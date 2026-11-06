"""Private frozen-config worker for real training with structured CUDA OOM."""

from argparse import Namespace
import json
import os
from pathlib import Path
import sys
import traceback

import torch

from library.training.auto_block_swap.process import write_result
from library.training.auto_block_swap.capacity import HostLimitReached
from .training_config import islands_enabled, require_training_contract, retry_configuration_errors, resolved_mode
from .training_runtime import NonfiniteSkip, TrainingRuntime


def _possible_oom(exc):
    message = str(exc).lower()
    return "out of memory" in message or "cublas_status_alloc_failed" in message


_TORCH_OOM_TYPES = tuple(
    dict.fromkeys(
        cls for cls in (getattr(torch, "OutOfMemoryError", None),
                        torch.cuda.OutOfMemoryError)
        if isinstance(cls, type)
    )
)


def _error_result(runtime, exc):
    traceback.print_exc()
    possible_oom = isinstance(exc, _TORCH_OOM_TYPES) or _possible_oom(exc)
    mismatch = "precision_contract_mismatch" in str(exc)
    return {
        **runtime.result("error"),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "failure_kind": "precision_contract_mismatch" if mismatch
        else "possible_oom" if possible_oom else "runtime",
        "reason": "precision_contract_mismatch" if mismatch
        else "possible_oom" if possible_oom else "runtime_error",
    }


def _native_cuda_oom(exc, runtime):
    # torch.cuda.OutOfMemoryError is an alias of torch.OutOfMemoryError.
    # Type alone therefore cannot distinguish a CPU allocation failure.
    message = str(exc).lower()
    if "cpu" in message or "defaultcpuallocator" in message or "host" in message:
        return False
    explicit_cuda = "cuda" in message
    # Generic typed OOM remains ambiguous even with a CUDA model present.
    return explicit_cuda


def execute(args, directory, trainer_factory):
    native = resolved_mode(args) == "native_fp16"
    if native:
        from .native_runtime import NativeStartupRuntime
    runtime = (NativeStartupRuntime if native else TrainingRuntime)(
        directory,
        precision_contract_id=getattr(args, "_adaptive_precision_contract_id", None),
        args=args,
    )
    trainer = None
    try:
        # A frozen worker must reject missing/drifted precision identity before
        # a newer mode's configuration checks can reclassify the failure.
        if native and (not isinstance(runtime.precision_contract_id, str)
                       or len(runtime.precision_contract_id) != 64
                       or any(c not in "0123456789abcdef" for c in runtime.precision_contract_id)):
            raise ValueError("precision_contract_mismatch: native precision identity missing or invalid")
        runtime.validate_precision()
        require_training_contract(args, world_size=int(os.environ.get("WORLD_SIZE", "1")))
        if native:
            from .contract import validate_native_sources
            from .native_retry_config import native_retry_configuration_errors
            validate_native_sources(args)
            errors = native_retry_configuration_errors(args)
        else:
            errors = retry_configuration_errors(lambda key, default=None: getattr(args, key, default))
        private = getattr(args, "_adaptive_training_worker", {})
        if (errors or getattr(args, "adaptive_oom_retry", False)
                or not (native or islands_enabled(args))
                or Path(private.get("directory", "")).resolve() != Path(directory).resolve()):
            raise ValueError("Invalid isolated training worker contract: " + "; ".join(errors))
        capacity = getattr(args, "_adaptive_swap_capacity", None)
        if native and capacity is None:
            raise ValueError("Native isolated training worker requires swap capacity")
        if capacity is not None:
            if (type(capacity.get("effective_cap")) is not int
                    or type(capacity.get("model_limit")) is not int
                    or not 0 <= args.blocks_to_swap <= capacity["effective_cap"]
                    <= capacity["model_limit"]):
                raise ValueError("Invalid isolated training worker swap capacity")
        # Validate H before model loading, which can OOM before any dtype
        # manifest exists and must remain eligible for a fresh-process retry.
        runtime.validate_precision()
        trainer = trainer_factory()
        trainer._adaptive_training_runtime = runtime
        trainer.train(args)
        if not runtime.finished:
            raise RuntimeError("Training returned without a completed adaptive training result")
        result = runtime.result("ok")
    except _TORCH_OOM_TYPES as exc:
        if not _native_cuda_oom(exc, runtime):
            result = _error_result(runtime, exc)
        else:
            result = {**runtime.result("cuda_oom"), "error_type": type(exc).__name__, "error": str(exc)}
    except HostLimitReached as exc:
        result = {**runtime.result("host_limit"), "error_type": type(exc).__name__, "error": str(exc)}
    except NonfiniteSkip as exc:
        # Distinct from ``error``/``cuda_oom``: the trial overflowed its skip budget,
        # which is a configuration/precision signal the supervisor can act on.
        result = {
            **runtime.result("nonfinite_skip"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failure_kind": "nonfinite_skip",
            "reason": "nonfinite_skip",
        }
    except Exception as exc:
        result = _error_result(runtime, exc)
    if result["status"] in {"ok", "cuda_oom"} or result.get("failure_kind") == "possible_oom":
        try:
            runtime.validate_precision(
                runtime._model if native else getattr(trainer, "_adaptive_precision_manifest_model", None)
            )
            result["precision_manifest_id"] = runtime.precision_manifest_id
        except Exception as exc:
            result = _error_result(runtime, exc)
    write_result(Path(directory) / "result.json", result)
    return result


def main(request):
    from library.training.train_bootstrap import install_stop_signal_handlers

    install_stop_signal_handlers()
    try:
        args = Namespace(**json.loads(request.read_text(encoding="utf-8")))
    finally:
        request.unlink(missing_ok=True)
    from train import AnimaTrainer

    execute(args, request.parent, AnimaTrainer)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
