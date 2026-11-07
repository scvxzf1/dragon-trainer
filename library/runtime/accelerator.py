"""HuggingFace Accelerate setup and FP16/BF16 plumbing.

Wraps the ``Accelerator`` construction so training scripts can stay out of the
logging-backend plumbing. Also hosts the state-resume helper (local dir or HF
repo) and the dtype resolver that maps ``--mixed_precision`` and
``--save_precision`` flags to torch dtypes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from typing import Optional

import torch
from accelerate import Accelerator
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def resume_from_local_or_hf_if_specified(accelerator, args):
    if not args.resume:
        return

    if not args.resume_from_huggingface:
        from library.training.adaptive_runtime.training_precision import reject_unplanned_resume
        reject_unplanned_resume(args)
        logger.info(f"resume training from local state: {args.resume}")
        accelerator.load_state(args.resume)
        return

    logger.info(f"resume training from huggingface state: {args.resume}")
    repo_id = args.resume.split("/")[0] + "/" + args.resume.split("/")[1]
    path_in_repo = "/".join(args.resume.split("/")[2:])
    revision = None
    repo_type = None
    if ":" in path_in_repo:
        divided = path_in_repo.split(":")
        if len(divided) == 2:
            path_in_repo, revision = divided
            repo_type = "model"
        else:
            path_in_repo, revision, repo_type = divided
    logger.info(
        f"Downloading state from huggingface: {repo_id}/{path_in_repo}@{revision}"
    )

    from huggingface_hub import list_repo_files

    list_files = list_repo_files(
        repo_id=repo_id,
        revision=revision,
        token=args.huggingface_token,
        repo_type=repo_type,
    )
    list_files = [f for f in list_files if f.startswith(path_in_repo)]

    async def download(filename) -> str:
        def task():
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type=repo_type,
                token=args.huggingface_token,
            )

        return await asyncio.get_event_loop().run_in_executor(None, task)

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        asyncio.gather(*[download(filename=filename) for filename in list_files])
    )
    if len(results) == 0:
        raise ValueError("No files found in the specified repo id")
    dirname = os.path.dirname(results[0])
    accelerator.load_state(dirname)


def resolve_run_log_dir(args: argparse.Namespace) -> Optional[str]:
    """Return the timestamped run-specific log directory for this invocation.

    Builds ``{args.logging_dir}/{log_prefix}{YYYYMMDD-HHMM}`` once and caches it
    on ``args._resolved_run_log_dir`` so the config-snapshot writer (which runs
    before ``prepare_accelerator``) and the accelerator setup land in the same
    dir even when they're a minute apart on the wall clock. Returns ``None``
    when ``--logging_dir`` is unset.
    """
    if args.logging_dir is None:
        return None
    cached = getattr(args, "_resolved_run_log_dir", None)
    if cached is not None:
        return cached
    if args.log_prefix is not None:
        log_prefix = args.log_prefix
    else:
        method = getattr(args, "method", None)
        preset = getattr(args, "preset", None)
        parts = [p for p in (method, preset) if p]
        log_prefix = ("_".join(parts) + "_") if parts else ""
    resolved = (
        args.logging_dir
        + "/"
        + log_prefix
        + time.strftime("%Y%m%d-%H%M", time.localtime())
    )
    args._resolved_run_log_dir = resolved
    return resolved


def prepare_accelerator(args: argparse.Namespace):
    logging_dir = resolve_run_log_dir(args)

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError("logging_dir is required when log_with is tensorboard")
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    # Always "NO": torch.compile is applied per-block by DiT.compile_blocks (the
    # block loop is the shape-stable hot path; pre/post-block regions vary per
    # bucket). Letting Accelerate full-compile the model on top of that would
    # double-compile and trigger graph-break / recompile storms.
    dynamo_backend = "NO"

    from library.training.adaptive_runtime.training_config import islands_enabled, require_training_contract
    from library.training.adaptive_runtime.native_precision import native_enabled
    precision_kwargs = {}
    if native_enabled(args):
        require_training_contract(args)
        from library.training.adaptive_runtime.native_precision import accelerator_handlers
        precision_kwargs["kwargs_handlers"] = accelerator_handlers(args)
    elif islands_enabled(args):
        require_training_contract(args)
        from library.training.adaptive_runtime.training_precision import accelerator_handlers
        precision_kwargs["kwargs_handlers"] = accelerator_handlers(args)
    # Loader configuration is merged, never replaced: the stateful dataloader
    # (adaptive OOM resume) and the opt-in seedable sampler (contract §13 V-2.4
    # cross-world-size comparison, t3 F-T3-1a) are independent switches and only
    # one ``DataLoaderConfiguration`` can be passed to ``Accelerator``.
    loader_config_kwargs: dict = {}
    if getattr(args, "adaptive_oom_retry_resume", False):
        loader_config_kwargs["use_stateful_dataloader"] = True
    from library.training.multigpu import loader_seed

    loader_config_kwargs.update(loader_seed.dataloader_config_kwargs(args))
    if loader_config_kwargs:
        from accelerate import DataLoaderConfiguration
        precision_kwargs["dataloader_config"] = DataLoaderConfiguration(
            **loader_config_kwargs
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        dynamo_backend=dynamo_backend,
        **precision_kwargs,
    )
    if native_enabled(args) or islands_enabled(args):
        require_training_contract(args, world_size=accelerator.num_processes)
        if accelerator.device.type != "cuda" or accelerator.scaler is None:
            raise ValueError("Adaptive FP16 training requires CUDA and the Accelerate FP16 scaler")
    _record_multiprocess_identity(accelerator, args)
    print("accelerator device:", accelerator.device)
    return accelerator


def _record_multiprocess_identity(accelerator, args) -> None:
    """Opt-in startup probe plus the per-run identity manifest (contract §7).

    Both are no-ops for single-process runs: no collective is issued and no file
    is written, so the single-card path keeps its exact behaviour.

    Multi-process: the manifest is written by rank 0 from an all-gathered view of
    every rank (device / precision / world size), and a manifest that cannot be
    written is a **named failure agreed across ranks** – a run without identity
    evidence must not continue as if it were verifiable (review t4 F2).
    """
    from library.training.multigpu.probe import maybe_run_startup_probe

    probe = maybe_run_startup_probe(accelerator, args)
    if probe.ran:
        logger.info(
            "multi-process startup probe ok: world_size=%d reduced=%.1f",
            probe.world_size,
            probe.reduced_total,
        )
    if int(getattr(accelerator, "num_processes", 1) or 1) <= 1:
        return

    from library.training.multigpu import identity as identity_mod

    written = identity_mod.record_or_fail(
        accelerator,
        args,
        getattr(args, "output_dir", None),
        accuracy_source=getattr(
            args, "_mixed_precision_source", "cli: explicit --mixed_precision"
        ),
    )
    if written is not None:
        logger.info("multi-process identity written to %s", written)


def prepare_dtype(args: argparse.Namespace):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    from library.training.adaptive_runtime.training_config import islands_enabled, require_training_contract
    from library.training.adaptive_runtime.native_precision import native_enabled
    if native_enabled(args):
        require_training_contract(args)
    elif islands_enabled(args):
        require_training_contract(args)
        weight_dtype = torch.float32

    save_dtype: Optional[torch.dtype] = None
    if args.save_precision == "fp16":
        save_dtype = torch.float16
    elif args.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif args.save_precision == "float":
        save_dtype = torch.float32

    return weight_dtype, save_dtype


def patch_accelerator_for_fp16_training(accelerator):
    """Force ``allow_fp16=True`` inside ``GradScaler._unscale_grads_``.

    Needed for full-fp16 training (as opposed to mixed-precision) where the
    optimizer holds fp16 params that the default scaler refuses to touch.
    """
    from accelerate import DistributedType

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        return

    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer
