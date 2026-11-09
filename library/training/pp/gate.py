"""Fail-closed gates between ``pipeline_parallel`` and the training entries.

This module is the single source of truth for the two questions every entry
point asks about pipeline parallel (PP):

1. **"May this config be served at all?"** --
   :func:`pipeline_parallel_runtime_issue` / :func:`require_pipeline_runtime`.
   ``library.training.compat_matrix`` (CLI preflight and WebUI preflight) and
   ``library.training.cli_entry`` both ask it, so the family switch
   (``PipelineParallelFamilySpec.runtime_backend``) and the compatibility gate
   cannot drift apart.
2. **"May *this* code path run while PP is on?"** -- the ``require_*`` guards
   called from the accelerate trainer hotspots (``bootstrap`` / ``loop`` /
   ``checkpoints`` / ``schedulers``).

Why the data-parallel trainer path is refused
---------------------------------------------
PP v0 owns stage placement, the 1F1B schedule, the fp16 transaction and the
per-rank checkpoint inside :mod:`library.training.pp`.  The accelerate trainer
path is the *data-parallel* topology, and four of its assumptions are simply
false under PP:

* ``accelerator.prepare`` wraps the whole network and installs DDP gradient
  sync (``bootstrap.prepare_with_accelerator``);
* one micro-batch is one whole-model forward through ``trainer.process_batch``
  (``loop._run_microbatch``);
* accelerate owns the accumulation boundary, i.e. the ``1/accumulation``
  gradient scaling (``loop``);
* ``accelerator.save_state`` writes **every** rank's state into one directory
  (``checkpoints.publish_state_with_bundle``).

A guard that let any of them continue would change the effective topology
without changing the checkpoint contract -- exactly the silent DDP fallback that
PP v0 exists to remove.  Each one therefore raises
:class:`~library.training.pp.errors.PipelineParallelError` with a stable code.

Adapter gradient scaling (why the schedule cannot own it)
--------------------------------------------------------
``torch.distributed.pipelining`` divides gradients by ``n_microbatches`` through
``stage.scale_grads(n_microbatches)``, which only walks
``self.submod.parameters()``.  The LoRA adapters hang off the network and are not
part of the base model's parameter tree, so that divide is a silent no-op for
exactly the parameters PP v0 trains (n=2 would leave adapter gradients at 2x the
reference).  PP v0 therefore builds the schedule with ``scale_grads=False`` and
applies the factor inside the last-stage loss --
:data:`PP_GRAD_SCALE_OWNER`.  Nothing in the trainer may add a second factor;
:func:`require_trainer_topology` is what keeps the trainer's own accumulation
scaling away from a PP run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from library.models.family_registry import (
    PIPELINE_PARALLEL_RUNTIME_BACKEND_1F1B,
    get_model_family_spec,
    normalize_registered_family,
)
from library.models.pipeline_parallel import PipelineParallelConfig

from .contract import (
    PipelineRuntimeConfig,
    validate_family_config,
    validate_no_quantization,
    validate_runtime_config,
)
from .errors import (
    CODE_ACCELERATE_CHECKPOINT,
    CODE_CONFIG_INVALID,
    CODE_RUNTIME_UNAVAILABLE,
    CODE_TRAINER_PATH,
    PipelineParallelError,
)

#: Compatibility code kept for "this family/config cannot be served".  Every PP
#: compatibility issue is keyed on ``key="pipeline_parallel"`` (the WebUI resume
#: gate and the multimodel tests depend on that key); a more specific ``pp_*``
#: code is only ever reported as the issue's ``code``, never as the key.
COMPAT_CODE_RUNTIME_UNAVAILABLE = "pipeline_parallel_runtime_unavailable"
#: Configuration rejections keep their historical compatibility code.
COMPAT_CODE_CONFIG_INVALID = "pipeline_parallel_config"

#: The one place the ``1/n_microbatches`` gradient factor is applied under PP.
PP_GRAD_SCALE_OWNER = "library.training.pp.schedule.make_last_stage_loss_fn"


@dataclass(frozen=True)
class PipelineGateIssue:
    """A named compatibility issue, ready to be reported by ``compat_matrix``."""

    code: str
    message: str


def _get(config: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def pipeline_parallel_enabled(config: Mapping[str, Any] | object) -> bool:
    """Is ``pipeline_parallel`` on?  A malformed toggle is a named error.

    ``pipeline_parallel = "maybe"`` must never be read as "off": that would turn a
    typo into a silent DDP run.  The common case (absent or literal ``False``)
    returns without parsing, because the loop asks this once per micro-batch.
    """

    raw = _get(config, "pipeline_parallel", False)
    if raw is False or raw is None:
        return False
    try:
        return bool(PipelineParallelConfig.from_config(config).enabled)
    except ValueError as exc:
        raise PipelineParallelError(CODE_CONFIG_INVALID, str(exc)) from exc


def family_runtime_backend(config: Mapping[str, Any] | object) -> tuple[str, str | None]:
    """``(canonical family, runtime_backend)`` for this config."""

    family = str(_get(config, "model_family", "anima") or "anima")
    try:
        canonical = normalize_registered_family(
            family,
            source="pipeline_parallel model_family",
            allow_aliases=True,
        )
    except ValueError as exc:
        raise PipelineParallelError(CODE_CONFIG_INVALID, str(exc)) from exc
    pipeline = get_model_family_spec(canonical).pipeline_parallel
    return canonical, (None if pipeline is None else pipeline.runtime_backend)


def _runtime_config(config: Mapping[str, Any] | object) -> PipelineRuntimeConfig:
    try:
        return PipelineRuntimeConfig.from_config(config)
    except ValueError as exc:  # family contract raised a bare ValueError
        raise PipelineParallelError(CODE_CONFIG_INVALID, str(exc)) from exc


def require_pipeline_runtime(
    config: Mapping[str, Any] | object,
    *,
    world_size: int | None = None,
) -> PipelineRuntimeConfig:
    """Validate the whole PP v0 runtime contract, or raise a named error.

    This is what "PP is available" means everywhere: the family publishes a
    stage-local runtime backend, the family contract accepts the config, and the
    runtime contract (schedule, microbatches >= stages, world size, quantization)
    holds in *this* environment (``torch.distributed.pipelining`` importable).
    """

    validate_family_config(config, world_size=world_size)
    runtime = _runtime_config(config)
    _, backend = family_runtime_backend(config)
    if backend != PIPELINE_PARALLEL_RUNTIME_BACKEND_1F1B:
        raise PipelineParallelError(
            CODE_RUNTIME_UNAVAILABLE,
            f"model_family={_get(config, 'model_family', 'anima')!r} has no "
            f"stage-local pipeline runtime (runtime_backend={backend!r}); only "
            f"{PIPELINE_PARALLEL_RUNTIME_BACKEND_1F1B!r} is implemented. Refusing to "
            "fall back to ordinary Accelerate data parallelism.",
        )
    validate_runtime_config(runtime, world_size=world_size)
    # Single source of truth: ``contract.quantization_markers`` (via
    # ``validate_no_quantization``).  The gate used to keep its own refined copy
    # of this rule, which made "is this a quantized base?" answer two ways.
    validate_no_quantization(config)
    return runtime


def compat_code(code: str) -> str:
    """Map a ``pp_*`` runtime code onto the compatibility-layer code space."""

    if code == CODE_CONFIG_INVALID:
        return COMPAT_CODE_CONFIG_INVALID
    if code.startswith("pp_"):
        return "pipeline_parallel_" + code[len("pp_") :]
    return code


def pipeline_parallel_runtime_issue(
    config: Mapping[str, Any] | object,
    *,
    world_size: int | None = None,
) -> PipelineGateIssue | None:
    """``None`` when this config may run PP v0, else the issue to report.

    ``compat_matrix`` reports the result as ``TrainingCompatIssue(code=..., key=
    "pipeline_parallel")``; nothing here changes that key.
    """

    try:
        require_pipeline_runtime(config, world_size=world_size)
    except PipelineParallelError as exc:
        return PipelineGateIssue(compat_code(exc.code), exc.detail)
    return None


def require_trainer_topology(config: Mapping[str, Any] | object, *, where: str) -> None:
    """Refuse to enter an accelerate/DDP code path while PP is enabled."""

    if not pipeline_parallel_enabled(config):
        return
    runtime = require_pipeline_runtime(config)
    raise PipelineParallelError(
        CODE_TRAINER_PATH,
        f"{where}: pipeline_parallel is enabled (stages={runtime.stages}, "
        f"schedule={runtime.schedule}, microbatches={runtime.microbatches}) but this "
        "is the Accelerate data-parallel trainer path: it wraps the whole network "
        "with accelerator.prepare, runs one whole-model forward per micro-batch "
        "through trainer.process_batch, owns the gradient-accumulation scaling, and "
        "saves every rank with accelerator.save_state. None of that is true for a "
        "pipeline. PP v0 runs through library.training.pp.build_anima_pipeline_runtime "
        "(stage-local optimizer, per-rank checkpoint, 1/microbatches applied in "
        f"{PP_GRAD_SCALE_OWNER}). Refusing to fall back to DDP.",
    )


def require_accelerate_checkpoint(
    config: Mapping[str, Any] | object, *, where: str
) -> None:
    """Refuse ``accelerator.save_state`` while PP is enabled.

    Under PP each rank owns a different stage, so one directory holding both
    ranks' state is not resumable: the per-rank PP checkpoint format is the only
    recoverable artifact.
    """

    if not pipeline_parallel_enabled(config):
        return
    raise PipelineParallelError(
        CODE_ACCELERATE_CHECKPOINT,
        f"{where}: accelerator.save_state writes every rank's state into one "
        "directory, which cannot be resumed under pipeline_parallel (each rank owns "
        "a different stage). Use the per-rank PP checkpoint path in "
        "library.training.pp. Refusing to write an unrecoverable checkpoint.",
    )


def lr_scheduler_num_processes(
    config: Mapping[str, Any] | object, num_processes: int
) -> int:
    """Divisor for the LR plan: ``1`` under PP, the data-parallel size otherwise.

    ``get_scheduler_fix`` multiplies ``max_train_steps`` by ``num_processes``
    because a data-parallel step only advances one gradient per rank.  Under PP a
    single step already covers the whole pipeline, so passing the world size would
    double the LR plan for ``stages=2``.
    """

    if pipeline_parallel_enabled(config):
        return 1
    return int(num_processes)


__all__ = [
    "COMPAT_CODE_CONFIG_INVALID",
    "COMPAT_CODE_RUNTIME_UNAVAILABLE",
    "PP_GRAD_SCALE_OWNER",
    "PipelineGateIssue",
    "compat_code",
    "family_runtime_backend",
    "lr_scheduler_num_processes",
    "pipeline_parallel_enabled",
    "pipeline_parallel_runtime_issue",
    "require_accelerate_checkpoint",
    "require_pipeline_runtime",
    "require_trainer_topology",
]
