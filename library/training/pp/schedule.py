"""Schedule1F1B wiring for the Anima PP runtime.

Two details are worth spelling out.

**The loss lives on the last stage only.**  ``torch.distributed.pipelining``
requires a ``loss_fn`` to enable backward at all (``_has_backward = loss_fn is
not None``), so *every* stage is given one; the schedule only ever invokes it on
the last stage (``_maybe_compute_loss``).  :func:`make_last_stage_loss_fn` keeps
that invariant honest by raising a named error if it is ever called elsewhere.

**Gradient scaling is explicit.**  ``scale_grads=False`` disables the schedule's
``p.grad /= n_microbatches`` post-pass -- which would be a silent no-op here,
because the stage module deliberately owns no registered parameters (the frozen
base model must not be reachable through ``stage.parameters()``).  The
``1/n_microbatches`` factor is applied inside the loss instead, so it cannot
depend on whether parameters happen to be registered.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .contract import (
    PP_SUPPORTED_SCHEDULES,
    PpComputePolicy,
    require_pipelining_api,
)
from .errors import (
    CODE_LOSS_STAGE,
    CODE_MICROBATCHES_BELOW_STAGES,
    CODE_SCHEDULE_UNSUPPORTED,
    CODE_STAGE_CONTRACT,
    PipelineParallelError,
)
from .layout import (
    adaln_lora_shape,
    boundary_hidden_shape,
    prediction_shape,
    split_batch_shape,
    timestep_embedding_shape,
)
from .placement import AnimaStageLayout
from .stages import (
    CONDITIONING_LOCAL,
    CONDITIONING_WIRE,
    FrozenBoundaryPipelineStage,
    frozen_boundary_indices,
)


def _meta(shape: Sequence[int], dtype: torch.dtype) -> Tensor:
    return torch.empty(tuple(int(value) for value in shape), dtype=dtype, device="meta")


def boundary_metas(
    model,
    layout: AnimaStageLayout,
    *,
    latent_shape: Sequence[int],
    microbatches: int,
    policy: PpComputePolicy,
    conditioning: str,
    timesteps_shape: Sequence[int] | None = None,
    context_shape: Sequence[int] | None = None,
    latent_dtype: torch.dtype | None = None,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """``(input_args, output_args)`` metadata for this stage.

    ``PipelineStage`` allocates its receive buffers from ``input_args`` once and
    reuses them for every microbatch, so these must be **per-microbatch** shapes.
    Supplying them explicitly (instead of letting torch run shape inference) keeps
    startup free of an extra collective and of a meta-device forward, which would
    re-enter ``autocast``/attention on the ``meta`` device type.

    Three dtypes are in play and a single ``dtype`` could not express them (D2):

    * ``latent`` -- what the trainer feeds stage 0 (``policy.activation_dtype``
      unless the caller overrides ``latent_dtype``);
    * ``activation`` -- what crosses the boundary.  With ``use_fp32`` the block
      stack promotes the residual stream (``models.py:1519``), so this is **fp32
      even for an fp16 run**;
    * ``prediction`` -- what the last stage returns.  The final layer casts back
      to the compute dtype, so this is fp16 for an fp16 run.

    Passing one dtype for all three produced
    ``PipeliningShapeError ... expected torch.float32 actual torch.float16`` on
    the first fp16 step.
    """

    latent_mb = split_batch_shape(latent_shape, microbatches)
    hidden_mb = boundary_hidden_shape(
        latent_mb,
        patch_spatial=int(model.patch_spatial),
        patch_temporal=int(model.patch_temporal),
        model_channels=int(model.model_channels),
    )
    activation_dtype = policy.boundary_dtype
    prediction_dtype = policy.prediction_dtype
    if latent_dtype is None:
        latent_dtype = policy.activation_dtype
    hidden_meta = _meta(hidden_mb, activation_dtype)
    prediction_meta = _meta(prediction_shape(latent_mb), prediction_dtype)
    if conditioning == CONDITIONING_LOCAL:
        boundary = (hidden_meta,)
    elif conditioning == CONDITIONING_WIRE:
        if timesteps_shape is None or context_shape is None:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                "wire conditioning needs timesteps_shape and context_shape to build "
                "the fixed boundary metadata",
            )
        boundary = (
            hidden_meta,
            _meta(
                timestep_embedding_shape(
                    split_batch_shape(timesteps_shape, microbatches),
                    int(model.model_channels),
                ),
                activation_dtype,
            ),
            _meta(split_batch_shape(context_shape, microbatches), activation_dtype),
            _meta(
                adaln_lora_shape(
                    split_batch_shape(timesteps_shape, microbatches),
                    int(model.model_channels),
                ),
                activation_dtype,
            ),
        )
    else:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT, f"unknown conditioning mode {conditioning!r}"
        )
    if layout.is_first:
        return (_meta(latent_mb, latent_dtype),), boundary
    return boundary, (prediction_meta,)


def build_pipeline_stage(
    stage_module,
    layout: AnimaStageLayout,
    *,
    device: torch.device,
    input_args: Sequence[Tensor],
    output_args: Sequence[Tensor],
    conditioning: str,
    group: Any = None,
) -> FrozenBoundaryPipelineStage:
    """Wrap a stage module in the torch pipelining stage, with the grad wire."""

    frozen = frozen_boundary_indices(conditioning)
    return FrozenBoundaryPipelineStage(
        stage_module,
        layout.stage_index,
        layout.stages,
        device,
        input_args=tuple(input_args),
        output_args=tuple(output_args),
        group=group,
        frozen_input_indices=() if layout.is_first else frozen,
        frozen_output_indices=() if layout.is_last else frozen,
    )


def make_last_stage_loss_fn(
    *,
    is_last: bool,
    microbatches: int,
    reduction: str = "mean",
) -> Any:
    """MSE loss with the ``1/n_microbatches`` factor baked in.

    ``reduction="mean"`` matches the training loop's per-sample loss, so the sum
    over microbatches has to be divided by the microbatch count to stay
    equivalent to the whole-batch step.
    """

    if reduction != "mean":
        raise PipelineParallelError(
            CODE_LOSS_STAGE,
            f"PP v0 only implements the mean-reduced loss, got {reduction!r}",
        )
    divisor = float(max(1, int(microbatches)))
    stage_index = "last" if is_last else "not-last"

    def loss_fn(prediction: Tensor, target: Tensor) -> Tensor:
        if not is_last:
            raise PipelineParallelError(
                CODE_LOSS_STAGE,
                "the pipeline loss function was invoked on a non-last stage "
                f"({stage_index}); the loss must be computed on the last stage only",
            )
        return F.mse_loss(prediction.float(), target.float()) / divisor

    return loss_fn


def build_anima_schedule(stage, *, microbatches: int, schedule: str, loss_fn: Any):
    """Build the requested schedule for this stage."""

    _, schedule_1f1b, schedule_gpipe = require_pipelining_api()
    if schedule not in PP_SUPPORTED_SCHEDULES:
        raise PipelineParallelError(
            CODE_SCHEDULE_UNSUPPORTED,
            f"schedule must be one of {', '.join(PP_SUPPORTED_SCHEDULES)}, got "
            f"{schedule!r}",
        )
    count = int(microbatches)
    if count < int(stage.num_stages):
        # torch 2.10 enforces this in PipelineScheduleSingle.__init__, which is the
        # shared base of Schedule1F1B *and* ScheduleGPipe -- so this is not an
        # "use GPipe instead" situation.
        raise PipelineParallelError(
            CODE_MICROBATCHES_BELOW_STAGES,
            f"{schedule} needs microbatches >= stages ({stage.num_stages}), got "
            f"{count}; PipelineScheduleSingle enforces it for GPipe as well in "
            "torch 2.10",
        )
    if schedule == "1f1b":
        return schedule_1f1b(
            stage, n_microbatches=count, loss_fn=loss_fn, scale_grads=False
        )
    return schedule_gpipe(stage, n_microbatches=count, loss_fn=loss_fn, scale_grads=False)


def run_pipeline_step(
    schedule,
    *,
    is_first: bool,
    latents: Tensor | None,
    kwargs: dict[str, Any],
    target: Tensor | None,
    losses: list | None = None,
) -> Any:
    """One schedule step.  Only the first stage takes positional model inputs."""

    if is_first:
        if latents is None:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT, "the first stage needs the microbatch latents"
            )
        return schedule.step(latents, losses=losses, **kwargs)
    if latents is not None:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            "a non-first stage receives its activations from the previous stage and "
            "must not be given positional latents",
        )
    return schedule.step(target=target, losses=losses, **kwargs)


__all__ = [
    "boundary_metas",
    "build_anima_schedule",
    "build_pipeline_stage",
    "make_last_stage_loss_fn",
    "run_pipeline_step",
]
