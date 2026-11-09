"""Top-level Anima pipeline-parallel v0 runtime for one rank.

:class:`AnimaPipelineRuntime` owns the four things that have to agree for a PP
step to be correct, and keeps them behind one fail-closed constructor:

* **placement** — :func:`library.training.pp.placement.build_anima_stage_module`
  (non-destructive, reuses ``make_pipeline_plan`` / ``build_anima_block_stage``);
* **the grad wire** — :class:`library.training.pp.stages.FrozenBoundaryPipelineStage`
  plus the cross-rank agreement check that makes a one-sided override impossible
  to reach;
* **the schedule** — ``Schedule1F1B`` (or ``ScheduleGPipe`` for one microbatch)
  with the loss on the last stage only;
* **the fp16 transaction** — one scale and one skip decision for the whole
  pipeline.

Token-bucket contract
---------------------
``PipelineStage`` freezes its receive buffers from the metadata passed to the
constructor, so a PP v0 run serves exactly **one** latent shape (one token
bucket -- e.g. the 4200 family).  A step whose latents do not match the built
shape is refused with a named error instead of failing deep inside the transport.
Serving a mixed-bucket dataloader needs one runtime per bucket, which is a
trainer-side decision and out of PP v0 scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .contract import (
    PipelineRuntimeConfig,
    require_pipelining_api,
    validate_mixed_precision,
    validate_no_quantization,
    validate_runtime_config,
)
from .errors import (
    CODE_SHAPE_CONTRACT,
    CODE_STAGE_CONTRACT,
    CODE_TOKEN_BUCKET_MISMATCH,
    PipelineParallelError,
)
from .layout import token_family, validate_latent_5d_shape
from .placement import (
    AnimaStageLayout,
    build_anima_stage_module,
)
from .schedule import (
    boundary_metas,
    build_anima_schedule,
    build_pipeline_stage,
    make_last_stage_loss_fn,
    run_pipeline_step,
)
from .contract import (
    PpComputePolicy,
    assert_compute_policy_agreement,
    validate_distributed_backend,
    validate_master_weight_dtype,
)
from .stages import (
    CONDITIONING_WIRE,
    FrozenBoundaryPipelineStage,
    assert_frozen_boundary_agreement,
)
from .transactions import PpFp16Transaction, PpStepOutcome, make_clip_gradients


@dataclass(frozen=True)
class PpStepResult:
    """Outcome of one pipeline step, as seen by this rank."""

    prediction: Any
    losses: tuple[float, ...]
    topology: dict[str, Any]

    @property
    def is_last(self) -> bool:
        return bool(self.topology["is_last"])


class AnimaPipelineRuntime:
    """One rank's Anima PP runtime: stage, schedule and fp16 transaction."""

    def __init__(
        self,
        *,
        model,
        layout: AnimaStageLayout,
        stage_module,
        stage: FrozenBoundaryPipelineStage,
        schedule,
        config: PipelineRuntimeConfig,
        loss_fn,
        latent_shape: Sequence[int],
        dtype: torch.dtype,
        compute_policy: PpComputePolicy,
        transaction: PpFp16Transaction | None,
        optimizer,
    ) -> None:
        self._model = model
        self._layout = layout
        self._stage_module = stage_module
        self._stage = stage
        self._schedule = schedule
        self._config = config
        self._loss_fn = loss_fn
        self._latent_shape = validate_latent_5d_shape(latent_shape)
        self._dtype = dtype
        self._compute_policy = compute_policy
        self._transaction = transaction
        self._optimizer = optimizer

    # -- read-only views -----------------------------------------------------

    @property
    def compute_policy(self) -> PpComputePolicy:
        """The pipeline-wide compute dtype decision (identical on every stage)."""

        return self._compute_policy

    @property
    def use_fp32(self) -> bool:
        """Shorthand for ``compute_policy.use_fp32``."""

        return bool(self._compute_policy.use_fp32)

    @property
    def layout(self) -> AnimaStageLayout:
        return self._layout

    @property
    def stage(self) -> FrozenBoundaryPipelineStage:
        return self._stage

    @property
    def schedule(self):
        return self._schedule

    @property
    def config(self) -> PipelineRuntimeConfig:
        return self._config

    @property
    def stage_module(self):
        """This stage's block-owning module (the per-stage checkpoint owner)."""

        return self._stage_module

    @property
    def optimizer(self):
        """This stage's optimizer, or ``None`` when the runtime owns no state."""

        return self._optimizer

    @property
    def transaction(self) -> PpFp16Transaction | None:
        """The fp16 transaction, or ``None`` when no optimizer was supplied."""

        return self._transaction

    @property
    def is_first(self) -> bool:
        return self._layout.is_first

    @property
    def is_last(self) -> bool:
        return self._layout.is_last

    def topology(self) -> dict[str, Any]:
        topology = self._layout.as_dict()
        topology.update(
            {
                "is_first": self._layout.is_first,
                "is_last": self._layout.is_last,
                "microbatches": self._config.microbatches,
                "schedule": self._config.schedule,
                "conditioning": self._config.conditioning,
                "mixed_precision": self._config.mixed_precision,
                "latent_shape": list(self._latent_shape),
                "token_family": token_family(self._boundary_shape()),
                "frozen_input_indices": list(self._stage.frozen_input_indices),
                "frozen_output_indices": list(self._stage.frozen_output_indices),
            }
        )
        return topology

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return self._stage_module.trainable_parameters()

    def adapter_key_map(self, local_key: str) -> str:
        return self._stage_module.adapter_key_map(local_key)

    def supports_latent_shape(self, latent_shape: Sequence[int]) -> bool:
        return tuple(int(value) for value in latent_shape) == self._latent_shape

    # -- training step -------------------------------------------------------

    def step(
        self,
        latents: Tensor,
        *,
        timesteps: Tensor,
        crossattn_emb: Tensor,
        target: Tensor | None = None,
        padding_mask: Tensor | None = None,
        fps: Tensor | None = None,
        h_offset: int = 0,
        w_offset: int = 0,
        losses: list | None = None,
    ) -> PpStepResult:
        """Run one 1F1B/GPipe step; only the last stage returns a prediction."""

        self._require_shape(latents)
        if self._transaction is not None:
            self._transaction.begin_step()
        kwargs: dict[str, Any] = {
            "timesteps": timesteps,
            "crossattn_emb": crossattn_emb,
        }
        if padding_mask is not None:
            kwargs["padding_mask"] = padding_mask
        if fps is not None:
            kwargs["fps"] = fps
        if h_offset:
            kwargs["h_offset"] = int(h_offset)
        if w_offset:
            kwargs["w_offset"] = int(w_offset)
        bucket: list[Any] = [] if losses is None else losses
        prediction = run_pipeline_step(
            self._schedule,
            is_first=self._layout.is_first,
            latents=latents if self._layout.is_first else None,
            kwargs=kwargs,
            target=target if self._layout.is_last else None,
            losses=bucket,
        )
        return PpStepResult(
            prediction=prediction,
            losses=tuple(float(value.detach()) for value in bucket),
            topology=self.topology(),
        )

    def optimizer_step(self) -> PpStepOutcome:
        """The symmetric fp16 optimizer boundary (unscale, agree, step/skip)."""

        if self._transaction is None:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                "this runtime was built without an optimizer/GradScaler, so it "
                "cannot own the optimizer boundary",
            )
        return self._transaction.step()

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        if self._optimizer is None:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                "this runtime was built without an optimizer",
            )
        self._optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def scale(self) -> float | None:
        return None if self._transaction is None else self._transaction.scale

    # -- internals -----------------------------------------------------------

    def _boundary_shape(self) -> tuple[int, ...]:
        from .layout import boundary_hidden_shape, split_batch_shape

        latent_mb = split_batch_shape(self._latent_shape, self._config.microbatches)
        return boundary_hidden_shape(
            latent_mb,
            patch_spatial=int(self._model.patch_spatial),
            patch_temporal=int(self._model.patch_temporal),
            model_channels=int(self._model.model_channels),
        )

    def _require_shape(self, latents: Tensor) -> None:
        shape = tuple(int(value) for value in latents.shape)
        if shape == self._latent_shape:
            return
        raise PipelineParallelError(
            CODE_TOKEN_BUCKET_MISMATCH,
            "this PP runtime serves exactly one latent shape (the pipeline wire "
            f"buffers are frozen at {self._latent_shape}, family "
            f"{token_family(self._boundary_shape())}); got {shape} (family "
            f"{token_family(_boundary_shape_of(self._model, shape))}). Build one "
            "runtime per token bucket -- mixing buckets in one PP run is not "
            "supported by PP v0",
        )


def _boundary_shape_of(model, latent_shape: Sequence[int]) -> tuple[int, ...]:
    from .layout import boundary_hidden_shape

    try:
        return boundary_hidden_shape(
            latent_shape,
            patch_spatial=int(model.patch_spatial),
            patch_temporal=int(model.patch_temporal),
            model_channels=int(model.model_channels),
        )
    except PipelineParallelError:
        return tuple(int(value) for value in latent_shape)


def build_anima_pipeline_runtime(
    model,
    *,
    config: PipelineRuntimeConfig,
    world_size: int,
    rank: int,
    device: torch.device,
    latent_shape: Sequence[int],
    dtype: torch.dtype | None = None,
    timesteps_shape: Sequence[int] | None = None,
    context_shape: Sequence[int] | None = None,
    network: Any = None,
    scaler: Any = None,
    optimizer: Any = None,
    optimizer_factory: Any = None,
    max_grad_norm: float | None = None,
    clip_gradients: Any = None,
    loss_fn: Any = None,
    group: Any = None,
) -> AnimaPipelineRuntime:
    """Build this rank's PP v0 runtime, or fail closed with a named code.

    ``optimizer_factory`` receives the freshly built stage module and returns the
    optimizer for *this stage's* trainable parameters.  It exists because the
    stage module is constructed inside this function, so a caller cannot build a
    per-stage optimizer before it.

    Three startup requirements are not optional; each fails closed with a named
    code so bootstrap/preflight can switch on it (details in the helpers):

    * ``dtype`` is the pre-block compute dtype and it fixes ``use_fp32`` for the
      whole pipeline (``pp_compute_policy_mismatch`` if the first stage is fed
      something else) -- see :func:`_build_stage_runtime`.
    * CUDA PP needs ``cpu:gloo,cuda:nccl``
      (``pp_backend_multibackend_required``) -- see
      :func:`~library.training.pp.contract.validate_distributed_backend`.
    * ``mixed_precision="fp16"`` needs fp32 master weights
      (``pp_fp16_master_weights_required``) -- see
      :func:`~library.training.pp.contract.validate_master_weight_dtype`.
    """

    scaler = _validate_build_request(config, rank=rank, world_size=world_size, scaler=scaler)
    policy, compute_dtype, layout, stage_module, optimizer = _build_stage_runtime(
        model,
        config,
        rank=rank,
        world_size=world_size,
        dtype=dtype,
        device=device,
        network=network,
        optimizer=optimizer,
        optimizer_factory=optimizer_factory,
        group=group,
    )
    stage = _build_stage(
        model,
        layout,
        stage_module,
        config=config,
        device=device,
        latent_shape=latent_shape,
        policy=policy,
        timesteps_shape=timesteps_shape,
        context_shape=context_shape,
        group=group,
    )
    base_loss = loss_fn or make_last_stage_loss_fn(
        is_last=layout.is_last, microbatches=config.microbatches
    )
    transaction = _build_transaction(
        layout,
        scaler=scaler,
        optimizer=optimizer,
        group=group,
        clip_gradients=clip_gradients,
        max_grad_norm=max_grad_norm,
    )
    if transaction is not None and layout.is_last:
        base_loss = transaction.wrap_loss(base_loss)
    schedule = build_anima_schedule(
        stage,
        microbatches=config.microbatches,
        schedule=config.schedule,
        loss_fn=base_loss,
    )
    return AnimaPipelineRuntime(
        model=model,
        layout=layout,
        stage_module=stage_module,
        stage=stage,
        schedule=schedule,
        config=config,
        loss_fn=base_loss,
        latent_shape=latent_shape,
        dtype=compute_dtype,
        compute_policy=policy,
        transaction=transaction,
        optimizer=optimizer,
    )


def _build_stage_runtime(
    model,
    config,
    *,
    rank: int,
    world_size: int,
    dtype,
    device,
    network,
    optimizer,
    optimizer_factory,
    group,
):
    """Resolve the compute policy, build this stage's module and its optimizer.

    The policy is resolved **once, here, before any stage exists**, from the dtype
    the trainer will feed.  That ordering is the D1 fix: a stage that could see a
    boundary tensor would re-derive ``use_fp32`` from it, get ``False`` (the
    boundary is fp32 under fp16) and run its blocks in fp16.
    """

    compute_dtype = dtype if dtype is not None else _model_dtype(model)
    policy = PpComputePolicy.from_dtype(compute_dtype)
    layout, stage_module = build_anima_stage_module(
        model,
        stage_index=int(rank),
        stages=int(config.stages),
        conditioning=config.conditioning,
        network=network,
        compute_policy=policy,
    )
    if optimizer is None and optimizer_factory is not None:
        optimizer = optimizer_factory(stage_module)
    _validate_runtime_preconditions(
        config,
        policy=policy,
        stage_module=stage_module,
        optimizer=optimizer,
        device=device,
        world_size=int(world_size),
        group=group,
    )
    return policy, compute_dtype, layout, stage_module, optimizer


def _validate_runtime_preconditions(
    config,
    *,
    policy: PpComputePolicy,
    stage_module,
    optimizer,
    device,
    world_size: int,
    group,
) -> None:
    """Everything that must hold before the first step, in one place.

    Three checks that only make sense once the policy, the stage module and the
    optimizer exist:

    * **D3** -- a CUDA PP run needs a multi-backend process group
      (``cpu:gloo,cuda:nccl``); a single backend cannot serve both the GPU
      activations and the CPU scalars the fp16 transaction exchanges.
    * **D4** -- fp16 PP needs fp32 master weights, because ``GradScaler`` refuses
      fp16 gradients and the non-last stages divide gradients by hand.
    * **D1** -- every rank must have resolved the same ``use_fp32``.
    """

    validate_distributed_backend(
        _distributed_backend(group),
        device_type=str(getattr(device, "type", device)),
        world_size=int(world_size),
    )
    parameters = (
        [p for g in optimizer.param_groups for p in g["params"]]
        if optimizer is not None
        else list(stage_module.trainable_parameters())
    )
    validate_master_weight_dtype(parameters, mixed_precision=config.mixed_precision)
    assert_compute_policy_agreement(policy, group=group, where="pp_build")


def _distributed_backend(group) -> str | None:
    """The process group's backend spec, when a group is live."""

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None
    if group is not None and hasattr(dist, "get_backend"):
        try:
            return str(dist.get_backend(group))
        except (ValueError, RuntimeError):  # pragma: no cover - exotic backends
            return None
    return str(dist.get_backend())


def _validate_build_request(config, *, rank: int, world_size: int, scaler: Any) -> Any:
    """Validate the contract and return the scaler this stage may own."""

    validate_runtime_config(config, world_size=world_size)
    is_last = int(rank) == int(config.stages) - 1
    # Only the stage that owns the loss can own the loss scale; the other stages
    # unscale from the pipeline-wide scale, so they must not require a scaler.
    validate_mixed_precision(
        config, has_scaler=True if not is_last else scaler is not None
    )
    if is_last:
        return scaler
    # A non-last stage never sees the loss, so it must not own a GradScaler: its
    # gradients are unscaled from the pipeline-wide scale instead.
    return None


def _build_stage(
    model,
    layout,
    stage_module,
    *,
    config,
    device,
    latent_shape,
    policy: PpComputePolicy,
    timesteps_shape,
    context_shape,
    group,
):
    """Boundary metadata + the torch pipelining stage with the grad wire."""

    if config.conditioning == CONDITIONING_WIRE and (
        timesteps_shape is None or context_shape is None
    ):
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            "conditioning=wire needs timesteps_shape and context_shape",
        )
    input_args, output_args = boundary_metas(
        model,
        layout,
        latent_shape=latent_shape,
        microbatches=config.microbatches,
        policy=policy,
        conditioning=config.conditioning,
        timesteps_shape=timesteps_shape,
        context_shape=context_shape,
    )
    stage = build_pipeline_stage(
        stage_module,
        layout,
        device=device,
        input_args=input_args,
        output_args=output_args,
        conditioning=config.conditioning,
        group=group,
    )
    assert_frozen_boundary_agreement(stage, group=group)
    return stage


def _build_transaction(
    layout, *, scaler, optimizer, group, clip_gradients, max_grad_norm
):
    """The symmetric fp16 transaction, when this rank owns an optimizer."""

    if optimizer is None:
        return None
    clip = clip_gradients
    if clip is None and max_grad_norm is not None:
        clip = make_clip_gradients(
            [p for g in optimizer.param_groups for p in g["params"]], max_grad_norm
        )
    return PpFp16Transaction(
        scaler,
        optimizer,
        group=group,
        is_last=layout.is_last,
        clip_gradients=clip,
    )


def _model_dtype(model) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def assert_runtime_capabilities(config: Mapping[str, Any] | object) -> None:
    """Convenience preflight: quantization + capability checks, no construction.

    ``config`` is the *source* configuration (``argparse`` Namespace or a plain
    mapping), because that is where the quantization switches live:
    ``base_compute`` / ``nf4`` / ``convrot_*`` are not fields of
    :class:`PipelineRuntimeConfig`, so passing the normalized runtime config here
    could never see a quantized base.  A ``PipelineRuntimeConfig`` is still
    accepted (it simply reports no markers).
    """

    require_pipelining_api()
    validate_no_quantization(config)


__all__ = [
    "AnimaPipelineRuntime",
    "PpStepResult",
    "assert_runtime_capabilities",
    "build_anima_pipeline_runtime",
]
