"""Shared 1F1B pipeline-parallel runtime engine and family plugin contract.

Consolidates the ~72% identical stage construction, negotiated fail-closed
preconditions, ``PpFp16Transaction`` lifecycle, ``Schedule1F1B`` execution, and
checkpoint topology reporting across Anima, MOSS-Image, and Krea-2 into one
shared engine (:class:`GenericPipelineRuntime` + :func:`build_generic_pipeline_runtime`)
parameterized by a thin :class:`PpFamilyPlugin`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor

from .contract import (
    PipelineRuntimeConfig,
    PpComputePolicy,
    assert_compute_policy_agreement,
    validate_distributed_backend,
    validate_master_weight_dtype,
    validate_mixed_precision,
    validate_runtime_config,
)
from .errors import CODE_STAGE_CONTRACT, PipelineParallelError
from .schedule import (
    build_anima_schedule,
    make_last_stage_loss_fn,
    run_pipeline_step,
)
from .stages import FrozenBoundaryPipelineStage, assert_frozen_boundary_agreement
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


class PpFamilyPlugin:
    """Base hook interface for family-specific PP wire/stage behavior."""

    family: str = "generic"
    negotiation_prefix: str = "pp_build"
    negotiate_build: bool = True
    non_last_scaler_policy: str = "forbid"  # "forbid" | "drop"
    require_nonempty_trainable_for_factory: bool = True

    def resolve_wire_spec(
        self,
        model: Any,
        *,
        config: Any,
        latent_shape: Sequence[int],
        max_text_tokens: int | None = None,
        **extra: Any,
    ) -> Any:
        raise NotImplementedError

    def build_stage_module(
        self,
        model: Any,
        *,
        rank: int,
        config: Any,
        wire: Any,
        network: Any,
        policy: PpComputePolicy,
        **extra: Any,
    ) -> tuple[Any, Any]:
        raise NotImplementedError

    def after_stage_module_built(
        self,
        model: Any,
        *,
        layout: Any,
        stage_module: Any,
        config: Any,
        rank: int,
        device: torch.device,
        network: Any,
        **extra: Any,
    ) -> None:
        del model, layout, stage_module, config, rank, device, network, extra

    def build_pipeline_stage(
        self,
        model: Any,
        *,
        layout: Any,
        stage_module: Any,
        config: Any,
        rank: int,
        device: torch.device,
        wire: Any,
        policy: PpComputePolicy,
        group: Any,
        **extra: Any,
    ) -> FrozenBoundaryPipelineStage:
        raise NotImplementedError

    def extra_topology(self, runtime: GenericPipelineRuntime) -> dict[str, Any]:
        del runtime
        return {}

    def precheck_step(
        self,
        runtime: GenericPipelineRuntime,
        latents: Tensor | None,
        step_kwargs: dict[str, Any],
        target: Tensor | None,
    ) -> None:
        del runtime, latents, step_kwargs, target

    def prepare_schedule_inputs(
        self,
        runtime: GenericPipelineRuntime,
        step_kwargs: dict[str, Any],
        target: Tensor | None,
    ) -> tuple[dict[str, Any], Tensor | None]:
        del runtime
        return step_kwargs, target

    def postprocess_prediction(
        self, runtime: GenericPipelineRuntime, prediction: Any
    ) -> Any:
        del runtime
        return prediction

    def format_step_result(
        self,
        runtime: GenericPipelineRuntime,
        prediction: Any,
        bucket: list[Any],
    ) -> Any:
        del runtime
        return prediction, bucket


class GenericPipelineRuntime:
    """Shared stage-local PP v0 runtime owning stage, schedule, and fp16 transaction."""

    _default_plugin: PpFamilyPlugin = PpFamilyPlugin()

    def __init__(
        self,
        *,
        model: Any,
        layout: Any,
        stage_module: Any,
        stage: FrozenBoundaryPipelineStage,
        schedule: Any,
        config: PipelineRuntimeConfig | Any,
        wire: Any,
        latent_shape: Sequence[int],
        dtype: torch.dtype,
        compute_policy: PpComputePolicy,
        transaction: PpFp16Transaction | None,
        optimizer: Any,
        loss_fn: Any = None,
        group: Any = None,
        plugin: PpFamilyPlugin | None = None,
    ) -> None:
        self._model = model
        self._layout = layout
        self._stage_module = stage_module
        self._stage = stage
        self._schedule = schedule
        self._config = config
        self._loss_fn = loss_fn
        self._wire = wire
        self._latent_shape = tuple(int(v) for v in latent_shape)
        self._dtype = dtype
        self._compute_policy = compute_policy
        self._transaction = transaction
        self._optimizer = optimizer
        self._group = group
        self._plugin = self._default_plugin if plugin is None else plugin

    @property
    def compute_policy(self) -> PpComputePolicy:
        return self._compute_policy

    @property
    def use_fp32(self) -> bool:
        return bool(self._compute_policy.use_fp32)

    @property
    def layout(self) -> Any:
        return self._layout

    @property
    def stage(self) -> FrozenBoundaryPipelineStage:
        return self._stage

    @property
    def schedule(self) -> Any:
        return self._schedule

    @property
    def config(self) -> Any:
        return self._config

    @property
    def stage_module(self) -> Any:
        return self._stage_module

    @property
    def wire(self) -> Any:
        return self._wire

    @property
    def optimizer(self) -> Any:
        return self._optimizer

    @property
    def transaction(self) -> PpFp16Transaction | None:
        return self._transaction

    @property
    def is_first(self) -> bool:
        return bool(self._layout.is_first)

    @property
    def is_last(self) -> bool:
        return bool(self._layout.is_last)

    @property
    def scale(self) -> float | None:
        return None if self._transaction is None else self._transaction.scale

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return self._stage_module.trainable_parameters()

    def supports_latent_shape(self, latent_shape: Sequence[int]) -> bool:
        return tuple(int(v) for v in latent_shape) == self._latent_shape

    def topology(self) -> dict[str, Any]:
        base = self._layout.as_dict() if hasattr(self._layout, "as_dict") else {}
        base.update(
            {
                "is_first": self.is_first,
                "is_last": self.is_last,
                "microbatches": self._config.microbatches,
                "schedule": self._config.schedule,
                "conditioning": self._config.conditioning,
                "mixed_precision": self._config.mixed_precision,
                "latent_shape": list(self._latent_shape),
            }
        )
        base.update(self._plugin.extra_topology(self))
        return base

    def _execute_step(
        self,
        latents: Tensor | None,
        *,
        step_kwargs: dict[str, Any],
        target: Tensor | None = None,
        losses: list[Any] | None = None,
    ) -> Any:
        self._plugin.precheck_step(self, latents, step_kwargs, target)
        if self._transaction is not None:
            self._transaction.begin_step()
        schedule_kwargs, schedule_target = self._plugin.prepare_schedule_inputs(
            self, step_kwargs, target
        )
        bucket: list[Any] = [] if losses is None else losses
        prediction = run_pipeline_step(
            self._schedule,
            is_first=self.is_first,
            latents=latents if self.is_first else None,
            kwargs=schedule_kwargs,
            target=schedule_target if self.is_last else None,
            losses=bucket,
        )
        prediction = self._plugin.postprocess_prediction(self, prediction)
        return self._plugin.format_step_result(self, prediction, bucket)

    def optimizer_step(self) -> PpStepOutcome:
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


def model_dtype(model: Any) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def distributed_backend(group: Any) -> str | None:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None
    if group is not None and hasattr(dist, "get_backend"):
        try:
            return str(dist.get_backend(group))
        except Exception:  # noqa: BLE001 - exotic backends must not mask the check
            return None
    if hasattr(dist, "get_backend"):
        try:
            return str(dist.get_backend())
        except Exception:  # noqa: BLE001
            return None
    return None


def negotiate_phase(
    error: tuple[str, str] | None, group: Any, *, where: str
) -> None:
    """Turn a rank-local build failure into one symmetric refusal before collectives."""

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        if error is not None:
            raise PipelineParallelError(*error)
        return
    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, error, group=group)
    failures = [item for item in gathered if item]
    if failures:
        code, detail = failures[0]
        raise PipelineParallelError(
            code,
            f"negotiated build refusal ({where}; both ranks raise before any "
            f"further communication): {detail}",
        )


def _negotiate_optimizer_ownership(optimizer: Any, group: Any) -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        gathered: list[Any] = [None] * dist.get_world_size(group)
        dist.all_gather_object(
            gathered, {"has_optimizer": optimizer is not None}, group=group
        )
        if len({entry["has_optimizer"] for entry in gathered if entry}) != 1:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                "the ranks disagree about owning a stage-local optimizer "
                f"({gathered}); a PP pipeline cannot be half-transactional "
                "-- every rank needs stage-local trainable parameters and an "
                "optimizer, or none does",
            )


def _run_local_build_phase(
    plugin: PpFamilyPlugin,
    model: Any,
    *,
    config: Any,
    rank: int,
    world_size: int,
    device: torch.device,
    latent_shape: Sequence[int],
    max_text_tokens: int | None,
    dtype: torch.dtype | None,
    network: Any,
    scaler: Any,
    optimizer: Any,
    optimizer_factory: Any,
    group: Any,
    extra: dict[str, Any],
) -> tuple[Any, PpComputePolicy, torch.dtype, Any, Any, Any, Any, list[torch.nn.Parameter]]:
    validate_runtime_config(config, world_size=world_size)
    is_last = int(rank) == int(config.stages) - 1
    if plugin.non_last_scaler_policy == "drop":
        validate_mixed_precision(
            config, has_scaler=True if not is_last else scaler is not None
        )
        if not is_last:
            scaler = None
    elif not is_last and scaler is not None:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            "a non-last stage never sees the loss and must not own a GradScaler",
        )

    wire = plugin.resolve_wire_spec(
        model,
        config=config,
        latent_shape=latent_shape,
        max_text_tokens=max_text_tokens,
        **extra,
    )
    compute_dtype = dtype if dtype is not None else model_dtype(model)
    policy = PpComputePolicy.from_dtype(compute_dtype)
    layout, stage_module = plugin.build_stage_module(
        model,
        rank=int(rank),
        config=config,
        wire=wire,
        network=network,
        policy=policy,
        latent_shape=latent_shape,
        max_text_tokens=max_text_tokens,
        **extra,
    )
    plugin.after_stage_module_built(
        model,
        layout=layout,
        stage_module=stage_module,
        config=config,
        rank=int(rank),
        device=device,
        network=network,
        **extra,
    )
    stage_parameters = list(stage_module.trainable_parameters())
    # PP fp16's GradScaler transaction owns gradients, so the stage-local
    # adapter masters must be fp32 even when the ordinary trainer built them
    # from a fp16 compute-dtype model.  Promote only trainable stage
    # parameters; the frozen base is not part of this list.
    if config.mixed_precision == "fp16":
        for parameter in stage_parameters:
            if parameter.requires_grad and parameter.dtype != torch.float32:
                parameter.data = parameter.data.to(dtype=torch.float32)
    if plugin.negotiate_build:
        validate_master_weight_dtype(
            stage_parameters, mixed_precision=config.mixed_precision, device=device
        )
    if optimizer is None and optimizer_factory is not None:
        if plugin.require_nonempty_trainable_for_factory and not stage_parameters:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                f"rank {rank} owns no stage-local trainable parameters; a PP "
                "stage-local optimizer needs at least one owned adapter",
            )
        optimizer = optimizer_factory(stage_module)
    parameters = (
        [p for g in optimizer.param_groups for p in g["params"]]
        if optimizer is not None
        else stage_parameters
    )
    validate_distributed_backend(
        distributed_backend(group),
        device_type=str(getattr(device, "type", device)),
        world_size=int(world_size),
    )
    validate_master_weight_dtype(
        parameters,
        mixed_precision=config.mixed_precision,
        device=device if plugin.negotiate_build else None,
    )
    return wire, policy, compute_dtype, layout, stage_module, scaler, optimizer, parameters


def build_generic_pipeline_runtime(
    plugin: PpFamilyPlugin,
    runtime_cls: type[GenericPipelineRuntime],
    model: Any,
    *,
    config: Any,
    rank: int,
    world_size: int,
    device: torch.device,
    latent_shape: Sequence[int],
    max_text_tokens: int | None = None,
    dtype: torch.dtype | None = None,
    network: Any = None,
    scaler: Any = None,
    optimizer: Any = None,
    optimizer_factory: Any = None,
    max_grad_norm: float | None = None,
    clip_gradients: Any = None,
    loss_fn: Any = None,
    group: Any = None,
    **extra: Any,
) -> Any:
    """Construct a stage-local PP runtime through the shared negotiated lifecycle."""

    prefix = plugin.negotiation_prefix
    if not plugin.negotiate_build:
        (
            wire,
            policy,
            compute_dtype,
            layout,
            stage_module,
            scaler,
            optimizer,
            parameters,
        ) = _run_local_build_phase(
            plugin,
            model,
            config=config,
            rank=rank,
            world_size=world_size,
            device=device,
            latent_shape=latent_shape,
            max_text_tokens=max_text_tokens,
            dtype=dtype,
            network=network,
            scaler=scaler,
            optimizer=optimizer,
            optimizer_factory=optimizer_factory,
            group=group,
            extra=extra,
        )
        assert_compute_policy_agreement(policy, group=group, where=prefix)
        stage = plugin.build_pipeline_stage(
            model,
            layout=layout,
            stage_module=stage_module,
            config=config,
            rank=int(rank),
            device=device,
            wire=wire,
            policy=policy,
            group=group,
            latent_shape=latent_shape,
            **extra,
        )
        assert_frozen_boundary_agreement(stage, group=group)
    else:
        stage = None
        layout = None
        stage_module = None
        parameters = []
        policy = None
        compute_dtype = torch.float32
        wire = None
        build_error: tuple[str, str] | None = None
        try:
            (
                wire,
                policy,
                compute_dtype,
                layout,
                stage_module,
                scaler,
                optimizer,
                parameters,
            ) = _run_local_build_phase(
                plugin,
                model,
                config=config,
                rank=rank,
                world_size=world_size,
                device=device,
                latent_shape=latent_shape,
                max_text_tokens=max_text_tokens,
                dtype=dtype,
                network=network,
                scaler=scaler,
                optimizer=optimizer,
                optimizer_factory=optimizer_factory,
                group=group,
                extra=extra,
            )
            stage = plugin.build_pipeline_stage(
                model,
                layout=layout,
                stage_module=stage_module,
                config=config,
                rank=int(rank),
                device=device,
                wire=wire,
                policy=policy,
                group=group,
                latent_shape=latent_shape,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001 - negotiated symmetrically below
            build_error = (
                getattr(exc, "code", CODE_STAGE_CONTRACT),
                f"rank {rank}: {type(exc).__name__}: {exc}",
            )
        negotiate_phase(build_error, group, where=f"{prefix}:local")
        assert policy is not None and stage is not None and layout is not None
        assert_compute_policy_agreement(policy, group=group, where=prefix)
        frozen_error: tuple[str, str] | None = None
        try:
            assert_frozen_boundary_agreement(stage, group=group)
        except Exception as exc:  # noqa: BLE001
            frozen_error = (
                getattr(exc, "code", CODE_STAGE_CONTRACT),
                f"rank {rank}: {type(exc).__name__}: {exc}",
            )
        negotiate_phase(frozen_error, group, where=f"{prefix}:boundary")
        _negotiate_optimizer_ownership(optimizer, group)

    base_loss = loss_fn or make_last_stage_loss_fn(
        is_last=layout.is_last, microbatches=config.microbatches
    )
    transaction = None
    if optimizer is not None:
        clip = clip_gradients
        if clip is None and max_grad_norm is not None:
            clip = make_clip_gradients(parameters, max_grad_norm)
        transaction = PpFp16Transaction(
            scaler,
            optimizer,
            group=group,
            is_last=layout.is_last,
            clip_gradients=clip,
        )
        if layout.is_last:
            base_loss = transaction.wrap_loss(base_loss)
    schedule = build_anima_schedule(
        stage,
        microbatches=config.microbatches,
        schedule=config.schedule,
        loss_fn=base_loss,
    )
    resolved_latent = (
        tuple(int(v) for v in wire.latent_shape)
        if hasattr(wire, "latent_shape")
        else tuple(int(v) for v in latent_shape)
    )
    return runtime_cls(
        model=model,
        layout=layout,
        stage_module=stage_module,
        stage=stage,
        schedule=schedule,
        config=config,
        wire=wire,
        latent_shape=resolved_latent,
        dtype=compute_dtype,
        compute_policy=policy,
        transaction=transaction,
        optimizer=optimizer,
        loss_fn=base_loss,
        group=group,
        plugin=plugin,
    )
