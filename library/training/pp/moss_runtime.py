"""MOSS-Image stage-local PP v0 runtime for one rank.

Reuses, unchanged, the Anima PP v0 machinery that carries no latent-shape
coupling: :mod:`~library.training.pp.contract` (fail-closed config, compute
policy, master weights, backend agreement), :mod:`~library.training.pp
.transactions` (one fp16 scale / one skip decision), :mod:`~library.training.pp
.schedule` (``build_anima_schedule`` / ``run_pipeline_step`` /
``make_last_stage_loss_fn``) and :class:`~library.training.pp.stages
.FrozenBoundaryPipelineStage`.  The family-specific part lives in
:mod:`library.models.moss_image.pipeline_parallel`.

Boundary contract (MOSS, 4D)
----------------------------
Stage 0 owns the embeddings, the text refinement (whose gradient must reach it
through the boundary) and the first layer range; stage 1 owns the rest plus the
video head.  The wire carries three slots per microbatch:

* ``hidden``  ``[B, H*W + max_text_tokens, dim]`` -- the *joint* image+text
  hidden, grad-carrying, padded to the frozen ``token_cap``;
* ``lengths`` ``[B]`` float32 (token counts are exact in fp32; torch pipelining
  force-enables ``requires_grad`` on every received tensor, which int tensors
  reject) -- per-sample true text length, int-valued metadata;
* ``mod``     ``[B, 6, dim]`` -- shared block modulation, frozen (detached).

``lengths`` and ``mod`` are the frozen slots (``(1, 2)``); the receiving stage
detaches them and the paired ``FrozenBoundaryPipelineStage`` overrides keep the
grad wire symmetric.  Per-sample RoPE is rebuilt locally on each stage from
``(H, W, S_i)``, matching the production forward exactly; transport padding is
sliced away per sample and therefore never enters any attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor

from library.models.moss_image.pipeline_parallel import (
    MOSS_LATENT_CHANNELS,
    MossStageLayout,
    _validate_text_wire,
    build_moss_stage_module,
    resolve_wire_capacity,
    validate_moss_latent_shape,
)

from .contract import (
    PpComputePolicy,
    assert_compute_policy_agreement,
    validate_distributed_backend,
    validate_master_weight_dtype,
    validate_runtime_config,
)
from .errors import (
    CODE_SHAPE_CONTRACT,
    CODE_STAGE_CONTRACT,
    CODE_TOKEN_BUCKET_MISMATCH,
    PipelineParallelError,
)
from .schedule import (
    build_anima_schedule,
    make_last_stage_loss_fn,
    run_pipeline_step,
)
from .stages import FrozenBoundaryPipelineStage, assert_frozen_boundary_agreement
from .transactions import PpFp16Transaction, PpStepOutcome, make_clip_gradients

#: Boundary slots that carry no gradient: ``(lengths, mod)``.  Slot 0 (the joint
#: hidden) is the differentiable surface the loss gradient travels through.
MOSS_BOUNDARY_FROZEN_INDICES: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class MossWireSpec:
    """The frozen per-run wire geometry (per-microbatch shapes)."""

    latent_shape: tuple[int, ...]
    microbatch_latent_shape: tuple[int, ...]
    max_text_tokens: int
    token_cap: int
    dim: int

    @property
    def batch_size(self) -> int:
        return int(self.latent_shape[0])

    @property
    def microbatch_batch_size(self) -> int:
        return int(self.microbatch_latent_shape[0])


def resolve_moss_wire_spec(
    model,
    *,
    latent_shape: Sequence[int],
    max_text_tokens: int,
    microbatches: int,
) -> MossWireSpec:
    """Validate the run geometry and freeze the microbatch wire shapes."""

    latent = validate_moss_latent_shape(latent_shape)
    if int(latent[0]) % int(microbatches):
        raise PipelineParallelError(
            CODE_TOKEN_BUCKET_MISMATCH,
            f"train batch size {latent[0]} is not divisible by "
            f"pipeline_parallel_microbatches={microbatches}; PP v0 freezes the "
            "wire buffers so every microbatch must be the same size",
        )
    microbatch = (int(latent[0]) // int(microbatches), *latent[1:])
    dim = int(getattr(model.video_embedder, "out_features"))
    token_cap = resolve_wire_capacity(int(latent[2]), int(latent[3]), max_text_tokens)
    return MossWireSpec(
        latent_shape=latent,
        microbatch_latent_shape=microbatch,
        max_text_tokens=int(max_text_tokens),
        token_cap=int(token_cap),
        dim=dim,
    )


def moss_boundary_metas(
    model,
    layout: MossStageLayout,
    *,
    wire: MossWireSpec,
    policy: PpComputePolicy,
    latent_dtype: torch.dtype,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """``(input_args, output_args)`` metadata for this MOSS stage.

    MOSS's residual stream keeps its compute dtype across the boundary (the
    production ``_Block`` never promotes it), so the activation dtype is
    ``policy.activation_dtype`` -- deliberately *not* the Anima
    ``boundary_dtype``, which encodes Anima's promoted stream.
    """

    def _meta(shape: Sequence[int], dtype: torch.dtype) -> Tensor:
        return torch.empty(tuple(int(value) for value in shape), dtype=dtype, device="meta")

    dtype = policy.activation_dtype
    mb = wire.microbatch_latent_shape
    hidden_meta = _meta((mb[0], wire.token_cap, wire.dim), dtype)
    # torch pipelining force-enables requires_grad on every received tensor,
    # so the integer metadata rides the wire as float32 (exact for token
    # counts) instead of an int tensor.
    lengths_meta = _meta((mb[0],), torch.float32)
    mod_meta = _meta((mb[0], 6, wire.dim), dtype)
    boundary = (hidden_meta, lengths_meta, mod_meta)
    if layout.is_first:
        return (_meta(mb, latent_dtype),), boundary
    prediction_meta = _meta(
        (mb[0], MOSS_LATENT_CHANNELS, mb[2], mb[3]),
        dtype,
    )
    return boundary, (prediction_meta,)


class MossPipelineRuntime:
    """One rank's MOSS PP runtime: stage, schedule and fp16 transaction."""

    def __init__(
        self,
        *,
        model,
        layout: MossStageLayout,
        stage_module,
        stage: FrozenBoundaryPipelineStage,
        schedule,
        config,
        loss_fn,
        wire: MossWireSpec,
        dtype: torch.dtype,
        compute_policy: PpComputePolicy,
        transaction: PpFp16Transaction | None,
        optimizer,
        group: Any = None,
    ) -> None:
        self._model = model
        self._layout = layout
        self._stage_module = stage_module
        self._stage = stage
        self._schedule = schedule
        self._config = config
        self._loss_fn = loss_fn
        self._wire = wire
        self._dtype = dtype
        self._compute_policy = compute_policy
        self._transaction = transaction
        self._optimizer = optimizer
        self._group = group

    # -- read-only views -----------------------------------------------------

    @property
    def compute_policy(self) -> PpComputePolicy:
        return self._compute_policy

    @property
    def wire(self) -> MossWireSpec:
        return self._wire

    @property
    def layout(self) -> MossStageLayout:
        return self._layout

    @property
    def stage(self) -> FrozenBoundaryPipelineStage:
        return self._stage

    @property
    def schedule(self):
        return self._schedule

    @property
    def config(self):
        return self._config

    @property
    def stage_module(self):
        return self._stage_module

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def transaction(self) -> PpFp16Transaction | None:
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
                "latent_shape": list(self._wire.latent_shape),
                "latent_channels": MOSS_LATENT_CHANNELS,
                "token_cap": self._wire.token_cap,
                "max_text_tokens": self._wire.max_text_tokens,
                "dim": self._wire.dim,
                "dtype": str(self._dtype),
                # MOSS's actual boundary dtype: the residual stream keeps the
                # compute dtype (``_Block`` never promotes it), so the wire is
                # the activation dtype -- ``PpComputePolicy.boundary_dtype``
                # encodes Anima's promoted-stream semantics and is NOT the
                # MOSS wire dtype.
                "wire_dtype": str(self._compute_policy.activation_dtype),
                "compute_policy": self._compute_policy.as_dict(),
                "frozen_input_indices": list(self._stage.frozen_input_indices),
                "frozen_output_indices": list(self._stage.frozen_output_indices),
                "adapter_names": sorted(
                    module.lora_name for module in self._stage_module.adapter_modules
                ),
            }
        )
        return topology

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return self._stage_module.trainable_parameters()

    def supports_latent_shape(self, latent_shape: Sequence[int]) -> bool:
        return tuple(int(v) for v in latent_shape) == self._wire.latent_shape

    # -- training step -------------------------------------------------------

    def _negotiate_step_precheck(
        self,
        latents: Tensor | None,
        text_hiddens: Tensor | None,
        text_mask: Tensor | None,
        timesteps: Tensor,
        target: Tensor | None = None,
    ) -> None:
        """Share the per-step verdict before any rank enters the schedule.

        Every rank validates what it can see locally (rank 0: the latent
        geometry/dtype and the text wire; both ranks: the timestep metadata;
        the last stage additionally the training target), then one CPU
        ``all_gather_object`` shares the verdicts.  A rank-local raise alone
        would strand the peer inside the schedule waiting for a collective
        that never comes, so a failure is refused *symmetrically*: both ranks
        raise the same ``pp_*`` code before any communication.
        """

        import torch.distributed as dist

        error: tuple[str, str] | None = None
        try:
            # Timesteps are needed on *every* stage (block modulation starts
            # at rank 0), so both ranks check presence and per-sample shape.
            if timesteps is None:
                raise PipelineParallelError(
                    CODE_STAGE_CONTRACT,
                    "every MOSS PP stage needs the batch timesteps for the "
                    "block modulation (rank 0 included)",
                )
            if torch.as_tensor(timesteps).numel() != self._wire.batch_size:
                raise PipelineParallelError(
                    CODE_SHAPE_CONTRACT,
                    f"timesteps must carry one value per sample "
                    f"({self._wire.batch_size}), got "
                    f"{torch.as_tensor(timesteps).numel()}",
                )
            if self._layout.is_first:
                if latents is None or text_hiddens is None or text_mask is None:
                    raise PipelineParallelError(
                        CODE_STAGE_CONTRACT,
                        "the first MOSS stage needs the batch latents, "
                        "text_hiddens and text_mask",
                    )
                shape = validate_moss_latent_shape(latents.shape)
                if shape != self._wire.latent_shape:
                    raise PipelineParallelError(
                        CODE_TOKEN_BUCKET_MISMATCH,
                        f"this MOSS PP runtime serves exactly one whole-step "
                        f"latent shape {self._wire.latent_shape} (microbatches "
                        f"{self._config.microbatches} -> per-microbatch "
                        f"{self._wire.microbatch_latent_shape}); got {shape}",
                    )
                if latents.dtype != self._compute_policy.activation_dtype:
                    raise PipelineParallelError(
                        "pp_compute_policy_mismatch",
                        f"the pre-block latent dtype is {latents.dtype} but the "
                        f"pipeline compute policy was built for "
                        f"{self._compute_policy.activation_dtype}",
                    )
                _validate_text_wire(
                    text_hiddens, text_mask, shape[0], self._wire.max_text_tokens
                )
                # C1: per-sample (not just per-batch) true-length precheck, so
                # every sample's real length is validated before any rank
                # enters the schedule -- not lazily inside stage 0's loop.
                image_tokens = shape[2] * shape[3]
                for index in range(shape[0]):
                    length = int(text_mask[index].sum().item())
                    if length < 1:
                        raise PipelineParallelError(
                            CODE_STAGE_CONTRACT,
                            f"sample {index} has no valid text token; the "
                            "production MOSS forward rejects the same case",
                        )
                    if length > self._wire.max_text_tokens:
                        raise PipelineParallelError(
                            CODE_TOKEN_BUCKET_MISMATCH,
                            f"sample {index} text length {length} exceeds the "
                            f"built wire capacity {self._wire.max_text_tokens}; "
                            "build one runtime per max_text_tokens",
                        )
                    if image_tokens + length > self._wire.token_cap:
                        raise PipelineParallelError(
                            CODE_TOKEN_BUCKET_MISMATCH,
                            f"sample {index} needs {image_tokens + length} joint "
                            f"tokens, above the built token_cap "
                            f"{self._wire.token_cap}",
                        )
            else:
                # Last stage: the training target is required for the loss,
                # and must fit the built wire shape (no silent shape drift).
                if target is None:
                    raise PipelineParallelError(
                        CODE_STAGE_CONTRACT,
                        "the last MOSS stage needs the batch target to compute "
                        "the loss",
                    )
                if tuple(target.shape) != tuple(self._wire.latent_shape):
                    raise PipelineParallelError(
                        CODE_SHAPE_CONTRACT,
                        f"target shape {tuple(target.shape)} does not match the "
                        f"built latent shape {self._wire.latent_shape}",
                    )
        except PipelineParallelError as exc:
            error = (exc.code, exc.detail)
        if not (dist.is_available() and dist.is_initialized()):
            if error is not None:
                raise PipelineParallelError(*error)
            return
        gathered: list[Any] = [None] * dist.get_world_size(self._group)
        dist.all_gather_object(gathered, error, group=self._group)
        failures = [item for item in gathered if item]
        if failures:
            code, detail = failures[0]
            raise PipelineParallelError(
                code,
                "negotiated step refusal (both ranks raise before any "
                f"communication): {detail}",
            )

    def step(
        self,
        latents: Tensor | None,
        *,
        timesteps: Tensor,
        text_hiddens: Tensor | None = None,
        text_mask: Tensor | None = None,
        target: Tensor | None = None,
        losses: list | None = None,
    ):
        """Run one 1F1B step for the whole batch.

        The schedule chunks the batch into ``n_microbatches`` pieces itself
        (the wire buffers are per-microbatch); callers pass the *full* batch,
        exactly like :meth:`AnimaPipelineRuntime.step`.  Only the last stage
        returns a prediction.
        """

        self._negotiate_step_precheck(
            latents, text_hiddens, text_mask, timesteps, target
        )
        if self._layout.is_first and latents is not None:
            self._require_shape(latents)
        if self._transaction is not None:
            self._transaction.begin_step()
        kwargs: dict[str, Any] = {"timesteps": timesteps}
        bucket: list[Any] = [] if losses is None else losses
        prediction = run_pipeline_step(
            self._schedule,
            is_first=self._layout.is_first,
            latents=latents if self._layout.is_first else None,
            kwargs=kwargs if not self._layout.is_first else {
                "timesteps": timesteps,
                "text_hiddens": text_hiddens,
                "text_mask": text_mask,
            },
            target=target if self._layout.is_last else None,
            losses=bucket,
        )
        return prediction, bucket

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

    @property
    def scale(self) -> float | None:
        return None if self._transaction is None else self._transaction.scale

    # -- internals -----------------------------------------------------------

    def _require_shape(self, latents: Tensor) -> None:
        shape = tuple(int(value) for value in latents.shape)
        if shape == self._wire.latent_shape:
            return
        raise PipelineParallelError(
            CODE_TOKEN_BUCKET_MISMATCH,
            f"this MOSS PP runtime serves exactly one whole-step latent shape "
            f"{self._wire.latent_shape} (microbatches "
            f"{self._config.microbatches} -> per-microbatch "
            f"{self._wire.microbatch_latent_shape}); got {shape}. Build one "
            "runtime per (batch, H, W) wire shape -- tail batches and mixed "
            "shapes are not supported by PP v0",
        )


def _negotiate_phase(
    error: tuple[str, str] | None, group: Any, *, where: str
) -> None:
    """Share one local build verdict before the next collective.

    A rank whose local construction raised would otherwise strand its peer
    inside the next collective (policy agreement / frozen-boundary gather /
    fp16 broadcast).  One CPU ``all_gather_object`` makes any local failure a
    symmetric, same-code refusal on every rank before that collective runs.
    """

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


def build_moss_pipeline_runtime(
    model,
    *,
    config,
    rank: int,
    world_size: int,
    device: torch.device,
    latent_shape: Sequence[int],
    max_text_tokens: int,
    dtype: torch.dtype | None = None,
    network: Any = None,
    scaler: Any = None,
    optimizer: Any = None,
    optimizer_factory: Any = None,
    max_grad_norm: float | None = None,
    clip_gradients: Any = None,
    group: Any = None,
) -> MossPipelineRuntime:
    """Build this rank's MOSS PP v0 runtime, or fail closed with a named code.

    Mirrors :func:`~library.training.pp.runtime.build_anima_pipeline_runtime`
    in ordering and checks (compute policy first, stage module, optimizer,
    master weights / backend / policy agreement, wire metadata, frozen-boundary
    agreement, schedule, fp16 transaction) while keeping every MOSS-specific
    decision in :mod:`library.models.moss_image.pipeline_parallel`.
    """

    # ---- local construction phase (no collectives) --------------------------
    # Everything below can fail rank-locally; C2 requires those failures to be
    # negotiated *before* the first collective, so the whole local phase is
    # captured into one verdict.
    stage = None
    layout = None
    stage_module = None
    parameters = None
    policy = None
    compute_dtype = torch.float32
    wire = None
    build_error: tuple[str, str] | None = None
    try:
        validate_runtime_config(config, world_size=world_size)
        wire = resolve_moss_wire_spec(
            model,
            latent_shape=latent_shape,
            max_text_tokens=max_text_tokens,
            microbatches=int(config.microbatches),
        )
        is_last = int(rank) == int(config.stages) - 1
        if not is_last and scaler is not None:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                "a non-last stage never sees the loss and must not own a GradScaler",
            )
        compute_dtype = dtype if dtype is not None else _model_dtype(model)
        policy = PpComputePolicy.from_dtype(compute_dtype)
        layout, stage_module = build_moss_stage_module(
            model,
            stage_index=int(rank),
            stages=int(config.stages),
            latent_shape=latent_shape,
            max_text_tokens=int(max_text_tokens),
            network=network,
            compute_policy=policy,
        )
        stage_parameters = list(stage_module.trainable_parameters())
        validate_master_weight_dtype(
            stage_parameters, mixed_precision=config.mixed_precision, device=device,
        )
        if optimizer is None and optimizer_factory is not None:
            # A stage without stage-local trainable parameters cannot be a PP
            # rank; this is named here so the negotiation reports it as a
            # pp_* failure instead of an optimizer crash after the barrier.
            if not stage_parameters:
                raise PipelineParallelError(
                    CODE_STAGE_CONTRACT,
                    f"rank {rank} owns no stage-local trainable parameters (no plain "
                    "LoRA adapter is attached to its layers); a PP stage-local "
                    "optimizer needs at least one owned adapter",
                )
            optimizer = optimizer_factory(stage_module)
        parameters = (
            [p for g in optimizer.param_groups for p in g["params"]]
            if optimizer is not None
            else stage_parameters
        )
        validate_master_weight_dtype(
            parameters, mixed_precision=config.mixed_precision, device=device,
        )
        validate_distributed_backend(
            _distributed_backend(group),
            device_type=str(getattr(device, "type", device)),
            world_size=int(world_size),
        )
        input_args, output_args = moss_boundary_metas(
            model, layout, wire=wire, policy=policy,
            latent_dtype=policy.activation_dtype,
        )
        stage = FrozenBoundaryPipelineStage(
            stage_module,
            int(rank),
            int(config.stages),
            device,
            input_args=tuple(input_args),
            output_args=tuple(output_args),
            group=group,
            frozen_input_indices=() if layout.is_first else MOSS_BOUNDARY_FROZEN_INDICES,
            frozen_output_indices=() if layout.is_last else MOSS_BOUNDARY_FROZEN_INDICES,
        )
    except Exception as exc:  # noqa: BLE001 - negotiated below, never rank-local
        build_error = (getattr(exc, "code", CODE_STAGE_CONTRACT),
                       f"rank {rank}: {type(exc).__name__}: {exc}")
    _negotiate_phase(build_error, group, where="pp_moss_build:local")

    # ---- collective phase ----------------------------------------------------
    assert_compute_policy_agreement(policy, group=group, where="pp_moss_build")
    frozen_error: tuple[str, str] | None = None
    try:
        assert_frozen_boundary_agreement(stage, group=group)
    except Exception as exc:  # noqa: BLE001 - negotiated below
        frozen_error = (getattr(exc, "code", CODE_STAGE_CONTRACT),
                        f"rank {rank}: {type(exc).__name__}: {exc}")
    _negotiate_phase(frozen_error, group, where="pp_moss_build:boundary")
    transaction = None
    # C2: the fp16 transaction opens with broadcasts; a half-transactional
    # pipeline (one rank built an optimizer, the other did not) would strand
    # the optimizer rank inside that broadcast.  Every rank takes part in this
    # capability gather, and both refuse with the same named error before it.
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        gathered: list[Any] = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, {"has_optimizer": optimizer is not None}, group=group)
        if len({entry["has_optimizer"] for entry in gathered if entry}) != 1:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                "the ranks disagree about owning a stage-local optimizer "
                f"({gathered}); a PP pipeline cannot be half-transactional "
                "-- every rank needs stage-local trainable parameters and an "
                "optimizer, or none does",
            )
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
        base_loss = make_last_stage_loss_fn(
            is_last=layout.is_last, microbatches=config.microbatches
        )
        if layout.is_last:
            base_loss = transaction.wrap_loss(base_loss)
    else:
        base_loss = make_last_stage_loss_fn(
            is_last=layout.is_last, microbatches=config.microbatches
        )
    schedule = build_anima_schedule(
        stage,
        microbatches=config.microbatches,
        schedule=config.schedule,
        loss_fn=base_loss,
    )
    return MossPipelineRuntime(
        model=model,
        layout=layout,
        stage_module=stage_module,
        stage=stage,
        schedule=schedule,
        config=config,
        loss_fn=base_loss,
        wire=wire,
        dtype=compute_dtype,
        compute_policy=policy,
        transaction=transaction,
        optimizer=optimizer,
        group=group,
    )


def _distributed_backend(group) -> str | None:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None
    if group is not None and hasattr(dist, "get_backend"):
        try:
            return str(dist.get_backend(group))
        except (ValueError, RuntimeError):  # pragma: no cover - exotic backends
            return None
    return str(dist.get_backend())


def _model_dtype(model) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


__all__ = [
    "MOSS_BOUNDARY_FROZEN_INDICES",
    "MossPipelineRuntime",
    "MossWireSpec",
    "build_moss_pipeline_runtime",
    "moss_boundary_metas",
    "resolve_moss_wire_spec",
]
