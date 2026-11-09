"""Anima stage modules and the torch 2.10 frozen-boundary grad wire.

Two things live here.

1. :class:`AnimaFirstStage` / :class:`AnimaLastStage` — non-owning stage views.
   They keep a reference to the *whole* production model (so the placement
   builder never mutates it) plus an :class:`AnimaBlockStage` slice produced by
   ``build_anima_block_stage``.  ``object.__setattr__`` is used for both, because
   registering them would make ``stage.to()`` / ``stage.state_dict()`` reach into
   the frozen base model.

2. :class:`FrozenBoundaryPipelineStage` — the torch 2.10 fix for frozen
   conditioning that *does* cross the boundary (``conditioning="wire"``).

Why the grad wire needs a paired override
-----------------------------------------
torch 2.10's ``PipelineStage`` has no ``input_grads`` / ``output_grads``
constructor arguments (those appear in 2.12), and
``_create_grad_send_info`` mirrors ``args_recv_info`` unconditionally: every
received input is assumed to need a gradient sent back, "it is up to the previous
stage to discard this gradient".  The receiving side builds ``grad_recv_info``
from ``act_send_info`` -- i.e. from *all* forward outputs -- completely
independently of what the sender decides.  Suppressing the send alone therefore
does not remove the receive; it turns a symmetric exchange into a one-sided one
and the job **hangs** instead of raising.

The only 2.10-compatible implementation is therefore paired, on both stages:

* ``_create_grad_recv_info`` (earlier stage) allocates buffers for the kept
  outputs only;
* ``_retrieve_recv_grads`` (earlier stage) re-expands those buffers into a full
  ``output_grads`` tuple with ``None`` in the dropped slots;
* ``_create_grad_send_info`` (later stage) returns ``None`` for the dropped
  inputs, so ``get_bwd_send_ops`` creates no op for them.

Because a one-sided override hangs rather than raising,
:func:`assert_frozen_boundary_agreement` compares the two keep-lists across ranks
*before* the first step.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.pipelining import PipelineStage

from .conditioning import attention_params, rope_for_shape, timestep_modulation
from .contract import PpComputePolicy
from .errors import (
    CODE_COMPUTE_POLICY_MISMATCH,
    CODE_FROZEN_BOUNDARY_MISMATCH,
    CODE_STAGE_CONTRACT,
    PipelineParallelError,
)
from .layout import model_pred_to_loss_4d
from .numerics import (
    AnimaPipelineUnsupported,
    assert_mini_forward_supported,
    assert_mod_guidance_indexable,
    finish_model,
    mod_guidance_probe,
    run_stage_blocks,
)

#: Frozen conditioning is rebuilt on the later stage; only ``hidden`` crosses.
CONDITIONING_LOCAL = "local"
#: Frozen conditioning crosses the boundary and its gradient is dropped.
CONDITIONING_WIRE = "wire"

#: Wire-mode boundary order: ``(hidden, t_embedding, crossattn_emb, adaln_lora)``.
WIRE_FROZEN_INDICES: tuple[int, ...] = (1, 2, 3)


def frozen_boundary_indices(conditioning: str) -> tuple[int, ...]:
    """Boundary slots whose gradient must be dropped.

    ``local`` conditioning puts no frozen tensor on the wire at all, so both
    stages get an empty set and the base torch behaviour is used unchanged.
    """

    if conditioning == CONDITIONING_LOCAL:
        return ()
    if conditioning == CONDITIONING_WIRE:
        return WIRE_FROZEN_INDICES
    raise PipelineParallelError(
        CODE_STAGE_CONTRACT,
        f"unknown conditioning mode {conditioning!r}; expected "
        f"{CONDITIONING_LOCAL!r} or {CONDITIONING_WIRE!r}",
    )


def _default_compute_policy(model) -> PpComputePolicy:
    """Fallback policy for a stage built without an explicit pipeline policy.

    Only single-stage unit tests take this path; the runtime always passes the
    shared :class:`PpComputePolicy`.  The model is read through ``getattr`` because
    the capability guards in this module are also exercised with a plain
    ``SimpleNamespace`` stand-in for the DiT.
    """

    parameters = getattr(model, "parameters", None)
    dtype = None
    if callable(parameters):
        try:
            dtype = next(parameters()).dtype
        except StopIteration:  # pragma: no cover - parameterless stub model
            dtype = None
    return PpComputePolicy.from_dtype(dtype or torch.float32)


class _AnimaStageBase(nn.Module):
    """Non-owning view of one Anima stage over an ``AnimaBlockStage`` slice."""

    def __init__(
        self,
        *,
        model: nn.Module,
        block_stage: nn.Module,
        stage_index: int,
        stages: int,
        block_range: Sequence[int],
        conditioning: str,
        adapter_modules: Sequence[nn.Module] = (),
        compute_policy: PpComputePolicy | None = None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_block_stage", block_stage)
        object.__setattr__(self, "_adapter_modules", tuple(adapter_modules))
        self.stage_index = int(stage_index)
        self.stages = int(stages)
        self.block_range = (int(block_range[0]), int(block_range[1]))
        self.conditioning = str(conditioning)
        # The pipeline's single ``use_fp32`` decision.  It is *not* re-derived
        # from the tensors this stage receives: under fp16 the boundary is fp32
        # (the block stack promotes the residual stream), so a boundary-derived
        # flag would be False and the tail would overflow in fp16 (D1).
        self.compute_policy = (
            compute_policy
            if compute_policy is not None
            else _default_compute_policy(model)
        )
        # ``library.anima.mini_forward`` owns the capability guards and raises its
        # own named error type.  A stage module is part of *this* package, so a
        # direct construction (bypassing ``placement.build_anima_stage_module``,
        # which already converts) must not leak a second error type: every
        # rejection from ``library.training.pp`` is a ``PipelineParallelError``
        # carrying the same stable ``pp_*`` code (see the package docstring).
        try:
            assert_mini_forward_supported(model)
            assert_mod_guidance_indexable(
                mod_guidance_probe(model, list(block_stage.blocks))
            )
        except AnimaPipelineUnsupported as exc:
            raise PipelineParallelError(exc.code, exc.detail) from exc

    @property
    def use_fp32(self) -> bool:
        """The pipeline-wide ``use_fp32``; identical on every stage."""

        return bool(self.compute_policy.use_fp32)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def block_stage(self) -> nn.Module:
        return self._block_stage

    @property
    def adapter_modules(self) -> tuple[nn.Module, ...]:
        """The adapter modules attached to this stage's blocks.

        They are held non-owningly on purpose: LoRA modules live in the adapter
        network (``network.unet_loras``) and are *not* children of the blocks they
        patch, so a stage cannot discover them from its own ``parameters()``.
        """

        return self._adapter_modules

    @property
    def is_first(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        return self.stage_index == self.stages - 1

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """Trainable parameters this stage owns.

        Both sources are covered: adapter modules attached to the owned blocks,
        and any trainable parameter that genuinely lives inside an owned block
        (method adapters such as soft tokens register there).
        """

        seen: set[int] = set()
        for module in self._adapter_modules:
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter
        for block in self._block_stage.blocks:
            for parameter in block.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

    def adapter_key_map(self, local_key: str) -> str:
        """Map a stage-local ``blocks.N...`` key back to the global block id."""

        return self._block_stage.global_state_dict_key(local_key)

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


class AnimaFirstStage(_AnimaStageBase):
    """Embedding + timestep modulation + the first contiguous block range."""

    def forward(
        self,
        latents: Tensor,
        *,
        timesteps: Tensor,
        crossattn_emb: Tensor,
        padding_mask: Tensor | None = None,
        fps: Tensor | None = None,
        h_offset: int = 0,
        w_offset: int = 0,
    ) -> Any:
        model = self._model
        hidden, rope_cos_sin = model.prepare_embedded_sequence(
            latents,
            fps=fps,
            padding_mask=padding_mask,
            h_offset=int(h_offset),
            w_offset=int(w_offset),
        )
        self._require_pre_block_dtype(hidden)
        embedding, adaln_lora = timestep_modulation(model, timesteps)
        hidden = run_stage_blocks(
            self._block_stage,
            model,
            hidden,
            embedding,
            crossattn_emb,
            attention_params(model),
            compute_dtype=self.compute_policy.activation_dtype,
            rope_cos_sin=rope_cos_sin,
            adaln_lora=adaln_lora,
        )
        if self.conditioning == CONDITIONING_LOCAL:
            return hidden
        self._require_grad_free((embedding, crossattn_emb, adaln_lora))
        return hidden, embedding, crossattn_emb, adaln_lora

    def _require_pre_block_dtype(self, hidden: Tensor) -> None:
        """The runtime must have been built for the dtype it is actually fed.

        The first stage is the only place that sees the pre-block tensor, so it is
        the only place that can prove the pipeline-wide ``use_fp32`` matches
        reality.  A mismatch here (for example building with ``dtype=None`` while
        the trainer runs an fp16 autocast) would otherwise surface as the exact
        silent fp16 overflow this policy exists to prevent.
        """

        expected = self.compute_policy.activation_dtype
        if hidden.dtype != expected:
            raise PipelineParallelError(
                CODE_COMPUTE_POLICY_MISMATCH,
                f"the pre-block hidden dtype is {hidden.dtype} but the pipeline "
                f"compute policy was built for {expected} "
                f"(use_fp32={self.use_fp32}); build the runtime with "
                f"dtype={hidden.dtype} so every stage shares one use_fp32 decision",
            )

    @staticmethod
    def _require_grad_free(tensors: Sequence[Tensor | None]) -> None:
        """Frozen outputs must carry no autograd edge at all.

        ``stage_backward`` would otherwise append them to the backward tensor
        list with a ``None`` grad slot, and ``torch.autograd.backward`` cannot
        handle a ``None`` gradient for a non-scalar output.  Failing here turns a
        confusing autograd error into a named contract violation.
        """

        for index, tensor in enumerate(tensors, start=1):
            if tensor is not None and tensor.requires_grad:
                raise PipelineParallelError(
                    CODE_STAGE_CONTRACT,
                    f"frozen boundary output {index} requires grad "
                    f"(shape {tuple(tensor.shape)}); the PP wire can only drop "
                    "gradients for tensors that carry no autograd edge",
                )


class AnimaLastStage(_AnimaStageBase):
    """The remaining block range + final layer + unpatchify, and the loss target."""

    def forward(
        self,
        hidden: Tensor,
        *frozen: Tensor,
        timesteps: Tensor | None = None,
        crossattn_emb: Tensor | None = None,
        padding_mask: Tensor | None = None,
        fps: Tensor | None = None,
        h_offset: int = 0,
        w_offset: int = 0,
    ) -> Tensor:
        model = self._model
        # ``padding_mask`` is accepted so both ranks can be driven with one kwargs
        # dict; only the pre-block region (``concat_padding_mask``) consumes it.
        del padding_mask
        embedding, context, adaln_lora = self._resolve_conditioning(
            frozen, timesteps=timesteps, crossattn_emb=crossattn_emb
        )
        rope_cos_sin = rope_for_shape(
            model,
            hidden.shape,
            fps=fps,
            h_offset=int(h_offset),
            w_offset=int(w_offset),
        )
        hidden = run_stage_blocks(
            self._block_stage,
            model,
            hidden,
            embedding,
            context,
            attention_params(model),
            compute_dtype=self.compute_policy.activation_dtype,
            rope_cos_sin=rope_cos_sin,
            adaln_lora=adaln_lora,
        )
        projected = finish_model(
            model, hidden, embedding, adaln_lora, use_fp32=self.use_fp32
        )
        return model_pred_to_loss_4d(projected)

    def _resolve_conditioning(self, frozen: Sequence[Tensor], *, timesteps, crossattn_emb):
        """Return ``(t_embedding, crossattn_emb, adaln_lora)`` for this stage.

        In ``wire`` mode the three tensors arrive as activations and are
        **detached**: the receive buffers are forced ``requires_grad_(True)`` by
        torch (``stage.py:1549-1551``), so without a detach they would accumulate
        a ``.grad`` and ``get_bwd_send_ops`` would try to send it.  Detaching is
        the receiving-side drop the PP contract asks for; the paired send-side
        suppression lives in :class:`FrozenBoundaryPipelineStage`.
        """

        if self.conditioning == CONDITIONING_LOCAL:
            if frozen:
                raise PipelineParallelError(
                    CODE_STAGE_CONTRACT,
                    f"local conditioning expects only 'hidden' from the previous "
                    f"stage, got {len(frozen)} extra tensor(s)",
                )
            if timesteps is None or crossattn_emb is None:
                raise PipelineParallelError(
                    CODE_STAGE_CONTRACT,
                    "local conditioning requires the per-microbatch timesteps and "
                    "crossattn_emb keyword arguments on the last stage",
                )
            embedding, adaln_lora = timestep_modulation(self._model, timesteps)
            return embedding, crossattn_emb, adaln_lora
        if len(frozen) != len(WIRE_FROZEN_INDICES):
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                "wire conditioning expects (hidden, t_embedding, crossattn_emb, "
                f"adaln_lora) from the previous stage, got {len(frozen) + 1} "
                "tensor(s)",
            )
        embedding, context, adaln_lora = (value.detach() for value in frozen)
        return embedding, context, adaln_lora


class FrozenBoundaryPipelineStage(PipelineStage):
    """``PipelineStage`` that drops gradients for frozen boundary slots.

    See the module docstring for why both sides must be overridden together.
    With empty index sets this class is behaviourally identical to
    ``PipelineStage``, which is what ``conditioning="local"`` uses.
    """

    def __init__(
        self,
        *args: Any,
        frozen_input_indices: Sequence[int] = (),
        frozen_output_indices: Sequence[int] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._frozen_input_indices = tuple(sorted({int(i) for i in frozen_input_indices}))
        self._frozen_output_indices = tuple(
            sorted({int(i) for i in frozen_output_indices})
        )
        self._validate_indices()

    @property
    def frozen_input_indices(self) -> tuple[int, ...]:
        return self._frozen_input_indices

    @property
    def frozen_output_indices(self) -> tuple[int, ...]:
        return self._frozen_output_indices

    def _validate_indices(self) -> None:
        inputs_meta = self.inputs_meta or ()
        outputs_meta = getattr(self, "_outputs_meta", None) or ()
        for index in self._frozen_input_indices:
            if not 0 <= index < len(inputs_meta):
                raise PipelineParallelError(
                    CODE_STAGE_CONTRACT,
                    f"frozen_input_indices {self._frozen_input_indices} out of range "
                    f"for {len(inputs_meta)} stage inputs",
                )
        for index in self._frozen_output_indices:
            if not 0 <= index < len(outputs_meta):
                raise PipelineParallelError(
                    CODE_STAGE_CONTRACT,
                    f"frozen_output_indices {self._frozen_output_indices} out of "
                    f"range for {len(outputs_meta)} stage outputs",
                )

    # -- paired overrides ----------------------------------------------------

    def _create_grad_send_info(self, args_recv_info) -> list:
        """Send no gradient for a frozen received input.

        torch's version appends ``a.source`` for every ``_RecvInfo`` regardless of
        ``requires_grad``; that is the send half of the pair.
        """

        destinations = super()._create_grad_send_info(args_recv_info)
        for index in self._frozen_input_indices:
            destinations[index] = None
        return destinations

    def _create_grad_recv_info(self, act_send_info):
        """Allocate receive buffers for the kept outputs only."""

        infos = super()._create_grad_recv_info(act_send_info)
        if not self._frozen_output_indices:
            return infos
        keep = [
            index
            for index in range(len(infos))
            if index not in self._frozen_output_indices
        ]
        return tuple(infos[index] for index in keep)

    def _retrieve_recv_grads(self, bwd_chunk_id: int):
        """Re-expand the kept grads into a full ``output_grads`` tuple."""

        grads = super()._retrieve_recv_grads(bwd_chunk_id)
        if not self._frozen_output_indices:
            return grads
        full: list[Any] = [None] * len(self.get_outputs_meta())
        keep = [
            index
            for index in range(len(full))
            if index not in self._frozen_output_indices
        ]
        if len(keep) != len(grads):
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                f"stage {self.stage_index} received {len(grads)} gradient buffer(s) "
                f"for {len(keep)} kept output(s); the grad wire is asymmetric",
            )
        for index, grad in zip(keep, grads, strict=True):
            full[index] = grad
        return tuple(full)


def assert_frozen_boundary_agreement(
    stage: FrozenBoundaryPipelineStage,
    *,
    group: Any = None,
    where: str = "pp_stage_build",
) -> None:
    """Fail closed when the two stages disagree about the frozen slots.

    An asymmetric override does not raise at runtime -- it deadlocks, because one
    side waits for a send the other never issues.  Comparing the keep-lists across
    ranks at build time is the only place that failure can be reported as an
    error.
    """

    if not (dist.is_available() and dist.is_initialized()):
        return
    payload = (
        int(stage.stage_index),
        tuple(int(i) for i in stage.frozen_input_indices),
        tuple(int(i) for i in stage.frozen_output_indices),
    )
    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, payload, group=group)
    by_stage = {item[0]: (tuple(item[1]), tuple(item[2])) for item in gathered}
    for index in sorted(by_stage):
        if index not in by_stage:
            raise PipelineParallelError(
                CODE_FROZEN_BOUNDARY_MISMATCH,
                f"{where}: stage {index} is missing from the gathered topology "
                f"{sorted(by_stage)}",
            )
        frozen_out = by_stage[index][1]
        following = by_stage.get(index + 1)
        if following is None:
            continue
        if frozen_out != following[0]:
            raise PipelineParallelError(
                CODE_FROZEN_BOUNDARY_MISMATCH,
                f"{where}: stage {index} drops gradients for outputs {frozen_out} "
                f"but stage {index + 1} drops them for inputs {following[0]}; a "
                "one-sided override deadlocks instead of raising, so this is "
                "refused at build time",
            )


__all__ = [
    "CONDITIONING_LOCAL",
    "CONDITIONING_WIRE",
    "WIRE_FROZEN_INDICES",
    "AnimaFirstStage",
    "AnimaLastStage",
    "FrozenBoundaryPipelineStage",
    "assert_frozen_boundary_agreement",
    "frozen_boundary_indices",
]
