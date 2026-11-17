"""Krea-2 pipeline-parallel stage split (probe level).

``SingleStreamDiT.forward`` is one monolithic method: input projection, text
fusion, timestep projection, the 28-block stack, the final layer and the
text-slice all live inside it.  A pipeline runtime needs that method cut into a
*leading* half (everything before the trailing block range) and a *trailing* half
(the remaining blocks + final layer + text slice), with an explicit tensor
hand-off between them.

This module is the probe cut: two pure functions that reproduce the monolith
**bit-exactly** when chained.  It deliberately does not touch the block
container, ownership, autocast policy or the pipeline schedule -- those belong to
the real wiring step.  The companion test
``tests/test_krea2_pp_stage_split.py`` asserts equality against
``model.forward`` on a tiny model.

Hand-off contents (``Krea2StageState``):

======================  ===========================  ==================
field                   shape                        grad
======================  ===========================  ==================
``hidden``              ``[B, token_cap, features]``  yes
``tvec``                ``[B, features]``             no (per step)
``freqs``               ``[B, token_cap, ...]``       no (per geometry)
``mask``                ``[B, token_cap]`` bool       no (per batch)
``tembed``              ``[B, 1, features]``          no (per step)
``txtlen`` / ``imglen`` python ints                   --
======================  ===========================  ==================

``token_cap`` is ``txtlen + imglen`` rounded up to the next multiple of 256 --
the same padding ``SingleStreamDiT.forward`` applies to stabilise compiled kernel
shapes.  Every field is fixed-shape for a given latent geometry, which is what a
pipeline boundary needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import re
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from library.models.krea2_raw.dit import _mask, temb

__all__ = [
    "Krea2StageState",
    "krea2_forward_first",
    "krea2_forward_last",
    "krea2_token_cap",
]


@dataclass
class Krea2StageState:
    """Everything the trailing half needs from the leading half."""

    hidden: Tensor
    tvec: Tensor
    freqs: Tensor
    mask: Tensor
    tembed: Tensor
    txtlen: int
    imglen: int


def krea2_token_cap(txtlen: int, imglen: int) -> int:
    """Padded joint sequence length, matching ``SingleStreamDiT.forward``."""

    total = int(txtlen) + int(imglen)
    pad = (-total) % 256
    return total + pad


def _run_block_range(
    blocks: nn.ModuleList | list[nn.Module],
    hidden: Tensor,
    tvec: Tensor,
    freqs: Tensor,
    mask4d: Tensor,
) -> Tensor:
    for block in blocks:
        hidden = block(hidden, tvec, freqs, mask4d)
    return hidden




def _txtprefix_checkpoint(
    module: nn.Module, hidden: Tensor, mask: Tensor
) -> Tensor:
    """Checkpoint ``txtfusion`` so its internals do not stay resident all stage."""

    from torch.utils.checkpoint import checkpoint as _checkpoint

    return _checkpoint(
        lambda h: module(h, mask=mask), hidden, use_reentrant=False
    )


def krea2_forward_first(
    model: nn.Module,
    blocks: nn.ModuleList | list[nn.Module],
    img: Tensor,
    *,
    timesteps: Tensor,
    context: Tensor,
    pos: Tensor,
    mask: Tensor,
) -> Krea2StageState:
    """Run ``model.forward`` up to (and including) the leading block range."""

    compute_dtype = img.dtype
    img = model.first(img).to(compute_dtype)

    t = model.tmlp(
        temb(timesteps, model.config.tdim, device=img.device, dtype=compute_dtype)
    ).to(compute_dtype)
    tvec = model.tproj(t).to(compute_dtype)

    txtmask = _mask(mask[:, : context.shape[1]])
    # txtfusion is the biggest first-stage-only module (0.64 GiB) and its input is
    # the full (B, 256, 12, 2560) text tensor, i.e. a 30720-wide sequence.  Measured
    # on the real 12.82B weights, its *internals* pin ~890 MiB until every block
    # finishes, because enable_gradient_checkpointing() only walks model.blocks and
    # never covers this prefix.  Recomputing it costs one extra text pass and buys
    # the whole 890 MiB back on the tightest stage.
    _txt_in = context.to(compute_dtype)
    if torch.is_grad_enabled() and model.training:
        context = _txtprefix_checkpoint(model.txtfusion, _txt_in, txtmask)
    else:
        context = model.txtfusion(_txt_in, mask=txtmask)
    context = context.to(compute_dtype)
    context = model.txtmlp(context).to(compute_dtype)

    txtlen, imglen = context.shape[1], img.shape[1]
    combined = torch.cat((context, img), dim=1).to(compute_dtype)

    # Same 256-multiple padding the monolith applies; mask and pos are padded in
    # lockstep so the trailing half can rebuild the 4D mask without the caller
    # having to know the padded length.
    fold = (-combined.shape[1]) % 256
    if fold > 0:
        combined = F.pad(combined, (0, 0, 0, fold))
        mask = F.pad(mask, (0, fold), value=False)
        pos = F.pad(pos, (0, 0, 0, fold))

    mask4d = _mask(mask)
    freqs = model.posemb(pos)
    hidden = _run_block_range(blocks, combined, tvec, freqs, mask4d)
    return Krea2StageState(
        hidden=hidden,
        tvec=tvec,
        freqs=freqs,
        mask=mask,
        tembed=t,
        txtlen=txtlen,
        imglen=imglen,
    )


def krea2_forward_last(
    model: nn.Module,
    blocks: nn.ModuleList | list[nn.Module],
    state: Krea2StageState,
) -> Tensor:
    """Run the trailing block range, the final layer and the image text-slice."""

    compute_dtype = state.hidden.dtype
    mask4d = _mask(state.mask)
    hidden = _run_block_range(blocks, state.hidden, state.tvec, state.freqs, mask4d)
    final = model.last(hidden, state.tembed).to(compute_dtype)
    # .contiguous(): the text-slice is a *view* of the padded joint sequence, so
    # its stride still carries token_cap and [[torch pipelining]] rejects it
    # against the frozen boundary metadata (expected (256, 64, 1), got
    # (16384, 64, 1)).  The slice is tiny, the copy is free.
    # ``.contiguous()`` is NOT enough here: the batch dim is 1, and torch treats a
    # size-1 dim's stride as irrelevant, so the slice stays "contiguous" and keeps
    # the parent's stride (16384, 64, 1) instead of the (256, 64, 1) the frozen
    # boundary metadata declares.  pipelining validates stride strictly, so force
    # a real allocation in contiguous format.
    sliced = final[:, state.txtlen : state.txtlen + state.imglen, :]
    out = sliced.clone(memory_format=torch.contiguous_format)
    import os as _os
    if _os.environ.get('KREA2_PP_TRACE'):
        print(f'[Krea2LastStage] out={tuple(out.shape)} stride={out.stride()} '
              f'contig={out.is_contiguous()} txtlen={state.txtlen} imglen={state.imglen}', flush=True)
    return out


# ---------------------------------------------------------------------------
# Stage modules (probe level)
# ---------------------------------------------------------------------------

#: Submodules that only the leading stage needs (they feed the joint sequence).
_FIRST_ONLY_SUBMODULES = ("first", "tmlp", "tproj", "txtfusion", "txtmlp")
#: Submodules that only the trailing stage needs.
_LAST_ONLY_SUBMODULES = ("last",)


def release_foreign_submodules(model: nn.Module, *, is_first: bool) -> tuple[str, ...]:
    """Drop the pre-block / post-block submodules this stage does not own.

    ``release_foreign_blocks`` only trims ``model.blocks``; on Krea-2 the 1.24 GiB
    that lives **outside** the block container is what actually decides whether
    the trailing stage fits: measured 13.37 GiB with it vs 12.13 GiB without, on
    a 16 GB card.  A leading stage keeps the fusions and drops ``last``; a
    trailing stage keeps ``last`` and drops the fusions.

    Destructive and opt-in, same contract as ``release_foreign_blocks``: call it
    after the stage module holds its own references.
    """

    dropped: list[str] = []
    names = _LAST_ONLY_SUBMODULES if is_first else _FIRST_ONLY_SUBMODULES
    for name in names:
        if getattr(model, name, None) is not None:
            setattr(model, name, None)
            dropped.append(name)
    return tuple(dropped)


class _Krea2StageBase(nn.Module):
    """Shared plumbing for the two probe stages.

    ``model`` and the block slice are attached with ``object.__setattr__`` on
    purpose: registering them as submodules would make ``stage.to(device)`` walk
    into the frozen base model and move weights this rank must not touch.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        blocks,
        stage_index: int,
        stages: int,
        block_range: tuple[int, int],
        activation_dtype: torch.dtype | None = None,
        network: object | None = None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_network", network)
        object.__setattr__(self, "_blocks", list(blocks))
        object.__setattr__(self, "_stage_index", int(stage_index))
        object.__setattr__(self, "_stages", int(stages))
        object.__setattr__(self, "_block_range", (int(block_range[0]), int(block_range[1])))
        object.__setattr__(self, "_activation_dtype", activation_dtype)

    # -- read-only views -----------------------------------------------------
    @property
    def model(self) -> nn.Module:
        return self._model  # type: ignore[attr-defined]

    @property
    def blocks(self) -> list[nn.Module]:
        return self._blocks  # type: ignore[attr-defined]

    @property
    def adapter_modules(self) -> tuple[nn.Module, ...]:
        """This stage's adapter modules.

        Needed by the PP checkpoint writer: it classifies every trainable
        parameter as "inside an adapter module" or not, and refuses to save the
        latter (they could not be restored).  Krea-2 adapters live on the network
        object, not as children of the blocks they patch, so a stage cannot
        discover them from its own ``parameters()``.
        """

        network = getattr(self, "_network", None)
        if network is None:
            return ()
        start, stop = self._block_range  # type: ignore[attr-defined]
        selected: list[nn.Module] = []
        for module in getattr(network, "unet_loras", []) or []:
            name = str(getattr(module, "lora_name", ""))
            # lora_name looks like "lora_unet_blocks_12_attn_wq".
            match = re.search(r"(?:^|_)blocks[_.](\d+)[_.]", name)
            if match is None or start <= int(match.group(1)) < stop:
                selected.append(module)
        return tuple(selected)

    @property
    def stage_index(self) -> int:
        return self._stage_index  # type: ignore[attr-defined]

    @property
    def stages(self) -> int:
        return self._stages  # type: ignore[attr-defined]

    @property
    def block_range(self) -> tuple[int, int]:
        return self._block_range  # type: ignore[attr-defined]

    @property
    def is_first(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        return self.stage_index == self.stages - 1

    def trainable_parameters(self):
        """Parameters this stage actually owns and can update.

        Not ``model.parameters()``: a PP rank must only optimise its own block
        range plus the pre/post submodules that live on its side, otherwise the
        two ranks would both update the same weights and the frozen foreign
        halves would silently be trained twice.
        """

        model = self._model  # type: ignore[attr-defined]
        modules: list[nn.Module] = list(self._blocks)  # type: ignore[attr-defined]
        names = _FIRST_ONLY_SUBMODULES if self.is_first else _LAST_ONLY_SUBMODULES
        for name in names:
            module = getattr(model, name, None)
            if module is not None:
                modules.append(module)
        seen: set[int] = set()
        owned: list[nn.Parameter] = []
        for module in modules:
            for parameter in module.parameters():
                if not parameter.requires_grad or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                owned.append(parameter)
        # Krea-2 adapters are *not* inserted into the base module tree -- apply_to
        # keeps them on the network object and patches the forward instead -- so
        # walking ``model`` alone finds zero adapter parameters.  Collect the
        # adapter tensors whose lora_name falls inside this rank's block range.
        network = getattr(self, "_network", None)
        if network is not None:
            start, stop = self._block_range  # type: ignore[attr-defined]
            for lora in getattr(network, "unet_loras", []) or []:
                name = str(getattr(lora, "lora_name", ""))
                # lora_name looks like "lora_unet_blocks_12_attn_wq".
                match = re.match(r"(?:.*_)?blocks[_.](\d+)[_.]", name)
                if match and not (start <= int(match.group(1)) < stop):
                    continue
                for parameter in lora.parameters():
                    if not parameter.requires_grad or id(parameter) in seen:
                        continue
                    seen.add(id(parameter))
                    owned.append(parameter)
        return owned

    def _autocast(self):
        """Mirror ``_AnimaStageBase._trainer_autocast`` for the fp16 contract.

        fp16 policy promotes the residual stream to fp32 while the weights stay
        fp16, so every Linear needs an ambient fp16 autocast region to run at
        all.  Anything that is not fp16 (bf16 / fp32 / CPU) gets a null context.
        """

        import contextlib

        dtype = self._activation_dtype  # type: ignore[attr-defined]
        if dtype != torch.float16:
            return contextlib.nullcontext()
        device = next(self._model.parameters()).device  # type: ignore[attr-defined]
        if device.type == "cpu":
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=torch.float16)


class Krea2FirstStage(_Krea2StageBase):
    """Input/text fusion + timestep projection + the leading block range."""

    def __init__(
        self, *, max_text_tokens: int, img_tokens: int, patch: int = 2, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_max_text_tokens", int(max_text_tokens))
        object.__setattr__(self, "_img_tokens", int(img_tokens))
        object.__setattr__(self, "_patch", int(patch))

    @property
    def max_text_tokens(self) -> int:
        return self._max_text_tokens  # type: ignore[attr-defined]

    @property
    def img_tokens(self) -> int:
        return self._img_tokens  # type: ignore[attr-defined]

    @property
    def token_cap(self) -> int:
        return krea2_token_cap(self.max_text_tokens, self.img_tokens)

    def forward(
        self,
        latents: Tensor,
        *,
        timesteps: Tensor,
        context: Tensor,
        mask: Tensor | None = None,
        **_shared_kwargs,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Take the *5D latent* (the ``forward_for_loss`` contract) and patchify.

        Patchification lives here rather than in the runtime so both pipeline
        ranks receive exactly the same kwargs -- pipelining splits kwargs the
        same way on every rank, and an empty kwargs set on the trailing rank
        makes it subscript ``None``.
        """

        from library.models.krea2_raw.family import prepare_img_tokens

        latents_4d = latents.squeeze(2)
        batch = int(latents_4d.shape[0])
        txtlen = int(context.shape[1])
        if mask is None:
            txtmask = torch.ones(batch, txtlen, dtype=torch.bool, device=latents_4d.device)
        else:
            txtmask = mask if mask.dtype == torch.bool else mask > 0.5
        img_tokens, pos, joint_mask = prepare_img_tokens(
            latents_4d, txtlen, int(self._patch), txtmask
        )
        with self._autocast():
            state = krea2_forward_first(
                self.model,
                self.blocks,
                img_tokens,
                timesteps=timesteps,
                context=context,
                pos=pos,
                mask=joint_mask,
            )
        # The wire is a fixed list of tensors; the boolean mask travels as
        # float32-encoded 0/1 so every slot has a stable dtype/shape.
        #
        # Everything except ``hidden`` is declared frozen in
        # KREA2_BOUNDARY_FROZEN_INDICES, so it must *really* be detached:
        # ``tvec = model.tproj(t)`` and ``tembed = model.tmlp(...)`` carry
        # gradients whenever their own weights are trainable, and [[torch
        # pipelining]] rejects a frozen slot that arrives carrying gradients
        # ("[1] for chunk 0 has gradients ...").
        return (
            state.hidden,
            state.tvec.detach(),
            state.freqs.detach(),
            state.mask.to(torch.float32),
            state.tembed.detach(),
        )


class Krea2LastStage(_Krea2StageBase):
    """The trailing block range + final layer + the image text-slice."""

    def __init__(self, *, max_text_tokens: int, img_tokens: int, **kwargs) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_max_text_tokens", int(max_text_tokens))
        object.__setattr__(self, "_img_tokens", int(img_tokens))

    def forward(
        self,
        hidden: Tensor,
        tvec: Tensor,
        freqs: Tensor,
        mask_float: Tensor,
        tembed: Tensor,
        **_shared_kwargs,
    ) -> Tensor:
        # ``_shared_kwargs`` absorbs the kwargs the schedule hands to *every*
        # stage (``timesteps``/``context``/``mask``); only the leading stage
        # consumes them.  The wire mask slot is named ``mask_float`` on purpose
        # so it cannot collide with a ``mask`` kwarg.
        #
        # The frozen slots are detached on the *receiving* side too: the blocks
        # consume ``tvec``/``freqs``, so without this their gradients flow back
        # into a slot the pipeline already declared gradient-free and torch
        # aborts with "expecting to send gradients to stage None".
        state = Krea2StageState(
            hidden=hidden,
            tvec=tvec.detach(),
            freqs=freqs.detach(),
            mask=mask_float > 0.5,
            tembed=tembed.detach(),
            txtlen=self._max_text_tokens,  # type: ignore[attr-defined]
            imglen=self._img_tokens,  # type: ignore[attr-defined]
        )
        with self._autocast():
            return krea2_forward_last(self.model, self.blocks, state)


class _OffloadAfterBackward(torch.autograd.Function):
    """Offload a swappable block's base weights once its gradient reaches x.

    Under Schedule1F1B with 2 microbatches a rank runs F(mb0) -> F(mb1) ->
    B(mb0) -> B(mb1), so a stateful prefetching offloader (which assumes the
    last forward is immediately followed by the last backward) breaks on
    F(mb1).  This hook is stateless: the checkpoint's recompute of ``_forward``
    loads the weights, and they are released only when every operator inside
    that block has finished back-propagating -- signalled by the gradient
    arriving at the block's input activation.
    """

    @staticmethod
    def forward(ctx, x: Tensor, block) -> Tensor:  # noqa: D102 - autograd API
        ctx.block = block
        return x

    @staticmethod
    def backward(ctx, grad_x: Tensor):  # noqa: D102 - autograd API
        block = ctx.block
        for parameter in block.parameters():
            pinned = getattr(parameter, "_cpu_pinned", None)
            if pinned is not None:
                parameter.data = pinned
        return grad_x, None


def enable_stage_block_swap(stage_blocks, device, num_swap: int = 2) -> list[int]:
    """Swap the base weights of the first ``num_swap`` non-leading blocks.

    Each Krea-2 block's frozen fp16 base is 809 MiB while its LoRA lives on the
    network object (not in ``block.parameters()``), so offloading the block's
    parameters releases exactly the base weights and leaves every trainable
    adapter tensor, gradient and optimizer state on the GPU.

    Starts at index 1 so the activation entering a swapped block already passed
    the previous block, i.e. ``x.requires_grad`` is guaranteed True and the
    offload hook fires.

    Returns the swapped block indices for logging.
    """

    # ``stage_blocks`` is the *whole* model's block list; only the blocks this
    # rank actually kept on the device belong to its stage (the rest were
    # released by release_foreign_submodules and never moved here).  Selecting
    # by position alone picked blocks 1..4 on the trailing rank too -- those
    # live on the leading rank, so offloading them freed 0 MiB and the swap did
    # nothing exactly where it was needed.
    owned = [
        (index, block)
        for index, block in enumerate(stage_blocks)
        if all(p.device.type == device.type for p in block.parameters())
    ]
    blocks = [block for _, block in owned]
    swapped: list[int] = []
    # Start at the second owned block: the activation entering the first one may
    # still be the stage input, whose requires_grad the offload hook relies on.
    for position, (global_index, block) in enumerate(owned[1 : 1 + num_swap], start=1):
        pinned_resident = False
        for parameter in block.parameters():
            if not hasattr(parameter, "_cpu_pinned"):
                parameter._cpu_pinned = (
                    parameter.data.detach().to("cpu", non_blocking=False).pin_memory()
                )
            parameter.data = parameter._cpu_pinned
            pinned_resident = True
        if not pinned_resident:
            continue

        original_forward = block.forward
        original_inner = block._forward
        block._in_outer_forward = False

        block._swap_index = global_index

        def _load(b=block):
            for parameter in b.parameters():
                pinned = getattr(parameter, "_cpu_pinned", None)
                if pinned is not None and parameter.data.device.type == "cpu":
                    # A fresh buffer, not a view of the pinned copy: the pinned
                    # tensor keeps its own cpu storage and this one is what the
                    # autograd graph will reference for this pass.
                    gpu_buf = torch.empty_like(pinned, device=device)
                    gpu_buf.copy_(pinned, non_blocking=True)
                    parameter._gpu_active = gpu_buf
                    parameter.data = gpu_buf

        def _offload(b=block):
            for parameter in b.parameters():
                pinned = getattr(parameter, "_cpu_pinned", None)
                gpu_buf = getattr(parameter, "_gpu_active", None)
                if pinned is not None:
                    parameter.data = pinned
                if gpu_buf is not None:
                    # Repointing ``.data`` is not enough: any transposed view
                    # saved by a linear's backward (``weight.t()``) keeps a
                    # strong reference to the same StorageImpl, so the 809 MiB
                    # stays resident even though no parameter points at it.
                    # Shrinking the storage is the FSDP way to actually hand the
                    # block back, and the recompute path re-creates the buffer
                    # from the pinned copy anyway.
                    try:
                        gpu_buf.untyped_storage().resize_(0)
                    except RuntimeError:
                        pass
                    parameter._gpu_active = None

        def _wrapped_forward(*args, _orig=original_forward, _b=block, _l=_load, _o=_offload, **kwargs):
            _b._in_outer_forward = True
            _l()
            try:
                return _orig(*args, **kwargs)
            finally:
                _b._in_outer_forward = False
                _o()

        def _wrapped_inner(x, *args, _orig=original_inner, _b=block, _l=_load, **kwargs):
            if not _b._in_outer_forward:
                # Called by the checkpoint's recompute, not by the outer forward.
                _l()
                if x.requires_grad:
                    x = _OffloadAfterBackward.apply(x, _b)
            return _orig(x, *args, **kwargs)

        block.forward = _wrapped_forward
        block._forward = _wrapped_inner
        swapped.append(global_index)
    return swapped


def enable_block_probes(stage_blocks, rank: int) -> int:
    """Log every block's ``_forward`` entry/exit with the live cuda allocation.

    The OOM report is not diagnostic: ``allocated`` at the moment of failure is
    by construction the ceiling minus the failing request, so every run prints
    the same number.  What actually distinguishes a working memory strategy is
    *how far backward gets* and *whether the footprint falls as it unwinds* --
    hence an entry/exit trace per block.
    """

    def _alloc_gib() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / 2**30

    count = 0
    for index, block in enumerate(stage_blocks):
        original = block._forward

        def _traced(x, *args, _orig=original, _idx=index, **kwargs):
            outer = not getattr(block, "_in_outer_forward", True)
            print(
                f"[blk] r{rank} idx={_idx} enter outer={not outer} "
                f"alloc={_alloc_gib():.2f}GiB",
                flush=True,
            )
            out = _orig(x, *args, **kwargs)
            print(
                f"[blk] r{rank} idx={_idx} exit  outer={not outer} "
                f"alloc={_alloc_gib():.2f}GiB",
                flush=True,
            )
            return out

        block._forward = _traced
        count += 1
    return count
