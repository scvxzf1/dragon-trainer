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
import math

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
    freqs: Tensor | None = None,
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
        if freqs is None:
            pos = F.pad(pos, (0, 0, 0, fold))

    mask4d = _mask(mask)
    if freqs is None:
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


    def _resolve_cached_freqs(
        self,
        *,
        batch: int,
        txtlen: int,
        latent_hw: tuple[int, int],
        patch: int,
        device: torch.device,
    ) -> Tensor:
        cache = getattr(self, "_freqs_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_freqs_cache", cache)
        key = (int(batch), int(txtlen), int(latent_hw[0]), int(latent_hw[1]), int(patch), device)
        cached = cache.get(key)
        if cached is not None:
            return cached
        from library.models.krea2_raw.family import _cached_krea2_pos_and_imgmask

        pos, _ = _cached_krea2_pos_and_imgmask(
            int(batch),
            int(latent_hw[0]) // int(patch),
            int(latent_hw[1]) // int(patch),
            int(txtlen),
            device,
        )
        fold = (-int(pos.shape[1])) % 256
        if fold > 0:
            pos = F.pad(pos, (0, 0, 0, fold))
        with torch.no_grad():
            freqs = self.model.posemb(pos).detach()
        if len(cache) >= 8:
            cache.pop(next(iter(cache)))
        cache[key] = freqs
        return freqs


class Krea2FirstStage(_Krea2StageBase):
    """Input/text fusion + timestep projection + the leading block range."""

    def __init__(
        self,
        *,
        max_text_tokens: int,
        img_tokens: int,
        patch: int = 2,
        latent_hw: tuple[int, int] | None = None,
        local_freqs: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_max_text_tokens", int(max_text_tokens))
        object.__setattr__(self, "_img_tokens", int(img_tokens))
        object.__setattr__(self, "_patch", int(patch))
        object.__setattr__(
            self,
            "_latent_hw",
            None if latent_hw is None else (int(latent_hw[0]), int(latent_hw[1])),
        )
        object.__setattr__(self, "_local_freqs", bool(local_freqs))

    @property
    def max_text_tokens(self) -> int:
        return self._max_text_tokens  # type: ignore[attr-defined]

    @property
    def img_tokens(self) -> int:
        return self._img_tokens  # type: ignore[attr-defined]

    @property
    def local_freqs(self) -> bool:
        return bool(self._local_freqs)  # type: ignore[attr-defined]

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
    ) -> tuple[Tensor, ...]:
        """Take the *5D latent* (the ``forward_for_loss`` contract) and patchify."""

        from library.models.krea2_raw.family import prepare_img_tokens

        latents_4d = latents.squeeze(2)
        batch, _, height, width = latents_4d.shape
        txtlen = int(context.shape[1])
        if mask is None:
            txtmask = torch.ones(batch, txtlen, dtype=torch.bool, device=latents_4d.device)
        else:
            txtmask = mask if mask.dtype == torch.bool else mask > 0.5
        patch = int(self._patch)
        img_tokens, pos, joint_mask = prepare_img_tokens(
            latents_4d, txtlen, patch, txtmask
        )
        freqs = self._resolve_cached_freqs(
            batch=batch,
            txtlen=txtlen,
            latent_hw=(int(height), int(width)),
            patch=patch,
            device=latents_4d.device,
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
                freqs=freqs,
            )
        if self.local_freqs:
            return (
                state.hidden,
                state.tvec.detach(),
                state.mask.to(torch.float32),
                state.tembed.detach(),
            )
        return (
            state.hidden,
            state.tvec.detach(),
            state.freqs.detach(),
            state.mask.to(torch.float32),
            state.tembed.detach(),
        )


class Krea2LastStage(_Krea2StageBase):
    """The trailing block range + final layer + the image text-slice."""

    def __init__(
        self,
        *,
        max_text_tokens: int,
        img_tokens: int,
        patch: int = 2,
        latent_hw: tuple[int, int] | None = None,
        local_freqs: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_max_text_tokens", int(max_text_tokens))
        object.__setattr__(self, "_img_tokens", int(img_tokens))
        object.__setattr__(self, "_patch", int(patch))
        object.__setattr__(
            self,
            "_latent_hw",
            None if latent_hw is None else (int(latent_hw[0]), int(latent_hw[1])),
        )
        object.__setattr__(self, "_local_freqs", bool(local_freqs))

    @property
    def local_freqs(self) -> bool:
        return bool(self._local_freqs)  # type: ignore[attr-defined]

    def forward(
        self,
        hidden: Tensor,
        tvec: Tensor,
        *wire_tail: Tensor,
        **_shared_kwargs,
    ) -> Tensor:
        if len(wire_tail) == 2:
            mask_float, tembed = wire_tail
            latent_hw = self._latent_hw  # type: ignore[attr-defined]
            if latent_hw is None:
                raise RuntimeError("Krea2LastStage with local_freqs requires latent_hw")
            freqs = self._resolve_cached_freqs(
                batch=int(hidden.shape[0]),
                txtlen=int(self._max_text_tokens),
                latent_hw=latent_hw,
                patch=int(self._patch),
                device=hidden.device,
            )
        elif len(wire_tail) == 3:
            freqs, mask_float, tembed = wire_tail
            freqs = freqs.detach()
        else:
            raise ValueError(
                f"Krea2LastStage expected 4 or 5 wire tensors; got {2 + len(wire_tail)}"
            )
        state = Krea2StageState(
            hidden=hidden,
            tvec=tvec.detach(),
            freqs=freqs,
            mask=mask_float > 0.5,
            tembed=tembed.detach(),
            txtlen=self._max_text_tokens,  # type: ignore[attr-defined]
            imglen=self._img_tokens,  # type: ignore[attr-defined]
        )
        with self._autocast():
            return krea2_forward_last(self.model, self.blocks, state)


class _SwappedBlockSlot:
    """Contiguous pinned-CPU master slabs and parameter slice table for one block."""

    def __init__(self, block: nn.Module, *, global_index: int, pin: bool) -> None:
        self.block = block
        self.global_index = int(global_index)
        by_dtype: dict[torch.dtype, list[nn.Parameter]] = {}
        for param in block.parameters():
            if param.requires_grad:
                continue
            by_dtype.setdefault(param.dtype, []).append(param)

        self.pinned_slabs: dict[torch.dtype, Tensor] = {}
        self.specs: list[tuple[nn.Parameter, torch.dtype, int, tuple[int, ...], Tensor]] = []
        for dt, params in by_dtype.items():
            total = sum(p.numel() for p in params)
            slab = torch.empty(total, dtype=dt, device="cpu", pin_memory=pin)
            offset = 0
            for param in params:
                numel = param.numel()
                shape = tuple(param.shape)
                cpu_view = slab.narrow(0, offset, numel).view(shape)
                cpu_view.copy_(param.data.detach(), non_blocking=False)
                param._cpu_pinned = cpu_view  # type: ignore[attr-defined]
                param.data = cpu_view
                self.specs.append((param, dt, offset, shape, cpu_view))
                offset += numel
            self.pinned_slabs[dt] = slab


class _AsyncStageBlockSwapper:
    """Ping-pong H2D prefetcher on a dedicated CUDA ``copy_stream``."""

    def __init__(self, slots: list[_SwappedBlockSlot], device: torch.device) -> None:
        self.slots = slots
        self.device = torch.device(device)
        all_ckpt = all(
            bool(getattr(s.block, "gradient_checkpointing", False)) for s in slots
        )
        # Checkpointed blocks discard saved tensors during outer forward and
        # recompute one block at a time during backward, so 2 ping-pong slots
        # bound GPU residency for any num_swap; non-checkpointed blocks must
        # retain one slot per swapped block until backward unwinds them.
        self.ring_size = (
            max(1, min(2, len(slots))) if all_ckpt else max(1, len(slots))
        )
        self.is_cuda = self.device.type == "cuda"
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.is_cuda else None
        self.max_elems: dict[torch.dtype, int] = {}
        for slot in slots:
            for dt, pslab in slot.pinned_slabs.items():
                self.max_elems[dt] = max(self.max_elems.get(dt, 0), int(pslab.numel()))
        self.ring_slabs: list[dict[torch.dtype, Tensor] | None] = [None] * self.ring_size
        self.ring_owner: list[int | None] = [None] * self.ring_size
        self.ring_ready: list[torch.cuda.Event | None] = [None] * self.ring_size
        self.ring_done: list[torch.cuda.Event | None] = [None] * self.ring_size

    def prefetch(self, slot_pos: int) -> None:
        if not self.is_cuda or not (0 <= slot_pos < len(self.slots)):
            return
        r = slot_pos % self.ring_size
        if self.ring_owner[r] == slot_pos and self.ring_slabs[r] is not None:
            return
        slot = self.slots[slot_pos]
        assert self.copy_stream is not None
        slabs = self.ring_slabs[r]
        if slabs is None:
            # Allocate on compute_stream so freed ring slabs return directly to
            # compute_stream's CUDACachingAllocator pool (zero dual-pool bloat).
            compute_stream = torch.cuda.current_stream(self.device)
            slabs = {
                dt: torch.empty(n, dtype=dt, device=self.device)
                for dt, n in self.max_elems.items()
            }
            self.ring_slabs[r] = slabs
            alloc_ev = torch.cuda.Event()
            alloc_ev.record(compute_stream)
            self.copy_stream.wait_event(alloc_ev)
        done_ev = self.ring_done[r]
        if done_ev is not None:
            self.copy_stream.wait_event(done_ev)
        with torch.cuda.stream(self.copy_stream):
            for dt, pslab in slot.pinned_slabs.items():
                slabs[dt].narrow(0, 0, pslab.numel()).copy_(pslab, non_blocking=True)
            ready_ev = torch.cuda.Event()
            ready_ev.record(self.copy_stream)
        self.ring_owner[r] = slot_pos
        self.ring_ready[r] = ready_ev

    def bind_and_prefetch_next(
        self, slot_pos: int, *, next_slot_pos: int | None
    ) -> None:
        if not self.is_cuda:
            return
        r = slot_pos % self.ring_size
        if self.ring_owner[r] != slot_pos or self.ring_slabs[r] is None:
            self.prefetch(slot_pos)
        compute_stream = torch.cuda.current_stream(self.device)
        ready_ev = self.ring_ready[r]
        if ready_ev is not None:
            compute_stream.wait_event(ready_ev)
            self.ring_ready[r] = None
        slabs = self.ring_slabs[r]
        assert slabs is not None
        slot = self.slots[slot_pos]
        for param, dt, offset, shape, _cpu_view in slot.specs:
            param.data = slabs[dt].narrow(0, offset, math.prod(shape)).view(shape)
        if (
            next_slot_pos is not None
            and 0 <= next_slot_pos < len(self.slots)
            and (next_slot_pos % self.ring_size) != r
        ):
            self.prefetch(next_slot_pos)

    def release(self, slot_pos: int, *, drain_pool: bool) -> None:
        slot = self.slots[slot_pos]
        for param, _dt, _offset, _shape, cpu_view in slot.specs:
            param.data = cpu_view
        if not self.is_cuda:
            return
        r = slot_pos % self.ring_size
        compute_stream = torch.cuda.current_stream(self.device)
        done_ev = torch.cuda.Event()
        done_ev.record(compute_stream)
        self.ring_done[r] = done_ev
        if drain_pool:
            for idx in range(self.ring_size):
                slabs = self.ring_slabs[idx]
                if slabs is not None:
                    for slab in slabs.values():
                        slab.record_stream(compute_stream)
                    self.ring_slabs[idx] = None
                self.ring_owner[idx] = None
                self.ring_ready[idx] = None


class _OffloadAfterBackward(torch.autograd.Function):
    """Release a swapped block's GPU slot once backward reaches its input ``x``."""

    @staticmethod
    def forward(ctx, x: Tensor, swapper: _AsyncStageBlockSwapper, slot_pos: int) -> Tensor:
        ctx.swapper = swapper
        ctx.slot_pos = int(slot_pos)
        return x

    @staticmethod
    def backward(ctx, grad_x: Tensor):
        ctx.swapper.release(ctx.slot_pos, drain_pool=(ctx.slot_pos == 0))
        return grad_x, None, None


class _PrefetchOnBackwardEntry(torch.autograd.Function):
    """Kick off H2D prefetch of ``last_slot`` as soon as backward enters the block after it."""

    @staticmethod
    def forward(ctx, y: Tensor, swapper: _AsyncStageBlockSwapper, slot_pos: int) -> Tensor:
        ctx.swapper = swapper
        ctx.slot_pos = int(slot_pos)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        ctx.swapper.prefetch(ctx.slot_pos)
        return grad_y, None, None


def enable_stage_block_swap(stage_blocks, device, num_swap: int = 2) -> list[int]:
    """Swap the base weights of the first ``num_swap`` non-leading stage blocks.

    Packs each swapped block's frozen parameters into a contiguous pinned CPU
    slab and streams them through a 2-slot GPU staging ring on a dedicated CUDA
    ``copy_stream``, overlapping H2D DMA with neighbor block compute and
    draining both staging slots to 0 bytes outside the swapped block range.
    """

    target_device = torch.device(device)
    owned = [
        (index, block)
        for index, block in enumerate(stage_blocks)
        if any(True for _ in block.parameters())
        and all(
            p.device.type == target_device.type
            for p in block.parameters()
            if not p.requires_grad
        )
    ]
    if len(owned) <= 1 or num_swap <= 0:
        return []

    pin = target_device.type == "cuda"
    slots: list[_SwappedBlockSlot] = []
    for global_index, block in owned[1 : 1 + int(num_swap)]:
        slot = _SwappedBlockSlot(block, global_index=global_index, pin=pin)
        if slot.specs:
            slots.append(slot)
    if not slots:
        return []

    swapper = _AsyncStageBlockSwapper(slots, target_device)
    last_slot_pos = len(slots) - 1

    # Trigger forward H2D prefetch of slot 0 while the leading resident block computes.
    lead_block = owned[0][1]
    lead_orig_forward = lead_block.forward

    def _lead_forward(*args, _orig=lead_orig_forward, _s=swapper, **kwargs):
        _s.prefetch(0)
        return _orig(*args, **kwargs)

    lead_block.forward = _lead_forward

    # Trigger backward H2D prefetch of the last swapped block when backward enters
    # the first resident block immediately following the swapped range.
    trailing_resident_idx = 1 + len(slots)
    if trailing_resident_idx < len(owned):
        trail_block = owned[trailing_resident_idx][1]
        trail_orig_forward = trail_block.forward

        def _trail_forward(*args, _orig=trail_orig_forward, _s=swapper, _last=last_slot_pos, **kwargs):
            out = _orig(*args, **kwargs)
            if torch.is_grad_enabled() and isinstance(out, Tensor) and out.requires_grad:
                out = _PrefetchOnBackwardEntry.apply(out, _s, _last)
            return out

        trail_block.forward = _trail_forward

    swapped: list[int] = []
    for slot_pos, slot in enumerate(slots):
        block = slot.block
        original_forward = block.forward
        original_inner = block._forward
        block._in_outer_forward = False
        block._swap_index = slot.global_index
        block._stage_swapper = swapper

        def _wrapped_forward(
            *args,
            _orig=original_forward,
            _b=block,
            _s=swapper,
            _pos=slot_pos,
            _last=last_slot_pos,
            **kwargs,
        ):
            retain_for_eager_bwd = bool(
                torch.is_grad_enabled()
                and not getattr(_b, "gradient_checkpointing", False)
            )
            _b._in_outer_forward = True
            _s.bind_and_prefetch_next(_pos, next_slot_pos=_pos + 1)
            ok = False
            try:
                out = _orig(*args, **kwargs)
                ok = True
                return out
            finally:
                _b._in_outer_forward = False
                if not (ok and retain_for_eager_bwd):
                    _s.release(_pos, drain_pool=(_pos == _last or not ok))

        def _wrapped_inner(
            x,
            *args,
            _orig=original_inner,
            _b=block,
            _s=swapper,
            _pos=slot_pos,
            **kwargs,
        ):
            if not _b._in_outer_forward:
                _s.bind_and_prefetch_next(_pos, next_slot_pos=_pos - 1)
            # Attach in BOTH outer forward and recompute: under non-reentrant
            # checkpointing (use_reentrant=False), outer forward constructs the
            # real backward graph while recompute must execute the identical op
            # sequence to satisfy saved-tensor hook replay.
            if torch.is_grad_enabled() and isinstance(x, Tensor) and x.requires_grad:
                x = _OffloadAfterBackward.apply(x, _s, _pos)
            return _orig(x, *args, **kwargs)

        block.forward = _wrapped_forward
        block._forward = _wrapped_inner
        swapped.append(slot.global_index)
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

        def _traced(x, *args, _orig=original, _idx=index, _blk=block, **kwargs):
            outer = not getattr(_blk, "_in_outer_forward", True)
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
