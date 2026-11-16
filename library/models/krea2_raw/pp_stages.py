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
