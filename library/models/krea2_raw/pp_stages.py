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
    context = model.txtfusion(context.to(compute_dtype), mask=txtmask).to(compute_dtype)
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
    return final[:, state.txtlen : state.txtlen + state.imglen, :]
