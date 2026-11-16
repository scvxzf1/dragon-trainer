"""Krea-2 PP v0 runtime wiring (probe level).

Mirrors the MOSS runtime skeleton: freeze the wire geometry once per run, derive
this stage's boundary metadata from it, then let ``torch.distributed.pipelining``
allocate the receive buffers.  Only the geometry half is wired here -- the stage
modules themselves live in :mod:`library.models.krea2_raw.pp_stages`.

Wire slots (fixed order, frozen indices are gradient-free):

===  ============  ==================================  ==========
idx  name          shape                               frozen
===  ============  ==================================  ==========
0    hidden        ``[mb, token_cap, features]``       no
1    tvec          ``[mb, 1, 6*features]``             yes
2    freqs         ``[mb, token_cap, headdim//2, 2, 2]`` yes
3    mask          ``[mb, token_cap] `` float32         yes
4    tembed        ``[mb, 1, features]``                yes
===  ============  ==================================  ==========

``token_cap`` is ``max_text_tokens + img_tokens`` rounded up to 256, matching the
padding ``SingleStreamDiT.forward`` applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from library.models.krea2_raw.pp_stages import krea2_token_cap
from library.training.pp.contract import PpComputePolicy
from library.training.pp.errors import PipelineParallelError

__all__ = [
    "KREA2_BOUNDARY_FROZEN_INDICES",
    "Krea2WireSpec",
    "resolve_krea2_wire_spec",
    "krea2_boundary_metas",
]

#: Wire slots that carry no gradient (everything except the residual stream).
KREA2_BOUNDARY_FROZEN_INDICES: tuple[int, ...] = (1, 2, 3, 4)

#: Latent channels of Krea-2 (patchified into the token stream).
KREA2_LATENT_CHANNELS = 16


@dataclass(frozen=True)
class Krea2WireSpec:
    """The frozen per-run wire geometry (per-microbatch shapes)."""

    latent_shape: tuple[int, ...]
    microbatch_latent_shape: tuple[int, ...]
    max_text_tokens: int
    token_cap: int
    dim: int
    img_tokens: int
    heads: int

    @property
    def batch_size(self) -> int:
        return int(self.latent_shape[0])

    @property
    def microbatch_batch_size(self) -> int:
        return int(self.microbatch_latent_shape[0])

    @property
    def headdim(self) -> int:
        return int(self.dim) // int(self.heads)

    @property
    def freqs_shape(self) -> tuple[int, ...]:
        return (
            self.microbatch_batch_size,
            self.token_cap,
            self.headdim // 2,
            2,
            2,
        )


def resolve_krea2_wire_spec(
    model,
    *,
    latent_shape: Sequence[int],
    max_text_tokens: int,
    microbatches: int,
) -> Krea2WireSpec:
    """Validate the run geometry and freeze the microbatch wire shapes."""

    latent = tuple(int(value) for value in latent_shape)
    if len(latent) != 5:
        raise PipelineParallelError(
            "pp_krea2_latent_shape_invalid",
            f"Krea-2 expects a 5D latent (B, C, T=1, H, W); got {latent}",
        )
    batch, channels, temporal, height, width = latent
    if channels != KREA2_LATENT_CHANNELS or temporal != 1:
        raise PipelineParallelError(
            "pp_krea2_latent_shape_invalid",
            f"Krea-2 expects (B, {KREA2_LATENT_CHANNELS}, 1, H, W); got {latent}",
        )
    if int(microbatches) <= 0 or batch % int(microbatches):
        raise PipelineParallelError(
            "pp_krea2_batch_not_divisible",
            f"train batch size {batch} is not divisible by "
            f"pipeline_parallel_microbatches={microbatches}; PP v0 freezes the "
            "wire buffers so every microbatch must be the same size",
        )
    patch = int(getattr(model, "PATCH_SIZE", 2))
    if height % patch or width % patch:
        raise PipelineParallelError(
            "pp_krea2_latent_shape_invalid",
            f"latent {height}x{width} is not divisible by patch size {patch}",
        )
    img_tokens = (height // patch) * (width // patch)
    config = getattr(model, "config", None)
    dim = int(getattr(config, "features"))
    heads = int(getattr(config, "heads"))
    return Krea2WireSpec(
        latent_shape=latent,
        microbatch_latent_shape=(batch // int(microbatches), *latent[1:]),
        max_text_tokens=int(max_text_tokens),
        token_cap=krea2_token_cap(int(max_text_tokens), img_tokens),
        dim=dim,
        img_tokens=img_tokens,
        heads=heads,
    )


def _meta(shape: Sequence[int], dtype: torch.dtype) -> Tensor:
    return torch.empty(tuple(int(value) for value in shape), dtype=dtype, device="meta")


def krea2_boundary_metas(
    stage,
    *,
    wire: Krea2WireSpec,
    policy: PpComputePolicy,
    latent_dtype: torch.dtype,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """``(input_args, output_args)`` metadata for this Krea-2 stage.

    Unlike Anima, Krea-2 keeps its residual stream in the *activation* dtype when
    ``fp32_residual`` is off, so the boundary dtype follows
    ``policy.activation_dtype`` rather than Anima's promoted ``boundary_dtype``.
    """

    dtype = policy.activation_dtype
    mb = wire.microbatch_batch_size
    boundary = (
        _meta((mb, wire.token_cap, wire.dim), dtype),
        _meta((mb, 1, 6 * wire.dim), dtype),
        _meta(wire.freqs_shape, dtype),
        # torch pipelining force-enables requires_grad on received tensors, so
        # the boolean mask rides as float32 (exact for 0/1) instead of bool.
        _meta((mb, wire.token_cap), torch.float32),
        _meta((mb, 1, wire.dim), dtype),
    )
    if stage.is_first:
        return (_meta(wire.microbatch_latent_shape, latent_dtype),), boundary
    prediction = _meta((mb, wire.token_cap, KREA2_LATENT_CHANNELS), dtype)
    return boundary, (prediction,)
