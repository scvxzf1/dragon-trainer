# Refs:
#   https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
#   https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import contextlib
import random

import torch
from library.log import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


def _absorb_channel_scale(
    weight: torch.Tensor, channel_scale: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """SmoothQuant-style channel-scale absorption into a Linear's input columns.

    Mutates ``weight`` ([out, in]) so ``W[:, c] *= s_norm[c]`` and returns
    ``inv_scale = 1 / s_norm`` (caller applies ``x * inv_scale`` at forward).
    Output is unchanged; the point is to rebalance per-column gradient magnitudes
    so each column's ``∂L/∂W[:,c]`` no longer scales with ``|x[c]|^2``.
    See ``bench/channel_stats/channel_dominance_analysis.md``.
    """
    assert channel_scale.ndim == 1, (
        f"channel_scale must be 1D, got shape {tuple(channel_scale.shape)}"
    )
    assert channel_scale.shape[0] == weight.shape[1], (
        f"channel_scale length {channel_scale.shape[0]} does not match "
        f"weight in_features {weight.shape[1]}"
    )
    s = channel_scale.detach().to(dtype=torch.float32).clamp_min(eps)
    s = s / s.mean().clamp_min(eps)
    with torch.no_grad():
        weight.mul_(s.to(weight).unsqueeze(0))
    # inv_scale must live on the same device as the weight it rebalances: ``s``
    # is seeded from the calibration file (CPU), but the buffer has to track
    # ``weight`` so the forward multiply and the save-time bake never straddle
    # cuda/cpu. fp32 storage is intentional — only the device moves.
    return (1.0 / s).to(weight.device).contiguous()


def _chunked_down(module, x: torch.Tensor, work: torch.dtype):
    """``_down(_rebalance(x.to(work)))`` over sequence slices.

    The upcast itself is the allocation: for an attention projection
    ``x.to(fp32)`` is 4352 * 6144 * 4 = 102.00 MiB, the single request that
    OOM'd backward on a 16 GB card.  ``_down`` projects down to ``rank``
    dimensions, so only the fp32 *input* window is large, and it is per-token:
    slicing the sequence is exact.  The concatenated result keeps ``_down``'s
    own dtype/dim, so ``_gate``/dropout/``_up`` downstream are untouched.
    """

    if work != torch.float32 or x.ndim < 3:
        return module._down(module._rebalance(x.to(work)), work)
    seq = int(x.shape[-2])
    chunks = 8 if seq % 8 == 0 and seq >= 2048 else 1
    if chunks == 1:
        return module._down(module._rebalance(x.to(work)), work)
    step = seq // chunks
    return torch.cat(
        [
            module._down(
                module._rebalance(
                    x[..., index * step : (index + 1) * step, :].contiguous().to(work)
                ),
                work,
            )
            for index in range(chunks)
        ],
        dim=-2,
    )


def _chunked_up(module, lx: torch.Tensor, work: torch.dtype, out_dtype: torch.dtype):
    """Run ``module._up`` over sequence slices, casting each slice on the way out.

    ``work`` is fp32 on sm70 with fp16 mixed precision (Bootstrap auto-enables
    ``lora_fp32_compute`` there for stability), so the up projection materialises
    ``[B, L, out_features]`` in fp32: 4352 * 6144 * 4 = 102.00 MiB for one
    attention projection -- the exact allocation that OOM'd backward.  Casting
    per slice keeps every fp32 window at 102/chunks while the concatenated
    result is fp16, i.e. the same tensor the caller builds one statement later.
    64 * 6144 * 4 = 1.5 MiB is the same arithmetic for the MLP's first half.

    Numerically identical: ``_up`` is a per-token linear map, so slicing the
    sequence is exact and casting element-wise commutes with concatenation.
    """

    if work != torch.float32 or lx.ndim < 3:
        return module._up(lx, work).to(out_dtype)
    seq = int(lx.shape[-2])
    chunks = 8 if seq % 8 == 0 and seq >= 2048 else 1
    if chunks == 1:
        return module._up(lx, work).to(out_dtype)
    step = seq // chunks
    # Pre-allocate and copy slices in place.  ``torch.cat`` would hold every
    # fp32 slice *and* the concatenated result at once, which is what kept the
    # peak at ~102 MiB even after chunking; filling a preallocated buffer means
    # only the finished fp16 output plus one fp32 slice are ever resident.
    out = None
    for index in range(chunks):
        piece = module._up(
            lx[..., index * step : (index + 1) * step, :].contiguous(), work
        ).to(out_dtype)
        if out is None:
            out = torch.empty(
                (*piece.shape[:-2], seq, piece.shape[-1]),
                dtype=out_dtype,
                device=piece.device,
            )
        out[..., index * step : (index + 1) * step, :] = piece
        del piece
    assert out is not None
    return out


class BaseLoRAModule(torch.nn.Module):
    """Shared scaffolding: alpha→scale, multiplier, dropouts, channel_scale,
    timestep masking, ``apply_to`` monkey-patching. Subclasses own ``forward``."""

    supports_conv2d: bool = False
    supports_packed_qkv: bool = False

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d" and not self.supports_conv2d:
            raise ValueError(f"{type(self).__name__} does not support Conv2d")

        self.lora_dim = lora_dim
        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.fp32_compute = False

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self._has_channel_scale = False
        # Default all-ones mask → identity multiply. Prefer applying the mask
        # only on the training path (scaffold: `_gate` runs after the eval
        # early-return; custom forwards should gate with `self.training`).
        # Always-a-tensor avoids None-vs-Tensor branches under torch.compile.
        # T-LoRA rebinds via LoRANetwork.set_timestep_mask.
        self.register_buffer(
            "_timestep_mask",
            torch.ones(1, lora_dim, dtype=torch.float32),
            persistent=False,
        )
        self.enabled = True

    def _register_channel_scale(
        self,
        target_weight: torch.Tensor,
        channel_scale,
        *,
        linear_only: bool = True,
    ) -> None:
        if channel_scale is None:
            return
        if linear_only and target_weight.dim() != 2:
            raise ValueError(
                "channel_scale is only supported for Linear LoRA modules, "
                f"got weight with dim {target_weight.dim()}"
            )
        inv_scale = _absorb_channel_scale(target_weight, channel_scale)
        self.register_buffer("inv_scale", inv_scale, persistent=True)
        self._has_channel_scale = True

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _skip_module(self) -> bool:
        return (
            self.module_dropout is not None
            and self.training
            and random.random() < self.module_dropout
        )

    def _rebalance(self, x: torch.Tensor) -> torch.Tensor:
        # inv_scale stays fp32 in storage (calibration precision); cast at
        # the multiply site so ``bf16 × fp32 → bf16`` instead of being
        # promoted back to fp32. The total fp32 buffer is in_features × 4 B
        # × N_modules — ~1 MiB on Anima, negligible vs activations.
        if not self._has_channel_scale:
            return x
        return x * self.inv_scale.to(device=x.device, dtype=x.dtype)

    def _apply_rank_dropout(self, lx: torch.Tensor):
        if self.rank_dropout is not None and self.training:
            mask = (
                torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                > self.rank_dropout
            )
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            lx = lx * mask
            return lx, self.scale * (1.0 / (1.0 - self.rank_dropout))
        return lx, self.scale

    def _rank_compute_dtype(
        self, org_forwarded: torch.Tensor, default_dtype: torch.dtype | None = None
    ) -> torch.dtype:
        base_dtype = default_dtype or org_forwarded.dtype
        if self.training and self.fp32_compute and org_forwarded.dtype == torch.float16:
            return torch.float32
        return base_dtype

    def _rank_autocast_context(self, x: torch.Tensor, work: torch.dtype):
        if work == torch.float32 and self.training and self.fp32_compute:
            return torch.autocast(device_type=x.device.type, enabled=False)
        return contextlib.nullcontext()

    def forward(self, x):
        if not self.enabled or getattr(self, "_fused", False):
            return self.org_forward(x)

        # Training with an fp32-master adapter keeps the base output and the
        # fp32 delta alive at the same time: 4352 * 6144 * 2 = 51 MiB each, and
        # the producer below only ever shrinks the *individual* windows.  Running
        # the whole adapter (base projection + delta) per sequence slice is the
        # only way to reduce the total: then just one slice's base and delta
        # coexist, and the preallocated output holds the finished fp16 result.
        seq = int(x.shape[-2]) if x.ndim >= 3 else 0
        chunks = 8 if self.training and seq % 8 == 0 and seq >= 2048 else 1
        if chunks == 1 or not self._uses_fp32_compute():
            return BaseLoRAModule.forward_with_base(self, x, self.org_forward(x))
        step = seq // chunks
        out = None
        for index in range(chunks):
            lo, hi = index * step, (index + 1) * step
            piece = x[..., lo:hi, :]
            value = BaseLoRAModule.forward_with_base(
                self, piece, self.org_forward(piece)
            )
            if out is None:
                out = torch.empty(
                    (*value.shape[:-2], seq, value.shape[-1]),
                    dtype=value.dtype,
                    device=value.device,
                )
            out[..., lo:hi, :] = value
            del value, piece
        assert out is not None
        return out

    def _uses_fp32_compute(self) -> bool:
        return bool(self.training and self.fp32_compute)

    def forward_with_base(self, x: torch.Tensor, org_forwarded: torch.Tensor) -> torch.Tensor:
        """Apply this adapter to a base output computed by a shared projection."""
        if not self.enabled or getattr(self, "_fused", False):
            return org_forwarded

        if not self.training:
            delta = self._eval_delta(x, org_forwarded)
            return org_forwarded + delta.to(org_forwarded.dtype)

        if self._skip_module():
            return org_forwarded

        work = self._rank_compute_dtype(org_forwarded)
        up_module = getattr(self, "lora_up", None)
        can_scale_in_bottleneck = getattr(up_module, "bias", None) is None
        with self._rank_autocast_context(x, work):
            lx = _chunked_down(self, x, work)
            lx = self._gate(lx, work)
            if self.dropout is not None:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)
            lx, scale = self._apply_rank_dropout(lx)
            lx = lx.to(work)
            if can_scale_in_bottleneck:
                # Scale inside the rank-r bottleneck before expanding to
                # [B, L, out_features]: avoids two full-output fp16 temporaries
                # and extra MulBackward0 graph state (~237 MiB saved per MLP).
                lx = lx * (self.multiplier * scale)
            lx = _chunked_up(self, lx, work, org_forwarded.dtype)
        # Cast to the base output's dtype *immediately*.  ``work`` is fp32 for a
        # fp32-master adapter, so ``lx`` is [B, L, out_features] fp32; the caller
        # casts right after anyway, but by then the allocation is already made --
        # measured as a 272 MiB single request (= 4352x16384x4) against 136 MiB
        # in fp16, which is what OOM'd a 16 GB card at the tail of the MLP.
        lx = lx.to(org_forwarded.dtype)
        if can_scale_in_bottleneck:
            return org_forwarded + lx
        return org_forwarded + (lx * self.multiplier * scale).to(org_forwarded.dtype)

    def compute_delta(
        self,
        x: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        *,
        base_output: torch.Tensor | None = None,
        base_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the adapter-only contribution for a shared base projection.

        Packed projections use this protocol to compute one base GEMM and keep
        each adapter's own dropout, timestep, and variant-specific semantics.
        ``base_output``/``base_weight`` are optional inputs for variants whose
        delta depends on the base projection.
        """
        del x, out_dtype, base_output, base_weight
        raise NotImplementedError(
            f"{type(self).__name__} does not support shared-base delta computation"
        )

    def _gate(self, lx: torch.Tensor, work: torch.dtype) -> torch.Tensor:
        return lx * self._timestep_mask

    def _down(self, x_lora: torch.Tensor, work: torch.dtype) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} uses the forward scaffold but does not "
            "implement _down"
        )

    def _up(self, lx: torch.Tensor, work: torch.dtype) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} uses the forward scaffold but does not "
            "implement _up"
        )

    def _eval_delta(self, x: torch.Tensor, org_forwarded: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} uses the forward scaffold but does not "
            "implement _eval_delta"
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
