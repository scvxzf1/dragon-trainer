"""Fuse the two casts around NF4 dir1's power-of-two gradient rescaling.

BF16 storage is accessed as uint16 so pre-Ampere does not need native BF16
instructions. The caller owns the scalar, clamped [-126, 126] shift. Other
layouts/dtypes and higher-order autograd keep the original torch expression.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scaled_cast_kernel(X, Shift, Y, N: tl.constexpr,
                        FROM_BF16: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.load(X + offsets, offsets < N, other=0)
    if FROM_BF16:
        value = (bits.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    else:
        value = bits.to(tl.float16, bitcast=True).to(tl.float32)
    shift = tl.load(Shift).to(tl.int32)
    # Exact normal FP32 power of two; no approximate exp2 instruction.
    scale = ((shift + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    scaled = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
        args=[value, scale], dtype=tl.float32, is_pure=True, pack=1,
    )
    if FROM_BF16:
        result = scaled.to(tl.float16).to(tl.uint16, bitcast=True)
    else:
        raw = scaled.to(tl.uint32, bitcast=True)
        # IEEE round-to-nearest-even for FP32 -> BF16; retain signed zero/Inf.
        rounded = (raw + 0x7FFF + ((raw >> 16) & 1)) >> 16
        is_nan = (raw & 0x7FFFFFFF) > 0x7F800000
        result = tl.where(is_nan, 0x7FC0, rounded).to(tl.uint16)
    tl.store(Y + offsets, result, offsets < N)


def scaled_cast(value: torch.Tensor, shift: torch.Tensor, dtype: torch.dtype):
    """Use the fast path only for dir1's ordinary first-order CUDA backward."""
    eligible = (
        value.device.type == "cuda"
        and value.is_contiguous()
        and not value.is_neg()
        and value.numel() > 0
        and (value.dtype, dtype) in (
            (torch.bfloat16, torch.float16),
            (torch.float16, torch.bfloat16),
        )
        and shift.device == value.device
        and shift.numel() == 1
        and not shift.is_neg()
        and shift.dtype == torch.int32
        and not torch.is_grad_enabled()
    )
    if not eligible:
        return torch.ldexp(value.float(), shift).to(dtype)
    result = torch.empty_like(value, dtype=dtype)
    with torch.cuda.device(value.device):
        _scaled_cast_kernel[(triton.cdiv(value.numel(), 1024),)](
            value.view(torch.uint16), shift, result.view(torch.uint16),
            value.numel(), value.dtype == torch.bfloat16, 1024,
        )
    return result
