"""Krea-2 attention backends shared by training and inference."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

KREA2_ATTENTION_MODES = frozenset({"torch", "flash"})

#: 把 SDPA 的 q/k/v 转 fp16 计算，并展开 GQA 成 MHA 形状。
#:
#: sm70 上 bf16 没有任何可用的 SDPA 内核（FLASH/EFFICIENT/CUDNN 全部
#: "No available kernel"，只剩 MATH），而 MATH 的显存是 O(N²)：4864 tokens
#: 单层就要 10 GiB。fp16 同样不行——memory-efficient 内核在
#: ``q_heads != kv_heads``（GQA）时直接回退 MATH。把 kv 显式展开成与 q 同
#: head 数后，fp16 能走 EFFICIENT，4096 tokens 峰值从 7424 MiB 降到 224 MiB。
#:
#: 反向不能裸转 fp16：深层 attention 的上游梯度 dO 是 1e-8 量级（findings §2.1），
#: 直接 ``.to(fp16)`` 会掉进 subnormal 区。SDPA 反向对 dO 严格齐次线性
#: （``SDPA_bwd(Q,K,V, 2^s·dO) = 2^s·(dQ,dK,dV)``），所以用 frexp/ldexp 把 dO
#: 平移到 fp16 正规区再算、出结果后平移回来，数学上无损。
_ATTN_FP16_ENABLED = False


def set_krea2_attn_fp16(enabled: bool) -> None:
    """启用 Krea-2 attention 的 fp16 内层 + GQA 展开（sm<80 专用）。"""
    global _ATTN_FP16_ENABLED
    if enabled and torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        if major >= 8:
            raise ValueError(
                f"Krea-2 attention fp16 路径只对 sm<80 有收益；当前 sm_{major}0 "
                "有原生 bf16 内核，走默认路径即可"
            )
    _ATTN_FP16_ENABLED = bool(enabled)


def is_krea2_attn_fp16_enabled() -> bool:
    return _ATTN_FP16_ENABLED


class _PreFp16SDPA(torch.autograd.Function):
    """q/k/v 转 fp16；反向把 dq/dk/dv 从缩放空间平移回 bf16。"""

    @staticmethod
    def forward(ctx, q, k, v, state):
        ctx.state = state
        ctx.dtype = q.dtype
        # sm70 的 memory-efficient 内核对非连续输入直接 "No available kernel"
        # 并静默回退 MATH（O(N²) 显存）。RMSNorm/rearrange 之后 v 常常是非连续的，
        # 这里统一转连续，成本只有一次 O(n) 拷贝。
        return (
            q.to(torch.float16).contiguous(),
            k.to(torch.float16).contiguous(),
            v.to(torch.float16).contiguous(),
        )

    @staticmethod
    def backward(ctx, dq16, dk16, dv16):
        shift = ctx.state.get("shift")
        if shift is None:  # 前向没产生梯度（推理/no_grad）：不做缩放
            return dq16.to(ctx.dtype), dk16.to(ctx.dtype), dv16.to(ctx.dtype), None
        neg = -shift
        return (
            torch.ldexp(dq16.float(), neg).to(ctx.dtype),
            torch.ldexp(dk16.float(), neg).to(ctx.dtype),
            torch.ldexp(dv16.float(), neg).to(ctx.dtype),
            None,
        )


class _PostFp16SDPA(torch.autograd.Function):
    """SDPA 的 fp16 输出转回 bf16；反向把 dO 平移到 fp16 正规区。"""

    @staticmethod
    def forward(ctx, out16, state, out_dtype):
        ctx.state = state
        return out16.to(out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # dO 是深层 1e-8 量级的 bf16 梯度：裸转 fp16 会 subnormal/归零。
        # 取 amax 的二进制指数做 2 的整数次幂平移，尾数一位不丢。
        exp = torch.frexp(grad_out.abs().max())[1]
        shift = torch.clamp(-exp, min=-126, max=126)
        ctx.state["shift"] = shift
        return torch.ldexp(grad_out.float(), shift).to(torch.float16), None, None


def normalize_krea2_attention_mode(value: object) -> str:
    mode = str(value or "flash").strip().lower().replace("-", "_")
    if mode in {"sdpa", "cudnn", "cudnn_sdpa"}:
        return "torch"
    if mode not in KREA2_ATTENTION_MODES:
        raise ValueError(
            "Krea-2 attn_mode supports only 'torch' (cuDNN SDPA) and "
            f"'flash' (packed FlashAttention varlen); got {value!r}"
        )
    return mode


def validate_krea2_attention_mode(
    mode: object,
    *,
    dtype: torch.dtype | None,
    compile_enabled: bool = False,
) -> str:
    """Validate a Krea-2 attention mode before loading the large DiT."""

    normalized = normalize_krea2_attention_mode(mode)
    if normalized == "flash":
        from networks import attention_dispatch

        if dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError(
                "Krea-2 attn_mode='flash' requires fp16 or bf16 compute; "
                f"got {dtype}"
            )
        if not attention_dispatch.flash_attn_available_for_dtype(dtype):
            provider = (
                "flash-attention-v100 (FP16 only)"
                if attention_dispatch.flash_attn_v100_provider
                else "FlashAttention 2"
            )
            raise RuntimeError(
                f"Krea-2 attn_mode='flash' requires {provider} with {dtype} support"
            )
        if compile_enabled:
            # Boolean packing produces a symbolic packed-token length. This is
            # enabled only for a run that explicitly selected the Flash backend.
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

    return normalized


def prepare_krea2_attention(
    model: object,
    mode: object,
    *,
    dtype: torch.dtype | None,
    compile_enabled: bool = False,
) -> str:
    """Validate and install a Krea-2 attention mode before optional compile."""

    normalized = validate_krea2_attention_mode(
        mode,
        dtype=dtype,
        compile_enabled=compile_enabled,
    )

    setter = getattr(model, "set_attention_mode", None)
    if not callable(setter):
        raise TypeError("Krea-2 model does not expose set_attention_mode()")
    setter(normalized)
    return normalized


def run_krea2_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Tensor | None = None,
    scale: float | None = None,
    gqa: bool = False,
    mode: str = "torch",
) -> Tensor:
    """Run Krea-2 self-attention with the model's padding-mask contract.

    This is an internal DiT dispatch, not a general masked-attention API.
    ``mask`` must come from ``dit._mask``: a boolean outer product of valid
    tokens with at least one valid token per batch row. Causal or arbitrary
    four-dimensional masks are intentionally unsupported by the varlen path.
    """
    if mode == "flash":
        return _flash_varlen_attention(q, k, v, mask=mask, scale=scale)
    if mode != "torch":
        raise RuntimeError(f"unconfigured Krea-2 attention mode: {mode!r}")
    return _cudnn_attention(q, k, v, mask=mask, scale=scale, gqa=gqa)


def _cudnn_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Tensor | None,
    scale: float | None,
    gqa: bool,
) -> Tensor:
    if _ATTN_FP16_ENABLED and q.dtype == torch.bfloat16:
        return _fp16_attention(q, k, v, mask=mask, scale=scale, gqa=gqa)
    try:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
            )
    except (RuntimeError, NotImplementedError):
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
    return rearrange(output, "B H L D -> B L (H D)")


def _fp16_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Tensor | None,
    scale: float | None,
    gqa: bool,
) -> Tensor:
    """fp16 内层 + GQA 展开 + 反向 frexp/ldexp 缩放。

    sm70 的 memory-efficient 内核对 GQA 直接回退 MATH（O(N²) 显存）。把 kv
    展开成与 q 同 head 数后形状变成 MHA，fp16 就能走 EFFICIENT：4096 tokens
    峰值 7424 MiB -> 224 MiB。展开多花 96 MiB，数学上与 GQA 等价（反向的
    ``repeat_interleave`` 会把梯度 sum 回原 kv head 数）。
    """
    if gqa and q.shape[1] != k.shape[1]:
        group = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        gqa = False
    state: dict = {}
    q16, k16, v16 = _PreFp16SDPA.apply(q, k, v, state)
    with sdpa_kernel(
        [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]
    ):
        out16 = F.scaled_dot_product_attention(
            q16, k16, v16, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
    output = _PostFp16SDPA.apply(out16, state, q.dtype)
    return rearrange(output, "B H L D -> B L (H D)")


@torch.compiler.disable(recursive=True)
def _flash_varlen_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Tensor | None,
    scale: float | None,
) -> Tensor:
    """Run dynamic token packing outside compiled checkpointed blocks.

    Boolean packing and the padded-output scatter have data-dependent shapes.
    AOTAutograd can compile the forward, but checkpoint recomputation lowers
    ``IndexPutBackward`` through ``aten.nonzero`` and fails in Inductor.  The
    FlashAttention kernel still runs normally; only its small packing wrapper
    remains eager while the surrounding DiT block stays compiled.
    """
    from networks import attention_dispatch

    _validate_flash_qkv(q, k, v)
    if not attention_dispatch.flash_attn_available_for_dtype(q.dtype):
        raise RuntimeError(
            f"Krea-2 FlashAttention does not support Q/K/V dtype {q.dtype}"
        )
    flash_varlen: Callable = attention_dispatch.flash_attn_varlen_func

    batch, _heads, sequence, _dim = q.shape
    if mask is None:
        valid = torch.ones(batch, sequence, device=q.device, dtype=torch.bool)
    else:
        expected_mask_shape = (batch, 1, sequence, sequence)
        if tuple(mask.shape) != expected_mask_shape or mask.dtype != torch.bool:
            raise ValueError(
                "Krea-2 FlashAttention expects a boolean (B,1,L,L) "
                f"outer-product mask; got shape={tuple(mask.shape)}, dtype={mask.dtype}"
            )
        valid = mask[:, 0].diagonal(dim1=-2, dim2=-1)

    lengths = valid.sum(dim=1, dtype=torch.int32)
    cu_seqlens = torch.zeros(batch + 1, device=q.device, dtype=torch.int32)
    cu_seqlens[1:] = lengths.cumsum(dim=0)
    q_packed = rearrange(q, "b h l d -> b l h d")[valid].contiguous()
    k_packed = rearrange(k, "b h l d -> b l h d")[valid].contiguous()
    v_packed = rearrange(v, "b h l d -> b l h d")[valid].contiguous()
    output_packed = flash_varlen(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens,
        cu_seqlens,
        sequence,
        sequence,
        softmax_scale=scale,
    )
    output = torch.zeros(
        batch,
        sequence,
        q.shape[1],
        q.shape[-1],
        device=q.device,
        dtype=q.dtype,
    )
    output[valid] = output_packed
    return rearrange(output, "b l h d -> b l (h d)")


def _validate_flash_qkv(q: Tensor, k: Tensor, v: Tensor) -> None:
    tensors = {"q": q, "k": k, "v": v}
    for name, tensor in tensors.items():
        if tensor.ndim != 4:
            raise ValueError(
                f"Krea-2 FlashAttention expects {name} in (B,H,L,D); "
                f"got shape={tuple(tensor.shape)}"
            )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            "Krea-2 FlashAttention requires matching Q/K/V dtypes; "
            f"got q={q.dtype}, k={k.dtype}, v={v.dtype}"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("Krea-2 FlashAttention requires Q/K/V on the same device")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Krea-2 FlashAttention requires matching Q/K/V batch sizes")
    if q.shape[2:] != k.shape[2:] or q.shape[2:] != v.shape[2:]:
        raise ValueError(
            "Krea-2 FlashAttention requires matching Q/K/V sequence and head dimensions"
        )
    if k.shape[1] != v.shape[1] or k.shape[1] == 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "Krea-2 FlashAttention GQA requires equal K/V heads and "
            "Q heads divisible by K/V heads"
        )


__all__ = [
    "KREA2_ATTENTION_MODES",
    "normalize_krea2_attention_mode",
    "prepare_krea2_attention",
    "validate_krea2_attention_mode",
]
