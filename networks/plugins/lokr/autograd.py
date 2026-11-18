"""Memory-saving projection helpers for the LoKR plugin."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from .triton_backward import (
    lokr_grad_w1_triton_available,
    reduce_lokr_grad_w1_triton,
)

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - import availability is environment-specific
    triton = None
    tl = None
    _LOKR_TRITON_IMPORT_ERROR = exc
else:
    _LOKR_TRITON_IMPORT_ERROR = None


DEFAULT_LOKR_PROJECT_CHUNK_BYTES = 4 * 1024 * 1024
DEFAULT_LOKR_GROUPED_DELTA_BACKEND = "triton"
DEFAULT_LOKR_GROUPED_DELTA_BACKWARD_BACKEND = "triton_grad_w1_w2_grad_x"
_LOKR_PROJECT_CHUNK_BYTES = DEFAULT_LOKR_PROJECT_CHUNK_BYTES
_LOKR_GROUPED_DELTA_BACKENDS = frozenset({"eager", "triton"})
_LOKR_GROUPED_DELTA_BACKWARD_BACKENDS = frozenset(
    {
        "eager",
        "triton_grad_x",
        "triton_grad_w2_partial",
        "triton_grad_w2_grad_x",
        "triton_grad_w1_w2_grad_x",
    }
)
# Pre-Ampere GPUs (SM70/SM75) lack TF32 tensor cores, so vectorized chunked
# cuBLAS GEMM is ~6x faster and ~100 MiB leaner than SIMT FP32 Triton tl.dot.
_MIN_TRITON_CC = (8, 0)
_LOKR_ENABLE_BACKWARD_PHASE_RANGES = False
_LOKR_BACKWARD_PHASE_NAMES = (
    "recompute_projected",
    "grad_w1_reduce",
    "grad_w2_reduce",
    "grad_x_writeback",
)


def _normalize_lokr_project_chunk_bytes(value) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(_LOKR_PROJECT_CHUNK_BYTES)
    return max(1, parsed)


def normalize_lokr_grouped_delta_backend(value) -> str:
    if value is None:
        return DEFAULT_LOKR_GROUPED_DELTA_BACKEND
    backend = str(value).strip().lower()
    if not backend:
        return DEFAULT_LOKR_GROUPED_DELTA_BACKEND
    if backend not in _LOKR_GROUPED_DELTA_BACKENDS:
        choices = ", ".join(sorted(_LOKR_GROUPED_DELTA_BACKENDS))
        raise ValueError(
            f"unsupported LoKr grouped-delta backend {value!r}; expected one of: {choices}"
        )
    return backend


def normalize_lokr_grouped_delta_backward_backend(value) -> str:
    if value is None:
        return DEFAULT_LOKR_GROUPED_DELTA_BACKWARD_BACKEND
    backend = str(value).strip().lower()
    if not backend:
        return DEFAULT_LOKR_GROUPED_DELTA_BACKWARD_BACKEND
    if backend not in _LOKR_GROUPED_DELTA_BACKWARD_BACKENDS:
        choices = ", ".join(sorted(_LOKR_GROUPED_DELTA_BACKWARD_BACKENDS))
        raise ValueError(
            "unsupported LoKr grouped-delta backward backend "
            f"{value!r}; expected one of: {choices}"
        )
    return backend


def set_lokr_backward_phase_ranges(enabled: bool) -> bool:
    global _LOKR_ENABLE_BACKWARD_PHASE_RANGES
    previous = _LOKR_ENABLE_BACKWARD_PHASE_RANGES
    _LOKR_ENABLE_BACKWARD_PHASE_RANGES = bool(enabled)
    return previous


def _lokr_record_function(name: str):
    if not _LOKR_ENABLE_BACKWARD_PHASE_RANGES:
        return nullcontext()
    from torch.autograd.profiler import record_function

    return record_function(name)


def _projection_chunk_rows(in_dim: int, out_dim: int, chunk_bytes=None) -> int:
    chunk_limit = _normalize_lokr_project_chunk_bytes(
        _LOKR_PROJECT_CHUNK_BYTES if chunk_bytes is None else chunk_bytes
    )
    bytes_per_row = max(1, int(in_dim) + int(out_dim)) * 4
    return max(1, chunk_limit // bytes_per_row)


def _device_supports_lokr_triton(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return torch.cuda.get_device_capability(device) >= _MIN_TRITON_CC
    except Exception:
        return False


def _can_use_lokr_grouped_delta_triton(
    base,
    x,
    w1,
    w2,
    gate_scale,
    factor,
    in_dim,
    out_dim,
) -> bool:
    if triton is None or tl is None:
        return False
    if not all(torch.is_tensor(t) for t in (base, x, w1, w2, gate_scale)):
        return False
    if not all(t.device == base.device for t in (x, w1, w2, gate_scale)):
        return False
    if not _device_supports_lokr_triton(base.device):
        return False
    if gate_scale.numel() != 1:
        return False
    if factor <= 0 or in_dim <= 0 or out_dim <= 0:
        return False
    if x.shape[-1] != factor * in_dim or base.shape[-1] != factor * out_dim:
        return False
    if w1.shape != (factor, factor) or w2.shape != (out_dim, in_dim):
        return False
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if base.dtype not in supported_dtypes or x.dtype not in supported_dtypes:
        return False
    if w1.dtype not in supported_dtypes or w2.dtype not in supported_dtypes:
        return False
    return (
        base.is_contiguous()
        and x.is_contiguous()
        and w1.is_contiguous()
        and w2.is_contiguous()
    )


if triton is not None:  # pragma: no branch - definition only when Triton imports

    @triton.jit
    def _lokr_grouped_delta_forward_kernel(
        base_ptr,
        x_ptr,
        w1_ptr,
        w2_ptr,
        gate_ptr,
        rows,
        in_dim,
        out_dim,
        base_row_stride,
        x_row_stride,
        w1_row_stride,
        w2_row_stride,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        FACTOR: tl.constexpr,
        USE_IEEE: tl.constexpr = False,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_out = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < rows
        mask_n = offs_n < out_dim

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for in_factor in range(FACTOR):
            coeff = tl.load(w1_ptr + pid_out * w1_row_stride + in_factor).to(tl.float32)
            for k_iter in range(0, tl.cdiv(in_dim, BLOCK_K)):
                offs_k = k_iter * BLOCK_K + tl.arange(0, BLOCK_K)
                mask_k = offs_k < in_dim

                x_ptrs = (
                    x_ptr
                    + offs_m[:, None] * x_row_stride
                    + in_factor * in_dim
                    + offs_k[None, :]
                )
                w2_ptrs = w2_ptr + offs_n[:, None] * w2_row_stride + offs_k[None, :]

                x_block = tl.load(
                    x_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                w2_block = tl.load(
                    w2_ptrs,
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                if USE_IEEE:
                    acc += (
                        tl.dot(x_block, tl.trans(w2_block), input_precision="ieee")
                        * coeff
                    )
                else:
                    acc += tl.dot(x_block, tl.trans(w2_block)) * coeff

        gate = tl.load(gate_ptr).to(tl.float32)
        base_ptrs = (
            base_ptr
            + offs_m[:, None] * base_row_stride
            + pid_out * out_dim
            + offs_n[None, :]
        )
        base_block = tl.load(
            base_ptrs,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        tl.store(base_ptrs, base_block + acc * gate, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _lokr_grouped_delta_grad_x_mix_kernel(
        grad_ptr,
        w1_ptr,
        gate_ptr,
        mixed_ptr,
        rows,
        out_dim,
        grad_row_stride,
        w1_row_stride,
        mixed_row_stride,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FACTOR: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_in = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_m = offs_m < rows
        mask_d = offs_d < out_dim

        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        gate = tl.load(gate_ptr).to(tl.float32)
        for out_factor in range(FACTOR):
            coeff = tl.load(w1_ptr + out_factor * w1_row_stride + pid_in).to(tl.float32)
            grad_ptrs = (
                grad_ptr
                + offs_m[:, None] * grad_row_stride
                + out_factor * out_dim
                + offs_d[None, :]
            )
            grad_block = tl.load(
                grad_ptrs,
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += grad_block * coeff * gate

        mixed_ptrs = (
            mixed_ptr
            + offs_m[:, None] * mixed_row_stride
            + pid_in * out_dim
            + offs_d[None, :]
        )
        tl.store(
            mixed_ptrs,
            acc,
            mask=mask_m[:, None] & mask_d[None, :],
        )

    @triton.jit
    def _lokr_grouped_delta_grad_w2_mix_kernel(
        grad_ptr,
        w1_ptr,
        mixed_ptr,
        rows,
        out_dim,
        grad_row_stride,
        w1_row_stride,
        mixed_row_stride,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FACTOR: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_in = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_m = offs_m < rows
        mask_d = offs_d < out_dim

        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for out_factor in range(GROUP):
            coeff = tl.load(w1_ptr + out_factor * w1_row_stride + pid_in).to(tl.float32)
            grad_ptrs = (
                grad_ptr
                + offs_m[:, None] * grad_row_stride
                + out_factor * out_dim
                + offs_d[None, :]
            )
            grad_block = tl.load(
                grad_ptrs,
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += grad_block * coeff

        mixed_ptrs = (
            mixed_ptr
            + offs_m[:, None] * mixed_row_stride
            + pid_in * out_dim
            + offs_d[None, :]
        )
        tl.store(
            mixed_ptrs,
            acc,
            mask=mask_m[:, None] & mask_d[None, :],
        )


def _save_lokr_add_grouped_delta_ctx(
    ctx,
    *,
    x,
    w1,
    w2,
    gate_scale,
    factor,
    in_dim,
    out_dim,
    group_size,
    chunk_bytes,
):
    ctx.save_for_backward(x, w1, w2, gate_scale)
    ctx.factor = int(factor)
    ctx.in_dim = int(in_dim)
    ctx.out_dim = int(out_dim)
    ctx.group_size = max(1, min(int(group_size), int(factor)))
    ctx.chunk_bytes = _normalize_lokr_project_chunk_bytes(chunk_bytes)
    ctx.leading_shape = x.shape[:-1]


def _can_use_lokr_grouped_delta_backward_triton_common(
    grad_out,
    x,
    w1,
    w2,
    gate_scale,
    factor,
    in_dim,
    out_dim,
) -> bool:
    if triton is None or tl is None:
        return False
    if not all(torch.is_tensor(t) for t in (grad_out, x, w1, w2, gate_scale)):
        return False
    if not all(t.device == grad_out.device for t in (x, w1, w2, gate_scale)):
        return False
    if not _device_supports_lokr_triton(grad_out.device):
        return False
    if gate_scale.numel() != 1:
        return False
    if factor <= 0 or in_dim <= 0 or out_dim <= 0:
        return False
    if x.shape[-1] != factor * in_dim or grad_out.shape[-1] != factor * out_dim:
        return False
    if w1.shape != (factor, factor) or w2.shape != (out_dim, in_dim):
        return False
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if grad_out.dtype not in supported_dtypes or x.dtype not in supported_dtypes:
        return False
    if w1.dtype not in supported_dtypes or w2.dtype not in supported_dtypes:
        return False
    return (
        grad_out.is_contiguous()
        and x.is_contiguous()
        and w1.is_contiguous()
        and w2.is_contiguous()
    )


def _can_use_lokr_grouped_delta_backward_triton_grad_x(
    grad_out,
    x,
    w1,
    w2,
    gate_scale,
    factor,
    in_dim,
    out_dim,
) -> bool:
    return _can_use_lokr_grouped_delta_backward_triton_common(
        grad_out,
        x,
        w1,
        w2,
        gate_scale,
        factor,
        in_dim,
        out_dim,
    )


def _can_use_lokr_grouped_delta_backward_triton_grad_w2_partial(
    grad_out,
    x,
    w1,
    w2,
    gate_scale,
    factor,
    in_dim,
    out_dim,
) -> bool:
    return _can_use_lokr_grouped_delta_backward_triton_common(
        grad_out,
        x,
        w1,
        w2,
        gate_scale,
        factor,
        in_dim,
        out_dim,
    )


def _can_use_lokr_grouped_delta_backward_triton_grad_w2_grad_x(
    grad_out,
    x,
    w1,
    w2,
    gate_scale,
    factor,
    in_dim,
    out_dim,
) -> bool:
    return _can_use_lokr_grouped_delta_backward_triton_common(
        grad_out,
        x,
        w1,
        w2,
        gate_scale,
        factor,
        in_dim,
        out_dim,
    )


def _can_use_lokr_grouped_delta_backward_triton_grad_w1_w2_grad_x(
    grad_out,
    x,
    w1,
    w2,
    gate_scale,
    factor,
    in_dim,
    out_dim,
) -> bool:
    return lokr_grad_w1_triton_available() and (
        _can_use_lokr_grouped_delta_backward_triton_common(
            grad_out,
            x,
            w1,
            w2,
            gate_scale,
            factor,
            in_dim,
            out_dim,
        )
    )


def _lokr_add_grouped_delta_forward_eager(
    base,
    x,
    w1,
    w2,
    gate_scale,
    factor,
    in_dim,
    out_dim,
    group_size,
    chunk_bytes,
):
    x_view = x.reshape(-1, factor, in_dim)
    result_view = base.reshape(-1, factor, out_dim)
    gate = gate_scale.float().reshape(-1)[0]
    w1_t = (w1.float() * gate).transpose(0, 1).contiguous()
    w2_t = w2.float().transpose(0, 1)
    # Each row holds ``factor`` slices of (in_dim + out_dim) fp32 intermediates.
    chunk_rows = max(
        1, _projection_chunk_rows(factor * in_dim, factor * out_dim, chunk_bytes)
    )

    for row_start in range(0, x_view.shape[0], chunk_rows):
        row_end = min(row_start + chunk_rows, x_view.shape[0])
        rn = row_end - row_start
        # Compute x @ w2.T once per row chunk via batched GEMM instead of
        # repeating it ceil(factor / group_size) * factor times in Python.
        projected = (
            x_view[row_start:row_end]
            .float()
            .reshape(rn * factor, in_dim)
            .matmul(w2_t)
            .reshape(rn, factor, out_dim)
        )
        mixed = projected.permute(0, 2, 1).matmul(w1_t).permute(0, 2, 1)
        result_view[row_start:row_end].add_(mixed.to(result_view.dtype))
    return base


def _launch_lokr_grouped_delta_triton(
    base,
    x,
    w1,
    w2,
    gate_scale,
    factor,
    in_dim,
    out_dim,
):
    base_view = base.reshape(-1, factor * out_dim)
    x_view = x.reshape(-1, factor * in_dim)
    rows = base_view.shape[0]
    block_m = 32 if rows >= 32 else 16
    block_n = 32 if out_dim >= 32 else 16
    block_k = 32 if in_dim >= 32 else 16
    num_warps = 8 if block_m == 32 and block_n == 32 else 4
    use_ieee = (
        base.device.type == "cuda"
        and torch.cuda.get_device_capability(base.device) < (8, 0)
    )
    grid = (
        triton.cdiv(rows, block_m),
        triton.cdiv(out_dim, block_n),
        factor,
    )
    _lokr_grouped_delta_forward_kernel[grid](
        base_view,
        x_view,
        w1,
        w2,
        gate_scale.reshape(-1),
        rows,
        in_dim,
        out_dim,
        base_view.stride(0),
        x_view.stride(0),
        w1.stride(0),
        w2.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        FACTOR=factor,
        USE_IEEE=use_ieee,
        num_warps=num_warps,
        num_stages=2,
    )
    return base


def _launch_lokr_grouped_delta_grad_x_triton(
    grad_out,
    x,
    w1,
    w2,
    gate_scale,
    factor,
    in_dim,
    out_dim,
    chunk_bytes,
):
    grad_view = grad_out.reshape(-1, factor * out_dim)
    rows = grad_view.shape[0]
    grad_x = torch.empty(
        (rows, factor * in_dim),
        device=grad_out.device,
        dtype=x.dtype,
    )
    block_d = 64 if out_dim >= 64 else 32
    chunk_rows = _projection_chunk_rows(in_dim, out_dim, chunk_bytes)
    w2_float = w2.float()
    for row_start in range(0, rows, chunk_rows):
        row_end = min(row_start + chunk_rows, rows)
        rows_this = row_end - row_start
        grad_chunk = grad_view[row_start:row_end]
        mixed = torch.empty(
            (rows_this, factor * out_dim),
            device=grad_out.device,
            dtype=torch.float32,
        )
        block_m = 32 if rows_this >= 32 else 16
        num_warps = 8 if block_m == 32 and block_d == 64 else 4
        grid = (
            triton.cdiv(rows_this, block_m),
            triton.cdiv(out_dim, block_d),
            factor,
        )
        _lokr_grouped_delta_grad_x_mix_kernel[grid](
            grad_chunk,
            w1,
            gate_scale.reshape(-1),
            mixed,
            rows_this,
            out_dim=out_dim,
            grad_row_stride=grad_chunk.stride(0),
            w1_row_stride=w1.stride(0),
            mixed_row_stride=mixed.stride(0),
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            FACTOR=factor,
            num_warps=num_warps,
            num_stages=2,
        )
        grad_x_chunk = mixed.reshape(rows_this * factor, out_dim).matmul(w2_float)
        grad_x[row_start:row_end].copy_(
            grad_x_chunk.reshape(rows_this, factor * in_dim).to(x.dtype)
        )
    return grad_x


def _launch_lokr_grouped_delta_grad_w2_mix_triton(
    grad,
    w1_chunk,
    factor,
    out_dim,
    mixed_buffer=None,
):
    rows_this = grad.shape[0]
    out_count = grad.shape[1]
    block_d = 64 if out_dim >= 64 else 32
    block_m = 32 if rows_this >= 32 else 16
    num_warps = 8 if block_m == 32 and block_d == 64 else 4
    grad_chunk = grad.reshape(rows_this, out_count * out_dim).contiguous()
    if (
        mixed_buffer is None
        or mixed_buffer.shape[0] < rows_this
        or mixed_buffer.shape[1] < factor * out_dim
    ):
        mixed = torch.empty(
            (rows_this, factor * out_dim),
            device=grad.device,
            dtype=torch.float32,
        )
    else:
        mixed = mixed_buffer[:rows_this, : factor * out_dim]
    grid = (
        triton.cdiv(rows_this, block_m),
        triton.cdiv(out_dim, block_d),
        factor,
    )
    _lokr_grouped_delta_grad_w2_mix_kernel[grid](
        grad_chunk,
        w1_chunk,
        mixed,
        rows_this,
        out_dim=out_dim,
        grad_row_stride=grad_chunk.stride(0),
        w1_row_stride=w1_chunk.stride(0),
        mixed_row_stride=mixed.stride(0),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        FACTOR=factor,
        GROUP=out_count,
        num_warps=num_warps,
        num_stages=2,
    )
    return mixed


def _reduce_lokr_grad_w2_from_mixed(
    mixed,
    x_chunk,
    factor,
    in_dim,
    out_dim,
):
    rows_this = x_chunk.shape[0]
    x_flat = x_chunk.reshape(rows_this * factor, in_dim).float()
    mixed_flat = mixed.reshape(rows_this * factor, out_dim)
    return mixed_flat.transpose(0, 1).matmul(x_flat)


def _reduce_lokr_grad_x_from_mixed(
    mixed,
    w2_float,
    factor,
    in_dim,
    out_dim,
    out_dtype,
):
    rows_this = mixed.shape[0]
    grad_x_chunk = mixed.reshape(rows_this * factor, out_dim).matmul(w2_float)
    return grad_x_chunk.reshape(rows_this, factor, in_dim).to(out_dtype)


def _lokr_add_grouped_delta_backward(
    grad_out,
    x,
    w1,
    w2,
    gate_scale,
    factor,
    in_dim,
    out_dim,
    group_size,
    chunk_bytes,
    leading_shape,
    backward_backend=DEFAULT_LOKR_GROUPED_DELTA_BACKWARD_BACKEND,
):
    x_view = x.reshape(-1, factor, in_dim)
    w1_float = w1.float()
    w2_float = w2.float()
    grad_view = grad_out.reshape(-1, factor, out_dim)
    gate = gate_scale.float().reshape(-1)[0]
    normalized_backward_backend = normalize_lokr_grouped_delta_backward_backend(
        backward_backend
    )

    grad_w1 = w1_float.new_zeros(w1_float.shape)
    grad_w2 = w2_float.new_zeros(w2_float.shape)
    use_triton_grad_x = (
        normalized_backward_backend == "triton_grad_x"
        and _can_use_lokr_grouped_delta_backward_triton_grad_x(
            grad_out,
            x,
            w1,
            w2,
            gate_scale,
            factor,
            in_dim,
            out_dim,
        )
    )
    use_triton_grad_w2_partial = (
        normalized_backward_backend == "triton_grad_w2_partial"
        and _can_use_lokr_grouped_delta_backward_triton_grad_w2_partial(
            grad_out,
            x,
            w1,
            w2,
            gate_scale,
            factor,
            in_dim,
            out_dim,
        )
    )
    use_triton_grad_w2_grad_x = (
        normalized_backward_backend == "triton_grad_w2_grad_x"
        and _can_use_lokr_grouped_delta_backward_triton_grad_w2_grad_x(
            grad_out,
            x,
            w1,
            w2,
            gate_scale,
            factor,
            in_dim,
            out_dim,
        )
    )
    use_triton_grad_w1_w2_grad_x = (
        normalized_backward_backend == "triton_grad_w1_w2_grad_x"
        and _can_use_lokr_grouped_delta_backward_triton_grad_w1_w2_grad_x(
            grad_out,
            x,
            w1,
            w2,
            gate_scale,
            factor,
            in_dim,
            out_dim,
        )
    )
    use_triton_grad_w2_mixed = (
        use_triton_grad_w2_partial
        or use_triton_grad_w2_grad_x
        or use_triton_grad_w1_w2_grad_x
    )
    grad_x = None
    if not use_triton_grad_x:
        grad_x = torch.zeros(
            (x_view.shape[0], factor, in_dim), dtype=x.dtype, device=x.device
        )
    w2_t = w2_float.transpose(0, 1)
    chunk_rows = _projection_chunk_rows(in_dim, out_dim, chunk_bytes)
    mixed_buffer = None
    grad_w1_partial_buffer = None
    if use_triton_grad_w2_mixed:
        mixed_buffer = torch.empty(
            (chunk_rows, factor * out_dim),
            device=grad_out.device,
            dtype=torch.float32,
        )

    if not use_triton_grad_w2_mixed:
        # Vectorized chunked GEMM backward for CPU / pre-Ampere CUDA (e.g. V100)
        # and explicit eager mode: computes x @ w2.T once per row chunk instead
        # of repeating it across out_groups * factor Python iterations.
        vec_chunk_rows = max(
            1, _projection_chunk_rows(factor * in_dim, factor * out_dim, chunk_bytes)
        )
        for row_start in range(0, x_view.shape[0], vec_chunk_rows):
            row_end = min(row_start + vec_chunk_rows, x_view.shape[0])
            rn = row_end - row_start
            x_chunk_f = x_view[row_start:row_end].float()
            grad_chunk_f = grad_view[row_start:row_end].float() * gate

            with _lokr_record_function("recompute_projected"):
                projected = (
                    x_chunk_f.reshape(rn * factor, in_dim)
                    .matmul(w2_t)
                    .reshape(rn, factor, out_dim)
                )

            with _lokr_record_function("grad_w1_reduce"):
                grad_w1.add_(
                    grad_chunk_f.permute(1, 0, 2)
                    .reshape(factor, rn * out_dim)
                    .matmul(
                        projected.permute(1, 0, 2)
                        .reshape(factor, rn * out_dim)
                        .transpose(0, 1)
                    )
                )

            with _lokr_record_function("grad_w2_reduce"):
                grad_projected = (
                    grad_chunk_f.permute(0, 2, 1)
                    .matmul(w1_float)
                    .permute(0, 2, 1)
                    .reshape(rn * factor, out_dim)
                )
                grad_w2.add_(
                    grad_projected.transpose(0, 1).matmul(
                        x_chunk_f.reshape(rn * factor, in_dim)
                    )
                )

            if grad_x is not None:
                with _lokr_record_function("grad_x_writeback"):
                    grad_x[row_start:row_end].copy_(
                        grad_projected.matmul(w2_float)
                        .reshape(rn, factor, in_dim)
                        .to(x.dtype)
                    )
    else:
        for out_start in range(0, factor, group_size):
            out_count = min(group_size, factor - out_start)
            out_slice = slice(out_start, out_start + out_count)
            for row_start in range(0, x_view.shape[0], chunk_rows):
                row_end = min(row_start + chunk_rows, x_view.shape[0])
                grad = (
                    grad_view[row_start:row_end, out_slice, :].float() * gate
                ).contiguous()
                x_chunk = x_view[row_start:row_end]
                mixed = None
                with _lokr_record_function("grad_w2_reduce"):
                    mixed = _launch_lokr_grouped_delta_grad_w2_mix_triton(
                        grad,
                        w1_float[out_slice, :].contiguous(),
                        factor,
                        out_dim,
                        mixed_buffer=mixed_buffer,
                    )
                    grad_w2.add_(
                        _reduce_lokr_grad_w2_from_mixed(
                            mixed,
                            x_chunk,
                            factor,
                            in_dim,
                            out_dim,
                        )
                    )
                if use_triton_grad_w2_grad_x or use_triton_grad_w1_w2_grad_x:
                    with _lokr_record_function("grad_x_writeback"):
                        grad_x[row_start:row_end].add_(
                            _reduce_lokr_grad_x_from_mixed(
                                mixed,
                                w2_float,
                                factor,
                                in_dim,
                                out_dim,
                                x.dtype,
                            )
                        )
                if use_triton_grad_w1_w2_grad_x:
                    with _lokr_record_function("grad_w1_reduce"):
                        grad_w1_contribution, grad_w1_partial_buffer = (
                            reduce_lokr_grad_w1_triton(
                                grad,
                                x_chunk,
                                w2_float,
                                factor,
                                in_dim,
                                out_dim,
                                partial_buffer=grad_w1_partial_buffer,
                            )
                        )
                        grad_w1[out_slice, :].add_(grad_w1_contribution)
                    continue
                for in_factor in range(factor):
                    x_slice = x_chunk[:, in_factor, :].float()
                    with _lokr_record_function("recompute_projected"):
                        projected = x_slice.matmul(w2_t)

                    with _lokr_record_function("grad_w1_reduce"):
                        grad_w1[out_slice, in_factor].add_(
                            torch.einsum("ngo,no->g", grad, projected)
                        )

                    if (
                        grad_x is not None
                        and not use_triton_grad_w2_grad_x
                        and not use_triton_grad_w1_w2_grad_x
                    ):
                        coeffs = w1_float[out_slice, in_factor]
                        with _lokr_record_function("grad_x_writeback"):
                            grad_projected = torch.einsum("g,ngo->no", coeffs, grad)
                            grad_x[row_start:row_end, in_factor, :].add_(
                                grad_projected.matmul(w2_float).to(x.dtype)
                            )
    if grad_x is None:
        with _lokr_record_function("grad_x_writeback"):
            grad_x = _launch_lokr_grouped_delta_grad_x_triton(
                grad_out,
                x,
                w1,
                w2,
                gate_scale,
                factor,
                in_dim,
                out_dim,
                chunk_bytes,
            ).reshape(*leading_shape, factor * in_dim)
    else:
        grad_x = grad_x.reshape(*leading_shape, factor * in_dim)

    return grad_x, grad_w1.to(w1.dtype), grad_w2.to(w2.dtype)


class LoKrProjectFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, factor, in_dim, out_dim):
        factor = int(factor)
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        leading_shape = x.shape[:-1]
        x_view = x.reshape(-1, factor, in_dim)
        rows = x_view.shape[0]
        w1_t = w1.float().transpose(0, 1).contiguous()
        w2_t = w2.float().transpose(0, 1)
        chunk_rows = max(
            1, _projection_chunk_rows(factor * in_dim, factor * out_dim)
        )
        mixed = torch.empty(
            (rows, factor, out_dim), device=x.device, dtype=torch.float32
        )
        for row_start in range(0, rows, chunk_rows):
            row_end = min(row_start + chunk_rows, rows)
            rn = row_end - row_start
            projected = (
                x_view[row_start:row_end]
                .float()
                .reshape(rn * factor, in_dim)
                .matmul(w2_t)
                .reshape(rn, factor, out_dim)
            )
            mixed[row_start:row_end] = (
                projected.permute(0, 2, 1).matmul(w1_t).permute(0, 2, 1)
            )
        ctx.save_for_backward(x, w1, w2)
        ctx.factor = factor
        ctx.in_dim = in_dim
        ctx.out_dim = out_dim
        ctx.leading_shape = leading_shape
        return mixed.reshape(*leading_shape, factor * out_dim)

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2 = ctx.saved_tensors
        factor = ctx.factor
        in_dim = ctx.in_dim
        out_dim = ctx.out_dim

        x_view = x.reshape(-1, factor, in_dim)
        rows = x_view.shape[0]
        w1_float = w1.float()
        w2_float = w2.float()
        w2_t = w2_float.transpose(0, 1)
        grad_view = grad_out.reshape(-1, factor, out_dim)

        chunk_rows = max(
            1, _projection_chunk_rows(factor * in_dim, factor * out_dim)
        )
        grad_w1 = w1_float.new_zeros(w1_float.shape)
        grad_w2 = w2_float.new_zeros(w2_float.shape)
        grad_x = torch.empty(
            (rows, factor, in_dim), device=x.device, dtype=x.dtype
        )
        for row_start in range(0, rows, chunk_rows):
            row_end = min(row_start + chunk_rows, rows)
            rn = row_end - row_start
            x_chunk_f = x_view[row_start:row_end].float()
            grad_chunk_f = grad_view[row_start:row_end].float()
            projected = (
                x_chunk_f.reshape(rn * factor, in_dim)
                .matmul(w2_t)
                .reshape(rn, factor, out_dim)
            )
            grad_w1.add_(
                grad_chunk_f.permute(1, 0, 2)
                .reshape(factor, rn * out_dim)
                .matmul(
                    projected.permute(1, 0, 2)
                    .reshape(factor, rn * out_dim)
                    .transpose(0, 1)
                )
            )
            grad_projected = (
                grad_chunk_f.permute(0, 2, 1)
                .matmul(w1_float)
                .permute(0, 2, 1)
                .reshape(rn * factor, out_dim)
            )
            grad_w2.add_(
                grad_projected.transpose(0, 1).matmul(
                    x_chunk_f.reshape(rn * factor, in_dim)
                )
            )
            grad_x[row_start:row_end] = (
                grad_projected.matmul(w2_float)
                .reshape(rn, factor, in_dim)
                .to(x.dtype)
            )
        grad_x = grad_x.reshape(*ctx.leading_shape, factor * in_dim)

        return (
            grad_x,
            grad_w1.to(w1.dtype),
            grad_w2.to(w2.dtype),
            None,
            None,
            None,
        )


class LoKrProjectFactorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, out_factor, factor, in_dim, out_dim, chunk_bytes):
        out_factor = int(out_factor)
        factor = int(factor)
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        chunk_bytes = _normalize_lokr_project_chunk_bytes(chunk_bytes)
        leading_shape = x.shape[:-1]
        x_view = x.reshape(-1, factor, in_dim)
        w1_float = w1.float()
        w2_float = w2.float()
        w2_t = w2_float.transpose(0, 1)
        out = w2_float.new_empty((x_view.shape[0], out_dim))
        out.zero_()
        chunk_rows = _projection_chunk_rows(in_dim, out_dim, chunk_bytes)
        for row_start in range(0, x_view.shape[0], chunk_rows):
            row_end = min(row_start + chunk_rows, x_view.shape[0])
            out_chunk = out[row_start:row_end]
            for in_factor in range(factor):
                x_slice = x_view[row_start:row_end, in_factor, :].float()
                projected = x_slice.matmul(w2_t)
                out_chunk.add_(projected * w1_float[out_factor, in_factor])
        ctx.save_for_backward(x, w1, w2)
        ctx.out_factor = out_factor
        ctx.factor = factor
        ctx.in_dim = in_dim
        ctx.out_dim = out_dim
        ctx.chunk_bytes = chunk_bytes
        ctx.leading_shape = leading_shape
        return out.reshape(*leading_shape, out_dim)

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2 = ctx.saved_tensors
        out_factor = ctx.out_factor
        factor = ctx.factor
        in_dim = ctx.in_dim
        out_dim = ctx.out_dim
        chunk_bytes = ctx.chunk_bytes

        x_view = x.reshape(-1, factor, in_dim)
        w1_float = w1.float()
        w2_float = w2.float()
        grad_view = grad_out.reshape(-1, out_dim)

        grad_w1 = w1_float.new_zeros(w1_float.shape)
        grad_w2 = w2_float.new_zeros(w2_float.shape)
        grad_x = torch.empty(
            (x_view.shape[0], factor, in_dim), dtype=x.dtype, device=x.device
        )
        w2_t = w2_float.transpose(0, 1)
        chunk_rows = _projection_chunk_rows(in_dim, out_dim, chunk_bytes)
        for row_start in range(0, x_view.shape[0], chunk_rows):
            row_end = min(row_start + chunk_rows, x_view.shape[0])
            grad = grad_view[row_start:row_end].float()
            for in_factor in range(factor):
                x_slice = x_view[row_start:row_end, in_factor, :].float()
                projected = x_slice.matmul(w2_t)
                grad_w1[out_factor, in_factor].add_((grad * projected).sum())

                coeff = w1_float[out_factor, in_factor]
                grad_projected = grad * coeff
                grad_w2.add_(grad_projected.transpose(0, 1).matmul(x_slice))
                grad_x[row_start:row_end, in_factor, :] = grad_projected.matmul(
                    w2_float
                ).to(x.dtype)
        grad_x = grad_x.reshape(*ctx.leading_shape, factor * in_dim)

        return (
            grad_x,
            grad_w1.to(w1.dtype),
            grad_w2.to(w2.dtype),
            None,
            None,
            None,
            None,
            None,
        )


class LoKrProjectFactorGroupFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, out_start, out_count, factor, in_dim, out_dim, chunk_bytes):
        out_start = int(out_start)
        out_count = int(out_count)
        factor = int(factor)
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        chunk_bytes = _normalize_lokr_project_chunk_bytes(chunk_bytes)
        out_count = max(1, min(out_count, factor - out_start))
        leading_shape = x.shape[:-1]
        x_view = x.reshape(-1, factor, in_dim)
        w1_float = w1.float()
        w2_float = w2.float()
        w2_t = w2_float.transpose(0, 1)
        out = w2_float.new_empty((x_view.shape[0], out_count, out_dim))
        out.zero_()
        chunk_rows = _projection_chunk_rows(in_dim, out_dim, chunk_bytes)
        for row_start in range(0, x_view.shape[0], chunk_rows):
            row_end = min(row_start + chunk_rows, x_view.shape[0])
            out_chunk = out[row_start:row_end]
            for in_factor in range(factor):
                x_slice = x_view[row_start:row_end, in_factor, :].float()
                projected = x_slice.matmul(w2_t)
                coeffs = w1_float[out_start : out_start + out_count, in_factor]
                out_chunk.add_(projected.unsqueeze(1) * coeffs.view(1, out_count, 1))
        ctx.save_for_backward(x, w1, w2)
        ctx.out_start = out_start
        ctx.out_count = out_count
        ctx.factor = factor
        ctx.in_dim = in_dim
        ctx.out_dim = out_dim
        ctx.chunk_bytes = chunk_bytes
        ctx.leading_shape = leading_shape
        return out.reshape(*leading_shape, out_count * out_dim)

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2 = ctx.saved_tensors
        out_start = ctx.out_start
        out_count = ctx.out_count
        factor = ctx.factor
        in_dim = ctx.in_dim
        out_dim = ctx.out_dim
        chunk_bytes = ctx.chunk_bytes

        x_view = x.reshape(-1, factor, in_dim)
        w1_float = w1.float()
        w2_float = w2.float()
        grad_view = grad_out.reshape(-1, out_count, out_dim)

        grad_w1 = w1_float.new_zeros(w1_float.shape)
        grad_w2 = w2_float.new_zeros(w2_float.shape)
        grad_x = torch.empty(
            (x_view.shape[0], factor, in_dim), dtype=x.dtype, device=x.device
        )
        w2_t = w2_float.transpose(0, 1)
        chunk_rows = _projection_chunk_rows(in_dim, out_dim, chunk_bytes)
        out_slice = slice(out_start, out_start + out_count)
        for row_start in range(0, x_view.shape[0], chunk_rows):
            row_end = min(row_start + chunk_rows, x_view.shape[0])
            grad = grad_view[row_start:row_end].float()
            for in_factor in range(factor):
                x_slice = x_view[row_start:row_end, in_factor, :].float()
                projected = x_slice.matmul(w2_t)
                grad_w1[out_slice, in_factor].add_(
                    torch.einsum("ngo,no->g", grad, projected)
                )

                coeffs = w1_float[out_slice, in_factor]
                grad_projected = torch.einsum("g,ngo->no", coeffs, grad)
                grad_w2.add_(grad_projected.transpose(0, 1).matmul(x_slice))
                grad_x[row_start:row_end, in_factor, :] = grad_projected.matmul(
                    w2_float
                ).to(x.dtype)
        grad_x = grad_x.reshape(*ctx.leading_shape, factor * in_dim)

        return (
            grad_x,
            grad_w1.to(w1.dtype),
            grad_w2.to(w2.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LoKrAddGroupedDeltaFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        base,
        x,
        w1,
        w2,
        gate_scale,
        factor,
        in_dim,
        out_dim,
        group_size,
        chunk_bytes,
        backward_backend,
    ):
        factor = int(factor)
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        group_size = max(1, min(int(group_size), factor))
        chunk_bytes = _normalize_lokr_project_chunk_bytes(chunk_bytes)
        backward_backend = normalize_lokr_grouped_delta_backward_backend(
            backward_backend
        )
        ctx.mark_dirty(base)
        result = _lokr_add_grouped_delta_forward_eager(
            base,
            x,
            w1,
            w2,
            gate_scale,
            factor,
            in_dim,
            out_dim,
            group_size,
            chunk_bytes,
        )
        _save_lokr_add_grouped_delta_ctx(
            ctx,
            x=x,
            w1=w1,
            w2=w2,
            gate_scale=gate_scale,
            factor=factor,
            in_dim=in_dim,
            out_dim=out_dim,
            group_size=group_size,
            chunk_bytes=chunk_bytes,
        )
        ctx.backward_backend = backward_backend
        return result

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2, gate_scale = ctx.saved_tensors
        grad_x, grad_w1, grad_w2 = _lokr_add_grouped_delta_backward(
            grad_out,
            x,
            w1,
            w2,
            gate_scale,
            ctx.factor,
            ctx.in_dim,
            ctx.out_dim,
            ctx.group_size,
            ctx.chunk_bytes,
            ctx.leading_shape,
            ctx.backward_backend,
        )

        return (
            grad_out,
            grad_x,
            grad_w1,
            grad_w2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LoKrAddGroupedDeltaTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        base,
        x,
        w1,
        w2,
        gate_scale,
        factor,
        in_dim,
        out_dim,
        group_size,
        chunk_bytes,
        backward_backend,
    ):
        factor = int(factor)
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        group_size = max(1, min(int(group_size), factor))
        chunk_bytes = _normalize_lokr_project_chunk_bytes(chunk_bytes)
        backward_backend = normalize_lokr_grouped_delta_backward_backend(
            backward_backend
        )
        ctx.mark_dirty(base)
        result = _launch_lokr_grouped_delta_triton(
            base,
            x,
            w1,
            w2,
            gate_scale,
            factor,
            in_dim,
            out_dim,
        )
        _save_lokr_add_grouped_delta_ctx(
            ctx,
            x=x,
            w1=w1,
            w2=w2,
            gate_scale=gate_scale,
            factor=factor,
            in_dim=in_dim,
            out_dim=out_dim,
            group_size=group_size,
            chunk_bytes=chunk_bytes,
        )
        ctx.backward_backend = backward_backend
        return result

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2, gate_scale = ctx.saved_tensors
        grad_x, grad_w1, grad_w2 = _lokr_add_grouped_delta_backward(
            grad_out,
            x,
            w1,
            w2,
            gate_scale,
            ctx.factor,
            ctx.in_dim,
            ctx.out_dim,
            ctx.group_size,
            ctx.chunk_bytes,
            ctx.leading_shape,
            ctx.backward_backend,
        )

        return (
            grad_out,
            grad_x,
            grad_w1,
            grad_w2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable(recursive=True)
def lokr_project(x, w1, w2, factor, in_dim, out_dim):
    """Project through the LyCORIS-compatible Kronecker LoKR weight.

    ``torch.kron(w1, w2)`` materializes the full ``out_features x in_features``
    adapter weight. On Anima DiT's wide projections that temporary is large
    enough to erase the memory saved by block swap, so training uses the
    equivalent two-stage contraction instead:

    1. apply ``w2`` inside each Kronecker factor slice;
    2. mix factor slices with ``w1``.
    """

    return LoKrProjectFn.apply(x, w1, w2, factor, in_dim, out_dim)


@torch.compiler.disable(recursive=True)
def lokr_project_factor(
    x, w1, w2, out_factor, factor, in_dim, out_dim, chunk_bytes=None
):
    """Project one output factor slice of the Kronecker LoKr weight."""

    return LoKrProjectFactorFn.apply(
        x,
        w1,
        w2,
        out_factor,
        factor,
        in_dim,
        out_dim,
        _normalize_lokr_project_chunk_bytes(chunk_bytes),
    )


@torch.compiler.disable(recursive=True)
def lokr_project_factor_group(
    x, w1, w2, out_start, out_count, factor, in_dim, out_dim, chunk_bytes=None
):
    """Project a small group of output factor slices.

    This keeps the peak below a full LoKr output tensor but avoids recomputing
    the same ``x @ w2.T`` once for every single output factor.
    """

    return LoKrProjectFactorGroupFn.apply(
        x,
        w1,
        w2,
        out_start,
        out_count,
        factor,
        in_dim,
        out_dim,
        _normalize_lokr_project_chunk_bytes(chunk_bytes),
    )


@torch.compiler.disable(recursive=True)
def lokr_add_grouped_delta_(
    base,
    x,
    w1,
    w2,
    gate_scale,
    factor,
    in_dim,
    out_dim,
    group_size,
    chunk_bytes=None,
    backend=DEFAULT_LOKR_GROUPED_DELTA_BACKEND,
    backward_backend=DEFAULT_LOKR_GROUPED_DELTA_BACKWARD_BACKEND,
):
    """Add the LoKr delta into ``base`` without materializing a full delta output.

    ``lokr_project_factor_group`` returns an ``N × group × out_dim`` tensor for
    every output-factor group. On Anima MLP layer1 that temporary is ~132MiB in
    fp32 when ``group_size=8``. This fused autograd function computes the same
    contraction row-chunk by row-chunk and writes directly into the frozen
    Linear output, while backward recomputes the projection from saved inputs.
    ``backend="triton"`` switches the forward path to an experimental fused
    CUDA kernel when the input layout and device allow it. ``backward_backend``
    can opt into the experimental Triton-assisted gradient paths; unsupported
    devices, dtypes, shapes, or layouts retain the eager recompute formula.
    """
    normalized_backend = normalize_lokr_grouped_delta_backend(backend)
    normalized_backward_backend = normalize_lokr_grouped_delta_backward_backend(
        backward_backend
    )
    args = (
        base,
        x,
        w1,
        w2,
        gate_scale,
        factor,
        in_dim,
        out_dim,
        group_size,
        _normalize_lokr_project_chunk_bytes(chunk_bytes),
        normalized_backward_backend,
    )
    if normalized_backend == "triton" and _can_use_lokr_grouped_delta_triton(
        base,
        x,
        w1,
        w2,
        gate_scale,
        int(factor),
        int(in_dim),
        int(out_dim),
    ):
        return LoKrAddGroupedDeltaTritonFn.apply(*args)
    return LoKrAddGroupedDeltaFn.apply(*args)
