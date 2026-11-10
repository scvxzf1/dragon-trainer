"""Shared packed Linear with the CUDA BF16 autocast output contract."""
from __future__ import annotations

import copy

import torch
from bitsandbytes.nn import Linear4bit

from .constants import STATE_KEYS

#: NF4 Linear 内层 fp16 GEMM（dir1）的受控开关，默认关闭。
#:
#: 只对**没有原生 bf16 tensor core** 的卡有意义（sm70/sm75）。在 sm80+ 上 bf16
#: 本来就走 tensor core，dir1 只是纯亏损，所以 ``set_dir1_enabled(True)`` 会直接拒绝。
#: 本开关**不改变** ``compute_dtype``（始终 bfloat16），因此不触碰
#: ``weights.py`` / ``quantize.py`` 的 BF16 fail-closed 契约，也不是"全局 FP16 NF4"。
_DIR1_ENABLED = False


def dir1_supported() -> bool:
    """当前设备是否属于 dir1 有收益的架构（无原生 bf16 tensor core）。"""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] < 8


def is_dir1_enabled() -> bool:
    return _DIR1_ENABLED


def set_dir1_enabled(enabled: bool) -> None:
    """启用/关闭 NF4 Linear 的内层 fp16 GEMM（dir1）。

    证据见 ``docs/proposal/krea2_nf4_fp16_layer_scaling.md``：V100 上算子级
    5~6.6×、端到端 3.33×，精度与 bf16 持平（权重侧 fp16-dequant 还能反超）。
    """
    global _DIR1_ENABLED
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("dir1 需要 CUDA 设备")
        major = torch.cuda.get_device_capability()[0]
        if major >= 8:
            raise ValueError(
                f"dir1 只对 sm<80（无原生 bf16 tensor core）有收益；"
                f"当前设备 sm_{major}0 上 bf16 已走 tensor core，启用只会变慢"
            )
    _DIR1_ENABLED = bool(enabled)


def dequant_fp16(weight: torch.Tensor, quant_state) -> torch.Tensor:
    """按 fp16 直接 dequant 出 10 位尾数的权重。

    bnb 的 dequant 内核在 FP32 寄存器里查表并乘 absmax，只在写回时按
    ``quant_state.dtype`` 转换。把 ``dtype`` 设为 fp16 可以省掉
    ``dequant(bf16) -> .half()`` 的 72 MB 读 + 72 MB 写（实测 0.581 -> 0.308 ms），
    同时把权重精度从 bf16 的 7 位尾数提到 fp16 的 10 位。

    用 ``copy.copy`` 浅拷贝 ``QuantState``（轻量 Python 壳，不复制 GPU 张量），
    原对象全程零改动，因此多线程并发 dequant 无竞争，不需要锁。

    不用 bnb 的 ``out=`` 参数：0.49.2 的 kernel 里 ``torch._check(out.shape == shape)``
    的 ``shape`` 经 ``torch.ops`` 的 ``int[]`` schema 被转成 list，而 ``out.shape`` 是
    ``torch.Size``；``torch.Size == list`` 恒为 False，传 Size/tuple/list 都会失败。
    """
    from bitsandbytes.functional import dequantize_4bit

    qs = copy.copy(quant_state)
    qs.dtype = torch.float16
    return dequantize_4bit(weight, qs)


class _Dir1MatMul4Bit(torch.autograd.Function):
    """NF4 packed Linear 的 fp16 内层 GEMM + 逐层 ``frexp``/``ldexp`` 缩放。

    外层 ``compute_dtype`` 保持 bfloat16：它的 8 位指数负责把 1e-9 量级的梯度跨层
    运送（fp16 从 1e-8 起就归零）。内层用 fp16 走真 tensor core，backward 时用
    ``frexp`` 取 amax 的二进制指数、``ldexp`` 平移（只改指数位，尾数无损），
    算完再平移回来，从而彻底消除 fp16 subnormal。

    三个实现要点：
    - ``torch.ldexp`` 第二参数必须是 Tensor，不能传 Python int；
    - 平移量必须双向 clamp：``min=-126`` 防向下溢出，``max=126`` 防
      ``amax=1e-40 -> exp=-132 -> -exp=+132`` 向上溢出；
    - ``frexp(0.0) == (0.0, 0)``，全零梯度天然免疫，不需要 eps/where 分支。
    """

    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        ctx.is_empty = A.numel() == 0
        ctx.state = quant_state
        ctx.dtype_A = A.dtype
        ctx.bias = bias
        ctx.tensors = (None, B)
        if ctx.is_empty:
            shape = quant_state.shape
            return torch.empty(A.shape[:-1] + tuple(shape[1:]), dtype=A.dtype, device=A.device)
        w = dequant_fp16(B, quant_state)
        # 前向不需要缩放：激活量级 O(1e-2~1e0)，本来就在 fp16 正规区。
        # bias 必须一起转 fp16，否则 F.linear 内部 matmul 报 dtype 不匹配。
        # 关键：训练跑在 torch.autocast(dtype=bfloat16) 里。autocast 会把
        # F.linear 的 fp16 输入静默转回 bf16 执行，于是整个 dir1 白做——GEMM
        # 仍然落到 sm70 的 MAGMA 软件路径（实测 27.3s/步，占 78%）。
        # 这段内部计算是纯 fp16 的，必须在 autocast 外执行。
        with torch.autocast(device_type="cuda", enabled=False):
            output = torch.nn.functional.linear(
                A.to(torch.float16),
                w.t(),
                None if bias is None else bias.to(torch.float16),
            )
        if output.dtype != torch.float16:
            output = output.to(torch.float16)

        return output.to(ctx.dtype_A)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            return torch.zeros_like(ctx.tensors[1]), None, None, None, None
        _, B = ctx.tensors
        w = dequant_fp16(B, ctx.state)
        exp = torch.frexp(grad_output.abs().max())[1]
        shift = torch.clamp(-exp, min=-126, max=126)
        g16 = torch.ldexp(grad_output.float(), shift).to(torch.float16)
        # Backward normally runs outside any autocast region, so this guard is
        # defensive: if a caller ever drives backward inside ``autocast(bf16)``
        # (custom training loop, hook, or an inner forward that re-enters), the
        # fp16 operands would be silently promoted back to bf16 -- dir1 would do
        # double work and lose precision at the same time.
        with torch.autocast(device_type="cuda", enabled=False):
            out = torch.matmul(g16, w.t())
        grad_A = torch.ldexp(out.float(), -shift).to(ctx.dtype_A)
        grad_bias = None
        if ctx.bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0, dtype=ctx.bias.dtype)
        return grad_A, None, None, grad_bias, None


class BF16AutocastLinear4bit(Linear4bit):
    """Keep bnb autograd/storage, but avoid restoring FP32 autocast outputs.

    Linear4bit casts its result back to the input dtype. Casting the input
    before its forward preserves ordinary Linear's BF16 autocast contract
    without allocating a full FP32 result or changing saved packed aliases.
    Other devices, compute dtypes and disabled/nested autocast use stock bnb.

    When ``set_dir1_enabled(True)`` is active on a sm<80 device, the GEMM runs
    in fp16 with per-layer frexp/ldexp scaling; ``compute_dtype`` stays bfloat16
    so the BF16 fail-closed contracts are untouched.
    """

    def forward(self, x: torch.Tensor):
        if _DIR1_ENABLED and x.device.type == "cuda":
            return self._forward_dir1(x)
        if (
            x.device.type == "cuda"
            and x.dtype in (torch.float32, torch.float16)
            and self.compute_dtype == torch.bfloat16
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16
        ):
            x = x.to(torch.bfloat16)
        return super().forward(x)

    def _forward_dir1(self, x: torch.Tensor):
        """复刻 ``Linear4bit.forward`` 的预处理，但把 GEMM 换成 dir1。"""
        from bitsandbytes.nn.modules import fix_4bit_weight_quant_state_from_module

        fix_4bit_weight_quant_state_from_module(self)
        quant_state = self.weight.quant_state
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)
        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True
        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)
        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        out = _Dir1MatMul4Bit.apply(x, self.weight.t(), None, bias, quant_state)
        return out.to(inp_dtype)

def _valid_nested_code(code) -> bool:
    from bitsandbytes.functional import create_dynamic_map
    expected = create_dynamic_map()
    if torch.equal(code, expected):
        return True
    # bnb's native CPU blockwise kernel changes only the first endpoint to -1.
    expected[0] = -1.0
    return torch.equal(code, expected)


def quant_state(state: dict, weight_key: str, shape) -> tuple[dict, object]:
    from bitsandbytes.functional import QuantState, get_4bit_type
    qs = {k: state[weight_key + '.' + k] for k in STATE_KEYS if weight_key + '.' + k in state}
    if set(qs) != STATE_KEYS or qs['quant_state.bitsandbytes__nf4'].numel() > 65536:
        raise ValueError(f'Incomplete or oversized NF4 state: {weight_key}')
    try:
        q = QuantState.from_dict(dict(qs), device=torch.device('cpu'))
    except Exception as exc:
        raise ValueError(f'Invalid NF4 quant_state: {weight_key}') from exc
    n = shape[0] * shape[1]
    if (q.quant_type != 'nf4' or q.blocksize != 64 or q.dtype != torch.bfloat16
            or tuple(q.shape) != tuple(shape) or not q.nested or q.state2.blocksize != 256
            or q.state2.dtype != torch.float32
            or q.absmax.dtype != torch.uint8 or tuple(q.absmax.shape) != (n // 64,)
            or q.code.dtype != torch.float32 or tuple(q.code.shape) != (16,)
            or q.state2.code.dtype != torch.float32 or tuple(q.state2.code.shape) != (256,)
            or q.state2.absmax.dtype != torch.float32
            or tuple(q.state2.absmax.shape) != ((n // 64 + 255) // 256,)
            or not torch.isfinite(q.code).all() or not torch.isfinite(q.state2.code).all()
            or not torch.isfinite(q.state2.absmax).all() or (q.state2.absmax < 0).any()
            or q.offset.numel() != 1 or not torch.isfinite(q.offset).all() or (q.offset < 0).any()
            or not torch.equal(q.code, get_4bit_type('nf4', device='cpu'))
            or not _valid_nested_code(q.state2.code)):
        raise ValueError(f'NF4 quant_state violates fixed policy: {weight_key}')
    return qs, q


def restore_nf4_linear(data, qs, *, in_features, out_features, bias=None, device='cpu'):
    """Restore validated packed storage directly, without requantization."""
    from bitsandbytes.nn import Params4bit
    shape = (out_features, in_features)
    if (in_features <= 0 or out_features <= 0 or in_features % 64 or out_features % 64
            or data.dtype != torch.uint8 or tuple(data.shape) != (in_features * out_features // 2, 1)):
        raise ValueError('Invalid packed NF4 weight shape or dtype')
    quant_state({'weight.' + k: v for k, v in qs.items()}, 'weight', shape)
    if bias is not None and (tuple(bias.shape) != (out_features,) or bias.dtype != torch.bfloat16):
        raise ValueError('Invalid NF4 bias')
    new = BF16AutocastLinear4bit(in_features, out_features, bias=bias is not None,
        compute_dtype=torch.bfloat16, compress_statistics=True, quant_type='nf4', device='meta')
    new.weight = Params4bit.from_prequantized(data, dict(qs), requires_grad=False,
                                             device=device, module=new)
    if bias is not None:
        new.bias = torch.nn.Parameter(bias.to(device), requires_grad=False)
    return new
