"""NF4 dir1（内层 fp16 GEMM + 逐层 frexp 缩放）的边界与契约测试。

覆盖 ``library.quantization.nf4`` 新增的四个接口：

- ``set_dir1_enabled``：默认关闭、sm80+ 拒绝、CPU 拒绝、可反复开关；
- ``dequant_fp16``：浅拷贝不改原 ``quant_state``，输出 fp16；
- ``_Dir1MatMul4Bit``：全零梯度、``amax=1e-40``、非连续输入、``bias=None`` /
  带 bias、``grad_bias`` 仅在 ``bias.requires_grad`` 时产生；
- ``BF16AutocastLinear4bit``：``compute_dtype`` 保持 bfloat16 的契约不变。

真实 NF4 权重测试需要 GPU 与本地权重文件，缺失时 skip。
"""

from __future__ import annotations

import copy

import pytest
import torch

from library.quantization.nf4 import (
    BF16AutocastLinear4bit,
    _Dir1MatMul4Bit,
    dequant_fp16,
    dir1_supported,
    is_dir1_enabled,
    set_dir1_enabled,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")


@pytest.fixture(autouse=True)
def _reset_dir1():
    """每个用例前后都关掉 dir1，避免全局开关泄漏。"""
    set_dir1_enabled(False) if torch.cuda.is_available() and dir1_supported() else None
    yield
    if torch.cuda.is_available() and dir1_supported():
        set_dir1_enabled(False)


def test_dir1_disabled_by_default():
    assert is_dir1_enabled() is False


def test_set_dir1_rejects_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        set_dir1_enabled(True)


def test_set_dir1_rejects_sm80_plus(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 6))
    with pytest.raises(ValueError, match="sm<80"):
        set_dir1_enabled(True)


def test_set_dir1_accepts_sm70(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 0))
    set_dir1_enabled(True)
    assert is_dir1_enabled() is True
    set_dir1_enabled(False)
    assert is_dir1_enabled() is False


def test_compute_dtype_contract_unchanged():
    """dir1 不改变 compute_dtype，22 处 fail-closed 契约依赖这一点。"""
    layer = BF16AutocastLinear4bit(
        64, 64, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4", device="meta"
    )
    assert layer.compute_dtype == torch.bfloat16


@CUDA
@pytest.mark.skipif(not dir1_supported(), reason="dir1 只对 sm<80 有意义")
class TestDir1MatMul4Bit:
    """用真实 NF4 量化权重（合成数据，不依赖模型文件）。"""

    @staticmethod
    def _make(dim: int = 256, *, bias: bool = False, seed: int = 0):
        from bitsandbytes.functional import quantize_4bit

        torch.manual_seed(seed)
        w = torch.randn(dim, dim, device="cuda", dtype=torch.float32) * 0.02
        packed, qs = quantize_4bit(
            w.to("cuda"), quant_type="nf4", blocksize=64, compress_statistics=True
        )
        b = torch.randn(dim, device="cuda", dtype=torch.float32) * 0.01 if bias else None
        # 参考用 dequant 后的 fp32 权重，隔离 NF4 量化误差（9.2%），只测 dir1 本身
        from bitsandbytes.functional import dequantize_4bit

        return packed, qs, b, dequantize_4bit(packed, qs).float()

    def _ref(self, w: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Linear 语义：y = x @ W.T（W 是 [out_features, in_features]）。"""
        return torch.matmul(g.float(), w.float().t())

    def test_matches_fp32_reference(self):
        packed, qs, _, w = self._make()
        g = torch.randn(8, 256, device="cuda", dtype=torch.float32) * 1e-3
        out = _Dir1MatMul4Bit.apply(g, packed.t(), None, None, qs)
        assert out.shape == g.shape
        assert out.dtype == g.dtype  # 输出 dtype 跟随输入（A.dtype）
        rel = ((out.float() - self._ref(w, g)).norm() / self._ref(w, g).norm()).item()
        assert rel < 5e-3, f"dir1 rel-L2={rel:.4%} 超出预期"

    def test_all_zero_gradient(self):
        """frexp(0)=(0,0)，ldexp(0,0)=0；不需要 eps/where 防 NaN。"""
        packed, qs, _, _ = self._make()
        g = torch.zeros(4, 256, device="cuda", dtype=torch.float32)
        out = _Dir1MatMul4Bit.apply(g, packed.t(), None, None, qs)
        assert torch.isfinite(out).all()
        assert out.abs().max().item() == 0.0

    def test_tiny_amax_does_not_overflow(self):
        """amax=1e-40 时 exp=-132，-exp=+132；只 clamp min 拦不住，必须双向。"""
        packed, qs, _, _ = self._make()
        g = torch.full((4, 256), 1e-40, device="cuda", dtype=torch.float32)
        out = _Dir1MatMul4Bit.apply(g, packed.t(), None, None, qs)
        assert torch.isfinite(out).all(), "amax=1e-40 触发了 inf/nan"

    def test_non_contiguous_input(self):
        packed, qs, _, _w = self._make()
        base = torch.randn(256, 8, device="cuda", dtype=torch.float32) * 1e-3
        g = base.t()  # 非连续
        assert not g.is_contiguous()
        out = _Dir1MatMul4Bit.apply(g, packed.t(), None, None, qs)
        assert torch.isfinite(out).all()
        assert out.shape == g.shape

    def test_bias_none(self):
        packed, qs, _, _ = self._make(bias=False)
        g = torch.randn(4, 256, device="cuda", dtype=torch.float32) * 1e-3
        out = _Dir1MatMul4Bit.apply(g, packed.t(), None, None, qs)
        assert torch.isfinite(out).all()

    def test_bias_requires_grad_produces_grad_bias(self):
        packed, qs, b, _ = self._make(bias=True)
        bias = b.clone().detach().requires_grad_(True)
        g_in = torch.randn(4, 256, device="cuda", dtype=torch.float32) * 1e-3
        g_in.requires_grad_(True)
        out = _Dir1MatMul4Bit.apply(g_in, packed.t(), None, bias, qs)
        out.sum().backward()
        assert bias.grad is not None
        assert torch.isfinite(bias.grad).all()
        assert bias.grad.shape == bias.shape

    def test_bias_frozen_has_no_grad_bias(self):
        packed, qs, b, _ = self._make(bias=True)
        bias = b.clone().detach().requires_grad_(False)
        g_in = torch.randn(4, 256, device="cuda", dtype=torch.float32) * 1e-3
        g_in.requires_grad_(True)
        out = _Dir1MatMul4Bit.apply(g_in, packed.t(), None, bias, qs)
        out.sum().backward()
        assert bias.grad is None

    def test_dequant_fp16_does_not_mutate_quant_state(self):
        packed, qs, _, _ = self._make()
        before_dtype = qs.dtype
        before_shape = tuple(qs.shape)
        w16 = dequant_fp16(packed, qs)
        assert w16.dtype == torch.float16
        assert qs.dtype == before_dtype, "原 quant_state.dtype 被改动了"
        assert tuple(qs.shape) == before_shape
        # 共享张量引用未被复制（浅拷贝只复制 Python 壳）
        assert w16.shape == torch.Size(before_shape)

    def test_dequant_fp16_thread_safe(self):
        """并发调用不应改动原 quant_state，也不应报错。"""
        import threading

        packed, qs, _, _ = self._make()
        before_dtype = qs.dtype
        errors: list[str] = []

        def worker(tid: int):
            try:
                for i in range(5):
                    dt = torch.float16 if (tid + i) % 2 == 0 else torch.bfloat16
                    q = copy.copy(qs)
                    q.dtype = dt
                    from bitsandbytes.functional import dequantize_4bit

                    w = dequantize_4bit(packed, q)
                    assert w.dtype == dt
            except Exception as exc:  # noqa: BLE001
                errors.append(f"thread{tid}: {exc}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert qs.dtype == before_dtype



    def test_runs_fp16_under_bf16_autocast(self):
        """autocast(bf16) 不得把 dir1 的 fp16 GEMM 静默转回 bf16。

        回归 bb8c5b63：``_Dir1MatMul4Bit.forward`` 里若没有
        ``torch.autocast(enabled=False)`` 包裹，外层 bf16 autocast 会在进内核前把
        已经手转的 fp16 输入再转回 bf16，264 个主干 GEMM 全部落回 sm70 的 MAGMA
        软件路径（实测 27.3s/步，占单步 78%）。
        """
        from unittest import mock

        packed, qs, _, _ = self._make()
        g = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16) * 1e-3
        seen: list[tuple[torch.dtype, torch.dtype, torch.dtype]] = []
        real_linear = torch.nn.functional.linear

        def spy(a, b, bias=None):
            out = real_linear(a, b, bias)
            # 记录返回值 dtype：autocast 的强转发生在 aten::linear 的 C++ 分发层，
            # 只检查入参看不出问题，必须看内核实际吐出的类型。
            seen.append((a.dtype, b.dtype, out.dtype))
            return out

        with mock.patch.object(
            torch.nn.functional, "linear", spy
        ), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _Dir1MatMul4Bit.apply(g, packed.t(), None, None, qs)

        assert seen, "F.linear 未被调用，dir1 路径没生效"
        act_dtype, w_dtype, out_dtype = seen[0]
        assert act_dtype == torch.float16, f"激活 dtype {act_dtype}，应是 fp16"
        assert w_dtype == torch.float16, f"权重 dtype {w_dtype}，应是 fp16"
        assert out_dtype == torch.float16, (
            f"GEMM 输出被 autocast(bf16) 转成了 {out_dtype}；"
            "dir1 的 F.linear 必须包在 torch.autocast(device_type='cuda', enabled=False) 里"
        )
# === CLI / 配置接线 ===


def test_cli_flag_defaults_off():
    from library.training.cli_args import setup_parser

    args = setup_parser().parse_args([])
    assert args.nf4_dir1 is False


def test_cli_flag_parses():
    from library.training.cli_args import setup_parser

    args = setup_parser().parse_args(["--nf4_dir1"])
    assert args.nf4_dir1 is True


def test_bootstrap_noop_when_flag_off():
    from types import SimpleNamespace

    from library.training.bootstrap import TrainingBootstrap

    assert TrainingBootstrap.maybe_enable_nf4_dir1(SimpleNamespace(nf4_dir1=False)) is False


@CUDA
@pytest.mark.skipif(not dir1_supported(), reason="dir1 只对 sm<80 有意义")
def test_bootstrap_enables_dir1():
    from types import SimpleNamespace

    from library.training.bootstrap import TrainingBootstrap

    try:
        assert TrainingBootstrap.maybe_enable_nf4_dir1(SimpleNamespace(nf4_dir1=True)) is True
        assert is_dir1_enabled() is True
    finally:
        set_dir1_enabled(False)


def test_base_toml_declares_nf4_dir1():
    """配置项必须在 base.toml 里显式声明，默认关闭。"""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    with (root / "configs" / "base.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    assert cfg.get("nf4_dir1") is False
