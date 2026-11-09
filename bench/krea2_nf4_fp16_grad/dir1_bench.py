"""dir1 的算子级基准与真实权重串联精度。

两件事：

1. **算子级速度**：V100 上 bf16 GEMM 走慢路径（无原生 bf16 tensor core），fp16 走
   真 tensor core。dir1 在付掉 ``frexp``/``ldexp``/``clamp`` 的额外开销后还剩多少净收益。
2. **真实权重 28 层串联精度**：用 ``blocks.0..27.attn.wq`` 的真实 NF4 权重串联，
   看真实奇异值分布下 dir1 与 bf16 的误差累积是否仍持平。

**基准必须用真实 token 数 M，不能用 M=1。** M=1 是「一个梯度向量乘权重」，此时
``6144×6144`` 的 GEMM 完全受显存带宽限制（主要开销是读 72 MB 权重），bf16 与 fp16
速度相同，dir1 的额外 kernel 反而让它慢一倍——**会得出完全相反的结论**。
本脚本默认同时打印 M=1 作对照，提醒这个陷阱。

用法::

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/dir1_bench.py
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/dir1_bench.py --skip-chain
"""

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2FP16_GPU", "1"))

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trajectory_gate import KREA2_NF4, load_krea2_dit

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device("cuda")
GATE = 0.61  # 由实测反推：顶层 4.4e-3 -> 深层 6.5e-9 共 5.8 decade / 28 层


def bench(fn, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def dir1_linear(x: torch.Tensor, w16: torch.Tensor) -> torch.Tensor:
    xb = x.to(torch.bfloat16)
    exp = torch.frexp(xb.abs().max())[1]
    shift = torch.clamp(-exp, min=-126, max=126)
    x16 = torch.ldexp(xb.float(), shift).to(torch.float16)
    out = torch.nn.functional.linear(x16, w16.t())
    return torch.ldexp(out.float(), -shift).to(torch.bfloat16)


def dir1_matmul(g: torch.Tensor, w16: torch.Tensor) -> torch.Tensor:
    gb = g.to(torch.bfloat16)
    exp = torch.frexp(gb.abs().max())[1]
    shift = torch.clamp(-exp, min=-126, max=126)
    g16 = torch.ldexp(gb.float(), shift).to(torch.float16)
    # 外层是 bf16，输出必须截断回 bf16 —— 不截断会虚高约 0.6pp（0.72% vs 1.35%）
    return torch.ldexp(torch.matmul(g16, w16).float(), -shift).to(torch.bfloat16).float()


def bench_ops(k: int) -> None:
    w32 = torch.randn(k, k, device=DEV, dtype=torch.float32) * 0.02
    w_bf = w32.to(torch.bfloat16)
    w16 = w32.to(torch.float16)

    print(f"=== 算子级基准 K={k}（cuda.Event，100 次中位数）===")
    print(
        f"{'M':>7} {'方向':<9} {'bf16 ms':>9} {'fp16 ms':>9} {'dir1 ms':>9} "
        f"{'bf16/fp16':>10} {'bf16/dir1':>10}"
    )
    for m in (1, 256, 1024, 4096, 16384):
        x = torch.randn(m, k, device=DEV, dtype=torch.float32) * 1e-2
        g = torch.randn(m, k, device=DEV, dtype=torch.float32) * 1e-8
        t_bf_f = bench(lambda x=x, w=w_bf: torch.nn.functional.linear(x.to(torch.bfloat16), w.t()))
        t_fp_f = bench(lambda x=x, w=w16: torch.nn.functional.linear(x.to(torch.float16), w.t()))
        t_d1_f = bench(lambda x=x, w=w16: dir1_linear(x, w))
        t_bf_b = bench(lambda g=g, w=w_bf: torch.matmul(g.to(torch.bfloat16), w))
        t_fp_b = bench(lambda g=g, w=w16: torch.matmul(g.to(torch.float16), w))
        t_d1_b = bench(lambda g=g, w=w16: dir1_matmul(g, w))
        tag = "  <- 带宽限制，勿用于结论" if m == 1 else ""
        print(
            f"{m:>7} {'forward':<9} {t_bf_f:>9.3f} {t_fp_f:>9.3f} {t_d1_f:>9.3f} "
            f"{t_bf_f / t_fp_f:>9.2f}x {t_bf_f / t_d1_f:>9.2f}x{tag}"
        )
        print(
            f"{'':>7} {'backward':<9} {t_bf_b:>9.3f} {t_fp_b:>9.3f} {t_d1_b:>9.3f} "
            f"{t_bf_b / t_fp_b:>9.2f}x {t_bf_b / t_d1_b:>9.2f}x"
        )
        del x, g
        torch.cuda.empty_cache()
    del w32, w_bf, w16
    torch.cuda.empty_cache()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    denom = b.norm()
    return ((a - b).norm() / denom).item() if denom > 0 else float("nan")


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def dequant_as(B: torch.Tensor, quant_state, dtype: torch.dtype) -> torch.Tensor:
    """按指定 dtype dequant（``copy.copy`` 浅拷贝 QuantState，原对象零改动）。

    不用 bnb 的 ``out=``：0.49.2 的 cuda kernel 里 ``torch._check(out.shape == shape)``
    的 ``shape`` 经 ``torch.ops`` 的 ``int[]`` schema 被转成 list，而 ``out.shape`` 是
    ``torch.Size``；``torch.Size == list`` 恒为 False，所以传 Size/tuple/list 都报
    ``Expected out.shape == [6144, 6144], got torch.Size([6144, 6144])``。
    """
    import bitsandbytes.functional as bnb_f

    qs = copy.copy(quant_state)
    qs.dtype = dtype
    return bnb_f.dequantize_4bit(B, qs)


def chain_precision(img_tag: str) -> None:
    dit = load_krea2_dit(KREA2_NF4, device=str(DEV), dtype=torch.bfloat16, eval=True)
    layers = [dict(dit.blocks[i].named_modules())["attn.wq"] for i in range(28)]
    k = layers[0].in_features
    print(f"\n=== 真实权重 28 层串联（blocks.*.attn.wq，{k}²，gate={GATE}）===")

    torch.manual_seed(0)  # 固定初始梯度，否则末层误差在 0.99%~1.37% 之间抖动
    g0 = torch.randn(1, k, device=DEV, dtype=torch.float32)
    g0 = g0 / g0.pow(2).mean().sqrt() * 4.4e-3
    cur = {name: g0.clone() for name in ("fp32", "bf16", "dir1", "fp16")}
    hist = {name: [] for name in cur}

    for i in range(28):
        wq = layers[i].weight
        # fp32 dequant 作为真值：若参考臂也用 bf16 dequant，bf16 臂天然零权重误差，
        # 比较不公平（第一版正是踩了这个坑，得出「dir1 略差」的错误结论）。
        w32 = dequant_as(wq.data, wq.quant_state, torch.float32)
        w_bf = dequant_as(wq.data, wq.quant_state, torch.bfloat16)
        w16 = dequant_as(wq.data, wq.quant_state, torch.float16)
        n_ref = cur["fp32"].norm()
        cur["fp32"] = torch.matmul(cur["fp32"], w32)
        cur["bf16"] = torch.matmul(cur["bf16"].to(torch.bfloat16), w_bf).float()
        cur["fp16"] = torch.matmul(cur["fp16"].to(torch.float16), w16).float()
        cur["dir1"] = dir1_matmul(cur["dir1"], w16)
        # 真实模型里 Linear 之后有 LayerNorm 类归一化；不做归一化时真实 wq 会让
        # 梯度放大 3e7 倍并让 fp16 直接溢出，那不是真实反向链的行为。
        updated = {}
        for name, val in cur.items():
            n = val.norm()
            if n > 0 and torch.isfinite(n):
                val = val / n * n_ref
            updated[name] = val * GATE
            hist[name].append(updated[name].clone())
        cur = updated
        del w32, w_bf, w16
        if i % 7 == 0:
            torch.cuda.empty_cache()

    print(f"{'层':>4} {'bf16':>10} {'dir1':>10} {'fp16':>10}")
    for i in (0, 4, 9, 13, 17, 21, 24, 27):
        print(
            f"{i:>4} {rel_l2(hist['bf16'][i], hist['fp32'][i]) * 100:>9.2f}% "
            f"{rel_l2(hist['dir1'][i], hist['fp32'][i]) * 100:>9.2f}% "
            f"{rel_l2(hist['fp16'][i], hist['fp32'][i]) * 100:>9.2f}%"
        )
    print(
        f"\n衰减 {hist['fp32'][0].norm().item():.3e} -> "
        f"{hist['fp32'][-1].norm().item():.3e}"
    )
    for name in ("bf16", "dir1", "fp16"):
        last = hist[name][-1]
        print(
            f"  {name:<6} 末层 rel-L2={rel_l2(last, hist['fp32'][-1]) * 100:>7.3f}% "
            f"cos={cos_sim(last, hist['fp32'][-1]):>8.5f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--skip-chain", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"=== GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} ===")
    bench_ops(args.k)
    if not args.skip_chain:
        chain_precision(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
