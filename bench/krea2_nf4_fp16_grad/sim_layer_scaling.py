"""逐层缩放（dir1）方案的合成矩阵模拟。

回答两个问题：
1. 单层上，dir1 能否把 fp16 的下溢抹平到与 bf16 同级？
2. 28 层串联反传时，dir1 的误差累积与 bf16 相比如何？

三个臂共用同一份权重、同一份初始梯度，fp32 为真值裁判：

- ``bf16``：``matmul(g.bf16(), w.bf16())``，输出再截断 bf16（模拟外层 bf16）
- ``fp16``：``matmul(g.fp16(), w.fp16())``，无缩放（裸 fp16，会下溢）
- ``dir1``：外层 bf16 → ``frexp`` 取 amax 的二进制指数 → ``ldexp`` 平移 →
  fp16 GEMM → ``ldexp`` 平移回来 → 截断 bf16

dir1 的关键实现细节（都是踩过的坑）：

- ``torch.ldexp`` 第二个参数必须是 Tensor，不能传 Python int；
- 平移量必须**双向** clamp：``min=-126`` 防向下溢出，``max=126`` 防向上溢出。
  只写 ``min=-126`` 拦不住 ``amax=1e-40 → exp=-132 → -exp=+132``，仍然出 inf；
- ``frexp(0.0) == (0.0, 0)``，所以全零梯度天然免疫，不需要 eps 或 where 分支。

用法::

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/sim_layer_scaling.py
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/sim_layer_scaling.py --layer-dim 6144
"""

from __future__ import annotations

import argparse
import math

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    denom = b.norm()
    return ((a - b).norm() / denom).item() if denom > 0 else float("nan")


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def dir1_matmul(g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """外层 bf16 语义 + 内层 frexp 归一化 fp16 GEMM。"""
    gb = g.to(torch.bfloat16)
    exp = torch.frexp(gb.abs().max())[1]
    shift = torch.clamp(-exp, min=-126, max=126)
    g16 = torch.ldexp(gb.float(), shift).to(torch.float16)
    out = torch.matmul(g16, w.to(torch.float16))
    return torch.ldexp(out.float(), -shift).to(torch.bfloat16).float()


def arm_matmul(g: torch.Tensor, w: torch.Tensor, arm: str) -> torch.Tensor:
    if arm == "fp32":
        return torch.matmul(g, w)
    if arm == "bf16":
        out = torch.matmul(g.to(torch.bfloat16), w.to(torch.bfloat16)).float()
        return out.to(torch.bfloat16).float()
    if arm == "fp16":
        out = torch.matmul(g.to(torch.float16), w.to(torch.float16)).float()
        return out.to(torch.float16).float()
    if arm == "dir1":
        return dir1_matmul(g, w)
    raise ValueError(f"unknown arm: {arm}")


def make_weight(dim: int, kind: str, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    if kind == "gauss":
        w = torch.randn(dim, dim, device=DEVICE, dtype=torch.float32, generator=gen)
        w = w / math.sqrt(dim)
    elif kind == "orth":
        w, _ = torch.linalg.qr(
            torch.randn(dim, dim, device=DEVICE, dtype=torch.float32, generator=gen)
        )
    else:
        raise ValueError(f"unknown weight kind: {kind}")
    return w.to(torch.bfloat16).float()


def unit_rms_grad(dim: int, rms: float, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    g = torch.randn(1, dim, device=DEVICE, dtype=torch.float32, generator=gen)
    return g / g.pow(2).mean().sqrt() * rms


def sweep_layer(dim: int, seed: int) -> None:
    """单层：dY rms 从 1e-10 扫到 1e-2，覆盖 1024² 最深层量级。"""
    w = make_weight(dim, "gauss", seed)
    print("=" * 92)
    print(f"实验 1：单层 {dim}x{dim} 量级扫描（权重 rms≈{w.std().item():.4f}，fp32 为真值）")
    print("=" * 92)
    print(
        f"{'dY rms':>10} | {'bf16 rel':>9} {'cos':>8} | "
        f"{'fp16 rel':>10} {'cos':>8} | {'dir1 rel':>9} {'cos':>8}"
    )
    print("-" * 92)
    for e in range(-10, -1):
        for mant in (1.0, 3.0):
            rms = mant * 10.0**e
            g = unit_rms_grad(dim, rms, seed + 1)
            ref = torch.matmul(g, w)
            vals = {}
            for arm in ("bf16", "fp16", "dir1"):
                out = arm_matmul(g, w, arm)
                vals[arm] = (rel_l2(out, ref) * 100, cos_sim(out, ref))
            print(
                f"{rms:>10.1e} | {vals['bf16'][0]:>8.3f}% {vals['bf16'][1]:>8.5f} | "
                f"{vals['fp16'][0]:>9.3f}% {vals['fp16'][1]:>8.5f} | "
                f"{vals['dir1'][0]:>8.3f}% {vals['dir1'][1]:>8.5f}"
            )


def chain(dim: int, layers: int, gate: float, kind: str, seed: int) -> None:
    """28 层串联：所有臂共享同一份权重与同一份初始梯度。"""
    w = make_weight(dim, kind, seed)
    g0 = unit_rms_grad(dim, 4.4e-3, seed + 2)

    arms = ("fp32", "bf16", "fp16", "dir1")
    cur = {a: g0.clone() for a in arms}
    hist = {a: [] for a in arms}
    for _ in range(layers):
        for a in arms:
            cur[a] = arm_matmul(cur[a], w, a) * gate
            hist[a].append(cur[a].clone())

    print()
    print("=" * 92)
    print(
        f"实验 2：{layers} 层串联反传（{dim}x{dim}，{kind} 权重，gate={gate}，"
        f"衰减 {hist['fp32'][0].norm().item() / hist['fp32'][-1].norm().item():.1e} 倍）"
    )
    print("=" * 92)
    print(f"{'臂':<10} {'末层 |g|':>12} {'rel-L2 vs fp32':>16} {'cos':>10}")
    print("-" * 92)
    for a in ("bf16", "fp16", "dir1"):
        last = hist[a][-1]
        print(
            f"{a:<10} {last.norm().item():>12.3e} "
            f"{rel_l2(last, hist['fp32'][-1]) * 100:>15.3f}% {cos_sim(last, hist['fp32'][-1]):>10.5f}"
        )

    print()
    print("误差随深度累积（rel-L2 vs fp32）：")
    print(f"{'层':>4} {'bf16':>10} {'fp16':>10} {'dir1':>10}")
    samples = sorted({0, 4, 9, 14, 19, 23, layers - 1})
    for i in samples:
        if i >= layers:
            continue
        row = " ".join(
            f"{rel_l2(hist[a][i], hist['fp32'][i]) * 100:>9.2f}%"
            for a in ("bf16", "fp16", "dir1")
        )
        print(f"{i:>4} {row}")


def boundary_checks(dim: int) -> None:
    """dir1 的三个边界：全零、amax 极小、负零。"""
    print()
    print("=" * 92)
    print("实验 3：dir1 边界条件")
    print("=" * 92)
    w = make_weight(min(dim, 512), "gauss", 99)
    cases = {
        "全零梯度": torch.zeros(1, w.shape[0], device=DEVICE),
        "amax=1e-40": torch.full((1, w.shape[0]), 1e-40, device=DEVICE),
        "负零": torch.full((1, w.shape[0]), -0.0, device=DEVICE),
    }
    for name, g in cases.items():
        out = dir1_matmul(g, w)
        print(
            f"  {name:<12} 输入 max={g.abs().max().item():.3e} "
            f"输出有限={bool(torch.isfinite(out).all())} 输出 max={out.abs().max().item():.3e}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer-dim", type=int, default=6144)
    ap.add_argument("--chain-dim", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--gate", type=float, default=0.61)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"=== device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} ===")

    if not args.skip_sweep:
        sweep_layer(args.layer_dim, args.seed)
    chain(args.chain_dim, args.layers, args.gate, "orth", args.seed + 10)
    chain(args.chain_dim, args.layers, args.gate, "gauss", args.seed + 20)
    boundary_checks(args.layer_dim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
