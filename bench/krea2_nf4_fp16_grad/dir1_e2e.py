"""dir1 端到端交替基准（bf16 vs dir1，中位数）。

单次测量的端到端耗时受运行顺序、显存碎片、预热影响极大（早期一次测量得到
9.0s -> 3.1s，但同一脚本内 fp32 反而比 bf16 快，说明顺序效应明显）。本脚本用
**交替运行**消除顺序偏差，并取中位数。

dir1 的实现包含 ``qs.dtype=torch.float16`` 的 dequant 优化：bnb 的 dequant 内核
在 FP32 寄存器里查表并乘 absmax，只在写回时按 ``quant_state.dtype`` 转换。把
``dtype`` 临时设为 fp16，可以直接写出 10 位尾数的 fp16 权重，省掉
``dequant(bf16) -> .half()`` 那一次 72 MB 读 + 72 MB 写（实测 0.581 -> 0.308 ms）。
副作用是权重精度从 bf16 的 7 位尾数升到 fp16 的 10 位。

用法::

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/dir1_e2e.py --img 256 --repeat 3
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2FP16_GPU", "1"))

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trajectory_gate import (
    KREA2_NF4,
    build_lora,
    load_krea2_dit,
    prepare_inputs,
    run_trajectory,
)

ROOT = Path(__file__).resolve().parents[2]
CUDA = torch.device("cuda")


def dequant_fp16(B: torch.Tensor, quant_state) -> torch.Tensor:
    """按 fp16 直接 dequant：省一次 bf16 写出 + bf16->fp16 拷贝，且保留 10 位尾数。"""
    import bitsandbytes.functional as bnb_f

    orig = quant_state.dtype
    quant_state.dtype = torch.float16
    try:
        return bnb_f.dequantize_4bit(B, quant_state)
    finally:
        quant_state.dtype = orig


def install_dir1():
    """替换 bnb 的 MatMul4Bit：内层 fp16 GEMM + frexp/ldexp 逐层缩放。

    三个实现要点（都踩过）：
    - ``torch.ldexp`` 第二参数必须是 Tensor；
    - 平移量双向 clamp：``min=-126`` 防向下溢出，``max=126`` 防 ``amax=1e-40``
      时 ``-exp=+132`` 向上溢出；
    - forward 的 bias 必须一起转 fp16，否则 F.linear 内部 matmul 报 dtype 不匹配。
    """
    from bitsandbytes.autograd import _functions as bnb_autograd

    original = bnb_autograd.MatMul4Bit

    class Dir1MatMul4Bit(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, B, out=None, bias=None, quant_state=None):
            ctx.is_empty = A.numel() == 0
            ctx.state = quant_state
            ctx.dtype_A = A.dtype
            ctx.bias = bias
            ctx.tensors = (None, B)
            if ctx.is_empty:
                shape = quant_state.shape
                return torch.empty(
                    A.shape[:-1] + shape[1:], dtype=A.dtype, device=A.device
                )
            w = dequant_fp16(B, quant_state)
            output = torch.nn.functional.linear(
                A.to(torch.float16),
                w.t(),
                None if bias is None else bias.to(torch.float16),
            )
            return output.to(ctx.dtype_A)

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.is_empty:
                return torch.zeros_like(ctx.tensors[1]), None, None, None, None
            _, B = ctx.tensors
            w = dequant_fp16(B, ctx.state)
            g = grad_output
            exp = torch.frexp(g.abs().max())[1]
            shift = torch.clamp(-exp, min=-126, max=126)
            g16 = torch.ldexp(g.float(), shift).to(torch.float16)
            out = torch.matmul(g16, w.t())
            grad_A = torch.ldexp(out.float(), -shift).to(ctx.dtype_A)
            grad_bias = None
            if ctx.bias is not None and ctx.needs_input_grad[3]:
                grad_bias = grad_output.sum(0, dtype=ctx.bias.dtype)
            return grad_A, None, None, grad_bias, None

    bnb_autograd.MatMul4Bit = Dir1MatMul4Bit
    return original


def restore(original) -> None:
    from bitsandbytes.autograd import _functions as bnb_autograd

    bnb_autograd.MatMul4Bit = original


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--lora-dim", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument(
        "--out-dir", type=Path, default=ROOT / "output" / "tests" / "krea2_nf4_fp16_grad"
    )
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / (
        f"dir1_e2e_{args.img}{('_' + args.tag) if args.tag else ''}.json"
    )

    print(f"=== GPU: {torch.cuda.get_device_name(0)} | img={args.img} | repeat={args.repeat} ===")
    blob = prepare_inputs(args.img, args.out_dir / "cache")
    dit = load_krea2_dit(KREA2_NF4, device=str(CUDA), dtype=torch.bfloat16, eval=False)
    dit.train()
    dit.enable_gradient_checkpointing()

    torch.manual_seed(args.seed)
    net = build_lora(dit, args.lora_dim, args.lora_dim * 1.0).to(CUDA)
    init_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    def one(dtype: torch.dtype) -> tuple[float, torch.Tensor]:
        t0 = time.perf_counter()
        r = run_trajectory(
            dit, net, blob, dtype=dtype, steps=1, lr=0.0,
            init_state=init_state, loss_scale=1.0, device=CUDA,
        )
        torch.cuda.synchronize()
        return time.perf_counter() - t0, r["grads"][0]

    times = {"bf16": [], "dir1": []}
    grads = {}
    # 预热：两种路径各跑一次，排除首次编译/显存分配的影响
    one(torch.bfloat16)
    orig = install_dir1()
    try:
        one(torch.bfloat16)
    finally:
        restore(orig)
    torch.cuda.empty_cache()

    for r in range(args.repeat):
        t_bf, g_bf = one(torch.bfloat16)
        times["bf16"].append(t_bf)
        grads["bf16"] = g_bf

        orig = install_dir1()
        try:
            t_d1, g_d1 = one(torch.bfloat16)
        finally:
            restore(orig)
        times["dir1"].append(t_d1)
        grads["dir1"] = g_d1
        print(
            f"  round {r + 1}: bf16 {t_bf:.2f}s  dir1 {t_d1:.2f}s  "
            f"speedup {t_bf / t_d1:.2f}x"
        )
        torch.cuda.empty_cache()

    med = {k: statistics.median(v) for k, v in times.items()}
    print(f"\n=== 端到端中位数（{args.repeat} 轮交替）===")
    for k in ("bf16", "dir1"):
        print(f"  {k:<6} {med[k]:.3f}s  (all: {[round(x, 2) for x in times[k]]})")
    print(f"  加速比 {med['bf16'] / med['dir1']:.2f}x")

    gb, gd = grads["bf16"], grads["dir1"]
    rel = ((gd - gb).norm() / gb.norm()).item()
    cos = torch.nn.functional.cosine_similarity(
        gd.float().flatten(), gb.float().flatten(), dim=0
    ).item()
    print(f"\n  dir1 vs bf16 梯度 rel-L2={rel * 100:.3f}%  cos={cos:.5f}")

    payload = {
        "probe": "bench/krea2_nf4_fp16_grad/dir1_e2e.py",
        "date": time.strftime("%Y-%m-%d"),
        "img": args.img,
        "lora_dim": args.lora_dim,
        "repeat": args.repeat,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(0),
        "note": "交替运行取中位数；dir1 含 qs.dtype=fp16 dequant 优化",
        "times_s": times,
        "median_s": med,
        "speedup": med["bf16"] / med["dir1"],
        "grad_rel_l2_vs_bf16": rel,
        "grad_cos_vs_bf16": cos,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float))
    print(f"\n[out] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
