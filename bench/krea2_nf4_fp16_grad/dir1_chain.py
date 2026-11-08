"""真实权重下的 dir1（逐层 frexp 缩放）验证。

合成矩阵模拟（``sim_layer_scaling.py``）只能给定性结论，因为真实 Krea-2 权重的
条件数与高斯/正交矩阵不同。本脚本用**真实 NF4 权重 + 真实 TE/VAE 缓存**跑一次
完整单步反传，直接对比四条路径的 LoRA 参数梯度（fp32 为真值裁判）：

- ``bf16``：原生 bf16 compute（当前生产路径）
- ``fp16``：裸 fp16 compute（对照，预期深层崩溃）
- ``dir1``：外层 bf16 不变，monkey-patch ``MatMul4Bit`` 让 NF4 Linear 的
  forward/backward 都走 fp16 GEMM；backward 用 ``frexp``/``ldexp`` 做 2 的整数次幂
  平移，消除 fp16 subnormal

判据不是「dir1 比 bf16 准」（模拟已证明做不到），而是：

1. dir1 与 bf16 的差距是否落在 bf16 自身的 fp32 偏差量级内；
2. dir1 是否显著优于裸 fp16；
3. 全程无非有限梯度。

生产 ``compute_dtype`` 保持 bfloat16、22 处契约不改；本脚本只在进程内
monkey-patch bnb 的 autograd Function。

用法::

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/dir1_chain.py --img 256
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2FP16_GPU", "1"))

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trajectory_gate import (
    KREA2_NF4,
    PROMPT,
    build_lora,
    load_krea2_dit,
    prepare_inputs,
    run_trajectory,
)

ROOT = Path(__file__).resolve().parents[2]
CUDA = torch.device("cuda")


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    denom = b.norm()
    return ((a - b).norm() / denom).item() if denom > 0 else float("nan")


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def install_dir1():
    """把 bnb 的 MatMul4Bit 换成 dir1 版本，返回原类以便恢复。

    三个实现要点都是踩过的坑：

    - ``torch.ldexp`` 第二参数必须是 Tensor，不能传 Python int；
    - 平移量必须双向 clamp：``min=-126`` 防向下溢出，``max=126`` 防
      ``amax=1e-40 -> exp=-132 -> -exp=+132`` 向上溢出；
    - ``frexp(0.0) == (0.0, 0)``，全零梯度天然免疫，不需要 eps/where 分支。
    """
    import bitsandbytes.functional as bnb_f
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
            w = bnb_f.dequantize_4bit(B, quant_state)
            # 前向不需要缩放：激活量级 O(1e-2~1e0)，本来就在 fp16 正规区。
            # bias 必须一起转 fp16，否则 F.linear 内部 matmul 报 dtype 不匹配。
            output = torch.nn.functional.linear(
                A.to(torch.float16),
                w.to(torch.float16).t(),
                None if bias is None else bias.to(torch.float16),
            )
            return output.to(ctx.dtype_A)

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.is_empty:
                return torch.zeros_like(ctx.tensors[1]), None, None, None, None
            _, B = ctx.tensors
            w = bnb_f.dequantize_4bit(B, ctx.state)
            g = grad_output
            exp = torch.frexp(g.abs().max())[1]
            shift = torch.clamp(-exp, min=-126, max=126)
            g16 = torch.ldexp(g.float(), shift).to(torch.float16)
            out = torch.matmul(g16, w.to(torch.float16).t())
            grad_A = torch.ldexp(out.float(), -shift).to(ctx.dtype_A)

            grad_bias = None
            if ctx.bias is not None and ctx.needs_input_grad[3]:
                grad_bias = grad_output.sum(0, dtype=ctx.bias.dtype)
            return grad_A, None, None, grad_bias, None

    bnb_autograd.MatMul4Bit = Dir1MatMul4Bit
    return original


def restore_matmul4bit(original) -> None:
    from bitsandbytes.autograd import _functions as bnb_autograd

    bnb_autograd.MatMul4Bit = original


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--lora-dim", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument(
        "--out-dir", type=Path, default=ROOT / "output" / "tests" / "krea2_nf4_fp16_grad"
    )
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / (
        f"dir1_chain_{args.img}{('_' + args.tag) if args.tag else ''}.json"
    )

    print(f"=== GPU: {torch.cuda.get_device_name(0)} | img={args.img} ===")
    blob = prepare_inputs(args.img, args.out_dir / "cache")
    dit = load_krea2_dit(KREA2_NF4, device=str(CUDA), dtype=torch.bfloat16, eval=False)
    dit.train()
    dit.enable_gradient_checkpointing()

    torch.manual_seed(args.seed)
    net = build_lora(dit, args.lora_dim, args.lora_alpha).to(CUDA)
    init_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"LoRA 可训参数 {n_train / 1e6:.2f}M")

    def one(dtype: torch.dtype, label: str) -> dict:
        t0 = time.time()
        r = run_trajectory(
            dit,
            net,
            blob,
            dtype=dtype,
            steps=1,
            lr=0.0,
            init_state=init_state,
            loss_scale=1.0,
            device=CUDA,
        )
        g = r["grads"][0]
        print(
            f"[{label}] loss={r['losses'][0]:.6f} "
            f"grad_flat={g.numel()} 非有限={bool(not torch.isfinite(g).all())} "
            f"({time.time() - t0:.1f}s)"
        )
        return {"loss": r["losses"][0], "grad": g, "nonfinite": bool(not torch.isfinite(g).all())}

    results = {}
    results["fp32"] = one(torch.float32, "fp32")
    torch.cuda.empty_cache()
    results["bf16"] = one(torch.bfloat16, "bf16")
    torch.cuda.empty_cache()
    results["fp16"] = one(torch.float16, "fp16")
    torch.cuda.empty_cache()

    original = install_dir1()
    try:
        results["dir1"] = one(torch.bfloat16, "dir1")
    finally:
        restore_matmul4bit(original)
    torch.cuda.empty_cache()

    ref = results["fp32"]["grad"]
    summary = {}
    print("\n=== LoRA 梯度对比（fp32 为真值裁判）===")
    print(f"{'臂':<10} {'loss':>12} {'rel-L2 vs fp32':>16} {'cos':>10} {'非有限':>8}")
    print("-" * 62)
    for name in ("bf16", "fp16", "dir1"):
        g = results[name]["grad"]
        summary[name] = {
            "loss": results[name]["loss"],
            "grad_rel_l2_vs_fp32": _rel_l2(g, ref),
            "grad_cos_vs_fp32": _cos(g, ref),
            "nonfinite": results[name]["nonfinite"],
        }
        s = summary[name]
        print(
            f"{name:<10} {s['loss']:>12.6f} {s['grad_rel_l2_vs_fp32'] * 100:>15.3f}% "
            f"{s['grad_cos_vs_fp32']:>10.5f} {s['nonfinite']!s:>8}"
        )

    summary["dir1_vs_bf16"] = {
        "grad_rel_l2": _rel_l2(results["dir1"]["grad"], results["bf16"]["grad"]),
        "grad_cos": _cos(results["dir1"]["grad"], results["bf16"]["grad"]),
    }
    print(
        f"\ndir1 vs bf16: rel-L2 = {summary['dir1_vs_bf16']['grad_rel_l2'] * 100:.3f}% "
        f"cos = {summary['dir1_vs_bf16']['grad_cos']:.5f}"
    )

    payload = {
        "probe": "bench/krea2_nf4_fp16_grad/dir1_chain.py",
        "date": time.strftime("%Y-%m-%d"),
        "img": args.img,
        "lora_dim": args.lora_dim,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
        "weights": KREA2_NF4.name,
        "gpu": torch.cuda.get_device_name(0),
        "prompt": PROMPT,
        "note": "生产 compute_dtype 保持 bf16；dir1 仅在进程内 monkey-patch MatMul4Bit",
        "loss": {k: v["loss"] for k, v in results.items()},
        "summary": summary,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float))
    print(f"\n[out] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
