"""用**真实 6144×6144 NF4 层**定位 stage6 指纹（rel-L2 35.65% / cos 0.9434）对应的 dY 量级。

stage6 的 35.65% 是单层 Linear 探针（非整模型），但当年没有记录 `grad_output` 的
量级。本脚本用模型里真实的 `blocks.0.attn.wq`（6144×6144，NF4 反量化权重），
对 dY 做对数网格扫描，找出哪个 rms 复现该指纹；再对照真实模型各层实测 dY 量级
（`capture_real.py` 输出），判断该量级在真实训练里是否出现。

为什么必须是真 GPU GEMM：V100 上 fp16 走真 tensor core（fp32 累加 + fp16 输出
舍入），纯 CPU 仿真会漏掉输出舍入这一层。边界：V100 是 sm70，bf16 走 PyTorch
的 fp32 上采样仿真，不是 sm86 cuBLAS 复现。

用法::

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/layer_repro.py \
      --out output/tests/krea2_nf4_fp16_grad/layer_repro.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2FP16_GPU", "1"))

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.models.krea2_raw.weights import load_krea2_dit

KREA2_NF4 = ROOT / "models" / "diffusion_models" / "krea2_raw_nf4_self_contained.safetensors"
FP16_MIN_NORMAL = 6.103515625e-05
FP16_MIN_SUBNORMAL = 5.960464477539063e-08
TARGET_REL = 0.3565
TARGET_COS = 0.9434
CUDA = torch.device("cuda")


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float((a - b).norm() / b.norm())


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def frac_below(t: torch.Tensor, thr: float) -> float:
    x = t.float().abs()
    return float((x < thr).float().mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=str, default="blocks.0.attn.wq")
    ap.add_argument("--tokens", type=int, default=1024, help="dY 的行数（序列长度）")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    device = CUDA
    out_path = args.out or (
        ROOT / "output" / "tests" / "krea2_nf4_fp16_grad" / "layer_repro.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== {torch.cuda.get_device_name(0)} | layer={args.layer} ===")
    dit = load_krea2_dit(KREA2_NF4, device=str(device), dtype=torch.bfloat16, eval=True)
    mod = dict(dit.named_modules())[args.layer]

    import bitsandbytes.functional as bnbF

    w32 = bnbF.dequantize_4bit(mod.weight.data, mod.weight.quant_state).to(torch.float32)
    w16 = w32.half()
    print(f"  W: {tuple(w32.shape)} |W|={w32.norm():.4e} rms={w32.pow(2).mean().sqrt():.4e}")

    # 对数网格：覆盖 fp16 正规区下沿到彻底冲零
    grid = [10 ** (-8 + 0.25 * i) for i in range(17)]
    torch.manual_seed(20261108)
    base = torch.randn(args.tokens, mod.out_features, device=device, dtype=torch.float32)

    rows = []
    t0 = time.time()
    for rms in grid:
        dy = base * rms
        ref = torch.matmul(dy, w32)
        b16 = torch.matmul(dy.bfloat16().float(), w32.bfloat16().float())
        f16 = torch.matmul(dy.half(), w16).float()
        rows.append(
            {
                "dY_rms": rms,
                "dY_max_abs": float(dy.abs().max()),
                "frac_lt_fp16_min_normal": frac_below(dy, FP16_MIN_NORMAL),
                "frac_lt_fp16_min_subnormal": frac_below(dy, FP16_MIN_SUBNORMAL),
                "fp16_rel_l2": rel_l2(f16, ref),
                "fp16_cos": cos(f16, ref),
                "fp16_norm_ratio": float(f16.double().norm() / ref.double().norm()),
                "bf16_rel_l2": rel_l2(b16, ref),
                "bf16_cos": cos(b16, ref),
            }
        )
        print(
            f"  rms={rms:.2e} <2^-14={rows[-1]['frac_lt_fp16_min_normal'] * 100:5.1f}% "
            f"fp16 rel={rows[-1]['fp16_rel_l2']:.4f} cos={rows[-1]['fp16_cos']:.4f} | "
            f"bf16 rel={rows[-1]['bf16_rel_l2']:.2e}"
        )
    secs = time.time() - t0

    best = min(
        rows,
        key=lambda r: abs(r["fp16_rel_l2"] - TARGET_REL) + abs(r["fp16_cos"] - TARGET_COS),
    )
    payload = {
        "probe": "bench/krea2_nf4_fp16_grad/layer_repro.py",
        "date": time.strftime("%Y-%m-%d"),
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "layer": args.layer,
        "weight_shape": list(w32.shape),
        "weight_rms": float(w32.pow(2).mean().sqrt()),
        "dY_tokens": args.tokens,
        "target": {"rel_l2": TARGET_REL, "cos": TARGET_COS},
        "best_match": best,
        "grid": rows,
        "elapsed_s": secs,
        "note": "dY = N(0,1)*rms 合成；真实训练 dY 由整模型反向决定。",
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"\n[out] {out_path}")
    print(
        f"=== 指纹最优点: dY rms={best['dY_rms']:.3e} → rel={best['fp16_rel_l2']:.4f} "
        f"cos={best['fp16_cos']:.4f} (目标 {TARGET_REL}/{TARGET_COS}) ==="
    )
    print(
        f"    该点 <2^-14 占比 {best['frac_lt_fp16_min_normal'] * 100:.1f}%, "
        f"<2^-24 占比 {best['frac_lt_fp16_min_subnormal'] * 100:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
