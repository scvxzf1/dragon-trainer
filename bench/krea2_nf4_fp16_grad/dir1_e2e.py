"""dir1 端到端交替基准（bf16 vs dir1，中位数）。

单次测量的端到端耗时受运行顺序、显存碎片、预热影响极大（早期一次测量得到
9.0s -> 3.1s，但同一脚本内 fp32 反而比 bf16 快，说明顺序效应明显）。本脚本用
**交替运行**消除顺序偏差，并取中位数。

dir1 的实现已收敛到 ``library.quantization.nf4``（单一真源），本脚本只负责计时。
启用 ``set_dir1_enabled(True)`` 后，``BF16AutocastLinear4bit.forward`` 会把 GEMM
切到 fp16 + 逐层 ``frexp``/``ldexp`` 缩放；``compute_dtype`` 仍是 bfloat16，
BF16 fail-closed 契约不变。

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trajectory_gate import (
    KREA2_NF4,
    build_lora,
    load_krea2_dit,
    prepare_inputs,
    run_trajectory,
)

from library.quantization.nf4 import set_dir1_enabled

CUDA = torch.device("cuda")


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

    print(
        f"=== GPU: {torch.cuda.get_device_name(0)} | img={args.img} | repeat={args.repeat} ==="
    )
    blob = prepare_inputs(args.img, args.out_dir / "cache")
    dit = load_krea2_dit(KREA2_NF4, device=str(CUDA), dtype=torch.bfloat16, eval=False)
    dit.train()
    dit.enable_gradient_checkpointing()

    torch.manual_seed(args.seed)
    net = build_lora(dit, args.lora_dim, args.lora_dim * 1.0).to(CUDA)
    init_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    def one() -> tuple[float, torch.Tensor]:
        t0 = time.perf_counter()
        r = run_trajectory(
            dit,
            net,
            blob,
            dtype=torch.bfloat16,
            steps=1,
            lr=0.0,
            init_state=init_state,
            loss_scale=1.0,
            device=CUDA,
        )
        torch.cuda.synchronize()
        return time.perf_counter() - t0, r["grads"][0]

    times = {"bf16": [], "dir1": []}
    grads = {}
    # 预热：两种路径各跑一次，排除首次编译/显存分配的影响
    one()
    set_dir1_enabled(True)
    try:
        one()
    finally:
        set_dir1_enabled(False)
    torch.cuda.empty_cache()

    for r in range(args.repeat):
        t_bf, g_bf = one()
        times["bf16"].append(t_bf)
        grads["bf16"] = g_bf

        set_dir1_enabled(True)
        try:
            t_d1, g_d1 = one()
        finally:
            set_dir1_enabled(False)
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
        "note": "交替运行取中位数；dir1 走 library.quantization.nf4 的单一实现",
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
