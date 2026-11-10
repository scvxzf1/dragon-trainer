"""用 nsys 抓取恰好一个训练 step，供时间线分析。

存在的理由：torch.profiler 汇总出的 ``self_device_time_total`` 在 krea2 NF4 +
内层 fp16 路径上是 4,923 ms/step，而同一脚本实测墙钟只有 2,560 ms/step
（192%）。单流不可能超过 100%，所以要么有多流并发，要么该汇总口径重复计算。

本脚本不做任何自己的计时推断，只负责：
  1. 把一步训练精确夹进 ``torch.cuda.profiler.start()/stop()``
  2. 记录这一步的墙钟
  3. 让 nsys 用真实时间线算 union 忙时

对比 ``nsys stats --report cuda_gpu_kern_sum`` 的合计与这里打印的墙钟，
就能判断 4,923 ms 那个数是重复计算还是确实有多流。

用法::

    nsys-v100-torch profile --trace=cuda,nvtx \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --output=/tmp/prof/step --force-overwrite=true \
      .venv/bin/python bench/krea2_nf4_fp16_grad/nsys_step.py --img 256
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2FP16_GPU", "1"))

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trajectory_gate import (  # noqa: E402
    KREA2_NF4,
    build_lora,
    load_krea2_dit,
    prepare_inputs,
    run_trajectory,
)

from library.quantization.nf4 import set_dir1_enabled  # noqa: E402

CUDA = torch.device("cuda")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--lora-dim", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-dir1", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "tests" / "krea2_nf4_fp16_grad",
    )
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    set_dir1_enabled(not args.no_dir1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    blob = prepare_inputs(args.img, args.out_dir / "cache")
    dit = load_krea2_dit(KREA2_NF4, device=str(CUDA), dtype=torch.bfloat16, eval=False)
    dit.train()
    dit.enable_gradient_checkpointing()

    torch.manual_seed(args.seed)
    net = build_lora(dit, args.lora_dim, args.lora_dim * 1.0).to(CUDA)
    init = {k: v.detach().clone() for k, v in net.state_dict().items()}

    def step() -> None:
        run_trajectory(
            dit,
            net,
            blob,
            dtype=torch.bfloat16,
            steps=1,
            lr=1e-4,
            init_state=init,
            device=CUDA,
        )

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    t0 = time.perf_counter()
    step()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    torch.cuda.profiler.stop()

    print(f"\n[nsys_step] arm={'bf16' if args.no_dir1 else 'dir1'} img={args.img}")
    print(f"[nsys_step] wall_clock_ms = {wall_ms:.1f}")
    print("[nsys_step] 与 cuda_gpu_kern_sum 的合计对比：")
    print("[nsys_step]   接近 wall → 之前 4,923ms 是重复计算")
    print("[nsys_step]   明显小于 wall → 存在空泡（GPU 没排满）")
    print("[nsys_step]   大于 wall → 有多流并发")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
