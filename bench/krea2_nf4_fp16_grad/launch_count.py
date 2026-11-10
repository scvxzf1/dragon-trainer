"""统计 krea2 NF4 + fp16 内层路径的 kernel launch 次数。

回答两个问题：
  1. 单个训练 step 有多少次 CUDA kernel launch
  2. 这些 launch 按 kernel 如何分布（谁贡献了最多的 launch）

launch 次数是 CPU 侧开销的直接指标。当 step 时间被大量小 kernel 主导时，
即使每个 kernel 很快，CPU 的 launch 分发也会成为瓶颈——这与
docs/findings/krea2_nf4_fp16_inner_path_20261109.md 里 elementwise 占
2.065 s / 11.06 s 的观察一致，本脚本量化这一层的规模。

用 torch.profiler 采集（不是 nsys）：我们只关心 launch **计数**，不需要
时间线或硬件计数器，profiler 更轻且不涉及 CUPTI 单例问题。

用法::

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/launch_count.py --img 256 --steps 2

    # 对比 bf16（不开内层 fp16）
    ... --no-dir1
"""

import argparse
import json
import os
import sys
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
    ap.add_argument("--steps", type=int, default=2, help="profile 窗口内的步数")
    ap.add_argument("--warmup", type=int, default=1, help="profile 前的预热步数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-dir1", action="store_true", help="关闭内层 fp16（bf16 基线）")
    ap.add_argument("--top", type=int, default=15, help="列出前 N 个 kernel")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "tests" / "krea2_nf4_fp16_grad",
    )
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    set_dir1_enabled(not args.no_dir1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arm = "bf16" if args.no_dir1 else "dir1"
    out_path = args.out_dir / f"launch_count_{args.img}_{arm}.json"

    print(
        f"=== GPU: {torch.cuda.get_device_name(0)} | img={args.img} | "
        f"arm={arm} | steps={args.steps} | warmup={args.warmup} ==="
    )

    blob = prepare_inputs(args.img, args.out_dir / "cache")
    dit = load_krea2_dit(KREA2_NF4, device=str(CUDA), dtype=torch.bfloat16, eval=False)
    dit.train()
    dit.enable_gradient_checkpointing()

    torch.manual_seed(args.seed)
    net = build_lora(dit, args.lora_dim, args.lora_dim * 1.0).to(CUDA)
    init_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    def step() -> None:
        run_trajectory(
            dit,
            net,
            blob,
            dtype=torch.bfloat16,
            steps=1,
            lr=1e-4,
            init_state=init_state,
            device=CUDA,
        )

    # 预热：把 lazy 初始化、cuDNN autotune、显存池都跑热，避免混进统计
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()

    # --- CPU 侧：显式的 cudaLaunchKernel 调用次数 ---
    launch_calls = 0
    for evt in prof.events():
        if evt.name == "cudaLaunchKernel":
            launch_calls += int(getattr(evt, "count", 1) or 1)
    # events() 对同名事件会聚合，count 已含次数；保险起见用 key_averages 复核
    launch_from_avg = 0
    for row in prof.key_averages():
        if row.key == "cudaLaunchKernel":
            launch_from_avg = int(row.count)

    # --- GPU 侧：实际执行的 kernel 实例数 ---
    kernel_rows = []
    for row in prof.key_averages():
        # 带 device 时间的行才是 GPU kernel（CPU 算子行没有 cuda_time）
        cuda_total = getattr(row, "self_device_time_total", None)
        if cuda_total is None:
            cuda_total = getattr(row, "self_cuda_time_total", None)
        if not cuda_total:
            continue
        kernel_rows.append(
            {
                "kernel": row.key,
                "count": int(row.count),
                "self_cuda_time_ms": round(cuda_total / 1000.0, 3),
            }
        )

    kernel_rows.sort(key=lambda r: (-r["count"], -r["self_cuda_time_ms"]))
    total_kernel_exec = sum(r["count"] for r in kernel_rows)
    total_kernel_time_ms = round(sum(r["self_cuda_time_ms"] for r in kernel_rows), 3)

    report = {
        "arm": arm,
        "img": args.img,
        "steps_profiled": args.steps,
        "warmup_steps": args.warmup,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_launch_calls_total": max(launch_calls, launch_from_avg),
        "cuda_launch_calls_per_step": round(
            max(launch_calls, launch_from_avg) / max(args.steps, 1), 1
        ),
        "kernel_exec_total": total_kernel_exec,
        "kernel_exec_per_step": round(total_kernel_exec / max(args.steps, 1), 1),
        "distinct_kernels": len(kernel_rows),
        "kernel_gpu_time_ms_total": total_kernel_time_ms,
        "kernel_gpu_time_ms_per_step": round(total_kernel_time_ms / max(args.steps, 1), 2),
        "top_kernels": kernel_rows[: args.top],
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"CUDA launch (CPU 侧 cudaLaunchKernel 调用): {report['cuda_launch_calls_total']}")
    print(f"  每 step: {report['cuda_launch_calls_per_step']}")
    print(f"GPU kernel 实例数: {total_kernel_exec}")
    print(f"  每 step: {report['kernel_exec_per_step']}")
    print(f"不同 kernel 种类: {len(kernel_rows)}")
    print(f"kernel GPU 时间合计: {total_kernel_time_ms} ms "
          f"({report['kernel_gpu_time_ms_per_step']} ms/step)")
    print()
    print(f"{'count':>10}  {'ms/step':>10}  kernel")
    for row in kernel_rows[: args.top]:
        print(
            f"{row['count']:>10}  "
            f"{row['self_cuda_time_ms'] / max(args.steps, 1):>10.2f}  {row['kernel'][:78]}"
        )
    print()
    print(f"写出: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
