"""Measure achieved matmul throughput per precision (compute-lock probe).

CMP mining cards ship with a compute restriction: certain precisions (in
practice the FP16/BF16 tensor-core path) run far below the card's silicon
capability until the card is unlocked. Reading `nvidia-smi` cannot tell you
whether that restriction is active -- only a measured throughput can. This probe
runs square matmuls in fp32 / fp16 / bf16 and compares the achieved TFLOP/s with
the theoretical rate derived from the visible SM count and the current SM clock.

How to read the result
----------------------
* ``fp16 / fp32 ratio`` ~= 8 (Turing, FP16 accumulate) or ~4 (FP32 accumulate)
  means the tensor-core path is running normally.
* ratio ~<= 2 means FP16 is only using the CUDA-core path (packed HFMA2) or is
  rate-limited: the compute lock is active.

Run it on an idle GPU; a co-running job scales every number down and makes the
fp32/fp16 ratio noisy (it is still a usable first signal, not a verdict).

Usage::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        .venv/bin/python -m bench.gpu_precision_probe --gpu 0 --size 4096
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess

import torch


def _sm_clock_mhz() -> float | None:
    """Current SM clock from nvidia-smi (torch exposes no clock API)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception:
        return None
    try:
        return float(out.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def _bench_matmul(dtype: torch.dtype, size: int, iters: int, device: torch.device) -> dict:
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        a @ b
        end.record()
        torch.cuda.synchronize(device)
        samples.append(start.elapsed_time(end))
    _ = a @ b  # keep the tensors alive until here
    flops = 2.0 * size**3
    seconds = statistics.median(samples) / 1000.0
    return {
        "dtype": str(dtype).replace("torch.", ""),
        "size": size,
        "median_ms": round(statistics.median(samples), 3),
        "tflops": round(flops / seconds / 1e12, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="visible CUDA device index")
    parser.add_argument("--size", type=int, default=4096, help="square matmul edge")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    clock_mhz = _sm_clock_mhz()

    rows = []
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        rows.append(_bench_matmul(torch.float32, args.size, args.iters, device))
        torch.backends.cuda.matmul.allow_tf32 = True
        rows.append(_bench_matmul(torch.float32, args.size, args.iters, device))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
    rows[-1]["dtype"] = "fp32(tf32-allowed)"
    rows.append(_bench_matmul(torch.float16, args.size, args.iters, device))
    rows.append(_bench_matmul(torch.bfloat16, args.size, args.iters, device))

    by_name = {row["dtype"]: row["tflops"] for row in rows}
    fp32 = by_name.get("float32") or 0.0
    fp16 = by_name.get("float16") or 0.0
    ratio = round(fp16 / fp32, 2) if fp32 else None

    theoretical = {}
    if clock_mhz:
        # Turing: 64 FP32 cores/SM (2 FLOP/FMA) and 512 FP16 FMA/cycle/SM in the
        # tensor cores.
        theoretical = {
            "fp32_tflops": round(props.multi_processor_count * 128 * clock_mhz * 1e6 / 1e12, 2),
            "fp16_tensor_tflops": round(props.multi_processor_count * 1024 * clock_mhz * 1e6 / 1e12, 2),
        }

    verdict = "unknown"
    if ratio is not None:
        if ratio >= 3.0:
            verdict = "tensor path active (no compute lock detected)"
        elif ratio <= 2.0:
            verdict = "fp16 ~ fp32 rate: compute lock ACTIVE (or tensor cores unavailable)"
        else:
            verdict = "ambiguous: rerun on an idle GPU"

    report = {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "sm_clock_mhz": clock_mhz,
        "rows": rows,
        "fp16_over_fp32_ratio": ratio,
        "theoretical_theoretical_note": "SM count x clock; Turing constants",
        "theoretical": theoretical,
        "verdict": verdict,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[precision] {verdict} (fp16/fp32={ratio})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
