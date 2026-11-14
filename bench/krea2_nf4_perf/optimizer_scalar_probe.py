"""Distinguish AdamW CPU scalar extraction from CUDA synchronization.

No model, data, training queue, or production config is loaded. Each arm uses
392 independent two-element FP32 parameter tensors, not real LoRA shapes.
Counts are diagnostic only: neither timing nor training performance is measured.
Run CUDA only after gpu-gate, under an external ``timeout 60``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode


class ScalarDevices(TorchDispatchMode):
    """Record the input device without reading a tensor value."""

    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            self.counts[str(args[0].device)] += 1
        return func(*args, **(kwargs or {}))


def probe_arm(device: torch.device, name: str, options: dict) -> dict:
    params = [torch.nn.Parameter(torch.ones(2, device=device)) for _ in range(392)]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0, **options)
    for param in params:
        param.grad = torch.ones_like(param)
    opt.step()  # initialize state outside the measurement
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    # This is a CPU operator event counter, not a CUDA time profiler.
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    counts = {row.key: row.count for row in prof.key_averages()}
    # Separate step: the dispatch observer must not contaminate event counts.
    observer = ScalarDevices()
    with observer:
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    finite = bool(torch.stack([p.detach() for p in params]).isfinite().all().item())
    assert finite, f"nonfinite tiny optimizer output: {name}"
    observed = dict(observer.counts)
    scalar_events = counts.get("aten::_local_scalar_dense", 0)
    assert scalar_events == sum(observed.values()), (name, scalar_events, observed)
    return {
        "arm": name,
        "optimizer_kwargs": options,
        "parameter_tensors": len(params),
        "elements_per_tensor": 2,
        "dtype": str(params[0].dtype),
        "step_state_devices": sorted({str(opt.state[p]["step"].device) for p in params}),
        "profile_scalar_events": scalar_events,
        "separate_dispatch_scalar_devices": observed,
        "finite_after_three_updates": finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path (never overwrite evidence)")
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA unavailable; no CPU fallback")
        torch.cuda.reset_peak_memory_stats(device)
    arms = [("default", {}), ("foreach", {"foreach": True})]
    arms.append(("fused", {"fused": True}) if device.type == "cuda"
                else ("single_tensor", {"foreach": False}))
    rows = [probe_arm(device, name, options) for name, options in arms]
    result = {
        "probe": "optimizer scalar-device audit, NOT training performance",
        "torch_version": torch.__version__,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "rows": rows,
        "peak_allocated_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                               if device.type == "cuda" else None),
        "elapsed_seconds_excluding_import": time.perf_counter() - started,
        "limitations": [
            "No Krea-2 model, attention, NF4, or real parameter shapes exercised.",
            "No CUPTI timeline; counts do not measure GPU busy time or speedup.",
            "Historical 786 events lack device provenance; this is a counterexample, not a recovered trace.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
