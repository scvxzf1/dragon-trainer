"""Short CUDA fp16 seam probe for the ``codex/fp16-nan-guards`` branch.

This probe deliberately exercises only small synthetic tensors.  It is not a
training benchmark and does not load a model or touch user data.  It covers:

* H1: Accelerate backward -> unscale/clip -> step versus the direct CUDA
  GradScaler seam, including bitwise parameter/gradient comparison.
* H2: CUDA fp16 gradient unscale representation-window samples.
* H8: CUDA GradScaler state_dict restore followed by an identical continuation.

Run one process per physical GPU, for example::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      .venv/bin/python bench/fp16_precision/run_cuda_probe.py --gpu-label gpu0

The process sees its selected physical card as ``cuda:0``.  Keep separate
output files when running both cards concurrently.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from library.runtime.accelerator import patch_accelerator_for_fp16_training


SCALE = 65536.0


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
        return False
    return torch.equal(left.detach().contiguous().view(torch.uint8), right.detach().contiguous().view(torch.uint8))


def _patch_scaler_allow_fp16(scaler: Any) -> None:
    original = scaler._unscale_grads_

    def replacement(optimizer, inv_scale, found_inf, allow_fp16):
        return original(optimizer, inv_scale, found_inf, True)

    scaler._unscale_grads_ = replacement


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _finish() -> None:
    torch.cuda.synchronize()


def _h1_accelerate(device: torch.device, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = torch.nn.Linear(8, 4, bias=True, device=device, dtype=torch.float16)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    inputs = torch.randn(3, 8, device=device, dtype=torch.float16)
    target = torch.randn(3, 4, device=device, dtype=torch.float16)

    accelerator = Accelerator(device_placement=False, mixed_precision="fp16")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = accelerator.prepare(optimizer)
    patch_accelerator_for_fp16_training(accelerator)

    with accelerator.autocast():
        loss = (model(inputs).float() - target.float()).square().mean()
    accelerator.backward(loss)
    grads_before = [parameter.grad.detach().clone() for parameter in model.parameters()]
    accelerator.clip_grad_norm_(model.parameters(), 1.0)
    grads_after = [parameter.grad.detach().clone() for parameter in model.parameters()]
    accelerator_scale_before = float(accelerator.scaler.get_scale())
    optimizer.step()
    accelerator_scale_after = float(accelerator.scaler.get_scale())
    _finish()

    return {
        "loss": float(loss.detach().cpu()),
        "scale_before": accelerator_scale_before,
        "scale_after": accelerator_scale_after,
        "step_was_skipped": bool(optimizer.step_was_skipped),
        "grads_before": grads_before,
        "grads_after": grads_after,
        "params_after": [parameter.detach().clone() for parameter in model.parameters()],
        "initial": initial,
    }


def _h1_direct(device: torch.device, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = torch.nn.Linear(8, 4, bias=True, device=device, dtype=torch.float16)
    inputs = torch.randn(3, 8, device=device, dtype=torch.float16)
    target = torch.randn(3, 4, device=device, dtype=torch.float16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", init_scale=SCALE)
    _patch_scaler_allow_fp16(scaler)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = (model(inputs).float() - target.float()).square().mean()
    scaler.scale(loss).backward()
    grads_before = [parameter.grad.detach().clone() for parameter in model.parameters()]
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    grads_after = [parameter.grad.detach().clone() for parameter in model.parameters()]
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    _finish()

    return {
        "loss": float(loss.detach().cpu()),
        "scale_before": scale_before,
        "scale_after": scale_after,
        "step_was_skipped": None,
        "grads_before": grads_before,
        "grads_after": grads_after,
        "params_after": [parameter.detach().clone() for parameter in model.parameters()],
    }


def run_h1(device: torch.device) -> dict[str, Any]:
    accelerate_result = _h1_accelerate(device, seed=20261106)
    direct_result = _h1_direct(device, seed=20261106)
    grad_before_equal = all(
        _bits_equal(left, right)
        for left, right in zip(accelerate_result["grads_before"], direct_result["grads_before"])
    )
    grad_after_equal = all(
        _bits_equal(left, right)
        for left, right in zip(accelerate_result["grads_after"], direct_result["grads_after"])
    )
    params_equal = all(
        _bits_equal(left, right)
        for left, right in zip(accelerate_result["params_after"], direct_result["params_after"])
    )
    params_moved = any(
        not _bits_equal(updated, initial_value)
        for updated, initial_value in zip(
            accelerate_result["params_after"], accelerate_result["initial"].values()
        )
    )
    return {
        "accelerate_loss": accelerate_result["loss"],
        "direct_loss": direct_result["loss"],
        "loss_bitwise_equal": accelerate_result["loss"] == direct_result["loss"],
        "scale_before_equal": accelerate_result["scale_before"] == direct_result["scale_before"],
        "scale_after_equal": accelerate_result["scale_after"] == direct_result["scale_after"],
        "accelerate_step_not_skipped": not accelerate_result["step_was_skipped"],
        "grads_before_bitwise_equal": grad_before_equal,
        "grads_after_unscale_clip_bitwise_equal": grad_after_equal,
        "params_after_bitwise_equal": params_equal,
        "params_moved_from_initial": params_moved,
        "pass": all((
            accelerate_result["loss"] == direct_result["loss"],
            accelerate_result["scale_before"] == direct_result["scale_before"],
            accelerate_result["scale_after"] == direct_result["scale_after"],
            not accelerate_result["step_was_skipped"],
            grad_before_equal,
            grad_after_equal,
            params_equal,
            params_moved,
        )),
    }


def _h2_one(device: torch.device, true_grad: float) -> dict[str, Any]:
    parameter = torch.nn.Parameter(torch.zeros(1, device=device, dtype=torch.float16))
    optimizer = torch.optim.SGD([parameter], lr=0.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=SCALE)
    _patch_scaler_allow_fp16(scaler)
    # ``unscale_`` requires one scaled backward first, so warm the scaler up on a
    # throwaway scalar and leave the probe parameter fully synthetic.
    warmup = torch.zeros((), device=device, dtype=torch.float16, requires_grad=True)
    scaler.scale(warmup).backward()
    parameter.grad = torch.tensor([true_grad * SCALE], device=device, dtype=torch.float16)
    scaled_value = float(parameter.grad.detach().cpu().item())
    scaler.unscale_(optimizer)
    unscaled_value = float(parameter.grad.detach().cpu().item())
    state = scaler._per_optimizer_states[id(optimizer)]
    found_inf = {str(key): float(value.detach().cpu().item()) for key, value in state["found_inf_per_device"].items()}
    _finish()
    return {
        "true_grad_requested": true_grad,
        "scaled_fp16": scaled_value,
        "after_unscale_fp16": unscaled_value,
        "found_inf_per_device": found_inf,
        "flushed_to_zero": unscaled_value == 0.0 and true_grad != 0.0,
    }


def run_h2(device: torch.device) -> dict[str, Any]:
    rows = [_h2_one(device, value) for value in (1e-8, 1e-7, 5.96e-8, 1e-3, 0.5, 1.5)]
    # A row is consistent when the scaler's found_inf flag matches whether the
    # scaled fp16 gradient actually overflowed, and only sub-scale gradients are
    # flushed to zero by the fp16 unscale.
    def _row_ok(row: dict[str, Any]) -> bool:
        scaled = row["scaled_fp16"]
        overflowed = scaled != scaled or abs(scaled) == float("inf")
        found_inf = any(value > 0 for value in row["found_inf_per_device"].values())
        if found_inf != overflowed:
            return False
        if row["flushed_to_zero"] and row["true_grad_requested"] * SCALE > 1e-3:
            return False
        return True

    return {
        "scale": SCALE,
        "fp16_min_subnormal": float(torch.finfo(torch.float16).tiny * 2**-10),
        "rows": rows,
        "pass": all(_row_ok(row) for row in rows),
    }


def _h8_step(scaler: Any, parameter: torch.nn.Parameter, optimizer: Any, overflow: bool) -> tuple[float, float, float]:
    optimizer.zero_grad(set_to_none=True)
    value = float("inf") if overflow else 1.0
    parameter.grad = torch.tensor([value], device=parameter.device, dtype=parameter.dtype)
    before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    after = float(scaler.get_scale())
    return before, after, float(parameter.detach().cpu().item())


def _h8_new(device: torch.device, init_scale: float, growth_interval: int):
    parameter = torch.nn.Parameter(torch.ones(1, device=device, dtype=torch.float16))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=init_scale, growth_interval=growth_interval)
    _patch_scaler_allow_fp16(scaler)
    # Same warm-up requirement as H2: ``step``/``update`` refuse to run until one
    # scaled backward has initialised the internal scale.
    warmup = torch.zeros((), device=device, dtype=torch.float16, requires_grad=True)
    scaler.scale(warmup).backward()
    return scaler, parameter, optimizer


def run_h8(device: torch.device) -> dict[str, Any]:
    scaler_a, parameter_a, optimizer_a = _h8_new(device, 8.0, 2)
    prefix = [_h8_step(scaler_a, parameter_a, optimizer_a, False) for _ in range(2)]
    saved = scaler_a.state_dict()
    saved_scale = float(saved["scale"])
    scaler_b, parameter_b, optimizer_b = _h8_new(device, 1.0, 99)
    scaler_b.load_state_dict(saved)
    restored_scale = float(scaler_b.get_scale())
    continuation_a = [_h8_step(scaler_a, parameter_a, optimizer_a, overflow) for overflow in (False, True, False)]
    continuation_b = [_h8_step(scaler_b, parameter_b, optimizer_b, overflow) for overflow in (False, True, False)]
    _finish()
    trajectory_equal = [row[0:2] for row in continuation_a] == [row[0:2] for row in continuation_b]
    restored_scale_equal = restored_scale == saved_scale and saved_scale != 1.0
    return {
        "prefix": prefix,
        "saved_state_keys": sorted(saved),
        "saved_scale": saved_scale,
        "restored_scale": restored_scale,
        "continuation_a": continuation_a,
        "continuation_b": continuation_b,
        "continuation_scale_trajectory_equal": trajectory_equal,
        "restored_object_picked_up_saved_scale": restored_scale_equal,
        "pass": trajectory_equal and restored_scale_equal,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-label", default="unspecified")
    parser.add_argument("--skip-h1", action="store_true")
    parser.add_argument("--skip-h2", action="store_true")
    parser.add_argument("--skip-h8", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    report: dict[str, Any] = {
        "gpu_label": args.gpu_label,
        "visible_cuda_index": 0,
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
    }
    if not args.skip_h1:
        report["H1"] = run_h1(device)
    if not args.skip_h2:
        report["H2"] = run_h2(device)
    if not args.skip_h8:
        report["H8"] = run_h8(device)
    print(json.dumps(_json_value(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
