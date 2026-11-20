"""P2-2 GPU L2 bad-value smoke: run_family_flow_step -> _check_flow_values CUDA branch.

These tests exercise the real CUDA branch of ``_check_flow_values`` (torch._assert_async)
end to end, reproducing the /tmp smoke that proved: 4 bad-value cases die with a
device-side assert, the positive case passes, and ``good_pred_fp16_overflow`` (fp16
overflow inside GradScaler territory) is NOT rejected.

Why subprocess isolation: ``torch._assert_async`` failures surface at the next CUDA
synchronize and permanently poison the CUDA context of the process that hit them. Each
case therefore runs in its own ``python`` subprocess (``sys.executable``); the parent
judges only by exit code plus device-side assert / CUDA error markers on stderr -- never
by Python exception type or message text, which are implementation details of the
assert path.

Gating: skipped entirely unless ``FLOW_STEP_CUDA=1`` AND ``torch.cuda.is_available()``.
Only the devices granted via the environment (``CUDA_VISIBLE_DEVICES``) are touched;
the tests never enumerate or select GPUs on their own.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _cuda_gate() -> None:
    if os.environ.get("FLOW_STEP_CUDA") != "1":
        pytest.skip("Set FLOW_STEP_CUDA=1 to run the CUDA L2 bad-value smoke")
    if not torch.cuda.is_available():
        pytest.skip("Set CUDA_VISIBLE_DEVICES to an available CUDA device")


# The subprocess runs ONE case per invocation so an async device-side assert
# cannot contaminate a sibling case's CUDA context.
_CHILD_SCRIPT = r'''
#!/usr/bin/env python3
"""One case of the P2-2 GPU L2 bad-value smoke (see tests/test_flow_step_assert_async.py)."""
import sys

sys.path.insert(0, {repo_root!r})

import torch


class Args:
    weighting_scheme = "none"
    min_snr_gamma = None
    p2_gamma = 1.0
    p2_k = 1.0
    gradient_checkpointing = False
    max_train_steps = 0
    gradient_accumulation_steps = 1


class Acc:
    def autocast(self):
        return torch.autocast("cuda", enabled=False)


class Ctx:
    def __init__(self):
        self.args = Args()
        self.accelerator = Acc()
        self.unet = None
        self.network = None


class Plugin:
    def __init__(self, mode):
        self.mode = mode
        self.family = None

    def sample_sigmas(self, b, h, w, args, device, context=None):
        if self.mode == "sigma_nan":
            s = torch.rand(b, device=device)
            s[0] = float("nan")
            return s
        if self.mode == "sigma_inf":
            s = torch.rand(b, device=device)
            s[0] = float("inf")
            return s
        if self.mode == "sigma_range":
            s = torch.rand(b, device=device)
            s[0] = 1.5  # outside [0, 1], finite
            return s
        if self.mode == "timestep_inf":
            context.timesteps = torch.tensor([0.5, float("inf")], device=device)
            return torch.rand(b, device=device)
        return torch.rand(b, device=device)

    def prepare_conditions(self, batch, conds, context=None, is_train=True):
        return None

    def forward_velocity_4d(self, unet, noisy, conditions, sigmas, context=None):
        out = torch.randn(noisy.shape, device=noisy.device, dtype=torch.float32)
        if self.mode == "good_pred_fp16_overflow":
            out = out.half()
            out[0, 0, 0, 0] = float("inf")  # fp16 overflow: GradScaler territory, allowed
        return out


def main():
    mode = sys.argv[1]
    torch.manual_seed(0)
    dev = torch.device("cuda")
    latents = torch.randn(2, 16, 4, 4, device=dev)
    batch = object()
    ctx = Ctx()
    plugin = Plugin(mode)
    from library.training.flow_step import run_family_flow_step

    pred, target, timesteps, weighting = run_family_flow_step(
        trainer=type("T", (), {{"_hydra_warmup_step": 0}})(),
        ctx=ctx,
        plugin=plugin,
        latents=latents,
        batch=batch,
        text_encoder_conds=None,
        is_train=False,
    )
    # Explicit synchronize: torch._assert_async is asynchronous on the device; without
    # forcing a sync here the assertion failure could stay pending and be silently
    # swallowed when the process exits before the driver reports it.
    torch.cuda.synchronize()
    print(
        f"[{{mode}}] ok pred={{tuple(pred.shape)}} dtype={{pred.dtype}} "
        f"finite={{bool(torch.isfinite(pred.float()).all())}} "
        f"max_memory_allocated_mib={{torch.cuda.max_memory_allocated()/1024/1024:.2f}}"
    )


if __name__ == "__main__":
    main()
'''.format(repo_root=str(_REPO_ROOT))

_BAD_CASES = ("sigma_nan", "sigma_inf", "sigma_range", "timestep_inf")
_GOOD_CASES = ("good", "good_pred_fp16_overflow")
_ASSERT_MARKERS = ("device-side assert", "CUDA error")


def _run_case(mode: str) -> subprocess.CompletedProcess:
    _cuda_gate()
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "flow_step_case.py"
        script.write_text(_CHILD_SCRIPT, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(script), mode],
            capture_output=True,
            text=True,
            timeout=300,
        )


@pytest.mark.parametrize("mode", list(_BAD_CASES))
def test_flow_values_bad_value_triggers_device_side_assert(mode: str) -> None:
    proc = _run_case(mode)
    assert proc.returncode != 0, (
        f"{mode}: expected nonzero exit, got {proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    stderr_lower = proc.stderr.lower()
    assert any(marker.lower() in stderr_lower for marker in _ASSERT_MARKERS), (
        f"{mode}: nonzero exit without device-side assert / CUDA error marker\n"
        f"stderr={proc.stderr}"
    )


@pytest.mark.parametrize("mode", list(_GOOD_CASES))
def test_flow_values_good_case_passes(mode: str) -> None:
    proc = _run_case(mode)
    assert proc.returncode == 0, (
        f"{mode}: expected clean exit, got {proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    # Positive case prints max_memory_allocated so the memory footprint is reproducible.
    assert "max_memory_allocated_mib=" in proc.stdout, proc.stdout
