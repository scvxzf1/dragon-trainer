"""P5-B4 hardware gate for FamilyInferenceRunner lifecycle on the two V100s.

Run only with both authorized V100 UUIDs visible:

    RUN_FAMILY_INFERENCE_RUNNER_HOT=1 \
    CUDA_VISIBLE_DEVICES=GPU-b4b014b2-953b-28e8-b8c1-425c4a401165,GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061 \
    .venv/bin/python -m pytest tests/test_family_inference_runner_hot.py -q

The gate intentionally fails if the opt-in is set but either V100 is missing or
any different device is visible.  It never runs on the GTX 1050.
"""

from __future__ import annotations

import os

import pytest
import torch

from library.inference.family_runner import FamilyInferenceRunner
from library.runtime.device import clean_memory_on_device

V100_UUIDS = (
    "GPU-b4b014b2-953b-28e8-b8c1-425c4a401165",
    "GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061",
)
HOT_ENABLE = os.environ.get("RUN_FAMILY_INFERENCE_RUNNER_HOT") == "1"
pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        not HOT_ENABLE,
        reason="set RUN_FAMILY_INFERENCE_RUNNER_HOT=1 with both V100 UUIDs",
    ),
]


def _require_authorized_v100s() -> None:
    if not torch.cuda.is_available():
        pytest.fail("hot test enabled but torch.cuda is unavailable")
    visible = {
        str(torch.cuda.get_device_properties(index).uuid)
        for index in range(torch.cuda.device_count())
    }
    wanted = {uuid.split("GPU-", 1)[1] for uuid in V100_UUIDS}
    assert visible == wanted, (
        "hot test requires exactly the two authorized V100s; "
        f"visible={sorted(visible)}"
    )


class _GpuStage(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(256 * 1024, device=device)
        )


def _device(index: int) -> torch.device:
    _require_authorized_v100s()
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def _baseline(device: torch.device) -> int:
    torch.cuda.synchronize(device)
    clean_memory_on_device(device)
    torch.cuda.synchronize(device)
    return int(torch.cuda.memory_allocated(device))


def _assert_released(device: torch.device, baseline: int, stages) -> None:
    torch.cuda.synchronize(device)
    clean_memory_on_device(device)
    torch.cuda.synchronize(device)
    assert int(torch.cuda.memory_allocated(device)) <= baseline
    assert all(stage.weight.device.type == "cpu" for stage in stages)


@pytest.mark.parametrize("device_index", [0, 1], ids=["v100-0", "v100-1"])
@pytest.mark.parametrize("case", ["success", "operation_failure", "loader_failure"])
def test_stage_lifecycle_releases_real_cuda_storage(device_index, case):
    device = _device(device_index)
    baseline = _baseline(device)
    stages = []
    cleanup_calls = []

    def cleanup(target):
        cleanup_calls.append(str(target))
        clean_memory_on_device(target)

    runtime = FamilyInferenceRunner("anima", device, cleanup=cleanup)

    if case == "loader_failure":
        def fail_loader():
            scratch = torch.empty(8 * 1024 * 1024, device=device)
            assert scratch.numel() > 0
            raise RuntimeError("loader failed")

        with pytest.raises(RuntimeError, match="loader failed"):
            runtime.run_stage("dit", fail_loader, lambda _loaded: None)
        assert cleanup_calls == [str(device)]
    else:
        stage = _GpuStage(device)
        stages.append(stage)
        if case == "success":
            result = runtime.run_stage(
                "dit",
                lambda: stage,
                lambda loaded: loaded.weight.sum().item(),
            )
            assert result == pytest.approx(256 * 1024)
        else:
            with pytest.raises(RuntimeError, match="operation failed"):
                runtime.run_stage(
                    "dit",
                    lambda: stage,
                    lambda _loaded: (_ for _ in ()).throw(
                        RuntimeError("operation failed")
                    ),
                )
        _assert_released(device, baseline, stages)
        assert cleanup_calls == [str(device)]


@pytest.mark.parametrize("device_index", [0, 1], ids=["v100-0", "v100-1"])
def test_cuda_oom_retry_reloads_and_releases_each_attempt(device_index):
    device = _device(device_index)
    baseline = _baseline(device)
    attempts = []
    stages = []
    cleanup_calls = []

    def cleanup(target):
        cleanup_calls.append(str(target))
        clean_memory_on_device(target)

    def loader():
        stage = _GpuStage(device)
        stages.append(stage)
        return stage

    def operation(stage):
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return stage.weight.sum().item()

    runtime = FamilyInferenceRunner("anima", device, cleanup=cleanup)
    result = runtime.run_stage("dit", loader, operation, max_oom_retries=1)

    assert result == pytest.approx(256 * 1024)
    assert attempts == [0, 1]
    assert len(stages) == 2
    assert all(stage.weight.device.type == "cpu" for stage in stages)
    assert cleanup_calls == [str(device), str(device)]
    _assert_released(device, baseline, stages)


@pytest.mark.parametrize("device_index", [0, 1], ids=["v100-0", "v100-1"])
def test_run_text_releases_te_on_success_and_failure(device_index):
    device = _device(device_index)
    baseline = _baseline(device)
    stages = []
    runtime = FamilyInferenceRunner(
        "anima",
        device,
        cleanup=lambda target: clean_memory_on_device(target),
    )

    stage_ok = _GpuStage(device)
    stage_fail = _GpuStage(device)
    stages.extend((stage_ok, stage_fail))
    assert runtime.run_text(
        lambda: stage_ok,
        lambda loaded: loaded.weight.sum().item(),
    ) == pytest.approx(256 * 1024)
    with pytest.raises(RuntimeError, match="text failed"):
        runtime.run_text(
            lambda: stage_fail,
            lambda _loaded: (_ for _ in ()).throw(RuntimeError("text failed")),
        )

    _assert_released(device, baseline, stages)
