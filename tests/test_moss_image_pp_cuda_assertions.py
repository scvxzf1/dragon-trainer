"""CPU mocks exercise the actual CUDA node's parent assertions, not GPU work."""
from copy import deepcopy
from pathlib import Path

import pytest

from tests import test_moss_image_pp_runtime as cuda_tests


@pytest.fixture
def payloads(monkeypatch):
    saved, restored = [], []
    for rank in range(2):
        records = [
            {"scale_used": 4.0, "losses": [4.0, 8.0] if rank == 1 else []},
            {"scale_used": 8.0, "losses": [8.0, 16.0] if rank == 1 else []},
        ]
        saved.append({
            "growth_interval": 4, "scale_after_continuation": 8.0,
            "tracker_after_continuation": 2, "saved_digest": {"step": 3},
            "after_continuation_digest": {"step": 5},
            "peak_allocated_mib": 64.0, "peak_reserved_mib": 128.0,
            "continuation_steps": deepcopy(records), "skipped": True,
        })
        restored.append({
            "dtype": "torch.float32", "growth_interval": 4, "scale_after_replay": 8.0,
            "tracker_after_replay": 2, "restored_digest": {"step": 3},
            "after_replay_digest": {"step": 5},
            "peak_allocated_mib": 64.0, "peak_reserved_mib": 128.0,
            "replay_steps": deepcopy(records),
        })

    def load(path, **kwargs):
        stem = Path(path).stem
        rank = int(stem[-1])
        return saved[rank] if stem.startswith("cuda_save_") else restored[rank]

    # No worker starts, CUDA context, device transfer or model loading.
    monkeypatch.setattr(cuda_tests, "run_pair", lambda *args, **kwargs: None)
    monkeypatch.setattr(cuda_tests.torch, "load", load)
    return saved, restored


@pytest.mark.parametrize("delta", [0.0, 0.005])
def test_real_parent_assertions_accept_registered_tolerance(payloads, tmp_path, delta):
    saved, restored = payloads
    assert cuda_tests.MOSS_CUDA_REPLAY_RTOL == 1e-2  # pre-registered contract
    restored[1]["replay_steps"][0]["losses"][0] *= 1.0 + delta
    cuda_tests.test_two_rank_cuda_fp16_runtime_and_resume(tmp_path)
    assert Path(tmp_path, "cuda_node_evidence.json").is_file()


@pytest.mark.parametrize("mutation", [
    "empty_both", "short_replay", "short_continuation", "extra_both", "outside_prior",
])
def test_real_parent_assertions_refuse_invalid_loss_evidence(payloads, tmp_path, mutation):
    saved, restored = payloads
    continuation = saved[1]["continuation_steps"][0]["losses"]
    replay = restored[1]["replay_steps"][0]["losses"]
    if mutation == "empty_both":
        continuation.clear()
        replay.clear()
    elif mutation == "short_replay":
        replay.pop()
    elif mutation == "short_continuation":
        continuation.pop()
    elif mutation == "extra_both":
        continuation.append(4.0)
        replay.append(4.0)
    else:
        replay[0] *= 1.025
    with pytest.raises(AssertionError):
        cuda_tests.test_two_rank_cuda_fp16_runtime_and_resume(tmp_path)
    assert not Path(tmp_path, "cuda_node_evidence.json").exists()
