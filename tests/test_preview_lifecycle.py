from types import SimpleNamespace

import pytest
import torch

from library.training.preview_lifecycle import preview_lifecycle


class Network(torch.nn.Module):
    def __init__(self, fail_clear=False):
        super().__init__()
        self.fail_clear = fail_clear
        self.clear_timestep_mask_calls = 0

    def clear_timestep_mask(self):
        self.clear_timestep_mask_calls += 1
        torch.rand(1)
        if self.fail_clear:
            raise RuntimeError("clear failed")


class DiT:
    def __init__(self, fail_entry=False, fail_exit=False):
        self.prepares = {"inference": 0, "training": 0}
        self.mode = "training"
        self.fail_entry = fail_entry
        self.fail_exit = fail_exit

    def prepare_block_swap_before_forward(self):
        self.prepares[self.mode] += 1
        torch.rand(1)
        if (self.mode == "inference" and self.fail_entry) or (
            self.mode == "training" and self.fail_exit
        ):
            raise RuntimeError(f"{self.mode} prepare failed")

    def switch_block_swap_for_inference(self):
        self.mode = "inference"
        self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        self.mode = "training"
        self.prepare_block_swap_before_forward()


@pytest.mark.parametrize("failure", ["clear", "entry", "body", "exit", "body_and_exit"])
@pytest.mark.parametrize("training", [True, False])
def test_restoration_errors_preserve_original_and_attempt_all_cleanup(failure, training):
    network = Network(fail_clear=failure == "clear").train(training)
    dit = DiT(fail_entry=failure == "entry", fail_exit="exit" in failure)
    before = torch.get_rng_state().clone()
    expected = {
        "clear": "clear failed", "entry": "inference prepare failed",
        "body": "sample failed", "exit": "training prepare failed",
        "body_and_exit": "sample failed",
    }[failure]
    with pytest.raises(RuntimeError, match=expected) as caught:
        with preview_lifecycle(dit, network):
            torch.rand(8)
            if failure.startswith("body"):
                raise RuntimeError("sample failed")
    assert network.training == training
    # Swap cleanup itself can consume randomness. The snapshot must still win.
    assert torch.equal(torch.get_rng_state(), before)
    assert network.clear_timestep_mask_calls == 2
    if failure == "body_and_exit":
        assert "training prepare failed" in caught.value.__notes__[0]


@pytest.mark.parametrize("result", [True, False, "partial_failure"])
def test_pause_resume_contract(result):
    dit = DiT()
    state = SimpleNamespace(pauses=0, resumes=0, paused=False)

    def pause():
        state.pauses += 1
        if result == "partial_failure":
            raise RuntimeError("pause failed")
        state.paused = result
        return result

    def resume():
        state.resumes += 1
        return state.paused

    dit.pause_block_swap = pause
    dit.resume_block_swap = resume
    if result == "partial_failure":
        with pytest.raises(RuntimeError, match="pause failed"):
            with preview_lifecycle(dit, Network(), disable_block_swap=True):
                pytest.fail("failed pause must not sample")
        assert dit.prepares == {"inference": 0, "training": 1}
    else:
        with preview_lifecycle(dit, Network(), disable_block_swap=True):
            assert state.pauses == 1
        assert dit.prepares["inference"] == 0
        assert dit.prepares["training"] == int(not result)
    assert state.resumes == int(result is True or result == "partial_failure")


def test_rng_cleanup_failure_does_not_prevent_cuda_rng_or_mode_recovery(monkeypatch):
    dit = DiT(fail_exit=True)
    network = Network()
    cuda_restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: "cuda snapshot")
    monkeypatch.setattr(torch.cuda, "set_rng_state", cuda_restored.append)
    def fail_rng(_state):
        raise RuntimeError("CPU RNG restore failed")
    monkeypatch.setattr(torch, "set_rng_state", fail_rng)
    with pytest.raises(RuntimeError, match="sample failed") as caught:
        with preview_lifecycle(dit, network):
            raise RuntimeError("sample failed")
    assert network.training is True
    assert cuda_restored == ["cuda snapshot"]
    assert len(caught.value.__notes__) == 2
    assert "CPU RNG restore failed" in caught.value.__notes__[1]
