from contextlib import nullcontext
import inspect
from types import SimpleNamespace

import pytest
import torch

from library.anima import training as anima
from library.models.krea2_raw import training_preview as krea
from library.models.moss_image import training_preview as moss
from library.models.qwen_image_2_1 import training_preview as qwen
from library.models.z_image import training_preview as zimage


class Accelerator:
    device = torch.device("cpu")
    process_index = 0
    num_processes = 1
    is_main_process = True

    def __init__(self):
        self.barriers = 0

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return nullcontext()

    def wait_for_everyone(self):
        self.barriers += 1


class DiT:
    def __init__(self, entry_failure=False, exit_failure=False):
        self.mode = "training"
        self.prepares = {"inference": 0, "training": 0}
        self.predictions = []
        self.entry_failure = entry_failure
        self.exit_failure = exit_failure

    def prepare_block_swap_before_forward(self):
        self.prepares[self.mode] += 1
        if self.mode == "inference" and self.entry_failure:
            raise RuntimeError("entry failed")
        if self.mode == "training" and self.exit_failure:
            raise RuntimeError("exit failed")

    def switch_block_swap_for_inference(self):
        self.mode = "inference"
        self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        self.mode = "training"
        self.prepare_block_swap_before_forward()

    def predict(self, size, cfg_pass):
        assert self.mode == "inference"
        self.predictions.append((size, cfg_pass))
        torch.rand(1)


@pytest.mark.parametrize("family", [anima, krea, qwen, moss, zimage])
def test_all_family_preview_entrypoints_use_shared_lifecycle_contract(family):
    module_source = inspect.getsource(family)
    sample_source = inspect.getsource(family.sample_images)

    assert "from library.training.preview_lifecycle import" in module_source
    assert "with preview_lifecycle(" in module_source
    assert "finish_preview_batch(" in sample_source


@pytest.mark.parametrize("family", [anima, krea, zimage])
@pytest.mark.parametrize("failure", [None, "second_prompt", "entry", "mkdir", "clear", "exit", "second_and_exit"])
def test_family_batch_lifecycle(monkeypatch, tmp_path, family, failure):
    accelerator = Accelerator()
    dit = DiT(entry_failure=failure == "entry", exit_failure="exit" in (failure or ""))
    network = torch.nn.Linear(1, 1).eval()
    def clear():
        if failure == "clear":
            raise RuntimeError("clear failed")
    network.clear_timestep_mask = clear
    decoded = []
    monkeypatch.setattr(anima, "_ensure_sample_compile_range", lambda *_: False)
    monkeypatch.setattr(anima, "decode_samples_for_live_preview", lambda *a, **k: decoded.append(1))
    prompt_index = 8 if family is anima else 8 if family is krea else 7
    def sample(*args):
        prompt = args[prompt_index]
        # Two differently sized prompts, multiple denoising steps and CFG
        # predictions all share the batch layout with no extra prepare.
        for _ in range(3):
            for cfg_pass in ("positive", "negative"):
                dit.predict((prompt["width"], prompt["height"]), cfg_pass)
        if failure in ("second_prompt", "second_and_exit") and prompt["enum"] == 1:
            raise RuntimeError("second prompt failed")
    monkeypatch.setattr(family, "_sample_image_inference" if family is anima else "_sample_image", sample)
    if failure == "mkdir":
        def bad_mkdir(*a, **k):
            raise OSError("mkdir failed")
        monkeypatch.setattr(family.os, "makedirs", bad_mkdir)
    args = SimpleNamespace(sample_at_first=True, sample_every_n_epochs=None,
        sample_every_n_steps=None, sample_prompts="unused", output_dir=str(tmp_path),
        disable_block_swap_for_eval=False)
    before = torch.get_rng_state().clone()
    def run():
        family.sample_images(accelerator, args, None, 0, dit, object(), None,
            None, None, {"cached": []}, network=network,
            sample_prompts_snapshot=[{"enum": 0, "width": 512, "height": 768},
                {"enum": 1, "width": 768, "height": 512}])
    if failure:
        with pytest.raises(Exception, match="failed") as caught:
            run()
        assert accelerator.barriers == 0
        assert decoded == []
        if failure == "second_and_exit":
            assert str(caught.value) == "second prompt failed"
            assert "exit failed" in caught.value.__notes__[0]
    else:
        run()
        assert accelerator.barriers == 2
        assert len(decoded) == 1
        assert len(dit.predictions) == 12
    assert network.training is False
    assert torch.equal(before, torch.get_rng_state())
    expected = 0 if failure == "clear" else 1
    assert dit.prepares == {"inference": expected, "training": expected}
