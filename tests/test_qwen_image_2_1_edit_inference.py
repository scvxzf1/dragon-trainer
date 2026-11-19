"""Standalone Edit public boundaries and shared VLM/VAE reference contracts."""

from types import SimpleNamespace
import weakref

import numpy as np
from PIL import Image
import pytest
import torch

from library.inference.family_runtime import prepare_inference_family
from library.models.qwen_image_2_1 import inference_conditions as conditions
from library.models.qwen_image_2_1 import inference_runner
from library.models.qwen_image_2_1 import inference_text_encoder


def _args(**overrides):
    values = dict(qwen_image_2_1_task="edit", reference_image=None,
                  reference_images=["first.png"], image_size=[64, 96],
                  text_encoder="encoder", vae="vae", prompt="positive",
                  negative_prompt="negative", text_encoder_cpu=True,
                  runtime_dtype="bf16", sampler="euler", infer_steps=1,
                  guidance_scale=2, seed=7)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("overrides, message", [
    ({"reference_images": None}, "edit requires"),
    ({"reference_images": ["x"] * 5}, "1..4"),
    ({"qwen_image_2_1_task": "t2i"}, "t2i does not accept"),
    ({"runtime_dtype": "fp16"}, "BF16"),
])
def test_public_edit_args_fail_closed(overrides, message):
    with pytest.raises(SystemExit, match=message):
        inference_runner.validate_qwen_image_2_1_inference_args(_args(**overrides))


@pytest.mark.parametrize("family", ["anima", "krea2_raw", "z_image"])
def test_qwen_flags_cannot_leak_into_another_family(family):
    with pytest.raises(SystemExit, match="Qwen task/reference flags"):
        prepare_inference_family(_args(), family, "single")


def test_cli_reference_flags_are_mutually_exclusive():
    import inference

    args = inference.parse_args(["--text_encoder", "encoder", "--save_path", "output/tests", "--prompt", "edit", "--qwen_image_2_1_task", "edit",
                                 "--reference_images", "a.png", "b.png"])
    assert args.reference_images == ["a.png", "b.png"]
    with pytest.raises(SystemExit):
        inference.parse_args(["--text_encoder", "encoder", "--save_path", "output/tests", "--prompt", "edit", "--reference_image", "a.png",
                              "--reference_images", "b.png"])


class _Resident(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 1)


def test_edit_uses_one_exif_resize_in_order_and_frees_models(monkeypatch, tmp_path):
    # EXIF turns the first reference into portrait. The second remains landscape.
    first = tmp_path / "portrait.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (64, 32), "red").save(first, exif=exif)
    second = tmp_path / "landscape.png"
    Image.new("RGBA", (32, 16), (0, 255, 0, 100)).save(second)
    observed = {"reads": [], "text": [], "pixels": [], "events": []}
    original_read = conditions.read_reference
    residents = {}

    def read(path):
        observed["reads"].append(path)
        return original_read(path)

    def load_te(path, *, device, dtype):
        assert device == "cpu" and dtype == torch.bfloat16
        model = _Resident()
        residents["te"] = weakref.ref(model)
        observed["events"].append("te")
        return model

    def encode(model, tokenizer, text, images):
        assert residents["te"]() is model
        observed["text"].append((text, images, [np.array(image).copy() for image in images]))
        slots = sum((image.width // 32) * (image.height // 32) for image in images)
        return (torch.zeros(slots + 2, 4), torch.ones(slots + 2, dtype=torch.bool),
                torch.arange(slots + 2) < slots)

    def load_vae(path, *, device, dtype):
        assert residents["te"]() is None
        assert device == torch.device("cpu") and dtype == torch.bfloat16
        model = _Resident()
        residents["vae"] = weakref.ref(model)
        observed["events"].append("vae")
        return model

    def encode_vae(model, pixels):
        assert torch.is_inference_mode_enabled()
        assert pixels.dtype == torch.bfloat16
        observed["pixels"].append(pixels.clone())
        return torch.zeros(1, 64, pixels.shape[-2] // 16, pixels.shape[-1] // 16,
                           dtype=torch.bfloat16)

    monkeypatch.setattr(conditions, "read_reference", read)
    monkeypatch.setattr(conditions, "load_qwen_image_2_1_text_encoder", load_te)
    monkeypatch.setattr(conditions, "QwenImage21EditTokenizeStrategy", lambda path: object())
    monkeypatch.setattr(conditions, "encode_edit_prompt", encode)
    monkeypatch.setattr(conditions, "load_qwen_image_2_1_vae", load_vae)
    monkeypatch.setattr(conditions, "encode_qwen_image_2_1_latents", encode_vae)
    monkeypatch.setattr(conditions, "clean_memory_on_device", lambda device: None)
    monkeypatch.setattr(inference_text_encoder, "clean_memory_on_device", lambda device: None)
    positive, negative, references = conditions.prepare_qwen_image_2_1_conditions(
        _args(reference_images=[str(first), str(second)]), torch.device("cpu"), torch.bfloat16)
    assert observed["reads"] == [first, second]
    assert observed["events"] == ["te", "vae"]
    assert residents["vae"]() is None
    assert observed["text"][0][1] is observed["text"][1][1]
    assert [(array.shape[1], array.shape[0]) for array in observed["text"][0][2]] == [(64, 96), (96, 64)]
    for index, array in enumerate(observed["text"][0][2]):
        expected = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16) / 127.5 - 1
        torch.testing.assert_close(observed["pixels"][index], expected)
    assert positive[2].dtype == negative[2].dtype == torch.bool
    assert [tuple(value.shape) for value in references] == [(1, 64, 6, 4), (1, 64, 4, 6)]


@pytest.mark.parametrize("damage", ["slots", "mask", "grid"])
def test_edit_rejects_inconsistent_conditions(monkeypatch, damage):
    monkeypatch.setattr(conditions, "_freeze_references", lambda args: (Image.new("RGB", (32, 32)),))
    values = (torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool),
              torch.tensor([[True, False]]))
    if damage == "slots":
        values = (*values[:2], torch.zeros(1, 2, dtype=torch.bool))
    if damage == "mask":
        values = (values[0], torch.ones(1, 3, dtype=torch.bool), values[2])
    monkeypatch.setattr(conditions, "_prepare_edit_text", lambda *args: (values, values))
    if damage == "grid":
        monkeypatch.setattr(conditions, "load_qwen_image_2_1_vae", lambda *a, **kw: _Resident())
        monkeypatch.setattr(conditions, "encode_qwen_image_2_1_latents", lambda *a: torch.zeros(1, 64, 3, 2))
        monkeypatch.setattr(conditions, "clean_memory_on_device", lambda *a: None)
    else:
        monkeypatch.setattr(conditions, "_prepare_reference_latents", lambda *a: (torch.zeros(1, 64, 2, 2),))
    with pytest.raises(ValueError, match="slots disagree|mask shape|even H/W"):
        conditions.prepare_qwen_image_2_1_conditions(_args(), torch.device("cpu"), torch.bfloat16)


@pytest.mark.parametrize("stage", ["text", "vae"])
def test_edit_releases_model_on_encoding_error(monkeypatch, stage):
    resident = []
    cleaned = []

    def load(*args, **kwargs):
        value = _Resident()
        resident.append(weakref.ref(value))
        return value

    def fail(*args):
        raise RuntimeError("encode failed")

    monkeypatch.setattr(conditions, "clean_memory_on_device", cleaned.append)
    monkeypatch.setattr(inference_text_encoder, "clean_memory_on_device", cleaned.append)
    monkeypatch.setattr(conditions, "load_qwen_image_2_1_text_encoder", load)
    monkeypatch.setattr(conditions, "load_qwen_image_2_1_vae", load)
    monkeypatch.setattr(conditions, "QwenImage21EditTokenizeStrategy", lambda *a: object())
    monkeypatch.setattr(conditions, "encode_edit_prompt", fail)
    monkeypatch.setattr(conditions, "encode_qwen_image_2_1_latents", fail)
    with pytest.raises(RuntimeError, match="encode failed"):
        if stage == "text":
            conditions._prepare_edit_text(_args(), (), torch.device("cpu"), torch.bfloat16)
        else:
            conditions._prepare_reference_latents(_args(), (Image.new("RGB", (32, 32)),), torch.device("cpu"))
    assert cleaned == [torch.device("cpu")]
    assert all(value() is None for value in resident)


def test_runner_forwards_all_edit_reference_latents(monkeypatch):
    reference = (torch.zeros(1, 64, 2, 4), torch.zeros(1, 64, 4, 2))
    positive = negative = (torch.zeros(1, 4, 2), torch.ones(1, 4, dtype=torch.bool),
                           torch.ones(1, 4, dtype=torch.bool))

    def sample(dit, prompt, pos, neg, refs, **kwargs):
        assert pos is positive and neg is negative and refs is reference
        return torch.zeros(1, 64, 4, 6)

    monkeypatch.setattr(inference_runner, "generate_preview", sample)
    inference_runner.generate_qwen_image_2_1(_args(), object(), None, positive, negative,
                                           torch.device("cpu"), 7, torch.bfloat16, reference=reference)


@pytest.mark.parametrize("fail", [False, True])
def test_facade_loads_dit_after_conditions_and_releases_it(monkeypatch, fail):
    import inference

    events = []
    resident = []
    monkeypatch.setattr(inference_runner, "validate_qwen_image_2_1_inference_args", lambda *a, **kw: None)

    def prepare(*args):
        events.append("conditions")
        return ((), (), (torch.zeros(1, 64, 2, 2),))

    def load(*args):
        assert events == ["conditions", "clean"]
        events.append("dit")
        model = _Resident()
        resident.append(weakref.ref(model))
        return model, None

    def generate(*args, reference):
        assert len(reference) == 1
        if fail:
            raise RuntimeError("sampling failed")
        return torch.zeros(1, 64, 4, 6)

    monkeypatch.setattr(conditions, "prepare_qwen_image_2_1_conditions", prepare)
    monkeypatch.setattr(inference_runner, "load_qwen_image_2_1_dit_for_inference", load)
    monkeypatch.setattr(inference_runner, "generate_qwen_image_2_1", generate)
    monkeypatch.setattr(
        inference_runner,
        "clean_memory_on_device",
        lambda *a: events.append("clean"),
    )
    settings = SimpleNamespace(device=torch.device("cpu"), runtime_dtype=torch.bfloat16)
    if fail:
        with pytest.raises(RuntimeError, match="sampling failed"):
            inference._generate_qwen_image_2_1_latent(_args(), settings)
    else:
        assert inference._generate_qwen_image_2_1_latent(_args(), settings).shape == (1, 64, 4, 6)
    assert events == ["conditions", "clean", "dit", "clean"]
    assert resident[0]() is None
