"""Final Phase 5 contract checks for the shared family runtime boundaries."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from library.inference.family_runner import FamilyInferenceRunner

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATHS = (
    ROOT / "library" / "inference" / "anima_legacy_runner.py",
    ROOT / "library" / "models" / "krea2_raw" / "inference_runner.py",
    ROOT / "library" / "models" / "z_image" / "inference_runner.py",
    ROOT / "library" / "models" / "qwen_image_2_1" / "inference_runner.py",
    ROOT / "library" / "models" / "moss_image" / "inference_runner.py",
)


def test_all_single_family_runners_use_shared_stage_runtime():
    for path in RUNNER_PATHS:
        source = path.read_text(encoding="utf-8")
        assert "FamilyInferenceRunner" in source, path


def test_shared_runtime_releases_te_dit_vae_in_order():
    events = []

    class StageModule(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.weight = torch.nn.Parameter(torch.ones(1))

        def to(self, *args, **kwargs):
            events.append(f"{self.name}:release")
            return super().to(*args, **kwargs)

    runtime = FamilyInferenceRunner(
        "phase5",
        torch.device("cpu"),
        cleanup=lambda _device: events.append("allocator"),
    )
    text = StageModule("te")
    dit = StageModule("dit")
    vae = StageModule("vae")

    runtime.run_text(lambda: text, lambda _loaded: "encoded")
    runtime.run_stage("dit", lambda: dit, lambda _loaded: "latent")
    runtime.run_stage("vae", lambda: vae, lambda _loaded: "image")

    assert events == [
        "te:release",
        "allocator",
        "dit:release",
        "allocator",
        "vae:release",
        "allocator",
    ]


def test_anima_single_prompt_releases_text_encoder_before_dit(monkeypatch):
    import library.inference.anima_legacy_runner as legacy
    import library.inference.generation as generation
    import library.inference.models as models
    import library.inference.precision as precision
    import library.inference.text as text

    events = []

    class Encoder(torch.nn.Module):
        def to(self, *args, **kwargs):
            events.append("te:release")

    class DiT(torch.nn.Module):
        device = torch.device("cpu")

        def to(self, *args, **kwargs):
            events.append("dit:release")

        def _preprocess_text_embeds(self, **kwargs):
            events.append("dit:condition")
            return torch.ones(1, 2, 8), None

    def encode(prompt, encoder):
        events.append(f"te:{prompt}")
        return torch.ones(1, 2, 8), torch.ones(1, 2), torch.zeros(1, 2), torch.tensor([[True, False]])

    def generate(args, settings, *, shared_models, precomputed_text_data):
        events.append("dit:generate")
        assert isinstance(shared_models["model"], DiT)
        context = precomputed_text_data["context"]["embed"][0]
        assert context.shape == (1, 512, 8)
        assert torch.count_nonzero(context[:, 1:]) == 0
        return "latent"

    monkeypatch.setattr(
        legacy,
        "clean_memory_on_device",
        lambda _device: events.append("allocator"),
    )
    monkeypatch.setattr(generation, "generate", generate)
    monkeypatch.setattr(models, "load_text_encoder", lambda *args, **kwargs: events.append("te:load") or Encoder())
    monkeypatch.setattr(models, "load_dit_model", lambda *args, **kwargs: events.append("dit:load") or DiT())
    monkeypatch.setattr(precision, "resolve_text_encoder_dtype", lambda args: torch.float32)
    monkeypatch.setattr(text, "ensure_text_strategies", lambda path: None)
    monkeypatch.setattr(text, "encode_anima_prompt", encode)

    result = legacy.AnimaLegacyInferenceRunner().generate_latent(
        SimpleNamespace(prompt="positive", negative_prompt="negative", text_encoder="unused"),
        SimpleNamespace(device=torch.device("cpu"), runtime_dtype=torch.float32),
    )

    assert result == "latent"
    assert events == [
        "te:load", "te:positive", "te:negative", "te:release", "allocator",
        "dit:load", "dit:condition", "dit:condition", "dit:generate", "dit:release", "allocator",
    ]


def test_anima_easycontrol_reference_vae_finishes_before_dit_load(monkeypatch):
    import library.anima.inference_references as references
    import library.inference.anima_legacy_runner as legacy
    import library.inference.generation as generation
    import library.inference.models as models
    import library.inference.precision as precision
    import library.inference.text as text

    events = []

    class Encoder(torch.nn.Module):
        def to(self, *args, **kwargs):
            events.append("te:release")

    class Vae(torch.nn.Module):
        def to(self, *args, **kwargs):
            events.append("vae:release")
            return self

    class DiT(torch.nn.Module):
        device = torch.device("cpu")

        def to(self, *args, **kwargs):
            events.append("dit:release")

        def _preprocess_text_embeds(self, **kwargs):
            events.append("dit:condition")
            return torch.ones(1, 2, 8), None

    monkeypatch.setattr(legacy, "clean_memory_on_device", lambda _device: events.append("allocator"))
    monkeypatch.setattr(models, "load_text_encoder", lambda *args, **kwargs: Encoder())
    monkeypatch.setattr(models, "load_dit_model", lambda *args, **kwargs: events.append("dit:load") or DiT())
    monkeypatch.setattr(precision, "resolve_text_encoder_dtype", lambda _args: torch.float32)
    monkeypatch.setattr(text, "ensure_text_strategies", lambda _path: None)
    monkeypatch.setattr(text, "encode_anima_prompt", lambda _prompt, _encoder: (
        torch.ones(1, 2, 8), torch.ones(1, 2), torch.zeros(1, 2), torch.tensor([[True, False]])
    ))
    monkeypatch.setattr(references, "easycontrol_reference", lambda _args: "reference.png")
    monkeypatch.setattr(references, "select_easycontrol_size", lambda *_args: events.append("select:ref-size"))
    monkeypatch.setattr(references, "load_easycontrol_vae", lambda *_args: events.append("vae:load") or Vae())
    monkeypatch.setattr(references, "encode_easycontrol_reference", lambda *_args: events.append("vae:encode") or torch.ones(1, 16, 32, 32))

    def generate(_args, _settings, *, shared_models, precomputed_text_data):
        events.append("dit:generate")
        assert shared_models["easycontrol_latent"].device.type == "cpu"
        return "latent"

    monkeypatch.setattr(generation, "generate", generate)
    result = legacy.AnimaLegacyInferenceRunner().generate_latent(
        SimpleNamespace(
            prompt="positive", negative_prompt="negative", text_encoder="unused",
            easycontrol_weight="adapter.safetensors", easycontrol_image="reference.png",
            vae="vae.safetensors",
        ),
        SimpleNamespace(device=torch.device("cpu"), runtime_dtype=torch.float32),
    )

    assert result == "latent"
    assert events.index("vae:release") < events.index("dit:load")
    assert events == [
        "te:release", "allocator", "select:ref-size", "vae:load", "vae:encode",
        "vae:release", "allocator", "dit:load", "dit:condition", "dit:condition",
        "dit:generate", "dit:release", "allocator",
    ]


def test_z_image_runner_uses_shared_text_then_dit_stages(monkeypatch):
    from library.models.z_image import inference_runner as z_image_runner

    events = []
    latent = torch.zeros(1, 16, 4, 4)

    class StageModule(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.weight = torch.nn.Parameter(torch.ones(1))

        def to(self, *args, **kwargs):
            events.append(f"{self.name}:release")
            return super().to(*args, **kwargs)

    dit = StageModule("dit")
    monkeypatch.setattr(
        z_image_runner,
        "clean_memory_on_device",
        lambda _device: events.append("allocator"),
    )
    monkeypatch.setattr(
        z_image_runner,
        "load_z_image_text_encoder",
        lambda *args, **kwargs: events.append("te:load") or object(),
    )
    monkeypatch.setattr(
        z_image_runner.ZImageInferenceRunner,
        "_encode_conditions",
        lambda self, *args, **kwargs: events.append("te:encode")
        or ("cond", None),
    )
    monkeypatch.setattr(
        z_image_runner,
        "load_z_image_transformer",
        lambda *args, **kwargs: events.append("dit:load") or dit,
    )
    monkeypatch.setattr(
        z_image_runner,
        "prepare_z_image_attention",
        lambda *args, **kwargs: events.append("dit:prepare"),
    )
    monkeypatch.setattr(
        z_image_runner,
        "generate_z_image",
        lambda *args, **kwargs: events.append("dit:sample") or latent,
    )
    args = SimpleNamespace(
        sampler="euler",
        image_size=[64, 64],
        lora_weight=None,
        prompt="prompt",
        negative_prompt="",
        guidance_scale=1.0,
        dit="dit",
        text_encoder="te",
        infer_steps=1,
        flow_shift=1.0,
        seed=7,
        attn_mode="torch",
        text_encoder_dtype="bf16",
    )

    result = z_image_runner.ZImageInferenceRunner().generate_latent(
        args,
        SimpleNamespace(
            device=torch.device("cpu"),
            runtime_dtype=torch.bfloat16,
        ),
    )

    assert result is latent
    assert events == [
        "te:load",
        "te:encode",
        "allocator",
        "dit:load",
        "dit:prepare",
        "dit:sample",
        "dit:release",
        "allocator",
    ]


def test_z_image_runner_uses_shared_vae_stage(monkeypatch, tmp_path):
    from library.models.z_image import inference_runner as z_image_runner

    events = []

    class StageModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def to(self, *args, **kwargs):
            events.append("vae:release")
            return super().to(*args, **kwargs)

    vae = StageModule()
    monkeypatch.setattr(
        z_image_runner,
        "clean_memory_on_device",
        lambda _device: events.append("allocator"),
    )
    monkeypatch.setattr(
        z_image_runner,
        "load_z_image_vae",
        lambda *args, **kwargs: events.append("vae:load") or vae,
    )
    monkeypatch.setattr(
        z_image_runner,
        "decode_z_image_latent",
        lambda *args, **kwargs: events.append("vae:decode") or object(),
    )
    monkeypatch.setattr(
        z_image_runner,
        "save_pil_image",
        lambda *args, **kwargs: events.append("vae:save"),
    )
    args = SimpleNamespace(
        output_type="images",
        image_size=[64, 64],
        vae="vae",
        seed=7,
        runtime_dtype="bf16",
        save_path=str(tmp_path),
    )

    z_image_runner.ZImageInferenceRunner().save_output(
        args,
        torch.zeros(1, 16, 4, 4),
        torch.device("cpu"),
    )

    assert events == [
        "vae:load",
        "vae:decode",
        "vae:save",
        "vae:release",
        "allocator",
    ]


def test_mixed_nf4_cli_excludes_moss_image():
    from scripts.quantization import convert_mixed_nf4 as cli

    assert cli.SUPPORTED_FAMILIES == (
        "anima",
        "krea2_raw",
        "z_image",
        "qwen_image_2_1",
    )
    assert "moss_image" not in cli.SUPPORTED_FAMILIES
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--family",
                "moss_image",
                "--source",
                "unused.safetensors",
                "--output",
                "unused-output.safetensors",
            ]
        )
