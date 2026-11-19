from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from library.models.krea2_raw import inference_runner, training_preview
from library.models.krea2_raw.dit import SingleStreamBlock, SingleStreamDiT
from library.training import sample_preview
from networks.lora_modules.base import BaseLoRAModule


class _FakeAccelerator:
    device = torch.device("cpu")
    process_index = 0
    num_processes = 1
    is_main_process = True

    def __init__(self):
        self.barriers = 0

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def autocast():
        return nullcontext()

    def wait_for_everyone(self):
        self.barriers += 1


class _FakeDiT:
    dtype = torch.float32

    def __init__(self):
        self.inference_calls = 0
        self.training_calls = 0
        self.prepare_calls = 0

    def switch_block_swap_for_inference(self):
        self.inference_calls += 1
        self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        self.training_calls += 1
        self.prepare_block_swap_before_forward()

    def prepare_block_swap_before_forward(self):
        self.prepare_calls += 1


class _FakeNetwork:
    def __init__(self):
        self.training = True
        self.cleared = 0

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def clear_timestep_mask(self):
        self.cleared += 1


def _args(tmp_path: Path, **overrides):
    values = {
        "model_family": "krea2_raw",
        "sample_at_first": False,
        "sample_every_n_epochs": None,
        "sample_every_n_steps": 150,
        "sample_prompts": str(tmp_path / "prompts.txt"),
        "sample_sampler": "euler",
        "disable_block_swap_for_eval": False,
        "output_dir": str(tmp_path),
        "output_name": "anima",
        "seed": 114,
        "mixed_precision": "bf16",
        "save_precision": None,
        "cache_text_encoder_outputs": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _cached_embeddings():
    hiddens = torch.zeros(1, 8, 12, 4)
    mask = torch.ones(1, 8, dtype=torch.bool)
    return {
        "test prompt": [hiddens, mask],
        "": [hiddens.clone(), mask.clone()],
    }


def test_sample_preview_dispatches_krea2_away_from_anima(monkeypatch, tmp_path) -> None:
    calls = []
    trainer = SimpleNamespace(
        get_models_for_text_encoding=lambda *_args: None,
        sample_prompts_te_outputs=_cached_embeddings(),
        sample_prompts_snapshot=[{"prompt": "test prompt"}],
    )

    monkeypatch.setattr(
        sample_preview.text_strategies.TextEncodingStrategy,
        "get_strategy",
        lambda: "encoding-strategy",
    )
    monkeypatch.setattr(
        sample_preview.text_strategies.TokenizeStrategy,
        "get_strategy",
        lambda: "tokenize-strategy",
    )
    monkeypatch.setattr(
        sample_preview.anima_train_utils,
        "sample_images",
        lambda *_args, **_kwargs: pytest.fail("Krea-2 preview entered Anima sampler"),
    )
    monkeypatch.setattr(
        training_preview,
        "sample_images",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    sample_preview.sample_images(
        trainer,
        _FakeAccelerator(),
        _args(tmp_path),
        None,
        150,
        torch.device("cpu"),
        object(),
        None,
        None,
        _FakeDiT(),
        network=_FakeNetwork(),
    )

    assert len(calls) == 1
    positional, keyword = calls[0]
    assert positional[7] == "tokenize-strategy"
    assert positional[8] == "encoding-strategy"
    assert positional[9] is trainer.sample_prompts_te_outputs
    assert keyword["sample_prompts_snapshot"] is trainer.sample_prompts_snapshot


def test_krea2_cached_two_item_text_output_becomes_embedding() -> None:
    cached = _cached_embeddings()

    embedding = training_preview._encode_prompt(
        "test prompt",
        text_encoder=None,
        tokenize_strategy=None,
        text_encoding_strategy=None,
        sample_prompts_te_outputs=cached,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert embedding.hiddens.shape == (1, 8, 12, 4)
    assert embedding.hiddens.dtype == torch.bfloat16
    assert embedding.mask.shape == (1, 8)
    assert embedding.mask.dtype == torch.bool


def test_krea2_cached_text_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="text shapes"):
        training_preview._as_embedding(
            [torch.zeros(1, 8, 11, 4), torch.ones(1, 7, dtype=torch.bool)],
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )


def test_krea2_dit_dtype_skips_nf4_byte_storage() -> None:
    quantized = torch.nn.Parameter(
        torch.zeros(1, dtype=torch.uint8), requires_grad=False
    )
    compute = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
    fake_dit = SimpleNamespace(parameters=lambda: iter((quantized, compute)))

    assert SingleStreamDiT.dtype.fget(fake_dit) == torch.bfloat16


def test_lora_eval_delta_preserves_bf16_base_output() -> None:
    fake_lora = SimpleNamespace(
        enabled=True,
        _fused=False,
        training=False,
        org_forward=lambda x: torch.zeros_like(x, dtype=torch.bfloat16),
        _eval_delta=lambda _x, org: torch.ones_like(org, dtype=torch.float32),
    )

    output = BaseLoRAModule.forward(fake_lora, torch.zeros(1, 4, dtype=torch.bfloat16))

    assert output.dtype == torch.bfloat16
    assert torch.equal(output, torch.ones_like(output))


def test_krea2_block_preserves_bf16_across_fp32_modulation() -> None:
    seen = {}

    def modulation(vec):
        return tuple(torch.zeros_like(vec, dtype=torch.float32) for _ in range(6))

    def attention(x, _freqs, _mask):
        seen["attention"] = x.dtype
        return torch.zeros_like(x, dtype=torch.float32)

    def mlp(x):
        seen["mlp"] = x.dtype
        return torch.zeros_like(x, dtype=torch.float32)

    fake_block = SimpleNamespace(
        mod=modulation,
        prenorm=lambda x: x,
        postnorm=lambda x: x,
        attn=attention,
        mlp=mlp,
    )
    x = torch.ones(1, 4, 8, dtype=torch.bfloat16)
    vec = torch.zeros(1, 8, dtype=torch.bfloat16)

    output = SingleStreamBlock._forward(fake_block, x, vec, torch.empty(0), None)

    assert seen == {"attention": torch.bfloat16, "mlp": torch.bfloat16}
    assert output.dtype == torch.bfloat16


def test_krea2_sampler_casts_accelerate_fp32_velocity_to_latent_dtype(
    monkeypatch,
) -> None:
    seen = {}
    dit = SimpleNamespace(
        config=SimpleNamespace(patch=2, channels=4),
    )
    args = SimpleNamespace(image_size=(32, 32), infer_steps=1, guidance_scale=0.0)
    embedding = SimpleNamespace()

    monkeypatch.setattr(
        inference_runner,
        "forward_for_loss",
        lambda _dit, latents, _text, _t: torch.zeros_like(latents, dtype=torch.float32),
    )

    def fake_sample(dit_forward, latents, cond, _uncond, _seq_len, **_kwargs):
        velocity = dit_forward(latents, cond, torch.ones(1, dtype=latents.dtype))
        seen["latent"] = latents.dtype
        seen["velocity"] = velocity.dtype
        return latents

    monkeypatch.setattr(inference_runner, "sample", fake_sample)

    result = inference_runner.generate_krea2(
        args,
        dit,
        None,
        embedding,
        embedding,
        torch.device("cpu"),
        1,
        torch.bfloat16,
    )

    assert seen == {"latent": torch.bfloat16, "velocity": torch.bfloat16}
    assert result.dtype == torch.bfloat16


def test_krea2_sample_image_uses_official_sampler_and_stages_latent(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    def fake_generate(args, dit, network, cond, uncond, device, seed, dtype):
        captured.update(
            args=args,
            dit=dit,
            network=network,
            cond=cond,
            uncond=uncond,
            device=device,
            seed=seed,
            dtype=dtype,
        )
        return torch.ones(1, 16, 1, 64, 64)

    monkeypatch.setattr(training_preview, "generate_krea2", fake_generate)
    monkeypatch.setattr(
        training_preview.time, "strftime", lambda *_args: "20260810002105"
    )
    dit = _FakeDiT()
    network = _FakeNetwork()
    args = _args(tmp_path)
    save_dir = tmp_path / "sample"

    latent_path = training_preview._sample_image(
        _FakeAccelerator(),
        args,
        dit,
        network,
        None,
        None,
        None,
        str(save_dir),
        {
            "prompt": "test prompt",
            "negative_prompt": "",
            "width": 1024,
            "height": 1024,
            "sample_steps": 28,
            "guidance_scale": 4.0,
            "sample_sampler": "euler",
            "seed": 114,
            "enum": 3,
        },
        None,
        150,
        _cached_embeddings(),
        None,
    )

    assert captured["args"].image_size == (1024, 1024)
    assert captured["args"].infer_steps == 28
    assert captured["args"].guidance_scale == 4.0
    assert captured["seed"] == 114
    assert captured["dtype"] == torch.bfloat16
    assert captured["cond"].hiddens.dtype == torch.bfloat16
    assert latent_path.endswith("anima_000150_03_20260810002105_114.pt")
    record = torch.load(latent_path, map_location="cpu")
    assert record["latents"].shape == (1, 16, 1, 64, 64)
    assert record["prompt"] == "test prompt"
    assert record["enum"] == 3


def test_krea2_sample_lifecycle_restores_training_state(monkeypatch, tmp_path) -> None:
    accelerator = _FakeAccelerator()
    dit = _FakeDiT()
    network = _FakeNetwork()
    decoded = []
    sampled = []
    cache = _cached_embeddings()
    snapshot = [{"prompt": "test prompt", "enum": 0}]

    monkeypatch.setattr(
        training_preview,
        "_sample_image",
        lambda *args, **kwargs: sampled.append((args, kwargs)),
    )
    from library.models.family_plugins import get_family_plugin

    monkeypatch.setattr(
        get_family_plugin("krea2_raw"),
        "decode_preview_samples",
        lambda *args, **kwargs: decoded.append((args, kwargs)),
    )
    rng_before = torch.get_rng_state().clone()

    training_preview.sample_images(
        accelerator,
        _args(tmp_path),
        None,
        150,
        dit,
        object(),
        None,
        object(),
        object(),
        cache,
        network=network,
        sample_prompts_snapshot=snapshot,
    )

    assert len(sampled) == 1
    assert len(decoded) == 1
    assert accelerator.barriers == 2
    assert dit.inference_calls == 1
    assert dit.prepare_calls == 2
    assert dit.training_calls == 1
    assert network.training is True
    assert network.cleared == 2
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_krea2_sample_error_restores_state_before_raising(
    monkeypatch, tmp_path
) -> None:
    accelerator = _FakeAccelerator()
    dit = _FakeDiT()
    network = _FakeNetwork()

    def fail_sample(*_args, **_kwargs):
        raise RuntimeError("sample failed")

    monkeypatch.setattr(training_preview, "_sample_image", fail_sample)

    with pytest.raises(RuntimeError, match="sample failed"):
        training_preview.sample_images(
            accelerator,
            _args(tmp_path),
            None,
            150,
            dit,
            object(),
            None,
            object(),
            object(),
            _cached_embeddings(),
            network=network,
            sample_prompts_snapshot=[{"prompt": "test prompt", "enum": 0}],
        )

    assert accelerator.barriers == 0
    assert dit.training_calls == 1
    assert network.training is True


def test_krea2_training_preview_rejects_anima_only_sampler(tmp_path) -> None:
    with pytest.raises(ValueError, match="only the official Euler sampler"):
        training_preview._sample_image(
            _FakeAccelerator(),
            _args(tmp_path),
            _FakeDiT(),
            None,
            None,
            None,
            None,
            str(tmp_path / "sample"),
            {"prompt": "test prompt", "sample_sampler": "lcm"},
            None,
            150,
            _cached_embeddings(),
            None,
        )
