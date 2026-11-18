from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from library.models.z_image import training_preview
from library.training import sample_preview
from library.training.compat_matrix import check_training_compat


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


class _RecordingRouterNetwork(_FakeNetwork):
    def __init__(self):
        super().__init__()
        self.events = []

    def set_step_index(self, value):
        self.events.append(("step", value))

    def set_sigma(self, value):
        self.events.append(("sigma", value.detach().clone()))

    def set_content(self, value):
        self.events.append(("content", value.detach().clone()))

    use_content_router = True


class _FakeScheduler:
    def __init__(self):
        self.sigma_min = None
        self.timesteps = torch.tensor([1000.0])
        self.model_output = None
        self.steps = None

    def set_timesteps(self, steps, device):
        self.steps = steps
        self.timesteps = self.timesteps.to(device)

    def step(self, model_output, _timestep, sample, return_dict=False):
        self.model_output = model_output
        return (sample - model_output,)


class _FakeTransformer:
    in_channels = 16

    def __call__(self, *, x, t, cap_feats):
        del t
        samples = [
            torch.full_like(image, float(embedding.mean()))
            for image, embedding in zip(x, cap_feats)
        ]
        return SimpleNamespace(sample=samples)


def _args(tmp_path: Path, **overrides):
    values = {
        "model_family": "z_image",
        "pretrained_model_name_or_path": "/models/z_image_turbo_bf16.safetensors",
        "sample_at_first": False,
        "sample_every_n_epochs": None,
        "sample_every_n_steps": 250,
        "sample_prompts": str(tmp_path / "prompts.txt"),
        "sample_sampler": "euler",
        "output_dir": str(tmp_path),
        "output_name": "z_image",
        "seed": 114,
        "mixed_precision": "bf16",
        "save_precision": None,
        "discrete_flow_shift": 6.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _cached_embeddings():
    hiddens = torch.zeros(1, 8, 4)
    mask = torch.ones(1, 8, dtype=torch.bool)
    return {
        "test prompt": [hiddens, mask],
        "": [torch.ones_like(hiddens), mask.clone()],
    }


def test_sample_preview_dispatches_z_image_to_family_module(
    monkeypatch, tmp_path
) -> None:
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
        training_preview,
        "sample_images",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    sample_preview.sample_images(
        trainer,
        _FakeAccelerator(),
        _args(tmp_path),
        None,
        250,
        torch.device("cpu"),
        object(),
        None,
        None,
        _FakeTransformer(),
        network=_FakeNetwork(),
    )

    assert len(calls) == 1
    assert calls[0][0][7] == "tokenize-strategy"
    assert calls[0][0][8] == "encoding-strategy"


def test_generate_z_image_uses_official_sign_and_cfg(monkeypatch) -> None:
    scheduler = _FakeScheduler()
    monkeypatch.setattr(training_preview, "_make_scheduler", lambda _shift: scheduler)
    cond = [torch.zeros(8, 4)]
    uncond = [torch.ones(8, 4)]

    result = training_preview.generate_z_image(
        _FakeTransformer(),
        cond,
        uncond,
        height=1024,
        width=1024,
        sample_steps=9,
        guidance_scale=2.0,
        flow_shift=3.0,
        seed=114,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert scheduler.steps == 9
    assert scheduler.sigma_min == 0.0
    assert scheduler.model_output.shape == (1, 16, 128, 128)
    assert torch.equal(
        scheduler.model_output, torch.full_like(scheduler.model_output, 2.0)
    )
    assert result.shape == (1, 16, 128, 128)


def test_generate_z_image_refreshes_router_for_each_cfg_branch(monkeypatch) -> None:
    scheduler = _FakeScheduler()
    monkeypatch.setattr(training_preview, "_make_scheduler", lambda _shift: scheduler)
    network = _RecordingRouterNetwork()
    transformer = _FakeTransformer()
    transformer._preview_network = network

    training_preview.generate_z_image(
        transformer,
        [torch.zeros(8, 4)],
        [torch.ones(8, 4)],
        height=1024,
        width=1024,
        sample_steps=9,
        guidance_scale=2.0,
        flow_shift=3.0,
        seed=114,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert [event[0] for event in network.events] == [
        "step", "sigma", "content", "step", "sigma", "content",
    ]
    assert torch.equal(network.events[2][1], torch.zeros(1, 8, 4))
    assert torch.equal(network.events[5][1], torch.ones(1, 8, 4))


def test_z_image_preview_uses_turbo_and_base_scheduler_shifts(tmp_path) -> None:
    assert training_preview._default_flow_shift(_args(tmp_path)) == 3.0
    assert (
        training_preview._default_flow_shift(
            _args(
                tmp_path,
                pretrained_model_name_or_path="/models/z_image_bf16.safetensors",
            )
        )
        == 6.0
    )


def test_z_image_sample_lifecycle_restores_training_state(
    monkeypatch, tmp_path
) -> None:
    accelerator = _FakeAccelerator()
    network = _FakeNetwork()
    sampled = []
    decoded = []
    monkeypatch.setattr(
        training_preview,
        "_sample_image",
        lambda *args, **kwargs: sampled.append((args, kwargs)),
    )
    from library.anima import training as anima_training

    monkeypatch.setattr(
        anima_training,
        "decode_samples_for_live_preview",
        lambda *args, **kwargs: decoded.append((args, kwargs)),
    )
    rng_before = torch.get_rng_state().clone()

    training_preview.sample_images(
        accelerator,
        _args(tmp_path),
        None,
        250,
        _FakeTransformer(),
        object(),
        None,
        object(),
        object(),
        _cached_embeddings(),
        network=network,
        sample_prompts_snapshot=[{"prompt": "test prompt", "enum": 0}],
    )

    assert len(sampled) == 1
    assert len(decoded) == 1
    assert accelerator.barriers == 2
    assert network.training is True
    assert network.cleared == 2
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_decode_pending_z_image_samples_reverses_latent_normalization(tmp_path) -> None:
    class FakeVae(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(scaling_factor=2.0, shift_factor=0.5)
            self.seen = None
            self.use_tiling = False

        @property
        def dtype(self):
            return self.anchor.dtype

        def decode(self, latents, return_dict=False):
            self.seen = latents.detach().clone()
            return (torch.zeros(1, 3, 8, 8, device=latents.device),)

    args = _args(tmp_path)
    latents_dir = tmp_path / "sample" / "latents"
    latents_dir.mkdir(parents=True)
    record_path = latents_dir / "sample.pt"
    torch.save({"latents": torch.full((1, 16, 1, 1), 2.0)}, record_path)
    vae = FakeVae()

    training_preview.decode_pending_samples(_FakeAccelerator(), args, vae)

    assert torch.equal(vae.seen, torch.full_like(vae.seen, 1.5))
    assert (tmp_path / "sample" / "sample.png").is_file()
    assert not record_path.exists()


def test_z_image_compat_allows_training_preview_schedule() -> None:
    result = check_training_compat(
        {
            "model_family": "z_image",
            "network_module": "networks.lora_anima",
            "mixed_precision": "bf16",
            "base_compute": "bf16",
            "attn_mode": "torch",
            "discrete_flow_shift": 6.0,
            "timestep_sampling": "uniform",
            "weighting_scheme": "none",
            "sample_every_n_steps": 250,
        }
    )

    assert "z_image_training_preview" not in {item.code for item in result.errors}
