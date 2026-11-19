from __future__ import annotations

import os

import pytest
import torch

from library.training.flow_step import (
    mix_latents,
    run_family_flow_step,
    unit_weighting,
    velocity_target,
)


def _flow_test_device():
    if os.environ.get("FLOW_STEP_CUDA") != "1":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("Set CUDA_VISIBLE_DEVICES to an available CUDA device")
    return torch.device("cuda")


def test_flow_step_shared_4d_math() -> None:
    latents = torch.ones(2, 4, 3, 5)
    noise = torch.full_like(latents, 3.0)
    sigmas = torch.tensor([0.25, 0.75])

    mixed = mix_latents(latents, noise, sigmas)
    assert torch.equal(mixed[0], torch.full_like(mixed[0], 1.5))
    assert torch.equal(mixed[1], torch.full_like(mixed[1], 2.5))
    assert torch.equal(velocity_target(noise, latents), torch.full_like(latents, 2.0))
    assert unit_weighting(sigmas, latents).shape == (2, 1, 1, 1)


@pytest.mark.parametrize(
    "latents, noise, sigmas",
    [
        (torch.ones(2, 4, 3), torch.ones(2, 4, 3), torch.ones(2)),
        (torch.ones(2, 4, 3, 5), torch.ones(1, 4, 3, 5), torch.ones(2)),
        (torch.ones(2, 4, 3, 5), torch.ones(2, 4, 3, 5), torch.ones(1)),
    ],
)
def test_flow_step_rejects_invalid_shapes(latents, noise, sigmas) -> None:
    with pytest.raises(ValueError, match="flow"):
        mix_latents(latents, noise, sigmas)


def test_shared_flow_engine_owns_mixing_router_scope_target_and_loss_weighting(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    latents = torch.ones(2, 4, 3, 5)
    seen = {}
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.full_like(tensor, 3.0))

    class Plugin:
        def sample_sigmas(self, batch_size, height, width, args, device, *, context):
            return torch.tensor([0.25, 0.75])

        def prepare_conditions(self, batch, conditions, *, context, is_train):
            return conditions

        def forward_velocity_4d(self, dit, noisy, conditions, sigmas, *, context):
            seen["noisy"] = noisy
            seen["sigmas"] = sigmas
            return torch.zeros_like(noisy)

    context = SimpleNamespace(
        args=SimpleNamespace(
            gradient_checkpointing=False,
            weighting_scheme="none",
            min_snr_gamma=None,
            p2_gamma=1.0,
            p2_k=1.0,
            max_train_steps=10,
            gradient_accumulation_steps=1,
        ),
        accelerator=SimpleNamespace(device="cpu", autocast=nullcontext),
        network=None,
        unet=None,
        weight_dtype=torch.float32,
    )
    trainer = SimpleNamespace(_hydra_warmup_step=0)
    prediction, target, sigmas, weighting = run_family_flow_step(
        trainer, context, Plugin(), latents, {}, ("conditions",)
    )

    torch.testing.assert_close(seen["noisy"][0], torch.full_like(latents[0], 1.5))
    torch.testing.assert_close(seen["noisy"][1], torch.full_like(latents[1], 2.5))
    torch.testing.assert_close(target, torch.full_like(latents, 2.0))
    assert torch.equal(seen["sigmas"], sigmas)
    assert prediction.shape == target.shape == latents.shape
    assert weighting.shape == (2, 1, 1, 1)
    assert torch.equal(sigmas, torch.tensor([0.25, 0.75]))


def test_shared_flow_engine_rejects_cpu_sigma_outside_unit_interval():
    from contextlib import nullcontext
    from types import SimpleNamespace

    class Plugin:
        def sample_sigmas(self, batch_size, height, width, args, device, *, context):
            return torch.tensor([1.5])

    context = SimpleNamespace(
        args=SimpleNamespace(gradient_checkpointing=False),
        accelerator=SimpleNamespace(device="cpu", autocast=nullcontext),
        network=None,
    )
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        run_family_flow_step(
            SimpleNamespace(), context, Plugin(), torch.ones(1, 4, 2, 2), {}, None
        )


def test_shared_flow_engine_keeps_unit_weighting_for_family_default(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.zeros_like(tensor))

    class Plugin:
        def sample_sigmas(self, batch_size, height, width, args, device, *, context):
            return torch.full((batch_size,), 0.5, device=device)

        def prepare_conditions(self, batch, conditions, *, context, is_train):
            return conditions

        def forward_velocity_4d(self, dit, noisy, conditions, sigmas, *, context):
            return torch.zeros_like(noisy)

    context = SimpleNamespace(
        args=SimpleNamespace(
            gradient_checkpointing=False,
            weighting_scheme="cosmap",
            min_snr_gamma=None,
            p2_gamma=1.0,
            p2_k=1.0,
            max_train_steps=10,
            gradient_accumulation_steps=1,
        ),
        accelerator=SimpleNamespace(device="cpu", autocast=nullcontext),
        network=None,
        unet=None,
        weight_dtype=torch.float32,
    )
    *_, weighting = run_family_flow_step(
        SimpleNamespace(), context, Plugin(), torch.ones(1, 4, 2, 2), {}, None
    )
    torch.testing.assert_close(weighting, torch.ones(1, 1, 1, 1))


def test_shared_flow_engine_accepts_family_weighting_hook(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.zeros_like(tensor))

    class Plugin:
        def sample_sigmas(self, batch_size, height, width, args, device, *, context):
            return torch.full((batch_size,), 0.5, device=device)

        def prepare_conditions(self, batch, conditions, *, context, is_train):
            return conditions

        def forward_velocity_4d(self, dit, noisy, conditions, sigmas, *, context):
            return torch.zeros_like(noisy)

        def compute_loss_weighting(self, sigmas, latents, args):
            return torch.full((latents.shape[0], 1, 1, 1), 3.0)

    context = SimpleNamespace(
        args=SimpleNamespace(
            gradient_checkpointing=False,
            weighting_scheme="none",
            max_train_steps=10,
            gradient_accumulation_steps=1,
        ),
        accelerator=SimpleNamespace(device="cpu", autocast=nullcontext),
        network=None,
        unet=None,
        weight_dtype=torch.float32,
    )
    *_, weighting = run_family_flow_step(
        SimpleNamespace(), context, Plugin(), torch.ones(1, 4, 2, 2), {}, None
    )
    torch.testing.assert_close(weighting, torch.full((1, 1, 1, 1), 3.0))


def test_krea2_plugin_keeps_legacy_flow_weighting_shape():
    from types import SimpleNamespace

    from library.models.krea2_raw.plugin import Krea2Plugin

    sigmas = torch.tensor([0.25, 0.75])
    latents = torch.ones(2, 16, 2, 2)
    args = SimpleNamespace(
        weighting_scheme="cosmap",
        min_snr_gamma=None,
        p2_gamma=1.0,
        p2_k=1.0,
    )
    weights = Krea2Plugin().compute_loss_weighting(sigmas, latents, args)

    assert weights.shape == (2, 1, 1, 1)
    torch.testing.assert_close(
        weights.flatten(), torch.full((2,), 2.0 / (torch.pi * 0.625))
    )


def test_krea2_plugin_executes_through_shared_flow_engine(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from library.models.krea2_raw import family
    from library.models.krea2_raw.plugin import Krea2Plugin

    device = _flow_test_device()
    latents = torch.ones(2, 16, 2, 3, device=device)
    sigma = torch.tensor([0.25, 0.75], device=device)
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.full_like(tensor, 3.0))

    def sample(context):
        context.noisy_input = (
            (1.0 - sigma.view(-1, 1, 1, 1)) * context.latents
            + sigma.view(-1, 1, 1, 1) * context.noise
        )
        context.timesteps = sigma
        return sigma

    monkeypatch.setattr(family, "sample_krea2_sigmas", sample)
    monkeypatch.setattr(
        family,
        "prepare_krea2_conditions",
        lambda trainer, context, batch, conditions: conditions,
    )
    monkeypatch.setattr(
        family,
        "forward_krea2_velocity",
        lambda dit, noisy, conditions, sigmas, context: torch.zeros_like(noisy),
    )
    context = SimpleNamespace(
        args=SimpleNamespace(
            gradient_checkpointing=False,
            weighting_scheme="cosmap",
            min_snr_gamma=None,
            p2_gamma=1.0,
            p2_k=1.0,
            max_train_steps=10,
            gradient_accumulation_steps=1,
        ),
        accelerator=SimpleNamespace(device=device, autocast=nullcontext),
        noise_scheduler=None,
        network=None,
        unet=object(),
        weight_dtype=torch.float32,
    )
    trainer = SimpleNamespace(_hydra_warmup_step=0)
    prediction, target, timesteps, weighting = Krea2Plugin().compute_noise_pred_and_target(
        trainer, context, latents, {}, ("conditions",)
    )

    torch.testing.assert_close(target, torch.full_like(latents, 2.0))
    torch.testing.assert_close(timesteps, sigma)
    torch.testing.assert_close(prediction, torch.zeros_like(latents))
    torch.testing.assert_close(
        weighting.flatten(), torch.full((2,), 2.0 / (torch.pi * 0.625), device=device)
    )


def test_krea2_sampler_returns_sigmas_and_preserves_distinct_timesteps(monkeypatch):
    from types import SimpleNamespace

    from library.models.krea2_raw.family import sample_krea2_sigmas
    from library.training.samplers import SAMPLER_REGISTRY, SamplerOut

    sigmas = torch.tensor([0.2, 0.8])
    timesteps = torch.tensor([0.25, 0.75])
    sampler_output = SamplerOut(
        noisy_input=torch.zeros(2, 4, 2, 2),
        timesteps=timesteps,
        sigmas=sigmas,
    )
    monkeypatch.setitem(
        SAMPLER_REGISTRY,
        "test-distinct-sigma-timestep",
        lambda sampler_context: sampler_output,
    )
    train_context = SimpleNamespace(
        args=SimpleNamespace(sampler="test-distinct-sigma-timestep"),
        noise_scheduler=object(),
        accelerator=SimpleNamespace(device=torch.device("cpu")),
        weight_dtype=torch.float32,
    )
    flow_context = SimpleNamespace(
        train_context=train_context,
        latents=torch.ones(2, 4, 2, 2),
        noise=torch.zeros(2, 4, 2, 2),
    )

    returned_sigmas = sample_krea2_sigmas(flow_context)

    torch.testing.assert_close(returned_sigmas, sigmas)
    torch.testing.assert_close(flow_context.timesteps, timesteps)
    assert flow_context.timesteps is not returned_sigmas
    assert flow_context.noisy_input is sampler_output.noisy_input

    seen = {}

    def forward(dit, latents, conditions, model_timesteps):
        seen["timesteps"] = model_timesteps
        return latents

    monkeypatch.setattr("library.models.krea2_raw.family.forward_for_loss", forward)
    from library.models.krea2_raw.family import forward_krea2_velocity

    prediction = forward_krea2_velocity(
        object(),
        torch.zeros(2, 4, 2, 2),
        None,
        returned_sigmas,
        flow_context,
    )
    torch.testing.assert_close(seen["timesteps"], timesteps)
    assert prediction.shape == (2, 4, 2, 2)


def test_anima_sampler_flattens_broadcast_sigmas(monkeypatch):
    from types import SimpleNamespace

    from library.anima.plugin import AnimaPlugin
    from library.training.samplers import SAMPLER_REGISTRY, SamplerOut

    sigmas = torch.tensor([0.2, 0.8])
    timesteps = torch.tensor([0.2, 0.8])
    sampler_output = SamplerOut(
        noisy_input=torch.zeros(2, 4, 2, 2),
        timesteps=timesteps,
        sigmas=sigmas.reshape(2, 1, 1, 1),
    )
    monkeypatch.setitem(
        SAMPLER_REGISTRY,
        "test-broadcast-sigmas",
        lambda sampler_context: sampler_output,
    )
    train_context = SimpleNamespace(
        noise_scheduler=object(),
        weight_dtype=torch.float32,
    )
    flow_context = SimpleNamespace(
        train_context=train_context,
        latents=torch.ones(2, 4, 2, 2),
        noise=torch.zeros(2, 4, 2, 2),
        noisy_input=None,
        timesteps=None,
    )

    returned_sigmas = AnimaPlugin().sample_sigmas(
        2,
        2,
        2,
        SimpleNamespace(sampler="test-broadcast-sigmas"),
        torch.device("cpu"),
        context=flow_context,
    )

    torch.testing.assert_close(returned_sigmas, sigmas)
    torch.testing.assert_close(flow_context.timesteps, timesteps)
    assert returned_sigmas.ndim == 1


def test_anima_forward_uses_sampler_timesteps_not_sigma(monkeypatch):
    from types import SimpleNamespace

    from library.anima.plugin import AnimaPlugin

    seen = {}

    class Dit:
        def __call__(self, noisy, model_timesteps, prompt_embeds, **kwargs):
            seen["timesteps"] = model_timesteps
            return torch.zeros_like(noisy)

    timesteps = torch.tensor([0.25, 0.75])
    sigmas = torch.tensor([0.2, 0.8])
    context = SimpleNamespace(
        train_context=SimpleNamespace(weight_dtype=torch.float32, network=None),
        timesteps=timesteps,
    )
    conditions = SimpleNamespace(
        crossattn_emb=None,
        prompt_embeds=torch.zeros(2, 1, 4),
        t5_input_ids=None,
        t5_attn_mask=None,
        attn_mask=None,
    )

    prediction = AnimaPlugin().forward_velocity_4d(
        Dit(), torch.zeros(2, 4, 2, 2), conditions, sigmas, context=context
    )

    torch.testing.assert_close(seen["timesteps"], timesteps)
    assert prediction.shape == (2, 4, 2, 2)


@pytest.mark.parametrize("family", ["moss_image", "qwen_image_2_1", "z_image"])
def test_family_plugin_executes_through_shared_flow_engine(monkeypatch, family):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from library.models.family_plugins import get_family_plugin

    device = _flow_test_device()
    module_names = {
        "moss_image": "library.models.moss_image.family",
        "qwen_image_2_1": "library.models.qwen_image_2_1.family",
        "z_image": "library.models.z_image.family",
    }
    family_module = __import__(module_names[family], fromlist=["*"])
    sigmas = torch.tensor([0.25, 0.75], device=device)

    if family == "moss_image":
        monkeypatch.setattr(
            family_module,
            "uniform_grid_sigmas",
            lambda batch_size, *, device, **kwargs: sigmas.to(device),
        )
        monkeypatch.setattr(
            family_module, "_prepare_text_conditions", lambda *args: args[-1]
        )
        monkeypatch.setattr(
            family_module,
            "forward_for_loss",
            lambda dit, noisy, conditions, sampled_sigmas: torch.zeros_like(noisy),
        )
        channels = 128
        conditions = (
            torch.zeros(2, 1, 8, device=device),
            torch.ones(2, 1, dtype=torch.bool, device=device),
        )
    elif family == "qwen_image_2_1":
        monkeypatch.setattr(
            family_module,
            "sample_sigmas",
            lambda batch_size, image_tokens, *, device: sigmas.to(device),
        )
        qwen_conditions = SimpleNamespace(
            hiddens=torch.zeros(2, 1, 4096, device=device),
            mask=torch.ones(2, 1, dtype=torch.bool, device=device),
            reference_latents=None,
            image_slot_mask=None,
        )
        monkeypatch.setattr(
            family_module,
            "prepare_training_conditions",
            lambda *args: qwen_conditions,
        )
        monkeypatch.setattr(
            family_module,
            "forward_for_loss",
            lambda dit, noisy, *args, **kwargs: torch.zeros_like(noisy),
        )
        channels = 64
        conditions = (
            torch.zeros(2, 1, 4096, device=device),
            torch.ones(2, 1, dtype=torch.bool, device=device),
        )
    else:
        monkeypatch.setattr(
            family_module,
            "shifted_uniform_sigmas",
            lambda batch_size, **kwargs: sigmas.to(kwargs["device"]),
        )
        monkeypatch.setattr(
            family_module, "_prepare_text_conditions", lambda *args, **kwargs: []
        )
        monkeypatch.setattr(
            family_module,
            "forward_for_loss",
            lambda dit, noisy, conditions, sampled_sigmas: torch.zeros_like(noisy),
        )
        channels = 16
        conditions = (
            torch.zeros(2, 1, 2560, device=device),
            torch.ones(2, 1, dtype=torch.bool, device=device),
        )

    latents = torch.ones(2, channels, 2, 2, device=device)
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.full_like(tensor, 3.0))
    context = SimpleNamespace(
        args=SimpleNamespace(
            gradient_checkpointing=False,
            weighting_scheme="cosmap",
            max_train_steps=10,
            gradient_accumulation_steps=1,
            discrete_flow_shift=6.0,
            qwen_image_2_1_task="t2i",
        ),
        accelerator=SimpleNamespace(device=device, autocast=nullcontext),
        network=None,
        unet=object(),
        weight_dtype=torch.float32,
    )
    trainer = SimpleNamespace(_hydra_warmup_step=0)

    prediction, target, timesteps, weighting = get_family_plugin(
        family
    ).compute_noise_pred_and_target(trainer, context, latents, {}, conditions)

    torch.testing.assert_close(target, torch.full_like(latents, 2.0))
    torch.testing.assert_close(timesteps, sigmas)
    torch.testing.assert_close(prediction, torch.zeros_like(latents))
    torch.testing.assert_close(weighting, torch.ones(2, 1, 1, 1, device=device))
