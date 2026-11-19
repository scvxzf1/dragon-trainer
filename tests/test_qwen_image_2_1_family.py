from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.models.family_registry import get_model_family_spec
from library.models.qwen_image_2_1.family import forward_for_loss, sample_sigmas
from library.models.qwen_image_2_1.latent import encode_qwen_image_2_1_latents
from library.models.qwen_image_2_1.lora_targets import qwen_image_2_1_target_kwargs
from library.models.qwen_image_2_1.strategy import QwenImage21LatentCache
from library.models.qwen_image_2_1.weights import _text_key, _vae_key, _vae_shape
from library.training.compat_matrix import check_training_compat


def test_registry_isolated_cache_and_supported_adapters() -> None:
    spec = get_model_family_spec("qwen_image_2_1")
    assert spec.latent_space.latent_channels == 64
    assert spec.latent_space.cache_suffix == "_qwen_image_2_1.npz"
    assert spec.text_cache.hidden_width == 4096
    assert spec.supported_network_specs == frozenset({"lora", "lokr", "loha"})
    assert spec.supported_tasks == frozenset({"t2i", "edit"})
    assert spec.plain_lora_only
    assert spec.supported_attention_modes == frozenset({"torch", "sdpa", "flash"})
    assert spec.flash_runtime_dtypes == frozenset({"bf16"})
    assert spec.supported_inference_modes == frozenset({"single"})
    assert spec.supported_inference_samplers == frozenset({"euler"})


def test_comfy_single_file_key_conversion() -> None:
    assert _text_key("model.layers.0.self_attn.q_proj.weight") == (
        "model.language_model.layers.0.self_attn.q_proj.weight"
    )
    assert _text_key("model.visual.patch_embed.proj.weight") == "model.visual.patch_embed.proj.weight"
    assert _vae_key("encoder.downsamples.2.downsamples.0.residual.2.weight") == (
        "encoder.down_blocks.2.resnets.0.conv1.weight"
    )
    assert _vae_key("decoder.upsamples.3.upsamples.3.resample.1.weight") == (
        "decoder.up_blocks.3.upsampler.resample.1.weight"
    )
    assert _vae_shape((4, 8, 1, 3, 3)) == (4, 8, 3, 3)
    assert _vae_shape((4, 8, 3, 3, 3)) == (4, 8, 3, 3, 3)


def test_rgba_vae_normalization() -> None:
    class FakeVAE:
        config = SimpleNamespace(latents_mean=[0.5] * 64, latents_std=[2.0] * 64)

        def encode(self, pixels):
            assert pixels.shape == (1, 4, 1, 32, 32)
            torch.testing.assert_close(pixels[:, 3], torch.ones_like(pixels[:, 3]))
            return SimpleNamespace(latent_dist=SimpleNamespace(
                mode=lambda: torch.full((1, 64, 1, 2, 2), 2.5)
            ))

    latents = encode_qwen_image_2_1_latents(FakeVAE(), torch.zeros(1, 3, 32, 32))
    assert latents.shape == (1, 64, 2, 2)
    torch.testing.assert_close(latents, torch.ones_like(latents))


def test_dit_forward_layout_and_gradient() -> None:
    class FakeDiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, **kwargs):
            assert kwargs["hidden_states"].shape == (2, 24, 64)
            assert kwargs["img_shapes"] == [[(1, 4, 6)], [(1, 4, 6)]]
            assert kwargs["img_mask"].shape == (2, 11)
            assert not kwargs["img_mask"][:, :5].any()
            assert kwargs["img_mask"][:, 5:].all()
            assert kwargs["encoder_hidden_states_mask"].dtype == torch.bool
            prefix = kwargs["hidden_states"].new_zeros(2, 5, 64)
            return (torch.cat([prefix, kwargs["hidden_states"]], dim=1) * self.scale,)

    dit = FakeDiT()
    latents = torch.ones(2, 64, 4, 6)
    hiddens = torch.zeros(2, 5, 4096)
    mask = torch.ones(2, 5, dtype=torch.bool)
    prediction = forward_for_loss(dit, latents, hiddens, mask, torch.tensor([0.2, 0.8]))
    assert prediction.shape == latents.shape
    prediction.mean().backward()
    assert dit.scale.grad is not None and dit.scale.grad.item() == pytest.approx(1.0)


def test_edit_dit_forward_packs_reference_then_target_and_returns_target_tail() -> None:
    class FakeEditDiT(torch.nn.Module):
        def forward(self, **kwargs):
            assert kwargs["hidden_states"].shape == (1, 32, 64)
            assert kwargs["img_shapes"] == [[(1, 2, 4), (1, 4, 6)]]
            assert kwargs["img_mask"].shape == (1, 11)
            assert kwargs["img_mask"][0].sum().item() == 8
            assert kwargs["encoder_hidden_states_mask"].shape == (1, 5)
            seq_len = 5 - 2 + 32
            output = torch.arange(seq_len, dtype=kwargs["hidden_states"].dtype).view(1, seq_len, 1)
            return (output.expand(1, seq_len, 64),)

    target = torch.zeros(1, 64, 4, 6)
    reference = torch.ones(1, 64, 2, 4)
    hiddens = torch.zeros(1, 5, 4096)
    mask = torch.ones(1, 5, dtype=torch.bool)
    slots = torch.tensor([[False, True, False, True, False]])
    prediction = forward_for_loss(
        FakeEditDiT(), target, hiddens, mask, torch.tensor([0.5]),
        reference_latents=reference, image_slot_mask=slots,
    )
    expected = torch.arange(11, 35, dtype=prediction.dtype).view(1, 1, 4, 6).expand_as(prediction)
    torch.testing.assert_close(prediction, expected)


def test_multi_edit_dit_preserves_reference_shapes_order_and_target_tail() -> None:
    class FakeEditDiT(torch.nn.Module):
        def forward(self, **kwargs):
            assert kwargs["hidden_states"].shape == (1, 56, 64)
            assert kwargs["img_shapes"] == [[(1, 2, 4), (1, 4, 6), (1, 4, 6)]]
            assert kwargs["img_mask"].shape == (1, 18)
            assert kwargs["img_mask"].sum().item() == 14
            torch.testing.assert_close(kwargs["hidden_states"][0, :8], torch.ones(8, 64))
            torch.testing.assert_close(kwargs["hidden_states"][0, 8:32], torch.full((24, 64), 2.0))
            sequence = torch.arange(12 - 8 + 56, dtype=torch.float32).view(1, -1, 1)
            return (sequence.expand(1, -1, 64),)

    target = torch.zeros(1, 64, 4, 6)
    references = [torch.ones(1, 64, 2, 4), torch.full((1, 64, 4, 6), 2.0)]
    hiddens = torch.zeros(1, 12, 4096)
    mask = torch.ones(1, 12, dtype=torch.bool)
    # 2x4 and 4x6 references contribute 2+6 image slots.
    slots = torch.tensor([[False] + [True] * 8 + [False] * 3])
    prediction = forward_for_loss(
        FakeEditDiT(), target, hiddens, mask, torch.tensor([0.5]),
        reference_latents=references, image_slot_mask=slots,
    )
    expected = torch.arange(36, 60, dtype=prediction.dtype).view(1, 1, 4, 6).expand_as(prediction)
    torch.testing.assert_close(prediction, expected)


def test_multi_edit_rejects_aggregate_slot_mismatch() -> None:
    with pytest.raises(ValueError, match="image-slot count"):
        forward_for_loss(
            torch.nn.Identity(), torch.zeros(1, 64, 4, 6), torch.zeros(1, 9, 4096),
            torch.ones(1, 9, dtype=torch.bool), torch.tensor([0.5]),
            reference_latents=(torch.zeros(1, 64, 2, 4), torch.zeros(1, 64, 4, 6)),
            image_slot_mask=torch.tensor([[True] * 7 + [False] * 2]),
        )


def test_edit_dit_rejects_output_with_unexpected_sequence_length() -> None:
    class BadDiT(torch.nn.Module):
        def forward(self, **_kwargs):
            return (torch.zeros(1, 34, 64),)

    with pytest.raises(ValueError, match="DiT output shape mismatch"):
        forward_for_loss(
            BadDiT(), torch.zeros(1, 64, 4, 6), torch.zeros(1, 5, 4096),
            torch.ones(1, 5, dtype=torch.bool), torch.tensor([0.5]),
            reference_latents=torch.zeros(1, 64, 2, 4),
            image_slot_mask=torch.tensor([[False, True, False, True, False]]),
        )


def test_plain_lora_targets_only_four_attention_linears() -> None:
    from diffusers import QwenImage21Transformer2DModel
    from networks.lora_anima.targeting import (
        collect_lora_target_candidates,
        compile_lora_target_patterns,
    )

    model = QwenImage21Transformer2DModel(
        num_layers=1, num_attention_heads=1, attention_head_dim=128,
        context_in_dim=128, mlp_ratio=1,
    )
    spec = qwen_image_2_1_target_kwargs()
    targets = collect_lora_target_candidates(
        root_module=model, prefix="lora_unet",
        target_replace_modules=spec["unet_target_replace_modules"],
        exclude_patterns=compile_lora_target_patterns(spec["exclude_patterns"]),
        include_patterns=[], is_unet=True, layer_start=None, layer_end=None,
        modules_dim=None, modules_alpha=None, reg_dims=None,
        default_dim=None, lora_dim=4, alpha=4,
    )
    assert {item.original_name for item in targets} == {
        f"transformer_blocks.0.attn.{name}"
        for name in ("to_q", "to_k", "to_v", "to_out.0")
    }


def test_small_official_transformer_cpu_forward_backward() -> None:
    from diffusers import QwenImage21Transformer2DModel

    model = QwenImage21Transformer2DModel(
        num_layers=1, num_attention_heads=1, attention_head_dim=128,
        context_in_dim=4096, mlp_ratio=1,
    )
    latents = torch.randn(1, 64, 2, 2)
    hiddens = torch.randn(1, 3, 4096)
    mask = torch.tensor([[True, True, False]])
    prediction = forward_for_loss(model, latents, hiddens, mask, torch.tensor([0.5]))
    assert prediction.shape == latents.shape
    prediction.square().mean().backward()
    assert model.transformer_blocks[0].attn.to_q.weight.grad is not None


def test_dynamic_sigma_in_range() -> None:
    sigmas = sample_sigmas(128, 4096, device=torch.device("cpu"))
    assert sigmas.shape == (128,)
    assert torch.all((sigmas > 0) & (sigmas < 1))
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch, "rand", lambda size, **kwargs: torch.full((size,), 0.5, **kwargs))
        at_max = sample_sigmas(1, 4096, device=torch.device("cpu"))
    assert at_max.item() == pytest.approx(torch.sigmoid(torch.tensor(1.15)).item())


def test_latent_cache_uses_sixteen_pixel_stride(monkeypatch) -> None:
    strategy = QwenImage21LatentCache(True, 1, False)
    called = []
    monkeypatch.setattr(strategy, "_default_is_disk_cached_latents_expected",
                        lambda *args, **kwargs: called.append(strategy.latents_stride) or True)
    monkeypatch.setattr(strategy, "_default_load_latents_from_disk",
                        lambda *args: called.append(strategy.latents_stride))
    assert strategy.is_disk_cached_latents_expected((32, 32), "cache.npz", False, False)
    strategy.load_latents_from_disk("cache.npz", (32, 32))
    assert called == [16, 16]


def test_web_cache_audit_uses_qwen_strategies() -> None:
    from web.services.config.cache_audit import _latent_strategy, _text_strategy

    assert isinstance(_latent_strategy("qwen_image_2_1", False), QwenImage21LatentCache)
    from library.models.qwen_image_2_1.strategy import QwenImage21TextCache

    assert isinstance(_text_strategy("qwen_image_2_1", False), QwenImage21TextCache)


@pytest.mark.parametrize("bad", [
    {"base_compute": "nf4"},
    {"blocks_to_swap": 31, "gradient_checkpointing": True},
    {"blocks_to_swap": -1, "gradient_checkpointing": True},
    {"blocks_to_swap": 1, "gradient_checkpointing": False},
    {"torch_compile": True, "compile_inductor_mode": "reduce-overhead"},
    {"caption_dropout_rate": 0.1},
    {"sample_at_first": True},
    {"network_train_unet_only": False},
])
def test_unsupported_training_options_fail_closed(bad) -> None:
    config = {"model_family": "qwen_image_2_1", "network_module": "networks.lora_anima", **bad}
    assert check_training_compat(config).errors


def test_qwen_swap_and_compile_are_allowed_with_full_checkpointing() -> None:
    config = {
        "model_family": "qwen_image_2_1",
        "network_module": "networks.lora_anima",
        "gradient_checkpointing": True,
        "blocks_to_swap": 24,
        "torch_compile": True,
        "compile_dynamic_seq": True,
        "compile_block_scope": "all",
    }
    assert not check_training_compat(config).errors


def test_qwen_flash_is_allowed_with_full_checkpointing() -> None:
    config = {
        "model_family": "qwen_image_2_1",
        "network_module": "networks.lora_anima",
        "attn_mode": "flash",
        "gradient_checkpointing": True,
        "blocks_to_swap": 24,
        "torch_compile": True,
        "compile_dynamic_seq": True,
        "compile_block_scope": "all",
    }
    assert not check_training_compat(config).errors
