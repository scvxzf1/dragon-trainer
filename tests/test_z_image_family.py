from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from library.models.family_registry import get_model_family_spec
from library.models.z_image.family import (
    forward_for_loss,
    prepare_prompt_embeds,
    shifted_uniform_sigmas,
)
from library.models.z_image.lora_targets import z_image_target_kwargs
from library.models.z_image.latent import encode_z_image_latents
from library.models.z_image.strategy import (
    ZImageLatentsCachingStrategy,
    ZImageTextEncoderOutputsCachingStrategy,
    ZImageTokenizeStrategy,
)
from library.models.z_image.weights import (
    ZImageCheckpointError,
    resolve_z_image_tokenizer_path,
    validate_z_image_component_config,
)
from library.training.compat_matrix import check_training_compat
from library.training.extra_args import assert_training_extra_args
from library.training.trainer_network_mixin import TrainerNetworkMixin
from networks.lora_anima.targeting import (
    collect_lora_target_candidates,
    compile_lora_target_patterns,
)


def test_z_image_registry_uses_isolated_cache_contract() -> None:
    spec = get_model_family_spec("z_image")
    assert spec.text_cache.suffix == "_z_image_te.safetensors"
    assert spec.text_cache.schema == "z_image_te_v1"
    assert spec.text_cache.hidden_width == 2560
    assert spec.plain_lora_only is False
    assert spec.supported_network_specs is None
    assert spec.supported_attention_modes == {"torch", "flash", "sdpa"}
    assert spec.flash_runtime_dtypes == {"bf16"}


def test_z_image_compat_disables_anima_compile_seq_bands() -> None:
    result = check_training_compat(
        {
            "model_family": "z_image",
            "network_module": "networks.lora_anima",
            "mixed_precision": "bf16",
            "base_compute": "bf16",
            "attn_mode": "torch",
            "torch_compile": False,
            "compile_dynamic_seq": False,
            "compile_seq_bands": True,
            "selective_checkpoint": "off",
            "discrete_flow_shift": 6.0,
            "timestep_sampling": "uniform",
            "weighting_scheme": "none",
            "v100_flash_stability": "off",
        }
    )

    assert result.ok
    assert {item.code for item in result.warnings} == {
        "z_image_compile_seq_bands"
    }
    assert [(item.key, item.value) for item in result.mutations] == [
        ("compile_seq_bands", False)
    ]


def test_z_image_tokenizer_resolves_comfyui_assets(tmp_path) -> None:
    checkpoint = (
        tmp_path / "models" / "text_encoders" / "zimage" / "qwen_3_4b.safetensors"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    tokenizer = tmp_path / "comfy" / "text_encoders" / "qwen25_tokenizer"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    assert resolve_z_image_tokenizer_path(str(checkpoint)) == str(tokenizer)


def test_z_image_tokenizer_keeps_comfyui_root_across_models_symlink(tmp_path) -> None:
    comfy_root = tmp_path / "ComfyUI"
    shared_models = tmp_path / "shared" / "models"
    checkpoint = shared_models / "text_encoders" / "zimage" / "qwen_3_4b.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    comfy_root.mkdir()
    (comfy_root / "models").symlink_to(shared_models, target_is_directory=True)
    tokenizer = comfy_root / "comfy" / "text_encoders" / "qwen25_tokenizer"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    user_path = comfy_root / "models" / checkpoint.relative_to(shared_models)

    assert resolve_z_image_tokenizer_path(str(user_path)) == str(tokenizer)


def test_z_image_tokenizer_uses_official_user_assistant_template(monkeypatch) -> None:
    captured = {}

    class FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"

        def __call__(self, prompts, **_kwargs):
            captured["prompts"] = prompts
            return SimpleNamespace(
                input_ids=torch.ones(1, 4, dtype=torch.long),
                attention_mask=torch.ones(1, 4, dtype=torch.long),
            )

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )
    strategy = ZImageTokenizeStrategy("unused")
    strategy.tokenize("a red car")

    assert captured["prompts"] == [
        "<|im_start|>user\na red car<|im_end|>\n<|im_start|>assistant\n"
    ]
    assert "system" not in captured["prompts"][0]


def test_shifted_uniform_sigmas_applies_official_shift(monkeypatch) -> None:
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *args, **kwargs: torch.tensor([0.25, 0.5], **kwargs),
    )
    sigmas = shifted_uniform_sigmas(2, device=torch.device("cpu"), dtype=torch.float32)
    base_sigmas = torch.tensor([0.75, 0.5])
    expected = 6 * base_sigmas / (1 + 5 * base_sigmas)
    assert torch.allclose(sigmas, expected)


def test_forward_uses_one_minus_sigma_and_negates_output() -> None:
    class FakeTransformer:
        def __call__(self, *, x, t, cap_feats):
            self.x = x
            self.t = t
            self.cap_feats = cap_feats
            return SimpleNamespace(sample=[sample * 2 for sample in x])

    model = FakeTransformer()
    latents = torch.arange(2 * 4 * 1 * 2 * 2, dtype=torch.float32).reshape(
        2, 4, 1, 2, 2
    )
    prompts = [torch.ones(3, 8), torch.ones(5, 8)]
    sigmas = torch.tensor([0.2, 0.7], dtype=torch.bfloat16)

    result = forward_for_loss(model, latents, prompts, sigmas)

    assert model.t.dtype == torch.float32
    assert torch.allclose(model.t, 1.0 - sigmas.float())
    assert [tuple(value.shape) for value in model.x] == [(4, 1, 2, 2)] * 2
    assert torch.equal(result, -2 * latents)


def test_tiny_diffusers_transformer_checkpoint_forward_backward() -> None:
    from diffusers import ZImageTransformer2DModel

    torch.manual_seed(0)
    model = ZImageTransformer2DModel(
        dim=128,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=1,
        n_kv_heads=1,
        cap_feat_dim=32,
        axes_dims=[32, 48, 48],
        axes_lens=[64, 64, 64],
    )
    # Direct construction leaves learned pad tokens uninitialized; production
    # from_pretrained() replaces them with checkpoint weights.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, 0.01)
            elif name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.zero_()
    model.enable_gradient_checkpointing()
    latents = torch.randn(2, 16, 1, 4, 4, requires_grad=True)
    prompts = [torch.randn(3, 32), torch.randn(5, 32)]
    output = forward_for_loss(model, latents, prompts, torch.tensor([0.2, 0.7]))
    output.mean().backward()
    assert output.shape == latents.shape
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()


def test_prompt_mask_produces_variable_length_list_and_rejects_empty() -> None:
    hiddens = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    prompts = prepare_prompt_embeds(hiddens, mask)
    assert [tuple(prompt.shape) for prompt in prompts] == [(2, 3), (3, 3)]
    with pytest.raises(ValueError, match="empty prompt embedding"):
        prepare_prompt_embeds(hiddens[:1], torch.zeros(1, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="expects hiddens"):
        prepare_prompt_embeds(hiddens, mask[:1])


def test_text_cache_writes_z_image_schema(tmp_path, monkeypatch) -> None:
    path = tmp_path / "sample_z_image_te.safetensors"
    strategy = ZImageTextEncoderOutputsCachingStrategy(cache_to_disk=True)
    monkeypatch.setattr(
        strategy,
        "_encode_captions",
        lambda *args, **kwargs: (
            torch.zeros(1, 4, 2560, dtype=torch.bfloat16),
            torch.ones(1, 4, dtype=torch.bool),
        ),
    )
    info = SimpleNamespace(
        caption="test",
        caption_variants=None,
        cache_caption_variants=False,
        caption_dropout_rate=0.0,
        text_encoder_outputs_npz=str(path),
    )
    strategy.cache_batch_outputs(None, [], None, [info])
    assert strategy.is_disk_cached_outputs_expected(str(path))
    with safe_open(path, framework="pt") as handle:
        assert handle.metadata() == {
            "model_family": "z_image",
            "cache_schema": "z_image_te_v1",
        }


def test_latent_cache_applies_affine_once(monkeypatch) -> None:
    class LatentDist:
        def mode(self):
            return torch.tensor([[[[1.0]]]])

    class FakeVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def encode(self, _images):
            return SimpleNamespace(latent_dist=LatentDist())

    strategy = ZImageLatentsCachingStrategy(False, 1, False)
    captured = {}

    def fake_cache(encode, *_args, **_kwargs):
        captured["latents"] = encode(torch.zeros(1, 3, 8, 8))

    monkeypatch.setattr(strategy, "_default_cache_batch_latents", fake_cache)
    strategy.cache_batch_latents(FakeVAE(), [], False, False, False)
    assert captured["latents"].item() == pytest.approx((1.0 - 0.1159) * 0.3611)
    assert strategy.get_image_size_from_disk_cache_path(
        "image.png", "image_1024x0768_z_image.npz"
    ) == (1024, 768)


def test_live_training_uses_z_image_vae_contract() -> None:
    class LatentDist:
        def mode(self):
            return torch.tensor([[[[1.0]]]])

    class FakeVAE:
        def encode(self, _images):
            return SimpleNamespace(latent_dist=LatentDist())

    args = SimpleNamespace(model_family="z_image")
    latents = TrainerNetworkMixin().encode_images_to_latents(
        args, FakeVAE(), torch.zeros(1, 3, 8, 8)
    )
    assert torch.equal(latents, encode_z_image_latents(FakeVAE(), None))
    assert latents.item() == pytest.approx((1.0 - 0.1159) * 0.3611)


@pytest.mark.parametrize(
    ("component", "config"),
    [
        ("text_encoder", {"hidden_size": 2560}),
        (
            "vae",
            {
                "latent_channels": 16,
                "shift_factor": 0.1159,
                "scaling_factor": 0.3611,
            },
        ),
        ("transformer", {"in_channels": 16, "cap_feat_dim": 2560}),
    ],
)
def test_component_config_validation_accepts_training_contract(
    component: str, config: dict
) -> None:
    validate_z_image_component_config(
        SimpleNamespace(config=SimpleNamespace(**config)), component
    )


def test_component_config_validation_rejects_wrong_text_width() -> None:
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=4096))
    with pytest.raises(ZImageCheckpointError, match="hidden_size=4096"):
        validate_z_image_component_config(model, "text_encoder")


def test_official_transformer_has_136_attention_only_targets() -> None:
    from diffusers import ZImageTransformer2DModel

    with torch.device("meta"):
        model = ZImageTransformer2DModel()
    kwargs = z_image_target_kwargs()
    candidates = collect_lora_target_candidates(
        root_module=model,
        prefix="lora_unet",
        target_replace_modules=kwargs["unet_target_replace_modules"],
        exclude_patterns=compile_lora_target_patterns(kwargs["exclude_patterns"]),
        include_patterns=[],
        is_unet=True,
        layer_start=None,
        layer_end=None,
        modules_dim=None,
        modules_alpha=None,
        reg_dims=None,
        default_dim=None,
        lora_dim=16,
        alpha=16,
    )
    active = [candidate for candidate in candidates if not candidate.skipped]
    assert len(active) == 136
    assert all(".attention." in candidate.original_name for candidate in active)


def test_z_image_compat_rejects_unverified_optimizations() -> None:
    # strict keeps the historical hard errors; the default warn policy relaxes
    # only the audited C1 codes (see tests/test_compat_unverified_policy.py).
    result = check_training_compat(
        {
            "compat_unverified_policy": "strict",
            "model_family": "z_image",
            "network_module": "networks.lora_anima",
            "mixed_precision": "fp16",
            "base_compute": "nf4",
            "attn_mode": "flash",
            "xformers": True,
            "torch_compile": True,
            "blocks_to_swap": 4,
            "selective_checkpoint": "every_other",
            "discrete_flow_shift": 3.0,
            "timestep_sampling": "shift",
            "weighting_scheme": "cosmap",
            "caption_dropout_rate": 0.1,
            "weighted_captions": True,
            "layer_start": 3,
            "loss_type": "huber",
            "huber_schedule": "constant",
            "t_min": 0.1,
            "sampler": "custom",
            "v100_flash_stability": "conservative",
            "sample_every_n_steps": 10,
        }
    )
    codes = {item.code for item in result.errors}
    assert {
        "z_image_attention_mode",
        "z_image_bf16_only",
        "z_image_base_compute",
        "z_image_torch_compile",
        "z_image_selective_checkpoint",
        "z_image_flow_shift",
        "z_image_timestep_sampling",
        "z_image_weighting_scheme",
        "z_image_caption_dropout",
        "z_image_weighted_captions",
        "z_image_layer_range",
        "z_image_loss_type",
        "z_image_timestep_range",
        "z_image_training_sampler",
        "z_image_v100_flash_stability",
    } <= codes


@pytest.mark.parametrize("attn_mode", [None, "flash", "torch", "sdpa", "native"])
def test_z_image_compat_accepts_supported_attention_modes(attn_mode) -> None:
    result = check_training_compat(
        {
            "model_family": "z_image",
            "network_module": "networks.lora_anima",
            "mixed_precision": "bf16",
            "base_compute": "bf16",
            "attn_mode": attn_mode,
            "xformers": False,
            "torch_compile": False,
            "selective_checkpoint": "off",
            "discrete_flow_shift": 6.0,
            "timestep_sampling": "uniform",
            "weighting_scheme": "none",
        }
    )

    assert "z_image_attention_mode" not in {item.code for item in result.errors}


def test_z_image_compat_accepts_block_swap_and_rejects_out_of_range() -> None:
    supported = check_training_compat(
        {
            "model_family": "z_image",
            "network_module": "networks.lora_anima",
            "mixed_precision": "bf16",
            "base_compute": "bf16",
            "attn_mode": "torch",
            "gradient_checkpointing": True,
            "blocks_to_swap": 20,
            "discrete_flow_shift": 6.0,
            "timestep_sampling": "uniform",
            "weighting_scheme": "none",
        }
    )
    assert "z_image_block_swap_range" not in {item.code for item in supported.errors}

    rejected = check_training_compat({"model_family": "z_image", "blocks_to_swap": 29})
    assert "z_image_block_swap_range" in {item.code for item in rejected.errors}


def test_z_image_compat_rejects_subset_caption_dropout() -> None:
    result = check_training_compat(
        {
            "model_family": "z_image",
            "network_module": "networks.lora_anima",
            "mixed_precision": "bf16",
            "base_compute": "bf16",
            "attn_mode": "torch",
            "discrete_flow_shift": 6.0,
            "datasets": [{"subsets": [{"caption_dropout_rate": 0.25}]}],
        }
    )
    assert "z_image_caption_dropout" in {item.code for item in result.errors}


def test_runtime_rejects_dataset_subset_caption_dropout_before_training() -> None:
    args = SimpleNamespace(
        model_family="z_image",
        cache_text_encoder_outputs_to_disk=False,
        cache_text_encoder_outputs=False,
        cache_llm_adapter_outputs=False,
        network_train_unet_only=True,
        selective_checkpoint="off",
        selective_checkpoint_blocks="",
        block_swap_transfer_dtype="bf16",
        block_swap_restore_mode="slab",
    )
    subset = SimpleNamespace(caption_dropout_rate=0.25)
    group = SimpleNamespace(
        datasets=[SimpleNamespace(subsets=[subset])],
        is_text_encoder_output_cacheable=lambda **_kwargs: True,
    )
    with pytest.raises(ValueError, match="dataset subset caption_dropout_rate=0"):
        assert_training_extra_args(args, group, None)
