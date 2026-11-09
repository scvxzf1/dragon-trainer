"""Unverified-combination policy contract.

Audit source: docs/findings/compat_unverified_audit_20261109.md.

``compat_unverified_policy`` has three levels:

* ``strict``: origin/dev behavior (every audited site is an error).
* ``warn`` (default): only the audited C1 codes become warnings.
* ``allow``: audited C2/C3 codes become warnings too, with an explicit note
  that the key is unwired in the family (or departs from the official recipe).

Input-validation (B) and semantic-conflict (A / B+C2 mixed) codes stay errors
under every policy.
"""

from __future__ import annotations

import argparse

import pytest

from library.training.cli_args import add_dit_training_arguments
from library.training.compat_matrix import (
    COMPAT_UNVERIFIED_POLICIES,
    DEFAULT_COMPAT_UNVERIFIED_POLICY,
    HARD_BLOCK_CODES,
    RELAXABLE_UNWIRED_CODES,
    UNVERIFIED_C1_CODES,
    UNVERIFIED_C2_CODES,
    UNVERIFIED_C3_CODES,
    check_training_compat,
    resolve_unverified_policy,
)
from library.training.metadata import build_training_metadata


def _codes(items) -> set[str]:
    return {item.code for item in items}


C1_TRIGGERS = {
    "qwen_image_2_1_mixed_precision": {
        "model_family": "qwen_image_2_1",
        "mixed_precision": "fp16",
    },
    "qwen_image_2_1_masked_loss": {
        "model_family": "qwen_image_2_1",
        "masked_loss": True,
    },
    "qwen_image_2_1_loss_type": {
        "model_family": "qwen_image_2_1",
        "loss_type": "huber",
    },
    "qwen_image_2_1_weighting_scheme": {
        "model_family": "qwen_image_2_1",
        "weighting_scheme": "cosmap",
    },
    "z_image_loss_type": {"model_family": "z_image", "loss_type": "huber"},
    "z_image_flow_shift": {"model_family": "z_image", "discrete_flow_shift": 3.0},
    "z_image_weighting_scheme": {"model_family": "z_image", "weighting_scheme": "cosmap"},
}

# Representative C2 (no code path in the family) and C3 (recipe/VRAM policy)
# sites; ``warn`` keeps them as errors and ``allow`` relaxes them.
C2_TRIGGERS = {
    "qwen_image_2_1_weighted_captions": {
        "model_family": "qwen_image_2_1",
        "weighted_captions": True,
    },
    "z_image_torch_compile": {"model_family": "z_image", "torch_compile": True},
    "krea2_selective_checkpoint": {
        "model_family": "krea2_raw",
        "selective_checkpoint": "mlp_only",
    },
}
C3_TRIGGERS = {
    "z_image_bf16_only": {
        "model_family": "z_image",
        "mixed_precision": "fp16",
        "attn_mode": "torch",
    },
    "block_swap_cpu_offload": {
        "blocks_to_swap": 4,
        "cpu_offload_checkpointing": True,
        "gradient_checkpointing": True,
    },
}

# A (semantic conflict), B (input validation) and B+C2 mixed sites.
HARD_BLOCK_TRIGGERS = {
    "selective_full_gradient_checkpointing": {
        "selective_checkpoint": "mlp_only",
        "gradient_checkpointing": True,
    },
    "selective_cpu_offload": {
        "selective_checkpoint": "mlp_only",
        "cpu_offload_checkpointing": True,
    },
    "selective_unsloth_offload": {
        "selective_checkpoint": "mlp_only",
        "unsloth_offload_checkpointing": True,
    },
    "unsloth_cpu_offload": {
        "unsloth_offload_checkpointing": True,
        "cpu_offload_checkpointing": True,
    },
    "convrot_block_swap_int8_mutex": {
        "base_compute": "w8a16_convrot",
        "block_swap_transfer_dtype": "int8",
    },
    "block_swap_soft_tokens": {
        "blocks_to_swap": 8,
        "network_module": "networks.methods.soft_tokens",
    },
    "block_swap_functional_loss": {
        "blocks_to_swap": 8,
        "functional_loss_weight": 0.1,
    },
    "invalid_base_compute": {"base_compute": "zzz"},
    "negative_blocks_to_swap": {"blocks_to_swap": -1},
    "invalid_model_family": {"model_family": "zzz"},
    "krea2_invalid_attn_mode": {"model_family": "krea2_raw", "attn_mode": "xformers"},
    "z_image_attention_mode": {
        "model_family": "z_image",
        "attn_mode": "flash",
        "xformers": True,
    },
    "pipeline_parallel_config": {
        "pipeline_parallel": True,
        "pipeline_parallel_stages": 3,
    },
    "sample_vae_decode_contract": {"sample_vae_decode": "invalid"},
    "auto_block_swap_contract": {
        "auto_block_swap": True,
        "network_module": "networks.methods.ip_adapter",
    },
}


def test_audit_tables_are_disjoint_and_cover_every_classified_code() -> None:
    assert DEFAULT_COMPAT_UNVERIFIED_POLICY == "warn"
    assert COMPAT_UNVERIFIED_POLICIES == ("strict", "warn", "allow")
    # The seven audited C1 codes; relaxing anything else under ``warn`` is a bug.
    assert UNVERIFIED_C1_CODES == frozenset({
        "qwen_image_2_1_mixed_precision",
        "qwen_image_2_1_masked_loss",
        "qwen_image_2_1_loss_type",
        "qwen_image_2_1_weighting_scheme",
        "z_image_loss_type",
        "z_image_flow_shift",
        "z_image_weighting_scheme",
    })
    assert len(UNVERIFIED_C2_CODES) == 38
    assert len(UNVERIFIED_C3_CODES) == 5
    assert len(HARD_BLOCK_CODES) == 25
    assert RELAXABLE_UNWIRED_CODES == UNVERIFIED_C2_CODES | UNVERIFIED_C3_CODES
    groups = [UNVERIFIED_C1_CODES, UNVERIFIED_C2_CODES, UNVERIFIED_C3_CODES, HARD_BLOCK_CODES]
    for index, group in enumerate(groups):
        for other in groups[index + 1:]:
            assert not group & other
    # 77 audited codes minus the two always-false sites deleted in this branch.
    assert sum(len(group) for group in groups) == 75
    assert not RELAXABLE_UNWIRED_CODES & HARD_BLOCK_CODES


def test_every_c1_trigger_reproduces_its_code_under_strict() -> None:
    for code, config in C1_TRIGGERS.items():
        result = check_training_compat(config, unverified_policy="strict")
        assert code in _codes(result.errors), code
        assert code not in _codes(result.warnings), code


@pytest.mark.parametrize("code", sorted(C1_TRIGGERS))
def test_default_warn_relaxes_c1_with_marker_and_reason(code: str) -> None:
    result = check_training_compat(C1_TRIGGERS[code])

    assert result.unverified_policy == "warn"
    assert code not in _codes(result.errors)
    [issue] = [item for item in result.warnings if item.code == code]
    assert "未验证" in issue.message
    assert "Reason:" in issue.message
    assert "compat_unverified_policy=warn" in issue.message


@pytest.mark.parametrize("code", sorted(C1_TRIGGERS))
def test_strict_restores_the_historical_error(code: str) -> None:
    config = C1_TRIGGERS[code]
    keyword = check_training_compat(config, unverified_policy="strict")
    config_key = check_training_compat({**config, "compat_unverified_policy": "strict"})

    for result in (keyword, config_key):
        assert code in _codes(result.errors)
        [issue] = [item for item in result.errors if item.code == code]
        # strict keeps the origin/dev message verbatim (no relaxation marker).
        assert "未验证" not in issue.message


def test_explicit_keyword_overrides_the_config_key() -> None:
    config = {**C1_TRIGGERS["z_image_loss_type"], "compat_unverified_policy": "strict"}

    assert "z_image_loss_type" in _codes(check_training_compat(config).errors)
    assert "z_image_loss_type" not in _codes(
        check_training_compat(config, unverified_policy="allow").errors
    )


@pytest.mark.parametrize("code", sorted(C2_TRIGGERS))
def test_warn_keeps_c2_errors_and_allow_relaxes_them(code: str) -> None:
    config = C2_TRIGGERS[code]

    assert code in _codes(check_training_compat(config).errors)
    [issue] = [
        item
        for item in check_training_compat(config, unverified_policy="allow").warnings
        if item.code == code
    ]
    assert "未验证" in issue.message
    assert "该键在本族未接线/会被忽略" in issue.message
    assert code not in _codes(check_training_compat(config, unverified_policy="allow").errors)


@pytest.mark.parametrize("code", sorted(C3_TRIGGERS))
def test_allow_relaxes_c3_with_unwired_note(code: str) -> None:
    config = C3_TRIGGERS[code]

    assert code in _codes(check_training_compat(config).errors)
    allow = check_training_compat(config, unverified_policy="allow")
    [issue] = [item for item in allow.warnings if item.code == code]
    assert "该键在本族未接线/会被忽略" in issue.message


@pytest.mark.parametrize("code", sorted(HARD_BLOCK_TRIGGERS))
def test_hard_blocks_stay_errors_under_every_policy(code: str) -> None:
    config = HARD_BLOCK_TRIGGERS[code]

    for policy in COMPAT_UNVERIFIED_POLICIES:
        result = check_training_compat(config, unverified_policy=policy)
        assert code in _codes(result.errors), (code, policy)
        assert code not in _codes(result.warnings), (code, policy)


def test_invalid_policy_value_is_a_hard_error() -> None:
    result = check_training_compat({"compat_unverified_policy": "bogus"})
    assert "invalid_compat_unverified_policy" in _codes(result.errors)
    # An explicit override wins over the config key, so the typo is only
    # reported when the config key is actually consulted.
    override = check_training_compat(
        {"compat_unverified_policy": "bogus"}, unverified_policy="allow"
    )
    assert not override.errors


def test_resolve_unverified_policy_aliases_and_defaults() -> None:
    assert resolve_unverified_policy({}) == ("warn", None)
    assert resolve_unverified_policy(None, None) == ("warn", None)
    assert resolve_unverified_policy({"compat_unverified_policy": "error"}) == ("strict", None)
    assert resolve_unverified_policy({"compat_unverified_policy": "WARN"}) == ("warn", None)
    policy, message = resolve_unverified_policy({"compat_unverified_policy": "bogus"})
    assert policy == "warn"
    assert message is not None and "bogus" in message


def test_cli_exposes_the_policy_with_warn_default() -> None:
    parser = argparse.ArgumentParser()
    add_dit_training_arguments(parser)

    assert parser.parse_args([]).compat_unverified_policy == "warn"
    for value in COMPAT_UNVERIFIED_POLICIES:
        assert parser.parse_args(["--compat_unverified_policy", value]).compat_unverified_policy == value
    with pytest.raises(SystemExit):
        parser.parse_args(["--compat_unverified_policy", "bogus"])


class _MetadataArgs:
    """Minimal args surface for metadata stamping (any other key reads as 0)."""

    def __init__(self, **values):
        self.__dict__.update(values)

    def __getattr__(self, name):
        return 0


def _metadata(**values):
    return build_training_metadata(
        _MetadataArgs(**values),
        session_id=1,
        training_started_at=0.0,
        text_encoder_lr=0.0,
        optimizer_name="adamw",
        optimizer_args="",
        model_version="test",
        num_train_images=1,
        num_val_images=0,
        num_reg_images=0,
        num_batches_per_epoch=1,
        num_train_epochs=1,
    )


def test_metadata_stamps_the_policy() -> None:
    assert _metadata()["ss_compat_unverified_policy"] == "warn"
    assert _metadata(compat_unverified_policy="strict")["ss_compat_unverified_policy"] == "strict"


def test_message_fixes_match_the_implemented_aliases() -> None:
    # Krea-2 accepts sdpa/cudnn/cudnn_sdpa as torch aliases.
    for mode in ("sdpa", "cudnn", "cudnn_sdpa"):
        result = check_training_compat({"model_family": "krea2_raw", "attn_mode": mode})
        assert "krea2_invalid_attn_mode" not in _codes(result.errors)
    krea2_error = check_training_compat(
        {"model_family": "krea2_raw", "attn_mode": "xformers"}
    ).errors
    [issue] = [item for item in krea2_error if item.code == "krea2_invalid_attn_mode"]
    assert "cudnn" in issue.message

    # Z-Image and Qwen normalize native -> torch.
    for family in ("z_image", "qwen_image_2_1"):
        result = check_training_compat({"model_family": family, "attn_mode": "native"})
        assert "attn_mode" not in {item.key for item in result.errors}

    # invalid_base_compute lists the whole whitelist.
    [issue] = check_training_compat({"base_compute": "zzz"}).errors
    for value in ("bf16", "fp16", "none", "off", "w8a16_convrot", "w8a8_convrot", "w8a16", "w8a8", "nf4"):
        assert value in issue.message

    # Z-Image base_compute distinguishes runtime NF4 from pre-quantized mixed NF4.
    [issue] = check_training_compat(
        {"model_family": "z_image", "base_compute": "nf4"}
    ).errors
    assert "pre-quantized mixed-NF4" in issue.message


STRICT_PARITY_BATTERY = {
    'empty': [],
    'unsloth_cpu': [
        'unsloth_cpu_offload',
    ],
    'soft_tokens_functional': [
        'block_swap_functional_loss',
        'block_swap_soft_tokens',
    ],
    'selective_conflicts': [
        'selective_cpu_offload',
        'selective_full_gradient_checkpointing',
        'selective_unsloth_offload',
        'unsloth_cpu_offload',
    ],
    'convrot_int8': [
        'convrot_block_swap_int8_mutex',
    ],
    'bad_base_compute': [
        'invalid_base_compute',
    ],
    'negative_swap': [
        'negative_blocks_to_swap',
    ],
    'swap_cpu_offload': [
        'block_swap_cpu_offload',
    ],
    'swap_unsloth_offload': [
        'block_swap_unsloth_offload',
    ],
    'bad_selective': [
        'invalid_selective_checkpoint',
    ],
    'invalid_family': [
        'invalid_model_family',
    ],
    'pipeline_parallel_config': [
        'pipeline_parallel_config',
    ],
    'auto_block_swap_contract': [
        'auto_block_swap_contract',
    ],
    'krea2_family_kitchen_sink': [
        'krea2_compile_inductor_mode',
        'krea2_invalid_attn_mode',
        'krea2_selective_checkpoint',
        'krea2_v100_flash_stability',
    ],
    'z_image_kitchen_sink': [
        'z_image_attention_mode',
        'z_image_base_compute',
        'z_image_bf16_only',
        'z_image_caption_dropout',
        'z_image_flow_shift',
        'z_image_layer_range',
        'z_image_loss_type',
        'z_image_selective_checkpoint',
        'z_image_timestep_range',
        'z_image_timestep_sampling',
        'z_image_torch_compile',
        'z_image_training_sampler',
        'z_image_v100_flash_stability',
        'z_image_weighted_captions',
        'z_image_weighting_scheme',
    ],
    'z_image_c1_only': [
        'z_image_flow_shift',
        'z_image_loss_type',
        'z_image_weighting_scheme',
    ],
    'qwen_c1_only': [
        'qwen_image_2_1_loss_type',
        'qwen_image_2_1_masked_loss',
        'qwen_image_2_1_mixed_precision',
        'qwen_image_2_1_weighting_scheme',
    ],
    'qwen_kitchen_sink': [
        'adaptive_precision_contract',
        'auto_block_swap_contract',
        'block_swap_cpu_offload',
        'qwen_image_2_1_adaptive_precision',
        'qwen_image_2_1_attn_mode',
        'qwen_image_2_1_auto_block_swap',
        'qwen_image_2_1_base_compute',
        'qwen_image_2_1_caption_dropout_rate',
        'qwen_image_2_1_compile_mode',
        'qwen_image_2_1_cpu_offload_checkpointing',
        'qwen_image_2_1_layer_start',
        'qwen_image_2_1_loss_type',
        'qwen_image_2_1_masked_loss',
        'qwen_image_2_1_network_train_unet_only',
        'qwen_image_2_1_preview_text_cache',
        'qwen_image_2_1_selective_checkpoint',
        'qwen_image_2_1_t_min',
        'qwen_image_2_1_timestep_sampling',
        'qwen_image_2_1_weighted_captions',
        'qwen_image_2_1_weighting_scheme',
        'selective_cpu_offload',
        'selective_full_gradient_checkpointing',
    ],
    'qwen_edit_lokr': [
        'family_plain_lora_only',
        'qwen_image_2_1_edit_latent_cache',
        'qwen_image_2_1_edit_subsets',
        'qwen_image_2_1_edit_text_cache',
    ],
    'qwen_fused_saved_projection': [
        'qwen_fused_projections_block_swap_transfer_dtype',
        'qwen_saved_projection_contract',
    ],
    'qwen_fused_non_qwen': [
        'qwen_fused_projections_family',
        'qwen_fused_projections_gradient_checkpointing',
    ],
    'qwen_fused_bad_mode': [
        'invalid_qwen_fused_projections',
    ],
    'auto_block_swap_adapter': [
        'auto_block_swap_contract',
    ],
    'sample_vae_decode': [
        'sample_vae_decode_contract',
    ],
    'qwen_preview_memory_on_anima': [
        'qwen_preview_memory_contract',
    ],
    'qwen_nf4_compile_experiment': [
        'qwen_nf4_compile_experiment',
    ],
    'qwen_compile_backend': [
        'qwen_image_2_1_compile_backend',
    ],
    'qwen_fused_contract_variants': [
        'qwen_fused_projections_base_compute',
        'qwen_fused_projections_cpu_offload_checkpointing',
        'qwen_fused_projections_gradient_checkpointing',
        'qwen_fused_projections_mixed_precision',
        'qwen_fused_projections_selective_checkpoint',
        'qwen_image_2_1_base_compute',
        'qwen_image_2_1_cpu_offload_checkpointing',
        'qwen_image_2_1_mixed_precision',
        'qwen_image_2_1_selective_checkpoint',
        'selective_cpu_offload',
    ],
    'qwen_auto_block_swap': [
        'auto_block_swap_contract',
        'qwen_image_2_1_auto_block_swap',
        'qwen_image_2_1_block_swap_checkpoint',
    ],
    'qwen_blocks31': [
        'qwen_image_2_1_blocks_to_swap',
    ],
    'qwen_fused_lokr': [
        'qwen_fused_projections_plain_lora',
    ],
    'z_image_swap29': [
        'z_image_block_swap_range',
    ],
    'z_image_offload': [
        'z_image_checkpoint_offload',
    ],
    'pp_krea2': [
        'pipeline_parallel_runtime_unavailable',
    ],
    'qwen_invalid_saved_blocks': [
        'invalid_qwen_projection_budget_mib',
        'invalid_qwen_saved_projection_blocks',
    ],
}

def _battery() -> dict[str, dict]:
    """Configs used to prove strict-mode parity with origin/dev (2026-11-09)."""

    return {
        "empty": {},
        "unsloth_cpu": {
            "unsloth_offload_checkpointing": True,
            "cpu_offload_checkpointing": True,
            "gradient_checkpointing": False,
        },
        "soft_tokens_functional": {
            "blocks_to_swap": 8,
            "network_module": "networks.methods.soft_tokens",
            "functional_loss_weight": 0.1,
        },
        "selective_conflicts": {
            "selective_checkpoint": "mlp_only",
            "gradient_checkpointing": True,
            "cpu_offload_checkpointing": True,
            "unsloth_offload_checkpointing": True,
        },
        "convrot_int8": {
            "base_compute": "w8a16_convrot",
            "block_swap_transfer_dtype": "int8",
        },
        "bad_base_compute": {"base_compute": "zzz"},
        "negative_swap": {"blocks_to_swap": -1},
        "swap_cpu_offload": {
            "blocks_to_swap": 4,
            "cpu_offload_checkpointing": True,
            "gradient_checkpointing": True,
        },
        "swap_unsloth_offload": {
            "blocks_to_swap": 4,
            "unsloth_offload_checkpointing": True,
            "gradient_checkpointing": True,
        },
        "bad_selective": {"selective_checkpoint": "zzz"},
        "invalid_family": {"model_family": "zzz"},
        "pipeline_parallel_config": {
            "pipeline_parallel": True,
            "pipeline_parallel_stages": 3,
        },
        "auto_block_swap_contract": {
            "auto_block_swap": True,
            "auto_block_swap_ram_ratio": 9.0,
        },
        "krea2_family_kitchen_sink": {
            "model_family": "krea2_raw",
            "attn_mode": "xformers",
            "compile_inductor_mode": "reduce-overhead",
            "selective_checkpoint": "mlp_only",
            "v100_flash_stability": "safe",
        },
        "z_image_kitchen_sink": {
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
            "t_min": 0.1,
            "sampler": "custom",
            "v100_flash_stability": "conservative",
            "sample_every_n_steps": 10,
        },
        "z_image_c1_only": {
            "model_family": "z_image",
            "network_module": "networks.lora_anima",
            "mixed_precision": "bf16",
            "base_compute": "bf16",
            "attn_mode": "torch",
            "discrete_flow_shift": 3.0,
            "weighting_scheme": "cosmap",
            "loss_type": "huber",
        },
        "qwen_c1_only": {
            "model_family": "qwen_image_2_1",
            "network_module": "networks.lora_anima",
            "mixed_precision": "fp16",
            "masked_loss": True,
            "loss_type": "huber",
            "weighting_scheme": "cosmap",
        },
        "qwen_kitchen_sink": {
            "model_family": "qwen_image_2_1",
            "network_module": "networks.lora_anima",
            "base_compute": "nf4",
            "adaptive_precision": "auto",
            "attn_mode": "xformers",
            "blocks_to_swap": 31,
            "gradient_checkpointing": True,
            "torch_compile": True,
            "compile_inductor_mode": "reduce-overhead",
            "auto_block_swap": True,
            "selective_checkpoint": "mlp_only",
            "cpu_offload_checkpointing": True,
            "network_train_unet_only": False,
            "weighted_captions": True,
            "caption_dropout_rate": 0.1,
            "masked_loss": True,
            "loss_type": "huber",
            "timestep_sampling": "shift",
            "weighting_scheme": "cosmap",
            "sample_at_first": True,
            "layer_start": 3,
            "t_min": 0.1,
        },
        "qwen_edit_lokr": {
            "model_family": "qwen_image_2_1",
            "qwen_image_2_1_task": "edit",
            "network_module": "networks.lora_anima",
            "use_lokr": True,
        },
        "qwen_fused_saved_projection": {
            "model_family": "qwen_image_2_1",
            "qwen_fused_projections": "all",
            "qwen_saved_projection_blocks": 2,
            "qwen_projection_budget_mib": 0,
            "block_swap_transfer_dtype": "int8",
            "gradient_checkpointing": True,
        },
        "qwen_fused_non_qwen": {
            "model_family": "anima",
            "qwen_fused_projections": "all",
        },
        "qwen_fused_bad_mode": {
            "model_family": "qwen_image_2_1",
            "qwen_fused_projections": "invalid",
        },
        "qwen_invalid_saved_blocks": {
            "model_family": "qwen_image_2_1",
            "qwen_saved_projection_blocks": -2,
            "qwen_projection_budget_mib": "x",
        },
        "auto_block_swap_adapter": {
            "auto_block_swap": True,
            "network_module": "networks.methods.ip_adapter",
        },
        "sample_vae_decode": {"sample_vae_decode": "invalid"},
        "qwen_preview_memory_on_anima": {
            "model_family": "anima",
            "qwen_sample_swap_mode": "custom",
            "qwen_sample_blocks_to_swap": 4,
        },
        "qwen_nf4_compile_experiment": {
            "model_family": "qwen_image_2_1",
            "qwen_nf4_compile_experiment": True,
        },
        "qwen_compile_backend": {
            "model_family": "qwen_image_2_1",
            "torch_compile": True,
            "dynamo_backend": "cudagraphs",
        },
        "qwen_fused_contract_variants": {
            "model_family": "qwen_image_2_1",
            "qwen_fused_projections": "all",
            "mixed_precision": "fp16",
            "base_compute": "w8a16_convrot",
            "selective_checkpoint": "mlp_only",
            "cpu_offload_checkpointing": True,
            "gradient_checkpointing": False,
        },
        "qwen_auto_block_swap": {
            "model_family": "qwen_image_2_1",
            "auto_block_swap": True,
        },
        "qwen_blocks31": {
            "model_family": "qwen_image_2_1",
            "blocks_to_swap": 31,
            "gradient_checkpointing": True,
        },
        "qwen_fused_lokr": {
            "model_family": "qwen_image_2_1",
            "qwen_fused_projections": "all",
            "use_lokr": True,
            "gradient_checkpointing": True,
        },
        "z_image_swap29": {
            "model_family": "z_image",
            "blocks_to_swap": 29,
            "gradient_checkpointing": True,
        },
        "z_image_offload": {
            "model_family": "z_image",
            "cpu_offload_checkpointing": True,
            "gradient_checkpointing": True,
        },
        "pp_krea2": {
            "model_family": "krea2_raw",
            "pipeline_parallel": True,
            "pipeline_parallel_stages": 2,
            "pipeline_parallel_microbatches": 2,
            "pipeline_parallel_schedule": "1f1b",
            "pipeline_parallel_balance": "balanced",
            "network_train_unet_only": True,
            "blocks_to_swap": 0,
            "torch_compile": False,
            "selective_checkpoint": "off",
        },
    }




def test_strict_error_codes_match_origin_dev_battery() -> None:
    """Every strict-mode error set equals the pre-change (origin/dev) snapshot."""

    battery = _battery()
    assert set(battery) == set(STRICT_PARITY_BATTERY)
    for label, config in battery.items():
        result = check_training_compat(config, world_size=2, unverified_policy="strict")
        assert sorted(_codes(result.errors)) == STRICT_PARITY_BATTERY[label], label
