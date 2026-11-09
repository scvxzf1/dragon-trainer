from __future__ import annotations

import argparse

import pytest

from library.training.compat_matrix import (
    apply_training_compat_mutations,
    check_training_compat,
)
from library.training.extra_args import assert_training_extra_args


def _codes(items) -> set[str]:
    return {item.code for item in items}


def test_matrix_rejects_unsloth_with_cpu_offload_and_suggests_grad_checkpoint() -> None:
    result = check_training_compat(
        {
            "unsloth_offload_checkpointing": True,
            "cpu_offload_checkpointing": True,
            "gradient_checkpointing": False,
        }
    )

    assert "unsloth_cpu_offload" in _codes(result.errors)
    assert "unsloth_enables_gradient_checkpointing" in _codes(result.warnings)
    assert [(m.key, m.value) for m in result.mutations] == [("gradient_checkpointing", True)]


def test_matrix_rejects_block_swap_soft_tokens_and_functional_loss() -> None:
    result = check_training_compat(
        {
            "blocks_to_swap": 8,
            "network_module": "networks.methods.soft_tokens",
            "functional_loss_weight": 0.1,
        }
    )

    assert {"block_swap_soft_tokens", "block_swap_functional_loss"} <= _codes(result.errors)


def test_matrix_warns_cpu_offload_without_full_checkpointing() -> None:
    result = check_training_compat(
        {
            "cpu_offload_checkpointing": True,
            "gradient_checkpointing": False,
        }
    )

    assert result.ok
    assert "cpu_offload_without_gradient_checkpointing" in _codes(result.warnings)


def test_matrix_downgrades_block_swap_cudagraph_compile_modes() -> None:
    cudagraphs = check_training_compat(
        {
            "blocks_to_swap": 8,
            "torch_compile": True,
            "dynamo_backend": "cudagraphs",
        }
    )
    max_autotune = check_training_compat(
        {
            "blocks_to_swap": 8,
            "torch_compile": True,
            "dynamo_backend": "inductor",
            "compile_inductor_mode": "max-autotune",
        }
    )

    assert [(m.key, m.value) for m in cudagraphs.mutations] == [("torch_compile", False)]
    assert [(m.key, m.value) for m in max_autotune.mutations] == [
        ("compile_inductor_mode", "max-autotune-no-cudagraphs")
    ]


def test_apply_training_compat_mutations_supports_namespace_and_dict() -> None:
    ns = argparse.Namespace(gradient_checkpointing=False)
    mapping = {"gradient_checkpointing": False}
    result = check_training_compat({"unsloth_offload_checkpointing": True})

    apply_training_compat_mutations(ns, result)
    apply_training_compat_mutations(mapping, result)

    assert ns.gradient_checkpointing is True
    assert mapping["gradient_checkpointing"] is True


def test_trainer_uses_shared_matrix_for_compile_mutations() -> None:
    import train

    args = argparse.Namespace(
        cache_text_encoder_outputs_to_disk=False,
        cache_text_encoder_outputs=False,
        cache_llm_adapter_outputs=False,
        network_train_unet_only=True,
        cpu_offload_checkpointing=False,
        unsloth_offload_checkpointing=False,
        gradient_checkpointing=False,
        blocks_to_swap=8,
        torch_compile=True,
        dynamo_backend="inductor",
        compile_inductor_mode="reduce-overhead",
        network_module="networks.lora_anima",
        functional_loss_weight=0.0,
        selective_checkpoint="off",
        selective_checkpoint_blocks="",
        block_swap_transfer_dtype="bf16",
        block_swap_restore_mode="foreach",
    )

    train.AnimaTrainer().assert_extra_args(args, _CacheableDataset(), None)

    assert args.torch_compile is True
    assert args.compile_inductor_mode is None


def test_trainer_raises_shared_matrix_errors() -> None:
    import train

    args = argparse.Namespace(
        cache_text_encoder_outputs_to_disk=False,
        cache_text_encoder_outputs=False,
        cache_llm_adapter_outputs=False,
        network_train_unet_only=True,
        cpu_offload_checkpointing=False,
        unsloth_offload_checkpointing=False,
        gradient_checkpointing=False,
        blocks_to_swap=8,
        torch_compile=False,
        dynamo_backend="inductor",
        compile_inductor_mode=None,
        network_module="networks.methods.soft_tokens",
        functional_loss_weight=0.0,
        selective_checkpoint="off",
        selective_checkpoint_blocks="",
        block_swap_transfer_dtype="bf16",
        block_swap_restore_mode="foreach",
    )

    with pytest.raises(ValueError, match="soft_tokens"):
        train.AnimaTrainer().assert_extra_args(args, _CacheableDataset(), None)


def test_matrix_rejects_convrot_with_block_swap_int8() -> None:
    result = check_training_compat(
        {
            "base_compute": "w8a16_convrot",
            "block_swap_transfer_dtype": "int8",
        }
    )
    assert "convrot_block_swap_int8_mutex" in _codes(result.errors)


def test_matrix_accepts_convrot_with_bf16_transfer() -> None:
    result = check_training_compat(
        {
            "base_compute": "w8a16_convrot",
            "block_swap_transfer_dtype": "bf16",
        }
    )
    assert "convrot_block_swap_int8_mutex" not in _codes(result.errors)
    assert "invalid_base_compute" not in _codes(result.errors)


def test_matrix_accepts_nf4_alone() -> None:
    result = check_training_compat({"base_compute": "nf4"})
    assert result.ok
    assert "invalid_base_compute" not in _codes(result.errors)
    assert "nf4_block_swap_unverified" not in _codes(result.errors)
    # nf4 单独 (无 blocks_to_swap) 不应触发 host_ram 告警
    assert "nf4_block_swap_host_ram" not in _codes(result.warnings)


def test_matrix_accepts_nf4_with_block_swap_warns_host_ram() -> None:
    """NF4 × block_swap 已端到端验证通过 (方向 A, offloader Params4bit 搬运),
    不再硬拒; 保留 warning 提醒主战场是 host RAM (pinned NF4 master)."""
    result = check_training_compat(
        {"base_compute": "nf4", "blocks_to_swap": 4}
    )
    # 不再拒: errors 不含旧的 unverified 码, 也不含新的 host_ram 码 (warning 而非 error)
    assert "nf4_block_swap_unverified" not in _codes(result.errors)
    assert "nf4_block_swap_host_ram" not in _codes(result.errors)
    # 告警: 提醒 host RAM 主战场
    assert "nf4_block_swap_host_ram" in _codes(result.warnings)


def test_matrix_rejects_unknown_base_compute_after_nf4_added() -> None:
    result = check_training_compat({"base_compute": "zzz"})
    assert "invalid_base_compute" in _codes(result.errors)


def test_matrix_warns_nf4_path_ignored_without_nf4_base_compute() -> None:
    """--nf4_prequantized_path 只在 base_compute=nf4 时生效; 给了路径但 base_compute
    不是 nf4 → warning (路径被静默忽略, 提醒用户配错)."""
    result = check_training_compat(
        {"base_compute": "bf16", "nf4_prequantized_path": "/path/to/nf4.safetensors"}
    )
    assert "nf4_path_ignored" in _codes(result.warnings)
    # base_compute=nf4 时给了路径不应触发该告警
    result_nf4 = check_training_compat(
        {"base_compute": "nf4", "nf4_prequantized_path": "/path/to/nf4.safetensors"}
    )
    assert "nf4_path_ignored" not in _codes(result_nf4.warnings)


def test_matrix_warns_convrot_attn_flash_compile_combo() -> None:
    """Four-lock path: scope covering attn + flash + compile + w8a16.

    Informational only — flash dispatcher hard-casts; do not hard-error.
    """
    hit = check_training_compat(
        {
            "base_compute": "w8a16_convrot",
            "convrot_scope": "all",
            "attn_mode": "flash",
            "torch_compile": True,
            "block_swap_transfer_dtype": "bf16",
        }
    )
    miss_mlp = check_training_compat(
        {
            "base_compute": "w8a16_convrot",
            "convrot_scope": "mlp",
            "attn_mode": "flash",
            "torch_compile": True,
            "block_swap_transfer_dtype": "bf16",
        }
    )
    miss_no_compile = check_training_compat(
        {
            "base_compute": "w8a16_convrot",
            "convrot_scope": "all",
            "attn_mode": "flash",
            "torch_compile": False,
            "block_swap_transfer_dtype": "bf16",
        }
    )
    miss_self_qkv = check_training_compat(
        {
            "base_compute": "w8a8_convrot",
            "convrot_scope": "self_attn_qkv",
            "attn_mode": "flash",
            "torch_compile": True,
            "block_swap_transfer_dtype": "bf16",
        }
    )

    assert hit.ok
    assert "convrot_attn_flash_compile_dtype" in _codes(hit.warnings)
    assert "convrot_attn_flash_compile_dtype" not in _codes(miss_mlp.warnings)
    assert "convrot_attn_flash_compile_dtype" not in _codes(miss_no_compile.warnings)
    assert "convrot_attn_flash_compile_dtype" in _codes(miss_self_qkv.warnings)


def test_matrix_accepts_krea2_plain_lora() -> None:
    result = check_training_compat(
        {
            "model_family": "krea2_raw",
            "network_module": "networks.lora_anima",
            "attn_mode": "torch",
            "selective_checkpoint": "off",
        }
    )


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        (["dora_wd=true"], "family_plain_lora_only"),
        (["use_ortho=true"], "family_plain_lora_only"),
        (["train_adaln=true"], "family_plain_lora_only"),
        (["num_registers=4"], "family_plain_lora_only"),
    ],
)
def test_qwen_plain_lora_gate_checks_effective_network_args(arguments, code) -> None:
    result = check_training_compat(
        {
            "model_family": "qwen_image_2_1",
            "network_module": "networks.lora_anima",
            "network_args": arguments,
        }
    )

    assert code in _codes(result.errors)


def test_qwen_plain_lora_gate_uses_network_args_precedence() -> None:
    result = check_training_compat(
        {
            "model_family": "qwen_image_2_1",
            "network_module": "networks.lora_anima",
            "num_registers": 4,
            "network_args": ["num_registers=0"],
        }
    )

    assert "family_plain_lora_only" not in _codes(result.errors)


@pytest.mark.parametrize("task", ["t2i", "edit"])
@pytest.mark.parametrize("variant", ["use_lokr", "use_loha"])
def test_qwen_allows_only_selected_plugin_specs(task, variant) -> None:
    config = {
        "model_family": "qwen_image_2_1",
        "qwen_image_2_1_task": task,
        "network_module": "networks.lora_anima",
        "network_args": [f"{variant}=true"],
    }
    result = check_training_compat(config)
    if task == "t2i":
        assert "family_plain_lora_only" not in _codes(result.errors)
        assert result.ok
    else:
        assert "family_plain_lora_only" in _codes(result.errors)

    for conflict in ("use_glora", "use_vera", "use_ortho", "dora_wd", "train_adaln"):
        rejected = check_training_compat({**config, conflict: True})
        assert "family_plain_lora_only" in _codes(rejected.errors), conflict

    fused = check_training_compat({**config, "qwen_fused_projections": "qkv"})
    assert "qwen_fused_projections_plain_lora" in _codes(fused.errors)


def test_qwen_rejects_lokr_loha_together_and_method_adapter() -> None:
    config = {
        "model_family": "qwen_image_2_1",
        "network_module": "networks.lora_anima",
        "use_lokr": True,
        "use_loha": True,
    }
    assert "family_plain_lora_only" in _codes(check_training_compat(config).errors)
    assert "family_plain_lora_only" in _codes(check_training_compat({
        **config, "use_loha": False, "network_module": "networks.methods.ip_adapter"
    }).errors)


@pytest.mark.parametrize(
    "override",
    [
        {"network_module": "networks.methods.ip_adapter"},
        {"use_ip_adapter": True},
        {"use_easycontrol": True},
        {"use_byg": True},
        {"use_lokr": True},
        {"use_ortho": True},
        {"use_moe_style": "shared_A"},
        {"route_per_layer": True},
        {"router_source": "input"},
        {"dora_wd": 0.1},
        {"step_expert_K": 2},
        {"functional_loss_weight": 0.1},
    ],
)
@pytest.mark.parametrize("family", ["anima", "krea2_raw", "z_image"])
def test_matrix_does_not_blanket_reject_model_family_variants(override, family) -> None:
    config = {
        "model_family": family,
        "network_module": "networks.lora_anima",
        "attn_mode": "torch",
        "selective_checkpoint": "off",
    }
    config.update(override)
    result = check_training_compat(config)
    assert "family_plain_lora_only" not in _codes(result.errors)


class _CacheableDataset:
    datasets: list[object] = []

    def is_text_encoder_output_cacheable(self, *, cache_supports_dropout: bool) -> bool:
        return True

    def verify_bucket_reso_steps(self, steps: int) -> None:
        assert steps == 16


def test_cli_preflight_applies_model_family_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("ANIMA_MODEL_FAMILY", "krea2_raw")
    args = argparse.Namespace(
        model_family=None,
        use_lokr=True,
        cache_text_encoder_outputs_to_disk=False,
        cache_text_encoder_outputs=False,
        cache_llm_adapter_outputs=False,
        network_train_unet_only=True,
        selective_checkpoint="off",
        selective_checkpoint_blocks="",
        block_swap_transfer_dtype="bf16",
        block_swap_restore_mode="slab",
    )

    assert_training_extra_args(args, _CacheableDataset(), None)

    assert args.model_family == "krea2_raw"
