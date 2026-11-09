from __future__ import annotations

import pytest

from library.training.compat_matrix import check_training_compat


def _codes(items) -> set[str]:
    return {item.code for item in items}


@pytest.mark.parametrize("attn_mode", [None, "torch", "flash", "sdpa", "cudnn", "cudnn_sdpa"])
def test_krea2_compat_accepts_supported_attention_modes(attn_mode) -> None:
    result = check_training_compat(
        {
            "model_family": "krea2_raw",
            "attn_mode": attn_mode,
            "compile_dynamic_seq": False,
            "compile_inductor_mode": "default",
            "selective_checkpoint": "off",
            "v100_flash_stability": "off",
        }
    )

    assert result.ok


@pytest.mark.parametrize(
    ("patch", "code"),
    [
        ({"attn_mode": "xformers"}, "krea2_invalid_attn_mode"),
        ({"attn_mode": "native"}, "krea2_invalid_attn_mode"),
        ({"compile_inductor_mode": "reduce-overhead"}, "krea2_compile_inductor_mode"),
        ({"selective_checkpoint": "mlp_only"}, "krea2_selective_checkpoint"),
        ({"v100_flash_stability": "safe"}, "krea2_v100_flash_stability"),
    ],
)
def test_krea2_compat_rejects_unsupported_family_options(patch, code) -> None:
    config = {
        "model_family": "krea2_raw",
        "attn_mode": "torch",
        "compile_dynamic_seq": False,
        "compile_inductor_mode": "default",
        "selective_checkpoint": "off",
        "v100_flash_stability": "off",
    }
    config.update(patch)

    result = check_training_compat(config)

    assert code in _codes(result.errors)


def test_krea2_compat_disables_dynamic_sequence_inherited_from_base() -> None:
    result = check_training_compat(
        {
            "model_family": "krea2_raw",
            "attn_mode": "flash",
            "compile_dynamic_seq": True,
            "compile_inductor_mode": "default",
            "selective_checkpoint": "off",
            "v100_flash_stability": "off",
        }
    )

    assert result.ok
    assert "krea2_compile_dynamic_seq" in _codes(result.warnings)
    assert [(item.key, item.value) for item in result.mutations] == [
        ("compile_dynamic_seq", False)
    ]


def test_krea2_compat_disables_per_band_sequence() -> None:
    result = check_training_compat(
        {
            "model_family": "krea2_raw",
            "attn_mode": "flash",
            "compile_dynamic_seq": False,
            "compile_seq_bands": True,
            "compile_inductor_mode": "default",
            "selective_checkpoint": "off",
            "v100_flash_stability": "off",
        }
    )

    assert result.ok
    assert "krea2_compile_seq_bands" in _codes(result.warnings)
    assert [(item.key, item.value) for item in result.mutations] == [
        ("compile_seq_bands", False)
    ]


def test_anima_keeps_general_attention_and_compile_options() -> None:
    result = check_training_compat(
        {
            "model_family": "anima",
            "attn_mode": "xformers",
            "compile_dynamic_seq": True,
            "compile_inductor_mode": "max-autotune",
            "selective_checkpoint": "mlp_only",
            "v100_flash_stability": "safe",
        }
    )

    assert not {code for code in _codes(result.errors) if code.startswith("krea2_")}
