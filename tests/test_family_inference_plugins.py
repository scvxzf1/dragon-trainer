from types import SimpleNamespace

import pytest

from library.inference.family_runtime import build_inference_runner
from library.models.family_plugins import (
    FamilyPluginContractError,
    get_family_plugin,
    validate_plugin_contract,
)
from library.models.family_registry import get_model_family_spec


@pytest.mark.parametrize(
    "family,runner_type",
    [
        ("anima", "AnimaLegacyInferenceRunner"),
        ("krea2_raw", "Krea2InferenceRunner"),
        ("qwen_image_2_1", "QwenImage21InferenceRunner"),
        ("z_image", "ZImageInferenceRunner"),
        ("moss_image", "MossImageInferenceRunner"),
    ],
)
def test_every_family_plugin_builds_identity_checked_runner(family, runner_type):
    runner = build_inference_runner(SimpleNamespace(), family)
    assert type(runner).__name__ == runner_type
    assert runner.family == family
    assert get_family_plugin(family).build_inference_runner(SimpleNamespace()).family == family


@pytest.mark.parametrize("family", ["z_image", "moss_image"])
def test_new_standalone_families_publish_single_euler_contract(family):
    spec = get_model_family_spec(family)
    assert spec.supported_inference_modes == frozenset({"single"})
    assert spec.supported_inference_samplers == frozenset({"euler"})


def test_only_anima_publishes_legacy_batch_and_interactive_modes():
    assert get_model_family_spec("anima").supported_inference_modes == frozenset(
        {"single", "batch", "interactive"}
    )
    for family in ("krea2_raw", "z_image", "qwen_image_2_1", "moss_image"):
        assert get_model_family_spec(family).supported_inference_modes == frozenset(
            {"single"}
        )


def test_family_plugin_runner_is_fail_closed_for_unknown_family():
    with pytest.raises(ValueError, match="unknown_family"):
        build_inference_runner(SimpleNamespace(), "unknown_family")


def test_default_plugin_contract_does_not_require_qwen_preview_hook():
    for family in ("anima", "krea2_raw", "z_image", "moss_image"):
        plugin = get_family_plugin(family)
        validate_plugin_contract(plugin, family)
        assert not callable(getattr(plugin, "cache_preview_text", None))


def test_qwen_preview_hook_is_required_only_when_requested(monkeypatch):
    plugin = get_family_plugin("qwen_image_2_1", capability="qwen_preview_text")
    assert callable(plugin.cache_preview_text)

    with monkeypatch.context() as scoped:
        scoped.setattr(plugin, "cache_preview_text", None)
        validate_plugin_contract(plugin, "qwen_image_2_1")
        with pytest.raises(FamilyPluginContractError, match="cache_preview_text"):
            validate_plugin_contract(
                plugin,
                "qwen_image_2_1",
                capability="qwen_preview_text",
            )
