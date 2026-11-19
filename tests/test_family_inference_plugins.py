from types import SimpleNamespace

import pytest

from library.inference.family_runtime import build_inference_runner
from library.models.family_plugins import get_family_plugin
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


def test_family_plugin_runner_is_fail_closed_for_unknown_family():
    with pytest.raises(ValueError, match="unknown_family"):
        build_inference_runner(SimpleNamespace(), "unknown_family")
