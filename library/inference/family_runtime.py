"""Fail-closed model-family preparation for the inference entrypoint."""

from __future__ import annotations

from library.models.family_plugins import (
    InferenceRunner,
    get_family_plugin,
    validate_runner_contract,
)
from library.models.family_registry import get_model_family_spec


_INFERENCE_OPTION_GROUPS = {
    "qwen_reference": (
        ("qwen_image_2_1_task", "--qwen_image_2_1_task"),
        ("reference_image", "--reference_image"),
        ("reference_images", "--reference_images"),
    ),
    "ip_adapter": (
        ("ip_adapter_weight", "--ip_adapter_weight"),
        ("ip_image", "--ip_image"),
        ("ip_scale", "--ip_scale"),
        ("ip_image_match_size", "--ip_image_match_size"),
    ),
    "easycontrol": (
        ("easycontrol_weight", "--easycontrol_weight"),
        ("easycontrol_image", "--easycontrol_image"),
        ("easycontrol_scale", "--easycontrol_scale"),
        ("easycontrol_image_match_size", "--easycontrol_image_match_size"),
    ),
}


def _reject_unsupported_capability_options(args, family: str) -> None:
    spec = get_model_family_spec(family)
    offenders = []
    for group, options in _INFERENCE_OPTION_GROUPS.items():
        if group in spec.supported_inference_option_groups:
            continue
        offenders.extend(
            flag
            for name, flag in options
            if _inference_option_is_set(getattr(args, name, None))
        )
    for group, weight_name, image_name in (
        ("ip_adapter", "ip_adapter_weight", "ip_image"),
        ("easycontrol", "easycontrol_weight", "easycontrol_image"),
    ):
        if group not in spec.supported_inference_option_groups:
            continue
        weight_set = _inference_option_is_set(getattr(args, weight_name, None))
        image_set = _inference_option_is_set(getattr(args, image_name, None))
        if weight_set != image_set:
            weight_flag = "--ip_adapter_weight" if group == "ip_adapter" else "--easycontrol_weight"
            image_flag = "--ip_image" if group == "ip_adapter" else "--easycontrol_image"
            required = image_flag if weight_set else weight_flag
            supplied = weight_flag if weight_set else image_flag
            offenders.append(f"{required} must be supplied together with {supplied}")
    for group, image_name, dependent_names in (
        ("ip_adapter", "ip_image", ("ip_scale", "ip_image_match_size")),
        (
            "easycontrol",
            "easycontrol_image",
            ("easycontrol_scale", "easycontrol_image_match_size"),
        ),
    ):
        if group not in spec.supported_inference_option_groups:
            continue
        if _inference_option_is_set(getattr(args, image_name, None)):
            continue
        for name in dependent_names:
            if _inference_option_is_set(getattr(args, name, None)):
                flag = "--ip_image" if group == "ip_adapter" else "--easycontrol_image"
                offender = "--ip_scale" if name == "ip_scale" else "--ip_image_match_size"
                if group == "easycontrol":
                    offender = (
                        "--easycontrol_scale"
                        if name == "easycontrol_scale"
                        else "--easycontrol_image_match_size"
                    )
                offenders.append(f"{offender} requires {flag}")
    if offenders:
        raise SystemExit(
            f"{spec.display_name} inference does not support these options:\n  - "
            + "\n  - ".join(offenders)
        )


def _inference_option_is_set(value) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (str, list, tuple, set, frozenset)):
        return bool(value)
    return True


def validate_family_inference_options(args, family: str) -> None:
    _reject_unsupported_capability_options(args, family)


def validate_inference_family_options(args, family: str) -> None:
    """Validate family-independent option ownership before mode dispatch."""
    validate_family_inference_options(args, family)


def prepare_inference_family(args, family: str, mode: str) -> InferenceRunner:
    """Build, validate, and prepare a family runner exactly once."""
    runner = build_inference_runner(args, family)
    validate_family_inference_options(args, family)
    spec = get_model_family_spec(family)
    if mode not in spec.supported_inference_modes:
        supported = (
            "single-prompt"
            if spec.supported_inference_modes == frozenset({"single"})
            else ", ".join(sorted(spec.supported_inference_modes))
        )
        raise SystemExit(
            f"{spec.display_name} inference does not support {mode!r} mode; "
            f"supported: {supported}."
        )

    runner.validate(args, mode=mode)
    runner.prepare(args, mode=mode)
    return runner


def build_inference_runner(args, family: str) -> InferenceRunner:
    """Build the registered family's inference facade through its plugin."""
    plugin = get_family_plugin(family, capability="inference")
    builder = getattr(plugin, "build_inference_runner", None)
    if not callable(builder):
        raise RuntimeError(f"Model family plugin {family!r} has no inference runner")
    runner = builder(args)
    return validate_runner_contract(runner, family)
