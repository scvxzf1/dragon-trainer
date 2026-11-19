"""Fail-closed model-family preparation for the inference entrypoint."""

from __future__ import annotations

from library.models.family_plugins import (
    InferenceRunner,
    get_family_plugin,
    validate_runner_contract,
)
from library.models.family_registry import get_model_family_spec


_REFERENCE_CONDITIONING_OPTIONS = (
    ("qwen_image_2_1_task", "--qwen_image_2_1_task"),
    ("reference_image", "--reference_image"),
    ("reference_images", "--reference_images"),
)


def _reject_unsupported_reference_options(
    args,
    runner: InferenceRunner,
) -> None:
    if getattr(runner, "supports_reference_conditioning", False):
        return
    offenders = [
        flag
        for name, flag in _REFERENCE_CONDITIONING_OPTIONS
        if getattr(args, name, None) is not None
    ]
    if offenders:
        raise SystemExit(
            "Qwen task/reference flags require a family runner that "
            "supports reference conditioning; use --model_family "
            f"qwen_image_2_1 (got {', '.join(offenders)})."
        )


def validate_inference_family_options(args, family: str) -> None:
    """Validate family-independent option ownership before mode dispatch."""
    _reject_unsupported_reference_options(args, build_inference_runner(args, family))


def prepare_inference_family(args, family: str, mode: str) -> InferenceRunner:
    """Build, validate, and prepare a family runner exactly once."""
    runner = build_inference_runner(args, family)
    _reject_unsupported_reference_options(args, runner)
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
    plugin = get_family_plugin(family)
    builder = getattr(plugin, "build_inference_runner", None)
    if not callable(builder):
        raise RuntimeError(f"Model family plugin {family!r} has no inference runner")
    runner = builder(args)
    return validate_runner_contract(runner, family)
