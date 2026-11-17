"""Read-only draft overrides for the distributed diagnosis."""

from typing import Any

DIAGNOSIS_OVERRIDE_KEYS = frozenset({
    "model_family", "mixed_precision", "train_batch_size", "blocks_to_swap",
    "torch_compile", "pipeline_parallel", "pipeline_parallel_stages",
    "pipeline_parallel_microbatches", "pipeline_parallel_schedule",
    "pipeline_parallel_split", "pipeline_parallel_blocks_to_swap",
})


def apply_diagnosis_overrides(
    cfg: dict[str, Any], overrides: Any,
) -> dict[str, Any]:
    """Copy the loaded config; accept only the card's scalar input fields."""
    if overrides is None:
        return dict(cfg)
    if not isinstance(overrides, dict):
        raise ValueError("config_overrides must be an object")
    unknown = set(overrides) - DIAGNOSIS_OVERRIDE_KEYS
    if unknown:
        raise ValueError(f"Unsupported diagnosis overrides: {', '.join(sorted(unknown))}")
    if any(not isinstance(value, (str, int, float, bool)) for value in overrides.values()):
        raise ValueError("Diagnosis overrides must be scalar values")
    return {**cfg, **overrides}
