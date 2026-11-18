"""Read-only draft overrides for the distributed diagnosis."""

from typing import Any

DIAGNOSIS_OVERRIDE_KEYS = frozenset({
    "model_family", "mixed_precision", "base_compute", "train_batch_size",
    "gradient_accumulation_steps", "gradient_checkpointing", "selective_checkpoint",
    "cpu_offload_checkpointing", "unsloth_offload_checkpointing",
    "blocks_to_swap", "auto_block_swap", "adaptive_precision", "adaptive_oom_retry",
    "torch_compile", "compile_dynamic_seq", "compile_seq_bands",
    "sample_at_first", "sample_every_n_steps", "sample_every_n_epochs",
    "use_vae_cache", "use_text_cache", "skip_cache_check",
    "loss_type", "weighting_scheme", "stage_schedule_enabled",
    "save_n_epoch_ratio", "save_every_n_epochs", "persistent_data_loader_workers",
    "router_source", "use_moe_style", "use_ip_adapter", "use_easycontrol",
    "use_byg", "caption_dropout_rate", "ip_noise_gamma", "network_train_unet_only",
    "pipeline_parallel", "pipeline_parallel_stages",
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
