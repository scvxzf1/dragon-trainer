"""Training contract for hardware-selected precision and bounded OOM retry.

The ``auto`` mode is resolved before the trainer is constructed.  It may select
plain BF16/FP32, native Anima FP16 AMP, or explicit FP16/FP32 islands.
"""

from collections.abc import Mapping
import math

ADAPTIVE_FAMILIES = ("anima", "krea2_raw", "z_image")


def enabled(config):
    get = config.get if isinstance(config, Mapping) else lambda k, d=None: getattr(config, k, d)
    return get("adaptive_precision", "off") != "off"


def resolved_mode(config):
    """Return the execution mode after an optional ``auto`` resolution."""
    get = config.get if isinstance(config, Mapping) else lambda k, d=None: getattr(config, k, d)
    mode = str(get("adaptive_precision", "off") or "off").strip().lower()
    if mode != "auto":
        return mode
    resolved = get("adaptive_resolved_mode", None)
    if resolved in {"bf16", "native_fp16", "fp16_fp32", "fp32"}:
        return resolved
    # A deterministic fallback keeps config/preflight validation usable on a
    # host where CUDA probing is unavailable.  The training entry resolves the
    # actual hardware policy before launching a worker.
    mixed = str(get("mixed_precision", "bf16") or "bf16").strip().lower()
    from library.models.family_registry import normalize_registered_family
    family = normalize_registered_family(get("model_family", "anima") or "anima",
                                         source="adaptive precision model_family", allow_aliases=True)
    fp16_mode = "native_fp16" if family == "anima" else "fp16_fp32"
    return {"bf16": "bf16", "fp16": fp16_mode, "no": "fp32"}.get(mixed, "fp32")


def islands_enabled(config):
    return resolved_mode(config) == "fp16_fp32"


def add_arguments(parser):
    parser.add_argument("--adaptive_precision", choices=("off", "auto", "native_fp16", "fp16_fp32"), default="off",
                        help="Hardware-selected precision, native Anima FP16 AMP, or FP16/FP32 islands.")
    parser.add_argument("--adaptive_fp32_modules", nargs="*", default=[],
                        help="Exact names or glob patterns of frozen Linear modules kept in FP32.")
    parser.add_argument("--adaptive_loss_scale", type=float, default=1024.0,
                        help="Initial Accelerate loss scale for native FP16 AMP or FP16/FP32 islands.")
    parser.add_argument("--adaptive_oom_retry", action="store_true",
                        help="[EXPERIMENTAL] Run training in a fresh-process swap-retry supervisor.")
    parser.add_argument("--adaptive_oom_retry_resume", action="store_true",
                        help="[EXPERIMENTAL] Save exact cursors and resume committed CUDA OOM steps.")
    parser.add_argument("--adaptive_oom_retry_max_attempts", type=int, default=4)
    parser.add_argument("--adaptive_oom_retry_swap_increment", type=int, default=2)
    parser.add_argument("--adaptive_oom_retry_max_swap", type=int, default=0,
                        help="Maximum swap blocks; 0 uses the local model's structural limit.")
    parser.add_argument("--adaptive_oom_retry_timeout", type=float, default=3600.0)
    parser.add_argument("--adaptive_max_nonfinite_skips", type=int, default=0,
                        help="Consecutive loss-scaler skips the adaptive trial tolerates before "
                             "failing with a nonfinite_skip result (0 keeps the strict contract); "
                             "skipped steps never count as effective steps or timing samples.")


def configuration_errors(config, *, world_size=1):
    get = config.get if isinstance(config, Mapping) else lambda k, d=None: getattr(config, k, d)
    try:
        mode = resolved_mode(config)
    except ValueError as exc:
        return [str(exc)]
    if mode == "native_fp16":
        from .native_precision import configuration_errors as native_errors
        return native_errors(config, world_size=world_size)
    if mode == "off":
        return (["adaptive_oom_retry requires adaptive_precision=fp16_fp32"]
                if get("adaptive_oom_retry", False) else [])
    if mode not in {"bf16", "fp16_fp32", "fp32"}:
        return ["Unknown adaptive_precision mode"]
    errors = []
    from library.models.family_registry import normalize_registered_family
    try:
        family = normalize_registered_family(get("model_family", "anima") or "anima",
                                             source="adaptive precision model_family", allow_aliases=True)
    except ValueError as exc:
        return [str(exc)]
    if family not in ADAPTIVE_FAMILIES:
        errors.append(f"adaptive precision does not support model_family={family!r}")
    if mode in {"bf16", "fp32"}:
        if get("adaptive_oom_retry", False):
            errors.append("adaptive_oom_retry requires adaptive_precision=fp16_fp32")
        if mode == "bf16" and get("mixed_precision", "bf16") != "bf16":
            errors.append("adaptive BF16 mode requires mixed_precision='bf16'")
        if mode == "fp32" and get("mixed_precision", "no") not in {"no", "fp32"}:
            errors.append("adaptive FP32 mode requires mixed_precision='no'")
        return errors
    if get("adaptive_oom_retry", False):
        errors.extend(retry_configuration_errors(get))
    required = {"mixed_precision": "fp16",
                "base_compute": "bf16",
                "network_module": "networks.lora_anima", "lora_adapter_kind": "lora",
                "network_train_unet_only": True, "gradient_checkpointing": True,
                "selective_checkpoint": "off", "block_swap_transfer_dtype": "bf16",
                "block_swap_restore_mode": "foreach", "max_data_loader_n_workers": 0}
    for name, value in required.items():
        if get(name, value) != value:
            errors.append(f"FP16/FP32 training requires {name}={value!r}")
    attn_mode = str(get("attn_mode", "torch") or "torch").strip().lower()
    if family == "krea2_raw" and attn_mode != "torch":
        errors.append("Krea-2 FP16/FP32 training requires attn_mode='torch'")
    if family == "z_image" and attn_mode not in {"torch", "sdpa"}:
        errors.append("Z-Image FP16/FP32 training requires attn_mode='torch' or 'sdpa'")
    if world_size != 1:
        errors.append("FP16/FP32 training currently requires one process")
    forbidden = (
        "full_fp16", "full_bf16", "torch_compile", "nf4_prequantized_path", "pipeline_parallel",
        "auto_block_swap", "cpu_offload_checkpointing", "unsloth_offload_checkpointing",
        "network_args", "base_weights", "lora_path", "stage_schedule", "use_moe_style",
        "use_ortho", "use_timestep_mask", "add_reft", "use_glora", "use_loha", "use_lokr",
        "use_vera", "use_dora", "dora_wd", "use_ip_adapter", "use_easycontrol", "use_byg",
        "use_chimera_hydra", "train_llm_adapter", "network_train_text_encoder_only",
        "sample_at_first", "sample_every_n_steps", "sample_every_n_epochs",
        "validate_every_n_steps", "validate_every_n_epochs", "functional_loss_weight",
        "vr_loss_weight", "resume_from_huggingface",
    )
    for name in forbidden:
        if get(name, False):
            errors.append(f"FP16/FP32 training does not yet support {name}")
    for name in ("use_vae_cache", "use_text_cache", "cache_latents", "cache_text_encoder_outputs"):
        if get(name, True) is False:
            errors.append(f"FP16/FP32 training requires {name}")
    scale = get("adaptive_loss_scale", 1024.0)
    if (isinstance(scale, bool) or not isinstance(scale, (float, int))
            or not math.isfinite(scale) or not 1 < scale <= 2**24):
        errors.append("adaptive_loss_scale must be finite and in (1, 2**24]")
    patterns = get("adaptive_fp32_modules", [])
    if not isinstance(patterns, (list, tuple)) or not all(isinstance(p, str) and p for p in patterns):
        errors.append("adaptive_fp32_modules must be a list of nonempty patterns")
    return errors


def require_training_contract(config, *, world_size=1):
    errors = configuration_errors(config, world_size=world_size)
    from .resume_runtime import configuration_errors as recovery_errors
    get = config.get if isinstance(config, Mapping) else lambda k, d=None: getattr(config, k, d)
    errors.extend(recovery_errors(get, world_size=world_size))
    if errors:
        raise ValueError("; ".join(errors))


def retry_bounds_errors(get):
    errors = []
    for key, default, minimum, maximum in (
        ("blocks_to_swap", 0, 0, None), ("adaptive_oom_retry_max_swap", 0, 0, None),
        ("adaptive_oom_retry_swap_increment", 2, 1, None),
        ("adaptive_oom_retry_max_attempts", 4, 1, 16),
    ):
        value = get(key, default)
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
            errors.append(f"{key} must be an integer {bound}")
    initial, maximum = get("blocks_to_swap", 0), get("adaptive_oom_retry_max_swap", 0)
    if type(initial) is int and type(maximum) is int and maximum and initial > maximum:
        errors.append("Initial block swap exceeds retry maximum")
    timeout = get("adaptive_oom_retry_timeout", 3600.0)
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        errors.append("adaptive_oom_retry_timeout must be finite and positive")
    return errors


def retry_configuration_errors(get):
    errors = retry_bounds_errors(get)
    for key in ("resume", "initial_step", "initial_epoch", "save_state", "save_state_on_train_end",
                "save_every_n_steps", "save_every_n_epochs", "checkpointing_epochs", "save_n_epoch_ratio",
                "huggingface_repo_id", "log_with", "debug_dataset", "artist_filter", "scale_weight_norms"):
        if get(key, False):
            errors.append(f"Experimental OOM supervisor does not support {key}")
    if get("gradient_accumulation_steps", 1) != 1:
        errors.append("Experimental OOM supervisor requires accumulation=1")
    if type(get("seed")) is not int:
        errors.append("Experimental OOM supervisor requires an explicit integer seed")
    return errors
