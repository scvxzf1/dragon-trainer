"""Fail-closed contracts for the first MOSS-Image text-to-image LoRA path."""

from collections.abc import Mapping
import logging

from library.config.normalize import is_flag_enabled


logger = logging.getLogger(__name__)


def _get(config, key, default=None):
    return config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)


def _enabled(value):
    return is_flag_enabled(value)


def _number(value, default=0):
    try:
        return float(value) if value is not None else float(default)
    except (ValueError, TypeError):
        return float("nan")


def _dataset_rows(config):
    general = _get(config, "general", {})
    if isinstance(general, Mapping):
        yield general
    for dataset in _get(config, "datasets", []) or []:
        if not isinstance(dataset, Mapping):
            continue
        yield dataset
        for subset in dataset.get("subsets", []) or []:
            if isinstance(subset, Mapping):
                yield subset


def check_moss_training_contract(out, config):
    """Append errors to the shared compatibility builder; never mutate settings."""
    def reject(key, active, detail):
        if active:
            out.error(f"moss_image_{key}", key, f"MOSS-Image v1 {detail}")

    for key, default, accepted in (
        ("mixed_precision", "bf16", {"bf16", "fp16", "no"}),
        ("base_compute", "bf16", {"bf16"}),
        ("moss_te_quantization", "nf4", {"nf4", "bf16"}),
        ("attn_mode", "torch", {"torch", "sdpa"}),
        ("selective_checkpoint", "off", {"off"}),
        ("adaptive_precision", "off", {"off"}),
        ("timestep_sampling", "uniform", {"uniform"}),
        ("weighting_scheme", "none", {"none"}),
        ("loss_type", "l2", {"l2"}),
        ("sampler", "default", {"default"}),
        ("sample_sampler", "euler", {"euler"}),
        ("block_swap_transfer_dtype", "bf16", {"bf16", "int8"}),
        ("v100_flash_stability", "off", {"off"}),
    ):
        value = str(_get(config, key, default) or default).strip().lower()
        reject(key, value not in accepted, f"requires {key} in {sorted(accepted)}; got {value!r}.")
        if key == "block_swap_transfer_dtype" and value == "int8":
            warning = (
                "MOSS-Image int8 block-swap transfer is experimental: observed "
                "per-step loss shifts are approximately 0.3%-2.0% versus the "
                "unquantized path; validate quality before production use."
            )
            logger.warning(warning)
            out.warning("moss_image_int8_experimental", key, warning)

    for key in (
        "gradient_checkpointing", "network_train_unet_only", "use_vae_cache", "use_text_cache",
    ):
        # Config-facing cache flags imply disk caching in CLI normalization.
        reject(key, not _enabled(_get(config, key, True)), f"requires {key}=true.")
    for legacy, canonical in (
        ("cache_latents", "use_vae_cache"),
        ("cache_latents_to_disk", "use_vae_cache"),
        ("cache_text_encoder_outputs", "use_text_cache"),
        ("cache_text_encoder_outputs_to_disk", "use_text_cache"),
    ):
        value = _get(config, legacy)
        reject(legacy, value is not None and not _enabled(value), f"requires disk caches ({canonical}=true).")

    for key in (
        "torch_compile", "compile_dynamic_seq", "compile_seq_bands",
        "cpu_offload_checkpointing", "unsloth_offload_checkpointing", "auto_block_swap",
        "network_train_text_encoder_only", "train_llm_adapter", "weighted_captions",
        "use_shuffled_caption_variants", "cache_llm_adapter_outputs", "xformers", "mem_eff_attn",
        "use_ip_adapter", "use_easycontrol", "use_byg", "use_lokr", "use_loha", "use_glora",
        "use_vera", "use_ortho", "use_chimera_hydra", "use_timestep_mask", "add_reft",
        "use_moe_style", "route_per_layer", "dora_wd", "train_adaln", "masked_loss",
        "scale_v_pred_loss_like_noise_pred", "v_pred_like_loss", "debiased_estimation_loss",
        "blank_prompt_preservation", "adaptive_personalization_observe",
        "adaptive_personalization_loss_weighting", "adaptive_personalization_affine",
        "qwen_nf4_compile_experiment", "adaptive_oom_retry",
    ):
        reject(key, _enabled(_get(config, key)), f"does not support {key}; disable it.")
    # ``pipeline_parallel`` is no longer blanket-rejected: MOSS-Image publishes a
    # stage-local PP v0 runtime backend and wired trainer session, so the shared
    # compatibility matrix and ``library.training.pp.gate`` decide whether the
    # requested PP config is servable. Block swap remains MOSS-legal alone and
    # the PP config validation rejects the combination.
    reject("network_module", _get(config, "network_module", "networks.lora_anima") not in
           {"", "networks.lora_anima"}, "supports networks.lora_anima plain LoRA only.")
    reject("router_source", str(_get(config, "router_source", "none") or "none").lower() != "none",
           "supports router_source=none only.")
    for key in (
        "functional_loss_weight", "multiscale_loss_weight", "vr_loss_weight", "min_snr_gamma",
        "velocity_direction_loss_weight", "prior_preservation_weight", "inverted_mask_prior_weight",
        "p2_gamma", "balance_loss_weight", "fera_fecl_weight", "contrastive_loss_weight",
        "num_registers", "channel_scaling_alpha",
    ):
        reject(key, _number(_get(config, key)) != 0, f"does not support {key}; set it to 0.")
    reject("step_expert_K", _number(_get(config, "step_expert_K"), 0) > 1,
           "supports a single plain LoRA expert only.")
    for key in ("layer_start", "layer_end", "t_min", "t_max", "nf4_prequantized_path"):
        reject(key, _get(config, key) not in (None, ""), f"does not support {key}.")
    reject("discrete_flow_shift", _number(_get(config, "discrete_flow_shift"), 1) != 1,
           "uses a uniform 1000-step flow grid with discrete_flow_shift=1.")
    length = _number(_get(config, "max_token_length"), 4096)
    reject("max_token_length", not (1 <= length <= 4096 and length.is_integer()),
           "requires an integer max_token_length between 1 and 4096.")
    swap = _number(_get(config, "blocks_to_swap"))
    reject("blocks_to_swap", not (0 <= swap <= 40 and float(swap).is_integer()),
           "requires an integer blocks_to_swap between 0 and 40.")
    rows = list(_dataset_rows(config))
    batches = [_get(config, "train_batch_size", 1), _get(config, "batch_size", 1)]
    batches.extend(row["batch_size"] for row in rows if "batch_size" in row)
    reject("train_batch_size", swap > 0 and any(_number(value) != 1 for value in batches),
           "block swap requires batch size 1, including every dataset override.")
    for key in ("caption_dropout_rate", "caption_tag_dropout_rate", "caption_dropout_every_n_epochs"):
        reject(key, _number(_get(config, key)) != 0 or
               any(_number(row.get(key)) != 0 for row in rows), f"requires {key}=0 at every dataset level.")
    for key in ("weighted_captions", "shuffle_caption"):
        reject(key, any(_enabled(row.get(key)) for row in rows), f"does not support dataset {key}.")
