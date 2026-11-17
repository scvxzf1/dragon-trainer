"""Persistence helpers for the LoRA-family network facade."""

import json
import os
import re
from typing import Dict

import torch

from library.env import normalize_model_family
from networks import lora_save
from networks.registry import (
    NETWORK_REGISTRY,
    NetworkSpec,
    preprocess_weights_from_plugins,
)
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.loading import (
    _refuse_split_chimera_keys,
    _refuse_split_hydra_keys,
    _refuse_split_stacked_experts_keys,
    _refuse_unfused_attn_lora_keys,
    _rename_dora_scale_for_load,
    _stack_chimera_lora_ups,
    _stack_lora_ups,
)


def stamp_lora_save_metadata(
    metadata: Dict[str, str],
    cfg: LoRANetworkCfg,
    spec: NetworkSpec,
    network=None,
) -> None:
    if metadata:
        metadata["ss_network_spec"] = spec.name

    if spec.name == "lokr":
        metadata.setdefault("ss_network_dim", str(cfg.lora_dim))
        metadata.setdefault("ss_network_alpha", str(cfg.alpha))
        uses_full_factor_layout = None
        if network is not None:
            from networks.plugins.lokr.module import LoKrModule

            lokr_modules = [
                lora
                for lora in getattr(network, "text_encoder_loras", [])
                + getattr(network, "unet_loras", [])
                if isinstance(lora, LoKrModule)
            ]
            if lokr_modules:
                uses_full_factor_layout = all(
                    (not getattr(lora, "_use_decomposed_w2", False))
                    for lora in lokr_modules
                )
        if uses_full_factor_layout is None:
            plugin_args = getattr(cfg, "plugin_args", {}) or {}
            raw_full = plugin_args.get("lokr_full_factor", False)
            if isinstance(raw_full, str):
                uses_full_factor_layout = raw_full.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            else:
                uses_full_factor_layout = bool(raw_full)
        metadata["ss_lokr_full_factor"] = (
            "true" if bool(uses_full_factor_layout) else "false"
        )

    # Hard sigma-band partition lives in non-persistent buffers (`_expert_band`)
    # and a Python attr (`_sigma_band_partition`); nothing of it survives
    # the state_dict write. Emit the two scalars needed to re-register the
    # partition at load time so inference and the ComfyUI node can reconstruct
    # the per-sample band mask. Only stamped when the partition is on, so
    # older non-band checkpoints stay byte-identical.
    if cfg.specialize_experts_by_sigma_buckets:
        metadata["ss_specialize_experts_by_sigma_buckets"] = "true"
        metadata["ss_num_sigma_buckets"] = str(int(cfg.num_sigma_buckets))
        if cfg.sigma_bucket_boundaries is not None:
            metadata["ss_sigma_bucket_boundaries"] = json.dumps(
                list(cfg.sigma_bucket_boundaries)
            )

    # Three-axis routing config (plan2 three-axis config). Stamped on every
    # save so the loader can reconstruct the exact router layout without
    # key-sniffing.
    if cfg.use_moe_style is not False:
        metadata["ss_use_moe_style"] = str(cfg.use_moe_style)
        metadata["ss_route_per_layer"] = (
            "true" if cfg.route_per_layer else "false"
        )
        metadata["ss_router_source"] = str(cfg.router_source)

    # T-LoRA is training-only (mask not in weights). Stamp the schedule so
    # continue-training / experiment replay can recover the exact prior even
    # when ss_network_args was not written.
    if cfg.use_timestep_mask:
        metadata["ss_use_timestep_mask"] = "true"
        metadata["ss_min_rank"] = str(int(cfg.min_rank))
        metadata["ss_alpha_rank_scale"] = str(float(cfg.alpha_rank_scale))

    if spec.name == "vera":
        plugin_args = getattr(cfg, "plugin_args", {}) or {}
        projection_key = plugin_args.get(
            "vera_projection_prng_key",
            plugin_args.get("projection_prng_key", 0),
        )
        d_initial = plugin_args.get(
            "vera_d_initial",
            plugin_args.get("d_initial", 0.1),
        )
        save_projection = plugin_args.get(
            "vera_save_projection",
            plugin_args.get("save_projection", False),
        )
        metadata["ss_vera_projection_prng_key"] = str(int(projection_key))
        metadata["ss_vera_d_initial"] = str(float(d_initial))
        metadata["ss_vera_save_projection"] = (
            "true"
            if str(save_projection).strip().lower() in {"1", "true", "yes", "on"}
            else "false"
        )

    if getattr(cfg, "ortho_centered_gate", False):
        metadata["ss_ortho_centered_gate"] = "true"

    if getattr(cfg, "use_dora", False):
        metadata["ss_network_spec"] = "dora"
        metadata["ss_adapter_variant"] = "dora"
        metadata["ss_dora_compatible_export"] = "true"

    if cfg.num_registers > 0:
        metadata["ss_num_registers"] = str(int(cfg.num_registers))
        metadata["ss_register_insert_block"] = str(int(cfg.register_insert_block))

    # FEI router params: router-source-specific scalars the loader needs to
    # size the router input.
    if cfg.router_source == "fei" and cfg.fei_feature_dim > 0:
        metadata["ss_fei_feature_dim"] = str(int(cfg.fei_feature_dim))
        metadata["ss_fei_sigma_low_div"] = str(float(cfg.fei_sigma_low_div))

    # ChimeraHydra: pool split and router flags that cannot be reconstructed
    # purely from the tensor keys.
    if cfg.use_chimera_hydra:
        metadata["ss_use_chimera_hydra"] = "true"
        metadata["ss_num_experts_content"] = str(int(cfg.num_experts_content))
        metadata["ss_num_experts_freq"] = str(int(cfg.num_experts_freq))
        metadata["ss_chimera_fei_feature_dim"] = str(int(cfg.fei_feature_dim))
        metadata["ss_chimera_sigma_feature_dim"] = str(int(cfg.sigma_feature_dim))
        metadata["ss_chimera_fei_sigma_low_div"] = str(float(cfg.fei_sigma_low_div))
        metadata["ss_chimera_freq_router_layer_norm"] = (
            "true" if cfg.freq_router_layer_norm else "false"
        )
        metadata["ss_chimera_content_router_source"] = str(cfg.content_router_source)
        if cfg.content_router_source == "crossattn_emb":
            metadata["ss_chimera_content_router_layer_norm"] = (
                "true" if cfg.content_router_layer_norm else "false"
            )
        if getattr(cfg, "chimera_centered_gate", False):
            metadata["ss_chimera_centered_gate"] = "true"

    # Family-aware target containers (Krea-2-Raw migration). Stamp only when
    # the cfg field is non-None — anima's None sentinel means "use the
    # ANIMA_TARGET_REPLACE_MODULE class default", so we omit the key to keep
    # anima checkpoints byte-identical and let old unstamped checkpoints fall
    # back to the anima default at load (from_weights leaves the field None).
    if cfg.unet_target_replace_modules is not None:
        metadata["ss_unet_target_replace_modules"] = json.dumps(
            list(cfg.unet_target_replace_modules)
        )
    if cfg.text_encoder_target_replace_modules is not None:
        metadata["ss_text_encoder_target_replace_modules"] = json.dumps(
            list(cfg.text_encoder_target_replace_modules)
        )

    # Model family stamp (Krea-2-Raw migration). Stamped only for non-anima
    # families so anima checkpoints stay byte-identical (anima is the implicit
    # default at load — absence is read as "anima"). The inference loader has
    # no ``args``; this is the only way it can dispatch DiT / text-encoder /
    # forward path from the checkpoint alone.
    model_family = normalize_model_family(
        getattr(cfg, "model_family", "anima") or "anima",
        source="LoRA checkpoint model_family",
    )
    if model_family != "anima":
        metadata["ss_model_family"] = model_family
    elif (
        network is not None
        and (layout := getattr(network, "_anima_checkpoint_layout", None)) is not None
        and layout.num_blocks == 40
    ):
        from library.anima.compat import adapter_identity_metadata

        metadata.update(
            adapter_identity_metadata(
                layout,
                getattr(network, "_anima_base_sha256", None),
            )
        )


def strip_orig_mod_keys(state_dict):
    """Strip torch.compile '_orig_mod_' from state_dict keys."""
    new_sd = {}
    for key, val in state_dict.items():
        new_key = re.sub(r"(?<=_)_orig_mod_", "", key)
        new_sd[new_key] = val
    return new_sd


def load_lora_network_weights(network, file):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        from library.anima.compat import validate_adapter_metadata

        weights_sd = load_file(file)
        with safe_open(file, framework="pt") as handle:
            file_metadata = dict(handle.metadata() or {})
        # Resume / warm-start must agree on family before any tensor lands in
        # the network: the cfg family selected the target containers at build
        # time, so a foreign checkpoint can only half-load.
        from networks.lora_anima.factory import require_adapter_family

        require_adapter_family(
            file_metadata,
            getattr(getattr(network, "cfg", None), "model_family", None),
            checkpoint=file,
        )
        validate_adapter_metadata(file_metadata, network)
    else:
        weights_sd = torch.load(file, map_location="cpu")

    # save_network_weights relays adaln keys to the ComfyUI layout
    # (adaln_up_{br} -> adaln_modulation_{br}_2), so a resume/soup/init load of
    # our own checkpoint must rename them back or every adaln module lands in
    # missing_keys and silently trains from scratch. Presence-gated — mirrors
    # create_network_from_weights (factory.py).
    from networks.lora_utils import (
        has_comfy_adaln_keys,
        relayout_adaln_comfy_to_runtime,
    )

    if has_comfy_adaln_keys(weights_sd):
        weights_sd = relayout_adaln_comfy_to_runtime(weights_sd)

    # Stack per-expert hydra ups into fused lora_up_weight (training form).
    # Also stacks per-expert `.lora_downs.{i}.weight` for the independent-A
    # layout. No-op for plain LoRA.
    weights_sd = _stack_lora_ups(weights_sd)
    weights_sd = _stack_chimera_lora_ups(weights_sd)
    # Refuse split stacked-experts first; its discriminator is per-expert
    # `lora_down_weight` 3-D, which the hydra refuser would otherwise
    # short-circuit on the absent shared `lora_down.weight`.
    weights_sd = _refuse_split_stacked_experts_keys(weights_sd)
    weights_sd = _refuse_split_hydra_keys(weights_sd)
    weights_sd = _refuse_split_chimera_keys(weights_sd)
    weights_sd = _refuse_unfused_attn_lora_keys(weights_sd)
    weights_sd = _rename_dora_scale_for_load(weights_sd)
    weights_sd = preprocess_weights_from_plugins(weights_sd)

    reabsorb_baked_inv_scale(network, weights_sd)

    return network.load_state_dict(weights_sd, False)


def reabsorb_baked_inv_scale(network, weights_sd: Dict[str, torch.Tensor]) -> None:
    """Resume guard for baked (inv_scale-folded) checkpoints."""
    for lora in network.unet_loras + network.text_encoder_loras:
        if not getattr(lora, "_has_channel_scale", False):
            continue
        name = lora.lora_name
        down_key = f"{name}.lora_down.weight"
        if f"{name}.inv_scale" in weights_sd or down_key not in weights_sd:
            continue
        inv_scale = lora.inv_scale  # (in,) fp32, == 1/s_norm
        down = weights_sd[down_key]
        s_norm = (
            inv_scale.to(device=down.device, dtype=torch.float)
            .clamp_min(1e-12)
            .reciprocal()
        )
        weights_sd[down_key] = (
            down.to(torch.float) * s_norm.unsqueeze(0)
        ).to(down.dtype)
        weights_sd[f"{name}.inv_scale"] = inv_scale.clone()


def save_lora_network_weights(network, file, dtype, metadata) -> None:
    spec: NetworkSpec = getattr(network, "_network_spec", NETWORK_REGISTRY["lora"])
    if metadata is None:
        metadata = {}
    stamp_lora_save_metadata(metadata, network.cfg, spec, network=network)

    lora_save.save_network_weights(
        network.state_dict(),
        file=file,
        dtype=dtype,
        metadata=metadata,
        save_variant=spec.save_variant,
    )
