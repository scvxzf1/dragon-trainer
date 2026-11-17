"""Model loading for Anima inference: DiT, text encoder, shared model management."""

import argparse
from dataclasses import dataclass
import logging
from typing import Optional, Dict

import torch
from safetensors.torch import load_file

from library.anima import models as anima_models, weights as anima_utils
from library.env import resolve_model_family
from library.inference.precision import resolve_runtime_dtype, resolve_text_encoder_dtype
from library.inference.selective_lora import (
    ANIMA_SELECTIVE_BLOCKS,
    apply_anima_selective_lora,
    enabled_blocks_from_anima_selective_strengths,
    normalize_anima_selective_block_strengths,
    normalize_anima_selective_blocks,
    normalize_anima_selective_preset,
    preset_strength_for_anima_selective,
)
from library.runtime.device import clean_memory_on_device

logger = logging.getLogger(__name__)

_TRUE_METADATA_VALUES = {"1", "true", "True"}
_NETWORK_MERGE_KINDS = {"DoRA", "GLoRA", "LoHa", "LoKr", "VeRA"}


@dataclass(frozen=True)
class AdapterCapability:
    path: str
    kind: str
    supports_static_merge: bool
    requires_dynamic_hook: bool
    exclusive: bool = False


def _metadata_flag_enabled(
    metadata: dict, key: str, *, case_insensitive_true: bool = False
) -> bool:
    value = str(metadata.get(key, "")).strip()
    if case_insensitive_true:
        return value.lower() == "true"
    return value in _TRUE_METADATA_VALUES


def _read_lora_header(path: str) -> tuple[list[str], dict[str, str]]:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        return list(f.keys()), dict(f.metadata() or {})


def _read_lora_metadata(path: str) -> dict[str, str]:
    try:
        _, metadata = _read_lora_header(path)
    except Exception:
        return {}
    return metadata


def _classify_adapter_capability(path: str) -> AdapterCapability:
    try:
        keys, metadata = _read_lora_header(path)
    except Exception:
        return AdapterCapability(
            path=path,
            kind="LoRA",
            supports_static_merge=True,
            requires_dynamic_hook=False,
        )

    lowered_keys = [key.lower() for key in keys]
    spec_name = str(metadata.get("ss_network_spec") or "").strip().lower()
    if _metadata_flag_enabled(metadata, "ss_turbo_per_step_expert") or spec_name == "step_expert":
        return AdapterCapability(
            path=path,
            kind="StepExpert LoRA",
            supports_static_merge=False,
            requires_dynamic_hook=True,
            exclusive=True,
        )

    if any(
        ".lora_ups." in key or ".lora_ups_c." in key or ".lora_ups_f." in key
        for key in lowered_keys
    ):
        return AdapterCapability(
            path=path,
            kind="HydraLoRA",
            supports_static_merge=False,
            requires_dynamic_hook=True,
        )

    from networks import continue_weight_kind_from_plugins

    if spec_name == "dora" or any(
        key.endswith((".magnitude", ".dora_scale", ".dora_magnitude"))
        for key in lowered_keys
    ):
        return AdapterCapability(
            path=path,
            kind="DoRA",
            supports_static_merge=True,
            requires_dynamic_hook=False,
        )

    plugin_kind = continue_weight_kind_from_plugins(keys, metadata)
    if plugin_kind in _NETWORK_MERGE_KINDS:
        return AdapterCapability(
            path=path,
            kind=plugin_kind,
            supports_static_merge=True,
            requires_dynamic_hook=False,
        )

    return AdapterCapability(
        path=path,
        kind="LoRA",
        supports_static_merge=True,
        requires_dynamic_hook=False,
    )


def _classify_adapter_capabilities(paths: list[str] | None) -> list[AdapterCapability]:
    if not paths:
        return []
    return [_classify_adapter_capability(str(path)) for path in paths]


def _validate_adapter_capabilities(capabilities: list[AdapterCapability]) -> None:
    exclusive = [cap for cap in capabilities if cap.exclusive]
    if exclusive and (len(capabilities) > 1 or len(exclusive) > 1):
        raise ValueError(
            "Per-step-expert turbo must be loaded alone. Composing it "
            "with other LoRAs or static merge is unsupported."
        )

    hydra = [cap for cap in capabilities if cap.kind == "HydraLoRA"]
    if hydra and len(hydra) != len(capabilities):
        raise ValueError(
            "Mixing HydraLoRA moe files with regular LoRA files in a "
            "single --lora_weight list is not supported. The static "
            "merge + dynamic hook interaction is untested. Pass them "
            "in separate invocations."
        )


def _network_merge_capabilities(
    capabilities: list[AdapterCapability],
) -> list[AdapterCapability]:
    return [
        cap
        for cap in capabilities
        if cap.supports_static_merge and cap.kind in _NETWORK_MERGE_KINDS
    ]


def _has_te_keys(path: str) -> bool:
    """Cheap header peek: does this safetensors LoRA carry any ``lora_te_*`` keys?

    Lets ``load_text_encoder`` skip a redundant ``load_file`` + empty-dict merge
    when the LoRA is DiT-only (the common case — turbo, plain LoRA, postfix, …).
    Returns False on any read error so the caller falls back to the no-LoRA TE
    load path (a truly broken file would have already tripped the DiT loader).
    """
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt") as f:
            return any(k.startswith("lora_te_") for k in f.keys())
    except Exception:
        return False


def _is_chimera_moe(path: str) -> bool:
    """Peek at safetensors metadata for ``ss_use_chimera_hydra="true"``.

    Chimera files share the Hydra-MoE on-disk shape but carry the dual-pool
    runtime contract — they
    additionally hold a top-level ``freq_router.*`` block and need the
    per-Linear router narrowed to K_c outputs. Inference / load paths
    must read this flag to wire the network correctly.
    """
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt") as f:
            md = f.metadata() or {}
            return _metadata_flag_enabled(
                md, "ss_use_chimera_hydra", case_insensitive_true=True
            )
    except Exception:
        return False


def _resolve_lora_multiplier_for_index(
    lora_multiplier: float | list[float] | None, index: int
) -> float:
    if isinstance(lora_multiplier, (int, float)):
        return float(lora_multiplier)
    if not lora_multiplier:
        return 1.0
    if len(lora_multiplier) == 1:
        return float(lora_multiplier[0])
    if index < len(lora_multiplier):
        return float(lora_multiplier[index])
    return float(lora_multiplier[-1])


def _apply_selective_lora_controls(
    args: argparse.Namespace, lora_sd: Dict[str, torch.Tensor], *, path: str
) -> Dict[str, torch.Tensor]:
    if not getattr(args, "anima_selective_lora", False):
        return lora_sd

    preset = normalize_anima_selective_preset(
        getattr(args, "anima_selective_preset", "default")
    )
    raw_block_strengths = getattr(args, "anima_selective_block_strengths", None)
    block_strengths = normalize_anima_selective_block_strengths(
        raw_block_strengths,
        preset=preset,
    )
    selected_blocks = enabled_blocks_from_anima_selective_strengths(
        block_strengths,
        preset=preset,
    )
    if not selected_blocks:
        selected_blocks = normalize_anima_selective_blocks(
            getattr(args, "anima_selective_blocks", None),
            preset=preset,
        )
    strength = float(getattr(args, "anima_selective_strength", 1.0) or 1.0)
    effective_strength = (
        strength
        if raw_block_strengths is not None
        else strength * preset_strength_for_anima_selective(preset)
    )
    filtered = apply_anima_selective_lora(
        lora_sd,
        selected_blocks,
        strength=effective_strength,
        preset=preset,
        block_strengths=block_strengths,
    )
    logger.info(
        "Anima selective LoRA enabled for %s: preset=%s, blocks=%s/%s, strength=%.3f, tensors=%s/%s",
        path,
        preset,
        len(selected_blocks),
        len(ANIMA_SELECTIVE_BLOCKS),
        effective_strength,
        len(filtered),
        len(lora_sd),
    )
    return filtered


def _load_lora_state_dict_for_inference(
    args: argparse.Namespace, path: str
) -> Dict[str, torch.Tensor]:
    lora_sd = load_file(path)
    return _apply_selective_lora_controls(args, lora_sd, path=path)


def attach_adapters(
    model: anima_models.Anima,
    args: argparse.Namespace,
    device: torch.device,
    *,
    pgraft_mode: bool,
    hydra_mode: bool,
    step_expert_mode: bool = False,
) -> None:
    """Attach LoRA-family adapters that ride as dynamic forward hooks.

    Covers the two routes that can't go through ``load_anima_model``'s static
    merge: **P-GRAFT** (toggleable mid-denoising) and **HydraLoRA moe / chimera**
    (router-live, runs per-sample). Both rehydrate a network, ``apply_to`` the
    already-loaded ``model`` in place, and stash it on ``model`` for the sampler
    toggle sites to find. No-op when neither mode is set. Mutates ``model``;
    returns nothing. The static-merge path and ``torch.compile`` stay in
    :func:`load_dit_model` — this does only the dynamic-hook attach.

    ``pgraft_mode`` / ``hydra_mode`` are passed in (not recomputed) because the
    caller already derives them to decide whether to skip the static merge.
    """
    runtime_dtype = resolve_runtime_dtype(args)

    # P-GRAFT: attach LoRA as dynamic hooks (can be toggled mid-denoising)
    if pgraft_mode and not hydra_mode and not step_expert_mode:
        from networks import lora_anima

        logger.info("P-GRAFT: Loading LoRA as dynamic hooks (not static merge)")
        for index, lora_weight_path in enumerate(args.lora_weight):
            lora_sd = _load_lora_state_dict_for_inference(args, lora_weight_path)
            lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
            if not lora_sd:
                logger.warning(
                    "P-GRAFT: no DiT LoRA tensors left after filtering, skip %s",
                    lora_weight_path,
                )
                continue

            multiplier = _resolve_lora_multiplier_for_index(
                args.lora_multiplier, index
            )
            network, weights_sd = lora_anima.create_network_from_weights(
                multiplier=multiplier,
                file=lora_weight_path,
                ae=None,
                text_encoders=[],
                unet=model,
                weights_sd=lora_sd,
                metadata=_read_lora_metadata(lora_weight_path),
                for_inference=True,
                model_family=resolve_model_family(args),
            )
            network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
            info = network.load_state_dict(weights_sd, strict=False)
            if info.unexpected_keys:
                logger.debug(
                    f"P-GRAFT: unexpected keys in LoRA state dict: {info.unexpected_keys[:5]}..."
                )
            network.to(device, dtype=runtime_dtype)
            network.eval()
            model._pgraft_network = network
            logger.info(
                f"P-GRAFT: LoRA attached with cutoff_step={getattr(args, 'lora_cutoff_step', None)}"
            )

    # HydraLoRA moe: rehydrate the trained router-live network and attach it
    # as dynamic forward hooks, identical shape to the P-GRAFT path above.
    # The router runs per-sample on each adapted module, so the net stays in
    # eval mode with requires_grad_(False).
    if hydra_mode:
        from networks import lora_anima

        logger.info("HydraLoRA: loading moe file as router-live dynamic hooks")
        from safetensors import safe_open

        for index, lora_weight_path in enumerate(args.lora_weight):
            # Read the three-axis routing stamps (and chimera stamps) from
            # on-disk __metadata__ — load_file() drops it. Chimera files
            # (dual-pool) carry top-level ``freq_router.*`` keys outside the
            # ``lora_unet_*`` namespace, so they must NOT be filtered; plain
            # Hydra moe keeps the lora_unet_* filter. Passing ``metadata=``
            # alongside ``weights_sd=`` lets both layouts go through one code
            # path — no more file=path vs weights_sd= fork.
            with safe_open(lora_weight_path, framework="pt") as f:
                lora_metadata = dict(f.metadata() or {})
            is_chimera = _is_chimera_moe(lora_weight_path)
            lora_sd = _load_lora_state_dict_for_inference(args, lora_weight_path)
            if is_chimera:
                logger.info("HydraLoRA: chimera file — dual-pool routing wired")
            lora_sd = {
                k: v for k, v in lora_sd.items() if not k.startswith("lora_te_")
            }
            if not lora_sd:
                logger.warning(
                    "HydraLoRA: no tensors left after filtering, skip %s",
                    lora_weight_path,
                )
                continue

            multiplier = _resolve_lora_multiplier_for_index(
                args.lora_multiplier, index
            )
            network, weights_sd = lora_anima.create_network_from_weights(
                multiplier=multiplier,
                file=None,
                ae=None,
                text_encoders=[],
                unet=model,
                weights_sd=lora_sd,
                metadata=lora_metadata,
                for_inference=True,
                model_family=resolve_model_family(args),
            )
            network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
            info = network.load_state_dict(weights_sd, strict=False)
            if info.unexpected_keys:
                logger.warning(
                    f"HydraLoRA: unexpected keys in state dict: {info.unexpected_keys[:5]}..."
                )
            if info.missing_keys:
                logger.warning(
                    f"HydraLoRA: missing keys in state dict: {info.missing_keys[:5]}..."
                )
            network.to(device, dtype=runtime_dtype)
            network.eval().requires_grad_(False)
            hydra_networks = list(getattr(model, "_hydra_networks", []))
            hydra_networks.append(network)
            model._hydra_networks = hydra_networks
            model._hydra_network = network
            # Reuse the P-GRAFT cutoff slot so existing toggle sites
            # (inference_pipeline loops + spectrum_denoise) honor
            # --lora_cutoff_step without further plumbing.
            model._pgraft_network = network
            logger.info(
                f"HydraLoRA: router-live attached "
                f"({len(network.unet_loras)} modules, "
                f"cutoff_step={getattr(args, 'lora_cutoff_step', None)})"
            )

    # Per-step-expert turbo: K up-heads cannot be merged into one static DiT
    # weight. Attach it dynamically and let generation.py select head i.
    if step_expert_mode:
        from safetensors import safe_open

        from networks.methods.turbo_dmd import load_step_expert_student

        logger.info("step-expert turbo: loading as router-free kept-live hooks")
        for index, lora_weight_path in enumerate(args.lora_weight):
            with safe_open(lora_weight_path, framework="pt") as f:
                se_metadata = dict(f.metadata() or {})
            lora_sd = _load_lora_state_dict_for_inference(args, lora_weight_path)
            lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
            if not lora_sd:
                logger.warning(
                    "step-expert turbo: no DiT LoRA tensors left after filtering, skip %s",
                    lora_weight_path,
                )
                continue
            multiplier = _resolve_lora_multiplier_for_index(
                args.lora_multiplier, index
            )
            network = load_step_expert_student(
                model, lora_sd, se_metadata, multiplier=multiplier
            )
            network.to(device, dtype=runtime_dtype)
            network.eval().requires_grad_(False)
            step_nets = list(getattr(model, "_step_expert_networks", []))
            step_nets.append(network)
            model._step_expert_networks = step_nets


def _merge_network_adapters_into_model(
    model: anima_models.Anima,
    args: argparse.Namespace,
    capabilities: list[AdapterCapability],
    *,
    device: torch.device,
    dtype: Optional[torch.dtype],
) -> None:
    merge_caps = _network_merge_capabilities(capabilities)
    if not merge_caps:
        return

    from networks import lora_anima

    for index, capability in enumerate(capabilities):
        if capability not in merge_caps:
            continue
        lora_weight_path = capability.path
        logger.info(
            "Merging %s adapter through network merge path: %s",
            capability.kind,
            lora_weight_path,
        )
        lora_sd = _load_lora_state_dict_for_inference(args, lora_weight_path)
        lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
        if not lora_sd:
            logger.warning(
                "%s: no DiT adapter tensors left after filtering, skip %s",
                capability.kind,
                lora_weight_path,
            )
            continue
        multiplier = _resolve_lora_multiplier_for_index(args.lora_multiplier, index)
        network, weights_sd = lora_anima.create_network_from_weights(
            multiplier=multiplier,
            file=lora_weight_path,
            ae=None,
            text_encoders=[],
            unet=model,
            weights_sd=lora_sd,
            metadata=_read_lora_metadata(lora_weight_path),
            for_inference=True,
            model_family=resolve_model_family(args),
        )
        network.merge_to(None, model, weights_sd, dtype=dtype, device=device)


def load_dit_model(
    args: argparse.Namespace,
    device: torch.device,
    dit_weight_dtype: Optional[torch.dtype] = None,
) -> anima_models.Anima:
    """Load DiT model with optional LoRA merge, P-GRAFT hooks, and torch.compile.

    Namespace-driven adapter over the explicit-argument primitive
    ``library.anima.weights.load_anima_model``: it pulls ``dit``/``attn_mode``/
    ``lora_weight``/etc. off ``args``, then hands the dynamic-hook adapter attach
    to :func:`attach_adapters` and applies ``torch.compile``. Reach for
    ``load_anima_model`` directly when you want just the weights and no Namespace.
    """

    loading_device = device

    capabilities = _classify_adapter_capabilities(args.lora_weight)
    _validate_adapter_capabilities(capabilities)
    step_expert_mode = any(cap.kind == "StepExpert LoRA" for cap in capabilities)
    hydra_mode = any(cap.kind == "HydraLoRA" for cap in capabilities)

    # P-GRAFT: load without LoRA merge, attach dynamic hooks instead
    pgraft_mode = (
        getattr(args, "pgraft", False)
        and args.lora_weight is not None
        and len(args.lora_weight) > 0
    )

    # Plain LoRA keeps the memory-efficient base-load merge. Richer static
    # variants (DoRA / LoHa / LoKr / VeRA / GLoRA) need their module-owned
    # merge_to implementation, so they are merged after the base DiT is loaded.
    if (
        not pgraft_mode
        and not hydra_mode
        and not step_expert_mode
        and capabilities
    ):
        lora_weights_list = []
        lora_multipliers = []
        for index, lora_weight in enumerate(args.lora_weight):
            capability = capabilities[index]
            if capability.kind != "LoRA":
                logger.info(
                    "Skip legacy static LoRA merge for %s adapter: %s",
                    capability.kind,
                    lora_weight,
                )
                continue
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = _load_lora_state_dict_for_inference(
                args, lora_weight
            )  # load on CPU, dtype is as is
            lora_sd = {
                k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")
            }  # only keep unet lora weights
            if not lora_sd:
                logger.warning(
                    "No DiT LoRA tensors left after filtering, skip static merge: %s",
                    lora_weight,
                )
                continue
            lora_weights_list.append(lora_sd)
            lora_multipliers.append(
                _resolve_lora_multiplier_for_index(args.lora_multiplier, index)
            )
        if not lora_weights_list:
            lora_weights_list = None
            lora_multipliers = None
    else:
        lora_weights_list = None
        lora_multipliers = None

    model = anima_utils.load_anima_model(
        device,
        args.dit,
        args.attn_mode,
        loading_device,
        dit_weight_dtype,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
    )

    # Modulation guidance: load trained pooled_text_proj weights before .to()
    # (pooled_text_proj params are meta tensors when not in the pretrained checkpoint)
    pooled_text_proj_path = getattr(args, "pooled_text_proj", None)
    if pooled_text_proj_path is not None:
        anima_utils.load_pooled_text_proj(model, pooled_text_proj_path, "cpu")

    target_dtype = dit_weight_dtype
    if target_dtype is not None:
        logger.info(f"Convert model to {target_dtype}")
        logger.info(f"Move model to device: {device}")
        model.to(device, dtype=target_dtype)
    else:
        logger.info(f"Move model to device: {device}")
        model.to(device)

    if not pgraft_mode:
        _merge_network_adapters_into_model(
            model,
            args,
            capabilities,
            device=device,
            dtype=target_dtype,
        )

    model.eval().requires_grad_(False)

    # Dynamic-hook adapters (P-GRAFT toggle / HydraLoRA router-live /
    # step-expert turbo) that can't ride the static merge above.
    attach_adapters(
        model,
        args,
        device,
        pgraft_mode=pgraft_mode,
        hydra_mode=hydra_mode,
        step_expert_mode=step_expert_mode,
    )

    if getattr(args, "compile", False):
        logger.info("Compiling DiT model with torch.compile...")
        model = torch.compile(model)
    elif getattr(args, "compile_blocks", False):
        model.compile_blocks(mode=getattr(args, "compile_inductor_mode", None))

    clean_memory_on_device(device)

    return model


def load_text_encoder(
    args: argparse.Namespace,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    lora_weights_list = None
    lora_multipliers = None
    if (
        args.lora_weight is not None
        and len(args.lora_weight) > 0
        and any(_has_te_keys(p) for p in args.lora_weight)
    ):
        lora_weights_list = []
        lora_multipliers = []
        for index, lora_weight in enumerate(args.lora_weight):
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = _load_lora_state_dict_for_inference(
                args, lora_weight
            )  # load on CPU, dtype is as is
            lora_sd = {
                "model_" + k[len("lora_te_") :]: v
                for k, v in lora_sd.items()
                if k.startswith("lora_te_")
            }  # only keep Text Encoder lora weights, remove prefix "lora_te_" and add "model_" prefix
            if not lora_sd:
                logger.warning(
                    "No Text Encoder LoRA tensors left after filtering, skip merge: %s",
                    lora_weight,
                )
                continue
            lora_weights_list.append(lora_sd)
            lora_multipliers.append(
                _resolve_lora_multiplier_for_index(args.lora_multiplier, index)
            )
        if not lora_weights_list:
            lora_weights_list = None
            lora_multipliers = None
    text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        args.text_encoder,
        dtype=dtype,
        device=device,
        lora_weights=lora_weights_list,
        lora_multipliers=lora_multipliers,
    )
    text_encoder.eval()
    return text_encoder


def load_shared_models(args: argparse.Namespace) -> Dict:
    """Load shared models for batch processing or interactive mode.
    Models are loaded to CPU to save memory. VAE is NOT loaded here.
    DiT model is also NOT loaded here, handled by process_batch_prompts or generate.
    """
    shared_models = {}
    text_encoder_dtype = resolve_text_encoder_dtype(args)
    text_encoder = load_text_encoder(
        args, dtype=text_encoder_dtype, device=torch.device("cpu")
    )
    shared_models["text_encoder"] = text_encoder
    return shared_models
