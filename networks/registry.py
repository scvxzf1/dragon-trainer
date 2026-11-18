"""NetworkSpec registry bootstrap and resolve APIs.

Types and tables live in :mod:`networks.registry_api`. This module owns
plugin loading and kwargs-based resolve helpers.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any, Mapping, Optional, Tuple

from library.config.normalize import parse_bool

from .registry_api import (
    ContinueWeightDetectionContext,
    ModuleCreationContext,
    NETWORK_REGISTRY,
    NetworkSpec,
    SAVE_HANDLERS,
    SHARED_KWARG_FLAGS,
    WeightDetectionContext,
    register_network_spec,
    register_save_handler,
)

logger = logging.getLogger(__name__)

_PLUGINS_LOADED = False


def all_network_kwargs() -> Tuple[str, ...]:
    """Return the union of shared + per-variant kwargs, sorted."""

    ensure_builtin_plugins_loaded()
    merged: set[str] = set(SHARED_KWARG_FLAGS)
    for spec in NETWORK_REGISTRY.values():
        merged.update(spec.kwarg_flags)
    return tuple(sorted(merged))


def _parse_bool_flag(kwargs: Mapping[str, Any], key: str) -> bool:
    return parse_bool(kwargs.get(key, False), default=False)


def ensure_builtin_plugins_loaded() -> None:
    """Import bundled plugin packages so they can register their specs."""

    global _PLUGINS_LOADED
    from .core_specs import register_core_network_specs

    register_core_network_specs()
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        plugins_pkg = importlib.import_module(f"{__package__}.plugins")
    except ModuleNotFoundError:
        return

    for module_info in pkgutil.iter_modules(
        plugins_pkg.__path__, plugins_pkg.__name__ + "."
    ):
        if module_info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        try:
            importlib.import_module(module_info.name)
        except Exception:
            logger.exception("Failed to import network plugin %s", module_info.name)
            raise


def resolve_network_spec(kwargs: Mapping[str, Any]) -> NetworkSpec:
    """Resolve which NetworkSpec to instantiate from create_network kwargs."""

    ensure_builtin_plugins_loaded()

    use_dora = _parse_bool_flag(kwargs, "dora_wd")
    use_ortho = _parse_bool_flag(kwargs, "use_ortho")
    use_chimera = _parse_bool_flag(kwargs, "use_chimera_hydra")

    for spec in NETWORK_REGISTRY.values():
        if spec.name in {
            "lora",
            "ortho",
            "dora",
            "hydra",
            "ortho_hydra",
            "chimera_hydra",
            "stacked_experts_global_fei",
        }:
            continue
        if spec.validate is not None:
            spec.validate(kwargs)
        if spec.selector is not None and spec.selector(kwargs):
            return spec

    if use_chimera:
        if use_dora:
            raise ValueError("dora_wd=True is incompatible with ChimeraHydra.")
        return NETWORK_REGISTRY["chimera_hydra"]

    raw_moe = kwargs.get("use_moe_style")
    if isinstance(raw_moe, str):
        moe_style = raw_moe.strip()
        if moe_style.lower() in ("false", "none", ""):
            moe_style = ""
    elif raw_moe is False or raw_moe is None:
        moe_style = ""
    else:
        raise ValueError(
            f"use_moe_style={raw_moe!r}: expected False, 'shared_A', or 'independent_A'."
        )
    if moe_style not in ("", "shared_A", "independent_A"):
        raise ValueError(
            f"use_moe_style={raw_moe!r}: expected False, 'shared_A', or 'independent_A'."
        )

    if use_dora:
        if use_ortho:
            raise ValueError("dora_wd=True is incompatible with use_ortho=True.")
        if moe_style:
            raise ValueError("dora_wd=True is only supported for plain LoRA.")
        return NETWORK_REGISTRY["dora"]

    if moe_style == "independent_A":
        return NETWORK_REGISTRY["stacked_experts_global_fei"]
    if moe_style == "shared_A":
        return (
            NETWORK_REGISTRY["ortho_hydra"] if use_ortho else NETWORK_REGISTRY["hydra"]
        )
    if use_ortho:
        return NETWORK_REGISTRY["ortho"]
    return NETWORK_REGISTRY["lora"]


def detect_network_spec_from_weights(
    weights_sd: Mapping[str, Any],
    metadata: Mapping[str, str],
    modules_dim: dict[str, int],
    modules_alpha: dict[str, Any],
    plain_module_names: set[str],
) -> dict[str, Any]:
    """Run plugin checkpoint sniffers and return their accumulated state."""

    ensure_builtin_plugins_loaded()
    state: dict[str, Any] = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        ctx = WeightDetectionContext(
            key=key,
            value=value,
            lora_name=lora_name,
            metadata=metadata,
            modules_dim=modules_dim,
            modules_alpha=modules_alpha,
            plain_module_names=plain_module_names,
            state=state,
        )
        for spec in NETWORK_REGISTRY.values():
            if spec.detect_from_weights is not None:
                spec.detect_from_weights(ctx)
    for spec in NETWORK_REGISTRY.values():
        if spec.finish_weight_detection is not None:
            state.update(
                spec.finish_weight_detection(state, modules_dim, modules_alpha)
            )
    return state


def preprocess_weights_from_plugins(weights_sd: dict[str, Any]) -> dict[str, Any]:
    """Let plugin variants rewrite checkpoint keys before core sniffing."""

    ensure_builtin_plugins_loaded()
    for spec in NETWORK_REGISTRY.values():
        if spec.preprocess_weights is not None:
            weights_sd = spec.preprocess_weights(weights_sd)
    return weights_sd


def continue_weight_kind_from_plugins(
    keys: list[str],
    metadata: Mapping[str, str],
) -> Optional[str]:
    ensure_builtin_plugins_loaded()
    lowered_keys = [str(key).lower() for key in keys]
    ctx = ContinueWeightDetectionContext(
        keys=keys,
        metadata=metadata,
        lowered_keys=lowered_keys,
    )
    for spec in NETWORK_REGISTRY.values():
        if spec.continue_weight_kind is None:
            continue
        kind = spec.continue_weight_kind(ctx)
        if kind:
            return kind
    return None


ensure_builtin_plugins_loaded()


__all__ = [
    "ContinueWeightDetectionContext",
    "ModuleCreationContext",
    "NETWORK_REGISTRY",
    "NetworkSpec",
    "SAVE_HANDLERS",
    "SHARED_KWARG_FLAGS",
    "WeightDetectionContext",
    "all_network_kwargs",
    "continue_weight_kind_from_plugins",
    "detect_network_spec_from_weights",
    "ensure_builtin_plugins_loaded",
    "preprocess_weights_from_plugins",
    "register_network_spec",
    "register_save_handler",
    "resolve_network_spec",
]
