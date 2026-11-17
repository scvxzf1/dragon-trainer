"""Bundled LoKR plugin registration."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import torch

from .autograd import (
    DEFAULT_LOKR_GROUPED_DELTA_BACKEND,
    DEFAULT_LOKR_GROUPED_DELTA_BACKWARD_BACKEND,
    DEFAULT_LOKR_PROJECT_CHUNK_BYTES,
    normalize_lokr_grouped_delta_backward_backend,
    normalize_lokr_grouped_delta_backend,
)
from .module import LoKrModule
from .save import save_lokr_weights
from ...attn_fuse import iter_split_groups, match_fused_spec
from ...registry_api import (
    ContinueWeightDetectionContext,
    ModuleCreationContext,
    NetworkSpec,
    WeightDetectionContext,
    register_network_spec,
    register_save_handler,
)

DEFAULT_LOKR_RUNTIME_BACKEND = DEFAULT_LOKR_GROUPED_DELTA_BACKEND

logger = logging.getLogger(__name__)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _bool_arg(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return _truthy(value)


def _selector(kwargs: Mapping[str, Any]) -> bool:
    return _truthy(kwargs.get("use_lokr"))


def _validate(kwargs: Mapping[str, Any]) -> None:
    if not _selector(kwargs):
        return
    if (
        _truthy(kwargs.get("use_ortho"))
        or _truthy(kwargs.get("use_chimera_hydra"))
        or _truthy(kwargs.get("use_vera"))
        or _truthy(kwargs.get("dora_wd"))
        or kwargs.get("use_moe_style") not in (None, False, "", "false", "False")
    ):
        raise ValueError(
            "use_lokr is mutually exclusive with dora_wd, use_ortho, use_vera, "
            "use_moe_style, and use_chimera_hydra."
        )

    lokr_full_factor = _truthy(kwargs.get("lokr_full_factor"))
    decompose_w2 = _truthy(kwargs.get("lokr_decompose_w2"))
    use_einsum = _bool_arg(kwargs.get("lokr_use_einsum"), default=True)
    if lokr_full_factor and decompose_w2:
        raise ValueError(
            "lokr_full_factor=true conflicts with lokr_decompose_w2=true. "
            "Full-factor LoKR cannot also decompose the second Kronecker factor."
        )
    if not use_einsum and decompose_w2:
        raise ValueError(
            "lokr_decompose_w2=true requires lokr_use_einsum=true."
        )

    # Historical full-factor sentinel from older forks. It also collapses the
    # training scale to network_alpha / 114514, so new training must opt out.
    network_dim = kwargs.get("network_dim", None)
    try:
        network_dim_i = int(network_dim) if network_dim is not None else None
    except (TypeError, ValueError):
        network_dim_i = None
    if network_dim_i == 114514:
        network_alpha = kwargs.get("network_alpha", 1.0)
        try:
            network_alpha_f = float(network_alpha)
        except (TypeError, ValueError):
            network_alpha_f = 1.0
        legacy_scale = network_alpha_f / float(network_dim_i)
        message = (
            "LoKR network_dim=114514 is a deprecated full-factor sentinel. "
            f"It also sets training scale=network_alpha/network_dim={legacy_scale:.8g}, "
            "which suppresses adapter output and gradients. Use "
            "network_dim=32, network_alpha=32, lokr_full_factor=true instead."
        )
        if not _truthy(kwargs.get("lokr_allow_legacy_dim")):
            raise ValueError(
                message
                + " Set lokr_allow_legacy_dim=true only to resume an old state "
                "without changing its historical scale."
            )
        logger.warning(
            "%s Legacy compatibility was explicitly enabled; the suppressed "
            "scale is preserved.",
            message,
        )


def _module_kwargs(ctx: ModuleCreationContext) -> dict[str, Any]:
    full_factor = _truthy(ctx.cfg.plugin_args.get("lokr_full_factor"))
    if ctx.cfg.plugin_args.get("lokr_decompose_w2") is None:
        decompose_w2 = False
    else:
        decompose_w2 = _truthy(ctx.cfg.plugin_args.get("lokr_decompose_w2"))
    if full_factor:
        # Explicit full-factor mode always keeps both Kronecker factors complete.
        decompose_w2 = False
    explicit_map = ctx.cfg.plugin_args.get("_lokr_explicit_shapes")
    explicit_shapes = (
        explicit_map.get(ctx.lora_name)
        if isinstance(explicit_map, dict)
        else None
    )
    out = {
        "factor": int(ctx.cfg.plugin_args.get("lokr_factor", 8)),
        "lokr_factor_group_size": int(
            ctx.cfg.plugin_args.get("lokr_factor_group_size", 8)
        ),
        "lokr_project_chunk_bytes": int(
            ctx.cfg.plugin_args.get(
                "lokr_project_chunk_bytes", DEFAULT_LOKR_PROJECT_CHUNK_BYTES
            )
        ),
        "lokr_grouped_delta_backend": normalize_lokr_grouped_delta_backend(
            ctx.cfg.plugin_args.get(
                "lokr_grouped_delta_backend", DEFAULT_LOKR_RUNTIME_BACKEND
            )
        ),
        "lokr_grouped_delta_backward_backend": normalize_lokr_grouped_delta_backward_backend(
            ctx.cfg.plugin_args.get(
                "lokr_grouped_delta_backward_backend",
                DEFAULT_LOKR_GROUPED_DELTA_BACKWARD_BACKEND,
            )
        ),
        "lokr_use_einsum": _bool_arg(
            ctx.cfg.plugin_args.get("lokr_use_einsum"), default=True
        ),
        "lokr_decompose_w2": decompose_w2,
        "lokr_full_factor": full_factor,
    }
    if explicit_shapes is not None:
        out["explicit_shapes"] = explicit_shapes
    return out


def _detect_from_weights(ctx: WeightDetectionContext) -> bool:
    key = ctx.key
    value = ctx.value
    try:
        network_dim = int(float(ctx.metadata.get("ss_network_dim", "")))
    except (TypeError, ValueError):
        network_dim = None
    ctx.state["lokr_network_dim_meta"] = network_dim
    if "lokr_factor_stamp" not in ctx.state:
        raw_factor = ctx.metadata.get("ss_lokr_factor")
        if raw_factor is not None:
            try:
                ctx.state["lokr_factor_stamp"] = int(float(raw_factor))
            except (TypeError, ValueError):
                pass
    if "lokr_einsum_stamp" not in ctx.state:
        raw_einsum = ctx.metadata.get("ss_lokr_use_einsum")
        if raw_einsum is not None:
            ctx.state["lokr_einsum_stamp"] = (
                str(raw_einsum).strip().lower() not in {"0", "false", "no", "off"}
            )
    if "lokr_full_factor_stamp" not in ctx.state:
        stamped = ctx.metadata.get("ss_lokr_full_factor")
        if stamped is not None:
            ctx.state["lokr_full_factor_stamp"] = (
                str(stamped).strip().lower() == "true"
            )
    shapes = ctx.state.setdefault("lokr_explicit_shapes", {}).setdefault(
        ctx.lora_name, {}
    )
    if key.endswith(".lokr_w1"):
        ctx.state["has_lokr"] = True
        ctx.state.setdefault("lokr_module_names", set()).add(ctx.lora_name)
        if value.dim() == 2:
            shapes["w1"] = (int(value.size(0)), int(value.size(1)))
            spec = match_fused_spec(ctx.lora_name)
            n_comp = (
                len(spec.component_letters)
                if spec is not None and value.size(0) % len(spec.component_letters) == 0
                else 1
            )
            sub_out_a = int(value.size(0)) // n_comp
            ctx.state.setdefault("lokr_factors", set()).add(sub_out_a)
            if sub_out_a != int(value.size(1)):
                ctx.state["lokr_non_square_w1"] = True
        ctx.modules_dim[ctx.lora_name] = network_dim or (
            value.size(0) if value.dim() >= 1 else 1
        )
        ctx.modules_alpha.setdefault(
            ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name]))
        )
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    if key.endswith(".lokr_w2"):
        ctx.state["has_lokr"] = True
        ctx.state["lokr_has_full_w2"] = True
        ctx.state.setdefault("lokr_module_names", set()).add(ctx.lora_name)
        if value.dim() == 2:
            shapes["w2"] = (int(value.size(0)), int(value.size(1)))
        ctx.modules_dim.setdefault(
            ctx.lora_name, ctx.state.get("lokr_network_dim_meta") or 1
        )
        ctx.modules_alpha.setdefault(
            ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name]))
        )
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    if key.endswith(".lokr_w2_a") or key.endswith(".lokr_w2_b"):
        ctx.state["has_lokr"] = True
        ctx.state["lokr_has_decomposed_w2"] = True
        ctx.state.setdefault("lokr_module_names", set()).add(ctx.lora_name)
        if value.dim() == 2:
            leaf = "w2_a" if key.endswith(".lokr_w2_a") else "w2_b"
            shapes[leaf] = (int(value.size(0)), int(value.size(1)))
        ctx.modules_dim.setdefault(
            ctx.lora_name, ctx.state.get("lokr_network_dim_meta") or 1
        )
        ctx.modules_alpha.setdefault(
            ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name]))
        )
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    return False


def _finish_weight_detection(
    state: dict[str, Any],
    modules_dim: dict[str, int],
    modules_alpha: dict[str, Any],
) -> dict[str, Any]:
    if not state.get("has_lokr"):
        return {}
    if state.get("lokr_network_dim_meta") is None:
        for lora_name in state.get("lokr_module_names", set()):
            alpha_value = modules_alpha.get(lora_name)
            if isinstance(alpha_value, torch.Tensor) and alpha_value.numel() == 1:
                modules_dim[lora_name] = max(
                    1,
                    int(float(alpha_value.detach().float().cpu().item())),
                )
    stamped_factor = state.get("lokr_factor_stamp")
    factor = (
        int(stamped_factor)
        if stamped_factor is not None
        else next(iter(sorted(state.get("lokr_factors", set()))), 8)
    )
    stamped_einsum = state.get("lokr_einsum_stamp")
    if stamped_einsum is not None:
        use_einsum = bool(stamped_einsum)
    else:
        use_einsum = (
            bool(state.get("lokr_has_decomposed_w2"))
            or bool(state.get("lokr_non_square_w1"))
            or not bool(state.get("lokr_has_full_w2"))
        )
    plugin_args: dict[str, Any] = {
        "lokr_factor": factor,
        "lokr_use_einsum": use_einsum,
    }
    explicit_shapes = state.get("lokr_explicit_shapes")
    if explicit_shapes:
        plugin_args["_lokr_explicit_shapes"] = dict(explicit_shapes)
    if state.get("lokr_has_decomposed_w2"):
        plugin_args["lokr_decompose_w2"] = True
    stamped_full_factor = state.get("lokr_full_factor_stamp")
    if stamped_full_factor is not None:
        plugin_args["lokr_full_factor"] = bool(stamped_full_factor)
        if stamped_full_factor and state.get("lokr_has_decomposed_w2"):
            raise RuntimeError(
                "LoKR checkpoint is stamped ss_lokr_full_factor=true but "
                "contains decomposed factor keys."
            )
    else:
        # Legacy full-factor checkpoints have no stamp. Full w2 factors and no
        # decomposed keys are sufficient to reconstruct the intended layout.
        if state.get("lokr_has_full_w2") and not state.get("lokr_has_decomposed_w2"):
            plugin_args["lokr_full_factor"] = True
    return {
        "detected_spec": "lokr",
        "plugin_args": plugin_args,
    }


def _continue_weight_kind(ctx: ContinueWeightDetectionContext) -> str | None:
    if any("lokr_w1" in key or "lokr_w2" in key for key in ctx.lowered_keys):
        return "LoKr"
    if str(ctx.metadata.get("ss_network_spec") or "").strip().lower() == "lokr":
        return "LoKr"
    return None


def _same_tensor(tensors: list[torch.Tensor]) -> bool:
    first = tensors[0]
    return all(
        t.shape == first.shape and torch.equal(t, first.to(device=t.device, dtype=t.dtype))
        for t in tensors[1:]
    )


def _same_alpha(values: list[Any | None]) -> bool:
    if all(v is None for v in values):
        return True
    if any(v is None for v in values):
        return False
    def _as_float(v: Any) -> float:
        return float(v.detach().float().cpu().item()) if isinstance(v, torch.Tensor) else float(v)
    first = _as_float(values[0])
    return all(_as_float(v) == first for v in values[1:])


def _preprocess_weights(weights_sd: dict[str, Any]) -> dict[str, Any]:
    """Fuse split q/k/v LoKR keys back to Anima's runtime qkv/kv modules."""

    for shared_prefix, spec in iter_split_groups(weights_sd, ".lokr_w1"):
        suffixes = spec.component_letters
        w1s: list[torch.Tensor] = []
        w2s: list[torch.Tensor] = []
        w2as: list[torch.Tensor] = []
        w2bs: list[torch.Tensor] = []
        alphas: list[Any | None] = []
        complete = True
        decomposed: bool | None = None

        for suffix in suffixes:
            prefix = f"{shared_prefix}{suffix}_proj"
            w1_key = f"{prefix}.lokr_w1"
            if w1_key not in weights_sd:
                complete = False
                break
            has_full = f"{prefix}.lokr_w2" in weights_sd
            has_decomp = f"{prefix}.lokr_w2_a" in weights_sd and f"{prefix}.lokr_w2_b" in weights_sd
            if has_full == has_decomp:
                complete = False
                break
            if decomposed is None:
                decomposed = has_decomp
            elif decomposed != has_decomp:
                complete = False
                break
            w1s.append(weights_sd[w1_key])
            if has_full:
                w2s.append(weights_sd[f"{prefix}.lokr_w2"])
            else:
                w2as.append(weights_sd[f"{prefix}.lokr_w2_a"])
                w2bs.append(weights_sd[f"{prefix}.lokr_w2_b"])
            alphas.append(weights_sd.get(f"{prefix}.alpha"))

        if not complete:
            continue
        if not _same_alpha(alphas):
            raise RuntimeError(
                f"Split LoKR checkpoint has inconsistent alpha values at {shared_prefix}*."
            )

        fused_prefix = f"{shared_prefix}{spec.fused_letters}_proj"
        if not decomposed:
            if _same_tensor(w2s):
                weights_sd[f"{fused_prefix}.lokr_w1"] = torch.cat(w1s, dim=0).contiguous()
                weights_sd[f"{fused_prefix}.lokr_w2"] = w2s[0].contiguous()
            else:
                raise RuntimeError(
                    f"Split LoKR checkpoint has q/k/v-specific inner lokr_w2 factors at "
                    f"{shared_prefix}*; Anima's fused {spec.fused_letters}_proj requires "
                    "shared lokr_w2 across split components."
                )
        else:
            if _same_tensor(w2as) and _same_tensor(w2bs):
                weights_sd[f"{fused_prefix}.lokr_w1"] = torch.cat(w1s, dim=0).contiguous()
                weights_sd[f"{fused_prefix}.lokr_w2_a"] = w2as[0].contiguous()
                weights_sd[f"{fused_prefix}.lokr_w2_b"] = w2bs[0].contiguous()
            else:
                raise RuntimeError(
                    f"Split LoKR checkpoint has q/k/v-specific inner lokr_w2_a/lokr_w2_b "
                    f"factors at {shared_prefix}*; Anima's fused {spec.fused_letters}_proj "
                    "requires shared decomposed inner factors."
                )

        if alphas[0] is not None:
            weights_sd[f"{fused_prefix}.alpha"] = alphas[0]
        for suffix in suffixes:
            prefix = f"{shared_prefix}{suffix}_proj"
            for leaf in ("lokr_w1", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "alpha"):
                weights_sd.pop(f"{prefix}.{leaf}", None)

    return weights_sd


register_save_handler("lokr", save_lokr_weights)
register_network_spec(
    NetworkSpec(
        name="lokr",
        module_class=LoKrModule,
        save_variant="lokr",
        kwarg_flags=(
            "use_lokr",
            "lokr_factor",
            "lokr_factor_group_size",
            "lokr_project_chunk_bytes",
            "lokr_grouped_delta_backend",
            "lokr_grouped_delta_backward_backend",
            "lokr_use_einsum",
            "lokr_decompose_w2",
            "lokr_full_factor",
            # Escape hatch for resuming historical states that used network_dim
            # as a full-factor sentinel and must preserve their old alpha/dim scale.
            "lokr_allow_legacy_dim",
        ),
        selector=_selector,
        validate=_validate,
        module_kwargs=_module_kwargs,
        preprocess_weights=_preprocess_weights,
        detect_from_weights=_detect_from_weights,
        finish_weight_detection=_finish_weight_detection,
        continue_weight_kind=_continue_weight_kind,
    )
)

__all__ = ["LoKrModule"]
