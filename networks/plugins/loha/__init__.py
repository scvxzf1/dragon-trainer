"""Bundled LoHa plugin registration."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .module import LoHaModule
from .save import save_loha_weights
from ...attn_fuse import iter_split_groups
from ...registry_api import (
    ContinueWeightDetectionContext,
    NetworkSpec,
    WeightDetectionContext,
    register_network_spec,
    register_save_handler,
)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _selector(kwargs: Mapping[str, Any]) -> bool:
    return _truthy(kwargs.get("use_loha"))


def _validate(kwargs: Mapping[str, Any]) -> None:
    if not _selector(kwargs):
        return
    if _truthy(kwargs.get("use_lokr")):
        raise ValueError("use_loha is mutually exclusive with use_lokr.")
    if _truthy(kwargs.get("use_vera")):
        raise ValueError("use_loha is mutually exclusive with use_vera.")
    if _truthy(kwargs.get("dora_wd")):
        raise ValueError("use_loha is mutually exclusive with dora_wd.")
    if (
        _truthy(kwargs.get("use_ortho"))
        or _truthy(kwargs.get("use_chimera_hydra"))
        or kwargs.get("use_moe_style") not in (None, False, "", "false", "False")
    ):
        raise ValueError(
            "use_loha is mutually exclusive with use_ortho, "
            "use_moe_style, and use_chimera_hydra."
        )


def _detect_from_weights(ctx: WeightDetectionContext) -> bool:
    key = ctx.key
    value = ctx.value

    suffixes = (".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b")
    if not key.endswith(suffixes):
        return False

    ctx.state["has_loha"] = True
    ctx.state.setdefault("loha_module_names", set()).add(ctx.lora_name)
    if value.dim() == 2:
        if key.endswith((".hada_w1_a", ".hada_w2_a")):
            rank = int(value.size(1))
        else:
            rank = int(value.size(0))
    else:
        rank = 1
    ctx.modules_dim[ctx.lora_name] = rank
    ctx.modules_alpha.setdefault(
        ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name]))
    )
    ctx.plain_module_names.add(ctx.lora_name)
    return True


def _finish_weight_detection(
    state: dict[str, Any],
    modules_dim: dict[str, int],
    modules_alpha: dict[str, Any],
) -> dict[str, Any]:
    del modules_dim, modules_alpha
    if not state.get("has_loha"):
        return {}
    return {"detected_spec": "loha"}


def _continue_weight_kind(ctx: ContinueWeightDetectionContext) -> str | None:
    if any("hada_w1_" in key or "hada_w2_" in key for key in ctx.lowered_keys):
        return "LoHa"
    if str(ctx.metadata.get("ss_network_spec") or "").strip().lower() == "loha":
        return "LoHa"
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
    """Fuse split q/k/v LoHa keys back to Anima's runtime qkv/kv modules."""

    for shared_prefix, spec in iter_split_groups(weights_sd, ".hada_w1_a"):
        suffixes = spec.component_letters
        complete = True
        w1_as: list[torch.Tensor] = []
        w1_bs: list[torch.Tensor] = []
        w2_as: list[torch.Tensor] = []
        w2_bs: list[torch.Tensor] = []
        alphas: list[Any | None] = []
        for suffix in suffixes:
            prefix = f"{shared_prefix}{suffix}_proj"
            keys = (
                f"{prefix}.hada_w1_a",
                f"{prefix}.hada_w1_b",
                f"{prefix}.hada_w2_a",
                f"{prefix}.hada_w2_b",
            )
            if any(k not in weights_sd for k in keys):
                complete = False
                break
            w1_as.append(weights_sd[keys[0]])
            w1_bs.append(weights_sd[keys[1]])
            w2_as.append(weights_sd[keys[2]])
            w2_bs.append(weights_sd[keys[3]])
            alphas.append(weights_sd.get(f"{prefix}.alpha"))
        if not complete:
            continue

        if not (_same_tensor(w1_bs) and _same_tensor(w2_bs)):
            raise RuntimeError(
                f"Split LoHa checkpoint has q/k/v-specific input factors at {shared_prefix}*; "
                f"Anima's fused {spec.fused_letters}_proj requires shared hada_w1_b/hada_w2_b."
            )
        if not _same_alpha(alphas):
            raise RuntimeError(
                f"Split LoHa checkpoint has inconsistent alpha values at {shared_prefix}*."
            )

        fused_prefix = f"{shared_prefix}{spec.fused_letters}_proj"
        weights_sd[f"{fused_prefix}.hada_w1_a"] = torch.cat(w1_as, dim=0).contiguous()
        weights_sd[f"{fused_prefix}.hada_w1_b"] = w1_bs[0].contiguous()
        weights_sd[f"{fused_prefix}.hada_w2_a"] = torch.cat(w2_as, dim=0).contiguous()
        weights_sd[f"{fused_prefix}.hada_w2_b"] = w2_bs[0].contiguous()
        if alphas[0] is not None:
            weights_sd[f"{fused_prefix}.alpha"] = alphas[0]

        for suffix in suffixes:
            prefix = f"{shared_prefix}{suffix}_proj"
            for leaf in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "alpha"):
                weights_sd.pop(f"{prefix}.{leaf}", None)

    return weights_sd


register_save_handler("loha", save_loha_weights)
register_network_spec(
    NetworkSpec(
        name="loha",
        module_class=LoHaModule,
        save_variant="loha",
        kwarg_flags=("use_loha",),
        selector=_selector,
        validate=_validate,
        preprocess_weights=_preprocess_weights,
        detect_from_weights=_detect_from_weights,
        finish_weight_detection=_finish_weight_detection,
        continue_weight_kind=_continue_weight_kind,
    )
)

__all__ = ["LoHaModule"]
