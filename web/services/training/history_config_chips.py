"""Derive history-list chip fields from config.snapshot.toml.

Aligned with the Next React history overview chips in
web/frontend-next/src/features/training-history: training variant family,
preprocess precision, and block-swap transfer dtype.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from library.training.pp.errors import PipelineParallelError
from web.services.training.distributed import effective_distributed_mode
from web.services.training.extra_args import merge_training_extra_args

_KNOWN_VARIANTS = frozenset(
    {
        "lora",
        "lokr",
        "loha",
        "vera",
        "glora",
        "dora",
        "hydralora",
        "reft",
        "tlora",
        "ortholora",
        "chimera",
        "chimera_hydra",
        "soft_tokens",
        "ip_adapter",
        "easycontrol",
    }
)

_EMPTY = {
    "model_family": "",
    "training_variant": "",
    "preprocess_precision": "",
    "block_swap_precision": "",
    "base_compute": "",
    "precision_preference": "",
    "distributed_mode": "",
}


def history_config_chips_for_task_dir(
    task_dir: Path, *, variant: str = "", legacy_command: list[str] | None = None,
) -> dict[str, str]:
    snapshot = Path(task_dir) / "config.snapshot.toml"
    try:
        if not snapshot.is_file():
            return dict(_EMPTY)
        text = snapshot.read_text(encoding="utf-8")
    except OSError:
        return dict(_EMPTY)
    return history_config_chips_from_snapshot_text(text, variant=variant, legacy_command=legacy_command)


def history_config_chips_from_snapshot_text(
    text: str, *, variant: str = "", legacy_command: list[str] | None = None,
) -> dict[str, str]:
    raw = str(text or "")
    if not raw.strip():
        return dict(_EMPTY)
    try:
        data = tomllib.loads(raw)
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    flat = _flatten_toml_dict(data)
    return {
        "model_family": _infer_model_family(flat),
        "training_variant": _infer_training_variant(flat, raw, variant=variant),
        "preprocess_precision": _norm_precision(flat.get("preprocess_precision_preference")),
        "block_swap_precision": _norm_precision(flat.get("block_swap_transfer_dtype")),
        "base_compute": _norm_precision(flat.get("base_compute")),
        "precision_preference": _infer_precision_preference(flat),
        "distributed_mode": _legacy_pipeline_mode(data, legacy_command) if legacy_command is not None else "",
    }


def _legacy_pipeline_mode(config: dict[str, Any], command: list[str]) -> str:
    """Recover PP from a historical config and any recorded Web launcher overrides."""

    extra_args = []
    for option in ("--config_file", "--methods_subdir"):
        if option in command:
            extra_args = command[command.index(option) + 2:]
            break
    try:
        mode = effective_distributed_mode(merge_training_extra_args(config, extra_args))
    except (ValueError, PipelineParallelError):
        return ""
    return "pp" if mode == "pp" else ""


def _flatten_toml_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Root keys win; also merge one-level tables so sectioned snapshots still work."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for nested_key, nested_val in value.items():
                out.setdefault(str(nested_key), nested_val)
        else:
            out[str(key)] = value
    return out


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _norm_precision(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text


def _infer_model_family(flat: dict[str, Any]) -> str:
    """Resolve the same canonical family values used by the model config UI."""
    from library.models.family_registry import normalize_registered_family

    value = str(flat.get("model_family") or "anima").strip().lower()
    try:
        return normalize_registered_family(value)
    except ValueError:
        return ""


def _infer_precision_preference(flat: dict[str, Any]) -> str:
    """Match formatHistoryPrecisionPreference / precisionPreferenceFromConfig.

    Snapshot usually drops the virtual ``precision_preference`` key and only
    keeps ``mixed_precision`` (+ optional full_fp16/full_bf16).
    """
    explicit = _norm_precision(flat.get("precision_preference"))
    if explicit in {"bf16", "fp16", "fp32"}:
        return explicit
    mixed = _norm_precision(flat.get("mixed_precision"))
    if mixed == "no":
        return "fp32"
    if mixed == "fp16" or _truthy(flat.get("full_fp16")):
        return "fp16"
    if mixed == "bf16" or _truthy(flat.get("full_bf16")):
        return "bf16"
    if not mixed and not explicit:
        return ""
    return "bf16"


def _infer_training_variant(flat: dict[str, Any], raw_text: str, *, variant: str) -> str:
    """Match the Next history overview's training-variant inference order."""
    module_name = str(flat.get("network_module") or "").lower()
    moe_style = str(flat.get("use_moe_style") or "").strip().lower()

    if _truthy(flat.get("use_chimera_hydra")) or "chimera" in module_name:
        return "chimera"
    if _truthy(flat.get("use_ip_adapter")) or "ip_adapter" in module_name:
        return "ip_adapter"
    if _truthy(flat.get("use_easycontrol")) or "easycontrol" in module_name:
        return "easycontrol"
    if "soft_tokens" in module_name:
        return "soft_tokens"
    if _truthy(flat.get("use_loha")):
        return "loha"
    if _truthy(flat.get("use_lokr")):
        return "lokr"
    if _truthy(flat.get("use_vera")):
        return "vera"
    if _truthy(flat.get("use_glora")):
        return "glora"
    if _truthy(flat.get("dora_wd")) or _truthy(flat.get("use_dora")):
        return "dora"
    if _truthy(flat.get("add_reft")):
        return "reft"
    if moe_style and moe_style not in {"", "false", "none", "0", "off"}:
        return "hydralora"
    if _truthy(flat.get("use_timestep_mask")):
        return "tlora"
    if _truthy(flat.get("use_ortho")):
        return "ortholora"
    if "lora_anima" in module_name:
        return "lora"

    # Frontend: hasSnapshot && !moduleName → lora
    # Avoid misclassifying fixtures that only have output_dir/output_name.
    has_snapshot = bool(raw_text.strip()) and not (
        raw_text.lstrip().startswith("#")
        and ("无配置快照" in raw_text or "无法生成配置快照" in raw_text)
    )
    if has_snapshot and not module_name:
        meaningful = {
            k
            for k in flat
            if k not in {"output_dir", "output_name"} and flat.get(k) not in (None, "")
        }
        if meaningful:
            return "lora"

    variant_key = str(variant or "").strip().lower()
    if variant_key in _KNOWN_VARIANTS:
        return "chimera" if variant_key == "chimera_hydra" else variant_key
    if variant_key.endswith("-8gb"):
        compact = variant_key[: -len("-8gb")]
        if compact in _KNOWN_VARIANTS:
            return "chimera" if compact == "chimera_hydra" else compact
    return ""
