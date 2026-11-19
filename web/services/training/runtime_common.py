"""Shared runtime configuration helpers for WebUI training runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import toml

from library.env import expand_env_vars_in_obj
from web.services import config_service

LOGGER = logging.getLogger(__name__)
from web.services.training.common import _positive_int_or_none
from web.services.training.runtime_paths import (
    _display_settings_path,
    _path_exists,
    _resolve_display_path,
)

RUNTIME_META_KEYS = (
    "run_dir",
    "runtime_config_file",
    "original_config_file",
    "dataset_config_file",
    "model_cache_dir",
    "dataset_cache_dir",
    "training_output_dir",
    "logs_dir",
    "history_source_config_file",
)


def load_merged_config(*args, **kwargs):
    return config_service.load_merged_config(*args, **kwargs)


def apply_auto_data_dirs(*args, **kwargs):
    return config_service.apply_auto_data_dirs(*args, **kwargs)


def apply_global_model_path_defaults(*args, **kwargs):
    return config_service.apply_global_model_path_defaults(*args, **kwargs)


def _dataset_rows_for_estimate(*args, **kwargs):
    return config_service._dataset_rows_for_estimate(*args, **kwargs)


def _build_dataset_config_doc(*args, **kwargs):
    return config_service._build_dataset_config_doc(*args, **kwargs)


def _normalize_path_pattern(*args, **kwargs):
    return config_service._normalize_path_pattern(*args, **kwargs)


def _normalize_trigger_clone(*args, **kwargs):
    return config_service._normalize_trigger_clone(*args, **kwargs)


def _normalize_nl_tag_mix(*args, **kwargs):
    return config_service._normalize_nl_tag_mix(*args, **kwargs)


def _nl_tag_mix_image_files(*args, **kwargs):
    return config_service._nl_tag_mix_image_files(*args, **kwargs)


def _nl_tag_mix_caption_source(*args, **kwargs):
    return config_service._nl_tag_mix_caption_source(*args, **kwargs)


def _classify_nl_tag_caption_text(*args, **kwargs):
    return config_service._classify_nl_tag_caption_text(*args, **kwargs)


def training_sample_sampler_status(*args, **kwargs):
    return config_service.training_sample_sampler_status(*args, **kwargs)


def _load_config_file_config(config_file: str) -> dict[str, Any]:
    return read_config_file_config(config_file) or {}


def read_config_file_config(config_file: str) -> dict[str, Any] | None:
    """Read a runtime config, distinguishing an empty TOML file from a missing one."""
    path = _resolve_display_path(config_file)
    if path is None or not _path_exists(path):
        return None
    try:
        parsed = toml.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("failed to parse training config file %s: %s", path, exc)
        return None
    expanded = expand_env_vars_in_obj(parsed)
    return expanded if isinstance(expanded, dict) else None


def toml_dumps_sorted(data: dict[str, Any]) -> str:
    try:
        return toml.dumps({key: data[key] for key in sorted(data)})
    except Exception:
        return json.dumps(data, ensure_ascii=False, indent=2)


def _default_preprocess_precision_preference(cfg: dict[str, Any]) -> str:
    raw = str(cfg.get("preprocess_precision_preference") or "").strip().lower()
    if raw in {"bf16", "fp16", "fp32"}:
        return raw
    mixed_precision = str(cfg.get("mixed_precision") or "").strip().lower()
    if mixed_precision == "fp16":
        return "fp16"
    if mixed_precision == "no":
        return "fp32"
    return "bf16"


def _runtime_dir_layout(run_dir: Path) -> dict[str, Path]:
    model_cache_dir = run_dir / "model_cache"
    training_output_dir = run_dir / "training_output"
    return {
        "model_cache_dir": model_cache_dir,
        "dataset_cache_dir": run_dir / "dataset_cache",
        "training_output_dir": training_output_dir,
        "sample_dir": training_output_dir / "sample",
        "logs_dir": model_cache_dir / "logs",
        "torchinductor_cache_dir": model_cache_dir / "torchinductor",
        "triton_cache_dir": model_cache_dir / "triton",
    }


def _ensure_runtime_dir_layout(run_dir: Path) -> dict[str, Path]:
    layout = _runtime_dir_layout(run_dir)
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _display_runtime_value(value: str | Path) -> str:
    if isinstance(value, Path):
        return _display_settings_path(value)
    return str(value or "")


def _build_runtime_payload(
    *,
    run_dir: Path,
    layout: dict[str, Path],
    runtime_config_file: str | Path,
    original_config_file: str | Path,
    dataset_config_file: str | Path,
    output_dir: str | Path,
    logs_dir: str | Path,
    history_source_config_file: str,
    data_dirs: dict[str, Any],
    dataset_dirs: list[dict[str, Any]] | None = None,
    sample_config: dict[str, Any] | None = None,
    resume_duration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "run_dir": _display_settings_path(run_dir),
        "runtime_config_file": _display_runtime_value(runtime_config_file),
        "original_config_file": _display_runtime_value(original_config_file),
        "dataset_config_file": _display_runtime_value(dataset_config_file),
        "output_dir": _display_runtime_value(output_dir),
        "sample_dir": _display_settings_path(layout["sample_dir"]),
        "model_cache_dir": _display_settings_path(layout["model_cache_dir"]),
        "dataset_cache_dir": _display_settings_path(layout["dataset_cache_dir"]),
        "training_output_dir": _display_runtime_value(output_dir),
        "logs_dir": _display_runtime_value(logs_dir),
        "torchinductor_cache_dir": _display_settings_path(layout["torchinductor_cache_dir"]),
        "triton_cache_dir": _display_settings_path(layout["triton_cache_dir"]),
        "history_source_config_file": history_source_config_file,
        "data_dirs": data_dirs,
    }
    if dataset_dirs is not None:
        payload["dataset_dirs"] = dataset_dirs
    if sample_config is not None:
        payload["sample_config"] = sample_config
    if resume_duration is not None:
        payload["resume_duration"] = resume_duration
    return payload


def _sample_config_from_cfg(cfg: dict[str, Any], extra_args: list[str]) -> dict[str, Any]:
    sample_prompts = cfg.get("sample_prompts")
    sample_every_n_epochs = cfg.get("sample_every_n_epochs")
    sample_every_n_steps = cfg.get("sample_every_n_steps")
    sample_at_first = bool(cfg.get("sample_at_first", False))
    sample_sampler_raw = str(cfg.get("sample_sampler") or "euler")

    overrides = _cli_arg_overrides(extra_args)
    if "sample_prompts" in overrides:
        sample_prompts = overrides["sample_prompts"]
    if "sample_every_n_epochs" in overrides:
        sample_every_n_epochs = overrides["sample_every_n_epochs"]
    if "sample_every_n_steps" in overrides:
        sample_every_n_steps = overrides["sample_every_n_steps"]
    if "sample_at_first" in overrides:
        sample_at_first = True
    if "sample_sampler" in overrides:
        sample_sampler_raw = str(overrides["sample_sampler"] or sample_sampler_raw)

    epoch_freq = _positive_int_or_none(sample_every_n_epochs)
    step_freq = _positive_int_or_none(sample_every_n_steps)
    sample_sampler, sample_sampler_status = training_sample_sampler_status(sample_sampler_raw)
    prompt_path = _resolve_display_path(str(sample_prompts or ""))
    prompt_exists = prompt_path.is_file() if prompt_path else False
    enabled = bool(prompt_path and prompt_exists and (epoch_freq is not None or step_freq is not None or sample_at_first))

    if not sample_prompts:
        message = "未设置 sample_prompts，训练不会生成样张"
    elif not prompt_exists:
        message = f"sample_prompts 文件不存在: {sample_prompts}"
    elif epoch_freq is None and step_freq is None and not sample_at_first:
        message = "未设置 sample_every_n_epochs 或 sample_every_n_steps，训练不会生成样张"
    else:
        message = "训练中采样已配置"

    return {
        "enabled": enabled,
        "sample_prompts": str(sample_prompts or ""),
        "sample_prompts_exists": prompt_exists,
        "sample_every_n_epochs": epoch_freq,
        "sample_every_n_steps": step_freq,
        "sample_at_first": sample_at_first,
        "sample_sampler": sample_sampler,
        "sample_sampler_raw": sample_sampler_raw,
        "sample_sampler_status": sample_sampler_status,
        "message": message,
    }


def _cli_arg_overrides(extra_args: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = {
        "--sample_prompts": "sample_prompts",
        "--sample_every_n_epochs": "sample_every_n_epochs",
        "--sample_every_n_steps": "sample_every_n_steps",
        "--sample_sampler": "sample_sampler",
    }
    for idx, arg in enumerate(extra_args):
        if arg == "--sample_at_first":
            out["sample_at_first"] = True
            continue
        if arg in keys and idx + 1 < len(extra_args):
            out[keys[arg]] = extra_args[idx + 1]
            continue
        for cli_key, config_key in keys.items():
            prefix = cli_key + "="
            if arg.startswith(prefix):
                out[config_key] = arg.split("=", 1)[1]
                break
    return out
