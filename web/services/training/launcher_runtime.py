"""Shared facade accessors for training launcher modules."""

from __future__ import annotations

from typing import Any

from library.runtime.launch import (
    ACCELERATE_MIXED_PRECISION_ENV,
    resolve_accelerate_mixed_precision,
)
from web.services.training.launch_support import (
    _command_option_value,
)


def _training_facade():
    from web.services import training_service as facade
    return facade


def _root():
    return _training_facade().ROOT


def _prepare_web_runtime_config(*args, **kwargs):
    return _training_facade()._prepare_web_runtime_config(*args, **kwargs)


def _clone_frozen_runtime_config(*args, **kwargs):
    return _training_facade()._clone_frozen_runtime_config(*args, **kwargs)


def _runtime_meta(*args, **kwargs):
    return _training_facade()._runtime_meta(*args, **kwargs)


def _display_project_path(*args, **kwargs):
    return _training_facade()._display_project_path(*args, **kwargs)


def _load_config_file_config(*args, **kwargs):
    return _training_facade()._load_config_file_config(*args, **kwargs)


def _normalize_gpu_whitelist(*args, **kwargs):
    return _training_facade()._normalize_gpu_whitelist(*args, **kwargs)


def _apply_gpu_whitelist(*args, **kwargs):
    return _training_facade()._apply_gpu_whitelist(*args, **kwargs)


def _apply_runtime_env(*args, **kwargs):
    return _training_facade()._apply_runtime_env(*args, **kwargs)


def _resolve_training_runtime_info(*args, **kwargs):
    return _training_facade()._resolve_training_runtime_info(*args, **kwargs)


def _runtime_from_config_file(*args, **kwargs):
    return _training_facade()._runtime_from_config_file(*args, **kwargs)


def _sample_config_from_cfg(*args, **kwargs):
    return _training_facade()._sample_config_from_cfg(*args, **kwargs)


def _resolve_display_path(*args, **kwargs):
    return _training_facade()._resolve_display_path(*args, **kwargs)


def _training_config_argument(config_file: str) -> str:
    """Resolve Web display paths before passing them to the standalone CLI."""
    path = _resolve_display_path(config_file)
    if path is None:
        raise ValueError(f"训练配置路径无效: {config_file}")
    return str(path.resolve())


def _ensure_training_data_dirs(*args, **kwargs):
    return _training_facade()._ensure_training_data_dirs(*args, **kwargs)


def preflight_training_config(*args, **kwargs):
    return _training_facade().preflight_training_config(*args, **kwargs)


def _accelerate_mixed_precision_for_training(
    config_file: str | None,
    extra_args: list[str] | None = None,
) -> str | None:
    from web.services.training.extra_args import merge_training_extra_args

    cfg = _load_config_file_config(config_file) if config_file else {}
    raw = merge_training_extra_args(cfg, extra_args).get("mixed_precision")
    value = str(raw).strip() if raw is not None else None
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return resolve_accelerate_mixed_precision({
        ACCELERATE_MIXED_PRECISION_ENV: normalized,
    })


