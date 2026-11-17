"""Global Web UI settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import toml

from library.env import (
    anima_home,
    get_configs_root,
    get_training_history_root,
    get_training_queue_root,
    get_webui_settings_file,
    normalize_model_family,
)
from web.services._dynamic_path import DynamicPath
from web.services.atomic_io import atomic_write_text

ROOT = anima_home()


def _default_settings_file() -> Path:
    return get_webui_settings_file()


SETTINGS_FILE = DynamicPath(_default_settings_file)
CONFIGS_DIR = DynamicPath(lambda: SETTINGS_FILE.parent)

DEFAULT_OUTPUT_ROOT = "output/runs"
DEFAULT_UI_SCALE = 100
DEFAULT_UI_SCALE_OVERRIDE = ""
DEFAULT_TAGGING_MAX_RETAINED_JOBS = 40
GLOBAL_MODEL_PATH_KEYS = (
    "pretrained_model_name_or_path",
    "qwen3",
    "vae",
)
GLOBAL_CONFIG_PATH_KEYS = (
    "configs_root",
    "history_root",
    "queue_root",
)
GLOBAL_UI_OVERRIDE_KEYS = (
    "ui_scale_config",
    "ui_scale_datasets",
    "ui_scale_training",
    "ui_scale_weight_analysis",
    "ui_scale_image_test",
    "ui_scale_settings",
    "ui_scale_model_config",
    "ui_scale_environment",
    "ui_scale_history_overview",
    "ui_scale_history_analysis",
    "ui_scale_history_preview",
    "ui_scale_history_logs",
    "ui_scale_history_config_files",
)
GLOBAL_UI_KEYS = (
    "ui_scale",
    *GLOBAL_UI_OVERRIDE_KEYS,
)
GLOBAL_DRAGON_UI_KEYS = (
    "dragon_motion_enabled",
    "dragon_config_help_always_visible",
    "dragon_config_tags_always_visible",
)
GLOBAL_IMAGE_TEST_KEYS = (
    "image_test_allow_home_search",
)
GLOBAL_IMAGE_TEST_PATH_KEYS = (
    "image_test_save_root",
)
GLOBAL_TAGGING_KEYS = (
    "tagging_max_retained_jobs",
)
# Model family selector (Krea-2-Raw migration, stage 6). Whose value picks the
# base model the LoRA is trained against. Empty → anima default (resolve_model_family
# falls back to env / DEFAULT_MODEL_FAMILY), so anima-only deployments leave it
# unset and nothing changes. ``model_family`` is a single enum string, not a
# path, so it has its own whitelist tuple rather than GLOBAL_MODEL_PATH_KEYS.
GLOBAL_FAMILY_KEYS = (
    "model_family",
)
def get_training_policy() -> dict[str, Any]:
    """Return durable training/queue policy defaults from web-ui-settings."""
    raw = _load_raw_settings()
    section = raw.get("training_policy") if isinstance(raw.get("training_policy"), dict) else {}
    defaults = _default_training_policy()
    out = dict(defaults)
    for key, value in section.items():
        if key in defaults:
            out[key] = value
    return {
        "ok": True,
        **_normalize_training_policy(out),
        "defaults": defaults,
    }


def save_training_policy(data: dict[str, Any]) -> dict[str, Any]:
    """Persist training/queue policy defaults (API-only, no frontend required)."""
    settings_file = Path(SETTINGS_FILE)
    raw = _load_raw_settings(settings_file)
    current = raw.get("training_policy") if isinstance(raw.get("training_policy"), dict) else {}
    merged = {**_default_training_policy(), **current, **(data or {})}
    normalized = _normalize_training_policy(merged)
    raw["training_policy"] = {
        key: normalized[key]
        for key in _default_training_policy()
    }
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(settings_file, toml.dumps(raw))
    return {"ok": True, **normalized}


def _default_training_policy() -> dict[str, Any]:
    return {
        "auto_retry": False,
        "max_attempts": 1,
        "retry_backoff_sec": 0.0,
        "max_queue_items": 200,
        "max_history_items": 100,
        "system_monitor_interval_sec": 2.0,
        "progress_poll_interval_sec": 1.0,
        "stop_grace_sec": 3.0,
    }


def _normalize_training_policy(data: dict[str, Any]) -> dict[str, Any]:
    defaults = _default_training_policy()
    auto_retry = data.get("auto_retry", defaults["auto_retry"])
    if isinstance(auto_retry, str):
        auto_retry = auto_retry.strip().lower() in {"1", "true", "yes", "on"}
    else:
        auto_retry = bool(auto_retry)

    def _clamp_int(value: Any, default: int, lo: int, hi: int) -> int:
        try:
            n = int(value)
        except (TypeError, ValueError):
            n = default
        return max(lo, min(hi, n))

    def _clamp_float(value: Any, default: float, lo: float, hi: float) -> float:
        try:
            n = float(value)
        except (TypeError, ValueError):
            n = default
        return max(lo, min(hi, n))

    return {
        "auto_retry": auto_retry,
        "max_attempts": _clamp_int(data.get("max_attempts"), defaults["max_attempts"], 1, 10),
        "retry_backoff_sec": _clamp_float(data.get("retry_backoff_sec"), defaults["retry_backoff_sec"], 0.0, 3600.0),
        "max_queue_items": _clamp_int(data.get("max_queue_items"), defaults["max_queue_items"], 10, 2000),
        "max_history_items": _clamp_int(data.get("max_history_items"), defaults["max_history_items"], 10, 5000),
        "system_monitor_interval_sec": _clamp_float(
            data.get("system_monitor_interval_sec"), defaults["system_monitor_interval_sec"], 0.5, 30.0
        ),
        "progress_poll_interval_sec": _clamp_float(
            data.get("progress_poll_interval_sec"), defaults["progress_poll_interval_sec"], 0.2, 10.0
        ),
        "stop_grace_sec": _clamp_float(data.get("stop_grace_sec"), defaults["stop_grace_sec"], 0.5, 60.0),
    }


class DistributedSettingsError(ValueError):
    """Invalid [distributed] settings payload (stable 400 for the API)."""


DISTRIBUTED_MODES = ("single", "ddp", "pp")
DISTRIBUTED_SCHEDULES = ("1f1b",)
DISTRIBUTED_SPLITS = ("balanced",)


def _default_distributed_settings() -> dict[str, Any]:
    return {
        "mode": "ddp",
        "gpu_whitelist": [],
        "pipeline_parallel_stages": 2,
        "pipeline_parallel_microbatches": 4,
        "pipeline_parallel_schedule": "1f1b",
        "pipeline_parallel_split": "balanced",
        "pipeline_parallel_blocks_to_swap": 0,
    }


def _normalize_distributed_gpu_whitelist(value: Any) -> list[int]:
    from web.services.training.gpu import normalize_gpu_whitelist

    return normalize_gpu_whitelist(value)


def _normalize_distributed_settings(data: dict[str, Any]) -> dict[str, Any]:
    from web.services.training.distributed import normalize_distributed_mode

    defaults = _default_distributed_settings()
    mode = normalize_distributed_mode(data.get("mode"), default=defaults["mode"])
    raw_mode = str(data.get("mode") or "").strip().lower()
    if raw_mode and raw_mode not in DISTRIBUTED_MODES:
        raise DistributedSettingsError(
            f"distributed.mode 必须是 {'/'.join(DISTRIBUTED_MODES)} 之一，收到 {raw_mode!r}"
        )
    schedule = str(data.get("pipeline_parallel_schedule") or defaults["pipeline_parallel_schedule"]).strip().lower()
    if schedule not in DISTRIBUTED_SCHEDULES:
        raise DistributedSettingsError(
            f"distributed.pipeline_parallel_schedule 必须是 {'/'.join(DISTRIBUTED_SCHEDULES)} 之一"
        )
    split = str(data.get("pipeline_parallel_split") or defaults["pipeline_parallel_split"]).strip().lower()
    if split not in DISTRIBUTED_SPLITS:
        raise DistributedSettingsError(
            f"distributed.pipeline_parallel_split 必须是 {'/'.join(DISTRIBUTED_SPLITS)} 之一"
        )

    def _non_negative_int(value: Any, default: int, field: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise DistributedSettingsError(
                f"distributed.{field} 必须是非负整数"
            ) from None
        if parsed < 0:
            raise DistributedSettingsError(f"distributed.{field} 必须是非负整数")
        return parsed

    def _positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise DistributedSettingsError(
                "distributed.pipeline_parallel_stages / pipeline_parallel_microbatches 必须是正整数"
            ) from None
        if parsed < 1:
            raise DistributedSettingsError(
                "distributed.pipeline_parallel_stages / pipeline_parallel_microbatches 必须是正整数"
            )
        return parsed

    return {
        "mode": mode,
        "gpu_whitelist": _normalize_distributed_gpu_whitelist(
            data.get("gpu_whitelist", defaults["gpu_whitelist"])
        ),
        "pipeline_parallel_stages": _positive_int(
            data.get("pipeline_parallel_stages", defaults["pipeline_parallel_stages"]),
            defaults["pipeline_parallel_stages"],
        ),
        "pipeline_parallel_microbatches": _positive_int(
            data.get("pipeline_parallel_microbatches", defaults["pipeline_parallel_microbatches"]),
            defaults["pipeline_parallel_microbatches"],
        ),
        "pipeline_parallel_schedule": schedule,
        "pipeline_parallel_split": split,
        # 0 is legal (disabled), so this is non-negative rather than positive.
        "pipeline_parallel_blocks_to_swap": _non_negative_int(
            data.get(
                "pipeline_parallel_blocks_to_swap",
                defaults["pipeline_parallel_blocks_to_swap"],
            ),
            defaults["pipeline_parallel_blocks_to_swap"],
            "pipeline_parallel_blocks_to_swap",
        ),
    }


def get_distributed_settings() -> dict[str, Any]:
    """Read the [distributed] section of the local settings file.

    缺失节/缺字段按默认值回填（mode 缺省 DDP，旧设置零迁移兼容）。
    """

    raw = _load_raw_settings()
    section = raw.get("distributed") if isinstance(raw.get("distributed"), dict) else {}
    defaults = _default_distributed_settings()
    merged = {**defaults, **(section if isinstance(section, dict) else {})}
    try:
        normalized = _normalize_distributed_settings(merged)
    except DistributedSettingsError:
        normalized = defaults
    return {"ok": True, **normalized, "defaults": defaults}


def save_distributed_settings(data: dict[str, Any]) -> dict[str, Any]:
    """Persist the [distributed] section via the existing settings channel."""

    settings_file = Path(SETTINGS_FILE)
    normalized = _normalize_distributed_settings({**_default_distributed_settings(), **(data or {})})
    raw = _load_raw_settings(settings_file)
    raw["distributed"] = normalized
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(settings_file, toml.dumps(raw))
    return {"ok": True, **normalized}


def _normalize_bool_setting(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_model_family(value: Any) -> str:
    """Normalize the selector; only a missing value may use the default chain.

    Stored as a lowercase family string. Empty string means "use the
    resolve_model_family() fallback (env / DEFAULT_MODEL_FAMILY=anima)" so
    anima-only deployments leave the field unset and behavior is unchanged.
    """
    if not str(value or "").strip():
        return ""
    return normalize_model_family(value, source="WebUI global model_family")

def get_global_settings() -> dict[str, Any]:
    settings = _load_settings()
    defaults = _default_global_settings()
    path_overrides = _load_path_overrides()
    return {
        "ok": True,
        "revision": _settings_revision(),
        **settings,
        "defaults": defaults,
        "path_overrides": path_overrides,
        "effective_paths": {
            key: str(settings.get(key) or "")
            for key in GLOBAL_CONFIG_PATH_KEYS
        },
    }


class GlobalSettingsConflictError(ValueError):
    pass


def _settings_revision(path: Path | None = None) -> str:
    from web.services.config.revisioned_text import file_revision

    return file_revision(path or Path(SETTINGS_FILE))


def save_global_settings(data: dict[str, Any]) -> dict[str, Any]:
    from web.services.config.revisioned_text import locked_text_file

    current_settings_file = Path(SETTINGS_FILE)
    with locked_text_file(current_settings_file):
        revision = data.get("revision")
        if revision is not None and revision != _settings_revision(current_settings_file):
            raise GlobalSettingsConflictError("全局设置已在其他位置修改，请重新读取后再保存")
        return _save_global_settings_locked(data, current_settings_file)


def _save_global_settings_locked(data: dict[str, Any], current_settings_file: Path) -> dict[str, Any]:
    current = _load_settings(current_settings_file)
    output_root = _normalize_output_root(
        str(data.get("output_root", current["output_root"]) or DEFAULT_OUTPUT_ROOT),
        allow_empty=False,
    )
    current_raw = _load_raw_settings(current_settings_file)
    current_section = current_raw.get("global") if isinstance(current_raw.get("global"), dict) else {}
    target_settings_file = _settings_file_for_payload(data, current_settings_file)
    target_raw = _load_raw_settings(target_settings_file)
    target_section = target_raw.get("global") if isinstance(target_raw.get("global"), dict) else {}
    defaults = _default_global_settings(settings_file=target_settings_file)
    next_global = {
        **target_section,
        **current_section,
        "output_root": output_root,
    }
    for key in GLOBAL_MODEL_PATH_KEYS:
        if key in data:
            value = _normalize_global_model_path(data.get(key))
            next_global[key] = value or current.get(key) or defaults.get(key, "")
        elif key not in next_global:
            next_global[key] = current.get(key, "") or defaults.get(key, "")
    for key in GLOBAL_CONFIG_PATH_KEYS:
        if key in data:
            value = _normalize_config_path(data.get(key))
            next_global[key] = value or defaults.get(key, "")
        elif key not in next_global:
            next_global[key] = current.get(key, "") or defaults.get(key, "")
    if "ui_scale" in data:
        value = _normalize_ui_setting("ui_scale", data.get("ui_scale"))
        next_global["ui_scale"] = value if value is not None else current.get("ui_scale", defaults.get("ui_scale"))
    elif "ui_scale" not in next_global:
        next_global["ui_scale"] = current.get("ui_scale", defaults.get("ui_scale"))
    for key in GLOBAL_UI_OVERRIDE_KEYS:
        if key not in data:
            continue
        value = _normalize_ui_setting(key, data.get(key))
        if value is None:
            next_global.pop(key, None)
        else:
            next_global[key] = value
    for key in GLOBAL_DRAGON_UI_KEYS:
        default = bool(defaults.get(key, False))
        if key in data:
            next_global[key] = _normalize_bool_setting(data.get(key), default=default)
        elif key not in next_global:
            next_global[key] = bool(current.get(key, default))
    for key in GLOBAL_IMAGE_TEST_KEYS:
        if key in data:
            next_global[key] = _normalize_bool_setting(data.get(key), default=False)
        elif key not in next_global:
            next_global[key] = bool(current.get(key, defaults.get(key, False)))
    for key in GLOBAL_IMAGE_TEST_PATH_KEYS:
        if key in data:
            next_global[key] = _normalize_image_test_save_root(data.get(key))
        elif key not in next_global:
            next_global[key] = str(current.get(key, defaults.get(key, "")) or "")
    for key in GLOBAL_TAGGING_KEYS:
        if key in data:
            next_global[key] = _normalize_tagging_job_retention(data.get(key))
        elif key not in next_global:
            next_global[key] = int(current.get(key, defaults.get(key, DEFAULT_TAGGING_MAX_RETAINED_JOBS)))
    # Model family (Krea-2-Raw migration). Stored as "" for anima (the default)
    # so the resolve_model_family() fallback chain (env / DEFAULT_MODEL_FAMILY)
    # still governs when the panel leaves it unset. A non-empty non-anima value
    # pins the family for new preset generation; explicit base.toml
    # ``model_family`` keys still win at training time
    # (bootstrap.build_net_kwargs re-resolves). ``anima`` is normalized to empty
    # on save too — writing an explicit ``model_family = "anima"`` would mask
    # the env-var override path, so the panel's Anima option emits ``""`` and we
    # defensively collapse "anima" to "" as well.
    if "model_family" in data:
        normalized_family = _normalize_model_family(data.get("model_family"))
        if normalized_family and normalized_family != "anima":
            next_global["model_family"] = normalized_family
        else:
            # Empty / anima → drop the key so anima-default holds.
            next_global.pop("model_family", None)
    elif "model_family" in next_global:
        existing = _normalize_model_family(next_global.get("model_family"))
        if existing and existing != "anima":
            next_global["model_family"] = existing
        else:
            next_global.pop("model_family", None)
    raw = {
        **current_raw,
        **target_raw,
    }
    raw["global"] = {
        **next_global,
    }
    target_settings_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target_settings_file, toml.dumps(raw))

    # 如果设置了路径覆盖，同时保存到项目根目录的专用配置文件
    path_overrides = {
        key: data[key]
        for key in ("configs_root", "history_root", "queue_root")
        if key in data
    }
    if path_overrides:
        _save_path_overrides(path_overrides)

    saved = _load_settings(target_settings_file)
    raw_path_overrides = _load_path_overrides()
    return {
        "ok": True,
        "revision": _settings_revision(target_settings_file),
        "message": "全局设置已保存",
        "requires_reload": current_settings_file.resolve() != target_settings_file.resolve(),
        **saved,
        "defaults": _default_global_settings(settings_file=target_settings_file),
        "path_overrides": raw_path_overrides,
        "effective_paths": {
            key: str(saved.get(key) or "")
            for key in GLOBAL_CONFIG_PATH_KEYS
        },
    }




def _normalize_image_test_save_root(value: Any) -> str:
    """Normalize image_test save root; empty means fallback to output/tests."""
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        return ""
    path = Path(clean)
    if ".." in path.parts:
        raise ValueError("image_test 保存目录不能包含 ..")
    if path.is_absolute():
        return path.resolve().as_posix()
    return path.as_posix().lstrip("/").rstrip("/")


def _normalize_tagging_job_retention(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = DEFAULT_TAGGING_MAX_RETAINED_JOBS
    return max(1, min(500, count))


def resolve_image_test_save_root(value: str | None = None) -> str:
    """Return effective image_test save dir display path (never empty)."""
    if value is None:
        raw = _load_settings().get("image_test_save_root", "")
    else:
        raw = value
    try:
        normalized = _normalize_image_test_save_root(raw)
    except ValueError:
        normalized = ""
    if normalized:
        return normalized
    # Compatible default used by image_test / preview inference dir.
    return "output/tests"

def resolve_output_root(value: str | None = None) -> Path:
    output_root = value if value is not None else _load_settings()["output_root"]
    return _resolve_output_root(output_root)


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_settings(settings_file: Path | None = None) -> dict[str, Any]:
    settings_file = settings_file or Path(SETTINGS_FILE)
    defaults = _default_global_settings(settings_file=settings_file)
    raw = _load_raw_settings(settings_file)
    section = raw.get("global", {}) if isinstance(raw, dict) else {}
    if not isinstance(section, dict):
        return defaults
    settings = {**defaults}
    try:
        settings["output_root"] = _normalize_output_root(
            str(section.get("output_root", defaults["output_root"]) or ""),
            allow_empty=False,
        )
    except ValueError:
        settings["output_root"] = defaults["output_root"]
    for key in GLOBAL_MODEL_PATH_KEYS:
        if key in section:
            settings[key] = _normalize_global_model_path(section.get(key)) or defaults.get(key, "")
    for key in GLOBAL_CONFIG_PATH_KEYS:
        if key in section:
            settings[key] = _normalize_config_path(section.get(key)) or defaults.get(key, "")
    for key in GLOBAL_UI_KEYS:
        if key in section:
            value = _normalize_ui_setting(key, section.get(key))
            settings[key] = value if value is not None else defaults.get(key)
    for key in GLOBAL_DRAGON_UI_KEYS:
        default = bool(defaults.get(key, False))
        if key in section:
            settings[key] = _normalize_bool_setting(section.get(key), default=default)
        else:
            settings[key] = default
    for key in GLOBAL_IMAGE_TEST_KEYS:
        if key in section:
            settings[key] = _normalize_bool_setting(section.get(key), default=bool(defaults.get(key, False)))
        else:
            settings[key] = bool(defaults.get(key, False))
    for key in GLOBAL_IMAGE_TEST_PATH_KEYS:
        if key in section:
            try:
                settings[key] = _normalize_image_test_save_root(section.get(key))
            except ValueError:
                settings[key] = str(defaults.get(key, "") or "")
        else:
            settings[key] = str(defaults.get(key, "") or "")
    for key in GLOBAL_TAGGING_KEYS:
        settings[key] = _normalize_tagging_job_retention(section.get(key, defaults.get(key)))
    # Model family: stored as "" (anima default) when unset so the
    # resolve_model_family() fallback chain still governs at training time.
    # An explicit on-disk ``model_family = "anima"`` also reads back as "" so
    # the env-var override path stays unmasked.
    normalized_family = _normalize_model_family(section.get("model_family"))
    settings["model_family"] = "" if normalized_family == "anima" else normalized_family

    # 显示当前实际使用的配置根目录（包括环境变量）
    actual_configs_root = settings_file.parent.resolve()
    try:
        settings["configs_root"] = actual_configs_root.relative_to(ROOT).as_posix()
    except ValueError:
        settings["configs_root"] = actual_configs_root.as_posix()
    try:
        history_root = get_training_history_root().resolve()
        settings["history_root"] = history_root.relative_to(ROOT).as_posix()
    except Exception:
        try:
            settings["history_root"] = get_training_history_root().as_posix()
        except Exception:
            settings["history_root"] = settings.get("history_root") or ""
    try:
        queue_root = get_training_queue_root().resolve()
        settings["queue_root"] = queue_root.relative_to(ROOT).as_posix()
    except Exception:
        try:
            settings["queue_root"] = get_training_queue_root().as_posix()
        except Exception:
            settings["queue_root"] = settings.get("queue_root") or ""

    return settings


def _load_raw_settings(settings_file: Path | None = None) -> dict[str, Any]:
    settings_file = settings_file or Path(SETTINGS_FILE)
    if not settings_file.exists():
        return {}
    try:
        raw = toml.loads(settings_file.read_text(encoding="utf-8"))
    except toml.TomlDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _default_global_settings(*, settings_file: Path | None = None) -> dict[str, Any]:
    return {
        "output_root": DEFAULT_OUTPUT_ROOT,
        "configs_root": "configs",
        "history_root": "",
        "queue_root": "",
        "ui_scale": DEFAULT_UI_SCALE,
        "dragon_motion_enabled": True,
        "dragon_config_help_always_visible": False,
        "dragon_config_tags_always_visible": False,
        "image_test_allow_home_search": False,
        "image_test_save_root": "",
        "tagging_max_retained_jobs": DEFAULT_TAGGING_MAX_RETAINED_JOBS,
        "model_family": "",
        **{key: DEFAULT_UI_SCALE_OVERRIDE for key in GLOBAL_UI_OVERRIDE_KEYS},
        **_load_base_model_path_defaults(settings_file=settings_file),
    }


def _load_base_model_path_defaults(*, settings_file: Path | None = None) -> dict[str, str]:
    defaults = {key: "" for key in GLOBAL_MODEL_PATH_KEYS}
    settings_file = settings_file or Path(SETTINGS_FILE)
    base_file = settings_file.parent / "base.toml"
    if not base_file.exists():
        return defaults
    try:
        raw = toml.loads(base_file.read_text(encoding="utf-8"))
    except toml.TomlDecodeError:
        return defaults
    if not isinstance(raw, dict):
        return defaults
    for key in GLOBAL_MODEL_PATH_KEYS:
        defaults[key] = _normalize_global_model_path(raw.get(key))
    return defaults


def _normalize_global_model_path(value: Any) -> str:
    # 模型路径保留用户写法：相对路径、绝对路径、环境变量字符串都原样进入配置模板。
    return str(value or "").strip()


def _normalize_output_root(value: str, *, allow_empty: bool) -> str:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        if allow_empty:
            return ""
        raise ValueError("输出文件夹不能为空")
    path = Path(clean)
    if ".." in path.parts:
        raise ValueError("输出文件夹不能包含 ..")
    if path.is_absolute():
        return path.resolve().as_posix()
    return path.as_posix().lstrip("/").rstrip("/") or DEFAULT_OUTPUT_ROOT


def _resolve_output_root(value: str) -> Path:
    normalized = _normalize_output_root(value, allow_empty=False)
    path = Path(normalized)
    if path.is_absolute():
        return path.resolve()
    return (ROOT / normalized).resolve()


def _normalize_ui_setting(key: str, value: Any) -> int | str | None:
    """Normalize UI settings."""
    if key == "ui_scale":
        return _normalize_required_ui_scale(value)
    if key in GLOBAL_UI_OVERRIDE_KEYS:
        return _normalize_optional_ui_scale(value)
    return None


def _normalize_required_ui_scale(value: Any) -> int:
    try:
        scale = int(value) if value is not None else DEFAULT_UI_SCALE
    except (ValueError, TypeError):
        return DEFAULT_UI_SCALE
    return _clamp_ui_scale(scale)


def _normalize_optional_ui_scale(value: Any) -> int | None:
    clean = str(value or "").strip()
    if not clean:
        return None
    try:
        scale = int(clean)
    except (ValueError, TypeError):
        return DEFAULT_UI_SCALE
    return _clamp_ui_scale(scale)


def _clamp_ui_scale(value: int) -> int:
    # 限制在 25% - 400% 之间
    if value < 25:
        return 25
    if value > 400:
        return 400
    return value


def _normalize_config_path(value: Any) -> str:
    """规范化配置路径（类似 _normalize_output_root）。"""
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        return ""
    path = Path(clean)
    if path.is_absolute():
        return path.resolve().as_posix()
    if ".." in path.parts:
        raise ValueError("配置路径不能包含 ..")
    return path.as_posix().lstrip("/").rstrip("/")


def resolve_config_path(key: str, value: str | None = None) -> Path:
    """解析配置路径（类似 resolve_output_root）。"""
    if value is None:
        value = _load_settings().get(key, "")
    normalized = _normalize_config_path(value)
    if not normalized:
        # 返回默认值
        defaults = _default_global_settings()
        normalized = defaults.get(key, "")
    path = Path(normalized)
    if path.is_absolute():
        return path.resolve()
    return (ROOT / normalized).resolve()


def _settings_file_for_payload(data: dict[str, Any], current_settings_file: Path) -> Path:
    # 显式设置文件覆盖（环境变量 / .anima-webui-settings.toml [paths].settings_file）
    # 优先于 configs_root 推导：否则用户改了 configs_root 会把设置写到别处，
    # 而读取端仍按覆盖路径读，导致「保存成功但设置不生效」。
    override = _explicit_settings_file_override()
    if override is not None:
        return override
    if "configs_root" not in data:
        return current_settings_file.resolve()
    return _settings_file_for_configs_root(data.get("configs_root"))


def _explicit_settings_file_override() -> Path | None:
    """返回显式设置文件覆盖路径；未配置时返回 None。"""
    configured = get_webui_settings_file()
    default_file = get_configs_root() / "web-ui-settings.toml"
    try:
        if configured.resolve() == default_file.resolve():
            return None
    except OSError:
        return None
    return configured


def _settings_file_for_configs_root(value: Any) -> Path:
    normalized = _normalize_config_path(value)
    if not normalized:
        normalized = "configs"
    path = Path(normalized)
    configs_dir = path.resolve() if path.is_absolute() else (ROOT / normalized).resolve()
    return configs_dir / "web-ui-settings.toml"


def _save_path_overrides(path_values: dict[str, Any]) -> None:
    """保存 configs/history/queue root 到项目根目录的专用配置文件。"""
    webui_paths_file = ROOT / ".anima-webui-settings.toml"

    raw: dict[str, Any] = {}
    if webui_paths_file.exists():
        try:
            raw = toml.loads(webui_paths_file.read_text(encoding="utf-8"))
        except toml.TomlDecodeError:
            raw = {}

    if "paths" not in raw or not isinstance(raw["paths"], dict):
        raw["paths"] = {}

    for key, value in path_values.items():
        if key not in {"configs_root", "history_root", "queue_root"}:
            continue
        normalized = _normalize_config_path(value)
        if normalized:
            raw["paths"][key] = normalized
        else:
            # 空值表示使用默认，从配置中删除
            raw["paths"].pop(key, None)

    atomic_write_text(webui_paths_file, toml.dumps(raw))


def _load_path_overrides() -> dict[str, str]:
    """Return raw path overrides separately from their resolved values."""
    webui_paths_file = ROOT / ".anima-webui-settings.toml"
    if not webui_paths_file.exists():
        return {key: "" for key in GLOBAL_CONFIG_PATH_KEYS}
    try:
        raw = toml.loads(webui_paths_file.read_text(encoding="utf-8"))
    except toml.TomlDecodeError:
        raw = {}
    paths = raw.get("paths") if isinstance(raw.get("paths"), dict) else {}
    overrides = {
        key: _normalize_config_path(paths.get(key)) if key in paths else ""
        for key in GLOBAL_CONFIG_PATH_KEYS
    }
    if not overrides["configs_root"]:
        configured = _normalize_config_path(os.environ.get("ANIMA_CONFIGS_ROOT"))
        overrides["configs_root"] = configured or "configs"
    return overrides


def _save_configs_root_override(configs_root: str) -> None:
    """兼容旧调用：只写 configs_root。"""
    _save_path_overrides({"configs_root": configs_root})
