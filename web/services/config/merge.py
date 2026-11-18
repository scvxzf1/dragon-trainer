"""Configuration merge and data-directory helpers.

This module is loaded by ``web.services.config_service`` as part of the
compatibility facade.  It keeps facade access lazy so the module can also be
imported directly without pulling in the legacy facade.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import toml

from library.config import io as config_io
from library.env import expand_env_vars_in_obj, get_configs_root, load_dotenv
from web.services.config import common as _config_common
from web.services.config.metadata import (
    HIDDEN_CONFIG_FILES,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
IMPORTED_CONFIGS_DIR = CONFIGS_DIR / "imported"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
WEB_FILE_GROUPS_FILE = CONFIGS_DIR / "web-file-groups.toml"
WEB_USER_LOCKS_FILE = CONFIGS_DIR / "web-user-locks.toml"
DEFAULT_MAX_TRAIN_STEPS = 0
DATASET_PRESETS_DIR = CONFIGS_DIR / "datasets"
DATASET_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif", ".jxl"}

load_dotenv()

_SYNC_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "GUI_METHODS_DIR",
    "IMPORTED_CONFIGS_DIR",
    "PRESETS_FILE",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
    "DEFAULT_MAX_TRAIN_STEPS",
    "DATASET_PRESETS_DIR",
    "resolve_output_root",
    "_display_settings_path",
    "save_raw_file",
    "load_raw_file",
    "delete_raw_file",
    "patch_raw_file_values",
    "preview_raw_file_patch",
    "get_config_file_meta",
    "list_config_file_groups",
    "move_config_file_to_group",
    "_inspect_network_weight",
    "LOGGER",
)

def _sync_from_facade() -> None:
    from web.services import config_service as _facade

    _exported_names = set(globals().get("__all__", ()))
    for _name in _SYNC_NAMES:
        if not hasattr(_facade, _name):
            continue
        _value = getattr(_facade, _name)
        if _name not in _exported_names:
            globals()[_name] = _value


def _exported(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _sync_from_facade()
        return fn(*args, **kwargs)

    return wrapper


def _sync_common_config_state() -> None:
    _config_common.ROOT = ROOT
    _config_common.CONFIGS_DIR = CONFIGS_DIR


def _load(p: Path) -> dict:
    _sync_common_config_state()
    return _config_common._load(p)


def _safe_config_subdir(subdir: str) -> Path | None:
    _sync_common_config_state()
    return _config_common._safe_config_subdir(subdir)


def _resolve_project_path(value: str) -> Path:
    _sync_common_config_state()
    return _config_common._resolve_project_path(value)


def _auto_data_dir_for_key(value: Any, source_path: Path, suffix: str) -> Path:
    _sync_common_config_state()
    return _config_common._auto_data_dir_for_key(value, source_path, suffix)


def _derived_data_dir(source_path: Path, suffix: str) -> Path:
    _sync_common_config_state()
    return _config_common._derived_data_dir(source_path, suffix)


def _is_builtin_default_data_dir(value: str) -> bool:
    _sync_common_config_state()
    return _config_common._is_builtin_default_data_dir(value)


def _display_path(path: Path) -> str:
    _sync_common_config_state()
    return _config_common._display_path(path)


__all__ = ['list_methods', 'list_variants', 'list_all_variants', 'list_presets', 'load_merged_config', 'suggest_data_dirs', 'suggest_dataset_dirs', 'apply_auto_data_dirs']

def list_methods() -> list[str]:
    """Discover method families from configs/methods; known only orders names."""
    known = [
        "lora", "lokr", "ortholora", "tlora", "hydralora",
        "reft", "chimera", "soft_tokens", "ip_adapter", "easycontrol",
        "spd",
    ]
    found: list[str] = []
    try:
        methods_dir = Path(CONFIGS_DIR) / "methods"
        if methods_dir.is_dir():
            for path in sorted(methods_dir.glob("*.toml")):
                name = path.stem
                if name and name not in found:
                    found.append(name)
    except OSError:
        pass
    # known only orders; missing files must not appear as ghosts.
    ordered = [name for name in known if name in found]
    extras = [name for name in found if name not in known]
    return ordered + extras


def list_variants(method: str) -> list[str]:
    if method == "spd":
        spd_config = CONFIGS_DIR / "methods" / "spd.toml"
        return ["spd"] if spd_config.exists() else []
    if not GUI_METHODS_DIR.exists():
        return []
    variants = [stem for _order, stem in _builtin_variants_by_family().get(method, [])]
    if not variants:
        variants = _legacy_exact_variant_for_method(method)
    variants.extend(_custom_gui_variants())
    return variants


def _builtin_variants_by_family() -> dict[str, list[tuple[int, str]]]:
    by_family: dict[str, list[tuple[int, str]]] = {}
    if not GUI_METHODS_DIR.exists():
        return by_family
    for path in GUI_METHODS_DIR.glob("*.toml"):
        if _display_path(path) in HIDDEN_CONFIG_FILES:
            continue
        meta = _read_variant_metadata(path)
        family = meta.get("family")
        if not isinstance(family, str) or not family:
            continue
        order = meta.get("order")
        order_int = order if isinstance(order, int) else 100
        by_family.setdefault(family, []).append((order_int, path.stem))
    for entries in by_family.values():
        entries.sort(key=lambda item: (item[0], item[1]))
    return by_family


def _read_variant_metadata(path: Path) -> dict[str, Any]:
    try:
        data = toml.loads(path.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError):
        return {}
    meta = data.get("variant")
    return meta if isinstance(meta, dict) else {}


def _legacy_exact_variant_for_method(method: str) -> list[str]:
    path = GUI_METHODS_DIR / f"{method}.toml"
    if not path.is_file() or _display_path(path) in HIDDEN_CONFIG_FILES:
        return []
    return [method]


def _custom_gui_variants() -> list[str]:
    custom_dir = GUI_METHODS_DIR / "custom"
    if not custom_dir.exists():
        return []
    return [f"custom/{p.stem}" for p in sorted(custom_dir.glob("*.toml"))]


def list_all_variants() -> list[str]:
    if not GUI_METHODS_DIR.exists():
        return []
    return sorted(
        p.stem for p in GUI_METHODS_DIR.glob("*.toml")
        if _display_path(p) not in HIDDEN_CONFIG_FILES
    )


def list_presets() -> list[str]:
    names: set[str] = set()
    if PRESETS_FILE.exists():
        data = toml.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        names.update(k for k, v in data.items() if isinstance(v, dict))
    base_dir = PRESETS_FILE.parent if PRESETS_FILE != CONFIGS_DIR / "presets.toml" else CONFIGS_DIR
    for subdir in config_io._CUSTOM_PRESET_SUBDIRS:
        custom_dir = Path(base_dir) / subdir
        if custom_dir.is_dir():
            names.update(p.stem for p in custom_dir.glob("*.toml"))
    return sorted(names)


def load_merged_config(variant: str, preset: str, methods_subdir: str = "gui-methods") -> dict[str, Any]:
    methods_dir = _safe_config_subdir(methods_subdir)
    if methods_dir is None:
        raise ValueError("配置目录不合法")
    # Share flatten/alias/schema core with training merge, then apply WebUI-only
    # auto data-dir derivation on top.
    try:
        merged = config_io.load_method_preset(
            variant,
            preset=preset,
            configs_dir=str(CONFIGS_DIR),
            methods_subdir=methods_subdir,
            strict=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    merged = dict(merged)
    merged.setdefault("max_train_steps", DEFAULT_MAX_TRAIN_STEPS)
    merged = expand_env_vars_in_obj(merged)
    return apply_auto_data_dirs(merged)


def suggest_data_dirs(source_image_dir: str, *, inspect: bool = False) -> dict[str, Any]:
    source_path = _resolve_project_path(str(source_image_dir or ""))
    if not str(source_image_dir or "").strip():
        return {"ok": False, "error": "请先填写源图像目录 / source_image_dir"}
    result = {
        "ok": True,
        "source_image_dir": _display_path(source_path),
        "resized_image_dir": _display_path(_derived_data_dir(source_path, "resized")),
        "lora_cache_dir": _display_path(_derived_data_dir(source_path, "lora_cache")),
    }
    if inspect:
        result.update(_inspect_source_image_dir(source_path))
    return result


def _inspect_source_image_dir(source_path: Path) -> dict[str, Any]:
    try:
        exists = source_path.exists()
        is_dir = exists and source_path.is_dir()
    except OSError as exc:
        return {
            "source_exists": False,
            "source_is_dir": False,
            "source_image_count": 0,
            "source_inspection_error": str(exc),
        }
    image_count = 0
    inspection_error = ""
    if is_dir:
        try:
            image_count = sum(
                1 for item in source_path.rglob("*")
                if item.is_file() and item.suffix.casefold() in DATASET_IMAGE_SUFFIXES
            )
        except OSError as exc:
            inspection_error = str(exc)
    return {
        "source_exists": exists,
        "source_is_dir": is_dir,
        "source_image_count": image_count,
        "source_inspection_error": inspection_error,
    }


def suggest_dataset_dirs(source_dirs: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(source_dirs):
        source = str(raw or "").strip()
        if not source:
            continue
        source_path = _resolve_project_path(source)
        rows.append({
            "index": idx,
            "source_dir": _display_path(source_path),
            "image_dir": _display_path(_derived_data_dir(source_path, "resized")),
            "cache_dir": _display_path(_derived_data_dir(source_path, "lora_cache")),
        })
    if not rows:
        return {"ok": False, "error": "请至少填写一个原始数据集路径"}
    return {"ok": True, "datasets": rows}


def apply_auto_data_dirs(cfg: dict[str, Any], *, create: bool = False) -> dict[str, Any]:
    next_cfg = dict(cfg)
    source_raw = str(next_cfg.get("source_image_dir") or "").strip()
    if not source_raw:
        return next_cfg
    source_path = _resolve_project_path(source_raw)
    resized_path = _auto_data_dir_for_key(next_cfg.get("resized_image_dir"), source_path, "resized")
    cache_path = _auto_data_dir_for_key(next_cfg.get("lora_cache_dir"), source_path, "lora_cache")
    next_cfg["resized_image_dir"] = _display_path(resized_path)
    next_cfg["lora_cache_dir"] = _display_path(cache_path)
    if create:
        resized_path.mkdir(parents=True, exist_ok=True)
        cache_path.mkdir(parents=True, exist_ok=True)
    return next_cfg




for _name in __all__:
    globals()[_name] = _exported(globals()[_name])
