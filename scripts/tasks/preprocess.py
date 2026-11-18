"""Default-dataset preprocessing: resize → VAE latents → text-embedding caches."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None
    import toml as _toml  # type: ignore[no-redef]
else:
    _toml = None

from ._common import PY, ROOT, _path, run

RUNTIME_PREPROCESS_ATTR_KEY = "preprocess"
CAPTION_SOURCE_MODES = {"auto", "txt", "json", "captions_json"}
_CAPTION_INDEX_VOCAB = "models/captioners/anima-tagger-v2/vocab.json"
_PREPROCESS_PRECISION_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
}


# Subfolders under the source dir are walked by default — matches the
# `recursive = true` subset default in configs/base.toml. Stems must stay
# unique across the tree (cache filenames are stem-keyed and flat). Pass
# `--no_recursive` (or edit configs) to opt out.
def _min_pixels_args() -> list[str]:
    """``--min_pixels <N>`` derived from the variant TOML's
    ``drop_lowres_images`` + ``min_pixels`` keys (resolved through the same
    base → preset → method merge chain training uses, via ``_path_overrides``
    in scripts/tasks/_common.py).

    Returns ``[]`` when both keys are absent so plain CLI use keeps each
    script's own argparse default (500_000 = 0.5MP). ``drop_lowres_images
    = false`` forces ``--min_pixels 0`` even when ``min_pixels`` is set, so
    the user can flip a single boolean to disable the filter."""
    from ._common import _path_overrides  # local import: avoids unused circular

    overrides = _path_overrides()
    if "drop_lowres_images" not in overrides and "min_pixels" not in overrides:
        return []
    if overrides.get("drop_lowres_images") is False:
        return ["--min_pixels", "0"]
    raw = overrides.get("min_pixels", 500_000)
    try:
        n = max(0, int(raw))
    except (TypeError, ValueError):
        return []
    return ["--min_pixels", str(n)]


def _config_min_pixels() -> int:
    """The configured ``min_pixels`` threshold, default 0.5MP."""
    from ._common import _path_overrides

    raw = _path_overrides().get("min_pixels", 500_000)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 500_000


def _resolve_lowres_filter(extra: list[str]) -> tuple[list[str], list[str]]:
    """Reconcile low-res filtering convenience flags with forwarded args.

    ``--min_pixels`` is authoritative when passed explicitly. ``--no_drop_lowres``
    keeps every image; ``--drop_lowres`` forces the configured threshold.
    """
    cleaned = list(extra)
    no_drop = "--no_drop_lowres" in cleaned
    drop = "--drop_lowres" in cleaned
    cleaned = [arg for arg in cleaned if arg not in ("--no_drop_lowres", "--drop_lowres")]

    if "--min_pixels" in cleaned:
        return [], cleaned
    if no_drop:
        return ["--min_pixels", "0"], cleaned
    if drop:
        return ["--min_pixels", str(_config_min_pixels())], cleaned
    return _min_pixels_args(), cleaned


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return _truthy(value)


def _caption_source_mode(value: Any = None, prefer_json: Any = None) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in CAPTION_SOURCE_MODES:
        return raw
    aliases = {
        "text": "txt",
        ".txt": "txt",
        "same_stem_json": "json",
        ".json": "json",
        "captions.json": "captions_json",
        "captions-json": "captions_json",
        "diffpipeforge": "captions_json",
    }
    if raw in aliases:
        return aliases[raw]
    return "json" if _truthy(prefer_json) else "auto"


def _explicit_caption_source_mode(value: Any) -> bool:
    return bool(str(value or "").strip())


def _caption_source_args(value: Any, prefer_json: Any = None) -> list[str]:
    if not _explicit_caption_source_mode(value):
        return []
    return ["--caption_source_mode", _caption_source_mode(value, prefer_json)]


def _vocab_pack_args() -> list[str]:
    """Forward the optional Anima pack only when the merged config selects one."""
    value = str(_path_overrides_value().get("vocab_pack_path") or "").strip()
    return ["--vocab_pack_path", value] if value else []


def _first_explicit_caption_source_mode(*values: Any) -> Any:
    for value in values:
        if _explicit_caption_source_mode(value):
            return value
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        nums = [_positive_int(item) for item in value]
        nums = [item for item in nums if item is not None]
        return max(nums) if nums else None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None



def _reuse_overwrite_args(row: dict, *, kind: str) -> list[str]:
    """Return ``--overwrite`` when policy asks to rebuild VAE or TE.

    Reads row-level overrides first, then global path overrides / env via
    ``load_path_overrides`` keys if present on the row.
    """
    from library.cache_pool.policy import parse_cache_reuse_policy

    # Prefer explicit row flags, then flatten common cfg keys from row.
    cfg = {}
    for key in (
        "reuse_dataset_cache_copy",
        "reuse_vae_latents",
        "reuse_text_encoder_cache",
        "cache_fingerprint_mode",
        "force_rebuild_preprocess_cache",
    ):
        if key in row:
            cfg[key] = row[key]
    # Also allow nested settings.
    settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    for key in list(cfg.keys()) or []:
        pass
    for key in (
        "reuse_dataset_cache_copy",
        "reuse_vae_latents",
        "reuse_text_encoder_cache",
        "cache_fingerprint_mode",
        "force_rebuild_preprocess_cache",
    ):
        if key not in cfg and key in settings:
            cfg[key] = settings[key]
    # Fall back to process path overrides (merged training config).
    try:
        from scripts.tasks._common import load_path_overrides  # type: ignore
    except Exception:
        load_path_overrides = None
    if load_path_overrides is not None:
        try:
            overrides = load_path_overrides()
            if isinstance(overrides, dict):
                for key in (
                    "reuse_dataset_cache_copy",
                    "reuse_vae_latents",
                    "reuse_text_encoder_cache",
                    "cache_fingerprint_mode",
                    "force_rebuild_preprocess_cache",
                ):
                    if key not in cfg and key in overrides:
                        cfg[key] = overrides[key]
        except Exception:
            pass
    policy = parse_cache_reuse_policy(cfg)
    overwrite = []
    if policy.force_rebuild:
        overwrite = ["--overwrite"]
    elif kind == "vae" and not policy.reuse_vae_latents:
        overwrite = ["--overwrite"]
    elif kind == "te" and not policy.reuse_text_encoder_cache:
        overwrite = ["--overwrite"]
    return [*overwrite, *_require_cache_identity_args(row)]


def _require_cache_identity_args(row: dict) -> list[str]:
    """Forward --require_cache_identity when the merged config asks for it.

    Lets a run that opts into identity checking also rebuild pre-identity
    sidecars during preprocessing instead of only rejecting them at training
    time. Read from the row, then its settings block, then the runtime / base /
    preset / method override chain (the same sources as the reuse policy).
    """
    settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    for source in (row, settings):
        if "require_cache_identity" in source:
            return (
                ["--require_cache_identity"]
                if _truthy(source["require_cache_identity"])
                else []
            )
    try:
        overrides = _path_overrides_value()
    except Exception:  # defensive: overrides are best-effort
        overrides = {}
    if "require_cache_identity" in overrides:
        return (
            ["--require_cache_identity"]
            if _truthy(overrides["require_cache_identity"])
            else []
        )
    return []


def _preprocess_cache_batch_sizes() -> tuple[int | str, int | str]:
    from ._common import _path_overrides  # local import: avoids unused circular
    from library.preprocess.batch_config import resolve_cache_batch_sizes

    return resolve_cache_batch_sizes(_path_overrides())


def _preprocess_precision_dtype() -> str:
    from ._common import _path_overrides  # local import: avoids unused circular

    overrides = _path_overrides()
    raw = str(overrides.get("preprocess_precision_preference") or "").strip().lower()
    if not raw:
        mixed_precision = str(overrides.get("mixed_precision") or "").strip().lower()
        if mixed_precision == "fp16":
            raw = "fp16"
        elif mixed_precision == "no":
            raw = "fp32"
        else:
            raw = "bf16"
    return _PREPROCESS_PRECISION_ALIASES.get(raw, "bfloat16")


def _preprocess_settings_from_custom_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    raw = attrs.get(RUNTIME_PREPROCESS_ATTR_KEY) if isinstance(attrs, dict) else None
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("resolution", "min_bucket_reso", "max_bucket_reso", "bucket_reso_steps"):
        value = _positive_int(raw.get(key))
        if value is not None:
            out[key] = value
    if "bucket_no_upscale" in raw:
        out["bucket_no_upscale"] = raw.get("bucket_no_upscale")
    if "enable_bucket" in raw:
        out["enable_bucket"] = raw.get("enable_bucket")
    return out


def _resolve_project_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _load_toml_file(path: Path) -> dict[str, Any]:
    if tomllib is not None:
        with path.open("rb") as f:
            return tomllib.load(f)
    return _toml.load(path)


def _dataset_path_value(value: Any, overrides: dict[str, Any]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return text.format_map({key: str(val) for key, val in overrides.items()})
    except (KeyError, ValueError, IndexError):
        return text


def _same_path_text(left: Any, right: Any) -> bool:
    def normalize(value: Any) -> str:
        return str(value or "").replace("\\", "/").rstrip("/")

    return bool(normalize(left)) and normalize(left) == normalize(right)


def _has_extra_path_override(extra: list[str], *flags: str) -> bool:
    prefixes = tuple(f"{flag}=" for flag in flags)
    return any(arg in flags or arg.startswith(prefixes) for arg in extra)


_DIFF_OUTPUT_PRESERVATION_FLAGS = (
    "--diff_output_preservation_trigger",
    "--diff_output_preservation_class",
)


def _split_diff_output_preservation_args(extra: list[str]) -> tuple[list[str], list[str]]:
    cleaned: list[str] = []
    dop_args: list[str] = []
    i = 0
    while i < len(extra):
        arg = extra[i]
        if arg in _DIFF_OUTPUT_PRESERVATION_FLAGS and i + 1 < len(extra):
            dop_args.extend([arg, extra[i + 1]])
            i += 2
        elif any(arg.startswith(f"{flag}=") for flag in _DIFF_OUTPUT_PRESERVATION_FLAGS):
            dop_args.append(arg)
            i += 1
        else:
            cleaned.append(arg)
            i += 1
    return cleaned, dop_args


def _diff_output_preservation_args(explicit_args: list[str]) -> list[str]:
    if explicit_args:
        return explicit_args

    from ._common import _path_overrides  # local import: avoids unused circular

    overrides = _path_overrides()
    out: list[str] = []
    for key, flag in (
        ("diff_output_preservation_trigger", "--diff_output_preservation_trigger"),
        ("diff_output_preservation_class", "--diff_output_preservation_class"),
    ):
        value = str(overrides.get(key) or "").strip()
        if value:
            out.extend([flag, value])
    return out


def _dataset_rows(dataset_config: Any, overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    overrides = overrides or {}
    path = _resolve_project_path(dataset_config)
    if path is None or not path.is_file():
        return []
    try:
        data = _load_toml_file(path)
    except Exception as e:  # noqa: BLE001
        print(f"warn: could not read dataset_config for preprocess rows: {e}")
        return []
    datasets = data.get("datasets") or []
    if not isinstance(datasets, list):
        return []
    general = data.get("general") if isinstance(data.get("general"), dict) else {}
    general_caption_extension = str(general.get("caption_extension") or "").strip()
    general_prefer_json = _optional_bool(
        general.get("prefer_json_caption", general.get("prefer_json"))
    )
    general_caption_source_mode = str(general.get("caption_source_mode") or "").strip()

    rows: list[dict[str, Any]] = []
    fallback_source = str(overrides.get("source_image_dir") or "")
    fallback_image = str(overrides.get("resized_image_dir") or fallback_source)
    fallback_cache = str(overrides.get("lora_cache_dir") or "")
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        subsets = dataset.get("subsets") or []
        if not isinstance(subsets, list):
            continue
        for subset in subsets:
            if not isinstance(subset, dict):
                continue
            attrs = subset.get("custom_attributes")
            if not isinstance(attrs, dict):
                attrs = {}
            image_dir = _dataset_path_value(subset.get("image_dir") or fallback_image, overrides)
            cache_dir = _dataset_path_value(subset.get("cache_dir") or fallback_cache, overrides)
            text_cache_dir = (
                _dataset_path_value(subset.get("text_cache_dir"), overrides)
                if str(subset.get("text_cache_dir") or "").strip()
                else ""
            )
            cond_cache_dir = (
                _dataset_path_value(subset.get("cond_cache_dir"), overrides)
                if str(subset.get("cond_cache_dir") or "").strip()
                else ""
            )
            if attrs.get("source_dir"):
                source_dir = _dataset_path_value(attrs.get("source_dir"), overrides)
            elif image_dir and _same_path_text(image_dir, fallback_image):
                source_dir = _dataset_path_value(fallback_source or image_dir, overrides)
            else:
                source_dir = image_dir or _dataset_path_value(fallback_source, overrides)
            if not image_dir and not source_dir:
                continue
            row = dict(dataset)
            row.pop("subsets", None)
            row.update(_preprocess_settings_from_custom_attributes(attrs))
            row.update(
                {
                    "source_image_dir": source_dir,
                    "resized_image_dir": image_dir,
                    "lora_cache_dir": cache_dir,
                    "text_cache_dir": text_cache_dir,
                    "cond_cache_dir": cond_cache_dir,
                    "recursive": subset.get("recursive", dataset.get("recursive", True)),
                    "path_pattern": subset.get("path_pattern", dataset.get("path_pattern", "*")),
                }
            )
            prefer_json = _optional_bool(
                dataset.get(
                    "prefer_json_caption",
                    subset.get("prefer_json_caption"),
                )
            )
            if prefer_json is None:
                prefer_json = general_prefer_json
            if prefer_json is not None:
                row["prefer_json_caption"] = prefer_json
            caption_source_mode = _first_explicit_caption_source_mode(
                dataset.get("caption_source_mode"),
                subset.get("caption_source_mode"),
                general_caption_source_mode,
            )
            if caption_source_mode is not None:
                row["caption_source_mode"] = _caption_source_mode(caption_source_mode, prefer_json)
            caption_extension = str(
                dataset.get(
                    "caption_extension",
                    subset.get("caption_extension", general_caption_extension),
                )
                or ""
            ).strip()
            if caption_extension:
                row["caption_extension"] = caption_extension
            rows.append(row)
    return rows


def _first_dataset_settings(dataset_config: Any) -> dict[str, Any]:
    path = _resolve_project_path(dataset_config)
    if path is None or not path.is_file():
        return {}
    try:
        data = _load_toml_file(path)
    except Exception as e:  # noqa: BLE001
        print(f"warn: could not read dataset_config for resize settings: {e}")
        return {}
    datasets = data.get("datasets") or []
    first = datasets[0] if datasets and isinstance(datasets[0], dict) else {}
    if not isinstance(first, dict):
        return {}
    first = dict(first)
    first.pop("subsets", None)
    return first


def _preprocess_rows() -> list[dict[str, Any]]:
    """Return one preprocess row per dataset_config subset, or legacy top-level paths."""
    from ._common import _path_overrides  # local import: avoids unused circular

    overrides = _path_overrides()
    rows = _dataset_rows(overrides.get("dataset_config"), overrides)
    if rows:
        return rows
    return [
        {
            **dict(overrides),
            "source_image_dir": _path("source_image_dir", "image_dataset"),
            "resized_image_dir": _path("resized_image_dir", "post_image_dataset/resized"),
            "lora_cache_dir": _path("lora_cache_dir", "post_image_dataset/lora"),
            "recursive": True,
        }
    ]


def _resize_bucket_args(settings: dict[str, Any] | None = None) -> list[str]:
    """Forward dataset bucket settings to resize_images.py.

    WebUI stores these in dataset_config, while tasks.py preprocess historically
    only read top-level path keys. Callers pass one dataset row at a time so
    multi-dataset WebUI presets get matching resized images and caches.
    """
    from ._common import _path_overrides  # local import: avoids unused circular

    overrides = _path_overrides()
    if settings is None:
        settings = dict(overrides)
        rows = _preprocess_rows()
        if rows:
            settings.update(rows[0])
        settings.update(_first_dataset_settings(overrides.get("dataset_config")))
    else:
        merged = dict(overrides)
        merged.update(settings)
        settings = merged

    out: list[str] = []
    if _model_family() == "qwen_image_2_1":
        out.extend(["--model_family", "qwen_image_2_1"])
    mapping = (
        ("resolution", "--resolution"),
        ("min_bucket_reso", "--min_bucket_reso"),
        ("max_bucket_reso", "--max_bucket_reso"),
        ("bucket_reso_steps", "--bucket_reso_steps"),
    )
    for key, flag in mapping:
        n = _positive_int(settings.get(key))
        if n is not None:
            out.extend([flag, str(n)])
    if _truthy(settings.get("bucket_no_upscale")):
        out.append("--bucket_no_upscale")
    if not _truthy(settings.get("enable_bucket", True)):
        out.append("--no_enable_bucket")
    return out


def _recursive_args(row: dict[str, Any]) -> list[str]:
    return ["--recursive"] if _truthy(row.get("recursive", True)) else []


def _path_pattern_args(row: dict[str, Any]) -> list[str]:
    pattern = str(row.get("path_pattern") or "*").strip() or "*"
    return ["--path_pattern", pattern] if pattern != "*" else []


def _caption_extension_for_row(row: dict[str, Any]) -> str:
    from ._common import _path_overrides  # local import: avoids unused circular

    return str(
        row.get("caption_extension")
        or _path_overrides().get("caption_extension")
        or ".txt"
    )


def _caption_extension_args_for_row(row: dict[str, Any]) -> list[str]:
    return ["--caption_extension", _caption_extension_for_row(row)]


def _caption_backup_dir_for_row(row: dict[str, Any]) -> Path | None:
    cache_raw = str(row.get("lora_cache_dir") or _path("lora_cache_dir", "post_image_dataset/lora"))
    cache_dir = _resolve_project_path(cache_raw)
    if cache_dir is None:
        return None
    return cache_dir.parent / "caption_backup"


def _run_caption_backup(row: dict[str, Any]) -> None:
    from library.preprocess import backup_caption_sidecars

    src = _resolve_project_path(row.get("source_image_dir") or _path("source_image_dir", "image_dataset"))
    backup_dir = _caption_backup_dir_for_row(row)
    if src is None or backup_dir is None:
        print("warn: caption backup skipped: source or backup directory is empty")
        return
    stats = backup_caption_sidecars(
        src,
        backup_dir,
        recursive=_truthy(row.get("recursive", True)),
        path_pattern=str(row.get("path_pattern") or "*"),
        caption_extension=_caption_extension_for_row(row),
        warn=lambda msg: print(f"warn: {msg}"),
    )
    print(
        "caption backup: "
        f"{stats.copied} files copied to {backup_dir} "
        f"({stats.missing} images without sidecar, {stats.failed} failed)"
    )


def _run_preprocess_resize(row: dict[str, Any], extra: list[str], *, copy_captions: bool = False) -> None:
    mp_args, extra = _resolve_lowres_filter(extra)
    src = str(row.get("source_image_dir") or _path("source_image_dir", "image_dataset"))
    dst = str(row.get("resized_image_dir") or _path("resized_image_dir", "post_image_dataset/resized"))
    if _same_path_text(src, dst) and not _has_extra_path_override(extra, "--src", "--dst"):
        print(f"skip resize: source_image_dir and resized_image_dir are the same: {src}")
        return
    run(
        [
            PY,
            "-m",
            "scripts.preprocess.resize_images",
            "--src",
            src,
            "--dst",
            dst,
            *([] if copy_captions else ["--no_copy_captions"]),
            *_recursive_args(row),
            *_path_pattern_args(row),
            *_resize_bucket_args(row),
            *mp_args,
            *extra,
        ]
    )


def _run_preprocess_vae(row: dict[str, Any], extra: list[str]) -> None:
    vae_batch_size, _ = _preprocess_cache_batch_sizes()
    dtype = "float32" if _model_family() == "moss_image" else _preprocess_precision_dtype()
    run(
        [
            PY,
            "-m",
            "scripts.preprocess.cache_latents",
            "--dir",
            str(row.get("resized_image_dir") or _path("resized_image_dir", "post_image_dataset/resized")),
            "--cache_dir",
            str(row.get("lora_cache_dir") or _path("lora_cache_dir", "post_image_dataset/lora")),
            "--vae",
            _path("vae", "models/vae/qwen_image_vae.safetensors"),
            "--model_family",
            _model_family(),
            "--batch_size",
            str(vae_batch_size),
            "--dtype",
            dtype,
            "--chunk_size",
            "64",
            *_recursive_args(row),
            *_path_pattern_args(row),
            *_reuse_overwrite_args(row, kind="vae"),
            *extra,
        ]
    )


def cmd_preprocess_te(extra):
    from .qwen_edit_preprocess import uses_qwen_edit

    if uses_qwen_edit(_model_family(), _path_overrides_value()):
        raise ValueError("Qwen Image 2.1 Edit requires the full `preprocess` task; preprocess-te is unsupported")
    from library.preprocess.caption_cache_settings import resolve_caption_cache_settings

    shuffle_variants, tag_dropout_rate = resolve_caption_cache_settings(
        _path_overrides_value()
    )
    for row in _preprocess_rows():
        _run_preprocess_te(row, extra, str(shuffle_variants), str(tag_dropout_rate))


def _run_preprocess_te(
    row: dict[str, Any],
    extra: list[str],
    shuffle_variants: str | None = None,
    tag_dropout_rate: str | None = None,
    backup_captions: bool = True,
) -> None:
    if backup_captions:
        _run_caption_backup(row)
    family = _model_family()
    from library.models.family_registry import get_family_plugin

    plugin = get_family_plugin(family)
    spec = plugin.text_preprocess_spec
    if spec.runner == "special":
        if family == "qwen_image_2_1":
            _run_preprocess_te_qwen_image_2_1(
                row, extra, shuffle_variants, tag_dropout_rate
            )
        elif family == "moss_image":
            _run_preprocess_te_moss_image(
                row, extra, shuffle_variants, tag_dropout_rate
            )
        else:  # pragma: no cover - guarded by plugin metadata
            raise RuntimeError(f"unsupported special text preprocess family: {family}")
        return
    _run_preprocess_te_from_plugin(
        row, extra, shuffle_variants, tag_dropout_rate, plugin=plugin
    )


def _run_preprocess_te_from_plugin(
    row: dict[str, Any],
    extra: list[str],
    shuffle_variants: str | None,
    tag_dropout_rate: str | None,
    *,
    plugin,
) -> None:
    """Build the common Anima/Krea-2/Z-Image text-cache command."""
    spec = plugin.text_preprocess_spec
    if spec.module is None:
        raise RuntimeError(f"text preprocess plugin has no module: {plugin.family}")
    if spec.strip_diff_output_preservation:
        extra, _ = _split_diff_output_preservation_args(list(extra))
    mp_args, extra = _resolve_lowres_filter(extra)
    shuffle_variants = shuffle_variants or os.environ.get(
        "CAPTION_SHUFFLE_VARIANTS", spec.default_shuffle_variants
    )
    tag_dropout_rate = tag_dropout_rate or os.environ.get(
        "CAPTION_TAG_DROPOUT_RATE", spec.default_tag_dropout_rate
    )
    prefer_json_env = os.environ.get("CAPTION_PREFER_JSON")
    prefer_json = (
        _truthy(prefer_json_env)
        if prefer_json_env is not None
        else _truthy(row.get("prefer_json_caption"))
    )
    source_mode_env = os.environ.get("CAPTION_SOURCE_MODE")
    source_mode = (
        source_mode_env
        if source_mode_env is not None
        else row.get("caption_source_mode")
    )
    _, text_batch_size = _preprocess_cache_batch_sizes()
    cmd = [
        PY,
        "-m",
        spec.module,
        "--dir",
        str(row.get("source_image_dir") or _path("source_image_dir", "image_dataset")),
        "--cache_dir",
        _text_cache_dir_for_row(row),
    ]
    if spec.qwen3_config_key:
        cmd.extend(
            [
                "--qwen3",
                _path(spec.qwen3_config_key, spec.qwen3_default or ""),
            ]
        )
    if spec.dit_config_key:
        cmd.extend(
            [
                "--dit",
                _path(spec.dit_config_key, spec.dit_default or ""),
            ]
        )
    cmd.extend(
        [
            "--batch_size",
            str(text_batch_size),
            "--dtype",
            _preprocess_precision_dtype(),
            "--caption_shuffle_variants",
            str(shuffle_variants),
        ]
    )
    if spec.supports_tag_dropout:
        cmd.extend(["--caption_tag_dropout_rate", str(tag_dropout_rate)])
    optional_args = [
        *_caption_source_args(source_mode, prefer_json),
        *_caption_extension_args_for_row(row),
        *( ["--prefer_json_caption"] if prefer_json else [] ),
        *_recursive_args(row),
        *_path_pattern_args(row),
        *mp_args,
    ]
    if spec.supports_diff_output_preservation:
        optional_args.extend(_diff_output_preservation_args(extra))
    optional_args.extend([*_reuse_overwrite_args(row, kind="te"), *extra])
    cmd.extend(optional_args)
    run(cmd)


def _run_preprocess_te_moss_image(row, extra, shuffle_variants=None, tag_dropout_rate=None):
    from scripts.moss_image.preprocess_te_cache import run_task

    run_task(row, extra, shuffle_variants, tag_dropout_rate)


def _run_preprocess_te_qwen_image_2_1(
    row: dict[str, Any], extra: list[str],
    shuffle_variants: str | None = None, tag_dropout_rate: str | None = None,
) -> None:
    from library.models.qwen_image_2_1.cache_policy import validate_cache_policy

    policy = validate_cache_policy(_path_overrides_value().get("qwen_text_encoder_cache_policy", "auto"))
    if float(tag_dropout_rate or 0):
        raise ValueError("Qwen Image 2.1 caption tag dropout is not supported")
    mp_args, extra = _resolve_lowres_filter(extra)
    _, text_batch_size = _preprocess_cache_batch_sizes()
    prefer_json_env = os.environ.get("CAPTION_PREFER_JSON")
    prefer_json = (
        _truthy(prefer_json_env)
        if prefer_json_env is not None
        else _truthy(row.get("prefer_json_caption"))
    )
    source_mode = os.environ.get("CAPTION_SOURCE_MODE") or row.get("caption_source_mode")
    run([
        PY, "-m", "scripts.qwen_image_2_1.preprocess_te_cache",
        "--cache_policy", policy,
        "--dir", str(row.get("source_image_dir") or _path("source_image_dir", "image_dataset")),
        "--cache_dir", _text_cache_dir_for_row(row),
        "--qwen3", _path("qwen3", "models/text_encoders/Qwen-Image-2.1"),
        "--batch_size", str(text_batch_size),
        "--dtype", _preprocess_precision_dtype(),
        "--caption_shuffle_variants", str(shuffle_variants or 0),
        *_caption_source_args(source_mode, prefer_json),
        *_caption_extension_args_for_row(row),
        *(["--prefer_json_caption"] if prefer_json else []),
        *_recursive_args(row),
        *_path_pattern_args(row),
        *mp_args,
        *_reuse_overwrite_args(row, kind="te"),
        *extra,
    ])


def _model_family() -> str:
    """Preprocess-side model family (anima default; krea2_raw → _krea2_te).

    Mirrors ``library.env.resolve_model_family`` but reads from the same
    ``_path_overrides`` merge chain the preprocess task already uses (runtime
    config / base.toml / method) so a WebUI run with ``model_family =
    "krea2_raw"`` in its runtime config dispatches TE caching to the Krea-2
    Qwen3-VL ChatML path + ``_krea2_te.safetensors`` suffix. Anim-only scripts
    and runs without the key fall back to anima.
    """
    from library.env import normalize_model_family

    raw = _path_overrides_value().get("model_family")
    if not str(raw or "").strip():
        return "anima"
    return normalize_model_family(raw, source="preprocess model_family")


def _path_overrides_value() -> dict[str, Any]:
    from ._common import _path_overrides  # local import: avoids unused circular

    return _path_overrides()


def _run_preprocess_te_krea2(
    row: dict[str, Any],
    extra: list[str],
    shuffle_variants: str | None = None,
    tag_dropout_rate: str | None = None,
) -> None:
    """Krea-2 TE cache: writes ``{stem}_krea2_te.safetensors`` via the
    Qwen3-VL ChatML path. T5/LLM-adapter/crossattn remain anima-only, while
    caption variants use the shared preprocessing contract."""
    shuffle_variants = shuffle_variants or os.environ.get("CAPTION_SHUFFLE_VARIANTS", "4")
    tag_dropout_rate = tag_dropout_rate or os.environ.get("CAPTION_TAG_DROPOUT_RATE", "0.1")
    extra, _dop_args = _split_diff_output_preservation_args(list(extra))
    mp_args, extra = _resolve_lowres_filter(extra)
    prefer_json_env = os.environ.get("CAPTION_PREFER_JSON")
    prefer_json = (
        _truthy(prefer_json_env)
        if prefer_json_env is not None
        else _truthy(row.get("prefer_json_caption"))
    )
    source_mode_env = os.environ.get("CAPTION_SOURCE_MODE")
    caption_source_value = source_mode_env if source_mode_env is not None else row.get("caption_source_mode")
    json_args = ["--prefer_json_caption"] if prefer_json else []
    source_args = _caption_source_args(caption_source_value, prefer_json)
    _, text_batch_size = _preprocess_cache_batch_sizes()
    dtype = _preprocess_precision_dtype()
    run(
        [
            PY,
            "-m",
            "scripts.krea2.preprocess_te_cache",
            "--dir",
            # TE encode only needs captions (no image pixels); run on the source
            # dir so ``captions.json`` / ``.txt`` sidecars are found (the resize
            # step mirrors ``.txt`` sidecars but not the ``captions.json`` master,
            # so the resized dir may have no captions at all). The cache file's
            # stem is derived from the image path sans extension, so a source
            # ``foo.jpg`` writes ``foo_krea2_te.safetensors`` into ``cache_dir`` —
            # matching the resized ``.png``'s latent ``foo_{WxH}_anima.npz`` stem
            # (resize keeps the stem, only swaps the extension). Mirrors anima's
            # ``_run_preprocess_te_anima`` which also uses ``source_image_dir``.
            str(row.get("source_image_dir") or _path("source_image_dir", "image_dataset")),
            "--cache_dir",
            _text_cache_dir_for_row(row),
            "--qwen3",
            _path("qwen3", "models/text_encoders/qwen3vl_4b_bf16.safetensors"),
            "--batch_size",
            str(text_batch_size),
            "--dtype",
            dtype,
            "--caption_shuffle_variants",
            shuffle_variants,
            "--caption_tag_dropout_rate",
            tag_dropout_rate,
            *source_args,
            *_caption_extension_args_for_row(row),
            *json_args,
            *_recursive_args(row),
            *_path_pattern_args(row),
            *mp_args,
            *_reuse_overwrite_args(row, kind="te"),
            *extra,
        ]
    )


def _run_preprocess_te_z_image(
    row: dict[str, Any],
    extra: list[str],
    shuffle_variants: str | None = None,
    tag_dropout_rate: str | None = None,
) -> None:
    shuffle_variants = shuffle_variants or os.environ.get(
        "CAPTION_SHUFFLE_VARIANTS", "0"
    )
    tag_dropout_rate = tag_dropout_rate or os.environ.get(
        "CAPTION_TAG_DROPOUT_RATE", "0.0"
    )
    mp_args, extra = _resolve_lowres_filter(extra)
    _, text_batch_size = _preprocess_cache_batch_sizes()
    dtype = _preprocess_precision_dtype()
    prefer_json_env = os.environ.get("CAPTION_PREFER_JSON")
    prefer_json = (
        _truthy(prefer_json_env)
        if prefer_json_env is not None
        else _truthy(row.get("prefer_json_caption"))
    )
    source_mode_env = os.environ.get("CAPTION_SOURCE_MODE")
    source_mode = (
        source_mode_env
        if source_mode_env is not None
        else row.get("caption_source_mode")
    )
    run(
        [
            PY,
            "-m",
            "scripts.z_image.preprocess_te_cache",
            "--dir",
            str(row.get("source_image_dir") or _path("source_image_dir", "image_dataset")),
            "--cache_dir",
            _text_cache_dir_for_row(row),
            "--qwen3",
            _path("qwen3", "models/diffusion_models/Z-Image"),
            "--batch_size",
            str(text_batch_size),
            "--dtype",
            dtype,
            "--caption_shuffle_variants",
            shuffle_variants,
            "--caption_tag_dropout_rate",
            tag_dropout_rate,
            *_caption_source_args(source_mode, prefer_json),
            *_caption_extension_args_for_row(row),
            *(["--prefer_json_caption"] if prefer_json else []),
            *_recursive_args(row),
            *_path_pattern_args(row),
            *mp_args,
            *_reuse_overwrite_args(row, kind="te"),
            *extra,
        ]
    )


def _run_preprocess_te_anima(
    row: dict[str, Any],
    extra: list[str],
    shuffle_variants: str | None = None,
    tag_dropout_rate: str | None = None,
) -> None:
    shuffle_variants = shuffle_variants or os.environ.get("CAPTION_SHUFFLE_VARIANTS", "4")
    tag_dropout_rate = tag_dropout_rate or os.environ.get("CAPTION_TAG_DROPOUT_RATE", "0.1")
    mp_args, extra = _resolve_lowres_filter(extra)
    prefer_json_env = os.environ.get("CAPTION_PREFER_JSON")
    prefer_json = (
        _truthy(prefer_json_env)
        if prefer_json_env is not None
        else _truthy(row.get("prefer_json_caption"))
    )
    source_mode_env = os.environ.get("CAPTION_SOURCE_MODE")
    caption_source_value = source_mode_env if source_mode_env is not None else row.get("caption_source_mode")
    json_args = ["--prefer_json_caption"] if prefer_json else []
    source_args = _caption_source_args(caption_source_value, prefer_json)
    _, text_batch_size = _preprocess_cache_batch_sizes()
    dtype = _preprocess_precision_dtype()
    run(
        [
            PY,
            "-m",
            "scripts.preprocess.cache_text_embeddings",
            "--dir",
            str(row.get("source_image_dir") or _path("source_image_dir", "image_dataset")),
            "--cache_dir",
            _text_cache_dir_for_row(row),
            "--qwen3",
            _path("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors"),
            "--dit",
            _path(
                "pretrained_model_name_or_path",
                "models/diffusion_models/anima-base-v1.0.safetensors",
            ),
            "--caption_shuffle_variants",
            shuffle_variants,
            "--caption_tag_dropout_rate",
            tag_dropout_rate,
            "--batch_size",
            str(text_batch_size),
            "--dtype",
            dtype,
            *_vocab_pack_args(),
            *source_args,
            *_caption_extension_args_for_row(row),
            *json_args,
            *_diff_output_preservation_args(extra),
            *_recursive_args(row),
            *_path_pattern_args(row),
            *mp_args,
            *_reuse_overwrite_args(row, kind="te"),
            *extra,
        ]
    )


def _text_cache_dir_for_row(row: dict[str, Any]) -> str:
    explicit = str(row.get("text_cache_dir") or "").strip()
    if explicit:
        return explicit
    return str(
        row.get("lora_cache_dir")
        or _path("lora_cache_dir", "post_image_dataset/lora")
    )


def cmd_preprocess_resize(extra):
    for row in _preprocess_rows():
        _run_preprocess_resize(row, extra)


def cmd_preprocess_vae(extra):
    from .qwen_edit_preprocess import uses_qwen_edit

    if uses_qwen_edit(_model_family(), _path_overrides_value()):
        raise ValueError("Qwen Image 2.1 Edit requires the full `preprocess` task; preprocess-vae is unsupported")
    for row in _preprocess_rows():
        _run_preprocess_vae(row, extra)


def cmd_preprocess_pooled(extra):
    """Cache pooled text embeddings (max over seq dim) from existing TE caches.

    Reads ``{stem}_anima_te.safetensors`` from the LoRA cache dir and writes
    ``{stem}_anima_pooled.safetensors`` sidecars next to them. Consumed by
    ``make distill-mod`` to skip a redundant ``.max(dim=1)`` per training
    microstep / val sigma. No GPU needed.
    """
    for row in _preprocess_rows():
        run(
            [
                PY,
                "-m",
                "scripts.preprocess.cache_pooled_text",
                "--dir",
                str(row.get("lora_cache_dir") or _path("lora_cache_dir", "post_image_dataset/lora")),
                *extra,
            ]
        )


def cmd_preprocess_pe(extra):
    """Cache PE-Core-L14-336 vision-encoder features.

    Reads pre-resized images from ``post_image_dataset/resized/`` (the
    standard LoRA pipeline source) and writes
    ``{stem}_anima_pe.safetensors`` sidecars into the LoRA cache dir so the
    dataset's existing ``cache_dir`` lookup finds them.

    Consumed by IP-Adapter when reading PE features off disk and by the
    DCW v4 fusion head's pooled-image-feature input channel.
    """
    for row in _preprocess_rows():
        run(
            [
                PY,
                "-m",
                "scripts.preprocess.cache_pe_encoder",
                "--dir",
                str(row.get("resized_image_dir") or _path("resized_image_dir", "post_image_dataset/resized")),
                "--cache_dir",
                str(row.get("lora_cache_dir") or _path("lora_cache_dir", "post_image_dataset/lora")),
                "--encoder",
                "pe",
                *_recursive_args(row),
                *_path_pattern_args(row),
                *extra,
            ]
        )


def cmd_caption_index(extra):
    """Build the method-agnostic typed-tag caption index."""
    run(
        [
            PY,
            "-m",
            "scripts.preprocess.build_caption_index",
            "--src",
            _path("source_image_dir", "image_dataset"),
            *extra,
        ]
    )


def _build_caption_index_best_effort() -> None:
    """Build caption-index when its optional vocab is already available."""
    vocab = _path("caption_index_vocab", _CAPTION_INDEX_VOCAB)
    if not os.path.exists(vocab):
        print(
            "  [preprocess] skipping caption-index: tagger vocab not found at "
            f"{vocab}. Run `python tasks.py download-tagger`, then "
            "`python tasks.py caption-index` if the selected method needs it."
        )
        return

    try:
        cmd_caption_index([])
    except (SystemExit, OSError, Exception) as e:  # noqa: BLE001
        print(f"  [preprocess] caption-index build failed: {e}")


def cmd_preprocess(extra):
    # PE features are intentionally NOT cached here — only IP-Adapter / CMMD /
    # DCW v4 need them, and those paths chain `preprocess-pe` explicitly (see
    # `exp-ip-adapter-preprocess`). Leaving PE out keeps the default LoRA
    # preprocess fast on machines that won't ever use the vision tower.
    non_te_extra, explicit_dop_args = _split_diff_output_preservation_args(list(extra))
    from .qwen_edit_preprocess import (
        run_edit_preprocess, snapshot_edit_captions, split_edit_options, uses_qwen_edit,
    )

    overrides = _path_overrides_value()
    if uses_qwen_edit(_model_family(), overrides):
        rows = _preprocess_rows()
        resize_extra, _ = split_edit_options(non_te_extra)
        rebuild = {kind: any(_reuse_overwrite_args({**overrides, **row}, kind=kind) for row in rows)
                   for kind in ("vae", "te")}
        if rebuild["vae"] != rebuild["te"] and "--overwrite" not in extra:
            raise ValueError("Qwen Image 2.1 Edit cannot rebuild only VAE or TE caches; use `preprocess --overwrite` to rebuild both")
        for row in rows:
            _run_caption_backup(row)
            _run_preprocess_resize(row, resize_extra, copy_captions=True)
            snapshot_edit_captions(
                row["source_image_dir"], row["resized_image_dir"], _caption_extension_for_row(row),
            )
        run_edit_preprocess(
            overrides, run=run, python=PY, path=_path,
            dtype=_preprocess_precision_dtype(),
            extra=[*extra, *(["--overwrite"] if rebuild["vae"] else [])],
        )
        return
    _, vae_extra = _resolve_lowres_filter(non_te_extra)
    for row in _preprocess_rows():
        _run_caption_backup(row)
        _run_preprocess_resize(row, non_te_extra)
        _run_preprocess_vae(row, vae_extra)
        _run_preprocess_te(row, explicit_dop_args, backup_captions=False)
    _build_caption_index_best_effort()


def cmd_preprocess_config(extra):
    """Preprocess the exact directories named in a ``--dataset_config`` TOML."""
    import time

    args = list(extra)
    cfg_path: str | None = None
    src_dir: str | None = None
    vae_path = _path("vae", "models/vae/qwen_image_vae.safetensors")
    qwen3_path = _path("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors")
    dit_path = _path(
        "pretrained_model_name_or_path",
        "models/diffusion_models/anima-base-v1.0.safetensors",
    )
    explicit_dop_args: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--dataset_config" and i + 1 < len(args):
            cfg_path = args[i + 1]
            i += 2
        elif args[i] == "--src" and i + 1 < len(args):
            src_dir = args[i + 1]
            i += 2
        elif args[i] == "--vae" and i + 1 < len(args):
            vae_path = args[i + 1]
            i += 2
        elif args[i] == "--qwen3" and i + 1 < len(args):
            qwen3_path = args[i + 1]
            i += 2
        elif args[i] == "--dit" and i + 1 < len(args):
            dit_path = args[i + 1]
            i += 2
        elif args[i] in (
            "--diff_output_preservation_trigger",
            "--diff_output_preservation_class",
        ) and i + 1 < len(args):
            explicit_dop_args.extend([args[i], args[i + 1]])
            i += 2
        elif any(args[i].startswith(f"{flag}=") for flag in _DIFF_OUTPUT_PRESERVATION_FLAGS):
            explicit_dop_args.append(args[i])
            i += 1
        else:
            rest.append(args[i])
            i += 1
    if not cfg_path or not src_dir:
        raise SystemExit("preprocess-config requires --dataset_config <path> and --src <dir>")
    vae_batch_size, text_batch_size = _preprocess_cache_batch_sizes()
    dtype = "float32" if _model_family() == "moss_image" else _preprocess_precision_dtype()

    last_err: OSError | None = None
    for attempt in range(10):
        try:
            cfg = _load_toml_file(Path(cfg_path))
            break
        except PermissionError as e:
            last_err = e
            time.sleep(0.2 * (attempt + 1))
    else:
        raise SystemExit(
            f"could not read {cfg_path} after retrying (last error: {last_err}). "
            "If this persists, exclude the dataset/temp dir from your antivirus."
        )

    general = cfg.get("general") if isinstance(cfg.get("general"), dict) else {}
    general_prefer_json = _optional_bool(general.get("prefer_json_caption", general.get("prefer_json")))
    general_caption_source_mode = str(general.get("caption_source_mode") or "").strip()
    general_caption_extension = str(general.get("caption_extension") or ".txt").strip() or ".txt"

    subset_entries: list[dict[str, Any]] = []
    for dataset in cfg.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        for subset in dataset.get("subsets") or []:
            if not isinstance(subset, dict) or not subset.get("image_dir"):
                continue
            prefer_json = _optional_bool(
                dataset.get(
                    "prefer_json_caption",
                    subset.get("prefer_json_caption"),
                )
            )
            if prefer_json is None:
                prefer_json = general_prefer_json
            prefer_json = bool(prefer_json)
            caption_source_mode = _first_explicit_caption_source_mode(
                dataset.get("caption_source_mode"),
                subset.get("caption_source_mode"),
                general_caption_source_mode,
            )
            caption_extension = str(
                dataset.get(
                    "caption_extension",
                    subset.get("caption_extension", general_caption_extension),
                )
                or ".txt"
            ).strip() or ".txt"
            subset_entries.append({
                "subset": subset,
                "prefer_json": prefer_json,
                "caption_source_mode": caption_source_mode,
                "caption_extension": caption_extension,
                "recursive": subset.get("recursive", dataset.get("recursive", True)),
                "path_pattern": subset.get("path_pattern", dataset.get("path_pattern", "*")),
            })
    if not subset_entries:
        raise SystemExit(f"no [[datasets.subsets]] with image_dir in {cfg_path}")

    for entry in subset_entries:
        sub = entry["subset"]
        image_dir = sub["image_dir"]
        latent_cache_dir = str(sub.get("cache_dir") or "").strip()
        text_cache_dir = str(sub.get("text_cache_dir") or "").strip()
        if not text_cache_dir:
            # TE walks the caption-bearing source tree. When no dedicated
            # cache root exists, redirect its sidecars to the training image
            # tree, matching the loader's alongside-image lookup.
            text_cache_dir = latent_cache_dir or str(image_dir)
        prefer_json = bool(entry["prefer_json"])
        json_args = ["--prefer_json_caption"] if prefer_json else []
        source_args = _caption_source_args(entry.get("caption_source_mode"), prefer_json)
        caption_extension_args = ["--caption_extension", str(entry.get("caption_extension") or ".txt")]
        filter_row = {
            "recursive": entry.get("recursive", True),
            "path_pattern": entry.get("path_pattern", "*"),
        }
        run(
            [
                PY,
                "-m",
                "scripts.preprocess.resize_images",
                "--src",
                src_dir,
                "--dst",
                image_dir,
                "--no_copy_captions",
                "--min_pixels",
                "0",
                "--bucket_reso_steps",
                "64",
                *_recursive_args(filter_row),
                *_path_pattern_args(filter_row),
                *rest,
            ]
        )
        run(
            [
                PY,
                "-m",
                "scripts.preprocess.cache_latents",
                "--dir",
                image_dir,
                *(["--cache_dir", latent_cache_dir] if latent_cache_dir else []),
                "--vae",
                vae_path,
                *(["--model_family", "moss_image"] if _model_family() == "moss_image" else []),
                "--batch_size",
                str(vae_batch_size),
                "--dtype",
                dtype,
                "--chunk_size",
                "64",
                *_recursive_args(filter_row),
                *_path_pattern_args(filter_row),
            ]
        )
        if _model_family() == "moss_image":
            _run_preprocess_te_moss_image(
                {**filter_row, "source_image_dir": src_dir, "text_cache_dir": text_cache_dir,
                 "prefer_json_caption": prefer_json,
                 "caption_source_mode": entry.get("caption_source_mode"),
                 "caption_extension": entry.get("caption_extension")},
                ["--qwen3", qwen3_path, "--min_pixels", "0"],
            )
            continue
        run(
            [
                PY,
                "-m",
                "scripts.preprocess.cache_text_embeddings",
                "--dir",
                src_dir,
                "--cache_dir",
                text_cache_dir,
                "--qwen3",
                qwen3_path,
                "--dit",
                dit_path,
                "--batch_size",
                str(text_batch_size),
                "--dtype",
                dtype,
                *source_args,
                *caption_extension_args,
                *json_args,
                *_diff_output_preservation_args(explicit_dop_args),
                *_recursive_args(filter_row),
                *_path_pattern_args(filter_row),
            ]
        )
