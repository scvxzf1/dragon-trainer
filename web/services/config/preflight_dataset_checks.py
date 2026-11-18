"""Dataset / image / cache sidecar checks for training preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from web.services.config.cache_audit import audit_dataset_row_caches
from library.preprocess.captions import normalize_caption_source_mode, read_caption_source_from_dirs
from web.services.config.metadata import DATASET_IMAGE_EXTS
from web.services.config.preflight_runtime import (
    _bool_value,
    _caption_detection_counts_text,
    _count_source_images,
    _dataset_image_files,
    _dataset_rows_for_estimate,
    _nl_tag_mix_caption_counts,
    _nl_tag_mix_enabled,
    _normalize_path_pattern,
    _normalize_trigger_clone,
    _resolve_project_path,
)

def _check_training_images(
    cfg: dict[str, Any],
    add,
    *,
    dataset_rows: list[dict[str, Any]] | None = None,
) -> None:
    rows = _dataset_rows_for_estimate(cfg) if dataset_rows is None else dataset_rows
    if not rows:
        rows = [{
            "source_dir": str(cfg.get("source_image_dir") or ""),
            "image_dir": str(cfg.get("resized_image_dir") or cfg.get("source_image_dir") or ""),
        }]
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    all_missing_captions: list[str] = []
    detected_caption_modes: dict[str, int] = {}
    detected_caption_total = 0
    checked_groups = 0
    for idx, row in enumerate(rows, start=1):
        image_dir = _resolve_project_path(str(row.get("image_dir") or row.get("source_dir") or ""))
        source_dir = _resolve_project_path(str(row.get("source_dir") or ""))
        settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
        recursive = _bool_value(row.get("recursive"), True)
        path_pattern = _normalize_path_pattern(row.get("path_pattern"))
        caption_extension = str(settings.get("caption_extension") or cfg.get("caption_extension") or ".txt")
        if not caption_extension.startswith("."):
            caption_extension = f".{caption_extension}"
        prefer_json_caption = _bool_value(
            settings.get("prefer_json_caption", cfg.get("prefer_json_caption")),
            False,
        )
        caption_source_mode = normalize_caption_source_mode(
            settings.get("caption_source_mode", cfg.get("caption_source_mode")),
            prefer_json_caption,
        )
        if not image_dir.is_dir():
            continue
        checked_groups += 1
        key = "training_images" if idx == 1 else f"dataset_{idx}_training_images"
        label = "缩放图像目录" if idx == 1 else f"第 {idx} 组缩放图像目录"
        try:
            images = _dataset_image_files(
                image_dir,
                image_exts,
                recursive=recursive,
                path_pattern=path_pattern,
            )
        except ValueError as exc:
            add("error", key, str(exc), image_dir)
            continue
        if not images:
            add("error", key, f"{label}里没有可训练图片，请先预处理生成训练图", image_dir)
            continue
        for image in images[:50]:
            source = read_caption_source_from_dirs(
                image,
                [source_dir, image.parent],
                prefer_json_caption=prefer_json_caption,
                caption_source_mode=caption_source_mode,
                caption_extension=caption_extension,
            )
            if source.path is None:
                all_missing_captions.append(image.name)
            else:
                detected_caption_modes[source.detected_mode] = (
                    detected_caption_modes.get(source.detected_mode, 0) + 1
                )
                detected_caption_total += len(source.caption_texts())
    if checked_groups == 0:
        return
    if all_missing_captions:
        sample = ", ".join(all_missing_captions[:3])
        add("warning", "captions", f"部分图片未找到同名标注，例如 {sample}")
    else:
        summary = _caption_detection_counts_text(detected_caption_modes, detected_caption_total)
        add("ok", "captions", f"抽样图片均找到标注；{summary}" if summary else "抽样图片均找到标注")

def _check_dataset_source_paths(
    cfg: dict[str, Any],
    add,
    *,
    dataset_rows: list[dict[str, Any]] | None = None,
) -> None:
    rows = _dataset_rows_for_estimate(cfg) if dataset_rows is None else dataset_rows
    if not rows:
        return
    for idx, row in enumerate(rows, start=1):
        source = _resolve_project_path(str(row.get("source_dir") or ""))
        key = "source_image_dir" if idx == 1 else f"dataset_{idx}_source_dir"
        label = "源图像目录" if idx == 1 else f"第 {idx} 组原始数据集目录"
        recursive = _bool_value(row.get("recursive"), True)
        path_pattern = _normalize_path_pattern(row.get("path_pattern"))
        trigger_clone = _normalize_trigger_clone(row.get("trigger_clone"))
        if trigger_clone["enabled"] and not trigger_clone["prompt"]:
            add(
                "error",
                f"{key}_trigger_clone_prompt",
                f"{label} 的触发提示词图像克隆已开启，但触发提示词为空",
                source,
            )
        if not str(row.get("source_dir") or "").strip():
            add("error", key, f"{label} 未填写")
        elif not source.exists():
            add("error", key, f"{label} 不存在", source)
        elif not source.is_dir():
            add("error", key, f"{label} 不是目录", source)
        elif _count_source_images(
            source,
            DATASET_IMAGE_EXTS,
            recursive=recursive,
            path_pattern=path_pattern,
        ) <= 0:
            # Directory may still contain only caches (.npz/.safetensors). That is not a source dataset.
            if trigger_clone["enabled"]:
                add("error", f"{key}_trigger_clone_images", f"{label} 中没有可克隆的训练图片", source)
            else:
                add(
                    "error",
                    f"{key}_images",
                    f"{label} 中没有可训练图片（常见原因：指到了缓存目录，或只有 .npz/.safetensors）",
                    source,
                )
        elif _nl_tag_mix_enabled(row):
            settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
            caption_extension = str(settings.get("caption_extension") or cfg.get("caption_extension") or ".txt")
            if not caption_extension.startswith("."):
                caption_extension = f".{caption_extension}"
            prefer_json_caption = _bool_value(
                settings.get("prefer_json_caption", cfg.get("prefer_json_caption")),
                False,
            )
            caption_source_mode = normalize_caption_source_mode(
                settings.get("caption_source_mode", cfg.get("caption_source_mode")),
                prefer_json_caption,
            )
            image_count, captioned_count = _nl_tag_mix_caption_counts(
                source,
                caption_source_mode=caption_source_mode,
                caption_extension=caption_extension,
                prefer_json_caption=prefer_json_caption,
                recursive=recursive,
                path_pattern=path_pattern,
            )
            if image_count <= 0:
                add("error", f"{key}_nl_tag_mix", f"{label} 中没有可训练图片", source)
            else:
                if captioned_count <= 0:
                    add(
                        "warning",
                        f"{key}_nl_tag_mix_captions",
                        f"{label} 未找到可读取标注，captions格式nl/tag权重调整会全部按 tag 处理",
                        source,
                    )
                add("ok", key, f"{label} 存在", source)
        elif not any(source.iterdir()):
            add("warning", key, f"{label} 为空", source)
        else:
            add("ok", key, f"{label} 存在", source)

def _check_dataset_paths(
    cfg: dict[str, Any],
    add,
    *,
    check_runtime_dirs: bool = True,
    dataset_rows: list[dict[str, Any]] | None = None,
) -> None:
    if not check_runtime_dirs:
        return
    rows = _dataset_rows_for_estimate(cfg) if dataset_rows is None else dataset_rows
    for idx, row in enumerate(rows, start=1):
        image_dir = _resolve_project_path(str(row.get("image_dir") or ""))
        prefix = f"dataset_{idx}"
        if not image_dir.exists():
            add("error", f"{prefix}_image_dir", f"第 {idx} 组缩放图路径不存在", image_dir)
        elif not image_dir.is_dir():
            add("error", f"{prefix}_image_dir", f"第 {idx} 组缩放图路径不是目录", image_dir)
        raw_cache_dir = str(row.get("cache_dir") or "").strip()
        if raw_cache_dir:
            cache_dir = _resolve_project_path(raw_cache_dir)
            if not cache_dir.exists():
                add("error", f"{prefix}_cache_dir", f"第 {idx} 组缓存路径不存在", cache_dir)
            elif not cache_dir.is_dir():
                add("error", f"{prefix}_cache_dir", f"第 {idx} 组缓存路径不是目录", cache_dir)

def _check_cache_sidecars(
    cfg: dict[str, Any],
    add,
    *,
    dataset_rows: list[dict[str, Any]] | None = None,
) -> None:
    rows = _dataset_rows_for_estimate(cfg) if dataset_rows is None else dataset_rows
    for idx, row in enumerate(rows, start=1):
        image_dir = _resolve_project_path(
            str(row.get("image_dir") or row.get("source_dir") or "")
        )
        if not image_dir.is_dir():
            continue
        try:
            images = _dataset_image_files(
                image_dir,
                DATASET_IMAGE_EXTS,
                recursive=_bool_value(row.get("recursive"), True),
                path_pattern=_normalize_path_pattern(row.get("path_pattern")),
            )
            if not images:
                continue
            results = audit_dataset_row_caches(
                cfg,
                row,
                images,
                resolve_path=_resolve_project_path,
            )
        except ValueError as exc:
            add("error", f"dataset_{idx}_cache_semantics", str(exc), image_dir)
            continue
        for result in results:
            key = result["kind"] if idx == 1 else f"dataset_{idx}_{result['kind']}"
            valid = result["valid"]
            total = result["total"]
            missing = result["missing"]
            invalid = result["invalid"]
            if not missing and not invalid:
                add(
                    "ok",
                    key,
                    f"第 {idx} 组{result['label']}完整：{valid}/{total}",
                    result["root"],
                )
                continue
            samples = [Path(path).name for path in [*missing, *invalid][:3]]
            details = []
            if missing:
                details.append(f"缺失 {len(missing)}")
            if invalid:
                details.append(f"无效 {len(invalid)}")
            sample_text = f"；例如 {', '.join(samples)}" if samples else ""
            level = "error" if invalid else result["missing_level"]
            if result["kind"] == "external_masks":
                suffix = "；缺失遮罩的图片会按全 1 mask 训练"
            elif result["kind"] == "condition_latent_cache":
                suffix = "；通用预处理不会生成条件 latent，请先运行当前方法的 condition prep"
            else:
                suffix = "；runtime 训练不会临时补缓存，请重新预处理"
            add(
                level,
                key,
                f"第 {idx} 组{result['label']}不完整：{valid}/{total}（{'，'.join(details)}）{sample_text}{suffix}",
                result["root"],
            )

def _check_cache_sidecar_pattern(
    add,
    cache_dirs: list[tuple[int, Path, bool]],
    pattern: str,
    key: str,
    label: str,
    missing_message: str,
) -> None:
    for idx, cache_dir, recursive in cache_dirs:
        matches = cache_dir.rglob(pattern) if recursive else cache_dir.glob(pattern)
        count = sum(1 for path in matches if path.is_file())
        item_key = key if idx == 1 else f"dataset_{idx}_{key}"
        if count:
            add("ok", item_key, f"第 {idx} 组找到 {count} 个{label}", cache_dir)
        else:
            add("warning", item_key, f"第 {idx} 组{missing_message}", cache_dir)


def _check_dataset_bucket_settings(
    cfg: dict[str, Any],
    add,
    *,
    dataset_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Catch impossible preprocess bucket params before launch.

    BucketManager asserts max_bucket_reso >= resolution. Leaving the UI default
    max_bucket_reso=1024 while raising resolution (e.g. 1536) used to hard-crash
    resize mid-preprocess.
    """
    rows = _dataset_rows_for_estimate(cfg) if dataset_rows is None else dataset_rows
    if not rows:
        return
    for idx, row in enumerate(rows, start=1):
        settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
        if not settings:
            continue
        try:
            resolution = int(settings.get("resolution") or 0)
        except (TypeError, ValueError):
            resolution = 0
        try:
            min_bucket = int(settings.get("min_bucket_reso") or 0)
        except (TypeError, ValueError):
            min_bucket = 0
        try:
            max_bucket = int(settings.get("max_bucket_reso") or 0)
        except (TypeError, ValueError):
            max_bucket = 0
        if resolution <= 0:
            continue
        key = "dataset_bucket" if idx == 1 else f"dataset_{idx}_bucket"
        label = "数据集" if idx == 1 else f"第 {idx} 组数据集"
        if max_bucket > 0 and max_bucket < resolution:
            add(
                "error",
                key,
                (
                    f"{label} 的 max_bucket_reso={max_bucket} 小于 resolution={resolution}；"
                    "预处理会失败。请把 max_bucket_reso 调到 ≥ resolution（通常直接填 resolution）"
                ),
            )
        if min_bucket > 0 and max_bucket > 0 and max_bucket < min_bucket:
            add(
                "error",
                f"{key}_minmax",
                f"{label} 的 max_bucket_reso={max_bucket} 小于 min_bucket_reso={min_bucket}",
            )
