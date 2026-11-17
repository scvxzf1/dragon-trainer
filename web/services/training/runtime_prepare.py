"""Runtime preparation helpers for WebUI training runs."""

from __future__ import annotations

import shutil
from typing import Any

from library.preprocess.captions import CAPTION_SOURCE_CAPTIONS_JSON
from web.services.training.runtime_common import (
    _build_runtime_payload,
    _build_dataset_config_doc,
    _dataset_rows_for_estimate,
    _default_preprocess_precision_preference,
    _ensure_runtime_dir_layout,
    _load_config_file_config,
    _normalize_path_pattern,
    _normalize_trigger_clone,
    _sample_config_from_cfg,
    apply_auto_data_dirs,
    apply_global_model_path_defaults,
    load_merged_config,
    toml_dumps_sorted,
)
from web.services.training.runtime_paths import (
    _display_project_path,
    _display_settings_path,
    _path_exists,
    _resolve_display_path,
    _safe_run_stem,
    _unique_runtime_dir,
    resolve_output_root,
)
from web.services.training import runtime_datasets as _runtime_datasets
from web.services.training.runtime_state import _write_runtime_run_meta
from web.services.config.preflight_stage_schedule import validate_stage_schedule_or_raise
from web.services.config.dataset_rows import merge_stage_schedule_from_dataset_config
from library.training.stage_schedule import STAGE_TARGET_GROUPS_KEY

_bool_value_for_row = _runtime_datasets._bool_value_for_row
_prepare_runtime_nl_tag_mix_source = _runtime_datasets._prepare_runtime_nl_tag_mix_source
_prepare_runtime_trigger_clone_source = _runtime_datasets._prepare_runtime_trigger_clone_source
_bind_subset_to_cache_pool = _runtime_datasets._bind_subset_to_cache_pool
_runtime_row_semantics = _runtime_datasets._runtime_row_semantics


def _resolve_training_runtime_info(
    variant: str,
    preset: str,
    methods_subdir: str,
    extra_args: list[str],
    config_file: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    output_dir = "output/ckpt"
    cfg: dict[str, Any] = {}
    try:
        if config_file:
            cfg = _load_config_file_config(config_file)
        else:
            cfg = load_merged_config(variant, preset, methods_subdir)
        output_dir = str(cfg.get("output_dir") or output_dir)
    except Exception:
        pass

    for idx, arg in enumerate(extra_args):
        if arg == "--output_dir" and idx + 1 < len(extra_args):
            output_dir = str(extra_args[idx + 1])
            break
        if arg.startswith("--output_dir="):
            output_dir = arg.split("=", 1)[1]
            break

    rel_output = _display_project_path(output_dir) or "output/ckpt"
    return rel_output, f"{rel_output.rstrip('/')}/sample", _sample_config_from_cfg(cfg, extra_args)


def _ensure_training_data_dirs(
    variant: str,
    preset: str,
    methods_subdir: str,
    *,
    config_file: str | None = None,
) -> dict[str, str]:
    if config_file:
        cfg = apply_auto_data_dirs(_load_config_file_config(config_file), create=True)
    else:
        cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir), create=True)
    return {
        "source_image_dir": str(cfg.get("source_image_dir") or ""),
        "resized_image_dir": str(cfg.get("resized_image_dir") or ""),
        "lora_cache_dir": str(cfg.get("lora_cache_dir") or ""),
    }


def _prepare_web_runtime_config(
    variant: str,
    preset: str,
    methods_subdir: str,
    *,
    source_config_file: str | None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    source_path = _resolve_display_path(source_config_file or "") if source_config_file else None
    if source_config_file and (source_path is None or not _path_exists(source_path) or not source_path.is_file()):
        raise FileNotFoundError(f"训练配置不存在: {source_config_file}")

    stem_source = source_path.stem if source_path is not None else variant
    run_stem = _safe_run_stem(stem_source or variant or "run")

    fallback_cfg = load_merged_config(variant, preset, methods_subdir)
    cfg = dict(fallback_cfg)
    if source_path is not None:
        source_cfg = _load_config_file_config(_display_settings_path(source_path))
        if source_cfg:
            cfg.update(source_cfg)
    cfg = apply_global_model_path_defaults(cfg, fallback=fallback_cfg)
    from web.services.training.extra_args import merge_training_extra_args

    cfg = merge_training_extra_args(cfg, extra_args)
    cfg = merge_stage_schedule_from_dataset_config(cfg)
    source_rows = _dataset_rows_for_estimate(cfg)
    if not source_rows:
        raise ValueError("请先配置至少一个数据集路径")

    validate_stage_schedule_or_raise(cfg, dataset_rows=source_rows)
    run_dir = _unique_runtime_dir(resolve_output_root(), run_stem)
    layout = _ensure_runtime_dir_layout(run_dir)

    runtime_rows: list[dict[str, Any]] = []
    stage_target_groups: list[list[int]] = []
    dataset_cache_bindings: list[dict[str, Any]] = []
    dataset_cache_dir = layout["dataset_cache_dir"]
    from library.cache_pool.store import default_pool_root
    pool_root = default_pool_root()
    run_id = run_dir.name
    for index, row in enumerate(source_rows, start=1):
        stage_members = [len(runtime_rows)]
        group_dir = dataset_cache_dir / f"dataset-{index:02d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        source_dir = str(
            row.get("source_dir")
            or row.get("source_image_dir")
            or row.get("image_dir")
            or ""
        ).strip()
        source_dir = _prepare_runtime_nl_tag_mix_source(row, group_dir, source_dir)
        binding = _bind_subset_to_cache_pool(
            cfg=cfg,
            row=row,
            group_dir=group_dir,
            pool_root=pool_root,
            run_id=run_id,
            source_dir=source_dir,
            resized_name="resized",
            lora_name="lora",
        )
        dataset_cache_bindings.append({**binding, "subset_index": index, "kind": "primary"})
        runtime_rows.append({
            "source_dir": source_dir,
            "image_dir": binding["image_dir"],
            "cache_dir": binding["cache_dir"],
            "num_repeats": row.get("num_repeats") or 1,
            "is_reg": _bool_value_for_row(row.get("is_reg"), False),
            "recursive": _bool_value_for_row(row.get("recursive"), True),
            "path_pattern": _normalize_path_pattern(row.get("path_pattern")),
            "settings": row.get("settings") if isinstance(row.get("settings"), dict) else {},
            **_runtime_row_semantics(row),
        })
        trigger_clone = _normalize_trigger_clone(row.get("trigger_clone"))
        if trigger_clone["enabled"]:
            clone_source_dir = _prepare_runtime_trigger_clone_source(row, group_dir, source_dir)
            clone_settings = dict(row.get("settings") if isinstance(row.get("settings"), dict) else {})
            clone_settings["caption_source_mode"] = CAPTION_SOURCE_CAPTIONS_JSON
            clone_settings["prefer_json_caption"] = False
            clone_row = dict(row)
            clone_row["settings"] = clone_settings
            clone_binding = _bind_subset_to_cache_pool(
                cfg=cfg,
                row=clone_row,
                group_dir=group_dir,
                pool_root=pool_root,
                run_id=f"{run_id}-clone-{index}",
                source_dir=clone_source_dir,
                resized_name="trigger-clone-resized",
                lora_name="trigger-clone-lora",
            )
            dataset_cache_bindings.append({**clone_binding, "subset_index": index, "kind": "trigger_clone"})
            runtime_rows.append({
                "source_dir": clone_source_dir,
                "image_dir": clone_binding["image_dir"],
                "cache_dir": clone_binding["cache_dir"],
                "num_repeats": trigger_clone["num_repeats"],
                "is_reg": _bool_value_for_row(row.get("is_reg"), False),
                "recursive": _bool_value_for_row(row.get("recursive"), True),
                "path_pattern": _normalize_path_pattern(row.get("path_pattern")),
                "settings": clone_settings,
                **_runtime_row_semantics(clone_row),
            })
            stage_members.append(len(runtime_rows) - 1)
        stage_target_groups.append(stage_members)

    original_config_path = run_dir / "config.original.toml"
    if source_path is not None:
        shutil.copy2(source_path, original_config_path)
    else:
        original_config_path.write_text(toml_dumps_sorted(cfg), encoding="utf-8")

    dataset_config_path = run_dir / "dataset.runtime.toml"
    runtime_cfg = dict(cfg)
    if bool(runtime_cfg.get("stage_schedule_enabled")):
        runtime_cfg[STAGE_TARGET_GROUPS_KEY] = stage_target_groups
    dataset_doc = _build_dataset_config_doc(
        runtime_rows,
        runtime_cfg,
        prefer_train_batch_size=True,
        include_preprocess_settings=False,
    )
    dataset_config_path.write_text(dataset_doc, encoding="utf-8")

    first_row = runtime_rows[0]
    data_dirs = {
        "source_image_dir": first_row["source_dir"],
        "resized_image_dir": first_row["image_dir"],
        "lora_cache_dir": first_row["cache_dir"],
    }
    runtime_cfg.update({
        "output_dir": _display_settings_path(layout["training_output_dir"]),
        "logging_dir": _display_settings_path(layout["logs_dir"]),
        "dataset_config": _display_settings_path(dataset_config_path),
        "source_image_dir": first_row["source_dir"],
        "resized_image_dir": first_row["image_dir"],
        "lora_cache_dir": first_row["cache_dir"],
    })
    runtime_cfg["preprocess_precision_preference"] = _default_preprocess_precision_preference(runtime_cfg)
    runtime_config_path = run_dir / "config.runtime.toml"
    runtime_config_path.write_text(toml_dumps_sorted(runtime_cfg), encoding="utf-8")

    history_source_config_file = _display_project_path(str(source_path)) if source_path is not None else ""
    _write_runtime_run_meta(
        run_dir,
        {
            "history_source_config_file": history_source_config_file,
            "source_config_file": history_source_config_file,
            "run_dir": _display_settings_path(run_dir),
            "runtime_config_file": _display_settings_path(runtime_config_path),
            "original_config_file": _display_settings_path(original_config_path),
            "dataset_config_file": _display_settings_path(dataset_config_path),
            "cache_pool_root": _display_settings_path(pool_root),
            "dataset_cache_bindings": dataset_cache_bindings,
        },
    )
    return _build_runtime_payload(
        run_dir=run_dir,
        layout=layout,
        runtime_config_file=runtime_config_path,
        original_config_file=original_config_path,
        dataset_config_file=dataset_config_path,
        output_dir=runtime_cfg["output_dir"],
        logs_dir=runtime_cfg["logging_dir"],
        history_source_config_file=history_source_config_file,
        data_dirs=data_dirs,
        dataset_dirs=runtime_rows,
        sample_config=_sample_config_from_cfg(runtime_cfg, []),
    )
