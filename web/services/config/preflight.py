"""Training preflight checks and runtime config path validation.

Compatibility facade. Implementation lives in:

- ``preflight_runtime``: roots, facade sync, path wrappers
- ``preflight_paths``: config path validation / web runtime detection
- ``preflight_compat``: checkpoint / network / sample / preprocess env checks
- ``preflight_history``: history output-dir reuse checks
- ``preflight_dataset_checks``: dataset image / source / cache checks

This module is loaded by ``web.services.config_service`` as part of the
compatibility facade.  It keeps facade access lazy so the module can also be
imported directly without pulling in the facade cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from library.models.family_registry import normalize_registered_family
from web.services.config.preflight_tasks import task_check_config, training_task_summary

from web.services.config.preflight_compat import (
    _check_checkpointing_config,
    _check_core_training_semantics,
    _check_network_weights,
    _check_no_dataset_regularization_config,
    _check_training_sample_config,
    _check_web_preprocess_environment,
    _compat_web_message,
    _inspect_network_weight,
    _inspect_network_weight_impl,
    _web_python_executable,
    training_sample_sampler_status,
)
from web.services.config.preflight_nf4 import check_mixed_nf4
from web.services.config.preflight_stage_schedule import check_stage_schedule
from web.services.config.dataset_rows import merge_stage_schedule_from_dataset_config
from web.services.config.schema_gate import validate_config_mapping
from web.services.config.preflight_dataset_checks import (
    _check_cache_sidecar_pattern,
    _check_cache_sidecars,
    _check_dataset_bucket_settings,
    _check_dataset_paths,
    _check_dataset_source_paths,
    _check_training_images,
)
from web.services.config.preflight_history import (
    _check_output_dir_history_reuse,
    _history_output_match_label,
    _history_task_output_candidates,
    _history_task_reuses_output_dir,
    _history_training_tasks_for_output_dir,
    _is_web_runtime_training_output_dir,
    _read_history_meta_for_output_reuse,
)
from web.services.config.preflight_paths import (
    _blank_model_path,
    _config_file_path,
    _config_path_from_display_path,
    _global_model_path_defaults,
    _has_web_runtime_dirs,
    _is_allowed_training_config_path,
    _is_output_run_snapshot_config,
    _is_web_runtime_config_tree,
    _load_training_config_for_web_run,
    _looks_like_web_runtime_config,
    apply_global_model_path_defaults,
    is_web_runtime_config,
)
from web.services.config.preflight_runtime import (
    CONFIGS_DIR,
    DATASET_PRESETS_DIR,
    GLOBAL_MODEL_PATH_KEYS,
    GUI_METHODS_DIR,
    IMPORTED_CONFIGS_DIR,
    PRESETS_FILE,
    ROOT,
    WEB_FILE_GROUPS_FILE,
    WEB_USER_LOCKS_FILE,
    _display_path,
    _exported,
    _missing_facade_dependency,
    _nonnegative_float_value,
    _nonnegative_int_value,
    _normalize_config_rel_path,
    _resolve_project_path,
    _safe_resolve,
    _sync_from_facade,
)

# Keep facade-filled placeholders re-exported for monkeypatch / legacy sync.
from web.services.config.preflight_runtime import (  # noqa: F401
    LOGGER,
    _bool_value,
    _caption_detection_counts_text,
    _count_source_images,
    _dataset_config_path_from_cfg,
    _dataset_image_files,
    _dataset_rows_for_estimate,
    _display_settings_path,
    _is_blank_output_name,
    _nl_tag_mix_caption_counts,
    _nl_tag_mix_enabled,
    _normalize_path_pattern,
    _normalize_trigger_clone,
    _positive_int_or_none,
    apply_auto_data_dirs,
    delete_raw_file,
    get_config_file_meta,
    list_config_file_groups,
    load_merged_config,
    load_raw_file,
    move_config_file_to_group,
    patch_raw_file_values,
    preview_raw_file_patch,
    resolve_output_root,
    save_raw_file,
)


__all__ = [
    "preflight_training_config",
    "_load_training_config_for_web_run",
    "_config_file_path",
    "is_web_runtime_config",
    "training_sample_sampler_status",
    "apply_global_model_path_defaults",
    "_check_training_images",
    "_check_dataset_source_paths",
    "_check_dataset_bucket_settings",
    "_check_dataset_paths",
    "_check_cache_sidecars",
]


def _validate_lokr_config(cfg: dict) -> list[str]:
    """Extra LoKR rules that are not pure argparse choices."""
    errors: list[str] = []
    use_lokr = cfg.get("use_lokr")
    if isinstance(use_lokr, str):
        use_lokr = use_lokr.strip().lower() in {"1", "true", "yes", "on"}
    if not use_lokr:
        return errors
    full_factor = cfg.get("lokr_full_factor")
    if isinstance(full_factor, str):
        full_factor = full_factor.strip().lower() in {"1", "true", "yes", "on"}
    decompose_w2 = cfg.get("lokr_decompose_w2")
    if isinstance(decompose_w2, str):
        decompose_w2 = decompose_w2.strip().lower() in {"1", "true", "yes", "on"}
    if full_factor and decompose_w2:
        errors.append("lokr_full_factor=true conflicts with lokr_decompose_w2=true")
    network_dim = cfg.get("network_dim")
    try:
        network_dim_i = int(network_dim) if network_dim is not None else None
    except (TypeError, ValueError):
        network_dim_i = None
    allow_legacy = cfg.get("lokr_allow_legacy_dim")
    if isinstance(allow_legacy, str):
        allow_legacy = allow_legacy.strip().lower() in {"1", "true", "yes", "on"}
    if network_dim_i == 114514 and not allow_legacy:
        errors.append(
            "LoKR network_dim=114514 is a deprecated full-factor sentinel "
            "that suppresses training via alpha/dim. Use network_dim=32, "
            "network_alpha=32, lokr_full_factor=true instead."
        )
    return errors


def _check_dataset_image_dirs(cfg: dict, add) -> None:
    """确认训练/预处理图片目录存在且非空。

    WebUI 用 ``config_file`` 覆盖启动时，覆盖文件里通常没有数据目录，会静默
    继承 ``base.toml`` 的 ``source_image_dir = "image_dataset"``。那个目录若是
    空的，预处理会生成 0 张缓存，直到 ``train_session`` 才报 ``No data found``
    ——错误信息指向训练侧，排查成本高。

    这里在预检就把「目录不存在」和「目录存在但没有图片」两种必然失败拦下。
    只在没有任何可用目录时报错，避免 ``dataset_config`` 指向外部文件、图片由
    预处理另行生成的合法用法被误伤。
    """
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}
    checked: set[str] = set()
    missing: list[tuple[str, Path]] = []
    empty: list[tuple[str, Path]] = []

    # 只读配置里声明的**原始**数据集目录。不要用 dataset_rows：那里已经过
    # apply_auto_data_dirs 派生（例如 source=.../resized 会派生出
    # .../resized_resized），预检阶段该派生目录必然还不存在，会全部误报。
    raw_dirs: list[str] = []
    for dataset in cfg.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        subsets = dataset.get("subsets") or []
        if subsets:
            for subset in subsets:
                if isinstance(subset, dict) and subset.get("image_dir"):
                    raw_dirs.append(str(subset["image_dir"]))
        elif dataset.get("image_dir"):
            raw_dirs.append(str(dataset["image_dir"]))
    if not raw_dirs and cfg.get("source_image_dir"):
        raw_dirs.append(str(cfg["source_image_dir"]))

    for raw_in in raw_dirs:
        raw = raw_in.strip()
        if not raw or raw in checked:
            continue
        checked.add(raw)
        path = _resolve_project_path(raw)
        if not path.exists():
            missing.append((raw, path))
            continue
        if not path.is_dir():
            add("error", "datasets", f"数据集图片目录不是目录: {raw}", path)
            continue
        if not any(
            item.is_file() and item.suffix.lower() in image_suffixes
            for item in path.iterdir()
        ):
            empty.append((raw, path))

    for raw, path in missing:
        add(
            "error",
            "datasets",
            f"数据集图片目录不存在: {raw}"
            "（config_file 覆盖启动时容易漏掉数据目录，会静默继承 base.toml 的默认值）",
            path,
        )

    if empty and not missing:
        for raw, path in empty:
            add("error", "datasets", f"数据集图片目录里没有图片: {raw}", path)

    if checked and not missing and not empty:
        add("ok", "datasets", f"数据集图片目录可用（{len(checked)} 个）")


def preflight_training_config(
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    *,
    config_file: str | None = None,
    world_size: int | None = None,
    gpu_whitelist: Any = None,
) -> dict[str, Any]:
    """Run every preflight check for one training config.

    ``gpu_whitelist`` is the WebUI GPU selection (physical nvidia-smi indices).
    It is forwarded to the sm<80-only switch rules so the hardware probe uses
    the GPU the training subprocess will actually see, not the Web process's
    default device 0.
    """
    cfg = _load_training_config_for_web_run(
        variant, preset, methods_subdir, config_file=config_file
    )
    cfg = merge_stage_schedule_from_dataset_config(cfg)
    checks: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    runtime_config = is_web_runtime_config(
        config_file
    ) or _looks_like_web_runtime_config(cfg)

    def add(level: str, key: str, message: str, path: Path | None = None, **flags) -> None:
        item = {
            "level": level,
            "key": key,
            "message": message,
        }
        if path is not None:
            item["path"] = _display_path(path)
        item.update(flags)
        checks.append(item)
        if level == "error":
            errors.append(item)
        elif level == "warning":
            warnings.append(item)

    def check_file(key: str, label: str, suffixes: tuple[str, ...] = ()) -> None:
        raw = cfg.get(key)
        if not raw:
            add("error", key, f"{label} 未填写")
            return
        path = _resolve_project_path(str(raw))
        if not path.exists():
            add("error", key, f"{label} 不存在", path)
            return
        if not path.is_file():
            add("error", key, f"{label} 不是文件", path)
            return
        if suffixes and path.suffix.lower() not in suffixes:
            add("warning", key, f"{label} 后缀不是常见格式 {', '.join(suffixes)}", path)
            return
        add("ok", key, f"{label} 存在", path)

    def check_dir(
        key: str, label: str, *, must_exist: bool, warn_empty: bool = False
    ) -> None:
        raw = cfg.get(key)
        if not raw:
            add("error", key, f"{label} 未填写")
            return
        path = _resolve_project_path(str(raw))
        if not path.exists():
            if must_exist:
                add("error", key, f"{label} 不存在", path)
            else:
                add("warning", key, f"{label} 不存在，训练/预处理可能会创建它", path)
            return
        if not path.is_dir():
            add("error", key, f"{label} 不是目录", path)
            return
        if warn_empty and not any(path.iterdir()):
            add("warning", key, f"{label} 为空", path)
            return
        add("ok", key, f"{label} 存在", path)

    def check_diffusers_component(key: str, label: str, component: str) -> None:
        raw = cfg.get(key)
        if not raw:
            add("error", key, f"{label} 未填写")
            return
        root = _resolve_project_path(str(raw))
        if not root.exists():
            add("error", key, f"{label} 不存在", root)
            return
        if not root.is_dir():
            add("error", key, f"{label} 需要 Diffusers 目录", root)
            return
        path = root / component if (root / component).is_dir() else root
        if not (path / "config.json").is_file():
            add("error", key, f"{label} 缺少 {component}/config.json", root)
            return
        has_weights = any(path.glob("*.safetensors")) or any(
            path.glob("*.safetensors.index.json")
        )
        if not has_weights:
            add("error", key, f"{label} 缺少 safetensors 权重", root)
            return
        add("ok", key, f"{label} Diffusers 组件存在", root)

    def check_z_image_component(key: str, label: str, component: str) -> None:
        raw = cfg.get(key)
        if not raw:
            add("error", key, f"{label} 未填写")
            return
        root = _resolve_project_path(str(raw))
        if not root.exists():
            add("error", key, f"{label} 不存在", root)
            return
        if root.is_dir():
            check_diffusers_component(key, label, component)
        elif root.is_file():
            from library.models.z_image.weights import (
                ZImageCheckpointError,
                validate_z_image_single_file,
            )

            try:
                validate_z_image_single_file(str(root), component)
            except ZImageCheckpointError as exc:
                add("error", key, f"{label} 单文件格式不兼容: {exc}", root)
                return
            add("ok", key, f"{label} Z-Image 单文件存在且格式匹配", root)
        else:
            add("error", key, f"{label} 既不是文件也不是目录", root)
            return

        if component == "text_encoder":
            from library.models.z_image.weights import resolve_z_image_tokenizer_path

            # Keep the user-facing path here. Resolving a ComfyUI ``models``
            # symlink first loses the sibling ``comfy/text_encoders`` tree.
            tokenizer = Path(resolve_z_image_tokenizer_path(str(raw)))
            has_config = (tokenizer / "tokenizer_config.json").is_file()
            has_vocab = (tokenizer / "tokenizer.json").is_file() or (
                (tokenizer / "vocab.json").is_file()
                and (tokenizer / "merges.txt").is_file()
            )
            if not has_config or not has_vocab:
                add(
                    "error",
                    "z_image_tokenizer",
                    "Z-Image Qwen3 缺少 tokenizer 资源",
                    tokenizer,
                )
            else:
                add(
                    "ok",
                    "z_image_tokenizer",
                    "Z-Image tokenizer 资源存在",
                    tokenizer,
                )

        if component == "transformer" and "turbo" in root.name.casefold():
            add(
                "warning",
                "z_image_variant",
                "检测到 Z-Image Turbo 蒸馏模型；可以加载训练，但标准 LoRA 训练会改变其少步蒸馏特性，基础版更适合作为通用训练底模",
                root,
            )

    def check_moss_image_component(key: str, label: str, component: str) -> None:
        """Validate one MOSS component without accepting single-file checkpoints."""
        raw = cfg.get(key)
        if not raw:
            add("error", key, f"{label} 未填写")
            return
        root = _resolve_project_path(str(raw))
        if not root.exists():
            add("error", key, f"{label} 不存在", root)
            return
        if not root.is_dir():
            add("error", key, f"{label} 需要本地 Diffusers 目录", root)
            return

        component_dir = root / component
        path = component_dir if component_dir.is_dir() else root
        if not (path / "config.json").is_file():
            add("error", key, f"{label} 缺少 {component}/config.json", root)
            return
        has_weights = any(path.glob("*.safetensors")) or any(
            path.glob("*.safetensors.index.json")
        )
        if not has_weights:
            add("error", key, f"{label} 缺少 safetensors 权重或索引", root)
            return
        add("ok", key, f"{label} MOSS Diffusers 组件存在", path)

        if component != "text_encoder":
            return
        tokenizer_candidates = [root / "tokenizer"]
        if path != root:
            tokenizer_candidates.append(path.parent / "tokenizer")
        else:
            tokenizer_candidates.append(root.parent / "tokenizer")
        tokenizer = next(
            (
                candidate
                for candidate in tokenizer_candidates
                if (candidate / "tokenizer_config.json").is_file()
            ),
            tokenizer_candidates[0],
        )
        has_config = (tokenizer / "tokenizer_config.json").is_file()
        has_vocab = (tokenizer / "tokenizer.json").is_file() or (
            (tokenizer / "vocab.json").is_file()
            and (tokenizer / "merges.txt").is_file()
        )
        if not has_config or not has_vocab:
            add(
                "error",
                "moss_image_tokenizer",
                "MOSS Qwen3 缺少 tokenizer 资源",
                tokenizer,
            )
        else:
            add(
                "ok",
                "moss_image_tokenizer",
                "MOSS tokenizer 资源存在",
                tokenizer,
            )

    family_valid = True
    try:
        family = normalize_registered_family(
            cfg.get("model_family") or "anima",
            allow_aliases=True,
        )
    except ValueError as exc:
        add("error", "model_family", str(exc))
        family_valid = False
        family = "anima"
    cfg = dict(cfg)
    cfg["model_family"] = family

    if "output_name" in cfg and _is_blank_output_name(cfg.get("output_name")):
        add("error", "output_name", "输出名称未填写")
    dataset_rows = _dataset_rows_for_estimate(cfg)
    _check_core_training_semantics(cfg, add, dataset_rows=dataset_rows)
    task_cfg = task_check_config(cfg, _dataset_config_path_from_cfg(cfg), add)
    _check_checkpointing_config(
        task_cfg, add, world_size=world_size, gpu_whitelist=gpu_whitelist
    )
    _check_no_dataset_regularization_config(cfg, add)
    _check_output_dir_history_reuse(cfg, add)
    if family == "z_image":
        check_z_image_component(
            "pretrained_model_name_or_path", "基础 DiT 模型", "transformer"
        )
        check_z_image_component("qwen3", "Qwen3 文本编码器", "text_encoder")
        check_z_image_component("vae", "VAE 模型", "vae")
    elif family == "moss_image":
        check_moss_image_component(
            "pretrained_model_name_or_path", "基础 DiT 模型", "transformer"
        )
        check_moss_image_component("qwen3", "Qwen3 文本编码器", "text_encoder")
        check_moss_image_component("vae", "VAE 模型", "vae")
    else:
        check_file(
            "pretrained_model_name_or_path",
            "基础 DiT 模型",
            (".safetensors", ".pt", ".pth", ".ckpt"),
        )
        check_file("qwen3", "Qwen3 文本编码器", (".safetensors", ".pt", ".pth", ".bin"))
        check_file("vae", "VAE 模型", (".safetensors", ".pt", ".pth", ".ckpt"))
    if cfg.get("pretrained_model_name_or_path"):
        check_mixed_nf4(cfg, _resolve_project_path(str(cfg["pretrained_model_name_or_path"])), add)
    _check_dataset_image_dirs(cfg, add)
    _check_network_weights(cfg, add, variant, preset, methods_subdir, config_file)
    dataset_config_path = _dataset_config_path_from_cfg(cfg)
    if cfg.get("dataset_config") and (
        runtime_config or (dataset_config_path and dataset_config_path.exists())
    ):
        check_file("dataset_config", "数据集配置", (".toml",))

    _check_dataset_source_paths(cfg, add)
    _check_dataset_bucket_settings(cfg, add)
    _check_dataset_paths(cfg, add, check_runtime_dirs=runtime_config)
    _check_training_sample_config(cfg, add)
    check_stage_schedule(
        cfg,
        dataset_rows=dataset_rows,
        add=add,
    )
    schema_errors, schema_warnings = validate_config_mapping(cfg)
    schema_errors.extend(_validate_lokr_config(cfg))
    for msg in schema_errors:
        add("error", "schema", msg)
    for msg in schema_warnings:
        add("warning", "schema", msg)
    if not runtime_config:
        _check_web_preprocess_environment(add)
    if runtime_config:
        _check_training_images(cfg, add)
        _check_cache_sidecars(cfg, add)

    return {
        "ok": not errors,
        "can_override": bool(errors) and all(item.get("overridable") is True for item in errors),
        "training_task": training_task_summary(task_cfg) if family_valid else None,
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "summary": {
            "errors": len(errors),
            "warnings": len(warnings),
            "checks": len(checks),
        },
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


for _name in __all__:
    globals()[_name] = _exported(globals()[_name])
