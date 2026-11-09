"""Compatibility / sampler / network-weight preflight checks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from library.datasets.subsets import normalize_sample_ratio
from library.training.compat_matrix import check_training_compat
from web.services.continue_lora_service import inspect_continue_lora_weight as _inspect_continue_lora_weight
from web.services.config.metadata import (
    DATASET_IMAGE_EXTS,
    LEGACY_TRAINING_SAMPLE_SAMPLERS,
    PREPROCESS_ENV_CHECK_KEY,
    PREPROCESS_ENV_REQUIRED_FILES,
    SUPPORTED_TRAINING_SAMPLE_SAMPLERS,
)
from web.services.config.preflight_runtime import (
    ROOT,
    _bool_value,
    _count_source_images,
    _nonnegative_float_value,
    _normalize_path_pattern,
    _positive_int_or_none,
    _resolve_project_path,
)

def _inspect_network_weight(
    path: str,
    *,
    variant: str,
    preset: str,
    methods_subdir: str,
    config_file: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _inspect_network_weight_impl(
        path,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=config_file,
        cfg=cfg,
    )

def _inspect_network_weight_impl(
    path: str,
    *,
    variant: str,
    preset: str,
    methods_subdir: str,
    config_file: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del config_file
    return _inspect_continue_lora_weight(
        path,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        cfg=cfg,
        root=ROOT,
    )

def _compat_web_message(item) -> str:
    messages = {
        "invalid_selective_checkpoint": (
            "selective_checkpoint 取值不受支持；请改为 off、adapter_aware、"
            "mlp_only、peak_blocks_mlp 等已知模式。"
        ),
        "krea2_invalid_attn_mode": (
            "Krea-2 注意力后端仅支持 torch 或 flash（sdpa 作为 torch 别名）。"
        ),
        "krea2_compile_dynamic_seq": (
            "Krea-2 使用两张固定 token-family 编译图；"
            "训练启动时会自动关闭 compile_dynamic_seq。"
        ),
        "compile_seq_bands_requires_dynamic_seq": (
            "compile_seq_bands 只在 compile_dynamic_seq=true 时生效；"
            "训练启动时会自动关闭。"
        ),
        "krea2_compile_seq_bands": (
            "Krea-2 使用固定 token-family 编译图；"
            "训练启动时会自动关闭 compile_seq_bands。"
        ),
        "z_image_compile_seq_bands": (
            "compile_seq_bands 仅适用于 Anima native-flatten 路径；"
            "Z-Image 训练启动时会自动关闭。"
        ),
        "krea2_compile_inductor_mode": (
            "Krea-2 仅支持 compile_inductor_mode=default。"
        ),
        "krea2_selective_checkpoint": (
            "Krea-2 选择性检查点仅支持 off 或 every_other。"
        ),
        "krea2_v100_flash_stability": (
            "v100_flash_stability 是 Anima 专用诊断项，Krea-2 下必须为 off。"
        ),
        "krea2_plain_lora_only": "Krea-2 当前仅支持 plain LoRA，不能组合高级 adapter 或路由。",
        "pipeline_parallel_krea2_only": (
            "流水线并行当前仅支持 Krea-2 Raw"
            "（model_family=krea2_raw 或别名 krea2）。"
        ),
        "pipeline_parallel_config": (
            "流水线配置无效；当前需要两阶段、1F1B、balanced，"
            "只训练 DiT，并关闭 block swap、torch.compile、选择性 checkpoint "
            "与 CPU/Unsloth activation offload。"
        ),
        "krea2_pipeline_parallel_config": (
            "Krea-2 流水线配置无效；当前需要两阶段、1F1B、balanced，"
            "只训练 DiT，并关闭 block swap、torch.compile、选择性 checkpoint "
            "与 CPU/Unsloth activation offload。"
        ),
        "pipeline_parallel_runtime_unavailable": (
            "当前模型族的流水线分层规划已接入，但主训练 loop 的 1F1B 调度尚未接入；"
            "当前会阻止启动，不会静默回退到 DDP。"
        ),
        "pipeline_parallel_experimental_runtime": (
            "Anima 流水线并行已解析到实验性的 stage-local PP v0 运行时"
            "（2 阶段 / 1F1B / fp16 事务）。它由 library.training.pp 驱动，"
            "主训练入口遇到 pipeline_parallel 会以 pp_trainer_path_unavailable "
            "具名拒绝，绝不静默回退 DDP。"
        ),
        "pipeline_parallel_microbatches_below_stages": (
            "pipeline_parallel_microbatches 必须 >= pipeline_parallel_stages；"
            "torch 2.10 的 Schedule1F1B 与 ScheduleGPipe 都会在构造时拒绝更小的值，"
            "所以 microbatches=1 不是 2 阶段流水线的 GPipe 替代方案。"
        ),
        "negative_blocks_to_swap": "blocks_to_swap 不能小于 0。",
        "selective_full_gradient_checkpointing": (
            "selective_checkpoint 是 DiT 选择性检查点模式，不能同时开启完整 "
            "gradient_checkpointing；请关闭完整检查点，保留 selective_checkpoint。"
        ),
        "selective_cpu_offload": (
            "selective_checkpoint 不支持 CPU activation offload；请关闭 cpu_offload_checkpointing。"
        ),
        "selective_unsloth_offload": (
            "selective_checkpoint 不能和 unsloth_offload_checkpointing 同时开启。"
        ),
        "block_swap_cpu_offload": (
            "blocks_to_swap 可以和普通 gradient_checkpointing 同用，但不能和 "
            "cpu_offload_checkpointing 同时开启。"
        ),
        "block_swap_unsloth_offload": (
            "blocks_to_swap 可以和普通 gradient_checkpointing 同用，但不能和 "
            "unsloth_offload_checkpointing 同时开启。"
        ),
        "unsloth_cpu_offload": (
            "unsloth_offload_checkpointing 不能和 cpu_offload_checkpointing 同时开启。"
        ),
        "unsloth_enables_gradient_checkpointing": (
            "unsloth_offload_checkpointing 已开启，训练启动时会自动开启 gradient_checkpointing。"
        ),
        "cpu_offload_without_gradient_checkpointing": (
            "cpu_offload_checkpointing 只有配合 gradient_checkpointing 才会生效；"
            "请开启 gradient_checkpointing 或关闭 cpu_offload_checkpointing。"
        ),
        "lokr_full_checkpoint_compile": (
            "LoKr + 完整 gradient_checkpointing + torch_compile 属于实验性叠加；"
            "启动编译时会提高 Dynamo graph/accumulated 预算并稳定 graph 查找顺序，"
            "blocks_to_swap 可继续保留。"
        ),
        "block_swap_cudagraphs_disable_compile": (
            "blocks_to_swap 会在 CPU/GPU 间移动 DiT block 权重，"
            "dynamo_backend='cudagraphs' 不安全；训练启动时会关闭 torch_compile。"
        ),
        "block_swap_soft_tokens": (
            "blocks_to_swap 不支持 Soft Tokens 这类 multi-forward 方法；请保持 blocks_to_swap=0。"
        ),
        "block_swap_functional_loss": (
            "blocks_to_swap 不支持 functional_loss_weight > 0 的 multi-forward 训练；"
            "请关闭 block swap。"
        ),
    }
    if item.code == "block_swap_compile_mode_cudagraphs":
        value = getattr(item, "value", None)
        target = value or "默认 Inductor mode"
        return f"blocks_to_swap 不兼容 Inductor CUDAGraph mode；训练启动时会改用 {target}。"
    return messages.get(item.code, item.message)


def _adaptive_precision_requires_runtime_resolution(cfg: dict[str, Any]) -> bool:
    """Return whether ``adaptive_precision=auto`` is still hardware-agnostic.

    WebUI preflight runs before the training process has selected a CUDA device,
    while the CLI resolves ``auto`` immediately before validating the worker
    contract.  Do not let the raw ``mixed_precision`` value stand in for that
    missing capability probe.
    """
    requested = str(cfg.get("adaptive_precision", "off") or "off").strip().lower()
    resolved = str(cfg.get("adaptive_resolved_mode", "") or "").strip().lower()
    return requested == "auto" and resolved not in {"bf16", "native_fp16", "fp16_fp32", "fp32"}


def _check_checkpointing_config(
    cfg: dict[str, Any],
    add,
    *,
    world_size: int | None = None,
) -> None:
    compat = check_training_compat(cfg, world_size=world_size)
    defer_adaptive_contract = _adaptive_precision_requires_runtime_resolution(cfg)
    for item in compat.errors:
        if defer_adaptive_contract and item.code == "adaptive_precision_contract":
            continue
        add("error", item.key, _compat_web_message(item))

    if defer_adaptive_contract:
        add(
            "warning",
            "adaptive_precision",
            "adaptive_precision=auto 将在训练启动时按实际 CUDA compute capability 解析；"
            "当前 WebUI 预检不会用 mixed_precision 代替硬件结果，启动阶段仍会执行完整兼容性校验。",
        )

    mutation_codes = {item.code for item in compat.mutations}
    seen_warning_codes: set[str] = set()
    for item in compat.warnings:
        if item.code == "block_swap_compile_mode_cudagraphs" and item.code in mutation_codes:
            continue
        seen_warning_codes.add(item.code)
        add("warning", item.key, _compat_web_message(item))
    for item in compat.mutations:
        if item.code in seen_warning_codes:
            continue
        add("warning", item.key, _compat_web_message(item))


def _check_core_training_semantics(
    cfg: dict[str, Any],
    add,
    *,
    dataset_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Check field relationships that the trainer resolves at runtime."""

    raw_epochs = cfg.get("max_train_epochs")
    epochs_configured = raw_epochs not in (None, "")
    epochs = _int_or_none(raw_epochs)
    steps = _int_or_none(cfg.get("max_train_steps"))
    if epochs_configured:
        if epochs is None or epochs <= 0:
            add(
                "error",
                "max_train_epochs",
                "max_train_epochs 已设置但不是大于 0 的整数；训练端仍会用它覆盖 max_train_steps。",
            )
        elif steps is not None and steps > 0:
            add(
                "warning",
                "max_train_steps",
                "max_train_epochs 已设置；训练端会按当前数据集重新计算并覆盖 max_train_steps。",
            )
    elif steps is None or steps <= 0:
        add(
            "error",
            "max_train_steps",
            "未设置有效训练时长：请填写大于 0 的 max_train_epochs，或使用大于 0 的 max_train_steps。",
        )

    raw_ratio = cfg.get("sample_ratio")
    if raw_ratio not in (None, ""):
        _check_sample_ratio_value(raw_ratio, "sample_ratio", add)
    for index, row in enumerate(dataset_rows or (), start=1):
        _check_dataset_sample_ratios(row, index, add)

    network_weights = str(cfg.get("network_weights") or "").strip()
    if _bool_value(cfg.get("dim_from_weights"), False) and not network_weights:
        add(
            "error",
            "dim_from_weights",
            "已开启从权重读取维度，但未填写 network_weights；训练端无法推断 rank/alpha。",
        )

    save_keep = _int_or_none(cfg.get("save_last_n_epochs"))
    if save_keep == 0:
        add(
            "warning",
            "save_last_n_epochs",
            "普通权重保留数量 0 不表示关闭；训练端会按 1 份处理。负数才表示不清理。",
        )
    checkpoint_keep = _int_or_none(cfg.get("checkpointing_last_n_epochs"))
    if checkpoint_keep == 0 or (
        checkpoint_keep is not None and checkpoint_keep < -1
    ):
        add(
            "warning",
            "checkpointing_last_n_epochs",
            "续训状态保留数量仅 -1 表示不清理；0 或小于 -1 的值都会按 1 份处理。",
        )

    _validation_configured, validation_effective, validation_auto_disabled = (
        _validation_state(dataset_rows or ())
    )
    if validation_auto_disabled:
        add(
            "warning",
            "validation_split",
            "验证集划分已配置，但有效训练图少于 100；训练端会自动关闭验证并把整组图片用于训练。",
        )
    if _bool_value(cfg.get("use_cmmd"), False) and not validation_effective:
        detail = (
            "训练端会因小数据集自动关闭验证"
            if validation_auto_disabled
            else "数据集没有有效验证集划分"
        )
        add(
            "warning",
            "use_cmmd",
            f"已开启 CMMD，但{detail}；本次训练不会运行 CMMD 验证。",
        )


def _check_sample_ratio_value(value: Any, key: str, add) -> None:
    try:
        normalize_sample_ratio(value)
    except ValueError:
        add("error", key, "sample_ratio 必须满足 0 < 值 <= 1；训练端不会静默回退。")


def _check_dataset_sample_ratios(row: dict[str, Any], index: int, add) -> None:
    values: list[tuple[str, Any]] = []
    if "sample_ratio" in row:
        values.append(("subset", row.get("sample_ratio")))
    for scope, map_key in (
        ("subset", "preserved_subset_fields"),
        ("dataset", "preserved_dataset_fields"),
        ("general", "preserved_general_fields"),
    ):
        mapping = row.get(map_key)
        if isinstance(mapping, dict) and "sample_ratio" in mapping:
            values.append((scope, mapping.get("sample_ratio")))
    for scope, value in values:
        _check_sample_ratio_value(
            value,
            f"dataset_{index}_{scope}_sample_ratio",
            add,
        )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validation_state(rows) -> tuple[bool, bool, bool]:
    configured = False
    effective = False
    auto_disabled = False
    for row in rows:
        if _bool_value(row.get("is_reg"), False):
            continue
        settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
        try:
            split_num = int(settings.get("validation_split_num") or 0)
            split_ratio = float(settings.get("validation_split") or 0.0)
        except (TypeError, ValueError):
            continue
        if split_num <= 0 and split_ratio <= 0.0:
            continue
        configured = True
        candidates = [
            str(row.get("image_dir") or "").strip(),
            str(row.get("source_dir") or "").strip(),
        ]
        image_dir = next(
            (
                path
                for raw_dir in candidates
                if raw_dir
                for path in (_resolve_project_path(raw_dir),)
                if path.is_dir()
            ),
            None,
        )
        image_count = (
            _count_source_images(
                image_dir,
                DATASET_IMAGE_EXTS,
                recursive=_bool_value(row.get("recursive"), True),
                path_pattern=_normalize_path_pattern(row.get("path_pattern")),
            )
            if image_dir is not None
            else 0
        )
        if 0 < image_count < 100:
            auto_disabled = True
            continue
        if split_num > 0:
            effective = effective or image_count > split_num
        elif 0.0 < split_ratio < 1.0 and image_count > 0:
            train_count = math.ceil(image_count * (1.0 - split_ratio))
            effective = effective or train_count < image_count
    return configured, effective, auto_disabled


def _check_no_dataset_regularization_config(cfg: dict[str, Any], add) -> None:
    prior_weight = _nonnegative_float_value(cfg.get("prior_preservation_weight"), 0.0)
    mask_weight = _nonnegative_float_value(cfg.get("inverted_mask_prior_weight"), 0.0)
    blank_enabled = _bool_value(cfg.get("blank_prompt_preservation"), False)
    dop_trigger = str(cfg.get("diff_output_preservation_trigger") or "").strip()
    dop_class = str(cfg.get("diff_output_preservation_class") or "").strip()
    use_text_cache = _bool_value(cfg.get("use_text_cache"), False)
    cache_llm_adapter_outputs = _bool_value(cfg.get("cache_llm_adapter_outputs"), False)

    if prior_weight > 0.0 and dop_trigger and not dop_class and not blank_enabled:
        add(
            "error",
            "diff_output_preservation_class",
            "DOP 已填写触发词，但未填写类提示；请填写 woman / character / style 等类提示，或关闭 prior_preservation_weight。",
        )
    elif prior_weight > 0.0 and not (blank_enabled or dop_class):
        add(
            "error",
            "prior_preservation_weight",
            "prior_preservation_weight 大于 0 时，需要开启 blank_prompt_preservation 或填写 DOP 类提示。",
        )

    if prior_weight > 0.0 and blank_enabled and dop_class:
        add(
            "error",
            "blank_prompt_preservation",
            "blank_prompt_preservation 不能和 DOP 类提示同时使用。",
        )

    if (prior_weight > 0.0 or mask_weight > 0.0) and not use_text_cache:
        add(
            "error",
            "use_text_cache",
            "无数据集正则化需要 use_text_cache=true；请开启文本缓存后重新预处理。",
        )
    prior_needs_adapter_cache = prior_weight > 0.0 and (blank_enabled or dop_class)
    if (prior_needs_adapter_cache or mask_weight > 0.0) and not cache_llm_adapter_outputs:
        add(
            "error",
            "cache_llm_adapter_outputs",
            "无数据集正则化需要 cache_llm_adapter_outputs=true；请开启 LLM adapter 输出缓存后重新预处理。",
        )

def _check_network_weights(
    cfg: dict[str, Any],
    add,
    variant: str,
    preset: str,
    methods_subdir: str,
    config_file: str | None,
) -> None:
    raw = str(cfg.get("network_weights") or "").strip()
    if not raw:
        return
    try:
        info = _inspect_network_weight(
            raw,
            variant=variant,
            preset=preset,
            methods_subdir=methods_subdir,
            config_file=config_file,
            cfg=cfg,
        )
    except Exception as exc:
        add("error", "network_weights", f"热启动权重不可用：{exc}", _resolve_project_path(raw))
        return

    weight_path = _resolve_project_path(str(info.get("abs_path") or raw))
    if not info.get("compatible", False):
        message = str(info.get("message") or "当前训练配置与 network_weights 不兼容")
        add("error", "network_weights", message, weight_path)
        return

    kind = str(info.get("kind") or "LoRA/LoHa/LoKr/GLoRA")
    message = f"热启动权重可用（{kind}）"
    if _bool_value(cfg.get("dim_from_weights"), False):
        message += "，将从权重读取维度"
    add("ok", "network_weights", message, weight_path)

def _check_training_sample_config(cfg: dict[str, Any], add) -> None:
    from library.training.preview_spec import validate_preview_file

    try:
        validate_preview_file(cfg)
    except (ValueError, OSError, TypeError) as exc:
        add("error", "sample_prompts", str(exc))
    sample_prompts = str(cfg.get("sample_prompts") or "").strip()
    epoch_freq = _positive_int_or_none(cfg.get("sample_every_n_epochs"))
    step_freq = _positive_int_or_none(cfg.get("sample_every_n_steps"))
    sample_at_first = _bool_value(cfg.get("sample_at_first"), False)

    if sample_prompts and epoch_freq is None and step_freq is None and not sample_at_first:
        add(
            "warning",
            "sample_prompts",
            "已填写 sample_prompts，但未启用训练前、按轮或按步采样，训练不会生成样张",
        )

    if epoch_freq is not None and step_freq is not None:
        add(
            "warning",
            "sample_schedule",
            "已同时启用按轮和按步采样，会分别在轮末和步数命中时生成样张，采样开销会增加",
        )

    raw_sampler = str(cfg.get("sample_sampler") or "euler").strip().lower()
    sampler, sampler_status = training_sample_sampler_status(raw_sampler)
    if sampler_status == "legacy":
        add(
            "warning",
            "sample_sampler",
            f"sample_sampler={raw_sampler} 是旧 Diffusers 采样器名，训练预览会按 {sampler} 兼容处理",
        )
    elif sampler_status == "unknown":
        add(
            "warning",
            "sample_sampler",
            f"sample_sampler={raw_sampler} 当前训练预览不支持，会按 {sampler} 处理",
        )

def _check_web_preprocess_environment(add) -> None:
    python_exe = Path(_web_python_executable())
    if not python_exe.is_file():
        add(
            "error",
            PREPROCESS_ENV_CHECK_KEY,
            f"预处理启动环境异常: Python 解释器不存在 {python_exe}",
            python_exe,
        )
        return
    missing = [rel for rel in PREPROCESS_ENV_REQUIRED_FILES if not (ROOT / rel).is_file()]
    if missing:
        add(
            "error",
            PREPROCESS_ENV_CHECK_KEY,
            f"预处理启动环境异常: 缺少 {', '.join(missing)}",
            ROOT / missing[0],
        )
        return
    add("ok", PREPROCESS_ENV_CHECK_KEY, "预处理启动环境文件检查通过", ROOT)

def _web_python_executable() -> str:
    from web.services.project_python import resolve_web_python_executable

    return resolve_web_python_executable(ROOT)

def training_sample_sampler_status(value: Any) -> tuple[str, str]:
    sampler = str(value or "euler").strip().lower()
    if sampler in SUPPORTED_TRAINING_SAMPLE_SAMPLERS:
        return sampler, "supported"
    if sampler in LEGACY_TRAINING_SAMPLE_SAMPLERS:
        return "euler", "legacy"
    return "euler", "unknown"
