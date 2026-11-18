"""Compatibility / sampler / network-weight preflight checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
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

@dataclass(frozen=True)
class _CudaDeviceProbe:
    """Local CUDA context for the sm<80-only fp16 inner-path switches."""

    available: bool
    capability: tuple[int, int] | None


@dataclass(frozen=True)
class _Fp16InnerPathHardware:
    """Hardware context fed into the shared compat rules.

    白名单非空时用 ``selected_capabilities``（训练子进程真正可见的物理 GPU）；
    白名单为空/未指定时退回 Web 进程 device-0 探针的 ``cuda_available`` /
    ``capability``（此时子进程不覆盖 ``CUDA_VISIBLE_DEVICES``，两者口径一致）。
    ``selection_unresolved`` 表示白名单给出但 capability 解析不到，compat 会把它
    降级成 warning 而不是硬阻断。
    """

    cuda_available: bool | None = None
    capability: tuple[int, int] | None = None
    selected_capabilities: tuple[tuple[int, int], ...] = ()
    selection_unresolved: bool = False
    device_label: str = ""


#: 只在 sm<80（无原生 BF16 Tensor Core）有意义的两个实验开关。
FP16_INNER_PATH_FLAGS = ("nf4_dir1", "krea2_attn_fp16")


def _flag_enabled(cfg: dict[str, Any], flag: str) -> bool:
    """Local truthiness for the two switches.

    Deliberately not ``preflight_runtime._bool_value``: that helper is only
    filled by the config-service facade sync, and the hardware probe must stay
    callable from unit tests (and from any future direct caller) without the
    facade import cycle.
    """
    value = cfg.get(flag)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _fp16_inner_path_requires_probe(cfg: dict[str, Any]) -> bool:
    return any(_flag_enabled(cfg, flag) for flag in FP16_INNER_PATH_FLAGS)


def _probe_cuda_device() -> _CudaDeviceProbe:
    """Read the local CUDA device, failing closed for preflight decisions.

    ``torch.cuda.is_available()`` 为 False 时按「无 CUDA」处理；
    设备可见但读不到 compute capability 时返回 ``(True, None)``，让兼容规则
    以「无法确认架构」降级为 warning，而不是假装 sm70/sm75 放行。
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - a broken torch install must fail closed
        return _CudaDeviceProbe(False, None)
    try:
        if not torch.cuda.is_available():
            return _CudaDeviceProbe(False, None)
    except Exception:  # noqa: BLE001 - driver errors must fail closed
        return _CudaDeviceProbe(False, None)
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001 - unknown architecture must fail closed
        return _CudaDeviceProbe(True, None)
    return _CudaDeviceProbe(True, (int(major), int(minor)))


def _probe_cuda_device_for(cfg: dict[str, Any]) -> _CudaDeviceProbe | None:
    """Probe CUDA only when one of the switches is on (avoids needless context)."""
    if not _fp16_inner_path_requires_probe(cfg):
        return None
    return _probe_cuda_device()


def _selected_device_capabilities(
    gpu_whitelist: Any,
) -> tuple[list[int], tuple[tuple[int, tuple[int, int]], ...]]:
    """Resolve ``(selected_indices, [(physical_index, capability), ...])``.

    返回配对而不是两个平行序列：部分解析时 ``selected`` 与 ``capabilities`` 长度
    不等，平行序列会让 label 把 capability 安到错误的物理 index 上。
    """
    from web.services.training.gpu import (
        normalize_gpu_whitelist,
        query_gpu_compute_capabilities,
    )

    selected = normalize_gpu_whitelist(gpu_whitelist)
    if not selected:
        return [], ()
    found = query_gpu_compute_capabilities(selected)
    paired = tuple((index, found[index]) for index in selected if index in found)
    return selected, paired


def _format_device_label(paired: tuple[tuple[int, tuple[int, int]], ...]) -> str:
    if not paired:
        return ""
    parts = [f"GPU{index} sm_{major}{minor}" for index, (major, minor) in paired]
    return "、".join(parts)


def _fp16_inner_path_hardware(
    cfg: dict[str, Any], *, gpu_whitelist: Any = None
) -> _Fp16InnerPathHardware | None:
    """Build the hardware context, preferring the task's whitelisted GPUs.

    ``gpu_whitelist`` 为空/未指定时保持原行为（Web 进程 device 0 探针）；非空时
    按 nvidia-smi 物理 index 解析每张被选设备的 compute capability，避免在异构
    多卡机上用错卡判断 sm 架构。
    """
    if not _fp16_inner_path_requires_probe(cfg):
        return None
    selected, paired = _selected_device_capabilities(gpu_whitelist)
    if selected:
        capabilities = tuple(capability for _, capability in paired)
        return _Fp16InnerPathHardware(
            selected_capabilities=capabilities,
            selection_unresolved=len(capabilities) != len(selected),
            device_label=_format_device_label(paired),
        )
    probe = _probe_cuda_device_for(cfg)
    if probe is None:  # pragma: no cover - guarded by requires_probe above
        return None
    label = (
        f"当前预检设备 sm_{probe.capability[0]}{probe.capability[1]}"
        if probe.capability is not None
        else ""
    )
    return _Fp16InnerPathHardware(
        cuda_available=probe.available,
        capability=probe.capability,
        device_label=label,
    )


def _compat_web_message(
    item, *, device: _Fp16InnerPathHardware | None = None
) -> str:
    from web.services.config.compat_messages_zh import format_compat_web_message

    return format_compat_web_message(item, device=device)


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
    gpu_whitelist: Any = None,
) -> None:
    device = _fp16_inner_path_hardware(cfg, gpu_whitelist=gpu_whitelist)
    compat = check_training_compat(
        cfg,
        world_size=world_size,
        cuda_available=None if device is None else device.cuda_available,
        device_capability=None if device is None else device.capability,
        selected_device_capabilities=(
            None
            if device is None or not device.selected_capabilities
            else device.selected_capabilities
        ),
        selection_unresolved=False if device is None else device.selection_unresolved,
    )
    defer_adaptive_contract = _adaptive_precision_requires_runtime_resolution(cfg)
    for item in compat.errors:
        if defer_adaptive_contract and item.code == "adaptive_precision_contract":
            continue
        add("error", item.key, _compat_web_message(item, device=device))
    _check_pipeline_trainer_admission(cfg, add, world_size=world_size)

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
        add("warning", item.key, _compat_web_message(item, device=device))
    for item in compat.mutations:
        if item.code in seen_warning_codes:
            continue
        add("warning", item.key, _compat_web_message(item, device=device))
    for item in compat.infos:
        add("info", item.key, _compat_web_message(item, device=device))


def _check_pipeline_trainer_admission(
    cfg: dict[str, Any],
    add,
    *,
    world_size: int | None = None,
) -> None:
    """Surface the wired PP trainer admission before the launch (contract §5).

    The compat layer stays family-agnostic; this adds the trainer-path knob
    view (preview 类字段、compile、router、缓存等) so the WebUI shows the
    named refusal the queue dispatch would hit instead of failing only in the
    worker log. A named trainer refusal is an error, matching the launch gate.
    """

    try:
        from library.models.pipeline_parallel import PipelineParallelConfig

        if not PipelineParallelConfig.from_config(cfg).enabled:
            return
    except ValueError:
        return  # the compat layer above already reports pipeline_parallel_config
    from library.training.pp.errors import PipelineParallelError
    from library.training.pp.trainer_admission import require_family_trainer_admission
    from web.services.config.preflight_pp import krea2_batch_checks

    reported_codes: set[str] = set()
    for check in krea2_batch_checks(cfg):
        extra = {}
        if check.get("code"):
            reported_codes.add(check["code"])
            extra["code"] = check["code"]
        add(check["level"], check["key"], check["message"], **extra)
    add(
        "info", "pipeline_parallel_blocks_to_swap",
        "PP 配置：pipeline_parallel_blocks_to_swap="
        f"{cfg.get('pipeline_parallel_blocks_to_swap', 0)}。",
    )

    try:
        require_family_trainer_admission(
            cfg, world_size=world_size, where="webui_preflight"
        )
    except PipelineParallelError as exc:
        if exc.code not in reported_codes:
            add(
                "error",
                "pipeline_parallel",
                f"PP 训练准入预检：启动时会以 {exc.code} 具名拒绝：{exc.detail}",
                code=exc.code,
            )


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
    from library.training.regularization_compat import validate_no_dataset_regularization

    for issue in validate_no_dataset_regularization(cfg):
        add("error", issue.key, issue.web_message)

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
