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
    messages = {
        "invalid_selective_checkpoint": (
            "selective_checkpoint 取值不受支持；请改为 off、adapter_aware、"
            "mlp_only、peak_blocks_mlp 等已知模式。"
        ),
        "krea2_invalid_attn_mode": (
            "Krea-2 注意力后端仅支持 torch、flash，以及 sdpa/cudnn/cudnn_sdpa"
            "（这三个都是 torch 的别名）；其余写法会被启动预检拒绝，"
            "加载阶段也会报错。"
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
        "nf4_dir1_ignored": (
            "nf4_dir1 只在 base_compute=nf4 时生效；当前 base_compute 没有 NF4 Linear "
            "可加速，开关会被忽略。请关闭该开关，或切换 base_compute=nf4。"
        ),
        "nf4_dir1_requires_cuda": (
            "nf4_dir1 仅面向无原生 BF16 Tensor Core 的 sm70/sm75（V100 / 2080Ti / T4），"
            "且需要 CUDA 设备：set_dir1_enabled(True) 在无 CUDA 时会直接抛 RuntimeError。"
            "当前预检未检测到可用 CUDA，会阻止启动，不会静默回退；请保持 nf4_dir1=false。"
        ),
        "nf4_dir1_requires_pre_ampere": (
            "nf4_dir1 仅面向无原生 BF16 Tensor Core 的 sm70/sm75（V100 / 2080Ti / T4）；"
            "训练设备上 bf16 已走 tensor core，set_dir1_enabled(True) 会直接抛 ValueError。"
            "sm80+ 请保持 nf4_dir1=false 以使用原生 BF16；当前会阻止启动，不会静默回退。"
        ),
        "nf4_dir1_architecture_unknown": (
            "无法解析训练设备的 compute capability（白名单里的 GPU 查不到架构，或当前 CUDA 设备读不到算力），"
            "不能确认属于 sm70/sm75；预检降级为警告、不阻断启动，"
            "运行时 set_dir1_enabled(True) 仍会在 sm80+ 上抛错。请确认显卡架构，"
            "或关闭 nf4_dir1。"
        ),
        "krea2_attn_fp16_requires_cuda": (
            "krea2_attn_fp16 仅面向无原生 BF16 Tensor Core 的 sm70/sm75"
            "（V100 / 2080Ti / T4），需要 CUDA 设备才能拿到它使用的 SDPA 内核。"
            "当前预检看不到可用 CUDA：运行时只是把全局 flag 置真后空转、不会抛错，"
            "所以这里按「开关不生效」给出警告而不阻断启动。请保持 krea2_attn_fp16=false。"
        ),
        "krea2_attn_fp16_requires_pre_ampere": (
            "krea2_attn_fp16 仅面向无原生 BF16 Tensor Core 的 sm70/sm75"
            "（V100 / 2080Ti / T4）；训练设备已有原生 bf16 SDPA 内核，"
            "set_krea2_attn_fp16(True) 会直接抛 ValueError。sm80+ 请保持 "
            "krea2_attn_fp16=false 以使用原生 BF16；当前会阻止启动，不会静默回退。"
        ),
        "krea2_attn_fp16_architecture_unknown": (
            "无法解析训练设备的 compute capability（白名单里的 GPU 查不到架构，或当前 CUDA 设备读不到算力），"
            "不能确认属于 sm70/sm75；预检降级为警告、不阻断启动，"
            "运行时 set_krea2_attn_fp16(True) 仍会在 sm80+ 上抛错。请确认显卡架构，"
            "或关闭 krea2_attn_fp16。"
        ),
        "krea2_attn_fp16_requires_torch_attention": (
            "krea2_attn_fp16 只在 attn_mode=torch/sdpa（cudnn、cudnn_sdpa 是同义别名）"
            "且 mixed_precision=bf16 时生效：运行时代码只在 cuDNN SDPA 分支里对 bf16 "
            "输入读取该开关。attn_mode=flash 会直接走 Flash varlen 并以自身 dtype 运行，"
            "compute 不是 bf16 时也不会进入该分支，两种情况下开关都是空转。"
            "要获得 attention 内层 fp16 + GQA 展开，请把 attn_mode 切到 torch"
            "（sdpa/cudnn/cudnn_sdpa 等价）并保持 mixed_precision=bf16；"
            "否则请关闭 krea2_attn_fp16。"
        ),
        "krea2_attn_fp16_ignored": (
            "krea2_attn_fp16 只在 Krea-2（model_family=krea2_raw）的 attention 路径生效；"
            "当前模型族不会走 Krea-2 attention，开关会被忽略。"
            "请关闭该开关，或切换模型族。"
        ),
        "nf4_fp16_inner_pairing": (
            "当前设备无原生 BF16 Tensor Core（sm70/sm75）：Krea-2 + NF4 下 nf4_dir1 负责 "
            "NF4 Linear 内层 fp16 GEMM，krea2_attn_fp16 负责 attention 内层 fp16 SDPA"
            "（GQA 展开为 MHA）。当前只开启其中一个，建议两个同时开启，"
            "以获得完整的内层 fp16 收益。"
        ),
    }
    if item.code == "block_swap_compile_mode_cudagraphs":
        value = getattr(item, "value", None)
        target = value or "默认 Inductor mode"
        return f"blocks_to_swap 不兼容 Inductor CUDAGraph mode；训练启动时会改用 {target}。"
    message = messages.get(item.code, item.message)
    if device is not None and item.code.endswith("_requires_pre_ampere"):
        if device.selected_capabilities:
            detail = device.device_label or ", ".join(
                f"sm_{major}{minor}"
                for major, minor in device.selected_capabilities
            )
            message = f"{message}（被选 GPU: {detail}）"
        elif device.capability is not None:
            major, minor = device.capability
            message = f"{message}（当前预检设备 sm_{major}{minor}）"
    return message


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
