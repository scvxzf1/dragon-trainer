"""Shared training compatibility checks for optimization flags.

The helpers here are intentionally pure: they inspect a mapping/object and
return errors, warnings, and suggested mutations. CLI and Web preflight can
consume the same rules without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from library.models.family_registry import (
    get_model_family_spec,
    normalize_registered_family,
)
from library.models.pipeline_parallel import (
    PipelineParallelConfig,
    validate_pipeline_parallel_config,
)


VALID_SELECTIVE_CHECKPOINTS = {
    "off",
    "adapter_aware",
    "every_other",
    "mlp_only",
    "mlp_layer1_only",
    "peak_blocks_adapter_aware",
    "peak_blocks_mlp",
    "peak_blocks_mlp_layer1",
}
#: ``sdpa``/``cudnn``/``cudnn_sdpa`` 与 ``normalize_krea2_attention_mode`` 同口径
#: 都是 ``torch`` 的别名（cli_entry 会把 sdpa 归一成 torch），预检必须放行。
KREA2_ATTN_MODES = {"", "torch", "flash", "sdpa", "cudnn", "cudnn_sdpa"}
KREA2_SELECTIVE_CHECKPOINTS = {"off", "every_other"}
#: sm>=80 有原生 BF16 Tensor Core。``nf4_dir1`` / ``krea2_attn_fp16`` 只服务
#: sm<80（sm70 V100、sm75 2080Ti/T4），与 ``set_dir1_enabled`` /
#: ``set_krea2_attn_fp16`` 的运行时抛错条件一一对应。
NATIVE_BF16_TENSOR_CORE_MIN_MAJOR = 8
#: 两个"内层 fp16"实验开关的运行时门禁文案；preflight 只复用这里的判断。
#: ``nf4_dir1`` 无 CUDA 时运行时直接抛 RuntimeError（error）；``krea2_attn_fp16``
#: 无 CUDA 时只是把全局 flag 置真后空转（runtime gate 不抛），故 preflight 只给
#: warning，避免把「Web 进程看不到 CUDA」误判成训练会崩。架构读不到时两者都只
#: 给 warning（无法证明 sm70/sm75，运行时门禁仍然在 set_* 处 fail-closed）。
_FP16_INNER_PATH_GATES: dict[str, dict[str, str]] = {
    "nf4_dir1": {
        "no_cuda": "set_dir1_enabled(True) raises RuntimeError without a CUDA device",
        "no_cuda_severity": "error",
        "newer_arch": "set_dir1_enabled(True) raises ValueError",
    },
    "krea2_attn_fp16": {
        "no_cuda": (
            "set_krea2_attn_fp16(True) only sets the global flag and is a no-op "
            "without CUDA"
        ),
        "no_cuda_severity": "warning",
        "newer_arch": "set_krea2_attn_fp16(True) raises ValueError",
    },
}
#: ``attn_mode`` 与 cli_entry / normalize_krea2_attention_mode 同口径的别名。
KREA2_ATTENTION_TORCH_ALIASES = {
    "sdpa": "torch",
    "cudnn": "torch",
    "cudnn_sdpa": "torch",
}
#: 让 ``krea2_attn_fp16`` 空转的非 bf16 计算精度（q.dtype != bfloat16）。
NON_BF16_COMPUTE_PRECISIONS = {"fp16", "no", "fp32"}
#: 会触发第 5 类告警的 inert 原因集合（flash 死开关 / 非 bf16 compute）。
_FP16_ATTENTION_INERT_REASONS = frozenset(
    {"flash"} | {f"compute={item}" for item in NON_BF16_COMPUTE_PRECISIONS}
)
KREA2_UNSUPPORTED_ADAPTER_FLAGS = (
    "use_ip_adapter",
    "use_easycontrol",
    "use_byg",
    "use_lokr",
    "use_loha",
    "use_glora",
    "use_vera",
    "use_ortho",
    "use_chimera_hydra",
    "use_timestep_mask",
    "add_reft",
    "train_llm_adapter",
)


@dataclass(frozen=True)
class TrainingCompatIssue:
    code: str
    key: str
    message: str


@dataclass(frozen=True)
class TrainingCompatMutation:
    code: str
    key: str
    value: Any
    message: str


@dataclass(frozen=True)
class TrainingCompatResult:
    errors: tuple[TrainingCompatIssue, ...]
    warnings: tuple[TrainingCompatIssue, ...]
    mutations: tuple[TrainingCompatMutation, ...]
    #: 非阻断建议（Web 预检渲染为 info）。默认空元组，保持既有调用方兼容。
    infos: tuple[TrainingCompatIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


class _CompatBuilder:
    def __init__(self) -> None:
        self.errors: list[TrainingCompatIssue] = []
        self.warnings: list[TrainingCompatIssue] = []
        self.mutations: list[TrainingCompatMutation] = []
        self.infos: list[TrainingCompatIssue] = []

    def error(self, code: str, key: str, message: str) -> None:
        self.errors.append(TrainingCompatIssue(code, key, message))

    def warning(self, code: str, key: str, message: str) -> None:
        self.warnings.append(TrainingCompatIssue(code, key, message))

    def info(self, code: str, key: str, message: str) -> None:
        self.infos.append(TrainingCompatIssue(code, key, message))

    def mutate(self, code: str, key: str, value: Any, message: str) -> None:
        self.mutations.append(TrainingCompatMutation(code, key, value, message))

    def build(self) -> TrainingCompatResult:
        return TrainingCompatResult(
            errors=tuple(self.errors),
            warnings=tuple(self.warnings),
            mutations=tuple(self.mutations),
            infos=tuple(self.infos),
        )


def _get(config: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _with_network_args(config: Mapping[str, Any] | object) -> Mapping[str, Any] | object:
    """Overlay network_args using the same precedence as training bootstrap."""
    if isinstance(config, Mapping):
        values = dict(config)
    else:
        try:
            values = vars(config).copy()
        except TypeError:
            return config
    arguments = values.get("network_args")
    if not isinstance(arguments, (list, tuple)):
        return values
    for argument in arguments:
        if not isinstance(argument, str) or "=" not in argument:
            continue
        key, value = argument.split("=", 1)
        if key not in {"model_family", "network_module"}:
            values[key] = value
    return values


def _bool_value(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", "none", ""}:
        return False
    return fallback


def _int_value(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _float_value(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _nested_caption_dropout_enabled(config: Mapping[str, Any] | object) -> bool:
    if not isinstance(config, Mapping):
        return False
    datasets = config.get("datasets")
    if not isinstance(datasets, (list, tuple)):
        return False
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            continue
        if _float_value(dataset.get("caption_dropout_rate"), 0.0) > 0:
            return True
        subsets = dataset.get("subsets")
        if not isinstance(subsets, (list, tuple)):
            continue
        if any(
            isinstance(subset, Mapping)
            and _float_value(subset.get("caption_dropout_rate"), 0.0) > 0
            for subset in subsets
        ):
            return True
    return False


def _normalized_capability(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        major, minor = value
        return int(major), int(minor)
    except (TypeError, ValueError):
        return None


def _fp16_inner_path_hardware_verdict(
    *,
    cuda_available: bool | None,
    capability: tuple[int, int] | None,
    selected_capabilities: tuple[tuple[int, int], ...],
    selection_unresolved: bool,
) -> tuple[str, tuple[int, int] | None]:
    """Collapse the caller's hardware context into one verdict.

    ``selected_capabilities`` describes every GPU the training subprocess will
    actually see (physical ``CUDA_VISIBLE_DEVICES`` indices). All of them must
    be sm<80; one sm80+ device is enough to make the switch raise at runtime.
    """
    if selected_capabilities:
        offending = [
            item
            for item in selected_capabilities
            if item[0] >= NATIVE_BF16_TENSOR_CORE_MIN_MAJOR
        ]
        if offending:
            return "newer_arch", offending[0]
        if selection_unresolved:
            # 白名单里还有设备没解析出来：已知的都是 sm<80，但不能证明全部，
            # 按「无法确认」降级为 warning，而不是假装放行。
            return "unresolved", selected_capabilities[0]
        return "ok", selected_capabilities[0]
    if selection_unresolved:
        return "unresolved", None
    if cuda_available is None and capability is None:
        return "unknown_context", None
    if cuda_available is False:
        return "no_cuda", None
    if capability is None:
        return "unresolved", None
    if capability[0] >= NATIVE_BF16_TENSOR_CORE_MIN_MAJOR:
        return "newer_arch", capability
    return "ok", capability


def _fp16_inner_path_issue(
    flag: str,
    verdict: str,
    offending: tuple[int, int] | None,
    *,
    selected_capabilities: tuple[tuple[int, int], ...] = (),
) -> tuple[bool, TrainingCompatIssue] | None:
    """Build the issue for one switch; ``(severity_is_error, issue)`` or ``None``.

    Severity mirrors the runtime gate: sm80+ raises for both switches, a missing
    CUDA device raises only for ``nf4_dir1`` and is a silent no-op for
    ``krea2_attn_fp16``, and an unreadable architecture is never a hard block
    because the runtime gate still fails closed on its own.
    """
    if verdict == "ok" or verdict == "unknown_context":
        return None
    gates = _FP16_INNER_PATH_GATES[flag]
    if verdict == "newer_arch":
        major, minor = offending or (0, 0)
        if selected_capabilities:
            device_detail = "selected device(s) " + ", ".join(
                f"sm{item[0]}{item[1]}" for item in selected_capabilities
            )
        else:
            device_detail = f"detected device sm{major}{minor}"
        return True, TrainingCompatIssue(
            f"{flag}_requires_pre_ampere",
            flag,
            f"--{flag} only targets sm70/sm75 devices (V100 / 2080Ti / T4) "
            f"without native BF16 tensor cores; {device_detail} already runs "
            f"BF16 on tensor cores and {gates['newer_arch']}. Training start is "
            "blocked instead of silently downgrading; keep the flag off on sm80+ "
            "to use native BF16.",
        )
    if verdict == "no_cuda":
        severity_is_error = gates["no_cuda_severity"] == "error"
        return severity_is_error, TrainingCompatIssue(
            f"{flag}_requires_cuda",
            flag,
            f"--{flag} only targets sm70/sm75 devices (V100 / 2080Ti / T4) "
            f"without native BF16 tensor cores and needs a CUDA device; "
            f"{gates['no_cuda']}. "
            + (
                "Training start is blocked instead of silently downgrading; "
                "keep the flag off."
                if severity_is_error
                else "Preflight cannot give the switch its SDPA kernels, so "
                "treat it as disabled; the runtime flag stays a silent no-op."
            ),
        )
    return False, TrainingCompatIssue(
        f"{flag}_architecture_unknown",
        flag,
        f"--{flag} could not confirm a sm70/sm75 device: the compute capability "
        "of the training device could not be resolved. Preflight keeps going "
        "with a warning; the runtime set_* gate still fails closed if the "
        "device turns out to be sm80+.",
    )


def _normalized_krea2_attention_mode(config: Mapping[str, Any] | object) -> str:
    """Normalize ``attn_mode`` the same way the runtime does (aliases → torch)."""
    raw_mode = str(_get(config, "attn_mode", "") or "").strip().lower()
    return KREA2_ATTENTION_TORCH_ALIASES.get(
        raw_mode.replace("-", "_"), raw_mode
    )


def _krea2_attn_fp16_inert_reason(config: Mapping[str, Any] | object) -> str | None:
    """Return why ``krea2_attn_fp16`` cannot take effect, or ``None``.

    Runtime consumers of ``_ATTN_FP16_ENABLED`` are the cuDNN SDPA branch only
    (``attention_backend._cudnn_attention``: ``if _ATTN_FP16_ENABLED and
    q.dtype == torch.bfloat16``). ``mode="flash"`` returns early through the
    packed Flash varlen path in its own dtype, and a non-bf16 compute dtype
    skips the branch as well. An unset ``attn_mode`` is treated as unspecified
    here (production configs always inherit ``flash`` from ``base.toml``).
    """
    mode = _normalized_krea2_attention_mode(config)
    if mode == "flash":
        return "flash"
    if mode not in {"", "torch"}:
        # Unsupported values already fail with krea2_invalid_attn_mode; never
        # recommend the pairing switch on top of an unloadable configuration.
        return "invalid"
    mixed_precision = (
        str(_get(config, "mixed_precision", "") or "").strip().lower()
    )
    if mixed_precision in NON_BF16_COMPUTE_PRECISIONS:
        return f"compute={mixed_precision}"
    return None


def _check_fp16_inner_path_flags(
    out: _CompatBuilder,
    config: Mapping[str, Any] | object,
    *,
    family_name: str,
    nf4_active: bool,
    cuda_available: bool | None,
    device_capability: tuple[int, int] | None,
    selected_device_capabilities: tuple[tuple[int, int], ...] | None = None,
    selection_unresolved: bool = False,
) -> None:
    """Check ``nf4_dir1`` / ``krea2_attn_fp16`` against runtime semantics.

    Five distinct branches, aligned with ``set_dir1_enabled``,
    ``set_krea2_attn_fp16`` and ``run_krea2_attention``:

    1. blocker: either switch on with a sm80+ device, or ``nf4_dir1`` without a
       CUDA device (its runtime gate raises); ``krea2_attn_fp16`` without CUDA
       and an unresolvable architecture only warn;
    2. warning: ``nf4_dir1`` without ``base_compute=nf4`` (no NF4 Linear);
    3. warning: ``krea2_attn_fp16`` outside ``model_family=krea2_raw``;
    4. info: sm70/sm75 + Krea-2 + NF4 with only one of the two switches on;
    5. warning: ``krea2_attn_fp16`` with ``attn_mode=flash`` or non-bf16
       compute, where the runtime never reads the switch (dead switch).
    """
    nf4_dir1 = _bool_value(_get(config, "nf4_dir1", False), False)
    attn_fp16 = _bool_value(_get(config, "krea2_attn_fp16", False), False)
    # 双关也不是无事可做：Krea-2 + NF4 在 sm<80 上用默认 bf16 时，每个 bf16 GEMM
    # 都落到 MAGMA 软件模拟（V100 实测 44.8 vs 11.06 s/step），需要一条 info 告诉
    # 用户有可开的内层 fp16 路径。其它组合仍然早退，避免为一条提示让所有训练都去
    # 探测 GPU。
    if (
        not nf4_dir1
        and not attn_fp16
        and (family_name != "krea2_raw" or not nf4_active)
    ):
        return
    capability = _normalized_capability(device_capability)
    selected_capabilities = tuple(
        item
        for item in (_normalized_capability(entry) for entry in selected_device_capabilities or ())
        if item is not None
    )
    verdict, offending = _fp16_inner_path_hardware_verdict(
        cuda_available=cuda_available,
        capability=capability,
        selected_capabilities=selected_capabilities,
        selection_unresolved=selection_unresolved,
    )

    blocked: set[str] = set()
    for flag, enabled in (("nf4_dir1", nf4_dir1), ("krea2_attn_fp16", attn_fp16)):
        if not enabled:
            continue
        issue = _fp16_inner_path_issue(
            flag,
            verdict,
            offending,
            selected_capabilities=selected_capabilities,
        )
        if issue is None:
            continue
        severity_is_error, item = issue
        if severity_is_error:
            out.error(item.code, item.key, item.message)
            blocked.add(flag)
        else:
            out.warning(item.code, item.key, item.message)

    # Branch 5: the runtime only consumes the attention switch inside the cuDNN
    # SDPA branch for bf16 inputs. Flash / non-bf16 compute makes it a dead
    # switch, so warn instead of silently accepting it (and never recommend
    # turning it on in that configuration).
    inert_reason = _krea2_attn_fp16_inert_reason(config)
    if (
        attn_fp16
        and inert_reason in _FP16_ATTENTION_INERT_REASONS
        and "krea2_attn_fp16" not in blocked
    ):
        if inert_reason == "flash":
            message = (
                "--krea2_attn_fp16 only takes effect on the Krea-2 attention "
                "path with attn_mode=torch/sdpa (cuDNN SDPA) and bf16 compute; "
                "attn_mode='flash' runs the packed Flash varlen path in its own "
                "dtype and never reads the switch, so enabling it is a no-op. "
                "Set attn_mode=torch (sdpa is a torch alias) and keep "
                "mixed_precision=bf16 to get the inner fp16 SDPA with GQA "
                "expansion."
            )
        else:
            message = (
                "--krea2_attn_fp16 only takes effect on the Krea-2 attention "
                "path with attn_mode=torch/sdpa (cuDNN SDPA) and bf16 compute; "
                f"current mixed_precision={inert_reason.split('=', 1)[-1]!r} "
                "already runs attention outside bf16, and the switch only "
                "rewrites bf16 inputs, so it is a no-op. Keep "
                "mixed_precision=bf16 to use it or turn the switch off."
            )
        out.warning(
            "krea2_attn_fp16_requires_torch_attention",
            "krea2_attn_fp16",
            message,
        )

    # Only mention "ignored" when the combination is actually reachable; a
    # blocked flag would otherwise produce a second, useless message.
    if nf4_dir1 and not nf4_active and "nf4_dir1" not in blocked:
        out.warning(
            "nf4_dir1_ignored",
            "nf4_dir1",
            "--nf4_dir1 only takes effect with --base_compute nf4; there is no "
            "NF4 Linear to accelerate under current base_compute. Drop the flag "
            "or switch to nf4.",
        )
    if attn_fp16 and family_name != "krea2_raw" and "krea2_attn_fp16" not in blocked:
        out.warning(
            "krea2_attn_fp16_ignored",
            "krea2_attn_fp16",
            "--krea2_attn_fp16 only takes effect on the Krea-2 attention path "
            f"(model_family=krea2_raw); current model_family={family_name!r} "
            "never runs Krea-2 attention, so the flag is ignored. Drop the flag "
            "or switch the family.",
        )

    supported_architecture = verdict == "ok" and not blocked
    if (
        supported_architecture
        and family_name == "krea2_raw"
        and nf4_active
        and nf4_dir1 != attn_fp16
        # Never recommend the attention switch when the current attention
        # configuration makes it dead (flash / non-bf16 compute).
        and inert_reason is None
    ):
        enabled_flag, missing_flag = (
            ("nf4_dir1", "krea2_attn_fp16") if nf4_dir1 else ("krea2_attn_fp16", "nf4_dir1")
        )
        representative = offending or capability or (0, 0)
        if selected_capabilities:
            device_detail = "/".join(
                f"sm{major}{minor}" for major, minor in selected_capabilities
            )
            lead = (
                f"Training devices {device_detail} have no native BF16 "
                "tensor cores."
            )
        else:
            lead = (
                f"sm{representative[0]}{representative[1]} has no native BF16 "
                "tensor cores."
            )
        out.info(
            "nf4_fp16_inner_pairing",
            missing_flag,
            f"{lead} Under Krea-2 + "
            "base_compute=nf4, --nf4_dir1 covers the NF4 Linear inner GEMM in "
            "fp16 and --krea2_attn_fp16 covers the attention inner SDPA in fp16 "
            f"(GQA expanded to MHA); only --{enabled_flag} is enabled. Enable "
            "both for the complete inner-fp16 path.",
        )

    # 双关：Krea-2 + NF4 在 sm<80 上用默认 bf16，两个内层 fp16 开关都没开。
    # 这不是错误（bf16 数值正确），但每个 bf16 GEMM 都会落到 MAGMA 软件模拟，
    # V100 实测 44.8 vs 11.06 s/step。只做 info，不做 warning：bf16 是 base.toml
    # 的生产默认，把它标成"有问题的配置"会制造噪音。
    # 也绝不建议把 compute 改成 fp16 —— 那条路已被
    # krea2_nf4_fp16_grad_underflow_20261108.md 的 35.65% 梯度偏差否决。
    if (
        supported_architecture
        and family_name == "krea2_raw"
        and nf4_active
        and not nf4_dir1
        and not attn_fp16
        # 只在 bf16 compute 下提示：两个开关都要求 compute=bf16，用户已经是
        # fp16/fp32 时建议开启它们同样不生效，属于误导。
        and str(_get(config, "mixed_precision", "bf16") or "bf16").strip().lower()
        == "bf16"
    ):
        device_detail = "/".join(
            f"sm{major}{minor}" for major, minor in selected_capabilities
        ) or f"sm{(offending or capability or (0, 0))[0]}{(offending or capability or (0, 0))[1]}"
        attention_hint = (
            " --krea2_attn_fp16 additionally moves the attention inner SDPA to "
            "fp16 (GQA expanded to MHA) and needs attn_mode=torch/sdpa; it is "
            "inert under attn_mode=flash."
            if inert_reason is None
            else ""
        )
        out.info(
            "nf4_fp16_inner_available",
            "nf4_dir1",
            f"Training devices {device_detail} have no native BF16 tensor "
            "cores, so every bf16 GEMM falls back to a software emulation path "
            "(V100 measured 44.8 vs 11.06 s/step). Neither inner-fp16 switch is "
            "enabled. Enabling --nf4_dir1 moves the NF4 Linear inner GEMM to "
            f"fp16 (about 4x on those GEMMs).{attention_hint} Both keep "
            "compute_dtype=bfloat16; do not switch mixed_precision to fp16 "
            "instead — that path is rejected for gradient underflow.",
        )


def _check_qwen_image_2_1_contract(out: _CompatBuilder, config, *,
                                   selective_checkpoint: str, blocks_to_swap: int,
                                   torch_compile: bool, cpu_offload_checkpointing: bool,
                                   unsloth_offload_checkpointing: bool) -> None:
    def forbid(flag: str, active: bool, message: str) -> None:
        if active:
            out.error(f"qwen_image_2_1_{flag}", flag, message)

    task = str(_get(config, "qwen_image_2_1_task", "t2i") or "t2i").strip().lower()
    family = "Qwen Image 2.1 Edit adapter" if task == "edit" else "Qwen Image 2.1 T2I adapter"
    forbid("mixed_precision", str(_get(config, "mixed_precision", "bf16") or "bf16").lower() != "bf16",
           f"{family} requires mixed_precision=bf16")
    forbid("base_compute", str(_get(config, "base_compute", "bf16") or "bf16").lower() != "bf16",
           f"{family} does not support NF4/ConvRot; use base_compute=bf16")
    forbid("adaptive_precision", str(_get(config, "adaptive_precision", "off") or "off").lower() != "off",
           f"{family} does not support adaptive precision")
    mode = str(_get(config, "attn_mode", "torch") or "torch").lower()
    forbid("attn_mode", mode not in {"torch", "sdpa", "flash"} or _bool_value(_get(config, "xformers")),
           f"{family} supports torch/sdpa and bf16 FlashAttention varlen")
    forbid("blocks_to_swap", blocks_to_swap > 30,
           f"{family} blocks_to_swap must be between 0 and 30")
    forbid("block_swap_checkpoint", blocks_to_swap > 0 and not _bool_value(_get(config, "gradient_checkpointing")),
           f"{family} block swap requires full gradient_checkpointing")
    forbid("compile_backend", torch_compile and str(_get(config, "dynamo_backend", "inductor") or "inductor") == "cudagraphs",
           f"{family} block compile does not support CUDA Graphs")
    forbid("compile_mode", torch_compile and _get(config, "compile_inductor_mode") not in (None, "default"),
           f"{family} block compile supports only default Inductor mode")
    forbid("auto_block_swap", _bool_value(_get(config, "auto_block_swap")),
           f"{family} AUTO block swap is not implemented")
    forbid("selective_checkpoint", selective_checkpoint != "off",
           f"{family} supports only full gradient checkpointing")
    forbid("cpu_offload_checkpointing", cpu_offload_checkpointing or unsloth_offload_checkpointing,
           f"{family} checkpoint offload is not implemented")
    forbid("network_train_unet_only", not _bool_value(_get(config, "network_train_unet_only"), True),
           f"{family} cannot train the text encoder")
    forbid("weighted_captions", _bool_value(_get(config, "weighted_captions")),
           f"{family} weighted captions are not implemented")
    forbid("caption_dropout_rate", _float_value(_get(config, "caption_dropout_rate")) > 0
           or _nested_caption_dropout_enabled(config),
           f"{family} requires caption_dropout_rate=0")
    forbid("masked_loss", _bool_value(_get(config, "masked_loss")),
           f"{family} masked loss is not implemented")
    forbid("loss_type", str(_get(config, "loss_type", "l2") or "l2").lower() != "l2",
           f"{family} supports L2 flow-matching loss only")
    forbid("timestep_sampling", str(_get(config, "timestep_sampling", "uniform") or "uniform").lower() != "uniform",
           f"{family} uses uniformly sampled dynamic-shift sigmas")
    forbid("weighting_scheme", str(_get(config, "weighting_scheme", "none") or "none").lower() != "none",
           f"{family} requires weighting_scheme=none")
    sampling = any(_bool_value(_get(config, key)) if key == "sample_at_first"
                   else _int_value(_get(config, key)) > 0
                   for key in ("sample_at_first", "sample_every_n_steps", "sample_every_n_epochs"))
    forbid("preview_text_cache", sampling and not _bool_value(_get(config, "cache_text_encoder_outputs")),
           f"{family} preview requires cached text encoder outputs")
    forbid("layer_start", _get(config, "layer_start") is not None or _get(config, "layer_end") is not None,
           f"{family} layer-range targeting is not implemented")
    forbid("t_min", _get(config, "t_min") is not None or _get(config, "t_max") is not None,
           f"{family} uses the full sigma range")


def check_training_compat(
    config: Mapping[str, Any] | object,
    *,
    world_size: int | None = None,
    cuda_available: bool | None = None,
    device_capability: tuple[int, int] | None = None,
    selected_device_capabilities: tuple[tuple[int, int], ...] | None = None,
    selection_unresolved: bool = False,
) -> TrainingCompatResult:
    """Validate optimization-flag combinations used by training and Web preflight.

    ``cuda_available`` / ``device_capability`` carry the caller's hardware
    context for the sm<80-only switches (``nf4_dir1`` / ``krea2_attn_fp16``).
    Leaving them ``None`` means "no hardware context": the architecture rule is
    skipped and the runtime ``set_*`` gate still fails closed.

    ``selected_device_capabilities`` is the capability of every GPU the
    training subprocess will actually see (physical ``CUDA_VISIBLE_DEVICES``
    indices); all of them must be sm<80. ``selection_unresolved`` marks a
    selection whose capability could not be resolved, which downgrades the
    architecture rule to a warning instead of a hard block.
    """

    out = _CompatBuilder()
    from library.quantization.qwen_nf4_compile_policy import qwen_nf4_compile_experiment_errors

    for message in qwen_nf4_compile_experiment_errors(_get(config, "model_family"), config):
        out.error("qwen_nf4_compile_experiment", "qwen_nf4_compile_experiment", message)
    if _get(config, "qwen_nf4_compile_experiment") is True and world_size not in (None, 1):
        out.error("qwen_nf4_compile_experiment", "qwen_nf4_compile_experiment", "single-process training required")
    config = _with_network_args(config)

    from library.training.auto_block_swap.config import configuration_errors

    for message in configuration_errors(config, world_size=world_size or 1):
        out.error("auto_block_swap_contract", "auto_block_swap", message)

    from library.training.adaptive_runtime.training_config import configuration_errors as precision_errors
    for message in precision_errors(config, world_size=world_size or 1):
        out.error("adaptive_precision_contract", "adaptive_precision", message)

    selective_checkpoint = (
        str(_get(config, "selective_checkpoint", "off") or "off").strip().lower()
    )
    gradient_checkpointing = _bool_value(_get(config, "gradient_checkpointing"), False)
    cpu_offload_checkpointing = _bool_value(
        _get(config, "cpu_offload_checkpointing"), False
    )
    unsloth_offload_checkpointing = _bool_value(
        _get(config, "unsloth_offload_checkpointing"), False
    )
    blocks_to_swap = _int_value(_get(config, "blocks_to_swap"), 0)
    if _bool_value(_get(config, "auto_block_swap"), False):
        blocks_to_swap = 1  # Check enabled-swap contracts before calibration.
    torch_compile = _bool_value(_get(config, "torch_compile"), False)
    use_lokr = _bool_value(_get(config, "use_lokr"), False)
    network_module = str(_get(config, "network_module", "") or "")
    functional_loss_weight = _float_value(_get(config, "functional_loss_weight"), 0.0)
    dynamo_backend = str(_get(config, "dynamo_backend", "") or "")
    compile_inductor_mode = _get(config, "compile_inductor_mode", None)
    model_family = str(_get(config, "model_family", "") or "anima").strip().lower()
    try:
        canonical_family = normalize_registered_family(
            model_family,
            source="training compatibility model_family",
            allow_aliases=True,
        )
        family_spec = get_model_family_spec(canonical_family)
    except ValueError as exc:
        out.error("invalid_model_family", "model_family", str(exc))
        return out.build()
    from library.training.task_contracts import check_training_task

    check_training_task(out, config, family_spec)
    krea2_family = family_spec.name == "krea2_raw"
    z_image_family = family_spec.name == "z_image"
    qwen_image_2_1_family = family_spec.name == "qwen_image_2_1"
    from library.models.qwen_image_2_1.preview_memory import configuration_errors as preview_memory_errors

    for key, message in preview_memory_errors(config, family=family_spec.name):
        code = "sample_vae_decode_contract" if key == "sample_vae_decode" else "qwen_preview_memory_contract"
        out.error(code, key, message)
    fused_projections = str(_get(config, "qwen_fused_projections", "off") or "off").strip().lower()
    saved_blocks = _get(config, "qwen_saved_projection_blocks", 0) or 0
    projection_budget = _get(config, "qwen_projection_budget_mib", 0) or 0
    if not isinstance(saved_blocks, int) or isinstance(saved_blocks, bool) or saved_blocks < 0:
        out.error("invalid_qwen_saved_projection_blocks", "qwen_saved_projection_blocks", "Expected a nonnegative integer")
    if not isinstance(projection_budget, int) or isinstance(projection_budget, bool) or projection_budget < 0:
        out.error("invalid_qwen_projection_budget_mib", "qwen_projection_budget_mib", "Expected a nonnegative integer")
    if isinstance(saved_blocks, int) and isinstance(projection_budget, int) and (
        bool(saved_blocks) != bool(projection_budget) or (saved_blocks > 0 and fused_projections != "all")
    ):
        out.error("qwen_saved_projection_contract", "qwen_saved_projection_blocks", "Saved projections require mode=all, a positive block count and MiB budget")
    if fused_projections not in {"off", "mlp", "qkv", "all"}:
        out.error("invalid_qwen_fused_projections", "qwen_fused_projections", "Expected off, mlp, qkv, or all")
    elif fused_projections != "off":
        if not qwen_image_2_1_family:
            out.error("qwen_fused_projections_family", "qwen_fused_projections", "Projection packing requires qwen_image_2_1")
        elif _bool_value(_get(config, "use_lokr")) or _bool_value(_get(config, "use_loha")):
            out.error("qwen_fused_projections_plain_lora", "qwen_fused_projections", "Projection packing requires plain LoRA; LoKr and LoHa are unsupported")
        for key, active, detail in (
            ("mixed_precision", str(_get(config, "mixed_precision", "bf16") or "bf16").lower() != "bf16", "BF16 mixed precision"),
            ("base_compute", str(_get(config, "base_compute", "bf16") or "bf16").lower() != "bf16", "BF16 base compute"),
            ("block_swap_transfer_dtype", str(_get(config, "block_swap_transfer_dtype", "bf16") or "bf16").lower() != "bf16", "BF16 block-swap transfer"),
            ("gradient_checkpointing", not gradient_checkpointing, "full gradient checkpointing"),
            ("selective_checkpoint", selective_checkpoint != "off", "selective checkpoint off"),
            ("cpu_offload_checkpointing", cpu_offload_checkpointing or unsloth_offload_checkpointing, "checkpoint offload disabled"),
        ):
            if active:
                out.error("qwen_fused_projections_" + key, key, "Experimental projection packing requires " + detail)
    pipeline_parallel_error: ValueError | None = None
    try:
        pipeline_parallel = PipelineParallelConfig.from_config(config).enabled
    except ValueError as exc:
        pipeline_parallel = False
        pipeline_parallel_error = exc
    compile_dynamic_seq = _bool_value(_get(config, "compile_dynamic_seq"), False)
    compile_seq_bands = _bool_value(_get(config, "compile_seq_bands"), False)
    v100_flash_stability = (
        str(_get(config, "v100_flash_stability", "off") or "off").strip().lower()
    )

    if pipeline_parallel_error is not None:
        out.error(
            "pipeline_parallel_config",
            "pipeline_parallel",
            str(pipeline_parallel_error),
        )
    elif pipeline_parallel:
        try:
            validate_pipeline_parallel_config(config, world_size=world_size)
        except ValueError as exc:
            out.error(
                "pipeline_parallel_config",
                "pipeline_parallel",
                str(exc),
            )
        else:
            # Planning support is not a runtime implementation. Keep every
            # family fail-closed until stage-local placement, 1F1B, optimizer
            # ownership, and checkpoint recovery are wired into the trainer.
            out.error(
                "pipeline_parallel_runtime_unavailable",
                "pipeline_parallel",
                f"{family_spec.display_name} pipeline_parallel has a validated "
                "stage planner, but the 1F1B schedule is not wired into the "
                "main trainer; disable pipeline_parallel or use the standalone "
                "PP probe.",
            )

    # Krea-2 gets a family-specific message/mutation below. Avoid emitting a
    # duplicate generic mutation when its fixed-padded gate is the real cause.
    if (
        compile_seq_bands
        and not compile_dynamic_seq
        and family_spec.name == "anima"
    ):
        message = (
            "compile_seq_bands only applies with compile_dynamic_seq; "
            "compile_seq_bands will be disabled."
        )
        out.warning(
            "compile_seq_bands_requires_dynamic_seq",
            "compile_seq_bands",
            message,
        )
        out.mutate(
            "compile_seq_bands_requires_dynamic_seq",
            "compile_seq_bands",
            False,
            message,
        )

    if selective_checkpoint not in VALID_SELECTIVE_CHECKPOINTS:
        out.error(
            "invalid_selective_checkpoint",
            "selective_checkpoint",
            "--selective_checkpoint must be one of: "
            + ", ".join(sorted(VALID_SELECTIVE_CHECKPOINTS)),
        )

    if family_spec.plain_lora_only:
        adapter_conflicts = []
        network_spec_name = None
        if network_module not in {"", "networks.lora_anima"}:
            adapter_conflicts.append(f"network_module={network_module!r}")
        if family_spec.supported_network_specs is not None:
            from networks.registry import resolve_network_spec

            try:
                network_spec = resolve_network_spec(config)
            except ValueError as exc:
                adapter_conflicts.append(f"network_args ({exc})")
            else:
                network_spec_name = network_spec.name
                if network_spec.name not in family_spec.supported_network_specs:
                    adapter_conflicts.append(f"network_spec={network_spec.name!r}")
        allowed_flags = {"use_lokr", "use_loha"} if qwen_image_2_1_family else set()
        adapter_conflicts.extend(
            flag
            for flag in KREA2_UNSUPPORTED_ADAPTER_FLAGS
            if flag not in allowed_flags
            if _bool_value(_get(config, flag), False)
        )
        qwen_task = str(_get(config, "qwen_image_2_1_task", "t2i") or "t2i").strip().lower()
        if qwen_image_2_1_family and qwen_task == "edit":
            if network_spec_name in {"lokr", "loha"} or any(
                _bool_value(_get(config, flag), False) for flag in ("use_lokr", "use_loha")
            ):
                adapter_conflicts.append("qwen_image_2_1_task=edit requires plain LoRA")
        moe_style = (
            str(_get(config, "use_moe_style", "false") or "false").strip().lower()
        )
        if moe_style not in {"", "0", "false", "none", "off"}:
            adapter_conflicts.append(f"use_moe_style={moe_style!r}")
        if _bool_value(_get(config, "route_per_layer"), False):
            adapter_conflicts.append("route_per_layer")
        router_source = (
            str(_get(config, "router_source", "none") or "none").strip().lower()
        )
        if router_source != "none":
            adapter_conflicts.append(f"router_source={router_source!r}")
        if _bool_value(_get(config, "dora_wd"), False):
            adapter_conflicts.append("dora_wd")
        if _int_value(_get(config, "step_expert_K"), 0) > 1:
            adapter_conflicts.append("step_expert_K")
        if _bool_value(_get(config, "train_adaln"), False):
            adapter_conflicts.append("train_adaln")
        if _int_value(_get(config, "num_registers"), 0) > 0:
            adapter_conflicts.append("num_registers")
        if functional_loss_weight > 0:
            adapter_conflicts.append("functional_loss_weight")
        if adapter_conflicts:
            out.error(
                ("krea2_plain_lora_only" if krea2_family else "family_plain_lora_only"),
                "network_module",
                f"{family_spec.display_name} training supports only "
                + ("plain LoRA, LoKr, or LoHa" if qwen_image_2_1_family else "plain LoRA")
                + "; unsupported: " + ", ".join(adapter_conflicts),
            )

    if family_spec.name == "moss_image":
        from library.training.moss_compat import check_moss_training_contract

        check_moss_training_contract(out, config)

    if qwen_image_2_1_family:
        _check_qwen_image_2_1_contract(
            out, config,
            selective_checkpoint=selective_checkpoint,
            blocks_to_swap=blocks_to_swap,
            torch_compile=torch_compile,
            cpu_offload_checkpointing=cpu_offload_checkpointing,
            unsloth_offload_checkpointing=unsloth_offload_checkpointing,
        )

    if krea2_family:
        krea2_attn_mode = str(_get(config, "attn_mode", "") or "").strip().lower()
        if krea2_attn_mode not in KREA2_ATTN_MODES:
            out.error(
                "krea2_invalid_attn_mode",
                "attn_mode",
                "Krea-2 attn_mode supports only torch or flash (sdpa is a "
                f"torch alias); got {krea2_attn_mode!r}.",
            )
        if compile_dynamic_seq:
            message = (
                "Krea-2 compile uses two fixed padded token-family graphs; "
                "compile_dynamic_seq will be disabled."
            )
            out.warning(
                "krea2_compile_dynamic_seq",
                "compile_dynamic_seq",
                message,
            )
            out.mutate(
                "krea2_compile_dynamic_seq",
                "compile_dynamic_seq",
                False,
                message,
            )
        if compile_seq_bands:
            message = (
                "Krea-2 compile uses fixed padded token-family graphs; "
                "compile_seq_bands will be disabled."
            )
            out.warning(
                "krea2_compile_seq_bands",
                "compile_seq_bands",
                message,
            )
            out.mutate(
                "krea2_compile_seq_bands",
                "compile_seq_bands",
                False,
                message,
            )
        normalized_inductor_mode = (
            str(compile_inductor_mode or "default").strip().lower()
        )
        if normalized_inductor_mode != "default":
            out.error(
                "krea2_compile_inductor_mode",
                "compile_inductor_mode",
                "Krea-2 compile supports only compile_inductor_mode=default; "
                f"got {compile_inductor_mode!r}.",
            )
        if selective_checkpoint not in KREA2_SELECTIVE_CHECKPOINTS:
            out.error(
                "krea2_selective_checkpoint",
                "selective_checkpoint",
                "Krea-2 selective_checkpoint supports only off or every_other; "
                f"got {selective_checkpoint!r}.",
            )
        if v100_flash_stability != "off":
            out.error(
                "krea2_v100_flash_stability",
                "v100_flash_stability",
                "v100_flash_stability is Anima-only and must be off for Krea-2.",
            )

    if z_image_family:
        if compile_seq_bands:
            message = (
                "compile_seq_bands is Anima-only; Z-Image does not use "
                "native-flatten dynamic sequence bands, so it will be disabled."
            )
            out.warning(
                "z_image_compile_seq_bands",
                "compile_seq_bands",
                message,
            )
            out.mutate(
                "z_image_compile_seq_bands",
                "compile_seq_bands",
                False,
                message,
            )
        attn_mode = str(_get(config, "attn_mode", "flash") or "flash").strip().lower()
        if attn_mode not in {"torch", "flash", "sdpa"} or _bool_value(
            _get(config, "xformers"), False
        ):
            out.error(
                "z_image_attention_mode",
                "attn_mode",
                "Z-Image supports attn_mode=torch or flash (sdpa is a torch "
                "alias); xformers must be disabled.",
            )
        mixed_precision = (
            str(_get(config, "mixed_precision", "bf16") or "bf16").strip().lower()
        )
        from library.training.adaptive_runtime.training_config import islands_enabled
        adaptive_islands = islands_enabled(config)
        if mixed_precision != "bf16" and not (
            (adaptive_islands and mixed_precision == "fp16" or mixed_precision == "no")
            and attn_mode in {"torch", "sdpa"}
        ):
            out.error(
                "z_image_bf16_only",
                "mixed_precision",
                "Z-Image v1 is validated with mixed_precision=bf16; adaptive "
                "FP16/FP32 or pure FP32 paths require torch/sdpa attention.",
            )
        base_compute = (
            str(_get(config, "base_compute", "bf16") or "bf16").strip().lower()
        )
        if base_compute != "bf16":
            out.error(
                "z_image_base_compute",
                "base_compute",
                "Z-Image v1 does not yet support NF4 or ConvRot base compute; use bf16.",
            )
        if torch_compile:
            out.error(
                "z_image_torch_compile",
                "torch_compile",
                "Z-Image torch.compile has not been validated; keep torch_compile=false.",
            )
        if blocks_to_swap > 28:
            out.error(
                "z_image_block_swap_range",
                "blocks_to_swap",
                "Z-Image has 30 main layers and requires at least two resident layers; "
                "use blocks_to_swap between 0 and 28.",
            )
        if selective_checkpoint != "off":
            out.error(
                "z_image_selective_checkpoint",
                "selective_checkpoint",
                "Z-Image v1 supports full gradient checkpointing only.",
            )
        if cpu_offload_checkpointing or unsloth_offload_checkpointing:
            out.error(
                "z_image_checkpoint_offload",
                "gradient_checkpointing",
                "Z-Image v1 does not support CPU/Unsloth activation offload.",
            )
        if (
            _get(config, "layer_start") is not None
            or _get(config, "layer_end") is not None
        ):
            out.error(
                "z_image_layer_range",
                "layer_start",
                "Z-Image layer_start/layer_end is not implemented for layers.N and "
                "refiner blocks; remove the layer range.",
            )
        if _bool_value(_get(config, "weighted_captions"), False):
            out.error(
                "z_image_weighted_captions",
                "weighted_captions",
                "Z-Image weighted captions are not implemented in v1.",
            )
        loss_type = str(_get(config, "loss_type", "l2") or "l2").strip().lower()
        if loss_type != "l2":
            out.error(
                "z_image_loss_type",
                "loss_type",
                "Z-Image v1 reproduces the official L2 flow-matching loss only.",
            )
        if _get(config, "t_min") is not None or _get(config, "t_max") is not None:
            out.error(
                "z_image_timestep_range",
                "t_min",
                "Z-Image v1 samples the complete official scheduler grid; "
                "t_min/t_max are not supported.",
            )
        sampler = str(_get(config, "sampler", "default") or "default").strip().lower()
        if sampler != "default":
            out.error(
                "z_image_training_sampler",
                "sampler",
                "Z-Image v1 supports only the official default training sampler.",
            )
        if v100_flash_stability != "off":
            out.error(
                "z_image_v100_flash_stability",
                "v100_flash_stability",
                "v100_flash_stability is Anima-only and must be off for Z-Image.",
            )
        flow_shift = _float_value(_get(config, "discrete_flow_shift"), 6.0)
        if abs(flow_shift - 6.0) > 1e-7:
            out.error(
                "z_image_flow_shift",
                "discrete_flow_shift",
                "Z-Image training requires the official discrete_flow_shift=6.0.",
            )
        timestep_sampling = (
            str(_get(config, "timestep_sampling", "uniform") or "uniform")
            .strip()
            .lower()
        )
        if timestep_sampling != "uniform":
            out.error(
                "z_image_timestep_sampling",
                "timestep_sampling",
                "Z-Image v1 uses the official shifted-uniform sigma schedule.",
            )
        weighting_scheme = (
            str(_get(config, "weighting_scheme", "none") or "none").strip().lower()
        )
        if weighting_scheme != "none":
            out.error(
                "z_image_weighting_scheme",
                "weighting_scheme",
                "Z-Image v1 supports weighting_scheme=none only.",
            )
        caption_dropout_rate = _float_value(_get(config, "caption_dropout_rate"), 0.0)
        if caption_dropout_rate > 0 or _nested_caption_dropout_enabled(config):
            out.error(
                "z_image_caption_dropout",
                "caption_dropout_rate",
                "Z-Image caption dropout requires cached empty-prompt embeddings and "
                "is not supported in v1.",
            )

    if blocks_to_swap < 0:
        out.error(
            "negative_blocks_to_swap",
            "blocks_to_swap",
            "blocks_to_swap must be greater than or equal to 0.",
        )

    block_swap_enabled = blocks_to_swap > 0
    selective_enabled = selective_checkpoint != "off"

    if selective_enabled and gradient_checkpointing:
        out.error(
            "selective_full_gradient_checkpointing",
            "gradient_checkpointing",
            "--selective_checkpoint is a selective DiT checkpoint mode; "
            "do not combine it with full --gradient_checkpointing.",
        )
    if selective_enabled and cpu_offload_checkpointing:
        out.error(
            "selective_cpu_offload",
            "cpu_offload_checkpointing",
            "--selective_checkpoint does not support CPU activation offload.",
        )
    if selective_enabled and unsloth_offload_checkpointing:
        out.error(
            "selective_unsloth_offload",
            "unsloth_offload_checkpointing",
            "--selective_checkpoint cannot be combined with "
            "--unsloth_offload_checkpointing.",
        )

    if block_swap_enabled and cpu_offload_checkpointing:
        out.error(
            "block_swap_cpu_offload",
            "cpu_offload_checkpointing",
            "blocks_to_swap supports standard gradient_checkpointing, but is "
            "not supported with cpu_offload_checkpointing",
        )

    if unsloth_offload_checkpointing:
        if not gradient_checkpointing and not selective_enabled:
            out.warning(
                "unsloth_enables_gradient_checkpointing",
                "gradient_checkpointing",
                "unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled",
            )
            out.mutate(
                "unsloth_enables_gradient_checkpointing",
                "gradient_checkpointing",
                True,
                "unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled",
            )
        if cpu_offload_checkpointing:
            out.error(
                "unsloth_cpu_offload",
                "cpu_offload_checkpointing",
                "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing",
            )
        if block_swap_enabled:
            out.error(
                "block_swap_unsloth_offload",
                "unsloth_offload_checkpointing",
                "blocks_to_swap supports standard gradient_checkpointing, but is "
                "not supported with unsloth_offload_checkpointing",
            )

    if cpu_offload_checkpointing and not gradient_checkpointing:
        out.warning(
            "cpu_offload_without_gradient_checkpointing",
            "cpu_offload_checkpointing",
            "cpu_offload_checkpointing only affects activation checkpointing; "
            "enable gradient_checkpointing or disable cpu_offload_checkpointing.",
        )

    if use_lokr and gradient_checkpointing and torch_compile:
        out.warning(
            "lokr_full_checkpoint_compile",
            "torch_compile",
            "LoKr with full gradient_checkpointing under torch_compile is "
            "experimental. Training keeps torch_compile enabled, pins larger "
            "Dynamo graph budgets, and traces the factorized LoKr forward "
            "in-graph; blocks_to_swap remains independent.",
        )

    base_compute = str(_get(config, "base_compute", "bf16") or "bf16").strip().lower()
    block_swap_transfer_dtype = (
        str(_get(config, "block_swap_transfer_dtype", "bf16") or "bf16").strip().lower()
    )
    convrot_active = base_compute in {
        "w8a16_convrot",
        "w8a8_convrot",
        "w8a16",
        "w8a8",
    }
    nf4_active = base_compute == "nf4"
    if convrot_active and block_swap_transfer_dtype in {"int8", "int8_linear", "i8"}:
        out.error(
            "convrot_block_swap_int8_mutex",
            "base_compute",
            "base_compute ConvRot paths are mutually exclusive with "
            "block_swap_transfer_dtype=int8 (double dequant / confused semantics). "
            "Use block_swap_transfer_dtype=bf16 when enabling ConvRot.",
        )
    # NF4 (Krea-2 QLoRA) 与 ConvRot 互斥: ConvRot 是 anima cross-attn/AdaLN 专属
    # 量化路径, NF4 是 Krea-2 single-stream 通用 bnb 4-bit, 两者架构假设不同.
    if nf4_active and convrot_active:
        out.error(
            "nf4_convrot_mutex",
            "base_compute",
            "base_compute=nf4 is mutually exclusive with ConvRot paths "
            "(w8a16_convrot/w8a8_convrot). NF4 is the Krea-2 QLoRA path; "
            "ConvRot is anima-specific. Pick one.",
        )
    # NF4 × block_swap 已验证通过 (方向 A: deepcopy master + Params4bit.to() 整体
    # 搬运, offloading.py isinstance 分流不碰 bf16/int8/fp8 路径). 端到端探针
    # (probe_nf4_blockswap.py, PG199, 1024, swap=4, 30 步): host RAM 18.18GB
    # (bf16 路径 22.64GB master 单项超它曾在 62GB 机宕机), GPU 10.26GB, loss
    # 0.0084->0.0017 单调下降, LoRA grad 非零, DiT frozen 不变. 保留 warning:
    # NF4 block swap 主战场是 host RAM (master 5.66GB) 而非 GPU, 提醒用户关注
    # pinned master 内存预算.
    if nf4_active and block_swap_enabled:
        out.warning(
            "nf4_block_swap_host_ram",
            "base_compute",
            "base_compute=nf4 + blocks_to_swap verified (offloader NF4 path "
            "via Params4bit integral transport, end-to-end probe green). "
            "Main constraint is host RAM: pinned NF4 masters ~5.7GB (bf16 "
            "path 22.64GB), watch pinned-master memory budget on low-RAM hosts.",
        )
    # nf4_prequantized_path: 只在 base_compute=nf4 时生效. 给了路径但 base_compute
    # 不是 nf4 → 路径被静默忽略, 提醒用户 (常见于从 NF4 配置切回 bf16 忘删路径).
    nf4_path = _get(config, "nf4_prequantized_path", None)
    if nf4_path and not nf4_active:
        out.warning(
            "nf4_path_ignored",
            "nf4_prequantized_path",
            "--nf4_prequantized_path only takes effect with --base_compute nf4; "
            "ignored under current base_compute. Drop the path or switch to nf4.",
        )
    # nf4_dir1 / krea2_attn_fp16: 两个"无原生 bf16 tensor core"内层 fp16 开关.
    # 架构 blocker、忽略提示与 sm70/sm75 双开建议统一在 helper 内处理, 条件与
    # set_dir1_enabled / set_krea2_attn_fp16 的运行时抛错严格对齐.
    _check_fp16_inner_path_flags(
        out,
        config,
        family_name=family_spec.name,
        nf4_active=nf4_active,
        cuda_available=cuda_available,
        device_capability=device_capability,
        selected_device_capabilities=selected_device_capabilities,
        selection_unresolved=selection_unresolved,
    )
    if base_compute not in {
        "bf16",
        "fp16",
        "none",
        "off",
        "w8a16_convrot",
        "w8a8_convrot",
        "w8a16",
        "w8a8",
        "nf4",
    }:
        out.error(
            "invalid_base_compute",
            "base_compute",
            f"unknown base_compute={base_compute!r}; expected bf16 | "
            "w8a16_convrot | w8a8_convrot | nf4",
        )

    # ConvRot on attention linears + flash + torch.compile can materialize
    # float32 Q/K/V into FlashAttention (FA only accepts fp16/bf16). The
    # dispatcher now hard-casts at the flash entry, so this is informational
    # rather than a hard block — still surface so operators know the combo is
    # on the known-risk path.
    attn_mode = str(_get(config, "attn_mode", "") or "").strip().lower()
    convrot_scope = str(_get(config, "convrot_scope", "mlp") or "mlp").strip().lower()
    convrot_touches_attn = False
    if convrot_active:
        scope_tokens = {
            item.strip() for item in convrot_scope.split(",") if item.strip()
        }
        attn_scope_tokens = {
            "all",
            "attention",
            "attn",
            "self",
            "self_attn",
            "self_attn_qkv",
            "self_qkv",
            "self_attn_out",
            "self_out",
            "cross",
            "cross_attn",
            "cross_attn_q",
            "cross_q",
            "cross_attn_kv",
            "cross_kv",
            "cross_attn_out",
            "cross_out",
            "attention_out",
            "attn_out",
        }
        convrot_touches_attn = bool(scope_tokens & attn_scope_tokens)
    if (
        convrot_active
        and convrot_touches_attn
        and attn_mode in {"flash", "flash4"}
        and torch_compile
    ):
        out.warning(
            "convrot_attn_flash_compile_dtype",
            "base_compute",
            "base_compute ConvRot with convrot_scope covering attention "
            f"({convrot_scope!r}) + attn_mode={attn_mode!r} + torch_compile can "
            "materialize float32 Q/K/V; FlashAttention only accepts fp16/bf16. "
            "The flash dispatcher hard-casts at entry as a safety net. Prefer "
            "convrot_scope=mlp to avoid the cast tax if attention quant is not required.",
        )

    if block_swap_enabled:
        if torch_compile and dynamo_backend == "cudagraphs":
            out.warning(
                "block_swap_cudagraphs_disable_compile",
                "torch_compile",
                "blocks_to_swap moves DiT block weights between CPU/GPU, "
                "so dynamo_backend='cudagraphs' is unsafe. Disabling torch_compile.",
            )
            out.mutate(
                "block_swap_cudagraphs_disable_compile",
                "torch_compile",
                False,
                "blocks_to_swap moves DiT block weights between CPU/GPU, "
                "so dynamo_backend='cudagraphs' is unsafe. Disabling torch_compile.",
            )
            torch_compile = False

        if torch_compile and compile_inductor_mode in {
            "reduce-overhead",
            "max-autotune",
        }:
            safe_mode = (
                "max-autotune-no-cudagraphs"
                if compile_inductor_mode == "max-autotune"
                else None
            )
            out.warning(
                "block_swap_compile_mode_cudagraphs",
                "compile_inductor_mode",
                "blocks_to_swap is incompatible with Inductor CUDAGraph modes "
                f"({compile_inductor_mode!r}); using {safe_mode or 'default'} mode instead.",
            )
            out.mutate(
                "block_swap_compile_mode_cudagraphs",
                "compile_inductor_mode",
                safe_mode,
                "blocks_to_swap is incompatible with Inductor CUDAGraph modes "
                f"({compile_inductor_mode!r}); using {safe_mode or 'default'} mode instead.",
            )

        if network_module == "networks.methods.soft_tokens":
            out.error(
                "block_swap_soft_tokens",
                "blocks_to_swap",
                "blocks_to_swap is not supported with soft_tokens. "
                "Keep blocks_to_swap=0 for this multi-forward method.",
            )
        if functional_loss_weight > 0.0:
            out.error(
                "block_swap_functional_loss",
                "blocks_to_swap",
                "blocks_to_swap is not supported with functional_loss_weight > 0. "
                "Disable block swap for postfix/functional multi-forward training.",
            )

    return out.build()


def apply_training_compat_mutations(
    target: Mapping[str, Any] | object,
    result: TrainingCompatResult,
) -> None:
    """Apply suggested mutations to a mutable mapping or argparse namespace."""

    for mutation in result.mutations:
        if isinstance(target, dict):
            target[mutation.key] = mutation.value
        else:
            setattr(target, mutation.key, mutation.value)
