"""Distributed training (single / DDP / PP) backend support for the WebUI.

本模块是「分布式训练」卡片的唯一后端服务面：

- :func:`normalize_distributed_mode`：三态模式归一（旧队列缺省 DDP 兼容）；
- :func:`wired_pp_families`：PP 已接线族表（前端据此置灰 PP 选项）；
- :func:`hardware_diagnosis`：PP 模式三层硬件风险（显存异构 / 低显存 /
  代际混搭），全部建议级，不阻挡启动；
- :func:`diagnose_pp_dataset_readiness`：PP 数据集就绪（多子集桶一致性 /
  batch_size 一致 / latent 与 TE 缓存完整），复用现有 census / cache audit；
- :func:`diagnose_distributed_training`：只读诊断端点入口，复用
  ``preflight`` 的配置装载 + ``check_training_compat`` + PP admission。

诊断结果三档：``level`` ∈ {``error``(🔴 必败) / ``warning``(🟡 风险) /
``info``(🔵 信息)}。前端不阻挡：启动路径的 fail-closed 仍由既有 admission
兜底，本模块只读不改任何配置。
"""

from __future__ import annotations

import subprocess
from typing import Any

from web.services.training.gpu import (
    classify_pp_hardware_risks,
    query_gpu_hardware_snapshot,
)

DISTRIBUTED_MODES = ("single", "ddp", "pp")
#: 旧队列/旧设置缺省兼容：无模式字段时按 DDP 处理。
DEFAULT_DISTRIBUTED_MODE = "ddp"


def normalize_distributed_mode(
    value: Any,
    *,
    default: str = DEFAULT_DISTRIBUTED_MODE,
) -> str:
    """Normalize a user/queue/config distributed mode value.

    缺失或空值回落到 ``default``（旧队列 resume 无模式字段 → DDP 兼容）；
    未知名一律 fail-closed 回落而不是猜测，非法值由调用方决定是否报错。
    """

    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in DISTRIBUTED_MODES:
        return text
    return default


def item_distributed_mode(item: dict[str, Any] | None) -> str:
    """Read a queue item's stored mode; old items without the field → DDP."""

    if not isinstance(item, dict):
        return DEFAULT_DISTRIBUTED_MODE
    return normalize_distributed_mode(item.get("distributed_mode"))


def wired_pp_families() -> list[dict[str, Any]]:
    """PP 已接线族表（供前端置灰未接线的 PP 选项）。"""

    from library.models.family_registry import model_family_capability_catalog
    from library.training.pp.gate import TRAINER_PP_WIRED_FAMILIES

    catalog = model_family_capability_catalog()
    families: list[dict[str, Any]] = []
    for item in catalog:
        pipeline = item.get("pipeline_parallel")
        if not pipeline:
            continue
        families.append({
            "family": item["name"],
            "display_name": item["display_name"],
            "wired": item["name"] in TRAINER_PP_WIRED_FAMILIES,
            "runtime_backend": pipeline.get("runtime_backend"),
            "stages": list(pipeline.get("stages") or []),
            "schedules": list(pipeline.get("schedules") or []),
        })
    return families


def hardware_diagnosis(
    mode: str,
    gpu_whitelist: Any = None,
    *,
    runner: Any = None,
) -> dict[str, Any]:
    """Hardware snapshot + suggestion-level risks (三层风险仅 PP 模式判定)."""

    from web.services.training.gpu import normalize_gpu_whitelist

    snapshot = query_gpu_hardware_snapshot(runner=runner)
    selected = normalize_gpu_whitelist(gpu_whitelist)
    scoped = snapshot
    if selected:
        selected_set = set(selected)
        scoped = [row for row in snapshot if row["index"] in selected_set]
    risks: list[dict[str, Any]] = []
    if mode == "pp":
        risks = classify_pp_hardware_risks(scoped)
    return {
        "gpus": scoped,
        "risks": risks,
    }


def _pp_check(
    level: str,
    key: str,
    message: str,
    *,
    detail: str | None = None,
) -> dict[str, str]:
    """One diagnosis row.

    ``message`` is always human wording for the panel; ``detail`` (optional)
    carries the raw error text so the UI can fold it behind "原始信息" instead
    of showing a traceback-shaped string as the primary line.
    """

    payload = {"level": level, "key": key, "code": f"pp_{key}", "message": message}
    if detail:
        payload["detail"] = detail
    return payload


def _row_batch_size(row: dict[str, Any], cfg: dict[str, Any]) -> int:
    """Per-subset batch size (settings → row → cfg train_batch_size → 1)."""

    settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    value = row.get("batch_size", settings.get("batch_size"))
    if value in (None, ""):
        value = cfg.get("train_batch_size")
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def _census_for_row(
    row: dict[str, Any],
    cfg: dict[str, Any],
    resolve_path: Any,
) -> dict[str, Any] | None:
    """Bucket census for one dataset subset row (read-only, reuse estimate_buckets)."""

    from web.services.config.estimate_buckets import inspect_estimate_buckets

    image_dir = resolve_path(str(row.get("image_dir") or row.get("source_dir") or ""))
    if not image_dir.is_dir():
        return None
    settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    family = str(cfg.get("model_family") or "anima")
    return inspect_estimate_buckets(
        image_dir,
        recursive=bool(row.get("recursive", True)),
        path_pattern=str(row.get("path_pattern") or "") or None,
        available=True,
        source_settings=settings,
        model_family=family,
    )


def _cache_checks_for_row(
    row: dict[str, Any],
    cfg: dict[str, Any],
    resolve_path: Any,
) -> list[dict[str, Any]]:
    """Latent / TE cache completeness audit for one row (reuse cache_audit)."""

    from web.services.config.cache_audit import audit_dataset_row_caches
    from web.services.config.dataset_media import _dataset_image_files
    from web.services.config.metadata import DATASET_IMAGE_EXTS

    image_dir = resolve_path(str(row.get("image_dir") or row.get("source_dir") or ""))
    if not image_dir.is_dir():
        return []
    try:
        images = _dataset_image_files(
            image_dir,
            set(DATASET_IMAGE_EXTS),
            recursive=bool(row.get("recursive", True)),
            path_pattern=str(row.get("path_pattern") or "") or None,
        )
    except ValueError:
        return []
    return audit_dataset_row_caches(cfg, row, images, resolve_path=resolve_path)


def diagnose_pp_dataset_readiness(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """PP 数据集就绪：多子集桶一致性 / batch_size 一致 / 缓存完整。

    全部复用现有实现（estimate_buckets census + cache_audit），只读。
    """

    from web.services.config.datasets import _dataset_rows_for_estimate
    from web.services.config.preflight_runtime import _resolve_project_path

    checks: list[dict[str, str]] = []
    rows = _dataset_rows_for_estimate(cfg)
    if not rows:
        checks.append(_pp_check(
            "info",
            "dataset_no_subset",
            "当前配置没有数据集子集，PP 数据集就绪诊断未执行。",
        ))
        return checks

    # 多子集桶一致性：PP wire 是固定形状，一次 PP 训练只允许一个桶。
    union_buckets: set[tuple[int, int]] = set()
    partial = False
    missing_dirs = 0
    for index, row in enumerate(rows, start=1):
        census = _census_for_row(row, cfg, _resolve_project_path)
        if census is None:
            missing_dirs += 1
            continue
        if census.get("status") != "ready":
            partial = True
        for bucket in census.get("buckets") or []:
            union_buckets.add((int(bucket["width"]), int(bucket["height"])))
    if len(rows) < 2:
        checks.append(_pp_check(
            "info",
            "dataset_single_subset",
            "单子集数据集：无多子集桶一致性问题。",
        ))
    if missing_dirs:
        checks.append(_pp_check(
            "warning",
            "dataset_dir_missing",
            f"{missing_dirs} 个子集目录不存在或不可读，桶一致性检测不完整。",
        ))
    if len(union_buckets) > 1:
        sample = "、".join(f"{w}x{h}" for w, h in sorted(union_buckets)[:5])
        checks.append(_pp_check(
            "error",
            "dataset_bucket_mismatch",
            f"多子集桶不一致（必败）：PP 训练一次只允许一个 token 桶，当前合并发现 "
            f"{len(union_buckets)} 个桶（{sample}）。请统一各子集分辨率或拆分训练。",
        ))
    elif union_buckets:
        width, height = next(iter(union_buckets))
        checks.append(_pp_check(
            "info",
            "dataset_bucket_uniform",
            f"所有子集桶一致（{width}x{height}），满足 PP 单桶 wire 约束。",
        ))
    if partial:
        checks.append(_pp_check(
            "warning",
            "dataset_census_partial",
            "桶统计不完整（存在不可读/未扫描图片），桶一致性结论可能有偏差。",
        ))

    # batch_size 一致：PP wire batch 是固定的，多子集不同 batch_size 必败。
    batch_sizes = {
        _row_batch_size(row, cfg)
        for row in rows
    }
    if len(batch_sizes) > 1:
        checks.append(_pp_check(
            "error",
            "dataset_batch_size_mismatch",
            "多子集 batch_size 不一致（必败）："
            + "、".join(str(value) for value in sorted(batch_sizes))
            + "。PP wire 按固定 batch 构建缓冲，请统一各子集 batch_size。",
        ))
    else:
        checks.append(_pp_check(
            "info",
            "dataset_batch_size_uniform",
            f"各子集 batch_size 一致（{sorted(batch_sizes)[0] if batch_sizes else 1}）。",
        ))

    # 缓存完整：latent 与 TE 缓存由现有 cache_audit 逐图核对。
    if not cfg.get("skip_cache_check"):
        missing_summary: list[str] = []
        for index, row in enumerate(rows, start=1):
            results = _cache_checks_for_row(row, cfg, _resolve_project_path)
            for result in results:
                missing = int(result.get("missing_count") or 0)
                if missing:
                    missing_summary.append(
                        f"子集 {index} {result.get('label') or result.get('kind')} 缺 {missing}"
                    )
        if missing_summary:
            checks.append(_pp_check(
                "error",
                "dataset_cache_incomplete",
                "缓存不完整（必败）：PP 只服务缓存好的固定形状数据。"
                + "；".join(missing_summary[:5])
                + "。请先补齐预处理缓存。",
            ))
        else:
            checks.append(_pp_check(
                "info",
                "dataset_cache_complete",
                "抽样图片的 latent / TE 缓存完整。",
            ))
    else:
        checks.append(_pp_check(
            "info",
            "dataset_cache_check_skipped",
            "skip_cache_check=true，本轮跳过缓存完整性核对。",
        ))
    return checks


_WIRED_FAMILY_HINT = (
    "当前模型族还没有接入流水线并行训练，请改用单卡或数据并行。"
    "已接线的族：Anima、MOSS-Image、Krea-2；Z-Image / Qwen-Image 2.1 暂不支持。"
)


def _parse_topology_pairs(text: str) -> list[tuple[int, int, str]]:
    """Parse ``nvidia-smi topo -m`` into ``(gpu_a, gpu_b, link)`` with ``a < b``."""

    pairs: list[tuple[int, int, str]] = []
    header: list[str] | None = None
    for raw in text.splitlines():
        line = raw.replace("\x1b[4m", "").replace("\x1b[0m", "").strip()
        if not line:
            continue
        cols = line.split()
        if header is None:
            if cols and cols[0] == "GPU0" and "CPU" in line:
                header = [c for c in cols if c.startswith("GPU")]
            continue
        if not cols[0].startswith("GPU"):
            continue
        try:
            left = int(cols[0][3:])
        except ValueError:
            continue
        for offset, token in enumerate(cols[1 : 1 + len(header)]):
            if token.upper() == "X" or offset >= len(header):
                continue
            try:
                right = int(header[offset][3:])
            except ValueError:
                continue
            if left < right:
                pairs.append((left, right, token.upper()))
    return pairs


def _pp_comm_path_check(
    device_indices: list[int] | None = None,
) -> dict[str, str] | None:
    """Report the GPU-to-GPU link NCCL will use for stage-to-stage traffic.

    NCCL already degrades on its own (NVLink -> PCIe P2P -> host bounce), so this
    row is informational: it tells the operator what to expect and never blocks a
    run.  Returns ``None`` when the topology cannot be read at all.

    ``device_indices`` are PCI-bus ordinals -- the same namespace the launch path
    uses (``CUDA_DEVICE_ORDER=PCI_BUS_ID`` + ``CUDA_VISIBLE_DEVICES``), so they can
    be matched against ``nvidia-smi topo -m`` rows directly.  With two or more
    indices only the links *between the selected GPUs* are reported: a box that
    has NVLink somewhere must not read as "your pipeline will use NVLink" when the
    chosen pair is a PCIe one.  ``None`` keeps the whole-machine summary.
    """

    try:
        proc = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    pairs = _parse_topology_pairs(proc.stdout)
    if not pairs:
        return None
    wanted = sorted({int(index) for index in (device_indices or [])})
    if wanted:
        if len(wanted) < 2:
            # A single card has no stage-to-stage link; saying anything about one
            # would be a machine summary dressed up as a PP finding.
            return None
        scope = set(wanted)
        pairs = [(a, b, link) for a, b, link in pairs if a in scope and b in scope]
        if not pairs:
            return None
    nvlink = [(a, b, link) for a, b, link in pairs if link.startswith("NV")]
    if nvlink:
        shown = "、".join(f"GPU{a}<->GPU{b} {link}" for a, b, link in nvlink[:4])
        return _pp_check(
            "info",
            "comm_path_nvlink",
            f"通信路径：检测到 NVLink（{shown}）。stage 间 p2p 会走 NVLink，"
            "实测约 87 GB/s；每步通信仅约 16 MB、占比 ~0.002%，不构成瓶颈。",
        )
    shown = "、".join(f"GPU{a}<->GPU{b} {link}" for a, b, link in pairs[:4])
    return _pp_check(
        "info",
        "comm_path_pcie",
        f"通信路径：未检测到 NVLink（{shown}）。NCCL 会自动回退到 PCIe P2P 或经 host 中转，"
        "不会阻断训练；实测带宽从 87 GB/s 压到 5.6 GB/s 时每步约慢 1%。",
    )


def _classify_admission_failure(exc: BaseException) -> tuple[str, str, str, str]:
    """``(level, key, human message, raw detail)`` for one admission refusal.

    A fault inside this diagnosis (bad import, missing attribute) is our bug,
    not the user's configuration: it must not render as a red "your PP config is
    wrong" row, so it degrades to a warning and says so plainly. The raw text is
    kept as ``detail`` for a folded disclosure either way.
    """

    raw = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, (ImportError, ModuleNotFoundError, ImportWarning)):
        return (
            "warning",
            "diagnosis_unavailable",
            "诊断组件自身异常（与你的配置无关），训练入口仍会独立校验；可先按当前配置提交。",
            raw,
        )
    if isinstance(exc, (AttributeError, NameError)):
        return (
            "warning",
            "diagnosis_unavailable",
            "诊断组件自身异常（与你的配置无关），训练入口仍会独立校验；可先按当前配置提交。",
            raw,
        )
    code = str(getattr(exc, "code", "") or "")
    if code == "pp_trainer_path_unavailable":
        return ("error", "trainer_path_unavailable", _WIRED_FAMILY_HINT, raw)
    if code == "pp_runtime_unavailable":
        return (
            "error",
            "runtime_unavailable",
            "该模型族没有可用的流水线并行运行时，PP 无法启动。",
            raw,
        )
    if code == "pp_config_invalid":
        return (
            "error",
            "config_invalid",
            "PP 参数超出当前支持范围，展开原始信息可看到具体键。",
            raw,
        )
    return ("error", "admission_refused", "流水线并行准入检查未通过。", raw)


def diagnose_pp_config(
    cfg: dict[str, Any],
    world_size: int | None,
    *,
    device_indices: list[int] | None = None,
) -> list[dict[str, str]]:
    """PP config/compat/admission diagnosis (复用现有实现，只读).

    ``device_indices`` (PCI-bus ordinals) narrows the communication-path row to the
    GPUs this card actually selected.
    """

    from library.models.pipeline_parallel import (
        PipelineParallelConfig,
        validate_pipeline_parallel_config,
    )
    from library.training.compat_matrix import check_training_compat
    from library.training.pp.gate import pipeline_parallel_enabled
    from library.training.pp.trainer_admission import require_family_trainer_admission

    checks: list[dict[str, str]] = []
    enabled = pipeline_parallel_enabled(cfg)
    if not enabled:
        return checks

    try:
        validate_pipeline_parallel_config(cfg, world_size=world_size)
    except ValueError as exc:
        checks.append(_pp_check("error", "config_invalid", str(exc)))
        return checks

    compatibility = check_training_compat(cfg, world_size=world_size)
    from web.services.config.preflight_pp import krea2_batch_checks

    batch_checks = krea2_batch_checks(cfg)
    reported_codes = {item["code"] for item in batch_checks if item.get("code")}
    checks.extend(batch_checks)
    for item in compatibility.errors:
        if item.key == "pipeline_parallel":
            checks.append(_pp_check("error", "compat_error", item.message))
    for item in compatibility.warnings:
        if item.key == "pipeline_parallel":
            checks.append(_pp_check("warning", "compat_warning", item.message))

    try:
        require_family_trainer_admission(
            cfg, world_size=world_size, where="distributed-diagnosis"
        )
    except Exception as exc:  # noqa: BLE001 - admission refusal becomes a finding
        if getattr(exc, "code", None) not in reported_codes:
            level, key, message, detail = _classify_admission_failure(exc)
            checks.append(_pp_check(level, key, message, detail=detail))
    else:
        pipeline = PipelineParallelConfig.from_config(cfg)
        checks.append(_pp_check(
            "info",
            "config_admitted",
            f"PP 配置通过现有 admission：stages={pipeline.stages}, "
            f"microbatches={pipeline.microbatches}, schedule={pipeline.schedule}, "
            f"pipeline_parallel_blocks_to_swap={pipeline.blocks_to_swap}。",
        ))
    comm = _pp_comm_path_check(device_indices)
    if comm is not None:
        checks.append(comm)
    return checks


def diagnose_distributed_training(
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    *,
    config_file: str | None = None,
    gpu_whitelist: Any = None,
    mode: Any = None,
    config_overrides: Any = None,
) -> dict[str, Any]:
    """Read-only diagnosis payload for the「分布式训练」card.

    按 ``variant/preset/config_file`` 装载与训练完全相同的配置（复用
    ``preflight`` 的装载链），再叠加模式归一、wired 族表、硬件风险（仅 PP）
    与 PP 数据集就绪检查。只读；任何检查失败都折叠成诊断 finding，不抛出。
    """

    from web.services.config_service import _load_training_config_for_web_run
    from web.services.training.gpu import normalize_gpu_whitelist
    from library.runtime.launch import resolve_training_world_size_for_gpu_selection
    from web.services.training.distributed_inputs import apply_diagnosis_overrides

    normalized_mode = normalize_distributed_mode(mode)
    gpu_selection = normalize_gpu_whitelist(gpu_whitelist)
    world_size = resolve_training_world_size_for_gpu_selection(gpu_selection)

    checks: list[dict[str, str]] = []
    try:
        cfg = _load_training_config_for_web_run(
            variant, preset, methods_subdir, config_file=config_file
        )
        cfg = apply_diagnosis_overrides(cfg, config_overrides)
    except Exception as exc:  # noqa: BLE001 - diagnosis never raises
        cfg = None
        checks.append({
            "level": "error",
            "key": "config_unreadable",
            "code": "pp_config_unreadable",
            "message": f"训练配置无法读取，PP 诊断降级：{exc}",
        })

    if cfg is not None and normalized_mode == "pp":
        try:
            checks.extend(
                diagnose_pp_config(cfg, world_size, device_indices=gpu_selection)
            )
        except Exception as exc:  # noqa: BLE001
            checks.append(_pp_check("error", "diagnosis_error", f"PP 配置诊断异常：{exc}"))
        try:
            checks.extend(diagnose_pp_dataset_readiness(cfg))
        except Exception as exc:  # noqa: BLE001
            checks.append(_pp_check("error", "diagnosis_error", f"PP 数据集诊断异常：{exc}"))

    hardware = hardware_diagnosis(normalized_mode, gpu_selection)
    return {
        "ok": True,
        "mode": normalized_mode,
        "modes": list(DISTRIBUTED_MODES),
        "world_size": world_size,
        "wired_families": wired_pp_families(),
        "checks": checks,
        "hardware": hardware,
    }


__all__ = [
    "DEFAULT_DISTRIBUTED_MODE",
    "DISTRIBUTED_MODES",
    "diagnose_distributed_training",
    "diagnose_pp_config",
    "diagnose_pp_dataset_readiness",
    "hardware_diagnosis",
    "item_distributed_mode",
    "normalize_distributed_mode",
    "wired_pp_families",
]
