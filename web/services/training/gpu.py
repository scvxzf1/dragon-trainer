"""GPU selection and nvidia-smi parsing helpers for WebUI training."""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any, Awaitable, Callable


CreateSubprocessExec = Callable[..., Awaitable[Any]]
NvidiaSmiRunner = Callable[[list[str]], str | None]

#: compute capability 的名称兜底表。只在 nvidia-smi 不支持 ``compute_cap`` 字段
#: （老驱动）时使用；未列出的型号返回 None，由调用方降级为 warning。
_CAPABILITY_BY_NAME: tuple[tuple[str, tuple[int, int]], ...] = (
    ("TITAN V", (7, 0)),
    ("V100", (7, 0)),
    ("T4", (7, 5)),
    ("RTX 2080", (7, 5)),
    ("RTX 2070", (7, 5)),
    ("QUADRO RTX", (7, 5)),
    ("A100", (8, 0)),
    ("A30", (8, 0)),
    ("A10", (8, 6)),
    ("A40", (8, 6)),
    ("RTX 30", (8, 6)),
    ("RTX 40", (8, 9)),
    ("L4", (8, 9)),
    ("L40", (8, 9)),
    ("H100", (9, 0)),
)


async def get_gpu_stats(
    gpu_whitelist: list[int] | None = None,
    *,
    create_subprocess_exec: CreateSubprocessExec | None = None,
    stdout_pipe: Any | None = None,
    stderr_devnull: Any | None = None,
) -> dict[str, Any]:
    try:
        runner = create_subprocess_exec or asyncio.create_subprocess_exec
        proc = await runner(
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
            stdout=stdout_pipe if stdout_pipe is not None else asyncio.subprocess.PIPE,
            stderr=stderr_devnull if stderr_devnull is not None else asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        rows = parse_gpu_stats_rows(stdout.decode(errors="replace"))
        selected = normalize_gpu_whitelist(gpu_whitelist)
        if selected:
            selected_set = set(selected)
            rows = [row for row in rows if row["index"] in selected_set]
        else:
            rows = rows[:1]
        return aggregate_gpu_stats_rows(rows)
    except Exception:
        pass
    return {}


def parse_gpu_stats_rows(text: str) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            rows.append({
                "index": int(parts[0]),
                "memory_used_mb": int(parts[1]),
                "memory_total_mb": int(parts[2]),
                "gpu_util": int(parts[3]),
                "gpu_temp": int(parts[4]),
            })
        except ValueError:
            continue
    return rows


def aggregate_gpu_stats_rows(rows: list[dict[str, int]]) -> dict[str, Any]:
    if not rows:
        return {}
    indices = [row["index"] for row in rows]
    used_mb = sum(row["memory_used_mb"] for row in rows)
    total_mb = sum(row["memory_total_mb"] for row in rows)
    return {
        "gpu_index": indices[0],
        "gpu_indices": indices,
        "vram_used_gb": round(used_mb / 1024, 2),
        "vram_total_gb": round(total_mb / 1024, 2),
        "gpu_util": max(row["gpu_util"] for row in rows),
        "gpu_temp": max(row["gpu_temp"] for row in rows),
    }


async def list_available_gpus(
    *,
    create_subprocess_exec: CreateSubprocessExec | None = None,
    stdout_pipe: Any | None = None,
    stderr_devnull: Any | None = None,
) -> list[dict[str, Any]]:
    try:
        runner = create_subprocess_exec or asyncio.create_subprocess_exec
        proc = await runner(
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
            stdout=stdout_pipe if stdout_pipe is not None else asyncio.subprocess.PIPE,
            stderr=stderr_devnull if stderr_devnull is not None else asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
    except Exception:
        return []

    gpus: list[dict[str, Any]] = []
    for line in stdout.decode(errors="replace").splitlines():
        item = parse_available_gpu_row(line)
        if item is not None:
            gpus.append(item)
    return gpus


def parse_available_gpu_row(line: str) -> dict[str, Any] | None:
    parts = [part.strip() for part in line.split(",", 2)]
    if len(parts) < 2:
        return None
    try:
        index = int(parts[0])
    except ValueError:
        return None
    memory_total_mb = int_or_none(parts[2]) if len(parts) >= 3 else None
    item: dict[str, Any] = {
        "index": index,
        "name": parts[1],
        "label": f"GPU {index} · {parts[1]}",
    }
    if memory_total_mb is not None:
        item["memory_total_mb"] = memory_total_mb
        item["memory_total_gb"] = round(memory_total_mb / 1024, 1)
    return item


def normalize_gpu_whitelist(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    raw_items = value if isinstance(value, list) else [value]
    out: list[int] = []
    for item in raw_items:
        try:
            index = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if index < 0 or index in out:
            continue
        out.append(index)
    return out


def apply_gpu_whitelist(env: dict[str, str], whitelist: list[int]) -> None:
    if whitelist:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in whitelist)


#: Stable named error for the whitelist resolution guard.
GPU_WHITELIST_UNAUTHORIZED = "gpu_whitelist_unauthorized_gpu"


def parse_gpu_identity_rows(text: str) -> dict[int, dict[str, str]]:
    """Parse ``index,uuid,name`` rows from ``nvidia-smi``."""

    rows: dict[int, dict[str, str]] = {}
    for line in str(text or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        index = int_or_none(parts[0])
        if index is None or not parts[1] or not parts[2]:
            continue
        rows[index] = {"uuid": parts[1], "name": parts[2]}
    return rows


def gpu_identity_authorized(identity: dict[str, str] | None) -> bool:
    """Legacy compat shim: every resolvable card is now authorized.

    GPU 选择已全卡开放（解析级 fail-closed 是唯一的硬守卫，见
    :func:`ensure_gpu_whitelist_authorized`）。该函数仅为旧调用面保留，
    恒返回 ``True``（unresolvable 情况由守卫自己在更早的位置拒绝）。
    """

    del identity
    return True


def query_gpu_identities(
    indices: Any = None,
    *,
    runner: NvidiaSmiRunner | None = None,
) -> dict[int, dict[str, str]]:
    """Resolve ``{index: {uuid, name}}`` for the physical nvidia-smi indices.

    The whitelist is index-based under ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` (see
    :func:`apply_gpu_whitelist`), so identity must be read from the same
    nvidia-smi index space.  Unknown indices are absent from the result.
    """

    rows = parse_gpu_identity_rows(
        _query_nvidia_smi("index,uuid,name", runner=runner) or ""
    )
    selected = normalize_gpu_whitelist(indices)
    if selected:
        return {index: rows[index] for index in selected if index in rows}
    return rows


def ensure_gpu_whitelist_authorized(
    whitelist: Any,
    *,
    runner: NvidiaSmiRunner | None = None,
) -> None:
    """Resolution-level fail-closed guard for the WebUI gpu_whitelist.

    GPU 选择全卡开放：不再按 UUID/型号拒绝任何物理卡（GTX 1050 等卡改为
    由分布式诊断给出建议级警告）。唯一 fail-closed 条件 = 白名单索引无法在
    nvidia-smi 中解析到物理卡，此时以命名错误
    :data:`GPU_WHITELIST_UNAUTHORIZED` 在入队/启动前拒绝。空白名单（默认
    单卡路径）不受影响，宿主机没有 nvidia-smi 时保持历史行为。
    """

    selected = normalize_gpu_whitelist(whitelist)
    if not selected:
        return
    resolved = query_gpu_identities(selected, runner=runner)
    missing = [index for index in selected if index not in resolved]
    if missing:
        raise ValueError(
            f"{GPU_WHITELIST_UNAUTHORIZED}: GPU 白名单索引 {missing} 无法在 "
            "nvidia-smi 中解析到物理卡（显存查询失败或索引越界）；"
            "为避免把未知设备拉进训练，拒绝启动。"
            "请改选在 nvidia-smi 中可见的 GPU 索引。"
        )


# ---------------------------------------------------------------------------
# Distributed hardware snapshot (one nvidia-smi query, TTL-cached)
# ---------------------------------------------------------------------------

#: Single nvidia-smi query powering the distributed hardware snapshot.
GPU_HARDWARE_QUERY_FIELDS = "index,uuid,name,memory.total,compute_cap"

#: Snapshot TTL: the WebUI polls often, and hardware topology changes rarely.
HARDWARE_SNAPSHOT_TTL_SECONDS = 30.0

_hardware_snapshot_cache: dict[str, Any] = {"monotonic": 0.0, "rows": []}


def parse_gpu_hardware_rows(text: str) -> list[dict[str, Any]]:
    """Parse ``index,uuid,name,memory.total,compute_cap`` rows.

    ``compute_cap`` is optional on old drivers; when absent the capability
    falls back to the GPU marketing-name table, else ``None`` (callers treat
    unknown capability as "no mixed-generation judgement", not an error).
    """

    rows: list[dict[str, Any]] = []
    for line in str(text or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        index = int_or_none(parts[0])
        if index is None or not parts[1]:
            continue
        name = parts[2]
        memory_total_mb = None
        capability = None
        if len(parts) >= 5:
            name = ",".join(parts[2:-2]).strip()
            memory_total_mb = int_or_none(parts[-2])
            capability = parse_compute_capability(parts[-1])
        elif len(parts) == 4:
            # 老驱动没有 compute_cap 列：index,uuid,name,memory.total
            memory_total_mb = int_or_none(parts[3])
        if capability is None:
            capability = capability_from_gpu_name(name)
        rows.append({
            "index": index,
            "uuid": parts[1],
            "name": name,
            "memory_total_mb": memory_total_mb,
            "capability": capability,
        })
    return rows


def query_gpu_hardware_snapshot(
    *,
    runner: NvidiaSmiRunner | None = None,
    ttl_seconds: float = HARDWARE_SNAPSHOT_TTL_SECONDS,
) -> list[dict[str, Any]]:
    """One nvidia-smi query for index/uuid/name/memory/capability, TTL-cached.

    Only the real nvidia-smi path is cached; injected ``runner`` call sites
    (tests, alternate probes) always run fresh so fixtures stay deterministic.
    """

    global _hardware_snapshot_cache
    import time as _time

    if runner is None:
        cache = _hardware_snapshot_cache
        now = _time.monotonic()
        if cache["rows"] and now - float(cache["monotonic"]) < ttl_seconds:
            return cache["rows"]
    rows = parse_gpu_hardware_rows(
        _query_nvidia_smi(GPU_HARDWARE_QUERY_FIELDS, runner=runner) or ""
    )
    if runner is None:
        _hardware_snapshot_cache = {"monotonic": _time.monotonic(), "rows": rows}
    return rows


def _format_gigabytes(memory_total_mb: int | None) -> str:
    if memory_total_mb is None:
        return "未知显存"
    return f"{round(memory_total_mb / 1024, 1)}GB"


def classify_pp_hardware_risks(snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suggestion-level PP hardware risks (仅 PP 模式判定).

    三层风险：显存异构 / 低显存（<8GB）/ 代际混搭。全部为建议级（warning /
    info），不阻挡启动；真正的拦截仍由训练入口的 admission 兜底。
    """

    risks: list[dict[str, Any]] = []
    if not snapshot:
        risks.append({
            "level": "info",
            "key": "pp_hardware_snapshot_unavailable",
            "code": "pp_hardware_snapshot_unavailable",
            "message": "无法通过 nvidia-smi 读取 GPU 硬件快照，PP 硬件风险检测降级为未知。",
        })
        return risks

    labels = {
        row["index"]: f"GPU {row['index']} · {row['name']}（{_format_gigabytes(row.get('memory_total_mb'))}）"
        for row in snapshot
    }

    low = [row for row in snapshot if (row.get("memory_total_mb") or 0) < 8192]
    if low:
        risks.append({
            "level": "warning",
            "key": "pp_low_vram",
            "code": "pp_low_vram",
            "message": "低显存风险（建议级）："
            + "、".join(labels[row["index"]] for row in low)
            + " 显存低于 8GB，PP 阶段放置大概率放不下底座分片。",
        })

    totals = {row["index"]: row.get("memory_total_mb") for row in snapshot}
    distinct = {value for value in totals.values() if value is not None}
    if len(distinct) > 1:
        risks.append({
            "level": "warning",
            "key": "pp_vram_heterogeneous",
            "code": "pp_vram_heterogeneous",
            "message": "显存异构（建议级）：各卡显存不一致（"
            + "、".join(
                f"GPU {index} {_format_gigabytes(value)}" for index, value in sorted(totals.items())
            )
            + "），建议按显存从小到大放置流水线阶段并降低 microbatches。",
        })

    capabilities = {row["index"]: row.get("capability") for row in snapshot}
    known = {value for value in capabilities.values() if value is not None}
    unknown_rows = [row for row in snapshot if row.get("capability") is None]
    if len(known) > 1:
        risks.append({
            "level": "warning",
            "key": "pp_mixed_generation",
            "code": "pp_mixed_generation",
            "message": "代际混搭（建议级）：参与 PP 的卡算力代际不一致（"
            + "、".join(
                f"GPU {index} sm_{cap[0]}{cap[1]}"
                for index, cap in sorted(capabilities.items()) if cap is not None
            )
            + "），跨代 NCCL/P2P 通常不可用，流水线吞吐会明显下降。",
        })
    elif unknown_rows:
        risks.append({
            "level": "info",
            "key": "pp_generation_unknown",
            "code": "pp_generation_unknown",
            "message": "部分 GPU 的算力代际无法解析（老驱动或未知型号），代际混搭检测未判定。",
        })
    return risks


def parse_compute_capability(value: Any) -> tuple[int, int] | None:
    """Parse ``"7.0"`` / ``"8.6"`` into ``(major, minor)``."""
    text = str(value or "").strip().replace("_", ".")
    if not text:
        return None
    parts = text.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        return None


def capability_from_gpu_name(name: Any) -> tuple[int, int] | None:
    """Best-effort capability from the GPU marketing name (fallback only)."""
    text = str(name or "").upper()
    for marker, capability in _CAPABILITY_BY_NAME:
        if marker in text:
            return capability
    return None


def parse_gpu_compute_capability_rows(text: str) -> dict[int, tuple[int, int]]:
    """Parse ``index,compute_cap`` rows (``nvidia-smi --format=csv,noheader``)."""
    rows: dict[int, tuple[int, int]] = {}
    for line in str(text or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        index = int_or_none(parts[0])
        capability = parse_compute_capability(parts[1])
        if index is None or capability is None:
            continue
        rows[index] = capability
    return rows


def parse_gpu_named_capability_rows(text: str) -> dict[int, tuple[int, int]]:
    """Parse ``index,name`` rows and map known names to a capability."""
    rows: dict[int, tuple[int, int]] = {}
    for line in str(text or "").splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) < 2:
            continue
        index = int_or_none(parts[0])
        capability = capability_from_gpu_name(parts[1])
        if index is None or capability is None:
            continue
        rows[index] = capability
    return rows


def _query_nvidia_smi(fields: str, *, runner: NvidiaSmiRunner | None = None) -> str | None:
    command = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    if runner is not None:
        try:
            output = runner(command)
        except Exception:  # noqa: BLE001 - probe failures degrade to "unknown"
            return None
        return None if output is None else str(output)
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def query_gpu_compute_capabilities(
    indices: Any = None,
    *,
    runner: NvidiaSmiRunner | None = None,
) -> dict[int, tuple[int, int]]:
    """Resolve ``compute capability`` per **physical** nvidia-smi index.

    The training subprocess is launched with ``CUDA_DEVICE_ORDER=PCI_BUS_ID``
    and ``CUDA_VISIBLE_DEVICES=<physical index>`` (see
    :func:`apply_gpu_whitelist`), so capability must be read from the same
    nvidia-smi index space instead of torch's re-numbered device order.
    Unknown indices are simply absent from the result.
    """
    found = parse_gpu_compute_capability_rows(
        _query_nvidia_smi("index,compute_cap", runner=runner) or ""
    )
    if not found:
        found = parse_gpu_named_capability_rows(
            _query_nvidia_smi("index,name", runner=runner) or ""
        )
    selected = normalize_gpu_whitelist(indices)
    if selected:
        return {index: found[index] for index in selected if index in found}
    return found


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
