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
