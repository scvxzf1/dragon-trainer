"""nvidia-smi telemetry sampler for bench runs.

A benchmark that only records step time cannot tell you *why* the number moved:
whether the SM clock was pinned or throttling, whether VRAM was at the edge, or
whether block swap ran on a x8 gen1 link instead of a x16 gen3 one. This module
samples the physical GPU in a background thread for the duration of a run and
summarizes the interesting fields.

Usage::

    from bench.gpu_telemetry import GpuTelemetry

    tel = GpuTelemetry(1, interval_ms=500, csv_path=run_dir / "telemetry.csv")
    tel.start()
    ...run the training job...
    tel.stop()
    metrics["telemetry"] = tel.summary()

``gpu_index`` is the **nvidia-smi** index (PCI bus order). Only use it together
with ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` in the child process, otherwise the CUDA
index and the nvidia-smi index disagree (see bench notes in the README).
"""

from __future__ import annotations

import csv
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

#: nvidia-smi query fields, in output order. ``name`` comes first so the summary
#: can prove *which* physical card was sampled.
FIELDS: tuple[str, ...] = (
    "name",
    "clocks.sm",
    "clocks.mem",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "temperature.gpu",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
)

#: Fields summarized as distributions; the rest are reported as observed sets.
_DISTRIBUTION_FIELDS = (
    "clocks.sm",
    "clocks.mem",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "temperature.gpu",
)

_LINK_FIELDS = (
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
)


def _parse_number(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3),
        "min": round(min(values), 3),
        "p50": round(_quantile(values, 0.50), 3),
        "p95": round(_quantile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


class GpuTelemetry:
    """Background nvidia-smi sampler; safe to call when nvidia-smi is absent."""

    def __init__(
        self,
        gpu_index: int,
        *,
        interval_ms: int = 500,
        csv_path: Path | str | None = None,
    ) -> None:
        self.gpu_index = int(gpu_index)
        self.interval_ms = max(200, int(interval_ms))
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.rows: list[dict[str, Any]] = []
        self.started_at: float | None = None
        self.duration_sec: float | None = None
        self.available = shutil.which("nvidia-smi") is not None
        self.error: str | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def _command(self) -> list[str]:
        fields = ",".join(FIELDS)
        return [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]

    def start(self) -> "GpuTelemetry":
        if not self.available:
            self.error = "nvidia-smi not found"
            return self
        try:
            self._proc = subprocess.Popen(
                self._command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as exc:  # pragma: no cover - environment dependent
            self.available = False
            self.error = str(exc)
            return self
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()
        return self

    def _consume(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != len(FIELDS):
                continue
            row: dict[str, Any] = {"ts": round(time.time(), 3)}
            for name, value in zip(FIELDS, parts):
                number = _parse_number(value)
                row[name] = number if number is not None else value
            self.rows.append(row)

    def stop(self) -> "GpuTelemetry":
        if self.started_at is not None:
            self.duration_sec = round(time.time() - self.started_at, 3)
        proc = self._proc
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._proc = None
        self._write_csv()
        return self

    def _write_csv(self) -> None:
        if self.csv_path is None or not self.rows:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["ts", *FIELDS])
            writer.writeheader()
            writer.writerows(self.rows)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "gpu_index_nvidia_smi": self.gpu_index,
            "interval_ms": self.interval_ms,
            "sample_count": len(self.rows),
            "duration_sec": self.duration_sec,
            "available": self.available,
        }
        if self.error:
            out["error"] = self.error
        if self.csv_path is not None:
            out["csv"] = str(self.csv_path)
        if not self.rows:
            return out

        names = sorted({str(r["name"]) for r in self.rows if "name" in r})
        if names:
            out["gpu_name"] = names

        for field in _DISTRIBUTION_FIELDS:
            values = [r[field] for r in self.rows if isinstance(r.get(field), float)]
            stats = _distribution(values)
            if stats:
                out[field] = stats

        for field in _LINK_FIELDS:
            values = sorted({r[field] for r in self.rows if field in r})
            if values:
                out[field] = values

        totals = [r["memory.total"] for r in self.rows if isinstance(r.get("memory.total"), float)]
        used = [r["memory.used"] for r in self.rows if isinstance(r.get("memory.used"), float)]
        if totals and used:
            total = statistics.fmean(totals)
            out["memory_total_mib"] = round(total, 1)
            out["memory_used_peak_pct"] = round(100.0 * max(used) / total, 2)
        busy = [r["utilization.gpu"] for r in self.rows if isinstance(r.get("utilization.gpu"), float)]
        if busy:
            out["utilization_gpu_mean_pct"] = round(statistics.fmean(busy), 2)
            out["utilization_gpu_busy_sample_pct"] = round(
                100.0 * sum(1 for v in busy if v >= 50.0) / len(busy), 2
            )

        # Model load / preprocessing samples are CPU-bound with a cold GPU and
        # would drag every mean down. Keep a separate "active" window: samples
        # where the GPU actually reported work.
        active = [
            r for r in self.rows
            if isinstance(r.get("utilization.gpu"), float) and r["utilization.gpu"] > 0.0
        ]
        if active:
            out["active_sample_count"] = len(active)
            out["active_sample_pct"] = round(100.0 * len(active) / len(self.rows), 2)
            active_stats: dict[str, Any] = {}
            for field in ("clocks.sm", "clocks.mem", "memory.used", "utilization.gpu", "power.draw"):
                values = [r[field] for r in active if isinstance(r.get(field), float)]
                stats = _distribution(values)
                if stats:
                    active_stats[field] = stats
            out["active"] = active_stats
        return out
