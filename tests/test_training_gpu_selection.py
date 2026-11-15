from __future__ import annotations

import asyncio

import pytest

from web.services import training_service
from web.services.training import gpu as gpu_helpers
from web.services.training_service import _apply_gpu_whitelist, _normalize_gpu_whitelist


def test_normalize_gpu_whitelist_filters_invalid_and_duplicates():
    assert _normalize_gpu_whitelist(["1", 0, "bad", 1, "-2", 2]) == [1, 0, 2]
    assert gpu_helpers.normalize_gpu_whitelist(["1", 0, "bad", 1, "-2", 2]) == [1, 0, 2]


def test_apply_gpu_whitelist_sets_cuda_visible_devices():
    env = {}

    _apply_gpu_whitelist(env, [2, 0])

    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "2,0"


def test_apply_gpu_whitelist_keeps_default_when_empty():
    env = {"CUDA_VISIBLE_DEVICES": "3"}

    _apply_gpu_whitelist(env, [])

    assert env["CUDA_VISIBLE_DEVICES"] == "3"


def test_get_gpu_stats_uses_selected_physical_gpu(monkeypatch):
    class FakeProcess:
        async def communicate(self):
            return (
                b"0, 1024, 24576, 0, 44\n"
                b"1, 5120, 24576, 88, 71\n",
                b"",
            )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    monkeypatch.setattr(training_service.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    stats = asyncio.run(training_service._get_gpu_stats([1]))

    assert stats["gpu_index"] == 1
    assert stats["gpu_indices"] == [1]
    assert stats["vram_used_gb"] == 5
    assert stats["vram_total_gb"] == 24
    assert stats["gpu_util"] == 88
    assert stats["gpu_temp"] == 71


def test_get_gpu_stats_without_selection_keeps_first_gpu(monkeypatch):
    class FakeProcess:
        async def communicate(self):
            return (
                b"0, 1024, 24576, 11, 44\n"
                b"1, 5120, 24576, 88, 71\n",
                b"",
            )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    monkeypatch.setattr(training_service.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    stats = asyncio.run(training_service._get_gpu_stats([]))

    assert stats["gpu_index"] == 0
    assert stats["gpu_indices"] == [0]
    assert stats["gpu_util"] == 11


def test_training_gpu_helper_lists_available_gpus_with_injected_runner():
    class FakeProcess:
        async def communicate(self):
            return (
                b"0, NVIDIA RTX 5060 Ti, 16384\n"
                b"bad,row\n"
                b"1, NVIDIA RTX 4090, 24576\n",
                b"",
            )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    gpus = asyncio.run(
        gpu_helpers.list_available_gpus(
            create_subprocess_exec=fake_create_subprocess_exec,
            stdout_pipe=object(),
            stderr_devnull=object(),
        )
    )

    assert gpus == [
        {
            "index": 0,
            "name": "NVIDIA RTX 5060 Ti",
            "label": "GPU 0 · NVIDIA RTX 5060 Ti",
            "memory_total_mb": 16384,
            "memory_total_gb": 16.0,
        },
        {
            "index": 1,
            "name": "NVIDIA RTX 4090",
            "label": "GPU 1 · NVIDIA RTX 4090",
            "memory_total_mb": 24576,
            "memory_total_gb": 24.0,
        },
    ]


def test_system_monitor_samples_immediately_and_keeps_short_interval():
    """Dashboard resource cards should not wait a long initial sleep."""
    from pathlib import Path

    constants = Path("web/services/training/constants.py").read_text(encoding="utf-8")
    monitor = Path("web/services/training/live_monitor.py").read_text(encoding="utf-8")

    assert "SYSTEM_MONITOR_INTERVAL_SECONDS = 2.0" in constants
    assert "SYSTEM_MONITOR_INTERVAL_SECONDS" in monitor
    assert "await asyncio.sleep(SYSTEM_MONITOR_INTERVAL_SECONDS)" in monitor
    assert "await asyncio.sleep(5)" not in monitor
    # First sample happens at the top of the loop, before sleeping.
    sample_idx = monitor.index("snapshot = await self._gpu_snapshot_cache.read(ttl=2.0)")
    sleep_idx = monitor.index("await asyncio.sleep(SYSTEM_MONITOR_INTERVAL_SECONDS)")
    assert sample_idx < sleep_idx


# ---------------------------------------------------------------------------
# t9→放宽: gpu_whitelist 守卫（全卡开放，唯一 fail-closed = 索引解析不到物理卡）
# ---------------------------------------------------------------------------

from web.services.training.gpu import (
    GPU_WHITELIST_UNAUTHORIZED,
    ensure_gpu_whitelist_authorized,
    gpu_identity_authorized,
    parse_gpu_identity_rows,
    query_gpu_identities,
)


def _runner(rows: str):
    def run(_command):
        return rows

    return run


_HOST_ROWS = "\n".join(
    [
        "0, GPU-410a51d5-c1e2-6616-b56d-ba2e3ee2aa1b, NVIDIA GeForce GTX 1050",
        "1, GPU-b4b014b2-953b-28e8-b8c1-425c4a401165, Tesla V100-SXM2-16GB",
        "2, GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061, Tesla V100-SXM2-16GB",
    ]
)


def test_parse_gpu_identity_rows():
    rows = parse_gpu_identity_rows(_HOST_ROWS)
    assert rows[0]["name"] == "NVIDIA GeForce GTX 1050"
    assert rows[1]["uuid"].endswith("425c4a401165")


def test_query_gpu_identities_selects_physical_nvidia_smi_indices():
    rows = query_gpu_identities([0, 1], runner=_runner(_HOST_ROWS))
    assert set(rows) == {0, 1}
    assert query_gpu_identities(None, runner=_runner(_HOST_ROWS)).keys() == {0, 1, 2}


def test_gpu_identity_authorized():
    # 全卡开放后 identity 授权检查保留为恒真 compat shim（解析级守卫在别处）。
    assert gpu_identity_authorized({"uuid": "GPU-b4b014b2-953b-28e8-b8c1-425c4a401165", "name": "Tesla V100-SXM2-16GB"})
    assert gpu_identity_authorized({"uuid": "GPU-410a51d5-c1e2-6616-b56d-ba2e3ee2aa1b", "name": "NVIDIA GeForce GTX 1050"})
    assert gpu_identity_authorized(None)
    assert gpu_identity_authorized({"uuid": "GPU-x", "name": "GeForce RTX 3080"})


def test_guard_allows_the_gtx_1050_index_with_warning_semantics():
    # A/B: t9 时代 1050 会被 UUID 守卫拒绝；现在全卡开放，1050 可选，
    # 不再 raise；不适配风险改由分布式诊断的建议级警告承载。
    ensure_gpu_whitelist_authorized([0, 1], runner=_runner(_HOST_ROWS))
    ensure_gpu_whitelist_authorized([0], runner=_runner(_HOST_ROWS))


def test_guard_refuses_an_unresolvable_index():
    with pytest.raises(ValueError) as excinfo:
        ensure_gpu_whitelist_authorized([7], runner=_runner(_HOST_ROWS))
    assert GPU_WHITELIST_UNAUTHORIZED in str(excinfo.value)
    assert "7" in str(excinfo.value)
    # A probe that returns nothing at all is also a fail-closed refusal.
    with pytest.raises(ValueError) as excinfo:
        ensure_gpu_whitelist_authorized([1], runner=_runner(""))
    assert GPU_WHITELIST_UNAUTHORIZED in str(excinfo.value)


def test_guard_allows_authorized_v100_indices():
    ensure_gpu_whitelist_authorized([1], runner=_runner(_HOST_ROWS))
    ensure_gpu_whitelist_authorized([2, 1], runner=_runner(_HOST_ROWS))


def test_guard_keeps_the_empty_whitelist_default_path():
    # No explicit selection: the historical single-GPU default is untouched
    # (also on hosts without nvidia-smi).
    ensure_gpu_whitelist_authorized([])
    ensure_gpu_whitelist_authorized(None)


def test_launcher_start_and_resume_gate_import_the_guard():
    import inspect

    from web.services.training import launcher_start, resume_pipeline_gate

    assert "ensure_gpu_whitelist_authorized" in inspect.getsource(launcher_start)
    assert "ensure_gpu_whitelist_authorized" in inspect.getsource(resume_pipeline_gate)
    assert launcher_start.ensure_gpu_whitelist_authorized is ensure_gpu_whitelist_authorized
    assert resume_pipeline_gate.ensure_gpu_whitelist_authorized is ensure_gpu_whitelist_authorized


def test_queue_enqueue_calls_the_guard_before_enqueue():
    import inspect

    from web.services.training import queue_enqueue

    source = inspect.getsource(queue_enqueue)
    assert source.count("ensure_gpu_whitelist_authorized(") >= 2, (
        "training enqueue and batch enqueue must both guard before enqueueing"
    )
