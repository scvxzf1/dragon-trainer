"""R3 CPU contract: window verdict and optimizer apply/drop must agree across ranks.

Covers M23 (no per-micro-batch collective), M24 (single process is a byte-for-byte
no-op) and the two attribution rows M26①② (propagated drop never impersonates an
originating cause; a locally proven non-finite loss is not overwritten by the world
decision).

The multi-rank evidence here is a **distributed simulation**: ``torch.distributed`` is
replaced by a fake whose ``all_reduce(MAX)`` combines this rank's value with a scripted
peer value, so the reduction semantics can be asserted exactly and cheaply. Real NCCL /
native DDP are M15–M17 and live in ``tests/test_fp16_transaction_cuda.py``; they are
explicitly **not** verified by this file.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.training import loop, window_accounting, window_coordination

try:
    from window_replay_harness import Harness
except ImportError:  # pragma: no cover
    from tests.window_replay_harness import Harness


class _FakeDist:
    """Minimal ``torch.distributed`` stand-in for one rank of a two-rank world."""

    def __init__(self, peer_values=None, fixed=None):
        self.peer_values = list(peer_values or [])
        self.fixed = fixed
        self.calls: list[tuple[str, float]] = []
        self.ReduceOp = SimpleNamespace(MAX="MAX", SUM="SUM")
        self.Backend = SimpleNamespace(NCCL="nccl", GLOO="gloo")

    # -- torch.distributed surface used by window_coordination ---------------
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_initialized() -> bool:
        return True

    @staticmethod
    def get_backend() -> str:
        return "gloo"

    def all_reduce(self, tensor, op=None, async_op=False):
        if self.fixed is not None:
            peer = self.fixed
        else:
            peer = self.peer_values.pop(0) if self.peer_values else float("-inf")
        local = float(tensor.reshape(-1)[0].item())
        self.calls.append((op, local))
        tensor.reshape(-1)[0] = (
            local if op == "SUM" else torch.tensor(max(local, float(peer)))
        )
        return None


def _install(monkeypatch, fake: _FakeDist) -> None:
    import torch.distributed as dist

    for name in ("is_available", "is_initialized", "get_backend", "all_reduce"):
        monkeypatch.setattr(dist, name, getattr(fake, name), raising=False)
    monkeypatch.setattr(dist, "ReduceOp", fake.ReduceOp, raising=False)


def _multi_rank_state(monkeypatch, fake: _FakeDist | None = None):
    """Harness state that *believes* it is rank 0 of a two-rank world."""
    harness = Harness(monkeypatch, num_steps=1, batches=1, loss_plan=[1.0])
    harness.state.accelerator.num_processes = 2
    if fake is not None:
        _install(monkeypatch, fake)
    return harness


# ---------------------------------------------------------------------------
# M24 — single process must stay a no-op
# ---------------------------------------------------------------------------
def test_single_process_makes_no_collective_at_all(monkeypatch):
    harness = Harness(monkeypatch, num_steps=2, batches=2, loss_plan=[1.0, 2.0])
    poisoned: list[float] = []

    def boom(*_a, **_k):  # pragma: no cover - must never run
        poisoned.append(1.0)
        raise AssertionError("single-process run must not touch torch.distributed")

    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "all_reduce", boom, raising=False)

    harness.run()

    assert poisoned == [], "M24：单进程零回归，不做任何规约"
    assert harness.state.window_commits == 1
    assert harness.state.step_attempts == 1


# ---------------------------------------------------------------------------
# M23 — one reduction per boundary, never per micro-batch
# ---------------------------------------------------------------------------
def test_one_collective_per_boundary_not_per_microbatch(monkeypatch):
    fake = _FakeDist(peer_values=[0.0, 0.0])  # peer is clean: verdict stays clean
    harness = _multi_rank_state(monkeypatch, fake)
    harness.state.window_plan = None
    harness.state.accelerator.gradient_state._set_sync_gradients(True)
    buffer = window_accounting.WindowBuffer()
    for loss in (1.0, 2.0, 3.0, 4.0):
        buffer.add(object(), loss, epoch=0, step=0)
        # The per-boundary agreement runs once when the accumulate-scope boundary
        # resolves; micro-batches themselves never reduce.
        window_coordination.agree_drop_update(harness.state, False)
    window_coordination.window_reason(harness.state, buffer)

    assert len(fake.calls) == 5, "4 个边界各 1 次 + 1 次窗口判定，且都只有 1 元素"
    assert all(op == "MAX" for op, _ in fake.calls)


def test_microbatch_body_reduces_only_at_the_boundary(monkeypatch):
    """同一窗口内 4 个微批只产生 1 次应用判定（窗口判定另计）。"""
    fake = _FakeDist(peer_values=[0.0, 0.0])
    harness = Harness(monkeypatch, num_steps=4, batches=4, loss_plan=[1.0, 2.0, 3.0, 4.0])
    harness.state.accelerator.num_processes = 2
    _install(monkeypatch, fake)

    harness.run()

    assert harness.state.window_commits == 1
    assert len(fake.calls) == 2, f"期望 1 次边界判定 + 1 次窗口判定，实际 {fake.calls}"


# ---------------------------------------------------------------------------
# verdict / apply-drop agreement semantics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("local", "peer", "expected"),
    [
        (None, 0.0, None),
        (None, window_coordination._CODE_NONFINITE, window_accounting.REASON_NONFINITE_LOSS),
        (None, window_coordination._CODE_UPDATE_SKIPPED, window_accounting.REASON_UPDATE_SKIPPED),
        (
            window_accounting.REASON_NONFINITE_LOSS,
            0.0,
            window_accounting.REASON_NONFINITE_LOSS,
        ),
    ],
)
def test_window_verdict_is_the_world_worst(monkeypatch, local, peer, expected):
    fake = _FakeDist(peer_values=[peer])
    _install(monkeypatch, fake)
    state = _multi_rank_state(monkeypatch).state

    assert window_coordination.agree_failure_reason(state, local) == expected
    assert len(fake.calls) == 1


def test_clean_rank_follows_a_world_drop_decision(monkeypatch):
    fake = _FakeDist(peer_values=[1.0])  # peer drops, this rank is clean
    _install(monkeypatch, fake)
    state = _multi_rank_state(monkeypatch).state

    assert window_coordination.agree_drop_update(state, False) is True
    assert window_coordination.agree_drop_update(state, True) is True
    fake.peer_values = [0.0]
    assert window_coordination.agree_drop_update(state, False) is False


# ---------------------------------------------------------------------------
# M26①② — attribution layers stay apart
# ---------------------------------------------------------------------------
def test_propagated_drop_does_not_impersonate_an_originating_cause(monkeypatch):
    state = _multi_rank_state(monkeypatch).state
    state.update_skipped = True  # dropped, but this rank proved nothing locally

    cause = window_coordination.note_attempt(
        state,
        skip_update=False,
        force_skip=False,
        loss_is_finite=True,
        step_attempted=True,
        scaler_skipped=False,
    )
    snap = window_coordination.snapshot(state)

    assert cause is None
    assert snap["propagated_drops"] == 1, "L2 单独计数"
    assert "nonfinite_loss" not in snap["originating_causes"]
    assert "scaler_overflow" not in snap["originating_causes"]
    assert snap["originating_causes"][window_coordination.CAUSE_NO_LOCAL_REASON] == 1


def test_local_nonfinite_loss_is_not_overwritten_by_the_world_decision(monkeypatch):
    state = _multi_rank_state(monkeypatch).state
    state.update_skipped = True

    window_coordination.note_attempt(
        state,
        skip_update=False,
        force_skip=False,
        loss_is_finite=False,
        step_attempted=True,
    )
    snap = window_coordination.snapshot(state)

    assert snap["originating_causes"][window_coordination.CAUSE_NONFINITE_LOSS] == 1
    assert snap["propagated_drops"] == 0, "本地原发不得同时计成传播 drop"


def test_window_discard_causes_stay_local_when_the_world_discards(monkeypatch):
    state = _multi_rank_state(monkeypatch).state
    window_accounting.ensure_counters(state)
    buffer = window_accounting.WindowBuffer()
    buffer.add(object(), 1.0, epoch=0, step=0)
    buffer.record_attempt()

    # 本 rank 干净、由世界决定丢弃：不得进入成因直方图
    window_accounting.discard_window(state, buffer, window_accounting.REASON_NONFINITE_LOSS,
                                     local_reason=None)
    window_coordination.note_window_decision(state, "discard", propagated=True)
    snap = window_coordination.snapshot(state)

    assert snap["window_discard_causes"] == {}
    assert snap["window_discard_propagated"] == 1
    assert snap["window_world_decisions"] == {"discard": 1}

    # 本 rank 自己观测到非有限：进 L1 直方图
    buffer2 = window_accounting.WindowBuffer()
    buffer2.add(object(), float("nan"), epoch=0, step=0)
    buffer2.record_attempt()
    window_accounting.discard_window(
        state,
        buffer2,
        window_accounting.REASON_NONFINITE_LOSS,
        local_reason=window_accounting.REASON_NONFINITE_LOSS,
    )
    snap = window_coordination.snapshot(state)
    assert snap["window_discard_causes"] == {window_accounting.REASON_NONFINITE_LOSS: 1}
    assert snap["window_discard_propagated"] == 1


def test_discarded_window_still_advances_consecutive_discards_once(monkeypatch):
    """世界一致不得把一个窗口的丢弃重复计数（否则预算会被提前触发）。"""
    # 对端恒定报告 nonfinite：本 rank 干净也必须跟随世界丢弃
    fake = _FakeDist(fixed=window_coordination._CODE_NONFINITE)
    harness = Harness(
        monkeypatch,
        num_steps=1,
        batches=1,
        loss_plan=[1.0],
        max_train_steps=1,
    )
    harness.state.accelerator.num_processes = 2
    _install(monkeypatch, fake)

    harness.run()

    state = harness.state
    assert state.window_discards == 1, "一个有终态的窗口只算一次丢弃"
    assert state.window_consecutive_discards == 1, "世界一致不得重复计数（否则预算提前触发）"
    assert state.window_replays == 1, "上限仍是 1 次重放"
    assert state.global_step == 0, "干净 rank 跟随丢弃时也不得推进 global_step"
    assert state.window_commits == 0
    assert state.step_attempts == state.window_attempts == 2, "原尝试 + 一次重放；I2 仍成立"
