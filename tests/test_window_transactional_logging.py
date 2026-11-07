"""R1 / R2 / R4 / R5 CPU contract: window-transactional loss publication and step accounting.

Covers the frozen acceptance rows that must be asserted on the **real** main path:
M1–M11, M19-R2 (①②③④⑤), M25 (①②③), M26⑥ and M8-R2 (①②③④).

Why the fixtures look the way they do (contract §2 hard constraint): the shipped harness
stubs ``_log_step`` / ``_maybe_run_step_validation`` by default, so nothing here may claim
R1/R2 by reading a stubbed object. Cases that assert "the step-level event fires / does not
fire" therefore run the harness in its additive ``stub_step_ticks=False`` mode, where the
production ``_log_step``, the progress bar, the progress sink and the step-validation call
are the real functions.

Evidence class: fake accelerator (scheduling / ledger) plus real-accelerator (CPU) where the
row asks for it. No CUDA, no models, no real data.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from library.training import loop, window_accounting, window_coordination
from library.training.loss_recorder import LossRecorder

try:  # pytest puts tests/ on sys.path; explicit fallback keeps direct imports working
    from window_replay_harness import Harness, RealAcceleratorHarness
except ImportError:  # pragma: no cover
    from tests.window_replay_harness import Harness, RealAcceleratorHarness


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _run(monkeypatch, *, ticks: bool = False, **kwargs) -> Harness:
    """Harness with a real ``LossRecorder``; ``ticks`` keeps _log_step/validation real."""
    harness = Harness(monkeypatch, stub_step_ticks=not ticks, **kwargs)
    if not ticks:
        harness.state.loss_recorder = LossRecorder()
    return harness


def _count_validations(monkeypatch) -> list[int]:
    seen: list[int] = []
    monkeypatch.setattr(loop, "run_validation", lambda *a, **k: seen.append(k.get("global_step")))
    return seen


def _enable_step_validation(state, every: int = 1) -> None:
    state.args.validate_every_n_steps = every
    state.validation_steps = 1
    state.val_ctx = SimpleNamespace(dataloader=[])


# ---------------------------------------------------------------------------
# M1 / M2 — a committed window publishes exactly its own micro-batches
# ---------------------------------------------------------------------------
def test_single_window_publishes_one_sample_and_ticks_once(monkeypatch):
    validations = None
    harness = _run(monkeypatch, ticks=True, num_steps=1, loss_plan=[2.0], batches=1)
    validations = _count_validations(monkeypatch)
    _enable_step_validation(harness.state)
    harness.run()

    state = harness.state
    assert state.loss_recorder.loss_list == [2.0]
    assert state.global_step == 1
    assert harness.progress_updates == 1
    assert harness.sink.events == [1]
    assert validations == [1]
    assert state.window_boundary_committed is True


@pytest.mark.parametrize("num_steps", [2, 4])
def test_multi_microbatch_window_publishes_every_microbatch_in_order(monkeypatch, num_steps):
    losses = [float(index + 1) for index in range(num_steps)]
    harness = _run(
        monkeypatch, ticks=True, num_steps=num_steps, loss_plan=list(losses), batches=num_steps
    )
    harness.run()

    state = harness.state
    assert state.loss_recorder.loss_list == losses, "整窗按微批顺序发布"
    assert state.loss_recorder.moving_average == pytest.approx(sum(losses) / len(losses))
    assert harness.progress_updates == 1, "一个窗口只出一个 step tick"
    assert state.window_commits == 1
    assert state.window_num_steps == num_steps


# ---------------------------------------------------------------------------
# M3 — a successful replay publishes the replay values, not the failed prefix
# ---------------------------------------------------------------------------
def test_replay_success_publishes_replay_values_only(monkeypatch):
    harness = _run(
        monkeypatch,
        num_steps=2,
        loss_plan=[float("nan"), 20.0, 30.0, 40.0],
        batches=2,
    )
    harness.run()
    state = harness.state

    assert state.window_replays == 1
    assert state.loss_recorder.loss_list == [30.0, 40.0], "只发布重放值，不含失败尝试的 20"
    assert state.window_total_loss == pytest.approx(70.0), "账本口径不因发布而变"
    assert state.window_commits == 1


# ---------------------------------------------------------------------------
# M4 — a discarded window publishes nothing and re-fires no step-level event
# ---------------------------------------------------------------------------
def test_discard_publishes_nothing_and_does_not_repeat_step_events(monkeypatch):
    harness = _run(
        monkeypatch,
        ticks=True,
        num_steps=2,
        batches=6,
        loss_plan=[1.0, 2.0, 10.0, float("nan"), float("nan"), float("nan"), 3.0, 4.0],
    )
    validations = _count_validations(monkeypatch)
    _enable_step_validation(harness.state)
    harness.run()
    state = harness.state

    assert state.window_discards == 1
    assert state.window_commits == 2
    assert state.global_step == 2, "丢窗不推进 global_step"
    assert state.loss_recorder.loss_list == [1.0, 2.0, 3.0, 4.0], (
        "失败尝试的有限前缀 10 与 NaN 都不得入账"
    )
    assert harness.progress_updates == 2, "丢窗不产生 step tick"
    assert harness.sink.events == [1, 2], "同一 global_step 不得重复发进度事件"
    assert validations == [1, 2], "同一 global_step 不得重复跑 step 级验证"
    assert state.window_discard_causes == {window_accounting.REASON_NONFINITE_LOSS: 1}


def test_placeholder_window_ledger_is_not_reported_as_training_statistics(monkeypatch):
    """丢窗不得把占位窗口账本（sum/steps/last）当成实际训练统计写出去。"""
    from library.training.adaptive_runtime.training_runtime import NonfiniteSkip

    harness = _run(
        monkeypatch, num_steps=1, batches=4, loss_plan=[float("nan")] * 8
    )
    with pytest.raises(NonfiniteSkip):
        harness.run()
    state = harness.state
    assert state.window_commits == 0, "全程 0 个提交窗口"
    assert state.window_total_loss == 0.0 and state.window_num_steps == 0
    assert state.window_last_loss is None
    assert state.loss_recorder.loss_list == [], "没有任何样本进入 LossRecorder"


# ---------------------------------------------------------------------------
# M5 — epoch-tail window (end_of_dataloader forces a short boundary batch)
# ---------------------------------------------------------------------------
def test_epoch_tail_window_commits_and_ticks_once(monkeypatch):
    harness = RealAcceleratorHarness(
        monkeypatch, num_steps=2, total_batches=3, stub_step_ticks=False
    )
    try:
        harness.state.loss_recorder = LossRecorder()
        validations = _count_validations(monkeypatch)
        _enable_step_validation(harness.state)
        loop._run_epoch_steps(harness.trainer, harness.state, 0)
        state = harness.state

        # 3 个 batch、num_steps=2：首个满窗 + 数据流末尾的短窗（1 个微批）。
        assert state.window_commits == 2, "尾窗仍然提交（不得因门控漏 tick）"
        assert len(state.loss_recorder.loss_list) == 3, "尾窗发布样本数 = 实际微批数"
        assert harness.progress_updates == 2, "每个提交窗口各有一次 tick"
        assert validations == [1, 2]
    finally:
        harness.close()


# ---------------------------------------------------------------------------
# M6 / M7 — cross-epoch slots and non-zero resume
# ---------------------------------------------------------------------------
def test_cross_epoch_slots_restart_without_discard_holes(monkeypatch):
    harness = _run(
        monkeypatch,
        num_steps=1,
        batches=2,
        loss_plan=[
            1.0, 2.0,                                  # epoch 0：两个提交窗口
            float("nan"), float("nan"),                 # epoch 1：两个窗口都丢（重放也失败）
            float("nan"), float("nan"),
            7.0, 8.0,                                  # epoch 2：恢复并提交
        ],
        max_train_steps=4,
    )
    state = harness.state
    for epoch in (0, 1, 2):
        loop._run_epoch_steps(harness.trainer, state, epoch)

    assert state.window_commits == 4 and state.window_discards == 2
    assert state.global_step == 4, "跨 epoch 的丢窗不推进 global_step"
    # LossRecorder 在 epoch > 0 时按 step 索引覆盖；丢窗既不留样本也不留 0 占位洞。
    assert state.loss_recorder.loss_list == [7.0, 8.0]
    assert all(value != 0.0 for value in state.loss_recorder.loss_list)
    assert math.isfinite(state.loss_recorder.moving_average)


def test_resume_initial_step_keeps_numbering_and_first_commit_ticks(monkeypatch):
    """M7：非零恢复时被跳过的 batch 仍占号，恢复后首个提交照常发布并 tick。"""
    harness = _run(
        monkeypatch, ticks=True, num_steps=1, batches=5, loss_plan=[1.0, 2.0, 3.0, 4.0, 5.0]
    )
    state = harness.state
    state.initial_step = 3  # 由 accelerator 侧回退 2 个 batch，再吞掉 1 个（仍占号）
    harness.accelerator.skip_first_batches = lambda loader, count: loader[count:]
    harness.run()

    assert state.global_step == 2, "剩余 2 个微批各提交一次"
    # 夹具的 loss 计划按 forward 次数推进（被跳过的 batch 不占 loss 序号），
    # 因此恢复后两个真实微批拿到 1.0 / 2.0；关键契约是它们照常发布并 tick。
    assert state.loss_recorder.loss_list == [1.0, 2.0], "恢复后的窗口照常发布"
    assert harness.progress_updates == 2
    assert state.window_boundary_committed is True


# ---------------------------------------------------------------------------
# M8-R2 — _run_step is a supported entry (AUTO probe) with its own accounting scope
# ---------------------------------------------------------------------------
def test_run_step_keeps_direct_route_semantics_without_window_publication(monkeypatch):
    harness = _run(monkeypatch, num_steps=1, batches=1, loss_plan=[1.0])
    state = harness.state
    state.loss_recorder = LossRecorder()
    boundary_calls: list[int] = []
    monkeypatch.setattr(loop, "_commit_step_boundary", lambda *a, **k: boundary_calls.append(1))
    harness.accelerator.gradient_state._set_sync_gradients(True)

    loop._run_step(harness.trainer, state, batch=object())

    assert state.loss_recorder.loss_list == [], "① 直达路由无整窗发布"
    assert boundary_calls == [1], "② 边界且未被丢弃 → 调用提交副作用"
    assert state.step_attempts == 0, "③ 直达路由不得进入窗口尝试计数"
    assert state.direct_route_boundaries == 1
    assert state.window_commits == 0 and state.window_discards == 0

    # ④ 主训练窗口计数在探针前后不被污染：窗口路由继续满足 I1
    harness2 = _run(monkeypatch, num_steps=1, batches=2, loss_plan=[1.0, 2.0])
    harness2.run()
    st2 = harness2.state
    assert st2.direct_route_boundaries == 0
    assert st2.step_attempts == st2.window_attempts


def test_run_step_skips_commit_side_effects_when_update_dropped(monkeypatch):
    harness = _run(monkeypatch, num_steps=1, batches=1, loss_plan=[float("nan")])
    state = harness.state
    boundary_calls: list[int] = []
    monkeypatch.setattr(loop, "_commit_step_boundary", lambda *a, **k: boundary_calls.append(1))
    harness.accelerator.gradient_state._set_sync_gradients(True)

    loop._run_step(harness.trainer, state, batch=object())

    assert state.update_skipped is True
    assert boundary_calls == [], "② 非有限 loss 丢弃更新时不得跑提交副作用"


# ---------------------------------------------------------------------------
# M9 — tracker / progress sink follow the commit, not the micro-batch
# ---------------------------------------------------------------------------
def test_tracker_and_progress_sink_events_follow_commits_only(monkeypatch):
    harness = _run(
        monkeypatch,
        ticks=True,
        num_steps=2,
        batches=6,
        loss_plan=[1.0, 2.0, 10.0, float("nan"), float("nan"), float("nan"), 3.0, 4.0],
    )
    state = harness.state
    state.is_tracking = True
    tracked: list[int] = []
    harness.trainer.step_logging = lambda _acc, _logs, step, _epoch: tracked.append(int(step))
    harness.trainer.generate_step_logs = lambda *a, **k: {}
    monkeypatch.setattr(loop, "collect_metrics", lambda *a, **k: {})
    monkeypatch.setattr(loop, "MetricContext", lambda **k: SimpleNamespace(**k))

    harness.run()

    assert tracked == [1, 2], "tracker 的 step 级写入只随新提交"
    assert harness.sink.events == [], "tracking 时走 tracker，不再额外发 sink 事件"


# ---------------------------------------------------------------------------
# M10 / M11 — budgets unchanged (adaptive 0 fails on the first discard, plain 3 on the 4th)
# ---------------------------------------------------------------------------
def test_adaptive_budget_zero_escalates_on_first_discard(monkeypatch):
    from library.training.adaptive_runtime.training_runtime import NonfiniteSkip

    harness = _run(
        monkeypatch,
        num_steps=1,
        batches=1,
        loss_plan=[float("nan"), float("nan")],
        adaptive_trial=True,
        adaptive_skip_budget=None,
    )
    with pytest.raises(NonfiniteSkip):
        harness.run()
    assert harness.state.window_discards == 1
    assert harness.state.window_commits == 0


def test_plain_budget_three_tolerates_then_escalates_on_fourth(monkeypatch):
    from library.training.adaptive_runtime.training_runtime import NonfiniteSkip

    harness = _run(
        monkeypatch,
        num_steps=1,
        batches=4,
        loss_plan=[float("nan")] * 8,  # 每窗 2 次尝试，全部失败
        window_discard_budget=None,
    )
    with pytest.raises(NonfiniteSkip):
        harness.run()
    state = harness.state
    assert state.window_discards == 4, "普通预算 3：第 4 次连续丢窗熔断"
    assert state.global_step == 0


# ---------------------------------------------------------------------------
# M19-R2 / M25 / M26⑥ — counting identities, their boundaries, budget neutrality
# ---------------------------------------------------------------------------
def _i1_holds(state) -> bool:
    return state.step_attempts == (
        state.window_commits + state.window_discards + state.window_replays
    )


def test_counting_identities_hold_on_the_window_route(monkeypatch):
    harness = _run(
        monkeypatch,
        num_steps=2,
        batches=6,
        loss_plan=[1.0, 2.0, 10.0, float("nan"), float("nan"), float("nan"), 3.0, 4.0],
    )
    harness.run()
    state = harness.state

    assert state.direct_route_boundaries == 0
    assert _i1_holds(state), "I1: step_attempts == committed + discarded + replayed"
    assert state.step_attempts == state.window_attempts, "I2 交叉校验"
    assert state.step_updates_applied <= state.step_attempts
    assert state.step_updates_applied == state.window_commits, "E3 路由下 effective == committed"
    published = len(state.loss_recorder.loss_list)
    assert published == 2 + 2, "提交窗口各 2 个微批，丢窗 0 样本"


def test_i1_must_not_be_derived_when_a_window_has_no_terminal_state(monkeypatch):
    """M25①：提交路径抛异常时尝试已记但窗口无终态 ⇒ 不得用 I1 反推窗口数。"""
    harness = _run(monkeypatch, num_steps=1, batches=2, loss_plan=[1.0, 2.0])
    state = harness.state

    def boom(*_args, **_kwargs):
        raise RuntimeError("interrupted between attempt and window terminal state")

    # 注入点在「边界尝试已记录、窗口尚无终态」之间
    monkeypatch.setattr(window_coordination, "agree_failure_reason", boom)
    with pytest.raises(RuntimeError):
        harness.run()

    assert state.step_attempts == 1
    assert state.window_commits == 0 and state.window_discards == 0
    assert not _i1_holds(state), "无终态时 I1 不成立：禁止用它推导窗口数"


def test_record_attempt_runs_once_per_try_plus_each_replay(monkeypatch):
    """M25③：num_steps=4 的单窗只 record_attempt 一次；每次重放各加一次。"""
    calls: list[int] = []
    original = window_accounting.WindowBuffer.record_attempt

    def counting(self, losses=None):
        calls.append(1)
        return original(self, losses)

    monkeypatch.setattr(window_accounting.WindowBuffer, "record_attempt", counting)

    clean = _run(monkeypatch, num_steps=4, batches=4, loss_plan=[1.0, 2.0, 3.0, 4.0])
    clean.run()
    assert len(calls) == 1

    calls.clear()
    replayed = _run(
        monkeypatch,
        num_steps=2,
        batches=2,
        loss_plan=[float("nan"), 1.0, 1.0, 1.0],
    )
    replayed.run()
    assert len(calls) == 2, "原尝试 + 一次重放"


def test_discard_budget_is_blind_to_attribution_fields():
    """M26⑥：预算只读连续丢窗数，不因成因字段变化。"""
    args = SimpleNamespace()
    assert window_accounting.discard_budget(args) == 3
    args.adaptive_max_nonfinite_skips = 0
    assert window_accounting.discard_budget(
        SimpleNamespace(
            adaptive_max_nonfinite_skips=0,
            _adaptive_training_worker={"directory": "fixture"},
        )
    ) == 0

    state = SimpleNamespace()
    window_accounting.ensure_counters(state)
    state.window_consecutive_discards = 2
    state.window_discard_causes = {"nonfinite_loss": 99, "update_skipped": 7}
    assert window_accounting.discard_budget(args) == 3
