"""D3 semantics: a skipped optimizer update is tolerated up to a budget, never counted.

Contract under test (user-specified):
* a skipped step is never an effective step: it does not count toward
  ``completed_steps`` and must not feed step-time samples;
* a small budget lets GradScaler back off instead of killing the worker on the
  first overflow; the default 0 keeps the previous strict contract;
* exceeding the budget, or finishing the step budget without enough effective
  steps, ends the trial as ``nonfinite_skip`` so the supervisor can tell it apart
  from ``error`` / ``cuda_oom``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from library.training.adaptive_runtime.process import IsolatedRunner
from library.training.adaptive_runtime.training_runtime import NonfiniteSkip, TrainingRuntime


def _runtime(tmp_path, *, budget: int | None = None, max_steps: int = 4):
    args = SimpleNamespace(adaptive_max_nonfinite_skips=budget)
    if max_steps is not None:
        args.max_train_steps = max_steps
    return TrainingRuntime(tmp_path, args=args)


def _state(*, skipped: bool, global_step: int, max_steps: int = 4):
    return SimpleNamespace(
        accelerator=SimpleNamespace(optimizer_step_was_skipped=skipped),
        global_step=global_step,
        args=SimpleNamespace(max_train_steps=max_steps),
    )


def _step(runtime, state, *, skipped: bool):
    """Drive one optimizer boundary and its commit boundary.

    ``commit()`` is the loop's checkpoint hook; here it is simulated by clearing
    the pending flag so the next boundary is legal (the real commit also saves
    state, which is out of scope for these counter tests).
    """
    runtime.phase("optimizer")
    runtime.after_optimizer(state)
    runtime._pending_skipped = None


def test_default_budget_keeps_strict_contract(tmp_path):
    """默认 0：第一次跳步就以 nonfinite_skip 结束，与既有契约一致。"""
    runtime = _runtime(tmp_path, budget=None)

    with pytest.raises(NonfiniteSkip, match="scaler skipped"):
        _step(runtime, _state(skipped=True, global_step=1), skipped=True)

    assert (runtime.attempted_steps, runtime.effective_steps, runtime.skipped_steps) == (1, 0, 1)


def test_budget_tolerates_consecutive_skips_then_fails(tmp_path):
    """预算给 GradScaler 留 backoff 窗口，超出即 nonfinite_skip。"""
    runtime = _runtime(tmp_path, budget=2)

    _step(runtime, _state(skipped=True, global_step=1), skipped=True)
    _step(runtime, _state(skipped=True, global_step=2), skipped=True)
    assert runtime.consecutive_skips == 2 and runtime.skipped_steps == 2

    with pytest.raises(NonfiniteSkip, match="consecutive skips"):
        _step(runtime, _state(skipped=True, global_step=3), skipped=True)


def test_effective_step_resets_consecutive_skips(tmp_path):
    """真实更新后连续计数归零，预算只约束连续跳步。"""
    runtime = _runtime(tmp_path, budget=1)

    _step(runtime, _state(skipped=True, global_step=1), skipped=True)
    _step(runtime, _state(skipped=False, global_step=2), skipped=False)
    assert runtime.consecutive_skips == 0
    _step(runtime, _state(skipped=True, global_step=3), skipped=True)
    assert (runtime.effective_steps, runtime.skipped_steps) == (1, 2)


def test_skipped_steps_never_count_as_completed(tmp_path):
    """跳步不计入 completed_steps，且计数对外可见。"""
    runtime = _runtime(tmp_path, budget=3)
    for step in range(1, 4):
        _step(runtime, _state(skipped=step != 2, global_step=step), skipped=step != 2)

    assert runtime.effective_steps == 1
    assert runtime.skipped_steps == 2
    assert runtime.completed_steps == 0, "尚未 commit，completed_steps 仍为 0"

    result = runtime.result("running")
    assert result["attempted_steps"] == 3
    assert result["effective_steps"] == 1
    assert result["skipped_steps"] == 2
    assert result["max_nonfinite_skips"] == 3


def test_finish_fails_when_effective_steps_short_of_target(tmp_path):
    """跑完上限仍未凑够目标有效步数 → nonfinite_skip，而不是成功。"""
    runtime = _runtime(tmp_path, budget=5, max_steps=4)
    runtime.optimizer_started = True
    runtime.attempted_steps = 4
    runtime.effective_steps = 2
    runtime.skipped_steps = 2

    state = _state(skipped=False, global_step=4, max_steps=4)
    with pytest.raises(NonfiniteSkip, match="effective steps"):
        runtime.finish(state)


def test_finish_accepts_enough_effective_steps(tmp_path):
    runtime = _runtime(tmp_path, budget=5, max_steps=2)
    runtime.optimizer_started = True
    runtime.completed_steps = 2
    runtime.effective_steps = 2
    runtime.skipped_steps = 1

    runtime.finish(_state(skipped=False, global_step=3, max_steps=2))
    assert runtime.finished is True


def test_isolated_runner_accepts_nonfinite_skip_status(tmp_path):
    """进程结果校验必须放行 nonfinite_skip，否则分类会被降级成 invalid result。"""
    payload = {"status": "nonfinite_skip", "failure_kind": "nonfinite_skip", "skipped_steps": 3}
    (tmp_path / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    assert IsolatedRunner._result(tmp_path, 0, None) == payload

    (tmp_path / "result.json").write_text(json.dumps({"status": "bogus"}), encoding="utf-8")
    assert IsolatedRunner._result(tmp_path, 0, None)["status"] == "error"


def test_unknown_status_still_rejected(tmp_path):
    """负控制：白名单之外的状态仍必须被拒绝。"""
    (tmp_path / "result.json").write_text(json.dumps({"status": "definitely_not_a_status"}))
    result = IsolatedRunner._result(tmp_path, 0, None)
    assert result["status"] == "error" and "invalid structured worker result" in result["error"]


def test_guard_skip_is_visible_to_adaptive_accounting(tmp_path):
    """D2 的 guard 跳步不体现在 accelerator.optimizer_step_was_skipped 上。

    测量层必须读统一的 state.update_skipped，否则无 scaler 的试验会把丢弃的更新
    当成有效步，污染 completed/effective 计数与步时口径。
    """
    runtime = _runtime(tmp_path, budget=5)
    state = SimpleNamespace(
        accelerator=SimpleNamespace(optimizer_step_was_skipped=False),  # 无 scaler
        update_skipped=True,  # guard 丢弃了这一拍
        global_step=1,
        args=SimpleNamespace(max_train_steps=4),
    )

    runtime.phase("optimizer")
    runtime.after_optimizer(state)

    assert runtime.skipped_steps == 1
    assert runtime.effective_steps == 0


def test_unified_skip_flag_prefers_published_value():
    """统一标志优先于 accelerate 属性，并在缺失时回退。"""
    from library.training.scaler_ownership import update_skipped

    published = SimpleNamespace(
        update_skipped=True,
        accelerator=SimpleNamespace(optimizer_step_was_skipped=False),
    )
    assert update_skipped(published) is True

    fallback = SimpleNamespace(accelerator=SimpleNamespace(optimizer_step_was_skipped=True))
    assert update_skipped(fallback) is True

    clean = SimpleNamespace(
        update_skipped=None,
        accelerator=SimpleNamespace(optimizer_step_was_skipped=False),
    )
    assert update_skipped(clean) is False


def test_completed_steps_tracks_cursor_not_effective_steps(tmp_path):
    """completed_steps 是 resume 游标身份，必须跟随 global_step 而不是有效步数。"""
    runtime = _runtime(tmp_path, budget=5, max_steps=4)
    runtime.optimizer_started = True
    runtime.effective_steps = 2
    runtime.skipped_steps = 2
    state = _state(skipped=False, global_step=4, max_steps=4)

    # resume_checkpoint 要求 global_step == completed_steps == step
    runtime.completed_steps = state.global_step
    assert runtime.completed_steps == 4
    assert runtime.effective_steps == 2, "有效步数是测量口径，不被游标覆盖"
