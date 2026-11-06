"""E3 窗口重放：非有限窗口重放 1 次、仍失败则整窗丢弃。

用户裁定（2026-11-06）：窗口内出现非有限（loss NaN/inf，或梯度非有限导致 scaler 跳步 /
D2 guard 丢弃）→ 整个累积窗口作废并**重放同一窗口数据**；最多重放 1 次，仍失败则丢弃该窗口、
计入 ``window_discards`` 并继续训练。丢弃与失败尝试都**不得产生任何进度副作用**。

本文件驱动真实 ``library/training/loop.py::_run_epoch_steps``（夹具见
``tests/window_replay_harness.py``），不加载模型、不建 CUDA 上下文。
"""

from types import SimpleNamespace

import pytest
import torch

from library.training import loop, window_accounting

try:  # pytest 把 tests/ 放进 sys.path；显式回退便于直接导入
    from window_replay_harness import FakeAccelerator, Harness
except ImportError:  # pragma: no cover
    from tests.window_replay_harness import FakeAccelerator, Harness


def test_nonfinite_microbatch_is_replayed_then_committed_once(tmp_path, monkeypatch):
    """窗口内首批 NaN → 整窗重放 → 第二次成功：计数只推进 1 次，账本只记成功那次。"""
    harness = Harness(
        monkeypatch,
        num_steps=1,
        loss_plan=[float("nan"), 1.5, 2.5],
        batches=2,
    )
    harness.run()
    state = harness.state

    assert state.global_step == 2, "两个窗口各自只提交一次"
    assert harness.optimizer.steps == 2, "重放的失败尝试不得真正更新权重"
    assert harness.forward_calls == 3, "第一个窗口 2 次尝试（原 + 重放）+ 第二个窗口 1 次"
    assert state.window_replays == 1
    assert state.window_discards == 0
    assert state.window_attempts == 3
    assert harness.scheduler.steps == 2, "scheduler 只随提交推进"
    assert harness.saves == [1, 2], "checkpoint 只随提交写"

    # 账本只记成功那次（1.5 / 2.5），NaN 不入账
    assert state.window_last_loss == pytest.approx(2.5, abs=0)
    assert state.window_num_steps == 1
    assert state.window_total_loss == pytest.approx(2.5, abs=1e-12)
    assert state.ema_observation["step"] == 2, "EMA 占位槽只在提交路径写"


def test_scaler_skip_replays_instead_of_counting_a_step(tmp_path, monkeypatch):
    """scaler 跳步（optimizer_step_was_skipped=True）同样触发重放，不直接吃步。"""
    harness = Harness(
        monkeypatch,
        num_steps=1,
        loss_plan=[1.0, 1.0, 1.0],
        skip_plan=[True, False, False],
        batches=2,
    )
    harness.run()
    state = harness.state

    assert state.window_replays == 1
    assert state.window_discards == 0
    assert state.global_step == 2
    assert harness.optimizer.steps == 2
    assert harness.forward_calls == 3
    assert state.update_skipped is False, "提交路径上的窗口必须是真正应用的更新"


def test_two_failures_discard_window_and_training_continues(tmp_path, monkeypatch):
    """连续两次失败 → 丢窗：计数与账本均不推进，训练继续到下一窗口。"""
    harness = Harness(
        monkeypatch,
        num_steps=1,
        loss_plan=[float("nan"), float("nan"), 3.0],
        batches=2,
    )
    harness.run()
    state = harness.state

    assert state.window_discards == 1
    assert state.window_replays == 1
    assert state.global_step == 1, "丢窗不推进 global_step；第二个窗口提交 1 次"
    assert harness.optimizer.steps == 1
    assert harness.scheduler.steps == 1
    assert harness.saves == [1], "丢窗不写 checkpoint"
    assert state.window_last_loss == pytest.approx(3.0, abs=0)
    assert state.window_total_loss == pytest.approx(3.0, abs=1e-12)


def test_mid_window_nonfinite_loss_invalidates_whole_window(tmp_path, monkeypatch):
    """累积窗口中间步 NaN（sync_gradients=False）也要整窗作废。"""
    harness = Harness(
        monkeypatch,
        num_steps=2,
        loss_plan=[1.0, float("nan"), 1.0, 1.0],
        batches=2,
    )
    harness.run()
    state = harness.state

    assert state.window_replays == 1, "窗口中间步 NaN 必须触发重放"
    assert harness.optimizer.steps == 1, "整个窗口只提交一次更新"
    assert state.window_num_steps == 2, "账本记满窗口的 2 个微批"
    assert state.window_total_loss == pytest.approx(2.0, abs=1e-12)
    assert state.global_step == 1


def test_clean_windows_are_bitwise_unchanged_by_replay_machinery(tmp_path, monkeypatch):
    """负控制：不触发重放时，边界口径与改动前一致（无重放、无丢窗、每窗口一次提交）。"""
    harness = Harness(monkeypatch, num_steps=2, loss_plan=[1.0, 2.0], batches=2)
    harness.run()
    state = harness.state

    assert state.window_replays == 0 and state.window_discards == 0
    assert state.window_attempts == 1
    assert state.global_step == 1
    assert harness.optimizer.steps == 1, "累积 2 个微批只走一次 optimizer.step"
    assert harness.forward_calls == 2, "干净路径不得重放"
    assert state.window_num_steps == 2
    assert state.window_total_loss == pytest.approx(3.0, abs=1e-12)
    assert state.update_skipped is False


def test_replay_sets_boundary_explicitly_when_dataloader_is_exhausted():
    """重放必须显式设边界：end_of_dataloader 下再走 accumulate 会每个微批都 sync。"""
    accelerator = FakeAccelerator(num_steps=4)
    accelerator.gradient_state.end_of_dataloader = True
    seen: list[bool] = []

    def run():
        with loop._forced_accumulate(
            accelerator, torch.nn.Linear(1, 1), boundary=index == 3
        ):
            seen.append(accelerator.sync_gradients)

    for index in range(4):
        run()

    assert seen == [False, False, False, True], "只有窗口最后一步是边界"
    assert accelerator.gradient_state.sync_gradients is False, "退出后恢复外层状态"


def test_window_buffer_failure_reasons():
    """账本策略单测：两种失败原因与「丢弃不动账本」的语义。"""
    buffer = window_accounting.WindowBuffer()
    buffer.add("batch", torch.tensor(1.0))
    assert buffer.failure_reason(update_skipped=False) is None
    assert buffer.failure_reason(update_skipped=True) == window_accounting.REASON_UPDATE_SKIPPED

    nan_buffer = window_accounting.WindowBuffer()
    nan_buffer.add("batch", torch.tensor(float("nan")))
    assert nan_buffer.failure_reason(update_skipped=False) == window_accounting.REASON_NONFINITE_LOSS

    state = SimpleNamespace()
    window_accounting.commit_window(state, buffer)
    before = window_accounting.ledger_snapshot(state)
    nan_buffer.record_attempt()
    dropped = window_accounting.discard_window(state, nan_buffer, "nonfinite_loss")
    after = window_accounting.ledger_snapshot(state)

    assert dropped["discards"] == 1
    assert after["window_discards"] == before["window_discards"] + 1
    for field in ("window_total_loss", "window_num_steps", "window_last_loss", "ema_observation"):
        assert after[field] == before[field], f"丢窗不得改动 {field}"


def test_discard_escalates_to_nonfinite_skip_when_budget_is_zero(monkeypatch):
    """预算层（D3 的 --adaptive_max_nonfinite_skips）保留：连续丢窗超预算即 nonfinite_skip。

    丢窗不动 adaptive 计数器（它看到的边界只有提交过的窗口），因此预算在窗口接缝上执行，
    默认 0 立刻升级——分类仍是 ``NonfiniteSkip``，供 supervisor 识别为可重试的 fp16 溢出。
    """
    from library.training.adaptive_runtime.training_runtime import NonfiniteSkip

    harness = Harness(
        monkeypatch,
        num_steps=1,
        loss_plan=[float("nan"), float("nan")],
        batches=1,
        nonfinite_skip_budget=0,
    )
    with pytest.raises(NonfiniteSkip, match="consecutive > budget 0"):
        harness.run()

    assert harness.state.window_discards == 1
    assert harness.state.window_replays == 1
    assert harness.state.global_step == 0
    assert harness.optimizer.steps == 0 and harness.scheduler.steps == 0


def test_discard_budget_is_consecutive_and_resets_on_effective_window(monkeypatch):
    """预算按「连续」计：一次丢窗被容忍 → 真实更新复位 → 再丢一次仍被容忍。"""
    harness = Harness(
        monkeypatch,
        num_steps=1,
        loss_plan=[float("nan"), float("nan"), 1.0, float("nan"), float("nan")],
        batches=4,
        nonfinite_skip_budget=1,
    )
    harness.run()
    state = harness.state

    assert state.window_discards == 2, "两个窗口被丢弃"
    assert state.window_replays == 2
    assert state.global_step == 2, "两个干净窗口各提交 1 次（第 4 个 batch 走默认 loss）"
    # 末尾窗口提交后连续计数归零；它能归零本身就是「中间那次提交复位过」的证据——
    # 否则第二次丢窗会是 consecutive=2 > budget=1，早已抛 NonfiniteSkip。
    assert state.window_consecutive_discards == 0, "提交路径必须复位连续丢窗计数"
    assert harness.optimizer.steps == 2
    assert state.window_total_loss == pytest.approx(1.0, abs=1e-12)
