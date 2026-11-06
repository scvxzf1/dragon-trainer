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
    from window_replay_harness import FakeAccelerator, Harness, RealAcceleratorHarness
except ImportError:  # pragma: no cover
    from tests.window_replay_harness import FakeAccelerator, Harness, RealAcceleratorHarness


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


def test_mid_window_nonfinite_loss_invalidates_whole_window(monkeypatch):
    """窗口**中间**步 NaN（sync_gradients=False）也必须让整窗作废，且失败尝试不动权重。

    这条专门覆盖 ``force_skip``：NaN 不在边界上，边界微批自己的 loss 是有限的，
    若没有 ``force_skip`` 就会正常 ``optimizer.step()``，failed attempt 便已改权重。
    窗口 3 个微批、NaN 落在第 1 个；重放 3 个有限微批后提交一次。
    """
    seen: dict[str, list[torch.Tensor]] = {}

    def snapshot_at_window_start(item):
        if item.forward_calls == 0:
            seen["start"] = [p.detach().clone() for p in item.model.parameters()]

    def snapshot_before_replay(item):
        seen["before_replay"] = [p.detach().clone() for p in item.model.parameters()]

    harness = Harness(
        monkeypatch,
        num_steps=3,
        loss_plan=[float("nan"), 1.0, 1.0] + [1.0, 1.0, 1.0],
        batches=3,
        on_before_forward=snapshot_at_window_start,
        on_before_replay=snapshot_before_replay,
    )
    harness.run()
    state = harness.state

    assert state.window_replays == 1, "窗口中间步 NaN 必须触发重放"
    assert "before_replay" in seen, "重放钩子未触发"
    for before, after in zip(seen["start"], seen["before_replay"]):
        assert torch.equal(before, after), (
            "窗口中间步 NaN 之后，边界微批不得推动权重（force_skip 必须生效）"
        )
    assert harness.optimizer.steps == 1, "整个窗口只提交一次更新（重放那次）"
    assert state.window_num_steps == 3, "账本记满窗口的 3 个微批"
    assert state.window_total_loss == pytest.approx(3.0, abs=1e-12)
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


def test_forced_accumulate_context_manager_sets_boundary_only():
    """只覆盖 ``_forced_accumulate`` 上下文管理器本身（显式边界 + 退出后恢复）。

    真实 accelerate 下的端到端行为（``manage_accumulate=False`` 才真正生效）由本文件
    末尾的 ``test_real_accelerator_*`` 三条真机级用例覆盖；本条不能替代它们。
    """
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


def test_plain_training_default_budget_tolerates_three_discards_then_escalates(monkeypatch):
    """裁定 B：普通训练默认 --max_nonfinite_window_discards=3。

    连续 3 个窗口重放后仍失败 → 都被容忍（丢窗计数、继续训练），第 4 个才 NonfiniteSkip。
    这让「重放 → 丢窗 → 下一窗恢复」在默认配置下真的可达，同时快速熔断真发散。
    """
    from library.training.adaptive_runtime.training_runtime import NonfiniteSkip

    harness = Harness(
        monkeypatch,
        num_steps=1,
        loss_plan=[float("nan")] * 8,  # 4 个窗口 × (原 + 重放)
        batches=4,
    )
    with pytest.raises(NonfiniteSkip, match="consecutive > budget 3"):
        harness.run()

    state = harness.state
    assert state.window_discards == 4, "前 3 次被容忍，第 4 次升级"
    assert state.window_replays == 4
    assert state.global_step == 0
    assert harness.optimizer.steps == 0 and harness.scheduler.steps == 0


def test_adaptive_trial_default_budget_escalates_on_first_discard(monkeypatch):
    """裁定 B：adaptive 试验仍用 --adaptive_max_nonfinite_skips，默认 0 = 第一次丢窗即终止。"""
    from library.training.adaptive_runtime.training_runtime import NonfiniteSkip

    harness = Harness(
        monkeypatch,
        num_steps=1,
        loss_plan=[float("nan"), float("nan")],
        batches=1,
        adaptive_trial=True,
        # 普通训练预算即使放宽到 9 也不影响 adaptive 判定
        window_discard_budget=9,
    )
    with pytest.raises(NonfiniteSkip, match="consecutive > budget 0"):
        harness.run()
    assert harness.state.window_discards == 1


def test_adaptive_budget_is_independent_from_plain_training_budget(monkeypatch):
    """两套预算互不影响：adaptive 用 2 可容忍 2 次丢窗，普通训练读自己的键。"""
    harness = Harness(
        monkeypatch,
        num_steps=1,
        loss_plan=[float("nan"), float("nan"), float("nan"), float("nan"), 1.0],
        batches=3,
        adaptive_trial=True,
        adaptive_skip_budget=2,
        window_discard_budget=0,  # 若误读普通预算，第 1 次丢窗就会抛
    )
    harness.run()
    state = harness.state

    assert state.window_discards == 2, "adaptive 预算 2 → 两次丢窗仍被容忍"
    assert state.global_step == 1, "第 3 个窗口提交"
    assert state.window_consecutive_discards == 0


# ---------------------------------------------------------------------------
# 真机级回归（真实 accelerate Accelerator + 真 DataLoaderShard，CPU 即可）：
# blocker B1 —— `_forced_accumulate` 的边界必须真的生效。
# ---------------------------------------------------------------------------
def test_real_accelerator_replay_at_end_of_dataloader_applies_one_update(monkeypatch):
    """eod=True 下重放整窗恰好 1 次优化器应用，权重只更新一次（B1 回归）。"""
    real = RealAcceleratorHarness(monkeypatch, num_steps=2, total_batches=2)
    try:
        before = real.weights()
        for index, _batch in enumerate(real.loader):
            if index != len(real.loader) - 1:
                continue
            assert real.accelerator.gradient_state.end_of_dataloader is True
            step_counter_before = real.accelerator.step
            losses, _last = loop._replay_window(real.trainer, real.state, list(real.data))
            assert real.accelerator.step == step_counter_before, "重放不得挪动 acc.step"
        assert all(torch.isfinite(torch.tensor(value)) for value in losses)
        assert real.applied == 1, "整窗只允许一次真实更新"
        assert sum(0 if torch.equal(b, p.detach()) else 1 for b, p in zip(before, real.weights())) == 2
        assert real.state.update_skipped is False
    finally:
        real.close()


def test_real_accelerator_replay_with_nan_leaves_weights_bitwise_unchanged(monkeypatch):
    """重放仍含 NaN → 该窗口必然被丢弃，权重必须逐位不变（B1 回归，最严的一条）。"""
    from library.training.adaptive_runtime.training_runtime import NonfiniteSkip

    real = RealAcceleratorHarness(
        monkeypatch, num_steps=2, total_batches=2, nan_calls={2}
    )
    try:
        before = real.weights()
        for index, _batch in enumerate(real.loader):
            if index != len(real.loader) - 1:
                continue
            assert real.accelerator.gradient_state.end_of_dataloader is True
            losses, _last = loop._replay_window(real.trainer, real.state, list(real.data))
        assert not all(torch.isfinite(torch.tensor(value)) for value in losses), "构造前提：重放含 NaN"
        assert real.applied == 0, "含 NaN 的窗口不得有任何一次真实更新"
        for left, right in zip(before, real.weights()):
            assert torch.equal(left, right), "丢弃窗口必须权重逐位不变"
        assert real.state.update_skipped is True
        assert isinstance(NonfiniteSkip("x"), RuntimeError)  # 升级类型未被改动
    finally:
        real.close()


def test_real_accelerator_window_sequence_keeps_sync_landing(monkeypatch):
    """重放之后，下一个窗口的 sync 落点不得提前/延后（num_steps=2、三个窗口）。"""
    real = RealAcceleratorHarness(
        monkeypatch, num_steps=2, total_batches=6, nan_calls={3}
    )
    try:
        real.state.args.max_train_steps = 3
        loop._run_epoch_steps(real.trainer, real.state, 0)
        state = real.state

        assert state.window_replays == 1, "第 2 个窗口的首个微批 NaN → 重放一次"
        assert state.window_discards == 0
        assert state.global_step == 3, "三个窗口各提交一次"
        assert real.applied == 3, "每个窗口恰好一次真实更新"

        landings = [forward_calls for in_sync, moved, forward_calls in real.sync_landing if moved]
        assert landings == [2, 6, 8], (
            f"边界落点必须是每个窗口的最后一个微批；实际 {landings}"
        )
        assert landings[2] - landings[1] == 2, "重放后下一个窗口仍是满 2 个微批才 sync"
    finally:
        real.close()
