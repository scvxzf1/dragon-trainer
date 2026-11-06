from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from library.training import loop
from library.training.adaptive_runtime.training_runtime import TrainingRuntime


def setup_loop(tmp_path, monkeypatch, *, failure=None, skipped=False, budget=None):
    model = torch.nn.Linear(2, 1)
    args = None if budget is None else SimpleNamespace(adaptive_max_nonfinite_skips=budget)
    runtime = TrainingRuntime(tmp_path, args=args)

    def forward(*args, **kwargs):
        if failure == "forward":
            raise torch.cuda.OutOfMemoryError("injected")
        return model(torch.ones(1, 2)).square().mean()

    def backward(loss):
        if failure == "backward":
            raise torch.cuda.OutOfMemoryError("injected")
        loss.backward()

    optimizer = torch.optim.AdamW(model.parameters())
    steps = []

    def optimizer_step(*args, **kwargs):
        assert runtime.optimizer_started and runtime.stage == "optimizer"
        if failure == "optimizer":
            raise torch.cuda.OutOfMemoryError("injected after entry")
        if not skipped:
            original_step(*args, **kwargs)
        steps.append("optimizer")

    optimizer_step._wrapped_by_lr_sched = True
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    original_step = optimizer.step
    monkeypatch.setattr(optimizer, "step", optimizer_step)
    trainer = SimpleNamespace(
        _adaptive_training_runtime=runtime, _cudagraph_mark_step=False,
        _state=SimpleNamespace(personalization_observer={}),
        on_step_start=lambda *a, **k: None, process_batch=forward,
        run_after_backward=lambda *a, **k: None,
    )
    state = SimpleNamespace(
        args=SimpleNamespace(
            max_grad_norm=0,
            log_every_n_steps=1,
            max_train_steps=1,
            # 生产里 loop 与 runtime 共用同一份解析后的 args；丢窗预算由 loop 侧读取。
            # 本夹具是 adaptive 试验（TrainingRuntime），按裁定 B 走
            # --adaptive_max_nonfinite_skips（默认 0 严格），因此要带 adaptive 试验标记；
            # 普通训练则走 --max_nonfinite_window_discards（默认 3）。
            adaptive_max_nonfinite_skips=0 if budget is None else budget,
            _adaptive_training_worker={"directory": str(tmp_path)},
        ),
        accelerator=SimpleNamespace(sync_gradients=True, optimizer_step_was_skipped=skipped,
                                    accumulate=lambda model: nullcontext(), unwrap_model=lambda model: model,
                                    backward=backward),
        network=model, training_model=model, optimizer=optimizer, lr_scheduler=scheduler,
        text_encoder=None, unet=None, train_ctx=None, on_step_start_for_network=lambda *a: None,
        profile_started=False, profile_range=None, global_step=0, initial_step=0, is_tracking=False,
        stage_index=-1, current_step=SimpleNamespace(value=0), train_dataloader=[object()],
        progress_bar=SimpleNamespace(update=lambda n: steps.append("progress")),
        saver=SimpleNamespace(maybe_save_step=lambda *a: None),
        optimizer_train_fn=lambda: steps.append("train_mode"),
    )

    def save(directory, **kwargs):
        assert steps == ["optimizer", "progress", "train_mode"]
        assert state.current_step.value + 1 == state.global_step == 1
        if failure == "checkpoint":
            raise torch.cuda.OutOfMemoryError("injected during save")
        path = Path(directory)
        for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "scaler.pt",
                     "random_states_0.pkl", "adaptive_precision.json"):
            (path / name).write_text("fixture")
        (path / "train_state.json").write_text(json.dumps({"current_step": 1}))

    state.accelerator.save_state = save
    for name in ("_maybe_apply_stage_schedule", "_record_recent_step_seconds", "_sample_at_step",
                 "_log_step", "_maybe_run_step_validation"):
        monkeypatch.setattr(loop, name, lambda *a, **k: None)
    monkeypatch.setattr(loop, "_maybe_scale_norm", lambda state: (None, None, None, {}))
    return trainer, state, runtime, steps


@pytest.mark.parametrize("failure,started", [("forward", False), ("backward", False),
                                            ("optimizer", True), ("checkpoint", True)])
def test_real_loop_failure_phase_and_sticky_optimizer_flag(tmp_path, monkeypatch, failure, started):
    trainer, state, runtime, _ = setup_loop(tmp_path, monkeypatch, failure=failure)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        loop._run_epoch_steps(trainer, state, 0)
    assert runtime.stage == failure and runtime.optimizer_started is started
    assert runtime.completed_steps == 0 and runtime.saved_state is None


def test_real_loop_commits_after_increment_and_train_mode(tmp_path, monkeypatch):
    trainer, state, runtime, _ = setup_loop(tmp_path, monkeypatch)
    loop._run_epoch_steps(trainer, state, 0)
    assert runtime.completed_steps == state.global_step == 1
    assert runtime.saved_state and not runtime.finished
    runtime.finish(state)
    assert runtime.finished


def test_real_loop_scaler_skip_is_replayed_then_discarded_without_progress(tmp_path, monkeypatch):
    """E3 裁定后：scaler 跳步先整窗重放一次，仍失败则丢窗——进度一律不推进。

    用户裁定（E3）把「跳步」从「计一步」改成「重放同一窗口，最多 1 次，仍失败则丢弃该窗口」，
    因此 D3 的「跳步照常 commit」语义被取代。本用例保留原断言的核心不变量（跳步不得推进
    scheduler、不得写快照、不得推进游标），并新增重放/丢窗计数断言。
    """
    from library.training.adaptive_runtime.training_runtime import NonfiniteSkip

    trainer, state, runtime, events = setup_loop(tmp_path, monkeypatch, skipped=True)
    scheduler_step = state.lr_scheduler.last_epoch
    # 默认预算 0 = 严格契约：重放 1 次仍失败 → 丢窗 → 立刻以 nonfinite_skip 升级终止。
    with pytest.raises(NonfiniteSkip, match="consecutive > budget 0"):
        loop._run_epoch_steps(trainer, state, 0)
    assert events == ["optimizer", "optimizer"], "同一窗口必须重放一次（两次边界尝试）"
    assert state.global_step == runtime.completed_steps == 0
    assert state.lr_scheduler.last_epoch == scheduler_step, "失败尝试不得推进 scheduler"
    assert not list(tmp_path.iterdir()), "丢窗不得写快照"
    assert state.window_discards == 1 and state.window_replays == 1
    # 重放期间不得把中间失败喂给 adaptive 运行时（只在最终提交时通知）
    assert (runtime.attempted_steps, runtime.effective_steps, runtime.skipped_steps) == (0, 0, 0)


def test_real_loop_budget_does_not_turn_replayed_skips_into_steps(tmp_path, monkeypatch):
    """预算 > 0 也不再让跳步「计步」：重放 → 丢窗优先于预算容忍。

    E3 裁定后，跳步窗口先重放、再退化丢弃，循环侧不推进任何计数；adaptive 预算代码没有改动，
    但它已经看不到中间失败，因此容错不再表现为「跳步算一步」。试运行最终由 finish() 以
    nonfinite_skip 收口（有效步数不足），不会被丢窗卡死。
    """
    from library.training.adaptive_runtime.training_runtime import NonfiniteSkip

    trainer, state, runtime, events = setup_loop(tmp_path, monkeypatch, skipped=True, budget=2)
    loop._run_epoch_steps(trainer, state, 0)

    assert events == ["optimizer", "optimizer"], "重放一次，仍失败后丢弃"
    assert (runtime.attempted_steps, runtime.effective_steps, runtime.skipped_steps) == (0, 0, 0)
    assert runtime.completed_steps == state.global_step == 0, "游标不得推进"
    assert not runtime.saved_state, "丢窗不得写 checkpoint"
    assert state.window_discards == 1

    with pytest.raises(NonfiniteSkip, match="effective steps"):
        runtime.finish(state)
