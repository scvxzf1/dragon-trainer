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
        args=SimpleNamespace(max_grad_norm=0, log_every_n_steps=1, max_train_steps=1),
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


def test_real_loop_scaler_skip_stops_before_scheduler_progress_or_snapshot(tmp_path, monkeypatch):
    trainer, state, runtime, events = setup_loop(tmp_path, monkeypatch, skipped=True)
    scheduler_step = state.lr_scheduler.last_epoch
    with pytest.raises(RuntimeError, match="scaler skipped"):
        loop._run_epoch_steps(trainer, state, 0)
    assert events == ["optimizer"]
    assert state.global_step == runtime.completed_steps == 0
    assert state.lr_scheduler.last_epoch == scheduler_step
    assert not list(tmp_path.iterdir())


def test_real_loop_tolerates_skip_within_budget_then_requires_effective_steps(tmp_path, monkeypatch):
    """预算 > 0 的端到端路径：跳步经真实 loop 被容忍、照常 commit，但不算有效步。

    覆盖 t14 指出的 M4 后半段：跳步不调用 optimizer.step、scheduler 与游标照常推进、
    commit 写完快照，最后由 finish() 因有效步数不足报 nonfinite_skip。
    """
    from library.training.adaptive_runtime.training_runtime import NonfiniteSkip

    trainer, state, runtime, events = setup_loop(tmp_path, monkeypatch, skipped=True, budget=2)
    loop._run_epoch_steps(trainer, state, 0)

    # 跳步仍走完整边界：optimizer 边界 → 进度条 → 训练模式（scheduler/快照照常推进）
    assert events == ["optimizer", "progress", "train_mode"]
    assert (runtime.attempted_steps, runtime.effective_steps, runtime.skipped_steps) == (1, 0, 1)
    assert runtime.completed_steps == state.global_step == 1, "游标照常推进"
    assert runtime.saved_state, "被容忍的跳步照常提交 checkpoint"

    with pytest.raises(NonfiniteSkip, match="effective steps"):
        runtime.finish(state)
