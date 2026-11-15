from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from library.training import loop as training_loop
from library.training.loop import (
    _log_step,
    _recent_step_seconds,
    _record_recent_step_seconds,
    _run_sample_with_probe,
)


def _rate_state() -> SimpleNamespace:
    return SimpleNamespace(
        recent_step_rate_last=None,
        recent_step_seconds=deque(maxlen=9),
    )


def test_recent_step_seconds_starts_after_second_step() -> None:
    state = _rate_state()

    _record_recent_step_seconds(state, 1, now=100.0)
    assert _recent_step_seconds(state) is None

    _record_recent_step_seconds(state, 2, now=102.0)
    assert _recent_step_seconds(state) == pytest.approx(2.0)


def test_recent_step_seconds_uses_sliding_median() -> None:
    state = _rate_state()
    for step, now in ((1, 100.0), (2, 102.0), (3, 104.0), (4, 112.0)):
        _record_recent_step_seconds(state, step, now=now)

    assert list(state.recent_step_seconds) == [2.0, 2.0, 8.0]
    assert _recent_step_seconds(state) == pytest.approx(2.0)


def test_recent_step_seconds_resets_on_non_monotonic_sample() -> None:
    state = _rate_state()
    _record_recent_step_seconds(state, 1, now=100.0)
    _record_recent_step_seconds(state, 2, now=102.0)
    _record_recent_step_seconds(state, 1, now=110.0)

    assert list(state.recent_step_seconds) == []
    assert state.recent_step_rate_last == (110.0, 1)


class _LossRecorder:
    moving_average = 0.5

    def add(self, **_kwargs) -> None:
        return None


class _ProgressBar:
    def __init__(self) -> None:
        self.postfix = None

    def set_postfix(self, *, refresh, **kwargs) -> None:
        self.postfix = {"refresh": refresh, **kwargs}


def test_log_step_adds_recent_rate_postfix_without_step_unit() -> None:
    progress_bar = _ProgressBar()
    state = SimpleNamespace(
        args=SimpleNamespace(log_every_n_steps=1, max_train_steps=10),
        accelerator=SimpleNamespace(
            sync_gradients=True,
            device="cpu",
            unwrap_model=lambda network: network,
        ),
        global_step=2,
        loss_recorder=_LossRecorder(),
        progress_bar=progress_bar,
        network=SimpleNamespace(),
        is_tracking=False,
        recent_step_rate_last=(100.0, 1),
        recent_step_seconds=deque([1.92], maxlen=9),
    )
    trainer = SimpleNamespace(progress_sink=None)
    loss = SimpleNamespace(detach=lambda: SimpleNamespace(item=lambda: 0.5))

    _log_step(
        trainer,
        state,
        loss=loss,
        step=1,
        epoch=0,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        max_mean_logs={},
    )

    assert progress_bar.postfix["refresh"] is False
    assert progress_bar.postfix["recent_s_per_step"] == "1.92"
    assert "s/step" not in progress_bar.postfix["recent_s_per_step"]


def test_sample_memory_probe_records_boundaries_and_exceptions() -> None:
    events: list[dict] = []

    class Probe:
        def should_record_step(self, _step):
            return True

        def snapshot(self, label, **fields):
            events.append({"label": label, **fields})

    args = SimpleNamespace(
        sample_at_first=True,
        sample_every_n_steps=2,
        sample_every_n_epochs=None,
    )
    accelerator = SimpleNamespace(device="cpu")
    trainer = SimpleNamespace(memory_probe=Probe())

    _run_sample_with_probe(
        trainer,
        accelerator,
        args,
        None,
        0,
        lambda: events.append({"label": "callback"}),
    )
    assert [event["label"] for event in events] == [
        "sample_before",
        "callback",
        "sample_after",
    ]
    assert events[0]["phase"] == "sample"
    assert events[0]["sample_epoch"] is None

    events.clear()
    with pytest.raises(RuntimeError, match="sample failed"):
        _run_sample_with_probe(
            trainer,
            accelerator,
            args,
            None,
            2,
            lambda: (_ for _ in ()).throw(RuntimeError("sample failed")),
        )
    assert [event["label"] for event in events] == ["sample_before", "sample_exception"]
    assert events[-1]["exception_type"] == "RuntimeError"


class _FakeAccelerator:
    is_main_process = True
    device = "cpu"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def print(self, *_args, **_kwargs) -> None:
        return None

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self) -> None:
        self.events.append("wait")


class _FakeNetwork:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def on_epoch_start(self, _text_encoder, _unet) -> None:
        self.events.append("epoch_start")


class _FakeSaver:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def maybe_save_epoch(self, _network, global_step, epoch, num_train_epochs, *, loader_cursor=None, digest_context=None) -> None:
        assert (global_step, epoch, num_train_epochs) == (6, 5, 6)
        self.events.append("save_epoch")

    def maybe_save_resumable(
        self, _network, global_step, epoch, num_train_epochs, *, loader_cursor=None, digest_context=None
    ) -> None:
        assert (global_step, epoch, num_train_epochs) == (6, 5, 6)
        self.events.append("save_resumable")


class _FakeTrainer:
    _adapters = []

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def sample_images(self, *_args, **_kwargs) -> None:
        self.events.append("sample")


def test_final_step_runs_epoch_end_artifacts_before_loop_exit(monkeypatch) -> None:
    events: list[str] = []
    state = SimpleNamespace(
        args=SimpleNamespace(max_train_steps=6),
        accelerator=_FakeAccelerator(events),
        current_epoch=SimpleNamespace(value=0),
        metadata={},
        epoch_to_start=5,
        num_train_epochs=6,
        text_encoder=None,
        unet=None,
        network=_FakeNetwork(events),
        global_step=5,
        saver=_FakeSaver(events),
        optimizer_eval_fn=lambda: events.append("optimizer_eval"),
        optimizer_train_fn=lambda: events.append("optimizer_train"),
        vae=None,
        tokenizers=None,
        stage_index=-1,
    )

    monkeypatch.setattr(
        training_loop, "_maybe_apply_stage_schedule", lambda *_args, **_kwargs: None
    )

    def run_steps(_trainer, loop_state, _epoch) -> None:
        loop_state.global_step = 6
        events.append("steps")

    monkeypatch.setattr(training_loop, "_run_epoch_steps", run_steps)
    monkeypatch.setattr(
        training_loop,
        "_run_epoch_validation",
        lambda *_args: events.append("validation"),
    )
    monkeypatch.setattr(
        training_loop, "_log_epoch_average", lambda *_args: events.append("epoch_log")
    )
    monkeypatch.setattr(
        training_loop,
        "_run_adapter_epoch_hooks",
        lambda *_args: events.append("adapter_hooks"),
    )

    training_loop.run_training_loop(_FakeTrainer(events), state)

    assert events == [
        "epoch_start",
        "steps",
        "validation",
        "epoch_log",
        "adapter_hooks",
        "wait",
        "optimizer_eval",
        "save_epoch",
        "save_resumable",
        "sample",
        "optimizer_train",
    ]
    assert state.current_epoch.value == 6
    assert "ss_training_finished_at" in state.metadata
