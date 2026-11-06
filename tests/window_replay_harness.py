"""共享夹具：驱动真实 ``library/training/loop.py::_run_epoch_steps`` 的最小 loop。

供 ``tests/test_window_replay.py``（E3 重放行为）与
``tests/test_fp16_nan_static_equivalence.py::TestE3WindowAccountingGap``（E3 规格 8 观测点）
共用。只把 trainer / accelerator / optimizer 填成最小数据容器，不加载模型、不建 CUDA 上下文。

文件名不带 ``test_`` 前缀，pytest 不会把它当用例收集。
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import torch

from library.training import loop


class FakeGradientState:
    """加速器梯度状态的最小替身：单调计数 + 显式 sync 开关。"""

    def __init__(self, num_steps: int) -> None:
        self.num_steps = num_steps
        self.step = 0
        self.sync_gradients = False
        self.end_of_dataloader = True  # 让 accumulate 与 replay 的分歧可被观察

    def _set_sync_gradients(self, value: bool) -> None:
        self.sync_gradients = bool(value)


class FakeAccelerator:
    def __init__(self, num_steps: int = 1) -> None:
        self.mixed_precision = "no"
        self.scaler = None
        self.num_processes = 1
        self.device = torch.device("cpu")
        self.use_distributed = False
        self.gradient_state = FakeGradientState(num_steps)
        self.optimizer_step_was_skipped = False
        self.events: list[str] = []

    @property
    def sync_gradients(self) -> bool:
        return self.gradient_state.sync_gradients

    def accumulate(self, model):
        del model

        @contextmanager
        def _cm():
            state = self.gradient_state
            state.step += 1
            # 真实 accelerate 在 end_of_dataloader 时会强制 sync（step 归零）。
            state._set_sync_gradients(state.step % state.num_steps == 0)
            yield

        return _cm()

    def no_sync(self, model):
        del model
        return nullcontext()

    def unwrap_model(self, model):
        return model

    def backward(self, loss):
        loss.backward()
        self.events.append("backward")

    def clip_grad_norm_(self, params, max_norm):
        self.events.append("clip")
        return torch.nn.utils.clip_grad_norm_(list(params), max_norm)

    def reduce(self, tensor, reduction="mean"):  # noqa: ARG002
        return tensor

    def skip_first_batches(self, loader, count):  # pragma: no cover - 未使用
        raise AssertionError("initial_step 必须为 0")


class FakeOptimizer:
    def __init__(self, parameter) -> None:
        self.param_groups = [{"params": [parameter]}]
        self.steps = 0  # 真正应用的更新次数（跳步不计）
        self.zero_calls = 0

    def step(self):
        self.steps += 1

    def zero_grad(self, set_to_none=True):  # noqa: ARG002
        self.zero_calls += 1


class FakeScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self):
        self.steps += 1


class Harness:
    """最小 loop 夹具。

    ``loss_plan`` 逐次给出 ``process_batch`` 返回的 loss 数值（精确值，同时保留计算图）；
    ``skip_plan`` 逐次给出边界处 scaler 是否跳步（模拟 ``optimizer_step_was_skipped``）。
    ``on_before_replay`` 在窗口重放开始前被调用，用于抓「失败尝试之后、重放之前」的快照。
    """

    def __init__(
        self,
        monkeypatch,
        *,
        num_steps: int = 1,
        max_train_steps: int = 8,
        loss_plan: list[float] | None = None,
        skip_plan: list[bool] | None = None,
        batches: int = 2,
        window_discard_budget: int | None = None,
        adaptive_skip_budget: int | None = None,
        adaptive_trial: bool = False,
        on_before_replay=None,
        on_after_commit=None,
        on_before_forward=None,
    ) -> None:
        self.model = torch.nn.Linear(2, 2)
        self.accelerator = FakeAccelerator(num_steps=num_steps)
        self.optimizer = FakeOptimizer(self.model.weight)
        self.scheduler = FakeScheduler()
        self.saves: list[int] = []
        self.loss_plan = list(loss_plan or [])
        self.skip_plan = list(skip_plan or [])
        self.forward_calls = 0
        self.last_loss_values: list[float] = []
        self.replay_calls = 0
        self.commits = 0
        self.on_before_replay = on_before_replay
        self.on_after_commit = on_after_commit
        self.on_before_forward = on_before_forward

        def forward(*_args, **_kwargs):
            if self.on_before_forward is not None:
                self.on_before_forward(self)
            value = self.loss_plan.pop(0) if self.loss_plan else 1.0
            base = self.model(torch.ones(1, 2)).square().mean()
            # 精确给定 loss 数值，同时保留真实计算图（backward 仍产生梯度）。
            loss = base * 0.0 + float(value)
            self.forward_calls += 1
            self.last_loss_values.append(float(loss.detach()))
            return loss

        trainer = SimpleNamespace(
            memory_probe=None,
            peak_probe=None,
            _cudagraph_mark_step=False,
            _state=SimpleNamespace(personalization_observer={}),
            on_step_start=lambda *_a, **_k: None,
            process_batch=forward,
            run_after_backward=lambda *_a, **_k: None,
        )
        self.state = SimpleNamespace(
            args=SimpleNamespace(
                max_grad_norm=0.0,
                log_every_n_steps=1,
                max_train_steps=max_train_steps,
                stage_schedule=None,
                stage_schedule_enabled=False,
                _stage_num_update_steps_per_epoch=0,
                # 裁定 B：普通训练读 --max_nonfinite_window_discards（默认 3），
                # adaptive 试验读 --adaptive_max_nonfinite_skips（默认 0）。
                **(
                    {"max_nonfinite_window_discards": window_discard_budget}
                    if window_discard_budget is not None
                    else {}
                ),
                **(
                    {"adaptive_max_nonfinite_skips": adaptive_skip_budget}
                    if adaptive_skip_budget is not None
                    else {}
                ),
                **(
                    {"_adaptive_training_worker": {"directory": "fixture"}}
                    if adaptive_trial
                    else {}
                ),
            ),
            accelerator=self.accelerator,
            network=self.model,
            training_model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
            text_encoder=None,
            unet=None,
            train_ctx=object(),
            on_step_start_for_network=lambda *_a: None,
            profile_started=False,
            profile_range=None,
            global_step=0,
            initial_step=0,
            is_tracking=False,
            stage_index=-1,
            current_step=SimpleNamespace(value=0),
            train_dataloader=[object() for _ in range(batches)],
            progress_bar=SimpleNamespace(update=lambda n: None),
            saver=SimpleNamespace(
                maybe_save_step=lambda *_a, **_k: self.saves.append(self.state.global_step)
            ),
            optimizer_train_fn=lambda: None,
            nonfinite_grad_skips=0,
        )
        self.trainer = trainer

        original_step = self.optimizer.step

        def step_with_skip_plan(*args, **kwargs):
            if not self.accelerator.sync_gradients:
                return  # 真实 AcceleratedOptimizer 在非边界是 no-op
            skipped = self.skip_plan.pop(0) if self.skip_plan else False
            # 真实 AcceleratedOptimizer.step: scaler 判定 found_inf 时不调内层 step，
            # 并把 step_was_skipped 置真。这里按同一顺序模拟。
            self.accelerator.optimizer_step_was_skipped = skipped
            if skipped:
                return
            original_step(*args, **kwargs)

        monkeypatch.setattr(self.optimizer, "step", step_with_skip_plan)

        for name in (
            "_maybe_apply_stage_schedule",
            "_record_recent_step_seconds",
            "_sample_at_step",
            "_log_step",
            "_maybe_run_step_validation",
            "_maybe_auto_block_swap",
        ):
            monkeypatch.setattr(loop, name, lambda *_a, **_k: None)
        monkeypatch.setattr(loop, "_maybe_scale_norm", lambda _state: (None, None, None, {}))
        monkeypatch.setattr(loop, "precision_observe", lambda *_a, **_k: None)
        monkeypatch.setattr(loop, "adaptive_notify", lambda *_a, **_k: None)
        monkeypatch.setattr(loop, "synchronize_optimizer_gradients", lambda *_a: None)

        real_commit = loop._commit_step_boundary

        def commit_with_hook(*args, **kwargs):
            real_commit(*args, **kwargs)
            self.commits += 1
            if self.on_after_commit is not None:
                self.on_after_commit(self)

        monkeypatch.setattr(loop, "_commit_step_boundary", commit_with_hook)

        real_replay = loop._replay_window

        def replay_with_hook(*args, **kwargs):
            self.replay_calls += 1
            if self.on_before_replay is not None:
                self.on_before_replay(self)
            return real_replay(*args, **kwargs)

        monkeypatch.setattr(loop, "_replay_window", replay_with_hook)

    # ------------------------------------------------------------------ helpers
    def run(self) -> None:
        loop._run_epoch_steps(self.trainer, self.state, 0)

    def readings(self) -> dict:
        """E3 规格的 8 个观测点读数（与 _WINDOW_ACCOUNTING_POINTS 同名）。"""
        state = self.state
        return {
            "optimizer": self.optimizer.steps,
            "scheduler": self.scheduler.steps,
            "global_step": state.global_step,
            "ema": getattr(state, "ema_observation", None),
            "ckpt": list(self.saves),
            "total_loss": float(getattr(state, "window_total_loss", 0.0) or 0.0),
            "num_steps": int(getattr(state, "window_num_steps", 0) or 0),
            "last_loss": getattr(state, "window_last_loss", None),
        }


# ---------------------------------------------------------------------------
# 真机级夹具（真实 accelerate Accelerator + 真 DataLoaderShard，CPU 即可）
# ---------------------------------------------------------------------------
class RealAcceleratorHarness:
    """真实 ``Accelerator`` 的最小 loop 夹具（blocker B1 回归用）。

    ``sync_landing`` 记录每次 ``optimizer.step`` 调用时的 (是否 sync, 权重是否变化,
    迄今 ``process_batch`` 调用次数)，用来验证重放后边界落点没有错位。
    """

    def __init__(
        self,
        monkeypatch,
        *,
        num_steps: int = 2,
        total_batches: int = 2,
        nan_calls: set[int] | None = None,
    ) -> None:
        from accelerate import Accelerator
        from accelerate.state import AcceleratorState, GradientState

        self._state_snapshot = (
            dict(AcceleratorState._shared_state),
            dict(GradientState._shared_state),
        )
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._shared_state.clear()

        self.accelerator = Accelerator(
            gradient_accumulation_steps=num_steps, cpu=True, mixed_precision="no"
        )
        self.model = torch.nn.Linear(2, 2)
        self.raw_optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.raw_optimizer)
        self.nan_calls = set(nan_calls or ())
        self.forward_calls = 0
        self.sync_landing: list[tuple[bool, bool, int]] = []
        self.applied = 0

        real_step = self.optimizer.step

        def counting_step(*args, **kwargs):
            before = [p.detach().clone() for p in self.model.parameters()]
            in_sync = bool(self.accelerator.sync_gradients)
            result = real_step(*args, **kwargs)
            moved = any(
                not torch.equal(b, p.detach()) for b, p in zip(before, self.model.parameters())
            )
            if in_sync and moved:
                self.applied += 1
            self.sync_landing.append((in_sync, moved, self.forward_calls))
            return result

        monkeypatch.setattr(self.optimizer, "step", counting_step)

        def process_batch(_ctx, batch, is_train=True):  # noqa: ARG001
            self.forward_calls += 1
            loss = self.model(batch).square().mean()
            if self.forward_calls in self.nan_calls:
                return loss * float("nan")
            return loss

        self.trainer = SimpleNamespace(
            memory_probe=None,
            peak_probe=None,
            _cudagraph_mark_step=False,
            _state=SimpleNamespace(personalization_observer={}),
            on_step_start=lambda *_a, **_k: None,
            process_batch=process_batch,
            run_after_backward=lambda *_a, **_k: None,
        )
        self.data = [torch.ones(1, 2) + index for index in range(total_batches)]
        self.loader = self.accelerator.prepare(
            torch.utils.data.DataLoader(self.data, batch_size=1)
        )
        self.state = SimpleNamespace(
            args=SimpleNamespace(
                max_grad_norm=0.0,
                log_every_n_steps=1,
                max_train_steps=total_batches,
                stage_schedule=None,
                stage_schedule_enabled=False,
                _stage_num_update_steps_per_epoch=0,
                max_nonfinite_window_discards=3,
            ),
            accelerator=self.accelerator,
            network=self.model,
            training_model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=SimpleNamespace(step=lambda: None),
            text_encoder=None,
            unet=None,
            train_ctx=object(),
            on_step_start_for_network=lambda *_a: None,
            profile_started=False,
            profile_range=None,
            global_step=0,
            initial_step=0,
            is_tracking=False,
            stage_index=-1,
            current_step=SimpleNamespace(value=0),
            train_dataloader=self.loader,
            progress_bar=SimpleNamespace(update=lambda _n: None),
            saver=SimpleNamespace(maybe_save_step=lambda *_a, **_k: None),
            optimizer_train_fn=lambda: None,
            nonfinite_grad_skips=0,
        )

        for name in (
            "_maybe_apply_stage_schedule",
            "_record_recent_step_seconds",
            "_sample_at_step",
            "_log_step",
            "_maybe_run_step_validation",
            "_maybe_auto_block_swap",
        ):
            monkeypatch.setattr(loop, name, lambda *_a, **_k: None)
        monkeypatch.setattr(loop, "_maybe_scale_norm", lambda _state: (None, None, None, {}))
        monkeypatch.setattr(loop, "precision_observe", lambda *_a, **_k: None)
        monkeypatch.setattr(loop, "adaptive_notify", lambda *_a, **_k: None)
        monkeypatch.setattr(loop, "synchronize_optimizer_gradients", lambda *_a: None)

    def weights(self) -> list[torch.Tensor]:
        return [p.detach().clone() for p in self.model.parameters()]

    def close(self) -> None:
        from accelerate.state import AcceleratorState, GradientState

        AcceleratorState._shared_state.clear()
        AcceleratorState._shared_state.update(self._state_snapshot[0])
        GradientState._shared_state.clear()
        GradientState._shared_state.update(self._state_snapshot[1])
