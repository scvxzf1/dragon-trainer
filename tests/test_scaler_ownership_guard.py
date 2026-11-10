"""Ownership-aware gradient guard: D1 (no false fatal) and D2 (no NaN into step).

The guard is judged at the optimizer boundary only, so these cases use small
fakes instead of a full loop harness: the decision logic is what changed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.training.finite_checks import check_trainable_grads_finite
from library.training.loop import _guard_nonfinite_grads
from library.training.scaler_ownership import (
    GradFiniteness,
    inspect_grad_ownership,
    inspect_trainable_grads,
    scaled_gradient_overflow_is_scaler_owned,
)


class _SpyOptimizer:
    def __init__(self) -> None:
        self.zero_grad_calls = 0

    def zero_grad(self, set_to_none: bool = True) -> None:  # noqa: ARG002
        self.zero_grad_calls += 1


def _accelerator(*, scaler=None, sync: bool = True, processes: int = 1, reduce_value=None):
    calls = {"reduce": 0}

    def _reduce(tensor, reduction="mean"):  # noqa: ARG001
        calls["reduce"] += 1
        if reduce_value is None:
            return tensor
        return torch.full_like(tensor, float(reduce_value))

    return SimpleNamespace(
        sync_gradients=sync,
        scaler=scaler,
        num_processes=processes,
        device=torch.device("cpu"),
        unwrap_model=lambda model: model,
        reduce=_reduce,
        reduce_calls=calls,
    )


def _state():
    return SimpleNamespace(nonfinite_grad_skips=0, optimizer=_SpyOptimizer())


def _args(*, debug: bool = False):
    return SimpleNamespace(debug_finite_checks=debug)


def _network(*, nonfinite: bool) -> torch.nn.Linear:
    network = torch.nn.Linear(2, 2)
    for parameter in network.parameters():
        parameter.grad = torch.ones_like(parameter)
        if nonfinite:
            parameter.grad.fill_(float("inf"))
    return network


def test_no_scaler_nonfinite_grads_skip_the_update():
    """D2：无 scaler 时非有限梯度必须丢弃更新，而不是写进权重。"""
    state = _state()
    guard = _guard_nonfinite_grads(
        _accelerator(scaler=None), state, _network(nonfinite=True), _args()
    )

    assert guard is not None and guard.must_skip_step is True
    assert state.nonfinite_grad_skips == 1
    # zero_grad 由 _run_step 在账目块里统一执行（与 scaler 跳步同位置），
    # guard 只做判定，避免两处都清梯度。
    assert state.optimizer.zero_grad_calls == 0


def test_scaler_owned_overflow_is_not_fatal_even_with_debug_checks():
    """D1：有 scaler 时同一读数归 scaler 所有，debug 打开也不得抛错或跳过。"""
    state = _state()
    guard = _guard_nonfinite_grads(
        _accelerator(scaler=object()), state, _network(nonfinite=True), _args(debug=True)
    )

    assert guard is not None and guard.scaler_owned is True
    assert guard.must_skip_step is False
    assert state.optimizer.zero_grad_calls == 0
    assert state.nonfinite_grad_skips == 0


def test_scaler_without_debug_cheap_short_circuit():
    """有 scaler 且未开 debug 时不做扫描：结论恒为「归 scaler」，扫描是纯开销。"""
    state = _state()
    guard = _guard_nonfinite_grads(
        _accelerator(scaler=object()), state, _network(nonfinite=True), _args()
    )

    assert guard is None
    assert state.nonfinite_grad_skips == 0


def test_bulk_scan_reports_only_non_finite_grads():
    """批量 isfinite 路径必须既找到坏梯度，也不把有限梯度误判。"""
    network = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    for index, parameter in enumerate(network.parameters()):
        parameter.grad = torch.zeros_like(parameter)
        if index == 2:
            parameter.grad[0, 0] = float("nan")
    offenders = inspect_trainable_grads(network)

    assert len(offenders) == 1
    assert offenders[0].startswith("1.weight:")


def test_finite_grads_are_left_alone():
    state = _state()
    guard = _guard_nonfinite_grads(
        _accelerator(scaler=None), state, _network(nonfinite=False), _args()
    )

    assert guard is not None and guard.bad is False
    assert state.optimizer.zero_grad_calls == 0


def test_mid_window_microstep_never_skips():
    """累积窗口内不判定：否则会在边界之前丢掉仍在累积的梯度。"""
    state = _state()
    guard = _guard_nonfinite_grads(
        _accelerator(scaler=None, sync=False),
        state,
        _network(nonfinite=True),
        _args(debug=True),
    )

    assert guard is None
    assert state.optimizer.zero_grad_calls == 0
    assert state.nonfinite_grad_skips == 0


def test_scan_is_skipped_between_boundaries_without_debug():
    """非边界且未开 debug 时不做扫描，避免每个 microstep 都付全参扫描成本。"""
    state = _state()
    guard = _guard_nonfinite_grads(
        _accelerator(scaler=None, sync=False), state, _network(nonfinite=False), _args()
    )

    assert guard is None


def test_reduce_agrees_skip_across_ranks():
    """任一 rank 看到 NaN 时全员跳过，避免 DDP 分支发散挂死。"""
    accelerator = _accelerator(scaler=None, processes=2, reduce_value=1.0)
    network = _network(nonfinite=False)  # 本 rank 干净，另一 rank 报坏
    finiteness = inspect_grad_ownership(accelerator, network)

    assert accelerator.reduce_calls["reduce"] == 1
    assert finiteness.bad is True and finiteness.scaler_owned is False


def test_single_process_skips_reduce():
    accelerator = _accelerator(scaler=None, processes=1)
    network = _network(nonfinite=True)
    inspect_grad_ownership(accelerator, network)

    assert accelerator.reduce_calls["reduce"] == 0


def test_check_trainable_grads_finite_owner_semantics():
    network = _network(nonfinite=True)
    with pytest.raises(FloatingPointError, match="non-finite trainable gradients"):
        check_trainable_grads_finite(network)

    reported = check_trainable_grads_finite(network, scaler_owned=True)
    assert reported and all("dtype=" in item for item in reported)


def test_scaled_overflow_ownership_truth_table():
    overflow = _network(nonfinite=True)
    finite = _network(nonfinite=False)
    grads_overflow = [parameter.grad for parameter in overflow.parameters()]
    grads_finite = [parameter.grad for parameter in finite.parameters()]

    assert scaled_gradient_overflow_is_scaler_owned(_accelerator(scaler=object()), grads_overflow)
    assert not scaled_gradient_overflow_is_scaler_owned(_accelerator(scaler=object()), grads_finite)
    assert not scaled_gradient_overflow_is_scaler_owned(_accelerator(scaler=None), grads_overflow)


def test_grad_finiteness_summary_reports_offenders():
    finiteness = GradFiniteness(offenders=("w: dtype=torch.float16 shape=(1, 1)",))
    assert "non-finite" not in finiteness.summary()
    assert finiteness.must_skip_step is True


def test_loop_skips_update_without_scaler_and_commits_nothing(monkeypatch):
    """loop 级集成：无 scaler 的 NaN 步**不提交**——权重、scheduler、账本全部不动。

    语义变更（2026-11-06，用户 E3 裁定 + captain 补充裁定 #2「未完成真实权重更新的窗口
    绝不推进进度副作用」）：本用例此前断言「跳步仍推进 scheduler / 记 precision skipped」，
    那是 D3 的旧语义。现在 ``_run_step`` 只在边界真的应用了更新时才执行提交路径
    （``_commit_step_boundary``），因此跳步既不推进 scheduler、也不产生
    attempted/effective/skipped 计数；窗口层的重放与丢窗见 ``tests/test_window_replay.py``。
    """
    from library.training import loop as training_loop

    events: list[str] = []

    class Network(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, 1))
            self.weight.grad = torch.full_like(self.weight, float("inf"))

        def get_trainable_params(self):
            return [self.weight]

    parameter = Network().weight

    class Accelerator:
        sync_gradients = True
        mixed_precision = "no"
        scaler = None
        num_processes = 1
        device = torch.device("cpu")

        @staticmethod
        def accumulate(model):
            del model
            from contextlib import nullcontext

            return nullcontext()

        @staticmethod
        def unwrap_model(model):
            return model

        @staticmethod
        def backward(loss):
            del loss
            events.append("backward")

        @staticmethod
        def clip_grad_norm_(params, max_norm):
            events.append("clip")

    class Optimizer:
        param_groups = [{"params": [parameter]}]

        @staticmethod
        def step():
            events.append("optimizer")

        @staticmethod
        def zero_grad(set_to_none):
            events.append("zero")

    class Scheduler:
        @staticmethod
        def step():
            events.append("scheduler")

    trainer = SimpleNamespace(
        memory_probe=None,
        peak_probe=None,
        _cudagraph_mark_step=False,
        _state=SimpleNamespace(personalization_observer={}),
        on_step_start=lambda *_args, **_kwargs: events.append("step_start"),
        process_batch=lambda *_args, **_kwargs: (
            events.append("forward") or torch.tensor(1.0, requires_grad=True)
        ),
        run_after_backward=lambda *_args, **_kwargs: events.append("after_hook"),
    )
    network = Network()
    state = SimpleNamespace(
        args=SimpleNamespace(max_grad_norm=1.0, log_every_n_steps=1, max_train_steps=1),
        accelerator=Accelerator(),
        network=network,
        training_model=network,
        optimizer=Optimizer(),
        lr_scheduler=Scheduler(),
        text_encoder=None,
        unet=None,
        train_ctx=object(),
        on_step_start_for_network=lambda *_args: None,
        profile_started=False,
        global_step=0,
        is_tracking=True,
        nonfinite_grad_skips=0,
    )

    observed: list[tuple] = []
    monkeypatch.setattr(
        training_loop,
        "precision_observe",
        lambda method, _state, **kwargs: observed.append((method, kwargs.get("after"), kwargs.get("skipped"))),
    )
    monkeypatch.setattr(training_loop, "debug_finite_enabled", lambda _args: False)
    monkeypatch.setattr(
        training_loop, "synchronize_optimizer_gradients", lambda *_args: events.append("sync")
    )

    returned = training_loop._run_step(trainer, state, batch=object())

    assert "optimizer" not in events, "无 scaler 且梯度非有限时不得执行 optimizer.step"
    assert "scheduler" not in events, "未完成真实更新的边界不得推进 LR scheduler"
    assert "zero" in events, "跳过后必须清理梯度"
    assert state.nonfinite_grad_skips == 1
    assert state.update_skipped is True, "loop 必须发布统一的每步跳过标志"
    assert observed == [], "未提交的边界不得产生 attempted/effective/skipped 计数"
    assert returned is not None and returned.requires_grad is False, "返回值必须已 detach"


def test_large_finite_fp16_grads_are_not_reported():
    """R1 负控制：大量但有限的 fp16 梯度不得被误判非有限。

    范数若在 fp16 内累加，4×40000 个 1.0 会得到 inf，把健康运行判成需要跳步。
    """
    network = torch.nn.Linear(40000, 4, bias=False).half()
    # 160k 个 200.0 的 L2 范数是 80000，超过 fp16 上限 65504：范数若以 fp16 结果返回
    # 就会变成 inf，而梯度本身全是有限值。
    network.weight.grad = torch.full_like(network.weight, 200.0)
    assert torch.isfinite(network.weight.grad).all(), "构造前提：元素全部有限"

    assert inspect_trainable_grads(network) == ()
    finiteness = inspect_grad_ownership(_accelerator(scaler=None), network)
    assert finiteness.bad is False and finiteness.must_skip_step is False
    assert check_trainable_grads_finite(network) == ()

    # 同一构造函数里注入真正的溢出仍需被抓到
    network.weight.grad[0, 0] = float("inf")
    assert inspect_trainable_grads(network)


def test_skipped_step_does_not_enter_step_time_samples():
    """F3：跳步只推进时间基线，不追加步时样本。

    ``block_swap_auto`` 的 maybe_promote 读 ``_recent_step_seconds``，溢出步的墙钟
    时间若混进去会稀释/尖峰化步时判断。
    """
    from library.training.loop import _recent_step_seconds, _record_recent_step_seconds

    state = SimpleNamespace(recent_step_rate_last=None, recent_step_seconds=[])

    _record_recent_step_seconds(state, 1, now=100.0)
    _record_recent_step_seconds(state, 2, now=101.0, record=False)  # 跳步：不采样
    assert list(state.recent_step_seconds) == []
    assert state.recent_step_rate_last == (101.0, 2), "跳步仍要推进基线"

    _record_recent_step_seconds(state, 3, now=103.0)  # 真实更新：只量这一步
    assert list(state.recent_step_seconds) == [2.0]
    assert _recent_step_seconds(state) == 2.0


def test_dropped_step_never_clips_so_scaler_stage_stays_ready(monkeypatch):
    """P0-3 回归：协商丢弃的步不得触发 clip，否则 GradScaler stage 卡在 UNSCALED。

    场景是「rank 本地完全干净、对端非有限」：本 rank 的 loss 与梯度都有限，所以旧的
    ``clip_grad_norm_`` 门（只判 rank 本地）会放行；``accelerator.clip_grad_norm_``
    内部先调 ``unscale_gradients()``（纯本地 ``scaler.unscale_()``，非集合通信），把
    stage 推到 UNSCALED 后才发现多进程协商决定 drop——而 drop 分支不调
    ``optimizer.step()``，于是 ``scaler.step()``/``update()`` 双双被跳过，stage 停在
    UNSCALED。下一个边界再 ``unscale_()`` 就抛
    ``RuntimeError: unscale_() has already been called``。

    修复把 clip 移到 ``agree_step`` 之后：drop 的步根本不 unscale，stage 保持 READY。
    """
    from library.training import loop as training_loop

    events: list[str] = []

    class Network(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, 1))
            # 本地梯度是有限的——旧的门只看本地，所以会放行 clip。
            self.weight.grad = torch.ones_like(self.weight)

        def get_trainable_params(self):
            return [self.weight]

    parameter = Network().weight

    class Accelerator:
        sync_gradients = True
        mixed_precision = "fp16"
        scaler = object()  # 非 None 即可：我们断言的是 clip 根本没被调用
        num_processes = 2
        device = torch.device("cpu")

        @staticmethod
        def accumulate(model):
            del model
            from contextlib import nullcontext

            return nullcontext()

        @staticmethod
        def unwrap_model(model):
            return model

        @staticmethod
        def backward(loss):
            del loss
            events.append("backward")

        @staticmethod
        def clip_grad_norm_(params, max_norm):
            del params, max_norm
            # 真实实现会先 unscale；这里只记录，是否被调用就是被测行为。
            events.append("clip")
            raise AssertionError("协商丢弃的步不得触发 clip")

    class Optimizer:
        def __init__(self) -> None:
            self.param_groups = [{"params": [parameter]}]

        @staticmethod
        def step():
            events.append("optimizer")

        @staticmethod
        def zero_grad(set_to_none):
            del set_to_none
            events.append("zero")

    class Scheduler:
        @staticmethod
        def step():
            events.append("scheduler")

    class _Agreement:
        def __init__(self) -> None:
            self.drop_update = True
            self.reasons: tuple[str, ...] = ("peer_gradient_nonfinite",)

    def _fake_agree_step(accelerator, **_kwargs):
        del accelerator
        events.append("agree")
        return _Agreement()

    trainer = SimpleNamespace(
        memory_probe=None,
        peak_probe=None,
        _cudagraph_mark_step=False,
        _state=SimpleNamespace(personalization_observer={}),
        on_step_start=lambda *_args, **_kwargs: events.append("step_start"),
        process_batch=lambda *_args, **_kwargs: (
            events.append("forward") or torch.tensor(1.0, requires_grad=True)
        ),
        run_after_backward=lambda *_args, **_kwargs: events.append("after_hook"),
    )

    def _new_state():
        network = Network()
        return SimpleNamespace(
            args=SimpleNamespace(max_grad_norm=1.0, log_every_n_steps=1, max_train_steps=2),
            accelerator=Accelerator(),
            network=network,
            training_model=network,
            optimizer=Optimizer(),
            lr_scheduler=Scheduler(),
            text_encoder=None,
            unet=None,
            train_ctx=object(),
            on_step_start_for_network=lambda *_args: None,
            profile_started=False,
            global_step=0,
            is_tracking=True,
            nonfinite_grad_skips=0,
        )

    monkeypatch.setattr(training_loop, "precision_observe", lambda *_a, **_k: None)
    monkeypatch.setattr(training_loop, "debug_finite_enabled", lambda _args: False)
    monkeypatch.setattr(
        training_loop, "synchronize_optimizer_gradients", lambda *_a: events.append("sync")
    )
    # 对端非有限 → 协商结果是「本 rank 也丢弃」
    monkeypatch.setattr(training_loop.multigpu, "agree_step", _fake_agree_step)
    monkeypatch.setattr(training_loop.multigpu, "negotiate_boundary_error", lambda *_a, **_k: None)

    state = _new_state()
    training_loop._run_step(trainer, state, batch=object())
    first = list(events)

    # 第二个边界：同样的丢弃。若第一步残留了 UNSCALED，真实的 scaler.unscale_()
    # 就会在这里抛 RuntimeError；修复后 clip 从未被调用，也就无从残留。
    events.clear()
    state.global_step = 1
    training_loop._run_step(trainer, state, batch=object())

    assert "clip" not in first, "协商丢弃的步不得调用 clip（否则 scaler stage 残留 UNSCALED）"
    assert "clip" not in events, "第二个丢弃步同样不得调用 clip"
    assert "optimizer" not in first and "optimizer" not in events, "丢弃的步不得更新参数"
    assert "agree" in first, "多进程路径必须经过 agree_step"
    assert state.update_skipped is True
