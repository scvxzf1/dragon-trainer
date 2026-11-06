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
    assert state.optimizer.zero_grad_calls == 1
    assert state.nonfinite_grad_skips == 1


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
