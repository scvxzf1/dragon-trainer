"""Window-boundary coordination: world-wide decisions + per-boundary attempt accounting.

Two responsibilities that both belong to the optimizer **boundary** live here (and only
here), so ``library/training/loop.py`` stays pure wiring and
``library/training/window_accounting.py`` keeps its single-process predicate/ledger role:

1. **World agreement** (R3) — the E3 window predicate only reads *this* rank's losses and
   ``update_skipped``. With more than one rank that is two different decisions: the rank
   that saw a bad micro-batch replays (one extra gradient all-reduce round) while the
   others commit, so the collective order no longer matches; and because the verdict is
   taken *after* the microbatch body returned, a clean rank has already called
   ``optimizer.step()`` for a window the world later declares bad. Both must be agreed
   **before** the branch is taken.
2. **Per-boundary attempt accounting** (R4/R5, three-layer attribution) — count every
   optimizer boundary attempt (replays and finally discarded windows included) and record
   *why* it applied or dropped, split into the three layers below.

Attribution layers (contract C4; the layers must never overwrite each other)
--------------------------------------------------------------------------
``L1 originating cause``   what **this** rank observed for this attempt:
                           ``nonfinite_loss`` | ``guard_drop_nonfinite_grad`` |
                           ``scaler_overflow`` (only with a live scaler that reported the
                           skip) — a rank that had no local reason has **no** L1 cause;
``L2 propagated drop``     this rank was clean but followed a world decision to drop;
``L3 world decision``      the branch finally taken (``apply`` / ``drop`` for a boundary,
                           ``commit`` / ``replay`` / ``discard`` for a window).

Wire format is not severity: the cross-rank reduction encodes ``0=clean /
1=nonfinite_loss / 2=update_skipped`` and reduces with ``MAX`` only so every rank can pick
the same branch. The encoding is **not** a severity ranking and it is **never** used to
infer an L1 cause — ``discard_causes`` and the L1 histogram are built from local evidence
only, and with no live scaler the ``scaler_overflow`` bucket is structurally 0.

Observability only: nothing here changes a training decision. The discard budget stays
``window_consecutive_discards`` (plain 3 / adaptive 0 / one replay). Boundaries that did
**not** come from the window route (``_run_step`` direct callers such as the AUTO swap
probe) are counted in their own bucket (``direct_route_boundaries``) and never enter
``step_attempts`` / ``windows.*``, so the window identity stays meaningful.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from library.training import window_accounting

logger = logging.getLogger(__name__)

#: Route label for boundaries produced by the E3 epoch loop (window transaction).
ROUTE_WINDOW = "window"
#: Route label for ``_run_step`` direct callers (AUTO block-swap probe, tests).
ROUTE_DIRECT = "direct"

#: Wire-format codes for the MAX reduction. **Not** a severity ranking — see module doc.
_CODE_CLEAN = 0.0
_CODE_NONFINITE = 1.0
_CODE_UPDATE_SKIPPED = 2.0
_CODE_BY_REASON = {
    None: _CODE_CLEAN,
    window_accounting.REASON_NONFINITE_LOSS: _CODE_NONFINITE,
    window_accounting.REASON_UPDATE_SKIPPED: _CODE_UPDATE_SKIPPED,
}

# --- L1 originating causes -------------------------------------------------
CAUSE_APPLIED = "applied"
CAUSE_NONFINITE_LOSS = "nonfinite_loss"
CAUSE_GRAD_GUARD = "guard_drop_nonfinite_grad"
CAUSE_WINDOW_PREFIX = "window_prefix_nonfinite"
CAUSE_SCALER_OVERFLOW = "scaler_overflow"
#: Bucket for a boundary where this rank had no local reason (clean or L2-propagated).
CAUSE_NO_LOCAL_REASON = "no_local_reason"

_COUNTER_FIELDS = (
    "step_attempts",
    "step_updates_applied",
    "step_updates_skipped_by_scaler",
    "step_updates_dropped",
    "step_drops_with_scaler",
    "step_drops_without_scaler",
    "propagated_drops",
    "direct_route_boundaries",
)


# ---------------------------------------------------------------------------
# Section 1 — world agreement (R3)
# ---------------------------------------------------------------------------
def _collectives_enabled(state: Any) -> bool:
    """Whether a distributed world exists and this rank must join its collectives."""
    accelerator = state.accelerator
    if int(getattr(accelerator, "num_processes", 1) or 1) <= 1:
        return False
    import torch.distributed as dist

    return bool(dist.is_available() and dist.is_initialized())


def _reduce_max(state: Any, value: float) -> float:
    import torch
    import torch.distributed as dist

    flag = torch.tensor([value], device=state.accelerator.device, dtype=torch.float32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return float(flag.item())


def agree_failure_reason(state: Any, reason: Optional[str]) -> Optional[str]:
    """Reduce the window verdict so every rank takes the same commit/replay/discard branch.

    Returns ``reason`` unchanged for single-process runs and when the ranks agree.
    """
    if not _collectives_enabled(state):
        return reason
    worst = _reduce_max(state, _CODE_BY_REASON.get(reason, _CODE_NONFINITE))
    if worst <= _CODE_CLEAN:
        return None
    if worst >= _CODE_UPDATE_SKIPPED:
        return window_accounting.REASON_UPDATE_SKIPPED
    return window_accounting.REASON_NONFINITE_LOSS


def window_reason(state: Any, buffer: Any) -> Optional[str]:
    """The E3 verdict for the window being closed, agreed across ranks (L3)."""
    local = buffer.failure_reason(
        update_skipped=bool(getattr(state, "update_skipped", False))
    )
    return agree_failure_reason(state, local)


def local_window_reason(state: Any, buffer: Any) -> Optional[str]:
    """This rank's own E3 verdict, before any agreement (L1 input for the cause histogram)."""
    return buffer.failure_reason(
        update_skipped=bool(getattr(state, "update_skipped", False))
    )


def agree_drop_update(state: Any, drop_update: bool) -> bool:
    """Agree the "drop this boundary's update" gate *before* ``optimizer.step()``."""
    if not _collectives_enabled(state):
        return drop_update
    return bool(_reduce_max(state, 1.0 if drop_update else 0.0) > 0.0)


# ---------------------------------------------------------------------------
# Section 2 — per-boundary attempt accounting (R4/R5, C4 layers, C8 scopes)
# ---------------------------------------------------------------------------
def ensure(state: Any) -> None:
    """Create the counters on lightweight / hand-built states."""
    window_accounting.ensure_counters(state)
    for name in _COUNTER_FIELDS:
        if not hasattr(state, name):
            setattr(state, name, 0)
    for name in ("step_drop_causes", "step_world_decisions"):
        if getattr(state, name, None) is None:
            setattr(state, name, {})


def _bump(state: Any, name: str, delta: int = 1) -> int:
    value = int(getattr(state, name, 0) or 0) + delta
    setattr(state, name, value)
    return value


def _bump_cause(state: Any, field: str, key: str) -> None:
    causes = dict(getattr(state, field, None) or {})
    causes[key] = int(causes.get(key, 0) or 0) + 1
    setattr(state, field, causes)


def local_cause(
    state: Any,
    *,
    skip_update: bool,
    force_skip: bool,
    loss_is_finite: bool,
    step_attempted: bool,
    scaler_skipped: Optional[bool] = None,
) -> Optional[str]:
    """L1 originating cause this rank can prove for the boundary, or ``None``."""
    if skip_update:
        return CAUSE_GRAD_GUARD
    if force_skip:
        return CAUSE_WINDOW_PREFIX
    if not loss_is_finite:
        return CAUSE_NONFINITE_LOSS
    if not step_attempted:
        # ``optimizer.step()`` never ran, so this rank has no originating cause of its
        # own: either another rank's verdict dropped it (L2) or nothing happened.
        return None
    if scaler_skipped is None:
        scaler_skipped = bool(
            getattr(getattr(state, "accelerator", None), "optimizer_step_was_skipped", False)
        )
    # A scaler skip is only attributable when a scaler exists and step() really ran.
    if scaler_skipped and getattr(getattr(state, "accelerator", None), "scaler", None) is not None:
        return CAUSE_SCALER_OVERFLOW
    return None


def classify(
    state: Any,
    *,
    skip_update: bool,
    force_skip: bool,
    loss_is_finite: bool,
    step_attempted: bool = True,
    scaler_skipped: Optional[bool] = None,
) -> tuple[Optional[str], bool]:
    """Return ``(L1 cause or None, dropped_by_world_propagation)`` for this boundary."""
    dropped = bool(getattr(state, "update_skipped", False))
    cause = local_cause(
        state,
        skip_update=skip_update,
        force_skip=force_skip,
        loss_is_finite=loss_is_finite,
        step_attempted=step_attempted,
        scaler_skipped=scaler_skipped,
    )
    if not dropped:
        return (cause, False)
    if cause is None:
        # Clean rank that dropped: the update was propagated-dropped by the world
        # agreement (L2). Never attributed to nonfinite_loss/scaler_overflow.
        return (None, True)
    return (cause, False)


def note_attempt(
    state: Any,
    *,
    route: str = ROUTE_WINDOW,
    skip_update: bool,
    force_skip: bool,
    loss_is_finite: bool,
    step_attempted: bool = True,
    scaler_skipped: Optional[bool] = None,
) -> Optional[str]:
    """Record one optimizer-boundary attempt. Returns the L1 cause (``None`` if none)."""
    ensure(state)
    if route != ROUTE_WINDOW:
        # C8.2: direct-route boundaries (AUTO probe, other direct callers) must not enter
        # the window counters, otherwise the window identity would be meaningless.
        _bump(state, "direct_route_boundaries")
        return None
    cause, propagated = classify(
        state,
        skip_update=skip_update,
        force_skip=force_skip,
        loss_is_finite=loss_is_finite,
        step_attempted=step_attempted,
        scaler_skipped=scaler_skipped,
    )
    applied = not bool(getattr(state, "update_skipped", False))
    _bump(state, "step_attempts")
    if applied:
        _bump(state, "step_updates_applied")
        _bump_cause(state, "step_drop_causes", CAUSE_APPLIED)
        _bump_cause(state, "step_world_decisions", "apply")
        return CAUSE_APPLIED
    _bump(state, "step_updates_dropped")
    if cause == CAUSE_SCALER_OVERFLOW:
        _bump(state, "step_updates_skipped_by_scaler")
    if getattr(getattr(state, "accelerator", None), "scaler", None) is not None:
        _bump(state, "step_drops_with_scaler")
    else:
        _bump(state, "step_drops_without_scaler")
    if propagated:
        _bump(state, "propagated_drops")
    _bump_cause(state, "step_drop_causes", cause or CAUSE_NO_LOCAL_REASON)
    _bump_cause(state, "step_world_decisions", "drop")
    return cause


def note_window_decision(state: Any, decision: str, *, propagated: bool = False) -> None:
    """Record the L3 world decision for one closed window (commit/replay/discard)."""
    ensure(state)
    windows = dict(getattr(state, "window_world_decisions", None) or {})
    windows[decision] = int(windows.get(decision, 0) or 0) + 1
    setattr(state, "window_world_decisions", windows)
    if propagated and decision == "discard":
        _bump(state, "propagated_window_discards")


def scaler_present(state: Any) -> bool:
    """Whether this process has a live ``GradScaler`` (attribution bookkeeping only)."""
    return getattr(getattr(state, "accelerator", None), "scaler", None) is not None


def snapshot(state: Any) -> dict[str, Any]:
    """Read-only view: L1 histogram + L2 counts + L3 decisions + the window ledger."""
    attempts = int(getattr(state, "step_attempts", 0) or 0)
    applied = int(getattr(state, "step_updates_applied", 0) or 0)
    window = window_accounting.ledger_snapshot(state)
    originating = dict(getattr(state, "step_drop_causes", None) or {})
    if not scaler_present(state):
        # C4.3-3: with no live scaler the bucket must be a hard 0, never a guess.
        originating.pop(CAUSE_SCALER_OVERFLOW, None)
    return {
        "step_attempts": attempts,
        "updates_applied": applied,
        # Effective = accepted by the transaction (no scaler skip, no guard, no
        # propagation). Never a bitwise parameter comparison (R4.1).
        "updates_effective": applied,
        "updates_skipped": max(0, attempts - applied),
        "updates_skipped_by_scaler": int(
            getattr(state, "step_updates_skipped_by_scaler", 0) or 0
        ),
        "updates_dropped": int(getattr(state, "step_updates_dropped", 0) or 0),
        "updates_dropped_with_scaler": int(getattr(state, "step_drops_with_scaler", 0) or 0),
        "updates_dropped_without_scaler": int(
            getattr(state, "step_drops_without_scaler", 0) or 0
        ),
        "direct_route_boundaries": int(getattr(state, "direct_route_boundaries", 0) or 0),
        # L1 histogram (originating causes), L2 count, L3 decisions.
        "originating_causes": originating,
        "propagated_drops": int(getattr(state, "propagated_drops", 0) or 0),
        "world_decisions": dict(getattr(state, "step_world_decisions", None) or {}),
        "window_world_decisions": dict(getattr(state, "window_world_decisions", None) or {}),
        "propagated_window_discards": int(
            getattr(state, "propagated_window_discards", 0) or 0
        ),
        "windows_committed": window["window_commits"],
        "windows_discarded": window["window_discards"],
        "window_replays": window["window_replays"],
        "window_attempts": window["window_attempts"],
        "window_consecutive_discards": window["window_consecutive_discards"],
        "window_discard_causes": window["window_discard_causes"],
        "window_discard_propagated": window["window_discard_propagated"],
    }


__all__ = [
    "CAUSE_APPLIED",
    "CAUSE_GRAD_GUARD",
    "CAUSE_NONFINITE_LOSS",
    "CAUSE_NO_LOCAL_REASON",
    "CAUSE_SCALER_OVERFLOW",
    "CAUSE_WINDOW_PREFIX",
    "ROUTE_DIRECT",
    "ROUTE_WINDOW",
    "agree_drop_update",
    "agree_failure_reason",
    "classify",
    "ensure",
    "local_cause",
    "local_window_reason",
    "note_attempt",
    "note_window_decision",
    "scaler_present",
    "snapshot",
    "window_reason",
]
