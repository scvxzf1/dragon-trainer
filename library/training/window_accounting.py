"""E3 accumulation-window accounting and replay policy.

User ruling (2026-11-06, ``codex/fp16-nan-guards``): when anything non-finite
happens inside an accumulation window — a non-finite micro-batch loss, or a
non-finite gradient that makes the ``GradScaler`` skip the update / the D2 guard
drop it — the **whole window is invalid**. The window is replayed with the same
data at most :data:`WINDOW_MAX_REPLAYS` time; when the replay fails too, the
window is discarded (counted in ``window_discards``) and training continues with
the next window instead of aborting.

Two budgets, deliberately decoupled (user ruling B, 2026-11-06):

* plain training uses ``--max_nonfinite_window_discards`` (**default 3**): the
  "replay once → count the discard → recover on the next window" fallback is
  reachable by default, while three consecutive dead windows still trip a fast
  fuse instead of hiding a real divergence;
* an adaptive/supervised trial uses ``--adaptive_max_nonfinite_skips``
  (**default 0**, strict): a dead window fails the trial with its own
  ``nonfinite_skip`` classification and the supervisor retries it.

Why this lives in its own module: ``library/training/loop.py`` is a hot file
(see ``AGENTS.md`` "反上帝代码守则"), so the ledger / failure predicate / policy
live here and ``loop.py`` only wires them into the epoch loop.

Ledger semantics
----------------
``window_attempts`` / ``window_replays`` / ``window_discards`` are cumulative run
counters. ``window_total_loss`` / ``window_num_steps`` / ``window_last_loss`` are
**window-level**: they describe the last *committed* window only (sum of its
micro-batch losses, how many micro-batches contributed, and the last one). A
discarded window leaves them untouched, which is exactly what the E3 spec in
``tests/test_fp16_nan_static_equivalence.py`` asserts ("丢弃整窗后账本原样等于窗口
起点快照").

Known limitation (documented, not fixed here): this repo has no model EMA, so
``state.ema_observation`` is an explicit placeholder slot written on the commit
path only — it proves a discarded window touches nothing, it is not an EMA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

#: User ruling: one replay per window, then degrade to "discard and continue".
WINDOW_MAX_REPLAYS = 1

#: Consecutive discarded windows plain training tolerates by default (ruling B).
DEFAULT_WINDOW_DISCARD_BUDGET = 3

#: Failure reasons surfaced in warnings / telemetry.
REASON_NONFINITE_LOSS = "nonfinite_loss"
REASON_UPDATE_SKIPPED = "update_skipped"
#: A peer rank asked to discard this window while this rank judged it clean.
#: The window-seam agreement makes every rank take the stricter outcome, so the
#: discard is attributed to the peer decision instead of inventing a local one.
REASON_PEER_WINDOW_DECISION = "peer_window_decision"


def window_size(accelerator: Any) -> int:
    """Micro-batches per accumulation window (``accelerate`` num_steps)."""
    gradient_state = getattr(accelerator, "gradient_state", None)
    num_steps = int(getattr(gradient_state, "num_steps", 1) or 1)
    return max(1, num_steps)


def loss_value(loss: Any) -> float:
    """Scalar float of a loss tensor; non-scalar / non-tensor input becomes NaN."""
    try:
        value = loss.detach() if hasattr(loss, "detach") else loss
        value = value.item() if hasattr(value, "item") else value
        return float(value)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return float("nan")


def losses_are_finite(losses: Iterable[float]) -> bool:
    return all(math.isfinite(value) for value in losses)


@dataclass
class WindowBuffer:
    """Batches and losses of the window currently being accumulated."""

    batches: list[Any] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    attempts: int = 0
    replays: int = 0

    def add(self, batch: Any, loss: Any) -> float:
        value = loss_value(loss)
        self.batches.append(batch)
        self.losses.append(value)
        return value

    def record_attempt(self, losses: Optional[list[float]] = None) -> None:
        """Count this attempt; a replay also replaces the recorded loss trace."""
        self.attempts += 1
        if losses is None:
            return
        self.replays += 1
        self.losses = list(losses)

    def failure_reason(self, *, update_skipped: bool) -> Optional[str]:
        """Why this window may not be committed, or ``None`` when it is clean."""
        if not losses_are_finite(self.losses):
            return REASON_NONFINITE_LOSS
        if update_skipped:
            return REASON_UPDATE_SKIPPED
        return None

    def reset(self) -> None:
        self.batches = []
        self.losses = []
        self.attempts = 0
        self.replays = 0


_COUNTER_FIELDS = (
    "window_attempts",
    "window_replays",
    "window_discards",
    "window_consecutive_discards",
    "window_total_loss",
    "window_num_steps",
)


def ensure_counters(state: Any) -> None:
    """Give lightweight / hand-built states the E3 fields ``LoopState`` declares."""
    defaults = {"window_total_loss": 0.0, "window_num_steps": 0, "window_last_loss": None,
                "ema_observation": None}
    for name in _COUNTER_FIELDS:
        if not hasattr(state, name):
            setattr(state, name, defaults.get(name, 0))
    for name, value in defaults.items():
        if not hasattr(state, name):
            setattr(state, name, value)
    # Resume-cursor counters (contract §8 P1-2). They are real ``LoopState``
    # fields; hand-built test states get the same "nothing consumed yet" default
    # here instead of forcing every harness to know about them.
    for name in ("batches_consumed", "batches_per_epoch", "resume_skipped_batches"):
        if not hasattr(state, name):
            setattr(state, name, 0)


def _counter(state: Any, name: str) -> int:
    return int(getattr(state, name, 0) or 0)


def commit_window(state: Any, buffer: WindowBuffer) -> dict[str, Any]:
    """Publish the committed window into the counters/ledger. Returns the snapshot."""
    snapshot = {
        "attempts": max(1, buffer.attempts),
        "replays": buffer.replays,
        "total_loss": float(sum(buffer.losses)),
        "num_steps": len(buffer.losses),
        "last_loss": buffer.losses[-1] if buffer.losses else None,
    }
    setattr(state, "window_attempts", _counter(state, "window_attempts") + snapshot["attempts"])
    setattr(state, "window_replays", _counter(state, "window_replays") + snapshot["replays"])
    setattr(state, "window_total_loss", snapshot["total_loss"])
    setattr(state, "window_num_steps", snapshot["num_steps"])
    setattr(state, "window_last_loss", snapshot["last_loss"])
    setattr(state, "ema_observation", ema_observation(state, buffer.losses))
    # A committed (effective) window resets the discard run, mirroring the D3
    # "consecutive" budget semantics.
    setattr(state, "window_consecutive_discards", 0)
    return snapshot


def discard_window(state: Any, buffer: WindowBuffer, reason: str) -> dict[str, Any]:
    """Drop the whole window: no counter, no ledger, no global_step moves."""
    setattr(state, "window_attempts", _counter(state, "window_attempts") + max(1, buffer.attempts))
    setattr(state, "window_replays", _counter(state, "window_replays") + buffer.replays)
    setattr(state, "window_discards", _counter(state, "window_discards") + 1)
    consecutive = _counter(state, "window_consecutive_discards") + 1
    setattr(state, "window_consecutive_discards", consecutive)
    return {
        "reason": reason,
        "consecutive_discards": consecutive,
        "attempts": max(1, buffer.attempts),
        "replays": buffer.replays,
        "discards": _counter(state, "window_discards"),
        # The ledger deliberately keeps the previous committed window's values.
        "total_loss": getattr(state, "window_total_loss", 0.0),
        "num_steps": getattr(state, "window_num_steps", 0),
        "last_loss": getattr(state, "window_last_loss", None),
    }


def adaptive_trial(args: Any) -> bool:
    """Whether this process is an adaptive/supervised trial.

    Reuses the repository's existing marker pair (see
    ``adaptive_runtime/resume_runtime.py`` ``configuration_errors`` and
    ``adaptive_runtime/training_runtime.py`` ``validate_entry``): the supervisor
    sets ``--adaptive_oom_retry`` and/or the private ``_adaptive_training_worker``
    dict. No new flag is introduced here.
    """
    return bool(
        getattr(args, "adaptive_oom_retry", False)
        or getattr(args, "_adaptive_training_worker", None)
    )


def discard_budget(args: Any) -> int:
    """How many *consecutive* window discards are tolerated (user ruling B).

    * adaptive trial → ``--adaptive_max_nonfinite_skips`` (default 0, strict);
    * plain training → ``--max_nonfinite_window_discards`` (default 3).

    Either way the budget is enforced at the window seam: a discarded window does
    not advance the adaptive runtime counters (it is not a boundary), so the
    runtime's own consecutive-skip counter cannot see it. "Consecutive" resets on
    a committed (effective) window.
    """
    if adaptive_trial(args):
        return max(0, int(getattr(args, "adaptive_max_nonfinite_skips", 0) or 0))
    return max(
        0,
        int(
            getattr(args, "max_nonfinite_window_discards", DEFAULT_WINDOW_DISCARD_BUDGET)
            or 0
        ),
    )


def record_loss(recorder: Any, *, epoch: int, step: int, loss: float) -> bool:
    """Add a training-loss sample unless it is non-finite.

    E3: a non-finite micro-batch loss belongs to an invalidated window (replayed or
    discarded), so it must never reach the loss ledger — otherwise one NaN poisons
    the epoch average for the rest of the epoch (observed in the 2026-11-07 real
    training run). Returns whether the sample was recorded.
    """
    if not losses_are_finite([loss]):
        return False
    recorder.add(epoch=epoch, step=step, loss=loss)
    return True


def ema_observation(state: Any, losses: Iterable[float]) -> dict[str, Any]:
    """Commit-path snapshot of the (absent) EMA slot — see module docstring."""
    values = list(losses)
    return {
        "step": int(getattr(state, "global_step", 0) or 0) + 1,
        "num_steps": len(values),
        "mean_loss": (sum(values) / len(values)) if values else None,
        "ema": "absent-in-this-repo",
    }


def ledger_snapshot(state: Any) -> dict[str, Any]:
    """Read-only view used by tests / telemetry."""
    return {
        "window_attempts": _counter(state, "window_attempts"),
        "window_replays": _counter(state, "window_replays"),
        "window_discards": _counter(state, "window_discards"),
        "window_consecutive_discards": _counter(state, "window_consecutive_discards"),
        "window_total_loss": float(getattr(state, "window_total_loss", 0.0) or 0.0),
        "window_num_steps": _counter(state, "window_num_steps"),
        "window_last_loss": getattr(state, "window_last_loss", None),
        "ema_observation": getattr(state, "ema_observation", None),
    }
