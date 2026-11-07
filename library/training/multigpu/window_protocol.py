"""Window health / skip / scaler agreement for the training loop.

The production loop decides "commit, replay or discard this accumulation window"
from rank-local state:

* ``loss_is_finite`` – this rank's micro-batch loss (``library/training/loop.py``);
* ``force_skip`` – this rank's window ledger;

When those differ, one rank can reach ``optimizer.step()`` (and therefore
``scaler.step()``/``scaler.update()`` inside ``AcceleratedOptimizer``) while the
other only zeroes its gradients. Scale and growth tracker then advance on one
rank only, and the commit/replay/discard decision diverges together with the
collective sequence.

This module turns those rank-local values into one agreed decision. It does not
decide *what* to do – the loop keeps owning that – it only guarantees that every
rank gets the same answer and that the scaler state stays symmetric.

Frozen contract: ``docs/proposal/multigpu_reference_validation_contract.md`` §6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from library.training.multigpu.agreement import (
    agree_any,
    all_ranks_equal,
    is_multiprocess,
)


@dataclass(frozen=True)
class StepAgreement:
    """The agreed outcome of one optimizer boundary.

    ``unhealthy`` is the OR of every rank-local reason the window may not be
    committed. There is deliberately **no** micro-batch-count field: it was
    always the constant ``1`` at the call site, and publishing a field that never
    carries a real observation would invent a symmetry source that does not
    exist (review t4). A rank walking a different number of micro-batches cannot
    reach the boundary at all – it hangs inside the gradient collective, which is
    the class-II failure documented in contract §9.
    """

    unhealthy: bool
    loss_nonfinite: bool
    gradient_nonfinite: bool
    force_skip: bool
    multiprocess: bool

    @property
    def drop_update(self) -> bool:
        """Whether the update must be dropped on **every** rank."""
        return self.unhealthy


def agree_step(
    accelerator,
    *,
    loss_nonfinite: bool,
    gradient_nonfinite: bool,
    force_skip: bool,
) -> StepAgreement:
    """Agree the boundary decision across ranks in a single collective.

    All four inputs are rank-local. The returned flags are the world-wide OR, so
    a rank that is locally clean still drops the update when a peer is not.
    """
    agreed = agree_any(
        accelerator,
        {
            "loss_nonfinite": bool(loss_nonfinite),
            "gradient_nonfinite": bool(gradient_nonfinite),
            "force_skip": bool(force_skip),
        },
    )
    return StepAgreement(
        unhealthy=any(agreed.values()),
        loss_nonfinite=agreed["loss_nonfinite"],
        gradient_nonfinite=agreed["gradient_nonfinite"],
        force_skip=agreed["force_skip"],
        multiprocess=is_multiprocess(accelerator),
    )


def agree_window_discard(accelerator, *, discard: bool) -> bool:
    """Agree the commit-versus-discard decision at the window seam.

    Replay counts are kept identical by :func:`agree_step` (a peer's skip reason
    makes every rank enter the same replay), but the *outcome* of a replay is
    again rank-local. Without this second agreement one rank can commit a
    replayed window while the other discards it, which desynchronises
    ``global_step`` and every later collective.
    """
    return agree_any(accelerator, {"discard": bool(discard)})["discard"]


def scaler_state(accelerator) -> Optional[dict[str, float]]:
    """Snapshot the scaler scale and growth tracker (``None`` without a scaler).

    ``get_scale`` is public API; the growth tracker has no public reader, so it
    is read defensively and simply omitted when unavailable. Nothing here writes
    scaler internals.
    """
    scaler = getattr(accelerator, "scaler", None)
    if scaler is None:
        return None
    state: dict[str, float] = {}
    get_scale = getattr(scaler, "get_scale", None)
    if callable(get_scale):
        state["scale"] = float(get_scale())
    tracker = getattr(scaler, "_growth_tracker", None)
    if tracker is not None and hasattr(tracker, "item"):
        state["growth_tracker"] = float(tracker.item())
    return state or None


def assert_scaler_symmetry(accelerator, *, where: str) -> None:
    """Fail loudly (on every rank) when scaler state diverged.

    Symmetry is expected by construction: the boundary decision is agreed before
    any rank may step, so either every rank runs ``scaler.step()``/``update()``
    or none does. A mismatch therefore means the protocol was bypassed, and the
    contract requires a bounded symmetric abort instead of a silent drift.
    """
    if not is_multiprocess(accelerator):
        return
    state = scaler_state(accelerator)
    if not state:
        return
    locally_equal = all_ranks_equal(accelerator, state)
    mismatch = not all(locally_equal.values())
    # OR the local verdict so the abort is taken by every rank, not only by the
    # ranks that happen to hold the outlier.
    if agree_any(accelerator, {"scaler_state_mismatch": mismatch})[
        "scaler_state_mismatch"
    ]:
        raise RuntimeError(
            "multi-process GradScaler state diverged at "
            f"{where}: local={state}. Scale and growth tracker must stay "
            "identical on every rank (contract §6 W-3.2); refusing to continue "
            "with an asymmetric optimizer state."
        )


def aggregate_skipped(state: Any) -> bool:
    """Convenience reader for the loop's ``update_skipped`` marker."""
    return bool(getattr(state, "update_skipped", False))


class BoundaryAgreementError(RuntimeError):
    """Raised on **every** rank when any rank failed inside the boundary block.

    Same shape as the P0-2 startup probe: one rank's local failure becomes a
    named, agreed failure for the whole job instead of a job that dies on one
    rank while its peers wait inside the next collective.
    """

    code = "multigpu_boundary_error"


#: Fragments that mark a *transport* failure rather than a local training fault.
_TRANSPORT_MARKERS = (
    "nccl",
    "gloo",
    "socket",
    "broken pipe",
    "connection reset",
    "connection closed",
    "timed out",
    "timeout",
    "processgroup",
    "process group",
    "distbackend",
    "cuda error",
)


def is_transport_error(exc: BaseException) -> bool:
    """Whether ``exc`` looks like a broken process group / transport failure.

    The contract (§9) distinguishes two classes: a **local** error on a healthy
    process group must be negotiated, while a transport failure must not start
    any new collective (the group is unusable, so agreeing would hang instead of
    failing). This classifier is deliberately conservative: anything that looks
    like a distributed/transport fault is treated as class II.
    """
    if exc is None:
        return False
    for base in type(exc).__mro__:
        if base.__module__.startswith("torch.distributed"):
            return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _TRANSPORT_MARKERS)


def negotiate_boundary_error(accelerator, error: Optional[BaseException], *, dropped: bool) -> None:
    """Agree a boundary-local failure; raise the same named error on all ranks.

    Called once per optimizer boundary on **every** rank, which keeps the
    collective order identical whether or not a rank failed.

    * single process – re-raises the captured error unchanged (no collective);
    * transport-looking failure – re-raised immediately, no new collective
      (class II: the group is already unusable);
    * anything else – OR-agreed through the same primitive used for the window
      decision, and every rank raises :class:`BoundaryAgreementError` when any
      rank reported one.
    """
    if error is None and not is_multiprocess(accelerator):
        return
    if error is not None and (is_transport_error(error) or not is_multiprocess(accelerator)):
        raise error
    flagged = agree_any(accelerator, {"boundary_error": error is not None})[
        "boundary_error"
    ]
    if flagged:
        raise BoundaryAgreementError(
            f"{BoundaryAgreementError.code}: a rank failed inside the optimizer "
            f"boundary (dropped_update={dropped}); aborting every rank instead of "
            "letting a peer enter the next collective alone"
            + (f" – local error: {type(error).__name__}: {error}" if error else "")
        )
