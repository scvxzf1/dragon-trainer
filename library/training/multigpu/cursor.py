"""Loader/sampler cursor bookkeeping for exact resume (contract §8 P1-2).

A resume is only *exact* when the data order continues where it stopped. The
checkpoint bundle therefore records a **real** per-rank position that the training
loop produced, and the resume path compares it with the position the resume plan
will actually skip to.

This module is deliberately tiny and side-effect free so both sides can share it:

* the saver asks :func:`make_cursor` for the payload of the *committed* state;
* the resume path asks :func:`expected_position` what the skip plan will produce
  and :func:`check_cursor` to compare the two.

Anything else (a missing key, a string placeholder such as ``"not_reported"``, a
different rank or epoch) is a named mismatch instead of a silent "close enough".
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from library.training.multigpu.resume_protocol import (
    CODE_CURSOR_MISMATCH,
    CODE_CURSOR_UNKNOWN,
    ResumeBundleError,
)

CURSOR_SCHEMA = "multigpu_loader_cursor_v1"

#: The fields two cursors must agree on. A cursor payload may carry extra
#: bookkeeping (device type/index, batches per epoch) that is recorded but never
#: compared - otherwise a device report would look like a position drift.
COMPARED_KEYS = ("sampler_position", "epoch", "rank")

#: Keys that must be present with the right *type* for a cursor to count as real.
_INT_KEYS = COMPARED_KEYS


def comparable(cursor: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Project a cursor payload onto the compared fields."""
    source = dict(cursor or {})
    return {key: source.get(key) for key in COMPARED_KEYS}


def make_cursor(
    *,
    position: int,
    epoch: int,
    rank: int,
    batches_per_epoch: Optional[int] = None,
) -> dict[str, Any]:
    """Build the cursor payload written into the resume bundle.

    ``position`` is the number of dataloader batches **this rank** has consumed
    since the start of the run (cumulative across epochs, including batches that a
    resume skipped). It is therefore comparable with :func:`expected_position`.
    """
    if not isinstance(position, int) or position < 0:
        raise ResumeBundleError(
            CODE_CURSOR_UNKNOWN, f"cursor position must be a non-negative int, got {position!r}"
        )
    payload: dict[str, Any] = {
        "schema": CURSOR_SCHEMA,
        "sampler_position": int(position),
        "epoch": int(epoch),
        "rank": int(rank),
    }
    if batches_per_epoch is not None:
        payload["batches_per_epoch"] = int(batches_per_epoch)
    return payload


def expected_position(
    *, epoch_to_start: int, initial_step: int, batches_per_epoch: int
) -> int:
    """Batches this rank must already have consumed when the run re-enters.

    Mirrors exactly what the loop does: ``range(epoch_to_start, ...)`` whole
    epochs, then ``skip_first_batches(initial_step - 1)`` **plus** the one batch the
    loop consumes through ``if state.initial_step > 0: ... continue`` inside the
    starting epoch. That is ``initial_step`` batches, so the first batch that really
    runs carries local dataloader index ``initial_step`` (t3 F-T3-2: the old
    ``initial_step - 1`` made the counter disagree with the real index by one, and
    the noise fixture pinned the wrong samples after every resume).
    """
    whole_epochs = max(0, int(epoch_to_start)) * max(0, int(batches_per_epoch))
    inside_epoch = max(0, int(initial_step))
    return whole_epochs + inside_epoch


def resume_skipped_batches(
    *, epoch_to_start: int, initial_step: int, batches_per_epoch: int
) -> int:
    """Batches this rank has really consumed when the resumed run re-enters.

    The single producer of the loop's starting ``batches_consumed`` counter: whole
    epochs the loop skips, plus the residual prefix of the starting epoch. The
    prefix is ``initial_step`` batches (``skip_first_batches(initial_step - 1)``
    plus the batch the loop consumes through ``if state.initial_step > 0: ...
    continue``), which is why this is the number the first *executed* batch's local
    dataloader index equals (t3 F-T3-2).
    """
    return (
        max(0, int(epoch_to_start)) * max(0, int(batches_per_epoch))
        + max(0, int(initial_step))
    )


def is_real_cursor(cursor: Optional[Mapping[str, Any]]) -> bool:
    """True only when the payload carries real typed values (never a placeholder)."""
    if not isinstance(cursor, Mapping):
        return False
    if cursor.get("schema") != CURSOR_SCHEMA:
        return False
    for key in _INT_KEYS:
        value = cursor.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            return False
    return int(cursor["sampler_position"]) >= 0


def check_cursor(
    saved: Optional[Mapping[str, Any]],
    *,
    expected: Mapping[str, Any],
) -> None:
    """Raise a named error unless ``saved`` matches ``expected`` exactly.

    Compared fields: ``sampler_position``, ``epoch`` and ``rank``. The saved
    cursor must also be a real cursor, so a bundle written before this protocol
    (with ``sampler_position="not_reported"``) cannot pass as exact.
    """
    if not is_real_cursor(saved):
        raise ResumeBundleError(
            CODE_CURSOR_UNKNOWN,
            "saved loader cursor is not a real per-rank cursor "
            f"(got {dict(saved or {})!r}): the data order cannot be reproduced",
        )
    for key in COMPARED_KEYS:
        if int(saved[key]) != int(expected[key]):
            raise ResumeBundleError(
                CODE_CURSOR_MISMATCH,
                f"loader cursor {key} mismatch: saved {saved[key]!r} != resume start "
                f"{expected[key]!r} (saved={dict(saved)!r}, expected={dict(expected)!r})",
            )
