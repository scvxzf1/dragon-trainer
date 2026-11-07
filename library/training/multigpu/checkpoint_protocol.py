"""Distributed checkpoint publish protocol (contract §8 P1-1/P1-2).

``Accelerator.save_state`` already makes every rank participate: each rank writes
its own shard into the target directory. What it does **not** do is publish
safely, and the pre-existing code showed both gaps:

* the ``state_dir`` ⇄ ``backup_dir`` ⇄ ``tmp_dir`` renaming ran on **every** rank,
  so several processes raced over the same directory names;
* the ``train_state.json`` sidecar was written outside any main-process gate.

The rules implemented here, all fail-closed:

1. every rank collects and validates its own share (that part stays with
   ``accelerator.save_state``) and then reports the capture verdict through **one**
   agreed flag *before* the pre-publish barrier, so a rank-local failure cannot
   leave one rank publishing while its peer rolls back — and cannot pair a failing
   rank's collective with a healthy rank's barrier (which deadlocked this save);
2. only rank 0 touches the directory names (atomic rename + marker), every other
   rank waits at a barrier — no rank ever sees a half-published ``state_dir``;
3. a **local** publish failure becomes a named, agreed failure on every rank and
   the previous checkpoint is restored, so old state and user files survive;
4. a **transport** failure is re-raised on the spot without starting a new
   collective: a broken process group is class II (contract §9), and "agreeing"
   on it would hang instead of failing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

CODE_PUBLISH_FAILED = "multigpu_checkpoint_publish_failed"
CODE_RECORD_FAILED = "multigpu_checkpoint_record_failed"

#: Fixed key set **and order** of the checkpoint agreement collectives.
#:
#: The key set and its order are part of the protocol: every rank enters both
#: agreements of a save (capture, then publish) at the same positions, and
#: ``agree_any`` packs the flags in mapping order into one tensor. If one rank
#: sent ``{"save_state_failed": ...}`` and a peer sent ``{"publish_failed": ...}``
#: the payloads would differ in shape and meaning while the collective still
#: matched positionally - the classic silent split. Every path therefore builds
#: its payload with :func:`agreement_flags` and reads the verdict with
#: :func:`agreement_failed`; never pass a hand-written dict here.
AGREEMENT_KEYS = ("publish_failed", "save_state_failed")


def agreement_flags(
    *, publish_failed: bool = False, save_state_failed: bool = False
) -> dict[str, bool]:
    """Build the checkpoint agreement payload with the fixed key set and order.

    Both the capture-failure path and the publish path call this, so a rank that
    failed to capture and a rank that captured fine still exchange the same two
    keys in the same order. The collective's key set/order is a protocol element,
    not an implementation detail.
    """
    flags = {
        "publish_failed": bool(publish_failed),
        "save_state_failed": bool(save_state_failed),
    }
    assert tuple(flags) == AGREEMENT_KEYS, "agreement key order must stay stable"
    return flags


def agreement_failed(flags: Mapping[str, bool]) -> bool:
    """True when *any* rank reported a capture or publish failure."""
    return any(bool(flags.get(key, False)) for key in AGREEMENT_KEYS)


class CheckpointPublishError(RuntimeError):
    """Raised on every rank when the rank-0 publish of a state dir failed."""

    code = CODE_PUBLISH_FAILED

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        if code is not None:
            self.code = code


@dataclass
class PublishOutcome:
    state_dir: str
    published: bool
    error: Optional[str] = None
    transport: bool = False


def is_multiprocess(accelerator) -> bool:
    return int(getattr(accelerator, "num_processes", 1) or 1) > 1


def _is_main(accelerator) -> bool:
    return int(getattr(accelerator, "process_index", 0) or 0) == 0


def _transport(exc: BaseException) -> bool:
    from library.training.multigpu.window_protocol import is_transport_error

    return is_transport_error(exc)


def _remove(path: str) -> None:
    from library.training.checkpoints import _remove_path

    if os.path.exists(path):
        _remove_path(path)


def recover_state_dirs(state_dir: str, tmp_dir: str, backup_dir: str) -> None:
    """Drop stale tmp/backup dirs so a fresh publish starts from a known state.

    Rank 0 only: the directories are shared, and letting every rank delete them
    is the race this module exists to remove.
    """
    if os.path.exists(tmp_dir):
        _remove(tmp_dir)
    if os.path.exists(backup_dir):
        if not os.path.exists(state_dir):
            os.replace(backup_dir, state_dir)
        else:
            _remove(backup_dir)


def rollback_publish(state_dir: str, backup_dir: str) -> None:
    """Undo a publish that the group then refused (peer capture failed)."""
    if os.path.exists(state_dir):
        _remove(state_dir)
    if os.path.exists(backup_dir):
        os.replace(backup_dir, state_dir)


def finalize_publish(backup_dir: str) -> None:
    if os.path.exists(backup_dir):
        _remove(backup_dir)


def publish_state_dir(
    state_dir: str,
    tmp_dir: str,
    backup_dir: str,
    *,
    after_publish: Optional[Callable[[str], None]] = None,
    recover: bool = False,
    keep_backup: bool = False,
) -> PublishOutcome:
    """Atomically publish ``tmp_dir`` as ``state_dir`` (rank 0 only).

    Order matters: the previous state is moved aside first, the new one is
    renamed into place, and the backup is only dropped once the new state exists.
    Any failure restores the backup, so the old checkpoint survives.
    """
    try:
        if recover:
            # Recovery drops ``tmp_dir``; it must happen *before* the ranks write
            # their shards, never afterwards (that deleted a freshly written save).
            recover_state_dirs(state_dir, tmp_dir, backup_dir)
        if not os.path.exists(tmp_dir):
            raise FileNotFoundError(f"no state written to {tmp_dir} before publish")
        if os.path.exists(state_dir):
            os.replace(state_dir, backup_dir)
        os.replace(tmp_dir, state_dir)
        if after_publish is not None:
            after_publish(state_dir)
        if not keep_backup and os.path.exists(backup_dir):
            _remove(backup_dir)
        return PublishOutcome(state_dir=state_dir, published=True)
    except BaseException as exc:  # restore first, re-raise/agree afterwards
        try:
            if os.path.exists(tmp_dir):
                _remove(tmp_dir)
            if os.path.exists(backup_dir):
                if os.path.exists(state_dir):
                    _remove(state_dir)
                os.replace(backup_dir, state_dir)
        except BaseException as restore_exc:  # pragma: no cover - disk level
            return PublishOutcome(
                state_dir=state_dir,
                published=False,
                error=f"{type(exc).__name__}: {exc}; restore failed: {restore_exc}",
                transport=_transport(exc) or _transport(restore_exc),
            )
        return PublishOutcome(
            state_dir=state_dir,
            published=False,
            error=f"{type(exc).__name__}: {exc}",
            transport=_transport(exc),
        )


def _agree(accelerator, **flags) -> dict[str, bool]:
    """One agreement collective with the fixed key set/order of :func:`agreement_flags`."""
    from library.training.multigpu.agreement import agree_any

    return agree_any(accelerator, agreement_flags(**flags))


def _drop_half_written(state_dir: str, tmp_dir: str, backup_dir: str) -> bool:
    """Remove a refused ``tmp_dir`` and put the previous state back (rank 0 only).

    Only the publisher owns the shared directory names: a peer that removes
    ``tmp_dir`` also destroys the shards its siblings just wrote.
    """
    ok = True
    try:
        if os.path.exists(tmp_dir):
            _remove(tmp_dir)
        if os.path.exists(backup_dir) and not os.path.exists(state_dir):
            os.replace(backup_dir, state_dir)
    except Exception:  # pragma: no cover - disk level
        ok = False
    return ok


def save_state_published(
    accelerator,
    tmp_dir: str,
    state_dir: str,
    backup_dir: str,
    *,
    after_publish: Optional[Callable[[str], None]] = None,
    before_publish: Optional[Callable[[str], None]] = None,
) -> PublishOutcome:
    """Full save/publish sequence that every rank calls.

    ``before_publish(tmp_dir)`` runs on **every** rank after its own shard landed
    and before rank 0 publishes; a barrier separates the two so a peer's per-rank
    artifact (for example its loader cursor) is inside the directory that gets
    renamed. It only touches this rank's own files, so it cannot race.

    ``accelerator.save_state(tmp_dir)`` is what makes the save *all-rank*; the
    publish itself is rank-0 only, and the outcome is agreed before any rank
    continues. Legacy single-process behaviour is unchanged: no barrier, no
    collective.

    **Collective order is part of the protocol.** A save enters exactly two
    agreements, always in this order and always on every rank:

    1. the **capture** verdict, right after this rank's own ``save_state`` and
       *before* the pre-publish barrier;
    2. the **publish** verdict, after rank 0 renamed the directory.

    Both use the fixed payload from :func:`agreement_flags` (``publish_failed``,
    ``save_state_failed``, in that order), because the flags are packed in mapping
    order into one tensor: a rank that failed to capture and a rank that captured
    fine must reach the same collective with the same shape, and both read the
    verdict through :func:`agreement_failed`, so either failure stops every rank.

    The capture agreement must precede the pre-publish barrier. Negotiating the
    failure *after* it, from inside the ``except`` block, paired a failing rank's
    collective against its sibling's barrier and hung the save until the process
    group timeout (review t25: only ``..._local_capture_failure_...`` hung, for
    exactly this reason).
    """
    if is_multiprocess(accelerator):
        # Rank 0 clears stale tmp/backup dirs first and everyone waits, so no rank
        # can write into a directory another rank is deleting.
        if _is_main(accelerator):
            recover_state_dirs(state_dir, tmp_dir, backup_dir)
        if hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()
    else:
        # Single process: stale tmp/backup dirs and leftover shards from a crashed
        # run must be cleared *before* this run writes, otherwise the published
        # state can mix two runs (review t22).
        recover_state_dirs(state_dir, tmp_dir, backup_dir)

    capture_error: Optional[BaseException] = None
    try:
        accelerator.save_state(tmp_dir)
    except Exception as exc:
        if not is_multiprocess(accelerator):
            # Single process: clean this rank's half-written directory and keep the
            # old state. Nobody else is writing, so the cleanup cannot race.
            _drop_half_written(state_dir, tmp_dir, backup_dir)
            raise
        if _transport(exc):
            # Class II: the group is unusable, so re-raise on the spot and do not
            # start a collective (contract §9). The shared directories are left
            # untouched on purpose: a peer may still be writing its shard, and
            # deleting ``tmp_dir`` under it turns one rank's transport error into
            # everyone's local write failure.
            raise
        # Class I (healthy group, rank-local capture failure): remember it and let
        # the phase-1 agreement below carry it to every rank.
        capture_error = exc

    if is_multiprocess(accelerator):
        # ---- phase 1: the capture verdict, at one fixed position ---------------
        flags = _agree(accelerator, save_state_failed=capture_error is not None)
        if agreement_failed(flags):
            if _is_main(accelerator):
                _drop_half_written(state_dir, tmp_dir, backup_dir)
            if hasattr(accelerator, "wait_for_everyone"):
                accelerator.wait_for_everyone()
            detail = (
                f"{type(capture_error).__name__}: {capture_error}"
                if capture_error is not None
                else "a peer rank reported the failure"
            )
            raise CheckpointPublishError(
                f"{CODE_PUBLISH_FAILED}: a rank failed inside save_state "
                f"({detail}); no rank publishes this state",
                code=CODE_PUBLISH_FAILED,
            ) from capture_error

    if before_publish is not None:
        before_publish(tmp_dir)
    if is_multiprocess(accelerator) and hasattr(accelerator, "wait_for_everyone"):
        # Peers write their own per-rank artifacts (cursor/identity) into tmp_dir;
        # rank 0 must not rename the directory before they all landed.
        accelerator.wait_for_everyone()
    if not is_multiprocess(accelerator):
        outcome = publish_state_dir(
            state_dir, tmp_dir, backup_dir, after_publish=after_publish, recover=False
        )
        if not outcome.published:
            raise CheckpointPublishError(
                f"{CODE_PUBLISH_FAILED}: {outcome.error}"
            )
        return outcome

    if _is_main(accelerator):
        # Keep the backup until the group agrees: if a peer reports a publish
        # failure we must undo this publish instead of leaving a state dir that no
        # rank can resume from.
        outcome = publish_state_dir(
            state_dir,
            tmp_dir,
            backup_dir,
            after_publish=after_publish,
            recover=False,
            keep_backup=True,
        )
    else:
        outcome = PublishOutcome(state_dir=state_dir, published=False)
    if outcome.transport:
        # Class II: the group is already broken; agreeing would hang.
        raise CheckpointPublishError(
            f"{CODE_PUBLISH_FAILED} (transport): {outcome.error}"
        )

    # ---- phase 2: the publish verdict, at one fixed position -------------------
    # Only rank 0 publishes, so only its outcome may count as a failure. Peers
    # contribute False and simply learn the verdict - otherwise every peer would
    # report "not published" and the group would always fail.
    local_failed = (not outcome.published) if _is_main(accelerator) else False
    flags = _agree(accelerator, publish_failed=local_failed)
    if agreement_failed(flags):
        if _is_main(accelerator) and outcome.published:
            rollback_publish(state_dir, backup_dir)
        raise CheckpointPublishError(
            f"{CODE_PUBLISH_FAILED}: rank 0 could not publish {state_dir} "
            f"({outcome.error or 'peer reported failure'}); the previous "
            "checkpoint was restored and this save is refused"
        )
    if hasattr(accelerator, "wait_for_everyone"):
        accelerator.wait_for_everyone()
    if _is_main(accelerator):
        finalize_publish(backup_dir)
    return PublishOutcome(state_dir=state_dir, published=True)


def checkpoint_records(accelerator, *, global_step: int, epoch: int) -> dict[str, Any]:
    """Per-rank checkpoint facts agreed across the world before publishing.

    A step/epoch that differs between ranks means the ranks disagree about what
    they just trained; publishing that state would create a checkpoint nobody can
    resume exactly.
    """
    local = {"global_step": int(global_step), "epoch": int(epoch)}
    if not is_multiprocess(accelerator):
        return local
    from library.training.multigpu.window_protocol import all_ranks_equal

    equal = all_ranks_equal(
        accelerator, {"global_step": float(global_step), "epoch": float(epoch)}
    )
    if not all(equal.values()):  # ``all_ranks_equal`` returns a mapping, not a bool
        raise CheckpointPublishError(
            f"{CODE_RECORD_FAILED}: ranks disagree about the checkpoint position "
            f"(local step={global_step}, epoch={epoch}); refusing to publish a "
            "state that cannot be resumed exactly",
            code=CODE_RECORD_FAILED,
        )
    local["agreed"] = True
    return local
