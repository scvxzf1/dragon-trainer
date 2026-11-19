"""Trainer-level resume metadata for the MOSS PP training path.

The per-rank PP checkpoint (:mod:`library.training.pp.checkpoint`) owns the
stage weights, the stage-local optimizer, the fp16 scale/policy and the skip
decision.  The *trainer* layer owns the rest of an exact resume: the LR
scheduler state, the per-rank RNG, the data cursor and the run identity.

Format
------
One ``trainer_state_rank{r}.pt`` per rank inside the PP run root, written by
that rank and negotiated like every other PP artifact: per-rank payloads are
validated (version/family/run identity/step against the negotiated manifest
step) and only an all-rank zero-error agreement releases the apply phase.  A
refused load mutates nothing.

Compatibility rule (frozen contract §4)
---------------------------------------
An old PP checkpoint without trainer-state files is **not** an exact resume:
there is no scheduler, RNG or cursor to restore, so continuing it silently
would replay data and drift the LR plan.  The default is a named refusal; an
explicit user opt-in (``--pipeline_parallel_resume_weights_only`` style TOML
key ``pipeline_parallel_resume_weights_only`` or the environment variable
``MOSS_PP_RESUME_WEIGHTS_ONLY=1``) continues with weights only, explicitly
logged as non-exact, and never claims scheduler/cursor recovery.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist

from .errors import (
    CODE_CHECKPOINT_IDENTITY_MISMATCH,
    CODE_CHECKPOINT_INCOMPLETE,
    CODE_CHECKPOINT_STATE_MISMATCH,
    CODE_CHECKPOINT_STEP_MISMATCH,
    CODE_RESUME_NO_WORK,
    PipelineParallelError,
)

logger = logging.getLogger(__name__)

TRAINER_RESUME_FORMAT = "moss_image_pp_trainer_resume"
TRAINER_RESUME_VERSION = 1

WEIGHTS_ONLY_ENV = "MOSS_PP_RESUME_WEIGHTS_ONLY"


def trainer_resume_format(family: str) -> str:
    """The per-family format stamp: ``<family>_pp_trainer_resume`` (§4).

    ``moss_image`` maps back to the historical constant, so existing MOSS
    payloads stay byte-compatible; Anima writes ``anima_pp_trainer_resume``.
    """

    from .trainer_adapters import trainer_resume_format as _per_family

    return _per_family(family)


def trainer_state_path(root, rank: int):
    from pathlib import Path

    return Path(root) / f"trainer_state_rank{int(rank)}.pt"


def trainer_state_present(root) -> bool:
    from pathlib import Path

    return (Path(root) / "pp_checkpoint.json").is_file() and any(
        trainer_state_path(root, rank).is_file() for rank in range(2)
    )


def weights_only_opt_in(args, family: str = "moss_image") -> bool:
    """The explicit user choice for continuing without trainer state."""

    from .gate import _get

    if bool(_get(args, "pipeline_parallel_resume_weights_only", False)):
        return True
    for name in (f"{str(family).upper()}_PP_RESUME_WEIGHTS_ONLY", WEIGHTS_ONLY_ENV):
        if os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}:
            return True
    return False


def _distributed(group) -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def _gather_errors(local: dict, *, group, context: str) -> None:
    """Share one per-rank verdict; any failure is a symmetric named refusal."""

    if not _distributed(group):
        if local["error"]:
            raise PipelineParallelError(
                local["code"] or CODE_CHECKPOINT_INCOMPLETE,
                f"{context} refused: {local['error']}",
            )
        return
    gathered: list = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, local, group=group)
    failed = [entry for entry in gathered if entry and entry["error"]]
    if failed:
        raise PipelineParallelError(
            failed[0]["code"] or CODE_CHECKPOINT_INCOMPLETE,
            f"{context} refused: " + "; ".join(entry["error"] for entry in failed),
        )


def capture_trainer_payload(
    *,
    rank: int,
    run_id: str,
    family: str,
    global_step: int,
    effective_step: int,
    epoch: int,
    batch_index: int,
    lr_scheduler,
    seed: int | None,
) -> dict:
    """The per-rank trainer-level state as a plain payload."""

    payload = {
        "format": trainer_resume_format(family),
        "version": TRAINER_RESUME_VERSION,
        "rank": int(rank),
        "run_id": str(run_id),
        "family": str(family),
        "global_step": int(global_step),
        "effective_step": int(effective_step),
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "scheduler": None
        if lr_scheduler is None
        else lr_scheduler.state_dict(),
        "rng_torch_cpu": torch.get_rng_state(),
        "rng_torch_cuda": (
            torch.cuda.get_rng_state()
            if torch.cuda.is_available()
            else None
        ),
        # The trainer layer owns *every* RNG the loop touches: torch (cpu/cuda),
        # python's `random` and numpy. A resume that restores only one of them
        # would let noise/dropout sources drift apart between ranks.
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "seed": None if seed is None else int(seed),
    }
    return payload


def save_trainer_resume(
    root,
    *,
    rank: int,
    payload: dict,
    group=None,
) -> str:
    """Write this rank's trainer state into the run root, negotiated.

    Each rank writes only its own file; a rank-local failure is captured and
    shared so both ranks fail together instead of one writing a half-resume.
    """

    error: dict = {"rank": int(rank), "error": None, "code": None}
    path = trainer_state_path(root, rank)
    try:
        torch.save(payload, path)
    except Exception as exc:  # noqa: BLE001 - negotiated below
        error["error"] = f"rank {rank}: {type(exc).__name__}: {exc}"
        error["code"] = getattr(exc, "code", None)
    _gather_errors(error, group=group, context="PP trainer resume save")
    return str(path)


def validate_trainer_payload(
    payload: dict, *, rank: int, run_id: str, family: str, expected_step: int
) -> None:
    """Pure validation; raises before any mutation."""

    expected_format = trainer_resume_format(family)
    if not isinstance(payload, dict) or payload.get("format") != expected_format:
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            f"rank {rank} trainer state is not {expected_format} "
            f"version 1 payload (format={payload.get('format') if isinstance(payload, dict) else type(payload).__name__!r}); "
            "cross-family PP resume payloads are refused",
        )
    if int(payload.get("version", -1)) != TRAINER_RESUME_VERSION:
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            f"rank {rank} trainer state version {payload.get('version')!r} is not "
            f"{TRAINER_RESUME_VERSION}",
        )
    if int(payload.get("rank", -1)) != int(rank):
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            f"trainer_state_rank{rank}.pt carries rank={payload.get('rank')!r}",
        )
    if str(payload.get("run_id")) != str(run_id):
        raise PipelineParallelError(
            CODE_CHECKPOINT_IDENTITY_MISMATCH,
            f"rank {rank} trainer state run_id={payload.get('run_id')!r} does not "
            f"match this run's identity {run_id!r}",
        )
    if str(payload.get("family")) != str(family):
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            f"rank {rank} trainer state family={payload.get('family')!r}, "
            f"expected {family!r}",
        )
    if int(payload.get("global_step", -1)) != int(expected_step):
        raise PipelineParallelError(
            CODE_CHECKPOINT_STEP_MISMATCH,
            f"rank {rank} trainer state step={payload.get('global_step')!r} does "
            f"not match the negotiated PP checkpoint step {expected_step}",
        )


def load_trainer_resume(
    root,
    *,
    rank: int,
    run_id: str,
    family: str,
    expected_step: int,
    lr_scheduler=None,
    group=None,
) -> dict:
    """Read, validate (negotiated), then apply this rank's trainer state."""

    path = trainer_state_path(root, rank)
    error: dict = {"rank": int(rank), "error": None, "code": None}
    payload: dict | None = None
    try:
        if not path.is_file():
            raise PipelineParallelError(
                CODE_CHECKPOINT_INCOMPLETE,
                f"missing {path}; this run root has no trainer-level state, so it "
                "cannot be resumed exactly (use the weights-only opt-in instead)",
            )
        payload = torch.load(path, weights_only=False)
        validate_trainer_payload(
            payload, rank=rank, run_id=run_id, family=family,
            expected_step=expected_step,
        )
    except Exception as exc:  # noqa: BLE001 - negotiated below
        error["error"] = f"rank {rank}: {type(exc).__name__}: {exc}"
        error["code"] = getattr(exc, "code", None)
    _gather_errors(error, group=group, context="PP trainer resume load")
    assert payload is not None  # zero-error agreement above
    # ---- apply phase (all validation passed, nothing mutated before) --------
    scheduler_state = payload.get("scheduler")
    if lr_scheduler is not None and scheduler_state is not None:
        lr_scheduler.load_state_dict(scheduler_state)
    torch.set_rng_state(payload["rng_torch_cpu"])
    if payload.get("rng_torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(payload["rng_torch_cuda"])
    random.setstate(payload["rng_python"])
    np.random.set_state(payload["rng_numpy"])
    return payload


def require_exact_resume_or_opt_in(root, args, *, family: str) -> bool:
    """Refuse a trainer-state-less PP checkpoint unless the user opted in.

    Returns ``True`` when the run may continue (exact trainer state present, or
    the explicit weights-only opt-in).  Never silently downgrades.
    """

    if trainer_state_present(root):
        return True
    if weights_only_opt_in(args, family=family):
        logger.warning(
            "resuming PP checkpoint %s without trainer-level state "
            "(scheduler/RNG/data cursor unavailable): weights-only continuation, "
            "NOT an exact resume", root,
        )
        return True
    raise PipelineParallelError(
        CODE_CHECKPOINT_INCOMPLETE,
        f"PP run root {root} has no trainer-level resume state (scheduler, "
        "per-rank RNG, data cursor). Continuing it silently would replay data "
        "and drift the LR plan. Re-run with the explicit weights-only opt-in "
        "(config key pipeline_parallel_resume_weights_only=true or env "
        f"{WEIGHTS_ONLY_ENV}=1) to continue non-exactly, or resume from a PP "
        "bundle saved by this trainer.",
    )


def resolve_resume_position(
    payload: dict, *, step: int, batches_per_epoch: int, total_steps: int, group=None
) -> tuple[int, int, int]:
    """Validate the shared cursor and return the epoch, offset and effective step."""

    raw_position = tuple(payload.get(key) for key in ("epoch", "batch_index", "effective_step"))
    position = tuple(value if type(value) is int else None for value in raw_position)
    if _distributed(group):
        plan = (position, step, batches_per_epoch, total_steps)
        plans = [None] * dist.get_world_size(group)
        dist.all_gather_object(plans, plan, group=group)
        if any(peer != plan for peer in plans):
            raise PipelineParallelError(
                CODE_CHECKPOINT_STATE_MISMATCH,
                f"PP ranks disagree on the resume cursor or training budget: {plans}",
            )
    if step >= total_steps or batches_per_epoch == 0:
        raise PipelineParallelError(
            CODE_RESUME_NO_WORK,
            f"checkpoint step={step}, target max_train_steps={total_steps}, "
            f"batches_per_epoch={batches_per_epoch}; PP resume requires new training work",
        )

    epoch, batch_index, effective_step = position
    if (
        None in position
        or epoch < 0
        or not 0 <= batch_index <= batches_per_epoch
        or not 0 <= effective_step <= step
        or epoch * batches_per_epoch + batch_index != step
    ):
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            f"PP checkpoint step={step} has an invalid resume cursor "
            f"(epoch, batch_index, effective_step)={raw_position!r} "
            f"for batches_per_epoch={batches_per_epoch}",
        )
    if batch_index == batches_per_epoch:
        return epoch + 1, 0, effective_step
    return epoch, batch_index, effective_step


__all__ = [
    "TRAINER_RESUME_FORMAT",
    "TRAINER_RESUME_VERSION",
    "WEIGHTS_ONLY_ENV",
    "capture_trainer_payload",
    "load_trainer_resume",
    "require_exact_resume_or_opt_in",
    "resolve_resume_position",
    "save_trainer_resume",
    "trainer_state_path",
    "trainer_state_present",
    "trainer_resume_format",
    "validate_trainer_payload",
    "weights_only_opt_in",
]
