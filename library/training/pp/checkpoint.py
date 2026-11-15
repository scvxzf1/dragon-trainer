"""Per-rank PP checkpoint protocol: new run root, manifest, negotiated restore.

A PP checkpoint is a *set* of per-rank stage artifacts plus one manifest that
binds them together; there is no single process that holds the whole training
state.  This module owns that protocol (the state itself, its content hashing and
its comparison live in :mod:`library.training.pp.checkpoint_state`).

Reuse, and why each piece is reused
-----------------------------------
* **Negotiate-before-apply.**  ``scripts/experiments/anima_parallel/checkpoint.py``
  ``load_distributed_checkpoint`` is the only correct per-rank resume shape in the
  repo: every rank first all-gathers ``{"step", "error"}`` and only then is *any*
  rank allowed to touch the pipeline.  A resume that raises on one rank while its
  peer already entered the schedule hangs instead of failing.  This module keeps
  that shape for both load **and** save (save adds ``sha256``/``identity``/
  ``last_skip`` to the negotiated payload).
* **Per-rank artifact ownership.**  ``library/training/multigpu/resume_protocol.py``
  :func:`rank_local_keys` refuses an artifact claimed by two ranks.  The manifest
  stores its output, so "which file belongs to which rank" is a checked fact.
* **``multigpu.identity`` supplies the record discipline, not the record builder.**
  Its ``collect_rank_summaries`` / ``write_runtime_identity`` establish the shape
  this module follows: every rank reports its own facts through one fixed-size
  collective and only rank 0 publishes the single file that describes the whole
  run (never rank 0's local view).  Its record *builder* is not reused because it
  reads ``accelerator.num_processes`` / ``process_index`` / ``device`` /
  ``mixed_precision`` / ``scaler_present``, and a PP rank owns a plain
  ``torch.distributed`` group, not an accelerator; its RNG/loader-cursor
  vocabulary belongs to the DDP resume bundle.  ``identity.record_resume_outcome``
  stays the entry point for writing a resume verdict into the run ledger, and that
  wiring belongs to the trainer path (``library/training/pp/gate.py``), not here.
* **Publish/rollback is *not* reused.**  ``multigpu.checkpoint_protocol`` exists
  because ``accelerator.save_state`` overwrites a live ``state_dir`` and needs a
  backup/rename dance.  A PP checkpoint is only ever written into a **newly
  created run root** (:func:`create_run_root` refuses an existing path), so there
  is no previous checkpoint to preserve and nothing to roll back.  Its
  ``agree_any`` agreement is also accelerate-bound, while PP checkpointing runs on
  a plain ``torch.distributed`` group.
* **``multigpu.assert_scaler_symmetry`` is deliberately not reused.**  Under PP the
  fp16 *scale* may legitimately differ per rank; what must agree is the **skip
  decision**, otherwise one half of the model steps and the other does not.  That
  is :data:`~library.training.pp.errors.CODE_SCALER_DECISION_ASYMMETRY`.

Everything is fail-closed: a changed block range, stage count, rank count,
schedule, latent shape, run identity or artifact byte, a missing manifest or
artifact, a step disagreement and a skip-decision disagreement all raise a named
``pp_checkpoint_*`` error on every rank.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch.distributed as dist

from .checkpoint_artifacts import (
    PpResumeState,
    _distributed,
    _mark_failed,
    _plan_record,
    _resolve_rank,
    _resolve_world_size,
    _validate_adapters,
    _write_manifest,
    create_run_root,
    read_manifest,
    read_stage_checkpoint,
    save_stage_checkpoint,
    stage_filename,
    stage_path,
)
from .checkpoint_state import (
    PP_CHECKPOINT_FORMAT,
    PP_CHECKPOINT_VERSION,
    capture_stage_snapshot,
    checkpoint_format_for_family,
    content_hash_file,
    topology_digest,
)
from .errors import (
    CODE_CHECKPOINT_IDENTITY_MISMATCH,
    CODE_CHECKPOINT_INCOMPLETE,
    CODE_CHECKPOINT_STATE_MISMATCH,
    CODE_CHECKPOINT_STEP_MISMATCH,
    CODE_SCALER_DECISION_ASYMMETRY,
    PipelineParallelError,
)


# ---------------------------------------------------------------------------
# distributed plumbing
# ---------------------------------------------------------------------------


def _all_gather_object(value: Any, group: Any = None) -> list[Any]:
    if not _distributed(group):
        return [value]
    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def _raise_agreed(outcomes: Iterable[Mapping[str, Any]], *, context: str) -> None:
    """Turn per-rank errors into one agreed, named failure on every rank."""

    errors = [
        (int(entry["rank"]), entry.get("code"), str(entry.get("error")))
        for entry in outcomes
        if entry.get("error")
    ]
    if not errors:
        return
    code = next((entry_code for _, entry_code, _ in errors if entry_code), None)
    detail = "; ".join(message for _, _, message in errors)
    raise PipelineParallelError(
        code or CODE_CHECKPOINT_INCOMPLETE, f"{context} refused: {detail}"
    )


def _agreement_payload(*, rank: int) -> dict[str, Any]:
    """Fixed key set of the save/load negotiation collective.

    The key set (and therefore the ``all_gather_object`` payload shape) must be
    identical on every rank and on both the save and the load path; a rank that
    reports ``{"step": ..., "error": ...}`` while a peer reports a richer dict
    would still "match" positionally and hide the difference.
    """

    return {
        "rank": int(rank),
        "step": None,
        "error": None,
        "code": None,
        "sha256": None,
        "identity": None,
        "topology": None,
        "topology_digest": None,
        "scale": None,
        "growth_tracker": None,
        "last_skip": None,
    }


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PpCheckpointResult:
    """What a save published: one run root, one manifest, one file per rank."""

    root: Path
    step: int
    manifest: dict[str, Any]
    files: dict[int, Path]

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "step": int(self.step),
            "files": {str(rank): str(path) for rank, path in sorted(self.files.items())},
            "world_size": int(self.manifest["world_size"]),
        }


def _build_manifest(
    outcomes: list[Mapping[str, Any]], *, world_size: int, step: int
) -> dict[str, Any]:
    """The manifest rank 0 publishes from the negotiated per-rank outcomes."""

    from library.training.multigpu.resume_protocol import rank_local_keys

    ranks = [
        {
            "rank": int(entry["rank"]),
            "file": stage_filename(int(entry["rank"])),
            "sha256": entry["sha256"],
            "topology": dict(entry["topology"]),
            "topology_digest": entry["topology_digest"],
            "scale": entry["scale"],
            "growth_tracker": entry["growth_tracker"],
            "last_skip": entry["last_skip"],
        }
        for entry in sorted(outcomes, key=lambda item: int(item["rank"]))
    ]
    # One artifact belongs to exactly one rank; ``rank_local_keys`` is the checked
    # form of that statement (it refuses a key claimed by two ranks).
    per_rank_files = rank_local_keys(
        {str(entry["rank"]): [stage_filename(int(entry["rank"]))] for entry in ranks}
    )
    first_topology = dict(ranks[0]["topology"])
    plan = _plan_record(first_topology, ranks)
    return {
        "format": checkpoint_format_for_family(first_topology.get("family")),
        "version": PP_CHECKPOINT_VERSION,
        "world_size": int(world_size),
        "stages": int(first_topology["stages"]),
        "step": int(step),
        "identity": dict(outcomes[0]["identity"]),
        "schedule": first_topology.get("schedule"),
        "conditioning": first_topology.get("conditioning"),
        "microbatches": first_topology.get("microbatches"),
        "mixed_precision": first_topology.get("mixed_precision"),
        "plan": plan,
        "rank_local_keys": per_rank_files,
        "ranks": ranks,
    }


def _create_run_root_agreed(root: str | Path, *, rank: int, group: Any) -> bool:
    """Rank 0 creates the run root; the first gather tells every rank the verdict.

    No rank may write a stage file into a root that was never created, and no rank
    may touch an existing directory, so the creation verdict is negotiated before
    anything else happens.
    """

    root_error: dict[str, Any] = {"rank": rank, "error": None, "code": None}
    created = False
    if rank == 0:
        try:
            create_run_root(root)
            created = True
        except Exception as exc:
            root_error["error"] = f"rank 0: {type(exc).__name__}: {exc}"
            root_error["code"] = getattr(exc, "code", None)
    _raise_agreed(
        _all_gather_object(root_error, group), context="PP checkpoint run root"
    )
    return created


def _write_stage_outcome(
    root: str | Path,
    runtime: Any,
    *,
    step: int,
    identity: Mapping[str, str] | None,
    rank: int,
) -> dict[str, Any]:
    """Capture and write *this* rank's artifact, or record why it failed.

    A rank-local failure is reported in the negotiated payload instead of raised,
    because a raise here would strand the peer inside the next collective.
    """

    outcome = _agreement_payload(rank=rank)
    try:
        snapshot = capture_stage_snapshot(runtime, step=step, identity=identity)
        path = save_stage_checkpoint(root, snapshot, rank=rank)
        outcome.update(
            step=snapshot.step,
            sha256=content_hash_file(path),
            identity=dict(snapshot.identity),
            topology=dict(snapshot.topology),
            topology_digest=topology_digest(snapshot.topology),
            scale=snapshot.scale,
            growth_tracker=snapshot.growth_tracker,
            last_skip=snapshot.last_skip,
        )
    except Exception as exc:  # negotiated by the caller, never raised rank-locally
        outcome["error"] = f"rank {rank}: {type(exc).__name__}: {exc}"
        outcome["code"] = getattr(exc, "code", None)
    return outcome


def _require_agreed_save(
    outcomes: list[Mapping[str, Any]], *, rank: int, root: str | Path, created: bool
) -> None:
    """One failure/step/identity/skip verdict for the whole pipeline.

    A save that fails on one rank and succeeds on the other is how a checkpoint
    becomes half-resumable, so every disagreement is raised on every rank.
    """

    try:
        _raise_agreed(outcomes, context="PP checkpoint save")
        steps = {entry["step"] for entry in outcomes}
        if len(steps) != 1:
            raise PipelineParallelError(
                CODE_CHECKPOINT_STEP_MISMATCH,
                "the stages saved different step counters "
                f"({sorted(str(value) for value in steps)}); a PP checkpoint can only "
                "resume the pipeline at one step",
            )
        identities = {
            json.dumps(entry["identity"], sort_keys=True) for entry in outcomes
        }
        if len(identities) != 1:
            raise PipelineParallelError(
                CODE_CHECKPOINT_IDENTITY_MISMATCH,
                "the stages disagree about the run identity "
                f"({sorted(identities)}); every rank must be saving the same run",
            )
        skips = {entry["last_skip"] for entry in outcomes}
        if len(skips) != 1:
            raise PipelineParallelError(
                CODE_SCALER_DECISION_ASYMMETRY,
                "the stages disagree about the last GradScaler skip decision "
                f"({sorted(str(value) for value in skips)}); one half of the model "
                "may have stepped while the other did not",
            )
    except Exception as exc:
        # Only a root *this save created* may be marked: an existing directory is
        # user data, and writing a marker into it would be the overwrite this
        # module exists to prevent.
        if rank == 0 and created:
            _mark_failed(root, str(exc))
        raise


def _publish_manifest_agreed(
    root: str | Path,
    outcomes: list[Mapping[str, Any]],
    *,
    rank: int,
    world_size: int,
    step: int,
    created: bool,
    group: Any,
) -> None:
    """Rank 0 publishes the manifest; the third gather releases the group."""

    publish_error: dict[str, Any] = {"rank": rank, "error": None, "code": None}
    if rank == 0:
        try:
            manifest = _build_manifest(outcomes, world_size=world_size, step=int(step))
            _write_manifest(root, manifest)
        except Exception as exc:
            publish_error["error"] = f"rank 0: {type(exc).__name__}: {exc}"
            publish_error["code"] = getattr(exc, "code", None)
    try:
        _raise_agreed(
            _all_gather_object(publish_error, group),
            context="PP checkpoint manifest publish",
        )
    except Exception as exc:
        if rank == 0 and created:
            _mark_failed(root, str(exc))
        raise


def save_distributed_checkpoint(
    root: str | Path,
    runtime: Any,
    *,
    step: int,
    identity: Mapping[str, str] | None = None,
    group: Any = None,
) -> PpCheckpointResult:
    """Save every rank's stage state, negotiating before anything is published.

    Every collective here is an all-rank ``all_gather_object`` with the *same*
    payload shape on every rank, and every failure is captured into that payload
    instead of raised locally.  That is what keeps the failure paths deadlock-free:
    a barrier that only the healthy rank reaches, or a rank-local raise while the
    peer waits in a collective, hangs instead of failing.

    Order, and why:

    1. :func:`_create_run_root_agreed` -- rank 0 creates the run root (refusing an
       existing path) and the **first gather** tells every rank the verdict;
    2. :func:`_write_stage_outcome` -- each rank captures and writes *its own*
       artifact and reports ``{step, sha256, identity, topology, scale,
       last_skip}`` or its error;
    3. :func:`_require_agreed_save` -- the second gather turns a rank-local failure
       into an agreed, named failure, and the group must agree on the step counter,
       the run identity and the skip decision (the fp16 *scale* may differ per rank
       by design);
    4. :func:`_publish_manifest_agreed` -- only then does rank 0 publish the
       manifest, and the third gather releases the group only once the manifest
       exists (or the publish failure is agreed).
    """

    rank = _resolve_rank(None, group)
    world_size = _resolve_world_size(None, group)
    created = _create_run_root_agreed(root, rank=rank, group=group)
    outcome = _write_stage_outcome(
        root, runtime, step=step, identity=identity, rank=rank
    )
    outcomes = _all_gather_object(outcome, group)
    _require_agreed_save(outcomes, rank=rank, root=root, created=created)
    _publish_manifest_agreed(
        root,
        outcomes,
        rank=rank,
        world_size=world_size,
        step=step,
        created=created,
        group=group,
    )
    manifest = read_manifest(
        root,
        format_name=checkpoint_format_for_family(
            dict(outcomes[0]["topology"]).get("family")
        ),
    )
    files = {
        int(entry["rank"]): stage_path(root, int(entry["rank"]))
        for entry in manifest["ranks"]
    }
    return PpCheckpointResult(
        root=Path(root), step=int(manifest["step"]), manifest=manifest, files=files
    )


# ---------------------------------------------------------------------------
# apply and load (only after the group agreed)
# ---------------------------------------------------------------------------


def _validate_optimizer_shape(state: PpResumeState, optimizer: Any) -> None:
    """The saved param groups must describe *this* optimizer's parameter count."""

    saved_groups = list(state.optimizer.get("param_groups") or ())
    live_groups = list(optimizer.param_groups)
    saved_sizes = [len(group.get("params") or ()) for group in saved_groups]
    live_sizes = [len(group["params"]) for group in live_groups]
    if saved_sizes != live_sizes:
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            f"rank {state.rank} optimizer state has parameter groups {saved_sizes} but "
            f"this stage has {live_sizes}; the checkpoint was written for a different "
            "set of trainable parameters",
        )


def validate_stage_checkpoint(state: PpResumeState, runtime: Any) -> dict[str, Any]:
    """Pure validation of a loaded state against this rank's runtime.

    Raises before **any** mutation when the state does not fit (C3): adapter
    ownership, optimizer param-group/state shape, transaction/scaler existence
    and scale presence are checked here so a predictable mismatch never leaves
    one rank mutated while its peer refuses.  Returns the pieces the mutation
    phase needs.
    """

    optimizer = getattr(runtime, "optimizer", None)
    if optimizer is None:
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            "this runtime owns no optimizer, so a per-stage checkpoint cannot be "
            "applied to it",
        )
    owned = _validate_adapters(state.adapters, runtime.stage_module)
    _validate_optimizer_shape(state, optimizer)
    # F2: payload semantics are pure local checks too -- scale/tracker
    # legality, GradScaler keys/ranges and optimizer state content are all
    # refused here, before any rank mutates anything.
    from library.training.pp.checkpoint_validation import validate_checkpoint_content, validate_optimizer_payload

    validate_checkpoint_content(state)
    validate_optimizer_payload(state.optimizer, optimizer)
    transaction = getattr(runtime, "transaction", None)
    if transaction is None:
        if state.scaler is not None or state.scale is not None:
            raise PipelineParallelError(
                CODE_CHECKPOINT_STATE_MISMATCH,
                f"rank {state.rank} checkpoint carries fp16 state but this runtime has "
                "no optimizer transaction to restore it into",
            )
        restore_args = None
    else:
        if state.scale is None:
            raise PipelineParallelError(
                CODE_CHECKPOINT_STATE_MISMATCH,
                f"rank {state.rank} checkpoint has no fp16 scale, but this runtime owns an "
                "optimizer transaction",
            )
        scaler = getattr(transaction, "scaler", None)
        if scaler is not None:
            if state.scaler is None:
                raise PipelineParallelError(
                    CODE_CHECKPOINT_STATE_MISMATCH,
                    f"rank {state.rank} is the stage that owns the GradScaler, but the "
                    "checkpoint has no scaler state",
                )
        elif state.scaler is not None:
            raise PipelineParallelError(
                CODE_CHECKPOINT_STATE_MISMATCH,
                f"rank {state.rank} checkpoint carries a GradScaler state, but this stage "
                "does not own the loss scale",
            )
        restore_args = {
            "scale": state.scale,
            "growth_tracker": state.growth_tracker,
            "last_skip": state.last_skip,
            "where": f"pp_checkpoint_rank{state.rank}",
        }
    return {"owned": owned, "optimizer": optimizer, "restore_args": restore_args}


def apply_stage_checkpoint(state: PpResumeState, runtime: Any) -> PpResumeState:
    """Load a validated checkpoint into this rank's runtime.

    Runs :func:`validate_stage_checkpoint` first; only when every validation
    passes does the mutation phase start, so a refused apply leaves this rank's
    adapter/optimizer/scaler state untouched.
    """

    plan = validate_stage_checkpoint(state, runtime)
    # ---- mutation phase (all validations passed) ----------------------------
    for name, module in plan["owned"].items():
        module.load_state_dict(state.adapters[name], strict=True)
    optimizer = plan["optimizer"]
    try:
        optimizer.load_state_dict(state.optimizer)
    except (ValueError, KeyError, RuntimeError) as exc:
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            f"rank {state.rank} optimizer state does not load: {exc}",
        ) from exc
    restore_args = plan["restore_args"]
    if restore_args is not None:
        transaction = getattr(runtime, "transaction", None)
        scaler = getattr(transaction, "scaler", None)
        if scaler is not None:
            try:
                scaler.load_state_dict(state.scaler)
            except (ValueError, KeyError, RuntimeError) as exc:
                raise PipelineParallelError(
                    CODE_CHECKPOINT_STATE_MISMATCH,
                    f"rank {state.rank} GradScaler state does not load: {exc}",
                ) from exc
        transaction.restore(**restore_args)
    return state


def load_distributed_checkpoint(
    root: str | Path,
    runtime: Any,
    *,
    identity: Mapping[str, str] | None = None,
    group: Any = None,
) -> PpResumeState:
    """Restore every rank, agreeing on validity before any rank is released.

    The negotiation shape is the one ``load_distributed_checkpoint`` in
    ``scripts/experiments/anima_parallel/checkpoint.py`` established and this
    module keeps: every rank gathers ``{"step", "error"}`` first, and only a group
    with zero errors, one step counter and one skip decision is allowed to apply
    anything.  Applying on one rank while its peer refuses leaves that peer inside
    the schedule waiting for a collective that will never come.
    """

    rank = _resolve_rank(None, group)
    outcome = _agreement_payload(rank=rank)
    state: PpResumeState | None = None
    try:
        state = read_stage_checkpoint(root, runtime, identity=identity, group=group)
        outcome.update(
            step=state.step,
            sha256=content_hash_file(state.path),
            identity=dict(state.identity),
            topology=dict(state.topology),
            topology_digest=topology_digest(state.topology),
            scale=state.scale,
            growth_tracker=state.growth_tracker,
            last_skip=state.last_skip,
        )
    except Exception as exc:  # negotiated below, never raised rank-locally
        outcome["error"] = f"rank {rank}: {type(exc).__name__}: {exc}"
        outcome["code"] = getattr(exc, "code", None)
    outcomes = _all_gather_object(outcome, group)
    _raise_agreed(outcomes, context="PP checkpoint load")
    steps = {entry["step"] for entry in outcomes}
    if len(steps) != 1:
        raise PipelineParallelError(
            CODE_CHECKPOINT_STEP_MISMATCH,
            "the ranks loaded different step counters "
            f"({sorted(str(value) for value in steps)}); the pipeline cannot resume "
            "at two different steps",
        )
    skips = {entry["last_skip"] for entry in outcomes}
    if len(skips) != 1:
        # The fp16 *scale* may legitimately differ per rank; the skip decision may
        # not, or one half of the model resumes on a different schedule.
        raise PipelineParallelError(
            CODE_SCALER_DECISION_ASYMMETRY,
            "the ranks loaded different GradScaler skip decisions "
            f"({sorted(str(value) for value in skips)}); one half of the model may "
            "have stepped while the other did not",
        )
    assert state is not None  # guaranteed by the zero-error agreement above
    # C3, phase 1: pure local validation (no mutation, no collective inside the
    # try) -- every rank reports its verdict, and only an all-rank agreement
    # releases the apply phase.  Predictable mismatches (adapter ownership,
    # optimizer shape, transaction/scaler/scale existence) refuse here on both
    # ranks with state untouched.
    validate_error = None
    validate_code = None
    try:
        validate_stage_checkpoint(state, runtime)
    except Exception as exc:  # noqa: BLE001 - negotiated below
        validate_code = getattr(exc, "code", None)
        validate_error = f"rank {rank}: {type(exc).__name__}: {exc}"
    _raise_agreed(
        _all_gather_object(
            {"rank": rank, "error": validate_error, "code": validate_code}, group
        ),
        context="PP checkpoint validate",
    )
    # C3, phase 2: the mutation phase is itself coordinated.  A rank-local
    # failure here (payload that loads differently on this process) must abort
    # both ranks instead of leaving one applied and the peer waiting.
    apply_error = None
    apply_code = None
    try:
        apply_stage_checkpoint(state, runtime)
    except Exception as exc:  # noqa: BLE001 - negotiated below
        apply_code = getattr(exc, "code", None)
        apply_error = f"rank {rank}: {type(exc).__name__}: {exc}"
    _raise_agreed(
        _all_gather_object(
            {"rank": rank, "error": apply_error, "code": apply_code}, group
        ),
        context="PP checkpoint apply",
    )
    # C3, phase 3: the policy collective runs only after both ranks applied;
    # every rank reaches it, so its gather schema is always symmetric.
    _restore_fp16_policy(state, runtime, group=group)
    return state


def _restore_fp16_policy(state: PpResumeState, runtime: Any, *, group: Any) -> None:
    """Adopt the checkpointed growth policy on every stage, not just the last.

    The policy is broadcast at *construction* from the last stage's scaler;
    after a load the last stage's scaler carries the checkpointed policy, so a
    non-last stage that kept its constructor guess would grow the scale at a
    different step than the stage that owns the loss.  One collective re-derives
    the policy from the restored scaler and every stage adopts it.
    """

    import torch.distributed as dist

    transaction = getattr(runtime, "transaction", None)
    if transaction is None:
        return
    scaler = getattr(transaction, "scaler", None)
    is_last = bool(getattr(transaction, "is_last", False))
    restore_policy = getattr(transaction, "restore_policy", None)
    if not callable(restore_policy):
        return
    payload: dict[str, Any] = {"policy": None}
    if is_last and scaler is not None:
        payload["policy"] = {
            "growth_factor": float(scaler.get_growth_factor()),
            "backoff_factor": float(scaler.get_backoff_factor()),
            "growth_interval": int(scaler.get_growth_interval()),
        }
    if dist.is_available() and dist.is_initialized():
        gathered: list[Any] = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, payload, group=group)
        policies = [entry.get("policy") for entry in gathered if entry]
        restored = next((item for item in policies if item), None)
        if restored is None:
            # No stage owns a GradScaler (non-fp16 or probe runtimes): there is
            # no policy to restore.
            return
        if len({json.dumps(item, sort_keys=True) for item in policies if item}) != 1:
            raise PipelineParallelError(
                CODE_SCALER_DECISION_ASYMMETRY,
                "the stages restored different GradScaler policies "
                f"({policies}); the checkpoint's policy must be one "
                "pipeline-wide fact",
            )
    else:
        restored = payload["policy"]
    if restored is None:
        return
    transaction.restore_policy(
        growth_factor=restored["growth_factor"],
        backoff_factor=restored["backoff_factor"],
        growth_interval=restored["growth_interval"],
        where=f"pp_checkpoint_rank{state.rank}",
    )


__all__ = [
    "PpCheckpointResult",
    "apply_stage_checkpoint",
    "load_distributed_checkpoint",
    "save_distributed_checkpoint",
]
