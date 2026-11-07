"""Versioned resume metadata and pre-resume validation (contract §8 P1-1..P1-3).

A distributed state directory is only *exactly* resumable when the topology that
produced it can be reproduced. The bundle written next to ``train_state.json``
records the facts that decision needs:

* ``version`` – bundle schema version (bumped when the meaning changes);
* ``world_size`` / ``backend`` – how many workers ran and over which backend;
* ``rank_devices`` – the real mapping rank → device, so a topology change is
  visible instead of inferred;
* ``rng`` – which RNG streams were captured (and whether they are present);
* ``loader_cursor`` – where the sampler stood, so a resume does not silently
  replay or skip samples.

Decisions are explicit and named, never "probably fine":

* no bundle → ``legacy``: resume still allowed, but **not** advertised as exact;
* version mismatch / unreadable → rejected;
* ``rank_devices`` describing only part of ``world_size`` (or more ranks than it
  declares) → **rejected** as ``multigpu_resume_bundle_invalid``, not degraded:
  the publish path builds that mapping from every rank's own report
  (``checkpoints.publish_state_with_bundle`` reads ``rank<N>.json`` for every N),
  so a partial mapping proves the state was *not* produced by a save of this
  size. "Degraded" is reserved for a **known** topology change (different
  world_size/backend), never for an inconsistent record;
* different ``world_size`` → rejected by default; a caller may opt into a named
  *degraded* resume, which is recorded as such instead of being called exact;
* same ``world_size`` **and** the required streams present → exact.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

BUNDLE_FILENAME = "multigpu_resume_bundle.json"
BUNDLE_VERSION = 1

CODE_BUNDLE_INVALID = "multigpu_resume_bundle_invalid"
CODE_TOPOLOGY_MISMATCH = "multigpu_resume_topology_mismatch"
CODE_LEGACY_NOT_EXACT = "multigpu_resume_legacy_not_exact"
CODE_TOPOLOGY_DEGRADED = "multigpu_resume_topology_degraded"
CODE_CURSOR_MISMATCH = "multigpu_resume_cursor_mismatch"
CODE_CURSOR_UNKNOWN = "multigpu_resume_cursor_unknown"
CODE_SHARED_RANK_KEY = "multigpu_resume_shared_key"
#: The RNG files the bundle claims to have verified are gone/unreadable **now**.
CODE_RNG_EVIDENCE_MISSING_AT_RESUME = "multigpu_resume_rng_evidence_missing"
#: The files are readable but the streams they carry no longer match the bundle.
CODE_RNG_STREAMS_DEGRADED_AT_RESUME = "multigpu_resume_rng_streams_degraded"

REQUIRED_STREAMS = ("python", "torch", "cuda")


@dataclass
class ResumeDecision:
    """Outcome of validating a state directory before resuming.

    ``code`` is the primary (most actionable) problem; ``codes`` keeps **every**
    named problem that was found, so a RNG problem is never hidden behind a cursor
    problem or the other way round.
    """

    exact: bool
    code: Optional[str] = None
    message: str = ""
    bundle: Optional[dict[str, Any]] = None
    codes: tuple[str, ...] = ()

    def raise_if_rejected(self) -> None:
        if self.code in (CODE_BUNDLE_INVALID, CODE_TOPOLOGY_MISMATCH):
            raise ResumeBundleError(self.code, self.message)


class ResumeBundleError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def rank_local_keys(
    keys_by_rank: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    """Per-rank **exclusive** file/cursor keys.

    A key may belong to exactly one rank: a checkpoint artifact that two ranks
    both claim (or one rank claiming a shared artifact) makes the per-rank resume
    story ambiguous, so it is refused here instead of being written into the
    bundle.
    """
    out: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    for rank, keys in keys_by_rank.items():
        name = str(rank)
        out[name] = sorted({str(key) for key in keys})
        for key in out[name]:
            if key in seen:
                raise ResumeBundleError(
                    CODE_SHARED_RANK_KEY,
                    f"key {key!r} claimed by both rank {seen[key]} and rank {name}",
                )
            seen[key] = name
    return out


def rng_evidence_from_disk(
    state_dir: str, *, world_size: int = 1, uses_cuda: bool = False
) -> dict[str, Any]:
    """Derive the RNG booleans from what actually landed on disk.

    Accelerate writes one ``random_states_<rank>.pkl`` per rank. Those files are
    the evidence: if they are missing, "RNG was saved" is not a fact, and a resume
    must not claim it.
    """
    found: list[str] = []
    try:
        names = sorted(os.listdir(state_dir))
    except OSError:
        names = []
    for name in names:
        if name.startswith("random_states") and name.endswith(".pkl"):
            path = os.path.join(state_dir, name)
            try:
                if os.path.getsize(path) > 0:
                    found.append(name)
            except OSError:
                continue
    complete = len(found) >= max(1, int(world_size))
    return {
        "python": complete,
        "torch": complete,
        "cuda": bool(complete and uses_cuda),
        "files": found,
        "ranks_covered": len(found),
    }


def bundle_path(state_dir: str) -> str:
    return os.path.join(state_dir, BUNDLE_FILENAME)


def build_bundle(
    accelerator,
    *,
    backend: str,
    loader_cursor: Mapping[str, Any] | None = None,
    rng: Mapping[str, bool] | None = None,
    global_step: Optional[int] = None,
    extra: Mapping[str, Any] | None = None,
    evidence_dir: Optional[str] = None,
    rank_local: Mapping[str, Iterable[str]] | None = None,
    uses_cuda: Optional[bool] = None,
) -> dict[str, Any]:
    """Describe this run's topology precisely enough to resume it exactly.

    ``evidence_dir`` makes the RNG booleans **observed** instead of asserted: the
    flags come from the ``random_states_*.pkl`` files that were actually written.
    """
    world_size = int(getattr(accelerator, "num_processes", 1) or 1)
    process_index = int(getattr(accelerator, "process_index", 0) or 0)
    device = getattr(accelerator, "device", None)
    cuda_topology = (
        uses_cuda
        if uses_cuda is not None
        else str(getattr(device, "type", "")) == "cuda"
    )
    if evidence_dir is not None:
        from library.training.multigpu import rng_evidence

        # Same reader as the resume-time check (``guard_resume``): one schema, one
        # vocabulary. Never write a bundle that claims RNG was saved when it was not.
        streams = rng_evidence.observe_state_dir(
            evidence_dir, world_size=world_size, uses_cuda=cuda_topology
        )
    else:
        streams = dict(rng) if rng is not None else {}
    payload: dict[str, Any] = {
        "version": BUNDLE_VERSION,
        "world_size": world_size,
        "backend": str(backend),
        "local_rank": process_index,
        "rank_devices": {
            str(process_index): {
                "type": str(getattr(device, "type", "unknown")),
                "index": getattr(device, "index", None),
            }
        },
        "rng": streams,
        "loader_cursor": dict(loader_cursor or {}),
        "global_step": None if global_step is None else int(global_step),
    }
    if rank_local is not None:
        payload["rank_local_keys"] = rank_local_keys(rank_local)
    if extra:
        payload["extra"] = dict(extra)
    return payload


def merge_rank_bundles(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold per-rank bundles into the single file rank 0 publishes."""
    if not bundles:
        raise ResumeBundleError(CODE_BUNDLE_INVALID, "no rank bundles to merge")
    merged = dict(bundles[0])
    devices: dict[str, Any] = {}
    for bundle in bundles:
        devices.update(bundle.get("rank_devices", {}))
    merged["rank_devices"] = dict(
        sorted(devices.items(), key=lambda item: int(item[0]))
    )
    merged["world_size"] = max(int(b.get("world_size", 1)) for b in bundles)
    return merged


def write_bundle(bundle: Mapping[str, Any], state_dir: str, *, is_main: bool = True) -> Optional[str]:
    """Atomically write the bundle (rank 0 only, tmp file + ``os.replace``)."""
    if not is_main:
        return None
    target = bundle_path(state_dir)
    tmp = target + ".tmp"
    os.makedirs(state_dir, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(dict(bundle), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(tmp, target)
    return target


def read_bundle(state_dir: str) -> Optional[dict[str, Any]]:
    target = bundle_path(state_dir)
    if not os.path.exists(target):
        return None
    try:
        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ResumeBundleError(CODE_BUNDLE_INVALID, f"unreadable bundle: {exc}")
    if not isinstance(data, dict):
        raise ResumeBundleError(CODE_BUNDLE_INVALID, "bundle is not an object")
    return data


def validate_bundle(
    bundle: Optional[Mapping[str, Any]],
    *,
    world_size: int,
    backend: Optional[str] = None,
    allow_degraded: bool = False,
    resume_cursor: Optional[Mapping[str, Any]] = None,
    rank: Optional[int] = None,
    live_rng: Optional[Mapping[str, Any]] = None,
) -> ResumeDecision:
    """Decide whether this state dir can be resumed exactly, degraded or not.

    ``resume_cursor`` compares against the bundle's **top-level**
    ``loader_cursor`` (which the publish path fills with rank 0's cursor). Pass
    ``rank`` to compare against *that* rank's own cursor from
    ``extra.rank_cursors`` instead — the form the training session uses, because
    every rank resumes from its own position.

    ``live_rng`` is the RNG evidence **re-read at resume time** (see
    :meth:`RngEvidence.observe_state_dir`); ``guard_resume`` passes it. The bundle
    records what the files looked like when the state was *written*, and accelerate
    silently swallows a failed RNG restore, so the recorded flags alone cannot
    prove that the streams will really be loaded.
    """
    if bundle is None:
        return ResumeDecision(
            exact=False,
            code=CODE_LEGACY_NOT_EXACT,
            message=(
                "no multigpu resume bundle: this state predates versioned resume, "
                "so the topology/RNG/loader cursor cannot be verified. Resume is "
                "allowed but must not be reported as an exact resume"
            ),
        )
    version = bundle.get("version")
    if version != BUNDLE_VERSION:
        return ResumeDecision(
            exact=False,
            code=CODE_BUNDLE_INVALID,
            message=f"bundle version {version!r} != supported {BUNDLE_VERSION}",
            bundle=dict(bundle),
        )
    saved_world = int(bundle.get("world_size", 0) or 0)
    if saved_world != int(world_size):
        if allow_degraded:
            return ResumeDecision(
                exact=False,
                code=CODE_TOPOLOGY_DEGRADED,
                message=(
                    f"world_size changed {saved_world} -> {world_size}; resuming as "
                    "an explicitly degraded run (shard layout differs, RNG streams "
                    "are not comparable)"
                ),
                bundle=dict(bundle),
            )
        return ResumeDecision(
            exact=False,
            code=CODE_TOPOLOGY_MISMATCH,
            message=(
                f"world_size changed {saved_world} -> {world_size}; exact resume is "
                "impossible. Re-run with the original worker count or pass the "
                "explicit degraded-resume opt-in"
            ),
            bundle=dict(bundle),
        )
    if backend and bundle.get("backend") and str(bundle["backend"]) != str(backend):
        return ResumeDecision(
            exact=False,
            code=CODE_TOPOLOGY_DEGRADED if allow_degraded else CODE_TOPOLOGY_MISMATCH,
            message=(
                f"backend changed {bundle['backend']} -> {backend}; collective "
                "ordering differs, so this is not an exact resume"
            ),
            bundle=dict(bundle),
        )
    # --- device mapping ------------------------------------------------------
    devices = dict(bundle.get("rank_devices") or {})
    expected_ranks = [str(index) for index in range(int(world_size))]
    missing_ranks = [name for name in expected_ranks if name not in devices]
    unexpected_ranks = sorted(
        (name for name in devices if name not in expected_ranks),
        key=lambda name: (len(name), name),
    )
    if missing_ranks or unexpected_ranks:
        # A bundle that only describes some workers cannot place the remaining
        # ranks, and one that describes extra ranks was written by a different
        # world size than it declares. Both are inconsistent records, not a
        # topology change, so they are refused instead of downgraded.
        problems = []
        if missing_ranks:
            problems.append(f"missing rank(s) {missing_ranks}")
        if unexpected_ranks:
            problems.append(f"unexpected rank(s) {unexpected_ranks}")
        return ResumeDecision(
            exact=False,
            code=CODE_BUNDLE_INVALID,
            message=(
                "rank_devices " + " and ".join(problems)
                + f" for world_size={world_size}: the state was not produced by "
                "this topology (every rank reports its own device before rank 0 "
                "publishes the bundle)"
            ),
            bundle=dict(bundle),
        )
    uses_cuda = any(
        str((device or {}).get("type", "")) == "cuda" for device in devices.values()
    )

    # --- loader cursor (checked **before** the RNG streams) ------------------
    # The RNG verdict is a generic "not exact"; a cursor drift is the specific
    # "the data order moved". Checking RNG first returned early and swallowed the
    # drift, so a resume that would replay the wrong samples could pass as a mere
    # degradation (t3 F-T3-3; combined with the off-by-one of F-T3-2 this is how a
    # resumed run pinned the wrong noise and still looked green).
    cursor = dict(bundle.get("loader_cursor") or {})
    position = cursor.get("sampler_position")
    cursor_problem: Optional[tuple[str, str]] = None
    if not isinstance(position, int) or isinstance(position, bool):
        cursor_problem = (
            CODE_CURSOR_UNKNOWN,
            "loader cursor has no integer sampler_position "
            f"(got {position!r}): the sample order cannot be reproduced, so this "
            "is not an exact resume",
        )
    elif resume_cursor is not None:
        expected = dict(resume_cursor)
        if rank is None:
            saved_cursor = dict(bundle.get("loader_cursor") or {})
        else:
            from library.training.multigpu import cursor as cursor_mod

            raw_rank_cursor = _rank_cursor(bundle, int(rank))
            if not cursor_mod.is_real_cursor(raw_rank_cursor):
                cursor_problem = (
                    CODE_CURSOR_UNKNOWN,
                    f"rank {int(rank)} has no real per-rank cursor in the bundle "
                    f"(got {raw_rank_cursor!r}): the sample order cannot be "
                    "reproduced, so this is not an exact resume",
                )
            else:
                saved_cursor = cursor_mod.comparable(raw_rank_cursor)
                expected = cursor_mod.comparable(expected)
        if cursor_problem is None and saved_cursor != expected:
            cursor_problem = (
                CODE_CURSOR_MISMATCH,
                ("saved per-rank loader cursor " if rank is not None else "saved loader cursor ")
                + f"{saved_cursor} != this run's resume start {expected}: the data "
                "order would differ, so this is not an exact resume",
            )
    # --- problems, collected independently -----------------------------------
    # Every check appends its own named problem; nothing returns early. A RNG gap
    # and a cursor drift are different failures and both must be visible (t3
    # F-T3-3 fixed the early return; F-T10-1 needs the same discipline for the
    # resume-time observation).
    problems: list[tuple[str, str]] = []
    if cursor_problem is not None:
        problems.append(cursor_problem)

    recorded = dict(bundle.get("rng") or {})
    required = ("python", "numpy", "torch") + (("cuda",) if uses_cuda else ())
    if recorded.get("source") != "loaded_random_states":
        problems.append(
            (
                CODE_TOPOLOGY_DEGRADED,
                "RNG streams were not verified by reading per-rank "
                "random_states_*.pkl when the state was written (source="
                f"{recorded.get('source', 'unset')!r}"
                + (f", error={recorded['error']}" if recorded.get("error") else "")
                + "): exact resume cannot be claimed",
            )
        )
    else:
        missing_recorded = [
            name for name in required if not bool(recorded.get(name, False))
        ]
        if missing_recorded:
            problems.append(
                (
                    CODE_TOPOLOGY_DEGRADED,
                    "missing RNG streams at save time "
                    + ", ".join(missing_recorded)
                    + ": data order will differ from the original run",
                )
            )

    if live_rng is not None:
        live = dict(live_rng)
        if live.get("source") != "loaded_random_states":
            problems.append(
                (
                    CODE_RNG_EVIDENCE_MISSING_AT_RESUME,
                    "the RNG evidence this state claims to verify is not readable "
                    f"now (source={live.get('source', 'unset')!r}"
                    + (f", error={live['error']}" if live.get("error") else "")
                    + (f": {live['detail']}" if live.get("detail") else "")
                    + "); accelerate swallows a failed RNG restore, so resuming "
                    "would continue on an unrelated RNG stream",
                )
            )
        else:
            missing_now = [name for name in required if not bool(live.get(name, False))]
            no_longer_backed = [
                name
                for name in required
                if bool(recorded.get(name, False)) and not bool(live.get(name, False))
            ]
            if missing_now or no_longer_backed:
                detail = []
                if missing_now:
                    detail.append("missing now: " + ", ".join(missing_now))
                if no_longer_backed:
                    detail.append(
                        "recorded in the bundle but gone from disk: "
                        + ", ".join(no_longer_backed)
                    )
                problems.append(
                    (
                        CODE_RNG_STREAMS_DEGRADED_AT_RESUME,
                        "RNG streams no longer match this state ("
                        + "; ".join(detail)
                        + "); the resume would run on a different RNG stream",
                    )
                )

    if problems:
        return ResumeDecision(
            exact=False,
            code=problems[0][0],
            message="; ".join(text for _, text in problems),
            bundle=dict(bundle),
            codes=tuple(dict.fromkeys(code for code, _ in problems)),
        )
    return ResumeDecision(
        exact=True,
        message="topology, RNG streams (re-read at resume time) and loader cursor match",
        bundle=dict(bundle),
    )


def _rank_cursor(bundle: Mapping[str, Any], rank: int) -> dict[str, Any]:
    """This rank's own cursor from the bundle's per-rank section."""
    cursors = dict((bundle.get("extra") or {}).get("rank_cursors") or {})
    return dict(cursors.get(str(int(rank))) or {})


def guard_resume(
    state_dir: str,
    *,
    world_size: int,
    backend: Optional[str] = None,
    allow_degraded: bool = False,
    resume_cursor: Optional[Mapping[str, Any]] = None,
    rank: Optional[int] = None,
    verify_rng_files: bool = True,
) -> ResumeDecision:
    """Validate a state dir before resuming; log the named outcome.

    The RNG streams are observed **again here**, on the real directory, because the
    bundle only records what was true when the state was written: accelerate
    swallows a failed RNG restore, so a half-broken state would otherwise resume on
    an unrelated stream while the bundle still claims ``exact``. This is a few
    local pickle reads - no collective, no bulk IO.
    """
    from library.training.multigpu import rng_evidence

    bundle = read_bundle(state_dir)
    live_rng: Optional[dict[str, Any]] = None
    if verify_rng_files and bundle is not None:
        devices = dict(bundle.get("rank_devices") or {})
        uses_cuda = any(
            str((device or {}).get("type", "")) == "cuda" for device in devices.values()
        )
        live_rng = rng_evidence.observe_state_dir(
            state_dir, world_size=int(world_size), uses_cuda=uses_cuda
        )
    decision = validate_bundle(
        bundle,
        world_size=world_size,
        backend=backend,
        allow_degraded=allow_degraded,
        resume_cursor=resume_cursor,
        rank=rank,
        live_rng=live_rng,
    )
    if decision.exact:
        logger.info("resume bundle verified for exact resume: %s", state_dir)
    elif decision.code in (CODE_BUNDLE_INVALID, CODE_TOPOLOGY_MISMATCH):
        logger.error("resume refused (%s): %s", decision.code, decision.message)
    else:
        logger.warning(
            "resume is not exact (%s): %s",
            ", ".join(decision.codes) or decision.code,
            decision.message,
        )
    return decision
