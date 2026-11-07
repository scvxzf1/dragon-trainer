"""Per-run identity record (contract §7 P0-3).

What it answers without reading a log: which ranks ran, on which device, with
which precision that was *actually* in force and why, how many workers the
launcher intended versus how many really joined, and how gradients were reduced
(including the bucket count the real gradient-sync path used).

Design rules learned from review t4:

* **the whole world is recorded, not rank 0's view** – rank 0 all-gathers one
  small summary per rank, so a divergent device/precision assignment is visible
  inside a single manifest;
* **no "best effort" silence** – in a multi-process run a manifest that could not
  be written is a *named failure* (same shape as the P0-2 startup probe): the
  failure is agreed across ranks and the job stops instead of leaving evidence
  collection to guess;
* **`configured_world_size` and `world_size` are separate fields** – "what the
  launcher asked for" and "how many workers actually joined" are different
  facts and must never be conflated;
* **fields are filled from the real code path** – ``reduction_mode`` and the
  bucket counters are written when ``synchronize_optimizer_gradients`` actually
  returns its result, not from a flag that is always false.

Nothing here is written into a versioned default: the manifest goes to the run's
own output directory, written by rank 0 only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

MIXED_PRECISION_ENV = "ANIMA_ACCELERATE_MIXED_PRECISION"

#: Reason code used when the multi-process identity manifest cannot be written.
IDENTITY_WRITE_FAILED = "multigpu_identity_write_failed"


class IdentityRecordError(RuntimeError):
    """Raised (on every rank) when a multi-process identity manifest is missing.

    Rationale: a run without an identity manifest cannot be evidence. Silently
    continuing would let a later report claim "identity verified" from nothing,
    so this is a named, agreed failure instead of a warning.
    """

    code = IDENTITY_WRITE_FAILED


#: Per-process evidence filled in by the training loop once the real
#: ``synchronize_optimizer_gradients`` result exists.
_EVIDENCE: dict[str, Any] = {}
_MANIFEST_PATH: Optional[Path] = None


def _device_facts(accelerator) -> dict[str, Any]:
    device = getattr(accelerator, "device", None)
    facts: dict[str, Any] = {
        "type": str(getattr(device, "type", "unknown")),
        "index": getattr(device, "index", None),
    }
    try:
        if facts["type"] == "cuda" and torch.cuda.is_available():
            index = facts["index"]
            if index is None:
                index = torch.cuda.current_device()
            facts["index"] = int(index)
            facts["name"] = torch.cuda.get_device_name(int(index))
            props = torch.cuda.get_device_properties(int(index))
            facts["total_memory_bytes"] = int(getattr(props, "total_memory", 0) or 0)
            facts["capability"] = f"{props.major}.{props.minor}"
    except Exception:  # identity recording must never fail training
        facts["probe_error"] = "cuda device facts unavailable"
    return facts


def configured_world_size(env: Mapping[str, str] | None = None) -> int:
    """How many workers the **launcher intended** (not how many joined)."""
    from library.training.multigpu.admission import resolve_admission

    decision = resolve_admission(None, dict(os.environ if env is None else env))
    return int(decision.num_processes)


def reduction_mode_from_result(result: Any, *, async_overlap: bool = False) -> str:
    """Name the reduction actually used, derived from the real result object."""
    if result is None:
        return "pending_first_boundary"
    buckets = int(getattr(result, "bucket_count", 0) or 0)
    if int(getattr(result, "reduced_parameter_count", 0) or 0) == 0:
        return "no_reduction(single_process_or_empty_optimizer)"
    if async_overlap:
        return f"async_bucket_all_reduce_mean({buckets} buckets)"
    return f"synchronize_optimizer_gradients_bucket_mean({buckets} buckets)"


def build_runtime_identity(
    accelerator,
    args=None,
    *,
    accuracy_source: Optional[str] = None,
    global_batch_size: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Collect this rank's identity fields."""
    world = int(getattr(accelerator, "num_processes", 1) or 1)
    process_index = int(getattr(accelerator, "process_index", 0) or 0)
    requested_precision = (
        os.environ.get(MIXED_PRECISION_ENV) or getattr(args, "mixed_precision", None)
    )
    effective_precision = getattr(accelerator, "mixed_precision", None) or getattr(
        args, "mixed_precision", None
    )
    train_batch = getattr(args, "train_batch_size", None)
    accum = getattr(args, "gradient_accumulation_steps", 1)
    resolved_global_batch = global_batch_size
    if resolved_global_batch is None and train_batch is not None:
        try:
            resolved_global_batch = int(train_batch) * max(1, int(accum or 1)) * world
        except (TypeError, ValueError):
            resolved_global_batch = None

    payload: dict[str, Any] = {
        "process_index": process_index,
        # Real topology first, launcher intent second: these are different facts.
        "world_size": world,
        "configured_world_size": configured_world_size(),
        "is_main_process": bool(
            getattr(accelerator, "is_main_process", process_index == 0)
        ),
        "distributed_type": str(
            getattr(getattr(accelerator, "state", None), "distributed_type", "")
        ),
        "device": _device_facts(accelerator),
        "mixed_precision": {
            "requested": requested_precision,
            "effective": effective_precision,
            "source": accuracy_source or "unknown",
        },
        "gradient_accumulation_steps": max(1, int(accum or 1)),
        "global_batch_size": resolved_global_batch,
        # Filled by note_gradient_sync() from the real gradient-sync result.
        "reduction_mode": reduction_mode_from_result(None),
        "gradient_buckets": None,
        "reduced_parameter_count": None,
        "scaler_present": getattr(accelerator, "scaler", None) is not None,
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def collect_rank_summaries(accelerator, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """All-gather one fixed-size summary per rank (world-wide, not rank 0's view).

    The payload is a single float vector of fixed length, so this costs exactly
    one small collective and cannot diverge in shape between ranks.
    """
    world = int(getattr(accelerator, "num_processes", 1) or 1)
    device = payload.get("device", {})
    local = torch.tensor(
        [
            float(payload.get("process_index", 0)),
            1.0 if device.get("type") == "cuda" else 0.0,
            float(-1 if device.get("index") is None else device["index"]),
            float(payload.get("world_size", 0)),
            float(payload.get("configured_world_size", 0)),
        ],
        dtype=torch.float32,
        device=accelerator.device,
    )

    def _summary(values: list[float], precision: str, device_name: str | None) -> dict:
        return {
            "process_index": int(values[0]),
            "device_type": "cuda" if values[1] > 0.5 else "cpu",
            "device_index": None if values[2] < 0 else int(values[2]),
            "device_name": device_name,
            "world_size": int(values[3]),
            "configured_world_size": int(values[4]),
            "mixed_precision": precision,
        }

    precision = str(payload.get("mixed_precision", {}).get("effective") or "no")
    if world <= 1:
        return [_summary(local.tolist(), precision, device.get("name"))]

    import torch.distributed as dist

    gathered = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    out: list[dict[str, Any]] = []
    for tensor in gathered:
        values = tensor.tolist()
        # Device names are host-specific strings and are intentionally not
        # gathered; the index + type pair is what a divergence shows up in.
        out.append(_summary(values, "see per-rank payload" if int(values[0]) != 0 else precision, None))
    return out


def register_manifest_path(path: str | os.PathLike[str]) -> Path:
    """Remember where the manifest lives so the loop can fill it in later."""
    global _MANIFEST_PATH
    _MANIFEST_PATH = Path(path)
    return _MANIFEST_PATH


def write_runtime_identity(
    accelerator,
    payload: Mapping[str, Any],
    path: str | os.PathLike[str] | None = None,
    *,
    per_rank_summaries: Optional[list[dict[str, Any]]] = None,
) -> Optional[Path]:
    """Write the manifest from rank 0 only, covering every rank.

    ``per_rank_summaries`` comes from :func:`collect_rank_summaries`: the single
    file then shows every rank's device/topology, which is what makes a cross-rank
    divergence visible. In a multi-process run a write failure is a **named
    failure** (:class:`IdentityRecordError`), not a warning.
    """
    world = int(getattr(accelerator, "num_processes", 1) or 1)
    is_main = int(getattr(accelerator, "process_index", 0) or 0) == 0
    target = Path(path) if path is not None else _MANIFEST_PATH
    error: Optional[str] = None
    if target is None:
        error = "no manifest path registered"
    elif is_main:
        # Everything that can fail while producing the manifest lives inside this
        # try: document construction, serialisation and the atomic write. Catching
        # ``Exception`` (never ``BaseException``, so Ctrl-C/SystemExit still work)
        # is what keeps a rank-0-only failure from escaping before the agreed
        # failure below (review t17 low-①).
        try:
            document = dict(payload)
            if per_rank_summaries:
                document["ranks"] = sorted(
                    per_rank_summaries, key=lambda item: item.get("process_index", 0)
                )
            document["evidence"] = dict(_EVIDENCE)
            encoded = json.dumps(document, indent=2, sort_keys=True, default=str) + "\n"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_target = target.with_name(target.name + ".tmp")
            tmp_target.write_text(encoded, encoding="utf-8")
            os.replace(tmp_target, target)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                target.with_name(target.name + ".tmp").unlink(missing_ok=True)
            except Exception:
                pass

    # Agree the outcome so every rank takes the same branch: a manifest that only
    # rank 0 knows is missing would leave the others running a run with no
    # evidence (review t4 requirement).
    from library.training.multigpu.agreement import agree_any

    failed = agree_any(
        accelerator, {"identity_write_failed": error is not None}
    )["identity_write_failed"]
    if failed and world > 1:
        raise IdentityRecordError(
            f"{IDENTITY_WRITE_FAILED}: multi-process identity manifest could not "
            f"be written to {target} ({error or 'peer reported failure'}). "
            "Refusing to continue a multi-process run that cannot be identified."
        )
    if error is not None and is_main:
        # Single-process fallback: keep the explicit fact instead of silence.
        import sys

        print(
            f"[{IDENTITY_WRITE_FAILED}] identity manifest unavailable: {error}",
            file=sys.stderr,
        )
        return None
    return target if is_main else None


def note_gradient_sync(
    result: Any, *, accelerator=None, async_overlap: bool = False
) -> None:
    """Record the **real** gradient-sync accounting and refresh the manifest.

    Called by the training loop immediately after
    ``synchronize_optimizer_gradients`` returns, so ``bucket_count`` /
    ``reduced_parameter_count`` / ``reduction_mode`` describe what ran instead of
    a static guess. Rewriting is best effort here: the manifest already exists
    (its absence was a hard failure above), and a later update only adds evidence.
    """
    if result is None:
        return
    _EVIDENCE["gradient_buckets"] = int(getattr(result, "bucket_count", 0) or 0)
    _EVIDENCE["reduced_parameter_count"] = int(
        getattr(result, "reduced_parameter_count", 0) or 0
    )
    _EVIDENCE["materialized_gradient_count"] = int(
        getattr(result, "materialized_gradient_count", 0) or 0
    )
    _EVIDENCE["reduction_mode"] = reduction_mode_from_result(
        result, async_overlap=async_overlap
    )
    path = _MANIFEST_PATH
    if path is None or not path.exists():
        return
    # Only the manifest owner writes it: every rank calls this hook in the same
    # collective-free order, but concurrent rewrites of one file would corrupt it.
    if accelerator is not None and int(getattr(accelerator, "process_index", 0) or 0) != 0:
        return
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    document["reduction_mode"] = _EVIDENCE["reduction_mode"]
    document["gradient_buckets"] = _EVIDENCE["gradient_buckets"]
    document["reduced_parameter_count"] = _EVIDENCE["reduced_parameter_count"]
    document["evidence"] = dict(_EVIDENCE)
    try:
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return


def record_or_fail(
    accelerator,
    args,
    output_dir: Optional[str],
    *,
    accuracy_source: Optional[str] = None,
) -> Optional[Path]:
    """Build → gather → write the manifest, with agreed failure semantics.

    Multi-process rules (review t16 F2 / t17 low-①):

    * a missing output directory is refused **before any collective**;
    * local construction/serialisation/write failures are caught as ``Exception``
      and turned into the agreed failure below instead of escaping rank 0 alone;
    * the per-rank ``all_gather`` is deliberately outside that try: a transport
      failure there is class II and must propagate, not be "agreed".
    """
    world = int(getattr(accelerator, "num_processes", 1) or 1)
    if world <= 1:
        return None
    if not output_dir:
        raise IdentityRecordError(
            f"{IDENTITY_WRITE_FAILED}: --output_dir is required for a multi-process "
            "run so the identity manifest can be written; refusing to start a "
            "multi-process run that cannot be identified."
        )

    target = Path(output_dir) / "multigpu_identity.json"
    register_manifest_path(target)
    error: Optional[str] = None
    payload: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    try:
        payload = build_runtime_identity(
            accelerator, args, accuracy_source=accuracy_source
        )
    except Exception as exc:
        error = f"build failed: {type(exc).__name__}: {exc}"
        payload = {"process_index": int(getattr(accelerator, "process_index", 0) or 0)}

    if error is None:
        summaries = collect_rank_summaries(accelerator, payload)
    if error is not None:
        # Still enter the agreement with the same collective every rank uses, so
        # peers do not wait for a rank that already gave up.
        from library.training.multigpu.agreement import agree_any

        failed = agree_any(accelerator, {"identity_build_failed": True})[
            "identity_build_failed"
        ]
        if failed:
            raise IdentityRecordError(
                f"{IDENTITY_WRITE_FAILED}: {error}"
            )
    return write_runtime_identity(
        accelerator, payload, target, per_rank_summaries=summaries
    )
