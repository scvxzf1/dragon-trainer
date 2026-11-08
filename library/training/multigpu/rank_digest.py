"""Per-rank commit-boundary digest: the missing observation behind group R.

Group R asks whether two ranks really hold the same state at a commit boundary
(R-a: bit-identical adapter weights, R-b: the same scaler, R-c: the same
optimizer/scheduler position). A state directory produced by
``accelerator.save_state`` cannot answer that on its own: it holds **one**
``model.safetensors`` (rank 0's), one ``scaler.pt`` and one optimizer shard per
rank — nothing that a third party can compare pairwise without loading every
rank's shard and re-deriving the state.

So every rank writes a small digest of what it holds at the boundary, and rank 0
summarizes them into the resume bundle. This is **observation only**:

* it runs at the commit boundary, never on the per-step path;
* it adds no collective — each rank hashes its own tensors locally and the
  comparison is a rank-0 read of files that already landed on disk;
* it is on by default and can be switched off (`ANIMA_MULTIGPU_RANK_DIGEST=0`),
  in which case nothing is written and nothing is compared.

A digest difference is reported as a named WARNING, never as a failed run: the
existing run keeps its semantics, and the verifier decides what a mismatch means
(usually: fp16 non-determinism or a real protocol bug).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

#: Default **on**; set to ``0`` / ``false`` / ``off`` / ``no`` to disable.
ENV = "ANIMA_MULTIGPU_RANK_DIGEST"
FILE = "multigpu_rankdigest_rank{rank}.json"
SCHEMA = "multigpu-rank-digest-v1"

CODE_DIGEST_MISMATCH = "multigpu_rank_digest_mismatch"
CODE_DIGEST_MISSING = "multigpu_rank_digest_missing"

#: Fields the cross-rank comparison looks at, in reporting order.
COMPARED_FIELDS = (
    "global_step",
    "epoch",
    "adapter",
    "scaler",
    "optimizer",
    "scheduler",
    "window",
)

#: The four faces of group R, each judged on its own so a report can say *which*
#: one failed instead of "the ranks differ somewhere".
AXES = ("adapter", "scaler", "optimizer", "window")

#: Window counters are *rank-local expectations*: they describe how much work this
#: rank did, and a replay/discard is decided per rank before the boundary is agreed.
#: They are recorded for observation and never compared element-wise, otherwise a
#: legitimate rank-local loss (sum of different micro-batches) would raise a
#: mismatch code that means nothing.
WINDOW_OBSERVATION_ONLY = (
    "window_total_loss",
    "window_last_loss",
    "ema_observation",
    "window_num_steps",
)

_DISABLED = {"0", "false", "no", "off"}


def enabled(args: Any = None, env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the digest is collected (default on, explicitly switchable off)."""
    values = dict(os.environ if env is None else env)
    if str(values.get(ENV, "")).strip().lower() in _DISABLED:
        return False
    flag = getattr(args, "multigpu_rank_digest", None)
    return True if flag is None else bool(flag)


# ---------------------------------------------------------------------------
# adapter tensors
# ---------------------------------------------------------------------------


def _named_tensors(source: Any) -> list[tuple[str, Any]]:
    """``(name, tensor)`` pairs for a module, a mapping or a pair sequence."""
    if source is None:
        return []
    if isinstance(source, Mapping):
        pairs = list(source.items())
    elif isinstance(source, (list, tuple)):
        pairs = list(source)
    else:
        named = getattr(source, "named_parameters", None)
        if not callable(named):
            return []
        pairs = list(named())
    return sorted(
        ((str(name), tensor) for name, tensor in pairs),
        key=lambda pair: pair[0],
    )


def _tensor_bytes(tensor: Any) -> bytes:
    """Raw bytes of a tensor's values, independent of device and layout."""
    import torch

    detached = tensor.detach()
    flat = detached.to("cpu").contiguous().reshape(-1)
    if flat.numel() == 0:
        return b""
    try:
        return flat.view(torch.uint8).numpy().tobytes()
    except Exception:  # pragma: no cover - exotic dtypes without a byte view
        return flat.to(torch.float32).numpy().tobytes()


def tensor_hash(tensor: Any) -> str:
    """``sha256`` over dtype, shape and the raw values of one tensor."""
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("utf-8"))
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def adapter_digest(source: Any) -> dict[str, Any]:
    """Deterministic per-tensor hashes plus an aggregate over the sorted names.

    Sorted by name, so the result depends on *what* the tensors are, never on the
    iteration order of ``named_parameters()``.
    """
    tensors: dict[str, str] = {}
    numel = 0
    for name, tensor in _named_tensors(source):
        tensors[name] = tensor_hash(tensor)
        try:
            numel += int(tensor.numel())
        except Exception:  # pragma: no cover - defensive
            pass
    aggregate = hashlib.sha256()
    for name in sorted(tensors):
        aggregate.update(f"{name}:{tensors[name]}\n".encode("utf-8"))
    return {
        "count": len(tensors),
        "numel": numel,
        "aggregate": aggregate.hexdigest(),
        "tensors": tensors,
    }


# ---------------------------------------------------------------------------
# scaler / optimizer / scheduler / window
# ---------------------------------------------------------------------------


def scaler_state(scaler: Any) -> dict[str, Any]:
    """``GradScaler`` scale and growth tracker (R-b).

    Accepts either a live scaler or a **loaded ``scaler.pt`` mapping**, so R-b can
    also be judged offline from a state directory that already exists.
    """
    if scaler is None:
        return {"present": False}
    if isinstance(scaler, Mapping):
        # ``torch.save(scaler.state_dict(), "scaler.pt")`` lands exactly this shape:
        # ``{"scale", "growth_factor", "backoff_factor", "growth_interval",
        # "_growth_tracker"}``. Note the private-looking key: the public
        # ``growth_tracker`` attribute no longer exists in current torch, which is
        # why reading it alone reported ``None`` for a scaler that had stepped.
        growth = scaler.get("_growth_tracker", scaler.get("growth_tracker"))
        return {
            "present": True,
            "type": "scaler_state_dict",
            "scale": None if scaler.get("scale") is None else float(scaler["scale"]),
            "growth_tracker": None if growth is None else int(growth),
        }
    scale: Optional[float] = None
    get_scale = getattr(scaler, "get_scale", None)
    if callable(get_scale):
        try:
            scale = float(get_scale())
        except Exception:  # pragma: no cover - scaler not initialised yet
            scale = None
    # ``torch.amp.GradScaler`` keeps the counter in ``_growth_tracker``; the old
    # public ``growth_tracker`` attribute is gone in current torch, and
    # ``state_dict()["growth_tracker"]`` is the documented view. Read all three, in
    # that order, so a real stepped scaler reports a number instead of ``None``.
    growth = getattr(scaler, "_growth_tracker", None)
    if growth is None:
        growth = getattr(scaler, "growth_tracker", None)
    if growth is None:
        state_dict = getattr(scaler, "state_dict", None)
        if callable(state_dict):
            try:
                growth = dict(state_dict()).get("growth_tracker")
            except Exception:  # pragma: no cover - defensive
                growth = None
    return {
        "present": True,
        "type": type(scaler).__name__,
        "scale": scale,
        "growth_tracker": None if growth is None else int(growth),
    }


def optimizer_step(optimizer: Any) -> Optional[int]:
    """The optimizer's own step counter (R-c), when it exposes one."""
    if optimizer is None:
        return None
    for attribute in ("step_count", "_step_count", "steps"):
        value = getattr(optimizer, attribute, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return int(value)
    for group in getattr(optimizer, "param_groups", None) or []:
        value = group.get("step")
        if isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    state_dict = getattr(optimizer, "state_dict", None)
    if callable(state_dict):
        try:
            state = dict(state_dict()).get("state") or {}
        except Exception:  # pragma: no cover - defensive
            state = {}
        steps = []
        for entry in state.values():
            if not isinstance(entry, Mapping):
                continue
            value = entry.get("step")
            try:
                steps.append(int(value))
            except (TypeError, ValueError):
                continue
        if steps:
            return max(steps)
    return None


def scheduler_lrs(scheduler: Any) -> Optional[list[float]]:
    """Current learning rate(s) from the scheduler (R-c)."""
    if scheduler is None:
        return None
    get_last_lr = getattr(scheduler, "get_last_lr", None)
    if callable(get_last_lr):
        try:
            return [float(value) for value in get_last_lr()]
        except Exception:  # pragma: no cover - scheduler without a state yet
            pass
    groups = getattr(scheduler, "param_groups", None) or []
    lrs = []
    for group in groups:
        try:
            lrs.append(float(group.get("lr")))
        except (TypeError, ValueError):
            continue
    return lrs or None


def window_counts(source: Any) -> dict[str, Any]:
    """Window accounting counters: attempts/replays/discards and consumption."""
    from library.training import window_accounting

    fields: Iterable[str] = tuple(window_accounting._COUNTER_FIELDS) + (
        "batches_consumed",
        "batches_per_epoch",
        "resume_skipped_batches",
    )
    out: dict[str, Any] = {}
    for name in fields:
        value = getattr(source, name, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[name] = value
    return out


# ---------------------------------------------------------------------------
# build / write / read
# ---------------------------------------------------------------------------


def build(
    *,
    rank: Optional[int] = None,
    global_step: Optional[int] = None,
    epoch: Optional[int] = None,
    adapter: Any = None,
    scaler: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    window: Any = None,
) -> dict[str, Any]:
    """One rank's digest payload (without the file name)."""
    return {
        "schema": SCHEMA,
        "rank": None if rank is None else int(rank),
        "global_step": None if global_step is None else int(global_step),
        "epoch": None if epoch is None else int(epoch),
        "adapter": adapter_digest(adapter),
        "scaler": scaler_state(scaler),
        "optimizer": {"step": optimizer_step(optimizer)},
        "scheduler": {"lrs": scheduler_lrs(scheduler)},
        "window": window_counts(window),
    }


def build_from_loop_state(
    state: Any,
    *,
    global_step: Optional[int] = None,
    epoch: Optional[int] = None,
    network: Any = None,
    args: Any = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[dict[str, Any]]:
    """The digest for this rank from a training-loop state (``None`` when off).

    Everything is read defensively: a hand-built state that has no scaler or no
    scheduler still produces a digest, with those sections marked absent rather
    than guessed.
    """
    if not enabled(args if args is not None else getattr(state, "args", None), env):
        return None
    accelerator = getattr(state, "accelerator", None)
    network = network if network is not None else getattr(state, "network", None)
    step = global_step if global_step is not None else getattr(state, "global_step", None)
    return build(
        rank=getattr(accelerator, "process_index", None),
        global_step=step,
        epoch=epoch,
        adapter=network,
        scaler=getattr(accelerator, "scaler", None),
        optimizer=getattr(state, "optimizer", None),
        scheduler=getattr(state, "lr_scheduler", None),
        window=state,
    )


def write(digest: Mapping[str, Any], state_dir: str) -> str:
    """Write this rank's digest file next to the other per-rank artifacts."""
    name = FILE.format(rank=int(digest["rank"]))
    path = os.path.join(state_dir, name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(dict(digest), handle, sort_keys=True, indent=1)
        handle.write("\n")
    os.replace(tmp, path)
    return name


def read(state_dir: str, rank: int) -> Optional[dict[str, Any]]:
    path = os.path.join(state_dir, FILE.format(rank=int(rank)))
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("%s: rank digest %s is unreadable (%s)", CODE_DIGEST_MISSING, path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def read_all(
    state_dir: str, world_size: int
) -> tuple[dict[str, dict[str, Any]], list[int]]:
    """``({rank: digest}, [missing ranks])`` for every rank of this topology."""
    digests: dict[str, dict[str, Any]] = {}
    missing: list[int] = []
    for rank in range(max(1, int(world_size))):
        payload = read(state_dir, rank)
        if payload is None:
            missing.append(rank)
        else:
            digests[str(rank)] = payload
    return digests, missing


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _comparable_window(window: Any) -> dict[str, Any]:
    """Window counters minus the rank-local observation-only fields."""
    if not isinstance(window, Mapping):
        return {}
    return {
        key: value
        for key, value in window.items()
        if key not in WINDOW_OBSERVATION_ONLY
    }


def compare(
    digests: Mapping[Any, Mapping[str, Any]], world_size: int
) -> dict[str, Any]:
    """Judge whether every rank's digest is bit-identical (the R-a/R-b/R-c entry).

    Returns ``{"equal", "codes", "missing", "differences", "message", "ranks"}``;
    ``differences`` names the offending field and what each rank reported, so a
    verifier does not have to diff JSON by hand.
    """
    ranks = [str(rank) for rank in range(max(1, int(world_size)))]
    present = {str(key): value for key, value in dict(digests).items()}
    missing = [int(rank) for rank in ranks if rank not in present]

    codes: list[str] = []
    if missing:
        codes.append(CODE_DIGEST_MISSING)

    reference_rank = next((rank for rank in ranks if rank in present), None)
    differences: dict[str, dict[str, Any]] = {}
    if reference_rank is not None:
        reference = present[reference_rank]
        for field in COMPARED_FIELDS:
            values = {
                rank: (
                    _comparable_window(present[rank].get(field))
                    if field == "window"
                    else present[rank].get(field)
                )
                for rank in ranks
                if rank in present
            }
            canonical = {rank: _canonical(value) for rank, value in values.items()}
            if len(set(canonical.values())) > 1:
                differences[field] = values
        if differences:
            codes.append(CODE_DIGEST_MISMATCH)

    if CODE_DIGEST_MISSING in codes and CODE_DIGEST_MISMATCH in codes:
        message = (
            f"rank digests missing for {missing} and "
            f"{sorted(differences)} disagree across ranks"
        )
    elif CODE_DIGEST_MISSING in codes:
        message = f"rank digests missing for rank(s) {missing}"
    elif CODE_DIGEST_MISMATCH in codes:
        message = f"rank digests disagree on {sorted(differences)}"
    else:
        message = "all rank digests are bit-identical"

    # ``None`` means "nothing was compared" (no rank reported a digest at all);
    # ``False`` means this face of group R disagrees.
    axes = {
        axis: (None if reference_rank is None else axis not in differences)
        for axis in AXES
    }
    # ``global_step``/``epoch`` live outside the four axes but still gate equality.
    position_equal = (
        None
        if reference_rank is None
        else ("global_step" not in differences and "epoch" not in differences)
    )
    return {
        "equal": not codes,
        "codes": codes,
        "missing": missing,
        "differences": differences,
        "message": message,
        "axes": axes,
        "adapter_equal": axes["adapter"],
        "scaler_equal": axes["scaler"],
        "optimizer_equal": axes["optimizer"],
        "window_equal": axes["window"],
        "position_equal": position_equal,
        "ranks": {rank: _summary(present[rank]) for rank in ranks if rank in present},
    }


def _summary(digest: Mapping[str, Any]) -> dict[str, Any]:
    """The bundle-sized view of one digest (per-tensor hashes stay in the file)."""
    adapter = dict(digest.get("adapter") or {})
    adapter.pop("tensors", None)
    return {
        "global_step": digest.get("global_step"),
        "epoch": digest.get("epoch"),
        "adapter": adapter,
        "scaler": digest.get("scaler"),
        "optimizer": digest.get("optimizer"),
        "scheduler": digest.get("scheduler"),
        "window": digest.get("window"),
    }


def summarize_state_dir(state_dir: str, world_size: int) -> dict[str, Any]:
    """Read every rank's digest from a published state dir and judge them."""
    digests, missing = read_all(state_dir, world_size)
    comparison = compare(digests, world_size)
    return {
        "schema": SCHEMA,
        "world_size": int(world_size),
        "complete": not missing,
        "equal": bool(comparison["equal"]),
        "codes": list(comparison["codes"]),
        "missing": list(comparison["missing"]),
        "message": comparison["message"],
        "differences": list(comparison["differences"]),
        "axes": dict(comparison["axes"]),
        "adapter_equal": comparison["adapter_equal"],
        "scaler_equal": comparison["scaler_equal"],
        "optimizer_equal": comparison["optimizer_equal"],
        "window_equal": comparison["window_equal"],
        "position_equal": comparison["position_equal"],
        "ranks": comparison["ranks"],
        "files": {
            rank: FILE.format(rank=int(rank)) for rank in range(max(1, int(world_size)))
        },
    }


def log_outcome(summary: Mapping[str, Any], state_dir: str) -> None:
    """Named WARNING on a gap, INFO when the ranks agree (never raises)."""
    codes = list(summary.get("codes") or [])
    if CODE_DIGEST_MISSING in codes or CODE_DIGEST_MISMATCH in codes:
        logger.warning(
            "rank digest check (%s) for %s: %s",
            ", ".join(codes),
            state_dir,
            summary.get("message"),
        )
        return
    logger.info(
        "rank digests agree for %s (%s)",
        state_dir,
        summary.get("message"),
    )
