"""Readable per-rank RNG evidence for exact resume (contract §8 P1-2).

``exact=True`` must mean "the streams were really saved and can really be read",
not "the bundle has a boolean field". This module therefore *loads* every
``random_states_<rank>.pkl`` and inspects its contents instead of trusting a flag.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

CODE_RNG_EVIDENCE_MISSING = "multigpu_rng_evidence_missing"
CODE_RNG_EVIDENCE_CORRUPT = "multigpu_rng_evidence_corrupt"

#: RNG streams accelerate can persist; CUDA only exists on a CUDA topology.
BASE_STREAMS = ("python", "numpy", "torch")

#: Stream name → the key accelerate **really** writes into
#: ``random_states_<rank>.pkl`` (``accelerate/checkpointing.py``:
#: ``states["random_state"|"numpy_random_seed"|"torch_manual_seed"|
#: "torch_cuda_manual_seed"]``). Reading the bundle's own vocabulary
#: (``python``/``numpy``/``torch``) out of that pickle always said "missing",
#: which made ``exact=True`` unreachable on a real save (t3 F-T3-3).
ACCELERATE_KEYS = {
    "python": "random_state",
    "numpy": "numpy_random_seed",
    "torch": "torch_manual_seed",
    "cuda": "torch_cuda_manual_seed",
}

#: Older in-tree fixtures wrote the *bundle* vocabulary into the pickle. They are
#: still readable so a state saved before this fix keeps its named verdict, but
#: the schema is reported per rank and never assumed.
LEGACY_KEYS = {"python": "python", "numpy": "numpy", "torch": "torch", "cuda": "cuda"}


class RngEvidenceError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def state_files(state_dir: str) -> dict[int, str]:
    """Map rank → ``random_states_<rank>.pkl`` path (existing files only)."""
    out: dict[int, str] = {}
    try:
        names = os.listdir(state_dir)
    except OSError:
        return out
    for name in names:
        if not (name.startswith("random_states_") and name.endswith(".pkl")):
            continue
        rank_part = name[len("random_states_") : -len(".pkl")]
        if not rank_part.isdigit():
            continue
        path = os.path.join(state_dir, name)
        try:
            if os.path.getsize(path) > 0:
                out[int(rank_part)] = path
        except OSError:
            continue
    return out


def _payload_schema(payload: Mapping[str, Any]) -> str:
    """Which key vocabulary this pickle uses (``accelerate`` wins if mixed)."""
    if any(ACCELERATE_KEYS[name] in payload for name in BASE_STREAMS):
        return "accelerate"
    return "legacy"


def _lookup(payload: Mapping[str, Any], stream: str) -> Any:
    """The value accelerate wrote for ``stream``, falling back to the legacy key."""
    if ACCELERATE_KEYS[stream] in payload:
        return payload[ACCELERATE_KEYS[stream]]
    return payload.get(LEGACY_KEYS[stream])


def _streams_present(payload: Mapping[str, Any]) -> dict[str, bool]:
    """Which streams a decoded pickle really carries."""
    present: dict[str, bool] = {}
    for name in BASE_STREAMS:
        present[name] = _lookup(payload, name) is not None
    cuda = _lookup(payload, "cuda")
    if isinstance(cuda, (list, tuple)):
        present["cuda"] = len(cuda) > 0
    else:
        present["cuda"] = cuda is not None
    return present


def observe_state_dir(
    state_dir: str, *, world_size: int, uses_cuda: bool
) -> dict[str, Any]:
    """``inspect_state_dir`` as a value instead of an exception.

    One reading path for both moments: ``build_bundle`` observes the files right
    after the save, and ``guard_resume`` observes the **same** files again when the
    state is actually resumed. The vocabulary (``source`` / per-stream booleans) is
    identical, so neither side can drift into a second convention.
    """
    try:
        return inspect_state_dir(
            state_dir, world_size=world_size, uses_cuda=uses_cuda
        )
    except RngEvidenceError as exc:
        return {
            "python": False,
            "numpy": False,
            "torch": False,
            "cuda": None,
            "source": (
                "missing"
                if exc.code == CODE_RNG_EVIDENCE_MISSING
                else "unreadable"
            ),
            "error": exc.code,
            "detail": str(exc),
        }


def inspect_state_dir(
    state_dir: str, *, world_size: int, uses_cuda: bool
) -> dict[str, Any]:
    """Load and describe every rank's RNG file; raise on missing/corrupt.

    Returns a dict with per-rank details plus the aggregate booleans consumed by
    the resume bundle. ``cuda`` is reported as ``None`` ("not applicable") on a
    CPU topology instead of being faked to ``False``/``True``.
    """
    files = state_files(state_dir)
    if not files:
        raise RngEvidenceError(
            CODE_RNG_EVIDENCE_MISSING,
            f"no readable random_states_*.pkl under {state_dir}; runtime RNG state "
            "was not persisted, so this state cannot be resumed exactly",
        )
    missing_ranks = [rank for rank in range(int(world_size)) if rank not in files]
    if missing_ranks:
        raise RngEvidenceError(
            CODE_RNG_EVIDENCE_MISSING,
            f"RNG state missing for rank(s) {missing_ranks} of world_size={world_size}",
        )

    import torch

    ranks: dict[str, dict[str, Any]] = {}
    for rank, path in sorted(files.items()):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RngEvidenceError(
                CODE_RNG_EVIDENCE_CORRUPT, f"{path}: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RngEvidenceError(
                CODE_RNG_EVIDENCE_CORRUPT, f"{path}: payload is not a mapping"
            )
        present = _streams_present(payload)
        ranks[str(rank)] = {
            "file": os.path.basename(path),
            "streams": present,
            "schema": _payload_schema(payload),
        }

    every = {name: all(ranks[r]["streams"][name] for r in ranks) for name in BASE_STREAMS}
    cuda_flag: Optional[bool]
    if uses_cuda:
        cuda_flag = all(ranks[r]["streams"].get("cuda", False) for r in ranks)
    else:
        cuda_flag = None  # not applicable on a CPU topology - never faked
    return {
        "schema": (
            "accelerate"
            if all(entry["schema"] == "accelerate" for entry in ranks.values())
            else "legacy"
        ),
        "python": every["python"],
        "numpy": every["numpy"],
        "torch": every["torch"],
        "cuda": cuda_flag,
        "files": [entry["file"] for entry in ranks.values()],
        "ranks": ranks,
        "ranks_covered": len(ranks),
        "source": "loaded_random_states",
    }
