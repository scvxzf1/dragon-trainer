"""Opt-in per-sample noise/timestep fixture for single-vs-dual comparisons.

The equivalence gate in the validation contract (§13 V-2.4) needs the *same*
``(noise, timestep)`` for a given sample in every run. Production draws both from
the global RNG **per batch shape**, so a 1x2 batch and two 1x1 batches never agree.

This module provides an explicit, opt-in injection point:

* disabled by default — with ``ANIMA_MULTIGPU_NOISE_FIXTURE`` unset the training
  path is byte-for-byte unchanged;
* when set, the JSONL file must carry one record per sample
  (``{"sample_id": ..., "noise": [...], "timestep": ...}``);
* a record that is missing, duplicated, or the wrong shape is a **named error**,
  never a silent fallback to the random draw;
* the sample ids come from the batch the trainer built. The Anima dataloader emits
  ``example["image_keys"]`` — the bucket's image keys for the samples in this
  batch (``library/datasets/dataset_getitem_stack.py``) — and that is the key this
  module resolves first. It is only populated when the dataset runs with
  ``debug_dataset`` (a *do-not-train* inspection mode), so a real training batch
  normally carries no per-sample id at all. For a real run, set
  ``ANIMA_MULTIGPU_NOISE_FIXTURE_IDS=cursor`` and the ids are derived from the
  **sampler position** the training loop commits (see
  :func:`sample_ids_from_position`): the same global sample slot then gets the
  same pinned draw in a single-worker and in a multi-worker run, which is exactly
  what the single-vs-dual gate compares. Without either source the fixture fails
  closed; it never falls back to an unpinned draw.

It is a measurement fixture, not a training feature: nothing here changes the
default behaviour, the loss math, or any admission decision.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch

ENV = "ANIMA_MULTIGPU_NOISE_FIXTURE"
#: How the per-sample ids are resolved: ``batch`` (default) reads them off the
#: training batch, ``cursor`` derives them from the committed sampler position.
ENV_IDS = "ANIMA_MULTIGPU_NOISE_FIXTURE_IDS"

CODE_FIXTURE_INVALID = "noise_fixture_invalid"
CODE_FIXTURE_SAMPLE_MISSING = "noise_fixture_sample_missing"
CODE_FIXTURE_BATCH_IDS_MISSING = "noise_fixture_batch_ids_missing"
CODE_FIXTURE_IDS_MODE = "noise_fixture_ids_mode_invalid"

IDS_BATCH = "batch"
IDS_CURSOR = "cursor"

#: Batch keys that may carry the stable per-sample identifier.
#:
#: ``image_keys`` comes first because it is the key the Anima dataloader really
#: produces (``library/datasets/dataset_getitem_stack.py``: the bucket's image keys
#: for the samples of this batch). The rest are kept for callers that batch
#: ``sample_ids``/``image_names``/``paths`` explicitly. Resolving a key the
#: trainer never emits is how the L3 fixture used to fail closed on every real
#: dataloader batch.
ID_KEYS = ("image_keys", "sample_ids", "image_names", "paths", "file_paths")

#: Prefix of a cursor-derived sample id (``s00000007`` = global sample slot 7).
CURSOR_ID_PREFIX = "s"


class NoiseFixtureError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _floats(value: Any, *, where: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise NoiseFixtureError(CODE_FIXTURE_INVALID, f"{where}: noise must be a non-empty list")
    out: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise NoiseFixtureError(CODE_FIXTURE_INVALID, f"{where}: noise entries must be numbers")
        out.append(float(item))
    return out


def load_fixture(path: str) -> dict[str, dict]:
    """Parse the JSONL fixture into ``sample_id -> {noise, timestep}``."""
    records: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise NoiseFixtureError(CODE_FIXTURE_INVALID, f"{path}:{lineno}: {exc}") from exc
            if not isinstance(row, dict):
                raise NoiseFixtureError(CODE_FIXTURE_INVALID, f"{path}:{lineno}: not an object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise NoiseFixtureError(
                    CODE_FIXTURE_INVALID, f"{path}:{lineno}: sample_id must be a non-empty string"
                )
            if sample_id in records:
                raise NoiseFixtureError(
                    CODE_FIXTURE_INVALID, f"{path}:{lineno}: duplicate sample_id {sample_id!r}"
                )
            timestep = row.get("timestep")
            if not isinstance(timestep, (int, float)) or isinstance(timestep, bool):
                raise NoiseFixtureError(
                    CODE_FIXTURE_INVALID, f"{path}:{lineno}: timestep must be a number"
                )
            if not 0.0 <= float(timestep) <= 1.0:
                raise NoiseFixtureError(
                    CODE_FIXTURE_INVALID,
                    f"{path}:{lineno}: timestep must be a sigma in [0, 1], got {timestep}",
                )
            records[sample_id] = {
                "noise": _floats(row.get("noise"), where=f"{path}:{lineno}"),
                "timestep": float(timestep),
            }
    if not records:
        raise NoiseFixtureError(CODE_FIXTURE_INVALID, f"{path}: no usable records")
    return records


def batch_sample_ids(batch: Any) -> list[str]:
    """Resolve the stable sample ids carried by one training batch."""
    if isinstance(batch, Mapping):
        for key in ID_KEYS:
            value = batch.get(key)
            if value is None:
                continue
            ids = [str(item) for item in _as_sequence(value)]
            if ids:
                return ids
    if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], Mapping):
        return batch_sample_ids(batch[0])
    raise NoiseFixtureError(
        CODE_FIXTURE_BATCH_IDS_MISSING,
        "the batch carries no sample ids (" + ", ".join(ID_KEYS) + "): with the noise "
        "fixture enabled the per-sample draw cannot be pinned, so refusing to fall "
        "back to an unpinned random draw",
    )


def ids_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """``batch`` (default) or ``cursor``; anything else is a named error."""
    values = dict(os.environ if env is None else env)
    mode = str(values.get(ENV_IDS, "") or "").strip().lower() or IDS_BATCH
    if mode not in (IDS_BATCH, IDS_CURSOR):
        raise NoiseFixtureError(
            CODE_FIXTURE_IDS_MODE,
            f"{ENV_IDS}={mode!r}: expected {IDS_BATCH!r} or {IDS_CURSOR!r}",
        )
    return mode


def sample_ids_from_position(
    *,
    local_batch_index: int,
    rank: int = 0,
    world_size: int = 1,
    batch_size: int,
) -> list[str]:
    """Stable per-sample ids derived from the committed sampler position.

    With the worker count pinned to 0 and a fixed seed (contract §13 V-2.2), the
    dataloader order is deterministic and accelerate hands global batch ``b`` to
    rank ``b % world_size`` as that rank's local batch ``b // world_size``. The
    global index of a sample is therefore::

        (local_batch_index * world_size + rank) * batch_size + row

    The same formula identifies the *same* sample slot in a single-worker run
    (``world_size == 1``), which is what makes a single-vs-dual comparison see
    identical ``(noise, timestep)`` for identical samples.
    """
    if int(local_batch_index) < 0 or int(batch_size) <= 0 or int(world_size) < 1:
        raise NoiseFixtureError(
            CODE_FIXTURE_BATCH_IDS_MISSING,
            "cannot derive sample ids from the sampler position "
            f"(local_batch_index={local_batch_index}, batch_size={batch_size}, "
            f"world_size={world_size}): the data order is unknown, so refusing to "
            "fall back to an unpinned random draw",
        )
    global_batch = int(local_batch_index) * int(world_size) + int(rank or 0)
    base = global_batch * int(batch_size)
    return [f"{CURSOR_ID_PREFIX}{base + row:08d}" for row in range(int(batch_size))]


def resolve_sample_ids(
    batch: Any,
    *,
    mode: str = IDS_BATCH,
    local_batch_index: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    batch_size: Optional[int] = None,
) -> list[str]:
    """Per-sample ids for this batch, from the batch itself or from the position."""
    if mode == IDS_CURSOR:
        if local_batch_index is None:
            raise NoiseFixtureError(
                CODE_FIXTURE_BATCH_IDS_MISSING,
                f"{ENV_IDS}={IDS_CURSOR} needs the committed sampler position, but "
                "the trainer did not report one for this micro-batch: refusing to "
                "fall back to an unpinned random draw",
            )
        return sample_ids_from_position(
            local_batch_index=int(local_batch_index),
            rank=int(rank),
            world_size=int(world_size),
            batch_size=int(batch_size if batch_size is not None else 0),
        )
    if mode != IDS_BATCH:
        raise NoiseFixtureError(
            CODE_FIXTURE_IDS_MODE, f"unknown id mode {mode!r}"
        )
    return batch_sample_ids(batch)


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, torch.Tensor)):
        return list(value)
    return [value]


class NoiseFixture:
    """Loaded fixture usable from the training forward."""

    def __init__(self, records: Mapping[str, dict], source: str):
        self.records = dict(records)
        self.source = source

    def apply(
        self,
        noise: torch.Tensor,
        latents: torch.Tensor,
        batch: Any,
        *,
        mode: str = IDS_BATCH,
        local_batch_index: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(noise, sigmas)`` pinned per sample id for this batch."""
        ids = resolve_sample_ids(
            batch,
            mode=mode,
            local_batch_index=local_batch_index,
            rank=rank,
            world_size=world_size,
            batch_size=int(latents.shape[0]),
        )
        if len(ids) != int(latents.shape[0]):
            raise NoiseFixtureError(
                CODE_FIXTURE_INVALID,
                f"batch exposes {len(ids)} sample ids for {int(latents.shape[0])} latents",
            )
        missing = [sample_id for sample_id in ids if sample_id not in self.records]
        if missing:
            raise NoiseFixtureError(
                CODE_FIXTURE_SAMPLE_MISSING,
                f"no fixture record for sample(s) {missing[:5]} (source={self.source})",
            )
        pinned_noise = noise.clone()
        sigmas = torch.empty(len(ids), device=noise.device, dtype=torch.float32)
        for index, sample_id in enumerate(ids):
            record = self.records[sample_id]
            values = torch.tensor(record["noise"], device=noise.device, dtype=noise.dtype)
            if values.numel() != noise[index].numel():
                raise NoiseFixtureError(
                    CODE_FIXTURE_INVALID,
                    f"sample {sample_id!r}: fixture noise has {values.numel()} values, "
                    f"latents need {noise[index].numel()}",
                )
            pinned_noise[index] = values.view_as(noise[index])
            sigmas[index] = float(record["timestep"])
        return pinned_noise, sigmas


_CACHE: dict[str, Optional[NoiseFixture]] = {}


def active(env: Optional[Mapping[str, str]] = None) -> Optional[NoiseFixture]:
    """The enabled fixture, or ``None`` when the env var is unset/empty."""
    values = dict(os.environ if env is None else env)
    path = str(values.get(ENV, "")).strip()
    if not path:
        return None
    if path not in _CACHE:
        _CACHE[path] = NoiseFixture(load_fixture(path), source=path)
    return _CACHE[path]


def reset_cache() -> None:
    _CACHE.clear()


def sample_ids_from_records(records: Mapping[str, dict]) -> Iterable[str]:
    return sorted(records)
