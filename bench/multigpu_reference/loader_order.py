"""Why single-batch2 and dual-batch1 see different images, and how to pin them.

t29 measured the real symptom: with the same seed, the same fixture noise per slot
and the same initial weights, the *images* differ between a one-worker batch-2 run
and a two-worker batch-1 run — s0 was ``pixiv_…2022-07-26`` on the single side and
``pixiv_…2025-03-09`` on the dual side. The 23 % loss gap follows from that alone.

The cause is the shuffle's **unit**, not the sharding formula:

* this trainer's dataset yields **one whole batch per item** (``library/training/
  bootstrap.py:889`` pins ``DataLoader(batch_size=1)`` because "one dataset item
  already is one batch"), so ``len(dataset)`` is the number of *bucket batches*:
  12 for 24 images at batch 2, 24 at batch 1;
* ``shuffle=True`` becomes accelerate's :class:`SeedableRandomSampler`, which draws
  ``torch.randperm(len(dataset), generator=g)`` — a permutation **of items**, and
  ``randperm(12)`` and ``randperm(24)`` are unrelated permutations whatever the
  seed;
* ``BatchSamplerShard(split_batches=False)`` then hands item ``k*W + rank`` to
  ``rank`` at step ``k``.

Put together, slot ``s`` gets item ``k`` of a 12-long permutation in the single run
and item ``2k``/``2k+1`` of a 24-long permutation in the dual run. The *sharding*
formula is right (that is what :func:`...noise_fixture.sample_ids_from_position`
assumes, and why the pinned noise still lines up by slot); the **permutation**
cannot line up, because its length depends on the batch size. No ``data_seed``
value makes two different permutation lengths agree.

So the S-d divergence is **a measurement limitation, not a product defect**: the
seedable-sampler gate aligns the *seed*, but the item-level shuffle is inherently
batch-size dependent.

:class:`PinnedOrderBatchSampler` is the instrument that removes it: iterate the
items in their *natural* order and shard by ``k*W + rank``, so that

* single (W=1, B=2): step ``k`` -> item ``k`` -> samples ``[2k, 2k+1]``
* dual   (W=2, B=1): step ``k`` -> item ``2k`` / ``2k+1`` -> samples ``2k`` / ``2k+1``

i.e. the same sample per slot in both topologies. Because the bucket order inside
the dataset is itself seeded from ``data_seed`` and identical for both runs, the
stems then match slot by slot. This is a **measurement instrument**: it changes the
sample order of the run that enables it, so it belongs to a validation run only,
exactly like ``ANIMA_MULTIGPU_NOISE_FIXTURE``.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Optional, Sequence

ENV = "ANIMA_MULTIGPU_PINNED_ORDER"

_TRUE = {"1", "true", "yes", "on"}


def enabled(env: Optional[dict[str, str]] = None) -> bool:
    values = dict(os.environ if env is None else env)
    return str(values.get(ENV, "")).strip().lower() in _TRUE


def identity_item_indices(item_count: int, *, world_size: int, rank: int) -> list[int]:
    """Item indices ``rank`` consumes, in natural order (no shuffle)."""
    if int(item_count) < 0 or int(world_size) < 1 or not 0 <= int(rank) < int(world_size):
        raise ValueError(
            f"invalid shard request (item_count={item_count}, world_size={world_size}, "
            f"rank={rank})"
        )
    return list(range(int(rank), int(item_count), int(world_size)))


def slot_sample_index(
    *, local_batch_index: int, world_size: int, rank: int, batch_size: int
) -> list[int]:
    """Global sample indices of one step's slot — the pinned-order contract.

    Identical to :func:`...noise_fixture.sample_ids_from_position`'s arithmetic, so
    the noise fixture and the data stream agree by construction.
    """
    base = (int(local_batch_index) * int(world_size) + int(rank)) * int(batch_size)
    return [base + row for row in range(int(batch_size))]


class PinnedOrderBatchSampler:
    """Yield one item index per step in natural order, sharded by ``k*W + rank``.

    Drop-in for accelerate's ``BatchSamplerShard`` when the dataset already yields
    whole batches (``DataLoader(batch_size=1)``).
    """

    def __init__(self, item_count: int, world_size: int = 1, rank: int = 0):
        self.item_count = int(item_count)
        self.world_size = int(world_size)
        self.rank = int(rank)

    def __len__(self) -> int:
        return len(
            identity_item_indices(
                self.item_count, world_size=self.world_size, rank=self.rank
            )
        )

    def __iter__(self) -> Iterator[list[int]]:
        for item in identity_item_indices(
            self.item_count, world_size=self.world_size, rank=self.rank
        ):
            yield [item]


def describe(*, item_count: int, world_size: int, batch_size: int) -> dict[str, Any]:
    """Facts a run identity can record, so an S-d comparison states its own basis."""
    return {
        "env": ENV,
        "enabled": enabled(),
        "item_count": int(item_count),
        "world_size": int(world_size),
        "batch_size": int(batch_size),
        "permutation_length": int(item_count),
        "pinned": enabled(),
        "note": (
            "item-level shuffle length depends on batch_size; pinned order removes "
            "it for cross-world comparison"
            if not enabled()
            else "natural item order: slot arithmetic matches the noise fixture"
        ),
    }


def first_divergence(
    single: Sequence[Sequence[str]], dual: Sequence[Sequence[str]]
) -> Optional[dict[str, Any]]:
    """First step where the two per-step sample lists differ (``None`` if equal)."""
    for step, (left, right) in enumerate(zip(single, dual)):
        if list(left) != list(right):
            return {"step": step, "single": list(left), "dual": list(right)}
    if len(single) != len(dual):
        return {
            "step": min(len(single), len(dual)),
            "single": list(single[min(len(single), len(dual))]) if single else [],
            "dual": list(dual[min(len(single), len(dual))]) if dual else [],
        }
    return None


#: The three assertions every pinned-stream measurement has to carry (V40-F5). They
#: live here so the bench and the full-chain tests assert the *same* contract instead
#: of each restating it.
def expected_per_rank(item_count: int, *, world_size: int, steps: int) -> int:
    """Items one rank must receive in ``steps`` steps of a pinned pass.

    ``BatchSamplerShard(split_batches=False)`` yields one item per step from the
    ``r, r+W, r+2W, ...`` selector, so this is ``steps`` and never more: a larger
    consumption means the stream was re-iterated (replay) or the shard advanced
    twice.
    """
    if item_count < 0:
        raise ValueError("item_count must be non-negative")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    available = -(-item_count // world_size)  # ceil
    return min(int(steps), available)


def assert_stream_contract(
    rows: Sequence[Sequence[str]],
    *,
    item_count: int,
    world_size: int,
    steps: int,
    loader_len: Optional[int] = None,
    label: str = "",
) -> dict[str, Any]:
    """Assert one rank's pinned consumption: count, cross-step uniqueness, length.

    ``rows`` is the per-step list of stems that rank consumed (one entry per step,
    each a list because a batch-2 item carries two stems). Returns the observed
    facts so a caller can record them next to the artifact.
    """
    if len(rows) != steps:
        raise AssertionError(
            f"{label}rank stopped after {len(rows)} steps, expected {steps}"
        )
    flat_per_step = [list(row) for row in rows]
    # One row per step is one dataset item; a batch-2 item carries two stems, so the
    # item count is the row count, not the stem count.
    consumed = len(flat_per_step)
    expected = expected_per_rank(item_count, world_size=world_size, steps=steps)
    if consumed != expected:
        raise AssertionError(
            f"{label}rank consumed {consumed} items in {steps} steps, expected "
            f"{expected} (= min(steps, ceil({item_count}/{world_size}))): a pinned "
            "pass must hand out exactly one item per step"
        )
    flat = [stem for row in flat_per_step for stem in row]
    if len(set(flat)) != len(flat):
        duplicates = sorted({stem for stem in flat if flat.count(stem) > 1})
        raise AssertionError(
            f"{label}rank saw the same stem twice in {steps} steps: {duplicates}"
        )
    per_pass = -(-item_count // world_size)
    if loader_len is not None and loader_len != per_pass:
        raise AssertionError(
            f"{label}len(loader)={loader_len} disagrees with one full pass over "
            f"{item_count} items on {world_size} rank(s) ({per_pass})"
        )
    if loader_len is not None and steps > loader_len:
        raise AssertionError(
            f"{label}{steps} steps exceed the {loader_len} batches one pass holds"
        )
    return {
        "steps": steps,
        "items": consumed,
        "expected_items": expected,
        "unique_stems": len(set(flat)),
        "loader_len": loader_len,
        "items_per_pass": per_pass,
        "world_size": world_size,
        "item_count": item_count,
    }
