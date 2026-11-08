"""Why single-batch2 and dual-batch1 see different images (t30).

t29 measured it on the real chain: same seed, same pinned noise per slot, same
initial weights, different *images* (slot s0: ``…2022-07-26`` single vs
``…2025-03-09`` dual), which is the whole 23 % loss gap.

These tests reproduce the mechanism with accelerate's **real** sampler classes and
no GPU, attribute it, and prove the harness instrument that removes it:

* :func:`test_the_real_seedable_sampler_still_disagrees_across_batch_sizes` — the
  reproduction, including the detail that matters: the noise keeps lining up by
  slot while the images do not.
* :func:`test_a_sample_level_sampler_is_why_the_old_gate_looked_green` — F-T3-1's
  contradiction: with a dataset whose ``__len__`` counts *samples* (a plain
  dataset, batches formed by the DataLoader) the union does align, which is exactly
  the shape of the old green result.
* :func:`test_pinned_order_puts_the_same_stem_in_the_same_slot` — the fix.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import BatchSampler, Dataset

from accelerate.data_loader import BatchSamplerShard, SeedableRandomSampler

from bench.multigpu_reference import loader_order

SEED = 20261107
N_SAMPLES = 24


class BucketedDataset(Dataset):
    """Stands in for this trainer's dataset: **one item is one whole batch**.

    ``library/training/bootstrap.py`` pins ``DataLoader(batch_size=1)`` for exactly
    that reason, so ``len(dataset)`` is the number of bucket batches and therefore
    depends on the batch size.
    """

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        # The bucket's own order: seeded by data_seed/epoch, identical for both runs.
        self.order = [f"stem{index:02d}" for index in range(N_SAMPLES)]

    def __len__(self) -> int:
        return (len(self.order) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        return self.order[start : start + self.batch_size]


class SampleDataset(Dataset):
    """A plain dataset: ``len`` counts samples, the DataLoader forms the batches."""

    def __init__(self):
        self.order = [f"stem{index:02d}" for index in range(N_SAMPLES)]

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index):
        return self.order[index]


def _seedable(dataset, world: int, rank: int):
    return SeedableRandomSampler(
        data_source=dataset,
        replacement=False,
        num_samples=None,
        generator=torch.Generator(),
        data_seed=SEED,
    )


def _steps_item_level(dataset, *, world: int, rank: int, steps: int = 3):
    """Item indices this rank consumes per step, through accelerate's real shard."""
    sampler = _seedable(dataset, world, rank)
    batch_sampler = BatchSampler(sampler, 1, False)  # DataLoader(batch_size=1)
    shard = BatchSamplerShard(
        batch_sampler, num_processes=world, process_index=rank, split_batches=False
    )
    out = []
    for step, batch in enumerate(shard):
        if step >= steps:
            break
        out.append(list(batch))
    return out


def _stems_per_step(dataset, worlds, steps: int = 3):
    """Per step, the stems of the whole slot row (rank 0 first)."""
    rows = []
    for step in range(steps):
        slots = []
        for rank in range(worlds):
            items = _steps_item_level(dataset, world=worlds, rank=rank, steps=steps)[step]
            for item in items:
                slots.extend(dataset[item])
        rows.append(slots)
    return rows


def test_the_real_seedable_sampler_still_disagrees_across_batch_sizes():
    """The t29 reproduction: same seed, different images, same slots."""
    single = _stems_per_step(BucketedDataset(batch_size=2), worlds=1)
    dual = _stems_per_step(BucketedDataset(batch_size=1), worlds=2)

    difference = loader_order.first_divergence(single, dual)
    assert difference is not None, "the two topologies must not agree at the item level"
    assert difference["step"] == 0
    assert difference["single"] != difference["dual"]

    # The slot arithmetic is untouched: slot s0 in the dual run still means "the
    # first sample of step 0", which is why the pinned noise lines up by slot even
    # though the image does not.
    assert loader_order.slot_sample_index(
        local_batch_index=0, world_size=1, rank=0, batch_size=2
    ) == [0, 1]
    assert loader_order.slot_sample_index(
        local_batch_index=0, world_size=2, rank=0, batch_size=1
    ) == [0]
    assert loader_order.slot_sample_index(
        local_batch_index=0, world_size=2, rank=1, batch_size=1
    ) == [1]


def test_the_item_level_permutation_length_is_what_breaks():
    """``randperm(12)`` and ``randperm(24)`` are unrelated — no seed fixes that."""
    twelve = torch.randperm(12, generator=torch.Generator().manual_seed(SEED)).tolist()
    twentyfour = torch.randperm(24, generator=torch.Generator().manual_seed(SEED)).tolist()
    assert twelve != twentyfour[:12]
    # Single-run item k holds samples 2k, 2k+1; dual-run step k holds 2k, 2k+1 of
    # the *other* permutation.
    assert twelve[0] * 2 != twentyfour[0]


def test_a_sample_level_sampler_is_why_the_old_gate_looked_green():
    """F-T3-1's contradiction, explained.

    When the dataset counts **samples** and the DataLoader forms the batches
    (``batch_size`` passed to ``BatchSampler``), the permutation length no longer
    depends on the batch size, and the two topologies align — which is the shape of
    the old green gate. On the real bucketed dataset the same gate is red.
    """
    def stems(dataset, *, world: int, batch_size: int, steps: int = 3):
        rows = []
        for step in range(steps):
            slots = []
            for rank in range(world):
                sampler = _seedable(dataset, world, rank)
                batch_sampler = BatchSampler(sampler, batch_size, False)
                shard = BatchSamplerShard(
                    batch_sampler, num_processes=world, process_index=rank,
                    split_batches=False,
                )
                for index, batch in enumerate(shard):
                    if index == step:
                        slots.extend(dataset[item] for item in batch)
                        break
            rows.append(slots)
        return rows

    single = stems(SampleDataset(), world=1, batch_size=2)
    dual = stems(SampleDataset(), world=2, batch_size=1)
    assert loader_order.first_divergence(single, dual) is None

    bucketed_single = _stems_per_step(BucketedDataset(batch_size=2), worlds=1)
    bucketed_dual = _stems_per_step(BucketedDataset(batch_size=1), worlds=2)
    assert loader_order.first_divergence(bucketed_single, bucketed_dual) is not None


def test_pinned_order_puts_the_same_stem_in_the_same_slot():
    """The regression: the fix must make slot s carry the same image in both."""

    def pinned(dataset, *, world: int, steps: int = 3):
        rows = []
        for step in range(steps):
            slots = []
            for rank in range(world):
                sampler = loader_order.PinnedOrderBatchSampler(
                    len(dataset), world_size=world, rank=rank
                )
                items = list(sampler)[step]
                for item in items:
                    slots.extend(dataset[item])
            rows.append(slots)
        return rows

    single = pinned(BucketedDataset(batch_size=2), world=1)
    dual = pinned(BucketedDataset(batch_size=1), world=2)
    assert loader_order.first_divergence(single, dual) is None
    assert single[0] == ["stem00", "stem01"]
    assert dual[0] == ["stem00", "stem01"]
    assert single[1] == dual[1] == ["stem02", "stem03"]


def test_identity_shards_partition_the_items_without_reordering():
    for world in (1, 2, 3):
        seen = []
        for rank in range(world):
            indices = loader_order.identity_item_indices(24, world_size=world, rank=rank)
            assert indices == sorted(indices)
            assert indices[0] == rank
            seen.extend(indices)
        assert sorted(seen) == list(range(24))

    with pytest.raises(ValueError):
        loader_order.identity_item_indices(4, world_size=2, rank=2)


def test_the_gate_reports_its_basis_for_the_run_identity(monkeypatch):
    monkeypatch.delenv(loader_order.ENV, raising=False)
    facts = loader_order.describe(item_count=12, world_size=1, batch_size=2)
    assert facts["enabled"] is False and facts["permutation_length"] == 12
    assert "depends on batch_size" in facts["note"]

    monkeypatch.setenv(loader_order.ENV, "1")
    facts = loader_order.describe(item_count=12, world_size=2, batch_size=1)
    assert facts["enabled"] is True
    assert "slot arithmetic matches" in facts["note"]
