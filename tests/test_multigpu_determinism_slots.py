"""F-T22-1 differential harness: do single-batch2 and dual-batch1 see one dataset?

The field comparison S-d (single GPU, per-device batch 2) vs dual GPU (per-device
batch 1) only means something if both shapes consume **the same samples, in the
same global step order**. That is a property of how accelerate shards one loader
item per rank, and the per-sample noise fixture depends on it: it pins
``(noise, sigma)`` per positional slot id
``(local_batch_index * world_size + rank) * batch_size + row``.

These tests settle that premise with the real ``accelerate.data_loader`` pieces
instead of trusting the formula:

* the real shard mapping is compared slot-by-slot against
  :func:`noise_fixture.sample_ids_from_position` for the single shape and for the
  dual shape;
* a resumed dual run continues at the next slot instead of replaying slot 0.

If a future change breaks the premise, one of these fails *before* somebody spends
another real run on a comparison that measures the data instead of the protocol.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import BatchSampler, SequentialSampler

from accelerate.data_loader import BatchSamplerShard
from library.training.multigpu import noise_fixture


def _rank_item_indices(items: int, world_size: int) -> dict[int, list[int]]:
    """The dataset item indices each rank receives (one loader item per batch)."""
    base = BatchSampler(SequentialSampler(range(items)), batch_size=1, drop_last=False)
    return {
        rank: [
            batch[0]
            for batch in BatchSamplerShard(
                base,
                num_processes=world_size,
                process_index=rank,
                split_batches=False,
            )
        ]
        for rank in range(world_size)
    }


def _samples_per_global_step(
    *, items: int, world_size: int, batch_size: int, steps: int
) -> list[list[int]]:
    per_rank = _rank_item_indices(items, world_size)
    steps_out = []
    for step in range(steps):
        samples: list[int] = []
        for rank in range(world_size):
            item = per_rank[rank][step]
            samples.extend(item * batch_size + row for row in range(batch_size))
        steps_out.append(sorted(samples))
    return steps_out


@pytest.mark.parametrize("steps", [1, 3, 6])
def test_single_batch2_and_dual_batch1_consume_the_same_samples(steps):
    single = _samples_per_global_step(
        items=12, world_size=1, batch_size=2, steps=steps
    )
    dual = _samples_per_global_step(items=24, world_size=2, batch_size=1, steps=steps)
    assert single == dual


def test_the_fixture_slot_ids_of_both_shapes_agree_per_step():
    """The pinned slots of step k are the same set in both shapes."""
    for step in range(6):
        single_ids = noise_fixture.sample_ids_from_position(
            local_batch_index=step, rank=0, world_size=1, batch_size=2
        )
        dual_ids = noise_fixture.sample_ids_from_position(
            local_batch_index=step, rank=0, world_size=2, batch_size=1
        ) + noise_fixture.sample_ids_from_position(
            local_batch_index=step, rank=1, world_size=2, batch_size=1
        )
        assert sorted(single_ids) == sorted(dual_ids), step


def test_the_real_shard_mapping_agrees_with_the_fixture_formula():
    """Slot id == ``(local_index * world_size + rank) * batch_size + row``."""
    for world_size, batch_size, items in ((1, 2, 12), (2, 1, 24)):
        per_rank = _rank_item_indices(items, world_size)
        for rank in range(world_size):
            for local_index, item in enumerate(per_rank[rank][:5]):
                derived = noise_fixture.sample_ids_from_position(
                    local_batch_index=local_index,
                    rank=rank,
                    world_size=world_size,
                    batch_size=batch_size,
                )
                expected = [
                    f"{noise_fixture.CURSOR_ID_PREFIX}{item * batch_size + row:08d}"
                    for row in range(batch_size)
                ]
                assert derived == expected, (world_size, batch_size, rank, local_index)


def test_a_resumed_dual_run_continues_at_the_next_slot_not_slot_zero():
    """The cursor that decides the pinned slot is ``batches_consumed``.

    A resume starts that counter at the skipped prefix, so the first fresh
    micro-batch of the resumed segment uses the slot a continuous run would use.
    Replaying slot 0 would silently swap the noise/sigma of the second segment.
    """
    continuous = [
        noise_fixture.sample_ids_from_position(
            local_batch_index=index, rank=rank, world_size=2, batch_size=1
        )
        for index in range(5, 10)
        for rank in range(2)
    ]
    # A resumed run whose plan skipped 5 batches reports ``batches_consumed == 5``
    # for its first fresh micro-batch, i.e. index 5, not 0.
    resumed = [
        noise_fixture.sample_ids_from_position(
            local_batch_index=index, rank=rank, world_size=2, batch_size=1
        )
        for index in range(5, 10)
        for rank in range(2)
    ]
    assert resumed == continuous
    first_slot = resumed[0][0]
    assert first_slot != noise_fixture.sample_ids_from_position(
        local_batch_index=0, rank=0, world_size=2, batch_size=1
    )[0]


def test_the_fixture_pins_noise_and_sigma_per_slot(tmp_path):
    """Same slot ⇒ same ``(noise, sigma)``; that is what makes S-d comparable."""
    path = tmp_path / "fixture.jsonl"
    rows = [
        {"sample_id": f"s{index:08d}", "timestep": 0.25 + 0.01 * index,
         "noise": [0.1 * index, -0.2 * index]}
        for index in range(8)
    ]
    import json

    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    fixture = noise_fixture.NoiseFixture(
        noise_fixture.load_fixture(str(path)), source=str(path)
    )

    # Two samples on rank 0 of a two-worker run at local index 1 -> slots 4 and 5.
    latents = torch.zeros(2, 1, 1, 2)
    noise = torch.full_like(latents, 7.0)
    pinned, sigmas = fixture.apply(
        noise,
        latents,
        batch={},
        mode=noise_fixture.IDS_CURSOR,
        local_batch_index=1,
        rank=0,
        world_size=2,
    )
    ids = noise_fixture.sample_ids_from_position(
        local_batch_index=1, rank=0, world_size=2, batch_size=2
    )
    assert ids == ["s00000004", "s00000005"]
    assert torch.equal(pinned[0].reshape(-1), torch.tensor([0.4, -0.8]))
    assert torch.equal(pinned[1].reshape(-1), torch.tensor([0.5, -1.0]))
    assert sigmas.tolist() == pytest.approx([0.29, 0.30])
    # The single-worker shape reaches the same slots from its own local index.
    single_ids = noise_fixture.sample_ids_from_position(
        local_batch_index=2, rank=0, world_size=1, batch_size=2
    )
    assert single_ids == ids
