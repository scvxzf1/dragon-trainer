"""Same slot must mean the same image in a single- and a two-worker run (t32).

t31 proved the S-d premise was never satisfied: with the *DataLoader* shuffle
disabled the same slot still received a different image, and one rank even saw the
same image twice. There are **two** layers of batch-size-dependent shuffling:

1. the DataLoader's ``shuffle=True``, hard-coded in ``bootstrap`` with no CLI switch;
2. ``DatasetBuckets.create_buckets`` → ``shuffle_buckets``, which shuffles the
   *batch descriptor* list with ``random.seed(seed + epoch)``. The list has one
   entry per batch, so it is 12 entries at batch 2 and 24 at batch 1 for the same
   24 images, and ``random.shuffle`` over 12 and over 24 items are unrelated
   permutations. That is why turning the loader shuffle off changed nothing.

Both layers answer to ``ANIMA_MULTIGPU_PINNED_ORDER`` (default off). With it on,
item ``k`` at batch 2 holds the same two samples as items ``2k`` and ``2k+1`` at
batch 1, so accelerate's ``BatchSamplerShard`` (``rank`` takes ``k*W + rank``)
puts the same image in the same slot.

These tests use the **real** ``DreamBoothDataset`` over the t13 cache: 24 real
stems, real buckets, no GPU and no VAE forward.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from torch.utils.data import BatchSampler

from accelerate.data_loader import BatchSamplerShard

from library.datasets.buckets import (
    apply_pinned_order,
    pinned_buckets_indices,
    pinned_order_enabled,
)
from library.datasets.dreambooth import DreamBoothDataset
from library.datasets.subsets import DreamBoothSubset
from library.training.bootstrap import _PinnedOrderSampler

CACHE_ROOT = Path(
    "output/tests/multigpu_reference_validation/t13-anima-hot-20261108/preprocess"
)
IMAGES = CACHE_ROOT / "images896"
CACHE = CACHE_ROOT / "cache"
SEED = 20261107

pytestmark = pytest.mark.skipif(
    not (IMAGES.is_dir() and len(list(IMAGES.glob("*.png"))) >= 4),
    reason="t13 preprocess images are not present in this checkout",
)


def _build(batch_size: int) -> DreamBoothDataset:
    subset = DreamBoothSubset(
        image_dir=str(IMAGES), is_reg=False, class_tokens=None,
        caption_extension=".txt", cache_info=False, alpha_mask=False,
        num_repeats=1, sample_ratio=1.0, caption_separator=",",
        keep_tokens=0, keep_tokens_separator="|||", secondary_separator=";;;",
        enable_wildcard=False, color_aug=False, flip_aug=False,
        face_crop_aug_range=None, random_crop=False, caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0, caption_tag_dropout_rate=0.0,
        caption_prefix=None, caption_suffix=None, token_warmup_min=1,
        token_warmup_step=0, cache_dir=str(CACHE),
    )
    dataset = DreamBoothDataset(
        subsets=[subset], is_training_dataset=True, batch_size=batch_size,
        network_multiplier=1.0, prior_loss_weight=1.0, debug_dataset=False,
        validation_split=0.0, validation_seed=None,
        resize_interpolation="lanczos", resolution=1024, enable_bucket=True,
        min_bucket_reso=256, max_bucket_reso=2048, bucket_reso_steps=64,
        bucket_no_upscale=False,
    )
    dataset.make_buckets()
    return dataset


class _Group:
    """Minimal stand-in for ``DatasetGroup`` (``apply_pinned_order`` reads it)."""

    def __init__(self, *datasets):
        self.datasets = list(datasets)


def _stems(dataset: DreamBoothDataset, item: int) -> list[str]:
    """The stems one dataset item really carries."""
    descriptor = dataset.buckets_indices[item]
    bucket = dataset.bucket_manager.buckets[descriptor.bucket_index]
    start = descriptor.batch_index * descriptor.bucket_batch_size
    keys = bucket[start : start + descriptor.bucket_batch_size]
    return [str(key).rsplit("_0896x1200", 1)[0] for key in keys]


def _stem_rows(dataset: DreamBoothDataset, world: int, steps: int = 3) -> list[list[str]]:
    """Per step, the stems of the whole slot row (rank 0 first) — real shard."""
    rows = []
    for step in range(steps):
        row: list[str] = []
        for rank in range(world):
            shard = BatchSamplerShard(
                BatchSampler(_PinnedOrderSampler(len(dataset)), 1, False),
                num_processes=world,
                process_index=rank,
                split_batches=False,
            )
            for index, batch in enumerate(shard):
                if index != step:
                    continue
                for item in batch:
                    row.extend(_stems(dataset, item))
                break
        rows.append(row)
    return rows


@pytest.fixture()
def gate_off(monkeypatch):
    monkeypatch.delenv("ANIMA_MULTIGPU_PINNED_ORDER", raising=False)
    random.seed(SEED)


def test_the_gate_is_off_by_default_and_changes_nothing(gate_off):
    """Negative control: the production path keeps its shuffled descriptor order."""
    assert pinned_order_enabled() is False
    dataset = _build(2)
    before = list(dataset.buckets_indices)
    assert apply_pinned_order(_Group(dataset)) == 0
    assert list(dataset.buckets_indices) == before
    # The descriptors are shuffled by shuffle_buckets, so the natural order is not
    # what the loader would iterate.
    assert before != pinned_buckets_indices(dataset)


def test_the_descriptor_list_length_depends_on_the_batch_size(gate_off):
    two, one = _build(2), _build(1)
    assert len(two) == 12 and len(one) == 24
    # ...which is exactly why one shuffle cannot align both.
    assert len(two.buckets_indices) != len(one.buckets_indices)


def test_pinned_order_makes_one_batch2_item_equal_two_batch1_items(monkeypatch):
    monkeypatch.setenv("ANIMA_MULTIGPU_PINNED_ORDER", "1")
    random.seed(SEED)
    two = _build(2)
    apply_pinned_order(_Group(two))
    random.seed(SEED)
    one = _build(1)
    apply_pinned_order(_Group(one))

    for item in range(4):
        assert _stems(two, item) == _stems(one, 2 * item) + _stems(one, 2 * item + 1)


def test_same_slot_same_stem_across_batch_sizes(monkeypatch):
    """The regression the captain asked for: one slot, one image."""
    monkeypatch.setenv("ANIMA_MULTIGPU_PINNED_ORDER", "1")
    random.seed(SEED)
    two = _build(2)
    apply_pinned_order(_Group(two))
    random.seed(SEED)
    one = _build(1)
    apply_pinned_order(_Group(one))

    single = _stem_rows(two, world=1)
    dual = _stem_rows(one, world=2)
    assert single == dual
    assert single[0] == _stems(two, 0)
    # Two different images per slot, no accidental repetition in the first steps.
    assert len(set(sum(single, []))) == 6


def test_without_the_descriptor_pin_the_two_topologies_still_disagree(monkeypatch):
    """The second layer: pinning only the loader shuffle is not enough.

    Since t39 the descriptor pin lives inside ``shuffle_buckets`` itself, so building
    a dataset with the gate open already pins it. Neutralising that one hook therefore
    reproduces the t31 situation (loader shuffle off, descriptors left shuffled).
    """
    monkeypatch.setenv("ANIMA_MULTIGPU_PINNED_ORDER", "1")
    import library.datasets.dataset_buckets as buckets_module

    monkeypatch.setattr(buckets_module, "_repin_pinned_order", lambda dataset: 0)
    random.seed(SEED)
    two = _build(2)
    random.seed(SEED)
    one = _build(1)
    assert list(two.buckets_indices) != pinned_buckets_indices(two)
    assert list(one.buckets_indices) != pinned_buckets_indices(one)
    assert _stem_rows(two, world=1) != _stem_rows(one, world=2)


def test_the_pinned_sampler_is_a_plain_identity_stream():
    sampler = _PinnedOrderSampler(7)
    assert list(sampler) == list(range(7))
    assert len(sampler) == 7
    assert isinstance(sampler, torch.utils.data.Sampler)


def test_the_only_repeat_mechanism_is_the_even_batches_wraparound():
    """Where a repeated image can come from, and why it is not the dataset.

    The t13 set holds 24 unique stems with ``num_repeats = 1`` and
    ``validation_split_num = 0``, so no item shares an image. The stack can still
    repeat one: ``BatchSamplerShard(even_batches=True)`` (accelerate's default)
    loops back to the start for the last ranks when ``len(dataset) % world_size``
    is not zero. That is a data-*fetch* repeat, not a data problem — and pinning the
    order keeps the counts divisible (12 and 24 are both even).
    """

    def ranks(item_count: int, world: int) -> list[list[int]]:
        out = []
        for rank in range(world):
            shard = BatchSamplerShard(
                BatchSampler(_PinnedOrderSampler(item_count), 1, False),
                num_processes=world, process_index=rank, split_batches=False,
            )
            out.append([batch[0] for batch in shard])
        return out

    # An odd item count makes the last rank wrap onto item 0.
    assert ranks(5, 2) == [[0, 2, 4], [1, 3, 0]]
    # The pinned configuration is divisible in both topologies, so nothing repeats.
    for item_count, world in ((24, 2), (12, 2), (24, 1)):
        for consumed in ranks(item_count, world):
            assert len(set(consumed)) == len(consumed)


# ---- hot_run plan carries the gate, and the artifact can prove it (t41 / V40-F3) ----


@pytest.fixture()
def hot_assets(tmp_path):
    train = tmp_path / "train.py"
    train.write_text("# fake entry\n", encoding="utf-8")
    config = tmp_path / "run.toml"
    config.write_text("[base]\n", encoding="utf-8")
    base = tmp_path / "anima-base.safetensors"
    base.write_bytes(b"0" * 16)
    fixture = tmp_path / "l3.jsonl"
    fixture.write_text(
        '{"sample_id": "s0", "noise": [0.1, 0.2], "timestep": 0.5}\n', encoding="utf-8"
    )
    return {
        "train_script": str(train),
        "config": str(config),
        "base": str(base),
        "fixture": str(fixture),
        "project_root": tmp_path,
    }


def _plan_hot_run(assets, **kwargs):
    from bench.multigpu_reference import hot_run

    return hot_run.plan_hot_run(
        kwargs.pop("name", "RUN-B"),
        train_script=assets["train_script"],
        config=assets["config"],
        base_assets=[assets["base"]],
        l3_fixture=assets["fixture"],
        project_root=assets["project_root"],
        env={hot_run.HOT_ENV: "1"},
        **kwargs,
    )


def test_hot_run_plan_records_the_pinned_gate_and_reads_it_back(hot_assets):
    """A run artifact must be able to prove whether the measurement gate was open."""
    from bench.multigpu_reference import hot_run, loader_order

    plan = _plan_hot_run(hot_assets, run_id="pinned-plan")
    assert plan.facts["pinned_order"] is True
    assert plan.facts["launch_env"][loader_order.ENV] == "1"
    # The env handed to the runner must carry it too, not just the facts block.
    readback = hot_run.verify_pinned_gate(plan, plan.facts["launch_env"])
    assert readback == {
        "pinned_order": True,
        "env_name": "ANIMA_MULTIGPU_PINNED_ORDER",
        "launch_env": "1",
        "handed_env": "1",
    }
    executed = hot_run.execute(plan, runner=None, base_env={})
    assert [phase["pinned_order"] for phase in executed["phases"]] == [
        "1" for _ in executed["phases"]
    ]


def test_hot_run_plan_can_request_the_production_shuffle(hot_assets):
    from bench.multigpu_reference import hot_run, loader_order

    plan = _plan_hot_run(hot_assets, run_id="unpinned-plan", pinned_order=False)
    assert plan.facts["pinned_order"] is False
    assert loader_order.ENV not in plan.facts["launch_env"]
    assert hot_run.verify_pinned_gate(plan, plan.facts["launch_env"])["handed_env"] is None
    executed = hot_run.execute(plan, runner=None, base_env={})
    assert all(phase["pinned_order"] is None for phase in executed["phases"])


def test_the_gate_readback_rejects_an_unprovable_run(hot_assets):
    from bench.multigpu_reference import hot_run, loader_order

    plan = _plan_hot_run(hot_assets, run_id="mismatch-plan", pinned_order=False)
    # Someone exported the gate by hand outside the plan: exactly the t35 failure.
    with pytest.raises(hot_run.HotRunRefused):
        hot_run.verify_pinned_gate(plan, {loader_order.ENV: "1"})


# ---- the shared stream contract (t41 / V40-F5, reused by the full-chain test) ----


def test_stream_contract_accepts_a_pinned_pass_and_rejects_a_replay():
    from bench.multigpu_reference import loader_order

    rows = [["s0", "s1"], ["s2", "s3"], ["s4", "s5"]]
    facts = loader_order.assert_stream_contract(
        rows, item_count=12, world_size=1, steps=3, loader_len=12
    )
    assert facts["items"] == facts["expected_items"] == 3
    assert facts["unique_stems"] == 6

    # A replay: the same two stems come back at step 1 instead of new ones.
    with pytest.raises(AssertionError):
        loader_order.assert_stream_contract(
            [["s0", "s1"], ["s0", "s1"], ["s2", "s3"]],
            item_count=12, world_size=1, steps=3, loader_len=12,
        )
    # Wrong count for the horizon (a shard that advanced twice per step).
    with pytest.raises(AssertionError):
        loader_order.assert_stream_contract(
            [["s0"], ["s1"], ["s2"]], item_count=24, world_size=2, steps=2,
        )
    # len(loader) disagreeing with the consumed count.
    with pytest.raises(AssertionError):
        loader_order.assert_stream_contract(
            rows, item_count=12, world_size=1, steps=3, loader_len=99,
        )


def test_expected_per_rank_is_ceil_of_the_shard_and_never_more_than_steps():
    from bench.multigpu_reference import loader_order

    assert loader_order.expected_per_rank(24, world_size=2, steps=3) == 3
    assert loader_order.expected_per_rank(24, world_size=2, steps=20) == 12
    assert loader_order.expected_per_rank(12, world_size=1, steps=20) == 12
    assert loader_order.expected_per_rank(5, world_size=2, steps=9) == 3
