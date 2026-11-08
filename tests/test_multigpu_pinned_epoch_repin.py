"""Pinned order must survive an epoch boundary (t38).

``BaseDataset.set_current_epoch`` calls ``shuffle_buckets()``, which reseeds with
``seed + epoch`` and shuffles the **batch descriptor** list. That list has one
entry per batch, so its length is the batch count: 24 images give 12 descriptors at
``batch_size=2`` and 24 at ``batch_size=1``. Two permutations of different lengths
cannot agree, so a pinned stream silently breaks at every epoch boundary.

This is not a corner case for the reference runs: on 2 GPUs with 24 samples an
epoch is **12 updates**, so the 20-step S-g segment and the second half of a 10+10
resume (steps 13-20) already run in epoch 1.

The fix lives in ``library/datasets/base.py`` (``_repin_pinned_order``, env gated,
default off) and reuses ``pinned_order.repin`` → ``buckets.pinned_buckets_indices``.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from accelerate.data_loader import BatchSamplerShard

from library.datasets.buckets import pinned_buckets_indices
from library.datasets.dreambooth import DreamBoothDataset
from library.datasets.subsets import DreamBoothSubset
from library.training.adaptive_runtime import pinned_order

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
    def __init__(self, *datasets):
        self.datasets = list(datasets)

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index):
        return self.datasets[0][index]

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)


def _stems(dataset: DreamBoothDataset, item: int) -> list[str]:
    descriptor = dataset.buckets_indices[item]
    bucket = dataset.bucket_manager.buckets[descriptor.bucket_index]
    start = descriptor.batch_index * descriptor.bucket_batch_size
    keys = bucket[start : start + descriptor.bucket_batch_size]
    return [str(key).rsplit("_0896x1200", 1)[0] for key in keys]


def _rows(loader, dataset: DreamBoothDataset, world: int, steps: int) -> list[list[str]]:
    rows = []
    for step in range(steps):
        row: list[str] = []
        for rank in range(world):
            shard = BatchSamplerShard(
                loader.batch_sampler,
                num_processes=world,
                process_index=rank,
                split_batches=False,
            )
            for index, batch in enumerate(shard):
                if index != step:
                    continue
                for item in batch:
                    row.extend(_stems(dataset, int(item)))
                break
        rows.append(row)
    return rows


@pytest.fixture()
def gate_open(monkeypatch):
    monkeypatch.setenv("ANIMA_MULTIGPU_PINNED_ORDER", "1")


@pytest.fixture()
def gate_closed(monkeypatch):
    monkeypatch.delenv("ANIMA_MULTIGPU_PINNED_ORDER", raising=False)


def test_an_epoch_is_twelve_updates_so_twenty_steps_cross_it():
    """The arithmetic that makes this fix necessary, measured on the real dataset."""
    random.seed(SEED)
    one = _build(1)
    two = _build(2)

    assert len(one) == 24 and len(two) == 12
    updates_per_epoch_dual = len(one) // 2      # W=2, batch 1
    updates_per_epoch_single = len(two) // 1    # W=1, batch 2
    assert updates_per_epoch_dual == 12
    assert updates_per_epoch_single == 12

    def epoch_of_step(step: int) -> int:        # 1-based global update index
        return (step - 1) // updates_per_epoch_dual

    assert epoch_of_step(1) == 0
    assert epoch_of_step(12) == 0
    assert epoch_of_step(13) == 1               # first update after the boundary
    assert epoch_of_step(20) == 1
    # 1+1 step and 10+10 resume: the resumed half splits across the boundary.
    assert [epoch_of_step(s) for s in (1, 2)] == [0, 0]
    assert [epoch_of_step(s) for s in range(11, 21)] == [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]


def test_epoch_advance_repins_the_descriptors(gate_open):
    random.seed(SEED)
    dataset = _build(2)
    before = list(dataset.buckets_indices)
    dataset.set_current_epoch(1)
    assert list(dataset.buckets_indices) == pinned_buckets_indices(dataset)


def test_three_items_per_epoch_two_epochs_keep_the_natural_order(gate_open):
    """The small case the task asks for: 3 items per epoch, 2 epochs."""
    random.seed(SEED)
    dataset = _build(8)                          # 24 images // 8 = 3 items/epoch
    assert len(dataset) == 3
    pinned_order.repin(dataset)                  # what the loaders do at build time
    epoch0 = [_stems(dataset, i) for i in range(3)]
    assert [len(item) for item in epoch0] == [8, 8, 8]
    dataset.set_current_epoch(1)
    assert list(dataset.buckets_indices) == pinned_buckets_indices(dataset)
    epoch1 = [_stems(dataset, i) for i in range(3)]
    assert epoch0 == epoch1


def test_slot_alignment_survives_the_epoch_boundary(monkeypatch):
    monkeypatch.setenv("ANIMA_MULTIGPU_PINNED_ORDER", "1")
    random.seed(SEED)
    two = _build(2)
    two_loader = pinned_order.train_dataloader(_Group(two), {})
    random.seed(SEED)
    one = _build(1)
    one_loader = pinned_order.train_dataloader(_Group(one), {})

    epoch0_single = _rows(two_loader, two, world=1, steps=3)
    epoch0_dual = _rows(one_loader, one, world=2, steps=3)
    assert epoch0_single == epoch0_dual

    two.set_current_epoch(1)
    one.set_current_epoch(1)

    epoch1_single = _rows(two_loader, two, world=1, steps=3)
    epoch1_dual = _rows(one_loader, one, world=2, steps=3)
    assert epoch1_single == epoch1_dual
    # Pinned order is epoch independent by design: the same three rows again.
    assert epoch1_single == epoch0_single


def test_closed_gate_keeps_the_previous_epoch_behaviour(gate_closed):
    """Negative control: without the gate the epoch reshuffle is unchanged."""
    random.seed(SEED)
    dataset = _build(2)
    before = list(dataset.buckets_indices)
    expected = list(before)
    random.seed(dataset.seed + 1)
    random.shuffle(expected)

    dataset.set_current_epoch(1)

    assert list(dataset.buckets_indices) == expected       # byte-for-byte previous behaviour
    assert list(dataset.buckets_indices) != pinned_buckets_indices(dataset)
    assert pinned_order.apply(_Group(dataset)) == 0
    assert pinned_order.repin(dataset) == 0


def test_other_epoch_side_effects_are_untouched(gate_open):
    random.seed(SEED)
    dataset = _build(2)
    assert dataset.current_epoch == 0
    dataset.set_current_epoch(1)
    assert dataset.current_epoch == 1
    largest = getattr(dataset, "_largest_bucket_index", None)
    assert largest is not None                # _largest_bucket_first still ran
    pinned_after_first = list(dataset.buckets_indices)

    # A non-incrementing call only sets the epoch; it must not reshuffle, exactly as
    # before the change.
    dataset.set_current_epoch(0)
    assert dataset.current_epoch == 0
    assert list(dataset.buckets_indices) == pinned_after_first


def test_without_the_repin_hook_the_epoch_boundary_breaks_alignment(monkeypatch):
    """Proof that the one line in ``set_current_epoch`` is load-bearing.

    ``_repin_pinned_order`` is neutralised exactly as the pre-t38 code behaved (the
    hook did not exist), everything else stays the same: the two topologies agree in
    epoch 0 and disagree from epoch 1 on. That is the uncontrolled variable that
    steps 13-20 of a 20-update run would have carried. Since t39 the hook lives in
    ``shuffle_buckets`` itself (t38 put it in ``base.set_current_epoch``, which left
    ``create_buckets``/``make_buckets`` uncovered).
    """
    monkeypatch.setenv("ANIMA_MULTIGPU_PINNED_ORDER", "1")
    import library.datasets.dataset_buckets as buckets_module

    monkeypatch.setattr(buckets_module, "_repin_pinned_order", lambda dataset: 0)

    random.seed(SEED)
    two = _build(2)
    two_loader = pinned_order.train_dataloader(_Group(two), {})
    random.seed(SEED)
    one = _build(1)
    one_loader = pinned_order.train_dataloader(_Group(one), {})

    assert _rows(two_loader, two, world=1, steps=3) == _rows(
        one_loader, one, world=2, steps=3
    )

    two.set_current_epoch(1)
    one.set_current_epoch(1)

    # Without the hook the descriptor lists are reshuffled at different lengths and
    # the pinned stream is broken again.
    assert list(two.buckets_indices) != pinned_buckets_indices(two)
    assert list(one.buckets_indices) != pinned_buckets_indices(one)
    assert _rows(two_loader, two, world=1, steps=3) != _rows(
        one_loader, one, world=2, steps=3
    )
