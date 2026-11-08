"""Pinned order must hold on the two loaders that t32 did not cover (t36).

Three loader construction sites exist in the training path: the primary loader in
``bootstrap`` (pinned since t32), the loader rebuilt on a stage switch in
``loop.py``, and the stateful recovery cursor loader in
``adaptive_runtime/resume_data.py``. The last two used to ignore
``ANIMA_MULTIGPU_PINNED_ORDER``, so ``stage_schedule`` and
``adaptive_oom_retry_resume`` runs stayed misaligned per world size — exactly the
runs that most need a comparable stream.

Everything here goes through the real code: the real
``pinned_order.train_dataloader`` that ``loop.py`` calls, the real
``make_cursor_dataloader``, the real ``DreamBoothDataset`` over the t13 cache, and
accelerate's real ``BatchSamplerShard`` for the rank split. No GPU, no VAE.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from accelerate.data_loader import BatchSamplerShard

from library.datasets.dreambooth import DreamBoothDataset
from library.datasets.subsets import DreamBoothSubset
from library.training.adaptive_runtime import pinned_order
from library.training.adaptive_runtime.resume_data import make_cursor_dataloader

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
    """What the real builders see: a `datasets` list plus item-index access."""

    def __init__(self, *datasets):
        self.datasets = list(datasets)

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index):
        return self.datasets[0][index]

    def set_current_epoch(self, epoch):
        # Mimic DatasetBuckets.shuffle_buckets: an epoch advance reshuffles the
        # batch descriptor list. That is the second, batch-size-dependent permuta-
        # tion the gate has to absorb, so the fake must reproduce it.
        for dataset in self.datasets:
            setattr(dataset, "current_epoch", epoch)
            random.shuffle(dataset.buckets_indices)

    def set_current_step(self, step):
        for dataset in self.datasets:
            setattr(dataset, "current_step", step)


def _stems(dataset: DreamBoothDataset, item: int) -> list[str]:
    descriptor = dataset.buckets_indices[item]
    bucket = dataset.bucket_manager.buckets[descriptor.bucket_index]
    start = descriptor.batch_index * descriptor.bucket_batch_size
    keys = bucket[start : start + descriptor.bucket_batch_size]
    return [str(key).rsplit("_0896x1200", 1)[0] for key in keys]


def _rows(loader, dataset: DreamBoothDataset, world: int, steps: int = 3) -> list[list[str]]:
    """Per step, the stems of the whole slot row, using the real shard class."""
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


def _stage_switch_loaders():
    """The two loaders a stage switch would build, one per topology."""
    random.seed(SEED)
    two = _build(2)
    two_loader = pinned_order.train_dataloader(_Group(two), {})
    random.seed(SEED)
    one = _build(1)
    one_loader = pinned_order.train_dataloader(_Group(one), {})
    return (two, two_loader), (one, one_loader)


@pytest.fixture()
def gate_closed(monkeypatch):
    monkeypatch.delenv("ANIMA_MULTIGPU_PINNED_ORDER", raising=False)


@pytest.fixture()
def gate_open(monkeypatch):
    monkeypatch.setenv("ANIMA_MULTIGPU_PINNED_ORDER", "1")


def test_stage_switch_loader_puts_the_same_stem_in_the_same_slot(gate_open):
    (two, two_loader), (one, one_loader) = _stage_switch_loaders()
    single = _rows(two_loader, two, world=1)
    dual = _rows(one_loader, one, world=2)
    assert single == dual
    assert single[0] == _stems(two, 0)


def test_stage_switch_loader_is_not_aligned_when_the_gate_is_closed(gate_closed):
    (two, two_loader), (one, one_loader) = _stage_switch_loaders()
    assert _rows(two_loader, two, world=1) != _rows(one_loader, one, world=2)


def test_closed_gate_builds_the_previous_loader(gate_closed):
    """Negative control: same sampler type and same index sequence as before."""
    dataset = _build(2)
    group = _Group(dataset)
    loader = pinned_order.train_dataloader(group, {})
    assert type(loader.sampler) is torch.utils.data.RandomSampler

    torch.manual_seed(1234)
    before = list(loader.sampler)
    torch.manual_seed(1234)
    reference = torch.utils.data.DataLoader(
        group, shuffle=True, batch_size=1, collate_fn=lambda batch: batch[0]
    )
    assert type(reference.sampler) is torch.utils.data.RandomSampler
    assert before == list(reference.sampler)
    # ...and the descriptors were left exactly as create_buckets shuffled them.
    assert pinned_order.apply(group) == 0


def test_the_open_gate_is_what_changes_the_loader(gate_open):
    dataset = _build(1)
    loader = pinned_order.train_dataloader(_Group(dataset), {})
    assert type(loader.sampler) is not torch.utils.data.RandomSampler
    assert list(loader.sampler) == list(range(len(dataset)))


def test_cursor_dataloader_puts_the_same_stem_in_the_same_slot(gate_open):
    random.seed(SEED)
    two = _build(2)
    two_loader = make_cursor_dataloader(_Group(two), seed=SEED)
    random.seed(SEED)
    one = _build(1)
    one_loader = make_cursor_dataloader(_Group(one), seed=SEED)

    single = _rows(two_loader, two, world=1)
    dual = _rows(one_loader, one, world=2)
    assert single == dual
    assert single[0] == _stems(two, 0)


def test_cursor_dataloader_is_not_aligned_when_the_gate_is_closed(gate_closed):
    random.seed(SEED)
    two = _build(2)
    two_loader = make_cursor_dataloader(_Group(two), seed=SEED)
    random.seed(SEED)
    one = _build(1)
    one_loader = make_cursor_dataloader(_Group(one), seed=SEED)
    assert _rows(two_loader, two, world=1) != _rows(one_loader, one, world=2)


def test_cursor_dataloader_keeps_its_recovery_contract(gate_open):
    """The gate must not weaken the cursor's validation or its state recording."""
    random.seed(SEED)
    dataset = _build(1)
    group = _Group(dataset)
    loader = make_cursor_dataloader(group, seed=SEED)
    assert loader.num_workers == 0
    with pytest.raises(ValueError):
        make_cursor_dataloader(group, seed=SEED, loader_kwargs={"num_workers": 2})
    with pytest.raises(ValueError):
        make_cursor_dataloader(group, seed=SEED, loader_kwargs={"shuffle": True})


def test_the_cursor_repins_after_an_epoch_advance(gate_open):
    """An epoch advance re-shuffles descriptors; the cursor must re-pin them."""
    random.seed(SEED)
    dataset = _build(2)
    group = _Group(dataset)
    loader = make_cursor_dataloader(group, seed=SEED)
    view = loader.dataset
    view.set_epoch(0)
    # set_epoch advances the group epoch, which reshuffles the descriptor list;
    # pinned order must be restored immediately afterwards.
    from library.datasets.buckets import pinned_buckets_indices

    assert list(dataset.buckets_indices) == pinned_buckets_indices(dataset)
    assert list(loader.sampler) == list(range(len(dataset)))
