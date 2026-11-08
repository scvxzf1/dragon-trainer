"""Full-chain pinned stream: real dataset + real DataLoader + real accelerate (t39).

t35's gate failed on the *index layer*: rank 0 saw item 0 twice in three steps
while ``len(dataset)`` and the item→sample expansion were both correct. The sampler
on its own always looked right, so this module drives the whole chain — the real
24-stem dataset, a real ``DataLoader``, and real ``Accelerator.prepare_data_loader``
in two Gloo processes — and records the exact item sequence each rank asks for.

Measured outcomes (see the report for the full matrix):

* one continuous pass at ``workers`` 0/2/4 gives exactly ``k*W + rank``;
* re-iterating the loader between steps replays the prefix (``0, 0, 0``) — the
  only mechanism that reproduces t35's duplicate, and the reason the pinned order
  is what makes it visible (the default shuffle hides a replay behind a fresh
  permutation);
* prepare twice gives ``0, 4, 8`` (skips, never repeats);
* ``even_batches`` cannot wrap here (24 % 2 == 0, 12 % 1 == 0).

The audit records ``passes``/``emitted`` so a replay is a diagnosable fact rather
than a mystery sequence.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest
import torch

CACHE_ROOT = Path(
    "output/tests/multigpu_reference_validation/t13-anima-hot-20261108/preprocess"
)
IMAGES = CACHE_ROOT / "images896"
CACHE = CACHE_ROOT / "cache"
SEED = 20261107
STEPS = 3

pytestmark = pytest.mark.skipif(
    not (IMAGES.is_dir() and len(list(IMAGES.glob("*.png"))) >= 4),
    reason="t13 preprocess images are not present in this checkout",
)


def _build(batch_size: int):
    from library.datasets.dreambooth import DreamBoothDataset
    from library.datasets.subsets import DreamBoothSubset

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


class _ItemIndexView(torch.utils.data.Dataset):
    """Expose the item index so the sequence is directly observable."""

    def __init__(self, group):
        self.group = group

    def __len__(self):
        return len(self.group)

    def __getitem__(self, index):
        return int(index)


def _first(batch):
    return batch[0]


def _stems(group, item: int) -> list[str]:
    descriptor = group.buckets_indices[item]
    bucket = group.bucket_manager.buckets[descriptor.bucket_index]
    start = descriptor.batch_index * descriptor.bucket_batch_size
    keys = bucket[start : start + descriptor.bucket_batch_size]
    return [str(key).rsplit("_0896x1200", 1)[0] for key in keys]


def _worker(rank: int, world: int, batch_size: int, mode: str, out_dir: str) -> None:
    """mode: ``pass`` (pinned), ``closed`` (gate off), ``reiterate`` (per-step iter)."""
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29571",
            "RANK": str(rank),
            "WORLD_SIZE": str(world),
            "LOCAL_RANK": str(rank),
            "ACCELERATE_USE_CPU": "true",
        }
    )
    if mode == "closed":
        os.environ.pop("ANIMA_MULTIGPU_PINNED_ORDER", None)
    else:
        os.environ["ANIMA_MULTIGPU_PINNED_ORDER"] = "1"
    from accelerate import Accelerator

    from library.training.adaptive_runtime import pinned_order

    random.seed(SEED)
    group = _build(batch_size)

    class _Group:
        datasets = [group]

        def __len__(self):
            return len(group)

        def __getitem__(self, index):
            return index

    view = _ItemIndexView(group)
    accelerator = Accelerator()
    pinned_order.reset_audit()
    if mode == "closed":
        # Production path: shuffled descriptors and a plain RandomSampler.
        pinned_order.repin(_Group())
        loader = pinned_order.train_dataloader(_Group(), {"batch_size": 1, "collate_fn": _first})
    else:
        pinned_order.apply(_Group())
        loader = torch.utils.data.DataLoader(
            view,
            sampler=pinned_order.audited_identity_sampler(len(view)),
            batch_size=1,
            collate_fn=_first,
            num_workers=0,
        )
    prepared = accelerator.prepare_data_loader(loader)

    items: list[int] = []
    if mode in ("pass", "closed"):
        for step, batch in enumerate(prepared):
            items.append(int(batch))
            if step == STEPS - 1:
                break
    elif mode == "reiterate":
        # A consumer that pulls one batch per fresh iterator — the t35 symptom.
        for _ in range(STEPS):
            items.append(int(next(iter(prepared))))

    payload = {
        "rank": rank,
        "world": world,
        "batch_size": batch_size,
        "items": items,
        "stems": [_stems(group, item) for item in items],
        "audit": pinned_order.audit(),
        "sampler_type": type(prepared.batch_sampler).__name__,
        "len_prepared": len(prepared),
        "len_dataset": len(group),
    }
    Path(out_dir, f"chain-{mode}-b{batch_size}-w{world}-r{rank}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _run(world: int, batch_size: int, mode: str, tmp: Path) -> list[dict]:
    import torch.multiprocessing as mp

    if world == 1:
        _worker(0, 1, batch_size, mode, str(tmp))
    else:
        mp.spawn(_worker, args=(world, batch_size, mode, str(tmp)), nprocs=world, join=True)
    return [
        json.loads(Path(tmp, f"chain-{mode}-b{batch_size}-w{world}-r{rank}.json").read_text())
        for rank in range(world)
    ]


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pinned-chain")
    single = _run(1, 2, "pass", tmp)     # world 1, batch 2
    dual = _run(2, 1, "pass", tmp)       # world 2, batch 1
    reiterate = _run(2, 1, "reiterate", tmp)
    closed = _run(2, 1, "closed", tmp)   # gate off, production path
    closed_single = _run(1, 2, "closed", tmp)
    return single, dual, reiterate, closed, closed_single


def test_dual_ranks_get_exactly_k_w_plus_rank(chain):
    _, dual, _, _, _ = chain
    assert [dual[0]["items"], dual[1]["items"]] == [[0, 2, 4], [1, 3, 5]]
    # Batch-1 items carry one stem each; the dataloader really ran the accelerate shard.
    assert all(len(item) == 1 for row in dual for item in row["stems"])
    assert dual[0]["sampler_type"] == "BatchSamplerShard"
    # len(loader) agrees with what a pass actually yields (V40-F5).
    assert dual[0]["len_dataset"] == 24 and dual[0]["len_prepared"] == 12
    assert chain[0][0]["len_prepared"] == 12


def test_each_dual_rank_sees_three_distinct_stems(chain):
    """Delegates to the shared contract so bench and tests assert the same thing."""
    from bench.multigpu_reference import loader_order

    _, dual, _, _, _ = chain
    for row in dual:
        facts = loader_order.assert_stream_contract(
            row["stems"],
            item_count=row["len_dataset"],
            world_size=row["world"],
            steps=STEPS,
            loader_len=row["len_prepared"],
            label=f"rank{row['rank']} ",
        )
        assert facts["items"] == facts["expected_items"] == STEPS
        assert facts["unique_stems"] == STEPS


def test_slots_agree_between_the_two_topologies(chain):
    single, dual, _, _, _ = chain
    # single: 3 items of 2 stems; dual: 3 steps of rank0+rank1, 1 stem each.
    assert len(single[0]["stems"]) == STEPS
    for step in range(STEPS):
        assert single[0]["stems"][step] == dual[0]["stems"][step] + dual[1]["stems"][step]


def test_the_audit_shows_one_continuous_pass(chain):
    _, dual, _, _, _ = chain
    for row in dual:
        entry = row["audit"][0]
        assert entry["passes"] == 1
        assert entry["duplicates"] == 0
        # The sampler emitted the whole identity stream; the shard selected from it.
        # Only the prefetch window is emitted for a 3-step measurement; it must be
        # the ascending identity prefix, never a replay.
        assert entry["emitted"] == list(range(len(entry["emitted"])))
        assert entry["item_count"] == 24


def test_reiterating_the_loader_replays_the_prefix(chain):
    """Root cause of t35's duplicate: a fresh iterator per step restarts the stream."""
    _, _, reiterate, _, _ = chain
    assert [reiterate[0]["items"], reiterate[1]["items"]] == [[0, 0, 0], [1, 1, 1]]
    for row in reiterate:
        assert len({stem for item in row["stems"] for stem in item}) == 1
        assert row["audit"][0]["passes"] == STEPS
        assert row["audit"][0]["duplicates"] > 0


def test_the_gate_closed_path_is_still_a_plain_random_sampler(monkeypatch):
    monkeypatch.delenv("ANIMA_MULTIGPU_PINNED_ORDER", raising=False)
    from library.training.adaptive_runtime import pinned_order

    random.seed(SEED)
    group = _build(2)

    class _Group:
        datasets = [group]

        def __len__(self):
            return len(group)

    loader = pinned_order.train_dataloader(_Group(), {})
    assert type(loader.sampler) is torch.utils.data.RandomSampler
    torch.manual_seed(1234)
    from_gate = list(loader.sampler)
    torch.manual_seed(1234)
    reference = torch.utils.data.DataLoader(_Group(), shuffle=True, batch_size=1)
    assert from_gate == list(reference.sampler)
    assert pinned_order.audit() == [] or all(
        entry["passes"] == 0 for entry in pinned_order.audit()
    )


def test_the_chain_case_has_discriminating_power(chain):
    """Gate closed must be *red* on the same assertions the open gate passes.

    Without this the suite would prove nothing: the open-gate test has to fail in
    the closed configuration, i.e. the same-slot-same-stem property must actually
    depend on the gate.
    """
    _, dual, _, closed_dual, closed_single = chain
    open_single_rows = chain[0][0]["stems"]           # per step: two stems
    closed_single_rows = closed_single[0]["stems"]   # per step: two stems
    closed_rows = [row["stems"] for row in closed_dual]  # per step: one stem each
    # Both closed runs consume *some* items, but the slot pairing does not hold.
    assert closed_rows[0] and closed_rows[1]
    closed_pairs = [
        closed_single_rows[step] == closed_rows[0][step] + closed_rows[1][step]
        for step in range(STEPS)
    ]
    assert not all(closed_pairs), (
        "gate-closed run accidentally aligned: the test case lost its discriminating "
        "power"
    )
    # ...while the open gate aligns on every step.
    open_pairs = [
        open_single_rows[step] == dual[0]["stems"][step] + dual[1]["stems"][step]
        for step in range(STEPS)
    ]
    assert all(open_pairs)
    assert type(chain[0][0]["sampler_type"]) is str
