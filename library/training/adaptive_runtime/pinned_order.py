"""Pinned data-order measurement gate, shared by every training-time loader.

The training path builds its data loader in three places, and all three have to
answer to the same gate — otherwise a single-worker and a two-worker run iterate
different images and every cross-topology comparison measures different data:

* ``library/training/bootstrap.py`` — the primary train loader (t32);
* ``library/training/loop.py`` — the loader rebuilt on a stage switch (t36);
* ``library/training/adaptive_runtime/resume_data.py::make_cursor_dataloader``
  — the stateful recovery cursor loader (t36).

This module is the one place that turns ``ANIMA_MULTIGPU_PINNED_ORDER`` into a
loader, so those sites cannot drift apart. It owns no logic of its own: the
identity sampler is the one implemented by the bootstrap and the descriptor
ordering is ``library.datasets.buckets.apply_pinned_order``.

Default off: with the variable unset every helper reproduces the pre-existing
behaviour byte for byte (``shuffle=True`` and the plain ``RandomSampler``), and
``apply`` is a no-op that returns ``0``.

Why pinning needs two halves: the loader shuffle is only one of two
batch-size-dependent permutations. ``DatasetBuckets.shuffle_buckets`` also shuffles
the *batch descriptor* list, whose length is the batch count (12 at batch 2 and 24
at batch 1 for 24 images), so disabling the loader shuffle alone still hands each
slot a different image. Hence every builder here re-applies the natural descriptor
order before constructing the loader.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "PINNED_ORDER_ENV",
    "apply",
    "audit",
    "enabled",
    "identity_sampler",
    "repin",
    "reset_audit",
    "train_dataloader",
]

PINNED_ORDER_ENV = "ANIMA_MULTIGPU_PINNED_ORDER"


def enabled() -> bool:
    """Whether the pinned-order measurement gate is open (default: closed)."""
    from library.datasets.buckets import pinned_order_enabled

    return pinned_order_enabled()


def apply(dataset_group: Any) -> int:
    """Sort the group's batch descriptors into natural order; ``0`` when closed."""
    from library.datasets.buckets import apply_pinned_order

    return apply_pinned_order(dataset_group)


def repin(target: Any) -> int:
    """Re-pin the descriptor order on a single dataset **or** a dataset group.

    ``apply`` only walks ``target.datasets`` (a group). Epoch advances reach the
    leaf datasets one by one (``BaseDataset.set_current_epoch``), so the same rule
    has to be reachable for a leaf; the ordering itself still comes from
    ``library.datasets.buckets.pinned_buckets_indices`` and nothing is reimplemented
    here. Returns the number of descriptors re-ordered (``0`` when the gate is
    closed).
    """
    if not enabled():
        return 0
    from library.datasets.buckets import pinned_buckets_indices

    members = getattr(target, "datasets", None)
    changed = 0
    for dataset in list(members) if members is not None else [target]:
        indices = getattr(dataset, "buckets_indices", None)
        if not indices:
            continue
        ordered = pinned_buckets_indices(dataset)
        if ordered != list(indices):
            dataset.buckets_indices = ordered
            changed += len(ordered)
    return changed


def identity_sampler(item_count: int) -> torch.utils.data.Sampler:
    """The shared identity sampler (single implementation lives in bootstrap)."""
    from library.training.bootstrap import _PinnedOrderSampler

    return _PinnedOrderSampler(item_count)


#: Emitters recorded by the last audited loader, so a measurement can explain an
#: unexpected item sequence instead of guessing (t39 / F-T35-1).
_AUDIT: list = []

#: How many emitted indices one audited sampler keeps verbatim. A long run grows
#: ``item_count * passes`` entries, so the buffer is capped and the totals are kept
#: as counters instead (t43 review, F-T43-3 follow-up).
EMITTED_CAP = 4096


class AuditedIdentitySampler(torch.utils.data.Sampler):
    """Identity stream that records what it emitted, pass by pass.

    The measured ``k*W + rank`` contract only holds for one continuous pass. A
    consumer that re-iterates the loader mid-run replays the prefix instead — which
    is invisible under the default shuffle (a fresh permutation each pass) and very
    visible under the pinned identity order ("step 1 got item 0 again"). Recording
    the emissions turns that into a diagnosable fact: ``passes`` counts how often
    ``__iter__`` restarted the stream, and ``emitted`` is the exact index order this
    process asked the dataset for.
    """

    def __init__(self, inner: torch.utils.data.Sampler):
        self.inner = inner
        self.item_count = len(inner)
        self.passes = 0
        self.emitted: list[int] = []
        self.emitted_total = 0
        self.duplicates = 0
        self.truncated = False
        self._seen: set[int] = set()

    def __iter__(self):
        self.passes += 1
        for index in self.inner:
            index = int(index)
            self.emitted_total += 1
            if index in self._seen:
                self.duplicates += 1
            else:
                self._seen.add(index)
            if len(self.emitted) < EMITTED_CAP:
                self.emitted.append(index)
            else:
                self.truncated = True
            yield index

    def __len__(self) -> int:
        return self.item_count

    def as_dict(self) -> dict:
        return {
            "item_count": self.item_count,
            "passes": self.passes,
            # ``emitted`` is the verbatim (capped) buffer; ``emitted_total`` is what
            # actually went out, so a capped long run stays countable.
            "emitted": list(self.emitted),
            "emitted_total": self.emitted_total,
            "emitted_truncated": self.truncated,
            "duplicates": self.duplicates,
            "expected_prefix": list(range(min(self.item_count, len(self.emitted)))),
        }


def reset_audit() -> None:
    _AUDIT.clear()


def audit() -> list[dict]:
    """The recorded emissions of every audited loader built in this process."""
    return [sampler.as_dict() for sampler in _AUDIT]


def audited_identity_sampler(item_count: int) -> torch.utils.data.Sampler:
    sampler = AuditedIdentitySampler(identity_sampler(item_count))
    _AUDIT.append(sampler)
    return sampler


def train_dataloader(dataset_group: Any, dataloader_kwargs: dict) -> torch.utils.data.DataLoader:
    """Build the training loader for one dataset group.

    Closed gate: exactly ``DataLoader(dataset_group, shuffle=True, **kwargs)``,
    which is what both call sites used before the gate existed. Open gate: natural
    descriptor order plus an identity item stream, so accelerate's
    ``BatchSamplerShard`` gives rank ``r`` the items ``r, r+W, r+2W, ...`` and a
    batch-2 item ``k`` holds the same samples as batch-1 items ``2k``/``2k+1``.
    """
    if not enabled():
        return torch.utils.data.DataLoader(dataset_group, shuffle=True, **dataloader_kwargs)
    apply(dataset_group)
    return torch.utils.data.DataLoader(
        dataset_group,
        sampler=audited_identity_sampler(len(dataset_group)),
        **dataloader_kwargs,
    )
