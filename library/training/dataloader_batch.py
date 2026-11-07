"""Where the per-device batch size really comes from.

``bootstrap`` used to build its ``DataLoader`` with a literal ``batch_size=1``.
That literal is correct — the *dataset* already assembles one item per batch
(``BaseDataset.__getitem__`` walks ``bucket_batch_size`` samples) — but it hid the
fact that the per-device batch size is a **configurable** value: it is the dataset
blueprint's ``batch_size`` (``--train_batch_size`` / ``datasets[].batch_size``).
Reading the literal as "the batch size is hard-coded to 1" is how contract §V-2.6
(one GPU with ``batch_size=2``) looked impossible (t3 F-T3-7/F-T3-1b).

This module states both numbers explicitly, so a run can prove which per-device
batch it trained with instead of inferring it from a literal:

* ``loader_batch_size`` — the ``DataLoader`` wrapper size, always ``1`` (the
  dataset item already is a batch; a wrapper size > 1 would stack dicts of
  tensors and break the training forward);
* ``per_device_batch_size`` — the samples per item, taken from the dataset group
  and validated to be uniform across subsets.

It also fails closed when the configured ``--train_batch_size`` disagrees with what
the datasets actually produce: the scheduler/step arithmetic is computed from the
configured value, so a silent disagreement would train a different batch than the
run reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

CODE_BATCH_MISMATCH = "dataset_batch_size_mismatch"

#: The dataset item is already a batch, so the loader wrapper never groups items.
LOADER_BATCH_SIZE = 1


class DatasetBatchError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class LoaderBatchPlan:
    loader_batch_size: int
    per_device_batch_size: int
    sources: tuple[tuple[str, int], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "loader_batch_size": self.loader_batch_size,
            "per_device_batch_size": self.per_device_batch_size,
            "dataset_batch_sizes": dict(self.sources),
        }


def _dataset_batch_sizes(dataset_group: Any) -> list[tuple[str, int]]:
    """``(dataset label, batch_size)`` for every member dataset that declares one."""
    out: list[tuple[str, int]] = []
    datasets = getattr(dataset_group, "datasets", None)
    if datasets is None:
        single = getattr(dataset_group, "batch_size", None)
        if single is not None:
            out.append(("dataset", int(single)))
        return out
    for index, dataset in enumerate(datasets):
        value = getattr(dataset, "batch_size", None)
        if value is None:
            continue
        out.append((f"dataset[{index}]", int(value)))
    return out


def resolve_loader_batch(
    args: Any, dataset_group: Any, *, strict: bool = True
) -> LoaderBatchPlan:
    """Resolve the loader wrapper size and the real per-device batch size."""
    sizes = _dataset_batch_sizes(dataset_group)
    if sizes:
        distinct = sorted({value for _, value in sizes})
        if len(distinct) > 1:
            raise DatasetBatchError(
                CODE_BATCH_MISMATCH,
                "subsets disagree about the per-device batch size "
                f"({dict(sizes)}); one training run must feed one batch shape",
            )
        per_device = distinct[0]
    else:  # a group that does not expose it: keep the historical behaviour
        per_device = LOADER_BATCH_SIZE

    configured = getattr(args, "train_batch_size", None)
    if strict and configured is not None:
        try:
            configured_int: Optional[int] = int(configured)
        except (TypeError, ValueError):
            configured_int = None
        if configured_int is not None and configured_int > 0 and configured_int != per_device:
            raise DatasetBatchError(
                CODE_BATCH_MISMATCH,
                f"--train_batch_size={configured_int} but the dataset group produces "
                f"{per_device} sample(s) per item ({dict(sizes)}); the step and "
                "scheduler arithmetic is computed from the configured value, so this "
                "run would train a different batch than it reports",
            )

    plan = LoaderBatchPlan(
        loader_batch_size=LOADER_BATCH_SIZE,
        per_device_batch_size=int(per_device),
        sources=tuple(sizes),
    )
    if plan.per_device_batch_size != LOADER_BATCH_SIZE:
        logger.info(
            "per-device batch size %d (loader wrapper %d): set by the dataset "
            "blueprint batch_size / --train_batch_size",
            plan.per_device_batch_size,
            plan.loader_batch_size,
        )
    return plan
