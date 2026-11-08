"""Main-thread stateful data cursor, with isolated data RNG and bucket order."""

from contextlib import contextmanager
from copy import deepcopy
import random
import threading

import numpy as np
import torch

from . import pinned_order
from .resume_buckets import capture_buckets, restore_buckets


def _rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state())


def _set_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def first_example(examples):
    """DatasetGroup items are already batches; never run a second collator."""
    return examples[0]


def make_cursor_dataloader(group, *, seed, loader_kwargs=None):
    from torchdata.stateful_dataloader.sampler import BatchSampler, RandomSampler

    options = dict(loader_kwargs or {})
    forbidden = {"batch_size", "batch_sampler", "sampler", "shuffle", "collate_fn", "generator"}
    if forbidden.intersection(options) or options.pop("num_workers", 0) != 0:
        raise ValueError("Recovery loader requires workers=0 and its own sampler/collator")
    data = StatefulDatasetView(group, seed=seed)
    if pinned_order.enabled():
        # Pinned-order measurement gate (t36): same semantics as the bootstrap and
        # stage-switch loaders — natural descriptor order plus an identity item
        # stream, so the cursor hands the same stem to the same slot in a
        # single-worker and a two-worker run.
        pinned_order.apply(group)
        sampler = pinned_order.identity_sampler(len(data))
    else:
        sampler = RandomSampler(
            data, generator=torch.Generator().manual_seed(seed + 1)
        )
    return torch.utils.data.DataLoader(
        data, batch_sampler=BatchSampler(sampler, 1, False), num_workers=0,
        collate_fn=first_example, generator=torch.Generator().manual_seed(seed + 2), **options)


def capture_loader(loader):
    return deepcopy({"schema": "recovery_loader_v1",
                     "loader": loader.state_dict(), "iteration": loader.iteration,
                     "data_identity": getattr(loader.dataset, "data_identity", None)})


def validate_loader_snapshot(snapshot):
    if (not isinstance(snapshot, dict) or snapshot.get("schema") != "recovery_loader_v1"
            or type(snapshot.get("iteration")) is not int or snapshot["iteration"] < 0
            or not isinstance(snapshot.get("loader"), dict)
            or type(snapshot["loader"].get("_iterator_finished")) is not bool):
        raise ValueError("Invalid stateful loader snapshot")
    data = snapshot["loader"].get("dataset_state")
    if (not isinstance(data, dict) or data.get("schema") != "recovery_dataset_v1"
            or type(data.get("epoch")) is not int or data["epoch"] < -1
            or type(data.get("fetched")) is not int or data["fetched"] < 0):
        raise ValueError("Invalid cursor dataset state")
    try:
        state = snapshot["loader"]
        for name in ("_sampler_iter_yielded", "_num_yielded"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError("Invalid loader progress")
        sampler = state["_sampler_iter_state"]
        if type(sampler["samples_yielded"]) is not int or sampler["samples_yielded"] != state["_num_yielded"]:
            raise ValueError("Sampler progress differs")
        random_sampler = sampler["sampler_iter_state"]
        if (type(random_sampler["yielded"]) is not int
                or random_sampler["yielded"] != sampler["samples_yielded"]):
            raise ValueError("Random sampler progress differs")
        torch.Generator().set_state(random_sampler["generator"])
        random.Random().setstate(data["rng"][0])
        np.random.RandomState().set_state(data["rng"][1])
        torch.Generator().set_state(data["rng"][2])
        buckets = data["buckets"]
        if buckets.get("schema") != "recovery_bucket_order_v1" or not buckets.get("datasets"):
            raise ValueError("Invalid cursor buckets")
        length = sum(member["length"] for member in buckets["datasets"])
        yielded = state["_num_yielded"]
        if (type(length) is not int or length <= 0 or not 0 <= yielded <= length
                or state["_sampler_iter_yielded"] != yielded
                or snapshot["iteration"] != data["epoch"]
                or data["fetched"] != snapshot["iteration"] * length + yielded
                or state["_iterator_finished"] != (yielded == length)):
            raise ValueError("Cursor epoch/fetched progress differs")
        from .resume_buckets import _validate_candidate
        for member in buckets["datasets"]:
            _validate_candidate(member, member)
    except (KeyError, TypeError, RuntimeError, AttributeError, IndexError) as exc:
        raise ValueError("Invalid cursor contents") from exc


def restore_loader(loader, snapshot):
    snapshot = deepcopy(snapshot)
    validate_loader_snapshot(snapshot)
    if snapshot.get("data_identity") != getattr(loader.dataset, "data_identity", None):
        raise ValueError("Data identity changed")
    loader.iteration = snapshot["iteration"]
    loader.load_state_dict(snapshot["loader"])
    # Materialize torchdata's lazy restore BEFORE Accelerate sets the epoch.
    # Otherwise an exhausted-state restore overwrites the next epoch's shuffle.
    materialized = loader.base_dataloader.state_dict()
    # Accelerate's state_dict() returns a cached prefetch-adjusted snapshot.
    # load_state_dict() only updates the base loader; there is no prefetch yet,
    # so publish the materialized committed cursor before the first iteration.
    loader.dl_state_dict = deepcopy(materialized)
    if snapshot["loader"]["_iterator_finished"]:
        loader.iteration += 1


class StatefulDatasetView(torch.utils.data.Dataset):
    def __init__(self, group, *, seed):
        self.group = group
        self.epoch = -1
        self.fetched = 0
        self._rng = (random.Random(seed).getstate(), np.random.RandomState(seed % 2**32).get_state(),
                     torch.Generator().manual_seed(seed).get_state())

    @contextmanager
    def _data_rng(self):
        if (threading.current_thread() is not threading.main_thread()
                or torch.utils.data.get_worker_info() is not None):
            raise ValueError("Data RNG prototype supports main-thread worker=0 only")
        previous = _rng_state()
        try:
            _set_rng(self._rng)
            yield
        finally:
            self._rng = _rng_state()
            _set_rng(previous)

    def __len__(self):
        return len(self.group)

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            with self._data_rng():
                self.group.set_current_epoch(epoch + 1)
                # The epoch advance re-shuffles the batch descriptors; re-pin them
                # so pinned order survives an epoch boundary (no-op when closed).
                pinned_order.apply(self.group)
            self.epoch = epoch

    def __getitem__(self, index):
        with self._data_rng():
            self.group.set_current_step(self.fetched)
            item = self.group[index]
            self.fetched += 1
            return item

    def state_dict(self):
        return deepcopy({"schema": "recovery_dataset_v1", "epoch": self.epoch,
                         "fetched": self.fetched, "rng": self._rng,
                         "buckets": capture_buckets(self.group)})

    def load_state_dict(self, state):
        state = deepcopy(state)
        if state.get("schema") != "recovery_dataset_v1":
            raise ValueError("Unknown data RNG snapshot")
        if type(state["fetched"]) is not int or state["fetched"] < 0:
            raise ValueError("Invalid fetched count")
        if type(state["epoch"]) is not int or state["epoch"] < -1:
            raise ValueError("Invalid data epoch")
        previous = _rng_state()
        try:
            _set_rng(state["rng"])
        finally:
            _set_rng(previous)
        restore_buckets(self.group, state["buckets"])
        self.epoch, self.fetched, self._rng = state["epoch"], state["fetched"], state["rng"]
