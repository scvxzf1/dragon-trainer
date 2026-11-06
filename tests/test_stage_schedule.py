"""Unit tests for percent-based stage schedule helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.training.stage_schedule import (
    StageSpec,
    active_subset_indices_for_step,
    attach_stage_schedule_from_config,
    full_dataset_updates_per_epoch,
    normalize_stage_dicts,
    normalize_stage_target_groups,
    parse_stage_specs,
    progress_from_steps,
    resolve_stage_index,
    stage_schedule_enabled,
    stage_target_count,
    validate_stage_specs,
)


def test_full_dataset_epoch_budget_matches_web_estimate():
    assert full_dataset_updates_per_epoch(420) == 420
    assert full_dataset_updates_per_epoch(
        420,
        num_processes=2,
        gradient_accumulation_steps=2,
    ) == 105
    assert 6 * full_dataset_updates_per_epoch(420) == 2520


def test_normalize_accepts_percent_0_100_and_fraction():
    stages = normalize_stage_dicts(
        [
            {"name": "a", "subset_index": 0, "start_pct": 0, "end_pct": 50},
            {"name": "b", "subset_index": 1, "start_pct": 0.5, "end_pct": 1.0},
        ]
    )
    assert stages[0]["start_pct"] == 0.0
    assert stages[0]["end_pct"] == 0.5
    assert stages[1]["start_pct"] == 0.5
    assert stages[1]["end_pct"] == 1.0


def test_validate_two_stage_cover():
    specs = parse_stage_specs(
        [
            {"subset_index": 0, "start_pct": 0.0, "end_pct": 0.4},
            {"subset_index": 1, "start_pct": 0.4, "end_pct": 1.0},
        ]
    )
    assert validate_stage_specs(specs, subset_count=2) == []


def test_validate_rejects_gap_and_bad_subset():
    specs = parse_stage_specs(
        [
            {"subset_index": 0, "start_pct": 0.0, "end_pct": 0.3},
            {"subset_index": 9, "start_pct": 0.5, "end_pct": 1.0},
        ]
    )
    problems = validate_stage_specs(specs, subset_count=2)
    assert any("贴齐" in p for p in problems)
    assert any("subset_index" in p for p in problems)


def test_resolve_stage_index_boundaries():
    specs = [
        StageSpec(0, 0.0, 0.5, "low"),
        StageSpec(1, 0.5, 1.0, "high"),
    ]
    assert resolve_stage_index(specs, 0.0) == 0
    assert resolve_stage_index(specs, 0.499) == 0
    assert resolve_stage_index(specs, 0.5) == 1
    assert resolve_stage_index(specs, 1.0) == 1


def test_validate_five_stage_cover_and_resolve():
    specs = parse_stage_specs(
        [
            {"subset_index": 0, "start_pct": 0.0, "end_pct": 0.2},
            {"subset_index": 1, "start_pct": 0.2, "end_pct": 0.4},
            {"subset_index": 0, "start_pct": 0.4, "end_pct": 0.6},
            {"subset_index": 2, "start_pct": 0.6, "end_pct": 0.8},
            {"subset_index": 1, "start_pct": 0.8, "end_pct": 1.0},
        ]
    )
    assert validate_stage_specs(specs, subset_count=3) == []
    assert resolve_stage_index(specs, 0.0) == 0
    assert resolve_stage_index(specs, 0.2) == 1
    assert resolve_stage_index(specs, 0.599) == 2
    assert resolve_stage_index(specs, 1.0) == 4


def test_active_subset_for_step():
    args = SimpleNamespace(
        stage_schedule_enabled=True,
        max_train_steps=1000,
        stage_schedule=[
            {"subset_index": 0, "start_pct": 0.0, "end_pct": 0.3},
            {"subset_index": 2, "start_pct": 0.3, "end_pct": 1.0},
        ],
    )
    assert stage_schedule_enabled(args)
    assert active_subset_indices_for_step(args, 0) == {0}
    assert active_subset_indices_for_step(args, 299) == {0}
    assert active_subset_indices_for_step(args, 300) == {2}
    assert progress_from_steps(300, 1000) == 0.3


def test_runtime_target_groups_keep_trigger_clone_with_source_row():
    args = SimpleNamespace(
        stage_schedule_enabled=True,
        max_train_steps=100,
        stage_schedule=[
            {"subset_index": 0, "start_pct": 0.0, "end_pct": 0.5},
            {"subset_index": 1, "start_pct": 0.5, "end_pct": 1.0},
        ],
        stage_schedule_target_groups=[[0, 1], [2]],
    )

    assert normalize_stage_target_groups(args.stage_schedule_target_groups) == [
        (0, 1),
        (2,),
    ]
    assert stage_target_count(args, object()) == 2
    assert active_subset_indices_for_step(args, 0) == {0, 1}
    assert active_subset_indices_for_step(args, 50) == {2}


def test_attach_stage_schedule_copies_runtime_target_groups():
    args = SimpleNamespace()

    attach_stage_schedule_from_config(
        args,
        {
            "stage_schedule_enabled": True,
            "stage_schedule": [
                {"subset_index": 0, "start_pct": 0.0, "end_pct": 1.0}
            ],
            "stage_schedule_target_groups": [[0, 1]],
        },
    )

    assert args.stage_schedule_enabled is True
    assert args.stage_schedule_target_groups == [[0, 1]]
    runtime_args = SimpleNamespace(**vars(args), max_train_steps=10)
    assert active_subset_indices_for_step(runtime_args, 0) == {0, 1}


def test_disabled_when_flag_off():
    args = SimpleNamespace(
        stage_schedule_enabled=False,
        max_train_steps=100,
        stage_schedule=[{"subset_index": 0, "start_pct": 0, "end_pct": 1}],
    )
    assert not stage_schedule_enabled(args)
    assert active_subset_indices_for_step(args, 50) is None


class _FakeInfo:
    def __init__(self, key: str, repeats: int = 1, is_reg: bool = False):
        self.image_key = key
        self.num_repeats = repeats
        self.is_reg = is_reg
        self.bucket_reso = (64, 64)
        self.image_size = (64, 64)
        self.resized_size = (64, 64)
        self.absolute_path = key
        self.mask_path = None
        self.preloaded_alpha_mask = None
        self.resize_interpolation = None


class _FakeSubset:
    def __init__(self, name: str):
        self.name = name
        self.sample_ratio = 1.0
        self.random_crop = False


class _FakeBucketDataset:
    """Minimal stand-in for DatasetBucketsMixin rebuild/snapshot semantics."""

    def __init__(self):
        self.subsets = [_FakeSubset("a"), _FakeSubset("b")]
        self.image_data = {
            "a1": _FakeInfo("a1"),
            "a2": _FakeInfo("a2"),
            "b1": _FakeInfo("b1"),
        }
        self.image_to_subset = {
            "a1": self.subsets[0],
            "a2": self.subsets[0],
            "b1": self.subsets[1],
        }
        self._all_image_data = None
        self._all_image_to_subset = None
        self._constant_token_buckets = True
        self.bucket_manager = None
        self._largest_bucket_index = None
        self.num_train_images = 3
        self.num_reg_images = 0
        self._length = 3
        self.seed = 0
        self.current_epoch = 0
        self.batch_size = 1
        self.enable_bucket = False
        self.resolution = 64
        self.min_bucket_reso = 64
        self.max_bucket_reso = 64
        self.bucket_reso_steps = 64
        self.bucket_no_upscale = False
        self.buckets_indices = []

    def snapshot_full_image_data(self, *, force: bool = False):
        if not force and self._all_image_data is not None:
            return
        self._all_image_data = dict(self.image_data)
        self._all_image_to_subset = dict(self.image_to_subset)

    def has_full_image_data_snapshot(self) -> bool:
        return self._all_image_data is not None

    def make_buckets(self, constant_token_buckets: bool = False):
        # Lightweight: one index entry per image key (no real bucket manager).
        self.buckets_indices = list(self.image_data.keys())
        self._length = len(self.buckets_indices)

    def rebuild_buckets_for_subsets(self, active_subset_indices=None) -> bool:
        # Mirror production: filter from snapshot, not from already-filtered live map.
        if not self.has_full_image_data_snapshot():
            if not self.image_data:
                return False
            self.snapshot_full_image_data()
        if active_subset_indices is None:
            allowed = None
        else:
            allowed_ids = {int(i) for i in active_subset_indices}
            allowed = {
                subset
                for index, subset in enumerate(self.subsets)
                if index in allowed_ids
            }
            if not allowed:
                return False
        source = self._all_image_data
        source_map = self._all_image_to_subset
        if allowed is None:
            filtered = source
            filtered_map = source_map
        else:
            filtered = {
                k: v for k, v in source.items() if source_map.get(k) in allowed
            }
            filtered_map = {k: source_map[k] for k in filtered}
        if not filtered:
            return False
        self.image_data = dict(filtered)
        self.image_to_subset = dict(filtered_map)
        self.num_train_images = len(filtered)
        self.make_buckets()
        return True


def test_snapshot_before_filter_allows_later_stage_recovery():
    """P0 regression: filtering stage0 first must not freeze stage0 as full set."""
    from library.training.stage_schedule import (
        apply_active_subsets_to_dataset,
        snapshot_full_image_data,
    )

    ds = _FakeBucketDataset()
    # Correct order: snapshot full map, then filter stage0, then stage1.
    snapshot_full_image_data(ds, force=True)
    assert apply_active_subsets_to_dataset(ds, {0}) is True
    assert set(ds.image_data) == {"a1", "a2"}
    assert apply_active_subsets_to_dataset(ds, {1}) is True
    assert set(ds.image_data) == {"b1"}
    # Restore full.
    assert apply_active_subsets_to_dataset(ds, None) is True
    assert set(ds.image_data) == {"a1", "a2", "b1"}


def test_partial_snapshot_blocks_later_stage_recovery():
    """If the frozen map was captured after a partial filter, later stages fail."""
    from library.training.stage_schedule import apply_active_subsets_to_dataset

    ds = _FakeBucketDataset()
    # Corrupt order: shrink live map, then force-snapshot that partial map.
    ds.image_data = {"a1": ds.image_data["a1"], "a2": ds.image_data["a2"]}
    ds.image_to_subset = {
        "a1": ds.image_to_subset["a1"],
        "a2": ds.image_to_subset["a2"],
    }
    ds.snapshot_full_image_data(force=True)
    assert set(ds._all_image_data) == {"a1", "a2"}
    assert apply_active_subsets_to_dataset(ds, {1}) is False
    assert set(ds.image_data) == {"a1", "a2"}


class _FakeGroupMember(_FakeBucketDataset):
    """One DatasetGroup member with a single local subset (WebUI multi-[[datasets]])."""

    def __init__(self, name: str, keys: list[str]):
        super().__init__()
        self.subsets = [_FakeSubset(name)]
        self.image_data = {k: _FakeInfo(k) for k in keys}
        self.image_to_subset = {k: self.subsets[0] for k in keys}
        self.num_train_images = len(keys)
        self._length = len(keys)
        self.buckets_indices = list(keys)
        self._all_image_data = None
        self._all_image_to_subset = None


class _FakeDatasetGroup:
    def __init__(self, members):
        self.datasets = list(members)
        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0
        self.refresh_concat_state()

    def refresh_concat_state(self):
        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0
        for ds in self.datasets:
            self.image_data.update(ds.image_data)
            self.num_train_images += ds.num_train_images
            self.num_reg_images += getattr(ds, "num_reg_images", 0)


def test_apply_active_dataset_members_for_multi_dataset_group():
    """Plan A: stage index selects DatasetGroup members, not local subset slots.

    kesul-style layout: 3 members × 1 local subset each. UI rows 0/1/2 must map
    to member0/1/2. Broadcasting local subset index 1/2 would empty every member.
    """
    from library.training.stage_schedule import (
        apply_active_subsets_to_dataset,
        snapshot_full_image_data,
    )

    m0 = _FakeGroupMember("res1024", ["a1", "a2"])
    m1 = _FakeGroupMember("res512", ["b1"])
    m2 = _FakeGroupMember("res1536", ["c1", "c2", "c3"])
    group = _FakeDatasetGroup([m0, m1, m2])
    snapshot_full_image_data(group, force=True)

    assert apply_active_subsets_to_dataset(group, {0}) is True
    assert set(group.image_data) == {"a1", "a2"}
    assert set(m0.image_data) == {"a1", "a2"}
    assert set(m1.image_data) == set()
    assert set(m2.image_data) == set()

    assert apply_active_subsets_to_dataset(group, {1}) is True
    assert set(group.image_data) == {"b1"}

    assert apply_active_subsets_to_dataset(group, {2}) is True
    assert set(group.image_data) == {"c1", "c2", "c3"}

    # Restore full set
    assert apply_active_subsets_to_dataset(group, None) is True
    assert set(group.image_data) == {"a1", "a2", "b1", "c1", "c2", "c3"}


def test_region_to_full_stage_switch_keeps_distinct_member_contracts():
    """Region/full rows may share images but must remain distinct stage members."""
    from library.training.stage_schedule import (
        apply_active_subsets_to_dataset,
        snapshot_full_image_data,
    )

    region = _FakeGroupMember("region-cache", ["same-image"])
    full = _FakeGroupMember("full-cache", ["same-image"])
    region.subsets[0].mask_dir = "post_image_dataset/masks"
    region.subsets[0].alpha_mask = True
    region.subsets[0].cache_dir = "post_image_dataset/lora_region"
    full.subsets[0].mask_dir = ""
    full.subsets[0].alpha_mask = False
    full.subsets[0].cache_dir = "post_image_dataset/lora_full"
    group = _FakeDatasetGroup([region, full])
    snapshot_full_image_data(group, force=True)

    assert apply_active_subsets_to_dataset(group, {0}) is True
    assert region.num_train_images == 1
    assert full.num_train_images == 0
    assert region.subsets[0].cache_dir != full.subsets[0].cache_dir

    assert apply_active_subsets_to_dataset(group, {1}) is True
    assert region.num_train_images == 0
    assert full.num_train_images == 1
    assert full.subsets[0].mask_dir == ""
    assert full.subsets[0].alpha_mask is False


def test_count_stage_targets_for_dataset_group_members():
    """Validation budget should use member count for multi-[[datasets]] groups."""
    from library.training.stage_schedule import count_stage_targets

    m0 = _FakeGroupMember("res1024", ["a1"])
    m1 = _FakeGroupMember("res512", ["b1"])
    m2 = _FakeGroupMember("res1536", ["c1"])
    group = _FakeDatasetGroup([m0, m1, m2])
    assert count_stage_targets(group) == 3

    # Single dataset with multiple local subsets still counts local subsets.
    leaf = _FakeBucketDataset()
    assert count_stage_targets(leaf) == 2


def test_six_thousand_steps_three_equal_stages():
    """S=6000, three equal stages: boundaries at 0/2000/4000/6000."""
    args = SimpleNamespace(
        stage_schedule_enabled=True,
        max_train_steps=6000,
        stage_schedule=[
            {"subset_index": 0, "start_pct": 0.0, "end_pct": 1 / 3},
            {"subset_index": 1, "start_pct": 1 / 3, "end_pct": 2 / 3},
            {"subset_index": 2, "start_pct": 2 / 3, "end_pct": 1.0},
        ],
    )
    assert active_subset_indices_for_step(args, 0) == {0}
    assert active_subset_indices_for_step(args, 1999) == {0}
    assert active_subset_indices_for_step(args, 2000) == {1}
    assert active_subset_indices_for_step(args, 3999) == {1}
    assert active_subset_indices_for_step(args, 4000) == {2}
    assert active_subset_indices_for_step(args, 5999) == {2}


def test_group_member_switch_updates_concat_length():
    """Emptied members must report zero length so ConcatDataset/DataLoader shrink."""
    from library.training.stage_schedule import (
        apply_active_subsets_to_dataset,
        snapshot_full_image_data,
    )

    class _LenMember(_FakeGroupMember):
        def __len__(self):
            return int(getattr(self, "_length", 0) or 0)

    class _LenGroup(_FakeDatasetGroup):
        def __init__(self, members):
            super().__init__(members)
            self.cumulative_sizes = []
            self.refresh_concat_state()

        def refresh_concat_state(self):
            super().refresh_concat_state()
            # Mirror torch ConcatDataset.cumsum on member lengths.
            total = 0
            sizes = []
            for ds in self.datasets:
                total += len(ds)
                sizes.append(total)
            self.cumulative_sizes = sizes

        def __len__(self):
            return int(self.cumulative_sizes[-1]) if self.cumulative_sizes else 0

    m0 = _LenMember("res1024", ["a1", "a2"])  # len 2
    m1 = _LenMember("res512", ["b1"])  # len 1
    m2 = _LenMember("res1536", ["c1", "c2", "c3"])  # len 3
    group = _LenGroup([m0, m1, m2])
    snapshot_full_image_data(group, force=True)
    assert len(group) == 6

    assert apply_active_subsets_to_dataset(group, {1}) is True
    assert len(m0) == 0
    assert len(m1) == 1
    assert len(m2) == 0
    assert len(group) == 1
    assert set(group.image_data) == {"b1"}


def test_empty_group_members_do_not_reuse_stale_bucket_indices():
    """Inactive members must stay length zero even with a populated bucket manager."""
    from library.training.stage_schedule import (
        apply_active_subsets_to_dataset,
        snapshot_full_image_data,
    )

    class _StaleBucketMember(_FakeGroupMember):
        def __init__(self, name: str, keys: list[str]):
            super().__init__(name, keys)
            self.bucket_manager = SimpleNamespace(buckets=[list(keys)])
            self.bucket_info = {"buckets": {0: {"count": len(keys)}}}
            self._largest_bucket_index = 0
            self.make_buckets_calls = 0

        def __len__(self):
            return int(getattr(self, "_length", 0) or 0)

        def make_buckets(self, constant_token_buckets: bool = False):
            self.make_buckets_calls += 1
            if not self.image_data and self.bucket_manager is not None:
                self.buckets_indices = list(self.bucket_manager.buckets[0])
                self._length = len(self.buckets_indices)
                return
            super().make_buckets(constant_token_buckets=constant_token_buckets)

    class _LenGroup(_FakeDatasetGroup):
        def refresh_concat_state(self):
            super().refresh_concat_state()
            total = 0
            self.cumulative_sizes = []
            for ds in self.datasets:
                total += len(ds)
                self.cumulative_sizes.append(total)

        def __len__(self):
            return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    inactive = _StaleBucketMember("res1024", ["a1", "a2"])
    active = _StaleBucketMember("res512", ["b1"])
    group = _LenGroup([inactive, active])
    snapshot_full_image_data(group, force=True)

    assert apply_active_subsets_to_dataset(group, {1}) is True
    assert inactive.image_data == {}
    assert inactive.buckets_indices == []
    assert inactive.bucket_manager is None
    assert inactive._largest_bucket_index is None
    assert inactive.make_buckets_calls == 0
    assert len(inactive) == 0
    assert len(group) == 1


def test_empty_stage_member_bucket_shuffle_is_noop():
    """Epoch propagation must tolerate inactive members without a bucket manager."""
    from library.datasets.dataset_buckets import DatasetBucketsMixin

    member = DatasetBucketsMixin()
    member.seed = 42
    member.current_epoch = 1
    member.buckets_indices = []
    member.bucket_manager = None
    member._largest_bucket_index = 3

    member.shuffle_buckets()

    assert member.buckets_indices == []
    assert member.bucket_manager is None
    assert member._largest_bucket_index is None


def test_stage_switch_prepares_rebuilt_dataloader_with_accelerator(monkeypatch):
    """Rebuilt stage loaders must retain Accelerate device placement/sharding."""
    from library.training import loop as training_loop

    raw_loader = object()
    prepared_loader = object()

    class _Accelerator:
        def __init__(self):
            self.prepared = []

        def prepare_data_loader(self, loader):
            self.prepared.append(loader)
            return prepared_loader

        def print(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        training_loop,
        "apply_active_subsets_to_dataset",
        lambda _dataset, _active: True,
    )
    monkeypatch.setattr(
        training_loop.torch.utils.data,
        "DataLoader",
        lambda *_args, **_kwargs: raw_loader,
    )

    accelerator = _Accelerator()
    state = SimpleNamespace(
        args=SimpleNamespace(
            stage_schedule_enabled=True,
            stage_schedule=[
                {
                    "name": "first",
                    "subset_index": 0,
                    "start_pct": 0.0,
                    "end_pct": 1.0,
                }
            ],
            max_train_steps=10,
        ),
        accelerator=accelerator,
        train_dataset_group=object(),
        dataloader_kwargs={"batch_size": 1},
        train_dataloader=object(),
        global_step=0,
        stage_index=-1,
    )

    training_loop._maybe_apply_stage_schedule(state, force=True)

    assert accelerator.prepared == [raw_loader]
    assert state.train_dataloader is prepared_loader
    assert state.stage_index == 0


def test_stage_epoch_recycles_short_loader_to_full_dataset_budget(monkeypatch):
    """A short active stage must still consume the full staged epoch budget."""
    from library.training import loop as training_loop

    calls = {"steps": 0}
    monkeypatch.setattr(training_loop, "_maybe_apply_stage_schedule", lambda *_a, **_k: None)
    monkeypatch.setattr(training_loop, "_profiler_step_begin", lambda *_a, **_k: None)
    monkeypatch.setattr(training_loop, "_profiler_step_end", lambda *_a, **_k: None)
    # E3 起 epoch 循环按微批驱动 ``_run_microbatch``，窗口提交在 ``_commit_step_boundary``；
    # 本用例只关心 loader 回收，因此把这两处都换成计数/空实现。
    monkeypatch.setattr(
        training_loop,
        "_run_microbatch",
        lambda *_a, **_k: calls.__setitem__("steps", calls["steps"] + 1) or torch.tensor(0.0),
    )
    monkeypatch.setattr(training_loop, "_commit_step_boundary", lambda *_a, **_k: None)
    monkeypatch.setattr(
        training_loop,
        "_maybe_scale_norm",
        lambda *_a, **_k: (False, None, None, None),
    )
    for name in (
        "_record_recent_step_seconds",
        "_sample_at_step",
        "_log_step",
        "_maybe_run_step_validation",
    ):
        monkeypatch.setattr(training_loop, name, lambda *_a, **_k: None)

    state = SimpleNamespace(
        args=SimpleNamespace(
            max_train_steps=3,
            _stage_num_update_steps_per_epoch=3,
            stage_schedule_enabled=True,
            stage_schedule=[
                {
                    "name": "only",
                    "subset_index": 0,
                    "start_pct": 0.0,
                    "end_pct": 1.0,
                }
            ],
        ),
        accelerator=SimpleNamespace(sync_gradients=True),
        initial_step=0,
        train_dataloader=[object()],
        train_dataset_group=object(),
        stage_index=0,
        current_step=SimpleNamespace(value=0),
        global_step=0,
        progress_bar=SimpleNamespace(update=lambda *_a, **_k: None),
        saver=SimpleNamespace(maybe_save_step=lambda *_a, **_k: None),
        network=object(),
        optimizer_train_fn=lambda: None,
    )

    training_loop._run_epoch_steps(object(), state, epoch=0)

    assert calls["steps"] == 3
    assert state.global_step == 3


def test_stage_epoch_rejects_empty_loader_instead_of_recycling_forever(monkeypatch):
    from library.training import loop as training_loop

    monkeypatch.setattr(training_loop, "_maybe_apply_stage_schedule", lambda *_a, **_k: None)
    state = SimpleNamespace(
        args=SimpleNamespace(
            max_train_steps=100,
            _stage_num_update_steps_per_epoch=100,
        ),
        accelerator=SimpleNamespace(),
        initial_step=0,
        train_dataloader=[],
        stage_index=0,
        global_step=50,
    )

    with pytest.raises(RuntimeError, match="no optimizer updates"):
        training_loop._run_epoch_steps(object(), state, epoch=0)
