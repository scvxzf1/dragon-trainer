"""Bucket assignment helpers for BaseDataset.

Owns resolution bucketing, alpha-mask preload, and epoch shuffle so the core
dataset class can stay focused on captioning and ``__getitem__``.
"""

from __future__ import annotations

import logging
import os
import random
import math
from typing import Any, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from library.datasets.buckets import BucketBatchIndex, BucketManager

logger = logging.getLogger(__name__)


def _repin_pinned_order(dataset: Any) -> int:
    """Re-pin the batch descriptor order after any reshuffle (measurement gate).

    Lazily imported so ``library.datasets`` never depends on ``library.training`` at
    import time. Returns ``0`` — and touches nothing — when the gate is closed.
    """

    from library.training.adaptive_runtime import pinned_order

    return pinned_order.repin(dataset)


class DatasetBucketsMixin:
    """Mixin for constant-token / resolution bucket management."""

    def make_buckets(self, constant_token_buckets: bool = False):
        """Assign every image to its nearest bucket resolution.

        With ``constant_token_buckets`` (the only training mode) buckets use
        the canonical 1024 table or a resolution-scaled variant for WebUI
        datasets such as 768 — native shapes, no padding.
        """
        self._constant_token_buckets = constant_token_buckets
        qwen = any(getattr(subset, "model_family", "") == "qwen_image_2_1" for subset in self.subsets)
        from library.datasets.qwen_image_geometry import align_qwen_bucket_manager, align_qwen_resolution
        logger.info("loading image sizes.")
        for info in tqdm(self.image_data.values()):
            if info.image_size is None:
                info.image_size = self.get_image_size(info.absolute_path)

        logger.info("make buckets")

        if self.bucket_manager is None:
            self.bucket_manager = BucketManager(
                max_reso=(self.resolution, self.resolution),
                min_size=self.min_bucket_reso,
                max_size=self.max_bucket_reso,
                reso_steps=self.bucket_reso_steps,
            )
            if self.enable_bucket and not self.bucket_no_upscale:
                self.bucket_manager.make_buckets(
                    constant_token_buckets=constant_token_buckets
                )
                if qwen:
                    align_qwen_bucket_manager(self.bucket_manager)
            else:
                self.bucket_manager.set_predefined_resos([])

        img_ar_errors = []
        for image_info in self.image_data.values():
            image_width, image_height = image_info.image_size
            if not self.enable_bucket:
                bucket_reso = (self.resolution, self.resolution)
                if qwen:
                    bucket_reso = align_qwen_resolution(bucket_reso)
                resized_size = self._cover_resized_size(
                    image_width, image_height, bucket_reso
                )
                ar_error = (bucket_reso[0] / bucket_reso[1]) - (image_width / image_height)
                self.bucket_manager.add_if_new_reso(bucket_reso)
                image_info.bucket_reso = bucket_reso
                image_info.resized_size = resized_size
            elif self.bucket_no_upscale:
                bucket_reso = (image_width, image_height)
                if qwen:
                    bucket_reso = align_qwen_resolution(bucket_reso)
                self.bucket_manager.add_if_new_reso(bucket_reso)
                image_info.bucket_reso = bucket_reso
                image_info.resized_size = self._cover_resized_size(image_width, image_height, bucket_reso)
                ar_error = 0
            else:
                image_info.bucket_reso, image_info.resized_size, ar_error = (
                    self.bucket_manager.select_bucket(image_width, image_height)
                )

            img_ar_errors.append(abs(ar_error))

        self.bucket_manager.sort()

        for image_info in self.image_data.values():
            for _ in range(image_info.num_repeats):
                self.bucket_manager.add_image(
                    image_info.bucket_reso, image_info.image_key
                )

        self.bucket_info = {"buckets": {}}
        logger.info("number of images (including repeats)")
        for i, (reso, bucket) in enumerate(
            zip(self.bucket_manager.resos, self.bucket_manager.buckets)
        ):
            count = len(bucket)
            if count > 0:
                self.bucket_info["buckets"][i] = {
                    "resolution": reso,
                    "count": len(bucket),
                }
                logger.info(f"bucket {i}: resolution {reso}, count: {len(bucket)}")

        if len(img_ar_errors) == 0:
            mean_img_ar_error = 0  # avoid NaN
        else:
            img_ar_errors = np.array(img_ar_errors)
            mean_img_ar_error = np.mean(np.abs(img_ar_errors))
        self.bucket_info["mean_img_ar_error"] = mean_img_ar_error
        logger.info(f"mean ar error (without repeats): {mean_img_ar_error}")

        # Drop incomplete last batches to keep batch dim constant for torch.compile,
        # but only when no subset uses sample_ratio (where every image matters more).
        has_sample_ratio = any(s.sample_ratio < 1.0 for s in self.subsets)
        self.buckets_indices: List[BucketBatchIndex] = []
        for bucket_index, bucket in enumerate(self.bucket_manager.buckets):
            if has_sample_ratio:
                batch_count = int(math.ceil(len(bucket) / self.batch_size))
            else:
                batch_count = len(bucket) // self.batch_size
            for batch_index in range(batch_count):
                self.buckets_indices.append(
                    BucketBatchIndex(bucket_index, self.batch_size, batch_index)
                )

        self.shuffle_buckets()
        self._length = len(self.buckets_indices)

        self._preload_alpha_masks()

    @staticmethod
    def _cover_resized_size(
        image_width: int,
        image_height: int,
        bucket_reso: Tuple[int, int],
    ) -> Tuple[int, int]:
        bucket_width, bucket_height = bucket_reso
        if image_width / image_height > bucket_width / bucket_height:
            return (round(bucket_height * image_width / image_height), bucket_height)
        return (bucket_width, round(bucket_width * image_height / image_width))

    def _preload_alpha_masks(self):
        """Load mask PNGs into memory once as uint8 [H, W] tensors at
        bucket_reso, so the dataloader hot path doesn't re-decode + resize a
        PNG on every fetch. Mask files generated by `make mask` are already at
        post-resize resolution (matches bucket_reso), so no resize is needed
        in the common case; we only resize as a safety net for stale masks.
        Skipped for subsets with random_crop=True since image size varies per
        fetch in that case.
        """
        targets = [
            info
            for info in self.image_data.values()
            if info.mask_path is not None
            and not self.image_to_subset[info.image_key].random_crop
        ]
        if not targets:
            return
        logger.info(f"preloading {len(targets)} alpha masks into memory...")
        n_resized = 0
        n_missing = 0
        for info in tqdm(targets, desc="preload masks"):
            if not os.path.exists(info.mask_path):
                n_missing += 1
                continue
            mask = Image.open(info.mask_path).convert("L")
            target_w, target_h = info.bucket_reso  # (W, H)
            if (mask.width, mask.height) != (target_w, target_h):
                mask = mask.resize((target_w, target_h), Image.NEAREST)
                n_resized += 1
            info.preloaded_alpha_mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        if n_missing:
            logger.warning(f"  {n_missing} mask files missing on disk")
        if n_resized:
            logger.info(
                f"  {n_resized} masks needed runtime resize (size != bucket_reso)"
            )

    def snapshot_full_image_data(self, *, force: bool = False) -> None:
        """Freeze the full image map before any stage filter is applied.

        Must be called while ``image_data`` still contains every subset.
        Subsequent ``rebuild_buckets_for_subsets`` filters from this snapshot,
        not from the already-filtered live map (otherwise later stages cannot
        recover subsets that were filtered out earlier).
        """
        if not force and getattr(self, "_all_image_data", None) is not None:
            return
        if not self.image_data:
            self._all_image_data = {}
            self._all_image_to_subset = {}
            return
        self._all_image_data = dict(self.image_data)
        self._all_image_to_subset = dict(self.image_to_subset)

    def has_full_image_data_snapshot(self) -> bool:
        return getattr(self, "_all_image_data", None) is not None

    def rebuild_buckets_for_subsets(self, active_subset_indices=None) -> bool:
        """Rebuild bucket indices using only images from selected subsets.

        ``active_subset_indices=None`` restores all registered images.
        Subset identity is the index in ``self.subsets``. Returns True when
        membership was applied successfully. Does not re-encode caches.

        Requires :meth:`snapshot_full_image_data` first. Lazily snapshotting
        here is intentionally rejected when no snapshot exists: bootstrap may
        already have filtered to stage 0, and a late snapshot would freeze that
        partial set as "full".
        """
        if not self.has_full_image_data_snapshot():
            # Safe only when live map is still the full set (no prior filter).
            # Callers that filter before loop must snapshot explicitly.
            if not self.image_data:
                return False
            self.snapshot_full_image_data()

        if active_subset_indices is None:
            allowed_subsets = None
        else:
            allowed = {int(i) for i in active_subset_indices}
            allowed_subsets = {
                subset
                for index, subset in enumerate(self.subsets)
                if index in allowed
            }
            if not allowed_subsets:
                logger.warning(
                    "rebuild_buckets_for_subsets: no subsets matched %s",
                    sorted(allowed),
                )
                return False

        source_data = self._all_image_data
        source_map = self._all_image_to_subset
        if allowed_subsets is None:
            filtered = source_data
            filtered_map = source_map
        else:
            filtered = {
                key: info
                for key, info in source_data.items()
                if source_map.get(key) in allowed_subsets
            }
            filtered_map = {
                key: source_map[key] for key in filtered if key in source_map
            }

        if not filtered:
            logger.warning("rebuild_buckets_for_subsets: filtered image set is empty")
            return False

        self.image_data = dict(filtered)
        self.image_to_subset = dict(filtered_map)
        self.num_train_images = sum(
            info.num_repeats
            for info in filtered.values()
            if not getattr(info, "is_reg", False)
        )
        self.num_reg_images = sum(
            info.num_repeats
            for info in filtered.values()
            if getattr(info, "is_reg", False)
        )

        # Reset bucket manager; make_buckets reassigns from image_data.
        constant_token = bool(getattr(self, "_constant_token_buckets", True))
        self.bucket_manager = None
        self._largest_bucket_index = None
        self.make_buckets(constant_token_buckets=constant_token)
        return True

    def shuffle_buckets(self):
        """Reshuffle the batch descriptor order, then honour the pinned-order gate.

        ``shuffle_buckets`` has several callers (``create_buckets`` while building,
        ``BaseDataset.set_current_epoch`` per epoch), and any of them can run after a
        loader was pinned. The re-pin therefore lives *here*, not in the callers, so a
        new caller cannot silently reintroduce a batch-size-dependent permutation
        (t39 / F-T35-1). No-op when ``ANIMA_MULTIGPU_PINNED_ORDER`` is unset.
        """
        self._shuffle_buckets_unpinned()
        _repin_pinned_order(self)

    def _shuffle_buckets_unpinned(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)

        random.shuffle(self.buckets_indices)
        # Stage scheduling intentionally empties inactive DatasetGroup members
        # and clears their stale bucket manager. Epoch propagation still visits
        # every member, so an empty member must be a valid no-op here.
        if self.bucket_manager is None:
            self._largest_bucket_index = None
            return
        self.bucket_manager.shuffle()
        self._largest_bucket_first()

    def _largest_bucket_first(self):
        """Pin one batch of the highest-token-count bucket to the front of the
        epoch order.

        With native-shape buckets each distinct token count traces its own
        ``torch.compile`` block graph, and the largest
        bucket also carries the biggest activations. Front-loading it forces
        that worst-case graph compile + peak allocation onto step 0, so a
        too-tight VRAM budget fails fast at start instead of OOMing mid-epoch
        when the big bucket happens to come up in the shuffle. Only the first
        batch is reordered; the rest of the epoch stays randomly shuffled.
        """
        if not self.buckets_indices:
            return
        # resos are (W, H); pixel area is the token-count proxy.
        if getattr(self, "_largest_bucket_index", None) is None:
            resos = self.bucket_manager.resos
            present = {bbi.bucket_index for bbi in self.buckets_indices}
            self._largest_bucket_index = max(
                present, key=lambda bi: resos[bi][0] * resos[bi][1]
            )
        for i, bbi in enumerate(self.buckets_indices):
            if bbi.bucket_index == self._largest_bucket_index:
                if i:
                    self.buckets_indices.insert(0, self.buckets_indices.pop(i))
                return
