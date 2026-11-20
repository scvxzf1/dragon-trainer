"""Latent / text-encoder cache helpers for BaseDataset.

Extracted from ``BaseDataset`` so caching completeness checks and disk/memory
cache writers live outside the core bucket + ``__getitem__`` surface.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, List, Tuple

from accelerate import Accelerator
from tqdm import tqdm

from library.runtime.device import clean_memory_on_device
from library.strategies import (
    LatentsCachingStrategy,
    TextEncoderOutputsCachingStrategy,
    TextEncodingStrategy,
    TokenizeStrategy,
)
from library.datasets import runtime_flags
from library.datasets.image_utils import (
    is_disk_cached_latents_is_expected,
    load_image,
)
from library.datasets.subsets import ImageInfo

logger = logging.getLogger(__name__)


def alpha_mask_required_for_cache(info: ImageInfo, subset: Any) -> bool:
    """Return whether an NPZ must contain an alpha mask for this image.

    A separate ``mask_dir`` is loaded and preloaded by the dataset bucket
    stage. In that mode the mask is authoritative outside the latent cache;
    requiring an ``alpha_mask`` NPZ key would reject valid dual-cache
    region/full curricula after preprocessing.

    External-mask mode (``mask_dir`` set) never requires an ``alpha_mask`` NPZ
    key: images with a mask file get it from the mask directory, and images
    without one train as ordinary full-image samples.
    """
    if not bool(getattr(subset, "alpha_mask", False)):
        return False
    if bool(getattr(subset, "mask_dir", None)):
        return False
    return True


class DatasetCacheMixin:
    """Mixin for VAE latent and text-encoder output caching."""

    def is_latent_cacheable(self):
        return all(
            [not subset.color_aug and not subset.random_crop for subset in self.subsets]
        )

    def is_text_encoder_output_cacheable(self, cache_supports_dropout: bool = False):
        return all(
            [
                not (
                    subset.caption_dropout_rate > 0
                    and not cache_supports_dropout
                    or subset.token_warmup_step > 0
                    or subset.caption_tag_dropout_rate > 0
                )
                for subset in self.subsets
            ]
        )

    def is_latents_cache_complete(self) -> bool:
        """True iff every image already has a valid on-disk latents cache.

        Read-only probe (no model, no GPU) used by the trainer to decide
        whether the VAE needs loading at all. Mirrors the per-file skip
        condition inside ``new_cache_latents``; honours ``skip_cache_check``
        via the strategy's ``is_disk_cached_latents_expected``.
        """
        caching_strategy = LatentsCachingStrategy.get_strategy()
        if caching_strategy is None or not caching_strategy.cache_to_disk:
            return False
        for info in self.image_data.values():
            subset = self.image_to_subset[info.image_key]
            if info.latents_npz is not None and not getattr(info, "reference_image_path", None):
                continue
            if info.latents_npz is None:
                npz_path = caching_strategy.get_latents_npz_path(
                    info.absolute_path,
                    info.image_size,
                    cache_dir=getattr(subset, "cache_dir", None),
                    image_dir=getattr(subset, "image_dir", None),
                )
            else:
                npz_path = info.latents_npz
            if not caching_strategy.is_disk_cached_latents_expected(
                info.bucket_reso,
                npz_path,
                subset.flip_aug,
                alpha_mask_required_for_cache(info, subset),
            ):
                return False
            reference_check = getattr(caching_strategy, "is_edit_reference_cache_expected", None)
            if reference_check is not None and not reference_check(info, subset):
                return False
        return True

    def is_text_encoder_outputs_cache_complete(self) -> bool:
        """True iff every image already has a valid on-disk text-encoder cache.

        Read-only probe (no model, no GPU) used by the trainer to decide
        whether the text encoder needs loading at all. Mirrors the per-file
        skip condition inside ``new_cache_text_encoder_outputs``.
        """
        caching_strategy = TextEncoderOutputsCachingStrategy.get_strategy()
        if caching_strategy is None or not caching_strategy.cache_to_disk:
            return False
        for info in self.image_data.values():
            subset = self.image_to_subset.get(info.image_key)
            path_for_info = getattr(caching_strategy, "get_outputs_npz_path_for_info", None)
            npz_path = (
                path_for_info(info, subset)
                if path_for_info is not None
                else caching_strategy.get_outputs_npz_path(
                    info.absolute_path,
                    cache_dir=(
                        getattr(subset, "text_cache_dir", None)
                        or getattr(subset, "cache_dir", None)
                    ),
                    image_dir=getattr(subset, "image_dir", None),
                )
            )
            expected_for_info = getattr(caching_strategy, "is_expected_for_info", None)
            cache_valid = (
                expected_for_info(npz_path, info)
                if expected_for_info is not None
                else caching_strategy.is_disk_cached_outputs_expected(npz_path)
            )
            if not cache_valid:
                return False
        return True

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        r"""
        a brand new method to cache latents. This method caches latents with caching strategy.
        normal cache_latents method is used by default, but this method is used when caching strategy is specified.
        """
        logger.info("caching latents with caching strategy.")
        caching_strategy = LatentsCachingStrategy.get_strategy()
        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution and some conditions
        class Condition:
            def __init__(self, reso, flip_aug, alpha_mask, random_crop):
                self.reso = reso
                self.flip_aug = flip_aug
                self.alpha_mask = alpha_mask
                self.random_crop = random_crop

            def __eq__(self, other):
                return (
                    other is not None
                    and self.reso == other.reso
                    and self.flip_aug == other.flip_aug
                    and self.alpha_mask == other.alpha_mask
                    and self.random_crop == other.random_crop
                )

        batch: List[ImageInfo] = []
        current_condition = None

        # support multiple-gpus
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        # define a function to submit a batch to cache
        def submit_batch(batch, cond):
            for info in batch:
                if info.image is not None and isinstance(info.image, Future):
                    info.image = info.image.result()  # future to image
            caching_strategy.cache_batch_latents(
                model, batch, cond.flip_aug, cond.alpha_mask, cond.random_crop
            )

            # remove image from memory
            for info in batch:
                info.image = None

        # define ThreadPoolExecutor to load images in parallel
        max_workers = min(os.cpu_count(), len(image_infos))
        max_workers = max(1, max_workers // num_processes)  # consider multi-gpu
        max_workers = min(
            max_workers, caching_strategy.batch_size
        )  # max_workers should be less than batch_size
        executor = ThreadPoolExecutor(max_workers)

        try:
            # iterate images
            logger.info("caching latents...")
            for i, info in enumerate(tqdm(image_infos, desc="Caching latents")):
                subset = self.image_to_subset[info.image_key]

                has_edit_reference = bool(getattr(info, "reference_image_path", None))
                if info.latents_npz is not None and not has_edit_reference:
                    continue

                # check disk cache exists and size of latents
                if caching_strategy.cache_to_disk:
                    if info.latents_npz is None:
                        info.latents_npz = caching_strategy.get_latents_npz_path(
                            info.absolute_path,
                            info.image_size,
                            cache_dir=getattr(subset, "cache_dir", None),
                            image_dir=getattr(subset, "image_dir", None),
                        )
                    reference_path_factory = getattr(
                        caching_strategy, "get_edit_reference_latent_path", None
                    )
                    if has_edit_reference and reference_path_factory is not None:
                        info.edit_reference_latent_path = reference_path_factory(info, subset)

                    # if the modulo of num_processes is not equal to process_index, skip caching
                    if i % num_processes != process_index:
                        continue

                    target_available = caching_strategy.is_disk_cached_latents_expected(
                        info.bucket_reso,
                        info.latents_npz,
                        subset.flip_aug,
                        alpha_mask_required_for_cache(info, subset),
                    )
                    reference_check = getattr(
                        caching_strategy, "is_edit_reference_cache_expected", None
                    )
                    reference_available = (
                        reference_check(info, subset)
                        if has_edit_reference and reference_check is not None
                        else True
                    )
                    cache_available = target_available and reference_available
                    if cache_available:  # do not add to batch
                        continue

                # if batch is not empty and condition is changed, flush the batch.
                condition = Condition(
                    info.bucket_reso,
                    subset.flip_aug,
                    subset.alpha_mask,
                    subset.random_crop,
                )
                if len(batch) > 0 and current_condition != condition:
                    submit_batch(batch, current_condition)
                    batch = []
                if condition != current_condition and runtime_flags.HIGH_VRAM:
                    clean_memory_on_device(accelerator.device)

                if info.latents_npz is None and info.image is None:
                    # load image in parallel
                    info.image = executor.submit(
                        load_image, info.absolute_path, condition.alpha_mask
                    )

                batch.append(info)
                current_condition = condition

                # if number of data in batch is enough, flush the batch
                if len(batch) >= caching_strategy.batch_size:
                    submit_batch(batch, current_condition)
                    batch = []

            if len(batch) > 0:
                submit_batch(batch, current_condition)

        finally:
            executor.shutdown()

    def cache_latents(
        self,
        vae,
        vae_batch_size=1,
        cache_to_disk=False,
        is_main_process=True,
        file_suffix=".npz",
    ):
        logger.info("caching latents.")

        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution and some conditions
        class Condition:
            def __init__(self, reso, flip_aug, alpha_mask, random_crop):
                self.reso = reso
                self.flip_aug = flip_aug
                self.alpha_mask = alpha_mask
                self.random_crop = random_crop

            def __eq__(self, other):
                return (
                    self.reso == other.reso
                    and self.flip_aug == other.flip_aug
                    and self.alpha_mask == other.alpha_mask
                    and self.random_crop == other.random_crop
                )

        batches: List[Tuple[Any, List[ImageInfo]]] = []
        batch: List[ImageInfo] = []
        current_condition = None

        logger.info("checking cache validity...")
        for info in tqdm(image_infos):
            subset = self.image_to_subset[info.image_key]

            if info.latents_npz is not None:  # fine tuning dataset
                continue

            # check disk cache exists and size of latents
            if cache_to_disk:
                info.latents_npz = os.path.splitext(info.absolute_path)[0] + file_suffix
                if not is_main_process:  # store to info only
                    continue

                cache_available = is_disk_cached_latents_is_expected(
                    info.bucket_reso,
                    info.latents_npz,
                    subset.flip_aug,
                    subset.alpha_mask,
                )

                if cache_available:  # do not add to batch
                    continue

            # if batch is not empty and condition is changed, flush the batch.
            condition = Condition(
                info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop
            )
            if len(batch) > 0 and current_condition != condition:
                batches.append((current_condition, batch))
                batch = []

            batch.append(info)
            current_condition = condition

            # if number of data in batch is enough, flush the batch
            if len(batch) >= vae_batch_size:
                batches.append((current_condition, batch))
                batch = []
                current_condition = None

        if len(batch) > 0:
            batches.append((current_condition, batch))

        if cache_to_disk and not is_main_process:
            return

        from library.datasets.image_utils import (
            cache_batch_latents as _cache_batch_latents,
        )

        # iterate batches: batch doesn't have image, image will be loaded in cache_batch_latents and discarded
        logger.info("caching latents...")
        for condition, batch in tqdm(batches, smoothing=1, total=len(batches)):
            _cache_batch_latents(
                vae,
                cache_to_disk,
                batch,
                condition.flip_aug,
                condition.alpha_mask,
                condition.random_crop,
            )

    def new_cache_text_encoder_outputs(
        self, models: List[Any], accelerator: Accelerator
    ):
        r"""
        a brand new method to cache text encoder outputs. This method caches text encoder outputs with caching strategy.
        """
        tokenize_strategy = TokenizeStrategy.get_strategy()
        text_encoding_strategy = TextEncodingStrategy.get_strategy()
        caching_strategy = TextEncoderOutputsCachingStrategy.get_strategy()
        batch_size = caching_strategy.batch_size or self.batch_size

        logger.info("caching Text Encoder outputs with caching strategy.")
        image_infos = list(self.image_data.values())

        # split by resolution
        batches = []
        batch = []

        # support multiple-gpus
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        logger.info("checking cache validity...")
        for i, info in enumerate(tqdm(image_infos, desc="Checking text cache")):
            subset = self.image_to_subset.get(info.image_key)
            # check disk cache exists and size of text encoder outputs
            if caching_strategy.cache_to_disk:
                path_for_info = getattr(caching_strategy, "get_outputs_npz_path_for_info", None)
                te_out_npz = (
                    path_for_info(info, subset)
                    if path_for_info is not None
                    else caching_strategy.get_outputs_npz_path(
                        info.absolute_path,
                        cache_dir=(
                            getattr(subset, "text_cache_dir", None)
                            or getattr(subset, "cache_dir", None)
                        ),
                        image_dir=getattr(subset, "image_dir", None),
                    )
                )
                info.text_encoder_outputs_npz = te_out_npz

                if i % num_processes != process_index:
                    continue

                expected_for_info = getattr(caching_strategy, "is_expected_for_info", None)
                cache_available = (
                    expected_for_info(te_out_npz, info)
                    if expected_for_info is not None
                    else caching_strategy.is_disk_cached_outputs_expected(te_out_npz)
                )
                if cache_available:
                    continue

            batch.append(info)

            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        if len(batches) == 0:
            logger.info("no Text Encoder outputs to cache")
            return

        # iterate batches
        logger.info("caching Text Encoder outputs...")
        for batch in tqdm(batches, smoothing=1, total=len(batches), desc="Caching text encoder"):
            caching_strategy.cache_batch_outputs(
                tokenize_strategy, models, text_encoding_strategy, batch
            )

    def cache_text_encoder_outputs(
        self,
        tokenizers,
        text_encoders,
        device,
        output_dtype,
        cache_to_disk=False,
        is_main_process=True,
    ):
        assert len(tokenizers) == 2, "only support SDXL"
        return self.cache_text_encoder_outputs_common(
            tokenizers,
            text_encoders,
            [device, device],
            output_dtype,
            [output_dtype],
            cache_to_disk,
            is_main_process,
        )

    def cache_text_encoder_outputs_sd3(
        self,
        tokenizer,
        text_encoders,
        devices,
        output_dtype,
        te_dtypes,
        cache_to_disk=False,
        is_main_process=True,
        batch_size=None,
    ):
        from library.datasets.image_utils import TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3

        return self.cache_text_encoder_outputs_common(
            [tokenizer],
            text_encoders,
            devices,
            output_dtype,
            te_dtypes,
            cache_to_disk,
            is_main_process,
            TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3,
            batch_size,
        )

    def cache_text_encoder_outputs_common(
        self,
        tokenizers,
        text_encoders,
        devices,
        output_dtype,
        te_dtypes,
        cache_to_disk=False,
        is_main_process=True,
        file_suffix=None,
        batch_size=None,
    ):
        from library.datasets.image_utils import TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX

        if file_suffix is None:
            file_suffix = TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX

        logger.info("caching text encoder outputs.")

        tokenize_strategy = TokenizeStrategy.get_strategy()

        if batch_size is None:
            batch_size = self.batch_size

        image_infos = list(self.image_data.values())

        logger.info("checking cache existence...")
        image_infos_to_cache = []
        for info in tqdm(image_infos):
            if cache_to_disk:
                te_out_npz = os.path.splitext(info.absolute_path)[0] + file_suffix
                info.text_encoder_outputs_npz = te_out_npz

                if not is_main_process:
                    continue

                if os.path.exists(te_out_npz):
                    continue

            image_infos_to_cache.append(info)

        if cache_to_disk and not is_main_process:
            return

        # prepare tokenizers and text encoders
        for text_encoder, device, te_dtype in zip(text_encoders, devices, te_dtypes):
            text_encoder.to(device)
            if te_dtype is not None:
                text_encoder.to(dtype=te_dtype)

        # create batch
        is_sd3 = len(tokenizers) == 1
        batch = []
        batches = []
        for info in image_infos_to_cache:
            if not is_sd3:
                input_ids1 = self.get_input_ids(info.caption, tokenizers[0])
                input_ids2 = self.get_input_ids(info.caption, tokenizers[1])
                batch.append((info, input_ids1, input_ids2))
            else:
                l_tokens, g_tokens, t5_tokens = tokenize_strategy.tokenize(info.caption)
                batch.append((info, l_tokens, g_tokens, t5_tokens))

            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        # iterate batches: call text encoder and cache outputs for memory or disk
        logger.info("caching text encoder outputs...")
        # Note: SD/SDXL/SD3 specific batch caching functions are not included in this stripped version.
        # Anima uses new_cache_text_encoder_outputs with caching strategy instead.
