"""Family-neutral helpers for the legacy NPZ latent cache contract."""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from library import cache_identity as _identity

logger = logging.getLogger(__name__)


class GenericLatentCacheStrategy:
    """Reusable NPZ cache mechanics independent of a VAE family.

    Family strategies keep ownership of cache suffixes, latent encoding and
    shape-specific validation. This mixin owns only the shared sidecar layout:
    optional resolution keys, flip latents, alpha masks, and VAE identity.
    """

    def _default_is_disk_cached_latents_expected(
        self,
        latents_stride: int,
        bucket_reso: Tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        apply_alpha_mask: bool,
        multi_resolution: bool = False,
    ) -> bool:
        if not self.cache_to_disk or not os.path.exists(npz_path):
            return False

        expected_latents_size = (
            bucket_reso[1] // latents_stride,
            bucket_reso[0] // latents_stride,
        )
        key_reso_suffix = (
            f"_{expected_latents_size[0]}x{expected_latents_size[1]}"
            if multi_resolution
            else ""
        )
        return self._disk_cache_has_required_latent_keys(
            npz_path,
            key_reso_suffix,
            flip_aug,
            apply_alpha_mask,
            cache_identity=self.cache_identity,
            require_cache_identity=self.require_cache_identity,
        )

    @staticmethod
    def _disk_cache_has_required_latent_keys(
        npz_path: str,
        key_reso_suffix: str,
        flip_aug: bool,
        apply_alpha_mask: bool,
        *,
        cache_identity: Optional[dict] = None,
        require_cache_identity: bool = False,
    ) -> bool:
        try:
            with np.load(npz_path) as npz:
                keys = set(npz.files)
                recorded = _identity.read_npz_stamp(npz)

            if cache_identity is not None:
                state = _identity.evaluate(
                    recorded,
                    cache_identity,
                    require_identity=require_cache_identity,
                )
                if state == "stale":
                    return False
                if _identity.compare(recorded, cache_identity) == "absent":
                    _identity.warn_legacy("VAE latent", npz_path)

            if "latents" + key_reso_suffix not in keys:
                return False
            if flip_aug and "latents_flipped" + key_reso_suffix not in keys:
                return False
            if apply_alpha_mask and "alpha_mask" + key_reso_suffix not in keys:
                return False
        except Exception:
            logger.exception("Error loading latent cache: %s", npz_path)
            raise
        return True

    def _default_cache_batch_latents(
        self,
        encode_by_vae: Callable,
        vae_device: torch.device,
        vae_dtype: torch.dtype,
        image_infos: List,
        flip_aug: bool,
        apply_alpha_mask: bool,
        random_crop: bool,
        multi_resolution: bool = False,
    ) -> None:
        from library.datasets.image_utils import load_images_and_masks_for_caching

        img_tensor, alpha_masks, original_sizes, crop_ltrbs = (
            load_images_and_masks_for_caching(
                image_infos, apply_alpha_mask, random_crop
            )
        )
        img_tensor = img_tensor.to(device=vae_device, dtype=vae_dtype)

        with torch.no_grad():
            latents_tensors = encode_by_vae(img_tensor).to("cpu")
        if flip_aug:
            img_tensor = torch.flip(img_tensor, dims=[3])
            with torch.no_grad():
                flipped_latents = encode_by_vae(img_tensor).to("cpu")
        else:
            flipped_latents = [None] * len(latents_tensors)

        for info, latents, flipped_latent, alpha_mask, original_size, crop_ltrb in zip(
            image_infos,
            latents_tensors,
            flipped_latents,
            alpha_masks,
            original_sizes,
            crop_ltrbs,
        ):
            latents_size = latents.shape[-2:]
            key_reso_suffix = (
                f"_{latents_size[0]}x{latents_size[1]}"
                if multi_resolution
                else ""
            )
            if self.cache_to_disk:
                self.save_latents_to_disk(
                    info.latents_npz,
                    latents,
                    original_size,
                    crop_ltrb,
                    flipped_latent,
                    alpha_mask,
                    key_reso_suffix,
                    cache_identity=self.cache_identity,
                )
            else:
                info.latents_original_size = original_size
                info.latents_crop_ltrb = crop_ltrb
                info.latents = latents
                if flip_aug:
                    info.latents_flipped = flipped_latent
                info.alpha_mask = alpha_mask

    def _default_load_latents_from_disk(
        self,
        latents_stride: Optional[int],
        npz_path: str,
        bucket_reso: Tuple[int, int],
    ):
        if latents_stride is None:
            key_reso_suffix = ""
        else:
            latents_size = (
                bucket_reso[1] // latents_stride,
                bucket_reso[0] // latents_stride,
            )
            key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}"

        with np.load(npz_path) as npz:
            if "latents" + key_reso_suffix not in npz:
                raise ValueError(f"latents{key_reso_suffix} not found in {npz_path}")
            latents = npz["latents" + key_reso_suffix]
            original_size = npz["original_size" + key_reso_suffix].tolist()
            crop_ltrb = npz["crop_ltrb" + key_reso_suffix].tolist()
            flipped_latents = (
                npz["latents_flipped" + key_reso_suffix]
                if "latents_flipped" + key_reso_suffix in npz
                else None
            )
            alpha_mask = (
                npz["alpha_mask" + key_reso_suffix]
                if "alpha_mask" + key_reso_suffix in npz
                else None
            )
        return latents, original_size, crop_ltrb, flipped_latents, alpha_mask

    def load_latents_from_disk(self, npz_path: str, bucket_reso: Tuple[int, int]):
        return self._default_load_latents_from_disk(None, npz_path, bucket_reso)

    def save_latents_to_disk(
        self,
        npz_path,
        latents_tensor,
        original_size,
        crop_ltrb,
        flipped_latents_tensor=None,
        alpha_mask=None,
        key_reso_suffix="",
        cache_identity=None,
    ) -> None:
        kwargs = {}
        if os.path.exists(npz_path):
            with np.load(npz_path) as npz:
                previous = _identity.read_npz_stamp(npz)
                stale = (
                    cache_identity is not None
                    and _identity.compare(previous, cache_identity) == "mismatch"
                )
                if stale:
                    logger.warning(
                        "VAE identity changed for %s; dropping stale latent keys",
                        npz_path,
                    )
                else:
                    for key in npz.files:
                        kwargs[key] = npz[key]

        if cache_identity is not None:
            kwargs[_identity.NPZ_KEY] = _identity.encode_npz(cache_identity)
        kwargs["latents" + key_reso_suffix] = latents_tensor.float().cpu().numpy()
        kwargs["original_size" + key_reso_suffix] = np.array(original_size)
        kwargs["crop_ltrb" + key_reso_suffix] = np.array(crop_ltrb)
        if flipped_latents_tensor is not None:
            kwargs["latents_flipped" + key_reso_suffix] = (
                flipped_latents_tensor.float().cpu().numpy()
            )
        if alpha_mask is not None:
            kwargs["alpha_mask" + key_reso_suffix] = alpha_mask.float().cpu().numpy()
        np.savez(npz_path, **kwargs)

