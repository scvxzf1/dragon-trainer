"""Family-neutral helpers for the legacy NPZ latent cache contract."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch

from library import cache_identity as _identity
from library.models.family_registry import get_model_family_spec
from library.strategies.latent_geometry import validate_latent_batch_geometry, validate_latent_shape
from library.io.cache import resolve_cache_path

logger = logging.getLogger(__name__)


class GenericLatentCacheStrategy:
    """Reusable NPZ cache mechanics independent of a VAE family.

    Family strategies keep ownership of VAE encoding and shape-specific
    validation. Geometry and cache naming come from the registered latent
    space; this mixin owns resolution keys, flip latents, alpha masks, and VAE
    identity.
    """

    MODEL_FAMILY = ""

    @property
    def latent_space_spec(self):
        if not self.MODEL_FAMILY:
            raise RuntimeError("latent-cache strategy must declare MODEL_FAMILY")
        return get_model_family_spec(self.MODEL_FAMILY).latent_space

    @property
    def latents_stride(self) -> int:
        return self.latent_space_spec.vae_spatial_compression

    @property
    def cache_suffix(self) -> str:
        return self.latent_space_spec.cache_suffix

    def get_latents_npz_path(
        self,
        absolute_path: str,
        image_size: Tuple[int, int],
        cache_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
    ) -> str:
        suffix = f"_{image_size[0]:04d}x{image_size[1]:04d}{self.cache_suffix}"
        return resolve_cache_path(
            absolute_path, suffix, cache_dir=cache_dir, image_dir=image_dir
        )

    def get_image_size_from_disk_cache_path(
        self, absolute_path: str, npz_path: str
    ) -> Tuple[int, int]:
        del absolute_path
        stem = npz_path[: -len(self.cache_suffix)]
        width, height = stem.rsplit("_", 1)[-1].split("x")
        return int(width), int(height)

    def _default_is_disk_cached_latents_expected(
        self,
        bucket_reso: Tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        apply_alpha_mask: bool,
        multi_resolution: bool = False,
    ) -> bool:
        if not self.cache_to_disk or not os.path.exists(npz_path):
            return False

        expected_latents_size = (
            bucket_reso[1] // self.latents_stride,
            bucket_reso[0] // self.latents_stride,
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
            bucket_reso=bucket_reso,
            latent_space_spec=self.latent_space_spec,
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
        bucket_reso: Tuple[int, int],
        latent_space_spec,
        cache_identity: Optional[dict] = None,
        require_cache_identity: bool = False,
    ) -> bool:
        try:
            with np.load(npz_path) as npz:
                keys = set(npz.files)
                recorded = _identity.read_npz_stamp(npz)
                latent_key = "latents" + key_reso_suffix
                flipped_key = "latents_flipped" + key_reso_suffix
                required_keys = {
                    latent_key,
                    "original_size" + key_reso_suffix,
                    "crop_ltrb" + key_reso_suffix,
                }
                if flip_aug:
                    required_keys.add(flipped_key)
                if apply_alpha_mask:
                    required_keys.add("alpha_mask" + key_reso_suffix)
                if not required_keys.issubset(keys):
                    return False
                latent_shape = npz[latent_key].shape
                flipped_shape = npz[flipped_key].shape if flip_aug else None

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

            try:
                validate_latent_shape(
                    latent_shape,
                    bucket_reso,
                    latent_space_spec,
                    label=f"{npz_path} latent cache",
                )
                if flipped_shape is not None:
                    validate_latent_shape(
                        flipped_shape,
                        bucket_reso,
                        latent_space_spec,
                        label=f"{npz_path} flipped latent cache",
                    )
                    if flipped_shape != latent_shape:
                        return False
            except ValueError as exc:
                logger.warning("Invalid latent cache geometry in %s: %s", npz_path, exc)
                return False
        except Exception:
            logger.exception("Error loading latent cache: %s", npz_path)
            raise
        return True

    def _validate_encoded_latents(self, latents, image_infos, *, label: str) -> None:
        bucket_sizes = []
        for index, info in enumerate(image_infos):
            bucket_reso = getattr(info, "bucket_reso", None)
            if bucket_reso is None or len(bucket_reso) != 2:
                raise ValueError(
                    f"{self.MODEL_FAMILY} latent cache requires image_info.bucket_reso "
                    f"(width, height) at index {index}"
                )
            bucket_sizes.append(tuple(map(int, bucket_reso)))
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"{self.MODEL_FAMILY} VAE encoder must return a tensor")
        validate_latent_batch_geometry(
            latents, bucket_sizes, self.latent_space_spec,
            label=f"{self.MODEL_FAMILY} VAE encoder {label} latent",
        )

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
        self._validate_encoded_latents(latents_tensors, image_infos, label="encoded")
        if flip_aug:
            img_tensor = torch.flip(img_tensor, dims=[3])
            with torch.no_grad():
                flipped_latents = encode_by_vae(img_tensor).to("cpu")
            self._validate_encoded_latents(
                flipped_latents,
                image_infos,
                label="flipped",
            )
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
        npz_path: str,
        bucket_reso: Tuple[int, int],
        multi_resolution: bool = True,
    ):
        if multi_resolution:
            latents_size = (
                bucket_reso[1] // self.latents_stride,
                bucket_reso[0] // self.latents_stride,
            )
            key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}"
        else:
            key_reso_suffix = ""

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
        spec = self.latent_space_spec
        validate_latent_shape(latents.shape, bucket_reso, spec, label=f"{npz_path} latent")
        if flipped_latents is not None:
            validate_latent_shape(
                flipped_latents.shape, bucket_reso, spec, label=f"{npz_path} flipped latent"
            )
            if flipped_latents.shape != latents.shape:
                raise ValueError(f"{npz_path} flipped latent shape differs from the regular latent")
        return latents, original_size, crop_ltrb, flipped_latents, alpha_mask

    def load_latents_from_disk(self, npz_path: str, bucket_reso: Tuple[int, int]):
        return self._default_load_latents_from_disk(npz_path, bucket_reso)

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


class _LatentsCachingStrategyContract:
    _strategy = None

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        *,
        cache_identity: Optional[dict] = None,
        require_cache_identity: bool = False,
    ) -> None:
        self._cache_to_disk = cache_to_disk
        self._batch_size = batch_size
        self.skip_disk_cache_validity_check = skip_disk_cache_validity_check
        self.cache_identity = cache_identity
        self.require_cache_identity = bool(require_cache_identity)

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(
                f"Internal error. {cls.__name__} strategy is already set"
            )
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls):
        return cls._strategy

    @property
    def cache_to_disk(self):
        return self._cache_to_disk

    @property
    def batch_size(self):
        return self._batch_size

    def get_latents_npz_path(self, absolute_path, image_size, cache_dir=None, image_dir=None):
        raise NotImplementedError

    def is_disk_cached_latents_expected(self, bucket_reso, npz_path, flip_aug, alpha_mask):
        raise NotImplementedError

    def cache_batch_latents(self, model, batch, flip_aug, alpha_mask, random_crop):
        raise NotImplementedError


class LatentsCachingStrategy(GenericLatentCacheStrategy, _LatentsCachingStrategyContract):
    """Family-neutral public strategy contract used by dataset loaders."""


class FamilyLatentCacheStrategy(LatentsCachingStrategy):
    """Spec-driven cache strategy with only VAE encoding left to each family."""

    MODEL_FAMILY = ""

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        *,
        model_family: str,
        encode_by_vae: Callable,
        clean_after_cache: bool = False,
        cache_identity: Optional[dict] = None,
        require_cache_identity: bool = False,
    ) -> None:
        super().__init__(
            cache_to_disk,
            batch_size,
            skip_disk_cache_validity_check,
            cache_identity=cache_identity,
            require_cache_identity=require_cache_identity,
        )
        self.MODEL_FAMILY = model_family
        self._encode_by_vae = encode_by_vae
        self._clean_after_cache = bool(clean_after_cache)

    def is_disk_cached_latents_expected(
        self, bucket_reso, npz_path, flip_aug, alpha_mask
    ):
        return self._default_is_disk_cached_latents_expected(
            bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True
        )

    def encode_vae(self, vae, images):
        return self._encode_by_vae(vae, images)

    def cache_batch_latents(self, vae, image_infos, flip_aug, alpha_mask, random_crop):
        parameters = iter(vae.parameters())
        try:
            parameter = next(parameters)
        except StopIteration as exc:
            raise ValueError(f"{self.MODEL_FAMILY} VAE has no parameters") from exc
        self._default_cache_batch_latents(
            lambda images: self.encode_vae(vae, images).to("cpu"),
            parameter.device,
            parameter.dtype,
            image_infos,
            flip_aug,
            alpha_mask,
            random_crop,
            multi_resolution=True,
        )
        if self._clean_after_cache:
            from library.datasets import base as datasets_base
            from library.runtime.device import clean_memory_on_device

            if not datasets_base.HIGH_VRAM:
                clean_memory_on_device(parameter.device)
