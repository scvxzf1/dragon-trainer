"""Strict, read-only latent verification before skipping an unused PP VAE."""

from typing import Any


def krea2_runtime_latents_complete(cfg: dict[str, Any]) -> bool:
    from library import cache_identity
    from library.anima.strategy import AnimaLatentsCachingStrategy
    from web.services.config.cache_audit import (
        _bool_value, _bucket_reso, _resolved_mask_config, _validate_latent,
    )
    from web.services.config.dataset_media import _dataset_image_files
    from web.services.config.datasets import _dataset_rows_for_estimate
    from web.services.config.metadata import DATASET_IMAGE_EXTS
    from web.services.config.preflight_runtime import _resolve_project_path

    strategy = AnimaLatentsCachingStrategy(
        True, 1, False,
        cache_identity=cache_identity.latent_cache_identity(
            family="krea2_raw", vae=cfg.get("vae"),
        ),
        require_cache_identity=_bool_value(cfg.get("require_cache_identity"), False),
    )
    rows = _dataset_rows_for_estimate(cfg)
    if not rows:
        return False
    try:
        for row in rows:
            image_dir = _resolve_project_path(str(row.get("image_dir") or row.get("source_dir") or ""))
            cache_dir = _resolve_project_path(str(row["cache_dir"])) if row.get("cache_dir") else None
            images = _dataset_image_files(
                image_dir, DATASET_IMAGE_EXTS,
                recursive=_bool_value(row.get("recursive"), True),
                path_pattern=str(row.get("path_pattern") or "") or None,
            )
            if not images:
                return False
            mask = _resolved_mask_config(row, _resolve_project_path)
            for image in images:
                status, _ = _validate_latent(
                    image, image_dir=image_dir, cache_dir=cache_dir,
                    suffix=strategy.cache_suffix, strategy=strategy,
                    flip_aug=_bool_value(row.get("flip_aug"), False),
                    embedded_alpha=mask.mode == "embedded",
                    bucket_reso=_bucket_reso(image, row, "krea2_raw"),
                )
                if status != "valid":
                    return False
    except (OSError, ValueError):
        return False
    return True
