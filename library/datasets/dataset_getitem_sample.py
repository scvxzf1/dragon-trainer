"""Per-sample loading helpers for BaseDataset.__getitem__."""

from __future__ import annotations

import logging
import os
import random
from typing import List

import numpy as np
import torch

from library.datasets.image_utils import trim_and_resize_if_required
from library.models.family_registry import get_model_family_spec

logger = logging.getLogger(__name__)


def with_runtime_caption_dropout_rate(text_encoder_outputs, rate: float):
    """Return cached outputs with the current subset's dropout rate.

    Text embeddings are reusable across runs, while whole-caption dropout is a
    training-time policy.  Cache writers therefore keep only a compatibility
    placeholder in the trailing slot; the active subset config is authoritative.
    """
    if not isinstance(text_encoder_outputs, (list, tuple)) or not text_encoder_outputs:
        return text_encoder_outputs
    outputs = list(text_encoder_outputs)
    outputs[-1] = torch.tensor(float(rate), dtype=torch.float32)
    return outputs


def _as_float32_tensor(array_or_tensor) -> torch.Tensor:
    if isinstance(array_or_tensor, torch.Tensor):
        return array_or_tensor.to(dtype=torch.float32)
    arr = np.asarray(array_or_tensor)
    if arr.dtype != np.float32 or not arr.flags.c_contiguous or not arr.flags.writeable:
        arr = np.array(arr, dtype=np.float32, copy=True, order="C")
    return torch.from_numpy(arr)


def _maybe_downsample_mask_to_latent(
    alpha_mask: torch.Tensor | None,
    latents: torch.Tensor | None,
    image: torch.Tensor | None,
) -> torch.Tensor | None:
    """Pre-downsample pixel masks in CPU workers when training on cached latents."""
    if alpha_mask is None or latents is None or image is not None:
        return alpha_mask
    lat_hw = tuple(int(d) for d in latents.shape[-2:])
    if tuple(int(d) for d in alpha_mask.shape[-2:]) == lat_hw:
        return alpha_mask
    return (
        torch.nn.functional.interpolate(
            alpha_mask.unsqueeze(0).unsqueeze(0), size=lat_hw, mode="area"
        )
        .squeeze(0)
        .squeeze(0)
    )


def _latent_target_size(latents: torch.Tensor, model_family: str) -> tuple[int, int]:
    """Convert cached latent spatial dimensions to source pixel dimensions."""
    height, width = (int(d) for d in latents.shape[-2:])
    compression = get_model_family_spec(
        model_family
    ).latent_space.vae_spatial_compression
    return width * compression, height * compression


def load_visual_sample(dataset, image_info, subset, flipped: bool):
    """Load image/latents/alpha/crop metadata for one training sample."""
    if image_info.latents is not None:
        original_size = image_info.latents_original_size
        crop_ltrb = image_info.latents_crop_ltrb
        if not flipped:
            latents = image_info.latents
            alpha_mask = image_info.alpha_mask
        else:
            latents = image_info.latents_flipped
            alpha_mask = (
                None
                if image_info.alpha_mask is None
                else torch.flip(image_info.alpha_mask, [1])
            )

        if dataset.force_load_images_for_ip:
            image = dataset._load_image_at_bucket(subset, image_info, flipped)
        else:
            image = None
    elif image_info.latents_npz is not None:
        latents, original_size, crop_ltrb, flipped_latents, alpha_mask = (
            dataset.latents_caching_strategy.load_latents_from_disk(
                image_info.latents_npz, image_info.bucket_reso
            )
        )
        if flipped:
            latents = flipped_latents
            alpha_mask = (
                None if alpha_mask is None else alpha_mask[:, ::-1].copy()
            )
            del flipped_latents
        latents = _as_float32_tensor(latents)
        if alpha_mask is not None:
            alpha_mask = _as_float32_tensor(alpha_mask)

        if dataset.force_load_images_for_ip:
            image = dataset._load_image_at_bucket(subset, image_info, flipped)
        else:
            image = None
    else:
        img, _, _, _, _ = dataset.load_image_with_face_info(
            subset, image_info.absolute_path, subset.alpha_mask
        )

        img, original_size, crop_ltrb = trim_and_resize_if_required(
            subset.random_crop,
            img,
            image_info.bucket_reso,
            image_info.resized_size,
            resize_interpolation=image_info.resize_interpolation,
        )

        aug = dataset.aug_helper.get_augmentor(subset.color_aug)
        if aug is not None:
            img_rgb = img[:, :, :3]
            img_rgb = aug(image=img_rgb)["image"]
            img[:, :, :3] = img_rgb

        if flipped:
            img = img[:, ::-1, :].copy()

        if image_info.mask_path is not None:
            if image_info.preloaded_alpha_mask is not None:
                # Will be filled in by the post-branch override below.
                alpha_mask = None
            else:
                from library.datasets.image_utils import load_mask_from_dir

                alpha_mask = load_mask_from_dir(
                    os.path.dirname(image_info.mask_path),
                    image_info.absolute_path,
                    (img.shape[1], img.shape[0]),
                )
                if alpha_mask is None:
                    alpha_mask = torch.ones(
                        (img.shape[0], img.shape[1]), dtype=torch.float32
                    )
                if flipped:
                    alpha_mask = torch.flip(alpha_mask, [1])
        elif subset.alpha_mask:
            if img.shape[2] == 4:
                alpha_mask = img[:, :, 3]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)
            else:
                alpha_mask = torch.ones(
                    (img.shape[0], img.shape[1]), dtype=torch.float32
                )
        else:
            alpha_mask = None

        img = img[:, :, :3]

        latents = None
        image = dataset.image_transforms(img)
        del img

    if image_info.preloaded_alpha_mask is not None:
        # mask_dir is the source of truth: override any alpha_mask coming
        # from the latent cache (npz / in-memory) or the raw-image branch.
        alpha_mask = image_info.preloaded_alpha_mask.float() / 255.0
        if flipped:
            alpha_mask = torch.flip(alpha_mask, [1])

    alpha_mask = _maybe_downsample_mask_to_latent(alpha_mask, latents, image)

    target_size = (
        (image.shape[2], image.shape[1])
        if image is not None
        else _latent_target_size(latents, subset.model_family)
    )

    if not flipped:
        crop_left_top = (crop_ltrb[0], crop_ltrb[1])
    else:
        crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

    reference_latent_loader = getattr(
        dataset.latents_caching_strategy, "load_edit_reference_latent", None
    )
    edit_reference_latent = (
        reference_latent_loader(image_info, subset)
        if reference_latent_loader is not None and image_info.reference_image_path
        else None
    )

    return {
        "image": image,
        "latents": latents,
        "alpha_mask": alpha_mask,
        "cond_latents": dataset._load_cond_latent(subset, image_info, flipped),
        "qwen_edit_reference_latent": edit_reference_latent,
        "original_size_hw": (int(original_size[1]), int(original_size[0])),
        "crop_top_left": (int(crop_left_top[1]), int(crop_left_top[0])),
        "target_size_hw": (int(target_size[1]), int(target_size[0])),
        "flipped": flipped,
    }



def resolve_ip_pair(dataset, image_info, subset, target_stem: str):
    """Resolve IP-Adapter reference stem / caption strip flags."""
    ip_ref_stem, ip_ref_subset, ip_ref_reldir = (
        None,
        subset,
        "",
    )
    ip_shuffled_stem = None
    strip_identity = False
    sampler = dataset.identity_pair_sampler
    if (
        sampler is not None
        and dataset.ip_features_cache_to_disk
        and sampler.has(target_stem)
    ):
        if dataset.ip_pair_is_validation:
            # Deterministic per target so the matched/shuffled deltas
            # are stable across epochs (the held-out gate).
            drng = random.Random(dataset.seed ^ (hash(target_stem) & 0xFFFFFFFF))
            ip_ref_stem, _ = sampler.resolve(target_stem, drng)
            ip_shuffled_stem, _ = sampler.shuffled(target_stem, drng)
        else:
            if random.random() < dataset.ip_pair_prob:
                ip_ref_stem, _ = sampler.resolve(target_stem, random)
            else:
                ip_ref_stem = target_stem  # self-pair in the mix
            strip_identity = (
                ip_ref_stem != target_stem
                and dataset.ip_pair_caption_strip_p > 0.0
                and random.random() < dataset.ip_pair_caption_strip_p
            )
        if ip_ref_stem and ip_ref_stem != target_stem:
            ip_ref_reldir = sampler.rel_dir(ip_ref_stem)
    return {
        "sampler": sampler,
        "ip_ref_stem": ip_ref_stem,
        "ip_ref_subset": ip_ref_subset,
        "ip_ref_reldir": ip_ref_reldir,
        "ip_shuffled_stem": ip_shuffled_stem,
        "strip_identity": strip_identity,
    }


def load_text_fields(dataset, image_info, subset, *, sampler, target_stem: str, strip_identity: bool):
    """Load caption / TE cache / token ids for one sample."""
    caption = image_info.caption
    if strip_identity:
        caption = dataset._strip_identity_tags(
            caption, sampler.image_meta.get(target_stem, {})
        )

    tokenization_required = (
        dataset.text_encoder_output_caching_strategy is None
        or dataset.text_encoder_output_caching_strategy.is_partial
    )
    # The caption-leakage strip only reaches the model when captions
    # are tokenized live. With cached TE outputs the model reads the
    # full (identity-bearing) embedding regardless, so the strip is
    # inert — warn once instead of silently doing nothing.
    if (
        sampler is not None
        and not dataset.ip_pair_is_validation
        and dataset.ip_pair_caption_strip_p > 0.0
        and not tokenization_required
        and image_info.text_encoder_outputs_npz is not None
        and not dataset._ip_pair_strip_warned
    ):
        dataset._ip_pair_strip_warned = True
        logger.warning(
            "[ip-pair] ip_pair_caption_strip_p>0 but text-encoder "
            "outputs are cached — the strip is inert. Set "
            "cache_text_encoder_outputs=false for the guard to take effect."
        )
    text_encoder_outputs = None
    input_ids = None

    if image_info.text_encoder_outputs is not None:
        text_encoder_outputs = image_info.text_encoder_outputs
        select_in_memory = getattr(
            dataset.text_encoder_output_caching_strategy,
            "select_in_memory_outputs",
            None,
        )
        if select_in_memory is not None:
            text_encoder_outputs = select_in_memory(text_encoder_outputs)
    elif image_info.text_encoder_outputs_npz is not None:
        text_encoder_outputs = (
            dataset.text_encoder_output_caching_strategy.load_outputs_npz(
                image_info.text_encoder_outputs_npz
            )
        )
    else:
        tokenization_required = True

    if text_encoder_outputs is not None:
        text_encoder_outputs = with_runtime_caption_dropout_rate(
            text_encoder_outputs,
            image_info.caption_dropout_rate,
        )

    if tokenization_required:
        caption = dataset.process_caption(subset, image_info.caption)
        input_ids = [ids[0] for ids in dataset.tokenize_strategy.tokenize(caption)]

    return {
        "caption": caption,
        "input_ids": input_ids,
        "text_encoder_outputs": text_encoder_outputs,
    }


def load_feature_sidecars(
    dataset,
    image_info,
    subset,
    *,
    target_stem: str,
    sampler,
    ip_ref_stem,
    ip_ref_subset,
    ip_ref_reldir,
    ip_shuffled_stem,
):
    """Load BYG / inversion / IP / contrastive-negative sidecars."""
    if dataset.byg_text_dir:
        byg_tuple = dataset._try_load_byg_tuple(image_info.absolute_path)
    else:
        byg_tuple = None

    if dataset.inversion_dir:
        inversion_runs = dataset._try_load_inversion_runs(image_info.absolute_path)
    else:
        inversion_runs = None

    if ip_ref_stem is None or ip_ref_stem == target_stem:
        ip_features = dataset._try_load_ip_features(image_info.absolute_path)
    else:
        ip_features = dataset._load_ip_features_for_stem(
            ip_ref_stem, ip_ref_subset, ip_ref_reldir
        )
    if ip_shuffled_stem is not None and ip_shuffled_stem != target_stem:
        ip_features_shuffled = dataset._load_ip_features_for_stem(
            ip_shuffled_stem, subset, sampler.rel_dir(ip_shuffled_stem)
        )
    else:
        ip_features_shuffled = (
            dataset._try_load_ip_features(image_info.absolute_path)
            if ip_shuffled_stem is not None
            else None
        )

    # Soft-tokens contrastive negatives: draw k unrelated stems and load
    # their cached text embeddings. Deterministic per target on the
    # rare chance this dataset is used for validation; random in
    # training. None when no sampler is attached or the target is absent
    # from the index (the adapter then skips the contrastive forward).
    neg_sampler = dataset.contrastive_neg_sampler
    if neg_sampler is not None and neg_sampler.has(target_stem):
        k = dataset.contrastive_neg_k
        mode = dataset.contrastive_neg_mode
        nrng = random.Random(dataset.seed ^ (hash(target_stem) & 0xFFFFFFFF))
        neg_feats: List[torch.Tensor] = []
        neg_jacc: List[float] = []
        for _ in range(k):
            neg_stem, _lvl = neg_sampler.draw(target_stem, mode, nrng)
            if neg_stem == target_stem:
                continue  # no distinct negative reachable
            feat = dataset._load_te_for_stem(
                neg_stem, subset, neg_sampler.rel_dir(neg_stem)
            )
            if feat is not None:
                neg_feats.append(feat)
                neg_jacc.append(
                    neg_sampler.tag_jaccard(target_stem, neg_stem)
                    if mode == "jaccard"
                    else 0.0
                )
        ok = len(neg_feats) == k
        neg_crossattn = torch.stack(neg_feats, dim=0) if ok else None
        neg_jaccard = (
            torch.tensor(neg_jacc, dtype=torch.float32)
            if (ok and mode == "jaccard")
            else None
        )
    else:
        neg_crossattn = None
        neg_jaccard = None

    return {
        "byg_tuple": byg_tuple,
        "inversion_runs": inversion_runs,
        "ip_features": ip_features,
        "ip_features_shuffled": ip_features_shuffled,
        "neg_crossattn": neg_crossattn,
        "neg_jaccard": neg_jaccard,
    }
