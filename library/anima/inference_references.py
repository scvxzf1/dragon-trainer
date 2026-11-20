"""Shared reference-image preparation used by Anima inference stages."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def easycontrol_reference(args) -> str | None:
    weight = getattr(args, "easycontrol_weight", None)
    image = getattr(args, "easycontrol_image", None)
    if weight is None and image is None:
        return None
    if weight is None or image is None:
        raise ValueError(
            "--easycontrol_weight and --easycontrol_image must be passed together "
            f"(got easycontrol_weight={weight!r}, easycontrol_image={image!r})"
        )
    return image


def select_easycontrol_size(args, image_path: str) -> None:
    if not getattr(args, "easycontrol_image_match_size", False):
        return

    from PIL import Image

    from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS

    with Image.open(image_path) as image:
        image_width, image_height = image.size
    aspect = image_width / image_height
    width, height = min(
        CONSTANT_TOKEN_BUCKETS,
        key=lambda size: abs((size[0] / size[1]) - aspect),
    )
    args.image_size = [height, width]
    logger.info(
        "EasyControl: image_size auto-picked from ref (aspect w/h=%.3f) -> %s (HxW)",
        aspect,
        tuple(args.image_size),
    )


def load_easycontrol_vae(args, device, dtype):
    from library.models import qwen_vae as qwen_image_autoencoder_kl

    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=getattr(args, "vae_chunk_size", None),
        disable_cache=getattr(args, "vae_disable_cache", False),
    )
    vae.to(dtype=dtype)
    vae.eval()
    return vae.to(device)


def encode_easycontrol_reference(args, image_path, vae, device, dtype):
    from PIL import Image
    from torchvision import transforms

    height, width = args.image_size
    image = Image.open(image_path).convert("RGB").resize(
        (width, height), Image.Resampling.LANCZOS
    )
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    pixels = transform(image).unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        latent = vae.encode_pixels_to_latents(pixels)
    if latent.ndim == 5 and latent.shape[2] == 1:
        return latent.squeeze(2)
    if latent.ndim != 4:
        raise ValueError(
            f"Anima EasyControl VAE must return [B,C,H,W] or [B,C,1,H,W], "
            f"got {tuple(latent.shape)}"
        )
    return latent
