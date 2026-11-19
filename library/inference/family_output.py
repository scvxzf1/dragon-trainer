"""Shared image persistence for standalone family runners."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def save_pil_image(args, image, *, seed: int, png_info_factory, prefix: str) -> None:
    os.makedirs(args.save_path, exist_ok=True)
    stem = f"{prefix}_{seed}"
    path = Path(args.save_path) / f"{stem}.png"
    image.save(path, pnginfo=png_info_factory(args, seed, stem))


def save_with_shared_qwen_vae(
    args,
    latent: torch.Tensor,
    device: torch.device,
    *,
    family: str,
) -> None:
    """Decode/save an Anima-layout latent through the shared Qwen VAE."""
    from library.inference.family_runner import FamilyInferenceRunner
    from library.inference.output import save_output
    from library.inference.precision import resolve_runtime_dtype
    from library.models import qwen_vae as qwen_image_autoencoder_kl
    from library.runtime.device import clean_memory_on_device

    def load_vae():
        return qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=args.vae_chunk_size,
            disable_cache=args.vae_disable_cache,
        )

    def decode_and_save(vae):
        vae.to(dtype=resolve_runtime_dtype(args))
        vae.eval()
        return save_output(args, vae, latent, device)

    runtime = FamilyInferenceRunner(
        family,
        device,
        cleanup=clean_memory_on_device,
    )
    runtime.run_stage("vae", load_vae, decode_and_save)
