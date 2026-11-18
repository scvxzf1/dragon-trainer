#!/usr/bin/env python3
"""Cache VAE latents for all images in a dataset directory.

Encodes images through the Qwen Image VAE and saves latent caches (.npz)
alongside the images (or under ``--cache_dir``).  Skips already-cached
entries (idempotent).

The walk → group-by-resolution → encode → save loop lives in
``library/preprocess/latents.py``; this file is argparse + VAE load + reporting.
"""

import argparse
from pathlib import Path

import torch


from library import cache_identity as _identity
from library.preprocess import cache_latents, tqdm_progress
from library.runtime.cli import add_io_args
from library.runtime.device import str_to_dtype
from library.models.family_registry import get_family_plugin, get_model_family_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_args(
        parser,
        cache_noun="latent caches",
        include_batch_size=True,
        batch_size_default="auto",
        allow_auto_batch_size=True,
    )
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE weights")
    parser.add_argument(
        "--model_family",
        choices=["anima", "krea2_raw", "z_image", "qwen_image_2_1", "moss_image"],
        default="anima",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="VAE spatial chunk size (default: 64)",
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        default=True,
        help="Disable VAE internal cache (default: True)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default=None,
        help="VAE compute dtype (default: float32 for MOSS, bfloat16 otherwise).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode and replace existing latent sidecars.",
    )
    parser.add_argument(
        "--require_cache_identity",
        action="store_true",
        help=(
            "Treat sidecars without a cache-identity stamp (pre-identity "
            "caches) as stale and rebuild them too. By default unstamped "
            "sidecars are reused with a warning, and only sidecars stamped "
            "with a different VAE are rebuilt."
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = str_to_dtype(args.dtype or ("float32" if args.model_family == "moss_image" else "bfloat16"))

    print(f"Loading VAE from {args.vae} ...")
    plugin = get_family_plugin(args.model_family)
    vae, encode_fn = plugin.load_preprocess_vae(args, dtype)
    latent_spec = get_model_family_spec(args.model_family).latent_space
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()

    stats = cache_latents(
        data_dir,
        vae,
        cache_identity=_identity.latent_cache_identity(
            family=args.model_family, vae=args.vae
        ),
        require_cache_identity=bool(args.require_cache_identity),
        cache_dir=cache_dir,
        recursive=args.recursive,
        path_pattern=args.path_pattern,
        batch_size=args.batch_size,
        progress=tqdm_progress("Caching latents"),
        overwrite=bool(args.overwrite),
        latent_space_name=latent_spec.name if latent_spec.name != "anima" else None,
        vae_spatial_compression=latent_spec.vae_spatial_compression,
        encode_fn=encode_fn,
    )
    print(
        f"\nLatent caching complete: {stats.written} cached, "
        f"{stats.skipped} skipped (already existed)"
    )

    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
