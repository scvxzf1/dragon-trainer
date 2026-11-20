"""Cache VAE latents for a dataset directory.

Orchestration extracted from ``preprocess/cache_latents.py`` (see
``docs/proposal/tooling_architecture.md`` §A). The script keeps only argparse +
VAE load; the walk → group-by-resolution → batched-encode → idempotent-save
loop lives here.

Idempotence note: a single ``{stem}_{WxH}_anima.npz`` can hold *multiple*
resolutions (one ``latents_{H}x{W}`` key each), so the skip is per-resolution
*inside* the encode loop rather than a whole-file existence check.
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from library import cache_identity as _identity
from library.io.cache import resolve_cache_path
from library.io.cache_names import latent_cache_suffix
from library.models.latent_space import LatentSpaceSpec
from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.preprocess._dataset import PreprocessStats, group_by_shape, walk_images
from library.preprocess._progress import ProgressFn
from library.preprocess.adaptive_batch import AutoBatcher, batch_size_arg, iter_batches
from library.strategies.latent_geometry import validate_latent_batch_geometry


def get_latents_npz_path(
    image_path: Path,
    image_size: tuple[int, int],
    cache_dir: Path | None = None,
    image_dir: Path | None = None,
    latent_space_name: str | None = None,
) -> Path:
    """Latent sidecar path: ``{stem}_{WxH}_anima.npz`` by default.

    ``latent_space_name`` selects the space suffix via
    :func:`library.io.cache_names.latent_cache_suffix` (e.g.
    ``"dcgen_f32c32"`` → ``_dcgen_f32c32.npz``). With ``cache_dir`` the cache
    is redirected there (nested under the source subpath when ``image_dir`` is
    given); otherwise it lives next to the image.
    """
    suffix = (
        f"_{image_size[0]:04d}x{image_size[1]:04d}"
        + latent_cache_suffix(latent_space_name)
    )
    if cache_dir is None:
        return image_path.with_name(image_path.stem + suffix)
    return Path(
        resolve_cache_path(
            str(image_path),
            suffix,
            cache_dir=str(cache_dir),
            image_dir=str(image_dir) if image_dir is not None else None,
        )
    )


def _latent_cached(
    npz_path: Path,
    w: int,
    h: int,
    vae_spatial_compression: int = 8,
    *,
    cache_identity: dict | None = None,
    require_cache_identity: bool = False,
) -> bool:
    """True iff ``npz_path`` already holds the ``(w, h)`` latent.

    A single NPZ can carry several resolutions (one ``latents_{H}x{W}`` key
    each), so the check is per-resolution, not whole-file existence. A
    truncated / unreadable NPZ counts as not-cached (it'll be re-encoded).
    ``vae_spatial_compression`` converts pixel ``(w,h)`` to latent ``(H,W)``."""
    if not npz_path.exists():
        return False
    key = f"latents_{h // vae_spatial_compression}x{w // vae_spatial_compression}"
    try:
        with np.load(npz_path) as npz:
            keys = set(npz.files)
            if cache_identity is not None:
                recorded = _identity.read_npz_stamp(npz)
        if cache_identity is not None:
            # A sidecar encoded by a different VAE must be rebuilt even when the
            # requested resolution key is present.
            if _identity.evaluate(
                recorded, cache_identity, require_identity=require_cache_identity
            ) == "stale":
                return False
            if _identity.compare(recorded, cache_identity) == "absent":
                _identity.warn_legacy("VAE latent", npz_path)
        return key in keys
    except Exception:
        return False


def count_pending_latents(
    data_dir: Path,
    *,
    cache_dir: Path | None = None,
    recursive: bool = False,
    path_pattern: str | None = None,
    overwrite: bool = False,
    latent_space_name: str | None = None,
    vae_spatial_compression: int = 8,
    cache_identity: dict | None = None,
    require_cache_identity: bool = False,
) -> tuple[int, int]:
    """Return ``(pending, total)`` latent caches **without loading the VAE**.

    ``pending`` is the number of images whose ``(W, H)`` latent isn't already
    on disk; ``total`` is every enumerated image. Mirrors the per-resolution
    skip in :func:`cache_latents`, so the entry point can skip the (slow) VAE
    load entirely when ``pending == 0``. Reads only NPZ headers (no decode).
    With ``overwrite`` every enumerated image counts as pending."""
    image_files = walk_images(data_dir, recursive=recursive, pattern=path_pattern)
    if overwrite:
        return len(image_files), len(image_files)
    pending = 0
    for (w, h), paths in group_by_shape(image_files).items():
        for p in paths:
            npz_path = get_latents_npz_path(
                p,
                (w, h),
                cache_dir=cache_dir,
                image_dir=data_dir,
                latent_space_name=latent_space_name,
            )
            if not _latent_cached(
                npz_path,
                w,
                h,
                vae_spatial_compression=vae_spatial_compression,
                cache_identity=cache_identity,
                require_cache_identity=require_cache_identity,
            ):
                pending += 1
    return pending, len(image_files)


def _decode_batch(
    batch_paths: list[Path],
    w: int,
    h: int,
    cache_dir: Path | None,
    data_dir: Path,
    overwrite: bool = False,
    latent_space_name: str | None = None,
    vae_spatial_compression: int = 8,
    cache_identity: dict | None = None,
    require_cache_identity: bool = False,
) -> tuple[
    list[Path],
    list[tuple[Path, str]],
    list[tuple[Path, tuple[int, int]]],
    "torch.Tensor | None",
]:
    """CPU stage: per-resolution skip-probe + decode + transform a batch.

    Returns ``(skipped, failed, kept, img_batch)`` — already-cached paths,
    ``(path, reason)`` for images that wouldn't decode (truncated / corrupt),
    the ``(path, (w, h))`` survivors, and their stacked CPU tensor (``None`` if
    nothing survived). A bad file is isolated to its own entry rather than
    raising, so one corrupt staged PNG can't abort the whole run. Runs on a
    worker thread (pure PIL/numpy/torch-CPU), overlapping the previous GPU encode."""
    skipped: list[Path] = []
    failed: list[tuple[Path, str]] = []
    kept: list[tuple[Path, tuple[int, int]]] = []
    tensors: list[torch.Tensor] = []
    for p in batch_paths:
        npz_path = get_latents_npz_path(
            p,
            (w, h),
            cache_dir=cache_dir,
            image_dir=data_dir,
            latent_space_name=latent_space_name,
        )
        if not overwrite and _latent_cached(
            npz_path,
            w,
            h,
            vae_spatial_compression=vae_spatial_compression,
            cache_identity=cache_identity,
            require_cache_identity=require_cache_identity,
        ):
            skipped.append(p)
            continue
        try:
            img_np = np.array(Image.open(p).convert("RGB"))
        except Exception as e:
            failed.append((p, f"{type(e).__name__}: {e}"))
            continue
        tensors.append(IMAGE_TRANSFORMS(img_np))
        kept.append((p, (w, h)))
    img_batch = torch.stack(tensors, dim=0) if tensors else None
    return skipped, failed, kept, img_batch


def _save_batch(
    items: list[tuple[Path, np.ndarray, tuple[int, int]]],
    *,
    cache_identity: dict | None = None,
) -> None:
    """IO stage: write each ``(npz_path, latent, size)``. Preserves any
    other-resolution keys already in the file (read-modify-write). Each npz_path
    is written exactly once per ``cache_latents`` call, so threaded saves of a
    batch don't race. Runs on a worker thread, overlapping the next GPU encode."""
    for npz_path, lat_np, size in items:
        key_reso_suffix = f"_{lat_np.shape[-2]}x{lat_np.shape[-1]}"
        kwargs: dict = {}
        if npz_path.exists():
            with np.load(npz_path) as npz:
                previous = _identity.read_npz_stamp(npz)
                stale = (
                    cache_identity is not None
                    and _identity.compare(previous, cache_identity) == "mismatch"
                )
                if stale:
                    # Other resolutions in this file came from the previous VAE
                    # and are equally stale, so they are dropped, not carried.
                    print(
                        f"  VAE identity changed for {npz_path.name}: dropping "
                        "latents encoded by the previous VAE."
                    )
                else:
                    for key in npz.files:
                        kwargs[key] = npz[key]
        if cache_identity is not None:
            kwargs[_identity.NPZ_KEY] = _identity.encode_npz(cache_identity)
        kwargs[f"latents{key_reso_suffix}"] = lat_np
        kwargs[f"original_size{key_reso_suffix}"] = np.array(list(size))
        kwargs[f"crop_ltrb{key_reso_suffix}"] = np.array([0, 0, size[0], size[1]])
        np.savez(npz_path, **kwargs)


def _encode_latent_batch(vae, images, encode_fn):
    images = images.to(device=vae.device, dtype=vae.dtype)
    with torch.no_grad():
        if encode_fn is not None:
            return encode_fn(vae, images).cpu()
        return vae.encode_pixels_to_latents(images).cpu()


def cache_latents(
    data_dir: Path,
    vae,
    *,
    cache_dir: Path | None = None,
    recursive: bool = False,
    path_pattern: str | None = None,
    batch_size: int | str = "auto",
    progress: ProgressFn | None = None,
    io_workers: int | None = None,
    overwrite: bool = False,
    latent_space_name: str | None = None,
    vae_spatial_compression: int = 8,
    encode_fn=None,
    cache_identity: dict | None = None,
    require_cache_identity: bool = False,
    latent_spec: LatentSpaceSpec | None = None,
) -> PreprocessStats:
    """Encode every image under ``data_dir`` through ``vae`` → latent NPZs.

    ``vae`` is supplied loaded + on-device (``device``/``dtype`` are read off
    it). Returns counts; pass ``progress`` for a per-image bar. With
    ``overwrite`` the per-resolution skip is bypassed and every (W,H) latent is
    re-encoded (the matching ``latents_{H}x{W}`` key is replaced in place,
    other-resolution keys preserved).

    The VAE forward stays serial on the calling thread (single GPU stream); the
    per-batch disk decode + image transform and the npz read-modify-write are
    CPU/IO, so they're farmed to thread pools that overlap the GPU — the GPU no
    longer idles between batches. ``io_workers`` sizes those pools (default
    ``min(8, cpu_count)``). Output is byte-identical to the serial path."""
    batch_size = batch_size_arg(batch_size)
    image_files = walk_images(data_dir, recursive=recursive, pattern=path_pattern)
    reso_groups = group_by_shape(image_files)
    stats = PreprocessStats(seen=len(image_files))

    if progress is not None:
        progress(0, total=len(image_files))

    tuners = {
        (w, h): AutoBatcher(vae.device, label=f"vae/{w}x{h}",
                           cleanup=getattr(vae, "clear_cache", None))
        for w, h in reso_groups
    } if batch_size == "auto" else {}

    def batches():
        for (w, h), paths in reso_groups.items():
            tuner = tuners.get((w, h))
            for group in iter_batches(paths, lambda: tuner.batch_size if tuner else batch_size):
                yield w, h, group

    workers = io_workers or min(8, (os.cpu_count() or 4))
    depth = 1 if tuners else max(2, workers // 2)
    max_saves = max(2, workers)  # in-flight npz writes before backpressure
    it = iter(batches())
    decode_q: deque = deque()
    save_q: deque = deque()
    failed_all: list[tuple[Path, str]] = []

    def _submit_decode(decode_ex) -> bool:
        b = next(it, None)
        if b is None:
            return False
        w, h, bp = b
        decode_q.append(
            decode_ex.submit(
                _decode_batch,
                bp,
                w,
                h,
                cache_dir,
                data_dir,
                overwrite,
                latent_space_name,
                vae_spatial_compression,
                cache_identity,
                require_cache_identity,
            )
        )
        return True

    with (
        ThreadPoolExecutor(max_workers=workers) as decode_ex,
        ThreadPoolExecutor(max_workers=workers) as save_ex,
    ):
        for _ in range(depth):  # prime the decode window
            if not _submit_decode(decode_ex):
                break
        while decode_q:
            skipped, failed, kept, img_batch = decode_q.popleft().result()
            _submit_decode(decode_ex)  # keep a decode in flight

            for p in skipped:
                stats.skipped += 1
                if progress is not None:
                    progress(1, detail=f"skip {p.name}")
            for p, reason in failed:  # isolate corrupt files; don't abort the run
                failed_all.append((p, reason))
                if progress is not None:
                    progress(1, detail=f"FAILED {p.name}")
            if img_batch is None:
                continue

            tuner = tuners.get(kept[0][1])
            if tuner is not None:
                latents = tuner.encode_all(
                    img_batch, lambda images: _encode_latent_batch(vae, images, encode_fn)
                )
            else:
                latents = _encode_latent_batch(vae, img_batch, encode_fn)

            if latent_spec is not None:
                validate_latent_batch_geometry(
                    latents,
                    [size for _, size in kept],
                    latent_spec,
                    label=f"{latent_spec.name} preprocess latent",
                )
            items: list[tuple[Path, np.ndarray, tuple[int, int]]] = []
            for i, (p, size) in enumerate(kept):
                npz_path = get_latents_npz_path(
                    p,
                    size,
                    cache_dir=cache_dir,
                    image_dir=data_dir,
                    latent_space_name=latent_space_name,
                )
                items.append((npz_path, latents[i].float().numpy(), size))
                stats.written += 1
                if progress is not None:
                    progress(1, detail=f"{p.name} → {size[0]}x{size[1]}")
            save_q.append(
                save_ex.submit(_save_batch, items, cache_identity=cache_identity)
            )

            while len(save_q) >= max_saves:  # backpressure on the save side
                save_q.popleft().result()
        for f in save_q:  # surface any write error + drain remaining writes
            f.result()

    if failed_all:
        print(
            f"\n⚠ {len(failed_all)} image(s) could not be decoded and were skipped "
            f"(no latent cached — re-stage these, e.g. delete + re-run mangafy):"
        )
        for p, reason in failed_all:
            print(f"  {p}  ({reason})")

    return stats
