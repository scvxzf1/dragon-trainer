"""Pairing and deterministic reference preprocessing for Qwen Image 2.1 Edit."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from library.datasets.image_utils import glob_images


REFERENCE_RESIZE_VERSION = "area-32-bicubic-v1"
EDIT_TEXT_CACHE_SUFFIX = "_qwen_image_2_1_edit_te.safetensors"


@dataclass(frozen=True)
class EditPairReport:
    pairs: Mapping[str, str]
    missing_references: tuple[str, ...]
    duplicate_references: tuple[tuple[str, tuple[str, ...]], ...]
    unused_references: tuple[str, ...]


def _pair_key(path: str, root: str) -> tuple[str, str]:
    rel_path = os.path.relpath(path, root)
    rel_dir, filename = os.path.split(rel_path)
    stem = os.path.splitext(filename)[0]
    return rel_dir.replace(os.sep, "/").casefold(), stem.casefold()


def inspect_edit_pairs(
    target_paths: Sequence[str],
    *,
    target_dir: str,
    reference_dir: str,
    recursive: bool = False,
) -> EditPairReport:
    """Match targets to one reference by relative directory and case-folded stem."""
    targets = list(target_paths)
    refs = glob_images(reference_dir, recursive=recursive)
    ref_index: dict[tuple[str, str], list[str]] = {}
    target_index: dict[tuple[str, str], list[str]] = {}
    for path in refs:
        ref_index.setdefault(_pair_key(path, reference_dir), []).append(path)
    for path in targets:
        target_index.setdefault(_pair_key(path, target_dir), []).append(path)

    duplicate_references = tuple(
        ("/".join(key), tuple(paths))
        for key, paths in sorted(ref_index.items())
        if len(paths) > 1
    )
    duplicate_targets = tuple(
        ("/".join(key), tuple(paths))
        for key, paths in sorted(target_index.items())
        if len(paths) > 1
    )
    pairs: dict[str, str] = {}
    missing: list[str] = []
    used: set[str] = set()
    for target in targets:
        matches = ref_index.get(_pair_key(target, target_dir), [])
        if len(matches) == 1:
            pairs[target] = matches[0]
            used.add(matches[0])
        elif not matches:
            missing.append(target)

    duplicates = tuple(sorted((*duplicate_references, *duplicate_targets)))
    return EditPairReport(
        pairs=pairs,
        missing_references=tuple(sorted(missing)),
        duplicate_references=duplicates,
        unused_references=tuple(sorted(path for path in refs if path not in used)),
    )


def require_complete_edit_pairs(report: EditPairReport, *, limit: int = 8) -> None:
    issues: list[str] = []
    if report.missing_references:
        examples = ", ".join(report.missing_references[:limit])
        issues.append(
            f"{len(report.missing_references)} target image(s) have no paired reference: {examples}"
        )
    if report.duplicate_references:
        examples = "; ".join(
            f"{key}: {', '.join(paths)}"
            for key, paths in report.duplicate_references[:limit]
        )
        issues.append(
            f"{len(report.duplicate_references)} ambiguous pair key(s): {examples}"
        )
    if issues:
        raise ValueError("Invalid Qwen Image 2.1 Edit dataset pairing. " + " | ".join(issues))


def _snap_32(value: float) -> int:
    return max(32, int(round(value / 32.0)) * 32)


def reference_size_for_bucket(
    image_size: tuple[int, int], target_bucket: tuple[int, int]
) -> tuple[int, int]:
    """Keep reference aspect ratio while matching target pixel area on a 32px grid."""
    width, height = image_size
    target_width, target_height = target_bucket
    if min(width, height, target_width, target_height) <= 0:
        raise ValueError("Qwen edit image and bucket dimensions must be positive")
    ratio = width / height
    area = target_width * target_height
    return _snap_32((area * ratio) ** 0.5), _snap_32((area / ratio) ** 0.5)


def prepare_reference_image(
    path: str, target_bucket: tuple[int, int]
) -> tuple[Image.Image, np.ndarray]:
    """Create one shared RGB resize for the VLM and VAE conditioning paths."""
    with Image.open(path) as source:
        source = source.convert("RGB")
        resized_size = reference_size_for_bucket(source.size, target_bucket)
        image = source.resize(resized_size, Image.Resampling.BICUBIC)
    return image, np.asarray(image, dtype=np.uint8)


_REFERENCE_HASH_STATE_CACHE: dict[tuple[str, int, int], "hashlib._Hash"] = {}
_REFERENCE_HASH_CACHE_MAX = 4096


def _reference_file_hash_prefix(reference_path: str):
    """Return a clonable SHA-256 state after ``reference_path`` bytes, memoized by ``(abspath, mtime_ns, size)``."""
    abspath = os.path.abspath(reference_path)
    st = os.stat(abspath)
    cache_key = (abspath, int(st.st_mtime_ns), int(st.st_size))
    cached = _REFERENCE_HASH_STATE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()
    file_hash = hashlib.sha256()
    with open(abspath, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            file_hash.update(chunk)
    if len(_REFERENCE_HASH_STATE_CACHE) >= _REFERENCE_HASH_CACHE_MAX:
        _REFERENCE_HASH_STATE_CACHE.pop(next(iter(_REFERENCE_HASH_STATE_CACHE)))
    _REFERENCE_HASH_STATE_CACHE[cache_key] = file_hash
    return file_hash.copy()


def edit_condition_fingerprint(
    reference_path: str,
    instruction: str,
    target_bucket: tuple[int, int],
) -> str:
    digest = _reference_file_hash_prefix(reference_path)
    digest.update(b"\0")
    digest.update(instruction.encode("utf-8"))
    digest.update(b"\0")
    digest.update(f"{target_bucket[0]}x{target_bucket[1]}:{REFERENCE_RESIZE_VERSION}".encode())
    return digest.hexdigest()


def edit_cache_suffix(
    *, task: str, fingerprint: str, target_bucket: tuple[int, int]
) -> str:
    width, height = target_bucket
    return f"_qwen_image_2_1_edit_{task}_{width}x{height}_{fingerprint[:16]}.safetensors"
