"""Image sizes read from latent cache file names, including a separate cache_dir.

Regression for t13 F-T13-4: the reader only globbed ``subset.image_dir``, so a
dataset using the documented ``cache_dir`` layout (sidecars mirrored out of the
image tree) matched 0/N and every size was read by opening the image again — which
also widened the mismatch between cache keys and buckets.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from library.anima.strategy import AnimaLatentsCachingStrategy
from library.datasets import dreambooth


def _strategy():
    """The real Anima strategy: real ``cache_suffix`` and real name parsing."""
    return AnimaLatentsCachingStrategy(False, 1, True)


def _images(image_dir, stems, subdir=""):
    target = image_dir / subdir if subdir else image_dir
    target.mkdir(parents=True, exist_ok=True)
    return [str(target / f"{stem}.png") for stem in stems]


def _touch_images(paths):
    for path in paths:
        with open(path, "wb") as handle:
            handle.write(b"not-really-a-png")


def _sidecars(directory, stems, resolution="1024x1024", subdir=""):
    target = directory / subdir if subdir else directory
    target.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        with open(target / f"{stem}_{resolution}_anima.npz", "wb") as handle:
            handle.write(b"x")


def _subset(image_dir, cache_dir=None, recursive=False):
    return SimpleNamespace(
        image_dir=str(image_dir),
        cache_dir=None if cache_dir is None else str(cache_dir),
        recursive=recursive,
    )


def test_sizes_come_from_a_separate_cache_dir(tmp_path, caplog):
    """Images in image_dir, sidecars in cache_dir: every image must be a hit."""
    caplog.set_level(logging.INFO)
    image_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    stems = [f"img{index:02d}" for index in range(5)]
    img_paths = _images(image_dir, stems)
    _touch_images(img_paths)
    _sidecars(cache_dir, stems)

    sizes = dreambooth.set_image_sizes_from_cache(
        _subset(image_dir, cache_dir), img_paths, recursive=False, strategy=_strategy()
    )

    assert sizes == [(1024, 1024)] * len(stems)
    assert sum(1 for size in sizes if size is not None) == len(stems)
    assert any("from cache_dir=" in record.getMessage() for record in caplog.records)
    # Nothing was moved into image_dir to make this work.
    assert not list(image_dir.glob("*.npz"))
    assert len(list(cache_dir.glob("*.npz"))) == len(stems)


def test_mirrored_cache_dir_subdirectories_are_matched_by_stem(tmp_path):
    """A recursive source tree mirrored into cache_dir matches per subdirectory."""
    image_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    first = _images(image_dir, ["same", "other"], subdir="charA")
    second = _images(image_dir, ["same", "other"], subdir="charB")
    _touch_images(first + second)
    _sidecars(cache_dir, ["same", "other"], resolution="768x1024", subdir="charA")
    _sidecars(cache_dir, ["same", "other"], resolution="1024x768", subdir="charB")

    sizes = dreambooth.set_image_sizes_from_cache(
        _subset(image_dir, cache_dir, recursive=True),
        first + second,
        recursive=True,
        strategy=_strategy(),
    )

    # Same stems in two subdirectories must not be confused with each other.
    assert sizes == [(768, 1024), (768, 1024), (1024, 768), (1024, 768)]


def test_sidecars_next_to_the_images_are_unchanged(tmp_path):
    """The historical layout keeps working exactly as before."""
    image_dir = tmp_path / "images"
    stems = [f"img{index:02d}" for index in range(3)]
    img_paths = _images(image_dir, stems)
    _touch_images(img_paths)
    _sidecars(image_dir, stems, resolution="512x768")

    sizes = dreambooth.set_image_sizes_from_cache(
        _subset(image_dir), img_paths, recursive=False, strategy=_strategy()
    )

    assert sizes == [(512, 768)] * len(stems)


def test_no_cache_anywhere_leaves_sizes_for_the_slow_path(tmp_path, caplog):
    """Unknown sizes stay ``None`` and the log says they will be opened."""
    caplog.set_level(logging.INFO)
    image_dir = tmp_path / "images"
    img_paths = _images(image_dir, ["a", "b"])
    _touch_images(img_paths)

    sizes = dreambooth.set_image_sizes_from_cache(
        _subset(image_dir), img_paths, recursive=False, strategy=_strategy()
    )

    assert sizes == [None, None]
    messages = [record.getMessage() for record in caplog.records]
    assert any("set image size from cache files: 0/2" in message for message in messages)
    assert any("read by opening the image" in message for message in messages)


def test_a_missing_cache_dir_is_named_not_fatal(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    image_dir = tmp_path / "images"
    img_paths = _images(image_dir, ["a"])
    _touch_images(img_paths)

    sizes = dreambooth.set_image_sizes_from_cache(
        _subset(image_dir, tmp_path / "does-not-exist"),
        img_paths,
        recursive=False,
        strategy=_strategy(),
    )

    assert sizes == [None]
    assert any(
        "cache_dir" in record.getMessage() and "does not exist" in record.getMessage()
        for record in caplog.records
    )


def test_ambiguous_duplicate_stems_are_not_guessed(tmp_path, caplog):
    """Two candidate sidecars for one flat image: refuse instead of guessing."""
    caplog.set_level(logging.WARNING)
    image_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    img_paths = _images(image_dir, ["only"])
    _touch_images(img_paths)
    _sidecars(cache_dir, ["only"], resolution="1024x1024", subdir="one")
    _sidecars(cache_dir, ["only"], resolution="512x512", subdir="two")

    sizes = dreambooth.set_image_sizes_from_cache(
        _subset(image_dir, cache_dir), img_paths, recursive=False, strategy=_strategy()
    )

    assert sizes == [None]
    assert any("ambiguous latent cache" in record.getMessage() for record in caplog.records)


def test_a_flat_cache_dir_is_accepted_for_recursive_images(tmp_path):
    """Callers that wrote the cache without the source root still match."""
    image_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    img_paths = _images(image_dir, ["nested"], subdir="deep")
    _touch_images(img_paths)
    _sidecars(cache_dir, ["nested"], resolution="832x1216")

    sizes = dreambooth.set_image_sizes_from_cache(
        _subset(image_dir, cache_dir, recursive=True),
        img_paths,
        recursive=True,
        strategy=_strategy(),
    )

    assert sizes == [(832, 1216)]
