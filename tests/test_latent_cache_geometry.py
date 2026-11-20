"""Shared latent-cache geometry contracts for every registered latent space."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from library.models.family_registry import MODEL_FAMILY_REGISTRY, get_model_family_spec
from library.preprocess.latents import cache_latents
from library.strategies.latent_cache import FamilyLatentCacheStrategy


BUCKET_RESO = (512, 256)


def _image_info():
    return SimpleNamespace(
        bucket_reso=BUCKET_RESO,
        latents_npz="unused.npz",
    )


def _strategy(family, encode_by_vae):
    return FamilyLatentCacheStrategy(
        False,
        1,
        False,
        model_family=family,
        encode_by_vae=encode_by_vae,
    )


def _mock_image_loading(monkeypatch):
    monkeypatch.setattr(
        "library.datasets.image_utils.load_images_and_masks_for_caching",
        lambda image_infos, use_alpha_mask, random_crop: (
            torch.zeros(
                len(image_infos),
                3,
                BUCKET_RESO[1],
                BUCKET_RESO[0],
            ),
            [None] * len(image_infos),
            [(BUCKET_RESO[0], BUCKET_RESO[1])] * len(image_infos),
            [(0, 0, 0, 0)] * len(image_infos),
        ),
    )


@pytest.mark.parametrize("family", tuple(MODEL_FAMILY_REGISTRY))
def test_encoded_latent_geometry_matches_registry_stride(family, monkeypatch):
    _mock_image_loading(monkeypatch)
    spec = get_model_family_spec(family).latent_space
    expected = (
        BUCKET_RESO[1] // spec.vae_spatial_compression,
        BUCKET_RESO[0] // spec.vae_spatial_compression,
    )
    latent = torch.zeros(1, spec.latent_channels, 1, *expected)
    strategy = _strategy(family, lambda _images: latent)
    info = _image_info()

    strategy._default_cache_batch_latents(
        strategy._encode_by_vae,
        torch.device("cpu"),
        torch.bfloat16,
        [info],
        False,
        False,
        False,
        multi_resolution=True,
    )

    assert tuple(info.latents.shape) == tuple(latent.shape[1:])
    torch.testing.assert_close(info.latents, latent[0])


@pytest.mark.parametrize("family", tuple(MODEL_FAMILY_REGISTRY))
def test_encoded_latent_geometry_rejects_bad_bucket_scale(family, monkeypatch):
    _mock_image_loading(monkeypatch)
    spec = get_model_family_spec(family).latent_space
    bad_latent = torch.zeros(
        1,
        spec.latent_channels,
        1,
        BUCKET_RESO[1] // spec.vae_spatial_compression + 1,
        BUCKET_RESO[0] // spec.vae_spatial_compression,
    )
    strategy = _strategy(family, lambda _images: bad_latent)

    with pytest.raises(ValueError, match="geometry.*bucket_reso"):
        strategy._default_cache_batch_latents(
            strategy._encode_by_vae,
            torch.device("cpu"),
            torch.bfloat16,
            [_image_info()],
            False,
            False,
            False,
            multi_resolution=True,
        )


def test_encoded_latent_geometry_rejects_batch_mismatch(monkeypatch):
    _mock_image_loading(monkeypatch)
    spec = get_model_family_spec("anima").latent_space
    latent = torch.zeros(
        2,
        spec.latent_channels,
        1,
        BUCKET_RESO[1] // spec.vae_spatial_compression,
        BUCKET_RESO[0] // spec.vae_spatial_compression,
    )
    strategy = _strategy("anima", lambda _images: latent)

    with pytest.raises(ValueError, match="latent batch 2 for 1 image"):
        strategy._default_cache_batch_latents(
            strategy._encode_by_vae,
            torch.device("cpu"),
            torch.bfloat16,
            [_image_info()],
            False,
            False,
            False,
            multi_resolution=True,
        )


def test_flipped_latent_geometry_is_validated(monkeypatch):
    _mock_image_loading(monkeypatch)
    spec = get_model_family_spec("anima").latent_space
    expected = (
        BUCKET_RESO[1] // spec.vae_spatial_compression,
        BUCKET_RESO[0] // spec.vae_spatial_compression,
    )
    outputs = iter(
        [
            torch.zeros(1, spec.latent_channels, 1, *expected),
            torch.zeros(
                1,
                spec.latent_channels,
                1,
                expected[0] + 1,
                expected[1],
            ),
        ]
    )
    strategy = _strategy("anima", lambda _images: next(outputs))

    with pytest.raises(ValueError, match="flipped latent geometry"):
        strategy._default_cache_batch_latents(
            strategy._encode_by_vae,
            torch.device("cpu"),
            torch.bfloat16,
            [_image_info()],
            True,
            False,
            False,
            multi_resolution=True,
        )


def test_invalid_on_disk_geometry_is_marked_for_rebuild(tmp_path):
    cache_path = tmp_path / "bad_0064x0128_anima.npz"
    np.savez(
        cache_path,
        latents_8x16=np.zeros((16, 8, 15), dtype=np.float32),
        original_size_8x16=np.array([128, 64]),
        crop_ltrb_8x16=np.array([0, 0, 0, 0]),
    )
    strategy = FamilyLatentCacheStrategy(
        True,
        1,
        False,
        model_family="anima",
        encode_by_vae=None,
    )

    assert not strategy.is_disk_cached_latents_expected(
        (128, 64), str(cache_path), False, False
    )


def test_valid_legacy_on_disk_geometry_remains_cacheable(tmp_path):
    cache_path = tmp_path / "valid_0064x0128_anima.npz"
    np.savez(
        cache_path,
        latents_8x16=np.zeros((16, 8, 16), dtype=np.float32),
        original_size_8x16=np.array([128, 64]),
        crop_ltrb_8x16=np.array([0, 0, 0, 0]),
    )
    strategy = FamilyLatentCacheStrategy(
        True,
        1,
        False,
        model_family="anima",
        encode_by_vae=None,
    )

    assert strategy.is_disk_cached_latents_expected(
        (128, 64), str(cache_path), False, False
    )


@pytest.mark.parametrize("sample_shape", [(16, 8, 16), (16, 1, 8, 16), (16, 2, 8, 16)])
def test_preprocessing_and_training_disk_read_share_latent_geometry(tmp_path, sample_shape):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (128, 64)).save(image_path)
    spec = get_model_family_spec("anima").latent_space
    vae = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
    strategy = _strategy("anima", None)
    cache_path = strategy.get_latents_npz_path(str(image_path), (128, 64))

    def encode(_vae, images):
        return torch.zeros(images.shape[0], *sample_shape)

    if len(sample_shape) == 4 and sample_shape[1] != 1:
        with pytest.raises(ValueError, match="preprocess latent geometry"):
            cache_latents(tmp_path, vae, batch_size=1, encode_fn=encode, latent_spec=spec)
        assert not list(tmp_path.glob("*.npz"))
        return

    stats = cache_latents(tmp_path, vae, batch_size=1, encode_fn=encode, latent_spec=spec)
    assert stats.written == 1
    latents, *_ = strategy.load_latents_from_disk(cache_path, (128, 64))
    assert latents.shape == sample_shape

    with np.load(cache_path) as stored:
        values = dict(stored)
    values["latents_flipped_8x16"] = np.zeros((16, 8, 15))
    np.savez(cache_path, **values)

    with pytest.raises(ValueError, match="flipped latent geometry"):
        strategy.load_latents_from_disk(cache_path, (128, 64))
