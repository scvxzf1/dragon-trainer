from __future__ import annotations

import argparse

import pytest
import torch
from PIL import Image

from library.config.loader import (
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
)
from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS, BucketManager
from library.datasets.dataset_getitem_sample import _latent_target_size
from library.datasets.qwen_image_geometry import (
    align_qwen_bucket_manager,
    align_qwen_resolution,
)
from library.preprocess.images import resize_to_buckets
from web.services.config.cache_audit import _bucket_reso


QWEN = "qwen_image_2_1"


@pytest.mark.parametrize(
    ("family", "shape", "expected"),
    [
        ("anima", (16, 32, 48), (384, 256)),
        ("qwen_image_2_1", (1, 64, 32, 48), (768, 512)),
        ("moss_image", (128, 1, 32, 48), (768, 512)),
    ],
)
def test_cached_latent_target_size_uses_family_compression_and_trailing_axes(
    family, shape, expected
):
    assert _latent_target_size(torch.zeros(shape), family) == expected


def _training_dataset(image_dir, settings, family=QWEN):
    config = {
        "general": {"caption_extension": ".txt"},
        "datasets": [{
            "batch_size": 1,
            "subsets": [{
                "image_dir": str(image_dir),
                "custom_attributes": {"preprocess": settings},
            }],
        }],
    }
    args = argparse.Namespace(
        model_family=family,
        train_batch_size=None,
        debug_dataset=False,
        max_token_length=None,
        prior_loss_weight=1.0,
    )
    blueprint = BlueprintGenerator(ConfigSanitizer(support_dropout=True)).generate(
        config, args
    )
    group, _ = generate_dataset_group_by_blueprint(
        blueprint.dataset_group, constant_token_buckets=True
    )
    dataset = group.datasets[0]
    assert dataset.subsets[0].model_family == family
    return dataset


def test_qwen_alignment_preserves_anima_canonical_table():
    canonical = list(CONSTANT_TOKEN_BUCKETS)
    assert len(canonical) == 24
    assert (896, 1200) in canonical
    assert align_qwen_resolution((896, 1200)) == (896, 1184)
    manager = BucketManager(
        max_reso=(1024, 1024), min_size=256, max_size=2048, reso_steps=64
    )
    manager.make_buckets(constant_token_buckets=True)
    assert manager.predefined_resos == canonical
    align_qwen_bucket_manager(manager)
    assert (896, 1184) in manager.predefined_resos
    assert all(w % 32 == h % 32 == 0 for w, h in manager.predefined_resos)
    assert list(CONSTANT_TOKEN_BUCKETS) == canonical
    anima = BucketManager(
        max_reso=(1024, 1024), min_size=256, max_size=2048, reso_steps=64
    )
    anima.make_buckets(constant_token_buckets=True)
    assert anima.predefined_resos == canonical
    assert anima.select_bucket(896, 1200)[0] == (896, 1200)


@pytest.mark.parametrize("size", [(31, 32), (32, 31), (0, 1200)])
def test_qwen_alignment_rejects_dimensions_smaller_than_patch(size):
    with pytest.raises(ValueError, match="at least 32"):
        align_qwen_resolution(size)


@pytest.mark.parametrize(
    "source_size, overrides, expected",
    [
        ((896, 1200), {}, (896, 1184)),
        ((896, 1200), {"bucket_no_upscale": True}, (896, 1184)),
        ((624, 900), {"bucket_no_upscale": True}, (608, 896)),
        ((1024, 1400), {"bucket_no_upscale": True}, (1024, 1376)),
        ((896, 1200), {"enable_bucket": False, "resolution": 1008}, (992, 992)),
    ],
)
def test_qwen_training_resize_and_audit_share_pixel_geometry(tmp_path, source_size, overrides, expected):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "portrait.png"
    Image.new("RGB", source_size, color=(20, 40, 60)).save(image)
    image.with_suffix(".txt").write_text("portrait", encoding="utf-8")
    settings = {
        "resolution": 1024,
        "min_bucket_reso": 256,
        "max_bucket_reso": 2048,
        "bucket_reso_steps": 64,
        "enable_bucket": True,
        "bucket_no_upscale": False,
        **overrides,
    }
    dataset = _training_dataset(source, settings)
    assert next(iter(dataset.image_data.values())).bucket_reso == expected
    assert _bucket_reso(image, {"settings": settings}, QWEN) == expected

    destination = tmp_path / "resized"
    stats, counts = resize_to_buckets(
        source, destination, **settings, model_family=QWEN,
        workers=1, min_pixels=0, verbose=False,
    )
    assert stats.written == 1
    assert counts == {expected: 1}
    output = destination / image.name
    with Image.open(output) as resized:
        assert resized.size == expected
        assert all(dimension % 32 == 0 for dimension in resized.size)
    assert output.with_suffix(".txt").read_text(encoding="utf-8") == "portrait"
    resized_dataset = _training_dataset(destination, settings)
    assert next(iter(resized_dataset.image_data.values())).bucket_reso == expected
    assert _bucket_reso(output, {"settings": settings}, QWEN) == expected


def test_anima_training_preserves_unaligned_canonical_bucket(tmp_path):
    image = tmp_path / "portrait.png"
    Image.new("RGB", (896, 1200)).save(image)
    settings = {"resolution": 1024, "enable_bucket": True}
    dataset = _training_dataset(tmp_path, settings, family="anima")
    assert next(iter(dataset.image_data.values())).bucket_reso == (896, 1200)
    assert _bucket_reso(image, {"settings": settings}) == (896, 1200)
