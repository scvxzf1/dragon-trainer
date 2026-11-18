from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from safetensors.torch import save_file

from library.datasets.dataset_getitem_sample import (
    _as_float32_tensor,
    _maybe_downsample_mask_to_latent,
)
from library.datasets.dataset_getitem_stack import none_or_stack_elements
from library.datasets import qwen_image_edit as qwen_edit
from library.models.krea2_raw.family import prepare_img_tokens
from library.models.qwen_image_2_1 import strategy as qwen_strategy
from library.training.losses import apply_inverted_masked_loss
from library.training.mask_loss import reduce_masked_loss


def test_as_float32_tensor_is_zero_copy_on_contiguous_float32_numpy() -> None:
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    tensor = _as_float32_tensor(arr)
    assert tensor.dtype == torch.float32
    assert tensor.data_ptr() == arr.ctypes.data

    non_contig = arr[:, ::-1, :]
    copied = _as_float32_tensor(non_contig)
    assert copied.is_contiguous()
    torch.testing.assert_close(copied, torch.tensor(non_contig.copy()))


def test_worker_pre_downsampled_mask_matches_loss_side_interpolation() -> None:
    torch.manual_seed(0)
    pixel_mask = torch.rand(1024, 1024, dtype=torch.float32)
    latent = torch.randn(16, 128, 128, dtype=torch.float32)

    pre = _maybe_downsample_mask_to_latent(pixel_mask, latent, image=None)
    assert pre is not None
    assert pre.shape == (128, 128)
    assert pre.numel() * 64 == pixel_mask.numel()

    loss_4d = torch.rand(1, 16, 128, 128, dtype=torch.float32)
    for norm in ("none", "foreground_mean"):
        r_pixel = reduce_masked_loss(
            loss_4d, {"alpha_masks": pixel_mask.unsqueeze(0)}, normalize=norm
        )
        r_pre = reduce_masked_loss(
            loss_4d, {"alpha_masks": pre.unsqueeze(0)}, normalize=norm
        )
        torch.testing.assert_close(r_pre, r_pixel, rtol=1e-6, atol=1e-6)

    inv_pixel = apply_inverted_masked_loss(
        loss_4d, {"alpha_masks": pixel_mask.unsqueeze(0)}
    )
    inv_pre = apply_inverted_masked_loss(loss_4d, {"alpha_masks": pre.unsqueeze(0)})
    torch.testing.assert_close(inv_pre, inv_pixel, rtol=1e-6, atol=1e-6)


def test_none_or_stack_elements_single_alloc_padding_1d_and_2d() -> None:
    rows = [
        [torch.tensor([1, 2]), torch.ones(2, 4)],
        [torch.tensor([3, 4, 5, 6]), torch.ones(5, 4) * 2.0],
    ]
    stacked = none_or_stack_elements(rows, lambda x: x)
    assert stacked is not None
    assert stacked[0].shape == (2, 4)
    assert stacked[0][0].tolist() == [1, 2, 0, 0]
    assert stacked[0][1].tolist() == [3, 4, 5, 6]
    assert stacked[1].shape == (2, 5, 4)
    assert torch.all(stacked[1][0, :2] == 1.0)
    assert torch.all(stacked[1][0, 2:] == 0.0)


def test_qwen_edit_fingerprint_and_reference_loader_memoize_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref_path = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), color=(12, 34, 56)).save(ref_path)

    raw_bytes = ref_path.read_bytes()
    ref_hash = hashlib.sha256()
    ref_hash.update(raw_bytes)
    ref_hash.update(b"\0edit prompt\0")
    ref_hash.update(f"64x64:{qwen_edit.REFERENCE_RESIZE_VERSION}".encode())
    expected_fp = ref_hash.hexdigest()

    qwen_edit._REFERENCE_HASH_STATE_CACHE.clear()
    fp1 = qwen_edit.edit_condition_fingerprint(str(ref_path), "edit prompt", (64, 64))
    fp2 = qwen_edit.edit_condition_fingerprint(str(ref_path), "edit prompt", (64, 64))
    assert fp1 == expected_fp == fp2

    # Mutating the file invalidates the stat-keyed cache entry immediately.
    Image.new("RGB", (128, 64), color=(99, 11, 22)).save(ref_path)
    fp_changed = qwen_edit.edit_condition_fingerprint(
        str(ref_path), "edit prompt", (64, 64)
    )
    assert fp_changed != fp1

    Image.new("RGB", (64, 64), color=(12, 34, 56)).save(ref_path)
    cache_strategy = qwen_strategy.QwenImage21LatentCache(
        cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False
    )
    latent_cache_path = tmp_path / "ref_latent.safetensors"
    save_file(
        {"latent": torch.ones(64, 4, 4, dtype=torch.bfloat16)},
        str(latent_cache_path),
        metadata={
            "model_family": "qwen_image_2_1",
            "edit_cache_schema": cache_strategy.EDIT_LATENT_CACHE_SCHEMA,
        },
    )
    info = SimpleNamespace(
        reference_image_path=str(ref_path),
        absolute_path=str(tmp_path / "target.png"),
        caption="edit prompt",
        bucket_reso=(64, 64),
        edit_reference_latent_path=str(latent_cache_path),
    )
    subset = SimpleNamespace(cache_dir=str(tmp_path), image_dir=str(tmp_path))

    open_calls = {"count": 0}
    orig_open = Image.open

    def _counting_open(*args, **kwargs):
        open_calls["count"] += 1
        return orig_open(*args, **kwargs)

    monkeypatch.setattr(qwen_strategy.Image, "open", _counting_open)
    for _ in range(8):
        lat = cache_strategy.load_edit_reference_latent(info, subset)
        assert lat.shape == (64, 4, 4)
    assert open_calls["count"] == 1


def test_krea2_prepare_img_tokens_reuses_static_pos_buffer() -> None:
    img1 = torch.randn(2, 16, 8, 8)
    img2 = torch.randn(2, 16, 8, 8)
    mask1 = torch.ones(2, 6, dtype=torch.bool)
    mask2 = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)

    _, pos1, out_mask1 = prepare_img_tokens(img1, 6, 2, mask1)
    _, pos2, out_mask2 = prepare_img_tokens(img2, 6, 2, mask2)
    assert pos1.data_ptr() == pos2.data_ptr()
    assert out_mask1.shape == out_mask2.shape == (2, 22)
    assert not torch.equal(out_mask1, out_mask2)
