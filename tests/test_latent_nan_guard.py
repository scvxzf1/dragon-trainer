"""Cached-latent NaN scrubbing: the guard must cover both latent sources."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.training.batch_step import _prepare_latents


class _Accelerator:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, message: str) -> None:
        self.messages.append(message)


class _Trainer:
    def __init__(self, *, encode_output=None) -> None:
        self._encode_output = encode_output
        self.encode_calls = 0

    def encode_images_to_latents(self, args, vae, images):  # noqa: ARG002
        self.encode_calls += 1
        if self._encode_output is None:
            return torch.zeros(images.shape[0], 4, 2, 2)
        return self._encode_output

    def shift_scale_latents(self, args, latents):  # noqa: ARG002
        return latents


def _nan_batch() -> dict:
    latents = torch.full((2, 4, 2, 2), float("nan"))
    return {"latents": latents}


@pytest.mark.parametrize("vae_batch_size", [None, 1])
def test_cached_nan_latents_are_scrubbed(vae_batch_size):
    """缓存 latent 分支曾经绕过 NaN 处置，坏 sidecar 会直接进 forward。"""
    accelerator = _Accelerator()
    trainer = _Trainer()
    args = SimpleNamespace(vae_batch_size=vae_batch_size)

    latents = _prepare_latents(args, trainer, None, _nan_batch(), accelerator, torch.float32)

    assert torch.isfinite(latents).all()
    assert trainer.encode_calls == 0, "缓存命中时不得回退到 VAE 编码"
    assert accelerator.messages == ["NaN found in cached latents, replacing with zeros"]


def test_encoded_nan_latents_are_still_scrubbed():
    accelerator = _Accelerator()
    trainer = _Trainer(encode_output=torch.full((1, 4, 2, 2), float("nan")))
    args = SimpleNamespace(vae_batch_size=None)
    batch = {"images": torch.zeros(1, 3, 8, 8)}

    latents = _prepare_latents(args, trainer, None, batch, accelerator, torch.float32)

    assert torch.isfinite(latents).all()
    assert trainer.encode_calls == 1
    assert accelerator.messages == ["NaN found in VAE latents, replacing with zeros"]


def test_chunked_encode_scrubs_nan():
    accelerator = _Accelerator()
    trainer = _Trainer(encode_output=torch.full((1, 4, 2, 2), float("nan")))
    args = SimpleNamespace(vae_batch_size=1)
    batch = {"images": torch.zeros(2, 3, 8, 8)}

    latents = _prepare_latents(args, trainer, None, batch, accelerator, torch.float32)

    assert torch.isfinite(latents).all()
    assert trainer.encode_calls == 2
    assert "NaN found in VAE latents, replacing with zeros" in accelerator.messages


def test_finite_latents_pass_through_untouched():
    accelerator = _Accelerator()
    trainer = _Trainer()
    args = SimpleNamespace(vae_batch_size=None)
    cached = torch.arange(32, dtype=torch.float32).reshape(2, 4, 2, 2)

    latents = _prepare_latents(args, trainer, None, {"latents": cached}, accelerator, torch.float32)

    assert torch.equal(latents, cached)
    assert accelerator.messages == []
