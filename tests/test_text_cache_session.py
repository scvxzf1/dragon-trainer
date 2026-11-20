from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors import safe_open

from library.models.krea2_raw.strategy import Krea2TextEncoderOutputsCachingStrategy
from library.strategies.text_cache import text_encoder_cache_session


def test_text_encoder_cache_session_restores_cpu_model() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with text_encoder_cache_session(model, "cpu", dtype=torch.float64) as prepared:
        assert next(prepared.parameters()).dtype is torch.float64
        assert next(prepared.parameters()).device.type == "cpu"

    assert next(model.parameters()).dtype is torch.float32
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())


def test_text_encoder_cache_session_restores_after_encoder_error() -> None:
    model = nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="encode failed"):
        with text_encoder_cache_session(model, "cpu"):
            raise RuntimeError("encode failed")

    assert next(model.parameters()).device.type == "cpu"


@pytest.mark.parametrize("fail_encode", [False, True])
def test_lazy_text_encoder_cache_session_restores_loader_resource(fail_encode) -> None:
    from contextlib import nullcontext

    events = []
    model = nn.Linear(2, 2)

    def load():
        events.append("load")
        return model

    session = text_encoder_cache_session(None, "cpu", dtype=torch.float64, load=load)
    assert events == []
    expectation = pytest.raises(ValueError, match="encode failed") if fail_encode else nullcontext()
    with expectation:
        with session as prepared:
            assert events == ["load"]
            assert next(prepared.parameters()).dtype is torch.float64
            if fail_encode:
                raise ValueError("encode failed")
    assert next(model.parameters()).dtype is torch.float32


def test_text_encoder_cache_session_restores_moss_runtime_anchor(tmp_path) -> None:
    from library.models.moss_image.text_encoder_runtime import load_moss_text_encoder

    encoder_path = tmp_path / "encoder"
    encoder_path.mkdir()
    (encoder_path / "config.json").write_text("{}", encoding="utf-8")
    model = load_moss_text_encoder(
        str(encoder_path), dtype=torch.bfloat16, device="cpu", quantization="bf16"
    )
    original_anchor = model._anchor

    with text_encoder_cache_session(model, "cpu") as prepared:
        assert prepared is model

    assert model._anchor is not original_anchor
    assert model._anchor.device.type == "cpu"
    assert model._anchor.dtype is torch.bfloat16
    assert model._encoder is None


def test_family_text_cache_facade_stamps_and_checks_identity(tmp_path) -> None:
    encoder = tmp_path / "encoder.safetensors"
    encoder.write_bytes(b"encoder-v1")
    cache_path = tmp_path / "sample_krea2_te.safetensors"
    strategy = Krea2TextEncoderOutputsCachingStrategy(
        identity_inputs={"text_encoder": str(encoder)},
        require_cache_identity=True,
    )
    strategy._encode_captions = lambda *args, **kwargs: (
        torch.zeros(1, 2, 12, 2560, dtype=torch.bfloat16),
        torch.ones(1, 2, dtype=torch.bool),
    )
    info = SimpleNamespace(
        caption="test",
        caption_variants=None,
        cache_caption_variants=False,
        caption_dropout_rate=0.0,
        text_encoder_outputs_npz=str(cache_path),
    )

    strategy.cache_batch_outputs(None, [], None, [info])

    with safe_open(str(cache_path), framework="pt") as handle:
        metadata = handle.metadata()
        assert metadata["model_family"] == "krea2_raw"
        assert metadata["cache_schema"] == "krea2_te_v1"
        assert "cache_identity" in metadata
    assert strategy.is_disk_cached_outputs_expected(str(cache_path))

    encoder.write_bytes(b"encoder-v2")
    assert not strategy.is_disk_cached_outputs_expected(str(cache_path))
