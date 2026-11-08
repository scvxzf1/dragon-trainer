from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from bitsandbytes.nn import Linear4bit
from library.quantization.nf4 import BF16AutocastLinear4bit
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from library.models.krea2_raw import weights
from library.models.krea2_raw.quantize import (
    NF4_FORMAT_NAME,
    NF4_FORMAT_VERSION,
    NF4_MODEL_STATE_PREFIX,
    inspect_nf4_checkpoint,
    load_nf4_dit_into,
    quantize_dit_to_nf4,
    save_nf4_dit,
    upgrade_nf4_checkpoint_to_self_contained,
)


class _TinyModel(torch.nn.Module):
    def __init__(self, _config=None) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(8, 4, bias=True)
        self.scale = torch.nn.Parameter(torch.ones(4))
        self.register_buffer("offset", torch.arange(4, dtype=torch.float32))

    def forward(self, x):
        return self.fc(x) * self.scale + self.offset.to(x.dtype)


@pytest.fixture(scope="module")
def nf4_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("krea2_nf4_v2")
    base_path = root / "base.safetensors"
    v2_path = root / "v2.safetensors"
    legacy_path = root / "legacy.safetensors"

    torch.manual_seed(42)
    model = _TinyModel().to(torch.bfloat16)
    base_state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(base_state, str(base_path))
    quantize_dit_to_nf4(model, torch.device("cpu"))

    inputs = torch.randn(2, 8, dtype=torch.bfloat16)
    reference = model(inputs).detach().clone()
    reference_code = model.fc.weight.data.detach().cpu().clone()
    save_nf4_dit(model, str(v2_path))

    v2_state = load_file(str(v2_path), device="cpu")
    legacy_state = {
        key: value
        for key, value in v2_state.items()
        if not key.startswith(NF4_MODEL_STATE_PREFIX)
    }
    save_file(legacy_state, str(legacy_path))
    return {
        "root": root,
        "base": base_path,
        "v2": v2_path,
        "legacy": legacy_path,
        "inputs": inputs,
        "reference": reference,
        "reference_code": reference_code,
    }


def _load_base_model(path: Path) -> _TinyModel:
    model = _TinyModel().to(torch.bfloat16)
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True, assign=True)
    return model


def test_inspect_distinguishes_base_legacy_and_v2(nf4_files) -> None:
    base = inspect_nf4_checkpoint(nf4_files["base"])
    legacy = inspect_nf4_checkpoint(nf4_files["legacy"])
    v2 = inspect_nf4_checkpoint(nf4_files["v2"])

    assert not base.is_nf4
    assert legacy.is_nf4 and legacy.version == 1 and not legacy.self_contained
    assert v2.is_nf4 and v2.version == NF4_FORMAT_VERSION
    assert v2.self_contained
    assert v2.metadata["krea2_nf4_format"] == NF4_FORMAT_NAME
    assert v2.metadata["model_tensor_count"] == "2"


def test_self_contained_v2_round_trip_is_exact(nf4_files, monkeypatch) -> None:
    construction_devices = []

    def record_linear4bit_device(*args, **kwargs):
        construction_devices.append(kwargs.get("device"))
        return BF16AutocastLinear4bit(*args, **kwargs)

    monkeypatch.setattr(
        "library.models.krea2_raw.quantize.BF16AutocastLinear4bit",
        record_linear4bit_device,
    )
    loaded = _TinyModel().to(torch.bfloat16)
    count = load_nf4_dit_into(
        loaded,
        str(nf4_files["v2"]),
        torch.device("cpu"),
        require_self_contained=True,
    )

    assert count == 1
    assert construction_devices == ["meta"]
    assert isinstance(loaded.fc, Linear4bit)
    assert loaded.fc.weight.bnb_quantized
    assert not any(parameter.is_meta for parameter in loaded.parameters())
    assert not any(buffer.is_meta for buffer in loaded.buffers())
    assert torch.equal(loaded.fc.weight.data.cpu(), nf4_files["reference_code"])
    assert torch.equal(loaded.scale.cpu(), torch.ones(4, dtype=torch.bfloat16))
    assert torch.equal(loaded.offset.cpu(), torch.arange(4, dtype=torch.bfloat16))
    assert torch.equal(loaded(nf4_files["inputs"]), nf4_files["reference"])


def test_legacy_overlay_remains_supported(nf4_files) -> None:
    loaded = _load_base_model(nf4_files["base"])
    count = load_nf4_dit_into(
        loaded,
        str(nf4_files["legacy"]),
        torch.device("cpu"),
    )

    assert count == 1
    assert torch.equal(loaded(nf4_files["inputs"]), nf4_files["reference"])
    with pytest.raises(ValueError, match="v1"):
        load_nf4_dit_into(
            _TinyModel(),
            str(nf4_files["legacy"]),
            torch.device("cpu"),
            require_self_contained=True,
        )


def test_upgrade_legacy_checkpoint_without_requantizing(nf4_files) -> None:
    output = nf4_files["root"] / "upgraded.safetensors"
    result = upgrade_nf4_checkpoint_to_self_contained(
        nf4_files["base"],
        nf4_files["legacy"],
        output,
    )

    assert result["format_version"] == NF4_FORMAT_VERSION
    assert result["model_tensor_count"] == 2
    loaded = _TinyModel().to(torch.bfloat16)
    load_nf4_dit_into(
        loaded,
        str(output),
        torch.device("cpu"),
        require_self_contained=True,
    )
    assert torch.equal(loaded.fc.weight.data.cpu(), nf4_files["reference_code"])
    assert torch.equal(loaded(nf4_files["inputs"]), nf4_files["reference"])


def test_v2_rejects_missing_model_state(nf4_files) -> None:
    broken = nf4_files["root"] / "broken.safetensors"
    state = load_file(str(nf4_files["v2"]), device="cpu")
    state.pop(f"{NF4_MODEL_STATE_PREFIX}scale")
    with safe_open(str(nf4_files["v2"]), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    save_file(state, str(broken), metadata=metadata)

    with pytest.raises(ValueError, match="model state|\u6a21型状态"):
        load_nf4_dit_into(
            _TinyModel(),
            str(broken),
            torch.device("cpu"),
            require_self_contained=True,
        )


def test_unknown_nf4_version_is_rejected(nf4_files) -> None:
    unknown = nf4_files["root"] / "unknown.safetensors"
    state = load_file(str(nf4_files["v2"]), device="cpu")
    with safe_open(str(nf4_files["v2"]), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    metadata["krea2_nf4_version"] = "999"
    save_file(state, str(unknown), metadata=metadata)

    with pytest.raises(ValueError, match="999"):
        inspect_nf4_checkpoint(unknown)


def test_load_krea2_dit_accepts_v2_as_base_path(monkeypatch, nf4_files) -> None:
    monkeypatch.setattr(weights, "SingleStreamDiT", _TinyModel)
    monkeypatch.setattr(weights, "init_empty_weights", nullcontext)

    loaded = weights.load_krea2_dit(
        nf4_files["v2"],
        device="cpu",
        dtype=torch.bfloat16,
        config=object(),
        eval=False,
        nf4=True,
    )

    assert isinstance(loaded.fc, Linear4bit)
    assert torch.equal(loaded.fc.weight.data.cpu(), nf4_files["reference_code"])
    assert torch.equal(loaded(nf4_files["inputs"]), nf4_files["reference"])


def test_load_krea2_dit_rejects_non_bf16_compute_dtype(monkeypatch, nf4_files) -> None:
    """NF4 compute_dtype 契约锁定在 bf16（见 bench/krea2_nf4_fp16_grad 的溯源）。"""
    monkeypatch.setattr(weights, "SingleStreamDiT", _TinyModel)
    monkeypatch.setattr(weights, "init_empty_weights", nullcontext)

    for dtype in (torch.float16, torch.float32):
        with pytest.raises(ValueError, match="NF4 量化要求 dtype=bfloat16"):
            weights.load_krea2_dit(
                nf4_files["v2"],
                device="cpu",
                dtype=dtype,
                config=object(),
                eval=False,
                nf4=True,
            )


def test_krea2_nf4_compute_dtype_is_bf16() -> None:
    """生产量化入口的 compute_dtype 不得被改成 fp16（fp16 会破坏反向传播）。"""
    from library.models.krea2_raw.quantize import COMPUTE_DTYPE

    assert COMPUTE_DTYPE == torch.bfloat16
