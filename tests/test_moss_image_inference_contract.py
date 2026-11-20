"""Standalone MOSS-Image text-encoder placement and quantization contracts."""

from types import SimpleNamespace

import pytest
import torch

from library.inference.family_runner import FamilyInferenceRunner
from library.models.moss_image import inference_runner as moss_runner
from library.models.moss_image.inference_runner import MossImageInferenceRunner
from library.models.moss_image.text_encoder_runtime import MossTextEncoderRuntime


def _moss_args(**overrides):
    values = {
        "sampler": "euler",
        "image_size": [64, 64],
        "lora_weight": None,
        "moss_te_quantization": "nf4",
        "device": torch.device("cuda:0"),
        "text_encoder_cpu": False,
        "runtime_dtype": "bf16",
        "text_encoder_dtype": "bf16",
        "text_encoder": "text-encoder",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_moss_nf4_rejects_cpu_text_stage_before_loading():
    with pytest.raises(SystemExit, match="--moss_te_quantization bf16"):
        MossImageInferenceRunner.validate(_moss_args(device=torch.device("cpu")))

    with pytest.raises(SystemExit, match="--moss_te_quantization bf16"):
        MossImageInferenceRunner.validate(
            _moss_args(text_encoder_cpu=True)
        )


def test_moss_bf16_allows_explicit_cpu_text_stage():
    args = _moss_args(
        device=torch.device("cpu"),
        text_encoder_cpu=True,
        moss_te_quantization="bf16",
    )

    MossImageInferenceRunner.validate(args)


def test_moss_rejects_non_bf16_text_compute_before_loading():
    with pytest.raises(SystemExit, match="--text_encoder_dtype bf16"):
        MossImageInferenceRunner.validate(
            _moss_args(runtime_dtype="fp16", text_encoder_dtype="same")
        )


@pytest.mark.parametrize(
    "quantization,text_encoder_cpu,expected_device",
    [
        ("nf4", False, torch.device("cuda:1")),
        ("bf16", True, torch.device("cpu")),
    ],
)
def test_moss_loader_receives_resolved_text_stage_device(
    monkeypatch,
    quantization,
    text_encoder_cpu,
    expected_device,
):
    seen = {}

    def fake_loader(path, *, dtype, device, quantization):
        seen.update(
            path=path,
            dtype=dtype,
            device=device,
            quantization=quantization,
        )
        return object()

    monkeypatch.setattr(moss_runner, "load_moss_text_encoder", fake_loader)
    args = _moss_args(
        device=torch.device("cuda:1"),
        moss_te_quantization=quantization,
        text_encoder_cpu=text_encoder_cpu,
    )
    text_encoder_device = MossImageInferenceRunner._resolve_text_encoder_device(
        args,
        torch.device("cuda:1"),
    )

    MossImageInferenceRunner._load_text_encoder(args, text_encoder_device)

    assert seen == {
        "path": "text-encoder",
        "dtype": torch.bfloat16,
        "device": expected_device,
        "quantization": quantization,
    }


def test_inference_cli_exposes_moss_text_quantization():
    import inference

    base = ["--text_encoder", "te", "--save_path", "out", "--prompt", "p"]
    assert inference.parse_args(base).moss_te_quantization == "nf4"
    assert (
        inference.parse_args(
            [*base, "--moss_te_quantization", "bf16"]
        ).moss_te_quantization
        == "bf16"
    )


@pytest.mark.parametrize("fail_operation", [False, True])
def test_moss_text_stage_releases_encoder_on_success_and_failure(
    tmp_path,
    fail_operation,
):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    text_encoder = MossTextEncoderRuntime(
        str(tmp_path),
        dtype=torch.bfloat16,
        device="cpu",
        quantization="bf16",
    )
    text_encoder._encoder = object()
    runtime = FamilyInferenceRunner(
        "moss_image",
        torch.device("cpu"),
        cleanup=lambda _device: None,
    )

    def operation(encoder):
        assert encoder is text_encoder
        if fail_operation:
            raise RuntimeError("encoding failed")
        return "encoded"

    if fail_operation:
        with pytest.raises(RuntimeError, match="encoding failed"):
            runtime.run_text(lambda: text_encoder, operation)
    else:
        assert runtime.run_text(lambda: text_encoder, operation) == "encoded"

    assert text_encoder._encoder is None
