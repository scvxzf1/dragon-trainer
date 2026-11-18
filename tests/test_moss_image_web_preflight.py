from __future__ import annotations

import json
from pathlib import Path

from tests.web_config_test_support import (
    _patch_config_service_paths,
    _write_selected_checkpoint_preflight_config,
)
from web.services import config_service


def _preflight() -> dict:
    return config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/moss.toml",
    )


def _write_component(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({}), encoding="utf-8")
    (path / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")


def _write_moss_config(configs: Path, *, dit: str, qwen3: str, vae: str) -> None:
    (configs / "imported" / "moss.toml").write_text(
        "\n".join(
            [
                'model_family = "moss_image"',
                'source_image_dir = "image_dataset/selected"',
                f'pretrained_model_name_or_path = "{dit}"',
                f'qwen3 = "{qwen3}"',
                f'vae = "{vae}"',
                'moss_te_quantization = "bf16"',
                'output_name = "moss-test"',
                'network_module = "networks.lora_anima"',
                'network_train_unet_only = true',
                'gradient_checkpointing = true',
                'use_vae_cache = true',
                'use_text_cache = true',
                'mixed_precision = "bf16"',
                'base_compute = "bf16"',
                'attn_mode = "torch"',
                'selective_checkpoint = "off"',
                'timestep_sampling = "uniform"',
                'weighting_scheme = "none"',
                'loss_type = "l2"',
                'sample_sampler = "euler"',
                'caption_dropout_rate = 0.0',
                'max_train_steps = 1',
            ]
        ),
        encoding="utf-8",
    )


def test_moss_preflight_accepts_direct_components_and_adjacent_tokenizer(
    tmp_path: Path, monkeypatch
) -> None:
    configs, _ = _write_minimal_moss_fixture(tmp_path, monkeypatch)
    model_root = tmp_path / "models" / "moss"
    _write_component(model_root / "transformer")
    _write_component(model_root / "text_encoder")
    _write_component(model_root / "vae")
    _write_tokenizer(model_root / "tokenizer")
    _write_moss_config(
        configs,
        dit="models/moss/transformer",
        qwen3="models/moss/text_encoder",
        vae="models/moss/vae",
    )

    result = _preflight()

    checks = {item["key"]: item for item in result["checks"]}
    assert checks["pretrained_model_name_or_path"]["level"] == "ok"
    assert checks["qwen3"]["level"] == "ok"
    assert checks["vae"]["level"] == "ok"
    assert checks["moss_image_tokenizer"]["level"] == "ok"
    assert result["ok"] is True


def test_moss_preflight_accepts_model_root_for_all_components(
    tmp_path: Path, monkeypatch
) -> None:
    configs, _ = _write_minimal_moss_fixture(tmp_path, monkeypatch)
    model_root = tmp_path / "models" / "moss"
    for component in ("transformer", "text_encoder", "vae"):
        _write_component(model_root / component)
    _write_tokenizer(model_root / "tokenizer")
    _write_moss_config(
        configs,
        dit="models/moss",
        qwen3="models/moss",
        vae="models/moss",
    )

    result = _preflight()

    checks = {item["key"]: item for item in result["checks"]}
    assert all(checks[key]["level"] == "ok" for key in (
        "pretrained_model_name_or_path", "qwen3", "vae", "moss_image_tokenizer"
    ))
    assert result["ok"] is True


def test_moss_preflight_rejects_missing_tokenizer_and_single_file(
    tmp_path: Path, monkeypatch
) -> None:
    configs, _ = _write_minimal_moss_fixture(tmp_path, monkeypatch)
    model_root = tmp_path / "models" / "moss"
    _write_component(model_root / "transformer")
    _write_component(model_root / "text_encoder")
    _write_component(model_root / "vae")
    _write_moss_config(
        configs,
        dit="models/moss/transformer",
        qwen3="models/moss/text_encoder",
        vae="models/moss/vae",
    )

    result = _preflight()

    checks = {item["key"]: item for item in result["checks"]}
    assert result["ok"] is False
    assert checks["moss_image_tokenizer"]["level"] == "error"

    _write_moss_config(
        configs,
        dit="models/moss/model.safetensors",
        qwen3="models/moss/text_encoder",
        vae="models/moss/vae",
    )
    (model_root / "model.safetensors").write_bytes(b"not-a-component")
    result = _preflight()
    assert result["ok"] is False
    dit_check = next(
        item for item in result["errors"] if item["key"] == "pretrained_model_name_or_path"
    )
    assert "Diffusers 目录" in dit_check["message"]


def _write_tokenizer(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer_config.json", "tokenizer.json"):
        (path / name).write_text("{}", encoding="utf-8")


def _write_minimal_moss_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    _write_selected_checkpoint_preflight_config(tmp_path, monkeypatch, [])
    configs = tmp_path / "configs"
    return configs, tmp_path / "configs" / "datasets" / "lora.toml"
