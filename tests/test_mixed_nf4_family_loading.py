"""Exercise real small official models through production family loaders on CPU."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file, load_file
from library.quantization import (
    BF16AutocastLinear4bit,
    save_mixed_nf4,
    inspect_mixed_nf4,
)
from library.quantization.training import validate_mixed_nf4_training


@pytest.fixture(params=["krea2_raw", "z_image", "qwen_image_2_1"])
def checkpoint(request, monkeypatch, tmp_path):
    from diffusers import ZImageTransformer2DModel, QwenImage21Transformer2DModel
    from library.models.krea2_raw import weights as krea
    from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT
    from library.models.z_image import weights as z
    from library.models.qwen_image_2_1 import weights as q

    family = request.param
    if family == "krea2_raw":
        config = SingleMMDiTConfig(
            features=64,
            tdim=64,
            txtdim=64,
            heads=4,
            multiplier=2,
            layers=1,
            patch=2,
            channels=16,
            txtlayers=1,
            txtheads=4,
            txtkvheads=4,
        )
        model = SingleStreamDiT(config)

        def loader(path, dtype=torch.bfloat16):
            return krea.load_krea2_dit(path, config=config, dtype=dtype)
    elif family == "z_image":
        config = dict(
            dim=192,
            n_layers=30,
            n_refiner_layers=1,
            n_heads=12,
            n_kv_heads=12,
            cap_feat_dim=2560,
            axes_dims=[4, 6, 6],
            axes_lens=[8, 8, 8],
        )
        model = ZImageTransformer2DModel(**config)
        monkeypatch.setattr(
            ZImageTransformer2DModel, "load_config", lambda *a, **k: config
        )

        def from_single_file(path, **kwargs):
            ordinary = ZImageTransformer2DModel(**config).to(torch.bfloat16)
            state = load_file(path)
            for prefix in (
                "noise_refiner.0",
                "context_refiner.0",
                *(f"layers.{i}" for i in range(30)),
            ):
                q, k, v = state.pop(prefix + ".attention.qkv.weight").chunk(3)
                for name, value in zip(("to_q", "to_k", "to_v"), (q, k, v)):
                    state[prefix + ".attention." + name + ".weight"] = value
            ordinary.load_state_dict(state, strict=True)
            return ordinary

        monkeypatch.setattr(
            ZImageTransformer2DModel, "from_single_file", from_single_file
        )

        def loader(path, dtype=torch.bfloat16):
            return z.load_z_image_transformer(str(path), dtype=dtype)
    else:
        config = dict(
            num_layers=32,
            num_attention_heads=4,
            attention_head_dim=16,
            context_in_dim=64,
            axes_dims_rope=(4, 6, 6),
            mlp_ratio=2,
        )
        model = QwenImage21Transformer2DModel(**config)
        monkeypatch.setattr(
            "diffusers.QwenImage21Transformer2DModel",
            lambda: QwenImage21Transformer2DModel(**config),
        )

        def loader(path, dtype=torch.bfloat16):
            return q.load_qwen_image_2_1_transformer(str(path), dtype=dtype)

    model = model.to(torch.bfloat16)
    state = dict(model.state_dict())
    if family == "z_image":
        for prefix in (
            "noise_refiner.0",
            "context_refiner.0",
            *(f"layers.{i}" for i in range(30)),
        ):
            state[prefix + ".attention.qkv.weight"] = torch.cat(
                [
                    state.pop(prefix + ".attention." + name + ".weight")
                    for name in ("to_q", "to_k", "to_v")
                ]
            )
    if family == "qwen_image_2_1":
        for i in range(32):
            prefix = f"transformer_blocks.{i}.img_mlp."
            state[prefix + "gate_up.weight"] = torch.cat(
                [
                    state.pop(prefix + "gate_layer.weight"),
                    state.pop(prefix + "proj.weight"),
                ]
            )
    source = tmp_path / "source.safetensors"
    save_file(state, str(source))
    ordinary = loader(source)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(ordinary.state_dict()[key], value, rtol=0, atol=0)
    mixed = tmp_path / "mixed.safetensors"
    save_mixed_nf4(ordinary, mixed, family=family, source_sha256="a" * 64)
    return family, loader, ordinary, mixed


def test_family_loader_roundtrip(checkpoint):
    family, loader, ordinary, mixed = checkpoint
    model = loader(mixed)
    manifest = inspect_mixed_nf4(mixed, family=family)
    packed = load_file(str(mixed))
    assert model._mixed_nf4_family == family
    for key, value in ordinary.state_dict().items():
        if key.removesuffix(".weight") not in manifest["modules"]:
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    for path in manifest["modules"]:
        module = model.get_submodule(path)
        assert isinstance(module, BF16AutocastLinear4bit)
        assert torch.equal(module.weight, packed[path + ".weight"])
    assert all(not value.is_meta for value in model.parameters())
    if family == "qwen_image_2_1":
        assert not model.time_text_embed.time_proj.freqs.is_meta
        assert all(not freq.is_meta for freq in model.pos_embed.freqs)
    if family == "z_image":
        assert not model.rope_embedder(torch.zeros(1, 3, dtype=torch.long)).is_meta
    with pytest.raises(ValueError, match="BF16"):
        loader(mixed, dtype=torch.float16)
    other = "z_image" if family != "z_image" else "krea2_raw"
    with pytest.raises(ValueError):
        inspect_mixed_nf4(mixed, family=other)


@pytest.mark.parametrize(
    "change",
    [
        dict(base_compute="nf4"),
        dict(mixed_precision="fp16"),
        dict(adaptive_precision="auto"),
        dict(block_swap_transfer_dtype="fp16"),
        dict(gradient_checkpointing=False),
        dict(selective_checkpoint="every_other"),
        dict(torch_compile=True),
        dict(use_lokr=True),
        dict(network_args=["dora_wd=True"]),
        dict(qwen_fused_projections="mlp"),
        dict(qwen_saved_projection_blocks=2),
        dict(qwen_segmented_training=True),
    ],
)
def test_training_contract(change):
    args = SimpleNamespace(
        gradient_checkpointing=True,
        **{k: v for k, v in change.items() if k != "gradient_checkpointing"},
    )
    if "gradient_checkpointing" in change:
        args.gradient_checkpointing = False
    model = SimpleNamespace(_mixed_nf4_family="qwen_image_2_1")
    with pytest.raises(ValueError):
        validate_mixed_nf4_training(model, args)
    validate_mixed_nf4_training(SimpleNamespace(), args)


def test_valid_training_and_cli_guards(tmp_path):
    validate_mixed_nf4_training(
        SimpleNamespace(_mixed_nf4_family="krea2_raw"),
        SimpleNamespace(gradient_checkpointing=True),
    )
    from scripts.quantization.convert_mixed_nf4 import convert

    with pytest.raises(ValueError, match="single"):
        convert(family="krea2_raw", source=tmp_path, output=tmp_path / "out")
    source = tmp_path / "source.safetensors"
    save_file({"x": torch.ones(64, dtype=torch.float16)}, str(source))
    with pytest.raises(ValueError, match="ordinary BF16"):
        convert(family="krea2_raw", source=source, output=tmp_path / "out")
    output = tmp_path / "exists"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        convert(family="krea2_raw", source=source, output=output)
    assert output.read_bytes() == b"preserve"


def test_cli_conversion_dispatch_and_hash(tmp_path, monkeypatch):
    import hashlib
    from scripts.quantization.convert_mixed_nf4 import convert
    from library.models.krea2_raw import weights
    import library.quantization.mixed_nf4 as core

    source = tmp_path / "source.safetensors"
    save_file({"ordinary": torch.ones(64, dtype=torch.bfloat16)}, str(source))
    model = object()
    calls = []

    def load(path, **kwargs):
        assert path == source and kwargs == {"dtype": torch.bfloat16, "device": "cpu"}
        return model

    def save(value, path, **kwargs):
        assert value is model
        calls.append(kwargs)
        return {"output": str(path)}

    monkeypatch.setattr(weights, "load_krea2_dit", load)
    monkeypatch.setattr(core, "save_mixed_nf4", save)
    result = convert(family="krea2_raw", source=source, output=tmp_path / "out")
    assert result["output"] == str(tmp_path / "out")
    assert calls == [
        {
            "family": "krea2_raw",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]


def test_cli_dispatches_anima_to_canonical_converter(tmp_path, monkeypatch):
    from scripts.quantization import convert_mixed_nf4 as cli
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    save_file({"ordinary": torch.ones(64, dtype=torch.bfloat16)}, str(source))
    calls = []

    def convert_checkpoint(source_path, output_path, *, device, threads=None):
        calls.append((source_path, output_path, device, threads))
        return {"output": str(output_path), "canonical": True}

    monkeypatch.setattr("library.anima.mixed_nf4.convert_checkpoint", convert_checkpoint)
    result = cli.convert(
        family="anima", source=source, output=output, policy="main-mlp", threads=3
    )
    assert result["canonical"] is True
    assert calls == [(source.resolve(), output.resolve(), "cpu", 3)]

    with pytest.raises(ValueError, match="fixed selective"):
        cli.convert(
            family="anima",
            source=source,
            output=tmp_path / "other.safetensors",
            policy="all-linear",
        )


def test_cli_verify_dispatches_anima_without_rewriting(tmp_path, monkeypatch):
    from scripts.quantization import convert_mixed_nf4 as cli
    artifact = tmp_path / "mixed.safetensors"
    artifact.write_bytes(b"artifact")
    calls = []

    def verify_checkpoint(path, source):
        calls.append((path, source))
        return {"output": str(path), "verified": True}

    monkeypatch.setattr("library.anima.mixed_nf4.verify_checkpoint", verify_checkpoint)
    result = cli.verify(family="anima", output=artifact, source=None)
    assert result["verified"] is True
    assert calls == [(artifact, None)]
    assert artifact.read_bytes() == b"artifact"
