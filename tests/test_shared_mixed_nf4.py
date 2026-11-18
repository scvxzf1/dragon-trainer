"""Real CPU NF4 fixtures for canonical multi-family checkpoints."""

import json
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from library.quantization import (
    MANIFEST_KEY,
    BF16AutocastLinear4bit,
    inspect_mixed_nf4,
    load_mixed_nf4_into,
    save_mixed_nf4,
    save_mixed_nf4_state_dict,
)
from library.quantization.policy import FAMILIES
from library.quantization.policy import policy_paths


def tiny(family, device="cpu", count=2):
    model = torch.nn.Module()
    root, mlp, names = FAMILIES[family]
    blocks = torch.nn.ModuleList()
    for _ in range(count):
        block = torch.nn.Module()
        ff = torch.nn.Module()
        for name in names:
            setattr(
                ff,
                name,
                torch.nn.Linear(64, 64, bias=True, device=device, dtype=torch.bfloat16),
            )
        setattr(block, mlp, ff)
        block.attn = torch.nn.Linear(64, 64, device=device, dtype=torch.bfloat16)
        block.norm = torch.nn.LayerNorm(64, device=device, dtype=torch.bfloat16)
        blocks.append(block)
    setattr(model, root, blocks)
    model.register_buffer("scale", torch.ones((), device=device, dtype=torch.float32))
    return model


def mutate(path, fn):
    state = load_file(str(path))
    with safe_open(str(path), framework="pt") as f:
        manifest = json.loads(f.metadata()[MANIFEST_KEY])
    fn(state, manifest)
    save_file(state, str(path), metadata={MANIFEST_KEY: json.dumps(manifest)})


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_roundtrip(family, tmp_path):
    torch.manual_seed(31)
    source = tiny(family)
    before = {k: t.clone() for k, t in source.state_dict().items()}
    path = tmp_path / "mixed.safetensors"
    result = save_mixed_nf4(source, path, family=family, source_sha256="a" * 64)
    expected_quantized = 4 if family == "anima" else 6
    assert result["quantized_weights"] == expected_quantized
    manifest = inspect_mixed_nf4(path, family=family)
    model = load_mixed_nf4_into(
        tiny(family, "meta"), path, family=family, dtype=torch.bfloat16, device="cpu"
    )
    assert model._mixed_nf4_family == family
    saved = load_file(str(path))
    for key, value in source.state_dict().items():
        assert torch.equal(value, before[key])
        if key.removesuffix(".weight") not in manifest["modules"]:
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    for p in manifest["modules"]:
        linear = model.get_submodule(p)
        assert isinstance(linear, BF16AutocastLinear4bit)
        assert torch.equal(linear.weight, saved[p + ".weight"])
        for k, value in linear.weight.quant_state.as_dict(packed=True).items():
            assert torch.equal(value, saved[p + ".weight." + k])
        x = torch.randn(2, 64, dtype=torch.bfloat16, requires_grad=True)
        y = linear(x)
        y.float().square().sum().backward()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    second = load_mixed_nf4_into(
        tiny(family, "meta"), path, family=family, dtype=torch.bfloat16, device="cpu"
    )
    x = torch.randn(2, 64, dtype=torch.bfloat16, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    p = manifest["modules"][0]
    y, yy = model.get_submodule(p)(x), second.get_submodule(p)(xx)
    torch.testing.assert_close(y, yy, rtol=0, atol=0)
    y.float().sum().backward()
    yy.float().sum().backward()
    torch.testing.assert_close(x.grad, xx.grad, rtol=0, atol=0)


@pytest.mark.parametrize("family", FAMILIES)
def test_prefixed_state_dict_export_roundtrip(family, tmp_path):
    source = tiny(family)
    path = tmp_path / "prefixed.safetensors"
    prefix = "net."
    root = FAMILIES[family][0]
    modules = policy_paths(family, len(getattr(source, root)))
    save_mixed_nf4_state_dict(
        {prefix + key: value for key, value in source.state_dict().items()},
        path,
        family=family,
        source_sha256="d" * 64,
        modules=modules,
        prefix=prefix,
    )
    manifest = inspect_mixed_nf4(path, family=family)
    assert manifest["prefix"] == prefix
    restored = load_mixed_nf4_into(
        tiny(family, "meta"), path, family=family, dtype=torch.bfloat16, device="cpu"
    )
    for key, value in source.state_dict().items():
        if key.removesuffix(".weight") not in modules:
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "change",
    ["family", "policy", "module", "shape", "missing", "extra", "fp16", "qstate"],
)
def test_corruption_rejected(change, tmp_path):
    family = "krea2_raw"
    path = tmp_path / "mixed.safetensors"
    save_mixed_nf4(tiny(family), path, family=family, source_sha256="b" * 64)

    def edit(state, m):
        if change == "family":
            m["family"] = "z_image"
        if change == "policy":
            m["policy"] = "all-linears"
        if change == "module":
            m["modules"].pop()
        if change == "shape":
            m["source_tensors"]["blocks.0.attn.weight"]["shape"] = [64, 128]
        if change == "missing":
            del state["blocks.0.attn.weight"]
        if change == "extra":
            state["surprise"] = torch.ones(1)
        if change == "fp16":
            m["compute_dtype"] = "float16"
        if change == "qstate":
            state["blocks.0.mlp.gate.weight.quant_map"].zero_()

    mutate(path, edit)
    with pytest.raises(ValueError):
        load_mixed_nf4_into(
            tiny(family, "meta"),
            path,
            family=family,
            dtype=torch.bfloat16,
            device="cpu",
        )


def test_source_and_destination_guards(tmp_path):
    path = tmp_path / "exists.safetensors"
    path.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        save_mixed_nf4(None, path, family="krea2_raw", source_sha256="a" * 64)
    assert path.read_bytes() == b"preserve"
    model = tiny("krea2_raw")
    for invalid in ["bad", "A" * 64]:
        with pytest.raises(ValueError):
            save_mixed_nf4(
                model, tmp_path / "bad", family="krea2_raw", source_sha256=invalid
            )
    model.blocks[0].mlp.gate.forward = lambda x: x
    with pytest.raises(ValueError):
        save_mixed_nf4(
            model, tmp_path / "bad", family="krea2_raw", source_sha256="a" * 64
        )
    assert inspect_mixed_nf4(tmp_path) is None
    assert inspect_mixed_nf4(tmp_path / "absent") is None
    with pytest.raises(ValueError):
        load_mixed_nf4_into(
            None, "absent", family="krea2_raw", dtype=torch.float16, device="cpu"
        )


def test_unmarked_packed_and_existing_krea_format(tmp_path):
    path = tmp_path / "packed.safetensors"
    state = {"x.weight.absmax": torch.ones(1)}
    save_file(state, str(path))
    with pytest.raises(ValueError):
        inspect_mixed_nf4(path)
    save_file(state, str(path), metadata={"krea2_nf4_format": "krea2_raw_nf4"})
    assert inspect_mixed_nf4(path) is None


@pytest.mark.parametrize(
    "mutation",
    ["extra_declared", "f32_parameter", "other_family", "model_shape", "packed_target"],
)
def test_model_contract_is_exact(mutation, tmp_path):
    family = "krea2_raw"
    path = tmp_path / "model.safetensors"
    save_mixed_nf4(tiny(family), path, family=family, source_sha256="c" * 64)
    model = tiny(family, "meta")
    if mutation == "extra_declared":

        def edit(state, m):
            state["extra"] = torch.ones(1, dtype=torch.bfloat16)
            m["source_tensors"]["extra"] = {"shape": [1], "dtype": "bfloat16"}

        mutate(path, edit)
    elif mutation == "f32_parameter":

        def edit(state, m):
            key = "blocks.0.attn.weight"
            state[key] = state[key].float()
            m["source_tensors"][key]["dtype"] = "float32"

        mutate(path, edit)
    elif mutation == "other_family":
        family = "z_image"
    elif mutation == "model_shape":
        model.blocks[0].attn = torch.nn.Linear(
            64, 128, device="meta", dtype=torch.bfloat16
        )
    elif mutation == "packed_target":
        model.blocks[0].mlp.gate = BF16AutocastLinear4bit(64, 64, device="meta")
    with pytest.raises(ValueError):
        load_mixed_nf4_into(
            model, path, family=family, dtype=torch.bfloat16, device="cpu"
        )


@pytest.mark.parametrize(
    "mutation", ["empty", "partial", "unaligned", "packed", "fp16"]
)
def test_source_policy_guards(mutation, tmp_path):
    model = tiny("krea2_raw", count=0 if mutation == "empty" else 2)
    if mutation == "partial":
        del model.blocks[1].mlp.down
    if mutation == "unaligned":
        model.blocks[0].mlp.gate = torch.nn.Linear(65, 64, dtype=torch.bfloat16)
    if mutation == "packed":
        model.blocks[0].attn = BF16AutocastLinear4bit(64, 64)
    if mutation == "fp16":
        model = model.half()
    with pytest.raises(ValueError):
        save_mixed_nf4(
            model, tmp_path / "rejected", family="krea2_raw", source_sha256="a" * 64
        )
    assert not (tmp_path / "rejected").exists()
