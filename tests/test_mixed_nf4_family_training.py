"""CUDA acceptance on real tiny family models, real LoRA and real offloading."""
import pytest
import torch

from library.quantization import BF16AutocastLinear4bit, load_mixed_nf4_into, save_mixed_nf4
from networks.lora_modules.lora import LoRAModule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA NF4 training acceptance")


def _construct(family):
    if family == "krea2_raw":
        from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT
        return SingleStreamDiT(SingleMMDiTConfig(features=64, tdim=64, txtdim=64,
            heads=4, multiplier=2, layers=4, patch=2, channels=16,
            txtlayers=1, txtheads=4, txtkvheads=4))
    if family == "z_image":
        from diffusers import ZImageTransformer2DModel
        return ZImageTransformer2DModel(dim=192, n_layers=4, n_refiner_layers=1,
            n_heads=12, n_kv_heads=12, cap_feat_dim=32,
            axes_dims=[4,6,6], axes_lens=[64,64,64])
    from diffusers import QwenImage21Transformer2DModel
    return QwenImage21Transformer2DModel(num_layers=4, num_attention_heads=1,
        attention_head_dim=128, context_in_dim=4096, mlp_ratio=1)


def _load(family, path):
    with torch.device("meta"):
        model = _construct(family).bfloat16()
    load_mixed_nf4_into(model, path, family=family, dtype=torch.bfloat16, device="cpu")
    if family == "qwen_image_2_1":
        from library.models.qwen_image_2_1.weights import _materialize_qwen_derived_tensors
        _materialize_qwen_derived_tensors(model)
    return model.requires_grad_(False)


def _adapters(model, family):
    root, projection = {"krea2_raw": ("blocks", "attn.wq"),
        "z_image": ("layers", "attention.to_q"),
        "qwen_image_2_1": ("transformer_blocks", "attn.to_q")}[family]
    result = torch.nn.ModuleList()
    for index, block in enumerate(getattr(model, root)):
        adapter = LoRAModule(f"lora_{index}", block.get_submodule(projection), lora_dim=4, alpha=4)
        adapter.apply_to()
        # Nonzero up ensures both LoRA matrices receive gradients on the first step.
        with torch.no_grad():
            adapter.lora_up.weight.normal_(0, .01)
        result.append(adapter)
    return result.to(device="cuda", dtype=torch.float32)


def _inputs(family):
    if family == "krea2_raw":
        return torch.randn(1, 4, 64, device="cuda", dtype=torch.bfloat16), torch.randn(1,3,1,64, device="cuda", dtype=torch.bfloat16)
    if family == "z_image":
        return torch.randn(16,1,4,4, device="cuda", dtype=torch.bfloat16), torch.randn(3,32, device="cuda", dtype=torch.bfloat16)
    return torch.randn(1,64,2,2, device="cuda", dtype=torch.bfloat16), torch.randn(1,3,4096, device="cuda", dtype=torch.bfloat16)


def _forward(model, family, x, text):
    t = torch.tensor([.5], device="cuda", dtype=torch.bfloat16)
    if family == "krea2_raw":
        return model(x, text, t, torch.zeros(1,7,3, device="cuda"), torch.ones(1,7, device="cuda", dtype=torch.bool))
    if family == "z_image":
        return model(x=[x], t=t, cap_feats=[text]).sample[0]
    from library.models.qwen_image_2_1.family import forward_for_loss
    return forward_for_loss(model, x, text, torch.ones(1,3, device="cuda", dtype=torch.bool), t)


def _run(model, adapters, family, inputs, backward=True):
    adapters.zero_grad(set_to_none=True)
    x = inputs[0].clone().requires_grad_(backward)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = _forward(model, family, x, inputs[1])
    if not backward:
        return output.detach()
    output.float().square().mean().backward()
    grads = [p.grad for p in adapters.parameters()]
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.count_nonzero()
    assert all(g is not None and torch.isfinite(g).all() and g.count_nonzero() for g in grads)
    return output.detach(), x.grad.clone(), [g.clone() for g in grads]


def _enable(model, family, count, mode):
    if family == "krea2_raw":
        model.enable_block_swap(count, torch.device("cuda"), restore_mode=mode, transfer_dtype="bf16")
    elif family == "z_image":
        from library.models.z_image.block_swap import enable_z_image_block_swap
        enable_z_image_block_swap(model, count, torch.device("cuda"), restore_mode=mode, transfer_dtype="bf16")
    else:
        from library.models.qwen_image_2_1.block_swap import enable_qwen_image_2_1_block_swap
        enable_qwen_image_2_1_block_swap(model, count, torch.device("cuda"), restore_mode=mode, transfer_dtype="bf16")
    model.move_to_device_except_swap_blocks(torch.device("cuda"))
    model.prepare_block_swap_before_forward(free_cache=False)
    return model.offloader


@pytest.mark.parametrize("family", ["krea2_raw", "z_image", "qwen_image_2_1"])
@pytest.mark.parametrize("count", [1,2])
@pytest.mark.parametrize("restore_mode", ["foreach", "slab"])
def test_real_family_mixed_nf4_training_swap_roundtrip(tmp_path, family, count, restore_mode):
    torch.manual_seed(37)
    source = _construct(family).bfloat16()
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, .03)
            elif name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.zero_()
    path = tmp_path / "mixed.safetensors"
    save_mixed_nf4(source, path, family=family, source_sha256="a"*64)
    baseline, swapped = _load(family, path), _load(family, path)
    base_adapter, swap_adapter = _adapters(baseline, family), _adapters(swapped, family)
    swap_adapter.load_state_dict(base_adapter.state_dict())
    baseline.cuda().train().enable_gradient_checkpointing()
    swapped.train().enable_gradient_checkpointing()
    offloader = _enable(swapped, family, count, restore_mode)
    # Bootstrap repeats this after adapter application.
    swapped.enable_gradient_checkpointing()
    inputs = _inputs(family)
    packed = [(module, module.weight, module.weight.quant_state, module.weight.detach().cpu().clone())
        for module in swapped.modules() if isinstance(module, BF16AutocastLinear4bit)]
    try:
        expected = _run(baseline, base_adapter, family, inputs)
        actual = _run(swapped, swap_adapter, family, inputs)
        # Identical BF16 computations and packed bases: no quantization tolerance.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        swapped.switch_block_swap_for_inference()
        with torch.no_grad():
            torch.testing.assert_close(_run(swapped, swap_adapter, family, inputs, False), expected[0], rtol=0, atol=0)
        offloader.restore_blocks_to_device(getattr(swapped, {"krea2_raw":"blocks", "z_image":"layers", "qwen_image_2_1":"transformer_blocks"}[family]), torch.device("cuda"))
        torch.cuda.synchronize()
        for module, weight, state, payload in packed:
            assert module.weight is weight and module.weight.quant_state is state
            assert weight.device.type == "cuda" and weight.grad is None
            assert state.absmax.device.type == "cuda"
            if module.bias is not None:
                assert module.bias.device.type == "cuda"
            torch.testing.assert_close(weight.cpu(), payload, rtol=0, atol=0)
        swapped.switch_block_swap_for_training()
        torch.testing.assert_close(_run(swapped, swap_adapter, family, inputs), expected, rtol=0, atol=0)
    finally:
        for index in list(offloader.futures):
            offloader._wait_blocks_move(index, phase="test_cleanup")
        offloader.thread_pool.shutdown(wait=True)


def test_krea2_nf4_mlp_uses_compute_dtype_not_packed_weight_dtype(tmp_path):
    """Packed NF4 storage must not leak uint8 into the activation path."""
    family = "krea2_raw"
    source = _construct(family).bfloat16()
    path = tmp_path / "mixed.safetensors"
    save_mixed_nf4(source, path, family=family, source_sha256="a" * 64)
    model = _load(family, path).cuda().train()
    block = model.blocks[0]
    assert block.mlp.gate.weight.dtype == torch.uint8
    assert block.mlp.gate.compute_dtype == torch.bfloat16
    hidden = torch.randn(1, 2, model.config.features, device="cuda", dtype=torch.bfloat16)
    output = block.mlp(hidden)
    assert output.dtype == torch.bfloat16
