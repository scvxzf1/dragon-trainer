"""Small CPU acceptance contracts; no checkpoints, downloads, or CUDA required.

Set MOSS_OFFICIAL_SOURCE to the released pipeline_moss_image.py to enable
independent official-model and scheduler differential checks.
"""
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from library.models.moss_image.dit import MOSSImageTransformer
from library.models.moss_image.family import compute_noise_pred_and_target, forward_for_loss
from library.models.moss_image.latent import encode_moss_latents, decode_moss_latents, pack_moss_latents
from library.models.moss_image.lora_targets import moss_image_target_kwargs
from library.models.moss_image.sampling import MOSSImageScheduler, generate_moss_image
from library.models.moss_image.strategy import MossImageTextCache
from library.models.moss_image.cache_policy import text_cache_metadata
from library.models.moss_image.text import MossImageTextEncodingStrategy
from library.runtime.offloading import ModelOffloader
from library.training.preview_lifecycle import preview_lifecycle


@pytest.fixture
def tiny_model():
    torch.manual_seed(43)
    return MOSSImageTransformer(dim=32, ffn_dim=48, num_heads=1,
                               num_layers=3, text_dim=8, freq_dim=8)


@pytest.fixture(scope="module")
def official():
    source = os.environ.get("MOSS_OFFICIAL_SOURCE")
    if not source:
        pytest.skip("Set MOSS_OFFICIAL_SOURCE for official differential checks")
    assert Path(source).is_file(), source
    spec = importlib.util.spec_from_file_location("moss_official_contract", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_state_dict_math_and_padding(tiny_model, official):
    reference = official.MOSSImageTransformer(**dict(tiny_model.config))
    assert {k: v.shape for k, v in reference.state_dict().items()} == {
        k: v.shape for k, v in tiny_model.state_dict().items()}
    reference.load_training_state_dict(tiny_model.state_dict())
    z, text = torch.randn(2, 128, 2, 3), torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 0, 1]], dtype=torch.bool)
    times = torch.tensor([123.25, 987.5])
    actual = tiny_model(z, times, text, mask).sample
    torch.testing.assert_close(actual, reference(z, times, text, mask).sample, rtol=0, atol=0)
    text[~mask] = float("nan")
    torch.testing.assert_close(actual, tiny_model(z, times, text, mask).sample, rtol=0, atol=0)
    for row in range(2):
        single = tiny_model(z[row:row + 1], times[row:row + 1],
                            text[row:row + 1, mask[row]]).sample
        torch.testing.assert_close(actual[row:row + 1], single)
    tiny_model.to(dtype=torch.bfloat16)
    reference.to(dtype=torch.bfloat16)
    torch.testing.assert_close(tiny_model(z, times, text, mask).sample,
                               reference(z, times, text, mask).sample, rtol=0, atol=0)
    assert tiny_model.rope.bands.dtype == torch.float32
    with pytest.raises(RuntimeError):
        tiny_model.load_training_state_dict({"unexpected": torch.ones(1)})


def test_official_scheduler_grid_and_euler_step(official):
    for steps, shift in [(1, 5.0), (50, 5.0), (1003, 1.0), (17, 2.5)]:
        actual, expected = MOSSImageScheduler(shift=shift), official.MOSSImageScheduler(shift=shift)
        actual.set_timesteps(steps, device="cpu")
        expected.set_timesteps(steps, device="cpu")
        torch.testing.assert_close(actual.timesteps, expected.timesteps, rtol=0, atol=0)
        torch.testing.assert_close(actual.sigmas, expected.sigmas, rtol=0, atol=0)
        z, velocity = torch.randn(1, 128, 1, 1), torch.randn(1, 128, 1, 1)
        torch.testing.assert_close(actual.step(velocity, actual.timesteps[0], z).prev_sample,
                                   expected.step(velocity, expected.timesteps[0], z).prev_sample,
                                   rtol=0, atol=0)


def test_cfg_sampler_matches_official_pipeline(tiny_model, official):
    reference = official.MOSSImageTransformer(**dict(tiny_model.config))
    reference.load_state_dict(tiny_model.state_dict())
    tiny_model.to(dtype=torch.bfloat16)
    reference.to(dtype=torch.bfloat16)
    pipeline = official.MOSSImagePipeline(reference, official.MOSSImageScheduler())
    pipeline.set_progress_bar_config(disable=True)
    positive = torch.randn(1, 3, 8, dtype=torch.bfloat16)
    negative = torch.randn(1, 2, 8, dtype=torch.bfloat16)
    pmask, nmask = torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool)
    z = torch.randn(1, 128, 1, 1)
    actual = generate_moss_image(tiny_model, [positive, pmask], [negative, nmask],
                                 height=16, width=16, sample_steps=3, guidance_scale=3,
                                 device="cpu", dtype=torch.bfloat16,
                                 uncond_skip_layers=[1], latents=z.clone())
    expected = pipeline(prompt_embeds=positive, prompt_attention_mask=pmask,
                        negative_prompt_embeds=negative, negative_prompt_attention_mask=nmask,
                        height=16, width=16, num_inference_steps=3, guidance_scale=3,
                        uncond_skip_layers=[1], latents=z.clone(), output_type="latent").images
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_effective_16x_pack_order_bn_and_inverse():
    native = torch.arange(32 * 4 * 6, dtype=torch.float32).reshape(1, 32, 4, 6)
    vae = SimpleNamespace(
        bn=SimpleNamespace(running_mean=torch.linspace(-2, 2, 128),
                           running_var=torch.linspace(0.5, 2.5, 128)),
        config=SimpleNamespace(batch_norm_eps=1e-4),
        encode=lambda images: SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: native)),
        decode=lambda z, **kw: (z,),
    )
    normalized = encode_moss_latents(vae, torch.zeros(1, 3, 32, 48))
    assert normalized.shape == (1, 128, 2, 3)
    # Spatial offsets occupy consecutive packed channels, as in FLUX.2.
    torch.testing.assert_close(pack_moss_latents(native)[0, :4, 0, 0],
                               torch.tensor([0., 1., 6., 7.]))
    expected_first = (torch.tensor([0., 1., 6., 7.]) - vae.bn.running_mean[:4]) / (
        vae.bn.running_var[:4] + vae.config.batch_norm_eps).sqrt()
    torch.testing.assert_close(normalized[0, :4, 0, 0], expected_first)
    torch.testing.assert_close(decode_moss_latents(vae, normalized), native)
    with pytest.raises(ValueError, match="divisible by 16"):
        encode_moss_latents(vae, torch.zeros(1, 3, 31, 48))


def test_flow_noising_target_and_timestep_contract(monkeypatch):
    import library.models.moss_image.family as family
    import library.training.router_conditioning as routing
    latents = torch.randn(2, 128, 2, 3)
    noise, sigma = torch.full_like(latents, 2.0), torch.tensor([0.25, 0.75])
    seen = {}
    def predict(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(sample=kwargs["hidden_states"] * 0)
    monkeypatch.setattr(torch, "randn_like", lambda x: noise)
    monkeypatch.setattr(family, "uniform_grid_sigmas", lambda *a, **k: sigma)
    monkeypatch.setattr(routing, "apply_router_conditioning", lambda **kw: 0)
    ctx = SimpleNamespace(network=None, unet=predict, weight_dtype=torch.float32,
                          args=SimpleNamespace(gradient_checkpointing=False),
                          accelerator=SimpleNamespace(device="cpu", autocast=nullcontext))
    conditions = [torch.randn(2, 3, 8), torch.ones(2, 3, dtype=torch.bool)]
    pred, target, times, weights = compute_noise_pred_and_target(
        SimpleNamespace(), ctx, latents, {}, conditions)
    torch.testing.assert_close(seen["hidden_states"][0], 0.75 * latents[0] + 0.25 * noise[0])
    torch.testing.assert_close(seen["hidden_states"][1], 0.25 * latents[1] + 0.75 * noise[1])
    torch.testing.assert_close(target, noise - latents)
    torch.testing.assert_close(times, torch.tensor([250., 750.]))
    torch.testing.assert_close(seen["timestep"], times)
    assert pred.shape == target.shape == latents.shape
    assert weights.shape == (2, 1, 1, 1) and weights.eq(1).all()
    with pytest.raises(ValueError, match="128"):
        forward_for_loss(predict, latents.unsqueeze(2), conditions, sigma)


def test_qwen_gdn_never_receives_padding():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.empty(0))
            self.lengths = []
        def forward(self, input_ids, attention_mask, **kwargs):
            assert attention_mask.all() and input_ids.ne(0).all()
            self.lengths.append(input_ids.shape[1])
            return SimpleNamespace(last_hidden_state=input_ids[..., None].expand(-1, -1, 4096).float())
    encoder = Encoder()
    ids = torch.tensor([[3, 4, 0], [5, 6, 7]])
    hidden, mask = MossImageTextEncodingStrategy().encode_tokens(None, [encoder], [ids, ids.ne(0)])
    assert encoder.lengths == [2, 3]
    assert torch.equal(mask, ids.ne(0)) and hidden[0, 2].eq(0).all()
    torch.testing.assert_close(hidden[..., 0], ids.float())


def test_cache_precision_identity_is_mandatory_even_when_checks_skipped(tmp_path):
    nf4 = MossImageTextCache(True, 1, True, quantization="nf4")
    bf16 = MossImageTextCache(True, 1, True, quantization="bf16")
    assert nf4.cache_suffix != bf16.cache_suffix
    path = str(tmp_path / "text.safetensors")
    tensors = {"hiddens": torch.ones(2, 4096, dtype=torch.bfloat16),
               "mask": torch.ones(2, dtype=torch.bool),
               "caption_dropout_rate": torch.tensor(0.)}
    save_file(tensors, path, metadata=text_cache_metadata("nf4"))
    assert nf4.is_disk_cached_outputs_expected(path)
    assert not bf16.is_disk_cached_outputs_expected(path)
    with pytest.raises(ValueError, match="quantization"):
        bf16.load_outputs_npz(path)
    metadata = text_cache_metadata("nf4")
    metadata["model_family"] = "anima"
    save_file(tensors, path, metadata=metadata)
    assert not nf4.is_disk_cached_outputs_expected(path)


def test_text_runtime_rejects_fp16_and_cpu_nf4_before_loading(tmp_path):
    from library.models.moss_image.text_encoder_runtime import MossTextEncoderRuntime
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="BF16"):
        MossTextEncoderRuntime(str(tmp_path), dtype=torch.float16, device="cpu", quantization="nf4")
    runtime = MossTextEncoderRuntime(str(tmp_path), dtype=torch.bfloat16,
                                     device="cpu", quantization="nf4")
    assert runtime._encoder is None and not runtime.training
    with pytest.raises(RuntimeError, match="requires CUDA"):
        runtime(input_ids=torch.ones(1, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="BF16"):
        runtime.to(dtype=torch.float16)


def test_plain_lora_attention_only_and_checkpoint_roundtrip(tiny_model, tmp_path):
    from networks.lora_anima import create_network, create_network_from_weights
    base = deepcopy(tiny_model).requires_grad_(False)
    tiny_model.requires_grad_(False)
    network = create_network(1.0, 2, 2, None, [], tiny_model,
                             model_family="moss_image", **moss_image_target_kwargs())
    assert len(network.unet_loras) == 4 * len(tiny_model.layers)
    assert not network.text_encoder_loras
    assert {m.lora_name for m in network.unet_loras} == {
        f"lora_unet_layers_{i}_proj_{p}" for i in range(3) for p in "qkvo"}
    network.apply_to([], tiny_model, False, True)
    with torch.no_grad():
        for p in network.parameters():
            p.normal_(0, 0.2)
    z, text, mask = torch.randn(1, 128, 2, 2), torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool)
    original = base(z, 500., text, mask).sample
    output = tiny_model(z, 500., text, mask).sample
    assert not torch.allclose(original, output)
    output.square().mean().backward()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in network.parameters())
    path = str(tmp_path / "moss-lora.safetensors")
    network.save_weights(path, torch.float32, metadata={})
    with safe_open(path, framework="pt") as f:
        assert f.metadata()["ss_model_family"] == "moss_image"
    restored, _ = create_network_from_weights(1.0, path, None, [], base)
    restored.apply_to([], base, False, True)
    restored.load_weights(path)
    torch.testing.assert_close(base(z, 500., text, mask).sample, output.detach())


class _LogicalSwap:
    """Exercise the real forward-only ring plan without allocating CUDA storage."""
    def __init__(self, depth):
        self.num_blocks, self.blocks_to_swap = depth, 1
        self.forward_only = False
        self.prepares, self.events = 0, []
    def set_forward_only(self, mode):
        self.forward_only = mode
    def prepare_block_devices_before_forward(self, blocks, **kwargs):
        self.prepares += 1
        self.resident = set(range(self.num_blocks - self.blocks_to_swap))
    def wait_for_block(self, index):
        assert index in self.resident, f"block {index} was not prefetched"
        self.events.append(index)
    def _submit_move_blocks(self, blocks, outgoing, incoming, **kwargs):
        self.resident.remove(outgoing)
        self.resident.add(incoming)
    submit_move_blocks = ModelOffloader.submit_move_blocks


@pytest.mark.parametrize("skip", [(), (1,)])
def test_cfg_swap_ring_keeps_both_branches_complete(tiny_model, skip):
    condition = [torch.randn(1, 2, 8), torch.ones(1, 2, dtype=torch.bool)]
    negative = [torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool)]
    expected = generate_moss_image(tiny_model, condition, negative, height=16, width=16,
                                   sample_steps=2, guidance_scale=3, device="cpu",
                                   dtype=torch.float32, uncond_skip_layers=skip)
    offloader = _LogicalSwap(len(tiny_model.layers))
    tiny_model.offloader, tiny_model.blocks_to_swap = offloader, 1
    with preview_lifecycle(tiny_model, None):
        result = generate_moss_image(tiny_model, condition, negative, height=16, width=16,
                                     sample_steps=2, guidance_scale=3, device="cpu",
                                     dtype=torch.float32, uncond_skip_layers=skip)
    assert result.shape == (1, 128, 1, 1) and result.dtype == torch.float32
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert offloader.prepares == 2  # Enter inference, restore training; no per-forward prepare.
    assert offloader.events == list(range(3)) * 4
    assert offloader.forward_only is False


def test_batch_gt_1_block_swap_ring_and_compat_contract(tiny_model):
    from library.training.compat_matrix import check_training_compat

    offloader = _LogicalSwap(len(tiny_model.layers))
    tiny_model.offloader, tiny_model.blocks_to_swap = offloader, 1
    tiny_model.enable_gradient_checkpointing()
    offloader.prepare_block_devices_before_forward(tiny_model.layers)

    z = torch.randn(2, 128, 2, 2, requires_grad=True)
    text = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.bool)
    times = torch.tensor([250.0, 750.0])

    out = tiny_model(z, times, text, mask).sample
    out.square().mean().backward()
    assert out.shape == (2, 128, 2, 2)
    assert z.grad is not None and torch.isfinite(z.grad).all()
    # Layer-outer execution visits each block once for the whole batch=2 forward.
    assert offloader.events == list(range(len(tiny_model.layers)))

    report = check_training_compat(
        {
            "model_family": "moss_image",
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            "network_train_unet_only": True,
            "use_vae_cache": True,
            "use_text_cache": True,
            "blocks_to_swap": 12,
            "train_batch_size": 2,
            "datasets": [{"batch_size": 2}],
        }
    )
    assert not any(
        issue.code == "moss_image_train_batch_size"
        for issue in (*report.errors, *report.warnings)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for packed/batched MOSS test")
def test_cuda_packed_and_batched_moss_matches_single_sample_and_swaps(tiny_model):
    device = torch.device("cuda")
    model = deepcopy(tiny_model).to(device=device, dtype=torch.float32)
    model.enable_gradient_checkpointing()

    z = torch.randn(2, 128, 2, 3, device=device, requires_grad=True)
    text = torch.randn(2, 4, 8, device=device)
    mask_varlen = torch.tensor(
        [[1, 1, 0, 0], [0, 1, 1, 1]], device=device, dtype=torch.bool
    )
    text_poisoned = text.clone()
    text_poisoned[~mask_varlen] = float("nan")
    times = torch.tensor([125.0, 875.0], device=device)

    out_packed = model(z, times, text_poisoned, mask_varlen).sample
    with torch.no_grad():
        ref0 = model(z[0:1], times[0:1], text[0:1, mask_varlen[0]]).sample
        ref1 = model(z[1:2], times[1:2], text[1:2, mask_varlen[1]]).sample
    torch.testing.assert_close(
        out_packed, torch.cat([ref0, ref1], dim=0), rtol=1e-4, atol=1e-4
    )
    out_packed.square().mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()

    z.grad = None
    mask_eq = torch.tensor(
        [[1, 1, 0, 0], [0, 1, 0, 1]], device=device, dtype=torch.bool
    )
    out_eq = model(z, times, text, mask_eq).sample
    with torch.no_grad():
        eq0 = model(z[0:1], times[0:1], text[0:1, mask_eq[0]]).sample
        eq1 = model(z[1:2], times[1:2], text[1:2, mask_eq[1]]).sample
    torch.testing.assert_close(
        out_eq, torch.cat([eq0, eq1], dim=0), rtol=1e-4, atol=1e-4
    )

    # Real LoRA training freezes the base DiT (so ModelOffloader actually swaps
    # frozen block parameters to CPU and relies on nn.Module.__call__ backward
    # hooks to bring them back during checkpoint recomputation).
    from networks.lora_anima import create_network

    noswap_ref = deepcopy(tiny_model).to(device=device, dtype=torch.float16).requires_grad_(False)
    noswap_net = create_network(
        1.0, 2, 2, None, [], noswap_ref, model_family="moss_image", **moss_image_target_kwargs()
    )
    noswap_net.apply_to([], noswap_ref, False, True)
    noswap_net.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for p in noswap_net.parameters():
            p.normal_(0, 0.1)

    swap_model = deepcopy(tiny_model).to(dtype=torch.float16).requires_grad_(False)
    swap_net = create_network(
        1.0, 2, 2, None, [], swap_model, model_family="moss_image", **moss_image_target_kwargs()
    )
    swap_net.apply_to([], swap_model, False, True)
    swap_net.to(device=device, dtype=torch.float32)
    swap_net.load_state_dict(noswap_net.state_dict())
    swap_model.enable_gradient_checkpointing()
    swap_model.enable_block_swap(1, device)
    swap_model.move_to_device_except_swap_blocks(device)
    swap_model.prepare_block_swap_before_forward()

    for _step in range(2):
        z_ref = z.detach().to(torch.float16).requires_grad_(True)
        z16 = z.detach().to(torch.float16).requires_grad_(True)
        t16 = text.to(torch.float16)
        noswap_net.zero_grad(set_to_none=True)
        swap_net.zero_grad(set_to_none=True)
        out_ref = noswap_ref(z_ref, times, t16, mask_varlen).sample
        out_ref.float().square().mean().backward()
        out_swap = swap_model(z16, times, t16, mask_varlen).sample
        out_swap.float().square().mean().backward()
        torch.testing.assert_close(out_swap, out_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(z16.grad, z_ref.grad, rtol=1e-3, atol=1e-3)
        for p_swap, p_ref in zip(swap_net.parameters(), noswap_net.parameters()):
            assert p_swap.grad is not None and p_ref.grad is not None
            torch.testing.assert_close(p_swap.grad, p_ref.grad, rtol=2e-3, atol=2e-3)
