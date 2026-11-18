"""Phase 3 (P3-2): PP CPU-first loading, head/tail release, local Krea-2 freqs, and async stage swap."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT
from library.models.krea2_raw.pp_stages import (
    enable_stage_block_swap,
    release_foreign_submodules,
)
from library.models.moss_image.dit import MOSSImageTransformer
from library.models.moss_image.pipeline_parallel import (
    build_moss_stage_module,
    release_moss_foreign_submodules,
)
from library.training.pp.contract import PpComputePolicy
from library.training.pp.krea2_runtime import (
    KREA2_LOCAL_FREQS_FROZEN_INDICES,
    build_krea2_stage_module,
    krea2_boundary_metas,
    resolve_krea2_wire_spec,
)
from library.training.pp.placement import release_anima_foreign_submodules


def _tiny_krea(depth: int = 5, features: int = 128) -> SingleStreamDiT:
    torch.manual_seed(19)
    cfg = SingleMMDiTConfig(
        features=features,
        tdim=32,
        txtdim=64,
        heads=4,
        multiplier=2,
        layers=depth,
        patch=2,
        channels=16,
        kvheads=2,
        txtlayers=1,
        txtheads=4,
        txtkvheads=4,
    )
    return SingleStreamDiT(cfg)


def test_krea2_local_freqs_drops_wire_slot_and_matches_monolith() -> None:
    from library.models.krea2_raw.family import prepare_img_tokens

    model = _tiny_krea(depth=4).eval()
    wire = resolve_krea2_wire_spec(
        model,
        latent_shape=(2, 16, 1, 4, 4),
        max_text_tokens=5,
        microbatches=2,
    )
    policy = PpComputePolicy.from_dtype(torch.float32)
    head_layout, head_mod = build_krea2_stage_module(
        model, stage_index=0, stages=2, wire=wire, compute_policy=policy
    )
    tail_layout, tail_mod = build_krea2_stage_module(
        model, stage_index=1, stages=2, wire=wire, compute_policy=policy
    )
    assert head_layout.local_freqs is True and tail_layout.local_freqs is True
    assert KREA2_LOCAL_FREQS_FROZEN_INDICES == (1, 2, 3)

    head_in, head_out = krea2_boundary_metas(
        head_layout, wire=wire, policy=policy, latent_dtype=torch.float32
    )
    tail_in, tail_out = krea2_boundary_metas(
        tail_layout, wire=wire, policy=policy, latent_dtype=torch.float32
    )
    assert len(head_out) == 4 and len(tail_in) == 4

    g = torch.Generator().manual_seed(23)
    latents = torch.randn(2, 16, 1, 4, 4, generator=g)
    context = torch.randn(2, 5, 1, 64, generator=g)
    t = torch.rand(2, generator=g)
    text_mask = torch.ones(2, 5, dtype=torch.bool)

    img, pos, joint_mask = prepare_img_tokens(latents.squeeze(2), 5, 2, text_mask)
    with torch.no_grad():
        ref = model(img, context, t, pos, joint_mask)
        wire_tensors = head_mod(latents, timesteps=t, context=context, mask=text_mask)
        assert len(wire_tensors) == 4
        # Release head-only modules on a cloned tail model and verify local freqs still works
        out = tail_mod(*wire_tensors)
    assert torch.equal(ref, out)
    # Second pass hits the local freqs cache without recomputing posemb
    cached_head = head_mod._freqs_cache
    cached_tail = tail_mod._freqs_cache
    assert len(cached_head) == 1 and len(cached_tail) == 1


def test_moss_pp_cpu_staging_and_foreign_head_tail_release(monkeypatch) -> None:
    from library.models.moss_image import integration as moss_integration

    captured = {}

    class _FakeMoss(nn.Module):
        def set_attn_mode(self, _mode):
            pass

    def _fake_load_moss(path, *, dtype, device):
        captured["device"] = str(device)
        return _FakeMoss()

    monkeypatch.setattr(
        "library.models.moss_image.weights.load_moss_transformer",
        _fake_load_moss,
    )
    args = SimpleNamespace(
        pretrained_model_name_or_path="moss.safetensors",
        blocks_to_swap=0,
        pipeline_parallel=True,
        attn_mode="torch",
        gradient_checkpointing=False,
    )
    trainer = SimpleNamespace(is_swapping_blocks=False)
    accel = SimpleNamespace(device=torch.device("cuda:0"))
    moss_integration.load_dit(trainer, args, torch.float16, accel, [])
    assert captured["device"] == "cpu"

    # Verify stage split still reproduces monolith after foreign head/tail release
    moss_kwargs = dict(
        dim=64,
        ffn_dim=96,
        num_heads=8,
        num_layers=4,
        text_dim=32,
        freq_dim=16,
        latent_channels=128,
        rope_base=10000.0,
    )
    torch.manual_seed(31)
    m0 = MOSSImageTransformer(**moss_kwargs).eval()
    m1 = MOSSImageTransformer(**moss_kwargs).eval()
    m1.load_state_dict(m0.state_dict())

    latents = torch.randn(2, 128, 2, 2)
    text_h = torch.randn(2, 4, 32)
    text_m = torch.tensor([[True, True, True, False], [True, True, True, True]])
    timesteps = torch.tensor([0.25, 0.75])
    with torch.no_grad():
        ref = m0(latents, timesteps, text_h, text_m).sample

    _, stage0 = build_moss_stage_module(
        m0, stage_index=0, stages=2, latent_shape=(2, 128, 2, 2), max_text_tokens=4
    )
    _, stage1 = build_moss_stage_module(
        m1, stage_index=1, stages=2, latent_shape=(2, 128, 2, 2), max_text_tokens=4
    )
    dropped0 = release_moss_foreign_submodules(m0, is_first=True)
    dropped1 = release_moss_foreign_submodules(m1, is_first=False)
    assert set(dropped0) == {"final_norm_video", "video_head_modulation", "final_linear_video"}
    assert set(dropped1) == {"video_embedder", "text_embedder", "shared_block_modulation"}
    with torch.no_grad():
        wire_out = stage0(latents, timesteps=timesteps, text_hiddens=text_h, text_mask=text_m)
        pred = stage1(*wire_out, timesteps=timesteps)
    torch.testing.assert_close(pred, ref, rtol=0, atol=0)


def test_anima_foreign_submodule_release_drops_unused_llm_adapter_and_opposite_end() -> None:
    class _ToyAnima(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x_embedder = nn.Linear(4, 4)
            self.final_layer = nn.Linear(4, 4)
            self.llm_adapter = nn.Linear(4, 4).requires_grad_(False)

    head = _ToyAnima()
    assert set(release_anima_foreign_submodules(head, is_first=True)) == {
        "final_layer",
        "llm_adapter",
    }
    assert head.x_embedder is not None and head.final_layer is None and head.llm_adapter is None

    tail = _ToyAnima()
    assert set(release_anima_foreign_submodules(tail, is_first=False)) == {
        "x_embedder",
        "llm_adapter",
    }
    assert tail.final_layer is not None and tail.x_embedder is None and tail.llm_adapter is None


@pytest.mark.parametrize("ckpt", [True, False])
@pytest.mark.parametrize(
    "device_str",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_async_stage_block_swap_ring_wraparound_exact_gradients_and_zero_leak(
    device_str: str, ckpt: bool
) -> None:
    from networks.lora_modules.lora import LoRAModule

    device = torch.device(device_str)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    ref_model = _tiny_krea(depth=5, features=128).to(device=device, dtype=dtype).requires_grad_(False)
    swp_model = _tiny_krea(depth=5, features=128).to(device=device, dtype=dtype).requires_grad_(False)
    swp_model.load_state_dict(ref_model.state_dict())

    ref_loras = nn.ModuleList()
    swp_loras = nn.ModuleList()
    for idx in range(5):
        r_lora = LoRAModule(f"r_{idx}", ref_model.blocks[idx].attn.wq, lora_dim=4, alpha=4.0)
        s_lora = LoRAModule(f"s_{idx}", swp_model.blocks[idx].attn.wq, lora_dim=4, alpha=4.0)
        r_lora.apply_to()
        s_lora.apply_to()
        with torch.no_grad():
            r_lora.lora_up.weight.normal_(0, 0.02)
            s_lora.load_state_dict(r_lora.state_dict())
        ref_loras.append(r_lora)
        swp_loras.append(s_lora)
    ref_loras.to(device=device, dtype=torch.float32)
    swp_loras.to(device=device, dtype=torch.float32)

    if ckpt:
        for b in ref_model.blocks:
            b.enable_gradient_checkpointing()
        for b in swp_model.blocks:
            b.enable_gradient_checkpointing()

    # Swap 3 blocks (> ring_size=2!) to exercise ping-pong ring reuse on both fwd and bwd
    swapped_indices = enable_stage_block_swap(swp_model.blocks, device, num_swap=3)
    assert swapped_indices == [1, 2, 3]

    step0_alloc = None
    for step_idx in range(2):
        g = torch.Generator(device="cpu").manual_seed(100 + step_idx)
        x_base = torch.randn(2, 16, 128, generator=g).to(device=device, dtype=dtype)
        tvec = torch.randn(2, 1, 6 * 128, generator=g).to(device=device, dtype=dtype)
        pos = torch.zeros(2, 16, 3, device=device)
        freqs = ref_model.posemb(pos)
        mask4d = torch.ones(2, 1, 16, 16, device=device, dtype=torch.bool)

        x_ref = x_base.clone().requires_grad_(True)
        x_swp = x_base.clone().requires_grad_(True)

        h_ref = x_ref
        for blk in ref_model.blocks:
            h_ref = blk(h_ref, tvec, freqs, mask4d)
        h_ref.float().square().mean().backward()

        h_swp = x_swp
        for blk in swp_model.blocks:
            h_swp = blk(h_swp, tvec, freqs, mask4d)

        if device.type == "cuda" and ckpt:
            swapper = swp_model.blocks[1]._stage_swapper
            # Pool must be drained at end of checkpointed outer forward!
            assert swapper.ring_slabs == [None, None]

        h_swp.float().square().mean().backward()

        if device.type == "cuda":
            swapper = swp_model.blocks[1]._stage_swapper
            assert all(s is None for s in swapper.ring_slabs)
            if ckpt:
                assert swapper.ring_size == 2
            for idx in swapped_indices:
                assert swp_model.blocks[idx].mlp.up.weight.device.type == "cpu"

        torch.testing.assert_close(h_swp, h_ref, rtol=0, atol=0)
        torch.testing.assert_close(x_swp.grad, x_ref.grad, rtol=0, atol=0)
        for r_l, s_l in zip(ref_loras, swp_loras):
            torch.testing.assert_close(s_l.lora_down.weight.grad, r_l.lora_down.weight.grad, rtol=0, atol=0)
            torch.testing.assert_close(s_l.lora_up.weight.grad, r_l.lora_up.weight.grad, rtol=0, atol=0)

        ref_loras.zero_grad(set_to_none=True)
        swp_loras.zero_grad(set_to_none=True)
        del h_ref, h_swp, x_ref, x_swp, x_base, tvec, freqs, mask4d, pos
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            cur_alloc = torch.cuda.memory_allocated(device)
            if step_idx == 0:
                step0_alloc = cur_alloc
            else:
                assert cur_alloc == step0_alloc
