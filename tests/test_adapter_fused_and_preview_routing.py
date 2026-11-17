"""Regression tests for fused QKV LoKr/LoHa round-trips, preview router reset, and cross-family layer ranges."""

from __future__ import annotations

import torch
from safetensors.torch import load_file

from library.training.preview_lifecycle import preview_lifecycle
from networks.lora_anima import create_network, create_network_from_weights
from networks.lora_anima.targeting import collect_lora_target_candidates


class _FusedSelfAttn(torch.nn.Module):
    def __init__(self, in_dim: int = 16, head_dim: int = 16):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(in_dim, 3 * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(head_dim, in_dim, bias=False)


class _FusedBlock(torch.nn.Module):
    def __init__(self, in_dim: int = 16, head_dim: int = 16):
        super().__init__()
        self.self_attn = _FusedSelfAttn(in_dim, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.self_attn.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        return self.self_attn.o_proj(q + k + v)


class _TinyFusedDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_FusedBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](x)


def test_anima_fused_qkv_lokr_roundtrip_and_kronecker_slice_equivalence(tmp_path):
    torch.manual_seed(0)
    dit = _TinyFusedDiT()
    net = create_network(
        1.0,
        network_dim=8,
        network_alpha=8.0,
        vae=None,
        text_encoders=[],
        unet=dit,
        use_lokr="true",
        lokr_factor="4",
        lokr_full_factor="true",
        unet_target_replace_modules='["_FusedBlock"]',
    )
    net.apply_to([], dit, apply_text_encoder=False, apply_unet=True)
    for lora in net.unet_loras:
        with torch.no_grad():
            lora.lokr_w1.normal_(0.0, 0.2)
            lora.lokr_w2.normal_(0.0, 0.2)

    net.eval()
    x = torch.randn(2, 5, 16)
    expected = dit(x).detach().clone()

    qkv_lora = next(m for m in net.unet_loras if m.lora_name.endswith("self_attn_qkv_proj"))
    fused_delta = torch.kron(qkv_lora.lokr_w1.detach(), qkv_lora.lokr_w2.detach())

    ckpt = tmp_path / "anima_lokr.safetensors"
    net.save_weights(str(ckpt), torch.float32, {"ss_model_family": "anima"})

    on_disk = load_file(str(ckpt))
    base = "lora_unet_blocks_0_self_attn"
    split_deltas = [
        torch.kron(on_disk[f"{base}_{c}_proj.lokr_w1"], on_disk[f"{base}_{c}_proj.lokr_w2"])
        for c in ("q", "k", "v")
    ]
    torch.testing.assert_close(torch.cat(split_deltas, dim=0), fused_delta, atol=1e-6, rtol=1e-6)

    dit_reloaded = _TinyFusedDiT()
    dit_reloaded.load_state_dict(dit.state_dict())
    net2, weights_sd = create_network_from_weights(
        1.0,
        str(ckpt),
        None,
        [],
        dit_reloaded,
        for_inference=True,
        unet_target_replace_modules='["_FusedBlock"]',
    )
    net2.apply_to([], dit_reloaded, apply_text_encoder=False, apply_unet=True)
    info = net2.load_state_dict(weights_sd, strict=True)
    assert not info.missing_keys and not info.unexpected_keys
    net2.eval()
    torch.testing.assert_close(dit_reloaded(x), expected, atol=1e-6, rtol=1e-6)


def test_anima_fused_qkv_loha_roundtrip(tmp_path):
    torch.manual_seed(1)
    dit = _TinyFusedDiT()
    net = create_network(
        1.0,
        network_dim=4,
        network_alpha=4.0,
        vae=None,
        text_encoders=[],
        unet=dit,
        use_loha="true",
        unet_target_replace_modules='["_FusedBlock"]',
    )
    net.apply_to([], dit, apply_text_encoder=False, apply_unet=True)
    for lora in net.unet_loras:
        with torch.no_grad():
            lora.hada_w1_a.normal_(0.0, 0.2)
            lora.hada_w1_b.normal_(0.0, 0.2)
            lora.hada_w2_a.normal_(0.0, 0.2)
            lora.hada_w2_b.normal_(0.0, 0.2)

    net.eval()
    x = torch.randn(2, 5, 16)
    expected = dit(x).detach().clone()

    ckpt = tmp_path / "anima_loha.safetensors"
    net.save_weights(str(ckpt), torch.float32, {"ss_model_family": "anima"})

    dit_reloaded = _TinyFusedDiT()
    dit_reloaded.load_state_dict(dit.state_dict())
    net2, weights_sd = create_network_from_weights(
        1.0,
        str(ckpt),
        None,
        [],
        dit_reloaded,
        for_inference=True,
        unet_target_replace_modules='["_FusedBlock"]',
    )
    net2.apply_to([], dit_reloaded, apply_text_encoder=False, apply_unet=True)
    info = net2.load_state_dict(weights_sd, strict=True)
    assert not info.missing_keys and not info.unexpected_keys
    net2.eval()
    torch.testing.assert_close(dit_reloaded(x), expected, atol=1e-6, rtol=1e-6)


def test_preview_lifecycle_resets_batch_bound_router_buffers():
    torch.manual_seed(2)
    dit = _TinyFusedDiT()
    net = create_network(
        1.0,
        network_dim=4,
        network_alpha=4.0,
        vae=None,
        text_encoders=[],
        unet=dit,
        use_moe_style="shared_A",
        num_experts="2",
        route_per_layer="true",
        router_source="sigma",
        unet_target_replace_modules='["_FusedBlock"]',
    )
    net.apply_to([], dit, apply_text_encoder=False, apply_unet=True)
    net.train()

    # Simulate a training step with batch_size = 3 (rebinds _sigma_features to (3, D)).
    net.set_sigma(torch.tensor([0.2, 0.5, 0.8]))
    _ = dit(torch.randn(3, 4, 16))

    # Entering preview with batch_size = 1 must not crash with shape mismatch (3 vs 1).
    with preview_lifecycle(dit, net):
        assert not net.training
        out_b1 = dit(torch.randn(1, 4, 16))
        assert out_b1.shape == (1, 4, 16)
    assert net.training


def test_layer_range_filters_layers_and_transformer_blocks_containers():
    class _Leaf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False)

    for container_name in ("blocks", "layers", "transformer_blocks"):
        root = torch.nn.Module()
        setattr(root, container_name, torch.nn.ModuleList([_Leaf(), _Leaf(), _Leaf()]))
        kept = collect_lora_target_candidates(
            root_module=root,
            prefix="lora_unet",
            target_replace_modules=["_Leaf"],
            exclude_patterns=[],
            include_patterns=[],
            is_unet=True,
            layer_start=1,
            layer_end=2,
            modules_dim=None,
            modules_alpha=None,
            reg_dims=None,
            default_dim=None,
            lora_dim=4,
            alpha=4.0,
        )
        assert [c.original_name for c in kept] == [f"{container_name}.1.proj"]


def test_heterogeneous_dims_and_unfused_te_llm_adapter_roundtrip(tmp_path):
    """Regression for verifier BLOCK-1 & BLOCK-2:
    - Unfused lora_te_* and llm_adapter q/k/v_proj must NOT be merged into fake qkv_proj.
    - Heterogeneous Anima dims (e.g. 32->96 self_attn, 16->64 cross_attn, 32->128 mlp)
      with factor=16 or factor=-1 under einsum=True must reload without shape mismatch.
    """

    class _UnfusedAttn(torch.nn.Module):
        def __init__(self, dim: int = 16):
            super().__init__()
            self.q_proj = torch.nn.Linear(dim, dim, bias=False)
            self.k_proj = torch.nn.Linear(dim, dim, bias=False)
            self.v_proj = torch.nn.Linear(dim, dim, bias=False)

    class _LLMAdapterBlock(torch.nn.Module):
        def __init__(self, dim: int = 16):
            super().__init__()
            self.self_attn = _UnfusedAttn(dim)

    class _HeteroBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.qkv_proj = torch.nn.Linear(32, 96, bias=False)
            self.cross_attn = torch.nn.Module()
            self.cross_attn.kv_proj = torch.nn.Linear(16, 64, bias=False)
            self.mlp = torch.nn.Linear(32, 128, bias=False)

    class _HeteroDiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([_HeteroBlock()])
            self.llm_adapter = torch.nn.Module()
            self.llm_adapter.blocks = torch.nn.ModuleList([_LLMAdapterBlock(16)])

    class _TinyTEBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _UnfusedAttn(16)

    class _TinyTE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_TinyTEBlock()])

    for label, extra in (
        ("lokr_f16", {"use_lokr": "true", "lokr_factor": "16", "lokr_full_factor": "true"}),
        ("lokr_auto", {"use_lokr": "true", "lokr_factor": "-1", "lokr_decompose_w2": "true"}),
        ("loha", {"use_loha": "true"}),
    ):
        dit = _HeteroDiT()
        te = _TinyTE()
        net = create_network(
            1.0,
            network_dim=4,
            network_alpha=4.0,
            vae=None,
            text_encoders=[te],
            unet=dit,
            unet_target_replace_modules='["_HeteroBlock", "_LLMAdapterBlock"]',
            text_encoder_target_replace_modules='["_TinyTEBlock"]',
            **extra,
        )
        net.apply_to([te], dit, apply_text_encoder=True, apply_unet=True)
        for lora in net.unet_loras + net.text_encoder_loras:
            for p in lora.parameters():
                with torch.no_grad():
                    p.normal_(0.0, 0.15)

        ckpt = tmp_path / f"{label}.safetensors"
        net.save_weights(
            str(ckpt),
            torch.float32,
            {
                "ss_model_family": "anima",
                "ss_unet_target_replace_modules": '["_HeteroBlock", "_LLMAdapterBlock"]',
                "ss_text_encoder_target_replace_modules": '["_TinyTEBlock"]',
            },
        )

        dit2 = _HeteroDiT()
        te2 = _TinyTE()
        net2, sd2 = create_network_from_weights(
            1.0,
            str(ckpt),
            None,
            [te2],
            dit2,
            for_inference=True,
            unet_target_replace_modules='["_HeteroBlock", "_LLMAdapterBlock"]',
            text_encoder_target_replace_modules='["_TinyTEBlock"]',
        )
        net2.apply_to([te2], dit2, apply_text_encoder=True, apply_unet=True)
        info = net2.load_state_dict(sd2, strict=True)
        assert not info.missing_keys and not info.unexpected_keys, (label, info)
