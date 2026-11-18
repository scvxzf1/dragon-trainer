from __future__ import annotations

import importlib
import math
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from library.runtime.int8_linear import (
    Int8FrozenLinear,
    patch_lora_frozen_base_forwards_with_int8,
)
from networks.plugins.loha import autograd as loha_autograd
from networks.plugins.loha.autograd import loha_linear, make_hada_weight
from networks.plugins.lokr import autograd as lokr_autograd


def _get_v100_compat_module():
    existing = sys.modules.get("networks.flash_attn_v100_compat")
    if existing is not None:
        return existing
    had_stub = "flash_attn_v100_cuda" in sys.modules
    if not had_stub:
        stub = types.ModuleType("flash_attn_v100_cuda")
        stub.fwd = None
        stub.bwd = None
        stub.varlen_fwd = None
        stub.varlen_bwd = None
        sys.modules["flash_attn_v100_cuda"] = stub
    try:
        return importlib.import_module("networks.flash_attn_v100_compat")
    finally:
        if not had_stub:
            sys.modules.pop("flash_attn_v100_cuda", None)


def _attn_bwd_math(
    qf: torch.Tensor,
    kf: torch.Tensor,
    vf: torch.Tensor,
    doutf: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytical attention backward without Autograd dispatch (custom_op runs post-autograd)."""
    hq, hk = qf.shape[-3], kf.shape[-3]
    rep = hq // hk
    kf_exp = kf.repeat_interleave(rep, dim=-3) if rep > 1 else kf
    vf_exp = vf.repeat_interleave(rep, dim=-3) if rep > 1 else vf
    scores = torch.matmul(qf, kf_exp.transpose(-1, -2)) * float(softmax_scale)
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        mask = torch.ones((sq, sk), device=qf.device, dtype=torch.bool).tril(
            diagonal=sk - sq
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    dv_exp = torch.matmul(probs.transpose(-1, -2), doutf)
    dp = torch.matmul(doutf, vf_exp.transpose(-1, -2))
    ds = probs * (dp - (dp * probs).sum(dim=-1, keepdim=True)) * float(softmax_scale)
    dq = torch.matmul(ds, kf_exp)
    dk_exp = torch.matmul(ds.transpose(-1, -2), qf)
    if rep > 1:
        dk = dk_exp.unflatten(-3, (hk, rep)).sum(dim=-3)
        dv = dv_exp.unflatten(-3, (hk, rep)).sum(dim=-3)
    else:
        dk, dv = dk_exp, dv_exp
    return dq, dk, dv


def _install_fake_v100_flash_kernels(monkeypatch: pytest.MonkeyPatch):
    v100_compat = _get_v100_compat_module()
    counts = {"varlen_fwd": 0, "varlen_bwd": 0, "dense_fwd": 0, "dense_bwd": 0}

    def fake_cuda_varlen_fwd(
        q,
        k,
        v,
        _out,
        cu_seqlens_q,
        cu_seqlens_k,
        _seqused_k,
        _leftpad_k,
        _block_table,
        _alibi_slopes,
        _max_seqlen_q,
        _max_seqlen_k,
        _dropout_p,
        softmax_scale,
        _zero_tensors,
        causal,
        _window_left,
        _window_right,
        _softcap,
        _return_softmax,
        _gen,
        *_extra,
    ):
        counts["varlen_fwd"] += 1
        batch = int(cu_seqlens_q.numel()) - 1
        outs = []
        lses = []
        for b in range(batch):
            q0, q1 = int(cu_seqlens_q[b].item()), int(cu_seqlens_q[b + 1].item())
            k0, k1 = int(cu_seqlens_k[b].item()), int(cu_seqlens_k[b + 1].item())
            qb = q[q0:q1].transpose(0, 1).float()  # [H, Sq, D]
            kb = k[k0:k1].transpose(0, 1).float()  # [Hk, Sk, D]
            vb = v[k0:k1].transpose(0, 1).float()
            if qb.shape[0] != kb.shape[0]:
                rep = qb.shape[0] // kb.shape[0]
                kb = kb.repeat_interleave(rep, dim=0)
                vb = vb.repeat_interleave(rep, dim=0)
            scores = torch.matmul(qb, kb.transpose(-1, -2)) * float(softmax_scale)
            if causal:
                sq, sk = scores.shape[-2], scores.shape[-1]
                mask = torch.ones((sq, sk), device=q.device, dtype=torch.bool).tril(
                    diagonal=sk - sq
                )
                scores = scores.masked_fill(~mask, float("-inf"))
            lse_b = torch.logsumexp(scores, dim=-1)
            probs = torch.softmax(scores, dim=-1)
            out_b = torch.matmul(probs, vb).transpose(0, 1).to(q.dtype)
            outs.append(out_b)
            lses.append(lse_b)
        out_flat = torch.cat(outs, dim=0).contiguous()
        lse_flat = torch.cat(lses, dim=1).contiguous()
        softmax = torch.empty((0,), device=q.device, dtype=q.dtype)
        rng_state = torch.zeros((2,), device="cpu", dtype=torch.int64)
        return out_flat, lse_flat, softmax, rng_state

    def fake_cuda_varlen_bwd(
        dout,
        q,
        k,
        v,
        _out,
        _lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        _alibi_slopes,
        _max_seqlen_q,
        _max_seqlen_k,
        _dropout_p,
        softmax_scale,
        _zero_tensors,
        causal,
        _window_left,
        _window_right,
        _softcap,
        _deterministic,
        _gen,
        _rng_state,
    ):
        counts["varlen_bwd"] += 1
        batch = int(cu_seqlens_q.numel()) - 1
        for b in range(batch):
            q0, q1 = int(cu_seqlens_q[b].item()), int(cu_seqlens_q[b + 1].item())
            k0, k1 = int(cu_seqlens_k[b].item()), int(cu_seqlens_k[b + 1].item())
            gq, gk, gv = _attn_bwd_math(
                q[q0:q1].transpose(0, 1).float(),
                k[k0:k1].transpose(0, 1).float(),
                v[k0:k1].transpose(0, 1).float(),
                dout[q0:q1].transpose(0, 1).float(),
                softmax_scale,
                causal,
            )
            dq[q0:q1].copy_(gq.transpose(0, 1).to(q.dtype))
            dk[k0:k1].copy_(gk.transpose(0, 1).to(k.dtype))
            dv[k0:k1].copy_(gv.transpose(0, 1).to(v.dtype))
        return dq, dk, dv, torch.empty((0,), device=q.device, dtype=torch.float32)

    def fake_cuda_fwd(
        q,
        k,
        v,
        _out,
        _alibi_slopes,
        _dropout_p,
        softmax_scale,
        causal,
        _window_left,
        _window_right,
        _softcap,
        _return_softmax,
        _gen,
    ):
        counts["dense_fwd"] += 1
        qf, kf, vf = q.float(), k.float(), v.float()
        scores = torch.matmul(qf, kf.transpose(-1, -2)) * float(softmax_scale)
        if causal:
            sq, sk = scores.shape[-2], scores.shape[-1]
            mask = torch.ones((sq, sk), device=q.device, dtype=torch.bool).tril(
                diagonal=sk - sq
            )
            scores = scores.masked_fill(~mask, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        out = torch.matmul(torch.softmax(scores, dim=-1), vf).to(q.dtype)
        softmax = torch.empty((0,), device=q.device, dtype=q.dtype)
        rng_state = torch.zeros((2,), device="cpu", dtype=torch.int64)
        return out.contiguous(), lse.contiguous(), softmax, rng_state

    def fake_cuda_bwd(
        dout,
        q,
        k,
        v,
        _out,
        _lse,
        dq,
        dk,
        dv,
        _alibi_slopes,
        _dropout_p,
        softmax_scale,
        causal,
        _window_left,
        _window_right,
        _softcap,
        _deterministic,
        _gen,
        _rng_state,
    ):
        counts["dense_bwd"] += 1
        gq, gk, gv = _attn_bwd_math(
            q.float(), k.float(), v.float(), dout.float(), softmax_scale, causal
        )
        dq.copy_(gq.to(q.dtype))
        dk.copy_(gk.to(k.dtype))
        dv.copy_(gv.to(v.dtype))
        return dq, dk, dv, torch.empty((0,), device=q.device, dtype=torch.float32)

    monkeypatch.setattr(v100_compat, "_cuda_varlen_fwd", fake_cuda_varlen_fwd)
    monkeypatch.setattr(v100_compat, "_cuda_varlen_bwd", fake_cuda_varlen_bwd)
    monkeypatch.setattr(v100_compat, "_cuda_dense_fwd", fake_cuda_fwd)
    monkeypatch.setattr(v100_compat, "_cuda_dense_bwd", fake_cuda_bwd)
    return v100_compat, counts


def test_v100_flash_func_uses_zero_copy_varlen_for_contiguous_blhd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v100_compat, counts = _install_fake_v100_flash_kernels(monkeypatch)
    torch.manual_seed(0)
    q = torch.randn(2, 12, 4, 16, dtype=torch.float16, requires_grad=True)
    k = torch.randn(2, 9, 2, 16, dtype=torch.float16, requires_grad=True)
    v = torch.randn(2, 9, 2, 16, dtype=torch.float16, requires_grad=True)

    out = v100_compat.flash_attn_func(q, k, v)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    assert counts == {
        "varlen_fwd": 1,
        "varlen_bwd": 1,
        "dense_fwd": 0,
        "dense_bwd": 0,
    }

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    out_ref = F.scaled_dot_product_attention(
        q_ref.transpose(1, 2),
        k_ref.transpose(1, 2).repeat_interleave(2, dim=1),
        v_ref.transpose(1, 2).repeat_interleave(2, dim=1),
        scale=1.0 / math.sqrt(16),
    ).transpose(1, 2)
    out_ref.backward(grad_out.float())

    torch.testing.assert_close(out.float(), out_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(q.grad.float(), q_ref.grad, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(k.grad.float(), k_ref.grad, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(v.grad.float(), v_ref.grad, rtol=3e-3, atol=3e-3)


def test_v100_flash_func_falls_back_to_dense_for_lse_or_unaligned_head_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v100_compat, counts = _install_fake_v100_flash_kernels(monkeypatch)
    q_lse = torch.randn(1, 8, 2, 16, dtype=torch.float16)
    out_lse, lse, _ = v100_compat.flash_attn_func(q_lse, q_lse, q_lse, return_attn_probs=True)
    assert out_lse.shape == q_lse.shape
    assert lse.shape == (1, 2, 8)
    assert counts["dense_fwd"] == 1
    assert counts["varlen_fwd"] == 0

    q_pad = torch.randn(1, 8, 2, 20, dtype=torch.float16, requires_grad=True)
    out_pad = v100_compat.flash_attn_func(q_pad, q_pad, q_pad)
    out_pad.sum().backward()
    assert counts["dense_fwd"] == 2
    assert counts["dense_bwd"] == 1
    assert counts["varlen_fwd"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Dynamo fullgraph test")
def test_v100_flash_zero_copy_varlen_compiles_fullgraph_dynamic_seq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v100_compat, _counts = _install_fake_v100_flash_kernels(monkeypatch)

    def block(x: torch.Tensor) -> torch.Tensor:
        b, s, width = x.shape
        qkv = x.reshape(b, s, 2, width // 2)
        return v100_compat.flash_attn_func(qkv, qkv, qkv).reshape(b, s, width)

    compiled = torch.compile(block, backend="inductor", fullgraph=True)
    for seq_len in (32, 48):
        x = torch.randn(2, seq_len, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 1, min=16, max=64)
        y = compiled(x)
        y.float().square().mean().backward()
        assert y.shape == x.shape
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_loha_chunked_autograd_matches_unsplit_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(7)
    monkeypatch.setattr(loha_autograd, "_LOHA_WEIGHT_CHUNK_ELEMS", 32)
    x = torch.randn(2, 6, 12, dtype=torch.bfloat16, requires_grad=True)
    w1_a = torch.randn(19, 3, requires_grad=True)
    w1_b = torch.randn(3, 12, requires_grad=True)
    w2_a = torch.randn(19, 3, requires_grad=True)
    w2_b = torch.randn(3, 12, requires_grad=True)
    scale = 0.375

    y_ref = (F.linear(x.float(), make_hada_weight(w1_a, w1_b, w2_a, w2_b)) * scale).to(
        x.dtype
    )
    grad_out = torch.randn_like(y_ref)
    y_ref.backward(grad_out)
    grads_ref = [t.grad.detach().clone() for t in (x, w1_a, w1_b, w2_a, w2_b)]
    for t in (x, w1_a, w1_b, w2_a, w2_b):
        t.grad = None

    y_chunk = loha_linear(x, w1_a, w1_b, w2_a, w2_b, scale=scale)
    y_chunk.backward(grad_out)

    torch.testing.assert_close(y_chunk, y_ref, rtol=1e-5, atol=1e-5)
    for got, exp in zip((x.grad, w1_a.grad, w1_b.grad, w2_a.grad, w2_b.grad), grads_ref):
        torch.testing.assert_close(got, exp, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for VRAM check")
def test_int8_frozen_linear_does_not_retain_dequantized_weight_in_autograd() -> None:
    device = torch.device("cuda")
    linear = nn.Linear(1024, 4096, bias=False, device=device, dtype=torch.float16)
    linear.weight.requires_grad_(False)
    wrapped = Int8FrozenLinear.from_linear(linear).to(device)
    del linear
    torch.cuda.empty_cache()

    x = torch.randn(2, 16, 1024, device=device, dtype=torch.float16, requires_grad=True)
    with torch.no_grad():
        _ = wrapped(x.detach())
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    y = wrapped(x)
    retained_bytes = torch.cuda.memory_allocated(device) - before - y.numel() * y.element_size()
    # A retained [4096, 1024] fp16 weight would add 8 MiB; custom Function holds 0 B.
    assert retained_bytes < 512 * 1024, f"retained {retained_bytes} B in autograd graph"
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    base = nn.Linear(32, 64, bias=False, device=device, dtype=torch.float16)
    base.weight.requires_grad_(False)

    class _FakeLoRA(nn.Module):
        def __init__(self, mod: nn.Linear) -> None:
            super().__init__()
            self.lora_name = "lora_unet_blocks_0_mlp_layer1"
            self.original_name = "blocks.0.mlp.layer1"
            self.org_module_ref = [mod]
            self.org_forward = mod.forward

    class _FakeNet(nn.Module):
        def __init__(self, mod: nn.Linear) -> None:
            super().__init__()
            self.unet_loras = nn.ModuleList([_FakeLoRA(mod)])

    net = _FakeNet(base)
    patch_lora_frozen_base_forwards_with_int8(net, scope="mlp")
    x_small = torch.randn(4, 32, device=device, dtype=torch.float16, requires_grad=True)
    out = net.unet_loras[0].org_forward(x_small)
    out.sum().backward()
    assert x_small.grad is not None and torch.isfinite(x_small.grad).all()


@pytest.mark.skipif(
    lokr_autograd.triton is None or not torch.cuda.is_available(),
    reason="CUDA + Triton required",
)
def test_lokr_triton_sm70_ieee_compiles_and_matches_vectorized_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    factor, in_dim, out_dim = 4, 32, 48
    x = torch.randn(2, 16, factor * in_dim, device=device, dtype=torch.float16, requires_grad=True)
    w1 = (torch.randn(factor, factor, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
    w2 = (torch.randn(out_dim, in_dim, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
    base = torch.randn(2, 16, factor * out_dim, device=device, dtype=torch.float16)
    gate = torch.tensor([[0.5]], device=device, dtype=torch.float32)
    grad_out = torch.randn_like(base)

    y_eager = lokr_autograd.lokr_add_grouped_delta_(
        base.clone(),
        x,
        w1,
        w2,
        gate,
        factor,
        in_dim,
        out_dim,
        2,
        65536,
        backend="eager",
        backward_backend="eager",
    )
    y_eager.backward(grad_out)
    grads_eager = (x.grad.clone(), w1.grad.clone(), w2.grad.clone())
    x.grad = w1.grad = w2.grad = None

    monkeypatch.setattr(lokr_autograd, "_MIN_TRITON_CC", (7, 0))
    y_triton = lokr_autograd.lokr_add_grouped_delta_(
        base.clone(),
        x,
        w1,
        w2,
        gate,
        factor,
        in_dim,
        out_dim,
        2,
        65536,
        backend="triton",
        backward_backend="triton_grad_w1_w2_grad_x",
    )
    y_triton.backward(grad_out)
    torch.testing.assert_close(y_triton, y_eager, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(x.grad, grads_eager[0], rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(w1.grad, grads_eager[1], rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(w2.grad, grads_eager[2], rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for NF4 dir1")
def test_bf16_autocast_linear4bit_dir1_emits_bf16_under_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("bitsandbytes")
    from library.quantization import nf4

    linear = nn.Linear(64, 64, bias=True, dtype=torch.bfloat16)
    q_linear = nf4.BF16AutocastLinear4bit(
        64, 64, bias=True, compute_dtype=torch.bfloat16, quant_type="nf4"
    )
    q_linear.load_state_dict(linear.state_dict(), strict=True)
    q_linear = q_linear.to("cuda")

    monkeypatch.setattr(nf4, "_DIR1_ENABLED", True)
    x_fp32 = torch.randn(4, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = q_linear(x_fp32)
    assert out.dtype == torch.bfloat16
    out.float().sum().backward()
    assert x_fp32.grad is not None
    assert x_fp32.grad.dtype == torch.float32
    assert torch.isfinite(x_fp32.grad).all()
