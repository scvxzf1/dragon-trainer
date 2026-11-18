from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from networks import attention_dispatch


def test_flash_varlen_shared_primitive_supports_key_padding_and_preserves_padding(monkeypatch):
    calls = SimpleNamespace()

    def fake_flash(q, k, v, cu_q, cu_k, max_q, max_k, *, softmax_scale=None):
        calls.values = (q.shape, k.shape, cu_q.tolist(), cu_k.tolist(), max_q, max_k, softmax_scale)
        return q

    monkeypatch.setattr(attention_dispatch, "flash_attn_available_for_tensor", lambda _tensor: True)
    monkeypatch.setattr(attention_dispatch, "flash_attn_varlen_func", fake_flash)
    q = torch.randn(2, 5, 4, 3, dtype=torch.float16, requires_grad=True)
    k = torch.randn(2, 5, 2, 3, dtype=torch.float16, requires_grad=True)
    valid = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])

    output = attention_dispatch.flash_varlen_with_bool_mask(q, k, k, mask=valid, softmax_scale=0.25)
    assert calls.values == (torch.Size([10, 4, 3]), torch.Size([7, 2, 3]), [0, 5, 10], [0, 3, 7], 5, 4, 0.25)
    assert output.shape == q.shape
    assert torch.isfinite(output).all()


def test_flash_varlen_shared_primitive_accepts_krea_outer_mask(monkeypatch):
    calls = SimpleNamespace()

    def fake_flash(q, k, v, cu_q, cu_k, max_q, max_k, *, softmax_scale=None):
        calls.values = (cu_q.tolist(), cu_k.tolist(), max_q, max_k)
        repeat = q.shape[1] // k.shape[1]
        return q + k.repeat_interleave(repeat, dim=1) + v.repeat_interleave(repeat, dim=1)

    monkeypatch.setattr(attention_dispatch, "flash_attn_available_for_tensor", lambda _tensor: True)
    monkeypatch.setattr(attention_dispatch, "flash_attn_varlen_func", fake_flash)
    q = torch.randn(2, 5, 4, 3, dtype=torch.float16)
    k = torch.randn(2, 5, 2, 3, dtype=torch.float16)
    valid = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    mask = valid[:, None, :, None] & valid[:, None, None, :]

    output = attention_dispatch.flash_varlen_with_bool_mask(q, k, k, mask=mask, max_seqlen=5)
    assert calls.values == ([0, 3, 7], [0, 3, 7], 5, 5)
    assert torch.count_nonzero(output[~valid]) == 0


@pytest.mark.parametrize("mask", [torch.ones(2, 5, 5, dtype=torch.bool), torch.ones(2, 1, 5, 5, dtype=torch.float32)])
def test_flash_varlen_shared_primitive_rejects_ambiguous_or_non_boolean_masks(monkeypatch, mask):
    monkeypatch.setattr(attention_dispatch, "flash_attn_available_for_tensor", lambda _tensor: True)
    with pytest.raises(ValueError, match="boolean|shaped"):
        attention_dispatch.flash_varlen_with_bool_mask(
            torch.randn(2, 5, 4, 3, dtype=torch.float16),
            torch.randn(2, 5, 2, 3, dtype=torch.float16),
            torch.randn(2, 5, 2, 3, dtype=torch.float16),
            mask=mask,
        )


def test_dispatch_flash_fails_closed_when_provider_is_missing(monkeypatch):
    monkeypatch.setattr(attention_dispatch, "flash_attn_func", None)
    monkeypatch.setattr(attention_dispatch, "flash_attn_varlen_func", None)
    q = torch.randn(1, 3, 2, 4, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="provider is unavailable"):
        attention_dispatch.dispatch_attention(
            [q, q, q],
            attn_params=attention_dispatch.AttentionParams.create_attention_params("flash"),
        )
