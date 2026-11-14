"""Focused fallback/autograd and NF4 backward equivalence for scaled casts."""
import pytest
import torch

from library.quantization import _scaled_cast


def reference(value, shift, dtype):
    return torch.ldexp(value.float(), shift).to(dtype)


def test_cpu_fallback_keeps_gradient():
    value = torch.tensor([0.25, -0.5], requires_grad=True)
    shift = torch.tensor(3, dtype=torch.int32)
    out = _scaled_cast.scaled_cast(value, shift, torch.float16)
    torch.testing.assert_close(out, reference(value, shift, torch.float16), rtol=0, atol=0)
    out.float().sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, 8.0), rtol=0, atol=0)


def test_empty_fallback():
    value = torch.empty(0, dtype=torch.bfloat16)
    out = _scaled_cast.scaled_cast(value, torch.tensor(0, dtype=torch.int32), torch.float16)
    assert out.shape == value.shape and out.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_real_nf4_bf16_backward_matches_reference(monkeypatch):
    from bitsandbytes.functional import quantize_4bit
    from library.quantization.nf4 import _Dir1MatMul4Bit

    torch.manual_seed(123)
    weight = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16) * 0.02
    packed, state = quantize_4bit(weight, quant_type='nf4', compress_statistics=True)
    value = torch.randn(2, 8, 256, device='cuda', dtype=torch.bfloat16)
    grad = torch.randn_like(value) * 1e-8
    candidate = _scaled_cast.scaled_cast
    results = []
    for implementation in (reference, candidate):
        monkeypatch.setattr(_scaled_cast, 'scaled_cast', implementation)
        x = value.detach().clone().requires_grad_(True)
        y = _Dir1MatMul4Bit.apply(x, packed.t(), None, None, state)
        y.backward(grad)
        results.append((y.detach(), x.grad.detach()))
    for expected, actual in zip(*results):
        assert torch.equal(expected.view(torch.uint16), actual.view(torch.uint16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.no_grad()
def test_negative_views_use_torch_fallback(monkeypatch):
    class MustNotLaunch:
        def __getitem__(self, grid):
            raise AssertionError('negative-bit views must not launch Triton')

    monkeypatch.setattr(_scaled_cast, '_scaled_cast_kernel', MustNotLaunch())
    value = torch.ones(4, device='cuda', dtype=torch.bfloat16)
    shift = torch.tensor(-4, device='cuda', dtype=torch.int32)
    for x, power in ((torch._neg_view(value), shift), (value, torch._neg_view(shift))):
        actual = _scaled_cast.scaled_cast(x, power, torch.float16)
        assert torch.equal(actual, reference(x, power, torch.float16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('source,target', [(torch.bfloat16, torch.float16),
                                          (torch.float16, torch.bfloat16)])
@torch.no_grad()
def test_masked_tail_and_storage_offset(source, target):
    value = torch.randn(1030, device='cuda', dtype=source)[1:1026]
    assert value.is_contiguous() and value.storage_offset() == 1
    shift = torch.tensor(3, device='cuda', dtype=torch.int32)
    actual = _scaled_cast.scaled_cast(value, shift, target)
    expected = reference(value, shift, target)
    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_grad_enabled_cuda_uses_differentiable_fallback(monkeypatch):
    class MustNotLaunch:
        def __getitem__(self, grid):
            raise AssertionError('grad-enabled path must not launch Triton')

    monkeypatch.setattr(_scaled_cast, '_scaled_cast_kernel', MustNotLaunch())
    x = torch.ones(4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    shift = torch.tensor(2, device='cuda', dtype=torch.int32)
    _scaled_cast.scaled_cast(x, shift, torch.float16).float().sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 4.0))
