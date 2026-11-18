from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from library.models.qwen_image_2_1.packed_qkv import FrozenPackedQKV
from networks.lora_modules.dora import DoRALoRAModule
from networks.plugins.lokr.module import LoKrModule


def _make_variant(variant: str, projection: nn.Linear, name: str):
    if variant == "lokr":
        return LoKrModule(
            name, projection, lora_dim=2, alpha=2, factor=2, lokr_use_einsum=True
        )
    if variant == "dora":
        return DoRALoRAModule(name, projection, lora_dim=2, alpha=2)
    raise AssertionError(variant)


@pytest.mark.parametrize("variant", ["lokr", "dora"])
def test_packed_variant_matches_split_forward_and_gradients(variant):
    torch.manual_seed(23)
    reference_weights = [nn.Linear(8, 8, bias=False).double() for _ in range(3)]
    packed_weights = [copy.deepcopy(module) for module in reference_weights]
    reference = [_make_variant(variant, module, f"ref_{index}").double() for index, module in enumerate(reference_weights)]
    packed_adapters = [_make_variant(variant, module, f"packed_{index}").double() for index, module in enumerate(packed_weights)]
    for source, target in zip(reference, packed_adapters):
        target.load_state_dict(source.state_dict())
        source.apply_to()
        target.apply_to()
        source.eval()
        target.eval()

    packed = FrozenPackedQKV(
        torch.cat([module.weight.detach() for module in packed_weights], dim=0),
        tuple(packed_adapters),
    ).double()
    hidden_reference = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    hidden_packed = hidden_reference.detach().clone().requires_grad_()

    expected = torch.cat([module(hidden_reference) for module in reference], dim=-1)
    actual = torch.cat(packed(hidden_packed), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-8)

    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(hidden_packed.grad, hidden_reference.grad, rtol=2e-6, atol=2e-8)
    for source, target in zip(reference, packed_adapters):
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            if source_parameter.grad is None or target_parameter.grad is None:
                continue
            torch.testing.assert_close(
                target_parameter.grad, source_parameter.grad, rtol=2e-6, atol=2e-8
            )
