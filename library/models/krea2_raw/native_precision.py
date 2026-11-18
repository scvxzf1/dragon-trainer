"""Explicit native AMP boundaries for the nonquantized Krea-2 model."""
from types import MethodType

import torch
import torch.nn.functional as F


def _fp32_linear(self, x):
    # Installed before LoRA: its org_forward retains this base computation;
    # callers still invoke module.forward, including the adapter contribution.
    with torch.autocast(x.device.type, enabled=False):
        return F.linear(x.float(), self.weight.float(),
                        None if self.bias is None else self.bias.float())


def _fp32_linear_installed(linear):
    from networks.lora_modules.lora import LoRAModule

    if getattr(linear, "_native_fp32_linear", False) is not True:
        return False
    forward = linear.forward
    owner = getattr(forward, "__self__", None)
    if isinstance(owner, LoRAModule):
        if (getattr(forward, "__func__", None) is not LoRAModule.forward
                or owner.org_module_ref[0] is not linear):
            return False
        forward = owner.org_forward
    return (getattr(forward, "__self__", None) is linear
            and getattr(forward, "__func__", None) is _fp32_linear)


def enable_native_guards(model):
    from .dit import SingleStreamDiT

    if type(model) is not SingleStreamDiT:
        raise ValueError("Native Krea-2 requires the official SingleStreamDiT structure")
    if any(type(p) is not torch.nn.Parameter or p.dtype != torch.float16
           for p in model.parameters()):
        raise ValueError("Native Krea-2 requires nonquantized FP16 base parameters")
    for p in model.parameters():
        if not torch.isfinite(p).all():
            raise ValueError("Native Krea-2 base contains nonfinite FP16 weights")
    model.requires_grad_(False)
    for block in [*model.blocks, *model.txtfusion.layerwise_blocks,
                  *model.txtfusion.refiner_blocks, model.last]:
        block.fp32_residual = True
    if not getattr(model.last.linear, "_native_fp32_linear", False):
        model.last.linear.forward = MethodType(_fp32_linear, model.last.linear)
        model.last.linear._native_fp32_linear = True


def observe_guards(model):
    from .dit import SingleStreamDiT

    if type(model) is not SingleStreamDiT:
        raise ValueError("precision_contract_mismatch: native Krea-2 structure missing")
    blocks = list(model.blocks)
    text = [*model.txtfusion.layerwise_blocks, *model.txtfusion.refiner_blocks]
    return {"family": "krea2_raw", "blocks": len(blocks),
            "fp32_residual": all(getattr(b, "fp32_residual", False) is True for b in blocks),
            "text_fp32_residual": all(getattr(b, "fp32_residual", False) is True for b in text),
            "final_fp32": getattr(model.last, "fp32_residual", False) is True,
            "final_linear_fp32": _fp32_linear_installed(model.last.linear)}


def block_forward(block, x, vec, freqs, mask):
    compute = getattr(block.attn.wq, "compute_dtype", None) or x.dtype
    x = x.float()
    modulation = tuple(v.float() for v in block.mod(vec.float()))
    prescale, preshift, pregate, postscale, postshift, postgate = modulation
    attn_input = ((1 + prescale) * block.prenorm(x) + preshift).to(compute)
    x = x + pregate * block.attn(attn_input, freqs, mask).float()
    mlp_input = ((1 + postscale) * block.postnorm(x) + postshift).to(compute)
    return x + postgate * block.mlp(mlp_input).float()


def text_forward(block, x, mask):
    compute = getattr(block.attn.wq, "compute_dtype", None) or x.dtype
    x = x.float()
    x = x + block.attn(block.prenorm(x).to(compute), mask=mask).float()
    return x + block.mlp(block.postnorm(x).to(compute)).float()


def final_forward(layer, x, tvec):
    with torch.autocast(x.device.type, enabled=False):
        scale, shift = layer.modulation(tvec.float())
        x = (1 + scale.float()) * layer.norm(x.float()) + shift.float()
        return layer.linear(x)
