"""Experimental frozen-projection packing for Qwen Image 2.1 training."""

from __future__ import annotations

import torch
from torch import nn

from library.env import resolve_model_family


MODES = frozenset({"off", "mlp", "qkv", "all"})


def install_training_block_projections(args, model: nn.Module, network) -> tuple[int, int]:
    """Preflight the entire requested conversion before changing any block."""
    mode = str(getattr(args, "qwen_fused_projections", "off") or "off").strip().lower()
    saved_blocks = int(getattr(args, "qwen_saved_projection_blocks", 0) or 0)
    budget_mib = int(getattr(args, "qwen_projection_budget_mib", 0) or 0)
    if saved_blocks < 0 or budget_mib < 0:
        raise ValueError("Qwen saved projection count and budget must be nonnegative")
    if bool(saved_blocks) != bool(budget_mib):
        raise ValueError("Qwen saved projections require both a block count and a positive MiB budget")
    if saved_blocks and mode != "all":
        raise ValueError("qwen_saved_projection_blocks requires qwen_fused_projections=all")
    if mode not in MODES:
        raise ValueError(f"qwen_fused_projections must be one of {sorted(MODES)}")
    if mode == "off":
        return 0, 0
    if resolve_model_family(args) != "qwen_image_2_1":
        raise ValueError("qwen_fused_projections is only supported for qwen_image_2_1")
    if str(getattr(args, "mixed_precision", "bf16") or "bf16").lower() != "bf16":
        raise ValueError("qwen_fused_projections requires mixed_precision=bf16")
    if str(getattr(args, "base_compute", "bf16") or "bf16").lower() != "bf16":
        raise ValueError("qwen_fused_projections requires base_compute=bf16")
    if str(getattr(args, "block_swap_transfer_dtype", "bf16") or "bf16").lower() != "bf16":
        raise ValueError("qwen_fused_projections does not support low-bit block-swap transfer")
    if str(getattr(args, "selective_checkpoint", "off") or "off").lower() != "off":
        raise ValueError("qwen_fused_projections requires full checkpointing")
    if not getattr(args, "gradient_checkpointing", False):
        raise ValueError("qwen_fused_projections requires gradient_checkpointing")
    if getattr(args, "cpu_offload_checkpointing", False) or getattr(args, "unsloth_offload_checkpointing", False):
        raise ValueError("qwen_fused_projections does not support checkpoint offload")
    if str(getattr(args, "network_module", "networks.lora_anima") or "networks.lora_anima") != "networks.lora_anima":
        raise ValueError("qwen_fused_projections requires plain LoRA")

    from library.models.qwen_image_2_1.fused_mlp import (
        _validate_source, install_qwen_image_2_1_fused_mlps,
    )
    from library.models.qwen_image_2_1.packed_qkv import (
        _check_install_timing, _validate_attention, install_qwen_image_2_1_packed_qkv,
    )
    from networks.lora_modules.base import BaseLoRAModule

    blocks = _check_install_timing(model)
    if saved_blocks:
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

        if saved_blocks > len(blocks):
            raise ValueError("qwen_saved_projection_blocks must be within 1..block_count")
        if not getattr(model, "gradient_checkpointing", False):
            raise ValueError("segmented Qwen training requires gradient checkpointing")
        if any(type(block) is not QwenImage21TransformerBlock for block in blocks[-saved_blocks:]):
            raise TypeError("segmented training requires native Qwen Image 2.1 blocks")
        if getattr(model, "_qwen_image_2_1_block_swap_adapter", None) is None and not callable(
            getattr(model, "_gradient_checkpointing_func", None)
        ):
            raise TypeError("segmented Qwen training requires a checkpoint function")
    offloader = getattr(model, "offloader", None)
    if mode in {"mlp", "all"} and offloader is not None and not hasattr(offloader, "_cpu_weight_masters"):
        raise RuntimeError("install fused MLPs before inspectable block-swap CPU masters")
    loras = getattr(network, "unet_loras", None)
    if not isinstance(loras, (list, tuple)) or not loras or any(
        not isinstance(lora, BaseLoRAModule) for lora in loras
    ):
        raise TypeError("qwen_fused_projections requires applied LoRA-family modules")
    if getattr(network, "text_encoder_loras", ()):
        raise ValueError("qwen_fused_projections does not support text-encoder LoRA")
    by_projection = {id(lora.org_module_ref[0]): lora for lora in loras}
    allowed_ids = set()
    for block in blocks:
        attn = getattr(block, "attn", None)
        if attn is None:
            raise TypeError("Qwen Image 2.1 block has no attention module")
        for module in attn.modules():
            if type(module) is nn.Linear:
                allowed_ids.add(id(module))
        if mode in {"mlp", "all"}:
            gate, up, out = _validate_source(getattr(block, "img_mlp", None), freeze=True)
            if any(linear.weight.dtype != torch.bfloat16 for linear in (gate, up, out)):
                raise ValueError("qwen_fused_projections requires BF16 MLP weights")
        if mode in {"qkv", "all"}:
            projections = tuple(getattr(attn, name, None) for name in ("to_q", "to_k", "to_v"))
            _validate_attention(attn, tuple(by_projection.get(id(p)) for p in projections), freeze=True)
            if any(p.weight.dtype != torch.bfloat16 for p in projections):
                raise ValueError("qwen_fused_projections requires BF16 QKV weights")
    if len(by_projection) != len(loras) or any(id(lora.org_module_ref[0]) not in allowed_ids for lora in loras):
        raise ValueError("qwen_fused_projections requires attention-only LoRA")

    mlp_count = install_qwen_image_2_1_fused_mlps(model, freeze=True) if mode in {"mlp", "all"} else 0
    qkv_count = install_qwen_image_2_1_packed_qkv(model, loras, freeze=True) if mode in {"qkv", "all"} else 0
    if saved_blocks:
        from library.models.qwen_image_2_1.segmented_training import install_segmented_training

        install_segmented_training(model, count=saved_blocks, budget_mib=budget_mib)
    return mlp_count, qkv_count
