"""Opt-in frozen QKV packing for Qwen Image 2.1 attention with split LoRA state."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from library.models.qwen_image_2_1.attention_backend import (
    QwenImage21FlashAttnProcessor,
)
from networks.lora_modules.base import BaseLoRAModule


class _RemovedProjection(nn.Module):
    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("split Q/K/V projection was replaced by packed QKV")


def _removed_lora_forward(_x: torch.Tensor) -> torch.Tensor:
    raise RuntimeError("split Q/K/V base was replaced by packed QKV")


class FrozenPackedQKV(nn.Module):
    """One registered base weight; three original, separately registered LoRAs."""

    def __init__(self, weight: torch.Tensor, loras: tuple[BaseLoRAModule, BaseLoRAModule, BaseLoRAModule]):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self._loras = loras  # A tuple does not re-register adapter parameters.

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        base = F.linear(hidden_states, self.weight)
        return tuple(
            part + lora.compute_delta(
                hidden_states,
                part.dtype,
                base_output=part,
                base_weight=weight_part,
            ).to(part.dtype)
            for lora, part, weight_part in zip(
                self._loras, base.chunk(3, dim=-1), self.weight.chunk(3, dim=0)
            )
        )


def prepare_packed_qkv(
    attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
):
    """Keep Diffusers' norm, RoPE and KV-cache ordering after packed projection."""
    return prepare_projected_qkv(
        attn, attn.packed_qkv(hidden_states), rotary_emb,
        layer_cache, kv_cache_mode, cache_write_slice,
    )


def prepare_projected_qkv(
    attn, projected, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
):
    from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen

    query, key, value = projected
    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))
    query = attn.norm_q(query).to(value.dtype)
    key = attn.norm_k(key).to(value.dtype)
    if rotary_emb is not None:
        query = apply_rotary_emb_qwen(query, rotary_emb, use_real=False)
        key = apply_rotary_emb_qwen(key, rotary_emb, use_real=False)
    if layer_cache is not None:
        if kv_cache_mode == "extract" and cache_write_slice is not None:
            layer_cache.store(key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone())
        elif kv_cache_mode == "cached":
            cached_k, cached_v = layer_cache.get()
            key = torch.cat([cached_k, key], dim=1)
            value = torch.cat([cached_v, value], dim=1)
    return query, key, value, query.shape[1]


class PackedQwenImage21AttnProcessor:
    """Native Diffusers attention dispatch with a packed projection source."""

    _parallel_config = None

    def __call__(
        self, attn, hidden_states, attention_mask=None, rotary_emb=None,
        layer_cache=None, kv_cache_mode=None, cache_write_slice=None,
        segments=None, key_valid=None,
    ):
        query, key, value, seq_len_q = prepare_packed_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
        )
        return self.attend_prepared(
            attn, query, key, value, seq_len_q, attention_mask, segments, key_valid
        )

    def attend_prepared(
        self, attn, query, key, value, seq_len_q, attention_mask, segments, key_valid
    ):
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        if segments is None:
            hidden_states = dispatch_attention_fn(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0,
                backend=None, parallel_config=self._parallel_config,
            )
        else:
            prefix_len = segments[-1][1] if segments else 0
            outputs = []
            for start, end, is_text in segments:
                seg_mask = None
                if is_text:
                    seg_len = end - start
                    seg_mask = torch.cat([
                        torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
                        torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device)),
                    ], dim=1)[None, None]
                if key_valid is not None:
                    valid = key_valid[:, None, None, :end]
                    seg_mask = valid if seg_mask is None else seg_mask & valid
                outputs.append(dispatch_attention_fn(
                    query[:, start:end], key[:, :end], value[:, :end], attn_mask=seg_mask,
                    dropout_p=0.0, backend=None, parallel_config=self._parallel_config,
                ))
            outputs.append(dispatch_attention_fn(
                query[:, prefix_len:], key, value,
                attn_mask=None if key_valid is None else key_valid[:, None, None, :],
                dropout_p=0.0, backend=None, parallel_config=self._parallel_config,
            ))
            hidden_states = torch.cat(outputs, dim=1)
        hidden_states = hidden_states[:, :seq_len_q].flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hidden_states))


def _validate_attention(attn: nn.Module, loras: Sequence[BaseLoRAModule], *, freeze: bool):
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21Attention, QwenImage21AttnProcessor,
    )

    if type(attn) is not QwenImage21Attention or "packed_qkv" in attn._modules:
        raise TypeError("expected an unpacked QwenImage21Attention")
    processor = attn.processor
    if type(processor) not in (QwenImage21AttnProcessor, QwenImage21FlashAttnProcessor):
        raise TypeError("packed QKV requires the native or Qwen Flash processor")
    if any(getattr(attn, name) for name in ("_forward_pre_hooks", "_forward_hooks")):
        raise RuntimeError("cannot replace attention with hooks")
    projections = tuple(getattr(attn, name) for name in ("to_q", "to_k", "to_v"))
    if len(loras) != 3 or any(not isinstance(lora, BaseLoRAModule) for lora in loras):
        raise TypeError("packed QKV requires exactly three LoRA-family adapters")
    if any(not getattr(lora, "supports_packed_qkv", False) for lora in loras):
        unsupported = sorted({type(lora).__name__ for lora in loras if not getattr(lora, "supports_packed_qkv", False)})
        raise TypeError(
            "packed QKV does not support adapter variants: " + ", ".join(unsupported)
        )
    dims = {(p.in_features, p.out_features, p.weight.dtype, p.weight.device) for p in projections if type(p) is nn.Linear}
    if len(dims) != 1 or any(type(p) is not nn.Linear or p.bias is not None for p in projections):
        raise TypeError("packed QKV requires matching bias-free Linear projections")
    for projection, lora in zip(projections, loras):
        if projection.weight.is_meta or "forward" not in projection.__dict__:
            raise RuntimeError("materialize weights and apply LoRA before QKV packing")
        if any(getattr(projection, name) for name in ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks")):
            raise RuntimeError("cannot discard projection hooks")
        current_forward = projection.forward
        original_forward = lora.org_forward
        if (
            lora.org_module_ref[0] is not projection
            or getattr(original_forward, "__self__", None) is not projection
            or getattr(current_forward, "__self__", None) is not lora
        ):
            raise RuntimeError("Q/K/V adapter does not own the expected projection")
        if lora._fused or lora.lora_down.in_features != projection.in_features or lora.lora_up.out_features != projection.out_features:
            raise RuntimeError("Q/K/V adapter shape or fusion is incompatible")
        if projection.weight.requires_grad and not freeze:
            raise RuntimeError("Q/K/V weights are trainable; pass freeze=True for frozen-base packing")
    return projections, processor


def _check_install_timing(model: nn.Module) -> nn.ModuleList:
    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError("packed QKV requires transformer_blocks ModuleList")
    if getattr(model, "_qwen_image_2_1_blocks_compiled", False) or any(hasattr(b, "_orig_mod") for b in blocks):
        raise RuntimeError("install packed QKV before block compilation")
    offloader = getattr(model, "offloader", None)
    if offloader is not None and (getattr(offloader, "_cpu_weight_masters", None) is not None or any(
        entry is not None for entry in getattr(offloader, "_block_module_maps", ())
    )):
        raise RuntimeError("install packed QKV before block-swap master capture")
    if getattr(model, "blocks_to_swap", 0) and offloader is None:
        raise RuntimeError("block swap state is not inspectable")
    return blocks


def install_qwen_image_2_1_packed_qkv(
    model: nn.Module, loras: Sequence[BaseLoRAModule], *, freeze: bool = False
) -> int:
    """Preflight all blocks, then replace split base weights without changing LoRA keys."""
    blocks = _check_install_timing(model)
    by_projection = {id(lora.org_module_ref[0]): lora for lora in loras}
    groups = []
    for block in blocks:
        attn = getattr(block, "attn", None)
        projections = tuple(getattr(attn, name, None) for name in ("to_q", "to_k", "to_v"))
        group = tuple(by_projection.get(id(p)) for p in projections)
        source, processor = _validate_attention(attn, group, freeze=freeze)
        groups.append((attn, source, group, processor))
    installed = []
    try:
        for attn, source, group, processor in groups:
            with torch.no_grad():
                weight = torch.cat([p.weight.detach() for p in source], dim=0)
            packed = FrozenPackedQKV(weight, group)
            packed.train(attn.training)
            if type(processor) is QwenImage21FlashAttnProcessor:
                new_processor = QwenImage21FlashAttnProcessor(
                    processor._api, prepare_qkv=prepare_packed_qkv
                )
            else:
                new_processor = PackedQwenImage21AttnProcessor()
            installed.append((attn, source, group, processor, tuple(lora.org_forward for lora in group)))
            # The old Parameters retain their identity for rollback, but no longer
            # retain three extra base-weight storages after this block commits.
            for projection, part in zip(source, packed.weight.detach().chunk(3, dim=0)):
                projection.weight.data = part
            for name, lora in zip(("to_q", "to_k", "to_v"), group):
                setattr(attn, name, _RemovedProjection())
                lora.org_forward = _removed_lora_forward
                lora.org_module_ref[0] = None
            attn.packed_qkv = packed
            attn.set_processor(new_processor)
    except Exception:
        for attn, source, group, processor, forwards in reversed(installed):
            for name, projection, lora, forward in zip(
                ("to_q", "to_k", "to_v"), source, group, forwards
            ):
                setattr(attn, name, projection)
                lora.org_forward = forward
                lora.org_module_ref[0] = projection
            if "packed_qkv" in attn._modules:
                del attn.packed_qkv
            attn.set_processor(processor)
        raise
    return len(groups)


__all__ = [
    "FrozenPackedQKV", "PackedQwenImage21AttnProcessor", "prepare_packed_qkv",
    "install_qwen_image_2_1_packed_qkv",
]
