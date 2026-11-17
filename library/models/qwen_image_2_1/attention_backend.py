"""Qwen Image 2.1 attention backend selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch


QWEN_IMAGE_21_ATTENTION_MODES = frozenset({"torch", "flash"})


@dataclass(frozen=True)
class _DiffusersAttentionApi:
    backend_names: type[Enum]
    processor_type: type[object]
    prepare_qkv: Callable
    dispatch_attention: Callable


def normalize_qwen_image_2_1_attention_mode(value: object) -> str:
    mode = str(value or "torch").strip().lower().replace("-", "_")
    if mode in {"sdpa", "native"}:
        return "torch"
    if mode not in QWEN_IMAGE_21_ATTENTION_MODES:
        raise ValueError(
            "Qwen Image 2.1 attn_mode supports only 'torch' (native SDPA) and "
            f"'flash' (FlashAttention varlen); got {value!r}"
        )
    return mode


def _load_diffusers_attention_api() -> _DiffusersAttentionApi:
    try:
        from diffusers.models import AttentionBackendName
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21AttnProcessor,
            _qwenimage21_prepare_qkv,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Qwen Image 2.1 FlashAttention requires the Diffusers Qwen 2.1 "
            "attention processor and dispatcher"
        ) from exc
    return _DiffusersAttentionApi(
        backend_names=AttentionBackendName,
        processor_type=QwenImage21AttnProcessor,
        prepare_qkv=_qwenimage21_prepare_qkv,
        dispatch_attention=dispatch_attention_fn,
    )


class QwenImage21FlashAttnProcessor:
    """Use Flash varlen for image queries while preserving causal text attention."""

    _parallel_config = None

    def __init__(self, api: _DiffusersAttentionApi, *, prepare_qkv: Callable | None = None) -> None:
        self._api = api
        self._prepare_qkv = prepare_qkv or api.prepare_qkv
        self._native_backend = api.backend_names.NATIVE
        self._flash_backend = api.backend_names.FLASH_VARLEN
        self._attention_backend = self._flash_backend

    def _attend(self, query, key, value, attn_mask, backend):
        if backend == self._flash_backend:
            return _flash_varlen_attention(query, key, value, mask=attn_mask)
        return self._api.dispatch_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            backend=self._native_backend,
            parallel_config=self._parallel_config,
        )

    def _attend_prefix_segment(self, query, key, value, segment, key_valid):
        start, end, is_text = segment
        segment_mask = None
        if is_text:
            segment_len = end - start
            segment_mask = torch.cat(
                [
                    torch.ones(segment_len, start, dtype=torch.bool, device=query.device),
                    torch.tril(
                        torch.ones(segment_len, segment_len, dtype=torch.bool, device=query.device)
                    ),
                ],
                dim=1,
            )[None, None]
        if key_valid is not None:
            valid_mask = key_valid[:, None, None, :end]
            segment_mask = valid_mask if segment_mask is None else (segment_mask & valid_mask)

        backend = self._native_backend if is_text else self._flash_backend
        return self._attend(query[:, start:end], key[:, :end], value[:, :end], segment_mask, backend)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask=None,
        rotary_emb=None,
        layer_cache=None,
        kv_cache_mode=None,
        cache_write_slice=None,
        segments=None,
        key_valid=None,
    ) -> torch.Tensor:
        query, key, value, seq_len_q = self._prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
        )
        return self.attend_prepared(
            attn, query, key, value, seq_len_q, attention_mask, segments, key_valid
        )

    def attend_prepared(
        self, attn, query, key, value, seq_len_q, attention_mask, segments, key_valid
    ) -> torch.Tensor:

        if segments is None:
            backend = (
                self._flash_backend
                if _is_key_padding_mask(attention_mask, batch=query.shape[0], key_length=key.shape[1])
                else self._native_backend
            )
            hidden_states = self._attend(query, key, value, attention_mask, backend)
        else:
            prefix_len = segments[-1][1] if segments else 0
            outputs = [
                self._attend_prefix_segment(query, key, value, segment, key_valid)
                for segment in segments
            ]
            target_mask = None if key_valid is None else key_valid[:, None, None, :]
            outputs.append(
                self._attend(query[:, prefix_len:], key, value, target_mask, self._flash_backend)
            )
            hidden_states = torch.cat(outputs, dim=1)

        hidden_states = hidden_states[:, :seq_len_q]
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


def _is_key_padding_mask(mask: object, *, batch: int, key_length: int) -> bool:
    if mask is None:
        return True
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        return False
    if mask.ndim == 1:
        return batch == 1 and mask.shape[0] == key_length
    return (
        mask.ndim == 4
        and mask.shape[0] == batch
        and mask.shape[1] == 1
        and mask.shape[2] == 1
        and mask.shape[3] == key_length
    )


def _normalize_key_valid_mask(mask, *, batch: int, key_length: int, device) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch, key_length), dtype=torch.bool, device=device)
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("Qwen Image 2.1 FlashAttention requires a boolean key-valid mask")
    if mask.device != device:
        raise ValueError(
            "Qwen Image 2.1 FlashAttention requires the key-valid mask on the same device "
            f"as Q/K/V; got mask on {mask.device} and Q/K/V on {device}"
        )
    if mask.ndim == 1 and batch == 1 and mask.shape[0] == key_length:
        return mask[None, :]
    if mask.ndim == 2 and tuple(mask.shape) == (batch, key_length):
        return mask
    if (
        mask.ndim == 4
        and mask.shape[0] == batch
        and mask.shape[1] == 1
        and mask.shape[2] == 1
        and mask.shape[3] == key_length
    ):
        return mask[:, 0, 0]
    raise ValueError(
        "Qwen Image 2.1 FlashAttention supports only a key-padding mask shaped "
        f"(B,K) or (B,1,1,K); got {tuple(mask.shape)}"
    )


@torch.compiler.disable(recursive=True)
def _flash_varlen_attention(query, key, value, *, mask):
    """Pack the actual valid K/V positions outside compiled checkpoint blocks."""
    from networks import attention_dispatch

    if not attention_dispatch.flash_attn_available_for_tensor(query):
        raise RuntimeError(
            f"Qwen Image 2.1 FlashAttention does not support Q/K/V dtype {query.dtype}"
        )
    if query.device.type != "cuda" or key.device != query.device or value.device != query.device:
        raise RuntimeError("Qwen Image 2.1 FlashAttention requires Q/K/V on the same CUDA device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Qwen Image 2.1 FlashAttention requires matching Q/K/V dtypes")
    if query.shape[0] != key.shape[0] or key.shape != value.shape:
        raise ValueError("Qwen Image 2.1 FlashAttention requires matching K/V and Q/K batches")
    if query.shape[2:] != key.shape[2:]:
        raise ValueError("Qwen Image 2.1 FlashAttention requires matching head count and dimension")

    batch, query_length, heads, head_dim = query.shape
    key_length = key.shape[1]
    valid = _normalize_key_valid_mask(
        mask, batch=batch, key_length=key_length, device=query.device
    )
    lengths = valid.sum(dim=1, dtype=torch.int32)
    if bool((lengths == 0).any()):
        raise ValueError("Qwen Image 2.1 FlashAttention does not accept empty key sequences")

    cu_q = torch.arange(batch + 1, device=query.device, dtype=torch.int32) * query_length
    cu_k = torch.zeros(batch + 1, device=query.device, dtype=torch.int32)
    cu_k[1:] = lengths.cumsum(dim=0)
    q_packed = query.reshape(batch * query_length, heads, head_dim).contiguous()
    k_packed = key[valid].contiguous()
    v_packed = value[valid].contiguous()
    max_key_length = int(lengths.max().item())
    output = attention_dispatch.flash_attn_varlen_func(
        q_packed,
        k_packed,
        v_packed,
        cu_q,
        cu_k,
        query_length,
        max_key_length,
        dropout_p=0.0,
    )
    return output.reshape(batch, query_length, heads, head_dim)


def prepare_qwen_image_2_1_attention(
    model: object,
    mode: object,
    *,
    dtype: torch.dtype | None,
) -> str:
    """Configure packed FlashAttention without changing Diffusers global state."""

    normalized = normalize_qwen_image_2_1_attention_mode(mode)
    if normalized == "torch":
        return normalized
    if dtype != torch.bfloat16:
        raise RuntimeError(
            f"Qwen Image 2.1 attn_mode='flash' is validated only for bf16 compute; got {dtype}"
        )

    try:
        from networks import attention_dispatch

        if not attention_dispatch.flash_attn_available_for_dtype(dtype):
            raise RuntimeError("installed FlashAttention provider does not support BF16")
    except (ImportError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "Qwen Image 2.1 attn_mode='flash' requires a compatible "
            "FlashAttention 2 varlen provider with BF16 support"
        ) from exc

    api = _load_diffusers_attention_api()

    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise TypeError("Qwen Image 2.1 transformer does not expose modules()")
    configured = 0
    for module in modules():
        processor = getattr(module, "processor", None)
        if not isinstance(processor, api.processor_type):
            continue
        setter = getattr(module, "set_processor", None)
        if not callable(setter):
            raise TypeError("Qwen Image 2.1 attention module cannot set its processor")
        setter(QwenImage21FlashAttnProcessor(api))
        configured += 1
    if configured == 0:
        raise RuntimeError(
            "Qwen Image 2.1 transformer has no supported attention processors; "
            "cannot configure FlashAttention"
        )
    return normalized


__all__ = [
    "QWEN_IMAGE_21_ATTENTION_MODES",
    "QwenImage21FlashAttnProcessor",
    "normalize_qwen_image_2_1_attention_mode",
    "prepare_qwen_image_2_1_attention",
]
