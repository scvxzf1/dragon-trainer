"""LoHa save/export hooks."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ...attn_fuse import defuse_qkv_state_dict, match_fused_spec


def defuse_loha_qkv(state_dict: Dict[str, torch.Tensor]) -> None:
    """Split runtime-fused LoHa qkv/kv keys into PEFT/LyCORIS-style slices."""

    fused_groups: list[tuple] = []
    for key in list(state_dict.keys()):
        if not key.endswith(".hada_w1_a"):
            continue
        prefix = key.removesuffix(".hada_w1_a")
        spec = match_fused_spec(prefix)
        if spec is not None:
            fused_groups.append((prefix, spec))

    for prefix, spec in fused_groups:
        w1_a = state_dict.pop(f"{prefix}.hada_w1_a")
        w1_b = state_dict.pop(f"{prefix}.hada_w1_b")
        w2_a = state_dict.pop(f"{prefix}.hada_w2_a")
        w2_b = state_dict.pop(f"{prefix}.hada_w2_b")
        alpha = state_dict.pop(f"{prefix}.alpha", None)

        w1_a_chunks = w1_a.chunk(len(spec.component_letters), dim=0)
        w2_a_chunks = w2_a.chunk(len(spec.component_letters), dim=0)
        base_prefix = prefix.removesuffix(spec.fused_frag)
        for letter, w1_a_chunk, w2_a_chunk in zip(
            spec.component_letters,
            w1_a_chunks,
            w2_a_chunks,
        ):
            new_prefix = base_prefix + spec.component_frag(letter)
            state_dict[f"{new_prefix}.hada_w1_a"] = w1_a_chunk
            state_dict[f"{new_prefix}.hada_w1_b"] = w1_b.clone()
            state_dict[f"{new_prefix}.hada_w2_a"] = w2_a_chunk
            state_dict[f"{new_prefix}.hada_w2_b"] = w2_b.clone()
            if alpha is not None:
                state_dict[f"{new_prefix}.alpha"] = alpha.clone()


def save_loha_weights(
    state_dict: dict[str, torch.Tensor],
    file: str,
    dtype: Optional[torch.dtype],
    metadata: Optional[dict[str, str]],
) -> bool:
    """Write a LoHa checkpoint using the standard safetensors path."""

    defuse_qkv_state_dict(state_dict, variant="loha")
    # Same runtime→ComfyUI adaln relayout the plain-LoRA writer does. This
    # handler short-circuits lora_save.save_network_weights, so without it a
    # LoHa trained with train_adaln would ship runtime-named adaln keys that
    # ComfyUI's generic key map silently drops. Presence-gated.
    from networks.lora_save import _relayout_adaln_to_comfy

    metadata = _relayout_adaln_to_comfy(state_dict, metadata)

    if dtype is not None:
        for key in list(state_dict.keys()):
            state_dict[key] = state_dict[key].detach().clone().to("cpu").to(dtype)

    if file.endswith(".safetensors"):
        from safetensors.torch import save_file
        from library.training.hashing import precalculate_safetensors_hashes

        if metadata is None:
            metadata = {}
        model_hash, legacy_hash = precalculate_safetensors_hashes(
            state_dict, metadata
        )
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash
        save_file(state_dict, file, metadata)
    else:
        torch.save(state_dict, file)
    return True
