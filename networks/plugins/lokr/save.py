"""LoKR save/export hooks."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ...attn_fuse import defuse_qkv_state_dict, match_fused_spec


def defuse_lokr_qkv(state_dict: Dict[str, torch.Tensor]) -> None:
    """Split runtime-fused LoKr keys into LyCORIS-compatible q/k/v slices."""

    fused_groups: list[tuple] = []
    for key in list(state_dict.keys()):
        if not key.endswith(".lokr_w1"):
            continue
        prefix = key.removesuffix(".lokr_w1")
        spec = match_fused_spec(prefix)
        if spec is not None:
            fused_groups.append((prefix, spec))

    for prefix, spec in fused_groups:
        n_comp = len(spec.component_letters)
        w1 = state_dict.pop(f"{prefix}.lokr_w1")
        alpha = state_dict.pop(f"{prefix}.alpha", None)
        base_prefix = prefix.removesuffix(spec.fused_frag)
        split_outer = w1.dim() >= 1 and w1.size(0) % n_comp == 0
        if f"{prefix}.lokr_w2" in state_dict:
            w2 = state_dict.pop(f"{prefix}.lokr_w2")
            if split_outer:
                w1_chunks = w1.chunk(n_comp, dim=0)
                for letter, w1_chunk in zip(spec.component_letters, w1_chunks):
                    new_prefix = base_prefix + spec.component_frag(letter)
                    state_dict[f"{new_prefix}.lokr_w1"] = w1_chunk.contiguous()
                    state_dict[f"{new_prefix}.lokr_w2"] = w2.clone()
                    if alpha is not None:
                        state_dict[f"{new_prefix}.alpha"] = alpha.clone()
            else:
                w2_chunks = w2.chunk(n_comp, dim=0)
                for letter, w2_chunk in zip(spec.component_letters, w2_chunks):
                    new_prefix = base_prefix + spec.component_frag(letter)
                    state_dict[f"{new_prefix}.lokr_w1"] = w1.clone()
                    state_dict[f"{new_prefix}.lokr_w2"] = w2_chunk.contiguous()
                    if alpha is not None:
                        state_dict[f"{new_prefix}.alpha"] = alpha.clone()
            continue

        w2a_key = f"{prefix}.lokr_w2_a"
        w2b_key = f"{prefix}.lokr_w2_b"
        if w2a_key not in state_dict or w2b_key not in state_dict:
            state_dict[f"{prefix}.lokr_w1"] = w1
            if alpha is not None:
                state_dict[f"{prefix}.alpha"] = alpha
            continue
        w2a = state_dict.pop(w2a_key)
        w2b = state_dict.pop(w2b_key)
        if split_outer:
            w1_chunks = w1.chunk(n_comp, dim=0)
            for letter, w1_chunk in zip(spec.component_letters, w1_chunks):
                new_prefix = base_prefix + spec.component_frag(letter)
                state_dict[f"{new_prefix}.lokr_w1"] = w1_chunk.contiguous()
                state_dict[f"{new_prefix}.lokr_w2_a"] = w2a.clone()
                state_dict[f"{new_prefix}.lokr_w2_b"] = w2b.clone()
                if alpha is not None:
                    state_dict[f"{new_prefix}.alpha"] = alpha.clone()
        else:
            w2a_chunks = w2a.chunk(n_comp, dim=0)
            for letter, w2a_chunk in zip(spec.component_letters, w2a_chunks):
                new_prefix = base_prefix + spec.component_frag(letter)
                state_dict[f"{new_prefix}.lokr_w1"] = w1.clone()
                state_dict[f"{new_prefix}.lokr_w2_a"] = w2a_chunk.contiguous()
                state_dict[f"{new_prefix}.lokr_w2_b"] = w2b.clone()
                if alpha is not None:
                    state_dict[f"{new_prefix}.alpha"] = alpha.clone()


def save_lokr_weights(
    state_dict: dict[str, torch.Tensor],
    file: str,
    dtype: Optional[torch.dtype],
    metadata: Optional[dict[str, str]],
) -> bool:
    """Write a LoKR checkpoint using the standard safetensors path."""

    defuse_qkv_state_dict(state_dict, variant="lokr")
    # Same runtime→ComfyUI adaln relayout the plain-LoRA writer does. This
    # handler short-circuits lora_save.save_network_weights, so without it a
    # LoKR trained with train_adaln would ship runtime-named adaln keys that
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
