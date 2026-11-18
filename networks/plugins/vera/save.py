"""VeRA save/export hooks."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ...attn_fuse import defuse_qkv_state_dict, match_fused_spec


def defuse_vera_qkv(state_dict: Dict[str, torch.Tensor]) -> None:
    """Split runtime-fused VeRA qkv/kv keys into per-component slices."""

    fused_groups: list[tuple] = []
    for key in list(state_dict.keys()):
        if not key.endswith(".vera_lambda_b"):
            continue
        prefix = key.removesuffix(".vera_lambda_b")
        spec = match_fused_spec(prefix)
        if spec is not None:
            fused_groups.append((prefix, spec))

    for prefix, spec in fused_groups:
        lambda_b = state_dict.pop(f"{prefix}.vera_lambda_b")
        lambda_d = state_dict.pop(f"{prefix}.vera_lambda_d")
        alpha = state_dict.pop(f"{prefix}.alpha", None)
        vera_A = state_dict.pop(f"{prefix}.projection_bank.vera_A", None)
        vera_B = state_dict.pop(f"{prefix}.projection_bank.vera_B", None)

        n = len(spec.component_letters)
        lambda_b_chunks = lambda_b.chunk(n, dim=0)
        vera_B_chunks = vera_B.chunk(n, dim=0) if vera_B is not None else None
        base_prefix = prefix.removesuffix(spec.fused_frag)
        for idx, (letter, lambda_b_chunk) in enumerate(
            zip(spec.component_letters, lambda_b_chunks)
        ):
            new_prefix = base_prefix + spec.component_frag(letter)
            state_dict[f"{new_prefix}.vera_lambda_b"] = lambda_b_chunk
            state_dict[f"{new_prefix}.vera_lambda_d"] = lambda_d.clone()
            if alpha is not None:
                state_dict[f"{new_prefix}.alpha"] = alpha.clone()
            if vera_A is not None:
                state_dict[f"{new_prefix}.projection_bank.vera_A"] = vera_A.clone()
            if vera_B_chunks is not None:
                state_dict[f"{new_prefix}.projection_bank.vera_B"] = vera_B_chunks[idx]


def save_vera_weights(
    state_dict: dict[str, torch.Tensor],
    file: str,
    dtype: Optional[torch.dtype],
    metadata: Optional[dict[str, str]],
) -> bool:
    """Write a VeRA checkpoint through the standard safetensors path."""

    defuse_qkv_state_dict(state_dict, variant="vera")
    # Same runtime→ComfyUI adaln relayout the plain-LoRA writer does. This
    # handler short-circuits lora_save.save_network_weights, so without it a
    # VeRA trained with train_adaln would ship runtime-named adaln keys that
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
