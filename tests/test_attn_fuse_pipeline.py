from __future__ import annotations

import torch

from networks.attn_fuse import (
    defuse_qkv_state_dict,
    refuse_qkv_state_dict,
)


def _fused_plain_state() -> dict[str, torch.Tensor]:
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    return {
        f"{prefix}.lora_down.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        f"{prefix}.lora_up.weight": torch.arange(24, dtype=torch.float32).reshape(12, 2),
        f"{prefix}.alpha": torch.tensor(2.0),
    }


def test_shared_defuse_entry_point_uses_standard_writer() -> None:
    state = _fused_plain_state()

    defuse_qkv_state_dict(state)

    assert "lora_unet_blocks_0_self_attn_qkv_proj.lora_down.weight" not in state
    for letter in ("q", "k", "v"):
        assert f"lora_unet_blocks_0_self_attn_{letter}_proj.lora_down.weight" in state
        assert state[f"lora_unet_blocks_0_self_attn_{letter}_proj.lora_up.weight"].shape == (4, 2)


def test_shared_refuse_entry_point_rebuilds_fused_plain_state() -> None:
    split = _fused_plain_state()
    defuse_qkv_state_dict(split)
    original_up = split["lora_unet_blocks_0_self_attn_q_proj.lora_up.weight"].clone()

    refuse_qkv_state_dict(split)

    fused = "lora_unet_blocks_0_self_attn_qkv_proj"
    assert f"{fused}.lora_down.weight" in split
    assert split[f"{fused}.lora_up.weight"].shape == (12, 2)
    assert torch.equal(split[f"{fused}.lora_up.weight"][:4], original_up)
    assert f"{fused}.alpha" in split
