"""Phase 3 (P3-1): Unified BlockSwapController, ExplicitBlockSwapMixin, and BlockSwapConfig."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT
from library.runtime.block_swap_auto import resolve_offloader_and_blocks
from library.runtime.block_swap_controller import (
    BlockSwapConfig,
    BlockSwapController,
    ExplicitBlockSwapMixin,
)


class _DummyBlock(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x)


class _WrapToyModel(nn.Module):
    def __init__(self, depth: int = 4, dim: int = 16) -> None:
        super().__init__()
        self.stem = nn.Linear(dim, dim)
        self.custom_blocks = nn.ModuleList(_DummyBlock(dim) for _ in range(depth))
        self.head = nn.Linear(dim, dim)
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = None

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = lambda fn, *a, **kw: torch.utils.checkpoint.checkpoint(
            fn, *a, use_reentrant=False, **kw
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.custom_blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                h = self._gradient_checkpointing_func(blk, h)
            else:
                h = blk(h)
        return self.head(h)


class _ExplicitToyModel(ExplicitBlockSwapMixin, nn.Module):
    _block_swap_container_attr = "stages"

    def __init__(self, depth: int = 4, dim: int = 16) -> None:
        super().__init__()
        self.stem = nn.Linear(dim, dim)
        self.stages = nn.ModuleList(_DummyBlock(dim) for _ in range(depth))
        self.blocks_to_swap = 0
        self.offloader = None
        self._paused_blocks_to_swap = None


def test_block_swap_config_from_args_and_probe_fields() -> None:
    args = SimpleNamespace(
        block_swap_transfer_dtype="int8",
        block_swap_restore_mode="slab",
        block_swap_int8_restore_mode="triton",
        block_swap_int8_scope="mlp",
        block_swap_int8_staging_ring=3,
        block_swap_int8_dual_stream=True,
        block_swap_int8_restore_chunk_rows=512,
    )
    cfg = BlockSwapConfig.from_args(args, profile_jsonl="/tmp/swap.jsonl")
    assert cfg.to_offloader_kwargs() == {
        "profile_jsonl": "/tmp/swap.jsonl",
        "transfer_dtype": "int8",
        "restore_mode": "slab",
        "int8_restore_mode": "triton",
        "int8_scope": "mlp",
        "int8_staging_ring": 3,
        "int8_dual_stream": True,
        "int8_restore_chunk_rows": 512,
    }
    assert cfg.probe_fields(12) == {
        "blocks_to_swap": 12,
        "block_swap_profile_jsonl": "/tmp/swap.jsonl",
        "block_swap_transfer_dtype": "int8",
        "block_swap_restore_mode": "slab",
    }


def test_generic_controller_registers_auto_tuner_and_idempotent_pause() -> None:
    model = _WrapToyModel(depth=4)
    model.enable_gradient_checkpointing()
    ctrl = BlockSwapController(
        model,
        2,
        torch.device("cpu"),
        container_attr="custom_blocks",
        min_blocks=0,
        family_label="WrapToy",
    )
    resolved = resolve_offloader_and_blocks(model)
    assert resolved is not None
    assert resolved[0] is ctrl.offloader
    assert resolved[1] is model.custom_blocks

    # Re-enabling checkpointing preserves swap dispatch
    model.enable_gradient_checkpointing()
    x = torch.randn(2, 4, 16, requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad is not None

    assert model.pause_block_swap() is True
    assert model.blocks_to_swap == 0
    # Second pause while already paused must be a clean False no-op
    assert model.pause_block_swap() is False
    assert model.resume_block_swap() is True
    assert model.blocks_to_swap == 2
    assert model.resume_block_swap() is False


def test_explicit_mixin_handles_custom_container_and_zero_swap() -> None:
    model = _ExplicitToyModel(depth=4)
    model._init_explicit_block_swap(2, torch.device("cpu"))
    assert model.blocks_to_swap == 2
    model.move_to_device_except_swap_blocks(torch.device("cpu"))
    assert len(model.stages) == 4
    assert model.pause_block_swap() is True
    assert model.blocks_to_swap == 0
    assert model.pause_block_swap() is False
    assert model.resume_block_swap() is True
    assert model.blocks_to_swap == 2


@pytest.mark.parametrize(
    "device_str",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_krea2_dit_gains_pause_and_resume_block_swap(device_str: str) -> None:
    cfg = SingleMMDiTConfig(
        features=64,
        tdim=32,
        txtdim=32,
        heads=2,
        multiplier=2,
        layers=4,
        patch=2,
        channels=4,
        kvheads=1,
        txtlayers=1,
        txtheads=2,
        txtkvheads=1,
    )
    device = torch.device(device_str)
    dit = SingleStreamDiT(cfg).to(dtype=torch.float32).requires_grad_(False)
    dit.enable_block_swap(2, device)
    dit.move_to_device_except_swap_blocks(device)
    dit.prepare_block_swap_before_forward()

    if device.type == "cuda":
        # Exactly 2 blocks should have parked 2D Linear weights on CPU while swap is active
        assert sum(b.mlp.up.weight.device.type == "cpu" for b in dit.blocks) == 2

    assert dit.pause_block_swap() is True
    assert dit.blocks_to_swap == 0
    if device.type == "cuda":
        # While paused, all blocks are resident on CUDA for zero-DMA eval
        assert all(b.mlp.up.weight.device.type == "cuda" for b in dit.blocks)

    assert dit.resume_block_swap() is True
    assert dit.blocks_to_swap == 2
    if device.type == "cuda":
        assert sum(b.mlp.up.weight.device.type == "cpu" for b in dit.blocks) == 2
    dit.flush_block_swap_profile(blocking=True)
