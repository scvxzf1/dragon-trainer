"""Block-swap adapter for the official Diffusers Z-Image transformer."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from library.runtime.block_swap_controller import BlockSwapController
from library.runtime.offloading import ModelOffloader

logger = logging.getLogger(__name__)


class ZImageBlockSwapAdapter(BlockSwapController):
    """Thin Z-Image specialization over the shared ``BlockSwapController``."""

    _default_container_attr = "layers"
    _default_min_blocks = 1
    _default_family_label = "Z-Image"
    _default_adapter_attr = "_z_image_block_swap_adapter"

    def __init__(
        self,
        model: nn.Module,
        blocks_to_swap: int,
        device: torch.device,
        **offloader_kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            blocks_to_swap,
            device,
            offloader_cls=ModelOffloader,
            **offloader_kwargs,
        )


def enable_z_image_block_swap(
    model: nn.Module,
    blocks_to_swap: int,
    device: torch.device,
    **offloader_kwargs: Any,
) -> ZImageBlockSwapAdapter:
    """Attach the shared training offloader to Z-Image's 30 main layers."""
    if getattr(model, "_z_image_block_swap_adapter", None) is not None:
        raise RuntimeError("Z-Image block swap is already enabled")
    adapter = ZImageBlockSwapAdapter(
        model,
        blocks_to_swap,
        device,
        **offloader_kwargs,
    )
    logger.info(
        "Z-Image block swap enabled: swapping %s/%s main layers on %s",
        blocks_to_swap,
        len(adapter.layers),
        device,
    )
    return adapter
