"""Training block swap for the Diffusers Qwen Image 2.1 transformer."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from library.runtime.block_swap_controller import BlockSwapController
from library.runtime.offloading import ModelOffloader

logger = logging.getLogger(__name__)


class QwenImage21BlockSwapAdapter(BlockSwapController):
    """Thin Qwen Image 2.1 specialization over ``BlockSwapController``."""

    _default_container_attr = "transformer_blocks"
    _default_min_blocks = 0
    _default_family_label = "Qwen Image 2.1"
    _default_adapter_attr = "_qwen_image_2_1_block_swap_adapter"
    _default_direct_checkpoint_attr = "_qwen_segmented_forward"

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


def enable_qwen_image_2_1_block_swap(
    model: nn.Module,
    blocks_to_swap: int,
    device: torch.device,
    **offloader_kwargs: Any,
) -> QwenImage21BlockSwapAdapter:
    if getattr(model, "_qwen_image_2_1_block_swap_adapter", None) is not None:
        raise RuntimeError("Qwen Image 2.1 block swap is already enabled")
    adapter = QwenImage21BlockSwapAdapter(
        model,
        blocks_to_swap,
        device,
        **offloader_kwargs,
    )
    logger.info(
        "Qwen Image 2.1 block swap enabled: %s/%s blocks on %s",
        blocks_to_swap,
        len(adapter.blocks),
        device,
    )
    return adapter
