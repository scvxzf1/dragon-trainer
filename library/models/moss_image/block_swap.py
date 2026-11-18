"""MOSS main-block residency using the shared ``ExplicitBlockSwapMixin``."""

from __future__ import annotations

import torch

from library.runtime.block_swap_controller import ExplicitBlockSwapMixin
from library.runtime.offloading import ModelOffloader


class MOSSBlockSwapMixin(ExplicitBlockSwapMixin):
    _block_swap_container_attr = "layers"

    def enable_block_swap(self, num_blocks: int, device: torch.device, **kwargs):
        return enable_moss_block_swap(self, num_blocks, device, **kwargs)


def enable_moss_block_swap(model, blocks_to_swap: int, device, **kwargs):
    """Enable swapping 1..depth-2 blocks; zero leaves swapping disabled."""
    if model.offloader is not None:
        raise RuntimeError("MOSS block swap is already configured")
    if type(blocks_to_swap) is not int or not 0 <= blocks_to_swap <= len(model.layers) - 2:
        raise ValueError(f"MOSS blocks_to_swap must be 0..{len(model.layers) - 2}")
    if blocks_to_swap == 0:
        return None
    return model._init_explicit_block_swap(
        blocks_to_swap,
        device,
        offloader_cls=ModelOffloader,
        **kwargs,
    )
