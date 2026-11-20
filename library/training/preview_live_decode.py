"""Family-neutral live preview decode orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Optional

import torch

from library.runtime.device import clean_memory_on_device

logger = logging.getLogger(__name__)


def _module_device(module) -> Optional[torch.device]:
    if module is None:
        return None
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(exc).lower()
    )


def _restore_dit_training_device(dit, device: torch.device) -> None:
    if getattr(dit, "blocks_to_swap", 0) and hasattr(
        dit, "move_to_device_except_swap_blocks"
    ):
        dit.move_to_device_except_swap_blocks(device)
    else:
        dit.to(device)


def decode_live_preview(
    accelerator,
    args,
    vae,
    *,
    dit=None,
    network=None,
    decode_pending_samples: Callable,
) -> None:
    """Park live modules, decode pending latents, and restore training state."""
    if vae is None or not accelerator.is_main_process:
        return
    import os

    latents_dir = os.path.join(args.output_dir, "sample", "latents")
    if not os.path.isdir(latents_dir) or not any(
        name.endswith(".pt") for name in os.listdir(latents_dir)
    ):
        return

    dit_device = _module_device(dit)
    network_device = _module_device(network)
    offloader = getattr(dit, "offloader", None)
    parked = False
    # Offloader placement owns active ConvRot carriers: ordinary dit.to(cpu)
    # cannot retire them. Transfer failures are fatal, outside decode's retry.
    if offloader is not None and dit_device is not None:
        from library.runtime.block_swap_lifecycle import model_swap_blocks

        parked_blocks = model_swap_blocks(dit)
        parked = offloader.park_blocks_on_cpu(parked_blocks)
        if not parked:
            raise RuntimeError(
                "Live decode requires blocks that are not already CPU-parked"
            )
    try:
        if dit is not None and dit_device is not None:
            dit.to("cpu")
        if network is not None and network_device is not None:
            network.to("cpu")
        clean_memory_on_device(accelerator.device)
        decode_pending_samples(accelerator, args, vae)
    except Exception as exc:  # noqa: BLE001 - keep training; final cleanup retries
        from library.training.preview_decode_policy import PreviewDecodeRestoreError

        if isinstance(exc, PreviewDecodeRestoreError):
            raise
        if _is_cuda_oom(exc):
            logger.warning(
                "Live sample decode skipped because CUDA memory is tight; "
                "latent previews will be decoded at stop/end."
            )
        else:
            logger.error(f"Live sample decode failed; will retry at stop/end: {exc}")
    finally:
        try:
            if dit is not None and dit_device is not None:
                _restore_dit_training_device(dit, dit_device)
        finally:
            try:
                if parked:
                    offloader.restore_parked_blocks(parked_blocks)
            finally:
                if network is not None and network_device is not None:
                    network.to(network_device)
        if dit is not None and hasattr(dit, "switch_block_swap_for_training"):
            dit.switch_block_swap_for_training()
        clean_memory_on_device(accelerator.device)


def decode_live_preview_for_family(
    accelerator,
    args,
    vae,
    *,
    dit=None,
    network=None,
) -> None:
    """Resolve the family plugin instead of depending on an Anima implementation."""
    from library.env import resolve_model_family
    from library.models.family_plugins import get_family_plugin

    family = resolve_model_family(args)
    get_family_plugin(family, capability="preview").decode_preview_samples(
        accelerator,
        args,
        vae,
        dit=dit,
        network=network,
    )


def decode_pending_samples_for_family(accelerator, args, vae) -> None:
    """Dispatch pending latent decoding without importing an Anima default."""
    from library.env import resolve_model_family
    from library.models.family_plugins import get_family_plugin

    family = resolve_model_family(args)
    get_family_plugin(family, capability="preview").decode_pending_samples(accelerator, args, vae)
