"""Spatial mask reduction helpers for training losses.

The legacy path multiplies a per-pixel loss by the mask and then averages over
the full latent. ``foreground_mean`` keeps the foreground gradient scale
independent of mask coverage while preserving the legacy default.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def mask_image_for_loss(
    loss: torch.Tensor, batch: dict[str, Any]
) -> torch.Tensor | None:
    """Return a broadcastable ``[B, 1, H, W]`` (or 5D ``[B, 1, T, H, W]``) mask, if present."""
    target_hw = tuple(int(d) for d in loss.shape[-2:])
    if "conditioning_images" in batch:
        cond = batch["conditioning_images"].to(
            device=loss.device, dtype=loss.dtype, non_blocking=True
        )[:, 0].unsqueeze(1)
        cond = cond * 0.5 + 0.5
        if tuple(int(d) for d in cond.shape[-2:]) != target_hw:
            cond = F.interpolate(cond, size=target_hw, mode="area")
        return cond.unsqueeze(2) if loss.ndim == 5 else cond
    alpha_masks = batch.get("alpha_masks")
    if alpha_masks is None:
        return None
    mask = alpha_masks.to(
        device=loss.device, dtype=loss.dtype, non_blocking=True
    ).unsqueeze(1)
    if tuple(int(d) for d in mask.shape[-2:]) != target_hw:
        mask = F.interpolate(mask, size=target_hw, mode="area")
    return mask.unsqueeze(2) if loss.ndim == 5 else mask


def apply_mask(loss: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    """Apply a spatial mask without changing the tensor shape."""
    mask = mask_image_for_loss(loss, batch)
    return loss if mask is None else loss * mask


def reduce_masked_loss(
    loss: torch.Tensor,
    batch: dict[str, Any],
    *,
    normalize: str = "none",
    foreground_weight: float = 1.0,
) -> torch.Tensor:
    """Reduce a per-pixel loss to ``[B]`` with optional foreground normalization.

    ``none`` is the historical full-latent mean. ``foreground_mean`` averages
    only over the soft mask and falls back to the unmasked mean for an empty
    mask, preventing NaNs and making malformed masks observable by callers.
    """
    reduce_dims = tuple(range(1, loss.ndim))
    mask = mask_image_for_loss(loss, batch)
    if mask is None:
        return loss.mean(dim=reduce_dims)
    expanded_mask = mask.expand_as(loss)
    masked = loss * expanded_mask
    if normalize == "none":
        return masked.mean(dim=reduce_dims)
    if normalize != "foreground_mean":
        raise ValueError(f"unsupported mask loss normalization: {normalize!r}")

    full_mean = loss.mean(dim=reduce_dims)
    area = expanded_mask.sum(dim=reduce_dims)
    valid = area > 1e-6
    denom = area.clamp_min(1e-6)
    foreground = masked.sum(dim=reduce_dims) / denom
    result = torch.where(valid, foreground * float(foreground_weight), full_mean)
    return result
