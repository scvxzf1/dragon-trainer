"""Family-neutral rectified-flow step primitives.

Model families keep their own sigma schedules, conditioning layouts, and DiT
forward contracts. This module owns only the shared 4D numerical boundary.
"""

from __future__ import annotations

import torch


def mix_latents(
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Build ``x_sigma = (1-sigma) * x0 + sigma * noise`` for 4D latents."""

    if latents.ndim != 4 or noise.shape != latents.shape:
        raise ValueError(
            "flow step expects matching 4D latents/noise, got "
            f"{tuple(latents.shape)} and {tuple(noise.shape)}"
        )
    sigma_view = sigmas.to(dtype=latents.dtype).reshape(-1, 1, 1, 1)
    if sigma_view.shape[0] != latents.shape[0]:
        raise ValueError(
            "flow sigma batch does not match latents: "
            f"{sigma_view.shape[0]} != {latents.shape[0]}"
        )
    return (1.0 - sigma_view) * latents + sigma_view * noise


def velocity_target(noise: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    """Return the rectified-flow velocity target ``noise - latents``."""

    if noise.shape != latents.shape:
        raise ValueError(
            "flow target expects matching noise/latents, got "
            f"{tuple(noise.shape)} and {tuple(latents.shape)}"
        )
    return noise - latents


def unit_weighting(sigmas: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    """Return the broadcastable unit loss weighting used by non-Anima families."""

    return torch.ones(
        (latents.shape[0], 1, 1, 1), device=sigmas.device, dtype=torch.float32
    )
