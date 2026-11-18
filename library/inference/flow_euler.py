"""Small shared Euler kernel for flow-matching preview and inference loops."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor


def combine_cfg(positive: Tensor, negative: Tensor, guidance_scale: float) -> Tensor:
    """Apply the standard positive-anchored classifier-free guidance rule."""
    if positive.shape != negative.shape:
        raise ValueError(
            f"CFG predictions must have matching shapes, got {positive.shape} and {negative.shape}"
        )
    return positive + float(guidance_scale) * (positive - negative)


def euler_step(latent: Tensor, velocity: Tensor, sigma_current: Tensor, sigma_next: Tensor) -> Tensor:
    """Advance one flow-matching Euler step without changing tensor dtype."""
    if latent.shape != velocity.shape:
        raise ValueError(
            f"latent and velocity must have matching shapes, got {latent.shape} and {velocity.shape}"
        )
    return latent + (sigma_next - sigma_current).to(device=latent.device) * velocity


def _as_sigma_sequence(sigmas: Tensor | Sequence[float], latent: Tensor) -> Tensor:
    values = torch.as_tensor(sigmas, device=latent.device)
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("Euler sampling requires a one-dimensional sigma sequence with at least two values")
    if not torch.is_floating_point(values):
        values = values.float()
    return values


def run_flow_euler_loop(
    latent: Tensor,
    sigmas: Tensor | Sequence[float],
    predict: Callable[[Tensor, Tensor], Tensor],
    *,
    guidance_scale: float = 0.0,
    predict_negative: Callable[[Tensor, Tensor], Tensor] | None = None,
    step_callback: Callable[[int, Tensor, Tensor, Tensor, Tensor], None] | None = None,
    step_start_callback: Callable[[int, Tensor, Tensor], None] | None = None,
    branch_start_callback: Callable[[int, int, Tensor, Tensor], None] | None = None,
    combine_predictions: Callable[[Tensor, Tensor, float], Tensor] = combine_cfg,
    use_guidance: bool | None = None,
    step_fn: Callable[[int, Tensor, Tensor, Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    """Run a scheduler-agnostic flow Euler loop.

    Scheduler/shift logic stays with each model family. ``predict`` and
    ``predict_negative`` receive the current latent and a one-element sigma
    tensor. A non-positive guidance scale intentionally never calls the
    negative predictor, which keeps unconditional preview work opt-in.
    """
    if not torch.isfinite(torch.as_tensor(float(guidance_scale))).item():
        raise ValueError("guidance_scale must be finite")
    sigma_values = _as_sigma_sequence(sigmas, latent)
    use_cfg = float(guidance_scale) > 0.0 if use_guidance is None else bool(use_guidance)
    if use_cfg and predict_negative is None:
        raise ValueError("predict_negative is required when guidance_scale is positive")

    current = latent
    for index, (sigma_current, sigma_next) in enumerate(
        zip(sigma_values[:-1], sigma_values[1:])
    ):
        sigma = sigma_current.reshape(1)
        if step_start_callback is not None:
            step_start_callback(index, current, sigma)
        if branch_start_callback is not None:
            branch_start_callback(index, 0, current, sigma)
        positive = predict(current, sigma)
        velocity = positive
        if use_cfg:
            if branch_start_callback is not None:
                branch_start_callback(index, 1, current, sigma)
            negative = predict_negative(current, sigma)
            velocity = combine_predictions(positive, negative, guidance_scale)
        if step_fn is None:
            current = euler_step(current, velocity, sigma_current, sigma_next)
        else:
            current = step_fn(index, current, velocity, sigma_current, sigma_next)
        if step_callback is not None:
            step_callback(index, current, sigma_current, sigma_next, velocity)
    return current


__all__ = ["combine_cfg", "euler_step", "run_flow_euler_loop"]
