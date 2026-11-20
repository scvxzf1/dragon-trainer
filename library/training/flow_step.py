"""Family-neutral rectified-flow step primitives.

Model families keep their own sigma schedules, conditioning layouts, and DiT
forward contracts. This module owns only the shared 4D numerical boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass
class FlowSamplingContext:
    trainer: Any
    train_context: Any
    latents: torch.Tensor
    batch: Any
    noise: torch.Tensor
    noisy_input: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    is_train: bool = True
    family_state: Any = None


def loss_weighting(
    sigmas: torch.Tensor,
    latents: torch.Tensor,
    args: Any,
) -> torch.Tensor:
    weights = compute_flow_loss_weighting(
        weighting_scheme=getattr(args, "weighting_scheme", "none"),
        sigmas=sigmas,
        min_snr_gamma=getattr(args, "min_snr_gamma", None),
        p2_gamma=getattr(args, "p2_gamma", 1.0),
        p2_k=getattr(args, "p2_k", 1.0),
    )
    return weights.reshape(latents.shape[0], *([1] * (latents.ndim - 1)))


def compute_flow_loss_weighting(
    weighting_scheme: str | None,
    sigmas: torch.Tensor,
    *,
    min_snr_gamma: float | None = None,
    p2_gamma: float = 1.0,
    p2_k: float = 1.0,
) -> torch.Tensor:
    """Compute the shared rectified-flow sample weighting from sigma."""
    sigmas_f = sigmas.float()
    if weighting_scheme == "sigma_sqrt":
        return sigmas_f.pow(-2.0)
    if weighting_scheme == "cosmap":
        denominator = 1.0 - 2.0 * sigmas_f + 2.0 * sigmas_f.square()
        return 2.0 / (math.pi * denominator)
    if weighting_scheme == "min_snr":
        gamma = 5.0 if min_snr_gamma is None else float(min_snr_gamma)
        sigma_safe = sigmas_f.clamp(1e-6, 1.0 - 1e-6)
        snr = ((1.0 - sigma_safe) / sigma_safe).square()
        return torch.minimum(snr, torch.full_like(snr, gamma)) / snr.clamp_min(
            1e-12
        )
    if weighting_scheme == "p2":
        sigma_safe = sigmas_f.clamp(1e-6, 1.0 - 1e-6)
        snr = ((1.0 - sigma_safe) / sigma_safe).square()
        return (float(p2_k) + snr).pow(-float(p2_gamma))
    return torch.ones_like(sigmas_f)


def run_family_flow_step(
    trainer,
    ctx,
    plugin,
    latents: torch.Tensor,
    batch,
    text_encoder_conds,
    *,
    is_train: bool = True,
):
    """Run the shared rectified-flow step around three family-specific hooks."""
    if latents.ndim == 5 and latents.shape[2] == 1:
        latents = latents.squeeze(2)
    if latents.ndim != 4:
        raise ValueError(
            f"family flow engine expects 4D [B,C,H,W] latents, got {tuple(latents.shape)}"
        )
    if any(size <= 0 for size in latents.shape):
        raise ValueError(f"family flow requires nonempty latent axes, got {tuple(latents.shape)}")
    family = getattr(plugin, "family", None)
    if family is not None:
        from library.models.family_registry import get_model_family_spec

        channels = get_model_family_spec(family).latent_space.latent_channels
        if latents.shape[1] != channels:
            raise ValueError(f"{family} flow expects {channels} latent channels, got {latents.shape[1]}")
    noise = torch.randn_like(latents)
    sampling_context = FlowSamplingContext(
        trainer=trainer,
        train_context=ctx,
        latents=latents,
        batch=batch,
        noise=noise,
        is_train=is_train,
    )
    input_shape = tuple(latents.shape)
    input_device = latents.device
    sigmas = plugin.sample_sigmas(
        latents.shape[0],
        latents.shape[-2],
        latents.shape[-1],
        ctx.args,
        latents.device,
        context=sampling_context,
    )
    latents = sampling_context.latents
    noise = sampling_context.noise
    _validate_flow_inputs(
        latents, noise, sampling_context.noisy_input, input_shape, input_device
    )
    if not isinstance(sigmas, torch.Tensor):
        raise TypeError("family sigma hook must return a torch.Tensor")
    if sigmas.ndim != 1 or sigmas.shape[0] != latents.shape[0]:
        raise ValueError(
            "family sigma hook must return one flat sigma per sample; "
            f"got {tuple(sigmas.shape)} for batch {latents.shape[0]}"
        )
    sigmas = sigmas.to(device=latents.device, dtype=torch.float32)
    _check_flow_values(
        ((sigmas >= 0) & (sigmas <= 1)).all(),
        "family sigma hook must return finite sigmas in [0,1]",
    )
    timesteps = sampling_context.timesteps
    if timesteps is None:
        timesteps = sigmas
    elif not isinstance(timesteps, torch.Tensor):
        raise TypeError("family model timestep hook must return a torch.Tensor")
    elif timesteps.shape != sigmas.shape or timesteps.device != sigmas.device:
        raise ValueError("family model timestep must have one value per sample on the latent device")
    else:
        _check_flow_values(torch.isfinite(timesteps).all(), "family model timestep must be finite")

    noisy = sampling_context.noisy_input
    if noisy is None:
        noisy = mix_latents(latents, noise, sigmas)
    elif noisy.shape != latents.shape:
        raise ValueError("family sampler noisy-input override must match latent shape")
    elif noisy.device != latents.device:
        raise ValueError("family sampler noisy-input override must match the latent device")
    prepare_inputs = getattr(plugin, "prepare_flow_inputs", None)
    if prepare_inputs is not None:
        sampling_context.noisy_input = noisy
        prepare_inputs(sampling_context)
        latents, noise, noisy = sampling_context.latents, sampling_context.noise, sampling_context.noisy_input
        _validate_flow_inputs(latents, noise, noisy, input_shape, input_device)
        if sampling_context.timesteps is not None:
            timesteps = sampling_context.timesteps
            if not isinstance(timesteps, torch.Tensor):
                raise TypeError("family model timestep hook must return a torch.Tensor")
            if timesteps.shape != sigmas.shape or timesteps.device != latents.device:
                raise ValueError(
                    "family model timestep hook must preserve one value per sample "
                    "on the latent device"
                )
            _check_flow_values(
                torch.isfinite(timesteps).all(),
                "family model timestep must be finite",
            )
    route_early = getattr(plugin, "route_before_conditions", False)
    if route_early:
        _route_flow_step(sampling_context, noisy, timesteps)
    if getattr(ctx.args, "gradient_checkpointing", False):
        noisy.requires_grad_(True)
    conditions = plugin.prepare_conditions(
        batch,
        text_encoder_conds,
        context=sampling_context,
        is_train=is_train,
    )

    if not route_early:
        _route_flow_step(sampling_context, noisy, timesteps, conditions)
    with torch.set_grad_enabled(is_train), ctx.accelerator.autocast():
        prediction = plugin.forward_velocity_4d(
            ctx.unet, noisy, conditions, sigmas, context=sampling_context
        )
    if not isinstance(prediction, torch.Tensor) or prediction.shape != latents.shape:
        raise ValueError(
            "family velocity hook must return a tensor matching 4D latents; "
            f"got {getattr(prediction, 'shape', None)} vs {tuple(latents.shape)}"
        )

    target = velocity_target(noise, latents)
    observe_step = getattr(plugin, "observe_flow_step", None)
    if observe_step is not None:
        observe_step(sampling_context, prediction, target)
    weighting_fn = getattr(plugin, "compute_loss_weighting", None)
    weighting = (
        weighting_fn(sigmas, latents, ctx.args)
        if callable(weighting_fn)
        else unit_weighting(sigmas, latents)
    )
    adjust_weighting = getattr(plugin, "adjust_flow_weighting", None)
    if adjust_weighting is not None:
        weighting = adjust_weighting(sampling_context, weighting)
    expected_weight_shape = (latents.shape[0], 1, 1, 1)
    if (
        not isinstance(weighting, torch.Tensor)
        or weighting.shape != expected_weight_shape
        or weighting.device != latents.device
    ):
        raise ValueError(
            "family loss-weighting hook must return a tensor shaped "
            f"{expected_weight_shape} on {latents.device}; "
            f"got {getattr(weighting, 'shape', None)} on "
            f"{getattr(weighting, 'device', None)}"
        )
    return prediction, target, timesteps, weighting


def _check_flow_values(valid: torch.Tensor, message: str) -> None:
    if valid.is_cuda:
        torch._assert_async(valid, message)
    elif not bool(valid):
        raise ValueError(message)


def _validate_flow_inputs(latents, noise, noisy, expected_shape, expected_device) -> None:
    for name, value in (("latents", latents), ("noise", noise)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"family flow {name} hook must preserve a tensor")
        if tuple(value.shape) != expected_shape or value.device != expected_device:
            raise ValueError(
                f"family flow {name} hook must preserve shape {expected_shape} "
                f"and device {expected_device}"
            )
    if noisy is not None:
        if not isinstance(noisy, torch.Tensor):
            raise TypeError("family sampler noisy-input hook must preserve a tensor")
        if tuple(noisy.shape) != expected_shape or noisy.device != expected_device:
            raise ValueError(
                f"family sampler noisy-input hook must preserve shape {expected_shape} "
                f"and device {expected_device}"
            )


def _route_flow_step(context, noisy, timesteps, conditions=None) -> None:
    from library.training.router_conditioning import apply_router_conditioning

    trainer, ctx = context.trainer, context.train_context
    trainer._hydra_warmup_step = apply_router_conditioning(
        network=ctx.network,
        noisy_model_input=noisy,
        timesteps=timesteps,
        is_train=context.is_train,
        warmup_step=int(getattr(trainer, "_hydra_warmup_step", 0)),
        max_train_steps=int(getattr(ctx.args, "max_train_steps", 0) or 0),
        crossattn_emb=getattr(conditions, "crossattn_emb", None),
        gradient_accumulation_steps=int(getattr(ctx.args, "gradient_accumulation_steps", 1) or 1),
    )


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
