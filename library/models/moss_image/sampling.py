"""MOSS nearest-training-grid Euler sampling with fp32 noise and CFG."""

from __future__ import annotations

import math

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config

from library.inference.flow_euler import run_flow_euler_loop


class MOSSImageScheduler(FlowMatchEulerDiscreteScheduler):
    """Official MOSS V2 grid; inference shift is applied before grid snapping."""

    @register_to_config
    def __init__(self, num_train_timesteps=1000, shift=5.0, training_shift=1.0):
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
        self.register_to_config(shift=shift, training_shift=training_shift)

    def set_timesteps(self, num_inference_steps=50, device=None):
        if (type(num_inference_steps) is not int or num_inference_steps < 1
                or self.config.num_train_timesteps < 1
                or not math.isfinite(self.config.shift) or self.config.shift <= 0
                or not math.isfinite(self.config.training_shift) or self.config.training_shift <= 0):
            raise ValueError("MOSS steps and shifts must be positive and finite")
        total = self.config.num_train_timesteps

        def grid(count, shift):
            lower = shift / (total + shift - 1)
            values = torch.linspace(1, lower, count, device="cpu", dtype=torch.float32)
            return shift * values / (1 + (shift - 1) * values)

        sigmas = grid(num_inference_steps, self.config.shift)
        training = grid(total, self.config.training_shift).flip(0)
        upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where(
            (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(),
            lower, upper,
        )
        sigmas = torch.unique_consecutive(training[nearest])
        self._shift = 1.0  # Already shifted and snapped; never shift twice.
        super().set_timesteps(device=device, sigmas=sigmas.tolist())


def _validate_skip_layers(dit, layers):
    layers = () if layers is None else tuple(layers)
    depth = len(dit.layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f"uncond_skip_layers must contain integer block indices in [0, {depth})")
    return layers


def _prepare_condition(encoded, *, device, dtype):
    if encoded is None or len(encoded) != 2:
        raise ValueError("MOSS prompt encoding must contain hiddens and mask")
    hidden, mask = (torch.as_tensor(value) for value in encoded)
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if hidden.ndim != 3 or mask.ndim != 2 or hidden.shape[:2] != mask.shape:
        raise ValueError("MOSS prompt shapes must be [B,S,D] and [B,S]")
    if not torch.all((mask == 0) | (mask == 1)) or not mask.bool().any(dim=1).all():
        raise ValueError("MOSS prompt masks must be binary with at least one token per sample")
    return hidden.to(device=device, dtype=dtype), mask.to(device=device, dtype=torch.bool)


@torch.no_grad()
def generate_moss_image(
    dit, cond_embeds, uncond_embeds=None, *, height=1024, width=1024,
    sample_steps=50, guidance_scale=3.0, flow_shift=5.0, seed=0,
    device, dtype=torch.bfloat16, uncond_skip_layers=(22,), latents=None,
    step_start_callback=None,
    branch_start_callback=None,
):
    """Return normalized [B,128,H/16,W/16] MOSS latents without loading TE/VAE.

    CFG is ``negative + scale * (positive - negative)``; scale=1 skips the
    negative forward. Tiny models must explicitly pass uncond_skip_layers=[].
    The caller owns the batch-wide block-swap and adapter inference lifecycle.
    """
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("MOSS height and width must be positive multiples of 16")
    if not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError("MOSS guidance_scale must be finite and nonnegative")
    skip_layers = _validate_skip_layers(dit, uncond_skip_layers)
    positive, positive_mask = _prepare_condition(cond_embeds, device=device, dtype=dtype)
    use_cfg = guidance_scale != 1.0
    if use_cfg:
        negative, negative_mask = _prepare_condition(uncond_embeds, device=device, dtype=dtype)
        if negative.shape[0] != positive.shape[0]:
            raise ValueError("MOSS positive and negative prompt batches must match")
    shape = (positive.shape[0], 128, height // 16, width // 16)
    if latents is None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(device)
    elif tuple(latents.shape) != shape:
        raise ValueError(f"MOSS latents must have shape {shape}")
    else:
        latents = latents.to(device=device, dtype=torch.float32)
    scheduler = MOSSImageScheduler(shift=float(flow_shift), training_shift=1.0)
    scheduler.set_timesteps(sample_steps, device=device)
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)

    def positive_predict(current_latent, sigma):
        timestep = sigma * 1000.0
        return dit(
            current_latent, timestep, positive, positive_mask, return_dict=True,
        ).sample.float()

    def negative_predict(current_latent, sigma):
        timestep = sigma * 1000.0
        return dit(
            current_latent, timestep, negative, negative_mask,
            return_dict=True, skip_layers=skip_layers,
        ).sample.float()

    def scheduler_step(index, current_latent, velocity, _sigma_current, _sigma_next):
        return scheduler.step(
            velocity, scheduler.timesteps[index], current_latent, return_dict=False
        )[0].float()

    return run_flow_euler_loop(
        latents,
        sigmas,
        positive_predict,
        guidance_scale=guidance_scale,
        predict_negative=negative_predict if use_cfg else None,
        combine_predictions=lambda positive, negative, scale: negative + scale * (positive - negative),
        use_guidance=use_cfg,
        step_fn=scheduler_step,
        step_start_callback=step_start_callback,
        branch_start_callback=(
            (lambda index, branch, current, sigma: branch_start_callback(
                index, branch, current, sigma,
                positive if branch == 0 else negative,
            ))
            if branch_start_callback is not None else None
        ),
    )
