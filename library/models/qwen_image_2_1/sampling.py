"""Official flow-match Euler scheduling for Qwen Image 2.1 previews."""

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from library.models.qwen_image_2_1.family import forward_for_loss
from library.models.qwen_image_2_1.preview_kv import PreviewKVForward, supports_preview_kv
from library.inference.flow_euler import run_flow_euler_loop


@torch.no_grad()
def generate_preview(dit, prompt, positive, negative, reference, *, device, dtype, seed,
                     forward=forward_for_loss, use_kv_cache=False,
                     step_start_callback=None, branch_start_callback=None):
    """Sample using the inference block layout prepared by the caller.

    With block swap enabled, call ``switch_block_swap_for_inference()`` before
    the prompt batch and restore training mode afterwards. The forward-only
    offloader maintains that layout across denoising and CFG forwards.
    The default uses B without DiT KV caching. ``use_kv_cache=True`` lets
    official causal-condition eval models reuse a separate cache per CFG branch;
    custom forwards and incompatible models keep their uncached behavior.
    """
    if not isinstance(use_kv_cache, bool):
        raise ValueError("Qwen preview use_kv_cache must be a boolean")
    width, height = int(prompt.get("width", 512)), int(prompt.get("height", 512))
    count = int(prompt.get("sample_steps", 28))
    scale = float(prompt.get("guidance_scale", prompt.get("scale", 1.0)))
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn((1, 64, height // 16, width // 16), generator=generator, device=device, dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    tokens = latent.shape[-2] * latent.shape[-1]
    mu = 0.5 + (tokens - 256) * (1.15 - 0.5) / (4096 - 256)
    scheduler.set_timesteps(count, device=device, sigmas=np.linspace(1.0, 1 / count, count), mu=mu)

    def condition(values):
        return tuple(value.to(device=device, dtype=dtype if index == 0 else torch.bool) for index, value in enumerate(values))

    positive = condition(positive)
    negative = condition(negative) if scale > 1 and prompt.get("negative_prompt") is not None and negative is not None else None
    if reference is not None:
        if isinstance(reference, torch.Tensor):
            reference = reference.to(device=device, dtype=dtype)
        else:
            reference = tuple(value.to(device=device, dtype=dtype) for value in reference)

    caches = [PreviewKVForward(dit) for _ in range(1 + (negative is not None))] if (
        use_kv_cache and supports_preview_kv(dit, forward)
    ) else []

    def predict(values, current_latent, sigma, branch):
        if caches:
            return caches[branch](current_latent, values[0], values[1], sigma,
                                  reference_latents=reference,
                                  image_slot_mask=values[2] if reference is not None else None)
        return forward(dit, current_latent, values[0], values[1], sigma,
                       reference_latents=reference,
                       image_slot_mask=values[2] if reference is not None else None)

    try:
        sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)

        def positive_predict(current_latent, sigma):
            return predict(positive, current_latent, sigma, 0)

        def negative_predict(current_latent, sigma):
            return predict(negative, current_latent, sigma, 1)

        def scheduler_step(index, current_latent, velocity, _sigma_current, _sigma_next):
            return scheduler.step(
                velocity, scheduler.timesteps[index], current_latent, return_dict=False
            )[0].to(dtype)

        latent = run_flow_euler_loop(
            latent,
            sigmas,
            positive_predict,
            guidance_scale=scale,
            predict_negative=negative_predict if negative is not None else None,
            use_guidance=negative is not None,
            combine_predictions=lambda positive, negative, scale: negative + scale * (positive - negative),
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
    finally:
        for cache in caches:
            cache.close()
    return latent
