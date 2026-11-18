"""Z-Image training-time previews using the live transformer and adapter."""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace
from typing import Optional

import torch

from library.training.preview_lifecycle import preview_lifecycle
from library.training.preview_lifecycle import apply_preview_router_conditioning
from library.datasets.buckets import snap_sample_size
from library.models.z_image.family import forward_for_loss, prepare_prompt_embeds
from library.runtime.accelerator import prepare_dtype
from library.runtime.device import clean_memory_on_device
from library.training.sample_preview_common import (
    finish_preview_batch,
    load_prompts,
    should_sample,
)

logger = logging.getLogger(__name__)


def _as_prompt_embeds(encoded, *, device: torch.device, dtype: torch.dtype):
    if encoded is None or len(encoded) < 2:
        return None
    hiddens = torch.as_tensor(encoded[0])
    mask = torch.as_tensor(encoded[1])
    if hiddens.ndim == 2:
        hiddens = hiddens.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    return prepare_prompt_embeds(
        hiddens.to(device=device, dtype=dtype),
        mask.to(device=device, dtype=torch.bool),
    )


def _encode_prompt(
    prompt: str,
    *,
    text_encoder,
    tokenize_strategy,
    text_encoding_strategy,
    sample_prompts_te_outputs,
    device: torch.device,
    dtype: torch.dtype,
):
    encoded = None
    if sample_prompts_te_outputs and prompt in sample_prompts_te_outputs:
        encoded = sample_prompts_te_outputs[prompt]
    elif text_encoder is not None:
        tokens = tokenize_strategy.tokenize(prompt)
        encoded = text_encoding_strategy.encode_tokens(
            tokenize_strategy, [text_encoder], tokens
        )
    return _as_prompt_embeds(encoded, device=device, dtype=dtype)


def _default_flow_shift(args) -> float:
    model_path = str(getattr(args, "pretrained_model_name_or_path", "") or "")
    if "turbo" in os.path.basename(model_path).lower():
        return 3.0
    return float(getattr(args, "discrete_flow_shift", 6.0) or 6.0)


def _resolve_sample_spec(args, prompt_dict, prompt_replacement):
    prompt = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get("negative_prompt", "") or ""
    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        negative_prompt = negative_prompt.replace(
            prompt_replacement[0], prompt_replacement[1]
        )
    sampler = (
        str(
            prompt_dict.get("sample_sampler", getattr(args, "sample_sampler", "euler"))
            or "euler"
        )
        .strip()
        .lower()
    )
    if sampler != "euler":
        raise ValueError(
            "Z-Image training preview supports only the official Euler sampler, "
            f"got {sampler!r}"
        )
    seed = prompt_dict.get("seed")
    if seed is None:
        seed = int(getattr(args, "seed", None) or 0) + int(prompt_dict.get("enum", 0))
    width, height = snap_sample_size(
        int(prompt_dict.get("width", 512)),
        int(prompt_dict.get("height", 512)),
    )
    sample_steps = int(prompt_dict.get("sample_steps", 28))
    if sample_steps <= 0:
        raise ValueError(
            f"Z-Image training preview sample_steps must be positive, got {sample_steps}"
        )
    raw_shift = prompt_dict.get("flow_shift")
    flow_shift = (
        _default_flow_shift(args) if raw_shift in (None, "") else float(raw_shift)
    )
    return SimpleNamespace(
        prompt=prompt,
        negative_prompt=negative_prompt,
        sample_steps=sample_steps,
        width=width,
        height=height,
        scale=float(prompt_dict.get("guidance_scale", prompt_dict.get("scale", 0.0))),
        flow_shift=flow_shift,
        seed=int(seed),
        enum=int(prompt_dict.get("enum", 0)),
    )


def _make_scheduler(flow_shift: float):
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=float(flow_shift),
        use_dynamic_shifting=False,
    )


def generate_z_image(
    dit,
    cond_embeds: list[torch.Tensor],
    uncond_embeds: Optional[list[torch.Tensor]],
    *,
    height: int,
    width: int,
    sample_steps: int,
    guidance_scale: float,
    flow_shift: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run the official Z-Image Euler flow-matching denoising loop."""
    latent_h = 2 * (int(height) // 16)
    latent_w = 2 * (int(width) // 16)
    channels = int(getattr(dit, "in_channels", 16))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        (1, channels, latent_h, latent_w),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device)
    scheduler = _make_scheduler(flow_shift)
    scheduler.sigma_min = 0.0
    scheduler.set_timesteps(sample_steps, device=device)

    use_cfg = guidance_scale > 0 and uncond_embeds is not None
    for index, timestep in enumerate(scheduler.timesteps):
        sigma = (timestep.float() / 1000.0).reshape(1)
        if use_cfg:
            preview_network = getattr(dit, "_preview_network", None)
            apply_preview_router_conditioning(
                preview_network,
                latents,
                sigma,
                step_index=index,
                crossattn_emb=cond_embeds[0].unsqueeze(0),
            )
            positive = (
                forward_for_loss(
                    dit,
                    latents.to(dtype=dtype).unsqueeze(2),
                    cond_embeds,
                    sigma,
                )
                .squeeze(2)
                .float()
            )
            apply_preview_router_conditioning(
                preview_network,
                latents,
                sigma,
                step_index=index,
                crossattn_emb=uncond_embeds[0].unsqueeze(0),
            )
            negative = (
                forward_for_loss(
                    dit,
                    latents.to(dtype=dtype).unsqueeze(2),
                    uncond_embeds,
                    sigma,
                )
                .squeeze(2)
                .float()
            )
            velocity = positive + guidance_scale * (positive - negative)
        else:
            apply_preview_router_conditioning(
                getattr(dit, "_preview_network", None),
                latents,
                sigma,
                step_index=index,
                crossattn_emb=cond_embeds[0].unsqueeze(0),
            )
            velocity = (
                forward_for_loss(
                    dit,
                    latents.to(dtype=dtype).unsqueeze(2),
                    cond_embeds,
                    sigma,
                )
                .squeeze(2)
                .float()
            )
        latents = scheduler.step(velocity, timestep, latents, return_dict=False)[
            0
        ].float()
    return latents


def _stage_latent(args, save_dir, latents, spec, epoch, steps: int) -> str:
    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    output_name = "" if args.output_name is None else args.output_name + "_"
    stem = f"{output_name}{num_suffix}_{spec.enum:02d}_{ts_str}_{spec.seed}"
    latents_dir = os.path.join(save_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)
    latent_path = os.path.join(latents_dir, stem + ".pt")
    torch.save(
        {
            "latents": latents.detach().to("cpu"),
            "prompt": spec.prompt,
            "enum": spec.enum,
            "model_family": "z_image",
        },
        latent_path,
    )
    return latent_path


def _sample_image(
    accelerator,
    args,
    dit,
    text_encoder,
    tokenize_strategy,
    text_encoding_strategy,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    spec = _resolve_sample_spec(args, prompt_dict, prompt_replacement)
    runtime_dtype, _ = prepare_dtype(args)
    cond_embeds = _encode_prompt(
        spec.prompt,
        text_encoder=text_encoder,
        tokenize_strategy=tokenize_strategy,
        text_encoding_strategy=text_encoding_strategy,
        sample_prompts_te_outputs=sample_prompts_te_outputs,
        device=accelerator.device,
        dtype=runtime_dtype,
    )
    if cond_embeds is None:
        logger.warning("Cannot encode Z-Image sample prompt; skipping sample")
        return None
    uncond_embeds = None
    if spec.scale > 0:
        uncond_embeds = _encode_prompt(
            spec.negative_prompt,
            text_encoder=text_encoder,
            tokenize_strategy=tokenize_strategy,
            text_encoding_strategy=text_encoding_strategy,
            sample_prompts_te_outputs=sample_prompts_te_outputs,
            device=accelerator.device,
            dtype=runtime_dtype,
        )
        if uncond_embeds is None:
            logger.warning("Cannot encode Z-Image negative prompt; skipping sample")
            return None
    logger.info(
        "  Z-Image prompt: %s, size: %sx%s, steps: %s, scale: %s, "
        "flow_shift: %s, sampler: euler, seed: %s",
        spec.prompt,
        spec.width,
        spec.height,
        spec.sample_steps,
        spec.scale,
        spec.flow_shift,
        spec.seed,
    )
    latents = generate_z_image(
        dit,
        cond_embeds,
        uncond_embeds,
        height=spec.height,
        width=spec.width,
        sample_steps=spec.sample_steps,
        guidance_scale=spec.scale,
        flow_shift=spec.flow_shift,
        seed=spec.seed,
        device=accelerator.device,
        dtype=runtime_dtype,
    )
    return _stage_latent(args, save_dir, latents, spec, epoch, steps)


def sample_images(
    accelerator,
    args,
    epoch,
    steps,
    dit,
    vae,
    text_encoder,
    tokenize_strategy,
    text_encoding_strategy,
    sample_prompts_te_outputs=None,
    prompt_replacement=None,
    network=None,
    sample_prompts_snapshot=None,
):
    if not should_sample(args, epoch, steps):
        return
    process_index = int(getattr(accelerator, "process_index", 0))
    num_processes = max(int(getattr(accelerator, "num_processes", 1)), 1)
    prompts = load_prompts(
        accelerator,
        args,
        text_encoder=text_encoder,
        sample_prompts_te_outputs=sample_prompts_te_outputs,
        sample_prompts_snapshot=sample_prompts_snapshot,
        num_processes=num_processes,
    )
    if not prompts:
        return
    local_prompts = prompts[process_index::num_processes]
    dit = accelerator.unwrap_model(dit)
    if text_encoder is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)
    net = accelerator.unwrap_model(network) if network is not None else None
    def run_local():
        with preview_lifecycle(
            dit, net, disable_block_swap=getattr(args, "disable_block_swap_for_eval", False)
        ):
            save_dir = os.path.join(args.output_dir, "sample")
            os.makedirs(save_dir, exist_ok=True)
            with torch.no_grad(), accelerator.autocast():
                for prompt_dict in local_prompts:
                    _sample_image(
                        accelerator,
                        args,
                        dit,
                        text_encoder,
                        tokenize_strategy,
                        text_encoding_strategy,
                        save_dir,
                        prompt_dict,
                        epoch,
                        steps,
                        sample_prompts_te_outputs,
                        prompt_replacement,
                    )

    finish_preview_batch(
        accelerator,
        args,
        processes=num_processes,
        run_local=run_local,
        failure_message="Z-Image sample generation failed on another process",
        vae=vae,
        dit=dit,
        network=net,
    )


def decode_pending_samples(accelerator, args, vae) -> None:
    """Decode normalized Flux latents with the shared preview VAE policy."""
    from library.training.preview_decode import decode_pending_samples as decode_shared_samples, decode_z_image_latent

    decode_shared_samples(accelerator, args, vae, decode_latent=decode_z_image_latent)
