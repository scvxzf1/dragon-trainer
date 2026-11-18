"""Live MOSS LoRA previews from startup-cached text conditions."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from library.models.moss_image.sampling import generate_moss_image
from library.runtime.accelerator import prepare_dtype
from library.training.preview_lifecycle import preview_lifecycle
from library.training.preview_lifecycle import apply_preview_router_conditioning
from library.training.sample_preview_common import (
    finish_preview_batch, load_prompts, should_sample,
)

logger = logging.getLogger(__name__)


def _resolve_sample_spec(args, prompt_dict, prompt_replacement):
    prompt = prompt_dict.get("prompt", "")
    negative = prompt_dict.get("negative_prompt", "") or ""
    if prompt_replacement is not None:
        prompt = prompt.replace(*prompt_replacement)
        negative = negative.replace(*prompt_replacement)
    sampler = str(prompt_dict.get("sample_sampler", getattr(args, "sample_sampler", "euler")) or "euler").strip().lower()
    if sampler != "euler":
        raise ValueError(f"MOSS training preview supports only Euler, got {sampler!r}")
    from library.training.preview_spec import validate_preview_prompt

    validate_preview_prompt(prompt_dict, "moss_image", sampler)
    width, height = int(prompt_dict.get("width", 512)), int(prompt_dict.get("height", 512))
    seed = prompt_dict.get("seed")
    if seed is None:
        seed = int(getattr(args, "seed", None) or 0) + int(prompt_dict.get("enum", 0))
    raw_shift = prompt_dict.get("flow_shift")
    return SimpleNamespace(
        prompt=prompt, negative_prompt=negative, width=width, height=height,
        sample_steps=int(prompt_dict.get("sample_steps", 50)),
        scale=float(prompt_dict.get("guidance_scale", prompt_dict.get("scale", 3.0))),
        flow_shift=5.0 if raw_shift in (None, "") else float(raw_shift),
        seed=int(seed), enum=int(prompt_dict.get("enum", 0)),
        uncond_skip_layers=prompt_dict.get("uncond_skip_layers", (22,)),
    )


def _cached_condition(outputs, prompt):
    if outputs is None or prompt not in outputs:
        raise ValueError(f"MOSS sample prompt was not cached before DiT loading: {prompt!r}")
    return outputs[prompt]


def _stage_latent(args, latents, spec, epoch, steps):
    directory = Path(args.output_dir) / "sample" / "latents"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    label = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    prefix = (getattr(args, "output_name", None) or "")
    prefix = prefix + "_" if prefix else ""
    path = directory / f"{prefix}{label}_{spec.enum:02d}_{stamp}_{spec.seed}.pt"
    torch.save({
        "latents": latents.detach().cpu(), "prompt": spec.prompt,
        "enum": spec.enum, "model_family": "moss_image",
        "metadata": {
            "width": spec.width, "height": spec.height, "sample_steps": spec.sample_steps,
            "guidance_scale": spec.scale, "flow_shift": spec.flow_shift,
            "sample_sampler": "euler", "seed": spec.seed, "step": steps,
        },
    }, path)
    return path


def _sample_local(accelerator, args, dit, network, prompts, outputs, replacement, epoch, steps):
    dtype, _ = prepare_dtype(args)
    with preview_lifecycle(dit, network, disable_block_swap=getattr(args, "disable_block_swap_for_eval", False)):
        with torch.no_grad(), accelerator.autocast():
            for prompt_dict in prompts:
                spec = _resolve_sample_spec(args, prompt_dict, replacement)
                positive = _cached_condition(outputs, spec.prompt)
                negative = _cached_condition(outputs, spec.negative_prompt) if spec.scale != 1.0 else None
                logger.info("MOSS preview %sx%s, steps=%s, CFG=%s, shift=%s, seed=%s",
                            spec.width, spec.height, spec.sample_steps, spec.scale, spec.flow_shift, spec.seed)
                latents = generate_moss_image(
                    dit, positive, negative, height=spec.height, width=spec.width,
                    sample_steps=spec.sample_steps, guidance_scale=spec.scale,
                    flow_shift=spec.flow_shift, seed=spec.seed,
                    device=accelerator.device, dtype=dtype, uncond_skip_layers=spec.uncond_skip_layers,
                    step_start_callback=lambda index, current, sigma: apply_preview_router_conditioning(
                        network, current, sigma, step_index=index
                    ),
                    branch_start_callback=lambda index, branch, current, sigma, condition: apply_preview_router_conditioning(
                        network,
                        current,
                        sigma,
                        step_index=index,
                        crossattn_emb=condition[0],
                    ),
                )
                _stage_latent(args, latents, spec, epoch, steps)


def sample_images(
    accelerator, args, epoch, steps, dit, vae, text_encoder, tokenize_strategy,
    text_encoding_strategy, sample_prompts_te_outputs=None, prompt_replacement=None,
    network=None, sample_prompts_snapshot=None,
):
    """Match the shared family-preview interface; sampling never reloads TE."""
    if not should_sample(args, epoch, steps):
        return
    processes = max(int(getattr(accelerator, "num_processes", 1)), 1)
    rank = int(getattr(accelerator, "process_index", 0))
    prompts = load_prompts(
        accelerator, args, text_encoder=None,
        sample_prompts_te_outputs=sample_prompts_te_outputs,
        sample_prompts_snapshot=sample_prompts_snapshot, num_processes=processes,
    )
    if not prompts:
        return
    dit = accelerator.unwrap_model(dit)
    network = accelerator.unwrap_model(network) if network is not None else None
    finish_preview_batch(
        accelerator,
        args,
        processes=processes,
        run_local=lambda: _sample_local(
            accelerator,
            args,
            dit,
            network,
            prompts[rank::processes],
            sample_prompts_te_outputs,
            prompt_replacement,
            epoch,
            steps,
        ),
        failure_message="MOSS sample generation failed on another process",
        vae=vae,
        dit=dit,
        network=network,
    )


def decode_moss_latent(vae, latent):
    """Convert MOSS normalized Flux.2 latents into the shared RGB PNG format."""
    from library.models.moss_image.latent import decode_moss_latents
    from library.training.preview_decode import pixels_to_image
    from library.training.preview_decode_policy import model_device

    if latent.ndim != 4 or latent.shape[1] != 128:
        raise ValueError("MOSS preview decoder expects [B,128,H,W]")
    latent = latent.to(device=model_device(vae), dtype=vae.dtype)
    return pixels_to_image(decode_moss_latents(vae, latent))


def decode_pending_samples(accelerator, args, vae):
    """Decode with shared placement/tiling/retry policy and WebUI naming."""
    from library.training.preview_decode import decode_pending_samples as decode_shared_samples

    decode_shared_samples(accelerator, args, vae, decode_latent=decode_moss_latent, wandb_enabled=True)
