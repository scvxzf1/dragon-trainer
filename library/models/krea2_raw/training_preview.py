"""Krea-2 training-time sample previews using the live training models."""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace
from typing import Any, Optional

import torch

from library.training.preview_lifecycle import preview_lifecycle
from library.datasets.buckets import snap_sample_size
from library.models.krea2_raw.family import Krea2TextEmbedding
from library.models.krea2_raw.inference_runner import generate_krea2
from library.runtime.accelerator import prepare_dtype
from library.training.sample_preview_common import (
    finish_preview_batch,
    load_prompts as _load_prompts,
    should_sample as _should_sample,
)

logger = logging.getLogger(__name__)


def _as_embedding(encoded, *, device: torch.device, dtype: torch.dtype):
    if encoded is None:
        return None
    if len(encoded) < 2:
        raise ValueError(
            "Krea-2 sample text encoding must contain hiddens and mask, "
            f"got {len(encoded)} item(s)"
        )
    hiddens, mask = encoded[:2]
    hiddens = torch.as_tensor(hiddens)
    mask = torch.as_tensor(mask)
    if hiddens.ndim == 3:
        hiddens = hiddens.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if (
        hiddens.ndim != 4
        or mask.ndim != 2
        or hiddens.shape[2] != 12
        or hiddens.shape[:2] != mask.shape
    ):
        raise ValueError(
            "Krea-2 sample text shapes must be [B,L,12,D] and [B,L], "
            f"got {tuple(hiddens.shape)} and {tuple(mask.shape)}"
        )
    return Krea2TextEmbedding(
        hiddens=hiddens.to(device=device, dtype=dtype),
        mask=mask.to(device=device, dtype=torch.bool),
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
    return _as_embedding(encoded, device=device, dtype=dtype)


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
            "Krea-2 training preview supports only the official Euler sampler, "
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
            f"Krea-2 training preview sample_steps must be positive, got {sample_steps}"
        )
    return SimpleNamespace(
        prompt=prompt,
        negative_prompt=negative_prompt,
        sample_steps=sample_steps,
        width=width,
        height=height,
        scale=float(prompt_dict.get("guidance_scale", prompt_dict.get("scale", 4.5))),
        seed=int(seed),
        enum=int(prompt_dict.get("enum", 0)),
    )


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
        },
        latent_path,
    )
    return latent_path


def _sample_image(
    accelerator,
    args,
    dit,
    network,
    text_encoder,
    tokenize_strategy,
    text_encoding_strategy,
    save_dir: str,
    prompt_dict: dict[str, Any],
    epoch,
    steps: int,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    spec = _resolve_sample_spec(args, prompt_dict, prompt_replacement)
    runtime_dtype, _ = prepare_dtype(args)

    logger.info(
        "  Krea-2 prompt: %s, size: %sx%s, steps: %s, scale: %s, "
        "sampler: euler, seed: %s",
        spec.prompt,
        spec.width,
        spec.height,
        spec.sample_steps,
        spec.scale,
        spec.seed,
    )
    cond_emb = _encode_prompt(
        spec.prompt,
        text_encoder=text_encoder,
        tokenize_strategy=tokenize_strategy,
        text_encoding_strategy=text_encoding_strategy,
        sample_prompts_te_outputs=sample_prompts_te_outputs,
        device=accelerator.device,
        dtype=runtime_dtype,
    )
    if cond_emb is None:
        logger.warning("Cannot encode Krea-2 sample prompt, skipping sample")
        return None
    uncond_emb = cond_emb
    if spec.scale > 0:
        uncond_emb = _encode_prompt(
            spec.negative_prompt,
            text_encoder=text_encoder,
            tokenize_strategy=tokenize_strategy,
            text_encoding_strategy=text_encoding_strategy,
            sample_prompts_te_outputs=sample_prompts_te_outputs,
            device=accelerator.device,
            dtype=runtime_dtype,
        )
        if uncond_emb is None:
            logger.warning("Cannot encode Krea-2 negative prompt, skipping sample")
            return None

    sample_args = SimpleNamespace(
        image_size=(spec.height, spec.width),
        infer_steps=spec.sample_steps,
        guidance_scale=spec.scale,
    )
    latents = generate_krea2(
        sample_args,
        dit,
        network,
        cond_emb,
        uncond_emb,
        accelerator.device,
        spec.seed,
        runtime_dtype,
    )
    return _stage_latent(args, save_dir, latents, spec, epoch, steps)


def _run_local_samples(
    accelerator,
    args,
    *,
    dit,
    net,
    text_encoder,
    tokenize_strategy,
    text_encoding_strategy,
    local_prompts,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
) -> Optional[Exception]:
    sample_error: Optional[Exception] = None
    try:
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
                        net,
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
    except Exception as exc:  # coordinate failures before entering a barrier
        sample_error = exc
    return sample_error


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
    """Generate Krea-2 previews without reloading the live training models."""
    if not _should_sample(args, epoch, steps):
        return

    process_index = int(getattr(accelerator, "process_index", 0))
    num_processes = max(int(getattr(accelerator, "num_processes", 1)), 1)
    if getattr(accelerator, "is_main_process", True):
        logger.info(
            "Generating Krea-2 sample images at step %s across %s process(es)",
            steps,
            num_processes,
        )
    prompts = _load_prompts(
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
    finish_preview_batch(
        accelerator,
        args,
        processes=num_processes,
        run_local=lambda: _run_local_samples(
            accelerator,
            args,
            dit=dit,
            net=net,
            text_encoder=text_encoder,
            tokenize_strategy=tokenize_strategy,
            text_encoding_strategy=text_encoding_strategy,
            local_prompts=local_prompts,
            epoch=epoch,
            steps=steps,
            sample_prompts_te_outputs=sample_prompts_te_outputs,
            prompt_replacement=prompt_replacement,
        ),
        failure_message="Krea-2 sample generation failed on another process",
        vae=vae,
        dit=dit,
        network=net,
    )


def decode_pending_samples(accelerator, args, vae) -> None:
    """Decode Krea-2's Anima-layout latents with the shared preview VAE policy."""
    from library.training.preview_decode import (
        decode_anima_latent,
        decode_pending_samples as decode_shared_samples,
    )

    decode_shared_samples(
        accelerator,
        args,
        vae,
        decode_latent=decode_anima_latent,
        wandb_enabled=True,
    )
