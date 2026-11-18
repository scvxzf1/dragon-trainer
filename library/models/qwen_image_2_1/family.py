"""Qwen Image 2.1 text-to-image flow-matching training forward."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def sample_sigmas(batch_size: int, image_tokens: int, *, device: torch.device) -> Tensor:
    # Match the pipeline's dynamic exponential time shift for this resolution.
    base = torch.rand(batch_size, device=device, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    mu = 0.5 + (image_tokens - 256) * (1.15 - 0.5) / (4096 - 256)
    scale = math.exp(mu)
    return scale / (scale + (1 / base - 1))


def forward_for_loss(
    dit: torch.nn.Module,
    latents: Tensor,
    hiddens: Tensor,
    mask: Tensor,
    sigmas: Tensor,
    *,
    reference_latents: Tensor | Sequence[Tensor] | None = None,
    image_slot_mask: Tensor | None = None,
) -> Tensor:
    if latents.ndim != 4 or latents.shape[1] != 64:
        raise ValueError(f"Qwen Image 2.1 expects [B,64,H,W] latents, got {tuple(latents.shape)}")
    batch, channels, height, width = latents.shape
    if height % 2 or width % 2:
        raise ValueError("Qwen Image 2.1 latent height and width must be even")
    if hiddens.ndim != 3 or hiddens.shape[:2] != mask.shape or hiddens.shape[2] != 4096:
        raise ValueError("Qwen Image 2.1 expects text [B,L,4096] and mask [B,L]")
    image_tokens = height * width
    target_tokens = latents.flatten(2).transpose(1, 2)
    if reference_latents is None:
        if image_slot_mask is not None:
            raise ValueError("Qwen Image 2.1 T2I must not receive an edit image-slot mask")
        tokens = target_tokens
        img_shapes = [[(1, height, width)] for _ in range(batch)]
        img_mask = torch.cat(
            [
                torch.zeros_like(mask, dtype=torch.bool),
                torch.ones(batch, image_tokens // 4, device=mask.device, dtype=torch.bool),
            ],
            dim=1,
        )
    else:
        if batch != 1:
            raise ValueError("Qwen Image 2.1 Edit currently requires batch_size=1")
        references = (reference_latents,) if isinstance(reference_latents, Tensor) else tuple(reference_latents)
        if not references:
            raise ValueError("Qwen Image 2.1 Edit requires at least one reference latent")
        for reference in references:
            if (
                not isinstance(reference, Tensor)
                or reference.ndim != 4
                or reference.shape[:2] != (batch, channels)
                or reference.shape[-2] % 2
                or reference.shape[-1] % 2
            ):
                raise ValueError(
                    "Qwen Image 2.1 Edit reference latents must be [1,64,H,W] with even H/W"
                )
        if image_slot_mask is None or image_slot_mask.shape != mask.shape:
            raise ValueError("Qwen Image 2.1 Edit requires an image-slot mask matching text mask")
        expected_slots = sum(reference.shape[-2] * reference.shape[-1] // 4 for reference in references)
        actual_slots = image_slot_mask.bool().sum(dim=1)
        if not torch.all(actual_slots == expected_slots):
            raise ValueError(
                "Qwen Image 2.1 Edit image-slot count does not match reference latent shape: "
                f"slots={actual_slots.tolist()}, expected={expected_slots}"
            )
        reference_tokens = torch.cat(
            [reference.flatten(2).transpose(1, 2) for reference in references], dim=1
        )
        image_token_count = int(actual_slots[0].item())
        tokens = torch.cat([reference_tokens, target_tokens], dim=1)
        img_shapes = [
            [(1, reference.shape[-2], reference.shape[-1]) for reference in references]
            + [(1, height, width)]
            for _ in range(batch)
        ]
        img_mask = torch.cat(
            [
                image_slot_mask.bool(),
                torch.ones(batch, image_tokens // 4, device=mask.device, dtype=torch.bool),
            ],
            dim=1,
        )
    output = dit(
        hidden_states=tokens,
        encoder_hidden_states=hiddens,
        encoder_hidden_states_mask=mask.bool(),
        timestep=sigmas.to(dtype=tokens.dtype),
        img_shapes=img_shapes,
        img_mask=img_mask,
        return_dict=False,
    )[0]
    expected_output_tokens = hiddens.shape[1] + image_tokens
    if reference_latents is not None:
        expected_output_tokens = (
            hiddens.shape[1]
            - image_token_count
            + reference_tokens.shape[1]
            + image_tokens
        )
    if (
        output.ndim != 3
        or output.shape[0] != batch
        or output.shape[1] != expected_output_tokens
        or output.shape[2] != channels
    ):
        raise ValueError(f"Qwen Image 2.1 DiT output shape mismatch: {tuple(output.shape)}")
    output = output[:, -image_tokens:]
    return output.transpose(1, 2).reshape(batch, channels, height, width)


def compute_noise_pred_and_target(
    trainer, ctx, latents: Tensor, batch, text_encoder_conds, *, is_train: bool = True,
):
    if latents.ndim == 5 and latents.shape[2] == 1:
        latents = latents.squeeze(2)
    if latents.ndim != 4 or latents.shape[1] != 64:
        raise ValueError("Qwen Image 2.1 requires its own 64-channel VAE latent cache")
    dropout_rates = batch.get("caption_dropout_rates") if isinstance(batch, dict) else None
    if dropout_rates is not None and torch.as_tensor(dropout_rates).gt(0).any():
        raise ValueError("Qwen Image 2.1 training requires caption_dropout_rate=0")
    if not text_encoder_conds or text_encoder_conds[0] is None:
        tokens = [value.to(ctx.accelerator.device) for value in batch["input_ids_list"]]
        with torch.no_grad(), ctx.accelerator.autocast():
            text_encoder_conds = ctx.text_encoding_strategy.encode_tokens(
                ctx.tokenize_strategy,
                trainer.get_models_for_text_encoding(ctx.args, ctx.accelerator, ctx.text_encoders),
                tokens,
            )
    hiddens = text_encoder_conds[0].to(device=latents.device, dtype=ctx.weight_dtype)
    mask = text_encoder_conds[1].to(device=latents.device, dtype=torch.bool)
    edit_task = getattr(ctx.args, "qwen_image_2_1_task", "t2i") == "edit"
    reference_latents = None
    image_slot_mask = None
    if edit_task:
        reference_latents = batch.get("qwen_edit_reference_latents")
        if reference_latents is None or len(text_encoder_conds) < 3:
            raise ValueError(
                "Qwen Image 2.1 Edit requires paired reference latents and cached image-slot masks"
            )
        if latents.shape[0] != 1:
            raise ValueError("Qwen Image 2.1 Edit currently requires batch_size=1")
        reference_latents = reference_latents.to(
            device=latents.device, dtype=ctx.weight_dtype
        )
        image_slot_mask = text_encoder_conds[2].to(
            device=latents.device, dtype=torch.bool
        )
    elif batch.get("qwen_edit_reference_latents") is not None:
        raise ValueError(
            "Qwen edit reference images are configured but qwen_image_2_1_task is not 'edit'"
        )
    noise = torch.randn_like(latents)
    sigmas = sample_sigmas(latents.shape[0], latents.shape[2] * latents.shape[3], device=latents.device)
    from library.training.flow_step import mix_latents, velocity_target, unit_weighting

    noisy = mix_latents(latents, noise, sigmas)
    if getattr(ctx.args, "gradient_checkpointing", False):
        noisy.requires_grad_(True)
    with torch.set_grad_enabled(is_train), ctx.accelerator.autocast():
        model_pred = forward_for_loss(
            ctx.unet,
            noisy,
            hiddens,
            mask,
            sigmas,
            reference_latents=reference_latents,
            image_slot_mask=image_slot_mask,
        )
    target = velocity_target(noise, latents)
    return model_pred, target, sigmas, unit_weighting(sigmas, latents)
