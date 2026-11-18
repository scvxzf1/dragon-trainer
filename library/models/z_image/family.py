"""Official Z-Image flow-matching training contract."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


Z_IMAGE_FLOW_SHIFT = 6.0
Z_IMAGE_NUM_TRAIN_TIMESTEPS = 1000


def shifted_uniform_sigmas(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    shift: float = Z_IMAGE_FLOW_SHIFT,
    num_train_timesteps: int = Z_IMAGE_NUM_TRAIN_TIMESTEPS,
) -> Tensor:
    """Sample the official shifted 1000-step scheduler grid."""
    if num_train_timesteps < 1:
        raise ValueError("num_train_timesteps must be positive")
    u = torch.rand(batch_size, device=device, dtype=torch.float32)
    indices = (u * num_train_timesteps).long()
    base_sigmas = 1.0 - indices.to(torch.float32) / num_train_timesteps
    sigmas = shift * base_sigmas / (1.0 + (shift - 1.0) * base_sigmas)
    return sigmas.to(dtype=dtype)


def prepare_prompt_embeds(hiddens: Tensor, mask: Tensor) -> list[Tensor]:
    if hiddens.ndim != 3 or mask.ndim != 2 or hiddens.shape[:2] != mask.shape:
        raise ValueError(
            "Z-Image prompt cache expects hiddens [B,L,D] and mask [B,L]; "
            f"got {tuple(hiddens.shape)} and {tuple(mask.shape)}"
        )
    if mask.device.type != "cpu":
        # Copy the tiny [B, L] bool mask to CPU once instead of triggering B
        # separate GPU->CPU aten.nonzero syncs via per-row GPU boolean indexing.
        mask_cpu = mask.detach().to(device="cpu", dtype=torch.bool)
        prompts: list[Tensor] = []
        for hidden, row_mask in zip(hiddens, mask_cpu):
            valid_count = int(row_mask.sum().item())
            if valid_count == 0:
                raise ValueError(
                    "Z-Image received an empty prompt embedding. Rebuild text caches and "
                    "keep caption_dropout_rate=0."
                )
            if bool(row_mask[:valid_count].all().item()):
                prompts.append(hidden[:valid_count])
            else:
                idx = row_mask.nonzero(as_tuple=False).flatten().to(device=hidden.device)
                prompts.append(hidden.index_select(0, idx))
        return prompts

    prompts = [hidden[row_mask.bool()] for hidden, row_mask in zip(hiddens, mask)]
    if any(prompt.shape[0] == 0 for prompt in prompts):
        raise ValueError(
            "Z-Image received an empty prompt embedding. Rebuild text caches and "
            "keep caption_dropout_rate=0."
        )
    return prompts


def forward_for_loss(
    dit: torch.nn.Module,
    latents_5d: Tensor,
    prompt_embeds: list[Tensor],
    sigmas: Tensor,
    **_kwargs: Any,
) -> Tensor:
    if latents_5d.ndim != 5 or latents_5d.shape[2] != 1:
        raise ValueError(
            f"Z-Image expects latent shape (B,C,T=1,H,W); got {tuple(latents_5d.shape)}"
        )
    images = [sample for sample in latents_5d]
    model_t = 1.0 - sigmas.reshape(-1).float()
    output = dit(x=images, t=model_t, cap_feats=prompt_embeds)
    samples = output.sample if hasattr(output, "sample") else output[0]
    if not isinstance(samples, (list, tuple)):
        raise TypeError("Z-Image transformer output must contain a per-sample list")
    if len(samples) != latents_5d.shape[0]:
        raise ValueError(
            "Z-Image transformer output batch mismatch: "
            f"{len(samples)} != {latents_5d.shape[0]}"
        )
    return -torch.stack(list(samples), dim=0)


def _prepare_text_conditions(trainer, ctx, batch, text_encoder_conds, *, is_train):
    accelerator = ctx.accelerator
    if not text_encoder_conds or text_encoder_conds[0] is None:
        tokens = [
            value.to(accelerator.device, non_blocking=True)
            for value in batch["input_ids_list"]
        ]
        with torch.no_grad(), accelerator.autocast():
            text_encoder_conds = ctx.text_encoding_strategy.encode_tokens(
                ctx.tokenize_strategy,
                trainer.get_models_for_text_encoding(
                    ctx.args, accelerator, ctx.text_encoders
                ),
                tokens,
            )

    dropout_rates = (
        batch.get("caption_dropout_rates") if isinstance(batch, dict) else None
    )
    if dropout_rates is not None and torch.as_tensor(dropout_rates).gt(0).any():
        raise ValueError("Z-Image training requires caption_dropout_rate=0")
    raw_hiddens, raw_mask = text_encoder_conds[0], text_encoder_conds[1]
    if raw_hiddens.device.type == "cpu" and raw_mask.device.type == "cpu":
        # Slice valid tokens on CPU before H2D: zero GPU stream syncs and only
        # unpadded prompt tokens cross PCIe.
        cpu_prompts = prepare_prompt_embeds(raw_hiddens, raw_mask.to(dtype=torch.bool))
        return [
            prompt.to(
                device=accelerator.device,
                dtype=ctx.weight_dtype,
                non_blocking=True,
            )
            for prompt in cpu_prompts
        ]
    hiddens = raw_hiddens.to(
        device=accelerator.device, dtype=ctx.weight_dtype, non_blocking=True
    )
    mask = raw_mask.to(device=accelerator.device, dtype=torch.bool, non_blocking=True)
    return prepare_prompt_embeds(hiddens, mask)


def compute_noise_pred_and_target(
    trainer,
    ctx,
    latents: Tensor,
    batch,
    text_encoder_conds,
    *,
    is_train: bool = True,
):
    args = ctx.args
    if latents.ndim == 5:
        latents = latents.squeeze(2)
    if latents.ndim != 4:
        raise ValueError(f"Z-Image expects 4D cached latents; got {latents.ndim}D")

    noise = torch.randn_like(latents)
    sigmas = shifted_uniform_sigmas(
        latents.shape[0],
        device=ctx.accelerator.device,
        dtype=torch.float32,
        shift=float(
            getattr(args, "discrete_flow_shift", Z_IMAGE_FLOW_SHIFT)
            or Z_IMAGE_FLOW_SHIFT
        ),
    )
    sigma_view = sigmas.to(dtype=latents.dtype).view(-1, 1, 1, 1)
    noisy_input = (1.0 - sigma_view) * latents + sigma_view * noise

    from library.training.router_conditioning import apply_router_conditioning

    trainer._hydra_warmup_step = apply_router_conditioning(
        network=ctx.network,
        noisy_model_input=noisy_input,
        timesteps=sigmas,
        is_train=is_train,
        warmup_step=int(getattr(trainer, "_hydra_warmup_step", 0)),
        max_train_steps=int(getattr(args, "max_train_steps", 0) or 0),
        gradient_accumulation_steps=int(
            getattr(args, "gradient_accumulation_steps", 1) or 1
        ),
    )
    if getattr(args, "gradient_checkpointing", False):
        noisy_input.requires_grad_(True)

    prompt_embeds = _prepare_text_conditions(
        trainer, ctx, batch, text_encoder_conds, is_train=is_train
    )
    with torch.set_grad_enabled(is_train), ctx.accelerator.autocast():
        model_pred = forward_for_loss(
            ctx.unet,
            noisy_input.unsqueeze(2),
            prompt_embeds,
            sigmas,
        )
    model_pred = model_pred.squeeze(2)
    target = noise - latents
    weighting = torch.ones_like(sigma_view, dtype=torch.float32)
    return model_pred, target, sigmas, weighting
