"""MOSS 4D flow-matching training contract.

Uniform sampling of the official training_shift=1 scheduler grid is the
initial finetuning policy, not a reconstruction of unpublished training data
or timestep sampling. Inference shift=5 is deliberately independent.
"""
from __future__ import annotations
import torch

MOSS_NUM_TRAIN_TIMESTEPS = 1000
MOSS_TRAINING_SHIFT = 1.0


def uniform_grid_sigmas(batch_size, *, device, dtype=torch.float32, generator=None):
    indices = torch.randint(
        MOSS_NUM_TRAIN_TIMESTEPS, (batch_size,), device=device, generator=generator
    )
    return (1.0 - indices.float() / MOSS_NUM_TRAIN_TIMESTEPS).to(dtype=dtype)


def sample_moss_flow(latents, *, generator=None):
    """The MOSS flow-matching training inputs from clean latents.

    Returns ``(noise, sigmas, noisy_input)``. This is the exact input/noise
    policy of :func:`compute_noise_pred_and_target`, extracted so the PP trainer
    can reuse it (and synchronize it across stages) without re-implementing the
    math. With ``generator=None`` the draw comes from the default global RNG in
    the same order as before, so the non-PP path is unchanged.
    """

    noise = (
        torch.randn_like(latents)
        if generator is None
        else torch.randn(
            latents.shape, generator=generator, device=latents.device, dtype=latents.dtype
        )
    )
    sigmas = uniform_grid_sigmas(
        latents.shape[0], device=latents.device, generator=generator
    )
    from library.training.flow_step import mix_latents

    noisy_input = mix_latents(latents, noise, sigmas)
    return noise, sigmas, noisy_input


def forward_for_loss(dit, latents, text_emb, sigmas, **kwargs):
    """4D [B,128,H,W] + (hiddens, binary mask) + sigma → 4D velocity."""
    if latents.ndim != 4 or latents.shape[1] != 128:
        raise ValueError("MOSS forward_for_loss requires [B,128,H,W]")
    if hasattr(text_emb, "hiddens"):
        hiddens, mask = text_emb.hiddens, text_emb.mask
    else:
        hiddens, mask = text_emb
    output = dit(
        hidden_states=latents, timestep=sigmas.reshape(-1).float() * 1000,
        encoder_hidden_states=hiddens, encoder_attention_mask=mask, **kwargs
    )
    return output.sample if hasattr(output, "sample") else output[0]


def _prepare_text_conditions(trainer, ctx, batch, conditions):
    if not conditions or conditions[0] is None:
        tokens = [value.to(ctx.accelerator.device) for value in batch["input_ids_list"]]
        with torch.no_grad(), ctx.accelerator.autocast():
            conditions = ctx.text_encoding_strategy.encode_tokens(
                ctx.tokenize_strategy,
                trainer.get_models_for_text_encoding(ctx.args, ctx.accelerator, ctx.text_encoders),
                tokens,
            )
    rates = batch.get("caption_dropout_rates") if isinstance(batch, dict) else None
    if rates is not None and torch.as_tensor(rates).gt(0).any():
        raise ValueError("MOSS initial training requires caption_dropout_rate=0")
    return (
        conditions[0].to(device=ctx.accelerator.device, dtype=ctx.weight_dtype),
        conditions[1].to(device=ctx.accelerator.device),
    )


def compute_noise_pred_and_target(
    trainer, ctx, latents, batch, text_encoder_conds, *, is_train=True
):
    from library.models.family_registry import get_family_plugin
    from library.training.flow_step import run_family_flow_step

    return run_family_flow_step(
        trainer,
        ctx,
        get_family_plugin("moss_image"),
        latents,
        batch,
        text_encoder_conds,
        is_train=is_train,
    )
