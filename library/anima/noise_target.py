"""Anima denoising forward with Anima-specific auxiliary objectives."""

from __future__ import annotations

import torch

from library.training.forward import (
    build_forward_kwargs,
    compute_inversion_func_loss,
    prepare_text_conds,
    run_prior_preservation_forward,
    run_vr_reference_forward,
)
from library.training.method_adapter import ForwardArtifacts, StepCtx
from library.training.samplers import SAMPLER_REGISTRY, SamplerContext
from library.training.prior_preservation import (
    blank_prompt_preservation_enabled,
    diff_output_preservation_enabled,
    inverted_mask_prior_enabled,
)
from library.training.contexts import TrainCtx
from library.training.adaptive_personalization import (
    dynamic_denoise_weights,
    observer_enabled,
    record_error,
    should_probe,
    update_observation,
)
from library.training.synchronous_affine import (
    affine_probabilities,
    apply_synchronous_affine,
)


def compute_noise_pred_and_target(
    trainer,
    ctx: TrainCtx,
    latents,
    batch,
    text_encoder_conds,
    *,
    is_train=True,
):
    from library.models.family_registry import get_family_plugin
    from library.training.flow_step import run_family_flow_step

    # Reset per-step adapter aux so stale tensors from a prior step can't
    # leak into the loss composer.
    trainer._state.extras_for_step = {}
    return run_family_flow_step(
        trainer, ctx, get_family_plugin("anima", capability="flow"), latents, batch, text_encoder_conds,
        is_train=is_train,
    )


def sample_anima_sigmas(context, args, device):
    from library.training.multigpu import noise_fixture

    ctx = context.train_context
    accelerator = ctx.accelerator if hasattr(ctx, "accelerator") else None
    sigma_override = None
    fixture = noise_fixture.active()
    if fixture is not None:
        # ``_multigpu_batch_index`` is stamped by the training loop: it is the
        # local dataloader index of this micro-batch, the only place that can
        # identify the samples in a real run (see noise_fixture.IDS_CURSOR).
        context.noise, sigma_override = fixture.apply(
            context.noise,
            context.latents,
            context.batch,
            mode=noise_fixture.ids_mode(),
            local_batch_index=getattr(args, "_multigpu_batch_index", None),
            rank=int(getattr(accelerator, "process_index", 0) or 0),
            world_size=int(getattr(accelerator, "num_processes", 1) or 1),
        )

    sampler_fn = SAMPLER_REGISTRY[getattr(args, "sampler", "default") or "default"]
    sampler_out = sampler_fn(
        SamplerContext(
            args=args,
            noise_scheduler=ctx.noise_scheduler,
            latents=context.latents,
            noise=context.noise,
            device=device,
            weight_dtype=ctx.weight_dtype,
            sigma_override=sigma_override,
        )
    )
    context.noisy_input = sampler_out.noisy_input
    context.timesteps = sampler_out.timesteps.reshape(-1)
    return sampler_out.sigmas.reshape(-1)


def prepare_anima_flow_inputs(context):
    trainer, ctx, batch = context.trainer, context.train_context, context.batch
    args = ctx.args
    context.family_state = context.is_train and observer_enabled(args) and should_probe(
        trainer._state.personalization_observer, args
    )
    affine_p = affine_probabilities(
        trainer._state.personalization_observer, args, context.timesteps
    )
    if context.is_train and affine_p is not None:
        context.latents, context.noise, context.noisy_input, affine_fraction = apply_synchronous_affine(
            context.latents, context.noise, context.noisy_input, batch, affine_p, args
        )
        trainer._state.personalization_observer["last_affine_fraction"] = affine_fraction

    # Method-adapter pre-forward priming must see the same transformed latent
    # and conditioning image that the DiT forward will consume.
    if trainer._adapters:
        step_ctx = StepCtx(
            args=args,
            accelerator=ctx.accelerator,
            network=ctx.network,
            weight_dtype=ctx.weight_dtype,
        )
        for adapter in trainer._adapters:
            adapter.prime_for_forward(step_ctx, batch, context.latents, is_train=context.is_train)


def prepare_anima_conditions(context, batch, text_encoder_conds):
    trainer, ctx = context.trainer, context.train_context
    args, network = ctx.args, ctx.network
    if getattr(args, "gradient_checkpointing", False):
        # Only require grads for text conditions when training the text encoder.
        # When using cached text encoder outputs (or training DiT-only), requiring grads here adds backward work.
        if trainer.is_train_text_encoder(args) and not args.cache_text_encoder_outputs:
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

    # Unpack text encoder conditions, H2D move, and on-device caption dropout.
    tc = prepare_text_conds(
        text_encoder_conds=text_encoder_conds,
        batch=batch,
        text_encoding_strategy=ctx.text_encoding_strategy,
        network=network,
        device=ctx.accelerator.device,
        weight_dtype=ctx.weight_dtype,
        uncond_crossattn_emb=trainer._state.uncond_crossattn_1,
    )
    crossattn_emb = tc.crossattn_emb

    # ChimeraHydra global content router (chimera with
    # ``content_router_source="crossattn"``): fire ONCE per step on the
    # pooled crossattn_emb. apply_router_conditioning above ran before
    # text conds were materialized, so the content router lives outside
    # that helper. No-op on non-chimera networks or per-Linear chimera.
    if (
        getattr(network, "use_content_router", False)
        and crossattn_emb is not None
        and hasattr(network, "set_content")
    ):
        network.set_content(crossattn_emb)

    # Network-level GlobalRouter routed on pooled text
    # (``router_source="crossattn_emb"``, route_per_layer=False). Same
    # timing rationale as the content router above — fires once per step
    # on the materialized cross-attn text features. No-op otherwise.
    if (
        getattr(network, "use_crossattn_router", False)
        and crossattn_emb is not None
        and hasattr(network, "set_crossattn_routing")
    ):
        network.set_crossattn_routing(crossattn_emb)
    return tc


def forward_anima_velocity(anima, noisy, conditions, sigmas, *, context):
    ctx = context.train_context
    trainer = getattr(context, "trainer", None)
    timesteps = context.timesteps if context.timesteps is not None else sigmas
    mask_shape = (noisy.shape[0], 1, noisy.shape[-2], noisy.shape[-1])
    mask_key = (mask_shape[0], *mask_shape[-2:], ctx.weight_dtype, noisy.device)
    cache = getattr(trainer, "_padding_mask_cache", {})
    padding_mask = cache.get(mask_key)
    if padding_mask is None:
        padding_mask = torch.zeros(mask_shape, dtype=ctx.weight_dtype, device=noisy.device)
        cache[mask_key] = padding_mask
    noisy_5d = noisy.unsqueeze(2)
    if conditions.crossattn_emb is None:
        prediction = anima(
            noisy_5d, timesteps, conditions.prompt_embeds,
            padding_mask=padding_mask,
            target_input_ids=conditions.t5_input_ids,
            target_attention_mask=conditions.t5_attn_mask,
            source_attention_mask=conditions.attn_mask,
        )
    else:
        forward = build_forward_kwargs(
            network=ctx.network,
            crossattn_emb=conditions.crossattn_emb,
            t5_attn_mask=conditions.t5_attn_mask,
            timesteps=timesteps,
        )
        prediction = anima(
            noisy_5d, timesteps, forward.crossattn_emb,
            padding_mask=padding_mask, **forward.kw,
        )
        if trainer is not None:
            _collect_anima_forward_extras(
                context, anima, noisy_5d, sigmas, timesteps, padding_mask, forward, prediction,
            )
    return prediction.squeeze(2)


def _collect_anima_forward_extras(
    context, anima, noisy_model_input, sigmas, timesteps, padding_mask, forward, model_pred,
):
    trainer, ctx, batch = context.trainer, context.train_context, context.batch
    args, accelerator, network, weight_dtype = ctx.args, ctx.accelerator, ctx.network, ctx.weight_dtype
    latents, noise, is_train = context.latents, context.noise, context.is_train
    crossattn_emb, kw, has_postfix = forward.crossattn_emb, forward.kw, forward.has_postfix

    # Method-adapter extra forwards (soft-tokens, …).
    # Each adapter sees the primary forward's inputs + 5D output
    # and may run additional anima(...) calls inside this same
    # autocast / grad scope, returning aux loss tensors keyed for
    # the LossComposer.
    if trainer._adapters:
        primary = ForwardArtifacts(
            anima_call=anima,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            padding_mask=padding_mask,
            forward_kwargs=kw,
            model_pred=model_pred,
            noise=noise,
            latents=latents,
            is_train=is_train,
        )
        step_ctx = StepCtx(
            args=args,
            accelerator=accelerator,
            network=network,
            weight_dtype=weight_dtype,
        )
        for adapter in trainer._adapters:
            out = adapter.extra_forwards(step_ctx, primary)
            if out:
                trainer._state.extras_for_step.update(out)

    if is_train and blank_prompt_preservation_enabled(args):
        from library.inference.uncond import uncond_for_batch

        if trainer._state.uncond_crossattn_1 is None:
            raise RuntimeError(
                "blank_prompt_preservation requires the T5('') sidecar "
                "to be staged before training starts"
            )
        blank_crossattn_emb = uncond_for_batch(
            trainer._state.uncond_crossattn_1, crossattn_emb
        ).to(device=accelerator.device, dtype=weight_dtype)
        prior_pred = run_prior_preservation_forward(
            anima_call=anima,
            network=network,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            crossattn_emb=blank_crossattn_emb,
            padding_mask=padding_mask,
            forward_kwargs={},
        )
        trainer._state.extras_for_step["prior_preservation"] = {
            "prior_pred": prior_pred.detach(),
            "mode": "blank_prompt",
        }
    elif is_train and diff_output_preservation_enabled(args):
        prior_crossattn_emb = batch.get("prior_crossattn_emb")
        if prior_crossattn_emb is None:
            raise RuntimeError(
                "diff_output_preservation requires TE caches with "
                "prior_crossattn_emb; re-run text caching/preprocess "
                "after setting diff_output_preservation_class."
            )
        prior_crossattn_emb = prior_crossattn_emb.to(
            device=accelerator.device,
            dtype=weight_dtype,
            non_blocking=True,
        )
        prior_pred = run_prior_preservation_forward(
            anima_call=anima,
            network=network,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            crossattn_emb=prior_crossattn_emb,
            padding_mask=padding_mask,
            forward_kwargs={},
        )
        trainer._state.extras_for_step["prior_preservation"] = {
            "prior_pred": prior_pred.detach(),
            "mode": "diff_output",
        }

    if is_train and inverted_mask_prior_enabled(args, batch):
        prior_pred = run_prior_preservation_forward(
            anima_call=anima,
            network=network,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            padding_mask=padding_mask,
            forward_kwargs=kw,
        )
        trainer._state.extras_for_step["inverted_mask_prior"] = {
            "prior_pred": prior_pred.detach(),
        }

    if context.family_state:
        try:
            observer_aux = trainer._state.extras_for_step.get(
                "inverted_mask_prior"
            )
            observer_pred = (
                observer_aux.get("prior_pred") if observer_aux else None
            )
            if observer_pred is None:
                observer_pred = run_prior_preservation_forward(
                    anima_call=anima,
                    network=network,
                    noisy_model_input=noisy_model_input,
                    timesteps=timesteps,
                    crossattn_emb=crossattn_emb,
                    padding_mask=padding_mask,
                    forward_kwargs=kw,
                )
            trainer._state.extras_for_step["personalization_observer"] = {
                "base_pred": observer_pred.detach()
            }
        except Exception as exc:
            record_error(trainer._state.personalization_observer, exc)

    # Functional MSE loss against a sampled stochastic inversion run.
    # The captures dict is populated by trainer-owned forward hooks
    # on cross_attn.output_proj at ``trainer._func_blocks``.
    trainer._func_loss = None
    if is_train and getattr(trainer, "_func_blocks", None):
        trainer._func_loss = compute_inversion_func_loss(
            anima_call=anima,
            captures=trainer._func_captures,
            block_indices=trainer._func_blocks,
            batch=batch,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            padding_mask=padding_mask,
            has_postfix=has_postfix,
            kw=kw,
            device=accelerator.device,
            dtype=weight_dtype,
        )

    # Variance-reduced FM control variate (AsymFlow §5.2). Stash the
    # residual `z` so the loss composer can blend `(y + λ·z)²`.
    if (
        is_train
        and float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0
    ):
        z_residual = run_vr_reference_forward(
            anima_call=anima,
            network=network,
            latents=latents,
            noise=noise,
            sigmas=sigmas.reshape(-1, 1, 1, 1).to(latents.dtype),
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            padding_mask=padding_mask,
            forward_kwargs=kw,
            weight_dtype=weight_dtype,
            fei_sigma_low_div=float(args.vr_fei_sigma_low_div),
        )
        trainer._state.extras_for_step["vr"] = {
            "z": z_residual.detach(),
            "state": trainer._state.vr,
        }
def observe_anima_flow(context, model_pred, target):
    trainer, args = context.trainer, context.train_context.args
    observer_aux = trainer._state.extras_for_step.get("personalization_observer")
    if observer_aux is not None:
        try:
            update_observation(
                trainer._state.personalization_observer,
                args,
                timesteps=context.timesteps,
                adapter_pred=model_pred,
                base_pred=observer_aux["base_pred"],
                target=target,
            )
        except Exception as exc:
            record_error(trainer._state.personalization_observer, exc)

def adjust_anima_weighting(context, weighting):
    trainer, args = context.trainer, context.train_context.args
    dynamic_weight = dynamic_denoise_weights(
        trainer._state.personalization_observer, args, context.timesteps
    )
    if dynamic_weight is not None:
        dynamic_weight = dynamic_weight.reshape(
            dynamic_weight.shape[0], *([1] * (weighting.ndim - 1))
        )
        weighting = weighting * dynamic_weight.to(
            device=weighting.device, dtype=weighting.dtype
        )

    return weighting
