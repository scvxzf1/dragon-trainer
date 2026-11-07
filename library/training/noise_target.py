"""Primary denoising forward: noise prediction, target, and per-step aux losses."""

from __future__ import annotations

import torch

from library.anima import models as anima_models
from library.anima import training as anima_train_utils
from library.training.forward import (
    build_forward_kwargs,
    compute_inversion_func_loss,
    prepare_text_conds,
    run_prior_preservation_forward,
    run_vr_reference_forward,
)
from library.training.method_adapter import ForwardArtifacts, StepCtx
from library.training.router_conditioning import apply_router_conditioning
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
    args = ctx.args
    accelerator = ctx.accelerator
    noise_scheduler = ctx.noise_scheduler
    unet = ctx.unet
    network = ctx.network
    weight_dtype = ctx.weight_dtype
    anima: anima_models.Anima = unet

    # Reset per-step adapter aux so stale tensors from a prior step can't
    # leak into the loss composer.
    trainer._state.extras_for_step = {}
    observer_probe = is_train and observer_enabled(args) and should_probe(
        trainer._state.personalization_observer, args
    )

    # Sample noise
    if latents.ndim == 5:  # Fallback for 5D latents (old cache)
        latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]

    noise = torch.randn_like(latents)

    # Opt-in measurement fixture (contract §13 V-2.4): pin (noise, sigma) per
    # sample id so a single-vs-dual comparison sees identical random inputs.
    # ``active()`` is None unless ANIMA_MULTIGPU_NOISE_FIXTURE is set, so the
    # production draw above is used unchanged in every normal run.
    from library.training.multigpu import noise_fixture

    sigma_override = None
    fixture = noise_fixture.active()
    if fixture is not None:
        # ``_multigpu_batch_index`` is stamped by the training loop: it is the
        # local dataloader index of this micro-batch, the only place that can
        # identify the samples in a real run (see noise_fixture.IDS_CURSOR).
        noise, sigma_override = fixture.apply(
            noise,
            latents,
            batch,
            mode=noise_fixture.ids_mode(),
            local_batch_index=getattr(args, "_multigpu_batch_index", None),
            rank=int(getattr(accelerator, "process_index", 0) or 0),
            world_size=int(getattr(accelerator, "num_processes", 1) or 1),
        )

    # Draw noisy input + timesteps via the sampler registry (M1).
    sampler_fn = SAMPLER_REGISTRY[getattr(args, "sampler", "default") or "default"]
    sampler_out = sampler_fn(
        SamplerContext(
            args=args,
            noise_scheduler=noise_scheduler,
            latents=latents,
            noise=noise,
            device=accelerator.device,
            weight_dtype=weight_dtype,
            sigma_override=sigma_override,
        )
    )
    noisy_model_input = sampler_out.noisy_input
    timesteps = sampler_out.timesteps  # [0,1]-scaled, float32
    sigmas = sampler_out.sigmas

    affine_p = affine_probabilities(
        trainer._state.personalization_observer, args, timesteps
    )
    if is_train and affine_p is not None:
        latents, noise, noisy_model_input, affine_fraction = apply_synchronous_affine(
            latents, noise, noisy_model_input, batch, affine_p, args
        )
        trainer._state.personalization_observer["last_affine_fraction"] = affine_fraction

    # Method-adapter pre-forward priming must see the same transformed latent
    # and conditioning image that the DiT forward will consume.
    if trainer._adapters:
        step_ctx = StepCtx(
            args=args,
            accelerator=accelerator,
            network=network,
            weight_dtype=weight_dtype,
        )
        for adapter in trainer._adapters:
            adapter.prime_for_forward(step_ctx, batch, latents, is_train=is_train)

    # Per-step network conditioning: timestep masks, σ/FEI routers, balance-loss warmup.
    trainer._hydra_warmup_step = apply_router_conditioning(
        network=network,
        noisy_model_input=noisy_model_input,
        timesteps=timesteps,
        is_train=is_train,
        warmup_step=int(getattr(trainer, "_hydra_warmup_step", 0)),
        max_train_steps=int(getattr(args, "max_train_steps", 0) or 0),
        gradient_accumulation_steps=int(
            getattr(args, "gradient_accumulation_steps", 1) or 1
        ),
    )

    # Gradient checkpointing support
    if args.gradient_checkpointing:
        noisy_model_input.requires_grad_(True)
        # Only require grads for text conditions when training the text encoder.
        # When using cached text encoder outputs (or training DiT-only), requiring grads here adds backward work.
        if _is_train_text_encoder(trainer, args) and not args.cache_text_encoder_outputs:
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

    # Unpack text encoder conditions, H2D move, and on-device caption dropout.
    tc = prepare_text_conds(
        text_encoder_conds=text_encoder_conds,
        batch=batch,
        text_encoding_strategy=ctx.text_encoding_strategy,
        network=network,
        device=accelerator.device,
        weight_dtype=weight_dtype,
        uncond_crossattn_emb=trainer._state.uncond_crossattn_1,
    )
    crossattn_emb = tc.crossattn_emb
    prompt_embeds = tc.prompt_embeds
    attn_mask = tc.attn_mask
    t5_input_ids = tc.t5_input_ids
    t5_attn_mask = tc.t5_attn_mask

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

    # Create padding mask
    bs = latents.shape[0]
    h_latent = latents.shape[-2]
    w_latent = latents.shape[-1]
    padding_mask_key = (bs, h_latent, w_latent, weight_dtype, accelerator.device)
    padding_mask = trainer._padding_mask_cache.get(padding_mask_key)
    if padding_mask is None:
        padding_mask = torch.zeros(
            bs, 1, h_latent, w_latent, dtype=weight_dtype, device=accelerator.device
        )
        trainer._padding_mask_cache[padding_mask_key] = padding_mask

    # Call model
    noisy_model_input = noisy_model_input.unsqueeze(
        2
    )  # 4D to 5D, [B, C, H, W] -> [B, C, 1, H, W]

    with torch.set_grad_enabled(is_train), accelerator.autocast():
        if crossattn_emb is None:
            model_pred = anima(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                padding_mask=padding_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
            )
        else:
            # crossattn_emb is already in target (T5-compatible) space.
            # Postfix splice kwargs.
            fk = build_forward_kwargs(
                network=network,
                crossattn_emb=crossattn_emb,
                t5_attn_mask=t5_attn_mask,
                timesteps=timesteps,
            )
            crossattn_emb = fk.crossattn_emb
            kw = fk.kw
            has_postfix = fk.has_postfix
            model_pred = anima(
                noisy_model_input,
                timesteps,
                crossattn_emb,
                padding_mask=padding_mask,
                **kw,
            )

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

            if observer_probe:
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
                    sigmas=sigmas,
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
    model_pred = model_pred.squeeze(2)  # 5D to 4D, [B, C, 1, H, W] -> [B, C, H, W]

    # Note: do NOT clear timestep mask here -- gradient checkpointing recomputes the forward
    # pass during backward, so the mask must remain set. It gets overwritten on the next step.

    # Rectified flow target: noise - latents
    target = noise - latents

    observer_aux = trainer._state.extras_for_step.get("personalization_observer")
    if observer_probe and observer_aux is not None:
        try:
            update_observation(
                trainer._state.personalization_observer,
                args,
                timesteps=timesteps,
                adapter_pred=model_pred,
                base_pred=observer_aux["base_pred"],
                target=target,
            )
        except Exception as exc:
            record_error(trainer._state.personalization_observer, exc)

    # Loss weighting
    weighting = anima_train_utils.compute_loss_weighting_for_anima(
        weighting_scheme=args.weighting_scheme,
        sigmas=sigmas,
        min_snr_gamma=getattr(args, "min_snr_gamma", None),
        p2_gamma=getattr(args, "p2_gamma", 1.0),
        p2_k=getattr(args, "p2_k", 1.0),
    )
    dynamic_weight = dynamic_denoise_weights(
        trainer._state.personalization_observer, args, timesteps
    )
    if dynamic_weight is not None:
        dynamic_weight = dynamic_weight.reshape(
            dynamic_weight.shape[0], *([1] * (weighting.ndim - 1))
        )
        weighting = weighting * dynamic_weight.to(
            device=weighting.device, dtype=weighting.dtype
        )

    return model_pred, target, timesteps, weighting

def _is_train_text_encoder(trainer, args) -> bool:
    return trainer.is_train_text_encoder(args)
