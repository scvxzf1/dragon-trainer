"""Single training/validation batch: latents, text conds, forward, loss compose."""

from __future__ import annotations

import typing

import torch

from library.training.losses import (
    LossContext,
    build_loss_composer,
    get_huber_threshold_if_needed,
)
from library.training import numerical_observation
from library.training.method_adapter import ComputeLossCtx
from library.training.contexts import TrainCtx
from library.models.family_registry import get_family_plugin


def _nonfinite_counts(tensor: torch.Tensor) -> dict[str, int]:
    """``{nan, inf, nonfinite, elements}`` in a single device→host sync."""
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    nan, inf = torch.stack(
        [torch.count_nonzero(nan_mask), torch.count_nonzero(inf_mask)]
    ).tolist()
    return {
        "nan": int(nan),
        "inf": int(inf),
        "nonfinite": int(nan) + int(inf),
        "elements": int(tensor.numel()),
    }


def _krea2_noise_pred_and_target(*args, **kwargs):
    from library.models.krea2_raw.family import compute_noise_pred_and_target

    return compute_noise_pred_and_target(*args, **kwargs)


def _z_image_noise_pred_and_target(*args, **kwargs):
    from library.models.z_image.family import compute_noise_pred_and_target

    return compute_noise_pred_and_target(*args, **kwargs)


def _moss_noise_pred_and_target(*args, **kwargs):
    from library.models.moss_image.family import compute_noise_pred_and_target

    return compute_noise_pred_and_target(*args, **kwargs)


def _qwen_image_2_1_noise_pred_and_target(*args, **kwargs):
    from library.models.qwen_image_2_1.family import compute_noise_pred_and_target

    return compute_noise_pred_and_target(*args, **kwargs)


def _get_noise_pred_and_target(trainer, ctx, latents, batch, text_encoder_conds, *, is_train=True):
    from library.env import resolve_model_family
    plugin = get_family_plugin(resolve_model_family(ctx.args), capability="flow")
    return plugin.compute_noise_pred_and_target(
        trainer, ctx, latents, batch, text_encoder_conds, is_train=is_train
    )


def _scrub_latents_nonfinite(
    latents: torch.Tensor,
    *,
    source: str,
    accelerator,
) -> torch.Tensor:
    counts = _nonfinite_counts(latents)
    if counts["nan"]:
        accelerator.print(f"NaN found in {source}, replacing with zeros")
        numerical_observation.record(
            numerical_observation.SITE_LATENTS,
            numerical_observation.KIND_SOURCE_NAN,
            nonfinite=counts["nan"],
            elements=counts["elements"],
            before_dtype=latents.dtype,
            after_dtype=latents.dtype,
            shape=tuple(latents.shape),
            detail=f"source={source}",
        )
        if counts["inf"]:
            numerical_observation.record(
                numerical_observation.SITE_LATENTS,
                numerical_observation.KIND_SOURCE_INF,
                nonfinite=counts["inf"],
                elements=counts["elements"],
                before_dtype=latents.dtype,
                after_dtype=latents.dtype,
                shape=tuple(latents.shape),
                detail=f"source={source}",
            )
        # Out of place: a cached ``batch["latents"]`` tensor already on device
        # would otherwise be rewritten through the dataloader's own storage.
        return torch.nan_to_num(latents, 0)
    if counts["inf"]:
        # Observation only: the established scrub triggers on NaN, so ±Inf is
        # counted and left in the tensor exactly as before (changing that
        # predicate is a separate user policy decision).
        numerical_observation.record(
            numerical_observation.SITE_LATENTS,
            numerical_observation.KIND_SOURCE_INF,
            nonfinite=counts["inf"],
            elements=counts["elements"],
            before_dtype=latents.dtype,
            after_dtype=latents.dtype,
            shape=tuple(latents.shape),
            detail=f"source={source} (Inf-only: NaN-gated scrub does not touch it)",
        )
    return latents


def _prepare_latents(args, trainer, vae, batch, accelerator, vae_dtype) -> torch.FloatTensor:
    """Load cached latents or encode images, then scrub non-finite values.

    The NaN scrub used to sit inside the VAE-encode branch only, so a corrupt
    cached sidecar went straight into the forward pass. It now applies to both
    sources, which is also what makes the cached path testable on its own.
    """
    with torch.no_grad():
        if "latents" in batch and batch["latents"] is not None:
            raw_latents = batch["latents"]
            source = "cached latents"
            if raw_latents.device.type == "cpu":
                # Scrub on CPU before async H2D so steady-state cached training
                # incurs zero GPU->CPU stream synchronization per batch.
                if not bool(torch.isfinite(raw_latents).all()):
                    raw_latents = _scrub_latents_nonfinite(
                        raw_latents, source=source, accelerator=accelerator
                    )
                latents = typing.cast(
                    torch.FloatTensor,
                    raw_latents.to(accelerator.device, non_blocking=True),
                )
                return typing.cast(
                    torch.FloatTensor, trainer.shift_scale_latents(args, latents)
                )
            latents = typing.cast(
                torch.FloatTensor,
                raw_latents.to(accelerator.device, non_blocking=True),
            )
        else:
            source = "VAE latents"
            if (
                args.vae_batch_size is None
                or len(batch["images"]) <= args.vae_batch_size
            ):
                latents = trainer.encode_images_to_latents(
                    args,
                    vae,
                    batch["images"].to(
                        accelerator.device, dtype=vae_dtype, non_blocking=True
                    ),
                )
            else:
                chunks = [
                    batch["images"][i : i + args.vae_batch_size]
                    for i in range(0, len(batch["images"]), args.vae_batch_size)
                ]
                list_latents = []
                for chunk in chunks:
                    with torch.no_grad():
                        chunk = trainer.encode_images_to_latents(
                            args,
                            vae,
                            chunk.to(
                                accelerator.device,
                                dtype=vae_dtype,
                                non_blocking=True,
                            ),
                        )
                        list_latents.append(chunk)
                latents = torch.cat(list_latents, dim=0)

        latents = typing.cast(
            torch.FloatTensor,
            _scrub_latents_nonfinite(
                latents, source=source, accelerator=accelerator
            ),
        )
        return typing.cast(torch.FloatTensor, trainer.shift_scale_latents(args, latents))


def process_batch_inner(
    trainer,
    ctx: TrainCtx,
    batch,
    *,
    is_train=True,
) -> torch.Tensor:
    """
    Process a batch for the network (original NetworkTrainer.process_batch logic)
    """
    args = ctx.args
    accelerator = ctx.accelerator
    network = ctx.network
    vae = ctx.vae
    text_encoders = ctx.text_encoders
    text_encoding_strategy = ctx.text_encoding_strategy
    tokenize_strategy = ctx.tokenize_strategy
    noise_scheduler = ctx.noise_scheduler
    vae_dtype = ctx.vae_dtype
    weight_dtype = ctx.weight_dtype
    train_text_encoder = ctx.train_text_encoder
    latents = _prepare_latents(args, trainer, vae, batch, accelerator, vae_dtype)

    text_encoder_conds = []
    text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
    if text_encoder_outputs_list is not None:
        text_encoder_conds = (
            text_encoder_outputs_list  # List of text encoder outputs
        )

    if (
        len(text_encoder_conds) == 0
        or text_encoder_conds[0] is None
        or train_text_encoder
    ):
        with (
            torch.set_grad_enabled(is_train and train_text_encoder),
            accelerator.autocast(),
        ):
            if args.weighted_captions:
                input_ids_list, weights_list = (
                    tokenize_strategy.tokenize_with_weights(batch["captions"])
                )
                encoded_text_encoder_conds = (
                    text_encoding_strategy.encode_tokens_with_weights(
                        tokenize_strategy,
                        trainer.get_models_for_text_encoding(
                            args, accelerator, text_encoders
                        ),
                        input_ids_list,
                        weights_list,
                    )
                )
            else:
                input_ids = [
                    ids.to(accelerator.device, non_blocking=True)
                    for ids in batch["input_ids_list"]
                ]
                encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                    tokenize_strategy,
                    trainer.get_models_for_text_encoding(
                        args, accelerator, text_encoders
                    ),
                    input_ids,
                )
            if args.full_fp16:
                encoded_text_encoder_conds = [
                    c.to(weight_dtype) for c in encoded_text_encoder_conds
                ]

        if len(text_encoder_conds) == 0:
            text_encoder_conds = encoded_text_encoder_conds
        else:
            for i in range(len(encoded_text_encoder_conds)):
                if encoded_text_encoder_conds[i] is not None:
                    text_encoder_conds[i] = encoded_text_encoder_conds[i]

    # Some methods own the entire objective instead of producing a standard
    # flow-matching prediction/target pair. BYG is the current user: it runs
    # bootstrap, prior, cycle and identity forwards inside its adapter.
    owners = [adapter for adapter in trainer._adapters if adapter.owns_training_step(args)]
    if owners:
        if len(owners) != 1:
            raise ValueError(
                "at most one adapter may own the training step; got "
                f"{[adapter.name for adapter in owners]}"
            )
        return owners[0].compute_loss(
            ComputeLossCtx(
                args=args,
                accelerator=accelerator,
                network=getattr(trainer, "_network", network),
                unet=ctx.unet,
                noise_scheduler=noise_scheduler,
                weight_dtype=weight_dtype,
                batch=batch,
                latents=latents,
                text_encoder_conds=text_encoder_conds,
                is_train=is_train,
            )
        )

    # sample noise, call unet, get target
    noise_pred, target, timesteps, weighting = _get_noise_pred_and_target(trainer,
        ctx,
        latents,
        batch,
        text_encoder_conds,
        is_train=is_train,
    )

    huber_c = get_huber_threshold_if_needed(args, timesteps, noise_scheduler)

    # Assemble aux dict for the composer: extra_forwards returns from each
    # method adapter plus the trainer-owned functional-loss capture.
    loss_aux: dict = dict(trainer._state.extras_for_step)

    func_loss = getattr(trainer, "_func_loss", None)
    if func_loss is not None:
        loss_aux["func_loss"] = func_loss

    net_for_loss = getattr(trainer, "_network", network)
    composer_sig = (
        id(args),
        id(net_for_loss),
        float(getattr(args, "vr_loss_weight", 0.0) or 0.0),
        float(getattr(args, "velocity_direction_loss_weight", 0.0) or 0.0),
        float(getattr(args, "prior_preservation_weight", 0.0) or 0.0),
        float(getattr(args, "inverted_mask_prior_weight", 0.0) or 0.0),
        float(getattr(args, "functional_loss_weight", 0.0) or 0.0),
        float(getattr(args, "multiscale_loss_weight", 0.0) or 0.0),
        float(getattr(net_for_loss, "_ortho_reg_weight", 0.0) or 0.0),
        float(getattr(net_for_loss, "_balance_loss_weight", 0.0) or 0.0),
        bool(getattr(net_for_loss, "_use_chimera_hydra", False)),
        float(getattr(net_for_loss, "fecl_weight", 0.0) or 0.0),
        getattr(getattr(net_for_loss, "cfg", None), "use_moe_style", False),
        float(getattr(net_for_loss, "_contrastive_target_weight", 0.0) or 0.0),
    )
    cached_composer = getattr(trainer, "_cached_loss_composer", None)
    if cached_composer is not None and cached_composer[0] == composer_sig:
        composer = cached_composer[1]
    else:
        composer = build_loss_composer(args, net_for_loss)
        try:
            trainer._cached_loss_composer = (composer_sig, composer)
        except Exception:
            pass

    def _build_loss_ctx(aux: dict) -> LossContext:
        return LossContext(
            args=args,
            batch=batch,
            model_pred=noise_pred,
            target=target,
            timesteps=timesteps,
            weighting=weighting,
            huber_c=huber_c,
            loss_weights=batch["loss_weights"],
            network=getattr(trainer, "_network", network),
            aux=aux,
            is_train=is_train,
        )

    return composer.compose(_build_loss_ctx(loss_aux))
