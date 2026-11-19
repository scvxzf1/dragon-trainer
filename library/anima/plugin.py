"""Lazy Anima strategy plugin."""

from __future__ import annotations

from library.models.family_plugins import TextPreprocessSpec


class AnimaPlugin:
    family = "anima"

    text_preprocess_spec = TextPreprocessSpec(
        runner="module",
        module="scripts.preprocess.cache_text_embeddings",
        qwen3_config_key="qwen3",
        qwen3_default="models/text_encoders/qwen_3_06b_base.safetensors",
        dit_config_key="pretrained_model_name_or_path",
        dit_default="models/diffusion_models/anima-base-v1.0.safetensors",
        default_shuffle_variants="4",
        default_tag_dropout_rate="0.1",
        supports_diff_output_preservation=True,
    )

    def load_preprocess_vae(self, args, dtype):
        from library.models import qwen_vae as qwen_image_autoencoder_kl

        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=getattr(args, "chunk_size", 64),
            disable_cache=getattr(args, "disable_cache", True),
        )
        return vae, self.encode_vae

    def encode_vae(self, vae, pixels):
        return vae.encode_pixels_to_latents(pixels)

    def compute_noise_pred_and_target(self, *args, **kwargs):
        from library.anima.noise_target import compute_noise_pred_and_target

        return compute_noise_pred_and_target(*args, **kwargs)

    def sample_sigmas(self, batch_size, height, width, args, device, *, context):
        from library.training.samplers import SAMPLER_REGISTRY, SamplerContext

        train_context = context.train_context
        sampler_fn = SAMPLER_REGISTRY[getattr(args, "sampler", "default") or "default"]
        sampler_out = sampler_fn(
            SamplerContext(
                args=args,
                noise_scheduler=train_context.noise_scheduler,
                latents=context.latents,
                noise=context.noise,
                device=device,
                weight_dtype=train_context.weight_dtype,
            )
        )
        context.noisy_input = sampler_out.noisy_input
        context.timesteps = sampler_out.timesteps.reshape(-1)
        return sampler_out.sigmas.reshape(-1)

    def prepare_conditions(self, batch, conditions, *, context, is_train):
        from library.training.forward.text_conds import prepare_text_conds

        del is_train
        train_context = context.train_context
        return prepare_text_conds(
            text_encoder_conds=conditions,
            batch=batch,
            text_encoding_strategy=train_context.text_encoding_strategy,
            network=train_context.network,
            device=train_context.accelerator.device,
            weight_dtype=train_context.weight_dtype,
            uncond_crossattn_emb=getattr(
                getattr(context.trainer, "_state", None), "uncond_crossattn_1", None
            ),
        )

    def forward_velocity_4d(self, dit, noisy_4d, conditions, sigmas, *, context):
        import torch

        from library.training.forward import build_forward_kwargs

        train_context = context.train_context
        timesteps = context.timesteps if context.timesteps is not None else sigmas
        noisy_5d = noisy_4d.unsqueeze(2)
        padding_mask = torch.zeros(
            noisy_4d.shape[0], 1, noisy_4d.shape[-2], noisy_4d.shape[-1],
            dtype=train_context.weight_dtype,
            device=noisy_4d.device,
        )
        if conditions.crossattn_emb is None:
            prediction = dit(
                noisy_5d,
                timesteps,
                conditions.prompt_embeds,
                padding_mask=padding_mask,
                target_input_ids=conditions.t5_input_ids,
                target_attention_mask=conditions.t5_attn_mask,
                source_attention_mask=conditions.attn_mask,
            )
        else:
            forward = build_forward_kwargs(
                network=train_context.network,
                crossattn_emb=conditions.crossattn_emb,
                t5_attn_mask=conditions.t5_attn_mask,
                timesteps=timesteps,
            )
            prediction = dit(
                noisy_5d,
                timesteps,
                forward.crossattn_emb,
                padding_mask=padding_mask,
                **forward.kw,
            )
        return prediction.squeeze(2)

    def compute_loss_weighting(self, sigmas, latents, args):
        from library.training.flow_step import loss_weighting

        return loss_weighting(sigmas, latents, args)

    def build_tokenize_strategy(self, args):
        from library.anima.strategy import AnimaTokenizeStrategy

        return AnimaTokenizeStrategy(
            qwen3_path=args.qwen3,
            t5_tokenizer_path=args.t5_tokenizer_path,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )

    def build_text_encoding_strategy(self, args):
        from library.anima.strategy import AnimaTextEncodingStrategy

        return AnimaTextEncodingStrategy()

    def build_latent_cache_strategy(self, args):
        from library.anima.strategy import AnimaLatentsCachingStrategy
        from library import cache_identity

        return AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk,
            args.vae_batch_size,
            args.skip_cache_check,
            encode_by_vae=self.encode_vae,
            cache_identity=cache_identity.latent_cache_identity(
                family=self.family, vae=getattr(args, "vae", None)
            ),
            require_cache_identity=bool(getattr(args, "require_cache_identity", False)),
        )

    def build_text_cache_strategy(self, args, weight_dtype):
        from library.anima.strategy import AnimaTextEncoderOutputsCachingStrategy

        return AnimaTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            False,
            cache_llm_adapter_outputs=getattr(args, "cache_llm_adapter_outputs", False),
            use_shuffled_caption_variants=getattr(
                args, "use_shuffled_caption_variants", False
            ),
            diff_output_preservation_trigger=getattr(
                args, "diff_output_preservation_trigger", None
            ),
            diff_output_preservation_class=getattr(
                args, "diff_output_preservation_class", None
            ),
            cache_dtype=weight_dtype,
            identity_inputs={
                "text_encoder": getattr(args, "qwen3", None),
                "t5_tokenizer": getattr(args, "t5_tokenizer_path", None),
                "vocab_pack": getattr(args, "vocab_pack_path", None),
                "dit": getattr(args, "pretrained_model_name_or_path", None),
            },
            require_cache_identity=(
                bool(getattr(args, "require_cache_identity", False))
                or bool(getattr(args, "vocab_pack_path", None))
            ),
        )


PLUGIN = AnimaPlugin()
