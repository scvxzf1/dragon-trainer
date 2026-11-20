"""Lazy Qwen Image 2.1 strategy plugin."""

from __future__ import annotations

from library.models.family_plugins import TextPreprocessSpec


class QwenImage21Plugin:
    family = "qwen_image_2_1"

    text_preprocess_spec = TextPreprocessSpec(
        runner="special",
        module="scripts.qwen_image_2_1.preprocess_te_cache",
        qwen3_config_key="qwen3",
        qwen3_default="models/text_encoders/Qwen-Image-2.1",
        supports_tag_dropout=False,
    )

    def load_preprocess_vae(self, args, dtype):
        from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_vae

        return (
            load_qwen_image_2_1_vae(args.vae, dtype=dtype, device="cpu"),
            self.encode_vae,
        )

    def encode_vae(self, vae, pixels):
        from library.models.qwen_image_2_1.latent import encode_qwen_image_2_1_latents

        return encode_qwen_image_2_1_latents(vae, pixels)

    def compute_noise_pred_and_target(
        self, trainer, ctx, latents, batch, text_encoder_conds, *, is_train=True
    ):
        from library.training.flow_step import run_family_flow_step

        return run_family_flow_step(
            trainer,
            ctx,
            self,
            latents,
            batch,
            text_encoder_conds,
            is_train=is_train,
        )

    def sample_sigmas(self, batch_size, height, width, args, device, *, context):
        from library.models.qwen_image_2_1.family import sample_sigmas

        return sample_sigmas(batch_size, height * width, device=device)

    def prepare_conditions(self, batch, conditions, *, context, is_train):
        from library.models.qwen_image_2_1.family import prepare_training_conditions

        return prepare_training_conditions(
            context.trainer, context.train_context, batch, conditions
        )

    def forward_velocity_4d(self, dit, noisy_4d, conditions, sigmas, *, context):
        from library.models.qwen_image_2_1.family import forward_for_loss

        return forward_for_loss(
            dit,
            noisy_4d,
            conditions.hiddens,
            conditions.mask,
            sigmas,
            reference_latents=conditions.reference_latents,
            image_slot_mask=conditions.image_slot_mask,
        )

    def build_inference_runner(self, args):
        from library.models.qwen_image_2_1.inference_runner import QwenImage21InferenceRunner

        return QwenImage21InferenceRunner()

    def cache_preview_text(self, trainer, args, accelerator, model, tokenizer, encoder):
        import torch

        from library.models.qwen_image_2_1.preview_conditions import cache_preview_text
        from library.models.qwen_image_2_1.text_encoder_runtime import loaded_text_encoder_for_cache
        from library.strategies.text_cache import text_encoder_cache_session

        policy = getattr(args, "qwen_text_encoder_cache_policy", "auto")

        def manage(encoder_model, device, dtype):
            return loaded_text_encoder_for_cache(
                encoder_model, dtype=dtype, device=str(device), cache_policy=policy,
            )

        target = "cpu" if policy == "cpu" else accelerator.device
        with text_encoder_cache_session(
            model, target, dtype=next(model.parameters()).dtype, manage=manage,
        ) as prepared:
            with accelerator.autocast(), torch.no_grad():
                cache_preview_text(trainer, args, prepared, tokenizer, encoder)

    def decode_preview_samples(
        self,
        accelerator,
        args,
        vae,
        *,
        dit=None,
        network=None,
    ):
        from library.models.qwen_image_2_1.preview_decode import decode_pending_samples
        from library.training.preview_live_decode import decode_live_preview

        return decode_live_preview(
            accelerator,
            args,
            vae,
            dit=dit,
            network=network,
            decode_pending_samples=decode_pending_samples,
        )

    def decode_pending_samples(self, accelerator, args, vae):
        from library.models.qwen_image_2_1.preview_decode import decode_pending_samples

        return decode_pending_samples(accelerator, args, vae)

    def build_tokenize_strategy(self, args):
        from library.models.qwen_image_2_1.strategy import (
            QwenImage21EditTokenizeStrategy,
            QwenImage21TokenizeStrategy,
        )

        if getattr(args, "qwen_image_2_1_task", "t2i") == "edit":
            return QwenImage21EditTokenizeStrategy(args.qwen3)
        return QwenImage21TokenizeStrategy(args.qwen3)

    def build_text_encoding_strategy(self, args):
        from library.models.qwen_image_2_1.strategy import QwenImage21TextEncodingStrategy

        return QwenImage21TextEncodingStrategy()

    def build_latent_cache_strategy(self, args):
        from library.models.qwen_image_2_1.strategy import QwenImage21LatentCache
        from library import cache_identity

        return QwenImage21LatentCache(
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
        from library.models.qwen_image_2_1.strategy import (
            QwenImage21EditTextCache,
            QwenImage21TextCache,
        )

        cache_type = (
            QwenImage21EditTextCache
            if getattr(args, "qwen_image_2_1_task", "t2i") == "edit"
            else QwenImage21TextCache
        )
        return cache_type(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            use_shuffled_caption_variants=getattr(
                args, "use_shuffled_caption_variants", False
            ),
        )


PLUGIN = QwenImage21Plugin()
