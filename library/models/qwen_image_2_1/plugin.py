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

        return sample_sigmas(
            batch_size, height * width, device=device,
        )

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
