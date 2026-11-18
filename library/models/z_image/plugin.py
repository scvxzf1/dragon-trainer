"""Lazy Z-Image strategy plugin."""

from __future__ import annotations

from library.models.family_plugins import TextPreprocessSpec


class ZImagePlugin:
    family = "z_image"

    text_preprocess_spec = TextPreprocessSpec(
        runner="module",
        module="scripts.z_image.preprocess_te_cache",
        qwen3_config_key="qwen3",
        qwen3_default="models/diffusion_models/Z-Image",
    )

    def load_preprocess_vae(self, args, dtype):
        from library.models.z_image.latent import encode_z_image_latents
        from library.models.z_image.weights import load_z_image_vae

        return load_z_image_vae(args.vae, dtype=dtype, device="cpu"), encode_z_image_latents

    def compute_noise_pred_and_target(self, *args, **kwargs):
        from library.models.z_image.family import compute_noise_pred_and_target

        return compute_noise_pred_and_target(*args, **kwargs)

    def build_tokenize_strategy(self, args):
        from library.models.z_image.strategy import ZImageTokenizeStrategy

        return ZImageTokenizeStrategy(args.qwen3)

    def build_text_encoding_strategy(self, args):
        from library.models.z_image.strategy import ZImageTextEncodingStrategy

        return ZImageTextEncodingStrategy()

    def build_latent_cache_strategy(self, args):
        from library.models.z_image.strategy import ZImageLatentsCachingStrategy
        from library import cache_identity

        return ZImageLatentsCachingStrategy(
            args.cache_latents_to_disk,
            args.vae_batch_size,
            args.skip_cache_check,
            cache_identity=cache_identity.latent_cache_identity(
                family=self.family, vae=getattr(args, "vae", None)
            ),
            require_cache_identity=bool(getattr(args, "require_cache_identity", False)),
        )

    def build_text_cache_strategy(self, args, weight_dtype):
        from library.models.z_image.strategy import ZImageTextEncoderOutputsCachingStrategy

        return ZImageTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            use_shuffled_caption_variants=getattr(
                args, "use_shuffled_caption_variants", False
            ),
        )


PLUGIN = ZImagePlugin()
