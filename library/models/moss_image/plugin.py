"""Lazy MOSS-Image strategy plugin."""

from __future__ import annotations

from library.models.family_plugins import TextPreprocessSpec


class MossImagePlugin:
    family = "moss_image"

    text_preprocess_spec = TextPreprocessSpec(
        runner="special",
        module="scripts.moss_image.preprocess_te_cache",
    )

    def load_preprocess_vae(self, args, dtype):
        from library.models.moss_image.latent import encode_moss_latents
        from library.models.moss_image.weights import load_moss_vae

        return load_moss_vae(args.vae, dtype=dtype, device="cpu"), encode_moss_latents

    def compute_noise_pred_and_target(self, *args, **kwargs):
        from library.models.moss_image.family import compute_noise_pred_and_target

        return compute_noise_pred_and_target(*args, **kwargs)

    def build_tokenize_strategy(self, args):
        from library.models.moss_image.strategy import MossImageTokenizeStrategy

        return MossImageTokenizeStrategy(
            args.qwen3, int(getattr(args, "max_token_length", None) or 4096)
        )

    def build_text_encoding_strategy(self, args):
        from library.models.moss_image.strategy import MossImageTextEncodingStrategy

        return MossImageTextEncodingStrategy()

    def build_latent_cache_strategy(self, args):
        from library.models.moss_image.strategy import MossImageLatentCache
        from library import cache_identity

        return MossImageLatentCache(
            args.cache_latents_to_disk,
            args.vae_batch_size,
            args.skip_cache_check,
            cache_identity=cache_identity.latent_cache_identity(
                family=self.family, vae=getattr(args, "vae", None)
            ),
            require_cache_identity=bool(getattr(args, "require_cache_identity", False)),
        )

    def build_text_cache_strategy(self, args, weight_dtype):
        from library.models.moss_image.strategy import MossImageTextCache

        return MossImageTextCache(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            quantization=getattr(args, "moss_te_quantization", "nf4"),
            max_text_tokens=int(getattr(args, "max_token_length", None) or 4096),
            use_shuffled_caption_variants=getattr(
                args, "use_shuffled_caption_variants", False
            ),
        )


PLUGIN = MossImagePlugin()
