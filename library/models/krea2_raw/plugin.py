"""Lazy Krea-2 strategy plugin."""

from __future__ import annotations

from library.models.family_plugins import TextPreprocessSpec


class Krea2Plugin:
    family = "krea2_raw"

    text_preprocess_spec = TextPreprocessSpec(
        runner="module",
        module="scripts.krea2.preprocess_te_cache",
        qwen3_config_key="qwen3",
        qwen3_default="models/text_encoders/qwen3vl_4b_bf16.safetensors",
        default_shuffle_variants="4",
        default_tag_dropout_rate="0.1",
        strip_diff_output_preservation=True,
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
        return vae, None

    def compute_noise_pred_and_target(self, *args, **kwargs):
        from library.models.krea2_raw.family import compute_noise_pred_and_target

        return compute_noise_pred_and_target(*args, **kwargs)

    def build_tokenize_strategy(self, args):
        from library.models.krea2_raw.strategy import Krea2TokenizeStrategy

        return Krea2TokenizeStrategy()

    def build_text_encoding_strategy(self, args):
        from library.models.krea2_raw.strategy import Krea2TextEncodingStrategy

        return Krea2TextEncodingStrategy()

    def build_latent_cache_strategy(self, args):
        from library.anima.strategy import AnimaLatentsCachingStrategy
        from library import cache_identity

        return AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk,
            args.vae_batch_size,
            args.skip_cache_check,
            cache_identity=cache_identity.latent_cache_identity(
                family=self.family, vae=getattr(args, "vae", None)
            ),
            require_cache_identity=bool(getattr(args, "require_cache_identity", False)),
        )

    def build_text_cache_strategy(self, args, weight_dtype):
        from library.models.krea2_raw.strategy import (
            Krea2TextEncoderOutputsCachingStrategy,
        )

        return Krea2TextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            use_shuffled_caption_variants=getattr(
                args, "use_shuffled_caption_variants", False
            ),
        )


PLUGIN = Krea2Plugin()
