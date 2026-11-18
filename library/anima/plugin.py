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
        return vae, None

    def compute_noise_pred_and_target(self, *args, **kwargs):
        from library.training.noise_target import compute_noise_pred_and_target

        return compute_noise_pred_and_target(*args, **kwargs)

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
