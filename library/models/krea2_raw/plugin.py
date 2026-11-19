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
        return vae, self.encode_vae

    def encode_vae(self, vae, pixels):
        return vae.encode_pixels_to_latents(pixels)

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
        from library.models.krea2_raw.family import sample_krea2_sigmas

        return sample_krea2_sigmas(context)

    def prepare_conditions(self, batch, conditions, *, context, is_train):
        from library.models.krea2_raw.family import prepare_krea2_conditions

        return prepare_krea2_conditions(
            context.trainer, context.train_context, batch, conditions
        )

    def forward_velocity_4d(self, dit, noisy_4d, conditions, sigmas, *, context):
        from library.models.krea2_raw.family import forward_krea2_velocity

        return forward_krea2_velocity(dit, noisy_4d, conditions, sigmas, context)

    def compute_loss_weighting(self, sigmas, latents, args):
        from library.training.flow_step import loss_weighting

        return loss_weighting(sigmas, latents, args)

    def build_inference_runner(self, args):
        from library.models.krea2_raw.inference_runner import Krea2InferenceRunner

        return Krea2InferenceRunner()

    def decode_preview_samples(
        self,
        accelerator,
        args,
        vae,
        *,
        dit=None,
        network=None,
    ):
        from library.models.krea2_raw.training_preview import decode_pending_samples
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
        from library.models.krea2_raw.training_preview import decode_pending_samples

        return decode_pending_samples(accelerator, args, vae)

    def build_tokenize_strategy(self, args):
        from library.models.krea2_raw.strategy import Krea2TokenizeStrategy

        return Krea2TokenizeStrategy()

    def build_text_encoding_strategy(self, args):
        from library.models.krea2_raw.strategy import Krea2TextEncodingStrategy

        return Krea2TextEncodingStrategy()

    def build_latent_cache_strategy(self, args):
        from library.strategies.latent_cache import FamilyLatentCacheStrategy
        from library import cache_identity

        return FamilyLatentCacheStrategy(
            args.cache_latents_to_disk,
            args.vae_batch_size,
            args.skip_cache_check,
            model_family=self.family,
            encode_by_vae=self.encode_vae,
            clean_after_cache=True,
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
