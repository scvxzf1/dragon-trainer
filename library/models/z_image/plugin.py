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
        from library.models.z_image.weights import load_z_image_vae

        return load_z_image_vae(args.vae, dtype=dtype, device="cpu"), self.encode_vae

    def encode_vae(self, vae, pixels):
        from library.models.z_image.latent import encode_z_image_latents

        return encode_z_image_latents(vae, pixels)

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
        import torch

        from library.models.z_image.family import shifted_uniform_sigmas
        from library.models.family_registry import get_model_family_spec

        spec = get_model_family_spec(self.family)
        return shifted_uniform_sigmas(
            batch_size,
            device=device,
            dtype=torch.float32,
            shift=float(getattr(args, "discrete_flow_shift", spec.image_test_flow_shift_default)
                        or spec.image_test_flow_shift_default),
        )

    def prepare_conditions(self, batch, conditions, *, context, is_train):
        from library.models.z_image.family import _prepare_text_conditions

        return _prepare_text_conditions(
            context.trainer,
            context.train_context,
            batch,
            conditions,
            is_train=is_train,
        )

    def forward_velocity_4d(self, dit, noisy_4d, conditions, sigmas, *, context):
        from library.models.z_image.family import forward_for_loss

        return forward_for_loss(dit, noisy_4d.unsqueeze(2), conditions, sigmas).squeeze(2)

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
            encode_by_vae=self.encode_vae,
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
