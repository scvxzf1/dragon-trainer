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
        from library.models.moss_image.weights import load_moss_vae

        return load_moss_vae(args.vae, dtype=dtype, device="cpu"), self.encode_vae

    def encode_vae(self, vae, pixels):
        from library.models.moss_image.latent import encode_moss_latents

        return encode_moss_latents(vae, pixels)

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
        from library.models.moss_image.family import uniform_grid_sigmas

        return uniform_grid_sigmas(batch_size, device=device)

    def prepare_conditions(self, batch, conditions, *, context, is_train):
        from library.models.moss_image.family import _prepare_text_conditions

        return _prepare_text_conditions(
            context.trainer, context.train_context, batch, conditions
        )

    def forward_velocity_4d(self, dit, noisy_4d, conditions, sigmas, *, context):
        from library.models.moss_image.family import forward_for_loss

        return forward_for_loss(dit, noisy_4d, conditions, sigmas)

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
            encode_by_vae=self.encode_vae,
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
