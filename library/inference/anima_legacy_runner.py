"""Plugin facade for the existing Anima inference implementation."""

from __future__ import annotations

from library.inference.family_output import save_with_shared_qwen_vae
from library.inference.family_runner import FamilyInferenceRunner
from library.runtime.device import clean_memory_on_device


class AnimaLegacyInferenceRunner:
    family = "anima"
    supports_reference_conditioning = False

    @staticmethod
    def validate(args, *, mode: str = "single") -> None:
        return None

    @staticmethod
    def prepare(args, *, mode: str = "single") -> None:
        from library.anima import strategy as strategy_anima
        from library.anima import text_strategies
        from library.inference.text import MAX_CROSSATTN_TOKENS

        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.text_encoder,
            t5_tokenizer_path=None,
            qwen3_max_length=MAX_CROSSATTN_TOKENS,
            t5_max_length=MAX_CROSSATTN_TOKENS,
        )
        text_strategies.TokenizeStrategy.set_strategy(tokenize_strategy)
        encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
        text_strategies.TextEncodingStrategy.set_strategy(encoding_strategy)

    def generate_latent(self, args, gen_settings):
        from library.inference.generation import generate

        runtime = FamilyInferenceRunner(
            self.family,
            gen_settings.device,
            cleanup=clean_memory_on_device,
        )
        return runtime.run_stage(
            "legacy_generation",
            lambda: None,
            lambda _: generate(args, gen_settings),
        )

    def save_output(self, args, latent, device):
        return save_with_shared_qwen_vae(
            args,
            latent,
            device,
            family=self.family,
        )
