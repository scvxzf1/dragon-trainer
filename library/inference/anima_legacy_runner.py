"""Plugin facade for the existing Anima inference implementation."""

from __future__ import annotations

from library.inference.family_output import save_with_shared_qwen_vae
from library.inference.family_runner import FamilyInferenceRunner
from library.runtime.device import clean_memory_on_device


class AnimaLegacyInferenceRunner:
    family = "anima"
    supports_reference_conditioning = True

    @staticmethod
    def validate(args, *, mode: str = "single") -> None:
        from library.inference.family_runtime import validate_family_inference_options

        validate_family_inference_options(args, "anima")
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
        import torch

        from library.anima.inference_references import (
            easycontrol_reference, encode_easycontrol_reference, load_easycontrol_vae,
            select_easycontrol_size,
        )
        from library.inference.generation import generate
        from library.inference.models import load_dit_model, load_text_encoder
        from library.inference.precision import resolve_text_encoder_dtype

        runtime = FamilyInferenceRunner(
            self.family,
            gen_settings.device,
            cleanup=clean_memory_on_device,
        )
        text_device = torch.device("cpu") if getattr(args, "text_encoder_cpu", False) else gen_settings.device
        text_runtime = FamilyInferenceRunner(self.family, text_device, cleanup=clean_memory_on_device)
        encoded = text_runtime.run_text(
            lambda: load_text_encoder(args, dtype=resolve_text_encoder_dtype(args), device=text_device),
            lambda encoder: self._encode_prompts(args, encoder),
        )
        reference = easycontrol_reference(args)
        reference_latent = None
        if reference is not None:
            device, dtype = gen_settings.device, gen_settings.runtime_dtype
            select_easycontrol_size(args, reference)
            reference_latent = runtime.run_stage(
                "reference_vae", lambda: load_easycontrol_vae(args, device, dtype),
                lambda vae: encode_easycontrol_reference(args, reference, vae, device, dtype).cpu(),
            )

        def denoise(model):
            return generate(
                args,
                gen_settings,
                shared_models={"model": model, "easycontrol_latent": reference_latent},
                precomputed_text_data=self._prepare_prompt_data(model, encoded),
            )

        return runtime.run_stage(
            "dit", lambda: load_dit_model(args, gen_settings.device, gen_settings.runtime_dtype), denoise,
        )

    @staticmethod
    def _encode_prompts(args, encoder):
        from library.inference.text import encode_anima_prompt, ensure_text_strategies, process_escape

        ensure_text_strategies(args.text_encoder)
        encoder.eval()
        prompts = {"context": args.prompt, "context_null": args.negative_prompt}
        if getattr(args, "pooled_text_proj", None) and getattr(args, "mod_w", 0.0) != 0.0:
            prompts.update(mod_pos=args.mod_pos_prompt, mod_neg=args.mod_neg_prompt)
        by_prompt = {}
        encoded = {}
        for key, prompt in prompts.items():
            prompt = process_escape(prompt)
            if prompt not in by_prompt:
                by_prompt[prompt] = tuple(value.cpu() for value in encode_anima_prompt(prompt, encoder))
            encoded[key] = (prompt, by_prompt[prompt])
        return encoded

    @staticmethod
    def _prepare_prompt_data(model, encoded):
        from library.inference.text import prepare_anima_prompt_context

        prepared = {}
        for key in ("context", "context_null"):
            prompt, embedding = encoded[key]
            crossattn = prepare_anima_prompt_context(model, embedding).cpu()
            prepared[key] = {"prompt": prompt, "embed": (crossattn, *embedding[1:])}
        if "mod_pos" in encoded:
            prepared["mod_guidance"] = tuple(
                prepare_anima_prompt_context(model, encoded[key][1]) for key in ("mod_pos", "mod_neg")
            )
        return prepared

    def save_output(self, args, latent, device):
        return save_with_shared_qwen_vae(
            args,
            latent,
            device,
            family=self.family,
        )
