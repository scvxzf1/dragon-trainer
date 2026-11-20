"""Standalone Z-Image inference using the family preview contract."""

from __future__ import annotations

import torch

from library.inference.family_runner import FamilyInferenceRunner
from library.inference.family_output import save_pil_image
from library.inference.output import _build_png_info
from library.models.z_image.attention_backend import prepare_z_image_attention
from library.models.z_image.strategy import ZImageTextEncodingStrategy, ZImageTokenizeStrategy
from library.models.z_image.training_preview import generate_z_image
from library.models.z_image.weights import (
    load_z_image_text_encoder,
    load_z_image_transformer,
    load_z_image_vae,
)
from library.training.preview_decode import decode_z_image_latent
from library.inference.precision import resolve_runtime_dtype, resolve_text_encoder_dtype
from library.inference.family_runtime import validate_family_inference_options
from library.runtime.device import clean_memory_on_device


class ZImageInferenceRunner:
    family = "z_image"
    supports_reference_conditioning = False

    @staticmethod
    def validate(args, *, mode: str = "single") -> None:
        validate_family_inference_options(args, "z_image")
        if mode != "single":
            raise SystemExit("Z-Image standalone inference supports only single-prompt mode.")
        if str(getattr(args, "sampler", "euler") or "euler").lower() != "euler":
            raise SystemExit("Z-Image standalone inference supports only the Euler sampler.")
        height, width = (int(value) for value in args.image_size)
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("Z-Image image_size must be positive and divisible by 16")
        if getattr(args, "lora_weight", None):
            raise SystemExit("Z-Image standalone inference currently does not support LoRA.")

    @staticmethod
    def prepare(args, *, mode: str = "single") -> None:
        return None

    @staticmethod
    def _load_text_encoder(args):
        return load_z_image_text_encoder(
            args.text_encoder,
            dtype=resolve_text_encoder_dtype(args),
            device="cpu",
        )

    @staticmethod
    def _encode_prompt(encoder, args, device, dtype, prompt: str):
        tokenizer = ZImageTokenizeStrategy(args.text_encoder)
        strategy = ZImageTextEncodingStrategy()
        values = strategy.encode_tokens(
            tokenizer,
            [encoder],
            tokenizer.tokenize(prompt),
        )
        hiddens, mask = values
        return [hiddens.to(device=device, dtype=dtype)[0][mask[0].bool()]]

    def _encode_conditions(self, encoder, args, device, dtype):
        cond = self._encode_prompt(
            encoder,
            args,
            device,
            dtype,
            args.prompt or "",
        )
        uncond = None
        if float(args.guidance_scale) > 0:
            uncond = self._encode_prompt(
                encoder,
                args,
                device,
                dtype,
                getattr(args, "negative_prompt", "") or "",
            )
        return cond, uncond

    def generate_latent(self, args, gen_settings):
        self.validate(args)
        device = gen_settings.device
        dtype = gen_settings.runtime_dtype
        runtime = FamilyInferenceRunner(
            self.family,
            device,
            cleanup=clean_memory_on_device,
        )
        cond, uncond = runtime.run_text(
            lambda: self._load_text_encoder(args),
            lambda encoder: self._encode_conditions(
                encoder,
                args,
                device,
                dtype,
            ),
        )
        return runtime.run_stage(
            "dit",
            lambda: load_z_image_transformer(
                args.dit,
                dtype=dtype,
                device=device,
            ),
            lambda dit: (
                prepare_z_image_attention(
                    dit,
                    getattr(args, "attn_mode", "torch"),
                    dtype=dtype,
                ),
                generate_z_image(
                    dit,
                    cond,
                    uncond,
                    height=int(args.image_size[0]),
                    width=int(args.image_size[1]),
                    sample_steps=int(args.infer_steps),
                    guidance_scale=float(args.guidance_scale),
                    flow_shift=float(
                        getattr(args, "flow_shift", 6.0) or 6.0
                    ),
                    seed=int(args.seed),
                    device=device,
                    dtype=dtype,
                ),
            )[1],
        )

    def save_output(self, args, latent, device):
        if args.output_type in {"latent", "latent_images"}:
            from library.inference.output import save_latent

            save_latent(latent, args, int(args.image_size[0]), int(args.image_size[1]))
        if args.output_type == "latent":
            return
        runtime = FamilyInferenceRunner(
            self.family,
            device,
            cleanup=clean_memory_on_device,
        )
        runtime.run_stage(
            "vae",
            lambda: load_z_image_vae(args.vae, dtype=resolve_runtime_dtype(args), device=device),
            lambda vae: save_pil_image(
                args,
                decode_z_image_latent(vae, latent),
                seed=int(args.seed),
                png_info_factory=_build_png_info,
                prefix="z_image",
            ),
        )
