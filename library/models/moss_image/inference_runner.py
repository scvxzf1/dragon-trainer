"""Standalone MOSS-Image inference using the training family contracts."""

from __future__ import annotations

import torch

from library.inference.family_runner import FamilyInferenceRunner
from library.inference.family_output import save_pil_image
from library.inference.output import _build_png_info
from library.inference.precision import resolve_runtime_dtype, resolve_text_encoder_dtype
from library.inference.family_runtime import validate_family_inference_options
from library.models.moss_image.sampling import generate_moss_image
from library.models.moss_image.strategy import MossImageTextEncodingStrategy, MossImageTokenizeStrategy
from library.models.moss_image.text_encoder_runtime import load_moss_text_encoder
from library.models.moss_image.training_preview import decode_moss_latent
from library.models.moss_image.weights import load_moss_transformer, load_moss_vae
from library.runtime.device import clean_memory_on_device


class MossImageInferenceRunner:
    family = "moss_image"
    supports_reference_conditioning = False

    @staticmethod
    def validate(args, *, mode: str = "single") -> None:
        validate_family_inference_options(args, "moss_image")
        if mode != "single":
            raise SystemExit("MOSS-Image standalone inference supports only single-prompt mode.")
        if str(getattr(args, "sampler", "euler") or "euler").lower() != "euler":
            raise SystemExit("MOSS-Image standalone inference supports only the Euler sampler.")
        if getattr(args, "lora_weight", None):
            raise SystemExit("MOSS-Image standalone inference currently does not support LoRA.")
        height, width = (int(value) for value in args.image_size)
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("MOSS-Image image_size must be positive and divisible by 16")
        configured_device = getattr(args, "device", None)
        if configured_device is not None:
            MossImageInferenceRunner._validate_text_encoder_config(
                args,
                MossImageInferenceRunner._resolve_text_encoder_device(
                    args,
                    configured_device,
                ),
            )

    @staticmethod
    def prepare(args, *, mode: str = "single") -> None:
        return None

    @staticmethod
    def _resolve_text_encoder_device(args, fallback_device):
        if getattr(args, "text_encoder_cpu", False):
            return torch.device("cpu")
        return torch.device(fallback_device)

    @staticmethod
    def _validate_text_encoder_config(args, text_encoder_device):
        quantization = str(
            getattr(args, "moss_te_quantization", "nf4") or "nf4"
        ).lower()
        if quantization not in {"nf4", "bf16"}:
            raise ValueError(
                "MOSS-Image text encoder quantization must be 'nf4' or 'bf16'"
            )
        if resolve_text_encoder_dtype(args) != torch.bfloat16:
            raise SystemExit(
                "MOSS-Image text encoding requires BF16 compute; "
                "set --text_encoder_dtype bf16."
            )
        if quantization == "nf4" and text_encoder_device.type != "cuda":
            raise SystemExit(
                "MOSS-Image NF4 text encoding requires a CUDA text-encoder "
                "stage; use --moss_te_quantization bf16 for CPU inference."
            )

    @staticmethod
    def _load_text_encoder(args, text_encoder_device):
        return load_moss_text_encoder(
            args.text_encoder,
            dtype=resolve_text_encoder_dtype(args),
            device=text_encoder_device,
            quantization=getattr(args, "moss_te_quantization", "nf4"),
        )

    @staticmethod
    def _encode_prompt(encoder, args, device, dtype, prompt: str):
        tokenizer = MossImageTokenizeStrategy(
            args.text_encoder,
            int(getattr(args, "max_token_length", None) or 4096),
        )
        values = MossImageTextEncodingStrategy().encode_tokens(
            tokenizer,
            [encoder],
            tokenizer.tokenize(prompt),
        )
        return (
            values[0].to(device=device, dtype=dtype),
            values[1].to(device=device),
        )

    def _encode_conditions(self, encoder, args, device, dtype):
        cond = self._encode_prompt(
            encoder,
            args,
            device,
            dtype,
            args.prompt or "",
        )
        uncond = None
        if float(args.guidance_scale) != 1.0:
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
        text_encoder_device = self._resolve_text_encoder_device(args, device)
        self._validate_text_encoder_config(args, text_encoder_device)
        runtime = FamilyInferenceRunner(
            self.family,
            device,
            cleanup=clean_memory_on_device,
        )
        cond, uncond = runtime.run_text(
            lambda: self._load_text_encoder(args, text_encoder_device),
            lambda encoder: self._encode_conditions(
                encoder,
                args,
                device,
                dtype,
            ),
        )
        return runtime.run_stage(
            "dit",
            lambda: load_moss_transformer(
                args.dit,
                dtype=dtype,
                device=device,
            ),
            lambda dit: generate_moss_image(
                dit,
                cond,
                uncond,
                height=int(args.image_size[0]),
                width=int(args.image_size[1]),
                sample_steps=int(args.infer_steps),
                guidance_scale=float(args.guidance_scale),
                flow_shift=float(
                    getattr(args, "flow_shift", 5.0) or 5.0
                ),
                seed=int(args.seed),
                device=device,
                dtype=dtype,
                uncond_skip_layers=tuple(
                    getattr(args, "uncond_skip_layers", (22,))
                ),
            ),
        )

    def save_output(self, args, latent, device):
        if args.output_type in {"latent", "latent_images"}:
            from library.inference.output import save_latent

            save_latent(
                latent,
                args,
                int(args.image_size[0]),
                int(args.image_size[1]),
            )
        if args.output_type == "latent":
            return
        runtime = FamilyInferenceRunner(
            self.family,
            device,
            cleanup=clean_memory_on_device,
        )
        runtime.run_stage(
            "vae",
            lambda: load_moss_vae(args.vae, dtype=resolve_runtime_dtype(args), device=device),
            lambda vae: save_pil_image(
                args,
                decode_moss_latent(vae, latent),
                seed=int(args.seed),
                png_info_factory=_build_png_info,
                prefix="moss_image",
            ),
        )
