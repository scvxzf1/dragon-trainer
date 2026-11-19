"""Standalone Qwen Image 2.1 T2I/Edit inference with Euler and plain LoRA."""

from __future__ import annotations

import gc
import logging
import math
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from library.inference.output import _build_png_info, get_time_flag, save_latent
from library.inference.family_runner import FamilyInferenceRunner
from library.models.qwen_image_2_1.attention_backend import (
    normalize_qwen_image_2_1_attention_mode,
    prepare_qwen_image_2_1_attention,
)
from library.models.qwen_image_2_1.inference_text_encoder import encode_with_text_encoder
from library.models.qwen_image_2_1.preview_decode import decode_latent
from library.models.qwen_image_2_1.sampling import generate_preview
from library.models.qwen_image_2_1.strategy import (
    QwenImage21TextEncodingStrategy,
    QwenImage21TokenizeStrategy,
)
from library.models.qwen_image_2_1.weights import (
    load_qwen_image_2_1_text_encoder,
    load_qwen_image_2_1_transformer,
    load_qwen_image_2_1_vae,
)
from library.runtime.device import clean_memory_on_device

logger = logging.getLogger(__name__)


class QwenImage21InferenceRunner:
    family = "qwen_image_2_1"
    supports_reference_conditioning = True

    @staticmethod
    def validate(args, *, mode: str = "single") -> None:
        validate_qwen_image_2_1_inference_args(args, mode=mode)

    @staticmethod
    def prepare(args, *, mode: str = "single") -> None:
        return None

    def generate_latent(self, args, gen_settings):
        self.validate(args, mode="single")
        device = gen_settings.device
        runtime_dtype = gen_settings.runtime_dtype
        from library.models.qwen_image_2_1.inference_conditions import (
            prepare_qwen_image_2_1_conditions,
        )

        runtime = FamilyInferenceRunner(
            self.family,
            device,
            cleanup=clean_memory_on_device,
        )
        positive, negative, reference = runtime.run_stage(
            "conditions",
            lambda: None,
            lambda _: prepare_qwen_image_2_1_conditions(
                args,
                device,
                runtime_dtype,
            ),
        )
        return runtime.run_stage(
            "dit",
            lambda: load_qwen_image_2_1_dit_for_inference(
                args,
                device,
                runtime_dtype,
            ),
            lambda loaded: generate_qwen_image_2_1(
                args,
                loaded[0],
                loaded[1],
                positive,
                negative,
                device,
                args.seed,
                runtime_dtype,
                reference=reference,
            ),
        )

    def save_output(self, args, latent, device):
        return save_qwen_image_2_1_output(args, latent, device)


_UNSUPPORTED_ANIMA_OPTIONS = {
    "pgraft": "--pgraft",
    "spectrum": "--spectrum",
    "spd": "--spd",
    "tiled_diffusion": "--tiled_diffusion",
    "dcw": "--dcw",
    "pooled_text_proj": "--pooled_text_proj",
    "ip_adapter_weight": "--ip_adapter_weight",
    "easycontrol_weight": "--easycontrol_weight",
    "soft_tokens_weight": "--soft_tokens_weight",
    "smc_cfg": "--smc_cfg",
    "compile": "--compile",
    "compile_blocks": "--compile_blocks",
    "anima_selective_lora": "--anima_selective_lora",
    "cns": "--cns",
}


def validate_qwen_image_2_1_inference_args(args, *, mode: str = "single") -> None:
    """Validate the deliberately small standalone Qwen inference contract."""
    if mode != "single":
        raise SystemExit(
            "Qwen Image 2.1 standalone inference currently supports only "
            "single-prompt mode."
        )
    from library.models.qwen_image_2_1.inference_conditions import validate_task

    validate_task(args)
    if str(getattr(args, "runtime_dtype", "bf16") or "bf16") != "bf16":
        raise SystemExit("Qwen Image 2.1 standalone inference requires BF16 runtime precision.")
    sampler = str(getattr(args, "sampler", "euler") or "euler").lower()
    if sampler != "euler":
        raise SystemExit("Qwen Image 2.1 standalone inference supports only the Euler sampler.")
    flow_shift = getattr(args, "flow_shift", None)
    if flow_shift is not None:
        # The official scheduler derives its dynamic shift from image token
        # count.  ``3.0`` is the legacy CLI default shared with Anima/Krea;
        # accept it for compatibility but never feed it into Qwen sampling.
        try:
            flow_shift = float(flow_shift)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid Qwen Image 2.1 flow_shift: {flow_shift!r}") from exc
        if not math.isfinite(flow_shift) or flow_shift not in {1.0, 3.0}:
            raise SystemExit(
                "Qwen Image 2.1 uses its automatic token-count flow shift; "
                "--flow_shift must remain 1.0 or the legacy default 3.0."
            )
    unsupported = [
        label
        for name, label in _UNSUPPORTED_ANIMA_OPTIONS.items()
        if getattr(args, name, None)
    ]
    if unsupported:
        raise SystemExit(
            "Qwen Image 2.1 standalone inference does not support:\n  - "
            + "\n  - ".join(unsupported)
        )
    lora_weights = getattr(args, "lora_weight", None)
    if isinstance(lora_weights, str):
        lora_weights = [lora_weights]
    if lora_weights and len(lora_weights) > 1:
        raise SystemExit("Qwen Image 2.1 standalone inference accepts at most one plain LoRA checkpoint.")
    try:
        infer_steps = int(getattr(args, "infer_steps", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Qwen Image 2.1 infer_steps: {getattr(args, 'infer_steps', None)!r}") from exc
    if infer_steps <= 0:
        raise ValueError(f"Qwen Image 2.1 infer_steps must be positive, got {infer_steps}")
    height, width = (int(value) for value in getattr(args, "image_size", (0, 0)))
    # The family forward flattens 2x2 image patches.  The 16x VAE stride
    # therefore also needs an even latent grid, i.e. pixel dimensions %32.
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError(
            "Qwen Image 2.1 image_size must contain positive dimensions divisible by 32; "
            f"got {(height, width)}"
        )


def _encode_prompt(
    text_encoder: torch.nn.Module,
    text_encoder_path: str,
    prompt: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = QwenImage21TokenizeStrategy(text_encoder_path)
    tokens = tokenizer.tokenize([prompt or ""])
    strategy = QwenImage21TextEncodingStrategy()
    with torch.inference_mode():
        hiddens, mask = strategy.encode_tokens(tokenizer, [text_encoder], tokens)
    return hiddens.to(device=device, dtype=dtype), mask.to(device=device, dtype=torch.bool)


def prepare_qwen_image_2_1_text(
    args,
    device: torch.device,
    runtime_dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Encode positive/negative prompts, then release the text encoder."""
    def encode(text_encoder):
        positive = _encode_prompt(
            text_encoder, args.text_encoder, args.prompt or "", device=device, dtype=runtime_dtype
        )
        negative = _encode_prompt(
            text_encoder,
            args.text_encoder,
            getattr(args, "negative_prompt", "") or "",
            device=device,
            dtype=runtime_dtype,
        )
        return positive, negative

    return encode_with_text_encoder(
        args, device, encode, loader=load_qwen_image_2_1_text_encoder
    )


def _load_plain_lora(args, dit: torch.nn.Module, device: torch.device, dtype: torch.dtype):
    paths = getattr(args, "lora_weight", None)
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        return None
    from library.inference.models import (
        _load_lora_state_dict_for_inference,
        _read_lora_metadata,
        _resolve_lora_multiplier_for_index,
    )
    from networks import lora_anima

    path = str(paths[0])
    metadata = _read_lora_metadata(path)
    # Qwen's network factory validates the family stamp and target spec.  The
    # explicit spec check here keeps unsupported LoKr/LoHa variants fail-closed
    # for this initial plain-LoRA-only standalone path.
    stamped_spec = str(metadata.get("ss_network_spec", "lora")).strip().lower() or "lora"
    if stamped_spec != "lora":
        raise ValueError(
            "Qwen Image 2.1 standalone inference currently supports plain LoRA only; "
            f"checkpoint {path} declares ss_network_spec={stamped_spec!r}"
        )
    lora_sd = _load_lora_state_dict_for_inference(args, path)
    unsupported_keys = sorted(key for key in lora_sd if not key.startswith("lora_unet_"))
    if unsupported_keys:
        raise ValueError(
            f"Qwen Image 2.1 LoRA checkpoint {path}: unexpected keys "
            f"({len(unsupported_keys)}): {unsupported_keys[:5]}"
        )
    if not lora_sd:
        raise ValueError(f"Qwen Image 2.1 LoRA has no DiT tensors: {path}")
    # The from-weights factory requires a saved alpha for each down weight;
    # unlike direct module construction, it does not fill missing alphas.
    missing_alpha = sorted(
        key.removesuffix(".lora_down.weight") + ".alpha"
        for key in lora_sd if key.endswith(".lora_down.weight")
        and key.removesuffix(".lora_down.weight") + ".alpha" not in lora_sd
    )
    if missing_alpha:
        raise ValueError(
            f"Qwen Image 2.1 LoRA checkpoint {path}: missing keys "
            f"({len(missing_alpha)}): {missing_alpha[:5]}"
        )
    multiplier = _resolve_lora_multiplier_for_index(getattr(args, "lora_multiplier", 1.0), 0)
    network, weights_sd = lora_anima.create_network_from_weights(
        multiplier=multiplier,
        file=path,
        ae=None,
        text_encoders=[],
        unet=dit,
        weights_sd=lora_sd,
        metadata=metadata,
        for_inference=True,
    )
    if getattr(getattr(network, "cfg", None), "model_family", None) != "qwen_image_2_1":
        raise ValueError(
            f"Qwen Image 2.1 inference requires checkpoint {path} stamped "
            "ss_model_family=qwen_image_2_1"
        )
    # apply_to both registers the adapter modules and monkey-patches DiT.
    # Inspect their future state first, excluding the temporary base-module
    # reference that apply_to removes, so malformed files leave DiT untouched.
    expected = dict(network.state_dict())
    for lora in network.unet_loras:
        expected.update({
            f"{lora.lora_name}.{key}": value
            for key, value in lora.state_dict().items()
            if not key.startswith("org_module.")
        })
    weights_sd = dict(weights_sd)
    missing = sorted(expected.keys() - weights_sd.keys())
    unexpected = sorted(weights_sd.keys() - expected.keys())
    mismatched = sorted(
        key for key in expected.keys() & weights_sd.keys()
        if expected[key].shape != weights_sd[key].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Qwen Image 2.1 LoRA checkpoint {path}: "
            f"missing keys ({len(missing)}): {missing[:5]}; "
            f"unexpected keys ({len(unexpected)}): {unexpected[:5]}; "
            f"shape mismatches ({len(mismatched)}): {mismatched[:5]}"
        )
    network.apply_to([], dit, apply_text_encoder=False, apply_unet=True)
    try:
        network.load_state_dict(weights_sd, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Qwen Image 2.1 LoRA checkpoint {path} failed strict loading: {exc}") from exc
    network.to(device=device, dtype=dtype).eval()
    logger.info("Qwen Image 2.1 plain LoRA attached: %s (multiplier=%s)", path, multiplier)
    return network


def load_qwen_image_2_1_dit_for_inference(
    args,
    device: torch.device,
    dit_weight_dtype: Optional[torch.dtype],
) -> Tuple[torch.nn.Module, Optional[Any]]:
    """Load Qwen DiT, configure attention, then attach one optional plain LoRA."""
    dtype = dit_weight_dtype or torch.bfloat16
    mode = normalize_qwen_image_2_1_attention_mode(getattr(args, "attn_mode", None))
    dit = load_qwen_image_2_1_transformer(args.dit, dtype=dtype, device=device)
    try:
        prepare_qwen_image_2_1_attention(dit, mode, dtype=dtype)
        dit.eval().requires_grad_(False)
        network = _load_plain_lora(args, dit, device, dtype)
        return dit, network
    except BaseException:
        del dit
        gc.collect()
        clean_memory_on_device(device)
        raise


def generate_qwen_image_2_1(
    args,
    dit: torch.nn.Module,
    network: Optional[Any],
    positive: tuple[torch.Tensor, ...],
    negative: tuple[torch.Tensor, ...],
    device: torch.device,
    seed: int,
    runtime_dtype: torch.dtype,
    *,
    reference: Optional[tuple[torch.Tensor, ...]] = None,
) -> torch.Tensor:
    """Run official dynamic-shift Euler sampling and return [1,64,H/16,W/16]."""
    del network  # attached hooks live on ``dit`` during sampling
    height, width = (int(value) for value in args.image_size)
    prompt = {
        "prompt": args.prompt or "",
        "negative_prompt": getattr(args, "negative_prompt", "") or "",
        "width": width,
        "height": height,
        "sample_steps": int(args.infer_steps),
        "guidance_scale": float(args.guidance_scale),
    }
    logger.info(
        "Qwen Image 2.1 sampling: steps=%s cfg=%s latent=(%s,%s)",
        prompt["sample_steps"], prompt["guidance_scale"], height // 16, width // 16,
    )
    return generate_preview(
        dit,
        prompt,
        positive,
        negative,
        reference,
        device=device,
        dtype=runtime_dtype,
        seed=int(seed),
    )


def save_qwen_image_2_1_output(
    args, latent: torch.Tensor, device: torch.device, *, original_base_name: Optional[str] = None,
) -> None:
    """Decode/save Qwen's 64-channel normalized latent with its own 16x VAE."""
    if latent.ndim != 4 or latent.shape[:2] != (1, 64):
        raise ValueError(f"Qwen Image 2.1 output latent must be [1,64,H,W], got {tuple(latent.shape)}")
    height, width = int(latent.shape[-2] * 16), int(latent.shape[-1] * 16)
    if args.output_type in {"latent", "latent_images"}:
        # Shared save_latent assumes the Anima 8x stride, so retain the same
        # metadata contract while passing Qwen's true pixel dimensions.
        old_size = args.image_size
        args.image_size = [height, width]
        try:
            save_latent(latent, args, height, width)
        finally:
            args.image_size = old_size
    if args.output_type == "latent":
        return
    from library.inference.precision import resolve_runtime_dtype

    vae_dtype = resolve_runtime_dtype(args)
    # The standalone caller releases DiT and clears its allocator before this
    # stage, so decode can use the selected runtime device without co-residency.
    logger.info("Qwen Image 2.1 VAE decoding on %s", device)
    runtime = FamilyInferenceRunner(
        "qwen_image_2_1",
        device,
        cleanup=clean_memory_on_device,
    )

    def save_decoded_image(vae):
        with torch.inference_mode():
            image = decode_latent(vae, latent)
        os.makedirs(args.save_path, exist_ok=True)
        stem = f"{get_time_flag()}_{args.seed}"
        if original_base_name is not None:
            stem += f"_{Path(original_base_name).name}"
        path = Path(args.save_path) / f"{stem}.png"
        image.save(path, pnginfo=_build_png_info(args, args.seed, stem))
        logger.info("Qwen Image 2.1 image saved to: %s", path)

    runtime.run_stage(
        "vae",
        lambda: load_qwen_image_2_1_vae(args.vae, dtype=vae_dtype, device=device),
        save_decoded_image,
    )
