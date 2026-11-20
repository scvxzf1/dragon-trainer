"""MOSS frozen-text and effective-16x latent caching strategies."""

from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from library.strategies.text import TextEncoderOutputsCachingStrategy
from library.io.cache import resolve_cache_path
from library.strategies.latent_cache import FamilyLatentCacheStrategy
from .cache_policy import (
    MOSS_HIDDEN_SIZE, MOSS_MAX_LENGTH, resolve_text_token_cap, text_cache_metadata,
    validate_cache_metadata, validate_quantization,
)
from .text import MossImageTextEncodingStrategy, MossImageTokenizeStrategy

__all__ = [
    "MossImageTokenizeStrategy", "MossImageTextEncodingStrategy",
    "MossImageTextCache", "MossImageLatentCache",
]


class MossImageTextCache(TextEncoderOutputsCachingStrategy):
    MODEL_FAMILY = "moss_image"

    def __init__(self, cache_to_disk, batch_size, skip_disk_cache_validity_check,
                 *, quantization="nf4", max_text_tokens=MOSS_MAX_LENGTH,
                 use_shuffled_caption_variants=False):
        if use_shuffled_caption_variants:
            raise ValueError("MOSS shuffled caption variants are unsupported")
        if batch_size == "auto":
            batch_size = 1  # Qwen GDN always executes one unpadded caption at a time.
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)
        self.quantization = validate_quantization(quantization)
        self.max_text_tokens = resolve_text_token_cap(max_text_tokens)
        self.use_shuffled_caption_variants = False

    @property
    def preprocess_batch_size(self):
        return max(1, int(self.batch_size or 1))

    @property
    def cache_suffix(self):
        return f"_moss_image_te_{self.quantization}.safetensors"

    def get_outputs_npz_path(self, image_abs_path, cache_dir=None, image_dir=None):
        return resolve_cache_path(image_abs_path, self.cache_suffix,
                                  cache_dir=cache_dir, image_dir=image_dir)

    def _validate(self, handle, path):
        # Identity remains mandatory even when callers skip expensive checks.
        validate_cache_metadata(
            handle.metadata(), self.quantization, path, self.max_text_tokens,
        )
        hidden = handle.get_slice("hiddens")
        mask = handle.get_slice("mask")
        shape = hidden.get_shape()
        if (len(shape) != 2 or not 0 < shape[0] <= 4096
                or shape[1] != MOSS_HIDDEN_SIZE or hidden.get_dtype() != "BF16"
                or mask.get_shape() != [shape[0]] or mask.get_dtype() != "BOOL"):
            raise ValueError(f"Invalid MOSS text cache layout: {path}")
        rate = handle.get_tensor("caption_dropout_rate")
        if rate.ndim != 0 or rate.dtype != torch.float32 or rate.item() != 0:
            raise ValueError(f"MOSS caption dropout must be zero: {path}")
        if "num_variants" in handle.keys():
            raise ValueError(f"MOSS caption variants are unsupported: {path}")

    def load_outputs_npz(self, npz_path):
        with safe_open(npz_path, framework="pt") as handle:
            self._validate(handle, npz_path)
            return [handle.get_tensor("hiddens"), handle.get_tensor("mask"),
                    handle.get_tensor("caption_dropout_rate")]

    def is_disk_cached_outputs_expected(self, npz_path, **kwargs):
        if not self.cache_to_disk or not Path(npz_path).is_file():
            return False
        try:
            with safe_open(npz_path, framework="pt") as handle:
                self._validate(handle, npz_path)
            return True
        except (OSError, ValueError, RuntimeError, KeyError, SafetensorError):
            return False

    def cache_batch_outputs(self, tokenize_strategy, models, text_encoding_strategy, batch):
        if models[0] is None:
            raise RuntimeError("Missing MOSS text cache; run text preprocessing before training")
        if getattr(models[0], "quantization", None) != self.quantization:
            raise ValueError("MOSS text encoder quantization must match the cache quantization")
        for info in batch:
            if float(getattr(info, "caption_dropout_rate", 0)) != 0:
                raise ValueError("MOSS caption_dropout_rate must be zero")
            if (getattr(info, "cache_caption_variants", False)
                    or len(getattr(info, "caption_variants", None) or []) > 1
                    or getattr(info, "caption_tag_dropout_rate", 0)):
                raise ValueError("MOSS shuffled/dropout caption variants are unsupported")
        hiddens, masks = text_encoding_strategy.encode_tokens(
            tokenize_strategy, models, tokenize_strategy.tokenize([info.caption for info in batch]),
        )
        for info, hidden, mask in zip(batch, hiddens, masks):
            length = int(mask.sum().item())
            tensors = {
                "hiddens": hidden[:length].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                "mask": mask[:length].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                "caption_dropout_rate": torch.tensor(0.0, dtype=torch.float32),
            }
            if self.cache_to_disk:
                path = Path(info.text_encoder_outputs_npz)
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file(tensors, str(path), metadata=text_cache_metadata(
                    self.quantization, self.max_text_tokens))
            else:
                info.text_encoder_outputs = list(tensors.values())


class MossImageLatentCache(FamilyLatentCacheStrategy):
    def __init__(
        self,
        cache_to_disk,
        batch_size,
        skip_disk_cache_validity_check,
        *,
        cache_identity=None,
        require_cache_identity=False,
        encode_by_vae=None,
    ):
        from .latent import encode_moss_latents

        super().__init__(
            cache_to_disk,
            batch_size,
            skip_disk_cache_validity_check,
            model_family="moss_image",
            encode_by_vae=encode_by_vae or encode_moss_latents,
            cache_identity=cache_identity,
            require_cache_identity=require_cache_identity,
        )

    def cache_batch_latents(self, vae, image_infos, flip_aug, alpha_mask, random_crop):
        parameter = next(vae.parameters())
        self._default_cache_batch_latents(
            lambda images: self.encode_vae(vae, images).to("cpu"),
            parameter.device, parameter.dtype, image_infos, flip_aug, alpha_mask,
            random_crop, multi_resolution=True,
        )
