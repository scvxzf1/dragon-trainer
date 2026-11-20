"""Qwen3 text and Flux-VAE cache strategies for Z-Image."""

from __future__ import annotations

from typing import Any, List, Optional, Union

import torch

from library.strategies.text import TextEncodingStrategy, TokenizeStrategy
from library.io.cache import resolve_cache_path
from library.models.family_registry import get_model_family_spec
from library.strategies.text_cache import FamilySafetensorsTextCacheStrategy
from library.strategies.latent_cache import FamilyLatentCacheStrategy
from library.models.z_image.latent import encode_z_image_latents
from library.models.z_image.weights import resolve_z_image_tokenizer_path


Z_IMAGE_MAX_LENGTH = 512
Z_IMAGE_CACHE_DTYPE = torch.bfloat16


class ZImageTokenizeStrategy(TokenizeStrategy):
    def __init__(self, tokenizer_path: str, max_length: int = Z_IMAGE_MAX_LENGTH):
        from transformers import AutoTokenizer

        self.max_length = max_length
        self.tokenizer_path = resolve_z_image_tokenizer_path(tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        captions = [text] if isinstance(text, str) else list(text)
        prompts = [
            f"<|im_start|>user\n{caption}<|im_end|>\n<|im_start|>assistant\n"
            for caption in captions
        ]
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return [encoded.input_ids, encoded.attention_mask]

    def tokenize_with_weights(self, text: Union[str, List[str]]) -> tuple:
        tokens = self.tokenize(text)
        return tokens, [torch.ones_like(tokens[0], dtype=torch.float32)]


class ZImageTextEncodingStrategy(TextEncodingStrategy):
    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        model = models[0]
        input_ids, attention_mask = tokens
        device = next(model.parameters()).device
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                output_hidden_states=True,
                use_cache=False,
            )
        return [outputs.hidden_states[-2], attention_mask.to(device).bool()]

    def apply_caption_dropout_inplace(self, caption_dropout_rates, **_kwargs) -> None:
        if torch.as_tensor(caption_dropout_rates).gt(0).any():
            raise ValueError(
                "Z-Image caption_dropout_rate > 0 is not supported until "
                "empty-prompt embeddings are cached explicitly"
            )


class ZImageTextEncoderOutputsCachingStrategy(FamilySafetensorsTextCacheStrategy):
    MODEL_FAMILY = "z_image"
    CACHE_LABEL = "Z-Image"
    EXPECTED_HIDDEN_RANK = 2

    def get_outputs_npz_path(
        self,
        image_abs_path: str,
        cache_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
    ) -> str:
        suffix = get_model_family_spec(self.MODEL_FAMILY).text_cache.suffix
        return resolve_cache_path(
            image_abs_path,
            suffix,
            cache_dir=cache_dir,
            image_dir=image_dir,
        )


class ZImageLatentsCachingStrategy(FamilyLatentCacheStrategy):
    def __init__(
        self,
        cache_to_disk,
        batch_size,
        skip_disk_cache_validity_check,
        *,
        cache_identity=None,
        require_cache_identity=False,
        encode_by_vae=encode_z_image_latents,
    ):
        super().__init__(
            cache_to_disk,
            batch_size,
            skip_disk_cache_validity_check,
            model_family="z_image",
            encode_by_vae=encode_by_vae,
            cache_identity=cache_identity,
            require_cache_identity=require_cache_identity,
        )
