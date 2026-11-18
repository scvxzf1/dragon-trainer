"""Anima text / latent strategy factories used by AnimaTrainer.

Family dispatch (stage 6): each ``get_*`` reads ``resolve_model_family(args)``
and returns the matching Anima or Krea-2-Raw strategy. Anima is the default and
keeps byte-identical behavior; Krea-2 dispatches to ``library.models.krea2_raw``
strategies (Qwen3-VL ChatML tokenize/encode + ``_krea2_te`` cache suffix).
"""

from __future__ import annotations

import torch

from library import cache_identity as _identity
from library.anima import strategy as strategy_anima
from library.env import resolve_model_family
from library.models.family_registry import get_family_plugin


def require_cache_identity(args) -> bool:
    """Whether unstamped (pre-identity) caches must be rebuilt for this run."""
    return bool(getattr(args, "require_cache_identity", False))


def _krea2_tokenize_strategy(args):
    from library.models.krea2_raw.strategy import Krea2TokenizeStrategy

    return Krea2TokenizeStrategy()


def _z_image_tokenize_strategy(args):
    from library.models.z_image.strategy import ZImageTokenizeStrategy

    return ZImageTokenizeStrategy(args.qwen3)


def _moss_tokenize_strategy(args):
    from library.models.moss_image.strategy import MossImageTokenizeStrategy

    return MossImageTokenizeStrategy(args.qwen3, int(getattr(args, "max_token_length", None) or 4096))


def _qwen_image_2_1_tokenize_strategy(args):
    from library.models.qwen_image_2_1.strategy import (
        QwenImage21EditTokenizeStrategy,
        QwenImage21TokenizeStrategy,
    )

    if getattr(args, "qwen_image_2_1_task", "t2i") == "edit":
        return QwenImage21EditTokenizeStrategy(args.qwen3)
    return QwenImage21TokenizeStrategy(args.qwen3)


def _anima_tokenize_strategy(args):
    return strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.qwen3,
        t5_tokenizer_path=args.t5_tokenizer_path,
        qwen3_max_length=args.qwen3_max_token_length,
        t5_max_length=args.t5_max_token_length,
    )


def get_tokenize_strategy(args):
    return get_family_plugin(resolve_model_family(args)).build_tokenize_strategy(args)


def get_tokenizers(tokenize_strategy):
    if isinstance(tokenize_strategy, strategy_anima.AnimaTokenizeStrategy):
        return [tokenize_strategy.qwen3_tokenizer]
    from library.models.krea2_raw.strategy import Krea2TokenizeStrategy

    if isinstance(tokenize_strategy, Krea2TokenizeStrategy):
        return [tokenize_strategy.tokenizer]
    from library.models.z_image.strategy import ZImageTokenizeStrategy

    if isinstance(tokenize_strategy, ZImageTokenizeStrategy):
        return [tokenize_strategy.tokenizer]
    from library.models.qwen_image_2_1.strategy import QwenImage21TokenizeStrategy

    if isinstance(tokenize_strategy, QwenImage21TokenizeStrategy):
        return [tokenize_strategy.tokenizer]
    from library.models.moss_image.strategy import MossImageTokenizeStrategy

    if isinstance(tokenize_strategy, MossImageTokenizeStrategy):
        return [tokenize_strategy.tokenizer]
    raise TypeError(
        f"Unsupported tokenize strategy: {type(tokenize_strategy).__name__}"
    )


def get_latents_caching_strategy(args):
    # VAE / latent cache is shared (same AutoencoderKLQwenImage); anima's
    # caching strategy is family-agnostic.
    family = resolve_model_family(args)
    return get_family_plugin(family).build_latent_cache_strategy(args)


def get_text_encoding_strategy(args):
    return get_family_plugin(resolve_model_family(args)).build_text_encoding_strategy(args)


def get_text_encoder_outputs_caching_strategy(args, weight_dtype: torch.dtype):
    family = resolve_model_family(args)
    qwen_edit = (
        family == "qwen_image_2_1"
        and getattr(args, "qwen_image_2_1_task", "t2i") == "edit"
    )
    if qwen_edit and not args.cache_text_encoder_outputs:
        raise ValueError("Qwen Image 2.1 Edit requires cached Qwen3-VL outputs")
    if not args.cache_text_encoder_outputs:
        return None
    return get_family_plugin(resolve_model_family(args)).build_text_cache_strategy(
        args, weight_dtype
    )


def _z_image_latents_caching_strategy(*args, **kwargs):
    from library.models.z_image.strategy import ZImageLatentsCachingStrategy

    return ZImageLatentsCachingStrategy(*args, **kwargs)


def _qwen_image_2_1_latents_caching_strategy(*args, **kwargs):
    from library.models.qwen_image_2_1.strategy import QwenImage21LatentCache

    return QwenImage21LatentCache(*args, **kwargs)


def _moss_latents_caching_strategy(*args, **kwargs):
    from library.models.moss_image.strategy import MossImageLatentCache

    return MossImageLatentCache(*args, **kwargs)


def get_models_for_text_encoding(args, accelerator, text_encoders):
    if args.cache_text_encoder_outputs:
        return None  # no text encoders needed for encoding
    return text_encoders
