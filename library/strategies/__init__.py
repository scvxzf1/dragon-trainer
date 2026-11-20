"""Shared cache strategies used by model-family training pipelines.

The package intentionally keeps imports lazy so dataset and strategy startup
paths do not eagerly load unrelated model-family implementations.
"""

__all__ = [
    "GenericLatentCacheStrategy",
    "FamilyLatentCacheStrategy",
    "LatentsCachingStrategy",
    "TokenizeStrategy",
    "TextEncodingStrategy",
    "TextEncoderOutputsCachingStrategy",
    "FamilySafetensorsTextCacheStrategy",
    "VariantSafetensorsTextCacheStrategy",
    "text_encoder_cache_session",
]


def __getattr__(name):
    if name in {
        "GenericLatentCacheStrategy",
        "FamilyLatentCacheStrategy",
        "LatentsCachingStrategy",
    }:
        from .latent_cache import (
            FamilyLatentCacheStrategy,
            GenericLatentCacheStrategy,
            LatentsCachingStrategy,
        )

        return locals()[name]
    if name in {
        "TokenizeStrategy",
        "TextEncodingStrategy",
        "TextEncoderOutputsCachingStrategy",
    }:
        from .text import (
            TextEncoderOutputsCachingStrategy,
            TextEncodingStrategy,
            TokenizeStrategy,
        )

        return locals()[name]
    if name in {
        "FamilySafetensorsTextCacheStrategy",
        "VariantSafetensorsTextCacheStrategy",
        "text_encoder_cache_session",
    }:
        from .text_cache import (
            FamilySafetensorsTextCacheStrategy,
            VariantSafetensorsTextCacheStrategy,
            text_encoder_cache_session,
        )

        return locals()[name]
    raise AttributeError(name)

