"""Shared cache strategies used by model-family training pipelines.

The package intentionally keeps imports lazy: the Anima compatibility facade
imports ``library.strategies.latent_cache`` while the text-cache engine uses
Anima's base protocol. Eager re-exports would create a circular import during
legacy strategy loading.
"""

__all__ = [
    "GenericLatentCacheStrategy",
    "FamilySafetensorsTextCacheStrategy",
    "VariantSafetensorsTextCacheStrategy",
    "text_encoder_cache_session",
]


def __getattr__(name):
    if name == "GenericLatentCacheStrategy":
        from .latent_cache import GenericLatentCacheStrategy

        return GenericLatentCacheStrategy
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

