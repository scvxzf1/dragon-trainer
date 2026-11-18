"""Lazy model-family plugin loading without importing runtime dependencies."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol

from library.models.family_registry import normalize_registered_family


class ModelFamilyPlugin(Protocol):
    """Small strategy-factory surface shared by family plugins."""

    family: str

    def build_tokenize_strategy(self, args: Any): ...

    def build_text_encoding_strategy(self, args: Any): ...

    def build_latent_cache_strategy(self, args: Any): ...

    def build_text_cache_strategy(self, args: Any, weight_dtype: Any): ...

    def compute_noise_pred_and_target(self, *args: Any, **kwargs: Any): ...

    def load_preprocess_vae(self, args: Any, dtype: Any): ...

    @property
    def text_preprocess_spec(self) -> "TextPreprocessSpec": ...


@dataclass(frozen=True)
class TextPreprocessSpec:
    """Static CLI contract for a family's text-cache preprocessing command."""

    runner: str
    module: str | None = None
    qwen3_config_key: str | None = None
    qwen3_default: str | None = None
    dit_config_key: str | None = None
    dit_default: str | None = None
    default_shuffle_variants: str = "0"
    default_tag_dropout_rate: str = "0.0"
    supports_tag_dropout: bool = True
    supports_diff_output_preservation: bool = False
    strip_diff_output_preservation: bool = False


_PLUGIN_MODULES = {
    "anima": "library.anima.plugin",
    "krea2_raw": "library.models.krea2_raw.plugin",
    "z_image": "library.models.z_image.plugin",
    "qwen_image_2_1": "library.models.qwen_image_2_1.plugin",
    "moss_image": "library.models.moss_image.plugin",
}
_PLUGINS: dict[str, ModelFamilyPlugin] = {}


def get_family_plugin(family: str) -> ModelFamilyPlugin:
    """Load exactly one registered family plugin on first use."""

    canonical = normalize_registered_family(family, source="family plugin")
    plugin = _PLUGINS.get(canonical)
    if plugin is None:
        module = importlib.import_module(_PLUGIN_MODULES[canonical])
        plugin = module.PLUGIN
        if plugin.family != canonical:
            raise RuntimeError(
                f"family plugin identity mismatch: {canonical!r} != {plugin.family!r}"
            )
        _PLUGINS[canonical] = plugin
    return plugin
