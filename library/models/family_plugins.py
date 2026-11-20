"""Lazy model-family plugin loading without importing runtime dependencies."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol

from library.models.family_registry import normalize_registered_family


class InferenceRunner(Protocol):
    """Method surface consumed by the generic inference entrypoint."""

    family: str
    supports_reference_conditioning: bool

    def validate(self, args: Any, *, mode: str = "single") -> None: ...

    def prepare(self, args: Any, *, mode: str = "single") -> None: ...

    def generate_latent(self, args: Any, gen_settings: Any): ...

    def save_output(self, args: Any, latent: Any, device: Any) -> None: ...


class FlowFamilyPlugin(Protocol):
    """Model-family-specific hooks around the shared flow-matching engine."""

    def sample_sigmas(
        self,
        batch_size: int,
        height: int,
        width: int,
        args: Any,
        device: Any,
        *,
        context: Any,
    ): ...

    def prepare_conditions(
        self,
        batch: Any,
        conditions: Any,
        *,
        context: Any,
        is_train: bool,
    ): ...

    def forward_velocity_4d(
        self,
        dit: Any,
        noisy_4d: Any,
        conditions: Any,
        sigmas: Any,
        *,
        context: Any,
    ): ...


class CacheFamilyPlugin(Protocol):
    def build_tokenize_strategy(self, args: Any): ...

    def build_text_encoding_strategy(self, args: Any): ...

    def build_latent_cache_strategy(self, args: Any): ...

    def encode_vae(self, vae: Any, pixels: Any): ...

    def build_text_cache_strategy(self, args: Any, weight_dtype: Any): ...

    def load_preprocess_vae(self, args: Any, dtype: Any): ...

    @property
    def text_preprocess_spec(self) -> "TextPreprocessSpec": ...


class InferenceFamilyPlugin(Protocol):

    def build_inference_runner(self, args: Any) -> InferenceRunner: ...


class PreviewFamilyPlugin(Protocol):
    def decode_preview_samples(
        self,
        accelerator: Any,
        args: Any,
        vae: Any,
        *,
        dit: Any = None,
        network: Any = None,
    ) -> None: ...

    def decode_pending_samples(self, accelerator: Any, args: Any, vae: Any) -> None: ...


class QwenPreviewTextPlugin(Protocol):
    def cache_preview_text(
        self,
        trainer: Any,
        args: Any,
        accelerator: Any,
        model: Any,
        tokenizer: Any,
        encoder: Any,
    ) -> None: ...


class ModelFamilyPlugin(
    FlowFamilyPlugin, CacheFamilyPlugin, InferenceFamilyPlugin, PreviewFamilyPlugin, Protocol
):
    """Compatibility view of the complete plugin surface."""

    family: str

    def compute_noise_pred_and_target(self, *args: Any, **kwargs: Any): ...


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


_CAPABILITY_METHODS = {
    "cache": (
        "build_tokenize_strategy", "build_text_encoding_strategy", "build_latent_cache_strategy",
        "build_text_cache_strategy", "encode_vae", "load_preprocess_vae",
    ),
    "flow": ("compute_noise_pred_and_target", "sample_sigmas", "prepare_conditions", "forward_velocity_4d"),
    "inference": ("build_inference_runner",),
    "preview": ("decode_preview_samples", "decode_pending_samples"),
    "qwen_preview_text": ("cache_preview_text",),
}
_DEFAULT_CAPABILITIES = ("cache", "flow", "inference", "preview")

_REQUIRED_RUNNER_METHODS = (
    "validate",
    "prepare",
    "generate_latent",
    "save_output",
)


class FamilyPluginContractError(RuntimeError, TypeError):
    """A plugin is missing part of the shared family contract."""


def _require_callable(
    value: Any,
    *,
    owner: str,
    attribute: str,
) -> None:
    """Fail closed when a plugin contract member is missing or non-callable."""
    if not callable(value):
        raise RuntimeError(f"{owner} has no callable {attribute}()")


def validate_plugin_contract(plugin: Any, family: str, *, capability: str | None = None) -> None:
    """Validate the capability used by the caller, or the full legacy surface."""
    if getattr(plugin, "family", None) != family:
        raise RuntimeError(
            f"family plugin identity mismatch: {family!r} != {getattr(plugin, 'family', None)!r}"
        )
    if capability is not None and capability not in _CAPABILITY_METHODS:
        raise ValueError(f"unknown plugin capability {capability!r}")
    groups = (capability,) if capability is not None else _DEFAULT_CAPABILITIES
    missing = [
        attribute
        for group in groups
        for attribute in _CAPABILITY_METHODS[group]
        if not callable(getattr(plugin, attribute, None))
    ]
    if missing:
        raise FamilyPluginContractError(
            f"family plugin {family!r} missing callable contract methods ({capability or 'all'}): "
            + ", ".join(
                f"{attribute} (no callable {attribute}())"
                for attribute in missing
            )
        )
    if capability in (None, "cache") and not isinstance(
        getattr(plugin, "text_preprocess_spec", None), TextPreprocessSpec
    ):
        raise FamilyPluginContractError(
            f"family plugin {family!r} must expose a TextPreprocessSpec"
        )


def validate_runner_contract(runner: Any, family: str) -> InferenceRunner:
    """Validate the generic runner surface without importing family code."""
    if getattr(runner, "family", None) != family:
        raise RuntimeError(
            f"Inference runner identity mismatch: "
            f"{getattr(runner, 'family', None)!r} != {family!r}"
        )
    supports_references = getattr(runner, "supports_reference_conditioning", None)
    if not isinstance(supports_references, bool):
        raise RuntimeError(
            f"Inference runner {family!r} must declare "
            "supports_reference_conditioning as bool"
        )
    for attribute in _REQUIRED_RUNNER_METHODS:
        _require_callable(
            getattr(runner, attribute, None),
            owner=f"inference runner {family!r}",
            attribute=attribute,
        )
    return runner


def get_family_plugin(family: str, *, capability: str | None = None) -> ModelFamilyPlugin:
    """Load exactly one registered family plugin on first use."""

    canonical = normalize_registered_family(family, source="family plugin")
    plugin = _PLUGINS.get(canonical)
    if plugin is None:
        module = importlib.import_module(_PLUGIN_MODULES[canonical])
        plugin = module.PLUGIN
    validate_plugin_contract(plugin, canonical, capability=capability)
    _PLUGINS[canonical] = plugin
    return plugin
