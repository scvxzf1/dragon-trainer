"""Canonical model-family metadata and fail-closed dispatch helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeVar

from library.models.latent_space import (
    ANIMA_F8C16_P2,
    Z_IMAGE_F8C16_P2,
    QWEN_IMAGE_21_F16C64_P1,
    MOSS_IMAGE_F16C128_P1,
    LatentSpaceSpec,
)


T = TypeVar("T")

#: Identifier of the only stage-local PP runtime that exists (PP v0).  A family
#: may only publish it once ``library/training/pp`` can really place, schedule and
#: checkpoint its stages; ``compat_matrix`` and ``library.training.pp.gate`` read
#: this same string, so the family switch and the compatibility gate cannot drift
#: apart.  ``None`` (the default) keeps a family fail-closed.
PIPELINE_PARALLEL_RUNTIME_BACKEND_1F1B = "torch_pipelining_1f1b"


class ModelFamilyHandlerError(RuntimeError):
    """Raised when a registered family lacks an operation handler."""


@dataclass(frozen=True)
class TextCacheSpec:
    suffix: str
    schema: str
    hidden_width: int | None = None

    def metadata(self, family: str) -> dict[str, str]:
        return {
            "model_family": family,
            "cache_schema": self.schema,
        }


@dataclass(frozen=True)
class PipelineParallelFamilySpec:
    """Static topology contract for pipeline planning and WebUI discovery."""

    block_container: str
    default_num_blocks: int
    known_num_blocks: frozenset[int]
    supported_stages: frozenset[int]
    supported_schedules: frozenset[str]
    supported_splits: frozenset[str]
    stage_zero_block_offset: int = 0
    runtime_backend: str | None = None

    @property
    def runtime_available(self) -> bool:
        return self.runtime_backend is not None


@dataclass(frozen=True)
class ModelFamilySpec:
    name: str
    display_name: str
    aliases: frozenset[str]
    latent_space: LatentSpaceSpec
    text_cache: TextCacheSpec
    supported_network_specs: frozenset[str] | None
    supports_method_adapters: bool
    plain_lora_only: bool
    supported_inference_modes: frozenset[str]
    supported_inference_samplers: frozenset[str]
    supported_attention_modes: frozenset[str]
    sdpa_aliases_to_torch: bool
    flash_runtime_dtypes: frozenset[str] | None
    image_test_flow_shift_default: float
    automatic_flow_shift: bool
    supports_anima_selective_lora: bool
    pipeline_parallel: PipelineParallelFamilySpec | None
    supported_tasks: frozenset[str] = frozenset({"t2i"})
    training_task_key: str | None = None
    supported_preview_tasks: frozenset[str] = frozenset({"t2i"})
    max_preview_references: int = 0


MODEL_FAMILY_REGISTRY: dict[str, ModelFamilySpec] = {
    "anima": ModelFamilySpec(
        name="anima",
        display_name="Anima",
        aliases=frozenset({"anima"}),
        latent_space=ANIMA_F8C16_P2,
        text_cache=TextCacheSpec(
            suffix="_anima_te.safetensors",
            schema="anima_te_v1",
        ),
        supported_network_specs=None,
        supports_method_adapters=True,
        plain_lora_only=False,
        supported_inference_modes=frozenset({"single", "batch", "interactive"}),
        supported_inference_samplers=frozenset({"euler", "er_sde", "lcm"}),
        supported_attention_modes=frozenset(
            {"flash", "torch", "mem_efficient", "sageattn", "flex", "xformers", "sdpa"}
        ),
        sdpa_aliases_to_torch=False,
        flash_runtime_dtypes=None,
        image_test_flow_shift_default=1.0,
        automatic_flow_shift=False,
        supports_anima_selective_lora=True,
        pipeline_parallel=PipelineParallelFamilySpec(
            block_container="blocks",
            default_num_blocks=28,
            known_num_blocks=frozenset({28, 40}),
            supported_stages=frozenset({2}),
            supported_schedules=frozenset({"1f1b"}),
            supported_splits=frozenset({"balanced"}),
            # PP v0 switch: only anima has a stage-local runtime today
            # (``library/training/pp``).  Keep every other family at ``None``.
            runtime_backend=PIPELINE_PARALLEL_RUNTIME_BACKEND_1F1B,
        ),
    ),
    "krea2_raw": ModelFamilySpec(
        name="krea2_raw",
        display_name="Krea-2",
        aliases=frozenset({"krea2", "krea2_raw"}),
        latent_space=ANIMA_F8C16_P2,
        text_cache=TextCacheSpec(
            suffix="_krea2_te.safetensors",
            schema="krea2_te_v1",
            hidden_width=2560,
        ),
        supported_network_specs=None,
        supports_method_adapters=False,
        plain_lora_only=False,
        supported_inference_modes=frozenset({"single"}),
        supported_inference_samplers=frozenset({"euler"}),
        supported_attention_modes=frozenset({"torch", "flash", "sdpa"}),
        sdpa_aliases_to_torch=True,
        flash_runtime_dtypes=frozenset({"fp16", "bf16"}),
        image_test_flow_shift_default=3.0,
        automatic_flow_shift=True,
        supports_anima_selective_lora=False,
        pipeline_parallel=PipelineParallelFamilySpec(
            block_container="blocks",
            default_num_blocks=28,
            known_num_blocks=frozenset({28}),
            supported_stages=frozenset({2}),
            supported_schedules=frozenset({"1f1b"}),
            supported_splits=frozenset({"balanced"}),
            stage_zero_block_offset=-1,
        ),
    ),
    "z_image": ModelFamilySpec(
        name="z_image",
        display_name="Z-Image",
        aliases=frozenset({"zimage", "z_image"}),
        latent_space=Z_IMAGE_F8C16_P2,
        text_cache=TextCacheSpec(
            suffix="_z_image_te.safetensors",
            schema="z_image_te_v1",
            hidden_width=2560,
        ),
        supported_network_specs=None,
        supports_method_adapters=False,
        plain_lora_only=False,
        supported_inference_modes=frozenset(),
        supported_inference_samplers=frozenset(),
        supported_attention_modes=frozenset({"torch", "flash", "sdpa"}),
        sdpa_aliases_to_torch=True,
        flash_runtime_dtypes=frozenset({"bf16"}),
        image_test_flow_shift_default=6.0,
        automatic_flow_shift=False,
        supports_anima_selective_lora=False,
        pipeline_parallel=PipelineParallelFamilySpec(
            block_container="layers",
            default_num_blocks=30,
            known_num_blocks=frozenset({30}),
            supported_stages=frozenset({2}),
            supported_schedules=frozenset({"1f1b"}),
            supported_splits=frozenset({"balanced"}),
        ),
    ),
    "qwen_image_2_1": ModelFamilySpec(
        name="qwen_image_2_1",
        display_name="Qwen Image 2.1",
        aliases=frozenset({"qwen_image_2_1", "qwen_image_21", "qwen21"}),
        latent_space=QWEN_IMAGE_21_F16C64_P1,
        text_cache=TextCacheSpec(
            suffix="_qwen_image_2_1_te.safetensors",
            schema="qwen_image_2_1_te_v1",
            hidden_width=4096,
        ),
        supported_network_specs=frozenset({"lora", "lokr", "loha"}),
        supports_method_adapters=False,
        plain_lora_only=True,
        supported_tasks=frozenset({"t2i", "edit"}),
        training_task_key="qwen_image_2_1_task",
        supported_preview_tasks=frozenset({"t2i", "edit"}),
        max_preview_references=4,
        supported_inference_modes=frozenset({"single"}),
        supported_inference_samplers=frozenset({"euler"}),
        supported_attention_modes=frozenset({"torch", "sdpa", "flash"}),
        sdpa_aliases_to_torch=True,
        flash_runtime_dtypes=frozenset({"bf16"}),
        image_test_flow_shift_default=1.0,
        automatic_flow_shift=True,
        supports_anima_selective_lora=False,
        pipeline_parallel=None,
    ),
    "moss_image": ModelFamilySpec(
        name="moss_image",
        display_name="MOSS-Image",
        aliases=frozenset({"moss", "moss_image"}),
        latent_space=MOSS_IMAGE_F16C128_P1,
        text_cache=TextCacheSpec(
            suffix="_moss_image_te_nf4.safetensors",
            schema="moss_image_te_v2",
            hidden_width=4096,
        ),
        supported_network_specs=frozenset({"lora"}),
        supports_method_adapters=False,
        plain_lora_only=True,
        supported_inference_modes=frozenset(),
        supported_inference_samplers=frozenset(),
        supported_attention_modes=frozenset({"torch", "sdpa"}),
        sdpa_aliases_to_torch=True,
        flash_runtime_dtypes=None,
        image_test_flow_shift_default=5.0,
        automatic_flow_shift=False,
        supports_anima_selective_lora=False,
        pipeline_parallel=None,
    ),
}


def known_model_families() -> tuple[str, ...]:
    return tuple(MODEL_FAMILY_REGISTRY)


def normalize_registered_family(
    value,
    *,
    source: str = "model_family",
    allow_empty: bool = False,
    allow_aliases: bool = False,
) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized and allow_empty:
        return ""
    if normalized in MODEL_FAMILY_REGISTRY:
        return normalized
    if allow_aliases:
        normalized = normalized.replace("-", "_")
        for spec in MODEL_FAMILY_REGISTRY.values():
            if normalized in spec.aliases:
                return spec.name
    allowed = ", ".join(known_model_families())
    raise ValueError(f"{source} must be one of: {allowed}; got {value!r}")


def get_model_family_spec(value, *, source: str = "model_family") -> ModelFamilySpec:
    family = normalize_registered_family(value, source=source)
    return MODEL_FAMILY_REGISTRY[family]


def model_family_capability_catalog() -> tuple[dict[str, object], ...]:
    """Return the JSON-safe public capability catalog used by both WebUIs."""

    items: list[dict[str, object]] = []
    for spec in MODEL_FAMILY_REGISTRY.values():
        pipeline = spec.pipeline_parallel
        pipeline_payload: dict[str, object] | None = None
        if pipeline is not None:
            pipeline_payload = {
                "configurable": True,
                "runtime_available": pipeline.runtime_available,
                "runtime_backend": pipeline.runtime_backend,
                "block_container": pipeline.block_container,
                "default_num_blocks": pipeline.default_num_blocks,
                "known_num_blocks": sorted(pipeline.known_num_blocks),
                "stages": sorted(pipeline.supported_stages),
                "schedules": sorted(pipeline.supported_schedules),
                "splits": sorted(pipeline.supported_splits),
            }
        items.append(
            {
                "name": spec.name,
                "display_name": spec.display_name,
                "aliases": sorted(spec.aliases),
                "latent_cache_suffix": spec.latent_space.cache_suffix,
                "text_cache_suffix": spec.text_cache.suffix,
                "text_cache_schema": spec.text_cache.schema,
                "supported_network_specs": (
                    sorted(spec.supported_network_specs)
                    if spec.supported_network_specs is not None
                    else None
                ),
                "supports_method_adapters": spec.supports_method_adapters,
                "plain_lora_only": spec.plain_lora_only,
                "supported_tasks": sorted(spec.supported_tasks),
                "supported_preview_tasks": sorted(spec.supported_preview_tasks),
                "max_preview_references": spec.max_preview_references,
                "supported_attention_modes": sorted(spec.supported_attention_modes),
                "pipeline_parallel": pipeline_payload,
            }
        )
    return tuple(items)


def dispatch_model_family(
    family,
    *,
    operation: str,
    handlers: Mapping[str, T],
) -> T:
    """Return a handler only when every registered family is covered."""

    canonical = normalize_registered_family(family, source=f"{operation} model_family")
    registered = set(MODEL_FAMILY_REGISTRY)
    provided = set(handlers)
    missing = sorted(registered - provided)
    unknown = sorted(provided - registered)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ModelFamilyHandlerError(
            f"{operation} family handlers are incomplete ({'; '.join(details)})"
        )
    return handlers[canonical]


def validate_text_cache_metadata(
    metadata: Mapping[str, str] | None,
    *,
    family: str,
    path: str,
) -> None:
    """Accept legacy unstamped caches, but reject explicit mismatches."""

    spec = get_model_family_spec(family, source="text cache family")
    metadata = dict(metadata or {})
    cached_family = metadata.get("model_family")
    cached_schema = metadata.get("cache_schema")
    if cached_family is not None and cached_family != spec.name:
        raise ValueError(
            f"text cache family mismatch for {path}: {cached_family!r} != {spec.name!r}"
        )
    if cached_schema is not None and cached_schema != spec.text_cache.schema:
        raise ValueError(
            f"text cache schema mismatch for {path}: "
            f"{cached_schema!r} != {spec.text_cache.schema!r}"
        )
