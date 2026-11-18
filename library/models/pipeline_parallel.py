"""Model-family-aware pipeline-parallel planning primitives.

This module provides topology validation, deterministic block ownership plans,
and borrowing probe wrappers. Stage-local 1F1B execution, communication,
optimizer state, and checkpoint recovery live in :mod:`library.training.pp`;
families without a wired runtime stay fail-closed via ``family_registry`` and
``library.training.pp.gate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from torch import nn

from library.models.family_registry import (
    PipelineParallelFamilySpec,
    dispatch_model_family,
    get_model_family_spec,
    normalize_registered_family,
)


PIPELINE_PARALLEL_SCHEDULES = ("1f1b",)
PIPELINE_PARALLEL_SPLITS = ("balanced",)
MAX_PIPELINE_PARALLEL_STAGES = 2
MAX_PIPELINE_PARALLEL_MICROBATCHES = 1024


def _get(config: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _bool(value: Any, default: bool, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    from library.config.normalize import parse_bool

    return parse_bool(value, default=default, strict=True, field=field)


def _int(value: Any, default: int, *, field: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    raise ValueError(f"{field} must be an integer, got {value!r}")


@dataclass(frozen=True)
class PipelineParallelConfig:
    """Normalized PP settings shared by CLI, preflight, and probes."""

    enabled: bool = False
    stages: int = 2
    microbatches: int = 4
    schedule: str = "1f1b"
    split: str = "balanced"
    # Per-stage block swap, deliberately a *separate* knob from blocks_to_swap:
    # PP swaps only the blocks this rank owns, after their own backward, and a
    # model that already carries blocks_to_swap>0 is refused (placement.py).
    # Sharing the plain key would silently mean different things.
    # Declared last so positional construction keeps its historical order.
    blocks_to_swap: int = 0

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | object
    ) -> "PipelineParallelConfig":
        schedule = _get(config, "pipeline_parallel_schedule", "1f1b")
        split = _get(config, "pipeline_parallel_split", "balanced")
        pp_blocks_to_swap = _int(
            _get(config, "pipeline_parallel_blocks_to_swap", 0),
            0,
            field="pipeline_parallel_blocks_to_swap",
        )
        if pp_blocks_to_swap < 0:
            raise ValueError(
                "pipeline_parallel_blocks_to_swap must be >= 0, "
                f"got {pp_blocks_to_swap}"
            )
        return cls(
            enabled=_bool(
                _get(config, "pipeline_parallel", False),
                False,
                field="pipeline_parallel",
            ),
            stages=_int(
                _get(config, "pipeline_parallel_stages", 2),
                2,
                field="pipeline_parallel_stages",
            ),
            microbatches=_int(
                _get(config, "pipeline_parallel_microbatches", 4),
                4,
                field="pipeline_parallel_microbatches",
            ),
            blocks_to_swap=pp_blocks_to_swap,
            schedule=str("1f1b" if schedule is None else schedule).strip().lower(),
            split=str("balanced" if split is None else split).strip().lower(),
        )


@dataclass(frozen=True)
class PipelinePlan:
    """Contiguous main-block ownership for one model-family topology."""

    family: str
    block_container: str
    num_blocks: int
    stages: int
    ranges: tuple[tuple[int, int], ...]

    def range_for_stage(self, stage_index: int) -> tuple[int, int]:
        if not 0 <= stage_index < self.stages:
            raise ValueError(
                f"stage_index must be in [0, {self.stages}), got {stage_index}"
            )
        return self.ranges[stage_index]

    def indices_for_stage(self, stage_index: int) -> tuple[int, ...]:
        start, end = self.range_for_stage(stage_index)
        return tuple(range(start, end))


def _pipeline_spec(family: object) -> tuple[str, PipelineParallelFamilySpec]:
    canonical = normalize_registered_family(
        family,
        source="pipeline_parallel model_family",
        allow_aliases=True,
    )
    family_spec = get_model_family_spec(canonical)
    pipeline_spec = family_spec.pipeline_parallel
    if pipeline_spec is None:
        raise ValueError(
            f"pipeline_parallel is not configurable for model_family={canonical}"
        )
    return canonical, pipeline_spec


def validate_pipeline_parallel_config(
    config: Mapping[str, Any] | object,
    *,
    world_size: int | None = None,
    num_blocks: int | None = None,
) -> PipelineParallelConfig:
    """Validate a registered family's PP contract without starting workers."""

    normalized = PipelineParallelConfig.from_config(config)
    if not normalized.enabled:
        return normalized

    family_value = _get(config, "model_family", "anima") or "anima"
    canonical, pipeline_spec = _pipeline_spec(family_value)
    # Per-stage block swap is implemented by exactly one family.  Without this
    # guard a config that sets it for any other family would be silently
    # ignored -- the worst failure mode, because the user believes memory was
    # freed while forward still OOMs.  Checked here (not in the runtime) so the
    # preflight / compat layer rejects it before any worker or NCCL starts, and
    # because every family's PipelineRuntimeConfig.from_config() funnels through
    # this function as well.
    if normalized.blocks_to_swap and not pipeline_spec.supports_stage_block_swap:
        raise ValueError(
            "pipeline_parallel_blocks_to_swap is only implemented for "
            f"krea2/krea2_raw, but model_family={canonical!r} sets it to "
            f"{normalized.blocks_to_swap}; set it to 0 or use krea2_raw"
        )
    family_spec = get_model_family_spec(canonical)
    block_count = _int(
        num_blocks,
        pipeline_spec.default_num_blocks,
        field="num_blocks",
    )

    if normalized.stages not in pipeline_spec.supported_stages:
        supported = ", ".join(map(str, sorted(pipeline_spec.supported_stages)))
        if pipeline_spec.supported_stages == frozenset({2}):
            raise ValueError(
                "pipeline_parallel_stages must be 2 in the current dual-GPU implementation"
            )
        raise ValueError(f"pipeline_parallel_stages must be one of: {supported}")
    if normalized.stages > block_count:
        raise ValueError(
            f"pipeline_parallel_stages={normalized.stages} exceeds "
            f"{family_spec.display_name} block count {block_count}"
        )
    if not 1 <= normalized.microbatches <= MAX_PIPELINE_PARALLEL_MICROBATCHES:
        raise ValueError(
            "pipeline_parallel_microbatches must be between 1 and "
            f"{MAX_PIPELINE_PARALLEL_MICROBATCHES}"
        )
    if normalized.schedule not in pipeline_spec.supported_schedules:
        raise ValueError(
            "pipeline_parallel_schedule must be one of: "
            + ", ".join(sorted(pipeline_spec.supported_schedules))
        )
    if normalized.split not in pipeline_spec.supported_splits:
        raise ValueError(
            "pipeline_parallel_split must be one of: "
            + ", ".join(sorted(pipeline_spec.supported_splits))
        )
    parsed_world_size = (
        None if world_size is None else _int(world_size, 1, field="world_size")
    )
    if parsed_world_size is not None and parsed_world_size != normalized.stages:
        raise ValueError(
            "pipeline_parallel_stages must equal the distributed world size: "
            f"stages={normalized.stages}, world_size={parsed_world_size}"
        )

    unsupported = _unsupported_combinations(config)
    if unsupported:
        raise ValueError(
            f"{family_spec.display_name} pipeline_parallel currently cannot be combined with: "
            + ", ".join(unsupported)
        )
    return normalized


def _unsupported_combinations(
    config: Mapping[str, Any] | object,
) -> list[str]:
    unsupported: list[str] = []
    if _int(_get(config, "blocks_to_swap", 0), 0, field="blocks_to_swap") > 0:
        unsupported.append("blocks_to_swap")
    if _bool(_get(config, "torch_compile", False), False, field="torch_compile"):
        unsupported.append("torch_compile")
    if str(_get(config, "selective_checkpoint", "off") or "off").strip().lower() != "off":
        unsupported.append("selective_checkpoint")
    if _bool(
        _get(config, "cpu_offload_checkpointing", False),
        False,
        field="cpu_offload_checkpointing",
    ):
        unsupported.append("cpu_offload_checkpointing")
    if _bool(
        _get(config, "unsloth_offload_checkpointing", False),
        False,
        field="unsloth_offload_checkpointing",
    ):
        unsupported.append("unsloth_offload_checkpointing")
    if not _bool(
        _get(config, "network_train_unet_only", True),
        True,
        field="network_train_unet_only",
    ):
        unsupported.append("network_train_unet_only=false")
    return unsupported


def make_pipeline_plan(
    *,
    family: str,
    stages: int,
    num_blocks: int | None = None,
) -> PipelinePlan:
    """Split a family's main blocks into deterministic contiguous ranges."""

    canonical, pipeline_spec = _pipeline_spec(family)
    if stages < 1:
        raise ValueError(f"stages must be positive, got {stages}")
    block_count = _int(
        num_blocks,
        pipeline_spec.default_num_blocks,
        field="num_blocks",
    )
    if block_count < stages:
        raise ValueError(f"num_blocks={block_count} must be >= stages={stages}")
    if stages not in pipeline_spec.supported_stages:
        supported = ", ".join(map(str, sorted(pipeline_spec.supported_stages)))
        raise ValueError(f"stages must be one of {supported} for {canonical}")

    base, remainder = divmod(block_count, stages)
    widths = [base + (1 if index < remainder else 0) for index in range(stages)]
    offset = pipeline_spec.stage_zero_block_offset
    if stages == 2 and offset:
        adjusted = (widths[0] + offset, widths[1] - offset)
        if min(adjusted) >= 1:
            widths[:] = adjusted

    ranges: list[tuple[int, int]] = []
    start = 0
    for width in widths:
        end = start + width
        ranges.append((start, end))
        start = end
    return PipelinePlan(
        family=canonical,
        block_container=pipeline_spec.block_container,
        num_blocks=block_count,
        stages=stages,
        ranges=tuple(ranges),
    )


class BorrowedBlockStage(nn.Module):
    """Common ownership metadata for a contiguous borrowing probe wrapper."""

    def __init__(
        self,
        blocks: list[nn.Module] | tuple[nn.Module, ...],
        *,
        source_block_container: str,
        stage_index: int = 0,
        block_range: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        block_items = tuple(blocks)
        start, end = block_range or (0, len(block_items))
        if start < 0 or end < start or end - start != len(block_items):
            raise ValueError(
                "block_range must be a non-negative half-open range matching "
                f"the borrowed block count, got range=({start}, {end}), "
                f"blocks={len(block_items)}"
            )
        self.stage_index = stage_index
        self.block_range = (start, end)
        self.source_block_container = source_block_container
        self.blocks = nn.ModuleList(block_items)

    @property
    def global_block_indices(self) -> tuple[int, ...]:
        return tuple(range(*self.block_range))

    def global_state_dict_key(self, local_key: str) -> str:
        """Map a wrapper-local ``blocks.N`` key back to the source-model key."""

        parts = local_key.split(".", 2)
        if len(parts) < 2 or parts[0] != "blocks":
            raise ValueError(
                f"stage state key must start with 'blocks.N', got {local_key!r}"
            )
        try:
            local_index = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"stage state key has invalid block index: {local_key!r}"
            ) from exc
        if not 0 <= local_index < len(self.blocks):
            raise ValueError(
                f"stage state key block index is out of range: {local_key!r}"
            )
        suffix = f".{parts[2]}" if len(parts) == 3 else ""
        global_index = self.block_range[0] + local_index
        return f"{self.source_block_container}.{global_index}{suffix}"

    def state_dict_key_map(self) -> dict[str, str]:
        return {key: self.global_state_dict_key(key) for key in self.state_dict()}


def build_pipeline_block_stage(
    model: nn.Module,
    *,
    family: str,
    stage_index: int,
    stages: int,
) -> tuple[PipelinePlan, BorrowedBlockStage]:
    """Build one family-specific borrowing wrapper from registry topology."""

    from library.anima.pipeline_parallel import AnimaBlockStage
    from library.models.krea2_raw.pipeline_parallel import Krea2BlockStage
    from library.models.moss_image.pipeline_parallel import MossImageBlockStage
    from library.models.z_image.pipeline_parallel import ZImageBlockStage

    canonical, pipeline_spec = _pipeline_spec(family)
    family_spec = get_model_family_spec(canonical)
    stage_type = dispatch_model_family(
        canonical,
        operation="pipeline block stage",
        handlers={
            "anima": AnimaBlockStage,
            "krea2_raw": Krea2BlockStage,
            "z_image": ZImageBlockStage,
            "moss_image": MossImageBlockStage,
            "qwen_image_2_1": lambda *_args, **_kwargs: (_ for _ in ()).throw(
                NotImplementedError("Qwen Image 2.1 pipeline parallel is not implemented")
            ),
        },
    )
    blocks = getattr(model, pipeline_spec.block_container, None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError(
            f"{family_spec.display_name} pipeline model must expose a "
            f"{pipeline_spec.block_container} ModuleList"
        )
    plan = make_pipeline_plan(
        family=canonical,
        stages=stages,
        num_blocks=len(blocks),
    )
    start, end = plan.range_for_stage(stage_index)
    stage = stage_type(
        tuple(blocks[start:end]),
        stage_index=stage_index,
        block_range=(start, end),
    )
    return plan, stage


__all__ = [
    "BorrowedBlockStage",
    "MAX_PIPELINE_PARALLEL_MICROBATCHES",
    "MAX_PIPELINE_PARALLEL_STAGES",
    "PIPELINE_PARALLEL_SCHEDULES",
    "PIPELINE_PARALLEL_SPLITS",
    "PipelineParallelConfig",
    "PipelinePlan",
    "build_pipeline_block_stage",
    "make_pipeline_plan",
    "validate_pipeline_parallel_config",
]
