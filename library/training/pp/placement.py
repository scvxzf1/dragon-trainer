"""Non-destructive Anima PP=2 stage placement.

The placement itself is **not** re-implemented here: the contiguous block split
comes from ``library.models.pipeline_parallel.make_pipeline_plan`` and the block
slice wrapper from ``library.anima.pipeline_parallel.build_anima_block_stage`` /
``AnimaBlockStage``.  What this module adds is the two things the experimental
probe got wrong:

1. **Non-destructive ownership.**  ``scripts/experiments/anima_parallel/stages.py``
   deletes the foreign submodules from the model in place (``delattr``), which
   makes the model unusable for anything else and is unrecoverable if placement
   is retried.  Here the stage only *references* the model and the slice; the
   model is left intact unless the caller explicitly asks for the memory back
   via :func:`release_foreign_blocks`.
2. **Startup-time capability checks.**  Everything the split forward cannot
   reproduce is rejected before the first step (pooled text modulation, flex
   attention, native-flatten compile, per-block modulation guidance on a sliced
   stage, block swap) instead of silently producing different numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from torch import nn

from library.anima.pipeline_parallel import build_anima_block_stage
from library.models.pipeline_parallel import PipelinePlan, make_pipeline_plan

from .contract import PP_CONDITIONING_MODES
from .errors import CODE_STAGE_CONTRACT, PipelineParallelError
from .numerics import (
    AnimaPipelineUnsupported,
    assert_mini_forward_supported,
    assert_mod_guidance_indexable,
    mod_guidance_probe,
)
from .stages import (
    CONDITIONING_LOCAL,
    CONDITIONING_WIRE,
    AnimaFirstStage,
    AnimaLastStage,
)

#: ``lora_unet_blocks_<N>_...`` is the kohya-compatible adapter naming used by
#: ``networks.lora_anima``; it is the only place a block owner can be recovered
#: from, because a LoRA module is not a child of the block it patches.
_ADAPTER_BLOCK_RE = re.compile(r"lora_unet_blocks_(\d+)_")

#: Submodules the split forward reads directly from the production model.
REQUIRED_MODEL_ATTRIBUTES = (
    "blocks",
    "x_embedder",
    "pos_embedder",
    "t_embedder",
    "t_embedding_norm",
    "final_layer",
    "unpatchify",
    "prepare_embedded_sequence",
    "patch_spatial",
    "patch_temporal",
    "model_channels",
)


@dataclass(frozen=True)
class AnimaStageLayout:
    """Which contiguous block range one rank owns, and where it sits."""

    family: str
    stage_index: int
    stages: int
    block_range: tuple[int, int]
    num_blocks: int

    @property
    def is_first(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        return self.stage_index == self.stages - 1

    @property
    def block_indices(self) -> tuple[int, ...]:
        return tuple(range(self.block_range[0], self.block_range[1]))

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "stage_index": self.stage_index,
            "stages": self.stages,
            "block_range": list(self.block_range),
            "num_blocks": self.num_blocks,
        }


def layout_from_plan(plan: PipelinePlan, stage_index: int) -> AnimaStageLayout:
    """Project a :class:`PipelinePlan` onto one stage."""

    if not 0 <= int(stage_index) < plan.stages:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"stage_index must be in [0, {plan.stages}), got {stage_index}",
        )
    return AnimaStageLayout(
        family=plan.family,
        stage_index=int(stage_index),
        stages=int(plan.stages),
        block_range=plan.range_for_stage(int(stage_index)),
        num_blocks=int(plan.num_blocks),
    )


def build_anima_stage_layout(
    *, stage_index: int, stages: int, num_blocks: int
) -> AnimaStageLayout:
    """Block ownership for one stage, without touching a model."""

    plan = make_pipeline_plan(family="anima", stages=int(stages), num_blocks=int(num_blocks))
    return layout_from_plan(plan, stage_index)


def assert_stage_capabilities(model: nn.Module) -> None:
    """Reject any model the split forward cannot reproduce, at build time."""

    missing = [name for name in REQUIRED_MODEL_ATTRIBUTES if not hasattr(model, name)]
    if missing:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            "the PP split forward needs the production Anima model surface; missing "
            + ", ".join(missing),
        )
    if not isinstance(model.blocks, nn.ModuleList):
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"model.blocks must be an nn.ModuleList, got {type(model.blocks).__name__}",
        )
    swap = int(getattr(model, "blocks_to_swap", 0) or 0)
    if swap:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"pipeline_parallel cannot own a block-swap model (blocks_to_swap={swap}); "
            "the offloader owns module residency and the stage owns block ownership",
        )
    try:
        assert_mini_forward_supported(model)
    except AnimaPipelineUnsupported as exc:
        raise PipelineParallelError(exc.code, exc.detail) from exc


def assert_stage_mod_guidance_supported(model: nn.Module, blocks: Sequence[nn.Module]) -> None:
    """Fail closed on active per-block modulation guidance.

    ``_run_blocks`` looks ``_mod_guidance_schedule`` up with the *local* block
    index, but the schedule is stored per *global* block id, so a sliced stage
    would apply an active schedule to the wrong blocks.  PP v0 does not support
    it; the probe carries the slice explicitly because the placement builder
    leaves the full ``model.blocks`` in place.
    """

    try:
        assert_mod_guidance_indexable(mod_guidance_probe(model, list(blocks)))
    except AnimaPipelineUnsupported as exc:
        raise PipelineParallelError(exc.code, exc.detail) from exc


def adapter_block_index(adapter_name: str) -> int:
    """Global block id an adapter patches, from its kohya-style name."""

    match = _ADAPTER_BLOCK_RE.search(str(adapter_name))
    if match is None:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"cannot resolve a block id from adapter name {adapter_name!r}; PP "
            "placement needs adapters named lora_unet_blocks_<N>_...",
        )
    return int(match.group(1))


def stage_adapter_modules(network, layout: AnimaStageLayout) -> list[nn.Module]:
    """The adapter modules attached to the blocks this stage owns."""

    owned = set(layout.block_indices)
    adapters = getattr(network, "unet_loras", None)
    if adapters is None:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"adapter network {type(network).__name__} does not expose unet_loras",
        )
    return [
        adapter
        for adapter in adapters
        if adapter_block_index(adapter.lora_name) in owned
    ]


def build_anima_stage_module(
    model: nn.Module,
    *,
    stage_index: int,
    stages: int,
    conditioning: str = CONDITIONING_LOCAL,
    network=None,
    compute_policy=None,
) -> tuple[AnimaStageLayout, nn.Module]:
    """Build this rank's stage module over a contiguous Anima block range.

    ``compute_policy`` is the pipeline-wide
    :class:`~library.training.pp.contract.PpComputePolicy`.  It is optional so a
    single-stage unit test can build a module directly (the default resolves it
    from the model dtype), but the runtime always passes the shared instance:
    every stage must run with the same ``use_fp32``.
    """

    if conditioning not in PP_CONDITIONING_MODES:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"conditioning must be one of {', '.join(PP_CONDITIONING_MODES)}, got "
            f"{conditioning!r}",
        )
    assert_stage_capabilities(model)
    plan, block_stage = build_anima_block_stage(
        model, stage_index=int(stage_index), stages=int(stages)
    )
    layout = layout_from_plan(plan, stage_index)
    assert_stage_mod_guidance_supported(model, block_stage.blocks)
    if (
        layout.is_last
        and conditioning == CONDITIONING_WIRE
        and not bool(getattr(model, "use_adaln_lora", False))
    ):
        raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                "wire conditioning ships the AdaLN-LoRA modulation across the "
                "boundary, which requires use_adaln_lora=true",
            )
    adapters = () if network is None else tuple(stage_adapter_modules(network, layout))
    stage_type = AnimaFirstStage if layout.is_first else AnimaLastStage
    module = stage_type(
        model=model,
        block_stage=block_stage,
        stage_index=layout.stage_index,
        stages=layout.stages,
        block_range=layout.block_range,
        conditioning=conditioning,
        adapter_modules=adapters,
        compute_policy=compute_policy,
    )
    return layout, module


def release_foreign_blocks(model: nn.Module, layout: AnimaStageLayout) -> tuple[int, int]:
    """Drop the blocks this stage does not own, to reclaim their memory.

    This is the only destructive operation in the package and it is **opt-in**:
    call it after the stage module (which holds its own references to the owned
    blocks) and the optimizer are built.  It intentionally does not touch the
    pre-block / post-block submodules -- head/tail residency is a trainer policy
    decision, not a placement detail.
    """

    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"model.blocks must be an nn.ModuleList, got {type(blocks).__name__}",
        )
    start, end = layout.block_range
    if (start, end) == (0, len(blocks)):
        return (start, end)
    model.blocks = nn.ModuleList([blocks[index] for index in range(start, end)])
    return (start, end)


__all__ = [
    "REQUIRED_MODEL_ATTRIBUTES",
    "AnimaStageLayout",
    "adapter_block_index",
    "assert_stage_capabilities",
    "assert_stage_mod_guidance_supported",
    "build_anima_stage_layout",
    "build_anima_stage_module",
    "layout_from_plan",
    "release_foreign_blocks",
    "stage_adapter_modules",
]
