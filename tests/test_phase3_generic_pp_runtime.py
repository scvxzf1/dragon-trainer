"""Phase 3 (P3-3): GenericPipelineRuntime + PpFamilyPlugin contract tests."""

from __future__ import annotations

from typing import Any, Sequence

import pytest
import torch
from torch import Tensor, nn

from library.training.pp.contract import PipelineRuntimeConfig, PpComputePolicy
from library.training.pp.errors import PipelineParallelError
from library.training.pp.generic_runtime import (
    GenericPipelineRuntime,
    PpFamilyPlugin,
    build_generic_pipeline_runtime,
)
from library.training.pp.krea2_runtime import Krea2PipelineRuntime
from library.training.pp.moss_runtime import MossPipelineRuntime
from library.training.pp.runtime import AnimaPipelineRuntime
from library.training.pp.stages import FrozenBoundaryPipelineStage


def test_all_three_family_runtimes_inherit_generic_pipeline_runtime() -> None:
    assert issubclass(AnimaPipelineRuntime, GenericPipelineRuntime)
    assert issubclass(MossPipelineRuntime, GenericPipelineRuntime)
    assert issubclass(Krea2PipelineRuntime, GenericPipelineRuntime)


class _ToyStageLayout:
    def __init__(self, stage_index: int, stages: int = 2) -> None:
        self.stage_index = stage_index
        self.stages = stages
        self.is_first = stage_index == 0
        self.is_last = stage_index == stages - 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": "toy_family",
            "stage_index": self.stage_index,
            "stages": self.stages,
            "block_range": [self.stage_index, self.stage_index + 1],
            "num_blocks": 2,
        }


class _ToyStageModule(nn.Module):
    def __init__(self, linear: nn.Linear, *, is_first: bool, is_last: bool) -> None:
        super().__init__()
        self.linear = linear
        self.is_first = is_first
        self.is_last = is_last
        self.adapter_modules = ()

    def trainable_parameters(self):
        return self.linear.parameters()

    def forward(self, x: Tensor, *, scale_factor: Tensor | None = None) -> Tensor:
        del scale_factor
        return self.linear(x)


class _ToyPpPlugin(PpFamilyPlugin):
    family = "toy_family"
    negotiation_prefix = "pp_toy_build"
    negotiate_build = True

    def resolve_wire_spec(
        self,
        model: Any,
        *,
        config: Any,
        latent_shape: Sequence[int],
        max_text_tokens: int | None = None,
        **extra: Any,
    ) -> tuple[int, ...]:
        del model, max_text_tokens, extra
        mb = int(latent_shape[0]) // int(config.microbatches)
        return (mb, int(latent_shape[1]))

    def build_stage_module(
        self,
        model: Any,
        *,
        rank: int,
        config: Any,
        wire: Any,
        network: Any,
        policy: PpComputePolicy,
        **extra: Any,
    ) -> tuple[_ToyStageLayout, _ToyStageModule]:
        del wire, network, policy, extra
        layout = _ToyStageLayout(rank, int(config.stages))
        return layout, _ToyStageModule(
            model.layers[rank], is_first=layout.is_first, is_last=layout.is_last
        )

    def build_pipeline_stage(
        self,
        model: Any,
        *,
        layout: Any,
        stage_module: Any,
        config: Any,
        rank: int,
        device: torch.device,
        wire: Any,
        policy: PpComputePolicy,
        group: Any,
        **extra: Any,
    ) -> FrozenBoundaryPipelineStage:
        del model, extra
        meta = torch.empty(wire, dtype=policy.activation_dtype, device="meta")
        return FrozenBoundaryPipelineStage(
            stage_module,
            rank,
            int(config.stages),
            device,
            input_args=(meta,),
            output_args=(meta,),
            group=group,
        )

    def extra_topology(self, runtime: GenericPipelineRuntime) -> dict[str, Any]:
        return {"wire_shape": list(runtime.wire)}


def test_minimal_four_hook_plugin_builds_and_validates_preconditions() -> None:
    class _ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])

    model = _ToyModel()
    cfg = PipelineRuntimeConfig(stages=2, microbatches=2, mixed_precision="fp16")
    plugin = _ToyPpPlugin()

    # Non-last stage owning a GradScaler must fail closed with pp_stage_contract_violation
    with pytest.raises(PipelineParallelError) as exc_info:
        build_generic_pipeline_runtime(
            plugin,
            GenericPipelineRuntime,
            model,
            config=cfg,
            rank=0,
            world_size=2,
            device=torch.device("cpu"),
            latent_shape=(4, 8),
            scaler=object(),
        )
    assert exc_info.value.code == "pp_stage_contract_violation"
