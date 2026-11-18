"""Top-level Anima pipeline-parallel v0 runtime backed by GenericPipelineRuntime."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .contract import (
    PipelineRuntimeConfig,
    PpComputePolicy,
    require_pipelining_api,
    validate_no_quantization,
)
from .errors import (
    CODE_SHAPE_CONTRACT,
    CODE_TOKEN_BUCKET_MISMATCH,
    PipelineParallelError,
)
from .generic_runtime import (
    GenericPipelineRuntime,
    PpFamilyPlugin,
    PpStepResult,
    build_generic_pipeline_runtime,
)
from .layout import (
    boundary_hidden_shape,
    split_batch_shape,
    token_family,
    validate_latent_5d_shape,
)
from .placement import (
    AnimaStageLayout,
    build_anima_stage_module,
)
from .schedule import (
    boundary_metas,
    build_pipeline_stage,
)
from .stages import CONDITIONING_WIRE, FrozenBoundaryPipelineStage


def _boundary_shape_of(model: Any, latent_shape: Sequence[int]) -> tuple[int, ...]:
    try:
        return boundary_hidden_shape(
            latent_shape,
            patch_spatial=int(model.patch_spatial),
            patch_temporal=int(model.patch_temporal),
            model_channels=int(model.model_channels),
        )
    except PipelineParallelError:
        return tuple(int(value) for value in latent_shape)


class AnimaPpPlugin(PpFamilyPlugin):
    """Anima PP v0 family plugin hooks over GenericPipelineRuntime."""

    family = "anima"
    negotiation_prefix = "pp_build"
    negotiate_build = False
    non_last_scaler_policy = "drop"
    require_nonempty_trainable_for_factory = False

    def resolve_wire_spec(
        self,
        model: Any,
        *,
        config: Any,
        latent_shape: Sequence[int],
        max_text_tokens: int | None = None,
        **extra: Any,
    ) -> None:
        del model, config, latent_shape, max_text_tokens, extra
        return None

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
    ) -> tuple[AnimaStageLayout, Any]:
        del wire, extra
        return build_anima_stage_module(
            model,
            stage_index=int(rank),
            stages=int(config.stages),
            conditioning=config.conditioning,
            network=network,
            compute_policy=policy,
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
        del rank, wire
        timesteps_shape = extra.get("timesteps_shape")
        context_shape = extra.get("context_shape")
        latent_shape = extra["latent_shape"]
        if config.conditioning == CONDITIONING_WIRE and (
            timesteps_shape is None or context_shape is None
        ):
            raise PipelineParallelError(
                CODE_SHAPE_CONTRACT,
                "conditioning=wire needs timesteps_shape and context_shape",
            )
        input_args, output_args = boundary_metas(
            model,
            layout,
            latent_shape=latent_shape,
            microbatches=config.microbatches,
            policy=policy,
            conditioning=config.conditioning,
            timesteps_shape=timesteps_shape,
            context_shape=context_shape,
        )
        return build_pipeline_stage(
            stage_module,
            layout,
            device=device,
            input_args=input_args,
            output_args=output_args,
            conditioning=config.conditioning,
            group=group,
        )

    def extra_topology(self, runtime: GenericPipelineRuntime) -> dict[str, Any]:
        assert isinstance(runtime, AnimaPipelineRuntime)
        return {
            "token_family": token_family(runtime._boundary_shape()),
            "frozen_input_indices": list(runtime.stage.frozen_input_indices),
            "frozen_output_indices": list(runtime.stage.frozen_output_indices),
        }

    def precheck_step(
        self,
        runtime: GenericPipelineRuntime,
        latents: Tensor | None,
        step_kwargs: dict[str, Any],
        target: Tensor | None,
    ) -> None:
        del step_kwargs, target
        assert isinstance(runtime, AnimaPipelineRuntime) and latents is not None
        runtime._require_shape(latents)

    def format_step_result(
        self,
        runtime: GenericPipelineRuntime,
        prediction: Any,
        bucket: list[Any],
    ) -> PpStepResult:
        return PpStepResult(
            prediction=prediction,
            losses=tuple(float(value.detach()) for value in bucket),
            topology=runtime.topology(),
        )


_ANIMA_PP_PLUGIN = AnimaPpPlugin()


class AnimaPipelineRuntime(GenericPipelineRuntime):
    """One rank's Anima PP runtime: stage, schedule and fp16 transaction."""

    _default_plugin = _ANIMA_PP_PLUGIN

    def __init__(self, *, latent_shape: Sequence[int], **kwargs: Any) -> None:
        validated = validate_latent_5d_shape(latent_shape)
        super().__init__(latent_shape=validated, **kwargs)

    def adapter_key_map(self, local_key: str) -> str:
        return self._stage_module.adapter_key_map(local_key)

    def step(
        self,
        latents: Tensor,
        *,
        timesteps: Tensor,
        crossattn_emb: Tensor,
        target: Tensor | None = None,
        padding_mask: Tensor | None = None,
        fps: Tensor | None = None,
        h_offset: int = 0,
        w_offset: int = 0,
        losses: list | None = None,
    ) -> PpStepResult:
        """Run one 1F1B/GPipe step; only the last stage returns a prediction."""

        kwargs: dict[str, Any] = {
            "timesteps": timesteps,
            "crossattn_emb": crossattn_emb,
        }
        if padding_mask is not None:
            kwargs["padding_mask"] = padding_mask
        if fps is not None:
            kwargs["fps"] = fps
        if h_offset:
            kwargs["h_offset"] = int(h_offset)
        if w_offset:
            kwargs["w_offset"] = int(w_offset)
        return self._execute_step(
            latents, step_kwargs=kwargs, target=target, losses=losses
        )

    def _boundary_shape(self) -> tuple[int, ...]:
        latent_mb = split_batch_shape(self._latent_shape, self._config.microbatches)
        return boundary_hidden_shape(
            latent_mb,
            patch_spatial=int(self._model.patch_spatial),
            patch_temporal=int(self._model.patch_temporal),
            model_channels=int(self._model.model_channels),
        )

    def _require_shape(self, latents: Tensor) -> None:
        shape = tuple(int(value) for value in latents.shape)
        if shape == self._latent_shape:
            return
        raise PipelineParallelError(
            CODE_TOKEN_BUCKET_MISMATCH,
            "this PP runtime serves exactly one latent shape (the pipeline wire "
            f"buffers are frozen at {self._latent_shape}, family "
            f"{token_family(self._boundary_shape())}); got {shape} (family "
            f"{token_family(_boundary_shape_of(self._model, shape))}). Build one "
            "runtime per token bucket -- mixing buckets in one PP run is not "
            "supported by PP v0",
        )


def build_anima_pipeline_runtime(
    model: Any,
    *,
    config: PipelineRuntimeConfig,
    world_size: int,
    rank: int,
    device: torch.device,
    latent_shape: Sequence[int],
    dtype: torch.dtype | None = None,
    timesteps_shape: Sequence[int] | None = None,
    context_shape: Sequence[int] | None = None,
    network: Any = None,
    scaler: Any = None,
    optimizer: Any = None,
    optimizer_factory: Any = None,
    max_grad_norm: float | None = None,
    clip_gradients: Any = None,
    loss_fn: Any = None,
    group: Any = None,
) -> AnimaPipelineRuntime:
    """Build this rank's Anima PP v0 runtime, or fail closed with a named code."""

    return build_generic_pipeline_runtime(
        _ANIMA_PP_PLUGIN,
        AnimaPipelineRuntime,
        model,
        config=config,
        rank=rank,
        world_size=world_size,
        device=device,
        latent_shape=latent_shape,
        dtype=dtype,
        network=network,
        scaler=scaler,
        optimizer=optimizer,
        optimizer_factory=optimizer_factory,
        max_grad_norm=max_grad_norm,
        clip_gradients=clip_gradients,
        loss_fn=loss_fn,
        group=group,
        timesteps_shape=timesteps_shape,
        context_shape=context_shape,
    )


def assert_runtime_capabilities(config: Mapping[str, Any] | object) -> None:
    """Convenience preflight: quantization + capability checks, no construction."""

    require_pipelining_api()
    validate_no_quantization(config)


__all__ = [
    "AnimaPipelineRuntime",
    "PpStepResult",
    "assert_runtime_capabilities",
    "build_anima_pipeline_runtime",
]
