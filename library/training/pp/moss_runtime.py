"""MOSS-Image stage-local PP v0 runtime backed by GenericPipelineRuntime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from library.models.moss_image.pipeline_parallel import (
    MOSS_LATENT_CHANNELS,
    MossStageLayout,
    _validate_text_wire,
    build_moss_stage_module,
    resolve_wire_capacity,
    validate_moss_latent_shape,
)

from .contract import PpComputePolicy
from .errors import (
    CODE_SHAPE_CONTRACT,
    CODE_STAGE_CONTRACT,
    CODE_TOKEN_BUCKET_MISMATCH,
    PipelineParallelError,
)
from .generic_runtime import (
    GenericPipelineRuntime,
    PpFamilyPlugin,
    build_generic_pipeline_runtime,
)
from .stages import FrozenBoundaryPipelineStage

#: Boundary slots that carry no gradient: ``(lengths, mod)``.
MOSS_BOUNDARY_FROZEN_INDICES: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class MossWireSpec:
    """The frozen per-run wire geometry (per-microbatch shapes)."""

    latent_shape: tuple[int, ...]
    microbatch_latent_shape: tuple[int, ...]
    max_text_tokens: int
    token_cap: int
    dim: int

    @property
    def batch_size(self) -> int:
        return int(self.latent_shape[0])

    @property
    def microbatch_batch_size(self) -> int:
        return int(self.microbatch_latent_shape[0])


def resolve_moss_wire_spec(
    model: Any,
    *,
    latent_shape: Sequence[int],
    max_text_tokens: int,
    microbatches: int,
) -> MossWireSpec:
    """Validate the run geometry and freeze the microbatch wire shapes."""

    latent = validate_moss_latent_shape(latent_shape)
    if int(latent[0]) % int(microbatches):
        raise PipelineParallelError(
            CODE_TOKEN_BUCKET_MISMATCH,
            f"train batch size {latent[0]} is not divisible by "
            f"pipeline_parallel_microbatches={microbatches}; PP v0 freezes the "
            "wire buffers so every microbatch must be the same size",
        )
    microbatch = (int(latent[0]) // int(microbatches), *latent[1:])
    dim = int(getattr(model.video_embedder, "out_features"))
    token_cap = resolve_wire_capacity(int(latent[2]), int(latent[3]), max_text_tokens)
    return MossWireSpec(
        latent_shape=latent,
        microbatch_latent_shape=microbatch,
        max_text_tokens=int(max_text_tokens),
        token_cap=int(token_cap),
        dim=dim,
    )


def moss_boundary_metas(
    model: Any,
    layout: MossStageLayout,
    *,
    wire: MossWireSpec,
    policy: PpComputePolicy,
    latent_dtype: torch.dtype,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """``(input_args, output_args)`` metadata for this MOSS stage."""

    del model

    def _meta(shape: Sequence[int], dtype: torch.dtype) -> Tensor:
        return torch.empty(tuple(int(value) for value in shape), dtype=dtype, device="meta")

    dtype = policy.activation_dtype
    mb = wire.microbatch_latent_shape
    hidden_meta = _meta((mb[0], wire.token_cap, wire.dim), dtype)
    lengths_meta = _meta((mb[0],), torch.float32)
    mod_meta = _meta((mb[0], 6, wire.dim), dtype)
    boundary = (hidden_meta, lengths_meta, mod_meta)
    if layout.is_first:
        return (_meta(mb, latent_dtype),), boundary
    prediction_meta = _meta(
        (mb[0], MOSS_LATENT_CHANNELS, mb[2], mb[3]),
        dtype,
    )
    return boundary, (prediction_meta,)


class MossPpPlugin(PpFamilyPlugin):
    """MOSS-Image PP v0 family plugin hooks over GenericPipelineRuntime."""

    family = "moss_image"
    negotiation_prefix = "pp_moss_build"
    negotiate_build = True
    non_last_scaler_policy = "forbid"
    require_nonempty_trainable_for_factory = True

    def resolve_wire_spec(
        self,
        model: Any,
        *,
        config: Any,
        latent_shape: Sequence[int],
        max_text_tokens: int | None = None,
        **extra: Any,
    ) -> MossWireSpec:
        del extra
        assert max_text_tokens is not None
        return resolve_moss_wire_spec(
            model,
            latent_shape=latent_shape,
            max_text_tokens=max_text_tokens,
            microbatches=int(config.microbatches),
        )

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
    ) -> tuple[MossStageLayout, Any]:
        del wire
        return build_moss_stage_module(
            model,
            stage_index=int(rank),
            stages=int(config.stages),
            latent_shape=extra["latent_shape"],
            max_text_tokens=int(extra["max_text_tokens"]),
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
        del extra
        input_args, output_args = moss_boundary_metas(
            model,
            layout,
            wire=wire,
            policy=policy,
            latent_dtype=policy.activation_dtype,
        )
        return FrozenBoundaryPipelineStage(
            stage_module,
            int(rank),
            int(config.stages),
            device,
            input_args=tuple(input_args),
            output_args=tuple(output_args),
            group=group,
            frozen_input_indices=() if layout.is_first else MOSS_BOUNDARY_FROZEN_INDICES,
            frozen_output_indices=() if layout.is_last else MOSS_BOUNDARY_FROZEN_INDICES,
        )

    def extra_topology(self, runtime: GenericPipelineRuntime) -> dict[str, Any]:
        wire: MossWireSpec = runtime.wire
        return {
            "latent_channels": MOSS_LATENT_CHANNELS,
            "token_cap": wire.token_cap,
            "max_text_tokens": wire.max_text_tokens,
            "dim": wire.dim,
            "dtype": str(runtime._dtype),
            "wire_dtype": str(runtime.compute_policy.activation_dtype),
            "compute_policy": runtime.compute_policy.as_dict(),
            "frozen_input_indices": list(runtime.stage.frozen_input_indices),
            "frozen_output_indices": list(runtime.stage.frozen_output_indices),
            "adapter_names": sorted(
                module.lora_name for module in runtime.stage_module.adapter_modules
            ),
        }

    def precheck_step(
        self,
        runtime: GenericPipelineRuntime,
        latents: Tensor | None,
        step_kwargs: dict[str, Any],
        target: Tensor | None,
    ) -> None:
        assert isinstance(runtime, MossPipelineRuntime)
        runtime._negotiate_step_precheck(
            latents,
            step_kwargs.get("text_hiddens"),
            step_kwargs.get("text_mask"),
            step_kwargs.get("timesteps"),
            target,
        )
        if runtime.is_first and latents is not None:
            runtime._require_shape(latents)

    def prepare_schedule_inputs(
        self,
        runtime: GenericPipelineRuntime,
        step_kwargs: dict[str, Any],
        target: Tensor | None,
    ) -> tuple[dict[str, Any], Tensor | None]:
        if runtime.is_first:
            return step_kwargs, target
        return {"timesteps": step_kwargs["timesteps"]}, target


_MOSS_PP_PLUGIN = MossPpPlugin()


class MossPipelineRuntime(GenericPipelineRuntime):
    """One rank's MOSS PP runtime: stage, schedule and fp16 transaction."""

    _default_plugin = _MOSS_PP_PLUGIN

    def _negotiate_step_precheck(
        self,
        latents: Tensor | None,
        text_hiddens: Tensor | None,
        text_mask: Tensor | None,
        timesteps: Tensor | None,
        target: Tensor | None = None,
    ) -> None:
        """Share the per-step verdict before any rank enters the schedule."""

        import torch.distributed as dist

        error: tuple[str, str] | None = None
        try:
            if timesteps is None:
                raise PipelineParallelError(
                    CODE_STAGE_CONTRACT,
                    "every MOSS PP stage needs the batch timesteps for the "
                    "block modulation (rank 0 included)",
                )
            if torch.as_tensor(timesteps).numel() != self._wire.batch_size:
                raise PipelineParallelError(
                    CODE_SHAPE_CONTRACT,
                    f"timesteps must carry one value per sample "
                    f"({self._wire.batch_size}), got "
                    f"{torch.as_tensor(timesteps).numel()}",
                )
            if self.is_first:
                if latents is None or text_hiddens is None or text_mask is None:
                    raise PipelineParallelError(
                        CODE_STAGE_CONTRACT,
                        "the first MOSS stage needs the batch latents, "
                        "text_hiddens and text_mask",
                    )
                shape = validate_moss_latent_shape(latents.shape)
                if shape != self._wire.latent_shape:
                    raise PipelineParallelError(
                        CODE_TOKEN_BUCKET_MISMATCH,
                        f"this MOSS PP runtime serves exactly one whole-step "
                        f"latent shape {self._wire.latent_shape} (microbatches "
                        f"{self._config.microbatches} -> per-microbatch "
                        f"{self._wire.microbatch_latent_shape}); got {shape}",
                    )
                if latents.dtype != self._compute_policy.activation_dtype:
                    raise PipelineParallelError(
                        "pp_compute_policy_mismatch",
                        f"the pre-block latent dtype is {latents.dtype} but the "
                        f"pipeline compute policy was built for "
                        f"{self._compute_policy.activation_dtype}",
                    )
                _validate_text_wire(
                    text_hiddens, text_mask, shape[0], self._wire.max_text_tokens
                )
                image_tokens = shape[2] * shape[3]
                lengths = text_mask.bool().sum(dim=1).detach().to(device="cpu").tolist()
                for index, raw_len in enumerate(lengths):
                    length = int(raw_len)
                    if length < 1:
                        raise PipelineParallelError(
                            CODE_STAGE_CONTRACT,
                            f"sample {index} has no valid text token; the "
                            "production MOSS forward rejects the same case",
                        )
                    if length > self._wire.max_text_tokens:
                        raise PipelineParallelError(
                            CODE_TOKEN_BUCKET_MISMATCH,
                            f"sample {index} text length {length} exceeds the "
                            f"built wire capacity {self._wire.max_text_tokens}; "
                            "build one runtime per max_text_tokens",
                        )
                    if image_tokens + length > self._wire.token_cap:
                        raise PipelineParallelError(
                            CODE_TOKEN_BUCKET_MISMATCH,
                            f"sample {index} needs {image_tokens + length} joint "
                            f"tokens, above the built token_cap "
                            f"{self._wire.token_cap}",
                        )
            else:
                if target is None:
                    raise PipelineParallelError(
                        CODE_STAGE_CONTRACT,
                        "the last MOSS stage needs the batch target to compute "
                        "the loss",
                    )
                if tuple(target.shape) != tuple(self._wire.latent_shape):
                    raise PipelineParallelError(
                        CODE_SHAPE_CONTRACT,
                        f"target shape {tuple(target.shape)} does not match the "
                        f"built latent shape {self._wire.latent_shape}",
                    )
        except PipelineParallelError as exc:
            error = (exc.code, exc.detail)
        if not (dist.is_available() and dist.is_initialized()):
            if error is not None:
                raise PipelineParallelError(*error)
            return
        gathered: list[Any] = [None] * dist.get_world_size(self._group)
        dist.all_gather_object(gathered, error, group=self._group)
        failures = [item for item in gathered if item]
        if failures:
            code, detail = failures[0]
            raise PipelineParallelError(
                code,
                "negotiated step refusal (both ranks raise before any "
                f"communication): {detail}",
            )

    def step(
        self,
        latents: Tensor | None,
        *,
        timesteps: Tensor,
        text_hiddens: Tensor | None = None,
        text_mask: Tensor | None = None,
        target: Tensor | None = None,
        losses: list | None = None,
    ):
        """Run one 1F1B step for the whole batch."""

        return self._execute_step(
            latents,
            step_kwargs={
                "timesteps": timesteps,
                "text_hiddens": text_hiddens,
                "text_mask": text_mask,
            },
            target=target,
            losses=losses,
        )

    def _require_shape(self, latents: Tensor) -> None:
        shape = tuple(int(value) for value in latents.shape)
        if shape == self._wire.latent_shape:
            return
        raise PipelineParallelError(
            CODE_TOKEN_BUCKET_MISMATCH,
            f"this MOSS PP runtime serves exactly one whole-step latent shape "
            f"{self._wire.latent_shape} (microbatches "
            f"{self._config.microbatches} -> per-microbatch "
            f"{self._wire.microbatch_latent_shape}); got {shape}. Build one "
            "runtime per (batch, H, W) wire shape -- tail batches and mixed "
            "shapes are not supported by PP v0",
        )


def build_moss_pipeline_runtime(
    model: Any,
    *,
    config: Any,
    rank: int,
    world_size: int,
    device: torch.device,
    latent_shape: Sequence[int],
    max_text_tokens: int,
    dtype: torch.dtype | None = None,
    network: Any = None,
    scaler: Any = None,
    optimizer: Any = None,
    optimizer_factory: Any = None,
    max_grad_norm: float | None = None,
    clip_gradients: Any = None,
    group: Any = None,
) -> MossPipelineRuntime:
    """Build this rank's MOSS PP v0 runtime, or fail closed with a named code."""

    return build_generic_pipeline_runtime(
        _MOSS_PP_PLUGIN,
        MossPipelineRuntime,
        model,
        config=config,
        rank=rank,
        world_size=world_size,
        device=device,
        latent_shape=latent_shape,
        max_text_tokens=max_text_tokens,
        dtype=dtype,
        network=network,
        scaler=scaler,
        optimizer=optimizer,
        optimizer_factory=optimizer_factory,
        max_grad_norm=max_grad_norm,
        clip_gradients=clip_gradients,
        group=group,
    )


__all__ = [
    "MOSS_BOUNDARY_FROZEN_INDICES",
    "MossPipelineRuntime",
    "MossWireSpec",
    "build_moss_pipeline_runtime",
    "moss_boundary_metas",
    "resolve_moss_wire_spec",
]
