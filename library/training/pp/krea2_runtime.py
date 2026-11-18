"""Krea-2 PP v0 runtime backed by GenericPipelineRuntime."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Sequence

import torch
from torch import Tensor

from library.models.krea2_raw.pipeline_parallel import make_krea2_pipeline_plan
from library.models.krea2_raw.pp_stages import (
    Krea2FirstStage,
    Krea2LastStage,
    enable_block_probes,
    enable_stage_block_swap,
    krea2_token_cap,
    release_foreign_submodules,
)
from library.training.pp.contract import PpComputePolicy
from library.training.pp.errors import CODE_STAGE_CONTRACT, PipelineParallelError
from library.training.pp.generic_runtime import (
    GenericPipelineRuntime,
    PpFamilyPlugin,
    build_generic_pipeline_runtime,
)
from library.training.pp.stages import FrozenBoundaryPipelineStage

__all__ = [
    "KREA2_BOUNDARY_FROZEN_INDICES",
    "KREA2_LOCAL_FREQS_FROZEN_INDICES",
    "Krea2PipelineRuntime",
    "Krea2StageLayout",
    "Krea2WireSpec",
    "adapter_modules_for_stage",
    "build_krea2_pipeline_runtime",
    "build_krea2_stage_module",
    "krea2_boundary_metas",
    "resolve_krea2_wire_spec",
]

#: Wire slots that carry no gradient in legacy 5-slot probe mode.
KREA2_BOUNDARY_FROZEN_INDICES: tuple[int, ...] = (1, 2, 3, 4)
#: Wire slots that carry no gradient when ``freqs`` is cached locally (4-slot wire).
KREA2_LOCAL_FREQS_FROZEN_INDICES: tuple[int, ...] = (1, 2, 3)

#: Latent channels of Krea-2 (patchified into the token stream).
KREA2_LATENT_CHANNELS = 16


@dataclass(frozen=True)
class Krea2WireSpec:
    """The frozen per-run wire geometry (per-microbatch shapes)."""

    latent_shape: tuple[int, ...]
    microbatch_latent_shape: tuple[int, ...]
    max_text_tokens: int
    token_cap: int
    dim: int
    img_tokens: int
    heads: int
    patch: int

    @property
    def batch_size(self) -> int:
        return int(self.latent_shape[0])

    @property
    def microbatch_batch_size(self) -> int:
        return int(self.microbatch_latent_shape[0])

    @property
    def headdim(self) -> int:
        return int(self.dim) // int(self.heads)

    @property
    def freqs_shape(self) -> tuple[int, ...]:
        return (
            self.microbatch_batch_size,
            self.token_cap,
            self.headdim // 2,
            2,
            2,
        )


def resolve_krea2_wire_spec(
    model: Any,
    *,
    latent_shape: Sequence[int],
    max_text_tokens: int,
    microbatches: int,
) -> Krea2WireSpec:
    """Validate the run geometry and freeze the microbatch wire shapes."""

    latent = tuple(int(value) for value in latent_shape)
    if len(latent) != 5:
        raise PipelineParallelError(
            "pp_krea2_latent_shape_invalid",
            f"Krea-2 expects a 5D latent (B, C, T=1, H, W); got {latent}",
        )
    batch, channels, temporal, height, width = latent
    if channels != KREA2_LATENT_CHANNELS or temporal != 1:
        raise PipelineParallelError(
            "pp_krea2_latent_shape_invalid",
            f"Krea-2 expects (B, {KREA2_LATENT_CHANNELS}, 1, H, W); got {latent}",
        )
    from library.training.pp.krea2_batch import batch_divisibility_message

    batch_error = batch_divisibility_message(batch, int(microbatches))
    if batch_error:
        raise PipelineParallelError(
            "pp_krea2_batch_not_divisible",
            batch_error,
        )
    patch = int(getattr(model, "PATCH_SIZE", 2))
    if height % patch or width % patch:
        raise PipelineParallelError(
            "pp_krea2_latent_shape_invalid",
            f"latent {height}x{width} is not divisible by patch size {patch}",
        )
    img_tokens = (height // patch) * (width // patch)
    config = getattr(model, "config", None)
    dim = int(getattr(config, "features"))
    heads = int(getattr(config, "heads"))
    return Krea2WireSpec(
        latent_shape=latent,
        microbatch_latent_shape=(batch // int(microbatches), *latent[1:]),
        max_text_tokens=int(max_text_tokens),
        token_cap=krea2_token_cap(int(max_text_tokens), img_tokens),
        dim=dim,
        img_tokens=img_tokens,
        heads=heads,
        patch=patch,
    )


def _meta(shape: Sequence[int], dtype: torch.dtype) -> Tensor:
    return torch.empty(tuple(int(value) for value in shape), dtype=dtype, device="meta")


def krea2_boundary_metas(
    stage: Any,
    *,
    wire: Krea2WireSpec,
    policy: PpComputePolicy,
    latent_dtype: torch.dtype,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """``(input_args, output_args)`` metadata for this Krea-2 stage."""

    dtype = policy.activation_dtype
    mb = wire.microbatch_batch_size
    if bool(getattr(stage, "local_freqs", False)):
        boundary = (
            _meta((mb, wire.token_cap, wire.dim), dtype),
            _meta((mb, 1, 6 * wire.dim), dtype),
            _meta((mb, wire.token_cap), torch.float32),
            _meta((mb, 1, wire.dim), dtype),
        )
    else:
        boundary = (
            _meta((mb, wire.token_cap, wire.dim), dtype),
            _meta((mb, 1, 6 * wire.dim), dtype),
            _meta(wire.freqs_shape, torch.float32),
            _meta((mb, wire.token_cap), torch.float32),
            _meta((mb, 1, wire.dim), dtype),
        )
    if stage.is_first:
        return (_meta(wire.microbatch_latent_shape, latent_dtype),), boundary
    prediction = _meta(
        (mb, wire.img_tokens, wire.patch * wire.patch * KREA2_LATENT_CHANNELS), dtype
    )
    return boundary, (prediction,)


@dataclass(frozen=True)
class Krea2StageLayout:
    """Contiguous block ownership for one Krea-2 pipeline stage."""

    stage_index: int
    stages: int
    block_range: tuple[int, int]
    num_blocks: int
    local_freqs: bool = False

    @property
    def is_first(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        return self.stage_index == self.stages - 1

    @property
    def local_blocks(self) -> int:
        return self.block_range[1] - self.block_range[0]

    @property
    def block_indices(self) -> tuple[int, ...]:
        return tuple(range(self.block_range[0], self.block_range[1]))

    def as_dict(self) -> dict[str, object]:
        return {
            "family": "krea2_raw",
            "stage_index": self.stage_index,
            "stages": self.stages,
            "block_range": list(self.block_range),
            "num_blocks": self.num_blocks,
        }


def build_krea2_stage_module(
    model: Any,
    *,
    stage_index: int,
    stages: int,
    wire: Krea2WireSpec,
    network: Any = None,
    compute_policy: PpComputePolicy | None = None,
    local_freqs: bool = True,
):
    """``(layout, stage_module)`` for this rank, trimmed to its own blocks."""

    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList):
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            "Krea-2 pipeline model must expose a blocks ModuleList",
        )
    plan = make_krea2_pipeline_plan(stages=int(stages), num_blocks=len(blocks))
    block_range = plan.ranges[int(stage_index)]
    layout = Krea2StageLayout(
        stage_index=int(stage_index),
        stages=int(stages),
        block_range=(int(block_range[0]), int(block_range[1])),
        num_blocks=len(blocks),
        local_freqs=bool(local_freqs),
    )
    start, stop = layout.block_range
    sliced = list(blocks[start:stop])
    if network is not None:
        adapter_markers = ("lora", "loha", "lokr", "vera", "glora")
        has_adapter = any(
            any(marker in name.lower() for marker in adapter_markers)
            or any(marker in type(module).__name__.lower() for marker in adapter_markers)
            for name, module in model.named_modules()
        )
        if not has_adapter:
            net_params = getattr(network, "parameters", None)
            if callable(net_params):
                has_adapter = any(True for _ in net_params())
        if not has_adapter:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                "network was passed but the model carries no adapter modules; "
                "apply the network to the live base model before building the "
                "Krea-2 PP stage",
            )
    kwargs = dict(
        model=model,
        blocks=sliced,
        stage_index=layout.stage_index,
        stages=layout.stages,
        block_range=layout.block_range,
        activation_dtype=(
            None if compute_policy is None else compute_policy.activation_dtype
        ),
        max_text_tokens=wire.max_text_tokens,
        img_tokens=wire.img_tokens,
        patch=wire.patch,
        latent_hw=(int(wire.latent_shape[3]), int(wire.latent_shape[4])),
        local_freqs=layout.local_freqs,
        network=network,
    )
    stage_module = Krea2FirstStage(**kwargs) if layout.is_first else Krea2LastStage(**kwargs)
    return layout, stage_module


def adapter_modules_for_stage(network: Any, layout: Krea2StageLayout) -> list:
    """Adapter modules whose ``lora_name`` falls inside this stage's blocks."""

    start, stop = layout.block_range
    selected: list = []
    for module in getattr(network, "unet_loras", []) or []:
        name = str(getattr(module, "lora_name", ""))
        match = re.search(r"(?:^|_)blocks[_.](\d+)[_.]", name)
        if match is None or start <= int(match.group(1)) < stop:
            selected.append(module)
    return selected


def _unpatchify(prediction: Tensor, wire: Krea2WireSpec) -> Tensor:
    """``[B, L_img, patch*patch*C]`` -> 5D latent velocity, like forward_for_loss."""

    from einops import rearrange

    patch = int(wire.patch)
    _, _, _t, height, width = wire.latent_shape
    latent_h, latent_w = height // patch, width // patch
    velocity_4d = rearrange(
        prediction,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=latent_h,
        w=latent_w,
        c=KREA2_LATENT_CHANNELS,
        ph=patch,
        pw=patch,
    )
    return velocity_4d.unsqueeze(2)


class Krea2PpPlugin(PpFamilyPlugin):
    """Krea-2 PP v0 family plugin hooks over GenericPipelineRuntime."""

    family = "krea2_raw"
    negotiation_prefix = "pp_krea2_build"
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
    ) -> Krea2WireSpec:
        del extra
        assert max_text_tokens is not None
        return resolve_krea2_wire_spec(
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
    ) -> tuple[Krea2StageLayout, Any]:
        del extra
        return build_krea2_stage_module(
            model,
            stage_index=int(rank),
            stages=int(config.stages),
            wire=wire,
            network=network,
            compute_policy=policy,
        )

    def after_stage_module_built(
        self,
        model: Any,
        *,
        layout: Any,
        stage_module: Any,
        config: Any,
        rank: int,
        device: torch.device,
        network: Any,
        **extra: Any,
    ) -> None:
        if bool(extra.get("release_foreign", False)):
            release_foreign_submodules(model, is_first=layout.is_first)
        if network is not None:
            adapter_ids = {id(p) for p in network.parameters()}
            for param in model.parameters():
                param.requires_grad_(id(param) in adapter_ids)
        for block in stage_module.blocks:
            block.to(device)
        for name, child in model.named_children():
            if name != "blocks":
                child.to(device)
        if network is not None:
            keep = {id(m) for m in adapter_modules_for_stage(network, layout)}
            for lora in getattr(network, "unet_loras", []) or []:
                if id(lora) not in keep:
                    lora.to("cpu")
        if device.type != "cpu":
            cpu_leftovers: list[str] = []
            for block in stage_module.blocks:
                cpu_leftovers += [
                    n for n, t in block.named_parameters() if t.device.type == "cpu"
                ]
            for name, child in model.named_children():
                if name != "blocks":
                    cpu_leftovers += [
                        f"{name}.{n}"
                        for n, t in child.named_parameters()
                        if t.device.type == "cpu"
                    ]
            if cpu_leftovers:
                raise PipelineParallelError(
                    CODE_STAGE_CONTRACT,
                    f"rank {rank} still holds {len(cpu_leftovers)} cpu tensors of its own after "
                    f"the stage move (first few: {cpu_leftovers[:6]}); a cpu weight surfaces as "
                    "an ATen device mismatch inside forward",
                )
        swap = int(getattr(config, "blocks_to_swap", 0) or 0)
        if not swap:
            swap = int(os.environ.get("KREA2_PP_SWAP_BLOCKS", "0") or 0)
        if swap:
            swapped = enable_stage_block_swap(model.blocks, device, num_swap=swap)
            if swapped:
                print(f"[pp] rank={rank} swapping blocks {swapped} to cpu", flush=True)
            if len(swapped) < swap:
                print(
                    f"[pp] rank={rank} warning: requested {swap} swapped blocks "
                    f"but only {len(swapped)} are owned-and-swappable "
                    f"(block 0 is never swapped); using {len(swapped)}",
                    flush=True,
                )
        if int(os.environ.get("KREA2_PP_PROBE", "0") or 0):
            probed = enable_block_probes(model.blocks, rank)
            print(f"[pp] rank={rank} probing {probed} blocks", flush=True)

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
        input_args, output_args = krea2_boundary_metas(
            layout,
            wire=wire,
            policy=policy,
            latent_dtype=policy.activation_dtype,
        )
        frozen_indices = (
            KREA2_LOCAL_FREQS_FROZEN_INDICES
            if layout.local_freqs
            else KREA2_BOUNDARY_FROZEN_INDICES
        )
        return FrozenBoundaryPipelineStage(
            stage_module,
            int(rank),
            int(config.stages),
            device,
            input_args=tuple(input_args),
            output_args=tuple(output_args),
            group=group,
            frozen_input_indices=() if layout.is_first else frozen_indices,
            frozen_output_indices=() if layout.is_last else frozen_indices,
        )

    def extra_topology(self, runtime: GenericPipelineRuntime) -> dict[str, Any]:
        wire: Krea2WireSpec = runtime.wire
        return {
            "max_text_tokens": wire.max_text_tokens,
            "dim": wire.dim,
            "patch": wire.patch,
            "dtype": str(runtime._dtype),
            "wire_dtype": str(runtime.compute_policy.activation_dtype),
            "compute_policy": runtime.compute_policy.as_dict(),
            "adapter_names": sorted(
                str(getattr(module, "lora_name", ""))
                for module in runtime.stage_module.adapter_modules
            ),
        }

    def precheck_step(
        self,
        runtime: GenericPipelineRuntime,
        latents: Tensor | None,
        step_kwargs: dict[str, Any],
        target: Tensor | None,
    ) -> None:
        del step_kwargs, target
        if runtime.is_first and latents is not None:
            shape = tuple(int(value) for value in latents.shape)
            if shape != runtime.wire.latent_shape:
                raise PipelineParallelError(
                    "pp_krea2_latent_shape_invalid",
                    f"this runtime serves exactly one whole-step latent shape "
                    f"{runtime.wire.latent_shape}; got {shape}",
                )

    def prepare_schedule_inputs(
        self,
        runtime: GenericPipelineRuntime,
        step_kwargs: dict[str, Any],
        target: Tensor | None,
    ) -> tuple[dict[str, Any], Tensor | None]:
        if target is not None and target.ndim in (4, 5):
            from einops import rearrange

            target_4d = target.squeeze(2) if target.ndim == 5 else target
            patch = int(runtime.wire.patch)
            target = rearrange(
                target_4d,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=patch,
                pw=patch,
            )
        return step_kwargs, target

    def postprocess_prediction(
        self, runtime: GenericPipelineRuntime, prediction: Any
    ) -> Any:
        if runtime.is_last and prediction is not None:
            return _unpatchify(prediction, runtime.wire)
        return prediction


_KREA2_PP_PLUGIN = Krea2PpPlugin()


class Krea2PipelineRuntime(GenericPipelineRuntime):
    """This rank's Krea-2 PP v0 runtime."""

    _default_plugin = _KREA2_PP_PLUGIN

    def step(
        self,
        latents: Tensor | None,
        *,
        timesteps: Tensor | None = None,
        context: Tensor | None = None,
        mask: Tensor | None = None,
        target: Tensor | None = None,
        losses: list | None = None,
    ):
        """Run one 1F1B step; callers pass the full 5D batch, the schedule chunks it."""

        return self._execute_step(
            latents,
            step_kwargs={
                "timesteps": timesteps,
                "context": context,
                "mask": mask,
            },
            target=target,
            losses=losses,
        )


def build_krea2_pipeline_runtime(
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
    release_foreign: bool = False,
) -> Krea2PipelineRuntime:
    """Build this rank's Krea-2 PP v0 runtime, or fail closed with a named code."""

    return build_generic_pipeline_runtime(
        _KREA2_PP_PLUGIN,
        Krea2PipelineRuntime,
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
        release_foreign=release_foreign,
    )
