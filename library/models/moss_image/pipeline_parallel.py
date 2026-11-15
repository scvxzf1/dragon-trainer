# SPDX-License-Identifier: Apache-2.0
"""MOSS-Image PP=2 stage components (family hooks for ``library.training.pp``).

This module owns everything MOSS-specific about a two-stage pipeline and
nothing about scheduling, the fp16 transaction or checkpointing: those are
reused from :mod:`library.training.pp` unchanged.  The split forward here is a
*slice* of the production :meth:`~library.models.moss_image.dit
.MOSSImageTransformer.forward` loop, not a second implementation:

* stage 0: ``video_embedder`` + ``text_embedder`` (text refinement, per-sample
  true-length RoPE) + ``shared_block_modulation`` + ``layers[0:k)``;
* stage 1: ``layers[k:42)`` + ``final_norm_video`` + ``video_head_modulation``
  gating + ``final_linear_video`` + unflatten back to ``[B,128,H,W]``.

The boundary carries the *joint* hidden (image + evolved text) differentiably,
so the text refinement gradient reaches stage 0.  Transport padding to the
frozen ``token_cap`` never enters any attention: the receiving stage slices
every sample to its true length from the per-sample lengths tensor that rides
the wire beside the hidden.

Adapter ownership is resolved from the real adapter objects (``lora_name`` is
the kohya-style ``lora_unet_layers_<N>_proj_<qkvo>`` produced by
``networks.lora_anima.targeting``); the Anima ``lora_unet_blocks_(N)_`` regex
does not match it and is deliberately not reused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from library.models.pipeline_parallel import BorrowedBlockStage

from library.training.pp.errors import (
    CODE_COMPUTE_POLICY_MISMATCH,
    CODE_MICROBATCHES_INVALID,
    CODE_SHAPE_CONTRACT,
    CODE_STAGE_CONTRACT,
    CODE_TOKEN_BUCKET_MISMATCH,
    PipelineParallelError,
)

#: MOSS latent channel count (packed BN-normalized form).
MOSS_LATENT_CHANNELS = 128

#: Official MOSS-Image topology: 42 main ``_Block`` layers in ``model.layers``.
MOSS_NUM_LAYERS = 42

#: Regex over the real kohya-style adapter name for a main-block projection.
_MOSS_ADAPTER_RE = re.compile(r"lora_unet_layers_(\d+)_proj_[qkvo]$")


def validate_moss_latent_shape(latent_shape: Sequence[int]) -> tuple[int, ...]:
    """``(B, 128, H, W)`` or a named ``pp_shape_contract_violation``.

    The packed 128-channel form has no parity requirement of its own: the even
    edge constraint applies to the pre-pack 32-channel latents only (see
    ``moss_image.latent.pack_moss_latents``).
    """

    shape = tuple(int(value) for value in latent_shape)
    if len(shape) != 4:
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            f"MOSS PP latents must be 4D [B, {MOSS_LATENT_CHANNELS}, H, W], got "
            f"shape {shape}",
        )
    if shape[1] != MOSS_LATENT_CHANNELS:
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            f"MOSS PP latents must carry {MOSS_LATENT_CHANNELS} packed channels, "
            f"got {shape[1]}",
        )
    if any(dimension < 1 for dimension in (shape[0], shape[2], shape[3])):
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            f"MOSS PP latent batch/height/width must be positive, got {shape}",
        )
    return shape


def resolve_wire_capacity(height: int, width: int, max_text_tokens: int) -> int:
    """Frozen wire capacity: ``H*W + max_text_tokens`` tokens per sample."""

    if int(max_text_tokens) < 1:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"max_text_tokens must be >= 1, got {max_text_tokens!r}",
        )
    return int(height) * int(width) + int(max_text_tokens)


def split_moss_batch(batch_size: int, microbatches: int) -> int:
    """Uniform microbatch size, or a named rejection.

    PP v0 freezes the wire receive buffers, so every microbatch must be the
    same size: a tail batch or a non-divisible batch is refused instead of
    padding a fake sample.
    """

    if int(microbatches) < 1:
        raise PipelineParallelError(
            CODE_MICROBATCHES_INVALID,
            f"microbatches must be >= 1, got {microbatches!r}",
        )
    if int(batch_size) < 1:
        raise PipelineParallelError(
            CODE_TOKEN_BUCKET_MISMATCH,
            f"batch_size must be >= 1, got {batch_size!r}",
        )
    if int(batch_size) % int(microbatches):
        raise PipelineParallelError(
            CODE_TOKEN_BUCKET_MISMATCH,
            f"train batch size {batch_size} is not divisible by "
            f"pipeline_parallel_microbatches={microbatches}; PP v0 freezes the "
            "wire buffers so every microbatch must be the same size (no tail "
            "batch, no fake padded sample)",
        )
    return int(batch_size) // int(microbatches)


@dataclass(frozen=True)
class MossStageLayout:
    """Which contiguous ``layers`` range one rank owns."""

    family: str
    stage_index: int
    stages: int
    layer_range: tuple[int, int]
    num_layers: int
    max_text_tokens: int

    @property
    def is_first(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        return self.stage_index == self.stages - 1

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(range(self.layer_range[0], self.layer_range[1]))

    def as_dict(self) -> dict[str, Any]:
        # ``block_range``/``num_blocks`` are the checkpoint protocol's shared
        # keys (``checkpoint_artifacts._plan_record`` recomputes the plan from
        # them); ``layer_range`` is the MOSS-native spelling.
        return {
            "family": self.family,
            "stage_index": self.stage_index,
            "stages": self.stages,
            "block_range": list(self.layer_range),
            "layer_range": list(self.layer_range),
            "num_blocks": self.num_layers,
            "max_text_tokens": int(self.max_text_tokens),
        }


def moss_adapter_block_index(adapter_name: str) -> int:
    """Global ``layers`` index an adapter patches, from its real lora_name."""

    match = _MOSS_ADAPTER_RE.search(str(adapter_name))
    if match is None:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"cannot resolve a layers index from adapter name {adapter_name!r}; "
            "MOSS PP placement needs adapters named "
            "lora_unet_layers_<N>_proj_[qkvo] (see "
            "library.models.moss_image.lora_targets)",
        )
    return int(match.group(1))


def moss_stage_adapter_modules(network, layout: MossStageLayout, model=None) -> list[nn.Module]:
    """Adapter modules attached to the ``layers`` this stage owns.

    Ownership is parsed from the real ``lora_name`` (the kohya-style name the
    production creation path actually builds).  When the model is supplied the
    target is cross-checked against the applied model: ``apply_to`` monkey-patch
    deletes the adapter's ``org_module`` reference, so the check reconstructs
    the dotted module path from the name and verifies the wrapped Linear's
    geometry -- an ``include_patterns`` override cannot silently move an
    adapter to a stage that does not run its target.
    """

    adapters = getattr(network, "unet_loras", None)
    if adapters is None:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"adapter network {type(network).__name__} does not expose unet_loras",
        )
    owned = set(layout.layer_indices)
    selected = []
    for adapter in adapters:
        index = moss_adapter_block_index(adapter.lora_name)
        if index not in owned:
            continue
        if model is not None:
            _assert_adapter_target_matches(adapter, model, index)
        selected.append(adapter)
    return selected


_MOSS_TARGET_RE = re.compile(r"^lora_unet_layers_(\d+)_proj_([qkvo])$")


def _assert_adapter_target_matches(adapter, model, index: int) -> None:
    match = _MOSS_TARGET_RE.fullmatch(str(adapter.lora_name))
    if match is None:  # pragma: no cover - ownership regex already matched
        return
    layer_index, projection = int(match.group(1)), match.group(2)
    if layer_index != index:  # pragma: no cover - same regex as ownership
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"adapter {adapter.lora_name!r} claims layers[{index}] but parses to "
            f"layers[{layer_index}]",
        )
    module = dict(model.named_modules()).get(f"layers.{layer_index}.proj.{projection}")
    if module is None:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"adapter {adapter.lora_name!r} targets layers.{layer_index}.proj."
            f"{projection}, which does not exist in this model; ownership and "
            "placement disagree",
        )
    in_features = getattr(module, "in_features", None)
    down = getattr(adapter, "lora_down", None)
    rank = getattr(getattr(down, "weight", None), "shape", None)
    if in_features is not None and rank is not None and rank[1] != in_features:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"adapter {adapter.lora_name!r} geometry ({tuple(rank)}) does not "
            f"match its target Linear (in_features={in_features})",
        )


def assert_moss_stage_capabilities(model) -> None:
    """Reject models the split forward cannot reproduce, before the first step."""

    layers = getattr(model, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            "MOSS PP needs model.layers as an nn.ModuleList, got "
            f"{type(layers).__name__}",
        )
    swap = int(getattr(model, "blocks_to_swap", 0) or 0)
    if swap:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"pipeline_parallel cannot own a block-swap model (blocks_to_swap="
            f"{swap}); MOSS block swap also forces batch_size=1, which the "
            "pipeline wire cannot serve",
        )
    text_embedder = getattr(model, "text_embedder", None)
    if text_embedder is None or not hasattr(text_embedder, "blocks"):
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            "MOSS PP needs model.text_embedder with its refinement blocks",
        )
    for attribute in ("video_embedder", "rope", "time_embedding",
                      "shared_block_modulation", "final_norm_video",
                      "video_head_modulation", "final_linear_video"):
        if not hasattr(model, attribute):
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                f"the MOSS PP split forward needs the production model surface; "
                f"missing {attribute!r}",
            )


def _moss_block_stage(model, layout: MossStageLayout) -> BorrowedBlockStage:
    layers = model.layers
    start, end = layout.layer_range
    return MossImageBlockStage(
        tuple(layers[start:end]),
        source_block_container="layers",
        stage_index=layout.stage_index,
        block_range=(start, end),
    )


class MossImageBlockStage(BorrowedBlockStage):
    """Borrowed ``layers`` slice; identifies MOSS in plan/topology records."""


class _MossStageBase(nn.Module):
    """Non-owning view of one MOSS stage over a borrowed ``layers`` slice."""

    def __init__(
        self,
        *,
        model: nn.Module,
        layout: MossStageLayout,
        latent_shape: Sequence[int],
        adapter_modules: Sequence[nn.Module] = (),
        compute_policy: Any = None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_block_stage", _moss_block_stage(model, layout))
        object.__setattr__(self, "_adapter_modules", tuple(adapter_modules))
        self.layout = layout
        self.stage_index = int(layout.stage_index)
        self.stages = int(layout.stages)
        self.max_text_tokens = int(layout.max_text_tokens)
        self.compute_policy = compute_policy
        latent = validate_moss_latent_shape(latent_shape)
        object.__setattr__(self, "_latent_shape", latent)
        self.height = int(latent[2])
        self.width = int(latent[3])
        self.image_tokens = self.height * self.width

    @property
    def activation_dtype(self) -> torch.dtype:
        """The pipeline-wide compute dtype (never re-derived from a boundary)."""

        if self.compute_policy is None:
            return _model_dtype(self._model)
        return self.compute_policy.activation_dtype

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def block_stage(self) -> BorrowedBlockStage:
        return self._block_stage

    @property
    def adapter_modules(self) -> tuple[nn.Module, ...]:
        return self._adapter_modules

    @property
    def is_first(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        return self.stage_index == self.stages - 1

    def trainable_parameters(self):
        """Trainable parameters this stage owns (stage-local plain LoRA only)."""

        seen: set[int] = set()
        for module in self._adapter_modules:
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

    def _time_embedding(self, timesteps: Tensor, batch: int) -> Tensor:
        """Production time path: shared fp64-sinusoid helper, fp32 into the MLP."""

        from .math import time_sinusoid

        model = self._model
        device = model.video_embedder.weight.device
        times = torch.as_tensor(timesteps, device=device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)  # production forward does the same expand
        embedding = time_sinusoid(times, model.config.freq_dim, device)
        return model.time_embedding(
            embedding.to(model.time_embedding[0].weight.dtype)
        ).float()


def _run_sample_layers(block_stage, x: Tensor, rope: Tensor, modulation: Tensor) -> Tensor:
    """The production per-sample block loop for this stage's slice."""

    for layer in block_stage.blocks:
        x = layer(x, rope, modulation)
    return x


class MossFirstStage(_MossStageBase):
    """Embeddings + text refinement + modulation + the first layer range.

    Returns the padded joint hidden ``[B, token_cap, dim]`` (grad-carrying),
    the per-sample true text lengths ``[B]`` (int metadata) and the shared
    block modulation ``[B, 6, dim]`` (frozen, detached).
    """

    def forward(
        self,
        latents: Tensor,
        *,
        timesteps: Tensor,
        text_hiddens: Tensor,
        text_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        model = self._model
        batch, _, height, width = validate_moss_latent_shape(latents.shape)
        if (height, width) != (self.height, self.width):
            raise PipelineParallelError(
                CODE_TOKEN_BUCKET_MISMATCH,
                f"this PP runtime serves exactly one latent geometry "
                f"({self.height}x{self.width}); got {height}x{width}. Build one "
                "runtime per latent shape",
            )
        _validate_text_wire(text_hiddens, text_mask, batch, self.max_text_tokens)
        token_cap = resolve_wire_capacity(height, width, self.max_text_tokens)
        if latents.dtype != self.activation_dtype:
            raise PipelineParallelError(
                CODE_COMPUTE_POLICY_MISMATCH,
                f"the pre-block latent dtype is {latents.dtype} but the pipeline "
                f"compute policy was built for {self.activation_dtype}; build the "
                "runtime with the dtype the trainer actually feeds",
            )
        video_dtype = model.video_embedder.weight.dtype
        mod = model.shared_block_modulation(
            self._time_embedding(timesteps, batch).to(
                model.shared_block_modulation.weight.dtype
            )
        ).view(batch, 6, -1)
        # The boundary tensor is allocated in the *compute* dtype, not the
        # latent dtype: a mixed fp32-latent/fp16-compute run must not quietly
        # demote every joint token to the latent dtype.
        hidden = torch.zeros(
            (batch, token_cap, model.video_embedder.out_features),
            dtype=self.activation_dtype,
            device=latents.device,
        )
        # The wire is grad-forced by torch pipelining on every received tensor,
        # so lengths ride the wire as float32 (exact for token counts) and are
        # converted back to ints on use.
        lengths = torch.zeros(batch, dtype=torch.float32, device=latents.device)
        image_tokens = height * width
        for index in range(batch):
            length = int(text_mask[index].sum().item())
            if length < 1:
                raise PipelineParallelError(
                    CODE_STAGE_CONTRACT,
                    "each sample needs at least one valid text token "
                    f"(sample {index} has none); the production MOSS forward "
                    "rejects the same case",
                )
            context = text_hiddens[index:index + 1, text_mask[index].bool()]
            video = latents[index:index + 1].flatten(2).transpose(1, 2)
            video = model.video_embedder(
                video.to(video_dtype).contiguous()
            )
            rope = model.rope(height, width, length, latents.device).to(video.dtype)
            text = model.text_embedder(context, rope[:, image_tokens:])
            sample = torch.cat((video, text), 1)
            sample = _run_sample_layers(
                self._block_stage, sample, rope, mod[index:index + 1]
            )
            if sample.shape[1] != image_tokens + length:
                raise PipelineParallelError(
                    CODE_STAGE_CONTRACT,
                    f"stage 0 produced {sample.shape[1]} joint tokens, expected "
                    f"{image_tokens + length}",
                )
            hidden[index:index + 1, :image_tokens + length] = sample
            lengths[index] = length
        return hidden, lengths, mod.detach()


class MossLastStage(_MossStageBase):
    """The remaining layer range + final norm/head/linear + unflatten.

    Padding is sliced away per sample before any attention or norm runs, so
    padded transport positions cannot influence q/k/v, RMSNorm statistics or
    the image RoPE (rebuilt locally from the true per-sample lengths).
    """

    def forward(
        self,
        hidden: Tensor,
        lengths: Tensor,
        modulation: Tensor,
        *,
        timesteps: Tensor,
    ) -> Tensor:
        model = self._model
        batch, token_cap, _dim = hidden.shape
        if hidden.dtype != self.activation_dtype:
            raise PipelineParallelError(
                CODE_COMPUTE_POLICY_MISMATCH,
                f"the boundary hidden dtype is {hidden.dtype} but the pipeline "
                f"compute policy was built for {self.activation_dtype}; every "
                "stage must run the pipeline's single compute-dtype decision",
            )
        expected_cap = self.image_tokens + self.max_text_tokens
        if token_cap != expected_cap:
            raise PipelineParallelError(
                CODE_TOKEN_BUCKET_MISMATCH,
                f"boundary hidden token capacity is {token_cap} but this stage "
                f"was built for {expected_cap} "
                f"({self.image_tokens} image + {self.max_text_tokens} text slots); "
                "build one runtime per wire shape",
            )
        if int(lengths.shape[0]) != batch:
            raise PipelineParallelError(
                CODE_SHAPE_CONTRACT,
                f"lengths must be [B={batch}], got {tuple(lengths.shape)}",
            )
        head = model.video_head_modulation(
            self._time_embedding(timesteps, batch).to(
                model.video_head_modulation.weight.dtype
            )
        ).view(batch, 2, -1)
        modulation = modulation.detach()
        outputs = []
        for index in range(batch):
            length = int(lengths[index].item())
            if length < 1 or self.image_tokens + length > token_cap:
                raise PipelineParallelError(
                    CODE_STAGE_CONTRACT,
                    f"sample {index} reports text length {length}, outside "
                    f"[1, {token_cap - self.image_tokens}]; the wire metadata "
                    "and the built capacity disagree",
                )
            sample = hidden[index:index + 1, :self.image_tokens + length]
            # The production forward casts RoPE to the activation dtype of the
            # tokens it rotates; the boundary dtype *is* the pipeline's compute
            # dtype, so both spellings agree -- stated explicitly here so the
            # stage cannot drift from ``_RoPE`` semantics.
            rope = model.rope(
                self.height, self.width, length, sample.device
            ).to(self.activation_dtype)
            sample = _run_sample_layers(
                self._block_stage, sample, rope, modulation[index:index + 1]
            )
            x = model.final_norm_video(
                sample[:, :self.image_tokens].to(
                    model.final_norm_video.weight.dtype
                )
            )
            x = x * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
            x = model.final_linear_video(
                x.to(model.final_linear_video.weight.dtype)
            )
            outputs.append(
                x.transpose(1, 2).reshape(1, -1, self.height, self.width)
            )
        return torch.cat(outputs)


def _validate_text_wire(text_hiddens: Tensor, text_mask: Tensor, batch: int,
                        max_text_tokens: int) -> None:
    if text_hiddens.ndim != 3 or text_hiddens.shape[0] != batch:
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            f"text_hiddens must be [B, {max_text_tokens}, text_dim] matching the "
            f"microbatch (B={batch}), got {tuple(text_hiddens.shape)}",
        )
    if text_hiddens.shape[1] != max_text_tokens:
        raise PipelineParallelError(
            CODE_TOKEN_BUCKET_MISMATCH,
            f"text_hiddens second dimension must equal the built wire capacity "
            f"text slots {max_text_tokens}, got {text_hiddens.shape[1]}; build "
            "one runtime per max_text_tokens",
        )
    if text_mask.shape != text_hiddens.shape[:2]:
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            f"text_mask must be [B, {max_text_tokens}] matching text_hiddens, got "
            f"{tuple(text_mask.shape)}",
        )
    if not torch.all((text_mask == 0) | (text_mask == 1)):
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            "text_mask must be a binary [B, slots] mask; non-binary values would "
            "make the true text length ambiguous (the production MOSS forward "
            "rejects the same case)",
        )
    if int(text_mask.sum().item()) < batch:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            "every sample needs at least one valid text token (mask sum < batch); "
            "the production MOSS forward rejects the same case",
        )


def build_moss_stage_module(
    model,
    *,
    stage_index: int,
    stages: int,
    latent_shape: Sequence[int],
    max_text_tokens: int,
    network=None,
    compute_policy=None,
) -> tuple[MossStageLayout, nn.Module]:
    """Build this rank's MOSS stage module over a contiguous layer range."""

    layers = getattr(model, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            "MOSS PP needs model.layers as an nn.ModuleList",
        )
    assert_moss_stage_capabilities(model)
    if int(stages) != 2:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"MOSS PP v0 supports exactly 2 stages, got {stages}",
        )
    from library.models.pipeline_parallel import make_pipeline_plan

    plan = make_pipeline_plan(
        family="moss_image", stages=int(stages), num_blocks=len(layers)
    )
    if int(stage_index) not in range(int(stages)):
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"stage_index must be in [0, {stages}), got {stage_index}",
        )
    layout = MossStageLayout(
        family="moss_image",
        stage_index=int(stage_index),
        stages=int(stages),
        layer_range=plan.range_for_stage(int(stage_index)),
        num_layers=len(layers),
        max_text_tokens=int(max_text_tokens),
    )
    adapters = () if network is None else tuple(
        moss_stage_adapter_modules(network, layout, model)
    )
    stage_type = MossFirstStage if layout.is_first else MossLastStage
    module = stage_type(
        model=model,
        layout=layout,
        latent_shape=latent_shape,
        adapter_modules=adapters,
        compute_policy=compute_policy,
    )
    return layout, module


def _model_dtype(model) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


__all__ = [
    "MOSS_LATENT_CHANNELS",
    "MOSS_NUM_LAYERS",
    "MossFirstStage",
    "MossImageBlockStage",
    "MossLastStage",
    "MossStageLayout",
    "assert_moss_stage_capabilities",
    "build_moss_stage_module",
    "moss_adapter_block_index",
    "moss_stage_adapter_modules",
    "resolve_wire_capacity",
    "split_moss_batch",
    "validate_moss_latent_shape",
]
