"""Krea-2 PP v0 runtime wiring (probe level).

Mirrors the MOSS runtime skeleton: freeze the wire geometry once per run, derive
this stage's boundary metadata from it, then let ``torch.distributed.pipelining``
allocate the receive buffers.  Only the geometry half is wired here -- the stage
modules themselves live in :mod:`library.models.krea2_raw.pp_stages`.

Wire slots (fixed order, frozen indices are gradient-free):

===  ============  ==================================  ==========
idx  name          shape                               frozen
===  ============  ==================================  ==========
0    hidden        ``[mb, token_cap, features]``       no
1    tvec          ``[mb, 1, 6*features]``             yes
2    freqs         ``[mb, token_cap, headdim//2, 2, 2]`` yes
3    mask          ``[mb, token_cap] `` float32         yes
4    tembed        ``[mb, 1, features]``                yes
===  ============  ==================================  ==========

``token_cap`` is ``max_text_tokens + img_tokens`` rounded up to 256, matching the
padding ``SingleStreamDiT.forward`` applies.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Sequence

import re
import torch
from torch import Tensor

from library.models.krea2_raw.pp_stages import krea2_token_cap
from library.training.pp.contract import PpComputePolicy
from library.training.pp.errors import PipelineParallelError

__all__ = [
    "KREA2_BOUNDARY_FROZEN_INDICES",
    "Krea2WireSpec",
    "resolve_krea2_wire_spec",
    "krea2_boundary_metas",
]

#: Wire slots that carry no gradient (everything except the residual stream).
KREA2_BOUNDARY_FROZEN_INDICES: tuple[int, ...] = (1, 2, 3, 4)

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
    model,
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
    if int(microbatches) <= 0 or batch % int(microbatches):
        raise PipelineParallelError(
            "pp_krea2_batch_not_divisible",
            f"train batch size {batch} is not divisible by "
            f"pipeline_parallel_microbatches={microbatches}; PP v0 freezes the "
            "wire buffers so every microbatch must be the same size",
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
    stage,
    *,
    wire: Krea2WireSpec,
    policy: PpComputePolicy,
    latent_dtype: torch.dtype,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """``(input_args, output_args)`` metadata for this Krea-2 stage.

    Unlike Anima, Krea-2 keeps its residual stream in the *activation* dtype when
    ``fp32_residual`` is off, so the boundary dtype follows
    ``policy.activation_dtype`` rather than Anima's promoted ``boundary_dtype``.
    """

    dtype = policy.activation_dtype
    mb = wire.microbatch_batch_size
    boundary = (
        _meta((mb, wire.token_cap, wire.dim), dtype),
        _meta((mb, 1, 6 * wire.dim), dtype),
        # RoPE frequencies are produced by ``posemb`` in fp32 regardless of the
        # activation dtype; declaring them fp16 here makes pipelining reject the
        # real output.  Keep fp32 -- casting would change the numerics.
        _meta(wire.freqs_shape, torch.float32),
        # torch pipelining force-enables requires_grad on received tensors, so
        # the boolean mask rides as float32 (exact for 0/1) instead of bool.
        _meta((mb, wire.token_cap), torch.float32),
        _meta((mb, 1, wire.dim), dtype),
    )
    if stage.is_first:
        return (_meta(wire.microbatch_latent_shape, latent_dtype),), boundary
    # The trailing stage hands back the model's raw per-image output:
    # [B, L_img, patch*patch*C] -- *not* the padded joint sequence.  The runtime
    # unpatchifies it into a 5D velocity afterwards.
    prediction = _meta(
        (mb, wire.img_tokens, wire.patch * wire.patch * KREA2_LATENT_CHANNELS), dtype
    )
    return boundary, (prediction,)


# ---------------------------------------------------------------------------
# Stage layout + runtime assembly (step 3, assembly half)
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass  # noqa: E402
from typing import Any  # noqa: E402

from library.models.krea2_raw.pipeline_parallel import (  # noqa: E402
    make_krea2_pipeline_plan,
)
from library.models.krea2_raw.pp_stages import (
    enable_stage_block_swap,  # noqa: E402
    Krea2FirstStage,
    Krea2LastStage,
    release_foreign_submodules,
)
from library.training.pp.contract import (  # noqa: E402
    assert_compute_policy_agreement,
    validate_distributed_backend,
    validate_master_weight_dtype,
    validate_runtime_config,
)
from library.training.pp.errors import (  # noqa: E402
    CODE_STAGE_CONTRACT,
    PipelineParallelError,
)
from library.training.pp.schedule import (  # noqa: E402
    build_anima_schedule,
    make_last_stage_loss_fn,
    run_pipeline_step,
)
from library.training.pp.stages import (  # noqa: E402
    FrozenBoundaryPipelineStage,
    assert_frozen_boundary_agreement,
)
from library.training.pp.transactions import (  # noqa: E402
    PpFp16Transaction,
    PpStepOutcome,
    make_clip_gradients,
)


@_dataclass(frozen=True)
class Krea2StageLayout:
    """Contiguous block ownership for one Krea-2 pipeline stage."""

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
    def local_blocks(self) -> int:
        return self.block_range[1] - self.block_range[0]


def _model_dtype(model) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _distributed_backend(group) -> str | None:
    """Backend spec of the default (or given) group, e.g. ``cpu:gloo,cuda:nccl``."""

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None
    if not hasattr(dist, "get_backend"):
        return None
    try:
        return str(dist.get_backend(group))
    except Exception:  # noqa: BLE001 - an unknown backend must not mask the check
        return None


def _negotiate_phase(error, group, *, where: str) -> None:
    """Turn a rank-local build failure into an agreed one (or none)."""

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        if error is not None:
            raise PipelineParallelError(error[0], error[1])
        return
    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, error, group=group)
    for entry in gathered:
        if entry is not None:
            raise PipelineParallelError(entry[0], f"{where}: {entry[1]}")


def build_krea2_stage_module(
    model,
    *,
    stage_index: int,
    stages: int,
    wire: Krea2WireSpec,
    network: Any = None,
    compute_policy: PpComputePolicy | None = None,
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
    )
    start, stop = layout.block_range
    sliced = list(blocks[start:stop])
    if network is not None:
        # Contract: the caller applies the adapter to the *live* base model
        # before this call, and freezes the base at the same time.  We must not
        # re-apply it here -- a PP rank owns only its own block range plus its
        # side's fusions, so applying on a trimmed model would silently miss the
        # other half.  ``trainable_parameters()`` then walks exactly this rank's
        # slice and keeps whatever still has requires_grad=True, which is the
        # adapter (plus any intentionally trained Norm/Bias) and nothing else.
        #
        # This is cheap to get wrong silently, so verify the slice really carries
        # adapter parameters: a rank that ends up with none would train nothing
        # while the pipeline still ran.
        # apply_to usually swaps the Linear for an adapter-typed module rather
        # than renaming it, so match on the class name as well as the path.
        adapter_markers = ("lora", "loha", "lokr", "vera", "glora")
        has_adapter = any(
            any(marker in name.lower() for marker in adapter_markers)
            or any(marker in type(module).__name__.lower() for marker in adapter_markers)
            for name, module in model.named_modules()
        )
        if not has_adapter:
            # ``apply_to`` may keep the original module names/classes (it can
            # inject plain Parameters or wrap without renaming), so fall back to
            # the adapter object itself: a built network always carries params.
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
    )
    # The adapter lives on the network object, not inside the base module tree
    # (apply_to patches forward instead of swapping modules), so the stage needs
    # the reference to report this rank's trainable parameters.
    kwargs["network"] = network
    stage_module = Krea2FirstStage(**kwargs) if layout.is_first else Krea2LastStage(**kwargs)
    return layout, stage_module



def adapter_modules_for_stage(network: Any, layout: Krea2StageLayout) -> list:
    """Adapter modules whose ``lora_name`` falls inside this stage's blocks.

    Krea-2's ``apply_to`` creates the adapter for **all** layers and keeps it on
    the network object (it patches forward instead of swapping modules), so a
    plain ``network.to(device)`` parks the other rank's dead weights on this
    card -- measured at 98 MiB on a 13/15 split.  Callers should move only this
    list.  Names look like ``lora_unet_blocks_12_attn_wq``.
    """

    start, stop = layout.block_range
    selected: list = []
    for module in getattr(network, "unet_loras", []) or []:
        name = str(getattr(module, "lora_name", ""))
        match = re.search(r"(?:^|_)blocks[_.](\d+)[_.]", name)
        if match is None or start <= int(match.group(1)) < stop:
            selected.append(module)
    return selected


class Krea2PipelineRuntime:
    """This rank's Krea-2 PP v0 runtime (probe level)."""

    def __init__(
        self,
        *,
        model,
        layout: Krea2StageLayout,
        stage_module,
        stage: FrozenBoundaryPipelineStage,
        schedule,
        config,
        wire: Krea2WireSpec,
        dtype: torch.dtype,
        compute_policy: PpComputePolicy,
        transaction: PpFp16Transaction | None,
        optimizer,
        group,
    ) -> None:
        self._model = model
        self._layout = layout
        self._stage_module = stage_module
        self._stage = stage
        self._schedule = schedule
        self._config = config
        self._wire = wire
        self._dtype = dtype
        self._compute_policy = compute_policy
        self._transaction = transaction
        self._optimizer = optimizer
        self._group = group

    # -- views ---------------------------------------------------------------
    @property
    def layout(self) -> Krea2StageLayout:
        return self._layout

    @property
    def stage(self) -> FrozenBoundaryPipelineStage:
        return self._stage

    @property
    def stage_module(self):
        return self._stage_module

    @property
    def schedule(self):
        return self._schedule

    @property
    def config(self):
        return self._config

    @property
    def wire(self) -> Krea2WireSpec:
        return self._wire

    @property
    def compute_policy(self) -> PpComputePolicy:
        return self._compute_policy

    @property
    def transaction(self) -> PpFp16Transaction | None:
        return self._transaction

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def is_first(self) -> bool:
        return self._layout.is_first

    @property
    def is_last(self) -> bool:
        return self._layout.is_last

    @property
    def scale(self) -> float | None:
        return None if self._transaction is None else self._transaction.scale

    # -- step ----------------------------------------------------------------
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
        """Run one 1F1B step; callers pass the full 5D batch, the schedule chunks it.

        Both ranks must pass the *same* kwargs: pipelining splits them the same
        way on every rank, and an empty set on the trailing rank makes it
        subscript ``None``.  Patchification lives in ``Krea2FirstStage`` (it needs
        the text mask anyway), so nothing rank-specific happens here -- the
        trailing rank simply ignores the inputs it receives and only the leading
        one consumes ``latents``.
        """

        if self._layout.is_first and latents is not None:
            shape = tuple(int(value) for value in latents.shape)
            if shape != self._wire.latent_shape:
                raise PipelineParallelError(
                    "pp_krea2_latent_shape_invalid",
                    f"this runtime serves exactly one whole-step latent shape "
                    f"{self._wire.latent_shape}; got {shape}",
                )
        # The fp16 transaction tracks one optimizer interval at a time and
        # refuses a second step() without begin_step() (the moss runtime does the
        # same).  Without this, step 2 onwards fails with
        # pp_scaler_decision_asymmetry.
        if self._transaction is not None:
            self._transaction.begin_step()
        kwargs: dict[str, Any] = {
            "timesteps": timesteps,
            "context": context,
            "mask": mask,
        }
        # The pipeline loss compares the trailing stage's raw per-image output
        # [B, N_img, patch*patch*C] against whatever target we hand it.  A
        # trainer passes the plain latent [B, C, H, W] (or 5D with the singleton
        # time axis), whose last dim is W -- mse_loss then broadcasts to rank 4
        # and reports "size of tensor a (64) must match tensor b (112) at
        # dimension 3" (64 = C*ph*pw, 112 = W).  Patchifying here is a pure
        # permutation of the same elements, so the mean is unchanged, and it
        # keeps runtime callers that already pass a 3D target working as-is.
        if target is not None and target.ndim in (4, 5):
            from einops import rearrange

            target_4d = target.squeeze(2) if target.ndim == 5 else target
            _patch = int(self._wire.patch)
            target = rearrange(
                target_4d,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=_patch,
                pw=_patch,
            )
        prediction = run_pipeline_step(
            self._schedule,
            is_first=self._layout.is_first,
            # A non-first stage receives its activations over the wire; handing it
            # the positional latents is a named contract violation.
            latents=latents if self._layout.is_first else None,
            kwargs=kwargs,
            target=target,
            losses=losses,
        )
        if self._layout.is_first or prediction is None:
            return prediction
        return _unpatchify(prediction, self._wire)

    def topology(self) -> dict[str, Any]:
        """Stage shape and adapter set, as the PP checkpoint writer records it.

        The checkpoint stores this next to the payload and refuses to resume a
        run whose topology or adapter names differ, so it must describe exactly
        what this rank is training.
        """

        layout = self._layout
        adapter_names = sorted(
            str(getattr(module, "lora_name", ""))
            for module in self._stage_module.adapter_modules
        )
        return {
            "family": "krea2_raw",
            "stage_index": layout.stage_index,
            "stages": layout.stages,
            "is_first": layout.is_first,
            "is_last": layout.is_last,
            "block_range": list(layout.block_range),
            "num_blocks": layout.num_blocks,
            "microbatches": self._config.microbatches,
            "schedule": self._config.schedule,
            "conditioning": self._config.conditioning,
            "mixed_precision": self._config.mixed_precision,
            "latent_shape": list(self._wire.latent_shape),
            "max_text_tokens": self._wire.max_text_tokens,
            "dim": self._wire.dim,
            "patch": self._wire.patch,
            "dtype": str(self._dtype),
            "wire_dtype": str(self._compute_policy.activation_dtype),
            "compute_policy": self._compute_policy.as_dict(),
            "adapter_names": adapter_names,
        }

    def optimizer_step(self) -> PpStepOutcome:
        if self._transaction is None:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                "this runtime was built without an optimizer/GradScaler",
            )
        return self._transaction.step()

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        if self._optimizer is None:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                "this runtime was built without an optimizer",
            )
        self._optimizer.zero_grad(set_to_none=set_to_none)


def build_krea2_pipeline_runtime(
    model,
    *,
    config,
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

    stage = None
    layout = None
    stage_module = None
    parameters = None
    policy = None
    compute_dtype = torch.float32
    wire = None
    build_error: tuple[str, str] | None = None
    try:
        validate_runtime_config(config, world_size=world_size)
        wire = resolve_krea2_wire_spec(
            model,
            latent_shape=latent_shape,
            max_text_tokens=max_text_tokens,
            microbatches=int(config.microbatches),
        )
        is_last = int(rank) == int(config.stages) - 1
        if not is_last and scaler is not None:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                "a non-last stage never sees the loss and must not own a GradScaler",
            )
        compute_dtype = dtype if dtype is not None else _model_dtype(model)
        policy = PpComputePolicy.from_dtype(compute_dtype)
        layout, stage_module = build_krea2_stage_module(
            model,
            stage_index=int(rank),
            stages=int(config.stages),
            wire=wire,
            network=network,
            compute_policy=policy,
        )
        if release_foreign:
            # Opt-in: drop the 1.24 GiB of fusions/heads this rank does not own.
            release_foreign_submodules(model, is_first=layout.is_first)
        # PP v0 trains the adapter only.  Krea-2's apply_to keeps the adapter off
        # the module tree, so a generic "train unet network" freeze never reaches
        # the base and autograd would retain ~192 MiB of fc2.weight dW per layer
        # (measured: 14.98 GiB allocated, OOM on a 16 GB card without this).
        if network is not None:
            _adapter_ids = {id(p) for p in network.parameters()}
            for _param in model.parameters():
                _param.requires_grad_(id(_param) in _adapter_ids)
        # Move this rank's weights onto the device *here*, before the optimizer
        # and the PipelineStage are built: doing it in a caller runs too late
        # (the factory below would capture cpu parameters) and any caller that
        # forgets it leaves model.first() on cpu -- the base model is loaded on
        # cpu under PP because the full 25 GB cannot fit on one 16 GB card.
        # After the release above, ``model`` only holds this rank's weights, so
        # no name guessing is needed.
        # Only this rank's blocks plus the surviving non-block children: walking
        # ``model.parameters()`` here would move all 28 layers and OOM at build
        # time (measured: 14.98 GiB allocated, 192 MiB short).
        for _block in stage_module.blocks:
            _block.to(device)
        for _name, _child in model.named_children():
            if _name == "blocks":
                continue
            _child.to(device)
        _moved = sum(
            1 for t in list(stage_module.parameters()) if t.device.type == "cuda"
        )
        # Only this rank's own tensors matter: the peer's blocks are still in
        # ``model.blocks`` and *must* stay on cpu (they are dropped by
        # release_foreign_blocks later in the caller).
        # Send the peer's adapter modules back to cpu: apply_to builds the LoRA
        # for *all* 28 layers and keeps it on the network object, so a plain
        # network.to(device) parks the other rank's dead adapter here (measured
        # 98 MiB on a 13/15 split).  Doing it here -- not in a caller -- is what
        # makes it unconditional for every entry point.
        if network is not None:
            _keep = {id(m) for m in adapter_modules_for_stage(network, layout)}
            for _lora in getattr(network, "unet_loras", []) or []:
                if id(_lora) not in _keep:
                    _lora.to("cpu")
        _cpu: list[str] = []
        for _block in stage_module.blocks:
            _cpu += [f"{n}" for n, t in _block.named_parameters() if t.device.type == "cpu"]
        for _name, _child in model.named_children():
            if _name == "blocks":
                continue
            _cpu += [
                f"{_name}.{n}" for n, t in _child.named_parameters() if t.device.type == "cpu"
            ]
        if _cpu:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                f"rank {rank} still holds {len(_cpu)} cpu tensors of its own after "
                f"the stage move (first few: {_cpu[:6]}); a cpu weight surfaces as "
                "an ATen device mismatch inside forward",
            )
        # Block swap: each Krea-2 block's frozen base is 809 MiB and the stage
        # is 11.75-12.13 GiB, which left the runtime ~52 MiB short of the 16 GB
        # ceiling.  Swapping 2 blocks frees 809 MiB of continuous space without
        # touching LoRA (adapters live on the network, not in block.parameters).
        _swap = int(os.environ.get("KREA2_PP_SWAP_BLOCKS", "0") or 0)
        if _swap:
            _swapped = enable_stage_block_swap(model.blocks, device, num_swap=_swap)
            if _swapped:
                print(
                    f"[pp] rank={rank} swapping blocks {_swapped} to cpu",
                    flush=True,
                )
        parameters = list(stage_module.trainable_parameters())
        validate_master_weight_dtype(
            parameters, mixed_precision=config.mixed_precision, device=device
        )
        if optimizer is None and optimizer_factory is not None:
            if not parameters:
                raise PipelineParallelError(
                    CODE_STAGE_CONTRACT,
                    f"rank {rank} owns no stage-local trainable parameters",
                )
            # The factory contract is the *stage module* (moss passes the
            # same): a trainer-side factory calls stage_module.
            # trainable_parameters() itself, so handing it the flat list
            # raises AttributeError, which the negotiation below would
            # report as a stage-contract violation.
            optimizer = optimizer_factory(stage_module)
        validate_distributed_backend(
            _distributed_backend(group),
            device_type=str(getattr(device, "type", device)),
            world_size=int(world_size),
        )
        input_args, output_args = krea2_boundary_metas(
            layout,
            wire=wire,
            policy=policy,
            latent_dtype=policy.activation_dtype,
        )
        stage = FrozenBoundaryPipelineStage(
            stage_module,
            int(rank),
            int(config.stages),
            device,
            input_args=tuple(input_args),
            output_args=tuple(output_args),
            group=group,
            frozen_input_indices=(
                () if layout.is_first else KREA2_BOUNDARY_FROZEN_INDICES
            ),
            frozen_output_indices=(
                () if layout.is_last else KREA2_BOUNDARY_FROZEN_INDICES
            ),
        )
    except Exception as exc:  # noqa: BLE001 - negotiated below, never rank-local
        build_error = (
            getattr(exc, "code", CODE_STAGE_CONTRACT),
            f"rank {rank}: {type(exc).__name__}: {exc}",
        )
    _negotiate_phase(build_error, group, where="pp_krea2_build:local")

    assert_compute_policy_agreement(policy, group=group, where="pp_krea2_build")
    frozen_error: tuple[str, str] | None = None
    try:
        assert_frozen_boundary_agreement(stage, group=group)
    except Exception as exc:  # noqa: BLE001 - negotiated below
        frozen_error = (
            getattr(exc, "code", CODE_STAGE_CONTRACT),
            f"rank {rank}: {type(exc).__name__}: {exc}",
        )
    _negotiate_phase(frozen_error, group, where="pp_krea2_build:boundary")

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        gathered: list[Any] = [None] * dist.get_world_size(group)
        dist.all_gather_object(
            gathered, {"has_optimizer": optimizer is not None}, group=group
        )
        if len({e["has_optimizer"] for e in gathered if e}) != 1:
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                "the ranks disagree about owning a stage-local optimizer "
                f"({gathered}); a PP pipeline cannot be half-transactional",
            )

    transaction = None
    base_loss = make_last_stage_loss_fn(
        is_last=layout.is_last, microbatches=config.microbatches
    )
    if optimizer is not None:
        clip = clip_gradients
        if clip is None and max_grad_norm is not None:
            clip = make_clip_gradients(parameters, max_grad_norm)
        transaction = PpFp16Transaction(
            scaler,
            optimizer,
            group=group,
            is_last=layout.is_last,
            clip_gradients=clip,
        )
        if layout.is_last:
            base_loss = transaction.wrap_loss(base_loss)
    schedule = build_anima_schedule(
        stage,
        microbatches=config.microbatches,
        schedule=config.schedule,
        loss_fn=base_loss,
    )
    return Krea2PipelineRuntime(
        model=model,
        layout=layout,
        stage_module=stage_module,
        stage=stage,
        schedule=schedule,
        config=config,
        wire=wire,
        dtype=compute_dtype,
        compute_policy=policy,
        transaction=transaction,
        optimizer=optimizer,
        group=group,
    )


def _prepare_first_stage_inputs(
    latents: Tensor,
    wire: Krea2WireSpec,
    *,
    timesteps: Tensor | None,
    context: Tensor | None,
    mask: Tensor | None,
):
    """5D latent -> (patchified tokens, 3D pos, joint key-padding mask)."""

    from library.models.krea2_raw.family import prepare_img_tokens

    if timesteps is None or context is None:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT, "the first stage needs timesteps and context"
        )
    latents_4d = latents.squeeze(2)
    batch = int(latents_4d.shape[0])
    txtlen = int(context.shape[1])
    if mask is None:
        txtmask = torch.ones(batch, txtlen, dtype=torch.bool, device=latents_4d.device)
    else:
        txtmask = mask if mask.dtype == torch.bool else mask > 0.5
        if txtmask.shape != (batch, txtlen):
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                f"text mask must be {(batch, txtlen)}, got {tuple(txtmask.shape)}",
            )
    return prepare_img_tokens(latents_4d, txtlen, int(wire.patch), txtmask)


def _unpatchify(prediction: Tensor, wire: Krea2WireSpec) -> Tensor:
    """``[B, L_img, patch*patch*C]`` -> 5D latent velocity, like forward_for_loss."""

    from einops import rearrange

    patch = int(wire.patch)
    channels = KREA2_LATENT_CHANNELS
    batch, _, _t, height, width = wire.latent_shape
    latent_h, latent_w = height // patch, width // patch
    velocity_4d = rearrange(
        prediction,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=latent_h,
        w=latent_w,
        c=channels,
        ph=patch,
        pw=patch,
    )
    return velocity_4d.unsqueeze(2)
