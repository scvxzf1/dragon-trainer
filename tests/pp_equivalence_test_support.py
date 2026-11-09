"""CPU + Gloo equivalence harness for Anima pipeline parallelism (PP=2).

Task t3.  The harness runs two paths on a tiny Anima DiT and compares them:

* **single-device reference** — the authoritative
  :meth:`library.anima.models.Anima.forward_mini_train_dit` on one process;
* **2-rank PP** — a real ``torch.distributed.pipelining`` stage split over two
  Gloo ranks on CPU.

Equality policy (deliberately narrow, do not widen it):

``microbatches == 1``
    The stage-boundary activation, the prediction, the loss and the adapter
    gradients must be **bit-identical** (``torch.equal``).  A single microbatch
    is the only case where the PP split changes neither the batch shapes nor the
    gradient accumulation order, so it is the only case where bit-equality is
    claimed.

``microbatches >= 2``
    Splitting the batch changes (a) the shapes each GEMM sees and (b) the order
    in which per-microbatch contributions are summed, so only **numerical**
    equivalence is claimed.  Every comparison carries ``max_abs_diff``,
    ``max_rel_diff`` and ``rel_to_scale`` (largest deviation relative to the
    largest compared magnitude) and the case is explicitly *not* labelled
    bit-exact.

    The report additionally compares against a *microbatched* single-device
    reference (per-microbatch forward/backward on one rank, summed and scaled by
    ``1/n``).  That isolates "PP added drift" from "microbatch splitting added
    drift"; when it comes out exactly equal it is reported as evidence, never as
    a licence to call the ``microbatches >= 2`` case bit-exact.

Torch 2.10 facts this harness had to work around (each is pinned by a test):

1. ``PipelineScheduleSingle.__init__`` rejects ``n_microbatches < num_stages``
   for **every** schedule, including :class:`ScheduleGPipe`
   (``schedules.py:579``).  A 2-stage pipeline therefore cannot run
   ``microbatches=1`` through a stock schedule; the harness drives that case
   through the public per-chunk stage API (``get_*_ops`` /
   ``forward_one_chunk`` / ``backward_one_chunk`` / ``perform_reduce_grad``),
   which is the op sequence ``ScheduleGPipe`` performs for a single chunk.
2. ``scale_grads`` divides ``self.submod.parameters()``.  A LoRA adapter that is
   monkey-patched onto frozen ``nn.Linear`` modules is **not** part of the
   model's parameter tree, so without registering the adapters inside the stage
   module the microbatches' gradients are never averaged and come out ``n``
   times too large (measured: 2x on this fixture, i.e. a 4.4e-2 absolute error).
   The stage module registers its owned adapters in ``self.adapters`` for exactly
   that reason; ``test_lora_adapters_are_not_in_the_model_parameter_tree`` pins
   the underlying fact.
3. ``Block``/``FinalLayer`` zero-initialise their AdaLN output projections
   (``models.py:1279``, ``:1468-1474``).  On a stock model the final layer's
   ``emb`` term is dead and the per-block ``adaln_up_*`` path is dead, so a
   fixture built on a fresh model cannot detect wrong-microbatch conditioning.
   :func:`build_micro_anima` de-zeroes both and
   ``test_fixture_conditioning_paths_are_live`` proves the paths are live.
4. ``_run_blocks`` indexes ``_mod_guidance_schedule`` with the **local** block
   index (``models.py:2565``), so per-block modulation guidance cannot be
   reproduced on a sliced stage.  The harness keeps guidance off and lets
   ``pp.assert_mod_guidance_indexable`` fail closed instead of misapplying it.

Frozen conditioning stays **off** the stage boundary (the option recommended by
the t1 audit): only the differentiable block-stack activation crosses the wire.
The post stage recomputes the timestep conditioning from its own copy of the
microbatch and holds the shape-derived RoPE cache as a non-persistent buffer, so
no frozen tensor is exchanged and no per-tensor gradient "skip" override is
needed.  :func:`grad_exchange_plan` documents the only consistent policy on
torch 2.10: the receiving stage mirrors the sender's activation send info, so
every tensor on the wire exchanges gradients — the only way to skip a tensor is
to keep it off the wire.

The harness is CPU-only by contract; it never touches CUDA and never asserts
anything about GPU numerics.

This module owns the fixture, the two stage modules and the comparison
primitives.  The two-rank driver (case configuration, process spawn, report) is
``tests/pp_equivalence_dist_test_support.py``; the assertions live in
``tests/test_anima_pp_equivalence.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- fixture constants -------------------------------------------------------

MICRO_BLOCKS = 4
MICRO_CHANNELS = 24
BLOCK_SPLIT = 2
MODEL_SEED = 123
ADALN_DENZERO_SEED_OFFSET = 7
ADAPTER_SEED = 4242
ADAPTER_UP_STD = 0.02
BATCH_SEED = 999
LORA_RANK = 4
LORA_ALPHA = 4.0
DIST_TIMEOUT = timedelta(seconds=90)
JOIN_TIMEOUT_SECONDS = 100

#: Relative-error floor: elementwise ``|diff| / max(|expected|, floor)`` with
#: ``floor = REL_FLOOR * max(|expected|)``.  Without a scale-aware floor a single
#: near-zero element makes the elementwise relative error meaningless.
REL_FLOOR = 1e-6

# --- boundary description ----------------------------------------------------


@dataclass(frozen=True)
class BoundaryEntry:
    """One tensor that may or may not cross the stage boundary."""

    name: str
    requires_grad: bool
    on_wire: bool
    reason: str


def boundary_spec(*, wire_conditioning: bool = False) -> tuple[BoundaryEntry, ...]:
    """Describe the boundary for one harness mode.

    ``wire_conditioning=False`` (default) sends only the block-stack activation.
    ``wire_conditioning=True`` additionally puts the frozen timestep
    conditioning on the wire, which exercises the "frozen tensor on the
    boundary" case that both stages must still exchange gradients for.
    """

    return (
        BoundaryEntry(
            name="hidden",
            requires_grad=True,
            on_wire=True,
            reason="block-stack activation; the only differentiable boundary value",
        ),
        BoundaryEntry(
            name="t_embedding",
            requires_grad=False,
            on_wire=bool(wire_conditioning),
            reason=(
                "frozen timestep embedding; sent only in wire_conditioning mode"
                if wire_conditioning
                else "frozen conditioning recomputed on the post stage"
            ),
        ),
        BoundaryEntry(
            name="adaln_lora",
            requires_grad=False,
            on_wire=bool(wire_conditioning),
            reason=(
                "frozen AdaLN-LoRA modulation; sent only in wire_conditioning mode"
                if wire_conditioning
                else "frozen conditioning recomputed on the post stage"
            ),
        ),
        BoundaryEntry(
            name="rope_cos_sin",
            requires_grad=False,
            on_wire=False,
            reason="shape-derived constant; materialised as a stage buffer",
        ),
        BoundaryEntry(
            name="crossattn_emb",
            requires_grad=False,
            on_wire=False,
            reason="per-microbatch data both ranks already hold",
        ),
    )


def grad_exchange_plan(spec: Sequence[BoundaryEntry]) -> tuple[bool, ...]:
    """Whether gradients are exchanged for each entry (``True`` = exchanged).

    torch 2.10 mirrors the sender's activation send info into the receiver's
    gradient receive info and sends a gradient for every received input
    (``stage.py:261-282``, ``:464-506``).  Skipping one tensor on the sender side
    therefore desynchronises the p2p chain and deadlocks.  The only consistent
    plan is "exchange for everything on the wire"; keeping a tensor off the wire
    is the way to skip it.
    """

    return tuple(entry.on_wire for entry in spec)


def skipped_entries(spec: Sequence[BoundaryEntry]) -> tuple[str, ...]:
    return tuple(entry.name for entry in spec if not entry.on_wire)


# --- fixture -----------------------------------------------------------------


def build_micro_anima(
    *,
    seed: int = MODEL_SEED,
    num_blocks: int = MICRO_BLOCKS,
    dezero_adaln: bool = True,
) -> tuple[nn.Module, Any]:
    """Build the tiny frozen Anima DiT plus its plain MLP LoRA adapter.

    ``dezero_adaln`` replaces the zero-initialised AdaLN output projections with
    small random values.  Without it the timestep path is partly dead on a fresh
    model (see the module docstring, fact 3) and the numeric comparisons below
    would be vacuous.
    """

    from library.anima.models import Anima
    from scripts.experiments.anima_parallel.common import create_mlp_lora

    torch.manual_seed(seed)
    model = Anima(
        max_img_h=4,
        max_img_w=4,
        max_frames=1,
        in_channels=4,
        out_channels=4,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=MICRO_CHANNELS,
        num_blocks=num_blocks,
        num_heads=2,
        mlp_ratio=2.0,
        crossattn_emb_channels=MICRO_CHANNELS,
        pos_emb_learnable=True,
        use_adaln_lora=True,
        adaln_lora_dim=8,
        use_llm_adapter=False,
        attn_mode="torch",
    )
    if dezero_adaln:
        _dezero_adaln(model, seed=seed)
    model.eval()
    model.requires_grad_(False)
    network = create_mlp_lora(
        model, seed=seed + 1, rank_dim=LORA_RANK, alpha=LORA_ALPHA
    )
    generator = torch.Generator().manual_seed(ADAPTER_SEED)
    with torch.no_grad():
        for lora in network.unet_loras:
            # ``create_network`` zero-initialises ``lora_up``; give it a small
            # non-zero value so the adapter actually moves the output and its
            # gradient is a real signal.
            lora.lora_up.weight.copy_(
                torch.randn(lora.lora_up.weight.shape, generator=generator)
                * ADAPTER_UP_STD
            )
    return model.train(), network


def _dezero_adaln(model: nn.Module, *, seed: int) -> None:
    """Undo the zero-initialised AdaLN output projections.

    ``Block`` zeroes ``adaln_up_self_attn/cross_attn/mlp`` (``models.py:1468``)
    and the modulation output projections are zeroed too (``models.py:1279``,
    ``:1472-1474``).  Both are replaced with ``N(0, 0.1)`` drawn from an explicit
    generator, so the fixture is reproducible independently of any other RNG
    use.
    """

    generator = torch.Generator().manual_seed(seed + ADALN_DENZERO_SEED_OFFSET)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.dim() < 2:
                continue
            if "adaln_modulation" not in name and "adaln_up_" not in name:
                continue
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def build_micro_batch(
    count: int, *, seed: int = BATCH_SEED
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Fixed-shape micro batch: ``(x, timesteps, context, padding_mask)``."""

    generator = torch.Generator().manual_seed(seed)
    inputs = (
        torch.randn(count, 4, 1, 4, 4, generator=generator),
        torch.rand(count, generator=generator),
        torch.randn(count, 3, MICRO_CHANNELS, generator=generator),
        torch.zeros(count, 1, 4, 4),
    )
    target = torch.randn(count, 4, 1, 4, 4, generator=generator)
    return inputs, target


def build_fixture(
    *,
    microbatches: int,
    dezero_adaln: bool = True,
    num_blocks: int = MICRO_BLOCKS,
    seed: int = MODEL_SEED,
) -> tuple[nn.Module, Any, tuple[torch.Tensor, ...], torch.Tensor]:
    """Model + adapter + batch.  Two calls with the same seed are identical."""

    model, network = build_micro_anima(
        seed=seed, num_blocks=num_blocks, dezero_adaln=dezero_adaln
    )
    inputs, target = build_micro_batch(microbatches)
    return model, network, inputs, target


def zero_adapter_grads(network: Any) -> None:
    for parameter in network.parameters():
        parameter.grad = None


def adapter_grads(
    network: Any, *, blocks: Iterable[int] | None = None
) -> dict[str, torch.Tensor]:
    """Adapter gradients keyed by ``<lora_name>.<parameter>``.

    ``blocks`` restricts the result to the adapter modules owned by one stage so
    the two ranks can be compared without exchanging gradients.
    """

    from scripts.experiments.anima_parallel.common import lora_block_index

    wanted = None if blocks is None else set(blocks)
    grads: dict[str, torch.Tensor] = {}
    for lora in network.unet_loras:
        if wanted is not None and lora_block_index(lora) not in wanted:
            continue
        for name, parameter in lora.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(
                    f"adapter parameter {lora.lora_name}.{name} has no gradient; "
                    "the backward pass did not reach this stage's adapter"
                )
            grads[f"{lora.lora_name}.{name}"] = parameter.grad.detach().clone()
    return grads


def owned_blocks(
    rank: int, *, split: int = BLOCK_SPLIT, total: int = MICRO_BLOCKS
) -> range:
    """Contiguous block ids owned by one rank."""

    if rank == 0:
        return range(0, split)
    return range(split, total)


# --- authoritative single-device references ----------------------------------


@dataclass(frozen=True)
class ReferenceResult:
    """Outcome of a single-device step."""

    prediction: torch.Tensor
    losses: tuple[float, ...]
    grads: dict[str, torch.Tensor]

    @property
    def loss(self) -> float:
        return sum(self.losses) / len(self.losses)


def reference_step(
    model: nn.Module,
    network: Any,
    inputs: Sequence[torch.Tensor],
    target: torch.Tensor,
    *,
    microbatched: bool,
) -> ReferenceResult:
    """Run one single-device training step.

    ``microbatched=False`` runs the authoritative full-batch forward and one
    backward over the whole batch — the reference the PP path must match.
    ``microbatched=True`` runs one forward/backward per sample and scales by
    ``1/n``, i.e. the same arithmetic the PP path performs, so it isolates
    pipeline drift from microbatch-splitting drift.
    """

    zero_adapter_grads(network)
    count = target.shape[0]
    if not microbatched:
        prediction = model(*inputs[:3], padding_mask=inputs[3])
        loss = F.mse_loss(prediction, target)
        loss.backward()
        return ReferenceResult(
            prediction=prediction.detach().clone(),
            losses=(float(loss.detach()),),
            grads=adapter_grads(network),
        )

    predictions: list[torch.Tensor] = []
    losses: list[float] = []
    for index in range(count):
        chunk = tuple(value[index : index + 1] for value in inputs)
        prediction = model(*chunk[:3], padding_mask=chunk[3])
        loss = F.mse_loss(prediction, target[index : index + 1])
        (loss / count).backward()
        predictions.append(prediction.detach().clone())
        losses.append(float(loss.detach()))
    return ReferenceResult(
        prediction=torch.cat(predictions),
        losses=tuple(losses),
        grads=adapter_grads(network),
    )


def reference_boundary(
    model: nn.Module, inputs: Sequence[torch.Tensor], *, split: int = BLOCK_SPLIT
) -> tuple[torch.Tensor, ...]:
    """Per-microbatch boundary activation the pre stage must emit.

    Runs the pre-block region on each microbatch separately through the same
    helpers the PP pre stage uses (``prepare_block_inputs`` /
    ``run_local_blocks`` restricted to ``blocks[:split]``).
    """

    from scripts.experiments.anima_parallel import pipeline_parallel as pp

    blocks = model.blocks
    boundaries: list[torch.Tensor] = []
    try:
        model.blocks = nn.ModuleList(list(blocks)[:split])
        for index in range(inputs[0].shape[0]):
            chunk = tuple(value[index : index + 1] for value in inputs)
            with torch.no_grad():
                hidden, embedding, params, kwargs = pp.prepare_block_inputs(
                    model, *chunk
                )
                hidden = pp.run_local_blocks(
                    model, hidden, embedding, chunk[2], params, kwargs
                )
            boundaries.append(hidden.detach().clone())
    finally:
        model.blocks = blocks
    return tuple(boundaries)


def reference_boundary_full_batch(
    model: nn.Module, inputs: Sequence[torch.Tensor], *, split: int = BLOCK_SPLIT
) -> tuple[torch.Tensor, ...]:
    """Boundary activation produced by one *full-batch* forward, split per sample.

    This is the boundary the single-device path would produce for the same
    microbatches, so comparing against it exposes the batching-order drift that
    only ``microbatches >= 2`` can have.
    """

    from scripts.experiments.anima_parallel import pipeline_parallel as pp

    blocks = model.blocks
    try:
        model.blocks = nn.ModuleList(list(blocks)[:split])
        with torch.no_grad():
            hidden, embedding, params, kwargs = pp.prepare_block_inputs(model, *inputs)
            hidden = pp.run_local_blocks(
                model, hidden, embedding, inputs[2], params, kwargs
            )
    finally:
        model.blocks = blocks
    return tuple(
        hidden[index : index + 1].detach().clone()
        for index in range(hidden.shape[0])
    )


def timestep_conditioning(
    model: nn.Module, timesteps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Timestep region of the authoritative pre-block code (``models.py:2676-2679``).

    The post stage has no ``x_embedder``/``pos_embedder`` (they belong to the pre
    stage) but must reproduce this region bit-exactly, so it is factored out
    here and pinned against ``pp.prepare_block_inputs`` by
    ``test_timestep_conditioning_matches_prepare_block_inputs``.
    """

    from scripts.experiments.anima_parallel import pipeline_parallel as pp

    pp.assert_mini_forward_supported(model)
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    embedding, adaln = model.t_embedder(timesteps)
    return model.t_embedding_norm(embedding), adaln


# --- PP stages ---------------------------------------------------------------


def lora_block_of(lora: Any) -> int:
    from scripts.experiments.anima_parallel.common import lora_block_index

    return lora_block_index(lora)


class AnimaPPStageModule(nn.Module):
    """One owned stage of a two-stage Anima split.

    The module keeps only the submodules this stage owns plus the frozen
    timestep conditioning it needs, and registers the adapters of the blocks it
    owns inside its own parameter tree (see module docstring, fact 2).
    """

    def __init__(
        self,
        model: nn.Module,
        network: Any,
        *,
        rank: int,
        split: int,
        rope: tuple[torch.Tensor, torch.Tensor],
        batch_inputs: Sequence[torch.Tensor],
        wire_conditioning: bool = False,
    ) -> None:
        super().__init__()
        from scripts.experiments.anima_parallel import pipeline_parallel as pp

        total = len(model.blocks)
        if rank not in (0, 1):
            raise ValueError(f"rank must be 0 or 1, got {rank}")
        if not 0 < split < total:
            raise ValueError(f"split must be inside (0, {total}), got {split}")
        pp.assert_mini_forward_supported(model)

        self.rank = rank
        self.split = split
        self.total_blocks = total
        self.wire_conditioning = bool(wire_conditioning)
        self.start, self.end = (0, split) if rank == 0 else (split, total)
        owned = set(range(self.start, self.end))

        blocks = list(model.blocks)
        model.blocks = nn.ModuleList(blocks[self.start : self.end])
        keep = {"blocks", "t_embedder", "t_embedding_norm"}
        keep |= {"x_embedder", "pos_embedder"} if rank == 0 else {"final_layer"}
        for name in list(model._modules):
            if name not in keep:
                delattr(model, name)
        self.model = model
        self.blocks = model.blocks

        self.adapters = nn.ModuleDict(
            {
                lora.lora_name: lora
                for lora in network.unet_loras
                if lora_block_of(lora) in owned
            }
        )
        self.register_buffer("rope_cos", rope[0].detach().clone(), persistent=False)
        self.register_buffer("rope_sin", rope[1].detach().clone(), persistent=False)
        # CPU-only harness: plain attributes need no device move.  Both ranks
        # hold the same micro batch (that is how a real PP run feeds a stage).
        self.batch_inputs = tuple(value.detach() for value in batch_inputs)
        self._cursor = 0
        self._hidden_shape: tuple[int, ...] | None = None
        self.boundary_out: list[torch.Tensor] = []
        self.boundary_in: list[torch.Tensor] = []
        # Fail closed when per-block modulation guidance is active: a sliced
        # block stack cannot index the global schedule (docstring, fact 4).
        pp.assert_mod_guidance_indexable(model)

    # -- helpers ---------------------------------------------------------

    def reset_microbatch_cursor(self) -> None:
        self._cursor = 0

    def _next_microbatch(self) -> int:
        index = self._cursor
        if index >= self.batch_inputs[0].shape[0]:
            raise RuntimeError(
                f"stage {self.rank} ran more forwards than there are microbatches "
                f"({self.batch_inputs[0].shape[0]})"
            )
        self._cursor += 1
        return index

    def _check_shape(self, hidden: torch.Tensor) -> None:
        shape = tuple(hidden.shape)
        if self._hidden_shape is None:
            self._hidden_shape = shape
        elif shape != self._hidden_shape:
            raise RuntimeError(
                "PP stage received a different fixed-shape microbatch: "
                f"{shape} != {self._hidden_shape}"
            )

    def _block_kwargs(self, adaln: torch.Tensor) -> dict[str, Any]:
        return {
            "rope_cos_sin": (self.rope_cos, self.rope_sin),
            # ``run_local_blocks`` normalises ``use_fp32`` and the RoPE dtype from
            # the compute dtype, so this value is never trusted.
            "adaln_lora_B_T_3D": adaln,
            "use_fp32": False,
        }

    def _attention_params(self):
        from networks import attention_dispatch

        return attention_dispatch.AttentionParams.create_attention_params(
            self.model.attn_mode,
            self.model.attn_softmax_scale,
            v100_flash_stability=self.model.v100_flash_stability,
            debug_finite_checks=self.model.debug_finite_checks,
        )

    # -- forward ---------------------------------------------------------

    def forward(self, *args):
        if self.rank == 0:
            return self._forward_first(*args)
        return self._forward_last(*args)

    def _forward_first(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        from scripts.experiments.anima_parallel import pipeline_parallel as pp

        hidden, embedding, params, kwargs = pp.prepare_block_inputs(
            self.model, x, timesteps, context, padding_mask
        )
        hidden = pp.run_local_blocks(
            self.model, hidden, embedding, context, params, kwargs
        )
        self.boundary_out.append(hidden.detach().clone())
        if self.wire_conditioning:
            return hidden, embedding, kwargs["adaln_lora_B_T_3D"]
        return hidden

    def _forward_last(self, *args):
        from scripts.experiments.anima_parallel import pipeline_parallel as pp

        if self.wire_conditioning:
            hidden, embedding, adaln = args
        else:
            (hidden,) = args
        self._check_shape(hidden)
        self.boundary_in.append(hidden.detach().clone())
        index = self._next_microbatch()
        context = self.batch_inputs[2][index : index + 1]
        if not self.wire_conditioning:
            timesteps = self.batch_inputs[1][index : index + 1]
            embedding, adaln = timestep_conditioning(self.model, timesteps)
        params = self._attention_params()
        hidden = pp.run_local_blocks(
            self.model, hidden, embedding, context, params, self._block_kwargs(adaln)
        )
        return pp.finish_model(self.model, hidden, embedding, adaln)


@dataclass(frozen=True)
class StageExamples:
    """Static wire metadata for ``PipelineStage``."""

    boundary: torch.Tensor
    prediction: torch.Tensor
    rope: tuple[torch.Tensor, torch.Tensor]
    t_embedding: torch.Tensor
    adaln: torch.Tensor


def stage_examples(
    model: nn.Module, inputs: Sequence[torch.Tensor], *, split: int = BLOCK_SPLIT
) -> StageExamples:
    """Derive the boundary/output/conditioning examples before module transfer."""

    from scripts.experiments.anima_parallel import pipeline_parallel as pp

    blocks = list(model.blocks)
    example = tuple(value[:1] for value in inputs)
    try:
        with torch.no_grad():
            hidden, embedding, params, kwargs = pp.prepare_block_inputs(model, *example)
            rope = kwargs["rope_cos_sin"]
            adaln = kwargs["adaln_lora_B_T_3D"]
            model.blocks = nn.ModuleList(blocks[:split])
            boundary = pp.run_local_blocks(
                model, hidden, embedding, example[2], params, kwargs
            )
            model.blocks = nn.ModuleList(blocks[split:])
            tail = pp.run_local_blocks(
                model, boundary, embedding, example[2], params, kwargs
            )
            prediction = pp.finish_model(model, tail, embedding, adaln)
    finally:
        model.blocks = nn.ModuleList(blocks)
    return StageExamples(
        boundary=boundary.detach().clone(),
        prediction=prediction.detach().clone(),
        rope=rope,
        t_embedding=embedding.detach().clone(),
        adaln=adaln.detach().clone(),
    )


@dataclass(frozen=True)
class LocalPPResult:
    """Result of running both stages in one process (no ``torch.distributed``)."""

    prediction: torch.Tensor
    boundary: tuple[torch.Tensor, ...]
    modules: tuple["AnimaPPStageModule", "AnimaPPStageModule"]


def local_pp_forward(
    inputs: Sequence[torch.Tensor],
    *,
    split: int = BLOCK_SPLIT,
    wire_conditioning: bool = False,
    dezero_adaln: bool = True,
    post_microbatch_order: Sequence[int] | None = None,
) -> LocalPPResult:
    """Run the split forward without any distributed machinery.

    This is the split *without* the wire: useful for pinning the decomposition
    against ``forward_mini_train_dit`` and as a negative control — passing a
    permuted ``post_microbatch_order`` feeds the post stage the wrong
    microbatch's conditioning, which the harness must be able to detect.
    """

    model0, network0 = build_micro_anima(dezero_adaln=dezero_adaln)
    rope = stage_examples(model0, inputs, split=split).rope
    model1, network1 = build_micro_anima(dezero_adaln=dezero_adaln)
    pre = AnimaPPStageModule(
        model0,
        network0,
        rank=0,
        split=split,
        rope=rope,
        batch_inputs=inputs,
        wire_conditioning=wire_conditioning,
    )
    post_inputs: Sequence[torch.Tensor] = inputs
    if post_microbatch_order is not None:
        order = torch.as_tensor(list(post_microbatch_order), dtype=torch.long)
        post_inputs = tuple(value.index_select(0, order) for value in inputs)
    post = AnimaPPStageModule(
        model1,
        network1,
        rank=1,
        split=split,
        rope=rope,
        batch_inputs=post_inputs,
        wire_conditioning=wire_conditioning,
    )

    predictions: list[torch.Tensor] = []
    boundaries: list[torch.Tensor] = []
    for index in range(inputs[0].shape[0]):
        chunk = tuple(value[index : index + 1] for value in inputs)
        produced = pre(*chunk)
        boundary = produced[0] if wire_conditioning else produced
        boundaries.append(boundary.detach().clone())
        prediction = post(*produced) if wire_conditioning else post(produced)
        predictions.append(prediction.detach().clone())
    return LocalPPResult(
        prediction=torch.cat(predictions),
        boundary=tuple(boundaries),
        modules=(pre, post),
    )


def build_counting_stage_class():
    """Build the counting ``PipelineStage`` subclass.

    ``torch.distributed.pipelining`` is imported inside the worker process, so
    the subclass cannot be declared at module import time.  The counters are the
    evidence that the boundary really is as small as the specification claims
    and that both stages agree on it.
    """

    from torch.distributed.pipelining import PipelineStage

    class _CountingPipelineStage(PipelineStage):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.communication: dict[str, int] = {
                "fwd_send_calls": 0,
                "fwd_send_bytes": 0,
                "fwd_recv_calls": 0,
                "fwd_recv_bytes": 0,
                "bwd_send_calls": 0,
                "bwd_send_bytes": 0,
                "bwd_recv_calls": 0,
                "bwd_recv_bytes": 0,
            }

        def _count(self, kind: str, ops):
            self.communication[f"{kind}_calls"] += len(ops)
            self.communication[f"{kind}_bytes"] += sum(
                op.tensor.numel() * op.tensor.element_size() for op in ops
            )
            return ops

        def get_fwd_send_ops(self, chunk_id):
            return self._count("fwd_send", super().get_fwd_send_ops(chunk_id))

        def get_fwd_recv_ops(self, chunk_id):
            return self._count("fwd_recv", super().get_fwd_recv_ops(chunk_id))

        def get_bwd_send_ops(self, chunk_id):
            return self._count("bwd_send", super().get_bwd_send_ops(chunk_id))

        def get_bwd_recv_ops(self, chunk_id):
            return self._count("bwd_recv", super().get_bwd_recv_ops(chunk_id))

    return _CountingPipelineStage


def build_pipeline_stage(
    module: AnimaPPStageModule,
    *,
    input_args: Sequence[torch.Tensor],
    output_args: Sequence[torch.Tensor],
):
    """Wrap a stage module in the counting ``PipelineStage``."""

    stage_type = build_counting_stage_class()
    return stage_type(
        module,
        module.rank,
        2,
        torch.device("cpu"),
        input_args=tuple(input_args),
        output_args=tuple(output_args),
    )


def _run_ops(ops) -> None:
    """Execute a batch of p2p ops and wait for them (``[]`` is a no-op)."""

    import torch.distributed as dist

    if not ops:
        return
    for work in dist.batch_isend_irecv(ops):
        work.wait()


def step_single_microbatch(
    stage,
    *,
    rank: int,
    inputs: Sequence[torch.Tensor],
    target: torch.Tensor,
) -> tuple[list[float], torch.Tensor | None]:
    """Drive one microbatch through the public per-chunk stage API.

    ``ScheduleGPipe``/``Schedule1F1B`` reject ``n_microbatches < num_stages``
    (``schedules.py:579``), so a 2-stage pipeline cannot run ``microbatches=1``
    through a stock schedule.  This is the same op sequence
    ``ScheduleGPipe._step_microbatches`` performs for a single chunk: receive,
    forward, send, compute the loss on the last stage, receive gradients,
    backward, send gradients, then scale.

    ``_prepare_forward_infra``/``_prepare_backward_infra`` are the stage's
    internal setup hooks; the stock schedules call exactly these.
    """

    stage.has_backward = True
    stage.clear_runtime_states()
    stage._prepare_forward_infra(1, tuple(inputs) if rank == 0 else (), {})
    stage._prepare_backward_infra(1)

    args = tuple(value[:1] for value in inputs) if rank == 0 else ()
    _run_ops(stage.get_fwd_recv_ops(0))
    output = stage.forward_one_chunk(0, args, {}, save_forward_output=True)
    _run_ops(stage.get_fwd_send_ops(0))
    loss = F.mse_loss(output.float(), target[:1].float()) if rank == 1 else None
    _run_ops(stage.get_bwd_recv_ops(0))
    stage.backward_one_chunk(0, loss=loss, last_backward=True)
    _run_ops(stage.get_bwd_send_ops(0))
    stage.perform_reduce_grad(1)
    return (
        [float(loss.detach())] if loss is not None else [],
        output.detach().clone() if rank == 1 else None,
    )


def build_schedule(stage, *, microbatches: int, schedule: str):
    """Build a stock schedule, failing loudly on the torch 2.10 constraint."""

    from torch.distributed.pipelining import Schedule1F1B, ScheduleGPipe

    normalized = schedule.strip().lower()
    if normalized not in {"gpipe", "1f1b"}:
        raise ValueError(f"unknown schedule {schedule!r}")
    if microbatches < 2:
        raise ValueError(
            "stock pipeline schedules require n_microbatches >= num_stages on "
            f"torch {torch.__version__}; got {microbatches} for 2 stages"
        )
    schedule_type = ScheduleGPipe if normalized == "gpipe" else Schedule1F1B
    return schedule_type(
        stage,
        n_microbatches=microbatches,
        loss_fn=lambda prediction, target: F.mse_loss(
            prediction.float(), target.float()
        ),
        scale_grads=True,
    )


def step_pipeline(
    schedule,
    *,
    rank: int,
    inputs: Sequence[torch.Tensor],
    target: torch.Tensor,
) -> tuple[list[float], torch.Tensor | None]:
    """Run one schedule step; only the last stage returns a prediction."""

    losses: list[float] = []
    prediction = None
    if rank == 0:
        schedule.step(*inputs, losses=losses)
    else:
        prediction = schedule.step(target=target, losses=losses)
    return (
        [float(value.detach()) for value in losses],
        None if prediction is None else prediction.detach().clone(),
    )


# --- comparison --------------------------------------------------------------


@dataclass(frozen=True)
class TensorComparison:
    """Bit-level and numeric comparison of two tensors.

    ``max_rel_diff`` is elementwise and therefore pessimistic for entries whose
    reference magnitude is near zero.  ``rel_to_scale`` is the largest absolute
    deviation relative to the largest reference magnitude in the compared set —
    the number the tests assert on, because it answers "how large is the drift
    compared with the quantity being compared".
    """

    name: str
    equal: bool
    max_abs_diff: float
    max_rel_diff: float
    rel_to_scale: float
    numel: int
    actual_abs_max: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "equal": self.equal,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "rel_to_scale": self.rel_to_scale,
            "numel": self.numel,
            "actual_abs_max": self.actual_abs_max,
        }


def compare_tensors(
    name: str, actual: torch.Tensor, expected: torch.Tensor
) -> TensorComparison:
    """Compare two tensors exactly (``equal``) and numerically."""

    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    diff = (actual - expected).abs()
    scale = float(expected.abs().max()) if expected.numel() else 0.0
    floor = max(REL_FLOOR * scale, 1e-12)
    denominator = expected.abs().clamp_min(floor)
    max_abs_diff = float(diff.max()) if diff.numel() else 0.0
    return TensorComparison(
        name=name,
        equal=bool(torch.equal(actual, expected)),
        max_abs_diff=max_abs_diff,
        max_rel_diff=float((diff / denominator).max()) if diff.numel() else 0.0,
        rel_to_scale=max_abs_diff / max(scale, 1e-12),
        numel=int(diff.numel()),
        actual_abs_max=float(actual.abs().max()) if actual.numel() else 0.0,
    )


def compare_tensor_sequences(
    name: str,
    actual: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
) -> TensorComparison:
    """Compare per-microbatch tensors and report the worst entry."""

    if len(actual) != len(expected):
        raise AssertionError(
            f"{name}: {len(actual)} tensors compared against {len(expected)}"
        )
    comparisons = [
        compare_tensors(f"{name}[{index}]", a, b)
        for index, (a, b) in enumerate(zip(actual, expected))
    ]
    worst = max(comparisons, key=lambda item: item.max_abs_diff)
    scale = max(item.actual_abs_max for item in comparisons)
    return TensorComparison(
        name=name,
        equal=all(item.equal for item in comparisons),
        max_abs_diff=worst.max_abs_diff,
        max_rel_diff=max(item.max_rel_diff for item in comparisons),
        rel_to_scale=worst.max_abs_diff / max(scale, 1e-12),
        numel=sum(item.numel for item in comparisons),
        actual_abs_max=scale,
    )


def compare_grads(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    name: str = "grads",
) -> dict[str, Any]:
    """Compare adapter gradients key by key, reporting the worst entry."""

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise AssertionError(
            f"{name}: key mismatch missing={missing[:4]} extra={extra[:4]}"
        )
    comparisons = [
        compare_tensors(key, actual[key], expected[key]) for key in sorted(expected)
    ]
    worst = max(comparisons, key=lambda item: item.max_abs_diff)
    scale = max(item.actual_abs_max for item in comparisons)
    return {
        "name": name,
        "equal": all(item.equal for item in comparisons),
        "max_abs_diff": worst.max_abs_diff,
        "max_rel_diff": max(item.max_rel_diff for item in comparisons),
        "rel_to_scale": worst.max_abs_diff / max(scale, 1e-12),
        "worst_entry": worst.name,
        "entries": len(comparisons),
        "numel": sum(item.numel for item in comparisons),
        "actual_abs_max": scale,
    }


#: Track ② threshold (user ruling): the PP gradient vector must *point the same
#: way* as the single-device full-batch reference.  Cosine similarity is the
#: metric because float addition is non-associative: splitting the batch changes
#: both the GEMM shapes and the order in which per-microbatch contributions are
#: summed, so an element-wise bound would either flake or hide a real defect.
COSINE_THRESHOLD = 0.9999


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Cosine similarity of two vectors, evaluated in float64."""

    left = actual.double().reshape(-1)
    right = expected.double().reshape(-1)
    left_norm = float(left.norm())
    right_norm = float(right.norm())
    if left_norm == 0.0 and right_norm == 0.0:
        # Two all-zero vectors agree by convention; the ratio would be 0/0.
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(torch.dot(left, right) / (left_norm * right_norm))


def gradient_cosine(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    name: str = "grads",
) -> dict[str, Any]:
    """Track ② metric: gradient cosine similarity, per tensor and aggregate.

    ``aggregate`` is computed on the concatenated (sorted-key) gradient vector,
    i.e. the direction agreement of the whole adapter update; ``min_per_tensor``
    catches the case where one adapter points elsewhere while the aggregate
    hides it behind the others.  Both are scale-invariant, so the *magnitude* of
    the update still needs its own (relative) check -- the element-wise
    comparisons next to this one in the equivalence suite.
    """

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise AssertionError(
            f"{name}: key mismatch missing={missing[:4]} extra={extra[:4]}"
        )
    keys = sorted(expected)
    per_tensor = {key: _cosine(actual[key], expected[key]) for key in keys}
    aggregate = _cosine(
        torch.cat([actual[key].reshape(-1) for key in keys]),
        torch.cat([expected[key].reshape(-1) for key in keys]),
    )
    return {
        "name": name,
        "aggregate": aggregate,
        "min_per_tensor": min(per_tensor.values()) if per_tensor else 1.0,
        "per_tensor": per_tensor,
        "entries": len(per_tensor),
    }


def compare_losses(
    name: str, actual: Sequence[float], expected: Sequence[float]
) -> dict[str, Any]:
    if len(actual) != len(expected):
        raise AssertionError(f"{name}: {len(actual)} losses vs {len(expected)}")
    diffs = [abs(a - b) for a, b in zip(actual, expected)]
    scale = max((abs(value) for value in expected), default=0.0)
    floor = max(REL_FLOOR * scale, 1e-12)
    rels = [diff / max(abs(b), floor) for diff, b in zip(diffs, expected)]
    max_abs_diff = max(diffs)
    return {
        "name": name,
        "equal": all(diff == 0.0 for diff in diffs),
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max(rels),
        "rel_to_scale": max_abs_diff / max(scale, 1e-12),
        "numel": len(diffs),
        "actual_abs_max": max((abs(value) for value in actual), default=0.0),
    }
