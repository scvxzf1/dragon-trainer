"""The wired MOSS PP trainer session (train_session dispatch target).

This module is the *only* execution entry for a pipeline-parallel training run
in the production trainer path.  ``train_session._run_training_session`` forks
here after the shared cache / model / network setup and before
``setup_optimizer_and_dataloader``; everything below reuses the shared
skeleton's helpers (dataset strategies, optimizer factory, scheduler builder,
metadata builder, LoRA saver) instead of copying the 840-line session or the
1853-line loop.

Step semantics (frozen contract §4):

* ``global_step`` counts **consumed pipeline steps** (skips included);
* ``effective_step`` advances only on a successful optimizer step;
* the LR scheduler advances only on a successful step -- never together with
  ``global_step``;
* the fp16 skip decision is the PP transaction's unanimous verdict.

Checkpoint semantics:

* per-rank stage weights / optimizer / fp16 state come from
  :mod:`library.training.pp.checkpoint` (new run root per save, negotiated,
  never overwriting user artifacts);
* scheduler / per-rank RNG / data cursor / run identity come from
  :mod:`library.training.pp.trainer_resume`, validated before apply;
* ``accelerator.save_state`` / ``load_state`` are never called (the
  ``pp_accelerate_checkpoint_unsupported`` guard stays armed);
* the final LoRA is written in the ordinary output format by rank 0 after
  gathering every stage's adapter state into its own network.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import time
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from library.training.pp.errors import (
    CODE_BACKEND_MULTIBACKEND_REQUIRED,
    CODE_CONFIG_INVALID,
    CODE_WORLD_SIZE_MISMATCH,
    PipelineParallelError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PpSessionHandoff:
    """Everything the shared session setup already prepared for this run."""

    args: Any
    accelerator: Any
    network: Any
    unet: Any
    vae: Any
    vae_dtype: Any
    text_encoders: Any
    train_dataset_group: Any
    val_dataset_group: Any
    collator: Any
    weight_dtype: torch.dtype
    save_dtype: Any
    tokenizers: Any
    training_started_at: float
    session_id: int
    model_version: Any
    use_user_config: bool
    use_dreambooth_method: bool
    text_encoding_strategy: Any
    tokenize_strategy: Any
    train_unet: bool
    train_text_encoder: bool
    net_kwargs: dict | None = None


@dataclass
class PpLoopCounters:
    global_step: int = 0
    effective_step: int = 0
    skipped: int = 0
    epoch: int = 0
    batch_index: int = 0
    last_loss: float | None = None


def _distributed(group=None) -> bool:
    import torch.distributed as dist

    return bool(dist.is_available() and dist.is_initialized())


def validate_pp_process_group(accelerator) -> None:
    """The shared validator: first init must be the multi-backend PP group."""

    from library.runtime.accelerator import validate_pp_process_group_backend

    validate_pp_process_group_backend(accelerator)


def build_pp_train_dataloader(args, dataset_group, collator):
    """The unsharded, per-epoch-deterministic loader both stages iterate.

    Same dataset, same collator and the same resolve_loader_batch contract as
    the ordinary path; no ``accelerator.prepare`` (that would install a
    DistributedSampler and shard the batch between the stages).  The shuffle
    order is pinned to ``(seed, epoch)`` through a dedicated generator, so the
    batch identity survives a resume without depending on mid-epoch RNG state.
    """

    from library.training.dataloader_batch import resolve_loader_batch

    dataset_group.set_current_strategies()
    batch_plan = resolve_loader_batch(args, dataset_group)
    args._dataloader_batch_plan = batch_plan.as_dict()
    n_workers = min(getattr(args, "max_data_loader_n_workers", 0) or 0, os.cpu_count() or 1)
    kwargs = {
        "batch_size": batch_plan.loader_batch_size,
        "collate_fn": collator,
        "num_workers": n_workers,
        "pin_memory": getattr(args, "dataloader_pin_memory", False),
    }
    if n_workers > 0:
        kwargs["prefetch_factor"] = getattr(args, "dataloader_prefetch_factor", 2)
    generator = torch.Generator()
    generator.manual_seed(int(getattr(args, "seed", 0) or 0))
    return torch.utils.data.DataLoader(dataset_group, shuffle=True, generator=generator, **kwargs)


def derive_run_id(args, family: str = "moss_image") -> str:
    """A deterministic, identical-on-every-rank run identity."""

    import hashlib

    # Run material only: the identity must survive a resume into a different
    # output directory while still binding seed / run name / step budget.
    material = "|".join(
        str(part)
        for part in (
            family,
            getattr(args, "seed", None),
            getattr(args, "output_name", None),
        )
    )
    # Full 64-hex digest: the PP checkpoint identity validator accepts
    # sha256 content digests only.
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def run_identity(run_id: str, args) -> dict[str, str]:
    """The run identity is one pipeline-wide fact: no rank component."""

    # Identity values must be sha256 content digests (validate_identity);
    # ``run_id`` already is the full sha256 of the run material.
    return {"run": run_id}


def build_moss_trainer_runtime(args, accelerator, network, unet, *, latent_shape, max_text_tokens, dtype=None):
    """Build this rank's stage-local runtime (real MOSS PP runtime only)."""

    import torch.distributed as dist

    from library.training.optimizers import get_optimizer
    from library.training.pp.contract import PipelineRuntimeConfig
    from library.training.pp.moss_runtime import build_moss_pipeline_runtime

    # LoRA masters stay FP32; the network moves device-only before the stage
    # module captures its parameter references (never a full-model dtype cast).
    network.to(device=accelerator.device)
    config = PipelineRuntimeConfig.from_config(args)
    world_size = int(accelerator.num_processes)
    rank = dist.get_rank() if _distributed() else 0
    is_last = rank == int(config.stages) - 1
    device = accelerator.device
    scaler = _last_stage_scaler(device, is_last)
    runtime = build_moss_pipeline_runtime(
        unet,
        config=config,
        rank=rank,
        world_size=world_size,
        device=device,
        latent_shape=latent_shape,
        max_text_tokens=max_text_tokens,
        dtype=dtype,
        network=network,
        scaler=scaler,
        optimizer_factory=_stage_optimizer_factory(args),
        max_grad_norm=getattr(args, "max_grad_norm", None),
    )
    _release_foreign_stage_weights(
        unet, runtime.layout, network=network, device=device
    )
    return runtime


def _last_stage_scaler(device, is_last: bool):
    if not is_last:
        return None
    try:
        scaler = torch.amp.GradScaler(device.type)
    except (TypeError, AttributeError):  # pragma: no cover - older torch
        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    if scaler is None:
        raise PipelineParallelError(
            "pp_grad_scaler_missing",
            "the last PP stage needs a GradScaler for plain fp16 training",
        )
    return scaler


def _stage_optimizer_factory(args):
    """Per-stage ``get_optimizer`` over that stage's trainable parameters."""

    from library.training.optimizers import get_optimizer

    return lambda stage_module: get_optimizer(
        args, list(stage_module.trainable_parameters())
    )[2]  # get_optimizer returns the (name, args, optimizer) triple


def _release_foreign_stage_weights(model, layout, *, network, device) -> None:
    """Drop the blocks this stage does not own so PP actually saves weights.

    Stage modules *borrow* their slice from the full model, so ``model.blocks``
    still holds every layer and each rank would carry the whole DiT.  The
    release is opt-in by design (it is the only destructive placement op), so
    the trainer has to ask for it explicitly.

    An attached adapter keeps an ``org_module_ref`` back-pointer to the layer it
    wraps; a foreign adapter's pointer would keep a dropped layer alive.  Those
    back-pointers are cleared for foreign adapters only -- their LoRA tensors
    stay untouched, so checkpoint saving and stage gathering are unaffected.
    """

    from library.training.pp.placement import release_foreign_blocks, stage_adapter_modules

    blocks = getattr(model, "blocks", None)
    container = "blocks"
    if blocks is None:
        blocks = getattr(model, "layers", None)
        container = "layers"
    if blocks is None:
        return
    block_range = getattr(layout, "block_range", None) or getattr(layout, "layer_range", None)
    if block_range is None:
        return
    start, end = block_range
    if (start, end) == (0, len(blocks)):
        return
    if network is not None and hasattr(layout, "block_indices"):
        # Adapter bookkeeping is Anima-shaped (``block_indices``); the MOSS
        # layout indexes layers only, so its adapters keep their references and
        # the block release below stands on its own.
        owned = {id(module) for module in stage_adapter_modules(network, layout)}
        for adapter in getattr(network, "unet_loras", ()) or ():
            if id(adapter) in owned:
                continue
            ref = getattr(adapter, "org_module_ref", None)
            if isinstance(ref, list) and ref:
                ref[0] = None
    release_foreign_blocks(model, layout, container=container, block_range=(start, end))
    gc.collect()
    if getattr(device, "type", "") == "cuda":
        torch.cuda.empty_cache()


def build_anima_trainer_runtime(args, accelerator, network, unet, *, latent_shape, max_text_tokens=None, dtype=None):
    """Build this rank's Anima stage-local runtime (library-level, wired).

    The 5D ``(B, C, 1, H, W)`` latent shape, ``build_anima_schedule`` and the
    shared fp16 transaction all come from the existing library runtime; this
    wrapper only supplies the trainer-side pieces (per-stage optimizer from
    ``get_optimizer``, last-stage GradScaler, fp32 LoRA masters).
    """

    import torch.distributed as dist

    from library.training.pp.contract import PipelineRuntimeConfig
    from library.training.pp.runtime import build_anima_pipeline_runtime

    network.to(device=accelerator.device)
    config = PipelineRuntimeConfig.from_config(args)
    world_size = int(accelerator.num_processes)
    rank = dist.get_rank() if _distributed() else 0
    is_last = rank == int(config.stages) - 1
    scaler = _last_stage_scaler(accelerator.device, is_last)
    runtime = build_anima_pipeline_runtime(
        unet,
        config=config,
        world_size=world_size,
        rank=rank,
        device=accelerator.device,
        latent_shape=latent_shape,
        dtype=dtype,
        network=network,
        scaler=scaler,
        optimizer_factory=_stage_optimizer_factory(args),
        max_grad_norm=getattr(args, "max_grad_norm", None),
    )
    # The base was loaded on CPU for the PP slice.  Release the foreign blocks /
    # submodules FIRST, then move whatever survived: after the release the model
    # only holds this rank's weights, so no name guessing is involved.  A
    # hand-written allow-list missed a module and surfaced as the ATen addmm
    # error "mat1 is on cuda:0, different from other tensors on cpu" (a cpu bias).
    _release_foreign_stage_weights(
        unet, runtime.layout, network=network, device=accelerator.device
    )
    moved = 0
    for tensor in list(unet.parameters()) + list(unet.buffers()):
        if tensor.device.type == "cpu":
            tensor.data = tensor.data.to(accelerator.device)
            moved += 1
    for tensor in list(runtime.stage_module.parameters()) + list(runtime.stage_module.buffers()):
        if tensor.device.type == "cpu":
            tensor.data = tensor.data.to(accelerator.device)
            moved += 1
    # NOTE: the base model is deliberately not a registered submodule of the
    # stage (``object.__setattr__`` keeps stage.to() from walking 25 GB), so any
    # check must walk ``unet`` directly -- asking the stage silently sees nothing.
    leftovers = [
        f"base:{n}" for n, t in unet.named_parameters() if t.device.type == "cpu"
    ] + [
        f"adapter:{n}"
        for n, t in network.named_parameters()
        if t.requires_grad and t.device.type == "cpu"
    ]
    if leftovers:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"rank {rank} still holds CPU tensors after the stage move "
            f"({len(leftovers)} found, first few: {leftovers[:8]}); a cpu weight "
            "here surfaces as an ATen device mismatch inside forward",
        )
    print(f"[pp] rank={rank} stage ready, moved {moved} tensors", flush=True)
    return runtime


def build_krea2_trainer_runtime(
    args, accelerator, network, unet, *, latent_shape, max_text_tokens=None, dtype=None
):
    """Build this rank's Krea-2 stage-local runtime (library-level, wired).

    Differs from the Anima wrapper in two ways that matter for 16 GB cards:

    * ``max_text_tokens`` is required (the Krea-2 wire declares its text length);
    * only **this rank's** adapter modules go to the card.  ``apply_to`` creates
      the adapter for all 28 layers and keeps it off the module tree, so a plain
      ``network.to(device)`` would park the peer's dead weights here (measured
      at ~98 MiB on the 13/15 split).
    """

    import torch.distributed as dist

    from library.training.pp.contract import PipelineRuntimeConfig
    from library.training.pp.krea2_runtime import (
        adapter_modules_for_stage,
        build_krea2_pipeline_runtime,
    )

    if max_text_tokens is None:
        raise PipelineParallelError(
            CODE_CONFIG_INVALID,
            "the Krea-2 PP runtime needs max_text_tokens from the first batch's "
            "cached text encoder output",
        )
    config = PipelineRuntimeConfig.from_config(args)
    world_size = int(accelerator.num_processes)
    rank = dist.get_rank() if _distributed() else 0
    is_last = rank == int(config.stages) - 1
    scaler = _last_stage_scaler(accelerator.device, is_last)
    network.to(device=accelerator.device)
    # PP v0 trains the adapter only.  Krea-2's apply_to keeps the adapter off the
    # module tree, so a generic "train unet network" freeze does not reach the
    # base weights and autograd would still retain ~192 MiB of fc2.weight dW per
    # layer (measured: 15.18 GiB allocated, OOM at 192 MiB on a 16 GB card).
    adapter_ids = {id(p) for p in network.parameters()}
    for parameter in unet.parameters():
        parameter.requires_grad_(id(parameter) in adapter_ids)
    runtime = build_krea2_pipeline_runtime(
        unet,
        config=config,
        world_size=world_size,
        rank=rank,
        device=accelerator.device,
        latent_shape=latent_shape,
        max_text_tokens=int(max_text_tokens),
        dtype=dtype,
        network=network,
        scaler=scaler,
        optimizer_factory=_stage_optimizer_factory(args),
        max_grad_norm=getattr(args, "max_grad_norm", None),
    )
    # Send the peer's adapter modules back to CPU now that we know our range.
    keep = {id(module) for module in adapter_modules_for_stage(network, runtime.layout)}
    for module in getattr(network, "unet_loras", []) or []:
        if id(module) not in keep:
            module.to("cpu")
    _release_foreign_stage_weights(
        unet, runtime.layout, network=network, device=accelerator.device
    )
    return runtime


def _loss_collective_device(runtime):
    """Device for the loss all-reduce: the PP group is 2-D (gloo + nccl)."""

    for name in ("device", "_device"):
        candidate = getattr(runtime, name, None)
        if isinstance(candidate, torch.device):
            return candidate
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _broadcast_step_loss(local, runtime):
    """Sum the per-rank step loss so the progress line is meaningful on every rank.

    Only the trailing stage evaluates the loss, so every other rank would log
    ``loss=None`` and the pipeline's loss curve would be invisible.  All ranks
    run ``pp_train_step``, so this collective is safe.  The (has_loss, loss)
    pair keeps "no rank produced a loss" distinguishable from a real zero.
    """

    if not dist.is_available() or not dist.is_initialized():
        return local
    group = getattr(runtime, "process_group", None) or getattr(runtime, "_group", None)
    payload = torch.zeros(2, dtype=torch.float64, device=_loss_collective_device(runtime))
    if local is not None:
        payload[0] = float(local)
        payload[1] = 1.0
    try:
        dist.all_reduce(payload, op=dist.ReduceOp.SUM, group=group)
    except Exception:  # pragma: no cover - backend without the collective
        logger.debug("loss all-reduce unavailable; keeping the local value", exc_info=True)
        return local
    return float(payload[0].item()) if float(payload[1].item()) > 0 else None


def pp_train_step(runtime, inputs, counters: PpLoopCounters, *, scheduler=None) -> None:
    """One whole pipeline step with the frozen §4 semantics (family-blind)."""

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    runtime.zero_grad()
    result = runtime.step(
        inputs.wire_for(runtime),
        target=inputs.target if runtime.is_last else None,
        losses=[],
        **inputs.step_kwargs(runtime),
    )
    # MOSS runtime.step returns (prediction, losses); the Anima library runtime
    # returns a PpStepResult. Normalize without changing either contract.
    if hasattr(result, "losses") and hasattr(result, "topology"):
        prediction, losses = result.prediction, list(result.losses)
    else:
        prediction, losses = result
    outcome = runtime.optimizer_step()
    counters.global_step += 1
    counters.skipped += 1 if outcome.skipped else 0
    if outcome.skipped:
        counters.last_loss = None
    else:
        counters.effective_step += 1
        if scheduler is not None:
            scheduler.step()
        if losses:
            # Display口径 = the whole-batch mean, not the last microbatch: each
            # logged loss already carries the schedule's 1/n_microbatches
            # factor, so summing the (scale-restored) values reproduces the
            # single-device whole-batch MSE exactly (equal-size microbatches).
            # Verified on the equivalence fixture: sum(scaled) == full-batch
            # MSE bit-equal, while losses[-1] alone reads ~0.5x low at m=2.
            values = [
                float(value if not torch.is_tensor(value)
                      else value.detach().float()) / max(outcome.scale, 1.0)
                for value in losses
            ]
            counters.last_loss = float(sum(values))
    # Every rank must reach this collective, so it sits outside the branches.
    # Without it the trailing stage's loss never appears in the progress line.
    counters.last_loss = _broadcast_step_loss(counters.last_loss, runtime)
    # Per-step wall time: the only number that says whether 1F1B + block swap is
    # worth its PCIe traffic.  Runs after every collective so it measures the
    # whole step, and is printed every step (a mean hides the first-step cost).
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(
        f"[pp] step {counters.global_step} wall={elapsed:.2f}s "
        f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB"
        if torch.cuda.is_available()
        else f"[pp] step {counters.global_step} wall={elapsed:.2f}s",
        flush=True,
    )
    del prediction


def _bundle_root(args, run_id: str, step: int):
    return os.path.join(str(args.output_dir), "pp_runs", f"{run_id}-step{int(step)}")


def save_pp_bundle(args, runtime, *, counters: PpLoopCounters, scheduler, run_id: str, accelerator, family: str = "moss_image") -> str:
    """Save the negotiated PP bundle + trainer-level state; never overwrite."""

    from library.training.pp.checkpoint import save_distributed_checkpoint
    from library.training.pp.trainer_resume import (
        capture_trainer_payload,
        save_trainer_resume,
    )

    root = _bundle_root(args, run_id, counters.global_step)
    marker = Path(root) / "pp_checkpoint.json"
    if marker.is_file():
        # Same-step periodic + end-of-run double save: the bundle at this step
        # already exists and was created by this run; skipping is idempotent,
        # while a *new* root still refuses to overwrite anything.
        logger.info("PP bundle for step %s already exists at %s; skipping", counters.global_step, root)
        return str(root)
    result = save_distributed_checkpoint(
        root, runtime,
        step=int(counters.global_step),
        identity=run_identity(run_id, args),
    )
    payload = capture_trainer_payload(
        rank=runtime.layout.stage_index,
        run_id=run_id,
        family=family,
        global_step=counters.global_step,
        effective_step=counters.effective_step,
        epoch=counters.epoch,
        batch_index=counters.batch_index,
        lr_scheduler=scheduler,
        seed=getattr(args, "seed", None),
    )
    save_trainer_resume(root, rank=runtime.layout.stage_index, payload=payload)
    if accelerator.is_main_process:
        pointer = os.path.join(str(args.output_dir), "pp_runs", "LATEST.json")
        with open(pointer, "w", encoding="utf-8") as handle:
            json.dump({"run_id": run_id, "root": str(result.root), "step": int(result.step)}, handle)
        logger.info("PP bundle saved: %s (step %s)", result.root, result.step)
    return str(result.root)


def resume_pp_bundle(args, runtime, *, scheduler, run_id: str, group=None, family: str = "moss_image"):
    """Restore stage + trainer state from ``args.resume`` (exact or opt-in)."""

    from library.training.pp.checkpoint import load_distributed_checkpoint
    from library.training.pp.trainer_resume import (
        load_trainer_resume,
        require_exact_resume_or_opt_in,
    )

    root = str(args.resume)
    require_exact_resume_or_opt_in(root, args, family=family)
    identity = run_identity(run_id, args)
    state = load_distributed_checkpoint(root, runtime, identity=identity, group=group)
    payload = load_trainer_resume(
        root,
        rank=runtime.layout.stage_index,
        run_id=run_id,
        family=family,
        expected_step=int(state.step),
        lr_scheduler=scheduler,
        group=group,
    )
    return state, payload


def _gather_final_adapter_state(runtime) -> dict[str, dict]:
    """Gather every stage's adapter state onto this rank (CPU tensors)."""

    import torch.distributed as dist

    from library.training.pp.checkpoint_state import stage_adapter_state

    local = stage_adapter_state(runtime.stage_module)
    if not _distributed():
        return local
    gathered: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    merged: dict[str, dict] = {}
    for entry in gathered:
        for name, state in (entry or {}).items():
            if name in merged:
                raise PipelineParallelError(
                    "pp_checkpoint_state_mismatch",
                    f"adapter {name!r} is claimed by two stages; the full LoRA "
                    "output cannot be assembled unambiguously",
                )
            merged[name] = state
    return merged


def save_final_lora(trainer, args, runtime, handoff, counters, metadata, minimum_metadata) -> str:
    """Write the ordinary-format LoRA output (rank 0) from gathered states."""

    from library.training.checkpoints import CheckpointSaver, get_last_ckpt_name

    saver = CheckpointSaver(
        args=args,
        accelerator=handoff.accelerator,
        save_dtype=handoff.save_dtype,
        metadata=metadata,
        minimum_metadata=minimum_metadata,
        get_sai_model_spec_fn=trainer.get_sai_model_spec,
        current_epoch=counters.epoch,
        current_step=counters.effective_step,
        progress_sink=None,
    )
    network = handoff.network
    unwrapped = handoff.accelerator.unwrap_model(network)
    module_by_name = {lora.lora_name: lora for lora in unwrapped.unet_loras}
    for name, state in _gather_final_adapter_state(runtime).items():
        module = module_by_name.get(name)
        if module is None:
            raise PipelineParallelError(
                "pp_checkpoint_state_mismatch",
                f"gathered adapter {name!r} has no module on rank 0's network",
            )
        module.load_state_dict(
            {key: value.float() for key, value in state.items()}, strict=True
        )
    saver.save_final(network, counters.effective_step, counters.epoch + 1)
    return os.path.join(args.output_dir, get_last_ckpt_name(args, "." + args.save_model_as))


__all__ = [
    "PpLoopCounters",
    "PpSessionHandoff",
    "build_anima_trainer_runtime",
    "build_moss_trainer_runtime",
    "build_pp_train_dataloader",
    "derive_run_id",
    "pp_train_step",
    "resume_pp_bundle",
    "run_identity",
    "run_pp_trainer_session",
    "save_final_lora",
    "save_pp_bundle",
    "validate_pp_process_group",
]


def _resolve_max_steps(args, updates_per_epoch: int) -> int:
    if getattr(args, "max_train_steps", None):
        return int(args.max_train_steps)
    epochs = int(getattr(args, "max_train_epochs", None) or 1)
    args.max_train_steps = epochs * updates_per_epoch
    return int(args.max_train_steps)


def _build_metadata(trainer, args, handoff, counters, loader):
    from library.training.metadata import (
        add_dataset_metadata,
        add_model_hash_metadata,
        build_training_metadata,
        finalize_metadata,
    )

    metadata = build_training_metadata(
        args,
        session_id=handoff.session_id,
        training_started_at=handoff.training_started_at,
        text_encoder_lr=getattr(args, "text_encoder_lr", None),
        optimizer_name=str(getattr(args, "optimizer_type", "AdamW8bit")),
        optimizer_args={},
        model_version=handoff.model_version,
        num_train_images=handoff.train_dataset_group.num_train_images,
        num_val_images=(
            handoff.val_dataset_group.num_train_images
            if handoff.val_dataset_group is not None
            else 0
        ),
        num_reg_images=handoff.train_dataset_group.num_reg_images,
        num_batches_per_epoch=len(loader),
        num_train_epochs=math.ceil(
            int(args.max_train_steps) / max(len(loader), 1)
        ),
    )
    trainer.update_metadata(metadata, args)
    add_dataset_metadata(
        metadata,
        handoff.train_dataset_group,
        args,
        use_user_config=handoff.use_user_config,
        use_dreambooth_method=handoff.use_dreambooth_method,
        # PP: one pipeline step consumes one whole batch; the world is the
        # stage count, not a data-parallel replication factor.
        total_batch_size=int(getattr(args, "train_batch_size", 1)),
    )
    add_model_hash_metadata(metadata, args)
    return finalize_metadata(metadata, net_kwargs=handoff.net_kwargs or None)


def _report_progress(accelerator, runtime, counters, scheduler, total_steps, *, log_every=1):
    if counters.global_step % log_every:
        return
    loss = counters.last_loss
    accelerator.print(
        f"pp step {counters.global_step}/{total_steps} "
        f"(effective {counters.effective_step}, skipped {counters.skipped}) "
        f"loss={loss if loss is None else f'{loss:.6f}'} "
        f"lr={scheduler.get_last_lr()[0] if scheduler is not None else 'n/a'} "
        f"scale={runtime.scale}"
    )


def run_pp_trainer_session(trainer, args, handoff: PpSessionHandoff) -> None:
    """The dedicated PP training session (dispatched from train_session)."""

    from library.training.optimizers import get_optimizer_train_eval_fn
    from library.training.pp.trainer_adapters import resolve_trainer_adapter
    from library.training.pp.trainer_inputs import build_family_step_inputs
    from library.training.schedulers import get_scheduler_fix

    accelerator = handoff.accelerator
    adapter = resolve_trainer_adapter(args)
    adapter.admission(
        args, world_size=int(accelerator.num_processes),
        where="train_session.pp_dispatch",
    )
    validate_pp_process_group(accelerator)
    if getattr(args, "gradient_checkpointing", False) and hasattr(handoff.unet, "enable_gradient_checkpointing"):
        # Real per-block activation checkpointing on the family DiT (the MOSS
        # contract requires it; Anima opts in the same way).
        handoff.unet.enable_gradient_checkpointing()

    loader = build_pp_train_dataloader(args, handoff.train_dataset_group, handoff.collator)
    first_batch = next(iter(loader))
    latent_shape, _raw_text_tokens = adapter.wire_shape_from_batch(first_batch)
    # The cached text-encoder outputs on disk carry the *real* token length (12
    # for a short caption); the padding to the bucket length happens in
    # prepare_cached_conditions.  Declaring the unpadded length made the wire
    # expect token_cap = 12 + 4200 -> 4352 while the forward actually ran on
    # 512 + 4200 = 4864, which pipelining rejected as a shape mismatch.
    _prepared = adapter.prepare_conditions(
        trainer,
        args,
        first_batch,
        accelerator=accelerator,
        weight_dtype=handoff.weight_dtype,
    )
    # Krea-2 names the bundle {"context", "mask"}; Anima/MOSS use "text_hiddens".
    _hiddens = _prepared.get("context")
    if _hiddens is None:
        _hiddens = _prepared["text_hiddens"]
    max_text_tokens = int(_hiddens.shape[1])
    # The LR plan is built from max_train_steps, so the steps budget is
    # resolved (epochs -> steps when needed) *before* the scheduler exists.
    total_steps = _resolve_max_steps(args, max(len(loader), 1))
    run_id = derive_run_id(args, family=adapter.name)
    runtime = adapter.build_runtime(
        args, accelerator, handoff.network, handoff.unet,
        latent_shape=latent_shape, max_text_tokens=max_text_tokens,
        dtype=handoff.weight_dtype,
    )
    scheduler = get_scheduler_fix(args, runtime.optimizer, 1)
    optimizer_train_fn, optimizer_eval_fn = get_optimizer_train_eval_fn(
        runtime.optimizer, args
    )

    counters = PpLoopCounters()
    resume_step = 0
    resume_batch_index = 0
    resume_epoch = 0
    if getattr(args, "resume", None):
        state, payload = resume_pp_bundle(
            args, runtime, scheduler=scheduler, run_id=run_id, family=adapter.name
        )
        resume_step = int(state.step)
        resume_epoch = int(payload.get("epoch", 0))
        resume_batch_index = int(payload.get("batch_index", 0))
        counters.global_step = resume_step
        counters.effective_step = int(payload.get("effective_step", resume_step))
        counters.epoch = resume_epoch
        logger.info(
            "PP exact resume from %s at global_step=%s epoch=%s batch_index=%s",
            args.resume, resume_step, resume_epoch, resume_batch_index,
        )
    metadata, minimum_metadata = _build_metadata(trainer, args, handoff, counters, loader)

    save_every = int(getattr(args, "save_every_n_steps", 0) or 0)
    save_at_end = bool(getattr(args, "save_state", False) or getattr(args, "save_state_on_train_end", False))
    optimizer_train_fn()
    total_steps = int(args.max_train_steps)
    start_epoch = counters.epoch
    for epoch in range(start_epoch, math.ceil(total_steps / max(len(loader), 1))):
        loader.generator.manual_seed(int(getattr(args, "seed", 0) or 0) + epoch)
        counters.epoch = epoch
        counters.batch_index = 0
        for batch in iter(loader):
            if counters.global_step >= total_steps:
                break
            if counters.batch_index < resume_batch_index:
                counters.batch_index += 1
                continue
            inputs = build_family_step_inputs(
                adapter,
                trainer, args, batch,
                accelerator=accelerator,
                vae=handoff.vae,
                vae_dtype=handoff.vae_dtype,
                weight_dtype=handoff.weight_dtype,
                text_encoding_strategy=handoff.text_encoding_strategy,
                is_first=runtime.is_first,
            )
            pp_train_step(runtime, inputs, counters, scheduler=scheduler)
            counters.batch_index += 1
            resume_batch_index = 0
            _report_progress(accelerator, runtime, counters, scheduler, total_steps)
            if save_every and counters.global_step % save_every == 0:
                save_pp_bundle(args, runtime, counters=counters, scheduler=scheduler, run_id=run_id, accelerator=accelerator, family=adapter.name)
        else:
            continue
        break
    if save_at_end or save_every == 0:
        save_pp_bundle(args, runtime, counters=counters, scheduler=scheduler, run_id=run_id, accelerator=accelerator, family=adapter.name)
    save_final_lora(trainer, args, runtime, handoff, counters, metadata, minimum_metadata)
    optimizer_eval_fn()
    accelerator.end_training()
