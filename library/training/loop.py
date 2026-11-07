"""Training-loop orchestration.

Owns the per-epoch / per-step body that used to live inline in
``AnimaTrainer.train()``. The entrypoint is :func:`run_training_loop`, which
takes a built :class:`LoopState` plus the trainer instance so override hooks
(``process_batch``, ``on_step_start``, ``sample_images``,
``generate_step_logs``, ``step_logging``, ``epoch_logging``) keep working
unchanged. The validation pass lives in :mod:`library.training.validation`.

State that used to be on ``self`` for cross-call signaling —
``_last_router_H_postfix``, ``_cudagraph_mark_step``, ``_hydra_warmup_step``,
``_adapters`` — stays on the trainer; this module reads them through the
``trainer`` handle.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import sys
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm

from library.training import multigpu, window_accounting
from library.training.finite_checks import (
    check_loss_finite,
    debug_finite_enabled,
)
from library.training.scaler_ownership import (
    GradFiniteness,
    inspect_grad_ownership,
    scaler_present,
    log_scaler_owned_overflow,
    log_skipped_step,
)
from library.training.gradient_sync import synchronize_optimizer_gradients
from library.training.method_adapter import MethodAdapter

from library.training.stage_schedule import (
    active_subset_indices_for_step,
    apply_active_subsets_to_dataset,
    log_stage_switch,
    parse_stage_specs,
    progress_from_steps,
    resolve_stage_index,
    stage_schedule_enabled,
)

from library import train_util
from library.datasets import LossRecorder
from library.runtime.device import clean_memory_on_device
from library.training.checkpoints import CheckpointSaver
from library.training.contexts import TrainCtx, ValCtx
from library.training.method_adapter import StepCtx
from library.training.metrics import MetricContext, collect_metrics
from library.training.validation import run_validation
from library.training.adaptive_personalization import metrics as personalization_metrics
from library.training.adaptive_runtime.training_runtime import notify as adaptive_notify
from library.training.adaptive_runtime.resume_runtime import startup_log
from library.training.sample_preview_common import should_sample
from library.training.precision_report import observe as precision_observe

logger = logging.getLogger(__name__)


def _memory_probe_for_step(trainer, state) -> Any | None:
    probe = getattr(trainer, "memory_probe", None)
    if probe is None:
        return None
    if not probe.should_record_step(state.global_step):
        return None
    return probe


def _async_gradient_sync_for_step(trainer, state):
    """Return the hook-based synchronizer when no adapter adds post-backward grads."""
    synchronizer = getattr(
        state.accelerator.unwrap_model(state.network),
        "_anima_async_gradient_sync",
        None,
    )
    if synchronizer is None or not state.accelerator.sync_gradients:
        return None
    if any(
        type(adapter).after_backward is not MethodAdapter.after_backward
        for adapter in getattr(trainer, "_adapters", ())
    ):
        return None
    return synchronizer


def _peak_probe_for_step(trainer, state) -> Any | None:
    probe = getattr(trainer, "peak_probe", None)
    if probe is None:
        return None
    if not probe.should_record_step(state.global_step):
        return None
    return probe


def _gradient_flow_probe_for_step(trainer, state) -> Any | None:
    probe = getattr(trainer, "gradient_flow_probe", None)
    if probe is None or not state.accelerator.sync_gradients:
        return None
    next_step = state.global_step + 1
    if not probe.should_record_step(
        next_step, max_train_steps=state.args.max_train_steps
    ):
        return None
    return probe


def _probe_exception_fields(exc: BaseException) -> dict[str, Any]:
    message = str(exc)
    return {
        "exception_type": type(exc).__name__,
        "exception_message": message[:1000],
        "is_cuda_oom": "out of memory" in message.lower(),
    }


def _probe_step(probe, state, label: str, **fields: Any) -> None:
    if probe is None:
        return
    probe.snapshot(
        label,
        device=state.accelerator.device,
        step=state.global_step,
        phase="train",
        sync_gradients=state.accelerator.sync_gradients,
        **fields,
    )


def _sample_probe_snapshot(
    trainer,
    accelerator,
    args,
    epoch,
    steps: int,
    label: str,
    **fields: Any,
) -> bool:
    """Record a sampling boundary without changing the sampling path."""
    probe = getattr(trainer, "memory_probe", None)
    if probe is None:
        return False
    try:
        scheduled = should_sample(args, epoch, steps)
    except AttributeError:
        scheduled = False
    if not scheduled:
        return False
    if not probe.should_record_step(steps):
        return False
    probe.snapshot(
        label,
        device=accelerator.device,
        step=steps,
        phase="sample",
        sample_epoch=epoch,
        **fields,
    )
    return True


def _run_sample_with_probe(trainer, accelerator, args, epoch, steps: int, callback) -> None:
    probed = _sample_probe_snapshot(
        trainer, accelerator, args, epoch, steps, "sample_before"
    )
    try:
        callback()
    except Exception as exc:
        if probed:
            _sample_probe_snapshot(
                trainer,
                accelerator,
                args,
                epoch,
                steps,
                "sample_exception",
                exception_type=type(exc).__name__,
                exception_message=str(exc)[:1000],
                is_cuda_oom="out of memory" in str(exc).lower(),
            )
        raise
    else:
        if probed:
            _sample_probe_snapshot(
                trainer, accelerator, args, epoch, steps, "sample_after"
            )


@dataclass
class LoopState:
    """Bundles every local that used to live in ``train()``'s for-epoch scope.

    Most fields are constants for the run; ``global_step``, ``profile_started``,
    ``profile_range``, ``initial_step``, and ``text_encoder(s)`` are mutated
    during the loop. ``current_epoch`` / ``current_step`` are mp.Value handles
    shared with :class:`CheckpointSaver` for state persistence.
    """

    args: Any
    accelerator: Accelerator
    train_ctx: TrainCtx
    val_ctx: ValCtx
    saver: CheckpointSaver

    network: Any
    unet: Any
    text_encoder: Any
    text_encoders: list
    vae: Any
    tokenizers: Any
    training_model: Any
    train_dataloader: Any
    optimizer: Any
    lr_scheduler: Any
    lr_descriptions: Optional[list]
    optimizer_train_fn: Callable
    optimizer_eval_fn: Callable
    weight_dtype: Any
    unet_weight_dtype: Any

    current_epoch: Any  # mp.Value
    current_step: Any  # mp.Value
    num_train_epochs: int
    epoch_to_start: int
    initial_step: int

    metadata: dict
    is_tracking: bool
    progress_bar: Any
    loss_recorder: LossRecorder
    val_step_loss_recorder: LossRecorder
    val_epoch_loss_recorder: LossRecorder

    validation_steps: int

    profile_range: Optional[tuple]
    on_step_start_for_network: Callable

    # Optional curriculum schedule: underlying dataset + loader factory kwargs.
    train_dataset_group: Any = None
    dataloader_kwargs: Optional[dict] = None
    stage_index: int = -1

    global_step: int = 0
    profile_started: bool = False
    #: Unified per-step answer to "did the weights move?" — set every optimizer
    #: boundary for both a GradScaler skip and a guard skip, and read by the
    #: adaptive runtime / AUTO probe / online swap controller.
    update_skipped: Optional[bool] = None
    #: Steps whose optimizer update was dropped because no GradScaler existed to
    #: absorb a non-finite gradient (see ``library/training/scaler_ownership.py``).
    #: Known limitation: only warned and counted for now — it is not published into
    #: ``precision_report.json``, so it is not yet a first-class metric.
    nonfinite_grad_skips: int = 0
    #: E3 accumulation-window accounting. ``window_attempts`` / ``window_replays``
    #: / ``window_discards`` are cumulative; the three ledger fields describe the
    #: last *committed* window and are left untouched by a replay/discard
    #: (see ``library/training/window_accounting.py``).
    window_attempts: int = 0
    window_replays: int = 0
    window_discards: int = 0
    #: Consecutive discards, reset by a committed window — drives the
    #: ``--adaptive_max_nonfinite_skips`` escalation (see window_accounting).
    window_consecutive_discards: int = 0
    window_total_loss: float = 0.0
    window_num_steps: int = 0
    window_last_loss: Optional[float] = None
    #: This repo has no model EMA; the slot is written on the commit path only so
    #: the E3 spec can prove a discarded window touches no ledger at all.
    ema_observation: Optional[dict] = None
    recent_step_rate_last: Optional[tuple[float, int]] = None
    recent_step_seconds: deque[float] = field(default_factory=lambda: deque(maxlen=9))
    #: Batches **this rank** has consumed from its dataloader, cumulative across
    #: epochs and including the prefix a resumed run skipped. It is the real
    #: producer of the checkpoint's ``sampler_position`` (contract §8 P1-2): a
    #: resume is exact only when this counter continues where it stopped.
    batches_consumed: int = 0
    batches_per_epoch: int = 0
    #: Batches the resume plan skipped before the first fresh update; the counter
    #: starts there so positions from two runs are comparable.
    resume_skipped_batches: int = 0


def loader_cursor_payload(state: "LoopState") -> Optional[dict]:
    """The committed per-rank cursor, or ``None`` when it cannot be trusted yet."""
    from library.training.multigpu import cursor as cursor_mod

    if not state.batches_per_epoch:
        # No dataloader length known: reporting a position would be a guess.
        return None
    return cursor_mod.make_cursor(
        position=int(state.batches_consumed),
        epoch=int(state.current_epoch.value),
        rank=int(getattr(state.accelerator, "process_index", 0) or 0),
        batches_per_epoch=int(state.batches_per_epoch),
    )


def _median_positive_finite(values: deque[float]) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value) and value > 0)
    if not finite:
        return None
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return (finite[mid - 1] + finite[mid]) / 2


def _record_recent_step_seconds(
    state: LoopState, step: int, now: float | None = None, *, record: bool = True
) -> None:
    """Track seconds-per-step over the last few boundaries.

    ``record=False`` advances the timestamp baseline without appending a sample.
    A skipped update moves no weights, so its wall time is not a sample of the
    workload the block-swap policy is trying to measure; skipping the sample keeps
    ``_recent_step_seconds`` (and therefore ``block_swap_auto`` promotion) free of
    overflow-driven spikes.
    """
    current_step = int(step)
    current_ts = time.monotonic() if now is None else float(now)
    if current_step <= 0 or not math.isfinite(current_ts):
        return

    if not record:
        state.recent_step_rate_last = (current_ts, current_step)
        return

    last = state.recent_step_rate_last
    if last is None:
        state.recent_step_rate_last = (current_ts, current_step)
        return

    last_ts, last_step = last
    if current_step == last_step:
        return
    if current_step < last_step or current_ts <= last_ts:
        state.recent_step_seconds.clear()
        state.recent_step_rate_last = (current_ts, current_step)
        return

    step_delta = current_step - last_step
    seconds_per_step = (current_ts - last_ts) / step_delta
    if math.isfinite(seconds_per_step) and seconds_per_step > 0:
        state.recent_step_seconds.append(seconds_per_step)
    state.recent_step_rate_last = (current_ts, current_step)


def _recent_step_seconds(state: LoopState) -> float | None:
    return _median_positive_finite(state.recent_step_seconds)


def build_loop_state(
    trainer,
    *,
    args,
    accelerator: Accelerator,
    saver: CheckpointSaver,
    network,
    unet,
    text_encoder,
    text_encoders,
    vae,
    tokenizers,
    training_model,
    train_dataloader,
    val_dataloader,
    val_dataset_group,
    optimizer,
    lr_scheduler,
    lr_descriptions,
    optimizer_train_fn,
    optimizer_eval_fn,
    weight_dtype,
    unet_weight_dtype,
    vae_dtype,
    text_encoding_strategy,
    tokenize_strategy,
    train_text_encoder,
    train_unet,
    current_epoch,
    current_step,
    num_train_epochs,
    epoch_to_start,
    initial_step,
    resume_global_step,
    metadata,
    batches_per_epoch: Optional[int] = None,
    resume_skipped_batches: int = 0,
) -> LoopState:
    """Build :class:`LoopState`. Mirrors the pre-loop setup that used to sit
    between ``_prepare_with_accelerator()`` and the for-epoch loop in
    ``train()``: noise scheduler, trackers, loss recorders, optional text
    encoder eviction, ``--sample_at_first``, train/val ctx construction,
    progress bar, profiler parsing.
    """
    noise_scheduler = trainer.get_noise_scheduler(args, accelerator.device)

    train_util.init_trackers(accelerator, args, "network_train")

    loss_recorder = LossRecorder()
    val_step_loss_recorder = LossRecorder()
    val_epoch_loss_recorder = LossRecorder()

    if hasattr(accelerator.unwrap_model(network), "on_step_start"):
        on_step_start_for_network = accelerator.unwrap_model(network).on_step_start
    else:

        def on_step_start_for_network(*args, **kwargs):
            return None

    if trainer.is_text_encoder_not_needed_for_training(args):
        startup_log(args, logger, "text_encoder is not needed for training. deleting to save memory.")
        for t_enc in text_encoders:
            del t_enc
        text_encoders = []
        text_encoder = None
        gc.collect()
        clean_memory_on_device(accelerator.device)

    # --sample_at_first
    optimizer_eval_fn()
    _run_sample_with_probe(
        trainer,
        accelerator,
        args,
        0,
        0,
        lambda: trainer.sample_images(
            accelerator,
            args,
            0,
            0,
            accelerator.device,
            vae,
            tokenizers,
            text_encoder,
            unet,
            network=network,
        ),
    )
    optimizer_train_fn()
    is_tracking = len(accelerator.trackers) > 0
    if is_tracking:
        accelerator.log({}, step=0)

    train_ctx = TrainCtx(
        args=args,
        accelerator=accelerator,
        network=network,
        unet=unet,
        vae=vae,
        text_encoders=text_encoders,
        noise_scheduler=noise_scheduler,
        text_encoding_strategy=text_encoding_strategy,
        tokenize_strategy=tokenize_strategy,
        vae_dtype=vae_dtype,
        weight_dtype=weight_dtype,
        train_text_encoder=train_text_encoder,
        train_unet=train_unet,
        optimizer_eval_fn=optimizer_eval_fn,
        optimizer_train_fn=optimizer_train_fn,
        is_tracking=is_tracking,
    )

    # The checkpoint plan keeps global_step in optimizer steps and initial_step
    # as the residual batch count within the resumed epoch.
    global_step = max(0, int(resume_global_step or 0))

    startup_log(args, logger, f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
    _ts_parts = [f"timestep_sampling={args.timestep_sampling}"]
    if args.timestep_sampling in ("sigmoid", "shift", "flux_shift"):
        _ts_parts.append(f"sigmoid_scale={args.sigmoid_scale}")
        _ts_parts.append(f"sigmoid_bias={getattr(args, 'sigmoid_bias', 0.0)}")
    if args.timestep_sampling in ("shift", "flux_shift"):
        _ts_parts.append(f"discrete_flow_shift={args.discrete_flow_shift}")
    if (
        getattr(args, "t_min", None) is not None
        or getattr(args, "t_max", None) is not None
    ):
        _ts_parts.append(
            f"σ∈[{getattr(args, 't_min', None)}, {getattr(args, 't_max', None)}]"
        )
    startup_log(args, logger, "sigma sampling: " + ", ".join(_ts_parts))
    for i, t_enc in enumerate(text_encoders):
        params_itr = t_enc.parameters()
        params_itr.__next__()  # skip the first parameter
        params_itr.__next__()  # skip the second parameter (CLIP first two are embeddings)
        param_3rd = params_itr.__next__()
        logger.info(
            f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}"
        )

    clean_memory_on_device(accelerator.device)

    progress_bar = tqdm(
        range(args.max_train_steps - global_step),
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )

    validation_steps = (
        min(args.max_validation_steps, len(val_dataloader))
        if args.max_validation_steps is not None
        else len(val_dataloader)
    )
    # Validate at fixed sigma values across the schedule:
    # 0.1 = near-clean / fine detail, 0.4 = mid / bulk structure,
    # 0.7 = high noise / coarse denoising (early inference steps).
    validation_sigmas = (
        args.validation_sigmas
        if args.validation_sigmas is not None
        else [0.1, 0.4, 0.7]
    )
    val_ctx = ValCtx(
        dataloader=val_dataloader,
        sigmas=validation_sigmas,
        steps=validation_steps,
        total_steps=validation_steps * len(validation_sigmas),
        train_loss_recorder=loss_recorder,
        original_t_min=args.t_min,
        original_t_max=args.t_max,
        dataset_group=val_dataset_group,
    )

    # nsys workflow: --profile_steps START-END toggles the cuda profiler API
    # around the requested step window. Wrap the launch with
    #   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...
    # so nsys only records that window.
    profile_range = trainer._parse_profile_steps(args)

    return LoopState(
        args=args,
        accelerator=accelerator,
        train_ctx=train_ctx,
        val_ctx=val_ctx,
        saver=saver,
        network=network,
        unet=unet,
        text_encoder=text_encoder,
        text_encoders=text_encoders,
        vae=vae,
        tokenizers=tokenizers,
        training_model=training_model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_descriptions=lr_descriptions,
        optimizer_train_fn=optimizer_train_fn,
        optimizer_eval_fn=optimizer_eval_fn,
        weight_dtype=weight_dtype,
        unet_weight_dtype=unet_weight_dtype,
        current_epoch=current_epoch,
        current_step=current_step,
        num_train_epochs=num_train_epochs,
        epoch_to_start=epoch_to_start,
        initial_step=initial_step,
        metadata=metadata,
        is_tracking=is_tracking,
        progress_bar=progress_bar,
        loss_recorder=loss_recorder,
        val_step_loss_recorder=val_step_loss_recorder,
        val_epoch_loss_recorder=val_epoch_loss_recorder,
        validation_steps=validation_steps,
        profile_range=profile_range,
        on_step_start_for_network=on_step_start_for_network,
        global_step=global_step,
        batches_per_epoch=(
            int(batches_per_epoch)
            if batches_per_epoch is not None
            else len(train_dataloader)
        ),
        resume_skipped_batches=max(0, int(resume_skipped_batches)),
    )


def run_training_loop(trainer, state: LoopState) -> None:
    """Run the full for-epoch training loop and the post-loop end-of-training
    metadata write. Mutates ``state.global_step``, profiler bookkeeping, and
    the metadata dict; the per-checkpoint saves go through ``state.saver``.
    """
    args = state.args
    accelerator = state.accelerator
    _maybe_apply_stage_schedule(state, force=True)

    for epoch in range(state.epoch_to_start, state.num_train_epochs):
        accelerator.print(f"\nepoch {epoch + 1}/{state.num_train_epochs}\n")
        state.current_epoch.value = epoch + 1
        state.metadata["ss_epoch"] = str(epoch + 1)

        # network.train() is invoked here
        accelerator.unwrap_model(state.network).on_epoch_start(
            state.text_encoder, state.unet
        )

        _run_epoch_steps(trainer, state, epoch)
        reached_step_limit = state.global_step >= args.max_train_steps
        _run_epoch_validation(trainer, state, epoch)
        _log_epoch_average(trainer, state, epoch)
        _run_adapter_epoch_hooks(trainer, state)

        accelerator.wait_for_everyone()

        state.optimizer_eval_fn()
        state.saver.maybe_save_epoch(
            state.network,
            state.global_step,
            epoch,
            state.num_train_epochs,
            loader_cursor=loader_cursor_payload(state),
        )
        state.saver.maybe_save_resumable(
            state.network,
            state.global_step,
            epoch,
            state.num_train_epochs,
            loader_cursor=loader_cursor_payload(state),
        )

        _run_sample_with_probe(
            trainer,
            accelerator,
            args,
            epoch + 1,
            state.global_step,
            lambda: trainer.sample_images(
                accelerator,
                args,
                epoch + 1,
                state.global_step,
                accelerator.device,
                state.vae,
                state.tokenizers,
                state.text_encoder,
                state.unet,
                network=state.network,
            ),
        )
        state.optimizer_train_fn()

        if reached_step_limit:
            break

    state.metadata["ss_training_finished_at"] = str(time.time())


def _current_stage_fields(state: LoopState) -> dict[str, object]:
    """Return optional stage_index/stage_name for progress events."""
    if int(getattr(state, "stage_index", -1)) < 0:
        return {}
    fields: dict[str, object] = {"stage_index": int(state.stage_index)}
    stages = parse_stage_specs(getattr(state.args, "stage_schedule", None))
    if 0 <= state.stage_index < len(stages):
        name = str(stages[state.stage_index].name or "").strip()
        if name:
            fields["stage_name"] = name
    return fields


def _maybe_apply_stage_schedule(state: LoopState, *, force: bool = False) -> None:
    """Switch active subset filter when progress crosses a stage boundary."""
    args = state.args
    if not stage_schedule_enabled(args):
        return
    if state.train_dataset_group is None or not state.dataloader_kwargs:
        return

    stages = parse_stage_specs(getattr(args, "stage_schedule", None))
    if not stages:
        return
    progress = progress_from_steps(state.global_step, int(args.max_train_steps or 1))
    next_index = resolve_stage_index(stages, progress)
    if not force and next_index == state.stage_index:
        return

    active = active_subset_indices_for_step(args, state.global_step)
    applied = apply_active_subsets_to_dataset(state.train_dataset_group, active)
    if not applied:
        # Do not advance stage_index: keep previous membership and fail loudly.
        raise RuntimeError(
            "stage schedule switch failed: filtered dataset is empty "
            f"(target_stage={next_index}, subset_indices={sorted(active or [])}, "
            f"step={state.global_step}/{args.max_train_steps}). "
            "Ensure snapshot_full_image_data ran before the first stage filter."
        )

    # Rebuild DataLoader so workers and __len__ see the new indices, then pass
    # it through Accelerate again. A vanilla replacement loader leaves cached
    # batch tensors on CPU while the model/loss live on the accelerator device.
    kwargs = dict(state.dataloader_kwargs)
    # Avoid inheriting a dead iterator / stale length under persistent workers.
    kwargs["persistent_workers"] = False
    raw_dataloader = torch.utils.data.DataLoader(
        state.train_dataset_group,
        shuffle=True,
        **kwargs,
    )
    state.train_dataloader = state.accelerator.prepare_data_loader(raw_dataloader)
    state.stage_index = next_index
    stage = stages[next_index]
    log_stage_switch(
        stage,
        next_index,
        state.global_step,
        int(args.max_train_steps or 1),
    )
    state.accelerator.print(
        f"[stage] #{next_index + 1} {stage.name or ''} "
        f"subset={stage.subset_index} "
        f"@ step {state.global_step}/{args.max_train_steps}"
    )


@contextmanager
def _accumulate_scope(accelerator, state: LoopState, *, manage: bool):
    """Enter ``accelerator.accumulate`` only when the caller lets accelerate manage sync.

    The replay path passes ``manage=False``: accelerate's ``_do_sync`` (1.13.0
    ``accelerator.py:1228-1235``) resets ``step``/forces ``sync_gradients`` whenever
    ``sync_with_dataloader and end_of_dataloader`` is true, which would override the
    boundary :func:`_forced_accumulate` just set and turn every replayed micro-batch
    into its own optimizer update.
    """
    if not manage:
        yield
        return
    with accelerator.accumulate(state.training_model):
        yield


@contextmanager
def _forced_accumulate(accelerator, model, *, boundary: bool):
    """Replay-path accumulate context with an explicit boundary.

    Used together with ``_run_microbatch(manage_accumulate=False)`` so accelerate
    never re-derives the boundary. ``accelerator.step`` (accelerate's monotonic
    ``_do_sync`` counter) is saved and restored around the replay: the replay does
    not consume accumulation slots, so the counter must keep pointing where the
    failed attempt left it or the *next* window would sync one micro-batch early
    or late.
    """
    gradient_state = getattr(accelerator, "gradient_state", None)
    set_sync = getattr(gradient_state, "_set_sync_gradients", None)
    if set_sync is None:
        # Only test doubles get here. This fallback hands the boundary back to
        # accelerate, which is exactly blocker B1: on ``end_of_dataloader``
        # micro-batches ``_do_sync`` forces a sync each time and a replayed window
        # becomes one optimizer step per micro-batch. Warn loudly so a real
        # accelerator that lost the internal API is never silently mis-replayed.
        logger.warning(
            "Replay boundary delegated to accelerator.accumulate(): "
            "accelerator.gradient_state._set_sync_gradients is unavailable. "
            "On epoch-end windows this can apply one update per replayed "
            "micro-batch (see findings E3 / blocker B1)."
        )
        with accelerator.accumulate(model):
            yield
        return
    previous_sync = bool(gradient_state.sync_gradients)
    previous_step = getattr(accelerator, "step", None)
    set_sync(bool(boundary))
    try:
        if boundary or not bool(getattr(accelerator, "use_distributed", False)):
            yield
        elif hasattr(accelerator, "no_sync"):
            with accelerator.no_sync(model):
                yield
        else:  # pragma: no cover - defensive: distributed without no_sync
            yield
    finally:
        set_sync(previous_sync)
        if previous_step is not None:
            accelerator.step = previous_step
            # The replay must not consume accumulation slots: if this drifts, the
            # next window syncs one micro-batch early or late.
            assert accelerator.step == previous_step, (
                "replay altered accelerate's accumulation counter"
            )


def _replay_window(trainer, state: LoopState, batches) -> tuple[list[float], torch.Tensor]:
    """Re-run a whole window once with the same data (E3 retry #1)."""
    accelerator = state.accelerator
    # The failed attempt left (possibly non-finite) partial gradients behind.
    state.optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    last_loss: Optional[torch.Tensor] = None
    total = len(batches)
    for index, batch in enumerate(batches):
        with _forced_accumulate(
            accelerator, state.training_model, boundary=index == total - 1
        ):
            last_loss = _run_microbatch(
                trainer,
                state,
                batch,
                force_skip=not window_accounting.losses_are_finite(losses),
                # The boundary is set by _forced_accumulate above; letting
                # accelerate re-derive it would override it (blocker B1).
                manage_accumulate=False,
            )
        losses.append(window_accounting.loss_value(last_loss))
    return losses, last_loss


def _run_epoch_steps(trainer, state: LoopState, epoch: int) -> None:
    """Inner per-step loop: walk the dataloader, execute the accumulate
    scope, run sample / save / log / step-validation ticks.

    When a percent-based stage schedule is enabled, crossing a stage boundary
    rebuilds the dataset indices and DataLoader, then continues consuming the
    new loader until the epoch budget or max_train_steps is hit.
    """
    args = state.args
    accelerator = state.accelerator
    stage_updates_per_epoch = int(
        getattr(args, "_stage_num_update_steps_per_epoch", 0) or 0
    )
    epoch_end_step = (
        min(int(args.max_train_steps), (epoch + 1) * stage_updates_per_epoch)
        if stage_updates_per_epoch > 0
        else None
    )

    skipped_dataloader = None
    if state.initial_step > 0:
        skipped_dataloader = accelerator.skip_first_batches(
            state.train_dataloader, state.initial_step - 1
        )
        state.initial_step = 1

    window_accounting.ensure_counters(state)
    if state.batches_consumed == 0 and state.resume_skipped_batches:
        # The resume plan rewound the dataloader; the counter starts at the skipped
        # prefix so a later save reports the same position the original run had.
        state.batches_consumed = state.resume_skipped_batches

    step = -1
    while state.global_step < args.max_train_steps and (
        epoch_end_step is None or state.global_step < epoch_end_step
    ):
        _maybe_apply_stage_schedule(state)
        loader = skipped_dataloader or state.train_dataloader
        skipped_dataloader = None
        stage_at_start = state.stage_index
        pass_start_step = state.global_step
        window = window_accounting.WindowBuffer()
        for batch in loader:
            step += 1
            state.current_step.value = state.global_step
            if state.initial_step > 0:
                state.initial_step -= 1
                continue

            _profiler_step_begin(state)
            loss = _run_microbatch(
                trainer,
                state,
                batch,
                # An earlier micro-batch of this window already poisoned the
                # accumulators: keep this attempt from touching the weights.
                force_skip=not window_accounting.losses_are_finite(window.losses),
            )
            _profiler_step_end(state)
            state.batches_consumed += 1
            window.add(batch, loss)

            keys_scaled, mean_norm, maximum_norm, max_mean_logs = _maybe_scale_norm(state)

            if accelerator.sync_gradients:
                # E3: the window boundary decides commit vs replay vs discard.
                window.record_attempt()
                reason = window.failure_reason(
                    update_skipped=bool(getattr(state, "update_skipped", False))
                )
                while (
                    reason is not None
                    and window.replays < window_accounting.WINDOW_MAX_REPLAYS
                ):
                    replay_losses, loss = _replay_window(trainer, state, window.batches)
                    window.record_attempt(replay_losses)
                    reason = window.failure_reason(
                        update_skipped=bool(getattr(state, "update_skipped", False))
                    )

                # The outcome of a replay is rank-local again. Agree it, so a peer
                # that must discard cannot let this rank commit – committing here
                # would advance ``global_step`` and every later collective out of
                # step with that peer (contract §6 W-2(c)).
                if multigpu.agree_window_discard(accelerator, discard=reason is not None):
                    reason = reason or window_accounting.REASON_PEER_WINDOW_DECISION

                if reason is None:
                    # Ledger first, then the boundary side effects (E3 contract E).
                    window_accounting.commit_window(state, window)
                    _commit_step_boundary(trainer, state, loss)
                    state.progress_bar.update(1)
                    state.global_step += 1
                    _record_recent_step_seconds(
                        state,
                        state.global_step,
                        record=not bool(getattr(state, "update_skipped", False)),
                    )
                    _sample_at_step(trainer, state)
                    state.saver.maybe_save_step(
                        state.network,
                        state.global_step,
                        epoch,
                        loader_cursor=loader_cursor_payload(state),
                    )
                    state.optimizer_train_fn()
                    adaptive_notify(trainer, "commit", state)
                else:
                    dropped = window_accounting.discard_window(state, window, reason)
                    state.optimizer.zero_grad(set_to_none=True)
                    budget = window_accounting.discard_budget(args)
                    logger.warning(
                        "Discarding accumulation window after %d attempt(s): %s "
                        "(window_discards=%d, consecutive=%d, budget=%d, steps %d/%d). "
                        "No counter, ledger or checkpoint moved for this window.",
                        dropped["attempts"],
                        reason,
                        dropped["discards"],
                        dropped["consecutive_discards"],
                        budget,
                        state.global_step,
                        int(args.max_train_steps),
                    )
                    if dropped["consecutive_discards"] > budget:
                        # D3 budget layer, kept as the escalation seam: the trial is
                        # stopped with its own ``nonfinite_skip`` classification
                        # instead of silently discarding windows forever.
                        from library.training.adaptive_runtime.training_runtime import (
                            NonfiniteSkip,
                        )

                        who = (
                            "Adaptive worker stopped"
                            if window_accounting.adaptive_trial(args)
                            else "Training stopped"
                        )
                        raise NonfiniteSkip(
                            f"{who}: accumulation window discarded "
                            f"({dropped['consecutive_discards']} consecutive > budget "
                            f"{budget}; reason={reason}, "
                            f"effective_steps={state.global_step})"
                        )
                window.reset()

            _log_step(
                trainer,
                state,
                loss=loss,
                step=step,
                epoch=epoch,
                keys_scaled=keys_scaled,
                mean_norm=mean_norm,
                maximum_norm=maximum_norm,
                max_mean_logs=max_mean_logs,
            )
            _maybe_run_step_validation(trainer, state, epoch)

            if state.global_step >= args.max_train_steps:
                return

            if (
                stage_schedule_enabled(args)
                and state.train_dataset_group is not None
                and stage_at_start >= 0
            ):
                stages = parse_stage_specs(getattr(args, "stage_schedule", None))
                progress = progress_from_steps(
                    state.global_step, int(args.max_train_steps or 1)
                )
                if stages and resolve_stage_index(stages, progress) != state.stage_index:
                    break
        else:
            # Exhausted current loader without mid-epoch stage change.
            if (
                stage_updates_per_epoch > 0
                and epoch_end_step is not None
                and state.global_step < epoch_end_step
            ):
                if state.global_step == pass_start_step:
                    raise RuntimeError(
                        "stage schedule dataloader yielded no optimizer updates; "
                        "refusing to recycle an empty loader indefinitely"
                    )
                # A stage can be smaller than one full staged-data epoch.
                # Recycle it until the full epoch step budget is consumed.
                continue
            return


def _guard_nonfinite_grads(
    accelerator, state: LoopState, network, args
) -> Optional[GradFiniteness]:
    """Decide what a non-finite gradient means for this step (findings D1/D2).

    * a ``GradScaler`` owns fp16 overflow: never abort the run, only report it
      when the debug checks are on (D1 — the old unconditional raise was a false
      positive);
    * without a scaler nothing else protects the step, so the update is skipped
      instead of pushing NaN weights through ``optimizer.step`` (D2).

    Scanning only happens at optimizer boundaries, or on every microstep when the
    debug checks are enabled (their previous cadence).
    """
    debug = debug_finite_enabled(args)
    # Only the optimizer boundary is judgeable: mid-window gradients are still
    # accumulating and a transient overflow there must not abort or skip a step.
    if not accelerator.sync_gradients and not debug:
        return None
    # With a scaler the decision is always "let the scaler own it", so scanning
    # would be pure overhead unless the debug checks asked for the detail.
    if not debug and scaler_present(accelerator):
        return None
    finiteness = inspect_grad_ownership(
        accelerator, accelerator.unwrap_model(network)
    )
    if not finiteness.bad:
        return finiteness
    if finiteness.scaler_owned:
        if debug:
            log_scaler_owned_overflow(finiteness)
        return finiteness
    if not accelerator.sync_gradients:
        return None
    log_skipped_step(finiteness)
    state.nonfinite_grad_skips += 1
    return finiteness


def _commit_step_boundary(trainer, state: LoopState, loss: torch.Tensor) -> None:
    """Boundary side effects that may only run once a window is **committed**.

    E3: a window that fails (non-finite loss, scaler skip, or D2 guard drop) is
    replayed before any of this runs, so a failed attempt can never advance the
    LR scheduler, the adaptive runtime or the precision counters.
    """
    _step_lr_scheduler(state, loss)
    adaptive_notify(trainer, "after_optimizer", state)
    skipped = bool(getattr(state, "update_skipped", False))
    precision_observe("optimizer", state)
    precision_observe("optimizer", state, after=True, skipped=skipped)
    state.optimizer.zero_grad(set_to_none=True)
    _maybe_auto_block_swap(state)


def _run_step(trainer, state: LoopState, batch) -> torch.Tensor:
    """Run one micro-batch and, at an optimizer boundary, commit its side effects.

    Kept as the single-batch entry for direct callers and tests; the epoch loop
    drives :func:`_run_microbatch` itself so a failed window can be replayed.
    """
    loss = _run_microbatch(trainer, state, batch)
    # Transaction semantics (E3): a boundary whose update was dropped (scaler skip
    # or D2 guard drop) commits nothing — no scheduler, no adaptive event, no
    # precision counter — exactly like the window loop's failed attempt.
    if state.accelerator.sync_gradients and not bool(getattr(state, "update_skipped", False)):
        _commit_step_boundary(trainer, state, loss)
    return loss


def _run_microbatch(
    trainer,
    state: LoopState,
    batch,
    *,
    force_skip: bool = False,
    manage_accumulate: bool = True,
) -> torch.Tensor:
    """The accumulate-scope body: on_step_start hooks, cudagraph mark, forward,
    backward gating, manual adapter sync, grad capture/clip, optimizer step +
    zero_grad. Returns a detached loss for logging and sample scheduling.

    Deliberately free of boundary side effects (LR scheduler, adaptive runtime,
    precision counters, block-swap promotion): those belong to
    :func:`_commit_step_boundary` and only run for a committed window.

    ``force_skip`` lets the window loop suppress the update of an attempt that is
    already known bad (an earlier micro-batch of the same window had a non-finite
    loss): a failed attempt must not move the weights even once. A non-finite
    micro-batch loss does the same on its own (E3: the check is unconditional).

    ``manage_accumulate=False`` makes the caller own the accumulation boundary
    (replay path): accelerate's ``accumulate()`` is not entered, so its ``_do_sync``
    cannot overwrite the boundary that :func:`_forced_accumulate` set.
    """
    from library.training.auto_block_swap.online import before_microstep, after_microstep

    adaptive_notify(trainer, "forward")
    before_microstep(state, batch)
    args = state.args
    # The opt-in noise/timestep fixture (contract §13 V-2.4) identifies samples by
    # their global slot; the loop is the only place that knows this micro-batch's
    # local dataloader index. One int per micro-batch, unread unless the fixture
    # is enabled.
    consumed = getattr(state, "batches_consumed", None)
    args._multigpu_batch_index = None if consumed is None else int(consumed)
    accelerator = state.accelerator
    network = state.network
    memory_probe = _memory_probe_for_step(trainer, state)
    peak_probe = _peak_probe_for_step(trainer, state)
    async_gradient_sync = _async_gradient_sync_for_step(trainer, state)
    if async_gradient_sync is not None:
        async_gradient_sync.begin_step()

    _probe_step(memory_probe, state, "step_enter")
    if peak_probe is not None:
        peak_probe.begin_step(state.global_step, device=accelerator.device)

    with _accumulate_scope(accelerator, state, manage=manage_accumulate):
        state.on_step_start_for_network(state.text_encoder, state.unet)

        # preprocess batch for each model
        trainer.on_step_start(state.train_ctx, batch, is_train=True)

        # Clear last-step gate/σ tensor refs + memoized router-stats caches
        # before the next forward. Called unconditionally — the cudagraph
        # branch below also needs it (lingering refs into the cudagraph
        # memory pool block pool reclamation, demoting the run to eager),
        # and the eager path needs it so per-step memoized stats
        # (``_router_stats_cache`` / ``_chimera_router_stats_cache``) get
        # invalidated each step instead of freezing at their first computed
        # values. Cost is ~60 Python attr writes; stats compute itself is
        # already log-step-gated by callers.
        net_unwrapped = accelerator.unwrap_model(network)
        if hasattr(net_unwrapped, "clear_step_caches"):
            net_unwrapped.clear_step_caches()

        # CUDAGraphs (reduce-overhead / max-autotune) also need an explicit
        # iteration boundary for inductor's cudagraph_trees. Without this
        # call, the "pending, uninvoked backwards" fast-path check fails
        # every step and cudagraphs silently fall back to the eager path —
        # you pay compile latency and keep launch overhead. Must be called
        # before the forward on every step.
        if trainer._cudagraph_mark_step:
            torch.compiler.cudagraph_mark_step_begin()

        if state.profile_started:
            torch.cuda.nvtx.range_push("forward")
        _probe_step(memory_probe, state, "before_forward")
        try:
            trainer._state.personalization_observer["global_step"] = state.global_step
            loss = trainer.process_batch(state.train_ctx, batch, is_train=True)
        except Exception as exc:
            _probe_step(memory_probe, state, "forward_exception", **_probe_exception_fields(exc))
            raise
        if debug_finite_enabled(args):
            check_loss_finite(
                loss,
                mixed_precision=getattr(accelerator, "mixed_precision", None),
            )
        # E3: unconditional (not debug-gated) — a non-finite micro-batch loss makes
        # the whole accumulation window invalid, so this attempt must not update.
        loss_is_finite = math.isfinite(window_accounting.loss_value(loss))
        _probe_step(
            memory_probe,
            state,
            "after_forward",
            loss_dtype=getattr(loss, "dtype", None),
            loss_device=getattr(loss, "device", None),
            loss_requires_grad=getattr(loss, "requires_grad", None),
        )
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

        if state.profile_started:
            torch.cuda.nvtx.range_push("backward")
        _probe_step(memory_probe, state, "before_backward")
        adaptive_notify(trainer, "backward")
        try:
            accelerator.backward(loss)
        except Exception as exc:
            _probe_step(memory_probe, state, "backward_exception", **_probe_exception_fields(exc))
            raise
        _probe_step(memory_probe, state, "after_backward")
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

        # Post-backward adapter hook (before clip/step) — injects extra grad
        # contributions that can't share the primary backward, e.g. soft-tokens
        # gradient-cached contrastive negatives under active block swapping.
        try:
            trainer.run_after_backward(state.train_ctx)
        except Exception as exc:
            _probe_step(memory_probe, state, "after_backward_hook_exception", **_probe_exception_fields(exc))
            raise

        if accelerator.sync_gradients:
            try:
                # Keep loss-scaled gradients scaled until every rank has the
                # same reduced values. Any FP16 overflow then propagates to all
                # ranks before GradScaler decides whether to skip the step.
                if async_gradient_sync is not None:
                    # The async path returns the same GradientSyncResult shape, so
                    # identity evidence is filled from real counts here too instead
                    # of staying at "pending" (review t17 low-②).
                    async_result = async_gradient_sync.finish_step()
                    try:
                        from library.training.multigpu import identity as _identity

                        _identity.note_gradient_sync(
                            async_result, accelerator=accelerator, async_overlap=True
                        )
                    except Exception as exc:  # evidence bookkeeping must not train-fail
                        logger.debug("identity evidence update skipped: %s", exc)
                else:
                    sync_result = synchronize_optimizer_gradients(
                        accelerator, state.optimizer
                    )
                    # The identity manifest records what the gradient path really
                    # did (bucket count / reduced parameters / mode), instead of a
                    # static guess made before the first boundary (review t4 F1).
                    try:
                        from library.training.multigpu import identity as _identity

                        _identity.note_gradient_sync(sync_result, accelerator=accelerator)
                    except Exception as exc:  # evidence bookkeeping must not train-fail
                        logger.debug("identity evidence update skipped: %s", exc)
            except Exception as exc:
                _probe_step(
                    memory_probe,
                    state,
                    "gradient_sync_exception",
                    **_probe_exception_fields(exc),
                )
                raise

        grad_finiteness = _guard_nonfinite_grads(accelerator, state, network, args)
        # No GradScaler means nothing else protects the update: drop it, but keep
        # the rest of the step accounting identical to a scaler-skipped step
        # (scheduler advances, counters increment, probes still run).
        skip_update = grad_finiteness is not None and grad_finiteness.must_skip_step
        gradient_flow_probe = None
        # Known gap when skip_update is True: capture_up_grad_stats below is skipped,
        # so the ``hydra_up_grad`` metric reports the previous step's stash (a stale
        # value, never an incorrect update). Fixing it needs the consumers to know
        # the metric is missing for this step.

        if accelerator.sync_gradients and not skip_update and not force_skip and loss_is_finite:
            net_unwrapped = accelerator.unwrap_model(network)
            gradient_flow_probe = _gradient_flow_probe_for_step(trainer, state)
            if gradient_flow_probe is not None:
                try:
                    scaler = getattr(accelerator, "scaler", None)
                    grad_scale = scaler.get_scale() if scaler is not None else 1.0
                    gradient_flow_probe.capture_before_optimizer(
                        state.global_step + 1,
                        net_unwrapped,
                        grad_scale=grad_scale,
                    )
                except Exception as exc:  # telemetry must not fail training
                    logger.warning("gradient-flow probe capture failed: %s", exc)
                    gradient_flow_probe.fail("before_optimizer", state.global_step + 1, exc)
                    gradient_flow_probe = None
            # Snapshot Hydra up-weight grad norms before zero_grad wipes them.
            # The metric ``hydra_up_grad`` reads this stash later in the step.
            # Also runs pre-clip so absolute magnitudes aren't distorted by
            # the global rescale (clipping preserves the below/above ratio).
            # Skip on non-log steps — the metric only fires at log cadence,
            # so capturing every step burns kernels whose output is never
            # read. global_step increments below, so predict the
            # post-increment value.
            _log_every = max(1, int(getattr(args, "log_every_n_steps", 1) or 1))
            _will_log_after = state.is_tracking and (
                ((state.global_step + 1) % _log_every == 0)
                or ((state.global_step + 1) >= args.max_train_steps)
            )
            if _will_log_after and hasattr(net_unwrapped, "capture_up_grad_stats"):
                net_unwrapped.capture_up_grad_stats()
            if args.max_grad_norm != 0.0:
                params_to_clip = accelerator.unwrap_model(
                    network
                ).get_trainable_params()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

        if state.profile_started:
            torch.cuda.nvtx.range_push("optimizer")
        _probe_step(memory_probe, state, "before_optimizer")
        adaptive_notify(trainer, "optimizer")
        try:
            # Cleared before the step so an exception cannot leave the previous
            # boundary's answer behind for a later reader.
            state.update_skipped = None
            boundary_error: Optional[BaseException] = None
            # Multi-process: agree the drop/step decision *before* any rank may
            # enter ``optimizer.step()``. A rank that is locally clean must still
            # drop when a peer is not, otherwise only one rank reaches
            # ``scaler.step()``/``update()`` inside AcceleratedOptimizer and the
            # scale/growth tracker diverge (contract §6 W-3.1/§6 W-3.2).
            if accelerator.sync_gradients:
                state.multigpu_step_agreement = multigpu.agree_step(
                    accelerator,
                    loss_nonfinite=not loss_is_finite,
                    gradient_nonfinite=bool(skip_update),
                    force_skip=bool(force_skip),
                )
                drop_update = state.multigpu_step_agreement.drop_update
            else:
                drop_update = skip_update or force_skip or not loss_is_finite
            if drop_update:
                state.update_skipped = True
                # Nothing applied: clear the (possibly poisoned) gradients here so a
                # replay or the next window starts clean. Applied updates are zeroed
                # by the commit path, which keeps ``optimizer → scheduler → zero``.
                # Mid-window (sync_gradients False) the inner step is already a no-op
                # through AcceleratedOptimizer; this call is the backstop for the
                # boundary case and keeps the grad buffers finite either way.
                #
                # Multi-process mid-window this call is **skipped on purpose**: the
                # boundary agreement above is what makes the final decision
                # symmetric, and deleting this rank's real accumulation while a
                # peer keeps accumulating is exactly the divergence the protocol
                # removes. Every rank then drops and zeroes at the boundary.
                if accelerator.sync_gradients or not multigpu.is_multiprocess(accelerator):
                    state.optimizer.zero_grad(set_to_none=True)
            else:
                # A rank-local failure here (adapter save hook, optimizer state,
                # instrumented probe) must not let this rank leave the boundary
                # while its peer enters the next collective. The failure is first
                # captured, then agreed through the *same* primitive every rank
                # calls in this order; the class-II transport case is excluded so
                # we never start a new collective on a broken process group
                # (contract §9; review t4 F4).
                try:
                    state.optimizer.step()
                except Exception as exc:
                    boundary_error = exc
                state.update_skipped = bool(
                    getattr(accelerator, "optimizer_step_was_skipped", False)
                )
                if state.update_skipped:
                    state.optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                multigpu.negotiate_boundary_error(
                    accelerator, boundary_error, dropped=bool(drop_update)
                )
                # Bounded symmetric abort: scale/growth tracker must be identical
                # on every rank, so a bypass of the agreement above fails loudly
                # on all ranks instead of drifting silently (contract §6 W-3.6).
                multigpu.assert_scaler_symmetry(accelerator, where="optimizer_boundary")
            if accelerator.sync_gradients and gradient_flow_probe is not None:
                try:
                    gradient_flow_probe.capture_after_optimizer(state.global_step + 1)
                except Exception as exc:  # telemetry must not fail training
                    logger.warning("gradient-flow probe update capture failed: %s", exc)
                    gradient_flow_probe.fail("after_optimizer", state.global_step + 1, exc)
        except Exception as exc:
            if accelerator.sync_gradients and gradient_flow_probe is not None:
                gradient_flow_probe.discard_pending()
            _probe_step(memory_probe, state, "optimizer_exception", **_probe_exception_fields(exc))
            raise
        _probe_step(memory_probe, state, "after_optimizer")
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

    if peak_probe is not None:
        peak_probe.end_step(device=accelerator.device)

    after_microstep(state, loss)
    # Callbacks above still receive the original graph. The outer loop only
    # needs the scalar value; retaining grad_fn across VAE decode keeps frozen
    # ConvRot/NF4 constants from a completed backward alive.
    return loss.detach()


def _maybe_auto_block_swap(state: LoopState) -> None:
    """Promote swapped blocks to resident once, at an optimizer boundary.

    Performance-only feature: the promotion runs with cleared gradients and no live
    graph, so the offloader can migrate residency transactionally. Any failure is
    logged and training continues with the manual blocks_to_swap.

    The decision needs a measured step time, because the promotion stops once the
    projected DMA time is already hidden behind compute. When no measurement exists
    yet (the first boundary has no previous timestamp), the call waits one more
    boundary instead of promoting blind.
    """
    if not getattr(state.args, "block_swap_auto", False):
        return
    if getattr(state, "_block_swap_auto_done", False):
        return
    now = time.perf_counter()
    previous = getattr(state, "_block_swap_auto_boundary_at", None)
    state._block_swap_auto_boundary_at = now
    step_ms = None
    recent = _recent_step_seconds(state)
    if recent:
        step_ms = float(recent) * 1000.0
    elif previous is not None:
        step_ms = (now - previous) * 1000.0
    if step_ms is None or step_ms <= 0:
        logger.info(
            "block_swap_auto: waiting one optimizer boundary for a measured step time"
        )
        return
    state._block_swap_auto_done = True
    try:
        from library.runtime.block_swap_auto import maybe_promote

        state._block_swap_auto_plan = maybe_promote(
            getattr(state, "unet", None),
            margin_mib=getattr(state.args, "block_swap_vram_margin_mib", None),
            step_ms=step_ms,
            dma_ratio_floor=getattr(state.args, "block_swap_auto_stop_ratio", 0.0) or None,
        )
    except Exception as exc:  # noqa: BLE001 - a perf feature must not fail a run
        logger.warning("block_swap_auto promotion failed: %s", exc)


def _step_lr_scheduler(state: LoopState, loss: torch.Tensor) -> None:
    scheduler = state.lr_scheduler
    inner_scheduler = getattr(scheduler, "scheduler", scheduler)
    if (
        getattr(inner_scheduler, "is_loss_aware_lr_scheduler", False)
        and state.accelerator.sync_gradients
    ):
        reduced_loss = state.accelerator.reduce(loss.detach(), reduction="mean")
        step_loss = float(reduced_loss.float().item())
        if hasattr(inner_scheduler, "set_step_loss"):
            inner_scheduler.set_step_loss(step_loss)
            try:
                scheduler.step()
            finally:
                if hasattr(inner_scheduler, "clear_step_loss"):
                    inner_scheduler.clear_step_loss()
            return
        scheduler.step(loss=step_loss)
        return
    scheduler.step()


def _profiler_step_begin(state: LoopState) -> None:
    if (
        state.profile_range
        and state.global_step == state.profile_range[0]
        and not state.profile_started
    ):
        state.accelerator.print(f"\n[profiler] starting at step {state.global_step}")
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        state.profile_started = True

    if state.profile_started:
        torch.cuda.nvtx.range_push(f"step={state.global_step}")


def _profiler_step_end(state: LoopState) -> None:
    if state.profile_started:
        torch.cuda.nvtx.range_pop()  # close per-step NVTX range
    if state.profile_started and state.global_step >= state.profile_range[1]:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        flush_block_swap_profile = getattr(state.unet, "flush_block_swap_profile", None)
        if callable(flush_block_swap_profile):
            try:
                flush_block_swap_profile(blocking=True)
            except Exception:
                pass
        state.accelerator.print(f"\n[profiler] stopped at step {state.global_step}")
        state.accelerator.print(
            "[profiler] open the .nsys-rep with the Nsight Systems GUI\n"
        )
        state.profile_started = False
        state.profile_range = None  # don't re-trigger
        # Hard-exit so the accelerate launcher exits and nsys finalizes the
        # report. sys.exit(0) hangs in interpreter shutdown here (DataLoader
        # workers + NCCL/CUDA atexit handlers all wait on futexes). os._exit
        # skips that cleanup; the profile buffer is already flushed by the
        # preceding cuda.synchronize() + cuProfilerStop, and the .nsys-rep is
        # owned by nsys, not this process, so the hard exit is safe.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _maybe_scale_norm(state: LoopState):
    args = state.args
    if args.scale_weight_norms:
        keys_scaled, mean_norm, maximum_norm = state.accelerator.unwrap_model(
            state.network
        ).apply_max_norm_regularization(
            args.scale_weight_norms, state.accelerator.device
        )
        max_mean_logs = {
            "Keys Scaled": keys_scaled,
            "Average key norm": mean_norm,
        }
        return keys_scaled, mean_norm, maximum_norm, max_mean_logs
    return None, None, None, {}


def _sample_at_step(trainer, state: LoopState) -> None:
    state.optimizer_eval_fn()
    _run_sample_with_probe(
        trainer,
        state.accelerator,
        state.args,
        None,
        state.global_step,
        lambda: trainer.sample_images(
            state.accelerator,
            state.args,
            None,
            state.global_step,
            state.accelerator.device,
            state.vae,
            state.tokenizers,
            state.text_encoder,
            state.unet,
            network=state.network,
        ),
    )


def _cuda_memory_logs(device) -> dict[str, float]:
    try:
        torch_device = torch.device(device)
    except Exception:
        return {}
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return {}
    try:
        idx = torch_device.index
        if idx is None:
            idx = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        max_allocated = torch.cuda.max_memory_allocated(idx)
        max_reserved = torch.cuda.max_memory_reserved(idx)
    except Exception:
        return {}
    gib = float(1024**3)
    return {
        "cuda/memory_allocated_gb": allocated / gib,
        "cuda/memory_reserved_gb": reserved / gib,
        "cuda/max_memory_allocated_gb": max_allocated / gib,
        "cuda/max_memory_reserved_gb": max_reserved / gib,
    }


def _log_step(
    trainer,
    state: LoopState,
    *,
    loss,
    step: int,
    epoch: int,
    keys_scaled,
    mean_norm,
    maximum_norm,
    max_mean_logs,
) -> None:
    args = state.args
    log_every = max(1, int(getattr(args, "log_every_n_steps", 1) or 1))
    # Gate on sync_gradients: with gradient_accumulation_steps > 1 the
    # dataloader fires this hook once per micro-batch but global_step only
    # advances on sync. Without the gate, log_every_n_steps decides the same
    # answer for every micro-batch in an accumulation cycle, producing bursts
    # of N back-to-back tracker writes followed by N silent ones (exactly the
    # "sometimes honored, sometimes ignored" pattern).
    should_log_step = state.accelerator.sync_gradients and (
        (state.global_step % log_every == 0)
        or (state.global_step >= args.max_train_steps)
    )

    current_loss = loss.detach().item()
    # E3: a non-finite micro-batch loss belongs to an invalidated window (replayed
    # or discarded), so it must never enter the training-loss ledger — otherwise a
    # single NaN poisons the epoch average for the rest of the epoch.
    loss_is_finite = window_accounting.losses_are_finite([current_loss])
    if should_log_step:
        from library.training.auto_block_swap.monitor import observe_training

        observe_training(args, state.accelerator.device, state.global_step)
    if loss_is_finite:
        window_accounting.record_loss(
            state.loss_recorder, epoch=epoch, step=step, loss=current_loss
        )
    avr_loss: float = state.loss_recorder.moving_average
    memory_logs = _cuda_memory_logs(state.accelerator.device) if should_log_step else {}
    recent_step_seconds = _recent_step_seconds(state)
    logs = {}
    if recent_step_seconds is not None:
        logs["recent_s_per_step"] = f"{recent_step_seconds:.2f}"
    logs["avr_loss"] = avr_loss
    logs.update(memory_logs)
    from library.training.auto_block_swap.online import metrics as dynamic_swap_metrics

    logs.update(dynamic_swap_metrics(state))
    logs.update(_current_stage_fields(state))
    logs.update(
        personalization_metrics(
            getattr(getattr(trainer, "_state", None), "personalization_observer", {})
        )
    )
    _unwrapped_net = state.accelerator.unwrap_model(state.network)
    # Refresh router_H only on log cadence — get_router_entropy → full
    # get_router_stats compute (with D2H syncs) is wasted if the only
    # consumer is the progress-bar postfix. Cache last value on trainer
    # so tqdm shows a stale value harmlessly between log steps.
    if getattr(_unwrapped_net, "_use_hydra", False) and should_log_step:
        _router_H = _unwrapped_net.get_router_entropy()
        if _router_H is not None:
            trainer._last_router_H_postfix = _router_H
    _router_H_cached = getattr(trainer, "_last_router_H_postfix", None)
    if _router_H_cached is not None:
        logs["router_H"] = f"{_router_H_cached:.3f}"
    state.progress_bar.set_postfix(refresh=False, **{**max_mean_logs, **logs})

    # The Phase-0 progress sink (GUI / daemon progress bar tails progress.jsonl)
    # needs `step` events even with no tracker configured. When tracking, the
    # step_logging call below already feeds the sink via dispatch_logs; emit a
    # lightweight event directly only when untracked, so the bar advances
    # without paying for the full generate_step_logs + collect_metrics path.
    progress_sink = getattr(trainer, "progress_sink", None)
    if should_log_step and not state.is_tracking and progress_sink is not None:
        progress_sink.log(logs, global_step=state.global_step, epoch=epoch + 1)

    if state.is_tracking and should_log_step:
        logs = trainer.generate_step_logs(
            args,
            current_loss,
            avr_loss,
            state.lr_scheduler,
            state.lr_descriptions,
            state.optimizer,
            keys_scaled,
            mean_norm,
            maximum_norm,
            None,  # mean_grad_norm — not tracked here
            None,  # mean_combined_norm — not tracked here
        )
        if recent_step_seconds is not None:
            logs["recent_s_per_step"] = recent_step_seconds
        logs.update(memory_logs)
        producers = [_unwrapped_net, *trainer._adapters]
        logs.update(
            collect_metrics(
                producers,
                MetricContext(args=args, network=_unwrapped_net),
            )
        )
        trainer.step_logging(state.accelerator, logs, state.global_step, epoch + 1)


def _maybe_run_step_validation(trainer, state: LoopState, epoch: int) -> None:
    args = state.args
    should_validate_step = (
        args.validate_every_n_steps is not None
        and state.global_step % args.validate_every_n_steps == 0
    )
    if (
        state.accelerator.sync_gradients
        and state.validation_steps > 0
        and should_validate_step
    ):
        run_validation(
            trainer,
            state.train_ctx,
            state.val_ctx,
            val_loss_recorder=state.val_step_loss_recorder,
            epoch=epoch,
            global_step=state.global_step,
            progress_bar=state.progress_bar,
            progress_desc="validation steps",
            postfix_label="val_avg_loss",
            log_avg_key="loss/validation/step_average",
            log_div_key="loss/validation/step_divergence",
            logging_fn=trainer.step_logging,
        )


def _run_epoch_validation(trainer, state: LoopState, epoch: int) -> None:
    args = state.args
    should_validate_epoch = (
        (epoch + 1) % args.validate_every_n_epochs == 0
        if args.validate_every_n_epochs is not None
        else True
    )
    if should_validate_epoch and len(state.val_ctx.dataloader) > 0:
        run_validation(
            trainer,
            state.train_ctx,
            state.val_ctx,
            val_loss_recorder=state.val_epoch_loss_recorder,
            epoch=epoch,
            global_step=state.global_step,
            progress_bar=state.progress_bar,
            progress_desc="epoch validation steps",
            postfix_label="val_epoch_avg_loss",
            log_avg_key="loss/validation/epoch_average",
            log_div_key="loss/validation/epoch_divergence",
            logging_fn=trainer.epoch_logging,
        )


def _log_epoch_average(trainer, state: LoopState, epoch: int) -> None:
    if not state.is_tracking:
        return
    logs = {"loss/epoch_average": state.loss_recorder.moving_average}
    trainer.epoch_logging(state.accelerator, logs, state.global_step, epoch + 1)


def _run_adapter_epoch_hooks(trainer, state: LoopState) -> None:
    """Per-method end-of-epoch hooks (IP-Adapter diagnostic dump, …).
    Main process only — adapters that need cross-rank reduction should do
    that internally."""
    if not (trainer._adapters and state.accelerator.is_main_process):
        return
    epoch_end_ctx = StepCtx(
        args=state.args,
        accelerator=state.accelerator,
        network=state.network,
        weight_dtype=state.weight_dtype,
    )
    for adapter in trainer._adapters:
        adapter.on_epoch_end(epoch_end_ctx)
