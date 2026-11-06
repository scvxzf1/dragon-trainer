"""Single-process runtime controller; all migrations precede a fresh update."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import torch

from library.models.krea2_raw.dynamic_compile import refresh_resident_compile
from .online_policy import OnlineSwapPolicy
from .resources import GIB, host_memory
from .probe import representative_batches
from .preferences import gpu_reserve_bytes, preference, swap_io_limit_bytes

logger = logging.getLogger(__name__)


class OnlineSwapController:
    def __init__(self, trainer, state):
        self.args = state.args
        self.trainer = trainer
        self.model = state.accelerator.unwrap_model(state.unet)
        self.device = state.accelerator.device
        self.offloader = self.model.offloader
        self.maximum = self.args._auto_swap_maximum
        self.policy = OnlineSwapPolicy(
            self.maximum, interval=self.args.auto_block_swap_interval,
            preference=preference(self.args),
        )
        self.policy.current = self.model.blocks_to_swap
        self.block_bytes = self.args._auto_swap_block_bytes
        self.reserve = gpu_reserve_bytes(
            self.args, torch.cuda.get_device_properties(self.device).total_memory
        )
        self.overhead = 0
        self.watermark = 0
        self.pending = None
        self.microsteps = 0
        self.shapes = []
        self.skipped_updates = 0
        self.consecutive_skips = 0
        group = getattr(state, "train_dataset_group", None)
        if group is None:
            group = getattr(getattr(state, "train_dataloader", None), "dataset", None)
        self.required_shapes = (
            {
                (case["batch_size"], case["height"] // 8, case["width"] // 8)
                for case in representative_batches(group)
            }
            if group is not None
            else set()
        )
        self.observed_shapes = set()
        self.started = None
        self.last_switch_seconds = 0
        self.switches = 0
        self.switch_seconds = 0
        self.controller_seconds = 0
        self.last_headroom = 0
        self.last_reason = "warmup"
        self.host = host_memory()
        self.path = Path(self.args._auto_swap_report).with_name("runtime.jsonl")
        self._update_peak()
        self._record(
            {"event": "start", "step": state.global_step, "maximum": self.maximum,
             "gpu_reserve": self.reserve, "preference": self.policy.preference}
        )

    def _resident_bytes(self, count):
        return sum(self.block_bytes[: len(self.block_bytes) - count])

    def _update_peak(self):
        peak = torch.cuda.max_memory_allocated(self.device)
        if peak > self.watermark:
            self.overhead = max(
                self.overhead, peak - self._resident_bytes(self.policy.current)
            )
            self.watermark = peak

    def _safe_count(self, *, promotion=False):
        free, _ = torch.cuda.mem_get_info(self.device)
        budget = free + torch.cuda.memory_reserved(self.device)
        margin = self.reserve + (
            max(512 * 1024**2, 2 * max(self.block_bytes)) if promotion else 0
        )
        self.last_headroom = (
            budget - self.overhead - self._resident_bytes(self.policy.current)
        )
        if promotion and not self.required_shapes <= self.observed_shapes:
            return self.maximum
        for count in range(self.maximum + 1):
            if self.overhead + self._resident_bytes(count) + margin <= budget:
                return count
        return self.maximum + 1

    def _record(self, fields):
        record = {"time": time.time(), "blocks": self.policy.current, **fields}
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        except OSError as exc:
            logger.warning("Dynamic AUTO report write failed: %s", exc)

    def _change(self, decision, step):
        if self.policy.preference == "ram":
            from .online_memory import check_master_growth

            if not check_master_growth(self, decision):
                self.policy.abandon(step)
                self._record({"event": "host_growth_refused", "step": step, "target": decision.blocks})
                return
        previous = self.policy.current
        started = time.perf_counter()
        try:
            changed = self.offloader.reconfigure(self.model.blocks, decision.blocks)
        except torch.cuda.OutOfMemoryError as exc:
            # No optimizer transaction is in progress; reconfigure rolled back.
            if decision.reason in {"gpu_pressure", "new_shape"}:
                self._record({"event": "safety_migration_oom", "step": step,
                              "target": decision.blocks, "reason": decision.reason})
                raise RuntimeError(
                    f"Dynamic AUTO safety migration failed ({decision.reason}); "
                    "refusing to start the batch"
                ) from exc
            self.policy.rejected_below = max(
                self.policy.rejected_below, decision.blocks
            )
            self.policy.retry_after = step + max(64, self.policy.interval * 8)
            self.policy.abandon(step)
            self._record(
                {"event": "migration_oom", "step": step, "target": decision.blocks}
            )
            return
        self.model.blocks_to_swap = decision.blocks
        refresh_resident_compile(self.model)
        self.args.blocks_to_swap = decision.blocks
        self.trainer.is_swapping_blocks = decision.blocks > 0
        elapsed = time.perf_counter() - started
        self.policy.changed(decision.blocks, seconds=elapsed)
        self.last_reason = decision.reason
        if changed:
            self.last_switch_seconds = elapsed
            self.switch_seconds += self.last_switch_seconds
            self.switches += 1
            logger.info(
                "Dynamic AUTO swap %s -> %s (%s), migration %.3fs",
                previous,
                decision.blocks,
                decision.reason,
                self.last_switch_seconds,
            )
            self._record(
                {
                    "event": "switch",
                    "step": step,
                    "previous": previous,
                    "reason": decision.reason,
                    "seconds": self.last_switch_seconds,
                }
            )

    def before(self, state, batch):
        shape = tuple(batch["latents"].shape)
        if self.microsteps:
            if (
                shape not in self.policy.known_inputs
                and self.policy.current < self.maximum
            ):
                raise RuntimeError(
                    "Dynamic AUTO encountered an uncalibrated shape during accumulation"
                )
        else:
            started = time.perf_counter()
            host = host_memory()
            swap_io = host.swap_in + host.swap_out - self.host.swap_in - self.host.swap_out
            if host.available <= host.reserve:
                raise RuntimeError("Dynamic AUTO host RAM/paging reserve exhausted")
            swap_limit = swap_io_limit_bytes(self.args)
            if swap_limit and swap_io > swap_limit:
                raise RuntimeError(
                    "Dynamic AUTO RAM/paging or configured system swap IO/paging limit exceeded"
                )
            safe = self._safe_count()
            guard = self.policy.guard(shape, step=state.global_step, safe_count=safe)
            if guard is not None:
                self.pending = None
                self._change(guard, state.global_step)
            elif self.pending is not None:
                decision, self.pending = self.pending, None
                floor = self._safe_count(
                    promotion=decision.blocks < self.policy.current
                )
                if decision.blocks >= floor:
                    self._change(decision, state.global_step)
                else:
                    self.policy.abandon(state.global_step)
            self.controller_seconds += time.perf_counter() - started
            self.started = torch.cuda.Event(enable_timing=True)
            self.started.record()
        self.policy.known_inputs.add(shape)
        self.observed_shapes.add((shape[0], shape[-2], shape[-1]))
        self.shapes.append(shape)
        self.microsteps += 1

    def after(self, state, loss):
        if not torch.isfinite(loss.detach()).all().item():
            raise RuntimeError(
                "Dynamic AUTO requires finite, completed optimizer updates"
            )
        if not state.accelerator.sync_gradients:
            return
        started = time.perf_counter()
        if state.accelerator.optimizer_step_was_skipped:
            # A skipped update moved no weights, so its wall time is not a sample of
            # the workload the policy is trying to measure. Drop it and keep the
            # microstep/shape accumulators clean for the next effective step.
            from library.training.adaptive_runtime.training_runtime import NonfiniteSkip

            self.skipped_updates += 1
            self.consecutive_skips += 1
            self._record(
                {"event": "skipped_update", "step": state.global_step + 1,
                 "consecutive": self.consecutive_skips}
            )
            self.shapes.clear()
            self.microsteps = 0
            self.controller_seconds += time.perf_counter() - started
            budget = max(0, int(getattr(self.args, "adaptive_max_nonfinite_skips", 0) or 0))
            if self.consecutive_skips > budget:
                raise NonfiniteSkip(
                    "Dynamic AUTO stopped: loss scaler skipped optimizer update "
                    f"({self.consecutive_skips} consecutive > budget {budget}); "
                    "skipped updates are never used as timing samples"
                )
            return
        self.consecutive_skips = 0
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        end.synchronize()
        seconds = self.started.elapsed_time(end) / 1000
        self._update_peak()
        step = state.global_step + 1
        # Match whole accumulation layouts; never compare different bucket mixes.
        shape = tuple(sorted(self.shapes))
        self.pending = self.policy.observe(
            shape,
            seconds,
            step=step,
            remaining=self.args.max_train_steps - step,
            promotion_floor=self._safe_count(promotion=True),
            switch_seconds=self.last_switch_seconds,
        )
        self._record(
            {
                "event": "update",
                "step": step,
                "seconds": seconds,
                "shapes": sorted(self.shapes),
                "headroom": self.last_headroom,
                "loss": float(loss.detach()),
                "phase": self.policy.phase,
                "verdict": self.policy.last_verdict,
                "switches": self.switches,
                "switch_seconds": self.switch_seconds,
                "controller_seconds": self.controller_seconds,
                "cpu_master_bytes": getattr(self.offloader, "_frozen_weight_master_bytes", 0),
            }
        )
        self.shapes.clear()
        self.microsteps = 0
        self.controller_seconds += time.perf_counter() - started


def attach(trainer, state):
    if (
        not getattr(state.args, "auto_block_swap", False)
        or getattr(state.args, "auto_block_swap_mode", "startup") != "dynamic"
    ):
        return
    state.auto_swap_controller = OnlineSwapController(trainer, state)


def before_microstep(state, batch):
    controller = getattr(state, "auto_swap_controller", None)
    if controller is not None:
        controller.before(state, batch)


def after_microstep(state, loss):
    controller = getattr(state, "auto_swap_controller", None)
    if controller is not None:
        controller.after(state, loss)


def metrics(state):
    controller = getattr(state, "auto_swap_controller", None)
    if controller is None:
        return {}
    return {
        "auto_swap/blocks": controller.policy.current,
        "auto_swap/switches": controller.switches,
        "auto_swap/headroom_gib": controller.last_headroom / GIB,
        "auto_swap/switch_seconds": controller.switch_seconds,
        "auto_swap/controller_seconds": controller.controller_seconds,
    }


def finish(state, *, completed):
    controller = getattr(state, "auto_swap_controller", None)
    if controller is not None:
        controller._record(
            {
                "event": "end",
                "step": state.global_step,
                "completed": completed
                and state.global_step >= state.args.max_train_steps,
                "switches": controller.switches,
                "switch_seconds": controller.switch_seconds,
                "controller_seconds": controller.controller_seconds,
            }
        )
