import ast
import importlib
import logging
import math
from typing import Any, Optional

import torch
from torch.optim import Optimizer

from library.training.optimizers import (
    is_schedulefree_optimizer,
    is_self_managed_lr_optimizer,
)

# transformers (~1.3s) and diffusers (~2s) are imported lazily inside
# get_scheduler_fix so that merely importing this module (and, transitively,
# library.train_util) doesn't pay for them. Only the actual scheduler-build
# path needs them, and only the adafactor/piecewise_constant branches reach
# into transformers.Adafactor / diffusers.optimization respectively.

logger = logging.getLogger(__name__)

LULU_LOSS_GATED_COSINE = "lulu_loss_gated_cosine"
LULU_LLOSS_WEIGHTED_ANNEALED_COSINE = "lulu_lloss_weighted_annealed_cosine"
LULU_LOSS_GATED_COSINE_ALIASES = {
    LULU_LOSS_GATED_COSINE,
    LULU_LLOSS_WEIGHTED_ANNEALED_COSINE,
}


def get_dummy_scheduler(optimizer: Optimizer) -> Any:
    class DummyScheduler:
        def __init__(self, optimizer: Optimizer):
            self.optimizer = optimizer

        def step(self):
            pass

        def get_last_lr(self):
            if hasattr(self.optimizer, "get_learning_rates"):
                return list(self.optimizer.get_learning_rates())
            return [group["lr"] for group in self.optimizer.param_groups]

    return DummyScheduler(optimizer)


class LuluLossWeightedAnnealedCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Cosine annealing scheduler whose late phase can pause on useful loss drops."""

    is_loss_aware_lr_scheduler = True

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        num_warmup_steps: int,
        num_training_steps: int,
        gamma: float = 2.0,
        loss_ema_beta: float = 0.90,
        loss_min_delta_rel: float = 0.002,
        loss_min_delta_abs: float = 0.0,
        plateau_patience_steps: int = 8,
        improvement_cooldown_steps: int = 4,
        max_hold_steps: int = 32,
        min_phase_advance_ratio: float = 0.05,
        loss_gate_start_weight: float = 0.25,
        min_lr_ratio: float = 0.0,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        if num_training_steps <= 0:
            raise ValueError("num_training_steps must be positive")
        if num_warmup_steps < 0:
            raise ValueError("num_warmup_steps must be non-negative")

        self.num_warmup_steps = int(num_warmup_steps)
        self.num_training_steps = int(num_training_steps)
        self.gamma = max(0.0, float(gamma))
        self.loss_ema_beta = min(0.9999, max(0.0, float(loss_ema_beta)))
        self.loss_min_delta_rel = max(0.0, float(loss_min_delta_rel))
        self.loss_min_delta_abs = max(0.0, float(loss_min_delta_abs))
        self.plateau_patience_steps = max(1, int(plateau_patience_steps))
        self.improvement_cooldown_steps = max(0, int(improvement_cooldown_steps))
        self.max_hold_steps = max(1, int(max_hold_steps))
        self.min_phase_advance_ratio = max(0.0, float(min_phase_advance_ratio))
        self.loss_gate_start_weight = min(1.0, max(0.0, float(loss_gate_start_weight)))
        self.min_lr_ratio = min(1.0, max(0.0, float(min_lr_ratio)))
        self.num_cycles = max(0.0, float(num_cycles))

        self.loss_ema: float | None = None
        self.best_loss: float | None = None
        self.plateau_steps = 0
        self.cooldown_steps = 0
        self.hold_steps = 0
        self.loss_gate_phase = 0.0
        self._pending_loss: float | None = None
        self._next_step_loss: float | None = None
        self._last_phase_progress = 0.0
        self._last_loss_weight = 0.0
        self._last_loss_gate_effect = 0.0

        super().__init__(optimizer, last_epoch=last_epoch)

    @property
    def decay_steps(self) -> int:
        return max(1, self.num_training_steps - self.num_warmup_steps)

    def set_step_loss(self, loss: float | torch.Tensor | None) -> None:
        """Queue one loss sample for the next scheduler step.

        Accelerate may call the wrapped scheduler multiple times per optimizer
        step on multi-process runs. Queueing lets only the first inner step
        update the loss-gate state while the remaining inner steps advance LR.
        """
        self._next_step_loss = self._loss_to_float(loss)

    def clear_step_loss(self) -> None:
        self._next_step_loss = None

    def step(self, epoch: int | None = None, loss: float | torch.Tensor | None = None):
        self._pending_loss = self._loss_to_float(loss)
        if self._pending_loss is not None:
            self._next_step_loss = None
        elif self._next_step_loss is not None:
            self._pending_loss = self._next_step_loss
            self._next_step_loss = None
        try:
            super().step(epoch=epoch)
        finally:
            self._pending_loss = None

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            warmup = max(1, self.num_warmup_steps)
            factor = max(0.0, float(self.last_epoch) / warmup)
            return [base_lr * factor for base_lr in self.base_lrs]

        step_progress = self._step_progress()
        loss_weight = step_progress**self.gamma
        loss_gate_effect = self._loss_gate_effect(step_progress, loss_weight)
        phase_progress = (
            step_progress * (1.0 - loss_weight) + loss_gate_effect * loss_weight
        )
        phase_progress = min(1.0, max(self._last_phase_progress, phase_progress))

        self._last_phase_progress = phase_progress
        self._last_loss_weight = loss_weight
        self._last_loss_gate_effect = loss_gate_effect

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * 2.0 * self.num_cycles * phase_progress)
        )
        factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        return [base_lr * factor for base_lr in self.base_lrs]

    def _step_progress(self) -> float:
        step_index = max(0, self.last_epoch - self.num_warmup_steps)
        return min(1.0, step_index / self.decay_steps)

    @staticmethod
    def _loss_to_float(loss: float | torch.Tensor | None) -> float | None:
        if loss is None:
            return None
        if isinstance(loss, torch.Tensor):
            loss = float(loss.detach().float().item())
        else:
            loss = float(loss)
        return loss if math.isfinite(loss) else None

    def _loss_gate_effect(self, step_progress: float, loss_weight: float) -> float:
        loss = self._pending_loss
        if loss is None:
            if self.loss_ema is None:
                self.loss_gate_phase = step_progress
                self.hold_steps = 0
                return step_progress
            return min(1.0, max(0.0, self.loss_gate_phase))

        improved = self._update_loss_stats(loss)
        if loss_weight < self.loss_gate_start_weight:
            self.loss_gate_phase = step_progress
            self.hold_steps = 0
            return step_progress

        if improved:
            self.plateau_steps = 0
            self.cooldown_steps = self.improvement_cooldown_steps
            should_hold = True
        elif self.cooldown_steps > 0:
            self.cooldown_steps -= 1
            should_hold = True
        else:
            self.plateau_steps += 1
            should_hold = self.plateau_steps < self.plateau_patience_steps

        if self.hold_steps >= self.max_hold_steps:
            should_hold = False

        if should_hold:
            self.hold_steps += 1
            min_advance = self.min_phase_advance_ratio / self.decay_steps
            self.loss_gate_phase = min(
                step_progress,
                max(self.loss_gate_phase, self.loss_gate_phase + min_advance),
            )
        else:
            self.hold_steps = 0
            self.loss_gate_phase = step_progress

        return min(1.0, max(0.0, self.loss_gate_phase))

    def _update_loss_stats(self, loss: float) -> bool:
        if self.loss_ema is None:
            self.loss_ema = loss
            self.best_loss = loss
            return False

        self.loss_ema = (
            self.loss_ema_beta * self.loss_ema + (1.0 - self.loss_ema_beta) * loss
        )
        if self.best_loss is None:
            self.best_loss = self.loss_ema
            return False

        min_delta = max(
            self.loss_min_delta_abs,
            abs(self.best_loss) * self.loss_min_delta_rel,
        )
        if self.loss_ema < self.best_loss - min_delta:
            self.best_loss = self.loss_ema
            return True
        return False


def get_scheduler_fix(args, optimizer: Optimizer, num_processes: int):
    """
    Unified API to get any scheduler from its name.
    """
    if is_schedulefree_optimizer(optimizer, args) or is_self_managed_lr_optimizer(
        optimizer, args
    ):
        return get_dummy_scheduler(optimizer)

    name = args.lr_scheduler
    # Under pipeline_parallel one optimizer step already covers the whole
    # pipeline, so the LR plan must not be multiplied by the world size (stages=2
    # would double max_train_steps). The gate owns that decision.
    from library.training.pp.gate import lr_scheduler_num_processes

    num_training_steps = args.max_train_steps * lr_scheduler_num_processes(
        args, num_processes
    )
    num_warmup_steps: Optional[int] = (
        int(args.lr_warmup_steps * num_training_steps)
        if isinstance(args.lr_warmup_steps, float)
        else args.lr_warmup_steps
    )
    num_decay_steps: Optional[int] = (
        int(args.lr_decay_steps * num_training_steps)
        if isinstance(args.lr_decay_steps, float)
        else args.lr_decay_steps
    )
    num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps
    num_cycles = args.lr_scheduler_num_cycles
    power = args.lr_scheduler_power
    timescale = args.lr_scheduler_timescale
    min_lr_ratio = args.lr_scheduler_min_lr_ratio

    lr_scheduler_kwargs = {}
    if args.lr_scheduler_args is not None and len(args.lr_scheduler_args) > 0:
        for arg in args.lr_scheduler_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            lr_scheduler_kwargs[key] = value

    def wrap_check_needless_num_warmup_steps(return_vals):
        if num_warmup_steps is not None and num_warmup_steps != 0:
            raise ValueError(
                f"{name} does not require `num_warmup_steps`. Set None or 0."
            )
        return return_vals

    if args.lr_scheduler_type:
        lr_scheduler_type = args.lr_scheduler_type
        logger.info(f"use {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler")
        if "." not in lr_scheduler_type:
            lr_scheduler_module = torch.optim.lr_scheduler
        else:
            values = lr_scheduler_type.split(".")
            lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
            lr_scheduler_type = values[-1]
        lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
        lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
        return wrap_check_needless_num_warmup_steps(lr_scheduler)

    if name.startswith("adafactor"):
        import transformers

        assert isinstance(optimizer, transformers.optimization.Adafactor), (
            "adafactor scheduler must be used with Adafactor optimizer"
        )
        initial_lr = float(name.split(":")[1])
        return wrap_check_needless_num_warmup_steps(
            transformers.optimization.AdafactorSchedule(optimizer, initial_lr)
        )

    # Gate on the literal value ("piecewise_constant") so the diffusers import
    # (~2s) is only paid when that scheduler is actually requested.
    if name == "piecewise_constant":
        from diffusers.optimization import (
            SchedulerType as DiffusersSchedulerType,
            TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
        )

        name = DiffusersSchedulerType(name)
        schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[name]
        return schedule_func(optimizer, **lr_scheduler_kwargs)

    if name in LULU_LOSS_GATED_COSINE_ALIASES:
        if num_warmup_steps is None:
            raise ValueError(
                f"{name} requires `num_warmup_steps`, please provide that argument."
            )
        lr_scheduler_kwargs.setdefault(
            "min_lr_ratio", min_lr_ratio if min_lr_ratio is not None else 0.0
        )
        return LuluLossWeightedAnnealedCosineScheduler(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **lr_scheduler_kwargs,
        )

    from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == SchedulerType.CONSTANT:
        return wrap_check_needless_num_warmup_steps(
            schedule_func(optimizer, **lr_scheduler_kwargs)
        )

    if num_warmup_steps is None:
        raise ValueError(
            f"{name} requires `num_warmup_steps`, please provide that argument."
        )

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs
        )

    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            timescale=timescale,
            **lr_scheduler_kwargs,
        )

    if num_training_steps is None:
        raise ValueError(
            f"{name} requires `num_training_steps`, please provide that argument."
        )

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            **lr_scheduler_kwargs,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power,
            **lr_scheduler_kwargs,
        )

    if name == SchedulerType.COSINE_WITH_MIN_LR:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles / 2,
            min_lr_rate=min_lr_ratio,
            **lr_scheduler_kwargs,
        )

    if name == SchedulerType.LINEAR or name == SchedulerType.COSINE:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **lr_scheduler_kwargs,
        )

    if num_decay_steps is None:
        raise ValueError(
            f"{name} requires `num_decay_steps`, please provide that argument."
        )
    if name == SchedulerType.WARMUP_STABLE_DECAY:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_stable_steps=num_stable_steps,
            num_decay_steps=num_decay_steps,
            num_cycles=num_cycles / 2,
            min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else 0.0,
            **lr_scheduler_kwargs,
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_decay_steps=num_decay_steps,
        **lr_scheduler_kwargs,
    )
