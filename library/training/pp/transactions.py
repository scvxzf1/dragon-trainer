"""Symmetric fp16 optimizer transaction for the two pipeline stages.

Why a dedicated transaction
---------------------------
fp16 loss scaling interacts with pipeline parallelism in a way that
``accelerator.step`` / ``GradScaler.step`` cannot express:

* the loss is scaled **once**, on the last stage, so the gradient that crosses
  the stage boundary is already multiplied by that scale factor.  Every stage
  must therefore unscale by the *same* factor; a per-rank scale would divide the
  first stage's gradients by a different number than the loss was scaled with and
  the two halves of the model would drift apart silently;
* each stage only sees *its own* parameters, so one rank can overflow while the
  other does not.  ``GradScaler.step`` decides per process, so the two stages
  could step one half of the model and skip the other -- corrupting the shared
  activation contract with no error anywhere.

Both problems are solved the same way: keep **one** fp16 scale and **one** skip
decision for the whole pipeline, derived from an all-reduced overflow flag.  The
last stage owns the ``GradScaler`` (it owns the loss, so it is the only stage
that can call ``scale``); every stage advances the scale from the *agreed*
decision, so the two processes stay in lockstep without extra collectives.

``GradScaler.update(new_scale=...)`` is the public hook that lets the agreed
decision drive the scaler, and the growth/backoff constants are read from the
scaler itself, so the policy is not re-implemented from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

from .errors import (
    CODE_GRAD_SCALER_MISSING,
    CODE_SCALER_DECISION_ASYMMETRY,
    PipelineParallelError,
)


def _distributed() -> bool:
    """Whether a process group is live (the PP group is always the world)."""

    return bool(dist.is_available() and dist.is_initialized())


def _all_reduce_sum(value: int, *, group: Any = None) -> int:
    tensor = torch.tensor([int(value)], dtype=torch.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return int(tensor.item())


def gradients_have_nonfinite(gradients: Iterable[Tensor | None]) -> bool:
    """Whether any gradient holds inf/NaN, in as few device syncs as possible.

    Overflow and NaN both survive into a float32 norm, so one ``_foreach_norm``
    plus a single ``isfinite`` check is enough -- a per-tensor scan would cost one
    sync per parameter and dominate the step at LoRA-sized parameter counts.
    """

    grads = [grad for grad in gradients if grad is not None]
    if not grads:
        return False
    foreach_norm = getattr(torch, "_foreach_norm", None)
    if foreach_norm is not None:
        try:
            norms = foreach_norm(grads, 2.0, dtype=torch.float32)
        except (RuntimeError, TypeError):  # pragma: no cover - backend dependent
            norms = None
        if norms is not None:
            stacked = torch.stack([norm.detach() for norm in norms])
            return not bool(torch.isfinite(stacked).all().item())
    return any(not bool(torch.isfinite(grad.detach()).all().item()) for grad in grads)


def divide_gradients(gradients: Iterable[Tensor | None], scale: float) -> None:
    """In-place ``grad /= scale`` for every present gradient."""

    if scale <= 0:
        raise PipelineParallelError(
            CODE_GRAD_SCALER_MISSING,
            f"fp16 scale must be positive, got {scale!r}",
        )
    for grad in gradients:
        if grad is not None:
            grad.div_(scale)


def agree_overflow(local_overflow: bool, *, group: Any = None) -> bool:
    """OR the per-stage overflow verdicts; every rank returns the same answer."""

    if not _distributed():
        return bool(local_overflow)
    total = _all_reduce_sum(1 if local_overflow else 0, group=group)
    return total > 0


def verify_skip_unanimity(
    local_skip: bool, *, group: Any = None, where: str = "pp_optimizer_boundary"
) -> None:
    """Fail closed when the stages disagree about skipping this step.

    The protocol above makes the decision unanimous by construction, so a
    mismatch means some rank bypassed it (a direct ``scaler.step`` /
    ``optimizer.step``).  Detecting that here turns "one half of the model
    advanced, the other did not" into a named, symmetric abort.
    """

    if not _distributed():
        return
    world = dist.get_world_size(group)
    total = _all_reduce_sum(1 if local_skip else 0, group=group)
    if total in (0, world):
        return
    raise PipelineParallelError(
        CODE_SCALER_DECISION_ASYMMETRY,
        f"GradScaler skip decision diverged at {where}: {total}/{world} rank(s) "
        "skipped this step. Under pipeline_parallel the skip decision is derived "
        "from an all-reduced overflow flag, so a mismatch means the protocol was "
        "bypassed; refusing to continue with one half of the model stepped.",
    )


def broadcast_scale(local_scale: float, *, source_stage: int, group: Any = None) -> float:
    """Share the last stage's fp16 scale with the other stages (build time)."""

    if not _distributed():
        return float(local_scale)
    tensor = torch.tensor([float(local_scale)], dtype=torch.float64)
    dist.broadcast(tensor, src=int(source_stage), group=group)
    return float(tensor.item())


#: Fallback fp16 policy for a stage that owns no ``GradScaler``.  It is used only
#: when there is no process group to broadcast the real policy over; in a real PP
#: run the last stage's policy is broadcast, so every stage advances identically.
DEFAULT_GROWTH_FACTOR = 2.0
DEFAULT_BACKOFF_FACTOR = 0.5
DEFAULT_GROWTH_INTERVAL = 2000


def _broadcast_fp16_policy(
    growth_factor: float,
    backoff_factor: float,
    growth_interval: int,
    *,
    source_stage: int,
    group: Any = None,
) -> tuple[float, float, int]:
    """Share the last stage's fp16 growth/backoff policy with the other stages.

    Broadcasting the *scale* alone is not enough.  ``_advance`` runs rank-locally,
    so a stage that merely guessed the torch defaults (2.0 / 0.5 / 2000) would
    grow its scale at a different step than the last stage and then unscale the
    boundary gradients by the wrong factor -- a silent 2x on the first half of the
    model (measured before this fix with ``growth_interval=10``: 16.0 on the last
    stage vs 8.0 elsewhere after 12 clean steps).  The policy, not just the value,
    has to be one pipeline-wide fact.
    """

    if not _distributed():
        return float(growth_factor), float(backoff_factor), int(growth_interval)
    tensor = torch.tensor(
        [float(growth_factor), float(backoff_factor), float(growth_interval)],
        dtype=torch.float64,
    )
    dist.broadcast(tensor, src=int(source_stage), group=group)
    return (
        float(tensor[0].item()),
        float(tensor[1].item()),
        int(round(tensor[2].item())),
    )


@dataclass(frozen=True)
class PpStepOutcome:
    """What one PP optimizer boundary decided."""

    skipped: bool
    local_overflow: bool
    agreed_overflow: bool
    scale: float
    next_scale: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "skipped": self.skipped,
            "local_overflow": self.local_overflow,
            "agreed_overflow": self.agreed_overflow,
            "scale": self.scale,
            "next_scale": self.next_scale,
        }


class PpFp16Transaction:
    """One symmetric fp16 optimizer transaction shared by every PP stage."""

    def __init__(
        self,
        scaler: Any,
        optimizer: Any,
        *,
        group: Any = None,
        is_last: bool = True,
        scale: float | None = None,
        clip_gradients: Any = None,
        where: str = "pp_optimizer_boundary",
    ) -> None:
        if optimizer is None:
            raise PipelineParallelError(
                CODE_GRAD_SCALER_MISSING,
                "the PP fp16 transaction needs the stage optimizer",
            )
        self._scaler = scaler
        self._optimizer = optimizer
        self._group = group
        self._is_last = bool(is_last)
        self._clip_gradients = clip_gradients
        self._where = where
        if self._is_last and scaler is None:
            raise PipelineParallelError(
                CODE_GRAD_SCALER_MISSING,
                "the last PP stage owns the fp16 loss scale and requires a "
                "GradScaler; mixed_precision=fp16 without one underflows silently",
            )
        if self._is_last:
            local_policy = (
                scaler.get_growth_factor(),
                scaler.get_backoff_factor(),
                scaler.get_growth_interval(),
            )
        else:
            local_policy = (
                DEFAULT_GROWTH_FACTOR,
                DEFAULT_BACKOFF_FACTOR,
                DEFAULT_GROWTH_INTERVAL,
            )
        (
            self._growth_factor,
            self._backoff_factor,
            self._growth_interval,
        ) = _broadcast_fp16_policy(
            *local_policy, source_stage=self._source_stage(), group=group
        )
        local_scale = float(scale) if scale is not None else self._local_scale(scaler)
        self._scale = broadcast_scale(
            local_scale, source_stage=self._source_stage(), group=group
        )
        self._growth_tracker = 0
        self._stepped = False
        self._last_skip: bool | None = None

    # -- construction helpers ------------------------------------------------

    @staticmethod
    def _local_scale(scaler: Any) -> float:
        if scaler is None:
            return 65536.0
        return float(scaler.get_scale())

    def _source_stage(self) -> int:
        """Rank index of the last stage inside the PP group (identity mapping)."""

        if not _distributed():
            return 0
        return dist.get_world_size(self._group) - 1

    # -- public API ----------------------------------------------------------

    @property
    def scale(self) -> float:
        return float(self._scale)

    @property
    def is_last(self) -> bool:
        return self._is_last

    @property
    def scaler(self) -> Any:
        """The ``GradScaler`` this stage owns, or ``None`` on earlier stages."""

        return self._scaler

    @property
    def growth_tracker(self) -> int:
        """Steps taken since the last scale growth (part of the checkpoint)."""

        return int(self._growth_tracker)

    @property
    def growth_factor(self) -> float:
        """Pipeline-wide growth factor (broadcast from the last stage's scaler)."""

        return float(self._growth_factor)

    @property
    def backoff_factor(self) -> float:
        """Pipeline-wide backoff factor (broadcast from the last stage's scaler)."""

        return float(self._backoff_factor)

    @property
    def growth_interval(self) -> int:
        """Pipeline-wide growth interval (broadcast from the last stage's scaler)."""

        return int(self._growth_interval)

    @property
    def last_skip(self) -> bool | None:
        """Whether the last optimizer boundary skipped, or ``None`` before one.

        This is the per-step decision that must agree across the pipeline; unlike
        the fp16 *scale* it is not allowed to differ per rank.
        """

        return self._last_skip

    def restore(
        self,
        *,
        scale: float,
        growth_tracker: int = 0,
        last_skip: bool | None = None,
        where: str | None = None,
    ) -> None:
        """Resume the pipeline-wide fp16 state from a checkpoint.

        The ``GradScaler`` itself is restored by the checkpoint module through
        ``load_state_dict``; this method then re-points the transaction at the
        restored value, so a rank without a scaler (every non-last stage) resumes
        at exactly the scale the last stage owns.
        """

        restored = float(scale)
        if not restored > 0:
            raise PipelineParallelError(
                CODE_GRAD_SCALER_MISSING,
                f"restored fp16 scale must be positive, got {scale!r}",
            )
        if int(growth_tracker) < 0:
            raise PipelineParallelError(
                CODE_GRAD_SCALER_MISSING,
                f"restored growth tracker must be non-negative, got {growth_tracker!r}",
            )
        if last_skip is not None and not isinstance(last_skip, bool):
            raise PipelineParallelError(
                CODE_SCALER_DECISION_ASYMMETRY,
                f"restored skip decision must be a bool or None, got {last_skip!r}",
            )
        if self._scaler is not None:
            actual = float(self._scaler.get_scale())
            if actual != restored:
                raise PipelineParallelError(
                    CODE_SCALER_DECISION_ASYMMETRY,
                    f"GradScaler scale {actual} after load_state_dict does not match "
                    f"the checkpointed pipeline-wide scale {restored} at "
                    f"{where or self._where}; the checkpoint's scaler and scale "
                    "entries disagree",
                )
        self._scale = restored
        self._growth_tracker = int(growth_tracker)
        self._last_skip = last_skip

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Multiply the raw loss by the shared fp16 scale.

        Only the last stage computes a loss, so only it can hand the tensor to
        ``GradScaler.scale``; the other stages never see a loss.
        """

        if not self._is_last:
            raise PipelineParallelError(
                CODE_SCALER_DECISION_ASYMMETRY,
                "scale_loss() is only valid on the last PP stage, which owns the loss",
            )
        if self._scaler is None:
            raise PipelineParallelError(
                CODE_GRAD_SCALER_MISSING,
                "scale_loss() requires a GradScaler on the last PP stage",
            )
        scaled = self._scaler.scale(loss)
        self._assert_scale_in_sync()
        return scaled

    def wrap_loss(self, loss_fn):
        """Wrap a loss function so the schedule backwards through scaled loss."""

        def scaled_loss(prediction, target):
            return self.scale_loss(loss_fn(prediction, target))

        return scaled_loss

    def begin_step(self) -> None:
        """Reset the per-step latch (called before each schedule step)."""

        self._stepped = False

    def step(self) -> PpStepOutcome:
        """Unscale, agree the overflow verdict, then step or skip on every stage."""

        if self._stepped:
            raise PipelineParallelError(
                CODE_SCALER_DECISION_ASYMMETRY,
                f"PpFp16Transaction.step() ran twice without begin_step() at "
                f"{self._where}",
            )
        scale = float(self._scale)
        local_overflow = self._unscale_and_scan(scale)
        agreed = agree_overflow(local_overflow, group=self._group)
        if agreed:
            skipped = True
        else:
            if self._clip_gradients is not None:
                self._clip_gradients()
            self._optimizer.step()
            skipped = False
        next_scale = self._advance(agreed)
        if self._scaler is not None and self._is_last:
            self._scaler.update(new_scale=next_scale)
        self._scale = next_scale
        outcome = PpStepOutcome(
            skipped=skipped,
            local_overflow=bool(local_overflow),
            agreed_overflow=bool(agreed),
            scale=scale,
            next_scale=next_scale,
        )
        self._assert_next_scale_agreement(next_scale)
        verify_skip_unanimity(skipped, group=self._group, where=self._where)
        self._stepped = True
        self._last_skip = bool(skipped)
        return outcome

    # -- internals -----------------------------------------------------------

    def _parameters(self) -> list[Tensor]:
        return [p for group in self._optimizer.param_groups for p in group["params"]]

    def _unscale_and_scan(self, scale: float) -> bool:
        gradients = [p.grad for p in self._parameters()]
        if self._scaler is not None and self._is_last:
            self._scaler.unscale_(self._optimizer)
        else:
            divide_gradients(gradients, scale)
        return gradients_have_nonfinite(gradients)

    def _assert_scale_in_sync(self) -> None:
        if self._scaler is None:
            return
        actual = float(self._scaler.get_scale())
        if actual != float(self._scale):
            raise PipelineParallelError(
                CODE_SCALER_DECISION_ASYMMETRY,
                f"GradScaler scale {actual} diverged from the pipeline-wide scale "
                f"{self._scale} at {self._where}; every stage must unscale by the "
                "same factor the loss was scaled with",
            )

    def _assert_next_scale_agreement(self, next_scale: float) -> None:
        """Refuse to continue if the stages advanced the scale differently.

        ``_assert_scale_in_sync`` only compares a stage against *its own*
        ``GradScaler``, and the non-last stages have none, so a policy or restore
        divergence would otherwise be invisible until the gradients came out
        wrong.  One ``all_gather`` per step turns that into a symmetric, named
        abort: every rank sees every value and therefore every rank raises.
        """

        if not _distributed():
            return
        world = dist.get_world_size(self._group)
        mine = torch.tensor([float(next_scale)], dtype=torch.float64)
        gathered = [torch.empty_like(mine) for _ in range(world)]
        dist.all_gather(gathered, mine, group=self._group)
        values = [float(item.item()) for item in gathered]
        if any(value != values[0] for value in values):
            raise PipelineParallelError(
                CODE_SCALER_DECISION_ASYMMETRY,
                f"pipeline-wide fp16 scale diverged at {self._where}: {values}. "
                "Every stage must advance the scale from the same policy and the "
                "same agreed skip decision; a per-rank divergence silently "
                "rescales the boundary gradients by the scale ratio.",
            )

    def _advance(self, overflow: bool) -> float:
        """Next pipeline-wide scale, from the *agreed* decision."""

        if overflow:
            self._growth_tracker = 0
            return float(self._scale) * self._backoff_factor
        self._growth_tracker += 1
        if self._growth_tracker >= self._growth_interval:
            self._growth_tracker = 0
            return float(self._scale) * self._growth_factor
        return float(self._scale)


def make_clip_gradients(parameters: Sequence[Tensor], max_norm: float):
    """A clip callback that does **not** unscale (the transaction already did).

    ``accelerator.clip_grad_norm_`` would unscale again, which torch rejects with
    "unscale_() has already been called"; this helper keeps the unscale inside
    the transaction, where the symmetric decision lives.
    """

    def clip() -> None:
        torch.nn.utils.clip_grad_norm_(list(parameters), float(max_norm))

    return clip


__all__ = [
    "PpFp16Transaction",
    "PpStepOutcome",
    "agree_overflow",
    "broadcast_scale",
    "divide_gradients",
    "gradients_have_nonfinite",
    "make_clip_gradients",
    "verify_skip_unanimity",
]
