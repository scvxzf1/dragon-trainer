"""Launch-time admission for multi-process training.

Two decisions are frozen here (contract §7 P0-1 and P0-4):

* an explicit ``ANIMA_ACCELERATE_NUM_PROCESSES`` may not contradict the GPU
  selection – the launcher refuses instead of starting "N cards, 1 worker";
* the process count that is actually in force is the ``world_size`` handed to
  the compatibility matrix, both from the CLI path and from the Web preflight.

Reason codes are stable strings: callers surface them verbatim, so a Web
preflight and a CLI run explain a rejection the same way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

NUM_PROCESSES_ENV = "ANIMA_ACCELERATE_NUM_PROCESSES"
LAUNCH_ENV = "ANIMA_ACCELERATE_LAUNCH"

#: Stable reason codes (never translated, never renumbered).
CODE_INVALID_NUM_PROCESSES = "multigpu_num_processes_invalid"
CODE_SELECTION_MISMATCH = "multigpu_num_processes_selection_mismatch"
CODE_SINGLE_GPU_MULTIPROCESS = "multigpu_single_gpu_multi_process"
CODE_LAUNCH_WITHOUT_WORKERS = "multigpu_launch_without_workers"


@dataclass(frozen=True)
class Admission:
    """The admitted process count plus why it was accepted or rejected."""

    ok: bool
    code: Optional[str]
    message: str
    num_processes: int
    world_size: int
    selection: tuple[int, ...]
    explicit: bool

    def raise_if_rejected(self) -> None:
        if not self.ok:
            raise ValueError(f"{self.code}: {self.message}")


def _env_flag_enabled(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_selection(gpu_selection: Optional[Sequence[int]]) -> tuple[int, ...]:
    out: list[int] = []
    for raw in gpu_selection or ():
        try:
            index = int(str(raw).strip())
        except (TypeError, ValueError):
            continue
        if index < 0 or index in out:
            continue
        out.append(index)
    return tuple(out)


def resolve_admission(
    gpu_selection: Optional[Sequence[int]],
    env: Mapping[str, str] | None = None,
    *,
    launch_requested: Optional[bool] = None,
) -> Admission:
    """Resolve the worker count implied by the selection and the explicit env.

    Rules (all fail closed, none of them silently rewrites a user value):

    * no selection and no explicit value → single process;
    * ``N >= 2`` selected GPUs and no explicit value → ``N`` workers (the
      existing WebUI default, kept);
    * a run that will launch workers must have ``num_processes == len(selection)``
      whenever a selection is given: fewer workers than cards (including the
      ``1 worker / N cards`` case, which gets its own reason code) **and** more
      workers than cards are both refused. "The user wrote it explicitly" is not
      an escape hatch – the contract has no approved multi-worker-per-GPU mode;
    * launch requested with a resolved worker count of ``1`` → reject, because
      that starts an accelerate job that cannot do anything multi-process.

    With ``ANIMA_ACCELERATE_LAUNCH`` explicitly disabled the process count is a
    dormant value for a single-process run and is left untouched.
    """
    values = dict(os.environ if env is None else env)
    selection = _normalize_selection(gpu_selection)
    raw = (values.get(NUM_PROCESSES_ENV) or "").strip()
    explicit = bool(raw)
    if launch_requested is None:
        launch_requested = _env_flag_enabled(values.get(LAUNCH_ENV)) or (
            len(selection) >= 2 and not values.get(LAUNCH_ENV)
        )

    if explicit:
        try:
            declared = int(raw)
        except ValueError:
            return Admission(
                ok=False,
                code=CODE_INVALID_NUM_PROCESSES,
                message=(
                    f"{NUM_PROCESSES_ENV} must be a positive integer, got {raw!r}"
                ),
                num_processes=1,
                world_size=1,
                selection=selection,
                explicit=True,
            )
        if declared < 1:
            return Admission(
                ok=False,
                code=CODE_INVALID_NUM_PROCESSES,
                message=(
                    f"{NUM_PROCESSES_ENV} must be a positive integer, got {raw!r}"
                ),
                num_processes=1,
                world_size=1,
                selection=selection,
                explicit=True,
            )
    else:
        declared = len(selection)

    num_processes = max(1, int(declared))

    # Consistency is only enforced for a run that will actually launch workers.
    # With ``ANIMA_ACCELERATE_LAUNCH=0`` the process count is a dormant value
    # (single-process run), and the existing behaviour is to leave it untouched.
    if not launch_requested:
        return Admission(
            ok=True,
            code=None,
            message=(
                f"launch disabled; {num_processes} worker value kept as-is over "
                f"selection {list(selection) or 'all visible'}"
            ),
            num_processes=num_processes,
            world_size=1,
            selection=selection,
            explicit=explicit,
        )

    if len(selection) >= 2 and num_processes == 1:
        return Admission(
            ok=False,
            code=CODE_SINGLE_GPU_MULTIPROCESS,
            message=(
                f"{NUM_PROCESSES_ENV}=1 contradicts the {len(selection)} selected "
                f"GPU(s) {list(selection)}; starting one worker would silently "
                f"train on a single card. Pass one card or {len(selection)} workers"
            ),
            num_processes=1,
            world_size=1,
            selection=selection,
            explicit=explicit,
        )

    # Any other mismatch between the selected cards and the worker count is
    # refused in **both** directions. There is deliberately no "explicit value
    # wins" escape hatch: a run that starts more workers than cards, or fewer,
    # is a launch error, not a user preference to be preserved (contract §7
    # P0-1; repair round 2 / T3-F1). Sharing one GPU between several workers
    # would need an explicit, separately evidenced opt-in that the contract does
    # not currently define, so it is rejected rather than assumed.
    if selection and num_processes != len(selection):
        return Admission(
            ok=False,
            code=CODE_SELECTION_MISMATCH,
            message=(
                f"{NUM_PROCESSES_ENV}={num_processes} does not match the "
                f"{len(selection)} selected GPU(s) {list(selection)}; the worker "
                "count and the card selection must agree"
            ),
            num_processes=num_processes,
            world_size=num_processes,
            selection=selection,
            explicit=explicit,
        )

    if num_processes < 2:
        return Admission(
            ok=False,
            code=CODE_LAUNCH_WITHOUT_WORKERS,
            message=(
                f"{LAUNCH_ENV} is set but the resolved worker count is "
                f"{num_processes}; a multi-process launch needs at least 2"
            ),
            num_processes=num_processes,
            world_size=1,
            selection=selection,
            explicit=explicit,
        )

    world_size = num_processes
    return Admission(
        ok=True,
        code=None,
        message=(
            f"{num_processes} worker(s) over selection {list(selection) or 'all visible'}"
        ),
        num_processes=num_processes,
        world_size=world_size,
        selection=selection,
        explicit=explicit,
    )


def resolve_world_size(
    gpu_selection: Optional[Sequence[int]] = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """The ``world_size`` to hand to the compatibility matrix.

    This is the number of workers the process will actually take part in, not a
    placeholder: the training process and the Web preflight both call it, so the
    multi-process incompatibilities are evaluated against the real topology.

    Inside a launched worker ``WORLD_SIZE`` is authoritative (torchrun/accelerate
    set it), so the in-process check never has to guess from the launcher intent.
    """
    values = dict(os.environ if env is None else env)
    raw = str(values.get("WORLD_SIZE", "")).strip()
    if raw:
        try:
            launched_world = int(raw)
        except ValueError:
            launched_world = 0
        if launched_world >= 1:
            # The launcher already created the group: its WORLD_SIZE is the
            # only authoritative answer, even when it is *smaller* than what the
            # admission decision would infer from the environment. The admission
            # value is a fallback for processes that were not launched by a
            # multi-process launcher (review t4).
            return launched_world
    return max(1, int(resolve_admission(gpu_selection, values).world_size))
