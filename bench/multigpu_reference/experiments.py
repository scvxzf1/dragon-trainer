"""Bench entries for multigpu reference experiments (t5).

Status: experimental. These entries are **not** part of any default flow and never
run as a side effect of a test session: each one requires an explicit env release,
a fresh output root and a declared update budget.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

#: Env var that releases GPU/hardware experiment execution.
RELEASE_ENV = "ANIMA_MULTIGPU_EXPERIMENT_RELEASE"
#: Default parent for experiment output roots (never the user's output dir).
DEFAULT_ROOT = "bench_output/multigpu_reference"

#: Budgets frozen by the team goal / contract §13.
GPU_STEP_BUDGET = {
    "single_gpu_steps": 10,
    "dual_gpu_steps": 20,
    "resume_steps": 20,  # 10 + 10 continuation
    "max_steps_per_run": 20,
    "max_updates_total": 50,
    "max_minutes_per_run": 20,
    "max_minutes_total": 80,
}

CODE_NOT_RELEASED = "experiment_release_missing"
CODE_BUDGET_INVALID = "experiment_budget_invalid"
CODE_OUTPUT_ROOT_INVALID = "experiment_output_root_invalid"
CODE_DEVICE_MISSING = "experiment_device_missing"


class ExperimentRefused(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class ExperimentPlan:
    name: str
    kind: str  # "cpu" | "gloo" | "cuda_small" | "real_training"
    world_size: int
    steps: int
    output_root: Path
    gated: bool = True
    notes: list[str] = field(default_factory=list)


def _released(env: Mapping[str, str]) -> bool:
    return str(env.get(RELEASE_ENV, "")).strip() in {"1", "true", "yes"}


def plan_experiment(
    name: str,
    *,
    kind: str,
    world_size: int,
    steps: int,
    output_root: str | os.PathLike[str] | None,
    env: Mapping[str, str] | None = None,
    device_count: Optional[int] = None,
) -> ExperimentPlan:
    """Validate gates before any experiment touches a device or the filesystem.

    Refusals are named and happen *before* work starts: a missing release, an
    over-budget step count, a shared/unset output root or a missing device are all
    rejections, never a silent downgrade or skip that later reads as verified.
    """
    values = dict(os.environ if env is None else env)
    if int(steps) <= 0 or int(world_size) <= 0:
        raise ExperimentRefused(CODE_BUDGET_INVALID, "steps and world_size must be >= 1")
    if int(steps) > GPU_STEP_BUDGET["max_steps_per_run"]:
        raise ExperimentRefused(
            CODE_BUDGET_INVALID,
            f"{steps} steps > per-run cap {GPU_STEP_BUDGET['max_steps_per_run']}",
        )
    if output_root is None or not str(output_root).strip():
        raise ExperimentRefused(
            CODE_OUTPUT_ROOT_INVALID,
            "an explicit output root is required; experiments never write into the "
            "user's default output directory",
        )
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise ExperimentRefused(
            CODE_OUTPUT_ROOT_INVALID,
            f"output root {root} already has content; use a fresh root per run",
        )
    if kind in {"cuda_small", "real_training"} and not _released(values):
        raise ExperimentRefused(
            CODE_NOT_RELEASED,
            f"set {RELEASE_ENV}=1 to release device work for {kind} experiments",
        )
    if kind == "cuda_small":
        required = 1 if world_size == 1 else world_size
        if device_count is not None and device_count < required:
            raise ExperimentRefused(
                CODE_DEVICE_MISSING, f"{required} device(s) required, {device_count} visible"
            )
    return ExperimentPlan(
        name=name,
        kind=kind,
        world_size=int(world_size),
        steps=int(steps),
        output_root=root,
        gated=kind in {"cuda_small", "real_training"},
        notes=[
            "budget: "
            f"max {GPU_STEP_BUDGET['max_steps_per_run']} steps/run, "
            f"{GPU_STEP_BUDGET['max_updates_total']} updates total",
            "output root must stay distinct from the user's output directory",
        ],
    )


def isolated_model_spec(*, layers: int = 4, width: int = 16, vocab: int = 32) -> dict:
    """Tiny deterministic model used by the FSDP2/TP experiments.

    Isolated on purpose: the frozen base model is never sharded in these tests, so
    a sharding experiment cannot alter production admission or assume that
    "inject after shard" is managed for us.
    """
    return {"layers": layers, "width": width, "vocab": vocab, "dtype": "float32"}
