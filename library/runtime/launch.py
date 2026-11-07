"""Shared launch argument helpers for training subprocesses."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

ACCELERATE_NUM_PROCESSES_ENV = "ANIMA_ACCELERATE_NUM_PROCESSES"
ACCELERATE_LAUNCH_ENV = "ANIMA_ACCELERATE_LAUNCH"
ACCELERATE_MIXED_PRECISION_ENV = "ANIMA_ACCELERATE_MIXED_PRECISION"
_ACCELERATE_MIXED_PRECISION_CHOICES = {"no", "fp16", "bf16"}


def _env_flag_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_accelerate_num_processes(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    raw = values.get(ACCELERATE_NUM_PROCESSES_ENV, "").strip()
    if not raw:
        return "1"
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{ACCELERATE_NUM_PROCESSES_ENV} must be a positive integer, got {raw!r}"
        ) from exc
    if count < 1:
        raise ValueError(
            f"{ACCELERATE_NUM_PROCESSES_ENV} must be a positive integer, got {raw!r}"
        )
    return str(count)


def resolve_accelerate_mixed_precision(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    raw = values.get(ACCELERATE_MIXED_PRECISION_ENV, "").strip().lower()
    if not raw:
        return "bf16"
    if raw not in _ACCELERATE_MIXED_PRECISION_CHOICES:
        raise ValueError(
            f"{ACCELERATE_MIXED_PRECISION_ENV} must be one of "
            f"{sorted(_ACCELERATE_MIXED_PRECISION_CHOICES)}, got {raw!r}"
        )
    return raw


def configure_accelerate_for_gpu_selection(
    env: MutableMapping[str, str],
    gpu_selection: Sequence[int],
) -> None:
    """Default WebUI multi-GPU selections to one Accelerate worker per GPU.

    The selection and any explicit ``ANIMA_ACCELERATE_NUM_PROCESSES`` are checked
    together first (:func:`library.training.multigpu.admission.resolve_admission`):
    a contradiction is refused here, so the Web launcher, the CLI and the daemon
    all fail closed with the same reason code instead of starting "N cards, 1
    worker" or "1 card, N workers".
    """
    from library.training.multigpu.admission import resolve_admission

    resolve_admission(gpu_selection, env).raise_if_rejected()

    if len(gpu_selection) < 2:
        return

    launch_value = env.get(ACCELERATE_LAUNCH_ENV, "").strip()
    if not launch_value:
        env[ACCELERATE_LAUNCH_ENV] = "1"
        launch_value = "1"
    if _env_flag_enabled(launch_value) and not env.get(ACCELERATE_NUM_PROCESSES_ENV, "").strip():
        env[ACCELERATE_NUM_PROCESSES_ENV] = str(len(gpu_selection))


def resolve_training_world_size_for_gpu_selection(
    gpu_selection: Sequence[int],
    env: Mapping[str, str] | None = None,
) -> int:
    """Resolve the worker count produced by the WebUI launch policy."""

    values = dict(os.environ if env is None else env)
    configure_accelerate_for_gpu_selection(values, gpu_selection)
    if not _env_flag_enabled(values.get(ACCELERATE_LAUNCH_ENV)):
        return 1
    return int(resolve_accelerate_num_processes(values))


def accelerate_training_command_prefix(
    python_exe: str,
    train_script: str | Path,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    values = os.environ if env is None else env
    if not _env_flag_enabled(values.get(ACCELERATE_LAUNCH_ENV)):
        return [python_exe, str(train_script)]
    return [
        python_exe,
        "-m",
        "accelerate.commands.accelerate_cli",
        "launch",
        "--num_processes",
        resolve_accelerate_num_processes(env),
        "--num_machines",
        "1",
        "--dynamo_backend",
        "no",
        "--num_cpu_threads_per_process",
        "3",
        "--mixed_precision",
        resolve_accelerate_mixed_precision(env),
        str(train_script),
    ]
