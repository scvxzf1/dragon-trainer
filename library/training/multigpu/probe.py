"""Opt-in startup collective probe (contract §7 P0-2).

Purpose: turn a broken transport into a loud failure *before* the first
optimizer boundary, instead of a hang inside the first gradient bucket.

Rules frozen by the contract:

* **off by default** – nothing runs unless ``ANIMA_MULTIGPU_STARTUP_PROBE`` (or
  ``--multigpu_startup_probe``) asks for it, so ordinary single-card and
  multi-card runs pay nothing;
* **only in multi-process mode** – a single process has no peer to reach;
* **failure means the whole job fails** – there is no "continue on one card"
  fallback, because silently downgrading a multi-card run to a single card
  changes the global batch the user asked for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

PROBE_ENV = "ANIMA_MULTIGPU_STARTUP_PROBE"


class StartupProbeError(RuntimeError):
    """Raised when the startup collective probe cannot complete."""


@dataclass(frozen=True)
class ProbeResult:
    ran: bool
    world_size: int
    reduced_total: float
    expected_total: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "world_size": self.world_size,
            "reduced_total": self.reduced_total,
            "expected_total": self.expected_total,
        }


def probe_requested(args=None, env: Mapping[str, str] | None = None) -> bool:
    """Whether the probe was explicitly requested by CLI flag or environment."""
    values = os.environ if env is None else env
    if str(values.get(PROBE_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return bool(getattr(args, "multigpu_startup_probe", False))


def run_startup_probe(
    accelerator,
    *,
    expected_world_size: Optional[int] = None,
) -> ProbeResult:
    """Run one tiny all-reduce and verify every rank took part.

    A sum of ``1.0`` from every rank must come back as exactly the world size.
    Any other value means a rank is missing (or double-counted), which is a
    transport/topology fault and must abort the job.
    """
    world = int(getattr(accelerator, "num_processes", 1) or 1)
    if world <= 1:
        return ProbeResult(ran=False, world_size=world, reduced_total=0.0, expected_total=0.0)

    expected = int(expected_world_size or world)
    try:
        flag = torch.ones(1, dtype=torch.float32, device=accelerator.device)
        total = float(accelerator.reduce(flag, reduction="sum").item())
    except Exception as exc:  # transport fault, not a training fault
        raise StartupProbeError(
            "multi-process startup probe failed: the first collective did not "
            f"complete ({type(exc).__name__}: {exc}). Refusing to continue – a "
            "multi-card run must not silently degrade to one card."
        ) from exc

    if abs(total - expected) > 1e-3:
        raise StartupProbeError(
            "multi-process startup probe failed: expected a summed flag of "
            f"{expected} from the configured world size but the collective "
            f"returned {total}. Refusing to continue with a partially joined "
            "process group."
        )
    return ProbeResult(
        ran=True,
        world_size=world,
        reduced_total=total,
        expected_total=float(expected),
    )


def maybe_run_startup_probe(accelerator, args=None, *, env=None) -> ProbeResult:
    """Run the probe only when it was explicitly requested."""
    if not probe_requested(args, env):
        return ProbeResult(ran=False, world_size=0, reduced_total=0.0, expected_total=0.0)
    return run_startup_probe(accelerator)
