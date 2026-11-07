"""Explicit cross-rank agreement primitives.

Why this module exists
----------------------
``accelerate.Accelerator.reduce`` is **not** a general reduction API. In
accelerate 1.13.0 the implementation (``accelerate/utils/operations.py``) has a
single ``if reduction == "mean":`` branch and everything else falls through to
``torch.distributed.all_reduce(..., ReduceOp.SUM)``; its docstring only promises
``'sum'``, ``'mean'`` and ``'none'``. ``reduction="max"`` therefore *is* a sum,
so a passing test says nothing about the requested op.

The frozen contract (``docs/proposal/multigpu_reference_validation_contract.md``
§6 W-3) allows exactly two explicit forms for an "any rank saw this" decision:

1. ``all_reduce(SUM)`` over ``{0.0, 1.0}`` followed by ``> 0`` – implemented
   here, with the encoding written down instead of implied;
2. ``all_reduce(op=ReduceOp.MAX)`` or an equivalent explicit boolean gather.

Everything in this module is a no-op for single-process runs, so single-card
training keeps its current behaviour and pays no collective.
"""

from __future__ import annotations

from typing import Mapping

import torch

#: Relative tolerance used when checking that every rank reported the same
#: value. Float sums are compared against ``value * world_size`` instead of
#: requiring bit-identical additions, which is exact enough for the scaler
#: scale/growth-tracker magnitudes and never hides a real divergence.
_EQUAL_REL_TOL = 1e-6


def is_multiprocess(accelerator) -> bool:
    """Whether this process belongs to a multi-process run.

    ``num_processes`` is the accelerate contract for "how many workers run this
    job"; anything below ``2`` keeps the single-card path byte-identical.
    """
    try:
        return int(getattr(accelerator, "num_processes", 1) or 1) > 1
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


def _reduce_sum(accelerator, tensor: torch.Tensor) -> torch.Tensor:
    """Run one ``SUM`` all-reduce through accelerate.

    The ``reduction="sum"`` argument is spelled out on purpose: it is the only
    reduction accelerate actually implements, and the value every rank gets back
    is the same tensor (``all_reduce``, not a gather to rank 0).
    """
    return accelerator.reduce(tensor, reduction="sum")


def agree_any(accelerator, flags: Mapping[str, bool]) -> dict[str, bool]:
    """OR the given rank-local booleans across all ranks in one collective.

    Every rank returns the same mapping, so "any rank raised this" is decided by
    the whole world rather than by rank 0's private view. Flags are packed into
    a single tensor, which keeps the collective sequence identical on all ranks
    (one call, fixed shape, fixed order) regardless of how many flags are asked
    about.

    Single-process runs short-circuit: no tensor is built and no collective is
    issued.
    """
    if not flags:
        return {}
    local = {name: bool(value) for name, value in flags.items()}
    if not is_multiprocess(accelerator):
        return local
    names = tuple(flags)
    packed = torch.tensor(
        [1.0 if local[name] else 0.0 for name in names],
        dtype=torch.float32,
        device=accelerator.device,
    )
    reduced = _reduce_sum(accelerator, packed).tolist()
    return {name: float(reduced[index]) > 0.0 for index, name in enumerate(names)}


def all_ranks_equal(accelerator, values: Mapping[str, float]) -> dict[str, bool]:
    """Check that every rank reports the same value for each key.

    The check is deliberately *not* "compare against rank 0": each rank reduces
    the sum of all reports and compares it with its own value times the world
    size, so a single divergent rank makes the comparison fail on the ranks that
    hold the outlier as well as on the others.
    """
    if not values:
        return {}
    if not is_multiprocess(accelerator):
        return {name: True for name in values}
    names = tuple(values)
    world = int(getattr(accelerator, "num_processes", 1) or 1)
    packed = torch.tensor(
        [float(values[name]) for name in names],
        dtype=torch.float64,
        device=accelerator.device,
    )
    totals = _reduce_sum(accelerator, packed).tolist()
    out: dict[str, bool] = {}
    for index, name in enumerate(names):
        expected = float(values[name]) * world
        tolerance = _EQUAL_REL_TOL * max(1.0, abs(expected))
        out[name] = abs(float(totals[index]) - expected) <= tolerance
    return out


def agree_mismatch(accelerator, flags: Mapping[str, bool]) -> dict[str, bool]:
    """Alias of :func:`agree_any` used when the flags mean "I detected a
    divergence". Kept as a separate name so call sites read as a symmetric
    abort decision instead of a feature flag."""
    return agree_any(accelerator, flags)
