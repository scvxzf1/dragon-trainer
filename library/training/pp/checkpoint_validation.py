"""Pure checkpoint content validators (F2, repair-round-3).

These helpers validate *payload semantics* before any rank mutates its
runtime: PP scale, growth tracker, GradScaler state dict keys/ranges, growth
policy compatibility and the optimizer state structure (AdamW moment shapes
and legal scalar steps).  Everything here is read-only and raises
``PipelineParallelError`` with a stable ``pp_*`` code so the load-time
negotiation in :func:`~library.training.pp.checkpoint.load_distributed_checkpoint`
can refuse symmetrically before the apply phase.

Leniency contract: only the *known* AdamW keys (``exp_avg``, ``exp_avg_sq``,
``max_exp_avg_sq``) are required to match the parameter shape.  Unknown keys
of a factorized or otherwise legitimate optimizer state keep their existing
interpretation -- no registry framework and no blanket parameter-shape rule
for unknown fields. Optimizer-specific hyperparameters are not exhaustively
validated by these payload checks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from library.training.pp.errors import PipelineParallelError

CODE_CHECKPOINT_STATE_MISMATCH = "pp_checkpoint_state_mismatch"

#: AdamW-family keys whose per-parameter state must match the parameter shape.
_ADAMW_MOMENT_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")

#: GradScaler state-dict keys the transaction needs to restore correctly.
_SCALER_REQUIRED_KEYS = (
    "scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker",
)


def _mismatch(detail: str) -> PipelineParallelError:
    return PipelineParallelError(CODE_CHECKPOINT_STATE_MISMATCH, detail)


def validate_pp_scale(scale: Any) -> float:
    """The PP loss scale must be a positive, finite real number."""

    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise _mismatch(f"PP scale must be a real number, got {scale!r}")
    value = float(scale)
    if not math.isfinite(value):
        raise _mismatch(f"PP scale must be finite, got {value!r}")
    if value <= 0.0:
        raise _mismatch(f"PP scale must be positive, got {value!r}")
    return value


def validate_pp_growth_tracker(tracker: Any) -> int:
    """The PP growth tracker is its own counter and must be a legal step."""

    if isinstance(tracker, bool) or not isinstance(tracker, int):
        raise _mismatch(f"PP growth tracker must be a non-negative int, got {tracker!r}")
    if tracker < 0:
        raise _mismatch(f"PP growth tracker must be non-negative, got {tracker!r}")
    return int(tracker)


def validate_scaler_state(scaler_state: Any) -> None:
    """GradScaler payload keys, types and ranges (its own tracker, not PP's)."""

    if not isinstance(scaler_state, Mapping):
        raise _mismatch(
            f"GradScaler state must be a mapping, got {type(scaler_state).__name__}"
        )
    missing = [key for key in _SCALER_REQUIRED_KEYS if key not in scaler_state]
    if missing:
        raise _mismatch(
            f"GradScaler state is missing required keys {missing}; the scaler "
            "cannot be restored without its internal tracker settings"
        )
    for key in _SCALER_REQUIRED_KEYS:
        value = scaler_state[key]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise _mismatch(
                    f"GradScaler state {key!r} must be a scalar, got a tensor "
                    f"with {value.numel()} elements"
                )
            value = value.item()
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _mismatch(f"GradScaler state {key!r} must be numeric, got {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise _mismatch(f"GradScaler state {key!r} must be finite, got {number!r}")
    if float(scaler_state["scale"]) <= 0.0:
        raise _mismatch(
            f"GradScaler scale must be positive, got {scaler_state['scale']!r}"
        )
    if float(scaler_state["growth_factor"]) <= 1.0:
        raise _mismatch(
            "GradScaler growth_factor must be > 1, got "
            f"{scaler_state['growth_factor']!r}"
        )
    backoff = float(scaler_state["backoff_factor"])
    if not 0.0 < backoff < 1.0:
        raise _mismatch(
            f"GradScaler backoff_factor must be in (0, 1), got {backoff!r}"
        )
    interval = scaler_state["growth_interval"]
    if isinstance(interval, torch.Tensor):
        interval = interval.item()
    if isinstance(interval, float) and not float(interval).is_integer():
        raise _mismatch(
            f"GradScaler growth_interval must be a whole number, got {interval!r}"
        )
    if int(interval) < 1:
        raise _mismatch(
            f"GradScaler growth_interval must be >= 1, got {interval!r}"
        )
    tracker = scaler_state["_growth_tracker"]
    if isinstance(tracker, torch.Tensor):
        tracker = tracker.item()
    if not float(tracker).is_integer() or tracker < 0:
        raise _mismatch(
            f"GradScaler _growth_tracker must be a non-negative whole number, got {tracker!r}"
        )


def _optimizer_parameter_map(optimizer_state, optimizer):
    """Match serialized IDs by group position, just as Optimizer.load_state_dict does."""
    saved_groups = optimizer_state.get("param_groups")
    live_groups = optimizer.param_groups
    if not isinstance(saved_groups, (list, tuple)) or len(saved_groups) != len(live_groups):
        raise _mismatch("optimizer parameter groups do not match this optimizer")
    by_id = {}
    for group_index, (saved_group, live_group) in enumerate(zip(saved_groups, live_groups)):
        if not isinstance(saved_group, Mapping):
            raise _mismatch(f"optimizer parameter group {group_index} must be a mapping")
        saved_ids = saved_group.get("params")
        live_params = live_group["params"]
        if not isinstance(saved_ids, (list, tuple)) or len(saved_ids) != len(live_params):
            raise _mismatch(f"optimizer parameter group {group_index} has incompatible params")
        for saved_id, parameter in zip(saved_ids, live_params):
            try:
                if saved_id in by_id:
                    raise _mismatch(f"optimizer parameter ID {saved_id!r} is duplicated")
                by_id[saved_id] = (parameter, saved_group)
            except TypeError as exc:
                raise _mismatch(f"optimizer parameter ID {saved_id!r} is not hashable") from exc
    return by_id


def _validate_optimizer_step(step, saved_id) -> None:
    if isinstance(step, torch.Tensor):
        if step.numel() != 1:
            raise _mismatch(f"optimizer step for ID {saved_id!r} must be scalar")
        step = step.item()
    if isinstance(step, bool) or not isinstance(step, (int, float)):
        raise _mismatch(f"optimizer step for ID {saved_id!r} must be a real scalar")
    if not math.isfinite(step) or step < 0:
        raise _mismatch(f"optimizer step for ID {saved_id!r} must be finite and non-negative")


def validate_optimizer_payload(
    optimizer_state: Mapping[str, Any], optimizer: Any
) -> None:
    """Validate parameter state using saved IDs, never treating IDs as offsets.

    Known Adam-family moment tensors must match their actual live parameter.
    Other optimizers' factorized state fields keep their existing leniency.
    """
    state = optimizer_state.get("state")
    if not isinstance(state, Mapping):
        raise _mismatch("optimizer state payload must carry a 'state' mapping")
    parameters = _optimizer_parameter_map(optimizer_state, optimizer)
    for saved_id, entry in state.items():
        if saved_id not in parameters:
            raise _mismatch(f"optimizer state ID {saved_id!r} is absent from param_groups")
        if not isinstance(entry, Mapping):
            raise _mismatch(f"optimizer state for ID {saved_id!r} must be a mapping")
        parameter, saved_group = parameters[saved_id]
        # An empty state is legal before this parameter's first optimizer step.
        if entry and isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            required = ["step", "exp_avg", "exp_avg_sq"]
            if saved_group.get("amsgrad", False):
                required.append("max_exp_avg_sq")
            missing = [key for key in required if key not in entry]
            if missing:
                raise _mismatch(f"Adam state for ID {saved_id!r} is missing {missing}")
        if "step" in entry:
            _validate_optimizer_step(entry["step"], saved_id)
        for moment_key in _ADAMW_MOMENT_KEYS:
            if moment_key not in entry:
                continue
            moment = entry[moment_key]
            if not isinstance(moment, torch.Tensor):
                raise _mismatch(f"optimizer {moment_key} for ID {saved_id!r} must be a tensor")
            if tuple(moment.shape) != tuple(parameter.shape):
                raise _mismatch(
                    f"optimizer {moment_key} for ID {saved_id!r} has shape "
                    f"{tuple(moment.shape)} but the parameter is {tuple(parameter.shape)}"
                )


def validate_checkpoint_content(state: Any) -> None:
    """Validate the payload-level invariants shared by every PP checkpoint."""

    scale = validate_pp_scale(state.scale)
    validate_pp_growth_tracker(state.growth_tracker)
    if state.scaler is not None:
        validate_scaler_state(state.scaler)
        scaler_scale = float(state.scaler["scale"])
        if scaler_scale != scale:
            raise _mismatch(
                f"PP scale {scale} does not match checkpointed GradScaler scale "
                f"{scaler_scale}; refuse before applying either state"
            )
