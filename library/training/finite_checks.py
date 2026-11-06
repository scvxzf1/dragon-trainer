"""Fail-fast checks for invalid training loss and adapter gradients."""

from __future__ import annotations

import torch

from library.training.scaler_ownership import inspect_trainable_grads
from library.training.v100_flash import env_flag


def debug_finite_enabled(args) -> bool:
    return bool(getattr(args, "debug_finite_checks", False)) or env_flag(
        "ANIMA_DEBUG_FINITE"
    )


def check_loss_finite(
    loss: torch.Tensor,
    *,
    mixed_precision: str | None = None,
) -> None:
    if torch.isfinite(loss.detach()).all():
        return
    hint = ""
    if mixed_precision == "fp16":
        hint = (
            " fp16 autocast is active; inspect batch/sigma, residual-range "
            "guards, and the fp16-safe FinalLayer projection path."
        )
    raise FloatingPointError(
        "non-finite training loss before backward; aborting to avoid logging "
        "NaN averages or saving an invalid checkpoint." + hint
    )


def check_trainable_grads_finite(network, *, scaler_owned: bool = False) -> tuple[str, ...]:
    """Scan trainable gradients and raise unless a GradScaler owns the overflow.

    ``scaler_owned=True`` means a ``GradScaler`` is present, so a non-finite
    gradient is an ordinary fp16 overflow that the scaler will skip; the caller
    gets the offenders back for logging/counting instead of a fatal error (D1).
    """
    bad = list(inspect_trainable_grads(network))
    if bad and not scaler_owned:
        raise FloatingPointError(
            "non-finite trainable gradients after backward: " + "; ".join(bad)
        )
    return tuple(bad)
