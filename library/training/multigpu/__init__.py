"""Multi-process (DDP) state protocol for the Anima training path.

This package owns the pieces that must agree across ranks before control flow is
allowed to diverge:

* :mod:`library.training.multigpu.agreement` – explicit collective primitives
  (a boolean OR is a ``SUM`` over ``{0,1}`` compared with ``> 0``; nothing here
  relies on ``accelerator.reduce(reduction="max")``, which is a silent alias for
  ``SUM`` in accelerate 1.13.0).
* :mod:`library.training.multigpu.window_protocol` – the per-boundary window /
  skip / scaler decision that ``library/training/loop.py`` calls once per
  optimizer boundary and once per window seam.
* :mod:`library.training.multigpu.admission` – launch-time admission: GPU
  selection versus an explicit process count, plus the real ``world_size``.
* :mod:`library.training.multigpu.probe` – opt-in startup collective probe.
* :mod:`library.training.multigpu.identity` – the per-run identity record.

Frozen contract: ``docs/proposal/multigpu_reference_validation_contract.md``
§6 (P0-0), §7 (P0-1..P0-5), §9 (two failure classes), §11 (single-card parity).
"""

from __future__ import annotations

from library.training.multigpu.agreement import (
    agree_any,
    all_ranks_equal,
    is_multiprocess,
)
from library.training.multigpu.window_protocol import (
    BoundaryAgreementError,
    StepAgreement,
    agree_step,
    agree_window_discard,
    assert_scaler_symmetry,
    is_transport_error,
    negotiate_boundary_error,
    scaler_state,
)

__all__ = [
    "BoundaryAgreementError",
    "StepAgreement",
    "agree_any",
    "agree_step",
    "agree_window_discard",
    "all_ranks_equal",
    "assert_scaler_symmetry",
    "is_multiprocess",
    "is_transport_error",
    "negotiate_boundary_error",
    "scaler_state",
]
