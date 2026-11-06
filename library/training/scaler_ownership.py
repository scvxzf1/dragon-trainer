"""Ownership-aware handling of non-finite adapter gradients.

Two documented gaps in ``docs/findings/fp16_nan_static_equivalence_20261021.md``
are decided here, because both hinge on the same question: *who owns a
non-finite gradient?*

* **D1 (false positive)**: with ``--debug_finite_checks`` the gradient scan ran
  before unscale/clip and read still-scaled gradients, so routine fp16 overflow
  was reported as a fatal training error even though the real ``GradScaler``
  would simply skip the step.
* **D2 (missing protection)**: the default path had no gradient finiteness
  handling at all. When a scaler exists that is acceptable — overflow is the
  scaler's job and it skips the update. When no scaler exists (bf16/fp32 or an
  fp16 run without autocast scaler) a non-finite gradient flows straight into
  ``optimizer.step`` and corrupts the weights.

Reordering the checks cannot fix D1 (an inf stays inf after unscale, and clipping
an inf produces ``inf * 0 = nan``), so the decision is made from ownership:
``GradScaler`` present means the overflow is scaler-owned and must not abort the
run; no scaler means this module owns the decision and the update is skipped.

Skipping mirrors the scaler semantics the loop already accepts: gradients are
dropped, the optimizer step and scheduler are not run, and ``global_step`` still
advances (exactly what a skipped ``GradScaler`` step does today).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

#: Upper bound on reported offenders; a NaN usually hits every parameter and the
#: full list would drown the log line without adding diagnostic value.
MAX_REPORTED_OFFENDERS = 8


@dataclass(frozen=True)
class GradFiniteness:
    """Result of inspecting one optimizer-boundary gradient set."""

    offenders: tuple[str, ...] = ()
    scaler_owned: bool = False

    @property
    def bad(self) -> bool:
        return bool(self.offenders)

    @property
    def must_skip_step(self) -> bool:
        """True when this module owns the decision and the update must be dropped."""
        return self.bad and not self.scaler_owned

    def summary(self) -> str:
        return "; ".join(self.offenders)


def scaler_present(accelerator) -> bool:
    """Whether ``accelerator`` carries a ``GradScaler`` (fp16 autocast)."""
    return getattr(accelerator, "scaler", None) is not None


def update_skipped(state) -> bool:
    """Whether the current optimizer boundary actually moved the weights.

    Two independent sources can skip an update: a ``GradScaler`` overflow (read
    through accelerate) and this module's guard when no scaler exists — the latter
    is invisible to ``optimizer_step_was_skipped``. The training loop publishes the
    unified answer as ``state.update_skipped``; measurement layers (adaptive
    runtime, AUTO probe, online swap controller) must read it instead of the
    accelerate property, otherwise a guard skip is counted as an effective step and
    pollutes step-time samples.
    """
    explicit = getattr(state, "update_skipped", None)
    if explicit is not None:
        return bool(explicit)
    accelerator = getattr(state, "accelerator", None)
    return bool(getattr(accelerator, "optimizer_step_was_skipped", False))


def scaled_gradient_overflow_is_scaler_owned(accelerator, gradients) -> bool:
    """Whether an overflowing gradient set belongs to a live ``GradScaler``.

    ``True`` means the protection layer must drop its own reading and let the
    scaler skip the step; ``False`` (no scaler) means no one else will protect the
    update. ``False`` also covers the non-overflow case: there is nothing to own.
    """
    if not scaler_present(accelerator):
        return False
    for grad in gradients:
        if grad is None:
            continue
        if not torch.isfinite(grad.detach()).all():
            return True
    return False


def _describe(name: str, grad: torch.Tensor) -> str:
    return f"{name}: dtype={grad.dtype} shape={tuple(grad.shape)}"


def _scan_norms_bulk(named_grads, limit: int) -> tuple[str, ...] | None:
    """One dispatcher call for the whole gradient set; ``None`` means unavailable.

    A per-tensor ``isfinite().all()`` costs one device sync per tensor, which
    dominated the step for LoRA-sized parameter counts (measured 42 ms/step at
    400 tensors). ``_foreach_norm`` reduces the set to one small tensor, so the
    scan costs two syncs regardless of how many parameters exist. Overflow (inf)
    and NaN both survive into a norm accumulated in fp32, so a finite stack proves
    every gradient is finite — accumulating in the gradient dtype would not, since
    large finite fp16 sets overflow the norm itself.
    """
    if not named_grads or not hasattr(torch, "_foreach_norm"):
        return None
    try:
        # Accumulate in fp32: the norm of a large-but-finite fp16 gradient set
        # overflows fp16 (4x40000 finite elements sum to inf), which would report
        # a healthy run as non-finite.
        norms = torch.stack(
            torch._foreach_norm(
                [grad.detach() for _, grad in named_grads], 2.0, torch.float32
            )
        )
    except Exception:  # sparse/odd dtypes, older torch: fall back to the exact loop
        return None
    finite = torch.isfinite(norms)
    if bool(finite.all()):
        return ()
    bad_indices = (~finite).nonzero(as_tuple=False).flatten().tolist()
    return tuple(
        _describe(named_grads[index][0], named_grads[index][1])
        for index in bad_indices[:limit]
    )


def inspect_trainable_grads(
    network, *, limit: int = MAX_REPORTED_OFFENDERS
) -> tuple[str, ...]:
    """Return ``name: dtype=... shape=...`` for non-finite trainable gradients."""
    named_grads = [
        (name, parameter.grad)
        for name, parameter in network.named_parameters()
        if parameter.grad is not None
    ]
    bulk = _scan_norms_bulk(named_grads, limit)
    if bulk is not None:
        return bulk
    offenders: list[str] = []
    for name, grad in named_grads:
        if torch.isfinite(grad.detach()).all():
            continue
        offenders.append(_describe(name, grad))
        if len(offenders) >= limit:
            break
    return tuple(offenders)


def _any_rank_bad(accelerator, bad: bool) -> bool:
    """All-reduce the skip decision so every rank takes the same branch.

    Divergent control flow would let one rank enter the next DDP collective while
    another skips it, which hangs the job instead of failing loudly. This costs one
    tiny all-reduce per optimizer boundary, and only on scaler-less multi-process
    runs; single-process runs never build the tensor or call ``reduce``.
    """
    processes = int(getattr(accelerator, "num_processes", 1) or 1)
    if processes <= 1:
        return bad
    flag = torch.tensor([1.0 if bad else 0.0], device=accelerator.device)
    flag = accelerator.reduce(flag, reduction="max")
    return bool(float(flag.item()) > 0.0)


def inspect_grad_ownership(accelerator, network) -> GradFiniteness:
    """Classify the current gradient set without touching scaler internal state.

    No ``unscale_`` happens here on purpose: calling it twice per iteration is an
    error in ``torch.amp.GradScaler``, and the scaled gradient already carries all
    the information needed to decide ownership.
    """
    scaler_owned = scaler_present(accelerator)
    offenders = inspect_trainable_grads(network)
    if scaler_owned:
        # The scaler already all-reduces ``found_inf`` across ranks, so every rank
        # reaches the same skip decision without help from this module.
        return GradFiniteness(offenders=offenders, scaler_owned=True)

    processes = int(getattr(accelerator, "num_processes", 1) or 1)
    if processes <= 1:
        return GradFiniteness(offenders=offenders, scaler_owned=False)

    # Without a scaler the skip decision must be identical on every rank, so the
    # flag is agreed even when this rank's own gradients are clean.
    if not _any_rank_bad(accelerator, bool(offenders)):
        return GradFiniteness(offenders=(), scaler_owned=False)
    if not offenders:  # pragma: no cover - another rank owns the observation
        offenders = ("<non-finite gradients reported by another rank>",)
    return GradFiniteness(offenders=offenders, scaler_owned=False)


def log_scaler_owned_overflow(finiteness: GradFiniteness) -> None:
    """Explain a scaler-owned overflow instead of failing the run."""
    logger.warning(
        "non-finite gradients after backward, owned by GradScaler: %s "
        "(step will be skipped; this is expected fp16 overflow, not a failure)",
        finiteness.summary(),
    )


def log_skipped_step(finiteness: GradFiniteness) -> None:
    """Warn once per offending step when no scaler exists to absorb the overflow."""
    logger.warning(
        "non-finite gradients after backward with no GradScaler to absorb them: %s "
        "(optimizer update skipped to keep weights finite; check batch/precision "
        "config if this repeats)",
        finiteness.summary(),
    )
