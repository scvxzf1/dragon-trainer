"""Training mixed-precision / VAE dtype policy helpers."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

CapabilityFn = Callable[[], tuple[int, int]]


def _default_get_capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability()


def resolve_mixed_precision(
    args,
    *,
    get_capability: Optional[CapabilityFn] = None,
) -> None:
    """Back-write ``args.mixed_precision`` for pre-Ampere GPUs in place.

    Only acts when current value is ``bf16`` and GPU major < 8.
    Capability probe failure is fail-closed: keep bf16 and warn.

    ``base_compute=nf4`` is exempt: NF4 hard-requires a bf16 ``compute_dtype``
    (``weights.py::load_krea2_dit`` rejects anything else), so rewriting to fp16
    would make the NF4 base fail to load on exactly the pre-Ampere cards NF4 is
    meant to serve. The bf16 outer dtype is what carries the tiny gradients
    across layers; ``--nf4_dir1`` recovers fp16 tensor cores in the inner GEMM
    without touching ``compute_dtype``.
    """
    if getattr(args, "mixed_precision", None) != "bf16":
        return
    if not torch.cuda.is_available():
        return
    if str(getattr(args, "base_compute", "bf16") or "bf16").strip().lower() == "nf4":
        logger.info(
            "base_compute=nf4 requires bf16 compute_dtype; skipping the sm<8 "
            "bf16->fp16 rewrite. Pass --nf4_dir1 to run the inner GEMM in fp16 "
            "tensor cores."
        )
        return

    probe = get_capability or _default_get_capability
    try:
        major, _minor = probe()
    except Exception:
        logger.warning(
            "could not read GPU compute capability; keeping --mixed_precision bf16."
        )
        return

    if major < 8:
        args.mixed_precision = "fp16"
        # Remember *why* the value changed: the runtime identity record must not
        # present the launcher hint as the precision that was actually used.
        try:
            setattr(
                args,
                "_mixed_precision_source",
                f"precision_policy:sm_{major}{_minor}<8 has no native bf16 "
                "(rewritten from bf16)",
            )
        except Exception:  # pragma: no cover - defensive: args may be frozen
            pass
        logger.warning(
            "GPU sm_%d%d has no native bf16 (bf16 autocast runs the slower "
            "fp32 emulation) — auto-switching --mixed_precision from bf16 to "
            "fp16. On pre-Ampere GPUs this switch applies whenever the value is "
            "bf16 (default or explicit); Ampere+ keeps bf16 natively.",
            major,
            _minor,
        )


def resolve_vae_dtype(
    args,
    weight_dtype: torch.dtype,
    *,
    get_capability: Optional[CapabilityFn] = None,
) -> torch.dtype:
    """Derive VAE dtype, forcing fp32 where fp16 decode is unsafe."""
    if getattr(args, "no_half_vae", False):
        return torch.float32
    if getattr(args, "half_vae", False):
        return weight_dtype
    if getattr(args, "mixed_precision", None) != "fp16":
        return weight_dtype
    if not torch.cuda.is_available():
        return weight_dtype

    probe = get_capability or _default_get_capability
    try:
        major, minor = probe()
    except Exception:
        logger.warning(
            "could not read GPU compute capability; keeping VAE dtype at "
            f"{weight_dtype} (fp16 decode artifacts possible on pre-Ampere)."
        )
        return weight_dtype

    if major < 8:
        logger.info(
            "pre-Ampere GPU (sm_%d%d) under fp16: forcing VAE to fp32 to avoid "
            "decode artifacts (花图/糊图). Pass --half_vae to allow half-precision "
            "VAE (not recommended).",
            major,
            minor,
        )
        return torch.float32
    return weight_dtype
