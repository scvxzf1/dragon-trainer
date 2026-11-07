"""Non-finite *observability* for cache/batch tensors (counters + one-shot warnings).

User ruling (2026-11-07): keep the existing ``nan_to_num``/NaN-gated cleaning, the
dtype and the cache data exactly as they are; only make the numbers *visible*.
Three things were invisible before:

* a ±Inf-only tensor produced no message at all (the scrub is NaN-gated);
* nothing said whether the non-finite values were **already** there before the
  dtype conversion or were **introduced by it** (a finite source that overflows a
  narrower storage/compute dtype);
* there was no identity (site / cache path / dtype / shape) and no counter that a
  test or an artifact could read.

This module is observation only — it never rejects, rewrites, stops training or
adds a collective. Every event bumps a counter and, the first time a
``(site, kind)`` pair is seen in this process, emits one warning; later events of
the same pair only increase the counters (no per-sample log spam).

Kinds
-----
``source_nan``      NaN in the tensor *before* the conversion/cleaning point;
``source_inf``      ±Inf in the tensor before that point;
``cast_overflow``   values that were finite before the conversion and non-finite
                    after it (a finite source that does not fit the narrower dtype).

Payload discipline: only scalars and *bounded* identity reach this module (site name,
dtype names, shape tuple, cache/tensor identity string, element counts). Tensors are
never copied, serialised or logged, and nothing is written to disk here — the counters
are published with the existing precision-report lifecycle writes, never per step.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

SITE_LATENTS = "batch_step.latents"

KIND_SOURCE_NAN = "source_nan"
KIND_SOURCE_INF = "source_inf"
KIND_CAST_OVERFLOW = "cast_overflow"

_COUNTERS: dict[str, dict[str, int]] = {}
_ELEMENTS: dict[str, dict[str, int]] = {}
_LOGGED: set[tuple[str, str]] = set()
#: Last bounded metadata per ``(site, kind)``: scalars/strings only, never a tensor.
_LAST: dict[tuple[str, str], dict[str, Any]] = {}


def _bucket(site: str) -> dict[str, int]:
    return _COUNTERS.setdefault(site, {})


def summarise(tensor: Any) -> dict[str, int]:
    """``{nan, inf, nonfinite, elements}`` of ``tensor`` in one host sync (no copy)."""
    import torch

    nan, inf = torch.stack(
        [torch.count_nonzero(torch.isnan(tensor)), torch.count_nonzero(torch.isinf(tensor))]
    ).tolist()
    return {
        "nan": int(nan),
        "inf": int(inf),
        "nonfinite": int(nan) + int(inf),
        "elements": int(tensor.numel()),
    }


def record(
    site: str,
    kind: str,
    *,
    nonfinite: int,
    elements: int,
    before_dtype: Any = None,
    after_dtype: Any = None,
    shape: Any = None,
    identity: Optional[str] = None,
    detail: str = "",
) -> bool:
    """Count one non-finite event; warn the first time ``(site, kind)`` is seen.

    Returns whether this call emitted the warning (used by tests).
    """
    if nonfinite <= 0:
        return False
    bucket = _bucket(site)
    bucket[kind] = int(bucket.get(kind, 0) or 0) + int(nonfinite)
    counts = _ELEMENTS.setdefault(site, {})
    counts[kind] = int(counts.get(kind, 0) or 0) + 1
    # Bounded provenance for the artifact: the last metadata seen per (site, kind),
    # scalars/strings only (never the tensor, and JSON-serialisable).
    _LAST[(site, kind)] = {
        "nonfinite": int(nonfinite),
        "elements": int(elements or 0),
        "before_dtype": None if before_dtype is None else str(before_dtype),
        "after_dtype": None if after_dtype is None else str(after_dtype),
        "shape": None if shape is None else [int(dim) for dim in shape],
        "identity": identity,
        "detail": detail,
    }
    if (site, kind) in _LOGGED:
        return False
    _LOGGED.add((site, kind))
    logger.warning(
        "non-finite observation [site=%s kind=%s]: %d/%s element(s), dtype %s -> %s, "
        "shape=%s, identity=%s%s. Behaviour unchanged (observation only; no rejection, "
        "no collective).",
        site,
        kind,
        int(nonfinite),
        elements,
        before_dtype,
        after_dtype,
        shape,
        identity or "<none>",
        f", {detail}" if detail else "",
    )
    return True


def snapshot() -> dict[str, Any]:
    """Functional read of the counters (for tests and artifacts).

    ``sites[site][kind]`` is the number of non-finite **elements**; ``events[site][kind]``
    is the number of *tensors* that carried them. Scope: this process only.
    """
    last: dict[str, dict[str, dict[str, Any]]] = {}
    for (site, kind), entry in _LAST.items():
        last.setdefault(site, {})[kind] = dict(entry)
    return {
        "scope": "local-process counters; observation only; identity is bounded metadata",
        "sites": {site: dict(kinds) for site, kinds in _COUNTERS.items()},
        "events": {site: dict(kinds) for site, kinds in _ELEMENTS.items()},
        "last": last,
    }


def reset() -> None:
    """Clear all counters and the one-shot warning memory (tests only)."""
    _COUNTERS.clear()
    _LAST.clear()
    _ELEMENTS.clear()
    _LOGGED.clear()


__all__ = [
    "KIND_CAST_OVERFLOW",
    "KIND_SOURCE_INF",
    "KIND_SOURCE_NAN",
    "SITE_LATENTS",
    "record",
    "reset",
    "snapshot",
    "summarise",
]
