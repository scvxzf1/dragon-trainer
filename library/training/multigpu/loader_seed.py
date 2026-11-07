"""Opt-in deterministic dataloader seeding for cross-world-size comparison.

Contract §13 V-2.4 (S-0) needs *the same sample in the same slot* in the single
and the dual run, otherwise Δθ / S-d / S-g compare different data and every number
built on them is meaningless.

accelerate 1.13.0 does not give that by default. In its multi-process branch
(``accelerate/data_loader.py``) a loader whose sampler carries a generator is
re-seeded from the **global** RNG and shuffled again, while the single-process
branch shuffles once with the global RNG; two runs with the same ``seed``
therefore produce different permutations (t3 F-T3-1a measured 3 of 12 samples
sharing a slot).

``DataLoaderConfiguration(use_seedable_sampler=True, data_seed=<seed>)`` makes both
branches shuffle with the same derived seed, and the measurement showed the single
and dual streams becoming identical.

This is an **explicit opt-in measurement gate**, exactly like
``ANIMA_MULTIGPU_NOISE_FIXTURE``: unset, no ``DataLoaderConfiguration`` is passed
and the training path keeps today's behaviour bit for bit. It is not a production
default and it changes the sample order of whatever run enables it - so it belongs
to a validation run whose reference numbers were produced with the same gate.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

ENV = "ANIMA_MULTIGPU_SEEDABLE_SAMPLER"
#: Optional explicit seed for the sampler; defaults to ``args.seed``.
SEED_ENV = "ANIMA_MULTIGPU_SEEDABLE_SAMPLER_SEED"

CODE_LOADER_SEED_INVALID = "multigpu_seedable_sampler_invalid"

_TRUE = {"1", "true", "yes", "on"}


class LoaderSeedError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def enabled(args: Any = None, env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the opt-in gate is open (env var first, then the CLI flag)."""
    values = dict(os.environ if env is None else env)
    if str(values.get(ENV, "")).strip().lower() in _TRUE:
        return True
    return bool(getattr(args, "multigpu_seedable_sampler", False))


def data_seed(args: Any = None, env: Optional[Mapping[str, str]] = None) -> int:
    """The seed the sampler is derived from: ``$SEED_ENV`` else ``args.seed``."""
    values = dict(os.environ if env is None else env)
    raw = values.get(SEED_ENV)
    if raw is not None and str(raw).strip():
        try:
            return int(str(raw).strip())
        except ValueError as exc:
            raise LoaderSeedError(
                CODE_LOADER_SEED_INVALID, f"{SEED_ENV}={raw!r} is not an integer"
            ) from exc
    seed = getattr(args, "seed", None)
    if seed is None:
        raise LoaderSeedError(
            CODE_LOADER_SEED_INVALID,
            "the seedable-sampler gate is open but no seed is available; pass "
            f"--seed or {SEED_ENV}",
        )
    try:
        return int(seed)
    except (TypeError, ValueError) as exc:
        raise LoaderSeedError(
            CODE_LOADER_SEED_INVALID, f"--seed={seed!r} is not an integer"
        ) from exc


def dataloader_config_kwargs(
    args: Any = None, env: Optional[Mapping[str, str]] = None
) -> dict[str, Any]:
    """``DataLoaderConfiguration`` fields for this run (empty when the gate is off).

    Returned as a plain dict so the caller merges it with any other loader
    configuration (for example ``use_stateful_dataloader``) instead of one option
    silently replacing the other.
    """
    if not enabled(args, env):
        return {}
    return {"use_seedable_sampler": True, "data_seed": data_seed(args, env)}


def describe(args: Any = None, env: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """Facts for the run identity / evidence files."""
    is_on = enabled(args, env)
    return {
        "seedable_sampler": is_on,
        "data_seed": data_seed(args, env) if is_on else None,
        "env": ENV,
    }
