"""Runtime contract for the Anima pipeline-parallel v0 runtime.

Everything in this module is fail-closed.  ``pipeline_parallel`` is either fully
available with the exact configuration validated here, or the caller gets a
:class:`~library.training.pp.errors.PipelineParallelError` with a stable code.
There is deliberately no "PP is unavailable, fall back to DDP" branch anywhere
in :mod:`library.training.pp`: a silent fallback changes the effective topology
without changing the checkpoint contract, which is exactly the failure mode the
PP v0 work exists to remove.

Scope of PP v0 (deliberate, not incidental):

* 2 stages, one process per stage (``world_size == stages``);
* ``Schedule1F1B`` (and ``ScheduleGPipe`` for the single-microbatch case);
* plain ``mixed_precision = "fp16"`` plus a ``GradScaler`` -- no NF4, no fp8, no
  ConvRot, no bf16 (bf16 has no scaler, so the fp16 transaction protocol in
  :mod:`library.training.pp.transactions` would not apply).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from library.anima.mini_forward import resolve_use_fp32
from library.models.pipeline_parallel import (
    MAX_PIPELINE_PARALLEL_MICROBATCHES,
    validate_pipeline_parallel_config,
)

from .errors import (
    CODE_BACKEND_MULTIBACKEND_REQUIRED,
    CODE_COMPUTE_POLICY_MISMATCH,
    CODE_CONDITIONING_UNSUPPORTED,
    CODE_CONFIG_INVALID,
    CODE_GRAD_SCALER_MISSING,
    CODE_MICROBATCHES_BELOW_STAGES,
    CODE_MICROBATCHES_INVALID,
    CODE_FP16_MASTER_WEIGHTS_REQUIRED,
    CODE_MIXED_PRECISION_UNSUPPORTED,
    CODE_NOT_ENABLED,
    CODE_QUANTIZATION_UNSUPPORTED,
    CODE_RUNTIME_UNAVAILABLE,
    CODE_SCHEDULE_UNSUPPORTED,
    CODE_WORLD_SIZE_MISMATCH,
    PipelineParallelError,
)

#: Schedules the PP v0 runtime can build.  ``1f1b`` is the requested default;
#: ``gpipe`` exists because torch 2.10 refuses 1F1B below ``num_stages``
#: microbatches, so the single-microbatch case has to be named for what it is.
PP_SUPPORTED_SCHEDULES = ("1f1b", "gpipe")
#: How the frozen conditioning reaches the later stage.  See ``stages.py``.
PP_CONDITIONING_MODES = ("local", "wire")
#: PP v0 is fp16-only (plain, with a ``GradScaler``).
PP_SUPPORTED_MIXED_PRECISION = ("fp16",)

#: Config/arg markers that mean "a quantized base is in play".
_QUANTIZATION_FLAG_KEYS = ("nf4", "use_nf4", "fp8", "use_fp8", "quantize_base")
_QUANTIZATION_PATH_KEYS = ("nf4_prequantized_path", "convrot_prequant_path")

#: ``base_compute`` is the runtime's *own* switch for a quantized base
#: (``library.runtime.convrot.checks.normalize_base_compute``): ``bf16`` is the
#: unquantized default and the aliases ``fp16``/``none``/``off`` all normalize to
#: it, so only the values returned by :func:`_normalized_base_compute` other than
#: ``bf16`` mean quantization.  The alias table lives in the runtime authority;
#: this module never keeps a second copy of it.
_BASE_COMPUTE_UNQUANTIZED = "bf16"


def _get(config: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


@dataclass(frozen=True)
class PipelineRuntimeConfig:
    """Normalized PP v0 runtime settings.

    ``stages`` / ``microbatches`` / ``schedule`` come from the existing
    ``pipeline_parallel_*`` config keys (the same ones the family contract and
    the WebUI preflight already read).  ``conditioning`` is a runtime choice, not
    a config key, because it is a transport detail rather than a user-facing
    capability.

    ``conditioning`` is not a capability switch
    -------------------------------------------
    ``"local"`` is the default **and the only production path**: each stage
    rebuilds the frozen modulation / RoPE products from ``timesteps`` and the
    latent geometry, so nothing that can never carry a useful gradient is put on
    the wire.

    ``"wire"`` exists **only** as an experiment control / debugging aid -- it is
    used by the equivalence harness to prove that the two conditioning modes agree
    bit-for-bit, and to reproduce the older "frozen conditioning crosses the
    boundary, its gradient is dropped on the receiving side" topology when
    comparing against it.  It is **not** a production path: it doubles the
    boundary payload, it exists to keep a comparison honest, and nothing in the
    trainer selects it.  Do not make it reachable from a config key.

    The allowed values are enforced by :func:`validate_runtime_config`
    (``CODE_CONDITIONING_UNSUPPORTED``) and by
    ``library.training.pp.stages.frozen_boundary_indices``; an unknown value
    fails closed instead of silently degrading to ``local``.
    """

    stages: int = 2
    microbatches: int = 2
    schedule: str = "1f1b"
    #: ``"local"`` (production default) or ``"wire"`` (experiment control only --
    #: see the class docstring; never reachable from a config key).
    conditioning: str = "local"
    mixed_precision: str = "fp16"

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | object) -> "PipelineRuntimeConfig":
        family_config = validate_pipeline_parallel_config(config)
        return cls(
            stages=int(family_config.stages),
            microbatches=int(family_config.microbatches),
            schedule=str(family_config.schedule),
            conditioning="local",
            mixed_precision=str(_get(config, "mixed_precision", "fp16") or "fp16")
            .strip()
            .lower(),
        )

    def with_overrides(self, **overrides: Any) -> "PipelineRuntimeConfig":
        values = {
            "stages": self.stages,
            "microbatches": self.microbatches,
            "schedule": self.schedule,
            "conditioning": self.conditioning,
            "mixed_precision": self.mixed_precision,
        }
        unknown = set(overrides) - set(values)
        if unknown:
            raise TypeError(f"unknown PipelineRuntimeConfig fields: {sorted(unknown)}")
        values.update(overrides)
        return PipelineRuntimeConfig(**values)


def require_pipelining_api() -> tuple[Any, ...]:
    """Import the torch pipelining API, or fail closed with a named code.

    A missing/short ``torch.distributed.pipelining`` is the one case where the
    historical trainer would silently keep running as DDP.  It raises here
    instead.
    """

    try:
        from torch.distributed.pipelining import (
            PipelineStage,
            Schedule1F1B,
            ScheduleGPipe,
        )
    except (ImportError, AttributeError) as exc:  # pragma: no cover - env dependent
        raise PipelineParallelError(
            CODE_RUNTIME_UNAVAILABLE,
            "torch.distributed.pipelining is not importable in this environment "
            f"({exc}); pipeline_parallel cannot be served and PP v0 will not fall "
            "back to DDP",
        ) from exc
    return PipelineStage, Schedule1F1B, ScheduleGPipe


def validate_runtime_config(
    config: PipelineRuntimeConfig,
    *,
    world_size: int | None = None,
    require_enabled: bool = True,
) -> PipelineRuntimeConfig:
    """Validate the PP v0 runtime contract for this process."""

    if require_enabled:
        require_pipelining_api()
    if config.schedule not in PP_SUPPORTED_SCHEDULES:
        raise PipelineParallelError(
            CODE_SCHEDULE_UNSUPPORTED,
            f"pipeline_parallel_schedule must be one of "
            f"{', '.join(PP_SUPPORTED_SCHEDULES)} for PP v0, got "
            f"{config.schedule!r}",
        )
    if config.conditioning not in PP_CONDITIONING_MODES:
        raise PipelineParallelError(
            CODE_CONDITIONING_UNSUPPORTED,
            f"conditioning must be one of {', '.join(PP_CONDITIONING_MODES)}, got "
            f"{config.conditioning!r}",
        )
    if not 1 <= int(config.microbatches) <= MAX_PIPELINE_PARALLEL_MICROBATCHES:
        raise PipelineParallelError(
            CODE_MICROBATCHES_INVALID,
            "pipeline_parallel_microbatches must be between 1 and "
            f"{MAX_PIPELINE_PARALLEL_MICROBATCHES}, got {config.microbatches}",
        )
    if int(config.microbatches) < int(config.stages):
        raise PipelineParallelError(
            CODE_MICROBATCHES_BELOW_STAGES,
            f"pipeline_parallel_microbatches must be >= stages ({config.stages}), "
            f"got {config.microbatches}. In torch 2.10 *both* "
            "PipelineScheduleSingle schedules (Schedule1F1B and ScheduleGPipe) "
            "raise this from __init__, so microbatches=1 is not a GPipe escape "
            "hatch for a 2-stage pipeline. Note that with 2 stages only "
            "microbatches >= 3 actually departs from the GPipe ordering.",
        )
    if world_size is not None and int(world_size) != int(config.stages):
        raise PipelineParallelError(
            CODE_WORLD_SIZE_MISMATCH,
            "pipeline_parallel_stages must equal the distributed world size: "
            f"stages={config.stages}, world_size={world_size}",
        )
    return config


def validate_mixed_precision(
    config: PipelineRuntimeConfig,
    *,
    has_scaler: bool,
) -> None:
    """PP v0 is plain fp16 plus a ``GradScaler``; everything else fails closed."""

    if config.mixed_precision not in PP_SUPPORTED_MIXED_PRECISION:
        raise PipelineParallelError(
            CODE_MIXED_PRECISION_UNSUPPORTED,
            "pipeline_parallel supports mixed_precision=fp16 only (plain fp16 + "
            f"GradScaler); got mixed_precision={config.mixed_precision!r}. bf16 has "
            "no scaler, so the symmetric skip protocol cannot apply, and fp32 is "
            "not a supported PP v0 configuration.",
        )
    if not has_scaler:
        raise PipelineParallelError(
            CODE_GRAD_SCALER_MISSING,
            "mixed_precision=fp16 under pipeline_parallel requires a GradScaler; "
            "without one fp16 gradients underflow silently and the two stages "
            "cannot agree on a skip decision",
        )


def _normalized_base_compute(config: Mapping[str, Any] | object) -> str:
    """``base_compute`` through the runtime authority, or fail closed.

    ``normalize_base_compute`` owns the alias table (``w8a16`` ==
    ``w8a16_convrot``, ``fp16``/``none``/``off`` == ``bf16``, ``nf4`` ...).  An
    unknown value is *not* treated as unquantized: it is returned verbatim so
    :func:`quantization_markers` can still report it and the caller keeps failing
    closed.  The runtime raises on such a value later anyway.
    """

    raw = _get(config, "base_compute", _BASE_COMPUTE_UNQUANTIZED)
    try:
        from library.runtime.convrot.checks import normalize_base_compute

        return normalize_base_compute(raw)
    except (ImportError, ValueError):
        return str(raw).strip().lower()


def convrot_selected(config: Mapping[str, Any] | object) -> bool:
    """Is ConvRot actually switched on for this config?

    The single source of truth for "``convrot_group_size`` means something":
    ``library/training/bootstrap.py::maybe_apply_convrot_base`` returns ``False``
    unless ``base_compute`` normalizes to a ConvRot mode, so the group size is a
    tuning parameter (CLI default 256) in every other configuration.
    """

    return _normalized_base_compute(config) in {"w8a16_convrot", "w8a8_convrot"}


def quantization_markers(config: Mapping[str, Any] | object) -> list[str]:
    """Quantization switches that PP v0 refuses, as ``key=value`` strings.

    This is the only implementation of the rule -- ``gate``, ``runtime`` and the
    compatibility layer all call it, so "is this a quantized base?" cannot answer
    two different ways.

    ``convrot_group_size`` is special: ``--convrot_group_size`` has an
    ``argparse`` default of 256 (``library/training/cli_args.py``), i.e. *every*
    CLI configuration carries a non-zero value.  Counting it unconditionally
    rejected every default run as ``pp_quantization_unsupported``.  It therefore
    only counts when :func:`convrot_selected` is true; the quantized
    ``base_compute`` itself is reported instead, which also closes the previous
    hole where ``base_compute=nf4`` was not detected at all.
    """

    markers: list[str] = []
    for key in _QUANTIZATION_FLAG_KEYS:
        value = _get(config, key, False)
        if value:
            markers.append(f"{key}={value!r}")
    for key in _QUANTIZATION_PATH_KEYS:
        value = _get(config, key, None)
        if value:
            markers.append(f"{key}={value!r}")
    base_compute = _normalized_base_compute(config)
    if base_compute != _BASE_COMPUTE_UNQUANTIZED:
        markers.append(f"base_compute={base_compute}")
    if convrot_selected(config):
        group_size = _get(config, "convrot_group_size", 0)
        try:
            if int(group_size or 0) > 0:
                markers.append(f"convrot_group_size={group_size}")
        except (TypeError, ValueError):
            markers.append(f"convrot_group_size={group_size!r}")
    return markers


def validate_no_quantization(config: Mapping[str, Any] | object) -> None:
    """Reject NF4 / fp8 / ConvRot bases: PP v0 is plain fp16 only."""

    markers = quantization_markers(config)
    if markers:
        raise PipelineParallelError(
            CODE_QUANTIZATION_UNSUPPORTED,
            "pipeline_parallel does not support a quantized base (PP v0 is plain "
            "fp16); found " + ", ".join(markers),
        )


def require_enabled(config: Mapping[str, Any] | object) -> None:
    """Raise unless ``pipeline_parallel`` is switched on in this config."""

    enabled = _get(config, "pipeline_parallel", False)
    if not enabled:
        raise PipelineParallelError(
            CODE_NOT_ENABLED,
            "pipeline_parallel is disabled in this config; the PP runtime must "
            "not be constructed (and must never be used as a DDP wrapper)",
        )


def validate_family_config(
    config: Mapping[str, Any] | object,
    *,
    world_size: int | None = None,
    num_blocks: int | None = None,
) -> None:
    """Run the family-level PP contract, re-raising with a named PP code."""

    try:
        validate_pipeline_parallel_config(
            config, world_size=world_size, num_blocks=num_blocks
        )
    except ValueError as exc:
        raise PipelineParallelError(CODE_CONFIG_INVALID, str(exc)) from exc


# ---------------------------------------------------------------------------
# pipeline-level compute policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PpComputePolicy:
    """The pipeline's single compute-dtype decision.

    ``use_fp32`` is not a per-stage tuning knob.  The single-GPU authority derives
    it once from the **pre-block hidden** dtype (``models.py:2665``) and applies
    the same value to all blocks and to the final layer.  When it is true the
    block stack promotes the residual stream (``models.py:1519``), so the tensor
    that crosses the PP boundary is **fp32 even though the stage is running fp16**.

    A stage that re-derived the flag from the tensor it *received* therefore
    computed ``False`` and ran the tail in fp16: T8 measured ``8.665e4 > 65504``
    -> non-finite, first bad block ``idx=32``.  The flag is resolved exactly once,
    from :attr:`activation_dtype` (the pre-block hidden dtype), and carried to
    every stage; :func:`library.training.pp.runtime.build_anima_pipeline_runtime`
    asserts cross-rank agreement before the first step.
    """

    #: dtype of the pre-block hidden tensor (the authority for ``use_fp32``)
    activation_dtype: torch.dtype
    #: ``resolve_use_fp32(activation_dtype)``, evaluated once
    use_fp32: bool

    @classmethod
    def from_dtype(cls, dtype: torch.dtype) -> "PpComputePolicy":
        """Resolve the policy from the pre-block compute dtype."""

        resolved = torch.empty(0).to(dtype).dtype
        return cls(activation_dtype=resolved, use_fp32=resolve_use_fp32(resolved))

    @property
    def boundary_dtype(self) -> torch.dtype:
        """dtype of the inter-stage activation (the promoted residual stream)."""

        return torch.float32 if self.use_fp32 else self.activation_dtype

    @property
    def prediction_dtype(self) -> torch.dtype:
        """dtype of the last stage's prediction (the final layer casts back)."""

        return self.activation_dtype

    def as_dict(self) -> dict[str, Any]:
        return {
            "activation_dtype": str(self.activation_dtype),
            "use_fp32": bool(self.use_fp32),
            "boundary_dtype": str(self.boundary_dtype),
            "prediction_dtype": str(self.prediction_dtype),
        }


def assert_compute_policy_agreement(
    policy: PpComputePolicy,
    *,
    group: Any = None,
    where: str = "pp_build",
) -> None:
    """Every rank must resolve the same ``use_fp32`` before the first step.

    A split decision is not self-correcting: the stage that resolves ``False``
    runs its blocks in fp16 and produces ``inf``/``NaN`` (D1), while the other
    stage keeps running, so the failure shows up as a silent numerical
    corruption rather than an error.
    """

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    payload = policy.as_dict()
    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, payload, group=group)
    if any(item != payload for item in gathered):
        raise PipelineParallelError(
            CODE_COMPUTE_POLICY_MISMATCH,
            f"the stages resolved different compute policies at {where}: {gathered}. "
            "use_fp32 must be decided once from the pre-block dtype and shared; a "
            "stage that re-derives it from the boundary tensor gets fp32 -> False "
            "and runs fp16 blocks that overflow.",
        )


# ---------------------------------------------------------------------------
# fp16 master weights (D4)
# ---------------------------------------------------------------------------


def validate_master_weight_dtype(
    parameters: Iterable[Any],
    *,
    mixed_precision: str | None = None,
) -> None:
    """fp16 PP needs fp32 master weights (``GradScaler`` refuses fp16 grads).

    ``GradScaler.unscale_``/``step`` raise ``ValueError: Attempting to unscale
    FP16 gradients``, and the PP fp16 transaction *divides* gradients by hand on
    every non-last stage, which silently underflows fp16.  A fp16 PP run is
    therefore only meaningful with fp32 adapter master weights and an autocast
    forward; the base model may stay fp16/bf16 because it carries no gradient.
    """

    if mixed_precision != "fp16":
        return
    offenders = sorted(
        {
            str(getattr(parameter, "dtype", "?"))
            for parameter in parameters
            if getattr(parameter, "requires_grad", False)
            and getattr(parameter, "dtype", None) != torch.float32
        }
    )
    if offenders:
        raise PipelineParallelError(
            CODE_FP16_MASTER_WEIGHTS_REQUIRED,
            f"mixed_precision=fp16 under pipeline_parallel requires fp32 master "
            f"weights for every trainable parameter; found {', '.join(offenders)}. "
            "GradScaler refuses fp16 gradients and the non-last stages divide "
            "gradients by the shared scale by hand, which underflows in fp16.",
        )


# ---------------------------------------------------------------------------
# process-group backend (D3)
# ---------------------------------------------------------------------------


def validate_distributed_backend(
    backend: str | None,
    *,
    device_type: str,
    world_size: int = 2,
) -> None:
    """CUDA PP needs a multi-backend process group: ``cpu:gloo,cuda:nccl``.

    The PP runtime moves *device* activations between stages (NCCL) but also
    exchanges **CPU** scalars for the fp16 scale/skip decision and the boundary
    agreement checks (:mod:`library.training.pp.transactions`).  A single-backend
    group cannot serve both: Gloo on CUDA P2P fails with
    ``gloo::IoException writev Bad address``, and NCCL rejects CPU tensors.  The
    requirement is a property of the runtime, so it is validated here rather than
    left to a confusing runtime abort.
    """

    if str(device_type) != "cuda":
        return
    if int(world_size) <= 1:
        return
    spec = str(backend or "")
    lowered = spec.lower()
    has_gloo = "gloo" in lowered
    has_nccl = "nccl" in lowered
    if has_gloo and has_nccl:
        return
    raise PipelineParallelError(
        CODE_BACKEND_MULTIBACKEND_REQUIRED,
        f"CUDA pipeline_parallel needs a multi-backend process group "
        f"(cpu:gloo,cuda:nccl); got backend={spec!r}. The PP runtime moves GPU "
        "activations between stages (needs NCCL) and CPU scalars for the fp16 "
        "scale / skip decision (needs Gloo): a Gloo-only group fails CUDA P2P "
        "with 'gloo::IoException writev Bad address', and an NCCL-only group "
        "cannot receive the CPU tensors.",
    )


__all__ = [
    "PP_CONDITIONING_MODES",
    "convrot_selected",
    "PP_SUPPORTED_MIXED_PRECISION",
    "PP_SUPPORTED_SCHEDULES",
    "PipelineRuntimeConfig",
    "PpComputePolicy",
    "assert_compute_policy_agreement",
    "quantization_markers",
    "require_enabled",
    "require_pipelining_api",
    "validate_distributed_backend",
    "validate_master_weight_dtype",
    "validate_family_config",
    "validate_mixed_precision",
    "validate_no_quantization",
    "validate_runtime_config",
]
