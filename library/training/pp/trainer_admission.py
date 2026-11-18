"""Admission preflight for the wired trainer-level PP path.

The CLI/session entry asks two questions here:

1. *May this PP config be served by the trainer path at all?* -- delegated to
   :func:`library.training.pp.gate.require_pipeline_runtime` (family contract,
   schedule/microbatch/world-size/quantization) plus the trainer-wiring
   admission (:data:`~library.training.pp.gate.TRAINER_PP_WIRED_FAMILIES`;
   every unwired family keeps the historical ``pp_trainer_path_unavailable``
   refusal).
2. *Is every trainer-facing knob this session will encounter either wired or
   explicitly rejected?* -- PP v0's first wired release serves exactly one
   configuration shape (cached single bucket, plain LoRA, grad accum 1, fp16,
   standard MSE).  Unsupported knobs are refused by name here, at argument
   validation time, instead of being silently ignored at training time.

This module only *admits*; the single execution dispatch stays in
``train_session._run_training_session``, and the DDP-interior guards in
``bootstrap`` / ``loop`` keep refusing any PP run that ever reaches them.
"""

from __future__ import annotations

from typing import Any, Mapping

from library.training.pp.errors import (
    CODE_CONFIG_INVALID,
    CODE_TRAINER_PATH,
    PipelineParallelError,
)
from library.training.pp.gate import (
    family_runtime_backend,
    require_pipeline_runtime,
    trainer_pp_wired,
)
from library.models.family_registry import PIPELINE_PARALLEL_RUNTIME_BACKEND_1F1B

_TRAINER_WIRED_BACKEND = PIPELINE_PARALLEL_RUNTIME_BACKEND_1F1B


def _get(config: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _enabled(value: Any) -> bool:
    return str(value or "").strip().lower() not in {"", "0", "false", "none", "off", "no"}


def _reject(config: Mapping[str, Any] | object, key: str, active: bool, detail: str) -> None:
    if active:
        raise PipelineParallelError(
            CODE_CONFIG_INVALID,
            f"pipeline_parallel trainer path requires {key}: {detail} "
            "(refused at entry instead of silently ignoring the setting)",
        )


def _require_wired_trainer_family(
    config: Mapping[str, Any] | object,
    *,
    world_size: int | None,
    where: str,
):
    """The family-agnostic wired check (shared by every family admission)."""

    require_pipeline_runtime(config, world_size=world_size)
    family, backend = family_runtime_backend(config)
    if not trainer_pp_wired(config):
        raise PipelineParallelError(
            CODE_TRAINER_PATH,
            f"{where}: model_family={family!r} has no wired trainer-level PP path "
            f"(runtime_backend={backend!r}, wired backend={_TRAINER_WIRED_BACKEND!r}). "
            "The accelerate data-parallel trainer cannot serve pipeline_parallel; "
            "Refusing to fall back to DDP.",
        )


def _reject_common_trainer_knobs(config: Mapping[str, Any] | object) -> None:
    """The family-neutral trainer knob refusal table (MOSS v0 table, verbatim)."""

    _reject(
        config,
        "gradient_accumulation_steps=1",
        int(_get(config, "gradient_accumulation_steps", 1) or 1) != 1,
        "the PP schedule owns the microbatch split and the 1/n factor; extra "
        "accumulation on top is not wired in the first release",
    )
    from library.training.train_bootstrap import sample_preview_enabled

    _reject(
        config,
        "no training preview",
        bool(_get(config, "sample_prompts", None)) or sample_preview_enabled(config),
        "sample-prompt preview decodes through the whole model on one rank",
    )
    # The PP trainer path serves the *cached* configuration the shared session
    # already prepared; a non-cached run would live-encode through encoders the
    # PP stages do not own.
    _reject(
        config,
        "cache_latents=true",
        not _enabled(_get(config, "cache_latents", None))
        and not _enabled(_get(config, "use_vae_cache", True)),
        "the PP stages consume cached latents (the session refuses incomplete "
        "caches); live VAE encoding is not wired",
    )
    _reject(
        config,
        "cache_text_encoder_outputs=true",
        not _enabled(_get(config, "cache_text_encoder_outputs", None))
        and not _enabled(_get(config, "use_text_cache", True)),
        "the PP first stage consumes cached text-encoder outputs; live TE "
        "encoding is not wired",
    )
    # Standard MSE on the flow-matching target is the only loss the schedule
    # owner implements; anything else would need a second loss path.
    _reject(
        config,
        "loss_type=l2",
        str(_get(config, "loss_type", "l2") or "l2").lower() != "l2",
        "the PP last-stage loss is the standard flow-matching MSE",
    )
    _reject(
        config,
        "weighting_scheme=none",
        str(_get(config, "weighting_scheme", "none") or "none").lower() != "none",
        "timestep weighting is not wired into the PP loss",
    )
    _reject(
        config,
        "mixed_precision=fp16",
        str(_get(config, "mixed_precision", "") or "").lower() != "fp16",
        "PP v0 is plain fp16 + GradScaler only (bf16 has no scaler, fp32 is "
        "not a supported PP configuration)",
    )
    from library.training.stage_schedule import stage_schedule_enabled

    stage_active = bool(_get(config, "stage_schedule_enabled", False)) or stage_schedule_enabled(config)
    _reject(
        config,
        "no stage_schedule",
        stage_active,
        "mid-run stage switches rebuild the dataloader outside the PP session",
    )
    _reject(
        config,
        "adaptive_oom_retry off",
        bool(_get(config, "adaptive_oom_retry", False)),
        "the adaptive OOM supervisor re-enters the DDP session",
    )
    for key in ("save_n_epoch_ratio", "save_every_n_epochs"):
        _reject(
            config,
            f"{key} unset",
            (_get(config, key, None) or 0) not in (None, 0),
            "the PP trainer path saves PP resume bundles by step and one final "
            "LoRA; epoch-based save schedules are not wired",
        )
    _reject(
        config,
        "persistent_data_loader_workers off",
        bool(_get(config, "persistent_data_loader_workers", False)),
        "the PP loader rebuilds its per-epoch generator; persistent workers would "
        "pin a different shuffle contract than the ordinary path",
    )
    for key in ("torch_compile",):
        _reject(config, f"{key} disabled", bool(_get(config, key, False)),
                f"{key} is not wired into the PP trainer path")


def require_moss_trainer_admission(
    config: Mapping[str, Any] | object,
    *,
    world_size: int | None = None,
    where: str = "trainer-admission",
):
    """Admit a legal MOSS PP config to the wired trainer path, or fail closed.

    Returns the validated :class:`PipelineRuntimeConfig`.  A PP config for a
    family without a wired trainer path raises ``pp_trainer_path_unavailable``
    (the historical code, unchanged); a legal MOSS config with an unsupported
    trainer knob raises ``pp_config_invalid`` naming the knob.
    """

    _require_wired_trainer_family(config, world_size=world_size, where=where)
    _reject_common_trainer_knobs(config)
    return require_pipeline_runtime(config, world_size=world_size)


def require_anima_trainer_admission(
    config: Mapping[str, Any] | object,
    *,
    world_size: int | None = None,
    where: str = "trainer-admission",
):
    """Admit a legal Anima PP config to the wired trainer path, or fail closed.

    The shared MOSS-v0 table applies unchanged, plus the Anima-specific named
    refusals from the frozen contract §3: compile bands, FEI/router
    conditioning, method adapters, caption dropout, ip-noise, text-encoder
    training.  Nothing is silently ignored; the single token-bucket hard
    constraint is enforced by the runtime's ``pp_token_bucket_mismatch``.
    """

    _require_wired_trainer_family(config, world_size=world_size, where=where)
    _reject_common_trainer_knobs(config)
    _reject(
        config,
        "compile_seq_bands / compile_dynamic_seq disabled",
        bool(_get(config, "compile_seq_bands", False))
        or bool(_get(config, "compile_dynamic_seq", False)),
        "per-band compile is not wired inside a PP stage (PP v0 compiles "
        "nothing on the pipeline path)",
    )
    router_source = str(_get(config, "router_source", "none") or "none").lower()
    _reject(
        config,
        "router_source='none'",
        router_source != "none",
        "FEI/sigma/crossattn routing state is per-step trainer state the PP "
        "stages do not share",
    )
    use_moe_style = _get(config, "use_moe_style", False)
    _reject(
        config,
        "use_moe_style=False",
        bool(use_moe_style) or str(use_moe_style or "").lower() not in {"", "false", "0", "none", "off", "no"},
        "MoE-style routing weights are not wired across PP stage boundaries",
    )
    for key in ("use_ip_adapter", "use_easycontrol", "use_byg"):
        _reject(
            config,
            f"{key} off",
            bool(_get(config, key, False)),
            "method adapters attach through on_network_built setup that the PP "
            "dispatch (after the fork point) does not cover",
        )
    _reject(
        config,
        "caption_dropout_rate=0",
        float(_get(config, "caption_dropout_rate", 0.0) or 0.0) > 0.0,
        "conditional/unconditional mixing needs the uncond crossattn cache "
        "and a per-stage dropout contract not wired under PP",
    )
    _reject(
        config,
        "ip_noise_gamma off",
        bool(_get(config, "ip_noise_gamma", False)),
        "input-perturbation noise changes the noising arithmetic the last "
        "stage reconstructs from the broadcast payload",
    )
    _reject(
        config,
        "network_train_unet_only=true",
        not bool(_get(config, "network_train_unet_only", False)),
        "the text encoder is not a PP stage; PP v0 trains only the DiT LoRA",
    )
    return require_pipeline_runtime(config, world_size=world_size)


def require_krea2_trainer_admission(
    config: Mapping[str, Any] | object,
    *,
    world_size: int | None = None,
    where: str = "trainer-admission",
):
    """Admit a legal Krea-2 PP config to the wired trainer path, or fail closed.

    Krea-2 has no FEI/router and no per-band compile, so the Anima-only refusals
    (compile bands, router_source, use_moe_style) do not apply.  The shared v0
    table plus the refusals that DO change the noising arithmetic still apply.
    """

    _require_wired_trainer_family(config, world_size=world_size, where=where)
    _reject_common_trainer_knobs(config)
    from .krea2_batch import batch_divisibility_message

    message = batch_divisibility_message(
        int(_get(config, "train_batch_size", 1) or 1),
        int(_get(config, "pipeline_parallel_microbatches", 4) or 4),
    )
    if message:
        raise PipelineParallelError("pp_krea2_batch_not_divisible", message)
    for key in ("use_ip_adapter", "use_easycontrol", "use_byg"):
        _reject(
            config,
            f"{key} off",
            bool(_get(config, key, False)),
            "method adapters attach through on_network_built setup that the PP "
            "dispatch (after the fork point) does not cover",
        )
    _reject(
        config,
        "caption_dropout_rate=0",
        float(_get(config, "caption_dropout_rate", 0.0) or 0.0) > 0.0,
        "conditional/unconditional mixing needs the uncond text cache and a "
        "per-stage dropout contract not wired under PP",
    )
    _reject(
        config,
        "ip_noise_gamma off",
        bool(_get(config, "ip_noise_gamma", False)),
        "input-perturbation noise changes the noising arithmetic the last "
        "stage reconstructs from the broadcast payload",
    )
    _reject(
        config,
        "network_train_unet_only=true",
        not bool(_get(config, "network_train_unet_only", False)),
        "the text encoder is not a PP stage; PP v0 trains only the DiT LoRA",
    )
    return require_pipeline_runtime(config, world_size=world_size)


def require_family_trainer_admission(
    config: Mapping[str, Any] | object,
    *,
    world_size: int | None = None,
    where: str = "trainer-admission",
):
    """Dispatch the family admission from the adapter table (single entry)."""

    from .trainer_adapters import resolve_trainer_adapter

    return resolve_trainer_adapter(config).admission(
        config, world_size=world_size, where=where
    )


__all__ = [
    "require_anima_trainer_admission",
    "require_family_trainer_admission",
    "require_moss_trainer_admission",
]
