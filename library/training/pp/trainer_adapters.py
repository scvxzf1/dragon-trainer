"""Per-family trainer-level PP adapters (the §2 generalization of the MOSS wiring).

The MOSS trainer modules (``trainer_admission`` / ``trainer_inputs`` /
``trainer_session`` / ``trainer_resume``) were MOSS-specific in four places.
This module converges every family difference into one explicit adapter table
so the trainer modules stay family-blind:

* the runtime builder (Anima: ``pp.runtime.build_anima_pipeline_runtime`` with
  its 5D ``(B, C, 1, H, W)`` wire; MOSS: ``pp.moss_runtime`` with its 4D wire);
* the step-condition bundle (Anima: max-padded ``crossattn_emb`` from the disk
  text cache — padding positions are attention sinks and are **never** trimmed;
  MOSS: ``family._prepare_text_conditions`` hiddens/mask);
* the noise/timestep math (Anima: the *same* ``noise_target`` sampler-registry
  path the non-PP trainer uses — no second noise math; MOSS:
  ``sample_moss_flow``);
* the resume format stamp (``anima_pp_trainer_resume`` /
  ``moss_image_pp_trainer_resume``).

MOSS behaviour is frozen: its adapter delegates to the exact functions the
wired MOSS path used before the generalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .errors import (
    CODE_CONFIG_INVALID,
    CODE_STAGE_CONTRACT,
    PipelineParallelError,
)


def trainer_resume_format(family: str) -> str:
    """The per-family resume format stamp (contract §4)."""

    return f"{str(family)}_pp_trainer_resume"


def _family_of(args) -> str:
    from .gate import family_runtime_backend

    family, _backend = family_runtime_backend(args)
    return family


@dataclass(frozen=True)
class FlowSample:
    """What rank 0 sampled for one whole step, plus the tensors to share."""

    #: The rank-0 noisy input straight from the family's noise math. ``None``
    #: never crosses this structure: rank 0 always samples.
    noisy_input: Tensor
    #: Ordered mapping of tensors broadcast rank0 -> every other rank. The
    #: insertion order is part of the wire contract (both ranks iterate it).
    broadcast: dict[str, Tensor]


class MossTrainerAdapter:
    """The frozen MOSS adapter: every call is the pre-generalization code."""

    name = "moss_image"
    resume_format = trainer_resume_format("moss_image")
    family_label = "MOSS"
    #: MOSS conditions are prepared only on the first stage (frozen protocol).
    conditions_on_every_rank = False

    def admission(self, config, *, world_size=None, where="trainer-admission"):
        from .trainer_admission import require_moss_trainer_admission

        return require_moss_trainer_admission(config, world_size=world_size, where=where)

    def build_runtime(self, args, accelerator, network, unet, *, latent_shape, max_text_tokens, dtype=None):
        from .trainer_session import build_moss_trainer_runtime

        return build_moss_trainer_runtime(
            args, accelerator, network, unet,
            latent_shape=latent_shape, max_text_tokens=max_text_tokens, dtype=dtype,
        )

    def wire_latents(self, latents: Tensor) -> Tensor:
        return latents

    def wire_shape_from_batch(self, batch) -> tuple[tuple[int, ...], int]:
        """The frozen wire geometry from the first batch (cached single bucket)."""

        latents = batch.get("latents")
        if latents is None:
            raise PipelineParallelError(
                CODE_CONFIG_INVALID,
                "the PP trainer path serves the cached-latents configuration; the "
                "first batch has no latents",
            )
        conds = batch.get("text_encoder_outputs_list")
        if not conds or conds[0] is None:
            raise PipelineParallelError(
                CODE_CONFIG_INVALID,
                "the PP trainer path serves the cached text-encoder configuration; "
                "the first batch has no cached text_encoder_outputs_list",
            )
        return tuple(int(v) for v in latents.shape), int(conds[0].shape[1])

    def prepare_conditions(self, trainer, args, batch, *, accelerator, weight_dtype, text_encoding_strategy=None):
        from .trainer_inputs import prepare_cached_conditions

        hiddens, mask = prepare_cached_conditions(
            trainer, args, batch, accelerator=accelerator, weight_dtype=weight_dtype
        )
        return {"text_hiddens": hiddens, "text_mask": mask}

    def finalize_conditions(self, latents: Tensor, conditions: dict) -> dict:
        return conditions

    def sample_flow(self, args, trainer, latents: Tensor, *, accelerator, weight_dtype) -> FlowSample:
        from library.models.moss_image.family import sample_moss_flow

        noise, sigmas, noisy_input = sample_moss_flow(latents)
        return FlowSample(
            noisy_input=noisy_input,
            broadcast={"noise": noise, "sigmas": sigmas},
        )

    def receive_flow(self, latents: Tensor, broadcast: dict[str, Tensor], *, weight_dtype):
        """Rebuild ``(noisy_input, timesteps)`` on the last stage."""

        noise = broadcast["noise"]
        sigmas = broadcast["sigmas"]
        sigma_view = sigmas.to(latents.dtype).view(-1, 1, 1, 1)
        noisy_input = (1 - sigma_view) * latents + sigma_view * noise
        timesteps = sigmas.reshape(-1).float() * 1000.0
        return noisy_input, timesteps

    def timesteps_from_flow(self, broadcast: dict[str, Tensor]) -> Tensor:
        return broadcast["sigmas"].reshape(-1).float() * 1000.0

    def empty_broadcast(self, latents: Tensor, weight_dtype) -> dict[str, Tensor]:
        return {
            "noise": torch.empty(latents.shape, dtype=latents.dtype, device=latents.device),
            "sigmas": torch.empty((latents.shape[0],), dtype=torch.float32, device=latents.device),
        }

    def make_target(self, latents: Tensor, noise: Tensor) -> Tensor:
        return noise - latents

    def build_inputs(self, *, noisy_input, timesteps, target, conditions, digest):
        from .trainer_inputs import MossStepInputs

        return MossStepInputs(
            noisy_input=noisy_input,
            sigmas=conditions["sigmas"],
            timesteps=timesteps,
            target=target,
            text_hiddens=conditions.get("text_hiddens"),
            text_mask=conditions.get("text_mask"),
            batch_digest=digest,
        )


class AnimaTrainerAdapter:
    """The Anima trainer adapter: the library-level runtime wired at trainer level."""

    name = "anima"
    resume_format = trainer_resume_format("anima")
    family_label = "Anima"
    #: Local conditioning rebuilds the timestep/crossattn modulation on *every*
    #  stage, so each rank reads the cached conditions from its own batch.
    conditions_on_every_rank = True

    def admission(self, config, *, world_size=None, where="trainer-admission"):
        from .trainer_admission import require_anima_trainer_admission

        return require_anima_trainer_admission(config, world_size=world_size, where=where)

    def build_runtime(self, args, accelerator, network, unet, *, latent_shape, max_text_tokens, dtype=None):
        from .trainer_session import build_anima_trainer_runtime

        return build_anima_trainer_runtime(
            args, accelerator, network, unet,
            latent_shape=latent_shape, max_text_tokens=max_text_tokens, dtype=dtype,
        )

    def wire_latents(self, latents: Tensor) -> Tensor:
        """Anima's 5D wire: ``(B, C, H, W)`` cache latents gain the T=1 axis.

        The explicit ``unsqueeze(2)`` is the frozen 4D/5D boundary (AGENTS
        invariant): no bare ``squeeze()`` anywhere in the PP path.
        """

        if latents.ndim == 4:
            return latents.unsqueeze(2)
        if latents.ndim == 5:
            return latents
        raise PipelineParallelError(
            CODE_CONFIG_INVALID,
            f"the Anima PP wire consumes (B, C, H, W) cached latents or an "
            f"existing 5D (B, C, 1, H, W) wire; got ndim={latents.ndim}",
        )

    def wire_shape_from_batch(self, batch) -> tuple[tuple[int, ...], int]:
        latents = batch.get("latents")
        if latents is None:
            raise PipelineParallelError(
                CODE_CONFIG_INVALID,
                "the Anima PP trainer path serves the cached-latents "
                "configuration; the first batch has no latents",
            )
        wire = self.wire_latents(latents)
        conds = batch.get("text_encoder_outputs_list")
        if not conds:
            raise PipelineParallelError(
                CODE_CONFIG_INVALID,
                "the Anima PP trainer path serves the cached text-encoder "
                "configuration; the first batch has no cached "
                "text_encoder_outputs_list (rebuild the Anima text cache)",
            )
        if len(conds) < 5 or conds[4] is None:
            raise PipelineParallelError(
                CODE_CONFIG_INVALID,
                "the Anima PP first stage consumes the cached max-padded "
                "crossattn_emb; this batch's text cache has no cached "
                "crossattn_emb entry (live TE encoding is not wired)",
            )
        # (B, T, C): T is the max-padded token count, never trimmed.
        max_text_tokens = int(conds[4].shape[-2])
        return tuple(int(v) for v in wire.shape), max_text_tokens

    def prepare_conditions(self, trainer, args, batch, *, accelerator, weight_dtype, text_encoding_strategy=None):
        """The cached max-padded ``(crossattn_emb)`` bundle for this batch.

        Padding positions are cross-attention attention sinks (the Anima
        invariant): the cached tensor is passed to the wire **as-is** — no
        length-based trimming, no mask that removes padding.
        """

        conds = batch.get("text_encoder_outputs_list", None)
        if not conds:
            raise PipelineParallelError(
                CODE_CONFIG_INVALID,
                "the Anima PP trainer path serves the cached text-encoder "
                "configuration only; this batch arrived without cached "
                "text_encoder_outputs_list (rebuild the Anima text cache)",
            )
        rates = batch.get("caption_dropout_rates", None) if isinstance(batch, dict) else None
        if rates is not None and any(float(rate or 0.0) > 0.0 for rate in rates):
            raise PipelineParallelError(
                CODE_CONFIG_INVALID,
                "caption_dropout_rate > 0 is not wired into the Anima PP path "
                "(the unconditional crossattn_emb would need the uncond cache "
                "and a per-stage dropout contract); disable caption dropout",
            )
        if len(conds) < 5 or conds[4] is None:
            raise PipelineParallelError(
                CODE_CONFIG_INVALID,
                "the Anima PP first stage consumes the cached max-padded "
                "crossattn_emb; live TE encoding is not wired (rebuild the "
                "Anima text cache with crossattn_emb)",
            )
        crossattn_emb = conds[4].to(accelerator.device, dtype=weight_dtype)
        return {"crossattn_emb": crossattn_emb}

    def finalize_conditions(self, latents: Tensor, conditions: dict) -> dict:
        """Attach the all-zero padding mask (attention sinks, never trimmed)."""

        return {**conditions, "padding_mask": anima_padding_mask(latents)}

    def sample_flow(self, args, trainer, latents: Tensor, *, accelerator, weight_dtype) -> FlowSample:
        """Rank 0 samples noise/timesteps with the *non-PP* sampler registry.

        ``noise_target.compute_noise_pred_and_target`` draws ``randn_like`` and
        delegates to ``SAMPLER_REGISTRY``; this is exactly that math on the 5D
        wire latents (the default sampler's sigma view already handles 5D), so
        the PP branch never owns a second noise policy.
        """

        from library.training.samplers import SAMPLER_REGISTRY, SamplerContext

        # The noise math runs on the 4D cache latents exactly like the non-PP
        # path (``noise_target`` squeezes the T axis first); the wire tensor is
        # the explicit ``unsqueeze(2)`` of the same draw.
        latents_4d = latents.squeeze(2)
        noise = torch.randn_like(latents_4d)
        sampler_fn = SAMPLER_REGISTRY[getattr(args, "sampler", "default") or "default"]
        scheduler = trainer.get_noise_scheduler(args, accelerator.device)
        sampler_out = sampler_fn(
            SamplerContext(
                args=args,
                noise_scheduler=scheduler,
                latents=latents,
                noise=noise.unsqueeze(2),
                device=accelerator.device,
                weight_dtype=weight_dtype,
            )
        )
        # timesteps are the flat per-sample sigma (sigma in [0,1] *is* the DiT
        # time argument) — identical values on both ranks after broadcast.
        timesteps = sampler_out.timesteps.reshape(-1).float()
        return FlowSample(
            noisy_input=sampler_out.noisy_input.to(dtype=weight_dtype),
            # The broadcast payload carries the 4D noise; the wire and the
            # target derive from it with the frozen unsqueeze/squeeze boundary.
            broadcast={"noise": noise, "timesteps": timesteps},
        )

    def receive_flow(self, latents: Tensor, broadcast: dict[str, Tensor], *, weight_dtype):
        """Rebuild the noisy input on the last stage with the same arithmetic.

        ``get_noisy_model_input_and_timesteps`` computes
        ``(1 - sigma) * latents + sigma * noise`` (``ip_noise_gamma`` is a
        named admission refusal, so this reconstruction is exact).
        """

        noise = broadcast["noise"]
        timesteps = broadcast["timesteps"]
        sigmas = timesteps.to(latents.dtype).view(-1, 1, 1, 1, 1)
        noisy_input = ((1.0 - sigmas) * latents + sigmas * noise.unsqueeze(2)).to(weight_dtype)
        return noisy_input, timesteps

    def timesteps_from_flow(self, broadcast: dict[str, Tensor]) -> Tensor:
        return broadcast["timesteps"]

    def empty_broadcast(self, latents: Tensor, weight_dtype) -> dict[str, Tensor]:
        return {
            "noise": torch.empty(latents.squeeze(2).shape, dtype=latents.dtype, device=latents.device),
            "timesteps": torch.empty((latents.shape[0],), dtype=torch.float32, device=latents.device),
        }

    def make_target(self, latents: Tensor, noise: Tensor) -> Tensor:
        """The rectified-flow target on the 4D latents (noise_target line 404)."""

        return noise - latents.squeeze(2)

    def build_inputs(self, *, noisy_input, timesteps, target, conditions, digest):
        return AnimaStepInputs(
            noisy_input=noisy_input,
            timesteps=timesteps,
            target=target,
            crossattn_emb=conditions.get("crossattn_emb"),
            padding_mask=conditions.get("padding_mask"),
            batch_digest=digest,
        )


@dataclass(frozen=True)
class AnimaStepInputs:
    """One whole-step Anima PP training input bundle (already synchronized)."""

    noisy_input: Tensor
    timesteps: Tensor
    target: Tensor
    crossattn_emb: Tensor | None
    padding_mask: Tensor | None
    batch_digest: str

    @property
    def batch_size(self) -> int:
        return int(self.noisy_input.shape[0])

    def step_kwargs(self, runtime) -> dict:
        """The ``runtime.step`` kwargs for the Anima wire (local conditioning)."""

        return {
            "timesteps": self.timesteps,
            "crossattn_emb": self.crossattn_emb,
            "padding_mask": self.padding_mask,
        }

    def wire_for(self, runtime) -> Tensor | None:
        """The Anima runtime validates the wire on every rank; the last stage
        passes its reconstructed wire (``run_pipeline_step`` drops it)."""

        return self.noisy_input


def anima_padding_mask(latents: Tensor) -> Tensor:
    """The all-zero padding mask, cached like the non-PP ``_padding_mask_cache``.

    Padding positions are attention sinks: the mask is all zeros and its shape
    ``(B, 1, H, W)`` follows the cached-latent wire, not any real-length mask.
    """

    _bs, _c, _t, h, w = (int(v) for v in latents.shape)
    return torch.zeros((int(latents.shape[0]), 1, h, w), dtype=latents.dtype, device=latents.device)


ADAPTERS: dict[str, object] = {
    MossTrainerAdapter.name: MossTrainerAdapter(),
    AnimaTrainerAdapter.name: AnimaTrainerAdapter(),
}


def resolve_trainer_adapter(config) -> object:
    """The adapter for this config's canonical family, or a named refusal.

    Only families in ``TRAINER_PP_WIRED_FAMILIES`` resolve; every other family
    keeps the historical ``pp_trainer_path_unavailable`` refusal one step
    earlier (in the admission), so this lookup is a pure table read.
    """

    family = _family_of(config)
    adapter = ADAPTERS.get(family)
    if adapter is None:
        from .errors import CODE_TRAINER_PATH

        raise PipelineParallelError(
            CODE_TRAINER_PATH,
            f"model_family={family!r} has no trainer-level PP family adapter; "
            "Refusing to fall back to DDP.",
        )
    return adapter


__all__ = [
    "ADAPTERS",
    "AnimaStepInputs",
    "AnimaTrainerAdapter",
    "FlowSample",
    "MossTrainerAdapter",
    "anima_padding_mask",
    "resolve_trainer_adapter",
    "trainer_resume_format",
]
