"""Trainer-level MOSS PP step inputs: batch prep, flow sampling, rank sync.

Both stages must process the *same* logical batch, the same noise/sigmas and
the same text conditions.  This module is the only place that owns that
synchronisation:

* the cached latents and cached text-encoder outputs are prepared exactly like
  the non-PP path (``batch_step._prepare_latents`` for the NaN scrub + shift /
  scale, ``family._prepare_text_conditions`` for the cached conditions), so no
  second data policy exists;
* per-rank prep failures are **negotiated** (one ``all_gather_object``) before
  any collective, so a rank whose batch prep raised cannot strand its peer;
* batch identity is proven with a SHA-256 hash over the raw latent bytes -- a
  strong, order-independent, dtype-exact digest (a sum/numel pair would collide
  under permutation, and a self-incrementing id would assert nothing);
* rank 0 samples the flow-matching inputs with
  :func:`library.models.moss_image.family.sample_moss_flow` (the extracted,
  unchanged MOSS math) and **broadcasts** ``(noise, sigmas)`` over the PP
  group, so the last stage's ``target = noise - latents`` is bit-identical
  without any RNG-order coupling between the processes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from library.models.moss_image.family import sample_moss_flow  # noqa: F401  (re-exported for the MOSS adapter contract)

from .errors import (
    CODE_CONFIG_INVALID,
    CODE_STAGE_CONTRACT,
    PipelineParallelError,
)


@dataclass(frozen=True)
class MossStepInputs:
    """One whole-step PP training input bundle (already synchronized)."""

    noisy_input: Tensor
    sigmas: Tensor
    timesteps: Tensor
    target: Tensor
    text_hiddens: Tensor | None
    text_mask: Tensor | None
    batch_digest: str

    @property
    def batch_size(self) -> int:
        return int(self.noisy_input.shape[0])

    def step_kwargs(self, runtime) -> dict:
        """The ``runtime.step`` kwargs for the MOSS wire (frozen form)."""

        return {
            "timesteps": self.timesteps,
            "text_hiddens": self.text_hiddens,
            "text_mask": self.text_mask,
        }

    def wire_for(self, runtime) -> Tensor | None:
        """The wire latents this rank passes into ``runtime.step`` (frozen)."""

        return self.noisy_input if runtime.is_first else None


def prepare_cached_conditions(trainer, args, batch, *, accelerator, weight_dtype):
    """The cached MOSS text conditions ``(hiddens, mask)`` for this batch.

    PP v1 requires the disk caches to be complete (the shared session setup
    already refuses incomplete caches); a batch without cached outputs is a
    named error instead of a silent live-encode detour.
    """

    from library.models.moss_image.family import _prepare_text_conditions

    conditions = batch.get("text_encoder_outputs_list", None)
    if not conditions or conditions[0] is None:
        raise PipelineParallelError(
            CODE_CONFIG_INVALID,
            "the MOSS PP trainer path serves the cached text-encoder configuration "
            "only; this batch arrived without cached text_encoder_outputs_list",
        )
    ctx = SimpleCtx(
        args=args,
        accelerator=accelerator,
        network=getattr(trainer, "_network", None),
        weight_dtype=weight_dtype,
    )
    return _prepare_text_conditions(trainer, ctx, batch, conditions)


class SimpleCtx:
    """Minimal stand-in for the loop's TrainCtx (cached-condition path only)."""

    def __init__(self, *, args, accelerator, network, weight_dtype):
        self.args = args
        self.accelerator = accelerator
        self.network = network
        self.weight_dtype = weight_dtype


def prepare_clean_latents(trainer, args, batch, *, accelerator, vae, vae_dtype):
    """The clean latents exactly as the non-PP path prepares them."""

    from library.training.batch_step import _prepare_latents

    return _prepare_latents(args, trainer, vae, batch, accelerator, vae_dtype)


def tensor_digest(tensor: Tensor) -> str:
    """SHA-256 over the exact raw bytes of ``tensor`` (CPU copy)."""

    payload = tensor.detach().cpu().contiguous().flatten()
    # view(torch.uint8) rejects 0-d/odd shapes; flatten + explicit byte view
    # keeps the digest dtype-exact and device-independent.
    raw = payload.numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _distributed(group=None) -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def _negotiate_prep(error: tuple[str, str] | None, *, group, where: str) -> None:
    """Share per-rank prep verdicts before the first collective of the step."""

    if not _distributed(group):
        if error is not None:
            raise PipelineParallelError(*error)
        return
    gathered: list = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, error, group=group)
    failures = [item for item in gathered if item]
    if failures:
        code, detail = failures[0]
        raise PipelineParallelError(
            code, f"negotiated batch-prep refusal ({where}): {detail}"
        )


def _refuse_mismatched_ranks(digest: str, *, group) -> None:
    """Every rank compares its digest against rank 0's and refuses identically."""

    if not _distributed(group):
        return
    gathered: list = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, digest, group=group)
    reference = gathered[0]
    if any(item != reference for item in gathered):
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            "pp_data_sync: the pipeline stages hold different logical batches "
            f"(latent SHA-256 digests {sorted(set(gathered))[:2]}...). Under PP "
            "both stages must see the same unsharded batch; a DDP-style loader "
            "split would silently train each half of the model on different data",
        )


def build_family_step_inputs(
    adapter,
    trainer,
    args,
    batch,
    *,
    accelerator,
    vae,
    vae_dtype,
    weight_dtype,
    text_encoding_strategy=None,
    group=None,
    is_first: bool,
):
    """Prepare and synchronize one whole step of PP training inputs (any family).

    The family-independent protocol (the generalization of the MOSS wiring):

    * cached latents are prepared exactly like the non-PP path, then mapped to
      the family wire shape (``adapter.wire_latents``) and digested;
    * per-rank prep failures are negotiated before any collective;
    * batch identity is proven with the SHA-256 latent digest;
    * rank 0 samples the family's noise/timestep math and broadcasts its
      payloads (same values on every stage, no RNG-order coupling); the last
      stage reconstructs its noisy input from the broadcast payloads;
    * the target and the step-input bundle come from the family adapter.
    """

    error: tuple[str, str] | None = None
    latents = None
    conditions: dict | None = None
    digest = ""
    try:
        raw = prepare_clean_latents(
            trainer, args, batch,
            accelerator=accelerator, vae=vae, vae_dtype=vae_dtype,
        ).to(dtype=weight_dtype)
        latents = adapter.wire_latents(raw)
        digest = tensor_digest(latents)
        if is_first or getattr(adapter, "conditions_on_every_rank", False):
            conditions = adapter.prepare_conditions(
                trainer, args, batch,
                accelerator=accelerator, weight_dtype=weight_dtype,
                text_encoding_strategy=text_encoding_strategy,
            )
    except PipelineParallelError as exc:
        error = (exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 - negotiated below, never rank-local
        error = ("pp_stage_contract_violation", f"{type(exc).__name__}: {exc}")
    _negotiate_prep(error, group=group, where="pp_trainer_inputs")
    _refuse_mismatched_ranks(digest, group=group)

    if is_first:
        flow = adapter.sample_flow(
            args, trainer, latents, accelerator=accelerator, weight_dtype=weight_dtype
        )
        broadcast = flow.broadcast
        noisy_input = flow.noisy_input
        if _distributed(group):
            for tensor in broadcast.values():
                dist.broadcast(tensor, src=0, group=group)
        noise = broadcast["noise"]
        timesteps = adapter.timesteps_from_flow(broadcast)
    else:
        broadcast = adapter.empty_broadcast(latents, weight_dtype)
        if not _distributed(group):
            raise PipelineParallelError(
                CODE_STAGE_CONTRACT,
                f"the last {adapter.family_label} PP stage needs a process "
                "group to receive the synchronized noise payloads",
            )
        for tensor in broadcast.values():
            dist.broadcast(tensor, src=0, group=group)
        noise = broadcast["noise"]
        noisy_input, timesteps = adapter.receive_flow(
            latents, broadcast, weight_dtype=weight_dtype
        )
    target = adapter.make_target(latents, noise)
    conditions = adapter.finalize_conditions(latents, {**(conditions or {}), **broadcast})
    return adapter.build_inputs(
        noisy_input=noisy_input,
        timesteps=timesteps,
        target=target,
        conditions=conditions,
        digest=digest,
    )


def build_step_inputs(
    trainer,
    args,
    batch,
    *,
    accelerator,
    vae,
    vae_dtype,
    weight_dtype,
    group=None,
    is_first: bool,
) -> MossStepInputs:
    """Prepare and synchronize one whole step of MOSS PP training inputs."""

    from .trainer_adapters import ADAPTERS

    return build_family_step_inputs(
        ADAPTERS["moss_image"], trainer, args, batch,
        accelerator=accelerator, vae=vae, vae_dtype=vae_dtype,
        weight_dtype=weight_dtype, group=group, is_first=is_first,
    )


__all__ = [
    "MossStepInputs",
    "build_step_inputs",
    "prepare_cached_conditions",
    "tensor_digest",
]
