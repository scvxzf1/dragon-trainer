"""Loss registry + composer.

M1 extraction (plan.md): the loss side of `_process_batch_inner` becomes a
registry of small callables. The composer calls the active handlers in three
phases matching the pre-refactor reduction order — break that ordering and
ortho / multiscale numerics shift.

Reduction order (must match train.py pre-refactor):
  1. Per-sample [B] stage:
       flow_match   — base FM (weighting + masked + loss_weights).
  2. Per-sample += scalar broadcast stage (was `post_process_loss`):
       ortho_reg     — OrthoLoRA orthogonality regularizer
       hydra_balance — MoE load-balance loss
       functional    — functional inversion MSE (weight-gated)
  3. Scalar stage (after `.mean()` reduction):
       multiscale    — avg_pool2d MSE on pred/target

The composer does not own forward passes. Functional-loss forwards still
happen inside the trainer (they need `anima()` and post_process_network
hooks). Those forwards stash their aux tensors on `LossContext.aux`, and the
composer consumes them.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from library.training.mask_loss import apply_mask, reduce_masked_loss


def add_custom_train_arguments(
    parser: argparse.ArgumentParser, support_weighted_captions: bool = True
):
    parser.add_argument(
        "--min_snr_gamma",
        type=float,
        default=None,
        help="gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by paper.",
    )
    parser.add_argument(
        "--scale_v_pred_loss_like_noise_pred",
        action="store_true",
        help="scale v-prediction loss like noise prediction loss",
    )
    parser.add_argument(
        "--v_pred_like_loss",
        type=float,
        default=None,
        help="add v-prediction like loss multiplied by this value",
    )
    parser.add_argument(
        "--debiased_estimation_loss",
        action="store_true",
        help="debiased estimation loss",
    )
    parser.add_argument(
        "--velocity_direction_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a FasterDiT-style auxiliary velocity direction loss. "
            "0 disables it. The term is training-only and keeps validation FM-MSE clean."
        ),
    )
    parser.add_argument(
        "--velocity_direction_loss_eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon for velocity direction cosine similarity.",
    )
    parser.add_argument(
        "--prior_preservation_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for no-extra-dataset prior-preservation MSE. "
            "0 disables it. The trainer supplies a no-grad base-model prediction "
            "with the adapter temporarily zeroed."
        ),
    )
    parser.add_argument(
        "--blank_prompt_preservation",
        action="store_true",
        default=False,
        help=(
            "Use T5('') as the prior-preservation text condition. Requires "
            "--prior_preservation_weight > 0 and reuses the staged max-padded "
            "unconditional cross-attention sidecar instead of running the text "
            "encoder in the training loop."
        ),
    )
    parser.add_argument(
        "--diff_output_preservation_trigger",
        type=str,
        default=None,
        help=(
            "Trigger text to replace when building DOP/class-prompt prior captions. "
            "Requires --diff_output_preservation_class and "
            "--prior_preservation_weight > 0."
        ),
    )
    parser.add_argument(
        "--diff_output_preservation_class",
        type=str,
        default=None,
        help=(
            "Class prompt used for DOP/class-prompt prior preservation, e.g. "
            "'woman' or 'character'. Text caches store its max-padded "
            "cross-attention embedding per image."
        ),
    )
    parser.add_argument(
        "--inverted_mask_prior_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for no-extra-dataset inverted-mask prior preservation. "
            "When alpha masks are present, the trainer compares the current "
            "prediction to an adapter-disabled base prediction only outside "
            "the mask."
        ),
    )
    parser.add_argument(
        "--adaptive_personalization_observe",
        action="store_true",
        help="Observe base-vs-adapter loss by timestep bin; does not alter training.",
    )
    parser.add_argument("--adaptive_personalization_timestep_bins", type=int, default=10)
    parser.add_argument("--adaptive_personalization_ema_decay", type=float, default=0.95)
    parser.add_argument("--adaptive_personalization_probe_every_n_steps", type=int, default=4)
    parser.add_argument("--adaptive_personalization_gamma_scale", type=float, default=1.0)
    parser.add_argument(
        "--adaptive_personalization_loss_weighting",
        action="store_true",
        help="Apply 1-gamma timestep-bin weights after observer warmup.",
    )
    parser.add_argument("--adaptive_personalization_min_bin_samples", type=int, default=16)
    parser.add_argument("--adaptive_personalization_denoise_weight_min", type=float, default=0.25)
    parser.add_argument(
        "--adaptive_personalization_affine",
        action="store_true",
        help="Apply gamma-driven synchronous affine augmentation after observation warmup.",
    )
    parser.add_argument("--adaptive_personalization_affine_probability_max", type=float, default=0.5)
    parser.add_argument("--adaptive_personalization_affine_rotation_deg", type=float, default=10.0)
    parser.add_argument("--adaptive_personalization_affine_translation", type=float, default=0.05)
    parser.add_argument("--adaptive_personalization_affine_scale_delta", type=float, default=0.05)
    parser.add_argument(
        "--p2_gamma",
        type=float,
        default=1.0,
        help="Gamma for weighting_scheme='p2' loss weighting.",
    )
    parser.add_argument(
        "--p2_k",
        type=float,
        default=1.0,
        help="k offset for weighting_scheme='p2' loss weighting.",
    )
    if support_weighted_captions:
        parser.add_argument(
            "--weighted_captions",
            action="store_true",
            default=False,
            help="Enable weighted captions in the standard style (token:1.3).",
        )


def apply_masked_loss(loss, batch) -> torch.FloatTensor:
    return apply_mask(loss, batch)


def apply_inverted_masked_loss(loss, batch) -> torch.FloatTensor:
    if "alpha_masks" not in batch or batch["alpha_masks"] is None:
        return loss.new_zeros(loss.shape)

    target_hw = tuple(int(d) for d in loss.shape[-2:])
    mask_image = batch["alpha_masks"].to(
        device=loss.device, dtype=loss.dtype, non_blocking=True
    ).unsqueeze(1)
    if tuple(int(d) for d in mask_image.shape[-2:]) != target_hw:
        mask_image = torch.nn.functional.interpolate(
            mask_image, size=target_hw, mode="area"
        )
    if loss.ndim == 5:
        mask_image = mask_image.unsqueeze(2)
    return loss * (1.0 - mask_image)


def get_huber_threshold_if_needed(
    args, timesteps: torch.Tensor, noise_scheduler
) -> Optional[torch.Tensor]:
    if args.loss_type == "pseudo_huber":
        b_size = timesteps.shape[0]
        return torch.full((b_size,), args.pseudo_huber_c, device=timesteps.device)
    if not (args.loss_type == "huber" or args.loss_type == "smooth_l1"):
        return None

    b_size = timesteps.shape[0]
    if args.huber_schedule == "exponential":
        # `timesteps` is σ∈[0,1] — Anima feeds the DiT time arg directly (see
        # runtime/noise.py). The original sd-scripts formula divided alpha by
        # num_train_timesteps because it expected timesteps∈[0,1000]; on the σ
        # scale that 1000× shrinks the exponent to ~0, pinning the threshold
        # flat at huber_scale. Drop the divisor so the schedule decays as
        # intended: huber_c**σ · huber_scale, i.e. huber_scale at σ=0 (clean)
        # down to huber_c·huber_scale at σ=1 (noise).
        alpha = -math.log(args.huber_c)
        result = torch.exp(-alpha * timesteps) * args.huber_scale
    elif args.huber_schedule == "snr":
        if not hasattr(noise_scheduler, "alphas_cumprod"):
            raise NotImplementedError(
                "Huber schedule 'snr' is not supported with the current model."
            )
        alphas_cumprod = torch.index_select(
            noise_scheduler.alphas_cumprod, 0, timesteps.cpu()
        )
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        result = (1 - args.huber_c) / (1 + sigmas) ** 2 + args.huber_c
        result = result.to(timesteps.device)
    elif args.huber_schedule == "constant":
        result = torch.full(
            (b_size,), args.huber_c * args.huber_scale, device=timesteps.device
        )
    else:
        raise NotImplementedError(f"Unknown Huber loss schedule {args.huber_schedule}!")

    return result


def conditional_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    reduction: str,
    huber_c: Optional[torch.Tensor] = None,
):
    if loss_type == "l2":
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "l1":
        loss = torch.nn.functional.l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = (
            2
            * huber_c
            * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        )
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "smooth_l1":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "pseudo_huber":
        if huber_c is None:
            raise ValueError("pseudo_huber_c is required for pseudo_huber loss")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")
    return loss


# Internal alias — still referenced below by the composer stages.
_conditional_loss = conditional_loss


# ---------------------------------------------------------------------------
# Context + types
# ---------------------------------------------------------------------------


@dataclass
class LossContext:
    args: argparse.Namespace
    batch: dict
    model_pred: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor
    weighting: Optional[torch.Tensor]
    huber_c: Optional[torch.Tensor]
    loss_weights: torch.Tensor
    network: object
    aux: dict = field(default_factory=dict)
    is_train: bool = True


LossFn = Callable[[LossContext], torch.Tensor]


# ---------------------------------------------------------------------------
# Per-sample losses ([B])
# ---------------------------------------------------------------------------


def _flow_match_loss(ctx: LossContext) -> torch.Tensor:
    """Base rectified-flow MSE with weighting, masked loss, per-sample weight.

    Mirrors train.py lines 1041–1055 (pre-M1). Returns a [B] tensor.
    """
    loss = _conditional_loss(
        ctx.model_pred.float(),
        ctx.target.float(),
        ctx.args.loss_type,
        "none",
        ctx.huber_c,
    )
    if ctx.weighting is not None:
        loss = loss * ctx.weighting
    if ctx.args.masked_loss or (
        "alpha_masks" in ctx.batch and ctx.batch["alpha_masks"] is not None
    ):
        return _reduce_loss_with_mask(ctx, loss)
    loss = loss.mean(dim=list(range(1, loss.ndim)))
    loss = loss * ctx.loss_weights
    return loss


def _reduce_loss_with_mask(ctx: LossContext, loss: torch.Tensor) -> torch.Tensor:
    """Reduce a masked spatial loss using the configured compatibility mode."""
    foreground_weight = getattr(ctx.args, "foreground_loss_weight", 1.0)
    if foreground_weight is None:
        foreground_weight = 1.0
    reduced = reduce_masked_loss(
        loss,
        ctx.batch,
        normalize=str(getattr(ctx.args, "mask_loss_normalize", "none") or "none"),
        foreground_weight=float(foreground_weight),
    )
    return reduced * ctx.loss_weights


def _flow_matching_vr_loss(ctx: LossContext) -> torch.Tensor:
    """AsymFlow §5.2 control-variate FM loss.

    Replaces ``y² → (y + λ·z)²`` per element, where::

        y = model_pred − target                (gradient flows here)
        z = ref_pred_L − (noise − x_0^L)       (no_grad, supplied by trainer)

    λ is estimated online as ``λ* = −Cov(y, z) / Var(z)`` on the *detached*
    residuals and tracked through an EMA across batches (β default 0.01).
    Theory only applies to the squared-error loss, so we always compute
    ``(y + λz)²`` regardless of ``args.loss_type``.

    Trainer contract: when active, ``train.py::get_noise_pred_and_target``
    stashes ``ctx.aux['vr'] = {'z': Tensor, 'state': mutable_dict}``; this
    handler updates ``state['lambda_ema']`` in place. If the aux entry is
    missing (e.g. validation step), falls back to standard flow-match.
    """
    vr_aux = ctx.aux.get("vr") or {}
    z = vr_aux.get("z")
    weight = float(getattr(ctx.args, "vr_loss_weight", 0.0) or 0.0)
    if weight <= 0.0 or z is None:
        return _flow_match_loss(ctx)

    y = ctx.model_pred.float() - ctx.target.float()
    z_f = z.float()

    # Per-batch λ_batch on detached residuals, then EMA across batches.
    with torch.no_grad():
        y_d = y.detach()
        cov = (y_d * z_f).sum()
        var = (z_f * z_f).sum().clamp_min(1e-12)
        lambda_batch = float(-(cov / var).item())

    beta = float(getattr(ctx.args, "vr_lambda_beta", 0.01) or 0.0)
    state = vr_aux.get("state")
    prev = state.get("lambda_ema") if isinstance(state, dict) else None
    if prev is None or not isinstance(prev, float):
        lambda_ema = lambda_batch
    else:
        lambda_ema = (1.0 - beta) * prev + beta * lambda_batch
    if isinstance(state, dict):
        state["lambda_ema"] = lambda_ema
        state["lambda_batch"] = lambda_batch

    diff = y + lambda_ema * z_f
    loss = diff.pow(2)
    if ctx.weighting is not None:
        loss = loss * ctx.weighting
    if ctx.args.masked_loss or (
        "alpha_masks" in ctx.batch and ctx.batch["alpha_masks"] is not None
    ):
        return weight * _reduce_loss_with_mask(ctx, loss)
    loss = loss.mean(dim=list(range(1, loss.ndim)))
    loss = loss * ctx.loss_weights
    return weight * loss


def _velocity_direction_loss(ctx: LossContext) -> torch.Tensor:
    """FasterDiT-style auxiliary direction loss for velocity prediction.

    The paper computes a cosine term between predicted and target velocity at
    each spatial position. Here `ctx.model_pred` / `ctx.target` have already
    been squeezed to Anima's 4D latent shape `[B,C,H,W]`, so cosine over dim=1
    gives a per-pixel direction residual. This term is training-only so
    validation remains a comparable FM-MSE signal across experiments.
    """
    if not ctx.is_train:
        return ctx.model_pred.new_zeros(ctx.model_pred.shape[0])
    weight = float(getattr(ctx.args, "velocity_direction_loss_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(ctx.model_pred.shape[0])

    eps = float(getattr(ctx.args, "velocity_direction_loss_eps", 1e-6) or 1e-6)
    pred = ctx.model_pred.float()
    target = ctx.target.float()
    loss = 1.0 - F.cosine_similarity(pred, target, dim=1, eps=eps)
    loss = loss.unsqueeze(1)  # [B,1,H,W], compatible with apply_masked_loss
    if ctx.args.masked_loss or (
        "alpha_masks" in ctx.batch and ctx.batch["alpha_masks"] is not None
    ):
        return weight * _reduce_loss_with_mask(ctx, loss)
    loss = loss.mean(dim=list(range(1, loss.ndim)))
    loss = loss * ctx.loss_weights
    return weight * loss


def _prior_preservation_loss(ctx: LossContext) -> torch.Tensor:
    """No-extra-dataset prior preservation against a base-model prediction."""
    if not ctx.is_train:
        return ctx.model_pred.new_zeros(ctx.model_pred.shape[0])
    weight = float(getattr(ctx.args, "prior_preservation_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(ctx.model_pred.shape[0])

    aux = ctx.aux.get("prior_preservation") or {}
    prior_pred = aux.get("prior_pred")
    if prior_pred is None:
        return ctx.model_pred.new_zeros(ctx.model_pred.shape[0])
    if prior_pred.dim() == 5:
        prior_pred = prior_pred.squeeze(2)

    loss = F.mse_loss(ctx.model_pred.float(), prior_pred.float(), reduction="none")
    loss = loss.mean(dim=list(range(1, loss.ndim)))
    loss = loss * ctx.loss_weights
    return weight * loss


def _inverted_mask_prior_loss(ctx: LossContext) -> torch.Tensor:
    """Preserve base-model behavior outside user-provided alpha masks."""
    if not ctx.is_train:
        return ctx.model_pred.new_zeros(ctx.model_pred.shape[0])
    weight = float(getattr(ctx.args, "inverted_mask_prior_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(ctx.model_pred.shape[0])

    aux = ctx.aux.get("inverted_mask_prior") or {}
    prior_pred = aux.get("prior_pred")
    if prior_pred is None or ctx.batch.get("alpha_masks") is None:
        return ctx.model_pred.new_zeros(ctx.model_pred.shape[0])
    if prior_pred.dim() == 5:
        prior_pred = prior_pred.squeeze(2)

    loss = F.mse_loss(ctx.model_pred.float(), prior_pred.float(), reduction="none")
    loss = apply_inverted_masked_loss(loss, ctx.batch)
    loss = loss.mean(dim=list(range(1, loss.ndim)))
    loss = loss * ctx.loss_weights
    return weight * loss


# ---------------------------------------------------------------------------
# Scalar-broadcast regularizers (added to the per-sample [B] tensor)
# ---------------------------------------------------------------------------


def _ortho_reg_loss(ctx: LossContext) -> torch.Tensor:
    weight = float(getattr(ctx.network, "_ortho_reg_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(())
    return weight * ctx.network.get_ortho_regularization()


def _hydra_balance_loss(ctx: LossContext) -> torch.Tensor:
    # Chimera bakes the warmup gate into its own per-pool sum (content
    # rides the outer warmup, freq fires from step 0). Consume the
    # scalar directly without re-multiplying; the early-exit on
    # ``_balance_loss_weight <= 0`` would otherwise zero out the freq
    # term during the warmup window.
    if getattr(ctx.network, "_use_chimera_hydra", False):
        return ctx.network.get_balance_loss()
    weight = float(getattr(ctx.network, "_balance_loss_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(())
    return weight * ctx.network.get_balance_loss()


def _functional_loss(ctx: LossContext) -> torch.Tensor:
    weight = float(getattr(ctx.args, "functional_loss_weight", 0.0) or 0.0)
    func_loss = ctx.aux.get("func_loss")
    if weight <= 0.0 or func_loss is None:
        return ctx.model_pred.new_zeros(())
    # Per-sample running loss is float32 (flow_match casts inputs via .float()).
    # Match the pre-refactor cast: `func_weight * func_loss.to(loss.dtype)`.
    return weight * func_loss.float()


def _soft_tokens_contrastive_loss(ctx: LossContext) -> torch.Tensor:
    """SoftREPA-style contrastive term on the soft-tokens bank.

    The InfoNCE scalar itself is computed by ``SoftTokensMethodAdapter``
    (it needs ``k`` extra DiT forwards) and stashed under
    ``aux["soft_tokens_contrastive"]``; this handler just applies the
    warmup-gated weight ``network._contrastive_weight`` (updated each step by
    ``SoftTokensNetwork.step_contrastive_warmup``).

    Training-only: gated on ``ctx.is_train`` so validation FM-MSE stays a clean
    per-token regression metric (the contrastive term is a separate objective
    that doesn't track held-out denoise quality on Anima —
    ``project_fm_val_loss_uninformative``).
    """
    if not ctx.is_train:
        return ctx.model_pred.new_zeros(())
    weight = float(getattr(ctx.network, "_contrastive_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(())
    con_loss = ctx.aux.get("soft_tokens_contrastive")
    if con_loss is None:
        return ctx.model_pred.new_zeros(())
    return weight * con_loss.float()


def _fera_fecl_bands(
    z: torch.Tensor, num_bands: int, fei_sigma_low_div: float
) -> list[torch.Tensor]:
    """Decompose ``z (B, C, H, W)`` into ``num_bands`` Laplacian-pyramid
    components (high → low). Uses ``library.runtime.fei.gaussian_blur_2d``;
    fp32 internally so bf16 latents don't underflow the squared norm.

    ``σ_low = min(H_lat, W_lat) / fei_sigma_low_div`` keeps band semantics
    aspect-invariant; subsequent σ's double outward.
    """
    if num_bands < 2:
        raise ValueError(f"num_bands must be >= 2, got {num_bands}")
    from library.runtime.fei import gaussian_blur_2d

    z = z.float()
    h_lat, w_lat = int(z.shape[-2]), int(z.shape[-1])
    sigma_low = float(min(h_lat, w_lat)) / float(fei_sigma_low_div)
    sigmas = [sigma_low * (2.0**k) for k in range(num_bands - 1)]
    pyr = [z]
    for s in sigmas:
        pyr.append(gaussian_blur_2d(pyr[-1], s))
    bands = [pyr[k] - pyr[k + 1] for k in range(num_bands - 1)]
    bands.append(pyr[-1])
    return bands


def _fera_fecl_loss(ctx: LossContext) -> torch.Tensor:
    """FeRA Frequency-Energy Consistency Loss (Yin et al. eq. 10).

    Bandwise consistency between adapter correction ``δ = z_fera − z_base``
    and residual ``r = z_fera − z_target``, weighted by the residual's
    per-band energy share. Active on the ``stacked_experts_global_fei``
    spec when ``fera_fecl_weight > 0`` — the trainer stashes ``z_base``
    (no-grad base-pass prediction with routing zeroed) in
    ``ctx.aux['fera']`` and this handler runs the band decomposition.

    The 2-band path collapses Eq. 10 to a content-free scalar (only two
    ratios that sum to 1), so production training should keep
    ``fera_fecl_weight = 0.0`` until bench-validated at 3 bands — see
    ``[[project_fera_probe_2band_decision]]``.
    """
    weight = float(
        getattr(ctx.network, "fecl_weight", None)
        or getattr(getattr(ctx.network, "cfg", None), "fera_fecl_weight", 0.0)
        or 0.0
    )
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(())

    fera_aux = ctx.aux.get("fera") or {}
    z_base = fera_aux.get("z_base")
    if z_base is None:
        return ctx.model_pred.new_zeros(())

    cfg = getattr(ctx.network, "cfg", None)
    num_bands = int(
        getattr(cfg, "fera_num_bands", None) or fera_aux.get("num_bands", 3)
    )
    fei_sigma_low_div = float(
        getattr(cfg, "fei_sigma_low_div", None)
        or fera_aux.get("fei_sigma_low_div", 4.0)
    )

    def _to4(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(2) if x.dim() == 5 else x

    z_base_4 = _to4(z_base).float()
    z_fera = _to4(ctx.model_pred).float()
    z_target = _to4(ctx.target).float()

    delta = z_fera - z_base_4
    resid = z_fera - z_target
    delta_bands = _fera_fecl_bands(delta, num_bands, fei_sigma_low_div)
    resid_bands = _fera_fecl_bands(resid, num_bands, fei_sigma_low_div)

    eps = 1e-8
    d_total = delta.flatten(1).pow(2).sum(-1).sqrt().clamp_min(eps)
    r_total = resid.flatten(1).pow(2).sum(-1).sqrt().clamp_min(eps)
    r_band_e = torch.stack([b.flatten(1).pow(2).sum(-1) for b in resid_bands], dim=-1)
    r_share = r_band_e / r_band_e.sum(-1, keepdim=True).clamp_min(eps)

    loss = z_target.new_zeros(z_target.shape[0])
    for k in range(num_bands):
        d_band = delta_bands[k].flatten(1).pow(2).sum(-1).sqrt()
        r_band = resid_bands[k].flatten(1).pow(2).sum(-1).sqrt()
        term = (d_band / d_total - r_band / r_total).pow(2)
        loss = loss + r_share[:, k] * term

    return weight * loss.mean()


# ---------------------------------------------------------------------------
# Scalar post-reduction losses (operate on the scalar mean of the per-sample)
# ---------------------------------------------------------------------------


def _multiscale_loss(ctx: LossContext) -> torch.Tensor:
    """Additional MSE term at 2x-downsampled resolution. Scalar output meant to
    be blended into the scalar mean via `(scalar + ms*ms_w) / (1 + ms_w)`.
    The composer applies that blend — this handler returns the raw MSE.
    """
    ms_weight = float(getattr(ctx.args, "multiscale_loss_weight", 0.0) or 0.0)
    if ms_weight <= 0.0:
        return ctx.model_pred.new_zeros(())
    h, w = ctx.model_pred.shape[-2:]
    side_length = math.sqrt(h * w) * 8
    if side_length < 1024 * 0.9 or h < 2 or w < 2:
        return ctx.model_pred.new_zeros(())
    pred_ds = torch.nn.functional.avg_pool2d(ctx.model_pred.float(), 2)
    target_ds = torch.nn.functional.avg_pool2d(ctx.target.float(), 2)
    return torch.nn.functional.mse_loss(pred_ds, target_ds)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


LOSS_REGISTRY: dict[str, LossFn] = {
    "flow_match": _flow_match_loss,
    "flow_matching_vr": _flow_matching_vr_loss,
    "velocity_direction": _velocity_direction_loss,
    "prior_preservation": _prior_preservation_loss,
    "inverted_mask_prior": _inverted_mask_prior_loss,
    "ortho_reg": _ortho_reg_loss,
    "hydra_balance": _hydra_balance_loss,
    "functional": _functional_loss,
    "multiscale": _multiscale_loss,
    "fera_fecl": _fera_fecl_loss,
    "soft_tokens_contrastive": _soft_tokens_contrastive_loss,
}


# Which stage each registered loss runs in (see module docstring).
# `flow_match` and `flow_matching_vr` are mutually exclusive — both produce
# the primary per-sample [B] tensor. `velocity_direction` is an optional
# FasterDiT-style auxiliary per-sample term.
_STAGE_PER_SAMPLE = (
    "flow_match",
    "flow_matching_vr",
    "velocity_direction",
    "prior_preservation",
    "inverted_mask_prior",
)
_STAGE_SCALAR_BROADCAST = (
    "ortho_reg",
    "hydra_balance",
    "functional",
    "fera_fecl",
    "soft_tokens_contrastive",
)
_STAGE_SCALAR_POST = ("multiscale",)
# _STAGE_SCALAR_POST is consulted by LossComposer.compose via the hard-coded
# multiscale branch; kept as a named constant for documentation / future
# extensibility.
__all__ = [
    "LossContext",
    "LossComposer",
    "LossFn",
    "LOSS_REGISTRY",
    "build_loss_composer",
    "_STAGE_PER_SAMPLE",
    "_STAGE_SCALAR_BROADCAST",
    "_STAGE_SCALAR_POST",
]


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


@dataclass
class LossComposer:
    """Holds the active loss entries and composes them in-order.

    `active_losses` is a list of names that exist in LOSS_REGISTRY. The
    composer groups them by stage and applies them. `build_loss_composer`
    decides which names to include based on `args` + `network`.
    """

    active_losses: list[str]

    def compose(self, ctx: LossContext) -> torch.Tensor:
        per_sample = ctx.model_pred.new_zeros(ctx.model_pred.shape[0])

        # Stage 1: per-sample losses.
        first = True
        for name in _STAGE_PER_SAMPLE:
            if name not in self.active_losses:
                continue
            contribution = LOSS_REGISTRY[name](ctx)
            per_sample = contribution if first else (per_sample + contribution)
            first = False
        if first:
            # exactly one of {flow_match, flow_matching_vr} must always be
            # present; defend against a caller passing an empty composer.
            raise RuntimeError(
                "LossComposer: no per-sample loss registered; "
                "one of {'flow_match', 'flow_matching_vr'} must be in active_losses"
            )

        # Stage 2: scalar-broadcast regularizers (added to the per-sample [B]).
        for name in _STAGE_SCALAR_BROADCAST:
            if name not in self.active_losses:
                continue
            reg = LOSS_REGISTRY[name](ctx)
            if reg is None:
                continue
            per_sample = per_sample + reg  # broadcast scalar -> [B]

        scalar = per_sample.mean()

        # Stage 3: scalar-level blend (multiscale).
        if "multiscale" in self.active_losses:
            ms_weight = float(getattr(ctx.args, "multiscale_loss_weight", 0.0) or 0.0)
            if ms_weight > 0.0:
                ms_loss = LOSS_REGISTRY["multiscale"](ctx)
                if ms_loss is not None and torch.is_tensor(ms_loss) and ms_loss.numel():
                    # pre-refactor: (scalar + ms * ms_w) / (1 + ms_w), only when
                    # side_length >= 0.9 * 1024. The guard is inside
                    # _multiscale_loss and returns 0 when it shouldn't apply —
                    # check against zero to preserve exact behavior.
                    if not (ms_loss == 0).all():
                        scalar = (scalar + ms_loss * ms_weight) / (1.0 + ms_weight)

        return scalar


def build_loss_composer(args: argparse.Namespace, network: object) -> LossComposer:
    """Inspect args + network and return the active LossComposer.

    Rules:
      - exactly one of flow_match / flow_matching_vr is active. VR wins when
        args.vr_loss_weight > 0 (the trainer is responsible for running the
        adapter-bypass no-grad forward and stashing ctx.aux['vr']).
      - velocity_direction active iff args.velocity_direction_loss_weight > 0.
      - prior_preservation active iff args.prior_preservation_weight > 0.
      - inverted_mask_prior active iff args.inverted_mask_prior_weight > 0.
      - ortho_reg active iff network._ortho_reg_weight > 0.
      - hydra_balance active iff network._balance_loss_weight > 0.
      - functional active iff args.functional_loss_weight > 0.
      - multiscale active iff args.multiscale_loss_weight > 0.
      - soft_tokens_contrastive active iff
        network._contrastive_target_weight > 0 (gated on the target,
        not the live warmup-held value).
    """
    fm_name = (
        "flow_matching_vr"
        if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0
        else "flow_match"
    )
    active: list[str] = [fm_name]

    if float(getattr(args, "velocity_direction_loss_weight", 0.0) or 0.0) > 0.0:
        active.append("velocity_direction")
    if float(getattr(args, "prior_preservation_weight", 0.0) or 0.0) > 0.0:
        active.append("prior_preservation")
    if float(getattr(args, "inverted_mask_prior_weight", 0.0) or 0.0) > 0.0:
        active.append("inverted_mask_prior")

    if float(getattr(network, "_ortho_reg_weight", 0.0) or 0.0) > 0.0:
        active.append("ortho_reg")
    # Chimera always activates hydra_balance — the freq pool's term fires
    # from step 0 (bypasses warmup), so we can't gate composer activation
    # on the warmup-held ``_balance_loss_weight``.
    if float(getattr(network, "_balance_loss_weight", 0.0) or 0.0) > 0.0 or bool(
        getattr(network, "_use_chimera_hydra", False)
    ):
        active.append("hydra_balance")
    if float(getattr(args, "functional_loss_weight", 0.0) or 0.0) > 0.0:
        active.append("functional")
    if float(getattr(args, "multiscale_loss_weight", 0.0) or 0.0) > 0.0:
        active.append("multiscale")
    # FeRA FECL: active iff a ``LoRANetwork`` carrying the
    # stacked_experts_global_fei spec has a positive ``fecl_weight``. The
    # trainer's base-pass forward gate (in
    # ``train.py::get_noise_pred_and_target``) is the same condition, so the
    # composer activation just mirrors it.
    fecl_weight = float(getattr(network, "fecl_weight", 0.0) or 0.0)
    if (
        fecl_weight > 0.0
        and getattr(getattr(network, "cfg", None), "use_moe_style", False)
        == "independent_A"
    ):
        active.append("fera_fecl")
    # soft_tokens contrastive: gate on the *target* weight (warmup may hold the
    # live ``_contrastive_weight`` at 0 for the first ratio*steps). The
    # SoftTokensMethodAdapter supplies the InfoNCE scalar via aux.
    if float(getattr(network, "_contrastive_target_weight", 0.0) or 0.0) > 0.0:
        active.append("soft_tokens_contrastive")

    return LossComposer(active_losses=active)
