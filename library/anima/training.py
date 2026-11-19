# Anima Training Utilities

import argparse
import gc
import os
import random
import time
from typing import List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm

from library.runtime.device import clean_memory_on_device, synchronize_device
from library import train_util
from library.training.preview_lifecycle import (
    apply_preview_router_conditioning,
    preview_lifecycle,
)
from library.training.sample_preview_common import finish_preview_batch
from library.datasets.buckets import snap_sample_size
from library.training.checkpoints import (
    get_epoch_ckpt_name,
    get_remove_epoch_no,
    get_remove_step_no,
    get_step_ckpt_name,
    save_sd_model_on_epoch_end_or_stepwise_common,
    save_sd_model_on_train_end_common,
)
from library.anima import models as anima_models, weights as anima_utils
from library.models import qwen_vae as qwen_image_autoencoder_kl
from library.inference import sampling as inference_sampling

from library.log import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


def _ensure_sample_compile_range(dit, network, prompts) -> bool:
    """Widen the training dynamic-seq compile range to cover these prompts.

    Prompt files are re-read at every sampling event, so a resolution added
    after startup can fall outside the compiled range. Expanding it here means
    the preview renders instead of being skipped by the guard in
    ``_sample_image_inference``. No-op when compile is off or not dynamic-seq.
    """
    from library.datasets.buckets import token_counts_for_sample_prompts
    from library.runtime.harness import ensure_training_compile_seq_range

    return ensure_training_compile_seq_range(
        dit,
        network,
        token_counts_for_sample_prompts(prompts),
        logger=logger,
    )


_TRAINING_SAMPLE_SAMPLERS = {"euler", "er_sde", "lcm"}
_LEGACY_TRAINING_SAMPLE_SAMPLERS = {
    "ddim",
    "pndm",
    "lms",
    "euler_a",
    "heun",
    "dpm_2",
    "dpm_2_a",
    "dpmsolver",
    "dpmsolver++",
    "dpmsingle",
    "k_lms",
    "k_euler",
    "k_euler_a",
    "k_dpm_2",
    "k_dpm_2_a",
}


def normalize_training_sample_sampler(value: object) -> str:
    """Return a sampler supported by Anima's flow-matching preview loop."""
    sampler = str(value or "euler").strip().lower()
    if sampler in _TRAINING_SAMPLE_SAMPLERS:
        return sampler
    if sampler in _LEGACY_TRAINING_SAMPLE_SAMPLERS:
        logger.warning(
            "sample_sampler=%s is a legacy Diffusers scheduler name; "
            "training preview falls back to euler.",
            sampler,
        )
        return "euler"
    logger.warning(
        "Unknown sample_sampler=%s; training preview falls back to euler.",
        sampler,
    )
    return "euler"


# Sentinel users can drop into captions that lack a real artist tag, so the
# shuffle/drop boundary keeps working. Stripped from caption variants before
# they reach the tokenizer (see _generate_caption_variants in
# scripts/preprocess/cache_text_embeddings.py). Callers of anima_smart_shuffle_caption
# that feed the result to a tokenizer must strip it themselves — kept inside
# the shuffle so the boundary index stays consistent with the input.
NO_ARTIST_SENTINEL = "@no-artist"


def _is_artist_tag(tag: str) -> bool:
    """True for @-prefixed artist handles; False for booru emoticons like ``@ @``.

    Matches the predicate the Anima Tagger uses (see
    ``scripts/anima_tagger/vocab.py``): a single ``@`` followed by non-space
    characters. ``@ @`` (``@_@`` after the corpus-wide ``_``→`` `` normalization)
    is a general-category eye-shape tag and must not trigger the shuffle
    boundary.
    """
    return len(tag) >= 2 and tag[0] == "@" and not tag[1].isspace()


def find_anima_prefix_end(tags: List[str]) -> int:
    """Index one past the trailing artist-handle in the leading run.

    Walks ``tags`` accepting any non-artist tags up front, then any consecutive
    artist tags, and stops at the first non-artist tag that follows an artist.
    Returns 0 if no artist tag is present anywhere (the no-artist case the
    ``@no-artist`` sentinel exists to fix). Multi-artist captions
    (e.g. ``@artist1, @artist2, …``) protect the full handle run, not just the
    first handle.
    """
    split_idx = 0
    saw_artist = False
    for idx, tag in enumerate(tags):
        if _is_artist_tag(tag):
            split_idx = idx + 1
            saw_artist = True
        elif saw_artist:
            break
    return split_idx


def strip_no_artist_sentinel(tags: List[str]) -> List[str]:
    """Drop every occurrence of :data:`NO_ARTIST_SENTINEL` from ``tags``."""
    return [t for t in tags if t != NO_ARTIST_SENTINEL]


def anima_smart_shuffle_caption(flex_tokens: List[str]) -> List[str]:
    """Shuffle caption tags with awareness of @artist prefix and 'on the ...' sections.

    - Tags up to and including the trailing artist tag of the leading run are
      kept in order (see :func:`find_anima_prefix_end`). Multi-artist captions
      and the ``@no-artist`` sentinel both preserve the full handle run.
    - Remaining tags are split into sections by 'on the ...' / 'in the ...'
      delimiters; tags within each section are shuffled independently.
    - The ``@no-artist`` sentinel is preserved in the output so the boundary
      index stays usable; callers that feed the result to a tokenizer must
      call :func:`strip_no_artist_sentinel` before joining.
    """
    split_idx = find_anima_prefix_end(flex_tokens)

    prefix = flex_tokens[:split_idx]
    suffix = flex_tokens[split_idx:]

    # Split suffix into sections delimited by "on the ..." tags
    sections: list[list[str]] = [[]]
    for tag in suffix:
        if tag.startswith("On the ") or tag.startswith("In the "):
            sections.append([tag])
        else:
            sections[-1].append(tag)

    result = list(prefix)
    for section in sections:
        if not section:
            continue
        if section[0].startswith("On the ") or section[0].startswith("In the "):
            header, body = [section[0]], section[1:]
        else:
            header, body = [], section
        shuffled = body.copy()
        random.shuffle(shuffled)
        result.extend(header + shuffled)
    return result


# Anima-specific training arguments


def add_anima_training_arguments(parser: argparse.ArgumentParser):
    """Add Anima-specific training arguments to the parser."""
    parser.add_argument(
        "--qwen3",
        type=str,
        default=None,
        help="Path to Qwen3-0.6B model (safetensors file or directory)",
    )
    parser.add_argument(
        "--llm_adapter_path",
        type=str,
        default=None,
        help="Path to separate LLM adapter weights. If None, adapter is loaded from DiT file if present",
    )
    parser.add_argument(
        "--llm_adapter_lr",
        type=float,
        default=None,
        help="Learning rate for LLM adapter. None=same as base LR, 0=freeze adapter",
    )
    parser.add_argument(
        "--self_attn_lr",
        type=float,
        default=None,
        help="Learning rate for self-attention layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--cross_attn_lr",
        type=float,
        default=None,
        help="Learning rate for cross-attention layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=None,
        help="Learning rate for MLP layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--mod_lr",
        type=float,
        default=None,
        help="Learning rate for AdaLN modulation layers. None=same as base LR, 0=freeze. Note: mod layers are not included in LoRA by default.",
    )
    parser.add_argument(
        "--t5_tokenizer_path",
        type=str,
        default=None,
        help="Path to T5 tokenizer directory. If None, uses bundled library/anima/configs/t5_old/",
    )
    parser.add_argument(
        "--vocab_pack_path",
        type=str,
        default=None,
        help=(
            "Path to the selected Anima vocab-pack manifest or resource directory. "
            "Its content digest invalidates stale TE caches without relying on caption mtime."
        ),
    )
    parser.add_argument(
        "--qwen3_max_token_length",
        type=int,
        default=512,
        help="Maximum token length for Qwen3 tokenizer (default: 512)",
    )
    parser.add_argument(
        "--t5_max_token_length",
        type=int,
        default=512,
        help="Maximum token length for T5 tokenizer (default: 512)",
    )
    parser.add_argument(
        "--cache_llm_adapter_outputs",
        action="store_true",
        help="Cache LLM adapter outputs (cross-attention embeddings) in the text encoder cache and skip running the adapter during training. "
        "Requires --cache_text_encoder_outputs. Incompatible with LoRA training for the LLM adapter.",
    )
    parser.add_argument(
        "--use_ip_adapter",
        action="store_true",
        help="Enable IP-Adapter image conditioning (decoupled cross-attention). "
        "Requires the network module to expose set_ip_tokens (e.g. networks.methods.ip_adapter). "
        "Live mode needs --no-use_vae_cache so batch['images'] carries the raw "
        "reference; pre-cache mode (--ip_features_cache_to_disk) reads PE features "
        "from sibling .safetensors and is compatible with --use_vae_cache.",
    )
    parser.add_argument(
        "--ip_features_cache_to_disk",
        action="store_true",
        help="Read image features from sibling sidecars "
        "({stem}_anima_{ip_encoder}.safetensors, produced by `make preprocess-pe`) "
        "instead of running the vision encoder live. "
        "Compatible with --use_vae_cache. Missing cache files raise FileNotFoundError.",
    )
    parser.add_argument(
        "--ip_image_drop_p",
        type=float,
        default=0.1,
        help="IP-Adapter image-conditioning dropout probability per batch (CFG dropout for image branch). "
        "Independent of text-side caption_dropout_rate; default 0.1 matches the original IP-Adapter recipe.",
    )
    parser.add_argument(
        "--ip_encoder",
        type=str,
        default="pe",
        help="IP-Adapter vision encoder name (registered in library/vision/encoders.py). "
        "Default 'pe' = PE-Core-L14-336 (dynamic resolution).",
    )
    parser.add_argument(
        "--ip_diagnostics_epochs",
        type=int,
        default=1,
        help="Number of epochs to keep IP-Adapter per-block diagnostics on for. "
        "Each enabled epoch adds 56 fp32 norm reductions per step (2 per block × 28 blocks). "
        "Default 1 logs the initial summary plus epoch-0 ratios, then auto-disables. "
        "Set to 0 to skip even the warm-up logs, or a large number to keep them on.",
    )
    parser.add_argument(
        "--ip_pair_mode",
        type=str,
        default="self",
        choices=["self", "identity", "identity_cross_artist"],
        help="IP-Adapter distinct-pair (identity) training mode. 'self' (default) "
        "= reference == VAE target (legacy, bit-identical). 'identity' draws the "
        "IP-path reference from a DIFFERENT image of the target's identity "
        "(character → franchise → artist back-off via the caption index), "
        "removing the self-pair copy shortcut. 'identity_cross_artist' additionally "
        "requires character/franchise matches from a different artist (drops the "
        "source style). Requires --ip_features_cache_to_disk. "
        "See docs/proposal/ip-adapter-identity-pairs.md.",
    )
    parser.add_argument(
        "--ip_pair_prob",
        type=float,
        default=0.8,
        help="Fraction of steps that draw a distinct reference under "
        "--ip_pair_mode!=self; the rest self-pair (keeps some self-pairs in the "
        "mix — reference recipes warm up better that way). Default 0.8.",
    )
    parser.add_argument(
        "--ip_pair_min_level",
        type=str,
        default="artist",
        choices=["character", "copyright", "artist"],
        help="Loosest tier the identity-pair sampler will back off to before "
        "self-pairing. 'character' = same-character only; 'artist' (default) = "
        "full character → franchise → artist back-off.",
    )
    parser.add_argument(
        "--ip_pair_caption_strip_p",
        type=float,
        default=0.0,
        help="Probability of dropping the target's character/copyright tokens "
        "from the caption on distinct-pair steps, forcing identity through the IP "
        "image path rather than the text. INERT while use_text_cache=true "
        "(the cached embedding still carries identity) — set it false to enable. "
        "Default 0.0 (off).",
    )
    parser.add_argument(
        "--use_easycontrol",
        action="store_true",
        help="Enable EasyControl image conditioning (extended self-attn KV with "
        "VAE-encoded reference). Requires the network module to expose set_cond_tokens "
        "(e.g. networks.methods.easycontrol). The cond input is the clean VAE latent of "
        "the reference image; for ref==target training (Phase 1) this reuses the "
        "existing VAE-cache output — no new sidecar required.",
    )
    parser.add_argument(
        "--easycontrol_drop_p",
        type=float,
        default=0.1,
        help="EasyControl image-conditioning dropout probability per batch (CFG dropout "
        "for image branch). Independent of text-side caption_dropout_rate; default 0.1.",
    )
    parser.add_argument(
        "--easycontrol_cond_noise_max",
        type=float,
        default=0.0,
        help="Per-step training-only additive Gaussian noise on the cond latent before "
        "set_cond. Sampled per-sample as sigma ~ U(0, this) and applied as "
        "cond + sigma * eps. 0.0 disables (default = current ref==target behavior). "
        "Use a small positive value (e.g. 0.3-0.7) to weaken cond's blueprint "
        "dominance and force text to carry the high-frequency residual; sigma=0 "
        "stays in the training distribution so clean-cond inference still works.",
    )
    parser.add_argument(
        "--use_shuffled_caption_variants",
        action="store_true",
        help="Consume preprocessed caption-shuffle variants from the text-encoder cache "
        "(written by `cache_text_embeddings.py --caption_shuffle_variants N`). When the "
        "cache file has a `num_variants` tensor, a random variant is drawn per sample. "
        "Falls back to single-variant silently if the cache has no variants. Inline "
        "multi-variant generation at training time is no longer supported — preprocess "
        "the variants first.",
    )
    parser.add_argument(
        "--artist_filter",
        type=str,
        default=None,
        help="If set, only train on images whose caption contains the `@<artist>` tag. "
        "Pass with or without the leading `@` (e.g. `--artist_filter sincos`). "
        "Output is redirected to output/ckpt-artist/<output_name>_<artist>.safetensors.",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=1.0,
        help="Timestep distribution shift for rectified flow training (default: 1.0)",
    )
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        default="sigmoid",
        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift"],
        help="Timestep sampling method (default: sigmoid (logit normal))",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help="Scale factor for sigmoid (logit_normal) timestep sampling (default: 1.0)",
    )
    parser.add_argument(
        "--sigmoid_bias",
        type=float,
        default=0.0,
        help="Logit-space mean shift for sigmoid/shift/flux_shift sampling: "
        "sigmoid(sigmoid_scale*randn + sigmoid_bias). >0 skews mass toward high σ "
        "(structure regime), <0 toward low σ. 0.0 (default) = unbiased logit-normal.",
    )
    parser.add_argument(
        "--lora_fp32_accumulation",
        action="store_true",
        help="[DEPRECATED, no-op] fp32 accumulation is now unconditional in "
        "LoRA/Hydra/ReFT bottleneck matmuls. Flag accepted for one release "
        "cycle; will be removed.",
    )
    parser.add_argument(
        "--attn_mode",
        choices=[
            "torch",
            "mem_efficient",
            "xformers",
            "flash",
            # "flash4",  # not supported yet (flash-attention-sm120 disabled)
            "sageattn",
            "flex",
            "sdpa",
        ],  # "sdpa" is for backward compatibility
        default=None,
        help="Attention implementation to use. mem_efficient forces PyTorch's "
        "efficient SDPA backend for each call; torch lets PyTorch choose. "
        "Default is None (torch). sageattn does not support training (inference only). "
        "This option overrides --xformers or --sdpa."
        "",
    )
    parser.add_argument(
        "--attn_softmax_scale",
        type=float,
        default=None,
        help="Custom softmax scale for attention (default: 1/sqrt(head_dim)). Larger values sharpen attention and improve bf16 precision."
        " For head_dim=128, default is ~0.088. Try 0.1-0.15 for better low-precision stability.",
    )
    parser.add_argument(
        "--v100_flash_stability",
        choices=["off", "hybrid", "safe"],
        default=None,
        help="V100 Flash diagnostic mode: off uses full Flash, hybrid routes "
        "cross-attention to torch SDPA, and safe enables fail-fast finite checks. "
        "ANIMA_V100_FLASH_STABILITY may provide the default.",
    )
    parser.add_argument(
        "--debug_finite_checks",
        action="store_true",
        help="Fail fast when attention tensors, block residuals, loss, or "
        "gradients contain NaN/Inf. Also enabled by ANIMA_DEBUG_FINITE=1.",
    )
    parser.add_argument(
        "--vae_chunk_size",
        type=int,
        default=None,
        help="Spatial chunk size for VAE encoding/decoding to reduce memory usage. Must be even number. If not specified, chunking is disabled (official behavior)."
        + "",
    )
    parser.add_argument(
        "--vae_disable_cache",
        action="store_true",
        help="Disable internal VAE caching mechanism to reduce memory usage. Encoding / decoding will also be faster, but this differs from official behavior."
        + "",
    )

    # EMA arguments
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Enable Exponential Moving Average. Requires --ema_decay to be set.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay rate. Typical values: 0.999, 0.9999. (default: 0.9999)",
    )
    parser.add_argument(
        "--ema_device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device for EMA shadow parameters. 'cpu' saves GPU VRAM but slower update. (default: gpu)",
    )
    parser.add_argument(
        "--ema_use_num_updates",
        action="store_true",
        help="Use warmup schedule for EMA decay: min(decay, (1+num_updates)/(10+num_updates))",
    )
    parser.add_argument(
        "--ema_use_feedback",
        action="store_true",
        help="Feed back EMA decay to training parameters (experimental)",
    )
    parser.add_argument(
        "--ema_param_multiplier",
        type=float,
        default=1.0,
        help="Multiply parameters each EMA update step (default: 1.0, no effect)",
    )
    parser.add_argument(
        "--ema_resume_path",
        type=str,
        default=None,
        help="Path to EMA model safetensors file to resume EMA state from a previous run",
    )

    # Variance-reduced flow-matching loss (AsymFlow §5.2, arXiv:2605.12964).
    # See bench/fm_vr_headroom/proposal.md. Gated off by default.
    parser.add_argument(
        "--vr_loss_weight",
        type=float,
        default=0.0,
        help="Weight of the VR control-variate correction on the FM loss. "
        "0 = standard FM (default). 1.0 = paper recipe. When > 0, the trainer "
        "runs one extra no-grad forward per step on the FEI-low-passed latent "
        "through the *same* trainable DiT with the adapter zeroed "
        "(network.set_multiplier(0)) — equivalent to a frozen base DiT for "
        "LoRA-family runs, without holding a second model copy in VRAM.",
    )
    parser.add_argument(
        "--vr_fei_sigma_low_div",
        type=float,
        default=4.0,
        help="Divisor for the FEI low-pass kernel used to build x_0^L "
        "(σ_low = min(H_lat, W_lat) / div). Matches live FEI default (4.0).",
    )
    parser.add_argument(
        "--vr_sigma_min",
        type=float,
        default=1e-3,
        help="Floor on σ_t in the VR loss denominator (AsymFlow §6.1). "
        "Defensive against low-σ instability in the 1/σ_t factor. 0 disables.",
    )
    parser.add_argument(
        "--vr_lambda_beta",
        type=float,
        default=0.01,
        help="EMA rate for the online λ estimator: "
        "λ_ema ← (1−β)·λ_ema + β·λ_batch. Default 0.01 over typically B=4.",
    )

    # Functional MSE loss against inversion runs (postfix-func)
    parser.add_argument(
        "--inversion_dir",
        type=str,
        default=None,
        help="Directory containing <stem>_inverted_run{i}.safetensors files. If set, "
        "inversions are loaded alongside each sample for functional-loss supervision.",
    )
    parser.add_argument(
        "--functional_loss_weight",
        type=float,
        default=0.0,
        help="Weight for functional MSE loss: MSE between cross_attn output_proj captures "
        "from (t5+postfix) forward and (inversion) forward, summed over functional_loss_blocks. "
        "0 disables (default).",
    )
    parser.add_argument(
        "--functional_loss_blocks",
        type=str,
        default="8,12,16,20",
        help="Comma-separated DiT block indices at which to capture cross_attn.output_proj "
        "outputs for functional MSE loss.",
    )
    parser.add_argument(
        "--functional_loss_num_runs",
        type=int,
        default=3,
        help="Number of inversion runs expected per image (aggregate_by of make invert). "
        "Each training step samples one run stochastically. Default: 3.",
    )


# Loss weighting


def compute_loss_weighting_for_anima(
    weighting_scheme: str,
    sigmas: torch.Tensor,
    *,
    min_snr_gamma: float | None = None,
    p2_gamma: float = 1.0,
    p2_k: float = 1.0,
) -> torch.Tensor:
    """Compute loss weighting for Anima training.

    `sigmas` are Anima's rectified-flow σ in [0,1], where
    x_σ = (1-σ)·x0 + σ·ε. For SNR-style schemes we use the natural RF analogue:

        SNR(σ) = ((1 - σ) / σ)²
    """
    from library.training.flow_step import compute_flow_loss_weighting

    return compute_flow_loss_weighting(
        weighting_scheme,
        sigmas,
        min_snr_gamma=min_snr_gamma,
        p2_gamma=p2_gamma,
        p2_k=p2_k,
    )


# Parameter groups (6 groups with separate LRs)
def get_anima_param_groups(
    dit,
    base_lr: float,
    self_attn_lr: Optional[float] = None,
    cross_attn_lr: Optional[float] = None,
    mlp_lr: Optional[float] = None,
    mod_lr: Optional[float] = None,
    llm_adapter_lr: Optional[float] = None,
):
    """Create parameter groups for Anima training with separate learning rates.

    Args:
        dit: Anima model
        base_lr: Base learning rate
        self_attn_lr: LR for self-attention layers (None = base_lr, 0 = freeze)
        cross_attn_lr: LR for cross-attention layers
        mlp_lr: LR for MLP layers
        mod_lr: LR for AdaLN modulation layers
        llm_adapter_lr: LR for LLM adapter

    Returns:
        List of parameter group dicts for optimizer
    """
    if self_attn_lr is None:
        self_attn_lr = base_lr
    if cross_attn_lr is None:
        cross_attn_lr = base_lr
    if mlp_lr is None:
        mlp_lr = base_lr
    if mod_lr is None:
        mod_lr = base_lr
    if llm_adapter_lr is None:
        llm_adapter_lr = base_lr

    base_params = []
    self_attn_params = []
    cross_attn_params = []
    mlp_params = []
    mod_params = []
    llm_adapter_params = []

    for name, p in dit.named_parameters():
        # Store original name for debugging
        p.original_name = name

        if "llm_adapter" in name:
            llm_adapter_params.append(p)
        elif ".self_attn" in name:
            self_attn_params.append(p)
        elif ".cross_attn" in name:
            cross_attn_params.append(p)
        elif ".mlp" in name:
            mlp_params.append(p)
        elif ".adaln_modulation" in name:
            mod_params.append(p)
        else:
            base_params.append(p)

    logger.info("Parameter groups:")
    logger.info(f"  base_params: {len(base_params)} (lr={base_lr})")
    logger.info(f"  self_attn_params: {len(self_attn_params)} (lr={self_attn_lr})")
    logger.info(f"  cross_attn_params: {len(cross_attn_params)} (lr={cross_attn_lr})")
    logger.info(f"  mlp_params: {len(mlp_params)} (lr={mlp_lr})")
    logger.info(f"  mod_params: {len(mod_params)} (lr={mod_lr})")
    logger.info(
        f"  llm_adapter_params: {len(llm_adapter_params)} (lr={llm_adapter_lr})"
    )

    param_groups = []
    for lr, params, name in [
        (base_lr, base_params, "base"),
        (self_attn_lr, self_attn_params, "self_attn"),
        (cross_attn_lr, cross_attn_params, "cross_attn"),
        (mlp_lr, mlp_params, "mlp"),
        (mod_lr, mod_params, "mod"),
        (llm_adapter_lr, llm_adapter_params, "llm_adapter"),
    ]:
        if lr == 0:
            for p in params:
                p.requires_grad_(False)
            logger.info(f"  Frozen {name} params ({len(params)} parameters)")
        elif len(params) > 0:
            param_groups.append({"params": params, "lr": lr})

    total_trainable = sum(
        p.numel() for group in param_groups for p in group["params"] if p.requires_grad
    )
    logger.info(f"Total trainable parameters: {total_trainable:,}")

    return param_groups


# EMA save helpers


def _get_ema_filename(ckpt_file):
    """Get EMA filename by adding ema_ prefix to the basename of a checkpoint file."""
    dirpath = os.path.dirname(ckpt_file)
    basename = os.path.basename(ckpt_file)
    return os.path.join(dirpath, f"ema_{basename}")


def _remove_old_ema_file(old_ckpt_file):
    """Remove old EMA file corresponding to an old checkpoint file (for save_last_n cleanup)."""
    if old_ckpt_file is None:
        return
    old_ema_file = _get_ema_filename(old_ckpt_file)
    if os.path.exists(old_ema_file):
        logger.info(f"removing old EMA checkpoint: {old_ema_file}")
        os.remove(old_ema_file)


def _save_ema_model(ema, dit, ckpt_file, sai_metadata, save_dtype):
    """Save EMA model as standard format with ema_ prefix.

    Builds the EMA state dict directly from shadow params without mutating model
    parameters in place — keeps stepwise save side-effect-free on the live model.
    """
    if ema.shadow_params is None:
        logger.warning("EMA shadow_params is None (worker process?), skipping EMA save")
        return

    ema_file = _get_ema_filename(ckpt_file)

    dit_sd = dit.state_dict()
    trainable_idx = 0
    for name, p in dit.named_parameters():
        if p.requires_grad:
            dit_sd[name] = ema.shadow_params[trainable_idx].data.to(p.device)
            trainable_idx += 1

    anima_utils.save_anima_model(ema_file, dit_sd, sai_metadata, save_dtype)
    logger.info(f"EMA model saved: {ema_file}")
    del dit_sd


# Save functions
def save_anima_model_on_train_end(
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    dit: anima_models.Anima,
    ema=None,
):
    """Save Anima model at the end of training."""

    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec_dataclass(
            args, lora=False
        ).to_metadata_dict()
        dit_sd = dit.state_dict()
        # Save with 'net.' prefix for ComfyUI compatibility
        anima_utils.save_anima_model(ckpt_file, dit_sd, sai_metadata, save_dtype)
        if ema is not None:
            _save_ema_model(ema, dit, ckpt_file, sai_metadata, save_dtype)

    save_sd_model_on_train_end_common(
        args, True, True, epoch, global_step, sd_saver, None
    )


def save_anima_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator: Accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    dit: anima_models.Anima,
    ema=None,
):
    """Save Anima model at epoch end or specific steps."""

    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec_dataclass(
            args, lora=False
        ).to_metadata_dict()
        dit_sd = dit.state_dict()
        anima_utils.save_anima_model(ckpt_file, dit_sd, sai_metadata, save_dtype)

        if ema is not None:
            _save_ema_model(ema, dit, ckpt_file, sai_metadata, save_dtype)

            # Clean up old EMA files (train_util only cleans normal model files)
            ext = ".safetensors"
            if on_epoch_end:
                remove_no = get_remove_epoch_no(args, epoch_no)
                if remove_no is not None:
                    old_ckpt_name = get_epoch_ckpt_name(args, ext, remove_no)
                    _remove_old_ema_file(os.path.join(args.output_dir, old_ckpt_name))
            else:
                remove_no = get_remove_step_no(args, global_step)
                if remove_no is not None:
                    old_ckpt_name = get_step_ckpt_name(args, ext, remove_no)
                    _remove_old_ema_file(os.path.join(args.output_dir, old_ckpt_name))

    save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        True,
        True,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        None,
    )


# Sampling for rectified flow training previews
def do_sample(
    height: int,
    width: int,
    seed: Optional[int],
    dit: anima_models.Anima,
    crossattn_emb: torch.Tensor,
    steps: int,
    dtype: torch.dtype,
    device: torch.device,
    guidance_scale: float = 1.0,
    flow_shift: float = 3.0,
    neg_crossattn_emb: Optional[torch.Tensor] = None,
    sampler: str = "euler",
    show_progress: bool = True,
    step_start_callback=None,
) -> torch.Tensor:
    """Generate a sample using Anima flow-matching sampling.

    Args:
        height, width: Output image dimensions
        seed: Random seed (None for random)
        dit: Anima model
        crossattn_emb: Cross-attention embeddings (B, N, D)
        steps: Number of sampling steps
        dtype: Compute dtype
        device: Compute device
        guidance_scale: CFG scale (1.0 = no guidance)
        flow_shift: Flow shift parameter for rectified flow
        neg_crossattn_emb: Negative cross-attention embeddings for CFG

    Returns:
        Denoised latents
    """
    # Latent shape: (1, 16, 1, H/8, W/8) for single image
    latent_h = height // 8
    latent_w = width // 8
    latent = torch.zeros(1, 16, 1, latent_h, latent_w, device=device, dtype=dtype)

    # Generate noise
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    noise = (
        torch.randn(
            latent.size(), dtype=torch.float32, generator=generator, device="cpu"
        )
        .to(dtype)
        .to(device)
    )

    sampler = normalize_training_sample_sampler(sampler)
    _timesteps, sigmas = inference_sampling.get_timesteps_sigmas(
        steps,
        float(flow_shift),
        device,
    )
    sigmas = sigmas.to(device=device, dtype=dtype)
    sampler_stepper = None
    if sampler == "er_sde":
        sampler_stepper = inference_sampling.ERSDESampler(
            sigmas.detach().float().to(device),
            seed=seed,
            device=device,
        )
    elif sampler == "lcm":
        sampler_stepper = inference_sampling.LCMSampler(
            sigmas.detach().float().to(device),
            seed=seed,
            device=device,
        )

    # Start from pure noise
    x = noise.clone()

    # Padding mask (zeros = no padding) — resized in prepare_embedded_sequence to match latent dims
    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=dtype, device=device)

    use_cfg = guidance_scale > 1.0 and neg_crossattn_emb is not None

    preview_net = getattr(dit, "_preview_network", None)
    for i in tqdm(range(steps), desc="Sampling", disable=not show_progress):
        sigma = sigmas[i]
        t = sigma.unsqueeze(0)  # (1,)
        if preview_net is not None:
            apply_preview_router_conditioning(
                preview_net, x, t, step_index=i, crossattn_emb=crossattn_emb
            )
        if step_start_callback is not None:
            step_start_callback(i, x, t)

        if use_cfg:
            # CFG: two separate passes to reduce memory usage
            pos_out = dit(x, t, crossattn_emb, padding_mask=padding_mask)
            pos_out = pos_out.float()
            if preview_net is not None:
                apply_preview_router_conditioning(
                    preview_net, x, t, step_index=i, crossattn_emb=neg_crossattn_emb
                )
            neg_out = dit(x, t, neg_crossattn_emb, padding_mask=padding_mask)
            neg_out = neg_out.float()

            model_output = neg_out + guidance_scale * (pos_out - neg_out)
        else:
            model_output = dit(x, t, crossattn_emb, padding_mask=padding_mask)
            model_output = model_output.float()

        if sampler_stepper is None:
            x = inference_sampling.step(x, model_output, sigmas, i)
        else:
            denoised = x.float() - sigma.float() * model_output.float()
            x = sampler_stepper.step(x, denoised, i)
        x = x.to(dtype)

    return x


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    dit: anima_models.Anima,
    vae,
    text_encoder,
    tokenize_strategy,
    text_encoding_strategy,
    sample_prompts_te_outputs=None,
    prompt_replacement=None,
    network=None,
    sample_prompts_snapshot=None,
):
    """Generate sample images during training.

    This is a simplified sampler for Anima - it generates images using the current model state.

    Mirrors the validation phase's model handling (see
    ``library/training/validation.py::run_validation``): the adapter ``network``
    is put in ``eval()`` for the duration (so dropout / rank-dropout match how
    samples would be generated at inference) and restored to ``train()``
    afterward, reusing the live in-GPU model rather than reloading anything. The
    optimizer eval/train swap is handled by the caller in ``loop.py``.
    """
    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        sample_this_call = False
        if epoch is not None and args.sample_every_n_epochs is not None:
            sample_this_call = epoch % args.sample_every_n_epochs == 0
        if epoch is None and args.sample_every_n_steps is not None:
            sample_this_call = steps % args.sample_every_n_steps == 0
        if not sample_this_call:
            return

    process_index = int(getattr(accelerator, "process_index", 0))
    num_processes = max(int(getattr(accelerator, "num_processes", 1)), 1)
    if getattr(accelerator, "is_main_process", True):
        logger.info(
            "Generating sample images at step %s across %s process(es)",
            steps,
            num_processes,
        )
    use_cached_prompt_snapshot = (
        sample_prompts_snapshot is not None
        and sample_prompts_te_outputs is not None
        and text_encoder is None
    )
    prompt_file_missing = not use_cached_prompt_snapshot and not os.path.isfile(
        args.sample_prompts
    )
    if _sample_failed_on_any_process(
        accelerator, prompt_file_missing, num_processes
    ):
        if getattr(accelerator, "is_main_process", True):
            logger.error(
                "Sample prompt file is unavailable on at least one process: %s",
                args.sample_prompts,
            )
        return

    prompt_error: Optional[Exception] = None
    prompts = []
    try:
        if use_cached_prompt_snapshot:
            prompts = [dict(prompt) for prompt in sample_prompts_snapshot]
        else:
            prompts = train_util.load_prompts(args.sample_prompts)
    except Exception as exc:  # keep all ranks on the same collective path
        prompt_error = exc
    if _sample_failed_on_any_process(
        accelerator, prompt_error is not None, num_processes
    ):
        if prompt_error is not None:
            raise prompt_error
        raise RuntimeError("Sample prompt loading failed on another process")

    local_prompts = prompts[process_index::num_processes]

    # Unwrap models
    dit = accelerator.unwrap_model(dit)
    if text_encoder is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)
    net = accelerator.unwrap_model(network) if network is not None else None

    # Before the block-swap pause below: pause_block_swap() zeroes
    # blocks_to_swap, and a recompile under that would treat every block as
    # resident and compile the swapped tail too — silently widening
    # compile_block_scope="resident" for the rest of the run.
    compile_error: Optional[Exception] = None
    try:
        _ensure_sample_compile_range(dit, net, prompts)
    except Exception as exc:  # keep all ranks on the same collective path
        compile_error = exc
    if _sample_failed_on_any_process(
        accelerator, compile_error is not None, num_processes
    ):
        if compile_error is not None:
            raise compile_error
        raise RuntimeError("Sample compile preparation failed on another process")

    def run_local():
        with preview_lifecycle(
            dit, net, disable_block_swap=getattr(args, "disable_block_swap_for_eval", False)
        ):
            save_dir = os.path.join(args.output_dir, "sample")
            os.makedirs(save_dir, exist_ok=True)
            with torch.no_grad(), accelerator.autocast():
                for prompt_dict in local_prompts:
                    _sample_image_inference(
                        accelerator,
                        args,
                        dit,
                        text_encoder,
                        vae,
                        tokenize_strategy,
                        text_encoding_strategy,
                        save_dir,
                        prompt_dict,
                        epoch,
                        steps,
                        sample_prompts_te_outputs,
                        prompt_replacement,
                    )

    finish_preview_batch(
        accelerator,
        args,
        processes=num_processes,
        run_local=run_local,
        failure_message="Sample generation failed on another process",
        vae=vae,
        dit=dit,
        network=net,
    )


def _sample_failed_on_any_process(
    accelerator: Accelerator,
    failed: bool,
    num_processes: int,
) -> bool:
    if num_processes <= 1:
        return failed
    failure_count = accelerator.reduce(
        torch.tensor(
            [int(failed)],
            device=accelerator.device,
            dtype=torch.int32,
        ),
        reduction="sum",
    )
    return bool(failure_count.item())


def _sample_image_inference(
    accelerator,
    args,
    dit,
    text_encoder,
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    tokenize_strategy,
    text_encoding_strategy,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    """Generate a single sample image."""
    prompt = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get("negative_prompt", "")
    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    scale = prompt_dict.get("guidance_scale", prompt_dict.get("scale", 7.5))
    seed = prompt_dict.get("seed")
    if seed is None:
        # No explicit `--d` on this prompt: pin to a stable per-prompt seed so
        # the same prompt renders from identical noise at every epoch — that's
        # what makes the per-epoch gallery comparable for spotting overfitting.
        # Offset by the prompt's index so different prompts still get distinct
        # noise. Falls back to 0 if the run somehow has no seed.
        base_seed = getattr(args, "seed", None)
        seed = (base_seed if base_seed is not None else 0) + int(
            prompt_dict.get("enum", 0)
        )
    flow_shift = prompt_dict.get("flow_shift", 3.0)
    sample_sampler = normalize_training_sample_sampler(
        prompt_dict.get("sample_sampler", getattr(args, "sample_sampler", "euler"))
    )

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt:
            negative_prompt = negative_prompt.replace(
                prompt_replacement[0], prompt_replacement[1]
            )

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # seed all CUDA devices for multi-GPU

    width, height = snap_sample_size(width, height)

    # Train-time sampling runs through the same compiled blocks as training.
    # sample_images() widens the compile range before this point when prompt
    # files gain new resolutions mid-run; this guard remains as the final
    # no-crash fallback for unusual compile states (a ConstraintViolationError
    # would otherwise take down the whole run).
    seq_len = (width // 16) * (height // 16)
    seq_range = getattr(dit, "_dynamic_seq_range", None)
    seq_bands = getattr(dit, "_dynamic_seq_bands", None) or (
        [seq_range] if seq_range is not None else None
    )
    from library.datasets.buckets import band_for_seq

    if (
        getattr(dit, "_dynamic_seq", False)
        and seq_bands is not None
        and band_for_seq(seq_bands, seq_len) is None
    ):
        logger.warning(
            f"Skipping sample prompt at {width}x{height} ({seq_len} tokens): outside "
            f"the compiled dynamic-seq token band(s) {seq_bands}. Automatic range "
            "expansion was unavailable for this compile state; lower --w/--h, "
            "restart training, or disable torch_compile."
        )
        return

    logger.info(
        f"  prompt: {prompt}, size: {width}x{height}, steps: {sample_steps}, scale: {scale}, flow_shift: {flow_shift}, sampler: {sample_sampler}, seed: {seed}"
    )

    # Encode prompt
    def encode_prompt(prpt):
        if sample_prompts_te_outputs and prpt in sample_prompts_te_outputs:
            return sample_prompts_te_outputs[prpt]
        if text_encoder is not None:
            tokens = tokenize_strategy.tokenize(prpt)
            encoded = text_encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens
            )
            return encoded
        return None

    encoded = encode_prompt(prompt)
    if encoded is None:
        logger.warning("Cannot encode prompt, skipping sample")
        return

    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = encoded

    # Convert to tensors if numpy
    if isinstance(prompt_embeds, np.ndarray):
        prompt_embeds = torch.from_numpy(prompt_embeds).unsqueeze(0)
        attn_mask = torch.from_numpy(attn_mask).unsqueeze(0)
        t5_input_ids = torch.from_numpy(t5_input_ids).unsqueeze(0)
        t5_attn_mask = torch.from_numpy(t5_attn_mask).unsqueeze(0)

    prompt_embeds = prompt_embeds.to(accelerator.device, dtype=dit.dtype)
    attn_mask = attn_mask.to(accelerator.device)
    t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
    t5_attn_mask = t5_attn_mask.to(accelerator.device)

    # Process through LLM adapter if available
    if dit.use_llm_adapter:
        crossattn_emb = dit.llm_adapter(
            source_hidden_states=prompt_embeds,
            target_input_ids=t5_input_ids,
            target_attention_mask=t5_attn_mask,
            source_attention_mask=attn_mask,
        )
        crossattn_emb[~t5_attn_mask.bool()] = 0
        # Pad to 512 tokens (model expects fixed-length context)
        if crossattn_emb.shape[1] < 512:
            crossattn_emb = torch.nn.functional.pad(
                crossattn_emb, (0, 0, 0, 512 - crossattn_emb.shape[1])
            )
    else:
        crossattn_emb = prompt_embeds

    # Encode negative prompt for CFG
    neg_crossattn_emb = None
    if scale > 1.0 and negative_prompt is not None:
        neg_encoded = encode_prompt(negative_prompt)
        if neg_encoded is not None:
            neg_pe, neg_am, neg_t5_ids, neg_t5_am = neg_encoded
            if isinstance(neg_pe, np.ndarray):
                neg_pe = torch.from_numpy(neg_pe).unsqueeze(0)
                neg_am = torch.from_numpy(neg_am).unsqueeze(0)
                neg_t5_ids = torch.from_numpy(neg_t5_ids).unsqueeze(0)
                neg_t5_am = torch.from_numpy(neg_t5_am).unsqueeze(0)

            neg_pe = neg_pe.to(accelerator.device, dtype=dit.dtype)
            neg_am = neg_am.to(accelerator.device)
            neg_t5_ids = neg_t5_ids.to(accelerator.device, dtype=torch.long)
            neg_t5_am = neg_t5_am.to(accelerator.device)

            if dit.use_llm_adapter:
                neg_crossattn_emb = dit.llm_adapter(
                    source_hidden_states=neg_pe,
                    target_input_ids=neg_t5_ids,
                    target_attention_mask=neg_t5_am,
                    source_attention_mask=neg_am,
                )
                neg_crossattn_emb[~neg_t5_am.bool()] = 0
                # Pad to 512 tokens (model expects fixed-length context)
                if neg_crossattn_emb.shape[1] < 512:
                    neg_crossattn_emb = torch.nn.functional.pad(
                        neg_crossattn_emb, (0, 0, 0, 512 - neg_crossattn_emb.shape[1])
                    )
            else:
                neg_crossattn_emb = neg_pe

    # Generate sample. Deliberately no empty_cache here: freed tensors return
    # to the caching allocator and get reused on the next prompt / when training
    # resumes, so VRAM stays flat across the train<->sample transition instead
    # of the shrink-then-regrow churn that torch.cuda.empty_cache() causes.
    latents = do_sample(
        height,
        width,
        seed,
        dit,
        crossattn_emb,
        sample_steps,
        dit.dtype,
        accelerator.device,
        scale,
        flow_shift,
        neg_crossattn_emb,
        sample_sampler,
    )

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i = prompt_dict.get("enum", 0)
    stem = (
        f"{'' if args.output_name is None else args.output_name + '_'}"
        f"{num_suffix}_{i:02d}_{ts_str}{seed_suffix}"
    )
    latents_dir = os.path.join(save_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)
    latent_path = os.path.join(latents_dir, stem + ".pt")
    torch.save(
        {"latents": latents.detach().to("cpu"), "prompt": prompt, "enum": i},
        latent_path,
    )
    return latent_path


def decode_samples_for_live_preview(
    accelerator: Accelerator,
    args,
    vae,
    *,
    dit=None,
    network=None,
) -> None:
    """Compatibility shim for the family-neutral live decode lifecycle."""
    from library.training.preview_live_decode import decode_live_preview

    return decode_live_preview(
        accelerator,
        args,
        vae,
        dit=dit,
        network=network,
        decode_pending_samples=decode_pending_samples,
    )


def decode_anima_pending_samples(accelerator: Accelerator, args, vae) -> None:
    """Decode Anima-layout sample latents into PNGs.

    Used both for live previews during training and as a final stop/end cleanup.
    Applies the selected VAE memory policy and decodes every
    ``output_dir/sample/latents/*.pt`` into a PNG in ``output_dir/sample/``,
    then parks the VAE back. Each latent file is removed after a successful
    decode; a failed one is left on disk so it can be recovered. No-op when
    sampling was disabled or the VAE isn't available.
    """
    if vae is None:
        return
    from library.training.preview_decode import decode_anima_latent, decode_pending_samples as decode_shared_samples

    decode_shared_samples(accelerator, args, vae, decode_latent=decode_anima_latent, wandb_enabled=True)


def decode_pending_samples(accelerator: Accelerator, args, vae) -> None:
    """Compatibility entrypoint; dispatch through the active family plugin."""
    from library.training.preview_live_decode import decode_pending_samples_for_family

    return decode_pending_samples_for_family(accelerator, args, vae)


def sample_image_to_tensor(
    *,
    accelerator: Accelerator,
    dit: anima_models.Anima,
    vae,
    height: int,
    width: int,
    crossattn_emb: torch.Tensor,
    neg_crossattn_emb: Optional[torch.Tensor] = None,
    sample_steps: int = 20,
    guidance_scale: float = 4.0,
    flow_shift: float = 3.0,
    seed: Optional[int] = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Sample one image and return the decoded pixel tensor in ``[-1, 1]``.

    Sibling of :func:`_sample_image_inference` that skips disk I/O and the
    PIL conversion. Returned tensor is ``[3, H, W]`` float (on
    ``accelerator.device``), suitable for direct PE-Core encoding.

    ``crossattn_emb`` is the prepared cross-attention embedding (already
    LLM-adapter'd and padded to the model's expected length); CMMD val
    builds it from cached TE outputs so we don't re-run the text encoder.
    """
    height = max(64, height - height % 16)
    width = max(64, width - width % 16)
    crossattn_emb = crossattn_emb.to(accelerator.device, dtype=dit.dtype)
    if neg_crossattn_emb is not None:
        neg_crossattn_emb = neg_crossattn_emb.to(accelerator.device, dtype=dit.dtype)

    clean_memory_on_device(accelerator.device)
    latents = do_sample(
        height,
        width,
        seed,
        dit,
        crossattn_emb,
        sample_steps,
        dit.dtype,
        accelerator.device,
        guidance_scale,
        flow_shift,
        neg_crossattn_emb,
        "euler",
        show_progress=show_progress,
    )

    gc.collect()
    synchronize_device(accelerator.device)
    clean_memory_on_device(accelerator.device)
    org_vae_device = vae.device
    vae.to(accelerator.device)
    decoded = vae.decode_to_pixels(latents)
    vae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)

    image = decoded.float()[0]
    if image.ndim == 4:
        # Drop temporal dim if the VAE returned [C, T, H, W].
        image = image[:, 0, :, :]
    return image.clamp(-1.0, 1.0)
