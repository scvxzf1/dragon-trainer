# Krea-2-Raw 推理 runner (阶段 6 配置收口 — 推理侧 family dispatch)

# 闭合记忆/AGENTS.md 标注的「generation.py denoising family dispatch 未闭合」缺口:
# 推理 CLI/WebUI 在 family=krea2_raw 时不再走 anima 的 load_anima_model (会因
# Anima 与 SingleStreamDiT 结构不同报 missing keys), 而是走本模块的独立路径。

# 反上帝守则: 所有 Krea-2 推理新逻辑集中在本模块, generation.py / inference.py
# 只加薄 family dispatch (十几行), 不把 Krea-2 采样/文本/LoRA 逻辑堆进热点文件。

# 蓝本: scripts/krea2/probe_sample.py (阶段 5 验证的完整采样序列). 复用既有:
#   - library.models.krea2_raw.weights.load_krea2_dit (strict 加载原生 key)
#   - library.models.krea2_raw.strategy.{load_krea2_text_encoder, Krea2*Strategy}
#   - library.models.krea2_raw.family.{Krea2TextEmbedding, forward_for_loss}
#   - library.models.krea2_raw.sampling.sample (mu-shift Euler ODE + CFG)
#   - networks.lora_anima.create_network_from_weights (读 ss_model_family stamp
#     自动构造 Krea-2 target spec, 复用 anima LoRA network, apply_to 透明)
#
# Krea-2 首日非目标 (提案 §1, 记忆确认): IP-Adapter / Spectrum / SPD / tiled
# diffusion / DCW / mod-guidance / soft-tokens / P-GRAFT / Hydra — 这些 anima-only
# 旁路在 krea2_raw family 下直接拒绝, 不静默跑错。VAE decode 复用 anima 的
# qwen_vae (R2 验证同一 AutoencoderKLQwenImage), 出图契约与 anima 一致 ([-1,1]).

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Tuple

import torch

from library.inference.family_output import save_with_shared_qwen_vae
from library.inference.family_runner import FamilyInferenceRunner
from library.inference.precision import resolve_text_encoder_dtype
from library.models.krea2_raw.dit import SingleStreamDiT
from library.models.krea2_raw.family import Krea2TextEmbedding, forward_for_loss
from library.models.krea2_raw.sampling import sample
from library.models.krea2_raw.strategy import (
    Krea2TextEncodingStrategy,
    Krea2TokenizeStrategy,
    load_krea2_text_encoder,
)
from library.models.krea2_raw.attention_backend import (
    prepare_krea2_attention,
    validate_krea2_attention_mode,
)
from library.models.krea2_raw.weights import load_krea2_dit
from library.runtime.device import clean_memory_on_device
from library.training.preview_lifecycle import apply_preview_router_conditioning

logger = logging.getLogger(__name__)


class Krea2InferenceRunner:
    family = "krea2_raw"
    supports_reference_conditioning = False

    @staticmethod
    def validate(args, *, mode: str = "single") -> None:
        validate_krea2_inference_args(args, mode=mode)

    @staticmethod
    def prepare(args, *, mode: str = "single") -> None:
        return None

    def generate_latent(self, args, gen_settings):
        self.validate(args, mode="single")
        device = gen_settings.device
        runtime_dtype = gen_settings.runtime_dtype
        runtime = FamilyInferenceRunner(
            self.family,
            device,
            cleanup=clean_memory_on_device,
        )

        cond_emb, uncond_emb = runtime.run_text(
            lambda: load_krea2_text_encoder_for_inference(args, device),
            lambda bundle: encode_krea2_prompts(
                args,
                bundle,
                device,
                runtime_dtype,
            ),
        )
        return runtime.run_stage(
            "dit",
            lambda: load_krea2_dit_for_inference(
                args,
                device,
                runtime_dtype,
            ),
            lambda loaded: generate_krea2(
                args,
                loaded[0],
                loaded[1],
                cond_emb,
                uncond_emb,
                device,
                args.seed,
                runtime_dtype,
            ),
        )

    def save_output(self, args, latent, device):
        return save_with_shared_qwen_vae(
            args,
            latent,
            device,
            family=self.family,
        )


# Krea-2 首日不支持的 anima-only 推理旁路. 触及即拒绝, 不静默跑错.
_KREA2_UNSUPPORTED_ANIMA_ONLY = {
    "pgraft": "--pgraft (P-GRAFT mid-denoising cutoff)",
    "spectrum": "--spectrum (Chebyshev feature forecasting)",
    "spd": "--spd (staged prompt-driven)",
    "tiled_diffusion": "--tiled_diffusion (tiled denoising)",
    "dcw": "--dcw (bias correction fusion head)",
    "pooled_text_proj": "--pooled_text_proj (modulation guidance)",
    "ip_adapter_weight": "--ip_adapter_weight (IP-Adapter)",
    "easycontrol_weight": "--easycontrol_weight (EasyControl)",
    "soft_tokens_weight": "--soft_tokens_weight (Soft Tokens)",
}


def validate_krea2_inference_args(args, *, mode: str = "single") -> None:
    """Reject modes and arguments that the Krea runner does not consume."""
    if mode not in {"single", "batch", "interactive"}:
        raise ValueError(f"Unknown Krea-2 inference mode: {mode!r}")
    if mode != "single":
        flag = "--from_file" if mode == "batch" else "--interactive"
        raise SystemExit(
            f"Krea-2-Raw inference currently supports only single-prompt mode; "
            f"{flag} is not implemented."
        )

    offenders = [
        label
        for flag, label in _KREA2_UNSUPPORTED_ANIMA_ONLY.items()
        if getattr(args, flag, None)
    ]

    sampler = str(getattr(args, "sampler", "euler") or "euler").strip().lower()
    if sampler != "euler":
        offenders.append(f"--sampler {sampler} (only euler is supported)")

    flow_shift = getattr(args, "flow_shift", 3.0)
    flow_shift = 3.0 if flow_shift is None else float(flow_shift)
    if not math.isfinite(flow_shift) or not math.isclose(flow_shift, 3.0, abs_tol=1e-9):
        offenders.append(
            "--flow_shift (Krea-2 uses its automatic official mu-shift; "
            "leave the CLI compatibility value at 3.0)"
        )

    if getattr(args, "smc_cfg", False):
        offenders.append("--smc_cfg")
    smc_lambda_raw = getattr(args, "smc_cfg_lambda", 5.0)
    smc_alpha_raw = getattr(args, "smc_cfg_alpha", 0.2)
    smc_lambda = 5.0 if smc_lambda_raw is None else float(smc_lambda_raw)
    smc_alpha = 0.2 if smc_alpha_raw is None else float(smc_alpha_raw)
    if not math.isfinite(smc_lambda) or not math.isclose(smc_lambda, 5.0, abs_tol=1e-9):
        offenders.append("--smc_cfg_lambda (SMC-CFG is unsupported)")
    if not math.isfinite(smc_alpha) or not math.isclose(smc_alpha, 0.2, abs_tol=1e-9):
        offenders.append("--smc_cfg_alpha (SMC-CFG is unsupported)")

    if getattr(args, "cns", None):
        offenders.append("--cns")
    cns_strength_raw = getattr(args, "cns_strength", 1.0)
    cns_strength = 1.0 if cns_strength_raw is None else float(cns_strength_raw)
    if not math.isfinite(cns_strength) or not math.isclose(
        cns_strength, 1.0, abs_tol=1e-9
    ):
        offenders.append("--cns_strength (CNS is unsupported)")

    if offenders:
        raise SystemExit(
            "Krea-2-Raw does not support these inference options:\n  - "
            + "\n  - ".join(offenders)
            + "\nSee docs/findings/backend_multi_model_audit_20260810.md."
        )


def _reject_anima_only_extras(args) -> None:
    """Compatibility wrapper for callers of the former narrow guard."""
    validate_krea2_inference_args(args, mode="single")


def require_krea2_checkpoint_family(network) -> None:
    if getattr(getattr(network, "cfg", None), "model_family", None) != "krea2_raw":
        raise ValueError(
            "Krea-2 inference requires a checkpoint stamped "
            "ss_model_family=krea2_raw; old or Anima checkpoints are rejected."
        )


def load_krea2_dit_for_inference(
    args,
    device: torch.device,
    dit_weight_dtype: Optional[torch.dtype],
) -> Tuple[SingleStreamDiT, Optional[Any]]:
    """加载 Krea-2 DiT + 可选 LoRA attach (推理侧).

    LoRA 用 attach 模式 (非 anima 的静态合并): load_krea2_dit 是 strict 原生 key
    加载, 无 anima 的 rename/concat hook 把 LoRA 合进 state dict. 所以先加载干净
    DiT, 再 create_network_from_weights (读 ss_model_family stamp 自动 family) +
    apply_to + load_weights, 与 anima P-GRAFT/Hydra attach 路径同构.

    返回 (dit, network). network 可能为 None (无 LoRA).
    """
    from networks import lora_anima
    from library.inference.models import (
        _load_lora_state_dict_for_inference,
        _read_lora_metadata,
        _resolve_lora_multiplier_for_index,
    )
    from library.inference.precision import resolve_runtime_dtype

    runtime_dtype = dit_weight_dtype or resolve_runtime_dtype(args)
    compile_enabled = bool(
        getattr(args, "compile", False) or getattr(args, "compile_blocks", False)
    )
    requested_attn_mode = validate_krea2_attention_mode(
        getattr(args, "attn_mode", None),
        dtype=runtime_dtype,
        compile_enabled=compile_enabled,
    )

    dit = load_krea2_dit(args.dit, device="cpu", dtype=runtime_dtype, eval=True)
    attn_mode = prepare_krea2_attention(
        dit,
        requested_attn_mode,
        dtype=runtime_dtype,
        compile_enabled=compile_enabled,
    )
    logger.info("Krea-2 inference attention mode: %s", attn_mode)
    dit = dit.to(device).eval()
    for p in dit.parameters():
        p.requires_grad_(False)

    lora_weight = getattr(args, "lora_weight", None)
    if not lora_weight:
        return dit, None

    network = None
    for index, lora_path in enumerate(lora_weight):
        lora_sd = _load_lora_state_dict_for_inference(args, lora_path)
        # Krea-2 LoRA 复用 anima lora_unet_* 命名空间 (family-aware target spec
        # 由 factory 从 ss_model_family stamp 自动注入, 不靠 key 前缀分叉).
        lora_sd = {k: v for k, v in lora_sd.items() if not k.startswith("lora_te_")}
        if not lora_sd:
            logger.warning(
                "Krea-2: no DiT LoRA tensors left after filtering, skip %s",
                lora_path,
            )
            continue
        multiplier = _resolve_lora_multiplier_for_index(args.lora_multiplier, index)
        network, weights_sd = lora_anima.create_network_from_weights(
            multiplier=multiplier,
            file=lora_path,
            ae=None,
            text_encoders=[],
            unet=dit,
            weights_sd=lora_sd,
            metadata=_read_lora_metadata(lora_path),
            for_inference=True,
        )
        require_krea2_checkpoint_family(network)
        network.apply_to([], dit, apply_text_encoder=False, apply_unet=True)
        info = network.load_state_dict(weights_sd, strict=False)
        if info.unexpected_keys:
            logger.debug(
                "Krea-2 LoRA unexpected keys: %s...",
                info.unexpected_keys[:5],
            )
        network.to(device, dtype=runtime_dtype)
        network.eval()
        logger.info(
            "Krea-2 LoRA attached: %s (multiplier=%s)", lora_path, multiplier
        )
    return dit, network


def _encode_prompt(
    te_model,
    tokenizer,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Krea2TextEmbedding:
    """encode 单 prompt -> Krea2TextEmbedding (hiddens+mask on device)."""
    tok = Krea2TokenizeStrategy()
    tokens = tok.tokenize([prompt])
    enc = Krea2TextEncodingStrategy()
    with torch.inference_mode():
        [hiddens, mask] = enc.encode_tokens(tok, [te_model], tokens)
    return Krea2TextEmbedding(
        hiddens.to(device=device, dtype=dtype),
        mask.to(device=device),
    )


def prepare_krea2_text(
    args,
    device: torch.device,
    runtime_dtype: torch.dtype,
) -> Tuple[Krea2TextEmbedding, Krea2TextEmbedding]:
    """加载 Qwen3-VL TE + encode cond/uncond(空串) + 释放 TE.

    遵循 lazy loading 不变量 (AGENTS.md): TE -> encode -> free -> 再加载 DiT.
    DiT 必须在 TE 释放后才加载 (26GB DiT + 8.9GB TE 同时驻留 PG199 会 OOM).
    因此本函数不依赖已加载的 DiT — runtime_dtype 由调用方从 args 解析传入.
    uncond 用空字符串 (走同一套 ChatML encode, 与 anima T5("")+LLM adapter 不同构).
    """
    runtime = FamilyInferenceRunner(
        Krea2InferenceRunner.family,
        device,
        cleanup=clean_memory_on_device,
    )
    return runtime.run_text(
        lambda: load_krea2_text_encoder_for_inference(args, device),
        lambda bundle: encode_krea2_prompts(
            args,
            bundle,
            device,
            runtime_dtype,
        ),
    )


def load_krea2_text_encoder_for_inference(args, device: torch.device):
    """Load the TE stage resource; release is owned by FamilyInferenceRunner."""
    te_device = (
        torch.device("cpu")
        if getattr(args, "text_encoder_cpu", False)
        else device
    )
    return load_krea2_text_encoder(
        args.text_encoder,
        dtype=resolve_text_encoder_dtype(args),
        device=str(te_device),
    )


def encode_krea2_prompts(
    args,
    encoder_bundle,
    device: torch.device,
    runtime_dtype: torch.dtype,
) -> Tuple[Krea2TextEmbedding, Krea2TextEmbedding]:
    """Encode positive/negative prompts while one TE instance is resident."""
    te_model, tokenizer = encoder_bundle
    cond_emb = _encode_prompt(
        te_model,
        tokenizer,
        args.prompt or "",
        device,
        runtime_dtype,
    )
    uncond_emb = _encode_prompt(
        te_model,
        tokenizer,
        getattr(args, "negative_prompt", "") or "",
        device,
        runtime_dtype,
    )
    return cond_emb, uncond_emb


def generate_krea2(
    args,
    dit: SingleStreamDiT,
    network: Optional[Any],
    cond_emb: Krea2TextEmbedding,
    uncond_emb: Krea2TextEmbedding,
    device: torch.device,
    seed: int,
    runtime_dtype: torch.dtype,
) -> torch.Tensor:
    """Krea-2 flow-matching Euler ODE 采样 + CFG, 返回 5D latent (B,C,1,H,W).

    复用 sampling.sample + forward_for_loss (训练/推理共用承重接口). mu shift 的
    seq_len 用纯图像 token 数 (patchify 后 h_*w_).
    """
    # image_size: anima 约定 [H, W]; Krea-2 DiT LATENT_CHANNELS=16, VAE 压缩 8×.
    height, width = args.image_size
    patch = dit.config.patch
    channels = dit.config.channels
    latent_h, latent_w = height // 8, width // 8
    img_seq_len = (latent_h // patch) * (latent_w // patch)

    steps = int(args.infer_steps)
    cfg = float(args.guidance_scale)

    # 初始 latent = randn (σ=1.0 隐含, 同 probe_sample). 5D 不变量.
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    init_latent = torch.randn(
        1, channels, 1, latent_h, latent_w, device=device, dtype=runtime_dtype,
        generator=gen,
    )

    # dit_forward: (latents_5d, text_emb, t) -> velocity_5d (forward_for_loss 签名).
    # network (LoRA) 已 apply_to 进 dit 的 Linear.forward, forward_for_loss 透明.
    def dit_forward(latents_5d, text_emb, t):
        velocity = forward_for_loss(dit, latents_5d, text_emb, t)
        return velocity.to(latents_5d.dtype)

    logger.info(
        f"Krea-2 采样: {steps} 步, cfg={cfg}, latent=({latent_h},{latent_w}), "
        f"img_seq_len={img_seq_len}"
    )

    final_latent = sample(
        dit_forward,
        init_latent,
        cond_emb,
        uncond_emb,
        img_seq_len,
        steps=steps,
        cfg=cfg,
        device=device,
        dtype=runtime_dtype,
        step_start_callback=lambda index, latent, sigma: apply_preview_router_conditioning(
            network, latent, sigma, step_index=index
        ),
    )
    return final_latent
