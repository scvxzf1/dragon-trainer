"""Krea-2-Raw ModelFamily 实现 (阶段 4).

沿 docs/proposal/krea2_raw_migration.md §3 的 ModelFamily 边界落地. 本模块是
Krea-2 在训练侧的承重接口: forward_for_loss 把 (5D latent, 文本 hiddens/mask,
timestep) 转成 Krea-2 single-stream DiT 需要的 (patchified img, context, pos,
mask) 并跑 forward, 还原成与 latent 同形的 velocity (5D).

阶段 4 在自包含训练探针里验证; 阶段 6 配置收口把 train.py 的 batch_step 通过
``compute_noise_pred_and_target`` 接到本模块 (反上帝守则: 热点文件 noise_target.py
保持 anima 路径, Krea-2 走 family 模块独立函数).

关键不变量:
- latent 5D (B, C, T=1, H, W), 单例时间轴 dim 2 (同 anima, AGENTS.md 不可破坏).
- 进 DiT 前 squeeze(2) 到 4D (B, C, H, W), 出 DiT 后 unsqueeze(2) 回 5D.
- text_emb = (hiddens (B, L, 12, 2560), mask (B, L) bool) — R1 契约: mask 屏蔽
padding, 不二次置零 (与 anima zero-sink 不同, 见 stage1 findings).
- timestep = σ ∈ [0, 1] float, DiT 内部 temb 做 t*tfactor(1e3) sinusoidal embedding.
- target = noise - latents (rectified-flow velocity; 与 anima flow-matching 同构,
  阶段 4 子代理核实 noise_target.py:381).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange, repeat
from torch import Tensor


@dataclass
class Krea2TextEmbedding:
    """Krea-2 文本编码输出 (forward_for_loss 的 text_emb 入参).

    hiddens: (B, L_txt, 12, 2560) — Qwen3-VL 选 12 层 stack dim=2, 切 prefix 34 后.
    mask: (B, L_txt) bool — True=有效 token, False=padding (R1: 屏蔽不置零).
    """

    hiddens: Tensor
    mask: Tensor


_KREA2_POS_CACHE: dict[
    tuple[int, int, int, int, torch.device], tuple[Tensor, Tensor]
] = {}
_KREA2_POS_CACHE_MAX = 16


def _cached_krea2_pos_and_imgmask(
    b: int,
    h_: int,
    w_: int,
    txtlen: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    key = (b, h_, w_, txtlen, device)
    cached = _KREA2_POS_CACHE.get(key)
    if cached is not None:
        return cached
    imgids = torch.zeros((h_, w_, 3), device=device)
    imgids[..., 1] = torch.arange(h_, device=device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=device, dtype=torch.bool)
    txtpos = torch.zeros(b, txtlen, 3, device=device)
    pos = torch.cat((txtpos, imgpos), dim=1)
    if len(_KREA2_POS_CACHE) >= _KREA2_POS_CACHE_MAX:
        _KREA2_POS_CACHE.pop(next(iter(_KREA2_POS_CACHE)))
    _KREA2_POS_CACHE[key] = (pos, imgmask)
    return pos, imgmask


def prepare_img_tokens(
    img: Tensor,
    txtlen: int,
    patch: int,
    txtmask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """patchify latent + 构造 3D pos / key-padding mask.

    移植自 krea-ai/krea-2 sampling.prepare (同 scripts/krea2/probe_*.py).
    img: (B, C, H, W) 4D latent (已 squeeze(2) 离开 5D 时间轴).
    返回:
      img_tokens (B, L_img, patch*patch*C),
      pos (B, L_txt+L_img, 3),
      mask (B, L_txt+L_img) bool.
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    pos, imgmask = _cached_krea2_pos_and_imgmask(b, h_, w_, txtlen, img.device)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    mask = torch.cat((txtmask, imgmask), dim=1)
    return img, pos, mask


def forward_for_loss(
    dit: torch.nn.Module,
    latents_5d: Tensor,
    text_emb: Krea2TextEmbedding | tuple[Tensor, Tensor],
    t: Tensor,
    **kw: Any,
) -> Tensor:
    """Krea-2 承重接口: 5D latent → DiT forward → 5D velocity.

    latents_5d: (B, C, T=1, H, W) — anima 5D 不变量.
    text_emb: Krea2TextEmbedding 或 (hiddens, mask) tuple.
    t: (B,) timestep = σ ∈ [0, 1].

    返回 velocity (B, C, T=1, H, W) — 与 latents_5d 同形, 用于 flow-matching loss.
    """
    if isinstance(text_emb, Krea2TextEmbedding):
        hiddens, txtmask = text_emb.hiddens, text_emb.mask
    else:
        hiddens, txtmask = text_emb

    # 5D → 4D (squeeze 单例时间轴 dim 2; 不用裸 squeeze 防误伤 batch=1).
    b, c, _t, h, w = latents_5d.shape
    latents_4d = latents_5d.squeeze(2)
    assert latents_4d.shape == (b, c, h, w)

    patch = dit.config.patch
    channels = dit.config.channels
    txtlen = hiddens.shape[1]

    img_tokens, pos, mask = prepare_img_tokens(
        latents_4d, txtlen, patch, txtmask
    )
    context = hiddens

    out = dit(img=img_tokens, context=context, t=t, pos=pos, mask=mask)
    # out: (B, L_img, patch*patch*C) → 还原 4D → 5D.
    latent_h, latent_w = h // patch, w // patch
    velocity_4d = rearrange(
        out,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=latent_h,
        w=latent_w,
        c=channels,
        ph=patch,
        pw=patch,
    )
    velocity_5d = velocity_4d.unsqueeze(2)
    return velocity_5d


def prepare_text_embedding_for_training(
    text_encoder_conds,
    batch,
    text_encoding_strategy,
    *,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Move Krea text conditions to device and apply cached-caption dropout."""
    hiddens, mask = text_encoder_conds[0], text_encoder_conds[1]
    caption_dropout_rates = (
        batch.get("caption_dropout_rates") if isinstance(batch, dict) else None
    )
    # copy=True avoids mutating reusable in-memory caches when device already
    # matches, without adding a second allocation to the normal CPU->GPU copy.
    copy = caption_dropout_rates is not None
    hiddens = hiddens.to(device=device, dtype=weight_dtype, copy=copy)
    mask = mask.to(device=device, copy=copy)
    if caption_dropout_rates is not None:
        text_encoding_strategy.apply_caption_dropout_inplace(
            caption_dropout_rates,
            hiddens=hiddens,
            mask=mask,
        )
    return hiddens, mask


def compute_noise_pred_and_target(
    trainer,
    ctx,
    latents: Tensor,
    batch,
    text_encoder_conds,
    *,
    is_train: bool = True,
):
    """Krea-2 训练 noise pred + target (阶段 6 配置收口).

    与 anima ``library.training.noise_target.compute_noise_pred_and_target``
    同返回契约: ``(model_pred, target, timesteps, weighting)`` — 这样
    batch_step 的下游 (loss composer, prior-preservation observer) 可保持
    共享. Krea-2 首日不实现 crossattn_emb / postfix / method-adapter extra
    forwards / VR loss / affine / observer (那些是 anima-only 能力, 见提案 §1
    非目标); 只做最简 rectified-flow 训练路径.

    复用 anima 的 sampler registry (M1) 取 noisy_input + timesteps + sigmas —
    sampler 只做 ``x_t = (1-σ)x0 + σ·noise``, 与 family 无关. target 走
    rectified-flow velocity (``noise - latents``), 与 anima flow-matching 同构.
    """
    args = ctx.args
    accelerator = ctx.accelerator
    unet = ctx.unet
    network = ctx.network
    weight_dtype = ctx.weight_dtype

    # 5D 不变量: 4D cache → 5D. 老 5D cache 兼容 squeeze.
    if latents.ndim == 5:
        latents = latents.squeeze(2)
    noise = torch.randn_like(latents)

    from library.training.samplers import SAMPLER_REGISTRY, SamplerContext

    sampler_fn = SAMPLER_REGISTRY[getattr(args, "sampler", "default") or "default"]
    sampler_out = sampler_fn(
        SamplerContext(
            args=args,
            noise_scheduler=ctx.noise_scheduler,
            latents=latents,
            noise=noise,
            device=accelerator.device,
            weight_dtype=weight_dtype,
        )
    )
    noisy_model_input = sampler_out.noisy_input
    timesteps = sampler_out.timesteps
    sigmas = sampler_out.sigmas

    # Per-step network conditioning (timestep masks / σ-FEI routers). Krea-2
    # 复用 anima LoRA network — apply_router_conditioning 对 SingleStreamBlock
    # 的 Linear 透明 (只 set_timestep_mask / set_fei by reference).
    from library.training.router_conditioning import apply_router_conditioning

    trainer._hydra_warmup_step = apply_router_conditioning(
        network=network,
        noisy_model_input=noisy_model_input,
        timesteps=timesteps,
        is_train=is_train,
        warmup_step=int(getattr(trainer, "_hydra_warmup_step", 0)),
        max_train_steps=int(getattr(args, "max_train_steps", 0) or 0),
        gradient_accumulation_steps=int(
            getattr(args, "gradient_accumulation_steps", 1) or 1
        ),
    )

    if args.gradient_checkpointing:
        noisy_model_input.requires_grad_(True)

    # Unpack Krea-2 text conds. Cache format: [hiddens (B,L,12,2560),
    # mask (B,L) bool]; live-encode path returns the same via
    # Krea2TextEncodingStrategy.encode_tokens.
    if not text_encoder_conds or text_encoder_conds[0] is None:
        # Live encode (uncached / TE training) — reuses the shared singleton.
        text_encoding_strategy = ctx.text_encoding_strategy
        tokenize_strategy = ctx.tokenize_strategy
        with torch.set_grad_enabled(is_train and False), accelerator.autocast():
            input_ids = [
                ids.to(accelerator.device) for ids in batch["input_ids_list"]
            ]
            encoded = text_encoding_strategy.encode_tokens(
                tokenize_strategy,
                trainer.get_models_for_text_encoding(
                    args, accelerator, ctx.text_encoders
                ),
                input_ids,
            )
        text_encoder_conds = encoded
    hiddens, mask = prepare_text_embedding_for_training(
        text_encoder_conds,
        batch,
        ctx.text_encoding_strategy,
        device=accelerator.device,
        weight_dtype=weight_dtype,
    )

    # 4D → 5D (anima 5D 不变量, DiT 承重接口入参).
    noisy_model_input = noisy_model_input.unsqueeze(2)

    with torch.set_grad_enabled(is_train), accelerator.autocast():
        model_pred = forward_for_loss(
            unet, noisy_model_input, (hiddens, mask), timesteps
        )
    model_pred = model_pred.squeeze(2)  # 5D → 4D

    from library.training.flow_step import velocity_target

    # Rectified-flow target: noise - latents (velocity; anima 同构).
    target = velocity_target(noise, latents)

    # Loss weighting (复用 anima 的 min-snr / p2 加权; Krea-2 首日沿用).
    from library.anima import training as anima_train_utils

    weighting = anima_train_utils.compute_loss_weighting_for_anima(
        weighting_scheme=args.weighting_scheme,
        sigmas=sigmas,
        min_snr_gamma=getattr(args, "min_snr_gamma", None),
        p2_gamma=getattr(args, "p2_gamma", 1.0),
        p2_k=getattr(args, "p2_k", 1.0),
    )
    return model_pred, target, timesteps, weighting
