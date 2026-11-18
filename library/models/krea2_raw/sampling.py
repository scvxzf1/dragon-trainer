"""Krea-2-Raw 推理采样器 (阶段 5).

移植自 krea-ai/krea-2 sampling.py (路径 B 裸移植, 非 diffusers).
公式与 anima 的 library/inference/sampling.py 不同: anima 用线性 mu shift
`(shift*sigmas)/(1+(shift-1)*sigmas)`, Krea-2 官方用 log-sigmoid shift
`exp(mu)/(exp(mu)+(1/ts-1)**sigma)`. 所以不复用 anima 的 get_timesteps_sigmas.

关键数值 (子代理核实 krea-ai/krea-2 sampling.py):
- Euler ODE, 默认 28 步, 确定性无噪声注入
- 更新: img = img + (tprev - tcurr) * v  (tprev < tcurr, ts 倒序)
- sigma 网格: linspace(1.0, 0.0, steps+1) 倒序
- mu 端点: (x1=256, y1=0.5) / (x2=6400, y2=1.15)
- mu 公式: mu = (y2-y1)/(x2-x1) * seq_len + (y1 - slope*x1)
  seq_len = 纯图像 token 数 (patchify 后 h_ * w_), 不含文本 token
- shift 应用: ts = exp(mu) / (exp(mu) + (1/ts - 1)**sigma), sigma=1.0
- CFG: v = cond + guidance*(cond - uncond), uncond=空字符串, 默认 cfg=4.5
- 初始 latent = randn, 无 sigma_max 缩放 (σ=1.0 隐含)
- 官方无 block swap / offload

阶段 5 仅在自包含探针里验证; 正式串通 generation.py 是阶段 6 配置收口的事
(反上帝守则, generation.py 是热点文件).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from library.inference.flow_euler import run_flow_euler_loop


def timesteps(
    seq_len: int,
    steps: int,
    x1: int = 256,
    x2: int = 6400,
    y1: float = 0.5,
    y2: float = 1.15,
    sigma: float = 1.0,
    mu: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Krea-2 官方 mu-shift 时间网格.

    seq_len: 纯图像 token 数 (patchify 后 h_*w_), 不含文本.
    返回 (steps+1,) 的 sigma 网格, 从 1.0 倒序到 0.0.
    """
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    # 网格: 1.0 -> 0.0 倒序, steps+1 个点
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    # log-sigmoid shift (官方公式, sigma=1.0)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts


def sample(
    dit_forward,
    latents_5d: Tensor,
    cond_emb,
    uncond_emb,
    img_seq_len: int,
    steps: int = 28,
    cfg: float = 4.5,
    x1: int = 256,
    x2: int = 6400,
    y1: float = 0.5,
    y2: float = 1.15,
    mu: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    step_start_callback=None,
) -> Tensor:
    """Krea-2 flow-matching Euler ODE 采样.

    dit_forward: callable(latents_5d, text_emb, t) -> velocity_5d (同 family.forward_for_loss 签名)
    cond_emb / uncond_emb: Krea2TextEmbedding 或 (hiddens, mask) tuple (uncond 用空 prompt 编码)
    img_seq_len: 图像 patch token 数 (h_*w_), 用于 mu shift
    cfg: guidance scale, 0 跳过 uncond (只跑 cond 一次)
    返回最终 5D latent (B, C, 1, H, W), σ=0.
    """
    ts = timesteps(
        img_seq_len, steps, x1=x1, x2=x2, y1=y1, y2=y2, mu=mu,
        device=device, dtype=dtype,
    )
    use_cfg = cfg > 0
    def predict(image, sigma):
        return dit_forward(image, cond_emb, sigma)

    def predict_negative(image, sigma):
        return dit_forward(image, uncond_emb, sigma)

    with torch.inference_mode():
        return run_flow_euler_loop(
            latents_5d,
            ts,
            predict,
            guidance_scale=cfg if use_cfg else 0.0,
            predict_negative=predict_negative if use_cfg else None,
            step_start_callback=step_start_callback,
        )
