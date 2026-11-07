# Krea-2-Raw 迁移阶段 0 可行性确认

状态：历史阶段快照 / 阶段 0 已完成
适用版本：2026-08-08 阶段 0；不作为当前能力说明
日期：2026-08-08
入口命令：`python scripts/krea2/probe_vae.py`（R2 验证脚本）
相关代码：`scripts/krea2/probe_vae.py`、`library/models/qwen_vae.py`、权重 `models/{diffusion_models,text_encoders,vae}/`
相关提案：[`../../_archive/docs/proposal/krea2_raw_migration.md`](../../_archive/docs/proposal/krea2_raw_migration.md)、[`../../_archive/docs/proposal/krea2_raw_migration_notes.md`](../../_archive/docs/proposal/krea2_raw_migration_notes.md)

> 本文保留阶段 0 当时的开放问题与下一步；当前能力以
> [`../multi_model_support.md`](../multi_model_support.md) 和实时源码为准。

> 阶段 0 的可行性确认实验记录。权重三件已就位（DiT 26.3GB + TE Qwen3-VL 8.9GB + VAE 253MB，源 `Comfy-Org/Krea-2` 单文件 bf16）。本文件记录 R1/R2/R4 的定论与基准。

---

## R2: VAE encode/decode 互逆验证 — 通过 ✓

**实验**（`scripts/krea2/probe_vae.py`，PG199 32GB，bf16）：加载 anima 的 `AutoencoderKLQwenImage`（`library/models/qwen_vae.py:997`）+ Krea-2 VAE 权重，对结构化图案（渐变+条纹+棋盘）做 encode→decode 互逆。

**结果**：

| 指标 | 结果 |
|---|---|
| VAE 加载 | ✓ "All keys matched successfully"（194 keys 全匹配） |
| `latents_mean` 与 Krea-2 公开值（Qwen/Qwen-Image config.json） | ✓ 逐元素一致（16 元素，误差 <1e-3） |
| `latents_std` 与 Krea-2 公开值 | ✓ 逐元素一致 |
| 256×256 PSNR | 37.73 dB |
| 512×768 PSNR | 40.91 dB |
| 768×512 PSNR | 39.25 dB |
| 5D 路径（B,C,1,H,W，unsqueeze(2)） | ✓ 工作，39.25 dB |
| latent shape | ✓ 全对 `(1, 16, H/8, W/8)` |
| encode 耗时（PG199 bf16） | ~100-500 ms |
| decode 耗时（PG199 bf16） | ~35-180 ms |

**结论**：Krea-2 VAE 与 anima VAE 是**同一个 `AutoencoderKLQwenImage`**，per-channel mean/std 逐元素相同，anima 已实现的 `encode_pixels_to_latents`（`qwen_vae.py:1398-1439`）可直接复用，encode/decode 严格互逆无损（`encode=(x-mean)*(1/std)` ↔ `decode=x/(1/std)+mean`）。**R2 闭合，利好确认。** 加载器 `library/models/krea2_raw/weights.py::load_vae` 可直接调 `library.models.qwen_vae.load_vae`，无需自建 encode 路径。

**注**：随机均匀像素 `torch.rand` 重建 PSNR 仅 10.36 dB（非自然分布，VAE 本就难重建）；结构化图案 37-41 dB 是 VAE 对该素材的正常水平。

---

## R4: DiT state_dict key 清单 + LoRA 注入点定论

**DiT 结构**（430 keys，`models/diffusion_models/krea2_raw_bf16.safetensors`，Comfy-Org 单文件 bf16，mmdit.py 原生命名）：

- `blocks.0..27`（28 层 single-stream block，每层 13 参数）
- `txtfusion.layerwise_blocks.{0,1}`（2 层文本融合，**非 24**——之前 prefix 计数误读）
- `txtfusion.refiner_blocks.{0,1}`（2 层精炼）
- `txtfusion.projector.weight`（12 层→1 的线性投影，shape `(1,12)`）
- `txtmlp.{0.scale,1,3}`（RMSNorm→Linear→GELU→Linear，2560→6144）
- `first.*`（patch embed）、`last.{linear,modulation.lin,norm.scale}`（输出层）
- `tmlp.{0,2}` + `tproj.1`（timestep embedding MLP）

**每个 `blocks.N` 的 13 参数**：

```
attn.wq.weight          # query proj (GQA 48 头, 6144→6144)
attn.wk.weight          # key proj   (GQA 12 头, 6144→1536)
attn.wv.weight          # value proj (GQA 12 头, 6144→1536)
attn.wo.weight          # output proj (6144→6144)
attn.gate.weight         # 门控 sigmoid 的 gate (标量权重, 残差乘性)
attn.qknorm.qnorm.scale # query RMSNorm (head_dim=128)
attn.qknorm.knorm.scale # key RMSNorm
mlp.up.weight            # SwiGLU up (6144→16384)
mlp.down.weight          # SwiGLU down (16384→6144)
mlp.gate.weight          # SwiGLU gate (6144→16384)
mod.lin                  # light modulation bias (6*6144=36864, 单 Parameter)
postnorm.scale           # 后 RMSNorm
prenorm.scale            # 前 RMSNorm
```

**对 R4 开放问题的定论**：

1. **q/k/v 不 fused**（分别是 `wq`/`wk`/`wv` 独立权重，不是 `qkv`）——**anima 的 `qkv` fuse spec 不适用**。加载器 `load_state_dict(strict=True, assign=True)` 直接吃原生命名，**无需 `_dit_concat_hook` 那套 qkv/kv/adaln fuse**（路径 B 裸移植，非 diffusers）。
2. **block 命名极规则**：`blocks.{i}.attn.{wq,wk,wv,wo,gate}` + `mlp.{up,down,gate}` + `mod.lin` + norms——regex `blocks\.(\d+)\.` 完全可用，比 anima 更规整。
3. **无独立 cross_attn**：single-stream，text/image 共享 attention。anima 的 `router_source="crossattn_emb"` 概念在 Krea-2 下消解；`CROSSATTN_EMB_DIM=1024` 不适用（Krea-2 text context 维度 2560）。
4. **LoRA 注入点（首日 plain LoRA，保守集）**：`mlp.up`/`mlp.down`/`mlp.gate` + `attn.wq`/`wk`/`wv`/`wo`。`mod.lin`（light modulation bias）和 `attn.gate`（sigmoid gate）首日**不挂**——它们影响调制/门控语义，需单独验证；先稳住标准 Linear。
5. **三轴路由首日**：`use_moe_style=False`、`router_source="none"`（plain LoRA）。Hydra/FEI 留待 single-stream 语义明确后。

---

## R1: Qwen3-VL padding / attention sink 不变量 — 定论

子代理核实 `krea-ai/krea-2` 的 `encoder.py` 后确认：

- **Krea-2 的 padding 语义与 anima 不同构**。anima 靠"zero 位作 cross-attn softmax sink"（`library/inference/text.py:21` 注释 + `strategy.py:137` 置零）；Krea-2 用 **attention mask 屏蔽 padding**，DiT 内部**不二次置零**。
- Qwen3-VL `Qwen3VLConditioner.forward` 返回 `(hiddens, mask)`，hiddens shape `(B, seqlen, 12, hidden_dim)`（12 层 stack 在 dim=2），mask 在 DiT attention 里屏蔽 padding 位。
- **结论**：迁移时 padding 契约要重写，**不能套 anima 的 `[~mask]=0` 置零逻辑**。strategy.py 只负责 Qwen3-VL 选 12 层 + stack + 返回 `(hiddens, mask)`；txtfusion 留在 DiT 内部。

**Qwen3-VL 加载**：`transformers 5.5.0` 的 `Qwen3VLForConditionalGeneration` 可 import（实测），**R5 的 transformers 版本冲突不是阻塞点**（官方钉 4.57.1，本地 5.5.0 仍可用）。

---

## R8: 加载路径定论 — 路径 B（裸移植 mmdit.py）

本地权重是 **mmdit.py 原生命名**（`blocks.N.attn.wq` 等），非 diffusers 命名（`to_q` 等）。`load_state_dict(strict=True, assign=True)` 直接吃，无需 diffusers key 重映射。**路径 A（diffusers Krea2Transformer2DModel）反而需 ComfyUI→diffusers 格式转换，工程量更大。路径 B 强烈优先。**

## 采样器定论（阶段 5 用）

- flow-matching Euler ODE，**不是 v-pred**（模型直接返回速度场 `v`）
- CFG：`v = cond + guidance * (cond - uncond)`
- mu shift：按图像 token seq_len 线性插值，端点 `(256, 0.5)` 和 `(6400, 1.15)`（base_image_seq_len=256, max=6400）
- sigma=1.0（单一定值）
- 默认 28 步，cfg 4.5（Raw）；Turbo 固定 mu=1.15
- 起始 t=1，标准正态噪声无 sigma 缩放

## 下一步

阶段 0 闭合。进入阶段 1（文本链路）+ 阶段 2（DiT 本体+加载器）+ 阶段 3（LoRA 注入），三者部分可并行。
