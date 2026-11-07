# Krea-2-Raw 迁移阶段 2：DiT 本体移植 + 加载器

状态：历史阶段快照 / 阶段 2 已完成
适用版本：阶段 2 DiT/loader 落地时点；不作为当前完整能力说明
日期：2026-08-08
入口命令：`python scripts/krea2/probe_dit.py`（阶段 2 验证脚本）
相关代码：`library/models/krea2_raw/dit.py`、`library/models/krea2_raw/weights.py`、`scripts/krea2/probe_dit.py`
相关提案：[`../../_archive/docs/proposal/krea2_raw_migration.md`](../../_archive/docs/proposal/krea2_raw_migration.md)

> “已知限制 / 后续”保留阶段 2 当时状态；compile、block swap、lazy loading 等后续
> 已落地能力以 [`../multi_model_support.md`](../multi_model_support.md) 和实时源码为准。
相关前置：[`krea2_raw_migration_stage0_findings.md`](krea2_raw_migration_stage0_findings.md)（R4/R8 定论）

> 阶段 2 出口验证：SingleStreamDiT 忠实移植 mmdit.py + 单文件 strict 加载 + 单 latent forward shape 对齐。

---

## 移植结论

### R8 路径 B 落地

`library/models/krea2_raw/dit.py` 是 krea-ai/krea-2 `mmdit.py` 的**搬家型移植**，计算语义完全一致，只做三处最小适配：

1. **去掉 `@torch.compile(fullgraph=True)` 装饰器**（RMSNorm/PositionalEncoding/LastLayer 原本各带一个）。anima 的 `compile_blocks()` 在 `network.apply_to` 之后统一编译 block `_forward`（Compile After Apply 不变量），内置装饰器会双重编译并冲突。
2. **`attention()` 保留 `sdpa_kernel(SDPBackend.CUDNN_ATTENTION)`（Krea-2 原始选择），加 fallback**：若当前环境无 CUDNN attention，退回默认 `F.scaled_dot_product_attention`（自动选 flash/cudnn/math）。anima 的 `networks/attention_dispatch.py` 用 `EFFICIENT_ATTENTION`，是 anima 适配层；Krea-2 GQA 48:12 + 特定 RoPE 用原版更安全，后续阶段再接入 anima dispatch。
3. **`SingleMMDiTConfig.krea2_raw()`**：从权重反推的默认 config（见下表），作为 dataclass classmethod。

### config 反推核验（从权重 shape 倒推，全部对齐）

| 字段 | 值 | 权重依据 |
|---|---|---|
| features | 6144 | `first.weight: (6144, 64)` |
| tdim | 256 | `tmlp.0.weight: (6144, 256)` |
| txtdim | 2560 | `txtmlp.1.weight: (6144, 2560)` |
| heads | 48 | `attn.wq: (6144,6144)`, headdim=128 |
| kvheads | 12 | `attn.wk/wv: (1536,6144)`, 1536/128=12 |
| headdim | 128 | `qknorm.qnorm.scale: (128,)` |
| multiplier | 4 | `mlp.up: (16384,6144)`, 16384/(2*6144/3)=4 |
| layers | 28 | `blocks.0..27` |
| patch | 2 | `first.weight: (6144, 64)`, 64=16*4, channels=16 |
| channels | 16 | 同上 |
| txtlayers | 12 | `txtfusion.projector.weight: (1, 12)` |
| bias | False | anima 习惯；Krea-2 Linear 默认无 bias |

参数量：**12.82B**。

### 关键架构差异（vs anima DiT）

- **single-stream**：text/image 拼接后共享 attention，无独立 cross_attn。anima 是 dual-stream Cosmos-Predict2（self_attn + cross_attn 分离）。
- **GQA 48:12**：query 48 头，KV 12 头，headdim=128。anima 用标准 MHA 16 头。
- **light modulation**：`mod.lin` 是单 `nn.Parameter(6*dim)` 的 bias（`DoubleSharedModulation`），不是 anima 的 AdaLN-LoRA（SiLU→Linear）。
- **attention 门控**：`sigmoid(gate)` 乘性作用于 attention 输出（`Attention.gate`）。anima 无此门控。
- **3D RoPE**：headdim 按轴拆 `[headdim - 12*(headdim//16), 6*(headdim//16), 6*(headdim//16)]` = `[96, 16, 16]`（headdim=128）。anima 用 `VideoRopePosition3DEmb`（dim_h=dim_w, dim_t=dim-2*dim_h）。
- **SwiGLU MLP**（gate*up→down），anima 用 GPT2FeedForward（GELU）。

### forward 签名

```python
SingleStreamDiT.forward(img, context, t, pos, mask) -> Tensor
# img: (B, L_img, patch*patch*channels) 已 patchify 的 latent 序列
# context: (B, L_txt, num_txt_layers=12, txtdim=2560) Qwen3-VL 多层 hidden states
# t: (B,) timestep
# pos: (B, L_total, 3) 3D 位置编码（H, W, token-type 轴）
# mask: (B, L_total) key-padding mask（True=有效，R1 定论用 mask 屏蔽 padding）
# 输出: (B, L_img, patch*patch*channels) 速度场 v
```

`pos`/`mask` 构造移植自 `krea-ai/krea-2 sampling.prepare()`：text pos 全 0，image pos = `(0, h_idx, w_idx)`，mask = text+image 拼接。

---

## 阶段 2 出口验证结果（probe_dit.py，PG199 bf16）

| 验证项 | 结果 |
|---|---|
| config 反推与 `krea2_raw()` 全对齐 | ✓ |
| strict 加载 0 missing / 0 unexpected（430 keys） | ✓ |
| 单 latent forward shape 对齐 (1,256,64)→(1,256,64) | ✓ |
| 多分辨率 forward shape 对齐（256/512） | ✓ |
| 数值有限（无 NaN/Inf） | ✓ |

### 性能基准（PG199 32GB，bf16，inference_mode）

| 分辨率 | forward 耗时（热） | 峰值显存 |
|---|---|---|
| 256×256 | 90 ms | 25.74 GB |
| 512×512 | 222 ms | 25.88 GB |

权重 25.6 GB 占绝大部分显存，激活仅 ~0.3 GB。GPU 总 33.87 GB，空闲 8.13 GB。训练时需要 LoRA + 优化器状态 + 梯度 + block swap 才能装下（阶段 4/5 验证）。

---

## 阶段 2 截止时的已知限制 / 后续（历史快照）

- **阶段 2 当时未接入 Anima `attention_dispatch`**：Krea-2 使用原版
  `sdpa_kernel(CUDNN_ATTENTION)` + fallback。当前仍由
  `library/models/krea2_raw/attention_backend.py` 维护专用 backend（仅按需复用
  provider 可用性检查），不等同于复用 Anima 的通用 layout dispatch。
- **阶段 2 当时未接入 `compile_blocks()`**：Krea-2 DiT 的 block forward 签名与
  Anima `Block._forward` 不同（single-stream vs dual-stream），不能直接套用 Anima
  的 native-flatten + compile。后续已实现 Krea-2 自有 compile 路径，当前入口见
  `library/models/krea2_raw/dit.py::compile_blocks`。
- **阶段 2 当时未接入 block swap / lazy loading**：后续阶段 5/6 已补齐，当前入口见
  `library/models/krea2_raw/dit.py` 的 offloader/block-swap 接口。
- **context 当时仍是合成随机张量**：阶段 1 完成后已由真实 Qwen3-VL hidden states
  和独立 text-cache 链路替代。

## 下一步

阶段 2 闭合。进入：
- 阶段 1（文本链路）：`library/models/krea2_raw/strategy.py` Qwen3-VL tokenize + 12 层 MFA + caching。
- 阶段 3（LoRA 注入）：`library/models/krea2_raw/lora_targets.py` single-stream block target spec。
