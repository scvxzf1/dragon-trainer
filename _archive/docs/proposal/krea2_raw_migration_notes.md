# Krea-2-Raw 迁移注意事项与风险登记册

状态：历史风险登记册 / 核心迁移已落地 / 待归档
适用版本：2026-08-08 迁移执行快照；正文的“待验证/未实现”不代表当前代码
日期：2026-08-08

> **一句话：** 这是迁移当时逐条核对的历史清单，不是当前能力说明。每条风险保留原始
> 触发条件、现象和缓解过程；已经落地的结论以当前源码、
> [`krea2_raw_migration.md`](krea2_raw_migration.md) 顶部状态和
> [`../multi_model_support.md`](../../../docs/multi_model_support.md) 为准。

---

## R1. Qwen3-VL padding / attention sink 不变量是否成立

**触发条件：** 阶段 1 写 `library/models/krea2_raw/strategy.py` 的 tokenize/encode，第一次把 Qwen3-VL 输出喂给 DiT cross-attention 时。

**现象（若不变量不成立）：**
- 裁剪到真实文本长度 → 黑图或纯噪声（anima 行为：`library/inference/text.py:21` 注释明确"裁剪会产生黑图"）。
- max-pad 到固定长度但 pad 位不置零或不被当 sink → softmax 分布异常，质量崩。
- 不同 prompt 长度下结果不一致（说明 padding 影响了 conditioning）。

**Anima 的不变量（基线，必须先理解再判断 Krea-2 是否沿用）：**
- `library/inference/text.py:21` `MAX_CROSSATTN_TOKENS=512`，注释："预训练模型将零填充位置视为交叉注意力 softmax 中的注意力汇聚点——裁剪到实际文本长度会产生黑色图像。"
- `library/anima/strategy.py:137` `prompt_embeds[~qwen3_attn_mask.bool()] = 0`：padding 位在 DiT 内部被置零。
- `library/anima/models.py:2505, 2736` `context[~crossattn_mask.bool()] = 0`：DiT 内部再次置零。
- `_trim_outputs`（`library/anima/strategy.py:479-496`）是直通空操作，明确不裁剪。
- 推理时运行时 pad 到 512：`library/inference/text.py:153-157, 185-189`。

**关键区别：** Anima 的文本是 Qwen3（纯文本 LLM）+ LLM Adapter 桥到 T5 space。Krea-2 用 Qwen3-VL-4B（VLM）+ 12 层 MFA 聚合。**VLM 的 pad token 语义、多层聚合后的 padding 行为，与 anima 的"zero 位作 sink"是否同构，完全没有验证。** 这是头号技术风险。

**缓解：**
1. 阶段 0 用官方 `Krea2Pipeline` 跑推理时，dump Qwen3-VL 的 `attention_mask`、padding 长度、MFA 输出 shape，看官方代码如何处理 padding。
2. 阶段 1 写策略前，先做对照实验：固定 prompt，分别用真实长度 / max-pad / max-pad+置零 三种方式喂 DiT，看输出差异。**结论写入本文件后再写策略代码。**
3. 若 VLM 不沿用 sink 语义，重新推导 padding 契约，不要假设 512。

**何时该暂停重估：** 若阶段 1 对照实验显示任何 padding 方式都产生黑图或质量崩，且无法定位原因 → 暂停，回到阶段 0 深读 `krea-ai/krea-2` 的 `encoder.py` padding 处理，不要硬写策略。

---

## R2. VAE encode/decode 互逆验证（虽是利好，仍需验证）

**触发条件：** 阶段 0 验证 VAE、阶段 2 写加载器、阶段 4 训练前 cache latents。

**利好前提：** Anima VAE 就是 `AutoencoderKLQwenImage`（`library/models/qwen_vae.py:997`），Krea-2 官方 `autoencoder.py` 也用同一个 from `Qwen/Qwen-Image`。anima 已实现 `encode_pixels_to_latents`（`qwen_vae.py:1398-1439`）。**理论上 encode 路径直接复用。**

**仍需验证的点：**
1. **per-channel mean/std 一致性**：anima 的 `latents_mean`/`latents_std`（`qwen_vae.py:1019-1054`，16 元素列表）必须与 Krea-2 checkpoint 里 VAE 的 `latents_mean`/`latents_std` 逐元素一致。若 Krea-2 用了不同归一化常量，encode 出的 latent 与官方 decode 不匹配。
2. **encode/decode 严格互逆**：用 anima 的 `encode_pixels_to_latents` 编码一张图，再用同一个 VAE decode 回来，确认像素级无损（或误差在可接受范围）。官方 `autoencoder.py` 只暴露 `decode`，没有 encode 的官方实现做参照，所以互逆性要自己证。
3. **5D 布局**：Krea-2 VAE 内部 `rearrange(x, "b c h w -> b c 1 h w")`（T=1），与 anima 的 5D 一致。确认 `unsqueeze(2)`/`squeeze(2)` 位置正确。

**缓解：**
1. 阶段 0 写一个一次性脚本：加载 Krea-2 VAE 权重 → `encode_pixels_to_latents` 一张测试图 → decode 回来 → 计算与原图的 PSNR/误差。**通过门槛：误差与 anima 自身 encode→decode 误差同量级。**
2. dump Krea-2 VAE config（`vae/config.json`，791 字节，gated 但接受 license 后可读）的 `latents_mean`/`latents_std`/`scaling_factor`，与 anima 硬编码值对比。
3. 若不一致，`library/models/krea2_raw/weights.py::load_vae` 要从 checkpoint 读 mean/std 而非硬编码。

**何时该暂停重估：** 若 encode→decode 互逆失败且无法通过调 mean/std 修复 → VAE 可能不是完全相同版本，需深查 diffusers `AutoencoderKLQwenImage` 是否对 `Qwen/Qwen-Image` VAE 做过改动。

---

## R3. compile after apply 顺序不变量无运行时防呆

**触发条件：** 阶段 4 写 `library/runtime/harness.py` 的 Krea-2 加载路径，或新写加载器时。

**现象（若乱序）：** `torch.compile` 静默追踪错误的 forward 图（原始 DiT forward 而非 adapter monkey-patch 后的 forward），表现为梯度错误 / 输出错误，**无报错，难定位**。

**Anima 的现状：** `harness.py:64 build_anima` 6 步顺序（load → requires_grad_(False) → adapter apply_to+load_weights → grad-ckpt → fp32 拥差 → compile_blocks）完全靠文档约定和字面执行顺序强制，**没有运行时断言**。`harness.py:192-193` 注释："最后编译。必须先安装 adapter monkey-patch，否则 torch.compile 会追踪到错误的 forward。" `compile_blocks_for_training` docstring（`:727-731`）重复此约束。

**迁移要求：** Krea-2 加载器必须照搬此顺序。**建议同时补一个运行时断言**作为防呆（anima 当前没有）：

```python
# 在 compile_blocks 入口处
if not getattr(network, "_applied", False):
    raise RuntimeError("compile_blocks called before network.apply_to; "
                        "torch.compile would trace the un-patched forward")
```

**缓解：**
1. 新加载器严格复刻 6 步顺序。
2. 加 `_applied` 标志断言（这是给 Krea-2 迁移顺带补的防呆，anima 侧可同步受益但属独立小改）。
3. 阶段 4 训练前，用 `torch._dynamo` 的 graph dump 确认 compile 追踪的是 patch 后的 forward。

**何时该暂停重估：** 若发现 compile 后梯度异常 / loss 不下降且无报错 → 立即怀疑乱序，dump dynamo graph 核对。

---

## R4. single-stream MMDiT 的 LoRA 注入点与 fuse spec 重设计

**触发条件：** 阶段 3 写 `library/models/krea2_raw/lora_targets.py`，挂 LoRA 到 SingleStreamBlock。

**Anima 的现状（dual-stream，不能直接套）：**
- 容器白名单 `networks/lora_anima/network.py:42-55`：`["Block","PatchEmbed","TimestepEmbedding","FinalLayer"]` + LLM adapter `["LLMAdapterTransformerBlock"]` + TE `["Qwen3Attention","Qwen3MLP",...]`。
- fuse spec `networks/attn_fuse.py:51-54`：`self_attn qkv` / `cross_attn kv`（dual-stream 有独立 cross_attn）。
- `CROSSATTN_EMB_DIM=1024`（`networks/lora_anima/routers.py:14`）硬常量，注释"Fixed by the Anima DiT"。
- 三轴路由 `router_source="crossattn_emb"` 假设有独立 cross-attention 可挂。

**Krea-2 的区别（single-stream）：**
- text/image token 共享同一套 attention + MLP 权重，**没有独立 cross_attn**。
- GQA 48:12（query 48 头，KV 12 头），不是 anima 的 16 头对称。
- 门控 sigmoid attention：attention 输出经 `F.sigmoid(gate)`，挂 LoRA 要考虑 gate 是否是合理 target。
- light modulation with bias（每 block 一个可调 bias），不是 AdaLN-LoRA。

**待回答的开放问题（阶段 0 dump key 后定）：**
1. SingleStreamBlock 里哪些 Linear 是合理 LoRA target？（q/k/v proj、o proj、MLP gate/up/down？）
2. GQA 下 q/k/v 是否 fused？fused spec 怎么写？（anima 是 `qkv` fuse，Krea-2 GQA 可能 q 单独、kv 共享）
3. `router_source="crossattn_emb"` 在 single-stream 下还有意义吗？若 text/image 共享 attention，"cross-attention embedding"这个概念是否消解？
4. `CROSSATTN_EMB_DIM` 在 Krea-2 下对应什么维度？（Krea-2 text context 维度是 2560，非 1024）
5. 门控 sigmoid 的 gate 要不要挂 LoRA？

**缓解：**
1. 阶段 0 dump transformer state_dict keys，对照 `krea-ai/krea-2` 的 `mmdit.py::SingleStreamBlock` 源码，逐一标注每个 Linear 的角色。
2. 阶段 3 先只挂最保守的 target（MLP 的 up/down/gate，不动 attention），验证训练能跑，再逐步加 attention target。
3. 三轴路由首日**只支持 plain LoRA**（`use_moe_style=False`、`router_source="none"`），Hydra/FEI/σ 路由留待 single-stream 语义明确后再定。

**何时该暂停重估：** 若 dump key 后发现 block 命名极不规则、无法用 regex 表达 → 考虑按模块类型而非层名匹配（`targeting.py:68-81` 已支持 class name + Linear 类型检测，可走这条路）。

---

## R5. 散落硬编码路径清单（迁移时易遗漏）

**触发条件：** 阶段 6 配置收口，清理 anima 字面量。

**当前 anima 路径硬编码散落在 ~13 处**（网络探子已枚举）。迁移时要么统一走 `family` 派发，要么至少确认这些字面量不影响 Krea-2 路径：

| 文件 | 行 | 字面量 |
|---|---|---|
| `configs/base.toml` | 1 | `pretrained_model_name_or_path = "models/diffusion_models/anima-base-v1.0.safetensors"` |
| `configs/base.toml` | 2 | `qwen3 = "models/text_encoders/qwen_3_06b_base.safetensors"` |
| `configs/base.toml` | 3 | `vae = "models/vae/qwen_image_vae.safetensors"` |
| `configs/base.toml` | 11 | `network_module = "networks.lora_anima"` |
| `scripts/tasks/_common.py` | 934 | `pretrained_model_name_or_path` 默认 |
| `scripts/tasks/_common.py` | 937 | `qwen3` 默认 |
| `scripts/tasks/_common.py` | 939 | `vae` 默认 |
| `scripts/tasks/preprocess.py` | 602, 662, 666, 811-815 | 预处理路径默认 |
| `scripts/experimental_tasks/training.py` | 151, 167, 171, 195, 199 | 实验训练路径默认 |
| `scripts/tasks/utilities.py` | 109 | 工具路径默认 |
| `library/inference/request.py` | 13-15 | 推理请求默认 |
| `scripts/distill_turbo/config.py` | 369 | 蒸馏配置 |
| `scripts/distill_mod/distill.py` | 100 | mod 蒸馏 |
| `scripts/merge_to_dit.py` | 176 | 合并脚本 |
| `scripts/dcw/collect_fei_sidecar.py` | 327, 332 | DCW sidecar |
| `scripts/dcw/measure_bias_args.py` | 20, 174, 212 | DCW 测量 |
| `scripts/distill_spd.py` | 214 | SPD 蒸馏 |
| `scripts/edit.py` | 13-15 | 编辑脚本 |

**cache 后缀硬编码：**

| 文件 | 行 | 字面量 |
|---|---|---|
| `library/io/cache.py` | 22 | `LATENT_CACHE_SUFFIX = "_anima.npz"` |
| `library/io/cache.py` | 23 | `TE_CACHE_SUFFIX = "_anima_te.safetensors"` |
| `library/anima/strategy.py` | 599 | `ANIMA_LATENTS_NPZ_SUFFIX` |
| `library/anima/strategy.py` | 215 | `ANIMA_TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX` |
| `library/datasets/feature_sidecars.py` | 43, 136, 243 | `_anima_{encoder}.safetensors` 等 |

**ModelSpec 架构标签：**

| 文件 | 行 | 字面量 |
|---|---|---|
| `library/models/sai_spec.py` | 20-25 | `ARCH_ANIMA_PREVIEW` / `ARCH_ANIMA_UNKNOWN` / `IMPL_ANIMA`（写入 safetensors metadata） |

**缓解：**
1. cache 后缀改读 `family.cache_suffix`，但**保留 `_anima.*` 字面量作为 anima family 的值**（向后兼容旧缓存）。
2. 路径字面量统一走 `family` 派发，散落处改读 family 提供的默认路径。
3. ModelSpec 标签：Krea-2 新增 `ARCH_KREA2` / `IMPL_KREA2`，anima 标签不动。
4. 蒸馏 / DCW / 编辑 / SPD 脚本若首日不支持 Krea-2（见主提案 §1 非目标），这些字面量可暂不动，但要加守卫：`model_family != "anima"` 时拒绝执行。

**何时该暂停重估：** 若发现某处字面量被 Krea-2 路径意外触发（如 `scripts/edit.py` 在 Krea-2 下被调用）→ 立即加 family 守卫，不要让 anima 路径假设泄漏到 Krea-2。

---

## R6. CONSTANT_TOKEN_BUCKETS 是否复用

**触发条件：** 阶段 2 确认 VAE 后、阶段 4 训练前。

**Anima 的现状：**
- `library/datasets/buckets.py:27-54`：24 个 (W,H) 分辨率，4032/4200 两个 token-count family。
- token count = `(W//16)*(H//16)`（16px patch grid = 8x VAE × 2 patch）。
- native flatten（`models.py:2528-2534`）让 dynamo 按 token count 复用图，24 bucket → 2 个编译图。
- `DCW_ASPECT_BUCKETS`（`buckets.py:69-75`）前 5 个，绑已发布 fusion-head checkpoint，**不重排**。

**Krea-2 的情况：**
- patch=2、VAE 8x，与 anima 一致 → 16px token grid，token count 公式相同。
- **理论上可复用同表**，但需确认：
  1. Krea-2 RoPE 上限是否支持 2016px（anima rope 上限 256 patch，`buckets.py:15-16`）？Krea-2 README 说 Raw "up to 1K"，2016px 可能超其训练分布。
  2. Krea-2 是否对分辨率有不同偏好（技术报告说预训练跨 256/512/1024 三档）？
  3. native flatten 的伪 5D 展平是否适用于 SingleStreamBlock？（anima 的展平在 `models.py:2528-2534`，Krea-2 DiT 要重写展平逻辑）

**缓解：**
1. 阶段 2 VAE 验证后，先确认 token count 公式一致。
2. 阶段 0 跑官方推理时，测试不同分辨率（512/768/1024）的输出质量，确定 Krea-2 的合理分辨率范围。
3. **首日可先用 anima 的 bucket 表子集**（只取 ≤1024 的那些），native flatten 逻辑在 Krea-2 DiT 里重写。
4. DCW 表保持冻结，Krea-2 不用 DCW（首日非目标）。

**何时该暂停重估：** 若 Krea-2 RoPE 不支持 anima 的大分辨率 bucket → 砍掉大分辨率 bucket，用子集，不要硬塞。

---

## R7. 13B 参数显存与训练可行性

**触发条件：** 阶段 4 训练前，阶段 5 推理前。

**现状：**
- Anima ~2-3B，Krea-2-Raw 13B（features=6144 / 28层 / GQA 48:12）。
- 即便 LoRA 只训少量参数，base model 前向仍要驻留显存（或 offload）。
- 现有 preset：`default`、`low_vram`、`low_vram_blockswap`、`balanced_16g`、`graft`、`half`、`quarter`、`tenth`、`debug`。

**待评估：**
1. 13B base model bf16 前向需要多少显存？哪些 preset 能承载？
2. block swap / offload（`library/runtime/offloading.py`）在 13B 上是否够用？
3. 是否需要新增 `low_vram_krea2` preset？
4. WebUI 训练队列的 GPU 白名单（`web/services/training/gpu.py`）是否要按 family 区分？

**缓解：**
1. 阶段 4 前用 `library/runtime/peak_probe.py` 实测 13B 前向峰值。
2. 优先复用 `low_vram_blockswap` preset，调 block swap 深度。
3. 若现有 preset 不够，新增 Krea-2 专用 preset（不破坏 anima preset）。
4. 训练 dtype：Krea-2 推荐 bf16，与 anima 一致。

**何时该暂停重估：** 若 13B 即便 block swap 也无法在目标 GPU（如 24GB 4090）上训练 → 缩小 LoRA rank / 限制可训层 / 评估是否需要 8bit base 加载（但 Krea-2 官方无 int8 base，需自建，工程量大）。

---

## R8. diffusers 主干版本依赖与备选方案

**触发条件：** 阶段 2 选 DiT 加载路径。

**现状：**
- Krea-2 model card：`pip install git+https://github.com/huggingface/diffusers.git`，**未进 release**。
- `Krea2Pipeline` 类存在于 diffusers 主干，但 release 节点未知。
- 依赖：`transformers==4.57.1`（pin 死）、`diffusers>=0.32.0`、`torch>=2.9`、Python>=3.12。

**两条加载路径：**

| 路径 | 优点 | 缺点 |
|---|---|---|
| **A. 借 diffusers `Krea2Transformer2DModel`** | 复用 diffusers 的 key 映射、config 加载、可能复用 attn_fuse | 依赖主干版本，可能与 anima 现有 diffusers 冲突；release 不稳定 |
| **B. 从 `krea-ai/krea-2` GitHub 裸移植 `mmdit.py`** | 无 diffusers 依赖，自包含 | 要自己写 diffusers 三分片 + index.json 加载、key 映射 |

**缓解：**
1. 阶段 0 先试路径 A，确认 `Krea2Pipeline` 能跑通官方推理。
2. 若路径 A 与 anima diffusers 版本冲突 → 回退路径 B，从 GitHub 移植 `SingleStreamDiT`。
3. `transformers==4.57.1` pin：检查 anima 当前 transformers 版本，若冲突考虑单独 venv 或协商（anima 是否能升到 4.57.1）。
4. 加载器 `library/models/krea2_raw/weights.py` 复用 `networks/lora_utils.py::load_safetensors_with_lora` 的 `00001-of-00004` 分片展开机制（Krea-2 是 `00001-of-00003`，同模式）。

**何时该暂停重估：** 若路径 A B 都因依赖冲突无法在现有环境跑 → 暂停，先解决依赖版本问题（可能需独立 venv），不要在冲突环境下硬写代码。

---

## R9. HF gated 访问与权重下载

**触发条件：** 阶段 0 下载权重。

**现状：**
- `krea/Krea-2-Raw` 是 gated repo，需 HF 登录并接受 Krea 2 Community License。
- 权重 ~33GB（transformer 三分片 ~24.5GB + text_encoder ~8.3GB + VAE ~484MB），不含重复的 `raw.safetensors`。
- License：可商用许可需联系 `opensource@krea.ai`，代码部分 Apache-2.0。

**缓解：**
1. `scripts/tasks/downloads.py` 新增 `cmd_download_krea2`，shell-out 到 `hf download krea/Krea-2-Raw`（需先 `hf auth login` + 接受 license）。
2. `download-models` 聚合命令（`downloads.py:264-298`）加 Krea-2 项，但**默认 continue-on-failure**，未接受 license 时不阻塞其他下载。
3. preflight：`model_family="krea2_raw"` 但权重缺失时，硬拒绝并提示 `python tasks.py download-krea2`。
4. 不要把 license 文本、PAT、cookie 写进仓库文件（AGENTS.md Git 规则）。

**何时该暂停重估：** 若用户未接受 license → 阻塞在阶段 0，提示用户先接受，不要绕过。

---

## R10. Anima `library/anima/` 搬家型重构的独立风险

**触发条件：** 若选择先做 anima 搬迁（`library/anima/` → `library/models/anima/`）再落 Krea-2。

**现状：** 主提案 §3.1 画了 `library/models/<family>/` 命名空间，但 anima 搬迁本身是独立前置子提案，不阻塞 Krea-2 设计。

**风险：**
1. 搬迁若一轮同时改架构和改行为，违反 AGENTS.md "搬家型重构" 守则。
2. `library/anima/` 被 `library/training/`、`library/inference/`、`networks/lora_anima/`、`scripts/` 大量 import，搬迁要同步改所有 import。
3. 现有测试（`tests/test_ensure_text_strategies.py` 等）硬编码 `library.anima` 路径。

**缓解：**
1. 搬迁作为独立子提案，**保持行为不变**：先抽模块到新位置，旧路径留兼容 shim，跑全量测试，再清 shim。
2. Krea-2 落地可暂不依赖 anima 搬迁完成——`library/models/krea2_raw/` 独立新建，`model_family="krea2_raw"` 走新路径，`model_family="anima"` 继续走 `library/anima/`。两者在 `library/models/factory.py` 派发，互不阻塞。
3. anima 搬迁完成后再做 import 收敛。

**何时该暂停重估：** 若搬迁过程中测试大面积红 → 回退，分更小批次搬迁，不要强行推进。

---

## 通用执行纪律

1. **先非目标再方案**：每个阶段开工前，先重读主提案 §1 非目标，确认不越界。
2. **搬家型重构**：任何 anima 侧改动保持行为不变，先搬后改，不在一轮同时改架构和改行为。
3. **热点文件守则**：`train.py`、`library/inference/generation.py`、`networks/lora_anima/network.py` 等热点文件改动 >50 行必须拆子模块（AGENTS.md 反上帝代码守则）。
4. **每个阶段回归 anima**：`model_family="anima"` 路径必须行为不变，跑相关 pytest。
5. **风险触发即登记**：执行中发现新风险，追加到本文件 R11、R12...，不要只在对话里讨论。
6. **代码事实优先**：本文件所有 file:line 引用基于 2026-08-08 的 main，执行前以实时源码为准核对。
