# Anima → Krea-2-Raw 迁移提案

> **2026-08-31 当前状态说明：** 训练、标准单提示词推理、Web 模型配置、NF4、
> compile/Flash 和 checkpoint 主链已经落地；本文保留迁移阶段的原始计划与历史状态，
> 其中“推理 dispatch 待续”等阶段性描述不再代表当前代码。当前兼容边界、风险和测试缺口
> 以 [`../multi_model_support.md`](../../../docs/multi_model_support.md) 和
> [`../findings/backend_multi_model_audit_20260810.md`](../../../docs/findings/backend_multi_model_audit_20260810.md)
> 的后续加固记录为准。

状态：历史迁移计划 / 核心阶段 0-6 已落地 / 待归档
适用版本：正文是 2026-08-08 的迁移计划；当前能力以多模型说明和 registry 为准
日期：2026-08-08
入口命令：见各阶段 findings；探针入口 `.venv/bin/python scripts/krea2/probe_{train,sample,blockswap,checkpoint}.py`

相关代码：

- 迁移目标 DiT：`library/anima/models.py`、`library/anima/weights.py`、`library/anima/strategy.py`
- 训练前向路径（迁移最大阻塞点）：`library/training/noise_target.py`、`library/training/model_loading.py`、`library/training/trainer_network_mixin.py`
- 梯度检查点（阶段 4 子设计）：`library/anima/models.py::UnslothOffloadedGradientCheckpointer`、`library/runtime/offloading.py`、`tests/test_compile_checkpoint_block_swap_hot.py`，落地背景见 [已归档设计](krea2_raw_gradient_checkpointing.md)
- 推理路径：`library/inference/generation.py`、`library/inference/models.py`、`library/inference/text.py`
- LoRA 挂载与配置：`networks/lora_anima/network.py`、`networks/lora_anima/config.py`、`networks/attn_fuse.py`、`library/runtime/token_counts.py`、`library/io/cache.py`
- 配置与下载：`configs/base.toml`、`scripts/tasks/downloads.py`、`scripts/tasks/_common.py`

相关文档：

- 当前多模型实现边界：[`../multi_model_support.md`](../../../docs/multi_model_support.md)（本提案是
  Krea-2-Raw 迁移的历史设计来源；最终实现采用静态 registry + 完整 handler dispatch，
  没有照搬本文拟议的动态 `ModelFamily` Protocol）
- 提案索引：[`README.md`](README.md)
- Anima 架构基线：[`../structure/anima.md`](../../../docs/structure/anima.md)

> **一句话：** Krea-2-Raw 与 Anima 在底层几何上同源（同一个 Qwen-Image VAE、16ch latent、5D 布局、patch=2），VAE/latent/cache/训练框架这大半套基建可平移；工程量集中在 **Qwen3-VL 文本编码器链路** 和 **single-stream MMDiT 架构本体 + 加载器** 两块。
> 本提案不写死 anima 替换，而是沿 `ModelFamily` 边界让 Krea-2-Raw 成为可切换的第二个 family。**首日只支持 LoRA 家族**，编辑类 adapter（IP-Adapter / EasyControl / DirectEdit）在 Krea-2 上首日不支持。

---

## 0. 目标

把当前面向 Anima 的 LoRA 训练器迁移到 [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw)，使其成为与 Anima 并存的可切换 `model_family`，而非替换。

落地标准（阶段 4 完成时）：

- `configs/base.toml` 可通过 `model_family = "krea2_raw"` 切到 Krea-2-Raw。
- Krea-2-Raw 上 plain LoRA 能跑通单 prompt 过拟合 + 小数据集 sweep（`python tasks.py lora` + WebUI 队列）。
- Krea-2-Raw 推理跑通官方 flow-matching 采样（`python tasks.py test`）。
- Anima 路径行为不变（`model_family = "anima"` 默认），现有 `_anima.npz` 缓存不受污染。

## 1. 非目标（明确不做）

1. **不替换 Anima。** Krea-2-Raw 作为第二个 family 并存，Anima 保持默认。
2. **首日不支持编辑类 adapter。** IP-Adapter / EasyControl / postfix / DirectEdit / img2img / inversion 在 Krea-2 上首日均不做——Krea-2 官方 `autoencoder.py` 只暴露 `decode`，编辑链路工程量陡增，留待后续提案。
3. **首日不支持 Turbo 蒸馏 / DCW / Spectrum / mod-guidance。** 这些是 Anima 特有 cross-attn / AdaLN 假设的产物，Krea-2 single-stream + light bias modulation 不直接对应，暂不做。
4. **不引入 Qwen3-VL 的图像输入通道。** Qwen3-VL 虽是 VLM，但 Krea-2 把它当文本编码器用（产出文本 conditioning），不喂参考图。style reference / prompt expansion 等 Krea 产品功能不在本提案范围。
5. **不写 ComfyUI custom node 的 Krea-2 vendor 副本。** 这是独立关切，不随训练和推理
   family dispatch 自动获得支持。
6. **不重排 `DCW_ASPECT_BUCKETS`。** 该表绑已发布 fusion-head checkpoint，即使 Krea-2 用新 bucket 表也保持 anima 的 DCW 表冻结。

## 2. 现状锚点：架构对照

下表是迁移可行性的地基。**两条被网络调研一度判成"高风险"的点，代码核实后是利好**——已用 ✅ 标出。

| 维度 | Anima（当前） | Krea-2-Raw（目标） | 是否一致 |
|---|---|---|---|
| 模型类型 | Cosmos-Predict2 DiT，dual-stream（self+cross+MLP） | single-stream MMDiT（text/image 共享 attn+MLP） | ❌ 不同 |
| 参数量 | ~2-3B（28层×2048dim×16头） | 13B（28层×6144dim×48头，GQA 48:12） | ❌ 更大更宽 |
| 调制 | AdaLN-LoRA（256 bottleneck，3路 up） | light modulation with bias（每 block 一个可调 bias） | ❌ 不同 |
| MLP | GPT2 风格 Linear→GELU→Linear | SwiGLU 4× + 门控 sigmoid attention | ❌ 不同 |
| **Latent channels** | **16**（`library/models/qwen_vae.py:1013` `z_dim=16`） | **16**（Qwen-Image VAE） | ✅ 一致 |
| **VAE** | **AutoencoderKLQwenImage**（8x，per-channel mean/std） | **AutoencoderKLQwenImage**（8x，per-channel mean/std） | ✅ **同一个** |
| **Latent 布局** | 5D `(B,C,T=1,H,W)`，dim 2 是单例时间轴 | 5D `(B,C,1,H,W)` | ✅ 一致 |
| **Patch** | `patch_spatial=2`（`library/anima/weights.py:171`）→ 16px token grid | `patch=2` → 16px token grid | ✅ 一致 |
| 文本编码器 | Qwen3-0.6B + T5 tokenizer + 6层 LLM Adapter 桥到 T5 space | Qwen3-VL-4B VLM + 12层 MFA + chat template caption | ❌ 完全不同 |
| 文本 padding | max-pad 512，zero 位作 cross-attn sink（裁剪→黑图） | VLM，多层聚合，padding 行为未确认 | ⚠️ 高风险 |
| 采样 | flow-matching | flow-matching v-pred + logit-normal shift + mu shift | ⚠️ 同族，参数不同 |
| 权重格式 | 单文件 `.safetensors`，前缀 `net.`/`model.diffusion_model.` | diffusers 三分片 + index.json | ❌ 不同 |
| 训练代码 | 有（本仓库） | 官方只有推理，无训练脚本 | — |
| img2img/编辑 | 有（DirectEdit/inversion/IP-Adapter） | 官方无，VAE 只暴露 `decode` | ❌ 缺失（本提案不做） |

**核心利好（VAE 与 latent 几何全等）的代码证据**：

- Anima VAE 就是 `AutoencoderKLQwenImage`，`library/models/qwen_vae.py:997` 类定义，`z_dim=16`、`spatial_compression_ratio=8`、`temperal_downsample=[False,True,True]`。
- Anima 已实现完整 `encode_pixels_to_latents`（`library/models/qwen_vae.py:1398-1439`），含 4D↔5D 切换（`:1412-1414` unsqueeze、`:1436-1437` squeeze）和 per-channel mean/std 归一化（`:1426-1434`）。
- Krea-2 官方 `autoencoder.py` 只写 `decode()`，但用的是同一个 `AutoencoderKLQwenImage` from `Qwen/Qwen-Image`。**anima 已写好且经训练验证的 encode 路径直接复用，无需自建。**（[迁移注意事项](krea2_raw_migration_notes.md) R2 记录了 encode/decode 互逆验证。）
- `unsqueeze(2)`/`squeeze(2)` 这套 5D 不变量、native flatten、shape 计算、token grid（16px）几何上完全平移，不用重算 latent 维度。

## 3. 设计：沿 `ModelFamily` 边界落地

以下是 2026-08-08 的拟议边界。Krea-2-Raw 后来成为多模型实现的首个第二 family，但当前
代码采用 `ModelFamilySpec` registry 和显式 handler 表；实际目录与协议以
[`../multi_model_support.md`](../../../docs/multi_model_support.md) 为准。

### 3.1 命名空间与目录

```
library/models/
├── base.py                 # ModelFamily protocol（首次落地，从 sketch 具体化）
├── factory.py              # load_family(name) -> ModelFamily
├── anima/                  # 从 library/anima/ 搬迁，保持行为不变
│   ├── dit.py              # was library/anima/models.py
│   ├── weights.py
│   ├── strategy.py
│   └── lora_targets.py
└── krea2_raw/              # 新增
    ├── dit.py              # SingleStreamDiT（移植自 krea-ai/krea-2 mmdit.py，或借 diffusers）
    ├── weights.py          # diffusers 三分片加载 + key 映射
    ├── strategy.py         # Qwen3-VL tokenize + MFA encode + caching
    └── lora_targets.py    # single-stream block 的 LoRA 注入点
```

> **搬家型重构约束（AGENTS.md）：** `library/anima/` 的搬迁必须保持行为不变，先抽模块再补测试再清旧 facade，不在一轮里同时改架构和改行为。Anima 搬迁本身可作为独立前置子提案，与本提案解耦推进。

### 3.2 `ModelFamily` Protocol（从 sketch 具体化）

```python
class ModelFamily(Protocol):
    name: str                          # "anima" | "krea2_raw"
    cache_suffix: str                  # "_anima" | "_krea2"
    latent_channels: int               # 16（两者一致，但仍显式）
    vae_spatial_compression: int       # 8（两者一致）
    patch_spatial: int                 # 2（两者一致）

    def load_dit(self, args) -> nn.Module: ...
    def load_text_encoder(self, args) -> nn.Module: ...
    def load_vae(self, args) -> nn.Module: ...

    def tokenize_strategy(self) -> TokenizeStrategy: ...
    def text_encoding_strategy(self) -> TextEncodingStrategy: ...

    def lora_target_spec(self) -> LoRATargetSpec: ...      # block regex + excludes + fuse specs
    def forward_for_loss(self, dit, latents, text_emb, t, **kw) -> Tensor: ...  # 承重接口
```

`forward_for_loss` 是 [`../multi_model_support.md`](../../../docs/multi_model_support.md) 标注的承重接口——Anima 的实现伸手进 `unet.llm_adapter` / `_mod_guidance_*` / fused-projection 假设 / cross-attn LSE 不变量（`library/training/noise_target.py`），Krea-2-Raw 的实现必须重写，回答 single-stream MMDiT 的 forward 怎么调。

### 3.3 配置面

`configs/base.toml` 新增：

```toml
model_family = "anima"   # 默认；切到 "krea2_raw" 走 Krea-2 路径
```

模型路径键沿用现有 `pretrained_model_name_or_path` / `qwen3` / `vae`，但语义随 family 切换：

- `model_family = "anima"` → 三键指向 anima DiT/Qwen3/Qwen VAE（现状不变）。
- `model_family = "krea2_raw"` → 三键指向 Krea-2 transformer 目录 / Qwen3-VL 目录 / Qwen-Image VAE。

> **注意：** Krea-2-Raw 是 diffusers 三分片格式（`transformer/diffusion_pytorch_model-0000N-of-00003.safetensors` + `index.json`），不是单文件。加载器要处理 `index.json` 分片展开，可复用 `networks/lora_utils.py::load_safetensors_with_lora` 的 `00001-of-00004` 分片机制。

### 3.4 cache suffix 隔离

`library/io/cache.py:22-23` 当前 `_anima.npz` / `_anima_te.safetensors` 硬编码。迁移后读 `family.cache_suffix`：

- Krea-2 用 `_krea2.npz` / `_krea2_te.safetensors`，与 anima 缓存在 `post_image_dataset/` 共存不冲突。
- Anima 旧缓存保持原后缀，向后兼容；当前 cache 隔离契约见
  [`../multi_model_support.md`](../../../docs/multi_model_support.md)。

## 4. 迁移路径：按子系统分层

按 [`../multi_model_support.md`](../../../docs/multi_model_support.md) 的耦合地图自上而下推进。每层标注迁移工作量。

### 4.1 大改（替换 / 重写）

| 子系统 | 当前文件 | 迁移动作 | 工作量 |
|---|---|---|---|
| 文本编码器链路 | `library/anima/strategy.py`（`AnimaTokenizeStrategy`/`AnimaTextEncodingStrategy`/`AnimaTextEncoderOutputsCachingStrategy`）、`library/anima/text_strategies.py`、`library/preprocess/text.py`、`library/inference/text.py` | 新写 `library/models/krea2_raw/strategy.py`：Qwen3-VL-4B tokenize（带 chat template caption）+ 12 层 MFA encode + caching。padding/sink 不变量重新推导（[注意事项 R1](krea2_raw_migration_notes.md)） | 大 |
| DiT 架构本体 | `library/anima/models.py`（`Anima`/`Block`/`Attention`/`PatchEmbed`/`FinalLayer`/`VideoRopePosition3DEmb`/`LLMAdapter*`，Cosmos-Predict2 专属） | 新写 `library/models/krea2_raw/dit.py`：移植 `SingleStreamDiT`/`SingleStreamBlock`（GQA 48:12、SwiGLU、门控 sigmoid、light bias modulation、3-axis Axial RoPE），来源 `krea-ai/krea-2` GitHub `mmdit.py` 或 diffusers `Krea2Transformer2DModel` | 大 |
| 加载器 | `library/anima/weights.py`（`dit_config` 写死 `:163-176`、`_DIT_PREFIXES=(net.,model.diffusion_model.)`、`_dit_concat_hook`/`_dit_rename_hook` qkv/kv/adaln fuse） | 新写 `library/models/krea2_raw/weights.py`：diffusers 三分片 + `index.json` 加载、Krea-2 key 命名映射（不再有 `_dit_concat_hook` 那套，但若 diffusers 已 fuse 则复用 `attn_fuse.py` spec） | 大 |
| LoRA 注入点 | `networks/lora_anima/network.py:42-55` 容器白名单、`networks/lora_anima/config.py:197-200` `_DEFAULT_EXCLUDE`、`networks/lora_anima/routers.py:14` `CROSSATTN_EMB_DIM=1024`、`networks/lora_anima/targeting.py:17` `_BLOCK_IDX_RE=blocks.N.`、`networks/attn_fuse.py:51-54` fuse spec | 新建 `library/models/krea2_raw/lora_targets.py`：按 single-stream block 命名重做 target spec。**single-stream 无独立 cross_attn，text/image 共享 attention**——fuse spec 和三轴路由策略要重新设计（[注意事项 R4](krea2_raw_migration_notes.md)） | 大 |

### 4.2 中改

| 子系统 | 当前文件 | 迁移动作 |
|---|---|---|
| 训练前向路径 | `library/training/noise_target.py`（5D `unsqueeze(2)`/`squeeze(2)` 调用点 `:64,180-182,375`、两种 `anima(...)` 签名 `:186,207`、`compute_loss_weighting_for_anima`）、`library/training/model_loading.py`、`library/training/anima_strategies.py`、`library/training/trainer_network_mixin.py` | 走 `family.forward_for_loss`，不再直接 import `library.anima`；5D 约定 Krea-2 一致可保留；loss weighting helper 按新 family 实现 |
| 推理生成 | `library/inference/generation.py`（denoising loop `:749-890`，`anima(latents,...)` `:783,806`）、`library/inference/models.py`、`library/inference/adapters.py`（`iter_hydra_networks` 读 `_hydra_networks` 等、`compute_and_set_hydra_fei` 5D squeeze `:204-205`） | denoising loop 改走 family 的 sampler；flow-matching + mu shift 参数化；FEI/router 钩子首日仅 plain LoRA 可禁用 |
| harness | `library/runtime/harness.py`（`build_anima` `:64`、`compile_blocks_for_training` `:708`） | 改为 `build_dit(family, ...)`，6 步顺序不变量（apply_to → load_weights → grad-ckpt → fp32 拥差 → convrot → compile）保持，补运行时断言防乱序（[注意事项 R3](krea2_raw_migration_notes.md)） |

### 4.3 小改（配置收口）

| 子系统 | 当前文件 | 迁移动作 |
|---|---|---|
| 配置 | `configs/base.toml:1-11`（三路径 + `network_module`） | 加 `model_family` 键，三路径语义随 family 切换 |
| 散落硬编码 | `scripts/tasks/_common.py:934` 等 ~13 处字面量 | 统一走 `family` 派发，清理字面量（[注意事项 R5](krea2_raw_migration_notes.md) 有完整清单） |
| cache 后缀 | `library/io/cache.py:22-23`、`library/anima/strategy.py:599,215`、`library/datasets/feature_sidecars.py:43,136,243` | 读 `family.cache_suffix` |
| 下载命令 | `scripts/tasks/downloads.py::cmd_download_anima`（`:135-170`，repo `circlestone-labs/Anima`） | 新增 `cmd_download_krea2`，指向 `krea/Krea-2-Raw`（gated，需 license） |
| token 常量 | `library/runtime/token_counts.py:7` `ANIMA_VAE_SPATIAL_COMPRESSION=8` | 参数化为 `family.vae_spatial_compression`（Krea-2 也是 8，值不变但走 family） |
| bucket 表 | `library/datasets/buckets.py` `CONSTANT_TOKEN_BUCKETS`（4032/4200 两个 family） | Krea-2 patch=2 与 anima 一致，token count 公式不变；**是否复用同表待 VAE encode 验证后定**（[注意事项 R6](krea2_raw_migration_notes.md)） |

### 4.4 可保住不动（搬家型重构资产）

这些与 Anima 解耦，迁移时复用：

- **训练循环内核**：`library/training/loop.py`、`optimizers.py`、`schedulers.py`、`checkpoints.py`、`losses.py`、`gradient_sync.py`（对 anima 零引用）。
- **bootstrap 编排**：`library/training/bootstrap.py`（`create_and_apply_network` / `setup_optimizer_and_dataloader` / `prepare_with_accelerator`，仅 fp32 拥差检测沾 anima 类名 `:548-552`）。
- **compile after apply 顺序不变量**：`harness.py:64` 6 步顺序是通用 torch.compile 卫生。
- **LoRA 挂载骨架**：`networks/lora_anima/targeting.py`（容器 class name 白名单 + Linear/Conv2d 类型扫描）、monkey-patch、LoRA 数学、三轴路由 cfg、shared buffer、T-LoRA mask——与架构弱耦合。
- **attention 后端路由**：`networks/attention_dispatch.py`（SDPA/xformers/flash/sageattn/flex + BHLD/BLHD 转换）纯通用。
- **配置基建**：`library/config/*` TOML 合并链 / schema、`library/runtime/offloading.py` / `peak_probe.py` / `fei.py`。
- **WebUI 启动链**：模型参数走 TOML 流入，`web/services/training/launcher_*.py` 不碰。

## 5. 阶段计划

| 阶段 | 目标 | 退出条件 | 依赖 |
|---|---|---|---|
| **0 可行性确认** | 跑通官方推理 + dump key + 验证 VAE encode 互逆 | transformer state_dict key 清单到手；`AutoencoderKLQwenImage` encode→decode 互逆无损 | HF license + ~33GB 下载 ✅ 完成（[stage0 findings](../../../docs/findings/krea2_raw_migration_stage0_findings.md)）|
| **1 文本链路** | `library/models/krea2_raw/strategy.py` 跑通 Qwen3-VL tokenize + MFA + caching | 单 prompt encode 出 context，padding 行为确认（[R1](krea2_raw_migration_notes.md)） | 阶段 0 ✅ 完成（[stage1 findings](../../../docs/findings/krea2_raw_migration_stage1_findings.md)）|
| **2 DiT 本体 + 加载器** | `library/models/krea2_raw/dit.py` + `weights.py`（路径 B 裸移植 mmdit.py，非 diffusers；见 [R8](krea2_raw_migration_notes.md)） | 单 latent forward 通过，shape 对齐 | 阶段 0 ✅ 完成（[stage2 findings](../../../docs/findings/krea2_raw_migration_stage2_findings.md)）|
| **3 LoRA 注入** | `library/models/krea2_raw/lora_targets.py`，注入点按 single-stream 重做 | 单 block LoRA attach + forward 正常 | 阶段 2 ✅ 完成（[stage3 findings](../../../docs/findings/krea2_raw_migration_stage3_findings.md)）|
| **4 训练串通** | `family.forward_for_loss` + `noise_target.py` 改造 + 训练循环（含 grad-ckpt，落地背景见 [已归档设计](krea2_raw_gradient_checkpointing.md)） | 单 prompt 过拟合 loss 下降；小数据集 sweep | 阶段 1+2+3 ✅ 完成（[stage4 findings](../../../docs/findings/krea2_raw_migration_stage4_findings.md)）|
| **5 推理串通** | `generation.py` + flow-matching sampler + mu shift | `python tasks.py test` 出图 | 阶段 2+3 ✅ 完成（[stage5 findings](../../../docs/findings/krea2_raw_migration_stage5_findings.md)）|
| **6 配置/WebUI/下载/命名收口** | `model_family` 键 + 下载命令 + sidecar 命名 + WebUI 表单 + 测试 + docs | `model_family="krea2_raw"` 全链路可用，anima 路径回归通过 | ✅ 核心收口完成：块交换、检查点、`forward_for_loss`、1024 grad-ckpt、metadata/WebUI 与独立 inference runner 已落地；原始阶段快照见 [stage6 findings](../../../docs/findings/krea2_raw_migration_stage6_findings.md) |

## 6. 验证计划

沿用仓库 Phase 风格（参考 `turbo_anima_dmd_lora.md`）。

- **Phase 0（可行性确认）**：`Krea2Pipeline.from_pretrained("krea/Krea-2-Raw")` 跑官方推理；dump transformer `state_dict` keys；用 anima 的 `encode_pixels_to_latents` 对 Krea-2 VAE 权重 encode→decode 一张图验证互逆。
- **Phase 1（单 prompt 过拟合）**：1 张图 + 固定 prompt，LoRA 训练，loss 应单调下降，重建可见。
- **Phase 2（小数据集 sweep）**：10-30 张图，sweep rank / lr，验证不崩、不模式坍塌。
- **Phase 3（推理出图）**：训练得到的 LoRA + 官方 sampler 出图，与 base 对比风格可控。
- **回归测试（每个阶段）**：`model_family="anima"` 路径行为不变；`_anima.npz` 缓存不被污染；相关 pytest 通过。

定向测试候选（按 AGENTS.md 验证策略）：

- 网络注册/config/metadata：`tests/test_network_registry.py`、`tests/test_network_cfg.py`、`tests/test_factory_metadata_flow.py`、`tests/test_method_network_lifecycle.py`
- 训练 bootstrap：`tests/test_training_bootstrap.py`
- 文本策略/bucket 不变量：`tests/test_ensure_text_strategies.py`、`tests/test_constant_token_buckets.py`、`tests/test_native_flatten.py`
- 文档完整性：`timeout 60 .venv/bin/python -m pytest tests/test_documentation_integrity.py -q`

## 7. 风险与失败模式

完整风险登记册见 [`krea2_raw_migration_notes.md`](krea2_raw_migration_notes.md)。本节列顶层风险。

### R1: Qwen3-VL padding / attention sink 不变量是否成立（未确认，头号风险）

Anima 的 max-pad 512 + zero 位作 cross-attn sink 是硬不变量（`library/inference/text.py:21` `MAX_CROSSATTN_TOKENS=512`）。Qwen3-VL 是 VLM，多层 MFA 聚合，padding 行为完全未验证。若 VLM 的 pad token 不充当 sink，裁剪或 padding 方式错误会产生黑图或质量崩。**阶段 0/1 必须先确认。**

### R2: single-stream MMDiT 的 cross-attention 与 LoRA 注入点重设计

Anima 是 dual-stream（self + cross-attn 分离），LoRA fuse spec 是 `self_attn qkv` / `cross_attn kv`。Krea-2 single-stream text/image 共享 attention，没有独立 cross_attn。fuse spec、三轴路由的 `router_source="crossattn_emb"`、`CROSSATTN_EMB_DIM=1024` 都要重设计。**这是 LoRA 注入层最大的开放问题。**

### R3: 13B 参数 + GQA 的显存与训练可行性

Krea-2-Raw 13B（anima ~2-3B），GQA 48:12。即便 LoRA 只训少量参数，base model 前向仍要驻留。需评估 block swap / offload 是否够用、哪些 preset 可承载。**可能需要新增 low_vram preset。**

### R4: diffusers 主干版本依赖

`Krea2Pipeline` 尚未进 release（model card 写 `pip install git+https://github.com/huggingface/diffusers.git`）。若借 diffusers 加载，需 pin 主干版本，可能与 anima 现有 diffusers 版本冲突。**备选：直接从 `krea-ai/krea-2` GitHub 裸代码移植，不依赖 diffusers pipeline。**

### R5: 依赖版本冲突

Krea-2 钉死 `transformers==4.57.1`、`torch>=2.9`、Python>=3.12。Anima 训练器 Python 3.13 + uv，transformers 版本未 pin。**可能需单独 venv 或版本协商。**

## 8. 兼容性与 preflight

- **Anima 路径不变**：`model_family` 默认 `"anima"`，所有现有配置/缓存/命令行为不变。
- **cache suffix 隔离**：`_anima.npz` 与 `_krea2.npz` 共存，迁移不触发缓存重建。
- **硬拒绝**：`model_family="krea2_raw"` 但 Krea-2 权重未下载 / license 未接受时，preflight 应硬拒绝并提示 `python tasks.py download-krea2`。
- **WebUI 表单**：全局设置加 `model_family` 切换，切换后模型路径键的占位文本和下载按钮随之变。
- **DCW 表冻结**：无论 Krea-2 用什么 bucket 表，`DCW_ASPECT_BUCKETS`（`library/datasets/buckets.py:69-75`）保持不动。

## 9. 与其他提案的关系

- **[`../multi_model_support.md`](../../../docs/multi_model_support.md)**：当前实现事实入口。本提案是历史
  设计来源；`model_family`、cache 隔离和 family-specific forward 已落地，但最终形态是
  `ModelFamilySpec` registry + 显式 handler，不是本文草拟的动态 Protocol/factory 目录树。
- **[`convrot_w8a_optimization_roadmap.md`](../../../docs/proposal/convrot_w8a_optimization_roadmap.md)** / **[`convrot_w8a_training_plan.md`](../../../docs/proposal/convrot_w8a_training_plan.md)**：ConvRot 优化绑 Anima cross-attn / AdaLN 假设，Krea-2 首日不做 ConvRot，互不阻塞。
- **[`turbo_anima_dmd_lora.md`](../../../docs/proposal/turbo_anima_dmd_lora.md)**：Turbo 蒸馏绑 Anima，Krea-2 首日不做，互不阻塞。Krea-2 自家有 Turbo 变体（`krea/Krea-2-Turbo`），官方工作流是"LoRA 在 Raw 上训练，直接在 Turbo 上推理"，但本提案首日只支持 Raw。

## 10. 不在本提案范围

- `ModelFamily` Protocol 的最终签名；实际实现选择了静态 `ModelFamilySpec` registry 和
  operation-specific handler，本提案中的 Protocol 仅保留为历史设计。
- Anima `library/anima/` → `library/models/anima/` 的搬家型重构本身（可作独立前置子提案推进，不阻塞本提案设计）。
- ComfyUI custom node 的 Krea-2 vendor 副本（独立关切）。
- Krea-2 编辑 / img2img / inversion / IP-Adapter / EasyControl（官方无，工程量大，留待后续提案）。
- Krea-2 Turbo 蒸馏、DCW、Spectrum、mod-guidance（Anima 特有假设产物）。
- Qwen3-VL 图像输入通道 / style reference / prompt expansion（Krea 产品功能，非 base checkpoint 能力）。
