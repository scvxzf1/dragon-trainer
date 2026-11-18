# 多模型支持现状

状态：稳定架构说明
适用版本：当前 `dev` / `main` 工作树
相关代码：`library/models/family_registry.py`、`library/training/compat_matrix.py`、`library/training/moss_compat.py`、`library/training/pp/`、`library/models/<family>/`

本文是仓库多模型能力边界、跨族隔离契约与耦合现状的**唯一文档事实源**。其他文档（`AGENTS.md`、根 `README.md`、各功能与配置页）仅作摘要引用；若发生冲突，以实时源码（`MODEL_FAMILY_REGISTRY` 与兼容门禁）及本文为准。

## 当前已注册模型族总表

`library/models/family_registry.py::MODEL_FAMILY_REGISTRY` 静态注册全部支持的模型族。所有跨族分派（`dispatch_model_family`）要求完整覆盖全部已注册模型族；未知模型族、缺失 handler、未注册推理模式/采样器或跨族权重一律 fail-closed 报错，绝不静默回退 Anima。

| 模型族 (`model_family`) | 别名 | 训练任务与定位 | Adapter 范围 | 训练预览 | 独立 CLI 推理 (`inference.py`) | Attention 后端 | 双卡 PP v0 (`1f1b`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `anima` | `anima` | 默认生产模型族（T2I） | 完整 LoRA 家族 + Method Adapters | 支持（T2I） | `single` / `batch` / `interactive`；`euler` / `er_sde` / `lcm` | `flash` / `torch` / `mem_efficient` / `sageattn` / `flex` / `xformers` / `sdpa` | **已接线**（28/40 `blocks` 均分） |
| `krea2_raw` | `krea2`, `krea2_raw` | 第二生产模型族（T2I） | 开放网络变体选择（禁用 Method Adapters） | 支持（T2I） | `single` + `euler` | `flash` / `torch` / `sdpa`（含 `cudnn` 别名） | **已接线**（28 `blocks` 13/15 切分，支持 `pipeline_parallel_blocks_to_swap`） |
| `z_image` | `zimage`, `z_image` | 训练与训练预览 v1（T2I） | 开放网络变体选择（禁用 Method Adapters；真机证据以 plain LoRA 为主） | 支持（T2I） | 未注册（空集合） | `flash` (`flash_varlen`) / `torch` / `sdpa`（BF16） | 仅拓扑规划（`runtime_available=False`，入口拒绝） |
| `qwen_image_2_1` | `qwen_image_2_1`, `qwen_image_21`, `qwen21` | T2I / Edit 训练与预览（Edit 支持 1–4 张参考图） | `plain_lora_only=True`：T2I 允许 `lora`/`lokr`/`loha`；Edit 与独立推理仅 `lora` | 支持（T2I / Edit，最多 4 图） | `single` + `euler`（T2I / Edit，最多 1 个 plain LoRA） | `torch` / `sdpa` / `flash`（BF16，默认 `torch`） | 未声明（`None`，入口拒绝） |
| `moss_image` | `moss`, `moss_image` | 文生图 plain LoRA 首版（T2I） | `plain_lora_only=True`：仅 `lora` | 支持（T2I，Euler shift=5） | 未注册（空集合） | `torch` / `sdpa` | **已接线**（42 `layers` 21/21 均分） |

说明：
- WebUI 模型选择下拉框暴露全部 5 个模型族，但训练预览支持不代表通用生图测试（WebUI `image-test`）或 `batch`/`interactive` 独立推理已可用。
- Anima、Krea-2 与 Z-Image 的 `supported_network_specs=None` 表示接口层允许尝试网络变体（不套用“非 Anima 仅 plain LoRA”白名单），不代表所有变体组合已完成长训质量验收；`supports_method_adapters=False` 仍拦截 IP-Adapter、EasyControl、BYG 等依赖专属训练 hook 的方法。

## Pipeline Parallel（PP v0）现状与契约

`library/models/pipeline_parallel.py` 负责静态拓扑规划，`library/training/pp/`（门禁集中在 `gate.py` 与 `trainer_admission.py`）承载 stage-local 1F1B 运行时与专用训练会话：

- **已接线生产入口的 3 个模型族**（`TRAINER_PP_WIRED_FAMILIES = {"anima", "krea2_raw", "moss_image"}`）：
  `runtime_backend = "torch_pipelining_1f1b"`（`runtime_available = True`），`train_session._run_training_session` 在进入普通 Accelerate DDP 路径前将其分派至专用 PP 会话，分阶段构建 optimizer 与 per-rank checkpoint 包（`pp_checkpoint.json`）。
- **未接线模型族**：`z_image` 仅保留 30 层拓扑规划（`runtime_backend = None`），`qwen_image_2_1` 未声明 PP；在 CLI、WebUI 预检、队列与续训入口均以 `pipeline_parallel_runtime_unavailable` / `pp_trainer_path_unavailable` 具名拒绝，**绝不静默回退为普通 DDP**。
- **共同硬约束**：
  - 固定 `stages = 2`（`world_size == 2`）、`schedule = "1f1b"`、`mixed_precision = "fp16"` + `GradScaler`（拒绝 `bf16`/`fp32`、任何基座量化与 `torch_compile`）。
  - 要求预先完成磁盘 VAE/TE 缓存（`use_vae_cache=true`、`use_text_cache=true`）、`gradient_accumulation_steps=1`、`loss_type="l2"`、`weighting_scheme="none"`、关闭训练内 sample preview 与阶段调度。
  - 普通 `blocks_to_swap` 必须为 `0`。**仅 `krea2_raw` PP 额外支持每 stage 块交换参数 `pipeline_parallel_blocks_to_swap`**（设为 `0` 时保留 `KREA2_PP_SWAP_BLOCKS` 环境变量兼容回退，非 Krea-2 族设非零直接拒绝），且要求 `train_batch_size` 被 `pipeline_parallel_microbatches` 整除。
  - WebUI 交互与选卡诊断见 [分布式训练卡片说明](experimental/distributed-training-card.md) 及 [PP v0 前置契约](experimental/pipeline_parallel_v0_prerequisites.md)。

## 跨族路由、精度与缓存隔离契约

### 1. 预处理先行与训练加载路由
- `train.py` 保留 `AnimaTrainer` 兼容门面，底层由 `library/training/model_loading.py` 与 `batch_step.py::_get_noise_pred_and_target` 按 `model_family` 全覆盖分派。
- **强制磁盘缓存**：`library/training/train_session.py` 要求训练启动前已由 `preprocess` 生成完整的 latent 与 text cache；常规缓存齐备且不开启采样预览或 CMMD 时，`train.py` 跳过加载文本编码器与 VAE。
- **采样批次块交换生命周期**：训练预览在一批样张进出时统一切换块交换并恢复 adapter 状态与 RNG（见 [采样生命周期验收](findings/sampling_batch_lifecycle_20261002.md)）；通用预览 VAE 分块解码策略 `sample_vae_decode` 覆盖全部 5 个模型族（见 [训练预览显存策略](configuration/sample-preview-memory.md)）。

### 2. 精度改写与 NF4 边界
- **Pre-Ampere（`sm<80`）精度复检**：`library/training/precision_policy.py::resolve_mixed_precision` 在不支持原生 BF16 Tensor Core 的设备（如 V100 `sm70`、Turing `sm75`）上会将默认 `bf16` 改写为 `fp16`，随后 `train_session.py` 强制执行 `recheck_compat_after_precision_resolve` 二次门禁，确保改写后的精度仍满足目标模型族契约。
- **NF4 路径区分**：
  - Krea-2 支持 `base_compute="nf4"`（外层 `compute_dtype` 恒为 BF16）；在 `sm<80` 上可通过显式实验开关 `--nf4_dir1` + `--krea2_attn_fp16`（`configs/methods/krea2_lora_nf4_fp16.toml`）启用外层 BF16 + 内层 FP16 逐层缩放路径（豁免全局 `bf16->fp16` 改写）。
  - Anima、Krea-2、Z-Image、Qwen Image 2.1 另支持按权重文件自动识别的预量化主干 MLP 混合 BF16/NF4 实验格式（限 BF16 eager + full checkpoint，不等同于开放 `base_compute="nf4"`，详见 [多模型族混合 NF4](experimental/multi-family-mixed-nf4.md)）；MOSS-Image 不支持 DiT NF4（仅文本编码器缓存支持 NF4）。

### 3. 跨族产物严格隔离
不同模型族的权重、缓存与 adapter 无兼容面（回归测试见 `tests/test_adapter_family_isolation.py`）：

| 模型族 | 训练 `forward_for_loss` 输入形状 | Latent Cache 后缀 | Text Cache 后缀与 Schema |
| --- | --- | --- | --- |
| `anima` | 5D `(B, 16, 1, H/8, W/8)` | `_anima.npz` | `_anima_te.safetensors` (`anima_te_v1`，max-padded) |
| `krea2_raw` | 5D `(B, 16, 1, H/8, W/8)` | `_anima.npz`（共用 Qwen VAE 空间） | `_krea2_te.safetensors` (`krea2_te_v1`, width=2560) |
| `z_image` | 5D `(B, 16, 1, H/8, W/8)` | `_z_image.npz`（Flux 仿射 `(x-0.1159)*0.3611`） | `_z_image_te.safetensors` (`z_image_te_v1`, 512 tokens) |
| `qwen_image_2_1` | 4D `[B, 64, H/16, W/16]` | `_qwen_image_2_1.npz`（64ch RGBA VAE 逐通道归一化） | `_qwen_image_2_1_te.safetensors` (`qwen_image_2_1_te_v1`, width=4096) |
| `moss_image` | 4D `[B, 128, H/16, W/16]` | `_moss_image.npz`（stride 16 packed BN） | `_moss_image_te_nf4.safetensors` / `_bf16.safetensors` (`moss_image_te_v2`，含 `max_text_tokens` 真实长度) |

LoRA checkpoint 通过 `ss_model_family` metadata stamp（未标记旧权重视为 `anima`）由 `networks/lora_anima/factory.py::require_adapter_family` 在加载张量前严格校验，防止跨族“零模块命中却静默放行”。

## 各非默认模型族关键边界摘要

### Krea-2 (`krea2_raw`)
- 起始配置：`configs/methods/krea2_lora.toml`（`sm<80` NF4 内层 FP16 为 `configs/methods/krea2_lora_nf4_fp16.toml`）。
- 单流 28-block MMDiT + Qwen3-VL 文本缓存；默认 full checkpoint、fixed resident block compile（拒绝 dynamic-seq 与 CUDA Graph compile 模式）及 `flash`/`torch`/`sdpa` attention。
- 独立推理走 `library/models/krea2_raw/inference_runner.py`（`single` + `euler`，自动 flow shift），拒绝 Anima 专属增强参数。性能与显存消融结论见 [Krea-2 速度总结](findings/krea2_3080_speed_final.md) 及 [PP 显存与 WebUI 验收](findings/krea2_pp_webui_acceptance_20261117.md)。

### Z-Image (`z_image`)
- 起始配置：`configs/methods/z_image_lora.toml`。
- 官方 Diffusers 30-layer transformer + Qwen3 ChatML 倒数第二层 512-token 缓存 + Flux VAE（shift `6`）；`blocks_to_swap` 支持 `0` 或 `1..28`（仅交换 30 个 main layers，refiner 与首尾层常驻）。
- 要求所有层级 `caption_dropout_rate = 0`；不支持 `torch.compile`、selective/offloaded checkpoint、`base_compute="nf4"`（预量化混合文件除外）及独立推理。

### Qwen Image 2.1 (`qwen_image_2_1`)
- 起始配置：`configs/methods/qwen_image_2_1_lora.toml`。组件为 32-block DiT、Qwen3-VL 8B 与 64-channel RGBA VAE。单文件 TE 需同级或上层存在匹配的 Qwen3-VL `tokenizer/` 目录（ComfyUI `qwen25_tokenizer` 仅作词表不等价的兼容回退）。
- 支持 `qwen_image_2_1_task = "t2i" | "edit"`：
  - **训练**：BF16、full checkpoint、`torch`/`sdpa`（默认 `torch`）或 opt-in packed varlen `flash`、`transformer_blocks` 块交换与 apply 后 per-block `torch.compile`。T2I 开放实验 `lokr`/`loha`（[变体热测](findings/qwen_image_2_1_adapter_variants_90hx_20260929.md)）；Edit 数据集按同配对名“编辑前/编辑后”子集一对一配对，要求 `batch_size=1`、关闭随机几何/颜色增强且仅支持 plain LoRA（见 [模型训练能力与编辑数据集](configuration/model-training-capabilities.md)）。受限 NF4 Edit 编译实验开关见 [Qwen NF4 编译实验](experimental/qwen-nf4-compile.md)。
  - **训练预览与独立推理**：预览块交换覆盖与 VAE 解码见 [训练预览显存策略](configuration/sample-preview-memory.md#qwen-预览块交换覆盖)，可选 `use_kv_cache=true` 前缀缓存见 [A/B/C 验收](findings/qwen_sampling_kv_abc_90hx_20261002.md)。独立 CLI 推理（`library/models/qwen_image_2_1/inference_runner.py`）支持 `single` + `euler` 的 T2I/Edit（1–4 张参考图）、最多 1 个 plain LoRA 与 `--latent_path` 再解码（[独立推理验收](findings/qwen_standalone_inference_20261005.md)）；不支持 WebUI `image-test`、`batch`/`interactive` 或 LoKr/LoHa 推理。

### MOSS-Image (`moss_image`)
- 起始配置：`configs/methods/moss_image_lora.toml`（及分辨率/文本上限档位 `_t768`、`_t512`、`_tecap1024`，完整使用说明见 [MOSS-Image 配置](configuration/moss-image.md)）。
- 42-layer 4D DiT（training shift=1，预览 Euler shift=5），仅支持文生图 plain LoRA、`torch`/`sdpa` attention、full checkpoint、`blocks_to_swap = 0..40` 及全层级 `caption_dropout_rate = 0`。
- 文本编码器默认 `moss_te_quantization="nf4"`（可选 `"bf16"`），缓存保存真实 token 长度并将 `max_token_length`（`1..4096`）写入 `moss_image_te_v2` 身份；切换量化方式或 `max_token_length` 必须重建文本缓存。支持双卡 1F1B PP 训练会话，不支持 `torch.compile`、DiT NF4 或独立 CLI 推理。

## 仍存在的架构耦合与修改清单

| 区域 | 当前事实 | 维护要求 |
| --- | --- | --- |
| Family Registry | 静态注册 5 个模型族，`dispatch_model_family` 强制全覆盖 | 新增或扩展模型族必须补齐全部 handler、前端 `model-family.js` 离线目录与 fail-closed 测试 |
| 训练辅助层 | `model_loading.py`、`noise_target.py`、`train_session.py` 仍含显式模型族分支 | 新模型族走显式分支与专属 `family.py` 校验，不得伪装成 Anima 5D 布局或借用 Anima fallback |
| 兼容门禁 | `compat_matrix.py` + `moss_compat.py` + `pp/gate.py` 共享 CLI/WebUI 预检 | 不支持的能力必须明确拒绝或纳入经审计的 `compat_unverified_policy` 分类，禁止无声忽略 |
| 推理与预览 | Anima、Krea-2、Qwen 有独立推理；Z-Image、MOSS 仅有训练内预览 | `supported_inference_modes` 为空时必须拒绝，不得借用 Anima 采样器 |

相关核心回归测试：`tests/test_model_family_registry.py`、`tests/test_model_family_fail_closed.py`、`tests/test_adapter_family_isolation.py`、`tests/test_pipeline_parallel_multimodel.py`、`tests/test_compat_unverified_policy.py`。
