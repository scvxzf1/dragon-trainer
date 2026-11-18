# MOSS-Image 文生图 LoRA 首版

状态：基础实现入口。真机训练、质量和性能证据需另附验收报告；配置中
`blocks_to_swap=20` 只是调参起点，不能据此推断任意硬件能运行。

使用 `configs/methods/moss_image_lora.toml`，模型族为 `moss_image`（别名 `moss`），
WebUI 显示为 MOSS-Image。组件路径相对项目根目录：

- DiT：`models/moss_image/MOSS-Image-DiT`
- 文本编码器（BF16 原始权重）：`models/moss_image/MOSS-Image-TextEncoder-BF16`
- 文本编码器（静态 NF4 权重）：`models/moss_image/MOSS-Image-TextEncoder-NF4`
- VAE：`models/moss_image/MOSS-Image-VAE`

先准备本地组件权重和数据，再通过 `python tasks.py preprocess METHOD=moss_image_lora` 预先生成完整的磁盘 latent 与文本缓存。
训练要求磁盘缓存预先就绪：`train.py` 在缓存完整且未启用预览采样 / CMMD 时不加载文本编码器与 VAE（直接加载冻结 DiT）；若启用训练预览，采样提示词编码与 VAE 解码按预览阶段独立编排并释放。
首版只支持普通文生图 plain LoRA，文本编码器始终冻结。

## 精度与缓存

`moss_te_quantization="nf4"` 是默认文本编码器路径：从已下载的 BF16 权重按需量化，
使用 BF16 compute。量化或依赖失败直接报错，不会自动切换 BF16。
保留 `moss_te_quantization="bf16"` 作为未量化对照。该字段已进入 CLI 和配置 schema。

DiT 支持 `mixed_precision="bf16"`、`"fp16"` 或 `"no"`（FP32）。使用 FP16 + AMP 时设置：

```toml
mixed_precision = "fp16"
adaptive_precision = "off"
full_fp16 = false
```

该路径以 FP16 加载冻结的 DiT，LoRA 可训练参数保持 FP32；前向使用 Accelerate
autocast，反向及优化器更新使用 GradScaler。pre-Ampere GPU 遵循共享策略，将 BF16
自动切换为 FP16。`base_compute` 保持 `"bf16"`，普通 AMP 路径的实际加载精度由
`mixed_precision` 决定；`base_compute="nf4"` 不代表文本量化，会被拒绝。
VAE 推荐 FP32（方法配置 `no_half_vae=true`）。仅切换 DiT 精度可复用原有文本缓存。

必须开启 `use_text_cache=true` 与 `use_vae_cache=true`，两者都使用磁盘缓存。
不同模型族的缓存不得混用。NF4 与 BF16 文本缓存也必须隔离：切换文本编码器量化方式时
选择新的缓存目录或重新预处理。文本长度上限为 4096 token。

文本缓存身份还包含本次运行的 `max_token_length`（schema `moss_image_te_v2` 的
`max_text_tokens` 字段）：MOSS 保存真实长度（不套 Anima 的 max-padding 规则），
所以 4096 档写入的缓存不能被 1024 档复用，反之亦然。改 `max_token_length` 后重新
预处理文本缓存即可。旧 `v1` 缓存仍可读取（不含该字段），但无法证明其截断上限；
需要按档位比较时应重建为 `v2`。

## 训练与预览

训练使用完整 gradient checkpointing、`torch` 或 `sdpa` attention、均匀采样的
1000-step flow grid，training shift 固定为 1。`blocks_to_swap` 允许整数 0..40，
0 关闭交换；启用交换时训练及每个 dataset 的 batch size 都必须为 1。

MOSS 不使用 P2 加权，方法配置固定 `p2_gamma=0`。默认每轮保存一个可恢复训练状态，
保留最近两个状态；状态包含 optimizer、scheduler 和训练步数，可用于中断后续训。

训练预览使用 Euler，默认 50 步、CFG 3、shift 5。训练 shift 与预览 shift 分开处理。
预览默认值由 MOSS 专属预览实现提供；普通推理/image-test 的支持以 registry 为准。

配置预检（`library/training/moss_compat.py`）明确拒绝 compile、DiT NF4、其他 adapter、
选择性/卸载 checkpoint、自适应精度、AUTO block swap、weighted caption、TE 训练和高级损失。
训练只接受 L2 flow loss、`timestep_sampling="uniform"`、`weighting_scheme="none"`。
caption dropout 在顶层、general、dataset 和 subset 都必须为 0。
对于流水线并行（`pipeline_parallel`）：MOSS 已发布实验性双卡 1F1B `pipeline_parallel`
runtime 并接线专用训练会话（仅限 `PP=2` / `1F1B` / `FP16` / plain LoRA / 完整磁盘缓存，
且与 `blocks_to_swap > 0`、compile 等互斥；其余未接线形状仍由门禁具名拒绝），详见
[moss_image_pipeline_parallel_v0.md](../experimental/moss_image_pipeline_parallel_v0.md) 与
[moss_image_pp_trainer_wiring.md](../experimental/moss_image_pp_trainer_wiring.md)。
不要把其他模型的编译、路由或损失配置直接带入 MOSS；这些配置会报错。

## 训练前契约核对与执行模板

以下取值越界会在配置预检阶段直接报错，不会静默降级（来源 `library/training/moss_compat.py`）：

| 键 | 允许值 / 要求 |
| --- | --- |
| `model_family` | `moss_image`（别名 `moss`） |
| `mixed_precision` | `bf16` / `fp16` / `no` |
| `base_compute` | 仅接受 `bf16`（`nf4` 不代表文本编码器量化，会被拒绝） |
| `moss_te_quantization` | `nf4`（默认，BF16 compute）/ `bf16`；失败即报错，不静默回退 |
| `attn_mode` | `torch` / `sdpa` |
| `blocks_to_swap` | 整数 `0..40`（`0 <= n <= len(layers) - 2`，当前 42 层即上界 40）；`> 0` 时训练及所有 dataset 的 `batch_size` 必须为 1 |
| `block_swap_transfer_dtype` | 推荐 `bf16`；`int8` 会打印实验性 WARNING 且存在逐 step 数值偏移，不建议作为正式训练默认 |
| `max_token_length` | `1..4096` 整数，且写入 `moss_image_te_v2` 缓存身份（`max_text_tokens`） |
| 缓存开关 | `use_text_cache = true` 且 `use_vae_cache = true`（完整磁盘缓存） |
| 损失与采样 | `loss_type = "l2"`、`timestep_sampling = "uniform"`、`weighting_scheme = "none"`、`discrete_flow_shift = 1`、`p2_gamma = 0` |
| Adapter | 仅 `networks.lora_anima` plain LoRA；`network_train_unet_only = true`、`router_source = "none"` |
| Caption | `caption_dropout_rate` / `caption_tag_dropout_rate` 在顶层、general、dataset、subset 全为 `0`；不支持 weighted captions 或 caption shuffle |
| `pipeline_parallel` | 实验性放行 `PP=2` / `1F1B` / `FP16` / plain LoRA / 完整缓存（与 `blocks_to_swap > 0` 互斥） |

通用 CLI 执行序列模板：

```bash
# 0) 预检：核对合并后的模型族、档位、缓存与块交换配置
.venv/bin/python tasks.py print-config METHOD=moss_image_lora PRESET=default

# 1) 预处理：resize -> VAE(float32) -> TE(NF4, 含 max_token_length 缓存身份)
.venv/bin/python tasks.py preprocess METHOD=moss_image_lora

# 2) 单卡训练（切换档位直接改 --method moss_image_lora_t768 等，不要单改 resolution）
.venv/bin/python train.py --method moss_image_lora --preset default --max_train_epochs 3
```

修改 `max_token_length` 或 `moss_te_quantization` 后必须重新运行文本缓存预处理或切换独立缓存目录。

## 分辨率 / token 预算档位

默认 `moss_image_lora.toml` 的 `resolution = 1024` 会解析为 24 个常量 token bucket
（图像 token 恒为 4032 / 4200）。该路径下 `min_bucket_reso` / `max_bucket_reso` /
`bucket_reso_steps` 不参与过滤，改它们没有效果。

需要控制计算量时用自带档位配置，且**不要只改 `resolution`**：单改会让 bucket 表塌缩
（768 → 8 个 bucket，512 → 只剩 1 个 512×512）。

| 配置 | resolution | min/max | bucket 数 | 图像 token |
|---|---|---:|---:|---|
| `moss_image_lora.toml`（默认） | 1024 | 672 / 1536 | 24 | 4032–4200 |
| `moss_image_lora_t768.toml` | 768 | 504 / 1152 | 14 | 2256–2394 |
| `moss_image_lora_t512.toml` | 512 | 336 / 768 | 13 | 1008–1064 |
| `moss_image_lora_tecap1024.toml` | 1024 | 672 / 1536 | 24 | 4032–4200 |

`tecap1024` 只把 `max_token_length` 从 4096 降到 1024。MOSS 文本与图像 token 在同一条
流里，长 caption 是显存与 attention 成本的主要变量，先压文本上限通常比降分辨率更划算。

档位写在方法配置的 `[[datasets]]` 里，经 `load_dataset_config_from_base(method=...)`
覆盖 base.toml 的 dataset blueprint；显式传 `--dataset_config` 时该文件优先，方法里的
`[[datasets]]` 不再生效。

数据契约：1024 档要求源图短边 ≥ 768 px，否则 `bucket_no_upscale` 会回退到
`min_bucket_reso` 兜底（512×512 变成 672×672），产生 1764–3528 的带外 token 数。
768 / 512 档按比例约需 576 / 384 px。

推导、成本模型与真机热测试协议见
[分辨率 / 图像 token 预算档位](../findings/moss_image_token_budget_tiers_20261017.md)，
实测结果见 [档位热测试：CMP 50HX 七行实测](../findings/moss_image_tier_hot_50hx_20261018.md)，
静态契约测试是 [test_moss_image_token_tiers.py](../../tests/test_moss_image_token_tiers.py)。

## 验收范围

配置可解析、兼容规则与前端能力入口属于基础实现证据。
完整权重的 forward/backward、LoRA 保存重载、预览和恢复，以及 NF4/BF16 对照、
显存与性能，需要分别记录真机环境、命令和结果后再作为推荐依据。

已有 [CMP 50HX 十步基准](../findings/moss_image_bf16_baseline10_50hx_20261017.md)：
单张 256×256 缓存图、22-token 提示、BF16 配置、plain LoRA rank32、swap20，
10 次有效更新；报告保留逐步耗时、allocator 显存和整卡显存采样。

2026-10-17 的 FP16 + AMP 短测使用 CMP 50HX、完整 5.08B DiT、单张 256×256 缓存图、
22-token 提示、plain LoRA rank32、swap20 和正常 `train.py` 入口：4 次有效更新、零跳步，
loss 与保存的 LoRA 张量全部有限，GradScaler 保持 65536。另有 7 个定向测试覆盖
FP16/BF16/FP32 strict local loading、FP16 RMSNorm、CUDA AMP、checkpoint、block swap
及 LoRA 保存重载；3 个共享精度用例通过。
完整模型短测的命令与原始证据位于本机
`output/runs/moss-image-fp16-amp-smoke-20261017-120954-917068/review-result.md`。
该结果覆盖上述短程训练配置；长训练、其他分辨率及质量尚未验收。
