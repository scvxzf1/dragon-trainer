# MOSS-Image 训练前置清单（真实训练准备）

状态：已归档（早期训练前置核对快照）
适用版本：历史 `dev` 快照（`0d294920`，2026-10-19）
相关代码：`library/training/moss_compat.py`、`library/models/moss_image/`、
`scripts/moss_image/preprocess_te_cache.py`、`bench/moss_tier_hot/`
相关配置：`configs/methods/moss_image_lora{,_t768,_t512,_tecap1024}.toml`

本文只做三件事：把散落的 MOSS 记录收敛成一张索引、把训练前必须核对的契约和
环境写成可勾选清单、把还缺的用户决策列出来。结论与数字以被链接的原文档和实时源码为准，
本页不复制会漂移的实验数字。

## 1. 现有记录索引

| 文档 | 覆盖范围 | 状态 |
| --- | --- | --- |
| [moss-image.md](../../../docs/configuration/moss-image.md) | 组件路径、NF4/BF16 文本精度、缓存身份、训练/预览、预检拒绝项、分辨率档位、验收范围 | 首版基础实现（当前入口） |
| [moss_image_bf16_baseline10_50hx_20261017.md](../../../docs/findings/moss_image_bf16_baseline10_50hx_20261017.md) | 256×256 / 22 token / 10 步基准：11.3386 s/步、5.85 GiB | 真机短程实测 |
| [moss_image_token_budget_tiers_20261017.md](../../../docs/findings/moss_image_token_budget_tiers_20261017.md) | 1024/768/512 档定义、只改 resolution 的塌缩陷阱、数据短边契约、attention 成本模型、静态验收 | 协议 + 静态契约已验收 |
| [moss_image_tier_hot_50hx_20261018.md](../../../docs/findings/moss_image_tier_hot_50hx_20261018.md) | 12 行档位热测、OOM 边界、swap/int8 对照、作废首轮矩阵的尸体记录 | 实测（质量未测） |
| [moss_image_int8_blockswap_diagnosis_20261018.md](../../../docs/findings/moss_image_int8_blockswap_diagnosis_20261018.md) | int8 传输的 0.94 s / 3.5 GiB 归属、fused 还原、loss 偏移归因 | 诊断结论 |
| [int8_encoding_floor_50hx_20261019.md](../../../docs/findings/int8_encoding_floor_50hx_20261019.md) | 1 Byte/元素编码的误差上限（3A 否定、3B 依赖分布） | 实验 |
| [block_swap_family_payload_50hx_20261019.md](../../../docs/findings/block_swap_family_payload_50hx_20261019.md) | 各族块载荷与晋升算术（**表内没有 MOSS 行**） | 实验（跨族算术） |
| [multi_model_support.md](../../../docs/multi_model_support.md) | MOSS 能力边界：plain LoRA、无通用推理 mode/sampler、缓存隔离 | 能力边界 |
| [optimization-configs-current.md](../../../docs/optimization-configs-current.md) | `blocks_to_swap` / `block_swap_*` 键与 int8 实验性 WARNING | 配置事实清单 |

代码与验证入口：

- 契约门禁：`library/training/moss_compat.py::check_moss_training_contract`（由 `library/training/compat_matrix.py` 在 `model_family == "moss_image"` 时调用）。
- 族实现：`library/models/moss_image/`（`dit.py`、`latent.py`、`strategy.py`、`cache_policy.py`、`block_swap.py`、`training_preview.py`、`preview_validation.py`）。
- 文本缓存：`scripts/moss_image/preprocess_te_cache.py`（经 `scripts/tasks/preprocess.py` 按 family 分派）。
- 静态契约测试：`tests/test_moss_image_token_tiers.py`、`test_moss_image_family.py`、`test_moss_image_web_group.py`、`test_moss_image_web_preflight.py`、`test_moss_image_fp16_amp.py`、`test_moss_int8_contract.py`。
- 热测工具：`bench/moss_tier_hot/`（`python tasks.py moss-tier-hot -- plan|all|report`）。

## 2. 训练前必须核对的契约

以下取值越界会在配置预检阶段直接报错，不会静默降级（来源 `moss_compat.py`）：

| 键 | 允许值 / 要求 |
| --- | --- |
| `model_family` | `moss_image`（别名 `moss`） |
| `mixed_precision` | `bf16` / `fp16` / `no` |
| `base_compute` | 只接受 `bf16` |
| `moss_te_quantization` | `nf4`（默认，BF16 compute）/ `bf16`；失败即报错，不回退 |
| `attn_mode` | `torch` / `sdpa` |
| `blocks_to_swap` | 整数 0..40（族内另有 `len(layers)-2` 上界；当前 42 层即 40）；`>0` 时**所有** batch size 必须为 1 |
| `max_token_length` | 1..4096 的整数，且**进入文本缓存身份**（`moss_image_te_v2` 的 `max_text_tokens`） |
| 缓存 | `use_text_cache` / `use_vae_cache` 必须为 true（磁盘缓存） |
| 损失 | `loss_type=l2`、`timestep_sampling=uniform`、`weighting_scheme=none`、`discrete_flow_shift=1`、`p2_gamma=0` |
| Adapter | 只支持 `networks.lora_anima` plain LoRA；`network_train_unet_only=true`、`router_source=none` |
| caption | `caption_dropout_rate` / `caption_tag_dropout_rate` 在顶层、general、dataset、subset 全为 0；不支持 weighted captions / shuffle |
| 明确拒绝 | compile（`torch_compile`/`compile_dynamic_seq`/`compile_seq_bands`）、DiT NF4、选择性/卸载 checkpoint、自适应精度、AUTO block swap、TE 训练、`use_ip_adapter`/`use_lokr`/`use_loha`/`add_reft` 等 |
| `pipeline_parallel` | 不再被族 shim 一刀切拒绝：MOSS 发布了 stage-local PP v0 backend（实验，见 `docs/experimental/moss_image_pipeline_parallel_v0.md`）；合法性由共享 compat/gate 判定，MOSS 已接线专用 PP 训练会话（其余族仍以 `pp_trainer_path_unavailable` 具名拒绝）；与 `blocks_to_swap>0`、compile 等组合仍被族契约拒绝 |

补充事实（源码核对）：

- `blocks_to_swap` 的真正上界来自 `library/models/moss_image/block_swap.py`：`0 <= n <= len(model.layers) - 2`，当前 DiT `num_layers=42` → 40。
- `block_swap_transfer_dtype=int8` 现在能过门禁，但会打印实验性 WARNING；CMP 50HX 上 loss 逐 step 偏移 0.3%~2.0%，**不建议作为真实训练默认**。
- 训练预览固定 Euler，默认 50 步 / CFG 3 / shift 5；宽高必须是 16 的倍数。

## 3. 性能与显存预算（CMP 50HX，9.56 GiB 可用）

摘自档位热测，仅用于本次预算，不跨硬件外推：

| 档位 | 图像 token | 短字幕 s/步 | 4096-token 字幕 |
| --- | ---: | ---: | --- |
| 1024（默认） | 4032–4200 | 5.666 | 首步 OOM（swap20）；cap2048 可跑 9.365；cap4096 需 swap28（14.120） |
| 768 | 2256–2394 | 3.439 | 第 3 步 OOM |
| 512 | 1008–1064 | 3.036 | 7.210 |

- 面积收益次线性：约 `2.1 s + 0.00084 s/token` 的固定项占主导，降分辨率换不到等比例时间。
- 文本长度是显存主变量；`tecap1024` 是 1024 档的可用性开关（7.234 s/步，峰值 9195 MiB）。
- 256×256 基线另有 11.34 s/步的对照（swap20 的固定开销在小分辨率下占比极高）。

## 4. 本机现状核对（2026-10-19 12:38 CST 实测）

| 项目 | 结果 |
| --- | --- |
| GPU | `GPU 0 = CMP 50HX`，10240 MiB / 空闲 9646 MiB，利用率 0%，47 °C（GPU 1 是 GTX 1050，sm_61 不被当前 PyTorch 支持） |
| 权重 | `models/moss_image` → `ComfyUI/local_models/diffusers/MOSS-Image-SFT`；DiT 9.5G、TE-BF16 16G、TE-NF4 5.5G、VAE 238M，四个组件目录齐全 |
| 源数据 | `image_dataset/style_8_4`：60 图 / 60 txt，短边 900–3456 px（**全部 ≥768**，1024 档不会掉出 token 带） |
| 字幕 | 211–372 字符，属短字幕；文本侧不是显存瓶颈 |
| 已 resize | `post_image_dataset/resized` 60 张，全部落在 4032 / 4200 两个 token family，边长 `% 16 == 0`，与 MOSS stride 16 兼容 |
| 现有缓存 | `post_image_dataset/lora` 只有 `_anima.npz` 与 `_krea2_te.safetensors`，**没有任何 `_moss_image` sidecar** |
| 配置合并 | `train.py --method moss_image_lora_tecap1024 --print-config` 能正确覆盖 `max_token_length=1024`、`blocks_to_swap=20`、`model_family=moss_image` |
| 契约测试 | `pytest tests/test_moss_image_token_tiers.py test_moss_image_family.py test_moss_image_web_group.py test_moss_image_web_preflight.py` → **44 passed, 3 skipped** |

## 5. 前置清单

- [x] 确认权重四组件齐全且路径可解析
- [x] 确认源图短边满足所选档位的数据契约
- [x] 确认现有缓存中没有可复用的 MOSS sidecar（必须重新预处理）
- [x] 确认契约测试与配置合并通过
- [ ] **确定档位**（1024 默认 / tecap1024 / 768 / 512）
- [ ] **确定数据来源**（当前默认走 `base.toml` 的 `image_dataset` 递归；换目录需改 `source_image_dir` 或传 `--dataset_config`）
- [ ] **确定缓存落点**（写进 `post_image_dataset/lora` 只新增 `_moss_image` 后缀；或改 `lora_cache_dir` 完全隔离）
- [ ] **确定训练预算**（epochs / `num_repeats` / 是否开预览采样）
- [ ] 决定是否需要独立 worktree（当前主工作树干净，可直接在 `dev` 上跑）

## 6. 决策后的执行序列（模板）

```bash
# 0) 预检：确认合并后的族、档位、缓存与 swap
.venv/bin/python train.py --method moss_image_lora --preset default \
  --print-config --no-config-snapshot | head -40

# 1) 预处理：resize -> VAE(float32) -> TE(NF4, 带 max_token_length 身份)
#    注意 METHOD 是环境变量，不是 train.py 的 --method
.venv/bin/python tasks.py preprocess METHOD=moss_image_lora

# 2) 训练（默认单卡直跑 train.py；tasks.py lora 会把 --method 固定成 lora）
.venv/bin/python train.py --method moss_image_lora --preset default \
  --max_train_epochs 3

# 3) 收尾：LoRA 与训练状态默认落在 output_dir/output_name
ls -la output/ckpt/moss_image_lora*
```

- 换档位只改方法配置名（`moss_image_lora_t768` 等），**不要单独改 `resolution`**。
- 换 `max_token_length` 必须重建 TE 缓存（缓存身份含该值）。
- 中断续训依赖 `checkpointing_epochs=1` + `checkpointing_last_n_epochs=2` 的状态目录。
- `pipeline_parallel=true`（实验）：只放行 MOSS PP=2/1F1B/FP16/plainLoRA/完整缓存的
  已接线形状，其余配置在入口具名拒绝；启动与恢复口径见
  [moss_image_pp_trainer_wiring.md](../../../docs/experimental/moss_image_pp_trainer_wiring.md)。
  PP 续训用 `resume` 指向 `pp_runs/<run>-step<N>` 包根（旧库级 checkpoint 只能
  显式 weights-only 非精确续训）。

## 7. 仍未验收的边界

- 质量对照从未做过：现有全部结论只覆盖速度、显存和数值有限性。
- 没有真实数据集规模的预处理吞吐数据；本轮 60 图属小规模。
- 长训练（热稳态、收敛、过拟合）无证据；热测只有 10 步窗口。
- MOSS 未注册通用推理 mode/sampler，训练产物只能用训练预览或外部管线验证。
- 本页不改变任何默认配置。

## 8. 受保护产物（清理磁盘前必读）

磁盘清理是最容易误伤实验证据的操作。以下路径**不属于可清理缓存**，任何人（含 Agent）
在清理磁盘时都不得把它们当作候选：

| 路径（相对工作树根） | 是什么 | 删除规则 |
| --- | --- | --- |
| `output/km-venv/` | B 侧（KohakuMossTrainer）实测运行环境：`torch 2.13.0+cu130` + `kohaku-engine`/`kohakuvault`/`anyschedule`/`lycoris-lora` 全套依赖，约 5.4 GB | **必须先向用户提出，并反复多次询问取得明确肯定回答**后才可删除；一次含糊同意不算数 |
| `output/km-src/` | B 侧源码副本（原仓库 `KohakuMossTrainer` 保持只读） | 随 `km-venv` 一并询问 |
| `output/analysis/moss-ab/` | A/B 对比与 512 幅面实测的全部证据：原始日志、逐步计时、配置快照、核验记录 | 不得删除；报告里每个数字都要靠它回查 |

重建成本说明：`km-venv` 之所以要保护，是因为本机默认 pip 源（tuna）已失效，必须显式走
`https://mirrors.aliyun.com/pypi/simple/`，且 `torch` 需从 `https://download.pytorch.org/whl/cu130`
拉取——整套重建是十几分钟到几十分钟的下载加安装，不是一条 `pip install` 就回来的东西。

同一规则在文件系统里另有两份副本：`output/PROTECTED-DIRS.md` 与 `output/km-venv/DO-NOT-DELETE.md`
（`output/` 被 gitignore，所以受版本控制的这份才是 `git clean` 之后仍然存在的记录）。
