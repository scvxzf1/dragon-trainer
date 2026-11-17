# 分布式训练卡片（WebUI 三态模式）使用说明

状态：实验（与 PP v0 同批）
适用版本：含 Krea-2 PP 接线的版本
相关代码：`web/services/training/distributed.py`、`web/services/training/gpu.py`、
`web/routes/training.py`、`web/frontend-next/src/features/training-config/TrainingDistributedCard.tsx`

## 入口与形态

Next 工作台「设备与性能（resources）」分组、「多卡与分布式训练」区块之下是
「分布式训练」三态卡片：

- 折叠态只有一行：模式按钮组（单卡 / 多卡·数据并行 / 流水线并行（实验））、
  当前模式徽标、展开箭头。不挤占主表单。
- 点展开箭头打开 Popover（portal 浮层），内含 GPU 选择、每卡属性徽标、
  诊断列表与 PP 参数。

三态语义：

| 模式 | 含义 | 前置条件 |
| --- | --- | --- |
| 单卡 | 单 GPU 训练 | 任选一张卡（radio，选一） |
| 多卡 · 数据并行（DDP） | 每卡一份完整模型副本 | 至少两张卡（checkbox 可多选） |
| 流水线并行（PP，实验） | 模型按阶段切分（stages=2 / 1F1B） | 模型族已接线（anima / moss_image / krea2_raw）且选择两张卡 |

PP 选项只对已接线模型族启用；其他族按钮置灰并提示
「模型族 X 尚未接入流水线并行（仅 anima / moss_image / krea2_raw 支持）」。
未接线族的 `pipeline_parallel` 启动仍被拦截（历史口径不变），已接线族放行，
常规预检与训练入口使用相同的 admission；已知必败设置在预检中作为 error 阻止启动。

## GPU 选择与守卫口径

- GPU **全卡开放**：任何 nvidia-smi 可见的卡都可选（含 GTX 1050 等
  非授权卡）。
- 唯一 fail-closed 守卫：显式选卡索引在 nvidia-smi 中**解析不到物理卡**
  （越界 / 查询失败）时，enqueue/start 前具名拒绝
  `gpu_whitelist_unauthorized_gpu`。
- 不适配卡的提醒降级为诊断建议（见下节），不再阻断。

## 诊断（🔴 必败 / 🟡 风险 / 🔵 信息）

Popover 展开时调用只读诊断端点
`POST /api/training/distributed/diagnosis`，返回：

- `hardware.gpus`：每卡快照（index / uuid / name / memory_total_mb /
  capability），Popover 内每卡徽标显示总显存与 `sm_XX` 代际。
- `hardware.risks`（仅 PP 模式，三层硬件风险，全部**建议级**）：
  🟡 低显存（<8GB）、🟡 显存异构、🟡 代际混搭；无法解析代际时 🔵 提示未判定。
- `checks`（仅 PP 模式，PP 数据集就绪，复用现有 census / cache audit 实现）：
  🔴 多子集桶不一致、🔴 多子集 batch_size 不一致、🔴 latent/TE 缓存缺失；
  🟡 子集目录缺失或桶统计不完整；🔵 单桶/单子集/batch 一致/缓存完整。
- `wired_families`：PP 已接线族表，卡片据此置灰未接线族的 PP 选项。

折叠行徽标带 🔴n / 🟡n 计数。诊断永不阻挡启动；真正的拦截仍在
常规预检和训练入口 admission（`pp_config_invalid`、`pp_trainer_path_unavailable` 等）。
卡片诊断读取当前参数草稿；启动和入队前会先保存草稿，再预检已保存的配置。

## 设置落点

PP 参数与模式快照保存在本机设置文件 `[distributed]` 节
（经 `GET/PUT /api/settings/distributed` 读写，WebUI 设置通道，无 schema 变化）：

- `mode`：`single` / `ddp` / `pp`（缺失缺省 `ddp`，旧队列兼容）。
- `gpu_whitelist`：GPU 索引白名单。
- `pipeline_parallel_stages` / `pipeline_parallel_microbatches` /
  `pipeline_parallel_schedule` / `pipeline_parallel_split` /
  `pipeline_parallel_blocks_to_swap`：卡片 PP 参数缺省值。

修改单个 PP 参数会合并写回该节，保留其他本机分布式设置，并更新当前训练草稿。
当前配置或草稿已有值时优先使用它们；仅缺值时使用本机默认。保存训练配置后才写入训练 TOML。
GPU 选择本身仍保存在浏览器本地（与既有训练设备卡片一致）。

## Krea-2 PP 配置

使用 `mixed_precision = "fp16"`，关闭 compile 和普通 `blocks_to_swap`，只训练 DiT LoRA。
微批数必须至少为 2 并整除 `train_batch_size`：batch=2 时设为 2，batch=4 时可设为 2 或 4。
不支持训练 preview、额外梯度累积和 epoch 保存的组合会在预检中明确报错。

`pipeline_parallel_blocks_to_swap` 是每阶段交换块数，不能复用普通 `blocks_to_swap`。
双 V100 16GB、896×1200 的已记录基线为 4；实际适用范围以
[显存实验记录](../findings/krea2_pp_vram_gap_20261116.md)为准。
配置正值优先；0 会回退 `KREA2_PP_SWAP_BLOCKS`，环境变量未设置或为 0 时才关闭交换。
`lora_fp32_compute` 可在 TOML 或 CLI 的 `--network_args lora_fp32_compute=false` 中显式设置。

## 边界

- PP 仍为实验能力：Anima、MOSS-Image 和 Krea-2 有专用训练会话，支持范围受各族准入约束；
  已记录的真机实验不代表全部硬件或配置组合均经过验收。
- 旧队列项（无 `distributed_mode` 字段）续训按 DDP 处理；PP 快照仍由
  runtime config 的 `pipeline_parallel` 键决定。
