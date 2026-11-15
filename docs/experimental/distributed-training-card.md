# 分布式训练卡片（WebUI 三态模式）使用说明

状态：实验（与 PP v0 同批）
适用版本：当前 dev（codex/moss-pp-v0-20261112 现场）
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
| 流水线并行（PP，实验） | 模型按阶段切分（stages=2 / 1F1B） | 模型族已接线（anima / moss_image）且至少两张卡 |

PP 选项只对已接线模型族启用；其他族按钮置灰并提示
「模型族 X 尚未接入流水线并行（仅 anima / moss_image 支持）」。
未接线族的 `pipeline_parallel` 启动仍被拦截（历史口径不变），已接线族放行，
最终由训练入口的 admission fail-closed 兜底——前端不阻挡。

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
训练入口 admission（`pp_config_invalid`、`pp_trainer_path_unavailable` 等）。

## 设置落点

PP 参数与模式快照保存在本机设置文件 `[distributed]` 节
（经 `GET/PUT /api/settings/distributed` 读写，WebUI 设置通道，无 schema 变化）：

- `mode`：`single` / `ddp` / `pp`（缺失缺省 `ddp`，旧队列兼容）。
- `gpu_whitelist`：GPU 索引白名单。
- `pipeline_parallel_stages` / `pipeline_parallel_microbatches` /
  `pipeline_parallel_schedule` / `pipeline_parallel_split`：卡片 PP 参数缺省值。

在卡片内切换模式或修改 PP 参数会即时写回该节；其他设置节不受影响。
GPU 选择本身仍保存在浏览器本地（与既有训练设备卡片一致）。

## 边界

- PP 仍为实验能力：仅 anima / moss_image、stages=2、1F1B、plain LoRA、
  完整缓存范围；真实大模型权重的端到端训练未验收。
- 旧队列项（无 `distributed_mode` 字段）续训按 DDP 处理；PP 快照仍由
  runtime config 的 `pipeline_parallel` 键决定。
