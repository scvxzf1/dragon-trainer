# 多卡训练实现研究计划（分支 codex/multi-gpu-training）

状态：实验 / 研究分支计划，未改动任何生产默认值
适用版本：分支基线 `a615a9e0`（dev）
入口命令：见各阶段小节
相关代码：`library/training/gradient_sync.py`、`library/runtime/launch.py`、
`library/models/pipeline_parallel.py::validate_pipeline_parallel_config`、
`library/training/compat_matrix.py`、`scripts/experiments/anima_parallel/`

## 目的

把“多卡训练”从一个隐含能力变成有边界、有证据、可验收的能力集合：

1. 明确当前哪条路径已可用（数据并行 adapter 训练）、哪条只是配置可见（PP）、哪条只是探针（TP）。
2. 在真实硬件（本机 2×V100-SXM2-16GB / NVLink）上逐条补齐或明确否决。
3. 每一条结论都落到代码位置、验收命令和 findings 记录，不用“配置能开”当能力证明。

## 现状盘点（代码事实）

| 能力 | 状态 | 代码位置 | 关键事实 |
| --- | --- | --- | --- |
| 数据并行（DDP） | 已接入生产训练路径 | `library/training/gradient_sync.py`、`bootstrap.py::prepare_network_for_manual_gradient_sync`、`loop.py::synchronize_optimizer_gradients` | adapter 容器**故意不包 DDP**：其 forward 由被打补丁的冻结 DiT/TE 模块驱动，容器没有匹配的 root forward。改为 rank0 广播参数/buffer、固定顺序 bucket 均值归约、`grad=None` 用零填充但不把全 rank 未用参数变实、CUDA bucket 从 backward hook 在通信流上重叠；`after_backward` 加梯度的 adapter 走同步回退 |
| 多卡启动 | 已接入 | `library/runtime/launch.py`：`ANIMA_ACCELERATE_LAUNCH` / `ANIMA_ACCELERATE_NUM_PROCESSES` / `ANIMA_ACCELERATE_MIXED_PRECISION`、`accelerate_training_command_prefix()` | WebUI 多选 GPU 时 `configure_accelerate_for_gpu_selection()` 自动置 launch=1 且 num_processes=已选卡数；CLI/daemon/队列共用同一命令构造 |
| 流水并行（PP） | 配置与规划已接入，**runtime 未接线** | `library/models/pipeline_parallel.py`、各族 `pipeline_parallel.py`、`family_registry.py::PipelineParallelFamilySpec` | 合法配置仍返回 `pipeline_parallel_runtime_unavailable`；`BorrowedBlockStage` 只是借用原 block 的探针 wrapper，不转移参数归属 |
| 张量并行（TP） | 仅实验探针 | `scripts/experiments/anima_parallel/{tensor_parallel,collectives,stages,benchmark,checkpoint,matrix,report}.py` | 同步 TP2 + 精确 LoRA 梯度集合通信；INT8 组-128 传输把 wire bytes 降 49.2% 但端到端只快约 0.23%，已否决为默认 |

历史证据：

- [公开训练器 DDP/TP/PP 调研](multigpu_trainers_ddp_tp_pp_survey_20261004.md)：sd-scripts/Musubi 与本项目同路线（Accelerate + adapter 显式梯度平均）；diffusion-pipe 是真正接入训练循环的 DP+PP 参考；TP 参考 TorchTitan/Megatron。
- [Anima 异构双卡 PP2/TP2 探针](../findings/anima_dual_gpu_parallel_probe_20260904.md)：170HX+3080 无 peer access 时 PP2 1.806 s/步、TP2 3.057 s/步；该结论不能外推到本机 NVLink 同构对。
- [本机 2×V100 NVLink 基线](../findings/multigpu_v100_nvlink_baseline_20261107.md)：all_reduce 36 MiB 0.678 ms（55.7 GB/s）、单次集合通信延迟下限约 0.18 ms、H2D 约 5 GB/s、BF16 只有 FP16 的约 9.2% 算力。

### 本机硬件约束（直接决定研究取舍）

- 2×V100-SXM2-16GB，SM70，NVLink（NV4），主机侧 PCIe gen3 x8。
- **bf16 是仿真路径**（8.41 对 fp16 91.24 TFLOP/s），而 `ANIMA_ACCELERATE_MIXED_PRECISION`
  默认值是 `bf16`：多卡启动必须显式 `fp16`，否则精度档默认值在本机就是错的。
- 单卡 15.8 GiB：数据并行不降低基座显存，因此 V100 上 PP/TP 的价值首先是“装得下”。

## 阶段计划

### S0 基线与复现（已完成首轮）

- 产出：[multigpu_v100_nvlink_baseline_20261107.md](../findings/multigpu_v100_nvlink_baseline_20261107.md)。
- 复现命令见该文；探针复用 `scripts/experiments/anima_parallel/transport_bench.py`，不新增重复实现。
- 待补：多 bucket 并发 all_reduce、Gloo 对照、长窗口稳定性。

### S1 DDP 在 V100×2 上的等价与验收

- 目标：证明“2 卡 DDP = 单卡 global batch”的等价性与可恢复性，并把 SM70 精度档固化。
- 必须观测：同 seed 同数据下 2 rank 归约后 adapter 参数逐位一致；loss 轨迹与单卡等效
  global batch 对齐；`GradScaler` 跳过决策全 rank 一致；checkpoint 保存/续训后继续一致。
- 复用判据：`tests/test_distributed_gradient_sync.py`、
  `scripts/experiments/convrot_distributed_probe.py` 的四轴检查（数学偏移/速度/峰值显存/备注）。
- 验收命令（草案）：`.venv/bin/torchrun --standalone --nproc_per_node=2 <probe>`；真实短训需要
  用户同意后执行。
- 明确不做：把 `num_processes` 语义扩展到 TP/PP 组，S1 只覆盖纯数据并行。

### S2 DDP 工程缺口收口

- 候选缺口（需逐个用源码核实，不预设结论）：
  - 每 rank 数据分片与 bucket/aspect 对齐：不能把 world ranks 当作独立数据 rank 之后又共享同一
    sampler 语义。
  - 参数初始化广播只在 rank0 且与 adapter 加载顺序的关系。
  - preview/采样/日志/进度上报是否只在主进程；sample prompt 与 RNG 是否需要 rank 一致。
  - 续训时 `world_size` 变化的语义（当前未文档化）。
  - WebUI 预检、`docs/guidelines/training.md` 与实际行为的一致性。
- 验收：每条缺口要么有测试/探针证据，要么记录为“未验证”并写进 findings。

### S3 PP 生产 runtime 接线

- 前置：[PP 提案](krea2_pipeline_parallel.md) 的“下一阶段约束”四条（apply_to/加载先于 stage 构造、
  固定 `Schedule1F1B`、不复用全局 adapter all-reduce、checkpoint 记录 family/topology/ranges）。
- 顺序：单卡等价 forward/backward → 2 rank Gloo smoke → NCCL microbatch 4/8 → 吞吐/通信占比/显存。
- 硬约束：一条 P2P 单向 35.9 GB/s、延迟 ~0.24 ms/次，microbatch 数与 payload 必须按此预算。
- 出口条件：真实训练循环接入并有 stage-local optimizer/ckpt 证据；否则只保留 fail-closed 拒绝。

### S4 TP 可行性研究

- 先做“collective 次数压降”评估再谈实现：每步 3960 次集合通信在 0.18 ms 延迟下就是约
  0.7 s/步下限，历史 TP2 的劣势主要来自此。
- 判据：同步 TP + plain LoRA 的 forward/backward 数值等价（rel-L2 门槛）、通信占比、显存；
  传输量化（INT8）不作为默认方向（已被历史证据否决）。

### S5 组合与取舍结论

- 输出一张表：本机（2×V100 NVLink）上 DDP / PP / TP 各自适用条件、收益上界、放弃理由。
- 只有通过 S1–S3 验收的路径才允许出现在默认配置或 WebUI 提示中。

## 不变量与风险

- 每个 adapter 参数每个 `sync_gradients` 边界只归约一次；`grad=None` 分歧不得让某个 rank 跳过
  collective 导致 hang。
- 梯度在 loss-scaled 状态下归约，保证 FP16 溢出传播到所有 rank，`GradScaler` 跳过决策一致。
- 非有限 loss 的处理、router/FEI 统计、梯度裁剪必须发生在同步之后（现有顺序不可破坏）。
- 主进程专属副作用（保存、预览、日志、tracker）不得被多进程重复触发。
- 不改变单卡默认行为与性能；多卡开关仍 opt-in。
- 本机 bf16 默认档问题：要么在启动层按设备能力拒绝/改写，要么明确要求用户显式设置。

## 非目标

- FSDP/ZeRO 全量实现、跨机多节点、异构混卡训练。
- 把 PP/TP 变成生产默认；未验收的路径不得放宽 fail-closed。
- MOSS 等其他模型族的多卡支持（先做 Anima / Krea-2 / Z-Image / Qwen-Image 2.1 已注册族）。
- 大模型下载、长训练、功耗或散热调整。

## 待确认（用户输入）

1. 优先级：先做 S1/S2（DDP 收口）还是 S3（PP runtime）？
2. 目标模型族与用途：只要 Anima，还是要求 Krea-2 / Z-Image / Qwen-Image 也覆盖？
3. 是否授权在 V100×2 上跑真实短训（如 20–50 step）作为验收证据？
