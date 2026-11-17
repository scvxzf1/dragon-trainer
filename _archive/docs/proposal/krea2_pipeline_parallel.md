# 多模型双卡 Pipeline Parallel 阶段 1

状态：通用配置、分层规划和探针已接入；生产训练 runtime 尚未接入。
适用硬件：双 GPU、每个 rank 一张卡；首轮通信目标是 PCIe 3.0 x8，microbatch BS=1。
兼容文件名：本文最初只覆盖 Krea-2，因此保留原文件名。

## 公共架构

- `library/models/family_registry.py::PipelineParallelFamilySpec` 是模型族 PP 能力的单一事实源，
  声明 block 容器、已知 block 数、stages/schedule/split 和 runtime backend 状态。
- `library/models/pipeline_parallel.py` 提供严格的 `PipelineParallelConfig`、连续范围
  `PipelinePlan`、通用互斥校验以及 `BorrowedBlockStage` 的全局 state-dict 键映射。
- family adapter 只负责各自的 block forward 签名：
  `library/anima/pipeline_parallel.py`、`library/models/krea2_raw/pipeline_parallel.py` 和
  `library/models/z_image/pipeline_parallel.py`。
- Krea-2 旧导入名 `Krea2PipelineParallelConfig`、`Krea2PipelinePlan`、
  `validate_krea2_pipeline_config()` 与 `Krea2BlockStage` 保留为兼容 facade。
- `/api/config/model-families` 将 registry 中的能力序列化给 classic 和 Dragon WebUI；
  前端共用同一个 family 别名归一化与 PP 能力判断模块。

## 模型族拓扑

| 模型族 | 主 block 容器 | 已知规模 | 双卡 balanced 起点 | stage 签名边界 |
| --- | --- | --- | --- | --- |
| `anima` | `model.blocks` | 28 / 40 | 14/14 或 20/20 | hidden + timestep/block embedding + cross-attention 与 RoPE |
| `krea2_raw` | `model.blocks` | 28 | 13/15 | combined hidden + tvec + freqs + mask |
| `z_image` | `model.layers` | 30 | 15/15 | unified hidden + mask + freqs + AdaLN/noise modulation |

Krea-2 的 13/15 是考虑 stage 0 还承担 input/text fusion/timestep projection 的初始启发式，
不是已验证的最优分割。Anima 和 Z-Image 暂无实测偏置依据，因此从主 block 均分开始。
Z-Image 只分配 30 个 main layers，不把 noise/context/siglip refiner 混入主分层。

## 配置与校验

三个模型族共用下列顶层字段：

- `pipeline_parallel=false`
- `pipeline_parallel_stages=2`
- `pipeline_parallel_microbatches=4`，可校验范围为 1--1024，首轮探针建议 4--8
- `pipeline_parallel_schedule="1f1b"`
- `pipeline_parallel_split="balanced"`

启用时必须恰好两个 worker，且当前不能与 `blocks_to_swap`、`torch_compile`、
selective checkpoint、CPU/Unsloth activation offload 或非 DiT-only 训练叠加。CLI、WebUI 预检、
直接续训和队列续训共用 `validate_pipeline_parallel_config()` 与统一错误码
`pipeline_parallel_config`。非空非法布尔/整数不会回退默认值。

## 当前边界

`torch.distributed.pipelining` 在当前 Python 环境可导入，但本阶段没有把它接入
`library/training/loop.py`。任一模型族开启 PP 后，合法配置仍会收到
`pipeline_parallel_runtime_unavailable`，CLI 也会明确拒绝启动，不会静默回退
Accelerate DDP。当前也不会保存或恢复 PP stage-local optimizer/RNG 状态。

`BorrowedBlockStage` 及其三个 family subclass 仅是借用原模型 block 的 probe wrapper；它们不转移
parameter ownership。对 wrapper 调用 `.to()` 会同时移动原模型中的同一批模块，不能用来实现
rank-local placement。配置可见、planner 成功也都不等于已能叠加显存。

## 下一阶段约束

1. LoRA `apply_to` 与权重加载必须先于 stage-local model 构造；每个 rank 只持有自己负责的
   base block 与 adapter 参数，activation 通信只传明确 schema 的 Tensor。
2. 首个真实 runtime 固定 `Schedule1F1B`、一 rank 一 GPU、固定 token bucket、microbatch
   BS=1，不开放未实测的 swap/compile/selective 组合。
3. 不得沿用现有全局 adapter gradient all-reduce；pipeline backward 后只更新当前 stage 的
   optimizer 参数，checkpoint 需记录 family、topology version、ranges 并逐 rank 校验恢复。
4. 每个 family 都必须先做单卡等价 forward/backward 与双卡 Gloo/NCCL smoke，再测 PCIe 3.0 x8
   下 microbatch 4/8 的吞吐、通信占比、显存峰值和 loss 曲线。
