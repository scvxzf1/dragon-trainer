# 流水线并行 v0：上手条件与硬件前提

状态：实验
适用版本：`dev` @ `5a4716fd`（2026-11-16）
入口命令：`python tasks.py web`（WebUI 卡片）/ `python train.py --pipeline_parallel ...`
相关代码：`library/training/pp/`、`library/training/train_session.py`、
`library/training/cli_entry.py`、`web/services/training/distributed.py`

本文只回答一件事：**换一台机器、换一个人，怎样才能真的把 PP=2 跑起来，以及跑起来会看到什么数字。**
实现细节见 [Anima PP v0](anima_pipeline_parallel_v0.md)、[Anima trainer 接线](anima_pp_trainer_wiring.md)、
[MOSS-Image PP v0](moss_image_pipeline_parallel_v0.md)。

## 1. 默认不是 PP

仓库默认分布式模式是 **DDP**：

```python
# web/services/training/distributed.py
DEFAULT_DISTRIBUTED_MODE = "ddp"
```

`configs/web-ui-settings.toml` 里没有 `[distributed]` 段，所以新环境必须在 WebUI 卡片里显式切到 PP，
或在 TOML 里写下面第 2 节那几行。不存在「双卡自动变 PP」的路径。

## 2. 最小可用配置

```toml
pipeline_parallel = true
pipeline_parallel_stages = 2
pipeline_parallel_microbatches = 2      # 必须 >= stages
pipeline_parallel_schedule = "1f1b"
pipeline_parallel_split = "balanced"
torch_compile = false                   # 与 PP 互斥
blocks_to_swap = 0                      # 与 PP 互斥
```

WebUI 侧对应「分布式训练」卡片的三态选择（单卡 / DDP / PP），GPU 可选全部卡，
多选只在卡片内部生效，外层选择保持单选。

## 3. 硬性约束（违反即 fail-closed）

| 约束 | 行为 / 错误码 |
| --- | --- |
| `stages` 只支持 **2** | 端点 `wired_families` 报 `stages: [2]` |
| `microbatches >= stages` | `pp_microbatches_below_stages`（torch 2.10 两种 Single schedule 都在 `__init__` 抛） |
| `microbatch_size` 硬编码为 **1** | global batch 恒等于 `microbatches` 数，不是配置项 |
| 与 `torch_compile` 互斥 | 配置校验直接报 `Anima pipeline_parallel currently cannot be combined with: torch_compile` |
| 与 `blocks_to_swap` 互斥 | `pipeline_parallel cannot own a block-swap model` |
| 族白名单 | 仅 `anima`、`moss_image`；其余族报 `pp_trainer_path_unavailable` |
| CUDA PP 需要多后端进程组 | 必须 `cpu:gloo,cuda:nccl`，否则 `pp_backend_multibackend_required` |

## 4. 硬件前提

**双卡即可，不需要 NVLink。** NCCL 自己会按 `NVLink → PCIe P2P → host 中转` 回退，PP 不额外写检测逻辑。

精度不限 fp16：`PpComputePolicy.from_dtype()` 接受 fp32/fp16/bf16 任一种，所以 3090/4090 上 bf16 可用；
V100 没有原生 bf16，必须 fp16（`mixed_precision="fp16"` 会自动要求 fp32 master weights）。

## 5. 实测基线（双 V100-SXM2 16GB / 200 W 限功耗 / fp16）

真实权重 `anima-preview3-base`（28 blocks，`3.94 GiB`），真实 cache，4032-token bucket，等效 batch=8：

| 配置 | 卡数 | 每卡/单卡显存 | 每步 | 吞吐 |
| --- | --- | --- | --- | --- |
| 单卡 bs8 一次前向（无累积） | 1 | **OOM** | — | — |
| 单卡 bs8 + block_swap16 | 1 | 15.05 GB | 14.53 s | 0.551 样本/s |
| 单卡 bs8 真梯度累积（`bs1×8`） | 1 | **5.65 GB** | 15.06 s | 0.531 样本/s |
| 单卡 真累积 + block_swap16 | 1 | **3.83 GB** | 19.18 s | 0.417 样本/s |
| **PP=2，batch=8** | 2 | **4.33 GB/卡**（总 8.66） | **8.53 s** | **0.938 样本/s** |
| PP=2，batch=2 | 2 | 4.10 GB/卡 | — | — |
| PP=2，batch=16 | 2 | 4.62 GB/卡（总 9.24） | 16.29 s | 0.982 样本/s |

要点：

- **单卡 batch=8 直接 OOM，PP=2 每卡 4.33 GB 跑通** —— 这是 PP 存在的理由。
- PP 的显存几乎不随 batch 增长（2→8→16 只从 4.10 涨到 4.62 GB），因为 `microbatch_size=1`。
- 单卡真正的低显存手段是**梯度累积**（`batch_size=1` + `grad_accum=N`），5.65 GB 就能跑等效 bs8；
  block swap 优先级在它之后。这也说明 **PP 相对「单卡 + 梯度累积」的优势是速度（1.77×），不是显存**。

权重释放：`_release_foreign_stage_weights()` 会在 stage module 与 optimizer 建好之后裁掉不属于本 rank 的
block。实测完整模型 `3940 MiB` → 释放后 `2172 MiB`（−46%）；若挂了 adapter，释放率取决于 adapter 是否
只持有 `org_module_ref`（`create_mlp_lora` 这类简化实现还会留另一条引用，只到 3.00 GiB）。

## 6. 通信路径

PP 的 stage 间通信是 `torch.distributed.pipelining.Schedule1F1B` 内部的 p2p send/recv，边界张量是 fp32
（`boundary_dtype: torch.float32`），每 microbatch 每方向约 1 MB。

双 V100 实测（NV4 拓扑，每链路 25.781 GB/s）：

| 路径 | 裸带宽 | PP 每步 |
| --- | --- | --- |
| NVLink P2P（默认） | 86.93 GB/s | 8.526 s |
| 经 host 中转（`NCCL_P2P_DISABLE=1`） | 5.59 GB/s | 8.630 s（+1.22%） |

**结论：通信不是瓶颈。** 每步 16 MB，NVLink 下约 0.17 ms；带宽压掉 15.5 倍也只慢 1.22%，因为 1F1B 的
p2p 是延迟敏感、带宽不敏感的，且流水线把传输掩盖掉了。所以换机器时**不必强求 NVLink**。

`web/services/training/distributed.py` 的诊断会给出一行 `comm_path_nvlink` / `comm_path_pcie`（info 级，
永不拦截），按卡片里实际选中的 GPU 收敛，不是笼统报「本机有 NVLink」。

## 7. 新增族时必读：`_trainer_autocast`

`library/training/pp/stages.py::_AnimaStageBase._trainer_autocast` 是 **Anima 专属**的 autocast 包装。
它套住整个 stage 前向，避免 fp16 注意力在 `autocast(bf16)` 里被反向优化（该缺陷曾把显存从
14.8 GiB 推到 5.05 GiB 的等价收益直接吃掉）。

任何新族要接 PP，**必须复刻同样的包装**，否则会重新引入这个 autocast 缺陷且很难看出来——
数值上可能仍然「能跑」，只是慢或显存异常。

## 8. 验证状态

| 项 | 状态 |
| --- | --- |
| 双 V100 straight + resume（权重摘要相等） | ✅ 实测 |
| 真实权重 + 真实 cache 的 PP=2 训练 | ✅ 实测（上表数据） |
| 未 wired 族 fail-closed | ✅ 测试覆盖 |
| WebUI 诊断 / 通信路径项 | ✅ 端点实测 |
| 其他硬件（3090 / 4090 / A100 / H100） | ❌ 未验证 |
| 其他 CUDA / 驱动 / torch 版本 | ❌ 未验证 |
| Windows | ❌ 未验证（p2p + gloo 组合未测） |

## 9. 排障速查

| 现象 | 处置 |
| --- | --- |
| `pp_microbatches_below_stages` | 把 `microbatches` 提到 ≥ 2 |
| `Anima pipeline_parallel ... torch_compile` | 关掉 `torch_compile` |
| `pipeline_parallel cannot own a block-swap model` | `blocks_to_swap = 0` |
| `pp_trainer_path_unavailable` | 当前族未接线，换 `anima` / `moss_image` |
| `pp_backend_multibackend_required` | 进程组后端要同时含 `gloo` 与 `nccl` |
| 单卡 batch 一大就 OOM | 这是预期：先降 `batch_size` 提 `grad_accum`，PP 才有意义 |
