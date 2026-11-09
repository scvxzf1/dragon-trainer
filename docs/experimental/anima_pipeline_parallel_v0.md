# Anima 流水线并行 v0（PP v0）

状态：实验
适用版本：当前 dev（torch 2.10）
入口命令：训练入口不变；PP 配置在训练入口会以具名错误 fail-closed，运行时由
`library/training/pp` 驱动。真机测量与 microbatch 扫描用
`scripts/experiments/anima_parallel/pp_v0_bench.py`（见「真机验证结果」节）
相关代码：`library/training/pp/`、`library/training/pp/gate.py`、
`library/models/family_registry.py`、`library/training/compat_matrix.py`

一句话：Anima 的两阶段 stage-local 流水线运行时已经落地（stage 放置、Schedule1F1B、
冻结 conditioning、fp16 事务对称、per-rank checkpoint），但**主训练循环还没有接入 PP**；
所有训练入口在 `pipeline_parallel=true` 时都会用具名错误拒绝，绝不静默回退 DDP。

## 支持范围（刻意收窄，不是偶然）

| 维度 | PP v0 支持 | 拒绝方式 |
| --- | --- | --- |
| 模型族 | 仅 `anima` | 其他族：`pipeline_parallel_runtime_unavailable` |
| stages | 2 | `pipeline_parallel_config` |
| schedule | `1f1b` | `pipeline_parallel_config`（见下） |
| split | `balanced` | `pipeline_parallel_config` |
| microbatches | **必须 >= stages**（2 阶段即 >= 2） | `pipeline_parallel_microbatches_below_stages` |
| 精度 | 纯 fp16 + `GradScaler` | `pipeline_parallel_mixed_precision_unsupported` / `pp_grad_scaler_missing` |
| 量化基座 | 不支持（NF4 / fp8 / ConvRot 全部拒绝） | `pipeline_parallel_quantization_unsupported` |
| token 桶 | 一个 run 只服务一个 latent shape | `pp_token_bucket_mismatch` |

> 量化判定的唯一口径在 `library/training/pp/contract.py::quantization_markers`：
> `convrot_group_size` 是**调参项**（CLI 默认 256），只有当 `base_compute` 真的选中 ConvRot
> 模式（`w8a16_convrot` / `w8a8_convrot`，含别名 `w8a16` / `w8a8`）时才计入
> （判定复用 `library.runtime.convrot.checks.normalize_base_compute`，不另抄别名表）；
> 量化的 `base_compute`（`nf4` / convrot 模式）本身就是一个 marker。
> ConvRot 的开关语义见 `library/training/bootstrap.py::maybe_apply_convrot_base`（非 convrot
> 值直接返回 `False`）。gate / runtime / compat 都调用同一个函数，不存在第二套判定。

### schedule 只接受 `1f1b`：拒绝发生在 family spec 层

`gpipe` 在 **family spec 层就被拒绝**（anima 的
`PipelineParallelFamilySpec.supported_schedules` 只有 `{"1f1b"}`），compat 报的是
`pipeline_parallel_config`，消息是
`pipeline_parallel_schedule must be one of: 1f1b`。CLI 更早一层也拦：`--pipeline_parallel_schedule`
的 `argparse` choices 只有 `1f1b`，所以从命令行根本传不进 `gpipe`；上面的 `pipeline_parallel_config`
是 TOML 配置文件路径的结果。实测（`model_family=anima`，`world_size=2`，配置文件值）：

```text
schedule='gpipe' -> [('pipeline_parallel_config', 'pipeline_parallel_schedule must be one of: 1f1b')]
schedule='1f1b'  -> []
schedule=''      -> [('pipeline_parallel_config', 'pipeline_parallel_schedule must be one of: 1f1b')]
```

运行时契约 `library/training/pp/contract.py` 里确实还有一条 `gpipe` 分支和
`pp_schedule_unsupported` 码，但**配置路径不可达**：走到 `validate_runtime_config` 之前，
family 校验已经用 `pipeline_parallel_config` 把非 `1f1b` 的 schedule 拒掉了。只有在直接
构造 `PipelineRuntimeConfig(schedule=...)` 的库内调用里才会看到 `pp_schedule_unsupported`。
写文档时不要把这两个码混为一谈。

### microbatches 必须 >= stages

`torch.distributed.pipelining` 的 `PipelineScheduleSingle.__init__`（2.10 的
`schedules.py:578-583`）对 **`Schedule1F1B` 和 `ScheduleGPipe` 都会**断言
`n_microbatches >= num_stages`。所以 2 阶段流水线配 `microbatches=1` 时，任何 stock
schedule 都构造不出来——它不是「改用 GPipe 就行」的情况。CLI 预检
（`library/training/cli_entry.py`）会用 `pp_microbatches_below_stages` 直接拒绝。

补充事实：2 阶段下 `microbatches=2` 的 1F1B 与 GPipe 顺序等价，**>= 3 才真正偏离**
GPipe 的调度。

### DEPRECATED：1.806 s/step 不是 1F1B 基线

**DEPRECATED（作废数字）**：`1.806 s/step` 不得作为任何基线引用。

`docs/findings/anima_dual_gpu_parallel_probe_20260904.md` 里的 1.806 s/step 来自
`scripts/experiments/anima_parallel_probe.py:69`（`_run_pp`）的手写**阻塞**
`dist.send` / `dist.recv`（`:96` / `:98` / `:102` / `:112`）+ `cuda.synchronize` 循环，
**完全没有使用 `torch.distributed.pipelining`**，也不是 `Schedule1F1B`（它是单
microbatch 的 fill-drain 往返）。因此它既不是 `Schedule1F1B` 的基线，也不能用来判断
PP v0 的收益。PP v0 的性能结论必须来自使用 PP v0 运行时本身的测量。该 findings 页与
`docs/findings/multigpu_v100_nvlink_baseline_20261107.md` 均已加同一条 DEPRECATED 标记。

## 双轨验收判据（track ① / track ②）

PP 的正确性分两轨验收，**两轨都要有证据，不能互相替代**：

- **track ① 结构等价（bit-exact）**：只对**相同 microbatch 切分与累加顺序的参考**声称逐位
  相等；`n == 1` 时该参考即全 batch。切分与累加顺序一致时，stage 边界、prediction、loss、
  adapter 梯度都必须与该参考逐位相同（`torch.equal`）。`n == 1` 的用例在
  `tests/test_anima_pp_equivalence.py::test_pp_cpu_gloo_microbatch1_is_bit_exact`（以及
  wire conditioning 变体）；`n >= 2` 的逐位断言只出现在同切分参考上
  （`test_harness_detects_wrong_microbatch_conditioning` 的 per-sample 参考）。
- **track ② 算法等价（对「同切分参考」：梯度余弦 > 0.9999、相对范数误差 ≤ 2%）**：
  `microbatches >= 2` 时 batch 被切开，GEMM 形状和逐 microbatch 贡献的累加顺序都变了，
  逐位相等不再成立，**也不允许**用它当验收条件。②轨的参考是**单卡按相同 microbatch 切分
  与累加顺序跑出的结果**（`*_vs_microbatched` 那组 key），判据为梯度余弦
  （阈值 `0.9999`，见 `tests/pp_equivalence_test_support.py` 的 `COSINE_THRESHOLD`，
  逐 tensor 与整体聚合都要过）与相对范数误差（**真机验收的 Tier 1 判据 ≤ 2%**；CPU harness
  里对应的逐元素上界是 `NUMERIC_ATOL` / `NUMERIC_RTOL`，不是 2%）。

  **参考基准为什么不是全 batch**（2026-11-09 真机实测）：同一张卡、同一权重、同一 batch，
  **只把 bs4 拆成 4×bs1 累加、完全不引入 PP**，梯度余弦就只有 `0.881679`、相对范数误差
  3.04%。也就是说「对全 batch 参考」测的是 fp16 分批累加的固有批不变性差异，不是 PP 的
  正确性；在 fp16 下把它当 PASS/FAIL 判据物理不可达。

  **全 batch 参考因此降级为诊断项**：仍然测量、仍然打印（`*_vs_full_batch` 那组 key），
  但不参与 PASS/FAIL。该数字反映 fp16 batching 的固有差异，与 PP 实现无关；真机上它和
  「无 PP、仅改切分」的对照实验几乎一致，正是这条的实证。

两轨的分工：余弦是尺度不变量，只能证明"方向一致"；它**看不见** torch 2.10
`scale_grads` 那个陷阱（adapter 梯度整体大 n 倍而方向不变）。所以 track ② 的用例里同时保留
`atol` / `rtol` 的逐元素断言（证明幅度正确），两者并存、互不删除：

| 证据 | 位置 |
| --- | --- |
| track ① 逐位相等 | `tests/test_anima_pp_equivalence.py`（`torch.equal`，microbatches=1） |
| track ② 余弦 + 逐元素幅度（**同切分参考**，正式判据） | `tests/test_anima_pp_equivalence.py` 的 `*_vs_microbatched` key（`_assert_gradient_cosine` 与 `_assert_numerically_equal` 并存，microbatches>=2） |
| track ② 负向回归（去掉 `1/n` 后梯度恰好放大 n 倍） | `tests/test_anima_pp_loss_scaling.py` |
| 全 batch 参考（**诊断项**，不判 PASS/FAIL） | `tests/test_anima_pp_equivalence.py` 的 `*_vs_full_batch` key |

## 开关点

唯一开关是 `library/models/family_registry.py` 里 anima 的
`PipelineParallelFamilySpec.runtime_backend`：

```python
runtime_backend=PIPELINE_PARALLEL_RUNTIME_BACKEND_1F1B  # "torch_pipelining_1f1b"
```

- 其他模型族保持 `runtime_backend=None`，继续 fail-closed。
- `compat_matrix` 与 `library/training/pp/gate.py` 读**同一个常量**，所以「家族开关」
  和「兼容层放行条件」不会漂移；改开关时必须同时满足运行时契约（schedule、microbatches、
  world size、量化、`torch.distributed.pipelining` 可导入）。
- `runtime_available` 只表示「有 stage-local 运行时」，**不表示主训练循环已接入 PP**。

## 启动要求（CUDA 上跑 PP v0 的三条硬性前提）

三条都在 `build_anima_pipeline_runtime` 里启动即校验，违反时抛具名
`PipelineParallelError.code`，不会「先跑起来再说」。CPU / Gloo 的等价 harness 只受第 3 条约束。

### 1. 必须用多后端进程组：`cpu:gloo,cuda:nccl`

PP runtime 同时需要两类通信，单后端必然崩：

| 通信 | 内容 | 需要的后端 |
| --- | --- | --- |
| stage 间 P2P | 边界激活（GPU 张量） | NCCL |
| fp16 事务 | scale / skip 决策的标量；`transactions.py:52` / `:133` / `:476` 构造的都是 **CPU** 张量 | Gloo |

- Gloo-only：CUDA P2P 报 `gloo::IoException writev Bad address`。
- NCCL-only：收不了上面那些 CPU 张量。

所以初始化必须写成 `cpu:gloo,cuda:nccl`。`contract.validate_distributed_backend`
（`contract.py:482`）在 `device_type="cuda"` 且 `world_size > 1` 时校验，单后端抛
`pp_backend_multibackend_required`。

### 2. fp16 隐含要求 fp32 master weights

`GradScaler` 拒绝 fp16 梯度，非末 stage 还要按共享 scale 手工反解，fp16 主权重会静默下溢。
因此 fp16 PP 要求**所有可训练参数为 fp32**；冻结基座可以是 fp16/bf16，它没有梯度。
`contract.validate_master_weight_dtype`（`contract.py:443`）在启动时检查，违反抛
`pp_fp16_master_weights_required`。

### 3. `use_fp32` 是 pipeline 级决策，不是每 stage 各推一次

单卡 `models.py:2665` 用 **pre-block hidden** 的 dtype 决定 `use_fp32`，并在
`models.py:1520` 把残差流提升到 fp32 —— 所以 fp16 下**跨边界的张量是 fp32，而这次 run 仍然是
fp16**。后段 stage 若拿「自己收到的边界 dtype」重推，会得到 `False`，后段就在 fp16 里跑本该
fp32 的 block（真机实测 `8.665e4 > 65504` → NaN，首个非有限 block `idx=32`）。

修复后的语义：

- `PpComputePolicy`（`contract.py:357`）在**任何 stage 构造之前**由 `runtime.py:435-436`
  决定一次：由 `dtype`（pre-block compute dtype）推出 `use_fp32`；
- 之后显式传给所有 stage：`build_anima_stage_module(compute_policy=)` →
  `run_stage_blocks(compute_dtype=policy.activation_dtype)` → `stage_block_kwargs`；
  末 stage 的 final layer 走 `finish_model(..., use_fp32=...)`；
- **不允许**任何 stage 从自己收到的边界张量重新推断 `use_fp32`。`numerics.stage_block_kwargs`
  的参数已从 `hidden_dtype` 改名为 `compute_dtype`，就是为了让调用点无法再传边界 dtype。

两条守卫：

| 守卫 | 位置 | 触发条件 | 具名错误 |
| --- | --- | --- | --- |
| 首 stage 自检 | `stages.py:262` `_require_pre_block_dtype` | 实际 pre-block dtype ≠ 策略的 activation dtype（例如 `dtype=None` 建 runtime 却在 fp16 autocast 下跑） | `pp_compute_policy_mismatch` |
| 跨 rank 一致性 | `contract.py:407` `assert_compute_policy_agreement`（`runtime.py:493` 调用） | 两个 rank 解析出的 policy 不一致 | `pp_compute_policy_mismatch` |

同一个 policy 也决定边界 metadata 的 dtype：边界是 `policy.boundary_dtype`（fp16 下 fp32）、
prediction 是 `policy.prediction_dtype`（fp16）。二者不同，`boundary_metas` 必须分开表达，
否则第一个 fp16 步就报 `PipeliningShapeError ... expected torch.float32 actual torch.float16`。

## 强制点清单（不允许静默回退 DDP）

| 位置 | 触发条件 | 具名错误 |
| --- | --- | --- |
| `library/training/compat_matrix.py:368`（`pipeline_parallel_runtime_issue`） | 兼容层预检：家族无运行时 / 契约不成立 | `pipeline_parallel_*`（`key` 恒为 `pipeline_parallel`） |
| `library/training/cli_entry.py:94` | CLI 预检：家族无运行时、microbatches < stages、world size 不匹配 | `pp_runtime_unavailable` / `pp_microbatches_below_stages` / `pp_world_size_mismatch` |
| `library/training/cli_entry.py:100` | CLI 早拒：合法 PP 配置也不进主训练路径（在创建 trainer / 加载模型之前） | `pp_trainer_path_unavailable` |
| `library/training/bootstrap.py:857`（`setup_optimizer_and_dataloader`） | 绕过 CLI 的入口（WebUI 队列 / daemon / 直接 `AnimaTrainer().train` / resume）进入 DDP 组装 | `pp_trainer_path_unavailable` |
| `library/training/bootstrap.py:1075`（`prepare_with_accelerator`） | `accelerator.prepare` 会把整模型包成 DDP | `pp_trainer_path_unavailable` |
| `library/training/loop.py:626`（`run_training_loop`） | accelerate 训练循环启动 | `pp_trainer_path_unavailable` |
| `library/training/loop.py:1175`（`_run_microbatch`） | `trainer.process_batch` 会做整模型单卡 forward | `pp_trainer_path_unavailable` |
| `library/training/schedulers.py:250` | LR 计划按 world size 翻倍 | 不报错：PP 下 `num_processes` 传 `1` |
| `library/training/checkpoints.py:629` / `:704` | `accelerator.save_state` 把两 rank 全状态写进一个目录 | `pp_accelerate_checkpoint_unsupported` |

所有错误都是 `library.training.pp.errors.PipelineParallelError`（`RuntimeError` 子类），
带稳定 `code` 字段，不依赖匹配报错文本。

### 兼容层的 key 不变

`TrainingCompatIssue.key` 恒为 `pipeline_parallel`（`web/services/training/resume_pipeline_gate.py`
与 `tests/test_pipeline_parallel_multimodel.py` 依赖它）。PP 的细分只体现在
`TrainingCompatIssue.code` 上：例如 `pipeline_parallel_microbatches_below_stages`
是新的 `code`，`key` 仍是 `pipeline_parallel`。

### LR 计划：PP 下 `num_processes` 传 1

`get_scheduler_fix(args, optimizer, num_processes)` 会把 `max_train_steps` 乘上
`num_processes`，因为 DDP 下每一步只推进一个 rank 的梯度。PP 下**一步就覆盖整条流水线**，
再乘 world size 会让 2 阶段配置的 LR 计划翻倍。该决策集中在
`library.training.pp.gate.lr_scheduler_num_processes`。

## adapter 梯度缩放（生产级坑）

`torch/distributed/pipelining/schedules.py` 会调用 `stage.scale_grads(n_microbatches)`，
而 `stage.py:570-582` 的 `scale_grads` 只遍历 `self.submod.parameters()`。Anima 的 LoRA
adapter 挂在 network 上、不在基座参数树里，所以 **`scale_grads` 对 adapter 完全无效**；
n=2 时 adapter 梯度恰好是参考的 2 倍。

PP v0 的处理：`library/training/pp/schedule.py` 用 `scale_grads=False` 构造 schedule，
把 `1/n_microbatches` 放进末 stage 的 loss
（`make_last_stage_loss_fn`）。因为因子不依赖参数注册，它对 adapter 与基座一致生效。
训练侧不得再加第二个因子——`pp_trainer_path_unavailable` 就是挡住 accelerate 自身
梯度累积缩放的那道门。

## checkpoint

PP v0 的 checkpoint 是 per-rank 的：每个 rank 只保存自己那一段的 adapter/optimizer 状态，
全 rank 先协商（step / topology / 内容哈希）再落盘，恢复时逐位一致。因此
`accelerator.save_state` 形态的「两 rank 全状态写进一个目录」在 PP 下不可恢复，被具名拒绝
（`pp_accelerate_checkpoint_unsupported`）。入口见 `library/training/pp/checkpoint.py`。

## 相关测试

```bash
timeout 300 .venv/bin/python -m pytest tests/test_anima_pp_wiring.py -q
timeout 300 .venv/bin/python -m pytest tests/test_pipeline_parallel_multimodel.py \
  tests/test_krea2_pipeline_parallel.py tests/test_krea2_pipeline_parallel_webui.py -q
```

`tests/test_anima_pp_wiring.py` 覆盖：anima 放行 / 其他三族具名拒绝、开关与放行条件
一致性、bootstrap 绕过 CLI 的拒绝、loop 与 checkpoint 的拒绝、LR 计划不翻倍、
microbatches < stages 具名拒绝、`scale_grads=False` 与 loss 侧 `1/n`。

## 真机验证结果（2026-11-09，2×V100-SXM2-16GB）

一次有界短跑（T8 复测 / t21），用来验证 D1 修复与 ②轨判据在真机 fp16 下成立。**数字原样记录，
不外推到持续训练**：4 步不足以观察长窗口热降频，也不构成吞吐基线。

**环境**：`CUDA_VISIBLE_DEVICES=0,1`（两张 V100，major 7）；多后端 PG `cpu:gloo,cuda:nccl`；
`attn_mode=torch`（SDPA，本 venv 无 flash_attn/xformers）；fp16 + GradScaler。

**配置**：PP=2 / Schedule1F1B / microbatches=4 / microbatch_size=1 / Global Batch=4 /
单 token 桶（4200 族）/ stage-local LoRA / 40 blocks（20+20）。

### D1 修复确认

- 两 rank 的 `compute_policy` 均 `use_fp32=True`（boundary fp32 / prediction fp16）。
- `first_non_finite_block=None`。
- 4 步全部 `skipped=false`，scale 恒 `65536`。
- loss = `0.279280` / `0.110215` / `0.146224` / `0.083531`。

修复前同一配置会因 fp16 溢出在 block `idx=32` 产生 NaN（见「启动要求」第 3 条）。

### 性能

| 指标 | PP=2 | 单卡同环境同 batch |
| --- | --- | --- |
| 每步 | mean **7.151 s/step** | 10.866 s/step |
| 每样本 | **1.788 s/sample** | 2.717 s/sample |
| 峰值显存（allocated / reserved） | rank0 **5.54 GB**（5.67）、rank1 4.85 GB（4.96） | 14.49 GB |

- 峰值显存 ≤ 6.0 GB → **PASS**（余量 0.46 GB）。
- 相对单卡加速比 **1.521×**（≥ 1.45×）→ **PASS**。
- 绝对时延目标 ≤ 1.60 s/sample → **未达标**（1.788，差 12%），原因是 D1 修复后残差流必须走
  fp32。**吞吐以加速比 ≥ 1.45× 为判定依据（用户裁定），绝对时延为参考项。**

### fp16 数值等价

| 参考 | 余弦 | 相对范数误差 | 结论 |
| --- | --- | --- | --- |
| **同切分参考（②轨正式判据）** | **0.999999521** | **0.0102%** | **Tier 1 PASS**（最小逐 tensor 0.999998） |
| 全 batch 参考（诊断项，不判 PASS/FAIL） | 0.881787 | 3.05% | 诊断数字 |
| 对照：无 PP、仅把 bs4 拆成 4×bs1 | 0.881679 | 3.04% | 证明偏差来自 fp16 分批累加 |

后两行几乎一致 ⇒ 全 batch 参考上的差距来自 fp16 分批累加，而非 PP 实现。

- 分 stage（全 batch 参考下）：stage1（20–39）`0.999888`；stage0（0–19）`0.750894`，
  逐 block 单调退化，且 19/20 边界无跳变（不是某一层突然崩）。
- 前向 loss 相对误差 **0.0014%**。

### 预算

4 次测量运行（57 / 70 / 70 / 72 s）+ 1 次 gate 自证 ≈ 4.8 分钟，未触达 10 次 / 40 分钟上限。

### microbatch 扫描（M=4/8/16/32）

上表是 M=4（Global Batch=4）的结果。把 microbatch 数往上提能压掉 1F1B 的填充/排空，
但收益递减，拐点在 M=16 与 M=32 之间：

| M | Global Batch | s/sample | 加速比 | 峰值 allocated |
| --- | --- | --- | --- | --- |
| 4 | 4 | 1.788 | 1.520× | 5.543 GB |
| 8 | 8 | 1.641 | 1.656× | 5.688 GB |
| **16** | 16 | **1.543** | **1.761×** | 5.981 GB |
| 32 | 32 | 1.519 | 1.789× | 6.566 GB |

**默认建议 M=16**：相对 M=4 每样本快 13.7%、加速比 +16%，显存只多 0.44 GB。
M=32 只再快 1.6% 却多花 0.59 GB，边际性价比掉约 8 倍。

完整数据、边际分析、与理论天花板 `M/(M+1)×2` 的对照、以及复现方式见
[anima_pp_microbatch_scan_20261109.md](../findings/anima_pp_microbatch_scan_20261109.md)。
**注意**：该扫描的加速比是每样本吞吐对比（单卡参考取 GBS=4），不是同 batch 的端到端对比。

## 已知边界（待验收项）

- 主训练循环（accelerate loop + `trainer.process_batch`）尚未接入 PP，因此
  `pipeline_parallel=true` 的正式训练仍会被具名拒绝；PP v0 目前由
  `library.training.pp.build_anima_pipeline_runtime` 驱动（等价 harness / 有界真机短跑）。
- WebUI 预检对 anima + 合法 PP 不再报错，改为警告提示「主训练入口会拒绝」。
- 真机数值与性能只来自一次 4 步有界短跑（见「真机验证结果」），不外推到持续训练、
  长窗口热降频或其它硬件；绝对时延 1.788 s/sample 未达 1.60 目标，判定以加速比 ≥ 1.45× 为准。
- 只做 Anima 族：Krea-2 / Z-Image / Qwen-Image 2.1 / MOSS-Image 的运行时行为未变。
