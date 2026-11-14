# Krea-2 NF4 性能探针口径审计：84.8% 不等于生产优化空间

状态：实验；口径审计完成，生产稳态重测待执行  
日期：2026-11-12  
适用版本：`6610f710fba4f74aab3b3c6e2fd1cab61c6f39c8`（本轮核对与远端 dev 一致）  
分支：`codex/krea2-nf4-perf-research`  
入口：[最小探针说明](../../bench/krea2_nf4_perf/README.md)

## 结论

1. **保留历史读数，不保留跨场景推断**：11 月 10 日报告记载 256² V100 探针
   kernel union 2361.2 ms / wall 2785.1 ms = **84.8%**。它不是 SM occupancy、
   Tensor Core 利用率或理论 FLOPS 效率，也不是当前生产训练的稳态读数。
2. **“786 次标量提取 = 786 次 GPU→CPU 同步”不成立**：L1 CPU 和 L2 CUDA 小探针
   都复现了大量纯 CPU 标量提取。L2 默认 AdamW 的 784 次事件全部来自 CPU；
   fused AdamW 为 0。没有历史逐事件设备证据，不能宣称历史 786 次已逐项归因。
3. **先修测量契约，不改训练 kernel**：旧探针没有启用 attention 内层 FP16，
   不使用生产默认 fused AdamW，每次采样还重置 optimizer/LoRA 并回传梯度及参数。
   因而不能把其 MAGMA 占比、launch 规模或空泡直接作为双开生产训练的瓶颈。
4. 已开隔离研究分支，只新增研究探针、原始结果、报告及索引。没有修改生产代码、
   配置契约、模型、数据、缓存、历史或队列，没有启动模型训练、推送或跨工作树同步。

## 1. 历史数据与算术边界（L0，引用历史 L3）

来源：[旧 launch 报告](krea2_nf4_fp16_launch_profile_20261110.md#L75-L91)。
本轮在主工作区检查历史 `/tmp/prof`，目录不存在（glob exit 2）；已有其他 nsys
资产不能替代这份时间线。本轮未恢复旧 CUPTI 原始事件，84.8% 仍是报告中的历史记录。

- `wall - union(kernel) ≈ 424 ms` 是未被 kernel 覆盖的窗口，不等于所有 GPU 引擎空闲；
  需另外识别 memcpy、CPU 工作、同步、队列等待与前后边界。
- 即使全部可消除，15.2% 也是该探针口径的延迟份额，不是已测得的生产提速。
- 旧报告两项 MAGMA 为 519.3 + 193.3 = **712.6 ms**，占 2361.2 ms kernel 总时长
  **30.18%**，不是正文所写的 22%；不能据此判断剩余生产 BF16 算子的占比。
- [原始 NF4 消融](krea2_nf4_ablation.jsonl)的 29.454→10.492GB 是峰值显存减少
  **64.38%**，不是旧文档的 72%；3.290→3.405s 是耗时增加 **3.495%**。
- 旧报告的 `50,413 kernel/step` 混计 CPU 算子；其 `4923 ms/step` 时间列也不可用。
  本轮保留旧文件，不覆盖其历史证据。

## 2. 探针与生产执行路径的差异（L0）

| 维度 | 旧 256² 探针 | 生产训练路径 |
| --- | --- | --- |
| Attention 内层 FP16 | `nsys_step.py` / `launch_count.py` 只开 dir1；attention 开关默认 false | bootstrap 分别消费 `nf4_dir1` 与 `krea2_attn_fp16`，V100 模板双开并选择 torch |
| Optimizer | 每次 `run_trajectory(steps=1)` 新建 AdamW，未指定 fused | `get_optimizer` 在 CUDA 的 AdamW 分支默认 `fused=True`，对象跨步持久化；显式用户覆盖另论 |
| LoRA 状态 | 每次采样 `net.load_state_dict(init_state)` | 普通连续步不重载 LoRA |
| 初始化成本 | 每次新 optimizer 的首步 lazy 初始化仍在测量内，外部 warmup 无法消除 | 稳态采样前 optimizer 状态已建立 |
| 梯度与权重回传 | 每步 `_flat_grad(...).cpu()`；返回时逐参数 `.float().cpu()` | 普通训练无这套全量数值对照回传；可选诊断/保存需单独标注 |
| 尺寸与 rank | 256²、rank8、swap0 | 11.06s 记录是 4864 tokens、swap20，不能混成同一基线 |
| 计时边界 | 包含整次轨迹函数的重置、初始化、训练及诊断 | 既有 profiler 钩子围住 `_run_microbatch`，不覆盖整个数据加载及 step 收尾 |

定位：
- [旧 nsys 入口](../../bench/krea2_nf4_fp16_grad/nsys_step.py#L65-L100)
- [旧 launch 入口](../../bench/krea2_nf4_fp16_grad/launch_count.py#L68-L115)
- [轨迹重置、优化器创建](../../bench/krea2_nf4_fp16_grad/trajectory_gate.py#L124-L128)
- [梯度和参数回传](../../bench/krea2_nf4_fp16_grad/trajectory_gate.py#L155-L181)
- [默认 fused AdamW](../../library/training/optimizers.py#L292-L297)
- [生产双开接线](../../library/training/bootstrap.py#L298-L340)
- [V100 方法模板](../../configs/methods/krea2_lora_nf4_fp16.toml#L50-L64)

## 3. 标量提取设备归属实验（L1 / L2）

### 方法

[optimizer_scalar_probe.py](../../bench/krea2_nf4_perf/optimizer_scalar_probe.py) 不加载模型。
每臂 392 个形状 `(2,)` 的 FP32 参数，warmup 更新后，使用 CPU profiler 统计一次
optimizer 更新中的 `_local_scalar_dense`。随后在独立一步用 `TorchDispatchMode`
记录该算子输入设备，只读 device metadata，不读取数值；断言两次总计数一致、
三次更新后的参数有限。不同臂不做性能计时比较，也不充当训练数值保真测试。

Torch 为 `2.10.0+cu129`。该版 Adam 在非 capturable、非 fused 时把 step tensor
放在 CPU；foreach 分别为 beta1/beta2 bias correction 取值，每参数可产生两次
CPU scalar extraction。**CPU 张量的 `.item()` 不构成 GPU→CPU 同步。**

### L2 CUDA 结果

先执行 gpu-gate：`--need 512 --count 1 --exclude 0,1 --json`，exit **0**，选择物理
GPU 2 V100；运行设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2`，
外层 `timeout 60`，探针 exit **0**。不触碰 GPU 0/1 的用户负载。

| CUDA 参数上的 AdamW | 标量事件数 | 事件输入设备 | step state |
| --- | ---: | --- | --- |
| 默认参数 | 784 | 全部 CPU | CPU |
| 显式 foreach=True | 784 | 全部 CPU | CPU |
| 显式 fused=True | 0 | 无事件 | CUDA |

峰值 `torch.cuda.max_memory_allocated` **0.9668 MiB**；探针执行段 **2.143s**
（不含 import，不是训练/optimizer 性能基准）。allocated 不包含 CUDA context
和驱动占用。原始数据：[GPU JSON](krea2_nf4_profile_audit_20261112_gpu.json)。

### L1 CPU 对照与 stderr

CPU 默认 / foreach / single-tensor 分别 **392 / 784 / 392** 次，全部来自 CPU；
断言通过，exit **0**。原始数据：[CPU JSON](krea2_nf4_profile_audit_20261112_cpu.json)。

两次均出现 profiler 的 single-cycle 提示 `Profiler clears events at the end of
 each cycle`；本探针仅一个 cycle，且独立 dispatch 计数一致，不需要跨 cycle 累积。
CPU 对照显式 `CUDA_VISIBLE_DEVICES=''`，stderr 另有
`DeviceProperties.cpp:47 gpuGetDeviceCount failed with code 100`。命令仍 exit 0，
CPU 输出与设备计数完整；该臂只计 L1，绝不将其视为 CUDA 验证。

这组结果推翻的是“标量事件必然是 GPU 同步”的判据。**未证明历史 786 次恰好是
784 次 CPU + 2 次 GPU，也未测得生产速度提升。**

## 4. 下一阶段：复用真实训练的单步采样入口

不再基于旧 `run_trajectory` 编写第二个性能训练循环。

1. 冻结 V100 实验配置：NF4、外层 BF16、dir1 + attention FP16、torch attention、
   full checkpoint、compile=false、持久 optimizer；记录实际 optimizer kwargs、
   token 数、rank、batch、swap、GPU UUID、软件版本、温度和数值诊断开关。
2. 先完成单步最小 GPU 闭环，再扩展到真实 1024² token family。不得把小图结果外推，
   也不得复用/覆盖用户缓存。大模型整机 profiling 本轮未执行。
3. 复用生产 `--profile_steps START-END`。在梯度累积为 1、无异常重放的单卡条件下，
   `3-3` 对应 global_step=3 的一个 microbatch；单写 `3` 会解析成 `3-5`，不是一步。
4. 使用 nsys `--capture-range=cudaProfilerApi --capture-range-end=stop`，输出到
   **新的唯一目录**，不使用覆盖已有报告的选项。先确认 CUPTI kernel 表存在。
5. 此钩子在 batch 取出后开始，optimizer 后、scheduler/日志/采样等收尾前结束，
   **不能把它当完整 wall-clock step**。另做不带 profiler 的连续步墙钟对照。
   stop 会调用 `os._exit(0)`，只用于隔离诊断进程，不能给正在运行的训练注入。
6. 分析 kernel 区间并集、memcpy、CPU launch 入队时刻与 NVTX 阶段，区分 launch
   供给不足、回传、同步和算子本体；再决定缩放/转换融合或残留 BF16 算子优化。

入口：[profile 参数解析](../../library/training/rng_state.py#L11-L27)、
[循环采样位置](../../library/training/loop.py#L897-L914)、
[开始/结束与硬退出](../../library/training/loop.py#L1561-L1602)。

## 未验证 / 不做的推断

- 未重新取得 84.8% 的原始 CUPTI 时间线；未完成历史 424ms 的逐段归因。
- 未重测当前双开配置 1024² 的 kernel 忙时、step latency、长期热态和质量。
- L2 仅验证优化器标量事件归属；**未做 Krea-2 模型级 GPU 数值验证**。
- 未验证任何 kernel/compile/量化优化带来的性能收益；不修改现有 BF16 fail-closed 契约。
- 当前主机没有 3080，本轮 V100 证据不代表 3080。
- 未运行 CPU 全量测试或长训练；没有安装依赖、下载模型、修改功耗/风扇。
