# Anima 1MP 训练速度与 GPU 遥测（CMP 50HX）

状态：实验 / 速度基线（本轮只记录，不做优化）
适用版本：`dev`（HEAD `3b85e43d`），2026-10-22 17:20~17:50 实测
入口命令：见「复现」
相关代码：`bench/block_swap_pipeline/run_probe.py`、`bench/gpu_telemetry.py`、`bench/gpu_precision_probe.py`

## 目的与测试定位

按用户指定的测试定位，测 Anima 在 **1MP 分辨率 / batch=1 / 完整检查点** 下的速度与硬件状态，
并把「核心频率 / 显存频率 / 迭代速度 / 显存占用 / PCIe 带宽」一次性记全：

| 维度 | 本轮取值 |
| --- | --- |
| 分辨率 | 1024×1024（= 1.048 MP，latent 128×128，4096 tokens） |
| batch | 1 |
| 完整检查点 | `gradient_checkpointing = true`（`selective_checkpoint = off`） |
| 块交换 | 主测 **0**（用户本轮要求关掉）；另有一次 **4** 的观测（见结果 2） |
| 步数 | 20 步（swap=0）；50 步计划在 36 步处按用户要求中断（swap=4） |
| 卡 | CMP 50HX（物理 iGPU 之外的独显，nvidia-smi 索引 1） |

训练产物不保留（无 sampler、无 save），本轮只要数字。

## 环境与硬件状态

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA CMP 50HX（`10de:1e09`，TU102），sm_75，**56 SM**，10240 MiB |
| 驱动 / 内核 | `580.178.04`（license `NVIDIA`）/ `6.8.0-136-generic` |
| VBIOS | `90.02.60.00.1A` |
| 显存频率 | 7000 MHz（全程恒定），最大值 7001 MHz |
| SM 频率 | 空闲 300 MHz，负载 1875~1890 MHz，最大 2100 MHz |
| 功耗 | **当前限制 120 W**（默认/最大 225 W），实测负载 39~108 W |
| 温度 | 空闲 42 ℃，负载峰值 70 ℃（slowdown 91 ℃） |
| PCIe 链路 | `gen.current = 1`、`width.current = 8`，`width.max = 16`；`gen.max` 在 17:46 前后由 1 变为 2 |

**算力锁状态（用户侧外部解锁）**：用 `bench/gpu_precision_probe.py` 实测 4096³ 方阵乘法，
对比 56 SM × 1.733 GHz 的理论值：

| 精度 | 实测 TFLOPS | 理论 TFLOPS | 占理论 |
| --- | --- | --- | --- |
| fp32 | 12.38 | 12.42 | 99.7% |
| fp16（tensor core） | 85.03 | 99.38 | 85.6% |
| bf16（sm_75 无原生，fp32 模拟） | 7.06 | — | — |

fp16/fp32 = **6.87×**，tensor 路径正常。若算力锁生效，fp16 会掉到 CUDA-core 档（≈2× fp32），
所以「解锁成功」在这台机上可以只用这一条判定。bf16 只有 7 TFLOPS，是 sm_75 没有原生 bf16 的
模拟开销，与锁无关。

## 测试配置（两轮共有）

`bench/block_swap_pipeline/run_probe.py --family anima --side 1024`，载体与参数：

| 项 | 值 |
| --- | --- |
| DiT / TE / VAE | `anima-base-v1.0-sha.safetensors`（4.18 GB）+ `anima_qwen_3_06b_base`（1.19 GB）+ `qwen_image_vae` |
| 数据 | 2 张 1024×1024 程序生成图 + 短 caption（latent/TE 缓存复用，训练期无磁盘瓶颈） |
| 适配器 | plain LoRA rank8 / alpha4，AdamW，lr 1e-4，constant，bf16 存储 |
| 计算精度 | 配置写 `bf16` → sm_75 由 `precision_policy` 自动切 **fp16** |
| 注意力 / 编译 | `attn_mode = "torch"`（Turing 无 flash）、`torch_compile = false` |
| 块交换 | `restore_mode = slab`、`h2d_only = true`、每块载荷 138,413,056 B（=132.0 MiB）、28 块 |

## 结果 1：swap=0（主测，20 步 × 5 轮，含显存频率梯度）

| 轮次 | run 目录（`output/bench/block_swap_pipeline/`） | 显存频率 |
| --- | --- | --- |
| 17:46 | `20261022-174639-1mp-bs1-swap0-fullckpt` | 7000 MHz（默认） |
| 18:06 | `20261022-180648-1mp-bs1-swap0-fullckpt-r2` | 6801 MHz |
| 18:10 | `20261022-181030-1mp-bs1-swap0-fullckpt-mem7500` | 8000 MHz |
| 18:12 | `20261022-181255-1mp-bs1-swap0-fullckpt-mem8000-r2` | 8000 MHz |
| 18:15 | `20261022-181548-1mp-bs1-swap0-fullckpt-mem7500-r2` | 7500 MHz |

步时口径统一为**自算**（`progress.jsonl` 里相邻 `loss/current` 步事件的间隔中位），避免汇总器把重复步事件算进均值。
后续 `显存频率 → 步时` 的对照都基于这一列。

| 轮次 | 显存频率 | 迭代速度（中位 / 最小~最大） | 耗时%（最慢=100） | 相对速度%（最慢=100） | SM 频率（负载均值） | GPU 利用率（负载） | 功耗（均值 / 峰值） | 显存峰值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17:46 | 7000 MHz | 2.394 s（2.316~2.416） | 98.56% | 101.46% | 1851 MHz | 49.4% | 55.8 W / 108.3 W | 6074 MiB |
| 18:06 | 6801 MHz | 2.429 s（2.362~2.460） | 100.00% | 100.00% | 1850 MHz | 49.3% | 56.1 W / 107.2 W | 6074 MiB |
| 18:15 | **7500 MHz** | **2.334 s**（2.249~2.411） | 96.09% | 104.07% | 1808 MHz | 90.3% | 85.9 W / 110.9 W | 6074 MiB |
| 18:10 | **8000 MHz** | **2.270 s**（2.187~2.300） | 93.45% | 107.00% | 1809 MHz | 90.3% | 86.2 W / 112.9 W | 6074 MiB |
| 18:12 | **8000 MHz** | **2.277 s**（2.186~2.361） | 93.74% | 106.68% | 1766 MHz | 90.2% | 88.3 W / 114.8 W | 6074 MiB |

百分比口径：以最慢一轮（18:06，2.429 s/步）为 100%。`耗时%` = 实际步时 ÷ 2.429，`相对速度%` = 2.429 ÷ 实际步时
（两者互为倒数）。对应吞吐：2.429 s = 1482 图/小时，2.394 s = 1504，2.334 s = 1542，2.277 s = 1581，2.270 s = 1586。

温度峰值各轮 69~71 ℃，PCIe 全程 gen1 ×8，显存峰值五轮逐 MiB 一致（6074 MiB / reserved 5.75 GiB）。

五点观察：

1. **默认显存频率档完全可复现**（2.394 / 2.429，差 1.5%）：1MP/bs1/完整检查点/无块交换是
   **2.39~2.43 s/步、约 56 W、利用率仅 49%**。
2. **8000 MHz 档也可复现**（2.270 / 2.277，差 0.3%）：稳定 **2.27 s/步**，比默认档快 **5~6%**。
3. **显存频率→步时是单调的**：6801 → 7500 降 3.9%，7500 → 8000 再降 2.6%，暂未见平台期；
   想要更准的拐点需要 8500/9000 再补两档。
4. **收益来源是显存带宽**：只要离开默认频率，GPU 利用率就从 49% 跳到 **90%**、功耗 56 → 86 W，
   说明默认档 SM 有近一半时间在等显存，不是算力不足。
5. **代价落在 120 W 功耗墙**：SM 负载均值随显存频率上升反而从 1851 降到 1766~1809 MHz，8000 档功耗峰值
   112.9 / **114.8 W** 已顶到墙（默认/最大 225 W），温度 71 ℃ 距 slowdown 91 ℃ 仍很远。当前是
   「显存带宽换核心频率」的功率受限状态，继续提速要先抬功耗墙。

## 结果 2：swap=4（观测，36/50 步中断）

run：`output/bench/block_swap_pipeline/20261022-172641-1mp-bs1-swap4-fullckpt/`（无 `result.json`，被中断）
2 步冒烟：`output/bench/block_swap_pipeline/20261022-172058-smoke-1mp-swap4-fullckpt/`（有 `result.json`）

| 指标 | 实测 |
| --- | --- |
| 稳态迭代速度 | **21.066 s/步**（中位，36 步），逐步区间 20.92~21.11 s |
| 显存占用 | 训练器 `max reserved 5.24 GiB`；telemetry 峰值 5589 MiB |
| SM 频率 | 1875~1890 MHz（冒烟轮全程恒定 1890） |
| 显存频率 | 7000 MHz 恒定 |
| 功耗 | 峰值 38.6 W |

块交换流水线明细（profile 37 步均值，`slab`）：

| 指标 | 实测 |
| --- | --- |
| 任务数 | 7.89/步（= 2×4，前向 4 + 反向 4；换出块为 24~27） |
| H2D 字节 | 1.092 GB/步（8 × 132 MiB） |
| DMA 忙时间 | 691.8 ms/步 |
| 单块 H2D | 94~102 ms → **有效 ≈1.38 GB/s**（链路 gen1 ×8） |
| 有效带宽（profile 口径） | 1.578 GB/s |
| 未掩盖等待 `exposed_wait` | 0.886 ms/步（几乎不阻塞） |
| 拷贝池排队 | 24.9 ms/步 |
| 预取 | `prefetch_lead = 24` 块、`slot_count = 24`，D2H = 0 |

## 关键观察与未决项

1. **swap=0 与 swap=4 相差 8.9×（2.363 → 21.066 s/步），但不能归因于搬运**：DMA 只占 0.69 s/步，
   且 `exposed_wait` 不足 1 ms，搬运已被计算掩盖。两轮之间还夹着用户的**外部解锁动作**
   （telemetry 里 `pcie.link.gen.max` 由 1 变成 2，fp16 85 TFLOPS 是在解锁确认之后测的），
   因此这两轮**不是干净的 A/B**，差异到底是「块交换」还是「解锁状态」目前无法分离。
   要落地这条结论，需要在同一解锁状态下补一次 swap=4 的 20 步重跑。
2. 历史参照：10-19 同一工作区、同一 1MP、`swap=24 + block_swap_auto`（晋升到 2）实测 2.5 s/步，
   与本轮 swap=0 的 2.36 s/步同量级——即「解锁态 + 少换块」才是 2.x s/步这一档。
3. **PCIe 链路是 gen1 ×8**（上限 ×16），纯 H2D 天花板约 2 GB/s，训练内单块实测 1.38 GB/s。
   历史文档里的 3.14 GB/s 是 Gen1×16 时期的记录，不能与当前 ×8 混用。
4. **功耗墙只有 120 W**（默认/最大 225 W）。当前 SM 频率已跑到 1875~1890（接近 2100 上限的 90%），
   但功耗墙是否压低长跑稳态值得单独确认；本轮 20 步窗口内没有热降频证据（70 ℃ < 91 ℃）。
5. **步时口径坑**：`progress.jsonl` 每步会写两条 `ev=step`（含一条只有 `loss/epoch_average` 的重复行），
   现用的 `_summarize_progress` 会把重复行当成间隔，导致 `avg_step_sec` 被稀释（本轮 1.559 s vs 中位 2.363 s）。
   **以中位/p90 为准**，`avg` 暂不可信。
6. 未测：PCIe 纯拷贝天花板（`copy_bench`，用户本轮叫停）、热稳态长跑、bf16/fp16/checkpoint 组合的对照。

## 复现

```bash
# 1MP / bs1 / 完整检查点 / 无块交换，20 步，带频率-显存-功耗遥测
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python -m bench.block_swap_pipeline.run_probe \
    --gpu 1 --telemetry --telemetry-gpu-index 1 --memory-probe \
    --side 1024 --steps 20 --blocks-to-swap 0 --metric-window 6-20 \
    --label 1mp-bs1-swap0-fullckpt

# 算力锁判定（fp16 tensor 是否满速）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    .venv/bin/python -m bench.gpu_precision_probe --gpu 0 --size 4096
```

产物：`result.json`（`speed.progress` / `speed.telemetry` / `speed.vram`）、`gpu_telemetry.csv`
（500 ms 采样：SM/显存频率、显存占用、利用率、功耗、温度、PCIe gen/width）、`logs/*.progress.jsonl`、
`logs/*.block_swap_profile.jsonl`、`train.log`。
