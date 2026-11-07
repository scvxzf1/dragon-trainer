# Block Swap 宏观流水线探针

独立于 `bench/cache_identity_hot/` 的性能/调度测量工具：它**只测量，不改变**任何训练行为。

## 为什么需要它

微观层面（单张量的 fused 恢复 + 环形缓冲 + 双流）已接近物理极限，下一步的杠杆在**宏观调度**：
每一步的 retire(D2H) + restore(H2D) 任务如何排队、被计算掩盖了多少、折返点前后有没有可用的空窗。
这些问题的答案只能在真机事件流里看，不能靠静态阅读推断。

## 载体

| 项 | 值 |
| --- | --- |
| 模型 | Anima（真实 `anima-base-v1.0-sha.safetensors` + 真实 Qwen3 TE + 真实 VAE） |
| 数据 | 2 张 512×512 程序生成图 + 短 caption |
| 配置 | plain LoRA rank8/alpha4、bf16、gradient_checkpointing、`attn_mode="torch"`（50HX 是 Turing，flash 不可用）、`torch_compile=false` |
| profiler | `block_swap_profile_jsonl = "auto"` → `<run>/logs/block_swap_pipeline.block_swap_profile.jsonl`，每个完成的 retire+restore 任务写一条事件 |

工作区（`output/bench/block_swap_pipeline/workspace/`）持久复用：图像、权重副本、latent/TE 缓存只建一次，
之后同一分辨率下的 A/B 只花一次 DiT 加载 + N 步训练。

## 命令

```bash
.venv/bin/python -m bench.block_swap_pipeline.run_probe --gpu 0 --steps 4 --blocks-to-swap 12 --label swap12
```

常用开关：

| 开关 | 作用 |
| --- | --- |
| `--blocks-to-swap N` | 固定水位 |
| `--block-swap-auto --margin-mib N` | 打开进程内水位调优（`block_swap_auto`），并在 `result.json` 的 `block_swap_auto` 字段回收日志行 |
| `--restore-mode foreach\|slab` | 固定恢复模式，做 A/B |
| `--side 256\|512` | 分辨率；非默认分辨率使用独立工作区目录（`workspace-<side>`）避免缓存互串 |
| `--family {anima,krea2_raw,z_image,qwen_image_2_1}` | 族预设：模型路径、sidecar 后缀、运行时键；非 anima 族用 `workspace-<family>[-<side>]` |

产物：`<run>/result.json`（原始 metrics）、`<run>/train.log`、`<run>/config.runtime.toml`、profile JSONL。

## 速度测试与 GPU 遥测

`--telemetry` 在训练期间用后台线程采样 `nvidia-smi`（SM 频率、显存频率、显存占用、利用率、
功耗、温度、PCIe 链路 gen/width），落 `<run>/gpu_telemetry.csv` 并把分布写进
`result.json` 的 `speed.telemetry`；`--memory-probe` 另外打开训练器自带显存探针
（`speed.vram`，按加载/训练阶段给出 allocated/reserved 峰值）。

| 开关 | 作用 |
| --- | --- |
| `--telemetry` | 训练阶段采样 nvidia-smi，写 `speed.telemetry` + `gpu_telemetry.csv` |
| `--telemetry-gpu-index N` | 采样的 **nvidia-smi** 索引；默认取 `--gpu`，两者只有在 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 时才一致（否则 CUDA 与 nvidia-smi 的序号顺序不同） |
| `--telemetry-interval-ms N` | 采样间隔，默认 500 |
| `--memory-probe` / `--memory-probe-max-steps N` | 打开训练器显存探针（`speed.vram`） |
| `--metric-window lo-hi` | 步时统计的稳态窗口（例如 `6-50`），默认全部步 |

速度口径一次给全：迭代速度用 `speed.progress`（avg/median/p90 步时、images/h、`peak_reserved_gb`），
搬运带宽用 profile 的 `gb_per_s`（每步 H2D 字节 ÷ DMA 时长），链路上限用 `copy_bench.py`。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python -m bench.block_swap_pipeline.run_probe \
    --gpu 1 --telemetry --telemetry-gpu-index 1 --memory-probe \
    --side 1024 --steps 50 --blocks-to-swap 4 --metric-window 6-50 --label 1mp-swap4
```

实测记录见 [1MP / bs1 / 完整检查点 / swap4 速度基线](../../docs/findings/anima_1mp_speed_50hx_20261022.md)。

## 跨族运行

`--family` 预设了每族的 dit/te/vae、缓存后缀与运行时键，并按族选择 TE 缓存脚本：
anima 用 `scripts/preprocess/cache_text_embeddings.py`，`krea2_raw` / `z_image` /
`qwen_image_2_1` 各有自己的脚本（编码器参数名统一为 `--qwen3`）。
Anima 沿用历史工作区 `workspace`，其它族使用 `workspace-<family>`，缓存互不串用。

各族还会覆盖运行时的族契约键：`z_image` 与 `qwen_image_2_1` 都显式写
`timestep_sampling = "uniform"`（`configs/base.toml` 默认 `sigmoid`，会被
`library/training/compat_matrix.py` 的族校验直接拒绝）。

`qwen_image_2_1` 另有 `tokenizer` 预设：`library/models/qwen_image_2_1/strategy.py`
从 TE 路径向上找 `tokenizer/` 目录，ComfyUI 单文件 TE 旁边没有，探针会在 workspace 的
TE 副本旁就地铺一份。

```bash
.venv/bin/python -m bench.block_swap_pipeline.run_probe --family krea2_raw \
    --gpu 0 --steps 4 --blocks-to-swap 26 --label krea2-swap26
```

50HX（Turing）上各族都只能用 `attn_mode="torch"`：flash 不可用，`sdpa` 在 Krea-2 上报
`No available kernel`；原生高分辨率还会因注意力内存 OOM（Krea-2 在 896×1200 实测如此）。
真实数据集的缓存重建流程与四个实测坑见
[Krea-2 跨族验收](../../docs/findings/block_swap_krea2_hot_50hx_20261019.md)。

## 指标口径

事件字段来自 `library/runtime/offloading.py::_build_profile_wait_event`。分析器（`analyze_profile`）按 step 聚合：

| 指标 | 含义 |
| --- | --- |
| `jobs` / `jobs_forward` / `jobs_backward` | 每步 retire+restore 任务数，分相位 |
| `dma_busy_ms` | 每步 DMA 占用时间下界 = Σ(`d2h_ms` + `h2d_ms`) |
| `exposed_wait_ms` | **未被掩盖**的等待 = Σ(`wait_ms`)，即主线程/GPU 真被卡住的时间 |
| `worker_queue_ms` | Σ(`submit_to_enqueue_ms`)：任务在单线程拷贝池里排队的时间，直接反映 `max_workers=1` 的串行化 |
| `step_span_ms` | 该步从第一个任务入队到最后一个任务等待返回的墙钟跨度 |
| `forward_tail_idle_ms` | 最后一个前向任务 ready → 第一个反向任务入队之间的空窗（跨折返点的可预取窗口） |
| `duplicate_restores` | 同一步内同一个 block 被恢复两次的目标块号。**注意口径**：前向与反向相位各自覆盖全部换出块，跨相位的同号是正常契约而非重复搬运；只有**同相位内**同号才是折返点证据（第 12 轮核对，见 [Qwen-Image 2.1 验收](../../docs/findings/block_swap_qwen_image_21_hot_50hx_20261019.md)） |
| `h2d_bytes` / `gb_per_s` | 每步 H2D 字节数与有效带宽（`Offloader._estimate_swap_bytes()` 从交换计划的 `stored_nbytes()` 推算） |

## 纯拷贝微基准

`copy_bench.py` 不加载任何模型，直接测链路上限与逐张量开销：

```bash
.venv/bin/python -m bench.block_swap_pipeline.copy_bench --gpu 0 --sizes-mb 100,50 --chunks 1,8,24
```

输出每档 `median_ms` / `GB/s` / `us/chunk`，写入 `<root>/<stamp>-copy-bench.json`；它是判断训练内搬运
是链路受限还是开销受限的参照系。

## 跨族分块载荷（不加载权重）

`family_bytes.py` 只读 safetensors 头部，算出每族的每块换出载荷、块数与 `swap` 上界，
用来把 Anima 上的 P1 结论换算到其它族：

```bash
.venv/bin/python -m bench.block_swap_pipeline.family_bytes --all
.venv/bin/python -m bench.block_swap_pipeline.family_bytes --family krea2_raw --free-mib 6550 --margin-mib 400
```

口径与运行时一致（bf16 传输 = `numel × 2`，等于 `Offloader._estimate_swap_bytes()` 的
`stored_nbytes()` 求和）；Anima 档会自校验是否等于 profiler 实测的 132.0 MiB/块。
数据与推论见[各族分块载荷与晋升算术](../../docs/findings/block_swap_family_payload_50hx_20261019.md)。

## 注意

- `exposed_wait_ms` 只统计"等待返回"的耗时；任务在等待点之前就 ready 的部分不计入，因此它是
  **暴露停顿的下界**，`dma_busy_ms` 是 DMA 占用的下界。两者都用于相对比较，不当作绝对物理量。
- 探针默认 `--blocks-to-swap` 由调用方给定；不要用它做显存压力测试（2 图 / 512×512 只用于调度观察）。
- 真机运行时不要与用户训练抢 GPU；本工具不检查 GPU 占用，调用方负责确认。
