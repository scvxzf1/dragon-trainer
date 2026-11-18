# 训练性能 Profiling 落地流程

状态：稳定
更新时间：`2026-06-29`

本文记录 Anima LoRA 训练器的性能观测路径。目标是先用低开销信号判断瓶颈类型，再决定
是优化算子、减少同步、调整 block swap / 显存策略，还是重构 CPU 侧组件。

## 结论先行

训练性能排查按三层漏斗推进：

1. 用 MFU / 步时判断“训练器是否整体算不满”。
2. 用 JSONL 探针判断“显存、block swap、数据搬运是否拖慢”。
3. 用 Nsight Systems 判断“具体时间花在 kernel、memcpy、CUDA API 同步还是 Python 间隙”。

不要在没有 profile 证据前直接押注 Rust 重构。Rust / C++ 只适合 CPU 热点；如果瓶颈在
GPU kernel、H2D/D2H 或 CUDA 同步，应该优先处理训练图、融合、compile、offload 和缓存。

## 快速观测路径

### 1. MFU 与步时基线

先获取当前 GPU 的理论峰值：

```bash
.venv/bin/python -m bench.mfu.gpu_theoretical --gpu-index 0
```

再做短跑基准。默认基准复用 `bench/mfu/` 的缓存训练场景：

```bash
.venv/bin/python -m bench.mfu.run_training --suite baseline --steps 80
.venv/bin/python -m bench.mfu.run_training --suite plain_lora --steps 80
```

维护新 MFU arm 时，不要在对应 `configs/gui-methods/` 变体里写 `max_train_epochs`。
训练 bootstrap 会让 `max_train_epochs` 覆盖 `--max_train_steps`，从而使
`bench.mfu.run_training --steps` 失去截断作用。

重点看 `summary.json` 和终端摘要中的字段：

- `avg_step_sec`：平均每步耗时。
- `median_step_sec` / `p90_step_sec`：步时稳定性。
- `achieved_tflops`：按 Anima DiT 结构估算出的实际吞吐。
- `MFU`：相对理论峰值的模型 FLOPs 利用率。

判读：

- MFU 高、步时稳定：主链计算已经比较饱和，继续优化要看具体 kernel。
- MFU 低、步时不稳定：优先排查 block swap、数据加载、同步和缓存命中。
- 同一数据下某方法比 plain LoRA 慢很多：先把 adapter、rank、checkpoint、block swap 的成本拆开。

MFU 套件说明见 `bench/mfu/README.md`。例如在 `RTX 3080 10GB` 对照中，
`plain LoRA + checkpointing + no block swap`
约 `36.44% MFU`，而 `LoKr + blocks_to_swap=26` 约 `7.16% MFU`。这说明低 MFU
不等于训练主链必然慢，可能是省显存策略的吞吐代价。

### 2. 训练器 JSONL 探针

对真实方法做短跑，打开显存和 block swap 探针：

```bash
.venv/bin/python tasks.py lora PRESET=default --max_train_steps 20 \
  --memory_probe_jsonl auto \
  --memory_probe_max_steps 5 \
  --block_swap_profile_jsonl auto
```

相关 CLI 参数：

- `--progress_jsonl`：结构化训练进度，默认写到 `<output_dir>/../logs/*.progress.jsonl`。
- `--memory_probe_jsonl auto`：写显存、组件、optimizer 诊断。
- `--memory_probe_max_steps 5`：只记录前 5 个训练 step 的细粒度显存快照。
- `--peak_probe_jsonl auto`（配合 `--peak_probe_level block|ops|lokr|full`）：写逐 DiT 块边界、算子或 LoKr 的细粒度 CUDA 峰值显存诊断。
- `--gradient_flow_probe_jsonl auto`：写按需开启的逐块梯度与真实优化器更新遥测。
- `--block_swap_profile_jsonl auto`：写 block swap 传输和等待观测。

`block_swap_profile_jsonl` 是诊断探针，不应作为正式吞吐对比的默认条件。长训或热测要比较
`avg_step_sec` 时，优先关闭它；需要定位 block swap 时再打开。CUDA 路径下 profile observer
默认使用轻量模式：

- `ANIMA_BLOCK_SWAP_PROFILE_POLL_MS`：后台 profile event poll 间隔，默认 `50`。
- `ANIMA_BLOCK_SWAP_PROFILE_GPU_WAIT=1`：开启完整 GPU wait timing，会在训练主 stream 上额外记录
  timing event，只适合短诊断窗口。

轻量模式下，`gpu_wait_ms` 默认为 `0`，`wait_ms` 主要代表 host 侧等待；判断 copy 是否提前完成
优先看 `prefetch_runway_ms`、`estimated_ready_slack_ms`、`h2d_ms` 和 `enqueue_ms`。需要精确拆分
GPU stream 等待时，再打开 `ANIMA_BLOCK_SWAP_PROFILE_GPU_WAIT=1` 做短窗口复测。

WebUI 队列会把 `auto` 解析到当前任务目录：

- `memory_probe.jsonl`
- `block_swap_profile.jsonl`

CLI 直接训练时，`auto` 通常解析到 `output/logs/<output_name>.*.jsonl`。

判读：

| 现象 | 优先方向 |
| --- | --- |
| `block_swap_profile` 中 H2D / D2H / wait 高 | 调整 `blocks_to_swap`、transfer dtype、checkpoint、方法选择或显存预算 |
| 显存峰值接近上限且 alloc/free 抖动 | 看 batch、bucket、compile、cache、checkpoint、预处理缓存 |
| `progress_jsonl` 步时 p90 远高于 median | 看保存、采样、数据加载、日志、同步和偶发 cache miss |
| JSONL 没显示明显传输问题但 MFU 低 | 进入 Nsight Systems 时间线 |

### 3. Nsight Systems 时间线

仓库已经把 `PROFILE_STEPS` 接到 `train.py --profile_steps` 和 nsys wrapper。它只抓稳定窗口，
避免把冷启动、缓存、compile warmup 混进结果。

`scripts/tasks/_common.py` 的 wrapper 会探测本机 nsys CLI 能力并自动降级：

- 新版 nsys：启用 `--python-sampling=true`、`--gpu-metrics-devices=cuda-visible` 和新版 stats report 名。
- 旧版 nsys 2022.x：跳过不支持的 Python sampling，改用 `--gpu-metrics-device=<id>`，
  stats report 改用 `gpukernsum` / `cudaapisum` / `gpumemtimesum` / `nvtxkernsum` 等旧名。
- 若 nsys 只能生成 `.qdstrm` 而没有 `.nsys-rep`，wrapper 会尝试调用
  `QdstrmImporter` 自动导入；可用 `NSYS_QDSTRM_IMPORTER=/path/to/QdstrmImporter`
  指定位置。
- 旧 importer 可能因为 CUDA runtime 太新而返回非零诊断，但只要 `.nsys-rep`
  已生成，wrapper 会继续跑 stats，并把诊断压成简短 warning。
- 若 GPU metrics 被 `ERR_NVGPUCTRPERM` 拒绝，wrapper 会自动跳过 metrics。启用 SM /
  tensor-core / memory-bandwidth counters 需要驱动级 perf counter 权限，通常要 root 设置
  `NVreg_RestrictProfilingToAdminUsers=0` 并重启。
- 混架构多卡机器上，`cuda-visible` 可能受设备顺序影响。若机器同时有不支持 GPU metrics
  的老卡，优先显式指定 metrics 设备，例如
  `NSYS_GPU_METRICS_DEVICES=1 NSYS_GPU_METRICS_SET=ga10x`。旧版 nsys 也可用兼容别名
  `NSYS_GPU_METRICS_DEVICE=1`。
- `NSYS_GPU_METRICS_FREQUENCY` 可覆盖默认 `10000` 采样频率；smoke 或低开销验证可降到
  `1000`。
- 新版 nsys 默认会启用 Python sampling；如果训练在 capture range 附近出现
  multiprocessing / dataloader 连接错误，或者 `.nsys-rep` 只有 NVTX 没有 CUDA trace，
  用 `NSYS_PYTHON_SAMPLING=0` 只抓 CUDA/NVTX 时间线。

启用非 root GPU metrics 的本机步骤：

```bash
sudo install -m 0644 /dev/stdin /etc/modprobe.d/nvidia-perf-counter.conf <<'EOF'
# Allow Nsight Systems/Compute performance counters for non-root profiling.
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF
sudo update-initramfs -u -k "$(uname -r)"
sudo reboot
```

重启后验证：

```bash
rg -n "RmProfilingAdminOnly" /proc/driver/nvidia/params
```

期望值是 `RmProfilingAdminOnly: 0`。如果仍是 `1`，当前 NVIDIA 模块还没用新参数加载。

`scripts/profile_nsys.sh` 也会自动导入 `.qdstrm`，并按本机 report 列表在新版名和
2022.x 旧版名之间自动切换。

推荐跑法：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
PROFILE_STEPS=3-7 \
NSYS_PYTHON_SAMPLING=0 \
NSYS_GPU_METRICS_DEVICES=1 \
NSYS_GPU_METRICS_SET=ga10x \
NSYS_OUT=output/nsys/lora_profile.nsys-rep \
.venv/bin/python tasks.py lora PRESET=default --max_train_steps 10
```

也可以用脚本：

```bash
METHOD=lora PRESET=default PROFILE_START=3 PROFILE_END=7 \
bash scripts/profile_nsys.sh
```

产物：

- `output/nsys/*.nsys-rep`：用 Nsight Systems GUI 打开，看完整时间线。
- `*_cuda_gpu_kern_sum.txt`：GPU kernel 总耗时排名。
- `*_nvtx_kern_sum.txt`：按训练 step / forward / backward / optimizer 聚合。
- `*_cuda_gpu_mem_time_sum.txt`：H2D / D2H / device memcpy 时间。
- `*_cuda_gpu_mem_size_sum.txt`：传输字节量。
- `*_cuda_api_sum.txt`：CUDA API 和同步耗时。
- `*_cuda_kern_exec_sum.txt`：kernel 排队、launch 与执行时间。

判读：

| Nsight 现象 | 解释 | 下一步 |
| --- | --- | --- |
| CUDA HW row 中 kernel 之间大块空白 | GPU 被 CPU、数据加载或同步饿住 | 看 Python sampling、dataloader、缓存、日志和 `.item()` |
| `cudaStreamSynchronize` / `cudaDeviceSynchronize` 很高 | host 侧同步过多或某些步骤阻塞 | 定位同步调用、减少 step 内读回和强制 synchronize |
| 很多极短 kernel 排队密集 | launch overhead / Python 调度占比高 | `torch.compile`、算子融合、批量化跨模块小操作 |
| H2D / D2H 时间或字节量高 | block swap、缓存缺失或 host-device 搬运重 | 降低 offload、压缩 transfer dtype、补缓存 |
| 少数 attention / GEMM kernel 占绝大多数 | GPU 计算主导 | 再用 Nsight Compute 针对单 kernel 看算力/带宽瓶颈 |
| `cudaGraphLaunch` 缺失 | compile / cudagraph 路径可能回退 eager | 查 graph break、shape guard、动态 batch / token family |

## 决策表

| 证据 | 不建议先做 | 建议先做 |
| --- | --- | --- |
| MFU 低且 block swap wait 高 | Rust 重写外围代码 | 降低 swap、换方法、优化 offload 传输 |
| MFU 低且 GPU 时间线有大空白 | 手写 CUDA kernel | 查 CPU 数据管线、同步、缓存、日志和调度 |
| 小 kernel 数量极多 | 增大训练步数掩盖问题 | 批量化小操作、compile、共享 buffer、减少 per-module Python loop |
| 单个 GEMM / attention kernel 主导 | 重构 WebUI / 配置层 | kernel 级分析、attention backend、shape / dtype / layout |
| CPU sampling 指向 JSON/TOML/队列扫描 | 调整 GPU 算子 | 结构化缓存、减少重复解析；必要时考虑 Rust / C++ |

## 推荐推进顺序

1. 固定同一数据、同一方法、同一 seed，先跑 `bench.mfu`。
2. 对慢配置做 `20` step JSONL 探针，不启动采样和长训练。
3. 如果 JSONL 指向 block swap，先做 `blocks_to_swap` / 方法 / checkpoint 消融。
4. 如果 JSONL 不能解释低 MFU，抓 `PROFILE_STEPS=3-7` 的 Nsight Systems。
5. 根据 nsys 证据分类优化：
   - GPU 小 kernel：融合、批量化、compile、共享 buffer。
   - GPU 大 kernel：attention backend、layout、dtype、shape。
   - CPU 空洞：缓存、数据加载、日志、配置解析；确认热点后再考虑 Rust。
   - 传输主导：offload 策略、缓存命中、H2D/D2H 体积。
6. 每次优化后复跑同一窗口，记录 `avg_step_sec`、MFU、关键 nsys 表和改动结论。

### 共享 GPU 口径卫生

做热跑前先看一眼 GPU 是否干净，尤其是 `bench.mfu.run_training` 这种想比较 `avg_step_sec`
的短基准。最少检查：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

如果目标卡上已经有别的 compute 进程长期占用几 GiB 显存或持续 SM 利用率，那么：

- `summary.json` 里的 `avg_step_sec`、`achieved_tflops`、`MFU` 不能直接和历史干净 run 比输赢。
- 这轮数据仍可用于看组件层探针，例如 `block_swap_profile.jsonl` 的 `enqueue_ms`、
  `h2d_ms`、`transfer_ms` 是否下降。
- 真正的“是否提升总步时”结论，必须在空闲卡上按同口径复跑。

一个实用经验是：如果起跑前 `nvidia-smi` 里目标卡已经有非常驻桌面负载，例如独立 Python /
Qt / 推理进程，先把这次 run 标记成“脏口径”，不要把它写成最终性能结论。

## 记录模板

新的性能结论建议归档到 `docs/findings/`：

```markdown
# <方法或配置> Profiling 记录：<简短主题>

时间：`YYYYMMDD-HHMM +0800`

## 结论

- ...

## 环境

- GPU：
- 方法 / preset：
- 数据与缓存：
- 训练步数：
- profile 窗口：

## 产物

- `output/.../*.progress.jsonl`
- `output/.../*.memory_probe.jsonl`
- `output/.../*.block_swap_profile.jsonl`
- `output/nsys/...`

## 证据

| 指标 | 数值 |
| --- | ---: |
| avg_step_sec | |
| median_step_sec | |
| p90_step_sec | |
| achieved_tflops | |
| MFU | |

## 解读

1. ...

## 下一步

- ...
```

## 注意事项

- 不要默认启动长训练、大模型下载或真实采样；profiling 先用短窗口。
- `output/`、`logs/`、`post_image_dataset/` 等产物默认是运行数据，不要作为普通源码清理。
- Nsight Systems 适合看时间线和调度；单 kernel 的算力/带宽细节再用 Nsight Compute。
- 修改 compile、bucket、token shape、attention layout 后，要补对应 invariant 测试。
