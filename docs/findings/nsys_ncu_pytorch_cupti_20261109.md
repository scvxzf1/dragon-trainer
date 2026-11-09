# nsys / ncu 在 PyTorch 上的可用性排查与多卡 PP 时间线实测

状态：实验
适用版本：当前 dev（2026-11-09，驱动 580.196.07 / torch 2.10.0+cu129 / Python 3.13.9）
入口命令：见 §4「可用配方」；下列路径均为**本机本地设施，不是仓库依赖**
相关代码：无（本文是工具链与环境文档，不改仓库代码）

> **本地环境声明**：文中出现的 `/opt/nvidia/cupti-shim/`、`/opt/nvidia/nsight-systems/2025.1.3/`、
> `/usr/local/bin/nsys-v100-torch`、`/usr/local/bin/ncu-v100` 是这台机器上额外搭建的本地设施，
> **不属于仓库内容**，换机器需要重新搭建。仓库内的任何脚本都不应依赖这些路径。

## 1. 结论

1. **系统自带的 `nsys` 在 PyTorch 进程里拿不到 CUPTI 活动表**，原因是 `libtorch_cuda.so`
   硬依赖 `libcupti.so.12`、经 `DT_RUNPATH` 解析到 venv 里的 pip 包，而 CUPTI 是**进程级单例**，
   在 `import torch` 时就被占住，nsys 注入自己那份时无法再注册活动表。
2. 因为 `DT_RUNPATH` 优先级**低于** `LD_LIBRARY_PATH`，用一个 shim 把 `libcupti.so.12`
   指到 nsys 自带的 CUPTI 12.9 即可让两边共用同一份，冲突消失。
3. **`ncu` 不受影响**：它靠 kernel replay 工作，不依赖活动表。同样的最小用例能直接出
   counter 级数据（SM 吞吐、occupancy、寄存器）。
4. 修好之后 `nsys` 在 2 rank PP 上给出了**精确时间线**，把「17% 空泡」定位到
   **步末一整段 1.027 s 的排空**，这是 `torch.profiler` 的算子级汇总给不出的结论。
5. 三种工具互补且不能互相替代：`torch.profiler`（算子级汇总）/ `nsys`（时间线与等待）/
   `ncu`（counter 级）。**`nsys` 与 `torch.profiler` 不能在同一进程同时开**（CUPTI 单例）。

## 2. 现象与隔离

在 2 rank PP 基准上跑：

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none -o out python bench.py
```

报告生成成功，但：

```
SKIPPED: out.sqlite does not contain CUDA kernel data.
```

直接查 sqlite：29 张表、`NVTX_EVENTS` 有 118 条、`TARGET_INFO_GPU` 有 3 行
（说明 nsys **认到了卡**），但 `CUPTI_ACTIVITY_KIND_KERNEL` / `_MEMCPY` / `_RUNTIME`
**一张都没有**。

**隔离方法**：用一段与项目无关的最小 CUDA 用例复现。

```bash
nsys profile --trace=cuda python -c "
import torch
a = torch.randn(1024, 1024, device='cuda'); b = a @ a
torch.cuda.synchronize(); print('ok', b.sum().item())
"
```

最小用例同样无 CUDA 数据 ⇒ **不是应用问题，是本机工具链问题**。

同时排除权限：`/proc/driver/nvidia/params` 中 `RmProfilingAdminOnly: 0`，且当前用户非 root
也能跑 nsys，说明不是「限制到管理员」那条路。

## 3. 根因

```
libtorch_cuda.so ──NEEDED──> libcupti.so.12
                                └─ DT_RUNPATH 解析到 venv 的 pip 包
                                   nvidia-cuda-cupti-cu12/lib/libcupti.so.12
```

- CUPTI 在 **`import torch` 那一刻**就被动态链接器加载，不是等 profiler 才加载。
- CUPTI 是**进程级单例**，nsys 注入自己那份 CUPTI 时单例已被占用 → 活动表注册失败。
- 这与 GPU 架构（Volta/A100）、Nsight 版本都无关，是 torch wheel 的打包方式决定的。

**关键点**：`DT_RUNPATH` 的查找优先级**低于** `LD_LIBRARY_PATH`，所以可以抢先。

## 4. 可用配方

### 4.1 一次性设施（本机，需 root 搭建）

```bash
# shim：让 torch 的 libcupti.so.12 解析到 nsys 自带的 CUPTI
mkdir -p /opt/nvidia/cupti-shim
ln -sf /opt/nvidia/nsight-systems/2025.1.3/target-linux-x64/libcupti.so.12.9 \
       /opt/nvidia/cupti-shim/libcupti.so.12

# 包装脚本：只对 PyTorch 运行生效
cat > /usr/local/bin/nsys-v100-torch <<'SH'
#!/bin/sh
LD_LIBRARY_PATH=/opt/nvidia/cupti-shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_LIBRARY_PATH
exec /opt/nvidia/nsight-systems/2025.1.3/bin/nsys "$@"
SH
chmod +x /usr/local/bin/nsys-v100-torch
```

`nsys-v100`（不带 shim）留给非 PyTorch 程序；**跑 torch 一律用 `nsys-v100-torch`**。

### 4.2 多卡采集：每个 rank 单独起 nsys

**不要用 `nsys profile ... torchrun`**——实测只会抓到 torchrun 父进程的 NVTX，
子进程的 CUDA 活动完全缺失（`--trace-fork-before-exec=true` 也没解决）。

```bash
PY=/path/to/venv/bin/python
ARGS=(--mode pp --output-dir /tmp/prof/run
  --dit-path  "/path/model.safetensors"
  --latent-path "/path/with space/latent.npz"     # 路径含空格必须逐项引号
  --text-path "/path/text.safetensors"
  --microbatches 4 --steps 2 --warmup-steps 1)

for R in 0 1; do
  CUDA_VISIBLE_DEVICES=0,1 \
  RANK=$R LOCAL_RANK=$R WORLD_SIZE=2 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29551 \
  NSYS_STEP=2 PYTHONPATH=/path/to/repo \
  timeout 400 nsys-v100-torch profile --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output=/tmp/prof/rank$R --force-overwrite=true \
    $PY bench.py "${ARGS[@]}" > /tmp/prof/rank$R.log 2>&1 &
done
wait
```

### 4.3 代码侧：用 API 精确卡住一步

```python
if os.environ.get("NSYS_STEP") == str(step):
    torch.cuda.synchronize(); torch.cuda.profiler.start()
...                    # 恰好一个 step
torch.cuda.synchronize(); torch.cuda.profiler.stop()
```

`--capture-range=cudaProfilerApi` 把采集窗口从「整个进程生命周期（含 30 s 模型加载）」
压到「一步 7 s」，报告体积从不可用降到 2.1 MB。

### 4.4 分析

```bash
nsys stats --report cuda_gpu_kern_sum rank0.nsys-rep
```

需要更细的结构时直接查 sqlite（见 §6）。

## 5. 踩过的坑

| # | 现象 | 原因 | 解法 |
| --- | --- | --- | --- |
| 1 | 报告无 CUPTI 表 | torch 的 libcupti 在 import 时占住单例 | shim + `nsys-v100-torch`（§4.1） |
| 2 | 包 torchrun 只抓到父进程 | nsys 未跟进 exec 出的子进程 | 每 rank 单独起 nsys（§4.2） |
| 3 | `Illegal --trace argument 'nccl'` | nsys 2025.1.3 无该值 | 用 `--trace=cuda,nvtx`；NCCL kernel 本就以 CUDA kernel 出现 |
| 4 | `unrecognized arguments: 13_31_23_...` | 变量拼接把含空格路径拆开 | bash 数组 + 逐项引号 |
| 5 | 报告巨大、噪声多 | 抓了整个进程生命周期 | `--capture-range=cudaProfilerApi` |
| 6 | `'int' object has no attribute 'lower'` | `demangledName` 是 `StringIds` 的整数 id | join `StringIds` |

## 6. 分析方法（出结论的关键）

### 6.1 用区间并集算真实 GPU 忙

多流并发时把 kernel 时长直接相加会**超过墙钟时间**（实测 rank0 直接相加得 8.57 s，
而步时只有 7.1 s）。必须算区间并集：

```python
def union_len(iv):
    iv = sorted(iv); tot = 0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce: tot += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    return tot + ce - cs
```

### 6.2 「GPU 忙」必须减去 NCCL

**NCCL 的 send/recv kernel 在等对端时仍占着 GPU**，所以计入「忙」。实测 rank0 显示
「忙 97.4%」，其中 1.506 s 是阻塞等待，真实传输只有约 7.7 ms（275 MB ÷ 35.9 GB/s）。

正确口径：

```
真计算 = union(全部 kernel) − union(NCCL kernel)
真空闲 = 采集窗口 − union(全部 kernel)
```

### 6.3 gap 分析定位空泡位置

算相邻 kernel 之间的空隙并按长度排序，可以一眼区分「空泡是散的」还是「一整段」：

```
rank1: 空闲 245 段，合计 1.113 s
       最长一段 1.027 s @ 6.180 s   ← 一整段步末排空
```

这是从「有 17% 空泡」到「空泡就是步末那 1 秒」的关键一步。

### 6.4 查询模板

```python
import sqlite3
c = sqlite3.connect("rank0.sqlite")
rows = c.execute("""
    select k.start, k.end, s.value
    from CUPTI_ACTIVITY_KIND_KERNEL k
    join StringIds s on s.id = k.demangledName
""").fetchall()
```

## 7. 工具分工（实测印证）

| 工具 | 能给什么 | 局限 | 本次用途 |
| --- | --- | --- | --- |
| `torch.profiler` | 算子级 CUDA 耗时汇总，最轻量，可嵌训练循环 | 无硬件计数器；**看不出时间线位置** | 算出 rank0 5.48 s / rank1 5.57 s，证明**计算平衡** |
| `nsys` | 时间线：NCCL 等待、空泡坐标、CPU 调度 | 与 `torch.profiler` **不能同进程**（CUPTI 单例） | 定位 `nccl[1] @0.763 s 时长 1.049 s` 与 rank1 `1.027 s @6.180 s` 是同一空泡两端 |
| `ncu` | 唯一 counter 级（SM/DRAM/L1/L2/occupancy/寄存器/stall） | 逐 kernel 重放、慢；单卡；需限 launch 数 | 本次未用（瓶颈不在 kernel） |

选择判据：

- 问「**哪一步慢、慢在等谁**」→ `nsys`
- 问「**哪个算子占多少**」→ `torch.profiler`
- 问「**这个 kernel 为什么不够快**」→ `ncu`

`ncu` 最小用例实测（1024³ sgemm，V100 SXM2）：

```
volta_sgemm_32x128_nn  Duration 196.54 us
  Compute (SM) Throughput  83.19%   L1/TEX 80.85%   Memory 77.00%   DRAM 10.21%
  Achieved Occupancy       45.35%   Registers/Thread 57   Block Limit Registers 4
```

即 196 µs ≈ 11 TFLOPS（V100 FP32 峰值 15.7），cuBLAS 已到约 70%；occupancy 受
`Block Limit Registers = 4` 压制，这是 cuBLAS 自身取舍，**不需要也无法通过改调用方解决**。

## 8. 实测案例：2 rank PP 的空泡定位

配置：Anima PP=2 / Schedule1F1B / microbatches=4 / microbatch_size=1 / Global Batch=4 /
单 token 桶（4200 族）/ fp16 + GradScaler / 2×V100-SXM2-16GB / SDPA。

| rank | 采集窗口 | GPU 真忙 | **真实空闲** | 非 NCCL 计算 | NCCL 阻塞 |
| --- | --- | --- | --- | --- | --- |
| rank0（首 stage） | 7.178 s | 6.991 s | **0.187 s（2.6%）** | 5.485 s（76.4%） | **1.506 s（21.0%）** |
| rank1（末 stage） | 7.209 s | 5.982 s | **1.227 s（17.0%）** | 5.567 s（77.2%） | 0.415 s（5.8%） |

口径严格按 §6.2 的定义：`真实空闲 = 采集窗口 − union(全部 kernel)`，
`非 NCCL 计算 = union(全部 kernel) − union(NCCL kernel)`。

- rank1 的空闲里，**一段 1.027 s（从 6.180 s 到步末）就占了 83.7%**，是步末排空。
- rank0 在 0.763 s 处进入一次 **1.049 s** 的 NCCL send/recv 阻塞。
- **两者是同一个空泡的两端**：rank0 等 rank1 的梯度，rank1 干完在等步末。
- 非 NCCL 计算部分 rank0 5.485 s / rank1 5.567 s，**差 1.5%** ⇒ **20/20 均分是合适的，
  unbalanced split 不会带来收益**。

> 注：早先用 gap 统计（过滤 <100 µs 的碎片）得到的是 rank0 1.2% / rank1 15.4%；
> 本文表格改用 §6.2 的并集口径，两者差异来自碎片过滤，不是测量变化。

## 9. 边界与未验证

- 本文所有 GPU 数据来自**单次短跑**（1 warmup + 2 测量步），不代表长跑稳态或热降频行为。
- 空泡比例 17.0% 与 1F1B 理论 `(S−1)/(M+S−1) = 1/5 = 20%` 同量级，但**未逐 microbatch
  对齐验证**，只做了区间级定位。
- `ncu` 只跑了最小用例，**未对训练 kernel 做 counter 级分析**。
- shim 方案依赖 nsys 2025.1.3 自带的 CUPTI 12.9；若将来 torch 换 CUPTI 主版本
  （如 `.so.13`），shim 需要同步调整。
- 未验证在 A100 / sm80+ 上是否同样表现（按根因推断应该一致，但**未实测**）。
