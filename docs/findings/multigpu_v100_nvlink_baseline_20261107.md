# 2×V100 NVLink 多卡链路与精度基线

状态：实验 / 单次合成探针，不是生产能力或训练质量结论
适用版本：分支 `codex/multi-gpu-training`，基线提交 `a615a9e0`（dev）
归档注记（2026-11-08）：该分支已归档（本地与远端分支均已删除），全部提交已并入 `dev`（`7a6b13c4`）
测量时间：2026-11-07 12:00 前后，机器空闲（GPU0/GPU1 利用率 0%）
相关代码：`scripts/experiments/anima_parallel/transport_bench.py`

## 目的

多卡训练（DDP / PP / TP）的取舍取决于两件事：设备间集合通信成本和各精度实际算力。
本机此前只有异构 PHB 组合（CMP 170HX 64GB + RTX 3080 10GB，无 peer access）的探针数据
（见 [anima_dual_gpu_parallel_probe_20260904.md](anima_dual_gpu_parallel_probe_20260904.md)），
不能外推到本机 NVLink 同构对。本记录补上同构 V100×2 基线。

## 硬件事实

| 项目 | 值 |
| --- | --- |
| GPU | 2 × Tesla V100-SXM2-16GB，compute capability 7.0（SM70） |
| 单卡显存 | 16,928,342,016 B（约 15.8 GiB） |
| 卡间链路 | `nvidia-smi topo -m` 报 **NV4**（NVLink，非 PCIe P2P） |
| 主机链路 | `nvidia-smi` 报 PCIe gen3 **x8**；两卡 NUMA 0 |
| 软件 | torch 2.10.0+cu129、accelerate 1.13.0，未设置任何 `NCCL_*` 环境变量 |

## 通信实测（NCCL，2 rank，repeats=10，warmup=3，CUDA 同步墙钟）

| Payload | all_reduce 中位 | all_reduce 有效带宽 | P2P roundtrip（两次串行传输）中位 | 单向等效带宽 |
| --- | ---: | ---: | ---: | ---: |
| 0.0625 MiB | 0.181 ms | 0.36 GB/s | 0.480 ms | 0.14 GB/s |
| 1 MiB | 0.184 ms | 5.70 GB/s | 0.448 ms | 2.34 GB/s |
| 16.406 MiB | 0.363 ms | 47.4 GB/s | 0.606 ms | 28.4 GB/s |
| 36 MiB | 0.678 ms | 55.7 GB/s | 1.050 ms | 35.9 GB/s |

主机传输（两 rank 并发，pinned）：

| Payload | H2D | D2H |
| --- | ---: | ---: |
| 1 MiB | 0.219 ms / 4.8 GB/s | 0.191 ms / 5.5 GB/s |
| 16.406 MiB | 3.371 ms / 5.1 GB/s | 3.777 ms / 4.6 GB/s |
| 36 MiB | 7.321 ms / 5.2 GB/s | 7.594 ms / 5.0 GB/s |

带宽按十进制 GB/s、`payload_bytes / median_seconds` 计算；all_reduce 为 ring 算法下
的 payload 口径（2 rank 时 busbw 与 payload 口径一致）。P2P roundtrip 是两个方向各一次
串行传输，故单向带宽取表中值的一半。

## 精度实测（4096³ matmul，30 次迭代，cuda:0）

| dtype | 中位耗时 | 等效算力 |
| --- | ---: | ---: |
| float32 | 12.25 ms | 11.22 TFLOP/s |
| float16 | 1.51 ms | 91.24 TFLOP/s |
| bfloat16 | 16.33 ms | 8.41 TFLOP/s |

`torch.cuda.is_bf16_supported()` 在本机返回 `True`，但那是包含仿真路径的判定：
SM70 没有原生 BF16 tensor core，实测 BF16 只有 FP16 的约 9.2%，甚至低于 FP32。
因此**在本机上 bf16 不能作为训练精度档**，与 `library/runtime/launch.py` 里
`ANIMA_ACCELERATE_MIXED_PRECISION` 的默认值 `bf16` 冲突，多卡启动必须显式设为 `fp16`。

## 对多卡方案的直接含义

1. **DDP（adapter 梯度归约）在 NVLink 上不是瓶颈**：LoRA 参数量级在一次 all_reduce
   有效带宽 47–56 GB/s 下，36 MiB 梯度只需 0.68 ms，相对单步训练时间可忽略。
   `library/training/gradient_sync.py` 的 16 MiB bucket 与该带宽匹配。
2. **TP 的主要成本是小消息延迟而非带宽**：单次集合通信有约 0.18 ms 固定延迟。
   历史探针 TP2 每步约 3960 次集合通信，按 0.18 ms 计就是约 0.7 s/步的纯延迟下限，
   与当时 TP2 3.057 s/步、PP2 1.806 s/步的差距方向一致。TP 若要可行，必须减少
   collective 次数（大 fusion/合并通信），不能靠换传输精度。
3. **真正的弱链路是 H2D（约 5 GB/s）**：PP 阶段切分后的 CPU offload、block swap、
   stage 参数装卸都会撞这条链路，PCIe x8 是硬约束，卡间 NVLink 帮不上忙。
4. **单卡 15.8 GiB 是容量上限**：数据并行不降低基座显存，因此 V100 上的多卡研究里
   PP/TP 的价值主要来自“单卡装不下”，而不是速度。

## 复现命令

```bash
cd <repo-root>
.venv/bin/torchrun --standalone --nproc_per_node=2 \
  -m scripts.experiments.anima_parallel.transport_bench \
  --output output/tests/parallel_transport_v100_nvlink_20261107.json \
  --sizes-mib 0.0625 1 16.40625 36 --repeats 10 --warmup 3
```

精度对照用同一解释器执行 4096³ matmul 循环即可，未落盘脚本。

## 边界

- 单次会话、单一消息尺寸序列，没有做 NCCL 参数调优、跨会话稳定性或多 bucket 并发测量。
- 全部为合成 tensor，不含真实 DiT 训练；不能据此给出 DDP/PP 的端到端加速比。
- H2D/D2H 结果是两 rank 并发口径，不是单卡峰值。
- 未测 NVLink 实际链路数量与理论带宽，NV4 是 `nvidia-smi` 报告的拓扑标签，不是带宽实测值。
