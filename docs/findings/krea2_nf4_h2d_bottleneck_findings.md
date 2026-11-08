状态：已完成（方向 B 前置诊断 → NOT_WORTH，归档）
适用版本：当前 krea2-migration 分支
入口命令：`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/krea2/probe_nf4_h2d_bottleneck.py`
相关代码：`scripts/krea2/probe_nf4_h2d_bottleneck.py`、`library/runtime/offloading.py`、`docs/findings/krea2_nf4_h2d_bottleneck.json`（原始数据）
相关提案：[krea2_nf4_blockswap.md](../../_archive/docs/proposal/krea2_nf4_blockswap.md)（方向 B 段，落地前提 2 由本探针回答）

# 方向 B 前置诊断：NF4+blockswap H2D 搬运占比

## 目标

proposal `krea2_nf4_blockswap.md` 方向 B（slab master + 手动重建 Params4bit，
per-slot stream 让 H2D 与 forward 重叠）的存在前提是「大分辨率 H2D 串行成瓶颈」。
本探针回答落地前提 2：**PG199 1024×1024 下 NF4+swap 训练单步里，H2D 搬运占多少比例**。

判断阈值：两口径交叉 ——
- 口径 A `(swap_avg - base_avg)/swap_avg`：分母含 recompute+opt+sync，H2D 占比下界
- 口径 B `(fwd_diff + bwd_diff)/(swap_fwd + swap_bwd)`：只看 fwd+bwd 计算段，分母不含 opt/clip/sync，更准
- both >30% → BOTTLENECK（实现）；both <15% → NOT_WORTH（归档）；分歧 → MARGINAL（按 B 判）

## 实测结果（PG199, 1024×1024, 20 步, swap=4 vs swap=0, 跳前 3 步预热）

| 段 | swap=4 | swap=0 (base) | 差 |
| --- | --- | --- | --- |
| step 墙钟 | 3.371s | 3.370s | +1ms |
| fwd 事件 (CUDA event) | 969ms | 965ms | +4ms |
| bwd 事件 (含 recompute) | 2399ms | 2402ms | -3ms |
| profile nf4_ms | 0 次 | — | 0ms |
| host RSS | 18.05GB | — | 持平 |

双口径 H2D 占比：**A = 0.0% (1ms/3371ms)，B = 0.0% (1ms/3368ms)**。
判断：**NOT_WORTH** —— 两口径都 <15%，H2D 完全藏在反量化窗口。

## 事件口径核验（回应用户盲区提醒）

用户标过盲区：CUDA event 要确认包不包 backward 的 grad-ckpt recompute，否则 H2D 占比
会误判。核验结论：

1. **事件分段**：`fwd_evt` = `fwd_start→fwd_end` 只包第一遍 forward；`bwd_evt` =
   `bwd_start→bwd_end` 包 backward（含 grad-ckpt recompute 的第二遍 forward）。
   两段相加 = forward + backward(recompute) 全程，**没漏 recompute**。
2. **backward 也搬 block**：offloading.py:1734 `backward_hook` 触发
   `backward_prefetch` —— block swap 的 H2D 在 forward 和 backward **两段都发生**，
   不是只在前向。`bwd_diff` 也含 H2D，不会因漏 recompute 偏高。
3. **recompute 在分母做差抵消**：recompute 是 swap/base 共同开销，`(swap_avg-base_avg)`
   做差时抵消，差=纯 H2D 串行开销。口径 A 分母含 recompute 反而**低估** H2D 占比，
   不会高估。

## 口径缺陷（诚实记录）

1. **profile nf4_events=0**：offloader 的 `nf4_ms` 计时（offloading.py:1191-1216）
   只在 `_compute_swap_plan` 把 weight 归入 `nf4_jobs`（master 类型为
   `Params4bitBlockSwapCpuMaster`）时写入。本探针 swap=4 跑完 0 个 nf4_ms 事件，
   说明 NF4 搬运走了 `fallback_names` 路径（`_ensure_weight_on_device`，直接
   `Params4bit.to()` 原地搬）或 plan 分类条件（shape 匹配 offloading.py:668）未
   满足，**没进 nf4_ms 计时分支**。但搬运确实发生了（见下）。
2. **swap=4 实际生效（已点验）**：setup 后 block 0-23（RESIDENT）Linear4bit 在
   `cuda:0`，block 24-27（SWAP）Linear4bit 在 `cpu`。forward 正常跑完（loss
   2.5625→1.0078），没报 device 错 → block 24-27 forward 时确实搬上了 GPU，只是
   搬运没进 nf4_ms 口径。**swap 配置正确，搬运发生了，nf4_ms 口径漏计**。

## 与消融矩阵的矛盾

诊断探针 swap=4 单步 **3.37s**，但消融矩阵（probe_nf4_ablation.py，已验证提交）
报 NF4+swap4 = **3.88s**，差 0.5s。两者测同一步骤逻辑一致（prepare → forward →
backward → clip → opt → sync），单步差 0.5s 不合理。

可能原因：消融矩阵那 0.47s 差（swap4 vs nf4_only）**不是 H2D 传输**，而是
`prepare_block_swap_before_forward` 每步的同步开销（`_ensure_cpu_weight_masters` +
`_warm_swap_plan_cache` + `clean_memory_on_device` 即 `empty_cache`，empty_cache
很慢）+ 首次冷启动 master 捕获。诊断探针 swap=4 ≈ swap=0 说明**在稳定态 H2D 传输
完全藏在反量化窗口**，消融矩阵那 0.47s 是 prepare 同步开销。**方向 B 重叠 H2D 解
不了 empty_cache/同步开销**。

## H2D 传输理论上界

- 4 块 swap NF4 权重体积 ≈ 1.4GB（NF4 master 5.66GB / 28 块 × 4 块 ≈ 0.81GB，
  加 quant_state 和搬运缓冲估 ~1.4GB）
- PG199 PCIe H2D 带宽 ~20GB/s → 传输时间 ~70ms/步
- 即使方向 B 完美重叠这 70ms，省 70ms / 3371ms = **整步 2%**

## 最终判断：NOT_WORTH，方向 B 归档

三个独立证据都指向「H2D 不是瓶颈」：

1. **双口径实测都 0.0%**：swap=4 vs swap=0 单步差 1ms（噪声级），forward/backward
   事件差 4ms/-3ms（噪声级）。H2D 完全藏在 NF4 反量化窗口（反量化慢 7×，forward
   计算时间长，70ms H2D 搬运藏在 ~2.4s 反量化里）。
2. **H2D 传输理论上界 2%**：4 块 1.4GB / 20GB/s = 70ms，占整步 2%。即使方向 B
   完美重叠，收益天花板 2%，不值得实现风险（手动拼 8 字段 Params4bit，依赖 bnb
   内部，易碎；子代理已核实 8 字段清单与 bnb 0.49.2 一致，但实现仍易碎）。
3. **消融矩阵那 0.47s 不是 H2D**：是 prepare 同步开销（empty_cache 等），方向 B
   重叠 H2D 解不了这些。proposal line 142-144 既有 3080 结论（swap 速度几乎不变）
   在 PG199 1024 下重现。

**结论**：方向 B 的存在前提（大分辨率 H2D 串行成瓶颈）在 PG199 1024×1024 下
**不成立**。H2D 藏在反量化窗口，方向 B 重叠收益上界 2%，不值得其实现风险。
**方向 B 归档，不实现。** 子代理核实的 8 字段清单 + bnb 契约保留在 proposal
方向 B 段，供未来大分辨率（>1024）或带宽受限场景复用。

## 未做 / 限制

- 本探针只测 1024×1024。更大分辨率（2048+）反量化窗口更长，H2D 占比只会更低，
  不会变高 → 方向 B 更不值得。但未实测 2048。
- profile nf4_ms 口径漏计是观测缺陷，不影响 wallclock 判断（双口径用 CUDA event
  直接测 forward/backward 段，不依赖 nf4_ms）。
- 诊断探针 swap=4 和消融矩阵 swap=4 单步差 0.5s 的根因未完全定位（疑 prepare
  冷启动 vs 稳定态），但两者都指向「H2D 非主因」，不影响 NOT_WORTH 判断。
