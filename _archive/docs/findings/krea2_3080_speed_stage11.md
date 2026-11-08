状态：阶段 11 已完成（eager vs compiled CUDA operator profile）
日期：2026-08-09
原始摘要：`krea2_3080_speed_stage11.json`
探针：`scripts/krea2/probe_nf4_profile_step.py`

# Krea-2 NF4 单步算子归因

## 方法

PG199 上使用同一 NF4 DiT + rank16 LoRA、full checkpoint、4032-token family
`(1008,1024)` 形状，先预热 eager 并 profile 一步，再编译全部 28 个 resident
block，预热后 profile 一步。仅保存 aggregate self CUDA time，不生成大型 trace。

## 结果

| 操作 | eager | compiled | 变化 |
| --- | ---: | ---: | ---: |
| step 墙钟 | 3.398s | 2.746s | -19.2% |
| GPU peak | 10.494GB | 11.057GB | +0.563GB |
| `aten::mm` | 1593.316ms | 1593.246ms | 持平 |
| cuDNN attention backward | 577.014ms | 576.418ms | 持平 |
| cuDNN attention forward | 270.410ms | 270.436ms | 持平 |
| `aten::mul` | 284.880ms | 融合 | 独立 kernel 消失 |
| `aten::copy_` | 185.169ms | 融合/减少 | 独立 kernel 大幅减少 |
| `aten::add` | 131.401ms | 融合 | 独立 kernel 消失 |
| `bitsandbytes::dequantize_4bit` | 102.237ms / 706 次 | 68.378ms / 486 次 | -33ms / -220 次 |

compiled 图中出现 `triton_poi_fused_*add_mul_silu*` 和
`*silu_backward_view*` 等融合 kernel。矩阵和 attention 内核本身时间完全不变，
所以 19% 收益不是 Inductor 找到了更快 GEMM，而是：

1. 融合 modulation、SwiGLU、residual 及其 backward 的 elementwise kernel。
2. 减少 copy/view/materialization。
3. 在 AOT/checkpoint 图中减少可见 NF4 反量化调用，代价是约 0.56GB
   更高峰值。

## 剩余上限

compiled 后 `mm + attention` 的 operator-level self time 约 2.44s，占 2.746s 墙钟的
约 89%。它们也是阶段 1 中 3080/PG199 差距最大的硬件算子。因此：

- 继续清理 Python、prepare、padding 或小 elementwise 不会再有两位数收益。
- 想明显突破必须减少大矩阵次数/尺寸、减少 checkpoint recompute，或更换吞吐更高的
  GPU；这些都有显存、质量或硬件代价。
- NF4 dequant 在 compiled 图中只占约 68ms，不支持为它重写脆弱的 bnb 内核。

该 profile 与阶段 1/3/4 的墙钟数据一致，给出了后续优化的实际天花板。
