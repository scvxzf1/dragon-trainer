# Krea-2 3080 / PG199 速度优化 12 阶段汇总

状态：研究记录 / 阶段汇总
适用版本：2026-08-09 各阶段实测；当前默认值以源码和 `configs/` 为准
入口命令：无（纯记录；探针见 `scripts/krea2/probe_nf4_ablation.py`）
相关代码：`library/models/krea2_raw/`、`scripts/krea2/probe_nf4_ablation.py`
配套文档：[最终结论](krea2_3080_speed_final.md)、[跨硬件矩阵](krea2_3080_speed_comparison_extended.md)

## 定位

本页把 12 个阶段的判定与关键数字合成一页便于查阅。**每阶段完整正文**（矩阵、探针口径、
注意事项、原始 json 摘要文件名）已归档到 `_archive/docs/findings/`，表格最后一列是直达链接。

本页不新增结论、不改写任何阶段判定；数字与判定均转述自各阶段原文。

## 阶段结论一览

| 阶段 | 主题 | 判定 | 关键数字 | 完整记录 |
| --- | --- | --- | --- | --- |
| 1 | 瓶颈定位 + 两个候选优化消融 | 两个候选均 `NOT_WORTH` | 完整步 3080/PG199 `3.60x`；BF16 Linear 慢 `4.4-5.2x`、NF4 Linear `4.1-4.6x`、attention `2.83x`；NF4 相对 BF16 额外开销仅 `13-25%` | [stage1](../../_archive/docs/findings/krea2_3080_speed_stage1.md) |
| 2 | 选择性梯度检查点 | PG199 `WORTH_IT` / 3080 `NOT_FEASIBLE` | PG199 预期 1024² NF4 `28.46GB / 2.90s`；3080 连放开 1 个 block 都无法容纳，单块预期收益约 1% | [stage2](../../_archive/docs/findings/krea2_3080_speed_stage2.md) |
| 3 | Krea-2 per-block compile | PG199 `+19.1%` | PG199 `3.370→2.726s`，峰值 `11.06GB`（eager `10.49GB`）；3080 仅 `3.3%` 冷态短窗口，已由阶段 5 修正 | [stage3](../../_archive/docs/findings/krea2_3080_speed_stage3.md) |
| 4 | compile × selective checkpoint 叠加 | 最优点仅作探针 | 最优点 `2.408s`（比 3.370s 快 `28.5%`）；安全推荐仍是 full checkpoint + compile `2.726s / 11.06GB`；3080 不适用 | [stage4](../../_archive/docs/findings/krea2_3080_speed_stage4.md) |
| 5 | 3080 20 步长窗口 + 热稳态 | 推翻 4 步短测结论 | compiled 首步 `25.167s`，第 2-4 步 `12.06s`，第 18-20 步 `12.57-12.65s`（20 步内漂移约 5%）；eager swap20 处于 OOM 边界 | [stage5](../../_archive/docs/findings/krea2_3080_speed_stage5.md) |
| 6 | FP16 compute dtype 前置消融 | `REJECT` | 以 `35.6%` 梯度偏移换速度不可接受；保留 NF4 `compute_dtype=bfloat16` 强制检查 | [stage6](../../_archive/docs/findings/krea2_3080_speed_stage6.md) |
| 7 | Inductor mode 消融 | `REJECT` | `reduce-overhead` 已证 checkpoint/CUDA Graph 运行时冲突；`max-autotune*` 未验证，显式拒绝而非默认转发 | [stage7](../../_archive/docs/findings/krea2_3080_speed_stage7.md) |
| 8 | LoRA rank 消融 | `NOT_WORTH_FOR_SPEED` | rank8/16 差异被冻结 DiT 大矩阵、NF4 反量化与 checkpoint 重算完全淹没；rank 应按容量/质量选 | [stage8](../../_archive/docs/findings/krea2_3080_speed_stage8.md) |
| 9 | multi-bucket compile 图复用 | `PASS` | 24 buckets 折叠；默认落为 `compile_dynamic_seq=false`、`compile_block_scope="resident"`、`compile_inductor_mode="default"` | [stage9](../../_archive/docs/findings/krea2_3080_speed_stage9.md) |
| 10 | compile × 完整检查点 | `PASS` | 两者正交：compiled graph 读取原位 reload 后的参数、optimizer reload 不破坏 backward、无需重新 `compile_blocks()`；检查点仍是 289MB | [stage10](../../_archive/docs/findings/krea2_3080_speed_stage10.md) |
| 11 | eager vs compiled CUDA operator profile | 给出优化天花板 | compiled 后 `mm + attention` self time 约 `2.44s`，占 `2.746s` 墙钟约 `89%`；NF4 dequant 仅约 `68ms` | [stage11](../../_archive/docs/findings/krea2_3080_speed_stage11.md) |
| 12 | FlashAttention varlen | 生产化为显式 opt-in | `L=4608` BF16：PG199 `27.93→14.80ms`（`-47.0%`）、3080 `47.12→30.94ms`（`-34.3%`）；3080 长窗口 `12.65→12.15s`（约 4%）；rel-L2 `0.00303`、cos `0.9999955`；默认仍为 cuDNN SDPA | [stage12](../../_archive/docs/findings/krea2_3080_speed_stage12.md) |

## 跨阶段主线

1. **瓶颈是硬件大矩阵吞吐**，不是 H2D、不是 3080 降频、也不是 padding 尾（阶段 1）。
2. **compile 是唯一被证明有效的软件手段**：PG199 `-19.1%`（阶段 3），并成为默认（阶段 9、10）。
3. **显存换速度只在 PG199 成立**：选择性 checkpoint 在 3080 上不可行（阶段 2、4）。
4. **已明确否决**：FP16 compute（阶段 6）、Inductor 激进 mode（阶段 7）、为速度降 rank（阶段 8）。
5. **软件优化的天花板已量化**：`mm + attention` 占墙钟约 89%，清理 Python/prepare/elementwise 不会再有两位数收益（阶段 11）。
6. **Flash varlen 是 3080 上首个长窗口仍有正收益的软件 attention 候选**，但只有约 4%，未改变 GEMM 主瓶颈（阶段 12）。

## 各阶段共同确立、当前仍有效的契约

- NF4 强制 BF16 compute，不提供全局 FP16 NF4 开关（阶段 6）。
- `compile_blocks` 只接受 `mode=None` / `"default"`；固定长度 opt-in 需 `compile_dynamic_seq=false` 与 `compile_block_scope="resident"`（阶段 7、9）。
- selective checkpoint 只支持 `off` / `every_other`；RTX 3080 保持 full checkpoint（阶段 2、4）。
- checkpoint/resume 后无需重复调用 `compile_blocks()`（阶段 10）。
- 冷态 compile 收益不能外推为持续收益；长窗口叠加热降频（阶段 3、5、12）。
