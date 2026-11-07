状态：阶段 4 已完成（PG199 compile × selective checkpoint 叠加）
日期：2026-08-09
原始摘要：`krea2_3080_speed_stage4.json`

# Krea-2 速度研究：阶段 4

## 目标

阶段 2 的 every-other checkpoint 单独快 13.9%，阶段 3 compile 单独快 19.1%。
本阶段验证两者是否可叠加，并用 20 步窗口复核最佳可运行点。

## 矩阵

| compile | checkpoint block | 结果 | 稳态 step | 峰值/失败点 |
| --- | ---: | --- | ---: | --- |
| off | 28/28 | PASS | 3.370s | 10.49GB |
| off | 14/28 every-other | PASS | 2.901s | 28.46GB |
| on | 28/28 | PASS | 2.726s | 11.06GB |
| on | 14/28 every-other | **OOM** | - | allocated 30.20GiB，余 39.9MiB，再需 144MiB |
| on | 16/28（blocks 0-15 ckpt） | PASS | **2.408s** | 探针报 31.55GB |

compile + every-other 不能直接叠加，首步 Inductor 计划和大 MLP 临时张量会用完
PG199 显存。多 checkpoint 两块、仅放开末尾 12 块后可运行。

## 20 步长窗口

`compile + checkpoint blocks 0-15 + uncheckpoint blocks 16-27`：

- 首步（含编译）：`15.034s`
- 后 19 步：`2.406-2.409s`，均值 `2.408s`
- 相对 full-eager `3.370s`：**-28.5%**
- loss：`0.011841→0.002274`，first5 `0.008398`，last5 `0.002438`
- 梯度：全步非零，无 NaN/Inf
- 报告峰值：`31.55GB`，等于该卡报告总容量

稳态方差极小，速度和数学训练都已复核。但首步峰值没有安全余量，任何桌面
进程、更长文本、不同 allocator 状态都可能让它 OOM。

## 判定

- **速度研究最优点**：2.408s/it，比原 PG199 3.370s 快 28.5%。
- **安全推荐点**：仍是 full checkpoint + compile，2.726s/it / 11.06GB，余量大。
- **实验点**：16/28 checkpoint + compile 只供 PG199 专用探针，不进入默认配置，
  也不扩展成通用 CLI 模式。
- **RTX 3080**：阶段 2 已证明放开单 block 都 OOM，不适用该叠加方案；其
  安全优化仍是 full checkpoint + resident compile，约 11.74s/it。

如要复现实验点，使用 `scripts/krea2/probe_nf4_ablation.py` 的
`K2_ABL_GRAD_CKPT=full_except`、`K2_ABL_UNCKPT_BLOCKS=16,...,27`、
`K2_ABL_COMPILE=1`；不要改生产默认。
