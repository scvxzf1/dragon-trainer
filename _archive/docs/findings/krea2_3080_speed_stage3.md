状态：阶段 3 已完成（Krea-2 per-block compile）
日期：2026-08-09
原始摘要：`krea2_3080_speed_stage3.json`

# Krea-2 RTX 3080 速度研究：阶段 3

## 实现

`SingleStreamDiT.compile_blocks()` 在 LoRA apply/load 和 checkpoint 设置之后编译
`SingleStreamBlock._forward`，不编译外层 checkpoint wrapper。这与 Anima 的
compile-after-apply 不变量一致，同时保留 Krea-2 自己的 combined sequence 256 对齐。

块交换默认 `compile_block_scope="resident"`：只编译常驻头部，不编译会在
CPU/CUDA 之间重绑 `Params4bit` 权重的交换尾部，避免 device guard 重编译。

当前只支持 `dynamic_seq=false`。固定 1024² 且 Qwen 文本缓存固定 512 token 时
只有一个 padded 序列长度；多 bucket/动态文本的重编译预算未完成，因此未更改
Krea-2 默认配置。

## 兼容性前置探针

PG199 小型 `SingleStreamBlock`，含标准 checkpoint backward：

| 权重 | eager | compiled | 提升 |
| --- | ---: | ---: | ---: |
| BF16 | 13.322ms | 7.807ms | 70.6% |
| NF4 | 20.548ms | 10.690ms | 92.2% |

输出 max delta `0.03125`（bf16 舍入量级），输入/条件梯度 max delta 不超过
`3.052e-5`。bitsandbytes `matmul_4bit`、cuDNN SDPA、checkpoint recompute 均可通过
Inductor。

## 完整训练消融

| GPU | 配置 | eager 稳态 | compiled 稳态 | 提升 | 首步 |
| --- | --- | ---: | ---: | ---: | ---: |
| PG199 | NF4, swap0, full ckpt, 28 resident | 3.370s | 2.726s | **19.1%** | 28.644s |
| RTX 3080 | NF4, swap20, full ckpt, 8 resident | 12.140s | 11.744s（短窗口） | **3.3%（冷态）** | 35.903s |

PG199 compiled 峰值 `11.06GB`（eager `10.49GB`），loss 有限、梯度非零。3080 三个
稳态步为 `11.645/11.777/11.809s`，无 OOM，loss 有限且总体下降。短探针
因 4 步的 first-N/last-N 窗口重叠而返回 1，不是训练失败。

3080 这个 3.3% 是短窗口冷态结果。阶段 5 的 20 步复测显示步时从
12.06s 漂到 12.65s，120 秒纯 GEMM 在 84°C 复现同方向退化；因此**不再
声称 60 步回本或持续快 3.3%**。compile 在 3080 上的已证价值是将当前
eager OOM 边界点变为 20 步可运行，见 [stage5](krea2_3080_speed_stage5.md)。

## 使用边界

固定 1024² opt-in：

```toml
torch_compile = true
compile_dynamic_seq = false
compile_block_scope = "resident"
```

- PG199 可与 `selective_checkpoint="every_other"` 组合，但两者组合峰值/数学尚未实测，
  本阶段不声称可叠加。
- RTX 3080 保持 full checkpoint + swap20，compile 仅覆盖 8 个 resident block，因此
  收益只有 3.3%，符合上限。
- `compile_dynamic_seq=true` 会显式拒绝；多分辨率训练在完成重编译预算前保持 eager。

下一阶段应测 PG199 `compile + every_other`的叠加性，并用 20-30 步稳态窗口
复核长训均值。
