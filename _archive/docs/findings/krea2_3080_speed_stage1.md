状态：阶段 1 已完成（瓶颈定位 + 两个候选优化消融）
日期：2026-08-09
原始摘要：`krea2_3080_speed_stage1.json`
相关探针：`scripts/krea2/probe_nf4_ablation.py`、`scripts/krea2/probe_nf4_h2d_bottleneck.py`

# Krea-2 RTX 3080 12s/it 速度研究：阶段 1

## 问题与口径

目标是解释同一台机器上 RTX 3080 10GB 训练 Krea-2 1024² NF4 LoRA 为什么
约 `12s/it`，而 32GB PG199 约 `3.37s/it`。所有训练对照都使用 bf16 compute、
LoRA dim 16、完整 28-block 梯度检查点；3080 使用已验证的 NF4 + swap20 工作点。

## 结论摘要

1. **12s/it 首要是硬件大矩阵吞吐差距**，不是 H2D，也不是 3080 降频。
2. 3080 的代表性 BF16 Linear 比 PG199 慢 `4.4-5.2x`，NF4 Linear 慢 `4.1-4.6x`；
   attention 慢 `2.83x`。完整步 `12.1/3.37=3.60x` 落在这两类算子之间。
3. NF4 在各卡上相对 BF16 的代表性 Linear 额外开销只是 `13-25%`。它是 10GB 卡
   可以训练的前提，但不是卡间 `3.6x` 差距的主因。
4. 本阶段测试的两个低风险候选均为 `NOT_WORTH`：每步 block prepare 不存在于生产
   稳态路径；文本 padding 尾裁剪在两卡均无可测加速，已回退行为改动。

## 同机双卡硬件与算子消融

| 指标 | PG199 32GB | RTX 3080 10GB |
| --- | ---: | ---: |
| compute capability / SM | 8.0 / 96 | 8.6 / 68 |
| memory bus / L2 | 4096-bit / 32MiB | 320-bit / 5MiB |
| BF16 Linear 6144×6144, 4608 tokens | 1.523ms | 7.844ms |
| NF4 Linear 6144×6144, 4608 tokens | 1.908ms | 8.870ms |
| BF16 Linear 16384×6144, 4608 tokens | 4.025ms | 17.831ms |
| NF4 Linear 16384×6144, 4608 tokens | 4.976ms | 20.354ms |
| cuDNN SDPA, GQA+dense mask | 4.952ms | 14.015ms |

3080 持续 BF16 GEMM 时实测 `P2`、`1905-1950MHz`、GPU utilization `100%`、
`315-319W / 320W`，不存在低时钟、低功耗或温度降频。PCIe 在负载下 Gen3 x16，
但既有 swap0/swapN 数据已证明 H2D 不是主导段。

Flash SDPA 本软件栈不能直接替代：Krea-2 是 `Hq=48/Hkv=8` GQA 且传入非空 dense
mask，PyTorch 2.12 明确拒绝 Flash backend（non-null mask + head mismatch）；强制改后端
不是可用优化。

## prepare 口径修正

PG199 1024²、NF4、swap4、8 步（跳 3 步预热）：

| 段 | swap4 | swap0 |
| --- | ---: | ---: |
| `prepare(free_cache=False)` | 195.66ms | 0.025ms |
| forward + backward | 3357.99ms | 3368.07ms |
| 含 prepare 墙钟 | 3.556s | 3.370s |

`library/training/unet_prepare.py` 只在 accelerator prepare 和 validation 恢复时调 prepare，
生产训练不是每步调用。旧探针每步 prepare 会多计约 196ms，但不能用
“去掉 196ms”声称生产加速，因为生产本来就没有这段。探针现已默认
`free_cache=False` 并单独记录 prepare。

## 文本 padding 尾裁剪消融

短提示词只有 11/512 个有效文本 token，裁剪后 single-stream 总长度从 4608
降到 4107。实测却无收益：

| 卡 | 未裁剪 | 裁剪 | 判定 |
| --- | ---: | ---: | --- |
| PG199, NF4, swap0 | 3.370s | 3.366s | 噪声级 |
| RTX 3080, NF4, swap20 | 12.14s（历史 10 步） | 12.48-12.52s（本轮稳态） | 无收益 |

显存仍为 `7.65GB`，loss 有限且下降。短复测因 host RSS 门槛 `<20GB` 与实测
`20.44GB` 不符而返回 1，但训练 6 步全部完成，该退出码不是数学或 OOM 失败。

## 下一阶段

按预期收益排序：

1. 使用单次加载、稳态复用的探针分离 grad-checkpoint 首次 forward、recompute 和
   backward，评估选择性 checkpoint 在 3080 约 1GB 显存余量内的可行性。
2. 为 Krea-2 建立 per-block `torch.compile` 接口后先在 PG199 做显存/编译时间/步时
   前置消融；未证明收益前不在 10GB 卡默认开启。
3. 不再投入 H2D overlap、每步 prepare 或 padding 尾裁剪，除非新硬件/新后端改变
   前提。
