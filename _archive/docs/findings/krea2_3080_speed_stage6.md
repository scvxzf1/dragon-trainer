状态：阶段 6 已完成（FP16 compute dtype 前置消融 → REJECT）
日期：2026-08-09
原始摘要：`krea2_3080_speed_stage6.json`

# Krea-2 RTX 3080 速度研究：阶段 6

## 动机

RTX 3080 的 FP16/BF16 tensor core 理论吞吐接近，但实际 cuBLAS 内核可能不同。
现有 `quantize.py` 和 `weights.py` 明确将 NF4 `compute_dtype` 锁定为 BF16，因此
本阶段只做独立层前置探针，不绕过生产拒绝逻辑。

## 算子速度

| GPU | dtype | Linear 6144×6144 | GQA attention |
| --- | --- | ---: | ---: |
| PG199 | BF16 | 1.517ms | 4.938ms |
| PG199 | FP16 | 1.517ms | 4.812ms |
| RTX 3080 | BF16 | 8.176ms | 12.489ms |
| RTX 3080 | FP16 | 7.052ms | 12.438ms |

FP16 在 PG199 没有 Linear 收益，在 3080 大 Linear 上快 13.7%；attention 基本持平。

## 同权重、同输入的 NF4 前后向

| 指标 | BF16 compute | FP16 compute |
| --- | ---: | ---: |
| forward | 8.512ms | 8.252ms |
| backward | 27.071ms | 10.018ms |
| 有限值 | PASS | PASS |

数学差异：

- 输出：max delta `0.015625`，rel-L2 `0.2885%`
- 输入梯度：rel-L2 **`35.65%`**，cosine `0.9434`

FP16 backward 速度收益很大，但梯度轨迹偏移远超现有 NF4 vs BF16 实现的可接受
数值噪声口径。这不是可以靠 loss scaling 简单证明安全的改动：偏移来自整个
dequantized FP16 前后向，不只是最终 loss 标量下溢。

## 判定

**REJECT**：保留 NF4 `compute_dtype=torch.bfloat16` 强制检查，不提供 FP16 开关，
不进行完整训练。RTX 3080 12s/it 的大矩阵瓶颈不应以 35.6% 梯度偏移
换取速度。

后续若研究 mixed precision，必须从更细粒度开始（例如仅非训练路径的特定
projection），并建立多步梯度 cosine/loss 轨迹门槛；不应修改当前全局 NF4 契约。
