状态：阶段 2 已完成（选择性梯度检查点）
日期：2026-08-09
原始摘要：`krea2_3080_speed_stage2.json`

# Krea-2 RTX 3080 速度研究：阶段 2

## 目标

梯度检查点在 backward 重算 28 个 block，是除硬件吞吐外最明确的固定速度成本。
本阶段通过逐块开关测量“少 checkpoint 一部分 block”的速度/显存曲线，再在
RTX 3080 上测最小可行颗粒。

## 结果

| GPU | block swap | 激活 checkpoint | GPU peak | 稳态 step | 结果 |
| --- | ---: | --- | ---: | ---: | --- |
| PG199 | 0 | 28/28（full） | 10.49GB | 3.370s | PASS |
| PG199 | 0 | 14/28（every-other） | 28.46GB | 2.901s | PASS |
| PG199 | 0 | 26/28（last2 不 ckpt） | 11.83GB | 3.299s | PASS |
| RTX 3080 | 22 | 20/22 可见尾块（last2 不 ckpt） | - | - | OOM |
| RTX 3080 | 20 | last1 不 ckpt | - | - | OOM |

PG199 `every_other` 相对 full 稳态快 **13.9%**，代价是增加 17.97GB 激活峰值；
28.46GB 在 32GB PG199 可容纳，因此是有实测价值的 opt-in 性能档。仅放开末尾
2 块只快 2.1%，不足以独立成为配置。

3080 两格均在 LoRA forward 追加 `144MiB` 临时张量时 OOM。swap22+last2 时
PyTorch 已分配 6.99GiB、reserved 未用 681MiB；swap20+last1 时已分配
6.98GiB、reserved 未用 604MiB。卡上还有约 436MiB 桌面/服务常驻。

## 判定

- **PG199：WORTH_IT**。实现 Krea-2 `selective_checkpoint=every_other`，必须同时关闭
  `gradient_checkpointing`，预期 1024² NF4 为 28.46GB / 2.90s。
- **RTX 3080：NOT_FEASIBLE**。当前 10GB 工作点连放开 1 个 block 都无法容纳，
  选择性 checkpoint 不能解决 12s/it。不应通过 allocator 环境变量或关桌面服务
  强行挤入：即使成功，单块预期收益也只约 1%，且无安全余量。

## 实现边界

Krea-2 `SingleStreamBlock` 只有整块 checkpoint 语义。本阶段仅实现与现有语义
完全一致的 `off/every_other`；`adapter_aware`、`mlp_only`、`peak_blocks_*` 需要
Krea block 内部新 checkpoint 切面，当前显式拒绝，不做静默降级。

## 使用

PG199 32GB 性能档：

```toml
gradient_checkpointing = false
selective_checkpoint = "every_other"
```

3080 10GB 继续使用：

```toml
gradient_checkpointing = true
selective_checkpoint = "off"
```

下一阶段转向 Krea-2 per-block compile 前置消融，先 PG199 BF16/NF4 兼容性，
不直接在 3080 开启。
