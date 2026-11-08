状态：阶段 10 已完成（compile × 完整检查点）
日期：2026-08-09
原始摘要：`krea2_3080_speed_stage10.json`

# Krea-2 compile × checkpoint 续训消融

## 风险

Krea-2 compile 在 LoRA apply/load 之后捕获 block `_forward`。中途 checkpoint reload 会将
LoRA 和 optimizer state 写回已 monkey-patch 的模块。如 compiled graph 捕获了旧参数值/
旧 callable，可能出现 reload 无效、重编译或 backward 失败。

## 实测

PG199、1024²、NF4、swap0、full activation checkpoint、default compile，10 步，第 5 步后
保存 LoRA + optimizer，立即 reload 并续训：

| 指标 | 结果 |
| --- | ---: |
| LoRA checkpoint | 96.403MB |
| optimizer state | 193.004MB |
| 保存时间 | 1.304s |
| LoRA round-trip max delta | **0** |
| forward round-trip max delta | **0** |
| reload 前/后 loss | 0.004242 → 0.004456 |
| loss jump | 0.000214 |
| GPU peak | 11.058GB |

步时：

```text
15.303, 2.731, 2.731, 2.731, 2.731,
2.730, 2.728, 2.729, 2.729, 2.730
```

首步含编译；reload 后第一步 2.730s，与 reload 前 2.731s 一致，没有新的
编译峰值或速度退化。loss 总体下降，梯度非零。

## 判定

**PASS**。默认 fixed resident compile 与现有完整 checkpoint 契约正交：

- compiled graph 读取原位 reload 后的 LoRA 参数，不捕获旧值。
- optimizer state reload 不破坏 backward。
- 不需要 reload 后重新调 `compile_blocks()`。
- NF4 DiT 仍是冻结 master，不进 checkpoint，保持原有 289MB 检查点体积。

该结果支持阶段 9 的 Krea 默认 compile 改动用于正常长训和续训。
