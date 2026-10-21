# fp16 精度静态探针（P1–P9）

状态：稳定（CPU-only）。适用版本：当前 dev。
入口命令：CUDA_VISIBLE_DEVICES='' .venv/bin/python bench/fp16_precision/run_probe.py
相关代码：library/training/loop.py、library/training/finite_checks.py、library/runtime/accelerator.py
相关文档：[fp16 保护与 NaN 后处理：静态等效判定与待热测](../../docs/findings/fp16_nan_static_equivalence_20261021.md)

本目录只放一个探针脚本，用途是给静态验收报告提供**可离线重跑**的数字来源。
报告里引用的 P1–P9 全部由本脚本产出，不需要 GPU，也不需要模型权重。

## 九个探针

| 编号 | 验什么 | 对应判据 |
|---|---|---|
| P1 | 真 torch.amp.GradScaler("cpu") 的 growth / backoff / state_dict 往返 | E4 |
| P2 | 溢出步：step_was_skipped、内层 optimizer.step 未调用、权重逐位不变、scale 退档 | E2 |
| P3 | --debug_finite_checks 在 unscale 之前读梯度 → 例行溢出被误报为训练失败（缺陷 D1） | E5 |
| P4 | 裁剪顺序反例：先 clip 后 unscale 时 wrong/correct = 1/scale | 假绿反例 |
| P5 | 无 scaler 路径：loss 对象身份保持、grads/params 逐位相等 | E1 |
| P6 | loop.py 关键行号与调用顺序的静态抽取 | E2/E3 |
| P7 | 生产代码里窗口记账 / 所有权判定符号的命中数（全 0） | E3/E5 |
| P8 | unscale → clip → step 的范数与权重增量对照 | 假绿反例 |
| P9 | fp16 下 unscale 的真梯度可用窗口 [2^-24, 65504/scale] | 数学边界 |

## 边界

- 全部结论只对 **CPU 可判定面** 成立；真实 CUDA autocast、unscale kernel 舍入、跨 rank
  found_inf 传播、长程数值漂移仍需 GPU 热测，见报告「待热测清单」H1–H8。
- P4 的比值**依赖 scale**（默认 65536）；换 init_scale 必须重算，不能直接引用。
- 本目录不参与测试套件；契约回归在 tests/test_fp16_nan_static_equivalence.py。
