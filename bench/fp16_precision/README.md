# fp16 精度静态探针（P1–P9）

状态：稳定（CPU-only）。适用版本：当前 dev。
入口命令：CUDA_VISIBLE_DEVICES='' .venv/bin/python bench/fp16_precision/run_probe.py
相关代码：library/training/loop.py、library/training/finite_checks.py、library/runtime/accelerator.py
相关文档：[fp16 保护与 NaN 后处理：静态等效判定与待热测](../../docs/findings/fp16_nan_static_equivalence_20261021.md)

本目录放两个探针脚本：CPU 探针 `run_probe.py` 给静态验收报告提供**可离线重跑**的数字来源
（报告中 P1–P9 全部由它产出，不需要 GPU 与模型权重）；CUDA 合成探针 `run_cuda_probe.py`
补 H1/H2/H8 的真机读数（需要 GPU，见下文）。

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
- 2026-11-06 起 P3/P6 只描述**历史**生产形态：D1/D2 修复后 `loop.py` 不再调用
  `check_trainable_grads_finite`，P6 的 `check_trainable_grads_finite_line` 为 -1、
  `order_loss_grads_clip_step_sched` 为 false；P3 仍复现「函数默认参数下对 scaled 梯度
  fail-closed」，但已不代表主循环路径。所有权语义见 findings「修复状态」。
- 实测环境是 torch 2.10.0+cu129 / accelerate 1.13.0（`pyproject` 要求 torch>=2.12.0,<2.13），
  对接缝结论不影响，但严格对齐版本需重跑。

## CUDA 合成探针（run_cuda_probe.py）

状态：实验（需要真 CUDA）。适用版本：当前 dev。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python bench/fp16_precision/run_cuda_probe.py --gpu-label gpu0
```

一个进程只占用一张卡（进程内是 `cuda:0`），只做合成张量验证，不加载模型、不读数据集。
可用 `--skip-h1/--skip-h2/--skip-h8` 单独跑。

| 探针 | 验什么 | 对应待热测 |
|---|---|---|
| H1 | accelerate `backward → unscale+clip → step` 与直接 `GradScaler` 接缝的 loss/scale/grads/params 逐字节对照，含「参数确实相对初值移动」负控制 | H1 |
| H2 | CUDA fp16 unscale 表示窗口采样，并要求 `found_inf` 与「是否上溢 / 是否被清零」语义一致 | H2 |
| H8 | `GradScaler.state_dict()` 保存后用不同 `init_scale`/`growth_interval` 的新对象恢复，续跑序列的 scale 轨迹一致 | H8 |

2026-11-06 在 2× Tesla V100-SXM2-16GB 上各自独占跑通，H1/H2/H8 均 pass，两卡输出除
`--gpu-label` 外逐字段一致。边界：只覆盖合成张量（`Linear(8,4)` 与 `(1,)` 参数），
不代表真实 trainer / 模型族 / 数据链路；H8 不含 checkpoint 文件往返与跨设备搬运。
