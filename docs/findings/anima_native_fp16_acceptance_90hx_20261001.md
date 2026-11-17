# Anima 旧 FP16：12格集中验收与质量预登记

日期：2026-10-01。状态：执行和独立审计完成，严格数值失败，质量尚未运行。
范围：Anima plain LoRA、CMP 90HX 10GB（SM86），固定旧 native FP16 AMP、scale1024。
本轮落实[收敛计划（已归档）](../../_archive/docs/proposal/adaptive_precision_delivery_reset_20261001.md)的单候选集中验收，
结束05/04默认扩张，未改生产默认、旧证据或通用数学比较器。

## 结论

12个案例全部 finite、各完成真实独立 AdamW 更新、零跳步，scale全程1024。
运行和独立NumPy FP64审计均exit0，证据完整性通过；原严格数值门槛 **0/12通过**。
prediction误差12/12在门槛内，输入和adapter梯度误差12/12均超标。
当前拒绝将旧FP16作为通过原严格标准的自动计划；保持显式实验候选身份，不追加层或scale搜索。
质量标准已提前登记，数据/prompt尚未绑定，`quality=not_run`；严格失败不直接证明训练质量失败。

| 指标 | 最差relL2（比例） | 原门槛 | 结果 |
| --- | ---: | ---: | --- |
| prediction | 0.001324490 | .01 | 12/12通过 |
| input gradient | 2.089836740 | .02 | 12/12失败 |
| adapter gradient | 0.653534682 | .02 | 12/12失败 |
| 实际AdamW更新 | 0.732433814 | 无独立接受阈值 | 仅记录 |

input/adapter梯度最小cosine为 -0.040380372 / 0.814322354，原门槛 .999。
实际更新的最大差异绝对L2为0.117576624、最小cosine0.733685081；
这里的绝对L2不是单个参数最大变化量。逐case最大参数变化仍约1e-4，确有有效更新。
三个更新分组最差relL2：MLP .737240356、self attention .770571624、cross attention .672786466。
这些更新指标不覆盖原严格梯度失败。

## 固定状态与比较坐标

复用批次2 representative 4032与heldout 4200 native token包，交叉sigma .2/.5/.8、
initial/nonzero adapter和AdamW状态。每格独立恢复参数、buffers、moments/step及四类RNG，
随后clip1、AdamW lr1e-4执行一次真实更新；两worker各6格，不是连续12步训练。
原reference使用BF16 checkpoint数值提升FP32计算，不是原生FP32权重来源。

旧source/replay/state文件身份保持不变；当前旧数值源码ID仍为
`1f5ea4b1d29e337500a5270075fe9f9fd2d7b9a8a0c1e5d0eab50c51c8caed71`。
新入口只在私有worker临时替换构建、校验和forward，调用原六格执行流程，finally恢复。
新执行源码和真实策略另绑定到plan及最终result；canonical signature描述来源比较合同，
不能将其中旧timestep来源描述理解成actual FP16模型边界。

固定FP32 latent/noise/sigma先混合，模型调用处noisy、timestep、prompt及padding转FP16，
target保持FP32；输入梯度仍捕获转换前的FP32 leaf，标记
`canonical_fp32_pre_boundary_cast`，实际FP16张量dtype/digest单独落盘并独立重算。
本轮复现固定随机样本的精度边界，不代表生产随机采样全过程。

首格 initial/sigma=.2 实际观察到block0四个attention/MLP Linear输出FP16、AMP开启，
最终投影输入/计算权重/输出FP32、内部AMP关闭。377个普通Linear权重存储FP16，
LoRA可训练参数FP32，28 block与final共29个保护标志生效；未枚举全部算子或backward dtype。

## 数据与资源

| role | 有效独立更新/跳步 | GPU峰MiB | 进程树RSS峰B | host available最低B | allocated/reserved峰GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| representative | 6 / 0 | 3528 | 11,259,269,120 | 103,954,042,880 | 3.0075 / 3.1777 |
| heldout | 6 / 0 | 3582 | 11,241,869,312 | 102,976,618,496 | 3.0618 / 3.2305 |

loss范围分别.122226–.261744、.095569–.188347，12格均finite。
遥测73/74条、错误均0，swap峰0，最高温52°C；结束目标GPU回到17MiB、0%利用率。
两个监控窗口78.986/79.625s，编排186.308s，产物522,821,832B（约499MiB）。
单case compute均值2.479/2.518s包含边界观察开销，不作纯kernel或连续训练稳态步时。
[此前五步生产链路](anima_legacy_fp16_comparison_90hx_20261001.md)的2.694s/3524MiB仍是独立短窗证据，
不能与此矩阵混成同一计时定义。

## 质量标准与停止条件

[质量协议与入口](../experimental/anima-native-fp16-acceptance.md)已在GPU执行前登记并绑定SHA。
未来有限窗口为3个配对seed、每路径50有效更新，生产BF16对照；独立validation与prompt需先绑定，
现有矩阵样本不再作为未见quality集。终点validation loss比值中位数≤1.01、最差seed≤1.03，
参考需有预定改善信号，FP16保留至少80%参考中位改善，并完成固定prompt盲评。
参考信号不足、评估缺失或预算未完成时为inconclusive。门槛是工程标准，不是统计置信度或通用安全证明。
本轮未启动这些50更新/生成任务，协议 `future_run_authorized=false`、`quality_status=not_run`。

单候选矩阵预算已完成，不继续05/04或逐层/scale搜索。质量验收与原严格数值分别记录，
今后的质量通过不会把本次严格失败改判；五步finite同样不认证质量。
旧默认scaler、长期收敛、Turing硬件、其他adapter及更新后OOM恢复仍未验证。

## 实现、审计与验证

新增[`native执行与窄比较器`](../../bench/adaptive_runtime/anima_native_validation.py)、
[`冻结计划与编排`](../../bench/adaptive_runtime/anima_native_acceptance.py)及
[`CPU契约测试`](../../tests/test_adaptive_anima_native_validation.py)。
准备阶段完整核验旧张量、state/replay来源；运行阶段重核全部冻结字节，避免重复张量解析。
CPU定向测试13通过（独立审核5.64s），Ruff通过；旧公共SOURCE_FILES未改。

原始目录：`output/runs/anima-adaptive-precision-90hx-20261001-delivery-batch2-legacy-fp16-acceptance/`（本地未入库产物）。
完整逐格报告（`output/runs/anima-adaptive-precision-90hx-20261001-delivery-batch2-legacy-fp16-acceptance/report.md`）、
独立 FP64 审计（`output/runs/anima-adaptive-precision-90hx-20261001-delivery-batch2-legacy-fp16-acceptance/independent-audit.json`）、
决策记录（`output/runs/anima-adaptive-precision-90hx-20261001-delivery-batch2-legacy-fp16-acceptance/decision.json`）
均已落盘。独立审计的`passed=true`仅表示证据与重算自洽；`strict_passes=0`、`strict_diagnostic=fail`。

首次启动命令漏`--run`，只在准备阶段拒绝已有目录，未启动GPU；日志保留。
修正后唯一实际GPU批次及独立CPU审计均exit0，命令/退出码/遥测均保留在新目录。
