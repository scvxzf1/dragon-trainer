# Anima 旧 FP16 AMP：90HX 五步补充对照

日期：2026-10-01。状态：实验 / 受限五步链路与独立审核完成。
后续更新：[12格集中验收](anima_native_fp16_acceptance_90hx_20261001.md)已完成，证据自洽、严格0/12通过。
以下“严格矩阵未运行”描述本五步实验当时范围；质量仍未运行，原短测结果保留。
适用范围：当前工作树、Anima plain LoRA、CMP 90HX 10GB（SM86）。
这是[批次2](anima_precision_delivery_batch2_90hx_20261001.md)遗漏的旧 FP16 基线补测，
保留原四组结果和严格矩阵结论；不修改生产默认或自动精度策略。

旧 native FP16 完成 **5 次有效更新、0 跳步、exit 0**，loss 与最终权重均有限。
短窗平均步时 **2.694 s**，整卡峰值 **3524 MiB**，明显优于主体 FP32 的候选05/04。
这条已有生产路径应成为后续 FP16 方案的核心对照。
当前结果没有认证旧 FP16 的严格梯度精度、长期质量或 Turing，不能将其直接写成通用自动选择方案。

## 精度实现与测试条件

测试配置为 `mixed_precision="fp16"`、`adaptive_precision="off"`、
`full_fp16=false`、`full_bf16=false`。生产入口保留 Accelerate autocast，基座存储 FP16，
LoRA 可训练参数 FP32；bootstrap 在 compile 前自动开启 FP32 residual/final guard。
本次 28 个 block 与 final guard 均生效。

源码边界：[`accelerator.py`](../../library/runtime/accelerator.py)的 `prepare_dtype` 与
`prepare_accelerator`，[`model_loading.py`](../../library/training/model_loading.py)的
`_load_anima_dit`，[`bootstrap.py`](../../library/training/bootstrap.py)的
`maybe_enable_fp32_residual`，以及 [`models.py`](../../library/anima/models.py)的
`FinalLayer._fp32_project`、`Block._gated_residual_add`。
普通 FP16 使用 AMP；新 FP16 islands 的主体 FP32、关闭 AMP，二者不可混称。

实测 377 个普通 Linear 权重存储 FP16；首步 `blocks.0` 的 self-attention
qkv/output projection 与 MLP layer1/2 输出 FP16，autocast 开启。
最终 `_fp32_project` 输入、计算权重与输出 FP32，内部 autocast 关闭、调用侧开启。
存储名单不能代表全部实际 GEMM；本次观察覆盖上述五处，未枚举所有算子和 backward dtype。

唯一额外数值设置是 run-local `GradScalerKwargs(init_scale=1024)`，五步维持 1024，
保留 AMP。旧默认 scaler 未测试；配置中的 `adaptive_loss_scale` 在 adaptive off 下不替代该覆盖。
其余与生产 BF16 对照匹配：批次2两个缓存样本、4032/4200 native token、max-padded text、
seed 20261001、rank/alpha 4、AdamW、lr 1e-4、batch/accum 1、clip 1、
swap 20、torch attention、full checkpoint、关闭 compile、BF16 最终保存、max steps 5。
初始 adapter 与 FP32 基线逐 tensor 完全相同。

## 统一五组对照

步时取 `step_enter.ts → after_optimizer.ts` 的有效更新 2..5，包含同步与取证开销。
显存、RSS 均为各 run 全程峰值；allocator GiB 与外部整卡 MiB 分开记录。

| 方案 | 步2..5均值 s | GPU峰 MiB | allocated / reserved 峰 GiB | 进程树RSS峰 GiB |
| --- | ---: | ---: | ---: | ---: |
| 生产 BF16 | 2.6240 | 3104 | 2.6334 / 2.7617 | 9.627 |
| 旧 FP16 AMP，scale 1024 | 2.6940 | 3524 | 3.0350 / 3.1719 | 10.232 |
| 同路径 FP32 | 7.3775 | 5232 | 4.7024 / 4.8340 | 17.883 |
| 候选05：blocks14..27 MLP2 FP16，其余FP32 | 7.0150 | 5730 | 4.4524 / 5.3223 | 14.115 |
| 候选04：blocks21..27 MLP2 FP16，其余FP32 | 7.19825 | 5674 | 4.4837 / 5.2637 | 15.861 |

五组均五次有效更新、零跳步。旧 FP16 相对 FP32 的步时减少 63.48%、GPU峰值减少32.65%；
相对05/04分别减少61.60%/62.57%步时、38.50%/37.89%显存。
相对生产 BF16 则增加2.67%步时、13.53%显存，当前 Ampere 短窗仍由 BF16 更省。
旧 FP16 步时中位数2.6805s；分别运行的冷启动、温度及取证版本差异仍存在，不能把这些比例外推稳态或 Turing。

旧 FP16 外部 RSS 峰10,986,840,064 B，主机 available 最低103,247,228,928 B，
65条遥测、0错误、swap峰0，全监控温度45–55°C。
监控总耗时70.302s与 `run_end.ts=45.815` 计时起点/覆盖范围不同，均不作纯训练步时。
没有改功耗/风扇、启动长训练或触碰另一张 GPU 的用户进程。

## 验收、证据与下一步

五个 loss 和五个更新范数有限，更新均大于零；最终840个保存张量全部finite BF16，
560个相对初始改变，`ss_steps=5`。旧 FP16 相对 FP32 的逐步loss最大绝对差2.82675e-5，
相对BF16为2.99871e-4；这是各自训练轨迹的观察，没有预注册质量阈值，也不是同状态 replay。

Luna 执行唯一 GPU 热测，Sol 独立 CPU 审核 `status=pass`，无阻断 P1/P2。
旧四组仍使用原 `production-audit.json`，新增汇总绑定每组原始文件SHA、当前数值源码、
热测工具及两份独立审核；CPU汇总命令exit 0，复得原四组步时与资源值。
早期 `cpu-selfcheck.json` 保留了最终观察补丁前的哈希，当前身份由冻结计划、训练manifest、
独立review及汇总核验补足，不能说旧快照执行了最新观察代码。

原始目录为 `output/runs/anima-adaptive-precision-90hx-20261001-delivery-batch2-legacy-fp16/`。
完整运行报告（`output/runs/anima-adaptive-precision-90hx-20261001-delivery-batch2-legacy-fp16/report.md`）、
五组汇总（`output/runs/anima-adaptive-precision-90hx-20261001-delivery-batch2-legacy-fp16/comparison.json`）、
独立审核（`output/runs/anima-adaptive-precision-90hx-20261001-delivery-batch2-legacy-fp16/review.json`）
均已落盘（本地未入库产物）。CPU复算可用 `.venv/bin/python <该目录>/comparison_cpu.py --output <新的JSON路径>`，
拒绝覆盖旧输出，不调用 GPU。

候选05/04仍只有严格矩阵数值优势，没有实用显存收益；旧 FP16 已显示短窗性能优势，
其严格12格矩阵尚未运行。prediction relL2≤.01、input/adapter grad relL2≤.02、
grad cosine≥.999及finite门槛保持不变，短训成功不能继承矩阵通过结论。
后续先将旧 FP16 纳入数值与预注册质量对照，再决定自动计划的接受/拒绝；
计划身份绑定、通用总报告、旧默认scale、长期质量及Turing硬件验证仍待完成。
此次也没有验证更新后OOM恢复。
