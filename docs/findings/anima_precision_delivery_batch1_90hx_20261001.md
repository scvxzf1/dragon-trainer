# Anima Linear精度交付批次1：CMP 90HX

日期：2026-10-01。状态：有界筛选、热测及独立CPU数值验收完成；生产计划未开放。
依据：[交付收敛计划（已归档）](../../_archive/docs/proposal/adaptive_precision_delivery_reset_20261001.md)。

## 结论

12个不同的可部署普通Linear组合中，10个通过本固定案例的严格prediction/梯度门槛。
候选05把blocks14..27的MLP2改为FP16，候选04仅改blocks21..27；其余普通Linear保持FP32，
残差流与LoRA参数保持FP32。这是实际dtype干预，没有替换参考激活或反向梯度。
两者进入下一批保留集与训练验证，不能直接作为生产推荐名单。

候选05的prediction、输入梯度、adapter梯度相对L2误差分别为0.005062%、0.156350%、
0.081213%，实际AdamW更新delta误差0.903123%。它只覆盖源权重元素的11.233%，
理论FP32权重payload减少5.616%。本次GPU峰值5738MiB，高于全FP32控制的5236MiB，
尚未证明显存或稳态速度收益。批次1回答了有限范围的数值可行性，实用收益仍未完成验证。

原梯度门槛、生产精度默认和OOM策略均未修改。BF16诊断控制在本案例失败，
不等于生产BF16训练失败；固定FP32重放输入及timestep路径与生产输入处理有差别。

## 固定条件与身份

| 项目 | 本批条件 |
| --- | --- |
| GPU | NVIDIA CMP 90HX，10GiB，SM86；UUID `GPU-806debd9-d254-da57-d515-ccae15b4ac6a` |
| 软件 | 项目Python 3.13环境；PyTorch `2.12.0+cu130` |
| 基座 | `anima-base-v1.0.safetensors`，源权重全部BF16；FP32参考为源值提升，不是原生FP32 checkpoint |
| 输入 | 一个真实缓存样本，latents/noise `[1,16,1,150,112]`，prompt `[1,512,1024]`；4200 image tokens |
| 固定状态 | 同adapter/buffers/输入/RNG重放；sigma=0.5，初始plain LoRA，rank/alpha=4/4 |
| 计算 | torch attention，TF32关闭，full checkpoint，swap20；islands loss scale=1024 |
| 优化器 | 真实AdamW，初始空moments/step，lr=1e-4，betas=.9/.999，eps=1e-8，weight decay=.01 |
| 更新 | 一次disposable optimizer step；unscale后clip，max norm=1，norm type=2；无scheduler |
| 保存 | 捕获张量、delta和证据；不保存训练权重或checkpoint |

来源权重SHA256：
`bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e`。
输入文件SHA256：
`0c2d6f12569596357efd5a1b0b652150f176ef37e2f475bda0bcd533ccc17d78`。
同pre-step参数摘要：
`e53baa61dae3d3ad85f3511bd0c7b4d3abc98d7293471c779a20734aff598bfa`。
完整代码/依赖身份保存在结果目录的`environment.json`与`final-audit.json`。

## 实现与证据契约

- [批次编排器](../../bench/adaptive_runtime/anima_precision_batch.py)复用worker、GPU空闲检查、
  monitor与比较器。一份参考、两个控制，然后按4+4+4分段运行，最多12个不同候选。
- [候选策略](../../bench/adaptive_runtime/anima_precision_batch_policy.py)从四个7-block区间起步，
  按实际误差组合/细分。并集重新跑整网，不继承组成方案的通过结论。
- [更新捕获/比较](../../bench/adaptive_runtime/anima_update_comparison.py)在真实clip/update前后
  保存FP32参数差值，验证空AdamW状态、相同pre-step参数、参数清单和optimizer选项。
  不支持非零moments的同状态更新重放，不能冒充该能力已完成。
- [探针](../../bench/adaptive_runtime/probe_anima_train.py)新增`--capture-update`，仅接受
  一步、sigma=.5、record/replay契约。实际update完成且AdamW step=1才产生完整更新证据。
  update manifest绑定training capture与最终worker result SHA。
- 候选FP16实际安装名单必须精确等于`blocks.<i>.mlp.layer2`目标；精确FP32补集保存到候选表。
  续段校验权重、输入、probe、更新/候选/批次源码、replay/capture/metrics及模型依赖SHA，
  并复核已有证据；本批三段没有改变数值源码。
- 输出预算2GiB在每个worker前后及候选后检查，包括参考与控制；它是检查式预算，
  不是OS硬配额。子捕获另有文件预算。worker限150秒、段限480秒、外层timeout540秒。

自动shortlist当前按通过候选的理论payload减少量、再按归一化数值误差排序，
因此选05和04。时间、allocated/reserved与遥测峰值已记录，尚未用未经预热的单步数字
做生产性能排序。这里的“确认轮”是新增更小子集筛选，没有重复05/04来证明跨运行稳定性。

## 全部数值结果

下表均为相对L2误差的**百分数**；JSON记录原始比例。现有门槛为prediction≤1%、
输入/adapter梯度≤2%、两种梯度cosine≥.999，且必须finite。
实际更新列是额外观测，未新增更新放行门槛，不能用它推翻旧梯度拒绝。

| worker | FP16 MLP2 blocks | prediction % | input grad % | adapter grad % | actual update % | 严格门槛 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| FP32控制 | 无 | 0 | 0.000467 | 0.000210 | 0.002303 | 通过 |
| BF16诊断控制 | 全局BF16路径 | 0.922020 | 97.588632 | 44.067794 | 75.721327 | 拒绝 |
| 01 | 0–6 | 0.015632 | 68.409726 | 25.6395 | 49.3891 | 拒绝 |
| 02 | 7–13 | 0.002903 | 20.0017 | 7.45661 | 18.6705 | 拒绝 |
| 03 | 14–20 | 0.001930 | 0.184345 | 0.086088 | 0.885580 | 通过 |
| **04** | **21–27** | **0.004670** | **0.034954** | **0.030776** | **0.348972** | **通过** |
| **05** | **14–27** | **0.005062** | **0.156350** | **0.081213** | **0.903123** | **通过** |
| 06 | 21–23 | 0.002109 | 0.058181 | 0.032211 | 0.375132 | 通过 |
| 07 | 24–27 | 0.004134 | 0.013005 | 0.016925 | 0.187143 | 通过 |
| 08 | 14–16 | 0.000917 | 0.217868 | 0.086026 | 0.822383 | 通过 |
| 09 | 24–25 | 0.001781 | 0.009618 | 0.011040 | 0.124271 | 通过 |
| 10 | 26–27 | 0.003722 | 0.006801 | 0.011587 | 0.122265 | 通过 |
| 11 | 21 | 0.000863 | 0.024922 | 0.017268 | 0.196555 | 通过 |
| 12 | 22–23 | 0.001929 | 0.021079 | 0.020121 | 0.234327 | 通过 |

原始通过范围为03–12，失败项没有改写或丢弃。更小子集不一定误差更低：例如08的输入
梯度误差高于03/05，不能据此作单调性剪枝或建立“安全层”普遍规律。

05/04的实际delta对照分别保存392个参数张量、5,160,960个元素；global delta余弦为
0.999959218 / 0.999993911。分组相对误差如下，反映整个adapter更新而非仅降精度块：

| 候选 | MLP update % | self-attention update % | cross-attention update % |
| --- | ---: | ---: | ---: |
| 05 | 0.997963 | 1.045856 | 0.449328 |
| 04 | 0.380576 | 0.413773 | 0.162986 |

`other`组无参数元素，不把其零误差作证据。global delta参考L2约0.162914；
05/04的绝对L2、逐组统计及tensor inventory均在`update-comparison.json`。

## 存储与实测资源

源文件共有2,091,068,928个元素，假设全部以FP32表示的值payload为8,364,275,712 bytes。
05选234,881,024个weight元素降至FP16，估计混合payload为7,894,513,664 bytes，减少
469,762,048 bytes（448MiB，5.616%）；04减少224MiB（2.808%）。这里只算源值表示，
不等于实际GPU常驻量、CPU masters、activations、swap传输量或FLOP覆盖比例。

以下为同卡、同swap单次冷态观测，不能宣称稳态性能优劣；单步包含capture与update IO：

| worker | 冷态捕获步 s | allocated GiB | reserved GiB | monitor GPU MiB | RSS树 GiB | worker端到端 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32控制 | 8.034 | 4.654 | 4.838 | 5236 | 18.330 | 59.885 |
| BF16诊断控制 | 3.277 | 2.601 | 2.811 | 3154 | 9.482 | 37.871 |
| 04 | 8.159 | 4.433 | 5.236 | 5646 | 16.321 | 55.309 |
| 05 | 7.875 | 4.401 | 5.330 | 5738 | 14.573 | 52.893 |

05的allocated较低，但reserved与monitor GPU峰值更高；单次RSS下降也不能归因于
精度策略。大部分模型仍为FP32，当前结果不足以满足“低显存且更快的FP16驱动训练”。

参考、两个控制及12个候选共15个worker全部退出0，均完成一步实际更新。
746次遥测的最高GPU占用5738MiB、RSS树峰值18.334GiB、最低host available
74.568GiB、最高温53°C；swap used/in/out均为0，遥测错误0。此聚合包含参考/控制，
不能混用只统计候选的较小峰值。结束后目标GPU为17MiB、0%利用率；未操作显示GPU。

| 分段 | 新候选 | preflight s | 段端到端 s |
| --- | ---: | ---: | ---: |
| 初筛（另含3个参考/控制） | 4 | 13.594 | 393.393 |
| 组合/细分 | 4 | 20.589 | 248.017 |
| 子集确认 | 4 | 25.306 | 256.951 |

分段计时累计898.360秒，约14.97分钟，不含段间等待、开发及最后独立审计。
这些计时不提供每个worker初始化/SHA/纯计算的完整分解，不能反推计算吞吐。
审计时原证据153文件、684,118,802 bytes（约652.43MiB），不含后来写入的审计/人类报告，
低于检查式2GiB预算。

## 命令与验收

首段使用新目录，后两段完全相同参数追加`--continue-run`：

```bash
timeout 540 .venv/bin/python -m bench.adaptive_runtime.anima_precision_batch \
  --weights /path/to/anima-base-v1.0.safetensors \
  --inputs output/runs/anima-adaptive-precision-90hx-20260930-first/inputs.safetensors \
  --output-dir output/runs/<fresh-delivery-batch1> \
  --gpu-uuid GPU-806debd9-d254-da57-d515-ccae15b4ac6a \
  --swap 20 --loss-scale 1024 --worker-timeout 150 --batch-timeout 480 \
  --max-new-candidates 4
```

录制参考前，相关6文件定向回归61 passed（6.76s），Ruff通过。实现的首次独立审查发现
续段缺数值依赖身份、控制组缺输出预算检查两个问题，已在本批录参考前修正并复测。
最终独立Sol验收执行两文件定向测试7 passed（5.78s），并核验全部来源/manifest/result
SHA、实际dtype、空optimizer契约、参数状态、12候选预算及finite更新。
其NumPy FP64复算覆盖两个控制、全部12候选的prediction/input/adapter梯度及实际delta，
包括global与各adapter分组；匹配容差rel=2e-9、abs=2e-11，无新增发现。
该CPU复算核对已保存张量统计，不独立重新执行模型forward/backward。

```bash
timeout 90 .venv/bin/python -m pytest -q \
  tests/test_adaptive_anima_precision_batch.py \
  tests/test_adaptive_anima_update_comparison.py \
  tests/test_adaptive_anima_search.py tests/test_adaptive_anima_probe.py \
  tests/test_adaptive_training_replay.py tests/test_adaptive_training_capture.py
```

全部证据位于结果目录报告
`output/runs/anima-adaptive-precision-90hx-20261001-delivery-batch1/REPORT.md`：
`summary.json`、`candidate-table.json`、`environment.json`、`final-audit.json`，以及每个worker
的result/supervisor/telemetry/log、training capture、实际update capture/comparison。
参考目录另存training replay。数据与失败结果保持原样，未提交或发布本批代码。

## 下一批的边界

1. 只让05/04进入新增两个真实样本、两种bucket、sigma .2/.5/.8与初始/非零adapter状态
   的12案例验证；代表集与保留集分开，不对保留集无限调参。
2. 本批仅交付空AdamW首步更新对照；非零moments/step的状态捕获与重放仍须补齐，
   不能因adapter状态已有replay而声称optimizer状态也已恢复。
3. 固定状态通过后，按收敛计划用真实`train.py`做3–5步受限链路测试，记录有效更新/跳步。
   预热后才比较纯计算与端到端步时、显存/RSS；没有收益就如实标注。
4. Turing kernel、容量与速度仍需实际Turing硬件。20–50步质量评估、生产选择结果桥接、
   与OOM计划联合验收属于后续批次；更新后OOM自动恢复仍未开放。
