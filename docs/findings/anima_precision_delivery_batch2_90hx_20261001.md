# Anima 自动精度交付批次2：保留集与五步训练链路

日期：2026-10-01。状态：矩阵、四组真实五步及独立验收完成。设备：NVIDIA CMP 90HX，10GiB，CUDA capability 8.6。本报告只评价两种普通 Linear FP16 islands、同路径 FP32 控制及生产 BF16 五步链路；不是长期质量、Turing 或生产自动选型验收。任务范围按[交付收敛计划（已归档）](../../_archive/docs/proposal/adaptive_precision_delivery_reset_20261001.md)第二批执行。原始证据位于 `output/runs/anima-adaptive-precision-90hx-20261001-delivery-batch2/`、`...-delivery-batch2-inputs/` 和 `...-delivery-batch2-production-prepared/`。

## 结论

- 固定状态矩阵：FP32 同路径控制、候选05、候选04各 12/12 严格通过；独立 CPU/NumPy FP64 对保存张量的 36 格复算通过，194 个 hash 核验，无 binding、数值或 worker 错误。候选05/04可进入短训链路，但不等于可推广的自动计划。
- 实际 `train.py` 五步：四组均 exit 0、五次有限正更新、零跳步，最终 adapter 有限、`ss_steps=5`，840 个保存张量中 560 个相对初始改变。run-local wrapper 捕获了实现的 dtype 名单及逐步 loss/update；普通 final save 本身不写此 manifest，不能把该旁路取证说成生产 logger 已交付。
- 实用收益未成立：同路径 FP32 的外部 GPU 使用峰值 5232 MiB，候选05/04分别 5730/5674 MiB，反而更高。理论源权重 payload 下降 5.616%/2.808% 不等于峰值显存或速度收益。当前不开放生产自动选择，不能称通用低显存 FP16 训练已达成。

## 旧FP16基线补充（2026-10-01）

原四组之外，现已补测已有生产 `mixed_precision=fp16`、`adaptive_precision=off` 路径，
保持原生AMP、FP32 LoRA/residual/final保护，仅用run-local GradScaler初始scale1024覆盖。
同样五次有效更新、零跳步，Luna热测与Sol独立CPU审核通过。步2..5均值2.694s，
整卡GPU峰3524MiB，allocator allocated/reserved峰3.0350/3.1719GiB，RSS峰10.232GiB。
相对05/04短窗步时低约62%、峰值显存低约38%，与BF16的2.624s接近；
FP16峰值仍比BF16高13.53%。该五步实验未测试旧默认scale与严格12格，不继承候选的数值通过结论。
其后[集中12格验收](anima_native_fp16_acceptance_90hx_20261001.md)已完成：全部finite且有效更新，
独立FP64证据自洽，但严格0/12通过；质量标准已预登记，quality仍未运行。
完整五组对照、实际dtype观察及限制见[旧FP16补充报告](anima_legacy_fp16_comparison_90hx_20261001.md)。
旧四组产物、审计与上节结论保留；后续FP16收益/质量评估应纳入这条基线。

## 样本、状态与绑定

输入是两个新增真实缓存样本：representative 为 768×1344、4032 native token，heldout 为 896×1200、4200 native token；Anima text cache 的 cross-attention shape 均为 `[512,1024]`，保留 max padding。图像、caption、latent、text-cache、输入张量 SHA256 和 schema 由 `...-inputs/selection.json` 登记；heldout 不来自批次1已知样本。旧批次1样本只作 warmup，不泄露到保留集结论。矩阵在每个样本上交叉 sigma 0.2/0.5/0.8 与初始/非零 adapter+AdamW moments，合计每方案 12 格；sigma 与状态独立变化。状态 bundle 以 `0→1`、`1→2` 的保存更新链形成；旧 TrainingReplay v1 不恢复 optimizer，不能替代这套实际 AdamW 状态比较。

候选05：blocks 14..27 的 14 个 `mlp.layer2` 为 FP16，其他普通 Linear 为 FP32；候选04：blocks 21..27 的 7 个 `mlp.layer2` 为 FP16，其余 FP32。LoRA/残差维持原边界。所谓 FP32 控制是 BF16 源基座权重提升到 FP32 计算，不是原生 FP32 checkpoint。矩阵各候选与参考使用相同输入、状态、RNG 和后端；保存张量的独立审计结果见 `final-audit.json`，逐格结果见 `validation-table.json`。

## 数值矩阵

以下为两样本合并后的最差相对 L2，JSON与本表均为比例而非百分数。严格门槛为 prediction≤.01、input/adapter-gradient≤.02、两种梯度 cosine≥.999、全部 finite；不因 actual-delta 或短训结果放宽。actual delta 是相同 pre-step 参数、AdamW moments/step、clip、epsilon 与 weight decay 的本步更新比较，额外记录、没有独立接受阈值；其接近不能证明长期收敛。

| 方案 | 严格通过 | prediction max | input grad max | adapter grad max | grad cosine min | actual delta max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 同路径 FP32 | 12/12 | 0 | 0.00000533331 | 0.00000372228 | ≥0.99999999998 | 0.0000430818 |
| 候选05 | 12/12 | 0.00012203017 | 0.00579932900 | 0.00612064106 | 0.99998199307 | 0.03802199406 |
| 候选04 | 12/12 | 0.00011253031 | 0.00073081765 | 0.00055003008 | 0.99999978192 | 0.00614644662 |

逐格更新 delta 的分组最差相对 L2（`validation-table.json`）：

| 方案 | MLP | self attention | cross attention |
| --- | ---: | ---: | ---: |
| 同路径 FP32 | 0.00005219966 | 0.00004570501 | 0.00001450695 |
| 候选05 | 0.04060807041 | 0.04459200031 | 0.02229174797 |
| 候选04 | 0.00704789689 | 0.00690993016 | 0.00263132866 |

矩阵 9 个 worker（含 warmup）结果均 `ok`，telemetry failure 0，原始 matrix source 未漂移。输出预算是经过检查的 4 GiB 空间要求，不是操作系统 quota，也不是未来运行量保证。独立审计 `final-audit.json` 的 `passed=true`、`strict_passes` 六组各 6、`hashes_checked=194`，比只看编排器汇总更强，但仍受两个样本与固定状态范围限制。

## 真实训练链路

CPU-only 的预备 `execution-plan.json`、`config-acceptance.json` 先确认四组配置及源名单；审查指出普通最终 adapter 保存无 manifest、默认 progress 的 `avr_loss` 不是精确单步 loss，因此先用 run-local launcher/evidence sidecar 取证，再启动候选。控制组先跑，随后添加最终BF16保存dtype验收与内部worker监控入口检查，未修改数值实现。控制组的旧wrapper已重建为 `control-wrapper-source/` 并与其manifest SHA逐字节核对；差异见 `instrumentation-revision-review.json`，不冒充四组同一wrapper版本。控制组按新证据规则复验通过，候选及BF16使用当前版本。五步 run 都逐步保存 loss、optimizer、gradient-flow、memory、progress、telemetry 与 realized precision manifest；独立审计使用保存文件和最终 adapter，不把 progress 条数当更新数。

| run | realized Linear dtype | 五步/跳步 | final adapter | GPU 峰值 MiB | 主机 RSS 峰值 B | 步 2..5 均值 s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 候选05 | FP16 14 / FP32 363 | 5 / 0 | 有限，560/840 改变，`ss_steps=5` | 5730 | 15,155,859,456 | 7.0150 |
| 候选04 | FP16 7 / FP32 370 | 5 / 0 | 有限，560/840 改变，`ss_steps=5` | 5674 | 17,030,389,760 | 7.19825 |
| 同路径 FP32 | FP32 377 | 5 / 0 | 有限，560/840 改变，`ss_steps=5` | 5232 | 19,202,187,264 | 7.3775 |
| 生产 BF16 | BF16 377 | 5 / 0 | 有限，560/840 改变，`ss_steps=5`；840 个保存张量均 BF16 | 3104 | 10,337,079,296 | 2.6240 |

候选与控制的 5 个更新范数均为有限正数，loss 也均有限；四组全部通过独立 `production-audit.json` 验收（`verdict=pass`、`counts.pass=4`）。文件SHA不同的初始adapter也经实际张量逐个比较一致。各 run 的外部 GPU 峰值、进程树 RSS、主机 available 与 swap 分开记录，不能用 PyTorch `allocated`/`reserved` 代替整卡峰值。候选05/04/FP32控制/BF16的allocator峰值分别4.4524/4.4837/4.7024/2.6334GiB，reserved峰值5.3223/5.2637/4.8340/2.7617GiB。四组swap峰值0、遥测错误0，峰温53--56°C；BF16主机available最低108,210,176,000B，最后目标GPU回到17MiB、0%利用率。短窗口步2..5包含同步和取证开销，仍不足称热稳态吞吐。05/04逐步loss与FP32控制最大绝对差分别4.16e-6/8.49e-7，BF16为3.28e-4；这些只作观察，没有预注册质量门槛，也不是同精确状态的训练replay。五步只验证连续train/save链路，20--50步及生成质量未做。

## 实现与验证入口

本批新增[状态bundle](../../bench/adaptive_runtime/anima_validation_state.py)、[实际更新捕获与比较](../../bench/adaptive_runtime/anima_validation_updates.py)、[六格重放](../../bench/adaptive_runtime/anima_validation_probe.py)、[编排器](../../bench/adaptive_runtime/anima_precision_validation.py)及[来源核验](../../bench/adaptive_runtime/anima_validation_evidence.py)；probe仅接入两个互斥模式 `--export-validation-states` / `--validation-states`。状态身份绑定family、基座/数值源码、后端、rank/alpha、优化器及scheduler，恢复参数、buffers、AdamW moments/step与CPU/CUDA/Python/NumPy RNG。每格独立恢复step0或step1，实际更新分别到step1或step2，不能串行累积到下一个sigma。

开发CPU回归67项通过；来源绑定修正后的17项定向回归通过，两次计数不相加。独立审核关闭了rebound结果伪造有效更新和丢失replay原件的缺口；编排器验证bundle/replay实际内容。最终Ruff及diff检查通过。运行目录CPU自检另确认零更新、FP32最终保存及绕过监控的worker被拒绝。普通生产模块与默认配置未因本批短训改变。

矩阵命令（第一段省略 `--continue-run`；后段保持相同来源/参数）：

```bash
timeout 540 .venv/bin/python -m bench.adaptive_runtime.anima_precision_validation \
  --weights /path/to/anima-base-v1.0.safetensors --warmup-inputs /path/to/batch1-inputs.safetensors \
  --selection /path/to/batch2-inputs/selection.json --output-dir output/runs/<fresh-batch2> \
  --gpu-uuid <target-gpu-uuid> --swap 20 --loss-scale 1024 --worker-timeout 240 \
  --batch-timeout 480 --max-new-workers 2 --continue-run
```

五步配置与启动器保存在 `...-production-prepared/`，只绑定此次输入与候选身份；不能当可迁移生产配置。完整四组审计可用 `.venv/bin/python <prepared-dir>/production_audit_cpu.py --output <new-audit.json>` 重跑，要求新输出路径，保留旧证据。

## 决策与未完成项

候选05的矩阵误差大于04，且二者在短训外部 GPU 峰值均高于 FP32；目前无证据将 selected plan 作为生产推荐。BF16 对照是原生生产路径，不能用 BF16 显存值证明 FP16 islands 有收益。第三批应实现计划身份绑定、加载时对model/family/adapter/实现/设备后端/配置/验证范围做fail-closed校验，并提供通用训练流程内的选择与拒绝总报告；无收益时直接报告拒绝理由。本批结果目录已保存此次决策，尚未成为通用训练功能。数值自动选择未开放，Turing kernel/容量/速度未测；更新后OOM的 `committed_checkpoint=null` 契约未解决，不能宣称中途恢复。
