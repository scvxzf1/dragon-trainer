# Qwen Image 2.1 阶段 3 开发验收

日期：2026-09-27 至 2026-09-28。状态：阶段 3 开发与本轮验收完成，默认关闭。数学、CPU/真实 GPU 小模型、生产 adapter 保存加载、完整模型确定性续训及 compiled 重算计数通过；三轮 12 组共 400 步（含四组 60 步）正常完成。采用结论：torch 下约 2.2%、Flash 下约 0.6–1.2% 的整步收益不足以支持默认开启，保留为显式实验选项。

## 实现与边界

三个实验参数为 `qwen_fused_projections=off|mlp|qkv|all`、`qwen_saved_projection_blocks` 和 `qwen_projection_budget_mib`，默认分别为 `off/0/0`。后两项必须同时为正且 mode 为 `all`。只支持 Qwen BF16 frozen base、BF16 swap transfer、full checkpoint、plain attention-only LoRA；不叠加 INT8。本轮完整模型包含 torch/Flash 两种 attention，各自完成三轮交替及 60-step 长窗口；范围为同一真实缓存样本、B1/L4107、BF16/full checkpoint/swap24/all-block dynamic compile。

- `fused_mlp.py`：初始化打包 gate/up，复用 out；不增加 MLP LoRA。
- `packed_qkv.py`：一次 frozen QKV 投影，保留三个独立 LoRA 对象、参数名和增量；不把三个 rank 合成一个。源 split 权重不重复注册，安装失败回滚。运行态 packed base 拒绝原地 merge/fuse。
- `segmented_training.py`：选择末 N 个逻辑 block，保留 QKV/gate-up 输出，其余 attention/MLP 后段 checkpoint。未选中块仍 full checkpoint；swap backward hooks 保持原有生命周期，compile 针对计算段。
- `training_blocks.py`：在 adapter apply/load 后、compile 与 CPU master capture 前安装；可预测的选块数、真实 checkpoint 状态、block 类型等拒绝在转换前完成。

预算只计保留投影输出的原始 payload，不是全显存上界。no-swap 模式在安装后再次调用 `enable_gradient_checkpointing()` 会覆盖 wrapper，此生命周期不支持。未修改生产默认配置，无 WebUI 开关；未提交或推送。主工作树原有修改通过三方合并保留。

## 独立验收

| 项目 | 结果与限制 |
| --- | --- |
| CPU 集成回归 | 240 passed；涵盖四个新模块、attention/swap、兼容层、原 LoRA forward/save 等。四新模块 Pyright 0 errors。独立审阅发现两处 Ruff 问题，修复后另一代理复核通过 |
| CPU 数值 | 原生小型 block，FP32、B=2、真实非恒等 RoPE、text/image segments、非全有效 key mask、非零 LoRA，连续两次 SGD；输出 max abs 1.1921e-7、dX 3.5763e-7、LoRA grads 7.4506e-8，atol/rtol 均 2e-5 |
| Adapter 文件兼容 | 实际 safetensors 存盘后严格加载到 fresh split/packed 路径，键与 tensor 值一致；前向最大差 2.3842e-7 |
| 真实 GPU 小模型 | 90HX、4 个原生 block、BF16 slab、swap2、non-reentrant checkpoint、真实 Inductor；末 1/3 块各两步，对同 segmented+compiled 无 swap 参考，输出/dX/全部 24 个非零 LoRA 梯度/更新后参数差为 0；两个 GPU slots 复用。该探针不含真实尺寸和 RoPE |
| 重算计数 | 小模型被选块 packed QKV 与 gate/up 调用各从 2 降为 1；不能代替完整模型 compiled GEMM profile |

验收产物位于仓库 `output/runs/`（本地运行数据，不默认纳入版本控制）：

- `qwen21-stage3-integration-review-20260927/review.md`：独立审阅命令和 240 项结果；保留当时 lint FAIL 原始记录。
- `qwen21-stage3-rope-acceptance-20260927/{accept_cpu.py,results.json,REPORT.md,adapter_roundtrip.safetensors}`：后续数值及 lint 复核。
- `qwen21-stage3-dev-acceptance-20260927/{probe.py,result.json,metadata.json,stdout-stderr.log}`：真实 GPU 小模型独立验收。

## 完整模型 smoke 与恢复执行

90HX 10GB，沿用原数据缓存、seed 1234、BF16、full checkpoint、swap24、all-block dynamic compile、torch attention。阶段 3 为 `all + saved_blocks=1 + budget=512 MiB`。

`qwen21-stage3-fullmodel-smoke-20260927/` 保存配置、源码 hash、完整日志、GPU 遥测：4 步 exit 0，无 OOM，记录 loss finite，peak allocated 5.66 GiB / reserved 7.80 GiB。保存了 LoRA 以及 model/optimizer/scheduler/sampler/RNG state。约 7.8 s 的少量编译后步时**不是速度验收结论**，没有同轮交替基线。

`qwen21-stage3-fullmodel-resume-20260927/` 从 step4 恢复至 step6，exit 0；optimizer 256 个 state 的计数均从 4 到 6，384 adapter 键及形状一致。但这次源训练的 horizon=4、恢复 horizon=6，不能作为严格连续性对照。

## 同 horizon 续训差异

另建 `qwen21-stage3-resume-parity-20260927/`，两条路径总步数都为 6。一条连续跑完并于 step4 保存，另一条从该中途 state 恢复到 step6，配置和 seed 一致。两次 exit 0、scheduler 相同，256 个 adapter 张量都发生有效更新；比较训练 state 的 FP32 权重而非 BF16 导出。

| 对照项 | 结果 |
| --- | --- |
| step5 loss | 两边均 0.9877434968948364 |
| step6 loss | 0.2847360372543335 / 0.28471800684928894，差 1.803e-5 |
| step6 adapter | 256 张量均有差异，max abs 9.3447e-6 |
| 与 step4→6 更新量比较 | down/up 差异 rel-L2 6.28% / 6.03% |
| optimizer | moments 有差异；up exp_avg rel-L2 95.73%，exp_avg_sq 930.62%，不能仅按绝对值小忽略 |

`audit_metrics.json` 是离线审计，不放宽阈值、不把 exit 0 当作严格连续性通过。state 权重/moments 为 FP32，alpha、param groups、计数、scheduler 一致，未发现恢复时显式 BF16 cast 的证据。

### 首次偏离定位

`qwen21-stage3-resume-gradient-20260928/` 再运行相同 horizon 的最小诊断，在第 5 次 optimizer 更新前退出，保留参考与恢复快照。

- 恢复刚完成、以及第 5 次更新前，256 个具名参数、AdamW step/exp_avg/exp_avg_sq 与连续运行逐位一致，param groups 也一致。
- 裁剪后的梯度有 10,606,172 个元素不同，max abs 1.90735e-6、整体 rel-L2 0.248%。
- 因此这次首次偏离在梯度生成或裁剪阶段，不在 checkpoint 读取或 optimizer state 映射。该插桩尚不能区分 backward 与裁剪，也不能单凭它断言是 cuDNN/Inductor 非确定性或阶段 3 错误。
- `qwen21-stage3-resume-deterministic-20260928/` 完成确定性诊断。仅在诊断脚本设置 `torch.use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、`cudnn.benchmark=False`，并在启动前设 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。没有改生产默认值。

确定性两次运行均 exit 0。step4 state 恢复后、step5 更新前的参数和 optimizer moments 逐位一致；新增的裁剪前梯度与裁剪后梯度，所有 256 张量均 finite 且逐位一致（max abs=0）。证据为该目录的 `comparison.json` 和 `deterministic_result.json`，配置、脚本、完整日志和 GPU 遥测同时保留。

这支持将此前差异归入非确定性计算路径，反驳“checkpoint 把 FP32 状态转成 BF16”或“optimizer 映射错位”的猜测。但同时改变了多个确定性设置，不能指定某一个 CUDA/cuDNN 内核为根因；也未证明原始 off 路径是否存在相同差异。两次诊断都停在第 5 次更新前，因此不能冒充确定性完整 6 步权重/optimizer 最终状态对照。

### 后续完整确定性轨迹验收

在固定工作树 `krea2-stage3-acceptance-20260928` 重做相同 horizon=6 的连续训练与 step4→6 恢复。产物根为 `output/runs/qwen21-stage3-acceptance-20260928/`，所有模型源码 SHA-256 进入 `source_sha256.json`。

`queue/det_reference/attempt-003` 和 `queue/det_resumed/attempt-001` 均 exit 0。最终 FP32 模型、optimizer（包括 moments 和 step）、scheduler 全部精确相同；256 个非 alpha 张量从 step4 到 step6 均有实际更新。两边 step5 loss 为 0.9877674579620361，step6 loss 为 0.2847459614276886，LR 也完全一致。检查了实际 resume 配置、成功加载日志、跳过 4 批日志，以及恢复运行只出现 step5/6 的事件，排除“其实从头重新跑 6 步”假通过。

最终独立 CPU 验收再次读取两份真实 state，结论 PASS；快照 7 个相关测试文件为 86 passed。证据见 `final_cpu_review.md`、`queue/comparison.json`、`queue/comparison_with_resume_evidence.json`。这是完整确定性多步恢复验收，补齐上一节首步探针的范围限制；仍不声称非确定性默认运行位级一致。

队列最初两次启动失败属于验收脚本/进程托管问题（脱离执行会话、脚本 `queue.py` 遮蔽标准库），原日志保留；修改为 `acceptance_queue.py` 后实际训练运行正常，不把这些失败记作模型训练异常。

## 首轮性能消融与后续交替验收

固定快照、同一缓存/seed、BF16/full checkpoint/torch/all-block dynamic compile，不用确定性诊断开关。每组 20 步，第 6–20 步的相邻 progress 事件差统计稳态，不包含首次编译。顺序为 off/swap24、MLP/swap24、all/swap24、all+saved1/swap24、all+saved4/swap24、all/swap21、off/swap24 复测。

首组 off/swap24 中位数 7.935 s/step，前后五步窗口漂移 −0.063%，峰值 allocated 5.4438 GiB、reserved 7.8047 GiB；稳态平均时钟 1714.4 MHz，平均温度 51.22°C。MLP/swap24 中位数 7.803 s/step（首轮缩短 1.66%），漂移 −0.161%，峰值 allocated 5.5378 GiB、reserved 7.8047 GiB，平均时钟 1714.6 MHz、温度 51.25°C。温频接近，但单轮小幅差异未达到采用门槛。后续结果见下表；不能据此给出持续收益或默认建议。

完成的单轮结果如下；交替长测另行执行：

| 配置 | 步时中位数（s） | 相对 off/swap24 缩短 | peak allocated（GiB） |
| --- | --- | --- | --- |
| off / swap24 | 7.935 | 基线 | 5.44 |
| MLP / swap24 | 7.803 | 1.66% | 5.54 |
| MLP+QKV / swap24 | 7.824 | 1.40% | 5.54 |
| MLP+QKV+saved1 / swap24 | 7.817 | 1.49% | 5.66 |
| MLP+QKV+saved4 / swap24 | 7.778 | 1.98% | 7.26 |
| MLP+QKV / swap21 | 7.425 | 6.43%，包含换块变化 | 6.76 |
| off / swap24 末尾复测 | 7.958 | 首尾基线相差约 0.3% | 5.44 |
| off / swap21 | 7.556 | 与 all/swap21 同换块数时，融合仅快约 1.73% | — |

保留 4 块相对不保留投影的融合路径仅快约 0.59%，却增加约 1.73 GiB peak allocated，暂不支持这种显存分配。swap21 少交换三块；补做的 off/swap21 为 7.556 s，对比 all/swap21 的 7.425 s，融合单轮收益约 1.73%，不能把跨 swap 配置的 6.43% 都归给融合。它的 peak reserved 为 9.23 GiB，和稳态驱动显存不是同一统计量，也不能据此保证更长文本/更大 batch 安全。以上是筛选数据，未达到独立、重复、长窗口的推荐条件。

主统计采用 `queue/aligned_results.json`。旧 `summary.json` 的 GPU 窗口使用了错误的启动时间原点，**不作为遥测依据**；新增 observer 记录每条已 flush progress 的到达区间，将真实 progress 时钟原点限制在上下界，再取交集。首组对齐不确定区间约 9 ms；稳态遥测仅使用完全落在窗口内的一秒采样点。每组源码起止均实际核对全部 552 个 Python 文件 hash。独立审阅确认该方法及首组结果，见 `final_cpu_review.md`。步时是实际相邻事件间隔，包含少量每步收尾/日志开销，不是纯 kernel 时间。

## 传输证据口径

独立审阅完整 20-step `block_swap.jsonl`：swap24 每组均 960 次 H2D event（forward/backward 各 480）；swap21 为 840 次（各 420），减少 12.5%。按配置中每块 BF16 payload 436,208,128 bytes 推算，整次运行传输量为 390 GiB 与 341.25 GiB；日志没有实际字节计数，因此这是估算。

swap24 的 off/MLP/all/saved1/saved4 及末尾基线传输次数相同，不能声称融合或保留投影减少换块量。异步 event 时长不能直接相加换算成墙钟耗时占比；`gpu_wait_ms=0` 是默认关闭计时的占位值，不能解释成零等待。日志 `step` 是 prepare/profile 计数，不能直接套用 optimizer 稳态窗口。

## Flash 小模型独立验收

`flash_correctness/attempt-001` exit 0，独立报告为 `flash_correctness/independent_review.md`。两步非零 FP32 LoRA、BF16 base/autocast、小型原生 block、B2、非恒等 RoPE、多段和非全有效 mask，比较 split Flash 与 packed/segmented Flash，以及 compiled 无 swap 与 slab swap2。

200 项比较全部通过事前门槛（逐元素 atol=0.0625/rtol=0.05，同时 rel-L2<=0.05、cos>=0.995）。最大绝对误差 6.103515625e-5、最大 rel-L2 0.008348166、最低 cosine 0.99996555；记录 128 次真实 FA2 varlen 调用，具有 Inductor 与真实 slab swap 证据。safetensors 键和值精确往返通过。

限制：两条路径共享 Flash mask/RoPE 语义，不能作为独立语义 oracle；此探针没有用标准 LoRA loader 加载到 fresh split 模型后做前向。后续生产 loader 的 CPU 集成补验见下一节；完整 Flash 模型三轮/长测见后文，二者各自限定证据范围。

## 生产 adapter writer/reader 补验

前面的裸 safetensors 和自定义加载往返不能代替生产 loader。随后新增 `production_loader/accept_cpu.py`，调用真实 `create_network/apply_to`、`LoRANetwork.save_weights/load_weights`，无 mock，使用冻结源码、CPU FP32 原生 tiny Qwen 两 block、8 个 attention LoRA、16 个非零可训张量。

packed all + final-block segmented 状态完成一次 SGD 更新后保存，fresh split 和 fresh packed+segmented 分别从同一 base 独立构建并通过标准 reader 加载。24 个标准 split 键及值精确一致，missing/unexpected 均为空；两个路径输出、dX 和全部 LoRA 梯度共 36 项比较通过原定 `atol=rtol=2e-5`。fresh split 输出/dX/LoRA 梯度 max abs 分别 1.1921e-7 / 7.1526e-7 / 7.4506e-8，fresh packed 重载误差为 0。

另一名代理在 `production_loader/independent/` 独立重跑，核对真实文件 metadata/hash、相同 base、全部 16 参数实际更新、冻结 base 无梯度且不变；额外实测 full checkpoint 与两个 segmented tail 均发生重计算。相关 26 项测试及 Ruff/Pyright 通过，结论见 `production_loader/independent_review.md`。这关闭本阶段 plain Qwen adapter 生产保存/加载互换的 CPU 集成缺口，不外推 inference factory、其他 adapter、GPU 加载或训练状态 resume；完整模型 torch 确定性 resume 证据仍见前节。

## 完整 compiled 路径重算复核

两份 CPU profiler 原始 `events.json` 以 `inconclusive` / exit 2 保留。独立离线复核发现原分析器只按 CPU 祖先寻找 `probe.backward`，漏掉 autograd 工作线程；此外 QKV 前向与 MLP down 的 dX 形状相同，不能只按 shape 计数。

`profile/reanalyze_projection_trace.py` 按同进程完整时间窗归属跨线程事件，结合 RHS stride、真实 `CompiledFunction` / `CompiledFunctionBackward` 调用范围区分重算与 dX。`profile/trace_review.md` 保存输入 SHA-256、事件索引和调用上下文，结果如下：

| 实际 dispatch | saved0 | saved4 |
| --- | --- | --- |
| Forward QKV / gate-up | 32 / 32 | 32 / 32 |
| Backward QKV / gate-up 重算 | 32 / 32 | 28 / 28 |
| QKV / gate-up dX | 32 / 32 | 32 / 32 |
| 所有 aten mm/addmm | 1416 | 1384 |

总差值 32 由 QKV 4、gate/up 4、两类 rank16 GEMM 各 12 闭合；无缺失 shape 或未归属 GEMM。分析器自检、Ruff、Pyright 通过，相关 CPU 测试 7 passed。证据证明实际 compiled GEMM dispatch 减少；trace 不含 CUDA kernel/runtime 事件，不能据此宣称硬件 kernel 数或 GPU 耗时占比。

## 三轮与长窗口队列

`final_queue.py --run` 于 2026-09-28 01:30（UTC+8）启动，固定冻结源码及 swap24，比较 torch/Flash × off/all+saved4（budget1536 MiB）。前两轮每组 20 步，第二轮反转顺序；第三轮每组 60 步，共 12 组 400 步。保留每组配置、命令、日志、遥测、源码起止 hash 和退出码，任何失败即停。该队列 12/12 均 exit 0，完成 400 步；最终四组各完整执行 60 步并保存，队列终态为 completed，训练与监控进程均已退出。

长窗口统计使用 `final_results.py`，覆盖第 6–60 步、6–20 / 21–60 子窗口、首末五步及对齐遥测；旧 `summary.json` 仅统计第 6–20 步且遥测原点有误，不适合作最终长测报告。分析器独立审阅 PASS，并以原始日志重算及合成 60-step 完整 `inspect()` 验证口径，见 `final_analysis_review.md`。合成数据不代替真机证据。源码冻结目录不随主仓文档更新改变。

最终 12/12 组配置、源码和 observer 检查通过，全部 step loss 有限。下面统一采用预热后的**平均步时**；前面的消融表采用中位数，不混用。

| 轮次 / 总步数 | torch off | torch all+saved4 | 阶段 3 缩短 | Flash off | Flash all+saved4 | 阶段 3 缩短 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 20 | 7.9611 s | 7.7838 s | 2.23% | 7.3733 s | 7.2836 s | 1.22% |
| 2 / 20（逆序） | 7.9340 s | 7.7580 s | 2.22% | 7.3453 s | 7.3043 s | 0.56% |
| 3 / 60 | 7.9347 s | 7.7569 s | 2.24% | 7.3658 s | 7.2925 s | 1.00% |

第三轮第 21–60 步：torch off/all+saved4 平均 7.93475 / 7.75725 s（缩短 2.24%），Flash 为 7.364525 / 7.28235 s（缩短 1.12%）。四组长测首末五步均值漂移分别为 −0.071%、+0.015%、−0.035%、+0.005%，未出现持续步时恶化。全轮稳态温度均值约 50–51°C，SM 平均时钟约 1690–1730 MHz；记录实测频率而非声称锁频。

| 配置 | peak allocated GiB | peak reserved GiB | 稳态驱动显存上限 MiB |
| --- | ---: | ---: | ---: |
| torch off | 5.4438 | 7.8047 | 8286 |
| torch all+saved4 | 7.2638 | 7.8047 | 8286 |
| Flash off | 5.5065 | 7.5898 | 8068 |
| Flash all+saved4 | 7.2638 | 7.5898 | 8068 |

saved4 相对 off 额外占用约 1.76–1.82 GiB allocated；allocator reserved/驱动显存相同不表示该激活没有成本。不能将本次 B1/L4107 的显存通过外推到更长文本、更大 batch 或其他模型。

同轮同 stage3 设置的后端对照：关闭阶段 3 时 Flash 三轮比 torch 缩短 7.38% / 7.42% / 7.17%；all+saved4 时为 6.43% / 5.85% / 5.99%。这是本机本配置的持续收益，不能宣称 Flash 普遍更快，也不能用本轮消除旧慢测的历史环境差异。

`checkpoints_final.json` 审计全部 12 份真实 safetensors，每份 384 键（128 alpha、256 非 alpha）、down/up 各 128，BF16 非 alpha 参数全部有限且各 tensor 非零，family/输出名/键结构一致。性能运行没有开启逐参数 `debug_finite_checks`，因此不声称逐步直接检查了所有梯度；梯度正确性由 CPU/真实 BF16 小模型、生产 loader 和确定性恢复对照共同约束，长测证据是完整执行、有限 loss、有效最终 adapter 及无 OOM/异常退出。模型质量收敛不在单样本性能实验的证明范围内。

`delivery_source_check.md` 确认四个新模块及八个接线文件与冻结验收源码逐字节一致。源文件、配置、日志及结果均保留在本地 artifact 根；这些运行数据不默认加入 Git。

## 验收结论与默认建议

1. **实现目标已验证**：冻结投影融合保持独立 LoRA 参数及磁盘 split keys；保存投影确实减少选中四块的 QKV/gate-up 重算 dispatch，输出、输入梯度、LoRA 梯度、生产保存加载和确定性续训已有独立证据。
2. **默认采用门槛未达到**：torch 的稳定收益约 2.2%，Flash 的收益约 0.6–1.2%，同时增加约 1.8 GiB allocated。保持 `qwen_fused_projections=off`、保存块数/预算为 0；实验实现保留 opt-in，不继续为这条路径增加复杂优化。
3. **本机性能选择**：90HX 在本次 BF16/full checkpoint/swap24/compile 条件下可显式选择 Flash，关闭阶段 3 即获得跨三轮约 7.2–7.4% 的耗时缩短。torch 默认回退策略保持不变；没有改发布默认值。
4. **后续研究方向而非本轮欠项**：若继续投入，应优先对比同显存预算用于增加常驻权重 slots 的价值，再考虑大矩阵吞吐；swap21 单轮结果不是普适显存建议。旧慢测根因、其他 shape/hardware 和长期训练质量需各自实验，不能由这次结果外推。

开发、机制、数值、文件兼容、状态恢复与三轮性能验收分别完成；性能采用结论为“不推荐默认开启阶段 3”，不是宣称达到了约 5% 目标。无需为了获得正向推荐继续增加改动。

关联：[开发设计与数学契约（已归档）](../../_archive/docs/proposal/qwen_image_2_1_training_block_stage3_20260927.md)、[实验使用说明](../experimental/qwen_image_2_1_fused_projections.md)。

最终独立完成审阅见本地产物 `final_completion_review.md`：12 组结果全量只读重算一致、60-step Flash 文件抽查通过、46 项 CPU 测试及 Ruff/Pyright 通过，无阻塞项。结论 PASS，完成本阶段验收；保持实验默认 off/0/0。
