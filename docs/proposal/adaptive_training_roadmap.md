# DiT 自适应混合精度与 OOM 恢复阶段开发计划

状态：半活跃提案 / 单进程受限中途恢复已落地，双卡分流至 DDP/PP，自动敏感层精度搜索待定
更新时间：2026-10-04（补充后续落地说明）
适用范围：当前 `dev` 中的实验实现（默认关闭）
当前说明：[实验入口与限制](../experimental/adaptive-runtime.md)、[训练中途 OOM 恢复](../experimental/adaptive-training-recovery.md)
历史证据：[实测与失败记录](../findings/adaptive_runtime_20260921.md)

> **后续落地补注**：单进程、磁盘缓存条件下的受限训练中途 OOM 恢复已通过显式 `adaptive_oom_retry_resume=true` 上线（见 [adaptive-training-recovery.md](../experimental/adaptive-training-recovery.md)）；双卡训练已转入独立 DDP 参考点验收与三族 PP v0 专用会话（见 [distributed-training-card.md](../experimental/distributed-training-card.md)），不再复用单进程 adaptive OOM supervisor。

## 2026-10-01 全局复盘后的执行优先级

用户要求加快推进。当时排序见[自动精度交付收敛计划（已归档）](../../_archive/docs/proposal/adaptive_precision_delivery_reset_20261001.md)：
暂停把token/head根因追踪作为交付前置，优先复用已有训练精度入口与probe，完成有预算的
可部署Linear组合搜索、同优化器状态的更新对照、保留集及真实训练收益验证。
下文各阶段的“下一步”保留历史研究语境；当前主线不再默认继续局部K反向恢复。
旧数值拒绝、门槛和生产默认不变；未开启 `adaptive_oom_retry_resume` 时更新后 OOM 恢复仍默认关闭。

批次1现已完成：3个参考/控制worker与12个不同MLP2候选，10个通过单输入、sigma=.5、
初始LoRA的严格门槛；同空状态AdamW的真实更新已落盘并经独立CPU复算。
候选05/04进入下一批；05理论权重payload减少5.616%，尚无GPU峰值或稳态速度收益证明。
完整范围见[批次1交付报告](../findings/anima_precision_delivery_batch1_90hx_20261001.md)。
随后[批次2](../findings/anima_precision_delivery_batch2_90hx_20261001.md)已完成两个新样本、
两种bucket、三档sigma与独立初始/非零AdamW状态的验证：05/04及FP32控制各12/12通过，
独立FP64复算36格一致；四组train.py各完成5次有效更新、零跳步，最终CPU审计通过。
候选05/04峰值显存仍高于FP32控制，生产BF16对照更省显存；实用收益尚未成立。
下一步是计划身份绑定、选择/拒绝报告与收益决策，当前仍不能计作生产自动精度交付完成。

随后[旧Anima FP16 AMP基线](../findings/anima_legacy_fp16_comparison_90hx_20261001.md)
补测五步有效、零跳步，独立审核通过；scale1024时步2..5均值2.694s、整卡峰3524MiB，
短窗明显优于主体FP32候选。原生AMP与FP32必要保护已实际观察；旧默认scale、严格矩阵、
长期质量及Turing未验收。下一阶段的收益/质量评估应纳入旧FP16基线，仍保留原数值门槛。

其后[旧FP16集中12格验收](../findings/anima_native_fp16_acceptance_90hx_20261001.md)已完成：
全部finite、真实更新、零跳步，独立FP64证据审核通过，但严格0/12通过，input/adapter梯度明显漂移。
05/04默认扩张与逐层/scale追加结束。质量标准已提前登记并绑定SHA，数据/prompt待绑定、quality=not_run；
严格失败不等同已证明训练质量失败，不能以finite覆盖它。执行与协议见
[集中验收入口](../experimental/anima-native-fp16-acceptance.md)。

## 当前执行范围（2026-09-30）

本轮授权更新启动 OOM 恢复、配置与 WebUI：`adaptive_oom_retry_max_swap=0` 表示
自动查询实际模型合法交换上限，正整数表示用户预算；模型结构及 CPU 主机内存只会
进一步收紧有效上限。重试按起始增量指数增长，最后一次可直达有效上限；尝试次数
包含首次训练。Anima、Krea-2 和 Z-Image 的上限按实际模型结构计算，不再使用跨族
固定的 26。此次仅进行 CPU/前端定向验证，不启动 GPU、整模训练或热测，也不改变
精度计划冻结约束。首次 optimizer 更新开始后的 OOM 仍不自动续训。

另有独立授权的 Anima 首阶段 disposable precision CLI 实测，记录于
[CMP 90HX 探针报告](../findings/anima_adaptive_precision_90hx_20260930.md)：FP16/BF16
候选均 finite 完成 3 updates 但梯度门槛失败，FP32 Linear 组仅作为通过的控制对照。
这不表示 production 精度计划解冻、生产 train 已桥接或 OOM 上限策略参与该实验；
OOM max-swap 自动上限更新仍按本节及[实验入口说明](../experimental/adaptive-runtime.md)
所述范围独立维护。

Anima 后续由 Luna 执行分支/scale 诊断矩阵，六个候选仍梯度不达标；缩放有效但不充分，
早期 block 是误差集中处而非已确认源头。证据与后续局部 VJP 定位方向见
[90HX 梯度定位](../findings/anima_adaptive_precision_diagnostics_90hx_20260930.md)。

后续单步固定状态的[局部 VJP/cotangent 对照](../findings/anima_local_vjp_90hx_20260930.md)
已由 Luna 运行；MLP-only 候选仍梯度不达标，局部量化与整网轨迹偏移尚未形成因果分解。
下一步定位 MLP2/GELU，不将 QKV-FP16 局部反事实作为候选实际配置。

MLP2/GELU 捕获及投影拆分已继续完成，见[定位报告](../findings/anima_mlp_path_90hx_20260930.md)。
真实样本 GELU 输入单独变化影响很小，但传入 cotangent 偏移很大；MLP1/MLP2 单独
降精度均失败，不能据此推荐层名单。后续需要前向/反向分离的受控诊断，而不是增加
FP32 层数量或放宽门槛。

同前向 FP32 base-dx [受控反事实](../findings/anima_backward_counterfactual_90hx_20260930.md)
已完成；两种 FP16 prediction 逐位一致，base-dx 干预未改善梯度漂移。后续沿 block
边界定位 cotangent 首次放大位置；此结果只收窄本固定样本解释，不开放生产 FP16。

29个[残差流边界对照](../findings/anima_block_boundaries_90hx_20260930.md)已完成，
FP32控制误差极小；boundary14首次超2%，block10/8两侧观察到明显放大。
下一步固定同一activation/cotangent做单block VJP，对照轨迹与局部计算；不将定位指标作为安全层名单。

block14[完整形状八组VJP](../findings/anima_block14_vjp_90hx_20260930.md)已完成；
同侧捕获重现通过，八格完整梯度CPU复算一致。固定输入/策略只换传入cotangent产生最大局部对照，
下一步沿block15..27的cotangent来源继续固定边界定位，不将三类对照相加成因果贡献。

[15/17/27上游对照](../findings/anima_upstream_vjp_90hx_20260930.md)已完成，24格完整梯度CPU复算一致。
block17的激活变化对照已与cotangent变化接近，而局部policy切换仍小；下一步拆查
block17内部attention/MLP及归一化/残差边界的激活扰动敏感性，尚不开放生产精度方案。

2026-10-01 [block17内部阶段对照](../findings/anima_block17_paths_90hx_20261001.md)已完成，
复用来源捕获，无新训练；固定参考阶段cotangent时self-attention完整VJP差异最大
（0.676%，cross-attention 0.081%、MLP 0.164%）。下一步细分自注意力前置
normalization/modulation、QKV、Q/K normalization、attention和输出投影；
该结果尚不区分具体算子根因，也不开放生产精度方案。

[自注意力两段/投影核心拆分](../findings/anima_attention_split_90hx_20261001.md)随后完成：
归一化固定h换输入对照0.022%--0.040%，换h约1.271%--1.296%；固定du的core
fused-QKV梯度差异1.130%，K分量1.923%，固定cotangent的两个投影VJP变化0。
下一步在显存预算内隔离重放core，校验与现有dp的fidelity后，再拆Q/K normalization
与attention后向；不能据此推荐生产安全层名单或认定最初FP16误差来源。

[隔离QK-norm四格](../findings/anima_qknorm_90hx_20261001.md)现已完成：原core u逐位/
dp fidelity通过，四格由真实weights与NumPy解析式独立验证，显存峰值1500MiB。
换norm输入对照0.116%--0.374%，换attention传回梯度约1.161%--1.319%，K分量
后者2.004%--2.365%。下一步在隔离attention路径做真实Q/K/V激活受控替换，
继续固定du与conditioning；四格不是可加的因果分解，生产精度方案仍未开放。

[post-RoPE QKV八格](../findings/anima_qkv_matrix_90hx_20261001.md)已完成热测与独立验收。
单独换K输入fused对照0.737%--0.947%，大于V/Q轴，且受V背景影响。
下一步优先在固定背景下细查K轨迹与V交互或K差异方向插值；
仍不能认定最初FP16舍入误差来源，不能将输入轴对照当成安全精度名单。

[K差异方向五点/V背景对照](../findings/anima_k_interpolation_90hx_20261001.md)已完成：
固定Q_R/du，参考V端点对照0.945%、候选V0.737%，五点响应非线性；
独立CPU统计审计通过，外层热测timeout124仍保留。
[已有梯度head/token分布](../findings/anima_k_distribution_90hx_20261001.md)随后完成：
两V背景端点约99%平方误差集中于top1% tokens，最大head13/token3343，独立复算通过。
[实际K激活漂移CPU核对](../findings/anima_k_activation_distribution_90hx_20261001.md)完成：
token3343承载77.75%激活误差，head13激活份额仅10.24%；激活top42覆盖约93.1%
fused梯度误差，集合交集13/42，独立NumPy核验通过。
[局部K替换](../findings/anima_k_local_90hx_20261001.md)已完成热测与产物验收：
top42距full响应残差约12%--15%，低误差集合响应小；head13仅10.24%漂移能量却产生
较大响应。下一步在候选K背景检查反向恢复，不将距离比作为贡献或生产精度认证，
不追加长训练或放宽门槛。

本轮 CPU 主回归为 106 passed，关联精度/旧 AUTO/预检回归为 230 passed、1 skipped
（CUDA 条件）；独立复核通过。前端定向测试 23 passed、帮助对话框隔离测试 5 passed，
类型检查与隔离候选构建通过。全量前端测试仍有历史页和打标相关三项失败，未计作通过；
此次未覆盖线上静态资源。正 swap 成功结果必须携带已测主机预算，加载阶段 OOM 可暂未测。

下文历史热测授权和“两线同步”的阶段安排不构成当前执行授权。三模型双卡的
最终目标保留，但未完成、未在本轮验证。

本轮新增前后向误差传播、LoRA 梯度、loss scaling 上下界和非单调 promotion 的
[数值说明及 CPU 反例](../structure/adaptive-precision-numerics.md)。新增 16 项测试
通过，相关回归 118 passed、1 CUDA 项 deselected；产物为
`output/adaptive-runtime-20260926/precision-math-cpu.xml`。尚不能据此生成真实
模型的自动 FP32 名单或宣称训练质量/加速通过。

2026-09-27 继续静态核验共享参数边界：新增 18 项通过测试与 3 项严格预期失败。
实际训练入口会在修改模型前拒绝共享 Parameter；底层 island 安装器缺少同一检查，
本轮仅记录缺口，不改运行时。相关 CPU 回归 136 passed、3 xfailed、1 CUDA 项
deselected；产物为 `output/adaptive-runtime-20260927/precision-aliases-regression-cpu.xml`。
关闭 xfail 的核验明确返回 3 failed，缺口不计作通过；数学约束与复现命令见上述数值说明。

同日继续验证统一 loss scale 的可行域：新增 19 项 CPU 测试通过，证明受控双分支
中统一 scale 可以无解，而定点提升大梯度分支 FP32 后 LoRA/输入梯度可恢复为
FP32 对照；另验证 finite gate 不能发现静默下溢。完整假设和反例见数值说明第 6 节。
回归 155 passed、3 xfailed、1 CUDA 项 deselected，产物为
`output/adaptive-runtime-20260927/precision-scale-feasibility-regression-cpu.xml`。
只增加测试和论证，不生成真实层名单、不扩展 OOM，不代表完成三模型双卡验收。

同日比较门槛审计新增 18 项通过、2 项严格预期失败：固定余弦分母使两个比较入口
误判相同小梯度，缺口未修复；另用精确分数证明全局梯度误差小不保证第一步 Adam
更新接近。范围、假设和现有保护见数值说明第 8 节。最终 CPU 回归 173 passed、
5 xfailed、1 CUDA 项 deselected，产物为
`output/adaptive-runtime-20260927/precision-comparison-math-final-cpu.xml`。
本轮没有修改比较门槛或生产策略。

同日块交换精度边界新增 6 项 CPU 测试通过：确认普通权重的 `bf16` 传输策略保留
原生 FP16/FP32 表示，两种精度布局在实际 CPU fallback 的三轮前后向交换中保持
权重、LoRA 梯度、Parameter 身份和 dtype manifest。数学条件与证据边界见数值说明
第 9.1 节。回归 185 passed、5 xfailed、1 CUDA 项 deselected；产物为
`output/adaptive-runtime-20260927/precision-swap-regression-cpu.xml`。
这不认证 CUDA stream、显存或吞吐，也未扩大 OOM 重试与自动块交换能力。

## 最新进度（2026-09-22）

用户已恢复开发并授权两张 GPU。T10、3080 均可见；T10 的实际 `train.py`
FP16/FP32 三步 smoke 通过，峰值 allocated 4.13GiB，最终 588 个 adapter 张量
与快照逐位一致、392 组 AdamW/scaler/scheduler 进度核验通过。两卡小模型 CUDA
scaler/clip/续步回归分别通过。随后按用户补充请求，3080 完成普通非量化 Krea BF16
三步训练，peak allocated 3.67GiB，588 个 adapter 张量和 392 组 AdamW 状态核验通过。
该卡仍有 Blender 负载，结果仅为链路基线；首版日志审计字段错误已修复并对原产物重审，
原失败记录保留。尚无同卡 BF16/混合/FP32 实际训练性能或质量对照。

硬件默认策略已按用户要求更新：SM80+ 为 BF16，SM70-79 为 FP16/FP32，SM70 以下
为 FP32，但 SM60 精确例外使用 FP16/FP32。独立自动实验入口同步接通三态路由；
训练入口现在也支持 `auto`：在三类已注册 DiT（Anima、Krea-2、Z-Image）上解析为
BF16、FP16/FP32 岛屿或纯 FP32；Krea-2/Z-Image 的 pre-Ampere 自动回退到 torch attention。
这仍是代码级合同接入，不是三族旧架构真机认证，也尚未把自动敏感层校准接入实际训练入口。
本次规则变更的 CPU 回归为 489 passed、2 skipped（CUDA 条件）、25 个既有警告，
35.35 秒；覆盖全部 adaptive 测试及混合精度 resolver、训练兼容、Krea attention 兼容。
XML：`output/adaptive-runtime-20260922/hardware-policy-tests.xml`。修改文件 Ruff 通过；
SM60/61/62/70/75/80 等分界和自动 CLI 路由通过模拟能力测试，未新增 GPU 热测。

P2 使用 torchdata 0.11.0 + Accelerate 实际单批 look-ahead，桶顺序与独立数据 RNG
原型通过跨 epoch 新进程小模型续训对照。尚未接入生产 checkpoint，数据内容/底模
身份与实际多 bucket/scaler 续训仍待验证。更新后 OOM 自动恢复继续关闭。

上述结果只完成 P1/P2 的受限子阶段，不标记任一整个阶段已完成。命令、范围和
证据见[本轮报告](../findings/adaptive_training_20260922.md)。后面的“暂停时基线”
保留为历史快照，以本节和实时源码作为当前状态入口。

## 1. 方向与交付定义

主线是把 FP16/FP32 混合精度用于实际 DiT 训练，同步建设 OOM 自动调参恢复。
不再把不断增加独立探针数量当作主要交付。探针只用于解决明确的训练阻塞，
结果必须回到训练配置、执行契约或验收报告。

最终用户体验：选择模型和训练目标后，系统检测设备能力，在允许的模型族能力范围内，
生成并验证精度候选；不支持原生 BF16 的设备使用经过验证的 FP16/FP32 组合。
冻结已选精度计划后，通过 block swap 等不改变数值策略的手段规划内存；训练 OOM 时，
在可以证明恢复一致性的检查点重新启动，有限调整内存配置，保留训练进度与审计记录。

必须把三类成功分开：

| 成功类型 | 必须证明 | 不能据此推断 |
| --- | --- | --- |
| 可运行 | 完成 forward/backward/有效更新，无非有限值 | 精度足够、训练质量正确 |
| 可恢复 | 恢复状态、数据位置和随机状态契约成立 | 精度策略更好、训练更快 |
| 可推荐 | 数值、短训、质量和稳态资源指标达到明确门槛 | 任意模型、数据或显卡通用 |

不承诺任意显卡都不会 OOM。目标是减少 OOM、正确处理可恢复的 OOM，并在资源不足、
预算耗尽或恢复条件不成立时明确停止，而不是通过偷偷降画质或丢失进度继续运行。

## 2. 暂停时的实现基线

### 2.1 已实现，但仍属实验

- `training_config.py` 提供显式 `adaptive_precision=fp16_fp32`、FP32 Linear glob 名单和初始 loss scale。
- `training_precision.py` 安装冻结普通 Linear 精度域；低精度 Linear 输出回到 FP32 残差流。
- 模型加载、adapter、offloader masters 的顺序已接线，避免后续整体 dtype cast 覆盖精度域。
- 训练使用 Accelerate 原生 FP16 scaler，禁用全局 autocast，保留 unscale-before-clipping。
- 精度名单保存为 `adaptive_precision.json`，缺失或改变名单时拒绝按相同计划恢复。
- CLI supervisor 使用合并后配置启动隔离 worker，结构化 CUDA OOM 才允许进入重试。
- 首次 optimizer 更新前的加载/搬运、forward、backward OOM 可增加 swap 后新进程重试。
- optimizer 更新开始前设置不可逆的进度标志；异常发生在部分更新之后也不会当成未开始。
- 完整更新后保存临时状态目录，验证文件与步数后原子发布；只保留最近一份已发布快照。
- 空数据导致的提前返回不算成功，GradScaler 跳过更新也不计作成功提交。
- worker 有主机内存保护、超时、取消和进程组清理；私有配置文件读取后清理。

当前 FP16/FP32 岛屿实际训练合同已覆盖三族的非量化 plain LoRA、单进程、完整梯度检查点、
数据 worker=0 等受限组合；Krea-2 仍要求 torch attention，Z-Image 要求 torch/sdpa。
默认关闭。Krea NF4、复杂 adapter、compile 和选择性 checkpoint 边界不变；三族长期质量/性能
仍待真机验收。

### 2.2 未完成与不可宣传项

| 项目 | 当前状态 |
| --- | --- |
| 自动选择敏感区域 | 训练入口仍要求显式名单；独立候选搜索不能视为自动训练已完成 |
| 更新后真实数据 OOM 自动续训 | 未开放；`committed_checkpoint=null` |
| 数据游标精确恢复 | 尚无 bucket/sampler/prefetch 完整协议 |
| 长期 scaler overflow 策略 | 当前实验 worker 遇跳步即停止，尚非可持续动态处理 |
| 新训练桥接整模热测 | 尚未验收；已有小模型和固定输入实验不可替代 |
| 长期加速与质量 | 均未证明；已有 Krea T10 对照未显示加速且梯度门槛未通过 |
| 双卡三模型真机覆盖 | 未完成；当前只完成三族入口/合同/重试代码与模拟测试 |
| WebUI 配置面 | `/next/` 已展示三族共享字段和默认值；运行进度展示仍待补齐 |
| 独立安装包 | 尚未交付；优先完成仓内训练契约后再抽公共核心 |

### 2.3 测试与环境快照

- 最近一次扩展回归：449 passed、2 skipped、14 个既有弃用警告，28.21 秒，退出码 0。
- XML：`output/adaptive-runtime-20260921/training-supervisor-tests.xml`。
- 覆盖 adaptive 系列、训练 loop/bootstrap、配置兼容、Krea attention 和分布式梯度同步等。
- 两个跳过项依赖 CUDA；该轮通过 `CUDA_VISIBLE_DEVICES=` 明确使用 CPU。
- 早先真实 CUDA 小模型验证已覆盖 LoRAModule、scaler、clip、adapter/optimizer/scheduler 状态重放。
- 最后调整了私有参数文件创建方式为 `O_EXCL + 0600`；其 Ruff 已通过，但未单独重跑该改动后的测试。
- 暂停时无仍在运行的测试或本任务训练进程。保持暂停，不因本计划自动启动后续工作。
- 最近硬件查询只枚举到 RTX 3080 20GB，约 1.6GiB 已用、88% GPU 利用率；T10 未枚举。
- 历史 T10/3080 实验按 GPU UUID 区分，不使用易漂移的设备序号推断卡身份。

## 3. 架构与不可越过的边界

### 3.1 分离精度决策与内存恢复

```text
硬件 + 模型族 + 训练配置
  -> 能力过滤
  -> 有界精度候选 + 数值验证
  -> 冻结 PrecisionPlan
  -> 内存预检与 MemoryPlan
  -> 实际训练 worker
       -> 有效更新 -> 状态提交
       -> 数值异常 -> 数值策略处理或停止
       -> CUDA OOM -> 恢复资格检查 -> 调整 MemoryPlan -> 新 worker
```

精度变化与 OOM 必须走不同原因码。OOM 不证明某层数值敏感，NaN/Inf 不证明显存不足。
OOM 恢复不得自动把已确认敏感的 FP32 区域降为 FP16。正式更新后不允许静默重选精度名单。
若未来需要重新校准，必须明确创建新的计划版本和恢复边界，而不是原地修改现有计划。

### 3.2 优先复用现有模块

| 责任 | 当前入口 / 拟扩展位置 |
| --- | --- |
| 配置与 fail-closed | `library/training/adaptive_runtime/training_config.py`、共享 compatibility layer |
| 精度安装与身份 | `training_precision.py`、`islands.py` |
| 训练事件边界 | `training_runtime.py`，session/loop 仅保留薄调用 |
| 原子状态存储 | `training_state.py`，复用 Accelerate 与现有 checkpoint hooks |
| 独立进程与资源保护 | `process.py`、`supervisor.py`、`training_worker.py` |
| OOM 策略 | `retry.py`；模型族 swap 上限通过能力契约提供 |
| 数值对照 | 现有 capture/replay/compare 模块，必要时抽取训练可复用部分 |
| 数据恢复 | 新建受限 data-state 模块；避免向 dataset base/loop 堆大段业务 |

不绕过 lazy loading、compile-after-apply、LoRA FP32 参数、attention layout 和 family dispatch。
不将 Krea NF4 的历史 FP16 拒绝路径包装为“实验自动支持”。Turing 没有原生 BF16 算力，
允许 BF16 来源权重并不意味着执行 BF16 算子；必须检查真实运行 dtype 和后端能力。

## 4. 阶段与依赖

| 阶段 | 核心产物 | 依赖 | 阶段结束时允许宣称 |
| --- | --- | --- | --- |
| P0 | 当前训练桥接可复现、限制和事件完整 | 恢复开发指令 | 受限入口完成代码级验证 |
| P1 | 真实数据 FP16/FP32 短训对照 | P0、可用设备/模型 | 指定配置可运行；性能/质量按结果报告 |
| P2 | 数据位置和检查点恢复契约 | P0，可与 P1 同步推进 | 已覆盖范围的数据状态可正确恢复 |
| P3 | 更新后 OOM 自动恢复 | P1 运行基线、P2 | 指定配置真实训练中途 OOM 可恢复 |
| P4 | 有界自动精度选择 | P1 数值基线；不必等待全部 P3 工作 | 特定设备/模型/案例候选通过验证 |
| P5 | 双卡多模型与长期质量性能验收 | P3、P4、各 family 实验适配 | 通过的组合可进入推荐候选 |
| P6 | WebUI、配置产品化与可复用包边界 | 契约稳定、P5 部分通过 | 明确支持矩阵内可发布实验功能 |

执行上每个开发批次同时给出“精度线产物”和“OOM 线产物”，避免恢复工程长期挤占
混合精度训练主线。并行指逻辑任务可独立推进，不授权同时争用 GPU 或绕过主机内存保护。

## 5. P0：训练桥接收口

### 工作项

1. 重跑最后文件权限改动的定向测试，验证文件从创建起即为 0600、拒绝覆盖、异常后清理。
2. 检查 CLI/TOML 合并到 worker 的完整参数类型，确保 JSON 冻结不会丢失有效配置或重新加载旧值。
3. 审计所有提前返回、异常和结束路径，保证没有完整目标步数就不能报告成功。
4. 审计实际 Accelerate 模型清单，确认快照保存 adapter 而非整份冻结 DiT。
5. 补充进程级测试：真正新进程启动、SIGINT/SIGTERM、超时、主机保护、保存期间终止、磁盘写失败。
6. 增加权重/激活 dtype 不变量测试，覆盖 adapter apply、CPU masters、GPU 搬运与 checkpoint 重算。
7. 评估全 FP32 非 Linear 区域与 FP32 输出转换的峰值/吞吐成本，不把省权重显存等同于总显存下降。

### 验收

- 现有 449 项回归保持通过，新权限与进程生命周期测试通过；跳过项说明原因。
- 每类终止只有一个权威终态，取消不会因尾随 worker 结果变成成功或触发重试。
- 失败尝试不能覆盖用户配置、既有输出或前一个已发布快照。
- 实际训练调用点保持薄接线，新增业务放在独立模块。

## 6. P1：先证明实际混合精度训练

### 6.1 测试准备

选择小而真实、缓存完整的数据集，至少包含两个 bucket、两类 token-count、不同文本和 timestep。
不重建用户现有缓存，不修改生产数据集配置。固定底模指纹、adapter 初始化、seed、optimizer、
scheduler、loss 配置与输入清单，所有运行使用独立输出目录。

先做 3-5 个有效更新的链路 smoke；授权后扩展到 20-50 步稳定性检查。
这些步数是建议预算，不是立即执行授权，也不是长训质量证明。

### 6.2 对照矩阵

| 对照 | 目的 |
| --- | --- |
| BF16 来源权重的 FP32 计算参考 | 控制组，不冒充原始 FP32 预训练权重 |
| 显式 FP16 Linear + FP32 残差 | 测量当前最小混合策略 |
| 显式敏感 Linear/子区域 FP32 | 验证候选干预是否实际改善数值 |
| 原有受支持 BF16 配置 | 在支持设备上比较真实训练性能，不强求与 FP32 逐位相同 |

每组记录有效更新数、scaler scale/跳步、loss、未裁剪梯度、裁剪后梯度、optimizer 状态、
峰值 allocated/reserved、主机内存、交换流量、保存耗时、冷启动和热稳态步时。
训练纯计算时间与端到端含 checkpoint 时间分别报告，OOM 失败尝试和校准耗时也计入总成本。

### 6.3 Scaler 策略

当前“跳步即停止”只适合保守联调。后续明确区分数据批次数、optimizer 尝试次数和有效更新数。
允许动态降低 loss scale 时，跳步不得推进 scheduler 或提交有效更新；需制定连续跳步上限和终止原因。
是否重放同一批输入必须与 P2 的数据/RNG 协议一致，不能临时在循环内重跑 forward。
对无法通过缩放避免的前向溢出，回到校准阶段或停止，不能仅无限降低 scale。

### 验收与停止条件

- 用实际训练入口完成计划步数，快照包含有效 adapter/optimizer/scheduler/scaler。
- 误差不达门槛仍保留失败结果；不得把“跑完”升级为可推荐名单。
- 没有稳态速度收益时标记“容量兼容模式”，不得宣称加速成功。
- 设备被占用、主机余量不足或缺底模时暂停对应热测，不降低保护阈值抢跑。

## 7. P2：数据游标与恢复身份协议

### 7.1 必须恢复的状态

| 状态 | 最低要求 |
| --- | --- |
| 模型与训练身份 | family、底模内容指纹、adapter 结构、有效配置签名、精度计划版本 |
| 数据身份 | subset/样本清单、缓存版本与必要内容指纹、bucket 配置和顺序 |
| 数据迭代 | epoch、sampler permutation、已消费位置、largest-bucket-first 状态 |
| 桶内部状态 | 原地 shuffle 后的 bucket_indices 和各 bucket 内容顺序 |
| 随机状态 | Python、NumPy、torch CPU/CUDA、独立 generator；缺少哪项就明确拒绝对应随机功能 |
| 预取状态 | 下一批已物化数据及其随机状态，或经验证可重建的等价描述 |
| 训练状态 | adapter、optimizer、scheduler、scaler、有效更新数、累积边界 |

只有路径、目录存在、文件大小或 mtime 不足以证明底模和数据完全相同。
大权重内容校验需有成本预算和缓存失效策略，不能每次重试重复扫描全部模型。

### 7.2 实现选型

先做两条小型可行性比较：兼容版本的 stateful loader 库，以及项目内显式 sampler/cursor。
优先采用维护良好的现成状态化组件，但必须确认它覆盖本项目的 dataset 原地变更与预取，
不能把“提供 state_dict”直接当作本项目恢复正确。增加依赖需先评估 Python/PyTorch/Accelerate 兼容性。

首版继续限制 worker=0、单进程、accumulation=1、固定数据集，恢复位置只在完整有效更新边界。
worker>0、分布式、跨 epoch 动态数据、stage schedule 和中途梯度累积单独设计，不顺便开放。

### 7.3 验收案例

1. 未中断 N 步与第 K 步保存后新进程恢复，后续 sample ID、bucket、文本变体逐项相同。
2. 比较实际进入训练的 latent/text/noise/timestep 张量及其摘要，而不只比最终 loss。
3. 覆盖跨 bucket、epoch 最后一批、下一 epoch 首批、largest-bucket-first 和空/损坏数据拒绝。
4. CPU 确定性小模型要求 adapter/optimizer/scheduler/scaler 逐位一致。
5. CUDA 对照预先声明确定性条件与容差；不能看到误差后临时放宽标准。
6. 数据或底模变化、精度名单变化、配置不兼容、快照缺文件/损坏时拒绝恢复。

交付独立数据状态 schema、兼容规则、测试及说明。通过前继续保持
`data_cursor_resume_supported=false`，禁止给通用恢复器提供可自动续训路径。

## 8. P3：训练中途 OOM 自动恢复

### 8.1 提交与重启协议

完整有效更新后，写入状态、数据游标、身份清单及校验信息，完成验证再发布。
只有发布并可重新验证的检查点才允许成为 `committed_checkpoint`。
OOM 后丢弃失败进程的全部未提交状态，恢复到最近已提交边界，禁止保存部分 optimizer 更新。

固定精度计划，首版只增加 swap。max swap 由模型族提供，并同时满足 CPU masters 的主机预算。
后续才评估 checkpoint 策略、micro-batch/accumulation；batch 调整即使有效 batch 相等，
也不保证梯度完全一致，必须单独授权和验收。分辨率、文本长度、rank、optimizer、量化不自动改。

### 8.2 故障矩阵

| 故障位置/类型 | 预期行为 |
| --- | --- |
| 首次加载/forward/backward CUDA OOM | 无更新时新进程启动，有限增加 swap |
| 已提交 K 步后 forward/backward CUDA OOM | 从 K 步完整状态恢复，保持精度计划 |
| optimizer 内 OOM | 视作可能部分更新，只允许恢复旧提交点 |
| 状态保存中 OOM/磁盘错误 | 临时状态不可用，旧提交点保持有效；磁盘错误不当 GPU OOM |
| 没有完整提交点、计划耗尽 | 停止并给出原因，不重新从零训练 |
| 主机内存保护、超时、用户取消 | 终止所属进程组，不自动继续尝试 |
| 校验失败、底模或数据变化 | 拒绝恢复，保留诊断证据 |

首版不扩展 optimizer/checkpoint 阶段的自动参数调整；即使恢复状态存在，也只对已授权阶段重试。

### 8.3 真实验证与保存开销

先在小模型进行确定性异常注入，再用受控分配器上限触发真实整模 CUDA OOM。
注入异常只能证明控制流，不能替代真正 CUDA 分配失败。需要同配置不中断对照，
检查恢复后批次序列、有效步数、scaler、scheduler 与最终 adapter。

当前每步保存成本较高。协议通过后再引入提交周期、磁盘预算和保留策略；周期变大意味着
OOM 时回放更多未提交步骤，日志/指标必须标记尝试与回放，不能把重复步骤当新增进度。
原子 rename 只保证可见性；若承诺断电恢复，需额外设计 fsync、清单落盘顺序与崩溃恢复测试。

## 9. P4：自动敏感区域与精度选择

### 9.1 候选生成

硬件仅决定候选集合，不决定数值安全性。默认 SM80+ 先评估 family 允许的 BF16；
SM70-79 及 SM60 精确例外评估 FP16/FP32；其余低于 SM70 的设备默认 FP32。
这是用户指定默认策略，仍需工具链/算子能力过滤，必要时报告没有满足资源与数值门槛的候选。
底模来源 dtype、真实计算 dtype、master 存储 dtype 必须分开记录。

先以有模型语义的子区域生成候选，例如 conditioning/modulation、norm 周边、attention
投影和 MLP。现有执行粒度为普通 Linear；整块 FP32/FP16 是后续 adapter 能力，
必须先明确 norm、RoPE、softmax、量化 Linear 和自定义模块的边界，不能直接全块 `.half()`。

### 9.2 校准闭环

1. 从真实数据抽取有界代表集，覆盖幅值、timestep、token family 和至少一个保留验证集。
2. 固定 adapter、运行输入和 RNG 进行参考重放，分离输入舍入与独立训练轨迹差异。
3. 局部输出/输入 VJP 找候选，再用整网 LoRA 梯度和预测误差验证组合。
4. 按“改善量、GPU/CPU 代价、稳态耗时”排序；组合不假设对 FP32 层数单调改善。
5. 通过固定状态后做独立短训轨迹与保留案例验证，失败时回退或终止。
6. 只有明确版本和验证范围的计划可供实际训练选用，未经验证的 hardware auto 标记为候选。

历史实验的 1% 输出、2% 梯度和 0.999 cosine 只作诊断门槛，不是长期质量标准。
制定推荐门槛时同时看绝对误差、参考梯度量级、重复运行与任务质量，避免微小分母误导。

### 9.3 预算和计划缓存

限制候选数、案例数、GPU 时间、CPU/GPU 峰值、capture 磁盘量和失败重试次数。
自动选择开销必须计入用户首训总时间，不能为一次短训练做无限校准。
缓存键至少包括模型内容身份、family/adapter 结构、精度实现版本、软件/后端版本、
设备能力及验证设备身份、输入分布摘要和训练数值配置。另一张卡上的通过不能直接当作本卡认证。
范围改变时失效或降级为需复核候选；不得把旧缓存静默视作新配置已验证。

## 10. P5：双卡与模型族验收

| 模型族 | RTX 3080 20GB | T10 16GB | 前置条件 |
| --- | --- | --- | --- |
| Krea-2 非量化 | BF16 基线、FP16/FP32 候选、启动/中途恢复 | FP16/FP32、FP32 计算参考、启动/中途恢复 | 优先完成实际训练闭环 |
| Z-Image | 既有 BF16 基线；新增混合路径需独立实验适配 | 实验 FP16/FP32，不冒充现有生产支持 | 保留生产 BF16 限制，验证 Diffusers/attention 合约 |
| Anima | BF16 基线与混合候选 | FP16/FP32 候选 | 用户提供有效完整底模；保留 padding sinks 与 5D latent 契约 |

设备恢复可见后先记录驱动、torch/CUDA、UUID、总/可用显存、主机余量和已有任务。
3080 改装显存容量只用于容量判断，算力和 BF16 能力以实时探测为准。
不修改功耗、风扇、频率，不中止用户现有任务。双卡均可能受 PCIe 和 CPU masters 争用影响，
性能对照默认同卡、同条件顺序运行，不能把跨卡吞吐差当作精度策略收益。

质量验收需要用户确认训练目标、样本集合、评价方式和长训预算。至少记录训练曲线、
固定 seed 预览、保留集指标及人工盲评；存在质量损失时明确收益/代价，不能只展示成功图片。
性能记录冷态、预热后、热稳态和重复运行，注明温度/时钟变化；3080 历史热降频结论仍适用。

每格独立标记：未测、仅控制流通过、真实短训通过、数值通过、恢复通过、质量/性能通过。
不再用一个“支持”勾选覆盖六种不同证据层级。

## 11. P6：配置、WebUI 与可复用组件

### 配置与界面

保留手动模式；自动模式清楚显示实际 dtype 分布、敏感区域数、候选验证范围、swap 值、
显存/主机余量和预计校准成本。展示校准、训练、保存、OOM 恢复与停止原因，用户可取消。
把尝试编号与逻辑训练步分开，worker 的结构化进度转发到当前训练任务；不能只把全部日志
隐藏在 `.adaptive-recovery` 中导致 WebUI 无响应。

后端校验为真相源，前端仅按 family 能力过滤；导入配置、队列、历史回放和普通 CLI 一致。
预览、日志 tracker、定期保存逐项解除当前限制，并验证输出不会在恢复时重复覆盖。

### “两个轮子”的边界

精度选择核心和 OOM 恢复核心可以分别复用，但共享版本化执行/检查点契约。
不要先复制成两个独立仓库再靠字符串配置拼装：先在仓内稳定纯策略与 family adapter 接口，
再决定是否提供独立 Python 包。核心不依赖 WebUI、不硬编码 Krea 层数、不自行加载用户模型。

包化候选接口包括 HardwareCapabilities、PrecisionPlan、MemoryPlan、CheckpointManifest、
TrainingAdapter 与 RecoveryEvent；均为设计方向，尚未新增公共 API。稳定后再补安装方式、
版本兼容、示例、无 CUDA 单元测试和支持矩阵。

## 12. 风险与处理

| 风险 | 处理原则 |
| --- | --- |
| FP32 敏感区太多，吞吐反降 | 同时优化误差和资源；允许结论为仅兼容/无加速 |
| 参考计算或 capture 自身 OOM | 有界参考、独立进程、保留主机保护；不静默降低参考精度 |
| 不同精度组合出现非单调误差 | 实际干预和整网验证，不能按层数推断 |
| 数据恢复看似正确但批次已漂移 | 逐批输入/随机摘要和跨 epoch 对照，先限制功能面 |
| checkpoint 保存掩盖计算收益 | 同时报纯步时和含保存的端到端时间 |
| 仅目录存在就接受恢复 | 身份、清单、完整性、游标资格联合验证 |
| 广泛修改训练热点造成回归 | 新模块承载逻辑，原有入口少量编排，旧模式回归 |
| T10 不可见或模型缺失 | 继续 CPU 协议工作，热测明确待条件满足；不擅自修驱动/下载 |
| 自动重试掩盖持续失败 | 全局预算、无重复候选、明确终态，不无限循环 |

## 13. 下一次恢复开发的具体顺序

1. 用户明确恢复开发后，读取本计划与当前 git diff，确认其他改动，保留所有用户数据和配置。
2. 收口 P0：先验证最后 0600 写入改动，再做私有 worker 生命周期和真实状态 inventory 测试。
3. 精度线推进 P1：设计一份真实缓存短训配置与 FP32/BF16/混合对照矩阵，不直接开长训。
4. OOM 线推进 P2：先做双 bucket、跨 epoch、小模型数据游标复现，确定 loader 方案。
5. 确认可用 GPU 和用户授权的短训预算后运行首个 Krea 实际训练 smoke，输出失败或成功证据。
6. P2 通过后才改 `committed_checkpoint` 与更新后恢复开关，进入 P3。
7. P1 暴露的具体数值问题驱动 P4；不再无目标地增加探针组合。

完成每批次后更新本计划状态、实际改动文件、测试命令与产物路径、仍未通过项；
研究失败结果写 findings，稳定使用说明写 experimental，不把长日志继续追加到根 AGENTS.md。

## 14. 待确认事项

- T10 何时恢复系统可见，是否允许在空闲时使用指定 GPU 做短训。
- Anima 完整底模的有效路径；下载需要单独授权。
- 首批真实训练数据集及可使用的模型/缓存，是否有可接受的代表性验证集。
- 单次短训、总 GPU 时间、主机内存保留量之外的磁盘预算和检查点保留要求。
- 最终优先目标是让 T10 可训练、相同质量下更快，还是尽量提高分辨率/容量；这些目标分别验收。
- 可接受的质量偏差和长训验收预算，以及何时考虑解除 preview/复杂 adapter 等限制。

上述问题不阻止后续获准后进行 CPU 协议开发，但不会被解释为热测或长训授权。
