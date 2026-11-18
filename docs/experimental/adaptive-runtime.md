# DiT 自适应精度与 OOM 恢复

状态：实验，正在实现；不改变现有训练默认行为。

## 2026-10-04 中途恢复验收

显式 `adaptive_oom_retry_resume=true` 已接通完整状态、sampler/bucket/预取恢复和
恢复后设备布局，并完成 Anima 90HX 三步真实 CUDA OOM / 连续训练对照。
默认关闭；下文历史记录中的“首次更新后停止”仍描述未启用该开关的启动重试。
配置、受限范围和逐步快照成本见 [中途恢复说明](adaptive-training-recovery.md)，
证据见 [恢复与独立预览交付](../findings/anima_committed_oom_preview_delivery_20261004.md)。
预览解码的 GPU tiled / CPU 回退独立于训练 supervisor，见
[预览内存策略](../configuration/sample-preview-memory.md)。

## 2026-10-02 原生 FP16 启动 OOM 回退

已将原生 AMP 接入受限新进程重试，保持精度、来源文件和后端不变，只增加交换块。
仅首次 optimizer 开始前的明确加载/前向/反向 CUDA OOM 可重试，首次更新开始后
停止；输入续训和自动交换不开放。正常周期保存、日志和训练后采样保留。
本轮限静态实现与 CPU 验证，未启动 GPU 或真实 OOM。
配置与完整边界见[原生入口](anima-native-fp16.md#启动-oom-回退)，
核验依据见[静态验收](../findings/anima_native_fp16_startup_oom_static_20261002.md)。

## 2026-10-02 原生 FP16 入口

Anima 新增 `adaptive_precision="native_fp16"`，采用原生 AMP 和可配置初始
scaler。`auto` 的 Anima FP16 候选进入此路径，安培继续优先 BF16；
旧显式/冻结的 FP16/FP32 岛屿合同保留。普通训练输出新增精度报告。
配置、支持范围和报告语义见[原生 FP16 使用说明](anima-native-fp16.md)。
此入口基于[200步 A/B 观察](../findings/anima_fp16_bf16_ab200_90hx_20261002.md)，
不改判旧严格矩阵；该入口首批当时未开放 native OOM 重试，新增启动期合同见上节。
未显式开启 `adaptive_oom_retry_resume=true` 时，首次更新后的中途自动恢复默认保持关闭（受限显式中途恢复见上文 2026-10-04 节与 [中途恢复说明](adaptive-training-recovery.md)）。

## 2026-10-01 旧FP16集中验收

旧native FP16 AMP、FP32必要保护、scale1024已完成固定12格同状态验收：
全部finite、真实有效更新、零跳步，独立NumPy FP64证据审核通过，但原严格数值0/12通过。
prediction最差relL2 .00132449在门槛内，input/adapter梯度最差2.08984/.653535均超标。
结束05/04默认扩张，不追加层或scale搜索；当前不作为通过原严格标准的自动计划。
质量标准已另行预登记并绑定SHA，数据/prompt待绑定，quality仍not_run。
完整证据见[集中验收报告](../findings/anima_native_fp16_acceptance_90hx_20261001.md)，
执行与质量门槛见[实验入口](anima-native-fp16-acceptance.md)。

## 2026-10-01 Linear精度交付批次2

`bench.adaptive_runtime.anima_precision_validation`对冻结的05/04候选执行两样本、
两种native bucket、sigma .2/.5/.8与独立初始/非零adapter和AdamW状态的12格验证。
新增StateBundle恢复实际optimizer moments/step；旧TrainingReplay v1仍只负责
adapter、输入及RNG重放。实际更新比较保留原梯度门槛，未增加更新放行阈值。

90HX矩阵已完成：05/04及同路径FP32控制各12/12通过，独立NumPy FP64复算36格一致。
随后四组实际train.py各完成5次有效更新、零跳步，独立CPU审计全部通过。普通最终
adapter不写精度manifest，本批通过运行目录内的取证wrapper记录实际dtype、精确loss、
更新和资源；此wrapper未成为通用生产入口。
05/04峰值显存5730/5674MiB，高于FP32控制5232MiB；生产BF16为3104MiB。
仅短步链路通过，未验证实用显存收益、稳态速度、质量或Turing，不开放生产自动选择。
完整命令、实现、证据与边界见[批次2报告](../findings/anima_precision_delivery_batch2_90hx_20261001.md)。

随后已补测[旧Anima FP16 AMP基线](../findings/anima_legacy_fp16_comparison_90hx_20261001.md)：
`mixed_precision=fp16`、`adaptive_precision=off`，原生AMP及FP32残差/最终投影保护，
仅覆盖初始GradScaler为1024。五步有效、零跳步，独立审核通过；步2..5均值2.694s、
整卡峰3524MiB，短窗优于主体FP32候选。旧默认scale、严格矩阵、长期质量及Turing未验证；
该取证wrapper与新CPU汇总均留在运行目录，未成为生产自动选择或通用logger。

## 2026-10-01 Linear精度交付批次1

有界筛选入口为 `bench.adaptive_runtime.anima_precision_batch`，复用现有probe、monitor、
capture/replay与梯度比较器；新增真实AdamW更新delta捕获。一个参考、全FP32 Linear
控制、BF16诊断控制只运行一次，再按实际整网误差组合/细分MLP2区间，最多12个不同候选。
每段最多4个新候选；`--continue-run`核验来源权重/输入/数值源码/既有证据SHA，不能跨版本续接。

90HX实测已完成，10/12候选通过固定案例的原严格门槛，05（MLP2 blocks14..27）与04
（21..27）进入shortlist。两个控制组与全部候选的梯度及实际更新已由独立CPU NumPy复算。
这是单缓存输入、sigma=.5、初始LoRA、空AdamW状态的单步筛选；更新指标未新增放行门槛。
05理论源权重payload减少5.616%，GPU峰值高于FP32控制，未证明实用性能收益。
完整结果与命令见[批次1报告](../findings/anima_precision_delivery_batch1_90hx_20261001.md)。
该shortlist的保留集与受限真实训练现已完成，见上节；仍不直接用于生产自动选择。

```bash
timeout 540 .venv/bin/python -m bench.adaptive_runtime.anima_precision_batch \
  --weights /path/to/anima-base-v1.0.safetensors \
  --inputs /path/to/inputs.safetensors \
  --output-dir output/runs/<fresh-batch1> \
  --gpu-uuid <target-gpu-uuid> --swap 20 --loss-scale 1024 \
  --worker-timeout 150 --batch-timeout 480 --max-new-candidates 4
```

后两段使用相同参数追加 `--continue-run`。整批输出在每个worker前后和候选完成后检查
2GiB预算，该检查不是OS硬配额。冷态单步计时包含capture/update IO，不能当稳态吞吐。
该批次的BF16诊断控制要求Ampere及以上；Turing测试需另立硬件流程。

## 历史证据与边界

精度请求与 OOM 重试的静态不变量证明见
[2026-09-24 静态契约论证](../findings/adaptive_precision_oom_static_proof_20260924.md)。
动态模型/主机上限和指数重试的条件证明与 CPU 枚举见
[2026-09-30 数学补充](../findings/adaptive_oom_bounds_math_20260930.md)。

2026-09-30 的 OOM 配置与 WebUI 契约批次只做 CPU/前端验证，不启动 GPU 或真实训练。
当时首次 optimizer 更新开始后的 OOM 尚未开放自动续训（后续显式开关 `adaptive_oom_retry_resume=true` 的单进程受限中途恢复见 [中途恢复说明](adaptive-training-recovery.md)）。前后向误差传播、
FP32 LoRA 和 loss scaling 的数值边界见
[混合精度数值论证](../structure/adaptive-precision-numerics.md)。CPU 反例不构成
真实模型数值认证，也不代表恢复双卡热测授权。

2026-09-30 至 2026-10-01 期间在 CMP 90HX 上完成了一组递进式 Anima FP16 梯度偏差归因诊断（全部仅作误差源定位，不放宽生产数值门槛或开放自动精度方案）：

- **整网与分支级初筛**：首阶段探针中 FP16-islands / BF16 虽完成 3 步有限更新，但梯度对照超限，仅全 FP32 Linear 控制组通过（[Anima 90HX 精度探针报告](../findings/anima_adaptive_precision_90hx_20260930.md)）；后续按分支与 loss scale 定位确认单独切 FP16 或仅提高 scale 均不足以消除偏差（[梯度定位报告](../findings/anima_adaptive_precision_diagnostics_90hx_20260930.md)），局部 VJP、MLP2/GELU 拆分与同前向 FP32 base-dx 反事实同样未达整网门槛（[局部 VJP 报告](../findings/anima_local_vjp_90hx_20260930.md)、[MLP2/GELU 报告](../findings/anima_mlp_path_90hx_20260930.md)、[反事实报告](../findings/anima_backward_counterfactual_90hx_20260930.md)）。
- **残差边界与单块 VJP 定位**：29 个残差边界采样定位到反向首次超 2% 出现在 boundary 14（[边界定位报告](../findings/anima_block_boundaries_90hx_20260930.md)）；block 14 完整 4200-token VJP 与上游 15/17/27 多块矩阵表明上游传入的 cotangent 差异占主导（[完整形状VJP报告](../findings/anima_block14_vjp_90hx_20260930.md)、[上游VJP报告](../findings/anima_upstream_vjp_90hx_20260930.md)），block 17 内部三分支拆查则显示 self-attention 差异（0.676%）显著高于 cross-attention（0.081%）与 MLP（0.164%）（[阶段VJP报告](../findings/anima_block17_paths_90hx_20261001.md)）。
- **自注意力 QK-norm 与 K 尖峰 token 细化**：`bench.adaptive_runtime.diagnose_anima_self_attention` 将自注意力拆为归一化段与投影/注意力核，定位主要放大来自注意力反向传回 K 分量的梯度（[自注意力拆分报告](../findings/anima_attention_split_90hx_20261001.md)）；进一步隔离 QK-norm、post-RoPE Q/K/V 八格替换、K 方向五点插值及 token/head 分布分析表明，超过 98.6% 的 fused 梯度误差能量集中在 top 1%（42 个）高幅值 K token 上（[隔离QK-norm报告](../findings/anima_qknorm_90hx_20261001.md)、[八格报告](../findings/anima_qkv_matrix_90hx_20261001.md)、[K插值报告](../findings/anima_k_interpolation_90hx_20261001.md)、[分布报告](../findings/anima_k_distribution_90hx_20261001.md)、[激活与梯度重合报告](../findings/anima_k_activation_distribution_90hx_20261001.md)、[局部替换报告](../findings/anima_k_local_90hx_20261001.md)）。

代表性复用分析命令（零新增训练步，硬限文件/批次字节预算）：

```bash
timeout 360 .venv/bin/python -m bench.adaptive_runtime.diagnose_anima_self_attention \
  --source-run output/runs/<completed-upstream-vjp> \
  --stage-run output/runs/<completed-block17-paths> \
  --weights /path/to/anima-base-v1.0.safetensors \
  --output-dir output/runs/<fresh-self-attention-analysis> \
  --gpu-uuid <target-gpu-uuid> --analysis-timeout 300
```

开发已于 2026-09-22 恢复，阶段依赖、剩余缺口与验收标准见
[阶段开发计划](../proposal/adaptive_training_roadmap.md)。T10 实际训练三步 smoke
与数据游标原型的新进程对照见[最新报告](../findings/adaptive_training_20260922.md)；
默认配置下更新后 OOM 仍不自动续训，显式开启 `adaptive_oom_retry_resume=true` 的单进程受限中途恢复见 [中途恢复说明](adaptive-training-recovery.md)。

## 交付目标

- 自动检测硬件，为 BF16、FP16/FP32 混合精度或 FP32 生成候选。
- 使用输出与梯度误差校准敏感区域，联合已有自动块交换规划显存。
- CUDA OOM 后在独立进程中有限重试，保留用户配置和完整恢复语义。
- Anima、Z-Image、Krea-2 分别在 RTX 3080 20GB 与 Tesla T10 16GB 热测试。
- 留存命令、环境、实际配置、数值误差、训练步时、显存和失败证据。

## 当前实现边界

### 硬件初选规则

2026-09-22 按用户指定策略更新。以 CUDA compute capability `(major, minor)` 为准，
不是 CUDA 软件版本或显存容量：

| SM | 默认候选 | 示例 |
| --- | --- | --- |
| SM80 及以上 | BF16 | RTX 3080，SM86 |
| SM70-79 | FP16/FP32 混合 | V100 SM70、T10 SM75 |
| SM70 以下，排除 SM60 | FP32 | SM61、SM62、SM5x |
| SM60，精确例外 | FP16/FP32 混合 | P100；不包含 SM61/62 |

规则统一由 `precision.py::preferred_candidate()` 提供。独立 `search_dit`、
`recover_krea` / `recover_z_image` 的 `--precision auto` 将三档分别映射为现有 worker 的
`bf16`、`fp16-islands`、`fp32-reference`；也可显式选择这些模式。FP32 不使用 FP16 loss scaler
或局部 FP32 名单，OOM 调整仍保持选定精度不变。这里的 FP32 是同一来源权重提升后的计算，
不代表获取了原始 FP32 预训练权重，也不绕过 Krea NF4 限制。

局部 capture/block 校准同样遵守三档策略。FP32 默认不再试探 FP16，而只检查 FP32 参考
有限值，报告标注 `low_precision_tested=false`、`validation_scope=fp32_reference_finiteness_only`。
这不是低精度校准通过。显式选择仍保留，硬件门槛、模型/后端限制和数值门槛继续独立生效。

这是默认策略，不是所有旧卡已验证可训练：SM60/SM70 及更老架构本轮仅覆盖模拟能力的
路由测试，实际运行仍取决于 PyTorch/CUDA wheel、算子、显存和主机预算。没有因此改动
生产 `train.py` 的手动精度配置，或接通敏感层名单自动选择；3080/T10 的默认候选不变。

### 实际训练入口：显式 FP16/FP32 与启动 OOM 重试

已接入 `train.py`，默认关闭。Anima、Krea-2 和 Z-Image 的非量化 plain LoRA
现在共用同一个训练入口、precision manifest 和 fresh-process OOM supervisor；
FP16/FP32 岛屿仍是实验合同，不改变 NF4、复杂 adapter、compile 或选择性 checkpoint
等既有边界。敏感 Linear 名单可以显式提供，但尚未实现经整网数值验证的自动名单选择，
也未证明长期加速或质量。

在已配置好底模、数据和输出目录的训练 TOML 中显式设置：

```toml
adaptive_precision = "fp16_fp32"
adaptive_fp32_modules = [] # 用模型中的 Linear 名称/glob 指定 FP32 区域；空列表不是质量推荐
adaptive_loss_scale = 1024.0
model_family = "krea2_raw"
mixed_precision = "fp16"
base_compute = "bf16"
attn_mode = "torch"
network_module = "networks.lora_anima"
lora_adapter_kind = "lora"
network_train_unet_only = true
gradient_checkpointing = true
selective_checkpoint = "off"
torch_compile = false
block_swap_transfer_dtype = "bf16"
block_swap_restore_mode = "foreach"
max_data_loader_n_workers = 0
gradient_accumulation_steps = 1
seed = 42
adaptive_oom_retry = true
adaptive_oom_retry_max_attempts = 4
adaptive_oom_retry_swap_increment = 2
adaptive_oom_retry_max_swap = 0 # 0 = detect the legal model cap automatically
adaptive_oom_retry_timeout = 3600.0
```

`model_family` 也可以是 `anima` 或 `z_image`。Z-Image 的 FP16/FP32 岛屿路径必须
使用 `attn_mode="torch"`（或 `"sdpa"`）；Krea-2 保持 `torch`，Anima 可使用已验证的
attention 后端。选择 `auto` 时，Krea-2/Z-Image 会在进入 FP16/FP32 岛屿前把不兼容的
Flash 请求收敛到 native torch attention。三类模型都必须满足 plain LoRA、单进程、完整梯度检查点和 workers=0
等合同，启动预检会拒绝不兼容组合。

`adaptive_oom_retry_max_swap=0` 自动查询当前模型合法交换上限；正整数是用户预算，
超过模型结构上限会被拒绝。有效上限还会受实际模型结构与 CPU 主机内存预算约束，
且只会进一步收紧。Anima checkpoint 头部 28/40 块对应上限 26/38，Krea-2 加载配置 28 块对应 26，
Z-Image 从实际 `n_layers=30` 得出 28；这些不是跨模型固定默认值。初始 `blocks_to_swap`
必须不超过有效上限；其数值不是自动硬件推荐。正式 supervisor 以起始增量指数增长：
起始值 0、增量 2、最多 4 次尝试时为 0、2、6、有效上限；最后一次可直接到达该上限。
`adaptive_oom_retry_max_attempts` 包含首次训练尝试。需缓存 latent/text，关闭 preview、validation、
复杂 adapter、自动块交换旧入口和手动保存周期等不兼容功能；配置校验会明确拒绝。
这些字段已在 `/next/` WebUI 的“精度与计算”资源组中对三个模型族可见；`auto` 会在
训练入口依据 compute capability 解析为 BF16、FP16/FP32 岛屿或 FP32。默认仍是 `off`，
因为普通方法、量化和复杂 adapter 尚未纳入该实验合同。直接 Python 调用 trainer 时，
`adaptive_oom_retry=true` 会拒绝绕过 supervisor。

预加载仅检查本地结构信息，不加载权重 tensor；加载成功后再核验实际 block 容器。
CPU 模型按安装后的真实 dtype 评估 masters 和传输暂存预算；GPU 常驻的初始尝试不
冒用 CPU 增量预算。模型尚未加载成功时允许主机预算未测，仍保持进程级主机保护；
启用交换的成功尝试必须提供已测预算。旧独立 bench 默认继续使用线性增量。

冻结普通 Linear 默认 FP16，匹配名单的 Linear 与外部残差/非 Linear 浮点状态为 FP32。
使用 Accelerate 原生 FP16 GradScaler，关闭全局 autocast，保留梯度 unscale/clip；
安装在 adapter、offloader masters 和 compile 之前。`adaptive_precision.json` 保存
精度契约，恢复时缺失或名单不同会拒绝，但该文件不验证底模或数据集身份。

开启重试后，每次使用新进程和冻结的合并配置。只把明确的 CUDA OOM 作为显存不足，
且只在首次 optimizer 更新前的加载/搬运、forward、backward 阶段增加 swap。
不改变精度名单、loss scale 初始值、分辨率或 batch。主机内存保护、超时、取消、
普通异常均停止；GradScaler 跳过更新也停止，不把跳步算作成功提交。

每次完整更新后原子发布 Accelerate 状态快照并仅保留最近一份，包含 adapter、optimizer、
scheduler、scaler、RNG、训练步数和精度契约。目录位于输出目录的 `.adaptive-recovery/`，
包含 `summary.json`、每次尝试的日志/结果及 `latest-state.json`。目录已存在则拒绝覆盖，
需为新任务选择新输出目录。每步保存带来额外 IO 成本，不代表最终性能方案。

**默认（`adaptive_oom_retry_resume=false`）在首次更新开始后遇到 OOM 不自动续训。** 未开启中途恢复合同的情况下，数据桶原地 shuffle 且 Accelerate 存在预取，仅恢复 RNG 并跳过 batch 不能证明下一批相同；此时快照只以 `saved_state` 展示（`committed_checkpoint=null`、`data_cursor_resume_supported=false`），不会拿残缺快照或错误数据位置继续训练。若需单进程、`workers=0`、磁盘缓存条件下的中途 OOM 续训，须显式开启 `adaptive_oom_retry_resume=true`（完整协议、stateful 游标与快照成本见 [中途恢复说明](adaptive-training-recovery.md)）。原子目录发布不承诺断电持久性。该边界不影响下方独立实验 worker 已有的固定输入恢复证据，两者不可混用。

### 校准与固定输入实验

`library/training/adaptive_runtime/precision.py` 提供独立单元校准，比较 FP32
参考的输出、输入梯度及可训练参数梯度。数值失败选择 FP32；OOM 向外传播，
不能把内存不足误判为精度敏感。输入应来自真实模型的代表性激活。
局部校准通过不代表完整训练质量已经验证。

`islands.py` 可为冻结的普通 Linear 安装独立精度域，保持 state-dict 名称，
低精度计算后返回 FP32，避免污染外部残差精度。必须在 adapter、offloader
master 和 compile 建立之前安装；不接受量化或可训练参数。

`library/training/adaptive_runtime/retry.py` 提供有界恢复策略。只接受结构化
CUDA OOM，按阶段增加 swap、启用 checkpoint。改变 micro-batch 必须显式
授权，并保持 micro-batch 与 accumulation 的乘积。optimizer 已开始后若没有
完整检查点，拒绝自动重启。实际训练入口的单进程显式中途恢复已通过 `adaptive_oom_retry_resume=true` 接线（见 [中途恢复说明](adaptive-training-recovery.md)）；多进程与多 worker 通用数据游标恢复仍未开放。

`process.py` 实现独立子进程执行、结构化结果、超时和主机内存保护。
非零退出码、超时、主机内存不足不会伪装成 CUDA OOM 继续重试。
主线程取消信号会回收所属 worker 并记录 cancelled；不处理 SIGKILL/主机崩溃恢复。

`bench/adaptive_runtime/recover_krea.py` 已在真实 Krea NF4/BF16 worker 上验证
OOM -> 增加 swap -> 从已提交第 1 步恢复并完成第 2、3 步。其 checkpoint
只覆盖固定输入、无 scheduler 的实验 worker，不是生产 Accelerate 状态格式。
`--precision auto` 当前只依据硬件选候选，不等于自动误差校准通过。

Krea 与 Z-Image 实验恢复入口为 `bench.adaptive_runtime.recover_krea` 和
`bench.adaptive_runtime.recover_z_image`，共享 `recovery.py` 编排。Z-Image
检查点必须使用真实固定输入缓存；底模目录按 transformer 配置/分片的路径、
大小和 mtime 校验，不把目录自身 mtime 当作所有权重未变的证据。
T10 Krea 已验证首次 backward OOM 后 swap20 -> 24 新进程完成三步；
此次失败前没有 optimizer 更新，不等同于中途 checkpoint 续训。
Z-Image 双卡也已完成真实加载 OOM 后自动增加 swap 的三步测试；两卡另从
第 1 步实验检查点显式恢复到第 3 步。自动 OOM 任务与独立恢复测试分别记录，
不把二者合并成“同一任务中途 OOM 自动续训”的证据。

`bench/adaptive_runtime/capture.py` 从真实 forward 采集有界 token 行（包含
最大幅值行），排除 backward checkpoint 重算。`calibrate_capture.py` 对同一份
冻结 Linear 权重与输入比较 FP32 计算和硬件低精度候选，保存逐案例输出/
输入 VJP 误差及局部计划。参考权重来自已加载的 BF16/FP16/FP32 快照，
不是未舍入的原始 FP32 checkpoint；不包含 LoRA 参数梯度或整网组合误差。
局部计划不会自动覆盖已有敏感层配置，`full_model_calibrated` 始终为 false。

现有 NF4 BF16 限制保持不变；不通过关闭检查为 Turing 虚构支持。
OOM 恢复策略不自动改变分辨率、文本长度、优化器、量化或敏感区域精度。

## 联合预检

`bench.adaptive_runtime.search_dit` 把有限值精度搜索与 OOM 内存规划合到一次
有界启动中，当前实验模型族为 `krea2` / `z_image`。它复用
`adaptive_runtime/joint.py`、`numerics.py` 与 `retry.py::next_plan`：

- 硬件 auto 只决定初始 BF16 / FP16-islands / FP32-reference 候选，遵循上表。
- CUDA OOM 只增加 swap（本实验固定 full checkpoint、batch=1）。
- FP16 非有限值只在首个输入有限的普通 Linear 上提升 FP32，且不重复提升已为 FP32 的层。
- 每次尝试必须是明确标记的 disposable worker；不加载训练 checkpoint，失败后的
  少量实验更新可丢弃，成功后也不保存可被误当正式产物的 LoRA/checkpoint。
- 成功状态为 `finite_only`，`precision_calibrated=false`、`production_ready=false`；
  不是正式训练续训，也不是整网误差或生成质量认证。

T10 已完成同一命令内真实 OOM -> swap20/24 -> 自动提升一个遗漏的敏感 Q
投影 -> 三步验证。该测试以已有 71 个 FP32 Linear 为种子，不是从零发现全部
敏感层；最终 72 个 FP32 Linear。完整命令和逐次结果见实测报告。
正式训练期间的 OOM 恢复仍使用检查点协议，不复用这种丢弃实验更新的行为。

## 整网短训对照

### 有界数值配置搜索

`bench.adaptive_runtime.search_z_image_precision` 接入固定状态梯度比较与 OOM
swap 调整，当前仅限 Z-Image disposable 实验。需要预先录制的同卡 FP32
`training-replay` 与相邻完整 `training-capture`，不会跨卡复用参考。

硬件规则选择 BF16 时默认先试 BF16，选择混合精度时从 seed 开始；这两档的后续候选为
conditioning、MLP、attention 分组配置。硬件规则选择 FP32 时，默认只运行 `fp32` profile，
对应 `fp32-reference`、loss scale=1、无精度岛，不自动降到 FP16。
这是显式候选集合搜索，不是任意 block 自动定位，
也不保证增加 FP32 层数就改善数值。候选运行成功后必须通过全部固定状态
案例的输出/输入梯度/LoRA 梯度门槛，才返回 `probe_validated`。
`precision_calibrated`、`full_model_calibrated`、`production_ready` 仍为 false，
含义只是所给短探针案例通过，不能外推生成质量、长训或新输入。

数值失败或非有限值转到下一配置；真实 CUDA OOM 只增加 swap，保持精度、
batch=1 和 full checkpoint 不变。尝试次数是跨配置、跨 OOM 的全局上限。
每次 fresh worker，无 checkpoint/resume。主机内存保护、超时、取消、普通
运行错误立即停止；比较失败写终态错误。记录 IO 失败直接抛出并停止，不保证
磁盘不可写时还能落盘终态，不继续启动无日志的候选。

`--profile` 可显式指定顺序。`fp32-control` 把全部普通 Linear 设为 FP32，
只有明确指定时才运行，用于检查搜索/显存/比较流程，不能作为混合精度加速
达成的证据。默认候选不包含这个控制项。它与上述 `fp32` profile 不同，仍用于显式
检查 FP16-islands 框架中全 Linear FP32 的控制路径及 loss scaling。
另有显式组合`conditioning-mlp`、`conditioning-attention`，不改变默认候选
顺序。T10前者真实三步重放仍失败，第1步LoRA梯度误差6.38%、第2步输入
梯度误差6.48%，不可因单独conditioning改善就假定组合会通过。
后者在T10的三个固定状态案例全部通过：208个FP32 Linear、68个FP16 Linear，
LoRA梯度rel-L2为0.91%/0.38%/0.38%，峰值约5.86GiB。这是短探针通过，
不是独立训练轨迹/长期质量认证，不改变生产默认或默认候选顺序。
相同组合在3080同卡参考下仍失败：第1步LoRA梯度误差2.90%、第2步输入
梯度误差6.37%。因此T10的通过不能跨卡推广，仍需逐设备校准和重复验证。
T10去掉参考状态重放后，独立三步轨迹的第2步也失败：输入梯度误差7.61%、
LoRA梯度误差2.32%，不能将固定状态通过升级为独立训练已认证。

T10 已完成同一命令内 seed swap20 OOM -> swap24 三步但数值拒绝 ->
显式 FP32 control swap24 OOM -> swap28 全案例通过。最终全276个Linear为
FP32，证明流程协作，不证明混合精度加速。独立 conditioning 候选仅增加约
99.5MiB峰值，中间步LoRA误差从15.20%降至1.90%，但输入梯度仍7.15%，
且第1步LoRA误差2.17%，故正确返回失败。完整命令与证据见实测报告。

Krea/Z-Image 实验 worker 支持显式 `--precision fp32-reference` 和
`--capture-training`。参考仍从同一 BF16 checkpoint 加载并提升到 FP32 计算，
不是额外获取的原始 FP32 预训练权重；Krea NF4 不允许进入此参考路径。
Krea参考可显式添加`--reference-bf16-storage`：只允许swap>0、disposable和
training capture，拒绝checkpoint/resume及局部capture。实验helper在普通
BF16权重驻留后保留不可变CPU BF16 masters，将GPU浮点参数/buffer及后续
交换目标提升为FP32，并检查模块执行参数/buffer和CUDA autocast状态。
仅支持foreach，不修改公共offloader默认行为。这是BF16来源权重的FP32
计算参考，不是原始FP32预训练权重。T10完整三步参考已成功，解决此前主机
内存保护中止的问题；GPU峰值约7.07GiB，CPU masters约22.64GiB。
同卡scale1024 FP16候选峰值4.21GiB，但三步LoRA梯度rel-L2为
8.17%/2.20%/1.76%，前两步未通过当前实验门槛；本组短测也未显示加速。
当前仍需敏感区域定位，不能把省显存或OOM恢复通过视为精度认证。

`training_capture.py` 在每次 backward 后、梯度裁剪/optimizer 更新前记录预测、
输入 loss 梯度和全部可训练 LoRA 参数的 loss 梯度，按字节预算保存 safetensors。

`bench.adaptive_runtime.compare_training` 要求成功 worker、完整步序列、相同
源输入 hash、底模文件指纹、初始 adapter hash、训练参数和 sigma 序列，
且参考和候选必须提供相同的非空 GPU UUID。
缺失梯度、未知精度、hash/shape 不匹配均拒绝。候选 BF16 的运行输入会按其
真实训练路径舍入，因此“相同输入”指相同源缓存，不保证运行张量逐位一致。

报告比较每步预测、输入梯度、整体 LoRA 梯度的 relative-L2/cosine，并分别
列出相对误差与绝对 L2 偏差最大的参数，避免微小参考梯度主导诊断排序。
偏差位置不等于误差源，不能直接映射为 FP32 提升名单。
后续步骤包含各自 optimizer 更新产生的轨迹差异，不是固定
同一 adapter 参数上的独立微分对照。实验门槛沿用 1% 输出、2% 梯度、0.999
梯度 cosine，未验证为长期质量标准；无论是否通过均不标记生产可用或整网已认证。
Krea FP32 参考首轮曾触发主机内存保护，后续完整参考见上文；Z-Image T10 FP32
三步参考及重复对照已完成。未缩放 72 层混合候选的 LoRA 梯度 rel-L2 为
13.7%/38.6%/17.9%，未通过整网短训门槛。受限 `--loss-scale 128` 对照将其
降为 1.35%/15.59%/1.05%，仍有一个 sigma 不通过，不能发布为验证过的默认精度名单。
进一步提高 scale 到 1024，LoRA 梯度误差为 1.36%/10.42%/1.15%，中间步
仍失败；不能把增大 scale 当作已经解决的自动策略。
loss scaling 默认仍为关闭；非检查点缩放实验只允许 disposable FP16 capture。
检查点缩放需使用下述显式开关，不改变现有恢复 worker 默认行为。
详情及候选比较见实测报告。
实验checkpoint API已新增可选scaler状态v2及CPU精确续步回归，v1未缩放
检查点保持兼容；带缩放但缺失scaler的旧状态拒绝。Z-Image和Krea实验worker现可
显式使用`--scaled-checkpoint --checkpoint-every-step --loss-scale 1024`进行
固定输入状态保存/恢复，必须使用FP16-islands及真实缓存，拒绝disposable、
capture和replay混用。两个模型的自动recovery CLI也支持相同显式开关，并在每次
fresh worker中保持相同loss scale和precision；这是独立实验格式，不直接加载到上述训练入口。
Krea此路径只接受非量化权重，仍在加载前拒绝NF4，不更改生产BF16限制。
该开关不表示精度方案已认证，也不等于已通过中途OOM自动续训。
T10已完成scale1024三步保存及step1新进程swap24->26恢复，最终272组AdamW
step与scaler增长计数均为3，检查点与该次输出一致。与不中断路径最大参数
差约3.93e-5，不能宣称逐位轨迹等价；未发生OOM，仍缺自动OOM接线实测。
后续T10自动入口实测已完成：5.5GiB分配器上限下swap24加载OOM，保持
scale1024与精度名单不变自动swap26三步成功，峰值约4.80GiB。最终scaler
计数/AdamW step均为3、检查点与输出一致。这是首次更新前重启，仍缺带缩放
的中途OOM检查点恢复证据；前述显式resume测试不可冒充这一证据。
3080也已完成相同scaled自动重试：swap24加载OOM->swap26三步成功，峰值
约4.80GiB。两卡均经`bench.adaptive_runtime.audit_recovery`核验worker历史、
最终检查点、scaler/AdamW进度与adapter一致；均明确未发生中途checkpoint
resume。这是恢复流程证据，不改变3080该精度配置数值门槛未通过的结论。

Krea T10现也完成显式scale1024自动重试：非量化底模、264个FP16 Linear与
FP32残差，9.05GiB分配器上限下swap20首次backward OOM，swap24三步成功。
峰值6,258,903,040 bytes（约5.83GiB）；v2检查点的scaler计数与392组AdamW
step均为3，最终adapter与检查点一致。本次没有checkpoint resume，也未完成
Krea整网FP32数值对照；不能据此认证混合精度质量或长期加速。

后续T10 Krea在6.0GiB分配器上限下已完成真实带缩放中途恢复：swap24提交
第1步后，第2步backward OOM；自动swap26从该检查点完成第2/3步。统一审计
`checkpoint_resume_observed=true`，scaler计数和392组AdamW step均为3，
恢复worker峰值约4.21GiB。最终588个adapter张量与先前不中断三步结果逐位
一致，仅为此固定输入短测证据，不涵盖生产scheduler、sampler或数据worker。
命令、5.75GiB未命中中途OOM的先行尝试及对照产物见实测报告。

3080 Z-Image 同卡 FP32/BF16 三步对照也已完成：LoRA 梯度 rel-L2 为
30.57%/26.97%/56.95%，全部未通过当前实验门槛。比较包含 BF16 输入舍入，
且此门槛不是已验证的长期质量标准，因此既不能认证候选，也不能据此宣布
生产 BF16 不可用。后续需要同 adapter、同运行输入的单因素重放。

## 验证进度

### 固定状态重放

Z-Image worker 新增 `--record-replay`（FP32 参考）和 `--replay-reference`
（参考的 `training-replay` 目录），仅允许 disposable + 真实输入 + training
capture，拒绝 checkpoint/resume 和局部 Linear capture。尚未接入 Krea/Anima
worker 或生产训练恢复。

每步 forward 前记录完整 network 参数、包括非持久化 timestep mask 在内的
全部 buffers、实际 noisy/target/prompts/sigma，以及 torch CPU/CUDA 和 Python
RNG。候选恢复后重新计算摘要；源 worker 必须成功且同 GPU，底模/输入/训练
签名一致，快照受 hash、shape、dtype、有限值和字节预算约束。磁盘 tensor
预算默认 1GiB，单案例最多 512MiB，不保存可作为正式训练输出的 adapter。

候选仍完整执行 backward 和 disposable optimizer 更新以保持探针生命周期，
但下一步会覆盖成参考的 pre-step 参数/buffers，候选的上一步更新不传入下一步
forward。优化器状态不是被校准对象；不能把这种实验当作续训或独立训练轨迹。
报告 scope 为 `fixed_pre_step_state_and_inputs_not_quality_certificate`，比较器
要求逐步状态/输入/RNG/快照摘要一致，否则拒绝输出固定状态对照结论。

与原 BF16 轨迹不同，重放候选使用参考记录的 FP32 noisy、target 和 prompt，
保留候选内部 autocast/计算精度。这用于剥离输入舍入与更新轨迹的影响，不代表
生产 BF16 路径默认保持这些输入为 FP32；两类结果不可混写为同一实验。

T10 已完成真实三步录制、72 层 FP32 + FP16 scale1024 重放与 FP32 重放
控制组。候选 LoRA 梯度 rel-L2 为 1.36%/15.20%/0.91%，中间步仍失败；
控制组三步预测逐位一致，LoRA 梯度 rel-L2 约 1.4e-6 到 2.7e-6。
目前可排除“误差全由之前 optimizer 更新不同导致”，但尚不能直接定位敏感 block。
3080 同卡固定状态 BF16 重放也已完成，三步 LoRA 梯度 rel-L2 为
19.47%/39.42%/27.56%，仍全部失败；同运行输入也不足以消除偏差。
两卡具体参考和候选目录、限制与误差表见实测报告。
T10 另试将全部 attention Linear 提升 FP32（共170个 FP32 Linear）：显存峰值
比72层方案增加约0.83GiB，中间步 LoRA 梯度误差反而为16.82%，未采用。
精度域组合的误差不能假设随 FP32 层数单调下降，仍需实际干预和整网梯度验证。

实际命令、单元测试和双卡真实权重单层结果见
[实测与失败证据报告](../findings/adaptive_runtime_20260921.md)。

2026-09-21 初始硬件探测：PyTorch 2.12.0+cu130；3080 SM 8.6 / 20GB；
T10 SM 7.5 / 16GB；主机内存约 62GiB。PyTorch 默认枚举与 nvidia-smi
序号相反，热测试必须通过 GPU UUID 选择设备。

| 模型 | 3080 | T10 | 实际权重 |
| --- | --- | --- | --- |
| Anima | 待测 | 待测 | 历史路径失效，等待定位 |
| Z-Image | 真实输入 3 更新、OOM swap16 -> 20、独立 checkpoint 恢复通过；4 层局部 BF16 校准通过，但同卡整网 FP32/BF16 梯度门槛失败 | 72 个 FP32 Linear 的有限值/OOM/独立恢复通过，但完整梯度门槛失败；AMP scaling 改善后仍有一个 sigma 失败 | 已在相邻 Z-Image Diffusers 目录找到 |
| Krea-2 | NF4/BF16 真实输入 3 更新及 OOM checkpoint 续训通过；非量化 3 层局部 BF16 校准通过 | 非量化 FP16/FP32 三步及真实 OOM swap20 -> 24 重试通过；3 层局部 FP16 校准通过，尚非整网校准 | 本仓 BF16 与 NF4 可用 |

所有待测项均未达到交付验收。小模型、随机输入和模拟 OOM 只能作为分层测试，
不能替代六个真实模型/设备组合的热测试。
