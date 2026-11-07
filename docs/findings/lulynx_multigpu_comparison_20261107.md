# Lulynx 多卡方案与本项目对照：可迁移机制、反例与分级路线图

状态：实验 / 源码比较研究（t5 收口版），未改动生产默认值，未运行任何多卡训练
适用版本：本项目分支 `codex/multi-gpu-training`，基线 `2987186c542b09a55c70a17c3dd523f3b11fb4d5`；
参考仓库 `/mnt/SD_Dataset/训练器相关/lulynx-trainer`，基线 `92f3c07cb63dc3ebc4aad31f9f2b49fc8883f9dd`
上游计划：[多卡训练实现研究计划][p-plan]
参考设计入口：[Lulynx 多卡训练方案文档][r-design]
本地基线记录：[2×V100 NVLink 基线][l-baseline]（本文 §6 对其中两处口径做了纠偏，基线文件本体不改）

## 1. 目的与边界

本轮回答一个问题：**Lulynx 的多卡设计里，哪些机制值得搬进本项目，搬的时候会在哪里坏。**

- 只做源码比较。未运行任何测试、探针、训练；未安装依赖；未使用 GPU。
- **未改两侧训练源码**：参考树未改（唯一未跟踪文件 `guide/fp16_nan_protection_design_2026-11-06.md` 未读未改）；本研究分支仅三份文档改变（本报告、[findings 索引][p-index]、[计划文档][p-plan]）。
- 证据等级：**A**=源码+调用链闭合；**B**=静态推断（写出推论所需前提）；**C**=需实测或受依赖版本漂移影响；**D**=文档声称但无可复核产物。
- 本批**没有任何真机训练证据**。参考仓侧曾有 2 进程 Gloo 契约，但脚本已删除（§3.2），本轮无法复核；CPU/Gloo 一律不充当 CUDA 验收。
- 依赖环境漂移：本工作树无 `.venv`；accelerate / torch 只在只读环境 `../krea2-webui/.venv-v100`（accelerate 1.13.0、torch 2.10.0+cu129，后者由[传输 artifact][l-artifact] 的 `torch_version` 记录佐证）。本树 [pyproject.toml][l-pyproject] 声明 `torch>=2.12.0,<2.13`，两者不一致，故涉及依赖内部行为的结论一律标 C，行号可能偏移，需按符号名重新定位。

## 2. 两边机制的根本差异（先定坐标，否则必然提出错误移植）

| 维度 | 本项目 | Lulynx |
| --- | --- | --- |
| DDP 包装对象 | **adapter 容器不入 DDP**：只 `accelerator.prepare(optimizer, loader, scheduler)`，adapter 参数与梯度由自有同步器归约（[gradient_sync.py:298-317][l-gsync-prepare]、[bootstrap.py:1049-1063][l-bootstrap]） | **整棵 DiT 子树入 DDP**：注入后的 `self.model.unet` 直接交给 `DistributedDataParallel`（[trainer_execution_optimizer_setup.py:393-402][r-opt-setup]、[distributed_runtime.py:163-254][r-runtime]） |
| 归约时机 | 只在 `sync_gradients` 边界归约一次（[loop.py:1143-1159][l-loop-reduce]），loss-scaled 状态下归约 | 每个 micro-batch 的 `backward()` 触发 reducer 归约，未见 `no_sync`（[step_compute.py:750-757][r-step-compute]、[accumulation_batches.py:38][r-accum]） |
| 梯度归集的覆盖面 | 由 **optimizer `param_groups`** 决定（[gradient_sync.py:227-241][l-gsync-params]）：登记在参数组里的 adapter 张量（含 TE adapter）都被覆盖；**未登记为 optimizer 参数的张量不会被广播或归约** | 由 **被包装模块的 root forward** 决定：可训练参数必须 ⊆ 被包装模块，否则不同步（§7 R1） |
| 数据结构差异 | 每个参数显式记录 presence 掩码，全 rank 都缺梯度时保持 `grad is None`（[gradient_sync.py:320-389][l-gsync-sync]） | 由 reducer 的默认语义处理；具体行为还受 `find_unused_parameters` 等配置影响（[distributed_runtime.py:283-285][r-ddp-wrap]），**不能简化为"缺梯度写零"** |

**结论：不能"换成原生 DDP 更省事"。** 两条路线各自的**正确性条件不同**：本项目是"参数集合 ⊆ optimizer param_groups"，参考仓是"可训练参数 ⊆ 被包装模块"。硬换会同时丢掉 `after_backward` 语义与 presence 掩码；反之，本项目的条件也要求 adapter 必须把自己的参数登记进 optimizer，否则同样会漏同步。

## 3. 参考文档与参考源码的一致性（我方独立核验）

### 3.1 成立（10 项）

| # | 文档声明 | 结论 | 证据 |
| --- | --- | --- | --- |
| V1 | `auto` 在 world>1 解析 DDP、world=1 折叠 none；TP/SP 请求时 auto 不抢占 | 成立 | [distributed_strategy.py:224-247][r-strategy]；单测 [test_distributed_strategy_unit.py:31-62][r-test-strategy] |
| V2 | 选卡数与 `num_processes` 对齐，显式不一致 fail closed | 成立 | [queue_worker_launch.py:101-114][r-queue] |
| V3 | DDP wrapper、`device_ids` 绑定、sampler 重建、stop 用 MAX 广播 | 成立 | [distributed.py:106-127,140-186][r-distributed] |
| V4 | `global_effective_batch = per-rank batch × accum × world` | 成立 | [training_step_breakdown.py:390,501][r-breakdown] |
| V5 | RNG 分层：DP=`seed+rank`、SP 共享、TP=`rank//degree` | 成立 | [trainer_model_prep_runtime_utils.py:57-82][r-prep-utils] |
| V6 | FSDP2 leaf-first、深优先、冻结优先，不退化 DDP | 成立 | [distributed_runtime.py:315-345,372-495][r-runtime] |
| V7 | FSDP2⊥DeepSpeed、DeepSpeed⊥LoRA、缺包、world≤1、FSDP2⊥Quanto 具名 code | 成立 | [distributed_strategy.py:382-497][r-strategy] |
| V8 | TP 要求 `degree==world`、与 DP 策略互斥；**先切裸模型再注入 adapter** | 成立 | [tensor_parallel_runtime_profile.py:76-101][r-tp-profile]；[trainer_prepare_adapter_inject_mixin.py:42][r-inject]；[trainer_prepare_tensor_parallel_mixin.py:166-197][r-tp-mixin] |
| V9 | SP 仅 Krea2 native + plain LoRA，`expanded_kv` 路由，模块不齐直接 raise | 成立 | [trainer_prepare_tensor_parallel_mixin.py:117-161][r-tp-mixin] |
| V10 | PP 是单进程跨卡 model sharding、无跨 rank 1F1B、自报 `microbatch_sequential` | 成立 | [pipeline_parallel.py:15-31][r-pp] |

### 3.2 需要收窄表述（3 项）

- **多卡 smoke 证据**：设计文档 §10.1 引用的 `ddp_smoke.py`、`distributed_strategy_preflight_smoke.py`、`ddp_gloo_dual_process_smoke.py` 在当前树不存在——由提交 `aab16963d`（2026-09-30）删除，同批还删除了 `parallel_tp_multirank_gloo_smoke.py`、`tensor_parallel_lora_dual_process_gloo_smoke.py`、`krea2_sequence_parallel_gloo_support.py`。[CUDA 发布清单 2026-08-18:46][r-inventory] 记录了当时的 PASS（preflight 15/15、`ddp_smoke.py` 24/24、`ddp_gloo_dual_process_smoke.py` 2/2 rank）。
  正确表述：**存在历史报告，但当前版本缺少相应脚本与运行产物，本轮未复核；不是"从未验证"，也不能沿用其通过计数。**
- **[multi_gpu_parallel_scorecard.py][r-scorecard]**：硬编码 `wired_into_trainer=False`/`promotion_ready=False`（[multi_gpu_parallel_scorecard.py:40-75][r-scorecard]），又与同文件 notes 的"degree>1 已接线"矛盾，全树无调用方。这是本次抽样的**一处**账本腐化；实际接线是真的（V8）。
- **run manifest 字段**：文档 §5.3 称 manifest 会写 `world_size`/`global_effective_batch`/`per_rank_microbatches_*`；实际这些键只在 API 侧预检 breakdown（[training_step_breakdown.py:493-505][r-breakdown]），trainer 侧 manifest 未写。

## 4. 可迁移机制与优先级

分两轴看：**控制面**（准入、判定、清单、恢复契约）与**梯度主干**（归约、跳过决策、scaler、collective 顺序）。后者是本轮风险最集中的地方，优先级高于控制面。

- **P0-0（最高）窗口健康判定 / 跳过决策 / scaler 状态的跨 rank 一致性**：这是**已知可达**的静态风险（§8.5、§10），必须先于任何真实短训。改的是梯度主干的控制流，不是"一行小修"，也不是"补一个 reduce"。
- **P0-1 选卡与 `num_processes` 对齐的 fail-closed 语义**（参考 REF-02）：本项目已有 ≥2 卡时自动置 `ANIMA_ACCELERATE_LAUNCH=1` 与 `NUM_PROCESSES=len(gpu_selection)`（[launch.py:50-63][l-launch]），但缺少"用户显式填的进程数与选卡数冲突就拒绝启动"的判定。
- **P0-2 启动期 collective 健康探针**（参考 REF-03 的 probe 部分）：注意它**不是零运行时影响**——探针本身会在启动期引入一次真实集合通信，且必须"失败即中止"（与参考仓的"降级继续"相反，见 §7 R3）。落地方式：仅多卡路径启用或显式开关，默认关。
- **P0-3 运行清单记录多卡身份**：`world_size`、`configured_world_size`、归约模式（同步/异步）、bucket 数、**实际生效的 `mixed_precision` 与改写原因**；rank0-only 写。
- **P0-4 准入原因码表 + 向训练进程内的兼容门传真实 `world_size`**：参考 REF-01/REF-06 的组织方式；同时修本节指出的真实裂缝：训练进程内调用 `check_training_compat()` 时未传 `world_size`（[bootstrap.py:162-172][l-bootstrap-compat]，[compat_matrix.py:241][l-compat]），规则按 `world_size or 1 = 1` 求值，只有 Web 预检传了真实值（[preflight_compat.py:191][l-preflight]），CLI 直连多卡会漏判。
- **P0-5 恢复语义显式化**：未启用 `use_seedable_sampler` 只能证明 accelerate 那条特定的
  `SeedableRandomSampler` 分支没有被显式启用（[accelerator.py:3678,3840][a-accel] 的 save/load 走
  `self._dataloaders` 注册表），**不能据此推断所有 sampler / RNG 状态都没保存**：实际还要看 loader 类型、
  全局 RNG 与 sampler 生成器的交互、epoch/游标来源以及已注册的 state hooks；恢复侧的
  `skip_first_batches`（[loop.py:801-805][l-loop-skip]）也只是跳过 N 个批。
  **结论收窄为**：当前证据**不能承诺精确恢复**；需要"同 world_size 的中断/续训对照"作为判定用例，
  并对"改变 world_size 后恢复"给出显式拒绝或降级策略（现状无任何提示，见计划文档 S2 与 P1-2）。
- **P1-1 checkpoint 三阶段协议**（参考 REF-11）。参考实现把保存拆成三个阶段：**所有 rank 都进入**
  `save_training_state`（[trainer_collective_state_save.py:51-152][r-ckpt-save]），先在 `save_prepare`
  阶段交换并校验 (epoch, step, final, 是否需要全 rank 采集)、identity 与 rank 拓扑，再各 rank 本地采集
  （rank-local CUDA RNG 清单、loader/sampler 快照，并从 bundle 里剔除 `SHARED_KEYS`），`save_capture`
  阶段再比对 global_step 是否全等，最后**只有 rank0 组装 bundle 并落盘**、把报告广播回所有 rank
  （传输层见 [distributed_checkpoint_transport.py:71-89][r-ckpt-transport]）。
  移植边界：本项目当前用 `accelerator.save_state()` + 手工 hook（[checkpoints.py][l-ckpt]），要走三阶段
  协议必须与之对齐（谁负责收集、谁负责 rename/上传）。
  失败语义要分两种情况：**进程组健康时，局部采集或 rank0 发布失败仍须交换状态，使所有 rank 作出一致
  失败决定**（参考实现即如此：局部 capture 异常经 `exchange("save_capture", …)` 带上 `errors` 汇总，
  rank0 的发布失败也通过 `exchange("save_publish", …)` 回到所有 rank）；**一旦 transport 失败或 peer 失联，
  不再依赖新的 collective**，转由 PG timeout / 进程退出机制终止。本轮只做源码定位，**未验证**任何容错
  或恢复保证。
- **P1-2 版本化 per-rank resume bundle**（参考 REF-12）。参考实现把恢复信息放进
  `distributed_resume_bundle`（[distributed_resume_state.py:20-31][r-resume-state]）：metadata 带
  `version`（必须为 1）、`world_size`、`backend`、`rank_mapping` 与 `rank_local_keys`，加载时校验
  version 与当前活组一致、rank↔device 映射逐项匹配、rank-local 字段不得含 `SHARED_KEYS`，
  `restore_rank_runtime_state` 负责恢复 loader 与 CUDA RNG 清单
  （[distributed_resume_state.py:88-150,164-186][r-resume-state]），`synchronize_resume_result`
  再交换 status 比对 global_step / epoch / scheduler / lr（[distributed_resume_state.py:188-217][r-resume-state]）；
  没有 bundle 的 legacy checkpoint 在新 world 下显式降级为 shared-only 并标 `exact_resume=False`。
  移植边界：要接到现有 Accelerate save/load 之上，并明确"拓扑不符时拒绝还是降级"。本轮只做源码定位，
  **不声称本项目已具备精确恢复能力**。
- **P1-3 带证据指针的能力账本**：结构清晰、成本中等，但同样需要先有实测证据才能填 `validated`。
- **P2 FSDP2 / TP 中高风险实验候选（不实现、不宣称收益）**：
  - **EXP-FSDP2-BASE**（open ownership 设计，不是既定顺序）：目标是对**冻结基座**分片以降低驻留显存，
    但**参数同步的归属必须先定清楚**，再谈注入与包装顺序：
    1) **归属**：冻结基座由 FSDP2 的 `fully_shard` 与某个 process group 负责；adapter 参数若继续走现有的
       手动同步（[gradient_sync.py:320-389][l-gsync-sync]），必须明确它与 FSDP2 的边界，**同一参数不得被
       两条路径各归约一次**（双重归约），也不得出现"谁都不管"的第三种状态。
    2) **不能默认**："shard 之后注入的 adapter 会自动被 FSDP2 管理"。`fully_shard` 是对**当时已存在的
       参数**做分片与 hook 注册，后注入的参数是否被纳入取决于 API 行为与参数身份，必须实测确认。
       因此**不从 TP 的"先切后注入"推导 FSDP2 同序**（那条结论的依据是 TP 准入里"注入器要能看见已替换层"，
       见 [trainer_prepare_adapter_inject_mixin.py:42][r-inject]，与 FSDP2 的 ownership 语义不同）。
    3) **顺序待验证**：注入、`fully_shard`、optimizer 构造三者的先后，必须按实际 API 与参数身份
       （`id(param)` 是否稳定、是否有参数别名/容器重写）逐项验证后再定，不在文档里预设。
    **收益前提**：基座显存确实是瓶颈；**已知成本**：all-gather 通信、激活峰值不降、wrap 粒度影响分片效率、
    参数别名与 monkey-patched adapter 的所有权风险。**拒绝组合**：与 TP/SP 混用、block swap/module offload、
    量化（NF4/FP8）、PP、未验证的 compile。
    **验收分两级**：CPU/Gloo 只用于验证本项目**自己的参数清单/拓扑协议或 mock**（例如断言
    `trainable_param_ids ⊆ optimizer_param_ids`、`requires_grad` 不变、wrap 计数正确），**不能证明
    `fully_shard` 真实运行**；真实的 FSDP2 门槛必须另列——受控小模型、CUDA/NCCL 前后数值对齐、保存/恢复、
    峰值显存，且**不在本批执行**。
  - **EXP-TP-COLROW**：TP column/row 替换 + 注入器识别 + 复制参数的**梯度同步协议**需显式定义（哪些参数在每个 rank 都有完整副本、在哪个边界同步），并把 **buffer/掩码类对象**的广播或共享状态协议与之**分开**——掩码不是梯度，不能按参数梯度归约处理；具体的 process group 与 sum/mean 语义必须按 TP 的数学契约逐项验证，不能假设所有对象都做一次梯度 all-reduce。**定位**：单卡容量不足时的候选之一（不是唯一路径）；**已知成本**：每 block 多次集合通信、小消息延迟主导、GQA/QKNorm/融合 QKV 家族必须在准入拒绝。**拒绝组合**：任何 DP 混用（参考仓明确要求 degree==world）、Krea2 attention（参考仓以 GQA 为由整段排除，[specs.py][r-specs]）、fused QKV。**最小验证**：TP1 与 `nn.Linear` 逐位一致、TP2 数值 rel-L2 门槛 + 集合通信次数统计。
  - 两者都**只到实验候选**：没有真机收益证据，不得写进默认配置或 UI 提示。

## 5. 已有等价机制：不要重复造轮子

| 参考机制 | 本项目的对应实现 | 准确的相同/不同处 |
| --- | --- | --- |
| 缺梯度的跨 rank 处理 | [gradient_sync.py:320-389][l-gsync-sync] presence 掩码 + 均值 | 本项目的均值分母是 **world_size**（与 DDP 均值一致）；差别在于"全 rank 都缺梯度"时**保持 `None`**，即该参数本次不推进 weight decay/优化器状态。这是**语义差异**，不是普遍的"更保守/更强" |
| 非有限梯度的一致性 | 归约在前（[loop.py:1143-1159][l-loop-reduce]）+ 无 scaler 时的显式 `reduce(max)`（[scaler_ownership.py:161-174][l-scaler-owner]） | 只覆盖"梯度本身非有限"这一支；**不覆盖"loss 非有限但梯度有限"**与"各 rank 窗口长度不等"（§8.5、§10） |
| 权重级 max-norm | [regularization.py:16,23-52][l-reg] 本地确定性缩放 | 输入相同则结果相同，不需要归约；**是否合规取决于参数一致性这个更大不变量**，不单列缺陷 |
| 采样/预览的 rank 编排 | [library/anima/training.py:986-1100][l-anima-preview]（prompt 按 rank 分片、失败先归约再统一上抛、双 barrier） | 覆盖参考 §6.2 的"全 rank 参与 collective + 仅 rank0 落盘" |
| 多卡/多精度 fail-closed 骨架 | [native_precision.py:66-67][l-native]、[resume_runtime.py:45-46][l-resume-runtime]、[resume_checkpoint.py:91,202][l-resume-ckpt] | 这些闸门同时意味着"多卡 + adaptive 精度/恢复目前不可组合"（§8.5） |
| 主进程副作用 | [checkpoints.py][l-ckpt] 多处 `is_main_process`、[log_dispatch.py:60][l-logdispatch] | 与参考 §6.2 原则一致 |

## 6. 对既有基线文档的口径纠偏（基线文件本体不改，以本节为准）

1. **精度档默认值**：`ANIMA_ACCELERATE_MIXED_PRECISION` 默认 `bf16`（[launch.py:37-46][l-launch]）只是 launcher 侧提示；训练侧 `Accelerator(mixed_precision=args.mixed_precision)` 显式传参（[accelerator.py:159-166][l-accel]），而 `--mixed_precision` 的**直接 CLI parser 默认是 `no`**（[cli_args.py:659-664][l-cliargs]）——注意这只是直接 CLI 入口的 parser 默认，**不能替代**任务配置/TOML 合并后的实际默认，需按入口分别确认。bf16→fp16 的自动改写来自 [precision_policy.py:43-52][l-precision]（SM<8），调用点在 [train_session.py:163][l-trainsession]，早于 `prepare_dtype()` 与 `Accelerator()` 构造。
   结论：SM70 上 bf16 走仿真（§6.4 的算力比值）仍是"不要选 bf16"的依据；风险从"静默按 bf16 训练"降级为"启动参数与实际精度不一致，导致日志/清单误导"。
2. **P2P 带宽的两种字节口径**（前轮已有 artifact，本轮只读复核：[parallel_transport_v100_nvlink_20261107.json][l-artifact]）：rank0 `all_reduce` 中位 0.0006776 s（[artifact:242-258][l-artifact-ar]）、`pp_roundtrip` 中位 0.0010504 s（[artifact:261-277][l-artifact-pp]），两者 payload 都是 37,748,736 B；artifact 的 `notes` 明确"PP roundtrip is two serialized tensor transfers"（[artifact:639][l-artifact-notes]）。
   因此存在两种**不同字节口径**：`S/t_rt`（每个方向各传 S、总时间 t_rt）与 `2S/t_rt`（两次传输总字节 / 总时间）。后者是"串行 ping-pong 的总吞吐"，**不自动等于真实单向峰值**；要得到单向时间必须是 `t_rt/2`，这依赖"两个方向对称"的假设（artifact 未测非对称负载）。16.40625 MiB 的精确 payload 是 17,203,200 B（[artifact:239][l-artifact-d2h]）。
   基线表把 `payload/roundtrip` 标注为"单向等效带宽"、正文又写"取表中值的一半"，属推导描述不一致；**两种口径都不能当链路峰值，也不能当 PP 安全保证**。
3. **小消息延迟与历史 TP 计数**：0.0625 MiB 时 all_reduce 中位 0.181 ms 是该探针在**该消息尺寸、该重复次数**下的统计值，**不是物理延迟下限**。历史异构机（170HX+3080，无 peer access）报告里的 `Calls/rank = 3960` 是"1 warmup + 8 optimizer steps"**整个窗口的累计**，`3960/9 ≈ 440` 只是**窗口平均**，不是每步逐条精确计数；本报告不再补算"每步 540–560 次"这类未经 artifact 复核的静态核算，也不把 `440 × 0.18 ms` 转成"每步 X 秒下界"。
4. **matmul 精度表属手工观测（D 级）**：按本会话前轮命令记录，实际是先 warmup 5 次、`cuda.synchronize()` 后做 30 次 matmul 取**总时间/30**，即**均值**而非中位数；无独立脚本、无原始 artifact，仅可作一次性观测。V100（SM70）**没有 TF32 Tensor Core**，因此"未记录 `allow_tf32` 会让 fp32 行极敏感"的说法不成立，已删除。表中 TFLOP/s 也不能用来断言训练吞吐。

## 7. 不可直接照搬项（含反例）

(A) 为"代码里确有什么/没有什么"，(B) 为推论所需前提。

- **R1（高）参考仓的 DDP 包装边界 ⊂ 可训练参数集合**（A）：`_wrap_ddp` 只接收 `self.model.unet`（[trainer_execution_optimizer_setup.py:393-402][r-opt-setup]），TE LoRA 走独立注入路径（[lora_injector_inject.py:48][r-inject-te]），且 [configs_parallel_interlock.py][r-interlock] 与 `parallel/` 下没有"TE 训练 × 多卡"互斥。推论（B）：多卡 + TE LoRA 时 TE 参数梯度不被 AllReduce，各 rank 静默发散。**迁移规则**：引入包装式并行时断言 `trainable_param_ids ⊆ wrapped_module_param_ids`，否则 fail closed。
- **R2（高）rank-local 判定 × 每微批归约**：非有限 loss 判定是进程本地的（[training_loop_gradient_mixin.py:139-151][r-gradient-mixin]），标志位 `_accumulation_window_invalid`/`_nonfinite_stop_requested` 也是本地字段，没有"本窗口作废"的跨 rank 协商；叠加每微批归约后，各 rank 的 backward/collective 次数可不等。与 §8.5/§10 属同一类问题：**归约梯度不构成"一切安全"的证明**。
- **R3（中高）PG 建组失败后按单进程继续**：[distributed_process_group.py:290-294][r-pg]（告警 + cleanup + return False）→ [trainer_model_prep_runtime_utils.py:111-126][r-prep-utils]（非 TP/SP 记日志继续）→ [distributed_runtime.py:203-215][r-runtime]（显式 FSDP2/DeepSpeed 拒绝，DDP 可返回未包装对象）。**边界**：问题限定在 DDP 意图路径，且**有告警日志**，其他策略已有 fail-closed。
- **R4（中）累积窗口按 rank 本地首批动态决定长度**（[accumulation_batches.py:88-105][r-accum]），叠加每微批归约即 R2 的另一触发路径。
- **R5（中）`drop_last=False` 时尾批复制补齐**（[distributed_dataloader.py:219-232][r-dataloader]）：同 epoch 内部分样本被多 rank 重复计入，全局 batch 与有效样本数失真，且未写入 manifest。
- **R6（中）stop OR 不是失联容错**：`broadcast_stop_signal` 只表达"有人要停"，peer 死亡只能等 PG timeout。本项目不用该机制（取消走进程树终止），代价是没有优雅 flush——属设计取舍。
- **R7（中）`LOCAL_RANK` 取模兜底**：`nproc_per_node` 大于可见设备数时多 rank 会静默落到同一张卡，应改为 fail closed。

## 8. 对本批其他审查推断的反证

结论：以下原推断**不成立或需降级**。

1. **`scale_weight_norms` 缺 all_reduce 不构成缺陷**（未证实假设）。[regularization.py:16,23-52][l-reg] 只读 `state_dict()` 的 down/up/alpha 与同一 `max_norm_value`：`norm = updown.norm().clamp(min=max_norm_value/2)`、`desired = clamp(norm, max=max_norm_value)`、`ratio = desired/norm`（[regularization.py:40-55][l-reg]）。这是确定性运算，**输入相同必然得到相同结果，不需要归约**；比值只会在 `(0, 1]` 区间内缩（`desired` 恒 ≤ `norm`，没有放大方向），先前"ratio 夹在 [0.5, 2]"的说法**无源码依据，已删除**。它唯一的隐含前提是"各 rank 参数一致"，属于 §10 参数一致性验收的一个观测点；**在没有给出可达的、会造成输入不同的来源之前，本项不是风险项，也不设"预期失败"探针**。
2. **`init_trackers` 不是多 rank 重复副作用**：accelerate 1.13.0 的 `init_trackers`（[accelerator.py:3259][a-accel]）与 `log` 带 `@on_main_process`，非主进程 tracker 列表为空，[log_dispatch.py:36-58][l-logdispatch] 空转。**风险不成立**（C）。仅留健壮性备注："只有 rank0 写"是靠列表为空间接获得，不是显式闸门。
3. **scheduler 乘 `num_processes` 不是步数翻倍**：[schedulers.py:245][l-sched] 乘了 `num_processes`，而 `AcceleratedScheduler.step()` 在默认 `step_with_optimizer=True`、`split_batches=False` 时每次 optimizer step 推进底层 scheduler `num_processes` 次（[scheduler.py:54-76][a-sched]），两者相抵。**风险不成立**（C）。前提：调度器经 `accelerator.prepare` 包装、每窗口只推进一次（[loop.py:995-1005][l-loop-boundary]）、未启用 `split_batches`。
4. **`load_state()` 不传 dataloaders ≠ 未恢复 dataloader 状态**：[accelerator.py:3678,3840][a-accel] 使用内部注册表 `self._dataloaders`（由 `prepare` 填充）。"没传参数"不是证据（C）。真实缺口按 P0-5 收窄：该分支未被显式启用只说明这一条特定路径，**不能推出所有 sampler/RNG 状态都未保存**；结论是"当前证据不能承诺精确恢复"，需要同 world_size 的中断对照与改变 world_size 的显式拒绝/降级策略。
5. **"native precision / replay 多卡不可达"必须拆成不同的合同**（原表述过宽，此处更正）。三类合同条件不同，**不能一概等同，也不能因为"默认 off"就放开现有拒绝**：
   - **native FP16 合同**：`world_size != 1` 即记 error（[native_precision.py:66-67][l-native]），由 [accelerator.py:167-170][l-accel] 用真实 world_size 执行。
   - **islands（FP16/FP32）合同**：与 native 同一条执行点（`native_enabled(args) or islands_enabled(args)`，[accelerator.py:167-170][l-accel]）。
   - **adaptive 恢复合同**：多进程或 `gradient_accumulation_steps != 1` 直接 `raise ValueError`（[resume_checkpoint.py:91,202][l-resume-ckpt]；[resume_runtime.py:45-46][l-resume-runtime]）。
   - **通用 window replay**：**没有任何世界规模前置**，走普通 epoch 路径且无条件执行（[loop.py:819-854][l-loop-replay]）。`--adaptive_precision` 默认 `off`（[training_config.py:18-23,38-43][l-adaptive-cfg]），`resolved_mode` 在 `off` 时直接返回 `off`，`native_enabled/islands_enabled` 均为假 ⇒ **普通 adaptive_precision=off 的多进程路径同样可达窗口重放**。
   因此正确写法是：**"多卡 + adaptive native/islands 精度与 adaptive 恢复"被现有闸门拒绝（保留该拒绝）；而通用 window replay 在普通多进程路径上可达，是 §10 的首要风险项。**
6. **"容器没有 root forward" 的准确表述**：DDP 可以对任意参数注册 hook；问题在**reducer 生命周期与实际 adapter 调用边界不匹配**（DDP 期望由 wrapper 的 forward 驱动 bucketing 与 `prepare_for_backward`，而本项目 adapter 前向由被 monkey-patch 的冻结 DiT/TE Linear 驱动）。结论仍是"不值得为 adapter 包 DDP"，但理由要写对。

## 9. 风险 / 成本矩阵与首批接口

| 候选 | 层面 | 收益 | 成本/影响 | 主要失败模式 | 回退 | 最小验收门槛 |
| --- | --- | --- | --- | --- | --- | --- |
| P0-0 窗口/跳过/scaler 一致性 | 梯度主干 | 消除可达的静默分叉与挂死 | 中高：需要状态协议（窗口健康、跳过决策、scaler 决策的跨 rank 一致来源） | 只补归约会把分叉挪位；用 rank-local 判定做"安全回退"仍会分叉 | 保持现状的拒绝/统一终止；不用局部判定兜底 | 2 进程 Gloo 四类场景（§10） |
| P0-1 选卡/进程数对齐 | 控制面 | 消除"选 N 卡只起 1 进程" | 纯判定 | 语义歧义（显式填 1 是单卡还是默认值） | 加来源标记，保留旧行为 | 单测：默认推断/冲突拒绝/单卡不受影响 |
| P0-2 启动期 reduce 探针 | 控制面（**引入运行期 collective**） | 通信错误前移 | 启动阶段多一次集合通信；失败即中止会改变启动结果 | 异构机上探针失败被当致命 | 显式开关，默认关；失败统一终止 | 单测 + 一次 2 进程 Gloo |
| P0-3 清单字段 | 控制面 | 实验可回溯 | 小 | schema 冲突 | 字段可选追加 | 单测：字段存在且 rank0-only |
| P0-4 原因码 + 传真实 world_size | 控制面 | 补上 CLI 直连漏判 | 中 | 原因码与判定漂移 | 单一函数产出 | 单测逐 code；CLI 直连断言 |
| P0-5 恢复语义显式化 | 控制面 | 不再宣称精确恢复 | 小 | 与用户预期冲突 | 文档级声明 | 文档 + Gloo 样本序列比对 |
| P1-1/1-2 checkpoint 三阶段、resume bundle | 控制面 | 可恢复性可判定 | 中高 | 阶段顺序错位导致 hang / 校验过严误拒 | legacy 降级 `exact_resume=False` | 2 进程 Gloo + 故障注入 |
| EXP-FSDP2-BASE | 模型/显存 | 冻结基座分片**可能**降低驻留显存（未验证） | 高：all-gather、wrap 粒度、参数别名；**参数同步归属未定**（FSDP2 vs 现有手动同步） | 双重归约或出现无人负责的参数、注入顺序被误设为既定方案 | 保持 fail-closed，不接默认；归属未定前不落地 | 先定归属并只做 CPU/Gloo 协议与 mock 检查；真实门槛另列 CUDA/NCCL 数值+保存恢复+显存 |
| EXP-TP-COLROW | 模型/显存 | 单卡容量不足时的候选之一（未验证） | 高：小消息延迟主导、准入面窄 | collective 次数爆炸、GQA/融合层错切 | 保持 fail-closed | TP1 逐位一致；TP2 rel-L2 + 通信次数统计 |

首批落点（暂不改训练实现；后续准入/启动改动会改变部分配置的拒绝行为，应保留单卡默认回归门槛）：[launch.py][l-launch]（P0-1）、[gradient_sync.py][l-gsync]（P0-2 探针位置，`prepare_network_for_manual_gradient_sync` 之前）、[bootstrap.py][l-bootstrap] + [compat_matrix.py][l-compat]（P0-4）、清单写入点（P0-3）。**拟新增**（尚无文件，不建链接）：`library/training/multigpu_admission.py`（纯函数原因码表）与窗口一致性状态协议模块。

## 10. 未验证项、验证契约与争议

**首要项（P0-0）静态风险，未实测**：窗口健康判定是 rank-local（[loop.py:1109][l-loop-1109]）→ 边界先归约梯度（[loop.py:1143-1151][l-loop-reduce]）→ `drop_update = skip_update or force_skip or not loss_is_finite`（[loop.py:1217-1228][l-loop-step]）：为真只 `zero_grad`，**不进入** `optimizer.step()`；否则走 `AcceleratedOptimizer.step()`，而它才执行 `scaler.step()` + `scaler.update()`（[optimizer.py:162-173][a-opt]）。后果：**只有真正进入 `optimizer.step()` 的 rank 才推进 scale / growth tracker**，只做 `zero_grad` 的 rank scale 不变；即使两边最终 `update_skipped` 都为 True、replay 次数也相同，**scaler 状态仍可能已分叉**，后续迭代梯度量级随之不同。可达子情形：(a) loss 非有限但梯度有限 ⇒ 该 rank 跳过更新，另一 rank 正常 `step`；(b) 梯度本身非有限 ⇒ 边界均值归约让各 rank 看到同一份 NaN，且无 scaler 时有 `reduce(max)` 守卫（[scaler_ownership.py:161-174][l-scaler-owner]），跳过决策因此对称——**但这只覆盖"梯度为 NaN"这一支，不能推广为"NaN 经 all_reduce 通常安全"**；(c) 各 rank 窗口长度不等（§7 R4）⇒ 归约次数本身不等。

**验证契约（本批未执行）**：2 进程 Gloo、CPU 假模型，至少四个场景——①坏 loss + 梯度有限；②坏 loss + 非有限梯度；③accum 尾窗（epoch 末尾不足一个窗口）；④正常窗口对照。每个场景同时比对：(1) 每步 `update_skipped` / replay 次数 / 提交窗口序列；(2) `GradScaler` 的 scale 与 growth tracker；(3) optimizer `step` 计数与 `param_group` lr；(4) 参数逐位一致。**任一项不同即判该路径未通过**，不能只比 replay 次数或只比最终参数。同一检查顺带覆盖 `scale_weight_norms`（§8.1）——作为观测点，无预设失败。

**修复方向（不承诺"一处 reduce"）**：需要一致性状态协议——窗口健康判定与跳过决策由单一来源（如 rank0 广播或显式 `reduce`）统一，并让 scaler 的推进路径在"跳过"分支上也对称；失败时**统一终止或保持现有拒绝**，不得把已知可能分歧的 rank-local 判定当作安全回退。

其余未关闭项：
- 参考仓结论仅对 `92f3c07cb` 成立；其双进程 Gloo 契约只有 2026-08-18 历史报告，脚本已删，**本轮不可复核**。
- 本批无 CUDA/NCCL 真机证据；S1（等价性）与 S2（工程缺口）仍未验收。§6.2/§6.3 的两个 artifact 支撑项只支持**口径比值**，不支持单向峰值或 PP 安全保证。
- 未逐一枚举：参考仓 `distributed_resume_state` 的全部 restore 调用点、FSDP2 state dict 逐行行为、多 rank 取消/终止链传导；本项目其它族 preview 的分布式分支。
- 争议保留：`num_processes` 语义（"每机进程数" vs "每机 GPU 数"）两边都用推断表达，若 UI 允许显式填 1 表示单卡，需要来源标记消歧。
- 已知注释级错误（不影响结论）：[scaler_ownership.py:187-188][l-scaler-owner] 称"The scaler already all-reduces `found_inf` across ranks"，而 torch 2.10 的 `amp/grad_scaler.py` 无该逻辑；真正保证来自"归约在 `optimizer.step()` 之前"。

## 11. 许可证与引用边界

- 参考仓 [LICENSE][r-license] = **PolyForm Noncommercial 1.0.0**（声明位置见 [pyproject.toml:10][r-pyproject]），无 NOTICE：允许符合条件的非商业使用、修改与分发，须保留条款，不可擅自再许可。
- 本项目为 **MIT**（[LICENSE][l-license]）。本轮**只提炼机制与接口契约**，不导入参考代码、不做近似改写；若后续确需复用实现，须先就许可合规与署名做出决定。
- 本文所有代码位置均为只读引用，用于说明机制与反例。

## 12. 结论

1. **DDP 方向不变，但不要把参考仓 DDP 当模板**：两边正确性条件不同（optimizer 参数集合 vs 被包装模块），硬换会丢 `after_backward` 语义与 presence 掩码。
2. **优先级：先修梯度主干的一致性（P0-0 窗口/跳过/scaler），再做控制面 P0-1..P0-5，之后才谈真实短训与 FSDP2/TP 实验**。真实短训不应在 P0-0 之前执行。
3. **参考仓最大的价值是它踩过的坑**：包装边界与可训练参数集合不一致（R1）、rank-local 判定破坏 collective 对称（R2）、PG 失败降级（R3）、证据账本腐化（§3.2）。
4. **FSDP2/TP 保留为有边界的中高风险实验候选**（EXP-FSDP2-BASE / EXP-TP-COLROW），收益与风险都写清；任何收益在目标机实测前不得量化。
5. **口径纪律**：微基准只支持它自己的口径比值；历史计数是窗口平均；matmul 表是手工均值观测。三者都不能升级成"下限""峰值"或"安全保证"。

<!-- 本地源码（相对本文件 docs/findings/） -->
[l-launch]: ../../library/runtime/launch.py#L37
[l-accel]: ../../library/runtime/accelerator.py#L159
[l-precision]: ../../library/training/precision_policy.py#L43
[l-trainsession]: ../../library/training/train_session.py#L163
[l-cliargs]: ../../library/training/cli_args.py#L659
[l-bootstrap]: ../../library/training/bootstrap.py#L1049
[l-bootstrap-compat]: ../../library/training/bootstrap.py#L162
[l-compat]: ../../library/training/compat_matrix.py#L241
[l-preflight]: ../../web/services/config/preflight_compat.py#L191
[l-gsync]: ../../library/training/gradient_sync.py#L298
[l-gsync-prepare]: ../../library/training/gradient_sync.py#L298
[l-gsync-params]: ../../library/training/gradient_sync.py#L227
[l-gsync-sync]: ../../library/training/gradient_sync.py#L320
[l-loop-replay]: ../../library/training/loop.py#L819
[l-loop-reduce]: ../../library/training/loop.py#L1143
[l-loop-1109]: ../../library/training/loop.py#L1109
[l-loop-step]: ../../library/training/loop.py#L1217
[l-loop-boundary]: ../../library/training/loop.py#L995
[l-loop-skip]: ../../library/training/loop.py#L801
[l-scaler-owner]: ../../library/training/scaler_ownership.py#L161
[l-sched]: ../../library/training/schedulers.py#L245
[l-ckpt]: ../../library/training/checkpoints.py
[l-logdispatch]: ../../library/training/log_dispatch.py#L36
[l-native]: ../../library/training/adaptive_runtime/native_precision.py#L66
[l-resume-runtime]: ../../library/training/adaptive_runtime/resume_runtime.py#L45
[l-resume-ckpt]: ../../library/training/adaptive_runtime/resume_checkpoint.py#L91
[l-adaptive-cfg]: ../../library/training/adaptive_runtime/training_config.py#L18
[l-reg]: ../../networks/lora_anima/regularization.py#L16
[l-anima-preview]: ../../library/anima/training.py#L986
[l-pyproject]: ../../pyproject.toml#L8
[l-baseline]: multigpu_v100_nvlink_baseline_20261107.md
[l-artifact]: ../../output/tests/parallel_transport_v100_nvlink_20261107.json
[l-artifact-ar]: ../../output/tests/parallel_transport_v100_nvlink_20261107.json#L242
[l-artifact-pp]: ../../output/tests/parallel_transport_v100_nvlink_20261107.json#L261
[l-artifact-notes]: ../../output/tests/parallel_transport_v100_nvlink_20261107.json#L639
[l-artifact-d2h]: ../../output/tests/parallel_transport_v100_nvlink_20261107.json#L239
[p-plan]: ../proposal/multi_gpu_training_plan.md
[p-index]: README.md

<!-- 参考仓源码（完整绝对路径；前缀 backend/core） -->
[r-design]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/devtools/docs/multi_gpu_training_design_current.md
[r-inventory]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/devtools/docs/Reports/cuda_release_inventory_2026-08-18.md#L46
[r-strategy]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/parallel/distributed_strategy.py#L224
[r-runtime]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/parallel/distributed_runtime.py#L163
[r-distributed]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/parallel/distributed.py#L106
[r-ddp-wrap]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/parallel/distributed_runtime.py#L283
[r-pg]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/parallel/distributed_process_group.py#L290
[r-queue]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/services/training/queue_worker_launch.py#L101
[r-breakdown]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/services/training/training_step_breakdown.py#L390
[r-prep-utils]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/trainer/trainer_model_prep_runtime_utils.py#L57
[r-opt-setup]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/trainer/trainer_execution_optimizer_setup.py#L393
[r-inject]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/trainer/trainer_prepare_adapter_inject_mixin.py#L42
[r-inject-te]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/lora_core/lora_injector_inject.py#L48
[r-tp-mixin]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/trainer/trainer_prepare_tensor_parallel_mixin.py#L117
[r-tp-profile]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/parallel/tensor_parallel_runtime_profile.py#L76
[r-specs]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/parallel/specs.py
[r-pp]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/parallel/pipeline_parallel.py#L15
[r-scorecard]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/parallel/multi_gpu_parallel_scorecard.py#L40
[r-step-compute]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/training_loop/step_compute.py#L750
[r-accum]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/training_loop/accumulation_batches.py#L38
[r-gradient-mixin]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/training_loop/training_loop_gradient_mixin.py#L139
[r-dataloader]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/parallel/distributed_dataloader.py#L219
[r-interlock]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/configs_parallel_interlock.py
[r-test-strategy]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/tests/test_distributed_strategy_unit.py#L31

[r-ckpt-save]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/trainer/trainer_collective_state_save.py#L51
[r-ckpt-transport]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/trainer/distributed_checkpoint_transport.py#L71
[r-resume-state]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/backend/core/lulynx_trainer/trainer/distributed_resume_state.py#L20
[r-license]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/LICENSE#L1
[r-pyproject]: /mnt/SD_Dataset/训练器相关/lulynx-trainer/pyproject.toml#L10
[l-license]: ../../LICENSE

<!-- 只读依赖环境（完整绝对路径） -->
[a-accel]: /mnt/SD_Dataset/训练器相关/krea2-webui/.venv-v100/lib/python3.13/site-packages/accelerate/accelerator.py#L3259
[a-opt]: /mnt/SD_Dataset/训练器相关/krea2-webui/.venv-v100/lib/python3.13/site-packages/accelerate/optimizer.py#L162
[a-sched]: /mnt/SD_Dataset/训练器相关/krea2-webui/.venv-v100/lib/python3.13/site-packages/accelerate/scheduler.py#L54
