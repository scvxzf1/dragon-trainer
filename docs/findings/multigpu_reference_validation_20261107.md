# 多卡全流程参考点验收：实现、分层证据与逐点裁定（集成报告）

状态：实验 / 研究分支 `codex/multi-gpu-training`，未改任何生产默认值
适用版本：HEAD `2e7da55e` + 12 条未提交差异（字节清单见 §1）；验收契约冻结在 `80dfdb02…`
入口命令：见 §10 复现步骤（CPU/Gloo 层可自跑；GPU/真实训练层需要显式门与已批准预算）
相关代码：`library/training/multigpu/`、`library/training/{loop,checkpoints,train_session}.py`、
`library/datasets/buckets.py`、`library/runtime/{launch,accelerator}.py`、`bench/multigpu_reference/`、
`tests/test_multigpu_*.py`

> **增量版本（§12–§17，2026-11-08 t37）**：本报告首版（§0–§11，405 行）描述 t6 轮读取的字节基线，
> 其数字与裁定在后续取证中被**更正**，不是被删除。更正集中在两处：**取数顺序三层缺陷**
> （§12）使 4 个 S-d/S-g 数字作废（§13）；**P0-5 / P1-2 的 V-REAL 数值面由 `failed` 改为
> `unverified`**（§13，待 t35 结论填入占位）。另新增 §14（合同只能靠 env 门控满足，F-T31-3）、
> §15（验证体系盲区，F-T33-2）、§16（流程与 captain 侧缺陷账）、§17（F-T3-1 旧结论降级）。
> **§18–§23（收口，2026-11-08 t46）**：按用户指示「现在就收口」对契约做**正式修订 v2**，解除数学不可达的
> S-d/S-g 数值门并换为 C1–C4；**P0-5 / P1-2 的 V-REAL 数值面已由 `unverified` 改判为「通过（v2 判据）」**（§18），
> 附旧数字作废清单（§19）、阈值来源与失效教训（§20）、`pinned_order` 正名（§21）、captain 判决失误清单（§22）。
> 增量章节落盘期间工作树由并行任务（t32/t34/t36）持续变动，因此这里**不绑定具体条数**；
> 本报告自身只碰 `docs/`（4 个 `M` + 1 个 `??`），未改 `library/`、`tests/`、`bench/`、`configs/`、
> 契约与 `output/`。§1 的字节绑定仍如实描述 t6 轮的读取基线，不回填。

契约：[多卡全流程参考点验收契约（DDP 优先）](../proposal/multigpu_reference_validation_contract.md)
（`sha256:80dfdb02…d76a2`，981 行 / 79,443 B，自冻结以来未被改写）。
使用说明与代码事实：[训练指南](../guidelines/training.md)（多卡节）。
原始未提交研究报告（保留、未改写）：[Lulynx 多卡实现对照](lulynx_multigpu_comparison_20261107.md)、
[本机 2×V100 NVLink 基线](multigpu_v100_nvlink_baseline_20261107.md)、
[多卡训练研究计划](../proposal/multi_gpu_training_plan.md)。

---

## 0. 一句话结论（先读这一节）

**本轮不是"多卡训练验收完成"。** 按契约的三态口径，实际状态是：

- **已实现 + 已验收（有本契约下真实执行的命令、退出码与 artifact）**：P0-0（V-CPU + V-GLOO +
  V-CUDA-SMALL 的 scaler 分支）、P0-1、P0-2、P0-3、P0-4、P1-1、P1-3，以及作为**实验候选被否决**的
  EXP-FSDP2-BASE / EXP-TP-COLROW、作为**回归项**的 PP 拒绝。
- **实现已修、但按契约判据判定失败（不得记"已验收"）**：**P0-5 / P1-2 的"同 world 精确恢复"**。
  RUN-C 的 S-g 门 `3.434e-03`、single-vs-dual 的 S-d 门 `1.219e-02`，均超**更宽的** fp16+GradScaler
  门 `2e-3`（bf16 门 `1e-3` 亦超）。阈值**未放宽**，需 captain 二选一裁定（§6.1）。
  > ⚠ **本节该条已被更正（2026-11-08，见 §13）**：这两个数字（以及 t22 的 fp32 对应值
  > `1.193e-02` / `3.106e-03`）来自**违反合同 §13 V-2 第 2 条（未关 shuffle）**的配置，
  > 属**取数错位所致，不可解释为协议差异**。**P0-5 / P1-2 的 V-REAL 数值面现记
  > `unverified`（未判定），不是 `failed`**；§6.1 的 (a)/(b) 二选一已被后续证据取代；
  > **后续（2026-11-08 t46）**：契约已出**修订 v2**，本点在 C1–C4 判据下改判为「**通过（v2）**」，见 §18。
  > （fp32 路径已由 t22 执行过，其数字同样作废）。最终结论待 t35 填入（§13 占位块）——**该占位已于 2026-11-08 关闭：pinned 复测由 t43/t45 完成，结论见 §18（通过）。**
- **未执行 / 环境受限（unverified，不得当作通过）**：R-b 逐 rank 真机留痕、恢复清单三字段的真机
  留痕、跨桶/其它 canonical 桶/尾窗/4032 族、fp32 参考路径、DeepSpeed/SP 拒绝回归（本树无代码无用例）、
  **4096 固定 shape 候选作为 L3 输入**（该候选未进入任何真跑，见 §7）。
- **真实短训确实执行过**：RUN-A/B/C 共 **50 次有效更新 / 372.74 s**，判别实验再 **+20 次**
  （累计 70，用户批准上限），**入口与 CPU 用例一律不计作真训练**（§5）。

一句话的可交付边界：**协议面在 CPU/Gloo/小模型上成立且可复现；真实 Anima 双卡能跑、能存、能续，
但"数值等价 / 逐位恢复"两项在 V100 fp16 上既没有通过、也没有被证明为协议缺陷**——缺的是
fp32 参考路径（判别力）与 per-rank 逐位留痕，不是缺入口。

---

## 1. 字节绑定（结论绑在哪一版代码上）

| 项 | 值 |
| --- | --- |
| 分支 / HEAD | `codex/multi-gpu-training` / `2e7da55e` |
| `git status --short` | **12 条**（9 个 `M` + 3 个 `??`；逐条见 §10 第 0 步）。**这 12 条是本报告的输入基线**；本报告自身的文档改动（4 个 `M` + 1 个 `??`）不在这 12 条里，追加后为 16 条，仍无源码改动 |
| 契约指纹 | `80dfdb02c4fb81e1cc4e798560aeb5533a3ca09f1ade855c3c7ae1b0a67d76a2`（981 行 / 79,443 B） |
| 契约 README 指纹 | `3c2cc7f21ae1ef02e63f9c8243e50c8601a897d21184d7a0b94a5f63a0ca137f` |
| 最后一次独立 review | review-round-3（`t21`）：**verdict=pass**，但**P0-5 的 V-REAL 数值面明确不在其通过范围内**，原样转交 captain |

本轮读取的源码字节（用于对照 t18/t19/t21 各轮 review）：

```
f4ae25e331c65f9175edd08e71a631ae314bb95d20c033aede0c727704aa104d  library/training/multigpu/identity.py
e72ba97632164b62038fd341a9ab8d6e7733492a153f11560f63851a924ce245  library/training/multigpu/resume_protocol.py
2ac180881026a82664337d618c3383b2c9d1ecd37090d13037edd80975008b35  library/training/multigpu/rank_digest.py
9c1a1ef880407b0ba8e9b15d9e6ef60f11b853164a58e51f0d2c6f2ca99718b3  library/training/multigpu/checkpoint_protocol.py
6d1bbd36e873c63d235082fc87d08f10f6cbcbc6c6b717866816f9d0a620b479  library/training/multigpu/cursor.py
ed2e98d775ec2b5978e190d5a012eca0eff5fc85acb299fe2ae23b1b0a7adf36  library/training/multigpu/rng_evidence.py
6c008a7138b168dbdfd17d1131fb69193e1828f9babeb5a7d744fb01d1bde39d  library/training/multigpu/window_protocol.py
ca444d5935820bdd75f8ac1cf17f4f3b1f659b7042fc12a8075915734ec27651  library/training/multigpu/admission.py
450657cab204482778b02cff2e85563493230e5d7abad959f67eb9cc18dc291c  library/training/multigpu/agreement.py
28fd3d0d5a692c5a4efe83f4f8ad5a10d19d1476366615f33ace3d63569df33e  library/training/checkpoints.py
f70f5bd01032fedede551f19a9bb3c748bec63c037ab66d206b9625bf39c8b26  library/training/loop.py
b2c9b133c091a79f9d3243e1d9cc72111e323b9aacf0775603456da1037ad2d6  bench/multigpu_reference/hot_run.py
```

### 三态定义（与承接账本一致，全文只用这三个词）

- **已实现**：源码/测试在工作树可定位，逻辑成立。**不等于**可用。
- **已验收**：有本契约下**独立执行**的命令 + 退出码 + 耗时 + artifact，并经独立 review 裁定 pass。
- **未执行**：没有任何该层实际执行证据（含"只有入口""只有单测""skip""xfail"四种情形）。

三态互斥、不可互相替代。**报告写完不等于能力通过**；本文件只做登记，不新增授权。

---

## 2. 环境契约（实测，不外推）

`output/tests/multigpu_reference_validation/t6-faultinj-cuda-20261108/environment_identity.txt`
（2026-11-08T01:24:27+08:00）：

| 项 | 值 |
| --- | --- |
| Python | 3.13.9（`krea2-webui/.venv-v100/bin/python`） |
| torch | **2.10.0+cu129** |
| accelerate | 1.13.0 |
| pytest | 9.1.1 |
| NCCL | 2.27.5（CPU 轮次为 `cuda-invisible`，未初始化） |
| 硬件 | 2× Tesla V100-SXM2-16GB（SM70，NVLink），驱动 580.196.07 |

**硬规则**：所有结论只对 torch 2.10.0 / accelerate 1.13.0 成立，**不得外推 torch ≥ 2.12**
（`pyproject` 声明的下限高于本机实测版本）。CPU/Gloo 证据一律不充当 CUDA 验收；小模型不充当
Anima 真实训练；真实短训不充当性能结论。

---

## 3. 逐参考点矩阵（P0 / P1 / EXP）

裁定列取值：**已验收** / **判定失败** / **已实现待观测** / **实验否决** / **未执行（unverified）** /
**本轮不适用**。每条都给源码位置 + 层级 + 命令/退出码 + artifact。

### 3.1 P0 控制面

| 参考点 | 实现（源码） | 层级与级别 | 实际命令 → 退出码 | 数字与 artifact | 裁定 |
| --- | --- | --- | --- | --- | --- |
| **P0-0** 窗口健康 / 跳过 / scaler 跨 rank 一致 | `multigpu/window_protocol.py`（`agree_step` 只剩 `loss_nonfinite/gradient_nonfinite/force_skip` 三 rank-local 输入，返回世界 OR）、`multigpu/agreement.py`（`agree_any`）、`loop.py`（`agree_step` → `optimizer` → `negotiate_boundary_error` → `assert_scaler_symmetry`）、`scaler_ownership._any_rank_bad` 显式 SUM>0 | V-CPU + V-GLOO + V-CUDA-SMALL | `pytest tests/test_multigpu_window_protocol.py -q -rs` → **exit 0**；`pytest … tests/test_multigpu_admission.py tests/test_distributed_gradient_sync.py -m 'integration and not hardware'` → **exit 0** | window_protocol **19 passed / 75.89 s 无 skip**；独立 Gloo harness **9/9 场景四项全等**（`t3-gloo-window-admission-20261107/harness/comparison.json`）；复跑 8 Gloo passed / 61.51 s（`t15-gloo-reverify-20261107/`）；V-CUDA-SMALL：step1 rank1 非有限 loss → 两 rank 同丢弃、`assert_scaler_symmetry=true`、**scale 65536→32768 两 rank 相同**（`t6-faultinj-cuda-20261108/harness/vcuda_out/rank{0,1}.json`） | **已验收**（V-REAL 的 scaler 分支**未覆盖**：三段真跑零 skip/replay/discard，按契约 §13 只能引用 V-GLOO 证据，不得写"真实运行验证了 scaler 对称性"） |
| **P0-1** 选卡 ↔ `num_processes` 双向 fail-closed | `multigpu/admission.py::resolve_admission`（`<` 与 `>` 一律具名拒绝）、`library/runtime/launch.py::configure_accelerate_for_gpu_selection`（抛同码） | V-CPU | `pytest tests/test_multigpu_admission.py tests/test_launch_config.py -q` → **exit 0**（v1 t17 review 复跑） | **59 passed / 6.41 s**；独立准入探针 **11/11**，其中方向 B（2 卡 4 进程）与（1 卡 2 进程）由"放行"改为"具名拒绝"（`t15-gloo-reverify-20261107/probe_admission_summary.json`）；`resolve_world_size` **6/6**；两方向拒绝用例含在 g2 的 225 passed 内 | **已验收**（真实多卡**启动**由 RUN-B/C 证明可用；选卡冲突**拒绝**路径只在 CPU 探针验证，无真机冲突用例） |
| **P0-2** 启动期 collective 探针 | `multigpu/probe.py`（默认关、单进程短路、失败抛 `StartupProbeError`、不降级、不写清单） | V-CPU + **V-GLOO 首次双进程** | 由 t3 轮 harness 执行（`startup_probe_out/`） | 默认关：**0 collective**、清单仍写；env 开启 / CLI 开启：两 rank 均执行、`reduced_total=2.0`；两种失败（collective 抛错 / 世界大小不足）**同名 `StartupProbeError`、不写清单、不降级**（`t6-faultinj-cuda-20261108/harness/startup_probe_out/startup_probe_report.json`） | **已验收** |
| **P0-3** 运行身份清单 | `multigpu/identity.py`（rank0 一次定长 `all_gather` 收全 rank 摘要、`configured_world_size` 与 `world_size` 分列、`reduction_mode`/bucket 由真实 `GradientSyncResult` 回填、写失败 = 具名 `multigpu_identity_write_failed`） | V-CPU + V-GLOO + **V-REAL** | `pytest`（P0-1 同命令）+ 真机两段 | Gloo：`world_size=2` 两 rank 摘要齐（`startup_probe_out/{env_on,cli_on}/run-…/multigpu_identity.json`）；真机：RUN-B/C `multigpu_identity.json` 记 `reduced_parameter_count=392`、`gradient_buckets=2`、`reduction_mode=async_bucket_all_reduce_mean(2 buckets)`、`mixed_precision.effective=fp16`（`source: precision_policy:sm_70<8 has no native bf16`） | **已验收**（**不含** P1-2 的恢复三字段，见下） |
| **P0-4** 准入原因码 + 训练进程内真实 `world_size` | `library/training/bootstrap.py` 传 `resolve_world_size(env=os.environ)`；原因码单一函数产出 | V-CPU | 同 P0-1 + 独立探针 | 原因码逐条具名（`multigpu_single_gpu_multi_process` / `multigpu_num_processes_selection_mismatch` / `multigpu_num_processes_invalid` / `multigpu_launch_without_workers`）；`WORLD_SIZE` 权威优先（`WORLD_SIZE=4` + 显式 2 → 按 4 求值） | **已验收** |
| **P0-5** 恢复语义显式化 | `multigpu/resume_protocol.py`（`validate_bundle` + `guard_resume`，跨 world 拒绝 / 显式降级 + 具名码） | V-GLOO 成立；**V-REAL 判定失败** | `pytest tests/test_multigpu_resume_protocol.py`（t12 g1）→ **exit 0** | **跨 world_size 拒绝与显式降级成立**（`chain_broken_state/chain_report.json`：`multigpu_resume_topology_mismatch` / `…_degraded`）；**同 world 精确恢复的 V-REAL 数值门失败**：S-g = `3.434e-03` > fp16 门 `2e-3`、bf16 门 `1e-3`（`t13-anima-hot-20261108/evidence/compare.json`，`within_tolerance_by_key={bf16:false, fp16_gradscaler:false}`） | **判定失败 → 待裁定**（契约 §13 V-3 明确"超阈即 `needs_revision`，不得以影响很小放过"，且观测后不得调阈值）。出路只有 §6.1 的 (a)/(b) —— **本格裁定已被 §13 更正为 `unverified`，理由：该数字来自未关 shuffle 的配置（取数错位）；再经契约修订 v2，现记「通过（v2 判据，§18）」** |

### 3.2 P1 checkpoint / 恢复 / 账本

| 参考点 | 实现（源码） | 层级 | 实际命令 → 退出码 | 数字与 artifact | 裁定 |
| --- | --- | --- | --- | --- | --- |
| **P1-1** 三阶段发布（全 rank capture 判决 → before_publish+barrier → rank0 发布 → publish 判决） | `multigpu/checkpoint_protocol.py`（`AGREEMENT_KEYS` 两处同键同序、rank0 recover→rename→marker→bundle、peer 失败 rollback、transport 类不追加 collective）、`library/training/checkpoints.py`（`maybe_save_step/epoch` 改**全 rank** 调 state 保存） | V-GLOO + V-CUDA-SMALL | `pytest tests/test_multigpu_checkpoint_protocol.py` → **exit 0** | **20 项 / 44 s**（对接前为 240 s/900 s 挂死）；`chain_happy/events-rank{0,1}.jsonl` 证明 collective 序列一致；类 I 局部采集失败与 rank0 发布失败均**全体同码失败 + 旧 checkpoint 存活**（`PRIOR-OK.marker`）；类 II transport 错误后 **collective=0、共享 tmp 与对端 shard 保留**；peer 死亡 **8 s 有界退出**（harness PG timeout 60 s，时延非生产值）；GPU 同构 `chain_gpu/` | **已验收** |
| **P1-2(a)** 版本化 per-rank resume bundle（version / world / backend / rank_devices / rank_local_keys + 逐项校验） | `resume_protocol.py` + `checkpoints.py::build_bundle` | V-GLOO | 同上 + `pytest tests/test_multigpu_cursor_wiring.py` | `chain_happy/t3-step00000010-state/multigpu_resume_bundle.json`；缺 rank / 多 rank / 无 `rank_devices` → `multigpu_resume_bundle_invalid`（**多出 rank 也拒绝**） | **已验收（结构面）** |
| **P1-2(b)** exact resume（真实 RNG 证据 + 恢复时刻二次核验） | `rng_evidence.py`（真实 `random_state / numpy_random_seed / torch_manual_seed / torch_cuda_manual_seed` 键映射）、`guard_resume(verify_rng_files=True)` 在恢复时刻重读 | V-GLOO + **V-REAL（单点）** | t12：`pytest` 四门 + `harness/guard_resume_falsification.py` → **exit 0** | 假 exact 已消除：完好目录 `exact=True`；删 `random_states_1.pkl` → `exact=False` + `multigpu_resume_rng_evidence_missing`（点名 `rank [1] of world_size=2`）；垃圾 pkl → 同码 + `multigpu_rng_evidence_corrupt`；**同一坏目录在 `verify_rng_files=False` 仍返回 `exact=True`**（证明判别点就是新增的时刻核验，不靠 accelerate 吞错）；缺流 + 游标漂移两码并存；legacy 词汇 pkl 可读且标 `schema=legacy`。成本：**2.84 ms/次、0 collective、RSS +288 kB/20 次**，每次恢复只调 1 次、不在步路径上。真机：`t13b-run-c/…-state` 两 rank RNG 文件 2/2 齐备、`guard_resume → exact=True, codes=[]` | **实现已修、判定未通过**：S-g 数值门失败（见 P0-5）+ 恢复清单三字段无真机留痕 ⇒ **不得记"已验收"** —— **本格裁定已被 §13 更正为 `unverified`（数值面作废原因同上：取数错位）；"恢复清单三字段无真机留痕"这一子项不受影响，仍记"已实现待观测"；**数值面再经契约修订 v2 记「通过（v2 判据，§18）」** |
| **P1-2(c)** loader cursor 真实生产链路 | `multigpu/cursor.py`（`expected_position` / `resume_skipped_batches` 唯一生产者）、`train_session.verify_resume_cursor` 走 `guard_resume(..., rank=本rank, resume_cursor=期望位置)`、`COMPARED_KEYS=("sampler_position","epoch","rank")` | V-GLOO | `pytest tests/test_multigpu_cursor_wiring.py -q` → **exit 0** | 差 1 已闭合（旧 `initial_step-1` 使计数与真实 index 差一，恢复后 fixture 钉错样本）；真机两 rank cursor 逐字段相同（`sampler_position=10`、`batches_per_epoch=12`、`epoch=1`），三处一致（rank0/rank1 文件 + bundle `extra.rank_cursors`） | **已实现 + V-GLOO 已验收**；"per-rank optimizer 逐位对比"未测（见 R-a） |
| **P1-3** 带证据指针的能力账本 | `output/tests/multigpu_reference_validation/t6-faultinj-cuda-20261108/capability_ledger.md` | 文档件 | 人工核对 | 每条带 state ∈ {supported, rejected, experimental, unverified} + 指针 + 日期；并含「集成阶段刷新（t18）」18 行表与「P0-5/P1-2 裁定待 captain」表 | **supported（本报告即在其基础上集成）** |

### 3.3 隔离实验与明确不适用项

| 项 | 实现 | 实测 | 裁定 |
| --- | --- | --- | --- |
| **EXP-FSDP2-BASE** | `bench/multigpu_reference/sharding.py`（`run_two_rank`、`materialize`、`fully_shard` 注入顺序） | NCCL 小模型真实跑：`value_rel_l2=0.0`、每参数 grad/Δθ `0.0`、归属 3/3 唯一且 `requires_grad` 不变、`buffers_not_sharded=true`、save/load roundtrip ok、wrap 顺序 `inject_adapters→wrap_shard→build_optimizer→register_sync_owner`；**峰值显存 dense 16.26 MiB = sharded 16.26 MiB（`sharded_lower=false`）**（`t6-faultinj-cuda-20261108/harness/fsdp2_tp_out/fsdp2_tp_report.json`） | **实验否决（不可采用）**：采用条件"峰值显存实测下降"未满足。注：`F-T3-4`（`aten.allclose` 拒绝 Tensor/DTensor 混比）**已修**，仓库自带 FSDP2 入口现可在 CPU Gloo 双进程真跑 |
| **EXP-TP-COLROW** | `bench/multigpu_reference/experiments.py`（column/row 替换 + 注入器识别） | TP1 与 `nn.Linear` **逐位一致**；column rel-L2 `0.0`、row `5.23e-8`（fp32）；fp16 下 row `3.60e-4`（≤2e-3）；通信 **每步 column 1×all_gather、row 1×all_reduce**；**仅单层**，buffer/掩码协议未覆盖 | **实验否决（不可采用）**：多层扩展性与 buffer/掩码协议未验收，通信次数只对单层测得 |
| DeepSpeed | 本树**无代码、无用例** | 无 | **本轮不适用 / unverified（具名缺口）**：不得引用"拒绝回归通过"——没有可执行的拒绝路径 |
| Sequence Parallel（SP） | 同上 | 无 | **本轮不适用 / unverified（具名缺口）** |
| Pipeline Parallel（PP） | `library/models/pipeline_parallel.py::validate_pipeline_parallel_config` 仍返回 `pipeline_parallel_runtime_unavailable` | `test_multigpu_admission.py` 用例通过（含在 225 passed 内） | **回归通过**（拒绝未被多卡改动放宽） |

---

## 4. 分层证据账（每层能证什么、实际跑出什么）

所有命令的解释器为 `/mnt/SD_Dataset/训练器相关/krea2-webui/.venv-v100/bin/python`，
CPU 轮次一律 `CUDA_VISIBLE_DEVICES=` 置空。

| 层级 | 命令（摘要） | 退出码 | 实测数字 | artifact 根 |
| --- | --- | --- | --- | --- |
| V-CPU / V-GLOO 门 g1 | `-m 'not hardware'`：`test_multigpu_{resume_protocol,cursor_wiring,checkpoint_protocol}.py` | **0** | **70 passed / 58.80 s** | `t12-f-t10-1-reverify-20261108/` |
| V-CPU / V-GLOO 门 g2 | 全量 multigpu 非 hardware（9 文件，含 `test_launch_config.py`） | **0** | **225 passed / 4 deselected / 161.21 s** | 同上 |
| 回归地板 g3 | 真名：`test_window_replay.py`、`test_fp16_nan_static_equivalence.py`、`test_training_checkpointing.py`、`test_training_resume_actions.py`、`test_training_resume_options.py` | **0** | **121 passed / 9.72 s** | 同上 |
| 硬件门 g4 | `MULTIGPU_REFERENCE_HARDWARE=1 -m hardware` | **0** | **1 passed / 2 skipped / 87 deselected / 10.42 s**；skip 具名（需外部 PG 的 FSDP2/TP 各 1 条，**不计通过、不计失败**） | 同上 |
| 最近一轮定向复跑 | `rank_digest + resume_protocol + dataset_cache_dir_sizes` | **0** | **61 passed / 17.78 s** | review-round-3（`t21`） |
| 收尾轮（v1 t25 承接） | verify1 / verify2 | **0** / **0** | **128 passed / 4 deselected / 80.91 s**；**116 passed / 12.44 s** | t2（CPU 范围，`CUDA_VISIBLE_DEVICES=` 空） |
| 固定 4096-token 前身失败层 | `tests/test_multigpu_reference_hot.py -m 'not hardware'` | **0** | 27 passed（t13 attempt 1）→ **38 passed / 1 deselected**（t14 修复后） | `t13-anima-hot-20261108/` |
| 真跑预算机检 | glob `*/anima_hot/summary.json`，断言单份 / 3 runs / 单次 ≤1200 s / 累计 ≤4800 s / 步数 ≤50 | **0** | `budget-ok [('RUN-A',10,151.025),('RUN-B',20,90.013),('RUN-C',20,131.704)]` | 同上 |

**数字口径警告**：t10 轮报的 **219 passed / g1 7 passed 已因字节漂移失效**（当时口径为
`-m 'integration and not hardware'`，且后续新增了用例）。**现行数字一律以 t12 重绑为准**，
其字节绑定记录在 `t12-f-t10-1-reverify-20261108/fingerprint_and_precheck.txt`（HEAD
`2987186c` + 32 条差异 + 逐文件 sha256 + 两卡/磁盘预检）。本轮结论绑定的则是 §1 的
`2e7da55e` + 12 条差异。

**skip 一览（逐条具名，一律不计通过）**：`test_fsdp2_matches_dense_on_forward_backward_optimizer_and_save_load`
（需已初始化进程组）、`test_tp_column_row_matches_dense_with_replicated_buffers`（同上）、
`test_bounded_anima_run`（`tests/test_multigpu_reference_hot.py:43`，设计上不自启训练）。

---

## 5. 真实短训：RUN 账本、50 更新与判别实验

证据根：`output/tests/multigpu_reference_validation/t13-anima-hot-20261108/`（`report4/5/6/7`）
与 `t17-determinism-20261108/`（`report8`）。全部经 `hot_run.plan_hot_run` +
`execute(runner=真实 subprocess)`，**不是**入口演示、不是 CPU 用例。

### 5.1 三段真跑（契约 §13 V-1）

| run | 设备 | 每设备 batch | 有效更新 | wall clock | 峰值显存 | 步时 | 资产 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RUN-A | cuda:0 单卡 | **2**（`--train_batch_size 2`） | **10** | **151.03 s** | 7.67 GB | 4.64 s/step | 单桶 896×1200（**4200 token 族**）、**24 样本** |
| RUN-B | cuda:0 + cuda:1 | 1 | **20** | **90.01 s** | 5.83 GB | 2.43 s/step | 同上 |
| RUN-C | cuda:0 + cuda:1 | 1 | **20**（10 commit + 10 resume） | **131.70 s** | 5.83 GB | 2.42/2.43 s/step | 同上 |
| **合计** | — | — | **50 / 50** | **372.74 s**（≤4800 s） | — | — | 单次上限 1200 s 全部满足 |

- 资产由**用户 2026-11-08 批准的一次性有界预处理**生成（项目自带
  `scripts.preprocess.resize_images` + `cache_latents --model_family anima --dtype float32`，
  非手搓 npz）；latent `(16,150,112)` fp32 = 268,800 元素/样本；TE cache 以 **只读 symlink**
  复用既有 24 个 `*_anima_te.safetensors`；**用户目录写入为零**（全部新产物在 run 根，145 MB）。
- 口径同源：三段均带 `--multigpu_seedable_sampler`、`--max_data_loader_n_workers 0`、
  `seed 20261107`、`lr_scheduler constant`、`caption_dropout_rate=0.0`、`attn_mode sdpa`；
  `mixed_precision` 请求 bf16 但 SM70 下由精度层改写为 **fp16 + GradScaler**（`identity` 里
  `mixed_precision.source` 记录改写原因）。
- L3 fixture 真机生效留痕：`ANIMA_MULTIGPU_NOISE_FIXTURE=l3_fixture.jsonl` +
  `ANIMA_MULTIGPU_NOISE_FIXTURE_IDS=cursor`，每 slot noise 长度 268,800 = 真实 latent 元素数；
  **fail-closed 实证**：slot 不足时真实训练直接抛 `noise_fixture_sample_missing: s00000024/s00000025`
  （RUN-B 第 2 次尝试），说明注入点确实在真实 dataloader 上、未命中不会静默通过。
  **注意**：fixture 需要的 slot 数 = `steps × world_size`（不是 `steps`），契约原文未写明（F-T13-8，已记入 hot_run 注释）。

### 5.2 判别实验（+20 更新，累计 70 = 用户批准上限）

`t17-determinism-20261108/report8.md`：det-a / det-b 是 RUN-C commit phase 的逐字复刻
（argv 差异仅 `--output_dir`，另 pin 了 GPU UUID）。

| 对照 | global rel-L2 | max_abs |
| --- | --- | --- |
| t13 baseline vs det-a | **3.225e-04** | 4.265e-04 |
| t13 baseline vs det-b | **3.988e-04** | 5.726e-04 |
| **det-a vs det-b（纯重复）** | **2.040e-04** | 3.595e-04 |

- 三次运行**没有一次逐位相同**（99.8% 元素不同）⇒ 噪声地板 ≈ **2.0e-4 ~ 4.0e-4**；
  契约的 `1e-3` 门只有噪声地板的 2.5–5 倍，`2e-3` 门 5–10 倍 ⇒ **该精度档的数值门处在可测性边缘**。
- 按预登记规则：重复跑**未达 1e-3**、也**不是 ≈0** ⇒ 落在**规则③**（区间 + 不确定性如实记录），
  **既不能判"协议缺陷"，也不能判"仅噪声"**。S-d 超噪声地板 30–60 倍、S-g 超 9–17 倍，但
  **S-g 只有 S-d 的 1/3.5**（恢复段差异远小于单卡 batch2 vs 双卡 batch1 的差异）。
- 结论：**唯一解法是 fp32 参考路径**（fp32 门 1e-6 远低于任何 fp16 噪声）。这是给 captain 的裁定输入。

### 5.3 组 R（rank 一致性）在真实双卡跑里的覆盖

| 子项 | t13 真机（无 per-rank 摘要） | t17 真机（t16 摘要上线后） | 现状 |
| --- | --- | --- | --- |
| R-a adapter 逐位 | **未测**（只有 rank0 `model.safetensors`） | **成立**：det-a/det-b 各轮 `multigpu_rankdigest_rank{0,1}.json` 聚合 sha256 相等（392 张量 / 5,160,960 元素），跨运行不同（与 §5.2 非确定性一致） | **已验收（真机，2 次）** |
| R-b scaler scale / growth | 部分（仅 rank0 `scaler.pt`：`scale=65536.0`，含 `_growth_tracker` 键） | scale 两 rank 相同；**`growth_tracker` 当轮为 `null`**（采集缺口 F-T17-3） | **实现已修（t18：`_growth_tracker`→旧公开键→`state_dict()`），逐 rank 真机留痕仍缺** ⇒ **unverified** |
| R-c optimizer step / lr / global_step / window 计数 | step/epoch/cursor 测了；per-rank optimizer 未测 | **成立**：两 rank `optimizer.step=10`、`lrs=[1e-4]`、`global_step=10` | **已验收（除 per-rank optimizer 逐位）** |
| R-d 跳过 / 重放 / 丢弃计数 | 测了：`skipping 10 steps: 0 full epochs`，无重放 | **成立**：两 rank `window_attempts=10 / replays=0 / discards=0 / resume_skipped_batches=0` | **已验收** |
| 附加 RNG | bundle `rng.ranks_covered=2`，两 rank 四流全 true | 同上 | **已验收** |
| window 观测 | — | `window_total_loss` rank0/rank1 天然不同 ⇒ 旧实现 `rank_digests.equal=false` 每次真跑都报"失败"（F-T17-2）；t18 已把 window_observation 项排除出等值判定（计数器仍参与） | **已修** |

### 5.4 组 S（single vs dual）门逐项

| ID | 判据 | 实测 | 结论 |
| --- | --- | --- | --- |
| S-0 | per-sample 随机输入 fixture 生效并落 traces | 显式 opt-in 门开、single/dual permutation 逐位一致（关门负对照不相等）；真机 fixture fail-closed 命中 | **成立 —— 但仅指"分片算术成立"，样本级对齐未验证（§17）** |
| S-a | 有效更新计数精确相等 | 10 / 20 / 20 | **成立** |
| S-b | 窗口 → 样本 id multiset 精确相等 | RUN-A(world1,batch2,10) 与 RUN-C commit(world2,batch1,10) 均为 `s00000000..s00000019`，`identical_set=true`、`identical_order=true`（`evidence/alignment.json`）；**诚实前提**：真实 batch 不带 per-sample id，对齐由确定性 sampler 公式 + 既有 cross-world permutation 同源推出 | **成立（带前提出处）** |
| S-c / S-f | loss 序列 / cos(Δθ) | 仅记录、不设门 | **观测项** |
| S-d | dtype-specific 数值门（fp16+scaler ≤2e-3） | **1.219e-02**（已作废，§19） | **failed（原判）→ `unverified`（§13 口径更正）→ 通过（契约 v2 / C1–C3，§18；正确数字 `1.5055e-04`）** |
| S-e | skip 决策序列逐窗一致 | 三段真跑零 skip，序列空集一致 | **成立但覆盖为空**（不得据此声称 scaler 分支在 V-REAL 被验证） |
| S-g | 10+10 续训 vs 连续 20 步逐位相等 | **3.434e-03**（已作废，§19） | **failed（原判）→ `unverified`（§13）→ 通过（契约 v2 / C4，§18；正确数字 `3.694e-05`）** |

---

## 6. 未通过项、限制与发布阻塞（原样可见）

### 6.1 唯一的裁定门：P0-5 / P1-2 数值面子项（**原为"待 captain 二选一"，2026-11-08 已被 §13 取代**）

> ⚠ **本节已被后续证据取代，保留只为不改写首版措辞**：选项 **(a)**（fp32 参考路径）**已由 t22 执行**，
> 但其数字（S-d `1.193e-02`、S-g `3.106e-03`）同样来自未关 shuffle 的配置 ⇒ **作废**；
> 因此 (a)/(b) 不再是需要二选一的分叉。现行口径：**P0-5 / P1-2 的 V-REAL 数值面记 `unverified`**，
> 结论待 t35（pinned 前提门通过后的 fp32 复测）填入，占位见 §13。**（已关闭：t43 完成 pinned 复测、t45 补齐地板与判别实验，v2 裁定见 §18。）**

| 选项 | 内容 | 影响 |
| --- | --- | --- |
| **(a)** | 批准**有界 fp32 参考路径**预算（fp32 门 `1e-6`）以归因 S-d / S-g | 判别力最强；该 run 顺带闭合 R-b 逐 rank 留痕；需要新预算与更慢的 fp32 跑 —— **已执行（t22），数字因取数错位作废** |
| **(b)** | 明文接受降级，按 §13 V-3 写 `exact_resume=False` 声明 + 发布限制 | 不新增预算；P0-5/P1-2 只能记"实现 + 协议证据"，**不得**记"已验收" |

现状：实现侧**未改任何阈值**（`3.434e-03` / `1.219e-02` 仍按 `1e-3` / `2e-3` 判超阈），
`compare.json` 已删掉混用的单值 `tolerance` 字段（原值把 `fp16_gradscaler` 的 key 配上 bf16 的
`1e-3`），只保留 `tolerance_keys_recorded={bf16:1e-3, fp16_gradscaler:2e-3}` 与
`within_tolerance_by_key`（两条皆 false），**实测值与阈值逐字未动**。

### 6.2 限制清单（进入发布/对外表述前必须同时引用）

| 编号 | 限制 | 依据 |
| --- | --- | --- |
| L-1 | **P0-5 / P1-2 的 V-REAL 数值面 = `unverified`（未判定）**，不是 failed、更不是通过：S-d `1.219e-02`(fp16)/`1.193e-02`(fp32)、S-g `3.434e-03`/`3.106e-03` **一律为取数错位所致，不可解释为协议差异**；前提门（同一 slot 同一 stem）在 pinned 后已通过，正确数字为 S-d `1.5055e-04` / S-g `3.694e-05`，**契约 v2 下记通过** | §18；§13；§12 |
| L-1b | **默认 shuffle 下不存在跨 world_size 的逐步等价**：等价只在 `ANIMA_MULTIGPU_PINNED_ORDER=1`（默认关、测量手段非产品能力）下成立；任何"多卡与单卡等价"的无条件表述都不得出现 | §12、§14（F-T31-3） |
| L-2 | **R-b 逐 rank 真机留痕缺失**（`_growth_tracker` 采集已修 + 有负对照，但修复晚于 t17 之后无真机保存） | §5.3；`t17-…/report8.md` F-T17-3 |
| L-3 | **恢复清单三字段**（`resume_exact` / `resume_codes` / `resume_bundle_version`）= 已实现 + 单测，**无真机留痕** ⇒ 记"已实现待观测" | `identity.record_resume_outcome`；`capability_ledger.md`「t20 更正」 |
| L-4 | **单桶证据**：本次真跑是 **896×1200 = 4200 族、24 样本**；跨桶、其它 canonical 桶、尾窗未见情形、**4032 族**一律 unverified | `t13-…/report6.md` §1 |
| L-5 | `library/datasets/dreambooth.py::set_image_sizes_from_cache` 的 `subset.cache_dir` 尺寸读取 = **单测覆盖、无真机对照**（只读既有缓存做对照，不改写用户缓存；不作能力声明） | `tests/test_dataset_cache_dir_sizes.py`（7 项） |
| L-6 | **fp32 参考路径未执行**——判别力问题的唯一解法 | §5.2 |
| L-7 | 所有结论仅对 **torch 2.10.0 / accelerate 1.13.0** 成立 | §2 |

### 6.3 发布阻塞

- 不解除任何生产默认：多卡仍非默认路径，`configs/`、WebUI 提示、兼容拒绝面**均未改动**。
- EXP-FSDP2-BASE / EXP-TP-COLROW **不得**进默认配置或 UI 文案（显存无下降 / 仅单层）。
- DeepSpeed / SP **没有**可执行的拒绝路径（本树无代码无用例），不得对外声称"已拒绝/已回归"。
- 真实短训不构成性能结论：`4.64 s/step`（单卡 batch2）与 `2.43 s/step`（双卡 batch1）只说明可运行性。

---

## 7. 4096 固定 shape 候选与 canonical 桶的口径（必须如实反映）

契约 §4 A-3 路径 3 曾把 `output/ab-anima-20261002/cache`（60 个 `*_1024x1024_anima.npz`，
内部 `latents_128x128: (16,128,128)` = **4096 token 单 shape**）列为**资产候选（L2）**，并明确
**不解锁 RUN-A/B/C**。实际演进如下，不得混写：

1. **该候选没有进入任何真跑**。`CONSTANT_TOKEN_BUCKETS`（24 桶，4032/4200 家族）**含 (1008,1024)、
   不含 (1024,1024)**；真实 `train.py` 把图像收成单一 canonical 桶 (1008,1024) 后按
   `{stem}_1008x1024_anima.npz` 找缓存 ⇒ `train_session.py:259` 抛
   `RuntimeError: Latent cache is incomplete. train.py requires a completed preprocess pass`。
   生成 canonical 桶缓存需要一次预处理 pass（会写缓存目录），在冻结边界内被禁止。
2. 因此 **L3 的输入不是 4096 固定 shape**，而是 §5.1 的 896×1200 单桶（4200 族）24 样本。
3. 由此推论：**"4096 固定 shape 的 L3 输入与数值门"= 未执行（unverified）**。任何把
   "路径 3 可用"读成"L3 已在 4096 上通过"的表述都是错的；`hot_run` 对 L2 候选持有
   `hot_run_l2_candidate_only` 的具名拒绝语义，本轮未放宽。
4. 反过来，L3 的数值门结论（S-d/S-g 超阈）也**只**对 4200 族/896×1200 单桶成立，不得外推到
   4032 族、跨桶配对或其它模型族。

---

## 8. v1 已披露偏差：逐项现状（不得抹去）

| 偏差 | v1 事实 | 本轮对当前字节的核对 | 现状 |
| --- | --- | --- | --- |
| **t14 / t16 changedPaths 少列** | v1 t17 review 逐一点名：t16 实改 10 路径，其中 `multigpu/window_protocol.py`、`multigpu/__init__.py`、`tests/test_multigpu_window_protocol.py` 超出其声明；加 t14 历史的 `tests/test_multigpu_admission.py`、`tests/test_multigpu_window_protocol.py`、`tests/test_launch_config.py`，**合计 6 个路径不得描述为"未改"** | 历史流程事实，无法在字节层面"修复"，只能登记 | **保留为流程偏差**；本报告的"相关代码"清单按实际改动写，不按声明写 |
| **t17 F5 恒 False**（`loop.py` 先 `update_skipped=None` 再把它传给 `agree_step`，该 OR 项恒 False；t16 曾声明"已修"，不成立） | v1 t17 判定"F5 实际未修（低危）" | `window_protocol.agree_step` 现签名只有 `loss_nonfinite / gradient_nonfinite / force_skip`，`StepAgreement` 无 `update_skipped` 字段 ⇒ 死参数**已删除**，恒假 OR 项不复存在 | **已修（参数删除）** |
| **identity.py 仅捕 OSError** | v1 t17 low-②：写块只捕 `OSError`，非 OSError 会在 `agree_any` 前逸出，peer 需等 PG timeout | `write_runtime_identity` 现为 `except Exception`（注释直接引用 review t17 low-①），仍不捕 `BaseException`（Ctrl-C 可退出） | **已修**；但 `note_gradient_sync` / `record_resume_outcome` 的读路径仍窄（`OSError`/`ValueError`），属**best-effort 旁路**，已用 WARNING/debug 显式化，不再影响协商同序 |
| **async 分支身份字段缺失** | v1 t17 low-③：async 梯度分支走 `finish_step()` 时不取 `sync_result`、不调 `note_gradient_sync` ⇒ `gradient_buckets`/`reduction_mode` 停在占位值 | `loop.py` async 分支现取 `async_result = async_gradient_sync.finish_step()` 并调 `note_gradient_sync(..., async_overlap=True)`，真机 `multigpu_identity.json` 记 `reduction_mode=async_bucket_all_reduce_mean(2 buckets)` | **已修（并有真机留痕）** |
| **`verify_resume_cursor` 的 epoch 仅信息性** | 旧实现只比位置，epoch 只写进 message | 现走 `guard_resume(..., resume_cursor=expected, rank=本rank)`，`cursor.COMPARED_KEYS=("sampler_position","epoch","rank")` 三项**逐项不等即具名报错**（`multigpu_resume_cursor_mismatch`） | **已修（epoch 进入硬比较）** |
| **`skip_first_batches` 重消费语义** | 生产跳过序列 = `skip_first_batches(initial_step-1)` **＋** 循环里一次 `continue` ⇒ 首个真正执行的 batch 的 local index 是 `initial_step`（旧代码按 `initial_step-1` 记账，fixture 恢复后钉错样本） | `cursor.resume_skipped_batches` 成为唯一生产者并在 docstring 写明"`initial_step` 条、不是 `initial_step-1`"，`train_session` 直接调用它 | **已修（语义已文档化 + 单测锁定）**；真机未单列观测该条 |
| **Torch 2.10 实测不得外推 2.12** | 契约 E-2 / G-3 | §2 已写明 | **保留为硬边界** |

补充：v1 还有一条 **F-T12-1（low，纯文案）**——`verify_rng_files=False` 时 `exact=True` 的消息
仍写 "re-read at resume time"，而那一刻并未重读。**该项现已修复**：`resume_protocol.py` 按
`rng_recheck` 分成两句（`RNG_RECHECK_DONE` 才用 "re-read at resume time"，否则用
"as recorded at save time (no resume-time re-read requested)"），日志侧同分支区分。v1 把它记为
"已知低危未修"，本报告按当前字节**更正为已修**。

---

## 9. 历史与更正（保留原判，不改写）

1. **v1 基础实现原判 failed，不可改写**：v1 t5（总体 review）判 `failed`，t3（故障注入 + CUDA/FSDP2/TP）
   终态 `failed`。本报告的"已验收"**不是**对 failed 的追认，而是建立在其后的 t8/t11/t14/t15/t16/t17/t18/t20/t21
   与 v2 t2 的修复 + 复验之上；`t5` 未通过的项按 §6 原样保留。
2. **t3 的 findings 未落库，本轮直读 artifact 重建**：t3 的 update 因 JSON 过大被判
   `MALFORMED_RESPONSE`，结构化 findings 丢失。本报告据
   `output/tests/multigpu_reference_validation/t6-faultinj-cuda-20261108/`（`report.md`、
   `summary.json`、`capability_ledger.md`、`harness/`）重建事实：3 项 blocker（P0-1 方向 B 放行、
   真实 RNG 键不匹配导致 exact 不可达、L3 slot 跨 world 不稳定）+ 2 项 medium（docstring 与行为矛盾、
   缺 SUM 形态回归）。这些均已由 t8/t11/t14/t16 修复并由 t10/t12/t13/t17 复验，但**t3 本身仍是 failed**。
3. **captain 自身两处契约错误的如实记录**：
   - v2 **t8 的 verify 命令误写**不存在的 `tests/test_training_window_replay.py` 与
     `tests/test_training_fp16_nan_static_equivalence.py`（真名**无 `training_` 前缀**：
     `tests/test_window_replay.py`、`tests/test_fp16_nan_static_equivalence.py`）。成员拒绝按错名
     执行，没有把 pytest 的 exit 4（用法错误）记成通过——这属于**captain 的编排错误，成员无责**。
   - v2 **t13 原有 verify 只有一条 hot 命令**（`-m hardware` 在无门时会结构性地只得到 skip），
     已由 captain amend 为三条：非 hardware 单元门 + hardware 预期 skip + `summary.json` 预算机器校验。
     本轮报告中的 g4 结果因此必须与"1 passed / 2 skipped"一起读，skip 不算通过。
4. **三次取消不得当成成员失败**：
   - **t4**（Anima 短训）因 **t11 正在写 `library/training/multigpu/`** 而取消——真机训练会对并发
     编辑中的库运行，证据不可复现；由 t13 接替。
   - **t9**（复验）因 captain 漏设 `dependencies` 而误派给验证者，对变动中的工作区复验无法绑定字节；
     由 t10 接替。
   - **t7** 是审批流自动生成的**空占位 work 任务**（无 description/objective/acceptance/verify/inScope），
     会占用唯一生产写入者的任务槽，由 captain 取消；实际工作由 t1..t6 覆盖。
5. **EXP 裁定原样保留**：EXP-FSDP2-BASE / EXP-TP-COLROW = `experimental` **不可采用**
   （显存无下降 / 仅单层）；DeepSpeed / SP **具名未执行**（本树无代码无用例）；PP 拒绝**回归通过**；
   全部 hardware skip **逐条具名、不计通过**。
6. **数字以最新绑定为准**：t10 的 219 passed / g1 7 passed 已因字节漂移失效，现行数字用 t12 重绑
   结果（g1 70 / g2 225 / g3 121 / g4 1 passed + 2 skipped），并引用其
   `fingerprint_and_precheck.txt` 作为字节绑定证据（§4）。
7. **用户提交与保留**：`HEAD 2e7da55e` 由用户在验证期间提交并推送；本轮未提交、未推送、
   未回滚、未清理。本轮开始时工作树为 12 条未提交差异（其中 `docs/` 只有
   `docs/guidelines/training.md` 一条，属 t18 引入的多卡使用说明）；本报告完成后 `docs/` 侧
   变为 4 个 `M` + 1 个 `??`（见 §1 与 §10 第 0 步），`library/`/`tests/`/`bench/` 与 `output/`
   的既有改动**逐条未动**。

---

## 10. 复现步骤

```bash
cd /mnt/SD_Dataset/训练器相关/krea2-multigpu
PY=/mnt/SD_Dataset/训练器相关/krea2-webui/.venv-v100/bin/python

# 0) 字节绑定（先跑，确认你复现的是本报告的版本）
git rev-parse --short HEAD && git status --short
sha256sum docs/proposal/multigpu_reference_validation_contract.md   # 期望 80dfdb02…

# 1) CPU/Gloo 四门（约 4 分钟；CUDA 置空，不占 GPU）
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD $PY -m pytest -q -m 'not hardware' \
  tests/test_multigpu_resume_protocol.py tests/test_multigpu_cursor_wiring.py \
  tests/test_multigpu_checkpoint_protocol.py
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD $PY -m pytest -q -m 'not hardware' \
  tests/test_multigpu_*.py tests/test_launch_config.py
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD $PY -m pytest -q \
  tests/test_window_replay.py tests/test_fp16_nan_static_equivalence.py \
  tests/test_training_checkpointing.py tests/test_training_resume_actions.py \
  tests/test_training_resume_options.py

# 2) 硬件门（只在放行时跑；无门时结果只可能是 skip，不算通过）
MULTIGPU_REFERENCE_HARDWARE=1 PYTHONPATH=$PWD $PY -m pytest -q -m hardware -rs \
  tests/test_multigpu_checkpoint_protocol.py tests/test_multigpu_resume_protocol.py \
  tests/test_multigpu_reference_experiments.py

# 3) 重放历史证据（不自启训练，只读既有 artifact）
$PY - <<'PY'
import json
r="output/tests/multigpu_reference_validation/"
print(json.load(open(r+"t13-anima-hot-20261108/evidence/compare.json"))["comparisons"])
print(json.load(open(r+"t6-faultinj-cuda-20261108/harness/fsdp2_tp_out/fsdp2_tp_report.json")["peak_memory_mib"])
PY
```

真实短训**不在**本步骤内：它需要 `MULTIGPU_REFERENCE_HOT=1`、D-5 磁盘门与**已批准的更新预算**，
由验证方按其 run 报告执行；复述数字时请引用 §5 与 `t13-anima-hot-20261108/report6.md`。

---

## 11. 不得外推（与契约 §1/G-3 一致）

- V-GLOO 只证明**本项目自己的** collective 顺序、状态协议与账本对称性，不证明 NCCL/CUDA 行为。
- V-CUDA-SMALL 与 toy FSDP2/TP 不证明 Anima 真实数值、质量或性能。
- 单桶 4200 族证据不外推到 4032 族、跨桶配对、尾窗未见情形、其它模型族。
- TLoRA/compile/NF4/block swap 等生产特性**均未**在本轮多卡验证范围内。
- 本文件是登记与边界说明，**不是**发布许可；生产默认值、契约阈值与授权范围均未改动。

---

## 12. 取数顺序三层缺陷与其修复（2026-11-08 增量，t27→t34 链条）

这一章解释为什么**首版报告里的 single-vs-dual 数字全部作废**。根因不在 DDP 协议，在**取数顺序**：
同一 `slot` 在两个 world_size 下取到了**不同的图**。三层缺陷叠加，任一层单独存在都足以让对照失效。

### 12.1 三层缺陷

| 层 | 位置 | 机制 | 后果 |
| --- | --- | --- | --- |
| **①** `SeedableRandomSampler` 按 `len(dataset)` 做 randperm | accelerate `data_loader.py:73`；`bootstrap.py:889` 把 DataLoader 钉成 `batch_size=1`（"一个 dataset item 已经是一个 batch"） | `shuffle=True` 走 `SeedableRandomSampler`，它 shuffle 的是 **item 级** `randperm(len(dataset))`。因为 item 即 batch，**`len(dataset)` = bucket batch 数**：24 图时 B=2→**12**、B=1→**24** | 同 seed 下 `randperm(12)` 与 `randperm(24)` **互不相关**；`show` 上用 `data_seed` 也救不了长度不同的两个排列 |
| **②（真凶）** `DatasetBuckets.shuffle_buckets()` 打乱 descriptor 列表 | `library/datasets/buckets.py`：`create_buckets` 建好 `buckets_indices`（一 batch 一个 descriptor）后**立刻** `shuffle_buckets()`，实现为 `random.seed(seed + epoch)` + `random.shuffle(buckets_indices)` | 列表长度 = batch 数（24 图：B=2→12、B=1→24），同 seed 下是**两个无关排列** | **即使关掉 DataLoader 的 shuffle，item 序依然错位**（这正是 t31 前提门在"关掉 seedable"之后仍然失败的直接原因）。真数据三张表：B=2 侧 item0=`[09_58_15,13_31_23]`、B=1 侧 item0/1=`[09_58_15]`/`[13_31_23]`；但两侧 descriptor 的 `batch_index` 分别是 `1,9,8,5,10,2` 与 `14,0,17,7,6,11` |
| **③** 训练 loader 硬编码 `shuffle=True`、CLI 无开关 | `library/training/bootstrap.py`（`DataLoader(train_dataset_group, shuffle=True, **kwargs)` 两行；t32 后移入 pinned 的 `else` 分支，默认行为不变）；`library/cli_args.py` 无 shuffle 参数 | 产品默认路径**无法用 CLI 关闭 shuffle**，只能靠 env 门控 | 合同 §13 V-2 第 2 条（关闭 shuffle 类随机）在本项目**只能靠 env 门控满足**（见 §14，F-T31-3） |

**为什么"位置→噪声对、位置→图像错"**：分片公式本身是对的——accelerate `BatchSamplerShard(split_batches=False)`
让 rank 取 item `k*W+rank`。所以按 slot 钉住的 noise/timestep 仍逐槽对齐（t31 实测噪声 head8
`0.001/0.002/0.003…`、sum `268.8/537.6/806.4…` 两侧一致），**但每个 slot 背后的 stem 不同**。
叠加机制还有一个：`BatchSamplerShard(even_batches=True)` 在 `len(dataset) % world ≠ 0` 时**回绕到开头**，
实测 n=5/W=2 → rank1=`[1,3,0]`，这就是 t31 里出现"三步内重复图"的来源。24 个 stem 全唯一、
`num_repeats=1`、`validation_split_num=0` ⇒ **没有数据层重复**，重复完全来自取数层。

结论：在这一配置下，single-batch2 与 dual-batch1 的 S-d / S-g **比的是不同数据**，
任何数值差都不能归因于协议（§13）。

### 12.2 修复：`ANIMA_MULTIGPU_PINNED_ORDER` 门控（默认关）

| 落点 | 内容 |
| --- | --- |
| `library/datasets/buckets.py` | `BucketManager.shuffle()` 提前 return（桶内保持插入序）+ `apply_pinned_order()` 把 descriptors 排成 `(bucket_index, batch_index)` 自然序；新增 `PINNED_ORDER_ENV` / `pinned_order_enabled` / `pinned_buckets_indices`（纯加法，`CONSTANT_TOKEN_BUCKETS` 等常量表零改动） |
| `library/training/bootstrap.py` | `_PinnedOrderSampler`（identity）替代 `shuffle=True`；accelerate 仍照常用 `BatchSamplerShard` ⇒ rank 取 item `k*W+rank`，**同 slot 同 stem**。原两行 `DataLoader(..., shuffle=True, ...)` 移入 `else` 分支，**与 HEAD 等价**（t34 逐 hunk 复核，diff 与 HEAD 逐字相同） |
| 默认行为 | 门控默认关；负对照断言 gate 关时 `apply_pinned_order` 返回 `0` 且 `buckets_indices` 前后**逐项相同** |
| `--multigpu_seedable_sampler` | 已**彻底移除**（t33/t34 复核 seedable 引用数为 0）；旧门只保证"排列算法一致"，不保证"同一批样本"（§17） |
| 回归 | `tests/test_multigpu_pinned_data_order.py`（7 passed，真 24 个真实 stem、无 GPU/VAE）：`test_same_slot_same_stem_across_batch_sizes`（W=1/B=2 与 W=2/B=1 的 3 步 stem 行完全相同）；只钉 loader、不钉 descriptor 仍错位（第二层证据）；负对照 / 长度对照 / `even_batches` 回绕说明 |

### 12.3 门控覆盖状态（**以实际结果为准，未完成即点名未覆盖**）

| 取数路径 | 覆盖状态 | 依据 |
| --- | --- | --- |
| `bootstrap.py` **主训练 loader** | **已覆盖** | t32 接线 + t34 复核 + `test_multigpu_pinned_data_order.py` 真数据集回归 |
| `buckets.py` descriptor 排序（`BucketManager.shuffle` / `apply_pinned_order`） | **已覆盖** | 同上（含负对照） |
| `loop.py` **stage-switch 重建 loader** | **未覆盖（点名）** | t34 交付说明第 6 条：本轮未获授权；t36（门控覆盖剩余两条取数路径）截至本文档写入**仍在运行、未产出结果** ⇒ `stage_schedule` / `adaptive_oom_retry_resume` 的 run **即使开门仍会错位** |
| `make_cursor_dataloader`（**cursor dataloader**） | **未覆盖（点名）** | 同上；t36 完成后须回填本表，并在 `bench/multigpu_reference/README.md` 同步 |
| 单卡默认路径逐位不变 | **已保证** | 门控默认关 + `else` 分支与 HEAD 逐字相同（t34） |

> **关于 t36 的现场观察（截至本文档写入，2026-11-08，只读 `git status`）**：工作树出现
> `library/training/adaptive_runtime/pinned_order.py`（新增）、`library/training/adaptive_runtime/resume_data.py`（修改）、
> `tests/test_multigpu_pinned_order_paths.py`（新增），与 t36「门控覆盖剩余两条取数路径」的方向一致。
> 但 **t36 尚未产出完成结论**，因此上表这两行**仍按"未覆盖（点名）"记录**——**不得**据"文件已出现"
> 推断"已覆盖"。t36 完成、且验证方在真机或真数据集上确认后，由后续任务把本表与
> `bench/multigpu_reference/README.md` 一并改成"已覆盖"并附证据指针。

### 12.4 链条账（谁发现了什么，便于复核）

| 轮次 | 结论 | 关键证据 |
| --- | --- | --- |
| t25 | 判据② `forward_shape_sensitive`：第一处差异在 `global_step=0` 的 `adapter_hash` | 埋点探针（bench 侧，`library/` 零改动） |
| t27 | 机械点名 S-d 首个不等字段 `noise_per_sample`（step0）⇒ "同一 slot 被喂了不同内容" | 逐样本内容比对；但缺原始噪声元素值，验收未完全达成 ⇒ 该轮 failed |
| t28 | 补 `head8 + sum + hash + dtype + shape`、捕获点、`fixture_provenance.sha256`、`LatentIndex` 溯源；CPU/Gloo 双进程自证捕获点一致 | 消除了"探针假象"（提交因 `MALFORMED_RESPONSE` 被拒，captain 代收） |
| t29 | 二分判定 = **真实取数缺陷**（非编排假象、非探针假象）：两侧 `fixture_provenance.sha256` 全等 `368e91e2…`，双卡 `noise_raw` 取到的是**另一张真实训练图** | `report12*`、`probe2/*.jsonl` |
| t30 | 用 accelerate **真类**归因为"加速库固有行为 + 本项目 item=batch 语义"；交付 `bench/multigpu_reference/loader_order.py::PinnedOrderBatchSampler`；同时**降级 F-T3-1**（§17） | CPU 复现 `randperm(12)` vs `randperm(24)`；`test_pinned_order_puts_the_same_stem_in_the_same_slot` |
| t31 | 前提门**仍未通过**，并发现 env 门控当时**根本没接线**（`grep -rn "loader_order\|PINNED_ORDER" library/` = 0）；同时发现第二层机制（关 shuffle 后 item 序仍不同、三步内出现重复图） | `report13.md`、`evidence/verdict_shape_noshuffle.json`；用量 12/45 |
| t32 | 定位**真凶** = `shuffle_buckets()`；交付 `ANIMA_MULTIGPU_PINNED_ORDER` 门控 + 真数据集回归 | 三张表（`len(dataset)`、item 内 stem、slot→stem）；7 passed |
| t33 | **阻断回归**：t32 误删 `@dataclass(frozen=True)`，任何真跑在 `prepare_dataset` 即崩（`TypeError: DatasetBuildResult() takes no arguments`，`bootstrap.py:549`）；单卡 8.14 s / 双卡 16.31 s 失败，均在首次 forward 之前 ⇒ 前提门未评估、预算 33 全额保留 | `evidence/dataclass_regression.txt`；F-T33-2（§15） |
| t34 | 装饰器恢复（`dataclasses.fields` 7 个、位置构造成功、`FrozenInstanceError` 正常）+ 补冒烟用例 + 新增第 4 条 verify（18 passed） | 逐 hunk 复核 t32 全量 diff，无第二处漂移 |

---

## 13. 四个数字的口径纠正与 P0-5 / P1-2 的裁定变更

### 13.1 四个数字一律作废

| 数字 | 出处 | 首版解读 | **现行口径** |
| --- | --- | --- | --- |
| `1.219e-02` | t13 fp16，S-d（单卡 batch2 vs 双卡 batch1） | 超 fp16 门 `2e-3` ⇒ failed | **取数错位所致，不可解释为协议差异** |
| `3.434e-03` | t13 fp16，S-g（连续 20 vs 10+恢复 10） | 同上 | 同上 |
| `1.193e-02` | t22 **fp32**，S-d | t22 曾判"协议缺陷坐实，非精度问题" | 同上（**该结论作废**：fp32 路径同样是 `shuffle=True` 配置） |
| `3.106e-03` | t22 **fp32**，S-g | 同上 | 同上 |

**为什么四个都作废**：它们全部来自**违反合同 §13 V-2 第 2 条（"同一份数据与顺序……关闭
`caption_dropout` / `shuffle` 类随机"）**的配置——训练 loader 硬编码 `shuffle=True`
且 CLI 无开关（§14），再叠加与 DataLoader 无关的 `shuffle_buckets()` 重排（§12）。
两侧 item 排列长度不同（12 vs 24）且互不相关 ⇒ **对照双方消费的不是同一批样本**。
t31 的前提门（同一 slot 同一 stem）**一次也没通过过**，因此 S-d 至今**没有一次有效测量**，S-g 同理。

**"换精度后量级不变"也不再是协议证据**：从 fp16 到 fp32，`1.219e-02→1.193e-02`、
`3.434e-03→3.106e-03` 几乎不变，当时被读成"与精度无关的确定性协议差异"；但在取数错位的
解释下这一现象**同样自然**——数据错位与精度无关。故它不构成协议缺陷证据，也不构成"经得起换精度检验"的证据。

**未改任何阈值**：`compare.json` 的 `tolerance_keys_recorded={bf16:1e-3, fp16_gradscaler:2e-3}`
与实测值逐字未动；本轮只是**改变解读**，不改判据、不改文件里的数字。

### 13.2 裁定变更：`failed` → `unverified`（**后经契约 v2 再改为「通过」，见 §18**）

**P0-5 / P1-2 的 V-REAL 数值面记 `unverified`（未判定），不是 `failed`。**

- 正向理由：契约 §13 V-3 对前提门的规定是"**未生效 ⇒ 单双卡数值等价直接记 `unverified`，不得判通过**"。
- 反向理由同样成立：**判 failed 也需要一个满足前提的对照**。前提门未通过时，超阈数字既不能证明
  "协议有缺陷"，也不能证明"协议无缺陷" ⇒ 只能 `unverified`。
- 不受影响的子项（仍按首版结论）：跨 world_size **拒绝/显式降级**（V-GLOO 已验收）、版本化
  bundle 结构面（已验收）、恢复时刻 RNG 二次核验与假 exact 消除（已验收）、R-a / R-c / R-d 真机
  （已验收）、恢复清单三字段（**仍为"已实现待观测"**）、R-b（**仍为 unverified**）。

### 13.3 结论占位（**已由 §18 的 v2 裁定关闭**；以下占位框保留原文，不再使用）

```text
================ P0-5 / P1-2 V-REAL 数值面结论：占位（未填） ================
占位状态      : 空 —— 等待 t35（verification-round-3）产出结论
占位落盘      : 2026-11-08，由 t37（交付文档并行刷新）建立
填入人        : captain（拿到 t35 结果后填入本节）
--------------------------------------------------------------
填入时必须给出以下 6 项，缺一项不得关闭本占位：
 1) pinned 前提门结果：同一 slot 同一 stem（单卡 batch2 vs 双卡 batch1）是否成立；
    两侧 len(dataset)（预期 12 / 24）与两侧前 3 步 stem 序列；
 2) S-d 测量：fp32、门 1e-6，测量值 + 对照段 argv 差异（逐 token）；
 3) S-g 测量：fp32、连续 20 vs 10+恢复 10、门 1e-6，测量值；
 4) 若仍超阈：首个超阈张量名 + 分组（lora_down / lora_up）+ 是否与 R-b 留痕状态相关；
 5) 裁定：`已验收` / `unverified` / `failed` 三者择一，并写明依据条款（§13 V-3 或 §10）；
 6) 预算：本次消耗的有效更新数与剩余额度（历史累计：t13 50 + t17 20 + t22 60，
    其中 t22 超限 8 个已由成员主动上报）。
--------------------------------------------------------------
填入前对本报告的全部约束：
 - 这两点只能以 `unverified` 出现，任何"等价 / 不等价"的措辞都不得出现；
 - 不得把 pinned 下的成立读成产品默认成立（见 §14）；
 - 不得引用 1.219e-02 / 1.193e-02 / 3.434e-03 / 3.106e-03 作为任何一方的证据。
=========================================================================
```

> **关闭记录（2026-11-08，t46）**：本占位已关闭——(a) 路线已执行（fp32 参考路径 + `ANIMA_MULTIGPU_PINNED_ORDER` 前提门），
> 结论见 **§18**：四项判据 C1–C4 全过，P0-5 / P1-2 的 V-REAL 数值面记「通过（v2）」。以上占位框内文字保留原文，不再维护。

---

## 14. 合同只能靠 env 门控满足（F-T31-3）

- **合同要求**：§13 V-2「对照成立的前提（必须同名冻结）」第 2 条——*同一份数据与顺序：同一 subset
  列表、同一 `num_repeats`、`--max_data_loader_n_workers=0`，**关闭 `caption_dropout` / `shuffle`
  类随机***。
- **实际缺口**：`library/training/bootstrap.py` 的训练 loader **硬编码 `shuffle=True`**，
  且 `library/cli_args.py` **没有任何 shuffle 开关**（无 `--shuffle` / `--no-shuffle`）。
  ⇒ **用 CLI 无法关闭训练 loader 的 shuffle**；当前唯一手段是 env 门控
  `ANIMA_MULTIGPU_PINNED_ORDER=1`（默认关）。
- **性质界定**：该门控是**测量手段，不是产品能力**。它把对照实验钉到同一取数顺序，不改变默认行为，
  也不构成"多卡生产路径已等价"的能力声明。合同若要把这个前提写成"可执行的命令"，需要先定义
  CLI/配置面（当前合同未定义，属合同侧待补项）。
- **本文档全文的准确表述（任何章节不得偏离）**：
  **产品默认仍按 `seed + epoch` 打乱（`shuffle=True` + `shuffle_buckets()`）；跨 world_size 的
  逐步等价只在 pinned 顺序下成立，默认 shuffle 下不成立。**
  因此不得出现无条件的"多卡与单卡等价"；也不得把 pinned 下的成立读成默认路径成立。
- **未闭合点**：① 无 CLI 开关，env 门控不可发现、也无法在 WebUI 表达；② `loop.py` stage-switch 与
  cursor dataloader 未走门控（§12.3，t36 未完成）；③ 若将来要把 pinned 变成产品能力，需要单独的
  采用条件、默认值与证据（合同未定义，需 captain 修订）。

---

## 15. 验证体系盲区（F-T33-2）

- **事实**：t32 误删 `@dataclass(frozen=True)` 后，**契约 verify 全绿**——verify1 `38 passed` / 17.72 s、
  verify2 `44 passed` / 8.32 s、`git diff --check` exit 0——但任何真跑在 `prepare_dataset` 即崩：
  `TypeError: DatasetBuildResult() takes no arguments`（`bootstrap.py:549`）。
  单卡 8.14 s、双卡 16.31 s 失败，均发生在首次 forward 之前。
- **根因**：那三条 verify 命令**没有一条 `import library/training/bootstrap.py`**。业务用例走的是
  probe / 依赖注入路径，`git diff --check` 只查空白。**纯 import 也发现不了**——模块能导入，
  只是那个类不再是 dataclass（`bootstrap import OK`）。
- **新规则（本轮已生效，建议写进 captain 规则）**：
  **凡改 `library/training/bootstrap.py`、`library/datasets/buckets.py`、`library/training/loop.py`
  这类训练期加载路径，verify 必须包含"import + 关键对象可构造"的冒烟用例**；
  不接受"只跑业务用例 + `git diff --check`"。t34 已把第 4 条 verify 命令（冒烟套件 18 passed）加入自验。
- **负对照发现（值得记住）**：**AST 遍历查不出被删的装饰器**（被删掉的声明根本不在 AST 里）。
  真正拦住它的是**按名钉住的断言**：`is_dataclass(DatasetBuildResult)` + 7 字段逐一比对 +
  位置构造 + `FrozenInstanceError`。所以"扫描声明集合"不能替代"按名钉对象"。
- **落地件**：`tests/test_training_bootstrap_dataclass_contract.py`（5 条：import 冒烟 + 冻结 dataclass
  七字段 + 每个 dataclass 声明逐个 `is_dataclass` + buckets 门控默认关 + loader 分支存在性）。
  负对照实测：临时删装饰器 → 第 2、3 条立即 fail，恢复后 5 passed。

---

## 16. 流程与 captain 侧缺陷（如实入账）

| 编号 | 事实 | 影响与处置 |
| --- | --- | --- |
| **①a** | **verify 命令写错（t8）**：`tests/test_training_window_replay.py`、`tests/test_training_fp16_nan_static_equivalence.py` 不存在；真名**无 `training_` 前缀**：`tests/test_window_replay.py`、`tests/test_fp16_nan_static_equivalence.py` | 成员**拒绝**按错名执行，**未把 pytest exit 4（用法错误）记成通过**；按真名重跑后通过 |
| **①b** | **verify 命令写错（t13）**：原 verify 只有一条 `-m hardware` 的 hot 命令，结构上**只能得 skip** | 已由 captain amend 为三条：非 hardware 单元门 + hardware 预期 skip（skip 不计通过）+ `summary.json` 预算机器校验 |
| **①c** | **verify 命令写错（t23）**：`tests/test_training_fp16_nan.py` 不存在；真名是 `tests/test_fp16_nan_static_equivalence.py` / `tests/test_fp16_residual_safe.py` | 同上，按真名重跑；未产生错误放行 |
| **②** | **t22 超预算**：fp16 R（R-b）探针要求 **≤2** 个更新，实际跑了 **10**（停止点取自 plan 的 step-10 state 目录而非 plan 的 stop 条件）⇒ 累计 70 超硬上限 62 共 **8 个更新** | 验证者**主动上报**并"发现即停"、未再发起运行。流程改进：预算控制点必须绑在 plan 的 stop 条件上，不能依赖"看 state 目录"这种事后判据 |
| **③** | **`MALFORMED_RESPONSE` 两次吞掉更新**：t3 的 **7 条 findings**、t28 的**完整提交** | 均已由 captain 代收或从证据目录重建：t3 由 `output/tests/multigpu_reference_validation/t6-faultinj-cuda-20261108/` 重建（3 blocker + 2 medium）；t28 的结论由 captain 代收后落库。**代价是结构化 findings 一度丢失，必须靠 artfact 反推** |
| **④** | **`changedPaths` 白名单机制导致真实改动无法完整申报**：一批真实改动落在契约 in-scope 之外 | t18、t20 已在 output 里逐条列明（而非写进 `changedPaths`，否则会被判 undeclared 打回）；机制问题是"白名单先于事实"，建议 captain 允许"事实优先、随时补录" |
| **⑤** | **scope 扩充披露**：`tests/test_multigpu_pinned_data_order.py`（t32，7 passed，真 24 stem 数据集回归）与 `tests/test_training_bootstrap_dataclass_contract.py`（t34，5 passed，F-T33-2 冒烟）**在契约 in-scope 之外落盘** | captain **事后批准纳入 scope**。两者均未进 `changedPaths`（不在白名单内），故在此显式登记，避免"未申报即未改"的误读 |

补充说明（不是 captain 侧缺陷，但同属流程账）：t4 / t9 / t7 三次取消的**理由**已在首版 §9.4 记录，
不得当成成员失败；本表只登记**编排与预算侧**的问题。

---

## 17. F-T3-1 旧结论降级：分片算术已验证，样本级对齐未验证

- **旧表述**（t8 引入、t10 复验通过）："门开 single==dual 逐位相等、**12/12 slot 对齐**"。
- **现行表述**：**分片算术已验证；样本级对齐未验证。**
- **依据（t30）**：t3 自述是"独立探针"，比的是 `_run_epoch_steps` / `gradient_sync` / 协议原语，
  **tensor 由探针自建，没过真 bucketed dataset、没走 hot_run 真 argv ⇒ 比的是 slot / 索引，而不是 stem**。
  机制上用真 accelerate 类即可看出差别：当 dataset 按**样本**计数时，排列长度与 `batch_size` 无关，
  两个拓扑确实逐位对齐（这正是旧结论"绿"的原因）；换成本项目的真实语义"**一个 item 即一个 batch**"
  （`bootstrap.py:889`）后，排列长度变成 12 vs 24 ⇒ **红**。用例
  `test_a_sample_level_sampler_is_why_the_old_gate_looked_green` 把这一点钉死。
- **效力范围**：`--multigpu_seedable_sampler`（t8 引入、现已移除）能保证"排列算法一致"，
  **不能保证"两侧取到同一批样本"**；后者只有 pinned 顺序（§12.2）才成立。
  相应地，t10 的「F-T3-1 复核通过」**只覆盖分片算术，不再覆盖样本级对齐**；
  §5.4 的 S-0 行"成立"应读作"分片算术成立"（该行已加注）。

---

## 18. P0-5 / P1-2 的 V-REAL 数值面：v2 判据下的明确裁定（2026-11-08 收口）

**裁定：在 [契约修订 v2](../proposal/multigpu_reference_validation_contract_amendment_v2.md) 的 C1–C4 判据下，
P0-5 与 P1-2 的 V-REAL 数值面记为「通过」。** 四项判据逐条对照如下，**每一条都给出实测值与判定动作**；
v1 的 S-d/S-g 原判据已被 v2 取代（见 §13、§19 与契约 §19）。

> 口径声明（先读）：这里的"通过"**不是**"多卡与单卡等价"。v2 判据的含义是——**在 C1 数据对齐成立的前提下，
> 两条路径的数值差异不大于同一张卡自己重跑一遍的抖动**。产品默认路径（`shuffle=True`）仍然**不等价**，
> 见 §14。任何"多卡=单卡"的无条件表述依旧禁止。

### 19.1 C1 数据对齐门 —— 通过

| 判据 | 实测（t43，pinned 首次有效测量） | 判定 |
| --- | --- | --- |
| 同 slot 同 stem（前 ≥3 步） | **6/6 slot 两侧 stem 全等** | 通过 |
| 索引满足 `k*W+rank` | 双卡 `rank0` = 单卡偶数位、`rank1` = 奇数位 | 通过 |
| 每 rank 连续步无重复 | **3/3 unique**（两侧）；`len(dataset)` 12/24 与 `assert_stream_contract` 两侧 passed | 通过 |

### 19.2 C2 步级梯度等价（ULP 口径）—— 通过

| 判据 | 实测（t45，step0 全批梯度，npz 逐元素） | 判定 |
| --- | --- | --- |
| `max\|ΔG\| ≤ 4 ULP(max\|G\|)` | `max\|ΔG\| = 6.023e-07`；`max\|G\| = 3.0` ⇒ ULP = `2.384e-07` ⇒ 4 ULP = `9.537e-07`；实测 **≈ 2.53 ULP** | 通过 |

- **观测项（不作判据）**：392 个张量中 **196 个逐位相同**（50.0%）。此项**不是**判据——
  "至少半数"是 captain 在看到 `196/392` 之后才写下的，属"用观测反推阈值"，已于终审中降级。

- **为什么不用 rel L2**：t45 同时给了张量级 rel L2 = `6.341e-05`（111× fp32 同配置梯度地板）。
  这不是"实现差异大"，而是 **rel L2 的分母/方向陷阱**：`ΔG` 与 `G` **近正交** ⇒ 模长几乎不变
  （`grad_norm_total` rel 仅 `3.7e-09`）而逐元素已变。这与 `lora_up` 零初始化陷阱**同源**。
- **为什么把"半数逐位"降级（诚实记录）**：该阈值是 captain 在 t45 报出 `196/392` 之后才定下的，
  属于"先有观测、再定阈值"，与本次修订要纠正的"拍脑袋阈值"同源。终审时已把它降为观测项，
  判据只保留与机器精度锚定的 ULP 一条（ULP 与实测量级无关，不会被观测反推）。

### 19.3 C3 权重漂移（参照实测噪声地板）—— 通过

| 判据 | 实测 | 判定 |
| --- | --- | --- |
| `rel L2(S-d) ≤ fp16 同配置噪声地板`（t17 最大者 `3.988e-04`） | **`1.5055e-04`**（max_abs `2.3414e-04`；最差层 `blocks_3_mlp_layer1.lora_up` `1.417e-02`） | 通过（约为界的 **38%**） |
| 参照集 | t17：`2.040e-04` / `3.225e-04` / `3.988e-04` —— **三者都比 S-d 的 1.5055e-04 大** | — |

- **禁止用 fp32 同配置地板（`5.634e-08`）作参照**：那是"同一归约顺序重复自己"，拿它当界等于要求
  "两种不同归约顺序逐位相同"，数学不可达（§20 与 amendment §1）。
- **严重性以 `max_abs 2.34e-04` 与 global rel L2 为准**：`lora_up` 零初始化 ⇒ 分层 rel 无意义
  （放大 = 分母小 18.98× × 误差大 1.52× = 28.9×；组内均值口径 41.4×）。

### 19.4 C4 恢复（S-g）两段 —— 通过

| 判据 | 实测（t43 B 段） | 判定 |
| --- | --- | --- |
| ① 续接逐位精确：stem 逐步全等；cursor / state sha256 可核 | 恢复臂第 11 步 stem 序列 == 连续臂第 11 步（**是**）；overlap 的 step10–19 **两 rank 逐步全等，`mismatching=[]`**；state 目录 11 文件 sha256 已记 | 通过 |
| ② 20 步散度 ≤ 同一 fp16 噪声地板（`3.988e-04`） | **`3.694e-05`**（约为界的 **9%**） | 通过 |
| ③ 附逐步曲线 / 首超 `1e-6` 步号 / epoch 边界 | 曲线：step11 `1.602e-09` → 12 `7.692e-08` → 13 `1.611e-07` → 14 `3.944e-07` → **15 `1.081e-06`（首超 `1e-6`）** → 16 `1.883e-06` → 17 `2.574e-06` → 18 `4.150e-06` → 19 `2.062e-05` → **20 `3.694e-05`**；至 step20 从未超 `1e-4`；**epoch 边界（step13）无台阶**（`1.61e-07→3.94e-07` 平滑） | 通过 |

### 19.5 结论与仍然开放的子项

| 项 | v2 裁定 | 说明 |
| --- | --- | --- |
| **P0-5** 恢复语义显式化 | **通过（v2）** | 跨 world_size 拒绝/显式降级（V-GLOO 已验收）+ 同 world 续训对照（C4 全过） |
| **P1-2** exact resume | **通过（v2）** | 恢复时刻 RNG 二次核验 / 假 exact 消除 / 版本化 bundle（已验收）+ C4 全过 |
| P1-2 的「恢复清单三字段真机留痕」 | **仍"已实现待观测"** | 与数值面无关，不随 v2 改变（见 §6.2 L-3） |
| R-b 逐 rank `growth_tracker` 真机留痕 | **仍 unverified** | v2 不覆盖组 R 的该子项 |
| 跨桶 / 4032 族 / 尾窗 / fp32 之外的精度档 | **仍 unverified** | 单桶 896×1200（4200 族）24 样本 |
| 「多卡与单卡等价」 | **不成立、不得声称** | v2 只主张"差异不超过同卡重跑抖动"；默认 `shuffle=True` 下仍不等价（§14） |

### 19.6 支撑事实（t45，必须与裁定一起读）

| 事实 | 数值 |
| --- | --- |
| **fp32 同配置重复并非逐位**（权重 rel L2） | single `5.634e-08`（max_abs 1.832e-06）/ dual `5.492e-08`（1.446e-06） |
| **fp32 同配置梯度层地板** | single `5.703e-07` / dual `5.759e-07`（392 张量中 **200 张逐位相同**） |
| 归约分解：2 步等时距 | `singleGA2` vs `dual2` = **`1.179e-06`** |
| 归约分解：10 步等时距（隔离 all-reduce） | `singleGA10` vs `dual10t` = **`7.344e-05`**（= S-d `1.5055e-04` 的 **49%**）⇒ **all-reduce 与形状差两者叠加**，不存在唯一源 |
| 单步 S-d（权重） | `6.726e-06` ⇒ 只能称「**单步权重等价（符号层）**」（Adam 首步 `m̂/√v̂` ≈ sign，幅值被压掉），**不得写成梯度等价** |
| 放大机制 | AdamW 符号化归一：10 步放大 **1393×**（几何 **2.24×/步**，lr `1e-4`） |
| **F-T45-1（low，按收口不修）** | `stream_audit` 已落盘但四次跑全为 `gate_open=true, loaders=[]`：主 loader 用的是 bootstrap 的 `_PinnedOrderSampler`，`AuditedIdentitySampler` **未被采用** ⇒ 审计计数采不到，t43 的 `passes/duplicates` 仍只能由 stem 流反推 |

---

## 19. 旧数字作废清单（逐条，含替代数字）

**作废的共同原因**：全部来自**违反契约 §13 V-2 第 2 条**（"关闭 `caption_dropout`/`shuffle` 类随机"）
的配置——训练 loader 硬编码 `shuffle=True`、CLI 无开关（§14），再叠加与 DataLoader 无关的
`shuffle_buckets()` 重排（§12）⇒ 两侧 item 排列长度不同（12 vs 24）且互不相关，**对照双方消费的不是同一批样本**。

| 作废数字 | 出处 | 当时身份 | 作废原因 | 替代（pinned 后正确数字） |
| --- | --- | --- | --- | --- |
| **`1.219e-02`** | t13（fp16）S-d | 曾判超 `2e-3` ⇒ failed | 取数错位（未关 shuffle，`randperm(12)` vs `randperm(24)` + `shuffle_buckets()` 重排） | **`1.5055e-04`**（fp32 pinned，C3） |
| **`1.193e-02`** | t22（fp32）S-d | 曾判"协议缺陷坐实，非精度问题" | 同上（fp32 路径同样 `shuffle=True`） | **`1.5055e-04`**；且"协议缺陷坐实"结论**一并作废** |
| **`3.434e-03`** | t13（fp16）S-g | 曾判"续训未逐位" | 同上（两臂数据序均错位，且跨 epoch 重排长度不同） | **`3.694e-05`**（fp32 pinned，C4） |
| **`3.106e-03`** | t22（fp32）S-g | 同上 | 同上 | **`3.694e-05`**（比 v1 数字小 **84×**） |
| t13/t22 组 R 的 `window_total_loss` 判定 | t16/t17 摘要 | 曾报 `rank_digests.equal=false` | 字段口径错误（rank-local loss 被纳入等值判定，F-T17-2） | 已由 t18 排除该字段（计数器仍参与） |
| 一切"pinned 之前"的 S-d / S-g 数字 | t13 / t22 / t31 前提门失败轮 | 曾用于 failed / unverified 判定 | 前提门未通过 ⇒ **无有效测量** | 见上行两条；其余一律引用 t43/t45 |
| **仍有效、不作废** | t17 三点离散度 `2.040e-04` / `3.225e-04` / `3.988e-04` | **噪声地板**（不是 S-d/S-g 结果） | — | 它们是 v2 的 C3/C4 参照集，用途是"定义什么可分辨" |

> 量化关系：旧的 `1.219e-02` 与新的 `1.5055e-04` 相差约 **81×**，其中绝大部分是取数错位，
> 剩余 `1.5e-04` 才是 fp32 归约顺序残差经 Adam 放大（t43 亦给出同量级判断：约 79× 来自取数）。

---

## 20. 本合同阈值的来源与失效教训

### 21.1 来源（可复核）

| 事实 | 证据 |
| --- | --- |
| 契约由**上一轮 AI 研究阶段**产出，是**纯源码对照研究** | 上游 [`lulynx_multigpu_comparison_20261107.md`](lulynx_multigpu_comparison_20261107.md) 第 14 行自述：「**只做源码比较。未运行任何测试、探针、训练；未安装依赖；未使用 GPU。**」；其状态行亦写"未运行任何多卡训练" |
| 契约成文与入库时间 | 契约 mtime **2026-11-07 19:35:08**；由用户在 **2026-11-08 03:29** 提交入库，commit **`2e7da55e`**，该 commit 首次新增该文件（**981 行 / 79,443 B**） |
| `≤1e-6` 的分布 | 在**受版本控制的文档与源码中**，门槛式 `≤1e-6` **只出现于契约第 804 行一处**；上游对照报告 **0 次**、研究计划 **0 次** ⇒ **无推导、无实测依据**（`output/` 下的 run 报告与 captain 日志里出现的 `≤1e-6` 都是**引用**契约，不是独立来源） |
| 上游计划对阈值的态度 | 研究计划只写"rel-L2 门槛"而**未给数值**；数值只在契约里第一次出现 |

### 21.2 失效教训（按严重度）

1. **验收阈值必须落在"可测量且可分辨"的指标上**：`1e-6` 落在"两种浮点求和顺序逐位相同"的口径里，
   而 fp32 加法不满足结合律 ⇒ 该类阈值**在原理上不可达**，写进合同就是把"必然失败"制度化。
2. **必须先测噪声地板，再定界**：同一张卡 fp32 同配置重跑就有 `5.634e-08`（权重）/ `5.703e-07`（梯度）
   的抖动；fp16 更到 `2–4e-04`。**先量化"什么是噪声"，才能定义"什么是差异"**。v2 的 C3/C4 就建在这条上。
3. **rel L2 有分母/方向陷阱**：`ΔG` 与 `G` 近正交时，模长几乎不变（`3.7e-09`）而逐元素已变（`6.34e-05`）；
   `lora_up` 零初始化时同理（分层 rel 放大 28.9×/41.4×）。**能用 ULP / max_abs 的口径，不要用 rel L2 定门。**
4. **换阈值前必须先量它，哪怕是"保守"的阈值**：本次修订第一版曾把替换判据写成"梯度 ≤ `1e-6`"，
   实测 `6.341e-05` ⇒ **又一个不可达阈值**，当轮即改（见 §22 第 3 条）。
5. **门要写在"前提成立"的语境里**：v1 的 S-d/S-g 没写"前提门不成立时如何记"，导致前提未通过时只能
   在 `failed` 与 `unverified` 之间反复摇摆（§13.2）。v2 的 C1 明确"任一不满足 ⇒ 记 unverified"。

---

## 21. `pinned_order` 正名：可复现的确定性数据顺序（产品特性）

`ANIMA_MULTIGPU_PINNED_ORDER` **不是"为通过合同而加的补丁"**，而是产品特性：
**可复现的确定性数据顺序（reproducible deterministic data order）**。

| 维度 | 内容 |
| --- | --- |
| 开关 | env `ANIMA_MULTIGPU_PINNED_ORDER=1`；**默认关** |
| 覆盖范围 | ① `bootstrap.py` 主训练 loader（`_PinnedOrderSampler` 替代 `shuffle=True`）；② `loop.py` stage-switch 重建的 loader；③ cursor dataloader；④ **跨 epoch 重钉**（钩子下沉到 `DatasetBuckets.shuffle_buckets()`，同时覆盖建库期 `create_buckets` 与 `make_buckets`/`feature_sidecars` 路径）；⑤ 门控开启时 descriptor 排序为 `(bucket_index, batch_index)` 自然序 |
| 明确未覆盖 | val loader（本已 `shuffle=False`）、`auto_block_swap/probe.py`（经 `DatasetGroup.set_current_epoch` → 同一钩子**间接覆盖**，但探针仅单进程 CUDA）、自建 loader 的非训练消费者 |
| 默认路径 | **字节不变**：门关时 `apply`/`repin` 返回 `0`、descriptor 逐项不动、`RandomSampler` 与索引序列逐项相等（t39/t42 均有负对照） |
| 用途 | **多卡可复现**（同一 `(seed, epoch, world_size, per-device batch)` 下取数顺序确定）与**故障重跑**（中断后继承同一数据序）；也是跨 world_size 对照实验能够成立的前提 |
| 产物自证 | `hot_run.plan_hot_run(..., pinned_order=True)` 把门写进 `facts["pinned_order"]` **与** `facts["launch_env"]`，`verify_pinned_gate(plan, env)` 对"手 export 与 plan 不一致"直接 `HotRunRefused`；`execute()` 每 phase 结果回读该值 ⇒ **门开着这件事由产物自证，不靠口头声明** |

**为什么正名重要**：把它写成"合同补丁"会让人以为默认路径有缺陷、门开才是"正确模式"。
事实相反——**默认 `shuffle=True` 是训练语义的一部分**（每个 epoch 换序），pinned 只服务于
"需要逐步可复现/可对照"的场景。两者都应当存在，且**默认保持训练语义**。

---

## 22. captain 判决失误清单（留给以后的教训，不美化）

| # | 预判 | 实测 | 错因 |
| --- | --- | --- | --- |
| 1 | step0 梯度张量 rel L2 在 **`1e-8`** 量级 | **`6.341e-05`**（max_abs `6.023e-07`，196/392 逐位相同） | 只看模长相对差（`grad_norm_total` `3.7e-09`）就外推张量级，忽略"`ΔG` 与 `G` 近正交 ⇒ 模长几乎不变、逐元素已变" |
| 2 | all-reduce 是**唯一**差异源 | **只占 49%**（`singleGA10` vs `dual10t` = `7.344e-05`，S-d 为 `1.5055e-04`） | 把"同形状输入"误当"同归约树"；形状差（`batch2` 一次 GEMM vs `1+1` 两次）同样贡献 |
| 3 | 第一版替换判据写"梯度层 ≤ `1e-6`" | 实测 `6.341e-05` ⇒ **又一个不可达阈值** | 换阈值前没有先量它；已按纠偏改为 **ULP 口径**（C2） |
| 4 | 契约内的 verify 命令 | **三次写错测试文件名**（t8 用了 `tests/test_training_window_replay.py` / `tests/test_training_fp16_nan_static_equivalence.py`；t23 用了 `tests/test_training_fp16_nan.py`；t13 原 verify 结构性只能 skip） | 真名无 `training_` 前缀（`tests/test_window_replay.py`、`tests/test_fp16_nan_static_equivalence.py`、`tests/test_fp16_residual_safe.py`）；成员均拒绝按错名执行，未把 exit 4 记成通过 |

**共同教训**：**判决阈值与 verify 命令都必须"先跑一次再写死"**。凡涉及"差异有多大"的判断，
先取得同配置噪声地板与被测口径的实测量级，再决定门的形态（ULP / max_abs / 地板倍数），
不要用"看起来很保守"的绝对值代替测量。

---

## 23. 修订记录

| 日期 | 内容 | 执行者 |
| --- | --- | --- |
| 2026-11-08 | 收口（t46）：新增 §18 P0-5/P1-2 在契约 v2 判据（C1–C4）下的明确裁定（**通过**，附逐条实测、边界说明与仍开放子项）、§19 旧数字作废清单（含替代数字）、§20 合同阈值来源与失效教训、§21 `pinned_order` 正名（可复现的确定性数据顺序）、§22 captain 判决失误清单；并对 §0/§3.1/§3.2/§5.4/§6.2/§13.2 加"已由 v2 裁定取代"标注。未改源码/测试/配置/阈值，未跑 GPU | release-integrator（集成任务 t46，只改文档） |
| 2026-11-08 | 增量（t37）：新增 §12 取数顺序三层缺陷与其修复、§13 四个数字口径纠正与 P0-5/P1-2 改判 `unverified`（含待 t35 填入的占位块）、§14 合同只能靠 env 门控满足（F-T31-3）、§15 验证体系盲区（F-T33-2）、§16 流程与 captain 侧缺陷账、§17 F-T3-1 旧结论降级；并对 §0 / §3.1 / §3.2 / §5.4 / §6.1 / §6.2-L-1 作**原地口径更正**（保留原判、不删原文）。未改源码/测试/契约/阈值/预算，未跑 GPU | release-integrator（集成任务 t37，只改文档） |
| 2026-11-08 | 首版：在 t1 承接账本、t2（v1 t25 收尾）、t8/t11/t14/t15/t16/t18/t20 修复、t10/t12/t13/t17 分层证据与 t21 独立 review 之上，集成为逐参考点矩阵 + RUN 账本 + 未通过项与偏差登记 | release-integrator（集成任务 t6，只改文档，未改源码/阈值/预算） |
