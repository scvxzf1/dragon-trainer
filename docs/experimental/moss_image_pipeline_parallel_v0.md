# MOSS-Image 流水线并行 v0

状态：实验（库级实现已独立验收：CPU/Gloo 回归与双 V100 CUDA tiny 编排/事务/
同窗口恢复均已通过（t18 独立验证 exit 0 + t4 最终审阅 pass + t5 集成交付）；
主训练入口仍 fail-closed，真权重训练/完整 trainer 续训不在本次验收范围）
适用版本：分支 `codex/moss-pp-v0-20261112`（基线 `origin/dev` `6610f710`）
入口命令：见文末；主训练入口：MOSS 族已接线专用 PP 训练会话
（见 [moss_image_pp_trainer_wiring.md](moss_image_pp_trainer_wiring.md)）；
其余族维持 `pp_trainer_path_unavailable` 具名拒绝（fail-closed 不变）
相关代码：`library/training/pp/moss_runtime.py`、
`library/models/moss_image/pipeline_parallel.py`、
`library/models/family_registry.py`、`library/training/pp/`、
需求契约：`docs/proposal/moss_image_pipeline_parallel_v0.md`（需求冻结原文）

一句话：MOSS-Image 复用 Anima PP v0 的调度/事务/checkpoint 骨架，落地了族专属的
双 stage 切分、真实文本长度 conditioning 与分 rank checkpoint；MOSS 发布与 Anima
同一个 stage-local runtime backend 常量，`library.training.pp` 按族分发组件。
MOSS 的主训练循环已接线（专用会话派发，绝不回退 DDP）；其余族任何训练入口遇到 `pipeline_parallel` 仍拒绝。

## 支持范围

| 维度 | v0 支持 | 拒绝方式 |
| --- | --- | --- |
| 模型族 | `moss_image`（共享 `torch_pipelining_1f1b` backend） | 其他族维持原具名拒绝 |
| stages / schedule / split | 2 / `1f1b` / `balanced`（registry 计划：42 层 → 0:21 / 21:42） | `pp_config_invalid` 族 |
| latent | 4D `[B,128,H,W]` packed（packed 边无奇偶要求，偶数约束仅 pre-pack 32ch） | `pp_shape_contract_violation` |
| 精度 | 纯 fp16 + `GradScaler`，fp32 LoRA master weights | `pp_mixed_precision_unsupported` / `pp_fp16_master_weights_required` |
| 文本 | per-sample 真实长度 mask/RoPE；wire 容量 `token_cap = H*W + max_text_tokens` | 非二进制 mask、空样本、超容均具名拒绝 |
| block swap / compile | 与 PP 互斥 | `pp_config_invalid` |
| token 桶 | 一个 run 一个 `(batch, H, W)` wire 形状；尾 batch/不可整除拒绝 | `pp_token_bucket_mismatch` |

## 族边界（MOSS 专属组件，不复制 Anima）

* stage 0：`video_embedder`、`text_embedder`（refinement，真实长度 RoPE）、
  `shared_block_modulation`、`layers[0:21)`；stage 1：`layers[21:42)`、
  `final_norm_video`、head 门控、`final_linear_video`、unflatten。
* boundary 三槽：joint hidden（可微，grad 回到 stage 0 的 text 路径）、
  per-sample 真实长度（float32，torch pipelining 对收到的张量强制
  `requires_grad`，int 不可用）、共享 modulation（冻结 detach）。frozen slots
  = `(1, 2)`，由 `FrozenBoundaryPipelineStage` 成对覆盖保证不挂起。
* transport padding 永不进 attention：接收 stage 按真实长度切片重建 per-sample
  序列；RoPE 由各 stage 从 `(H, W, S_i)` 本地重建，与生产 forward 同源。
* adapter 归属按真实 `lora_unet_layers_<N>_proj_[qkvo]` 名解析，并对已 apply
  网络做 target 模块几何交叉校验；Anima 的 `lora_unet_blocks_` 正则不适用。
* checkpoint：`moss-image-pp-stage-checkpoint` 新格式（与 Anima 格式双向具名拒绝，
  读前校验），manifest 记录 family/层计划/wire 形状/adapter 归属表/max_text_tokens
  /dtype。fp16 增长策略（factor/backoff/interval）经 last-stage GradScaler state
  持久化，load 后向所有 stage 重新协商（restore 端构造参数不同也不会漂移）。
* step 前各 rank 本地校验 + 一次 CPU `all_gather_object` 协商，任一失败两 rank
  以同一 `pp_*` code 在任何通信前对称拒绝。

## 复用与共享面

`pp.contract` / `pp.transactions` / `pp.schedule`（调度与 loss）/ `pp.stages`
（frozen-boundary wire）/ `pp.checkpoint*`（run root/manifest/协商/apply）原样或
最小扩展复用；`pp.layout`（Anima 5D）不复用。共享扩展保持最小：最初两处
（`PpFp16Transaction.restore_policy` 策略恢复与 checkpoint 格式按 family 分流的
参数化）加上后续修复轮新增的 `pp/checkpoint_validation.py` 纯校验 helper 与
`validate_master_weight_dtype` 的可选 device 检查（均为纯校验/可选参数，不扩
架构、不加 optimizer 注册框架）。MOSS 侧计算 dtype 政策：boundary dtype =
activation dtype（`_Block`
不提升 residual stream）；`PpComputePolicy.boundary_dtype` 是 Anima 专属语义。

## 验收与边界

* 已执行（CPU/Gloo，共享解释器，`CUDA_VISIBLE_DEVICES=''`）：
  `tests/test_moss_image_pp_runtime.py`、`tests/test_moss_image_pp_checkpoint.py`、
  `tests/test_moss_image_pp_wiring.py`（`-m 'not hardware'`），含：
  staged vs 生产 forward 逐位相等（fp32，非前缀 mask、奇数 packed 边、非零 LoRA）、
  padding NaN/inf 双层不变性、双 rank loss/梯度 vs 非 PP 参考、单侧 overflow 双 rank
  skip、fresh 进程 restart/replay（per-step loss 位相等 + 增长边界跨越 + 策略从
  checkpoint 恢复）、双向格式拒绝、协商对称拒绝。
* 冻结 hardware node：`tests/test_moss_image_pp_runtime.py::
  test_two_rank_cuda_fp16_runtime_and_resume`（双 CUDA tiny、fp16 autocast、
  fresh-pair 重启、≤2GiB/卡），由 verification 任务在授权 GPU 上执行；本实现任务
  未使用 GPU。
## 交付状态（集成核对，2026-11-13）

* 交付现场：独立工作树 `krea2-pp-moss`，分支 `codex/moss-pp-v0-20261112`，
  HEAD=6610f710（基线 origin/dev，无新 commit）；17 个已跟踪文件修改
  （356+/66-）+ 14 个新增未跟踪文件，全部未提交。
* 验收绑定（review 实际 hash）：`tests/test_moss_image_pp_runtime.py`
  sha256=3aafdbbc8583e957af3e78d5bf08bf9b185655928e1d2288f048c750a09e8633、
  `tests/test_moss_image_pp_cuda_assertions.py`
  sha256=0d18daa508f3495e3be8752881a377c3aed922480db3aba74f0b561c94c423f8、
  本文档在 t17/t18 审阅绑定时点的 sha256=bf4e8bda6075eae0cfb1fcc6b14bf999776804703faaa96747480c1c2654efd2；
  其后的收尾措辞修正（状态/共享扩展/时点说明）不改变验收语义，最新字节以团队
  记录中的最终 SHA 为准。
* 分层语义（勿混淆）：
  - CPU 非PP：staged forward 与生产 forward 的预测逐位相等；双 rank PP 的 loss／梯度
    按预登记容差对照非 PP 参考通过，不能将梯度容差通过写成逐位相等。
    库级 PP=2/1F1B/FP16+FP32 master 编排／事务／checkpoint 有 CPU/Gloo 证据。
  - GPU L2 仅覆盖 tiny 编排/事务/同窗口恢复/overflow 对称/显存峰值：真实 GPU
    证据为 t18 单次授权双 V100 执行（exit 0，31.25s），落盘 JSON
    `/tmp/pytest-of-scv/pytest-102/test_two_rank_cuda_fp16_runtim0/cuda_node_evidence.json`
    （峰值 allocated 23.23/21.39 MiB、scale 32768/tracker 2/interval 4 恢复一致、
    双 rank overflow skipped）；CPU mock 同名 JSON（pytest-100/101）不是 GPU 证据。
    不覆盖真权重训练、独立 GPU 梯度数值、性能/吞吐/收敛。
  - 不恢复 RNG/loader/LR scheduler；production trainer 继续 fail-closed
    （`pp_trainer_path_unavailable`）。
* 历史失败/自动上限记录保留（t10/t12 各轮、F9 挂起原因未知、web preflight 2 例
  环境性失败）；本次收尾未 stash/commit/push/deploy（早期 t2 的 stash 流程偏差另有记录）。
  基线仍为 `6610f710`；本地
  `origin/dev` 跟踪引用其后前进了 8 个提交，本次收尾未 fetch/pull/rebase，
  变更现场保持未提交。团队 escalated/自动上限历史如实保留，系统 delivery 闸门
  状态不由本文档声明。
* 未验证/不支持：真实权重长训练、正式训练循环接线、RNG/loader/LR 完整 trainer
  续训、通用推理、其他模型族运行时。阈值原则：数值阈值来自 dtype 与非 PP 参考
  噪声基线（`tests/moss_pp_test_support.py` 记录），不从候选实现误差反推。

## 修复轮（repair-round-2，C1-C5）真实证据

> 时点说明：以下修复轮各节（C1-C5、F1-F10、t11/t13/t14/t15/t17）是**历史时点的
> 过程记录**，其中"尚未通过/待独立验收"的表述均已由后续轮次解决——最终以
> t18 独立验证（双 V100 CUDA tiny exit 0）与 t4 最终审阅 pass 为准；团队
> escalated/自动上限历史如实保留，不因最终通过而重写或清零。

* C1 逐样本 precheck：step 协商 precheck 现按样本校验有效 token 数（长度 [4,0] 这类
  总数合法但存在空样本的输入在进入 schedule 前被全部 rank 以同一
  `pp_stage_contract_violation` 拒绝；`test_empty_sample_refused_symmetrically_before_schedule`）。
* C2 构造前对称协商：本地构造阶段（校验/stage module/optimizer/boundary metas/
  FrozenBoundaryPipelineStage）全部结果先经一次协商，再进入 policy/frozen boundary
  collective；optimizer factory 抛错 → 双 rank 同 code 快速退出
  （`test_build_failure_negotiates_before_any_collective`）；half-transactional
  pipeline（单 rank 无 optimizer）在 transaction 广播前被能力协商双双拒绝
  （`test_single_side_missing_transaction_refuses_symmetrically`）。
* C3 checkpoint 恢复分四段：纯本地 validate（`validate_stage_checkpoint`，零
  mutation）→ 全 rank validate 协商 → 本地 apply → 全 rank apply 协商 → policy
  collective（不再把带 collective 的函数放进可能被单 rank 跳过的 try 段）。可预测
  mismatch（optimizer 组不符、scaler payload 缺失）双 rank 拒绝且 adapter 状态
  逐位不变（`test_single_side_optimizer_mismatch_refuses_symmetrically`、
  `test_single_side_scaler_state_missing_refuses_symmetrically`）。Anima 旧格式与
  入口兼容由 `tests/test_training_pp_checkpoint.py` 等回归覆盖。
* C4：`test_moss_checkpoint_restart_and_replay` 用同一训练轨迹的保存（step 2）/
  未中断 continuation（steps 3-4）/fresh-pair replay（steps 3-4），interval=4，两条
  独立轨迹各自跨过真实增长边界（scale 翻倍、tracker 归零），恢复端构造 interval=777
  与 checkpoint 的 4 不同；断言 policy/tracker/scale、逐步未缩放 loss（按各步
  `outcome.scale` 还原）及更新后 snapshot digest 逐字段相等。
* C5：CUDA tiny node 重写为同窗口比较——保存后记录未中断 continuation per-step
  `{scale_used, losses}`，fresh resume 对比相同窗口（steps 4-5），逐步用该步实际
  scale 还原 loss（`MOSS_CUDA_REPLAY_RTOL=1e-2` 为 dtype 先验预登记；不得依据候选误差
  放宽，CUDA 独立验证尚待取得）；worker/run_pair 参数契约已修正。
  本任务仍只做 CPU 验证，CUDA 节点执行归 verification。
* 本轮回归（`CUDA_VISIBLE_DEVICES=''`，真实退出码 0）：MOSS 三文件 39 passed +
  1 deselected（/tmp/t7v1.log）；Anima 共享五文件 153 passed（/tmp/t7v2.log）；
  `git diff --check` 干净。

## 修复轮 3（repair-round-3，F1-F3）真实证据

* F1 输入协商：step precheck 把 timesteps 必要性/逐样本形状改为双 rank 通用校验
  （rank 0 也需要 timesteps），末 stage 增加 target None/shape 校验；负向双 rank
  用例 `test_missing_target_refused_symmetrically_before_schedule`、
  `test_bad_timesteps_refused_symmetrically_on_both_ranks`（均同 code 在进入
  schedule 前拒绝）。
* F2 checkpoint 内容校验：新增 `library/training/pp/checkpoint_validation.py`
  纯校验 helper（PP scale 正有限、growth tracker 合法、GradScaler 必要键/类型/
  范围、增长策略合法、AdamW 已知 moment 键逐参数形状、step 合法标量；未知合法
  因子化状态不做一刀切参数形状拒绝），接入 `validate_stage_checkpoint` 纯校验
  阶段。负向用例：一个 save pair + 六种内容变异（NaN/负 scale、负 tracker、缺
  scaler key、非法 growth_interval、moment 形状不符）逐一 fresh restore pair，
  全部双 rank `pp_checkpoint_state_mismatch` 且 adapter 状态逐位不变
  （`test_content_mutation_refused_symmetrically_before_apply`）。
* F3 的历史报告曾声称两个 helper 均使用唯一 rendezvous，并记录两次 42 passed。
  t10 源码复核发现普通 `run_pair` 仍复用固定路径；因此旧通过次数不作为该缺陷
  已修复的证据。下节记录用户授权的人工修复，保留先前失败与返修历史。
* 本轮回归（`CUDA_VISIBLE_DEVICES=''`，真实退出码 0）：MOSS 三文件 42 passed +
  1 deselected ×2 独立进程；Anima 共享五文件见 /tmp/t9v2.log；`git diff --check`
  干净。CUDA 实际验收归 t10。

## 用户授权的主代理人工修复（t11，F4/F5）

自动返修在 t10 到达上限后，用户明确授权主代理接管；旧失败、过强声明及
升级记录保留，不将人工修复视为独立审阅已经通过。

* F4：两个 pair helper 共用调用局部的执行器，每次用新目录隔离 FileStore 和
  每 rank 的异常回执；调用者的模型结果输出路径保持不变。成功／预期失败均
  检查真实子进程退出码，旧 `.error`、单 rank 失败或被父进程终止的超时不能
  让负向用例通过。新增[同步与证据回归](../../tests/test_moss_image_pp_harness.py)。
* F5：[纯校验](../../library/training/pp/checkpoint_validation.py)要求 scaler 的
  `_growth_tracker` 存在且合法，并在 apply 前拒绝 PP/scaler 不一致的正 scale；
  不强行要求两个独立 tracker 相等。新增[纯校验回归](../../tests/test_moss_image_pp_checkpoint_validation.py)。
* [分进程负向用例](../../tests/test_moss_image_pp_checkpoint.py)的每个内容变异从
  有效 checkpoint 的独立副本开始，避免前一缺陷掩盖后一缺陷。缺 tracker 和
  scale 不一致均验证两 rank 同时拒绝，adapter／optimizer／scaler 的完整状态
  摘要保持不变。
* 主代理 CPU-only 自查，pytest 直连、每组实际退出码均为 0：快速回归 **14 passed**；
  MOSS checkpoint **11 passed**；MOSS runtime/wiring **31 passed、1 deselected**；
  Anima checkpoint/runtime/import **92 passed**。deselected 为尚未执行的 CUDA 节点。
* 完整独立验收、双 CUDA tiny 和最终审阅仍待完成；上述结果不代表真实权重训练、
  端到端吞吐或长期收敛验收通过。

## 主代理继续修复优化器状态边界（t13，F6/F7）

t12 保留 F4/F5 已通过证据，但双方独立 CPU 探针又复现了两处校验遗漏：
非连续保存 ID 会漏检 moment 形状；NaN step 可导致有限梯度更新后参数 NaN。
原通过数量不覆盖这些新边界，不能作为 checkpoint 整体放行的依据。

* F6：[校验 helper](../../library/training/pp/checkpoint_validation.py)按保存组内 ID
  与 live 参数的位置对应建表，匹配方式与 PyTorch 加载一致；重复／未引用的 ID
  具名拒绝。合法非连续 ID、不同参数形状及 AMSGrad 状态有正向加载回归。
* F7：step 必须为有限非负实数标量；支持合法单元素 tensor。初始化后的 Adam／AdamW
  必需状态键也在 apply 前校验，其他优化器的因子化字段不做参数形状一刀切。
  本轮不宣称穷尽所有优化器特有超参数或状态格式。
* [纯校验回归](../../tests/test_moss_image_pp_checkpoint_validation.py)实测 **34 passed、exit 0**。
  [双 rank 内容变异](../../tests/test_moss_image_pp_checkpoint.py)加入重编号错误 moment
  和 NaN step，仍从独立有效副本恢复，检查双方同码拒绝及完整状态不变；
  MOSS 两个关键节点与 Anima checkpoint 全文件组合实测 **32 passed、exit 0**。
* 不使用 GPU、不改变数值阈值；修复完成后原 t12 使用新 attempt 独立复验，旧失败与
  自动循环到顶记录保留，CUDA／最终审阅尚未通过。

## 主代理修复 CUDA master 放置（t14，F8）

t12 attempt2 在授权卡空闲后实际执行 CUDA 节点，暴露 LoRA 参数仍在 CPU：
base 已迁移并不自动迁移独立 network。旧两次运行均失败／超时，不作为数值证据；
无代码变更的第二次重跑不能用于绑定版本，该措辞已撤回，版本绑定使用 SHA256。

* 两个 CUDA worker 共用的 helper 在 builder／optimizer factory 前显式迁移 network，
  只移 device，保持 FP32 master 和冻结 base 的精度。
* [公共 master 校验](../../library/training/pp/contract.py)支持可选 device，旧 Anima
  调用不变；MOSS 在 factory 前及优化器绑定后检查，错放由原构造协商双方具名拒绝。
* [CPU/meta 与放置顺序回归](../../tests/test_moss_image_pp_devices.py)：**10 passed、exit 0**，
  包括错放 rank 不调用 factory／后续 policy、真实 tiny 的 base FP16/master FP32
  及 loss 记录缺失／非有限时在 optimizer 前拒绝。它们不是 CUDA 执行证据。
* CPU runtime/wiring、Anima runtime、import boundary 组合：**93 passed、1 deselected、exit 0**。
  后续只加强 CUDA-only 断言和修正文案：逐步完整有限 loss、overflow 前后 adapter／optimizer
  hash 不变、allocated/reserved 分别限额；固定先验阈值不变。loss 记录 guard 有 CPU 回归，
  其余 CUDA-only 断言及 GPU 正确性仍待独立实跑，不把 deselected 算通过。
* 修复后按最新字节、即时资源闸门，由原 t12 新 attempt 单次验证 CUDA；不无改动重跑，
  不声称真实权重训练或总体交付已完成。

## 超时预算与生命周期证据（t15，F9）

t12 attempt3 的 CPU/CUDA timeout 仍是失败；F9“destroy_process_group 死锁”因果已撤回：
日志仅为 bootstrap traceback，不能证明清理栈。主代理在未改库源码／线程数时，单对保存＋
损坏 tracker 诊断 **19.07s、exit 0**；完整内容变异诊断也 exit 0，各 rank 销毁约 20ms。
这不消除潜在环境／退出等待问题，不将原独立超时洗成通过。

* 修复的是可证实的预算冲突：旧 helper 每个 child 等 120s，外 CUDA 节点却仅 60s。
  [harness](../../tests/moss_pp_test_support.py)现在每 pair 共用绝对 deadline，CUDA 两阶段
  共用 45s，保留有界清理／报告空间；外部 timeout60 和全部步骤／阈值不变。
* 各调用独立目录写进入／返回／异常生命周期与延迟线程栈；真实 exitcode 在 cleanup 前快照。
  只 TERM/KILL 本调用 Process，worker 返回或清理后的退出不能充当真正成功／预期坏状态拒绝。
  不改 checkpoint collective、destroy 顺序或训练数学。
* [harness/device 回归](../../tests/test_moss_image_pp_harness.py)：**19 passed、exit 0**。
  内容变异／MOSS restart／runtime/wiring 组合：**33 passed、1 deselected、exit 0**。
  GPU 尚未通过；不把实现者自查、diag plugin 通过或 deselected 算独立 L2 验收。
* 新 t16 在 t15 完成前以完整当前合同建立，避免旧 t12 prompt 混用旧 verify；原失败、
  review／integration 任务及自动上限记录保持，由独立成员单次 GPU 取证后再审阅交付。

## CUDA loss 判据完整性（t17，F10）

t16 的正确 CPU 合同已独立通过（19＋33）；CUDA 两阶段实跑 31.95s，前置 dtype、恢复、
scale/tracker/digest 与峰值断言完成，但最后比较引用未定义常量，节点 **exit 1**。
这仍不是完整 L2 通过；F10 原失败保留，不借 partial 证据交付。

* 将此前登记的 FP16 accumulation prior **RTOL=1e-2** 实际定义为模块常量，
  不取本轮候选误差拟合；`zip` 前双方 loss 长度都必须为 last rank 的 2，非 last 为 0。
* [CPU 父断言回归](../../tests/test_moss_image_pp_cuda_assertions.py)mock 两个 worker 与加载，
  直接调用原 hardware 节点的父比较：**7 passed、exit 0**，正常／容差内成立，
  空、截短、额外 loss 或容差外拒绝。fake JSON 仅为 CPU 测试文件，不作 GPU 证据。
* runtime 测试入口收集 **17/18、1 deselected、exit 0**；这是入口／导入检查，不算17项执行通过。
  未改库、步骤、scale、同窗口参照或内存预算；t18 按新完整合同单次独立 GPU 复验后才审阅交付。
