# MOSS-Image 正式主训练 PP 接线与训练级恢复契约（需求冻结 v1）

状态：需求冻结（wiring round 1，本文档只冻结契约与验收合同，不包含实现）
适用版本：分支 `codex/moss-pp-v0-20261112`，基线 HEAD `6610f710`
入口命令：见 §8（后继 verification 任务的固定入口）
相关代码：`library/training/cli_entry.py`、`library/training/train_session.py`、`library/training/trainer_bootstrap_mixin.py`、`library/training/bootstrap.py`、`library/training/loop.py`、`library/training/checkpoints.py`、`library/runtime/accelerator.py`、`library/runtime/launch.py`、`library/training/pp/`、`library/models/moss_image/`、`library/models/family_registry.py`、`library/training/moss_compat.py`
前置文档：[moss_image_pipeline_parallel_v0.md](moss_image_pipeline_parallel_v0.md)（库级 PP v0 契约，其 §1–§7 冻结条款继续生效，本文不重复）

---

## 0. 结论摘要（冻结项）

1. **接线目标**：把已验收的库级 MOSS PP runtime（`library/training/pp` + `library/models/moss_image/pipeline_parallel.py`）接入正式主训练调用链，使 `model_family=moss_image` + `pipeline_parallel` 的配置能通过 **`train.py` → `AnimaTrainer.train` → `run_training_session`** 这条唯一生产入口真实运行；不是放开开关，而是新增一个有实际实现支撑的 PP 训练派发分支。
2. **唯一 PP 派发点**：`library/training/train_session.py::_run_training_session`（当前 L121 起的会话编排内）。派发点在 `verify_training_args`/compat 检查之后、DDP 专属的 optimizer/dataloader 组装（`bootstrap.setup_optimizer_and_dataloader`）之前分叉；禁止在 `cli_entry`、`bootstrap`、`loop`、`checkpoints` 内出现第二个 PP 分支。
3. **fail-closed 面不降级**：`gate.require_trainer_topology`、`gate.require_accelerate_checkpoint`（`checkpoints.publish_state_with_bundle:707-713`）与 `gate.require_pipeline_runtime` 的**guard 函数与 DDP 内部保护语义全部保持**，不得改成 no-op、不得删除；也不得让合法 PP 配置进入普通 `accelerator.prepare`/`accumulate`/`save_state`。接线不是"所有 guard callsite 原样不变"：`cli_entry:86-100` 当前对一切 PP 配置具名拒绝（`pp_trainer_path_unavailable`），接线后 CLI 对**满足 MOSS trainer admission 的配置**改为走 PP 预检分支而非拒绝；唯一执行派发仍在 `train_session::_run_training_session`。其余 bootstrap/loop/checkpoints callsite 的 guard 继续原样生效（PP 分支根本不该到达；若到达即 bug），允许在这些层做必要的 backend/预检条件分支，但**不得出现第二套执行器**。PP 训练分支成功派发后不再回到 trainer 的 DDP hotspots——它调 `pp.moss_runtime.build_moss_pipeline_runtime`（原生 4D `MossPipelineRuntime`，非 Anima 5D 路径；MOSS 族 stage 组件由 `moss_image.pipeline_parallel` 提供）自建 stage-local optimizer/1F1B 调度/分 rank checkpoint，并在会话结束/失败时具名退出。
4. **首轮可运行范围（只放行这一格）**：`model_family=moss_image` + `pipeline_parallel`（stages=2 / balanced 切分 / schedule=1f1b / microbatches∈[2, MAX] 且 batch 可整除）/ `mixed_precision=fp16`（plain + last-stage GradScaler）/ plain LoRA（`networks.lora_anima`，`layers.N.proj.{q,k,v,o}`）/ 完整 latent+TE disk caches / 固定单一 wire 形状 / 确定性数据加载。registry 已发布 `moss_image` 的 `runtime_backend=torch_pipelining_1f1b`（`family_registry.py:231-244`），gate 因此会放行该族——**放行即承诺有真实 runtime 支撑**，实现轮不得只改 registry。
5. **训练状态恢复**：库级分 rank checkpoint（`pp.checkpoint.save/load_distributed_checkpoint`）为唯一恢复格式；trainer 级新增 scheduler / RNG / 数据游标 / run identity 的记录与恢复（§5）。旧库级 checkpoint（无 trainer 状态）只能恢复到"可继续训练"，不得宣称精确续训。
6. **非 PP 与旧行为不变**：Anima/Krea-2/Z-Image/Qwen 全部族的非 PP 训练路径逐字节不受影响；`moss_image` 非 PP 训练路径不变。验收包含非 PP 回归。
7. **GPU 边界**：真实主入口 tiny 双 V100 热测仅用固定两卡（UUID `GPU-b4b014b2-953b-28e8-b8c1-425c4a401165`、`GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061`），每卡 allocator allocated/reserved ≤2GiB，单节点 timeout 60s，短步数，tiny 注入模型与临时 cached 数据；不下载真实权重、不触碰用户数据。

## 1. 当前源码调用链（已核对基线 `6610f710`）

正式主训练调用链与各 choke point（行号对应当前工作树）：

```text
train.py __main__
  └ cli_entry.run_training_cli                      cli_entry.py:26
      ├ verify_command_line_training_args           cli_entry.py:43
      ├ resolve_adaptive_precision / adaptive supervisor   cli_entry.py:54-78
      ├ pipeline_parallel_enabled(args)?            cli_entry.py:86
      │   ├ require_pipeline_runtime(args, world_size)     cli_entry.py:94   # PP v0 运行时契约预检
      │   └ require_trainer_topology(args, "cli_entry")    cli_entry.py:100  # F6/F10：训练器拓扑具名拒绝
      └ trainer.train(args)                         cli_entry.py:117
          └ AnimaTrainer.train → run_training_session      trainer_bootstrap_mixin.py:138-141
              └ train_session.run_training_session        train_session.py:111
                  └ _run_training_session                 train_session.py:121
                      ├ validate_entry / verify_training_args / check_training_compat   L127/137
                      ├ _prepare_dataset / cache 完整性检查 / resolve_mixed_precision   L139-227
                      ├ 模型加载（lazy：TE→cache→VAE→cache→DiT+network）
                      ├ bootstrap.setup_optimizer_and_dataloader   bootstrap.py:896-912  # DDP optimizer/dataloader
                      ├ bootstrap.prepare_with_accelerator         bootstrap.py:1099-1128 # accelerator.prepare → DDP 包装
                      ├ loop.run_training_loop                     loop.py:613-626        # whole-model forward + accumulate
                      │   └ loop._run_microbatch                   loop.py:1149-1197      # require_trainer_topology 每微批
                      └ checkpoints.publish_state_with_bundle      checkpoints.py:682-713 # require_accelerate_checkpoint
```

- **Accelerator/process-group 现状**：`runtime/accelerator.py:174-207` 构造 `Accelerator(gradient_accumulation_steps=..., mixed_precision=..., dataloader_config=...)`——这是 DDP 拓扑的资产。PP 分支不经过它拿 process group：PP runtime 用纯 `torch.distributed` group（`pp/runtime._distributed_backend`），`world_size == stages == 2`（`launch.py:50-114` 的 accelerate launch 包装负责把两进程拉起，PP 复用该启动面但不复用 DDP prepare）。
- **trainer 热点的四个 PP 假设**（gate docstring `pp/gate.py:17-48` 已冻结）：`accelerator.prepare` 全网 DDP 包装、单微批=整模 forward、accelerate 拥有 accumulation 缩放、`accelerator.save_state` 全 rank 单目录。接线实现不得触碰这四点，只能绕开。
- **LR 除数**：`gate.lr_scheduler_num_processes`（`pp/gate.py:244-257`）——PP 下计划除数为 1。
- **梯度缩放唯一归属**：`PP_GRAD_SCALE_OWNER = pp.schedule.make_last_stage_loss_fn`（1/n_microbatches 内置于 loss，`scale_grads=False`）；trainer 任何位置不得二次缩放。

## 2. 接线契约（实现轮的接口面）

### 2.1 唯一派发点与分叉时机

- 在 `_run_training_session` 内、`verify_training_args`（compat/moss 合同已跑）之后、`setup_optimizer_and_dataloader` 之前检测 `pipeline_parallel_enabled(args)`；为真且 family==moss_image 且满足 MOSS trainer admission 时进入 PP 训练编排（建议新小模块，如 `library/training/pp/trainer_session.py`，不往热点文件堆业务——AGENTS 反上帝代码守则）。gate 函数本身与 DDP 内部保护不改：`cli_entry` 只是把"满足 admission 的 MOSS PP 配置"从具名拒绝改为预检放行（见 §0.3），执行派发的唯一入口仍是这里。
- 派发分支必须复用 train_session 已完成的前置工作（dataset/caches 完整性、seed、`resolve_mixed_precision`、metadata/session 记账），不得另起一条平行 bootstrap。
- PP 分支职责：构建 `MossPipelineRuntime`（每 rank stage-local，原生 4D wire，非 Anima 5D 路径）、数据侧同步（§3）、stage-local optimizer + LR（§4）、1F1B step 循环、checkpoint/resume（§5）、进度与状态上报（§4）。
- 派发失败（family 无 backend、runtime 构造抛 `pp_*`）一律具名拒绝并中止会话，**不得静默回退 DDP**。

### 2.2 不得做的事（负向冻结）

- 不得把 `require_trainer_topology`/`require_accelerate_checkpoint` 改 no-op、加豁免参数或放宽 `COMPAT_CODE_*`。
- 不得让 PP run 走 `prepare_with_accelerator`、`_run_microbatch` 的 accumulate 路径或 `publish_state_with_bundle`——这些 guard 在 PP 分支里根本不该被到达（若到达即 bug，guard 继续报警是对的）。
- 不得复制/改写 `pp.schedule`、`pp.transactions`、`pp.checkpoint*`。"不复制"不是禁止合同内的最小兼容扩展（如 v0 §9-D 已批准的 `PpFp16Transaction.restore_policy` 与 checkpoint 格式按 family 分流）；禁止的是绕开合同另起一套副本实现。
- 不得绕过 `moss_image.family.compute_noise_pred_and_target`/`forward_for_loss` 语义另造训练 forward。

## 3. 训练级数据/noise 同步（每步两 rank 必须一致）

- **确定性加载**：两 stage 消费**同一个 logical batch 与同一组 noise/sigma**——二者是 pipeline 的两个切片视角，不是 DDP 各消费不同样本。同步手段二选一：rank 0 生成后 broadcast 给 rank 1，或两侧按同一确定性顺序独立重建并逐位校验一致；禁止任何"按 rank 分数据"的 scatter 语义。冻结要求：PP run 全程 `caption_dropout_rate=0`（moss_compat 已强制）、无 shuffle 抖动源、`shuffle_caption`/`weighted_captions` 关闭。
- **noise/sigma 同步**：`uniform_grid_sigmas` 由 batch 大小决定，但 noise 用 `torch.randn_like`——**sigma/noise 依赖 RNG 状态，不只依赖 batch 大小**：任一方案（rank0 生成后 broadcast noise+latents+text+sigma，或两侧同 seed 同 RNG 轨迹重放并校验）均可，但同窗口样本 id/noise/step 的逐位一致是 §5.3 对比判据的前提。实现轮二选一并写入 checkpoint 元数据（`data_sync_mode`），resume 后沿用同一模式。
- **固定 wire 形状**：一个 PP run 冻结一个 `(B, H, W, token_cap)`；`token_cap = H*W + max_text_tokens`（`resolve_wire_capacity`）。动态桶 / 多分辨率混跑在 PP v0 明确具名拒绝（`pp_token_bucket_mismatch`）；dataloader 必须配置为只出该形状桶，若当前 bucket 机制无法保证单桶，加载前具名拒绝（不得运行中撞上才失败）。
- **尾批**：`train_batch_size % microbatches == 0` 且尾批等大；不满足在 run 启动时即拒绝（dataloader 长度可预计算），不得运行中报 `pp_token_bucket_mismatch` 中途炸。

## 4. 步进与 LR 语义（冻结定义）

| 术语 | 定义（MOSS PP=2/microbatches=m 下） |
| --- | --- |
| `batch` / `train_batch_size` | 一个 pipeline step 消费的样本数；必须被 `m` 整除 |
| `microbatch` | `batch / m`；1F1B 调度的调度单元；wire 缓冲按它冻结 |
| `gradient_accumulation_steps` | **PP v0 必须为 1**（具名拒绝其他值）：一个 pipeline step 本身已聚合 m 个微批，梯度缩放由 last-stage loss 内置 1/m 完成；trainer 的 accumulation 语义整体不参与 |
| `global_step` | optimizer 步（pipeline step）口径下的**消费步数**：含 skip 步；与 checkpoint `step`、进度分母 `max_train_steps` 同一口径，但**不**与 LR scheduler 推进同一计数（scheduler 只在成功步前进） |
| `effective_step` | **成功更新步数** = `global_step − 累计 skip 步数`：只计真正更新了权重的步；进 metadata/日志，不进 scheduler |
| `overflow` / skip | `pp.transactions` 单 scale 协议：任一 rank inf/nan grad → 两 rank 同 skip（`agree_overflow`/`verify_skip_unanimity`），scale backoff 由 last stage 广播；skip 时 `global_step` 照常 +1（消费了一步），`effective_step` 不变、权重未更新 |
| LR 推进 | scheduler **只在成功步（未 skip）前进**；skip 步 LR 冻结。scheduler 步数规划除数 = 1（`lr_scheduler_num_processes`），**禁止**按 DDP 的 num_processes 倍乘 |
| 进度 | `progress = global_step / max_train_steps`，与既有 run 目录进度解析兼容；每步记 `loss`（last stage 返回值）、`scale`、`skip`、样本 id 指针 |

## 5. 训练级 resume 契约

### 5.1 覆盖面（trainer 级新增状态）

分 rank 侧沿用 `pp.checkpoint*`（adapters per-rank state_dict、optimizer state、scaler、step、拓扑 digest、run identity、last_skip、scale 增长策略恢复）。trainer 级**新增**到 manifest 或并列文件：

- **LR scheduler state**：`optimizer.state_dict()` **不含** scheduler，scheduler 必须独立序列化（rank 0 权威、每 rank 一份或随 rank 文件均可）并独立校验恢复后的 step/last_lr 一致；不得暗示它会随 optimizer state 自动带上。
- **RNG 状态**：每 rank 的 `torch`(CPU+GPU 若有)/`random`/`numpy` 状态，随该 rank stage 文件存；
- **数据游标**：`data_sync_mode`、epoch、全局样本指针（deterministic 顺序下的索引）；恢复后下一 step 的 batch 与不中断 run 逐位一致；
- **run identity**：库级 checkpoint 已带 run identity（`pp.checkpoint_state.validate_identity`）；trainer 级沿用同一 run root，禁止把一次 resume 记成新 run。

### 5.2 恢复协议（fail-closed）

1. **应用/运行态修改前完成纯验证**：manifest/元数据字段（format/version/拓扑 digest/adapter 归属/step/identity）先校验，再解码 payload；全部**纯验证在任何应用动作（load_state_dict、optimizer/scaler/scheduler 恢复、游标重置）之前完成**，并按 §5.2-2 协商——不允许"读一点、改一点"的交错。
2. **全 rank 协商**：`load_distributed_checkpoint` 的 all-gather `{"step","error"}` 协商——任一 rank 校验失败全体具名 abort（`pp_checkpoint_*`），不出现单 rank 恢复。
3. **旧库级 checkpoint 边界**：无 trainer 级元数据（scheduler/RNG/游标）的库级包**默认拒绝精确 resume**（具名拒绝，不得默默重置 LR/RNG/游标继续跑）；只有用户/调用方**明确请求 weights-only / nonexact 模式**时才允许降级恢复，且 resume metadata 必须标注 `trainer_state: absent → 非精确续训（weights-only）`。加载 trainer 级 checkpoint 时恢复后单步与不中断 run 对齐（§5.3）才可记 `exact`。
4. **数据游标协商**：恢复后的首个 batch 在两 rank 间一致性校验（样本 id 全 gather 比对），不一致具名拒绝。

### 5.3 一致性判据（同窗口比较，冻结容差产生方式）

- 比较：同 seed 同窗口的**样本 id 序列（逐位）、noise 张量（逐位）、`global_step`、每步 loss（相对差）、fp16 scale 轨迹、skip 事件序列、恢复后首步梯度方向余弦**。
- 容差：沿用 v0 文档 §7-E 原则——先按 dtype 与独立非 PP 参考噪声基线确定并记录阈值（进 `tests/moss_pp_test_support.py`），再比较；禁止看 PP 结果后拟合阈值，禁止照抄 Anima 数字，禁止"FP16 逐位相等"判据。
- 双 rank restart/replay 以新进程对整体重启为准（v0 §9-F）。

### 5.4 导出与 run 目录保护

- 最终交付格式是**普通 LoRA safetensors**：训练结束时把两 rank 的 adapter 合并/汇总为标准 kohya 命名（`lora_unet_layers_<N>_proj_<qkvo>`）单文件，由 rank 0 写出；PP 分 rank checkpoint 只是中间恢复格式，不得作为交付格式。
- PP run 的输出进**新的独立 run root**（`create_run_root` 拒绝已存在路径，天然保护既有 run 目录）；PP 分支不得写入/改写任何既有 DDP run 的 state_dir、history、队列文件。

## 6. 未接线能力的具名拒绝（加载前，逐项）

以下配置在 PP 分支进入模型加载**之前**具名拒绝（复用 `pp_*`/`moss_image_*` 稳定 code，不得静默忽略）：

| 能力 | 处置 |
| --- | --- |
| 动态桶 / 多分辨率 | 拒绝：`pp_token_bucket_mismatch`（run 启动时无法保证单桶即拒） |
| preview / validation / sample | 用户**启用**时具名拒绝（PP v0 不接这些路径）或给出真实支持，**禁止**"日志声明跳过"式静默忽略用户启用项；未启用时不得仅凭默认路径字段非空（如 `sample_prompts` 的默认文件存在）误判为已启用 |
| `gradient_accumulation_steps != 1` | 拒绝（§4） |
| NF4 / fp8 / ConvRot / bf16 compute | 拒绝：`pp_quantization_unsupported` / `pp_mixed_precision_unsupported`（fp16-only） |
| `blocks_to_swap > 0` × PP | 拒绝：`pp_stage_contract_violation`（互斥，v0 §3.3） |
| compile / `compile_dynamic_seq` | 拒绝（moss_compat 已拒 `torch_compile`） |
| router / adaptive / 复杂 loss（moss_compat 长清单） | 拒绝（`moss_compat.check_moss_training_contract` 继续生效，PP 分支必须先跑它） |
| 非 plain LoRA network_module | 拒绝（moss_compat） |
| 非 moss_image family 的 PP | gate 现行为不变（`pipeline_parallel_runtime_unavailable` 或族内约束） |
| 非 PP 的 moss_image / 一切 Anima 旧行为 | **不变**，不接受任何配置改写 |

## 7. 热验收合同（后继 verification 任务执行，本轮只冻结设计）

### 7.1 CPU/Gloo tiny（先行，无 GPU）

- 必须走正式入口：`train.py`（或 `AnimaTrainer().train(args)`）→ `run_training_session` → PP 派发分支 → `build_moss_pipeline_runtime` 真实 1F1B step，`gloo` backend、CPU tensor、tiny MOSS fixture（42 层缩 dim，结构同名同序）、临时目录内注入的完整 latent/TE cache 假 sidecar。
- **不得**用 mock `run_pair`/loop/runtime 冒充接线验收；mock 只允许出现在库级单测（v0 §8 的既有文件）。
- 验收点：派发点唯一性（正反向）、§4 步进语义、§5.2 恢复协议（含旧库级 checkpoint 的非 exact 标注）、§6 各具名拒绝、非 PP 回归不触发 PP 分支。

### 7.2 双 CUDA tiny（正式主入口）

- 命令面：`torchrun/accelerate launch --num_processes 2` 经 `launch.py` 包装启动 `train.py`，`CUDA_VISIBLE_DEVICES=GPU-b4b014b2-953b-28e8-b8c1-425c4a401165,GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061`（固定两 UUID，禁止 GTX 1050 或替代卡）。
- 硬限制：执行前 `gpu-gate.sh --count 2 --json` 确认两目标卡 free/usable=yes（low-impact 不构成空闲许可）；单节点 timeout 60s；两 phase 内部总 deadline 45s + 清理；每卡 allocator allocated/reserved ≤2GiB（`PYTORCH_CUDA_ALLOC_CONF` 记录进证据）。
- tiny 注入模型 + 临时 cached 数据；短步数（个位数）；真实权重、用户数据、长训一律不在本轮范围。
- 证据：命令、rc、两卡 UUID、每卡峰值显存、`global_step`/loss/scale/skip 序列、（如跑 resume）恢复前后对齐记录。新字节单次测试，失败报真实 rc + 完整原始证据，禁止无改动 GPU 重跑。

### 7.3 测试入口（冻结文件名）

`tests/test_moss_image_pp_trainer.py`（trainer 接线：派发点、步进语义、数据/noise 同步）、`tests/test_moss_image_pp_trainer_checkpoint.py`（trainer 级 resume：scheduler/RNG/游标/run identity、旧库级包默认拒绝）、`tests/test_moss_image_pp_trainer_hot.py`（双 CUDA 热测，固定 nodeid `test_two_rank_cuda_trainer_resume`）。fixtures 进 `tests/moss_pp_test_support.py`。

## 8. 验收命令（verification 任务 t2/t3 按此执行，本轮不执行）

```bash
# 公共只读校验
git diff --check

# （v3-N 追认）两个 verify 列内但不在实现 in-scope 的测试文件接受最小必要更新：
# tests/test_pp_import_boundary.py（__all__ 计数 96→111，因 pp 包新增 trainer 导出）
# tests/test_training_loop_runtime.py（_FakeSaver 补 loop 已用 kwarg；HEAD archive
#   独立复跑 rc=1 证实既有失败，见 §9 v3-O）

# CPU 定向组（venv=/mnt/SD_Dataset/训练器相关/krea2-webui/.venv/bin/python，CUDA_VISIBLE_DEVICES='' PYTHONPATH=.）
timeout 300 <venv> -m pytest -q -rs tests/test_moss_image_pp_trainer.py tests/test_moss_image_pp_trainer_checkpoint.py
timeout 300 <venv> -m pytest -q -rs tests/test_moss_image_pp_wiring.py tests/test_anima_pp_wiring.py tests/test_krea2_pipeline_parallel.py tests/test_multigpu_admission.py tests/test_pp_import_boundary.py
timeout 300 <venv> -m pytest -q -rs tests/test_training_bootstrap.py tests/test_training_loop_runtime.py tests/test_training_checkpointing.py

# 热测收集（timeout 60）
timeout 60 <venv> -m pytest --collect-only -q tests/test_moss_image_pp_trainer_hot.py

# 双 CUDA tiny 热测（仅 t3；gpu-gate 先行确认两目标卡 free/usable=yes）
/mnt/SD_Dataset/训练器相关/dsh-trainer-preset/gpu-gate.sh --count 2 --json
env CUDA_VISIBLE_DEVICES='GPU-b4b014b2-953b-28e8-b8c1-425c4a401165,GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061' \
  PYTHONPATH=. PYTHONUNBUFFERED=1 timeout 60 <venv> -u -m pytest -q -rs -s \
  tests/test_moss_image_pp_trainer_hot.py::test_two_rank_cuda_trainer_resume
```

## 9. 合同修订记录

- v1（本文档初版，2026-11-14，基线 `6610f710`）：基于实时源码复核冻结——唯一派发点定于 `_run_training_session`（§2.1）、四个 DDP 假设/guard 不降级（§0.3/§1）、步进与 LR 语义表（§4）、trainer 级 resume 四态覆盖与非 exact 标注（§5）、未接线能力具名拒绝清单（§6）、正式入口 tiny 热测合同与 GPU 边界（§7）。
- v2（t6 修订，2026-11-14，captain 纠偏；仅修本文，不洗掉 v1 原判）：
  - A（对 §0.3/§2.1/§7.1）：builder 名纠偏——实际实现是 `pp.moss_runtime.build_moss_pipeline_runtime` / `MossPipelineRuntime`（原生 4D wire），v1 误写 `build_anima_pipeline_runtime`/`AnimaPipelineRuntime` 及不存在的"族 hook 注入"表述。
  - B（对 §0.3/§2.1/§2.2）：guard 语义澄清——v1"guard callsite 全部保持"会使 `cli_entry:100` 对合法 MOSS PP 配置永远拒绝、接线不可能成立。改为：guard 函数与 DDP 内部保护语义保留；`cli_entry` 对满足 MOSS trainer admission 的配置改走 PP 预检分支而非拒绝；唯一执行派发仍在 `train_session`；其他层允许必要的 backend/预检条件分支，但不得出现第二套执行器。
  - C（对 §4）：`global_step` = 消费步数（含 skip），`effective_step` = 成功更新步数，scheduler 只在成功步前进；删除 v1"global_step 与 LR scheduler 同一口径"的错误捆绑与 LR 推进的"二选一"矛盾。
  - D（对 §5.1）：`optimizer.state_dict()` 不含 scheduler——scheduler 必须独立序列化并独立校验，v1 不得暗示自动包含。
  - E（对 §5.2-1）："读 payload 前完成逐字段校验"物理上不可能；改为 manifest/元数据校验先行解码、全部纯验证在任何应用/运行态修改前完成并协商。
  - F（对 §5.2-3）：旧库级包（无 trainer 元数据）**默认拒绝**精确 resume；仅明确请求 weights-only/nonexact 模式才降级；禁止默默重置 LR/RNG/游标。
  - G（对 §2.2）：澄清"不复制 checkpoint"非禁止合同内最小兼容扩展。
  - H（对 §7.3/§8）：测试入口与验收命令按实际 t2/t3 合同修正——三个新增测试 `tests/test_moss_image_pp_trainer.py` / `test_moss_image_pp_trainer_checkpoint.py` / `test_moss_image_pp_trainer_hot.py`（固定 hot nodeid `::test_two_rank_cuda_trainer_resume`）、CPU 三组 timeout 300、gpu-gate 先行、双 CUDA 命令原文入档；替换 v1 的旧 wiring60 单命令。
  - I（对 §3）：数据同步语义纠偏——PP 两 stage 消费**同一** logical batch/noise（broadcast 或同序校验），不是 DDP scatter 不同样本；sigma/noise 依赖 RNG 状态，不只依赖 batch 大小。
  - J（对 §6）：preview/validation 用户启用时必须拒绝或真实支持，禁止"日志声明跳过"；未启用时不得仅凭默认路径字段非空误判。
- v3（t7 修订，2026-11-15，实现者 t2 措辞差核验与范围外追认；不洗 v1/v2）：
  - K（对 §5.1 run identity 实现事实）：`run_id` 是 run material 的 **sha256 全 64-hex content digest**（`pp/trainer_session.py:136-153`，identity validator 只接受 sha256 content digest，无截断）；run material = `family|seed|output_name` 三字段，**不含** `output_dir`（identity 跨输出目录存活）也**不含** `max_train_steps`。§5.1 的"run identity"据此解释。
  - L（对 §5 恢复文件事实）：trainer 级 bundle 的 manifest 文件名实际为 `pp_checkpoint.json`（`pp/trainer_session.py:252`、`pp/trainer_resume.py:62`）。
  - M（对 §5.2 保存语义补充）：同一 step 的周期保存与结束保存**幂等跳过**——bundle root 已存在（`pp_checkpoint.json` marker）即跳过并日志声明（`pp/trainer_session.py:252-258`）；新建 root 仍拒绝覆盖任何已有路径。
  - N（对 §7.3/§8 范围外追认）：t2 对两个 verify 列内但不在实现 in-scope 的测试文件做了最小必要更新，captain 已追认——①`tests/test_pp_import_boundary.py`：`__all__` 计数 96→111（因 `library/training/pp/__init__.py` 新增 trainer 模块导出）；②`tests/test_training_loop_runtime.py`：`_FakeSaver` 补 `loop.run_training_loop` 已使用的 kwarg。**②的"HEAD 既有失败"主张已独立证实**（见 O）。
  - O（archive 独立复跑，2026-11-15）：`git archive HEAD`（`6610f710`）解包到临时目录 `/tmp/t7-archive-head`，工作树未动（无 stash/checkout），在临时目录直跑 `env CUDA_VISIBLE_DEVICES='' PYTHONPATH=/tmp/t7-archive-head timeout 300 <venv> -m pytest -q tests/test_training_loop_runtime.py`，**真实 rc=1**：`1 failed, 5 passed`，失败用例 `test_final_step_runs_epoch_end_artifacts_before_loop_exit`（`library/loop.py:323` `AttributeError: 'types.SimpleNamespace' object has no attribute 'batches_per_epoch'`）。结论：t2 对该文件的 `_FakeSaver` 补 kwarg 是**对 HEAD 既有失败的修复**，不是新需求；该测试的其余 5 项在 HEAD 已通过、工作树改动未使其回归（t3 组3 79 passed 含此文件）。
