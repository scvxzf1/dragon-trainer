# MOSS PP v0 独立验证报告（t3，2026-11-12）

状态：验证报告（verification round 1，只读验证；发现项交 review 裁决，不含实现修改）
执行根：`/mnt/SD_Dataset/训练器相关/krea2-pp-moss`，分支 `codex/moss-pp-v0-20261112`（HEAD `6610f710` + 未提交工作树，23 个改动/新增文件）
解释器：`/mnt/SD_Dataset/训练器相关/krea2-webui/.venv/bin/python`（torch `2.10.0+cu129`，`torch.__file__` = `/mnt/SD_Dataset/训练器相关/krea2-webui/.venv/lib/python3.13/site-packages/torch/__init__.py`，实测打印核验）
本任务边界：只写本文档与 `docs/findings/README.md` 索引；未运行 stash/checkout、未改用户数据、未改任何实现；CUDA tiny 未执行（分配卡被占用，见 §4）。

---

## 1. 执行环境核对（实测）

- 工作树/分支：`git log --oneline -1` → `6610f710`；`git status --short` 23 条（全部为本任务族改动，无用户数据目录）。
- 解释器来源：`sys.executable` = 共享 venv；torch 版本与安装路径实测打印（见上）。
- `torch.distributed.pipelining` API 可导入（t1 已验，本轮未复跑——基线口径不变）。

## 2. CPU/Gloo 回归（全部命令与退出码）

### 2.1 MOSS 新用例 + 原生 + fail-closed 组

```
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=. timeout 300 <venv>/bin/python \
  -m pytest -q -rs tests/test_moss_image_pp_runtime.py tests/test_moss_image_pp_checkpoint.py \
  tests/test_moss_image_pp_wiring.py tests/test_moss_image_family.py \
  tests/test_moss_image_fp16_amp.py tests/test_model_family_fail_closed.py -m 'not hardware'
```
结果：**exit 0，51 passed, 3 skipped, 4 deselected**（104.37s）。逐 skip 理由（-rs）：
- `tests/test_moss_image_family.py:49/:74/:87` ×3：`Set MOSS_OFFICIAL_SOURCE for official differential checks`（官方差分对照环境门）。
- 4 deselected = `tests/test_moss_image_fp16_amp.py` 的 hardware 标记用例（`-m 'not hardware'` 排除）。
- 新 MOSS PP 用例**无一 skip**，不借历史环境门。

### 2.2 Anima 既有 PP 回归组

```
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=. timeout 300 <venv>/bin/python -m pytest -q -rs \
  tests/test_anima_pp_wiring.py tests/test_training_pp_runtime.py tests/test_training_pp_checkpoint.py \
  tests/test_anima_pp_equivalence.py tests/test_anima_pp_loss_scaling.py \
  tests/test_pp_import_boundary.py tests/test_pipeline_parallel_multimodel.py
```
结果：**exit 0，172 passed**（155.78s），无 skip、无失败。Anima PP 接口无回归。

### 2.3 Web preflight 2 例失败——基线隔离证据（只读，未 stash）

```
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=. timeout 300 <venv>/bin/python -m pytest -q -rs \
  tests/test_moss_image_web_preflight.py tests/test_moss_image_web_group.py
```
结果：exit 0（pytest 进程码），**2 failed, 3 passed**（6.51s）。失败为
`test_moss_preflight_accepts_direct_components_and_adjacent_tokenizer` 与 `test_moss_preflight_accepts_model_root_for_all_components`，均断言 `result["ok"] is True` 失败（traceback 已录，`tests/test_moss_image_web_preflight.py:82` 与 `:106`）。
隔离证据（不改文件、不 stash，用同一 fixture 的等价复现脚本直接调 `config_service.preflight_training_config`，打印全部非 ok 项）：

```
ok: False
warning dataset_config 数据集配置尚未生成，无法确认数据集任务；训练前将重新检查
error datasets 数据集图片目录里没有图片: image_dataset
```

即唯一 error 是 **datasets 检查**：MOSS fixture 的 TOML（`tests/test_moss_image_web_preflight.py:37-58`）**不声明任何 dataset**，preflight 回退 `base.toml` 的 `source_image_dir = "image_dataset"`（`web/services/config/preflight.py:175` 注释明示该继承），而本工作树没有未跟踪的 `image_dataset/` 用户数据目录；四个模型组件检查（`pretrained_model_name_or_path`/`qwen3`/`vae`/`moss_image_tokenizer`）全部 `ok`。结论：与本 PP 改动无关的环境性失败（缺用户数据），非放行豁免而是逐项归因证据；该两条用例在含 `image_dataset/` 的原工作区应通过。

## 3. 核心缺陷独立核对（C1–C5，全部成立）

以下逐条在当前代码复核，观察/推断/实测分层标注。**这五条是 t2"完成声明"与合同之间的缺口，须 repair 后复审；本轮 CPU 测试通过不能抵消它们。**

### C1 per-sample 空样本穿过 preflight（实测 + 源码）

- 源码事实：`library/training/pp/moss_runtime.py:327` 与 `library/models/moss_image/pipeline_parallel.py:573` 的 mask 校验均为 `int(text_mask.sum().item()) < batch`（总数），非逐样本；逐样本 `length<1` 只在 stage-0 forward 内 raise（`pipeline_parallel.py:435-441`），发生在 schedule 通信中。
- 实测（captain 探针，本机复认）：`_validate_text_wire(torch.zeros(2,6,32), [[1,1,1,1,0,0],[0]*6], 2, 6)` **未拒绝**，`per_sample_lengths=[4,0]`，exit 0 表示"观察到未接受拒绝"。
- 后果（源码路径推断，未跑双 rank 死锁复现）：`[4,0]` 穿过 `_negotiate_step_precheck` 后，rank0 在 stage-0 forward 中段局部 raise，rank1 已进 schedule 等 recv → 非对称失败。既有 `test_non_binary_or_empty_mask_is_refused` 只构造总数 1 < B=2，掩盖该洞。
- 必修：preflight 逐样本长度校验（`(mask.sum(dim=1) >= 1).all()`）+ 双 rank 负向用例。

### C2 build 期 empty-owned 仍是本地 raise，未协商（源码事实）

`library/training/pp/moss_runtime.py:493-503`：空 owned adapter 现为具名 `PipelineParallelError`，但仍在 `build_moss_pipeline_runtime` 内、`assert_compute_policy_agreement`（:517）与 load 协商 collective 之前本地 raise——只是把 `AdamW([])` 的 ValueError 改名，对端若已进入 `assert_compute_policy_agreement` 的 `all_gather_object` 或 load 协商仍会挂到超时。t6-A 残余风险未关闭（同构校验≠协商）。必修：构造期错误纳入与 step/checkpoint 同款的对称协商通道。

### C3 apply 顺序违反"读前校验"（源码事实）

`library/training/pp/checkpoint.py:446-455`：`apply_stage_checkpoint` 先 `module.load_state_dict`（adapter 权重已改写，:447）再 `_validate_optimizer_shape`（:448）；scaler/transaction 一系列校验（:456-497）也都在 apply 内。`load_distributed_checkpoint:556-557` 的协商 gather 在 apply **之前**，但 gather payload 只含 step/topology/scale/tracker/last_skip——optimizer 参数组形状与 fp16 状态适配性不在读前校验面内；一 rank 形状不适配会"先改 adapter 再局部失败"，对端随后卡 `_restore_fp16_policy`（:558）的新 collective。必修：optimizer 形状/scaler 存在性/transaction 适配性全部前移进 `read_stage_checkpoint`（纯读校验），apply 阶段只做无副作用的写入。

### C4 CPU growth 测试：增长边界未越过 + tracker 断言是窗口巧合（实测）

- 实测（独立探针，`PpFp16Transaction` 单进程，interval=4，逐 step 打印）：tracker 逐步 1→2→3→**4（scale 65536→131072，tracker 归 0）**→1。增长只在第 4 步发生。
- 测试轨迹（当前 `tests/test_moss_image_pp_checkpoint.py`）：save 侧 = save 前 1 步（tracker1）+ save + continuation 1 步（tracker2，记录 2）；restore 侧 = load（恢复 tracker=**1**，即 save 时刻值）+ replay 1 步（tracker2）。两条分支各自最多 2 步，**增长边界（第 4 步）在两条轨迹上都未发生**；`growth_interval=4` 注释（"tracker reaches 4 inside the 4 total steps"）不成立——4 是 save/restore 两条**不可相加**的分支之和。
- 实测 payload（pytest tmp 产物，rank1）：saved `scale=65536, tracker=2, interval=4, cont=[40547.14, 41844.03]`；restored `scale=65536, tracker=2, interval=4, replay=[40547.14, 41844.03]`。`restored.tracker==saved.tracker`（2==2）成立纯属窗口巧合（恢复值=save 时刻 1，+1 步 replay=2；save 记录=continuation 后 2），**不证明 tracker 恢复语义正确**；scale 全程 65536 未变，策略恢复的实际数值效果（interval 777→4 采纳）只体现在 interval 值，未在增长边界上验证。
- 保留项：rank1 逐位相等的 replay（fp32 CPU）是真实等价证据；但只有 1 步，且断言 `restored["growth_tracker"] == saved["growth_tracker"]` 的语义需重写（对齐同一窗口或断言恢复值而非事后值）。
- 必修：save 侧在 save 后继续跑 ≥3 步使 tracker 越过 4，restore 侧重放同窗口，比较对应窗口的未缩放 loss 与每步 tracker/scale 序列。

### C5 CUDA 测试参照窗口/状态时点错误（源码事实，未运行）

`tests/test_moss_image_pp_runtime.py:540-734`：save 侧 = 2 clean 步（scale 65536，`clean_losses` 记此二步）+ 1 步单侧 overflow（skip、scale 减半、tracker 归 0）+ save(step=3)；restore 侧 fresh pair = load 后 2 clean 步，此时 scale=32768（减半后）、tracker 0→2。则：
- `:729` `replay_losses ≈ clean_losses rel=1e-2` 比较的是**不同 scale 下的 raw scaled loss**（65536 vs 32768，比值 2×）——不是容差问题，是参照窗口错误；
- `:722-725` `restored["growth_tracker"] == saved["growth_tracker"]` 比较"恢复后走 2 步的 tracker 2"对"overflow 后的 tracker 0"——必不等。
- fresh pair（`run_pair` 两次）、autocast（:567/:649）、mask/times `_cuda_batch` 内 `.to(rank_device)`（:529-537）、恢复策略取自 checkpoint（restore 构造 777，断言恢复为 4）这些 t6 项确已修复。
- 必修：save 侧在 save 之后继续原进程跑 2 clean 步作为 step4/5 参考（记录未缩放 loss 与每步 tracker/scale），fresh restore 走同窗口并比较对应量；不得以放大容差代偿。

### CUDA 未执行（环境阻塞，如实报告）

即时占用闸门未通过：分配的两张 V100 各被用户进程占用——
`nvidia-smi --query-compute-apps` 实测：`GPU-b4b014b2…(VLLM::Worker_TP0, pid 206623, 14390 MiB)`、`GPU-379b8c19…(VLLM::Worker_TP1, pid 206624, 14390 MiB)`（另有 kwin_wayland/lact 桌面进程；GTX 1050 `GPU-410a51d5…` 亦忙，且本来禁用）。按合同：不换卡、不抢占、不终止用户进程，**CUDA tiny 本轮未验证**；`gpu-gate` 工具在本机不可用（command not found），闸门以 nvidia-smi 读数代替并如实记录。待 captain 协调卡空闲后补跑（timeout 60、每卡 ≤2GiB、记录 dtype/scale/skip/峰值与 UUID）。

## 4. 结论分层

- **实测通过**：CPU/Gloo 全量回归（MOSS 51 passed/3 skip + Anima 172 passed，exit 0）；CPU fresh-pair checkpoint 位相等与 1 步 replay 位相等（rank1 payload 实测）；双向格式拒绝、归属冲突对称 abort、单侧 overflow 双 rank skip、step 前对称协商机制存在（`moss_runtime.py:280-362`，部分无效输入仍可穿透，见 C1）。
- **实测确认缺陷**：C1（探针）、C4（tracker/scale 探针 + pytest payload）、web preflight 2 失败（等价 fixture 复现归因 datasets 用户数据缺失）。
- **源码确认缺陷（未运行复现）**：C2、C3、C5。
- **未验证**：双 CUDA tiny 节点（卡忙）；真实权重训练/性能/收敛（合同外，本轮无任何此类声明）。
## 5. t10 round-3 独立复验（2026-11-12 18:0x，attempt af97017f）

**实测通过**：`git diff --check` 退出码 0；MOSS 三文件（含 `-rs`）pytest 直连退出码 **0**，42 passed / 1 deselected / 0 failed（254.58s，`/tmp/t10_moss.log`）；Anima 共享八文件 pytest 退出码 **0**，177 passed / 0 skip / 0 failed（157.96s，`/tmp/t10_anima.log`）。解释器/版本绑定同 §1；工作树 HEAD `6610f710` + 26 文件未提交改动（含新增 `library/training/pp/checkpoint_validation.py`）。

**修复项核验（源码）**：
- C1/F1 **关闭**：`library/training/pp/moss_runtime.py:280-290` precheck 签名含 `target`，timesteps 双 rank 通用校验（:305-316，rank0 也要求）、末 stage target None/形状校验（:371-382），`step()`（:419-421）把 target 传入协商。
- C3/F2 **部分关闭**：`library/training/pp/checkpoint_validation.py` 落地纯校验——scale 正有限（:41-50）、growth tracker、GradScaler 必要 keys/数值范围（:64-110）、AdamW exp_avg/exp_avg_sq 逐参数形状（:150-172）；`validate_checkpoint_content`（:175-181）接入。
- C2（构造期对称协商）、C4（同轨迹 save/continuation/fresh replay 跨增长边界）由 t7/t9 实现并经本轮 42 用例回归覆盖。

**本轮新确认缺陷（源码证据，交 repair）**：
- **G1**：`tests/moss_pp_test_support.py:204` 普通 `run_pair` 仍硬编码 `Path(tmp_path)/"init"`，唯一化 helper `_next_rendezvous`（:190-198）只被 `run_pair_expect_failure`（:229）调用——t9"F3 两个 helper 均改用"与源码不符；t8 观测到的 restore 阶段 `Gloo connectFullMesh Connection refused` 竞争并未修复（t9 三次独立运行恰好未触发）。另 :206 的 `errors` 用 tmp 全局 `glob("*.error")` 收集，期望失败阶段残留的旧 `.error` 文件可能污染后续断言信息（假失败信息/潜在假通过面）。
- **G2**：`library/training/pp/checkpoint_validation.py:34` `_SCALER_REQUIRED_KEYS` 缺 `"_growth_tracker"`（torch 2.10 `GradScaler.load_state_dict` 必读该键），缺失 payload 会在 validate 通过、apply 的 `scaler.load_state_dict` 才失败；且 `validate_checkpoint_content`（:175-181）对 `state.scale` 与 `state.scaler["scale"]` **无一致性检查**，两个不同正 scale 会延迟到 `transaction.restore` 阶段才拒。
- GPU：`gpu-gate.sh --count 2 --json` exit 3（双 V100 verdict=training、usable=no）——本轮不为源头缺陷占 GPU，CUDA tiny 仍未验证，阈值先验纪律不变（不从候选误差反推）。

**结论分层**：CPU 全量回归实测通过；C1 关闭、C3/F2 大部分关闭；G1/G2 为新确认源码缺陷（须 repair 后复审）；CUDA tiny 未验证（卡忙）。旧报告 §2.3 的 "exit 0 (pytest 进程码)" 已由 t8 澄清更正为 wrapper 码口径，本轮全部命令直连记录真实 pytest 退出码。

（§1–§4 为 t3/t8 轮次历史证据，保留不改。）

## 6. t12 人工接管修复独立核验（attempt 5f901f1f，进行中）

- **t11 绑定**：5 个源文件 SHA256 实测与 t11 声明全部一致；HEAD `6610f710` + 28 文件。
- **F4/F5 修复源码确认**：`tests/moss_pp_test_support.py:198-224` run_pair/_spawn_pair 统一 `_next_rendezvous` 唯一路径、错误回执按本次 call_dir 收集、exitcodes 在 terminate 前快照；`library/training/pp/checkpoint_validation.py:34-36/:118-123` `_growth_tracker` 必需键+范围校验、`:187-196` state.scale 与 scaler scale 一致性检查。
- **实测（直连真实退出码）**：`git diff --check`=0；`tests/test_moss_image_pp_harness.py`+`test_moss_image_pp_checkpoint_validation.py` pytest exit 0（14 passed，/tmp/t12_harness.log）；定向三节点（内容变异对称拒/MOSS restart_replay/Anima two_rank roundtrip restart）pytest exit 0（3 passed，/tmp/t12_targeted.log）。
- **CUDA**：`gpu-gate.sh --count 2 --json` exit 3（双 V100 各 10951 MiB 被 vLLM 占用）——闸门未通过，tiny 节点未验证，等待资源窗口；L2 执行纪律：既有 dtype 先验阈值（不据候选调整，node docstring "re-frozen" 旧措辞作废）、last rank 每步完整 n_microbatches 个有限 loss（防 zip 空列表假通过）、overflow 双 rank 一致+同窗口 digest、allocated/reserved 分开记录、只称 FP16 编排/事务/恢复覆盖不外推真权重。
- **新确认缺陷（独立探针实测复现，交 captain 授权修复）**：
  - **F6**：`library/training/pp/checkpoint_validation.py:150-152`（validate_optimizer_payload）对越界保存参数索引 `continue` 跳过校验——实测：tiny AdamW(2 参数) 正常 state 将保存 ID 重编号为 91 且 exp_avg/exp_avg_sq 形状 (3,)，validate 通过、torch `load_state_dict` 通过、`optimizer.step()` 实抛 `RuntimeError: size 3 vs 2`（apply 后延迟失败）。正确做法：按 saved param_groups.params ID 与 live 组 params 逐组 zip 映射校验，不越界跳过。
  - **F7**：`checkpoint_validation.py:155-166` step 校验仅查标量尺寸/粗类型，无有限/非负——实测：step 张量改 NaN，validate 通过，load 后有限梯度一步更新把参数变 NaN（`isfinite=False`）。需补 step 有限且非负合法计数校验。
- **结论**：t12 失败收束于 F6/F7 软件缺陷（非 GPU 资源）；F4/F5 通过证据保留；CUDA tiny 仍未验证（环境）。

## 7. t12 attempt-2（t13 修复后）独立复验（2026-11-12 20:4x）

- **t13 绑定（实测 SHA256）**：`checkpoint_validation.py`=`bbcb5d50…0251`、`tests/test_moss_image_pp_checkpoint_validation.py`=`b07e3d98…45d4`、`tests/test_moss_image_pp_checkpoint.py`=`fdfebcbb…6fbd`（与 t13 声明一致）；`tests/test_moss_image_pp_harness.py`=`6ba1de1c…`（F4 harness 未动）。
- **F6/F7 独立探针复验（实测，修复后代码）**：重编号 ID 91+错形 (3,) state → validate 具名拒绝 `pp_checkpoint_state_mismatch`（不再过 torch load）；step=NaN → 拒绝（finite and non-negative）；负 step → 拒绝；合法原样 state 仍接受。修复按保存 ID 与 live 组 params 逐组映射（`_optimizer_parameter_map`），未知/缺失 ID 读前拒绝，Adam/AdamW 必需字段+empty-state+非 Adam 因子化字段宽容均落地。
- **实测（直连真实退出码）**：`git diff --check`=0；harness+validation 双文件 pytest exit 0（**39 passed**，/tmp/t12b_harness.log）；定向三节点 exit 0（3 passed，/tmp/t12b_targeted.log）。
- **CUDA 节点（闸门通过、真实执行、真实失败）**：`gpu-gate.sh --count 2 --json` exit 0（双 V100 verdict=free、usable=yes）；以授权 UUID 执行唯一授权节点两次独立运行（/tmp/t12_cuda.log、/tmp/t12b_cuda.log，非重试至绿——第二次为绑定当前字节确认）均失败：Stage 0 forward 在 `layers` 首个 `_Block` 的 LoRA `_down` 抛 `RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cpu … cuda:0`，随后 timeout 60（CUDA_EXIT=124）。
- **新确认缺陷**：**F8**——CUDA worker 在 `model.to(device=cuda, fp16)` 之后 `tiny_lora(model)`（tests/test_moss_image_pp_runtime.py:800-801），LoRA `lora_down/up` 参数仍创建在 CPU 且未随 network 迁移（fixture 未调用 `network.to(device)`），fp16 CUDA 路径首步即 device mismatch；节点无法产生任何 L2 数值证据。修复方向：apply 后 `network.to(device)`（或 tiny_lora 按 base 参数设备创建），fp32 master 语义保持（device 迁移不改 dtype）。
- **结论分层**：CPU 侧 C1–C5/F1–F7 全部关闭并实测（探针+回归）；CUDA tiny 因 F8 fixture 缺陷未能产出任何数值/峰值/恢复证据——L2 仍未验证；阈值先验未动。

## 8. t12 attempt-3（t14 修复后）独立复验（2026-11-12 22:3x–22:5x）

- **t14 绑定（实测 SHA256）**：`tests/test_moss_image_pp_runtime.py`=`76eead2f…c4f7d`（F8 修复：`network.to(device=device)` 在 builder/factory 之前，device-only 迁移保持 fp32 master，:714-717）、`library/training/pp/moss_runtime.py`=`243c3314…e09e`（master 可选 device 校验）、`tests/test_moss_image_pp_devices.py`=`31afffa1…87e9`（10 个新 CPU 负向/守护用例）。checkpoint 三文件与 t13 一致（bbcb5d50/b07e3d98/fdfebcbb），按旧绑定证据复用不重跑。
- **实测通过（直连真实退出码）**：`git diff --check`=0；harness+validation pytest exit 0（39 passed，/tmp/t12c_harness.log）；隔离复验 MOSS restart/replay + Anima two_rank roundtrip restart pytest exit 0（2 passed，88.24s，/tmp/t12c_t3.log）。
- **新确认缺陷 F9（high，实测三次复现）**：`tests/test_moss_image_pp_checkpoint.py::test_content_mutation_refused_symmetrically_before_apply` 在 t14 字节上挂起——完整命令 timeout 300 杀（TARGETED_EXIT=124，空日志）、单测 timeout 200 杀（T1_EXIT=124）。`PYTHONFAULTHANDLER=1`+SIGABRT dump（/tmp/t12c_t2.log）显示两 rank 均已在 `checkpoint.py:600` 的 validate 协商 gather 处按预期抛出 `pp_checkpoint_state_mismatch`（损坏 growth_tracker=-1 被拒——拒绝逻辑本身正确），但此后 spawn 子进程长时间存活（分钟级，实测 pgrep 可见 `spawn_main` 两子进程），即 worker finally 的 `dist.destroy_process_group()` 挂起，测试永不返回。同条件下 restart/replay 与 Anima restart 两 spawn-pair 用例正常通过（88s），排除系统负载解释；对照 attempt-2 同命令 138s 通过，挂起与 t14 变更（moss_runtime 243c3314 的 master-device 校验协商路径）时间相关。修复方向：排查 validate 阶段拒绝后两 rank 的 collective/destroy 顺序（确认协商 gather 后无 rank 仍停留在未完成 collective 中再 destroy；或 destroy 加超时+具名中止），并使该用例在拒绝后可靠退出。
- **CUDA 节点（闸门通过、单次执行、无证据产出）**：gate exit 0（双 V100 verdict=free/usable=yes，即时核对无残留测试进程——vLLM 已退出）；以授权 UUID 单次执行 `test_two_rank_cuda_fp16_runtime_and_resume`：CUDA_EXIT=124（timeout 60），**日志为空**（/tmp/t12d_cuda.log，0 行——SIGTERM 直接杀导致 python 缓冲输出丢失，无法区分早期挂起与 60s 内未完成），未产出任何 L2 数值/峰值/恢复证据。按"失败不重复"纪律未重跑。结合 F9 的挂起表现，不排除 t14 变更引入的早期阻塞；需 F9 修复后在新绑定下单次重试。
- **结论分层**：F4–F7 与核心恢复契约独立验证通过；F8 关闭（device 迁移已入 _cuda_runtime，且 devices 新用例覆盖 wrong-master-device 拒绝/device-only 迁移保持 fp32 master）；F9 挂起为新确认缺陷（实测三次）；CUDA L2 仍未验证（单次运行无证据，失败保留）。

## 9. t16 独立复验（attempt c86ed014，2026-11-13 00:4x–01:0x）

- **绑定（实测 SHA256）**：t15 三件——`tests/moss_pp_test_support.py`=`1e1b8c65…f6f4`、`tests/test_moss_image_pp_harness.py`=`f5725fbb…fb005`、`tests/test_moss_image_pp_runtime.py`=`9c375cb5…54306`；t13/t14 库源码未变——`checkpoint_validation.py`=`bbcb5d50…`、`moss_runtime.py`=`243c3314…`、`contract.py`=`1f55b686…`。HEAD `6610f710`，30 文件未提交。
- **CPU 实测（直连真实退出码）**：`git diff --check`=0；t16 合同② harness+devices pytest exit 0（**19 passed**，30.12s，/tmp/t16_harness.log）；合同③ content_mutation + restart/replay + runtime 全文件 + wiring pytest exit 0（**33 passed, 1 deselected**，179.15s，/tmp/t16_combo.log）——**t15 共享 deadline 下 content_mutation 用例正常完成**（t12 attempt3 的挂起不再复现；F9 底层原因仍按未知记录，不宣称"已修复"）。
- **CUDA 节点（闸门通过、单次执行、真实失败）**：即时核对无残留测试进程（vLLM 已退出，仅桌面进程）；gate exit 0（双 V100 verdict=free/usable=yes）；以授权 UUID + PYTHONUNBUFFERED=1 + 外 timeout60 单次执行 `test_two_rank_cuda_fp16_runtime_and_resume`：**CUDA_EXIT=1（真实 pytest 码），31.95s 内两 phase 全部跑完**（未超时），失败于 `tests/test_moss_image_pp_runtime.py:986` 的 `NameError: name 'MOSS_CUDA_REPLAY_RTOL' is not defined`——该常量在 tests/library 全仓无定义（rg 证实仅 :942 docstring 与 :986 两处出现）。
- **该次运行的真实数值证据（在 NameError 之前的断言全部通过，来自 /tmp/t16_cuda.log traceback 路径）**：fresh 双进程 save（2 clean autocast fp16 步 + 单侧 overflow skip + checkpoint + 同窗口 continuation steps4-5）与 fresh 双进程 restore（构造 interval≠checkpoint，重放 steps4-5）完成；`dtype=fp32 master`、checkpoint 恢复的 `growth_interval=4`（构造值不同）、`scale_after_replay==scale_after_continuation`、`tracker_after_replay==tracker_after_continuation`、`restored_digest==saved_digest`、`after_replay_digest==after_continuation_digest`、每 rank `peak_allocated_mib<=2048` 与 `peak_reserved_mib<=2048`、每步 `scale_used` 一致、两 rank `skipped=True` 全部通过；仅最后的逐步未缩放 loss 数值比较在 NameError 处中止。**L2 未通过**（断言不完整即不算通过），证据已记录，未据此调整任何阈值。
- **新确认缺陷 F10（high）**：`tests/test_moss_image_pp_runtime.py:986` 引用未定义常量 `MOSS_CUDA_REPLAY_RTOL`（全仓无定义），节点最后一次数值断言以 NameError 崩溃；且 :985 的 `zip(cont["losses"], rep["losses"])` 未先断言两侧长度均为 n_microbatches，存在静默截短风险。requiredFix：在模块级定义预登记 dtype 先验 RTOL（不据候选误差调整）；zip 前断言 `len(cont["losses"])==len(rep["losses"])==2`；修复后新绑定单次重跑该节点。
- **结论分层**：CPU 全部合同命令通过（19+33，直连 exit 0）；CUDA 节点单次执行到达最终断言但因 F10 崩溃——数值/恢复/峰值/overflow 证据已产出且为真，**L2 不判通过**；无真权重/训练/性能结论。

## 10. t18 独立复验与 L2 完成（attempt 5b4f037b，2026-11-13 01:3x）

- **t17 绑定（实测 SHA256）**：`tests/test_moss_image_pp_runtime.py`=`3aafdbbc…8633`（F10 修复：模块级 `MOSS_CUDA_REPLAY_RTOL = 1e-2`，:714 注释"pre-registered fp16 prior; never candidate-fitted"；zip 双侧完整长度检查）、新增 `tests/test_moss_image_pp_cuda_assertions.py`=`0d18daa5…23f8`；未变源码复核——`moss_pp_test_support.py`=`1e1b8c65…`、`test_moss_image_pp_harness.py`=`f5725fbb…`、`moss_runtime.py`=`243c3314…`、`checkpoint_validation.py`=`bbcb5d50…`。`git diff --check`=0。
- **CPU 实测（直连真实退出码）**：cuda_assertions pytest exit 0（**7 passed**，6.50s，/tmp/t18_assert.log——mock 双 phase 直接执行实际 hardware 节点父断言，正常/容差内通过、空/截短/额外 loss 与超 1e-2 容差拒绝）；runtime collect-only exit 0（17/18 collected，1 deselected——仅为收集数，不计执行）。
- **GPU 单次执行（t17 字节首次，gate exit 0：双 V100 verdict=free/usable=yes，nvidia-smi 无测试残留进程）**：授权 UUID + PYTHONUNBUFFERED=1 + 外 timeout60 执行 `test_two_rank_cuda_fp16_runtime_and_resume` → **CUDA_EXIT=0，1 passed，31.25s**（/tmp/t18_cuda.log）。
- **本次运行真实 L2 证据（本次 tmpdir `/tmp/pytest-of-scv/pytest-102/test_two_rank_cuda_fp16_runtim0/cuda_node_evidence.json`，非 /tmp 全局 glob）**：
  - **逐步 loss 完整性**：last rank（rank1）continuation steps4/5 与 replay steps4/5 每步各 **2 个有限 loss**（[19861.7109375, 20359.970703125] / [19480.47265625, 19822.568359375]）；rank0 losses 为空（first stage，正常）；无空列表/截短被当作通过（长度断言在测试内）。
  - **同窗口恢复一致性**：每步 `scale_used=32768` 两侧一致；逐步未缩放 loss 两侧**逐位相同**；`scale_after_continuation==scale_after_replay==32768`、`tracker_after_continuation==tracker_after_replay==2`、恢复的 `growth_interval=4`（来自 checkpoint，构造值不同）；digest 相等（restored==saved、after_replay==after_continuation，测试内断言通过）。
  - **overflow 一致**：两 rank `skipped=True`（单侧 overflow 步双方同 skip，无半模型推进）。
  - **FP32 master/FP16 compute**：`dtype='torch.float32'`（rank1 记录）。
  - **显存**：allocated 23.23/21.39 MiB、reserved 28/26 MiB 分列，各 ≤2GiB（ allocator 峰值，非整卡总峰）。
- **结论分层（供 t4/t5）**：L0/L1/L2 全链在最新字节上完成——CPU（19+33+7，含 mock 父断言）与真实双 V100 tiny 节点（单次、完整断言、exit 0）；L2 覆盖边界：**FP16 编排/事务/同窗口恢复/overflow 对称/显存峰值**，不外推真权重训练、独立 GPU 梯度对照或性能/收敛；先验 RTOL=1e-2 未据候选调整；t12/t16 各 attempt 失败与更正历史保留。
