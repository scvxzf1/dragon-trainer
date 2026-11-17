# MOSS 主训练 PP 接线独立验证与双卡恢复热路径（t3，2026-11-14/15）

状态：验证报告（round-1 至 round-3 attempt6 六轮真实失败逐轮收敛 F1–F8，全部为测试侧缺陷、实现侧行为持续正确；round-3 attempt7（t11 第 9 轮，hot SHA `07da5ddf…`）**rc=0 完整通过**——MOSS 主训练 PP 接线在真实生产入口 tiny 双 V100 上取得全套热验证证据）
适用版本：分支 `codex/moss-pp-v0-20261112`，工作树基线 HEAD `6610f710`（t2 实现为工作树未提交改动）
验收依据：[moss_image_pp_trainer_wiring.md（已归档契约）](../../_archive/docs/proposal/moss_image_pp_trainer_wiring.md)（**v2**，SHA256 `50c3e4d0f6737f4cc5d23fcedf0069de7cd0d8421677dc86d0bf651d391fa4c9`；v1 的 Anima builder/CLI guard 不变/旧 wiring60 口径一律不作验收依据）
验证者：trainer-verifier（独立于 t2 实现者）

---

## 1. 结论摘要

1. **CPU 侧全部通过**：三组定向命令真实 exit 0（19 / 127 / 79 passed），`git diff --check` 干净；接线事实抽查（admission 预检、唯一分叉、backend/master-weight/optimizer 校验）与需求 v2 一致。
2. **GPU hot 首跑失败（rc=1）**：失败点是**测试/支持模块的 deadline 语义接线 bug**，pair 从未 spawn，与 GPU 环境、显存或 vLLM 驻留无关。按"无改动不重跑"纪律本轮不再执行第二次 GPU。
3. **验收判定**：CPU 验收项通过；`test_two_rank_cuda_trainer_resume` 未取得通过证据，GPU 验收项**未通过**，下游不得宣称双卡热路径已验证。

## 2. 代码身份绑定（本次验证所测字节）

| 文件 | SHA256（前16） |
| --- | --- |
| library/training/pp/trainer_admission.py | `df4d36ef423c4326` |
| library/training/pp/trainer_inputs.py | `b1dddfd8f6488b44` |
| library/training/pp/trainer_resume.py | `0a556adb8bfe8f97` |
| library/training/pp/trainer_session.py | `003772022b34c363` |
| library/training/pp/moss_runtime.py | `243c33140a5b1b62` |
| library/training/train_session.py | `5039caa281db4080` |
| library/training/cli_entry.py | `540b2453bc1c7706` |
| tests/test_pp_import_boundary.py（范围外追认） | `a38c6206a0be60ce` |
| tests/test_training_loop_runtime.py（范围外追认） | `8a458c3c0751b34a` |
| tests/test_moss_image_pp_trainer_hot.py | `c7b57d4a57674ea5` |
| docs/proposal/moss_image_pp_trainer_wiring.md（v2） | `50c3e4d0f6737f4c` |

## 3. CPU 验收证据（真实 exit code）

| 命令 | rc | 结果 |
| --- | --- | --- |
| `git diff --check` | 0 | 干净 |
| `pytest -q -rs tests/test_moss_image_pp_trainer.py tests/test_moss_image_pp_trainer_checkpoint.py`（`CUDA_VISIBLE_DEVICES='' PYTHONPATH=. timeout 300`，venv=krea2-webui/.venv） | 0 | **19 passed**，78.61s |
| `pytest -q -rs tests/test_moss_image_pp_wiring.py tests/test_anima_pp_wiring.py tests/test_krea2_pipeline_parallel.py tests/test_multigpu_admission.py tests/test_pp_import_boundary.py`（同环境） | 0 | **127 passed**，29.61s |
| `pytest -q -rs tests/test_training_bootstrap.py tests/test_training_loop_runtime.py tests/test_training_checkpointing.py`（同环境） | 0 | **79 passed**，6.25s |

接线事实抽查（源码实读，非 mock 判断）：

- `cli_entry.py:84-100`：PP 预检分支调用 `require_pipeline_runtime` + `require_moss_trainer_admission`（来自新模块 `pp/trainer_admission.py`）；`require_trainer_topology`/`require_accelerate_checkpoint` guard 函数保留，DDP 内部保护未删——与 v2 §0.3/§2.1 一致。
- `train_session.py:508-520`：唯一执行分叉，`pipeline_parallel_enabled(args)` 时调用 `pp.trainer_session.run_pp_trainer_session`，共享前置 setup 后薄派发——无第二套执行器。
- `pp/moss_runtime.py`：`MossPipelineRuntime`（:160）/`build_moss_pipeline_runtime`（:510）原生 4D；`validate_distributed_backend`/`validate_master_weight_dtype`（:52-53,:575）与 stage-local optimizer 归属（:177,:225-226,:443-458）在位。

## 4. GPU gate 与资源窗口（如实记录）

- 2026-11-15 01:1x（第一次）：`gpu-gate.sh --count 2 --json` rc=3，两 V100（index1/2）mem_used 11792/16384 MiB、verdict=training、usable=no。原始 JSON 存 `moss_pp_trainer_wiring_cuda_gate_20261115t0110.json`。任务保持 in_progress 报告资源窗口，未执行 GPU。
- 2026-11-15 02:1x（用户明确授权 GPU 使用：单次最大 6000s、多次无上限、不允许 1050）：重跑 gate rc=0 但两卡 `usable:"no"` 不变——驻留进程实查（`nvidia-smi --query-compute-apps`）：两目标 UUID 各有用户自己的 `VLLM::Worker_TP0/TP1`（pid 17816/17817）各 11778 MiB、util 0%；每卡余量 4592 MiB ≥ 测试需求 2048 MiB。用户授权视为资源所有者对驻留状态的知情放行（记录于此），1050 未使用。原始 JSON 存 `moss_pp_trainer_wiring_cuda_gate_pre-hot.json`。
- 无本团队旧测试进程残留（`pgrep` 实查）。

## 5. GPU hot 首跑（单次，真实失败）

命令（与 t3 合同逐字一致，另加测试自身要求的 `MOSS_PP_HOT_ENABLE=1`）：

```bash
env MOSS_PP_HOT_ENABLE=1 CUDA_VISIBLE_DEVICES='GPU-b4b014b2-953b-28e8-b8c1-425c4a401165,GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061' \
  PYTHONPATH=. PYTHONUNBUFFERED=1 timeout 60 <venv> -u -m pytest -q -rs -s \
  tests/test_moss_image_pp_trainer_hot.py::test_two_rank_cuda_trainer_resume
```

- **rc = 1**，`1 failed`，5.94s。完整原始输出存 `moss_pp_trainer_wiring_cuda_hot_run1.log`。
- 失败点：`run_pair(tmp_path, _cuda_worker, "straight", str(tmp_path), deadline=60.0)` → `tests/moss_pp_test_support.py:240: TimeoutError("PP pair deadline expired before spawning children")`。**两个 rank 子进程从未启动**，无任何 GPU 计算、显存分配或检查点写入发生。

### Finding F1（blocker，测试接线）

- **file/line**：`tests/test_moss_image_pp_trainer_hot.py:78-79`（调用侧）对 `tests/moss_pp_test_support.py:239-242`（`_spawn_pair`）。
- **问题**：`_spawn_pair` 的 `deadline` 参数是**绝对** `monotonic()` 端点（`end = deadline`），仓库既有约定同此：`tests/test_moss_image_pp_runtime.py:954` `deadline = monotonic() + 45`；`tests/test_moss_image_pp_harness.py:123-131` 明确断言"过期 deadline 不 spawn"。新 hot 测试把**相对**秒数 `60.0` 当 deadline 传入，`monotonic()`（系统启动以来秒数）≫ 60，`end <= monotonic()` 立即成立 → 支持模块在 spawn 前抛 `TimeoutError`。两个 phase（straight/resume）各自传独立 60.0 也与测试 docstring"60s outer budget"的单窗口共享预算不符（对照 runtime 测试两 phase 共享一个端点）。
- **影响**：hot node 在任何机器上必然立即失败；t2 的"hot collect 1"只证明收集不证明执行。
- **最小修法**（二选一，改测试侧为最小）：`tests/test_moss_image_pp_trainer_hot.py:77-79` 改为共享绝对端点，例如 `outer = time.monotonic() + 60.0` 后两处 `run_pair(..., deadline=outer)`（对齐 `test_moss_image_pp_runtime.py:954-956` 模式）；不动 `moss_pp_test_support.py` 语义。修复属 tests 范围，超出本验证任务 in-scope，本轮不改。
- **非根因排除**：不是 GPU 忙/OOM/vLLM 驻留（未 spawn、5.94s 内失败、无 CUDA 初始化子进程）；不是环境缺失（两卡可用、`_two_cuda_devices()` 为真、skipif 未触发）。

### 重跑纪律

本轮不重跑。修复 F1（tests 侧最小改动）后，按同一预算（用户已授权单次 6000s 上限；hot 测试自身 60s 外限/45s 相预算/每卡 ≤2GiB 维持）重跑单次 `test_two_rank_cuda_trainer_resume` 取证。阈值不据候选调整。

## 6. verification-round-2（t9，2026-11-15 02:2x；t8 修复 F1 后）

### 6.1 CPU 与收集（全部真实 exit 0）

| 命令 | rc | 结果 |
| --- | --- | --- |
| `git diff --check` | 0 | 干净 |
| 组1 trainer + trainer_checkpoint | 0 | 19 passed，76.72s |
| 组2 wiring/anima/krea2/multigpu/import_boundary | 0 | 127 passed，28.94s |
| 组3 bootstrap/loop_runtime/checkpointing | 0 | 79 passed，6.18s |
| `pytest --collect-only -q tests/test_moss_image_pp_trainer_hot.py`（timeout 60） | 0 | 1 test collected |

t8 修复字节绑定：`tests/test_moss_image_pp_trainer_hot.py` SHA256 `dac7db192aadfc8ff3faa7deff47ae98a0742cddd819165ece1176df9c30cd09`（与 t8 报告一致）；`pp/trainer_session.py` `003772022b34c363…`、`pp/trainer_resume.py` `0a556adb8bfe8f97…` 未变。

### 6.2 GPU gate

`gpu-gate.sh --count 2 --json`（原始存 `moss_pp_trainer_wiring_cuda_gate_t9.json`）：两 V100（index1/2）仍各驻留用户 vLLM 11792 MiB、usable=no（余量 4592 MiB ≥ 2048 需求）；沿用用户 2026-11-15 02:08 的明确 GPU 授权（单次 ≤6000s、多次无上限、禁 1050）执行。无旧测试进程残留（pgrep 实查）。

### 6.3 hot 单次运行（rc=1，真实失败——新缺陷 F2）

命令与 round-1 相同（`MOSS_PP_HOT_ENABLE=1` + 双 UUID + timeout 60）。**rc=1，1 failed，14.67s**。原始输出存 `moss_pp_trainer_wiring_cuda_hot_t9_run1.log`；两 rank 错误回执存 `t9_rank_receipts/rank{0,1}.{error,lifecycle}`。

真实进展（对比 round-1）：pair 成功 spawn，两 rank 均通过 admission/runtime 构建/数据同步，进入生产 `run_pp_trainer_session`——F1 已修复并越过；失败发生在训练循环**之前**的元数据构建。

### Finding F2（blocker，hot 夹具不满足共享 metadata 契约）

- **file/line**：`tests/test_moss_image_pp_trainer_hot.py:224`（`_CudaDatasetGroup.datasets = ()`）与 `:165`（`use_user_config=False`）→ `library/training/pp/trainer_session.py:520→441`（`_build_metadata` 在训练循环前无条件调 `add_dataset_metadata`）→ `library/training/metadata.py:258`（`assert len(train_dataset_group.datasets) == 1`）。
- **现象**：两 rank 同点同错 `AssertionError: There should be a single dataset but 0 found`，exitcodes [1,1]，各留新错误回执；无任何 step 执行、无 checkpoint/LoRA 写出。
- **归属**：实现侧契约是合理的——生产 PP session 的 dataset group 来自 `_prepare_dataset`（真实单数据集组）；hot 夹具用注入 cached batch 的空 stub 不满足该契约。属测试夹具缺陷，非 PP 实现缺陷，也非 gate/DDP 问题。
- **最小修法**（tests 侧，二选一）：①handoff 改 `use_user_config=True`（`metadata.py` 的 user-config 分支遍历 `datasets` 空元组不触发 assert，产出空数据集 metadata，与"注入 cached batch"语义一致）；或②给 `_CudaDatasetGroup` 补一个满足 `add_dataset_metadata` 遍历的最小单数据集 stub。不改 library。
- **非根因排除**：两 rank 对称失败（非 rank 间协商/竞态）；发生在 GPU 计算开始前（非数值/显存问题）；gate/进程组/模型构建均已在失败点之前走通。

### 重跑纪律

本轮不重跑。F2 修复（tests 侧最小改动）后按同一预算单次重跑取证；阈值不据候选调整。

## 6a. verification-round-3（t11，2026-11-15 02:3x；t10 修复 F2 方案①后）

### 6a.1 CPU 与收集（全部真实 exit 0）

`git diff --check` 干净；组1 **19 passed**（77.00s）、组2 **127 passed**（29.20s）、组3 **79 passed**（6.21s）、hot collect **1 test**。t10 修复字节绑定：`tests/test_moss_image_pp_trainer_hot.py` SHA256 `918466e5fd65b07904975d6befa935e4519ed53e4d2386ed64962f63716dd4bd`（与 t10 报告一致）；library 未变。

### 6a.2 GPU gate

`gpu-gate.sh --count 2 --json`（原始存 `moss_pp_trainer_wiring_cuda_gate_t11.json`）：两 V100 仍各驻留用户 vLLM 11792 MiB、usable=no（余量 4592 MiB ≥ 2048）；沿用用户 2026-11-15 02:08 明确 GPU 授权执行；无旧测试进程残留。

### 6a.3 hot 单次运行（rc=1，真实失败——新缺陷 F3）

命令同前两轮。**rc=1，1 failed，14.52s**。原始输出 `moss_pp_trainer_wiring_cuda_hot_t11_run1.log`；两 rank 回执 `t11_rank_receipts/rank{0,1}.{error,lifecycle}`。两 rank 同点同错，exitcodes [1,1]，零 step 执行。

### Finding F3（blocker，t10 修复不完整：删除属性而非保留空元组）

- **file/line**：`tests/test_moss_image_pp_trainer_hot.py`（t10 改动处，`_CudaDatasetGroup` 现无 `datasets` 属性）→ `library/training/metadata.py:252`。
- **问题**：t10 采用方案①（`use_user_config=True`）但其"连带回退"把 `_CudaDatasetGroup.datasets` 属性**整个删除**（上一轮是 `datasets = ()`）；而 t10 的核验主张"user-config 分支只遍历 datasets（空组安全返回空 dict、**不访问属性**）"**不成立**——`metadata.py:252` `_build_dataset_metadata_user_config(train_dataset_group.datasets)` 本身就访问该属性，空元组才会安全，**无属性**则 `AttributeError: '_CudaDatasetGroup' object has no attribute 'datasets'`。
- **最小修法**（tests 侧，一行）：在 `_CudaDatasetGroup` 恢复 `datasets = ()` 类属性（保留 t10 的 `use_user_config=True`），空元组经 `:252` 遍历产出空 dataset metadata。不改 library。
- **非根因排除**：两 rank 对称失败（非协商/竞态）；发生在 GPU 计算前（非数值/显存）；gate/进程组/runtime 构建/admission 均已在失败点之前走通（F1/F2 修复持续有效）。

### 重跑纪律

本轮不重跑。F3 修复后按同一预算单次重跑取证；阈值不据候选调整。

## 6b. verification-round-3 attempt 2（t11 第 4 轮，2026-11-15 02:4x；F3 修复后）

### 6b.1 CPU 与收集（全部真实 exit 0）

`git diff --check` 干净；组1 **19 passed**（77.12s）、组2 **127 passed**（29.14s）、组3 **79 passed**（6.19s）、hot collect **1 test**。修复字节绑定：`tests/test_moss_image_pp_trainer_hot.py` SHA256 `6b478649409fd1286cbf89799c82e5d5a19134bf69f8134860c5d310cff98b09`（恢复 `datasets = ()` + 保留 `use_user_config=True`，正是 F3 最小修法）；library 未变。

### 6b.2 GPU gate

`gpu-gate.sh --count 2 --json`（原始存 `moss_pp_trainer_wiring_cuda_gate_t11r4.json`）：两 V100 仍各驻留用户 vLLM 11792 MiB、usable=no（余量 4592 MiB ≥ 2048）；沿用用户 2026-11-15 02:08 明确 GPU 授权执行；无旧测试进程残留。

### 6b.3 hot 单次运行（rc=1，34.27s——**两 phase 真实跑完 GPU 训练与恢复，仅末尾断言失败**）

原始输出 `moss_pp_trainer_wiring_cuda_hot_t11r4_run.log`；两 phase 的 rank0/rank1 JSON 回执与显存回执存 `t11r4_phase_receipts/`。**首次取得的生产入口热证据（全部真实）**：

- **straight phase（4 步，双 rank GPU）**：step1 正常更新（rank1 loss=1.148719，scale 65536）；**step2 真实 fp16 overflow → 两 rank 一致 skip**（`skipped 1`，scale 65536→32768 backoff，rank0/1 loss=None 同 skip）；step3/4 正常更新（loss 1.193529/1.178362）；`effective` 计数（1→1→2→3）与 `global_step`（1..4）符合 §4 语义；LR 恒 0.001（skip 步不推进的语义与 4 步短窗一致）。
- **保存 + 新进程 resume**：straight 写出 step-4 bundle（run root=`...pp_runs/<run_id>-step4`，run_id 为 64-hex）；resume phase 新进程对从 step-2 接续，`resume_step=2`，step3/4 接续完成。
- **双 rank 一致性**：两 rank `final_param_digest` 全等（straight `b6c2ac08…` = resumed `b6c2ac08…`——**恢复后同窗口与不中断 run 权重一致**）；两 rank `final_lr` 全等 `[0.001]`。
- **显存**：allocated 22,052,352 / 21,003,776 B，reserved 27,262,976 B（rank0/rank1），远低于 2GiB cap。

### Finding F4（blocker，测试设计与 shuffle 置换错配；实现侧行为正确）

- **file/line**：`tests/test_moss_image_pp_trainer_hot.py:95`（断言 resumed 窗口含 `skipped 1)`）及 `:92-93` 设计注释（"the overflow batch is step 3"）↔ `:237`（`scale = 400.0 if index == 2`）↔ `library/training/pp/trainer_session.py:120-133`（loader `shuffle=True`，置换由 `(seed, epoch)` 钉死）。
- **现象**：断言失败——resumed 窗口（step3/4）`skipped 0`，scale 恒 32768，rank1 loss 有限（1.193529/1.178362）。
- **根因**：放大样本（列表 index 2）在 seed 钉死的 shuffle 置换下落在**step 2**（straight 的 skip 发生在 step2），先于 resume 接续点；resumed 窗口消费的置换位 3/4 都是非放大样本 → 无 overflow。测试注释假设"放大样本=step 3"，与实际置换不符。`_batch_digests`（:319-327）按**列表序**生成，不能反映 loader 置换序，掩盖了这一错配。
- **归属**：测试设计缺陷。**实现侧行为正确**——本轮已实证：真实更新、单 rank inf 梯度触发两 rank 一致 skip、scale backoff、global/effective 语义、批次身份确定性、保存、新进程 resume、双 rank 参数 digest 与 LR 全等。
- **最小修法**（tests 侧）：fixture 按 loader 同源置换计算放大位置——用 `torch.randperm(4, generator=torch.Generator().manual_seed(args.seed))` 求出 shuffle 序，对**置换位 2**（=step 3，恰在 resumed 窗口内）的样本施加 400× 放大（替换固定 `index == 2`）；或等价地放大置换位落进 step3/4 的样本。不改 library。
- **非根因排除**：非数值/显存（skip 机制在 straight phase 已真实触发且两 rank 一致）；非 resume 恢复缺陷（param digest 跨 phase 全等）；非协商/竞态（两 rank 回执对称）。

### 重跑纪律

本轮不重跑。F4 修复后按同一预算单次重跑取证；阈值不据候选调整。

## 6c. verification-round-3 attempt 3（t11 第 5 轮，2026-11-15 02:5x；t13 同源置换修复后）

### 6c.1 CPU 与收集（全部真实 exit 0）

`git diff --check` 干净；组1 **19 passed**（76.82s）、组2 **127 passed**（28.78s）、组3 **79 passed**（6.24s）、hot collect **1 test**。字节绑定：`tests/test_moss_image_pp_trainer_hot.py` SHA256 `527050fa863aa30b9d7b8aa49778e54b657de8fd8c245466c55d5eb32f492112`（t13 同源置换修复在位：fixture 以 `torch.randperm(4, generator(seed))` 排布样本、对置换位 2 施加 400× 放大，断言零改动）；library 四模块 SHA 与 t2 完全一致未变。

**置换独立复算**：本验证者独立执行 `torch.randperm(4, generator=torch.Generator().manual_seed(41))` → `[0, 3, 2, 1]`，与实现者声明一致；epoch0 顺序下置换位 2 = step 3（resumed 窗口内）。置换推导成立。

### 6c.2 GPU gate

`gpu-gate.sh --count 2 --json`（原始存 `moss_pp_trainer_wiring_cuda_gate_t11r5.json`）：两 V100 仍各驻留用户 vLLM 11792 MiB、usable=no（余量 4592 MiB ≥ 2048）；沿用用户 2026-11-15 02:08 明确 GPU 授权执行；无旧测试进程残留。

### 6c.3 hot 单次运行（rc=1，34.02s——F4 修复未达设计意图，暴露更深根因 F5）

原始输出 `moss_pp_trainer_wiring_cuda_hot_t11r5_run.log`；双 phase 回执 `t11r5_phase_receipts/`。straight 与 resumed 再次全程真实跑完（param digest 跨 phase 全等 `b6c2ac08…`、resume_step=2、显存 22.0MB/27.3MB），但 **straight 的 skip 仍发生在 step 2**（而非置换位 step 3），resumed 窗口 `skipped 0`、scale 恒 32768、rank1 loss 有限（1.193529/1.178362）。

### Finding F5（blocker，overflow 驱动假设错误：初始 scale 自然溢出主导，放大样本在 32768 下不溢出）

- **file/line**：`tests/test_moss_image_pp_trainer_hot.py`（fixture 放大驱动与 `:95` 断言的设计假设）。
- **决定性对照证据**：round-4（放大样本=列表 index2，置换位 step3）与 round-5（同源置换，放大样本=置换位 step3）**两轮的 skip 都发生在 step 2**——与放大样本位置无关。结论：step-2 overflow 由**初始 GradScaler scale=65536 下 tiny fp16 模型的普通梯度自然溢出**触发（backoff 65536→32768），而非 400× 放大样本；backoff 后 400× 放大批次（resumed step3 消费，digest `aadb6b28…`）在 scale=32768 下不再溢出 → resumed 窗口不可能 skip。置换修复（F4）方向正确但未触及真正驱动。
- **附带观察（供 engineer 裁定，非本轮判定）**：straight 结束时累计 skipped=1，resumed 的 progress 行从 `skipped 0` 重新计数——若 §5.3 要求 skip 事件序列跨 resume 一致，累计 skip 计数是否应随 trainer 状态恢复需实现者确认（可能是展示计数而非恢复缺陷）。
- **最小修法**（tests 侧，两选一）：①把放大因子提高到能在 scale=32768 下溢出（先在 CPU 上实测该 tiny 模型 raw 梯度幅度确定系数，预登记后写入 fixture，禁止看 GPU 结果拟合）；②保留 400× 但把断言改为"straight phase skip + resume 从 checkpoint 恢复 skip 语义（last_skip/计数）"，需先厘清上一条附带观察。不改 library。
- **非根因排除**：非实现缺陷——skip 协议、backoff、effective/global 语义、保存、resume 参数/LR 全等、显存回执全部持续正确；非置换推导错误（本验证者已独立复算 perm=[0,3,2,1]）。

### 重跑纪律

本轮不重跑。F5 修复后按同一预算单次重跑取证；阈值不据候选调整。

## 6d. verification-round-3 attempt 4（t11 第 6 轮，2026-11-15 03:3x；F5 放大重设计后）

### 6d.1 CPU 与收集（全部真实 exit 0）

`git diff --check` 干净；组1 **19 passed**（77.16s）、组2 **127 passed**（29.15s）、组3 **79 passed**（6.13s）、hot collect **1 test**。字节绑定：hot 测试 SHA256 `4b18e81c973c2ff13c843d57709d4a0d60092f477118614e54dcbe287899e5c6`（F5 修复：放大系数预登记 200000×（CPU 推导：放大 latents 达 fp16 max 的 10.5×，前向即 Inf），放大批次置于 loader 位 1、2；library 未变——`pp/trainer_session.py` 仍 `00377202…`）。

### 6d.2 GPU gate

`gpu-gate.sh --count 2 --json`（原始存 `moss_pp_trainer_wiring_cuda_gate_t11r6.json`）：两 V100 仍各驻留用户 vLLM 11792 MiB、usable=no（余量 4592 MiB ≥ 2048）；沿用用户 2026-11-15 02:08 明确 GPU 授权执行；无旧测试进程残留。

### 6d.3 hot 单次运行（rc=1，33.47s——**真实轨迹与预登记逐项吻合，仅回执口径断言失败**）

原始输出 `moss_pp_trainer_wiring_cuda_hot_t11r6_run.log`；双 phase 回执 `t11r6_phase_receipts/`。**实测轨迹（两 rank 一致，param digest 跨 phase 全等 `b319de88…`、resume_step=2、显存 22.0MB/27.3MB）**：

- straight：step1 eff@65536 → step2 skip（→32768）→ step3 skip（→16384）→ step4 eff@16384（累计 skipped=2）——与预登记轨迹**完全一致**（放大 latents 前向 Inf → 两 rank 一致 skip、双重 backoff）；
- resumed：resume_step=2 接续，step3 消费放大批次 → **skip（→16384）**，step4 eff@16384——resumed 窗口内真实 overflow skip **首次取得**。

### Finding F6（blocker，收尾级：回执口径错配——progress 每步打印 vs 断言假设只含 skip 行）

- **file/line**：`tests/test_moss_image_pp_trainer_hot.py:106`（`assert len(straight["skip_lines"]) == 2`，实测 4）↔ `library/training/pp/trainer_session.py` `_report_progress(log_every=1)`（每步都向 `args._printed_skips` 追加 progress 行——实现在 :508-548 的设计即每步打印）。
- **问题**：断言假设回执列表只含 skip 事件行（预登记轨迹按 skip 行写），但实现按合同每步打印 progress（`pp step N/M (effective E, skipped K) loss=L lr=R scale=S`）；4 步 → 4 行 → `len==2` 失败。断言的**内容**核验全部通过（"step 2/4"、"skipped 2)"、"scale=16384.0"、resumed "step 3/4"/"skipped 1)"/"scale=16384.0" 均在实测行中）。
- **归属**：测试回执口径缺陷（F5 修复时重写了断言组但未适配"每步一行"的回执）。实现侧每步打印符合进度报告语义，非缺陷。
- **最小修法**（tests 侧，一处）：写回执时过滤为 **skip 事件行**——保留累计 `skipped` 计数相对上一行增加的行（straight → 2 行、resumed → 1 行），或直接按 `"skipped K)"` 计数变化推导；断言组本身（预登记轨迹）零改动即可通过。不改 library。
- **非根因排除**：非数值/驱动问题（预登记轨迹逐项吻合）；非 resume 缺陷（digest 全等）；非协商问题（双 rank 对称）。

### 重跑纪律

本轮不重跑。F6 修复后按同一预算单次重跑取证；阈值不据候选调整。


## 6e. verification-round-3 attempt 5（t11 第 7 轮，2026-11-15 03:4x；F6 过滤器修复后）

### 6e.1 CPU 与收集（全部真实 exit 0）

`git diff --check` 干净；组1 **19 passed**（77.56s）、组2 **127 passed**（28.92s）、组3 **79 passed**（6.27s）、hot collect **1 test**。字节绑定：hot 测试 SHA256 `72cd73c15a3e33d6c8e361c5c7cc4ba9d1eb8c031c846b69fe3f1a4c4c23320a`（F6 过滤器：仅在累计 skipped 计数增加时记录回实行）；library 未变（`pp/trainer_session.py` 仍 `00377202…`）。

### 6e.2 GPU gate

`gpu-gate.sh --count 2 --json`（原始存 `moss_pp_trainer_wiring_cuda_gate_t11r7.json`）：两 V100 仍各驻留用户 vLLM 11792 MiB、usable=no（余量 4592 MiB ≥ 2048）；沿用用户 2026-11-15 02:08 明确 GPU 授权执行；无旧测试进程残留。

### 6e.3 hot 单次运行（rc=1，33.41s——实质轨迹与 r6 完全一致，仅过滤器首行残留）

原始输出 `moss_pp_trainer_wiring_cuda_hot_t11r7_run.log`；双 phase 回执 `t11r7_phase_receipts/`。实测：straight step1 eff@65536 → step2 skip(→32768) → step3 skip(→16384) → step4 eff（param digest `b319de88…` 跨 phase 全等、resume_step=2、resumed step3 skip→16384、显存 22.0MB/27.3MB）——与 r6 及预登记轨迹一致。

### Finding F7（blocker，一行残留：过滤器首行无条件追加）

- **file/line**：`tests/test_moss_image_pp_trainer_hot.py:239`（`if not receipt or _cum(line) > _cum(receipt[-1])`）。
- **问题**：`not receipt` 分支在首行**无条件追加**，未检查计数>0——step1 的 `skipped 0`（非 skip 事件）被收进回执（straight 3 行 vs 断言 2 行）。step2/3/4 的过滤行为正确（计数增加才追加）。
- **最小修法**（tests 侧一行）：首行追加条件改为 `_cum(line) > 0`，即 `if (not receipt and _cum(line) > 0) or (receipt and _cum(line) > _cum(receipt[-1]))`。
- **非根因排除**：实质训练/恢复行为与 r6 逐项一致且全部正确（预登记轨迹、双 rank 对称、digest 全等、显存达标）；非实现缺陷。

### 重跑纪律

本轮不重跑。F7 修复后按同一预算单次重跑取证；阈值不据候选调整。

## 6f. verification-round-3 attempt 6（t11 第 8 轮，2026-11-15 03:5x；F7 一行修复后）

### 6f.1 CPU 与收集（全部真实 exit 0）

`git diff --check` 干净；组1 **19 passed**（79.03s）、组2 **127 passed**（28.84s）、组3 **79 passed**（6.10s）、hot collect **1 test**。字节绑定：hot 测试 SHA256 `241110e0ef3a5bc9c65ab388533946f1494ea47447c07a0fe3c1541f38fd3ad7`（F7 修复：首行追加加 `_cum(line) > 0` 条件）；library 未变（`pp/trainer_session.py` 仍 `00377202…`）。

### 6f.2 GPU gate

`gpu-gate.sh --count 2 --json`（原始存 `moss_pp_trainer_wiring_cuda_gate_t11r8.json`）：两 V100 仍各驻留用户 vLLM 11792 MiB、usable=no（余量 4592 MiB ≥ 2048）；沿用用户 2026-11-15 02:08 明确 GPU 授权执行；无旧测试进程残留。

### 6f.3 hot 单次运行（rc=1，33.64s——预登记轨迹断言（:96-117）全部通过，仅剩回执缺键）

原始输出 `moss_pp_trainer_wiring_cuda_hot_t11r8_run.log`；双 phase 回执 `t11r8_phase_receipts/`。**F7 修复生效**：straight `skip_lines` 恰 2 行（step2 `skipped 1)+scale=32768.0`、step3 `skipped 2)+scale=16384.0`）、resumed 恰 1 行（step3 `skipped 1)+scale=16384.0`）——`:96-117` 预登记轨迹/digest/final_lr 断言**全部通过**（param digest 跨 phase 全等 `b319de88…`、显存 22.0MB/27.3MB）。

### Finding F8（blocker，最后一个回执缺口：straight 相位回执从未写入 `batch_digests`）

- **file/line**：`tests/test_moss_image_pp_trainer_hot.py` straight 回执写出（`_phase_straight` 尾部，仅含 `final_param_digest`/`final_lr`/`root`/`skip_lines`）↔ `:118`（`resumed["batch_digests"][2:] == straight["batch_digests"][2:]`）。
- **问题**：`KeyError: 'batch_digests'`——straight 相位 payload 历来不含该键（r4-r8 各轮回执均如此），此前轮次从未走到 :118（被更早断言拦截）；resumed 相位有 `"batch_digests": _batch_digests(handoff)`。
- **最小修法**（tests 侧一行）：straight 回执 payload 补 `"batch_digests": _batch_digests(handoff)`。语义上两侧同 group 同 list 序，`:118` 比较应直接通过（resumed 侧实测 `[efeb90cf…, eb440925…, eb440925…, efeb90cf…]`）。
- **非根因排除**：非实现缺陷（实现侧 batch 身份由 loader/协商对账保证，r6-r8 恒定）；纯测试回执缺键。

### 重跑纪律

本轮不重跑。F8 修复后按同一预算单次重跑取证；阈值不据候选调整。

## 6g. verification-round-3 attempt 7（t11 第 9 轮，2026-11-15 03:5x；F8 修复后——**rc=0 全绿**）

### 6g.1 CPU 与收集（全部真实 exit 0）

`git diff --check` 干净；组1 **19 passed**（78.18s）、组2 **127 passed**（29.15s）、组3 **79 passed**（6.07s）、hot collect **1 test**。字节绑定：hot 测试 SHA256 `07da5ddf2d2dc617ee6f56bd8a882303fe5e3bc746eecd394928ffc69a0d156e`（F8 修复：straight 回执补 `batch_digests`）；library 未变（`pp/trainer_session.py` 仍 `00377202…`）。

### 6g.2 GPU gate

`gpu-gate.sh --count 2 --json`（原始存 `moss_pp_trainer_wiring_cuda_gate_t11r9.json`）：两 V100 仍各驻留用户 vLLM 11792 MiB、usable=no（余量 4592 MiB ≥ 2048）；沿用用户 2026-11-15 02:08 明确 GPU 授权执行；无旧测试进程残留。

### 6g.3 hot 单次运行——**rc=0，1 passed，33.35s（首次完整通过）**

原始输出 `moss_pp_trainer_wiring_cuda_hot_t11r9_run.log`；双 phase 回执 `t11r9_phase_receipts/`。**全部断言通过，完整热证据如下（全部真实，无 mock）**：

- **straight 轨迹（4 步，双 rank GPU，global 4 / effective 2 / skipped 2）**：step1 eff@65536 → step2 skip（→32768）→ step3 skip（→16384）→ step4 eff@16384；skip_lines 恰 2 行，末行 `skipped 2) … scale=16384.0`；放大 latents 前向 Inf（200000× 预登记驱动）触发两 rank 一致 skip。
- **保存 + 新进程 resume**：step-4 bundle（run_id 64-hex，`pp_checkpoint.json` manifest）；resume `resume_step=2`（step-2 bundle，scale 32768）接续。
- **resumed 窗口**：step3 消费放大批次 → **真实 skip（→16384）**，恰 1 行 `step 3/4 … skipped 1) … scale=16384.0`；step4 eff@16384 且 loss 与 straight step4 逐位相同。
- **恢复全等**：`final_param_digest` 跨 phase 双 rank 全等（`b319de881c11f6d99df53f35af086fd463d21daadd5eb19d24d6af9c53590d8b`）、`final_lr` 全等 `[0.001]`、`batch_digests` 两侧全等（[efeb90cf…, eb440925…, eb440925…, efeb90cf…]）。
- **显存**：allocated 22,052,352 / 21,003,776 B，reserved 27,262,976 B（两卡），cap 2GiB——远低于上限。

### 6g.4 结论

`test_two_rank_cuda_trainer_resume` 在固定双 V100、timeout 60s / 内部 45s / 每卡 ≤2GiB 预算内**完整通过**（rc=0）：MOSS 主训练 PP 接线在真实生产入口（`run_pp_trainer_session`）下完成真实 fp16 训练、确定性 overflow 双 rank 一致 skip 与双重 backoff、分 rank checkpoint 保存、新进程 resume 同窗口参数/LR/批次身份全等恢复。F1–F8 全部修复闭环。验收依据：需求 v3（含 §4/§5 步进与恢复语义）。



## 7. 证据文件清单

| 文件 | 内容 |
| --- | --- |
| `moss_pp_trainer_wiring_cuda_gate_20261115t0110.json` | 第一次 gate（rc=3，usable=no） |
| `moss_pp_trainer_wiring_cuda_gate_pre-hot.json` | 授权后 gate（rc=0，两卡仍 usable=no + vLLM 驻留） |
| `moss_pp_trainer_wiring_cuda_hot_run1.log` | hot 首跑完整原始输出（rc=1，F1） |
| `moss_pp_trainer_wiring_cuda_gate_t9.json` | round-2 gate（rc=0，两卡仍 usable=no + vLLM 驻留） |
| `moss_pp_trainer_wiring_cuda_hot_t9_run1.log` | round-2 hot 完整原始输出（rc=1，F2） |
| `t9_rank_receipts/rank{0,1}.{error,lifecycle}` | round-2 两 rank 的 worker 错误回执与生命周期记录 |
| `moss_pp_trainer_wiring_cuda_gate_t11.json` | round-3 gate（rc=0，两卡仍 usable=no + vLLM 驻留） |
| `moss_pp_trainer_wiring_cuda_hot_t11_run1.log` | round-3 hot 完整原始输出（rc=1，F3） |
| `t11_rank_receipts/rank{0,1}.{error,lifecycle}` | round-3 两 rank 的 worker 错误回执与生命周期记录 |
| `moss_pp_trainer_wiring_cuda_gate_t11r4.json` | round-3 attempt2 gate（rc=0，两卡仍 usable=no + vLLM 驻留） |
| `moss_pp_trainer_wiring_cuda_hot_t11r4_run.log` | round-3 attempt2 hot 完整原始输出（rc=1，F4） |
| `t11r4_phase_receipts/straight.rank{0,1}.json` | straight phase 双 rank 回执：4 步 loss/scale/skip 线、param digest、run root |
| `t11r4_phase_receipts/resumed.rank{0,1}.json` | resume phase 双 rank 回执：resume_step=2、batch digests、param digest、final_lr |
| `t11r4_phase_receipts/memory.rank{0,1}.json` | 每卡显存回执（allocated 22.0/21.0 MB，reserved 27.3 MB，cap 2GiB） |
| `moss_pp_trainer_wiring_cuda_gate_t11r5.json` | round-3 attempt3 gate（rc=0，两卡仍 usable=no + vLLM 驻留） |
| `moss_pp_trainer_wiring_cuda_hot_t11r5_run.log` | round-3 attempt3 hot 完整原始输出（rc=1，F5） |
| `t11r5_phase_receipts/` | attempt3 双 phase rank 回执与显存回执（param digest 跨 phase 全等） |
| `moss_pp_trainer_wiring_cuda_gate_t11r6.json` | round-3 attempt4 gate（rc=0，两卡仍 usable=no + vLLM 驻留） |
| `moss_pp_trainer_wiring_cuda_hot_t11r6_run.log` | round-3 attempt4 hot 完整原始输出（rc=1，F6） |
| `t11r6_phase_receipts/` | attempt4 双 phase rank 回执与显存回执（轨迹与预登记逐项吻合） |
| `moss_pp_trainer_wiring_cuda_gate_t11r7.json` | round-3 attempt5 gate（rc=0，两卡仍 usable=no + vLLM 驻留） |
| `moss_pp_trainer_wiring_cuda_hot_t11r7_run.log` | round-3 attempt5 hot 完整原始输出（rc=1，F7） |
| `t11r7_phase_receipts/` | attempt5 双 phase rank 回执与显存回执 |
| `moss_pp_trainer_wiring_cuda_gate_t11r8.json` | round-3 attempt6 gate（rc=0，两卡仍 usable=no + vLLM 驻留） |
| `moss_pp_trainer_wiring_cuda_hot_t11r8_run.log` | round-3 attempt6 hot 完整原始输出（rc=1，F8） |
| `t11r8_phase_receipts/` | attempt6 双 phase rank 回执与显存回执（预登记轨迹断言全过） |
| `moss_pp_trainer_wiring_cuda_gate_t11r9.json` | round-3 attempt7 gate（rc=0，两卡仍 usable=no + vLLM 驻留） |
| `moss_pp_trainer_wiring_cuda_hot_t11r9_run.log` | round-3 attempt7 hot 完整原始输出（**rc=0 全绿**） |
| `t11r9_phase_receipts/` | attempt7 双 phase rank 回执与显存回执（全部断言通过的最终证据） |

## 8. 未验证项（最终口径）

- ~~真实更新/overflow skip/resume 对齐/显存~~：**已于 round-3 attempt7（rc=0）全部取得**（见 §6g）。
- 仍不在本轮范围：长训、真实权重、训练质量结论；多桶/动态分辨率 PP；非 tiny 规模的外推。
