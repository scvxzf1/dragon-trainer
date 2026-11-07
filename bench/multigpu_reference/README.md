# Multi-GPU reference experiments

状态：实验
适用版本：当前 `codex/multi-gpu-training` 工作树
入口：`bench/multigpu_reference/experiments.py`（被 `tests/test_multigpu_reference_*.py` 引用的门禁与预算定义）

本目录只放**隔离小模型**实验与真实训练的**入口门禁**，不放生产默认值，也不改动生产准入。

## 门禁

| 门 | 变量/条件 | 拒绝原因码 |
| --- | --- | --- |
| 释放设备工作 | `ANIMA_MULTIGPU_EXPERIMENT_RELEASE=1` | `experiment_release_missing` |
| 步数预算 | `<= 20` 步/次运行 | `experiment_budget_invalid` |
| 输出根 | 必须显式、且为空目录 | `experiment_output_root_invalid` |
| 设备数量 | `cuda_small` 需要 `world_size` 张可见卡 | `experiment_device_missing` |

预算常量与团队目标一致：单卡 10 步、双卡 20 步、双卡 10+10 续训，累计 ≤50 次有效更新，
单次运行 ≤20 分钟，全部真实训练 ≤80 分钟。

## 边界

- FSDP2 与 TP 实验只使用 `isolated_model_spec()` 生成的小模型；**冻结基座不做 shard**，
  也不假设"shard 之后注入会被自动管理"。
- 未设置释放变量时，设备实验一律 skip 或拒绝，不得把 skip 当成已验证。
- 真实 Anima 有界训练由验证任务在独立审批后执行，测试套件不会顺带启动它。

## t3 四项发现的闭合记录（t8，CPU/Gloo）

t3（v1 t6）的证据在 `output/tests/multigpu_reference_validation/t6-faultinj-cuda-20261108/`，
本次修复只改生产/入口代码与定向用例，**未改动该目录**。逐项对照：

| 发现 | 根因（源码事实） | 修复位置 | 证据（命令 → 用例） |
| --- | --- | --- | --- |
| F-T3-1a S-0 | accelerate 1.13.0 多进程分支用全局 RNG 抽 seed 重新 shuffle，单进程分支不这样做 ⇒ 同 seed 下 single/dual permutation 不同 | `library/training/multigpu/loader_seed.py`（显式 opt-in 门）、`library/runtime/accelerator.py`（`DataLoaderConfiguration` 合并）、`cli_args.py --multigpu_seedable_sampler`、`hot_run.py`（RUN-A/B/C 固定该 flag） | `tests/test_multigpu_reference_hot.py::test_seedable_sampler_gate_is_off_by_default`（默认不传给 Accelerator）、`::test_seedable_sampler_gate_is_opt_in_and_merges`、`::test_single_and_dual_permutations_match_only_with_the_gate`（真实单进程 vs 真实双 Gloo 进程，逐位相等；关门负对照不相等） |
| F-T3-1b batch_size | `bootstrap.py` 的 `DataLoader(batch_size=1)` 是包装层，真正的 per-device batch 来自 dataset blueprint 的 `batch_size`（默认 1） | `library/training/dataloader_batch.py`（`resolve_loader_batch` + 具名 `dataset_batch_size_mismatch`）、`bootstrap.py` 接线 | `tests/test_multigpu_reference_hot.py::test_loader_batch_plan_keeps_the_wrapper_at_one_and_reads_the_blueprint`、`::test_single_batch2_and_dual_batch1_cover_the_same_window_multiset` |
| F-T3-2 游标差 1 | 生产跳过序列 = `skip_first_batches(initial_step-1)` + 循环里一次 `continue`，首个真正执行的 batch 的 local index 是 `initial_step` | `cursor.resume_skipped_batches`（唯一生产者）、`cursor.expected_position`、`train_session` 改为调用它 | `tests/test_multigpu_cursor_wiring.py::test_resume_counter_equals_the_real_first_batch_index`（真实 DataLoader + `skip_first_batches` 语义）、`::test_resumed_state_is_exact_at_the_real_first_batch_position`（cursor + 真实 RNG 证据 + fixture slot 一条链） |
| F-T3-3 RNG 证据 | accelerate 真实键是 `random_state / numpy_random_seed / torch_manual_seed[ / torch_cuda_manual_seed]`，旧读取端按 `python/numpy/torch` 判定 ⇒ 恒缺失、`exact` 不可达、cursor 漂移被 `..._degraded` 早退吞掉 | `library/training/multigpu/rng_evidence.py`（真实键映射 + 每 rank 记录 schema）、`resume_protocol.validate_bundle`（cursor 判定提前到 RNG 之前，次要问题仍写进 message） | `tests/test_multigpu_resume_protocol.py::test_real_accelerator_save_produces_readable_rng_evidence`（真实 `Accelerator.save_state` 产物 → exact=True 可达）、`::test_cursor_drift_is_not_swallowed_by_rng_degradation` |
| F-T3-4 DTensor 比较 | `torch.allclose` 在 torch 2.10 拒绝 `Tensor`/`DTensor` 混比 | `bench/multigpu_reference/sharding.py::materialize`（`full_tensor`/`to_local`）、`_task_device()`（按 backend 选设备） | `tests/test_multigpu_reference_experiments.py::test_fsdp2_numeric_through_the_two_process_entry_on_cpu_gloo`（真实双 Gloo 进程 CPU 通过）、`::test_materialize_unwraps_a_dtensor_like_value` |

补充约定：hardware 标记的用例（NCCL/CUDA、需要外部进程组的 FSDP2/TP）仍然 **skip**，
skip 不计通过；`run_two_rank` 在 `nccl` 且未 `MULTIGPU_REFERENCE_HARDWARE=1` 时于 spawn
之前具名拒绝，不占 GPU。

## 真跑前置（t13 F-T13-2 / F-T13-3）

`plan_hot_run` 产出的命令可以**真的**执行，但有三件事不在仓库里，必须由跑的人补上：

1. **解释器**：runner 必须用项目 venv 的解释器
   `/mnt/SD_Dataset/训练器相关/krea2-webui/.venv-v100/bin/python`，不能依赖 `PATH`。
   `hot_run` 现在默认取 `sys.executable`（也就是启动 harness 的那个 python），并接受
   显式参数 `python=<绝对路径>`；传入裸名（如 `"python"`）会被
   `hot_run_interpreter_invalid` 拒绝。t13 的第 2 次尝试就是因为前缀是裸 `python`，
   子进程解析到没有 torch 的解释器，9 秒后 `ModuleNotFoundError: torch`。
   自检：`plan.command[0]` 必须是绝对路径，且 `[plan.command[0], "-c", "import torch"]` 能过。

2. **基座权重**：树内默认 `base_assets` 路径当前是**断链**，真跑必须显式覆盖到实际存在的
   文件，例如
   `/mnt/SD_Dataset/训练器测试/comfyui/models/diffusion_models/anima/anima-base-v1.0.safetensors`。
   这是运行参数，不改 `configs/`、不改库默认值；断链在 `plan_hot_run` 之前的资产校验里
   就会失败，不会退化成一个"看起来跑过了"的结果。

3. **训练超参走 CLI，不写进 `--config_file`**：`--config_file` 是数据集/方法 blueprint，
   训练超参（如 `--network_train_unet_only`，本来就是 `store_true` 旗标）直接加在命令行。
   不要为了"让某个 config 生效"去扩 `--config_file` 的语义，也不要放宽
   `extra_args.py` 对"缓存 TE 输出却仍训练 TE"这类无意义组合的断言。

另外：`launch_env` 里不适用于本次运行的键现在**直接省略**（单卡不再出现
`ANIMA_ACCELERATE_NUM_PROCESSES=None`，`Popen` 遇到 `None` 值会抛 `TypeError`，
t13 第 1 次尝试就是这么挂的）；`execute` 侧再过滤一次，并把发射键叠加到当前环境之上，
真实 runner 拿到的 `env` 不会是"只有两三个键"的残缺环境。
