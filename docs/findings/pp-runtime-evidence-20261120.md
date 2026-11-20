# P2 门禁②③ 真机证据：WebUI 极小 MOSS-Image PP 队列 + 续训（2026-11-20）

状态：完成（t10，双 V100 真机实跑；规模极小 max≈3 步，非收敛性验证）

## 卡位自证（CUDA 枚序 ≠ nvidia-smi 索引）

```
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -c '...get_device_properties(i).name/uuid'
→ 0 Tesla V100-SXM2-16GB uuid=b4b014b2-...；1 Tesla V100-SXM2-16GB uuid=379b8c19-...
CUDA_VISIBLE_DEVICES=0,1,2 → 2 NVIDIA GeForce GTX 1050（sm_61，PyTorch 明确不兼容告警）
```
WebUI 训练子进程由 gpu_whitelist 走 `CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=<nvidia-smi 索引>`（`web/services/training/gpu.py:164-165`），因此 API 层 gpu_whitelist 传 nvidia-smi 索引 `[1,2]` = 两张 V100。

## 复现命令（全部真实退出码）

```bash
# 1. WebUI（工作树根，仅 127.0.0.1）
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python tasks.py web --host 127.0.0.1 --port 20177
# 2. 预检 + 预处理 + 入队（POST /api/training/{preflight,preprocess,queue}）
#    config: t10 系 TOML（已备份 /tmp/gpu-smoker/*.toml），要点：
#    model_family="moss_image" pipeline_parallel=true stages=2 microbatches=2
#    schedule="1f1b" split="balanced" mixed_precision="fp16" save_every_n_epochs=0
#    persistent_data_loader_workers=false max_train_epochs=2 batch=2 resolution=512
# 3. 续训：同 config + seed=3747612563 + resume=<pp_runs/...-step2 包根> + max_train_epochs=3
```

PP 准入 preflight 真实拒绝过的错误（fail-closed 具名）：`pp_microbatches_below_stages`（mb=1）、`save_every_n_epochs unset`、`persistent_data_loader_workers`、配置越界（config 必须在项目内）。

## 门禁②：队列/历史/报告/计数

| 项 | 证据 |
| --- | --- |
| 队列模式记录 | `POST /api/training/queue` 返回 item `distributed_mode="pp"`（5 个队列项全部 pp，含 3 个失败项） |
| 历史模式记录 | `configs/web-training-history/20261120-150246…` 与 `…152805…`（两条成功训练）meta `distributed_mode="pp"`；旧记录零改写（本轮之前历史为空） |
| 首轮报告计数 | `precision_report.json`: `updates={attempted:2, effective:2, skipped:0, total_global_step:2}`，`status="completed"`（非 incomplete/setup） |
| 续训报告计数 | `updates={attempted:1, effective:1, skipped:0, total_global_step:3}`（含继承 2，`resume.inherited_global_step=2`），`status="completed"` |
| fp16 GradScaler | 首轮 `amp: {native_amp:true, autocast:"fp16", scaler_present:true, scaler_initial:65536.0, scaler_final:65536.0}`；续训 `scaler_resumed:65536.0`（=首轮终值，跨进程持久化一致），日志 `pp step 1/2 … scale=65536.0` |
| 显存峰值 | 每步日志 `[pp] step N wall≈7s peak=7.5GiB / 6.5GiB`（rank0/rank1）；报告 `cuda_peak_allocated_bytes=8064627712` |
| MOSS NF4/BF16 TE 契约 | 预处理产 `*_moss_image_te_nf4.safetensors`（NF4 缓存）；训练全程只读缓存、不加载 TE；MOSS TE 运行时强制 BF16 计算且 frozen（`requires_grad_(True)` 具名拒绝） |

## 门禁③：续训

- **具名拒绝 1（默认路径，权重复用未显式 opt-in）**：把 step2 包去掉 `trainer_state_rank*.pt` 后 resume → 双 rank 对称 `PipelineParallelError: pp_checkpoint_incomplete: … no trainer-level resume state … Re-run with the explicit weights-only opt-in`（训练 rc=1，queue item error）。历史任务 `20261120-150718…`。
- **具名拒绝 2（身份绑定，bonus）**：resume 配置改 `output_name`/缺 `seed` → `pp_checkpoint_identity_mismatch`（identity = sha256(`family|seed|output_name`)，`trainer_session.py:142`；首轮 seed 随机生成，跨进程续训必须在配置里钉住同一 seed）。历史任务 `20261120-151625…`/`…152127…`。
- **精确续训成功**：`seed=3747612563` + `resume=<…-step2>` → 日志 `restore … at global_step=2, next epoch=2 batch_index=0`，恰好 +1 步（`pp step 3/3 (effective 3, skipped 0)`），新 bundle `…-step3`。per-rank 游标对照（trainer_state）：

| bundle | rank | global_step | effective | epoch | batch_index |
| --- | --- | --- | --- | --- | --- |
| step2 | 0/1 | 2 | 2 | 1 | 1 |
| step3 | 0/1 | 3 | 3 | 2 | 1 |

## 未验证

- 权重-only 续训（`pipeline_parallel_resume_weights_only=true`）的放行路径未实跑（只验了默认拒绝）。
- `pp_resume_no_work`（target ≤ checkpoint step）分支未触发。
- 真实 fp16 overflow skip 与 scale 退避（65536→…）未出现（2+1 步全部 effective）；该行为已有独立热测验收（`test_two_rank_cuda_trainer_resume`）。
- 收敛质量、长训练、Krea-2 族同链路未覆盖。

## 待复核

- `seed is None` 时 `train_session.py:144` 每次随机生成 → run identity 含随机种子，WebUI 跨进程续训必须显式钉 seed，否则必撞 `pp_checkpoint_identity_mismatch`（本次实测两次）。建议 WebUI 续训入口自动继承历史 seed 或给出提示。
- 大 JSON 原件：`output/runs/t10_moss_pp_*-{150209,152721}/training_output/{precision_report.json,pp_runs/}`（未跟踪运行数据）。
