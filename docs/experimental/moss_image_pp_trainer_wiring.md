# MOSS-Image 主训练 PP 接线（trainer-level wiring）

状态：实验（CPU/Gloo 双 rank 与双 V100 CUDA tiny 热测均已独立验收通过——
`tests/test_moss_image_pp_trainer_hot.py::test_two_rank_cuda_trainer_resume`
t11 第 9 轮 rc=0/1 passed/33.35s，真实生产入口、无 mock；真实大模型权重训练未验收）
适用版本：分支 `codex/moss-pp-v0-20261112`
入口命令：见下文「启动」；普通非 PP 训练入口与行为完全不变
相关代码：`library/training/pp/trainer_session.py`（唯一执行派发）、
`library/training/pp/trainer_admission.py`（入口预检）、
`library/training/pp/trainer_inputs.py`（batch/噪声同步）、
`library/training/pp/trainer_resume.py`（trainer 级恢复元数据）、
`library/training/train_session.py`（冻结分叉点）、
`library/training/cli_entry.py`、`library/runtime/accelerator.py`
需求契约：`docs/proposal/moss_image_pp_trainer_wiring.md`（t1 冻结原文）

一句话：已验收的 MOSS 库级 PP runtime（`moss_runtime.py`）现在由正式
CLI / `AnimaTrainer.train` 调用链真实进入：`train_session._run_training_session`
在共享 cache/模型/network setup 之后、数据并行 optimizer/dataloader 装配之前
分叉到专用 PP 训练会话；非 PP 与 Anima 旧行为零改动。

## 放行范围（fail-closed，全部具名拒绝）

只放行已接线且验收的配置形状；每个未支持项在入口（`trainer_admission`）具名
拒绝（`pp_config_invalid` / `pp_trainer_path_unavailable`），不做静默忽略：

| 维度 | 放行值 |
| --- | --- |
| 模型族 / stages / schedule / split | `moss_image` / 2 / `1f1b` / `balanced` |
| 精度 | `mixed_precision=fp16`（纯 fp16 + GradScaler，fp32 LoRA master） |
| adapter | `networks.lora_anima` plain LoRA，`network_train_unet_only` |
| 缓存 | `cache_latents` / `cache_text_encoder_outputs` 完整磁盘缓存 |
| loss | 标准 flow-matching MSE（`loss_type=l2`、`weighting_scheme=none`），microbatch 归一化只在 schedule owner 一处 |
| accumulation | `gradient_accumulation_steps=1` |
| 其他 | 无 preview/compile/stage_schedule/adaptive OOM/epoch 级保存计划/persistent workers；未接线族（krea2_raw / z_image / qwen_image_2_1）维持 `pp_trainer_path_unavailable`。**Anima 状态**：本接线的四个 trainer 模块已泛化为族适配器（`pp/trainer_adapters.py`），Anima 已接线专用 PP 训练会话并独立热测验收（rc=0）——见 [anima_pp_trainer_wiring.md](anima_pp_trainer_wiring.md)；MOSS 分支行为冻结不变 |

进程组：首次（也是唯一一次）初始化在 `prepare_accelerator` 构造 Accelerator
**之前** 注入多后端组 `cpu:gloo,cuda:nccl`（CPU 标量走 gloo、设备激活走 nccl，
这是 PP 事务的硬要求）；随后用共享 validator 校验，绝不重建组掩盖错误后端。

## 数据 / 噪声 / 步进语义

* loader 复用 `resolve_loader_batch`（dataset item 本身就是整 batch，wrapper
  恒为 1），两个 stage 迭代同一确定性 per-epoch（`seed+epoch` 生成器）顺序，
  绝不 `accelerator.prepare`（那会装 DistributedSampler 分片 batch）。
* 每个 step 双方先对 latents 做 SHA-256 字节摘要 all-gather 对账（强摘要，
  非自增 id、非 sum/numel 弱校验）；rank0 用提取出的
  `family.sample_moss_flow`（与非 PP 完全同一流匹配输入/target 数学）采样并
  广播 `noise/sigmas`，stage1 的 `target=noise-latents` 位级一致。
* batch prep 失败先 `all_gather_object` 协商后再进入任何 collective；单 rank
  失败 = 双方同码具名拒绝。
* 步进：`global_step` 计消耗的 pipeline step（含 skip）；`effective_step` 与
  LR scheduler 只在成功 step 前进；overflow 由 fp16 事务全 rank 一致 skip，
  skip 不推 LR、不动参数。

## 保存与恢复

* PP resume 包 = 每 rank `pp_stage-rank{r}.pt`（权重/opt/scaler，沿用
  `pp.checkpoint` 协商协议）+ 每 rank `trainer_state_rank{r}.pt`（scheduler、
  torch/python/numpy 三路 RNG、数据 cursor `(epoch, batch_index)`、run
  identity）；save/load 全部 read→validate（版本/family/run identity/step）→
  all-gather 同意→apply，拒绝发生在任何 mutation 之前。
* run root 每次新建（拒绝覆盖既有目录），rank0 写 `pp_runs/LATEST.json` 指针；
  同 step 周期保存与结束保存幂等跳过。
* 最终 LoRA 走普通输出格式：rank0 `all_gather_object` 汇齐两 stage 的 adapter
  状态后经生产 `CheckpointSaver.save_final` 写 `<output_name>.safetensors`。
* 不触碰 `accelerator.save_state/load_state`（`pp_accelerate_checkpoint_unsupported`
  守卫保持武装）；DDP 内部守卫（bootstrap/loop 的 `require_trainer_topology`）
  全部保留——接线的 MOSS 族从分叉点离开共享路径，任何误路由到 DDP 内部的 PP
  运行都会被原样具名拒绝。
* 旧 PP 包缺 trainer 级状态默认具名拒绝；仅当显式 `pipeline_parallel_resume_weights_only=true`
  或 `MOSS_PP_RESUME_WEIGHTS_ONLY=1` 时按 weights-only 非精确续训（日志声明）。

## 启动（复用现有 launch 支持，无新命令框架）

```bash
# CPU/Gloo 冒烟（2 进程 = 2 stage）
CUDA_VISIBLE_DEVICES='' WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 \
  accelerate launch --num_processes 2 --cpu train.py --config <moss-pp-config.toml>

# 双 V100 生产形态（每卡一个 rank；空闲固定两卡，单节点 timeout 60 档热测归独立验证）
CUDA_VISIBLE_DEVICES=<uuid0>,<uuid1> PYTORCH_CUDA_ALLOC_CONF="" \
  accelerate launch --num_processes 2 --multi_gpu train.py --config <moss-pp-config.toml>
```

配置要点：`model_family="moss_image"`、`pipeline_parallel=true`、
`pipeline_parallel_stages=2`、`pipeline_parallel_microbatches=2`、
`pipeline_parallel_schedule="1f1b"`、`pipeline_parallel_split="balanced"`、
`mixed_precision="fp16"`、完整 latents/TE 缓存。续训把 `resume` 指向某个
`pp_runs/<run>-step<N>` 包根目录。不自动改用户 configs。

## 测试与验收面

* `tests/test_moss_image_pp_trainer.py`：入口路由（接线族放行、其他族与 DDP
  内部守卫保持）、进程组契约（真 Gloo 双 rank）、§4 步进语义（真 2-rank，
  真实 fp16 overflow skip）、loader 合同。
* `tests/test_moss_image_pp_trainer_checkpoint.py`：真双 rank 真会话入口
  straight(6) vs split(4)+resume(2) 的参数/LR/scale/RNG/cursor 全等对照、
  bundle/trainer 状态/最终 LoRA 产物、旧包默认拒绝与 weights-only 显式放行、
  坏 payload apply 前具名拒绝。
* `tests/test_moss_image_pp_trainer_hot.py`：固定热测节点（双 CUDA UUID gate、
  ≤2GiB/卡、45s/60s 双层预算、真实 skip、唯一调用目录与原始证据）。第 9 轮独立
  验收实测 rc=0（1 passed，33.35s）：straight 4 步真实 fp16 更新
  （global 4/effective 2/skipped 2，200000× 预登记 Inf 驱动两 rank 一致 skip 与
  双重 backoff，scale 65536→32768→16384）→ 分 rank checkpoint → 新进程
  resume_step=2 接续（step-2 bundle，scale 32768）→ step3 消费放大批次真实
  skip（→16384）、step4 loss 与 straight 逐位相同；final_param_digest 跨 phase
  双 rank 全等、final_lr 全等 [0.001]、batch_digests 全等；显存 allocated
  22.0/21.0 MB、reserved 27.3 MB ≪ 2 GiB。证据（docs/findings）：moss_pp_trainer_wiring_validation_20261114.md §6g、
  moss_pp_trainer_wiring_cuda_hot_t11r9_run.log、
  moss_pp_trainer_wiring_cuda_gate_t11r9.json、t11r9_phase_receipts/、
  独立审阅 moss_pp_trainer_wiring_review_20261114.md（verdict=pass）。

## 集成状态与字节绑定（t5）

工作树：分支 `codex/moss-pp-v0-20261112`（基线 HEAD `6610f710`，未提交开发成果，
无 commit/push）。本轮审阅通过并绑定的关键字节（SHA256，t4/t11r9 一致）：

| 文件 | SHA256 前 8 |
| --- | --- |
| `library/training/pp/trainer_session.py` | `00377202` |
| `library/training/pp/trainer_admission.py` | `df4d36ef` |
| `library/training/pp/trainer_inputs.py` | `b1dddfd8` |
| `library/training/pp/trainer_resume.py` | `0a556adb` |
| `tests/test_moss_image_pp_trainer_hot.py` | `07da5ddf` |

可复现命令（全部真实退出码，CPU 四条由实现侧多轮执行、GPU 单次由独立验证执行）：

```bash
# CPU 三套件（19/127/79 passed，rc=0）
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=. timeout 300 .venv/bin/python -m pytest -q -rs   tests/test_moss_image_pp_trainer.py tests/test_moss_image_pp_trainer_checkpoint.py
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=. timeout 300 .venv/bin/python -m pytest -q -rs   tests/test_moss_image_pp_wiring.py tests/test_anima_pp_wiring.py   tests/test_krea2_pipeline_parallel.py tests/test_multigpu_admission.py tests/test_pp_import_boundary.py
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=. timeout 300 .venv/bin/python -m pytest -q -rs   tests/test_training_bootstrap.py tests/test_training_loop_runtime.py tests/test_training_checkpointing.py
# 双 V100 热测（t11r9 rc=0；双 UUID gate，空闲窗口单次执行）
env MOSS_PP_HOT_ENABLE=1 CUDA_VISIBLE_DEVICES='GPU-b4b014b2-953b-28e8-b8c1-425c4a401165,GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061'   PYTHONPATH=. PYTHONUNBUFFERED=1 timeout 60 .venv/bin/python -u -m pytest -q -rs -s   tests/test_moss_image_pp_trainer_hot.py::test_two_rank_cuda_trainer_resume
```

## Web 启动链与后续工作（已更新）

WebUI/配置链（t5/t6/t9 轮）已补齐并独立验证：compat 预检按族感知（已接线族
提示「专用 PP 训练会话」，准入具名拒绝在预检显形）、`pipeline_parallel` 与
`model_family` 成对键强制（启动链 + 续训 gate 双侧）、队列链路真实发起证据
（enqueue → accelerate 双进程 → 逐级具名拒绝/准入通过）、gpu_whitelist 守卫
（2026-11-15 放开为解析级 fail-closed：全卡可选，仅索引解析不到物理卡时具名拒绝，
非授权卡改建议级警告，见 `docs/experimental/distributed-training-card.md`）。从 WebUI 发起 PP 训练的配置
面已可用；真实大模型权重的端到端训练仍未验收。
