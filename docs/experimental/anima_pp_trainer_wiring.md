# Anima 主训练 PP 接线（trainer-level wiring）

状态：实验（CPU/Gloo 双 rank 与双 V100 CUDA tiny 热测均已独立验收通过——
`tests/test_anima_pp_trainer_hot.py::test_two_rank_cuda_trainer_resume`
run14/15/16 rc=0（≈11.1s，straight+resume 参数 digest 全等），真实生产入口、
无 mock；真实大模型权重训练未验收）
适用版本：分支 `codex/moss-pp-v0-20261112`（基线 HEAD `6610f710` + 未提交接线成果）
入口命令：见下文「启动」；普通非 PP 训练入口与行为完全不变
相关代码：`library/training/pp/trainer_adapters.py`（族适配器表，本轮新增）、
`library/training/pp/trainer_session.py`（族盲派发）、
`library/training/pp/trainer_admission.py`（族准入）、
`library/training/pp/trainer_inputs.py`（族盲 batch/噪声同步引擎）、
`library/training/pp/trainer_resume.py`（按族格式戳）、
`library/training/train_session.py`（冻结分叉点）、`library/training/cli_entry.py`
需求契约：`docs/proposal/anima_pp_trainer_wiring.md`（t1 冻结原文）
上位文档：[moss_image_pp_trainer_wiring.md](moss_image_pp_trainer_wiring.md)
（MOSS 原始接线）、[anima_pipeline_parallel_v0.md](anima_pipeline_parallel_v0.md)
（库级 runtime）

一句话：把已验收的 MOSS trainer 级 PP 接线经**族适配器**推广到 Anima——
不重写 runtime，`build_anima_trainer_runtime` 薄壳复用库级
`build_anima_pipeline_runtime`（5D wire、`build_anima_schedule`、共享 fp16
事务与 checkpoint 协议）；`train_session` 分叉点位置不动，派发按族路由。

## 族适配器（差异收敛面）

四个 trainer 模块保持族盲；MOSS 与 Anima 的全部差异收敛在
`trainer_adapters.py` 的显式适配器表（两实现同接口，无第二执行器）：

| 维度 | anima | moss_image（冻结不动） |
| --- | --- | --- |
| runtime builder | `build_anima_trainer_runtime` → 共享 `build_anima_pipeline_runtime` | `build_moss_trainer_runtime` → `moss_runtime` |
| wire latents | 4D cache 显式 `unsqueeze(2)` → 5D `(B,C,1,H,W)` 单 bucket | 4D |
| 条件 | 缓存 max-padded `crossattn_emb` **原样进 wire（attention-sink 不裁剪）** + 全零 `padding_mask`；每 rank 各自读缓存（local conditioning） | `family._prepare_text_conditions`（仅 rank0 准备） |
| 噪声/时间 | rank0 用非 PP 同一 `SAMPLER_REGISTRY` 数学（`noise_target` 同源），广播 `noise(4D)/timesteps`；target=4D `noise-latents` | rank0 `sample_moss_flow`，广播 `noise/sigmas` |
| step 形参面 | `timesteps / crossattn_emb / padding_mask` | `timesteps / text_hiddens / text_mask` |
| resume 格式戳 | `anima_pp_trainer_resume` | `moss_image_pp_trainer_resume`（历史常量不变） |
| run identity | `sha256(family\|seed\|output_name)`，family=`anima` | 同构 |

## 放行范围（fail-closed，全部具名拒绝）

共享 MOSS-v0 拒绝表逐条继承（accumulation=1、无 preview、完整磁盘缓存、
标准 MSE、fp16+GradScaler、无 stage_schedule/adaptive OOM/epoch 级保存/
persistent workers/torch_compile），加 Anima 专属具名拒绝（`pp_config_invalid`）：
`compile_seq_bands`/`compile_dynamic_seq`、FEI/router 条件（`router_source`≠
`none`、`use_moe_style`）、method adapters（`use_ip_adapter`/`use_easycontrol`/
`use_byg`）、caption dropout、`ip_noise_gamma`（last-stage 重建算术前提）、
`network_train_unet_only=true`。单 token bucket 硬约束由 runtime 的
`pp_token_bucket_mismatch` 承担；`pp_trainer_path_unavailable` 仍挡 krea2_raw /
z_image / qwen_image_2_1 与一切 DDP 内部误路由。

## WebUI/配置链

* compat 预检按族感知：anima 合法 PP 放行并提示「专用 PP 训练会话」；
  训练准入的具名拒绝（preview/compile/router/缓存等）在 WebUI 预检即以
  `pp_config_invalid` 等 warning 显形。
* 队列/runtime config：`pipeline_parallel` 与 `model_family` 必须成对进入
  （启动链 `launcher_start` 与续训 gate 双侧强制，非 PP 零改动）。
* gpu_whitelist 守卫（2026-11-15 放开）：GPU 全卡可选，唯一 fail-closed 条件是
  显式选卡索引在 nvidia-smi 中解析不到物理卡（越界/解析失败在 enqueue/start
  前具名拒绝 `gpu_whitelist_unauthorized_gpu`）；曾经的 UUID 级授权集与
  GTX 1050 具名拒绝已移除，不适配卡改由分布式诊断端点给出建议级警告。
  空白名单默认路径零改动。

## 启动

```bash
# CPU/Gloo 冷测（先过，再上 CUDA）
CUDA_VISIBLE_DEVICES='' WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29521 \
  accelerate launch --num_processes 2 --cpu train.py --config <anima-pp-config.toml>

# 双 V100 生产形态（每卡一个 rank；热测当时按双 V100 UUID 选卡）
CUDA_VISIBLE_DEVICES='GPU-b4b014b2-953b-28e8-b8c1-425c4a401165,GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061' \
  accelerate launch --num_processes 2 --multi_gpu train.py --config <anima-pp-config.toml>
```

配置要点：`model_family="anima"` + `pipeline_parallel=true` / stages=2 /
microbatches≥2 / `1f1b` / `balanced` / `mixed_precision="fp16"` /
`network_train_unet_only=true` / 完整 latents+TE 缓存（含 `crossattn_emb`）。
续训把 `resume` 指向 `pp_runs/<run>-step<N>` 包根目录。

## 测试与验收面

* `tests/test_anima_pp_trainer.py`（34）：准入全表、5D wire 边界、
  untrimmed padding、按族格式戳、真 2-rank Gloo 步进语义与 digest/timesteps 对账。
* `tests/test_anima_pp_trainer_checkpoint.py`（5）：真双 rank 真会话入口
  straight(6) vs split(4)+resume(2) 参数 digest 全等、`anima_pp_trainer_resume`
  戳、跨族 payload 具名拒绝。
* `tests/test_anima_pipeline_parallel_webui.py`（9）：预检接入/准入显形/
  成对键/续训 gate/gpu UUID 守卫。
* 热证据：`docs/findings/anima_pp_trainer_wiring_cuda_hot_run1-16.log`
  （run14/15/16 rc=0，straight+resume digest 全等；GPU 边界全程仅双 V100）。

## 当前边界

真实大模型权重训练未验收；真实质量/性能数字不外推。共享基建
（`contract/schedule/transactions/checkpoint` 与 `build_anima_pipeline_runtime`）
语义未改；MOSS 行为零回归（MOSS trainer 套件全绿）。
