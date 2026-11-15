# Anima 主训练 PP 接线需求契约（trainer-level wiring，CLI + WebUI 全程）

状态：需求冻结 v1（2026-11-15，本轮只读实读源码定稿，未做任何实现改动）
适用版本：分支 `codex/moss-pp-v0-20261112`，基线 = `6610f710` + 全部未提交 MOSS 接线现场（本文以工作树实时源码为准，不洗 MOSS 契约原文）
入口命令：见 §7/§8；普通非 PP 训练入口与行为完全不变
相关代码：`library/training/pp/runtime.py`（Anima 库级 runtime，已存在）、
`library/training/pp/{trainer_admission,trainer_inputs,trainer_resume,trainer_session}.py`（MOSS trainer 四模块，本轮泛化为族适配器）、
`library/training/pp/gate.py`、`library/training/pp/{contract,schedule,transactions,checkpoint}.py`（共享基建，不改语义）、
`library/training/train_session.py`（冻结分叉点）、
`library/models/family_registry.py`、`library/anima/pipeline_parallel.py`、`library/training/noise_target.py`
上位契约：`docs/proposal/moss_image_pp_trainer_wiring.md`（MOSS v1/v2/v3 修订账，本文逐条继承其已验收语义）

一句话：把已验收的 MOSS trainer 级 PP 接线（`train_session` 分叉 → 专用 PP 训练会话 →
共享 pp 基建）推广到 Anima 族：复用 `build_anima_pipeline_runtime`（库级 runtime 已存在，
不重写），把 MOSS 专属的 trainer 四模块泛化为族适配器，admission 增族 Anima，
fail-closed 边界不变；阶段二做 WebUI（TOML schema / web services 字段校验 / 队列）。

## 0. 范围与基线

- 阶段一（CLI 主训练）：Anima trainer 级 PP 接线。只改 `library/training/pp/` 内的
  trainer 层与 gate/admission、`train_session` 分叉点内部条件（分叉点位置不动）、
  必要的 family registry 元数据；共享基建（`contract/schedule/transactions/checkpoint/
  checkpoint_state/checkpoint_artifacts/placement/layout/stages/numerics/conditioning/errors`
  与 `build_anima_pipeline_runtime` 本体）语义不改。
- 阶段二（WebUI）：TOML schema/compat、`web/services/config/form_metadata.py` 字段、
  `web/services/training/*` 字段校验与队列（`resume_pipeline_gate.py` 的
  `model_family`/`pipeline_parallel` 键已存在，见 §6）。
- 非 PP 路径、其他族（krea2_raw / z_image / qwen_image_2_1）、MOSS 已验收行为：零改动，
  继续具名拒绝（`pp_trainer_path_unavailable`）。
- 禁项继承 MOSS 契约：无 fetch/pull/rebase/stash/commit/push/部署；不下载真大模型；
  保护未提交现场与原 krea2-webui 工作区。

## 1. 只读现状事实（实读源码，2026-11-15）

1. **Anima 是唯一已有 `runtime_backend=1F1B` 注册的族**：`family_registry.py:88-117`
   `pipeline_parallel=PipelineParallelFamilySpec(block_container="blocks",
   supported_stages={2}, supported_schedules={"1f1b"}, supported_splits={"balanced"},
   runtime_backend=PIPELINE_PARALLEL_RUNTIME_BACKEND_1F1B)`；krea2_raw/z_image/qwen 的
   spec 无 1F1B backend。
2. **Anima 库级 PP runtime 已存在且被共享基建引用**：
   `pp/runtime.py::build_anima_pipeline_runtime / AnimaPipelineRuntime`。
   - wire：单一 latent shape（`_require_shape` 一个 PP run 只服务一个 token bucket，
     `pp_token_bucket_mismatch` 具名拒绝），5D `(B,C,T=1,H,W)`，boundary hidden shape
     由 `layout.boundary_hidden_shape(latent_mb, patch_spatial, patch_temporal,
     model_channels)` 推导。
   - step 入参：`timesteps / crossattn_emb / target / padding_mask / fps / h_offset /
     w_offset / losses`；`fps/h_offset/w_offset` 是 Anima 可选 kwargs（MOSS wire 没有）。
   - 调度：`schedule.build_anima_schedule`（Anima 专用 schedule 工厂，microbatch 归一化
     在 last-stage loss 一处）；fp16 事务 `PpFp16Transaction`（overflow 全 rank 一致
     skip、scale broadcast）与 MOSS 共享同一 `transactions.py`。
   - builder 硬约束（已实现，继承）：`dtype` 固定 `use_fp32` 策略、CUDA 必须
     `cpu:gloo,cuda:nccl` 多后端组、fp16 必须 fp32 master weights。
3. **stage 切分**：`library/anima/pipeline_parallel.py::AnimaBlockStage(BorrowedBlockStage)`
   复用共享 `build_pipeline_block_stage` 的借块协议（`global_state_dict_key` /
   `state_dict_key_map`），Anima 28/40 block 两种拓扑均已登记（`known_num_blocks`）。
4. **trainer 级接线目前只对 MOSS 开**：`pp/gate.py:205-217`
   `TRAINER_PP_WIRED_FAMILIES = frozenset({"moss_image"})`；`train_session._run_training_session`
   的分叉点（`pipeline_parallel_enabled(args)` → `run_pp_trainer_session`）已冻结且族无关
   ——本任务不需要移动分叉点，只需要让它能服务 Anima。
5. **MOSS trainer 四模块的 MOSS 专属点**（泛化对象）：
   - `trainer_inputs.MossStepInputs`：`noisy_input/sigmas/timesteps/target/text_hiddens/
     text_mask/batch_digest`；条件从 `moss_image.family._prepare_text_conditions` 取。
   - `trainer_admission.require_moss_trainer_admission`：名内硬编码 MOSS 语义
     （cache knobs、loss、accumulation、preview 拒绝表）。
   - `trainer_session.build_moss_trainer_runtime`（调 `build_moss_pipeline_runtime`）、
     `derive_run_id(args, family="moss_image")`（family 已参数化）、
     `run_identity`、`pp_train_step`（调 `runtime.step` 的 MOSS 形参面）。
   - `trainer_resume`：`TRAINER_RESUME_FORMAT = "moss_image_pp_trainer_resume"`（字符串
     格式戳），`family` 已是显式参数；格式戳需按族区分。
6. **Anima 条件链（非 PP 真相）**：`library/training/noise_target.py::compute_noise_pred_and_target`
   —— timesteps 由 sampler registry（`get_timesteps_sigmas`，`timestep_sampling` 配置）
   采样，`[0,1]` float32；`crossattn_emb` 来自 text conditioning 链（Anima TE 必须
   max-padded，padding 位置是 cross-attention attention sink，**不可裁剪、不可 mask 掉**）；
   `padding_mask` 按 `(bs,h,w,dtype,device)` 缓存的全零张量；FEI/content routing 在
   `router_source != "none"` 时每步设置。
7. **WebUI 现状**：`web/services/config/form_metadata.py:241-258,447-451` 已有
   `pipeline_parallel{,_stages,_microbatches,_schedule,_split}` 五字段；TOML 层
   `library/models/pipeline_parallel.py::validate_pipeline_parallel_config`（族 spec 驱动）；
   `web/services/training/resume_pipeline_gate.py` 已把 `model_family`/`pipeline_parallel`
   列入 resume 键；compat 审计已有 `pipeline_parallel_config` 硬 code。
   `tests/test_krea2_pipeline_parallel_webui.py` 是 WebUI 侧既有先例。
8. **Anima PP 已有定向测试**：`tests/test_anima_pp_wiring.py`（库级接线）、
   `tests/test_training_pp_runtime.py`、`tests/test_training_pp_checkpoint.py`；
   MOSS trainer 侧先例：`tests/test_moss_image_pp_trainer*.py`（hot 固定 nodeid
   `test_two_rank_cuda_trainer_resume`）。

## 2. 族适配器协议（wire shape / 条件格式 / forward / 调度差异的参数化）

四个 trainer 模块不改名、不复制执行器；MOSS 分支保持现行为（等价重构级差异可接受，
但语义差分必须为零）。族差异全部收敛到一个显式的 per-family 适配器表（建议
`pp/trainer_adapters.py` 新模块或各 trainer 模块内的注册 dict，实现任务按此冻结）：

| 维度 | anima | moss_image（现状，冻结不动） |
| --- | --- | --- |
| runtime builder | `pp.runtime.build_anima_pipeline_runtime` | `pp.moss_runtime.build_moss_pipeline_runtime` |
| wire latents | 5D `(B,C,1,H,W)`，单 bucket | 4D，单 bucket |
| wire 条件 | `timesteps[B]`(float32,[0,1]) + `crossattn_emb[B,T,C]`(max-padded) + `padding_mask` 全零 + 可选 `fps/h_offset/w_offset`(v1 全 0/None) | `timesteps` + `text_hiddens` + `text_mask` |
| 条件来源 | Anima text conditioning 链的 disk cache（`_anima_te.safetensors`，schema `anima_te_v1`），max-padded 原样进 wire | `moss_image.family._prepare_text_conditions` |
| 噪声/时间采样 | rank0 用 `noise_target` 同一 sampler registry 数学采 `timesteps+noise`，broadcast；target = 共享 noise_target 的 flow-matching target（与非 PP 位级同数学） | rank0 `sample_moss_flow`，target=noise−latents |
| step 形参面 | `runtime.step(latents, timesteps=…, crossattn_emb=…, target=…, padding_mask=…)` | `runtime.step(…, text_hiddens=…, text_mask=…)` |
| stage 切分 | `AnimaBlockStage` / 借块协议，28 或 40 blocks | MOSS 原生 wire |
| schedule | `build_anima_schedule` | moss schedule |
| run identity | `run_identity = sha256(family\|seed\|output_name)`，family=`anima` | 同构 |
| resume 格式戳 | `anima_pp_trainer_resume`（新） | `moss_image_pp_trainer_resume` |

适配器协议最小接口（实现必须覆盖，命名可调但面不变）：

1. `builder(args, accelerator, network, unet, *, latent_shape, …) -> runtime`
   —— Anima 分支调 `build_anima_pipeline_runtime`，`optimizer_factory` 复用 MOSS 的
   per-stage `get_optimizer` 模式；scaler 仍只在 last stage。
2. `prepare_step_conditions(trainer, args, batch, *, accelerator, weight_dtype) -> 条件 bundle`
   —— Anima 返回 `(timesteps, crossattn_emb, padding_mask)`；batch 无缓存 TE 输出 =
   具名拒绝（与 MOSS 同 code，文案点名 Anima 缓存）。
3. `sample_noise_and_target(clean_latents, …) -> (noisy_input, timesteps, target)`
   —— 必须调用与非 PP 完全相同的 family 数学（Anima：`noise_target` 用的 sampler
   registry 路径）；禁止在 PP 分支重写第二套噪声数学。
4. `step_inputs -> runtime.step kwargs` 映射 + 全 batch SHA-256 digest 对账
   （`trainer_inputs` 的强摘要/all-gather 语义族无关，直接复用）。

不变量（继承 MOSS v2-I/v3 语义，族无关）：
- 两 stage 消费同一 logical batch（loader 不做 DistributedSampler 分片、不
  `accelerator.prepare` loader）与同一 noise/timesteps（rank0 采样 + broadcast + digest 对账）。
- batch prep 失败先 `all_gather_object` 协商，单 rank 失败双方同码具名拒绝。
- `global_step` = 消耗的 pipeline step（含 skip）；`effective_step` 与 LR scheduler 只在
  成功步前进；fp16 事务全 rank 一致 skip。
- crossattn_emb 进 wire 前不裁剪 padding（Anima attention-sink 不变量，任何"按真实长度
  裁剪"的优化在 PP 路径同样禁止）。

## 3. Admission 增族边界（fail-closed）

- `pp/gate.py::TRAINER_PP_WIRED_FAMILIES` 增 `"anima"`（1F1B backend 判定逻辑不变；
  其他族仍 `pp_trainer_path_unavailable`，DDP-interior `require_trainer_topology`
  guard 语义不降级）。
- admission 泛化：`require_moss_trainer_admission` 抽出族无关拒绝 + 族专属拒绝表。
  Anima 放行形状（v1）：`model_family=anima` / stages=2 / schedule=1f1b / split=balanced /
  单 token bucket 缓存数据集 / fp16 mixed precision + fp32 master / plain 或已注册 LoRA
  networks（`network_train_unet_only`）/ 完整 `cache_latents` + `cache_text_encoder_outputs`
  / `gradient_accumulation_steps=1` / 标准 flow-matching MSE。
- Anima 专属具名拒绝（`pp_config_invalid`，逐条点名 knob，不做静默忽略）：
  - `compile_seq_bands` / `compile_dynamic_seq` / compile 开启（PP stage 内 compile 未接线）；
  - FEI / router 条件（`router_source != "none"` 或 `use_moe_style` 路由，PP 跨 stage
    未接线 FEI 每步状态）；
  - method adapters（EasyControl/IP-Adapter 等 `on_network_built` 附加物，PP 分叉点之后
    的 setup 未覆盖）；
  - timestep sampling 之外的 preview / training preview / validation 启用（继承 MOSS v2-J）；
  - 多 bucket / 非 constant bucket 数据集（wire 单 bucket 硬约束）；
  - `text_encoder training`（TE 属 stage0 外的 encoder，PP v0 只训 DiT LoRA）；
  - MOSS 表中其余项（accumulation、preview、非缓存、非标准 loss、stage_schedule、
    adaptive OOM、epoch 级保存计划、persistent workers）全部沿用。
- 准入失败信息必须含 family/runtime_backend 与"refusing to fall back to DDP"文案
  （现 gate 文案模板直接复用）。

## 4. 恢复语义

- PP resume 包结构族无关：每 rank `pp_stage-rank{r}.pt` + 每 rank
  `trainer_state_rank{r}.pt` + `pp_checkpoint.json` manifest；save/read→validate
  （version/family/run identity/step）→apply 顺序、幂等跳过、拒绝覆盖——全部沿用共享
  `pp.checkpoint` / `trainer_resume` 协议。
- `trainer_resume.TRAINER_RESUME_FORMAT` 按族实例化：Anima 用 `anima_pp_trainer_resume`，
  校验 `payload.family == "anima"`；跨族包（anima 读 moss 戳、反之）= 具名拒绝。
- 旧库级包（无 trainer 元数据）默认拒绝精确 resume，仅显式 weights-only/nonexact
  请求才降级（MOSS v3-F 继承）。
- run identity 三字段 `family|seed|output_name` 的 sha256 全 digest，跨输出目录存活；
  Anima 的 `derive_run_id(args, family="anima")`。

## 5. 阶段二 WebUI 冻结范围

- TOML/CLI：`pipeline_parallel*` 五键族 spec 校验已由 `validate_pipeline_parallel_config`
  承担；本阶段补 Anima 侧 schema/compat 一致性——`anima` 的 stages=2/1f1b/balanced 从
  registry 读取（不新增白名单副本）；`compat_unverified_policy` 不新增 Anima 降级面
  （`pipeline_parallel_config` 恒 error code 不动，新接线验收后是否降级另立审计）。
- web services：`form_metadata.py` 五字段族无关已存在；需要补的是
  Anima PP 专属字段的表单/校验可见性（如 timestep_sampling 在 PP 下仍合法但 preview
  类字段必须拒绝），以及 `training_service` / `resume_pipeline_gate.py` 对 Anima PP run
  的 `model_family` 传递正确性。
- 队列：PP run 的排队/resume 走现有队列；queue item 的 `pipeline_parallel` 与
  `model_family` 键必须成对进入 runtime config，不允许只传其一；非 PP/其他族行为零改动。
- 前端（frontend-next）本阶段只做字段透传与文案，不新增 PP 专属工作台功能。

## 6. 热测试入口与预算（GPU 优先，双 V100 tiny）

- 入口：新增 `tests/test_anima_pp_trainer_hot.py`，固定 nodeid
  `test_two_rank_cuda_trainer_resume`（与 MOSS hot 同构：真实生产入口
  `train_session` → `run_pp_trainer_session` → `build_anima_pipeline_runtime`，
  无 mock；straight + resume 两段）。
- 模型：tiny Anima（`library/anima/mini_forward.py` / 小配置 DiT，仓库内构造，不下载）；
  tiny 数据集 fixture（单 bucket，完整磁盘 cache）。
- GPU 边界：只允许 UUID `GPU-b4b014b2-953b-28e8-b8c1-425c4a401165` 与
  `GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061`（两张 V100）；**GTX 1050 严禁使用**
  （会卡死）；启动前 `gpu-gate.sh --count 2 --json` 确认两卡 free/usable。
- 预算：单轮 `timeout 60`（与 MOSS hot 一致；MOSS 实测 33.35s，Anima tiny 同量级）；
  allocator 有界（`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 类有界配置可按需），
  超时/失败可多轮授权重跑（用户已授权任意有界热测试），但每轮失败必须先记录证据再重跑，
  禁止无限循环。CPU/Gloo 双 rank 冷测先过（`CUDA_VISIBLE_DEVICES=''`），再上 CUDA。
- 验收口径：straight 段 loss 有限、无具名拒绝；resume 段 scheduler/RNG/数据游标/run
  identity 逐字段恢复、续训 step 正常；非 PP Anima 回归（`tests/test_anima_pp_wiring.py`
  及既有定向组）全绿。

## 7. 启动命令（阶段一热测）

```bash
# 双 CUDA tiny Anima hot（仅 GPU 验收任务执行；先 gpu-gate）
/mnt/SD_Dataset/训练器相关/dsh-trainer-preset/gpu-gate.sh --count 2 --json
env CUDA_VISIBLE_DEVICES='GPU-b4b014b2-953b-28e8-b8c1-425c4a401165,GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061' \
  PYTHONPATH=. PYTHONUNBUFFERED=1 timeout 60 .venv/bin/python -u -m pytest -q -rs -s \
  tests/test_anima_pp_trainer_hot.py::test_two_rank_cuda_trainer_resume

# CPU 定向组（阶段一验收）
timeout 300 .venv/bin/python -m pytest -q -rs tests/test_anima_pp_trainer.py tests/test_anima_pp_trainer_checkpoint.py
timeout 300 .venv/bin/python -m pytest -q -rs tests/test_anima_pp_wiring.py tests/test_training_pp_runtime.py tests/test_training_pp_checkpoint.py tests/test_pp_import_boundary.py
```

（实现/验收任务可在此契约上细化文件名，但 hot nodeid、GPU 边界、timeout 预算为冻结项。）

## 8. v1 冻结记录

- v1（2026-11-15，本文档初版）：只读实读 `pp/` 21 模块、`family_registry.py`、
  `library/anima/pipeline_parallel.py`、`noise_target.py`、`train_session.py` 分叉点、
  `gate.py` wired-family 判定、WebUI `form_metadata.py`/`resume_pipeline_gate.py` 后冻结。
  关键判定：①Anima 库级 runtime 已存在（`build_anima_pipeline_runtime`），本任务**不重写
  runtime，只做 trainer 层族适配**；②分叉点已族无关、位置不动；③MOSS 四模块的 MOSS
  专属点收敛为 §2 适配器表，MOSS 行为零改动；④admission 增 Anima + Anima 专属具名拒绝表；
  ⑤resume 格式戳按族实例化；⑥WebUI 五字段已存在，阶段二为一致性补齐而非新功能。
