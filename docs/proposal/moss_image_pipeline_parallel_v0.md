# MOSS-Image Pipeline Parallel v0：复用与验收契约（需求冻结）

状态：需求冻结（v0 round 1，本文档只冻结契约，不包含实现）
适用版本：分支 `codex/moss-pp-v0-20261112`，基线 `origin/dev` `6610f710`
入口命令：见 §8（CPU/Gloo 定向用例）；独立 CUDA tiny 归后续 verification 任务
相关代码：`library/training/pp/`、`library/models/moss_image/`、`library/models/family_registry.py`、`library/training/moss_compat.py`、`library/models/pipeline_parallel.py`

---

## 0. 结论摘要（冻结项）

1. **首轮范围**：仅 `model_family=moss_image` 的**库级** PP=2 / balanced 切分 / `Schedule1F1B` / plain fp16 + GradScaler / stage-local plain LoRA。不含大模型下载、真实权重长训练、正式训练循环接线、其他族放行、提交推送。
2. **主训练 fail-closed 不变**：`library.training.pp.gate.require_trainer_topology` 与 `require_accelerate_checkpoint` 的拒绝路径在 MOSS 下继续生效；`moss_image` 未注册 stage-local runtime backend 之前，`pipeline_parallel_runtime_issue` 继续对一切 family 报 `pipeline_parallel_runtime_unavailable`。
3. **复用现有 `family_registry` + 最小 family hooks**：不新建 layout registry。MOSS 的 PP 能力入口是给 `moss_image` 填 `ModelFamilySpec.pipeline_parallel = PipelineParallelFamilySpec(...)`，配一个 MOSS 专属的 stage 构造 hook（替代 `library.anima.pipeline_parallel.build_anima_block_stage` 这一个注入点）；`library/training/pp/{contract,gate,transactions,checkpoint*,schedule 编排骨架}` 按 §4 复用，不复制改写。
4. **MOSS 形状契约是 4D**：`[B,128,H,W]` packed latent（`moss_image/latent.py`）、`layers` 主层（42）、文本 refinement（`_TextProjection` 2 个 `_TextBlock`）、每样本真实文本长度 mask/RoPE、时间输入 `sigma*1000`。**不得套用** Anima 的 5D `(B,C,T=1,H,W)`、max-padding text encoder、`blocks`/`lora_unet_blocks_N` 命名或 4032/4200 token bucket 家族名。
5. **联合序列跨 stage 可微**：boundary 携带 image+已演化 text 的联合 hidden，text 路径梯度可微地回到第一 stage 的 text refinement/embedder；text 不是在接收端重新生成的冻结 conditioning。MOSS 的 per-sample 真实长度决定图像 RoPE（`math._RoPE.forward` 的 `text_length` 参数），transport 层 padding 不得进入任何 attention。
6. **判据不沿用 Anima 数字**：MOSS CPU 同切分参考与幅度/方向检查、双 rank restart/replay、单侧 overflow、padding 不变性等，均在 MOSS tiny fixture 上**首轮实测后冻结阈值**；禁止把 Anima 等价性实验的数值当 MOSS 通过判据；禁止"全 batch FP16 逐位相等"这类不可实现判据。

## 1. 共享 PP 接口（复用面，Anima 行为不回归）

以下 Anima 侧接口在本轮**零改动**，并作为 MOSS 复用的硬约束（现有 `tests/test_training_pp_runtime.py`、`tests/test_training_pp_checkpoint.py`、`tests/test_anima_pipeline_runtime.py`、`tests/test_pp_import_boundary.py` 必须继续全绿）：

| 接口 | 复用方式 | MOSS 侧约束 |
| --- | --- | --- |
| `pp.contract.PipelineRuntimeConfig` | 原样复用（stages/microbatches/schedule/mixed_precision） | `stages=2`、`microbatches>=stages`、`schedule="1f1b"`；`conditioning="local"` 语义按 §3 重新定义，`"wire"` 仍仅实验对照 |
| `pp.contract` 校验族 | 原样复用 | fp16-only + GradScaler；量化 base（NF4/fp8/ConvRot）具名拒绝 `pp_quantization_unsupported`；fp32 master weights `pp_fp16_master_weights_required` |
| `pp.errors` 稳定 `pp_*` code | 原样复用 | 新增 MOSS 专属 code 走同一模块（如 `pp_stage_contract_violation` 细分消息），不改既有 code 语义 |
| `pp.transactions` | **原样复用，零 latent-shape 耦合**（现状即无耦合：只碰 optimizer grads 与 CPU 标量 collective） | 不得为 MOSS 增加 shape 参数；单 scale/单 skip 协议、`agree_overflow`、`verify_skip_unanimity`、scale growth policy 广播全部照旧 |
| `pp.schedule` 编排骨架 | 复用 `build_pipeline_stage`/`build_anima_schedule`/`run_pipeline_step`/`make_last_stage_loss_fn`（loss 含 `1/n_microbatches`，`scale_grads=False`） | boundary meta 的形状计算是 Anima-5D 的，MOSS 需要族专属 boundary-meta 函数（§3），但调度/损失/grad-scale 所有权语义不变 |
| `pp.checkpoint*` 协议 | 复用 run root/manifest/content-hash/拓扑校验协议骨架 | format/version/拓扑字段为 MOSS 新命名（§6）；不读旧 Anima 格式 |
| `pp.gate` | 原样复用 | `family_runtime_backend` 依赖 registry 的 `pipeline_parallel.runtime_backend`；MOSS 注册后自动接入 |
| `pp.layout` | **不复用**（Anima 5D 专属） | MOSS 用族内 4D 校验函数；不共享 `latent_to_model_5d`/`token_family` 等 helper |

## 2. Registry 与 family hook（不造第二个 registry）

`library/models/family_registry.py` 中 `moss_image` 当前 `pipeline_parallel=None`（第 231 行），gate 因此 fail-closed——这是放行前的正确状态。v0 冻结的注册形态：

- `PipelineParallelFamilySpec`（复用现有 dataclass，不扩展字段语义）：
  `runtime_backend="moss_1f1b_stage_local"`（新常量，命名族专属；`gate.require_pipeline_runtime` 只认 registry 发布的 backend，结构上已拒绝其他族）；
  `runtime_available=False` 直至验收通过；
  `block_container="layers"`（`MOSSImageTransformer.layers`，42 个主 `_Block`）；
  `default_num_blocks=42`、`known_num_blocks={42}`（官方默认拓扑）；
  `supported_stages={2}`、`supported_schedules={"1f1b"}`、`supported_splits={"balanced"}`。
- **最小 family hooks**（这是本轮允许新增的全部族专属实现面）：
  1. stage 布局：`make_pipeline_plan(family="moss_image", ...)` 走现有 `library/models/pipeline_parallel.py` 的 balanced 计划，不新建 planner；
  2. stage 模块构造：MOSS 版 `build_<family>_stage_module`（对应 Anima 的 `placement.build_anima_stage_module` 注入点），负责 §3 的 head/tail 划分与 adapter 归属；
  3. boundary meta：MOSS 版形状函数（§3.4），替代 `pp.schedule.boundary_metas` 里的 Anima-5D 计算；
  4. adapter 归属映射（§5）。
- 显式禁止：复制 `pp.layout`/`pp.stages` 后改名的"第二个实现"；把 `library/models/pipeline_parallel.py` 的 `PIPELINE_PARALLEL_SCHEDULES=("1f1b",)` 扩成多值；给 `PipelineParallelFamilySpec` 加 MOSS-only 字段。

## 3. MOSS 专属接口契约

### 3.1 latent 与 forward 形状

- 训练 latent：4D `[B,128,H,W]`（`moss_image/latent.py::encode_moss_latents` 的 BN 归一化 packed 形态）；H、W 为 latent 分辨率（图像 `//16` 后不 pack、pack 后 `//2`）。族内校验函数拒绝非 4D、非 128 通道、奇数边（对齐 `forward_for_loss` 现有 ValueError 语义，但在 PP 侧用 `pp_shape_contract_violation` 具名报错）。
- forward：`family.forward_for_loss` 是唯一训练参考路径（`timestep=sigma*1000`、`encoder_attention_mask=mask`、内部 per-sample 循环）。PP 实现不得绕开它另造一份"参考 forward"。
- 时间：`uniform_grid_sigmas` 的 sigma 以 `*1000` 进 DiT；MOSS 的 sinusoid 在 fp64 计算、fp32 进 MLP（`dit.py` L82–85），PP stage 不得改变该精度链路。

### 3.2 联合序列与 per-sample 真实长度

MOSS 不是 Anima 的"image tokens + 冻结 cross-attn KV"结构；每个样本的序列是 `cat(video_tokens(HW), text_tokens(true_len))`，且：

- **text 实际长度参与图像 RoPE**：`_RoPE.forward(height,width,text_length)` 给 image token 的轴 1 填 `text_length`、给 text token 填序号——同一文本长度变化会改变图像 token 的 RoPE 相位。因此 text hidden（refinement 之后、refinement 之前经 `_TextProjection.proj`）必须在第一 stage 就位，RoPE 由持有真实长度的 stage 构建。
- **跨 stage 梯度**：loss 只在 last stage；boundary 上是第 k 层输出后的联合 hidden `[B, HW+S_i, dim]`（S_i 为各样本真实长度）。backward 穿过 boundary 时，text token 的梯度必须可微地回到第一 stage 的 `text_embedder`/`video_embedder` 与 LoRA（若有归属）。**禁止**把 text 当作接收端重新计算的冻结 conditioning（Anima 的 `conditioning="local"` 语义在此不适用）；MOSS 的 `conditioning` 取值保留枚举但 `"local"` 的定义改为"per-sample 真实长度 metadata 本地重建 RoPE/mod，hidden 本体走 wire"。
- **transport padding 不得进入 attention**：wire 上为满足固定容量可以 pad 到统一 token 数，但接收 stage 必须按随 wire 一起传的 per-sample 有效长度切片，重建 per-sample attention；任何 padded 位置的值不得影响 q/k/v 或 RMSNorm 统计。padding 不变性是 §8 L1 判据。

### 3.3 stage 划分（balanced，2 stage）

- head（stage 0 独有）：`video_embedder`、`text_embedder`（文本 refinement，含其 2 个 `_TextBlock`）、`rope` 构建、`time_embedding`/`shared_block_modulation`/`video_head_modulation`（per-sample modulation，可作为 metadata 或 head 输出随 wire 走，冻结量不携带梯度）；
- 主层：`layers[0:21]` stage 0、`layers[21:42]` stage 1（balanced 计划切分；43 层不可整除时按 `make_pipeline_plan` 现有规则，计划内容进 checkpoint 拓扑）；
- tail（stage 1 独有）：`final_norm_video`、`video_head_modulation` 门控应用、`final_linear_video`、unflatten 回 `[B,128,H,W]`。
- MOSS 的 `blocks_to_swap`（0..40，且强制 batch=1）与 PP 互斥：`pp_stage_contract_violation` 具名拒绝（对齐 `placement.assert_stage_capabilities` 对 Anima 的拒绝方式）。`gradient_checkpointing` 在 PP 下允许（`_Block` 已支持 non-reentrant checkpoint），但属 trainer 决策，不属于 PP 契约。

### 3.4 固定 wire 容量 vs 可变真实文本长度（具名拒绝）

`PipelineStage` 接收缓冲按构造时 metadata 冻结，因此：

- **容量定义**：每个 PP run 冻结一个 `(B_micro, H, W, token_cap)` wire 形状；`token_cap = H*W + max_text_tokens`（`max_text_tokens` 来自 MOSS text cache 的 `resolve_text_token_cap` 上限，进入 manifest 拓扑）。
- **可变长度**：每样本真实长度 `S_i` 以 int metadata（不携带梯度）随 wire 传输；`1 <= S_i <= max_text_tokens`，`S_i=0` 具名拒绝（对齐 `dit.py` "at least one valid text token"）。
- **具名拒绝**（全部 fail-closed，不得静默重切或重编译）：
  - step 内 latent 形状与构建时不符 → `pp_token_bucket_mismatch`（同 Anima 语义，MOSS 的 bucket 定义为 `(B_micro,H,W)`，文本长度变化**不**触发该错——只受 token_cap 约束）；
  - `S_i > token_cap - H*W` → `pp_stage_contract_violation`（容量不足，需重建 runtime）；
  - 尾 batch（最后一个 microbatch batch < B_micro）→ **PP v0 拒绝**：dataloader 必须保证 `train_batch_size % microbatches == 0` 且尾批等大；做不到时该 epoch 以 `pp_token_bucket_mismatch` 具名中止，而不是 pad 一个假样本；
  - batch 不可整除 microbatches → 同上具名拒绝（对齐 `layout.split_batch_shape` 语义，MOSS 侧重新实现校验）。

## 4. Adapter 归属与命名

- MOSS plain LoRA 注入点固定为 `layers.N.proj.{q,k,v,o}`（`moss_image/lora_targets.py::MOSS_IMAGE_LORA_INJECTION_POINTS`）；无 text encoder LoRA、无 AdaLN/train_adaln。
- Anima 的 `pp.placement._ADAPTER_BLOCK_RE = lora_unet_blocks_(\d+)_` **不匹配** MOSS 命名——MOSS stage 归属用 `layers\.(\d+)\.proj\.[qkvo]$` 解析全局层号，禁止改 Anima 侧正则去"兼容"。
- checkpoint 中 adapter 归属按全局 `layers.N` 记录 per-rank 拥有集合；stage-local optimizer 只收本 rank 拥有的 LoRA 参数（fp32 master，§7）。
- 排除规则沿用 `lora_targets.py` 的 `exclude_patterns`；PP 侧不得引入"把 text_embedder LoRA 摊给两 rank"之类的中间态。

## 5. Optimizer / scaler / 共享 scale 策略（冻结）

复用 `pp.transactions` 既有协议，MOSS 侧不加策略分支：

- stage-local optimizer（每 rank 只持有本 rank LoRA 参数）；last stage 独占 `GradScaler`；
- 单一 pipeline-wide scale：last stage `scale_loss`，非 last stage `divide_gradients`；growth/backoff/growth_interval 策略由 last stage 广播（`_broadcast_fp16_policy`）；
- overflow 全体 OR 后统一 skip；`verify_skip_unanimity` 兜底；
- `1/n_microbatches` 由 last-stage loss 内置，`scale_grads=False`；trainer 不得二次缩放（`gate.require_trainer_topology` 继续拒绝 accelerate 路径）。

## 6. Checkpoint 兼容与验收矩阵

### 6.1 格式

- **MOSS 新格式**：`PP_CHECKPOINT_FORMAT = "moss-image-pp-stage-checkpoint"`（独立常量），沿用 content-hash/run-root/manifest 协议骨架；manifest 拓扑字段必须含：family=`moss_image`、stages=2、schedule、microbatches、layer 计划（每 rank 的 `layers` 全局区间）、wire 形状 `(B_micro,H,W,token_cap)`、adapter 归属表（`layers.N.proj.*` → rank）、`max_text_tokens`、dtype/compute policy、fp16 scale/growth_tracker/last_skip、step。
- **Anima 旧格式兼容（负向）**：MOSS loader 读到 `anima-pp-stage-checkpoint` 必须**具名拒绝**（format mismatch），不得猜测字段；反之 Anima loader 读 MOSS 格式同样拒绝。format/version 校验先行于任何字段读取（"读前校验"）。
- **Anima 正向兼容**：本轮不改 Anima pp 路径，已有 Anima checkpoint 由现有测试保证 round-trip 不回归。

### 6.2 恢复协议（验收矩阵）

| 条目 | 判据 | 层级 |
| --- | --- | --- |
| 读前校验 | format/version/拓扑 digest/adapter 归属逐字段校验，任何不符在读 payload 前失败 | L1 |
| 跨 rank 失败协商 | save 时 manifest 原子写 + 失败标记（`_mark_failed` 语义）；任一 rank 校验失败全体 abort，不出现"单 rank 恢复成功" | L1 |
| 恢复后下一步一致性 | save→step→load→step：恢复后单步 loss 与不中断 run 同种子重放一致（**相对容差判据**，见 §7）；fp16 scale、growth_tracker、last_skip 精确恢复；scale 增长策略（factor/backoff/interval）恢复后与 last stage 广播值一致 | L1 |
| apply 前对称拒绝 | frozen-boundary/拓扑/adapter 归属不一致在 optimizer/apply 前由 all-gather 具名拒绝（对齐 `assert_frozen_boundary_agreement`/`assert_compute_policy_agreement`），两 rank 同报同 abort | L1 |
| 双 rank restart/replay | rank 任一先后 kill-restart 后重放结果与连续 run 一致（相对容差） | L1 |
| 单侧 overflow | 注入单 rank inf grad → 两 rank 同 skip、scale 同步 backoff、无半模型推进 | L1 |

## 7. 分层验收与容差原则

- **L0（环境/契约）**：纯 CPU，无 torch.distributed；registry gate、config 校验、形状校验、具名拒绝。本轮已执行基线见 §8。
- **L1（CPU/Gloo 行为）**：`gloo` backend、CPU tensor、tiny MOSS fixture（官方默认 42 层按比例缩小 dim/ffn 的 fixture 也允许，但**层结构必须与官方同名同序**）；同切分参考 = 生产 `MOSSImageTransformer`/`forward_for_loss` 不切分前向（**不得**用复制改写的 PP forward 当唯一参考）；比较判据为梯度幅度比与方向余弦 + loss 相对差，**阈值由 MOSS tiny 首轮实测数据冻结并记录**，禁止直接抄 Anima 等价性报告的数字。显式禁止"全 batch FP16 逐位相等"判据。
- **L2（双 CUDA tiny）**：仅后续 verification 任务；仅授权 GPU UUID `GPU-b4b014b2…,GPU-379b8c19…` 双 V100、每卡峰值 ≤2GiB、timeout 60、tiny 固定 fixture，无真实权重。所有 L2 证据须带命令、退出码、GPU UUID、峰值显存。L2 未跑不得宣称 CUDA 通过；skip/timeout 不算通过。
- **真机/真实权重长训练**：不在本轮任何层级，报告分层标注"未验证"。

## 8. 已执行基线与测试入口（冻结）

环境：共享解释器 `/mnt/SD_Dataset/训练器相关/krea2-webui/.venv/bin/python`（只读复用），`PYTHONPATH=/mnt/SD_Dataset/训练器相关/krea2-pp-moss`，`CUDA_VISIBLE_DEVICES=''`。

- torch 版本：`2.10.0+cu129`；`torch.distributed.pipelining` 的 `PipelineStage/Schedule1F1B/ScheduleGPipe` 可导入。注释与实现的差异已核对：`pp.contract` 注释所说的"torch 2.10 对 GPipe 也要求 microbatches>=stages"与 `schedule.build_anima_schedule` 的 `PipelineScheduleSingle.__init__` 行为一致，非过时注释。
- 已执行定向基线（命令与退出码在完成报告附实际结果）：
  ```
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=<root> timeout 300 <venv>/bin/python -m pytest \
    tests/test_training_pp_runtime.py tests/test_training_pp_checkpoint.py \
    tests/test_anima_pipeline_runtime.py tests/test_moss_image_family.py \
    tests/test_moss_image_token_tiers.py tests/test_moss_image_fp16_amp.py \
    tests/test_pp_import_boundary.py tests/test_pipeline_parallel_multimodel.py -q
  ```
- **测试入口冻结（文件名以阶段 2 (t2) 声明为准）**：`tests/test_moss_image_pp_runtime.py`、`tests/test_moss_image_pp_checkpoint.py`、`tests/test_moss_image_pp_wiring.py`、`tests/moss_pp_test_support.py`（共享 fixture/支持代码）。上述新增文件不存在于基线，由实现轮创建；本文档冻结其入口与职责划分：
  - runtime：L0/L1 gate、4D 形状、wire 容量与具名拒绝、联合序列可微性、padding 不变性、同切分幅度/方向参考；
  - checkpoint：读前校验、旧 Anima 格式拒绝、跨 rank 协商、恢复后下一步一致性（含 scale 增长策略恢复）、apply 前对称拒绝、双 rank restart/replay、单侧 overflow；
  - wiring：registry `runtime_backend` 接入 gate、`moss_compat` 对 `pipeline_parallel` 键的族内放行（仅当 backend 已注册）、主训练 fail-closed 回归。
- 不跑全仓测试套件。

## 9. 合同修订记录

- v0（本文档初版）：基于实现轮初稿的 captain 纠偏并入——不造 layout registry（§2）、transactions 零 latent-shape 耦合（§1）、text 联合 hidden 可微跨 stage 且实际长度参与图像 RoPE（§3.2）、不预设 wire 必须传原始 encoder hiddens（§3.4）、测试文件名沿 t2 声明 `test_moss_image_pp_*`（§8）、恢复检查覆盖 scale 增长策略恢复/apply 前对称拒绝/旧 Anima 格式（§6.2）。
- v1（t2 执行期修订，captain 有记录 amend，2026-11-12）：
  - A（对 §2/§0.3）：不引入 `moss_1f1b_stage_local` 独立 backend 常量与 `runtime_available` 伪字段——`runtime_available` 只是 `runtime_backend != None` 的派生属性；MOSS 复用共享常量 `torch_pipelining_1f1b`，族差异由 family hook（`library/models/moss_image/pipeline_parallel.py`）承载，gate 不改判定逻辑。
  - B（对 §3.1）：packed 128ch latent 不新增奇数边拒绝；偶数约束仅适用于 pack 前 32ch（`pack_moss_latents`）。PP 侧形状校验为 4D/128ch/正整数尺寸，测试含奇数 packed 边 fixture。
  - C（对 §4）：`layers.N.proj.q` 是原始模块名，实际 `lora_name` 为 `lora_unet_layers_<N>_proj_[qkvo]`（`targeting.py` 点转下划线）；归属按真实 adapter 对象的 lora_name 解析并对已 apply 网络做 target 几何交叉校验，不套用 §4 原文的 dotted 正则。
  - D（对 §1"零改动/原样复用"）：读作兼容 API/语义不变、不做无关重写；允许的共享最小扩展为 `PpFp16Transaction.restore_policy`（增长策略恢复）与 checkpoint 格式按 family 分流。
  - E（对 §7/L1）：数值阈值不得在看过 PP 结果后适配——先按 dtype 与独立非 PP 参考噪声基线确定并记录（`tests/moss_pp_test_support.py`），再比较。
  - F（对 §6.2 双 rank restart/replay）：恢复以**新进程对整体重启**为准，不承诺存活进程组内任一 rank 随时 kill/rejoin；CUDA tiny 节点必须 fresh-pair 分相执行。
