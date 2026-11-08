# 多卡全流程参考点验收契约（DDP 优先）

状态：实验 / 验收契约（冻结版 v1），本阶段**不实现代码、不跑训练、不改依赖**
适用版本：分支 `codex/multi-gpu-training`，基线 `2987186c542b09a55c70a17c3dd523f3b11fb4d5`
入口命令：见 §15；相关代码：`library/training/loop.py`、`library/training/window_accounting.py`、
`library/training/gradient_sync.py`、`library/training/scaler_ownership.py`、
`library/training/checkpoints.py`、`library/runtime/launch.py`、`library/runtime/accelerator.py`、
`library/training/compat_matrix.py`、`library/training/bootstrap.py`
上游研究：[Lulynx 多卡对照报告](../findings/lulynx_multigpu_comparison_20261107.md)（下称 **对照报告**）、
[多卡训练实现研究计划](multi_gpu_training_plan.md)

本文把前轮研究结论转成**可执行、可判定、责任到阶段**的验收合同。凡本文与旧文档冲突，以本文为准；
凡本文结论与实时源码冲突，以源码为准并须走 §17 修订流程。

---

## 0. 一句话范围

本契约冻结四类东西：**环境与资产能不能跑**（§3、§4）、**梯度主干怎么算通过**（§6）、
**控制面/checkpoint/恢复怎么算通过**（§7–§10）、**50 次有效更新的真实短训怎么算通过**（§13）。
不冻结具体实现写法，也不批准任何生产默认值变更。

---

## 1. 证据级别与观测层级

沿用对照报告的级别定义，并按本轮需要补一层"观测层级"。

| 级别 | 含义 |
| --- | --- |
| **A** | 源码调用链闭合（本树符号级，含行号可复核） |
| **B** | 静态推断（写出所需前提） |
| **C** | 依赖第三方库内部行为或版本敏感，需实测/复核 |
| **D** | 仅有文档或人工声称，无可复核产物 |

观测层级（每个验收项必须标注它由哪一层判定）：

| 层级 | 形态 | 能证明什么 | 不能证明什么 |
| --- | --- | --- | --- |
| **V-CPU** | 单进程 CPU，`CUDA_VISIBLE_DEVICES=""` | 纯逻辑/账本/协议形状 | 任何分布式或数值行为 |
| **V-GLOO** | 2 进程 CPU/Gloo，`mp.spawn` + `file://` init | 本项目**自己的** collective 顺序、状态协议、账本对称性、故障协商 | NCCL/CUDA 行为、显存、真实吞吐 |
| **V-CUDA-SMALL** | 2 卡小模型（`MULTIGPU_REFERENCE_HARDWARE=1`） | NCCL 路径、fp16+scaler 真实分支、每 rank 设备/参数一致性 | Anima 真实数值/质量 |
| **V-REAL** | Anima 真实有界短训（`MULTIGPU_REFERENCE_HOT=1`） | 端到端可运行性与 50 更新预算内的等价性观测 | 性能结论、长训稳定性 |

**硬规则**：CPU/Gloo 一律不充当 CUDA 验收；小模型不充当 Anima 真实训练；真实短训不充当性能结论。

---

## 2. 本阶段边界（不做的事）

- 不新增/修改任何训练源码（`library/`、`networks/`、`scripts/`、`tests/`、`configs/` 全部只读）。
- 不跑 GPU 任务、不跑真实训练、不建分布式进程组。
- 不装包、不升级、不 `uv sync`、不下载模型、不做预处理、不写 `post_image_dataset/`。
- 不写 `krea2-webui`、`anima_lora`、`lulynx-trainer`（全部只读）。不动任何用户进程/队列/缓存。
- 不提交、不推送、不部署。

---

## 3. 环境契约（E）

### E-1 冻结解释器与版本 [A]

| 项 | 冻结值 | 取证方式 |
| --- | --- | --- |
| 解释器 | `/mnt/SD_Dataset/训练器相关/krea2-webui/.venv-v100/bin/python` → `/mnt/qlc/miniconda3/bin/python3.13` | `ls -l` |
| Python | `3.13.9` | `platform.python_version()` |
| torch | `2.10.0+cu129` | `torch.__version__` |
| accelerate | `1.13.0` | `accelerate.__version__` |
| pytest | `9.1.1` | `pytest.__version__` |
| transformers / diffusers / safetensors / einops / bitsandbytes / torchvision | `5.17.0` / `0.41.0` / `0.9.0-rc.0` / `0.8.2` / `0.49.2` / `0.25.0+cu129` | `__version__` |
| NCCL（torch 自带包） | `2.27.5` | `torch.cuda.nccl.version()` → `(2, 27, 5)` |

`import` 归属 [A]：研究树**没有** `.venv`，venv 内**没有** `library` / `networks` / `anima_lora` 包或
editable 安装。因此 `PYTHONPATH=<研究树>` 时 `library` / `networks` 来自研究树源码，`torch` 来自 venv
site-packages（已逐符号打印 `__file__` 核对）。

### E-2 与项目声明的漂移台账（先记清，不修） [A]

| 依赖 | `pyproject.toml` 声明 | 实际安装 | 处置 |
| --- | --- | --- | --- |
| torch | `>=2.12.0,<2.13`（第 8 行附近） | `2.10.0+cu129` | **漂移**。本契约所有结论只对 2.10.0 成立，**不外推 torch≥2.12** |
| torchvision | `>=0.27.0,<0.28` | `0.25.0+cu129` | 漂移，同上 |
| diffusers | git pin `e0118ade…` | `0.41.0`（PyPI dist-info，无 `direct_url.json`） | 漂移；本轮不使用 diffusers 行为做判定 |
| CUDA 索引 | `cu130`（`[tool.uv] extra-index-url`） | `cu129` 运行时 | 漂移；不得据 `[tool.uv]` 推断本机行为 |
| accelerate | `>=1.13.0` | `1.13.0` | 一致（下界命中） |

**规则**：凡结论依赖第三方库内部行为的项一律标 **C**，且必须在 2.10.0/1.13.0 上复核；
升级 torch/accelerate 属于环境变更，须 captain 同意后另开任务，不得在本任务链内顺手做。

### E-3 冻结的验证环境变量 [A]

```bash
export PY=/mnt/SD_Dataset/训练器相关/krea2-webui/.venv-v100/bin/python
export R=/mnt/SD_Dataset/训练器相关/krea2-multigpu
# 纯 CPU 检查一律关闭可见 GPU，避免与用户/其他会话争抢
CUDA_VISIBLE_DEVICES= PYTHONPATH=$R timeout 60 $PY -m pytest tests/<target>.py -q
```

GPU 层级（V-CUDA-SMALL / V-REAL）**必须**显式 `MULTIGPU_REFERENCE_HARDWARE=1` 或
`MULTIGPU_REFERENCE_HOT=1`，并由**唯一 GPU 执行者**运行（§12、§17）。GPU 执行许可来自
**已批准的团队计划本身**：显式环境变量 + §3 E-5 前置 + §5 D-5 磁盘门即视为受控执行，
**不另设二次审批门**。默认（未设环境变量）一律不占 GPU。

### E-4 已执行的 CPU 定向基线（本轮实测，作为回归地板）

| # | 命令（均已带 `CUDA_VISIBLE_DEVICES=`、`PYTHONPATH=$R`） | 结果 |
| --- | --- | --- |
| 1 | `timeout 60 $PY -m pytest tests/test_window_replay.py -q` | **14 passed** in 5.76s |
| 2 | `timeout 120 $PY -m pytest tests/test_distributed_gradient_sync.py -q` | **10 passed** in 12.44s（含 2 进程 Gloo 用例） |
| 3 | `timeout 120 $PY -m pytest tests/test_distributed_sampling.py -q` | **7 passed** in 25.67s（含 2 进程 Gloo 用例） |
| 4 | `timeout 120 $PY -m pytest tests/test_fp16_nan_static_equivalence.py -q` | **24 passed** in 8.64s |
| 5 | `timeout 120 $PY -m pytest tests/test_training_checkpointing.py -q` | **55 passed** in 7.00s |

合计 **110 passed**，无失败、无跳过导致的虚假通过。
说明：第 2/3/4/5 项用了 120s 而不是 60s，因为 2 进程用例有 `mp.spawn` 启动开销；单进程的
第 1 项严格按 60s 执行。**这些数字是本契约的回归地板**：后续实现若让任一文件变红即为回归。

**可复现性（requirements attempt 6 独立复跑）**：同一组命令再次执行结果**完全一致** ——
`14 / 10 / 7 / 24 / 55` = **110 passed**（5.64s / 12.23s / 25.50s / 8.40s / 6.54s），
无 skip、无 xfail、`CUDA_VISIBLE_DEVICES=` 为空（全程无 GPU 参与）。因此该地板**有两次独立观测**。

额外一次 2 进程 Gloo 语义探针（本轮新写、临时脚本在 `/tmp`，不进仓库；配方见 §7.2）：

| # | 目的 | 结果 |
| --- | --- | --- |
| 6 | 判定 `accelerator.reduce(reduction="max")` 的真实语义（两 rank flag 均 1.0） | `sum=2.0`、`mean=1.0`、**`max=2.0`** → `"max"` 实为 SUM（见 §7.1 N-3） |

### E-5 CUDA 准入方案（不执行，仅冻结条件）

真实 GPU 放行前必须同时满足：

1. `nvidia-smi` 只读复核：2×Tesla V100-SXM2-16GB 均空闲（本轮采样时均为 `0 %` / 已用 ≤17 MiB）。
2. 无用户训练/队列/探针进程占用目标卡；不得终止任何非本任务进程。若被占用 → 报告 captain，不排队、不抢卡。
3. `MULTIGPU_REFERENCE_HARDWARE=1` / `MULTIGPU_REFERENCE_HOT=1` 由执行者显式设置，默认不设。
4. 磁盘门通过（§5 D-5）。
5. 环境仍是 E-1 冻结组合；若解释器或版本已变，先回到 E-2 重新登记漂移，禁止"就地适配"。

### E-6 精度口径 [A]

`library/training/precision_policy.py:45-52`：SM 主版本 `<8` 且 `--mixed_precision=bf16` 时**自动改写为
fp16` 并告警；`library/runtime/launch.py:37-46` 的 `bf16` 默认只是 launcher 侧提示。
V100（SM70）上的真实短训因此**实际走 fp16 + `GradScaler`**，`library/runtime/accelerator.py:159-165`
显式传 `--mixed_precision`。旧 fp16 文档中的过期注释不作为当前行为依据。

> 由此得出一条契约级联动：**V-REAL 环境天然覆盖 P0-0 的 scaler 分支**（只要真实运行出现一次 skip）。
> 若整轮真实运行零 skip，则 scaler 分支**未被真实运行覆盖**，只能用 V-GLOO 证据判定，不得宣称"真实运行验证过"。

---

## 4. 资产契约（A）

### A-1 Anima 基座与配套权重 [A]

路径取自 `krea2-webui` 外部设置文件（只读，未读取任何凭据字段）：

| 角色 | 路径 | 大小 |
| --- | --- | --- |
| DiT 基座 | `/mnt/SD_Dataset/训练器测试/comfyui/models/diffusion_models/anima/anima-base-v1.0.safetensors` | 4,182,218,328 B |
| Text encoder | `/mnt/SD_Dataset/训练器相关/anima_lora/models/text_encoders/qwen_3_06b_base.safetensors` | 1,192,135,096 B |
| VAE | `/mnt/SD_Dataset/训练器相关/krea2-webui/models/vae/qwen_image_vae.safetensors` | 253,806,246 B |

三者均为可直接被 `train.py` 消费的单文件 safetensors（与 `configs/web-ui-settings.toml`
及外部 `krea2-configs/web-ui-settings.toml` 的 anima 分节一致）。**无需下载、无需转换**。

### A-2 latent cache：可用 [A]

`krea2-webui/post_image_dataset/lora/`：60 个 `*_{WxH}_anima.npz`（VAE latent），
对应 `post_image_dataset/resized/` 的 60 张图 + 60 个 caption。bucket 分布（按像素尺寸）：
`0896x1200`×24、`0672x1536`×15、`1344x0768`×9、`1008x1024`×3、`1200x0896`×2、`0768x1344`×2、
`0800x1344`×2、`0960x1120`×2、`1344x0800`×1。

### A-3 Anima text cache：**逐根记账**（必须报告，不得用 toy 替代）

- **实测口径（2026-11-07 requirements r2 复核，替换早先误记的"全树 = 0"）**：
  `krea2-webui` 全树 `*_anima_te.safetensors` = **92 个，全部位于 `output/**`；`output/` 之外为 0**。
  按根分布（8 个）：
  `output/ab-anima-20261002/cache` = **60**、
  `output/runs/anima-fp16-bf16-ab200-90hx-20261002/data/cache` = **24**、
  `output/runs/anima-adaptive-precision-90hx-20261001-delivery-batch2-inputs/cache` = **2**、
  `output/bench/block_swap_pipeline/workspace-1024/cache` = **2**、
  `output/runs/model-family-fused-flash-smoke-20260903/cache/anima` = 1、
  `output/runs/auto-families-hot-20260909/{anima,anima-retry,anima-formal-recovery}/cache` = 各 1。
- **文件总数 ≠ 成对完整，也 ≠ 与 `post_image_dataset/lora` 60 张图匹配**：`post_image_dataset/lora`
  侧 `*_anima_te.safetensors` = **0**，同目录 60 个是 **Krea-2** 的 `*_krea2_te.safetensors`
  （schema 不同，不可复用）。只有 `output/ab-anima-20261002` 这一根是内部成对完整的（见 A-3.1），
  且它对应的是**另一批图**，不是 `post_image_dataset/lora` 的 60 张。
- sidecar 命名真相源：`library/io/cache_names.py:22-23`、`library/io/cache.py:23`；
  加载点 `library/datasets/feature_sidecars.py:240-243`。
- 结论：**`post_image_dataset/lora` 不是 Anima 的"完整 cache"**，只有 latent 侧完整。

可选补救路径（**路径 3 不越"不预处理"边界**；路径 1 越界需用户批准；路径 2 前置未解析）：

1. **就地只读根之外做有界 TE 预处理**：用 A-1 的 TE 为这 60 张图生成 `*_anima_te.safetensors`，
   写入**新输出根**（不改 `krea2-webui`），并把 `cache_text_encoder_outputs_to_disk=true` 指向该根。
   需要先确认 `max-padded` padding 契约（AGENTS.md「Text Encoder Padding」不变量）不被改变。
2. **改用已验证完整的邻近只读 cache**：`/mnt/SD_Dataset/训练器相关/anima_lora/post_image_dataset/rokkotsu_goddess`
   实测含 **57 个 `*.npz` + 57 个 `*_anima_te.safetensors`**（成对完整）以及 `_anima_uncond_te.safetensors`；
   对应本项目 blueprint `configs/datasets/rokkotsu_goddess.toml`（`cache_dir` 同名，`image_dir` 由
   `ANIMA_ROKKOTSU_DATASET_DIR` 提供）。bucket 分布：`0896x1200`×27、`1792x2400`×30。
   代价：需要确认 `image_dir` 实际可指到（本轮未取该变量），且该仓不在 `krea2-webui` 根下。
3. **路径 3（推荐，无需任何预处理）**：`krea2-webui/output/ab-anima-20261002/` 已是一套只读、
   内部配对完整的集合——`cache/` 有 **60 `*_anima.npz` + 60 `*_anima_te.safetensors`（stem 1:1）**，
   同级 `images-1024/` 有 **60 png + 60 txt**；TE sidecar header 为
   `crossattn_emb`/`prompt_embeds [512,1024]` + `attn_mask`/`t5_attn_mask [512]` + `t5_input_ids [512]`
   （max-padded 512，60 个文件均为 2,105,884 B），符合 AGENTS.md「Text Encoder Padding」不变量。
   只需把现成 blueprint `krea2-webui/configs/datasets/anima-ab-1024-20261002.toml`
   （`enable_bucket=false`、`resolution=[1024,1024]`、`batch_size=1`、`caption_dropout_rate=0.0`、
   `image_dir`/`cache_dir` 指向上述两目录）**复制并重基路径到 `/mnt/SD_Dataset/...`，写入唯一新输出根**
   （原 blueprint 的 `/home/scv/nvme0n1p1/...` 前缀在本机不存在；**不写 `krea2-webui`**）。
   前提与代价：该集与 `post_image_dataset/lora` 的 60 张**不是同一数据集**，且全部 1024x1024，
   **不属 `CONSTANT_TOKEN_BUCKETS`**（见下面的 token 数学）。因此它只用于 §13 的 50 更新对照，
   **不得**被描述成"在 lora 那 60 张图上验证"，也不得据此外推 canonical 桶结论。

**1024x1024 固定 shape 的 token/兼容核算（A，只读源码）**

| 环节 | 事实 | 位置 |
| --- | --- | --- |
| VAE 压缩 | `latent_h = height // 8`、`latent_w = width // 8` → 1024x1024 ⇒ 128x128 latent | `library/anima/training.py:876-877` |
| patch 大小 | `patch_spatial` 默认 **2** | `library/anima/weights.py:142` |
| token 数 | (128/2)×(128/2) = **64×64 = 4096** → 既非 4032 也非 4200 家族，是**第三个** token-count 族 | 推导 |
| 无 bucket 时的 shape | `enable_bucket=false` ⇒ 对所有样本统一 `bucket_reso=(resolution,resolution)`，**无 intra-bucket padding** | `library/datasets/dataset_buckets.py:64-72` |
| shape 合法性 | 唯一硬断言是 H/W 可被 `spatial_patch_size` 整除（128%2==0 成立） | `library/anima/models.py:1218` |
| 编译键 | native flatten 以 token 数作编译 key；4096 自成一族，只多一张图、不违反编译契约 | `library/anima/models.py:2140-2146` |

- 全树**没有**"token 数必须 ∈ {4032, 4200}"的断言（对 `4032|4200` 的 grep 只命中注释、convrot 尾部
  掩码说明与 bucket 表本身），因此 4096 固定 shape **在本树可执行**。
- 但它是**固定 shape 单独对照**，不是 canonical 桶结论：4096 只覆盖"单一 shape、每窗口 token 数
  完全相同"这一最理想情形；**不得**外推到 4032/4200 混合桶、跨桶配对或 epoch 尾窗不等长样本。
  若后续要给出 canonical 桶结论，必须另跑一条 4032/4200 数据集的对照。
- **不因此解锁 RUN-A/B/C**：解锁与否仍取决于 §13 V-3 实测判据（含 per-sample 随机输入、rank 逐位
  一致性与 dtype-specific 数值门），本核算只证明"该 shape 不被 shape/token 契约挡掉"。

**冻结规则**：真实短训**必须**使用真实 Anima 基座 + 真实图像 cache；缺 TE cache 时只能走上面三条
已冻结的路径，其中**路径 3 是无需预处理的资产候选（不等于 RUN 解锁，见 A-3.1 的 L1/L2/L3）**、
路径 1 需用户明确批准、路径 2 需先补齐 `image_dir`；**禁止**用随机小模型/合成 latent 冒充 V-REAL。

#### A-3.1 独立复核（r2 content-bound 复核；下表标注处沿用 r1 attempt 6 既有实测行，不改写其结论）

| 检查 | 实测结果 |
| --- | --- |
| `post_image_dataset/resized` 组成 | 120 项 = **60 图 + 60 caption** |
| `post_image_dataset/lora` latent | **60** 个 `*_{WxH}_anima.npz`，stem 与 resized 图一一对应（抽样 3 组一致） |
| `post_image_dataset/lora` TE | `*_anima_te` = **0**、`*_krea2_te` = **60** → 该目录 TE 侧确为缺口 |
| `*_anima_te.safetensors` 全树 / `output/` 外 | **92 / 0**（8 个根，见 A-3 首条） |
| `output/ab-anima-20261002` 配对性 | **60 TE + 60 npz + 60 png + 60 txt**，`comm -3` 比对 stem 差集为空；TE 60 个均 2,105,884 B |
| `rokkotsu_goddess` cache | **57** npz + **57** `_anima_te`（成对完整） |
| `ANIMA_ROKKOTSU_DATASET_DIR` | **未设置**：`anima_lora/.env` 无该键；`krea2-multigpu/.env.example` 仅有占位 `/path/to/rokkotsu/images`（未读取任何凭据字段） |

**新增源码事实（A）：`image_dir` 是强制依赖，不能只靠 cache 训练。**
`library/datasets/dreambooth.py:132`（`if not os.path.isdir(subset.image_dir)` 直接报错）与
`:170`（`img_paths = glob_images(subset.image_dir, "*", ...)` 才是样本枚举来源）。
因此"cache 完整 ⇒ 可直接训练"在本树**不成立**——路径 3 之所以成立，是因为它**同时**具备
`images-1024/` 的 60 图 + 60 caption，而不是因为 cache 完整。

**据此冻结的可用路径**（不改变任何判据/阈值/预算）：

- **路径 3（资产侧候选，唯一无需预处理者）**：`ab-anima-1024` 重基后即可被 Anima loader 枚举与
  读取，**无预处理、无下载、不写只读仓**；它落在既有"不预处理"授权内，不构成越界动作。
  **"资产可用"不等于"RUN 已解锁"**——执行门槛见下面的 L1/L2/L3 三层区分。
- **路径 1（越界，需用户明确批准）**：为 `post_image_dataset` 那 60 张图生成 TE cache；
  当前**不在**用户已授权范围内，只能记为可选项，**不得**把既有"短训练"授权改写成预处理授权。
- **路径 2（前置未解析）**：`image_dir` 未设置，且仅有 cache 无法枚举样本。

**三层区分（本合同的唯一口径；t5/t7 的脚本按此消费，不另立解释）**

| 层 | 成立的内容 | 不成立 / 不得外推 |
| --- | --- | --- |
| **L1 资产只读可用** | 从指定只读 `images-1024/`（60 png + 60 txt）与 `cache/`（60 npz + 60 TE）可枚举 60 个样本，且可被 Anima loader 读取；**无需任何预处理**（路径 3）。 | 不等于"已在该数据上训练过"，也不等于"该数据已被本任务验证过"。 |
| **L2 固定 shape 证据范围** | 1024x1024 固定 shape = **4096 token**（既非 4032 也非 4200 族）；该 shape 下每窗口 token 数恒等，"均值之均值"成立。 | **不得**外推到 canonical 4032/4200 桶、跨桶配对、混合 bucket，或未受测的 epoch 尾窗；**source 类别（这 60 张图）、seed、样本数都只适用于本次固定 shape 单独 run**。 |
| **L3 RUN-A/B/C 准入门（未因路径 3 放宽）** | 必须同时满足：S-0 的 per-sample `(noise, timestep)` fixture 生效、单卡 `batch_size=2` 与双卡 `batch_size=1` 的同窗口 multiset 对照、组 R 的 rank 逐位一致性，以及 S-d（dtype-specific 梯度/Δθ 门）、S-e（skip 序列一致）、S-g（10+10 与连续 20 步逐位相等）。 | 任何资产候选（含路径 3）**都不自动 unlock** RUN-A/B/C；50 次有效更新预算只可由后续验证任务在**已批准计划 + §3 E-5 + §5 D-5 资源门通过**后执行，并以 t6 前置与 t7 验收为准。 |

⇒ 准确表述是：**资产侧不再阻塞（L1 成立）；执行侧仍未解锁（L3 未满足）**。剩余阻塞不在资产，
而在"必须先完成 S-0/S-d/S-e/S-g 与 t6→t7 的串行验收"；若诉求是"必须在
`post_image_dataset/lora` 那 60 张图上训练"，则另需路径 1 的用户授权。该区分只影响 §13
的数据集选择与执行顺序，**不影响** t2–t6 的 CPU/Gloo 与小模型 GPU 工作。

### A-4 uncond sidecar

`krea2-webui/post_image_dataset/_anima_uncond_te.safetensors`（1,050,792 B）存在；
`anima_lora/…/rokkotsu_goddess/` 同级也有。走 CFG 相关路径时以此为准，不需重建。

### A-5 只读边界的落地方式

- 所有 cache/模型只**读**；任何 sidecar、checkpoint、日志只写 §5 的唯一新输出根。
- 不写 `krea2-webui`、`anima_lora`、`post_image_dataset`；不覆盖、不移动、不删除既有产物。

---

## 5. 输出根、磁盘门与 checkpoint 体积预算（D）

### D-1 唯一新输出根 [A]

```text
output/tests/multigpu_reference_validation/<run-id>/
├── runs/{single-10,dual-20,dual-10plus10}/
├── states/…
├── traces/        # 每 rank 设备/参数指纹/scaler/loader 轨迹
└── logs/          # stdout + 运行参数 + 有效/跳过计数
```

`<run-id>` 形如 `20261107T1530-a1b2c3`（时间戳 + 短哈希），**一次任务只生成一个 run-id、
只创建一个根，不新增第二个**；RUN-A/B/C 与所有 traces/states/日志全部落在这一个根内。
该根**只在 §4 A-3.1 的 L3 准入门通过、且后续验证任务按已批准计划实际执行 RUN 时才创建**；
资产候选（L1）本身不产生任何写盘。`output/` 已在 `.gitignore`（第 8 行），沿用既有
`output/tests/` 惯例（同目录已有 `parallel_transport_v100_nvlink_20261107.json`）。

### D-2 单次训练必须落盘的证据（不可裁剪）

每 rank：设备名与索引、参数量与参数指纹（sha256 前 16 位）、`GradScaler` scale 与 growth tracker
（无 scaler 记 `None`）、`update_skipped`、`window_attempts/replays/discards`、
`optimizer` step 计数、`param_group` lr、窗口内样本 id 列表。
全局：完整 stdout、合并后的运行参数（含实际生效 `mixed_precision` 与改写原因）、
有效更新/跳过计数汇总、checkpoint 路径 + hash、loader/RNG 轨迹。

### D-3 checkpoint 体积预算（实测锚点 + 公式） [A]

实测锚点（只读，`krea2-webui/output/convrot-90hx-20261002-210307/s4-resume-bf16-first5/s4-resume-bf16-first5-step00000005-state/`）：
`du 473 MB` = `model.safetensors 165,221,616 B` + `optimizer.bin 330,635,135 B` +
`random_states_0.pkl 14,629 B` + `scheduler.bin 1,389 B` + `train_state.json 39 B`。

估算规则：

```text
state_dir ≈ P·b_save                 # model.safetensors（b_save=2 B/半精度, 4 B/fp32）
          + k·P·4 B                  # optimizer.bin，k=每参数状态数（AdamW:2, Lion/schedulefree:1）
          + n_rank·15 KB             # random_states_{rank}.pkl
          + 忽略项                    # scheduler/train_state
```
其中 `P` = adapter 可训练参数量（对照实测平铺 LoRA 产物 98.4 MB → `P≈5.2×10⁷`，bf16 存储）。

### D-4 硬体量上限（先登记，后执行）

| 项 | 上限 | 触发动作 |
| --- | --- | --- |
| 整个新输出根 | **≤ 6 GiB** | 超限前停止并报告，不加盘、不删他人产物 |
| 单次运行目录 | **≤ 2 GiB** | 超过即判该运行失败 |
| state 目录数 | **≤ 2 个/运行**（step-10、step-20） | 只保留续训边界必需的那个 |
| 单个 state 目录 | **≤ 700 MiB** | 超过先查是否把冻结基座写进去了 |
| `state/model.safetensors` | **≤ 300 MiB** | **超过即为故障**：说明 `save_model_hook` 没剥掉冻结 DiT（基线文件 4.18 GB） |

### D-5 运行前磁盘门（必须在每次 V-REAL 前执行并记录）

```bash
df -B1 /mnt/SD_Dataset    # 2026-11-07 实测：916G 总 / 48,911,024,128 B 可用（≈45.6 GiB）/ 96% 使用
```

- 要求启动前可用空间 **≥ 15 GiB**；否则**报告资源不足并停止**（按团队约定"资源不足先报告，不延长"）。
- **保存前门（实现阶段必须内建，不是人工检查）**：每次落盘前计算
  `projected = P·b_save + k·P·4 + 15 KB·world_size`，要求
  `当前可用 − projected ≥ 10 GiB`；不满足则**拒绝保存并一致失败**（走 §9 类 I 协商），
  不允许"先写了再说"，也不允许自动删旧证据腾空间。
- **不得假设两种极端之一**：
  - 「反正只保存 LoRA」——`save_model_hook` 的剔除依赖 hook 已注册且类型匹配
    （`library/training/checkpoints.py:717-740`），一旦失效就会把 4.18 GB 冻结基座写进 `state/`；
  - 「最坏就是复制一份全基座」——真实最坏是**每个 rank × 每个保存边界**各一份，
    3 个边界 × 2 rank × 4.18 GB ≈ 25 GB，足以把 45.6 GiB 打穿。
  因此 §5 D-4 的 `state/model.safetensors ≤ 300 MiB` 是**故障判定线**，不是优化目标。
- 禁止 `--save_state_on_train_end` 与每步保存叠加；保存 flag 冻结见 §13 V-4。

### D-6 每个 case 的保留量与预算封顶（冻结）

| case | 保留物 | 数量 | 预估体积 | 上限 |
| --- | --- | --- | --- | --- |
| RUN-A（单卡 10 步） | 最终 adapter safetensors + stdout/argv/counters/trace | 1 adapter，0 state | ≈ 0.1 GiB | 0.5 GiB |
| RUN-B（双卡 20 步） | step-20 state + 最终 adapter + 上述日志 | 1 state + 1 adapter | ≈ 0.7 GiB | 1.0 GiB |
| RUN-C（双卡 10+10） | step-10 state、step-20 state + 最终 adapter + 上述日志 | 2 state + 1 adapter | ≈ 1.3 GiB | 1.5 GiB |
| 合计 | — | 3 state + 3 adapter | ≈ 2.1 GiB | **≤ 6 GiB** |

- 体积以 `P = 5.2×10⁷`（对照 98.4 MB bf16 平铺 LoRA 实测）与 AdamW `k=2` 估算；**t2 落地前必须
  用真实 adapter 的 `P` 复算并写进证据**，若 `state/` 超过 700 MiB 先查是否写入了冻结基座。
- RUN-C 用它自己的 step-20 产物与 RUN-B 的 step-20 做 §13 S-e 的逐位比对；两份都必须保留到
  比对完成并落盘 hash 之后，才允许按上限回收**本任务自己产生的**中间件。
- 只回收本任务新根下的内容；**不得**触碰任何既有 `output/` 产物、模型、cache。

---

## 6. P0-0 冻结协议：窗口健康 / 跳过决策 / scaler 状态的跨 rank 一致性

### W-1 冻结的源码事实（A）

| # | 事实 | 位置 |
| --- | --- | --- |
| W1 | 微批 loss 有限性判定是 **rank-local**，无条件执行 | `library/training/loop.py:1109` |
| W2 | 只在 `accelerator.sync_gradients` 边界做一次梯度归约（loss-scaled 状态） | `library/training/loop.py:1140-1152` |
| W3 | 无 scaler 时的跳过判定**会**做 `reduce(max)` 汇合；有 scaler 时直接返回"交给 scaler" | `library/training/scaler_ownership.py:161-174`、`177-200` |
| W4 | `drop_update = skip_update or force_skip or not loss_is_finite`；为真只 `zero_grad`，**不进入** `optimizer.step()` | `library/training/loop.py:1217-1226` |
| W5 | `AcceleratedOptimizer.step()` 内部才调用 `scaler.step()` + `scaler.update()` | venv `accelerate/optimizer.py:161-180` |
| W6 | 窗口重放/丢弃判定完全 rank-local，且**无条件**走普通 epoch 路径 | `library/training/loop.py:830-856`、`library/training/window_accounting.py:107-113` |
| W7 | 重放预算 `WINDOW_MAX_REPLAYS = 1`；丢弃预算是 rank-local 的连续计数 | `library/training/window_accounting.py:47,50,161-180,200-215` |
| W8 | scheduler 只在 **commit 边界**推进一次，skip 路径不推进 | `library/training/loop.py:986-1005`、`1303-1322` |
| W9 | 无任何"窗口健康/跳过/scaler"的跨 rank 状态协议（全树无对应 collective） | 与 W2/W3 同源，`loop.py` 内无 `reduce`/`broadcast` 参与该判定 |
| W10 | 安装的 accelerate 1.13.0 与 torch 2.10.0 **都没有** `found_inf` 的跨 rank 归约 | 对 site-packages 全包 grep `found_inf` 命中 0；`torch/amp/grad_scaler.py` 无 `all_reduce` |
| W11 | 现有"OR 汇合"用的 `accelerator.reduce(flag, reduction="max")` **不是 max**：accelerate 的 `reduce` 只对 `"mean"` 分支做除法，其余字符串一律退化为 `all_reduce(SUM)` | 源码 `accelerate/utils/operations.py:745-762`（唯一分支是 `if reduction == "mean"`）+ 本轮 V-GLOO 实测（两 rank 均置 1.0 时 `sum=2.0`、`mean=1.0`、**`max=2.0`**） |

**因此 `library/training/scaler_ownership.py:187-188` 的注释（"The scaler already all-reduces
`found_inf` across ranks"）是源码级错误**，与对照报告 §10 末条一致。真实对称性只来自 W2（归约在前），
而 W2 **只覆盖"梯度本身非有限"** 这一支。

### W-2 冻结的结论（B，前提写明）

在 W1–W7 的组合下，下列情形**静态可达**且当前无任何机制阻止：

- **(a) 坏 loss + 有限梯度**：一个 rank 走 `drop_update` 分支，另一个 rank 进 `optimizer.step()`
  → 只有后者推进 `scaler.step()/update()`，`scale` 与 `growth tracker` 分叉；后续梯度量级随之不同。
- **(b) 各 rank 窗口长度不等**（epoch 尾窗、`drop_last=False`、数据分片不均）：归约次数本身不等
  → 不是"值不一致"，而是 **collective 次数不一致**。
- **(c) 重放/丢弃决策分叉**：任一侧判"需要重放"而另一侧判"可提交"，则一侧多跑一次 backward，
  多出的一次归约会把另一侧的累积梯度当作本窗口梯度吃掉 → 静默污染或 PG 超时。
- **(d) 丢弃预算分叉**：一侧连续丢弃耗尽预算而中止，另一侧继续 → 单 rank 退出，其余卡在 collective。

`force_skip`（同窗口早前微批已知坏）同样由本 rank 的 `window.losses` 决定，不改变上述性质。

### W-3 冻结决策（不可协商，实现阶段必须逐条满足）

1. **全 rank OR 决策**：窗口健康（`nonfinite_loss` / `update_skipped` / 需要重放 / 需要丢弃）
   必须由**单一来源**决定并对全体 rank 生效。允许的实现形态**只有两种**，且必须显式写在代码注释里：
   (i) `all_reduce(SUM)` on `{0,1}` 标志后判 `> 0`；(ii) `torch.distributed.all_reduce(op=ReduceOp.MAX)`
   或等价的显式 boolean gather。**不允许** rank 本地判定直接驱动控制流。
   **禁止**继续使用 `accelerator.reduce(..., reduction="max")`（W11：它其实是 SUM）——现有
   `library/training/scaler_ownership.py:161-174` 的 0/1 + `>0` 组合**结果正确**、可以在实现前保留，
   但必须在 t2 改成上面两种显式形态之一，并补一条回归：两个 rank 都为 1 时归约结果为
   `2`（SUM 形态）而非 `1`，同时 OR 判定恒为 `True` 且**与 world_size 无关**。
2. **推进/增长计数对称**：`GradScaler` 的 `scale` 与 `_growth_tracker` 在"全体跳过"和"全体提交"两种
   情形下都必须**逐 rank 相同**；不允许靠"反正后面会追上"解释差异。若实现选择"跳过时也调用
   `scaler.update()` 之外的路径"，必须显式写出该选择及其理由，并以观测证明对称。
3. **optimizer/scheduler/提交游标对称**：`optimizer` step 计数、`param_group` lr、`global_step`、
   `window_attempts/replays/discards`、`window_consecutive_discards` 必须逐 rank 相同；scheduler
   仍只在 commit 边界推进（W8 不变）。
4. **不以 rank0 状态代替全体**：任何"只看 rank0 是否坏"的判定都不合格；必须能在**任一 rank**
   本地观测到坏状态时让全体一致转入同一分支（即 OR，不是 rank0 采样）。
5. **不默认"补一个 reduce 就够"**：只加一次归约、却仍让 scaler 推进路径不对称或重放次数分叉，
   判 **未通过**。必须同时覆盖"窗口健康 → 跳过决策 → scaler 推进 → 提交游标"这条链。
6. **失败即统一终止或保持现有拒绝**：不得把已知可能分叉的 rank-local 判定当作"安全回退"。

### W-4 验收场景（每个场景独立判定，V-GLOO 为最低层级）

| 场景 | 构造 | 必须比对 |
| --- | --- | --- |
| **S①** | 坏 loss + **有限梯度**（人为让 `loss.item()` 非有限，梯度仍有限） | 四项全等 |
| **S②** | 坏 loss + **非有限梯度** | 四项全等 |
| **S③** | accum 尾窗（epoch 末尾不足一个窗口） | 四项全等 + 不 hang |
| **S④** | 正常窗口对照（无异常） | 四项全等 |

四个场景每个都必须比对**四项**（缺一即不合格）：

1. 每步 `update_skipped` / replay 次数 / 提交窗口序列；
2. `GradScaler` 的 `scale` 与 growth tracker；
3. `optimizer` step 计数与 `param_group` lr；
4. 参数逐位一致（每一 commit 边界，`torch.equal`）。

**任一项不同即判该路径未通过**；不能只比 replay 次数，也不能只比最终参数。

### W-5 P0-0 采用/否决条件

| 结果 | 判定 |
| --- | --- |
| 四场景 × 四项全等，且实现满足 W-3 六条 | **采用**，允许进入 S1 真实短训前置 |
| 任一场景任一项不等 | **否决**该实现；`needs_revision` + 具体 findings，不得以"影响很小"放过 |
| 只覆盖"梯度 NaN"一支而漏 loss-nonfinite / 尾窗 | 否决（对照报告 §8.2 的口径收窄） |
| 实现让单卡行为变化 | 否决（§11 R-1 单卡回归） |

责任阶段：**DDP 核心实现 → CPU/Gloo 独立验证 → 核心 review**，必须早于任何 V-REAL。

---

## 7. 控制面 P0-1..P0-5

| 项 | 验证目标 | 证据级别 | 采用条件 | 否决条件 | 责任阶段 | 观测层级 |
| --- | --- | --- | --- | --- | --- | --- |
| **P0-1** 选卡数与 `num_processes` 对齐（fail-closed） | 显式 `gpu_selection` 非空时，`num_processes` 必须 **`== len(selection)`**；`<` 与 `>` **两个方向**的非等均都是具名拒绝。未显式指定进程数时才按选卡数推断，单卡默认仍为 1。选 N 卡起 M(<N) 或 >N worker **均不视作合法用户选择**，同卡多 worker 不豁免。 | A（现状缺口）：`library/runtime/launch.py:49-63` 只在 env **未设置**时补 `num_processes`，已设置则原样保留，无一致性判定 | 单测覆盖四种：默认推断正确 / `<` 具名拒绝 / **`>` 具名拒绝** / 单卡=1 行为不变；错误码由**单一函数**产出且 launcher 抛同码（§7.3-1） | 只拒 `<` 而放行 `>`；或以"同卡多 worker / 用户显式选择 / 性能需要"为由豁免任一方向；或把单卡默认改为 >1 | DDP 核心实现 + CPU 验证 | V-CPU |
| **P0-2** 启动期 collective 健康探针 | 通信错误前移，**失败即中止** | A（参考实现可用性）+ B（本树无此路径） | 仅多卡路径启用或显式开关、**默认关**；失败统一终止；一次 V-GLOO 双进程证明可通 | 默认开启（引入不可控启动期 collective）；失败后"降级继续" | DDP 核心实现 | V-CPU + V-GLOO |
| **P0-3** 运行清单记录多卡身份 | `world_size`、`configured_world_size`、归约模式、bucket 数、**实际生效 `mixed_precision` 与改写原因** | A（参考字段存在）+ B（本树未写） | 字段存在、rank0-only 写、与 E-6 改写原因一致 | 多 rank 重复写同一文件（见 §7.1 新发现 N-2）；或把 launcher 提示值当生效值 | DDP 核心实现 | V-CPU |
| **P0-4** 准入原因码 + 训练进程内传真实 `world_size` | CLI 直连多卡时也走同一套 fail-closed 判定 | A（缺口闭合）：`library/training/bootstrap.py:162-172` 调 `check_training_compat({...}, )` **不传** `world_size`；`library/training/compat_matrix.py:241,250,256,260` 用 `world_size or 1` 求值；只有 Web 预检传了真实值（`web/services/config/preflight_compat.py:189-191`） | 单测逐原因码；CLI 直连断言；原因码由**单一函数**产出 | 原因码与判定漂移（同一规则两处真值）；只改 Web 侧 | DDP 核心实现 + CPU 验证 | V-CPU |
| **P0-5** 恢复语义显式化 | 不再宣称"精确恢复"；对"改变 world_size 恢复"给出显式拒绝或降级 | A（现状）：`library/training/checkpoints.py:435-441` 的 `resume_skip_plan` / `444-526` 的 `plan_resume_start` 只按步数切 epoch 与跳批，**无 world_size 校验**；`library/training/loop.py:801-805` 的 `skip_first_batches` 只是跳过 N 个批 | 同 world_size 的中断/续训对照（§13 V-3 的强门槛）+ 跨 world_size **显式拒绝**并具名 | 静默允许跨 world_size 续训；或用"skip N 批"当恢复正确性的证明 | checkpoint/恢复实现 | V-GLOO + V-REAL |

### 7.1 本轮新增的两处源码事实（前轮报告未列，A）

- **N-1 目录 rename 竞态**：`library/training/checkpoints.py:596-625` 的 `save_checkpoint_state`
  在 `accelerator.save_state(tmp_dir)` 之后，**每个 rank 都**执行
  `os.replace(state_dir, backup_dir)` → `os.replace(tmp_dir, state_dir)` → 删除 `backup_dir`，
  且 `_recover_checkpoint_state_dirs`（`:549-557`）在每个 rank 入口先跑一遍。
  **没有 `is_main_process` 门，也没有 `wait_for_everyone()`**。多卡下两个 rank 会对同一路径并发
  rename/删除：可能出现一侧 `FileNotFoundError`、或一侧把另一侧刚写好的 state 移进 backup 再删掉。
  本项列入 **P1-1** 的必测内容（不是"文档补充"）。
- **N-2 `train_state.json` 全 rank 写**：`library/training/checkpoints.py:717-740` 的 `save_model_hook`
  只对"剔除非 network 模型"这一段加了 `is_main_process`（`:718`），随后的
  `open(train_state_file, "w")` + `json.dump`（`:727-740`）**对所有 rank 无条件执行**，写的是同一个
  共享路径。内容虽小且同值，但违反"主进程专属副作用"不变量，且与 N-1 叠加放大风险。
  列入 **P0-3/P1-1** 的观测点与修复面。

> 这两条都属对照报告 §5"主进程副作用与参考原则一致"的**过度概括**修正：`is_main_process` 确实
> 多处存在，但保存路径上的 rename 与 `train_state.json` 写入**不在其中**。

- **N-3 `reduce(reduction="max")` 是未文档化的 SUM**：`library/training/scaler_ownership.py:170`
  调用 `accelerator.reduce(flag, reduction="max")`；accelerate 1.13.0 的
  `reduce`（`accelerate/utils/operations.py:745-762`）对 `reduction` 只识别 `"mean"`，
  其他任何字符串都执行 `all_reduce(..., ReduceOp.SUM)`。官方 `Accelerator.reduce` 文档字符串
  （`accelerate/accelerator.py:3139-3140`）声明的合法值只有 `'sum' | 'mean' | 'none'`。
  本轮 V-GLOO 实测（两进程、两个 rank 的 flag 均置 `1.0`）：`sum=2.0`、`mean=1.0`、`max=2.0`
  —— 即 `"max"` 与 `"sum"` 同值。因标志是 `{0,1}` 且判据是 `> 0`，**今天的 OR 语义仍然正确**，
  属"能工作但契约未表述"的实现；风险在于：任何把标志改成非布尔/带权重的改动都会静默改变语义，
  且此 API 用法不受 accelerate 承诺。→ 列入 **t2 必修**（§6 W-3 第 1 条）。

### 7.2 本轮实测得到的探针配方（供 t2/t3 复用，A）

- **不要**用裸 `mp.spawn` 跑 accelerate 的 `reduce`/`PartialState` 语义探针：实测 `Accelerator()`
  在这种环境下报告 `distributed_type=DistributedType.NO`（`accelerate/state.py:806-820` 的
  MULTI_CPU 分支要求 `cpu=True`），`reduce` 于是短路成 clone，会得出"没有归约"的错误结论。
- **可用配方**（本轮已跑通，2 进程 Gloo）：
  `Accelerator(cpu=True, kwargs_handlers=[InitProcessGroupKwargs(backend="gloo", timeout=timedelta(seconds=60))])`
  + 子进程内先设 `RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT` → `distributed_type=MULTI_CPU`，
  `num_processes=2`，`reduce` 走真实 `all_reduce`。
- GPU 侧等价探针必须用 `accelerate launch` / `torchrun` 生成的环境，不得手工拼 env 冒充。

### 7.3 实现态记录（t14 repair-round-2，不改变判据）

本节只记录**实现已经冻结的口径**与取证入口，不新增或放宽任何判据/阈值。

1. **选卡↔进程数冲突两方向一律拒绝**（P0-1 的落地口径）。
   `library/training/multigpu/admission.py::resolve_admission` 在「显式 `--gpu` 选择非空且 launch
   生效」时要求 `num_processes == len(selection)`：
   - `<`：`1 worker / N cards` → `multigpu_single_gpu_multi_process`；`1 < n < N` →
     `multigpu_num_processes_selection_mismatch`；
   - `>`：worker 多于卡数 → `multigpu_num_processes_selection_mismatch`。
   **不存在"显式值优先、原样放行"的例外**；一卡多 worker 需要一份本合同尚未定义的显式 opt-in
   与配套证据，因此按拒绝处理，而不是按用户偏好放过。
   `library/runtime/launch.py::configure_accelerate_for_gpu_selection` 在入口抛同一原因码，所以
   CLI、daemon、Web 启动路径口径一致。回归：`tests/test_multigpu_admission.py`
   （两方向 + launcher 同码）、`tests/test_launch_config.py`
   （`test_gpu_selection_world_size_rejects_more_workers_than_cards`）。
2. **OR 原语必须看真实归约值**（W-3.1）。回归：`tests/test_multigpu_window_protocol.py`
   的双进程 Gloo 用例断言两 rank 全 1 时 `reduce("sum") == 2.0`、`agree_any` 判定为 `True`
   且与 world_size 无关；只有一 rank 置真时归约值为 `1.0` 而判定仍为 `True`。
3. **`window failure reason` 允许 rank-local，但判定与计数必须一致**。
   原因字符串（`nonfinite_loss` / `update_skipped` / `peer_window_decision`）描述的是**该 rank 自己
   观测到的**触发因素，多卡下天然可以不同；合同要求一致的是**决策结果**
   （drop/commit/discard）与**计数**（`window_attempts/replays/discards`、`global_step`、
   `optimizer` step、`param_group` lr、`GradScaler` scale/growth tracker）。后续 review 不得把
   "两个 rank 的 reason 字符串不同"判成分叉；反之，判定或计数不同即为缺口。
   回归：`tests/test_multigpu_window_protocol.py` 的 `_assert_symmetric` 逐项比对判定与计数
   （不比对 reason 字符串）。
4. **运行身份（P0-3）的失败语义**：多进程下 identity 清单写不出 = **具名失败**
   （`multigpu_identity_write_failed`，与 P0-2 启动探针同构），由 `agree_any` 广播后
   全体一致终止；`--output_dir` 缺失同样拒绝启动。清单由 rank0 用一次小 `all_gather`
   收集**全 rank** 的 device/precision/world_size 后写单文件，因此跨 rank 分叉可见。
   `configured_world_size`（launcher 意向）与 `world_size`（真实加入数）**分列**；
   `reduction_mode` / `gradient_buckets` / `reduced_parameter_count` 由循环在第一个边界
   从真实 `synchronize_optimizer_gradients` 返回的 `GradientSyncResult` 回填，未发生边界前
   记 `pending_first_boundary` / `None`，不伪造观测。
5. **边界局部异常**（类 I 的实际覆盖范围）：优化器 step 段落的 rank-local 异常被捕获后，
   通过**同一** agreement 原语（每个边界所有 rank 同序调用一次）广播 `boundary_error`，
   任一侧报错则全体以 `multigpu_boundary_error` 终止。**transport 类异常例外**：类型来自
   `torch.distributed` 或消息含 NCCL/Gloo/socket/timeout 等标记时直接抛出，不再发起新
   collective（类 II，进程树终止）。step 段落之外的异常仍按类 II 处理，W-3.6 不声称覆盖。
6. **不发布虚假对称性字段**：`StepAgreement` 不再携带 micro-batch 计数——调用点恒为 1，
   发布它只会制造不存在的对称性来源。各 rank 微批数不同时**到不了边界**（会挂在梯度
   集体里），属类 II，不在本协议能力范围内。
7. **`WORLD_SIZE` 权威优先**：进程内已存在合法 `WORLD_SIZE` 时直接以其为准，
   admission 推断值仅作回退（包括 `WORLD_SIZE` **小于** admission 值的场景）。




---

## 8. checkpoint / 恢复协议 P1-1..P1-3

### P1-1 checkpoint 三阶段协议

| 维度 | 冻结内容 |
| --- | --- |
| 验证目标 | 保存过程在所有 rank 参与下**顺序确定**：进入 → 交换/校验 (step, identity, world_size, 是否需要全 rank 采集) → 各 rank 本地采集 → 比对 `global_step` 全等 → **仅 rank0 组装并落盘**（含目录 rename）→ 报告广播回全体 |
| 证据级别 | A（参考实现成立）+ A（本树缺口：N-1/N-2、无阶段交换） |
| 采用条件 | ①四类元数据交换齐全且校验失败可判；②rename/publish 只有 rank0 执行，其余 rank 在 barrier 后再决定；③发布成功/失败都广播回全体；④state 目录体积满足 §5 D-4；⑤V-GLOO 下 `torch.equal` 逐 rank 一致 |
| 否决条件 | 仍有多 rank 并发 rename；失败只在本 rank 体现；把 OSError 当"重试即可"吞掉；rank0 落盘后不广播 |
| 责任阶段 | checkpoint/恢复与隔离实验实现 → 分层 CPU/Gloo/CUDA 验证 |
| 观测层级 | V-CPU + V-GLOO（真实 CUDA 保存另在 V-REAL 观测） |

### P1-2 版本化 per-rank resume bundle

| 维度 | 冻结内容 |
| --- | --- |
| 验证目标 | resume 元数据显式版本化：`version`（必须匹配）、`world_size`、`backend`、`rank↔device` 映射、`rank_local_keys`；加载时逐项校验；rank-local 段不得含共享键 |
| 证据级别 | A（参考实现）+ C（本树 `accelerate.load_state` 走内部 `_dataloaders` 注册表，`library` 依赖其行为） |
| 采用条件 | ①拓扑不符**拒绝**（见 §10）；②legacy checkpoint 无 bundle 时**显式**降级并标 `exact_resume=False`（日志 + 清单字段，不静默）；③恢复后 `global_step/epoch/scheduler/lr` 全 rank 比对一致 |
| 否决条件 | 无版本校验；拓扑不符静默继续；"没传 dataloaders 参数"被当成"未恢复"或"已恢复"的证据（对照报告 §8.4） |
| 责任阶段 | checkpoint/恢复实现 → 分层验证 |
| 观测层级 | V-GLOO（主）+ V-REAL（一次 10+10） |

### P1-3 带证据指针的能力账本

| 维度 | 冻结内容 |
| --- | --- |
| 验证目标 | 每条多卡能力一行：`state ∈ {supported, rejected, experimental, unverified}` + 证据指针（测试名/命令/artifact 路径）+ 最后验证日期 |
| 采用条件 | `validated` 只能由**本契约下真实跑过的命令**填；未跑的写 `unverified`；`rejected` 必须给出拒绝代码位置 |
| 否决条件 | 出现无指针的 `validated`；或把 V-CPU 证据当 V-REAL |
| 责任阶段 | 总体独立 review 前的收口（不阻断实现） |

---

## 9. 故障语义：两类必须分开处理

### 类 I：**PG 健康时的局部错误**（必须协商出一致决定）

- 形态：某 rank 的本地 I/O 失败、rank 局部采集失败、rank0 发布失败、adapter 保存失败、窗口判定异常。
- 要求：**仍必须完成一次状态交换**，把 `errors` 汇总给全体，再由全体做**一致决定**
  （一致中止，或一致降级到预先声明的状态）。不允许"一侧抛异常退出、另一侧继续进入下一次 collective"。
- 验收：故障注入（V-GLOO）——单侧抛错，另一侧必须收到同样的结论；**不许 hang**。
  判定依据：`update_skipped`/状态等价物逐 rank 相同 + 进程退出码一致 + 无 PG 超时。

### 类 II：**transport 失联 / PG 破坏**（不再依赖新 collective）

- 形态：peer 进程死亡、NCCL/Gloo transport 报错、PG 被销毁。
- 要求：**不得**再尝试新的 collective 来协商；由 PG timeout 与进程树终止处理。
  本项目不覆盖 timeout：`library/runtime/accelerator.py:159-165` 未传 `InitProcessGroupKwargs`，
  因此取 torch 后端默认（实测 `default_pg_timeout=0:30:00`、`default_pg_nccl_timeout=0:10:00`）[A]。
- 必须写明的代价：**无优雅 flush**（与对照报告 R6 同结论）——被杀时可能丢失最后一个窗口的日志/
  checkpoint，这是设计取舍，不是缺陷。
- 验收：故障注入——`kill -9` 一个 rank，另一 rank 必须在 ≤ PG 默认超时内以非零退出码终止，
  不产生 hang；**不得**声称"数据已 flush"。

**冻结**：类 I 不允许"统一终止"以外的兜底（除非降级状态已在文档中显式声明并逐项验收）；
类 II 不允许写成"已容错"。

---

## 10. world-size 变化语义

| 情形 | 冻结行为 |
| --- | --- |
| 同 `world_size` 续训 | 必须可判定：§13 V-3 的强门槛（第 20 步参数逐位相等）通过才算"精确恢复" |
| 不同 `world_size` 续训（P1-2 落地后） | **默认拒绝**（fail closed），错误码具名，提示"拓扑已变" |
| 不同 `world_size` 续训（legacy checkpoint，无 bundle） | 允许**显式降级**：`exact_resume=False` + 警告 + 清单字段，且**不得**被描述为"恢复了" |
| 单卡 → 多卡 / 多卡 → 单卡 | 同上面两条，不设"自动适配" |
| `num_processes != len(gpu_selection)`（**含 `>`**，两个方向） | P0-1 具名拒绝启动，**无豁免**（§7.3-1） |

现状（A）：上述校验**一概不存在**——`library/training/checkpoints.py:435-441`、`444-526` 与
`library/training/loop.py:801-805` 只按步数跳批。因此 P0-5/P1-2 之前的任何"多卡续训"都只能标
**unverified**。

---

## 11. 单卡兼容与"已有等价机制"：一律作回归项，不重做

### R-1 单卡默认行为不变（硬门）

- 单卡路径不得因多卡实现产生行为变化：`num_processes<=1` 时归约/探针/清单字段/拒绝逻辑全部空转。
- 已有多处短路可复用：`library/training/gradient_sync.py:270`（`synchronize_optimizer_state`）、
  `:333`（`synchronize_optimizer_gradients`）、`:104`（`begin_step`）、
  `library/training/scaler_ownership.py:161-174`（`processes<=1` 直接返回，不建 tensor）。
- 验收：§15 的 CPU 回归地板（110 passed）保持全绿 + 一条"多卡开关全关"的端到端形状断言。

### R-2 已有等价机制清单（回归项）

| 机制 | 现有实现 | 本契约的立场 |
| --- | --- | --- |
| adapter 梯度同步（presence 掩码 + world 均值） | `library/training/gradient_sync.py:321-389`（含 `:367-382` 掩码、`:199-201` 均值） | **回归**，不重写；只加"同一参数每次边界归约一次"的断言 |
| 参数/优化器状态 rank0 广播 | `library/training/gradient_sync.py:262-296`（`from_process=0`） | 回归 |
| 采样/预览的 rank 编排（分片、失败先归约、双 barrier） | `library/anima/training.py` 预览段；`tests/test_distributed_sampling.py` 已有 Gloo 用例 | 回归，不重做 |
| 无 scaler 时跳过判定的跨 rank 汇合 | `library/training/scaler_ownership.py:161-174` | 回归 + 在 P0-0 里扩展 scaler 分支 |
| 2 进程 Gloo 梯度同步契约 | `tests/test_distributed_gradient_sync.py:456-520` | **回归地板**，新增测试必须复用其 init 风格（`file://` + `mp.spawn`） |
| 窗口重放单进程行为 | `tests/test_window_replay.py`（14 passed）+ 共享夹具 `tests/window_replay_harness.py` | **回归地板**；多进程协议测试必须复用该夹具，不另写第二套假 loop |
| checkpoint 现有保存/恢复 | `tests/test_training_checkpointing.py`（55 passed） | 回归地板 |
| E3 账本 8 观测点 | `tests/test_fp16_nan_static_equivalence.py`（24 passed） | 回归地板 |

### R-3 前轮**已撤回**的误报：不得重做

以下三项在对照报告 §8 已被反证，**不进入本契约验证清单**，也不设"预期失败探针"：

1. `scale_weight_norms` 缺 `all_reduce`（`networks/lora_anima/regularization.py`）——确定性运算，
   输入相同必然相同；只作为"参数一致性"的**观测点**，无预设失败。
2. `init_trackers` 多 rank 重复副作用——accelerate 1.13.0 的 `@on_main_process` 使非主进程 tracker 列表为空。
3. scheduler 乘 `num_processes` 导致步数翻倍——`AcceleratedScheduler.step()` 默认每次推进
   `num_processes` 次，两者相抵；前提：经 `prepare` 包装、每窗口只推进一次、未启用 `split_batches`。

> 三条中任何一条若在后续实测翻案，必须有新的源码/测量证据，走 §17 修订；不得凭印象恢复。

### R-4 native/islands 门不挡普通 window replay（口径收窄，A）

- `native` / `islands` 的执行点确实带真实 world_size：`library/runtime/accelerator.py:166-170`
  传 `world_size=accelerator.num_processes`；规则本体 `library/training/adaptive_runtime/native_precision.py:66`
  与 `library/training/adaptive_runtime/training_config.py:111` 在 `world_size != 1` 时报错。
- adaptive 恢复另有硬门：`library/training/adaptive_runtime/resume_checkpoint.py:91,202`
  与 `library/training/adaptive_runtime/resume_runtime.py:39-46`（多进程或 accum≠1 直接 raise）。
- **但通用 window replay 没有任何世界规模前置**：`library/training/loop.py:830-856` 走普通 epoch 路径
  且无条件执行；`--adaptive_precision` 默认 `off` 时 `native_enabled/islands_enabled` 均为假。
- 冻结结论：**保留**"多卡 + adaptive native/islands 精度 + adaptive 恢复"的现有拒绝；
  同时承认"普通 `adaptive_precision=off` 的多进程路径同样可达窗口重放"，因此 P0-0 是**普通多卡路径**
  的必修项，不是可选实验。

---

## 12. 隔离实验与明确不适用项

### EXP-FSDP2-BASE（实验候选，不进生产默认）

| 维度 | 冻结内容 |
| --- | --- |
| 验证目标 | 冻结基座分片**是否**降低驻留显存；并**先定归属**（FSDP2 与现有手动同步不得双重归约，也不得出现无人负责的参数） |
| 证据级别 | B（收益未验证）+ A（拒绝组合与边界可确定） |
| 最小验证（本批） | 仅小模型：`trainable_param_ids ⊆ optimizer_param_ids`、`requires_grad` 不变、wrap 计数正确、注入/`fully_shard`/optimizer 构造三序的参数身份（`id(param)` 稳定性、别名）逐项实测 |
| 采用条件 | 归属唯一且可断言；CUDA/NCCL 前后数值对齐 ≤ 预登记阈值；保存/恢复可用；峰值显存**实测**下降 |
| 否决条件 | 任何参数被两条路径各归约一次；或"注入的 adapter 自动被 `fully_shard` 管理"被当默认假设；或与 TP/SP、block swap/module offload、NF4/FP8、PP、未验证 compile 组合 |
| 责任阶段 | 隔离实验实现 → 分层验证（CUDA 仅小模型） |
| 不得外推 | CPU/Gloo 只能证明**本项目的**参数清单/拓扑协议，不能证明 `fully_shard` 真实运行 |
| 生产门 | **不解除**现有生产兼容拒绝；不写默认配置/UI 提示 |

### EXP-TP-COLROW（实验候选，不进生产默认）

| 维度 | 冻结内容 |
| --- | --- |
| 验证目标 | TP column/row 替换 + 注入器识别 + 复制参数的**梯度同步协议**；buffer/掩码类对象协议**分开**定义 |
| 证据级别 | B（收益未验证）+ A（准入拒绝面确定） |
| 最小验证 | TP1 与 `nn.Linear` **逐位一致**；TP2 数值 rel-L2 ≤ 预登记阈值 + 每步**集合通信次数统计**（计数口径：窗口累计与每步均值分开记） |
| 采用条件 | 逐位一致 + rel-L2 达标 + 通信次数不随层数爆炸；掩码/缓冲协议独立验收 |
| 否决条件 | 任何 DP 混用；Krea2 attention；fused QKV；把"掩码按梯度归约"当默认；把窗口平均计数当每步精确计数 |
| 责任阶段 | 隔离实验实现 → 分层验证（CUDA 仅小模型） |
| 生产门 | 同上，**不解除** |

### 明确不适用 / 不迁移（本次 Anima 任务）

| 项 | 结论 | 需要的证据形态 |
| --- | --- | --- |
| DeepSpeed | **不适用、不迁移** | 拒绝路径证据：准入处给出具名拒绝（缺包/策略互斥），并有回归用例断言拒绝仍成立 |
| Sequence Parallel（SP） | **不适用、不迁移** | 同上（`expanded_kv` 路由与模块齐备性检查保留在参考侧，不在本树落地） |
| Pipeline Parallel（PP） | **不迁移**（本树 runtime 未接线，配置层仍返回 `pipeline_parallel_runtime_unavailable`） | 回归证据：该拒绝仍在；不因多卡改动被放宽 |

### 许可边界

参考仓为 PolyForm Noncommercial 1.0.0；本树 MIT。**只提炼机制与接口契约，不复制代码、不做近似改写**。
任何 PyTorch/Accelerate 候选均以**独立实现**为准，并以其官方公开 API 为依据。

---

## 13. 真实短训预算与预登记数值门槛

### V-1 三次运行，总计 50 次有效更新

| 运行 | 卡数 | 步数 | 有效更新 | 用途 |
| --- | --- | --- | --- | --- |
| **RUN-A** | 1 | 10 | 10 | 前 10 步 global-batch 对照基准 |
| **RUN-B** | 2 | 20 | 20 | 双卡连续 20 步 |
| **RUN-C** | 2 | 10 + 10 | 20 | 前 10 步后保存，恢复续 10 步 |
| 合计 | — | — | **50** | 超出即违约 |

硬限制：单次运行 **≤ 20 分钟**（`timeout 1200`），全部真实训练 **≤ 80 分钟**，单次 ≤ 20 步。
资源不足 → 报告并停止，**不延长、不扩为长训**。

### V-2 对照成立的前提（必须同名冻结）

1. **同一原始 init**：三次运行从同一 adapter 初始化（关闭随机初始化差异：固定 `seed` + 记录初始参数指纹）。
2. **同一份数据与顺序**：同一 subset 列表、同一 `num_repeats`、`--max_data_loader_n_workers=0`
   （保证 `skip_first_batches` 可精确重放）、关闭 `caption_dropout`/`shuffle` 类随机。
3. **同一 global batch 组成**：RUN-A 的每一步消费的样本 multiset == RUN-B/RUN-C 对应窗口消费的样本 multiset。
   做法：把 60（或 57）个样本按固定顺序编号，窗口 `k` 的两个样本固定为 `(2k, 2k+1)`；
   双卡每 rank 各 1 个样本（rank0 取 `2k`、rank1 取 `2k+1`），单卡 `batch_size=2` 取同一对。
4. **同一 per-sample 随机输入（硬前提，不是可选增强）**：同一 `sample_id` 在三次运行中必须拿到
   逐位相同的 `(noise, timestep)`（以及由它们决定的 target）。**当前实现下这一点不自动成立**：
   noise/timestep 由全局 RNG **按 batch 形状**抽取（`library/training/noise_target.py:66`
   `torch.randn_like(latents)`；`library/runtime/noise.py:104-116` 的
   `torch.rand((bsz,), device=device)`），`batch_size=2` 与两次 `batch_size=1` 不是同一随机序列；
   本树也没有 rank 分层 RNG（`library/training/rng_state.py:30-49` 只存/取全局 CPU/CUDA/python 状态，
   `library/training/samplers.py:42` 只包装 noise 契约）。
   ⇒ 单双卡数值对照**必须**经由一个显式的 **per-sample 随机输入 fixture/debug 入口**进行：
   fixture 按 `sample_id` 提供固定的 `(noise, timestep)` 注入点（测试侧 monkeypatch 或既有 sampler
   契约上的 debug 入口），**不改变生产默认行为**。没有该入口时，单双卡数值等价**不得判定**
   （只能用组 R 的 rank 一致性）。fixture 的位置、注入点与逐样本取值必须落进 traces。
5. **同一 shape / 无 padding**：每个窗口内两 rank 的样本 token 数完全相等。
   原因 [A]：Anima loss 是逐样本 `mean` 后按窗口平均（`library/training/losses.py:626`
   `weight * loss.mean()`），DDP 侧是"每 rank 梯度逐元素 `mean` 归约"
   （`library/training/gradient_sync.py:199-201`，`accelerator.reduce(..., "mean")`）；只有每 rank
   元素数相等时"均值之均值"才等于"全局均值"。**跨桶配对在数学上就不等价**，此时任何数值接近都是巧合。
   路径 3（固定 4096-token）天然满足；若改用 canonical 桶数据集，则两 rank 必须同桶。
   不引入 pad-to-static，不改 `CONSTANT_TOKEN_BUCKETS`，不做 intra-bucket padding。
6. **单卡运行用 batch_size=2**：RUN-A 必须一次消费同一对样本（`(2k, 2k+1)`），因此需要一个
   `batch_size=2` 的 blueprint 副本写入唯一新输出根（**不写 `krea2-webui`**）；双卡运行用
   `batch_size=1` + `num_processes=2`。两次运行的 per-window 样本 multiset 必须逐窗相等。

> **⚠ 已被 v2 取代（2026-11-08）——本节两条判据失效，原文保留不改写**：
> 本节的 **S-d**（**v1 版第 804 行**／插入本标注后当前第 813 行，原文：「fp32：grad rel-L2 ≤ 1e-6、Δθ rel-L2 ≤ 1e-6、
> `max|Δθ| ≤ 1e-7·max|θ|`」）与 **S-g**（「**第 20 步参数逐位相等**」，v1 版第 807–809 行／当前第 816–818 行）已被
> [修订 v2：数值门替换](multigpu_reference_validation_contract_amendment_v2.md) 的 **C1–C4** 取代。
> 取代原因：这两条等价于要求「单卡 `batch_size=2` 一次归约」与「双卡 `1+1` 再 all-reduce 求均值」
> 两种求和顺序**逐位相同**，而 fp32 加法不满足结合律 ⇒ 数学上不可达（实测：S-d `1.5055e-04`、
> S-g `3.694e-05`；同一张卡 fp32 同配置重复跑本身就有 `5.634e-08` 级非确定性）。
> **S-a / S-b / S-c / S-e / S-f 及其余全部条款继续按原文有效**；修订记录见本文件末尾 **§19**。

### V-3 预登记数值门槛（观测前固定；分两组，不混用）

**组 R（rank 一致性，同 world_size=2）——强门槛，逐位**

| ID | 判据 | 门槛 |
| --- | --- | --- |
| R-a | 每 commit 边界，全部 adapter 参数逐 rank | `torch.equal == True`（V-GLOO 明确；V-REAL 同样要求，若实测失败即缺口成立） |
| R-b | `GradScaler` state（scale、growth tracker）逐 rank | 完全相等 |
| R-c | `optimizer` step 计数、`param_group` lr、`global_step`、`window_*` 计数逐 rank | 完全相等 |
| R-d | 跳过/重放/丢弃计数逐 rank | 完全相等（含"全 0"这一情形必须被显式记录） |
| R-e | 不以 rank0 代表全体 | 任一侧观测到的坏状态必须被全体同分支处理（故障注入观测） |

**组 S（single vs dual）——同 per-sample 随机输入下的数值门槛**

前提门 **S-0**：§13 V-2 第 4 条的 per-sample 随机输入 fixture 必须已生效并落进 traces。
**未生效 ⇒ 单双卡数值等价直接记 `unverified`，不得判通过**（此时只有组 R 有效）。

判据以 **gradient 与参数更新量**为准，**loss 序列只作观测记录、不作等价判定门**
（10 步内 loss 由单窗口噪声主导，用它判等价既不可靠也会掩盖真正的 rank 缺口）。

| ID | 判据 | 门槛（预登记） |
| --- | --- | --- |
| S-a | 有效更新计数 | 精确相等：RUN-B 前 10 窗 == RUN-A 的 10 步；RUN-C 合计 20 == RUN-B 的 20 |
| S-b | 样本组成 | 窗口 → 样本 id multiset **精确相等** |
| S-c | committed loss 窗口均值序列（前 10 窗） | **仅记录，不设门**（观测项，写进 summary.md 供人看） |
| S-d | **dtype-specific 数值门**：每窗口的 adapter 梯度与 Δθ | fp32（CPU/Gloo、fp32 CUDA）：grad rel-L2 ≤ **1e-6**、Δθ rel-L2 ≤ **1e-6**、`max|Δθ| ≤ 1e-7·max|θ|`；bf16：≤ **1e-3** / ≤ **1e-3** / ≤ **1e-4·max|θ|`；fp16+GradScaler（V100 走此路径）：≤ **2e-3** / ≤ **2e-3** / ≤ **1e-3·max|θ|` |
| S-e | skip 决策序列 | 两轮逐窗口 `update_skipped` 序列必须**完全一致**；不一致则该窗口不计入 S-d，且整轮判 `needs_revision` |
| S-f | 参数增量方向 `cos(Δθ_A, Δθ_B)` | **仅记录，不设门**（10 步内 Δθ 由噪声方向主导，O(1) 差异合法） |
| S-g | 双卡 10+10 续训 vs 连续 20 步 | **第 20 步参数逐位相等**（强门槛）。依据：accelerate `save_state`/`load_state`
  保存并恢复每个 rank 的 RNG 与 optimizer/scheduler（实测 state 目录含 `random_states_0.pkl`），
  `skip_first_batches`（`library/training/loop.py:801-805`）跳批；配 `workers=0` 与固定顺序后应逐位可复现 |

> **S-g 失败的处理**：视为 P0-5/P1-2 缺口成立 → `needs_revision`，**不允许**把"近似相等"当通过，
> 也不允许事后放宽门槛。若失败但可逐项解释为已知仓库限制，须写成 `exact_resume=False` 的显式声明，
> 并仍以 `needs_revision` 收口，由 captain 决定是否接受降级。

> 阈值锚定方式：以各 dtype 的机器精度为锚（fp32 eps≈1.2e-7、bf16 尾数 8 bit≈4e-3、fp16 尾数 10 bit
> ≈1e-3），按"10 步累加 + 单窗口归约"留约 2 倍余量登记；**观测后一律不得调整**，只能由 captain
> 修订 §18 并写明实测值与理由。超阈即 `needs_revision`，不得以"影响很小"放过。

> **零 skip 情形**：若三轮真实运行都未出现 skip/replay/discard，则判据 R-c/R-d 的 scaler 分支**未在 V-REAL
> 覆盖**，必须在报告里写明，并只引用 V-GLOO 证据（§6 W-4）。不得写"真实运行验证了 scaler 对称性"。

### V-4 checkpoint 保存 flag 冻结（配合 §5 D-4）

- RUN-A：不保存 state（只保留最终 adapter + 日志 + 轨迹）。
- RUN-B：只在 **step 20** 保存一次 state（作为"连续运行终态"对照），不按 epoch 存。
- RUN-C：在 **step 10** 与 **step 20** 各保存一次 state。
- 统一关闭：`--save_every_n_steps 0`、`--save_last_n_steps 0`、`--checkpointing_epochs 0`、
  `--save_state_on_train_end false`；`--save_state` 仅在上述边界由运行脚本显式触发。
- 具体 flag 名在实施阶段按 `library/training/cli_args.py:391,415,421,427,432,443` 的实时定义复核；
  若名字漂移，以实时 CLI 为准并在证据里记录实际命令行。

### V-5 输出清单（唯一新根，见 §5 D-1）

`runs/<run>/stdout.log`、`runs/<run>/argv.json`、`runs/<run>/counters.json`（有效/跳过/重放/丢弃）、
`traces/rank{0,1}-<run>.jsonl`（设备/参数指纹/scaler/lr）、`states/`（RUN-B/C）、
`summary.md`（逐项对照 §13 V-3 的判据结果）。**不写第二份散落文件**。

---

## 14. 新增测试接口（本阶段只冻结接口，不实现）

| 文件 | 覆盖 | marker | 默认行为 | 显式启用 |
| --- | --- | --- | --- | --- |
| `tests/test_multigpu_window_protocol.py` | §6 W-4 的 S①–S④ 四场景 × 四项比对（2 进程 Gloo）；**并含 OR 原语回归**（§7.1 N-3：两 rank 均坏时归约值为 2 且判定与 world_size 无关）；复用 `tests/window_replay_harness.py` | `integration` | 只跑 CPU/Gloo，不占 GPU | 无 |
| `tests/test_multigpu_admission.py` | P0-1..P0-5 的准入/原因码/清单字段/world_size 传递；P0-1 覆盖四种：默认推断 / `<` 拒绝 / **`>` 拒绝** / 单卡=1 不变（含 launcher 同码） | 无（纯单测，进 core 层） | 默认在 core 层执行 | 无 |
| `tests/test_multigpu_checkpoint_protocol.py` | P1-1 三阶段顺序、rank0-only rename、体积门、N-1/N-2 回归 | `integration` | CPU/Gloo | 无 |
| `tests/test_multigpu_resume_protocol.py` | P1-2 bundle 版本/拓扑校验、legacy 降级、跨 world_size 拒绝 | `integration` | CPU/Gloo | 无 |
| `tests/test_multigpu_reference_experiments.py` | 小模型 2 卡 GPU 参考实验（形状/预算/收敛烟测），不加载 Anima 基座 | `integration` + `hardware` | **skip**（不占 GPU） | `MULTIGPU_REFERENCE_HARDWARE=1` |
| `tests/test_multigpu_reference_hot.py` | 真实 Anima 有界短训（RUN-A/B/C 的脚本化版本或最小等价） | `integration` + `hardware` | **skip** | `MULTIGPU_REFERENCE_HOT=1` |

约束：

1. 复用既有 marker（`integration`、`hardware`），不新增 marker（`pyproject.toml:97-104` 的现行清单）。
2. `hardware` 已在分层选择里被排除：`scripts/tasks/utilities.py:21`
   `CORE_TEST_MARKERS = "not integration and not hardware and not benchmark and not probe"`。
   **但单独 `pytest tests/test_x.py` 仍会执行**，所以 GPU 用例必须**再加环境变量门**：
   未设 `MULTIGPU_REFERENCE_HARDWARE=1` / `MULTIGPU_REFERENCE_HOT=1` 时 `pytest.skip`。
3. 每个新增测试文件 **≤ 1200 行**（AGENTS.md 反上帝代码守则）；超出即拆分。
4. GPU 用例只允许由**唯一 GPU 执行者**按**已批准计划**运行（显式环境变量 + §3 E-5 + §5 D-5
   即受控执行，不设二次审批门）；默认测试套件不得占卡。
5. `test_multigpu_reference_hot.py` 不实现"下载模型/预处理"，只消费 §4 冻结的只读资产；
   资产缺失时 `pytest.skip` 并打印**具名缺口**，禁止用 toy 数据冒充。

---

## 15. 可执行清单（环境 / 数据 / 命令）

### C-1 环境（每次会话开头，只读）

```bash
export PY=/mnt/SD_Dataset/训练器相关/krea2-webui/.venv-v100/bin/python
export R=/mnt/SD_Dataset/训练器相关/krea2-multigpu
CUDA_VISIBLE_DEVICES= $PY -c "import sys,torch,accelerate;print(sys.version.split()[0],torch.__version__,accelerate.__version__)"
# 期望：3.13.9 2.10.0+cu129 1.13.0；不一致即回到 §3 E-2 重新登记漂移
```

### C-2 数据/资产（只读存在性检查）

```bash
stat -c '%s %n' \
  /mnt/SD_Dataset/训练器测试/comfyui/models/diffusion_models/anima/anima-base-v1.0.safetensors \
  /mnt/SD_Dataset/训练器相关/anima_lora/models/text_encoders/qwen_3_06b_base.safetensors \
  /mnt/SD_Dataset/训练器相关/krea2-webui/models/vae/qwen_image_vae.safetensors
ls /mnt/SD_Dataset/训练器相关/krea2-webui/post_image_dataset/lora | grep -c '_anima\.npz$'          # 期望 60
ls /mnt/SD_Dataset/训练器相关/krea2-webui/post_image_dataset/lora | grep -c '_anima_te\.'           # 期望 0 → 具名缺口 §4 A-3
ls /mnt/SD_Dataset/训练器相关/anima_lora/post_image_dataset/rokkotsu_goddess | grep -c '_anima_te\.'  # 期望 57（备选路径）
```

### C-3 CPU/Gloo 回归地板（本任务已执行，见 §3 E-4）

```bash
cd $R
CUDA_VISIBLE_DEVICES= PYTHONPATH=$R timeout 60  $PY -m pytest tests/test_window_replay.py -q
CUDA_VISIBLE_DEVICES= PYTHONPATH=$R timeout 120 $PY -m pytest tests/test_distributed_gradient_sync.py -q
CUDA_VISIBLE_DEVICES= PYTHONPATH=$R timeout 120 $PY -m pytest tests/test_distributed_sampling.py -q
CUDA_VISIBLE_DEVICES= PYTHONPATH=$R timeout 120 $PY -m pytest tests/test_fp16_nan_static_equivalence.py -q
CUDA_VISIBLE_DEVICES= PYTHONPATH=$R timeout 120 $PY -m pytest tests/test_training_checkpointing.py -q
```

### C-4 分层测试（实施后启用，按 marker 选择）

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=$R timeout 120 $PY -m pytest tests/test_multigpu_window_protocol.py -q
CUDA_VISIBLE_DEVICES= PYTHONPATH=$R timeout  60 $PY -m pytest tests/test_multigpu_admission.py -q
CUDA_VISIBLE_DEVICES= PYTHONPATH=$R timeout 120 $PY -m pytest tests/test_multigpu_checkpoint_protocol.py -q
CUDA_VISIBLE_DEVICES= PYTHONPATH=$R timeout 120 $PY -m pytest tests/test_multigpu_resume_protocol.py -q
# GPU 层级：必须显式门 + captain 放行
MULTIGPU_REFERENCE_HARDWARE=1 PYTHONPATH=$R timeout 600 $PY -m pytest tests/test_multigpu_reference_experiments.py -q
MULTIGPU_REFERENCE_HOT=1      PYTHONPATH=$R timeout 1200 $PY -m pytest tests/test_multigpu_reference_hot.py -q
```

### C-5 真实短训（受控执行：显式环境变量 + E-5 + D-5；三次合计 50 更新）

多卡命令前缀由现有 launcher 构造（`library/runtime/launch.py:80-102`）：
`$PY -m accelerate.commands.accelerate_cli launch --num_processes <N> --num_machines 1 --dynamo_backend no --num_cpu_threads_per_process 3 --mixed_precision <mp> train.py …`。
参数取值、seed、样本顺序与 save flag 按 §13 冻结；每次运行前执行 §5 D-5 磁盘门，运行包 `timeout 1200`。
**命令原文与实际 argv 必须写入 `output/tests/multigpu_reference_validation/<run-id>/runs/<run>/argv.json`。**

---

## 16. 责任阶段映射（与团队 DAG 对齐）

| DAG 阶段 | 本契约中必须交付/判定的项 |
| --- | --- |
| 需求与环境契约（本阶段） | §3、§4、§5、§6 W-1..W-5、§7、§8、§9、§10、§13、§14 |
| DDP 核心实现 | P0-0（W-3 六条，含 N-3 的 OR 原语改写）、P0-1、P0-2、P0-3、P0-4 |
| CPU/Gloo 独立验证 | §6 W-4 四场景、P0-1..P0-4 单测、§11 R-1/R-2 回归地板 |
| 核心 review | W-3 六条逐条核对；R-3 三个已撤回项不得复活 |
| checkpoint/恢复与隔离实验实现 | P0-5、P1-1（含 N-1/N-2）、P1-2、EXP-FSDP2-BASE、EXP-TP-COLROW 的最小验证 |
| 分层 CPU/Gloo/CUDA 验证 | §14 六个文件的四层观测；§9 两类故障注入 |
| Anima 真实有界短训 | §13 全部（50 更新、预登记门槛、唯一输出根） |
| 总体独立 review | §13 V-3 逐项结果 + P1-3 账本 + 所有 `unverified` 项的显式清单 |
| 集成交付 | 只交付通过件；未通过项保持 fail-closed，不写默认值/UI 提示 |

**串行约束**：仅一位实现者改代码；验证阶段仅一位 GPU 执行者；禁止共享工作树并行写入。
---

## 17. 门禁、修订与不成立项

### G-1 本契约的放行门

1. 任何 V-CUDA-SMALL / V-REAL 的进入条件 = §3 E-5 五条 + §5 D-5 磁盘门 + 显式
   `MULTIGPU_REFERENCE_HARDWARE=1` / `MULTIGPU_REFERENCE_HOT=1`；GPU 许可以已批准计划为准，
   **不设二次审批门**。资源不足时按团队约定报告并停止，不延长、不扩为长训。
2. 任何依赖变更（装包/升级/`uv sync`）需 captain 同意；`needs_revision` 由实现者修，不由验证者改代码。
3. 预登记阈值（§13 V-3）**观测后不得调整**；确需调整只能由 captain 修订本文件并记录理由与时间。

### G-2 修订流程

- 只有 captain 可以 amend 本契约；amend 必须新增"修订记录"小节，写明：改了什么、为什么、
  原判据的实测结果（若有）、谁批准。
- 源码事实变化（行号/行为）不构成 amend 理由，但**必须在下次使用前重新核对并记录**。

### G-3 明确不成立 / 未验证清单（本阶段诚实边界）

| 项 | 状态 |
| --- | --- |
| "P0-0 已修复" | **未成立**：本阶段无任何源码修改 |
| "多卡精确恢复已具备" | **未成立**：无 world_size 校验、无 per-rank bundle |
| "`krea2-webui` 全树没有任何 Anima TE cache" | **不成立（已更正）**：全树实测 92 个 `*_anima_te.safetensors`，**全部在 `output/**`，`output/` 之外为 0**；按 8 个根计入 §4 A-3 |
| `rokkotsu_goddess` 的完整 cache 可直接用于 V-REAL | **不成立**：`image_dir` 未解析，且 `library/datasets/dreambooth.py:132,170` 要求 image_dir 存在（cache-only 不可行） |
| `krea2-webui` 侧 latent/图像前置完整、可作 TE 预处理起点 | **成立**（60 图 + 60 caption + 60 npz 实测齐备）；但 TE 预处理**不在用户已授权范围**，只能是候选选项，**不得**把既有"短训练"授权改写成预处理授权（§4 A-3.1） |
| `output/ab-anima-20261002` 可否作 V-REAL 数据集 | **仅资产侧候选（L1，无需预处理）**：60 TE + 60 npz + 60 img + 60 caption 配对完整、路径重基后可用；证据只覆盖**固定 4096-token 单 shape**（L2），不外推 canonical 4032/4200、跨桶或尾窗；**RUN-A/B/C 仍未解锁**，须先满足 S-0/S-d/S-e/S-g 且经 t6→t7（L3，§4 A-3.1） |
| CUDA/NCCL 任何行为 | **未观测**（本轮只用 CPU；GPU 为空闲但未使用） |
| FSDP2/TP 收益 | **未验证**（不得量化） |
| 依赖内部行为结论 | **C 级**，仅对 torch 2.10.0 / accelerate 1.13.0 成立 |
| DeepSpeed / SP / PP | **不适用/不迁移**，只保留拒绝与回归证据 |
| 参考仓双进程 Gloo 历史通过计数 | **不可复核**（脚本已删除），不得引用其计数 |

---

## 18. 修订记录

| 日期 | 版本 | 内容 | 批准 |
| --- | --- | --- | --- |
| 2026-11-07 | v1 | 首版冻结：E/A/D/W/P0/P1/EXP 全套契约、50 更新预算、预登记阈值、六个新测试接口 | requirements 阶段冻结（已被 v1.1/v1.2 补充，现行内容以最新行为准） |
| 2026-11-07 | v1.1 | **非 amend 性补充**（未改任何判据、阈值、预算与采用/否决条件）：①§3 E-4 增记 attempt 6 独立复跑，110 passed 完全复现 → 回归地板有两次观测；②新增 §4 A-3.1，把"V-REAL 资产可用路径"从"择一待批"收窄为结论：路径 1 前置齐备但需用户明确批准有界 TE 预处理（不在已授权范围），路径 2 因 `image_dir` 未解析而不可执行（依据 `library/datasets/dreambooth.py:132,170` 的强制枚举依赖）；③§17 G-3 补两行资产状态；④E-1/E-2/W-1/W-4/W-8/N-1/N-2/§14 marker 全部逐条复核无漂移（行号仍准确） | requirements 阶段自检，**不替代 captain amend** |
| 2026-11-07 | v1.2 | **定向修订（仅修下列 5 项，未改 50 更新预算、磁盘预算、P0/P1/EXP 采用否决条件、E-1/E-2 环境结论）**：①§4 A-3 把"全树 `*_anima_te` = 0"改为实测口径——92 个全在 `output/**`、`output/` 之外 0，按 8 个根列数，并声明"文件总数 ≠ 成对完整 ≠ 与 lora 60 张匹配"；②新增 A-3 路径 3（`output/ab-anima-20261002`，60 TE+60 npz+60 img+60 caption，路径重基即可、**无预处理**）与 1024x1024→**4096 token** 固定 shape 的源码核算表，明确它是**固定 shape 单独对照、非 canonical 4032/4200 桶结论**，且**不解锁 RUN-A/B/C**；③§5 D-1/§15 C-5 输出根统一为 `output/tests/multigpu_reference_validation/<run-id>/` 并写明"一次任务一个 run-id、不新增第二个根"；④§13 V-2 把"同一 per-sample 随机输入"升为硬前提并新增 fixture/debug 入口要求与单卡 `batch_size=2` 要求，V-3 组 S 删除 loss 百分比门（loss 仅观测）、改为 dtype-specific grad/Δθ 数值门 + skip 序列门，续训逐位门重编号 S-g；⑤删除 GPU 层二次放行与首版遗留的等待确认措辞（许可来自已批准计划 + E-5 + D-5 + 显式环境变量），预处理表述改为"需用户明确批准、不在已授权范围" | requirements r3 定向修订；**修订前指纹**：contract `sha256:a47ac461cc1e3a7f313f3566c8c4d2d4e0766831a9a853eef50f696fafe83f58`、README `sha256:3c2cc7f21ae1ef02e63f9c8243e50c8601a897d21184d7a0b94a5f63a0ca137f` |
| 2026-11-07 | v1.4 | **定向修订（只消歧措辞，不改 P0/P1 判据、数值容差、50 更新预算、授权边界）**：①§4 A-3.1 删除旧的"资产解阻 + 路径 3 直接执行 §13"表述，改为 **L1/L2/L3 三层区分**——L1 资产只读可枚举可读（无预处理）、L2 仅固定 4096-token 单 shape 证据（不外推 canonical 4032/4200、跨桶、未测尾窗；source/seed/样本数仅限本次单独 run）、L3 RUN-A/B/C 仍须 S-0 fixture + 单卡 batch2 / 双卡 batch1 + 组 R + S-d/S-e/S-g，任何候选不自动 unlock，预算执行须走已批准计划 + E-5 + D-5 + t6→t7；②§4 A-3 冻结规则与 §17 G-3 对应行同步改为"资产候选、非 RUN 解锁"；③§7 P0-1 行按 §7.3-1 已冻结口径补齐双向措辞（`num_processes != len(selection)`，`<`/`>` 均具名拒绝、同卡多 worker 不豁免，测试面四种）；④§10 与 §14 对应行同步。 | requirements t13 定向修订；**修订前指纹**：contract `sha256:2c355a491daafa0e51914db7b0afd95eebc70a13ecf0ea2a0bbaa3718c26f157`、README `sha256:3c2cc7f21ae1ef02e63f9c8243e50c8601a897d21184d7a0b94a5f63a0ca137f` |
| 2026-11-07 | v1.3 | **实现态记录（t14 repair-round-2，非 amend：未改判据/阈值/预算/采用否决条件）**：①新增 §7.3，记录 P0-1 已落地的「选卡↔进程数冲突两方向一律拒绝」口径与两方向原因码，明确不存在『显式值优先放行』例外；②记录 W-3.1 原语回归要求（两 rank 全 1 → SUM=2.0 且 OR=True，判定与 world_size 无关）；③明确 `window failure reason` 可 rank-local，但判定（drop/commit/discard）与计数必须逐 rank 一致，避免后续 review 误判 | t3 验证 findings + captain 确认按选择①修 |
| 2026-11-07 | v1.4 | **实现态记录（t16 repair-round-2，非 amend：未改判据/阈值/预算/采用否决条件）**：①§7.3 补 4–7 条：identity 写失败为具名失败 + rank0 全 rank 视图 + configured/real world_size 分列 + bucket/reduction 由真实 GradientSyncResult 回填；②边界局部异常的类 I 协商与 transport 类 II 退出边界；③删除虚假 micro-batch 计数字段；④WORLD_SIZE 权威优先（含小于 admission 值的场景） | t4 review findings 修复 |

## 19. 修订记录 v2（数值门替换，2026-11-08）

> 本节是**追加记录**，不修改也不删除本文件任何旧文。上面 §13 V-3 的 S-d / S-g 判据原文仍在原处
> （仅在该节标题上方加了一条「已被 v2 取代」的标注）。

| 项 | 内容 |
| --- | --- |
| **被修订的 v1 指纹** | `sha256:80dfdb02c4fb81e1cc4e798560aeb5533a3ca09f1ade855c3c7ae1b0a67d76a2`（981 行 / 79,443 B） |
| **v2 生效时间** | **2026-11-08**（修订文档落盘即生效；t46 派单 12:34:52 +08:00） |
| **修订文档** | [多卡全流程参考点验收契约 · 修订 v2（数值门替换）](multigpu_reference_validation_contract_amendment_v2.md) —— **v2 判据以该文件为准** |
| **被取代条款（准确位置）** | ① **§13 V-3 S-d**，本文件 **v1 版第 804 行**行首（§19 追加后当前为第 813 行）：「fp32（CPU/Gloo、fp32 CUDA）：grad rel-L2 ≤ **1e-6**、Δθ rel-L2 ≤ **1e-6**、`max\|Δθ\| ≤ 1e-7·max\|θ\|`」（bf16 ≤1e-3 与 fp16+GradScaler ≤2e-3 两档同时失效）；② **§13 V-3 S-g**，本文件 **v1 版第 807–809 行**（当前第 816–818 行）：「**第 20 步参数逐位相等**（强门槛）」 |
| **替换为** | C1 数据对齐门（同 slot stem 全等 + `k*W+rank` + 无重复）、C2 步级梯度 ULP 门（**唯一判据**：`max\|ΔG\| ≤ 4 ULP(max\|G\|)`；"逐位相同的张量计数"仅为观测项、不作判据）、C3 权重漂移 ≤ fp16 同配置实测噪声地板（t17：2.040e-04 / 3.225e-04 / 3.988e-04）、C4 恢复两段（续接逐位精确 + 20 步散度 ≤ 同一噪声地板，附曲线/首超步号/epoch 边界） |
| **取代原因** | 原两条判据等价于要求两种求和/归约顺序**逐位相同**，fp32 加法不满足结合律 ⇒ 数学上不可达；且 `1e-6` 在该口径下无推导、无实测依据（受版本控制的文档与源码中只出现于本文件第 804 行一处） |
| **原判据实测结果** | S-d（pinned 后首次有效测量）`1.5055e-04`（max_abs `2.3414e-04`）；S-g `3.694e-05`（step11 `1.602e-09` → step15 首超 `1e-6` → step20 `3.694e-05`，未超 `1e-4`）；前提门全过、续接逐位精确。详见修订文档 §4–§5 |
| **谁批准** | **用户**，2026-11-08，原话：「现在就收口，别再往工装上加码，按照你的a倾向来」（A 倾向 = `pinned_order` 保留在产品中并正名为「可复现的确定性数据顺序」，视为产品特性而非通过合同用的补丁）。批准记录全文见修订文档 §6 |
| **未解除项** | 其余全部条款（P0-0..P0-4、P1-1、P1-3、组 R、S-a/S-b/S-c/S-e/S-f、预算、资产、环境、磁盘门、采用/否决条件）继续按 v1 有效；v2 不证明"多卡与单卡等价" |
