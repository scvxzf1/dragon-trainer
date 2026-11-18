# AGENTS.md

本文件是 AI Agent 维护本仓库的根级工作协议，覆盖整个仓库；子目录若有 `AGENTS.md` 或 `CLAUDE.md`（如 [`networks/CLAUDE.md`](networks/CLAUDE.md)、[`custom_nodes/comfyui-hydralora/CLAUDE.md`](custom_nodes/comfyui-hydralora/CLAUDE.md)），作为离目标文件更近的补充约束。根级 [`CLAUDE.md`](CLAUDE.md) 仅为指向本文件的入口重定向。**代码、配置与测试始终优先于文档**；发现文档与实时源码不一致时，先核对源码与测试，再修正文档。

## 总体原则与数据安全

- 默认使用简体中文沟通；代码标识符、命令、配置键、报错文案保持项目既有语言风格。
- 先定位子系统，再读取最小必要上下文；搜索优先使用 `rg` / `rg --files`。
- 保持改动聚焦准确。禁止顺手重构、重排大文件、格式化无关文件或改写历史实验记录。
- **严禁破坏用户运行数据**：不得把 `.venv/`、`.venv-v100/`、`models/`、`output/`、`post_image_dataset/`、`image_dataset/`、`logs/`、`configs/imported/`、`configs/web-training-history/`、`configs/web-training-queue/`、`configs/.restore-backups/` 当作源码清理或覆盖。
- 未经用户明确同意，不得终止正在运行的训练、清空队列、批量移动输出、下载大模型、启动长训练或修改显卡功耗/风扇/频率设置。
- 遇到未提交改动或其他本地 worktree/分支，默认是用户或其他 Agent 的工作现场；不得擅自 `revert`、`reset`、`clean` 或跨工作树覆盖。

## Graphify 按需导航

- 跨文件调用链、模块依赖或影响面分析时，可按需查询本地 `graphify-out/graph.json`（若存在）；已知位置修改、单文件阅读和简单搜索直接用 `rg` 与源码。
- 图谱仅作线索，不是代码事实源，关键关系必须核对当前源码；图谱缺失或过期时直接回退源码，不自动重建。
- 调用时必须设置 `GRAPHIFY_QUERY_LOG_DISABLE=1 GRAPHIFY_NO_AUTO_REFRESH=1`；`graphify-out/` 为本地产物，不暂存提交。未经额外授权不启用外部语义 API、服务或 watcher。

## 子代理委派与并发工作树

- 主代理可在用户授权任务范围内自主委派探索、推导、实现、测试与独立验收；子代理通过自包含任务说明交接（不依赖隐式上下文）。
- 实现委派须明确文件归属、接口契约与验收条件。**多个写代理严禁共用同一工作树和分支并行写入**，必须使用 `.worktrees/` 独立 git worktree 隔离后由主代理合入。
- 测试与独立验收允许异步推进；仅在后续决策依赖结果、命中关键风险或准备发布时阻塞等待，普通开发交付明确区分“实现完成”与“待验收项”。

## 反上帝代码守则

- 热点编排文件默认只做 facade、分派、兼容 shim 或小范围修复，不堆砌新业务逻辑：
  `train.py`、`inference.py`、`library/inference/generation.py`、`library/datasets/base.py`、`networks/lora_anima/{network,config}.py`、`web/services/training_service.py`、`web/services/config/_legacy.py`。
- 修改热点文件超过约 50 行时优先拆入子模块；确需保留在热点文件时，须在回复中说明原因、后续瘦身路径与检查范围。
- 规模参考上限：单个新增 Python 函数 ≤ 100 行、新增类 ≤ 400 行、新增 JS/TS 函数 ≤ 80 行、单个 React 组件 ≤ 200–300 行、单个新增测试文件 ≤ 1200 行；已超 1000 行的文件默认不再承载新业务。
- 重构优先“搬家型重构”（先等价抽模块，再改行为），不在同一轮混杂大规模架构搬迁与数值变更。
- 新增配置键、CLI 参数、模型族能力或 WebUI 字段时，同步核对 schema、预检矩阵、帮助分册、文档索引与兼容面。

## 环境和命令入口

- 运行环境为 Python 3.13 + `uv`（V100 专用环境见 `./setup-v100-linux.sh`）；部署说明以 [根 README](README.md) 与 [Linux 部署指南](docs/guidelines/linux-deployment.zh.md) 为准。
- [`tasks.py`](tasks.py) 是唯一命令注册真相（`Makefile` 仅作薄转发）；实时命令面通过 `.venv/bin/python tasks.py --help` 查询，实现位于 [`scripts/tasks/`](scripts/tasks/) 与 [`scripts/experimental_tasks/`](scripts/experimental_tasks/)。
- 本仓维护执行优先使用 `.venv/bin/python`；用户文档主入口写 `python tasks.py <cmd>` 或 `.venv/bin/python tasks.py <cmd>`（`make <target>` 仅作兼容说明）。支持尾随 `KEY=value`（如 `python tasks.py print-config METHOD=lora PRESET=default`）。
- 常用入口（按需选用，非每次固定步骤）：
  - WebUI：`.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102`（或一键构建启动 `./webui.sh`，默认 `20203`）
  - Next 前端检查 / 构建：`.venv/bin/python tasks.py web-next-check` / `web-next-build` / `web-next-e2e`
  - 定向单测：`timeout 60 .venv/bin/python -m pytest tests/<test_file>.py -q`
  - 分层测试：`test-fast`、`test-core`、`test-integration`、`test-hardware`、`test-experimental`、`test-all`

## Git 推送和回滚

- 线上主仓由 URL `github.com/scvxzf1/dragon-trainer` 唯一确定（操作前先 `git remote -v` 核实本机别名，不假定必叫 `origin`）。`main` 为发布分支，`dev` 为日常集成分支；指向上游 `sorryhyun/anima_lora` 的 `upstream` 只 fetch 不 push（移植提交标 `port(upstream): ...`，血缘规范见 [`docs/project-lineage.md`](docs/project-lineage.md)）。
- 日常独立开发在干净工作树同步 `dev` 后创建 `codex/<task>` 分支；脏工作树先保护现场并在 `.worktrees/` 隔离开发，不得擅自 `switch` / `pull` / `rebase` / `stash`。详细流程见 [`docs/guidelines/git-sync-policy.md`](docs/guidelines/git-sync-policy.md)。
- 暂存必须使用**明确路径白名单**，严禁 `git add -A` 带入本机路径、`.anima-webui-settings.toml`、凭证、历史/队列或运行输出。
- 未指定分支的线上推送默认 `<target-remote> dev:dev`；仅明确要求发布时才推 `main`（须经 review 与对应风险验收）。
- `git reset --hard`、`git checkout -- <path>`、`git clean -fd`、`git push --force*` 均为高风险操作，未经用户明确指定目标与确认影响严禁执行。

## 项目地图

- `tasks.py` / `scripts/tasks/` / `scripts/experimental_tasks/`：CLI 任务注册表与实现。
- `train.py` / `library/training/`：统一训练入口与编排（bootstrap、loop、compat_matrix、optimizer、checkpoint、loss、adaptive_runtime、multigpu）。
- `library/training/pp/`：多模型共享双卡 1F1B 流水线并行（PP v0）runtime、门禁（`gate.py`）、族适配器与分 rank checkpoint。
- `inference.py` / `library/inference/`：独立推理入口、采样、adapter 加载、Spectrum、DCW、SMC-CFG、DirectEdit。
- `library/models/family_registry.py`：**单一模型族事实源**（当前注册 5 族：`anima`、`krea2_raw`、`z_image`、`qwen_image_2_1`、`moss_image`），完整边界以 [`docs/multi_model_support.md`](docs/multi_model_support.md) 为准：
  - `library/anima/`：Anima DiT、权重加载、max-padded Qwen3/T5 文本策略与编译。
  - `library/models/krea2_raw/`：Krea-2 single-stream MMDiT、BF16/NF4（含 `sm<80` 内层 FP16）加载、fixed resident compile、训练与 single/euler 推理。
  - `library/models/z_image/`：Z-Image Diffusers transformer、Flux VAE 归一化、Qwen3 文本缓存、block swap、训练与训练预览。
  - `library/models/qwen_image_2_1/`：Qwen Image 2.1 T2I/Edit 4D 训练、专属 64ch VAE / Qwen3-VL 缓存、block compile、训练预览与 single/euler 独立推理。
  - `library/models/moss_image/` & `library/training/moss_compat.py`：MOSS-Image 42 层 4D 训练、BF16/NF4 真实长度文本缓存、训练预览与专属契约门禁。
- `library/config/` & `library/env.py`：TOML 合并链、schema 校验与外置配置根解析。
- `library/preprocess/` & `preprocess/`：resize、VAE/TE/PE sidecar 缓存与打标预处理。
- `networks/`：LoRA 三轴路由家族、`plugins/`（LoHa/LoKr/VeRA/GLoRA）与 `methods/`（IP-Adapter/EasyControl/Soft Tokens/BYG 等）；修改前必读 [`networks/CLAUDE.md`](networks/CLAUDE.md)。
- `web/`：aiohttp 后端（`routes/`、`services/`）与 `frontend-next/` React/Vite 工作台（构建发布至 `web/static/dragon-next/`）。
- `custom_nodes/`：ComfyUI 节点（`_vendor/` 由 `python tasks.py vendor-sync` 从 live 源码生成）。
- `configs/`、`tests/`、`bench/`、`docs/`、`_archive/docs/`：配置、测试、基准实验、活跃文档树与历史归档。

## 配置与缓存约定

- **外置配置边界**（[`library/env.py`](library/env.py)、[`docs/configuration/external-configs.md`](docs/configuration/external-configs.md)）：相对路径锚定项目根，允许绝对路径/`$HOME`/`~`，拒绝 `..`；本机路径覆盖只写已忽略的 `.anima-webui-settings.toml`。
- **训练配置合并链**：`configs/base.toml -> configs/presets.toml[<preset>] -> configs/<methods_subdir>/<method>.toml -> CLI args`（默认 `methods_subdir="methods"`，WebUI 自包含变体用 `gui-methods`）。
- **按族隔离的缓存 sidecar**（严禁跨族混用或改名复用）：
  - Anima：`{stem}_{WxH}_anima.npz`、`{stem}_anima_te.safetensors`、`{stem}_anima_pe.safetensors`。
  - Krea-2：复用 `_anima.npz` latent 空间 + 独立 `{stem}_krea2_te.safetensors`。
  - Z-Image：`{stem}_{WxH}_z_image.npz`、`{stem}_z_image_te.safetensors`。
  - Qwen Image 2.1：`{stem}_{WxH}_qwen_image_2_1.npz`、`{stem}_qwen_image_2_1_te.safetensors`。
  - MOSS-Image：`{stem}_{WxH}_moss_image.npz`、`{stem}_moss_image_te_nf4.safetensors`（或 `_bf16.safetensors`，schema 含 `max_token_length`）。

## 不可破坏的不变量

1. **Anima Text Encoder Padding**：Anima 依赖 max-padded 文本输出作为 cross-attention softmax sinks；严禁按有效长度裁剪或通过 `crossattn_seqlens` mask 掉 padding（注：MOSS-Image 相反，保存真实长度并绑定 `max_token_length`）。
2. **Constant Token Buckets 与各族编译隔离**：`library/datasets/buckets.py::CONSTANT_TOKEN_BUCKETS` 使用 4032/4200 原生无内填充桶；不要恢复旧 4096 静态填充。Anima 使用 union dynamic-seq flatten（`compile_seq_bands` 默认关闭）；Krea-2 仅编译 resident block `_forward`（fixed seq + default mode）；Qwen Image 2.1 独立编译 block inner forward；Z-Image 与 MOSS-Image 禁止 `torch_compile`。
3. **Preprocess-First 与分阶段显存隔离**：`train.py`（`library/training/train_session.py`）强制要求训练前由 `preprocess` 完成磁盘 latent/TE 缓存（缺失直接报错）；常规缓存齐备且无采样/CMMD 时训练进程不加载 TE/VAE。预处理与采样解码严禁让 TE + VAE + DiT 同时常驻显存。
4. **Compile After Apply**：支持编译的模型族必须在 `network.apply_to` 及可选权重加载**之后**再调用 `compile_blocks_for_training()` / `compile_qwen_image_2_1_blocks()` / `build_anima`，严禁在脚本或 bench 中倒置顺序。
5. **模型族 Fail-Closed 分派**：`MODEL_FAMILY_REGISTRY` 与 `dispatch_model_family()` 对未知 family、缺失 handler、未注册推理 mode/sampler 或跨族 `ss_model_family` adapter 一律报错，严禁静默回退 Anima。各族能力边界见 [`docs/multi_model_support.md`](docs/multi_model_support.md)。
6. **DiT Latent Shape 边界**：Anima、Krea-2、Z-Image 的 `forward_for_loss` 接口接收 5D `(B, C, T=1, H, W)`（跨 4D/5D 显式 `unsqueeze(2)` / `squeeze(2)`，严禁裸 `squeeze()`）；Qwen Image 2.1 使用 4D `[B, 64, H, W]`；MOSS-Image 使用 4D `[B, 128, H, W]`。
7. **精度策略、NF4 与 Pre-Ampere 复检**：
   - NF4 外层 `compute_dtype` 必须保持 BF16（无全局 FP16 NF4）；`sm<80`（如 V100/Turing）仅在 Krea-2 开放显式实验开关 `--nf4_dir1` + `--krea2_attn_fp16`（外层仍为 BF16，`precision_policy.py` 对其豁免 `bf16->fp16` 自动改写）。
   - Z-Image/Qwen/Anima/Krea-2 的预量化 `main-mlp-nf4-v1` 混合文件仅限 BF16 eager + full checkpoint，不等同于开放通用 `base_compute="nf4"`。
   - `train_session.py` 在 `resolve_mixed_precision` 改写精度后必须调用 `recheck_compat_after_precision_resolve` 二次门禁，防止 `sm<80` 自动降为 `fp16` 后绕过 BF16-only 约束。
8. **Pipeline Parallel（PP v0）边界**：`library/training/pp/gate.py` 是唯一门禁；当前 `anima`、`krea2_raw`、`moss_image` 已启用 `torch_pipelining_1f1b` 专用训练会话（2 stages、1F1B、`mixed_precision="fp16"` + GradScaler、禁用量化与普通 `blocks_to_swap`；仅 Krea-2 PP 支持专属 `pipeline_parallel_blocks_to_swap` 且要求 `train_batch_size` 整除微批数），其余族与 Accelerate DDP 内部路径严格 fail-closed。
9. **兼容预检策略（`compat_unverified_policy`）**：`library/training/compat_matrix.py::check_training_compat` 默认 `warn`（仅降级已审计的 7 个 C1 code 并打日志/metadata `ss_compat_unverified_*`）；`allow` 仅额外放行 14 个本族未读取的 C2-ignored no-op 键；`C2-raise`、`C3`、输入校验与语义冲突在三档恒为 error。改动降级面必须同步 [`docs/findings/compat_unverified_policy_20261109.md`](docs/findings/compat_unverified_policy_20261109.md) 与 `tests/test_compat_unverified_policy.py`。
10. **LoRA 路由、Attention、Mask 与 Merge**：
    - LoRA 路由仅由三轴（`use_moe_style`、`route_per_layer`、`router_source`）与新 metadata stamp 表达，不恢复旧 `ss_use_hydra` fallback。
    - 全局 FEI（`route_per_layer=False, router_source="fei"`）每步由 `network.set_fei(z_t)` 计算并按引用共享；T-LoRA 时间步 mask 同样通过单步共享 buffer 更新，不穿透 Linear 签名。
    - Attention 统一经 `networks/attention_dispatch.py::dispatch_attention()` 明确 BHLD/BLHD 布局；磁盘拆分 `q/k/v_proj` 与运行期融合 `qkv_proj`/`kv_proj` 以 `networks/attn_fuse.py` 为唯一真相。
    - `python tasks.py merge` 仅允许可静态折叠进 Linear 的 adapter，不可折叠方法（MoE/ReFT/IP-Adapter/EasyControl/BYG/Soft Tokens 等）必须显式拒绝。
11. **ComfyUI Vendor 同步**：`custom_nodes/*/_vendor/` 严禁手改；修改受影响的 live `library/` 或 `networks/` 后运行或提醒 `python tasks.py vendor-sync`。

## WebUI、Daemon 与子系统维护

- **当前唯一前端入口**：`/next`（`web/frontend-next/` React/Vite 工作台，开发说明见 [`web/frontend-next/README.md`](web/frontend-next/README.md)）；`/` 与 `/?ui=dragon|classic` 仅作兼容重定向。旧 `web/static/` 静态源码已退役（历史背景见 [`_archive/docs/features/dragon-ui.md`](_archive/docs/features/dragon-ui.md)），不承载任何新功能。
- **WebUI 边界**：后端位于 `web/routes/` 与 `web/services/`；前端遵循 `src/{app,features,api,components,styles}/` 模块划分，不引入隐式 `globalThis` 总线，不手改 `web/static/dragon-next/` 构建产物。各类草稿独立，保存不自动启动训练，mutation 禁用盲目自动重试；所有删除操作受 `resolve_output_root()` 约束；打标凭据保留在服务端；编辑训练字段帮助后需运行 `node web/frontend-next/scripts/generate-field-help-summary.mjs` 保持摘要同步。
- **Daemon 与队列**：CLI 入口为 `daemon`、`daemon-attach`、`daemon-jobs`、`daemon-log`、`daemon-kill`、`daemon-terminate`（实现见 `scripts/daemon/`，端口由 pidfile/metadata 发现，禁止硬编码）；WebUI 队列与历史实现位于 `web/services/training/`。

## 文档维护与单一事实源

- 文档总入口为 [`docs/README.md`](docs/README.md)；分区职责与生命周期规则以该页为准，不在多处重复抄写同一配置表或模型矩阵：
  - 多模型架构与能力总表：唯一事实源为 [`docs/multi_model_support.md`](docs/multi_model_support.md)。
  - WebUI 任务标签与兼容预检说明：[`docs/configuration/model-training-capabilities.md`](docs/configuration/model-training-capabilities.md)。
  - 方法配置与 CLI 参数：集中在 `docs/methods/`，`docs/structure/` 只写数学与架构原理并单向链接 `docs/methods/`。
- 新增或移动 Markdown 必须保持从 `docs/README.md` 及所在分区 `README.md` 可达；`experimental/`、`findings/`、`proposal/` 文档前 25 行必须包含 `状态：`（或 `Status:`）标签；`docs/experimental/` 与 `docs/proposal/` 不得保留同名文件并存。
- 已完成、已替代或一次性阶段计划归档至 `_archive/docs/<subdir>/`，并同步登记 [`docs/archive-index.md`](docs/archive-index.md) 与对应归档 `README.md`。
- 文档内严禁将未受 Git 跟踪的运行目录（如 `output/...`、`/tmp/...`）或本机绝对路径写成 Markdown 链接 `[...](...)`，一律使用反引号纯文本。

## 验证策略

- 功能实现与集中验收分离：低风险文案、样式、局部配置以差异审阅、`git diff --check -- <changed_paths>` 和必要的语法/类型检查收尾，不默认新增测试或跑全量套件。
- 定向验收聚焦真实风险边界：数据删除/路径越界、latent shape 与缓存 schema、checkpoint/adapter 保存加载与恢复、核心数值逻辑、模型族与 PP fail-closed 分派、公开配置/API 兼容契约。优先复用最小相关用例（默认加 `timeout 60 .venv/bin/python -m pytest <file> -q`），通过后无新改动不重复执行。
- 大规模文档、索引或字段目录整理时运行 `timeout 60 .venv/bin/python -m pytest tests/test_documentation_integrity.py -q`（及前端目录契约检查）。
- `test-core`、`test-integration`、`test-hardware`、`test-experimental`、`test-all` 及真实大模型 GPU 热测放在明确的集中验收或发布阶段，避免争抢用户训练中的 GPU；未执行或超时的检查如实声明边界，绝不报告为已通过。
- PR 等级、bench 证据与 checklist 统一以 [`CONTRIBUTING.md`](CONTRIBUTING.md) 为准。

## 完成前检查

- `git diff --check` 对本次修改路径干净，未误碰任何用户数据目录。
- 用户可见命令、配置 schema、WebUI 表单/帮助与单一文档事实源保持同步，无悬空死链。
- 涉及 live `library/` 或 `networks/` 且影响 ComfyUI 节点时，执行或说明 `python tasks.py vendor-sync`。
- 最终回复用简短中文说明改了什么、实际验证了什么，明确区分“实现完成”与“待验收项”。
