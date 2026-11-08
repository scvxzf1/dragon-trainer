# AGENTS.md

本文件是给 AI Agent 长期维护本仓库用的根级工作协议。它覆盖整个
本仓库；子目录如果另有 `AGENTS.md` 或 `CLAUDE.md`，以离目标文件更近
的说明为补充约束。根级 `CLAUDE.md` 是指向本文件的入口重定向；不要依赖
工具自动展开引用，直接阅读本文件，并以实时源码和测试核对代码事实。

## 总体原则

- 默认用简体中文沟通，代码、命令、配置键、错误信息保持项目原有语言风格。
- 先定位子系统，再读最小必要上下文；优先用 `rg` / `rg --files` 查找。
- 保持改动小而准确。不要顺手重构、重排大文件、格式化无关文件或改写历史文档。
- 保护用户运行数据。不要把 `.venv/`、`models/`、`output/`、`post_image_dataset/`、
  `logs/`、`configs/imported/`、`configs/web-training-history/`、
  `configs/web-training-queue/` 当作普通源码清理或覆盖。
- 不要擅自删除模型、缓存、训练结果、历史任务、队列文件或用户导入配置。
- 不要擅自终止训练、清空队列、批量移动输出、下载大模型或启动长训练。确需执行时，
  先说明影响并取得用户明确同意。
- 遇到用户未提交改动，默认那是用户或其他 Agent 的工作。只在任务必须触碰同一文件时
  谨慎合并，不要 revert。
- 代码事实优先于文档。若本文件、旧说明和源码不一致，先读源码和测试，再更新文档。

## Graphify 按需导航

- 跨文件调用链、模块依赖、架构理解或改动影响分析时，由 Agent 自主判断是否先查询
  现有 `graphify-out/graph.json`，不要求用户每次提醒；已知位置的小修改、单文件阅读和
  简单文本搜索直接使用源码与 `rg`，不强制查询图谱。
- 图谱只用于缩小阅读范围，不是代码事实源。重要关系必须核对当前源码和测试；同名符号、
  动态调用、反射及运行时关系可能存在歧义或遗漏。无结果不代表不存在依赖。
- 查询前确认图谱存在，并留意生成后发生的改动、重构或分支切换。图谱缺失、过期或查询失败
  时直接回退源码，不阻塞任务，也不自动重建；需要重建时说明范围并取得用户同意。
- 每次调用设置 `GRAPHIFY_NO_AUTO_REFRESH=1`；查询同时设置
  `GRAPHIFY_QUERY_LOG_DISABLE=1`。例如：
  `GRAPHIFY_QUERY_LOG_DISABLE=1 GRAPHIFY_NO_AUTO_REFRESH=1 graphify query "<问题>" --graph graphify-out/graph.json`。
  精确关系使用同样环境变量调用 `graphify path` / `graphify explain`，限制查询和返回范围。
- 经授权建图时仅扫描本项目，使用本地 `--code-only --no-cluster`，排除依赖、缓存、数据集、
  权重、日志和训练输出。顶层数据目录使用根锚定排除规则（如 `/models/`），不得误排除
  `library/models/` 等源码。现有图谱先备份或使用独立输出目录。
- 未经额外授权，不分析文档、PDF、图片或视频，不启用语义 API、LLM 标注、MCP、服务或
  watcher，不自动安装升级工具或修改 hooks / Agent 规则。
- `graphify-out/` 是本地产物，不随业务代码暂存、提交或推送；使用明确路径暂存，保留用户
  已有文件和改动。本规则不自动修改 `.gitignore`。

## 子代理委派与模型

- 主代理可在用户已授权的任务范围内自主规划和委派探索、数学推导、代码实现、测试及独立验收，无需逐次申请委派许可；不再限定子代理只能探索或使用默认角色。
- 子代理模型选择与模型回退限制统一遵循全局 `~/.codex/AGENTS.md`，本项目不另设模型白名单。
- 保持 `fork_turns="none"`，通过自包含任务说明交接，减少重复上下文与主线程重复阅读。
- 实现任务必须明确文件归属、接口契约和验收条件；子代理必须保护他人改动。多代理并行写入遵守下文独立工作树规则。
- 实现者负责必要的局部自查；仅在下文「验证策略」列出的关键风险或明确的发布要求下，
  将独立验收作为交付条件。普通改动不默认再派测试代理；主代理负责集成、审阅关键差异与已有证据。
- 测试/验收代理默认允许异步工作，主代理继续推进不依赖其结果的开发、集成和文档工作。
  只在当前决定确实依赖验收结果、命中关键风险或准备发布时等待；普通开发交付可明确标注待验收项。
  关键风险的独立验收由不同于实现者的代理承担，也可以安排在集中的验收阶段。

## 反上帝代码守则

本节用于防止后续维护继续把复杂逻辑堆回少数大文件。

- 热点文件默认只做 facade、编排、兼容 shim 或小范围修复，不新增大块业务逻辑：
  - `train.py`
  - `inference.py`
  - `library/inference/generation.py`
  - `library/datasets/base.py`
  - `networks/lora_anima/network.py`
  - `networks/lora_anima/config.py`
  - `web/services/training_service.py`
  - `web/services/config/_legacy.py`
  - `web/static/js/features/anima-app/chunks/*`
- 修改热点文件超过约 50 行时，优先拆到现有子模块或新模块；若确实不能拆，最终回复要说明：
  - 为什么必须改热点文件。
  - 为什么不能放到新模块。
  - 后续如何继续瘦身。
  - 做了哪些必要检查，还有哪些待验收项。
- 单个新增 Python 函数建议不超过 100 行；超过时优先拆成 helper、pipeline step 或策略对象。
- 单个新增 Python 类建议不超过 400 行；超过时优先按状态、IO、策略、验证、保存/加载拆分。
- 单个新增 JS 函数建议不超过 80 行；新增 UI 逻辑必须按 feature、store、api、renderer 拆分。
- 单个新增测试文件建议不超过 1200 行；超过时按领域拆成多个测试文件，不要继续加大现有超大测试。
- 已超过 1000 行的源码或测试文件，除搬迁、兼容 shim、删除旧逻辑外，默认不继续承载新业务。
- 重构优先采用“搬家型重构”：先保持行为不变地抽模块，再按实际风险复用或补充验证并清理旧 facade；不要一轮同时改架构和改行为。
- 新增配置、CLI、adapter、WebUI 表单或队列/历史行为时，必须同步考虑文档入口、测试入口和旧兼容面，避免逻辑散落。

## 环境和命令入口

- 项目运行环境是 Python 3.13，依赖管理优先使用 `uv`；安装和模型准备以
  [根 README](README.md) 与 [Linux 部署指南](docs/guidelines/linux-deployment.zh.md) 为准。
- [`tasks.py`](tasks.py) 是命令入口真相；`Makefile` 只是薄转发。实时命令面使用
  `python tasks.py --help` 查看，稳定与实验命令实现分别位于
  [`scripts/tasks/`](scripts/tasks/) 和 [`scripts/experimental_tasks/`](scripts/experimental_tasks/)。
- 维护和验证命令优先使用 `.venv/bin/python`；只有确认无需项目虚拟环境，或 `.venv/`
  不存在时，才回退到系统 `python`。
- 直接使用原生 Shell 命令；搜索优先 `rg`，读取时限定相关文件和行范围。
  保留关键错误、退出状态与验证证据，避免无关的大量输出占用上下文。
- 跨平台用户文档可写成 `python tasks.py <command>`，本仓维护执行优先写成
  `.venv/bin/python tasks.py <command>`。用户文档可把 `make <target>` 作为兼容写法，
  但不要作为唯一入口。
- `python tasks.py <command> KEY=value` 支持 Make 风格尾随环境变量，例如：
  `python tasks.py print-config METHOD=lora PRESET=default`。
- 实验命令通常以 `exp-*` 开头，可能变动或删除。修改时同步检查实验任务实现、配置、文档和测试。
- 常用维护入口（测试命令按下文「验证策略」选用，不作为每次开发的固定步骤）：
  - WebUI：`.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102`
  - 单元测试：`timeout 60 .venv/bin/python -m pytest tests/<test_file>.py`
  - 快速反馈：`timeout 60 .venv/bin/python tasks.py test-fast`
  - Core layer：`timeout 300 .venv/bin/python tasks.py test-core`
  - 分层测试：`test-integration`、`test-hardware`、`test-experimental`
  - 完整测试：`.venv/bin/python tasks.py test-all`（`test-unit` 保留为兼容别名）
  - 合并配置：`.venv/bin/python tasks.py print-config METHOD=<name> PRESET=<name>`
- 老文档中的命令若不在 `tasks.py --help` 当前列表中，按历史或兼容入口处理，不要直接假定仍可用。

## Git 推送和回滚

- 默认线上仓库由 URL 确定，而不是由本机 remote 别名确定：
  `github.com/scvxzf1/dragon-trainer`（2026-10-19 由 `anima_lora_webui` 改名，旧 URL 自动重定向）。`main` 是发布分支，`dev` 是日常集成分支。
  用户说“拉取线上更新”或“同步线上 main”时，按指定分支操作；未指定分支的日常开发推送
  默认使用 `dev`，只有明确要求发布或推送 `main` 时才写入发布分支。
- 不要把 `main` 当作日常开发分支。开始独立开发任务时，在干净且适合切换的工作树中
  先同步 `dev`，再优先创建 `codex/<task>`；单人小改动才直接在 `dev` 上提交。
  只读审查、用户指定的现有分支续作以及脏工作区不自动执行该同步流程。
  多个 Agent 不要共用同一工作树和分支进行并行写入。
- 操作前先运行 `git remote -v`，找到 URL 匹配目标仓库的 remote；它在不同 checkout 中可能叫
  `origin`、`webui` 或其他名字。命令和汇报使用实际 remote，不要凭文档假定别名存在。
- 指向个人 fork、私有镜像或 `sorryhyun/anima_lora` 等参考仓的 remote 不是默认发布目标。
  除非用户明确点名，不要向它们 pull、push、reset，也不要把上游参考合入和线上发布混为一谈。
- 本仓与 `sorryhyun/anima_lora` 是血缘关系而不是跟踪关系：指向上游的 remote 统一命名
  `upstream`，只 fetch 不 push，不要求 merge-base 对齐；有价值的上游改动按需移植，并在
  提交信息标注 `port(upstream): ...`。对外表述以
  [项目血缘与上游政策](docs/project-lineage.md) 为准，不要把 README 写回
  「基于某某的扩展项目」。
- 日常开发在干净工作树中的最小同步流程是：
  ```text
  git status --short --branch
  git fetch <target-remote> --prune
  git switch dev
  git pull --rebase <target-remote> dev
  ```
  工作树有未提交或未跟踪内容时，先区分代码、用户配置和运行数据，不自动执行
  `switch` / `pull` / `rebase`，也不自动 stash、提交或清理。优先保留当前现场并在
  独立工作树隔离开发；若必须处理原工作区，先说明保留方案并取得用户确认。
  不得用切换、reset 或清理命令偷换或删除已有内容。
- 首次使用本流程时，在没有需要保留的未提交改动的工作树中，从已同步的 `main` 创建并发布 `dev`：
  ```text
  git switch main
  git pull --ff-only
  git switch -c dev
  git push -u <target-remote> dev:dev
  ```
  若线上已经存在 `dev`，改用 `git switch --track <target-remote>/dev`，不要用本地分支覆盖
  线上历史。
- 推送前至少检查：`git status --short --branch`、`git fetch <target-remote> --prune`、
  对日常分支使用 `git rev-list --left-right --count HEAD...<target-remote>/dev`，对发布分支
  使用对应的 `main` 比较，并查看 `git log --oneline <target-remote>/<branch>..HEAD`，再跑
  按下文「验证策略」选定的必要检查。
- 未跟踪文件默认不随推送发布，除非用户明确要求或本次任务确认需要纳入版本控制。
- 用户要求直接“推送更新到线上”但未指定分支时，默认目标是 `<target-remote> dev:dev`；明确
  发布时才使用 `<target-remote> main:main`。若当前身份没有写权限，使用个人 fork 分支和 PR，
  不要因此把个人 fork 改称线上主仓。完成后汇报仓库、remote、分支、最新提交 hash，以及是否
  还有未提交或未跟踪改动。
- 发布 `main` 前应先将 `dev` 的提交通过 review，并完成与风险和发布要求对应的定向验收，再使用 fast-forward 或普通 PR 合入；
  不要为了绕过分支差异强推。逻辑相关的代码、测试和文档可拆成独立提交，但每个提交都必须
  保持可解释、可验证。
- 暂存时使用明确的路径白名单，不要在含有本机设置、模型路径、训练历史、队列、日志、导入
  prompt 或输出的工作树执行 `git add -A`。`configs/web-ui-settings.toml` 的本机覆盖应迁到
  已忽略的 `.anima-webui-settings.toml` 或外置配置根；不能把绝对路径带入提交。
- 用户说“回滚”时，先分清是哪一种：
  - 本地工作区回退：丢弃未提交改动。只有用户明确要求时才做，执行前说明会丢失哪些内容。
  - 本地提交回退：撤销本地一个或多个提交。共享分支默认优先 `git revert`，不要默认改写历史。
  - 线上回退：撤回目标仓库分支上的已发布提交。必须先确认仓库、remote、分支和目标提交。
- 需要以线上仓库为准同步本地时，先 fetch 并比较 `HEAD` 和目标分支，再决定是否 reset。
  不要在没核对差异前直接做破坏性操作。
- `git reset --hard`、`git checkout -- <path>`、`git clean -fd`、`git push --force`、
  `git push --force-with-lease` 都视为高风险操作。除非用户已经明确要求，或已经明确给出
  目标提交/分支并接受影响，否则不要执行。
- 如果确实要改写线上历史，优先使用 `--force-with-lease`，并在执行前说明会覆盖哪个仓库和
  分支、抹掉哪些提交、是否影响其他协作者。
- 可以使用环境变量中的凭据或本机已配置的 SSH key 推送，但不要把 PAT、cookie、私钥或
  带密钥的远程 URL 写进仓库文件、文档、日志样例或长期说明。
- fetch/push 协议应尽量保持一致并使用已配置的 credential helper；SSH 不可达时可临时改用
  HTTPS，但不要把令牌嵌入 remote URL，也不要把认证诊断输出写入仓库。
- 面向人的精简操作说明见 [Git 同步规则](docs/guidelines/git-sync-policy.md)。

## 项目地图

- `tasks.py`：所有稳定命令注册表。
- `train.py`：统一训练入口；`AnimaTrainer` 类名作为兼容 facade 保留，实际按
  `model_family` dispatch Anima、Krea-2、Z-Image 和 Qwen-Image 2.1；族清单以注册表为准。
- `inference.py`：独立推理入口。
- `anima_lora/`：可安装包门面，给嵌入式调用暴露精选 API。
- `library/`：训练、推理、配置、数据、runtime、模型、captioning、vision 等核心逻辑。
- `library/anima/`：Anima DiT、权重加载、token/text strategy 和模型配置。
- `library/models/family_registry.py`：模型族名称、别名、cache、adapter、推理和 attention
  能力的单一事实源；operation handler 必须覆盖全部已注册 family。
- `library/models/krea2_raw/`：Krea-2-Raw 第二生产模型族（迁移过程见
  `_archive/docs/proposal/krea2_raw_migration.md`）。`dit.py` 提供 single-stream MMDiT、block swap、
  checkpoint 和 fixed resident compile；`weights.py` / `quantize.py` 负责 strict BF16/NF4
  加载；`strategy.py`、`family.py`、`sampling.py`、`inference_runner.py` 分别承载 Qwen3-VL
  cache、训练 forward、Euler flow sampling 和独立推理 dispatch。当前开放网络变体选择，但不代表
  所有组合均经过验证；支持 single/euler 推理以及 `flash`（默认）/`torch`/`sdpa` attention。完整迁移与消融证据进入
  `docs/findings/`，不要继续把阶段日志堆进本项目地图。
- `library/models/z_image/`：Z-Image 训练与训练预览 v1。使用官方 Diffusers transformer、
  Flux VAE latent normalization、Qwen3 prompt/cache 和 plain LoRA；`family.py` 提供
  flow-matching 训练契约，`strategy.py` 使用 `_z_image_te.safetensors`，`weights.py` 支持
  Diffusers 目录和受校验的单文件组件，`attention_backend.py` 将默认 `flash` 映射到
  Diffusers `flash_varlen`，`block_swap.py` 把官方 `model.layers` 接到共享 `ModelOffloader`。
  通用 image-test/独立推理尚未注册。
- `library/models/qwen_image_2_1/`：Qwen-Image 2.1 T2I/Edit 训练与训练预览。
  `family.py` 承载 4D latent/token forward，`strategy.py` 和 `weights.py` 负责专属
  text cache 与权重加载，`compile.py` 提供独立 block 编译。能力边界见
  [多模型支持](docs/multi_model_support.md)，adapter 组合按实时源码和验收证据判断。
- Krea-2 性能研究只在本协议保留维护边界，不再复制逐阶段实验数字。根因与否决路径见
  [最终结论](docs/findings/krea2_3080_speed_final.md)，跨硬件、冷/热稳态矩阵见
  [扩展对照](docs/findings/krea2_3080_speed_comparison_extended.md)；历史后端默认值以
  当前源码和方法配置为准。生产安全边界见下文「Krea-2 训练与性能边界」。
- `library/config/`：TOML 读取、合并、normalize、schema 校验。
- `library/training/`：训练 bootstrap、loop、optimizer、scheduler、checkpoint、loss 等。
- `library/inference/`：generation、sampling、adapter 加载、DirectEdit、DCW、输出处理。
- `library/preprocess/`：预处理和缓存编排。
- `networks/`：adapter/network 实现。修改这里前读 `networks/CLAUDE.md`。
- `scripts/tasks/`：稳定命令实现。
- `scripts/experimental_tasks/`：实验命令实现。
- `web/`：aiohttp WebUI 后端、`frontend-next/` React/Vite 工作台及静态发布资源。
- `preprocess/`：CLI 预处理脚本，底层编排通常在 `library/preprocess/`。
- `custom_nodes/`：ComfyUI 节点；发布副本通过 `_vendor/` 同步。
- `configs/`：base、presets、methods、gui-methods、datasets、Web 设置、历史和队列。
- `tests/`：pytest 测试，按文件名定位覆盖面。
- `bench/`：方法、性能、显存、推理和实验验证。
- `docs/`：文档主树，入口是 `docs/README.md`。
- `_archive/docs/`：历史或缺失上下文资料，不当作当前实现说明。

## 先分类再动手

把需求归入一个主子系统，跨子系统时按依赖顺序处理。

- WebUI：页面、表单、API、训练队列、历史、预览、全局设置、权重分析。
- 配置/数据/runtime：TOML 合并、数据集蓝图、缓存目录、路径解析、运行时配置。
- 训练：bootstrap、dataset、forward、loss、optimizer、resume、checkpoint、progress。
- Adapter/network：LoRA family、LoHa、LoKr、VeRA、GLoRA、Hydra、FeRA、ReFT、
  Chimera、IP-Adapter、EasyControl、BYG、Turbo、SPD、Soft Tokens。
- 推理：generation、sampling、adapter 加载、Spectrum、DCW、SMC-CFG、DirectEdit。
- 预处理：resize、VAE cache、TE cache、PE cache、pooled text cache、caption index、mask。
- Anima Tagger / captioning：tag taxonomy、训练缓存、阈值、DirectEdit source caption。
- Daemon / 队列：本地训练守护进程、WebUI 队列、runtime config、日志追踪。
- Custom nodes：ComfyUI hydralora loader、trainer、tagger、directedit、blockcompile。
- Bench/docs/tests：实验声明、文档维护、验证脚本、测试覆盖。

## 配置和数据约定

### 配置目录外置

配置根目录、训练历史和队列支持外置。优先级、路径展开、安全边界和迁移方法以
[外置配置目录说明](docs/configuration/external-configs.md)及
[`library/env.py`](library/env.py) 为准。

维护时必须保留以下边界：相对路径锚定项目根目录；允许绝对路径、`$HOME` 和 `~`；拒绝
包含 `..` 的路径；未配置覆盖项时继续回退到仓库内 `configs/`。本机路径覆盖写入已忽略的
`.anima-webui-settings.toml`，不要把本机绝对路径写进受版本控制的默认配置。

### 训练配置合并链

```text
configs/base.toml
  -> configs/presets.toml[<preset>]
  -> configs/<methods_subdir>/<method_or_variant>.toml
  -> CLI args
```

- 默认 `methods_subdir="methods"`；WebUI 友好变体使用历史命名目录 `configs/gui-methods/`。
- [`configs/base.toml`](configs/base.toml) 包含共享基础路径、optimizer、compile flag 和默认数据集蓝图。
- [`configs/presets.toml`](configs/presets.toml) 放硬件/采样 profile，不要把硬件 profile 复制进方法配置；实时预设以该文件为准。
- [`configs/methods/`](configs/methods/) 放算法 family 配置。
- [`configs/gui-methods/`](configs/gui-methods/) 放自包含用户变体，不使用注释切换块；实时变体以目录内容为准。
- [`configs/datasets/`](configs/datasets/) 放可复用数据集蓝图。
- `configs/imported/` 是 WebUI 导入或用户配置，默认视为用户数据。
- `configs/sample-prompts/` 放按配置分叉保存的 sample prompts。
- `configs/web-ui-settings.toml` 保存 WebUI 全局设置和模型路径；不要把本机绝对路径写进默认值或文档。

默认数据和缓存：

- 源图片通常在 `image_dataset/`，同名 `.txt` 是 caption master。
- 常见产物：`post_image_dataset/resized/`、`post_image_dataset/lora/`、
  `post_image_dataset/masks/`、`output/ckpt/`、`output/tests/`、`output/runs/`。
- subset 可以设置 `cache_dir`，让 VAE/text/PE sidecar 写入专用缓存目录。
- 重要 sidecar：
  - `{stem}_{WxH}_anima.npz`：VAE latent cache。
  - `{stem}_anima_te.safetensors`：text encoder cache。
  - `{stem}_anima_pe.safetensors`：PE-Core vision feature cache。
  - `{stem}_krea2_te.safetensors`：Krea-2 Qwen3-VL text cache；Krea-2 latent 仍复用共享
    VAE/Anima latent cache 策略。
  - `{stem}_{WxH}_z_image.npz`：Z-Image Flux VAE latent cache（分辨率 infix 由调用方加入）。
  - `{stem}_z_image_te.safetensors`：Z-Image text encoder cache。
  - `{stem}_{WxH}_qwen_image_2_1.npz`：Qwen-Image 2.1 专属 latent cache。
  - `{stem}_qwen_image_2_1_te.safetensors`：Qwen-Image 2.1 text encoder cache。
    不得复用其他族的 latent normalization 或 text cache schema。

## 方法和能力入口

不要在总协议里维护会漂移的方法清单。开始修改某个方法前，按成熟度和任务类型查阅：

- 稳定能力：[方法索引](docs/methods/README.md)。
- 可运行实验：[实验索引](docs/experimental/README.md)。
- 原理与架构：[结构索引](docs/structure/README.md)。
- 实验结论与失败路径：[研究结论索引](docs/findings/README.md)。
- 当前命令入口：[`tasks.py`](tasks.py) 与 `python tasks.py --help`。

文档只用于定位和理解；最终以实时源码、配置和测试为准。

## 不可破坏的不变量

### Text Encoder Padding

Anima 预训练模型需要 max-padded text encoder outputs。padding 位置会作为
cross-attention softmax 的 attention sinks。

- 不要按真实文本长度裁剪 text encoder 输出。
- 不要通过 `crossattn_seqlens` mask 掉 padding。
- tokenizer 或 padding 行为变化后，需要重建磁盘 `.npz` / `.safetensors` 缓存。
- 相关区域：`library/anima/strategy.py`、`library/anima/text_strategies.py`、
  `library/preprocess/text.py`、`library/inference/text.py`。

### Constant Token Buckets 和 Native Flatten

`library/datasets/buckets.py::CONSTANT_TOKEN_BUCKETS` 当前是 24 个 `(W, H)` 分辨率，
分成 4032 和 4200 两个 token-count family。每个 bucket 精确填满自己的 token count，
没有 intra-bucket padding。

- Anima/Krea-2 的共享分桶路径使用 native shapes；不要恢复旧的 pad-to-static 4096 路径。
- Anima 的 `compile_blocks()` 会开启 native-shape flatten，让图按 token count 复用。
  Krea-2 使用自己的 fixed resident block 编译，Qwen-Image 2.1 使用独立的 token/编译路径；
  不要把 Anima 的 flatten 或 band 实现当作所有模型族的通用契约。
- `compile_seq_bands` 是 Anima-only 实验开关，默认 `false`，且必须同时启用
  `compile_dynamic_seq=true`；Krea-2 和 Z-Image 由 compatibility layer 自动关闭，
  Qwen-Image 2.1 不复用该 band 编译路径。
- Anima 初始 bands 来自当前 `bucket_resolutions`，缺省才回退 `CONSTANT_TOKEN_BUCKETS`。采样事件
  发现新 prompt token count 时由 `ensure_training_compile_seq_range()` 加 singleton range
  并重编译，不要宣称 canonical bucket 自动包含任意 sample 分辨率。
- 64GB synthetic 多 band A/B 数值通过，但比 union 慢约 43--44% 且无显存收益，生产默认
  Anima 继续使用 union dynamic-seq。证据见
  `docs/findings/anima_perband_dynamic_seq_20260830.md`。
- 改 bucket 表、token count、compile flatten、sample prompt 分辨率参与预算时，必须补
  shape/invariant 测试。
- DCW aspect bucket 的顺序会影响已发布 fusion-head checkpoint，不要随意重排。
- 相关测试：`tests/test_constant_token_buckets.py`、`tests/test_native_flatten.py`、
  `tests/test_runtime_harness_cli.py`。

### Lazy Model Loading

训练为了避免 OOM，加载顺序必须保持：

```text
text encoder -> cache -> free
VAE -> cache -> free
DiT -> attach network -> training loop
```

不要把 DiT 提前加载到预处理阶段。WebUI、task runner 和 daemon 启动训练时也要保持
这个顺序。

### Compile After Apply

`torch.compile` 必须 trace adapter monkey-patched forward；支持编译的模型族都必须在
`network.apply_to` 和可选 `load_weights` 后编译，不要在 bench、scripts、preprocess 中
手写易错顺序。入口按模型族选择：

- Anima/Krea-2 训练复用 `compile_blocks_for_training()`；Anima runtime 可复用
  `library/runtime/harness.py::build_anima`。
- Qwen-Image 2.1 由训练 bootstrap 调用
  `library/models/qwen_image_2_1/compile.py::compile_qwen_image_2_1_blocks()`，编译 block
  inner forward，swap dispatch 留在图外；不复用 Anima native flatten/bands。
- Z-Image 当前不支持 compile，不能因为共享训练入口而绕过兼容拒绝。
- Anima distillation 的 `compile_dit_blocks_for_pool()` 仍使用一个 union range，
  不复用训练侧 `compile_seq_bands`。

### Krea-2 训练与性能边界

- 默认配置保留 full checkpoint、fixed resident compile 和 Flash varlen；
  `torch`/`sdpa` 可显式回退，block swap 按实际可用显存选择。NF4 是显式选择的路径，
  启用时只允许 BF16 compute，不将其描述为基座加载默认值。
- NF4 必须遵守 `weights.py` / `quantize.py` 的 BF16 compute 检查；FP16 NF4 已被
  数值验收否决，不新增全局 FP16 NF4 开关。
- 编译目标是 adapter apply/load 后的 resident block `_forward`，不是 checkpoint
  wrapper 或 swapped tail。使用 fixed sequence 和 None/default mode；拒绝
  dynamic-seq、`reduce-overhead`、`max-autotune*` 等未支持配置。
- selective checkpoint 只支持 `off/every_other`；RTX 3080 10GB 保持 full checkpoint，
  不套用 PG199 的 selective 实验。compile + selective 叠加没有通用安全余量，不进默认档。
- 性能由大矩阵吞吐主导，长窗口叠加热降频；冷态 compile 收益不能外推持续收益。
  不要把 rank 降低、padding 尾裁剪或每步 `prepare_block_swap_before_forward` 当作加速方案。
  checkpoint/resume 已验证无需重复调用 `compile_blocks()`。
- 修改功耗、风扇或降压前必须取得用户明确同意。各实验的适用条件与数值证据查阅
  项目地图链接的 findings，不将某张卡的结果外推到其他硬件。

### DiT Latent Shape

latent shape 是模型族接口契约，不是所有底层 transformer 的统一输入格式。

- Anima DiT 和 Krea-2/Z-Image 的训练 `forward_for_loss` 接口使用
  5D `(B, C, T=1, H, W)`，单例时间轴是 dim 2；族内再转换成底层模型要求的格式。
- 这些路径的 VAE、cache、训练 inner loop 和很多 spectral helper 使用
  4D `(B, C, H, W)`；跨 4D/5D 边界显式 `unsqueeze(2)` / `squeeze(2)`，
  不要用裸 `squeeze()` 或 `squeeze(0)`。
- Qwen-Image 2.1 的 `forward_for_loss` 使用 4D `[B,64,H,W]`，在族内展平为 image tokens；
  不要套用其他族的时间轴转换。以各族 `family.py` 的 shape 校验为准。

### Model Family Dispatch、Z-Image 和 Qwen-Image 2.1

- `MODEL_FAMILY_REGISTRY` 和 `dispatch_model_family()` 必须 fail closed；未知 family、缺失
  handler、未注册推理 mode/sampler 不得回退 Anima。
- Z-Image 当前开放网络变体选择，但真机训练证据仍以 plain LoRA 为限，不得外推全部变体已验证。
  支持 BF16 `torch`/`sdpa`/Flash varlen attention、full checkpoint、训练和训练 preview；
  NF4、compile、selective/offloaded checkpoint 及通用推理均不支持。
- Z-Image `blocks_to_swap=0` 表示关闭；启用时必须满足
  `1 <= blocks_to_swap <= len(model.layers)-2`，当前 30 main layers 即 `1..28`。
- Z-Image dataset/subset `caption_dropout_rate` 必须为 0；其 text cache 和 latent affine
  不得借用 Anima/Krea sidecar 或 normalization。
- Qwen-Image 2.1 支持 T2I/Edit 训练和训练 preview；通用 image-test/独立推理的
  mode/sampler 尚未注册，不得从训练 preview 外推通用推理支持。
- Qwen adapter 支持以实时 registry、compatibility layer 和定向验收证据为准；
  接口放行不等于所有组合已真机验证，正在推进的 LoKr/LoHa 扩展须区分代码支持与验证范围。
- 详细边界见 `docs/multi_model_support.md`，相关测试为
  `tests/test_model_family_registry.py`、`tests/test_model_family_fail_closed.py`、
  `tests/test_z_image_family.py` 和 `tests/test_z_image_block_swap.py`。

### LoRA Family 三轴路由

LoRA family 路由由三轴配置表达，不要恢复旧 metadata fallback：

- `use_moe_style`: `False` / `"shared_A"` / `"independent_A"`
- `route_per_layer`: `True` / `False`
- `router_source`: `"none"` / `"input"` / `"sigma"` / `"fei"` / `"crossattn_emb"`

关键位置：

- `networks/lora_anima/config.py::LoRANetworkCfg.from_kwargs`
- `networks/__init__.py::resolve_network_spec`
- `networks/lora_anima/network.py`
- `networks/lora_modules/*`

旧 metadata 如 `ss_use_hydra`、`ss_use_fei_router` 不再加载；新 stamp 是
`ss_use_moe_style`、`ss_route_per_layer`、`ss_router_source`。

### GlobalRouter / FEI

当 `route_per_layer=False` 且 `router_source="fei"` 时：

- `network.set_fei(z_t)` 每步计算一次 FEI 和 router。
- routing weights 通过引用写入每个 routing-aware module。
- 训练 loop 和推理 denoising loop 都需要在当前 step 前设置 FEI。

相关位置：`library/runtime/fei.py`、`library/training/router_conditioning.py`、
`library/inference/generation.py`、`networks/lora_anima/network.py`。

### Attention Layout 和 Fused Projection

- `networks/attention_dispatch.py::dispatch_attention()` 是 attention backend layout 转换入口。
- SDPA/sageattn 常见 BHLD；xformers/flash-attn 常见 BLHD。新增 call site 必须明确布局。
- runtime fused `qkv_proj` / `kv_proj` 与 on-disk split `q/k/v_proj` 的唯一真相源是
  `networks/attn_fuse.py`。保存和加载都依赖它。

### Timestep Masking

T-LoRA mask 是共享 buffer，每个 denoising step 更新一次。

- 不要把 `t` 穿透到每个 LoRA forward。
- 新 timestep-aware variant 应复用 buffer pattern。
- `networks/lora_anima/factory.py` 和 `networks/lora_anima/network.py` 是设置/清理 mask 的主要位置。

### Merge 边界

`python tasks.py merge` 只适合可折叠进 DiT Linear 权重的 adapter。

- LoRA / OrthoLoRA / DoRA / T-LoRA 通常可折叠。
- ReFT / Hydra MoE / postfix / IP-Adapter / EasyControl / BYG / Turbo / Soft Tokens 通常不能完整折叠。
- 新方法必须更新 merge 支持或拒绝逻辑，并在文档说明。

### ComfyUI Vendor 树

`custom_nodes/*/_vendor/` 是 live `library/` / `networks/` 的发布副本。

- 先改 live source。
- 再运行或提醒运行 `python tasks.py vendor-sync` / `make vendor-sync`。
- 不要手工让 vendor 和 live source 分叉。

## 前端实现约束

构建或修改 WebUI 前端时，必须避免生成臃肿的石山代码。

- 新增前端业务代码默认尽量控制在 1000 行以内；需求复杂、无法合理控制时，先说明原因、
  拆分方案和预计代码规模。
- 当前前端为 `web/frontend-next/` 的 React/Vite 工作台，按职责拆分页面、组件、hooks、
  utils、api、constants、styles；遵循 `src/app/`、`src/features/`、`src/api/`、
  `src/components/` 和 `src/styles/` 模块边界。旧 `web/static/js/` / `css/` 仅适用于
  遗留代码维护，不承载新工作台业务。
- 禁止把大量逻辑堆进单个 `App`、页面文件或超大组件；单个组件建议不超过 200-300 行，
  单个函数建议不超过 80 行。
- 重复 UI 和重复逻辑必须抽象复用，但不要为压缩行数牺牲可读性、功能完整性和可测试性。
- 样式应简洁克制，避免大段重复 CSS、硬编码结构和无法维护的视觉特殊分支。

## WebUI 维护

入口：

- 后端路由：`web/routes/config.py`、`training.py`、`settings.py`、`preview.py`、`analysis.py`、
  `tagging.py`。
- 业务服务：`web/services/config_service.py`、`settings_service.py`、`training_service.py`、
  `preview_service.py`、`weight_analysis_service.py`。
- 拆分业务：`web/services/config/`、`web/services/training/` 和 `web/services/tagging/`。
- 当前用户入口：`/next`（Next React 工作台）；`/` 和历史 `/?ui=dragon` / `/?ui=classic` 仅作兼容重定向，最终进入该入口。
- 前端 bootstrap：`web/frontend-next/src/main.tsx`；壳、路由和 UI 偏好在 `src/app/`，
  业务模块在 `src/features/`，共享 API、组件和样式在相应 `src/` 子目录。
- 开发与部署说明：[前端 README](web/frontend-next/README.md)。工具链以其说明、
  `package.json` 和锁文件为准；开发使用 Vite 并单独启动后端，生产由 Python 服务托管
  `web/static/dragon-next/` 构建产物，不需要 Node 常驻。
- 旧 `web/static/app.js`、`index.html`、`style.css` 和 `js/features/anima-app/`
  仅为遗留静态源码，不是当前主入口或用户故障回退入口；活跃功能文档和新维护应以 Next 路由及 feature 目录为准。

规则：

- 新工作台业务放入独立 feature，不依赖旧界面 DOM 或隐式全局业务状态；旧字段目录和
  能力规则只能按纯模块复用，不挂载旧 DOM。不要恢复 `js/features/legacy-app.js`。
- `globalThis` 只允许作为旧代码迁移桥或第三方库兼容桥；新 WebUI 业务默认使用 `export/import`
  和显式 `ctx` / store，不要新增隐式全局状态总线。
- `web/static/js/features/anima-app/chunks/` 是遗留机械拆分过渡层；仅在维护遗留代码时
  适用迁出状态和函数的规则，不把新工作台功能堆回该目录。
- 事件绑定、拖拽、筛选、弹窗、状态渲染等重复前端逻辑应抽到 shared helper 或 feature-local
  helper，不要复制一套近似 DOM 操作。
- Next 构建通过哈希资源与原子替换入口发布；修改源码后走 `web-next-build`，不手改
  `web/static/dragon-next/` 产物。旧 `previous-index.html` 只用于静态版本发布回退。
- 仅维护旧静态前端时：更新 import 同步 cache token，改 DOM id 前先搜索 selector
  和相关测试，CSS 新文件从旧 `style.css` 可达并遵守 import 顺序。
- 工作台各类草稿独立，保存不自动启动训练；mutation 不自动重试，连接中断且结果未知时
  必须先核对服务器状态，不能盲目重试。
- `configs/web-ui-settings.toml [global]` 由 `settings_service.py` 管理，`output_root` 默认
  `output/runs`。
- runtime、history、preview 图片、队列项删除必须受 `resolve_output_root()` 边界约束。
- `memory_probe_jsonl = "auto"` 和 `block_swap_profile_jsonl = "auto"` 应解析到当前任务目录，
  不要写回用户配置固定路径。
- sample prompts 默认 `configs/sample_prompts.txt`，按配置分叉到
  `configs/sample-prompts/<methods_subdir>/<config-stem>.txt`。保留注释、空行和用户格式。
- 历史任务模式只保留 `collection` / `collections`，不要恢复旧 `config` / `flat` 模式。
- 工作台复用共享 HTTP API；打标 provider 凭据与调用必须留在服务端，审核草稿和写回
  任务不并入训练队列。

前端开发按改动需要做类型检查或构建，低风险文案、样式和局部 UI 修改不默认新增测试或运行 E2E。
需要类型检查和单元测试的集中检查时使用 `.venv/bin/python tasks.py web-next-check`；
业务逻辑优先复用相关定向用例，`web-next-e2e` 用于关键交互验收、明确的发布要求或用户指定验证。
部署构建使用 `web-next-build`；具体命令以前端 README 和工具链为准，维护执行使用原生命令。
E2E 使用隔离端口和 mock API/训练 WebSocket，不对用户后端执行写操作，不复用或终止
已有服务；具体工具链、端口和并发隔离方式见前端 README。
旧静态前端维护才参考
[前端健康度评分卡](docs/features/frontend-health-scorecard.md)及
[`tests/test_training_frontend_*.py`](tests/)。后端需要测试时，按改动领域从 [`tests/`](tests/) 中选择
配置、预览、队列、历史或权重分析的最小定向用例，不在本协议复制容易漂移的测试清单。

## Adapter 和 Network 维护

修改 `networks/` 前先读 `networks/CLAUDE.md`。常见定位：

- LoRA family：`networks/lora_anima/config.py`、`factory.py`、`network.py`、`loading.py`。
- 单个 LoRA 变体：`networks/lora_modules/*.py`。
- LoHa/LoKr/VeRA/GLoRA 插件：`networks/plugins/<name>/module.py` 和 `save.py`。
- IP-Adapter/EasyControl/Soft Tokens/BYG 等：`networks/methods/`。
- 保存：`networks/lora_save.py`。
- fused/split projection：`networks/attn_fuse.py`。
- attention backend：`networks/attention_dispatch.py`。

新增或修改方法时检查：

- config：`configs/methods/`、必要时 `configs/gui-methods/`。
- registry：`networks/__init__.py` 或插件注册点。
- 保存/加载 metadata 和兼容拒绝逻辑。
- 推理加载：`library/inference/adapters.py` 或方法专属路径。
- ComfyUI loader 是否需要同步。
- tests、bench、docs 和 `tasks.py` 入口。

## 训练、推理和预处理

训练：

- 入口是 `train.py`，可复用逻辑通常在 `library/training/`。
- 新 CLI 参数要检查 `library/training/cli_args.py`、config schema、WebUI、README/docs。
- WebUI/daemon 启动训练还要查 `library/runtime/launch.py`、`scripts/tasks/_common.py`、
  `web/services/training_service.py` 和 `web/services/training/*`。
- 涉及 GPU、accelerate、compile 且需要测试时，优先小 fixture 或 monkeypatch；
  真实大模型、真机数值和性能验证放到明确的验收阶段。

推理：

- adapter 加载先查 `library/inference/adapters.py` 和具体 network loading。
- sampler/denoising 改动先查 `library/inference/generation.py`、`sampling.py`、
  `sampler_context.py`。
- DCW、Spectrum、SPD、SMC-CFG、CNS、mod-guidance 应组合或明确互斥；改一个不要静默破坏另几个。
- DirectEdit 涉及 Anima Tagger 和 inversion，先读 `docs/experimental/directedit_editing_v3.md`。

预处理：

- task wrapper 在 [`scripts/tasks/preprocess.py`](scripts/tasks/preprocess.py)。
- CLI 与编排实现分别位于 [`preprocess/`](preprocess/)、[`scripts/preprocess/`](scripts/preprocess/)
  和 [`library/preprocess/`](library/preprocess/)；修改前用 `rg` 定位当前调用链，不在总协议维护脚本清单。
- 缓存路径和 sidecar 命名不要随意改；改后需要迁移说明或兼容读取。
- caption source 支持 `.txt`、`.caption`、`captions.json` 和结构化 tag groups；variant cache 的
  `caption_shuffle_variants` / `caption_tag_dropout_rate` 按环境变量、配置、默认值的优先级解析。
  修改 caption 来源或 variant 参数后必须重建对应 text encoder cache。

## Daemon 和队列

- 命令入口：`tasks.py` 的 `daemon`、`daemon-attach`、`daemon-kill`、`daemon-terminate`。
- wrapper：`scripts/tasks/daemon.py`。
- daemon 实现：`scripts/daemon/`。
- WebUI 队列：`configs/web-training-queue/`。
- WebUI 历史：`configs/web-training-history/`。
- 队列失败策略、GPU 白名单、runtime config 和进度解析在
  `web/services/training/{queue,gpu,runtime_config,progress_parser,launcher,live_monitor}.py`。
- 不要硬编码 daemon 端口；发现机制以 pidfile / daemon metadata 为准。

## Custom Nodes

- 先看对应节点 `README.md`。
- Hydra loader 还要读 `custom_nodes/comfyui-hydralora/CLAUDE.md`。
- 修改 live `library/` 或 `networks/` 后，如影响发布节点，运行或提醒
  `python tasks.py vendor-sync`。
- 不要在 `_vendor/` 里做独立修复；先改源，再同步。

## 外部工具和父目录依赖

- 父目录里常见配套仓库包括 `../comfy/`、`../sam3/` 等。跨仓排障前先
  `ls ..` 确认真实存在的工具目录，不要假定所有外部依赖都已装好。
- `custom_nodes/`、ComfyUI 工作流、SAM3 和一些推理/预处理流程会间接依赖这些父目录仓库；
  改路径、说明文档或集成逻辑时要把这层依赖写清楚。

## 文档维护

- 文档入口是 `docs/README.md`。
- 当前 WebUI 用户与开发入口是 `web/frontend-next/README.md`；旧 Dragon/classic 文档只能作为
  明确标注的历史记录，不得将其描述为当前可用入口或故障回退路径。
- 根 `README.md` 只做项目介绍、部署快照和最高频入口；完整文档必须从根 README 明确链接到
  `docs/README.md`。
- 用户安装、WebUI 流程和启动命令变更：更新根 `README.md`。
- 文档索引、方法入口、坏链整理：更新 `docs/README.md`；如果分区有独立索引，也要同步更新
  对应 `README.md`。
- 新增 `docs/**/*.md` 时，必须让它从 `docs/README.md` 或一个分区索引可达。超过 5 篇文档、
  或长期增长的分区应维护自己的 `README.md`。
- 历史计划、完成报告、一次性上游合并材料、过期提案默认归档到 `_archive/docs/`，不要继续放在
  活跃 `docs/proposal/` 里。
- 活跃或半活跃提案放 `docs/proposal/`，归档时同步更新 `docs/proposal/README.md`、
  `docs/archive-index.md` 和 `_archive/docs/<subdir>/README.md`。
- 文档顶部可用状态块标注适用范围，特别是实验、历史、占位和归档文档：
  `状态：稳定 / 实验 / 历史 / 已归档 / 占位`、`适用版本：当前 main / 指定提交`、
  `入口命令：python tasks.py ...`、`相关代码：path/to/file.py`。
- 稳定能力使用说明：`docs/methods/`。
- 可运行但实验中的能力：`docs/experimental/`。
- 原理、数学、架构图解：`docs/structure/`。
- 配置、路径、环境变量和外置配置：`docs/configuration/`。
- WebUI 独立功能说明：`docs/features/`。
- 实验结论、失败路径、审计报告：`docs/findings/`。
- compile、kernel、显存和性能优化记录：`docs/optimizations/`。
- 活跃或半活跃提案：`docs/proposal/`。
- 过期或缺失上下文材料：`_archive/docs/`，并标注历史状态。
- bench 说明：对应 `bench/<method>/README.md`。
- 纯文档改动只对本次修改的文档路径运行 `git diff --check -- <changed-doc-paths>`；
  新增或修改链接时检查受影响链接，不默认运行测试套件。
- 大规模文档整理还要跑本地 Markdown 链接检查，确认真实坏链为 0；外部链接只在需要时人工抽查。
- 用户可见命令优先写 `python tasks.py <command>` 或 `.venv/bin/python tasks.py <command>`；
  `make <target>` 可作为兼容说明，但不要作为唯一入口。

## 验证策略

- 默认将功能实现与测试验收分开。日常开发以差异审阅、`git diff --check` 和必要的
  语法、类型或构建检查收尾；低风险文案、样式、简单配置和局部逻辑默认不新增测试、不运行测试套件。
- 验证范围由具体失败风险决定；跨文件、跨目录、改动行数或新增函数本身不触发扩大测试。
  优先复用能覆盖风险的已有用例，必要时只运行相关 case；一次必要检查通过后，只有新改动、
  失败或未解决的疑点才触发追加验证，避免重复跑相同检查。
- 新增或扩展自动化测试优先用于真实缺陷的回归、稳定对外接口的重要边界和关键运行时不变量。
  避免复述实现、穷举低价值组合；普通新行为不要求逐项补测试或解释为什么未补测试。
- 以下变化在交付相应行为前需要最小定向验收：用户数据删除或越界写入边界、latent shape
  和缓存 schema、checkpoint/adapter 保存加载和训练恢复、核心数值算法、模型族 fail-closed
  分派，以及改变已有配置或公开 API 的兼容契约。优先验证被改动的契约，不自动扩大到全套测试。
  本文件明确要求的 shape/invariant 测试仍须补充。关键风险的独立验收按上文委派规则执行。
- `test-core`、`test-integration`、`test-hardware`、`test-experimental`、`test-all`
  及完整收集默认放在集中验收或发布阶段，仅在有对应风险、明确发布要求或用户指定时选用，
  不作为普通开发交付的默认阻塞步骤。性能或数值结论必须有对应的测量证据；未测量时仅报告实现状态。
- 测试/验收可异步进行；主代理继续不依赖结果的工作。等待只用于必要的关键验收或发布条件，
  普通改动可以先交付实现并标明待验收范围；不要为了凑齐测试结果反复派发或升级测试套件。
- 执行必要的短定向测试时默认加 `timeout 60`；集中验收按任务设置明确时限，例如
  `timeout 300 .venv/bin/python tasks.py test-core`。超时、环境缺失或未执行不得报告为通过，
  不自动延长时限或重复重试。
- 需要项目 Python 依赖的验证命令，默认使用 `.venv/bin/python`，避免系统 Python
  缺少 torch、pytest 插件或本仓依赖导致误判。
- 大模型、真实训练、下载类验证不要默认执行；小 fixture 或 monkeypatch 用于局部契约验证，
  不作为真机数值或性能结论的证据。验收避免占用用户正在训练的 GPU，或争抢开发所需资源。
- 环境不足以完成关键验收时，可以交付实现，但必须明确尚未验证的契约和发布限制，
  不宣称相关能力已验证。普通低风险改动只需简短说明实际做过的检查。
- lint/format 会改文件时，只在范围明确时运行。

## 贡献规范

贡献等级、bench 证据、方法文档和 PR checklist 统一以
[`CONTRIBUTING.md`](CONTRIBUTING.md) 为准，本协议不维护第二份摘要。
测试的新增、范围与执行时机统一以上文「验证策略」为准；明确的 bench、数值及方法能力
证据要求在 PR 或发布验收阶段落实。

## 完成前检查

- `git diff --check` 对你改过的路径干净。
- 只改了任务相关文件，没有误碰用户数据目录。
- 命中关键风险的改动有必要的定向验收证据，或明确标注待验收的契约和发布限制；
  普通新行为不以新增测试作为完成条件。
- 用户可见命令、配置字段、WebUI 表单、文档入口保持同步。
- 如果改了 custom nodes 相关 live source，说明是否需要 `vendor-sync`。
- 最终回复用简短中文说明改了什么、实际验证了什么；有待验收项时区分“实现完成”和“验收通过”。
