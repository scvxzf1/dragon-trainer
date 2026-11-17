# 待推送变更审计与线上 dev 推送计划书

状态：待执行；本轮仅完成范围核查与计划编制，尚未完成代码审计、测试、提交或推送。
日期：2026-09-27
目标仓库：`https://github.com/scvxzf1/anima_lora_webui`
实际 remote：`origin`；目标分支：`dev`

## 1. 目标与完成定义

审计全部待推送提交及拟纳入的工作区改动，修正发布阻塞问题，生成可复现的候选提交，完成普通 Git 推送并核验远端结果。执行范围以本计划冻结的文件清单和候选提交为准。

完成需同时满足：

1. 已提交与未提交改动均有范围记录；新增源码、测试、配置、文档依赖完整。
2. 所有 P0/P1 问题关闭，必要验证通过；跳过、超时及未执行项有明确影响说明。
3. 通过验证的源码树与待推送候选提交一致，不依赖原工作区未跟踪文件或本机设置。
4. `origin/dev` 包含经审计的候选提交；记录推送前后 hash、命令结果及远端 CI 状态（若配置）。
5. 原工作区的本机配置、用户数据、未纳入改动保持原样；交付残留清单。

本计划的“线上”指 GitHub `dev`。推送不会自动完成 WebUI 静态产物部署；`main` 发布、运行服务替换、真实训练及外部 provider 写操作不属于此次默认范围。

## 2. 已核实的基线

以下为 2026-09-27 执行 `rtk git fetch origin --prune` 后的快照；执行阶段必须重新核对。

| 项目 | 核实结果 |
| --- | --- |
| 当前分支 | `dev` |
| 本地 HEAD | `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27` |
| `origin/dev` | `2e96bd0ce3a7c2803d542e5b0b7dfeac77369d8b` |
| ahead / behind | `13 / 0` |
| 已提交净差异 | 740 文件，新增 80,472 行、删除 3,703 行 |
| 未提交文件 | 修改 193、删除 10、未跟踪 106，共 309；按 `--untracked-files=all` 展开统计 |
| 其他 remote | `krea-upstream` 指向 `scvxzf1/krea2-webui`，不是本计划推送目标 |

文件统计不包含本计划和新增索引行；已提交、未提交两组文件有重叠，不能相加作为唯一文件总数。

13 个待推送提交（旧到新）：

| 提交 | 范围 |
| --- | --- |
| `1b01b531` | 自适应预处理批次与 bucket 尺寸 |
| `2737308c` | family cache 与自适应 block swap |
| `0e27ff5c` | NF4 DoRA、OrthoLoRA、ReFT 契约 |
| `86e1582c` | swap 校准与 pipeline 实验探针 |
| `4224ac44` | Web 配置校验与分页工作区 API |
| `60e42936` | 旧版配置工作区与模块契约 |
| `1d1e6b5e` | Next 训练、历史、打标和蒙版工作台 |
| `a00210e1` | 能力、实验和验证边界文档 |
| `04e5265d` | 实验精度与恢复 runtime |
| `c9804dc8` | adaptive 实验边界与 UI |
| `bb4fe09d` | 数据集蒙版入口 |
| `f8bde2e4` | 忽略本机调试产物 |
| `5b60d111` | adaptive 诊断与前端状态测试修正 |

工作区还包含 Qwen Image 2.1、新的 adaptive precision 契约、daemon 历史/GPU 快照、Next 页面与隔离验证等更新。审计范围不能缩成最后一个提交，也不能只审 `git diff HEAD`。

## 3. 阶段 A：冻结范围、隔离候选

交付：文件纳入清单、排除清单、基线提交与工作区快照摘要。

1. 重查 remote、branch、staged/unstaged/untracked 状态；保存路径、状态、内容哈希及待推提交列表。原始差异和可能含私有信息的日志只保存在本机审计目录。
2. 分别登记 `origin/dev..HEAD`、index、未暂存和未跟踪文件。未跟踪源码不能因默认不提交而遗漏：逐个判断其 import、测试、配置和文档依赖。
3. 默认排除下表内容，原工作区不恢复、不删除、不暂存。若排除后功能不完整，提取可发布的通用部分或修复默认配置，不把本机数据带入仓库。
4. 从新 fetch 的 `origin/dev` 建立独立 `codex/audit-push-20260927` 分支/worktree。远端仍为当前基线时，fast-forward 纳入原 HEAD 的 13 个提交；若远端推进，先比较并正常合并，解决冲突后重新验证，不改写原 `dev` 历史。
5. 按明确路径/补丁白名单把选定工作区改动带入候选 worktree，保留原工作区。记录每个源文件哈希，搬运期间发生变化时重新冻结该文件，避免混入一半更新。
6. 候选环境使用自己的依赖与临时配置/队列/历史/输出根；不复制模型或训练结果，不让候选服务连接现有 daemon 或真实队列。

| 内容 | 默认处置 |
| --- | --- |
| `configs/web-ui-settings.toml` | 排除本机覆盖；若含通用功能变更，逐 hunk 提取并保持默认值可移植 |
| `configs/datasets/` 下 10 个删除及 `colorize.toml` 修改 | 保留本机状态，默认不发布这些删除或本机数据路径 |
| `configs/sample-prompts/imported/` 两份新增 TXT | 作为用户导入数据排除 |
| 模型、缓存、日志、输出、队列、历史、依赖目录 | 不纳入提交，不作为清理对象 |
| 新 Qwen method、实现、测试、通用说明 | 经审计确认后成组纳入，不能只提交接线而漏实现 |
| 审计截图、探针结果、实验脚本 | 按可复现性、隐私和体积逐项裁定；报告只引用可发布证据 |

退出标准：每个变更有纳入/排除理由，候选无意外丢文件，原工作区内容未被覆盖。

## 4. 阶段 B：按风险审计并修复

主代理负责方案取舍和修改。可按下列边界派发独立只读探子，要求返回严重度、`file:line`、触发条件、关键原文和验证建议；主代理亲读将修改的代码与架构/设计原文。探子不承担并行写入。

| 审计批次 | 重点 | 必须取得的证据 |
| --- | --- | --- |
| 模型族与数据链 | Qwen Image 2.1 registry/weights/attention/edit pairs；Anima/Krea/Z-Image 回归；latent、文本和图像缓存命名/shape | 未知 family fail closed，能力限制一致，编辑配对校验，缓存隔离与旧配置兼容 |
| 训练与恢复 | adaptive precision、OOM/retry、block swap、checkpoint/resume、pipeline、GPU 选择 | 数值/dtype 边界；失败不伪装成功；恢复状态和数据游标一致；实验默认与声明一致 |
| Adapter 与预处理 | 13 个已提交提交内的 NF4 DoRA/OrthoLoRA/ReFT、cache batching、bucket | padding、native shape、lazy loading、compile-after-apply 不变量；必要时检查 vendor 发布副本 |
| Web 后端与 daemon | revisioned text、dataset pairs、路径边界、队列/历史/GPU 记录、tagging | 冲突 409、越界拒绝、幂等与结果未知、根目录隔离、凭据只在服务端 |
| Next 与旧前端 | 配置/数据集/图片工作台、caption、history、tools、入口和静态资源 | 草稿保护、请求乱序、mutation 不重试、断连/空/错/加载态；旧 UI 和 cache token 回归 |
| 交付与文档 | 依赖锁、启动脚本、命令入口、文档索引、构建产物 | 冻结依赖可安装，干净 checkout 可构建，能力声明不超出证据，引用可达 |

问题分级：P0 为泄密/数据破坏等严重问题；P1 为训练结果错误、主要流程不可用或默认路径不安全；P2 为有限影响问题；P3 为非阻塞改进。每项记录复现、修复提交和复验结果。P0/P1 必须清零；P2 需修复或说明范围、替代路径和接受理由，不能只标“已知”。

特别复查以下历史遗留证据；它们是复验线索，不是当前代码已经失败的结论：

- [9 月 13 日审计收尾](../findings/uncommitted_audit_closure_20260913.md)：Core 300 秒超时；后续 fail-fast 有 bitsandbytes CPU backend 失败，单测独立通过但整套污染原因未定位；完整 E2E 修复后未全量重跑；vendor 尚未同步。
- [9 月 25 日 S5 记录](../findings/dragon_next_s5_release_candidate_20260925.md)：HTTP raw-config 契约测试失败、S1 人工/读屏矩阵及 S4 部署态联通未关闭，整体为 `BLOCKED`。
- Next README 当前描述根路径默认进入 Next，而 S5 历史记录没有完成入口切换签收。核查当前路由、默认入口、实际部署文档是否一致；历史报告保留其时间语义，在新报告中关闭或继续列明缺口。
- revision 隔离已有阶段证据，但不带 revision 的旧客户端兼容写入不等同 Next 并发保证。针对跨标签保存、旧客户端和根切换重新核验。

退出标准：高风险调用链和旧遗留项均有明确结论，没有用跳过测试、降低断言或删除失败用例掩盖问题。

## 5. 阶段 C：分层验证

所有命令在候选 worktree 执行。先记录 Python/uv、Node/pnpm、系统及依赖版本；Python 使用候选 `.venv`，前端遵循 Node `>=24.13.0 <25`、pnpm `11.9.0`。锁文件安装需先核对依赖来源与 CUDA 约束，不升级依赖来消除未知失败。

每条验证记录：候选源码树/提交 hash、命令、版本、退出码、耗时、passed/failed/skipped、报告路径。超时记为超时，不能记 PASS；历史报告不能替代本轮结果。

### C1. 静态与定向回归

- 审查所有待推提交及新增文件中的密钥、本机路径、大文件和数据误纳入；只记录命中文件及类别，不把秘密打印进报告。已进入待推历史的秘密不能靠后续删除解决，需在推送前处理泄漏及历史可发布性。
- 运行变更范围的 Python lint、配置/文档完整性、import/CLI 和相关旧前端契约测试；lint 不自动格式化无关文件。
- 按模块执行 CPU/小 fixture 定向测试，至少覆盖模型族 fail-closed、Qwen edit/weights/attention/swap、adaptive precision/retry/resume、daemon/history/GPU、Web HTTP/路径/revision、缓存与 adapter 不变量。
- 在执行测试前核查新增测试的 marker 与 fixture，避免未标记用例隐式下载模型、占用真实 GPU 或读取用户数据。

起步命令示例（按实际测试清单补充分组，不用这一组代替全部覆盖）：

```bash
rtk test timeout 60 .venv/bin/python -m pytest -q \
  tests/test_model_family_registry.py tests/test_model_family_fail_closed.py
rtk test timeout 60 .venv/bin/python -m pytest -q \
  tests/test_web_static_server.py tests/test_web_http_contracts.py
rtk test timeout 60 .venv/bin/python tasks.py test-fast
rtk test timeout 300 .venv/bin/python tasks.py test-core
```

Core 若 300 秒未完成，先保存失败/慢例证据，再按显式 nodeid 清单分批运行、核对收集总数与覆盖，不以一次 fail-fast 通过代替全套。涉及多进程/HTTP 集成的变更运行对应 integration 层；hardware/probe 仅运行已审定且符合资源授权的子集。

### C2. 前端单测、浏览器和候选包

```bash
rtk test timeout 180 .venv/bin/python tasks.py web-next-check
rtk test timeout 600 .venv/bin/python tasks.py web-next-e2e
rtk test timeout 180 env DRAGON_NEXT_DESTINATION=/tmp/dev-push-audit-next-candidate \
  .venv/bin/python tasks.py web-next-build
rtk test timeout 60 env DRAGON_S5_CANDIDATE=/tmp/dev-push-audit-next-candidate \
  node web/frontend-next/scripts/verify-release-candidate.mjs
```

执行时使用本轮唯一临时目录；上面的目录仅为示例。E2E 预先选择空闲 `DRAGON_E2E_PORT` 并设置独立输出目录；配置没有命名为 chromium 的 project，不额外传 `--project=chromium`。首次完整运行用于建立总清单，失败修复后按受影响范围复验，最终必须有覆盖候选版本的完整结果。

这些验证各有边界：Vitest 不验证 Python；Playwright 默认 mock API/WS；候选脚本验证资源与模拟发布/回退，不验证真实业务写入。默认 build 会更新本机静态目录，因此必须显式设置候选输出目录。

### C3. 真实后端隔离契约与页面核验

- 在临时配置、历史、队列和输出根运行 aiohttp fixture/隔离服务，复用真实 route/service，测试保存、409、队列/历史 mutation、蒙版/打标写回的本地 fixture 路径。
- 使用 `ANIMA_CONFIGS_ROOT`、`ANIMA_TRAINING_HISTORY_ROOT`、`ANIMA_TRAINING_QUEUE_ROOT` 及临时设置中的 `output_root`；启动前核查所有服务实际解析路径和 daemon 发现行为，不能仅凭设置三个环境变量推断已隔离。
- 浏览器必须访问该候选对应的静态资源与后端，记录入口/资源 hash。对候选服务运行 `verify-production.mjs`，显式设置 `DRAGON_VERIFY_URL` 和 `DRAGON_VERIFY_OUTPUT`；该脚本仅允许 loopback，拦截非只读 API 请求。
- 人工检查主要操作的键盘/焦点、窄屏、缩放、草稿离开保护、旧 UI 回退入口。自动截图通过不能替代读屏或人工矩阵的未执行项。
- GPU 短训、真实续训/停止、provider 调用、下载和用户 TXT 写回不是上述隔离测试。若其证据对默认功能放行必需，则提出明确资源、时长、目录和影响后执行；缺证据时修正功能边界/默认启用方式，不能直接宣称已验证。

退出标准：必需门禁全部通过；实验未验证项明确标注且不会通过默认路径影响稳定能力。默认工作流的重要缺口不能仅凭“推到 dev”豁免。

## 6. 阶段 D：组织提交与候选放行

1. 保留已有 13 个提交；后续按依赖顺序拆分模型/数据、训练 runtime、Web 服务、前端、文档与修复，每组包含必要测试和使用说明。跨层不可拆的功能保持原子提交。
2. 使用明确文件白名单暂存，不使用 `git add -A`。检查 staged diff、删除项、新增实现和锁文件；本计划及其索引若纳入交付则一并提交。
3. 从提交生成干净 checkout 验证依赖完整性：不借用原工作区未跟踪源码、用户配置或生产静态产物。已有验证若基于相同源码树且环境一致可复用；发生源码/依赖/合并修正则重跑受影响门禁。
4. 检查 live source 是否影响 ComfyUI vendor；需要同步时使用仓库 vendor-sync 入口，在候选树审查生成差异并跑节点契约测试。不能让用户得到一套已修 live source 和一套过期发布副本。
5. 写审计报告与验收表，所有结果绑定候选提交。扫描内容覆盖全部待推历史与最终树；核对候选外未提交改动是否仅为生成物。

放行表：

| 门禁 | 放行要求 |
| --- | --- |
| 范围与数据保护 | 纳入/排除清单完整，无本机数据或秘密进入待推历史 |
| 代码审计 | P0/P1 清零，P2 结论可复核 |
| 后端/核心 | 定向、Core 及所需集成覆盖完成，历史失败已复验归因 |
| 前端 | TS、Vitest、E2E、候选 build 与资源检查通过 |
| 默认流程 | 真实后端隔离契约通过，默认入口与文档一致，遗留验收缺口已处置 |
| 可交付性 | 干净 checkout 可复现，vendor/文档/依赖一致 |
| 候选身份 | 记录唯一候选提交、tree hash 和构建证据 |

## 7. 阶段 E：完成推送并验收

所有门禁通过后执行，不用强推绕过分支变化。以下命令在隔离候选分支执行；`HEAD:dev` 明确把该候选写到 `origin/dev`，避免切换或修改仍承载用户改动的本地 `dev`。

```bash
rtk git remote -v
rtk git status --short --branch
rtk git fetch origin --prune
rtk git rev-list --left-right --count HEAD...origin/dev
rtk git log --oneline origin/dev..HEAD
rtk git diff --check origin/dev...HEAD
rtk git merge-base --is-ancestor origin/dev HEAD
rtk git push --dry-run origin HEAD:dev
rtk git push origin HEAD:dev
rtk git fetch origin --prune
rtk git rev-parse HEAD origin/dev
rtk git ls-remote origin refs/heads/dev
```

- ancestor 检查必须成功；若远端已推进且不在候选祖先链，停止推送、正常合并并重验受影响部分，再重新放行。
- dry-run 只验证推送可行性，不替代真实推送与远端核验。被并发更新拒绝时重新 fetch，不能 force。
- 推送成功后远端应等于候选 hash；若其他协作者随即推进，检查候选是否为远端祖先并审查新增差异，不能误报候选丢失或覆盖其工作。
- 若仓库配置 CI，核验该候选的检查结果；无 CI 则明确记录。必需检查未完成时不得记整体完成。
- 无直写权限或分支保护要求 PR 时，保留候选，走正常功能分支/PR 路径；PR 待合入时只汇报“等待合入”，不宣称 dev 已推送完成。

交付报告包含：仓库 URL、remote、目标分支、推送前后完整 hash、提交清单、验证结果、遗留边界、原工作区未提交/未跟踪项，以及“已推送 dev；main/运行服务部署未执行”的准确范围说明。

## 8. 失败处置与回退

- 审计或测试失败：保持候选可追踪，修复或缩减相应功能范围并重新验证；不清理原工作区来消除失败。
- 推送失败：保留本地提交与报告，按权限/网络/并发原因处理；失败不能记为完成。
- 已推送代码需要撤回：先核实并确认仓库、`origin/dev`、需撤回提交和影响，再用新的 `git revert` 提交，经定向验证普通推送；不重写共享历史。
- Git 回退不会回滚训练数据。前端静态回退是独立部署动作，本计划仅在临时候选根演练，不覆盖当前服务的 `index.html`。

## 9. 执行进度记录

| 阶段 | 当前状态 | 交付物 |
| --- | --- | --- |
| 基线核查与计划编制 | 完成 | 本计划、remote/提交/文件规模核实 |
| A 范围冻结与隔离 | 待执行 | 文件 manifest、排除清单、候选 worktree |
| B 审计与修复 | 待执行 | 分域问题表、修复及复验记录 |
| C 分层验证 | 待执行 | 测试/浏览器/候选包证据 |
| D 提交与放行 | 待执行 | 候选 hash、干净 checkout 验证、放行表 |
| E 推送与验收 | 待执行 | 远端 hash、CI 结果与最终报告 |

执行按退出标准推进，不预先承诺固定总耗时；主要不确定项是 Core 整套初始化失败、Next 历史验收缺口以及新增模型/精度实验的证据边界。计划执行结束后，将一次性材料按文档归档规则迁移并更新索引。
