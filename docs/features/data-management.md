# 数据管理

状态：稳定
适用版本：当前 WebUI Next 工作台 `/next/data`
入口：Next 工作台侧栏「数据管理」，路径 `/next/data`
相关代码：`web/services/data_management_service.py`、`web/frontend-next/src/features/data-management/`

## 能做什么

数据管理页集中查看项目中的模型、原始数据、训练历史、队列、配置和运行产物。打开页面时先取根目录元数据（`GET /api/data-management/roots?mode=meta`），卡片立即渲染；每个根目录的容量再由自己的 `entries` 请求并发回填，慢的目录不会阻塞首屏。容量用 `os.scandir` 递归统计并在服务端按物理路径缓存，重叠根目录（如 `output` 与 `output/runs`）只扫描一次；下钻直接读缓存。扫描受文件系统权限影响时，页面会标记为“部分统计”。

点击“刷新”会发起一次完整重扫（`?refresh=true`），结果同时写回每个根目录的容量缓存。

可清理范围固定在项目输出和后处理数据集两类根目录，以及带 WebUI runtime 标记的训练运行目录：

- `output/` 和 `post_image_dataset/` 内任意深度的普通文件或目录，只要没有被运行中/排队中的任务引用；
- 带 WebUI runtime 标记的训练运行目录。

模型、原始图像、配置、训练历史和训练队列只读；符号链接、根目录和越界路径始终受保护。页面不会把任意配置中的外部路径自动加入删除范围。

## 清理流程

1. 在同一个数据目录中选择一个或多个可清理条目。
2. 点击“清理所选”，先查看服务端生成的清理计划。
3. 核对条目和大小，勾选“我已核对上述列表，并确认永久删除”。
4. 服务端会重新检查路径、目录状态和运行中任务；预览变化、条目被引用或存在活动任务时，提交会被拒绝。

清理是永久删除，不提供回收站。训练、预处理、生图测试或打标任务启动时会避开正在执行的清理租约；清理期间新的相关任务也会被阻止。

## 检查点与权重智能清理

训练跑完几十组实验后，占空间的通常不是最终权重，而是每个实验里的中间检查点、`optimizer.bin` / `scheduler.bin` / `random_states.pkl` 等断点恢复状态。页面顶部的「检查点与权重智能清理」按钮（以及选择条目后悬浮栏里的同名入口）打开策略抽屉，按**每个实验单元独立**执行保留策略。

### 识别与分组

扫描受限于所选目录（最多 64 个目录、6 万文件、1.2 万目录、10 层深度），不跟随符号链接。每个“包含检查点 / 主权重 / 恢复状态的目录”是一个独立实验单元，策略只在单元内部排序和截取，绝不跨实验混算。文件角色：

| 角色 | 判定 | 默认动作 |
| --- | --- | --- |
| 中间检查点 | 带 `step` / `epoch` / `iter` / `ckpt` / `checkpoint` 数字标记的文件或目录，如 `...-step1500-step00001000.safetensors`、`checkpoint-500/` | 由保留模式决定 |
| 最终 / 主权重 | 单标记权重（`...-step1500.safetensors`）、无编号主权重，或名称含 `final` / `best` / `last` / `ema` | 保留 |
| 优化器 / 恢复状态 | `optimizer*`、`scheduler*`、`scaler*`、`rng_state*`、`random_states*`、`*-state/`、`*state_dict*` | 按开关清理 |
| 实验内缓存 | `cache/`、`latents/` 等目录，以及目录内或名称含 cache 的 `.npz` | 默认保留，需显式开启 |
| 分析中间张量 | `dump/`、`activation(s)/` 目录、`.npy`，以及缓存目录之外的 `.npz` | 默认保留，需显式开启 |
| 配置 / 日志 / 指标 | `.json` `.jsonl` `.tsv` `.csv` `.yaml` `.toml` `.log` `.txt` `.md` `.png` 等 | 永远保留 |
| 符号链接 | 任意软链（含 `data/cache/**/*.npz` 这类共享缓存软链） | 扫描阶段静默跳过：不跟随、不删除、不报错 |

数字按自然序比较（`step1000` 排在 `step200` 之后），日期形态的 8 位数字（`20260904`）不会被误判成训练步数，目录名也不会走“末尾数字”回退规则，避免把 `default_20260904-1315` 这类日志目录当成检查点。

### 策略参数

- 排序基准：训练步数 / Epoch（默认，从新到旧）、修改时间、文件体积；方向可切换。
- 保留模式：`keep_top_n`（保留前 N 个）、`delete_top_m`（清理前 M 个）、`keep_first_last`（保留最早 X + 最新 Y）、`delete_all_checkpoints`（清理全部中间检查点）。
- 权重规则：`keep_final_weights`（默认开）、`keep_weight_patterns`（逗号分隔，支持 `*best*` 通配或关键词）、`ensure_at_least_one_weight`（默认开，保底至少留 1 个权重）、`clean_optimizer_states`（默认开）。
- 空间扩展：`clean_data_cache`（实验内预处理缓存）、`clean_analysis_tensors`（分析 / 调试中间张量）默认关闭；只删除明确识别的缓存 / 张量文件，包含未知文件、子目录、权重、日志或预览图的泛名目录会整体跳过。汇总 `.json` / `.tsv` / `.md` 始终保留。
- 单实验豁免：`exclude_units`（单元相对路径列表）让指定实验完全不参与清理，汇总容量实时扣减，并进入 `plan_revision`。
- 被阻止的条目（包含日志/预览的检查点或缓存目录、活动任务引用）从清理计划中剔除并单独列出，不阻塞其余安全条目；符号链接不再进入阻止列表。

### 交互

抽屉为左右双栏（宽 1180px）：左栏配置规则并独立滚动，右栏是实时预览 + 常驻底部确认区，避免多层滚动条和按钮被挤出视口。

- 右栏默认只显示“有待清理项”的实验，并按预计释放空间从大到小排序，0 B 单元不再刷屏；取消勾选“仅显示有待清理项的实验”可查看全部。
- 每个实验左侧的复选框默认勾选，取消即豁免该实验（服务端按 `exclude_units` 重算），顶部汇总和确认按钮的释放容量同步扣减。
- 改动任意规则、作用范围或豁免会立即撤销确认，并在 350ms 后自动重算；确认按钮在草稿与服务端计划同步前保持禁用。明细按实验单元分组，按释放量排序，所有会被提交删除的分组都可展开核对。
- 确认前必须勾选“我已核对…”，提交时服务端重新生成计划并比对 `expected_items` 与 `plan_revision`，目录状态变化会返回 409 要求重新预览。

### 接口

- `POST /api/data-management/retention/preview`：`{root_id, dirs, policy, exclude_units}`，返回分组明细（含 `excluded` 标记）、阻止项、`exclude_units` 回显与 `plan_revision`。
- `POST /api/data-management/retention/cleanup`：追加 `expected_items`、`plan_revision`、`exclude_units`、`confirmed: true`，复用普通清理的路径/符号链接/活动任务/租约校验。

删除同样经过 `data_cleanup_lease()` 与 `_cleanup_lock`，单批最多 400 项；在生成预览时先统计完整候选集合，超过时返回 400，绝不截断后静默部分执行。

## 相关验证

```bash
.venv/bin/python -m pytest -q tests/test_data_management.py tests/test_data_management_retention.py
pnpm --dir web/frontend-next typecheck
```
