# 历史任务全局搜索扩展筛选设计

状态：已批准（方案 A）  
日期：2026-07-27  
范围：WebUI「训练 → 历史任务」工具栏筛选  
相关代码：

- `web/static/index.html`（历史工具栏 DOM）
- `web/static/js/features/anima-app/state/history-state.js`
- `web/static/js/features/history-list/collections-workbench.js`
- `web/static/js/features/history-list/task-collections.js`
- `web/static/js/features/app-shell/event-listeners-setup.js`
- `web/static/js/features/app-shell/event-listeners-contract.js`
- `web/static/js/features/app-shell/beginner-tooltips.js`
- `web/static/js/features/history-detail/overview.js`（语义来源）
- `web/services/training/history_store.py` / `history_meta.py`

## 1. 目标

在历史任务「全局搜索」下方的现有筛选栏中，新增三个与历史详情 **概览 → 实时指标** chip 语义一致的下拉筛选：

| 筛选项 | 详情 chip 标签 | 真实含义 |
|---|---|---|
| 训练变体 | 训练变体 | 方法族（如 `lokr` / `lora` / `hydralora`），**不是**配置文件 stem `task.variant` |
| 预处理精度 | 预处理精度 | `preprocess_precision_preference` |
| 块交换精度 | 块交换精度 | `block_swap_transfer_dtype`（UI 文案沿用详情「块交换精度」） |

约束：

- 风格与现有 `类型 / 状态 / 归档 / 来源 / 排序` 完全一致（同级 `<label><span>…</span><select>…</select></label>`）。
- 不得偏离现有 history-manager 交互：默认「全部」、change 即重渲染、控件与 `historyManagerFilters` 双向同步。
- 不做无关重构；不改用户历史数据目录内容（只读 snapshot 推导列表字段）。

## 2. 非目标

- 不新增「训练精度」筛选（详情有 chip，但本次用户只点名三项）。
- 不把 `task.variant`（导入配置名如 `okkotsu_goddess_…`）当作训练变体筛选项。
- 不在启动路径强制回写 `meta.json`（方案 C 不做）。
- 不改详情页 chip 展示文案或视觉。
- 不改队列页、预处理列表其它筛选。

## 3. 背景与现状

### 3.1 现有筛选链路

1. DOM：`#history-filter-kind|state|archived|source` + `#history-sort-mode` + `#history-manager-search`
2. 状态：`historyState.historyManagerFilters`
3. 绑定：`event-listeners-setup.js` 的 `historyFilterMap`
4. 同步：`syncHistoryFilterControls()` / `historyManagerFilterDefault()`
5. 过滤：`historyManagerBaseFilteredTasks()`（`collections-workbench.js`）
6. 统计快捷：`applyHistoryStatFilter()` 会重建 filter 对象（只保留 sort）

### 3.2 语义真相源

详情概览用 `payload.config_toml`（即 `config.snapshot.toml` 文本）本地解析：

- `formatHistoryTrainingVariant(task, configText)`
- `formatHistoryPreprocessPrecision(configText)` → `preprocess_precision_preference`
- `formatHistoryBlockSwapPrecision(configText)` → `block_swap_transfer_dtype`

列表 API（`list_history_tasks` → `_history_summary`）当前**不**带这些字段，只暴露 `variant`（配置 stem）、路径、runtime 路径等。

## 4. 方案选择

| 方案 | 做法 | 结论 |
|---|---|---|
| **A. 列表摘要读 snapshot 补字段** | `_history_summary` 轻量解析 snapshot，吐出三字段；前端三 select | **采用** |
| B. 前端按需拉详情再筛 | 列表后再 N 次 detail | 拒绝：慢且别扭 |
| C. 启动写入 meta + 旧任务回填 | 改启动与迁移 | 拒绝：范围大，本次不必要 |

方案 A 与详情 chip 一致，旧任务可筛，约数百条小 TOML 可接受。

## 5. 数据契约

### 5.1 列表 task 新增字段

在 `_history_summary(meta, task_dir)` 返回中增加（始终存在，字符串）：

```text
training_variant: str          # 方法族小写；未知/无快照为 ""
preprocess_precision: str      # bf16|fp16|fp32|…；无则 ""
block_swap_precision: str      # bf16|fp8_e4m3|int8|…；无则 ""
```

命名说明：

- 前端筛选用 `training_variant`，避免与已有 `variant`（配置 stem）混淆。
- `block_swap_precision` 对应配置键 `block_swap_transfer_dtype`，标签文案仍用「块交换精度」。

### 5.2 解析来源优先级

对每个历史任务目录：

1. 优先读 `task_dir / "config.snapshot.toml"`（与详情 `config_toml` 同源）。
2. 若 snapshot 缺失或关键键缺失，**不**再扫 runtime 大文件（保持列表轻量）；字段留空。
3. 解析失败（坏 TOML / 读权限）→ 三字段均为 `""`，不抛、不阻断列表。

实现注意：可用 `tomllib` 解析 dict，也可用与前端一致的行级 key 扫描；优先 **tomllib 读 dict**，更稳。布尔/字符串规范化与详情推断规则对齐。

### 5.3 训练变体推断规则

与 `formatHistoryTrainingVariant` **行为对齐**（列表与详情同值），核心顺序：

1. chimera（`use_chimera_hydra` 或 module 含 chimera）→ `chimera`
2. ip_adapter / easycontrol / soft_tokens（flag 或 module 名）
3. loha / lokr / vera / glora（对应 `use_*`）
4. dora（`dora_wd` 或 `use_dora`）
5. reft（`add_reft`）
6. hydralora（`use_moe_style` 非空且非 false/none/0/off）
7. tlora（`use_timestep_mask`）
8. ortholora（`use_ortho`）
9. lora（`network_module` 含 `lora_anima`，或有有效 snapshot 但无 module 时默认 lora）
10. 回退：若 `task.variant`（小写，去 `-8gb` 后缀）落在 known 集合，则用之；`chimera_hydra` 归一为 `chimera`
11. 否则 `""`（详情显示 `-`，列表筛「非全部」时不匹配）

Known 集合（筛选项 + 归一目标）：

```text
lora, lokr, loha, vera, glora, dora, hydralora, reft, tlora,
ortholora, chimera, soft_tokens, ip_adapter, easycontrol
```

实现要求：

- **Python 与 JS 不得长期各写一套互漂逻辑。**
  - 后端：在 `web/services/training/` 增加小 helper（例如 `history_config_chips.py`），供 `_history_summary` 调用。
  - 前端：把 `overview.js` 内的 `formatHistoryTrainingVariant` / precision helpers 抽到可 import 的 shared 模块（例如 `history-list/config-chips.js` 或 `history-detail/config-chips.js`），overview 与 filter 共用。
  - 若短期为降风险只在后端算字段、前端只消费 API 字段做过滤，则 **过滤必须以 API 字段为准**；详情 chip 仍可走原函数，但应用同一规则文本/单测钉死两边输出一致。
- 推荐落地：**后端列表吐字段；前端筛选只读 task 字段；详情继续用 snapshot 文本，但 helper 抽到 shared 并加对拍测试**，避免二次维护。

### 5.4 精度字段规范化

- 转小写、去引号空白。
- 预处理：常见 `bf16` / `fp16` / `fp32`；其它原样保留（筛选项可动态并入）。
- 块交换：常见 `bf16` / `fp8_e4m3`；若出现 `int8` 等也原样保留。
- 空 / 缺省 → `""`。

## 6. 前端 UI 与状态

### 6.1 DOM（`index.html`）

在 `#history-filter-source` 与 `#history-sort-mode` 之间（或 source 之后、sort 之前）插入三个 label，风格照抄现有：

```html
<label>
  <span>训练变体</span>
  <select id="history-filter-training-variant">
    <option value="all">全部</option>
    <!-- 固定方法族 options，见下 -->
  </select>
</label>
<label>
  <span>预处理精度</span>
  <select id="history-filter-preprocess-precision">
    <option value="all">全部</option>
    <option value="bf16">bf16</option>
    <option value="fp16">fp16</option>
    <option value="fp32">fp32</option>
  </select>
</label>
<label>
  <span>块交换精度</span>
  <select id="history-filter-block-swap-precision">
    <option value="all">全部</option>
    <option value="bf16">bf16</option>
    <option value="fp8_e4m3">fp8_e4m3</option>
  </select>
</label>
```

训练变体固定 options（value = 显示文本，小写方法族）：

```text
lora, lokr, loha, vera, glora, dora, hydralora, reft, tlora,
ortholora, chimera, soft_tokens, ip_adapter, easycontrol
```

第一版选项表固定如上，**不做**历史动态 append。若后续常见 `int8` 等固定表外取值，再单独加动态选项。

布局：沿用 `.history-manager-tools` grid；训练页侧栏已是 2 列，新控件自然换行，**不强制改 CSS**，除非视觉明显溢出再微调。

### 6.2 状态键

`historyManagerFilters` 增加：

```js
trainingVariant: 'all',
preprocessPrecision: 'all',
blockSwapPrecision: 'all',
```

- `historyManagerFilterDefault(key)`：这三键默认 `'all'`（与 kind/state/source 相同分支）。
- `createHistoryState()` 初始值同步。

### 6.3 事件与契约

以下三处 map/列表同步加入：

| 位置 | 内容 |
|---|---|
| `event-listeners-setup.js` `historyFilterMap` | id → key |
| `task-collections.js` `syncHistoryFilterControls` | 同上 |
| `event-listeners-contract.js` | 三个新 id |
| `beginner-tooltips.js` | 三条简短中文说明 |

id → key：

```text
history-filter-training-variant   → trainingVariant
history-filter-preprocess-precision → preprocessPrecision
history-filter-block-swap-precision → blockSwapPrecision
```

### 6.4 过滤逻辑

在 `historyManagerBaseFilteredTasks()` 中，`source` 判断之后、`search` 之前（或同级任意稳定位置）增加：

```text
if trainingVariant !== 'all' and task.training_variant !== filter → exclude
if preprocessPrecision !== 'all' and task.preprocess_precision !== filter → exclude
if blockSwapPrecision !== 'all' and task.block_swap_precision !== filter → exclude
```

比较前对 task 字段 `String(...).trim().toLowerCase()`；空字符串在非 `all` 时不匹配。

可抽 `historyTaskMatchesChipFilter(task, filters)` 小函数，避免把 base filter 拉长；非必须。

### 6.5 统计卡片快捷筛选

`applyHistoryStatFilter` 重建 `next` 时：

- **重置**三个新键为 `'all'`（与 search/kind/state/source 被清空的行为一致）。
- `historyStatFilterIsActive`：**不**要求检查新键——若用户在 stat 激活后再改新筛，stat 高亮按现有 base 条件可能仍为 true；为更干净，可在 `base` 条件中要求三键均为 `all`。  
  **采用：三键均为 `all` 才允许 stat active**（避免「看起来像只按训练筛选，实际还叠了 lokr」的误导）。

### 6.6 搜索文本

`historyTaskSearchText` **并入**三字段（`training_variant` / `preprocess_precision` / `block_swap_precision`），便于全局搜索命中 `lokr` / `fp8_e4m3`；不改变 select 语义。

## 7. 后端实现要点

- 新 helper 模块宜小：读 snapshot 文本/dict → 三字段。
- `_history_summary` 调用 helper，写入 `out`。
- 不修改 `meta.json` 落盘；纯派生。
- 列表路径已有 per-task 目录访问；多读一个通常已存在的小文件。**第一版不做** mtime 进程内缓存；若列表明显变慢再加。
- 预处理任务：snapshot 若仍含精度键则照常填充；训练变体按同一规则，可能为 `""` 或方法族——筛选「训练变体=lokr」时预处理任务通常被排除，符合预期。

## 8. 测试

### 8.1 后端

- 构造临时 task_dir：`meta.json` + `config.snapshot.toml`（含 `use_lokr=true`、`preprocess_precision_preference="bf16"`、`block_swap_transfer_dtype="bf16"`）→ summary 三字段为 `lokr`/`bf16`/`bf16`。
- 无 snapshot → 三字段 `""`。
- 变体边界：`use_moe_style="shared_A"` → `hydralora`；`use_timestep_mask=true` → `tlora`；`chimera` 优先于 hydra。

### 8.2 前端（现有字符串/结构断言风格）

- `index.html` 含三个新 id 与中文 label。
- `historyFilterMap` / `syncHistoryFilterControls` / contract / tooltips 含新项。
- `historyManagerBaseFilteredTasks` 或等价逻辑包含三字段比较。
- `createHistoryState` / `applyHistoryStatFilter` 含新键默认 `all`。
- 若抽 shared helper：对 snapshot 样例与详情函数输出一致的单测或 node 断言。

### 8.3 验证命令（实现时）

```text
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_history.py -q
# 以及新增/相关的 history summary 测试文件
```

按改动面可再补 `tests/test_training_frontend_state.py` / modules 中 DOM contract 相关断言。

## 9. 风险与兼容

| 风险 | 缓解 |
|---|---|
| 列表变慢 | snapshot 小；第一版无缓存；若明显变慢再加 mtime 缓存 |
| 前后端变体推断漂移 | shared 规则 + 对拍测试；列表过滤以 API 字段为准 |
| 旧任务无 snapshot | 字段空；筛「全部」仍可见；筛具体值时隐藏（可接受） |
| 与 `variant` 名混淆 | API/状态键用 `training_variant` / `trainingVariant` |
| 工具栏拥挤 | 沿用现有 grid 换行；不改主题除非必要 |
| `applyHistoryStatFilter` 丢新键 | 显式重置为 all，并纳入 stat active 判定 |

## 10. 完成标准

- 历史工具栏可见三个新下拉，默认「全部」，风格一致。
- 选 `训练变体=lokr` 时，仅 `training_variant === 'lokr'` 的任务保留；与详情概览 chip 为 `lokr` 的任务一致。
- 预处理/块交换精度同理。
- 相关 pytest 通过；`git diff --check` 干净。
- 不触碰 `configs/web-training-history` 用户数据写入。

## 11. 实现顺序（供 plan 展开）

1. 后端 helper + `_history_summary` 字段 + 单测  
2. 前端 DOM + state + 事件 map/contract/tooltips  
3. 过滤逻辑 + stat filter 重置/active  
4. （建议）抽 shared 变体/精度 helper，详情改用 import  
5. 前端 history 相关测试断言更新  
6. 定向 pytest + 完成说明  
