# Web 前端石山只读体检 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按已确认 spec 对当前工作树做一次全前端只读健康体检，填实审计快照（四域分数、Top10 债、先别动/可先动、命令摘要），不改任何业务代码。

**Architecture:** 先冻结 branch/HEAD/dirty 快照；再并行只读采集结构债、护栏测试、CSS/DOM、配置表面四类证据；最后由父代理按评分卡加权打分并写入同一 design 文档的 §8。禁止在本 plan 中修改 `web/static/**` 业务逻辑。

**Tech Stack:** bash、`rg`、`python3` 本地扫描、`.venv/bin/python -m pytest`、Markdown 审计文档。

**Spec:** `docs/superpowers/specs/2026-07-11-web-frontend-boulder-audit-design.md`  
**Scorecard:** `docs/features/frontend-health-scorecard.md`  
**Related design:** `docs/superpowers/specs/2026-07-11-frontend-config-optimization-design.md`

## Global Constraints

- 用户可见说明用简体中文；路径/命令/标识保持仓库原样。
- 测试一律 `timeout 60` + 优先 `.venv/bin/python`。
- **禁止**修改 `web/static/**`、训练语义、用户 history/queue/output/models。
- **允许**修改的写集仅限：
  - `docs/superpowers/specs/2026-07-11-web-frontend-boulder-audit-design.md`（填 §8 及状态行）
  - 可选：`docs/superpowers/README.md` / `docs/features/frontend-health-scorecard.md` 增加“最近复评指针”（非必须）
- 分数必须有证据；无证据不得上调相对 IR5 的叙事。
- IR5 ~78 是估计口径；本次以实测为准。
- 工作区若 dirty，报告标题与 §8 必须写「当前工作树」并附 dirty 摘要。
- max_depth=1：子任务只读，不再 spawn；最终文档串行由执行者写入。
- 每个 Task 结束要有可验收产物；最后统一 commit 审计结果。

## File Map

| 区域 | 文件 | 责任 |
|---|---|---|
| 设计 + 快照 | `docs/superpowers/specs/2026-07-11-web-frontend-boulder-audit-design.md` | §1–7 设计已冻结；本 plan 只填 §8 |
| 评分尺子 | `docs/features/frontend-health-scorecard.md` | 四域权重与门禁命令模板 |
| 入口 | `web/static/app.js` | 确认仍为薄 bootstrap（只读） |
| 装配 | `web/static/js/features/anima-app/index.js` | bridge/chunk 加载顺序（只读） |
| 过渡层 | `web/static/js/features/anima-app/chunks/*`、`helpers/*-bridge.js` | 结构债主战场（只读） |
| 业务 feature | `web/static/js/features/{config-form,dataset-editor,toml-manager,app-shell,history-*,queue,...}` | 反向依赖与体验抽查（只读） |
| 样式/DOM | `web/static/style.css`、`web/static/css/*`、`web/static/index.html` | C 域证据（只读） |
| catalog | `web/static/js/config/catalog/*` | D 域抽查（只读） |
| 护栏测试 | `tests/test_training_frontend_modules.py`、`tests/test_training_frontend_dom.py` | B 域证据 |

## Baseline Numbers (design-time；执行时必须复测)

| 指标 | 设计摸底约值 | 执行动作 |
|---|---:|---|
| features 目录 | 19 | 复测 |
| chunks | 45 / ~14.4k 行 | 复测 |
| `*-bridge.js` | 37 | 复测 |
| `legacyRoot = globalThis` bridge | 24 | 复测名单 |
| feature→chunks 反向 import 文件数 | 12 | 复测边列表 |
| R0 总分 | 61 / D | 对照 |
| IR5 估计总分 | ~78 / C+ | 对照，不照抄 |

---

### Task 1: 冻结工作树快照

**Files:**
- Modify: `docs/superpowers/specs/2026-07-11-web-frontend-boulder-audit-design.md`（仅 §8 表头字段：分支/HEAD/dirty；可先记在草稿纸，Task 6 一并写入也可——**本 Task 要求把快照原文保存到执行笔记，并在 Task 6 写入 §8**）
- Read: 无业务写

**Interfaces:**
- Consumes: 无
- Produces: 快照对象  
  `AuditSnapshotMeta = { branch: str, head: str, dirty_summary: str, captured_at: str }`

- [ ] **Step 1: 采集 git 元数据**

Run:

```bash
git status --short --branch
git rev-parse --short HEAD
git rev-parse --abbrev-ref HEAD
date -Iseconds
```

Expected: 打印当前分支名（例如 `feat/...` 或 `main`）、短 SHA、可能有大量 `M`/`??` 行。

- [ ] **Step 2: 生成 dirty 摘要（最多 30 行 + 计数）**

Run:

```bash
python3 - <<'PY'
from subprocess import check_output
raw = check_output(['git', 'status', '--short'], text=True)
lines = [ln for ln in raw.splitlines() if ln.strip()]
print(f'dirty_count={len(lines)}')
# 按路径前缀粗分
from collections import Counter
c = Counter()
for ln in lines:
    path = ln[3:].split(' -> ')[-1]
    top = path.split('/', 1)[0] if '/' in path else path
    c[top] += 1
print('by_top:')
for k, v in c.most_common(15):
    print(f'  {k}: {v}')
print('--- first 30 ---')
print('\n'.join(lines[:30]))
PY
```

Expected: 得到 `dirty_count` 与 top 目录分布。若 `dirty_count=0`，§8 写「工作区干净」。

- [ ] **Step 3: 固定 AuditSnapshotMeta 文本块（复制到后续 Task）**

把下面模板填实（执行者本地保留）：

```markdown
| 项 | 值 |
|---|---|
| 分支 | <branch> |
| HEAD | <shortsha> |
| dirty 摘要 | <dirty_count> 项；top: ... |
| 采集时间 | <ISO time> |
```

- [ ] **Step 4: 本 Task 验收**

- 已有 branch + short HEAD
- dirty 有计数（可为 0）
- **未**改任何业务文件

---

### Task 2: 结构债扫描（S1 structure-auditor）

**Files:**
- Read: `web/static/js/features/**`
- Read: `web/static/app.js`
- Produce data for §8（Task 6 写入）

**Interfaces:**
- Consumes: `AuditSnapshotMeta`
- Produces: `StructureEvidence`：

```text
StructureEvidence = {
  features_count: int,
  chunks_count: int,
  chunks_lines: int,
  shim_chunks: list[str],          # 文件几乎只有 re-export
  heavy_chunks: list[{path, lines}],  # 行数 Top10
  bridges_count: int,
  legacy_root_bridges: list[str],
  reverse_import_edges: list[{from_file, to_chunk}],
  app_js_lines: int,
  index_js_notes: str,             # 装配是否分组并行
}
```

- [ ] **Step 1: 规模扫描**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path('web/static/js/features')
features = sorted([p for p in root.iterdir() if p.is_dir()])
chunks = sorted(Path('web/static/js/features/anima-app/chunks').glob('*.js'))
bridges = sorted(Path('web/static/js/features/anima-app/helpers').glob('*-bridge.js'))

def lines(p: Path) -> int:
    return len(p.read_text(encoding='utf-8', errors='ignore').splitlines())

print('features', len(features))
print('chunks', len(chunks), 'lines', sum(lines(p) for p in chunks))
print('bridges', len(bridges), 'lines', sum(lines(p) for p in bridges))
print('app.js', lines(Path('web/static/app.js')))
print('anima-app/index.js', lines(Path('web/static/js/features/anima-app/index.js')))
print('--- feature domain lines ---')
rows = []
for d in features:
    n = sum(lines(p) for p in d.rglob('*.js'))
    rows.append((n, d.name))
for n, name in sorted(rows, reverse=True):
    print(f'{n:6d}  {name}')
print('--- chunk top 15 by lines ---')
for p in sorted(chunks, key=lines, reverse=True)[:15]:
    print(f'{lines(p):5d}  {p.name}')
PY
```

Expected: 打印 features/chunks/bridges 数量与行数；`app.js` 应很小（约数十行）。

- [ ] **Step 2: shim vs 重业务 chunk**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import re
chunks = sorted(Path('web/static/js/features/anima-app/chunks').glob('*.js'))
shim, heavy = [], []
for p in chunks:
    text = p.read_text(encoding='utf-8', errors='ignore')
    n = len(text.splitlines())
    # 启发式：很短且主要是 export * from / re-export
    export_star = len(re.findall(r'export\s+\*\s+from', text))
    if n <= 40 and export_star >= 1:
        shim.append((p.name, n))
    elif n >= 200:
        heavy.append((p.name, n))
print('shim_like', len(shim))
for name, n in shim:
    print(f'  {n:3d}  {name}')
print('heavy_ge_200', len(heavy))
for name, n in sorted(heavy, key=lambda x: -x[1])[:20]:
    print(f'  {n:4d}  {name}')
PY
```

Expected: 至少能分出若干 shim（如 queue-view/event-listeners 一类）与一长串 heavy chunk。

- [ ] **Step 3: legacyRoot bridge 名单**

Run:

```bash
rg -n "legacyRoot\s*=\s*globalThis|const legacyRoot = globalThis" \
  web/static/js/features/anima-app/helpers \
  --glob '*-bridge.js'
echo '--- count ---'
rg -l "legacyRoot\s*=\s*globalThis|const legacyRoot = globalThis" \
  web/static/js/features/anima-app/helpers \
  --glob '*-bridge.js' | wc -l
echo '--- fail-fast style bridges (sample) ---'
rg -n "throw new Error|is not configured|bridge is not configured" \
  web/static/js/features/anima-app/helpers \
  --glob '*-bridge.js' | head -40
```

Expected: 得到 legacyRoot bridge 文件列表与数量；以及哪些已 fail-fast。

- [ ] **Step 4: feature → chunks 反向依赖边**

Run:

```bash
rg -n "from ['\"].*anima-app/chunks/" web/static/js/features --glob '*.js'
```

Expected: 多行 `features/<domain>/...js` import `anima-app/chunks/...`。整理为边表（from → chunk）。

- [ ] **Step 5: 读入口与装配（人工 5 分钟）**

Read:

- `web/static/app.js`（全文）
- `web/static/js/features/anima-app/index.js`（全文）

记录：

- `app.js` 是否只 bootstrap
- `index.js` 是否仍大批量 import chunks、bridge configure 顺序是否敏感

- [ ] **Step 6: 本 Task 验收**

- `StructureEvidence` 数字齐全
- reverse_import_edges 非空则列入后续 High/Med
- **未**改业务文件

---

### Task 3: 护栏测试（S2 test-auditor）

**Files:**
- Test/Run: `tests/test_training_frontend_modules.py`、`tests/test_training_frontend_dom.py`
- Read-only on source

**Interfaces:**
- Consumes: `AuditSnapshotMeta`
- Produces: `TestEvidence = { command: str, exit_code: int, passed: int|None, failed: int|None, summary: str, missing: bool }`

- [ ] **Step 1: 确认 pytest 与 venv**

Run:

```bash
test -x .venv/bin/python && .venv/bin/python -V
.venv/bin/python -c "import pytest; print(pytest.__version__)"
```

Expected: Python 3.13.x；pytest 可 import。若失败：`TestEvidence.missing=true`，B 域按证据缺失扣分，跳到 Step 3 记录原因。

- [ ] **Step 2: 跑固定门禁包**

Run:

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_modules.py::test_frontend_module_graph_follows_production_entrypoint \
  tests/test_training_frontend_modules.py::test_frontend_module_cache_tokens_match_entrypoint \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  tests/test_training_frontend_modules.py::test_anima_app_global_this_writes_do_not_grow \
  tests/test_training_frontend_modules.py::test_split_frontend_features_do_not_write_global_this \
  tests/test_training_frontend_dom.py \
  -q --tb=line
```

Expected examples:

- 全绿：`N passed in Xs`
- 有红：失败用例名 + 一行 traceback；把**失败名与断言摘要**记入 `summary`
- 超时：exit 124 → `missing` 或 timeout 记入摘要，不重跑超过 1 次

- [ ] **Step 3: 粗扫 frontend 测试面（不跑全量）**

Run:

```bash
ls tests/test_training_frontend_*.py
wc -l tests/test_training_frontend_*.py | sort -n
```

Expected: 列出 modules/dom/history/queue/config 等文件与行数，用于 B 域“覆盖广度”叙述。

- [ ] **Step 4: 本 Task 验收**

- `TestEvidence` 有 exit_code 与 summary
- 红灯不得写成全绿
- **未**为了让测试绿而改业务代码（本 plan 禁止修业务；若红灯，债清单记录即可）

---

### Task 4: CSS / DOM 扫描（S3 css-dom-auditor）

**Files:**
- Read: `web/static/style.css`、`web/static/css/*`、`web/static/index.html`

**Interfaces:**
- Consumes: 无
- Produces: `CssDomEvidence = { css_total_lines, css_top: list[{file,lines}], style_imports: list[str], responsive_position: str, html_lines, dom_id_count, docs_features_count }`

- [ ] **Step 1: CSS 体量**

Run:

```bash
wc -l web/static/css/*.css | sort -n
wc -l web/static/style.css
```

Expected: 多份 >1000 行 CSS；记下 Top5 文件名与行数。

- [ ] **Step 2: style.css import 顺序**

Run:

```bash
cat web/static/style.css
```

检查：

- 是否先 tokens/base，后 forge，最后 `90-responsive`
- `90-responsive` 是否在其它页面 CSS **之后**（作为兜底）

把 import 列表按顺序列出到 `style_imports`。

- [ ] **Step 3: DOM id 规模**

Run:

```bash
wc -l web/static/index.html
python3 - <<'PY'
from pathlib import Path
import re
text = Path('web/static/index.html').read_text(encoding='utf-8', errors='ignore')
ids = re.findall(r'\bid="([^"]+)"', text)
print('html_lines', text.count('\n')+1)
print('dom_id_count', len(ids))
print('unique_ids', len(set(ids)))
# 重复 id
from collections import Counter
c = Counter(ids)
dups = [k for k,v in c.items() if v>1]
print('duplicate_ids', len(dups))
if dups[:20]:
    print('dup_sample', dups[:20])
PY
```

Expected: `dom_id_count` 约数百；若有 duplicate_ids，记为 High/Med。

- [ ] **Step 4: docs/features 对齐粗查**

Run:

```bash
ls docs/features
wc -l docs/features/*.md | sort -n
```

对照 scorecard / 真实 UI 域：history、queue、preview、dataset、global-settings 等是否有文档。缺口记 Low/Med。

- [ ] **Step 5: 本 Task 验收**

- Top CSS + import 顺序 + dom_id_count 齐全
- **未**改 CSS/HTML

---

### Task 5: 配置体验表面抽查（S4 config-surface-auditor）

**Files:**
- Read: `web/static/js/config/catalog/**`
- Read（抽样）: `web/static/js/features/config-form/**`、相关 chunk 仅当定位 provenance/live compat 时

**Interfaces:**
- Consumes: 五轮 design 中 C 轨声称项（provenance、live compat、guide、命名）
- Produces: `ConfigSurfaceEvidence = { residuals: list[{id, title, evidence_path}], positives: list[str] }`

- [ ] **Step 1: catalog 规模与 defaults 线索**

Run:

```bash
wc -l web/static/js/config/catalog/*.js | sort -n
rg -n "FORM_UI_DEFAULTS|ui_default|ui_only|provenance|source:" \
  web/static/js/config web/static/js/features \
  --glob '*.js' | head -80
```

Expected: 能看到 defaults/help/guide 文件；以及 provenance 相关符号是否仍存在。

- [ ] **Step 2: live compat / dirty / path label 残留抽查**

Run:

```bash
rg -n "liveCompat|live.?compat|compatibility|formatPathLabel|fieldProvenance|dirty" \
  web/static/js/features --glob '*.js' | head -80
```

把“五轮已做、代码仍在”的记入 `positives`；“声称有但找不到 / 仍可能覆盖消息”的记入 `residuals`。

- [ ] **Step 3: gui-methods 与 guide 同步粗查（文件名级）**

Run:

```bash
ls configs/gui-methods/*.toml 2>/dev/null | xargs -I{} basename {} .toml | sort > /tmp/gui-methods.txt
rg -n "METHOD_|methodKey|choiceGuide|guides" web/static/js/config/catalog --glob '*.js' | head -40
wc -l /tmp/gui-methods.txt
```

若无法自动 diff guide 列表，至少记录：gui-methods 数量 + catalog guide 入口路径，并在债中写「需人工点开 guide 是否与目录一致」（Med，除非已有测试证明同步）。

- [ ] **Step 4: 本 Task 验收**

- `positives` 与 `residuals` 至少各尝试列出；允许 positives 多、residuals 少
- **未**改 catalog 业务

---

### Task 6: 打分、Top10、写入 §8 并提交

**Files:**
- Modify: `docs/superpowers/specs/2026-07-11-web-frontend-boulder-audit-design.md`
  - 更新文首「状态」为：已执行审计快照
  - 填满 §8（含 8.1–8.4）
  - 勾选 §10 中审计 DoD 已完成项
- Optional Modify: `docs/features/frontend-health-scorecard.md` 增加一行“最近复评指针”指向本 spec §8（可选，默认跳过以免扩大写集）
- Modify: `docs/superpowers/README.md` 仅当 plans 索引缺少本 plan 时补一行

**Interfaces:**
- Consumes: `AuditSnapshotMeta` + `StructureEvidence` + `TestEvidence` + `CssDomEvidence` + `ConfigSurfaceEvidence`
- Produces: 填实的 §8；最终 commit

- [ ] **Step 1: 按评分卡计算四域分**

规则（必须写入 §8.1「证据摘要」列）：

```text
总分 = round(A*0.30 + B*0.25 + C*0.20 + D*0.25)
```

打分时强制对照 Task 2–5 证据：

| 域 | 上调条件（示例） | 下调条件（示例） |
|---|---|---|
| A | 入口仍薄；shim 占比上升；部分 bridge fail-fast | legacyRoot 仍多；反向 import 多；heavy chunk 仍主业务 |
| B | 门禁全绿；测试文件面仍在 | 红灯/超时/行为测试仍薄 |
| C | import 顺序正确；docs 增多 | 单文件 CSS >2k；DOM id 很多；重复 id |
| D | provenance/live compat 仍在且可用 | defaults 混层残留；guide 漂移；状态消息互盖 |

对比列必须保留 R0=61、IR5~78 与本次实测。

- [ ] **Step 2: 写 Top10 债**

每条必须含 spec §6 字段：`id, severity, domain, title, evidence, impact, suggested_next, do_not`。

排序：

1. silent bridge / 正确性  
2. 迁移永久化（chunks 新业务 + 反向依赖）  
3. 体量/性能/契约  
4. 体验残留  

High 尽量列全；总数主表 10 条。

- [ ] **Step 3: 写先别动 / 可先动**

至少包含：

**先别动**

- 一次删除全部 chunks
- 无 DOM 契约测试下批量改 id
- 未标 ui_only 就改写入默认值语义
- 碰 history/queue/output/models

**可先动（需另开实现 plan）**

- 按域切断 feature→chunks 反向依赖
- 高频 bridge 去 legacyRoot / fail-fast
- heavy chunk 搬家到 feature 目录
- 巨型 CSS 按面板拆分
- 行为级 frontend 测试补强

- [ ] **Step 4: 写入 §8.4 命令摘要**

只贴摘要，例如：

```text
HEAD: <sha>
dirty_count: N
chunks: 45 / lines L
legacyRoot bridges: K
reverse import files: F
pytest: X passed / Y failed (or timeout)
css top: 21-history-panels.css = ...
dom_id_count: ...
```

- [ ] **Step 5: 更新状态与 DoD 勾选**

在 design 文首：

```markdown
状态：审计快照已填写（只读体检完成）
```

§10 审计 DoD 项全部勾 `[x]`（设计阶段用户审阅项保持已完成）。

- [ ] **Step 6: 自检（执行者）**

Run:

```bash
rg -n "TBD|尚未执行" docs/superpowers/specs/2026-07-11-web-frontend-boulder-audit-design.md
git diff -- docs/superpowers/specs/2026-07-11-web-frontend-boulder-audit-design.md | head -200
# 确认无 web/static 进入暂存
git status --short
```

Expected:

- §8 不再出现「尚未执行」
- 分数表实测列无空
- `git status` 中**不应**为了本任务出现新的 `web/static/**` 改动；若工作区原本就有前端 dirty，不得把它们 stage 进本 commit

- [ ] **Step 7: 确保 plans 索引挂上本 plan**

若 `docs/superpowers/README.md` 的 Plans 表无本文件，追加：

```markdown
| [plans/2026-07-11-web-frontend-boulder-audit.md](plans/2026-07-11-web-frontend-boulder-audit.md) | Web 前端石山只读体检执行计划 |
```

- [ ] **Step 8: Commit**

```bash
git add docs/superpowers/specs/2026-07-11-web-frontend-boulder-audit-design.md \
        docs/superpowers/plans/2026-07-11-web-frontend-boulder-audit.md \
        docs/superpowers/README.md
# 仅当 Task 6 改了 scorecard 才 add：
# git add docs/features/frontend-health-scorecard.md

git commit -m "$(cat <<'EOF'
docs: fill web frontend boulder audit snapshot

Record read-only health scores, top debts, and gate evidence for the
current worktree without changing WebUI business code.
EOF
)"
```

Expected: commit 只含文档；`git show --stat HEAD` 无 `web/static`。

- [ ] **Step 9: 本 Task / 全 plan 验收（DoD）**

对照 spec §10：

- [x] 四域分数 + 总分/等级 + 每域证据  
- [x] R0 / IR5 / 实测对比表  
- [x] Top10 债字段完整  
- [x] 先别动 / 可先动  
- [x] 未改业务代码与用户数据  
- [x] design §8 已落盘  
- [x] 最小下一步已写明（停诊断 or 另开拆山 writing-plans）

---

## Plan Self-Review

| 检查 | 结果 |
|---|---|
| Spec 覆盖 | §4 证据包→Task2–5；§5 评分→Task6；§6 Top债→Task6；§7 并行角色→Task2–5；§8 填写→Task6；§10 DoD→Task6 Step9 |
| Placeholder | 无“implement later”；扫描/pytest/写入步骤均有具体命令 |
| 写集边界 | 全局约束禁止 web/static 业务改动 |
| 与 TDD 字面差异 | 本 plan 交付是审计文档；“红绿”体现在护栏 pytest 与 §8 自检，不为制造业务测试而改产品代码 |

## Execution Handoff

Plan 完成后，执行者可选：

1. **Subagent-Driven（推荐）** — Task 2–5 可并行只读子代理，Task 1/6 串行  
2. **Inline Execution** — 本会话按 Task 顺序跑完  

拆山实现**不在**本 plan 范围；若用户要拆山，另开实现 plan。
