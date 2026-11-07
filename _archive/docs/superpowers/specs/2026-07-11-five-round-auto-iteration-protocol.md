# 五轮自动迭代协议（前端强化版）

状态：启用  
适用分支：`docs/backend-config-optimization` 及后续前端优化实现分支  
日期：2026-07-11

关联文档：

- 评分卡：`docs/features/frontend-health-scorecard.md`
- 前端设计：`docs/superpowers/specs/2026-07-11-frontend-config-optimization-design.md`
- 前端计划：`docs/superpowers/plans/2026-07-11-frontend-config-optimization.md`
- 迭代日志：`docs/superpowers/plans/2026-07-11-fullstack-auto-iteration-log.md`
- 后端对照：`docs/superpowers/specs/2026-07-11-backend-config-optimization-design.md`

---

## 1. 目标

一句话：用固定节奏“评分 → 并行审核 → 选优化项 → 严格 debug 测试 → 复盘”，把前端从 D 档推到 C/B，并且每轮都能留下可复用证据。

本协议强制要求：

- 每轮有评分卡
- 每轮有 High 清单收敛情况
- 每轮有真实测试命令结果（不是空口“看起来没问题”）
- 每轮最多推进有限写集，避免一次改爆

---

## 2. 每轮输入 / 输出

| 方向 | 内容 |
|---|---|
| 输入 | 当前分支源码、上轮评分卡、未关闭 High、计划任务进度 |
| 输出 | 前端评分卡、本轮完成项、测试门禁结果、下轮焦点、是否熔断 |
| 可选输出 | 后端评分卡（全栈合流轮） |

---

## 3. 规范评分结构

前端固定使用：

`docs/features/frontend-health-scorecard.md`

| 域 | 权重 |
|---|---:|
| A 结构与迁移 | 30% |
| B 测试与门禁 | 25% |
| C CSS/DOM | 20% |
| D 配置体验 | 25% |

等级：A90+ / B80-89 / C70-79 / D60-69 / F<60

禁止：

- 无证据上调分数
- 用文案润色代替测试
- 为提分去碰用户 history/queue/output

---

## 4. 子代理分工

| 角色 | 任务 | 权限 | 并行 |
|---|---|---|---|
| structure-auditor | 结构/chunks/bridge | 只读 | 可并行 |
| test-auditor | 测试与门禁 | 只读 | 可并行 |
| css-ux-auditor | CSS/DOM/交互 | 只读 | 可并行 |
| config-surface-auditor | 配置体验 | 只读 | 可并行 |
| planner | 合流文档/计划 | 写 docs | 串行收口 |
| implementer | 按计划改代码 | 限定 write_scope | 按任务拆分 |

规则：

- auditor 默认可并行
- implementer 写集不得重叠
- `max_depth=1`，子代理不得再 spawn 孙代理
- 父代理负责汇总、裁决冲突、更新日志

---

## 5. 严格 Debug 测试流程

一句话：每个任务都必须走“红灯 → 最小实现 → 绿灯 → 域回归 → 记录”。

### 5.1 单任务流程

```mermaid
flowchart LR
  T1[写/改失败测试] --> T2[确认红灯]
  T2 --> T3[最小实现]
  T3 --> T4[定向绿灯]
  T4 --> T5[域回归 <=60s]
  T5 --> T6[记命令与结果]
  T6 --> T7[提交或记入日志]
```

### 5.2 通过标准

- 退出码 0
- 无 failed/error
- `skipped` 仅允许缺 `node`，且必须在日志写明；有 node 时 skipped=0
- 失败信息必须能定位：文件 / DOM id / token / globalThis / 配置键
- 不允许靠放宽断言“刷绿”

### 5.3 固定门禁包

#### 快速红灯（每轮开工前）

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_modules.py::test_frontend_module_graph_follows_production_entrypoint \
  tests/test_training_frontend_modules.py::test_frontend_module_cache_tokens_match_entrypoint \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  tests/test_training_frontend_modules.py::test_anima_app_global_this_writes_do_not_grow \
  tests/test_training_frontend_modules.py::test_split_frontend_features_do_not_write_global_this \
  tests/test_training_frontend_dom.py \
  -q
```

#### R1 架构护栏

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_modules.py \
  tests/test_training_frontend_dom.py \
  tests/test_training_frontend_state.py \
  -q
```

#### R2 实时训练 + 队列

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_live.py \
  tests/test_training_frontend_queue.py \
  tests/test_training_queue.py \
  tests/test_training_queue_resume.py \
  -q
```

#### R3 历史 + 预览

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_history.py \
  tests/test_preview_service.py \
  tests/test_training_history_list.py \
  tests/test_training_history_delete.py \
  tests/test_training_history_artifacts.py \
  tests/test_training_history_timeline.py \
  -q
```

#### R4 配置表单 + 配置服务

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_config_ui.py \
  tests/test_training_frontend_misc.py \
  tests/test_web_config_service.py \
  tests/test_web_preflight_compat_matrix.py \
  tests/test_config_provenance.py \
  -q
```

#### R5 全前端收口

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_*.py \
  tests/test_weight_analysis_service.py \
  tests/test_image_test_service.py \
  -q
```

### 5.4 失败诊断顺序

1. cache token 是否漏改？
2. globalThis / baseline 是否漂移？
3. DOM id 契约是否被改？
4. bridge 是否未 configure？
5. 是行为回归，还是字符串契约噪音？
6. 是否误触用户数据路径？

---

## 6. 五轮目标递进（实现向）

| 轮次 | 目标 | 主要任务 ID | 目标健康分 |
|---|---|---|---:|
| R1 | 基线锁定 + 护栏止血 | S0, T0, U0, C2 | 61→66 |
| R2 | 配置可信度 P0 | C2, C3, C4, T1 | 66→70 |
| R3 | 工程债收敛 + 来源可见 | E1, E2, C1, C6 | 70→74 |
| R4 | 兼容前移 + 工作台一致 | C5, U1, E3 | 74→76 |
| R5 | 收口冻结 + 文档对齐 | E4, U2, D1, Freeze | 76→78+ |

说明：

- 旧日志里“R1-R5 文档空转”保留为历史
- 本协议定义的是**下一阶段可执行五轮**
- 若某轮只改文档，不得宣称健康分上升

---

## 7. 熔断与恢复

触发任一条件立即熔断：

- 同一 High 连续 2 轮无收敛
- 门禁无法在 60s 给出有效信号
- 写集冲突导致反复互相覆盖
- 测试失败率 >40% 且无法定位

恢复条件：

- 先缩 scope
- 先补 smoke / baseline
- 写冲突改串行
- 重新评分后再开并行

---

## 8. 每轮日志模板

```markdown
### Round N — YYYY-MM-DD

| 项 | 值 |
|---|---|
| 分支 | |
| HEAD | |
| 前端总分/等级 | |
| A/B/C/D | |
| 本轮焦点 | |
| 完成任务 | |
| 关闭 High | |
| 新增 High | |
| 测试门禁 | 命令 + 结果摘要 |
| 熔断? | 否/是（原因） |
| 下轮焦点 | |
```

---

## 9. 完成定义（五轮后）

全部满足才算本轮协议阶段完成：

- 前端健康分 ≥ 78，或明确记录未达标原因与残留 High
- 评分卡、设计、计划、日志互相链接且一致
- F/C/E/U/T 任务队列已冻结，每项有测试命令
- 不新增 globalThis 业务总线
- 不删除用户 history/queue/runtime/output
