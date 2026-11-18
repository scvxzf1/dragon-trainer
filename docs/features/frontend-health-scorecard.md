# WebUI 前端健康度评分卡

状态：维护参考（集中审阅评分模板）
适用范围：`web/frontend-next/` React/Vite 工作台的可选集中审阅；不构成日常开发固定门禁（日常检查遵循 [`AGENTS.md#验证策略`](../../AGENTS.md#验证策略)）。

相关代码：
- `web/frontend-next/src/main.tsx`、`src/app/router.tsx`
- `web/frontend-next/src/features/**`、`src/styles/**`
- `web/frontend-next/src/**/*.test.*`、`web/frontend-next/e2e/**`

历史轮次审计记录：
- 2026-09-13 迁移期审计：[webui_frontend_audit_20260913.md](../findings/webui_frontend_audit_20260913.md)
- 2026-07-11 早期五轮迭代协议（已归档）：[`2026-07-11-five-round-auto-iteration-protocol.md`](../../_archive/docs/superpowers/specs/2026-07-11-five-round-auto-iteration-protocol.md)

---

## 1. 用途与边界

在明确安排的分支集中审核或重构里程碑中，用统一的四域百分制评估前端工程健康度、识别最高优先级技术债与配置体验缺口。本评分卡不替代真实用户验收，也不对后端数值训练逻辑打分。

---

## 2. 快速审核流程（15–30 分钟）

```bash
# 0) 核对工作树与提交
git status --short --branch
git rev-parse --short HEAD

# 1) 结构扫描：入口、路由、feature 划分与样式层
sed -n '1,120p' web/frontend-next/src/main.tsx
sed -n '1,180p' web/frontend-next/src/app/router.tsx
find web/frontend-next/src/features -mindepth 1 -maxdepth 1 -type d | sort
find web/frontend-next/src/styles -maxdepth 1 -type f | sort

# 2) 按审阅范围选用类型检查与单元测试
.venv/bin/python tasks.py web-next-check

# 3) 关键交互或发布验收时运行隔离 E2E
.venv/bin/python tasks.py web-next-e2e
```

---

## 3. 四域加权评分结构（100 分）

```text
总分 = round(A * 0.30 + B * 0.25 + C * 0.20 + D * 0.25)
```

| 等级 | 分数区间 | 含义 |
| --- | ---: | --- |
| A | 90–100 | 模块边界清晰，测试与配置契约稳固 |
| B | 80–89 | 主路径稳健，局部技术债可控 |
| C | 70–79 | 功能可用，存在需优先偿还的结构或配置体验债 |
| D | 60–69 | 模块耦合或配置可信度不足，需专项治理 |
| F | <60 | 架构或数据安全契约风险高，优先止血 |

| 域（权重） | 子项与满分 | 核心评分线索 |
| --- | --- | --- |
| **A. 结构与模块边界（30%）** | 入口装配 (20) / Feature 职责 (25) / 状态管理 (25) / 依赖方向 (15) / 单文件体量 (15) | `main.tsx` 与 `app/` 是否薄；页面、组件、hooks、store、API 是否按职责拆分；是否零隐式全局状态、零循环依赖；组件是否控制在合理规模 |
| **B. 测试与契约门禁（25%）** | 关键风险覆盖 (25) / 契约护栏 (25) / 行为仿真隔离 (25) / 反馈速度 (15) / 按风险选测 (10) | 草稿保护、模型族过滤、队列/历史 revision 与删除边界是否有定向覆盖；Vitest 与 Playwright 是否完全 mock/隔离且不写用户后端 |
| **C. 样式、交互与可访问性（20%）** | 设计令牌与复用 (25) / 层叠顺序 (20) / 交互与草稿保护 (30) / 响应式与键盘焦点 (15) / 文档同步 (10) | `controlDefaults.css` 与 Tailwind/组件层叠是否清晰；未保存草稿、断线未知结果提示、窄屏布局是否稳健；`docs/features/` 是否反映真实 UI |
| **D. 配置与能力表面（25%）** | 帮助准确性 (25) / 继承与默认值可信度 (30) / 实时披露与预检对齐 (20) / 概念分层 (15) / 后端同源性 (10) | 283 项 V2 帮助与摘要是否同步；覆写状态与继承基准是否诚实；`CONFIG_FIELD_CATALOG`、`model-family.js` 与后端 `MODEL_FAMILY_REGISTRY` / `compat_matrix` 是否对齐 |

---

## 4. 评审记录模板

````markdown
### 前端健康评分 — <branch> @ <shortsha> — <YYYY-MM-DD>

| 域 | 原始分 | 权重 | 加权 |
| --- | ---: | ---: | ---: |
| A 结构与模块边界 |  | 0.30 |  |
| B 测试与契约门禁 |  | 0.25 |  |
| C 样式与交互壳 |  | 0.20 |  |
| D 配置与能力表面 |  | 0.25 |  |
| **总分 / 等级** |  |  | ** / ** |

#### 关键证据与已执行命令
-

#### High / Medium 待跟进项
1.
````

> 新产出的具体轮次审计报告请写入 `docs/findings/` 并标注日期与提交快照，不要把单次临时日志堆回本模板页。
