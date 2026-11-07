# WebUI 设计系统控制台升级设计

状态：已完成  
实施进度注记：P0 设计系统底座 + P1 训练监控/队列 + P2 历史 board/面板降噪已合入本分支；P3 配置/数据集/工具四页回灌已完成；cache token 为 frontend-chain-20260712-ds-final。  
日期：2026-07-12  
范围：Web 前端视觉再升级（设计系统优先，信息架构基本不动）  
前置：[仪表面板换皮设计](../../proposal/webui-instrument-panel-reskin/specs/2026-07-12-webui-instrument-panel-reskin-design.md) 已完成并归档\
相关代码：

- `web/static/css/00-tokens.css`
- `web/static/css/01-base.css`
- `web/static/css/02-buttons.css`
- `web/static/css/13-shared-fields.css`
- `web/static/css/20-training-core.css`
- `web/static/css/21-history-panels.css`
- `web/static/css/22-training-queue.css`
- `web/static/css/33-training-forge.css`
- `web/static/css/33-training-history-theme.css`
- `web/static/css/90-responsive.css`
- `web/static/style.css`
- `web/static/index.html`（仅允许新增 class / 最小包裹层）

## 1. 背景与目标

上一轮“精密仪器台换皮”已经完成：

- 七个主 Tab 视觉语言初步统一
- 字段/控件可读性 token 落地
- 壳层变薄，16:9 密度有改善

但仍有结构性问题：

- 页级 forge 仍大量硬编码，系统层不够强
- 组件语义分散在多个 CSS 文件，复用靠复制
- 训练监控/历史虽已降噪，但还不是真正的“控制台骨架”
- 质感、层级、密度仍可再升一级，但不能再靠局部补丁

### 1.1 已确认决策

| 项 | 决策 |
|---|---|
| 改动类型 | 视觉再升级（信息架构基本不动） |
| 边界 | 中等：可不改 DOM id / 不减配置项 / 不加功能；允许新增 class 与最小 HTML 包裹层 |
| 成功目标优先级 | 1 密度扫读 + 2 组件系统 + 3 空间骨架 + 4 质感表现 |
| 推进路径 | A：先完整设计系统，再铺页 |
| 首批消费页 | 先组件系统，再训练监控/历史 |
| 主题 | 浅色优先，深色独立精修 |

### 1.2 目标

1. 建立可复用的 WebUI 设计系统（token + primitives + patterns）
2. 在不改 IA 的前提下，把布局骨架、密度、组件一致性、质感再升级
3. 用训练监控与历史验证系统是否真能支撑复杂控制台
4. 为后续配置/数据集/工具四页迁移提供稳定契约，避免半新半旧长期化

### 1.3 非目标

- 不重做 7 个主 Tab 导航结构
- 不新增业务功能
- 不删减任何配置项
- 不改 DOM id 契约
- 不一开始就全站七页同时完美翻修
- 不为抽象而预造训练/历史用不到的巨型组件库

## 2. 设计原则

1. **系统先于页面**：页面只消费系统，不再各自发明主按钮/字段/侧栏语言
2. **字只升不降**：密度提升靠壳变薄、间距节奏、信息分层，不靠缩小主字段
3. **主次唯一**：每个区域最多一个 primary 行动；highlight 永远弱于 primary
4. **骨架稳定**：顶栏/工具条/主区/侧栏/粘性区有固定节奏
5. **浅色优先**：默认展示与验收以浅色 16:9 为主，深色单独精修
6. **可测试**：token、组件 API、关键布局契约必须有自动化门禁

## 3. 架构

### 3.1 分层

```text
tokens
  → primitives（button/field/segmented/badge/card/toolbar/sidebar/stat/sticky）
    → patterns（page-shell / workbench / monitor-board / history-board）
      → pages（训练/历史先接；其余页后迁）
```

### 3.2 文件策略

新增设计系统层（名称可在实施计划中微调，但职责固定）：

- `web/static/css/ds/00-tokens-extend.css`：扩展 token
- `web/static/css/ds/10-primitives.css`：基础组件
- `web/static/css/ds/20-patterns.css`：页面骨架模式
- 可选：`web/static/css/ds/30-states.css`：状态色与 focus/disabled 汇总

既有页级文件策略：

- `33-training-forge.css` / `20-training-core.css` / `22-training-queue.css`
  改为“调用系统 + 训练特化”
- `33-training-history-theme.css` / `21-history-panels.css`
  改为“调用系统 + 历史特化”
- 旧硬编码色、字号、阴影逐步删除或降为 fallback

`web/static/style.css` 必须按稳定顺序导入系统层，并同步刷新 cache token。

### 3.3 HTML / JS 边界

- 允许：新增 class（如 `ui-btn`、`ui-field`、`monitor-board`）
- 允许：为布局需要增加无语义破坏的包裹层
- 禁止：修改既有 DOM id
- 禁止：删除配置项控件
- 默认不改 JS 业务；若必须为 class 钩子改 JS，只能做最小接线，不改行为

## 4. Token 设计

在现有 instrument-panel token 上扩展，不推倒重来。

### 4.1 类型

| Token | 用途 |
|---|---|
| `--font-size-title` | 页/区标题 |
| `--font-size-section` | 分组标题 |
| `--font-size-field-label` | 字段标签（保持高可读） |
| `--font-size-field` | 输入值 |
| `--font-size-meta` | 次级说明/eyebrow/badge |
| `--font-size-mono` | 路径、ID、代码感文本 |

### 4.2 控件与间距

| Token | 用途 |
|---|---|
| `--control-height-sm/md/lg` | 小/标准/大控件；默认 md |
| `--space-1..5` | 唯一间距阶梯 |
| `--radius-sm/md` | 小组件/面板圆角 |
| `--panel-shadow` / `--panel-shadow-soft` | 系统阴影，禁止页级重阴影回流 |

### 4.3 表面与状态

表面：

- `--surface-page`
- `--surface-panel`
- `--surface-raised`
- `--surface-input`
- `--surface-sticky`

状态三件套（bg/border/text）：

- idle / running / warning / error / success

深色要求：

- 低发光，不霓虹
- muted 文本仍可读
- 边框不发糊、不靠重阴影制造层级

## 5. 组件 API（Primitives）

> 以下为 CSS class 契约。实施时可加 BEM 修饰符，但语义不得漂移。

### 5.1 `ui-btn`

- 变体：`ui-btn--secondary` / `--primary` / `--highlight` / `--danger`
- 尺寸：默认吃 `--control-height-md`
- 规则：
  - primary 是最强填充
  - highlight 只做次强调，视觉弱于 primary
  - danger 用于破坏性/紧急操作
  - 图标+文字按钮保持同一高度

### 5.2 `ui-field`

结构：

```text
ui-field
  ui-field__label
  ui-field__control
  ui-field__help
```

规则：

- label/control 强制消费 field token 与 control height
- help 默认收敛，不默认撑高一屏
- 只读/禁用有独立表面，不靠透明度糊掉

### 5.3 `ui-segmented`

- 用于训练视图切换、来源模式、同类二分/多分段
- 选中态：底刻度线（与主 Tab 语言一致）
- 不使用大块实心填充作为默认选中

### 5.4 `ui-card` / `ui-toolbar` / `ui-sidebar`

- `ui-card`：薄头、弱边、内容优先
- `ui-toolbar`：矮工具条，主按钮唯一
- `ui-sidebar`：服务主区，索引/队列/筛选密度高于装饰

### 5.5 `ui-stat` / `ui-sticky`

- `ui-stat`：数字大、标签弱、空态低对比但不消失
- `ui-sticky`：粘性导航/操作条，占高受控，不遮挡首屏关键内容

## 6. 布局模式（Patterns）

### 6.1 `page-shell`

- 统一内容安全区与顶栏关系
- 负责页面背景与基础节奏，不承载业务结构

### 6.2 `workbench`

- 主区 + 侧栏（左或右）
- 侧栏宽度使用系统 clamp，不由页级随意拍脑袋

### 6.3 `monitor-board`（训练监控）

```text
[ 矮工具条：标题/状态/主操作 ]
[ 分段：实时/队列等同语言切换 ]
[ 主指标带：Loss / LR / Step / ETA / VRAM ]
[ 主区图表/日志 ]   [ 密侧栏：队列与最近任务 ]
```

规则：

- 顶栏只留高频操作
- 紧急停止保留强状态，但不靠巨型装饰
- 指标带用 `ui-stat`
- 侧栏弱于主监控区

### 6.4 `history-board`（训练历史）

```text
[ 侧栏：筛选/统计/批量工具条 ]
[ 主区：集合/配置组列表 + 详情 ]
```

规则：

- 批量条是工具条，不是第二皮肤
- 卡片头更薄
- 拖拽手柄可点但不抢色
- 搜索/筛选输入统一 `ui-field`
- 详情弹层走系统 panel 表面

## 7. 分阶段实施

### P0 设计系统底座

交付：

- token 扩展
- primitives
- patterns（至少 page-shell / workbench / monitor-board / history-board 骨架）
- 组件契约测试

完成定义：

- 不依赖具体业务页，也能验证按钮层级、字段高度、分段选中、表面 token
- style 入口与 cache token 双端同步

### P1 训练监控接入

交付：

- 训练页改用 monitor-board + primitives
- 队列侧栏密扫读
- 去掉与系统冲突的 forge 硬编码主路径

完成定义：

- 16:9 浅色下主指标与图表首屏更清楚
- 相关 frontend live/queue/dom 测试不因 CSS 契约回归

### P2 历史接入

交付：

- 历史页改用 history-board + primitives
- 批量条/筛选/卡片头系统化

完成定义：

- 与训练页同一组件语言
- 历史相关 frontend 测试不因 CSS 契约回归

### P3 回灌其余页

交付：

- 配置/数据集/ΔW/设置/环境/生图迁移到系统组件
- 清理半新半旧 forge 残留

完成定义：

- 七页无两套主按钮/字段/侧栏语言
- High=0

## 8. 审核与门禁

每阶段强制：

1. 有限写集
2. 补充审核
3. 交叉审核（visual / readability / theme / contract）
4. 自动化门禁
5. 16:9 浅/深走查记录

### 8.1 补充审核清单

- [ ] 无新功能
- [ ] 无配置项减少/隐藏
- [ ] 无 DOM id 改动
- [ ] 主字段可读性未回退
- [ ] 密度提升不靠缩小主字号
- [ ] 浅色主展示可用
- [ ] 深色未明显发糊/过霓虹
- [ ] primary/highlight 层级正确

### 8.2 自动化门禁

最低集合：

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_webui_visual_tokens.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  tests/test_training_frontend_dom.py \
  -q
```

阶段附加：

- P1：training live/queue 相关
- P2：history 相关
- P3：misc/config 布局契约 + 全 Tab 相关前端契约

### 8.3 熔断

触发任一条件立即停扩写：

1. 连续两阶段 High 未清零
2. 主字段字号被回退
3. DOM id 被改或配置项丢失
4. 系统组件被页级 forge 大面积覆盖且无测试保护
5. 16:9 密度明显变差且无补偿

## 9. 成功标准

1. 存在稳定设计系统层，页面以消费系统为主
2. 训练监控与历史成为同一控制台语言的首批样板
3. 16:9 默认缩放下，扫读效率与有效信息密度优于当前 reskin 结果
4. 组件层级清楚：primary > highlight > secondary，danger 独立
5. 浅色优先、深色精致
6. 无功能回归、无配置项丢失、无 DOM 契约破坏

## 10. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 系统抽象空转 | P0 只做训练/历史真实需要的组件 |
| 页级 CSS 盖掉系统 | 契约测试锁定主路径消费 token/class |
| HTML class 迁移牵连选择器 | 不改 id；新增 class；改选择器必补测 |
| 工期被七页同时摊薄 | 严格 A 路径：先系统，再训练/历史，最后回灌 |
| 与旧 reskin 冲突 | 以本 spec 为下一阶段权威；旧 reskin 作为已完成基线 |

## 11. 文档与产出

- 设计：`docs/superpowers/specs/2026-07-12-webui-design-system-console-upgrade-design.md`
- 实施计划：后续 `docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade.md`
- 迭代日志：后续 `docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade-iteration-log.md`

## 12. 开放决策（实施前已关闭）

| 问题 | 结论 |
|---|---|
| 是否先铺配置页？ | 否，先系统，再训练/历史 |
| 是否就地强化旧 forge？ | 否，设计系统优先 |
| 是否允许改 DOM id？ | 否 |
| 是否允许新增 class / 最小包裹层？ | 是 |
| 是否七页并行大翻修？ | 否，P3 再回灌 |
