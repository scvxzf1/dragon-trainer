# Dragon 数据集编辑器重建规格

状态：已确认范围 / 实现前规格<br>
优先级：P0<br>
关联历史设计：[`2026-07-11-dataset-page-stage-schedule-ia-design.md`](2026-07-11-dataset-page-stage-schedule-ia-design.md)

## 1. 用户目标

用户可以在一个稳定、密集但可扫描的工作台中完成：

- 管理数据集预设和自定义分组；
- 在独立编辑栏中维护多个 subset；
- 拖动 subset 排序，拖动预设进行组内排序或跨组移动；
- 预览每个 subset 的真实图片和 caption；
- 保存蓝图、应用到当前训练配置，并配置分阶段调度。

## 2. 页面结构

```text
数据集蓝图
├─ 左栏：预设库
│  ├─ 搜索、刷新、导入、新建、创建分组
│  ├─ 可折叠分组
│  └─ 可排序/跨组拖动的预设项
└─ 主栏：当前预设编辑器
   ├─ 文件身份、dirty/只读状态、保存/另存/应用
   ├─ subset 工具条：添加、阶段调度
   ├─ 可排序 subset 列表
   └─ 图片预览与高级设置 dialog
```

移动端不强行保持双栏：预设库使用抽屉或切换视图，主编辑区保持单列，所有操作可在
不依赖 hover 的情况下完成。

## 3. 领域模型

```ts
type DatasetPreset = {
  file: string;
  readonly: boolean;
  defaults: Record<string, unknown>;
  datasets: DatasetSubset[];
  stageScheduleEnabled: boolean;
  stageSchedule: StageSpec[];
};

type DatasetLibraryGroup = {
  id: string;
  label: string;
  renamable: boolean;
  deletable: boolean;
  movable: boolean;
  orderedFiles: string[];
};
```

后端实际响应类型由 API 契约测试锁定；上述类型不得通过 `any` 掩盖不一致。

## 4. 状态边界

| 状态 | 所有者 |
| --- | --- |
| 预设库和分组 | Query cache |
| 当前预设服务器快照 | Query cache |
| 当前编辑草稿、dirty、字段错误 | Form state |
| 当前 subset、展开组、对话框 | feature local state |
| 拖动中的 active/over 项 | DnD context |
| 图片预览列表 | 独立 preview query |
| 当前训练配置身份 | app context store |

预设保存和“应用到训练配置”是两个独立 mutation，不允许保存成功后隐式改变训练配置引用。

## 5. 分组与拖动

### 分组管理

- 支持新建、重命名和删除自定义组。
- 系统、只读或锁定组严格遵守后端 capability flags。
- 删除分组只删除分组元数据，不删除其中 TOML 文件。
- 搜索期间禁止产生含糊的排序写入；拖动入口需明确禁用。

### 预设拖动

- 支持组内重排和跨组移动。
- drop 前验证目标组 `movable` 和源文件权限。
- 使用 optimistic UI 时必须提供失败回滚和明确错误提示。
- Pointer、touch 和 keyboard 路径具有相同业务结果。
- 不能依赖 DOM selector 推导业务顺序；顺序来自显式 item id 数组。

### subset 拖动

- 支持 before/after 排序和长列表自动滚动。
- 排序后保持当前选中 subset 的稳定 identity；不能只依赖易变数组下标。
- 写盘前仍转换为后端现有数组顺序。
- 删除至少保留一个普通训练 subset，不删除任何磁盘图片或缓存。

## 6. 图片预览

- 只有已保存且非 dirty 的预设可以请求预览，避免磁盘内容与草稿错位。
- 请求包含 `file`、`dataset_index`、`source=source` 和有界 `limit`。
- 展示图片、尺寸、caption、caption 来源、重复、分辨率、分桶和 validation 摘要。
- 图片 lazy-load；损坏图片有明确占位，不阻断其他图片。
- caption 支持复制；图片支持大图查看、键盘关闭和焦点恢复。
- 刷新必须取消或忽略旧请求，禁止较慢旧响应覆盖新状态。
- 空目录、路径失效、只读、部分 caption 失败和 HTTP 错误均有独立状态。

## 7. 分阶段调度

沿用既有 `stage_schedule_enabled` 和 `stage_schedule[]` 后端语义，但在新前端中独立实现：

- 不 import `stage-resolution-ui.js`、anima-app state 或任何 bridge。
- 纯领域函数负责 normalize、validate、添加、删除、移动和贴齐区间。
- dialog 只消费 typed model，不直接查询 Classic/Dragon DOM。
- 保持 1/N 段、`0 -> 100%` 全覆盖、合法 `subset_index` 等现有不变量。

## 8. 保存和危险操作

- 切换预设、新建、刷新、离开路由或关闭窗口时统一执行 dirty guard。
- 新建、复制、另存、重命名、导入同名和删除必须有清晰冲突行为。
- 删除预设只删除 TOML，不删除图片、caption 或缓存目录。
- 只读预设允许另存，不允许覆盖源文件。
- mutation 成功后以服务器响应更新 query cache，不靠页面重载恢复一致性。

## 9. API 契约清单

实施前逐项从实时路由和测试固定 request/response：

- 数据集预设列表、读取、保存、另存、删除、导入和图片列表；
- file group 新建、改名、删除、组排序、文件排序和跨组移动；
- 应用数据集预设到训练配置；
- 当前训练配置上下文和 merged config。

API 层必须区分 HTTP 错误、`ok=false`、冲突、只读、路径越界和网络取消。

## 10. 测试门禁

### 领域单测

- subset reorder、删除下限、稳定 identity；
- group reorder、跨组移动、权限拒绝和失败回滚；
- stage schedule N=1/2/5、空洞、重叠、越界和 subset 删除后的引用错误。

### 组件测试

- 新建/改名/删除组；
- pointer 与 keyboard 排序；
- dirty guard；
- 预览 loading/empty/error/stale response；
- 只读和系统组 capability。

### Playwright

1. 新建预设并添加三个 subset，拖动排序后保存并刷新验证顺序。
2. 新建两个分组，把预设跨组移动并刷新验证持久化。
3. 打开图片预览，检查 caption、复制、大图和刷新。
4. 修改草稿后验证切换预设、离开路由和刷新保护。
5. 配置两段/五段 stage schedule，保存训练配置并重新加载。
6. 在桌面和移动端完成关键工作流，无水平溢出和控件遮挡。

## 11. 完成条件

- 新数据集功能树不存在对 `web/static/js/features/anima-app`、Classic DOM 或 bridge 的 import。
- 本规格所有行为有直接测试证据。
- 功能矩阵中数据集 P0 项全部更新为 `已验证`。
- 当前 Dragon 的 `legacyStageDatasetState`、`legacyStageConfigState` 依赖可被删除。
- 真实用户数据目录只读验证通过，测试不修改或删除用户预设、图片和缓存。
