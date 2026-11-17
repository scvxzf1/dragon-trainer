# Dragon Next 训练配置整理与分阶段验收计划

状态：活跃提案 / T0 基线切片、T1 依赖切片、T2 布局与键盘基线、T3 Qwen 首批切片已落地，持续验收中
审查日期：2026-09-27
适用范围：`/next/training`，React Next 训练配置工作台
基线：当前本地 `dev` 工作树及 `http://127.0.0.1:20203/next/training`；包含已有未提交开发，不等同于线上分支快照。

本轮落地（隔离开发服务 `http://127.0.0.1:20522/next/training`）：

- T1：统一 `visible/enabled/reason/code`，合并共享披露规则；epochs→steps 冲突禁用，搜索不绕过适用性，不适用分支和未知键仅在审计中可见且不可编辑；预检定位禁用控件时聚焦字段行。
- T2：方法簇顺序固定为“方法契约 → Adapter → 热启动 → 容量”，资源区拆出“自适应精度”和“OOM 重试”，配置库侧宽度上限 42%，保留窄屏纵向布局。
- T3 首批：Qwen `t2i/edit` 成为正式输入字段；旧配置缺失时默认 `t2i`，模型族不匹配时默认隐藏并禁用，后端 capability catalog 暴露 `supported_tasks`。
- T0 增量：从 Anima、Krea-2、Z-Image、Qwen 四个真实方法 TOML 构建逐键 manifest，比较公共字段的类型、页面分组及 catalog owner；修正 `use_moe_style`、检查点间隔、Flow 偏移、CPU 检查点卸载和运行时布尔字段的类型漂移；`dim_from_weights=false` 保持布尔语义，未知键进入审计 owner。
- T1 增量：覆盖 AUTO／手动 swap 双向切换、compile 子项依赖、旧值保留和仅父项变更时的 patch；禁用原因与 UI 控件状态均有测试。
- T2 增量：浏览器验证配置库宽度 42% 上限、260px 最小宽度、侧／顶布局持久化；编辑区在 1440×900 达到 420px、1280×720 达到 300px；768×1024 与 390×844 无横溢出；1280×720 明暗主题、200% 缩放与调整器键盘操作通过。
- 证据：`web-next-check src/features/training-config`、`tests/test_dragon_config_boolean_controls_frontend.py`、`training-field-availability.spec.ts`、`training-preflight-locate.spec.ts`、`training-library-resize.spec.ts`、`scale-keyboard.spec.ts`、`training-draft-flow.spec.ts` 和 `model-task-capabilities.spec.ts` 定向检查通过。mock E2E 未连接真实训练写接口；未修改配置、运行真实预检、启动训练或入队。

仍未完成：T0 完整字段 schema（选项、默认／未设置、依赖、序列化位置和来源）及持久化差异产物；T1 自适应精度 × family × OOM retry 完整矩阵；T2 训练字段完整键盘遍历；T3 其余确认缺项；T4 字段级 provenance、恢复继承 API 和隐藏修改摘要；T5 综合签收。

## 目标与推荐顺序

保留「输入准备 → 方法配置 → 训练计划 → 设备与性能」四阶段，整理每阶段的选择方式、字段顺序、联动规则与缺失入口。
用户应当能回答：当前训练什么、应该改哪些参数、哪些参数实际生效、修改会写入哪里、是否具备启动条件。

实施顺序为 **字段对账 → 选择与禁用规则 → 分组和布局 → 补充入口 → 来源与变更审查 → 综合验收**。
优先交付 T0–T2：先建立可信的字段规则和可用编辑空间，再扩充表单。

沿用此前[渐进披露提案（已归档）](../../_archive/docs/proposal/dragon_config_progressive_disclosure.md)的唯一归属、隐藏值保留和能力约束原则。
该提案主要描述旧 Dragon，不能把其「已实现」直接当作 Next 的验收证据，也不沿用其中历史字段数量和已经变化的 family 限制。
Next 现有四阶段采用 tab 切换；本计划保留这个交互，不同时改为长页滚动导航。

## 实施前基线与当前验证

审查包括真实页面只读操作、源码交叉检查、前端定向测试、代表 TOML 字段 manifest 和 mock 浏览器回归。未修改训练配置、运行真实预检、启动训练或加入队列。下表记录实施前问题；当前处理状态以上方落地记录和实时源码、测试为准，不能将表中观察当作尚存缺陷。
本轮未把完整浏览器套件或生产构建当作通过依据；缓存 token 相关旧 Dragon 套件仍有与本切片无关的基线失败，详见最终验收记录。

页面加载稳定后，当前 Krea-2 配置显示 `163 / 249`。这是该配置与 preset 下的一次观测，不能作为全局字段常量。
自动化视口覆盖了 `1440×900`（编辑区 ≥420px）、`1280×720`（编辑区 ≥300px、明暗主题、200% 缩放）、`768×1024`、`390×844` 无整页横溢出，以及 `1600×900`／`1200×900` 的配置库拖拽和键盘调整。尚未覆盖训练字段完整键盘遍历和全部模型族组合。

| 优先级 | 实施前观察 | 对使用的影响 | 处理方向 |
| --- | --- | --- | --- |
| P1 | `max_train_epochs` 已设置时，搜索 `max_train_steps` 可以找到并编辑该项，旁边同时提示该值将被覆盖 | 用户改了数值却不生效 | 将可见、可编辑、有效值三种语义分开；被覆盖项禁用并能定位控制项 |
| P1 | 搜索会越过「当前适用／已修改」过滤；编辑器的 `disabled` 只接收页面 busy 等状态，没有合并字段 availability | 搜索可重新打开不可用字段；「已修改」搜索也可能混入未改项 | 搜索只负责匹配，适用性和编辑权限独立判定；审计入口显式展示不适用结果 |
| P1 | `fieldsForConfig()` 将基础字段、旧 catalog、merged config 的键合并；动态字段类型先看当前值，再看选项 | 同一个字段的类型、空值处理和选择方式依赖所读配置，目录缺口不易被发现 | 为正式字段显式声明类型、枚举、单位、范围、来源、依赖和 owner |
| P1 | 方法页先显示可自由填写的 `network_module`；Adapter 选择位于容量之后且默认收起 | 选择方法时需要理解 Python 模块名，决策顺序不直观 | 常规方法／Adapter 选择在前，模块路径和原始参数作为兼容入口 |
| P1 | 1280×720、配置库展开时，参数滚动区域约 180px 高，只露出首个字段附近 | 编辑时持续滚动，输入上下文容易丢失 | 合并顶部工具行、收紧上下文，保障编辑区高度 |
| P1 | Qwen 有明确的后端精度、采样、preview 等限制，但前端选项过滤只覆盖部分字段 | 可选择的值不一定可启动 | 按后端能力与兼容规则补齐 UI 限制，保留预检最终校验 |
| P2 | 计算分组中，混合精度之后紧接自适应精度和 OOM 重试子项，基础计算路径与 attention 排在后面 | 常规配置被实验细节打断 | 将常规计算项前置，实验精度和 OOM 恢复各自成簇 |
| P2 | 字段来源只有「当前文件／继承或预设」；没有字段级的完整来源和覆盖管理 | 难判断本文件值、合并值、草稿值及恢复继承的区别 | 补充来源详情、撤销修改；恢复继承需明确后端删除覆盖语义 |

另一个可改善点：跨阶段搜索已有 tab 命中数量。例如在设备阶段搜索步数，训练计划显示 `(1)`，当前 tab 显示空态。
因此不能称为「没有跨阶段搜索」；可补「前往训练计划的 1 项结果」快捷入口，避免只给空态。

### 已有能力，不能再计为缺失

- 四阶段分组、中文名／键名搜索、当前适用／已修改／审计视图、分组折叠。
- 配置库、模型组合选择、设备选择、数据集选择与编辑、训练量估算、样张提示词和 TOML 入口。
- `mixed_precision`、`adaptive_precision`、`adaptive_fp32_modules`、loss scale 和 OOM retry 控件已存在；需要整理与能力联动。
- 变更预览、只提交修改项、未知 `network_args` 保留、保存冲突反馈、保存失败阻断后续执行已有实现和测试。
- 预检结果已有跨阶段定位与自动切换审计／搜索的 E2E。后续扩展到禁用控件和数据集字段，不重新建设一套定位系统。
- Qwen Edit 数据集已有编辑前／后配对、引用目录处理和应用确认；确认应用时能写入 `qwen_image_2_1_task`。训练页缺的是正式任务入口和一致的状态呈现，不是整个 Edit 数据流程。

## 目标排列与选择方式

### 页面层次

1. 配置文件与硬件 preset：保留当前配置、修改状态和来源入口。
2. 页面操作：保存、预检、启动／入队是主要操作；另存、TOML、样张提示词、估算与结果查看保持可发现的次级入口。预检动作和查看已有结果明确命名。
3. 训练上下文：模型组合、当前 family／方法摘要、设备。一个配置键只有一个主编辑位置，快捷选择复用同一草稿。
4. 参数搜索、视图和四阶段导航。
5. 参数编辑区：固定簇顺序；常用簇展开，专属簇按能力与方法显示，实验簇明确标记。

配置库在宽屏继续独立滚动；中窄屏改为可呼出的侧层或抽屉，避免展开时长期挤占参数区。
保留配置库展开偏好的合理迁移，不通过清空所有 UI 偏好来实现新默认布局。

### 四阶段的建议顺序

| 阶段 | 分组顺序 | 关键调整与字段示例 |
| --- | --- | --- |
| 输入准备 | 模型与任务 → 数据集 → 标注与遮罩 → 筛选 → 缓存策略 → 预处理与数据加载 | `model_family` 在路径前；Qwen 的 `qwen_image_2_1_task` 紧随 family；模型组合快捷入口靠近路径组；`use_vae_cache/use_text_cache` 在 reuse／校验子项前 |
| 方法配置 | 方法与 Adapter → 热启动 → 容量与训练目标 → 当前 Adapter 专属项 → 可组合扩展 → 原始参数 | 先选方法和 `lora_adapter_kind`，再决定 `network_weights/dim_from_weights/network_dim/network_alpha`；LoKr／VeRA 等专属项紧随其选择器；`network_module/network_args` 保留受控兼容入口 |
| 训练计划 | 任务身份 → 训练量与批次 → 优化器与学习率 → 时间步与损失 → 正则化 → 训练预览 → 保存与续训点 → 日志 | 明确 epochs／steps 优先级；optimizer 在专属参数前，scheduler 在 warmup 等参数前；preview 由当前 family 能力约束；区分权重保存和训练 state 保存 |
| 设备与性能 | 常规精度与计算 → 显存策略 → 编译 → 实验精度 → OOM 恢复 → 诊断 → 实验审计 | `mixed_precision/precision_preference/base_compute/attn_mode` 集中前置；checkpoint、AUTO／手动 swap 及各自子项按依赖排序；自适应精度和 OOM 重试不夹在基础精度与 attention 之间 |

预处理／DataLoader 继续归入当前 Next 的「输入准备」，避免照搬旧提案再次搬回资源阶段。
设备选择沿用常驻快捷区，性能页只呈现同一状态的摘要，不能再建一份设备选择状态。
流水线参数继续作为实验审计项；当前主训练 runtime 未接入，不能因重排就开放为可运行模式。

### 控件和状态规范

| 类型 | 目标控件与规则 | 必须保留的语义 |
| --- | --- | --- |
| 封闭枚举 | 下拉选择，中文解释与原始值对应；按 family／方法约束选项 | 老配置未知值显示原值及原因，不自动替换成第一项 |
| 可扩展值 | 常用选项加明确的自定义入口 | 不把开放 optimizer／模块参数误当封闭枚举 |
| 布尔 | 勾选或开关；来源／覆盖操作独立呈现 | 显式 `false`、继承的 `false` 和缺失不是同一持久化状态 |
| 数值 | 按整数、小数、科学计数、比例、时间等声明单位和范围 | `0`、未设置、默认值分别处理；HTML `min/step` 不能替代域校验 |
| 路径 | 全宽输入，复用模型库／数据集等现有选择器 | 草稿路径与已保存路径区别清楚，长路径可读，不自动改写 |
| 数组／原始参数 | 专属编辑器或 JSON／原始编辑退路 | 未知项保留，结构化字段与 `network_args` 冲突遵循已有 patch 合并契约 |
| 可解除冲突 | 可见但禁用，说明由哪项控制并可跳转 | 例如 epochs 覆盖 steps、权重决定 rank、AUTO 决定 swap |
| 不适用分支 | 默认隐藏，审计中显示原因与旧值 | 显隐不清空 baseline／draft；旧非法值提供明确修复动作 |

审计不是无条件编辑模式。不可用字段的修复可以是跳转父项、选择允许值或移除本文件覆盖；具体动作按字段能力定义。
禁用后，预检定位应将焦点放到可聚焦的字段说明容器或修复按钮，不能继续要求所有 `input` 都能获得焦点。

## 需要补充什么

| 项目 | 当前状态 | 本轮建议 |
| --- | --- | --- |
| Qwen T2I／Edit 任务入口 | 后端及数据集应用链已有；训练目录没有正式类型化 owner，可能通过 merged 未知键兜底出现 | 在「模型与任务」补 `t2i/edit` 选择和配对契约摘要；与数据集应用共用同一字段与规则 |
| 方法语义选择 | 已有模板和 Adapter；`network_module` 是普通文本 | 使用现有方法／Adapter 元数据提供可理解的选择，保留自定义模块入口；不能把网络变体开放等同于真机验证 |
| family 能力说明 | attention 等已有过滤；部分精度、loss、preview、swap 限制仍主要在兼容层 | 当前选择附近展示允许范围和原因；未知 family／能力加载失败保持阻断执行 |
| 高级损失等字段 | `loss_type/weighted_captions/t_min/t_max/layer_start/layer_end` 等需与当前 merged 集合对账；不应直接称为完全不可见 | T0 判定正式字段、审计字段或不属于 Web 支持面；先为 Qwen 固定 L2 等契约补说明，其他支持面确认后再开放编辑 |
| 优化器参数 | 已有原始参数，不适合把所有键复制到主表单 | 对实际支持且常用的参数增加结构化提示／编辑，保留扩展项；不新增未经后端支持的 optimizer |
| 有效值与来源 | 只有二值来源标签及变更预览 | 分开展示磁盘覆盖、合并有效值、当前草稿；优先加撤销该项，其次实现恢复继承 |
| 隐藏修改审查 | 修改项可保留并通过视图找回，但没有完整的专属摘要 | 保存前明确列出「当前不适用但已修改」和「旧配置不兼容值」，提供定位／撤销 |

数据集 subset、bucket、mask、参考图目录仍以数据集工作台为 owner；训练页补摘要和跳转，不复制整套蓝图编辑器。
`mixed_precision`、自适应精度、OOM retry、训练估算、预检定位都算已有能力的完善，不列入新增功能数量。

## 分阶段落地与验收门槛

每阶段独立提交、独立演示、通过门槛后再进入依赖阶段。实现阶段开始时重新确认工作树和运行构建版本；不把当前有未提交开发的工作树强行切换或重置。

| 阶段 | 实施内容与交付物 | 验收门槛 | 依赖 |
| --- | --- | --- | --- |
| T0 字段基线 | 生成逐键清单：来源、类型、选项、默认／未设置、family／method、阶段／簇／顺序、父项、序列化位置、测试入口；对账静态目录与代表配置 merged 集合 | 代表场景每个正式字段恰好一个 owner；缺项、重复、未知依赖、环、无类型被报告；未知键进入明确审计；输出逐键差异而非只比总数 | 起点 |
| T1 选择与状态 | 接通字段 `visible/enabled/reason`；修复不可用仍可编辑与搜索绕过规则；稳定字段类型、枚举和数值校验；后端能力同步 | epochs→steps、dim-from-weights→rank、AUTO→手动 swap、compile→子项，以及 family×自适应精度×mixed precision×OOM retry 矩阵通过；搜索／审计不改变权限；非法当前值保留且可修复；隐藏值不丢、未经编辑值不进 patch | T0 |
| T2 排列与空间 | 按上表调整簇顺序；方法选择前置；基础计算前置；自适应与 OOM 独立分组；合并顶部操作区，中窄屏配置库侧层化 | 1280×720 常规桌面编辑区目标至少 300px 高；1440×900 至少 420px；390×844 无整页横向溢出，工具可折叠且所有入口可达；DOM／键盘／视觉顺序一致；切组搜索草稿不变 | T1 |
| T3 补充入口和联动 | Qwen task 正式字段；模型／方法感知的约束摘要；分批补清单确认的正式遗漏；可选 optimizer 参数编辑 | Qwen T2I/Edit 与数据集配对、batch=1、缓存及不支持 preview 的规则一致；不支持值在选择时反馈；保存→重载→merged→preflight 含义一致；新增字段均有 owner、帮助及测试 | T0–T2 |
| T4 来源与变更审查 | 区分磁盘／合并／草稿；单项撤销；隐藏修改摘要；明确恢复继承的 API 语义；扩展预检定位 | 未设置、空串、`false`、`0`、继承、显式覆盖、未知字段全部 round-trip；恢复继承确实移除本文件覆盖；409／网络结果未知保留草稿且不重试；错误可定位隐藏／禁用／数据集项 | T1、T3 |
| T5 综合签收 | 模型与方法矩阵、视觉／键盘／缩放、保存与执行保护回归；隔离候选构建；更新文档；生产只读复核 | 本计划关键用例全部通过；新增阻断与数据回写问题为 0；当前审查中的 P1 均关闭或明确说明残余边界；交付版本、截图、测试报告、回退入口 | T2–T4 |

T2 的高度是本次提出的设计验收目标，不是现有性能数据。若硬件列表、缩放使固定高度目标不合理，应以可收起上下文保证编辑区，并记录调整后的基准。
T4 若需要后端提供来源链或删除覆盖操作，拆成单独接口变更验收；不能用写入空串、`false` 或 `null` 假装恢复继承。

第一批可评审切片已包含：epochs／steps 联动、方法选择前置、资源分组拆分和顶部布局收紧；Qwen 任务入口已作为补充切片落地。
下一批先完成 T0 完整 schema/差异产物、T1 自适应精度与 OOM retry 矩阵，以及 T2 训练字段键盘遍历；然后设计并单独验证 T4 provenance 与恢复继承 API，最后进入 T5 综合签收。

## 验收场景与证据

| ID | 场景 | 可判定的通过标准 |
| --- | --- | --- |
| A1 目录完整性 | Anima LoRA／LoKr／VeRA，Krea-2，Z-Image，Qwen T2I／Edit，未知 family，含未知键的旧配置 | 动态目录无重复、无未说明遗漏；类型与顺序不因当前值偶然变化；未知 family 明确阻断能力执行 |
| A2 联动与审计 | 上述四类父子依赖；切换 family／Adapter；搜索不适用项；父项关闭后存在非默认子值 | 默认显示规则正确，禁用原因具体，审计可解释并可修复；切换前后 draft／baseline 原值保持 |
| A3 选择与类型 | false／0／空值、科学计数学率、整数边界、枚举旧值、原始 JSON、结构化与原始 args 同改 | UI 到 patch 的类型与含义准确；不把空值变为 0，不把字符串 false 当 true；未知 args 不丢 |
| A4 持久化 | 临时配置根中读取→编辑→preview→保存→重载→merged→预检；无改动、只读另存、409、保存失败 | 仅有意修改的键变化；原始注释与未知键符合现有 patch 契约；失败不触发启动／入队，结果未知不自动重试 |
| A5 工作流 | 新建／加载→模型→数据→方法→训练量→性能→预检；dataset apply 与 Qwen mode 双向切换 | 全流程能解释每个生效值；不要求反复修改原始 TOML；未保存数据集与已保存训练引用状态可分辨 |
| A6 定位 | 普通字段、不可用字段、隐藏字段、数据集子项、已删除／未知键错误 | 自动选对阶段、展开正确簇、滚动可见、焦点可见；无法直接定位时明确提供所属编辑器入口 |
| A7 布局与键盘 | 1440×900、1280×720、768×1024、390×844；明暗主题、200% 缩放、长路径、库展开／收起 | 无主要操作被遮挡、无整页横向溢出；移动控件可触达；Tab 与箭头导航顺序稳定；弹窗焦点可恢复 |
| A8 执行边界 | 相同配置／preset／GPU 的预检、启动确认和暂停入队；双击、pending、失败与跨配置根 | payload 一致，预检失败不能执行，操作最多一次；GPU 选择维持运行上下文属性，不写回训练 TOML；根切换后无旧请求污染 |
| A9 实验精度 | family × `adaptive_precision` × 解析后的精度模式 × `mixed_precision` × OOM retry；设备或模式切换 | 复用后端自适应契约；关闭／BF16／FP32 模式不能启用仅适用于 FP16/FP32 路径的重试；`auto` 以解析后的模式判断；Qwen 拒绝自适应；单进程、attention、compile／swap、缓存、loss scale 和重试边界正确；保存重载不改变含义 |

自适应 `auto` 不能只按下拉框字符串推断执行精度。UI 展示策略与实际解析结果；硬件尚未确定时明确显示待解析，并由预检核验。
具体限制消费 [training_config.py](../../library/training/adaptive_runtime/training_config.py) 的契约；测试同时核对 family 兼容层，不在页面内复制一份可能漂移的判断矩阵。

布局回归使用合成配置截图，避免真实路径和数据出现在提交或外部评审材料中。
完整 GPU 训练质量、速度、多卡实验不作为本次表单整理的必需门槛；若触及训练语义，另立最小硬件验证并记录具体模型和组合，不能用 mocked E2E 证明训练能力。

### 现有测试入口与拟补覆盖

| 领域 | 现有入口 | 本轮需要补充 |
| --- | --- | --- |
| 字段与分组 | `trainingForm.test.ts`、`stageGroups.test.tsx`、`TrainingResourceGroups.test.tsx` | 四族真实 TOML 的 key/type/group/owner 差异、完整字段 schema、选项合法性、依赖拓扑、未知键审计 |
| 草稿／提交 | `TrainingWorkspace.test.tsx`、`TrainingLaunchDialog.test.tsx` | 隐藏修改摘要、禁用不写、恢复继承、同值显式覆盖 |
| 旧共享目录／披露 | `tests/test_dragon_config_stage_catalog_frontend.py`、`tests/test_dragon_config_disclosure_frontend.py` | Next 与共享纯规则的一致性；不能只跑旧 Dragon 测试 |
| 真实 Python 契约 | `tests/test_dragon_next_training_config_acceptance.py`、`tests/test_web_preflight_compat_matrix.py`、`tests/test_model_family_variant_selection.py` | 临时根保存后的 merged 与预检一致、Qwen task／能力边界 |
| Qwen 数据 | `editDataset.test.ts`、`DatasetApplyDialog.test.tsx`、`tests/test_web_qwen_image_edit_dataset.py` | 训练页任务选择与已有 dataset apply 的一致性 |
| 实验精度 | `tests/test_adaptive_training_precision.py`、`tests/test_adaptive_precision_contract.py` | 前端选项／禁用与后端模式解析、OOM retry 契约一致；需要补相应 React 测试 |
| 浏览器 | `training-draft-flow.spec.ts`、`training-preflight-locate.spec.ts`、`training-library-resize.spec.ts`、`scale-keyboard.spec.ts`、`model-task-capabilities.spec.ts` | 三态字段、恢复搜索、禁用焦点、补项与模型切换矩阵、编辑区高度、训练字段键盘路径 |

下列命令供后续实施执行，从仓库根目录运行；逐阶段选择相关文件，避免每次都跑完整前端。

```bash
# 类型检查 + 训练配置单元/组件测试；超过 60 秒时先拆小集合。
rtk test timeout 60 .venv/bin/python tasks.py web-next-check src/features/training-config

# 真实 Python 配置行为，测试使用临时根和 monkeypatch。
rtk test timeout 60 .venv/bin/python -m pytest tests/test_dragon_next_training_config_acceptance.py tests/test_web_preflight_compat_matrix.py

# 浏览器使用独立 Vite 端口与 mock API/WS；不对真实训练服务执行写操作。
rtk test timeout 60 .venv/bin/python tasks.py web-next-e2e e2e/training-preflight-locate.spec.ts
# draft-flow 用例较多，单独安排较长超时。
rtk test timeout 180 .venv/bin/python tasks.py web-next-e2e e2e/training-draft-flow.spec.ts

# 完成阶段的全前端检查，按实际执行时间安排并保留失败列表。
rtk test timeout 180 .venv/bin/python tasks.py web-next-check

# 隔离候选构建：必须设置目标，默认 web-next-build 会发布生产静态入口。
DRAGON_NEXT_DESTINATION=/tmp/dragon-next-training-config-candidate rtk test timeout 180 .venv/bin/python tasks.py web-next-build

# 服务上线后的只读验证：拦截非 GET/HEAD/OPTIONS；不启动或重启服务。
DRAGON_VERIFY_URL=http://127.0.0.1:20203 DRAGON_VERIFY_OUTPUT=/tmp/dragon-next-training-config-review rtk proxy node web/frontend-next/scripts/verify-production.mjs
```

隔离 mock 浏览器测试和临时根后端测试共同覆盖保存行为；生产只读检查仅证明已部署页面加载与显示，不能替代真实后端写契约验证。
构建结果、服务端版本和源码基线分别记录；前端静态更新不等于后端能力接口已更新。

## 实现边界与回退

- 按 `field schema/分组规则/能力适配/控件/来源状态` 拆职责，复用现有 feature 和 API；不把新业务继续堆进 `useTrainingWorkspace.ts` 或旧 chunks。
- catalog 可先通过适配层消费共享纯模块，逐步收敛事实源；不为一次排序全面重构旧前端。
- 每阶段保留输入输出契约，布局调整与训练数值／默认值变更分开提交。更改默认值必须有单独依据。
- Krea-2／Z-Image 当前 registry 已开放网络变体选择，不能恢复旧提案的固定 plain-LoRA 限制；同时不能将开放选择标成所有组合已真机验证。Qwen 当前仍按 plain LoRA 契约处理。
- 不修改用户导入配置、默认硬件选择、训练历史、队列和模型数据来通过验收；所有 mutation 验收使用隔离数据。
- 候选静态包验证后再发布；回退使用上一版静态入口或明确旧界面链接，不回滚业务数据，不停止训练。

## 主要代码依据

| 依据 | 核查点 |
| --- | --- |
| [fieldCatalog.ts](../../web/frontend-next/src/features/training-config/fieldCatalog.ts) | `fieldsForConfig` 动态字段集合／类型；`fieldAvailability`；选项过滤 |
| [trainingForm.ts](../../web/frontend-next/src/features/training-config/trainingForm.ts) | 显式基础字段、草稿类型、dirty 比较、patch 与 own-key 标签 |
| [TrainingFieldEditor.tsx](../../web/frontend-next/src/features/training-config/TrainingFieldEditor.tsx) | 控件禁用状态和不可用原因由合并后的字段规则驱动 |
| [useTrainingWorkspace.ts](../../web/frontend-next/src/features/training-config/useTrainingWorkspace.ts) | `visibleFields` 中搜索与视图条件、保存与执行状态 |
| [TrainingEditor.tsx](../../web/frontend-next/src/features/training-config/TrainingEditor.tsx) | 搜索计数、tab 切换、上下文与字段编辑器 |
| [stageGroups.ts](../../web/frontend-next/src/features/training-config/stageGroups.ts)、[resourceGroups.ts](../../web/frontend-next/src/features/training-config/resourceGroups.ts) | 簇顺序、默认展开、实验字段归组 |
| [family_registry.py](../../library/models/family_registry.py)、[compat_matrix.py](../../library/training/compat_matrix.py) | 当前模型族选择能力与最终训练组合约束 |
| `model-family.js`、`config-field-availability.js`、`config-field-disclosure-rules.js`（旧 Classic/Dragon 静态实现） | 历史共享能力、字段依赖与披露参考；当前 Next 实施入口见本表上方的 `web/frontend-next/src/features/training-config/` 文件 |
| [DatasetApplyDialog.tsx](../../web/frontend-next/src/features/dataset-editor/DatasetApplyDialog.tsx)、[editDataset.ts](../../web/frontend-next/src/features/dataset-editor/editDataset.ts) | 已有 Edit 配对及写入 task 的确认流程 |
| [training-preflight-locate.spec.ts](../../web/frontend-next/e2e/training-preflight-locate.spec.ts) | 已有定位流程与焦点断言，禁用改造需要同步扩展 |
| [前端工程说明](../../web/frontend-next/README.md)、[任务 wrapper](../../scripts/tasks/web.py) | 隔离测试、构建发布、生产只读检查边界 |

历史决策参考：2026-09-03 配置重组讨论及其渐进披露提案；具体支持范围与当前实现以以上实时源码为准。
