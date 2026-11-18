# Dragon 训练配置渐进披露重组提案

状态：四阶段目录、首批动态披露与紧凑参数工作台已实现

基线日期：2026-09-03

基线清单：[Dragon 训练器 215 项配置清单](../configuration/dragon-training-config-215.md)

## 当前工作台（2026-09-06）

本节记录当前已实现行为；下方旧进度与目标设计保留为演进记录，冲突时以本节和实时源码为准。

- 无显式子路由时默认进入参数工作台，旧分组视图保留在“参数显示设置”中，Classic 不变。
- 桌面配置选择侧栏默认常驻，支持文件搜索、分组折叠与原有管理操作；小屏使用可关闭侧栏，不改成只有下拉框。
- 默认显示全部当前适用参数。旧版本的自动折叠和等级偏好不再决定新工作台的首次布局；用户在新版主动选择的范围与折叠状态继续保存，双语等其他偏好保留。
- 四阶段导航位于编辑区上方；字段按 catalog 的阶段和语义簇顺序，使用稳定行序的两列表单，小屏单列。路径和多行值占整行，不再采用固定 132px 卡片和 dense 补位。
- 快速模型配置放在模型路径标题旁，数据集工具留在输入阶段，步数估算紧随“训练量”，保存与修改摘要固定在编辑区底部。
- 搜索匹配参数名、配置键和当前值，临时跨越显示等级，但不绕过模型/方法适用性。搜索和查看修改时隐藏空阶段、空字段簇，普通浏览保留阶段锚点。
- 清空搜索恢复原浏览位置；点击阶段会退出搜索和修改筛选。桌面内层滚动、小屏页面滚动都使用同一定位与焦点可见性模块。
- 渲染重排没有缩减候选字段、改变序列化或清理隐藏值。保存、预检、入队和启动仍沿用原有训练上下文与草稿流程。

新增运行时回归位于 `tests/test_dragon_config_workspace_runtime.py`，覆盖偏好迁移、搜索与隐藏草稿、侧栏筛选/折叠和双滚动上下文；字段全集与依赖顺序继续由 stage catalog 测试约束。

## 历史实现进度（2026-09-04）

已落地的首个切片：

- `config-field-catalog.js` 成为 Dragon 全参数画布的字段 owner、阶段、字段簇、位置、策略和依赖顺序目录。
- 全参数画布由八章节收敛为四个固定阶段，当前 218 项归属为 `21 / 105 / 44 / 48`；三个 gradient-flow probe 归入“资源与预检 / 诊断与兼容”。
- 阶段导航只执行滚动定位和 scrollspy，已删除“定位 / 过滤”双模式及其持久化状态。
- 搜索、只看修改、旧等级和隐藏不可用仍作为兼容过滤；四个阶段本身始终保留，空结果显示阶段空态。
- 冻结 215 项和当前 218 项分别有覆盖回归，并校验唯一 owner、未知阶段/字段簇、缺失引用、逆向依赖、依赖环和关键依赖顺序。

随后落地的第二个切片：

- 新增独立的 `config-field-disclosure-rules.js`，首批覆盖 LoRA / LoKr / VeRA / DoRA、Krea-2 / Z-Image plain LoRA、Ortho / ReFT / MoE / Chimera / IP-Adapter / EasyControl / Soft Tokens / SPD，以及 pipeline、ConvRot、compile、checkpoint、block swap、cache 和 probe 父子条件。
- 主画布已经启用真正的 `visible=false`；当前上下文不相关字段不再占位，“查看全部候选”可临时审计隐藏字段及原因。
- 顶部显示动态的“适用 N / 候选 M”，搜索会报告隐藏匹配数，四阶段标题计数随当前披露和搜索结果更新。
- 完整候选 DOM、`scopeKeys`、`baselineValues` 和 `draftValues` 仍保留；显隐不参与 patch 取值，隐藏但已修改的字段会计入 dirty 摘要并可从“查看修改”定位和撤销。
- `plain_lora_only` 现在先于旧配置中的 Ortho、ReFT、MoE、Chimera 等开关求值，Krea-2 / Z-Image 不会因遗留真值或原始 `network_module` / `network_args` 入口重新暴露不支持的方法分支；未知模型族对 Adapter、方法分支和能力字段 fail closed。
- 迁移期保留的旧分组视图也消费同一份披露状态，未知 / `audit_only` 字段默认隐藏且不可编辑，不能再从旧入口绕过规则写回配置。
- LoKr 模板保留完整 9 个 LoKr 专属字段，切回普通 LoRA 后即时隐藏；当前适用数量随 family、方法和通用父子项呈现规则动态变化，不作为固定验收常量。

根据旧分组页截图完成的默认面校正：

- 默认显示范围覆盖截图中 7 个必填项、20 个常用项，以及可完整辨认的 35 个“显存与速度优化”项；完整记录见 [215 项清单的截图补充](../configuration/dragon-training-config-215.md#补充旧分组页的默认暴露基线)。
- 流水线并行、block swap、选择性重算、compile、显存/峰值/梯度流探针和 cache reuse 等通用子项不再因父值关闭而隐藏；它们保持渲染并显示为禁用，在当前模型族和能力契约允许时，父值开启后即时恢复编辑。
- Adapter 变体选择器改为常态渲染；Anima 可直接切换，Krea-2 / Z-Image 固定显示 plain LoRA 并给出禁用原因，遗留的非 plain-LoRA 值仍允许切回 `lora`。
- ConvRot、方法、Adapter 和模型族专属参数分支仍按当前上下文隐藏，避免普通 LoRA 看见 LoKr / VeRA 子项，或 Krea-2 / Z-Image 绕过 plain-LoRA 限制。
- 旧存储 id 暂时兼容，界面文案改成“精简 / 默认 / 全部适用”；无偏好时进入“默认”，不再以“新人 / 初学者”描述用户。

仍待后续完成：为全部能力字段补齐更细的 family capability（尤其 compile / checkpoint）、统计“隐藏且非默认”的旧配置值、让 preflight 错误自动打开审计入口，以及最终移除新人 / 初学者等级和旧分组视图。

## 结论

按需求中的例子，本文把“纰漏式”理解为**渐进披露式**：先按当前训练上下文显示长期通用项，再由模型族、方法、Adapter 和功能开关逐层披露相关参数。

建议最终：

1. 移除“新人 / 初学者 / 高级”静态字段等级。
2. 不再同时维护“五个导航分类”和“八个画布章节”。
3. 统一成一个连续编辑流，只保留四个稳定阶段：`输入准备`、`方法配置`、`训练计划`、`资源与预检`。
4. 默认只渲染当前上下文适用的字段；不相干字段不占主画布位置。
5. 保留“查看全部候选”审计入口，用于检查隐藏字段、旧配置值和未分类新字段。
6. 四阶段导航只负责定位和显示进度，不再同时承担“定位 / 过滤”两种模式。

“实验性”“必填”“当前不可用”继续作为字段属性或状态，不再充当一级分类。

## 为什么要重组

当前页面给同一批字段叠加了三套组织维度：

- 五个导航分类：`7 / 38 / 6 / 60 / 104`。
- 八个画布章节：`7 / 35 / 6 / 43 / 13 / 5 / 60 / 46`。
- 历史三档显示等级：新人 `59`、初学者 `12`、高级 `144`；当前仅保留其存储 id 兼容，界面显示为“精简 / 默认 / 全部适用”。

这三套维度解决的是不同历史问题，但组合后产生了几个明显缺陷：

- “高级”收纳了 `104 / 215` 个字段，已经失去分类意义。
- “优化”同时包含显存、预处理、损失实验和正则化，任务语义不一致。
- LoKr 固定出现在“常用”，又在画布里归入“方法架构”；普通 LoRA 用户仍会看到 LoKr 专用项。
- “高级”既可以表示字段难度，也可以表示兜底位置；用户无法判断它是风险等级还是功能类别。
- 新字段默认进入高级层级，可能在过滤状态下无提示消失。
- 此前默认等级实际是 `all`，所以新人/初学者并未形成稳定的入门流程，只增加了一套过滤状态；截图校正后，无偏好默认使用明确的“默认”字段面。
- 在 `1280 x 720`、右侧预设库展开的实测页面中，八章节导航已经变成横向滚动条，首屏只能看到前几个章节；分类顺序本身也需要滚动查找。

历史上五分类的首要目标是让 Dragon 与 Classic 共享完整字段目录，避免字段丢失；它不是最终的信息架构。全部参数视图随后解决了跨页查找，但仍主要依靠“保留显示 + 禁用”表达不适用字段。

本轮在当前工作树重新打开页面时，运行时目录已从冻结快照的 `215` 项变为 `218` 项，“实验与高级”也从 `46` 项变为 `49` 项。新增的
`gradient_flow_probe_jsonl`、`gradient_flow_probe_every_n_steps`、
`gradient_flow_probe_dense_steps` 再次通过兜底逻辑追加到末尾。本文继续以已归档的
`215` 项清单作为可复核基线，不把 `218` 混入下方统计；这次漂移本身说明字段 owner、簇和顺序必须由机器可读 catalog 强制声明。按本文语义，这三个新字段应进入“资源与预检 / 诊断”。

## 目标信息架构

### 三层页面结构

页面不再被理解成一排分类按钮加一张长表，而是三个职责不同的层次：

| 层次 | 内容 | 是否参与四阶段导航 |
| --- | --- | --- |
| 页面操作层 | 当前配置文件、保存、预检、入队、开始训练 | 否 |
| 训练上下文层 | 模型族、方法、Adapter、数据集、运行覆盖和设备 | 否 |
| 连续编辑层 | 输入准备、方法配置、训练计划、资源与预检 | 是 |

页面操作不改变字段 owner；训练上下文决定下方披露集合；连续编辑层才是用户逐段完成配置的主体。这样避免把“配置文件管理”“训练对象选择”和“参数分类”混成同一排控件。

### 分类、披露、状态和顺序分离

新的 catalog 不再用一个“分类”同时承担四种职责：

| 维度 | 回答的问题 | 表达方式 |
| --- | --- | --- |
| 稳定归属 | 这个字段去哪里找？ | `location + stage + cluster`，每个字段唯一归属 |
| 动态披露 | 当前任务需不需要它？ | family / method / Adapter / feature 条件 |
| 字段状态 | 它是否必填、实验性、冲突或已修改？ | 徽标、禁用原因、错误和 dirty 状态 |
| 依赖顺序 | 用户必须先决定什么？ | `controlledBy / dependsOn` 拓扑顺序 |

“通用”也不是第五个分类，而是披露策略 `always`。例如学习率长期显示在“训练计划 / 优化器与学习率”，LoKr factor 永远归“方法配置 / Adapter 分支”，只在 LoKr 上下文披露。这样字段切换方法时只是显隐变化，不会在分类之间跳来跳去。

### 顶部固定区

顶部固定区分成两个带区，不计入四阶段正文：

- **配置与操作带**：当前训练配置文件、保存、预检、入队、开始训练。
- **训练上下文带**：模型族 `model_family`、当前方法、LoRA family 内的 Adapter `lora_adapter_kind`、数据集 preset / `dataset_config`、训练设备和运行覆盖 preset。

逻辑求值与键盘顺序固定为：

```text
配置文件 -> 模型族 -> 方法 -> Adapter -> 数据集 -> 设备 -> 运行覆盖
```

模型族先约束方法能力；方法再决定是否需要 Adapter 子类型以及数据集的额外契约；设备先确定执行目标，运行覆盖再对该目标应用硬件档位。`output_name` 等不改变字段披露的值不塞进上下文带，仍放在所属阶段。

这里的“方法”是决定披露集合的**方法选择**，在进入正文前已经确定；后面的“方法配置”阶段只编辑 rank、目标层和当前方法分支，不再选择一次方法。同理，顶部先选择数据集，随后“输入准备”按已经选定的方法检查并披露该数据集需要的配对图、mask、caption 或缓存契约。这样保留 `输入 -> 方法参数` 的编辑顺序，同时满足 `方法选择 -> 数据契约` 的依赖。

这不是要求把七个控件硬塞进一行。当前 `1280 x 720` 页面中，右侧预设库打开后，三组选择器和三个操作按钮已经接近宽度上限。桌面端应把“配置文件 + 页面操作”与“训练上下文”分开；上下文内部可按 `模型 / 方法 / Adapter`、`数据集 / 设备 / 运行覆盖` 自然换成两行。窄屏按同一 DOM 顺序纵向排列，不能为了视觉拼贴改变 Tab 顺序。

这些值决定下方字段集合。切换后应显示一条简短摘要，例如：

```text
Krea-2 Raw · LoRA · default
当前适用 N 项 · 已隐藏 H 项 · 保留 K 项非当前配置值
```

数量是当前上下文的结果，不写成永久常量。

### 四个连续阶段

| 顺序 | 一级阶段 | 导航短名 | 二级字段簇 | 收纳原则 |
| ---: | --- | --- | --- | --- |
| 1 | 输入准备 | 输入 | 模型路径、数据集、caption / mask、筛选、缓存语义 | 确认训练输入及其准备状态 |
| 2 | 方法配置 | 方法 | 热启动、rank / alpha、训练目标层、当前方法分支 | 决定训练哪些参数、使用哪种结构 |
| 3 | 训练计划 | 训练 | 任务身份、训练量、优化器与学习率、时间步与损失、正则化、训练预览、保存与日志 | 决定如何训练、验证和保存 |
| 4 | 资源与预检 | 资源 | 设备拓扑、计算路径、checkpoint、block swap、compile、预处理执行资源、诊断 | 适配当前硬件并确认配置可运行 |

一级阶段只回答“我现在进行到训练配置的哪一步”。字段难度、实验性和可用性由徽标及状态表达，不再改变字段所在位置。完整标题用于正文，短名用于窄屏或紧凑导航。

完整任务流因此收敛为：

```text
加载配置 -> 锁定训练上下文 -> 输入 -> 方法 -> 训练 -> 资源 -> 预检 -> 保存 / 入队 / 开始
```

这里故意把“方法配置”放在“训练计划”之前：Adapter、rank 和扩展架构会改变可训练参数量、合理学习率及显存压力。“资源与预检”最后配置，因为它依赖模型、数据分辨率、方法容量、batch 和训练目标。

“预检”是跨阶段检查命令，不作为第五个分类；保存、预检、入队和启动继续靠近页面级操作区。

### 冻结 215 项的候选归属

按上述语义对冻结快照逐项归属，静态候选分布为：

| 阶段 owner | 候选项 | 当前引擎判定可用 | 当前引擎判定不可用 |
| --- | ---: | ---: | ---: |
| 输入准备 | 21 | 21 | 0 |
| 方法配置 | 105 | 52 | 53 |
| 训练计划 | 44 | 41 | 3 |
| 资源与预检 | 45 | 34 | 11 |
| **合计** | **215** | **148** | **67** |

这是**目标设计的重新归属**，不是当前源码已经提供的四阶段统计。它按以下步骤从现有八章快照复算：

1. 初始聚合：输入准备 `7 + 13 = 20`；训练计划 `35 + 6 + 5 + 7 = 53`；资源与预检 `43`；方法配置 `60 + (46 - 7) = 99`。其中实验与高级章里的 7 个 loss weighting 字段先归训练计划。
2. `model_family` 从末尾高级兜底移到输入准备。
3. `network_train_unet_only`、rank / alpha、Adapter、DoRA、LoKr factor、VeRA 3 项、热启动权重和 `dim_from_weights` 共 11 项，从核心训练移到方法配置。
4. `lr_warmup_steps` 从显存与速度移到训练计划；`prior_loss_weight` 从高级兜底移到训练计划。
5. DataLoader workers、VAE chunk / cache、pin memory、persistent workers 共 5 项，从数据与缓存移到资源与预检。
6. cache copy / latent / text reuse、fingerprint、force rebuild 共 5 项，从显存与速度移到输入准备。
7. `compile_dynamic_seq`、`compile_seq_bands`、`activation_memory_budget` 共 3 项，从高级兜底移到资源与预检。

完成上述唯一移动后得到 `21 / 105 / 44 / 45`。这些归属现已落入机器可读 catalog，并由测试从字段键集合直接生成统计；该表继续作为冻结 215 项的提案基线。

当前工作树新增的三个 gradient-flow probe 均由现有引擎判为可用。它们加入上述 owner 后，当前观测分布为 `21 / 105 / 44 / 48 = 218`，可用 / 不可用为 `151 / 67`。这组数字不回写冻结的 215 项清单；实现同时保留“215 历史快照回归”和“当前运行时 M 项全覆盖”两类测试。

`105` 个方法候选不是需要继续拆出“高级”的信号。它们由互斥的方法分支组成，默认视图只呈现当前方法；静态候选数量不应被当作页面长度。

归属与披露条件保持正交：

- `weight_decay` 即使只在 SPD 下生效，语义 owner 仍是“训练计划 / 优化器与损失”。
- `ip_features_cache_to_disk` 即使只在 IP-Adapter 下出现，owner 仍是“输入准备 / 缓存策略”。
- `compile_dynamic_seq` 即使只被部分 family 支持，owner 仍是“资源与预检 / Compile”。
- 顶部上下文中的 `model_family`、`dataset_config`、`lora_adapter_kind` 仍分别计入唯一 owner，不能在正文重复渲染。

### 二级字段簇的固定顺序

以下均为**待实现的目标顺序**。当前 `FORM_SECTION_DEFS` 尚未承载四阶段映射，compile 动态字段等仍会由未分类字段兜底补到末尾。

#### 1. 输入准备

1. **模型路径**：语义顺序为 `model_family` -> `pretrained_model_name_or_path` -> `qwen3` -> `vae`；`model_family` 由顶部上下文条承载，正文不重复控件。
2. **数据集**：`dataset_config` 及其状态摘要；选择器由顶部上下文承载，本阶段显示方法感知的契约检查，详细 subset / bucket 编辑继续留在数据集工作台。
3. **Caption 与遮罩**：caption 变体、dropout、`masked_loss`。
4. **输入筛选**：`path_pattern`、`drop_lowres_images`、`min_pixels`。
5. **缓存策略**：`use_vae_cache`、`use_text_cache` 先于 cache reuse、fingerprint、skip check 和 force rebuild；IP 视觉缓存按方法披露。

DataLoader worker、VAE chunk 和预处理 batch 不放在这里。它们不改变训练输入语义，统一移到“资源与预检”。

#### 2. 方法配置

1. **方法契约**：顶部上下文已经承载规范化 method 和 `lora_adapter_kind`，正文只显示当前方法摘要与能力限制，不重复第二套选择器。
2. **热启动**：`network_weights` -> `dim_from_weights`。
3. **容量与目标层**：`network_dim`、`network_alpha`、`network_train_unet_only`、`train_adaln`；从权重读取维度时 rank / alpha 保持可见禁用。
4. **当前 Adapter 分支**：DoRA，或 LoKr 的 factor + 8 个子项，或 VeRA 的 3 个子项。
5. **可组合扩展**：Ortho / Timestep mask -> ReFT -> MoE / router / FEI -> Chimera；每个父开关先于子项。
6. **当前独立方法**：IP-Adapter、EasyControl、Soft Tokens 只呈现当前命中的一个方法簇。
7. **原始参数**：`network_args` 放在该阶段末尾的审计/兼容区，避免先于结构化控件出现。

方法阶段标题可以动态显示为“方法 · LoRA”“方法 · LoKr”或“方法 · IP-Adapter”。普通 LoRA 没有额外分支时，不渲染空小节。

SPD 不走 Web 普通训练按钮，而是 `scripts/distill_spd.py` 的专用实验流程。其字段为兼容旧配置仍进入 catalog 和保存 round-trip，但普通四阶段编辑流不显示；只有显式进入 SPD 实验模式时才使用专用流程界面，其他时候仅能从审计入口查看。当前 availability 只是把非 SPD 字段返回为 `enabled=false`，迁移后必须明确变为：主画布 `visible=false`，审计入口 `visible=true, enabled=false` 并保留原因；不能把现有禁用行为误当成目标已经实现。

#### 3. 训练计划

1. **训练身份**：`output_name`。
2. **训练量**：epochs / steps 模式 -> 对应上限 -> batch -> gradient accumulation -> sample ratio。
3. **优化器与学习率**：`optimizer_type` -> `learning_rate` -> `optimizer_args` -> `lr_scheduler` -> `lr_warmup_steps`；优化器类型先控制其专用参数，通用学习率仍保持在首屏可见位置。
4. **时间步与损失**：`timestep_sampling` -> `discrete_flow_shift` -> `weighting_scheme` -> 对应 Min-SNR / P2 / Sigmoid 子项 -> velocity loss。
5. **正则化**：prior、blank prompt、diff output 等驱动值先于依赖项；dataset reg 和无数据集正则化在说明中明确区分。
6. **训练预览**：prompts -> epoch / step 频率 -> `sample_at_first` -> sampler -> `seed`。
7. **保存与续训点**：保存格式 / 精度 -> 普通权重周期与保留数 -> training state 周期与保留数。
8. **日志与验证**：日志频率 / 目录 / 后端 -> CMMD 和方法专用诊断周期。

训练预览和正则化保留独立二级锚点，但不升级为一级分类。

#### 4. 资源与预检

1. **设备拓扑**：runtime 可用时，pipeline 开关 -> stages -> microbatches -> schedule -> split；拓扑选择先于单卡显存策略。
2. **计算路径**：训练精度 -> `base_compute` -> family 支持的 `attn_mode`；ConvRot 子项紧随其计算路径。
3. **激活显存**：gradient checkpoint / offload -> selective checkpoint -> 非 `off` 时的 blocks。
4. **模型驻留 / Block swap**：`blocks_to_swap` -> transfer dtype -> restore mode -> eval 行为。
5. **Compile**：`torch_compile` -> scope / mode -> dynamic seq / bands / activation budget -> custom autograd；swap 已确定后再判断可编译范围。
6. **预处理执行资源**：cache batch、memory profile、precision preference、DataLoader workers / pin / persistent、VAE chunk。
7. **诊断与兼容摘要**：block-swap profile、memory probe、peak probe、gradient-flow probe；每个输出开关先于 max steps / level，并显示最近一次预检的跨阶段结果。

这里按“决策依赖”排列，不按训练进程的时间线机械排列：设备拓扑会改变后续 checkpoint、swap 和 compile 的合法组合，因此先于它们；预处理虽然先执行，但其 worker / chunk 是较低层的资源微调。顶部“训练前检查”是全页唯一的预检触发器；本阶段只显示资源摘要与最近结果，不放第二个预检按钮。结果可以回链到任何阶段的具体字段。

### 导航与排序契约

四阶段导航采用单一行为：点击后滚动到对应阶段，滚动时用 scrollspy 更新当前位置。删除当前“定位 / 过滤”模式切换；渐进披露已经负责去除无关字段，章节按钮再过滤一次只会制造第二套可见性状态。

- 导航顺序永久为 `输入 -> 方法 -> 训练 -> 资源`，不会因某阶段当前字段少而换位。
- 空的二级字段簇不渲染；四个一级阶段和导航锚点始终保留。某阶段没有当前适用字段时显示紧凑空态，不隐藏或让其他阶段补位。
- 阶段导航默认不显示字段数量，避免继续诱导用户比较静态桶大小，也避免 context / body / audit 三种 location 造成口径噪声。只显示当前位置，以及确有意义的错误或已修改徽标。
- `21 / 105 / 44 / 45` 只用于文档审计和 catalog 覆盖测试，不硬编码进界面。顶部 `N / M` 是唯一总量入口。
- 搜索、只看修改和审计入口可以缩小结果，但不得重排字段或改变 canonical Tab 顺序。
- “查看全部候选”沿用同一 canonical 顺序，只额外插入隐藏原因，不恢复旧五分类或八章节顺序。
- 条件分支紧跟父控件出现。选择 LoKr 后，9 个 LoKr 字段插入 Adapter 分支固定槽位，而不是追加到方法区末尾。

每个字段应在 catalog 中显式声明完整契约，而不是只有一个数字顺序：

```js
{
  location: 'context' | 'body' | 'audit_only',
  stage: 'input' | 'method' | 'training' | 'resources',
  cluster: 'stable-cluster-id',
  policy: { kind: 'always' | 'conditional' | 'audit_only' },
  controlledBy: ['visibility-or-enable-driver'],
  dependsOn: ['ordering-predecessor'],
  siblingOrder: 30,
}
```

- `stage` 始终表示唯一语义 owner，即使 `location=context` 的控件不在正文重复渲染。
- `controlledBy` 只描述披露或启用关系，可以引用顶部上下文或 canonical 顺序中更早的字段。
- `dependsOn` 生成排序边；引用字段必须位于相同或更早的 stage / cluster，不能靠拓扑排序把子字段拉过阶段边界。
- `siblingOrder` 只处理没有依赖边的同级字段；仍相同时用配置键作确定性 tie-break，不用旧 catalog 索引继续固化历史乱序。
- 缺少引用、未知 stage / cluster、重复 owner、逆向依赖或依赖环必须让 catalog 测试失败。运行时遇到未分类新字段时只进入 `audit_only / unclassified` 列表并显示维护警告，不能自动追加到正文末尾。

导航、DOM、视觉栅格和键盘顺序必须消费同一份排序结果。阶段 1 还应生成按 stage / cluster 排序的逐键快照和旧位置 -> 新位置 diff；只有总数相等不足以证明 105 个方法候选或 11 个迁移项归属正确。

### 当前最需要修正的逆序

| 当前问题 | 调整后位置 |
| --- | --- |
| `model_family` 位于第 210 项，晚于所有能力相关字段 | 顶部上下文和“模型路径”首位 |
| `lora_adapter_kind` 位于第 18 项，但 `network_module` 到第 118 项才出现 | 顶部上下文按 method -> Adapter 连续排列 |
| `lokr_factor` 与其余 8 个 LoKr 字段分散 | 当前 Adapter 分支内连续排列 |
| cache reuse / fingerprint 先于 `use_vae_cache`、`use_text_cache` | 缓存主开关先于重用和校验子项 |
| `weighting_scheme` 及损失参数位于方法架构之后 | 紧随 timestep / flow shift |
| compile 主字段与第 211--213 项动态 compile 字段分散 | Compile 小节连续排列 |
| `prior_loss_weight` 位于第 215 项，脱离正则化 | 训练计划 / 正则化 |
| 共享 `seed` 当前被 SPD 方法作用域误伤 | 待修复：训练预览正常显示；SPD 只复用同一值和场景说明 |

### 排列备选

- `输入准备 -> 训练计划 -> 方法配置 -> 资源与预检` 更接近传统表单，但切换方法后往往要回头重调学习率和显存，不推荐。
- `输入准备 -> 方法配置 -> 资源与预检 -> 训练计划` 适合硬件基准工作台，但会让普通训练用户在明确训练目标前先面对复杂资源参数。
- 单独增加“输出与预览”第五阶段能让静态数字更均衡，但只有十余个高频字段，不值得增加一级导航；保留二级锚点即可。

### 为什么不是四个静态大组

静态候选数仍会失衡，尤其“方法配置”拥有大量候选字段。但渐进披露后，用户不会同时看到所有方法：

- 选择普通 LoRA：只显示通用 Adapter 容量与普通 LoRA 选项。
- 选择 LoKr：在“方法配置”的 Adapter 固定槽位追加 9 个 LoKr 专用字段，隐藏 VeRA 等其他分支。
- 选择 IP-Adapter：显示 IP 条件与视觉编码相关字段，隐藏 Soft Tokens、EasyControl 和 SPD。
- Krea-2 / Z-Image 的 `plain_lora_only` 能力成立时，不渲染不受支持的方法分支。

因此目标应优化“当前适用字段数”，而不是让静态候选字段平均分配。

## 渐进披露规则

### 决策顺序

字段集合按以下顺序求交集：

```text
全部候选字段
  -> 模型族能力
  -> 当前方法
  -> Adapter 种类
  -> 功能开关或模式
  -> 字段间依赖
  -> 搜索 / 只看修改 / 审计视图
```

搜索和视图过滤只能缩小已经适用的字段集合，不能绕过模型能力约束；四阶段导航只定位，不参与求交。

### 三种呈现状态

| 状态 | 使用条件 | 示例 |
| --- | --- | --- |
| 显示且可编辑 | 当前上下文适用 | LoRA 下的 rank、alpha、学习率 |
| 显示但禁用 | 字段与当前任务有关，且用户能从附近控件理解或解除冲突 | `max_train_epochs` 已设置时的 `max_train_steps`；从权重读取维度时的 rank / alpha |
| 默认隐藏 | 与当前模型族、方法、Adapter 或专属计算模式无关 | LoRA 下的 LoKr 后端；未启用 ConvRot 时的 7 个 ConvRot 子项 |

不再把所有不适用字段都渲染成禁用卡片。禁用态用于解释可解除的冲突；真正无关的分支直接隐藏。

### 首批应覆盖的条件簇

| 驱动字段 / 上下文 | 长期显示 | 条件显示 |
| --- | --- | --- |
| `model_family` | 当前模型族、已声明的支持能力摘要 | family 不支持的方法和 attention 选项隐藏；compile / checkpoint 必须先有独立能力声明，不能从现有 registry 猜测 |
| `lora_adapter_kind` | Adapter 选择、rank、alpha、热启动 | LoKr 9 项仅 `lokr`；VeRA 3 项仅 `vera`；DoRA 仅普通 `lora` |
| `pipeline_parallel` | 开关及 stages、microbatches、schedule、split 长期显示 | 父开关关闭或 runtime 不可用时，子项显示但禁用 |
| `base_compute` | 计算路径 | 7 个 ConvRot 参数仅 `w8a16_convrot` / `w8a8_convrot` |
| `selective_checkpoint` | checkpoint 模式与 blocks 长期显示 | `off` 时 blocks 显示但禁用 |
| `blocks_to_swap` | 交换块数及 transfer dtype、restore mode、eval/profile 长期显示 | 数量为 0 时子项显示但禁用 |
| `torch_compile` | compile 开关及通用子项长期显示 | 关闭时子项显示但禁用；family 明确不支持时按能力拒绝 |
| `memory_probe_jsonl` | 显存探针及 max steps 长期显示 | `off` 时 max steps 显示但禁用 |
| `peak_probe_jsonl` | 峰值探针及 max steps、level 长期显示 | `off` 时子项显示但禁用 |
| `gradient_flow_probe_jsonl` | 梯度流探针及 every-n-steps、dense-steps 长期显示 | `off` 时子项显示但禁用 |
| 训练预览频率 / `sample_at_first` | 是否生成训练预览 | prompts、sampler、seed 在预览会实际运行时显示 |
| `use_vae_cache` / `use_text_cache` | 缓存主开关与复用项长期显示 | 主开关关闭时各自复用项显示但禁用 |
| `add_reft` | ReFT 开关 | dim、alpha、layers 仅开启且 family 支持 |
| `use_moe_style` | MoE 风格 | 专家数、均衡、router 与 sigma 子树按风格和来源显示 |
| `use_ip_adapter` 或方法上下文 | IP-Adapter 入口 | 配对、编码器、resampler、PE-LoRA 子项按功能继续展开 |
| `use_easycontrol` 或方法上下文 | EasyControl 入口 | drop、noise、gate、FFN、token 子项 |
| `use_chimera_hydra` 或方法上下文 | Chimera 入口 | 内容池、频率池和各自 router 子项 |
| `weighting_scheme` | 损失加权方案 | Min-SNR、P2、Sigmoid 参数仅对应方案 |
| 正则化驱动值 | prior / blank / diff 的入口值 | class、mask weight 等依赖项按实际启用状态显示 |

`seed` 当前同时被预览和 SPD 复用，不应因为 SPD 未启用而在预览中禁用。它仍是一个配置键和一个持久化值，由预览与 SPD 两个场景共同引用；切换场景只改变说明和显示位置，不复制字段、重命名键或互相覆盖值。

LoKr 的完整分支是 9 项：顶层 `lokr_factor` 加上 8 个 `network_args` 映射字段。实现不能只遍历 `NETWORK_ARG_FIELD_MAP`，否则会漏掉顶层字段。

### 能力来源缺口

现有模型族 registry 已声明 network、plain LoRA、attention、部分 selective LoRA 和 pipeline 拓扑能力，但尚未完整声明 compile、gradient checkpoint、cache、probe 等训练配置能力。因此：

- 第一阶段只能按已有明确能力收缩方法和 attention 分支。
- compile、checkpoint、cache、probe 在新增后端能力契约前，继续使用各自经过测试的 compatibility 规则。
- 不允许用 `plain_lora_only` 推导“该模型不支持 compile”之类无关结论。
- 新能力字段应由后端 registry / API 提供，前端披露规则只消费，不复制支持矩阵。

Dragon 训练配置页现在对未知模型族 fail closed：模型族值本身仍可修复，Adapter 与方法专属分支默认隐藏，`network_module`、attention 等能力字段禁用；保存和 preflight 继续负责最终拒绝。共享的模型族 option helper 仍保留历史兼容行为，其他页面若接入未知 family，必须显式采用同等门禁，不能依赖该 fallback 推断支持能力。

### 方法上下文规范化

“当前方法”与“当前 Adapter”是两层概念：

1. 方法优先由 `methodsSubdir + variant` 的已知映射确定。
2. 再由 `network_module` 和明确的 method flags 识别 IP-Adapter、EasyControl、Soft Tokens、Chimera 等方法。
3. `lora_adapter_kind` 只在 LoRA family 内区分 LoRA / LoHa / LoKr / GLoRA / VeRA，不能代替方法识别。

规则评估器应复用一个规范化函数，页面渲染、保存摘要和 preflight 不得各自推导一次。

## 当前页面可缩减多少

基线页面有 215 个配置块：

- 当前可编辑 148 项。
- 当前禁用 67 项。
- 仅打开“隐藏不可用配置项”，最多只能收缩到 148 项。

这仍不是完整的渐进披露，因为当前引擎没有覆盖 compile、swap、cache、probe、ReFT、MoE、IP、EasyControl 等大量父子依赖。

早期方向性测算曾按“父功能关闭即隐藏子树”估计 `krea2_raw + plain LoRA` 为 90--100 项。截图确认通用父子项应长期渲染后，该区间已经失效。当前不再预设适用项数量；验收以字段规则正确、默认面覆盖截图基线、方法分支无串线和隐藏值不丢失为准。

阶段 1 完成后，应由同一规则评估器输出每个测试场景的精确数字，并把字段增减 diff 写入快照测试。验收关注“规则正确且无遗漏”，不要求数字落在某个预设区间。

建议界面同时显示：

```text
当前适用 N / 全部候选 M
```

`M` 由当前运行时 catalog 生成；冻结快照中为 `215`，当前工作树实测为 `218`。数字点击后打开审计入口，而不是恢复新人/高级等级。

## 查看全部候选

动态隐藏必须提供可发现、可审计的退路：

- 默认视图：仅当前适用字段。
- 次级入口“查看全部候选”：展示隐藏字段及隐藏原因。
- 搜索无结果时提示“另有 N 个不适用字段匹配”，允许临时查看。
- 未分类新字段不得静默隐藏；统一进入“未分类”列表并显示维护警告。
- 当前配置中存在非默认但已隐藏的值时，顶部显示数量和查看入口。
- preflight 指向隐藏字段时，自动打开审计入口并定位该字段，保证旧配置仍可修复。

“查看全部候选”是调试与迁移工具，不应重新成为默认的 `M` 项长画布。

## 数据安全与保存语义

渐进披露只改变渲染，不改变配置数据：

1. 隐藏字段的 baseline、draft 和磁盘值必须保留。
2. 切换 Adapter 或模型族不得自动删除旧方法参数。
3. 用户已编辑的字段随后因上下文变化而隐藏时，保存摘要必须单列“隐藏但已修改 N 项”。
4. 保存仍以完整 `baselineValues / draftValues / scopeKeys` 计算显式改动；呈现状态不得改变 diff 结果。隐藏但已编辑的项可以保留在 patch 中，但提交前必须让用户看见并可定位。
5. preflight 继续对最终合并配置做 fail-closed 校验，不能把“页面没显示”当作配置合法。
6. 配置切换、preset 切换和未保存离开确认继续保留。
7. “查看全部候选”使用同一 draft，不创建第二份表单状态；从审计入口修复后必须回写同一配置键。

这是实施中的最高优先级约束。

## 实现方向

### 规则单一事实源

当前 `config-field-availability.js` 已能表达部分禁用条件，但只返回 `enabled / reason / code`，且规则覆盖不完整。建议拆成：

- `config-field-disclosure-rules.js`：声明字段的 family、method、adapter、feature 和 dependency 规则。
- `config-field-availability.js`：评估规则并返回呈现状态，保留现有兼容导出。

推荐返回值：

```js
{
  visible: true,
  enabled: true,
  reason: '',
  code: null,
  controlledBy: [],
}
```

旧调用方迁移规则：

- `visible=false`：主画布不创建字段控件，但字段仍存在于 draft / baseline 状态。
- `visible=true, enabled=false`：沿用当前 disabled、原因帮助和键盘不可聚焦契约。
- 审计入口忽略 `visible=false` 的主视图过滤，但仍尊重 `enabled` 和 `reason`。
- 序列化不得从 DOM 可见控件反推全集；必须从配置状态和显式 dirty keys 生成 patch。

模型族能力继续来自 `library/models/family_registry.py` 对应的前端 capability 数据，不在规则文件中复制另一份模型支持矩阵。

这里的“运行时字段全集 `M`”严格定义为：指定配置与 preset 合并后，经过内部字段、退休字段、dataset blueprint 字段等静态排除，再去重得到的 Dragon 运行时配置块；`dataset_config` 工具卡按一个配置键计数。它不包含 425 项后端 schema 中未进入当前页面的候选，也不等同于 210 项静态布局目录。`215` 是该集合在冻结场景中的一次取值，不是类型定义或永久常量。

测试应分两层：静态 catalog 的所有键都具有 location / stage / cluster / order / policy；代表性 `model_family × method × adapter` 矩阵分别产生无重复的运行时字段集合。冻结的 215 项快照只是其中一个基线场景。

每个进入运行时字段集合的字段必须满足其一：

- 显式 `always`；
- 有一条披露规则；
- 被标记为 `unclassified` 并在审计入口可见。

这样新增字段不会因为默认高级等级而静默消失。

### 单一页面结构

目标状态不再保留“分组视图 / 积木流 Beta”两套主编辑模式：

- 四个阶段在同一文档流中。
- 顶部四个阶段导航只负责定位，当前阶段由滚动位置自动高亮。
- 二级字段簇使用稳定的 CSS Grid；不使用会改变阅读和 Tab 顺序的 Masonry。
- “查看全部候选”使用次级抽屉或独立审计状态。

迁移期间可以暂时保留旧分组视图，待字段集合和保存行为对账后再删除。

## 分阶段迁移

### 阶段 1：建立字段策略清单

状态：已完成首版 catalog 和完整性测试；真实披露规则求值留在阶段 2。

- 为静态 catalog 逐项声明 location、stage、cluster、依赖顺序和披露策略。
- 补齐 family / method / adapter / feature 上下文。
- 加入双层覆盖测试：冻结的 215 项快照继续作为历史回归；当前运行时 catalog 的 `M` 个字段（本轮观测为 218）必须恰好一次，三个 gradient-flow probe 必须正式归入资源诊断簇，而不是只记录漂移告警。
- 其他代表性矩阵同样无遗漏、无重复、无静默默认，并生成逐键 owner / 顺序快照。
- 暂不改变页面默认渲染。

### 阶段 2：引入三态呈现

状态：首批规则、主画布三态、plain-LoRA / 未知 family 门禁和旧分组视图一致性已实现；完整 family capability 矩阵待补齐。

- 在现有画布中应用 `visible / enabled / reason`。
- 保留旧等级控件作为临时回退。
- 首先覆盖 LoRA / LoKr / VeRA、Krea-2 / Z-Image plain LoRA、ConvRot、pipeline、compile、checkpoint。
- 同步改造 `config-page` 的字段创建、过滤、draft / baseline / scopeKeys、dirty 摘要和 `/api/config/raw` patch 链，再验证配置切换和保存 round-trip。

### 阶段 3：切换默认视图

状态：默认适用集、候选审计、隐藏搜索提示和隐藏 dirty 摘要已实现；隐藏非默认值摘要与 preflight 定位待实现。

- 默认只显示当前适用字段。
- 增加“查看全部候选”和隐藏非默认值摘要。
- 移除新人/初学者筛选对渲染的影响，但暂时忽略旧 localStorage 值以兼容升级。

### 阶段 4：统一四个连续阶段

状态：导航、画布阶段和响应式底座已提前完成；待动态披露后再完成错误 / 已修改阶段徽标和旧分组视图下线。

- 将五导航和八章节统一为 `输入准备 -> 方法配置 -> 训练计划 -> 资源与预检`。
- 阶段导航移除字段数和“定位 / 过滤”模式切换，仅保留定位、scrollspy 与错误 / 已修改状态。
- 完成搜索、过滤、空组、窄屏和键盘顺序验证。

### 阶段 5：清理旧系统

- 删除 `config-field-tiers.js` 及对应菜单、CSS 和测试。
- 删除 localStorage 中 `visibilityLevel` 的读写；保留同一对象里的 view、preset library、bilingual 等其他偏好。旧 `capsuleMode` 已随双模式导航删除并停止读取。
- 旧 `visibilityLevel` 值只忽略，不清空整个 `anima_dragon_config_ui`。
- 确认 Classic 是否继续使用自己的 `showAdvanced`；不得把两套实现误删为一套。

## 验收标准

- 冻结的 215 项快照作为回归 fixture 保留；当前运行时 catalog 的全部 `M` 项进入 owner、顺序和披露规则覆盖测试。新增字段在开发 / CI 中阻断静态覆盖测试，在运行时进入带警告的审计列表；dataset editor / internal / merged-only 字段不混入训练字段全集。
- `krea2_raw + lora` 不渲染 LoKr、VeRA、Soft Tokens、IP-Adapter、EasyControl、SPD 等不相干参数。
- Anima 选择 LoKr 后只追加 LoKr 的 9 个专用字段；切回 LoRA 后这些字段隐藏但值不丢失。
- Krea-2 和 Z-Image 的 family 限制与后端 registry 一致；未知 family 在前端禁用受能力约束字段，并由保存 / preflight fail closed。
- 开关、模式和父字段改变后，子字段即时披露或隐藏，顶部总量与阶段状态同步更新。
- 需要用户解除冲突的字段保持可见禁用，并显示明确原因。
- 普通浏览、搜索、章节定位和“查看全部候选”使用同一字段策略结果。
- 隐藏字段不会被保存流程静默清空；隐藏但已修改的字段在保存摘要中可见。
- 旧配置中的隐藏非默认值可从审计入口定位和修复；preflight 错误不会指向一个用户无法打开的字段。
- 切换配置文件、preset、模型族和 Adapter 后不残留上一个上下文的字段集合。
- 桌面和移动端不出现空章节、重排跳动、横向溢出或 DOM / Tab 顺序不一致。

## 不建议的方案

- **只把 8 章合成 5 章**：仍保留静态长表和另一套导航，根因没有解决。
- **默认隐藏所有 disabled 字段**：会把可解除冲突和真正无关字段混为一谈。
- **按新人/专家角色继续扩展等级**：字段风险取决于模型族和当前方法，不取决于用户身份。
- **真正的 Masonry 瀑布流**：容易造成视觉顺序与键盘顺序不一致，动态披露时布局跳动。
- **切换方法时清理旧字段**：存在配置数据丢失风险，违反配置编辑器的保留语义。

## 历史代码抓手与当前入口

以下旧 catalog/Dragon 模块文件名来自 Classic/Dragon 静态界面的实现快照，保留用于
解释原有分类、披露和兼容行为；新配置工作台的实施入口是
`web/frontend-next/src/features/training-config/`，不能将旧文件当作 Next 的当前事实源。

- 五分类（历史快照）：旧 `form-category-defs.js`
- 字段章节与顺序（历史快照）：旧 `form-layout.js`
- 八章节聚合（历史快照）：旧 `config-block-metadata.js`
- 静态等级（历史快照）：旧 `config-field-tiers.js`
- 动态可用性（历史快照）：旧 `config-field-availability.js`
- 页面上下文与过滤（历史快照）：旧 `config-page.js`
- 可见性偏好（历史快照）：旧 `config-ui-preferences.js`
- 当前 Next 字段目录与披露实现：`web/frontend-next/src/features/training-config/`
- 模型族能力：`library/models/family_registry.py`
