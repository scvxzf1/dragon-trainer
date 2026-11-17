# 优化路线图

本文从“重复、缺口、风险、自动化机会”四个角度整理未来优化方向。范围限定为显存、速度、稳定性、质量、易用性和配置复用。

状态：规划参考，执行前按实时实现复核。
表中的显存档位与方法组合以 Anima 经验为背景，是候选方向，不是所有模型族的推荐默认值或完成状态。
当前配置与能力边界见[优化配置事实清单](optimization-configs-current.md)和[多模型支持](multi_model_support.md)。

## 优先级约定

- `P0`：短期应优先做，主要是减少误用和维护成本。
- `P1`：中期建设项，需要小规模实现和验证闭环。
- `P2`：长期能力项，需要更多实验数据或产品化设计。

## 短期方向

| 方向 | 目标 | 当前痛点 | 可能方案 | 影响范围 | 难度 | 验证方式 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 统一配置命名和说明 | 让用户能区分 `low_vram`、`lora-8gb`、`balanced_16g`、`LoKr 16G`、`graft` | 硬件档、方法 variant 和快捷按钮命名来自不同层，含义容易混淆 | 在 `configs/presets.toml`、Next 配置界面与文档中统一术语：硬件 preset、方法 variant、资源快捷按钮分层说明 | 文档、Next 配置界面、配置目录 | 低 | 对照配置键与 guide；修改表单逻辑时按风险做类型或交互检查 | P0 |
| 清理重复/过期配置清单 | 降低维护文档与实时目录不一致风险 | 方法/变体列表散落在 `CLAUDE.md`、skill reference、Next guide | 以 `rg --files configs/methods configs/gui-methods` 为源生成或检查列表；文档只写“以目录实时列表为准” | 文档、测试辅助脚本 | 低 | 对照实时目录；仅在生成器或配置契约变化时选用相关用例 | P0 |
| 显式标注兼容边界 | 防止把互斥优化项一起打开 | block swap、Unsloth、CPU offload、selective checkpoint、Soft Tokens、functional loss 有硬边界 | 在 Next 字段帮助、preflight、文档矩阵中统一展示“不能同用” | Next preflight、训练启动、文档 | 中 | 文案对照兼容矩阵；规则变化时验收被改动的允许/拒绝契约 | P0 |
| 对齐表单默认值与合并值 | 让用户知道当前值来自 base、preset、method 还是表单默认 | Next 表单默认值与训练 merge chain 来源不总是直观 | 在表单字段旁显示来源：base/preset/method/runtime/用户改动；保存前展示 diff | Next config form、runtime config | 中 | 对照 `print-config` 与字段来源；合并或保存契约变化时定向验收 | P0 |
| 补齐关键 CLI-only 开关说明 | 让只在 CLI 存在的性能开关可被发现 | `dataloader_prefetch_factor`、`profile_steps`、`cpu_offload_checkpointing` 等未进入常用 UI | 文档列出 CLI-only；只把低风险字段加入高级区，危险/诊断字段留 CLI | 文档、WebUI advanced form | 低到中 | 对照 CLI/schema 定义；新增映射或校验时选用相关用例 | P1 |
| 建立“事实记录”模板 | 后续新增优化项能按同一格式记录 | 当前配置字段多，新增实验容易只写 findings 不回填清单 | 在 docs 中固定字段：名称、位置、作用、默认/候选、场景、风险、UI 暴露 | 文档维护流程 | 低 | 文档 review checklist | P0 |
| 强化 Next 快捷按钮提示 | 减少把实验按钮当默认训练方案 | `FP8 测试`、`OOM 兜底`、`LoKr 16G` 都是特定上下文按钮 | 快捷按钮 tooltip 加“适用方法/显存/是否实验”；应用按钮时显示将修改的字段 | Next config form | 中 | 提示文案与字段 diff 审阅；改变应用行为时定向检查提交值 | P1 |
| 最小化 profile 默认开销 | 让正式训练默认少写诊断日志 | 已将 `balanced_16g` 和正式训练快捷按钮的 block swap profile 改为 `off`；LoKr 快捷仍保留 memory probe 用于首跑显存确认 | 后续可再拆“诊断短跑”和“长期训练”两组快捷动作 | WebUI 快捷按钮、历史 artifact | 低 | 短跑检查 artifact 是否按预期生成 | P1 |

## 中期方向：显存档位矩阵

目标是建立 8GB / 12GB / 16GB / 24GB 的配置矩阵，并按方法 family 标注可用性、速度预期、已验证程度。

| 方向 | 目标 | 当前痛点 | 可能方案 | 影响范围 | 难度 | 验证方式 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8GB LoRA/T-LoRA 档 | 给 8GB 用户明确可跑 baseline | 当前有 `lora-8gb`、`tlora-8gb` 和 `low_vram`，但关系需更清晰 | 建立 `8gb_lora_safe` 文档档：`gradient_checkpointing=true`、`unsloth=true`、`blocks_to_swap=0`；明确速度代价 | GUI 变体、Web guide、docs | 中 | 8GB 或限制显存环境短跑；配置 merge 测试 | P1 |
| 12GB 过渡档 | 填补 8GB 和 16GB 之间的推荐空白 | 目前主要是 8GB 和 16GB 经验，12GB 用户只能手调 | 从 `low_vram_blockswap=8` 起步，测试 `blocks_to_swap=8/12`、`gradient_checkpointing` 组合 | presets、Web 快捷按钮 | 中 | 20-50 step 短跑，记录 reserved、sec/step、OOM | P1 |
| 16GB 普通 LoRA 档 | 固化 `balanced_16g` 为普通 LoRA 推荐档 | 已有 findings，但矩阵里未形式化 | 保留 `blocks_to_swap=12`；增加 `blocks_to_swap=16` 手动档；明确不代表 LoKr | presets、Web guide、docs | 低 | 复用已有测量；档位或性能声明变化时定向短跑 | P0 |
| 16GB LoKr 档 | 单独维护 LoKr 16GB 救场方案 | LoKr OOM 根因不同于普通 LoRA，不能套 `balanced_16g` | 建立 LoKr 行：`blocks_to_swap=23`、`lokr_factor_group_size=8`、allocator fallback、probe 策略 | Web 快捷按钮、LoKr docs、preflight | 中 | LoKr 10/50 step 短跑；`tests/test_lokr.py`、`test_block_swapping.py` | P0 |
| 24GB 快速档 | 给高显存用户避免误开保命开关 | 高显存用户可能仍沿用低显存配置导致变慢 | 建立 `gpu_full` 或 `fast_24g`：`blocks_to_swap=0`、`gradient_checkpointing=false`、`torch_compile=true`，按方法验证 batch/worker | presets、Web 快捷按钮 | 中 | 24GB 实卡 benchmark；样张和 loss 对比 | P1 |
| 方法兼容矩阵 | 明确每种方法可用的显存优化组合 | Soft Tokens、IP/Easy、Chimera、LoKr 与 block swap/checkpoint 兼容性不同 | 表格维度：方法 x block swap x full ckpt x selective ckpt x cache x compile | docs、preflight、config schema | 中 | 对照已改动的配置组合；允许/拒绝契约变化时选用 preflight 定向用例 | P1 |
| 短跑 benchmark harness | 让显存档验证可复现 | 当前 findings 多依赖手工短跑和临时目录 | 封装 10/50 step profile runner，统一输出 sec/step、reserved、allocated、profile p95、run status | scripts、docs/findings | 中到高 | harness 契约定向检查；性能结论另做 GPU 测量，需用户确认 | P1 |

## 中期方向：质量和速度实验闭环

| 方向 | 目标 | 当前痛点 | 可能方案 | 影响范围 | 难度 | 验证方式 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 训练记录分析 | 从历史任务自动提炼速度、显存、loss、OOM | `progress.jsonl`、`memory_probe.jsonl`、`block_swap_profile.jsonl` 已存在但分析分散 | 做 `scripts/analyze_training_runs.py` 或 WebUI 历史分析页，输出按配置聚合的指标 | scripts、WebUI history | 中 | 用现有 `configs/web-training-history` 只读样本跑解析测试 | P1 |
| 质量指标对比 | 把速度/显存与样张质量分开记录 | 优化项常只看能否跑和 sec/step，质量退化需人工看 | 对 CMMD、validation loss、固定 sample prompt 结果建立对比模板 | training loop、history、docs/findings | 中到高 | 固定 prompt + 固定 seed；CMMD/validation artifacts 对比 | P1 |
| 采样频率策略 | 避免 sample 过密拖慢训练 | `sample_every_*` 很容易被误设过小 | WebUI 根据数据集 step/epoch 估算建议采样间隔，提示额外耗时 | WebUI config estimator | 中 | frontend state + runtime config tests | P1 |
| checkpoint 策略提示 | 平衡磁盘、恢复能力和保存频率 | `save_every_n_epochs` 与 `checkpointing_epochs` 容易混淆 | 表单提示普通权重和可恢复状态区别；根据总 epoch 给建议 | WebUI field help、docs | 低 | 文案与保存语义对照；保存或恢复契约变化时定向验收 | P1 |
| optimizer/scheduler 组合模板 | 降低随意切换优化器导致训练不稳 | `optimizer_type` 候选多，但学习率/scheduler 搭配不明显 | 给 AdamW/CAME/AdamW8bit/Prodigy 系列建立最小模板和风险提示 | docs、Web guide、可选 preset | 中 | 短跑 loss 曲线对比；配置 schema choices 测试 | P2 |

## 长期方向

| 方向 | 目标 | 当前痛点 | 可能方案 | 影响范围 | 难度 | 验证方式 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 自动显存探测 | 启动前识别 GPU 显存、当前空闲、后台占用 | 用户需要手动判断选 8GB/16GB/LoKr 档 | WebUI 调用 GPU service/NVML，结合模型和方法估算推荐档 | WebUI backend、frontend、preflight | 高 | mock GPU 单测；实卡 dry-run；不启动长训 | P2 |
| 自动推荐配置 | 根据方法、显存、目标速度自动生成配置 | 目前用户在 preset、variant、快捷按钮之间手选 | 规则引擎：输入 GPU、方法、数据集尺寸、是否 LoKr/IP/Easy；输出字段 diff 和解释 | WebUI config form、runtime config | 高 | 推荐结果 snapshot tests；真实短跑验证 | P2 |
| 自动风险预检 | 在训练前拦截高风险组合 | 训练入口已有硬错误，但 WebUI 可以更早提示 | 将 `train.py` 兼容规则前移到 config preflight API：互斥、路径、缓存、显存档 | WebUI preflight、config schema | 中到高 | `tests/test_web_config_preflight.py`、`tests/test_web_preflight_compat_matrix.py` | P1 |
| 性能基准对比库 | 形成可查询的硬件/方法/配置基准 | findings 是文档型，难横向查询 | 保存 benchmark manifest：GPU、driver、method、preset、metrics、commit、dataset fingerprint | docs/findings、Web history、scripts | 高 | 基准 JSON schema 测试；重复 benchmark 方差检查 | P2 |
| 自适应训练记录分析 | 从历史任务自动发现 OOM、变慢、过拟合信号 | 用户需要手动读日志和样张 | 规则：OOM 阶段 -> 推荐 probe 或 fallback；sec/step 飙升 -> profile；validation/sample 退化 -> 降采样频率或调质量参数 | WebUI history、analysis service | 高 | 用历史任务 fixture 做规则单测；人工审核建议 | P2 |
| 自动缓存失效判断 | 减少 caption/图像/分桶变化后缓存错配 | 目前依赖用户理解何时重建缓存 | 缓存 sidecar 增加 fingerprint，preprocess/training 对比数据源和配置摘要 | preprocess、dataset、training bootstrap | 高 | 修改 caption/image 后应提示重建；旧缓存兼容测试 | P2 |
| 方法级 adapter 推荐 | 根据训练目标推荐 LoRA/LoKr/LoHa/VeRA/ReFT/Hydra | 用户面对方法很多，容易过早使用复杂方法 | 用 guide + 历史结果建立“先 LoRA，再进阶”的推荐流程 | WebUI guide、docs、可选 wizard | 中 | 文案对照方法能力；推荐逻辑变化时定向检查选择结果 | P2 |

## 建议实施顺序

1. `P0`：先完成命名统一、兼容边界、事实模板、16GB 普通 LoRA/LoKr 档说明。
2. `P1`：建立显存档矩阵和短跑 benchmark harness，把 WebUI preflight 与历史分析接起来。
3. `P2`：在有足够历史数据后，再做自动显存探测、自动推荐和性能基准库。

## 验证原则

- 范围与执行时机遵循 [AGENTS.md 验证策略](../AGENTS.md#验证策略)。表中的验证方式是针对候选行为的取证选择，不是每次改动都要执行的清单。
- 纯文档、文案和配置清单整理只做差异审阅及受影响链接/字段检查，不默认新增测试或运行配置测试。实际改变配置合并、schema 或兼容契约时，选择覆盖该契约的最小已有用例。
- Next 局部改动按风险选择类型检查或相关交互 case；`web-next-check`、完整 E2E 和构建用于对应风险的集中验收，不作为每次表单或文案修订的固定前置步骤。
- block swap、LoKr 等运行时改动先定位具体风险，再选择对应 case；保存/恢复、数值或模型族分派变化仍需必要定向验收，不因文件名或子系统自动扩大测试。
- 实现与验收可以分开推进，验收期间继续不依赖结果的工作；尚未取得证据时只报告实现状态。性能和数值结论必须有适用模型、硬件与配置下的测量证据。
- 需要执行的短定向测试使用 `timeout 60 .venv/bin/python -m pytest <相关文件> -k <相关case>`。真实 GPU 短跑或长跑会占用显卡，应单独确认后执行，不作为普通文档维护默认步骤。
