# 模型训练能力与编辑数据集

模型配置页的“普通文生图”“编辑数据集”“仅支持 LoRA”是后端派生的只读能力标签。
标签随模型族切换，不是可以自由填写的授权开关，也不通过模型文件名猜测能力。
当前 Qwen Image 2.1 支持普通文生图和编辑训练；其他现有模型族支持普通文生图。
这些标签表示本训练器已接入的训练能力，不代表该模型在所有推理工具中的能力。

训练配置中的“采样样张”能力单独由 `supported_preview_tasks` 和
`max_preview_references` 声明，不从训练标签推断。当前 Qwen Image 2.1 支持文生图和最多 4 张参考图的
编辑采样；单条样张中所有参考图按目标尺寸缩放后的总像素数不得超过 4 Mi。参考图在启动时复制为本次
运行的快照，正向/负向条件与参考 latent 在 DiT 加载前缓存。采样使用训练中当前 LoRA 权重，输出编辑
对比图并保留原始结果。跨运行的 Qwen 条件持久缓存和独立推理入口暂未实现，不能把“编辑采样”误解为
通用推理能力。整份样张文件中所有参考图缩放后的累计像素数不得超过 16 Mi（16777216 像素）；真实 GPU 上的多参考图采样尚未完成热态验证。

## 使用流程

1. 在模型配置页选择模型族和组件路径，可按训练能力筛选模型组合。
2. 数据集页以同一配对名称关联“编辑前”和“编辑后”子集，可保存多组配对。
   数据集编辑本身不要求当前选中某一模型。
3. 应用数据集时，后端检查目标模型能力和任务契约，再写入对应运行参数。
   未通过校验不会写入目标训练配置。
4. 训练配置预检查显示当前模型、支持任务、当前任务和数据集任务。
   外部 `dataset_config` 优先于内嵌数据集参与任务检查。
5. 启动与 CLI 数据集加载阶段仍执行共享兼容检查，不能通过手改标签绕过。

普通预处理流程允许数据集 TOML 尚未生成，此时显示警告和“尚未确认”，不假称已识别数据任务。
显式编辑训练的关联数据集缺失、无法解析或为空时会阻止启动。

## 兼容与边界

- 保留 `qwen_image_2_1_task = "t2i" / "edit"`，缺省仍为 `t2i`。
  发现参考图不会静默切换任务；应用编辑预设时会显式写入 `edit`。
- `qwen21` 和 `qwen_image_21` 别名在 Apply 与预检查入口统一规范化。
- Qwen 编辑训练继续要求条件/latent 缓存、batch size 1、完整配对和确定性变换；
  能力标签不解除 adapter、精度、正则或增强限制。
- 模型库 API 中 `training_tasks`、`capability_labels` 仅为响应字段，
  不写回模型库 TOML。旧配置不需要迁移，也不依赖模型库 ID 才能预检查。
- 模型识别以最终训练配置的 `model_family` 为准；本功能不新增权重内容鉴定。
- 编辑采样样张使用现有提示词文件的单行 JSON 记录。多参考图通过有序数组 `reference_images` 指定，
  最多 4 张；顺序会保留。旧字段 `reference_image`（单一路径）继续兼容，但同一条记录不能同时填写
  `reference_image` 和 `reference_images`。例如：
  `{"prompt":"给照片上色","sample_task":"edit","reference_images":["/data/ref-a.png","/data/ref-b.png"],"width":512,"height":512,"seed":42,"sample_steps":28,"guidance_scale":1}`。
  缩放到样张目标尺寸后，数组中参考图的总像素数最多为 4 Mi；超过会在启动预检查时拒绝。旧的纯文本提示词行继续有效；路径导入会复制到配置根的 `sample-references/`，不会改动数据集目录。
- 采样多参考图只影响训练过程中的编辑预览，不改变编辑训练数据集的配对契约：目标图仍与其训练参考图一对一配对。

## 兼容预检策略（`--compat_unverified_policy`）

训练启动前 `library/training/compat_matrix.py::check_training_compat` 会做跨族兼容预检；
WebUI 预检查、CLI 启动和 `bootstrap` 共用同一份判定。2026-11 起默认 `warn`：

| 档位 | 语义 | 放行范围 |
| --- | --- | --- |
| `strict` | 复现旧版判定，排障与回退用 | 全部 error（与旧版差异见策略文档 §1） |
| `warn`（默认） | 只放行“实现会生效、仅缺真机验收”的组合 | 审计 C1 的 7 个 code → warning |
| `allow` | 额外放行族内从未读取、放行即 no-op 的键 | 再加 14 个 C2-ignored → warning |

- 入口：CLI `--compat_unverified_policy strict|warn|allow`，或训练配置同名键；WebUI 表单
  尚未暴露该字段，WebUI 启动沿用默认 `warn`。
- 真矛盾、输入校验和“放行即下游 raise”的拦截在任何档位都是 error，`allow` 不会绕过。
- 每条降级 warning 带 `[unverified 未验证]` 与原因；启动日志另打一条
  `compat_unverified_policy=<policy> relaxed: [codes]` 汇总，便于确认本次实际放行了什么。
- checkpoint metadata 记录 `ss_compat_unverified_policy`（档位）与
  `ss_compat_unverified_relaxed`（本次实际降级的 code 列表），事后可判断该次运行是否用过
  未验证组合。
- 逐条依据见[审计](../findings/compat_unverified_audit_20261109.md)与
  [策略文档](../findings/compat_unverified_policy_20261109.md)（含 strict 差异账与已知后续）。

## 实现入口

- `library/models/family_registry.py`：`supported_tasks` 及旧任务键绑定。
- `library/training/task_contracts.py`：任务投影、能力门禁、模型专属编辑契约。
- `library/training/compat_matrix.py`：共享训练兼容入口。
- `web/services/config/preflight_tasks.py`：读取实际数据集及任务摘要。
- `GET /api/config/model-families`：前端只读能力来源。

新增编辑模型必须同时提供训练实现、能力声明、任务键和编辑契约；仅增加标签不会放行。
定向回归入口为 `tests/test_training_task_capabilities.py`、
`tests/test_web_qwen_image_edit_dataset.py`、`tests/test_model_config_service.py`，
前端 `ModelCapabilities.test.tsx`、`editDataset.test.ts` 和
`e2e/model-task-capabilities.spec.ts`。浏览器测试使用隔离 API fixture，不启动真实训练。
