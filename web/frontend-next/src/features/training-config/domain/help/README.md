# 训练字段帮助维护

状态：当前 Next 工作台，2026-10-05 完成全量改写。

帮助正文统一使用 `FieldHelpV2`，由上一级 `field-help.js` 汇总。旧七段说明数据和 builder 已删除，`resolveFieldHelp.ts` 只处理键查找与 `qwen_sample_vae_decode` 别名。

## 内容分册

| 分册 | 条目数 | 内容 | 主要核对入口 |
| --- | ---: | --- | --- |
| [basics.js](basics.js) | 5 | 模型族、标注后缀、图片过滤、预览解码 | `library/models/family_registry.py`、`library/preprocess/captions.py`、`images.py`、`library/training/preview_decode_policy.py` |
| [input.js](input.js) | 23 | 数据集、验证切分、标注来源、日志、Qwen 任务 | `library/datasets/`、`library/preprocess/captions.py`、`web/services/config/dataset_edit_pairs.py` |
| [plan.js](plan.js) | 36 | 预算、批量、优化器、损失、预览和保存 | `library/training/bootstrap.py`、`schedulers.py`、`checkpoints.py`、各模型族 `training_preview.py` |
| [resources.js](resources.js) | 56 | 注意力、检查点、块交换、量化、诊断和预处理 | `library/training/compat_matrix.py`、`auto_block_swap/`、`gradient_flow_probe.py`、`library/runtime/offloading.py` |
| [runtime.js](runtime.js) | 37 | 编译、精度、OOM 重试、缓存、路径和格式 | `library/training/adaptive_runtime/`、`bootstrap.py`、`mask_loss.py`、`web/services/training/runtime_prepare.py` |
| [adapters.js](adapters.js) | 67 | 网络容量、专家路由和适配器变体 | `networks/lora_anima/config.py`、`routing_state.py`、`networks/plugins/`、`library/training/losses.py` |
| [advanced.js](advanced.js) | 58 | Chimera、对比目标、PE-LoRA、Soft Tokens、SPD | `networks/methods/ip_adapter.py`、`ip_adapter_pe_lora.py`、`soft_tokens.py`、`configs/methods/spd.toml` |

合计 282 项：保留原有 266 个帮助键，并补齐 Qwen 任务、正则化子集、三个梯度流探针参数、按步保存间隔、vocab pack 路径，以及 `nf4_dir1`、`krea2_attn_fp16` 两个 Krea-2 实验开关。
当前字段目录和表单显式字段均有帮助。旧键 `use_hydra`、`use_sigma_router`、`batch_size` 保留迁移说明。

源码核对说明实现行为，不等于新增真机验证。界面允许选择的组合、代码支持和已有硬件证据须分别表述。

## 新增或修订内容

- `version: 2`、简短 `summary` 和可操作的 `recommendation` 必填。
- 枚举写 `options`，回答什么时候选。数值参数解释调高、调低的影响，不给没有适用条件的通用最优值。
- 真正会踩坑的条件放在可见的 `cautions`，实现细节放在默认折叠的 `advanced`。
- `guidance` 只用于必要的解释，不设固定栏目。每个自然段最多两句话，不用分号串句。
- `relatedKeys` 必须能在总目录中解析。枚举值复用表单目录，弹窗按当前模型族的可选值过滤。
- 修改相应 `.js` 数据及类型声明。新增分册时加入总目录和现有目录测试的分册列表，避免重复键被覆盖。

完整帮助仅在打开弹窗时加载，参数行使用同源生成的轻量摘要。从仓库根目录执行：

```bash
node web/frontend-next/scripts/generate-field-help-summary.mjs
node web/frontend-next/scripts/generate-field-help-summary.mjs --check
```

不要在 `trainingForm.ts` 再保存一份相同的字段说明，也不要手改生成的摘要。
相关参数当前只显示名称与键，恢复页面默认和帮助内导航仍是独立的后续交互工作。

## 定向检查

```bash
pnpm --dir web/frontend-next typecheck
pnpm --dir web/frontend-next exec vitest run src/features/training-config/domain/field-help.test.ts src/features/training-config/TrainingFieldHelpDialog.test.tsx src/features/training-config/trainingForm.test.ts
```

目录检查覆盖 V2 结构、摘要一致性、现有表单覆盖、分册重复键、关联键及枚举值。
文案应核对实时源码，不用逐字快照代替内容审阅。浏览器验收使用隔离端口和只读合成数据，按前端 README 的流程执行。
