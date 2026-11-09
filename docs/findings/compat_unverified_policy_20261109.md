# 未验证类兼容拦截的降级策略（compat_unverified_policy）

状态：已实现（默认 warn，可 strict 回退 / allow 显式放开）
适用版本：2026-11-09，隔离工作树 `krea2-compat-relax`（分支 `codex/compat-unverified-relax`）
相关代码：`library/training/compat_matrix.py`、`library/training/cli_args.py`、
`library/training/metadata.py`、`web/services/config/preflight_compat.py`
关联文档：[逐条审计](compat_unverified_audit_20261109.md)、
[多模型支持](../multi_model_support.md)、
[多族混合 NF4](../experimental/multi-family-mixed-nf4.md)

## 1. 结论

`check_training_compat` 新增三级策略 `compat_unverified_policy`：

| 档位 | 语义 | 审计 C1（7） | 审计 C2（38） | 审计 C3（5） | A/B/B+C2（25） |
| --- | --- | --- | --- | --- | --- |
| `strict` | 与 origin/dev 逐条一致（全部 `error`） | error | error | error | error |
| `warn`（默认） | 只放行“实现会生效、仅缺真机验收”的组合 | **warning** | error | error | error |
| `allow` | 连“未接线 / 官方配方与显存策略”也放行 | warning | **warning** | **warning** | error |

- 取值优先级：`check_training_compat(..., unverified_policy=...)` 显式关键字 >
  配置键 `compat_unverified_policy`（CLI `--compat_unverified_policy` / TOML / WebUI 表单）
  > 默认 `warn`。非法取值产出 `invalid_compat_unverified_policy` 硬错误并回落到 `warn`。
- 只有**审计表内**的 code 会被降级；未知 code（含 `library/training/task_contracts.py`
  的 `invalid_training_task` / `unsupported_training_task` / `training_task_unavailable` /
  `qwen_image_2_1_edit_*`）在任何档位都保持 `error`，新增检查不会被默认放行。
- 降级后的 warning 文案带 `[unverified 未验证]` 标记；C1 追加逐条原因（为什么“实现会生效”），
  `allow` 档追加 `该键在本族未接线/会被忽略，或偏离官方配方/显存预算` 说明。
- `strict` 档消息保持 origin/dev 原文（不追加标记），便于逐条对照审计清单。

## 2. warn 档放行的 7 个 C1 code（严格限定，不得扩大）

| code | 依据 |
| --- | --- |
| `qwen_image_2_1_mixed_precision` | qwen torch/sdpa 路径无 fp16 守卫，只有 flash 要求 bf16；缺 qwen fp16 真机验收 |
| `qwen_image_2_1_masked_loss` | 共享掩码 flow-matching 损失适用于 qwen `[B,64,H,W]` 目标；缺数据/质量验收 |
| `qwen_image_2_1_loss_type` | `library/training/losses.py` 在 qwen 路径应用 l1/huber；官方配方只验收 L2 |
| `qwen_image_2_1_weighting_scheme` | `library/training/noise_target.py` 在 qwen 路径读取 `args.weighting_scheme` |
| `z_image_loss_type` | 同上（z_image 路径） |
| `z_image_flow_shift` | `library/models/z_image/family.py` 把 `args.discrete_flow_shift` 传给 `shifted_uniform_sigmas` |
| `z_image_weighting_scheme` | `library/training/noise_target.py` 在 z_image 路径读取 `args.weighting_scheme` |

## 3. 保持 error 的硬边界（A/B/B+C2，共 25 个 code）

- A（语义互斥 / 结构性冲突）：`selective_full_gradient_checkpointing`、`selective_cpu_offload`、
  `selective_unsloth_offload`（selective 先 disable 全部 block，bootstrap 随后又全量 enable）、
  `unsloth_cpu_offload`（`library/anima/models.py` 实现层互相清零）、
  `convrot_block_swap_int8_mutex`（与 `library/runtime/convrot/checks.py` 双重守卫，double dequant）、
  `block_swap_soft_tokens`、`block_swap_functional_loss`（多前向与 swap 单前向调度冲突）。
- B（输入校验）：`invalid_base_compute`、`invalid_selective_checkpoint`、`invalid_model_family`、
  `negative_blocks_to_swap`、`pipeline_parallel_config`、`krea2_invalid_attn_mode`、
  `z_image_attention_mode`、`z_image_block_swap_range`、`qwen_image_2_1_attn_mode`、
  `qwen_image_2_1_blocks_to_swap`、`invalid_qwen_saved_projection_blocks`、
  `invalid_qwen_projection_budget_mib`、`invalid_qwen_fused_projections`。
- B+C2 混合：`auto_block_swap_contract`、`adaptive_precision_contract`、
  `sample_vae_decode_contract`、`qwen_preview_memory_contract`、`qwen_saved_projection_contract`
  （同一 code 既承载非法区间也承载未实现项，拆 code 之前不放松）。

## 4. 删除的恒假死代码站点

| code | 为何恒假 | 处理 |
| --- | --- | --- |
| `krea2_plain_lora_only` | 站点位于 `if family_spec.plain_lora_only:` 内，而 `krea2_raw` 的 `plain_lora_only=False`（`library/models/family_registry.py`）；实测 krea2 + `use_lokr/use_ip_adapter/dora_wd` 均不报该 code | 删除 krea2 分支，统一产出 `family_plain_lora_only`；同时清理 `preflight_compat.py` 的孤立中文文案 |
| `nf4_convrot_mutex` | `convrot_active` 与 `nf4_active` 由同一个 `base_compute` 字符串派生，恒不同真；实测 `base_compute=nf4 + convrot_scope=all` 不报该 code | 删除站点与注释；NF4 与 ConvRot 的真实互斥由 `base_compute` 单值枚举本身保证 |

## 5. 同批修正的文案/白名单漂移

| 位置 | 修正 |
| --- | --- |
| `krea2_invalid_attn_mode` | 白名单补 `cudnn` / `cudnn_sdpa`（实现把三者归一为 torch），文案改为“torch 与 flash；sdpa/cudnn/cudnn_sdpa 是 torch 别名” |
| `z_image_attention_mode` / `qwen_image_2_1_attn_mode` | 白名单补 `native`（两族实现都归一为 torch），文案同步 |
| `z_image_bf16_only` | adaptive FP16/FP32 的 torch/sdpa 白名单同步接受 `native` |
| `invalid_base_compute` | 文案列出全部 9 个合法值（补 `fp16/none/off/w8a16/w8a8`） |
| `z_image_base_compute` / `qwen_image_2_1_base_compute` | 文案区分“运行时 NF4 未接入”与“预量化 mixed-NF4 可用（保持 `base_compute=bf16`）” |

`web/services/config/preflight_compat.py` 同时补齐 7 个 C1 code 与
`invalid_compat_unverified_policy` 的中文映射（含“未验证”与 strict 回退提示）。

## 6. 可观测性

- 训练运行身份：`--compat_unverified_policy` 进入 args，`library/training/metadata.py`
  以 `ss_compat_unverified_policy` 写入 checkpoint metadata，便于事后判断该次运行是否
  使用了未验证组合。
- CLI 训练入口（`library/training/extra_args.py`）把降级项作为 `logger.warning` 输出，
  错误仍以 `ValueError` 抛出；WebUI 预检把 `compat.warnings` 记为 warning 级提示。

## 7. 验证

- `tests/test_compat_unverified_policy.py`：
  - 审计表完整性与互斥（C1 7 / C2 38 / C3 5 / 硬边界 25 = 75，已扣除 2 个删除的死代码）；
  - 7 个 C1 的「默认 warn 放行 + strict 恢复 error」定向用例（含关键字与配置键两条路径）；
  - C2/C3 在 `warn` 仍为 error、`allow` 降级且文案含“该键在本族未接线/会被忽略”；
  - A/B/B+C2 在三个档位下全部保持 error；
  - `strict` 档对 35 组配置的错误 code 集合与改造前快照逐条相等（origin/dev 一致性）；
  - 4 处文案修正与 `native`/`cudnn` 别名、CLI 默认值与 metadata 落章。
- 定向回归：`tests/test_training_compat_matrix.py`、`tests/test_z_image_family.py`、
  `tests/test_model_family_registry.py`、`tests/test_krea2_attention_compat.py`、
  `tests/test_qwen_image_2_1_training_blocks.py`、`tests/test_qwen_image_2_1_family.py`、
  `tests/test_shared_preview_decode.py`、`tests/test_adaptive_training_precision.py`、
  `tests/test_pipeline_parallel_multimodel.py`、`tests/test_web_preflight_compat_matrix.py` 全绿。

## 8. 未验证项

- 本轮只做 CPU/纯逻辑验证：`warn` 档放行的 qwen fp16 / masked loss / 非 L2 损失 /
  非 none 权重方案、z_image 非 6.0 flow shift 等组合**仍缺真机数值与质量验收**，
  降级只代表“实现会生效”，不代表可接受。
- `allow` 档按设计会把 C2/C3 变成 warning；放行后可能在下游 raise 或被静默忽略，
  不应作为生产默认，也不替代后续把 C2 接线的实现工作。
