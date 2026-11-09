# 未验证类兼容拦截的降级策略（compat_unverified_policy）

状态：已实现（默认 warn，可 strict 回退 / allow 显式放开；第 2 轮按 t5 评审拆分 C2 子类）
适用版本：2026-11-09，隔离工作树 `krea2-compat-relax`（分支 `codex/compat-unverified-relax`）
相关代码：`library/training/compat_matrix.py`、`library/training/cli_args.py`、
`library/training/metadata.py`、`web/services/config/preflight_compat.py`
关联文档：[逐条审计](compat_unverified_audit_20261109.md)、
[多模型支持](../multi_model_support.md)、
[多族混合 NF4](../experimental/multi-family-mixed-nf4.md)、
[损失与 huber 调度](../../tests/test_huber_schedule.py)

## 1. 结论

`check_training_compat` 提供三级策略 `compat_unverified_policy`：

| 档位 | 语义 | C1（7） | C2-ignored（14） | C2-raise（24） | C3（5） | A/B/B+C2（25） |
| --- | --- | --- | --- | --- | --- | --- |
| `strict` | origin/dev 判定（见下方差异账） | error | error | error | error | error |
| `warn`（默认） | 只放行“实现会生效、仅缺真机验收”的组合 | **warning** | error | error | error | error |
| `allow` | 只额外放行“未接线/无效果”的开关 | warning | **warning** | error | error | error |

- **allow 只放行 C2-ignored**（族内从不读取该键，放行只是让开关变成 no-op）。凡是
  “放行即在下游 raise / TypeError / 静默 AttributeError”的 C2-raise、C3 与 A/B/B+C2
  code，在任何档位都保持 `error`——否则等于把明确预检错误推迟成更难定位的崩溃。
- 取值优先级：`check_training_compat(..., unverified_policy=...)` 显式关键字 >
  配置键 `compat_unverified_policy`（CLI `--compat_unverified_policy` / TOML）
  > 默认 `warn`。非法取值产出 `invalid_compat_unverified_policy` 硬错误并回落到 `warn`。
- 只有**审计表内**的 code 会被降级；未知 code（含 `library/training/task_contracts.py`
  的 `invalid_training_task` / `unsupported_training_task` / `training_task_unavailable` /
  `qwen_image_2_1_edit_*`）在任何档位都保持 `error`，新增检查不会被默认放行。
- 降级后的 warning 文案带 `[unverified 未验证]` 标记；C1 追加逐条原因（为什么“实现会生效”），
  `allow` 档追加“该键在本族未接线/会被忽略（放行不会改变训练行为，也不会在下游 raise）”。
- **strict 与 origin/dev 的差异账**（`git show origin/dev:library/training/compat_matrix.py`
  加载旧模块，对 35 组配置逐条比对 `(code, key)` 多重集）：
  1. §5 的 **6 处白名单/文案修正**，共导致 **12 个 code 的消息文本**变化（清单见 §5）：
     `krea2_invalid_attn_mode`、`z_image_attention_mode`、`qwen_image_2_1_attn_mode`、
     `invalid_base_compute`、`z_image_base_compute`、`qwen_image_2_1_base_compute`、
     `qwen_image_2_1_masked_loss`、`qwen_image_2_1_loss_type`、
     `qwen_image_2_1_weighting_scheme`、`z_image_loss_type`、`z_image_flow_shift`、
     `z_image_weighting_scheme`；
  2. **4 组白名单放宽**导致的判定变化（有意放宽，逐组见 §5.2）；
  3. 第 2 轮新增的 **A 类硬错误** `loss_type_huber_schedule_unsupported`（§2.2）；
  4. 第 2 轮把 qwen `mixed_precision` 拆出的两个子 code
     `qwen_image_2_1_mixed_precision_flash` / `qwen_image_2_1_mixed_precision_nf4`
     （原 code 在这两个组合下同样为 error，只是 code 名更具体，§2.3）。
  除以上四类，strict 与 origin/dev 的 code/key 多重集逐条一致。

## 2. warn 档放行的 7 个 C1 code（严格限定，不得扩大）

| code | 依据 |
| --- | --- |
| `qwen_image_2_1_mixed_precision` | qwen 非 flash、且未加载预量化 mixed-NF4 权重时该组合会生效；缺 qwen fp16 真机验收（子条件见 §2.3） |
| `qwen_image_2_1_masked_loss` | 共享掩码 flow-matching 损失适用于 qwen `[B,64,H,W]` 目标；缺数据/质量验收。**条件性生效**：数据集无任何掩码时掩码归约退化为普通均值（见 §8 F7） |
| `qwen_image_2_1_loss_type` | `library/training/losses.py` 在 qwen 路径应用 l1/huber；官方配方只验收 L2（`huber_schedule` 子条件见 §2.2） |
| `qwen_image_2_1_weighting_scheme` | `library/training/noise_target.py` 在 qwen 路径读取 `args.weighting_scheme` |
| `z_image_loss_type` | 同上（z_image 路径） |
| `z_image_flow_shift` | `library/models/z_image/family.py` 把 `args.discrete_flow_shift` 传给 `shifted_uniform_sigmas` |
| `z_image_weighting_scheme` | `library/training/noise_target.py` 在 z_image 路径读取 `args.weighting_scheme` |

### 2.1 C2-ignored（allow 可降级，14 个）

`qwen_image_2_1_base_compute`、`qwen_image_2_1_adaptive_precision`、
`qwen_image_2_1_selective_checkpoint`、`qwen_image_2_1_timestep_sampling`、
`qwen_image_2_1_layer_start`、`qwen_image_2_1_t_min`、`krea2_v100_flash_stability`、
`z_image_base_compute`、`z_image_selective_checkpoint`、`z_image_layer_range`、
`z_image_timestep_range`、`z_image_training_sampler`、`z_image_v100_flash_stability`、
`z_image_timestep_sampling`。

依据：qwen/z_image loader 从不读取 `base_compute` 量化、selective checkpoint 只在
anima/krea2 装配（`library/training/model_loading.py` 仅两处 `enable_selective_checkpointing`）、
`v100_flash_stability` 是 Anima-only 诊断、timestep/sampler/layer-range 键在该族训练路径
没有被消费。

### 2.2 huber 损失调度子条件（A 类硬错误，三档恒 error）

`loss_type ∈ {huber, smooth_l1}` 且 `huber_schedule == "snr"` 时，`library/training/losses.py`
的 `get_huber_threshold_if_needed` 需要 `noise_scheduler.alphas_cumprod`，而
`get_noise_scheduler` 恒返回 `FlowMatchEulerDiscreteScheduler`，第一步即
`NotImplementedError`。CLI 的 `--huber_schedule` 默认就是 `snr`，因此
`--loss_type huber` 单独出现即被 `loss_type_huber_schedule_unsupported` 拒绝；
只有 `constant` / `exponential` 才落到 C1 的 `*_loss_type` 降级。

### 2.3 qwen mixed_precision 子条件（三档恒 error）

- `fp16 + attn_mode=flash` → `qwen_image_2_1_mixed_precision_flash`：
  `library/models/qwen_image_2_1/attention_backend.py` 对非 bf16 直接 `RuntimeError`。
- `fp16 + 预量化 mixed-NF4 权重` → `qwen_image_2_1_mixed_precision_nf4`：
  `library/quantization/training.py` 要求 mixed-NF4 必须 BF16 compute。
  预量化权重通过 `pretrained_model_name_or_path` 的 safetensors manifest 识别
  （`library/quantization/format.py::inspect_mixed_nf4`，只读头部）。
- 其余 fp16（`attn_mode ∈ {torch, sdpa, native}` 且非 mixed-NF4）才按 C1 降级。

## 3. 保持 error 的硬边界

- A/B/B+C2（25 个，审计 A/B/mixed 全集）：
  - A：`selective_full_gradient_checkpointing`、`selective_cpu_offload`、
    `selective_unsloth_offload`（selective 先 disable 全部 block，bootstrap 随后又全量 enable）、
    `unsloth_cpu_offload`（`library/anima/models.py` 实现层互相清零）、
    `convrot_block_swap_int8_mutex`（与 `library/runtime/convrot/checks.py` 双重守卫）、
    `block_swap_soft_tokens`、`block_swap_functional_loss`（多前向与 swap 单前向调度冲突）。
  - B：`invalid_base_compute`、`invalid_selective_checkpoint`、`invalid_model_family`、
    `negative_blocks_to_swap`、`pipeline_parallel_config`、`krea2_invalid_attn_mode`、
    `z_image_attention_mode`、`z_image_block_swap_range`、`qwen_image_2_1_attn_mode`、
    `qwen_image_2_1_blocks_to_swap`、`invalid_qwen_saved_projection_blocks`、
    `invalid_qwen_projection_budget_mib`、`invalid_qwen_fused_projections`。
  - B+C2 混合（拆 code 之前不放松）：`auto_block_swap_contract`、
    `adaptive_precision_contract`、`sample_vae_decode_contract`、
    `qwen_preview_memory_contract`、`qwen_saved_projection_contract`。
- C2-raise（24 个，放行即下游 raise/TypeError/AttributeError，永不进 allow）：
  qwen `compile_backend` / `compile_mode` / `auto_block_swap` /
  `cpu_offload_checkpointing` / `network_train_unet_only` / `weighted_captions` /
  `caption_dropout_rate` / `preview_text_cache`，全部 8 个 `qwen_fused_projections_*`，
  `pipeline_parallel_runtime_unavailable`，krea2 `compile_inductor_mode` /
  `selective_checkpoint`，z_image `torch_compile` / `checkpoint_offload` /
  `weighted_captions` / `caption_dropout`，以及 `family_plain_lora_only`。
  其中 `family_plain_lora_only` 是 C1+C2 混合：method adapter 侧有实现层 raise
  （`library/training/adapter_resolver.py`），LoRA 变体侧才是 C1——因此
  **必须按冲突项拆 code 之后才允许部分降级**，本轮保持整体 error。
- C3（5 个，官方配方/显存策略或必然 raise）：`qwen_image_2_1_block_swap_checkpoint`
  （swap 无 checkpointing 直接吃满显存）、`qwen_nf4_compile_experiment`（策略模块拒绝）、
  `z_image_bf16_only`（flash+fp16 在 attention backend raise）、`block_swap_cpu_offload`、
  `block_swap_unsloth_offload`。`block_swap_cpu_offload` 属 **A/C2 且与族相关**：
  Anima 下是显存资源策略；非 Anima 族走 `bootstrap.py` 的
  `unet.enable_gradient_checkpointing(cpu_offload=True)` 会直接 `TypeError`（Diffusers
  transformer 不接受该关键字），因此不能进入 allow。
- 第 2 轮新增硬错误（3 个）：`loss_type_huber_schedule_unsupported`（§2.2）、
  `qwen_image_2_1_mixed_precision_flash` / `qwen_image_2_1_mixed_precision_nf4`（§2.3）。

## 4. 删除的恒假死代码站点

| code | 为何恒假 | 处理 |
| --- | --- | --- |
| `krea2_plain_lora_only` | 站点位于 `if family_spec.plain_lora_only:` 内，而 `krea2_raw` 的 `plain_lora_only=False`（`library/models/family_registry.py`）；实测 krea2 + `use_lokr/use_ip_adapter/dora_wd` 均不报该 code | 删除 krea2 分支，统一产出 `family_plain_lora_only`；同时清理 `preflight_compat.py` 的孤立中文文案 |
| `nf4_convrot_mutex` | `convrot_active` 与 `nf4_active` 由同一个 `base_compute` 字符串派生，恒不同真；实测 `base_compute=nf4 + convrot_scope=all` 不报该 code | 删除站点与注释；NF4 与 ConvRot 的真实互斥由 `base_compute` 单值枚举本身保证 |

删除后 `family_plain_lora_only` 成为唯一的 adapter 白名单 code；按 §3 的结论，它必须
先按冲突项拆分（method adapter 保留 error、LoRA 变体可 warn）才允许部分降级。

## 5. 同批修正的文案/白名单漂移

### 5.1 6 处白名单/文案修正（12 个 code 消息差异）

| 位置 | 修正 |
| --- | --- |
| `krea2_invalid_attn_mode` | 白名单补 `cudnn` / `cudnn_sdpa`（实现把三者归一为 torch），文案改为“torch 与 flash；sdpa/cudnn/cudnn_sdpa 是 torch 别名” |
| `z_image_attention_mode` | 白名单补 `native`（实现归一为 torch），文案同步 |
| `qwen_image_2_1_attn_mode` | 白名单补 `native`（实现归一为 torch），文案同步 |
| `z_image_bf16_only` | adaptive FP16/FP32 的 torch/sdpa 白名单同步接受 `native` |
| `invalid_base_compute` | 文案列出全部 9 个合法值（补 `fp16/none/off/w8a16/w8a8`） |
| `z_image_base_compute` / `qwen_image_2_1_base_compute` | 文案区分“运行时 NF4 未接入”与“预量化 mixed-NF4 可用（保持 `base_compute=bf16`）” |

消息文本实际变化的 12 个 code（origin/dev 对比）：
`krea2_invalid_attn_mode`、`z_image_attention_mode`、`qwen_image_2_1_attn_mode`、
`invalid_base_compute`、`z_image_base_compute`、`qwen_image_2_1_base_compute`（6 个白名单/文案）
+ `qwen_image_2_1_masked_loss`、`qwen_image_2_1_loss_type`、
`qwen_image_2_1_weighting_scheme`、`z_image_loss_type`、`z_image_flow_shift`、
`z_image_weighting_scheme`（6 个 C1，原先把已实现的能力写成 “not implemented” / “supports … only”）。

`web/services/config/preflight_compat.py` 同时补齐 7 个 C1 code、
`invalid_compat_unverified_policy` 与两个新 qwen 子 code 的中文映射（含“未验证”与 strict 回退提示）。

### 5.2 4 组“有意放宽”的白名单（strict 判定 vs origin/dev）

| # | 配置 | origin/dev | 当前 strict | 说明 |
| --- | --- | --- | --- | --- |
| 1 | `model_family=krea2_raw, attn_mode=cudnn`（`cudnn_sdpa` 同） | `krea2_invalid_attn_mode` | 通过 | 实现本来就把 `cudnn/cudnn_sdpa` 归一为 torch |
| 2 | `model_family=z_image, attn_mode=native` | `z_image_attention_mode` | 通过 | 实现把 `native` 归一为 torch |
| 3 | `model_family=qwen_image_2_1, attn_mode=native` | `qwen_image_2_1_attn_mode` | 通过 | 实现把 `native` 归一为 torch |
| 4 | `model_family=z_image, mixed_precision=fp16, attn_mode=native, adaptive_precision=fp16_fp32` | `z_image_attention_mode` + `z_image_bf16_only` + `adaptive_precision_contract` | 仅剩 `adaptive_precision_contract` | FP16 岛要求 torch/sdpa，故整体仍被拒；两个 z_image code 不再误报 |

`tests/test_compat_unverified_policy.py::test_strict_whitelist_relaxations_are_intentional`
把这 4 组配置在 strict 下的 code 集合钉死，防止文档与实现漂移。

## 6. 可观测性

- 训练运行身份：`--compat_unverified_policy` 进入 args，`library/training/metadata.py`
  以 `ss_compat_unverified_policy` 记录档位，并以 **`ss_compat_unverified_relaxed`**
  记录该次运行实际被降级的 code 列表（由
  `compat_matrix.apply_training_compat_mutations` 写回 args）。
- 启动汇总日志：`check_training_compat` 在存在降级项时打一条
  `compat_unverified_policy=<policy> relaxed: [codes]` 的 WARNING；训练 bootstrap
  路径（只对 errors 抛错）因此不会再静默丢掉降级记录，WebUI 预检也把
  `compat.warnings` 记为 warning 级提示。
- metadata 记录的是档位 + 实际降级清单；历史运行复盘时以
  `ss_compat_unverified_relaxed` 与启动日志为准。

## 7. 验证

- `tests/test_compat_unverified_policy.py`：
  - 审计表完整性与互斥（C1 7 / C2-ignored 14 / C2-raise 24 / C3 5 / A-B-mixed 25 = 75，
    已扣除 2 个删除的死代码；另有 3 个第 2 轮新增硬 code）；
  - 7 个 C1 的「默认 warn 放行 + strict 恢复 error」定向用例（关键字与配置键两条路径）；
  - C2-ignored 在 `warn` 仍 error、`allow` 降级且文案含“该键在本族未接线/会被忽略”
    与“不会在下游 raise”；
  - C2-raise / C3 / A/B/B+C2 / 第 2 轮新硬 code 在三个档位下全部保持 error；
  - huber+snr 三档 error、huber+constant 在 warn 下为 C1 warning（qwen 与 z_image）；
  - qwen fp16+flash、fp16+mixed-NF4 三档 error（NF4 分支用 monkeypatch 命中头部探测）；
  - `strict` 档对 35 组配置的错误 code 集合与 origin/dev 快照逐条相等；
  - 4 组白名单放宽、CLI 默认值、args/metadata 落章与启动汇总日志。
- 定向回归：`tests/test_training_compat_matrix.py`、`tests/test_z_image_family.py`、
  `tests/test_model_family_registry.py`、`tests/test_krea2_attention_compat.py`、
  `tests/test_qwen_image_2_1_training_blocks.py`、`tests/test_qwen_image_2_1_family.py`、
  `tests/test_shared_preview_decode.py`、`tests/test_adaptive_training_precision.py`、
  `tests/test_pipeline_parallel_multimodel.py`、`tests/test_web_preflight_compat_matrix.py` 全绿。

## 8. 未验证项与后续任务（follow-up）

- 本轮只做 CPU/纯逻辑验证：`warn` 放行的 qwen fp16（非 flash、非 mixed-NF4）、
  masked loss、非 L2 损失、非 none 权重方案、z_image 非 6.0 flow shift 等组合
  **仍缺真机数值与质量验收**，降级只代表“实现会生效”，不代表可接受。
- `allow` 只放行 C2-ignored；`C2-raise`/C3 依旧 error，也不替代后续把这些键真正接线的实现工作。
- **F7（deferred）**：`masked_loss` 生效但数据集无任何掩码时目前没有额外 warning；
  掩码归约会退化为普通均值（`library/training/mask_loss.py`）。后续应在
  `UNVERIFIED_C1_REASONS` 与 §2 的 masked_loss 行写清“条件性生效”，并补一条 warning。
- **F8（deferred，由 t6 承接）**：WebUI 暂无 `compat_unverified_policy` 表单字段；
  当前入口是 CLI `--compat_unverified_policy` 与 TOML 键，WebUI 沿用默认 `warn`。
  用户文档与 WebUI 入口由 t6 补齐，本文件不先行改写。
- **F9（deferred）**：`compat_matrix.py` 已因分类表增长到约 1300 行，超过
  AGENTS.md「已超 1000 行不再承载新业务」的软约束。下轮应把分类表 + 策略解析 +
  `_CompatBuilder` 的策略分支搬到 `library/training/compat_policy.py`，让
  `check_training_compat` 只保留 facade；本轮受 inScope（6 条真实路径）限制未拆。
