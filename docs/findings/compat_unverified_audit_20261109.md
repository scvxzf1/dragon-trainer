# 未验证类兼容拦截逐条审计（compat_matrix 50 处 out.error + 7 处 C1 unverified）

状态：审计（只读）；实现状态见[降级策略](compat_unverified_policy_20261109.md)（两个恒假死代码站点已在 `codex/compat-unverified-relax` 删除）。第 2 轮按 t5 评审补记：本表 `block_swap_cpu_offload` 改判 **A/C2 且与族相关**、`family_plain_lora_only` 标注“必须拆 code 后才允许部分降级”；另新增 3 条审计外硬错误（`loss_type_huber_schedule_unsupported`、`qwen_image_2_1_mixed_precision_flash`、`qwen_image_2_1_mixed_precision_nf4`），详见策略文档 §2.2/§2.3。
适用版本：2026-11-09，隔离工作树 `krea2-compat-relax`（分支 `codex/compat-unverified-relax`）
相关代码：`library/training/compat_matrix.py`（约 1364 行；`out.error` 50 个站点、C1 `out.unverified` 7 个站点，另有第 2 轮新增的 3 条硬 code）、
`library/models/family_registry.py`、`library/models/{krea2_raw,z_image,qwen_image_2_1}/`、
`library/training/{task_contracts,auto_block_swap,adaptive_runtime,pp}`、
`library/quantization/`、`library/runtime/convrot/`、`web/services/config/preflight_compat.py`
关联文档：[多模型支持现状](../multi_model_support.md)、
[各族运行时键约束](../configuration/auto-block-swap.md)、
[多族混合 NF4](../experimental/multi-family-mixed-nf4.md)

## 0. 审计目的与口径

本审计为「默认 warn、可 strict 回退」策略提供逐条依据：**只有“代码路径真实存在、
只是未被验证”的拦截才允许降级为 warning**。凡是放行后会在下游 raise、或会让用户
开关被静默忽略、或两个开关语义互斥的站点，都必须继续 `error`，否则放松等于把
明确报错换成更难定位的隐性失败。

分类口径：

| 分类 | 含义 | 建议动作 |
| --- | --- | --- |
| **A** | 真矛盾 / 结构性冲突 / 必然崩：两个开关语义互斥，或放行后运行时必 raise | 保留 error |
| **B** | 输入校验：非法枚举、越界范围、类型错误 | 保留 error |
| **C1** | 仅未验证：实现路径存在且**会真正生效**，只是缺少真机验收 | **可降级 warning**（strict 下恢复 error） |
| **C2** | 未实现：族内没有对应代码路径；放行后要么下游 raise，要么开关被静默忽略 | 保留 error |
| **C3** | 官方配方 / 资源策略限制：实现上可跑，但偏离官方配方或会突破显存预算 | 保留 error（如需放开应走显式实验开关，不走默认 warn） |

另外两类不计入 A/B/C，但必须记录：

- **死代码**：条件在现有能力表下恒假，站点永不触发。
- **重复门**：compat 与实现层各有一份同义校验（放行后仍有运行时守卫，属“安全降级”，
  但会让报错从 preflight 推迟到模型加载）。

统计（审计时点 2026-11-09 快照：文件 984 行、52 个 `out.error` 站点，经模板展开后为
**79 个 code 位、77 个不同 code**；`qwen_nf4_compile_experiment` 在 L249/L251 各写一次，
`pipeline_parallel_config` 在 L345/L354 各写一次）。实现阶段删除 2 个恒假站点后为
50 个 `out.error` + 7 个 C1 `out.unverified` 站点，并按真机结果新增 3 条硬 code（§12-c）。
按不同 code 计：

| 分类 | code 数 | 说明 |
| --- | --- | --- |
| A | 7 | 语义互斥 / 结构性冲突 |
| B | 13 | 纯输入校验 |
| B + C2 混合（同一 code 承载两类） | 5 | `auto_block_swap_contract`、`adaptive_precision_contract`、`sample_vae_decode_contract`、`qwen_preview_memory_contract`、`qwen_saved_projection_contract` |
| C1（建议降级 warning） | 7 | 见下 |
| C2 | 38 | 未实现 |
| C3 | 5 | 官方配方 / 资源策略 |
| 死代码 | 2 | `krea2_plain_lora_only`、`nf4_convrot_mutex` |
| 合计 | 77 | — |

**建议降级为 warning 的 7 个 code**：
`qwen_image_2_1_mixed_precision`、`qwen_image_2_1_masked_loss`、`qwen_image_2_1_loss_type`、
`qwen_image_2_1_weighting_scheme`、`z_image_loss_type`、`z_image_flow_shift`、
`z_image_weighting_scheme`。

其余 70 个 code 建议在默认档继续 `error`；其中 2 个（`krea2_plain_lora_only`、`nf4_convrot_mutex`）
是恒假死代码，建议直接删除或改由能力表驱动，而不是“降级”。

## 1. Qwen Image 2.1 契约（站点 L185，模板展开 21 个 code）

站点：`compat_matrix.py:183-235` 的 `forbid()` → `out.error(f"qwen_image_2_1_{flag}", ...)`。
族门：`family_spec.name == "qwen_image_2_1"`（`family_registry.py:183-209`，`plain_lora_only=True`）。
关联测试：`tests/test_qwen_image_2_1_family.py:236-248`
（参数化 8 条只断言 `check_training_compat(config).errors` 非空，**不断言具体 code**）。

| code | 所属族 | 分类 | 判定依据 | 建议动作 | 关联测试 |
| --- | --- | --- | --- | --- | --- |
| `qwen_image_2_1_mixed_precision` | qwen_image_2_1 | C1 未验证 | compat L189；实现侧无 fp16 守卫，唯一硬约束是 `qwen_image_2_1/attention_backend.py:254`（flash 只验证 bf16），torch/sdpa 下 fp16 会真正生效；文档口径见 [多模型支持](../multi_model_support.md)「当前仅允许 BF16」 | **降级 warning** | `test_qwen_image_2_1_family.py:236-248`（仅断言有 error） |
| `qwen_image_2_1_base_compute` | qwen_image_2_1 | C2 未实现 | compat L191；`base_compute=nf4` 的运行时量化对 qwen 不存在，NF4 只能走预量化 mixed-NF4 + `base_compute=bf16`（`library/quantization/training.py:18-20`，[多族混合 NF4](../experimental/multi-family-mixed-nf4.md)「Z-Image/Qwen 的该配置值仍拒绝」） | 保留 error（文案应改为“运行时 NF4 未接入；预量化 mixed-NF4 请保持 base_compute=bf16”） | 同上 |
| `qwen_image_2_1_adaptive_precision` | qwen_image_2_1 | C2 未实现 | compat L193；`ADAPTIVE_FAMILIES` 不含 qwen（`library/training/adaptive_runtime/training_config.py:85-86`） | 保留 error | `tests/test_adaptive_training_precision.py`（族集合） |
| `qwen_image_2_1_attn_mode` | qwen_image_2_1 | B 枚举校验 | compat L196 白名单 `{torch,sdpa,flash}` + xformers 互斥；实现 `library/models/qwen_image_2_1/attention_backend.py:24-32` 另收 `native`→torch | 保留 error；**补 `native` 别名或改文案**（见 §8-2） | `tests/test_qwen_image_2_1_attention.py` |
| `qwen_image_2_1_blocks_to_swap` | qwen_image_2_1 | B 范围校验 | compat L198（>30 报错）；实现 `block_swap.py:36-42` 为 `0 <= n <= len(blocks)-2`，本族真实 32 块 → 上界 30（[各族分块载荷](block_swap_family_payload_50hx_20261019.md) 第 12 轮更正） | 保留 error | `tests/test_qwen_image_2_1_block_swap.py`、`tests/test_qwen_image_2_1_family.py:238` |
| `qwen_image_2_1_block_swap_checkpoint` | qwen_image_2_1 | C3 资源策略 | compat L200；`block_swap.py:20-68` 构造时不校验 checkpointing，属“不重算必爆激活”的显存策略，非结构性不可能 | 保留 error（如需放开应作为显式实验开关） | `test_qwen_image_2_1_family.py:240` |
| `qwen_image_2_1_compile_backend` | qwen_image_2_1 | C2 未实现 | compat L202；`library/models/qwen_image_2_1/compile.py:26-27` 对 `cudagraphs` 直接 `raise ValueError` | 保留 error（重复门） | `test_qwen_image_2_1_family.py:241` |
| `qwen_image_2_1_compile_mode` | qwen_image_2_1 | C2 未实现 | compat L204；`compile.py:26-27` 只接受 `None/default` | 保留 error（重复门） | 同上 |
| `qwen_image_2_1_auto_block_swap` | qwen_image_2_1 | C2 未实现 | compat L206；AUTO 载荷表只覆盖 `anima`/`krea2_raw`（`library/training/auto_block_swap/capacity.py:62-63`） | 保留 error | `tests/test_auto_block_swap_config.py` |
| `qwen_image_2_1_selective_checkpoint` | qwen_image_2_1 | C2 未实现 | compat L208；qwen 族内无 selective checkpoint 实现（`grep selective library/models/qwen_image_2_1/` 仅命中 fused-projections 的 `!= off` 断言） | 保留 error | — |
| `qwen_image_2_1_cpu_offload_checkpointing` | qwen_image_2_1 | C2 未实现 | compat L210；qwen 族内无 activation offload 接线 | 保留 error | — |
| `qwen_image_2_1_network_train_unet_only` | qwen_image_2_1 | C2 未实现 | compat L212；`library/training/extra_args.py:54` 在缓存 TE 输出时断言不能训练 TE | 保留 error | — |
| `qwen_image_2_1_weighted_captions` | qwen_image_2_1 | C2 未实现 | compat L214；`library/training/batch_step.py:207-219` 的加权路径需 `encode_tokens_with_weights`，qwen strategy 只有 `tokenize_with_weights`（`strategy.py:109`），无对应编码器 → AttributeError / 静默忽略 | 保留 error | — |
| `qwen_image_2_1_caption_dropout_rate` | qwen_image_2_1 | C2 未实现 | compat L216；双重守卫：`library/models/qwen_image_2_1/strategy.py:251-253` 与 `family.py:131` 均 raise | 保留 error（重复门） | `test_qwen_image_2_1_family.py:242` |
| `qwen_image_2_1_masked_loss` | qwen_image_2_1 | C1 未验证 | compat L219；通用掩码损失已实现（`library/training/losses.py:310-330` `_flow_match_loss` → `_reduce_loss_with_mask`），qwen 返回 `[B,64,H,W]`（`family.py:119`）可走同一空间归约；缺的是 qwen 数据/质量验收，`docs/configuration/auto-block-swap.md:256` 只记录“必须关闭”而无否决证据 | **降级 warning** | — |
| `qwen_image_2_1_loss_type` | qwen_image_2_1 | C1 未验证 | compat L221；`_conditional_loss(pred, target, ctx.args.loss_type, ...)`（`losses.py:316-321`）支持 l1/huber 等，会真正生效；官方配方只用 L2 | **降级 warning** | — |
| `qwen_image_2_1_timestep_sampling` | qwen_image_2_1 | C2 未实现（开关被忽略） | compat L223；qwen 用固定动态 shift 采样 `family.py:12-17`，从不读 `args.timestep_sampling`；该键只被 anima 路径 `library/runtime/noise.py:110-130` 使用 | 保留 error（或降级但文案必须写“该键对 Qwen 无效”） | — |
| `qwen_image_2_1_weighting_scheme` | qwen_image_2_1 | C1 未验证 | compat L225；`library/training/noise_target.py:420-427` 用 `args.weighting_scheme` 计算权重，会真正生效 | **降级 warning** | — |
| `qwen_image_2_1_preview_text_cache` | qwen_image_2_1 | C2 未实现 | compat L230；`library/models/qwen_image_2_1/training_preview.py:50-52` 在 conditions 未缓存时 `raise ValueError("Qwen preview conditions were not cached before DiT loading")` | 保留 error（重复门） | `tests/test_qwen_image_2_1_preview_kv.py` |
| `qwen_image_2_1_layer_start` | qwen_image_2_1 | C2 未实现 | compat L232；qwen 无 layer range 目标层选择实现 | 保留 error | — |
| `qwen_image_2_1_t_min` | qwen_image_2_1 | C2 未实现 | compat L234；`family.py:12-17` 固定全区间 sigmas | 保留 error | — |

## 2. Qwen 融合/保存投影（站点 L308、L310、L314、L316、L319、L321、L331）

实现集中在 `library/models/qwen_image_2_1/training_blocks.py` 与 `segmented_training.py`，
compat 与实现层是**同义双门**（放行后仍有运行时 raise）。

| code | 所属族 | 分类 | 判定依据 | 建议动作 | 关联测试 |
| --- | --- | --- | --- | --- | --- |
| `invalid_qwen_saved_projection_blocks`（L308） | qwen_image_2_1 | B 输入校验 | compat L307-308；`training_blocks.py:20-24,59`、`segmented_training.py:101` 同义 raise | 保留 error | `tests/test_qwen_image_2_1_training_blocks.py:112-115`（测实现层，未断言 code） |
| `invalid_qwen_projection_budget_mib`（L310） | qwen_image_2_1 | B 输入校验 | compat L309-310；`segmented_training.py:103` | 保留 error | 同上 |
| `qwen_saved_projection_contract`（L314） | qwen_image_2_1 | B/C2 | compat L311-314（要求 `mode=all` + 正整数 + MiB 预算）；`training_blocks.py:22-24,59` 为硬前提 | 保留 error | 同上 |
| `invalid_qwen_fused_projections`（L316） | qwen_image_2_1 | B 枚举校验 | compat L315-316；`training_blocks.py` 只接受 off/mlp/qkv/all | 保留 error | `test_qwen_image_2_1_training_blocks.py:157` |
| `qwen_fused_projections_family`（L319） | qwen_image_2_1 | C2 未实现（其他族） | compat L317-319；投影打包只在 qwen 族实现 | 保留 error | `test_qwen_image_2_1_training_blocks.py:152` |
| `qwen_fused_projections_plain_lora`（L321） | qwen_image_2_1 | C2 未实现 | compat L320-321；`training_blocks.py:44` raise（只接受 `networks.lora_anima`） | 保留 error（重复门） | `tests/test_training_compat_matrix.py`（`qwen_fused_projections_plain_lora`） |
| `qwen_fused_projections_mixed_precision`（L331） | qwen_image_2_1 | C2 未实现 | compat L322-331；`training_blocks.py:32` raise | 保留 error | — |
| `qwen_fused_projections_base_compute`（L331） | qwen_image_2_1 | C2 未实现 | 同上（`training_blocks.py:34`） | 保留 error | — |
| `qwen_fused_projections_block_swap_transfer_dtype`（L331） | qwen_image_2_1 | C2 未实现 | `training_blocks.py:36` raise | 保留 error | `test_qwen_image_2_1_training_blocks.py:154` |
| `qwen_fused_projections_gradient_checkpointing`（L331） | qwen_image_2_1 | C2 未实现 | `training_blocks.py:40` raise | 保留 error | — |
| `qwen_fused_projections_selective_checkpoint`（L331） | qwen_image_2_1 | C2 未实现 | `training_blocks.py:38` raise | 保留 error | — |
| `qwen_fused_projections_cpu_offload_checkpointing`（L331） | qwen_image_2_1 | C2 未实现 | `training_blocks.py:42` raise | 保留 error | — |

## 3. 委托型契约站点（L249、L251、L257、L261、L291、L303、L345、L354、L370）

这些站点本身只是把下游模块的 `configuration_errors()` 结果包成统一 code，
**判定必须看被委托模块**。

| code | 所属族 | 分类 | 判定依据 | 建议动作 | 关联测试 |
| --- | --- | --- | --- | --- | --- |
| `qwen_nf4_compile_experiment`（L249/L251） | qwen_image_2_1 | C3 实验配方 | `library/quantization/qwen_nf4_compile_policy.py:20-73`：整条实验配方被钉死（compile/attn/blocks/rank/task 等 11 项 + 禁用 20 余项），L250-251 另加单进程要求 | 保留 error | `tests/test_qwen_nf4_compile_experiment.py` |
| `auto_block_swap_contract`（L257） | 通用 | B + C2 混合 | `library/training/auto_block_swap/config.py:62-160`：非法区间为 B；`RAM/dynamic 仅 Krea-2`、`单进程`、`max_data_loader_n_workers=0` 等为 C2/C3 | 保留 error | `tests/test_auto_block_swap_config.py`（测实现层，未断言 code） |
| `adaptive_precision_contract`（L261） | 通用 | B + C2 混合 | `library/training/adaptive_runtime/training_config.py:64-137`：模式/数值校验为 B；`ADAPTIVE_FAMILIES = ("anima","krea2_raw","z_image")`（`:10`，qwen/moss 不在内 → C2）、`forbidden` 列表为 C2 | 保留 error | `tests/test_adaptive_training_precision.py`、`test_adaptive_training_supervisor.py`、`test_multigpu_admission.py`、`test_native_startup_oom.py` |
| `invalid_model_family`（L291） | 通用 | B 输入校验 | `family_registry.py` 的 `normalize_registered_family`（fail-closed，未知族/别名失败） | 保留 error | `tests/test_pipeline_parallel_multimodel.py`、`test_model_config_service.py`、`test_training_task_capabilities.py` |
| `sample_vae_decode_contract` / `qwen_preview_memory_contract`（L303） | qwen_image_2_1 | B + C2 | `library/training/preview_decode_config.py:19-22`（枚举校验，B）；`library/models/qwen_image_2_1/preview_memory.py:20-44`（非 qwen 族使用 Qwen 预览覆盖 → C2；custom swap 需 BF16 无损 → C3） | 保留 error | `tests/test_shared_preview_decode.py`、`tests/test_qwen_preview_memory.py` |
| `pipeline_parallel_config`（L345/L354） | 通用 | B 输入校验 | `library/training/pp/gate.py` 的 `validate_pipeline_parallel_config`；键固定为 `pipeline_parallel` 以便 WebUI 续训门复用 | 保留 error | `tests/test_pipeline_parallel_multimodel.py`、`test_anima_pp_wiring.py`、`test_krea2_pipeline_parallel.py` |
| `pipeline_parallel_runtime_unavailable`（L370，经 `issue.code`） | 通用 | C2 未实现 | `pp/gate.py:76-83,185-197`；`family_registry.py:119` 只有 anima 发布 `runtime_backend`，其余族（含已声明 PP 规划的 krea2/z_image）在运行期 fail closed | 保留 error | `tests/test_pipeline_parallel_multimodel.py` |

补充：`compat_matrix.py:293-295` 调用 `check_training_task(out, config, family_spec)`，
会额外产出 `invalid_training_task`、`unsupported_training_task`、`training_task_unavailable`、
`qwen_image_2_1_edit_*`（`library/training/task_contracts.py:58-166`）。这些**不在 52 个站点内**，
但同属“兼容拦截面”，实现为 B（枚举/数据契约）+ C2（未接线的编辑任务契约），建议同步纳入
strict 名单；本轮不动。

## 4. 通用选择性检查点与适配器白名单（L407、L466）

| code | 所属族 | 分类 | 判定依据 | 建议动作 | 关联测试 |
| --- | --- | --- | --- | --- | --- |
| `invalid_selective_checkpoint`（L407） | 通用 | B 枚举校验 | compat L406-412 白名单 8 项；`library/anima/models.py:2081-2095` 运行时同义 raise | 保留 error（重复门） | — （无直接断言） |
| `krea2_plain_lora_only`（L466） | krea2_raw | **死代码** | 站点位于 `if family_spec.plain_lora_only:`（L414）内，而 `krea2_raw` 的 `plain_lora_only=False`、`supported_network_specs=None`（`family_registry.py:132-134`）；[多模型支持](../multi_model_support.md) 明确“不再套用非 Anima 仅 plain LoRA 白名单”。实测 `use_lokr/use_ip_adapter/dora_wd` 在 krea2 下均不报错 | 保留 error（**等价 no-op**）；建议删除该分支或改由能力表驱动，并清理 `web/services/config/preflight_compat.py:100` 的孤立文案 | `tests/test_training_compat_matrix.py:267,368`（反向断言“不出现”） |
| `family_plain_lora_only`（L466） | qwen_image_2_1 / moss_image | C1+C2 混合 | `family_registry.py:195,222`；qwen 放行 `use_lokr/use_loha`（L430），其余 LoRA 变体与 method adapter 被拒。method adapter 侧另有运行时门（`library/training/adapter_resolver.py:41-50`，`supports_method_adapters=False`）→ C2；LoRA 变体侧无实现门 → C1 | 保留 error；**必须拆 code 后才允许部分降级**：method adapter 冲突保留 error，LoRA 变体子项才可挪进 warn 白名单 | `tests/test_model_family_registry.py`、`tests/test_training_compat_matrix.py:270-336` |

## 5. Krea-2 族门（L492、L534、L541、L548）

| code | 所属族 | 分类 | 判定依据 | 建议动作 | 关联测试 |
| --- | --- | --- | --- | --- | --- |
| `krea2_invalid_attn_mode`（L492） | krea2_raw | B 枚举校验 | compat L491 白名单 `KREA2_ATTN_MODES = {"", torch, flash, sdpa}`（L33）；实现 `library/models/krea2_raw/attention_backend.py:17-26` 把 `sdpa/cudnn/cudnn_sdpa` 归一为 `torch`。**白名单与实现不同步**（见 §8-1/§8-2） | 保留 error；补 `cudnn`/`cudnn_sdpa` 别名或明确拒绝并改文案 | `tests/test_krea2_attention_compat.py:11-30` |
| `krea2_compile_inductor_mode`（L534） | krea2_raw | C2 未实现 | compat L530-539；`library/models/krea2_raw/dit.py:580-586` raise（只接受 `None/default`）；默认档证据见 [krea2 速度阶段总结](krea2_3080_speed_stage_summary.md) | 保留 error（重复门） | `tests/test_krea2_attention_compat.py:33-51` |
| `krea2_selective_checkpoint`（L541） | krea2_raw | C2 未实现 | compat L540-546；`dit.py:526-546` 只实现整块 checkpoint，`every_other` 是唯一同义模式并直接 raise 其他值 | 保留 error（重复门） | `tests/test_krea2_attention_compat.py:33-51`、`tests/test_krea2_selective_checkpoint.py` |
| `krea2_v100_flash_stability`（L548） | krea2_raw | C2 未实现 | compat L547-552；该开关是 Anima-only 的 Flash 诊断路径（[V100 Flash 支持](v100_flash_attention_support.md)），krea2 无对应实现 | 保留 error | `tests/test_krea2_attention_compat.py:33-51` |

## 6. Z-Image 族门（L575–L702，共 17 个 code）

| code | 所属族 | 分类 | 判定依据 | 建议动作 | 关联测试 |
| --- | --- | --- | --- | --- | --- |
| `z_image_attention_mode`（L575） | z_image | B 枚举校验 | compat L571-580 白名单 `{torch,flash,sdpa}` + xformers 互斥；实现 `library/models/z_image/attention_backend.py:16-23` 另收 `native`→torch（**白名单漂移**，§8-2） | 保留 error；补 `native` 或改文案 | `tests/test_z_image_family.py:338-401`、`tests/test_z_image_attention_backend.py` |
| `z_image_bf16_only`（L590） | z_image | C3 配方（含 A 子条件） | compat L581-595；`attention_backend.py:56-60`：`flash` 硬性要求 bf16（A）；adaptive islands 下 fp16/fp32 允许走 torch/sdpa | 保留 error | `tests/test_z_image_family.py:366`、`tests/test_adaptive_training_precision.py` |
| `z_image_base_compute`（L600） | z_image | C2 未实现 | compat L596-604；z_image **没有** `base_compute=nf4` 运行时量化，但支持**预量化** mixed-NF4 权重（`library/models/z_image/weights.py:238-251`，需 bf16 计算），[多族混合 NF4](../experimental/multi-family-mixed-nf4.md) 明确该配置值仍拒绝 | 保留 error（文案应区分“运行时 NF4 未接入 / 预量化 mixed-NF4 可用”） | `tests/test_z_image_family.py:367` |
| `z_image_torch_compile`（L606） | z_image | C2 未实现 | compat L605-610；z_image 族内无任何 compile 代码（`grep compile library/models/z_image/` 无命中），且 AGENTS.md 不变量写明“Z-Image 当前不支持 compile，不能因为共享训练入口而绕过兼容拒绝” | 保留 error | `tests/test_z_image_family.py:368` |
| `z_image_block_swap_range`（L612） | z_image | B 范围校验 | compat L611-617；`library/models/z_image/block_swap.py:41-44` 为 `1 <= n <= len(layers)-2`，本族 30 层 → 上界 28 | 保留 error | `tests/test_z_image_family.py:404-422` |
| `z_image_selective_checkpoint`（L619） | z_image | C2 未实现 | compat L618-623；z_image 无 selective checkpoint 实现 | 保留 error | `tests/test_z_image_family.py:369` |
| `z_image_checkpoint_offload`（L625） | z_image | C2 未实现 | compat L624-629；`block_swap.py:145-149` 把 `*args/**kwargs` 原样透传给 Diffusers 的 `enable_gradient_checkpointing`，无 cpu/unsloth offload 语义 | 保留 error | — （无测试） |
| `z_image_layer_range`（L634） | z_image | C2 未实现 | compat L630-639；z_image 无 `layers.N`/refiner 层范围实现 | 保留 error | `tests/test_z_image_family.py:375` |
| `z_image_weighted_captions`（L641） | z_image | C2 未实现 | compat L640-645；加权路径需 `encode_tokens_with_weights`（`batch_step.py:207`），z_image strategy 只有 `tokenize_with_weights`（`strategy.py:50`） | 保留 error | `tests/test_z_image_family.py:374` |
| `z_image_loss_type`（L648） | z_image | C1 未验证 | compat L646-652；通用损失 `library/training/losses.py:315-321` 支持 l1/huber 并会真正生效，缺的是官方配方外的验收 | **降级 warning** | `tests/test_z_image_family.py:376` |
| `z_image_timestep_range`（L654） | z_image | C2 未实现 | compat L653-659；`family.py:117-121` 固定官方 shifted-uniform 网格，无 t_min/t_max 注入点 | 保留 error | `tests/test_z_image_family.py:377` |
| `z_image_training_sampler`（L662） | z_image | C2 未实现（开关被忽略） | compat L660-666；`sampler` 在 z_image 训练路径完全未被读取（krea2 才用 `SAMPLER_REGISTRY`，`library/models/krea2_raw/family.py:184`） | 保留 error（或降级但文案须写“对 Z-Image 无效”） | `tests/test_z_image_family.py:378` |
| `z_image_v100_flash_stability`（L668） | z_image | C2 未实现 | compat L667-672；同 krea2，Anima-only 诊断（[V100 Flash 支持](v100_flash_attention_support.md)） | 保留 error | `tests/test_z_image_family.py:379` |
| `z_image_flow_shift`（L675） | z_image | C1 未验证 | compat L673-679；`family.py:117-121` **会读取** `args.discrete_flow_shift` 并传给 `shifted_uniform_sigmas`，非 6.0 值真实生效 | **降级 warning** | `tests/test_z_image_family.py:370` |
| `z_image_timestep_sampling`（L686） | z_image | C2 未实现（开关被忽略） | compat L680-690；`family.py` 固定 `shifted_uniform_sigmas`，不读 `timestep_sampling`；该键只在 anima 路径 `library/runtime/noise.py:110-130` 生效 | 保留 error | `tests/test_z_image_family.py:371` |
| `z_image_weighting_scheme`（L695） | z_image | C1 未验证 | compat L691-699；`library/training/noise_target.py:420-427` 使用 `args.weighting_scheme` 计算权重，会真正生效 | **降级 warning** | `tests/test_z_image_family.py:372` |
| `z_image_caption_dropout`（L702） | z_image | C2 未实现 | compat L700-707；双重守卫：`library/models/z_image/strategy.py:74-78` raise（“empty-prompt embeddings are cached explicitly” 之前不支持） | 保留 error（重复门） | `tests/test_z_image_family.py:373,425-437` |

## 7. 通用 block swap / offload / 量化基座（L710–L859、L957、L964）

| code | 所属族 | 分类 | 判定依据 | 建议动作 | 关联测试 |
| --- | --- | --- | --- | --- | --- |
| `negative_blocks_to_swap`（L710） | 通用 | B 输入校验 | compat L709-714 | 保留 error | `tests/test_block_swapping.py` |
| `selective_full_gradient_checkpointing`（L720） | 通用 | A 语义互斥 | compat L719-725；`library/anima/models.py:2103-2121` 的 selective 实现**先对每个 block `disable_gradient_checkpointing()`** 再按模式重开，而 `library/training/bootstrap.py:675-680` 在之后统一 `unet.enable_gradient_checkpointing()` 会重新打开全部 block → selective 被静默覆盖 | 保留 error | — （无直接断言） |
| `selective_cpu_offload`（L727） | 通用 | A 语义互斥 | 同上；selective 重开 block 时硬编码 `cpu_offload=False, unsloth_offload=False`（`models.py:2108-2110`），offload 永不生效 | 保留 error | — |
| `selective_unsloth_offload`（L733） | 通用 | A 语义互斥 | 同上（unsloth 分支同样被绕过） | 保留 error | — |
| `block_swap_cpu_offload`（L741） | 通用（族相关） | **A/C2：非 Anima 族放行即 TypeError** | compat L740-746；`bootstrap.py:675-680` 只把 `cpu_offload` 传给 checkpoint 包装。**族相关性**：只有 Anima 的 `unet.enable_gradient_checkpointing(cpu_offload=True)` 存在，属显存资源策略；**Krea-2**（`library/models/krea2_raw/dit.py:361` 的 block 级与 `:516` 的 DiT 级 `enable_gradient_checkpointing()` 都不接受 kwargs，t4 真机在 `krea2_raw` 实测 `TypeError: ... got an unexpected keyword argument 'cpu_offload'`）与 Qwen/Z-Image 的 Diffusers transformer 同样不接受 `cpu_offload` → 放行即 `TypeError`；与 swap 的 block 驻留调度叠加还会同时占 CPU 激活与 pinned block 缓冲 | 保留 error（任何策略档均不降级） | t4 `output/tests/krea2-c3-smoke`（`output/` 不入库） |
| `unsloth_cpu_offload`（L762） | 通用 | A 语义互斥 | compat L761-766；`library/anima/models.py:1412` `self.cpu_offload_checkpointing = cpu_offload if not unsloth_offload else False`——两者在实现层就互相清零 | 保留 error | `tests/test_training_compat_matrix.py:19-30` |
| `block_swap_unsloth_offload`（L768） | 通用 | C3 资源策略 | compat L767-773；CLI 帮助已声明互斥（`library/training/cli_args.py:2038-2039`） | 保留 error | — |
| `convrot_block_swap_int8_mutex`（L805） | 通用 | A 双重反量化 | compat L804-811 与 `library/runtime/convrot/checks.py:101-113` 同义硬互斥（double dequant / confused semantics），依据 `docs/experimental/convrot_int8_training.md §G.23`（由 `checks.py:116-140` docstring 引用） | 保留 error（重复门） | `tests/test_training_compat_matrix.py:154,164` |
| `nf4_convrot_mutex`（L815） | 通用 | **死代码** | `convrot_active`（L797-802）与 `nf4_active`（L803）都从**同一个** `base_compute` 字符串派生，二者恒不同真 → L814 条件恒假；实测 `base_compute=nf4 + convrot_scope=all` 不报该 code | 保留 error（**等价 no-op**）；建议删除或改为“nf4 + ConvRot 专用键”的真实校验 | — （无测试） |
| `invalid_base_compute`（L859） | 通用 | B 枚举校验 | compat L848-864；`library/runtime/convrot/checks.py:68-90` 同义 raise。**文案与白名单不一致**（§8-3） | 保留 error；修文案 | `tests/test_training_compat_matrix.py:165,171,192` |
| `block_swap_soft_tokens`（L957） | 通用 | A 结构性冲突 | compat L956-962；`networks/methods/soft_tokens.py` 走“每 block monkey-patch forward + 额外对比/负向前向”的多前向路径，与 swap 的“单前向 + 每步一次 `prepare_block_swap_before_forward`”调度不兼容 | 保留 error | `tests/test_training_compat_matrix.py:32-41`、`tests/test_web_preflight_compat_matrix.py`、`tests/test_config_tools.py` |
| `block_swap_functional_loss`（L964） | 通用 | A 结构性冲突 | compat L963-969；`library/training/functional_hooks.py:18-55` 在 `unet.blocks[bi].cross_attn.output_proj` 上挂 forward hook 并依赖 inversion 的第二次前向快照，swap 会移动这些 block 且多前向超出 swap 调度 | 保留 error | `tests/test_training_compat_matrix.py:32-41` |

## 8. 消息文案与实际白名单 / 实现不一致清单

这些是“发现即记录”的偏差，均已在上面表格对应行标注；实现阶段应一并修正文案或白名单，
否则 warn 化后用户看到的提示会与实际行为矛盾。

| 编号 | 位置 | 偏差 | 影响 |
| --- | --- | --- | --- |
| 1 | `compat_matrix.py:491-497`（`krea2_invalid_attn_mode`） | 白名单 `KREA2_ATTN_MODES` 收 `sdpa`（L33），文案写“supports only torch or flash (sdpa is a torch alias)”；实现另收 `cudnn`/`cudnn_sdpa`（`krea2_raw/attention_backend.py:19`）但 compat 拒绝 | 用户按实现文档写 `attn_mode=cudnn` 会在 preflight 被拒 |
| 2 | `compat_matrix.py:571-580`（`z_image_attention_mode`）、`:195-197`（`qwen_image_2_1_attn_mode`） | 两族实现都把 `native` 归一为 `torch`（`z_image/attention_backend.py:17`、`qwen_image_2_1/attention_backend.py:26`），compat 白名单不含 `native` | 同上 |
| 3 | `compat_matrix.py:859-864`（`invalid_base_compute`） | 白名单 9 值（`bf16/fp16/none/off/w8a16_convrot/w8a8_convrot/w8a16/w8a8/nf4`），文案只列 4 值（`bf16 / w8a16_convrot / w8a8_convrot / nf4`），漏 `fp16/none/off/w8a16/w8a8` | 用户被误导为 `fp16` 非法 |
| 4 | `compat_matrix.py:600-604`（`z_image_base_compute`） | 文案“does not yet support NF4”与实现不符：z_image **支持预量化** mixed-NF4（`z_image/weights.py:238-251`），仅 `base_compute=nf4` 运行时量化未接入 | 用户误以为 z_image 完全不能 NF4 |
| 5 | `compat_matrix.py:466-472`（`krea2_plain_lora_only`） | 该 code 恒不可达，但 `web/services/config/preflight_compat.py:100` 仍保留中文文案 | 死文案，误导维护者 |
| 6 | `web/services/config/preflight_compat.py:101-114` | 仍保留 `pipeline_parallel_krea2_only` / `krea2_pipeline_parallel_config` 文案，而当前 PP v0 只由 anima 发布（`family_registry.py:110-120`） | 死文案 |
| 7 | `compat_matrix.py:33` vs `library/models/krea2_raw/attention_backend.py:14` | 同一族 attention 白名单存在两份独立字面量（`KREA2_ATTN_MODES` / `KREA2_ATTENTION_MODES`），已出现不同步（§8-1） | 漂移风险 |
| 8 | `compat_matrix.py:540-546` 与 `:618-623` | 当 `selective_checkpoint` 为非法值时，z_image 会同时产出 `invalid_selective_checkpoint`（B）与 `z_image_selective_checkpoint`（C2）两条 error | 重复报错，warn 化后会出现“一条 warn + 一条 error” |

## 9. 给实现阶段的建议（默认 warn / strict 回退）

> 本节是 t1 审计时点的建议，落地结果见[策略文档](compat_unverified_policy_20261109.md)。
> 现状：第 1/2/6 条已落地；第 3/4 条已完成（4 处文案修正 + 2 个死代码站点单独提交删除）；
> 第 5 条已补 7 个 C1 code 与 invalid 值的中文映射。**allow 档已收窄**：不再是“C2/C3 全部
> 降级”，而是只放行 14 个 C2-ignored（族内从不读取的键），C2-raise 与 C3 恒 error。

1. **默认 warn 白名单（7 个）**：§0 列出的 7 个 C1 code。其余保持 error。
2. **strict 开关语义**：建议按“按 code 粒度”回退，而不是按族整族回退——同一族内
   C1 与 C2 混杂（如 qwen 21 个 code 中只有 4 个是 C1）。
3. **实现前先修文案**：§8 的 1–4 项会让 warn 化后的提示自相矛盾，建议同批修正；
   第 5、6 项死文案可顺手删除。
4. **两个死代码站点**（`krea2_plain_lora_only`、`nf4_convrot_mutex`）不要纳入 warn 白名单，
   它们本来就是 no-op；建议单独一个提交删除或改为能力表驱动，便于审阅。
5. **WebUI 面**：`web/services/config/preflight_compat.py:196-227` 把所有 `compat.errors`
   直接 `add("error", ...)`、把 `warnings`/`mutations` 记为 warning。因此 C1 降级后
   WebUI 预检会自动放行，无需额外前端改动；但 `_compat_web_message` 缺 qwen/z_image
   新 code 的中文映射，会回落到 `item.message`（英文），建议补映射。
6. **不要放松的硬边界**：`convrot_block_swap_int8_mutex`、`unsloth_cpu_offload`、
   `selective_*` 三连、`block_swap_soft_tokens`、`block_swap_functional_loss` 属 A 类；
   `z_image_torch_compile`、qwen/z_image 的 `caption_dropout`、`weighted_captions`、
   `preview_text_cache` 属“放行即下游 raise”的 C2，放松只会把报错推迟。

## 10. 覆盖缺口（供后续补测试参考）

> 审计时点记录。实现阶段已由 `tests/test_compat_unverified_policy.py`（78 例）覆盖，
> 含六张触发器表（C1 / C2-ignored / C2-raise / C3 / 新增硬 code / A-B-mixed）、
> 三档判定、strict 35 组快照与 4 组白名单放宽；本节保留用于说明缺口当初如何评估。

以下 38 个 code 在 `tests/` 中**没有任何断言**（既无正向也无反向）：

`qwen_image_2_1_*`（21 个 code 全部只被 `tests/test_qwen_image_2_1_family.py:236-248`
的“errors 非空”间接覆盖 8 条配置，无一断言具体 code）、
`invalid_qwen_saved_projection_blocks`、`invalid_qwen_projection_budget_mib`、
`qwen_saved_projection_contract`、`qwen_fused_projections_{mixed_precision,base_compute,gradient_checkpointing,selective_checkpoint,cpu_offload_checkpointing}`、
`auto_block_swap_contract`（仅实现层）、`invalid_selective_checkpoint`、
`nf4_convrot_mutex`、`selective_full_gradient_checkpointing`、`selective_cpu_offload`、
`selective_unsloth_offload`、`block_swap_cpu_offload`、`block_swap_unsloth_offload`、
`z_image_checkpoint_offload`。

其中 **7 个建议降级为 warning 的 code 里有 4 个当前零测试**
（`qwen_image_2_1_mixed_precision`、`qwen_image_2_1_masked_loss`、
`qwen_image_2_1_loss_type`、`qwen_image_2_1_weighting_scheme`），
warn 化前建议补一条“默认档产出 warning 而非 error、strict 档仍为 error”的定向用例。

## 11. 证据强度与未核实项

- 本审计全部结论基于**当前工作树源码 + 运行时探测**（用主工作区 `.venv/bin/python`
  在隔离工作树内执行 `check_training_compat` 只读探测），未做 GPU 真机验证。
- “C1 未验证”只表示**实现会生效**，不代表数值/质量可接受；warn 化后应由用户在
  strict 开关与真机验收之间自行承担风险。
- `qwen_image_2_1_block_swap_checkpoint` 判为 C3 而非 A，依据是 `qwen_image_2_1/block_swap.py:20-68`
  构造时不校验 checkpointing；未实测“swap + 无 checkpointing”在 10 GB 卡上的实际
  失败形态（预期 OOM），若后续实测为必崩可上调为 A。
- `block_swap_soft_tokens` / `block_swap_functional_loss` 判为 A 依据的是实现层的
  前向次数与 hook 挂载位置，未构造真机反例；两者均无 findings 专文，证据强度为
  “代码结构 + 测试断言”而非实测。
- `z_image_checkpoint_offload` 的失败形态推断自 `z_image/block_swap.py:145-149` 的透传签名，
  未实跑 `cpu_offload_checkpointing=true`。
- t4 的真机 L2 冒烟（qwen C1、krea2 C2/C3、qwen huber+snr）已实测出三处与本审计推断不同的
  结论（`block_swap_cpu_offload` 对 Krea-2 为 C2-raise、huber+snr 必 raise、qwen fp16 子条件
  必 raise）；正文表格已按实测订正，逐条见 §12-c。


## 12. 已知后续（follow-up，2026-11-09 收口）

按 t8 复审与 captain 裁定，以下项本轮不修，登记为后续（实现状态见[策略文档](compat_unverified_policy_20261109.md) §8）：

| # | 事项 | 现状与下一步 |
| --- | --- | --- |
| a | **F9**：`compat_matrix.py` 已约 1364 行，超过 AGENTS.md「已超 1000 行不再承载新业务」的软约束 | 把审计分类表 + `_CompatBuilder` 的策略分支搬到 `library/training/compat_policy.py`（搬家型重构：行为不变、单独验收、单独提交）；**优先做** |
| b | C2 的长期方向 | 现在依赖人工维护的 14（ignored）/24（raise）划分；长期改为能力表驱动——由 `library/models/family_registry.py` 声明每族真正消费的键，兼容层从能力表派生 ignored/raise，避免人工清单再次漂移 |
| c | t4 真机发现的分类订正 | 已在实现中落地，本表口径同步：① `block_swap_cpu_offload` 对 `krea2_raw` 实测 `TypeError`（`krea2_raw/dit.py:361,516` 签名无 kwargs），实为 **C2-raise** 而非 C3 资源策略；② `loss_type=huber` + 默认 `huber_schedule=snr` 实测第一步 `NotImplementedError`（`losses.py:212`）→ 新增 A 类硬错；③ qwen `fp16 + mixed-NF4` 实测 `ValueError: Mixed NF4 requires BF16 compute`、`fp16 + flash` 由 attention backend 直接 `RuntimeError` → 拆成两个恒 error 子 code |
| d | **F7**：`masked_loss` 的语义 | 实现会进入 mask 分支，但数据集无 alpha mask 时 `mask_image_for_loss` 返回 None、`reduce_masked_loss` 回退 full-latent mean（`library/training/losses.py:324`、`library/training/mask_loss.py:16-46`）——属“**生效但不改变语义**”，不应按普通 C1 宣传。后续在 C1 原因与 warning 文案里写明该前提，并在无掩码时补一条提示 |
| e | **F8**：WebUI 入口 | 表单尚未暴露 `compat_unverified_policy`，WebUI 预检沿用默认 `warn`；CLI `--compat_unverified_policy` 与 TOML 同名键已可用，入口说明见 `docs/configuration/model-training-capabilities.md` |
| f | **O1**：汇总日志重复 | `compat_unverified_policy=... relaxed: [...]` 会在 `library/training/extra_args.py`、bootstrap（`validate_qwen_dataset_config`）与 WebUI 预检三处各打一次（各入口共用 `check_training_compat`）。建议保留 `check_training_compat` 内的一次，其余调用点降到 debug，或按 (policy, code 集合) 去重 |
| g | **O3**：NF4 探测覆盖不足 | `qwen_image_2_1_mixed_precision_nf4` 的文件探测（`library/quantization/format.py::inspect_mixed_nf4` 读 safetensors 头）目前只被 monkeypatch 用例覆盖；未下载的 NF4 checkpoint 会预检放行、加载时才报错。拆分轮补基于临时 safetensors 头的集成用例（构造最小 manifest fixture） |
| h | 与本次无关的既有缺陷 | `save_every_n_steps=0` / `save_every_n_epochs=0` 分别在 `library/training/checkpoints.py:1166`、`:1212` 触发 `ZeroDivisionError: integer modulo by zero`（CLI 未拒绝 0，`get_remove_ckpt_no` 的取模同样会炸）。标为已知问题，**本分支不修**；建议下一轮在 `library/training/cli_args.py` 校验 `>= 1`，或按 `None` 语义处理 |

## 13. 附：GPU L2 跑测规范

t4 真机冒烟的完整记录在 `output/tests/GPU_L2_SMOKE_SUMMARY.md`（`output/` 被 gitignore，
不随分支提交，只作为运行数据留存）。可复用的规范：

- **卡号对齐**：`nvidia-smi` 的序号与 CUDA 默认的 `FASTEST_FIRST` 枚举不一致；
  `CUDA_VISIBLE_DEVICES` 必须搭配 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，才能与 gpu-gate 返回的
  卡号对齐。例：`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2`。缺该变量时，
  `1,2` 曾实际落到别的卡（混入被禁用的 GTX 1050 4 GiB 卡）并导致一次 OOM。
- **gpu-gate 误报**：桌面进程（CherryStudio、kwin_wayland 等）占用 util 时，gpu-gate 会把
  该卡判成 `training` 而拒绝分配；判卡应以显存占用 + `nvidia-smi --query-compute-apps`
  的计算进程为主，util 只作参考，误报时人工复核后再决定是否放宽白名单。
- **单次冒烟口径**：`timeout 900`，记录退出码、完成步数、峰值显存与首/末步 loss；
  skip、OOM、0 步与“未执行”一律不计通过，也不写成“已验证”。
