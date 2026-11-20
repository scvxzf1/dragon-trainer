# P5 通用层与族插件契约修订

状态：Phase 5 代码/契约与定向验收收口完成（P5-B1..B8、F-01..F-03、Phase 5 closure tests）；F-04 的 P4→P5 正式合入仍为外部发布 gate；F-05 为后续协议拆分。
适用分支：`codex/abstraction-wiring-research`
审计基线：`93aaeddc` 上的 P5 工作树

本文记录 P5 审计阻断项的修订与验证状态。它不把“代码已改”写成“真机已验收”；
GPU 结果必须单独补记。

## 阻断项修订表

| 编号 | 阻断项 | 修订内容 | 验证 | 状态 |
| --- | --- | --- | --- | --- |
| P5-B1 | `family_runtime` 按五族硬编码 validator/installer，MOSS 错指向 Z-Image | 通用入口只经 `get_family_plugin()` 构建 runner，调用 runner 的 `validate/prepare`；删除五族 handler 表 | `tests/test_family_inference_contracts.py`、`tests/test_family_inference_plugins.py` | 实现完成；CPU 定向通过 |
| P5-B2 | Krea/Qwen runner 反向导入顶层 `inference.py`，Krea 借用 Anima 保存函数 | 采样主体迁入族 runner；`inference.py` 只保留兼容 shim；Anima/Krea 共用中立 Qwen VAE 保存 helper | 族 runner 契约测试、Qwen facade 测试、源码反向依赖检查 | 实现完成；CPU 定向通过 |
| P5-B3 | 插件只校验 `family`，runner 只校验 `family` | 插件必需方法面与 runner 的 `validate/prepare/generate_latent/save_output` 改为 fail-closed 校验；新增 `supports_reference_conditioning` 能力位 | `tests/test_family_inference_contracts.py` | 实现完成；CPU 定向通过 |
| P5-B4 | `run_text()` 成功路径不清理；`max_oom_retries` 无人启用；各族手工 `del`/`gc`/`clean_memory` | `run_text(loader, operation)` 成功/失败都释放；stage 默认 OOM 重试 1 次；Krea/Z/MOSS/Qwen 改走 stage 生命周期 | `tests/test_family_inference_runner.py`、`tests/test_family_inference_runner_hot.py`、Qwen/Krea 定向测试 | 实现完成；CPU 定向通过；双 V100 hot `10 passed` |
| P5-B5 | 训练预览默认从 Anima 导入解码并保留族内分支 | 新增中性 `library/training/preview_live_decode.py`；五族插件提供 `decode_preview_samples`；`sample_preview_common`/`train_bootstrap` 走插件 | `tests/test_family_preview_lifecycle.py`、`tests/test_block_swap_cpu_park.py`、`tests/test_deferred_sample_cleanup.py` | 实现完成；CPU 定向通过 |
| P5-B6 | MOSS `latent_images` 不保存 latent | MOSS runner 与 Z-Image 对齐：`latent` 和 `latent_images` 都写 latent，只有 `latent` 提前返回 | `tests/test_family_inference_contracts.py::test_moss_latent_images_writes_latent_and_renders_image` | 实现完成；CPU 定向通过 |
| P5-B7 | Z-Image/MOSS 文档仍写“无独立推理”，与 registry 不一致 | 同步多模型矩阵、推理参考与 MOSS 配置页，明确 `single` + `euler` 及尚未开放的边界 | `tests/test_documentation_integrity.py` | 实现完成；CPU 文档完整性通过 |
| P5-B8 | P4/P5 工作树分叉，插件契约与训练/推理生命周期文件重叠 | P4 工作树保持只读；将 P4 的 flow/cache 契约重放到 P5：Anima noise-target 迁入族目录、训练层保留兼容 shim，五族插件同时保留 P4 flow/cache hooks 与 P5 inference/decode hooks | P4 契约套件（P4 测试文件、P5 实现）`68 passed / 3 skipped`；P5-B8 定向套件 `184 passed`（含 F-02 新增 2 项）；目标文件 `py_compile` 与 `git diff --check` 通过 | 实现完成；插件契约实现为 P5 超集；待按 P4→P5 顺序正式合入并跑合并后二次回归 |

## 审计修订记录（2026-11-19）

| 审计项 | 优先级 | 修订内容 | 验证 | 状态 |
| --- | --- | --- | --- | --- |
| F-01 MOSS 默认 NF4 与 CPU placement 冲突 | P1 | 独立 CLI 新增 `--moss_te_quantization {nf4,bf16}`；NF4 将 TE stage 放到选定 CUDA 设备并在 `run_text` 收尾释放；CPU/TE-CPU 组合要求显式 BF16，BF16 compute 不做隐式回退 | `tests/test_moss_image_inference_contract.py`、`tests/test_family_inference_runner.py`；真实 MOSS V100 smoke（BF16 TE + 在线 NF4，1 step，64×64，latent） | 实现完成；CPU 定向与真实权重 smoke 均通过 |
| F-02 runner 生命周期覆盖范围过宽 | P2 | 保持 registry 兼容 Anima 的 `single/batch/interactive`，但文档和分派测试明确只有 `single` 调用 `generate_latent/save_output`；Anima 的 `batch`/`interactive` 继续走 legacy 入口 | `tests/test_family_inference_contracts.py::test_main_generates_through_family_runner_only_in_single_mode`、`tests/test_family_inference_plugins.py::test_only_anima_publishes_legacy_batch_and_interactive_modes` | 实现完成；目录与文档契约已锁定 |
| F-03 latent cache 未校验空间步长 | P2 | 共享写入边界校验 batch、channel、`bucket_reso` 对齐及 registry `vae_spatial_compression` 对应的 H/W；正向与翻转 latent 均校验 | `tests/test_latent_cache_geometry.py`（五族正向/错误 shape、batch mismatch、flip） | 实现完成；CPU 定向通过 |
| F-04 P4/P5 重叠文件需冲突感知集成 | P2 | 本轮不覆盖 P4 工作树；保留 P5 插件超集与 P4 flow/cache 现场，等待维护者先提交 P4，再在 P4 提交上三方重放 P5 | P4 契约集与 P5-B8 定向集已有历史结果；正式合并后需重跑 | 阻断于 P4 未提交现场，未执行覆盖式合并 |
| F-05 主插件协议耦合全部能力 | P3 | 不在本轮拆协议；记录为后续按 Flow / LatentCache / TextCache / InferenceRunner / Preview 能力位拆分的演进项，当前五族仍满足完整 fail-closed 契约 | 现有 `validate_plugin_contract` 测试 | 后续演进，当前不阻断 |

## Phase 5 收官证据（2026-11-19）

- 新增 `tests/test_phase5_closure.py`：锁定五族 runner 的共享 stage 入口、TE→DiT→VAE 顺序释放、Anima legacy generation 的 stage 包裹、Z-Image 的 TE→DiT 与 VAE stage 顺序，以及 Mixed-NF4 CLI 对 `moss_image` 的显式排除；`6 passed`。
- 修复旧 Anima Mixed-NF4 兼容边界：`library/quantization/format.py` 不再把 `anima.mixed_nf4.manifest` 的旧 packed artifact 误判为 canonical 四族格式；`tests/test_mixed_nf4_preflight_risks.py` 回归通过。
- 历史 CPU 收口组合 `176 passed`；本轮新增 Z-Image stage 测试后的 closure 定向组 `57 passed`；既有 P5-B8 套件 `184 passed`；Phase 5 预览/Flow/Registry/族级回归 `160 passed / 3 skipped`；文档完整性 `13 passed`。
- 双 V100 `FamilyInferenceRunner` 生命周期 hot：`10 passed`（授权 UUID；未使用 GTX 1050）。
- 真实 MOSS-Image V100 smoke（`2026-11-19 19:50:21`）：`--infer_steps 1 --image_size 64 64 --output_type latent --moss_te_quantization nf4 --text_encoder_dtype bf16 --attn_mode torch` 完成 TE→DiT 采样，产物 `(1,128,4,4)`、`float32`、全有限；临时产物为 `/tmp/p5-moss-smoke-final-20261119/20261119-195021-395_1234_latent.safetensors`，证明默认 NF4/CUDA placement 路径可执行。
- V100 Flash 边界保持 fail-closed：Krea-2 `NF4 + BF16 outer + V100 Flash` 不宣称通过，生产契约要求 `attn_mode="torch"`；该限制是已验证的安全边界，不是未实现分支。

## 验收边界

- 已完成的是无 GPU 的契约、生命周期与定向回归验证：本轮 closure 与
  F-01/F-02/F-03 定向组 `57 passed`，B8 既有套件重跑
  `184 passed`，共享缓存跨族回归 `104 passed`，文档完整性 `13 passed`；跳过项为
  测试自身环境条件，不是新增失败。
- P5-B4 已在授权的两张 V100 上执行真实 CUDA 生命周期 hot：
  `10 passed`。覆盖成功释放、操作失败释放、loader 失败释放、真实
  `torch.cuda.OutOfMemoryError` 注入后的重试，以及 `run_text` 成功/失败释放；
  两卡分别以 `v100-0` / `v100-1` 运行；F-01/F-03 修订后复跑仍为 `10 passed`。
- Anima 的 `library.inference.generation.generate()` 仍作为单个 composite legacy
  stage 被通用 runner 包裹（统一成功/失败清理与 OOM 重试）；其内部 DiT/TE 拆分保持
  既有 Anima 数值路径，尚未在本次改动中重排。
- MOSS-Image 独立推理真实模型 smoke 已完成；Z-Image 因本机没有本地权重，未执行
  真实模型 smoke，其 runner 由 TE→DiT→VAE stage 顺序测试覆盖。Krea/Qwen 的
  V100 端到端推理尚未单独执行；P5-B4 的两张 V100 hot 已完成，不代表其他族
  runner 的端到端推理已完成。
- F-01 的 NF4/CUDA 真实 MOSS 权重 smoke 与 BF16/CPU 契约测试均已通过；BF16 CPU
  分支目前由契约测试覆盖，未单独跑真实权重 smoke。F-04 的正式合入顺序仍必须先
  保护并提交 P4 工作树，再进行冲突感知重放。
- P4 工作树仍未提交，本轮未修改、合并或清理该现场。正式合入顺序必须为：
  P4 先提交/合入，P5 再重放/合并，最后在合并结果上执行全套回归；禁止用 P5
  工作树直接覆盖 P4。
- P5-B8 的 P4 契约复验使用 P4 工作树中的测试文件、以 P5 工作目录作为实现源
  执行；测试文件未复制到 P5，避免正式合入时制造重复文件与冲突。
- 两张 V100 的 UUID 仅在用户后续授权热测时使用；不使用 GTX 1050。

## 变更文件索引

- `library/inference/family_runtime.py`
- `library/inference/family_runner.py`
- `library/inference/family_output.py`
- `library/inference/anima_legacy_runner.py`
- `library/models/family_plugins.py`
- `library/training/flow_step.py`
- `library/strategies/latent_cache.py`
- `library/strategies/text_cache.py`
- `library/anima/noise_target.py`
- `library/training/noise_target.py`
- `library/models/{krea2_raw,z_image,qwen_image_2_1,moss_image}/inference_runner.py`
- `library/models/{krea2_raw,z_image,qwen_image_2_1,moss_image}/plugin.py`
- `library/anima/plugin.py`
- `library/training/preview_live_decode.py`
- `library/training/sample_preview_common.py`
- `library/training/train_bootstrap.py`
- `library/anima/training.py`
- `tests/test_family_inference_runner_hot.py`
- `tests/test_moss_image_inference_contract.py`
- `tests/test_latent_cache_geometry.py`
- `tests/test_phase5_closure.py`
- `library/quantization/format.py`

## 复验命令

```bash
/mnt/SD_Dataset/训练器相关/krea2-webui/.venv/bin/python -m pytest \
  tests/test_family_inference_contracts.py \
  tests/test_family_inference_plugins.py \
  tests/test_family_inference_runner.py \
  tests/test_qwen_image_2_1_inference.py \
  tests/test_qwen_image_2_1_edit_inference.py \
  tests/test_family_preview_lifecycle.py \
  tests/test_preview_lifecycle.py \
  tests/test_deferred_sample_cleanup.py \
  tests/test_block_swap_cpu_park.py \
  tests/test_model_family_registry.py \
  tests/test_flow_step.py \
  tests/test_text_cache_session.py -q
# 184 passed

/mnt/SD_Dataset/训练器相关/krea2-webui/.venv/bin/python -m pytest \
  tests/test_moss_image_inference_contract.py \
  tests/test_latent_cache_geometry.py \
  tests/test_family_inference_contracts.py \
  tests/test_family_inference_plugins.py \
  tests/test_family_inference_runner.py \
  tests/test_moss_image_family.py \
  tests/test_model_family_registry.py \
  tests/test_text_cache_session.py -q -p no:cacheprovider
# 92 passed / 3 skipped

/mnt/SD_Dataset/训练器相关/krea2-webui/.venv/bin/python -m pytest \
  tests/test_documentation_integrity.py -q -p no:cacheprovider
# 13 passed
```

GPU 热测完成后，应将实际命令、退出码和未覆盖范围追加到本表，不得只写“通过”。

P5-B8 的 P4 契约复验使用 P4 工作树中的 `test_flow_step.py`、
`test_model_family_registry.py`、`test_moss_image_family.py`、
`test_qwen_image_2_1_family.py` 与 `test_text_cache_session.py`，在 P5 工作目录
执行，结果为 `68 passed / 3 skipped`。目标文件 `py_compile` 与
`git diff --check` 通过。

P5-B4 已执行：

```bash
RUN_FAMILY_INFERENCE_RUNNER_HOT=1 \
CUDA_VISIBLE_DEVICES=GPU-b4b014b2-953b-28e8-b8c1-425c4a401165,GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061 \
/mnt/SD_Dataset/训练器相关/krea2-webui/.venv/bin/python -m pytest \
  tests/test_family_inference_runner_hot.py -q -p no:cacheprovider
# 10 passed
```
