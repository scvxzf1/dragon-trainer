# 上游核心合并计划 - 2026-07-02

状态：已归档

## 目标

基于同步源报告
`/home/scv/nvme0n1p1/训练器相关/anima_lora-同步源项目更新/UPSTREAM_UPDATE_REPORT_2026-07-02.md`
和本仓实时文件状态，规划从 `f68e7ca1` 到 `0cd59cbf` 的上游核心增量合并。

本计划只覆盖训练、推理、预处理、adapter/network、配置和测试相关核心代码。
不合并文档库和 GUI 部分；不做整包 merge。

## 执行状态（截至 2026-07-03）

- `P0.1` 已完成：
  - 已执行工作区保护审计：
    - `git status --short --branch`
    - `git diff --check`
  - 已确认本轮继续排除 `docs/**`、`_archive/**`、`gui/**`、`web/**`、`custom_nodes/**`
  - 已确认 `.worktrees/`、`tmp/` 只作为本地辅助目录，不纳入本轮提交范围
- `P0.2` 已完成：
  - 已接入 `expandable_segments:True` 默认值
  - 已接入 `partitioner_recompute_views` / `partitioner_aggressive_recomputation`
  - 已验证：
    - `timeout 60 .venv/bin/python -m pytest tests/test_partitioner_tuning.py tests/test_unsloth_checkpoint.py`
    - `timeout 60 .venv/bin/python -m pytest tests/test_runtime_harness_cli.py tests/test_block_swapping.py`
- `P0.3` 已完成：
  - 已新增独立 `register` method，不覆盖默认 `lora.toml`
  - 已验证：
    - `timeout 60 .venv/bin/python -m pytest tests/test_register_method.py`
    - `timeout 60 .venv/bin/python tasks.py print-config METHOD=register PRESET=default`
    - `timeout 60 .venv/bin/python -m pytest tests/test_config.py`
- `P0.4` 已完成：
  - 已把默认 VAE cache batch 收紧到 `2`
  - 已验证：
    - `timeout 60 .venv/bin/python -m pytest tests/test_preprocess_paths.py tests/test_preprocess_dataset.py`
    - `timeout 60 .venv/bin/python -m pytest tests/test_caption_index.py`
- `P1.1` 已完成本地闭环：
  - 已为 LoRA-family 接入 register tokens 的 kwarg 放行、cfg 解析、save/load、merge refusal、optimizer lr group、compile dynamic-seq 上界放宽和 block-swap 风险警告
  - 已补最小 `REPA` trim 落点：`library/training/repa.py::pool_dit_tokens_to_grid()`
  - 已新增本地测试 `tests/test_lora_register_tokens.py`
  - 已验证：
    - `timeout 60 .venv/bin/python -m pytest tests/test_lora_register_tokens.py tests/test_training_bootstrap.py tests/test_network_registry.py`
    - `timeout 60 .venv/bin/python -m pytest tests/test_factory_metadata_flow.py tests/test_network_cfg.py tests/test_runtime_harness_cli.py tests/test_register_method.py`
    - `timeout 60 .venv/bin/python -m pytest tests/test_config.py`
- `P1.2` 暂缓：
  - 该切片命中推理 core，且本轮硬约束是不主动碰本地推理链用户工作
- `P1.3` 暂缓：
  - 完整 Turbo / REPA 收束不纳入本轮
  - 仅保留 `P1.1` 所需的最小 `REPA` trim 工具，不继续扩展到 Turbo 训练栈
- 前置锚点和四个热运行测试已完成：
  - `pre_anchor_bf16_lora`、`post_fp16_lora`、`post_fp32_lora`、`post_bf16_lokr`、`post_bf16_lora` 均已完成 30/30
  - 5 份 block-swap profile 均已落盘，并包含 `block_swap_config` 与 forward/backward wait 或 transfer 事件
  - `post_fp32_lora` 因当前环境 `FlashAttention` 不支持 fp32，runtime TOML 已固定 `attn_mode = "torch"`，复跑通过
  - 热测 runtime TOML 已固定 dataset/model/GPU 映射修正，见 `/tmp/anima-upstream-core-merge-20260702/MANIFEST.md`

## 当前基线

- 本仓当前 HEAD：`165e3845 feat: add web image test and inference runtime tuning`
- 本仓 `origin/main`：`f68e7ca1 fsg`
- 同步源 `upstream-snapshot/main`：`0cd59cbf registernode fix`
- 本轮上游增量：`origin/main..upstream-snapshot/main`
- 同步源报告统计：206 个文件变化，新增 25939 行，删除 2936 行，新增文件 110 个。
- 本仓当前工作区未提交改动以本轮核心合并文件和新增测试/辅助文件为主。
- 当前 `git status --short` 未显示 `docs/**`、`_archive/**`、`gui/**`、`web/**`、`custom_nodes/**` 的本轮改动。

## 硬约束

1. 不执行 `git merge upstream-snapshot/main`。
2. 不整包 cherry-pick 大型提交。
3. 不合并以下路径：
   - `docs/**`
   - `_archive/**`
   - `gui/**`
   - `README*.md`
   - `CLAUDE.md`
   - `AGENTS.md`
4. 默认不合并 WebUI 路径，避免覆盖当前本地 WebUI 大量未提交工作：
   - `web/**`
   - `configs/web-ui-settings.toml`
   - `configs/web-user-locks.toml`
   - `configs/sample_prompts.txt`
   - `configs/sample-prompts/**`
   - `configs/imported/**`
   - `configs/web-training-history/**`
   - `configs/web-training-queue/**`
5. 不合并 `custom_nodes/**` 的节点替换；ComfyUI 节点只作为后续独立发布议题。
6. 不把上游 `configs/methods/lora.toml` 的 register 默认值直接覆盖到本地默认 LoRA。
7. 不删除本地 bench、WebUI、GUI、用户配置、训练历史、模型、缓存或输出。
8. 不默认运行长训练、下载模型、清空队列或启动生产训练。

## 禁入和暂缓路径

禁入路径：

```text
docs/**
_archive/**
gui/**
web/**
custom_nodes/**
README*.md
CLAUDE.md
AGENTS.md
configs/web-ui-settings.toml
configs/web-user-locks.toml
configs/sample_prompts.txt
configs/sample-prompts/**
configs/imported/**
configs/web-training-history/**
configs/web-training-queue/**
```

暂缓路径：

```text
sr/**
bench/**
configs/gui-methods/**
scripts/tasks/sr.py
```

说明：

- `sr/**` 是大体积 ResShift/RSD 侧线，含独立依赖和论文 PDF，不纳入本轮 LoRA/LoKr 热测目标。
- `bench/**` 中的上游资料可只读参考，不作为本轮合并交付。
- `configs/gui-methods/**` 不从上游合并；LoKr 热测可以读取本仓已有 LoKr 配置，或生成临时 runtime TOML。

## 合并策略

采用功能切片手工移植：

1. 每个切片先看 `origin/main..upstream-snapshot/main` 的最小 diff。
2. 再对照本仓 HEAD 和未提交工作区，判断是否已有等价实现。
3. 只移植必要代码，不做整文件覆盖。
4. 对本地已重构模块，适配本地架构，而不是恢复上游旧结构。
5. 每个切片完成后跑对应单测或最小验证。
6. 全部合并完成后跑四个 30 步真实热运行复测。

## 前置锚点

前置锚点只跑 `bf16 + lora` 一项真实热运行，不跑 fp16、fp32、LoKr。

执行状态：

- 已于 2026-07-03 执行并通过
- 日志：
  - `/tmp/anima-upstream-core-merge-20260702/logs/pre_anchor_bf16_lora.log`
- profile：
  - `output/ckpt/upstream-core-merge/logs/pre_anchor_bf16_lora.block_swap_profile.jsonl`
- 结果：
  - 完成 30/30
  - 未见 OOM、`ConstraintViolationError`、block-swap desync、checkpoint mismatch 或 `NaN`

锚点定义：

- adapter：LoRA
- precision：`mixed_precision = "bf16"`
- 完整 checkpoint：`gradient_checkpointing = true`
- selective checkpoint：`selective_checkpoint = "off"`
- block swap：`blocks_to_swap = 12`
- compile：`torch_compile = true`
- dynamic seq compile：`compile_dynamic_seq = true`
- steps：真实训练 30 步

执行要求：

1. 生成临时 runtime TOML，放在 `/tmp/anima-upstream-core-merge-20260702/pre_anchor_bf16_lora.toml`。
2. 运行时确保 `max_train_steps = 30`，且 runtime TOML 中不要保留 `max_train_epochs`，避免 epoch 配置覆盖 30 步目标。
3. 禁用 sample preview，避免把锚点变成推理或 VAE 压力测试。
4. 输出目录使用独立子目录，例如 `output/ckpt/upstream-core-merge/pre_anchor_bf16_lora`。
5. 记录训练是否到达 step 30、是否生成 block-swap profile、是否出现 compile/checkpoint/block-swap 相关异常。

建议命令形态：

```bash
timeout 3600 .venv/bin/python train.py \
  --config_file /tmp/anima-upstream-core-merge-20260702/pre_anchor_bf16_lora.toml
```

临时 TOML 必备覆盖项：

```toml
method = "lora"
methods_subdir = "methods"
mixed_precision = "bf16"
max_train_steps = 30

gradient_checkpointing = true
unsloth_offload_checkpointing = false
cpu_offload_checkpointing = false
selective_checkpoint = "off"

blocks_to_swap = 12
block_swap_transfer_dtype = "bf16"
block_swap_restore_mode = "foreach"
block_swap_profile_jsonl = "auto"
disable_block_swap_for_eval = false

torch_compile = true
compile_dynamic_seq = true
compile_inductor_mode = ""
activation_memory_budget = 1.0

sample_every_n_steps = 0
sample_every_n_epochs = 0
use_cmmd = false

save_every_n_steps = 999999
save_every_n_epochs = 999999
checkpointing_epochs = 999999
```

通过标准：

- 训练完成 30 步。
- 无 OOM、无 `ConstraintViolationError`、无 block-swap forward/backward desync。
- 日志中能看到 `blocks_to_swap=12`、`gradient_checkpointing=True`、`torch_compile=True` 的有效配置。
- block-swap profile 中存在 forward/backward wait 或 transfer 事件。

执行状态：

- 已执行并通过
- 运行说明：
  - 使用 `/tmp/anima-upstream-core-merge-20260702/pre_anchor_bf16_lora.toml`
  - runtime TOML 已显式绑定可用的 `dataset.runtime.toml` 与 `anima-preview3-base.safetensors`
  - runner 固定 `CUDA_VISIBLE_DEVICES=0`，对应当前 PyTorch 可正常工作的 `RTX 3080`
- 工件：
  - 日志：`/tmp/anima-upstream-core-merge-20260702/logs/pre_anchor_bf16_lora.log`
  - profile：`output/ckpt/upstream-core-merge/logs/pre_anchor_bf16_lora.block_swap_profile.jsonl`
  - 输出目录：`output/ckpt/upstream-core-merge/pre_anchor_bf16_lora`
- 备注：
  - 输出目录保留 `pre_anchor_bf16_lora.snapshot.toml` 与 `pre_anchor_bf16_lora_moe.safetensors`
  - 日志仍会打印 plain checkpoint 路径 `pre_anchor_bf16_lora.safetensors`，先按当前 LoRA 保存行为记录，不阻断锚点判定
- 已生成 runtime TOML：
  - `/tmp/anima-upstream-core-merge-20260702/pre_anchor_bf16_lora.toml`

## P0 合并切片

### P0.1 保护本地状态

实施前动作：

```bash
git status --short --branch
git diff --stat
git diff --check
```

执行状态：

- 已完成工作区审计
- 已确认当前脏改动仍以本地核心合并文件为主，未触碰禁入路径
- `git diff --check` 对本轮相关路径保持干净

要求：

- 不清理、不覆盖本地未提交 WebUI/LoKr 工作。
- 如果需要实际实施，先创建独立分支或独立 worktree。
- 当前工作区中 `library/inference/*` 的本地改动必须先读清楚，再碰推理 core。

### P0.2 VRAM / compile / partitioner 旋钮

上游候选：

```text
library/runtime/allocator.py
library/runtime/harness.py
library/config/cli_args.py
train.py
tests/test_partitioner_tuning.py
tests/test_unsloth_checkpoint.py
```

合并意图：

- 新增 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 默认值。
- 新增 AOT partitioner tuning：
  - `partitioner_recompute_views`
  - `partitioner_aggressive_recomputation`
- 在 `compile_blocks_for_training()` 中接入这些旋钮。

本地适配点：

- 本地 `library/runtime/harness.py` 已有 dynamic-seq / repeated compile 安全补丁，不能整文件覆盖。
- `train.py` 已有 LoKr + full checkpoint + compile 警告逻辑，必须保留。
- partitioner tuning 在 `gradient_checkpointing=true` 下必须跳过。

验证：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_partitioner_tuning.py tests/test_unsloth_checkpoint.py
timeout 60 .venv/bin/python -m pytest tests/test_runtime_harness_cli.py tests/test_block_swapping.py
```

### P0.3 独立 register 方法

上游候选：

```text
configs/methods/register.toml
networks/methods/register.py
networks/register_injection.py
scripts/tasks/training.py
tasks.py
tests/test_register_method.py
```

合并意图：

- 新增独立 register-token adapter。
- register method 通过 `network_module = "networks.methods.register"` 加载。
- register checkpoint kept-live，不进入静态 merge。

本地适配点：

- 本地已有 `networks/methods/base.py::AdapterNetworkBase`，上游 register 可基于它移植。
- 本地 `tasks.py` 支持 Make-style 尾随 env，合并 register 命令时只添加命令项，不回退任务入口结构。
- 不从上游合并 `custom_nodes/comfyui-anima-register/**`。
- 不修改默认 `configs/methods/lora.toml` 为 register 默认。

验证：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_register_method.py
timeout 60 env METHOD=register PRESET=default .venv/bin/python tasks.py print-config
timeout 60 .venv/bin/python tasks.py print-config METHOD=register PRESET=default
```

### P0.4 预处理和缓存修正

上游候选：

```text
library/io/cache_names.py
library/preprocess/images.py
library/preprocess/pe.py
library/preprocess/reconcile.py
scripts/preprocess/cache_latents.py
scripts/preprocess/cache_pe_encoder.py
scripts/tasks/preprocess.py
```

合并意图：

- VAE cache batch size 更保守。
- `pe_spatial` 孤儿缓存扫描更完整。
- Danbooru tag KB auto-fetch 相关预处理补丁。

本地适配点：

- 本地已有 `f10fc00c preprocess: harden latents and PE cache...`，先比对是否已覆盖同等能力。
- 不改变缓存 sidecar 命名。
- 不改变 `image_dataset/`、`post_image_dataset/`、`configs/imported/` 用户数据边界。

验证：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_preprocess_dataset.py tests/test_preprocess_paths.py
timeout 60 .venv/bin/python -m pytest tests/test_caption_index.py
```

## P1 合并切片

### P1.1 LoRA-family register tokens

上游候选：

```text
networks/lora_anima/config.py
networks/lora_anima/factory.py
networks/lora_anima/network.py
networks/register_injection.py
library/anima/merge.py
library/training/repa.py
tests/test_lora_register_tokens.py
```

合并意图：

- 允许普通 LoRA family 通过 `num_registers > 0` 训练 register tokens。
- 保存 `register_tokens`。
- 加载时 sniff `register_tokens` 并保持 kept-live。
- merge 时拒绝 register tokens。
- REPA token grid pooling 支持 trim register tail。

本地适配点：

- 本地 LoRA family 已 plugin/registry 化，不能按上游旧 `networks/__init__.py` 设计照搬。
- 需要把 `num_registers`、`register_insert_block`、`register_lr_scale`、`register_init_std` 加入本地 registry allowlist。
- 本地 LoKr 插件近期有大量优化，LoRA-register 改动不得破坏 `use_lokr` 互斥和 plugin 检测。
- 不把上游 `configs/methods/lora.toml` 默认切到 register；只保留可选注释或单独 variant。

执行状态：

- 已完成本地闭环
- 已接入：
  - registry allowlist 放行 register kwargs
  - `LoRANetworkCfg` 字段解析与 metadata 回放
  - 从权重 `register_tokens` sniff register mode
  - kept-live register injector 与 `extra_seq_tokens`
  - 独立 optimizer lr group
  - `register_tokens` 场景下 merge refusal
  - `compile_dynamic_seq` 的 `seq_range` 上界补入 `extra_seq_tokens`
  - `blocks_to_swap > 0` 时给出 register-token 风险提示
- 已补最小 `REPA` trim 支撑，不继续扩展完整 Turbo / REPA wiring

当前测试覆盖点：

- mid-stack register insert / strip
- plain LoRA 无 register 回归
- optimizer group + grad
- save/load roundtrip
- merge refusal
- REPA trim tail

验证：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_lora_register_tokens.py
timeout 60 .venv/bin/python -m pytest tests/test_network_cfg.py tests/test_network_registry.py
timeout 60 .venv/bin/python -m pytest tests/test_factory_metadata_flow.py tests/test_lora_custom_autograd.py
```

### P1.2 推理 correction pure-core 抽离

上游候选：

```text
library/inference/corrections/cns_core.py
library/inference/corrections/fsg_core.py
library/inference/corrections/mist_core.py
library/inference/corrections/mod_guidance_core.py
library/inference/corrections/cns.py
library/inference/corrections/fsg.py
library/inference/corrections/mod_guidance.py
library/inference/corrections/smc_cfg.py
library/inference/cfg_delta_probe.py
networks/spd.py
networks/spd_core.py
networks/spectrum.py
networks/spectrum_forecast.py
networks/spectrum_sea.py
```

合并意图：

- 抽出纯计算 core，减少 ComfyUI/vendor 与主仓推理实现漂移。
- `project_pooled()` 成为 modulation guidance 共享投影真相。
- SPD / Spectrum 拆出 core/forecast。

本地适配点：

- 当前工作区已修改：
  - `library/inference/corrections/mod_guidance.py`
  - `library/inference/generation.py`
  - `library/inference/models.py`
- 本地新增 `library/inference/precision.py`，推理精度路径必须与 core 抽离兼容。
- 该切片必须在 P0 完成并固定工作区后再做。

验证：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_generation_request.py tests/test_experimental_inference_tasks.py
timeout 60 .venv/bin/python -m pytest tests/test_edit_dispatcher.py tests/test_directedit_v_injection.py
```

### P1.3 Turbo / REPA 收束

上游候选：

```text
configs/methods/turbo.toml
scripts/distill_turbo/config.py
scripts/distill_turbo/distill.py
scripts/distill_turbo/metrics.py
library/training/repa.py
tests/test_repa.py
```

合并意图：

- 移除被上游 refuted 的 Turbo + REPA wiring。
- 保留 REPA 作为正式训练正则能力，但不默认绑到 LoRA。

本地适配点：

- 不合并 `docs/methods/repa.md` 或任何 docs 晋级。
- 不删除本地已有 Turbo/REPA bench 文件。
- 若本地配置仍依赖旧 Turbo REPA 入口，改为显式拒绝或保留兼容但默认关闭。

验证：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_repa.py
timeout 60 .venv/bin/python -m pytest tests/test_distill_runtime.py tests/test_turbo_metrics.py
```

## P2 暂缓

### P2.1 SR / ResShift / RSD

暂缓原因：

- 新增 `sr/**` 体积大，含独立依赖和 PDF。
- 与本轮 LoRA/LoKr 热测目标无直接关系。
- 需要独立环境策略，不应塞进核心合并。

后续若要合并，单独开计划：

```text
sr/**
scripts/tasks/sr.py
```

### P2.2 custom nodes

暂缓原因：

- 本轮约束不合并 GUI 部分。
- `custom_nodes/comfyui-anima-blockcompile/**` 被上游删除并替换为 register node，属于发布面变化。
- 本仓 vendor 树要求先改 live source，再跑 vendor-sync；不应在核心合并里手工分叉。

### P2.3 docs / archive / bench 资料重排

暂缓原因：

- 用户明确约束不合并文档库。
- 本轮大量 changes 是 docs、proposal、findings、archive、bench 重排。
- 这些资料可以只读参考，不进入本轮代码合并。

## 合并后复测矩阵

合并完成后必须跑四个真实热运行测试。四个测试都叠加：

- 完整 gradient checkpoint：`gradient_checkpointing = true`
- selective checkpoint 关闭：`selective_checkpoint = "off"`
- block swap：`blocks_to_swap = 12`
- torch compile：`torch_compile = true`
- dynamic seq compile：`compile_dynamic_seq = true`
- 真实训练步数：30
- sample preview 关闭
- CMMD 关闭

四个 case：

| case | method | methods_subdir | precision |
| --- | --- | --- | --- |
| `post_fp16_lora` | `lora` | `methods` | `mixed_precision = "fp16"` |
| `post_fp32_lora` | `lora` | `methods` | `mixed_precision = "no"` |
| `post_bf16_lokr` | `lokr` | `gui-methods` 或临时 LoKr runtime TOML | `mixed_precision = "bf16"` |
| `post_bf16_lora` | `lora` | `methods` | `mixed_precision = "bf16"` |

已生成执行工件：

- `/tmp/anima-upstream-core-merge-20260702/post_fp16_lora.toml`
- `/tmp/anima-upstream-core-merge-20260702/post_fp32_lora.toml`
- `/tmp/anima-upstream-core-merge-20260702/post_bf16_lokr.toml`
- `/tmp/anima-upstream-core-merge-20260702/post_bf16_lora.toml`
- `/tmp/anima-upstream-core-merge-20260702/run_all_hot_tests.sh`
- `/tmp/anima-upstream-core-merge-20260702/run_one_case_with_log.sh`
- `/tmp/anima-upstream-core-merge-20260702/HOT_RUN_RESULTS_TEMPLATE.md`
- `/tmp/anima-upstream-core-merge-20260702/MANIFEST.md`

建议执行方式：

- 不直接一把运行 `run_all_hot_tests.sh`
- 按 case 串行执行 `run_one_case_with_log.sh <case>`，让每个 case 单独落日志
- 执行完成后，把结论回填到 `HOT_RUN_RESULTS_TEMPLATE.md`
- `nvidia-smi -L` 显示物理卡顺序为 `GPU 0 = GTX 960`、`GPU 1 = RTX 3080`，但当前 PyTorch / CUDA 可见设备映射与其相反：
  - `CUDA_VISIBLE_DEVICES=0` -> `RTX 3080`
  - `CUDA_VISIBLE_DEVICES=1` -> `GTX 960`
- `/tmp` 下两个 runner 已默认设置 `CUDA_VISIBLE_DEVICES=0`；若后续需要改卡，必须显式覆盖，并避免 `CUDA_VISIBLE_DEVICES=1`
- 当前工作区缺少 repo-local `image_dataset/` 与 `post_image_dataset/resized/`，热测 runtime TOML 已改为显式绑定现成可用的 `dataset.runtime.toml`
- 当前工作区缺少 `models/diffusion_models/anima-base-v1.0.safetensors`，热测 runtime TOML 已改为显式绑定现成本地可用的 `models/diffusion_models/anima-preview3-base.safetensors`
- `post_fp32_lora` 需显式切到 `attn_mode = "torch"`，因为当前环境里的 `FlashAttention` 仅支持 `fp16/bf16`

建议命令形态：

```bash
timeout 3600 .venv/bin/python train.py --config_file /tmp/anima-upstream-core-merge-20260702/post_fp16_lora.toml
timeout 3600 .venv/bin/python train.py --config_file /tmp/anima-upstream-core-merge-20260702/post_fp32_lora.toml
timeout 3600 .venv/bin/python train.py --config_file /tmp/anima-upstream-core-merge-20260702/post_bf16_lokr.toml
timeout 3600 .venv/bin/python train.py --config_file /tmp/anima-upstream-core-merge-20260702/post_bf16_lora.toml
```

`post_fp32_lora` 说明：

- 本仓 argparse 使用 `mixed_precision = "no"` 表示 fp32 / no mixed precision。
- 若保存精度也要对齐，可设置 `save_precision = "float"`。

`post_bf16_lokr` 说明：

- 不从上游合并 `configs/gui-methods/**`。
- 运行时可以只读取本仓现有 `configs/gui-methods/lokr.toml` 生成临时 runtime TOML。
- 更稳妥做法是把 LoKr 必要标量复制到 `/tmp` runtime TOML：
  - `network_module = "networks.lora_anima"`
  - `use_lokr = true`
  - `lokr_factor = 8`
  - `lokr_factor_group_size = 8`
  - `lokr_project_chunk_bytes = 4194304`
  - `lokr_grouped_delta_backend = "eager"`

通过标准：

- 四个 case 均完成 30 步。
- 无 OOM、NaN、compile bound violation、checkpoint recompute mismatch。
- LoKr case 日志保留 LoKr + full checkpoint + compile 的实验警告，但不中断。
- block-swap profile 能看到 12-block swap 的 forward/backward 事件。
- fp16 case 不触发本地 fp16 acceleration patch 回退错误。
- fp32 case 明确显示 `mixed_precision=no`，没有误走 bf16/fp16。

执行状态：

- 已于 2026-07-03 串行执行并通过
- 结果汇总：
  - `post_fp16_lora`：完成 30/30；未见本地 fp16 acceleration patch fallback 错误
  - `post_fp32_lora`：首次因 `FlashAttention only support fp16 and bf16` 失败；切到 `attn_mode = "torch"` 后复跑完成 30/30，且日志明确显示 `weight/unet dtype = torch.float32`
  - `post_bf16_lokr`：完成 30/30；保留 `LoKr with full gradient_checkpointing under torch_compile is experimental` 等预期 warning，但未中断
  - `post_bf16_lora`：完成 30/30
- profile 汇总：
  - 4 份 profile 均存在，且包含 `block_swap_config` 与 forward/backward wait 或 transfer 事件
- 日志目录：
  - `/tmp/anima-upstream-core-merge-20260702/logs`
- 结果模板已回填：
  - `/tmp/anima-upstream-core-merge-20260702/HOT_RUN_RESULTS_TEMPLATE.md`

## 常规复测

P0/P1 完成后，再跑以下常规测试：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_config.py tests/test_runtime_harness_cli.py
timeout 60 .venv/bin/python -m pytest tests/test_block_swapping.py tests/test_compile_checkpoint_block_swap_hot.py
timeout 60 .venv/bin/python -m pytest tests/test_network_cfg.py tests/test_network_registry.py tests/test_lokr.py
timeout 60 .venv/bin/python -m pytest tests/test_preprocess_dataset.py tests/test_preprocess_paths.py
timeout 60 .venv/bin/python -m pytest tests/test_generation_request.py tests/test_experimental_inference_tasks.py
```

如果只完成 P0，不跑 P1 推理 core，则最后一条推理测试可延后。

执行状态（截至 2026-07-03）：

- 已完成：
  - `timeout 60 .venv/bin/python -m pytest tests/test_config.py tests/test_runtime_harness_cli.py`
  - `timeout 60 .venv/bin/python -m pytest tests/test_block_swapping.py tests/test_compile_checkpoint_block_swap_hot.py`
  - `timeout 60 .venv/bin/python -m pytest tests/test_network_cfg.py tests/test_network_registry.py tests/test_lokr.py`
  - `timeout 60 .venv/bin/python -m pytest tests/test_preprocess_dataset.py tests/test_preprocess_paths.py`
  - `timeout 60 .venv/bin/python -m pytest tests/test_partitioner_tuning.py tests/test_unsloth_checkpoint.py tests/test_register_method.py`
  - `timeout 60 .venv/bin/python -m pytest tests/test_lora_register_tokens.py tests/test_training_bootstrap.py tests/test_factory_metadata_flow.py`
  - `timeout 60 .venv/bin/python tasks.py print-config METHOD=register PRESET=default`
- 当前结果：
  - 上述命令均通过
  - `test_block_swapping.py` 与 `test_lokr.py` 仍会打印本机 `GPU1 GTX 960 / sm_52` 的兼容性 warning，但不影响通过
- 延后：
  - `timeout 60 .venv/bin/python -m pytest tests/test_generation_request.py tests/test_experimental_inference_tasks.py`
  - 原因：`P1.2` 推理 core 抽离本轮暂缓，且当前目标不主动碰本地推理链未提交工作

## 完成条件

本计划完成的最低标准：

1. P0.1 到 P0.4 已实施或明确标记跳过原因。
2. 前置 `bf16+lora` 锚点和合并后四个热测都有日志记录。
3. 不出现 `docs/**`、`_archive/**`、`gui/**`、`web/**`、`custom_nodes/**` 的合并改动。
4. `configs/methods/lora.toml` 本地默认不被上游 register 默认覆盖。
5. 新增 register method 如合并，必须有单测与 `print-config` 验证。
6. LoRA-family register tokens 如合并，必须通过 save/load/merge refusal/REPA trim 测试。
7. `git diff --check` 对本轮改过的路径干净。

## 建议执行顺序

```text
S0  保护工作区 + 建立前置 bf16+lora 锚点
S1  合并 allocator/partitioner 旋钮
S2  合并独立 register method，不改默认 LoRA
S3  合并预处理/cache 小修
S4  跑 P0 常规测试
S5  决策是否进入 LoRA-family register tokens
S6  决策是否进入 inference correction pure-core
S7  四个 30 步热运行复测
S8  汇总结果，列出未合并的 docs/gui/SR/custom_nodes/bench 项
```

优先级建议：

- 先做 P0.2、P0.3、P0.4。
- P1.1 和 P1.2 不要并行改同一轮，二者都碰到训练/推理关键路径，适合分两次提交。
- P2 本轮只保留评估记录，不进入代码合并。
