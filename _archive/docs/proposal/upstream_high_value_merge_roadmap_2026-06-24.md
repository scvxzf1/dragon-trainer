# 上游高价值内容合并 Roadmap - 2026-06-24

状态：已归档

## 目标

在不破坏当前项目 WebUI 和本地维护约束的前提下，评估并择优合并
`anima_lora-同步源项目更新` 中的高价值上游内容。

本计划不追求整包同步，而是按功能切片移植。每个切片必须可独立验证、可回退，并且
不能引入 GUI 代码或覆盖当前 WebUI。

## 当前基线

- 当前项目分支：`main`
- 当前项目 HEAD：`d3d7b498 feat: add webui environment check panel`
- 当前项目相对 `private/main`：ahead 7
- 上游同步源 HEAD：`f68e7ca fsg`
- 上游同步范围：从 `4fbf2db` 到 `f68e7ca`，约 114 个提交
- 当前项目与同步源不是简单快进关系：
  - `main..upstream-snapshot/main`：322 个提交
  - `upstream-snapshot/main..main`：90 个提交

当前项目已有未提交改动，后续实施前必须先确认并保护：

- `.gitignore`
- `AGENTS.md`
- `CLAUDE.md` 删除
- `library/config/loader.py`
- `web/services/config/datasets.py`
- `docs/findings/agent_audit_20260622/`

注意：本计划明确禁止修改 `AGENTS.md`，该文件即使已有本地改动，也不纳入本次合并工作。

## 硬约束

1. 不修改 `AGENTS.md`。
2. 不合并 `gui/**` 任何代码、资源、说明或测试。
3. 不接受上游对 `web/**` 的删除、替换或结构迁移。
4. 不删除或覆盖当前 WebUI 配置、运行状态和用户数据：
   - `configs/web-ui-settings.toml`
   - `configs/web-user-locks.toml`
   - `configs/sample_prompts.txt`
   - `configs/sample-prompts/**`
   - `configs/imported/**`
   - `configs/web-training-history/**`
   - `configs/web-training-queue/**`
5. 不整包执行 `git merge upstream-snapshot/main`。
6. 不整包 cherry-pick 大型上游提交。
7. 不直接采用上游 LoRA 实验默认值覆盖当前生产默认配置。
8. 不默认运行真实训练、下载大模型、清空队列或触碰训练输出。
9. 每个阶段完成后必须跑相关测试；后台测试必须设置超时。

## 禁入路径

以下路径只允许只读评估，不能从上游合并：

```text
AGENTS.md
gui/**
web/**
configs/web-ui-settings.toml
configs/web-user-locks.toml
configs/sample_prompts.txt
configs/sample-prompts/**
configs/imported/**
configs/web-training-history/**
configs/web-training-queue/**
```

以下路径需要极高谨慎，默认不接受上游删除：

```text
configs/gui-methods/**
configs/datasets/**
tests/test_web_config_service.py
tests/test_training_queue.py
tests/test_preview_service.py
tests/test_training_frontend_state.py
tests/test_weight_analysis_service.py
```

## 高价值候选分级

### P0：优先合并

这些内容收益明确，且可以通过小切片降低风险。

#### 预处理健壮性

候选能力：

- 空 caption fallback
- `use_repa` 缺 checkpoint 时 fail-fast
- PE auto-chain
- load-skip pre-flight
- cache reconcile
- mask 输出重定向到 path scope 下的 `masks/<scope>`
- 避免重复 TE image open

候选路径：

```text
library/preprocess/images.py
library/preprocess/latents.py
library/preprocess/pe.py
library/preprocess/text.py
library/preprocess/reconcile.py
library/io/cache_names.py
scripts/preprocess/cache_latents.py
scripts/preprocess/cache_pe_encoder.py
scripts/preprocess/cache_text_embeddings.py
scripts/preprocess/reconcile_caches.py
scripts/preprocess/resize_images.py
scripts/tasks/preprocess.py
tests/test_preprocess_tasks.py
tests/test_preprocess_dataset.py
tests/test_te_cache_pruned_load.py
```

约束：

- 不全量迁移 free-fit。
- 不删除当前 constant-token/native-flatten 相关测试，除非已经完成等价替代并通过回归。
- 不改变 WebUI dataset editor 的存储契约。

#### Dynamic-seq / compile 安全补丁

候选能力：

- sample preview 超出原 compile token range 时自动扩展编译范围
- repeated compile 时避免把已编译 wrapper 当作新的 base forward
- EasyControl two-stream compile 保留原始 inner source

候选路径：

```text
library/anima/models.py
library/anima/training.py
library/runtime/harness.py
networks/methods/easycontrol.py
tests/test_native_flatten.py
tests/test_runtime_harness_cli.py
tests/test_sample_token_budget.py
```

约束：

- 必须保护当前项目已有本地实现。
- 只接受经过测试证明的最小差异。

### P1：建议合并

#### Caption / Tagger 质量控制

候选能力：

- caption order correction
- duplicate caption trigger insertion 修复
- caption correction trigger
- caption variant sidecar
- tag group / taxonomy 增强

候选路径：

```text
library/captioning/correction.py
library/captioning/preprocess.py
library/captioning/tag_groups.py
library/captioning/tag_rules.py
library/captioning/taxonomy.py
library/preprocess/caption_variants.py
scripts/preprocess/correct_captions.py
scripts/anima_tagger/autotag.py
scripts/anima_tagger/autotag_server.py
scripts/anima_tagger/derive_groups.py
scripts/anima_tagger/build_english_tag_csv.py
tests/test_caption_correction.py
tests/test_caption_variant_sidecars.py
tests/test_caption_shuffle.py
tests/test_tag_groups.py
tests/test_tag_taxonomy.py
```

约束：

- 不引入 GUI caption editor。
- 不改变 WebUI dataset editor 的现有字段含义。
- sidecar 文件写入必须受路径边界约束。

#### LoRA Merge 干扰分析

候选能力：

- 多 LoRA merge 前的权重空间干扰分析
- CLI 合并脚本
- 后端分析工具，可供 WebUI 后续接入

候选路径：

```text
library/anima/merge.py
library/anima/merge_analysis.py
scripts/merge_loras.py
bench/lora_merge_interference/**
tests/test_merge_interference.py
```

约束：

- 不合并 GUI merge tab。
- 如需 WebUI 暴露，只做小型后端 API 或复用 `weight_analysis_service`。
- 不改变现有 `tasks.py merge` 的兼容边界，除非同时补测试和拒绝逻辑。

### P2：单独评估

#### Inference / FSG

候选能力：

- FSG correction
- resident inference server
- correction 组合能力增强

候选路径：

```text
library/inference/corrections/fsg.py
library/inference/corrections/dave.py
library/inference/args.py
library/inference/request.py
library/inference/generation.py
scripts/inference_server.py
tests/test_fsg_invariant.py
```

约束：

- 不直接替换当前 `inference.py`。
- 不改变 WebUI preview 默认推理行为。
- 先以可选 correction 接入，再评估服务化入口。

#### 打包资产与 uncond sidecar

候选能力：

- bundled `T5("")` uncond sidecar
- package-data 打包 `library/anima/assets/*.safetensors`

候选路径：

```text
library/anima/assets/_anima_uncond_te.safetensors
library/anima/uncond.py
library/preprocess/uncond.py
pyproject.toml
tests/test_inference_arg_closure.py
```

约束：

- 只有确认预处理或推理实际依赖该资产时才合并。
- 不为了资产合并而引入无关 inference 大改。

## 暂缓或拒绝内容

### 暂缓

- free-fit 全量替换 constant-token bucket
- 训练架构大规模迁移
- daemon 子系统大改
- release/calibration 脚本大规模迁移
- 大量 bench 目录迁移
- docs 树重排
- custom_nodes vendor sync

### 拒绝

- 上游 `web/**` 删除
- GUI 重构和 GUI 资源
- GUI 字体、theme、image tab、caption editor
- 上游删除当前 WebUI 测试
- 上游删除当前 WebUI 配置文件
- 直接覆盖当前 `configs/gui-methods/**`
- 直接覆盖当前生产默认 LoRA 配置为 REPA dog target / artist shard 默认值

## 阶段计划

### 阶段 0：基线与护栏

目标：建立可回归基线，确认当前项目状态。

任务：

1. 记录当前工作区状态：

   ```bash
   git status --short --branch
   git log -1 --oneline --decorate
   ```

2. 刷新同步源引用：

   ```bash
   git fetch upstream-snapshot main --prune
   ```

3. 确认禁入路径未被纳入候选 diff：

   ```bash
   git diff --name-status main..upstream-snapshot/main -- AGENTS.md gui web
   ```

4. 跑 WebUI 基线测试：

   ```bash
   timeout 60 python -m pytest tests/test_web_config_service.py
   timeout 60 python -m pytest tests/test_training_queue.py
   timeout 60 python -m pytest tests/test_preview_service.py
   timeout 60 python -m pytest tests/test_training_frontend_state.py
   timeout 60 python -m pytest tests/test_weight_analysis_service.py
   ```

5. 跑命令入口基线：

   ```bash
   timeout 60 .venv/bin/python tasks.py print-config METHOD=lora PRESET=default
   timeout 60 env METHOD=lora PRESET=default .venv/bin/python tasks.py print-config
   ```

验收：

- WebUI 五件套测试通过，或失败项被记录为既有问题。
- `tasks.py <command> KEY=value` 形式保持可用。
- 没有任何待执行计划需要修改 `AGENTS.md`、`gui/**` 或删除 `web/**`。

### 阶段 1：预处理健壮性切片

目标：移植低风险、高收益的预处理修复，不做 free-fit 全量迁移。

任务：

1. 从上游提取以下能力的最小实现：
   - 空 caption fallback
   - `use_repa` checkpoint fail-fast
   - PE auto-chain
   - load-skip pre-flight
   - cache reconcile
   - mask path scope
2. 对比当前 WebUI dataset/preflight/runtime config 依赖：
   - `web/services/config/preflight.py`
   - `web/services/config/datasets.py`
   - `web/services/training/runtime_config.py`
3. 补或移植最小测试：
   - `tests/test_preprocess_tasks.py`
   - `tests/test_preprocess_dataset.py`
   - `tests/test_te_cache_pruned_load.py`

验收：

```bash
timeout 60 python -m pytest tests/test_preprocess_tasks.py
timeout 60 python -m pytest tests/test_preprocess_dataset.py
timeout 60 python -m pytest tests/test_web_config_service.py
timeout 60 python -m pytest tests/test_training_queue.py
```

回滚点：

- 若 WebUI dataset editor 发生契约变化，回滚本阶段。
- 若 free-fit 改动无法和当前 bucket/native-flatten 共存，拆出单独 RFC，不在本阶段继续。

### 阶段 2：Dynamic-seq / compile 补丁审查

目标：确认并固化当前项目与同步源中的 compile 安全补丁。

任务：

1. 对比以下文件的当前项目、本地补丁、同步源差异：
   - `library/anima/models.py`
   - `library/anima/training.py`
   - `library/runtime/harness.py`
   - `networks/methods/easycontrol.py`
2. 只合并证明必要的最小差异。
3. 保留或补充回归测试：
   - `tests/test_native_flatten.py`
   - `tests/test_runtime_harness_cli.py`
   - `tests/test_sample_token_budget.py`

验收：

```bash
timeout 60 python -m pytest tests/test_native_flatten.py
timeout 60 python -m pytest tests/test_runtime_harness_cli.py
timeout 60 python -m pytest tests/test_sample_token_budget.py
```

回滚点：

- 若 compile cache 行为或 EasyControl two-stream 行为出现不确定性，保持当前项目实现，不合并上游版本。

### 阶段 3：Caption / Tagger 增强

目标：将 caption 清洗和变体侧车能力纳入预处理链路。

任务：

1. 移植 caption correction 核心：
   - `library/captioning/correction.py`
   - `library/captioning/preprocess.py`
2. 移植 sidecar 支撑：
   - `library/preprocess/caption_variants.py`
3. 移植 CLI：
   - `scripts/preprocess/correct_captions.py`
4. 检查 WebUI dataset editor：
   - 不新增 GUI image tab 功能。
   - 不改变已有 caption_source_mode 的语义。
   - 所有 sidecar 写入必须在项目允许路径内。

验收：

```bash
timeout 60 python -m pytest tests/test_caption_correction.py
timeout 60 python -m pytest tests/test_caption_variant_sidecars.py
timeout 60 python -m pytest tests/test_caption_shuffle.py
timeout 60 python -m pytest tests/test_web_config_service.py
```

回滚点：

- 若 sidecar 规则破坏现有 WebUI dataset 保存/加载，回滚 sidecar 接入，仅保留纯 correction 工具。

### 阶段 4：LoRA Merge 干扰分析

目标：合并后端分析能力，不引入 GUI merge tab。

任务：

1. 移植：
   - `library/anima/merge.py`
   - `library/anima/merge_analysis.py`
   - `scripts/merge_loras.py`
   - `tests/test_merge_interference.py`
2. 检查是否能复用：
   - `web/services/weight_analysis_service.py`
3. 如接入 WebUI，仅新增后端返回字段或独立 API；前端改动另开计划。

验收：

```bash
timeout 60 python -m pytest tests/test_merge_interference.py
timeout 60 python -m pytest tests/test_weight_analysis_service.py
timeout 60 python -m pytest tests/test_network_registry.py
```

回滚点：

- 若 merge 分析依赖上游已删除或重构的 network/config 路径，先保留 CLI 独立工具，不接入服务。

### 阶段 5：Inference / FSG 可选接入

目标：评估并接入 FSG correction，但保持当前 preview 默认行为稳定。

任务：

1. 移植：
   - `library/inference/corrections/fsg.py`
   - `tests/test_fsg_invariant.py`
2. 最小适配：
   - `library/inference/args.py`
   - `library/inference/request.py`
   - `library/inference/generation.py`
3. 不直接替换当前 `inference.py`。
4. 不默认启用 FSG。
5. 检查 WebUI preview：
   - `web/services/preview_service.py`

验收：

```bash
timeout 60 python -m pytest tests/test_fsg_invariant.py
timeout 60 python -m pytest tests/test_generation_request.py
timeout 60 python -m pytest tests/test_preview_service.py
```

回滚点：

- 若 FSG 需要大规模改写 generation/request 模型，暂缓本阶段。
- 若 preview 默认行为变化，回滚默认接入，只保留库级实验入口。

### 阶段 6：依赖与打包最小化

目标：只合并确有运行价值的依赖和 package-data。

任务：

1. 审查上游 `pyproject.toml` 差异。
2. 判断 `send2trash>=2.1.0` 是否只服务 GUI/curation 删除：
   - 如果只服务 GUI，不合并。
   - 如果被非 GUI 路径实际需要，再合并。
3. 判断 `_anima_uncond_te.safetensors` 是否必须打包：
   - 如果 FSG/preprocess 不依赖，不合并。
   - 如果依赖，加入 package-data 并补测试。
4. 不默认运行 `uv sync --frozen`；只有依赖实际变更后再提示或执行。

验收：

```bash
timeout 60 python -m pytest tests/test_smoke.py
timeout 60 python -m pytest tests/test_config.py
```

### 阶段 7：最终 WebUI 健全性回归

目标：确认核心能力合并后，当前 WebUI 仍可用。

任务：

1. 跑 WebUI 五件套：

   ```bash
   timeout 60 python -m pytest tests/test_web_config_service.py
   timeout 60 python -m pytest tests/test_training_queue.py
   timeout 60 python -m pytest tests/test_preview_service.py
   timeout 60 python -m pytest tests/test_training_frontend_state.py
   timeout 60 python -m pytest tests/test_weight_analysis_service.py
   ```

2. 跑核心配置和命令入口：

   ```bash
   timeout 60 python -m pytest tests/test_config.py tests/test_network_registry.py
   timeout 60 .venv/bin/python tasks.py print-config METHOD=lora PRESET=default
   ```

3. 检查禁入路径：

   ```bash
   git diff --name-only -- AGENTS.md gui web configs/web-ui-settings.toml configs/web-user-locks.toml
   ```

4. 检查空白/格式：

   ```bash
   git diff --check
   ```

验收：

- `AGENTS.md` 无本次新增改动。
- `gui/**` 无本次新增改动。
- `web/**` 未被上游删除或替换。
- WebUI 相关测试通过。
- `tasks.py print-config METHOD=lora PRESET=default` 继续可用。

## 实施策略

### 推荐分支

```bash
git switch -c codex/upstream-high-value-merge
```

### 推荐提交拆分

1. `preprocess: harden cache and caption preflight`
2. `training: preserve dynamic compile range safety`
3. `captioning: add correction and sidecar helpers`
4. `merge: add LoRA interference analysis`
5. `inference: add optional FSG correction`
6. `tests: preserve webui and command regression coverage`

每个提交必须做到：

- 不包含 `AGENTS.md`。
- 不包含 `gui/**`。
- 不删除 `web/**`。
- 有对应测试或明确说明无法低成本验证。

## 风险矩阵

| 风险 | 等级 | 缓解 |
| --- | --- | --- |
| 上游删除 `web/**` 被误合入 | 高 | 禁止整包 merge；每次提交前检查 `git diff --name-status -- web` |
| free-fit 全量迁移破坏当前 bucket/native-flatten | 高 | 阶段 1 只摘取健壮性修复，不做策略替换 |
| 上游默认 LoRA 实验配置覆盖本地生产默认 | 高 | 不直接覆盖 `configs/methods/lora.toml`，逐项审查 |
| caption sidecar 破坏 WebUI dataset editor | 中 | 先跑 `test_web_config_service.py`，必要时仅保留 CLI |
| FSG 接入改变 preview 默认行为 | 中 | FSG 默认关闭，preview 回归必须通过 |
| package-data 增加二进制资产影响打包 | 中 | 只有真实依赖时合并，补 smoke/config 测试 |
| 当前工作区已有用户改动被覆盖 | 高 | 阶段 0 记录状态；每次只改目标文件；不使用 destructive git 命令 |

## 决策点

实施前建议确认以下选择：

1. 是否接受阶段 1 不做 free-fit 全量迁移，只合并预处理健壮性修复。
2. 是否希望 LoRA merge 干扰分析只保留 CLI，还是同步接入 WebUI 后端。
3. 是否把 FSG 作为实验入口保留，还是接入正式 `inference.py` 参数。
4. 是否允许新增 package-data 二进制资产 `_anima_uncond_te.safetensors`。

## 完成定义

本轮合并完成必须同时满足：

- 已合并的内容均来自明确功能切片。
- `AGENTS.md` 未被本次修改。
- `gui/**` 未被本次修改。
- 当前 WebUI 未被删除、替换或结构性降级。
- WebUI 五件套测试通过，或所有失败都有明确既有原因。
- 相关核心测试通过。
- `git diff --check` 通过。
- 最终报告列明：
  - 合并了什么。
  - 明确没合并什么。
  - 跑了哪些测试。
  - 残余风险和后续建议。
