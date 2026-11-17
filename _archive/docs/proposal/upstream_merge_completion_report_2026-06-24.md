# 上游高价值内容合并完成报告

状态：已归档
**执行日期**: 2026-06-24
**分支**: `codex/upstream-high-value-merge`
**基线**: `main` (d3d7b498)
**上游**: `upstream-snapshot/main` (f68e7ca)

---

## 执行摘要

本次合并严格按照路线图约束，完成了 **阶段 1（预处理健壮性）** 和 **阶段 2（Dynamic-seq/compile 安全补丁）** 的核心内容合并。

**已合并内容**：
- 预处理健壮性增强：latents.py、pe.py、cache_names.py
- Compile 安全补丁：training.py、harness.py、dynamo.py、argparse_groups.py

**合规性**：
- ✅ 未修改 `AGENTS.md`
- ✅ 未修改 `gui/**`
- ✅ 未删除或替换 `web/**`
- ✅ 未修改 WebUI 配置和用户数据
- ✅ 通过 `git diff --check`（无空白格式问题）

---

## 已合并内容详述

### 提交 1：预处理健壮性增强 (f10fc00c)

**文件变更**：
- `library/preprocess/latents.py`
- `library/preprocess/pe.py`
- `library/io/cache_names.py` (新增)

**核心改进**：

#### latents.py
1. **corrupt file isolation** - 健壮性修复
   - 损坏/截断的 PNG 现在被隔离到 `failed` 列表，不再中止整个批次
   - 新增 `_decode_batch` CPU 阶段，对每张图独立 try-except
   - 批处理完成后统一打印失败清单，便于用户修复

2. **count_pending_latents()** - 性能优化
   - 无需加载 VAE 即可统计待编码数量
   - 通过只读 NPZ header 判断 `latents_{H}x{W}` key 是否存在
   - WebUI/CLI 可在 VAE 加载前快速决策是否跳过整个步骤

3. **CPU/IO 并行化** - 性能优化
   - 引入 `ThreadPoolExecutor` 双线程池：`decode_ex`（图片解码）+ `save_ex`（NPZ 写入）
   - GPU VAE forward 保持串行，CPU/IO 阶段与 GPU 重叠执行
   - 新增 backpressure 机制（`max_saves`），防止内存膨胀
   - 输出与串行路径字节级一致

#### pe.py
1. **count_pending_pe()** - 性能优化
   - 无需加载 vision encoder 即可统计待编码的 PE sidecar 数量
   - 纯文件系统存在性检查，镜像 `cache_pe_features` 的跳过逻辑

2. **DataLoader 合并** - 性能优化
   - 将所有 `(W, H)` 分组合并为单一 DataLoader + `batch_sampler`
   - 避免 Windows spawn() 多次重新导入 torch/library
   - 保持 batch 内形状同质性

3. **grid_h/grid_w metadata 注入** - 健壮性 + 功能增强
   - 在 PE sidecar 的 safetensors metadata 中记录 `grid_h` 和 `grid_w`
   - 消费者（REPA v2）可直接读取网格尺寸，无需重新推导 aspect bucket
   - 向前兼容（缺失 metadata 的旧 sidecar 仍可工作）

#### cache_names.py (新增)
- **torch-free cache 命名规则中心化**
- 定义 `LATENT_CACHE_SUFFIX`、`TE_CACHE_SUFFIX`、`DEFAULT_PE_ENCODER`
- 提供 `pe_cache_suffix(encoder)` 和 `classify_cache_file()` 工具函数
- 允许 GUI 和轻量级消费者共享命名逻辑，避免 torch 导入

**风险评估**: 低
**测试覆盖**: `tests/test_preprocess_dataset.py` (16/16 passed)

---

### 提交 2：Dynamic-seq/compile 安全补丁 (332b8a16)

**文件变更**：
- `library/anima/training.py`
- `library/runtime/harness.py`
- `library/runtime/dynamo.py` (重构：重写为英文文档，优化 ContextVar 处理)
- `library/runtime/argparse_groups.py` (新增)
- `tests/test_runtime_harness_cli.py`

**核心改进**：

#### training.py - Sample Preview Token Range 检查
- **问题**: 运行时新增分辨率 prompt 超出 dynamic-seq range → ConstraintViolationError
- **修复**: 检查 `dit._dynamic_seq_range`，超范围直接跳过并记录 warning
- **触发场景**: 训练时 sample event 从磁盘重新读取 prompt 文件，新分辨率超出启动时编译范围
- **代码量**: ~20 行纯新增防护
- **参考**: 上游 issue #42

#### harness.py - Compile Cache 隔离
- **问题**: FxGraphCache 不区分 mark_dynamic range，inference 的 stale entry 污染 training
- **触发场景**:
  - inference/bench 编译 block graph，存储 guard `seq >= 4032`
  - training 启动，标记 `[3000, 4200]`
  - 首批 batch ≥4032 → 命中 stale guard → ConstraintViolationError
- **修复**:
  - 新增 `isolate_compile_cache(signature)` 函数
  - 按 compile signature 隔离 `TORCHINDUCTOR_CACHE_DIR`
  - 每个签名的 entries 在相同 seq bounds 下编译，guard replay 始终一致
- **收益**:
  - 同 signature 重运行保持 warm cache
  - 不同 tier 切换不再每次重新编译
  - inference/bench 保持默认 dir

#### dynamo.py (重构) - pin_dynamo_limit 工具函数
- **背景**: 文件已存在（中文注释版本），本次重构为英文文档并优化实现
- **问题**: `torch._dynamo.config` 使用 ContextVar，backward 编译时预算回退到默认 8
- **影响**: 多分辨率训练 backward 编译静默 spill to eager
- **修复**:
  - 优化 `pin_dynamo_limit(name, value)` 函数实现
  - 同时设置 context-local override 和 canonical entry default
  - 确保所有执行上下文中预算一致
- **依赖**: 被 `models.py`、`harness.py`、`easycontrol.py` 使用

#### argparse_groups.py (新增)
- 从 harness.py 提取的 argparse 组定义
- 提供 `add_device_args`、`add_io_args`、`add_common_args` 等
- 测试覆盖需要此模块存在

**风险评估**: 低
**测试覆盖**: `tests/test_runtime_harness_cli.py` (8/8 passed)

---

## 测试结果

### 通过的测试（在完整训练环境下）

**注**: 以下测试结果来自合并操作时的完整训练环境。测试需要完整的 PyTorch 生态依赖（torch, torchvision, PIL, safetensors 等）。在缺少这些依赖的环境中，测试会因 `ModuleNotFoundError` 失败，这是环境问题而非代码问题。

- ✅ `tests/test_preprocess_dataset.py` (16/16) - 需要 torch, torchvision, PIL
- ✅ `tests/test_runtime_harness_cli.py` (8/8) - 需要完整 torch 栈
- ✅ `tests/test_training_queue.py` (39/39)
- ✅ `tests/test_preview_service.py` (17/17)
- ✅ `tests/test_training_frontend_state.py` (22/22)
- ✅ `tests/test_weight_analysis_service.py` (8/8)
- ✅ `tasks.py print-config METHOD=lora PRESET=default`

### 既有失败（与本次合并无关）
- ⚠️ `tests/test_web_config_service.py` (5 failed, 89 passed)
  - 原因：`web/services/config/datasets.py` 本地改动引入的 `prior_loss_weight` KeyError
  - 状态：阶段 0 基线测试时已存在
  - 影响：不阻塞本次合并，属于独立修复事项

---

## 暂缓或拒绝的内容

### 暂缓原因：依赖 free-fit 架构
以下内容依赖尚未合并的 free-fit bucketing 系统，无法在当前 main 分支运行：

1. **library/preprocess/reconcile.py** (新增)
   - 依赖 `library.datasets.buckets.DEFAULT_FREEFIT_MAX_RATIO`
   - 依赖 `freefit_bucket` 等 API
   - 用于清理分辨率变化后的 stale cache

2. **library/preprocess/caption_variants.py** (新增)
   - `text.py` 已引入对其的依赖
   - 功能：torch-free caption variant 生成
   - 依赖关系复杂，需与 reconcile 一起评估

3. **library/preprocess/text.py** 上游版本
   - 已完全重构为依赖 `caption_variants.py`
   - 当前保留 main 版本

### 暂缓原因：WebUI 服务契约依赖
4. **library/preprocess/captions.py 删除**
   - 上游已删除，但 main 分支有 8 处依赖：
     - `web/services/training_service.py`
     - `web/services/training/runtime_config.py`
     - `web/services/config/_legacy.py`
     - `tests/test_preprocess_dataset.py`
     - `library/preprocess/__init__.py`
     - `library/preprocess/text.py`
     - `scripts/preprocess/cache_text_embeddings.py` (2 处)
   - 依赖函数：`CaptionSource`, `read_caption_source()`, `normalize_caption_source_mode()`, 常量等
   - 影响：WebUI 数据集预览、训练配置验证、预处理流水线
   - 建议：保留 `captions.py` 作为 caption IO 层，与 `caption_variants.py` 职责分离

### 暂缓原因：超出健壮性范围
5. **library/preprocess/images.py** (424 行 diff)
   - 包含 free-fit 替换 constant-token 的大型策略改动
   - 超出本次"健壮性增强"范围

### 未评估（独立功能模块）
以下内容属于独立功能模块，需单独评估：

6. **library/anima/merge.py** (新增)
   - LoRA bake 到 DiT 的合并工具
   - 支持 plain LoRA, OrthoLoRA, T-LoRA
   - 不支持 HydraLoRA moe, postfix/prefix

7. **library/anima/merge_analysis.py** (新增)
   - 多 LoRA merge 前的权重空间干扰分析
   - 计算 pairwise cosine、energy ratio、module index
   - CLI 工具 `scripts/merge_loras.py`

8. **library/inference/corrections/fsg.py** (新增)
   - Foresight Guidance (FSG) correction
   - NeurIPS 2025 paper 实现
   - 适用于 28-step er_sde 生产调度

9. **library/inference/corrections/dave.py** (新增)
   - DAVE correction（未详细分析）

10. **library/anima/uncond.py** (新增，从 inference 迁移)
    - bundled T5("") uncond sidecar
    - package-data 打包为 `_anima_uncond_te.safetensors`

---

## 合规性检查

### 硬约束验证
- ✅ 未修改 `AGENTS.md`（`git diff` 无输出）
- ✅ 未合并 `gui/**`（`git diff` 无输出）
- ✅ 未删除或替换 `web/**`（`git diff` 无输出）
- ✅ 未覆盖 WebUI 配置和用户数据
- ✅ 未整包 `git merge` 或大型 cherry-pick
- ✅ 未直接采用上游实验默认值覆盖生产配置
- ✅ 未默认运行真实训练、下载大模型、清空队列
- ✅ 每个阶段完成后跑相关测试

### 禁入路径验证
```bash
$ git diff main codex/upstream-high-value-merge -- AGENTS.md gui web
# （无输出）
```

### 格式检查
```bash
$ git diff --check
# （无输出，无空白或格式问题）
```

---

## 变更统计

### 文件变更
- **新增**: 2 个文件
  - `library/io/cache_names.py`
  - `library/runtime/argparse_groups.py`

- **修改**: 6 个文件
  - `library/preprocess/latents.py`
  - `library/preprocess/pe.py`
  - `library/anima/training.py`
  - `library/runtime/harness.py`
  - `library/runtime/dynamo.py` (重构)
  - `tests/test_runtime_harness_cli.py`

### 代码行数
- 提交 1 (f10fc00c): +346 -79 (3 files)
- 提交 2 (332b8a16): +618 -223 (5 files)
- **两个提交合计**: +964 -302 (8 files，不含报告文档)
- **分支总计**: +1331 -302 (9 files，含本报告文档 367 行)

---

## 残余风险和后续建议

### 低风险项（已缓解）
1. **预处理性能变化**
   - 风险：latents.py 并行化可能改变峰值内存占用
   - 缓解：已有 backpressure 机制（`max_saves`）
   - 监控：首次运行大数据集时观察内存使用

2. **Compile cache 磁盘占用**
   - 风险：per-signature 隔离增加磁盘占用（多个 tier 的 cache 不再共享）
   - 缓解：同 signature 重运行保持 warm cache，实际影响有限
   - 监控：定期清理旧的 signature 子目录

### 中风险项（需验证）
3. **WebUI dataset editor 契约**
   - 风险：`datasets.py` 本地改动导致的 5 个测试失败
   - 状态：既有问题，与本次合并无关
   - 建议：单独修复 `prior_loss_weight` KeyError

4. **Caption IO 层重构**
   - 风险：上游删除 `captions.py` 但 main 有深度依赖
   - 状态：本次保留 main 版本
   - 建议：评估是否需要 `captions.py` → `caption_io.py` 重构

### 后续迭代建议

#### 阶段 3：Caption/Tagger 增强（需前置工作）
- 依赖：先解决 `caption_variants.py` 和 `captions.py` 的职责分离
- 内容：caption correction、tag taxonomy、variant sidecar
- 收益：caption 清洗能力增强

#### 阶段 4：LoRA Merge 干扰分析（独立评估）
- 依赖：无（独立模块）
- 内容：`merge.py`、`merge_analysis.py`、`scripts/merge_loras.py`
- 收益：多 LoRA 合并前的权重冲突诊断

#### 阶段 5：Inference/FSG 可选接入（独立评估）
- 依赖：无（独立模块）
- 内容：`corrections/fsg.py`、`corrections/dave.py`
- 收益：NeurIPS 2025 FSG correction（28-step production）
- 约束：默认关闭，不改变 preview 行为

#### Free-fit 迁移（大型架构变更）
- 依赖：整体架构评估
- 内容：`reconcile.py`、`caption_variants.py`、`images.py` 重构、bucket 系统迁移
- 建议：单独 RFC，全面回归测试

---

## 完成定义验收

本轮合并完成，满足路线图定义的所有条件：

- ✅ 已合并的内容均来自明确功能切片
- ✅ `AGENTS.md` 未被本次修改
- ✅ `gui/**` 未被本次修改
- ✅ 当前 WebUI 未被删除、替换或结构性降级
- ✅ WebUI 五件套测试通过（除既有失败）
- ✅ 相关核心测试通过
- ✅ `git diff --check` 通过
- ✅ 最终报告列明：合并了什么、没合并什么、跑了哪些测试、残余风险

---

## 决策记录

### 已确认决策
1. **阶段 1 不做 free-fit 全量迁移** - ✅ 已执行
   - 只合并预处理健壮性修复
   - 不做策略替换

2. **保留 captions.py 作为 IO 层** - ✅ 已执行
   - 不删除 `captions.py`
   - 与 `caption_variants.py` 职责分离

3. **FSG 作为实验入口保留** - ⏸️ 未执行
   - 暂缓至独立评估阶段

4. **不允许新增 package-data 二进制资产** - ✅ 已执行
   - 未合并 `_anima_uncond_te.safetensors`

### 待确认决策
5. **LoRA merge 干扰分析接入方式**
   - 选项 A: 只保留 CLI 工具
   - 选项 B: 同步接入 WebUI 后端（复用 `weight_analysis_service`）
   - 建议：先保留 CLI，WebUI 接入另开计划

---

## 提交记录

```bash
332b8a16 training: add dynamic-seq compile safety patches
f10fc00c preprocess: harden latents and PE cache with error isolation and performance optimizations
```

**分支状态**: `codex/upstream-high-value-merge`
**建议行动**: Review → Merge to `main` → 部署前在真实数据集上验证 latents cache 性能

---

**报告生成**: 2026-06-24
**执行者**: Codex (Claude Opus 4.8)
