# Contribution Priorities

状态：维护待办

适用版本：当前 `dev` / `main`；开始任务前必须用实时源码和 Issue 重新确认

PR 规范：[`../CONTRIBUTING.md`](../CONTRIBUTING.md)

本页承接原先混在 `CONTRIBUTING.md` 开头的项目待办。它记录值得推进的方向、已有入口和验收设想，**不是当前实现的事实来源，也不会降低 PR 验收要求**。多模型族（Anima、Krea-2、Z-Image、Qwen Image 2.1、MOSS-Image）与流水线并行的当前实现边界以 [`multi_model_support.md`](multi_model_support.md) 为准。认领较大任务前，请先在 Issue 中确认范围和优先级。

## 当前概览

| 方向 | 当前入口 | 主要缺口 | 建议等级 |
| --- | --- | --- | --- |
| EasyControl adapters | [`experimental/easycontrol.md`](experimental/easycontrol.md) | adapter、数据规范、toy dataset、控制保真评估 | Tier 1 / 1.5 |
| DP-DMD / Turbo | [`experimental/dpdmd.md`](experimental/dpdmd.md) | 分阶段实证、完整 bench、组合验证 | Tier 1.5 / 2 |
| DCW calibration | [`methods/dcw.md`](methods/dcw.md) | σ̂²、tiled、CFG、Spectrum、per-LoRA 校准 | Tier 1 / 1.5 / 2 |
| Bench 基础设施 | [`../bench/`](../bench/) | README、标准结果 envelope、基线解释 | Tier 1 |
| 翻译与本地化 | [`README.md`](README.md) | WebUI 文案、帮助信息、结构图和索引 | Tier 1 |

等级的正式定义和交付要求以 [`../CONTRIBUTING.md`](../CONTRIBUTING.md) 为准。

## 1. EasyControl adapters

EasyControl 在 frozen DiT 的 self-attention 和 FFN 上训练条件 LoRA 与门控。方法实现已经有实验入口，当前更适合由彼此独立的控制类型贡献补齐生态。

- **训练好的 adapter**：canny、depth、pose、lineart、scribble、segmentation 等。每个控制类型单独提交，附模型卡、训练配置、样例和可复现实证。计划中的 Hugging Face collection 名称必须在发布前由维护者确认。*[Tier 1.5]*
- **按任务划分的数据规范**：每个控制类型说明 pair 格式、许可、信号图生成方式、建议规模和数据质量检查。原设想约为 2k pairs，但提交前应通过 Issue 确认。*[Tier 1]*
- **Toy dataset**：为每个控制类型提供约 200 对、许可清晰的小数据集，使贡献者能先验证流程。不要把大数据或权重直接提交到仓库。*[Tier 1]*
- **单命令训练入口**：在现有 [`configs/methods/easycontrol.toml`](../configs/methods/easycontrol.toml)、[`configs/gui-methods/`](../configs/gui-methods/) 和 [`tasks.py`](../tasks.py) 约定内增加按任务配置及入口，例如 canny/depth；不要新建另一套任务系统。*[Tier 1]*
- **控制保真评估**：为约 100 对 held-out 数据重新提取生成结果中的控制信号，并与输入信号比较；报告指标、基线和失败样例，而不只提供主观图片。当前 [`bench/easycontrol/`](../bench/easycontrol/) 只有等价性和 smoke 探针。*[Tier 1.5]*

## 2. DP-DMD / Turbo 少步蒸馏

目标是把 28-step、CFG=4 的 Anima teacher 蒸馏为 4–8 step generator，并验证 turbo adapter 与普通 concept LoRA 的组合能力。当前 DP-DMD 实现说明见 [`experimental/dpdmd.md`](experimental/dpdmd.md)，早期设计背景见归档 [`_archive/docs/proposal/turbo_anima_dmd_lora.md`](../_archive/docs/proposal/turbo_anima_dmd_lora.md) 与 [`_archive/docs/proposal/dpdmd.md`](../_archive/docs/proposal/dpdmd.md)。

原待办写于实现前。当前仓库已经具有：

- `exp-turbo`、`exp-turbo-prep`、`exp-test-turbo` 任务入口；
- [`scripts/distill_turbo/`](../scripts/distill_turbo/) 训练实现；
- [`configs/methods/turbo.toml`](../configs/methods/turbo.toml)；
- [`bench/dpdmd/probe_first_step_anchor.py`](../bench/dpdmd/probe_first_step_anchor.py)。

这些入口存在不等于阶段验收已经完成。尚待用标准结果目录和对照实验回答：

1. **Phase 0：单 prompt overfit**。固定 seed，对比 teacher@28 与 student@4，记录配置、曲线和图片；原计划为 batch 1、约 2k iterations。*[Tier 2 方法验证]*
2. **Phase 1：100 prompt sweep**。报告 Image Reward、HPS v2.1 和 aspect breakdown（1024²、832×1248、1248×832）；原门槛为 student ≥ teacher 的 80%，且任一 aspect 不低于 60%。*[Tier 2 延续]*
3. **Phase 2：完整 HPS bench**。约 1k COCO prompts，覆盖设计文档中的 schedule ablation，并保存标准 `result.json`。*[Phase 1 落地后可作为 Tier 1.5 实证补充]*
4. **Phase 3：组合测试**。对比 turbo only、concept LoRA@28、turbo + concept LoRA@4，至少覆盖三个 concept checkpoint。*[Tier 1.5]*

保留原停止规则：Phase 1 在一次合理的 rank 调整后仍失败，就记录失败结论并停止扩大实验，不无限追加训练成本。

## 3. DCW calibration

DCW v4 已有训练和推理入口；当前开放问题集中在校准覆盖面。详细状态和限制以 [`methods/dcw.md`](methods/dcw.md) 为准。

- **σ̂² channel 三 seed 重训**：用多 seed pool 重新训练 fusion head 并复测 Gate B；若仍失败，记录结论并把 shrinkage-off 作为稳定策略。*[Tier 1.5]*
- **Tiled inference**：比较全局 `c_pool` / `g_obs` 广播与继续明确 no-op 两种策略；任何实现都必须说明 tile 边界语义。*[Tier 1.5]*
- **CFG drift**：补 CFG=1 / CFG=7 校准和选择策略，特别检查 CFG=1 与 direction sign-flip 的交互。新的校准池按实证型方法贡献审核。*[Tier 2-shaped]*
- **Cached-Spectrum `x0_pred` ablation**：增加一行可复现 bench，确认 Chebyshev forecaster 误差下的校正行为。*[Tier 1.5]*
- **Per-LoRA fusion heads**：为重要发布 checkpoint 生成并发布对应 `<lora_name>.fusion_head.safetensors`，记录生成配置和兼容范围。*[Tier 1]*
- **`scripts/dcw/` 帮助文本**：改善脚本 docstring、`--help` 和可执行示例。*[Tier 1]*

## 4. Bench 基础设施缺口

实时目录以 [`../bench/`](../bench/) 为准。当前直接可认领的缺口包括：

| 目录 | 当前事实 | 待办 |
| --- | --- | --- |
| [`bench/dcw/`](../bench/dcw/) | 有多组分析脚本，无 README | 说明每个脚本、标准运行命令、输出布局、代表性结果和解读 |
| [`bench/easycontrol/`](../bench/easycontrol/) | 有 equivalence/smoke 脚本，无 README | 增加 README，并为控制保真 harness 预留明确入口 |
| [`bench/dpdmd/`](../bench/dpdmd/) | 仅有 first-step anchor 探针，无 README | 说明当前探针边界，并逐步承接 Turbo 分阶段结果 |

原待办还提到 `bench/fera/`、`bench/hydralora/` 和 `bench/spectrum/`，但这些目录当前不存在，不能继续描述为“只缺 README”。若要恢复，先在 Issue 中确认应从历史重建还是由当前实现建立新 bench。现有 README 形状可参考 [`bench/spd/README.md`](../bench/spd/README.md) 和 [`bench/spectrum_pareto/README.md`](../bench/spectrum_pareto/README.md)。

通用缺口：

- 旧 bench 逐步接入 [`bench/_common.py`](../bench/_common.py) 的 `make_run_dir` / `write_result`，生成带 script、git SHA、环境、参数、指标和 artifact 的 `result.json`。
- 加载 DiT 的新 bench 使用 [`bench/_anima.py`](../bench/_anima.py)，保持 adapter apply/load 后再 compile 的顺序。
- 每次迁移只处理一个 bench，保留原始结果语义，并提供迁移前后可比的运行命令。*[Tier 1]*

## 5. 翻译与本地化

- **WebUI 文案和帮助**：修改 `web/frontend-next/src/features/training-config/domain/` 或对应 Next feature 模块中的现有字符串来源，不建立平行的桌面 GUI 翻译表。LoRA、MoE、σ-bucket、VAE 等技术术语应保持可识别。
- **文档和结构图**：英文基准结构图位于 `docs/structure_images/`，其他语言可使用 `docs/structure_images_<lang>/` sibling tree；提交时列出引用图片的 Markdown 文件并更新索引。
- **翻译文档入口**：新增 `<name>.<code>.md` sibling 时，必须说明如何从 [`docs/README.md`](README.md) 或分区索引发现它。
- **规范文件**：不要建立翻译后独立演化的 `AGENTS.md`。仓库级开发约束始终以根 [`../AGENTS.md`](../AGENTS.md) 为准。

翻译改动属于 Tier 1；若影响布局或交互，应在浏览器中检查对应流程，并在 PR 中附截图或说明无法截图的原因。

## 维护本页

- 开始任务前，以实时代码、配置、测试和相关方法文档核实状态。
- 任务落地后，在同一 PR 更新对应条目：标记完成、链接 Issue/PR/结果目录，或写明失败结论。
- 只有优先级或范围改变时才更新本页；具体 PR 验收规则只写在 [`../CONTRIBUTING.md`](../CONTRIBUTING.md)。
