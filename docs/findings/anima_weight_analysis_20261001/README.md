# Anima 权重静态调查归档（2026-10-01）

状态：历史调查快照
适用范围：2026-10-01 本批静态权重分析

本目录保存已完成的历史调查快照：画风10文件、人物5文件，共15文件、5,700模块（4,132 + 1,568）。仅归档报告、统计、图表、脚本与验证证据；未复制原始权重，未重新运行分析。

- [两组摘要与可比性边界](summary.md)
- [画风调查](style/report.md)
- [人物调查与人物 vs 画风对照](character/report.md)
- [文件来源、复制哈希与修改记录](manifest.json)

脚本为原始取证快照，包含当时的路径约定；复现须复制到新的分析工作目录并调整输入与对照目录，见各报告的复现说明。CSV、JSON、图表、证据与脚本均逐字节复制；报告只调整导航、快照状态和复现说明。

## 画风文件清单

- [style/data/aggregate.json](style/data/aggregate.json)
- [style/scripts/analysis.py](style/scripts/analysis.py)
- [style/figures/block_heatmap.png](style/figures/block_heatmap.png)
- [style/figures/common_targets_heatmap.png](style/figures/common_targets_heatmap.png)
- [style/evidence/independent_validation.json](style/evidence/independent_validation.json)
- [style/figures/module_heatmap.png](style/figures/module_heatmap.png)
- [style/evidence/partial.json](style/evidence/partial.json)
- [style/data/per_file.csv](style/data/per_file.csv)
- [style/data/per_layer.csv](style/data/per_layer.csv)
- `style/evidence/progress.log`（运行过程日志，按 `.gitignore` 未入库）
- [style/report.md](style/report.md)
- [style/evidence/validation.json](style/evidence/validation.json)

## 人物文件清单

- [character/data/aggregate.json](character/data/aggregate.json)
- [character/scripts/analysis.py](character/scripts/analysis.py)
- [character/figures/block_cosine.png](character/figures/block_cosine.png)
- [character/figures/block_heatmap.png](character/figures/block_heatmap.png)
- [character/data/common_target_summary.csv](character/data/common_target_summary.csv)
- [character/figures/comparison_blocks.png](character/figures/comparison_blocks.png)
- [character/evidence/independent_validation.json](character/evidence/independent_validation.json)
- [character/figures/module_heatmap.png](character/figures/module_heatmap.png)
- [character/evidence/partial.json](character/evidence/partial.json)
- [character/data/per_file.csv](character/data/per_file.csv)
- [character/data/per_layer.csv](character/data/per_layer.csv)
- `character/evidence/progress.log`（运行过程日志，按 `.gitignore` 未入库）
- [character/report.md](character/report.md)
- [character/evidence/validation.json](character/evidence/validation.json)
