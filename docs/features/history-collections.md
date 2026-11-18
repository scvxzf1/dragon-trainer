# 历史任务与集合

状态：稳定
适用版本：当前 WebUI Next 工作台 `/next/history`
入口命令：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

相关代码：
- `web/frontend-next/src/features/training-history/`（`HistoryPage.tsx`、`HistoryOverview.tsx`、`HistoryAssets.tsx`、`HistoryResume.tsx`）
- `web/services/training/`（历史列表、详情、删除、续训与配置指纹提取 `history_config_chips.py`）
- `configs/web-training-history/`（随外置配置根或 `history_root` 迁移）

---

## 1. 核心能力与布局

`/next/history` 管理已执行或中断的训练与预处理记录（监控页右上角“最近训练”抽屉也可原地预览最近 20 条记录）：

- **集合与布局切换**：左侧按 **集合（collection / collections）** 与配置分组导航；列表工具栏支持 **按配置堆叠（`stack`）** 与 **平铺任务（`list`）** 两种展示方式。
- **搜索与多维筛选**：服务端关键词搜索支持任务名、配置名、路径及 `组:<名称>`、`配置:<名称>` 语法；已加载列表支持按状态、归档状态、来源（队列/续训/热启动）及快照推导的配置指纹（训练变体、预处理精度、块交换精度、底模计算路径 `base_compute`、精度倾向）筛选与排序。
- **批量管理、对比与合并分析**：勾选任务后可批量归档/取消归档、修改集合、发起 **对比记录（2–4 条）**、进入 **合并训练视图（`/next/history/aggregate`）**，或执行分级删除。

---

## 2. 任务详情与动作边界

点击任一任务进入详情页，包含五个子视图：

1. **概览（`HistoryOverview`）**：展示执行结果状态条、3 项核心训练摘要（末次有效 Loss、实际目标步数、运行时长）、最新样张缩略图与产物统计、从 `config.snapshot.toml` 提取的 `HistoryFingerprint` 标签、来源与结果双栏，以及默认折叠的「诊断信息与完整路径」。
2. **指标**：Loss + 学习率双轴合并曲线、CMMD 曲线，以及支持按 `0–2000` 等半开采样区间分页拉取（`/api/training/history/{task_id}/system`）的 GPU 显存/利用率/温度曲线。
3. **产物（`HistoryAssets`）**：展示训练样张（支持最新优先或按 Prompt/Seed/参考图演进分组，点击打开左右分栏可缩放样张详情弹窗）、checkpoint 权重列表（显示精确步数对应的训练 Loss、`FINAL MODEL` 标记、公共父目录复制、下载及一键应用权重热启动到训练草稿）与结果文件卡片。
4. **日志**：稀疏索引虚拟滚动与服务端全文搜索，详见 [历史任务日志](history-logs.md)。
5. **配置**：只读展示可搜索、可复制的冻结 `config.snapshot.toml` 快照。

### 续训与以此配置新建训练的区别
- **原地断点续训（`HistoryResume`）**：在历史任务详情内直接选择该任务的 checkpoint 状态包（含 PP 任务的 `pp_runs/<run>-step<N>` 包），可调整追加步数（`duration_overrides.max_train_steps`）后立即启动或加入队列。
- **以此配置新建训练**：从该任务的 `config.snapshot.toml` 另存一份新项目配置文件并跳转到 `/next/training` 编辑，不覆盖原配置，也不自动启动训练。

---

## 3. 删除与风险边界

历史删除明确区分两档操作：
- **仅删除历史记录**：只移除 `web-training-history` 中的元数据卡片，保留磁盘运行目录与权重产物；
- **删除记录及运行目录**：同时永久清理关联运行目录（严格受全局 `output_root` / `resolve_output_root()` 边界约束；若挂载了预处理共享缓存池软链，只拆链接不删共享池，见 [预处理缓存复用](preprocess-cache-reuse.md)）。

---

## 4. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_history_list.py \
  tests/test_training_history_delete.py \
  tests/test_training_history_artifacts.py \
  tests/test_training_history_timeline.py \
  tests/test_web_history_system_page.py \
  tests/test_training_resume_options.py \
  tests/test_training_resume_actions.py \
  -q
```
