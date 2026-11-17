# 训练队列

状态：稳定
适用版本：当前 WebUI Next 工作台 `/next/queue`
入口命令：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

相关代码：
- `web/frontend-next/src/features/training-queue/`（`QueuePage.tsx`、`QueuePolicyForm.tsx`）
- `web/services/training/queue.py`、`launcher.py`
- `configs/web-training-queue/`（随外置配置根或 `queue_root` 迁移）

---

## 1. 页面布局与功能

`/next/queue` 采用**左侧队列条目列表 + 右侧队列策略面板（`QueuePolicyForm`）**双栏独立页面布局，按顺序调度从 `/next/training` 或历史续训入队的训练任务：

- **顶部状态筛选与平铺批量操作**：按状态统计筛选队列项，直接提供 `暂停 / 恢复`、`中止后续队列`、`强制中止`、`取消全部等待`、`清理已完成`、`清理已取消` 操作按钮。
- **入队配置冻结**：条目入队时即生成独立的冻结 runtime 配置（及关联数据集快照与当次风险确认标记）；后续在配置工作台修改原项目 TOML 不会改变已在队列中等待的条目。
- **队列策略与自动重试（`QueuePolicyForm`）**：
  - **失败策略**：`pause`（任务失败后暂停队列等待人工排查）或 `continue`（失败后继续启动下一条）；
  - **自动重试**：可配置 `auto_retry` 开关、最大尝试次数 `max_attempts` 与退避间隔 `retry_backoff_sec`。
- **并发版本保护**：批量变更与排序携带 `expected_revision`（及停止运行任务的 `task_id` 绑定），若队列状态已变化会返回 409 并要求刷新后再操作，防止误杀刚切换的新任务。

---

## 2. 风险与注意事项

- **强制中止**：会中断当前运行中的训练进程，未落盘步数的中间状态不会保存。
- **失败后继续下一个（`continue`）**：前置任务因配置或环境故障失败时不会停队，后续条目会继续占用 GPU 启动。
- **清理已完成 / 已取消**：仅清理队列记录文件，不会删除 `/next/history` 中的历史记录或 `output/runs/` 下的训练权重。

---

## 3. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_queue.py \
  tests/test_training_queue_resume.py \
  tests/test_training_queue_retry_wake.py \
  tests/test_queue_revision_contract.py \
  tests/test_queue_item_retry_override.py \
  -q
```
