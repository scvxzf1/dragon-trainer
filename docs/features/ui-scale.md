# UI 缩放

状态：稳定
适用版本：当前 WebUI Next 工作台 `/next/settings`
相关代码：`web/frontend-next/src/features/settings/SettingsPage.tsx`、`settingsForm.ts`、`web/frontend-next/src/app/uiPreferences.ts`、`web/services/settings_service.py`

---

## 1. 缩放机制

WebUI 提供两级百分比缩放（有效范围 `25`–`400`，推荐 `75`–`150`，默认 `100`）：

1. **全局默认缩放（`ui_scale`）**：全站所有页面的基准比例，在 `/next/settings`「界面显示」中通过快捷按钮（`80%`–`125%`）、滑块或数字框调整。
2. **分页面独立缩放**：在 `/next/settings`「分页面缩放」平铺数字输入框中为特定页面填写百分比；**输入框留空即自动跟随全局默认比例**，无需额外勾选复选框。修改后点击 **保存设置** 持久化到 `web-ui-settings.toml`。

---

## 2. 路由覆盖范围

- **支持独立缩放的主页面与辅助工具页**：
  - 训练配置（`/next/training` → `ui_scale_config`）
  - 数据集蓝图（`/next/datasets` → `ui_scale_datasets`）
  - 当前监控（`/next/monitor` → `ui_scale_training`）
  - 模型配置（`/next/models` → `ui_scale_model_config`）
  - 全局设置（`/next/settings` → `ui_scale_settings`）
  - 权重分析（`/next/weight-analysis` → `ui_scale_weight_analysis`）
  - 生图测试（`/next/image-test` → `ui_scale_image_test`）
  - 环境检测（`/next/environment` → `ui_scale_environment`）
- **历史任务与详情子视图（`/next/history`）**：
  - 概览（默认 → `ui_scale_history_overview`）
  - 指标（`?view=metrics` → `ui_scale_history_analysis`）
  - 产物（`?view=artifacts` → `ui_scale_history_preview`）
  - 日志（`?view=logs` → `ui_scale_history_logs`）
  - 配置（`?view=config` → `ui_scale_history_config_files`）
- **始终跟随全局 `ui_scale` 的页面**：训练队列（`/next/queue`）、预览工作区（`/next/preview`）、打标工作台（`/next/captioning`）、数据管理（`/next/data`）。

---

## 3. 注意事项

- 极端比例（接近 `25%` 或 `400%`）会影响密集表单与图表的可读性，日常建议保持在 `75%`–`150%`。
- 若仅单个页面字号异常，检查「分页面缩放」对应输入框是否残留了历史独立数值，清空并保存即可恢复跟随全局。

---

## 4. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest tests/test_ui_scale_settings.py -q
.venv/bin/python tasks.py web-next-e2e -- scale-keyboard.spec.ts
```
