# 全局设置

状态：稳定
适用版本：当前 WebUI Next 工作台 `/next/settings`
入口命令：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

相关代码：
- `web/frontend-next/src/features/settings/SettingsPage.tsx`、`settingsForm.ts`
- `web/services/settings_service.py`
- `library/env.py`（外置配置根与路径解析）

---

## 1. 用途与持久化位置

全局设置页面管理本机目录边界、生图测试目录、界面显示偏好与分页面缩放比例：
- 常规设置写入当前配置根目录下的 `web-ui-settings.toml`（`[global]` 分区；分布式训练默认项存于同文件 `[distributed]` 分区，由 `/next/training` 分布式卡片维护）；
- 外置配置根目录等路径覆盖写入项目根已忽略的 `.anima-webui-settings.toml`（`[paths]` 分区），不随版本控制提交；
- 基础模型路径组合已独立至 [全局模型配置](global-model-configs.md)（`/next/models`）。

配置根目录解析优先级（从高到低，详见 [外置配置目录说明](../configuration/external-configs.md)）：
1. `.anima-webui-settings.toml` 的 `[paths].configs_root`
2. `ANIMA_CONFIGS_ROOT` 环境变量
3. 仓库默认 `configs/`

---

## 2. 页面四组配置项

在 `/next/settings` 修改后点击 **保存设置**（提交成功后自动失效非设置类查询缓存，必要时可按提示刷新页面），或点击 **恢复默认**：

| 分组 | 字段键 | 界面名称与作用 |
| --- | --- | --- |
| **文件与任务** | `output_root` | 训练输出根目录（默认 `output/runs`，支持目录浏览弹窗 `GET /api/settings/directories`） |
| **文件与任务** | `configs_root` / `history_root` / `queue_root` | 配置根目录、训练历史目录、训练队列目录外置覆盖（拒绝包含 `..` 的路径） |
| **文件与任务** | `tagging_max_retained_jobs` | 打标工作台内存中保留已结束任务的上限（默认 `40`，范围 `1..500`） |
| **生图测试** | `image_test_save_root` | 生图测试输出保存目录 |
| **生图测试** | `image_test_allow_home_search` | 是否允许生图测试在用户主目录下搜索模型文件 |
| **界面显示** | `ui_scale` | 全局默认界面缩放比例（`25%`–`400%`，默认 `100%`，含 `80%`–`125%` 快捷档与滑块） |
| **界面显示** | `dragon_motion_enabled` | 页面过渡与动效开关（默认开启；若操作系统开启 `prefers-reduced-motion` 则系统偏好优先） |
| **界面显示** | `dragon_config_help_always_visible` / `dragon_config_tags_always_visible` | 训练配置卡片始终完整显示帮助说明 / 标签徽章 |
| **分页面缩放** | `ui_scale_<page>` / `ui_scale_history_<tab>` | 各支持页面及历史详情子标签页的独立缩放百分比；**留空**即跟随全局 `ui_scale`（详见 [UI 缩放](ui-scale.md)） |

---

## 3. 风险与边界

- **修改 `output_root`**：仅影响之后新启动的任务与输出边界校验（`resolve_output_root()`），已有历史运行目录不会自动迁移。
- **修改 `configs_root` / `history_root` / `queue_root`**：配置列表、历史记录或队列视图会切换到新目录，切换前确认目标目录归属。
- **恢复默认**：会重置当前设置中的本机自定义项，但不会删除磁盘上的训练产物、历史记录或模型文件。

---

## 4. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_web_settings_paths.py \
  tests/test_ui_scale_settings.py \
  tests/test_settings_image_test.py \
  tests/test_global_settings_runtime.py \
  tests/test_env_config_paths.py \
  -q
```
