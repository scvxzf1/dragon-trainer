# configs 可变内容外置计划书

状态：已归档
日期：2026-06-24

## 目标

在 WebUI 的“全局设置 / 路径与默认模型”中新增一个“配置数据文件夹”设置，
把 `configs/` 里由 WebUI 产生或长期变化的内容移动到用户指定目录。

核心原则：

- 不整体搬走 `configs/`，只外置可变数据。
- 保持前端、历史记录和训练配置里的逻辑路径仍是 `configs/...`。
- 底层解析逻辑把部分 `configs/...` 映射到外部物理目录。
- 迁移只复制，不自动删除旧文件；清理旧目录必须另行确认。

## 当前 configs 分类

### 继续留在仓库内

这些是源码模板、CLI 合并链或测试/文档约定的一部分，应继续由 git 管理：

- `configs/base.toml`
- `configs/presets.toml`
- `configs/methods/*.toml`
- `configs/gui-methods/*.toml`
- `configs/bench/*`
- `configs/sam_mask.yaml`
- `configs/sample_prompts.txt` 作为默认样张提示词模板保留

理由：

- `train.py` / `tasks.py` 当前训练链固定读取
  `configs/base.toml -> configs/presets.toml -> configs/<methods_subdir>/*.toml`。
- 文档、测试和 `python tasks.py print-config` 都依赖这些内置路径。
- 系统方法配置属于项目版本的一部分，不应该被用户数据目录覆盖。

### 建议外置到配置数据文件夹

这些是用户配置、WebUI 状态或运行历史，适合移动到外部目录：

- `configs/imported/`
- `configs/datasets/`
- `configs/sample-prompts/`
- `configs/custom/`，当前可能不存在，但 CLI 已支持自定义 preset
- `configs/web-file-groups.toml`
- `configs/web-user-locks.toml`
- `configs/web-training-history/`
- `configs/web-training-queue/`
- `configs/.restore-backups/` 的后续新备份

备注：

- `configs/datasets/easycontrol.toml` 和 `configs/datasets/ip_adapter.toml`
  当前被代码视为系统只读数据集蓝图。外置后应通过“仓库内置只读 + 外部用户数据”
  的叠加读取保留。
- 当前 `configs/web-training-history/` 是体积主要来源，约 45M 中大部分来自日志、
  progress、metrics 和 block swap profile。
- 当前 `configs/web-ui-settings.toml` 应暂时继续留在仓库内，作为全局设置入口和外部
  数据根目录的指针；否则会出现“需要先读设置才能知道设置在哪里”的引导问题。

## 推荐目录模型

新增全局设置字段：

```toml
[global]
config_data_root = "/path/to/anima-config-data"
```

默认值建议仍为 `configs`，保证升级后行为不变。用户在全局设置里改成外部目录后，
WebUI 新写入的可变数据落到新目录。

建议外部目录结构：

```text
<config_data_root>/
  imported/
  datasets/
  sample-prompts/
  custom/
  web-file-groups.toml
  web-user-locks.toml
  web-training-history/
  web-training-queue/
  .restore-backups/
  migration-manifest.json
```

逻辑路径到物理路径映射：

```text
configs/imported/foo.toml
  -> <config_data_root>/imported/foo.toml

configs/datasets/foo.toml
  -> <config_data_root>/datasets/foo.toml

configs/sample-prompts/imported/foo.txt
  -> <config_data_root>/sample-prompts/imported/foo.txt

configs/web-training-history/<task>/
  -> <config_data_root>/web-training-history/<task>/
```

前端、历史分组、TOML 字段仍显示和保存 `configs/...`，避免破坏已有历史记录。

## 文件级改造计划

### 1. 全局设置服务

文件：

- `web/services/settings_service.py`
- `web/routes/settings.py`
- `web/static/index.html`
- `web/static/js/config/catalog/defaults.js`
- `web/static/js/features/anima-app/chunks/26-load-global-settings.js`
- `web/static/js/features/anima-app/chunks/36-setup-event-listeners.js`

改造：

- 增加 `config_data_root` 字段。
- 增加 `resolve_config_data_root()`、`display_config_data_root()`。
- 路径规则复用 `output_root`：允许绝对路径或项目相对路径，禁止 `..`。
- 全局设置 UI 增加“配置数据文件夹”输入框和帮助文案。
- 环境检测增加外部配置数据目录可写性检查。

### 2. WebUI 配置路径解析层

文件：

- `web/services/config/paths.py`
- `web/services/config_service.py`
- `web/services/config/_legacy.py`
- `web/services/config/*.py`

改造：

- 引入逻辑路径解析函数，例如：
  - `resolve_config_logical_path("configs/imported/a.toml")`
  - `resolve_config_glob("configs/imported/*.toml")`
  - `display_config_logical_path(path)`
- `configs/base.toml`、`configs/presets.toml`、`configs/methods/`、
  `configs/gui-methods/` 默认解析到仓库内。
- `configs/imported/`、`configs/datasets/`、`configs/sample-prompts/`、
  `configs/custom/` 默认解析到 `config_data_root`。
- 读取时可做兼容回退：外部不存在时读取仓库旧路径。
- 写入时统一写外部数据根目录。

重点替换点：

- `_safe_resolve()`
- `_safe_config_subdir()`
- `_glob_config_files()`
- `_normalize_dataset_preset_path()`
- `_normalize_prompt_file_path()`
- `_sample_prompts_path_for_config()`
- `save_raw_file()` / `load_raw_file()` / `delete_raw_file()`

### 3. 配置分组和锁定状态

文件：

- `web/services/config/file_groups.py`

改造：

- `WEB_FILE_GROUPS_FILE` 改为外部数据根目录下的 `web-file-groups.toml`。
- `WEB_USER_LOCKS_FILE` 改为外部数据根目录下的 `web-user-locks.toml`。
- 分组文件内部继续保存 `configs/...` 逻辑路径。
- `_glob_config_files()` 需要从外部用户目录和仓库内置目录合并结果。
- `restore_system_presets()` 的新备份写到外部 `.restore-backups/`。

兼容：

- 如果外部 `web-file-groups.toml` 不存在，先读取仓库旧文件。
- 首次保存后写入外部目录。

### 4. 数据集预设和样张提示词

文件：

- `web/services/config/datasets.py`
- `web/services/config/sample_prompts.py`
- `web/services/config/preflight.py`
- `web/services/training/runtime_config.py`

改造：

- 用户保存的数据集预设写到 `<config_data_root>/datasets/`。
- 系统只读数据集蓝图从仓库 `configs/datasets/` 兜底读取。
- per-config 样张提示词写到 `<config_data_root>/sample-prompts/...`。
- `sample_prompts = "configs/sample-prompts/..."` 继续写入 TOML。
- WebUI 生成运行时配置时，必须把外部逻辑路径解析成可被训练子进程读取的实际文件，
  或继续生成 `dataset.runtime.toml` / `config.runtime.toml` 到 `output_root` 下。

### 5. 历史记录和训练队列

文件：

- `web/services/training_service.py`
- `web/services/training/history.py`
- `web/services/training/queue.py`
- `web/services/training/runtime_config.py`
- `web/services/config/preflight.py`

改造：

- 将 `HISTORY_DIR` / `QUEUE_DIR` / `QUEUE_FILE` 从模块级固定常量改为解析函数：
  - `resolve_training_history_dir()`
  - `resolve_training_queue_dir()`
  - `resolve_training_queue_file()`
- `collections.json` 跟随 history 目录。
- `preflight.py::_history_training_tasks_for_output_dir()` 不再硬编码
  `CONFIGS_DIR / "web-training-history"`。
- 删除、归档、续训检查仍必须限制在解析后的 history / output root 边界内。

兼容：

- 单测目前大量 monkeypatch `training_service.HISTORY_DIR` 和 `QUEUE_FILE`。
  第一轮改造可保留这些常量作为测试覆盖入口，但生产路径由解析函数提供。

### 6. 环境检测和文档

文件：

- `web/services/environment_check_service.py`
- `README.md`
- `docs/README.md` 或对应 WebUI 文档
- `AGENTS.md`
- `.gitignore`

改造：

- 环境检测展示：
  - 全局输出根目录可写
  - 配置数据文件夹可写
  - WebUI 设置文件位置
- 文档说明哪些配置仍在仓库，哪些会写入外部数据目录。
- `.gitignore` 补充忽略后续新生成的可变配置数据：
  - `configs/custom/`
  - `configs/sample-prompts/`
  - `configs/.restore-backups/`
  - 已有的 `configs/imported/`、`configs/datasets/`、`web-training-*` 保留。

## 迁移流程建议

新增一个显式迁移工具，不在保存全局设置时自动搬动旧文件：

```text
python tasks.py web-config-migrate --target /path/to/anima-config-data --dry-run
python tasks.py web-config-migrate --target /path/to/anima-config-data
```

迁移步骤：

1. 扫描旧 `configs/` 可变内容。
2. 生成 dry-run 清单。
3. 复制到目标目录，保留相对结构。
4. 写入 `migration-manifest.json`，记录来源、目标、大小、mtime。
5. 更新 `configs/web-ui-settings.toml [global].config_data_root`。
6. 不删除旧文件；只在用户明确确认后再清理。

迁移对象：

```text
configs/imported/                 -> <config_data_root>/imported/
configs/datasets/                 -> <config_data_root>/datasets/
configs/sample-prompts/           -> <config_data_root>/sample-prompts/
configs/custom/                   -> <config_data_root>/custom/
configs/web-file-groups.toml      -> <config_data_root>/web-file-groups.toml
configs/web-user-locks.toml       -> <config_data_root>/web-user-locks.toml
configs/web-training-history/     -> <config_data_root>/web-training-history/
configs/web-training-queue/       -> <config_data_root>/web-training-queue/
configs/.restore-backups/         -> <config_data_root>/.restore-backups/
```

## 兼容风险

- `train.py --config_file configs/imported/foo.toml` 在用户手动 CLI 场景下可能无法读取外部
  物理文件。WebUI 应优先传运行时配置文件；CLI 兼容可在第二阶段增加
  `ANIMA_CONFIG_DATA_ROOT` 解析。
- 历史任务里的 `history_source_config_file` 和 `history_group_key` 存的是
  `configs/imported/...` 字符串。必须保持逻辑路径稳定。
- 前端测试和 UI 文案大量写死 `configs/datasets/`、`configs/imported/`。
  迁移不应把这些显示路径改成绝对路径。
- `configs/datasets/` 同时含系统只读蓝图和用户数据。解析层必须支持“仓库内置 + 外部用户”
  的叠加模型。
- 当前 `settings_service._load_base_model_path_defaults()` 从
  `SETTINGS_FILE.parent / "base.toml"` 读取默认模型路径；如果设置文件仍在
  `configs/web-ui-settings.toml`，短期可保持不变。后续若设置文件也外置，必须改为显式读取
  仓库 `configs/base.toml`。

## 验证计划

最小测试：

```bash
timeout 60 python -m pytest tests/test_web_config_service.py
timeout 60 python -m pytest tests/test_training_queue.py
timeout 60 python -m pytest tests/test_training_resume.py
timeout 60 python -m pytest tests/test_preview_service.py
timeout 60 python -m pytest tests/test_environment_check_service.py
timeout 60 python -m pytest tests/test_training_frontend_state.py
timeout 60 python tasks.py print-config METHOD=lora PRESET=default
```

手动验证：

- 打开 WebUI 全局设置，设置外部配置数据文件夹。
- 新建/导入训练配置，确认写入外部 `imported/`。
- 新建数据集预设，确认写入外部 `datasets/`。
- 保存样张提示词，确认写入外部 `sample-prompts/`。
- 加入训练队列，确认 `web-training-queue/queue.json` 写入外部。
- 启动一次 dry-run 或最小训练前置流程，确认 history 写入外部。
- 旧历史记录仍能读取、归档、续训检查和预览。

## 分阶段落地

### Phase 1：路径解析和 UI 设置

- 增加 `config_data_root` 设置和 UI。
- 新增配置逻辑路径解析层。
- WebUI 新写入的 imported / datasets / sample-prompts / 分组状态落到外部目录。
- 不移动历史和队列。

### Phase 2：历史和队列外置

- history / queue 改成解析函数。
- 兼容旧 `configs/web-training-history` 和 `configs/web-training-queue`。
- 增加迁移 dry-run。

### Phase 3：CLI 兼容和清理

- 可选支持 `ANIMA_CONFIG_DATA_ROOT`，让手动 CLI 也能解析外部 `configs/imported/...`。
- 更新文档和 `.gitignore`。
- 在用户确认后提供旧目录清理命令。

## 推荐结论

推荐实现“逻辑路径不变、物理位置可配置”的外置方案。

不要把整个 `configs/` 直接搬走；那会破坏 CLI 合并链、系统方法模板、文档链接和
大量测试。最稳妥的边界是：仓库继续保存内置模板，WebUI 可变数据由全局设置指向外部
配置数据文件夹。
