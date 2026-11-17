# 配置目录外置功能

状态：稳定
适用版本：当前 main
相关代码：`library/env.py`、`web/services/settings_service.py`

## 概述

Dragon Trainer 支持将配置目录外置到独立位置，便于：
- 多实例部署时隔离配置
- 容器化环境中分离系统配置和用户数据
- 将运行时数据（训练历史、队列）从代码仓库分离
- 在不同磁盘上管理配置和数据

## 配置方式

### 方法 1：WebUI 全局设置（推荐）

在 Next WebUI 的“全局设置”（`/next/settings`）页面配置：

1. 打开「文件与任务」分组。
2. 设置「配置根目录」。
3. 点击保存全局设置。

该值会写入项目根目录的 `.anima-webui-settings.toml [paths]`，这是本机
配置文件，已被 `.gitignore` 忽略。

### 方法 2：环境变量

创建 `.env` 文件（项目根目录）：

```bash
# 配置根目录（包含 base.toml, methods/, datasets/, web-training-history/, web-training-queue/ 等）
ANIMA_CONFIGS_ROOT=/path/to/configs
```

## 路径解析规则

- **相对路径**：相对于项目根目录（`dragon-trainer/`，旧目录名如 `anima_lora/` 同理）解析
- **绝对路径**：直接使用
- **环境变量扩展**：支持 `$HOME`、`~` 等扩展
- **安全检查**：自动拒绝包含 `..` 的路径（防止路径遍历）

## 使用示例

### 示例 1：外置到用户主目录

```bash
# .env
ANIMA_CONFIGS_ROOT=$HOME/.local/share/anima/configs
```

### 示例 2：外置到数据盘

```bash
# .env
ANIMA_CONFIGS_ROOT=/mnt/data/anima/configs
```

### 示例 3：容器化部署

```bash
# .env
ANIMA_CONFIGS_ROOT=/etc/anima/configs
```

**Docker Compose 示例**：

```yaml
services:
  dragon-trainer:
    image: dragon-trainer:latest
    environment:
      ANIMA_CONFIGS_ROOT: /etc/anima/configs
    volumes:
      # 配置根还包含历史、队列和 WebUI 设置，默认需要可写。
      - ./configs:/etc/anima/configs
```

如果必须把方法配置挂载为只读，请另外通过
`ANIMA_TRAINING_HISTORY_ROOT` 和 `ANIMA_TRAINING_QUEUE_ROOT` 指向可写目录，
并确认 WebUI 不需要在该配置根下保存或导入 TOML。

## 默认行为

未设置任何环境变量时，保持原有行为：
- 配置根目录：`dragon-trainer/configs`（即项目根下的 `configs/`）
- 包含所有配置文件、训练历史和队列

无需任何迁移，完全向后兼容。

## 环境变量优先级

配置解析优先级（从高到低）：
1. 项目根 `.anima-webui-settings.toml [paths].configs_root`
2. `ANIMA_CONFIGS_ROOT` 环境变量
3. 默认 `configs/` 目录

> 如果 WebUI 中已保存 `configs_root`，后来仅修改 `ANIMA_CONFIGS_ROOT` 不会覆盖
> 该本机设置。需在 WebUI 中更改/恢复默认，或手动调整
> `.anima-webui-settings.toml`。

## 迁移指南

如果已有 `configs/` 目录数据，可以：

### 选项 1：使用符号链接（简单）

```bash
# 1. 移动 configs 到新位置
mv configs /path/to/new/configs

# 2. 创建符号链接
ln -s /path/to/new/configs configs

# 3. 设置环境变量（可选）
echo "ANIMA_CONFIGS_ROOT=/path/to/new/configs" >> .env
```

### 选项 2：完整外置（推荐）

```bash
# 1. 复制整个 configs 目录
cp -r configs /path/to/new/configs

# 2. 设置环境变量
echo "ANIMA_CONFIGS_ROOT=/path/to/new/configs" > .env

# 3. 验证并备份原目录
.venv/bin/python tasks.py print-config METHOD=lora PRESET=default
mv configs configs.backup
```

## 故障排查

### 检查当前配置

通过正式配置入口确认外置目录可被读取和合并：

```bash
.venv/bin/python tasks.py print-config METHOD=lora PRESET=default
```

### 常见问题

**Q: 设置环境变量后路径没有变化？**

A: 确保 `.env` 文件在项目根目录，并重启 WebUI 或训练进程。

**Q: 相对路径相对于哪里？**

A: 所有相对路径都相对于项目根目录（如 `dragon-trainer/`），而不是当前工作目录。

**Q: 外置后原 configs/ 目录会被删除吗？**

A: 不会。外置是通过环境变量切换路径，不会自动删除任何文件。

## 性能影响

路径通过轻量的动态路径封装获取，便于 WebUI 保存全局设置后切换配置根。
相对于配置 IO 和训练开销，该路径解析开销可忽略。

## 安全性

- 自动加载 `.env` 文件时不会覆盖已有的环境变量
- `.env` 文件已被 `.gitignore` 排除，不会意外提交敏感路径
- 路径安全检查防止 `..` 遍历攻击
- 建议将外置配置目录设置适当的文件权限

## 技术细节

### 实现位置

- `library/env.py` - 路径获取和环境变量解析
- `web/services/settings_service.py` - 全局设置管理
- `web/services/training_service.py` - 训练服务路径
- `.env.example` - 环境变量模板
- `.anima-webui-settings.toml.example` - WebUI 本机路径覆盖模板

### 测试覆盖

- `tests/test_env_config_paths.py` - 路径解析单元测试
- `tests/test_global_settings_runtime.py` - WebUI 配置根热切换与运行时路径测试

## 反馈和支持

如有问题或建议，请提交 Issue 或 Pull Request。
