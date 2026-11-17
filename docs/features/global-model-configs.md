# 全局模型配置

状态：稳定
适用版本：当前 WebUI Next 工作台 `/next/models`
相关代码：`web/frontend-next/src/features/settings/ModelConfigPage.tsx`、`web/services/model_config_service.py`、`web/services/settings_service.py`

## 1. 入口与用途

通过 Next 主导航的 **模型配置**（`/next/models`）管理可复用的模型路径组合库。页面采用**左侧分组列表 + 右侧配置编辑**双栏布局，只记录路径与模型族，不移动、下载或删除磁盘上的权重文件。

当前支持全部五个已注册模型族：

| 界面显示名称 | `model_family` 持久化值 | 组件路径字段 | 只读能力标签（后端派生） |
| --- | --- | --- | --- |
| Anima | `anima` | DiT、`qwen3`、VAE | 普通文生图；完整推理与方法面 |
| Krea-2 | `krea2_raw` | DiT、`qwen3`（Qwen3-VL）、VAE | 普通文生图 |
| Z-Image | `z_image` | DiT、`qwen3`、VAE（Diffusers 目录或单文件） | 普通文生图 |
| Qwen Image 2.1 | `qwen_image_2_1` | DiT、`qwen3`（Qwen3-VL）、VAE | 普通文生图 + 编辑训练（采样支持最多 4 图参考） |
| MOSS-Image | `moss_image` | DiT、`qwen3`（BF16/NF4 TE 目录）、VAE | 普通文生图、仅支持 LoRA |

> 页面品牌名 `Dragon Trainer` 不是模型格式；实际训练与推理分派完全由 `model_family` 决定。Anima 2.9B Preview v1（40 blocks）仍使用 `model_family = "anima"`，层数由训练器读取权重头自动识别。

## 2. 创建、筛选与排序

1. 左侧可按关键词搜索，或通过 **筛选训练能力**（全部 / 编辑训练 / 普通文生图）过滤模型组合。
2. 点击 **新建配置**，选择模型族并填写名称与三条组件路径；每条路径旁支持通过服务端目录浏览器（`GET /api/settings/model-files`）选择权重文件或 Diffusers 目录，并实时显示路径状态徽章（`GET /api/settings/model-path-status`：文件体积、目录就绪或缺失提示）。
3. 点击 **保存配置** 写入模型配置库；点击 **设为默认** 将该项设为新建预设和旧回退链的默认来源（删除默认项前须先将另一项设为默认）。
4. 支持新建、重命名和删除分组（删除分组不会删除配置，组内条目自动并入首个保留分组）。
5. 条目与分组通过 **上移 / 下移** 按钮及编辑区的 **所属分组** 下拉框调整顺序与归属；搜索或筛选激活期间禁用移动按钮以保护持久化顺序。

## 3. 训练配置页联动

在 `/next/training`「输入准备 → 模型」中点击选择模型组合弹窗，确认后会同时将 `model_family`、`pretrained_model_name_or_path`、`qwen3`、`vae` 填入当前训练草稿（仍需保存训练配置后才落盘）。

## 4. 持久化与兼容镜像

模型配置库保存在当前配置根目录的 `web-ui-settings.toml`（`[model_config_library]` 的 `default_id`、`items`、`groups`），保存请求受 `revision` 乐观锁保护（跨标签页并发修改返回 409 提示刷新，不静默覆盖）。

默认项在原子写入时同步镜像到 `[global]` 的三条路径与 `model_family`（`anima` 沿用历史兼容省略显式键，其余四个非默认模型族均显式写入各自的 `model_family`），保证旧预设创建与路径预检兼容。

## 5. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_model_config_service.py \
  tests/test_settings_model_family.py \
  tests/test_model_file_service.py \
  -q
```

前端单元测试与 E2E：
- `web/frontend-next/src/features/settings/modelLibrary.test.ts`
- `web/frontend-next/src/features/settings/ModelConfigPageGroups.test.tsx`
- `web/frontend-next/src/features/settings/ModelCapabilities.test.tsx`
- `web/frontend-next/e2e/model-library-order-path.spec.ts`
