# 配置工作台

状态：稳定
适用版本：当前 WebUI Next 工作台 `/next/training`
入口命令：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

相关代码：
- `web/frontend-next/src/features/training-config/`
- `web/services/config_service.py`、`web/services/config/`
- `web/services/training_service.py`、`library/training/compat_matrix.py`

---

## Next React 配置工作台

`/next/training` 是管理可复用训练 TOML 配置、执行兼容预检并立即启动或加入训练队列的核心工作台（历史运行快照在 `/next/history` 查看，完整 checkpoint 续训在历史任务详情发起）。

### 1. 页面结构与双视图布局

- **配置库与顶栏选择**：配置库支持侧边、顶部或自动排列（拖拽宽度保存在当前浏览器），支持读取/搜索、保存、另存、文件重命名、分组新建/重命名/删除、文件移动排序以及 TOML 原文导入/导出（页面不提供删除配置文件操作，避免误删用户配置）。顶部紧凑选择栏集中管理当前训练配置、硬件预设（`configs/presets.toml`）与全局模型组合填充。
- **分组视图 / 积木流视图**：可在顶部操作栏切换，共用同一份配置草稿。四个阶段依次为 **输入准备**（模型、数据集、标注与遮罩、筛选、预处理与缓存）、**方法配置**（基础方法、热启动、容量及适配器分支）、**训练计划**（任务名称、训练量、优化器、损失、正则化、预览、保存、日志）、**设备与性能**（分布式设备卡片、精度与计算后端、块交换与内存卸载、梯度检查点、编译加速、诊断）。目录之外的未知字段保留在“其他配置与审计”，不会在保存时丢失。
- **简要说明与字段帮助**：顶栏“配置 → 简要说明”支持“完全隐藏 / 单行显示 / 完整显示”；点击字段 `?` 按需加载 283 项 V2 结构化帮助（含模型路径读取的实际 block 数与交换上限）。仅当字段值与预设/方法继承基准已知不同时才显示覆写标记，支持“恢复继承值”与“撤销未保存修改”（均只改草稿，不自动写盘）。
- **模型路径与样张提示词**：DiT、`qwen3`、`vae` 输入框旁可通过只读 `GET /api/settings/model-files` 浏览项目 `models/` 及已存模型目录；样张提示词支持图形化逐条编辑（含 Qwen Edit 多参考图缩略图与大图预览）与原文模式，保存时写入配置专属分叉文件 `configs/sample-prompts/<methods_subdir>/<stem>.txt`。

### 2. 五模型族动态披露

工作台支持全部五个注册模型族（`anima`、`krea2_raw`、`z_image`、`qwen_image_2_1`、`moss_image`），表单字段与枚举按 `GET /api/config/model-families`（及同构离线目录 `domain/model-family.js`）动态过滤：
- Qwen Image 2.1 暴露 `qwen_image_2_1_task = "t2i" | "edit"`（T2I 允许实验 LoKr/LoHa，Edit 仅 plain LoRA）及预览换块覆盖；
- MOSS-Image 仅在该族下展示 `max_token_length` 与 `moss_te_quantization`（切换其他模型族时隐藏控件但保留草稿值）；
- 界面开放选择不等于全部组合已完成真机质量验收，完整边界以 [多模型支持现状](../multi_model_support.md) 为准。

### 3. 分布式训练卡片与 Pipeline Parallel（PP v0）

「设备与性能」阶段顶部的分布式训练卡片（`TrainingDistributedCard`）管理运行拓扑与物理 GPU 选择（通过 `GET/PUT /api/settings/distributed` 与 `POST /api/training/distributed/diagnosis` 交互）：

- **三种模式**：**单卡**（单选 GPU）、**数据并行 DDP**（至少两张卡）、**流水线并行 PP**（实验 2-stage 1F1B）。选卡保存在本机设置/浏览器上下文，启动与入队时作为 `gpu_whitelist` 传递（显式选卡索引在 `nvidia-smi` 中不存在时会以 `gpu_whitelist_unauthorized_gpu` 拒绝）。
- **PP 模型族边界**：当前 **Anima、MOSS-Image、Krea-2** 已接线专用 PP 训练会话（要求 `mixed_precision="fp16"`、plain LoRA、完整磁盘缓存、`gradient_accumulation_steps=1`、普通 `blocks_to_swap=0`）；Z-Image 与 Qwen Image 2.1 仍具名拒绝。
- **Krea-2 专属 PP 参数**：分布式详情中的「PP 块交换」对应 `pipeline_parallel_blocks_to_swap`（仅 Krea-2 PP 可设非零，编辑时同步更新当前草稿与本机 `[distributed]` 缺省值；设为 `0` 时兼容回退 `KREA2_PP_SWAP_BLOCKS` 环境变量）。Krea-2 的 `pipeline_parallel_microbatches` 必须整除 `train_batch_size` 且不少于 2。详见 [分布式训练卡片说明](../experimental/distributed-training-card.md)。

---

## 训练预检与风险确认

### 1. 共享兼容预检与 `--compat_unverified_policy`

WebUI 预检查、启动/入队确认与 CLI `train.py` 共用 `library/training/compat_matrix.py::check_training_compat`：
- **默认 `warn` 档**：WebUI 表单未暴露 `compat_unverified_policy` 字段，启动沿用默认 `warn`——仅将已审计“代码生效但缺真机验收”的 7 个 C1 组合降级为带 `[unverified 未验证]` 标记的 warning（记录到启动日志与 checkpoint metadata `ss_compat_unverified_*`）。
- **硬拦截不变**：输入错误、互斥冲突（如 selective checkpoint × full checkpoint、普通 `blocks_to_swap` × PP）以及放行即下游报错的 C2-raise / C3 组合在任何档位均为 error，阻止启动。CLI/TOML 如需复现旧版严格拦截可设 `compat_unverified_policy = "strict"`，详见 [模型训练能力与兼容预检策略](../configuration/model-training-capabilities.md#兼容预检策略--compat_unverified_policy)。
- **冻结 runtime 直训特例**：仅当冻结 runtime 配置的 VAE/text 缓存逐图有效、latent identity 匹配且不开启采样预览/CMMD 时，预检允许跳过缺失但实际不会加载的 VAE 路径；常规需要预处理的配置仍要求 VAE 路径存在。

### 2. NF4 编译风险确认与附加参数边界

- 预量化混合 NF4 与 Qwen Image 2.1 NF4 编译默认阻止启动（梯度差异背景见 [Qwen NF4 编译定位](../findings/qwen_nf4_compile_90hx_20261005.md)）。仅当所有阻碍项均属可确认风险时，启动/入队弹窗才提供「我已了解以上风险，仍要继续」勾选，确认后仅对当次请求或该队列项生效（CLI 对应 `--allow_unverified_nf4_compile`），不写回训练 TOML；配置或设备变更后须重新确认。
- Web `extra_args` 按训练 CLI 规则解析参与预检，但禁止覆盖 `config_file`、`method`、`preset`、`methods_subdir` 或触发只打印/快照类开关。

---

## 相关验证

后端契约测试：

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_web_config_service.py \
  tests/test_web_config_sample_prompts.py \
  tests/test_web_config_file_groups.py \
  tests/test_web_config_preflight.py \
  tests/test_mixed_nf4_preflight_risks.py \
  tests/test_training_preset_file_actions.py \
  tests/test_dragon_config_stage_catalog_frontend.py \
  -q
```

前端单元测试与类型检查：

```bash
pnpm --dir web/frontend-next exec vitest run src/features/training-config
pnpm --dir web/frontend-next run typecheck
```
