# 数据集编辑器

状态：稳定
适用版本：当前 WebUI Next 工作台 `/next/datasets`
入口命令：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

- `web/frontend-next/src/features/dataset-editor/`（Next 数据集工作区）
- `web/services/config/` 中的 dataset 预设读写
- `tests/test_web_config_datasets.py`

---

## 1. 这是干什么的

一句话：在「数据集」页管理 `configs/datasets/` 下可复用的数据集蓝图，再给训练配置引用。

你可以：

- 新建 / 复制 / 重命名 / 删除数据集预设
- 分组整理、导入导出
- 编辑每个 subset 的路径、caption、缓存、正则化等
- 预览原始图片与 caption

---

## 2. 入口

1. 打开 Next 主导航 **数据集蓝图**，进入 `/next/datasets`。
2. 左侧列表选择一个数据集预设。
3. 右侧编辑器修改字段并点击 **保存**（每个已保存子集标题旁可通过 **图片工作台** 进入预览、手动蒙版与打标）。
4. 可直接在数据集页点击 **应用到训练** 更新训练配置的 `dataset_config` 绑定，或在 `/next/training` 的数据集选择器中选用。

常用按钮：

- **新建预设 / 复制 / 重命名 / 新建分组**
- **导入 / 导出**
- **删除**
- **刷新**

图片预览在切回浏览器标签页或窗口重新获得焦点时自动刷新；隐藏页面不主动刷新，
连续滚动模式会更新当前保留的分页窗口。相同路径的图片替换后通过文件版本 URL 更新，
已打开的大图也会更新，并在新版本到达时自动恢复此前的加载失败状态。

图片工作台的预览页可通过“信息”按钮折叠或展开左侧信息栏。“连续滚动”默认关闭；
两项状态保存在当前浏览器中，蓝色边框表示信息栏展开或连续滚动开启；
开启后，上下滚动会按需读取相邻页，页码跟随当前浏览位置更新。最多同时保留五页，
关闭后返回当前浏览页，仍可使用翻页与跳页按钮。加载失败时点击重试，不会自动反复请求。

---

## 3. 关键配置项

| 项目 | 说明 |
| --- | --- |
| 预设文件 | 落在配置根目录下的 `datasets/` |
| 图片路径 / 源路径 | `source_dir` 是原始图片和 caption 来源。`image_dir` 是 resize/预处理后的训练图片目录 |
| caption 来源 | `auto` 依次查找 `captions.json`、同名 JSON、`caption_extension` 文本。也可显式选择 `txt`、`json` 或 `captions_json` |
| cache / training 相关目录 | `cache_dir` 可按 subset 保存 latent、text 等 sidecar 缓存 |
| regularization | `is_reg` 标记正则化子集，配合 `prior_loss_weight` 使用 |
| 校验划分 | `validation_split` 按比例切分，`validation_split_num` 大于 0 时优先按数量切分。每个蓝图组筛选后的非正则 `image_dir` 图片为 1～99 张时会自动关闭切分 |
| 图片预览 | 读取所选行的图片与 caption，只读预览 |

数据集页改的是“数据蓝图”，不是立刻开始训练。训练真正使用哪份数据，以配置页绑定结果和启动时 runtime 冻结为准。

普通数据行和正则化数据行可以保留各自的分辨率、batch 和缓存设置；训练加载时会在构建 bucket 前按全组普通图片 repeats 配平正则化图片。正则化数据暂不支持分阶段调度，保存时会显式拒绝该组合。

### 通用规则与分组设置

Next 数据集工作区顶部的“通用规则”是新建数据集组的默认基线。修改基线不会
隐式覆盖已有组；只有点击“同步到所有组”后，现有组的对应设置才会更新。

保存预设时，WebUI 会将这些基线值独立写入
`general.custom_attributes.webui_dataset_defaults`。该表只用于恢复 WebUI 表单，
不会替代每个 `[[datasets]]` 中实际生效的分组设置。旧文件没有该元数据时，
仍从第一个数据集组推导基线，保持向后兼容。

### 字段帮助与常用规则

36 项字段帮助采用简短说明与高亮建议，按需显示选项、注意和默认折叠的进阶内容。
字段标签旁的问号不会修改表单值，只读蓝图也可以查看帮助。按 Escape 先关闭帮助，阶段配置弹窗仍保持打开。
完整规则以保存时的校验和训练启动时的运行时配置为准。

- **路径**：相对路径相对项目根目录，也可以填写绝对路径。`source_dir` 读取原始图片与 caption。`image_dir` 指向已处理的训练图片，`cache_dir` 保存 latent、text 等 sidecar。开启“递归扫描子目录”后才会读取子目录。`path_pattern` 使用相对路径匹配，多个规则用 `|` 表示“或”，`*` 表示不筛选。
- **caption**：`auto` 的查找顺序是 `captions.json`、同名 JSON、`caption_extension` 文本。显式选择 `txt`、`json` 或 `captions_json` 会固定来源。“优先 JSON 标注”仅供未指定来源模式的旧配置兼容使用，先尝试同名 JSON，再回退文本；它不改变显式 `auto`。JSON 来源主要用于预处理和缓存；没有文本缓存、直接训练读取标注时仍读取 `caption_extension` 文本。预览可读取 JSON 来源。`keep_tokens` 按分隔符（默认逗号）计数，保护开头的标签片段不被标签丢弃或逐步加入标签规则移除；它不是分词数量，也不保护整条标注丢弃。
- **验证集**：`validation_split_num` 大于 0 时优先使用固定数量，否则才使用 `validation_split` 比例。`validation_seed` 固定抽样顺序。每个蓝图组分别统计其非正则 `image_dir` 图片，并先应用递归扫描和路径筛选；筛选后为 1～99 张时自动关闭验证切分。判断发生在 sample_ratio 抽样和 repeats 展开之前，不按 `source_dir` 或整份蓝图总数计算。验证集构建时重复次数固定为 1，`num_repeats` 只影响训练集。
- **分桶**：常规分桶由 `min_bucket_reso`、`max_bucket_reso` 和 `bucket_reso_steps` 控制范围与间距。启用固定 token 分桶且训练分辨率为 1024 时使用固定尺寸表，这三个字段不筛选或生成该表；其他分辨率下固定表先缩放，再可按最小和最大边长筛选。`bucket_no_upscale` 优先选择不大于原图的桶，但极小图片在后备路径仍可能被放大，不能作严格保证。
- **蒙版**：`none` 不使用蒙版，`embedded` 读取图片 Alpha，`external` 从 `mask_dir` 查找 `{stem}_mask.png`，`auto` 兼容旧的 masks 目录。外部蒙版支持镜像子目录和 flat fallback，找不到蒙版时回退为整图权重。
- **Qwen Edit**：每个配对由一个 `before` 参考子集和一个 `after` 目标子集组成，两侧填写相同的配对编号。保存时只校验角色编号；应用蓝图或开训时才检查图片：按相对子目录和不区分大小写的文件主名配对，每对须各有一张 before 与 after。after 子集的 batch size 固定为 1。编辑预设不能混用 `normal`、`is_reg` 正则子集或分阶段调度。
- **高级规则**：NL/Tag 混合按 `captions.json` 中每张图片的多条标注调整 NL 与 Tag 变体权重；只有一条标注的图片会完整保留原文，不改变其在训练中的最终图片比例，也不会生成新标注。触发词克隆会新增独立副本子集，原子集仍保留；副本整条 caption 替换为指定 prompt，且其重复次数独立于原子集。开启这些规则后，建议先看预检与训练量估算结果。
- **分阶段调度**：普通训练保持关闭，按顺序学习不同子集时再开启。所有阶段必须连续覆盖 0% 到 100%，不能混用正则化或 Qwen 编辑数据；阶段名称只用于显示，移动阶段只交换名称和子集绑定，不改变时间区间。

---

## 4. 危险项

- **删除数据集预设**：会删掉 `configs/datasets/` 里的蓝图文件；已启动任务不会自动回滚，但新任务会找不到该预设。
- **改路径但不重建缓存**：VAE / text / PE sidecar 与路径绑定；路径变了却继续用旧 cache，容易训到脏缓存。
- **导入同名预设**：默认可能拒绝覆盖，避免误覆盖；覆盖前先确认目标文件。
- **系统只读预设**：可能允许另存，不允许直接改源文件。
- **正则化权重 / validation 设置**：填错会改变训练步数估计和真实训练行为。

---

## 5. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_web_config_datasets.py \
  tests/test_web_config_estimation.py \
  -q
```
