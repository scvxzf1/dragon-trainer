# 打标工作台（外部 API 与本地 ONNX）

状态：外部 API 与本地 ONNX 打标均已接线；本地模型权重按需下载
适用界面：Dragon Next (`/next/captioning`)

## 入口与联动

直接入口：

```text
http://127.0.0.1:20102/next/captioning
```

该入口由根路径兼容重定向统一导向 Next；旧 Classic/Dragon 页面已退役。

也可以在 **数据集** 页面选择并保存一个数据集预设，然后点击 **打开打标工作台**。页面会带入当前
`dataset_file`、数据组序号和原始图目录；这段上下文只保存在一次性的 `sessionStorage` 项中，不包含
API Key。

React Next 的数据集子集「图片工作台」还提供内嵌的 **打标** 视图：进入时按 URL 中的 `dataset` 与
`subset` 直接预选蓝图和图片组；结果审阅及接入预设、提示词、本地资源仍使用打标工作台原有功能。

## 基本流程

1. 选择数据集预设、图片组和原始图/训练图目录。
2. 勾选当前页图片并填写打标提示词。
3. 在右上角设置按钮中选择接入预设；外部预设可配置 OpenAI-compatible API 地址、模型、超时、重试和并发数。
4. 发送任务，等待逐张候选 caption 返回。
5. 在工作台的两行任务摘要中查看状态，使用常驻的刷新或“打开最终结果”入口。
6. 审阅并修改候选文本，再选择写回当前选中项或全部候选。

任务运行期间工作台只同步进度、状态和必要的结果节点，不会周期性替换整个页面；结果与日志页也采用增量更新。
浏览器标签页切到后台时会暂停轮询，回到前台后再恢复，避免隐藏页面持续占用请求和渲染资源。

外部模型返回结果不会自动写盘。普通文本 caption 使用原子写入；已有 `.json`、`.jsonl` 或共享
`captions.json` 的结构化标注源本身不会被覆盖，但显式写回仍会在图片旁生成或覆盖同名 `.txt` 旁路文件。

## 最终结果与任务保留

Next 在 `/next/captioning` 内直接提供任务审阅面板。
页面使用与训练数据预设库一致的
数据集选择器，可搜索、分组和预览首图；选中预设后只显示该 `dataset_file` 对应的
保留任务。任务工具栏固定为两行：第一行显示任务状态、刷新和重新打标，第二行放查看模式和批量写回操作。

结果按可用宽度自动分列为图片卡片，窄屏时回落为单列。每张卡片支持两种同步编辑模式：

- **Tag**：把逗号分隔的标签显示为可编辑胶囊，可增加、删除和拖动排序。
- **原文**：直接编辑完整 caption。

点击卡片图片会打开与数据集编辑器一致的原图详情预览，同时显示文件路径，并提供与卡片相同的
Tag/原文编辑、翻译和保存操作。卡片与预览共用同一份草稿，在任一位置或模式中的修改会立即反映到其他位置。“保存修改”只更新当前
WebUI 任务中的候选文本；内容未改变或改回原值时该按钮保持禁用。只有显式点击写回按钮才会修改图片同名 `.txt`。切换数据集、任务或离开页面时，
尚未保存的草稿会先要求用户确认。

对已结束的任务可选择任一当前可用接入预设执行“重新打标”。该操作复用原任务的数据集上下文、
图片和提示词，并保留原任务 ID 和任务列表位置；有勾选时只重置并处理勾选的单张或多张图片，
没有勾选时处理整个原任务。未选中的条目和已写回信息保持不变，重新生成的候选结果不会自动覆盖 TXT。

全局设置的“打标任务保留上限”对应 `tagging_max_retained_jobs`，默认为 `40`，有效范围为
`1..500`。超过上限时只按创建顺序清理最旧的已结束任务；排队中或运行中的任务不会被清理，图片、已写回的 TXT、
模型和下载缓存也不在该策略的删除范围内。任务仍只保存在当前 WebUI 进程中。

新版 `/next/captioning` 的任务列表标题旁提供“清理已结束任务”按钮。确认后清除当前列表中
已结束任务的内存记录及候选标注（包括未写回内容），不可恢复；运行中、正在写回或重跑的任务
会跳过。原始图片、已写回标注和模型不受影响，诊断日志仍按独立保留上限管理。
对应接口为 `POST /api/captioning/jobs/cleanup`，请求体为 `{"job_ids": ["任务 ID"]}`。

## 本地中英标签翻译

结果卡片的“中文 / EN”按钮用于在英文 Danbooru tag 与对应中文名称之间切换。这不是 Google
Translate API：标注不会发送到 Google 或其他第三方。首次使用时，页面会先弹窗说明来源和大小，
只有用户确认后才下载约 23 MB 的固定版本本地词典。仓库和默认安装均不包含该资源。

词典固定为 ffdkj/Danbooru Tag Chinese-English Translation Table 的 commit
`bc2953723a76e1841e9564297c6812723223ecb0`，下载大小和 SHA-256 均在服务端校验。默认发布到：

```text
$ANIMA_HOME/models/tag-dictionaries/danbooru-zh-en/
```

可用 `ANIMA_TAG_DICTIONARY_ROOT` 改变词典目录。相对路径锚定 `$ANIMA_HOME`，包含 `..` 的路径会被拒绝。

## 本地 ONNX 打标与模型资产

接入预设与模型资源配置面板提供 WD14 和 CLTagger 的本地 ONNX 接入。仓库不包含任何
模型权重；页面只显示一份版本固定的 manifest，并按资产 ID 显示本机状态。用户点击“下载模型”后，
服务端才会从 manifest 指定的 HTTPS 仓库获取文件，下载到：

```text
$ANIMA_HOME/models/captioners/<provider>/<asset_id>/
```

下载器只接受 manifest 中的资产 ID，不接受浏览器传入的 URL 或目标路径。每个文件都会校验声明的
大小和 SHA-256，先写入 `.part`/staging 文件，全部通过后再原子发布。取消、失败或 WebUI 关闭时会
清理临时文件；已经通过校验的最终文件不会被自动删除。下载状态保存在当前进程内，页面通过轮询展示
进度，重启后需要重新点击下载（已完整安装的文件会复用）。

当前固定资产：

- `wd14-eva02-large-v3`（SmilingWolf WD14 EVA02 Large v3）
- `wd14-vit-v3`（SmilingWolf WD14 ViT Tagger v3）
- `wd14-vit-large-v3`（SmilingWolf WD14 ViT Large Tagger v3）
- `wd14-convnext-v2`（SmilingWolf WD v1.4 ConvNeXt Tagger v2）
- `cltagger-v1-02`（cella110n CLTagger v1.02）
- `cltagger-v2-01a`（cella110n CLTagger v2.01a SigLIP2）

WD14 的四个资产共享 `selected_tags.csv` 标签格式，但模型容量和显存需求不同；可按设备选择。
CLTagger v2 使用 gated 仓库，必须先在 Hugging Face 接受模型条款并登录，再下载同一目录下的
`model.onnx`、`model.onnx.data` 和 `model_vocabulary.json` 三个文件。v2 的 `.data` 文件约 2.2 GB，
页面会将授权失败显示为下载错误，不会把凭据写入 profile 或普通设置。

可选地设置 `ANIMA_CAPTIONER_MODELS_ROOT` 将资产根目录迁移到其他磁盘；该设置仍由服务端规范化，
不能通过页面修改。模型运行时和模型权重是两个独立步骤：前者由用户在 Python 环境中显式安装，后者
由用户在接入预设页显式点击下载。默认安装和启动 WebUI 都不会安装 ONNX Runtime，也不会下载权重。

本地接入的执行设备支持 `自动`、`CPU` 和 `CUDA`。选择 `CUDA` 后可从与训练配置相同的
`GET /api/training/gpus` 设备列表中指定一张物理 GPU；服务端只在该任务的 Worker 子进程中设置
`CUDA_VISIBLE_DEVICES=<gpu_index>`，ONNX Runtime 在子进程内使用逻辑设备 `0`，不会修改 WebUI
主进程或其他并发任务的设备环境。旧预设可继续使用“默认可见 GPU”；显式指定 GPU 时采用严格模式，
目标设备或 CUDA provider 不可用会直接失败，不会静默回退 CPU。

每个本地任务使用一个短生命周期 Worker 子进程。正常完成、失败或取消后，Worker 会关闭 ONNX
Session 并退出，CUDA Context 由操作系统回收；取消流程有超时强杀兜底，WebUI 正常关闭时也会停止
全部活跃 Worker。切换页面只停止当前页面轮询，不会取消后台打标，因此任务仍在运行时不会释放模型；
任务完成或用户明确取消后才释放。

### 安装可选运行时

在项目虚拟环境中二选一：

```bash
# CPU，适合无 CUDA 或希望使用最小运行时的环境
uv sync --extra onnx-cpu

# CUDA，需确认 onnxruntime-gpu 与本机驱动/CUDA 环境兼容
uv sync --extra onnx-cuda
```

两个 extra 只安装 Python 运行时，不包含 WD14/CLTagger 权重。不要同时安装 CPU 和 CUDA 两个
extra；若更换运行时，请在同一虚拟环境中卸载旧的 `onnxruntime`/`onnxruntime-gpu` 后再同步。
安装完成后，回到接入预设页检查运行时状态，再单独点击对应资产的“下载模型”。

资产 API（`/api/captioning`，同时保留 `/api/tagging` alias）：

```text
GET  /model-assets
GET  /model-assets/<asset_id>
POST /model-assets/<asset_id>/download
GET  /downloads
GET  /downloads/<download_id>
POST /downloads/<download_id>/cancel
```

## 外部 API 配置

当前基座使用 OpenAI-compatible：

```text
GET  <base_url>/models
POST <base_url>/chat/completions
```

视觉请求把数据集图片编码为 `data:<mime>;base64,...`，同时发送系统提示词和当前任务提示词。设置页支持：

- `base_url`、`model`、`system_prompt`
- `timeout_seconds`、`retry_count`、`retry_interval_seconds`
- `concurrency`
- `allow_private_network`

API Key 可通过以下环境变量提供，按从左到右的顺序取首个非空值：

```text
ANIMA_CAPTIONING_API_KEY
ANIMA_TAGGING_API_KEY
TAGGING_API_KEY
```

页面保存的普通设置位于已忽略的 `configs/captioning/settings.toml`；本地密钥位于已忽略且权限尽量
收紧为 `0600` 的 `.anima-captioning-secrets.toml`。GET/PUT 响应只返回 `api_key_configured`，不会回填
密钥明文。环境变量密钥不能从页面删除。

## 安全与当前边界

- 默认拒绝本机、私网、link-local 和保留地址；使用本地模型服务时必须明确启用私网 API。
- DNS 解析结果在 aiohttp connector 层再次检查；外部 API 重定向不会跟随，避免跨主机转发密钥。
- 单张图片最大 64 MiB，单个响应体最大 4 MiB；任务并发上限为 8。
- 任务和候选审阅状态保存在当前 WebUI 进程内，重启后不会恢复，也不进入训练队列。
- Next 图片扫描按每页 60 张分页，支持跨页累计选择；Next 任务审阅按每页 50 条显示，服务端单任务上限为 500 张。

## API 与验证

canonical API 前缀为 `/api/captioning`，同时保留 `/api/tagging` alias。主要端点包括设置、连通测试、
任务创建/轮询/取消、候选编辑和写回。

Next 候选编辑先保存在本地草稿；点击“保存候选”才 PATCH `/jobs/{job}/items/{item}`，
只有显式“写回”才 POST `/jobs/{job}/commit`。存在未保存草稿时，写回按钮会阻止提交并提示仍有草稿。

定向测试：

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_tagging_local_worker.py \
  tests/test_tagging_profiles.py \
  tests/test_tagging_providers.py \
  tests/test_tagging_service.py \
  tests/test_tagging_routes.py \
  tests/test_tagging_batch_update.py -q
```

## 本任务批量标签管理

Next 审阅面板对当前打标任务的全部候选（最多 500，包含其他页）统计标签频次，统计和筛选包含尚未保存的草稿；这不是磁盘整个数据集的统计。点击标签筛选列表，当前检查对象保留，定位筛选外对象会恢复全部列表。

可明确选择全任务、当前筛选或勾选项，执行追加、删除、替换、触发词置顶与去重；空勾选不回退全任务。差异弹窗显示受影响项，应用只生成本地草稿，不发送请求。未改变的原文保持字节格式，批量清空候选不应用。应用前会再次核对预览原文与服务器基线，期间发生变化须重新预览。标签精确匹配，大小写、空格和下划线不互换。撤销/重做保留最近 30 次 delta，历史累计文本最多 2 MiB（UTF-8），超限只淘汰历史，不丢弃当前草稿，保存后清除历史；任务切换隔离草稿和历史。

WD14/CLTagger 默认开放标签管理；外部 API 和未知接入默认关闭，需确认逗号拆分规则再切换。运行中、结果尚未核对或存在服务器冲突时禁用批量操作。

“保存全部草稿”通过 `PATCH /api/captioning/jobs/{id}/items`（`/api/tagging` 同样支持）发送 `{items:[{id, proposed_caption, expected_caption}]}`。服务端先核对全部 ID、数量、结束状态和原候选，再一次更新内存候选；任一冲突返回 409 且全部不变。不写 TXT。连接中断或 409 时先重新读取并比对服务器；草稿保留，冲突逐项显示服务器候选与本地草稿，必须显式采用服务器新基线后再次保存，不自动重试。

旧单项保存及显式 TXT 写回保留，磁盘写回仍受原接口的逐项冲突/原子文件边界限制，不是跨文件事务。
