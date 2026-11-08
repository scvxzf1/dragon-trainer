# 预处理缓存复用设计（LoRA / LoKr 重复测试）

状态：草案（已完成头脑风暴分段确认，待用户审阅后进入 implementation plan）  
适用版本：当前 main  
相关代码：

- `web/services/training/runtime_prepare.py`
- `web/services/training/runtime_datasets.py`
- `web/services/training/history_store.py`
- `library/preprocess/`、`library/io/cache.py`
- `scripts/tasks/preprocess.py`
- 配置/前端：训练配置表单与 WebUI 配置保存路径

---

## 1. 背景与问题

一句话：同一数据集、同一训练变体反复测试时，WebUI 仍会为每次 run 物化独立 `dataset_cache`，并可能重复跑 VAE / TE 预处理，浪费磁盘和时间；这与“任务隔离”主张在历史管理处冲突。

### 1.1 现状

| 层 | 行为 | 可复用性 |
|---|---|---|
| CLI 默认 `post_image_dataset/*` | 共享目录；preprocess 对已存在 sidecar 幂等 skip | 已可复用 |
| WebUI `output/runs/<run>/dataset_cache/` | 每任务准备独立 resized / lora 目录；续训等路径会 `copytree` | 重复测试最痛 |
| 历史删除 | 训练任务与同 `run_dir` 的预处理任务绑定清理；产物清理受 `output_root` 约束 | 若缓存“归任务所有”，共享后易误删 |

预处理侧已有能力：

- VAE / TE 写入支持 `cache_dir` 重定向与 nested stem 布局（`library/io/cache.py::resolve_cache_path`）
- 已存在 sidecar 可 skip（idempotent）
- 命名约定集中在 `library/io/cache_names.py`

缺口主要在 **WebUI runtime 物化策略** 与 **历史对缓存所有权的语义**，而不是底层 encode 管线从零重写。

### 1.2 目标用户场景

- 数据集不变
- 训练变体不变或仅改与预处理无关的超参（lr、epoch、LoRA/LoKr 结构等）
- 短周期反复测试
- 需要前端可配置：哪些层允许复用

### 1.3 成功标准

1. 同一数据 + 同一预处理签名下，第二次及以后启动可显著减少 `dataset_cache` 全量拷贝与 VAE/TE 重算。
2. 用户可在训练配置中独立开关 A/B/C 三层复用。
3. 删除历史训练任务默认不删除仍被其他 run 引用的共享缓存。
4. 触发词克隆、`nl_tag_mix` 等改写输入路径不与原数据集混池。
5. 训练产物（权重、日志、样张）继续 per-run 隔离。

---

## 2. 需求锁定（头脑风暴结论）

| 项 | 选择 |
|---|---|
| 优先级 | 两边都要；第一期主线 = 共享缓存复用，历史仅最小防误删 |
| 复用层 | **A** dataset_cache 拷贝；**B** VAE latents；**C** TE cache |
| 前端控制 | 三个独立开关，任意组合 |
| 作用域 | **按训练配置保存**（非全局设置） |
| 默认值 | A/B/C **默认全开** |
| 严格度 | **默认轻量指纹**；可切换 **全内容 hash（仅输入图 + caption）** |
| 架构路线 | **方案 A：共享内容池 + run 挂引用** |

不在第一期：

- PE 复用开关
- 全局设置兜底 / 配置覆盖链
- 后台自动 GC 守护进程
- 跨机器缓存同步
- 改变 CLI 默认 `post_image_dataset/*` 语义（CLI 已共享；重点修 WebUI）

---

## 3. 方案对比与选定

| 方案 | 思路 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| **A 共享池 + 引用** | `cache_pool/<fp>/` 存内容；run 的 `dataset_cache` 挂链接/引用 | 少拷贝 + 少重建；历史语义清晰 | 需 pool + 引用管理；symlink 兼容 | **选定** |
| B 直接全局路径 | run 直接用配置里的 resized/lora 路径 | 实现最短 | 隔离弱；删历史易误伤；多策略易污染 | 否 |
| C per-run + hardlink | 目录仍 per-run，文件 hardlink 去重 | 展示改动小 | 主要省空间；重建逻辑仍要另做 | 否 |

选定 A 的原因：同时满足 A/B/C 可配复用，并保留“任务产物隔离”，把“可共享内容”从“任务私有文件”中拆出，化解历史管理瓶颈。

---

## 4. 架构

### 4.1 目录布局

```text
output/cache_pool/                      # 默认池根（与 runs 同级；实现期可配置）
  <fingerprint>/
    manifest.json                       # schema、签名、模式、清单摘要、创建时间
    resized/                            # 缩放图与所需 caption sidecar
    lora/                               # VAE latents + TE caches
    refs.json                           # run_id 引用列表（最小引用实现）

output/runs/<run>/
  dataset_cache/dataset-01/
    resized  -> cache_pool/<fp>/resized # A 开启时：symlink / hardlink / junction
    lora     -> cache_pool/<fp>/lora
  training_output/                      # 永远独立
  model_cache/                          # 永远独立
  config.runtime.toml
  run meta: pool 绑定信息
```

约定：

- 多 subset：`dataset-01`、`dataset-02`… 各自 fingerprint / 池条目。
- 池根默认建议：`output/cache_pool/`，不进入 `configs/web-training-history/`。
- 展示路径仍通过现有 display/resolve helpers，避免把本机绝对路径写死进可提交配置默认值。

### 4.2 逻辑分层（反上帝代码）

新逻辑优先独立模块，热点文件只做薄编排：

| 模块（建议） | 职责 |
|---|---|
| `library/cache_pool/fingerprint.py` | 扫描输入、算 light/content 指纹、配置签名 |
| `library/cache_pool/store.py` | 池路径、manifest、原子创建、命中查询 |
| `library/cache_pool/refs.py` | 引用增减、无引用枚举 |
| `library/cache_pool/mount.py` | symlink → hardlink → copy 回退 |
| `web/services/training/runtime_*` | 调用池 API，改 materialize，不塞大块算法 |
| 配置 schema / WebUI 表单 | 暴露开关与严格度 |
| 历史删除路径 | 默认跳过共享池；可选清理无引用 |

禁止把完整指纹/池管理堆进 `training_service.py` 或超大 chunk JS。

### 4.3 数据流

```text
配置(A/B/C + mode)
  -> 每 subset：materialize 特殊 source（nl_tag_mix / trigger_clone）
  -> 扫描输入 + 算 fingerprint
  -> force_rebuild? / 池命中?
  -> A: 挂引用或私有物化
  -> B/C: skip 或只重建对应层
  -> 写 run_meta 绑定
  -> 训练产物进 training_output
```

---

## 5. Fingerprint 与严格度

### 5.1 纳入签名的字段

| 类别 | 是否纳入 | 说明 |
|---|---|---|
| 源数据规范化路径 | 是 | 不同目录不混用 |
| 图片清单 | 是 | 相对路径 + size + mtime（light） |
| caption 对应关系 | 是 | sidecar/json 的 size + mtime（light） |
| resize 规则 | 是 | 分辨率 / min_pixels / drop_lowres 等影响 resized 与 latent shape 的项 |
| model family / TE 相关配置 | 是 | TE 输出绑定模型与 padding 约定 |
| VAE 相关关键配置 | 是 | 影响 latent |
| 训练方法（LoRA/LoKr…）、lr、epoch、seed | 否 | 与预处理缓存无关，避免无谓拆池 |

`schema_version` 变化时旧池不自动命中，避免静默脏读。

### 5.2 两档严格度

| 模式 | 算法 | 开销（本机量级，约 180MB/s 全文件 SHA256 外推） |
|---|---|---|
| `light`（默认） | 路径 + 预处理配置签名 + 文件 name/size/mtime 清单 | 接近枚举目录成本 |
| `content` | 在 light 基础上对**输入图字节 + caption 文本**做 SHA256 | 50 张 ~1s 内；300 张数秒；2000 张约数十秒量级（视盘与 OS cache） |

明确不做（默认路径）：

- 不对 `*.npz` / `*_anima_te.safetensors` 做全量内容 hash（成本高且对“输入是否变化”增益有限）
- 缓存文件完整性校验可作为后续诊断工具，不进入日常复用热路径

### 5.3 指纹形态

```text
fingerprint = short_hash(
  schema_version,
  fingerprint_mode,       # light | content
  normalized_source_paths,
  inventory,              # light stats 或 content digests
  preprocess_signature
)
```

- 目录名：短 hex（建议 16–32）
- 完整输入：`manifest.json`

### 5.4 特殊输入

| 场景 | 处理 |
|---|---|
| `nl_tag_mix` | 先生成改写后的 source，再对**结果树**算指纹 |
| 触发词克隆 | 同上；不得与原数据集共享池条目 |
| 多 subset | 每 subset 独立 fp |
| 池命中但缺 B 或 C 文件 | 只补缺层，不整池删除（除非强制重建策略触发） |

---

## 6. 配置与前端

### 6.1 配置键

写入**训练配置**（配置级，非全局设置）：

| 键 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `reuse_dataset_cache_copy` | bool | `true` | A：复用 dataset_cache 拷贝（挂共享池） |
| `reuse_vae_latents` | bool | `true` | B：复用 VAE latents |
| `reuse_text_encoder_cache` | bool | `true` | C：复用 TE cache |
| `cache_fingerprint_mode` | `"light"` \| `"content"` | `"light"` | 指纹严格度 |
| `force_rebuild_preprocess_cache` | bool 或一次性动作 | `false` | 本次强制重建 |

前端形态：

- 三个独立开关（A/B/C）
- 严格度切换（轻量 / 全内容）
- 「强制重建」按钮（优先一次性动作，避免配置里长期卡在 true）

同步面：

- config schema / loader
- WebUI 配置表单与保存
- `config.runtime.toml` / 历史 snapshot 可观测
- 文档入口（features 或 configuration 短说明）
- 相关测试

### 6.2 开关语义

| 开关 | 开 | 关 |
|---|---|---|
| A | run `dataset_cache` 挂到池，避免 `copytree` 整包 | 物化到本 run 私有目录（仍可从池 copy 加速） |
| B | 池/目标中已有匹配 VAE → preprocess skip | 强制重算 VAE |
| C | 已有匹配 TE → skip | 强制重算 TE |

强制重建写回策略（默认，偏安全）：

- 普通「强制重建」：**写入本 run 私有 cache**，避免污染共享池
- 显式「重建并更新共享池」才覆盖/替换池内容（可第二期；第一期至少文档化并避免静默写脏共享池）

---

## 7. 运行时流程

```text
1. 读配置开关与 fingerprint_mode
2. 解析 subset 行；处理 nl_tag_mix / trigger_clone 等改写
3. 扫描输入，计算每 subset fingerprint
4. 若 force_rebuild：按 B/C 决定重建层与写入目标
5. 否则查询 cache_pool/<fp>
   - 未命中：创建池目录，preprocess 写入池
   - 命中：进入挂载/补缺
6. A 开：mount resized/lora 到 run.dataset_cache
   A 关：copy 或本地物化
7. B/C：对缺失或强制层只跑对应 preprocess
8. refs 增加 run_id；写 run_meta 绑定
9. 启动训练；产物只进 training_output / model_cache
```

并发同 fp：

- 池写入使用临时目录 + 原子 rename
- 竞争失败方复用已发布池结果

挂载回退：

```text
symlink -> hardlink/junction -> copy
```

失败时日志/预检 warning，不硬崩（除非用户要求严格链接模式——第一期不做额外严格链接模式）。

---

## 8. 历史与删除（最小防误删）

### 8.1 run_meta 记录

| 字段 | 用途 |
|---|---|
| `cache_pool_root` | 池根 |
| `dataset_cache_bindings[]` | 每 subset：`fingerprint`、`pool_path`、`link_mode`、`reuse_flags`、`fingerprint_mode` |
| `cache_inventory_summary` | 文件数 / 缺层摘要（详情展示） |

历史详情建议展示：是否命中共享池、模式、A/B/C 生效、挂载方式。

### 8.2 删除默认语义

| 对象 | 默认 |
|---|---|
| 历史 meta | 删 |
| `training_output` / 权重 / 样张 / 日志 | 删（`output_root` 边界内） |
| `model_cache` | 删 |
| 本 run 私有 dataset_cache（copy 模式） | 删 |
| 链接挂载点 | 拆引用 / 删空壳，**不跟删池内容** |
| `cache_pool/<fp>` | **默认不删** |

删除 run 时：从 `refs.json` 移除该 `run_id`。

### 8.3 可选清理

独立动作：**清理无引用共享缓存**

- 枚举池条目
- 引用为 0 才删除
- 删除确认文案明确路径与数量
- 第一期不做后台自动 GC

### 8.4 与旧链接预处理任务关系

现有 `_linked_preprocess_tasks_for_training` 按同 `run_dir` 绑定预处理历史。引入共享池后：

- 同 run 的预处理历史记录仍可按 run 绑定展示/删除
- **共享池生命周期与“单次 preprocess 历史任务”解耦**
- 文档需写清：删预处理历史 ≠ 删共享池

---

## 9. 错误处理与可观测性

| 情况 | 处理 |
|---|---|
| symlink 不可用 | hardlink → copy + warning |
| content 模式大集偏慢 | 预检可提示；允许改回 light |
| manifest schema 过旧 | 视为未命中，新建条目 |
| 池缺部分 sidecar | 只补 B/C 缺层 |
| 并行写池 | 临时目录 + 原子发布 |
| 旧 run 无 pool 字段 | 历史列表/删除保持兼容 |

日志至少包含：fp、mode、hit/miss、link_mode、skip/rebuild 的 B/C 统计。

---

## 10. 测试计划

优先定向测试（`timeout 60` + `.venv/bin/python`）：

| 主题 | 断言 |
|---|---|
| fingerprint 稳定 | 同输入同配置 → 同 fp |
| fingerprint 敏感 | 改 caption / mtime 清单 / resize 签名 → fp 变 |
| A 开 | 无全量 copytree；存在链接或等价挂载 |
| A 关 | run 私有目录有实体文件 |
| B/C 开 | 已有 sidecar 时 skip |
| B/C 关 | 对应层重算 |
| light vs content | 模式写入 manifest；可切换 |
| 删除 | 删训练历史不删仍被引用池 |
| 无引用清理 | refs=0 才可删池 |
| 特殊路径 | trigger_clone / nl_tag_mix 独立 fp |
| 兼容 | 无 pool 字段的旧 run 仍可列/删 |
| runtime 回归 | 现有 `tests/test_training_runtime_config_*.py` 等适配新默认行为 |

不默认跑真实大模型 preprocess。

---

## 11. 实现分期

### 第一期（本设计范围）

1. fingerprint（light + content 输入 hash）
2. cache_pool store / mount / refs
3. runtime materialize 接入 A/B/C
4. 配置键 + WebUI 三开关 + 严格度 + 强制重建入口
5. 历史删除默认不删池 + 引用移除
6. 可选“清理无引用池”
7. 测试 + 简短用户文档

### 第二期（非本设计必做）

- 显式「重建并发布到共享池」产品化
- PE 开关
- 全局默认 + 配置覆盖
- 引用 GC 策略增强 / 池配额
- Windows 挂载体验细化
- 历史详情更丰富的缓存命中 UI

---

## 12. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 脏缓存训错 | 默认 light 覆盖常见增删改；提供 content 与强制重建 |
| 共享池被误删 | 删除默认不碰池；清理需显式且 refs=0 |
| 链接权限/跨盘 | 三级回退 copy |
| 与完全隔离主张冲突 | 隔离边界上移到“产物隔离 + 内容寻址共享” |
| 热点文件膨胀 | 新模块承载，runtime 只编排 |
| 默认全开改变现有行为 | 文档说明；可用 A/B/C 全关回到偏隔离 |

---

## 13. 明确的非目标（再次强调）

- 不改变 Text Encoder padding、token bucket、lazy load、compile-after-apply 等训练不变量
- 不在第一期做 PE 复用产品开关
- 不把共享池默认放进可提交的 configs 树当作用户配置源
- 不默认执行长训练或大模型下载验证

---

## 14. 开放实现细节（plan 阶段可定，不阻塞本 spec 方向）

以下不影响架构选定，可在 implementation plan 落具体 API 名与文件切分：

1. 池根精确配置键名与是否允许 `.anima-webui-settings.toml` 覆盖（第一期可用固定默认 `output/cache_pool`）
2. `force_rebuild` 用配置 bool 还是 API 一次性参数
3. refs 用每池 `refs.json` 还是中心索引
4. 前端控件放在配置页“预处理/数据集”哪一块 DOM

---

## 15. 验收清单（DoD）

- [ ] A/B/C 三开关按配置保存且默认 true
- [ ] light / content 可切换
- [ ] 命中共享池时二次 run 不再全量 copy dataset_cache（A 开）
- [ ] B/C 开时已有匹配 sidecar 跳过对应 preprocess
- [ ] 删历史默认保留仍被引用的 `cache_pool/<fp>`
- [ ] 特殊改写路径独立指纹
- [ ] 定向测试通过
- [ ] 用户文档说明复用与删除语义

---

## 变更记录

| 日期 | 说明 |
|---|---|
| 2026-07-11 | 初稿：头脑风暴确认方案 A 与分段设计 1–3 |
