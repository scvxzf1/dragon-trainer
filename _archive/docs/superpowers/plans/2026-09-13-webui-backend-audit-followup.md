# WebUI 后端审计后续落地计划（2026-09-13）

- **状态：** 审计快照已整理；整改尚未开始
- **推进窗口：** 5 小时
- **审计范围：** `web/routes/`、`web/services/`、相关运行时契约与测试
- **上一份后端全景基线：** [2026-08-15 WebUI 后端审计](../../../../docs/findings/webui_backend_audit_20260815.md)

## 本轮约束

- 本报告基于 2026-09-13 的本地 `dev` 工作树快照。工作树已有大量其他未提交修改，结论不代表线上 `dev` 或发布版状态。
- 只把本计划及其索引改动归入本轮；不得回退、覆盖或清理其他工作树内容。
- 不启动训练、不操作队列、不修改用户运行数据，不执行 `git commit` 或 `git push`。
- 本轮只审计和整理计划；后续代码整改按阶段执行，且每个风险先补可复现测试。

## 当前结论

1. **P1（条件性策略风险，非命令注入）：训练启动的 `extra_args` 绕过预检输入面。** `training.py` 的启动、预处理、队列入口接收扩展参数；直接训练在 `launcher_start.py` 将它们原样追加到 `train.py` 命令，预检没有检查最终 argv。重复追加 `--config_file`、`--output_dir`、`--network_weights` 等已注册选项可使实际 CLI 值不同于 WebUI 已检查的配置、路径和 runtime 元数据；`--train_data_dir` 是否能替换 `dataset_config` 数据源取决于训练加载分支，本轮不作确定断言。参数通过 `create_subprocess_exec` 传递，未发现 shell 注入。若 WebUI 只服务完全受信任的本机用户，这是高级功能边界；若请求方不完全受信任，则是明确的策略绕过。现有测试覆盖扩展参数透传和 `--mixed_precision` 覆盖，未覆盖受保护参数、未知参数或畸形列表拒绝。
2. **P2：小数据集验证划分被估算和预检错误忽略。** `estimation.py::_training_pool_count` 与 `preflight_compat.py::_validation_state` 在图片数小于 100 时把已配置的验证划分视为无效；训练实际通过 `library/datasets/subsets.py::split_train_val` 执行划分。50 张图、`validation_split_num=10` 或 `validation_split=0.2` 时，运行端都是 40 train + 10 validation；估算仍按 50 张训练池计算（若 `sample_ratio=0.2`，估算为 10 张训练图，而运行端为 8 张训练图 + 10 张验证图）。preflight 的“自动关闭验证”提示也与训练端实现不符。
3. **P2：历史游标分页仍先全量读取和排序。** `training_history_list.py` 将 `limit=0` 传给 `list_history_tasks`，随后在内存中过滤游标、排序并取一页；搜索也会扫描任务摘要。游标目前只限制响应大小，没有限制每页的存储读取量，历史增长后分页请求成本随总量增加。

### 本轮验证记录

- 后端 smoke：**237 passed，2 warnings**（`NotAppKeyWarning`，来自现有测试夹具）。
- 配置/估算、启动、历史四个定向文件：**89 passed，2 failed**；复核窄回归（历史游标、日志索引、mask editor、启动）为 **37 passed，2 failed**。两次失败均是旧启动夹具缺少有效训练时长，被当前 core preflight gate 拦截；是否保留该门禁需要兼容决策，详见下文。
- 文档完整性：**7 passed，1 failed**；失败来自工作树原有的 `docs/findings/auto_block_swap_families_20260909.md` 和 `docs/findings/anima_dual_gpu_parallel_probe_20260904.md` 缺少状态标签，不属于本轮新增文档问题。

### 需要兼容决策的回归

本轮使用项目 `.venv` 对四个定向文件运行 pytest：**89 passed、2 failed**（16.71s）；复核窄回归确认同一原因。失败为 `tests/test_training_start_preprocess.py` 中两个旧启动路由用例：其临时配置没有 `max_train_steps` 或有效 `max_train_epochs`，而当前工作树新增的 core preflight gate 会拒绝缺少有效训练时长的配置。应保留该门禁还是兼容旧调用需要产品决策；若门禁是预期行为，测试夹具应提供有效训练时长，并另加缺时长时返回 400 的负例。不要为让旧夹具通过而直接放宽预检。

## 已核对但暂不列为缺陷

- `mask_editor` 对非整数查询值抛出的 `web.HTTPBadRequest` 是 aiohttp 约定的 HTTP 异常，应由框架返回 400；现有 HTTP 测试已覆盖非法 `dataset_index` 返回 400，仍可补 `offset` 和其他 handler 的正文契约。
- 历史分页按 `(started_at, id)` 排序；生产目录枚举以任务目录名作为 ID，未发现会产生重复 ID 的路径。纯分页 helper 对外部传入的重复键不防御，但目前不升级为生产缺陷。
- 日志分页的 `offset/total` 当前按非空 JSONL 行位置计数；坏行会使一页返回的有效对象少于 `limit`，但下一页按行偏移继续，不会因此跳过后续有效行。需补坏行跨页用例，明确这是 API 约定还是需要改成有效对象游标。
- 日志读写并发追加的快照一致性尚未证明有可见故障，不先按缺陷处理；阶段 4 用交错测试确定契约。

## 分阶段计划

### 阶段 0：收敛审计基线与测试夹具

- [ ] 确认缺少训练时长时 preflight 的预期 HTTP 契约；补正两个旧测试用例或调整门禁，并新增负例。
- [ ] 建立新增测试的独立文件，避免继续堆入大型历史/路由测试文件。
- [ ] 记录本计划所用测试命令、通过数和当前工作树基线；不据此推断未运行的全量测试通过。

完成条件：预检门禁的产品意图明确；四个定向测试文件通过；缺时长配置的拒绝行为有独立断言。

### 阶段 1：封闭 `extra_args` 的预检绕过（P1）

- [ ] 枚举 start、preprocess、queue single、queue batch、pending train-after 和队列恢复路径上的扩展参数来源与落点。
- [ ] 明确扩展参数契约：优先使用结构化字段；若需保留 argv 透传，建立集中白名单并拒绝路由已拥有的键、路径/配置选择键、未知键、畸形列表和重复冲突键。`train_data_dir` 的实际覆盖能力需先由训练数据加载分支测试确认。
- [ ] 在请求边界和可绕过 HTTP 的 service/queue 边界验证同一契约；队列调度对既有持久化条目也要 fail closed。
- [ ] 预检必须覆盖最终允许的有效配置；不得检查一份配置后再通过 CLI 覆盖未检查字段。
- [ ] 新增 HTTP 与 service 测试：合法的受支持参数仍可用；`--config_file`、`--output_dir`、`--network_weights` 等受保护参数不会启动、入队或写 runtime；对 `train_data_dir` 先锁定其在不同 dataset 配置下的优先级；queue batch 错误需报告对应 item。

完成条件：所有启动入口对危险/畸形参数返回明确 4xx；没有子进程或队列/runtime 副作用；合法参数和 preflight 一致。

### 阶段 2：统一验证集运行语义与估算（P2）

- [ ] 用训练端 `split_train_val` 行为作为唯一基准，消除两个 Web helper 的“小于 100 张图自动关闭验证”分支。
- [ ] 对比 `validation_split_num`、`validation_split`、验证耗尽、无图片、regularization subset 与边界数量的训练池计数。
- [ ] preflight 的 CMMD/validation warning 只描述训练端实际会发生的行为。
- [ ] 新增小数据集估算与 preflight 测试：50 张图分别验证 `split_num=10` 和 `split_ratio=0.2`；确认估算步数按 40 张训练池计算，并确认验证状态一致。

完成条件：代表性配置的 Web 训练池计数与训练数据集实际切分完全一致；旧的 100 张门槛无残留误导文案。

### 阶段 3：使历史分页真正有界（P2）

- [ ] 先用合成的 1k/10k 条任务摘要记录当前分页读量、排序成本和搜索耗时，保留可复现实测基线。
- [ ] 设计 store 层 keyset/cursor 读取，避免每个游标请求都先构造并排序全部任务摘要；游标排序键须唯一、稳定，并支持 archive/filter/search 语义。
- [ ] 精确 `total` 若必须扫描全量，应明确其成本并考虑独立统计/缓存；不得为了有界页误报 total 或漏记录。
- [ ] 增加跨页无重复/无遗漏、重复时间戳、唯一 ID、归档过滤、搜索和大量任务读量测试。

完成条件：单页的摘要读取/排序工作有明确上界或有被测的可失效索引；分页契约在新增任务并发写入时有定义。

### 阶段 4：锁定边界行为与后端总回归

- [ ] 为 mask editor 的 `offset` 及各 handler 的无效整数参数补 aiohttp HTTP 400 契约测试；`dataset_index` 已有基础覆盖。
- [ ] 为包含空行、坏 JSON、非对象 JSON 的日志分页补跨页测试，并明确 `offset/total/returned` 的计数单位。
- [ ] 为日志追加与分页读取交错补确定性测试；只有发现跳页、异常或违反已确认契约时再修改实现。
- [ ] 运行 config、training/queue/history、preview/tagging、HTTP/WS、path-boundary 定向回归和 `tasks.py test-backend-smoke`。
- [ ] 更新本计划的执行记录；如形成稳定现状结论，再另写 `docs/findings/` 快照，不覆盖 2026-08-15 历史审计。

完成条件：定向回归及 backend smoke 通过；失败均有明确归因；文档结论与实时代码/测试一致。全程不提交。

## 验证入口

在本仓维护执行中优先使用 `.venv/bin/python`：

```bash
timeout 60 .venv/bin/python -m pytest -q tests/test_training_start_preprocess.py
timeout 60 .venv/bin/python -m pytest -q tests/test_web_estimate_buckets.py tests/test_web_config_preflight.py
timeout 60 .venv/bin/python -m pytest -q tests/test_training_history_list.py tests/test_queue_revision_contract.py
timeout 180 .venv/bin/python tasks.py test-backend-smoke
timeout 60 .venv/bin/python -m pytest -q tests/test_documentation_integrity.py
```

每个失败都需区分实现缺陷、旧 fixture 与预期兼容变化；本计划不授权清理工作树、运行训练或提交代码。
