# 项目清理强制长跑目标书：多轮持续推进版

一句话：这份任务书专门解决“二十分钟就收口”的问题，把长时间持续推进写成硬性完成条件，而不是建议。

日期：2026-07-05
状态：已完成归档
完成提交：`bd591b83 test: extend sustained cleanup coverage`
完成记录：见 `docs/findings/project_cleanup_checkpoint_20260705.md` 第 15 节和 R4 正式收尾验证
后续入口：`docs/findings/project_cleanup_sustained_goal_20260706.md`
前置已完成：

- `project_cleanup_long_running_goal_20260705.md`：已完成归档，提交 `f74b8255`
- `project_cleanup_next_stage_goal_20260705.md`：已完成归档，提交 `7c5c277c`

默认发布目标：本地 `main` -> `webui/main`
目标强度：强制长跑，不允许完成一个小闭环就停。
建议运行窗口：至少 2 小时，理想 3 到 5 小时。

> ⚠️ 这份文档现在只作为历史目标记录保留，不再作为新的活跃目标重复执行。
> 下一次要跑长目标时，请使用 `project_cleanup_sustained_goal_20260706.md`。

---

## 🎯 1. 总目标

一句话：下一轮必须持续推进多轮低风险工程清理，直到满足“时间 + 阶段 + 验证”三重条件。

目标名称：

```text
强制长跑项目清理：LoRA save/load 拆分保护 + config facade 收口 + type-check 试点扩大
```

总目标：

- 继续 `TASK-07`：LoRA save/load/builder/router 周边只做有测试保护的小拆分。
- 继续 `TASK-09`：Web config facade 的导入面、兼容壳和 split module 复用继续收口。
- 继续 `TASK-10`：类型检查试点范围只按小目录扩大，不做全仓一次性切换。
- 持续更新 checkpoint，确保每轮结果可追踪、可回滚、不可夸大。

---

## ⛔ 2. 硬完成条件

一句话：这次不能只因为几个测试通过就提前 complete。

只有同时满足以下条件，才允许标记目标完成：

1. **最低耗时门槛**
   - `goal.timeUsedSeconds >= 7200`，也就是至少 2 小时。
   - 如果当前环境没有 goal 时间工具，就用开始时记录的 `date +%s` 和当前 `date +%s` 计算。
   - 不允许靠 `sleep`、空等、反复无意义轮询来凑时间。

2. **最低阶段数量**
   - 至少完成 `12` 个可验收小阶段。
   - 每个小阶段必须有以下至少一种产物：
     - 新测试
     - 小范围源码拆分
     - 文档审计结果
     - 验证命令结果
     - checkpoint 更新

3. **最低轮次数量**
   - 至少完成 `3` 个推进轮。
   - 每个推进轮必须包含：
     - 只读审计
     - 至少 1 个测试或源码小改
     - 至少 1 组验证
     - checkpoint 更新

4. **最终验证**
   - 最后一轮必须跑总验证。
   - 如果某个验证 60 秒内跑不完，必须拆成更窄命令并记录。

5. **提交推送**
   - 最终必须显式 stage 实际修改文件。
   - 不允许 `git add -A`。
   - 提交并推送到 `webui/main`。

如果 12 个阶段做完但耗时未满 2 小时：

- 不允许 complete。
- 必须进入 **扩展阶段池** 继续做低风险小阶段。

如果耗时满 2 小时但阶段不足 12 个：

- 不允许 complete。
- 必须继续做阶段，直到达到 12 个。

如果遇到阻塞：

- 只有连续 3 轮同一阻塞且无法安全推进，才允许标记 blocked。
- blocked 前必须写清楚已完成阶段、阻塞原因、下一步最小解除条件。

---

## 🛡️ 3. 禁止事项

一句话：长跑不等于乱跑，危险操作仍然禁止。

禁止：

- 不跑真实训练。
- 不下载模型。
- 不删除、移动或清理用户数据。
- 不删除 `_legacy.py`。
- 不改 LoRA checkpoint key 格式。
- 不改 LoRA public API。
- 不改三轴路由语义：`use_moe_style` / `route_per_layer` / `router_source`。
- 不做 `git reset --hard`。
- 不做 `git checkout -- <path>`。
- 不做 force push。
- 不用 `git add -A`。
- 不为了凑时长做无意义格式化。
- 不为了凑阶段拆出没有价值的空 helper。

禁止默认触碰目录：

- `.venv/`
- `.worktrees/`
- `models/`
- `output/`
- `logs/`
- `post_image_dataset/`
- `configs/imported/`
- `configs/web-training-history/`
- `configs/web-training-queue/`
- `web/configs/`
- `bench/mfu/assets/`
- `tmp/`

---

## 🧭 4. 执行规则

一句话：每一轮都要“审计 -> 小改 -> 验证 -> checkpoint”，然后继续下一轮。

启动时必须做：

```bash
git status --short --branch
git log -1 --oneline --decorate
git rev-list --left-right --count HEAD...webui/main
date +%s
```

每轮开始必须做：

```bash
git status --short --branch
git diff --name-only
```

每轮结束必须做：

```bash
git diff --check
```

每 30 分钟左右必须做一次状态盘点：

- 已完成阶段数。
- 当前轮次。
- 已跑验证。
- 剩余安全阶段。
- 是否已达到 2 小时门槛。

不允许提前停止：

- 如果计划阶段提前完成，就进入扩展阶段池。
- 如果扩展阶段也完成，就做更多只读审计和测试保护。
- 只有满足硬完成条件后，才允许提交最终总结。

---

## 🧩 5. 推进轮总览

一句话：本目标至少跑 3 轮，每轮都有多个小阶段，避免一个闭环就结束。

| 轮次 | 主线 | 最低阶段数 | 预计耗时 | 风险 |
|---|---|---:|---:|---|
| R0 | 基线、目标启动、已完成文档确认 | 2 | 10-20 分钟 | Low |
| R1 | TASK-07 LoRA save/load 保护加宽 | 4 | 45-75 分钟 | Low-Medium |
| R2 | TASK-09 Web config facade/import surface 收口 | 4 | 45-75 分钟 | Low-Medium |
| R3 | TASK-10 type-check 试点小范围扩大 | 3 | 35-60 分钟 | Low-Medium |
| R4 | 总验证、checkpoint、提交推送 | 2 | 25-45 分钟 | Low |
| EXT | 扩展阶段池 | 不限 | 直到满足 2 小时 | Low-Medium |

最低要求：

- 至少完成 R0、R1、R2、R4。
- R3 可在风险过高时跳过，但必须用 EXT 补足阶段数和耗时。
- 总阶段数必须达到 12。

---

## ✅ 6. R0：基线和目标启动

一句话：先确认当前仓库、前置文档和线上状态，建立本轮长跑计数。

### S00 基线确认

目标：

- 确认本地 `main` 与 `webui/main` 状态。
- 记录本轮开始时间。

命令：

```bash
git status --short --branch
git log -1 --oneline --decorate
git rev-list --left-right --count HEAD...webui/main
date +%s
```

验收：

- 输出写进 checkpoint 的本轮记录草稿。

### S01 已完成目标归档确认

目标：

- 确认前两份目标书都已完成归档。
- 确认本文件是当前活跃长跑目标。

命令：

```bash
rg -n "状态：已完成归档|完成阶段|后续入口" docs/findings/project_cleanup_*goal_20260705.md
```

验收：

- checkpoint 记录“当前活跃目标为本文件”。

---

## 💾 7. R1：TASK-07 LoRA save/load 保护加宽

一句话：这一轮继续给 LoRA 保存/加载边界补保护，再做很小的 helper 拆分。

### S02 save metadata 空字典行为保护

目标：

- 明确 `metadata={}` 时是否写入 `ss_network_spec`。
- 明确 `metadata={"x": "y"}` 时是否保留现有 stamp 行为。

建议写入：

- `tests/test_lora_save_pipeline.py`
- 或 `tests/test_lora_network_construction.py`

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_network_construction.py -k "metadata or save_weights"
```

### S03 lora_save fallback 分流保护

目标：

- 覆盖 `save_variant` 缺省但 state_dict 含 `.lora_up_weight` 时，仍走 Hydra `*_moe.safetensors` 的 fallback。

建议写入：

- `tests/test_lora_save_pipeline.py`

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py
```

### S04 loading malformed key 追加边界

目标：

- 继续补 `_stack_lora_ups` / `_refuse_split_*` 的坏 key 报错测试。
- 优先覆盖专家编号不连续、q/k/v 缺组件、Chimera pool 缺一侧。

建议写入：

- `tests/test_lora_loading_keys.py`

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py
```

### S05 loading helper 只读审计

目标：

- 只读审计 `networks/lora_anima/loading.py`，列出下一个可拆但暂不拆的 helper 候选。

建议写入：

- `docs/findings/project_cleanup_checkpoint_20260705.md`

验收：

- 写清楚候选 helper、风险、为什么本轮不扩大。

### S06 小范围 helper 拆分候选

目标：

- 只有在 S02-S04 测试充分时，才允许拆一个很小 helper。
- 候选：
  - save metadata helper 的内部子函数。
  - loading key 分组的纯检测函数。

禁止：

- 不改 checkpoint key。
- 不改对外函数名。
- 不改保存文件名。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_lora_network_construction.py tests/test_factory_metadata_flow.py
git diff --check
```

---

## 🧱 8. R2：TASK-09 Web config facade/import surface 收口

一句话：这一轮不删除 `_legacy.py`，只把保留理由、导入面和兼容测试继续做扎实。

### S07 import surface grep 审计

目标：

- 查所有 `config_service` 和 `_legacy` 依赖。
- 输出“必须保留 / 可迁移 / 需要测试”的清单。

命令：

```bash
rg -n "config_service\\.|from web\\.services import config_service|from web\\.services\\.config import|web\\.services\\.config\\._legacy" web tests docs scripts
```

建议写入：

- `docs/findings/project_cleanup_checkpoint_20260705.md`
- 或新增 `docs/findings/config_import_surface_audit_20260705.md`

### S08 facade 常量导出补测试

目标：

- 继续保护 `_legacy.py` / `config_service.py` 仍要导出的 metadata 常量。
- 不新增不必要常量，只测真实外部依赖。

建议写入：

- `tests/test_web_config_service.py`

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "metadata_exports or legacy"
```

### S09 split module direct-import smoke

目标：

- 给一个尚未覆盖充分的 split module 补 direct import smoke。
- 重点是不能依赖 facade 已经初始化。

候选模块：

- `web.services.config.merge`
- `web.services.config.output_runs`
- `web.services.config.file_groups`
- `web.services.config.datasets`

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "import_without_facade_cycle or direct"
```

### S10 common.py 复用候选审计

目标：

- 只读找 split modules 里仍重复的 path/coercion/helper 逻辑。
- 先写清单，不急着迁移。

命令：

```bash
rg -n "expand_env_vars|resolve|display_path|coerce|toml\\.loads|read_text" web/services/config/*.py
```

验收：

- checkpoint 记录候选和风险。

### S11 一个低风险 common.py 复用

目标：

- 只选一个无副作用 helper 做复用。
- 不碰用户配置文件，不写 runtime 数据。

建议验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "common_config_helpers or legacy or merge or dataset or file_group"
git diff --check
```

---

## 🧪 9. R3：TASK-10 type-check 试点小范围扩大

一句话：类型检查只按一个小目录扩大，避免一次性全仓切换。

### S12 只读选择候选

目标：

- 找一个低依赖、低噪声的目录或脚本加入 pyright 试点。

候选：

- `scripts/config_compat.py`
- `scripts/config_explain.py`
- 已覆盖脚本旁边的小工具
- 某个纯文档/配置 helper 模块

命令：

```bash
sed -n '1,220p' pyproject.toml
timeout 60 .venv/bin/python tasks.py type-check
```

### S13 新候选单独 pyright

目标：

- 对候选路径单独跑 pyright。
- 如果噪声过多，记录原因，不强上。

命令示例：

```bash
timeout 60 .venv/bin/python tasks.py type-check <candidate-path>
```

### S14 小范围纳入默认 type-check

目标：

- 只有候选单独通过时，才纳入默认 `tasks.py type-check`。
- 同步测试或文档。

验证：

```bash
timeout 60 .venv/bin/python tasks.py type-check
git diff --check
```

如果 R3 风险过高：

- 跳过 R3。
- 必须改做 EXT 阶段补足阶段数和时间。

---

## 📚 10. R4：总验证、checkpoint、提交推送

一句话：最后一轮只做收口，不再扩大重构。

### S15 checkpoint 总结

必须写入：

- `docs/findings/project_cleanup_checkpoint_20260705.md`

必须记录：

- 完成阶段编号。
- 修改文件列表。
- 验证命令和结果。
- 跳过阶段原因。
- 未做事项：
  - 没真实训练。
  - 没下载模型。
  - 没改 checkpoint key。
  - 没删除 `_legacy.py`。
  - 没建立全仓类型检查，除非真的做了。

### S16 总验证

建议命令：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_lora_network_construction.py tests/test_factory_metadata_flow.py tests/test_global_router.py tests/test_network_cfg.py tests/test_router_compute.py
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "metadata_exports or legacy or dataset or file_group or preflight or merge or output_run or common_config_helpers"
timeout 60 .venv/bin/python tasks.py type-check
git diff --check
```

如果 Web config 宽筛 60 秒内跑不完：

- 拆成以下命令：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "metadata_exports or legacy or merge or common_config_helpers"
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "dataset and not runtime_preflight"
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "file_group or output_run"
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "preflight"
```

### S17 提交推送

命令：

```bash
git status --short --branch
git diff --name-only
git add <本轮实际修改文件>
git diff --cached --check
git commit -m "refactor: continue sustained cleanup"
git fetch webui --prune
git log webui/main..HEAD
git push webui main:main
git status --short --branch
git rev-parse HEAD webui/main
```

提交前检查：

- 阶段数是否 >= 12。
- 轮次数是否 >= 3。
- 是否满足 2 小时。
- 不满足则不能 commit final，必须进入 EXT。

---

## 🔁 11. EXT：扩展阶段池

一句话：如果计划做完但时间或阶段不足，就从这里继续拿任务，不允许提前 complete。

EXT 阶段按优先级选择，每次只做一个：

| 编号 | 目标 | 风险 |
|---|---|---|
| E01 | 给 `lora_save.py` 增加一个 metadata 透传边界测试 | Low |
| E02 | 给 `loading.py` 增加一个 malformed key 错误信息测试 | Low |
| E03 | 给 `create_network_from_weights` 增加一个 no-metadata actionable error 测试变体 | Low |
| E04 | 审计 `network.py::save_weights` 剩余可拆 metadata 逻辑，只写文档 | Low |
| E05 | 审计 `factory.py` 剩余 key sniff 逻辑，只写文档 | Low |
| E06 | 给 `config_service` facade 增加一个真实导入面兼容测试 | Low |
| E07 | 审计 `web/services/config/*.py` 的 direct-import 风险，只写文档 | Low |
| E08 | 给一个 split module 增加 direct import smoke | Low |
| E09 | 审计 `tasks.py type-check` 扩围候选，只写文档 | Low |
| E10 | 对一个候选路径单独跑 type-check 并记录结果 | Low |
| E11 | 更新 checkpoint 的“不能对外说”边界 | Low |
| E12 | 更新下一轮目标草稿，不作为完成替代 | Low |
| E13 | 检查 docs 中仍指向旧目标书的引用，只写修正 | Low |
| E14 | 补一个 `git diff --check -- docs/findings` 验证记录 | Low |
| E15 | 补一个 py_compile 针对本轮源码文件 | Low |

EXT 规则：

- 每做一个 EXT 阶段，阶段计数 +1。
- 每做 3 个 EXT 阶段，必须跑一次相关测试。
- 不能用纯文档 EXT 连续超过 3 个；第 4 个必须是测试或源码小改。

---

## 🧾 12. 已完成目标 Prompt 历史记录

一句话：原 prompt 已在提交 `bd591b83` 收口，后续不要复制旧目标重复施工。

```text
请按 docs/findings/project_cleanup_sustained_goal_20260706.md 执行跨子系统强制长跑项目清理目标。
```

---

## 📌 13. 不能夸大的边界

一句话：强制长跑完成后，也只能按证据说话。

不能说：

- 不能说全仓技术债清完。
- 不能说 LoRA save/load/builder/router 已彻底拆完，除非对应链路真的完成并有测试。
- 不能说 `_legacy.py` 可以删除，除非外部 import surface 已迁移并验证。
- 不能说建立全仓 type-check，除非默认 type-check 真覆盖全仓并通过。
- 不能说训练性能提升，除非跑过 bench 或真实训练证据。

可以说：

- 完成了哪些阶段。
- 跑了哪些验证。
- 哪些 helper 被小步拆分。
- 哪些文档入口已经归档或替换。
- 哪些风险仍然保留。
