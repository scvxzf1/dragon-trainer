# 项目清理下一阶段目标书：保存链路与导入面收口

一句话：这份文档接在已完成的 `project_cleanup_long_running_goal_20260705.md` 后面，继续给 Agent 一个能连续推进数小时的多阶段目标。

日期：2026-07-05
状态：已完成归档
完成阶段：`N0`、`N1`、`N2`、`N3`、`N4`、`N5`、`N6`
完成记录：见 `docs/findings/project_cleanup_checkpoint_20260705.md` 第 13 节
后续入口：`docs/findings/project_cleanup_sustained_goal_20260706.md`
前置完成提交：`f74b8255 refactor: continue staged project cleanup`
前置完成阶段：`P0`、`P1`、`P2`、`P5`、`P6`、`P7`
默认发布目标：本地 `main` -> `webui/main`
建议总时长：3 到 5 小时

> ⚠️ 这份文档现在只作为历史目标记录保留，不再作为新的活跃目标重复执行。
> 下一次要跑长目标时，请使用 `project_cleanup_sustained_goal_20260706.md`。

---

## 🎯 1. 总目标

一句话：下一阶段不再重复已完成阶段，而是继续推进 LoRA 保存链路保护、加载 helper 边界和 Web config 导入面审计。

目标名称：

```text
下一阶段项目清理：LoRA save/load 边界保护 + Web config import surface 收口
```

总目标：

- 继续 `TASK-07`：补齐 `lora_save.py` 保存分流 characterization tests，再做最多一个低风险保存 metadata helper 拆分。
- 继续 `TASK-07`：给 `loading.py` 的 split/refuse helper 补边界测试，但不改 checkpoint key。
- 继续 `TASK-09`：审计 `web.services.config_service` / `_legacy.py` / split modules 的外部导入面，明确哪些 facade 还必须保留。
- 更新 checkpoint，把完成项、验证证据、未做边界写清楚。

完成定义：

- 至少完成 4 个小阶段，或遇到明确阻塞。
- 每个阶段都有测试、命令输出或文档证据。
- 不改 public API、checkpoint key 格式或三轴路由语义。
- 不删除 `_legacy.py`。
- 最后提交并推送到 `webui/main`。

---

## 🛡️ 2. 总约束

一句话：本阶段只做小步保护和边界收口，不做危险清理。

禁止事项：

- 不跑真实训练。
- 不下载模型。
- 不删除或移动用户数据。
- 不删除 `_legacy.py`。
- 不改 LoRA checkpoint key 格式。
- 不改 LoRA public API。
- 不改三轴路由语义：`use_moe_style` / `route_per_layer` / `router_source`。
- 不做 `git reset --hard`。
- 不做 force push。
- 不用 `git add -A`。

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

## 🧭 3. 执行规则

一句话：每个阶段都要小步闭环，完成后继续下一阶段，不要只做一条测试就停。

执行规则：

1. 先读：
   - `AGENTS.md`
   - `docs/findings/project_cleanup_checkpoint_20260705.md`
   - `docs/findings/project_cleanup_long_running_goal_20260705.md`
   - 本文件
2. 先只读审计，再补测试，再改代码。
3. 每阶段只允许一个主写入范围。
4. 如果一个测试 60 秒内跑不完，拆成更窄的测试集合并记录。
5. 如果发现需要大改，写进 checkpoint 的后续风险，不在本轮扩大。
6. 每完成一个阶段，更新 checkpoint，再继续下一阶段。

固定开头检查：

```bash
git status --short --branch
git log -1 --oneline --decorate
git diff --name-only
```

固定收尾检查：

```bash
git diff --check
```

---

## 🧩 4. 阶段总览

一句话：下面 7 个阶段接续上轮，足够支撑几个小时连续推进。

| 阶段 | 主线 | 预计耗时 | 风险 | 是否可跳过 |
|---|---|---:|---|---|
| N0 | 基线复核和已完成文档确认 | 10-15 分钟 | Low | 不可跳过 |
| N1 | TASK-07 save variant characterization | 35-55 分钟 | Low | 可跳过 |
| N2 | TASK-07 save metadata helper 小拆分 | 35-60 分钟 | Medium | 可跳过 |
| N3 | TASK-07 loading split/refuse helper 边界测试 | 45-60 分钟 | Medium | 可跳过 |
| N4 | TASK-09 config import surface 审计 | 30-45 分钟 | Low | 可跳过 |
| N5 | TASK-09 facade 保留清单和风险文档 | 30-45 分钟 | Low | 可跳过 |
| N6 | 文档、验证、提交和推送收口 | 30-45 分钟 | Low | 不可跳过 |

最低完成要求：

- 必须完成 `N0` 和 `N6`。
- 中间至少完成 2 个阶段。
- 若跳过某阶段，必须在 checkpoint 写明原因。

---

## ✅ 5. N0：基线复核和已完成文档确认

一句话：历史执行时曾确认上轮文档已归档，并确认本文件是当时的活跃入口。

目标：

- 确认当前 `main` 与 `webui/main` 对齐。
- 确认 `project_cleanup_long_running_goal_20260705.md` 已标记为完成归档。
- 确认本文件在当时是新的活跃目标入口；当前已完成归档。

建议命令：

```bash
git status --short --branch
git log -1 --oneline --decorate
rg -n "状态：已完成归档|后续入口|完成阶段" docs/findings/project_cleanup_long_running_goal_20260705.md docs/findings/project_cleanup_next_stage_goal_20260705.md
```

验收标准：

- 工作区状态已记录。
- 旧长目标不会再被误作为活跃目标。
- checkpoint 后续会指向本文件。

---

## 💾 6. N1：TASK-07 save variant characterization

一句话：先用测试钉住保存分流，不急着拆保存代码。

目标：

- 给 `networks/lora_save.py::save_network_weights()` 补 1 到 2 条 characterization tests。
- 优先覆盖保存文件名和 key 形态：
  - Hydra / OrthoHydra -> `*_moe.safetensors`
  - StackedExperts -> `*_moe.safetensors`
  - ChimeraHydra -> `*_chimera.safetensors`

建议写入：

- `tests/test_global_router.py`
- 或新增 `tests/test_lora_save_pipeline.py`

禁止写入：

- 先不改 `networks/lora_save.py`。
- 不改保存格式。

验收标准：

- 测试能证明文件名、metadata 透传、关键 key 形态不变。
- 不要求真实训练。

建议验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_global_router.py
git diff --check
```

---

## 🧰 7. N2：TASK-07 save metadata helper 小拆分

一句话：N1 有保护后，才允许拆 `save_weights()` 里的 metadata stamp 逻辑。

目标：

- 从 `LoRANetwork.save_weights()` 中抽一个私有 metadata helper。
- 只移动 metadata 组装，不改 `state_dict()`，不改 `lora_save.save_network_weights()` 调用。

候选 helper：

```python
def _stamp_lora_save_metadata(metadata, cfg, spec) -> None:
    ...
```

注意事项：

- 当前 `ss_network_spec` 的写入行为不能无意改变。
- `metadata is None` 和空 dict 的现有行为要用测试保护。
- MoE 分支 `metadata or {}` 的保存行为不能改变。

建议写入：

- `networks/lora_anima/network.py`
- 对应测试文件

验收标准：

- N1 测试仍通过。
- LoRA 综合测试仍通过。
- public API 不变。

建议验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_network_construction.py tests/test_factory_metadata_flow.py tests/test_global_router.py
git diff --check
```

---

## 🔗 8. N3：TASK-07 loading split/refuse helper 边界测试

一句话：给 `loading.py` 的 key 转换边界补测试，不碰真实 checkpoint。

目标：

- 给以下 helper 补至少 1 条边界测试：
  - `_stack_lora_ups`
  - `_refuse_split_hydra_keys`
  - `_refuse_split_stacked_experts_keys`
  - `_refuse_split_chimera_keys`
  - `_refuse_unfused_attn_lora_keys`

优先覆盖：

- q/k/v split -> fused round trip。
- plain fallback leg 不被 Hydra/StackedExperts 误收。
- malformed key 有明确错误。

建议写入：

- `tests/test_global_router.py`
- 或新增 `tests/test_lora_loading_keys.py`

禁止写入：

- 不改 checkpoint key 格式。
- 不改 loader 对外函数签名。

验收标准：

- 新测试通过。
- 没有改变现有保存/加载格式。

建议验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_global_router.py tests/test_factory_metadata_flow.py tests/test_lora_network_construction.py
git diff --check
```

---

## 🔍 9. N4：TASK-09 config import surface 审计

一句话：只读梳理外部还在依赖哪些 config facade 名字，不急着删。

目标：

- 审计 `web.services.config_service`、`web.services.config._legacy` 和 split modules 的导入面。
- 找出仍被测试、Web 路由、服务层或文档引用的 facade 名字。
- 形成“必须保留 / 可迁移 / 需补测试”清单。

建议命令：

```bash
rg -n "config_service\\.|from web\\.services import config_service|from web\\.services\\.config import|web\\.services\\.config\\._legacy" web tests docs scripts
rg -n "__all__|def _call_|def _make_|def _restore_|def " web/services/config/_legacy.py web/services/config/*.py
```

建议写入：

- `docs/findings/project_cleanup_checkpoint_20260705.md`
- 必要时新增 `docs/findings/config_import_surface_audit_20260705.md`

验收标准：

- 不改源码也可以完成。
- 有明确清单说明 `_legacy.py` 为什么暂不能删除。

---

## 🧾 10. N5：TASK-09 facade 保留清单和风险文档

一句话：把 N4 的审计结果变成可执行的下一步清单。

目标：

- 给 checkpoint 增加 facade 保留清单：
  - 必须保留的常量。
  - 必须保留的 public functions。
  - 只剩 shim 调度的内部函数。
  - 删除前必须迁移的 import surface。
- 如发现低风险缺口，可只补测试，不做删除。

建议写入：

- `docs/findings/project_cleanup_checkpoint_20260705.md`
- `tests/test_web_config_service.py` 中很窄的 facade 兼容测试

禁止写入：

- 不删除 `_legacy.py`。
- 不改配置文件根目录。

建议验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "metadata_exports or legacy or merge or common_config_helpers"
git diff --check
```

---

## 📚 11. N6：文档、验证、提交和推送收口

一句话：最后把完成项和边界写清楚，再显式提交推送。

必须更新：

- `docs/findings/project_cleanup_checkpoint_20260705.md`

应记录：

- 完成了哪些 N 阶段。
- 改了哪些文件。
- 跑了哪些测试。
- 哪个测试如果 60 秒内未跑完，要写清楚拆分验证结果。
- 没做哪些事：
  - 没跑真实训练。
  - 没下载模型。
  - 没改 checkpoint key。
  - 没删除 `_legacy.py`。
  - 没建立全仓类型检查。

总验证建议：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_network_construction.py tests/test_factory_metadata_flow.py tests/test_global_router.py tests/test_network_cfg.py tests/test_router_compute.py
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "metadata_exports or legacy or dataset or file_group or preflight or merge or output_run or common_config_helpers"
timeout 60 .venv/bin/python tasks.py type-check
git diff --check
```

提交推送：

```bash
git status --short --branch
git diff --name-only
git add <本轮实际修改文件>
git diff --cached --check
git commit -m "test: cover next cleanup boundaries"
git fetch webui --prune
git log webui/main..HEAD
git push webui main:main
```

推送后核验：

```bash
git status --short --branch
git log -1 --oneline --decorate
git rev-parse HEAD webui/main
```

---

## 🧾 12. 后续目标入口

一句话：这份目标书已经完成归档，后续不要再复制旧 N 阶段目标重复执行。

```text
请按 docs/findings/project_cleanup_sustained_goal_20260706.md 执行跨子系统强制长跑项目清理目标。
```

---

## 📌 13. 不能夸大的边界

一句话：下一阶段即使完成，也只能按证据说话。

不能说：

- 不能说 LoRA builder/router/load/save 已经彻底拆完，除非真的完成保存、加载、构造三条链路并有测试。
- 不能说 `_legacy.py` 可以删除，除非外部 import surface 已迁移并验证。
- 不能说全仓类型检查已建立，除非 `tasks.py type-check` 扩到全仓并通过。
- 不能说训练性能变好了，除非跑过 bench 或真实训练证据。

可以说：

- 完成了哪些 N 阶段。
- 哪些测试通过。
- 哪些文档入口已经归档。
- 哪些风险仍保留。
