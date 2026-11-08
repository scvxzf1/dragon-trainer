# 项目清理长目标执行书：连续多阶段小步推进

一句话：这份文档给后续 Agent 一个可以连续跑几个小时的目标包，要求一轮接一轮小步推进，而不是完成一个小闭环就停。

状态：已完成归档。
完成提交：`f74b8255 refactor: continue staged project cleanup`
完成阶段：`P0`、`P1`、`P2`、`P5`、`P6`、`P7`
后续入口：`docs/findings/project_cleanup_sustained_goal_20260706.md`

> ⚠️ 这份文档现在只作为历史记录保留，不再作为新的活跃目标执行。
> 下一阶段请使用 `project_cleanup_sustained_goal_20260706.md`。

日期：2026-07-05
适用仓库：`/home/scv/nvme0n1p1/训练器相关/anima_lora`
默认发布目标：本地 `main` -> `webui/main`
建议总时长：3 到 5 小时

---

## 🎯 1. 总目标

一句话：用很多个低风险小阶段，继续减少项目技术债，同时保证每一步都能验证、能回滚。

目标名称：

```text
连续推进项目清理：LoRA core 小拆分 + Web config legacy 收口 + 文档检查点维护
```

总目标：

- 继续推进 `TASK-07`：LoRA builder / router / load / save 的拆分前保护和低风险 helper 拆分。
- 继续推进 `TASK-09`：Web config `_legacy.py` facade 的剩余薄化和 split module 复用。
- 每个阶段都必须小、窄、可测，不做大重构。
- 不跑真实训练，不下载模型，不碰用户数据目录。
- 每完成一阶段，要更新检查点文档，再继续下一阶段；不要完成一小步就停。

完成定义：

- 至少完成 4 个小阶段，或遇到明确阻塞。
- 每个完成阶段都有测试或检查证据。
- 高风险项没有被绕过。
- `docs/findings/project_cleanup_checkpoint_20260705.md` 已更新。
- 最终工作树干净，且改动已提交并推送到 `webui/main`。

---

## 🛡️ 2. 总约束

一句话：这些约束是防止长时间自动推进时失控的护栏。

禁止事项：

- 不跑真实训练。
- 不下载模型。
- 不清理、删除、移动用户数据。
- 不做 `git reset --hard`。
- 不做 `git checkout -- <path>`。
- 不做 force push。
- 不用 `git add -A`。
- 不改 checkpoint key 格式。
- 不改 LoRA public API。
- 不改三轴路由语义：`use_moe_style` / `route_per_layer` / `router_source`。

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

每阶段固定开头检查：

```bash
git status --short --branch
git diff --name-only
```

每阶段固定收尾检查：

```bash
git diff --check
```

---

## 🧭 3. 长目标执行规则

一句话：后续 Agent 必须按阶段推进，完成一个阶段后自动进入下一个阶段。

执行规则：

1. 先读：
   - `AGENTS.md`
   - `docs/findings/project_cleanup_checkpoint_20260705.md`
   - 本文件
2. 先做只读审计，再写测试，再改代码。
3. 每阶段只允许一个主写入范围。
4. 如果一个阶段超过 60 分钟还没有可验证产物，收缩阶段范围。
5. 如果测试失败，先修当前阶段，不开新阶段。
6. 如果发现高风险重构需求，写入 checkpoint 的“后续风险”，不要当场扩大。
7. 阶段完成后继续下一阶段，直到完成定义满足或出现阻塞。

推荐并行方式：

- 只读审计可以并行。
- 写同一文件时必须串行。
- 多代理结果必须先汇总，再进入下一轮。

---

## 🧩 4. 阶段计划总览

一句话：这 8 个阶段按低风险到中风险排序，足够支撑几个小时连续推进。

| 阶段 | 主线 | 预计耗时 | 风险 | 是否可跳过 |
|---|---|---:|---|---|
| P0 | 基线确认和检查点复读 | 15 分钟 | Low | 不可跳过 |
| P1 | TASK-07 loader detection 保护测试加宽 | 30-45 分钟 | Low | 可跳过 |
| P2 | TASK-07 router metadata 纯 helper 拆分 | 30-45 分钟 | Low-Medium | 可跳过 |
| P3 | TASK-07 save variant 判定 characterization | 30-45 分钟 | Low | 可跳过 |
| P4 | TASK-07 save metadata 组装 helper 拆分 | 30-60 分钟 | Medium | 可跳过 |
| P5 | TASK-09 `_legacy.py` 剩余函数分类审计 | 30 分钟 | Low | 可跳过 |
| P6 | TASK-09 split module 复用 `common.py` 小步推进 | 45-60 分钟 | Low-Medium | 可跳过 |
| P7 | 文档和验证收口 | 30-45 分钟 | Low | 不可跳过 |

如果时间还够，可以追加：

| 追加阶段 | 主线 | 预计耗时 | 风险 |
|---|---|---:|---|
| P8 | LoRA `loading.py` split/refuse helper 补边界测试 | 45-60 分钟 | Medium |
| P9 | Web config import surface grep 审计报告 | 30-45 分钟 | Low |

---

## ✅ 5. P0：基线确认和检查点复读

一句话：先确认当前仓库真干净、线上目标正确、上一轮成果不是记忆里的幻觉。

输入范围：

- `git status --short --branch`
- `git remote -v`
- `git branch -vv`
- `git log -1 --oneline --decorate`
- `docs/findings/project_cleanup_checkpoint_20260705.md`

验收标准：

- 确认本地在 `main`。
- 确认 `main` 跟踪 `webui/main`。
- 确认没有未提交改动，或先记录已有改动归属。
- 确认当前 TASK-07 / TASK-09 状态。

建议命令：

```bash
git status --short --branch
git remote -v
git branch -vv
git log -1 --oneline --decorate
rg -n "TASK-07|TASK-09|不能对外说|下一步" docs/findings/project_cleanup_checkpoint_20260705.md
```

---

## 🧪 6. P1：TASK-07 loader detection 保护测试加宽

一句话：先继续补测试，保护 `create_network_from_weights()` 的 key sniff 行为。

目标：

- 给 `factory.py` 的 checkpoint key 扫描补 1 到 2 条 characterization tests。
- 优先覆盖旧格式拒绝、Chimera dual-A 标记、StackedExperts 判定中的一个。

建议候选：

- 旧 `_hydra_router.*` key 必须抛出明确错误。
- 混合 OrthoHydra/plain OrthoLoRA 能保留 `hydra_router_names`。
- StackedExperts 的 3D `lora_down_weight` 不被误判为 shared-A Hydra。

允许写入：

- `tests/test_factory_metadata_flow.py`
- `tests/test_lora_network_construction.py`
- 必要时新增一个窄测试文件，例如 `tests/test_lora_factory_key_scan.py`

禁止写入：

- `network.py` 主构造循环
- `lora_save.py`
- checkpoint 格式逻辑

验收标准：

- 新增测试先失败或明确保护当前行为。
- 修复后相关测试通过。

建议验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_factory_metadata_flow.py tests/test_lora_network_construction.py
git diff --check
```

---

## 🧱 7. P2：TASK-07 router metadata 纯 helper 拆分

一句话：只拆 metadata 解析或路由名单计算，不碰构造和保存主逻辑。

目标：

- 从 `create_network_from_weights()` 中继续抽一个纯 helper。
- 优先选择：
  - three-axis stamp 读取和规范化。
  - `hydra_router_names` 计算。
  - `sigma_router_names` / `fei_router_names` 推断中的局部纯逻辑。

允许写入：

- `networks/lora_anima/factory.py`
- 对应测试文件

禁止写入：

- `networks/lora_anima/network.py` 大构造循环。
- `networks/lora_save.py` 保存分流。

验收标准：

- public API 不变。
- checkpoint key 不变。
- 原有测试全部通过。
- 新 helper 是私有函数，不外露成新接口。

建议验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_factory_metadata_flow.py tests/test_lora_network_construction.py tests/test_network_cfg.py
git diff --check
```

---

## 💾 8. P3：TASK-07 save variant 判定 characterization

一句话：先用测试钉住保存分流行为，再决定要不要拆保存 helper。

目标：

- 给 `lora_save.save_network_weights()` 的 variant dispatch 补 characterization tests。
- 覆盖至少一个 MoE 保存文件名约定：
  - Hydra -> `*_moe.safetensors`
  - StackedExperts -> `*_moe.safetensors`
  - ChimeraHydra -> `*_chimera.safetensors`

允许写入：

- `tests/test_global_router.py`
- 或新增 `tests/test_lora_save_pipeline.py`

禁止写入：

- 先不改 `lora_save.py`。

验收标准：

- 测试能证明保存文件名、metadata 透传、关键 key 形态不变。
- 不要求真实训练。

建议验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_global_router.py
git diff --check
```

---

## 🧰 9. P4：TASK-07 save metadata 组装 helper 拆分

一句话：如果 P3 已有保护，可以把保存 metadata stamp 拆成窄 helper。

目标：

- 从 `LoRANetwork.save_weights()` 中抽出私有 helper。
- 只抽 metadata stamp，不改 `state_dict()`、不改 `lora_save.save_network_weights()` 调用。

候选 helper：

```python
def _stamp_lora_save_metadata(metadata: dict[str, str], cfg: LoRANetworkCfg, spec: NetworkSpec) -> None:
    ...
```

注意：

- 当前 `ss_network_spec` 的写入依赖 `metadata` 是否非空，不能无意改变。
- MoE 分支 `metadata or {}` 的保存行为不能改变。
- 不要新增公开 API。

允许写入：

- `networks/lora_anima/network.py`
- 对应测试文件

验收标准：

- P3 的 characterization tests 仍通过。
- `save_weights()` 行为不变。

建议验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_network_construction.py tests/test_global_router.py tests/test_factory_metadata_flow.py
git diff --check
```

---

## 🔍 10. P5：TASK-09 `_legacy.py` 剩余函数分类审计

一句话：先只读分类，不急着删 `_legacy.py`。

目标：

- 统计 `_legacy.py` 中剩余非转发函数。
- 分类为：
  - shim 调度
  - facade 状态恢复
  - 真业务逻辑残留
  - 测试专用兼容
- 写入检查点文档或新增审计小节。

允许写入：

- `docs/findings/project_cleanup_checkpoint_20260705.md`
- 必要时新增 `docs/findings/config_legacy_remaining_audit_20260705.md`

禁止写入：

- 先不改 `_legacy.py`。

建议命令：

```bash
rg -n "^def |^async def |__all__|TODO|legacy|facade" web/services/config/_legacy.py web/services/config
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "legacy or facade or raw_files or file_group"
git diff --check
```

---

## 🧩 11. P6：TASK-09 split module 复用 `common.py` 小步推进

一句话：只挑一个重复 helper 做复用，不做大范围迁移。

目标：

- 从 TASK-09 审计结果中挑一个低风险重复逻辑。
- 让 split module 复用 `common.py` 里的已有 helper。
- 保留 `_legacy.py` facade 行为。

优先候选：

- 路径展开 / 归一化小 helper。
- scalar coercion 小 helper。
- mtime / label / id 生成这类无副作用 helper。

允许写入：

- `web/services/config/common.py`
- 一个 split module，例如 `datasets.py`、`file_groups.py`、`output_runs.py`
- 对应测试

禁止写入：

- 不删除 `_legacy.py`。
- 不改用户配置目录。
- 不写入 `configs/imported/`、`configs/web-training-history/`、`configs/web-training-queue/`。

建议验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "dataset or file_group or output_run or legacy"
git diff --check
```

---

## 📚 12. P7：文档和验证收口

一句话：最后必须把本轮做了什么、没做什么、不能夸大的边界写清楚。

必须更新：

- `docs/findings/project_cleanup_checkpoint_20260705.md`

应写入：

- 完成了哪些阶段。
- 改了哪些文件。
- 跑了哪些验证。
- 哪些没做：
  - 没跑真实训练。
  - 没下载模型。
  - 没改 checkpoint key。
  - 没删除 `_legacy.py`。
  - 没建立全仓类型检查。

总验证建议：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_network_construction.py tests/test_factory_metadata_flow.py tests/test_global_router.py tests/test_network_cfg.py tests/test_router_compute.py
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "legacy or dataset or file_group or preflight or merge or output_run"
timeout 60 .venv/bin/python tasks.py type-check
git diff --check
```

---

## 🚀 13. 提交和推送规则

一句话：最后只提交目标文件，推送到 `webui/main`。

提交前：

```bash
git status --short --branch
git diff --name-only
git diff --check
```

显式 stage，示例：

```bash
git add <本轮实际修改的源码/测试/文档文件>
git diff --cached --check
```

提交信息建议：

```bash
git commit -m "refactor: continue staged project cleanup"
```

推送：

```bash
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

## 🧾 14. 已完成目标 Prompt 历史记录

一句话：原可执行 prompt 已完成，避免后续误用同一目标重复施工。

原目标已在提交 `f74b8255` 收口，完成记录见
`docs/findings/project_cleanup_checkpoint_20260705.md` 第 11 节。

新的可复制目标请使用：

```text
请按 docs/findings/project_cleanup_sustained_goal_20260706.md 执行跨子系统强制长跑项目清理目标。
```

---

## 📌 15. 不能夸大的边界

一句话：长目标完成后，也只能按证据说话。

不能说：

- 不能说全仓技术债清完。
- 不能说 LoRA builder/router/load/save 已经彻底拆完，除非真的完成并有测试。
- 不能说 `_legacy.py` 可以删除，除非外部 import surface 已迁移并验证。
- 不能说全仓类型检查已建立，除非 `tasks.py type-check` 扩到全仓并通过。
- 不能说训练性能变好了，除非跑过对应 bench 或真实训练证据。

可以说：

- 完成了哪些小阶段。
- 哪些测试通过。
- 哪些文件被小步拆分。
- 哪些风险仍保留。
