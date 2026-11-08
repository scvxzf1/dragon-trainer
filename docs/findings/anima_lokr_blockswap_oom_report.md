# Anima LoKr Block Swap OOM 技术报告

状态：研究记录 / 阶段快照
适用版本：以文中日期、提交和运行环境为准；不作为当前 main 操作说明

日期：2026-06-04

## 结论摘要

这轮问题不是 `block swap` 没生效，也不是 H2D 等待拖死训练。`block_swap_profile.jsonl` 反复显示 bf16 frozen block H2D 大约 `14-19ms`，实际 `wait_ms` 大多低于 `1ms`。真正的 OOM 主因是 LoKr 在 Anima DiT 的宽 Linear 上额外制造了训练峰值：先是完整 Kronecker 权重 / fp32 投影临时张量，然后是 custom path 里的整块输出副本，最后暴露为 DiT forward 本身只剩几十 MiB 的临界余量。

已完成修复：

- `use_custom_down_autograd=true` 现在会启用 `196` 个 LoKr module 的 `use_custom_lokr_autograd`。
- LoKr custom projection 不再 materialize `torch.kron(w1, w2)`。
- LoKr projection 改为 output-factor slice + token-row chunk，避免一次性 fp32 大临时。
- LoKr custom forward 不再 `torch.empty_like(org_forwarded)` 分配完整 result，而是在 frozen Linear 输出上按 slice 原地累加 delta。
- LoKr projection 新增 factor group 计算，默认 `lokr_factor_group_size=8`，减少重复的 `x @ w2.T` 投影。
- LoKr G8 进一步改成 fused row-chunk delta apply：不再把每个 output-factor group 的完整 fp32 delta tensor 返回给 Python，再写入 base output；而是在 custom autograd forward 里按 row chunk 直接写入 frozen Linear 输出，并在 backward 重算 delta。
- WebUI 资源快捷按钮新增 `LoKr 16G`，一键设置实测救场参数。

最终建议：

- 普通 `balanced_16g` 不改，继续面向普通 LoRA，默认 `blocks_to_swap=12`。
- LoKr 在 15.58GiB 级别显卡上不要指望 `balanced_16g` 或 `graft=20` 稳定训练。
- LoKr 手动救场推荐：

```toml
blocks_to_swap = 23
block_swap_transfer_dtype = "bf16"
selective_checkpoint = "off"
gradient_checkpointing = false
unsloth_offload_checkpointing = false
torch_compile = true
attn_mode = "flash"
lokr_factor_group_size = 8
block_swap_profile_jsonl = "auto"
memory_probe_jsonl = "auto"      # 首次验证建议打开
memory_probe_max_steps = 3
```

`blocks_to_swap=22` 是本轮三步短跑里能通过的最低点，但最小 forward 后余量只有约 `90MiB`，不建议作为用户推荐值。`blocks_to_swap=23` 是当前 LoKr 16G 推荐块交换基线；最新 G8 300-step 耐久短跑通过，但 compiled forward 仍会出现约 `46MiB` 的瞬时低点，所以长期训练仍建议配合 allocator 环境变量并保持后台显存占用很低。

补充消融显示：如果 `blocks_to_swap=23` 在用户环境仍偶发 OOM，优先尝试 allocator 环境变量，而不是单独打开 `mlp_only` 或盲目加到 `blocks_to_swap=24`：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
```

D36 使用该 allocator 配置三步通过；D37 的 `selective_checkpoint=mlp_only` 和 D38 的 `blocks_to_swap=24` 都在第二步 forward OOM，错误点仍是 compiled DiT block 内的 MLP/attention 临时 buffer。后续 50-step 复测也确认：更细粒度的 `mlp_layer1_only` / `peak_blocks_mlp_layer1` / `peak_blocks_mlp` 没有把最低 free 拉到 `300MiB+`；fused LoKr delta apply 降低了 `max allocated`，但 allocator `reserved` 仍会把 NVML free 压到约 `66MiB`。

## 固定约束

本轮消融遵守用户约束：

- 不关闭 `torch_compile`。
- 不调整注意力后端，保持 `attn_mode="flash"`。
- 不启用 Unsloth offload。
- 不删除用户训练历史、缓存、模型或输出。

## 数据来源

用户失败任务：

| history task | cache run | 关键结果 |
| --- | --- | --- |
| `20260603-161431-training-imported-ichika87_style--tag-es1` | `ichika87_style--tag-es1-20260603-161318` | LoKr `blocks_to_swap=16`，step0 forward OOM |
| `20260603-161744-training-imported-ichika87_style--tag-es1` | `ichika87_style--tag-es1-20260603-161631` | LoKr `blocks_to_swap=12 + mlp_only`，step0 forward OOM |

短实验目录：

```text
/tmp/anima-lokr-blockswap-phaseA/
/tmp/anima-lokr-group-ab/
```

每个实验保留：

- `D*.progress.jsonl`
- `D*.memory_probe.jsonl`
- `D*.block_swap_profile.jsonl`
- `D*.stderr.log`
- `D*.stdout.log`
- `G*.progress.jsonl`
- `G*.memory_probe.jsonl`
- `G*.block_swap_profile.jsonl`

## 根因链路

原始 LoKr 路径的问题分三层：

1. `networks/lora_anima/factory.py` 以前只识别普通 LoRA 的 `use_custom_down_autograd`，不会打开 LoKr 的 `use_custom_lokr_autograd`。
2. LoKr training forward 会执行 `torch.kron(self.lokr_w1, self.lokr_w2)`，再用 `F.linear(x.float(), weight.float())`，在 Anima 的宽 MLP/attention projection 上制造大 fp32 临时。
3. 第一版 no-kron custom path 仍然会 `torch.empty_like(org_forwarded)` 创建完整输出副本；在 4096 token 的 MLP `layer1` 上，这个副本约 `66MiB`，D19/D20 都死在这里。

修掉第 3 层后，D21 的 OOM 点后移到 base `Linear` 自身输出的 `18MiB` 分配。这说明 LoKr 临时峰值确实被削掉了，但 20 个 swapped blocks 仍不够稳定，剩余问题变成整体 DiT forward 峰值太贴 16G 上限。

## 实验结果

下表中的 `avg_s/step` 是从 run_start 到 run_end 粗算，包含短跑首步 warmup 和保存开销；只用于同组近似比较，不作为完整训练吞吐。

| run | blocks | status | final_step | avg_s/step | min forward free | max allocated | max reserved | wait p50/p95/max | h2d p50/p95/max | 失败点 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| D16 | 20 | ok | 1 | 49.3s | 0.283GiB | 13.648GiB | 14.234GiB | 0.040/0.678/0.869ms | 14.37/15.68/17.02ms | 单步通过 |
| D18 | 19 | ok | 1 | 44.6s | 0.929GiB | 13.416GiB | 13.729GiB | 0.043/0.672/0.873ms | 14.05/15.40/16.85ms | 单步通过 |
| D19 | 19 | error | 1 | 46.2s | 0.931GiB | 13.657GiB | 14.207GiB | 0.043/0.681/0.821ms | 14.29/16.45/16.61ms | step1 forward，`66MiB` full result |
| D20 | 20 | error | 1 | 46.3s | 1.043GiB | 13.525GiB | 14.201GiB | 0.042/0.660/0.924ms | 14.34/16.35/17.32ms | step1 forward，`66MiB` full result |
| D21 | 20 | error | 1 | 47.6s | 1.034GiB | 13.427GiB | 14.209GiB | 0.044/0.681/0.901ms | 13.99/14.71/15.23ms | step1 forward，base Linear `18MiB` |
| D22 | 21 | error | 0 | n/a | n/a | 13.310GiB | 13.855GiB | 0.517/0.750/0.771ms | 13.82/14.50/14.77ms | step0 final_layer `34MiB` |
| D23 | 22 | ok | 3 | 30.7s | 0.090GiB | 13.487GiB | 14.092GiB | 0.333/0.872/1.047ms | 14.04/17.25/18.49ms | 三步通过但余量过薄 |
| D24 | 23 | ok | 3 | 26.5s | 0.868GiB | 13.009GiB | 13.279GiB | 0.073/0.682/1.006ms | 15.58/17.85/19.55ms | 三步通过，推荐手动值 |
| D36 | 23 + allocator | ok | 3 | 30.4s | 1.084GiB | 13.301GiB | 14.443GiB | 0.350/0.901/1.007ms | 16.35/18.03/19.29ms | `expandable_segments` 三步通过 |
| D37 | 23 + mlp_only | error | 1 | n/a | 0.047GiB | 13.004GiB | 14.463GiB | 0.044/0.888/0.978ms | 16.64/19.01/19.70ms | step1 forward，compiled block `34MiB` |
| D38 | 24 | error | 1 | n/a | 0.099GiB | 12.747GiB | 14.428GiB | 0.469/0.734/0.867ms | 16.54/18.76/20.53ms | step1 MLP layer1 `66MiB` |

说明：D24 是前期 23-block 无 allocator 的 3-step baseline，原始临时文件已不在当前 `/tmp` 目录；它的结论现在由 D36/D37/D38 的可复现短跑再次确认。

关键观察：

- D19/D20 的 `66MiB` OOM 被 LoKr 原地累加修复消除。
- D21 失败点后移到 base Linear，说明 LoKr custom path 已经不再是最先爆的分配点。
- D23 虽然通过，但 step1/2 forward 后只有约 `90MiB`，长期训练、采样或桌面后台占用都可能让它再次 OOM。
- D24 保留约 `0.87GiB` 最小余量，是本轮最接近“能给用户推荐”的 16G LoKr 配置。
- D36 说明 allocator fragmentation 是残余风险之一。`expandable_segments` 能让 `blocks_to_swap=23` 在同一 3-step 短跑里通过，但峰值 reserved 仍到约 `14.44GiB`，它是救场环境变量，不是新的默认 preset。
- D37 说明 `mlp_only` 不能单独作为 LoKr 16G 的首选 fallback：第二步 forward 仍在 compiled block 内申请 `34MiB` 时 OOM。
- D38 说明单纯把 `blocks_to_swap` 从 `23` 提到 `24` 也不一定更稳：第二步 MLP `layer1` 的 `66MiB` buffer 仍 OOM，且 reserved/unallocated 更高。
- `wait_ms` 仍然很低，`blocks_to_swap=23` 的 p95 wait 约 `0.68ms`；主要成本是 H2D 本身和更窄的驻留窗口，不是现场等待长尾。

## LoKr grouped projection 速度消融

外部 Kron-LoRA / Kronecker adapter 资料指出，LoKr 可以用 reshape + 小矩阵乘法避免完整 Kronecker materialization。当前实现已采用 no-kron 路径；本轮进一步把输出 factor 从逐片计算改成小分组计算，减少重复的 `x @ w2.T`。

固定配置：

```toml
blocks_to_swap = 23
block_swap_transfer_dtype = "bf16"
selective_checkpoint = "off"
torch_compile = true
attn_mode = "flash"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
max_train_steps = 3 / 10 / 30 / 50 / 300
```

| run | lokr_factor_group_size | status | final_step | step delta avg | steady avg | min forward free | max allocated | max reserved | wait p95 | h2d p95 | 结论 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G1 | 1 | ok | 3 | 22.19s | 20.88s | 0.183GiB | 13.164GiB | 14.518GiB | 0.797ms | 13.821ms | 旧逐 factor 基线 |
| G2 | 2 | ok | 3 | 15.86s | 16.33s | 0.087GiB | 13.307GiB | 14.533GiB | 0.603ms | 13.938ms | 明显提速，显存稳定 |
| G4 | 4 | ok | 3 | 8.97s | 10.63s | 0.105GiB | 13.269GiB | 14.514GiB | 0.630ms | 14.728ms | 稳定性 fallback |
| G8 | 8 | ok | 3 | 5.02s | 4.88s | 0.062GiB | 13.368GiB | 14.488GiB | 0.586ms | 13.570ms | 3-step 速度候选 |
| G8-10step | 8 | ok | 10 | 5.71s | 5.44s | 0.082GiB | 13.368GiB | 14.516GiB | 0.580ms | 13.814ms | 10-step 稳定 |
| G8-30step | 8 | ok | 30 | 5.45s | 5.37s | 0.052GiB | 13.391GiB | 14.553GiB | 0.633ms | 13.829ms | 30-step 稳定 |
| G8-50step | 8 | ok | 50 | 5.47s | 5.42s | 0.048GiB | 13.392GiB | 14.555GiB | 0.593ms | 13.774ms | 50-step 稳定 |
| G8-300step | 8 | ok | 300 | 5.43s | 5.43s | 0.046GiB | 13.396GiB | 14.564GiB | 0.596ms | 13.774ms | 当前推荐默认值，但余量极薄 |

关键观察：

- G2 相对 G1 的 step delta 平均时间约降低 `28.5%`。
- G4 相对 G1 的 step delta 平均时间约降低 `59.6%`，相对 G2 约降低 `43.4%`。
- G8 300-step 通过，steady avg 为 `5.43s`，tail10 avg 为 `5.41s`，相对 G4 的短跑同口径明显更快；max reserved 从 30-step 的 `14.553GiB` 到 300-step 的 `14.564GiB`，没有出现随步数继续膨胀的趋势。
- G8 会一次计算完整 LoKr delta 输出，min forward free 仍只有约 `46MiB`，所以它是当前速度默认值，不是稳定性兜底值。若用户环境仍 OOM，先把 `lokr_factor_group_size` 退回 `4`，再退 `2/1`。

## WebUI 入口

新增 WebUI 资源快捷按钮：

| 按钮 | 作用 |
| --- | --- |
| `LoKr 16G` | 设置 `blocks_to_swap=23`、`block_swap_transfer_dtype=bf16`、`selective_checkpoint=off`、`block_swap_profile_jsonl=auto`、`memory_probe_jsonl=auto`、`memory_probe_max_steps=3`、`lokr_factor_group_size=8`、`torch_compile=true`、`unsloth_offload_checkpointing=false` |

使用建议：

1. LoKr 16G 首次试跑直接点 `LoKr 16G`。
2. 成功跑通几步后，可以把 `memory_probe_jsonl` 改回 `off`，保留 `block_swap_profile_jsonl=auto` 做速度分析。
3. 如果仍 OOM，优先加 allocator 环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256` 后重试；不要先开 Unsloth，也不要改 attention backend。
4. 如果速度默认值 `lokr_factor_group_size=8` 仍 OOM，先退回 `4`；这只牺牲 LoKr grouped projection 吞吐，不改变权重格式。
5. `selective_checkpoint=mlp_only` 和 `blocks_to_swap=24` 本轮短跑都单独失败，不作为首选 fallback。只有在 allocator 仍失败时，才把它们作为进一步诊断变量组合测试。

## 风险和下一轮

本轮结论已经从 3-step 扩展到 10-step、30-step、50-step 和 300-step，但仍不等同于完整训练稳定性承诺。长训练还要继续观察：

- sample/eval 阶段是否安全 pause/resume block swap。
- 不同 bucket 顺序下是否仍有足够余量；G8 300-step 已通过，但 min forward free 已贴到 `46MiB` 左右。
- `blocks_to_swap=24` 单独未通过 3-step 短跑；如果还要继续，应测试 `blocks_to_swap=24 + expandable_segments`，而不是只加 swap。
- `block_swap_transfer_dtype=fp8_e4m3` 是否能降低 H2D 时间，但它会量化 frozen base 权重，不应作为默认训练建议。

## gradient_checkpointing 消融

本轮额外测试了 `gradient_checkpointing=true` 的两种组合，仍固定 `lokr_factor_group_size=8`、`torch_compile=true`、`attn_mode="flash"`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256`。

| run | blocks_to_swap | gradient_checkpointing | status | final_step | avg step | steady avg | max allocated | max reserved | min forward free | 结论 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G8-50step | 23 | false | ok | 50 | 5.47s | 5.42s | 13.392GiB | 14.555GiB | 0.048GiB | 速度最好，余量极薄 |
| GC_no_swap_50 | 0 | true | ok | 50 | 6.99s | 7.00s | 5.330GiB | 5.938GiB | 8.696GiB | 显存最宽，但慢很多 |
| GC_swap_50 | 23 | true | ok | 50 | 7.02s | 7.02s | 2.361GiB | 2.514GiB | 12.049GiB | 显存最省，但速度不优 |

关键观察：

- full `gradient_checkpointing=true` 确实把显存压力大幅压低了；不开交换块时，`max_reserved` 只有约 `5.94GiB`。
- 但 `gradient_checkpointing=true` 的速度代价很大，50-step 平均约 `7.0s/step`，比 G8 无检查点交换块慢约 `28%`。
- `gradient_checkpointing=true + blocks_to_swap=23` 并没有比单独 checkpoint 更快，反而仍在 `7.02s/step` 左右。
- 结论是：全量 checkpoint 适合作为省显存兜底，不适合作为 LoKr 16G 的首选提速方案；当前速度默认仍应保留 `gradient_checkpointing=false` 与 `lokr_factor_group_size=8`。

## selective_checkpoint 同口径复测

2026-06-04 追加复测：由于 `/tmp/anima-lokr-group-ab/` 旧 runtime 已被系统清理，本轮用当前仍存在的 `ichika87_style--61a1-20260602-173252` 缓存重建等价 LoKr G8 runtime。该表只在本节内部横向比较，不与上方 `tag-es1` 旧短跑直接混算。

固定配置：

```toml
blocks_to_swap = 23
block_swap_transfer_dtype = "bf16"
lokr_factor_group_size = 8
gradient_checkpointing = false
torch_compile = true
attn_mode = "flash"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
max_train_steps = 50
```

| run | selective_checkpoint | status | final_step | avg step | steady avg | tail10 avg | max allocated | max reserved | min forward free | wait p95 | h2d p95 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SC_off_50 | off | ok | 50 | 6.35s | 6.23s | 5.88s | 13.392GiB | 14.014GiB | 0.060GiB | 0.902ms | 17.120ms | 同口径 baseline |
| SC_mlp_only_50 | mlp_only | ok | 50 | 5.97s | 5.91s | 5.95s | 13.392GiB | 14.066GiB | 0.047GiB | 0.981ms | 14.847ms | 速度最好，但余量反而更薄 |
| SC_every_other_50 | every_other | ok | 50 | 6.34s | 6.21s | 6.15s | 13.396GiB | 14.006GiB | 0.049GiB | 0.993ms | 17.615ms | 更慢，余量未改善 |

关键观察：

- 现有 `selective_checkpoint=mlp_only/every_other` 都能跑通 50-step，但没有把 min forward free 拉到 `300MiB+`；最低仍只有 `47-49MiB`，甚至比 `off` 组的 `60MiB` 更差。
- `mlp_only` 虽然是三组里最快的，但只是把均速拉回到接近 G8 baseline，没换来更高显存余量；`every_other` 反而更慢。
- 三组 `wait p95` 仍都在 `1ms` 左右，说明这轮差异依旧不是 block swap 等待导致，而是 forward 峰值和 allocator 行为主导。
- 因此当前实现下不建议把 `selective_checkpoint` 纳入默认 LoKr 16G 方案。它可以保留为诊断开关，但真正有价值的下一步应是更细粒度的高峰分支/高峰 block checkpoint，而不是现有粗粒度模式。
- allocator 碎片行为已由 D36/D37/D38 验证为残余主因之一；当前推荐把 `expandable_segments` 作为 LoKr 16G 的第一 fallback。

## peak_probe 峰值定位阶段

2026-06-04 追加实现了独立 `peak_probe_jsonl`，用于 LoKr 16G 后续定点优化。它和 `memory_probe_jsonl` 分离，支持 `peak_probe_level`：

- `block`：只在 DiT block 边界记录，插桩点在 compiled block 外，适合 50-step baseline。
- `ops`：记录 block 内 self-attn / cross-attn / MLP 阶段；会在 compiled block 内引入诊断 graph break，只适合短跑。
- `lokr`：记录 LoKr base output 与 delta apply 前后；同样只适合短跑。
- `full`：全量诊断，当前会明显扰动 16G 极限余量，不适合 50-step。

### 50-step block 级 baseline

输出目录：

```text
/tmp/anima-lokr-peak-probe/
  baseline_g8_50.runtime.toml
  progress.jsonl
  memory_probe.jsonl
  block_swap_profile.jsonl
  peak_probe.jsonl
```

固定配置仍是 LoKr G8 速度默认：

```toml
blocks_to_swap = 23
lokr_factor_group_size = 8
selective_checkpoint = "off"
gradient_checkpointing = false
torch_compile = true
attn_mode = "flash"
peak_probe_level = "block"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
max_train_steps = 50
```

| run | status | final_step | avg step | steady avg | tail10 avg | max allocated | max reserved | min before-forward free | wait p95 | h2d p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_g8_50 + block peak | ok | 50 | 6.28s | 6.21s | 6.11s | 13.384GiB | 14.006GiB | 0.025GiB | 0.847ms | 15.357ms |

block 边界 top low-free 事件集中在 step 17，token shape 为 `[1, 1, 4032, 1, 2048]`：

| rank | label | block_idx | free | allocated | reserved | 说明 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | block_after | 11 | 0.023GiB | 6.124GiB | 13.832GiB | block 11 后 |
| 2 | block_before | 12 | 0.023GiB | 6.124GiB | 13.832GiB | 同一低余量传到下个 block |
| 3 | block_after | 10 | 0.023GiB | 5.716GiB | 13.832GiB | block 10 后 |
| 4 | block_before | 11 | 0.023GiB | 5.716GiB | 13.832GiB | 同一低余量传到下个 block |
| 5 | block_after | 7 | 0.024GiB | 4.495GiB | 13.832GiB | block 7 后 |

解释：

- 这组 50-step 跑通，但最低 `cuda_free_gb` 只有约 `25MiB`，比旧 G8 300-step 的 `46MiB` 还薄。
- 低点不是 H2D wait：`wait p95=0.847ms`。
- block 边界的 allocated 只有 `~6GiB`，但 reserved 已到 `13.832GiB`，说明低 free 很大程度来自 allocator/reserved 与非 PyTorch 常驻占用；单看 block 边界无法捕获 compiled block 内部瞬时高峰。

### 短跑 op / LoKr 定位

由于在 compiled block 内记录 `ops/full` 会改变图分段并放大峰值，`ops_probe_3` 在 step0 OOM；这本身验证了内部余量极薄。失败点是 `blocks.27.mlp.layer1` 的 LoKr delta apply 附近：

```text
ops_probe_3: step0 forward OOM
CUDA tried to allocate 132MiB
stack: Block._forward -> self.mlp -> GPT2FeedForward.layer1 -> LoKrModule.forward
```

`ops_probe_3.peak_probe.jsonl` 在 OOM 前的最低事件：

| label | block_idx | phase | free | allocated | reserved |
| --- | ---: | --- | ---: | ---: | ---: |
| cross_attn_after_projection | 27 | cross_attn | 0.349GiB | 13.557GiB | 13.688GiB |
| mlp_before_norm | 27 | mlp | 0.349GiB | 13.557GiB | 13.688GiB |
| mlp_after_projection | 26 | mlp | 0.449GiB | 13.277GiB | 13.588GiB |

`lokr_probe_1` 用 `peak_probe_level="lokr"` 单步通过，最低 LoKr 事件进一步确认末端 block 的 MLP LoKr 是最危险路径：

| label | op | block_idx | free | allocated | reserved |
| --- | --- | ---: | ---: | ---: | ---: |
| lokr_after_delta_apply | mlp_layer2 | 27 | 0.787GiB | 13.012GiB | 13.258GiB |
| lokr_before_delta_apply | mlp_layer2 | 27 | 0.787GiB | 12.980GiB | 13.258GiB |
| lokr_after_delta_apply | mlp_layer1 | 27 | 0.787GiB | 13.028GiB | 13.258GiB |
| lokr_after_base | mlp_layer1 | 27 | 1.100GiB | 12.900GiB | 12.945GiB |

阶段结论：

- 50-step baseline 说明当前 G8 默认仍是“能跑但极薄”，最低 free 只有 `~25MiB`，没有达到 `300MiB` 目标。
- 细粒度短跑把风险收敛到后段 DiT block，尤其是 `block 27` 的 `MLP layer1/layer2` LoKr delta apply。
- 现有粗粒度 `mlp_only/every_other` 已证明不能有效抬高余量；因此后续实现了更细粒度候选：`mlp_layer1_only`、`peak_blocks_mlp_layer1`、`peak_blocks_mlp`。结果见下一节。
- `ops/full` 探针不能作为 50-step 常规 profile 模式；后续 50-step 消融应使用 `peak_probe_level="block"`，短跑才开 `ops/lokr`。

## 定点 checkpoint 与 fused LoKr delta 消融

2026-06-04 继续推进后实现并测试了三类更细粒度候选：

- `selective_checkpoint = "mlp_layer1_only"`
- `selective_checkpoint = "peak_blocks_mlp_layer1"`，分别覆盖 `25-27` 与 `24-27`
- `selective_checkpoint = "peak_blocks_mlp"`，覆盖 `25-27`

随后又实现了 fused LoKr delta apply：custom autograd forward 不再返回完整 `N × group × out_dim` fp32 delta 临时，而是按 row chunk 直接把 delta 写入 frozen Linear 的 base output；backward 从保存的 `x/w1/w2` 重算梯度。该实现保持权重格式不变，不 materialize `torch.kron`，也不量化 trainable adapter。

固定配置：

```toml
blocks_to_swap = 23
block_swap_transfer_dtype = "bf16"
lokr_factor_group_size = 8
gradient_checkpointing = false
unsloth_offload_checkpointing = false
torch_compile = true
attn_mode = "flash"
peak_probe_level = "block"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
max_train_steps = 50
```

输出目录：

```text
/tmp/anima-lokr-peak-probe/
  B0_baseline_g8_current_50.*
  C1_mlp_layer1_only_50.*
  C2_peak_25_27_layer1_50.*
  C3_peak_24_27_layer1_50.*
  C4_peak_25_27_mlp_50.*
  F0_fused_lokr_add_g8_50.*
  F1_fused_lokr_add_g8_alloc64_gc_50.*
```

| run | selective_checkpoint | blocks | status | final_step | steady avg | tail10 avg | max allocated | max reserved | min before free | min peak free | wait p95 | h2d p95 | 结论 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0_baseline_g8_current_50 | off | - | ok | 50 | 5.470s | 5.449s | 13.392GiB | 13.891GiB | 65MiB | 65MiB | 0.749ms | 13.953ms | 当前同口径 baseline |
| C1_mlp_layer1_only_50 | mlp_layer1_only | - | ok | 50 | 5.620s | 5.677s | 13.388GiB | 13.904GiB | 73MiB | 73MiB | 0.753ms | 14.570ms | 只增加约 8MiB，未达标 |
| C2_peak_25_27_layer1_50 | peak_blocks_mlp_layer1 | 25-27 | ok | 50 | 5.618s | 5.692s | 13.388GiB | 13.910GiB | 69MiB | 69MiB | 0.798ms | 14.570ms | 未达标 |
| C3_peak_24_27_layer1_50 | peak_blocks_mlp_layer1 | 24-27 | ok | 50 | 5.486s | 5.430s | 13.395GiB | 13.910GiB | 64MiB | 64MiB | 0.707ms | 14.324ms | 未达标 |
| C4_peak_25_27_mlp_50 | peak_blocks_mlp | 25-27 | ok | 50 | 5.713s | 5.820s | 13.396GiB | 13.908GiB | 76MiB | 76MiB | 0.809ms | 14.092ms | 本组最高 free 也只有 76MiB |
| F0_fused_lokr_add_g8_50 | off | - | ok | 50 | 6.040s | 5.971s | 13.251GiB | 13.910GiB | 66MiB | 66MiB | 0.832ms | 13.934ms | `max allocated` 降约 145MiB，但 free 仍未抬高 |
| F1_fused_lokr_add_g8_alloc64_gc_50 | off | - | ok | 50 | 5.903s | 5.875s | 13.247GiB | 13.922GiB | 66MiB | 66MiB | 0.726ms | 13.873ms | allocator GC 会阶段性释放 cache，但峰值 free 仍未达标 |

关键观察：

- 定点 checkpoint 没有解决目标问题。`mlp_layer1_only`、`peak_blocks_mlp_layer1`、`peak_blocks_mlp` 全部 50-step 通过，但最低 free 仍只有 `64-76MiB`，离 `300MiB` 门槛很远。
- fused LoKr delta apply 确实降低了 PyTorch `max allocated`：从 baseline `13.392GiB` 降到 `13.247-13.251GiB`，说明完整 fp32 delta group 临时已经被削掉一部分。
- 但 fused 后 `max reserved` 仍在 `13.91GiB` 左右，NVML free 仍只有 `~66MiB`。也就是说当前最小 free 主要被 allocator reserved / CUDA context / 桌面常驻显存共同决定，而不是单个 MLP checkpoint 可以释放的 saved activation。
- `wait p95` 仍低于 `1ms`，H2D 约 `13.9-14.6ms`，再次证明 block swap 等待不是当前瓶颈。
- F1 的 `max_split_size_mb=64,garbage_collection_threshold=0.8` 能在部分 step 后把 `memory_reserved_gb` 降到 `4-5GiB`，但 forward 峰值仍会重新顶到 `13.9GiB+`，没有把最低 free 提到 300MiB。

阶段结论：

- 新增细粒度 checkpoint 模式可以保留为诊断/手动开关，但不推荐作为 LoKr 16G 默认。
- fused LoKr delta apply 是正确方向，能降低 `max allocated` 且速度损失约 `7.5%~10.9%`，符合速度门槛；但它单独不足以达成 `min free >= 300MiB`。
- 下一轮不应继续堆 checkpoint；应围绕“降低 allocator 峰值 reserved / 减少 CUDA context 外常驻 / 更彻底地 fused base+delta Linear 输出”继续优化。

## 验证记录

已通过：

```bash
timeout 60 .venv/bin/python -m py_compile train.py networks/plugins/lokr/autograd.py networks/plugins/lokr/module.py tests/test_lokr.py
timeout 60 .venv/bin/python -m pytest tests/test_lokr.py -q
timeout 60 .venv/bin/python -m pytest tests/test_network_registry.py -k lokr -q
timeout 60 .venv/bin/python -m pytest tests/test_block_swapping.py -q
timeout 60 .venv/bin/python -m pytest tests/test_config.py -q
timeout 60 .venv/bin/python -m pytest tests/test_training_resume.py -k "block_swap or progress_jsonl" -q
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_state.py -k "block_swap or resource" -q
```

本轮新增验证：

```bash
timeout 60 .venv/bin/python -m py_compile library/runtime/peak_probe.py library/anima/models.py networks/plugins/lokr/module.py library/training/loop.py train.py library/training/cli_args.py tests/test_lokr.py tests/test_block_swapping.py tests/test_config.py
timeout 60 .venv/bin/python -m pytest tests/test_lokr.py -q
timeout 60 .venv/bin/python -m pytest tests/test_block_swapping.py -q
timeout 60 .venv/bin/python -m pytest tests/test_config.py -q
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_state.py -k "block_swap or resource or selective_checkpoint" -q
```

短实验通过：

```text
D23: blocks_to_swap=22, max_train_steps=3, status=ok, 但最小 forward free 约 0.09GiB
D24: blocks_to_swap=23, max_train_steps=3, status=ok, 最小 forward free 约 0.87GiB
D36: blocks_to_swap=23 + expandable_segments/max_split_size_mb=256, max_train_steps=3, status=ok
D37: blocks_to_swap=23 + selective_checkpoint=mlp_only, status=error, step1 forward OOM
D38: blocks_to_swap=24, status=error, step1 MLP layer1 OOM
B0/C1/C2/C3/C4: 定点 selective checkpoint 50-step 全部 status=ok，但 min peak free 仅 64-76MiB
F0/F1: fused LoKr delta apply 50-step status=ok，max allocated 降到约 13.25GiB，但 min peak free 仍约 66MiB
```

## 2026-06-05 allocator / blocks / FP8 追加消融

本轮在用户授权后继续沿 `goal` 推进，固定约束不变：

```toml
use_lokr = true
use_custom_down_autograd = true
lokr_factor_group_size = 8
selective_checkpoint = "off"
gradient_checkpointing = false
unsloth_offload_checkpointing = false
torch_compile = true
attn_mode = "flash"
peak_probe_level = "block"
max_train_steps = 50
```

输出目录：

```text
/tmp/anima-lokr-allocator-ablation/
/tmp/anima-lokr-blocks-ablation/
```

### allocator 消融

| run | status | final | avg step | steady avg | tail10 avg | min peak free | max allocated | max reserved | wait p95 | h2d p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alloc256_baseline | ok | 50 | 5.847s | 5.816s | 5.790s | 67.2MiB | 13.251GiB | 13.986GiB | 0.718ms | 14.573ms |
| alloc128 | ok | 50 | 6.006s | 5.927s | 5.830s | 53.9MiB | 13.247GiB | 14.145GiB | 0.728ms | 14.664ms |
| alloc64_gc08 | ok | 50 | 5.954s | 5.893s | 6.018s | 78.5MiB | 13.239GiB | 13.912GiB | 0.736ms | 14.640ms |
| alloc32_gc08 | ok | 50 | 5.856s | 5.809s | 5.981s | 70.9MiB | 13.247GiB | 13.910GiB | 0.806ms | 14.453ms |

结论：allocator 调参能改变 `reserved` 形态，但不能把最低 free 从 `~50-80MiB` 拉到目标 `300MiB+`。`max_split_size_mb=64 + garbage_collection_threshold=0.8` 是本组最低 `max_reserved` / 最高最低 free 的候选，但收益只有约十几 MiB，不能作为根本修复。

### blocks_to_swap=24 复测

在 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8` 下，把 `blocks_to_swap` 从 23 提到 24：

| run | status | final | avg step | steady avg | tail10 avg | min peak free | max allocated | max reserved | wait p95 | h2d p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bs24_alloc64_gc08 | ok | 50 | 5.831s | 5.775s | 5.747s | 78.7MiB | 13.118GiB | 13.898GiB | 0.737ms | 14.507ms |

结论：多换 1 个 block 会降低 `max allocated`，但最低 free 仍只有约 `79MiB`，并没有形成 300MiB 级别余量。当前不建议继续盲目把 LoKr 16G 默认推到更高 `blocks_to_swap`；收益已被 allocator/reserved 与 compiled forward 峰值掩盖。

### FP8 transfer 消融

同样固定 `blocks_to_swap=24 + alloc64_gc08`，只把 frozen block swap transfer 从 bf16 改为 `fp8_e4m3`：

| run | transfer | status | final | avg step | steady avg | tail10 avg | min peak free | max allocated | max reserved | wait p95 | h2d p95 | CPU master |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bs24_alloc64_gc08 | bf16 | ok | 50 | 5.831s | 5.775s | 5.747s | 78.7MiB | 13.118GiB | 13.898GiB | 0.737ms | 14.507ms | 3.61GiB |
| bs24_alloc64_gc08_fp8 | fp8_e4m3 | ok | 50 | 5.935s | 5.868s | 5.830s | 82.0MiB | 13.118GiB | 13.906GiB | 0.637ms | 11.102ms | 1.80GiB |

FP8 量化统计：`fp8_relative_l2_by_block` 最大约 `0.083`，未见 saturated tensor。

结论：FP8 transfer 有工程价值：CPU pinned master 减半，H2D p95 从约 `14.5ms` 降到约 `11.1ms`，wait p95 也略降。但它不降低 GPU 上 active block 的 bf16 执行权重，也不解决最低 free；50-step 最低 free 仍只有约 `82MiB`。因此 FP8 transfer 可以作为速度/PCIe 优化开关，不应被宣传为 LoKr 16G OOM 根因修复。

## 2026-06-05 next-goal 收口

原 `anima_lokr_16g_next_goal.md`（已归档到 `_archive/docs/findings/`）的活跃目标已经由上述 allocator / blocks / FP8 追加消融收口。该目标的约束保持不变：

- `torch_compile = true`
- 不修改 attention backend，继续使用 `attn_mode = "flash"`
- 不把 full `gradient_checkpointing` 作为默认方案
- trainable LoKr / router / adapter 常驻 GPU
- block swap 只处理 frozen base block

目标曾设为把最低 forward free 从当前几十 MiB 拉到 `>= 300MiB`，同时把速度损失控制在 `<= 10%~12%`。追加消融结论是：现有安全路线已经接近极限，allocator、`blocks_to_swap=24` 和 FP8 transfer 都不能把最低 free 拉到目标区间。

当前推荐用户配置保持：

```toml
blocks_to_swap = 23
block_swap_transfer_dtype = "bf16"
selective_checkpoint = "off"
gradient_checkpointing = false
unsloth_offload_checkpointing = false
torch_compile = true
attn_mode = "flash"
lokr_factor_group_size = 8
block_swap_profile_jsonl = "auto"
memory_probe_jsonl = "auto"
memory_probe_max_steps = 3
```

首选 fallback：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
```

暂停方向：

- 盲目提高 `blocks_to_swap`
- 把 full checkpoint 设为默认
- 关闭 `torch_compile`
- 更换 attention backend
- 继续扩大粗粒度 `selective_checkpoint`
- 未经 profiling 直接写 Triton 全量重构

若强制要求 `300MiB+` 余量，下一阶段必须转向更深层 kernel/graph/autograd 生命周期重构，或接受 full checkpoint / 降 token 峰值的代价。

## 当前最终结论

在当前约束下，已经验证过以下路线：

- no-kron LoKr forward
- grouped LoKr projection
- fused LoKr delta apply
- `blocks_to_swap=23/24`
- allocator `max_split_size_mb=256/128/64/32` 与 GC 阈值
- coarse / fine selective checkpoint
- bf16 vs fp8 block-swap transfer

这些路线能把 LoKr 16G 从“直接 OOM”推进到 50/300-step 可跑，但**仍无法稳定达到 `min forward free >= 300MiB`**。当前最佳短跑最低 free 仍在 `~80MiB` 量级。

因此后续若必须达到 300MiB+ 余量，只剩三类方向：

1. 接受更大数学/速度代价：full `gradient_checkpointing=true` 或降低 token/resolution/batch 峰值。
2. 做更深层 kernel/graph 级重构：融合 base Linear backward 与 LoKr delta backward，减少 compiled graph / autograd saved tensor 生命周期，而不是继续调 block swap。
3. 降低外部常驻显存与 allocator 压力：独占 GPU、关闭桌面/浏览器占用、保持 `expandable_segments`，但这只能救场，不能保证 300MiB+。

面向用户的稳定建议保持不变：默认仍用 `LoKr 16G` 速度方案；如果仍 OOM，优先用 allocator fallback 和降低 `lokr_factor_group_size`，最后才考虑 full checkpoint。FP8 transfer 可手动开启用于 PCIe/H2D 消融，不作为显存余量修复。
