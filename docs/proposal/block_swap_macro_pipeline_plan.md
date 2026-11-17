# Block Swap 宏观流水线开发计划（P0'–P3）

状态：半活跃 / P0'–P2 四族已收口（`--block_swap_auto` 已落地；仅剩 NF8 选择性接入与 P3 后续子项待定）
适用版本：当前 `dev`（原开发分支 `codex/block-swap-int8` 已合入）
相关代码：`library/runtime/offloading.py`、`library/runtime/block_swap_config.py`、`library/runtime/block_swap_masters.py`、`bench/block_swap_pipeline/`
证据基线：[Block Swap 宏观流水线基线（CMP 50HX）](../findings/block_swap_pipeline_baseline_50hx_20261019.md)

## 背景

单张量恢复（fused 单算子 + 环形缓冲 + 双流）已接近微观极限，因此本轮起把视角切到
**宏观调度**（谁在什么时候搬运、被掩盖了多少）与 **1 Byte 编码的信噪比**。
原始输入给的是四个方向（U-Turn Pinning / 交错换出 / 离群裁剪·NF8 / auto swap 水位）。
第 1 轮真机基线对其中的 P0 前提做了实测修正，见下。

## 第 1 轮实测修正（重要）

```
swap=0  稳态步时 ≈ 1.01~1.09 s
swap=12 稳态步时 ≈ 1.48~1.57 s        → swap 代价 ≈ +0.45~0.49 s/步
profiler: 24 任务/步（=2S）、重复恢复 0 次、exposed_wait ≈ 1.6 ms/步
```

- **原 P0"折返点重复搬运"不存在**：训练态前向只在 `block_idx < blocks_to_swap` 提交交换，最后一个换出块
  前向用完仍驻留，反向第一时间直接开算；反向早块恢复也已由尾块钩子形成 lead≈1 的流水。
- 真实机制是**串行 DMA 链**：12 次 H2D 合计 ≈ 1.07 s/步，与 swap=0 的计算（≈1.05 s）同量级；
  单块有效带宽仅 ≈1.3~2.3 GB/s，单线程拷贝池把它们排成一条链，于是表现为步时被 DMA 顶住，
  而不是在等待点暴露停顿。
- 结论：调度类优化（交错、预取、折返点驻留）头部空间有限；**减少搬运字节**与**提高单块搬运效率**才是杠杆。

## 阶段与状态

| 阶段 | 内容 | 状态 | 验收口径 |
| --- | --- | --- | --- |
| **P0'** | 压缩串行 DMA 链：量化单块 H2D 有效带宽，对比 `restore_mode`（foreach / slab） | **已完成（结论：链路饱和）** | 见[第 2 轮证据](../findings/block_swap_restore_link_limit_50hx_20261019.md)：链路天花板 3.14 GB/s，训练内已达 3.02（foreach）/ 3.12（slab）GB/s；建议显式设 `block_swap_restore_mode="slab"`（DMA +3.3%、步时 −4%） |
| **P1** | 进程内水位调优 `--block_swap_auto`（+ 可选停止条件 `--block_swap_auto_stop_ratio`） | **已实现 + 四族真机验收通过（Anima 512/256/1024、Krea-2-Raw、Z-Image、Qwen-Image 2.1），本阶段收口** | 见[第 3/7 轮证据](../findings/block_swap_auto_tuner_50hx_20261019.md)：默认 512 下 24 → 2（字节 6644 → 554 MB，窗口 2115 → ~720 ms，步时 0.92 s）；`stop_ratio=0.5` 在 256 下 24 → 13（字节 → 3599 MB，窗口 → ~1140 ms，步时 1.34 s，省约 1.45 GB 显存）；决策需实测步时故延后一个优化器边界 |
| **P2** | 1 Byte 高信噪比编码：3A（离群裁剪）与 3B（NF8 查表）；并归因 loss 偏移 | **已完成（结论：3A 否定；3B 部分有效；1 Byte 下 <0.2% 不可达）** | 见[第 4~6 轮证据](../findings/int8_encoding_floor_50hx_20261019.md)：int8 权重误差固有 0.4%~2%（相对 L2），3A 上限 ≤2.01%（平方口径）；固定 NF8 真实行 −13.7%~64.3%，高斯行回归 70%~477% → 按行择优；toy 端到端 loss 偏移 0.108%（gate_pass），把 1.09% 压到 <0.2% 需约 5.5× 误差下降，1 Byte 不可达 → int8 定为"省一半流量 + 显式记录偏移"的选项，NF8 待选择性接入 |
| **P3** | 交错换出拓扑 / 跨步预取 | 降级（收益受限） | 仅在 P0'/P1 完成后复查：链路已饱和（96~99%）、`exposed_wait` 实测 1.6~7 ms/步；调度重排不改变总字节数。**第 12 轮补充**：P1 把每步搬运量直接腰斩才换来步时腰斩，说明真正的杠杆是「少搬」而不是「换顺序」 |

## 每轮工作约定

1. 每轮先做**真机测量**，再改代码；改动必须能在同一工作区 A/B（`bench/block_swap_pipeline` 支持持久工作区）。
2. 每轮结束把结果写进 `docs/findings/`（含复现命令、边界与未证明项），并更新本文件的阶段状态。
3. 文档写完后核对：数字与 `result.json` / 训练日志一致、链接可达、`git diff --check` 干净。
4. 只在本分支（`codex/block-swap-int8`）开发；证据与实现分开提交，便于回滚单轮结论。

## 推进记录

| 轮次 | 日期 | 内容 | 结果 |
| --- | --- | --- | --- |
| 1 | 2026-10-19 | 建立 `bench/block_swap_pipeline` 探针（工作区复用 + 相位切分 + 指标口径），跑 swap=12 / swap=0 A/B | P0 前提被实测否定并修正为 P0'；swap 代价 ≈ +0.45~0.49 s/步；重复搬运 0；exposed_wait ≈ 1.6 ms |
| 2 | 2026-10-19 | 纯拷贝微基准（`copy_bench.py`）+ profiler 加 `h2d_bytes` 字节计量 + `foreach` vs `slab` A/B | 链路天花板 3.14 GB/s；训练内 3.02 / 3.12 GB/s = 96% / 99%；分块粒度无关；P0' 完成并推荐 slab；下一步 P1 |
| 3 | 2026-10-19 | 实现 `library/runtime/block_swap_auto.py` + CLI（`--block_swap_auto` / `--block_swap_vram_margin_mib`）+ 优化器边界挂点 + 10 项单测；真机跑 auto（24 起点）与固定 swap=5 对照 | 512 下自动 24 → 2（bytes 6644 → 554 MB/步，窗口 2118 → ~720 ms）；但 swap≤5 后与固定 5 持平 → 需加停止条件；256/1024 与其它模型族待验收 |
| 7 | 2026-10-19 | 给 `--block_swap_auto` 加可选停止条件（`--block_swap_auto_stop_ratio` / env），修掉链路测量 device 传参 bug，把决策改为"延后一个边界拿实测步时"；真机跑 512（默认模式）与 256（stop 0.5） | 512 默认 24 → 2（窗口 2115 → ~720 ms，步时 0.92 s）；256+stop0.5 按设计停在 24 → 13（窗口 → ~1140 ms，步时 1.34 s，省 ~1.45 GB 显存）；实测换出载荷与分辨率无关（256/512 在 swap=24 都是 6644 MB/步） |
| 8 | 2026-10-19 | 1024 档真机验收（先触达 Anima 换出上界：`num_blocks=28` → 合法 `1..26`，swap=30 被断言拒绝）；核对跨族 resolver 属性名；修正文档中已过期的 `restore_mode` 默认值表述（上游 `e8ad25b3` 已改为 `slab`） | 1024 默认模式 24 → 2（DMA 2291/2133 → 178/180 ms，步时 3.88/3.94 → 3.42/3.19 s）；绑定约束是契约（26 上限）而非显存；1024 由算力主导，DMA 占比降至约 5% |
| 9 | 2026-10-19 | 新增 `bench/block_swap_pipeline/family_bytes.py`（只读 safetensors 头部算各族分块载荷，含 `plan_promotion` 同口径算术）；核对跨族缓存入口 | 四族均为均匀块：anima 132.0 / z_image 345.1 / krea2_raw 828.1 / qwen_image_2_1 **416.0** MiB，上界 26/28/26/30（qwen 行在第 12 轮由 648.2/58 更正：旧值取自会被 `_validate_header` 拒绝的 60 块 fp8 架构）；Anima 档与 profiler 138.4 MB 逐字一致（工具先在 `llm_adapter.blocks` 同号处算错 32 MiB 后被修正）；非 Anima 族省一块值 0.23~0.55 s/步；跨族真机缺口在探针（`tasks.py preprocess-te` 已按 `model_family` 分派五族，仅 `scripts/preprocess/cache_text_embeddings.py` 为 Anima 专用），下一步给 `run_probe.py` 加族预设 |
| 10 | 2026-10-19 | Krea-2-Raw 跨族真机验收（bf16 全量、Turing 下只能用 `attn_mode=torch`）；打通真实数据集的缓存重建与自包含工作区 | 日志 `22.64 GiB across 28 blocks` 与头部推算相差 0.011%；每步 52 任务、45147 MB（预测 45153）、3.114 GB/s（天花板 99.2%）、DMA 占步时 ~96%；无余量时正确 `keeping blocks_to_swap=26`，且单块 828 MiB > 空闲 686 MiB → 该卡上 Krea-2 无法晋升；896×1200 原生分辨率在 torch 注意力下 OOM，512 档通过 |
| 12 | 2026-10-19 | Qwen-Image 2.1 第四族真机验收（tokenizer 就地铺设、`timestep_sampling` 漏配修正、改用本族真实 32 块权重）；修正第 9 轮 qwen 载荷数字并重判 `duplicate_restores` | 固定 `swap=30`：60 任务/步、24.38 GiB、3.130 GB/s；自动晋升 `30 → 14`（free=7.00 GiB、margin=0.39 GiB、block=416.0 MiB）→ 28 任务/步、11.38 GiB（−53.3%），步跨度 7848 → 3877 ms（−50.6%）；每块 416.0 MiB 与头部推算逐字节一致，**P1 四族收口** |
| 11 | 2026-10-19 | 给探针加 `--family` 族预设（各族源路径/缓存后缀/TE 缓存入口/运行时键；目录型组件原地引用；非 Anima 族图文同目录），并在 Z-Image 真机验收自动水位 | 日志 `10.11 GiB across 30 blocks` 与头部推算一致；固定 swap=28：56 任务/步、20260 MB（预测 20262）、3.117 GB/s、DMA 占步时 ~93%；**非 Anima 族正向晋升首次实测** `28 → 13`（15 块），步时 7.69/8.12 → 5.30/5.38 s（−30%），每步少搬 10853 MB，一次性晋升 ~1.6 s 一步回本 |
| 6 | 2026-10-19 | 用现成 `int8_blockswap_equivalence_probe.py` 在 50HX 跑 bf16 vs int8 端到端对照（6 块/换 3/3 步），并逐模块核对编码链 | `gate_pass=true`；loss 偏移 0.108%、输出 0.62%、cosine 0.999982、int8 master 字节 56.6%；权重相对 L2 0.37% 与理论 amax/(2·127)=9.84e-4 吻合 → **归因完成：1 Byte 下无法把 loss 偏移压到 <0.2%** |
| 4~5 | 2026-10-19 | 新增 `library/runtime/int8_encode.py`（裁剪搜索 + NF8 码本）与 `bench/block_swap_int8/encode_error.py` 探针；对 4 组合成分布 + 3 个真实 Anima 张量量化 4 种编码的误差上限；15 项单测 | **3A 否定**（上限 ≤2.01%，已贴噪声地板）；**3B 分布依赖**（真实行 +13.7~64.3%，高斯行回归 70~477%）→ 须选择性接入；1.09% loss 偏移仍未归因 |
