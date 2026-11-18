# Qwen Image 2.1 编辑 LoRA 1024 双图采样性能实测

状态：研究记录 / 阶段快照
日期：2026-10-03。结果：正常完成，进程退出 0，输出两张 1024×1024 PNG，latent finite。

使用此前训练的 step400 LoRA（严格加载全部键通过），CMP 90HX 10 GiB，BF16、torch attention、swap24/slab、32块 Inductor dynamic compile、方案 B，关闭 DiT K/V 缓存。两条独立单参考图编辑分别生成两张结果，不是同一次生成输入两个参考图。Euler28、CFG1、seed42/43，保持此前 heavy-paint 提示词。a-2 原图是输入，b-2 仅在验收对照图中展示，没有输入采样。

## 耗时

| 指标 | 本轮实测 |
| --- | ---: |
| 第1图去噪 | 130.24 秒 |
| 第2图去噪 | 128.82 秒 |
| 两图去噪合计 | 259.07 秒 |
| 平均去噪每步 | 4.626 秒 |
| 正式批次（含块交换进入/恢复、latent保存） | 266.81 秒 |
| 块交换进入 / 恢复 | 3.88 / 3.85 秒 |
| 文本/参考条件准备 | 35.34 秒 |
| 模型加载与编译接线 | 30.27 秒 |
| 双图各2步预热批次 | 40.50 秒 |
| 第1 / 第2图解码 | 0.70 / 0.67 秒 |
| VAE加载、双图解码和PNG保存 | 3.11 秒 |
| 完整进程墙钟（含准备、预热、全部保存） | 386.42 秒 |

每图去噪前后 CUDA synchronize；不含 VAE 解码和PNG保存。平均每步由整张去噪耗时除28计算，不是逐步时间分布。模型加载项只包括 compile 接线，首次执行编译包含在预热。条件缓存复用旧实验 helper，还缓存一条正式不用的CFG条件（总3条），故条件准备不等于优化后的最短启动成本。

正式采样前后 Dynamo counters 完全相同（unique_graphs=1），本轮未观察到正式阶段新增编译。正式批次 prepare 计数进入1、恢复1，去噪内0。

## 显存与硬件遥测

| 阶段 | PyTorch allocated峰值 | PyTorch reserved峰值 | NVML显存观测峰值 |
| --- | ---: | ---: | ---: |
| 预热 | 4.61 GiB | 8.23 GiB | 8718 MiB |
| 正式B | 4.29 GiB | 4.67 GiB | 5072 MiB |
| 解码 | 7.20 GiB | 8.67 GiB | 9164 MiB |

正式B阶段每约1秒采集，共267点（包含进入/恢复）：GPU利用率均值99.89%，功耗均值186.75W、峰值237.64W；温度46–55°C；SM频率1530–1830MHz。遥测无错误。NVML是采样观测峰值，可能漏掉瞬时峰值，不能与allocator指标等同。解码仅3个遥测点，特别不宜视为精确NVML峰值。

显存的最高阶段是VAE解码，不是正式DiT去噪；本轮解码前已释放DiT/adapter。测试结束GPU计算进程已退出，没有修改原训练配置、checkpoint或重启WebUI。

## 样图与边界

视觉核验：两图均保留参考图主要人物和场景结构，有细节/质感变化；宽高比由非方形改为方形，构图存在变化。未做无LoRA同seed对照，不能据此量化LoRA净贡献或质量提升。

这是一个设备、一个checkpoint、两条样本的单轮测试，不是长期稳定吞吐或跨分辨率严格性能A/B。不能将此前低分辨率耗时直接作为1024倍率基准（参考图形状、缓存与硬件温频也会影响结果）。

- 样图见上述输出目录的 B_image0.png / B_image1.png
- 原图/训练目标/采样结果对照：reference_target_result.jpg
- 原始指标：summary.json、events.jsonl、gpu_telemetry.csv、metrics.json
- checkpoint SHA256：`9ea1d6118403274aa762e1e64d1cfc3913688225d455e6bdedb69c41ce4a3def`

复现脚本：`scripts/qwen_image_2_1/sampling_benchmark.py`，具体输入及source hashes保存在summary.json。输出目录：`output/tests/qwen-edit-1024-dual-20261003`。
