# Anima 异型 NF4/BF16 换层修复与短程显存追踪

状态：换层兼容修复通过；训练阶段显存略降，含采样的全程峰值没有下降。

CMP 90HX 10GB；既有60张数据的独立1024×1024副本/cache；plain LoRA rank/alpha32、BF16 compute、swap12、full checkpoint、Flash、无compile。
BF16、旧混合换层、最终优化版各8步，在第4/8步使用同一caption/seed采样一张1024图（Euler8步、CFG4）。原500步配置和输出未覆盖。

## 实测 allocated（GiB）

| 指标 | BF16 | 混合修复前 | 混合修复后 |
|---|---:|---:|---:|
| 普通第2步结束 | 2.8464 | 2.3117 | 2.2727 |
| 采样前 forward 峰值 | 3.6458 | 3.7456 | 3.7065 |
| 采样前 backward 峰值 | 4.0186 | 4.0621 | 4.0231 |
| 全部8步 backward 阶段峰值 | 4.0186 | 4.2965 | 4.2184 |
| 含采样的进程峰值 | 4.6998 | 4.9341 | 5.0513 |

采样前反向峰值与普通步末各降40MiB；采样后训练阶段峰值降80MiB。含采样进程峰值却增加120MiB，不能宣称端到端显存改善。
逐阶段探针只覆盖forward/backward/optimizer，采样内部的扩散、VAE加载/解码没有独立细分，因此目前只能将新全程高水位约束到采样边界。
诊断在边界同步并重置raw峰值；公共max API维护累计值，日志口径保持一致。这些运行不用于吞吐比较。

## 修复内容

- 按逻辑矩阵shape验证NF4/原生dense交换，拒绝损坏或不匹配表示；消除双端留GPU fallback。
- 训练混合槽提前分配稳定uint8 backing，dense/NF4使用不同view；轮转owner并原地恢复，保留Parameter/quant_state及bnb原forward保存的packed alias。
- 当前5个混合槽各预留32MiB，替代双端8+32MiB；净驻留收益约40MiB，而不是160MiB。最初循环就绪时会比旧版多预留120MiB，不能只看一个时点。
- 采样复用训练缓冲池，避免每个推理槽预留dense最大容量，或释放重分配时暂留旧packed引用。
- dynamic reconfigure将新槽分配和计划校验纳入事务，失败恢复旧拓扑/hooks；兼容稀疏CPU masters；纯dense prepare保留计划缓存identity。
- 计划计数包含预热查询，不是真实执行次数；最终fallback查询分类为0，旧版为324。

## 数值证据与限制

CPU/CUDA回归覆盖swap12/20、多跳、foreach/slab、non-reentrant checkpoint开关、多步训练→采样→训练；小夹具output/input_grad/全部adapter_grad逐位相同。
真实独立8步最终LoRA全部finite，但初始化未证明一致，不能作为逐位等价证据；该差异记录在numerical-verification.json。
补充真实两步受控测试：加载同一份LoRA，在loop_ready重置seed；初始网络SHA256一致 `e61092f16d65e50b0ad6f76a4a799a1d5f2c3f98003f9bd0990d5f90ccf9f4e2`。
两步loss逐位相同：[0.22462502121925354, 0.1566736102104187]。保存LoRA整体relative-L2=5.5979335e-06，max_abs=3.2246113e-05；301/840个tensor逐位相同。
存在微小数值漂移，不能宣称真实训练梯度或最终权重逐位等价，也没有长期质量验收。受控测试不证明所有adapter或其他模型族的混合精度支持。

## 验收与证据

- 最终CPU/CUDA定向回归152项通过（新增hash探针测试另行通过）；独立CPU验收55通过24跳过；目标typecheck无错误；新增文件Ruff通过。
- 三个8步对照exit0，最终优化版采样后继续训练与保存成功；两个2步受控对照exit0且无诊断错误。
- 保留中间迭代和失败诊断日志，不计入最终对照统计（以下均为本地未入库运行产物）：
- `output/anima-mixed-swap-fix-20261002-163852/metrics.json`
- `output/anima-mixed-swap-fix-20261002-163852/memory-comparison.png`
- `output/anima-mixed-swap-fix-20261002-163852/optimized-status.json`
- `output/anima-mixed-swap-fix-20261002-163852/controlled-verified-status.json`
- `output/anima-mixed-swap-fix-20261002-163852/numerical-verification.json`
- `output/anima-mixed-swap-fix-20261002-163852/offloading-before.py`
- bf16：`output/anima-mixed-swap-fix-20261002-163852/bf16.toml`、`output/anima-mixed-swap-fix-20261002-163852/bf16.phase.jsonl`、`output/anima-mixed-swap-fix-20261002-163852/bf16.log`。
- mixed-before：`output/anima-mixed-swap-fix-20261002-163852/mixed-before.toml`、`output/anima-mixed-swap-fix-20261002-163852/mixed-before.phase.jsonl`、`output/anima-mixed-swap-fix-20261002-163852/mixed-before.log`。
- mixed-optimized：`output/anima-mixed-swap-fix-20261002-163852/mixed-optimized.toml`、`output/anima-mixed-swap-fix-20261002-163852/mixed-optimized.phase.jsonl`、`output/anima-mixed-swap-fix-20261002-163852/mixed-optimized.log`。
- controlled-before-verified：`output/anima-mixed-swap-fix-20261002-163852/controlled-before-verified.toml`、`output/anima-mixed-swap-fix-20261002-163852/controlled-before-verified.phase.jsonl`、`output/anima-mixed-swap-fix-20261002-163852/controlled-before-verified.log`。
- controlled-after-verified：`output/anima-mixed-swap-fix-20261002-163852/controlled-after-verified.toml`、`output/anima-mixed-swap-fix-20261002-163852/controlled-after-verified.phase.jsonl`、`output/anima-mixed-swap-fix-20261002-163852/controlled-after-verified.log`。
