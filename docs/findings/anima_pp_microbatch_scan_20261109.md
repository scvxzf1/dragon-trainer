# Anima PP v0 的 microbatch 扫描：M=4/8/16/32 与拐点

状态：实验
适用版本：当前 dev（2026-11-09；测量基线 `7476f3a0`）
入口命令：见 §6 复现（`scripts/experiments/anima_parallel/pp_v0_bench.py`）
相关代码：`library/training/pp/runtime.py`、`library/training/pp/schedule.py`、`library/training/pp/contract.py`

## 1. 结论

在 2×V100-SXM2-16GB / Anima PP=2 / Schedule1F1B / 单 token 桶（4200 族）/ fp16 的固定配置下，
把 microbatch 数从 4 提到 32 的实测结果：

| M | Global Batch | s/step | s/sample | 加速比 | 峰值 allocated | 峰值 reserved |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 4 | 7.151 | 1.788 | 1.520× | 5.543 GB | 5.667 GB |
| 8 | 8 | 13.127 | 1.641 | 1.656× | 5.688 GB | 5.847 GB |
| **16** | 16 | 24.688 | **1.543** | **1.761×** | 5.981 GB | 6.149 GB |
| 32 | 32 | 48.605 | 1.519 | 1.789× | 6.566 GB | 6.761 GB |

**推荐 M=16**：相对 M=4 每样本快 13.7%、加速比从 1.520× 提到 1.761×，显存只多 0.44 GB。
**不建议 M=32**：只再快 1.6%，却多花 0.59 GB。

**拐点在 M=16 与 M=32 之间**，判据是边际性价比而非绝对速度（见 §3）。

## 2. 测量条件

| 项 | 值 |
| --- | --- |
| 硬件 | 2×Tesla V100-SXM2-16GB（`CUDA_VISIBLE_DEVICES=0,1`，major 7） |
| 软件 | torch 2.10.0+cu129 / Python 3.13.9 / 驱动 580.196.07 |
| 模型 | `Anima-2.9B-preview-v1.safetensors`，40 blocks，两 stage 各 20 |
| 精度 | fp16 + GradScaler；`PpComputePolicy` 由 pre-block dtype 决定（`use_fp32=True`、boundary fp32、prediction fp16） |
| 调度 | `Schedule1F1B`，microbatch_size 恒为 1（M 即 Global Batch） |
| 数据 | 单 token 桶 4200 族，latent `(B, 16, 1, 150, 112)` |
| attention | `torch`（SDPA）；本 venv 无 flash_attn / xformers |
| 进程组 | 多后端 `cpu:gloo,cuda:nccl` |
| 步数 | 1 warmup + 2 测量步 |

四档全部 `skipped=false`、`scale` 恒 65536、`first_non_finite_block=None`、loss 有限。

**注意**：四档的 loss 数值不可横向比较——每档要载入 `3×M` 个 batch 再拼成一步
（分别是 12 / 24 / 48 / 96 个），拼出来的内容不同，实测 loss 也确实不同。
本文只关心耗时与显存，不关心收敛。

## 3. 拐点：边际性价比断崖

| 区间 | 每样本省下 | 多花显存 | 性价比 |
| --- | --- | --- | --- |
| M=4 → 8 | 0.147 s（−8.2%） | +0.145 GB | **1.01 s/GB** |
| M=8 → 16 | 0.098 s（−6.0%） | +0.293 GB | **0.33 s/GB** |
| M=16 → 32 | 0.024 s（−1.6%） | +0.585 GB | **0.041 s/GB** |

从 M=16 到 M=32，多花 10% 显存只换来 1.6% 速度，**性价比掉约 8 倍**。

## 4. 与 1F1B 理论天花板的关系

2 stage 时，1F1B 的空泡比例是 `(S−1)/(M+S−1) = 1/(M+1)`，
因此加速比天花板 = `M/(M+1) × 2`。

| M | 理论加速 | 实测 | 达成率 |
| --- | --- | --- | --- |
| 4 | 1.600× | 1.520× | 95.0% |
| 8 | 1.778× | 1.656× | 93.1% |
| 16 | 1.882× | 1.761× | 93.5% |
| 32 | 1.939× | 1.789× | 92.2% |

达成率在 92–95% 区间小幅下滑：M 越大，每步的 NCCL 同步次数随之增加，但实现本身没有劣化趋势。

**推论**：1F1B 的天花板本身趋近 2.0×，而实测达成率稳定在 92–95%，
因此**这套实现在 2 卡上的实测上限收敛在约 1.85–1.9×**。
无论 batch 加到多大都不会超过它，想突破只能换调度（interleaved / 虚拟 stage）
或减少单卡计算量。

## 5. 显存不是限制

M=32 只用 6.57 GB（reserved 6.76），16 GB 卡还剩一半以上。
**限制来自收益递减，不是显存。**

一个反直觉的观察：从 M=4 到 M=8，显存只涨 0.145 GB。
原因是 gradient checkpointing + 1F1B 让**同时存活的激活数有界**，
不随总 batch 线性增长——所以"batch 翻倍 → 激活翻倍"的直觉估算在这里是错的。

## 6. 复现

测量脚本已纳入版本控制：[`scripts/experiments/anima_parallel/pp_v0_bench.py`](../../scripts/experiments/anima_parallel/pp_v0_bench.py)。

PP v0 要求恰好两个 rank，每 rank 一个进程：

```bash
for R in 0 1; do
  CUDA_VISIBLE_DEVICES=0,1 RANK=$R LOCAL_RANK=$R WORLD_SIZE=2 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29561 \
  .venv/bin/python -m scripts.experiments.anima_parallel.pp_v0_bench \
    --mode pp --output-dir <输出目录> \
    --microbatches 16 --steps 2 --warmup-steps 1 \
    --dit-path <Anima-2.9B-preview-v1.safetensors> \
    --latent-path <*_anima.npz，需为 150x112 的 4200 族> \
    --text-path <*_anima_te.safetensors> &
done
wait
```

`--microbatches` 同时决定 wire buffer 的 batch 维度（脚本内 `batch_shapes()` 推导），
**不需要再手改常量**；microbatch_size 恒为 1，因此 M 即 Global Batch。

`CUDA_VISIBLE_DEVICES=0,1`（**不是** nvidia-smi 序号 `1,2`，本机两者不一致）。

## 7. 边界与未验证

- 每档只有 **2 个测量步**，未做长时间稳态；不含热降频、不含 dataloader 真实吞吐。
- **加速比是每样本吞吐对比，不是同 batch 的端到端对比**：参考值是单卡 GBS=4 实测
  2.717 s/sample（峰值 14.49 GB）。单卡 GBS=16 若沿用本 bench 的「全部 microbatch 拼成
  一个 batch、做一次前向」口径，显存需约 58 GB（14.49 × 4 线性外推），**物理上不可测**，
  因此没有同 batch 的单卡基线。
  这与 §5 不矛盾：§5 说的是 PP 路径下 1F1B 让存活的激活数有界，而单卡 bench 是把整批
  拼成一次前向，两者**口径不同**。若改成梯度累积口径，单卡显存会低得多，但那不是本次
  测量的基准。
- 未测 M>32；未测 3 stage 及以上；未测非 4200 token 桶。
- 未做 fp16 数值等价复测（M 改变会改变累加顺序，`track ②` 的同切分参考需要重新取证）。
- 采集期间曾出现一次外部并发进程与本次测量争抢同一物理 GPU 的显存与算力
  （resource contention）；该轮采集已整体作废并从基准数据中剔除，本文四档均在
  GPU 独占窗口内取得。
