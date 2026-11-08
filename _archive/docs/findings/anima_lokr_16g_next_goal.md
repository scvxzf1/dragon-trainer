# Anima LoKr 16G Next Goal

状态：研究记录 / 阶段快照
适用版本：以文中日期、提交和运行环境为准；不作为当前 main 操作说明

日期：2026-06-05

## 活跃 goal

在保持以下约束不变的前提下，继续推进 LoKr 16G 训练稳定性与速度：

- `torch_compile = true`
- 不修改 attention backend，继续使用 `attn_mode = "flash"`
- 不把 full `gradient_checkpointing` 作为默认方案
- trainable LoKr / router / adapter 常驻 GPU
- block swap 只处理 frozen base block

目标指标：

- 最低 forward free 从当前 `25-76MiB` 提升到 `>= 300MiB`，理想 `500MiB+`
- 速度损失控制在 `<= 10%~12%`
- 不牺牲当前 G8 300-step 已验证的可运行性

## 已完成基线

当前仓库已经具备：

1. LoKr no-kron custom path：训练路径不再 materialize `torch.kron(w1, w2)`。
2. grouped projection：`lokr_factor_group_size=8` 是当前速度默认。
3. fused LoKr delta apply：按 row chunk 写入 frozen Linear base output，不返回完整 fp32 group delta。
4. `peak_probe_jsonl`：支持 `block` / `ops` / `lokr` / `full` 粒度。
5. 细粒度 selective checkpoint：`mlp_layer1_only`、`peak_blocks_mlp_layer1`、`peak_blocks_mlp`。

关键结论见 `docs/findings/anima_lokr_blockswap_oom_report.md`：

- G8 300-step 可跑，但最低 free 约 `46MiB`，仍然极薄。
- 50-step block probe 最低 free 约 `25MiB`。
- `ops/lokr` 短跑把危险点收敛到后段 block，尤其 `block 27` 的 MLP LoKr apply。
- 定点 checkpoint 没有把最低 free 拉到 `300MiB+`。
- fused LoKr delta apply 降低 `max allocated` 约 `145MiB`，但 `max reserved` 仍接近 `13.91GiB`，NVML free 仍只有约 `66MiB`。
- `wait_ms p95 < 1ms`，当前瓶颈不是 block swap 等待。

## 下一阶段优先级

### P0：保持当前实现可验证

每轮修改前后至少跑：

```bash
.venv/bin/python -m py_compile library/runtime/peak_probe.py library/anima/models.py networks/plugins/lokr/module.py networks/plugins/lokr/autograd.py library/training/loop.py train.py library/training/cli_args.py tests/test_lokr.py tests/test_block_swapping.py tests/test_config.py
.venv/bin/python -m pytest tests/test_lokr.py -q
.venv/bin/python -m pytest tests/test_config.py -q
```

### P1：降低 allocator/reserved 峰值

当前最小 free 主要被 reserved / CUDA context / 桌面常驻显存共同决定。下一轮优先测试 allocator 组合，而不是继续堆 checkpoint。

候选：

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8`
- 对比当前推荐 `expandable_segments:True,max_split_size_mb:256`

实验：

- 固定 `blocks_to_swap=23`
- 固定 `lokr_factor_group_size=8`
- 固定 `selective_checkpoint="off"`
- 固定 `torch_compile=true`、`attn_mode="flash"`
- 先 50-step，再对最优 allocator 300-step

成功判据：

- `min peak free >= 300MiB` 或至少较当前 `~66MiB` 明显提升
- step time 不比当前 G8 baseline 慢 `> 10%~12%`
- `max_reserved` 不持续膨胀

### P2：更彻底 fused base+delta Linear

当前 fused LoKr 只把 delta 写进 base output；base Linear 自身仍由 `org_forward` 先分配完整输出，再进行 LoKr 写入。下一阶段研究是否能在 LoKr wrapper 中把 base Linear 和 LoKr delta 的写入进一步融合，减少峰值输出/临时生命周期。

约束：

- 不改变数学路径
- 不改变权重格式
- 不量化 trainable LoKr
- 先 PyTorch/custom autograd 原型，Triton 仅作为后续优化

成功判据：

- `max allocated` 继续下降至少 `100MiB+`
- `min free` 明显抬升
- 数值与旧路径对齐，`tests/test_lokr.py` 继续通过

### P3：FP8/int8 frozen block 传输压缩

该项是速度/带宽增强，不是第一修复点。只允许作用于 frozen base block 权重的 CPU→GPU transfer/staging，不作用于 LoKr trainable 参数、optimizer state 或梯度。

成功判据：

- H2D p95 明显下降
- wait p95 不上升
- loss/样本无明显漂移

## 暂停方向

暂时不要继续投入：

- 盲目提高 `blocks_to_swap`
- 把 full checkpoint 设为默认
- 关闭 `torch_compile`
- 更换 attention backend
- 继续扩大粗粒度 `selective_checkpoint`
- 未经 profiling 直接写 Triton 全量重构

## 当前推荐用户配置

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

## 2026-06-05 goal 推进结论

已执行 allocator / blocks_to_swap / FP8 transfer 追加消融。结论：

- allocator 组合没有把最低 free 拉到 300MiB；最佳约 78.5MiB。
- `blocks_to_swap=24 + alloc64_gc08` 50-step 可跑，但最低 free 仍约 78.7MiB。
- `fp8_e4m3` transfer 把 H2D p95 从约 14.5ms 降到约 11.1ms，CPU master 从 3.61GiB 降到 1.80GiB，但最低 free 仍只有约 82MiB。

因此当前 goal 的工程结论是：现有安全路线已经接近极限；FP8 是传输/速度优化，不是 OOM 根因修复。若强制要求 300MiB+，下一阶段必须转向更深层 kernel/graph/autograd 生命周期重构，或接受 full checkpoint / 降 token 峰值的代价。
