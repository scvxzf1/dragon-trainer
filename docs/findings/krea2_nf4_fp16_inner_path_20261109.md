# Krea-2 NF4 fp16 内层路径：autocast 静默截胡的两次实例与 V100 实测

状态：实验
适用版本：commit `afa372ae`（分支 `codex/krea2-fp16-layer-scaling`）
入口命令：
- `.venv/bin/python train.py --method krea2_lora --base_compute nf4 --nf4_prequantized_path models/diffusion_models/krea2_raw_nf4_self_contained.safetensors --attn_mode torch --krea2_attn_fp16 --nf4_dir1 --blocks_to_swap 20`
- 诊断模式：`K2_ATTN_STRICT=1`（attention）、`_DIR1_STATS`（dir1 调用计数）
相关代码：
- `library/quantization/nf4.py`（`_Dir1MatMul4Bit`、`BF16AutocastLinear4bit`）
- `library/models/krea2_raw/attention_backend.py`（`_PreFp16SDPA` / `_PostFp16SDPA` / `_fp16_attention`）
- `docs/proposal/krea2_nf4_fp16_layer_scaling.md`（设计推导）
- `docs/findings/krea2_nf4_fp16_grad_underflow_20261108.md`（数值可行性前置）

## 1. 结论

**`torch.autocast(dtype=bfloat16)` 会把 fp16 内层计算的输入静默转回 bf16，使两条 fp16 优化路径同时失效且不报错。** 这不是一个实现缺陷，而是一个反复出现的模式——本次在两个互相独立的位置各命中一次：

| 位置 | 表现 | 修复 |
|---|---|---|
| `_fp16_attention` 的 `F.scaled_dot_product_attention` | EFFICIENT 内核 `No available kernel`，静默回退 MATH（单层 4.23 GiB） | `torch.autocast(device_type="cuda", enabled=False)` 包裹（`e783f584`） |
| `_Dir1MatMul4Bit.forward` 的 `F.linear` | fp16 输入被转回 bf16，264 个主干 GEMM 全部落到 MAGMA 软件路径 | 同上（`bb8c5b63`） |

V100（sm70）上的实测结果：

| 配置 | s/step | 20 步总耗时 | peak allocated | loss |
|---|---:|---:|---:|---:|
| 仅 `--krea2_attn_fp16`，swap 26 | 44.8 | 15:20 | 5.05 GiB | 0.169 |
| 双开但 dir1 被 autocast 截胡，swap 20 | 35.0 | — | 6.89 GiB | — |
| **双开且两者都生效，swap 20** | **11.06** | **3:47** | **6.89 GiB** | **0.205** |

修复前后差 **4.05×**。`recent_s_per_step` 在 20 步内稳定在 11.02~11.06，无热降频。

## 2. 根因：autocast 的 dtype 抢占

`torch.autocast(dtype=torch.bfloat16)` 会对列入 autocast 白名单的算子（含 `F.linear`、`F.scaled_dot_product_attention`）做输入 dtype 收敛：**任何落入白名单算子的 fp16 输入都会被强制转成 autocast 的 dtype**。

在 sm70 上这特别致命——V100 没有 bf16 tensor core，也不存在 bf16 SDPA 内核：

- bf16 GEMM 落到 **MAGMA 软件实现**（`magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16>`），吞吐比 tensor core 低一个数量级；
- bf16 SDPA 只剩 **MATH** 后端，物化 `O(N²)` 注意力矩阵。

于是「外层 bf16 + 内层 fp16」的设计意图被 autocast 完整抹平，且**全程无异常、无警告**。

### 2.1 attention 侧的呈现

严格模式（`K2_ATTN_STRICT=1`）强制 EFFICIENT 后拿到完整诊断：

```
q: shape=(1, 48, 4608, 128) dtype=torch.float16 contig=True
mask: shape=(1, 1, 4608, 4608) dtype=torch.bool contig=True mask.sum=17007376
original: RuntimeError: No available kernel. Aborting execution.
```

`q` 明明是 fp16、连续、mask 合法，EFFICIENT 仍然拒绝。最小复现把变量隔离出来：

| 条件 | EFFICIENT |
|---|---|
| autocast OFF + fp16 | OK |
| **autocast bf16 ON + fp16** | **`No available kernel`** |
| autocast fp16 ON + fp16 | OK |

同一个 L=4608 + 20% padding bool mask 的组合，在 autocast 外完全正常——**mask 与长度都不是原因**，autocast 才是。

### 2.2 dir1 侧的呈现

`dir1` 的开关、日志、调用计数都显示「已启用」：

```
[DIR1DBG] args.nf4_dir1=True type=Namespace has_attr=True
INFO     NF4 dir1 enabled: inner GEMM runs fp16 ...
```

`_forward_dir1` 也确实被调用。但探针打在 `F.linear` 两侧，看到的是：

```
[DIR1MM] A=bf16->fp16  w=fp16  autocast=True acdtype=bf16
[DIR1MM] out=bf16      expect=fp16          <-- 输出被 autocast 改回 bf16
```

**输入已经手工转成 fp16，输出仍然是 bf16。** 这一层转换发生在 PyTorch 内部，Python 侧不可见。

## 3. 诊断方法论

本次定位依赖三类证据，按可信度排序。

### 3.1 CUDA memory snapshot 定位「哪一行在分配大显存」

`torch.cuda.memory._record_memory_history(max_entries=..., stacks="all")` + OOM 时 `_dump_snapshot()`，再按 block 大小排序取调用栈。

第一次命中的是 `_fp16_attention` 内部的 SDPA 调用点，`4332 MiB = 48 × 4864² × 4 bytes`——直接把「回退 MATH」这个假设钉死。

注意两点：
- `stacks="all"` 记录的栈前几帧是 snapshot 工具自身（`torch::cuda::gather_with_cpp`），要按 `/mnt/SD_Dataset` 过滤出仓库内帧；
- snapshot 只列 `active_allocated` 的块；瞬时分配的峰值要看 `device_traces`。

### 3.2 profiler 按 dtype + shape 聚合

`torch.profiler` 的 trace 里，CUDA kernel 的 `args.External id` 与 CPU op 的 `args.External id` 对应。对每个 CPU op 取 `Input type` / `Input Dims`，再把关联 kernel 的 `dur` 归到该 op 上，就能得到 **dtype × shape × GPU 耗时** 的精确分布。

本次输出的表：

```
12.396s x112  aten::mm  BFloat16  [[4864, 6144], [6144, 16384]]   MLP up/gate
 6.642s x168  aten::mm  BFloat16  [[4864, 6144], [6144, 6144]]    attn QKV + out
 6.022s x56   aten::mm  BFloat16  [[4864, 16384], [16384, 6144]]  MLP down
 1.157s x112  aten::mm  BFloat16  [[4864, 6144], [6144, 1536]]    AdaLN 调制
 0.457s x420  aten::mm  BFloat16  [[4864, 6144], [6144, 16]]      LoRA A
```

调用次数与 28 层结构完全吻合（`112 = 28 × 4`、`56 = 28 × 2`），可以作为形状归属正确性的交叉校验。

**限制**：PyTorch 2.10 的 trace 里即使 `with_stack=True` 也没有可直接读取的 `Call stack` 字段，调用栈只能由 shape + 次数 + 代码路径推断。

### 3.3 计数器与 dtype 探针

对「开关到底有没有穿透」这类问题，布尔值不够。有效做法是在目标分支入口累加计数、在算子两侧打印输入输出 dtype——本次正是 `out=bf16` 与 `expect=fp16` 的对比直接指认了 autocast。

## 4. 修复

两处都是同一个形状：**把纯 fp16 的内层计算移出 autocast 作用域**。

```python
# library/models/krea2_raw/attention_backend.py
with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel([...]):
    out16 = F.scaled_dot_product_attention(q16, k16, v16, ...)
```

```python
# library/quantization/nf4.py — _Dir1MatMul4Bit.forward
with torch.autocast(device_type="cuda", enabled=False):
    output = torch.nn.functional.linear(
        A.to(torch.float16), w.t(),
        None if bias is None else bias.to(torch.float16),
    )
if output.dtype != torch.float16:   # 兜底：防止外层再改 dtype
    output = output.to(torch.float16)
```

`compute_dtype` 全程保持 bfloat16，`weights.py` / `quantize.py` 的 22 处 BF16 fail-closed 契约未改动。

### 4.1 顺带修掉的第二个隐患

`_PreFp16SDPA.forward` 对 q/k/v 统一加 `.contiguous()`。原因是 RMSNorm / rearrange 之后 v 常常是非连续的，而 sm70 的 memory-efficient 内核对非连续输入同样以 `No available kernel` 静默回退 MATH。成本是一次 O(n) 拷贝。

## 5. 显存与 swap 的关系（重要的口径澄清）

`blocks_to_swap` 对显存的影响是单调的，但对速度**几乎没有影响，甚至 swap 多的略快**：

| swap | peak allocated | reserved | s/step |
|---:|---:|---:|---:|
| 26 | 5.05 GiB | 5.49 | 44.8（当时 dir1 未生效） |
| 20 | 6.26 GiB | 6.68 | 45.9 |
| 10 | 8.28 GiB | 8.73 | 47.1 |
| 0 | 14.80 GiB | — | **OOM** |

**因此「5.05 GiB」不是一个可以和 bf16 基线直接对比的数字**——它是 `blocks_to_swap 26` 换来的。等效口径应取 swap 10 的 **8.28 GiB**，与 3080 上 bf16 + flash 的 8.25~9.63 GiB 基本持平。

### 5.1 与 3080 基线的差异来源

V100 与 3080 的 fp16 算力相近（125 vs 120 TFLOPS），但：

| | V100 (sm70) | 3080 (sm86) |
|---|---|---|
| bf16 tensor core | 无 | 有 |
| flash-attn | 不可用（依赖 sm80+ 指令） | 可用 |
| `torch.compile` bf16 | **inductor 直接跳过** | 原生支持 |

`Tesla V100-SXM2-16GB does not support bfloat16 compilation natively, skipping` 意味着 V100 上不存在「编译后变慢」的中间态，而是根本没有编译。这也是为什么当前实测命令使用 `torch_compile = false`。

## 6. 未验证边界

- **训练质量**：20 步 loss 0.205，远未到收敛，不能作为数值正确性的证明。完整数值链验证见 `krea2_nf4_fp16_grad_underflow_20261108.md`（固定种子下 bf16 1.316%、dir1 0.988%、裸 fp16 135.315%）。
- **11s/step 的剖面**：修复后未重新 profiler，剩余 11s 中 fp16 tensor core / attention / swap 搬运的占比未知。
- **单项开关**：`--nf4_dir1` 单独开启（不含 attention fp16）的耗时未测。
- **3080 的 3.6s 基线口径**：`krea2_3080_speed_final.md` 未记录该读数的 token 数；按 4864 tokens 的算力推算（约 332.7 TFLOPs / 120 TFLOPS ≈ 2.8s 理论下限，实际效率 30~50% 即 6~9s），3.6s 不太可能对应 4864 tokens，需要回到原始实验条件确认。
- **swap 0 的真实峰值**：只观察到 OOM，未测得具体数值。

## 7. 复现

```bash
# 主路径（双开）
.venv/bin/python train.py --method krea2_lora \
  --base_compute nf4 \
  --nf4_prequantized_path models/diffusion_models/krea2_raw_nf4_self_contained.safetensors \
  --max_train_steps 20 --attn_mode torch \
  --krea2_attn_fp16 --nf4_dir1 --blocks_to_swap 20

# attention 严格诊断（强制 EFFICIENT，失败时打印全量张量信息）
K2_ATTN_STRICT=1 .venv/bin/python train.py ...（同上）

# 显存分配栈（OOM 时 dump snapshot）
#   用 runpy + torch.cuda.memory._record_memory_history(stacks="all")
#   包住 train.py，异常时 _dump_snapshot()

# profiler（dtype × shape × GPU 耗时）
#   torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True)
#   按 args["External id"] 关联 CPU op 与 CUDA kernel
```

## 8. 相关文档

- [NF4 fp16 梯度下溢与逐层缩放判据](krea2_nf4_fp16_grad_underflow_20261108.md)——本方案的数值可行性前置
- [Krea-2 NF4 fp16 逐层缩放提案](../proposal/krea2_nf4_fp16_layer_scaling.md)——设计推导与实现边界
- [Krea-2 3080 最终结论](krea2_3080_speed_final.md)——另一硬件平台的基线
