# Krea-2 在 V100 上的 autocast(bf16) 双重截胡：attention 与 dir1 的 fp16 内层都被吃掉

状态：稳定（已修复并实测）
适用版本：当前 `codex/krea2-fp16-layer-scaling`
相关代码：`library/quantization/nf4.py`、`library/models/krea2_raw/attention_backend.py`
复现脚本：`bench/krea2_nf4_fp16_grad/attn_fp16_probe.py`

## 1. 背景：V100 (sm70) 没有 bf16 tensor core

Krea-2 生产 `compute_dtype` 固定 bfloat16（22 处 fail-closed 契约）。但 sm70
上 bf16 没有任何可用的 SDPA 内核（FLASH / EFFICIENT / CUDNN 全部
`No available kernel`，只剩 MATH），且任何 bf16 GEMM 都会被 PyTorch 抛给
**MAGMA 软件实现**：

```
magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16, ...>
```

这是纯通用 CUDA core 的软模拟，吞吐比 fp16 tensor core 低一个数量级。

因此有两条内层 fp16 方案（外层仍 bf16，22 处契约不动）：

- **dir1**：`Linear4bit` 的 GEMM 切 fp16 + `frexp`/`ldexp` 逐层缩放
- **attention fp16**：GQA 展开成 MHA 让 fp16 走 EFFICIENT + 反向 `dO` 缩放

## 2. 根因：`torch.autocast(dtype=bfloat16)` 会截胡已经手转 fp16 的 op

PyTorch 的 AutocastCUDA 策略表把 `aten::linear`、`aten::scaled_dot_product_attention`
等列为 **CAST_TO_AUTOCAST_DTYPE**：即使调用方已经显式 `.to(torch.float16)`，
只要外层 `autocast(dtype=bfloat16)` 还开着，C++ 分发器会在进内核前把 fp16
张量**再转回 bfloat16**。

两个内层方案都被这一条吃掉了：

### 2.1 attention：静默回退 MATH

```
autocast OFF   + fp16 -> EFFICIENT OK
autocast bf16  + fp16 -> No available kernel   <-- 训练的实际组合
autocast fp16  + fp16 -> EFFICIENT OK
```

4864 tokens 单层 MATH 需要 `48 × 4864² × 4 bytes = 4.23 GiB`，直接 OOM。

### 2.2 dir1：探针抓到的铁证

```
[DIR1MM] A=bf16->fp16  w=fp16  autocast=True acdtype=bf16
[DIR1MM] out=bf16      expect=fp16          <-- 被转回来了
```

dir1 明明启用了（`args.nf4_dir1=True`、`NF4 dir1 enabled` 日志出现、
`_forward_dir1` 被调用），但 `F.linear` 的输出仍是 bf16，264 个主干 GEMM
全部落回 MAGMA。

## 3. profiler 证据

修复前单步 35s，CUDA kernel 27.3s（78%）是 bf16 MAGMA。按 dtype + shape 聚合：

| GPU 耗时 | 次数 | dtype | shape | 归属 |
|---:|---:|---|---|---|
| 12.396s | 112 | bf16 | `[[4864,6144],[6144,16384]]` | MLP up/gate |
| 6.642s | 168 | bf16 | `[[4864,6144],[6144,6144]]` | attn QKV + out |
| 6.022s | 56 | bf16 | `[[4864,16384],[16384,6144]]` | MLP down |
| 1.157s | 112 | bf16 | `[[4864,6144],[6144,1536]]` | AdaLN 调制 |
| 0.457s | 420 | bf16 | `[[4864,6144],[6144,16]]` | LoRA A |
| 0.322s | 112 | bf16 | `[[4864,16384],[16384,16]]` | LoRA B |

次数自洽：`112 = 28 层 × 4`（前向 + 反向重算），`56 = 28 × 2`。

修复后单步 10.78s：

| 类别 | 耗时 | 占比 |
|---|---:|---:|
| fp16 tensor core（dir1） | 3.372s | 31.3% |
| attention (fmha) | 2.878s | 26.7% |
| 逐元素/拷贝 | 2.065s | 19.1% |
| **bf16 MAGMA（残留）** | **1.420s** | 13.2% |
| 其它 | 0.906s | 8.4% |
| NF4 dequant | 0.143s | 1.3% |

## 4. 修复

两处都用 `torch.autocast(device_type="cuda", enabled=False)` 包住内层计算：

```python
# library/quantization/nf4.py — _Dir1MatMul4Bit.forward
with torch.autocast(device_type="cuda", enabled=False):
    output = torch.nn.functional.linear(
        A.to(torch.float16),
        w.t(),
        None if bias is None else bias.to(torch.float16),
    )

# library/models/krea2_raw/attention_backend.py — _fp16_attention
with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel([...]):
    out16 = F.scaled_dot_product_attention(q16, k16, v16, ...)
```

顺带修掉两个隐患：

- `_PreFp16SDPA.forward` 对 q/k/v 统一 `.contiguous()`：EFFICIENT 对非连续
  输入同样报 `No available kernel` 并静默回退 MATH。
- 新增 `K2_ATTN_STRICT=1` 严格模式：强制 EFFICIENT，失败时打印
  shape/stride/dtype/contiguity/mask 统计，取代静默回退。这是本次能定位根因的关键工具。

## 5. 实测（V100-SXM2-16GB，Krea-2 1024² = 4864 tokens，NF4）

| 配置 | s/step | 20 步总耗时 | peak alloc | avr_loss |
|---|---:|---:|---:|---:|
| 修复前（仅 attn fp16） | 44.8 | 15:20 | 5.05 GiB | 0.169 |
| 双开但 dir1 被截胡 | 35.0 | — | 6.89 GiB | — |
| **修复后（双开生效）** | **11.06** | **3:47** | **6.89 GiB** | 0.205 |

`recent_s_per_step` 稳定在 11.02~11.06，无明显热降频。

显存随 `blocks_to_swap` 的分布（修复后）：

| swap | peak allocated |
|---:|---:|
| 26 | 5.05 GiB |
| 20 | 6.89 GiB |
| 10 | 8.28 GiB |
| 0 | 14.80 GiB OOM |

**口径注意**：与 3080 bf16+flash 的 8.25~9.63 GiB 对比时，应使用 swap 10 的
8.28 GiB；swap 26 的 5.05 GiB 是换出来的，不是净收益。

### 5.1 长训收敛验证（150 步，固定 seed=42）

| step | avr_loss | s/step |
|---:|---:|---:|
| 15 | 0.2100 | 11.04 |
| 30 | 0.1900 | 11.07 |
| 45 | 0.1870 | 11.08 |
| 60 | 0.1830 | 11.11 |
| 75 | 0.1800 | 11.77 |
| 90 | 0.1860 | 11.78 |
| 105 | 0.1800 | 11.76 |
| 120 | 0.1850 | 11.05 |
| 135 | 0.1820 | 11.76 |
| 150 | 0.1790 | 11.76 |

首/末/最低/最高 = 0.1770 / 0.1790 / 0.1760 / 0.2350，**全程无 NaN/inf**。
`s/step` 稳定在 11.04~11.78，峰值显存 6.89 GiB。

loss 在 0.18 附近进入平台是正常的：扩散训练每步随机采样的 σ 不同，单步 loss
本身就有数倍波动（首步 0.235 到 0.177 的差异主要来自 σ 采样，不是模型退化）。
150 步只够验证「不炸、不掉速、稳定下降」，不足以判断最终收敛质量。

## 6. 防坑模式（写给后来者）

**在 sm70 上，任何"内层手切 fp16"的优化都必须用 `autocast(enabled=False)`
包住对应 op，否则外层 bf16 autocast 会静默把你转回去。**

自查方法：

1. 打一行 `print(output.dtype)` 确认真的是 fp16；
2. profiler 里看有没有 `magma_sgemmEx_kernel<... bf16 ...>` 占大头——
   有就是被截胡了；
3. SDPA 相关用 `K2_ATTN_STRICT=1` 直接报错，不要依赖静默回退。

## 7. 剩余优化空间

修复后 10.78s 里还有两块可压：

- **bf16 MAGMA 残留 1.42s**：AdaLN 调制层和 LoRA 旁路是普通 `nn.Linear`，
  不受 dir1 覆盖。要压需要把它们也纳入 fp16 内层。
- **逐元素/拷贝 2.07s**：RoPE / RMSNorm / gate 未编译。V100 上 `torch.compile`
  会跳过 bf16 图（`does not support bfloat16 compilation natively`），
  要压需要让 block 内部整体走 fp16。
