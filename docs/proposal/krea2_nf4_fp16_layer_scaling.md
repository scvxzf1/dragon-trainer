状态：活跃提案 / 未实现（原型已验证，生产未落地）
日期：2026-11-09
适用版本：当前 dev（分支 `codex/krea2-fp16-layer-scaling`）
前置证据：[NF4 fp16 梯度偏差机制溯源](../findings/krea2_nf4_fp16_grad_underflow_20261108.md)
相关代码：`library/quantization/nf4.py`（唯一介入点）
探针：`bench/krea2_nf4_fp16_grad/sim_layer_scaling.py`、`bench/krea2_nf4_fp16_grad/dir1_chain.py`

# Krea-2 NF4 逐层 fp16 缩放（dir1）方案

## 1. 要解决什么

V100（sm70）与 2080Ti（sm75）**没有原生 BF16 Tensor Core**，bf16 矩阵乘法只能走
PyTorch 的 fp32 上采样仿真路径，吞吐接近 fp32。NF4 训练里 264 个 `6144×6144`
大矩阵的前后向因此完全拿不到硬件加速。

直接用 fp16 又不行：[findings](../findings/krea2_nf4_fp16_grad_underflow_20261108.md)
已证明真实中间层 `grad_output` rms 落在 `6.5e-9 ~ 1.7e-7`，100% 低于 fp16 最小正规数
`6.1e-5`，裸 fp16 会让 LoRA 梯度偏离 fp32 参考 114.7%。全局 loss scaling 也救不回来：
k 的硬上限是 15（`loss×2^16 = 65536 > 65504` → 种子 inf → 全 NaN），而 1024² 需要
k≥16~18，窗口在 >512² 关闭。

**dir1 把缩放从「全局 loss 标量」下沉到「每个 Linear 的 backward」**，绕开这个上限。

## 2. 方案

外层 `compute_dtype` 保持 `bfloat16` 不动，只在 NF4 Linear 算子内部换成 fp16 GEMM。

```
前向：F.linear(x.fp16(), dequant(B).fp16().T, bias.fp16()).to(bf16)
      不需要缩放 —— 激活量级 O(1e-2~1e0)，本来就在 fp16 正规区

后向：exp   = torch.frexp(g.abs().max())[1]              # amax 的二进制指数
      shift = torch.clamp(-exp, min=-126, max=126)       # 双向 clamp，见 §4
      g16   = torch.ldexp(g.float(), shift).to(fp16)     # amax ∈ [0.5, 1)
      out   = torch.matmul(g16, dequant(B).fp16().T)
      grad  = torch.ldexp(out.float(), -shift).to(bf16)
```

三个关键设计点：

**① 用 2 的整数次幂平移，而不是 `g / amax`**

`ldexp` 只改 IEEE 754 的指数位，尾数一个 bit 都不动；而 `g / amax` 会引入额外舍入。
副作用是跨 batch 稳定性：只要相邻 batch 的 amax 落在同一个 2 倍区间，`exp` 完全相同，
缩放因子恒定，**不需要维护 running amax 或任何跨步状态**。

**② 全零梯度天然免疫**

`torch.frexp(0.0) == (0.0, 0)`，随后 `ldexp(0.0, 0) == 0.0`。所以不需要 `eps` 或
`torch.where` 防除零——对照 `g * (1.0 / amax)` 在 `amax == 0` 时直接产出 NaN。

**③ 动态范围由外层 bf16 兜底**

这是方案能成立的根本原因。梯度从顶层 `4.4e-3` 衰减到深层 `6.5e-9`，跨 6 个数量级。
**bf16 有 8 位指数（与 fp32 同范围），能把 1e-12 完好送达 Linear 门口；fp16 从 1e-8 起
直接归零。** 实测：

| 梯度值 | bf16 存储 | fp16 存储 |
|---:|---|---|
| 4.4e-3 | 4.3945e-3 | 4.3983e-3 |
| 1e-8 | **1.0012e-08** | **0.0000e+00** |
| 6.5e-9 | **6.4902e-09** | **0.0000e+00** |
| 1e-12 | **1.0019e-12** | **0.0000e+00** |

所以外层必须是 bf16（或 fp32）；改成 fp16 的话，梯度在 `grad * gate` 这一步就归零，
进 `Linear4bit.backward` 后再做 `ldexp` 也救不回来——那就必须回到 GradScaler 路线。

## 3. 实测结果

### 3.1 精度（真实 NF4 权重，256²，fp32 为裁判）

`bench/krea2_nf4_fp16_grad/dir1_chain.py --img 256`：

| 臂 | loss | LoRA 梯度 rel-L2 vs fp32 | cos | 单步耗时 |
| --- | ---: | ---: | ---: | ---: |
| fp32（裁判） | 0.027427 | 0 | 1.00000 | 7.0s |
| bf16（当前生产） | 0.027981 | 40.614% | 0.92764 | **9.0s** |
| fp16（裸，无缩放） | 0.027369 | **114.741%** | 0.36736 | 2.4s |
| **dir1** | 0.028151 | **40.974%** | 0.92553 | **3.1s** |

- **dir1 把裸 fp16 的 114.7% 救回到 40.97%**，与 bf16 的 40.61% 同一量级。
- **dir1 没有超越 bf16**（差 0.36pp），这是预期内——见 §5。
- dir1 与 bf16 的直接差异：`rel-L2 14.317%`、`cos 0.99246`。两条数值路径不完全一致，
  但都离 fp32 的 40% 是模型固有偏差（findings §4.2 已测 bf16 对 fp32 就是 40.66%）。
- 全程无非有限梯度。

**端到端单步 8.911s → 2.679s（3.33×）**。这是 `dir1_e2e.py` 交替运行 3 轮取中位数
的结果（bf16 极差 0.03s、dir1 极差 0.05s），含 attention、grad-ckpt 重算与 LoRA 前反向
——Linear 只占其中一部分。算子级收益见 §3.2。

> 早期一次顺序运行的测量给出 9.0s → 3.1s（2.9×），但同一脚本内 fp32 反而比 bf16 快，
> 说明顺序效应明显。交替运行后稳定在 3.33×。

### 3.2 算子级基准（V100，cuda.Event，100 次中位数）

**关键：必须用真实 token 数 M，不能用 M=1。** M=1 是「一个梯度向量乘权重」，此时
`6144×6144` 的 GEMM 完全受显存带宽限制（主要开销是读 72 MB 权重），bf16 与 fp16
速度相同，dir1 的额外 kernel 反而让它慢一倍。**M=1 的基准会得出完全相反的结论。**

真实训练里 M 是序列 token 数（256² → 256 tokens，1024² → 4096 tokens），此时是算力限制：

| M | 方向 | bf16 ms | fp16 ms | dir1 ms | bf16/fp16 | **bf16/dir1** |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 256 | forward | 2.998 | 0.322 | 0.605 | 9.32× | **4.95×** |
| 256 | backward | 2.972 | 0.308 | 0.579 | 9.64× | **5.14×** |
| 1024 | forward | 9.590 | 1.166 | 1.561 | 8.22× | **6.15×** |
| 1024 | backward | 9.557 | 1.089 | 1.502 | 8.78× | **6.36×** |
| 4096 | forward | 37.447 | 4.692 | 5.860 | 7.98× | **6.39×** |
| 4096 | backward | 37.447 | 4.220 | 5.728 | 8.87× | **6.54×** |
| 16384 | forward | 149.062 | 18.476 | 23.293 | 8.07× | **6.40×** |
| 16384 | backward | 149.040 | 16.695 | 22.690 | 8.93× | **6.57×** |

**V100 上 bf16 比 fp16 慢 8~9×**——这直接确认了 sm70 没有原生 bf16 tensor core，
bf16 走的是慢路径（甚至比 fp32 的 11478 GFLOPs 还低）。dir1 在付掉 `frexp`/`ldexp`/
`clamp` 的额外开销后，**净收益 5~6.6×**。

纯 matmul 对照（非 NF4，`K=6144`，M=K 方阵）：

| 精度 | 中位数 ms | GFLOPs |
| --- | ---: | ---: |
| fp32 | 40.411 | 11478 |
| bf16 | 55.638 | 8337 |
| fp16 | 6.757 | — |

bf16 比 fp32 还慢 1.38×，这是 V100 上 bf16 仿真路径的直接证据。

### 3.2b dequant 路径优化：`qs.dtype = torch.float16`

bnb 的 `dequantize_4bit` 在 CUDA 内核里用 FP32 寄存器查表并乘 absmax，**只在写回时**
按 `quant_state.dtype` 转换。所以 `dequantize_4bit(...).half()` 会先写出 72 MB 的 bf16，
再启动第二个内核做 72 MB 读 + 72 MB 写。把 `dtype` 临时设为 fp16 可以一步写出：

```python
orig = qs.dtype
qs.dtype = torch.float16
try:
    w16 = bnb_f.dequantize_4bit(B, qs)   # 直接 10 位尾数 fp16
finally:
    qs.dtype = orig
```

实测（6144²，单次 dequant，100 次中位数）：

| 路径 | 中位数 ms | 相对 |
| --- | ---: | --- |
| dequant → bf16 | 0.324 | — |
| dequant → bf16 → `.half()` | 0.581 | +0.257 |
| **dequant → fp16（改 `qs.dtype`）** | **0.308** | **省 0.272 ms（1.88×）** |

**双重收益**：既省掉 144 MB 显存读写，又把权重精度从 bf16 的 7 位尾数提到 fp16 的 10 位。

| dequant 输出 vs fp32 真值 | rel-L2 |
| --- | ---: |
| bfloat16 | 0.16418% |
| **float16** | **0.02061%（准 8 倍）** |

注意 bnb 的 `out=` 参数在当前版本有缺陷（报 `Expected out.shape == [6144, 6144],
got torch.Size([6144, 6144])`，形状实际一致），所以用 `qs.dtype` 而不是 `out=`。

### 3.3 合成矩阵模拟（单层扫描 + 28 层串联）

`bench/krea2_nf4_fp16_grad/sim_layer_scaling.py`，单层 `6144×6144`，dY rms 扫 `1e-10~1e-2`：

| dY rms | bf16 rel-L2 | 裸 fp16 rel-L2 | dir1 rel-L2 |
| ---: | ---: | ---: | ---: |
| 1e-10 | 0.235% | **100.000%** | **0.237%** |
| 1e-9 | 0.236% | **100.000%** | **0.240%** |
| 1e-8 | 0.236% | **100.000%** | **0.238%** |
| 1e-7 | 0.235% | 20.210% | 0.238% |
| 1e-6 | 0.234% | 2.026% | 0.235% |
| 1e-2 | 0.236% | 0.030% | 0.239% |

dir1 在**全部量级**上恒定在 `0.235%~0.240%`，与 bf16 持平；裸 fp16 在 `1e-8` 以下崩溃。

28 层串联（正交权重，衰减 6.3e5 倍），误差累积：

| 层 | bf16 | dir1 | 裸 fp16 |
| ---: | ---: | ---: | ---: |
| 0 | 0.24% | 0.24% | 0.03% |
| 9 | 0.74% | 0.76% | 0.10% |
| 14 | 0.91% | 0.96% | **0.74%** |
| 19 | 1.05% | 1.11% | **8.24%** |
| 23 | 1.15% | 1.22% | **59.42%** |
| 27 | 1.24% | 1.31% | **580.50%** |

裸 fp16 从第 14 层开始崩，到第 27 层完全失控；dir1 与 bf16 全程重合（差 0.07pp）。

**真实权重串联**（`blocks.0..27.attn.wq`，6144²，范数保持 + gate=0.61，
**fp32 dequant 作为真值**）：

| 层 | bf16 | dir1（bf16 权重） | **dir1（fp16 权重）** | 裸 fp16 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.24% | 0.25% | 0.20% | 0.14% |
| 9 | 0.86% | 0.85% | 0.61% | 0.46% |
| 17 | 1.16% | 1.13% | 0.90% | 2.45% |
| 24 | 1.34% | 1.33% | 1.03% | 62.58% |
| 27 | **1.316%** | 1.501% | **0.988%** | 135.315% |

**用 fp16 dequant 权重后，dir1 的末层误差 0.988% 反超 bf16 的 1.316%（好 0.33pp）。**
固定种子后两次运行逐位相同（`dir1_bench.py` 已加 `torch.manual_seed(0)`；未固定时
末层误差会在 0.99%~1.37% 之间抖动）。
这与 §3.2b 的单层权重精度一致（0.021% vs 0.164%），说明 28 层累积没有吃掉这部分优势。

> 若参考臂也用 bf16 dequant（而不是 fp32），bf16 臂会天然零权重误差，比较就不公平。
> 第一版脚本正是踩了这个坑，得出「dir1 略差」的错误结论。

**注意**：不做范数保持时，真实 `attn.wq` 会让梯度放大 3e7 倍（5.1e-1 → 1.5e7），
fp16 直接溢出成 NaN。真实模型里 Linear 之后有 LayerNorm 类归一化，所以脚本必须补上，
否则模拟的不是真实反向链。

## 4. 实现陷阱（四个都踩过）

**① `torch.ldexp` 第二个参数必须是 Tensor**

```python
torch.ldexp(x, -22)        # TypeError: argument 'other' must be Tensor, not int
torch.ldexp(x, -exp)       # OK，exp 来自 torch.frexp
```

**② clamp 必须双向，只写 `min` 拦不住向上溢出**

`amax = 1e-40` 时 `exp = -132`，取负得 `+132`：

```python
torch.clamp(+132, min=-126)          # → +132，仍然出 inf
torch.clamp(+132, min=-126, max=126) # → +126，正确
```

`min=-126` 只防「向下平移溢出」，防不了「向上平移溢出」——因为 `+132 > -126`，
`min` 根本不触发。**必须同时写 `max=126`。**

**③ forward 的 bias 必须一起转 fp16**

`F.linear(x.fp16(), w.fp16().T, bias_bf16)` 会报
`RuntimeError: self and mat2 must have the same dtype, but got BFloat16 and Half`。
bias 不能漏。

**④ bnb 0.49.2 的 `out=` 参数不可用**

报错看起来自相矛盾：

```
RuntimeError: Expected out.shape == [6144, 6144], got torch.Size([6144, 6144])
```

根因在 bnb cuda kernel 的检查：`torch._check(out.shape == shape)`。`shape` 经
`torch.ops` 的 `int[]` schema 分发后被转成 **list**，而 `out.shape` 是 `torch.Size`。
Python 里 `torch.Size` 是 `tuple` 的子类：

```python
torch.Size([6144,6144]) == torch.Size([6144,6144])  # True
torch.Size([6144,6144]) == (6144,6144)              # True
torch.Size([6144,6144]) == [6144,6144]              # False  <-- 这里炸
```

实测传 `Size` / `tuple` / `list` 三种形式**全部失败**（分发层统一转 list），
所以改 `qs.shape` 绕不过去。**不要用 `out=`**，用下面的 `qs.dtype` 浅拷贝方案。

## 5. 精度：与 bf16 持平，用 fp16-dequant 可反超

**输出侧不提升精度。** dir1 的输出仍是 bf16，所以激活流的精度上限是 bf16 的 7 位尾数；
内层 fp16 的 10 位尾数只在 GEMM 累加中起作用。端到端 LoRA 梯度实测 dir1 40.974% vs
bf16 40.614%（差 0.36pp），基本持平。

**但权重侧可以反超。** §3.2b 的 `qs.dtype=fp16` 让 dequant 直接写出 10 位尾数的 fp16
权重（单层 vs fp32 真值：0.021% vs bf16 的 0.164%，准 8 倍），28 层串联末层误差从
bf16 的 1.316% 降到 **0.988%**。

所以准确表述是：

- **激活/梯度流**：精度与 bf16 持平（外层 bf16 决定）；
- **权重**：fp16-dequant 后比 bf16 准 8 倍，且能传导到 28 层累积误差上；
- **端到端 LoRA 梯度**：仍与 bf16 同量级（40.97% vs 40.61%），因为激活流的 bf16 截断
  是主导项，权重精度的改善被它掩盖。

**如果目标是「端到端比 bf16 更准」，dir1 做不到**——那需要外层换 fp32，代价是激活显存
翻倍且要放开 22 处契约。

## 6. 收益与代价

| 维度 | 结果 |
| --- | --- |
| V100/sm75 上 Linear 前后向 | ✅ 走真 tensor core：**算子级 5~6.6×**（M=256~16384），端到端单步 **8.911s → 2.679s（3.33×）** |
| 1024² 深层下溢 | ✅ 逐层归一化，彻底消除（模拟全量级恒定 0.24%） |
| 精度 | ⚠️ 端到端与 bf16 持平（40.97% vs 40.61%）；**权重侧 fp16-dequant 后准 8 倍，28 层累积 0.988% 反超 bf16 的 1.316%** |
| 22 处 fail-closed 契约 | ✅ **0 改动**（`compute_dtype` 保持 bf16） |
| GradScaler | ✅ **不需要**（外层 bf16 保动态范围，逐层缩放替代全局 k） |
| 改动面 | `library/quantization/nf4.py` 一个文件，约 60~90 行 |
| 运行开销 | 每层多一次 `abs().max()` 规约 + 两次 `ldexp`；改用 `qs.dtype=fp16` 后每层 dequant 反而省 0.272 ms |
| sm80+ 显卡 | ⚠️ 收益消失——那些卡有原生 bf16 tensor core，bf16 已经够快 |

**注意最后一行**：dir1 只在**没有原生 bf16 tensor core**的卡上有价值（V100 / 2080Ti /
T4 等 sm70/sm75）。在 3080/3090/4090 上做这件事是纯亏损（多一层转换、没有提速）。

## 7. 工程安全性：QuantState 浅拷贝

`dequant_fp16` 的正确实现不是原地改 `quant_state.dtype`，而是浅拷贝：

```python
import copy

def dequant_fp16(B, quant_state):
    qs = copy.copy(quant_state)      # 只复制 Python 壳，不复制 GPU 张量
    qs.dtype = torch.float16
    return bnb_f.dequantize_4bit(B, qs)
```

`QuantState` 只是轻量 Python 壳（持有 `absmax` / `code` / `state2` 等 GPU 张量的
**引用**和几个标量），`copy.copy` 不复制显存、耗时亚微秒。实测验证：

| 检查 | 结果 |
| --- | --- |
| 浅拷贝后原 `qs.dtype` | 仍为 `bfloat16`（零改动） |
| 共享张量是否被复制 | 否（`absmax.data_ptr()` 相同，无额外显存） |
| 8 线程 × 20 次混合 fp16/bf16 并发 dequant | **0 错误**，原 `qs.dtype` 结束仍为 bf16 |
| `torch.compile(dir1_core, fullgraph=True)` | **通过**，与 eager 的 rel-L2 = 0.000000% |
| 端到端性能（copy.copy vs 原地改） | 3.31× vs 3.33×，无损失 |

因为全程 Zero Mutation，多线程并发读无竞争，**不需要锁、不需要预分配 `out=`、
不需要全局 monkey-patch `dequantize_4bit`**，对模型其他部分完全透明。

## 8. 与其它方案的比较

| 方案 | 精度 | 提速 | 契约改动 | GradScaler | 结论 |
| --- | --- | --- | --- | --- | --- |
| 维持 bf16（现状） | bf16 水平 | 无（sm70 上慢） | 0 | 不需要 | 当前生产 |
| 裸 fp16 + GradScaler | 崩溃（114.7%） | 有 | 22 处 | 需要 | 否决 |
| 全局 loss scaling | 窗口在 >512² 关闭 | 有 | 22 处 | 需要 | 否决 |
| 按分辨率分档（≤512² 用 fp16） | 可行 | 仅小图 | 22 处 | 需要 | 收益低 |
| **dir1（本提案）** | **bf16 水平** | **2.9×** | **0** | **不需要** | **推荐（限 sm70/sm75）** |
| 外层 fp32 + 内层 fp16 | 最高 | 有 | 22 处 | 不需要 | 显存翻倍，暂不采用 |

## 9. 落地前还缺什么

1. **端到端基准仍需多次测量**：算子级已用 cuda.Event 取 100 次中位数（§3.2），
   但 9.0s → 3.1s 的端到端数字是单次测量且受运行顺序影响，需交替运行取中位数。
   **注意不要用 M=1 做基准**，那会得出反向结论（见 §3.2）。
2. **完整训练验证**：≥500 步，确认 14.3% 的 dir1↔bf16 分歧不随训练放大。
3. **边界测试**：全零梯度、`amax=1e-40`、非连续输入、`bias=None`、block-swap 路径。
   （`copy.copy` 的线程安全与 `torch.compile fullgraph` 已验证，见 §7）
4. **backward 的 `grad_bias`**：当前实现保留了 `grad_output.sum(0)`，但 NF4 Linear 的
   bias 通常 `requires_grad=False`，需要确认。
5. **与 `torch.compile` 的兼容性**：`frexp`/`ldexp`/`clamp` 在编译图里的支持情况未验证。

## 10. 复现

```bash
# 合成矩阵模拟（单层扫描 + 28 层串联 + 边界检查）
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/sim_layer_scaling.py

# 真实权重四臂对比（fp32 / bf16 / fp16 / dir1）
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/dir1_chain.py --img 256
```
