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

**单步耗时 9.0s → 3.1s（2.9×）**，这是目标 A 的核心收益。

### 3.2 合成矩阵模拟（单层扫描 + 28 层串联）

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

## 4. 实现陷阱（三个都踩过）

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

## 5. 精度上限：dir1 不提精度

**这是必须写清的边界。** dir1 的输出仍是 bf16，所以整条链的精度上限是 bf16 的 7 位尾数，
不是 fp16 的 10 位。内层 fp16 的 10 位尾数只在 GEMM 累加过程中起作用（减少累加误差），
不能恢复输入的精度。

实测也印证：dir1 的 40.97% 与 bf16 的 40.61% 基本一致，甚至略差（多两次 dtype 转换）。

**如果目标是「比 bf16 更准」，dir1 做不到**——那需要外层换 fp32，代价是激活显存翻倍
且要放开 22 处契约。

## 6. 收益与代价

| 维度 | 结果 |
| --- | --- |
| V100/sm75 上 Linear 前后向 | ✅ 走真 tensor core，单步端到端 **9.0s → 3.1s（2.9×）** |
| 1024² 深层下溢 | ✅ 逐层归一化，彻底消除（模拟全量级恒定 0.24%） |
| 精度 | ⚠️ **与 bf16 持平（略差 0.36pp），不提升** |
| 22 处 fail-closed 契约 | ✅ **0 改动**（`compute_dtype` 保持 bf16） |
| GradScaler | ✅ **不需要**（外层 bf16 保动态范围，逐层缩放替代全局 k） |
| 改动面 | `library/quantization/nf4.py` 一个文件，约 60~90 行 |
| 运行开销 | 每层多一次 `abs().max()` 规约 + 两次 `ldexp`，约 1%~1.5% |
| sm80+ 显卡 | ⚠️ 收益消失——那些卡有原生 bf16 tensor core，bf16 已经够快 |

**注意最后一行**：dir1 只在**没有原生 bf16 tensor core**的卡上有价值（V100 / 2080Ti /
T4 等 sm70/sm75）。在 3080/3090/4090 上做这件事是纯亏损（多一层转换、没有提速）。

## 7. 与其它方案的比较

| 方案 | 精度 | 提速 | 契约改动 | GradScaler | 结论 |
| --- | --- | --- | --- | --- | --- |
| 维持 bf16（现状） | bf16 水平 | 无（sm70 上慢） | 0 | 不需要 | 当前生产 |
| 裸 fp16 + GradScaler | 崩溃（114.7%） | 有 | 22 处 | 需要 | 否决 |
| 全局 loss scaling | 窗口在 >512² 关闭 | 有 | 22 处 | 需要 | 否决 |
| 按分辨率分档（≤512² 用 fp16） | 可行 | 仅小图 | 22 处 | 需要 | 收益低 |
| **dir1（本提案）** | **bf16 水平** | **2.9×** | **0** | **不需要** | **推荐（限 sm70/sm75）** |
| 外层 fp32 + 内层 fp16 | 最高 | 有 | 22 处 | 不需要 | 显存翻倍，暂不采用 |

## 8. 落地前还缺什么

1. **基准测试**：当前 9.0s → 3.1s 是单次测量，含 grad-ckpt 重算与 attention，
   需要分离出纯 Linear GEMM 的加速比，并做多次取中位数。
2. **完整训练验证**：≥500 步，确认 14.3% 的 dir1↔bf16 分歧不随训练放大。
3. **边界测试**：全零梯度、`amax=1e-40`、非连续输入、`bias=None`、
   block-swap 路径、`torch.compile` 下的行为。
4. **backward 的 `grad_bias`**：当前实现保留了 `grad_output.sum(0)`，但 NF4 Linear 的
   bias 通常 `requires_grad=False`，需要确认。
5. **与 `torch.compile` 的兼容性**：`frexp`/`ldexp`/`clamp` 在编译图里的支持情况未验证。

## 9. 复现

```bash
# 合成矩阵模拟（单层扫描 + 28 层串联 + 边界检查）
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/sim_layer_scaling.py

# 真实权重四臂对比（fp32 / bf16 / fp16 / dir1）
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/dir1_chain.py --img 256
```
