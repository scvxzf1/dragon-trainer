状态：稳定（CPU + V100 实测；结论为机制判定，非 sm86 复现）
日期：2026-11-08
入口命令：见各节；探针在 `bench/krea2_nf4_fp16_grad/`
相关代码：`library/models/krea2_raw/quantize.py`、`library/models/krea2_raw/weights.py`、
`library/quantization/nf4.py`、`.venv/lib/python3.13/site-packages/bitsandbytes/autograd/_functions.py`
原始数据：`docs/findings/krea2_nf4_fp16_grad_underflow_20261108.json`；
探针输出在 `output/tests/krea2_nf4_fp16_grad/`（本地产物，不入库）

# Krea-2 NF4 fp16 梯度偏差（35.65%）机制溯源

## 结论

**机制判定（已闭环）**：stage6 的 35.65% 是 fp16 指数范围下溢，不是归约精度（候选①），
也不是小梯度分母放大（候选②）。

**精度判定（2026-11-08 追加，推翻 stage6 的否决理由）**：以 fp32 为唯一裁判，
**fp16 compute + loss scaling（k=14）在已测的所有维度上优于 bf16**——

| 维度 | bf16 | fp16 (k=14) |
| --- | ---: | ---: |
| 单步 LoRA 梯度 vs fp32（256²） | 40.66% | **4.58%** |
| 单步 cos vs fp32 | 0.9250 | **0.99897** |
| 10 步 loss 轨迹平均偏差 vs fp32 | 4.5% | **0.8%** |
| 10 步末步 LoRA 参数 rel-L2 | 17.55% | **8.56%** |

stage6 当年拿「fp16 vs bf16」互比得出 35.65% 并据此否决，但 **bf16 自己对 fp32 就有
40.66% 偏差**——那个判据把 bf16 的误差算成了 fp16 的账。

**为什么机制结论不受影响**：下溢仍然真实存在（§2、§3、§4.1 全部成立），
loss scaling 确实是必需项——不做 scaling 时 fp16 单步梯度是 114.80%，做了之后 4.58%。

**但窗口随分辨率关闭（决定性）**：k 的自收敛曲线（§5.2，两臂直接比较）是
256² `0.123%` → 512² `0.994%` → 768² `4.090%` → 1024² `5.900%`，**单调上升**。
而 k 的硬上限是 15（`loss×2^16 = 65536 > 65504`）。**768² 与 1024² 在 k=15 仍未饱和，
需要 k≥16~18，超出 fp16 标量能表达的范围——loss scaling 窗口在 512² 关闭。**

1024² 之所以能测（而不是停在「不可达」），是因为用 kv 显式展开绕过了 sm70 的
GQA kernel 回退（§5.3）。**这一步的价值是给出否定结论，不是让 fp16 可用。**

**决策：不放开生产契约。** 生产分辨率 1024² 不可行，且这是实测结论而非「未验证」。
22 处生产代码与 AGENTS.md 不变量本轮未改动。

## 1. 先确认参考口径（缺口 3）

stage6 只有 6 个标量，`docs/findings/krea2_3080_speed_stage6.json` 的
`same_nf4_weight_same_input` 块里**没有 ref_path / ref_dtype / reference 字段**，
整个 JSON 也没有 software 版本块。

全历史考古（`git log --all -S"input_gradient_rel_l2"`、`-S"same_nf4_weight_same_input"`、
`-S"dequantize_4bit"`、`--diff-filter=D` 全量删除清单、`git stash list`、4 个 worktree、
`/tmp`、兄弟 checkout 的 `output/tests/krea2_stage6/`）：

- **当年探针脚本从未入库**，唯一相关提交 `a97a3b74` 只加了 2 个文档 + 改 AGENTS/索引。
- **没有任何 fp32 参考臂**。md 表格分 BF16/FP16 两列，但「数学差异」只给一组数字，
  措辞是「同权重、同输入的 NF4 前后向」→ 最可能是**两臂互比（fp16 vs bf16）**，
  但没有任何历史证据能证明；可以确定的是**不存在 fp32 参考**。
- **「bf16 自身相对真值的偏差」历史上从未测过。**

本轮补上了：用 fp32 compute 臂做模型级参考（`--fp32-arm 1`），得到 bf16 与 fp16
各自对 fp32 的偏差（见 §3）。

## 2. 真实梯度量级（缺口 1，R2）

### 2.1 中间层 grad_output 全部落在 fp16 下溢区

`bench/krea2_nf4_fp16_grad/capture_real.py --img 256 --tag img256`（V100 #1，
自包含 NF4 v2 权重，grad-ckpt，fp32 参考臂），逐层 hook 真实 `grad_output`：

| 层 | dY rms | dY max | <6.1e-5 | <6e-8 | ==0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `first`（patch embed） | 4.51e-06 | 5.03e-04 | 100.0% | 6.5% | 0.0% |
| `blocks.2.mlp.up` | 1.74e-07 | 6.16e-05 | 100.0% | 80.6% | 65.2% |
| `blocks.7.attn.wq` | 7.81e-08 | 8.23e-06 | 100.0% | 91.9% | 65.2% |
| `blocks.9.mlp.down` | 4.66e-08 | 2.87e-06 | 100.0% | 89.7% | 65.2% |
| `blocks.19.mlp.down` | 1.31e-08 | 1.33e-06 | 100.0% | 99.6% | 65.2% |
| `blocks.25.mlp.gate` | 1.95e-08 | 8.80e-06 | 100.0% | 98.7% | 65.2% |
| `blocks.27.attn.wq` | 6.54e-09 | 7.97e-07 | 100.0% | 99.8% | 66.7% |
| `txtfusion.refiner_blocks.1.attn.wq` | 2.94e-08 | 4.28e-06 | 100.0% | 99.2% | 97.9% |
| `tproj.1`（time proj） | 2.09e-04 | 5.63e-03 | 44.9% | 0.1% | 0.0% |

（`==0` 是 fp32 参考臂自己的稀疏度，来自 text padding 与门控结构，不是 fp16 造成的。）

**结论**：除紧邻输入的 `first` / `tproj.1` 外，所有 264 个 Linear 的 `grad_output`
**没有一个元素落在 fp16 正规区**。fp16 在这个量级上每元素只剩 0~3 位有效位。

### 2.2 分辨率的真实影响：1/N 不是主因

MSE 是全元素均值，`dL/dv ∝ 1/N`，理论上分辨率翻倍 dY 应降 4×。但实测不支持这个强度：

| 分辨率 | 抽样层数 | 参考臂 | 中间层 dY rms | fp16 臂 latent-grad |
| --- | ---: | --- | ---: | --- |
| 256² | 24 | fp32 | 6.5e-9 ~ 1.7e-7 | L2 4.40e-3 / max 2.65e-4 |
| 512² | 8 | bf16 | 5.7e-9 ~ 2.0e-8 | L2 2.50e-4 / max 2.40e-5 |

512² 的 fp16 臂 latent grad **100% 低于 6.1e-5、69.3% 被冲成精确 0**。

> **初版勘误**：初版据此写「分辨率越高 fp16 越糟」。但两次抽样层不同、参考臂不同
> （512² 用 bf16 臂），**这个比较不可靠**。决定性证据是 §5.2 的 k 饱和曲线——
> 512² 与 768² 都在 **k=14** 就饱和，与 256² 相同，说明所需 scale 不随分辨率上升，
> **`1/N` 不是 dY 量级的主因**（残差链与 AdaLN 门控才是）。

### 2.3 模型级：bf16 参考 0.1%，fp16 差两个数量级

`--fp32-arm 1`（256²，参考臂 = fp32）：

| 臂 | output rel-L2 | output cos | latent-grad rel-L2 | latent-grad cos | 范数比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 vs fp32 | 0.93% | 0.999957 | **102.5%** | 0.6458 | 1.330 |
| fp16 vs fp32 | 0.13% | 0.999999 | **112.2%** | −0.1093 | 0.412 |
| fp16 vs bf16 | 0.95% | 0.999955 | 107.6% | −0.0987 | 0.310 |

注意 **bf16 臂的 latent-grad 对 fp32 也是 102%**——这不是 bf16 有问题，而是
`x_t` 的 latent 梯度经 28 个 residual block 反向累积，误差被放大到该指标失去分辨力。
**单步 latent-grad 不能当门槛**（见 §5）。逐层与 LoRA 参数梯度才是有效口径。

## 3. 机制定位：真实层上的量级指纹（R1 的前置）

`bench/krea2_nf4_fp16_grad/layer_repro.py`：取真实 `blocks.0.attn.wq`
（6144×6144，NF4 反量化权重 rms 2.58e-2），dY = N(0,1)×rms 扫 17 档，
V100 真 tensor core fp16 GEMM（fp32 累加 + fp16 输出舍入）：

| dY rms | <6.1e-5 | fp16 rel-L2 | fp16 cos | bf16 rel-L2 |
| ---: | ---: | ---: | ---: | ---: |
| 1.00e-08 | 100% | 1.0022 | 0.0501 | 1.66e-03 |
| 3.16e-08 | 100% | 0.6532 | 0.8344 | 1.65e-03 |
| **5.62e-08** | 100% | **0.3710** | **0.9373** | 1.66e-03 |
| 1.00e-07 | 100% | 0.2092 | 0.9789 | 1.65e-03 |
| 1.78e-07 | 100% | 0.1178 | 0.9931 | 1.63e-03 |
| 5.62e-07 | 100% | 0.0370 | 0.9993 | 1.63e-03 |
| 1.00e-06 | 100% | 0.0207 | 0.9998 | 1.66e-03 |
| 1.00e-05 | 100% | 0.0021 | 1.0000 | 1.62e-03 |
| 1.00e-04 | 45.8% | 0.0003 | 1.0000 | 1.65e-03 |

**`rms=5.62e-8` 复现出 rel `0.3710` / cos `0.9373`，目标 `0.3565` / `0.9434`。**
真实中间层 dY rms（256²: 1.3e-8~1.7e-7；512²: 5.7e-9~2.0e-8）正好覆盖并越过该点，
与 stage6 的 35.65% 完全自洽。

**同表最右列是关键的反证**：同一个 dY、同一个权重，bf16 操作数的误差
**恒定在 1.6e-3，与量级无关**（bf16 指数范围与 fp32 同宽）。
所以「35% 里有多少是 bf16 自身偏差」= **约 0.16%，占比 0.5%**。

### 候选②（小梯度分母放大）已被排除

在同一批极小 dY 上，bf16 的 rel-L2 恒为 `1.0e-3 ~ 1.7e-3`。如果 35% 来自
「dX 本身很小所以相对误差被放大」，bf16 会被同样放大。它没有。
分母是同一个分母，差异只在 dtype 的指数范围。

### 候选①（归约精度）是二阶项（R3）

`capture_real.py` 的 `reduction_scan`：真实 fp16 张量 GEMM，切
`torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction` 两档：

| 层 | dY rms | red=on | red=off | on-vs-off | bf16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `blocks.19.mlp.down` | 1.31e-08 | 0.4506 | 0.4506 | 0.0000 | 1.28e-03 |
| `blocks.27.attn.wq` | 6.54e-09 | 0.6497 | 0.6497 | 0.0000 | 9.40e-04 |
| `blocks.25.mlp.gate` | 1.95e-08 | 0.2838 | 0.2265 | **0.2084** | 1.24e-03 |
| `blocks.22.mlp.up` | 3.81e-08 | 0.1240 | 0.1077 | 0.0751 | 1.07e-03 |
| `blocks.2.mlp.up` | 1.74e-07 | 0.0456 | 0.0426 | 0.0202 | 1.31e-03 |
| `tproj.1` | 2.09e-04 | 0.0011 | 0.0002 | 0.0010 | 6.33e-04 |

24 层抽样里 16 层 on/off **逐位相同**；最大差异 `0.2084`，且只在本来就已经
`0.28` 的层上把误差降 20%。**没有任何一层被拉回 bf16 的 1e-3 量级。**
V100 上该开关默认 `True`。

## 4. loss-scale 扫描（R1）

### 4.1 逐层：严格单调，每个 k 精确减半

`capture_real.py --img 256 --tag ls --fp32-arm 0 --replay-layers 6 --loss-scale-k 4,8,12,16`，
真实 dY × 2^k。该次跑用 bf16 臂的 dY 做输入（`--fp32-arm 0`），所以本表 `first` 的
rms 是 `6.23e-06`，与 §2.1（fp32 臂）的 `4.51e-06` 是同量级的不同参考臂，不影响结论：

| 层 | dY rms | k=0 | k=1 | k=2 | k=3 | k=4 | k=6 | k=8 | k=10 | k=12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blocks.19.mlp.gate` | 6.09e-08 | 0.1090 | 0.0607 | 0.0328 | 0.0172 | 0.0089 | 0.0022 | 0.0005 | 0.0001 | 0.0000 |
| `blocks.26.attn.wk` | 4.27e-08 | 0.1770 | 0.1020 | 0.0539 | 0.0272 | 0.0136 | 0.0035 | 0.0008 | 0.0002 | 0.0000 |
| `blocks.13.attn.wq` | 7.47e-08 | 0.0991 | 0.0512 | 0.0260 | 0.0131 | 0.0066 | 0.0016 | 0.0003 | 0.0000 | 0.0000 |
| `blocks.6.attn.wo` | 8.33e-08 | 0.0640 | 0.0323 | 0.0162 | 0.0082 | 0.0041 | 0.0010 | 0.0002 | 0.0000 | 0.0000 |
| `tproj.1` | 2.37e-04 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `first` | 6.23e-06 | 0.0006 | 0.0003 | 0.0002 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**单调性成立，且是「每 +1 个 k，误差 ÷2」的精确关系**——fp16 subnormal 每多一位
指数就多一位尾数位。这是 subnormal 下溢独有的签名：归约误差、输出舍入误差都不会
对 k 呈 2^-k 线性。

纯 CPU 合成探针（`synth_underflow.py`，dY 1e-7 档）给出同一形状：
k=0 `0.1721` → k=4 `0.0109` → k=8 `0.0018`（≈bf16 的 0.00234）→ k=10 起饱和。

### 4.2 模型级 loss scaling：原论证作废，改用 fp32 裁判重做

> **勘误（2026-11-08）**：本节初版用「latent-grad 停在 ~80%」判定 loss scaling 无效，
> 但 §2.3 已证明 latent-grad 指标本身底噪就有 80%~102%（bf16 臂对 fp32 也是 102.5%）。
> 拿有底噪的指标判死是逻辑错位，原论证作废。

**k 的硬上限是 15，不是"顶层溢出"**。实测（`/tmp/probe_nan_source.py`）：
k=15 时梯度有限，k=16 时 `x_t.grad` 全部为 NaN 且 **inf=0**；把 loss 提到 fp32 也无效。
逐层算 `max|dY| × 2^16`，24 层**没有一层超过 65504**（最大的 `tproj.1` 只有 369）。
真正原因是 **fp16 标量上限 65504**：`loss × 2^16 = 65536` 作为 autograd 种子进 fp16 图后
变 inf，与有限值相减产生 NaN。**k ≤ 15。**

**用 LoRA 梯度 + fp32 裁判重做**（256²，`--fp32-reference 1`）：

| 臂 | 单步 LoRA grad vs fp32 | cos | 10 步最差 grad | 10 步最差 cos | 10 步最差 loss 偏差 | 10 步末步 LoRA 参数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 40.66% | 0.9250 | 56.00% | 0.8424 | 18.70% | 17.55% |
| fp16 (k=0) | 114.80% | 0.3668 | — | — | — | — |
| **fp16 (k=14)** | **4.58%** | **0.99897** | 35.42% | 0.9356 | **1.78%** | **8.56%** |

**多步下 fp16 会漂移，但 bf16 漂得更远**：fp16 从 Step 0 的 4.58% 涨到 Step 8 的 35.42%，
bf16 则始终在 25%~56% 之间波动。漂移来自 AdamW 迭代放大，不是算子错误——
证据是 loss 轨迹：fp16 逐 step 偏差 0.21%/0.40%/0.27%/1.78%/…（平均 0.8%），
bf16 是 2.02%/18.70%/4.95%/6.94%/…（平均 4.5%）。**fp16 的 loss 几乎贴着 fp32 走。**

## 5. 多步梯度 / loss 轨迹门槛（stage6 自述的后续要求）

stage6 文档要求「建立多步梯度 cosine/loss 轨迹门槛」。本轮建了可执行版本
`bench/krea2_nf4_fp16_grad/trajectory_gate.py`，阈值沿用
`library/training/adaptive_runtime/precision.py::Tolerances`
（grad rel-L2 ≤ 0.02、cos ≥ 0.999、loss 相对偏差 ≤ 1%），指标落在 **LoRA 参数梯度**
（不是 latent 梯度，理由见 §2.3）。

`--img 256 --steps 10 --lora-dim 8`，两臂同一份 LoRA 初始化、同一批激活、AdamW lr=2e-3：

| step | loss bf16 | loss fp16 | loss 相对偏差 | LoRA grad rel-L2 | cos | 范数比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.027954 | 0.027374 | 2.07% | 112.2% | 0.3518 | 0.971 |
| 1 | 0.025513 | 0.027344 | 7.18% | 96.1% | 0.2781 | 0.263 |
| 2 | 0.013489 | 0.027069 | 100.7% | 216.8% | 0.1668 | 2.097 |
| 3 | 0.012878 | 0.026199 | 103.4% | 157.6% | −0.3162 | 0.942 |
| 5 | 0.006561 | 0.024033 | 266.3% | 357.2% | 0.2783 | 3.719 |
| 8 | 0.003708 | 0.024673 | 565.4% | 604.6% | −0.0057 | 5.957 |
| 9 | 0.003738 | 0.026306 | 603.7% | 562.7% | 0.2813 | 5.826 |

- bf16 臂：0.0280 → 0.0037，正常收敛。
- fp16 臂：0.0274 → 0.0263，**几乎不动，末段还回升**（步长方向错误）。
- 末步 LoRA 参数 rel-L2 **29.3%**。

**门槛判定：`GATE: FAIL`**。grad rel-L2（限 2%）最差 604.6%、cos（限 0.999）最差
−0.316、loss 相对偏差（限 1%）最差 603.7%。只有「轨迹不塌」一项 PASS（两臂都在动）。

### 5.1 加上 loss scaling 与 fp32 裁判之后

`trajectory_gate.py` 随后支持了 `--loss-scale-k`（GradScaler 语义：backward 前放大、
step 前 unscale、比较用 unscale 后梯度）与 `--fp32-reference`（fp32 compute 臂当裁判），
并把 MSE 改成在 fp32 里算（`mse_loss(velocity.float(), target.float())`）——
因为留在 fp16 时 backward 内部先算 `2*(v-t)/N ≈ 2e-5`（fp16 subnormal，只剩 0~2 位），
再乘 scale 也救不回已丢的位。

结果见 §4.2 表。要点：

- **Step 0 单步就是 4.58%，不是"从微小放大"**（k=0 是 114.80%），说明分歧发生在单步算子内。
- 多步漂移到 35.42%，但 bf16 同期是 56.00%，且 bf16 的 loss 轨迹更差。
- **2% 绝对门槛两臂都不过**，说明该门槛不适合多步轨迹；应改用
  「单步梯度 + loss 轨迹 + 末步参数」三件套。

### 5.2 四个分辨率下的 k 自收敛曲线（决定性结果）

自收敛判据：若 subnormal 已被彻底消灭且未触发上溢，相邻两档 k 应给出**同一条梯度**
（直接比较 `_rel_l2(grad_k14, grad_k15)`，不是「两臂对同一裁判的偏差之差」）。

> **判据勘误**：初版把「k14 vs bf16」与「k15 vs bf16」两个数相减当自收敛指标
> （0.001pp / 0.055pp）。**这是错的**——两臂都离 bf16 约 40%，相减接近 0 只说明它们
> 对 bf16 的偏差量级相同，不说明彼此相同。改成两臂直接比较后，结论完全反转。

| 分辨率 | k14↔k15 梯度 rel-L2 | cos | 末步 LoRA 参数 rel | 判定 |
| --- | ---: | ---: | ---: | --- |
| 256² | **0.123%** | 0.999999 | 0.029% | 饱和 |
| 512² | **0.994%** | 0.999951 | 0.071% | 基本饱和 |
| 768² | **4.090%** | 0.999164 | 0.239% | **未饱和** |
| 1024² | **5.900%** | 0.998265 | 0.677% | **未饱和** |

**所需 k 随分辨率单调上升**：0.12% → 0.99% → 4.09% → 5.90%，每翻倍约 ×4~8，
与 `dY ∝ 1/N` 的直觉一致。

**但 k 的硬上限是 15**（§4.2：`loss×2^16 = 65536 > fp16 max 65504` → 种子 inf → 全 NaN）。
768² 与 1024² 在 k=15 仍未饱和，意味着它们需要 k≥16~18——**超出 fp16 标量能表达的
范围，loss scaling 窗口在 512² 关闭**。

（此前 §5.2 初版写「512²/768² 都在 k=14 饱和，所需 k 不随分辨率上升」，
并据此外推 1024² 也只需 k≈14-15。该结论建立在错误判据上，**予以撤回**。）

### 5.3 1024² 怎么跑起来的：GQA 展开绕过

1024² 的 img token 是 4096。原本三条路都 OOM：全驻留 / `--blocks-to-swap 20` /
`--blocks-to-swap 26`（上限），因为 `blocks_to_swap` 只搬 block 权重、搬不动 attention
激活。根因是 GQA 触发的 kernel 回退：

| 配置（fp16，4096 tokens，48 heads，D=128） | 峰值 | backend |
| --- | ---: | --- |
| MHA 48q / 48kv，fp16 | 240 MiB | EFFICIENT |
| MHA 48q / 48kv，fp32 | 384 MiB | EFFICIENT |
| MHA 48q / 48kv，bf16 | 7448 MiB | MATH（bf16 无可用内核） |
| **GQA 48q / 12kv（生产形状），fp16** | **7424 MiB** | **MATH** |
| **GQA 48q / 12kv，bf16** | **7424 MiB** | **MATH** |

Krea-2 用 GQA（`kvheads=12`），sm70 的 memory-efficient 内核在 `q_heads != kv_heads` 时
直接回退 MATH，**连 fp16 也一样**。

**绕过办法**：进 SDPA 前把 kv 显式展开成与 q 同 head 数
（`k.repeat_interleave(4, dim=1)`，多花 96 MiB），数学上与 GQA 等价，可走回 EFFICIENT：

| 指标 | `enable_gqa=True` | 展开后 MHA |
| --- | ---: | ---: |
| 峰值显存 | 7424 MiB | **224 MiB** |
| 前向 loss | 2.5205e+10 | 2.5205e+10 |
| dq / dk / dv rel-L2 | — | 3.99e-4 / 2.71e-4 / 2.89e-4 |
| dk 形状 | [1,12,4096,128] | [1,12,4096,128] |

~3e-4 是 fp16 下不同 kernel 的正常差异；反向梯度会正确 sum 回 12 个 kv head。
探针用 `--expand-gqa` 启用这个 patch（monkey-patch `F.scaled_dot_product_attention`，
**不改生产 `attention_backend.py`**）。

**这一步的价值不是「让 fp16 可用」，而是「把 1024² 从不可验证变成可验证」**——
而验证结果是 1024² 不饱和（§5.2），窗口关闭。若没有这个 patch，我们只能停在
「未验证」，无法给出否定结论。

### 5.4 后续：逐层缩放（dir1）方案与它的精度上限

§5.2 的结论是「全局 loss scaling 窗口在 512² 关闭」。一个自然的追问是：能不能把
缩放从「全局 loss 标量」下沉到「每个 Linear 的 backward」？答案是**能，但它只换速度，
不换精度**。完整提案见 [krea2_nf4_fp16_layer_scaling](../proposal/krea2_nf4_fp16_layer_scaling.md)。

机制：外层 `compute_dtype` 保持 bf16（8 位指数，负责把 1e-9 的梯度跨层送达），
只在 NF4 Linear 内部用 `frexp` 取 amax 的二进制指数、`ldexp` 平移后跑 fp16 GEMM，
算完再平移回来。

真实 NF4 权重、256²、fp32 裁判：

| 臂 | LoRA 梯度 rel-L2 vs fp32 | cos | 单步耗时 |
| --- | ---: | ---: | ---: |
| bf16（生产） | 40.614% | 0.92764 | 9.0s |
| 裸 fp16 | **114.741%** | 0.36736 | 2.4s |
| **dir1** | **40.974%** | 0.92553 | **3.1s** |

**dir1 把裸 fp16 的 114.7% 救回 40.97%，端到端速度 3.33×，但没有超越 bf16（差 0.36pp）。**

算子级基准（V100，`cuda.Event` 100 次中位数，K=6144）给出更关键的读数：**bf16 比 fp16
慢 8~9×**（sm70 无原生 bf16 tensor core 的直接证据），dir1 在付掉 `frexp`/`ldexp` 开销后
**净收益 5.0~6.6×**：

| M（token 数） | bf16/fp16 | bf16/dir1 |
| ---: | ---: | ---: |
| 1（带宽限制，勿用） | 0.99× | **0.51×（反而更慢）** |
| 256 | 9.2~9.9× | 5.05~5.48× |
| 1024 | 7.9~8.8× | 6.11~6.37× |
| 4096 | 8.0~8.9× | 6.38~6.57× |
| 16384 | 8.1~8.9× | 6.42~6.56× |

**M=1 会得出反向结论**：单个梯度向量乘权重时 GEMM 受显存带宽限制（主要开销是读 72 MB
权重），bf16 与 fp16 同速，dir1 的额外 kernel 让它慢一倍。基准必须用真实 token 数。

原因：dir1 的**输出**仍是 bf16，激活流的精度上限是 bf16 的 7 位尾数。所以端到端 LoRA
梯度与 bf16 同量级——dir1 的价值主要是在 sm70/sm75 这类**没有原生 bf16 tensor core**
的卡上拿回矩阵乘吞吐。

**但权重侧可以反超。** bnb 的 dequant 内核在 FP32 寄存器里算，只在写回时按
`quant_state.dtype` 转换。把 `qs.dtype` 临时设为 fp16 可以直接写出 10 位尾数的权重，
省掉一次 72 MB 读 + 72 MB 写（0.581 → 0.308 ms，省 0.272 ms），并且：

| dequant 输出 vs fp32 真值 | 单层 rel-L2 | 28 层串联末层 |
| --- | ---: | ---: |
| bfloat16 | 0.16418% | 1.316% |
| **float16** | **0.02061%** | **0.988%** |

即权重精度准 8 倍，28 层累积后反超 bf16 **0.33pp**。端到端 LoRA 梯度之所以看不出这个
改善，是因为激活流的 bf16 截断是主导项。

端到端交替中位数（3 轮，256²）：bf16 8.911s / dir1 2.679s，**3.33×**（极差 0.03~0.05s）。

合成矩阵模拟（`sim_layer_scaling.py`）给出同样的结论：单层 dY 从 `1e-10` 扫到 `1e-2`，
dir1 恒定 `0.235%~0.240%`，bf16 恒定 `0.232%~0.238%`，裸 fp16 在 `1e-8` 以下崩到 100%。

## 6. 决策（R4）

**结论：不放开 fp16 NF4。理由从「无法验证」升级为「已实测不可行」。**

| 选项 | 结论 | 理由 |
| --- | --- | --- |
| 维持现状（本轮执行） | 保留 `COMPUTE_DTYPE=bfloat16` 与 22 处强制检查 | 1024²（生产分辨率）在 k=15 仍未饱和，需要 k≥16 但会撞 fp16 标量上限 65504；窗口在 512² 关闭 |
| 放开 fp16（仅 ≤512² 训练） | **技术上可行，但收益不明** | 256²/512² 自收敛 0.12%/0.99%，且 fp16 对 fp32 优于 bf16（§4.2）。但 512² 不是生产默认分辨率，需单独评估是否有场景值得引入 loss scaling 全链路复杂度 |
| 收紧 | 不采纳 | 22 处拒绝已足够；本轮新增 2 条回归断言即止 |
| backward-only 宽 dtype | 仍否决，理由修正 | 原写「会失去 NF4 显存收益」是**错的**：bnb 训练无融合 kernel，逐层 `dequantize_4bit` 临时解压单层（6144² bf16 = 72MiB）后立即释放，自定义 `autograd.Function` 不改权重存储、不影响 block swap 与 LoRA。真正否决理由是收益：V100 backward 实测 `9.30s→3.26s`（2.85×）、stage6 3080 `27.07→10.02ms`（2.70×），砍掉反向只剩 forward 的 1.124ms/35.6ms ≈ 3.2%（单层探针口径），端到端更低 |

**如果将来要重新评估**，前置条件是：`dY` 量级不再随分辨率下降（即模型结构或 loss 归一化
方式改变），或 fp16 之外的格式（如 fp16 存储 + fp32 累加的自定义 kernel）能同时提供
宽指数范围与 tensor core 吞吐。

## 7. 缺口与边界

**已闭合**
- 参考口径（缺口 3）：历史无 fp32 臂、探针未入库；本轮用 fp32 compute 臂重建。
- 真实梯度量级（缺口 1）：§2.1，逐层 dY rms 与 subnormal 占比。
- 探针入库（缺口 2）：`bench/krea2_nf4_fp16_grad/` 四个脚本。
- 机制判定：下溢，非归约、非分母放大。
- **精度判定与分辨率窗口**：fp16(k14) 在 ≤512² 优于 bf16；768²/1024² 需要 k≥16~18，
  超出 fp16 标量上限，窗口关闭（§5.2、§5.3）。

**未闭合（按风险排序）**
1. **长训练未测**：最多 10 步。35% 的多步漂移在 500+ 步后是否累积成发散，未知。
2. **真实数据集未测**：合成图案 + 单 prompt + 固定 σ=0.5。
3. **AdaLN 门控衰减未做消融**：把 dY 量级归因到门控是合理推测，但没测过"关掉 gate 后 dY 是否上升"。
4. **≤512² 场景的实际收益未评估**：技术上可行，但引入 GradScaler 全链路是否值得，未做工程权衡。
5. 512²/768²/1024² 的 k14↔k15 比较缺 fp32 裁判（两臂互比，只判自收敛不判绝对精度）。

**边界**
- **本机无 bf16 原生卡**（2×V100-16G sm70 + GTX1050 sm61）。V100 的 bf16 matmul 走
  PyTorch fp32 上采样仿真（操作数 bf16 舍入 + fp32 累加），语义等价 sm86 bf16 tensor core，
  但**不是** sm86 cuBLAS 的逐 bit 复现。fp16 在 V100 上是真 tensor core（fp32 累加 + fp16
  输出舍入），这部分可外推。
- 本报告的 fp32 臂在 V100 上是真 fp32 GEMM（无 TF32），可当参考。
- 1024² 的 k 曲线是在 `--expand-gqa`（EFFICIENT kernel）路径下测的，与 256²~768² 的
  MATH 路径不同；自收敛判据只比较同路径两臂，不受此影响，但**跨分辨率的绝对值不宜直接横向对比**。
- 图像是合成图案（`probe_train.make_test_image`）。影响激活分布细节，不影响
  「dY 量级 → fp16 subnormal」这条链——量级由残差链与门控主导。
- 未做 3080/PG199 复测。

## 8. 复现命令

```bash
# CPU 合成探针（无 GPU）
CUDA_VISIBLE_DEVICES='' .venv/bin/python bench/krea2_nf4_fp16_grad/synth_underflow.py --repeat-check

# 真实权重逐层量级 + 归约两档（占一张 V100，约 3 分钟）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python bench/krea2_nf4_fp16_grad/capture_real.py --img 256 --tag img256

# 真实层指纹定位
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/layer_repro.py

# 单步 fp32 裁判（256²，三臂）
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/trajectory_gate.py \
  --img 256 --steps 1 --lora-dim 8 --loss-scale-k 0,14 --fp32-reference 1 --tag fp32ref

# 10 步 fp32 裁判（256²，含 loss scaling）
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/trajectory_gate.py \
  --img 256 --steps 10 --lora-dim 8 --loss-scale-k 14 --fp32-reference 1 --tag fp32ref10

# k 饱和曲线（512² / 768²；768² 需 block swap）
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/trajectory_gate.py \
  --img 512 --steps 1 --lora-dim 8 --loss-scale-k 14,15 --tag img512
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/trajectory_gate.py \
  --img 768 --steps 1 --lora-dim 8 --loss-scale-k 14,15 --blocks-to-swap 26 --tag img768swap

# k 自收敛曲线（两臂直接比较；--expand-gqa 绕过 sm70 GQA 内核回退）
for IMG in 256 512 768 1024; do
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/trajectory_gate.py \
    --img $IMG --steps 1 --lora-dim 8 --loss-scale-k 14,15 \
    --expand-gqa --skip-bf16 --tag direct$IMG
done
```

## 附录：生产 dtype 契约清单（22 处，均未改动）

关键 4 处：

- `library/models/krea2_raw/quantize.py:46` `COMPUTE_DTYPE = torch.bfloat16`
- `library/models/krea2_raw/weights.py:117-123` 加载期 `dtype != bfloat16` 硬拒绝
- `library/anima/mixed_nf4/runtime.py:12,22-23` `restore_linear` 硬编码 + 加载期拒绝
- `library/quantization/nf4.py:1-27` `BF16AutocastLinear4bit` 的 BF16 autocast 输出契约

其余见 `library/quantization/{nf4,format,mixed_nf4}.py`、`library/anima/{weights,mixed_nf4/*}.py`、
`library/training/{model_loading,adaptive_runtime/native_precision,compat_matrix}.py`、
`library/training/cli_args.py:1120`、`web/services/config/preflight_nf4.py`、
`web/frontend-next/.../labels-options.js:303` 与 `help/resources.js:201`，
以及 `tests/test_shared_mixed_nf4.py:112`、`tests/test_mixed_nf4_preflight_risks.py:66`、
`tests/test_anima_mixed_nf4.py:51,104`。

本轮新增两条回归断言（`tests/test_krea2_nf4_self_contained.py`）：
`load_krea2_dit` 对 `float16`/`float32` 都抛 `NF4 量化要求 dtype=bfloat16`，
以及 `COMPUTE_DTYPE == torch.bfloat16`。
