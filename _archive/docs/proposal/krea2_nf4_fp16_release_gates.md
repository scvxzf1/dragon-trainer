状态：已否决 / 2026-11-09 实测关闭（保留作门槛记录）
日期：2026-11-08
适用版本：当前 dev（分支 `krea2-fp16-2026118`）
前置证据：[NF4 fp16 梯度偏差机制溯源](../../../docs/findings/krea2_nf4_fp16_grad_underflow_20261108.md)
相关代码：`library/models/krea2_raw/quantize.py`、`library/models/krea2_raw/weights.py`、
`library/quantization/nf4.py`、`library/runtime/accelerator.py`
探针入口：`bench/krea2_nf4_fp16_grad/`

# Krea-2 NF4 + fp16 compute 的放开门槛与验证计划

## 1. 这份提案解决什么

stage6（2026-08-09）以「同权重同输入下 fp16 的输入梯度 rel-L2 = 35.65%」为由，
把 NF4 `compute_dtype` 永久锁在 bf16，并留下「后续若研究 mixed precision，必须建立
多步梯度 cosine/loss 轨迹门槛」的要求。

2026-11-08 的溯源（[findings](../../../docs/findings/krea2_nf4_fp16_grad_underflow_20261108.md)）证明：

- 35.65% 是 **fp16 指数范围下溢**（真实中间层 dY rms `6.5e-9 ~ 1.7e-7`，
  100% 低于 fp16 最小正规数 6.1e-5），不是 NF4 量化的固有缺陷。
- stage6 的判据是 **fp16 vs bf16 互比**，而 **bf16 自己对 fp32 就有 40.66% 偏差**
  （Step 0，权重完全相同）。那个判据把 bf16 的误差算在了 fp16 头上。
- 补上 **fp32 loss + loss scaling（k=14）** 后，fp16 在已测维度**全面优于 bf16**：

| 维度（256²，fp32 裁判） | bf16 | fp16 (k=14) |
|---|---:|---:|
| 单步 LoRA 梯度 rel-L2 | 40.66% | **4.58%** |
| 单步 cos | 0.9250 | **0.99897** |
| 10 步 loss 轨迹平均偏差 | 4.5% | **0.8%** |
| 10 步末步 LoRA 参数 rel-L2 | 17.55% | **8.56%** |

- 512² 与 768² 实测 **k=14 即饱和**（k14↔k15 差 0.001pp / 0.055pp），
  所需 scale 不随分辨率上升。

**但分辨率窗口把它挡回来了。** 2026-11-09 补测了 k 的自收敛曲线（两臂直接比较）：
256² `0.123%` → 512² `0.994%` → 768² `4.090%` → 1024² `5.900%`，**单调上升**。
而 k 的硬上限是 15，768²/1024² 需要 k≥16~18 → 溢出。

**最终结论：loss scaling 窗口在 512² 关闭，生产分辨率 1024² 不可行。**
本提案据此否决，`COMPUTE_DTYPE=bfloat16` 与 22 处 fail-closed 检查保持不变。
下文保留三个前置条件作为记录——它们的设计仍然正确，只是前置 1 已经用实测
给出了否定答案，无需再等硬件。

## 2. 硬性前置条件

### 前置 1：1024² 数值验证 —— **已执行，结论为不通过**

**2026-11-09 更新：这一项已经跑通并给出了否定结论，不再是阻塞项。**

原本以为 1024² 在 16GB V100 上硬件不可达（attention 分数矩阵 `4096² × 48 heads × 2B ≈ 1.6GiB`，
backward 需 3.80GiB）。真实根因是 **GQA 触发 kernel 回退**：Krea-2 用 `kvheads=12`，
sm70 的 memory-efficient 内核在 `q_heads != kv_heads` 时直接回退 `MATH`，**连 fp16 也一样**。

**绕过办法**（探针 `--expand-gqa`，monkey-patch `F.scaled_dot_product_attention`，
不改生产代码）：进 SDPA 前把 kv `repeat_interleave` 展开成与 q 同 head 数，数学等价，
显存从 7424MiB 降到 224MiB，反向梯度正确 sum 回 12 个 kv head（rel-L2 ~3e-4）。

跑通后得到的 k 自收敛曲线（两臂直接比较 `_rel_l2(grad_k14, grad_k15)`）：

| 分辨率 | k14↔k15 梯度 rel-L2 | 判定 |
| --- | ---: | --- |
| 256² | 0.123% | 饱和 |
| 512² | 0.994% | 基本饱和 |
| 768² | 4.090% | 未饱和 |
| **1024²** | **5.900%** | **未饱和** |

**k 的硬上限是 15**（`loss×2^16 = 65536 > fp16 max 65504` → 种子 inf → 全 NaN）。
768²/1024² 在 k=15 仍未饱和，需要 k≥16~18，**超出 fp16 标量能表达的范围**。

**结论：loss scaling 窗口在 512² 关闭，生产分辨率 1024² 不可行。本项判定为不通过，
不进入后续前置条件。**

### 前置 2：长步漂移收敛性（真实数据）—— 未执行，因前置 1 已否决

**为什么**：单步/10 步的高余弦相似度不等于训练收敛。实测 fp16(k=14) 对 fp32 的
LoRA 梯度从 Step 0 的 4.58% 涨到 Step 8 的 35.42%——这是 AdamW 迭代放大，
不是算子错误（同期 bf16 是 56.00%，且 bf16 的 loss 轨迹更差）。但 500+ 步后
是否累积成发散，未知。

**满足方式**：

- 真实数据集（非合成图案）、真实 σ 采样、≥500 steps 的 LoRA 训练；
- 双臂（bf16 / fp16+k14）同初始化，逐 50 步记录：loss、LoRA 梯度 cosine、
  生成样本（固定 seed / 固定 prompt）的 FID 或人工对照；
- 判据：fp16 臂的 loss 曲线不劣于 bf16，且末段生成质量无可见退化。

**门槛口径说明**：不要用「梯度 rel-L2 ≤ 2%」当唯一判据——该门槛对 bf16 也不成立
（bf16 对 fp32 是 40.66%）。应改用「单步梯度 + loss 轨迹 + 末步参数」三件套。

### 前置 3：AdaLN 门控衰减消融 —— 未执行，因前置 1 已否决

**为什么**：findings 把 dY 量级归因到「MSE 的 1/N 均值缩放 + AdaLN 门控衰减」，
但**门控衰减这一项从未做消融**。目前只有间接证据：`first` 层 dY rms `4.5e-6` 与
`blocks.*` 的 `1e-8` 相差约 2 个数量级，且 §2.2 已确认 `1/N` 不是主因。

**满足方式**：关掉（或固定）block 的 AdaLN gate，重测逐层 dY rms 与 fp16 下溢占比；
若 dY 显著上升且所需 k 下降，则归因成立并可写入文档。

**为什么列为硬条件**：如果门控不是主因，说明还有别的算子在压低 dY——那可能在
别的模型族/分辨率下表现不同，前置 1 的结论就不能外推。

## 3. 实施前必须一并落地的工程项

即使三个前置条件全部满足，放开也不能只改一个常量：

| 项 | 要求 | 依据 |
|---|---|---|
| loss 计算 | **必须在 fp32**：`mse_loss(pred.float(), target.float())` | 留在 fp16 时 backward 内部先算 `2*(v-t)/N ≈ 2e-5`（fp16 subnormal，只剩 0~2 位），再乘 scale 也救不回 |
| loss scaling | 接 GradScaler，**不能硬编码 k=14** | 固定 k 只在已测分辨率成立；k 的硬上限是 15（`loss×2^16 = 65536 > 65504` → 种子 inf → NaN，实测全 NaN 无 inf） |
| unscale 顺序 | `backward → unscale → clip → step` | 先 clip 后 unscale 会让裁剪阈值失真（`bench/fp16_precision/` P4/P8 已有反例） |
| 溢出处理 | 溢出步跳过、scale 退档、state_dict 往返 | 复用 `bench/fp16_precision/` 的 P1–P9 与 H1/H2/H8 |
| 配置面 | 走既有 `compute_dtype` 契约，**不新增全局 FP16 NF4 开关** | AGENTS.md「Krea-2 训练与性能边界」不变量 |
| 22 处 fail-closed | 逐条复核哪些需要放开、哪些保留 | findings 附录清单 |

## 4. 明确不做的事

- **不做 backward-only 宽 dtype**（前向 fp16 / 反向 bf16）。原 findings 写的
  「会失去 NF4 显存收益」是错的——bnb 训练时无融合 kernel，逐层 `dequantize_4bit`
  临时解压单层（6144² bf16 = 72MiB）后立即释放，自定义 `autograd.Function` 不改权重
  存储、不影响 block swap 与 LoRA。**真正否决理由是收益太薄**：V100 backward 实测
  `9.30s→3.26s`（2.85×）、stage6 3080 `27.07→10.02ms`（2.70×），砍掉反向只剩 forward
  的 1.124ms/35.6ms ≈ 3.2%（单层探针口径），端到端更低。
- **不把 fp16 当默认**。即使放开，也应是显式 opt-in，默认继续 bf16。
- **不在没有 1024² 证据的情况下改生产契约**。

## 5. 当前状态

- 生产契约：**未改动**（`COMPUTE_DTYPE=bfloat16` 与 22 处检查原样）。
- 已交付：四个探针 + 两条回归断言 + findings 文档与 JSON。
- 阻塞项：前置 1（本机硬件不可达）、前置 2（未跑）、前置 3（未做）。

## 6. 复现入口

```bash
# 单步 fp32 裁判
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/trajectory_gate.py \
  --img 256 --steps 1 --lora-dim 8 --loss-scale-k 0,14 --fp32-reference 1 --tag fp32ref

# 10 步 fp32 裁判
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/trajectory_gate.py \
  --img 256 --steps 10 --lora-dim 8 --loss-scale-k 14 --fp32-reference 1 --tag fp32ref10

# k 饱和曲线（768² 需 block swap）
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/trajectory_gate.py \
  --img 768 --steps 1 --lora-dim 8 --loss-scale-k 14,15 --blocks-to-swap 26 --tag img768swap
```
