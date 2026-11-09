# krea2_nf4_fp16_grad — Krea-2 NF4 fp16 backward 下溢探针

状态：稳定（机制结论已定；含 CPU 仿真 + V100 实测，非 sm86 复现）
适用版本：当前 dev（2026-11-08）
相关代码：`library/models/krea2_raw/quantize.py`、`library/models/krea2_raw/weights.py`、
`library/quantization/nf4.py`、bnb `bitsandbytes/autograd/_functions.py::MatMul4Bit`
相关文档：[Krea-2 NF4 fp16 梯度偏差机制溯源](../../docs/findings/krea2_nf4_fp16_grad_underflow_20261108.md)

本目录回答两个问题：**① stage6 记录的「NF4 只把 compute_dtype 从 bf16 换成 fp16，
输入梯度 rel-L2 就跳到 35.65%」到底是什么造成的；② 把缩放下沉到逐层（dir1）能否既消除
下溢又拿到 V100 的 fp16 tensor core 吞吐。** 八个脚本，各自独立可跑。

## 脚本

| 脚本 | 需要 GPU | 做什么 | 输出 |
|---|---|---|---|
| `synth_underflow.py` | 否 | 纯 CPU 数学仿真：dY 量级扫描、误差归因、loss-scale 扫描、fp16 归约上界**模拟** | `synth_underflow.json`（同目录） |
| `capture_real.py` | 是（1 张） | 真实 NF4 v2 权重 + 真实 TE/VAE 缓存，bf16/fp16/fp32 三臂单步 forward+backward，逐层抓 dY/dX 量级、`<6.1e-5`/`<6e-8` 占比、算子级拆解、归约两档 | `output/tests/krea2_nf4_fp16_grad/capture_img<N>_<tag>.json` |
| `layer_repro.py` | 是（1 张） | 用真实 6144×6144 NF4 层扫 dY 量级，定位 stage6 指纹（rel 35.65% / cos 0.9434）对应的量级 | `output/tests/krea2_nf4_fp16_grad/layer_repro.json` |
| `sim_layer_scaling.py` | 是（1 张，或 CPU 改小维度） | 合成矩阵模拟：单层 6144² 全量级扫描（dY rms 1e-10~1e-2）、28 层串联误差累积、dir1 三个边界（全零 / amax=1e-40 / 负零）。三臂共用同一权重与初始梯度，fp32 为裁判 | 终端表格 |
| `dir1_bench.py` | 是（1 张） | 算子级基准（`cuda.Event`，K=6144，M=1/256/1024/4096/16384）+ 真实权重 28 层串联精度。**基准必须用真实 token 数 M，M=1 是带宽限制场景，会得出反向结论** | 终端表格 |
| `dir1_e2e.py` | 是（1 张） | 端到端**交替**基准（bf16 vs dir1，3 轮取中位数），dir1 含 `qs.dtype=fp16` dequant 优化。单次顺序测量有 2.9× vs 3.33× 的偏差，必须交替 | `output/tests/krea2_nf4_fp16_grad/dir1_e2e_<N>_<tag>.json` |
| `dir1_chain.py` | 是（1 张） | **真实 NF4 权重**四臂对比（fp32 / bf16 / 裸 fp16 / dir1），进程内 monkey-patch `MatMul4Bit` 验证逐层 frexp 缩放。生产 `compute_dtype` 保持 bf16，契约不改 | `output/tests/krea2_nf4_fp16_grad/dir1_chain_<N>_<tag>.json` |
| `trajectory_gate.py` | 是（1 张） | 多步 LoRA 训练轨迹门槛：逐 step 比 loss / LoRA 梯度 rel-L2 / cosine。支持 `--loss-scale-k`（GradScaler 语义）、`--fp32-reference`（fp32 裁判臂）、`--blocks-to-swap`、`--expand-gqa`（绕过 sm70 GQA 内核回退）、`--skip-bf16` | `output/tests/krea2_nf4_fp16_grad/trajectory_<N>_<tag>.json` |

```bash
# CPU
CUDA_VISIBLE_DEVICES='' .venv/bin/python bench/krea2_nf4_fp16_grad/synth_underflow.py --repeat-check

# GPU（占一张卡，跑完释放；脚本内只读权重，不写用户数据）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python bench/krea2_nf4_fp16_grad/capture_real.py --img 256 --tag img256
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/layer_repro.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/trajectory_gate.py \
  --img 256 --steps 10 --lora-dim 8
# dir1（逐层缩放）方案：合成模拟 + 真实权重四臂对比
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/sim_layer_scaling.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/dir1_chain.py --img 256
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/dir1_bench.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/dir1_e2e.py --img 256 --repeat 3

# 带 loss scaling 与 fp32 裁判
CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/trajectory_gate.py \
  --img 256 --steps 10 --lora-dim 8 --loss-scale-k 14 --fp32-reference 1 --tag fp32ref10
# k 自收敛曲线（两臂直接比较，需 --expand-gqa 才能在 1024² 上跑）
for IMG in 256 512 768 1024; do
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/trajectory_gate.py \
    --img $IMG --steps 1 --lora-dim 8 --loss-scale-k 14,15 --expand-gqa --skip-bf16 --tag direct$IMG
done
```

## 结论摘要（详见 findings 文档）

- 真实训练里中间层 `grad_output` 的 rms 是 `6.5e-9 ~ 1.7e-7`，**100% 元素低于 fp16
  最小正规数 6.1e-5**，74%~99.8% 低于最小 subnormal 6e-8。bf16 指数范围与 fp32 同宽，
  同一批 dY 上误差恒定在 `1.6e-3`。**35% 与 0.16% 的差距就是 fp16 的指数范围。**
- 真实 dY × 2^k 的误差**严格每个 k 减半**（subnormal 每多一位指数多一位尾数），
  k=10~12 回到 fp32 噪声底。
- `allow_fp16_reduced_precision_reduction` 是二阶项：24 层抽样里 16 层逐位相同，
  最大差异 0.2084，没有任何一层回到 bf16 量级。
- 裸 fp16（无 loss scaling）多步轨迹 **FAIL**：10 步内 loss 相对偏差最大 603.7%、cos 最差 −0.316。
- **加上 fp32 loss + loss scaling（k=14）后，fp16 在已测维度全面优于 bf16**：单步梯度对
  fp32 是 4.58% vs bf16 的 40.66%；10 步 loss 轨迹偏差 0.8% vs 4.5%；末步 LoRA 参数
  8.56% vs 17.55%。stage6 的否决判据把 bf16 自己的误差算在了 fp16 头上。
- **k 窗口随分辨率关闭**（两臂直接比较）：256² `0.123%` → 512² `0.994%` → 768² `4.090%`
  → 1024² `5.900%`，单调上升。k 硬上限是 15（`loss×2^16 = 65536 > 65504`），
  768²/1024² 需要 k≥16~18 → 溢出。**窗口在 512² 关闭。**
- **1024² 能跑起来靠 `--expand-gqa`**：sm70 的 memory-efficient 内核在 `q_heads != kv_heads`
  时回退 MATH（7424MiB）；把 kv `repeat_interleave` 展开后走 EFFICIENT（224MiB），
  数学等价（rel-L2 ~3e-4），反向梯度正确 sum 回 12 个 kv head。**生产代码未改。**
- 生产契约维持 `COMPUTE_DTYPE=bfloat16`（本轮未改动 22 处生产代码与 AGENTS.md 不变量）。

## 边界

- **本机无 bf16 原生卡**（2×V100-16G sm70 + GTX1050 sm61）。V100 的 bf16 matmul 走
  PyTorch 的 fp32 上采样仿真（操作数 bf16 舍入 + fp32 累加），语义等价 sm86 bf16
  tensor core，但**不是** sm86 cuBLAS 的逐 bit 复现。fp16 在 V100 上是真 tensor core
  （fp32 累加 + fp16 输出舍入），这部分是实测。
- `synth_underflow.py` 只覆盖「操作数舍入 + 浮点范围」这条数学链；它的 fp16 归约结果是
  **chunked 累加模拟**（JSON 里 `simulated: true`），不是真机 kernel 测量。
- `capture_real.py` / `layer_repro.py` 用合成图案与合成 dY 分布（N(0,1)×rms），
  不是真实数据集样本。影响分布细节，不影响「dY 量级 → fp16 subnormal」这条链。
- 原始 JSON 写在 `output/tests/krea2_nf4_fp16_grad/`，是本地产物，不入库；
  入库的是 `synth_underflow.json` 与 `docs/findings/krea2_nf4_fp16_grad_underflow_20261108.json`。
