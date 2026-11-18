# ConvRot int8 训练探索（W8A16 / W8A8）

状态：实验 / **可运行（默认关闭）**
适用版本：当前 main（2026-07-24 起接入）
日期：2026-07-24（研究） / 2026-07-24（M1–M5 实现接入）
入口命令：

```bash
# 单元测试
timeout 60 .venv/bin/python -m pytest tests/test_convrot_*.py -q

# 多 seed toy probe（CI 友好）
.venv/bin/python scripts/experiments/convrot_equivalence_probe.py --mode w8a16 --seeds 0,1,2,3,4

# 训练（显式开启；默认 bf16 不变）
python tasks.py lora ... --base_compute w8a16_convrot
# 可选：--convrot_group_size 256 --convrot_scope mlp --convrot_hadamard sylvester
# 质量 opt-in：--convrot_hadamard regular --convrot_group_size 64
# W8A8（真 int8 GEMM 优先，形状不支持时回退 float）：
python tasks.py lora ... --base_compute w8a8_convrot
# 可选：
#   ANIMA_CONVROT_INT8_GEMM=auto|int_mm|float
#   ANIMA_CONVROT_FUSED=1|0              # 默认 1：单 autograd.Function 融合 RHT+GEMM
#   ANIMA_CONVROT_RHT=dense|fwht         # 默认 dense（本机 3080 更快；fwht 仅 sylvester）
#   ANIMA_CONVROT_HADAMARD=sylvester|regular  # 也可由 --convrot_hadamard 写入
#   ANIMA_CONVROT_W8A16_KERNEL=dequant|int8pack  # 默认 dequant（int8pack 本机更慢）
#   ANIMA_CONVROT_STE_TF32=0|1               # 默认 0；W8A8 STE 用 TF32（更快但可能破 grad gate）

# P1-G / P1-F 可选（默认全关，行为与 Phase 1 相同）
#   --convrot_min_in_features 4096
#   --convrot_largest_in_features_only
#   --convrot_large_layer_mode w8a8 --convrot_large_min_in_features 4096

# P0-A step profiler（bf16 / w8a16 / w8a8 单 step 饼图）
.venv/bin/python scripts/experiments/convrot_step_profile_probe.py \
  --cases bf16,w8a16_free,w8a8_auto \
  --json-out output/tests/convrot_step_profile.json

# P0-C：导出 / 加载 prequant 权重（去掉 apply 期 online weight RHT+quant）
.venv/bin/python scripts/experiments/convrot_export_prequant.py \
  --scope mlp --group-size 256 \
  --out output/tests/convrot_prequant_mlp_g256.safetensors
python tasks.py lora ... --base_compute w8a16_convrot \
  --convrot_weight_source prequant_checkpoint \
  --convrot_prequant_path output/tests/convrot_prequant_mlp_g256.safetensors
```

相关代码：

- `library/runtime/convrot/`（RHT / quant / W8A16 / W8A8 / apply / free_base / gemm / fused / prequant / checks / metadata）
- `library/training/bootstrap.py::maybe_apply_convrot_base`（compile 前薄钩子）
- `library/training/cli_args.py`（`base_compute` / `convrot_*`）
- 探针：`scripts/experiments/convrot_{equivalence,checkpoint,short_train,mem_speed,step_profile,export_prequant,fusion_microbench}_probe.py`（export / fusion 脚本名见上；fusion 文件为 `convrot_fusion_microbench.py`）
- 非 ConvRot 对照：`library/runtime/int8_linear.py`、`block_swap_transfer_dtype=int8`

相关 findings：[`../findings/anima_int8_base_linear_audit.md`](../findings/anima_int8_base_linear_audit.md)、[`../methods/channel_scaling.md`](../methods/channel_scaling.md)、`bench/channel_stats/`
落地规格：[`../../_archive/docs/proposal/convrot_w8a_training_plan.md`](../../_archive/docs/proposal/convrot_w8a_training_plan.md)
后续优化：[`../proposal/convrot_w8a_optimization_roadmap.md`](../proposal/convrot_w8a_optimization_roadmap.md)

> **一句话：** ConvRot 训练路径已按战略 C 自建接入；默认 `base_compute=bf16`。
> **不要**把 `int8_linear` 裸 rowwise 存储或 block-swap int8 传输当成 ConvRot。
> **不要**与 `--block_swap_transfer_dtype int8` 同时开启。

---

## 实现状态（2026-07-24）

| 里程碑 | 状态 | 说明 |
| --- | --- | --- |
| M1 算法核 | ✅ | `rht` / `quant` / `linear_w8a16` + unit tests |
| M2 训练 apply | ✅ | `apply_convrot_to_lora_network` + bootstrap 薄钩 + CLI + 与 block-swap int8 互斥 |
| M3 质量 gate | ✅ | toy + full-checkpoint + 20-step short-train（W8A16/W8A8 均已跑） |
| M4 W8A8 | ✅ | 真 int8 GEMM（`torch._int_mm`，形状不支持时 float 回退）+ STE |
| M5 产品化钩子 | ✅ 最小集 + WebUI MVP | metadata stamp、merge 拒绝、DoRA 拒绝、skip 日志；WebUI：`base_compute` / `convrot_group_size` / `convrot_scope=mlp`（配置页 → 显存与速度优化） |
| 显存优化 | ✅ | quant 后默认释放 base `Linear.weight`（meta 占位，去掉双份 bf16） |
| 融合路径 | ✅ | 单 `autograd.Function` 融合 RHT+quant+GEMM；默认 **dense RHT + dequant**（见 §F） |
| P0-A step profile | ✅ | `torch.profiler` 三 case 饼图（见 §G）；**不**满足 Triton 门槛 |
| P0-A2 bf16 计算 | ✅ | W8A16 dequant linear 用 bf16 TC（见 §G.4） |
| P0-C prequant 加载 | ✅ | 原生 `anima_lora_convrot_prequant_v1` + `weight_scale` 别名（见 §H） |
| P0-D regular Hadamard | ✅ | Kronecker `H_{4^k}` 实现 + multi-seed 对照（见 §I）；**默认仍 sylvester** |

### 验证矩阵总览（本机，2026-07-24 / 07-25）

| 层级 | 脚本 | W8A16 | W8A8 |
| --- | --- | --- | --- |
| Unit / apply / free / gemm / fused / profile / prequant / rht | `tests/test_convrot_*.py` | PASS（66） | PASS |
| Toy multi-seed | `scripts/experiments/convrot_equivalence_probe.py` | PASS（5 seed） | PASS（3 seed） |
| Full DiT + LoRA 1-step | `scripts/experiments/convrot_checkpoint_probe.py` | 2/3 严格 gate | 2/3 严格 gate |
| 20-step short-train + sample | `scripts/experiments/convrot_short_train_probe.py` | loss rel ≪1% | loss rel ≪1% |
| Mem/speed（free-base 后） | `scripts/experiments/convrot_mem_speed_probe.py` | peak **低于** bf16 | peak **低于** bf16 |
| Step profile 饼图 | `scripts/experiments/convrot_step_profile_probe.py` | 1.77× 更慢 | 1.46× 更慢 |

共同设置（checkpoint / short-train）：

- DiT：`models/diffusion_models/anima-preview3-base.safetensors`
- 数据：`post_image_dataset/rokkotsu_goddess` cached latent/TE
- adapter：plain LoRA r=4 / α=4，`scope=mlp`，`group=256`，`patched=56`
- 严格 smoke：`out_rel≤3%`、`loss_rel≤5%`、`grad_rel≤5%`（仅 1-step probe）
- 产物目录：`output/tests/convrot_ckpt_*.json`、`output/tests/convrot_short_train/`

### A. Toy probe

```text
W8A16 group=16 seeds=0..4
  max_out_rel≈0.009  max_grad_rel≈0.007  → gates PASS
W8A8  group=16 seeds=0..2
  max_out_rel≈0.013  max_grad_rel≈0.009  → gates PASS
```

### B. Full checkpoint 1-step probe（bf16 LoRA vs ConvRot base）

JSON：`output/tests/convrot_ckpt_w8a16_seeds012.json`、`output/tests/convrot_ckpt_w8a8_seeds012.json`

| mode | seed0 out/loss/grad | seed1 | seed2 | 严格 gate | 汇总 |
| --- | --- | --- | --- | --- | --- |
| **W8A16** | 0.0062 / 0.0004 / **0.0594 FAIL** | 0.0049 / 0.0003 / 0.0010 PASS | 0.0155 / 0.0001 / 0.0126 PASS | **2/3** | out_max=0.0155；grad_max=0.0594；mean_grad≈0.024；peak≈6.2 GB |
| **W8A8** | 0.0077 / 0.0003 / **0.1290 FAIL** | 0.0063 / 0.0009 / 0.0005 PASS | 0.0215 / 0.0004 / 0.0034 PASS | **2/3** | out_max=0.0215；grad_max=0.1290；mean_grad≈0.044；peak≈6.7 GB |

结论：

- **输出 / loss**：两模式全 seed 都远低于阈值。
- **adapter grad**：多数 seed 稳定；seed0 对两种模式都是最差 seed。W8A16 略超 5%（~5.9%），W8A8 更松（~12.9%），符合「act quant 额外噪声」预期。
- **不发散**：失败 seed 仍是有限相对误差，不是 NaN / 量级爆炸。

### C. Short-train sample 对照（20 steps，seed=0）

JSON：`output/tests/convrot_short_train/short_train_w8a16_seed0.json`、`short_train_w8a8_seed0.json`
图：`sample_bf16_seed0.png`、`sample_w8a16_seed0.png`、`sample_w8a8_seed0.png`
并排：`sample_bf16_vs_w8a16_seed0.png`、`sample_bf16_vs_w8a8_seed0.png`、`sample_bf16_w8a16_w8a8_seed0.png`

| 路径 | first → last loss | last_loss rel vs bf16 | per-step mean rel | sample pixel rel vs bf16 | peak GB（优化前） |
| --- | --- | --- | --- | --- | --- |
| bf16 | 0.9520 → 0.1353 | — | — | — | ≈5.0 |
| **W8A16** | 0.9521 → 0.1355 | **≈0.09%** | ≈0.036% | ≈1.6%（MAE≈1.1/255） | ≈6.2（双份权重） |
| **W8A8** | 0.9523 → 0.1354 | **≈0.09%** | ≈0.060% | ≈2.2%（MAE≈1.6/255） | ≈6.7（双份权重） |

结论：20 步内两条 ConvRot 路径的 loss 轨迹与 bf16 几乎重合；sample 像素差很小。上表 peak 为 **free-base 优化前**；优化后 peak 见 §E.3。

### D. 总判定（对照 proposal §4.1）

| 标准 | W8A16 | W8A8 |
| --- | --- | --- |
| adapter grad 多 seed 稳定 | **大体通过**（1/3 seed 略超 5% smoke） | **可用但更松**（1/3 seed grad 12.9%） |
| 短训 sample 接近同超参 bf16 | **通过** | **通过** |
| 默认 bf16 零回归 | 保持（默认关闭） | 保持（默认关闭） |
| Phase 1 优先推荐 | **是** | 实验可用；真 int8 GEMM 已接 |

### E. 任务 1+2：释放 base 权重 + 真 int8 GEMM（2026-07-24 续）

#### E.1 释放 GPU 上的 bf16 base（任务 1）

实现：`library/runtime/convrot/free_base.py`，`apply_convrot_to_lora_network(..., free_base_weights=True)` 默认开启。

- quant 成功后把 patched 的 `nn.Linear.weight` 换成 **meta** 占位（形状/dtype 保留，**0 字节存储**）
- 仅替换 `org_forward` 的约定不变；误走原 `Linear.forward` 会因 meta 立刻失败
- dry_run / `free_base_weights=False` 不释放
- mlp scope 实测 **freed ≈ 1792 MiB**（56 个 Linear）

#### E.2 真 int8 GEMM（任务 2）

实现：`library/runtime/convrot/gemm.py` + `linear_w8a8.py`。

- 前向：act per-token absmax → int8；`int8×int8 → int32 → ×scales → float`
- CUDA 优先 `torch._int_mm`（`M>16` 且 K/N 合理）；否则 float 模拟
- 反传：STE（对 act quant 视为恒等；base 冻结无 W 梯度）
- 环境变量：`ANIMA_CONVROT_INT8_GEMM=auto|int_mm|float`（默认 `auto`）

#### E.3 Mem/speed 对照（free-base 后，8/6 step microbench）

脚本：`scripts/experiments/convrot_mem_speed_probe.py`
JSON：`output/tests/convrot_mem_speed.json`

| case | peak GB | alloc after apply GB | sec/step | meta bases | freed MiB |
| --- | --- | --- | --- | --- | --- |
| **bf16** | **5.00** | 3.96 | **1.84** | 0 | 0 |
| **w8a16_free** | **4.43** | 3.10 | 2.94 | 56 | 1792 |
| w8a16_nofree（对照） | 7.07 | 5.74 | 2.96 | 0 | 0 |
| **w8a8_auto**（int_mm） | **4.52** | 3.10 | 2.54 | 56 | 1792 |
| w8a8_float | 4.52 | 3.10 | 3.23 | 56 | 1792 |

解读：

1. **显存：** 释放 base 后 W8A16/W8A8 peak **低于** bf16（约 −0.5 GB）；不释放时 peak 反而到 ~7 GB（双份权重）。
2. **速度：** 端到端仍慢于 bf16（online RHT + quant 开销主导）。W8A8 `auto` 快于 `float`（约 2.54 vs 3.23 s/step），但仍慢于 bf16。
3. **内核微基准**（单次大 GEMM `64×2048 @ 8192`）：本机 3080 上 `_int_mm` 与 float 模拟同量级（甚至 float 略快）——端到端收益主要来自 **少一次完整 dequant 权重物化**，不是单 kernel 碾压 bf16 Tensor Core。
4. **质量：** 同 loss 量级；free-base 不改变 quant 数值路径。

### F. 任务 3：融合 RHT + quant + GEMM（2026-07-24 续）

实现：`library/runtime/convrot/fused.py` + `group_fwht`（`rht.py`）。

融合内容（训练安全、无 Triton 依赖）：

1. **单 `autograd.Function`**：W8A16 / W8A8 前向把 RHT →（act quant）→ GEMM 收成一次 forward/backward，避免 Python 层中间图。
2. **RHT 后端**（`ANIMA_CONVROT_RHT`）：
   - `dense`（**默认**）：缓存 `H/√G` + `matmul`（cuBLAS）
   - `fwht`：O(D log G) butterfly（与 dense Sylvester 数值等价）
3. **W8A16 权重 GEMM**（`ANIMA_CONVROT_W8A16_KERNEL`）：
   - `dequant`（**默认 / auto**）：`dequant(w_q) → F.linear`
   - `int8pack`：`torch._weight_int8pack_mm`（opt-in）
4. **W8A8**：RHT → absmax act quant → `int8_mm_scaled`（`_int_mm` / float 回退）
5. **开关**：`ANIMA_CONVROT_FUSED=1`（默认）/ `0` 回退非融合路径

#### F.1 内核微基准（RTX 3080，本机）

| 路径 | 相对结果（同 shape 量级） |
| --- | --- |
| dense RHT vs FWHT | dense **约 10× 更快**（例 FWHT ~6.3 ms vs dense ~0.6 ms） |
| dequant+`F.linear` vs `_weight_int8pack_mm` | dequant **显著更快**（例 ~3 ms vs ~97 ms @ 512×2048@8192） |

#### F.2 端到端 mem/speed（融合默认 vs 错误默认）

JSON：

- 推荐默认（fused + dense RHT + dequant）：`output/tests/convrot_mem_speed_fused_dense.json`
- free-base 初版对照：`output/tests/convrot_mem_speed.json`（§E.3）
- 曾误默认 FWHT/int8pack 的 fused 对照：`output/tests/convrot_mem_speed_fused.json`

| case | peak GB | sec/step | 备注 |
| --- | --- | --- | --- |
| bf16 | 5.00 | **1.70** | 基线 |
| w8a16 free + fused dense/dequant（**当前默认**） | **4.37** | **2.98** | 可用 |
| w8a8 free + fused auto int_mm（**当前默认**） | **4.49** | **2.55** | 可用 |
| w8a16 free + fused 旧默认（FWHT/int8pack） | 4.55 | **47.6** | 不可用；已废弃为默认 |
| w8a8 free + fused 旧默认 | 4.55 | 3.61 | 仍慢于 dense 路径 |

解读：

1. **融合接口已落地**，但「融合 ≠ 更快」。本机 3080 上 **dense RHT + dequant** 才是可用默认。
2. FWHT / `_weight_int8pack_mm` 保留为 **opt-in**（研究 / 其它 GPU 再 profile）。
3. 真正的吞吐跃迁仍需 **Triton 真融合 kernel** 或 prequant checkpoint 去掉 online rotate 热路径。
4. 数值：dense 与 FWHT 相对误差 <1e-6；fused vs legacy dequant 路径 <1e-4（unit tests）。

**尚未做：** 完整 ComfyUI MixedPrecisionOps INT8-ConvRot 布局自动适配（仅 best-effort 键名）；Triton 真融合 kernel；完整 `train.py` 长训；WebUI 高级项（prequant / Hadamard / kernel env）；Hydra 等变体 `org_module_ref` 全覆盖。
**WebUI MVP：** 配置页 → 优化 →「显存与速度优化」：`base_compute` / `convrot_group_size` / `convrot_scope=mlp`（默认 bf16）。

### G. P0-A：单 step CUDA profiler（2026-07-24）

脚本：`scripts/experiments/convrot_step_profile_probe.py`
JSON：`output/tests/convrot_step_profile.json`
环境：RTX 3080 / torch 2.12.0+cu130；默认 dense RHT + dequant fused；warmup=2；capture 1 train step + 3-step wall microbench。

#### G.1 端到端

| case | sec/step（无 profiler） | peak GB | CUDA self 合计 | vs bf16 wall |
| --- | --- | --- | --- | --- |
| bf16 | **1.73 s** | 4.99 | 3.91 s | 1.0× |
| w8a16_free | **3.06 s** | **4.37** | 7.86 s | **1.77×** |
| w8a8_auto | **2.52 s** | **4.49** | 6.48 s | **1.46×** |

#### G.2 CUDA self-time 粗桶占比（%）

| bucket | bf16 | w8a16_free | w8a8_auto |
| --- | --- | --- | --- |
| gemm_generic | 51.7 | 52.6 | 37.8 |
| convrot_gemm（标记） | — | **12.0** | **11.7** |
| convrot_rht | — | **1.4** | **1.7** |
| convrot_dequant | — | 0.6 | — |
| convrot_act_quant | — | — | 2.6 |
| attention | 10.0 | 5.4 | 6.3 |
| memcpy_cast | 6.0 | 6.1 | 11.0 |
| other | 30.4 | 21.0 | 27.6 |

显式标记事件（top）：

- W8A16：`convrot::gemm_dequant_linear` ≈ **941 ms**（单 step self）；RHT 仅 ~1.4%
- W8A8：`convrot::gemm_int8` / `_int_mm` 可见；act quant ~2.6%；RHT 仍小

说明：profiler 的 `self` 时间会把部分 matmul 记在 `aten::mm`（gemm_generic）下；`convrot::*` 是注入的 `record_function` 边界，用于归因 ConvRot 路径，不是与 wall 秒严格一一对应。

#### G.3 决策（路线图分支）

自动启发式（基于 w8a16_free）：

```text
branch = fix_w8a16_keep_bf16_compute
reason = W8A16 dequant linear ~12% 而 RHT 仅 ~1.4%：
         更像 fp32 F.linear 相对 bf16 Tensor Core 的税，而非 RHT 内存链
convrot_tax ≈ 14%  << 50%  →  **不** 开 P2-K Triton
wall_ratio ≈ 1.77×
```

解读：

1. **Triton 真融合门槛未满足**（ConvRot 链 <50% CUDA self）。
2. **W8A16 慢的主因** 更像 **dequant 后走 fp32 `F.linear`**（top kernel 出现 `ampere_sgemm_*` / cutlass sgemm），而 bf16 路径吃 `bf16` Tensor Core gemm。
3. **RHT 本身不是主税**（~1–2%）；继续优化 FWHT/融合 RHT ROI 低。
4. **下一刀建议（按 ROI）**：
   - **已做（P0-A2）**：W8A16 dequant/`F.linear` 改用 bf16/fp16 计算 dtype（见 §G.4）
   - **已做（P0-C）**：prequant 加载/导出（见 §H；去掉 apply 期 online weight quant）
   - 仍 **不要** 默认 int8pack / FWHT；Triton 门槛仍未达

#### G.4 P0-A2 落地：W8A16 保持 bf16 计算（同日续）

代码：`fused.py` / `linear_w8a16.py` — RHT 仍 float32；dequant linear 用 act 的 bf16/fp16（CUDA 上 fp32 输入回退 bf16 TC）。

Mem/speed 复测 JSON：`output/tests/convrot_mem_speed_bf16_compute.json`（6 steps）

| case | peak GB | sec/step | vs bf16 | 备注 |
| --- | --- | --- | --- | --- |
| bf16 | 5.00 | **1.62** | 1.0× | 基线 |
| w8a16_free（A2 后） | **4.34** | **1.99** | **1.23×** | 此前 ~3.0 s / 1.77× |
| w8a8_auto | 4.49 | 2.55 | 1.57× | 与 A2 前同量级 |

结论：A2 把 W8A16 端到端从 **~1.77× 慢收到 ~1.23×**，且 peak 仍低于 bf16。剩余差距主要来自 RHT + dequant 额外链与其它开销，**不是**「再写一个 Python fusion」能抹平；Triton 仍非当前最高 ROI。

#### G.5 单 op microbench + fusion 上界（2026-07-25，步骤 1+2）

脚本：`scripts/experiments/convrot_fusion_microbench.py`

JSON：`output/tests/convrot_fusion_microbench.json`

测试：`tests/test_convrot_fusion_microbench.py`

环境：RTX 3080 / torch 2.12.0+cu130；Anima MLP 形状 `layer1 [8192,2048]` / `layer2 [2048,8192]`；M∈{64,512,4032}；dtype=bf16。

**目的（对应优化讨论步骤 1+2）：**

1. 固化单 op 基准：bf16 `F.linear` / W8A16 dequant+linear / predequant linear / int8pack / W8A8 `_int_mm`+post-scale / `_int_mm` only / float fallback / 反传 full vs chunked
2. **不写 Triton**，用差值估 fusion 上界，再决定要不要开 P2

端到端锚点（未在本脚本重跑，来自 A2 mem_speed）：

| case | sec/step | peak GB |
| --- | --- | --- |
| bf16 | 1.62 | 5.00 |
| w8a16_free | 1.99（1.23×） | 4.34 |
| w8a8_auto | 2.55（1.57×） | 4.49 |

**bucket-M（M=4032，训练 token 量级）摘要：**

| 路径 | 相对 | 解读 |
| --- | --- | --- |
| W8A16 dequant+linear vs bf16 | **~1.21×** | 与 e2e 1.23× 同量级 |
| W8A16 **predequant** vs bf16 | **~1.00×** | 免费 dequant 的天花板 = bf16 linear |
| W8A16 dequant 税（占 dequant 路径） | **~17.5%** | 小 M 会虚高到 60–90%，不能拿来开 P2 |
| W8A8 post-scale vs bf16 | **~1.60×** | 仍慢 |
| W8A8 **int_mm only** vs bf16 | **~1.31×** | 去掉 scale 也赢不了 bf16 TC |
| W8A8 post-scale 税（占 scaled 路径） | **~17%** | 有税，但 body 才是主因 |
| `_weight_int8pack_mm` vs dequant | **~9–100× 更慢** | 继续禁止默认 |

**反传 `grad_x = gy @ dequant(W)` vs out-dim chunked：**

| | 速度 | 峰值 |
| --- | --- | --- |
| chunked vs full | **~5× 更慢**（平均） | **~0.67×** 峰值（有省） |
| 数值 | rel err ~1e-2 量级 | — |

**决策（脚本 `decision` 字段，以 bucket-M 为主）：**

| 项 | 结论 |
| --- | --- |
| W8A8 epilogue 真融合实现 | **否**（tax≥15% 但 int_mm-only 已 1.31× bf16，融完仍输） |
| W8A16 K-loop / Triton dequant 融合 | **否**（天花板 = bf16 linear，且 bucket-M dequant 税仅 ~17%） |
| 反传 chunked 为速度 | **否** |
| 反传 chunked 为峰值 | **可选**（OOM / 大层时） |
| **P2 Triton 现在就开** | **否** |
| 下一步默认 | 继续 **P1**（compile 友好 / 缩 scope / 混精层）；产品 KPI 仍是显存 |

一句话：步骤 1+2 完成——**基准已冻、上界已量**；「先融进 K 循环再对标」被数据否定为下一刀。

### G.6 P1-F/G/H：缩 scope / 混精 / compile 友好（2026-07-25）

实现面（默认全关，**不改变**既有 `mlp` 全量 patch 行为）：

| 旋钮 | 作用 | 入口 |
| --- | --- | --- |
| `convrot_min_in_features` | 跳过 `in_features` 小于阈值的层（P1-G） | CLI / WebUI / apply |
| `convrot_largest_in_features_only` | scope 内只 patch 最大 `in_features`（Anima mlp → 常为 layer2） | 同上 |
| `convrot_large_layer_mode` + `convrot_large_min_in_features` | 大层覆盖 mode（如默认 w8a16、大层 w8a8）（P1-F） | 同上 |
| 预计算 `_convrot_hadamard` | dense RHT 矩阵挂 buffer，传入 fused，避免 hot-path env/cache（P1-H） | apply 默认 |

示例：

```bash
# 只 patch 最大 in 的 MLP 层
python tasks.py lora ... --base_compute w8a16_convrot --convrot_largest_in_features_only

# 小层 W8A16，in>=4096 的层 W8A8
python tasks.py lora ... --base_compute w8a16_convrot \
  --convrot_large_layer_mode w8a8 --convrot_large_min_in_features 4096
```

测试：`tests/test_convrot_apply.py`（min_in / largest_only / mixed mode / hadamard buffer）、`tests/test_convrot_bootstrap_hook.py`、`tests/test_convrot_fused.py`（hadamard 进 fused）。

#### G.6.1 P1 热测（2026-07-25，RTX 3080，scope=mlp，steps=6）

命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/experiments/convrot_mem_speed_probe.py \
  --cases p1 --steps 6 --group-size 256 --scope mlp \
  --json-out output/tests/convrot_mem_speed_p1.json
```

| 路径 | peak GB | sec/step | ×bf16 | patched | freed MB | loss |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 5.00 | 1.430 | 1.00 | 0 | 0 | 0.0944 |
| w8a16_free（全 mlp） | **4.34** | 1.735 | 1.21 | 56 | 1792 | 0.0944 |
| w8a16_largest（仅 max in） | 4.77 | **1.619** | **1.13** | 28 | 896 | 0.0944 |
| w8a16_min4096 | 4.77 | 1.623 | 1.14 | 28 | 896 | 0.0944 |
| w8a16 + large@4096→w8a8 | 4.49 | 1.958 | 1.37 | 56 | 1792 | 0.0944 |
| w8a8_auto（全 mlp） | 4.49 | 2.140 | 1.50 | 56 | 1792 | 0.0945 |

解读：

- **largest / min4096 等价**（Anima mlp 只有 in=2048 / 8192 两档）：各 28 层，只 patch layer2。
- 缩 scope 相对 full w8a16：**更快**（1.62 vs 1.74 s，≈1.13× bf16），但 **省显存减半**（peak 4.77 vs 4.34；freed 896 vs 1792 MB）——小层仍是 bf16 全量权重。
- **混精（小层 W8A16 + 大层 W8A8）** 峰值与 full W8A8 同档（4.49 GB），速度介于 full W8A16 与 full W8A8 之间（1.96 s），**未** 同时拿到「最快 + 最省」。
- 产品默认仍建议：**full mlp W8A16 free**（峰值最低、×1.21 可接受）；若更在意 step 且能接受 ~0.4 GB 回吐，用 `largest_only` / `min_in=4096`。
- 本轮 loss 与 bf16 一致（1-batch short probe，非质量门替代）。

JSON：`output/tests/convrot_mem_speed_p1.json`

**未做（当时）：** trust-mask STE（P1-J）；compile_blocks 下 re-profile P1-H 收益。

### G.7 P1.5：dtype 交通税（2026-07-25）

实现（**数学语义不变**，只砍强制 fp32 往返 / 多余 cast）：

| 改动 | 位置 | 作用 |
| --- | --- | --- |
| `dequantize_weight(..., dtype=bf16)` 直接在目标 dtype 乘 | `quant.py` | 去掉 full fp32 `[out,in]` 中间张量 |
| fused W8A16 RHT+GEMM 全程 act TC dtype | `fused.py` | 去掉 `x→fp32→RHT→bf16` 与 `y→fp32→x.dtype` |
| fused W8A8 激活 RHT 用 bf16 | `fused.py` / `linear_w8a8.py` | absmax quant 内部仍 fp32 |
| hadamard buffer 默认 bf16（CUDA） | `apply.py` | 与 act dtype 对齐，省 per-call cast |
| `int8_mm_scaled` 就地 `mul_` scale | `gemm.py` | 少两个 out-of-place 临时 |
| hadamard 不进 `save_for_backward` | `fused.py` | 常量 buffer 挂 `ctx`，减 autograd 图 |

热测（RTX 3080，scope=mlp，steps=6）：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/experiments/convrot_mem_speed_probe.py \
  --cases bf16,w8a16_free,w8a8_auto --steps 6 \
  --json-out output/tests/convrot_mem_speed_p15.json
```

| 路径 | peak GB | sec/step | ×bf16 | vs P1 sec | vs P1 peak |
| --- | --- | --- | --- | --- | --- |
| bf16 | 5.00 | 1.433 | 1.00 | ≈ | ≈ |
| **w8a16_free** | **4.14** | **1.508** | **1.05** | 1.735→1.508（−13%） | 4.34→4.14 |
| w8a8_auto | 4.43 | 2.062 | 1.44 | 2.140→2.062（−4%） | 4.49→4.43 |

单层 sanity：W8A16/W8A8 rel vs bf16 linear ≈ 0.8% / 1.0%，grad 有限。

JSON：`output/tests/convrot_mem_speed_p15.json`

**结论：** W8A16 已逼近 bf16 step（+5%）；主 KPI（显存）再降 ~0.2 GB。W8A8 仍慢在 `_int_mm`+act quant，非 dtype 交通主因。

**未做：** P1-J trust-mask；W8A8 预转置 `w_q.T` 常驻 buffer；compile_blocks 下 re-profile。

### G.8 P1.6 / P1.7：W8A8 kn 布局 + 反传无完整 dequant + 共享 Hadamard（2026-07-25）

| 改动 | 作用 | 热测结论 |
| --- | --- | --- |
| W8A8 存 `[K,N]`（`weight_layout=kn`） | 去掉每步 `w_q.t().contiguous()` | 相对 P1.5 **中性**（sec ±2% 噪声） |
| act quant：half 上 amax 再升 fp32 | 少一次 full fp32 abs | 同上 |
| bwd：`grad @ W` → `(gy*scale) @ W_q` | 不物化完整 dequant `W` | peak 不变（已有 ckpt 主导） |
| 同 group 共享一个 Hadamard buffer | 56×128KiB → 1×128KiB | alloc_after_apply 仍 3.10GB（探针粒度下不可见） |

热测 JSON：`output/tests/convrot_mem_speed_p16.json`、`p17.json`  
相对 P1.5：W8A16 仍 **~1.05× / 4.14GB**；W8A8 仍 **~1.47× / 4.43GB**。

**产品默认不变：** full-mlp free-base **W8A16**。P1.6/1.7 是正确性/清洁实现，不单独构成速度叙事。

### G.9 P1.8 dequant scratch + compile 热测（2026-07-25）

| 项 | 结论 |
| --- | --- |
| 共享 dequant scratch（flat `max(N*K)`） | 实现保留；**默认关闭**（`ANIMA_CONVROT_DEQUANT_SCRATCH=1` opt-in）。2D max(N)×max(K) 会误扩到 8192²；flat 修后 step 仍中性、常驻 buffer 略抬 peak |
| probe `--torch-compile` | 对齐训练：apply → `compile_blocks_for_training` |

compile 热测（RTX 3080，mlp，steps=4，inductor + grad-ckpt）：

| 路径 | peak GB | sec/step | ×bf16 |
| --- | --- | --- | --- |
| bf16 + compile | 4.95 | 1.139 | 1.00 |
| **w8a16_free + compile** | **4.11** | **1.180** | **1.036** |
| w8a8_auto + compile | 4.17 | 1.657 | 1.455 |

JSON：`output/tests/convrot_mem_speed_compile_p18.json`  
无 compile 基线（P1.5）：W8A16 **1.05× / 4.14GB**。

**结论：** 训练常用 compile 路径上 W8A16 已逼近 bf16（+3.6%），显存仍省 ~0.8 GB。W8A8 在 compile 下仍 ~1.45×，主税在 int8 GEMM/act quant，非 dtype 交通。

### G.10 质量复测 + scope=all 显存（2026-07-25，P1.5–1.8 后）

#### 质量（相对历史 2/3 gate 模式，**非回归**）

| 探针 | 结果 |
| --- | --- |
| toy equivalence W8A16 seeds 0–2 | out≤0.9% / grad≤0.7% **PASS** |
| toy equivalence W8A8 seeds 0–2 | out≤1.3% / grad≤0.9% **PASS** |
| full-ckpt W8A16 mlp syl@256 | **2/3** PASS（seed2 grad 6.6% >5%；seed0/1 OK）— 与历史 `seeds012` / `syl_g256` 同型 |
| full-ckpt W8A8 mlp | **3/3** PASS（优于历史 2/3） |
| full-ckpt W8A16 **scope=all** | **2/3** PASS（seed0 grad 7.2% >5%） |

JSON：`output/tests/convrot_ckpt_w8a16_p18.json`、`convrot_ckpt_w8a8_p18.json`、`convrot_ckpt_w8a16_scope_all.json`

#### scope=all 热测（eager / compile）

| 路径 | peak GB | sec/step | ×同 scope bf16 | patched | freed |
| --- | --- | --- | --- | --- | --- |
| bf16 LoRA@all | 5.03 | 1.57 | 1.00 | 0 | 0 |
| **W8A16 free@all** | **3.44** | 1.70 | **1.08** | 196 | **3360 MB** |
| W8A8 free@all | 3.69 | 2.80 | 1.78 | 196 | 3360 MB |
| bf16@all + compile | 5.00 | 1.20 | 1.00 | 0 | 0 |
| **W8A16@all + compile** | **3.43** | 1.30 | **1.08** | 196 | 3360 MB |

JSON：`output/tests/convrot_mem_speed_scope_all.json`、`convrot_mem_speed_scope_all_compile.json`

**产品叙事：**

- **速度优先 / 默认：** `scope=mlp` + compile → W8A16 **~1.04×**，peak ~4.1 GB。  
- **显存优先：** `scope=all`（mlp+attn）+ free_base → peak **~3.4 GB**（再省 ~0.7 GB vs mlp-only），step ~1.08× 同 scope bf16；质量门仍是历史同型 2/3。  
- W8A8 仍不作为速度默认；scope=all 时更慢。

### G.11 regular@64 质量 + 20-step short-train（2026-07-25）

| 探针 | 结果 |
| --- | --- |
| W8A16 **regular@64** ckpt seeds 0–2 | **3/3 PASS**（grad_max 1.0%，out_max 1.6%）— 优于 sylvester@256 的 2/3 |
| short-train 20-step W8A16 syl@256 seed0 | last_loss rel vs bf16 **0.07%**（0.13536 vs 0.13546） |

JSON：`output/tests/convrot_ckpt_w8a16_reg64_seeds012_p18.json`、`output/tests/convrot_short_train_p18/short_train_w8a16_seed0.json`

**质量建议（仍 opt-in，不改默认 sylvester@256）：**

```bash
export ANIMA_CONVROT_HADAMARD=regular
python tasks.py lora ... --base_compute w8a16_convrot --convrot_group_size 64
```

默认仍 sylvester@256（兼容 / 与既有 prequant 对齐）；若 seed 敏感或要收紧 grad gate，优先 regular@64。

### G.12 W8A8 STE 瘦身 + out_dtype（2026-07-25，P1.9）

| 改动 | 作用 |
| --- | --- |
| `_W8A8IntLinearFn` 不再 `save_for_backward(x_rot)` | 只存 `x_dtype`；STE 不需要激活 |
| `int8_mm_scaled(..., out_dtype=bf16)` | fused 路径 y 直接 bf16，少一次 full cast |

热测（mlp）：

| 路径 | peak | sec/step | 备注 |
| --- | --- | --- | --- |
| w8a8 eager | 4.43 GB | ~2.11 s | 相对 P1.5 中性 |
| **w8a8 + compile** | **4.14 GB** | **1.582 s** | 此前 compile 1.657 s（~−4.5%） |

JSON：`output/tests/convrot_mem_speed_w8a8_compile_p19.json`

W8A8 仍 ~1.4× bf16@compile；主税仍是 `_int_mm` + act quant，非 save/cast。

### G.13 产品 profile 对照（2026-07-25）

| Profile | scope / 旋钮 | peak | sec/step | ×bf16 | 用途 |
| --- | --- | --- | --- | --- | --- |
| **速度默认** | mlp + free + compile | **4.11 GB** | 1.18 s | **1.036** | 日常训 |
| largest-only + compile | mlp + largest | 4.51 GB | 1.17 s | 1.031 | 略快、省显存少 |
| attention_out | out-proj only | 4.66 GB | 1.40 s | 1.014 | 几乎不省显存 |
| **显存优先** | all + free + compile | **3.43 GB** | 1.30 s | **1.08** | 极限 VRAM |
| W8A8 mlp + compile | mlp | **4.14 GB** | **1.58 s** | **~1.39** | 非速度默认（见 G.15） |

WebUI：`convrot_scope` 选项扩展为 `mlp | all | attention_out | attn | mlp,attn`。

### G.14 P1.10 scale bf16 + `convrot_hadamard` 产品化（2026-07-25）

| 项 | 内容 |
| --- | --- |
| scale 存储 | **W8A16** CUDA 上 per-out scale 存 bf16；**W8A8 保持 float32**（bf16 scale  alone 会把 seed0 grad 推过 5% gate） |
| 目的 | W8A16 dequant / `(gy*scale)` 少一次 per-step cast；W8A8 保质量 |
| CLI | `--convrot_hadamard {sylvester,regular}` 默认 `sylvester` |
| Bootstrap | 写入 `ANIMA_CONVROT_HADAMARD`（CLI/WebUI 覆盖裸 env） |
| Metadata | `ss_convrot_hadamard` |
| WebUI | 同字段 + help；质量建议 `regular` + group `64` |
| 热测 | p110 eager 与 p15 同量级（dtype 已吃完主税）；regular@64 ~1.04× |

默认仍 **sylvester@256**；seed 敏感 / 收紧 grad 时 WebUI 或 CLI 切 **regular@64**。

### G.15 P1.11 W8A8 加速尝试：**否决默认**（2026-07-25）

尝试与结果（mlp full-ckpt gate：out≤3% / loss≤5% / grad≤5%）：

| 尝试 | 速度（compile） | 质量 | 结论 |
| --- | --- | --- | --- |
| half act quant（无 full `x.fp32`） | **1.27 s**（~1.13×） | **1/3**（seed2 grad_rel ~1.4） | **否决默认** |
| half quant + 仍 fp32 STE | ~1.58 s | **1/3** | 质量仍坏 → 根因是 **codes** 非 bwd |
| bf16 STE bwd | ~1.27 s 档 | **1/3** | **否决** |
| TF32 STE bwd（`matmul_precision=high`） | **1.42 s**（~1.25×） | **2/3**（seed0 grad 9.9%） | **仅 opt-in** |
| **默认保持** fp32 act codes + fp32 STE | **1.58 s**（~1.39×） | **3/3**（P1.9 / p18） | **产品默认** |

落地：

- act quant **保持** half amax → fp32 除法取 codes（与 P1.9 一致）。
- STE bwd **默认 fp32**；`ANIMA_CONVROT_STE_TF32=1` 可开 TF32（速度换质量，不默认）。
- **W8A8 scale 存 float32**（P1.10 的 CUDA bf16 scale 仅限 W8A16）。
- 文档/WebUI **不**把 1.27 s 当承诺数字。

JSON：`convrot_ckpt_w8a8_p111*.json`（**p111e 恢复 3/3**）、`convrot_mem_speed_w8a8_compile_p111*.json`。

### G.16 Phase 1 平台期（2026-07-25）

**产品默认（速度 + 显存）**

| 项 | 值 | 依据 |
| --- | --- | --- |
| `base_compute` | `w8a16_convrot`（实验开启时） | free-base peak **~4.11 GB** vs bf16 **~4.95** |
| `convrot_scope` | `mlp` | compile **~1.05×** bf16（`p112`） |
| `convrot_group_size` / hadamard | `256` / `sylvester` | 兼容 prequant；质量 2/3 |
| `torch_compile` | 建议开 | 吃掉多数 Python 税 |
| 显存优先 | `scope=all` | **~3.43 GB** / **~1.08×** |
| **替代 KPI** | 同峰抬 `network_dim` / batch | rank：§G.17；batch：§G.19（**all@b2** / **all@r32@b2**） |

**质量 opt-in**

| 项 | 值 | 依据 |
| --- | --- | --- |
| hadamard + group | `regular` + `64` | full-ckpt **3/3**；CLI/WebUI 已接线 |
| W8A8 | 非速度默认 | compile **~1.39×**；gate 3/3；scale **必须 fp32** |

**已否决 / 低 ROI（本机 3080）**

- Triton K-loop / epilogue fusion（microbench）
- 默认 FWHT / int8pack
- W8A8 half act codes、bf16 STE、默认 TF32 STE（破 grad gate）
- 共享 dequant scratch 默认开（peak 回吐、step 中性）
- **P1-J trust-mask STE**：base 冻结时 `org_forward` 不对 `w_q` 反传，LoRA 梯度差来自 **前向 quant 噪声** 而非权重 STE；投入产出不匹配

**残余税（W8A16@compile）** ≈ 4–5% step：`convrot_gemm` dequant+linear + 少量 RHT；再抠需换硬件或真 int8 TC 融合（P2）。

**Phase 1 KPI 结论：** 显存已达成；质量 short-train / regular@64 可用；**step ≤ bf16 非 KPI** 且 3080 上不现实。

### G.17 同显存更大 rank（替代 KPI，2026-07-25）

探针：`convrot_mem_speed_probe.py --lora-rank N --torch-compile`（mlp，6 steps，3080）。

| base | rank | peak GB | sec/step | vs bf16@r4 peak |
| --- | --- | --- | --- | --- |
| bf16 | 4 | **4.95** | 1.13 | 1.00× |
| bf16 | 16 | 4.99 | 1.13 | +0.04 |
| bf16 | 32 | 5.04 | 1.14 | +0.09 |
| **W8A16 free** | 4 | **4.11** | 1.18 | **−0.84** |
| W8A16 free | 16 | 4.15 | 1.18 | −0.80 |
| **W8A16 free** | **32** | **4.20** | 1.18 | **−0.75** |

解读：

1. **W8A16@r32 峰值仍比 bf16@r4 低 ~0.75 GB** —— 同机可把 rank 提到 32 仍更省显存。
2. rank 4→32 在 W8A16 上只抬 peak **~0.09 GB**（LoRA 相对 free-base 后的底模很小）；step 几乎不变。
3. 产品叙事优先写 **「同显存训更大 rank」**，而不是「比 bf16 更快」。
4. JSON：`output/tests/convrot_rank_{bf16,w8a16_free}_r{4,16,32}.json`。

**scope=all 抬 rank（compile）：**

| base | rank | scope | peak GB | sec/step |
| --- | --- | --- | --- | --- |
| W8A16 free | 4 | all | **3.43** | 1.29 |
| W8A16 free | **32** | all | **3.63** | 1.29 |
| bf16 | 4 | mlp | 4.95 | 1.14 |
| bf16 | 32 | mlp | 5.04 | 1.13 |

`all@r32` 仍比 `bf16@r4` 低 **~1.3 GB**；rank 税在 free-base 后几乎可忽略。

### G.18 P1.12 frozen weight 不进 `save_for_backward`（2026-07-26）

`w_q` / `w_scale` 是冻结 buffer，fused / W8A8 STE 改为挂 `ctx` 属性，避免 autograd packing。

| 路径 | peak | sec/step | ×bf16 |
| --- | --- | --- | --- |
| W8A16 compile（ctx） | 4.11 GB | 1.178 s | ~1.05 |

速度中性清洁改动；不改变质量门。

### G.19 同显存更大 batch（替代 KPI，2026-07-26）

探针：`convrot_mem_speed_probe.py --batch-size N --torch-compile`（repeat 1 cached sample；4 steps；3080）。

**mlp scope（默认速度档）：**

| base | bs | peak GB | sec/step | under bf16@b1 peak? | samp/s |
| --- | --- | --- | --- | --- | --- |
| bf16 | 1 | **4.95** | 1.14 | — | 0.88 |
| bf16 | 2 | 5.89 | 2.20 | no | 0.91 |
| bf16 | 4 | 7.79 | 4.33 | no | 0.92 |
| W8A16 free | 1 | **4.11** | 1.18 | **yes (−0.84）** | 0.85 |
| W8A16 free | 2 | 5.06 | 2.28 | **no（+0.11）** | 0.88 |
| W8A16 free | 4 | 6.97 | 4.46 | no | 0.90 |

mlp free-base 省下的 ~0.84 GB **不够** 在同峰下把 microbatch 从 1 翻到 2（激活/激活梯度主导 batch 税）。

**scope=all（显存优先档）：**

| base | bs | peak GB | sec/step | under bf16@b1? | samp/s |
| --- | --- | --- | --- | --- | --- |
| W8A16 free | 1 | **3.43** | 1.29 | yes | 0.77 |
| **W8A16 free** | **2** | **4.42** | 2.49 | **yes（−0.53）** | **0.80** |
| W8A16 free | 4 | 6.36 | 4.86 | no | 0.82 |

**组合：all + r32 + b2 仍进 bf16@b1 峰：**

| config | peak GB | sec/step | under bf16@b1? |
| --- | --- | --- | --- |
| all@r32@b1 | 3.63 | 1.30 | yes |
| **all@r32@b2** | **4.62** | 2.49 | **yes（−0.33）** |

解读：

1. **「同显存更大 batch」只在 `scope=all` 上成立**（b=2 peak 4.42 < bf16@b1 4.95）。
2. mlp 默认档更适合「同显存更大 rank」（§G.17），不适合同峰翻 batch。
3. batch 税近似线性：peak ≈ base_storage + O(bs)；free-base 砍的是 storage，不砍 activation。
4. 吞吐（samp/s）随 bs 略升（~3–5%），不是主收益；主收益是 **同 10GB 卡上可训更大有效 batch / 更大 rank**。
5. bwd `w_q` cast scratch 微测：相对每步 `to(dtype)` 仅 ~0.1 ms、顺序层 peak ~16 MB —— **不实现**（ROI 低于文档成本）。
6. JSON：`output/tests/convrot_bs_{bf16,w8a16_free}_b{1,2,4}.json`、`convrot_bs_all_w8a16_free_b{1,2,4}.json`、`convrot_bs_all_w8a16_free_r32_b{1,2}.json`。

**产品叙事（Phase 1 冻结）：**

| 目标 | 推荐配置 | 依据 |
| --- | --- | --- |
| 速度优先 | mlp + compile + r4–32 | ~1.05× bf16；r32 仍 < bf16@r4 peak |
| 显存 / 更大 batch | **all + compile**；可 **b=2** 且仍 < bf16@b1 | §G.19 |
| 显存 + 表达力 | **all + r32 + b2** | peak **4.62 GB** < 4.95 |
| 质量优先 | regular@64 | gate 3/3 |

### G.20 分辨率 KPI 阻断 + compile mode 残余税（2026-07-26）

**同显存更高分辨率：本机不可测（Anima 不变量）。**

- `post_image_dataset/rokkotsu_goddess` 下 57 对 cache（文件名含 0896×1200 与 1792×2400）**全部**是 `latents_144x112`，`H×W = 16128` token（CONSTANT_TOKEN_BUCKETS 的 4200-family 变体之一经 VAE 压缩后的实际 grid）。
- 像素文件名不同 ≠ DiT token 数不同；native flatten 按 token-count family 复用图。
- 因此「同峰训更大分辨率」在 **当前 cache + bucket 表** 上 **没有可区分的探针轴**。要测需改 bucket / 真不同 token family 的 cache（超出 Phase 1 micro-opt）。
- 探针仍加了 `--min-latent-tokens` / `--cache-index` 以备未来多 family cache。

**`torch.compile` mode 不关合残余税：**

| mode | bf16 sec | W8A16 sec | ×bf16 | 备注 |
| --- | --- | --- | --- | --- |
| default (`None`) | 1.133 | 1.180 | **1.042** | 训练常用 |
| `max-autotune-no-cudagraphs` | 1.103 | 1.148 | **1.041** | 两端各快 ~2.7%；**相对税不变** |
| `max-autotune` / `reduce-overhead` | — | — | — | CUDAGraph overwrite，与 grad-ckpt 训练 step 不兼容 |

结论：autotune 是 **通用 compile 旋钮**，不是 ConvRot 专属 ROI；残余 ~4% 仍是 dequant+RHT 相对 bf16 TC 的结构税。

**探针工程：**

- `convrot_mem_speed_probe.py` 自动挑选 CC≥7.5 GPU（避免 PCI 序下 GTX 960 当 cuda:0）。
- `--compile-mode`、`--batch-size`、`--lora-rank`、`--min-latent-tokens` 已齐。

JSON：`output/tests/convrot_compile_{default,max-autotune-no-cudagraphs}_{bf16,w8a16_free}.json`。

### G.21 产品变体 `lora-convrot-vram`（2026-07-26）

WebUI / `gui-methods` 自包含变体：

- 路径：`configs/gui-methods/lora-convrot-vram.toml`（本机外置 configs 根需同步一份）
- 默认：`base_compute=w8a16_convrot`、`convrot_scope=all`、`network_dim=32`、`group=256` sylvester
- 叙事：同显存更大 rank / 可尝试更大 batch（§G.17–G.19）；**不是**速度档
- 标签：`LoRA + ConvRot W8A16 (VRAM)` / 中文「LoRA + ConvRot W8A16 显存档（实验）」

```bash
# 合并检查（METHODS_SUBDIR 选 gui-methods）
METHODS_SUBDIR=gui-methods .venv/bin/python tasks.py print-config \
  METHOD=lora-convrot-vram PRESET=default | rg convrot
```

**与 `blocks_to_swap` 叠用（2026-08-04 residency 修复后）：**

- 训练顺序仍是 `enable_block_swap` → ConvRot free-base → 首次 prepare。
- free-base 把 patched `Linear.weight` 置 meta；offloader **跳过**
  `_convrot_weight_freed` / meta（`is_weight_swap_excluded`），只 master **残余**
  冻结权重（adaln 等）。CPU masters 实测约 **0.33 GiB**（scope=all）。
- ConvRot quantized weight/scale 由实际 base `Linear` 下的 runtime-only carrier 持有；
  carrier 属于对应 DiT block，并复用 frozen-weight swap protocol 在 CPU/GPU 间迁移。
- carrier 不注册为 parameter/buffer，不进入 optimizer 或 `state_dict`；普通 `.to()` 与
  offloader-owned placement 的边界由 `BlockSwapManagedTensor` 管理。
- `block_swap_transfer_dtype=int8` 与 ConvRot **仍硬互斥**。
- 主 VRAM 叙事仍是 `scope=all` + 更大 rank/batch + GC；swap 是残余路径，不是速度档。
- 热测见 §G.22。

### G.22 `scope=all` × GC × `blocks_to_swap=20`（历史方案 A）

探针：`scripts/experiments/convrot_mem_speed_probe.py`（`--blocks-to-swap`、
`--no-gradient-checkpointing`、host RSS）。顺序：`enable_block_swap` →
LoRA apply → ConvRot free-base → `switch_block_swap_for_training` → 步进。
平台：RTX 3080，`w8a16` free-base，`scope=all`，group 256 sylvester，r4，eager，6 step。

#### G.22.1 修复前（硬挂）

| 组合 | GC | swap | status | 备注 |
| --- | --- | --- | --- | --- |
| all+GC+swap20 / all+swap20 | ± | 20 | **error** | `_capture_cpu_master(meta)` → `NotImplementedError` |
| all+GC | on | 0 | ok | ~3.43 GB / ~1.75 s |

JSON：`output/tests/convrot_stack_all_*.json`。

#### G.22.2 方案 A 后（skip free-base / meta）

代码：`library/runtime/device.py::is_weight_swap_excluded`、
`block_swap_masters._can_swap_frozen_weight_to_cpu` / `_ensure_weight_on_device`、
`warn_convrot_blocks_to_swap` 文案更新。

| 组合 | GC | swap | status | math | peak VRAM | sec/step | host RSS | 相对 all+GC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **all+GC+swap20** | on | 20 | **ok** | finite | **3.17 GB** | **2.04 s** | **2.33 GB**（Δ+1.60） | peak **−0.27 GB**；速度 **1.16×**；RSS **+0.75 GB** |
| **all+GC** | on | 0 | **ok** | finite | **3.43 GB** | **1.76 s** | **1.57 GB**（Δ+0.84） | 基线 |
| **all+swap20**（无 GC） | off | 20 | **OOM** | — | peak 尝试 ~9.3 GB | n/a | ~2.14 GB | 无 GC 激活值爆炸；**不要**关 GC 叠 swap |

同 seed 一步 loss：swap vs no-swap `loss_rel≈4e-4`（对齐）；`grad_norm` 相对差 ~0.20（步进噪声/设备布局，非 gate 失败）。
masters 日志：`Block swap frozen CPU masters prepared: 0.33 GiB across 28 blocks`。

JSON：`output/tests/convrot_stack_a_all_{gc_swap20,gc,swap20}.json`。

**数学（all+GC，无 swap 对照）**：`convrot_checkpoint_probe --scope all` seeds 0/1/2 →
gate 2/3（syl@256 历史水平）。JSON：`output/tests/convrot_ckpt_all_gc.json`。

**四维结论（方案 A）**

| 轴 | all+GC+swap20 vs all+GC |
| --- | --- |
| 数学 | 可跑且 loss 对齐；有限 loss/grad |
| 速度 | **更慢**（~1.16× sec/step，swap 税） |
| 显存 | **略省**（3.17 vs 3.43，约 −0.27 GB；残余 adaln 等） |
| 主机内存 | **更高**（RSS ~2.33 vs 1.57；0.33 GiB masters + pin 开销） |

产品建议：

1. **默认仍 `all+GC`、`blocks_to_swap=0`**（速度/简单/主 VRAM 叙事）。
2. **极限再抠 ~0.3 GB** 可开 `blocks_to_swap`（须 **GC on** + transfer **bf16**）；接受更慢与更高 host RSS。
3. **禁止** `all+swap` 且关 GC（本机 OOM）。
4. 本节数据产生于 quant payload residency 修复前；当前行为和结论以 §G.23 为准。

### G.23 block-owned payload residency × `blocks_to_swap=0/12/26`（2026-08-04）

实现：`library/runtime/block_swap_payload.py`、`library/runtime/convrot/apply.py`、
`library/runtime/offloading.py`。探针新增 `--blocks-to-swap-grid 0,12,26`，并记录
apply 后和 step 后的 payload bytes/count。RTX 3080，compile，GC on，scope=all，r32，
8-step 独立进程复测：

| 模式 | swap | sec/step | peak VRAM | apply 后 allocated | payload CPU/GPU tensors |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0 | 1.954 | 5.200 GB | 4.031 GB | 0 / 0 |
| W8A16 | 0 | 2.039 | 3.634 GB | 2.402 GB | 0 / 392 |
| W8A8 | 0 | 3.019 | 3.679 GB | 2.466 GB | 0 / 392 |
| BF16 | 12 | 2.849 | 3.669 GB | 2.464 GB | 0 / 0 |
| W8A16 | 12 | 2.748 | 2.776 GB | 1.541 GB | 168 / 224 |
| W8A8 | 12 | 3.988 | 3.093 GB | 1.606 GB | 168 / 224 |
| BF16 | 26 | 3.613 | 1.839 GB | 0.635 GB | 0 / 0 |
| W8A16 | 26 | 3.148 | 1.771 GB | 0.535 GB | 364 / 28 |
| W8A8 | 26 | 4.884 | 2.085 GB | 0.599 GB | 364 / 28 |

共 196 个 patched Linear，每个包含 quantized weight/scale 两个 carrier，即 392 tensors。
计数严格符合 block 布局：swap12 为 `12×7×2=168` 个 CPU、`16×7×2=224` 个 GPU；
swap26 为 `26×7×2=364` 个 CPU、`2×7×2=28` 个 GPU。W8A16 payload bytes 在 swap12
为 CPU/GPU `755,613,696 / 1,007,484,928`，swap26 为
`1,637,163,008 / 125,935,616`。apply 后与 step 后 residency 相同，全部 case
`status=ok`、`math_ok=true`。

结论：residency 修复成立，不再存在“ConvRot payload 全量常驻 GPU”的结构性问题。
代价是交换越重，step latency 越高；W8A16 从 swap0 到 swap26 的 peak 降约 1.86 GB，
但 step 慢约 54%，W8A8 peak 降约 1.59 GB、step 慢约 62%。因此 swap0 是显存允许时
的速度档，swap12/26 是按显存预算换速度；`block_swap_transfer_dtype=int8` 仍禁止。

JSON：

- `output/tests/convrot_residency_ablation_swap0_20260804.json`
- `output/tests/convrot_residency_ablation_swap12_20260804.json`
- `output/tests/convrot_residency_ablation_swap26_20260804.json`

### G.24 SM86 `_int_mm` 4200-token 对齐修复（2026-08-05）

失败任务：`anima缓存/84-sk-20260804-231854-retry-20260805-002201`。RTX 3080，
W8A8、scope=all、rank64、compile、swap26。任务完成 step 1 后在 4200-token family
的 compiled `_int_mm([4200,2048], [2048,6144])` 失败：

```text
CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul
m 6144 n 4200 k 2048
```

这不是 OOM：失败前 peak allocated/reserved 仅 `2.42/2.87 GB`。`config.runtime.toml`
中的 `block_swap_transfer_dtype` 为 `bf16`，block swap 本身也没有报错。根因是 SM86
cuBLASLt INT8 路径要求 activation GEMM 的 M 维按 32 对齐：同卡实测 4032 成功，
4200/4208 失败，4224/4256 成功。原来的 Python `try/except RuntimeError` 在 eager
可回退，但 Inductor 会把 `_int_mm` 提升成外部 kernel，运行期错误无法由该分支捕获。

修复位于 `library/runtime/convrot/gemm.py::int8_mm_scaled`：仅当 `M % 32 != 0` 时，
把 INT8 activation 零填充到下一个 32 倍数，执行 `_int_mm` 后裁回原 M，再做 scale。
4200 因此以 4224 行进入 GEMM，模型边界 shape、数值和梯度 shape 不变；4032 已对齐，
不会额外复制。

验证：

- eager/Inductor `4200×64×64` 与 float reference 对齐。
- 失败日志的精确 compiled shape `4200×2048×6144` 输出 BF16、全 finite。
- 使用失败任务自身 cache 的真实 DiT 单步：latent `[1,16,1,150,112]`，即 4200
  DiT tokens；W8A8/r64/swap26/compile 为 `status=ok`、`math_ok=true`，peak
  `2.31 GB`，payload residency CPU/GPU `364/28`。

探针：`output/tests/convrot_w8a8_4200_fix_20260805.json`。原失败任务只完成 step 1，
没有 checkpoint 或权重，修复后必须 fresh retry，不能 resume。

### G.25 ConvRot × distributed（Accelerate 数据并行）热测（2026-08-06）

冻结后落地的 distributed adapter 训练（`library/training/gradient_sync.py`，
adapter 故意不 DDP-wrap、改为手动 mean-reduce）与 ConvRot 的交互此前从未热测。
代码审计结论：量化 payload 跟随 `weight.device`（无 `cuda:0` 硬编码）、被 free 的
meta base 不进 optimizer / `network.buffers()`、grad sync 只碰 adapter 参数、
FEI/routing 无 collective（per-rank 本地路由是正确数据并行语义）、GlobalRouter 参数
进 optimizer——五项均 SAFE。本节用热测验证集成行为。

新探针 `scripts/experiments/convrot_distributed_probe.py`（torchrun 多进程，复用
`build` 路径加载缓存 latent + 真实 `prepare_network_for_manual_gradient_sync` /
`synchronize_optimizer_gradients`）。每个 rank 喂**同一**缓存样本（等价于 global
batch = world × 重复样本），使 distributed 结果可与单进程参考逐位对比。

```bash
.venv/bin/torchrun --standalone --nproc_per_node=2 \
  scripts/experiments/convrot_distributed_probe.py \
  --mode w8a16 --steps 4 --rank-dim 32 --scope all \
  --json-out output/tests/convrot_dist_w8a16.json
```

2-GPU（GPU0 CMP 90HX + GPU1 RTX 3080，均 sm86；scope=all / rank32 / batch1 / GC on /
4-step）。`manual adapter gradient sync enabled: world_size=2, parameters=392,
buffers=393, mode=cuda-stream-overlap`：

| mode | loss_last | sec/step | peak/rank | vs bf16 | payload_ok | cross_rank_diff |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 0.086509 | 2.28 | 5.23 GB | 1.0× | True | 0 |
| W8A16 | 0.086503 | 2.43 | 5.85 GB | 1.07× | True | 0 |
| W8A8 | 0.086546 | 3.77 | 5.87 GB | 1.65× | True | 0 |

- **数学偏移**：单进程参考（`--no-sync-each-step`）loss_last `0.086505`；2-rank
  mean-reduce 后 `0.086503`（W8A16），差 ~2e-6 浮点噪声。两 rank 间 sync 后
  `cross_rank_param_max_diff = 0`（逐位一致）→ mean-reduce + 一致 AdamW 正确。
- **payload 归属**：两 rank `payload_on_local_device=True`，量化 `w_q`/`w_scale`
  落在各自的 `cuda:0` / `cuda:1`（非全被 broadcast 到 cuda:0）。
- **W8A8 `_int_mm`** 在非默认设备 `cuda:1` 上正确，无 cublasLt/NCCL 报错。
- **速度/显存**：W8A16 速度税 ~1.07×（与单卡一致），NCCL overlap 无可见额外开销；
  peak 见下节说明（本探针 `enable_gradient_checkpointing` 默认不开 unsloth offload）。

证据：`output/tests/convrot_dist_{bf16,w8a16,w8a8}_rank{0,1}.json`、
`convrot_ref_w8a16_nosync_rank0.json`。block_swap + ConvRot + 多进程三者叠加本仓
默认不触发（`lora-convrot-vram.toml` 默认 `blocks_to_swap=0`），如需启用另行热测。

### G.26 apply 期在线量化瞬时峰 + incremental free-base（2026-08-06）

**根因**：`apply_convrot_to_lora_network` 此前在**全部** patched module 量化完之后
才一次性 `free_base_weights_for_patches`。apply 期间 GPU 同时驻留「全部 bf16 base
权重（~3.97GB）」+「累积的全部 int8 payload（~1.9GB）」，形成一次性瞬时峰。

这解释了一个长期的测量口径矛盾：`convrot_mem_speed_probe` 报 W8A16 peak **3.64GB**
（apply+warm 后才 `reset_peak_memory_stats`，丢弃瞬时峰），而
`convrot_short_train_probe` 报 **5.86GB**（加载前 reset、之后从不 reset，把 apply
瞬时峰计入全局峰）。两者都「对」，只是测的不是同一阶段。

**优化（incremental free-base）**：apply 循环内，每个 module 的 int8 payload 装好、
`org_forward` patch 就位后，立即 `free_linear_weight_storage` 释放该 module 的 bf16
base（幂等，0 表示已 free/meta），把「全量共存」收敛为「单 module 共存」。末尾保留
一次幂等的批量 free 作兜底；`freed_modules`/`freed_bytes` 仍按 base-module id 去重、
bytes 幂等累加，metadata 语义不变。纯提前释放，不改任何数值。

| 指标（scope=all/rank32/batch1，GPU0） | 改前 | 改后 |
| --- | --- | --- |
| apply 后 GPU 驻留（`allocated_after_apply_gb`） | 4.03 GB（bf16 base 未释放期） | **2.41 GB** |
| 全局 peak（含 apply 瞬时峰，短训口径） | 5.86 GB | **5.24 GB**（=bf16 全局峰） |
| 训练稳态 peak（mem_speed 口径） | 3.64 GB | 3.64 GB（不变） |
| 数学偏移（loss_last rel vs bf16） | — | 0.0002 |

收益：**apply 瞬时峰 −0.62GB**，ConvRot 加载路径不再比 bf16 多付一次性峰值；稳态
显存优势（offload 下 3.32 vs bf16 4.91GB，省 1.58GB）不受影响。「same-VRAM larger
rank」在真实训练（`lora-convrot-vram.toml` 开 `unsloth_offload_checkpointing`）
依旧成立。offload 与 ConvRot 正交：frozen `w_q`/`w_scale` 挂 `ctx` 普通属性、不走
`save_for_backward`，host offload 只迁移激活。

证据：`output/tests/convrot_short_r32_incrfree/short_train_w8a16_seed0.json`、
`convrot_incrfree_memspeed.json`。测试：`tests/test_convrot_free_base.py::
test_apply_frees_base_weights_incrementally_across_modules`（全套 convrot 91 passed）。

### G.27 本机两代卡说明（2026-08-06）

冻结基线（§G.16–G.24）多在 RTX 3080 上测；本节 §G.25–G.26 部分数据在 CMP 90HX
（同 sm86、同 10GB、无显示输出）上测。二者 compute capability 相同，单卡结论互相
吻合（§G.25 单卡 sweep 复现 §G.23），但绝对 sec/step 因卡个体差异略有出入，跨卡
对比时以相对倍率为准。

### H. P0-C prequant 实现注记

实现：`library/runtime/convrot/prequant.py` + `apply.py` 接线；导出：`scripts/experiments/convrot_export_prequant.py`。

#### H.1 原生格式 `anima_lora_convrot_prequant_v1`

| 项 | 内容 |
| --- | --- |
| 张量 | `{original_name}.weight` int8 `[out,in]`（**已在旋转域**） |
| scale | `{original_name}.scale` float32 `[out]`（per-out absmax） |
| metadata | `format=anima_lora_convrot_prequant_v1`、`group_size`、`rht=sylvester`、可选 `mode` |
| 别名 | 亦接受 `{name}.weight_scale`（Comfy 风格键名 best-effort） |
| 名称前缀 | 自动尝试 `model.` / `diffusion_model.` 去前缀/加前缀匹配 |

**明确不做：** 完整解析 Comfy `MixedPrecisionOps` / 任意社区 INT8-ConvRot 布局；不保证 obsxrver 权重开箱即用（键名兼容了，但 RHT 定义/group 必须一致）。

#### H.2 行为

- `--convrot_weight_source prequant_checkpoint --convrot_prequant_path <file>`
- 文件 `group_size` 与 CLI 不一致时 **strict raise**（可用 apply 参数 `prequant_group_size_strict=False` 警告并用文件值）
- 缺层 / shape 不符：该层 skip；若最终 patched=0 则 raise
- 仍默认 **free base**（meta）；merge 仍拒绝
- **只去掉 apply 期 online weight RHT+quant**；每步 **act RHT 仍在**

#### H.3 测试

| 测试 | 结果 |
| --- | --- |
| `tests/test_convrot_prequant.py` | save/load 往返、weight_scale 别名、apply 与 online 数值一致、free base、缺层/gs 错/缺 path |
| 全量 `tests/test_convrot_*.py` | **58 passed** |

#### H.4 用法

```bash
# 从本机 bf16 DiT 导出 mlp scope
.venv/bin/python scripts/experiments/convrot_export_prequant.py \
  --scope mlp --group-size 256 \
  --out output/tests/convrot_prequant_mlp_g256.safetensors

# 训练时加载（示例）
python tasks.py lora ... --base_compute w8a16_convrot \
  --convrot_weight_source prequant_checkpoint \
  --convrot_prequant_path output/tests/convrot_prequant_mlp_g256.safetensors \
  --convrot_group_size 256 --convrot_scope mlp

# mem/speed 对照（含 prequant cases）
.venv/bin/python scripts/experiments/convrot_mem_speed_probe.py \
  --steps 6 \
  --cases bf16,w8a16_free,w8a16_prequant,w8a8_auto,w8a8_prequant \
  --prequant-path output/tests/convrot_prequant_mlp_g256.safetensors \
  --json-out output/tests/convrot_mem_speed_prequant.json
```

#### H.5 Mem/speed 热测（2026-07-25，RTX 3080，6 steps）

JSON：`output/tests/convrot_mem_speed_prequant.json`
prequant 文件：`output/tests/convrot_prequant_mlp_g256.safetensors`（56 layers, g=256）

| case | peak GB | sec/step | apply wall | vs bf16 step | 备注 |
| --- | --- | --- | --- | --- | --- |
| bf16 | 5.00 | **1.65** | — | 1.0× | 基线 |
| w8a16_free（online） | **4.34** | **2.03** | 0.18 s | 1.23× | A2 后 |
| **w8a16_prequant** | **4.34** | **2.05** | 0.39 s | 1.24× | step ≈ online |
| w8a8_auto（online） | 4.49 | 2.49 | 0.11 s | 1.51× | |
| **w8a8_prequant** | 4.49 | 2.50 | 0.35 s | 1.51× | step ≈ online |

解读：

1. **稳态 step time：prequant ≈ online**（W8A16 2.05 vs 2.03；W8A8 2.50 vs 2.49）——符合「act RHT 仍在、权重 quant 只发生一次」的设计。
2. **apply 时间：prequant 更慢**（读 safetensors + 设备拷贝 ≈0.35–0.39 s vs online RHT+quant ≈0.11–0.18 s）；只影响启动，不影响每 step。
3. **peak 显存：与 free-base online 相同**（仍低于 bf16）。
4. **因此 P0-C 的 KPI 是「可复现加载 / 去掉 apply 期 quant / 与社区权重接口」**，不是 step 加速；当初验收以 unit 为主是因为路线图已写明「不承诺 step 大加速」。

### I. P0-D：regular Hadamard 质量对齐（2026-07-25）

实现：`library/runtime/convrot/rht.py`

| 构造 | 阶 | 列 discrepancy \(\|H^\top 1\|_\infty\) | 入口 |
| --- | --- | --- | --- |
| **sylvester**（默认） | \(2^k\) | \(=n\)（全 1 列） | `ANIMA_CONVROT_HADAMARD=sylvester` |
| **regular**（论文） | \(4^k\)（4/16/64/256/1024…） | \(=\sqrt{n}\)（最小） | `ANIMA_CONVROT_HADAMARD=regular` |

- regular：`H_4` 基 + Kronecker `H_{4^{k+1}}=H_{4^k}⊗H_4`（ConvRot Thm 3.3）
- FWHT **仅**等价 sylvester；regular 强制 dense matmul
- **默认不改**：仍 sylvester + group=256（兼容已有 prequant / 探针基线）

#### I.1 Checkpoint multi-seed（W8A16，seeds 0–2，strict grad≤5%）

JSON：`output/tests/convrot_ckpt_w8a16_{syl_g256,reg_g64,reg_g256,reg_g1024}.json`

| 配置 | seed0 out/loss/**grad** | seed1 | seed2 | 严格 pass | grad_max |
| --- | --- | --- | --- | --- | --- |
| sylvester g=256 | 0.0063 / 0.0009 / **0.0875 FAIL** | PASS | PASS | **2/3** | 0.0875 |
| **regular g=64** | 0.0062 / 0.0000 / **0.0475 PASS** | PASS | 0.0161 / 0.0002 / **0.0509 FAIL** | **2/3** | **0.0509** |
| regular g=256 | 0.0063 / 0.0001 / **0.0841 FAIL** | PASS | PASS | **2/3** | 0.0841 |
| regular g=1024 | 0.0065 / 0.0005 / **0.1487 FAIL** | PASS | PASS | **2/3** | 0.1487 |

W8A8 regular g=256：seed0 **grad_rel≈4.1**（异常大，seed1/2 正常）→ **不推荐** 默认切 regular@256 给 W8A8。

#### I.2 解读与默认策略

1. **输出 / loss**：所有配置全 seed 远低于阈值。
2. **seed0 grad** 仍是最难 seed；regular **不保证**全面碾压 sylvester。
3. **regular g=64** 在本矩阵里 **seed0 首次过 5%**，grad_max 也最低（~5.1%），但 seed2 略超；值得作为 **质量 opt-in**。
4. regular g 越大（256→1024）seed0 grad **变差**（本机）——与「更大旋转窗更好」的朴素预期相反，需更多 seed 才下强结论。
5. **默认保持 sylvester + g=256**；实验可：
   `ANIMA_CONVROT_HADAMARD=regular` + `--convrot_group_size 64`。
6. 若已导出 prequant，切换 hadamard/group **必须重导出**（旋转域不同）。

### Phase 1 支持面

- plain LoRA + `scope=mlp`（默认）
- `online_from_bf16` 权重量化
- `prequant_checkpoint` 加载（原生 v1 + weight_scale 别名）
- quant 后默认释放 base bf16 权重（meta）
- W8A8：真 int8 GEMM（可回退 float）
- 融合 autograd 路径（默认 dense RHT + dequant；FWHT/int8pack opt-in）
- Hadamard：`sylvester`（默认）/ `regular`（opt-in，阶须 \(4^k\)）
- step profiler 探针（P0-A）
- DoRA / 无 `org_module_ref` 变体：跳过或拒绝（禁止 0 patch 静默成功）
- merge / fuse：默认拒绝 ConvRot base

---

## 一句话结论

**ConvRot 是修正「DiT 上裸 rowwise int8 效果差」的正确方向**（group-wise Hadamard 压 outlier）。本仓已自建 W8A16 / W8A8；free-base 后显存可低于 bf16；compile 下 W8A16 ~1.05× bf16。Phase 1 KPI = 显存/质量；**step ≤ bf16 非 KPI**。替代叙事：**同显存更大 rank**（mlp@r32）与 **同显存更大 batch**（仅 `scope=all`@b2 / all@r32@b2）。分辨率同峰本机无独立轴（全 cache 同 token family）。默认 sylvester@256；质量 opt-in regular@64。Triton 门槛未达；**Phase 1 micro-opt 冻结**。

---
## 1. ConvRot 是什么

| 项 | 内容 |
| --- | --- |
| 论文 | [ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers](https://arxiv.org/abs/2512.03673)（Huang et al., arXiv:2512.03673, 2025-12） |
| 本质 | **Group-wise Regular Hadamard Transform (RHT)**；名称里的 Conv **不是**“卷积层量化” |
| 论文目标 | DiT **免重训 PTQ**；主推 **W4A4**；生态扩展出 INT8 ConvRot / W8A8 推理 |
| 运行时 | 权重：旋转后量化存盘；激活：在线旋转 + 动态量化再 GEMM；bias / norm / embedding 等常保留高精度 |
| 模块概念 | `ConvLinear4bit`：旋转 → 量化 → GEMM → 反量化，作为可替换 Linear |

复杂度宣称：全维旋转 \(\mathcal{O}(K^2)\) → group-wise \(\mathcal{O}(K)\)。组大小论文/社区常见 `64 / 256 / 1024`（须能整除层维度）。

### 与 rowwise int8 失败的关系

| 机制 | 问题 |
| --- | --- |
| 裸 rowwise / per-channel absmax | DiT 有强 row-wise + column-wise outlier；少数通道支配 scale，其余通道被压扁 |
| QuaRot 类 full FWHT | 论文指出 Sylvester/FWHT 含全 1 列，可能**放大** DiT row-wise outlier |
| AdaLN | 破坏 LLM 那套“旋转融进相邻 Linear、省 online rotate”的路径 |
| ConvRot group RHT | 组内旋转摊平 outlier；更适配 DiT 局部结构 |

本仓已有旁证：

- `bench/channel_stats/`：Anima frozen base 存在 20–100× DC-bias outlier（LLM.int8 类 outlier feature，不是 attention sink）。
- [`../methods/channel_scaling.md`](../methods/channel_scaling.md)：SmoothQuant 风格 **adapter 侧** 预缩放，**不是** base int8 GEMM。
- 社区 latent 基准（[ComfyUI-INT8-Fast Metrics](https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/Metrics.md)，含 Anima）：

```text
GGUF Q8 > INT8 ConvRot > MXFP8 > FP8 ≥ INT8 Row > INT8 Tensorwise
```

社区定义：

> INT8 ConvRot = row-wise INT8，但在量化前对 weight/activation 做 ConvRot 旋转。

因此：**以前 int8 训练效果差，很大概率不是“int8 本身不行”，而是“裸 rowwise 未处理 DiT outlier”。**

---

## 2. W8A16 / W8A8 语义（训练目标）

| 模式 | Weight | Activation | 生态现状 | 训练含义 |
| --- | --- | --- | --- | --- |
| **W8A16** | int8（旋转后） | bf16 / fp16 | 几乎无人单独作为发布格式 | **第一期训练优先**：权重量化 + 高精度 act |
| **W8A8** | int8 | **动态** int8 | 推理有（Whole W8A8 / native int8 ConvRot） | 真 int8×int8 GEMM；act 每步旋转+量化 |
| **W4A4** | int4 packed | 动态 int4 | 论文主推 + ComfyUI 布局 | 推理向；训练更难 |

必须区分的三条路径：

| 路径 | 实际在算什么 | 是否等于目标 |
| --- | --- | --- |
| **QLoRA** | 存低比特 → **dequant 到 bf16 再 matmul**；通常不量化激活 | 结构可参考，**不是** W8A8 训练 GEMM |
| **本仓 `int8_linear` / block-swap int8** | int8 存储或 CPU master；执行时反量化回执行 dtype 再 `F.linear` / 高精度算子 | **存储/传输压缩**，不是 ConvRot，也不是 int8 GEMM 训练 |
| **真 ConvRot W8A8** | 存旋转 int8 权重；forward 对 act 做 RHT+quant，再 **int8 GEMM** | 目标形态 |

---

## 3. GEMM「融合」指什么

论文与社区说的 fusion 主要是 **推理算子链**：

```text
x (bf16)
  → group RHT
  → quant
  → int8 / int4 GEMM(W_q)
  → dequant
  → y
```

公开生态要点（时间敏感，约 2026-06～07）：

- ComfyUI **v0.27.0**：native **int8 ConvRot** 加载；含 int8 上 apply LoRA / requant 修复（**服务推理**，不是 `train.py`）。
- ComfyUI PR **#14859**：`convrot_w4a4` layout，`convrot_groupsize` 默认 256。
- Anima 推理加速示例：[ComfyUI-AnimaTurbo](https://github.com/TheLegendOfKitty/ComfyUI-AnimaTurbo)（shared rotated act cache、warp-FHT quantize、fuse QKV）。
- 转换工具方向：[`convert_to_quant`](https://github.com/silveroxides/convert_to_quant) 一类
  `--int8 --scaling_mode row --convrot --convrot-group-size {64,256,1024} --comfy_quant`。
- 公开 Anima 权重示例：[obsxrver/ComfyUI-Native-INT8_ConvRot](https://huggingface.co/obsxrver/ComfyUI-Native-INT8_ConvRot) 中的 `anima-preview3-base-int8-ConvRot`。

**没有**公开的「冻结 base ConvRot + LoRA 训练」fused kernel。
训练第一期可用 **可微 / STE 的 fake path** 验数值；速度第二期再接 Triton / comfy-kitchen 类 kernel。

---

## 4. 相关方法对照

| 方法 | 训练？ | 机制 | 对 Anima 训练的启示 |
| --- | --- | --- | --- |
| 裸 rowwise int8 | 可做但 Anima 差 | 无旋转 | 已踩过的失败模式 |
| SmoothQuant | 多为 PTQ | scale 迁移 outlier | 本仓 channel_scaling 是 adapter 侧近亲 |
| LLM.int8 / bitsandbytes | 推理 / 优化器 | outlier 通道混精；AdamW8bit | 优化器已有；**不是** ConvRot |
| QLoRA | 训练 | 存储 4bit + dequant bf16 GEMM | 冻结 base + 高精度 adapter 结构可参考 |
| QuaRot | PTQ | full Hadamard | DiT row-wise 可能更差；Anima 基准弱于 ConvRot |
| SpinQuant | PTQ 学旋转 | Cayley SGD | 推理向；非本仓训练栈 |
| **ConvRot 论文** | **仅 PTQ** | group RHT | 算法源 |
| **Comfy INT8 ConvRot** | 推理 + LoRA apply | 原生 loader | 有 Anima 量化权重 |
| **Ostris ARA** | 实验训练 | 冻结 quant base + 小 LoRA 补误差；另有 QAT 向实验 | **最接近「训练」的公开方向** |
| HSWQ | PTQ | 敏感层保护 + 余下 ConvRot | 打包推理用 |

### 不要当既定事实写入设计的说法

对抗校验中被驳回或不可靠的表述（勿写入默认承诺）：

- 某社交帖对 FLUX Whole W8A8 的具体 VRAM 数字（如固定 16.30 GiB / −32.3%）
- HSWQ 营销向 SSIM / 体积数字（未过校验）
- SpinQuant「相对 QuaRot 45.1%」一类原句营销表述（未过校验）

---

## 5. 训练支持：已证实 vs 缺口

### 已证实（高置信）

1. ConvRot 是 DiT 向 rotation **PTQ**；主证据在推理 W4A4 / W8A8。
2. 运行时：旋转量化权重存盘 + 激活在线旋转动态量化。
3. Comfy 生态成熟于 **load / forward / serialize**，不是 training loop。
4. QLoRA ≠ 训练期 W8A8 GEMM。
5. Anima **推理** INT8 ConvRot 质量显著优于裸 INT8 Row（社区 latent 基准）。
6. Anima `model_channels=2048`、`mlp_ratio=4 → 8192`，**64 / 256 / 1024 均可整除**，group size 约束友好。

### 未证实 / 工程缺口

1. **没有主源**给出 ConvRot vs rowwise 在 **LoRA 训练 loss / 画质** 上的对照实验。
2. **W8A16 训练**不是生态标准选项，需本仓定义。
3. LoRA 与 **group 旋转边界**、**requant 后权重域** 在训练侧未标准化。
4. Anima 特有约束需单独验证：5D latent、max-padded TE sink、FEI / Hydra router、`compile_blocks` after `apply_to`。
5. AdaLN / modulation / final layer 应排除在 int8 候选外（论文与社区 arch filter 一致）。

### 与本仓现状对照

| 已有 | 缺 |
| --- | --- |
| rowwise / per-channel 存 int8（`library/runtime/int8_linear.py`） | group RHT |
| block-swap CPU master `transfer_dtype=int8`（实验） | 在线 act 旋转 + quant |
| AdamW8bit 等 bnb 优化器 | int8 GEMM 训练 wrapper |
| channel_scaling（adapter 侧） | W8A16 / W8A8 配置面、WebUI、metadata |
| int8 probe / gate 习惯 | 与 LoRA `org_forward` 正确域残差相加 |

---

## 6. 建议的产品语义（若落地）

不要把「加载 Comfy int8-ConvRot 权重做推理」和「训练期 W8A*」混成一个开关。

### 分期

**Phase 0 — 对齐（可选）**

- 用 `convert_to_quant` 或公开 `anima-*-int8-ConvRot` 做推理数值对齐。
- 确认 group size、layer 排除（AdaLN / final）与 Comfy 布局一致。

**Phase 1 — W8A16 训练（优先）**

- 冻结 base 候选 Linear：`RHT + int8 weight`。
- 激活保持 bf16 → W8A16 GEMM（或 dequant 权重 × bf16 act 作为数值 baseline）。
- LoRA 在 **原空间 / dequant 后** 残差相加：

```text
y = base_w8a16(x) + lora(x)
```

- STE 或 stop-grad 穿过 frozen quant 权重；adapter 正常反传。
- scope：先 `mlp`，再 attention projection；排除 AdaLN / modulation / final。

**Phase 2 — W8A8 训练**

- 同 group RHT 后对 act **动态 quant**。
- int8×int8 GEMM + dequant。
- 独立 gate：output L2、adapter grad、短训 sample。
- 默认 **off**，仅实验开关。

### 第一期明确不做

- 把现有 `Int8FrozenLinear` 贴牌成 ConvRot。
- 只改 `block_swap_transfer_dtype=int8` 当训练质量方案。
- 默认写入 `configs/base.toml`。
- 全层含 AdaLN 一起 int8。
- 把 Comfy 推理 quant 权重格式直接当训练图。

### 配置草图（尚未实现）

```toml
# 建议语义，当前 main 不存在这些键
base_compute = "bf16" | "w8a16_convrot" | "w8a8_convrot"  # 默认 bf16
convrot_group_size = 256   # 64 | 256 | 1024
convrot_scope = "mlp"      # mlp | attention | all（仍排除 AdaLN）
convrot_weight_source = "online_from_bf16" | "prequant_checkpoint"
```

### 集成触点（实现时）

1. `network.apply_to` + `load_weights` **之后**，`compile_blocks` **之前** patch base / `org_forward`。
2. 不可 `replace Linear` 破坏 LoRA monkey-patch 链（对齐现有 `patch_lora_frozen_base_forwards_with_int8` 的思路，但换成 ConvRot 算子）。
3. 保存：默认只存 adapter；base quant 状态写 metadata（例如 `ss_base_compute=...`）；merge 默认拒绝或需先 dequant。
4. 测试：RHT 正交/可逆、shape、STE 梯度、toy bf16 vs W8A16、可选加载公开 Anima INT8-ConvRot 做推理对齐。
5. 评估必须以 **adapter grad + 短训 sample** 为准，不能只看权重反量化 L2（旧 int8_linear audit 不够）。

### 三条战略（按投入）

| 路线 | 内容 | 收益 | 风险 |
| --- | --- | --- | --- |
| A. 仅推理/服务 | 文档 + 可选加载 int8-ConvRot 推理 | 低成本对接社区 | 不解决 int8 **训练** |
| B. ARA / 冻结 quant base + LoRA | 预量化 base 上训 adapter | 有社区先例；可降显存 | 可能是补 quant 误差，而非真低精 GEMM 训练 |
| C. 真 W8A16→W8A8 训练路径 | 自研 ConvRot Linear + 训练 hook | 直接打 rowwise 失败根因 + 可冲 GEMM | Tier 1.5/2；kernel / compile / 数值工作量大 |

目标若是 **W8A16 + W8A8 训练支持**，应走 **C**；B 可作对照实验。

---

## 7. 开放问题

1. 第一期是否只实现 W8A16，接口预留 W8A8？（建议：接口齐、实现先 W8A16）
2. 权重来源：在线从 bf16 做 RHT+quant，还是加载社区 `*-int8-ConvRot.safetensors`？
3. 成功标准：短训 sample 接近 bf16 LoRA，还是显存 / 速度 KPI？
4. 范围：仅 LoRA family，还是 Hydra / T-LoRA / FEI 全开？（router 与 online act rotate 共存需单独设计）
5. 是否存在开源 **训练期** W8A16/W8A8 GEMM + STE 穿过 ConvRot 冻结权重的实现，及其在 V100/A100 级上的实测收益？

---

## 8. 实现顺序（历史清单；核心路径已落地）

1. [x] 确认 Phase 1 范围与权重来源。
2. [x] 模块：`library/runtime/convrot/`（RHT、quantize、W8A16/W8A8 Linear wrapper）。
3. [x] 训练钩子：`apply_to` 之后 patch frozen base / `org_forward`。
4. [x] 配置 / CLI / WebUI MVP 实验项（默认 `bf16`，UI 标明实验）。
5. [x] 测试：unit + probe gate；公开权重推理对齐仍为 best-effort。
6. [~] 文档：本页为可运行实验说明；稳定后再考虑迁 `docs/methods/`。

**当前默认不做**：改训练默认配置、长训、下载大模型、推送。

---

## 9. 主要来源

- [arXiv:2512.03673 ConvRot](https://arxiv.org/abs/2512.03673)
- [FLUX.1-dev-ConvRot 模型卡](https://huggingface.co/SearchingMan/FLUX.1-dev-ConvRot)
- [ComfyUI v0.27.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.27.0) / [PR #14859](https://github.com/Comfy-Org/ComfyUI/pull/14859)
- [Anima INT8 ConvRot 权重](https://huggingface.co/obsxrver/ComfyUI-Native-INT8_ConvRot)
- [INT8 质量基准（含 Anima）](https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/Metrics.md)
- [convert_to_quant](https://github.com/silveroxides/convert_to_quant)
- [ComfyUI-AnimaTurbo](https://github.com/TheLegendOfKitty/ComfyUI-AnimaTurbo)
- [QLoRA](https://arxiv.org/abs/2305.14314) / [SmoothQuant](https://arxiv.org/abs/2211.10438) / [QuaRot](https://arxiv.org/abs/2404.00456) / [SpinQuant](https://arxiv.org/abs/2405.16406)
- 本仓：[`../findings/anima_int8_base_linear_audit.md`](../findings/anima_int8_base_linear_audit.md)、`library/runtime/int8_linear.py`、`bench/channel_stats/`
