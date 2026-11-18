# ConvRot W8A* 训练后续优化路线图

状态：提案 / **半活跃（P0-A/A2/C/D 完成；默认仍 sylvester）**  
适用版本：当前 main（2026-07-24 核心路径；2026-07-25 prequant + regular Hadamard）  
日期：2026-07-24 / 2026-07-25  
入口命令：

```bash
# 当前实现验证
timeout 60 .venv/bin/python -m pytest tests/test_convrot_*.py -q
.venv/bin/python scripts/experiments/convrot_mem_speed_probe.py --steps 6
.venv/bin/python scripts/experiments/convrot_step_profile_probe.py \
  --cases bf16,w8a16_free,w8a8_auto \
  --json-out output/tests/convrot_step_profile.json

# prequant 导出 + 训练加载
.venv/bin/python scripts/experiments/convrot_export_prequant.py \
  --scope mlp --group-size 256 \
  --out output/tests/convrot_prequant_mlp_g256.safetensors
python tasks.py lora ... --base_compute w8a16_convrot \
  --convrot_weight_source prequant_checkpoint \
  --convrot_prequant_path output/tests/convrot_prequant_mlp_g256.safetensors

# 训练仍须显式开启；默认 bf16 不变
python tasks.py lora ... --base_compute w8a16_convrot
python tasks.py lora ... --base_compute w8a8_convrot
```

相关代码：

- `library/runtime/convrot/`（rht / quant / w8a16 / w8a8 / free_base / gemm / fused / prequant / apply）
- `library/training/bootstrap.py::maybe_apply_convrot_base`
- 探针：`scripts/experiments/convrot_{equivalence,checkpoint,short_train,mem_speed,step_profile,export_prequant}*.py`

相关文档：

- 落地规格：[`../../_archive/docs/proposal/convrot_w8a_training_plan.md`](../../_archive/docs/proposal/convrot_w8a_training_plan.md)
- 实验与实测：[`../experimental/convrot_int8_training.md`](../experimental/convrot_int8_training.md)
- 旧 int8 审计：[`../findings/anima_int8_base_linear_audit.md`](../findings/anima_int8_base_linear_audit.md)

> **一句话：** 核心 W8A16/W8A8 路径已可跑且 **省显存**；Python 级 RHT+quant+GEMM「融合」在本机 **未加速**。  
> 本文件冻结 **后续优化 ROI 排序** 与成功标准，避免把 4090 推理数字误写成 3080 训练预期。

---

## 0. 现状锚点（本机 RTX 3080，2026-07-24）

| 路径 | peak VRAM | sec/step | 备注 |
| --- | --- | --- | --- |
| bf16（eager） | ~5.0 GB | **~1.43 s** | P1.5 热测 |
| free-base W8A16（eager, P1.5） | **~4.14 GB** | **~1.51 s（1.05×）** | 推荐默认（无 compile） |
| bf16 + torch.compile | ~4.95 GB | **~1.14 s** | 2026-07-25 compile 热测 |
| free-base W8A16 + compile | **~4.11 GB** | **~1.18 s（1.036×）** | **训练常用路径推荐** |
| free-base W8A8 + compile | **~4.17 GB** | ~1.66 s（1.45×） | 仍慢在 int8 GEMM |
| free-base W8A16（P1 前 / A2） | ~4.34 GB | ~1.74–1.99 s | 历史锚点 |
| 误默认 FWHT + int8pack | ~4.55 GB | **~47.6 s** | 已废弃为默认 |

已落地（参见规格书 M1–M5 + 任务 1–3）：

- group RHT + online_from_bf16 quant + LoRA `org_forward` 补丁  
- quant 后 free base（meta 占位）  
- W8A8：`torch._int_mm` + float 回退 + STE  
- 单 `autograd.Function` 融合接口；默认 **dense RHT + dequant**（FWHT / int8pack 仅 opt-in）

质量门（摘要）：

- full-checkpoint 1-step：W8A16 / W8A8 均为 **2/3** 严格 gate（seed0 grad 最差）  
- 20-step short-train：last_loss rel vs bf16 ≪ 1%；sample 像素差小  

**结论：** 双份权重显存问题已解决；**step time 未解决**。瓶颈不在「少一层 Python 图」，而在 **online 激活旋转 + 非 Tensor-Core 友好路径**。

JSON 证据：

- `output/tests/convrot_mem_speed.json`  
- `output/tests/convrot_mem_speed_fused.json`（旧慢默认对照）  
- `output/tests/convrot_mem_speed_fused_dense.json`（当前默认）  
- **`output/tests/convrot_step_profile.json`（P0-A 饼图，2026-07-24）**

### 0.1 P0-A 结论（已完成）

| 指标 | bf16 | w8a16_free | w8a8_auto |
| --- | --- | --- | --- |
| sec/step | 1.73 | 3.06（1.77×） | 2.52（1.46×） |
| peak GB | 4.99 | **4.37** | **4.49** |
| convrot_rht % | — | **1.4** | **1.7** |
| convrot_gemm 标记 % | — | **12.0** | **11.7** |
| gemm_generic % | 51.7 | 52.6 | 37.8 |

自动分支：`fix_w8a16_keep_bf16_compute`  
- ConvRot 链合计 **~14% ≪ 50%** → **不** 开 P2-K Triton  
- W8A16 top 标记是 `convrot::gemm_dequant_linear`；kernel 侧出现 **fp32 sgemm**，相对 bf16 TC 是额外税  
- RHT 不是主瓶颈；FWHT/融合 RHT ROI 低  

**下一刀（更新后的队列）：**

1. ✅ **P0-A2 已做**：W8A16 dequant linear 保持 bf16/fp16 → sec/step **3.06 → 1.99**（~1.23× bf16）  
2. ✅ **P0-C 已做**：`prequant_checkpoint` 原生 v1 + 导出脚本（去掉 apply 期 online weight quant；act RHT 仍在）  
3. ✅ **P0-D 已做**：regular Hadamard（Kronecker \(4^k\)）+ multi-seed 对照；**默认仍 sylvester@256**；质量 opt-in：`ANIMA_CONVROT_HADAMARD=regular` + `group=64`  
4. 仍不做默认 Triton / FWHT / int8pack  
5. 可选：A2 后 re-profile / P1-G 缩 scope / 更多 seed 稳 regular@64 默认切换决策

---

## 1. 文献 / 生态与本仓目标的错位

### 1.1 ConvRot 论文是推理 PTQ，不是训练

来源：[arXiv:2512.03673](https://arxiv.org/abs/2512.03673)

- 目标：DiT **免重训 W4A4 推理**；`ConvLinear4bit` = 旋转 + 量化 + GEMM + 反量化  
- FLUX.1-dev @ **RTX 4090 推理**：约 **2.26× 更快、4.05× 更省显存**（相对 bf16 **推理**）  
- **AdaLN** 破坏 LLM 那套「把旋转融进相邻权重、几乎不 online rotate」的技巧 → 激活侧 online 旋转在 DiT 上很难消掉  
- 论文强调 **regular Hadamard**（组大小常为 **4 的幂**），并指出 Sylvester/FWHT 的全 1 列可能 **放大** DiT row-wise outlier  
- 「融合」主要是复用成熟 quant GEMM 管线，不是「训练 + STE 的 Triton 银弹」

**约束：** 论文速度数字 **不得** 直接写成「3080 上 frozen base + LoRA 训练 step」预期。

### 1.2 生态权重 = 离线 prequant，仅推理

| 资源 | 角色 |
| --- | --- |
| [obsxrver/ComfyUI-Native-INT8_ConvRot](https://huggingface.co/obsxrver/ComfyUI-Native-INT8_ConvRot) | Anima INT8-ConvRot 权重；Comfy 推理加载 |
| [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) | bf16 → Comfy int8/ConvRot 离线转换 |
| [ComfyUI native INT8 PR #14636](https://github.com/Comfy-Org/ComfyUI/pull/14636) | 推理加载/forward；训练路径基本不在范围 |
| [ComfyUI-INT8-Fast Metrics](https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/Metrics.md) | Anima INT8-ConvRot 推理质量排序 |

本仓 **P0-C 已实现** 原生 `anima_lora_convrot_prequant_v1` 加载/导出（`library/runtime/convrot/prequant.py`）。  
接上这些权重 **省的是 apply 期 online weight quant**，**不能单独** 消掉每步 act RHT。  
社区 Comfy INT8-ConvRot 仅 best-effort 键名兼容（`weight`+`weight_scale`），**不保证** obsxrver 权重开箱数值一致。

### 1.3 旋转量化族：多数 LLM 推理 / QAT

| 系统 | 可借鉴 | 对本仓限制 |
| --- | --- | --- |
| [QuaRot](https://arxiv.org/abs/2404.00456) / [code](https://github.com/spcl/QuaRot) | 权重离线旋转 + 少量 online HT；CUTLASS INT4 | LLM 计算不变性；DiT AdaLN 难完整吸收 |
| [QuEST](https://arxiv.org/abs/2502.05003) / [code](https://github.com/IST-DASLab/QuEST) | HT + 改进梯度估计；推理核在 4090 有层加速 | **QAT 训整模**，不是冻结 base + LoRA |
| 更快 Hadamard 核（HadaCore / FlashHadamard 类） | 加速 RHT 算子 | 本机 dense matmul 已快于 naive FWHT |
| kernel-embedded Hadamard+GEMM | 真融合方向 | 训练可用、DiT shape 对齐的开源极少 |

---

## 2. 本机未加速的原因（按优先级）

1. **训练 matmul 的 M 往往偏小**  
   DiT token×batch 常吃不满 int8 Tensor Core；`_int_mm` / pack 还带 layout 与 fallback 成本。本机已见：`_weight_int8pack_mm` ≪ dequant + `F.linear`。

2. **bf16 Tensor Core 在 Ampere 消费卡上已经很强**  
   W8A16 若最终 dequant 再 `F.linear`，等于 **多付 RHT + dequant**，几乎不可能更快。

3. **online act RHT 是固定税**  
   DiT AdaLN 导致 LLM 式 offline 吸收不完整。Python 级 fusion 消不掉 HBM 往返。

4. **W4A4 推理加速 ≠ W8A\* 训练加速**  
   4090 上 INT4 推理叙事，换到 3080 上 W8 + 反传 + grad checkpoint + LoRA，完全另一张表。

5. **实现与论文差异（次要但要记）**  
   当前默认 **Sylvester** group RHT；论文推 **regular Hadamard**。这首先影响 outlier/质量，不是当前 ~1.7× 变慢的主因。

---

## 3. 成功标准冻结（防预期漂移）

| 目标 | 3080 现实性 | 是否当前 KPI |
| --- | --- | --- |
| peak VRAM < bf16 | **已达成** | **是** |
| short-train sample / loss 接近 bf16 | **大体达成** | **是** |
| 默认 bf16 零回归 | **保持** | **是** |
| step time ≤ bf16 | **当前不现实**（需 P2 + 可能换卡） | **否**（非 Phase 1 KPI） |
| 同显存训更大 rank / 分辨率 | **高价值替代 KPI** | 建议作为速度失败时的产品叙事 |

**禁止：** 把「融合已做」写成「训练已加速」；把 4090 W4A4 推理 2.26× 写进训练用户文档当预期。

---

## 4. ROI 排序路线图

### P0 — 高 ROI / 低–中成本（下一步默认队列）

| ID | 方向 | 预期 | 验收 |
| --- | --- | --- | --- |
| **A** | nsys/ncu 或 `torch.profiler` 拆 short-train step（bf16 / w8a16_free / w8a8） | 决定后面砍哪块 | ✅ 2026-07-24：`convrot_step_profile.json` + experimental §G |
| **B** | 保持默认 dense RHT + dequant；FWHT / int8pack 仅 env opt-in | 避免回归 | ✅ 已保持 |
| **C** | 实现 `prequant_checkpoint` 加载（原生 v1 + weight_scale 别名） | 去掉 apply 期 online weight quant；启动与一致性更好；**不承诺** step 大加速 | ✅ 2026-07-25：`prequant.py` + export 脚本 + `tests/test_convrot_prequant.py` |
| **D** | 质量对齐：regular Hadamard + group ∈ {64,256,1024}（4 的幂）对照 | 可能改善 seed0 grad；为混精铺路 | ✅ 2026-07-25：实现 + multi-seed；regular@64 seed0 PASS；默认仍 sylvester |
| **E** | 文档/产品口径：Phase 1 KPI = 显存 / 可训，速度非 KPI | 防错误预期 | ✅ 同步中 |
| **A2** | **（profile 新增）W8A16 dequant linear 保持 bf16/fp16 计算** | 收回 fp32 sgemm 税 | ✅ 2026-07-24：mem_speed 1.99s/step（原 ~3.0）；见 `convrot_mem_speed_bf16_compute.json` |
| **A3** | 单 op microbench + fusion 上界（bf16 / dequant / predequant / int_mm±scale / bwd chunk；**不写 Triton**） | 决定是否开 P2 K/epilogue | ✅ 2026-07-25：`convrot_fusion_microbench.py` + JSON；**P2 不开**；见 experimental §G.5 |

### P1 — 中 ROI / 中成本（有 profile 后再开）

| ID | 方向 | 触发条件 | 预期 |
| --- | --- | --- | --- |
| **F** | 混合精度层：敏感层 bf16/W8A16，大 MLP W8A8（论文 ~20% 敏感层留高精） | out/grad 紧、大层占算力 | ✅ 2026-07-25 实现；热测 mixed 1.96s / 4.49GB（见 `convrot_mem_speed_p1.json`）— **不** 优于 full W8A16 默认 |
| **G** | 缩小 patch 范围：只 patch 最大 `in_features` 的 Linear | profile 显示小层 RHT 固定开销大 | ✅ 实现 + 热测：largest/min4096 ≈1.62s（1.13×）、peak 4.77GB（回吐 ~0.4GB） |
| **H** | compile 友好：固定 shape dense RHT + `F.linear` 吃满 `torch.compile` | compile_blocks 后仍慢 | ✅ 2026-07-25：apply 预计算 `_convrot_hadamard` 并传入 fused；compile 下收益未单独 re-profile |
| **I** | 3080 上 W8A16 **不** 默认 pack GEMM；W8A8 仅大 M 强制 `_int_mm` | microbench 已否 pack | 防 47 s/step 回归 |
| **J** | 梯度数值：trust-mask 类（QuEST）替代朴素 STE | seed0 grad_rel 过大 | **⏸ 搁置**：冻结 base 时 `w_q` 无 grad；LoRA 差来自前向 quant 噪声（见 experimental §G.16） |

### P2 — 低–不确定 ROI / 高成本（明确「冲速度」才开）

| ID | 方向 | 条件 | 风险 |
| --- | --- | --- | --- |
| **K** | Triton 真融合：`group_RHT + absmax + int8 GEMM` 单核（前向）；反传 STE 另核 | profile 链 >50% **且** microbench 证明 free-fusion 体能 ≤~1.15× bf16（A3：**未满足**，bucket-M int_mm-only 已 ~1.31×） | 工程量大；3080 仍可能输 bf16 TC；**当前默认不开** |
| **L** | 移植 CUTLASS / QuaRot / QuEST 推理核到训练 `org_forward` | 形状对齐、有维护意愿 | 许可证 / API / 反传缺失 |
| **M** | 换硬件再测：4090 / A100 / Hopper+ | 有卡 | 3080 结论不外推 |
| **N** | W4A4 推理路径或「训完再 PTQ」 | 产品要的是出图吞吐 | 与 LoRA 训练目标分叉 |

### 明确低 ROI / 不建议

- 再堆 Python 级 fusion 或默认 FWHT  
- 默认 `_weight_int8pack_mm`（本机已证慢）  
- 指望 Comfy INT8 节点 magically 支持训练  
- 无 profile 就写完整 Triton 框架  
- 把 4090 W4A4 推理加速写进训练预期  

---

## 5. 「怎样才可能快过 bf16」（诚实门槛）

在 **8–12GB 消费卡、frozen base + LoRA 训练** 上要 **step time < bf16**，需同时接近：

1. 激活侧 online 成本接近 0，或被 **真融合** 进 GEMM（DiT AdaLN 使 offline 吸收很难）  
2. 真正的 int8/int4 Tensor Core 路径，且 **M/K/N 大到核饱和**  
3. **不再** dequant 回 bf16 再 `F.linear`（否则是 bf16 GEMM + 额外税）  
4. 硬件最好是 4090 级及以上或数据中心 GPU；**3080 不是理想战场**

---

## 6. 建议执行顺序（下一迭代）

```text
1. [DONE] P0-A  step profile → experimental §G；Triton 门槛未达
2. [DONE] P0-A2 修 W8A16：dequant/F.linear 用 bf16/fp16 → ~1.23× bf16
3. [DONE] P0-C   prequant_checkpoint 原生 v1 加载/导出
4. [DONE] P0-D   regular Hadamard + multi-seed（默认仍 sylvester；opt-in regular@64）
5. [DONE] P1.*   dtype 交通税 / kn 布局 / compile / scope profiles / hadamard 产品化
6. [DONE] P1.11  否决 W8A8 half/TF32 默认加速；W8A8 scale fp32
7. [DONE] Phase1 平台期文档 §G.16 — 下一刀仅当：换卡 / 真要 step≤bf16 / 或训整模 QAT
8. [DONE] 替代 KPI：同显存更大 rank（§G.17）+ 同显存更大 batch（§G.19；**all@b2**）
9. [DONE] §G.20：分辨率轴阻断；compile autotune 不关合税 → **Phase 1 micro-opt 冻结**
10. 仅当未来 profile 显示 RHT+quant 内存链 >50% 时，再开 P2-K Triton spike
```

依赖关系：

- **A/A2/C/D/P1 已完成** → 否决「立刻 Triton」；prequant 不承诺 step 加速  
- **regular 默认切换** 可选（WebUI 已有）；sylvester@256 仍兼容默认  
- **P1-J 搁置**（冻结 base 路径无权重 STE 可修）  
- **K/L 强依赖「convrot tax ≥50%」**，当前 ~8% 不满足  
- **Phase 1 冻结**：rank/batch 替代 KPI 已验证；分辨率本机无轴；残余 ~4% step 需换硬件或 P2；不再为抠 1% 开默认 half/TF32/scratch

---

## 7. 与现有代码的接口约束（实现时勿破坏）

- 只替换 LoRA `org_forward`；不 swap 子 `nn.Linear`  
- apply 在 `apply_to` / load / grad-ckpt / fp32 residual 之后，`compile_blocks` 之前  
- 默认 `base_compute=bf16`；与 `block_swap_transfer_dtype=int8` 互斥  
- free base 默认开启；merge 拒绝 ConvRot base  
- 新逻辑继续放在 `library/runtime/convrot/`，bootstrap 只薄调用  
- 环境变量已有：
  - `ANIMA_CONVROT_INT8_GEMM=auto|int_mm|float`
  - `ANIMA_CONVROT_FUSED=1|0`
  - `ANIMA_CONVROT_RHT=dense|fwht`
  - `ANIMA_CONVROT_W8A16_KERNEL=dequant|int8pack`
  - `ANIMA_CONVROT_HADAMARD=sylvester|regular`（也可 `--convrot_hadamard`）
  - `ANIMA_CONVROT_STE_TF32=0|1`（W8A8 STE TF32，默认关）
  - `ANIMA_CONVROT_DEQUANT_SCRATCH=0|1`（默认关）

---

## 8. 主要来源

- [ConvRot arXiv:2512.03673](https://arxiv.org/abs/2512.03673)  
- [QuaRot arXiv:2404.00456](https://arxiv.org/abs/2404.00456) · [spcl/QuaRot](https://github.com/spcl/QuaRot)  
- [QuEST arXiv:2502.05003](https://arxiv.org/abs/2502.05003) · [IST-DASLab/QuEST](https://github.com/IST-DASLab/QuEST)  
- [obsxrver INT8 ConvRot weights](https://huggingface.co/obsxrver/ComfyUI-Native-INT8_ConvRot)  
- [convert_to_quant](https://github.com/silveroxides/convert_to_quant)  
- [ComfyUI INT8 PR #14636](https://github.com/Comfy-Org/ComfyUI/pull/14636)  
- [ComfyUI-INT8-Fast Metrics](https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/Metrics.md)  
- 本仓实测：`output/tests/convrot_mem_speed*.json`、`docs/experimental/convrot_int8_training.md`

---

## 9. 修订记录

| 日期 | 变更 |
| --- | --- |
| 2026-07-24 | 初版：基于本机 3080 实测 + deep-research 交叉文献，冻结 P0/P1/P2 ROI 路线图与成功标准 |
| 2026-07-24 | **P0-A 完成**：`convrot_step_profile_probe.py` + JSON；convrot tax~14% 否决 Triton；新增 P0-A2（W8A16 保持 bf16 计算） |
| 2026-07-24 | **P0-A2 完成**：W8A16 dequant linear 用 bf16 TC；mem_speed 3.06→1.99 s/step（peak 4.34GB） |
| 2026-07-25 | **P0-C 完成**：`prequant.py` 原生 v1 + apply 接线 + export 脚本 + unit tests；CLI help 更新 |
| 2026-07-25 | P0-C **mem/speed 热测**：online vs prequant step 几乎相同（W8A16 2.03 vs 2.05 s）；apply 因读盘更慢；peak 同 free-base；见 `convrot_mem_speed_prequant.json` |
| 2026-07-25 | **P0-D 完成**：regular Hadamard（Kronecker \(4^k\)）+ `ANIMA_CONVROT_HADAMARD`；W8A16 multi-seed：regular@64 seed0 过 5% gate，grad_max 最低；默认仍 sylvester@256 |
| 2026-07-25 | **P0-A3 完成**：fusion microbench 否决 P2 K/epilogue；**P1-F/G/H 实现 + 热测** `convrot_mem_speed_p1.json`：full W8A16 仍最省（4.34GB/1.21×）；largest 最快（1.13×）但 peak 回吐 |
| 2026-07-25 | **P1.5 dtype 交通税**：dequant 目标 dtype、RHT/GEMM 去强制 fp32、hadamard bf16 buffer、scale `mul_`；热测 W8A16 **1.05× / 4.14GB**（`convrot_mem_speed_p15.json`） |
| 2026-07-25 | **P1.6/1.7**：W8A8 kn 布局、bwd 免完整 dequant、共享 Hadamard；热测相对 P1.5 **中性**（`p16`/`p17.json`） |
| 2026-07-25 | **P1.8 + compile 热测**：dequant scratch 默认关；`--torch-compile` 下 W8A16 **1.036× / 4.11GB**（`convrot_mem_speed_compile_p18.json`）；step profile tax~8% |
| 2026-07-25 | **质量复测 + scope=all**：ckpt W8A16 仍 2/3、W8A8 3/3；scope=all free W8A16 **3.44GB / 1.08×**（compile 3.43GB / 1.08×） |
| 2026-07-25 | **regular@64 3/3 gate** + short-train 20-step last_rel **0.07%**；质量 opt-in 建议写入 experimental §G.11 |
| 2026-07-25 | **P1.9 W8A8**：STE 不存 x_rot；`int8_mm` `out_dtype=bf16`；compile 下 1.657→**1.582 s** / peak 4.14GB |
| 2026-07-25 | **P1.10**：W8A16 CUDA scale 存 bf16；`--convrot_hadamard` CLI/bootstrap/metadata/WebUI；regular@64 可配置质量档 |
| 2026-07-25 | **P1.11 否决默认 W8A8 加速**：half quant/bf16 STE 破 gate；TF32 STE opt-in；**W8A8 scale 改回 fp32**（p111e 3/3） |
| 2026-07-25 | **Phase 1 平台期**：§G.16 产品默认；P1-J 搁置（冻结 base 无权重 STE）；残余 ~5% 需 P2/换卡 |
| 2026-07-25 | **同显存更大 rank**：W8A16@r32 peak **4.20GB** 仍 < bf16@r4 **4.95GB**（§G.17）；probe 支持 `--lora-rank` |
| 2026-07-26 | **scope=all@r32** peak **3.63GB**；P1.12 frozen `w_q/scale` 不进 `save_for_backward`（速度中性清洁） |
| 2026-07-26 | **同显存更大 batch**：mlp@b2 越峰；**all@b2 4.42GB**、**all@r32@b2 4.62GB** 仍 < bf16@b1 4.95（§G.19）；probe `--batch-size`；bwd cast scratch 否决 |
| 2026-07-26 | **分辨率 KPI 阻断**（全 cache 同 16128 token）；`max-autotune-no-cudagraphs` 两端各 ~2.7% 但 **×bf16 仍 1.04**（§G.20）；probe 自动选 CC≥7.5 GPU + `--compile-mode` |
| 2026-07-26 | **产品变体** `configs/gui-methods/lora-convrot-vram.toml`（scope=all / dim=32 / W8A16；§G.21） |
| 2026-07-26 | **否决** ConvRot∥block_swap 作为下一刀：apply 顺序 free-base 在 swap 后；meta weight 与 offloader 未审计 |
| 2026-07-26 | **热测 §G.22**：`all+GC` 可叠（3.43GB / 1.75s eager / math ok）；`all±GC+swap20` free-base 后首次 prepare **硬挂** meta master（`convrot_stack_*.json`）；probe 增 `--blocks-to-swap` / RSS |
| 2026-07-26 | **方案 A**：`is_weight_swap_excluded` 跳过 free-base/meta；`all+GC+swap20` **3.17GB / 2.04s / RSS 2.33GB**（vs all+GC 3.43/1.76/1.57）；无 GC+swap **OOM**；§G.22.2 |
| 2026-08-04 | **block-owned residency 完成**：ConvRot weight/scale carrier 归属实际 DiT block 并随 swap 迁移；0/12/26 的 payload CPU/GPU 计数严格匹配 block 布局；8-step 消融见实验文档 §G.23 |
| 2026-08-05 | **SM86 `_int_mm` 4200-token 修复**：`int8_mm_scaled` 对 M%32≠0 零填充到 32 倍数再裁回，compiled 4200 家族不再 `CUBLAS_STATUS_NOT_SUPPORTED`；实验文档 §G.24 |
| 2026-08-06 | **ConvRot × distributed 热测**：新探针 `convrot_distributed_probe.py`；2-GPU mean-reduce 数学等价（cross_rank_diff=0）、payload 落各自 rank、W8A8 `_int_mm` 在 cuda:1 正确；实验文档 §G.25 |
| 2026-08-06 | **incremental free-base**：apply 循环内装好 payload 即 free bf16 base（原批量 free 留作幂等兜底），apply 瞬时峰 5.86→5.24GB（−0.62GB）、apply 后驻留 4.03→2.41GB，稳态/数学不变；实验文档 §G.26 |
