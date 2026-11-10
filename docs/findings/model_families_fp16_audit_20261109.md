# 全模型族 FP16 / 混合精度支持现状与分级诊断

状态：审计（只读静态普查 + 一次 sm70 GPU 冒烟探针；不含训练、不含源码改动）
适用版本：分支 `codex/model-fp16-paths`，静态扫描点 HEAD `9d530588`；探针环境 torch 2.10.0+cu129 /
Tesla V100-SXM2-16GB（sm70）/ bitsandbytes 0.49.2
相关代码：`library/models/family_registry.py`、`library/training/compat_matrix.py`、
`library/training/precision_policy.py`、`library/training/model_loading.py`、
`library/training/adaptive_runtime/{precision,native_precision,native_family}.py`、
`library/training/{losses,loop,bootstrap}.py`、`library/training/moss_compat.py`、
`library/quantization/{nf4,mixed_nf4,format,policy}.py`、`library/models/krea2_raw/attention_backend.py`、
`networks/attention_dispatch.py`
关联文档：[未验证类兼容拦截逐条审计](compat_unverified_audit_20261109.md)、
[未验证类兼容拦截的降级策略](compat_unverified_policy_20261109.md)、
[Krea-2 NF4 fp16 梯度偏差](krea2_nf4_fp16_grad_underflow_20261108.md)、
[Krea-2 NF4 fp16 内层路径](krea2_nf4_fp16_inner_path_20261109.md)、
[多模型支持](../multi_model_support.md)、[V100 FlashAttention 支持](v100_flash_attention_support.md)

## 证据等级说明

本文所有数字都从既有记录或本次探针抄来，没有新增测量。三档标注：

- **L0 静态**：只读源码 + 已测事实的演绎，未运行。t1/t2a/t2b/t2c 的主体属于这一档。
- **L1 既有真机记录**：引用 `docs/findings/` 里已落盘的真机训练/探针结果。
- **L2 GPU 冒烟**：本次 sm70 探针（合成随机权重、单/双层、非训练步），结论对「后端与 dtype 判定」可信，
  对性能与数值只作定性。

## 0. 结论摘要

1. **权威族清单是 5 族，不是 4 族**：`MODEL_FAMILY_REGISTRY` 含 `moss_image`，而 `AGENTS.md` 只写 4 族；
   MOSS 的精度契约也不在 `compat_matrix.py`，在 `library/training/moss_compat.py`（L0）。
2. **没有任何一族的「支持 fp16」依据是对 fp32 裁判的梯度偏差达标**。有实测的三组全部不达
   `Tolerances`（output rel-L2 0.01 / grad rel-L2 0.02 / cos 0.999）：anima 严格 12 格 **0/12**、
   anima islands 三档 **3/3 拒**、krea2 NF4-fp16（k=14）**4.58%**（L1）。唯一通过的是全 FP32 控制组。
3. **Anima 是「假收敛」的教科书样本**：200 步配对 loss 与验证 MSE 与 BF16 打平（FP16 甚至低 0.0253%），
   同一模型的严格梯度矩阵 0/12（input grad 2.0898）——「看 loss 正常所以 fp16 没问题」在本仓已被直接反驳（L1）。
4. **sm70 上 SDPA 后端由 autocast dtype 决定，不是输入 dtype**（L2，本次探针头条）：
   fp16 → EFFICIENT（16.0 MiB/层）、`autocast(bf16)` → MATH（2408~2449 MiB/层）、
   `autocast(bf16)` 内套 `autocast(enabled=False)` → 回到 EFFICIENT。
5. **因此 Krea-2 `_fp16_attention` 不是「提速武器」，是 bf16 回退守卫**；在 V100 fp16 生产路径上
   平移给 Anima 的步时收益 ≈ 0（L2）。
6. **dir1 对 Anima 是「数值上 100% 适用、运行时 0% 可达」**：算子零 Krea-2 假设，但
   `{anima, mixed-NF4, dir1, sm<80}` 被三道约束夹死（L0 + L2）。
7. **系统级 P0-1**：`check_training_compat` 跑在 `resolve_mixed_precision` **之前**，sm<8 的 bf16→fp16
   就地改写绕过全部 fp16 fail-closed 规则，`z_image_bf16_only` 与 `qwen_image_2_1_mixed_precision_flash`
   这两个硬 code 从未生效（L0）。
8. **系统级 P1**：`precision_policy` 的改写与所有预量化 mixed-NF4 族在 sm<80 上互斥，唯一豁免只接
   Krea-2 loader；preflight 在改写前判定故放行，失败只在 model load 炸（L0）。

## 1. 一图流：全族 FP16 / 混合精度支持现状与潜力矩阵

| 模型族 | 量化格式 | 当前允许精度 | sm70 (V100) 现状 | sm80+ (3080) 现状 | 核心物理瓶颈 | 可复用的 Krea-2 资产 | 推荐优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `anima` | bf16 基座；可选 mixed-NF4（Anima 私有 manifest `mlp-nf4-except-layer1-blocks14-20`，49/56 MLP 模块；算子是共享 `BF16AutocastLinear4bit` 的纯别名，`library/anima/mixed_nf4/linear.py:4`） | fp16（legacy AMP / `native_fp16` / islands）、bf16、fp32；`compat_matrix` 无 mixed_precision 拦截（L0） | **生产即 fp16**（`precision_policy.py:57-58` 在 sm<8 把 bf16 就地改写）；attention 实测 EFFICIENT 16.0 MiB/层（L2）；1MP **1.961 s/步**（L1）；但梯度保真 **0/12**（L1） | 未测（无 3080 直接证据；sm≥8 时 `preferred_candidate` 选 bf16） | 反向 dY 尾数截断/subnormal：fp32 残差守卫（`models.py:2665`）只防 inf，不保精度 | 无（对照族；其 `native_fp16` + `enable_fp32_residual` 是 Krea-2 native 守卫的参照实现） | P1：按「可运行、未达梯度保真门槛」管理，不新增能力 |
| `krea2_raw` | bf16 基座；NF4（在线/预量化/自包含，`compute_dtype` 恒 bf16，5 处 fail-closed）；两个内层 fp16 开关 `nf4_dir1` / `krea2_attn_fp16` | bf16（默认）；fp16 无 compat 拦截（legacy AMP 会把 13B 基座整份载成 fp16，`model_loading.py:366`）；`native_fp16` 显式且装 fp32 残差守卫；NF4 只允许 bf16 compute | dir1 + attn_fp16 **双开 44.8 → 11.06 s/step（4.05×）**（L1）；裸 fp16 LoRA 梯度 114.741%、dir1 40.974%、bf16 40.614%（L1）；1024² **loss-scale 窗口关闭**（k 上限 15、自收敛偏差 5.900%）；基座 fp16 梯度偏差**未测** | dir1 在 sm≥80 直接 `ValueError`（`nf4.py:41-46`）；3080 的 FP16 路线已被否决（L1）；native fp16 仅 90HX 3 步 smoke | 反向下溢：中深层 dY rms 6.5e-9~1.7e-7（264 个 Linear 的 grad_output 除 2 个外全部低于 fp16 正规下限 6.1e-5）+ 非 native 路径**无 fp32 残差守卫**（`bootstrap.py:296-302` 只覆盖 Anima/native） | 资产源头（dir1 / attn_fp16 / native guards） | P0：补非 native fp16 守卫或显式拒绝（P0-2） |
| `z_image` | bf16 基座；mixed-NF4 需 bf16（`weights.py:243`）；compile 三档恒 error | 仅 bf16，或 adaptive islands + fp16（且 attn∈{torch,sdpa,native}），或纯 fp32（`compat_matrix.py:1349-1358`） | 未测（sm70 无 z_image fp16 记录）；越界实测在 90HX：islands 全模型首步 loss 非有限（L1） | 未测（无 3080 fp16 记录；bf16 是既有生产档） | 前向物理越界且**层局部**（`layers.28.attention.to_q` σ=0.8、`noise_refiner.0.feed_forward.w2` 三 σ 非有限） | 无：dir1 前向不缩放（`nf4.py:98`）且开关是全局 `_DIR1_ENABLED`（无 per-layer 粒度）→ 越界层救不了 | 不建议（纯 fp16）；islands 已是正解 |
| `qwen_image_2_1` | bf16 基座；mixed-NF4 为 all-linear（232 模块）；fused projections | fp16 = C1（默认 warn 降为 warning）；`flash`+fp16 与 mixed-NF4+fp16 恒 error（`compat_matrix.py:886-900`） | 未测（**零 fp16 数值证据**，`grep fp16 docs/findings/qwen*.md` 为空） | 未测（无 fp16 记录；bf16 为既有生产档） | 契约性而非物理：唯一「允许 fp16 且完全没有 fp32 岛」的族；fused projections 在 fp16 下未测 | dir1 宿主最完整（all-linear 232 模块），但 dir1 只在 sm<80 生效 → 与 3080 无缘 | P1：放开前须补 islands + 数值证据 |
| `moss_image` | **无 DiT NF4**（仅 TE NF4，且 TE 被 `integration.py:10` 硬钉 bf16） | bf16 / fp16 / no 全部放行（`moss_compat.py:45`），**零精度拦截** | 未测于 sm70；sm75 512² 16–32 步 loss 全 finite（L1，原文声明不构成稳定结论）；1024² 未测；梯度偏差**未测** | 未测 | 前向越界风险：残差无 fp32 累加（`math.py:107-112`）、调制门**无 tanh 上界**；同架构 B 侧 13/20 步 NaN（L1） | 无（无 `Linear4bit` 宿主，dir1 不可嫁接） | P0：先补守卫（残差 fp32 + 门控上界），再谈 fp16 |

矩阵读法：`当前允许精度` 是**契约面**（`compat_matrix` / `moss_compat` 是否放行），
`核心物理瓶颈` 是**物理面**（是否会溢出/下溢）。两者不一致正是本报告要区分的东西——
象限三的「死因3」就是「契约拒绝但物理无证据」。

## 2. 象限一（已支持 FP16 的族）：健康度审计与优化空间

### 2.1 谁允许 fp16，以及实际走哪条路

| 族 | 允许路径 | 代码依据 |
| --- | --- | --- |
| `anima` | ① legacy AMP（`mixed_precision=fp16`）② `native_fp16` ③ islands | `compat_matrix` 无拦截；`adaptive_runtime/native_family.py:22-23` `enable_fp32_residual()` |
| `krea2_raw` | ① legacy AMP（无拦截）② `native_fp16`（显式 + 全 fp16 基座） | `bootstrap.py:277-303` 只给 Anima 与 native Krea-2 装守卫 |
| `z_image` | 仅 islands + fp16；裸 fp16 硬拒 `z_image_bf16_only` | `compat_matrix.py:1344-1362` |
| `qwen_image_2_1` | fp16 = C1（warn 档降为 warning）；flash / mixed-NF4 + fp16 硬拒 | `compat_matrix.py:886-903` |
| `moss_image` | 显式接受 `{bf16, fp16, no}` | `moss_compat.py:45` |

**关键放大器**：`library/training/precision_policy.py:57-58` 的 `resolve_mixed_precision` 在 `major<8` 时把
`bf16` **就地改写成 `fp16`**，全族生效、无族判断，调用点在 `train_session.py:217`（在
`prepare_dtype()` 与 Accelerator 构造之前、`load_unet_lazily()` 之前）。所以 **Krea-2 / MOSS 的无守卫
fp16 路径不需要用户显式选 fp16**，V100 / T4 上跑默认 bf16 配置就会踩到（L0）。

### 2.2 逐族健康度

**`anima` — 有隐患（全族唯一同时有 loss 与严格梯度矩阵的族，两者结论相反）**

- 200 步配对（L1）：200/200 有效更新、0 跳步、scale 全程 1024；共同 BF16 evaluator 上 FP16 MSE
  0.204031005 vs BF16 0.204082673（FP16 低 0.0253%）；配对 loss 最大相对差 0.4745%。
- 严格 12 格（L1）：prediction rel-L2 0.001324（门槛 0.01，12/12 过）；**input grad 2.0898 / adapter
  grad 0.6535（门槛 0.02，12/12 失败）**；最小 cos −0.0404 / 0.8143（门槛 0.999）；`strict_passes=0`。
  测量就在生产 4032 / 4200 token 包上。
- islands 三档（L1）：FP16 seed 90.15%/31.65%、conditioning 120.56%/32.02%、
  conditioning+attention-fp32 41.77%/14.14%，**全部拒绝**；只有全 FP32 控制组通过（0.000357%/0.000178%）。
- 守卫的真实保护面：`models.py:2665` `use_fp32 = hidden.dtype == torch.float16` +
  `_gated_residual_add` 把残差流提升 fp32 → **只防溢出（inf）**，不保护子层 GEMM 的 dY 尾数截断/subnormal，
  而后者正是 209% / 65% 的来源。
- 判定：**loss 健康、梯度保真已实测失败；守卫是「防 NaN」不是「保精度」。**

**`krea2_raw` — 有隐患（机制已实测，路径未测）**

- 非 native fp16：`dit.py:370-376` `x = x + pregate*attn_output` 全在 compute dtype（fp16）里累加；
  `compute_dtype = x.dtype`，而 noisy 输入由 `library/runtime/noise.py:189` cast 成 `weight_dtype=fp16`
  → **残差流就是 fp16**。`bootstrap.py:296-302` 的 `if not isinstance(unet, anima_cls): return False`
  让 Krea-2 完全跳过守卫；只有 `fp32_residual=True`（native）才走 `native_precision.py:66-74` 的 fp32 累加。
- 数值证据（L1，fp32 为唯一裁判）：256² 单步 LoRA 梯度 rel-L2 —— bf16 **40.614%**（cos 0.92764）/
  裸 fp16 **114.741%**（cos 0.36736）/ dir1 **40.974%**（cos 0.92553）；加 loss scaling k=14 后
  fp16 降到 **4.58%**（cos 0.99897），**仍超 0.02 门槛**。512² latent grad 100% < 6.1e-5、69.3% 精确归零；
  中深层 dY rms 6.5e-9~1.7e-7。
- **窗口随分辨率关闭**（L1）：k 自收敛偏差 256² 0.123% → 512² 0.994% → 768² 4.090% → 1024² 5.900%，
  单调上升；k 硬上限 15（`loss×2^16 = 65536 > 65504`），768²/1024² 在 k=15 仍未饱和、需 k≥16~18
  超出 fp16 标量表达范围 → **生产 1024² 的 loss-scaling 窗口关闭**。
- 口径边界：上述是 **NF4 fp16 内层 GEMM**，不是 `mixed_precision=fp16` 基座。基座 fp16 的梯度偏差
  **未测**；`native_fp16` 只有 90HX 3 步 smoke（3/3 更新、0 跳步、scaler 1024 不变，原文明确「没有质量
  结论、没有长期数值稳定性结论」）。但 dY 量级由残差链 + AdaLN 门控决定，与是否 NF4 无关，
  机制可外推、数字不可外推。

**`moss_image` — 有隐患（结构最脆弱的一族）**

- `math.py:111/114` `x = x + proj_o(...)*gate_a`、`return x + proj_ffn_down(...)*gate_f`，无任何 fp32 提升；
  `dit.py:94` 把残差流定成 `weight_dtype`（fp16）。`_RMSNorm` 内部 fp32 但输出 `.to(x.dtype)` 回 fp16。
  `_situ` 本身有界（≤4 / ≤25），但**调制门 gate_a / gate_f 无 tanh 上界**（对比 qwen 的 `img_gate.tanh()`）。
- 唯一实测（L1）：sm75、512²、bf16 被 precision_policy 改成 fp16，swap 5 档各 16–32 步 **loss 全 finite、
  0 Xid**；但原文写明「不构成 A 在 sm75 上 fp16 稳定的结论」。同文档对照项（显式 fp16 + `fp32_residual`）
  同窗口 13/20 步非有限、loss_scale 4096→1，说明「加 fp32 残差」也不是充分条件。梯度偏差**未测**。

**`qwen_image_2_1` — 证据不足（结构上最保守的 fp16 候选）**

- 残差：`transformer_qwenimage21.py:641/644` `hidden_states = hidden_states + img_gate.tanh()*attn_output`，
  且 `family.py:161-165` 的 noisy 张量**不 cast** 到 weight_dtype（latents 恒 fp32）→ 残差流自然留在 fp32，
  门有 tanh 上界。
- 兜底：`segmented_training.py:89` `clip(-65504, 65504)` 仅 segmented 路径且仅在已 fp16 时触发，
  **有损**：饱和区梯度被改，救不了已发生的 NaN。
- fp16 属 C1（`compat_matrix.py:900-903` 明说「实现会生效、缺真机验收」）。**零 fp16 数值证据。**

**`z_image` — 证据不足（唯一族级 fail-closed，但可被绕过）**

- 只允许 islands + fp16（`compat_matrix.py:1344-1362`）；`islands.py:19-33` 的 `_island_forward` 在
  `autocast(enabled=False)` 内跑 fp16 单元、**结果回写 fp32**，结构上是最难出事的一条路。
- 但 Z-Image 自身零 fp16 数值证据；且该硬拒可被 2.1 的 sm<8 改写绕过（见 §5 P0-1）。
- `z_image/attention_backend.py:58-60`：flash 仅 bf16，fp16 抛错——**运行期 fail-closed，不是静默降级**。

### 2.3 假收敛 / 隐性下溢（跨族判据）

1. **没有任何族「支持 fp16」的依据是梯度对 fp32 裁判的偏差达标**（见 §0-2）。
2. **anima 是假收敛的教科书样本**：200 步零跳步、loss 与验证 MSE 与 BF16 打平，同模型严格梯度 0/12。
3. **1024² 深层 subnormal**：Krea-2 已实测并给出 k 窗口在 512² 关闭的结论；anima 的 209%/65% 是在
   生产 4032/4200 token 包（≈1024² 级）上测的，同样落在下溢区。z_image / qwen / moss **未做逐层 dY 测量**，
   机制可类推，**数字不可**冒充。
4. **守卫与保真是两件事**：anima 的 fp32 残差、qwen 的 tanh 门 + clip、krea2 的 `fp32_residual`
   都属于「防 inf」，都无法恢复 fp16 操作数已经丢掉的有效位。写文档必须分开表述。

### 2.4 Loss 与 GradScaler

**loss 全族 fp32（已核实）**：`losses.py:310-330` 的 `_flow_match_loss` 对 `model_pred` / `target`
双 `.float()` 后进 conditional_loss；VR、velocity-direction、prior、fera 同样先 `.float()`。各族只产出
`(model_pred, target)`，都经此入口。

**但 fp32 loss ≠ fp32 梯度**：`.float()` 是带 backward 的 cast，梯度要舍回 `model_pred` 的 fp16；
Krea-2 文档 §5.1 实测该缺口（`2*(v-t)/N ≈ 2e-5`，在 fp16 里只剩 0~2 位，靠乘 scale 抢救）。
**「loss 用 fp32 算」不能作为精度论证。**

**GradScaler 面（`mixed_precision=="fp16"` 且有 CUDA 设备才由 accelerate 创建；bf16 天然无 scaler）**：
init_scale legacy = torch 默认 65536；native / islands 走 `GradScalerKwargs(init_scale=adaptive_loss_scale`，
默认 1024)，且 `accelerator.py:208-211` 对 native/islands **硬性要求 scaler 非 None**。无 scaler 时
`accelerator.py:115-141` 只 warning（用户裁定），由 `scaler_ownership.py` + `loop.py:1311`
接管「非有限梯度丢弃本次 update」。

**unscale_ 时机正确**：`accelerator.backward`(:1248) → `_guard_nonfinite_grads`(:1311，**故意在 unscale
前读 scaled 梯度**做所有权判定) → `accelerator.clip_grad_norm_`(:1361，内部先 `unscale_gradients()`) →
`state.optimizer.step()`(:1420，`AcceleratedOptimizer.step` → `scaler.step()` + `scaler.update()`)。
unscale 在 clip 之前、step 之前，且不会二次 unscale；`max_grad_norm==0` 时顺序仍正确。

### 2.5 GradScaler 新发现（P0-3，静态阅读未复现）

**drop-after-unscale 会把 scaler 的 per-optimizer stage 卡在 UNSCALED。** `loop.py:1326` 的 clip 门是
`sync_gradients and not skip_update and not force_skip and loss_is_finite`（**rank 本地**判定），而
`multigpu.agree_step` 在 `:1376-1386`，**在 clip 之后**。多进程下若 rank B 本地 loss 有限、rank A 非有限：
B 已 clip（= 已 unscale）→ 协议一致决定 drop → B 走 `zero_grad` 分支、**不调用 `scaler.step/update`** →
`_per_optimizer_states[id(opt)]["stage"]` 停在 UNSCALED → **下一步 B 的 clip → `unscale_` 抛
`RuntimeError: unscale_() has already been called`**。

现有防护抓不到：`assert_scaler_symmetry` 只比 scale + growing tracker，不含 per-optimizer stage；
`tests/test_window_rank_consistency.py` 的 drop 路径未挂真 scaler。修法（任一）：drop 分支补
`accelerator.scaler.update()`；把 clip 移到 `agree_step` 之后；或在 clip 前先 agree loss/grad 有限性。
影响面：所有用 scaler 的 fp16 运行（legacy / native / islands），单进程不受影响。

### 2.6 autocast 截胡：先更正任务前提

**`_fp16_attention` 与 `_Dir1MatMul4Bit.forward` 不是「无保护的手写 cast」，而是已修好的参照实现**：
`attention_backend.py:283` `with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel([...])`、
`nf4.py:104` 同款。它们由 [Krea-2 NF4 fp16 内层路径](krea2_nf4_fp16_inner_path_20261109.md) 实测确认是
**修好之后**的状态（双开生效 11.06 s/step vs 未修 44.8 s/step，4.05×）；失效方式是「被 autocast 抢占 →
静默回退慢内核」——**性能事故，不是精度事故**。

逐族扫查结果（L0）：

| 族 | 结论 |
| --- | --- |
| `anima` | 无未保护截胡点。`models.py:748-776` 是 QKV dtype 对齐（fp16 autocast 下本就不触发）；`_fp32_project`、RoPE 均已 `enabled=False` |
| `krea2_raw` | 无。`native_precision.py:11/85`、`attention_backend.py:270/283` 均有 `enabled=False` |
| `z_image` | 无。diffusers rotary 内部 `autocast("cuda", enabled=False)`；其余 `.to(dtype)` 的 dtype 就是 autocast dtype |
| `qwen_image_2_1` | 无。`native_layer_norm.py:74-79` 显式 `.float()` 属 LayerNorm 的 autocast fp32 策略 |
| `moss_image` | 两处**同模式但当前无实害**：`math.py:106` `y.to(dtype)`、`math.py:147` `q.to(v.dtype)` 后接白名单 SDPA。因为 `moss_compat` 要求 `adaptive_precision=off`，权重 dtype 恒等于 autocast dtype。**风险是脆性**：一旦给 MOSS 加 islands/native 会静默失效或降精度 |

残留小缺陷：`_Dir1MatMul4Bit.backward`（`nf4.py:124` 附近）的 `torch.matmul(g16, w.t())` **没有**
`enabled=False`。当前 backward 在 autocast 区外执行，无实害；若将来把 backward 放进 autocast，fp16 操作数
会被抬回 bf16 → dir1 白做且精度反降。**建议补保护或注释。**

`networks/attention_dispatch.py:166-197` `_cast_qkv_for_flash` 不属于截胡，但**在 autocast 关闭
（`mixed_precision=no`）时把 fp32 QKV 静默降成 bf16**（`_cast_qkv_for_flash` 的 fallback 分支返回 bf16）
→ 「fp32 训练」实际以 bf16 attention 跑。当前 compat 未覆盖，建议显式报错或 warning。

### 2.7 象限一优化点清单

| 编号 | 内容 | 依据 |
| --- | --- | --- |
| P0-1 | 打通 sm<8 改写的 fail-closed：`resolve_mixed_precision` 之后复跑一次 `check_training_compat` | §5 |
| P0-2 | Krea-2 非 native fp16 要么装守卫要么拒绝：`dit.py:370` 已有 `fp32_residual` 分支，把 `bootstrap.py:296-302` 的 `isinstance(unet, anima_cls)` 扩到 `SingleStreamDiT`；不装就加硬 code `krea2_legacy_fp16_no_fp32_residual`（代价：每 block 一次 fp32 转换/回写，必须实测步时与显存） | 2.2 |
| P0-3 | 修 drop-after-unscale：drop 分支补 `scaler.update()` | 2.5 |
| P1-1 | MOSS 残差流强制 fp32 + 评估给门加上界；把「moss fp16 = 无守卫」写进 [多模型支持](../multi_model_support.md) | 2.2 |
| P1-2 | `_Dir1MatMul4Bit.backward` 补 `autocast(enabled=False)` | 2.6 |
| P1-3 | `mixed_precision=no` + `attn_mode=flash` 显式报错 | 2.6 |
| P2-1 | 文档口径：把所有「支持 fp16」改成「可运行、未达梯度保真门槛」，以 §2.2 三组实测数字为唯一证据面 | 2.3 |
| P2-2 | 补齐证据缺口：z_image / qwen / moss / krea2-基座 各做一次 `bench/krea2_nf4_fp16_grad` 口径的逐层 dY 探针（只读、离线） | §6 |

## 3. 象限二（可拓展族）：Krea-2 资产（dir1 + attn_fp16）平移可行性与预期 ROI

本章严格分栏：**已实测**（有落盘数字）与**估算**（公式 + 假设明示）不混写。

### 3.1 dir1 → Anima：算子零成本，运行时不可达

**可复用性成立（L0，逐条核对）**

- Anima 的 NF4 线性就是共享算子：`library/anima/mixed_nf4/linear.py:2/4`
  `AnimaMixedNF4Linear = BF16AutocastLinear4bit`（**纯别名**），`format.py:10` 的 quant_state 校验也委托回
  共享实现 → `BF16AutocastLinear4bit.forward` 里的 `if _DIR1_ENABLED: return self._forward_dir1(x)`
  天然对 Anima 的 49 个 NF4 MLP 生效。
- `--nf4_dir1`（`cli_args.py:1146-1162`）→ `bootstrap.py:298-319 maybe_enable_nf4_dir1`
  **无任何 family gate**，调用点在 family 无关的 `create_and_apply_network`（`bootstrap.py:607/754`）；
  对照紧邻的 `maybe_enable_krea2_attn_fp16` 才是 Krea-2 专属。
- `_Dir1MatMul4Bit`（`nf4.py:72-129`）**零 Krea-2 专属假设**：不认 head 数 / GQA / 层类型；转置方向靠
  bnb `dequantize_4bit` 的 `A.shape[0] == 1` 判据，`_forward_dir1` 恒传 `weight.t()`；唯一隐含假设是
  bias 分支的 `grad_output.sum(0)` 需 2D，而两族 MLP 都是 `bias=False`（`anima/models.py:590-591`、
  krea2 `SingleMMDiTConfig.bias=False`）。

**但「零成本」只到算子层——三约束把可达集合压缩到只有 Krea-2（L0）**

1. `set_dir1_enabled(True)` 在 `major>=8` 直接 `ValueError`（`nf4.py:41-46`）→ dir1 只能 sm70/sm75。
2. Anima / 共享 mixed-NF4 都要求 bf16 compute（`anima/weights.py:175-176`、
   `anima/mixed_nf4/runtime.py:22-23`、`quantization/mixed_nf4.py:56-57`）。
3. `precision_policy.py:57-58` 在 sm<8 把 bf16 强制改写成 fp16，唯一豁免 `base_compute == "nf4"`
   （`:40-46`）只接 Krea-2 loader（`model_loading.py:373-409`）；Anima 侧**没有任何 `base_compute` 消费者**，
   `preflight_nf4.py:11` 又因 `policy.FAMILIES` 不含 anima 而整段早退 → Anima NF4 在 preflight 阶段
   完全没有检查。

**⇒ `{anima, mixed-NF4, --nf4_dir1, sm<80}` 必然在 model load 处失败（"Anima mixed NF4 requires BF16
compute"），而 sm≥80 上 dir1 又拒绝启用。dir1 对 Anima 是「物理上够得着、逻辑上不可达」。**

**数值适用性（L2 探针，合成随机权重、非真实权重）**

- 前向 `.half()` 安全：进 GEMM 的 fp16 输入 amax = **5.28**（MLP layer1 / qkv_proj，单位 RMS 输入）、
  **5.56**（MLP layer2，真 GELU 输入，rms 0.64），相对 65504 余量 **11776~12403×**；需放大 ~2000×
  才碰上限。RMSNorm 结构上把块输入钉在单位 RMS，故与真实权重无关。
- 反向 frexp/ldexp **必要**（条件性）：上游梯度 O(1) 时 dO amax 3.0~5.3、无一下溢；但真实 `mean(y²)`
  loss（0.399）下单层 `dX.amax = 8.98e-7`，即上一层 dO 已在 **0.0147× fp16 正规下限（6.1e-5）**、
  约 min-subnormal 的 15×；上游缩到 1e-6 时 layer1 有 **37.6%** 元素 < 5.96e-8、裸 `.half()` 直接归零 **24.5%**。
- 数值一致性（同权重 A/B，dir1 开 vs 关）：loss 0.398795 vs 0.398803（rel 1.8e-5）、out rel_l2 4.5e-3、
  dX rel_l2 4.8e-3，全 finite、无 inf/nan（差异量级即 bf16 输出舍入 ε/2 ≈ 3.9e-3）。

**ROI：已实测 vs 估算**

*已实测（L1，Krea-2，V100，`cuda.Event` 100 次中位）*：

| M | bf16/fp16 fwd | bf16/fp16 bwd | bf16/dir1 fwd | bf16/dir1 bwd |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 9.32× | 9.64× | **4.95×** | 5.14× |
| 1024 | 8.22× | 8.78× | 6.15× | 6.36× |
| 4096 | 7.98× | 8.87× | 6.39× | 6.54× |
| 16384 | 8.07× | 8.93× | 6.40× | **6.57×** |

*端到端（L1）*：V100、4864 tokens、NF4、swap20、20 步、**dir1 + `--krea2_attn_fp16` 双开**
44.8 → 11.06 s/step（4.05×）、峰值 6.89 GiB。**dir1 单开从未测过**（该文档 §6 自列未验证边界）。

*估算（非实测，公式明示）*：Anima 每 block 每 token 的 linear MACs ≈ 62.9M，其中 NF4 可覆盖的 MLP
只占 33.6M = **53%**；Anima 的 mixed-NF4 还只覆盖 49/56 个 MLP 模块（`mlp-nf4-except-layer1-blocks14-20`
排除了 blocks 14-20 的 layer1）。若 bf16 基线不变、只吃 dir1：
`t_new/t_old ≈ 0.42/5.5 + 0.58 ≈ 0.656` → **上限约 1.5×**（dir1 免费也只到 1.7×）。
对照 Anima **原生 fp16** 让 100% 的 linear 走 fp16 tensor core（V100 实测 fp16 96.4 vs bf16 11.38 TFLOPS）
→ 量级上 **4~8×**，严格支配 dir1 的 1.5×。

**⇒ ROI 判定：Anima + dir1 = 零代码成本、零可用收益（被原生 fp16 严格支配）。不建议投入。**

移植时缺一不可的 5 个前提：①卡必须 sm<80；②宿主必须是 NF4 packed `Linear4bit`（普通 bf16 `nn.Linear`
不吃 dir1）；③外层必须仍是 bf16 autocast（靠 8 位指数把 1e-9 梯度跨层运送）；④M 必须是真实 token 数
（M=1 时带宽受限，dir1 反而慢一倍）；⑤外层 `autocast(bf16)` 必须被 `autocast(enabled=False)` 屏蔽，
否则 fp16 输入被静默改回 bf16，优化全废。

### 3.2 attention：Anima 是 MHA，且 V100 默认本来就不是 bf16

**形态修正（L0）**：`library/anima/models.py:632-663` `Attention(n_heads=16, head_dim=128)`，self 侧
`qkv_proj(2048→6144)`、cross 侧 `q_proj + kv_proj` 全用同一个 `n_heads`；`compute_qkv`（:703-709）只用
`self.n_heads`，**没有第二个 head 数** → **MHA 16/16，不是 GQA**（Krea-2 才是 48:12）。
⇒ `_fp16_attention` 里的 GQA→MHA `repeat_interleave` 展开段（`attention_backend.py:257-261`）对 Anima 是
**死代码**。

**序列长度**：Anima 训练 N = 4032 / 4200（`CONSTANT_TOKEN_BUCKETS`，1MP 探针 4096；patch stride 16，
`(W//16)*(H//16)`）；Krea-2 单流合并序列 4608 / 4864（= 4032/4200 + 576/664 文本 token）→ **同量级**
（差 13–16%）。Anima 文本走 cross-attention，不占 self-attn 的 N²。

**t5 探针判定（L2，sm70 / torch 2.10.0+cu129 / 合成权重）**

| 条件 | 后端 | 单层峰值增量 |
| --- | --- | ---: |
| fp16 输入，无 autocast | EFFICIENT（`fmha_cutlassF_f16_aligned_32x128_rf_sm70`） | **16.0 MiB** |
| fp16 输入，`autocast(bf16)` | **MATH**（`magma_sgemmEx_kernel` 14.1 ms + `volta_sgemm_64x64_nn/tn` 5.95/5.53 ms） | **2408~2449 MiB** |
| fp16 输入，`autocast(bf16)` + 内层 `enabled=False` | 回到 EFFICIENT | 16.0 / 128.5 MiB |
| fp16 输入，`autocast(fp16)` | EFFICIENT | 16.0 MiB |

干扰项全部否证：mask（all-True bool / fp16 float）、`dropout_p=0.1`、自定义 scale、v 非连续
（`unbind` 视图 + `v_norm=Identity`）都不翻转后端。走真实 `library/anima/models.py::Attention` +
`dispatch_attention(attn_mode="torch")`：weights fp16 + `autocast(fp16)` = **128.0 MiB**；
weights fp32 + `autocast(fp16)`（accelerate 保 fp32 场景）= **152.2 MiB**；只有 `autocast(bf16)` 是
**2440.6 MiB**。两条 V100 生产路径都已经是 EFFICIENT。

**⇒ t2b 遗留的「未决冲突」判定**：[anima_1mp_speed_v100_20261025.md](anima_1mp_speed_v100_20261025.md):31
的「fp16 仍走 SDPA math 后端」在 torch 2.10.0+cu129 / V100 上**不可复现**；同一条记录里可复现的只有
bf16 态（数字吻合：2.4 GiB、`magma_sgemmEx` / `volta_sgemm`）。最可能是把 bf16 记录误写成 fp16
（`precision_policy` 的 bf16→fp16 改写发生在训练初始化，若测量早于/绕过该改写即得 MATH）。

**⇒ 平移 `_fp16_attention` 给 Anima，在 V100 fp16 生产路径上的步时收益 ≈ 0**（kernel 名、16 MiB 峰值、
调用次数完全一致）。t2b 原先估算的「30~50% 步时」**作废**（那是隐含「Anima 外层是 bf16」的估算，
该前提在 V100 默认路径不成立）。**它的真实价值是守卫**：任何让注意力在 `autocast(bf16)` 下执行的路径
都会瞬间从 16 MiB/层跌到 2.4 GiB/层（28 block 即 ~68 GiB，不可行），`autocast(enabled=False)` 是唯一
阻断点。同理 `_PreFp16SDPA.contiguous()` 在 torch 2.10 上已非必要（非连续仍 EFFICIENT），退化为保险。

**若要平移，需要（L0 清单，都不大）**：①注入点应放在 `networks/attention_dispatch.py` 的 `torch` 分派分支
（`attn_mode=="torch"` 处），而不是照搬 Krea-2 专用的 `_cudnn_attention`（它绑定 `dit._mask` 的 (B,1,L,L)
布尔外积 mask + `rearrange` 输出契约）；②砍掉 GQA 展开段，只保留 `_PreFp16SDPA` / `_PostFp16SDPA` 的
frexp/ldexp 三明治；③必须在 `autocast(enabled=False)` + `sdpa_kernel([...])` 里执行；④判断条件
`q.dtype == torch.bfloat16` 要改成「外层 bf16 且内层要 fp16」；⑤Anima 侧没有 `_ATTN_FP16_ENABLED`
消费点（走 `Attention.forward → dispatch_attention`），且 compat 的 `krea2_attn_fp16_ignored` 会对
`anima` 误报。

### 3.3 逐族 ROI 汇总

| 族 | dir1 | attention fp16 内层 | 结论 |
| --- | --- | --- | --- |
| `anima` | 算子零改可复用，但 `{sm<80 + mixed-NF4}` 被夹死 → 不可达；ROI ≤1.5× 被原生 fp16 的 4~8× 支配 | **收益 ≈ 0**（fp16 已 EFFICIENT）；仅作 bf16 回退守卫 | 不建议投入；先把 `precision_policy` 的 NF4 豁免补上（独立小改 bug fix） |
| `krea2_raw` | 已在生产（唯一豁免者），双开实测 4.05× | 已在生产（`--krea2_attn_fp16`） | 基准族，无嫁接需求 |
| `z_image` | 不可行：`base_compute` 必须 bf16 → sm<80 被改写成 fp16 → loader 抛错 | 不需要：`attn_mode=flash` 走 Diffusers varlen | 不可行 / 无收益 |
| `qwen_image_2_1` | 宿主最完整（all-linear 232 模块）但 `base_compute` 必须 bf16；且 flash/mixed-NF4 组合恒 error | 不需要（同上） | 潜在最大资产，但需先解契约 + 补数值证据（见 §4.3） |
| `moss_image` | 无 DiT NF4（`policy.FAMILIES` 无 moss）→ **N/A** | N/A | 不适用 |

### 3.4 已实测 vs 估算 一览

| 断言 | 类别 | 证据 |
| --- | --- | --- |
| Krea-2 双开 44.8 → 11.06 s/step（4.05×） | 已实测 L1 | [inner_path](krea2_nf4_fp16_inner_path_20261109.md) §1 |
| bf16/dir1 M=256 4.95× → M≥4096 6.4–6.6× | 已实测 L1 | §3.1 表 |
| bf16/fp16 M=256 9.32–9.64× → M≥1024 8.0–8.9× | 已实测 L1 | §3.1 表（**契约里的「9.2–9.9×」只在 M=256 成立**） |
| Anima V100 1MP 1.961 s/步、fp16 96.4 vs bf16 11.38 TFLOPS | 已实测 L1 | [anima_1mp_speed_v100](anima_1mp_speed_v100_20261025.md) |
| Anima fp16 MHA hd=128 → EFFICIENT 16.0 MiB/层 | 已实测 L2（合成权重） | §3.2 表 |
| dir1 对 Anima 上限约 1.5× | **估算** | §3.1 公式（MLP 占 53%、dir1 5.5× 保守假设） |
| 「平移 attention 给 Anima 省 30~50% 步时」 | **作废的估算** | 前提（外层 bf16）在 V100 默认路径不成立 |
| dir1 单开耗时 | **未测** | §6 |

## 4. 象限三（不支持 FP16 的族）：物理/架构拒止根因档案

### 4.1 死因定义

| 编号 | 名称 | 判别特征 | 能否靠 Krea-2 资产救 |
| --- | --- | --- | --- |
| 死因1 | **真·前向物理越界**（层局部或全模型） | fp16 forward 出现非有限输出（幅值超 65504） | **不能**：dir1 前向不做缩放（`nf4.py:98` 注释假设激活 O(1e-2~1e0)）且开关是全局 `_DIR1_ENABLED`（`nf4.py:17`，无 per-layer 粒度） |
| 死因2 | **真·反向下溢 + 全局窗口关闭** | 前向正常，但 dY 全落 fp16 subnormal 区，loss-scale k 撞标量上限 | **能**：外层保 bf16 承载 1e-9 量级梯度 + 内层 dir1 fp16 GEMM |
| 死因3 | **契约一刀切（假防御）** | 只有 dtype 检查，没有任何量被证明会溢出；零数值证据 | **视宿主而定**：有 `Linear4bit` 宿主就能嫁接 dir1 |

### 4.2 分类总表

| 族 | 死因 | 是否潜在可放开 | 一句话判据 |
| --- | --- | --- | --- |
| `krea2_raw` | **死因2** | 纯 fp16 否（已被实测否决）；dir1 属**已兑现资产** | 1024² k 自收敛 5.900% 而上限 15 |
| `z_image` | **死因1** | 局部（islands 已开且是正解）；纯 fp16 不建议 | 唯一通过配置是 276 个 Linear 全 FP32 |
| `qwen_image_2_1` | **死因3** | **是**（最有希望），但须先补 islands + 数值证据 | 零数值证据 + flash 的 bf16-only 是假的 |
| `moss_image` | **死因1** | 不是「放开」问题，是「要补守卫」 | 零拦截但无 fp32 守卫；同架构 B 侧 13/20 步 NaN |
| `anima` | 不在象限C（已支持，对照锚点） | — | 12/12 finite 但严门槛 0/12 |

### 4.3 逐族档案

**`krea2_raw` → 死因2（真·反向下溢 + 全局窗口关闭）**

- 拦截分三层，注意 **compat 层其实没有拦截**：`compat_matrix.py` 精度相关只有 `krea2_attn_fp16` 的
  warning/info（:644-668、:742-770）与 `krea2_invalid_attn_mode`（:1253-1259）；唯一硬拦截在 NF4 面
  （`weights.py:78`、`weights.py:119`、`quantize.py:46`、`mixed_nf4.py:56`、`format.py:24`）。
- 「全网纯 fp16」入口：`model_loading.py:366` `loading_dtype = bf16 if adaptive else weight_dtype`
  → 非 adaptive 时 `mixed_precision=fp16` 会把 13B 基座整份载成 fp16。
- 前向确实正常（与死因1 的分界）：[adaptive_runtime_20260921.md](adaptive_runtime_20260921.md) 三层校准
  （`blocks.0/27.attn.wq`、`blocks.27.mlp.down`）T10 fp16 全通过；t10-krea 264 个 FP16 Linear 三步 loss 有限。
- 外层 bf16 + 内层 dir1 fp16 成立：`nf4.py:72-129` 外层 `compute_dtype` 保持 bf16 承载 1e-9 梯度、
  内层 fp16 GEMM + frexp/ldexp 平移；实测端到端 3.31~3.33×、双开 4.05×、150 步无 NaN。
- **是否可放开：不是待放开资产**（纯 fp16 已被实测否决）；**dir1 属已兑现资产**。

**`z_image` → 死因1（真·前向物理越界，层局部）**

- 拦截：`compat_matrix.py:1349-1358` `z_image_bf16_only`（C3，恒 error；放行条件 = adaptive islands+fp16
  或 fp32，且 `attn_mode ∈ {torch, sdpa, native}`）；`weights.py:243`（mixed-NF4 必须 bf16）；
  `attention_backend.py:58`（flash 必须 bf16）。
- 判别依据是**数值性、本仓实测**：[adaptive_runtime_20260921.md](adaptive_runtime_20260921.md)：T10
  `fp16-islands` 全模型首个 forward loss 非有限；逐层校准 `layers.28.attention.to_q` σ=0.8 输出非有限、
  `noise_refiner.0.feed_forward.w2` 三个 σ 全非有限；最终唯一通过配置是 **276 个 Linear 全 FP32**
  （原文「实际没有 FP16 Linear，不属于混合精度加速成果」）。非有限发生在 Linear 输出 = 幅值超 fp16 上限，
  与硬件无关，可外推到 sm80+。
- **gate 比证据更宽**：实测证据只指向个别层，而契约一律拒绝到 islands。附带文档缺陷：
  `compat_matrix.py:227` 把该 code 的理由写成「flash+fp16 raises」，与 1349-1358 的实际条件
  （torch+fp16 无 islands 也拒）不一致，建议后续修正。
- dir1 不适用：前向不缩放 + 全局开关无 per-layer 粒度 → 越界层无法单独排除。

**`qwen_image_2_1` → 死因3（契约一刀切，最大潜在资产）**

- 拦截：`compat_matrix.py:888-903`（flash → `qwen_image_2_1_mixed_precision_flash` 恒 error；
  mixed-NF4 → `_nf4` 恒 error；其余 fp16 组合仅 C1 warning）；`attention_backend.py:252`（flash dtype≠bf16
  raise）；`training_blocks.py:87/92`（fused mlp/qkv 要求 bf16 权重）；`nf4_compile_runtime.py:49-58`
  （232 个 packed module `compute_dtype` 必须 bf16）；`mixed_nf4.py:56`。
- 判别依据是**契约性**：所有检查对象都是 dtype 本身，没有任何量被证明会溢出；
  `grep -rn "fp16|float16" docs/findings/qwen*.md` 结果为空；C1 理由自述「the qwen torch/sdpa path has
  no fp16 guard (only flash requires bf16); qwen fp16 has no accepted real-machine run」
  （`compat_matrix.py:142-145`）。
- **flash 的「bf16-only」是假的**：`networks/attention_dispatch.py:129-130`
  `flash_attn_available_for_dtype(torch.float16) → True`（provider 本身支持 fp16），qwen 侧却在问 provider
  之前按 dtype 拒。
- 结构性风险（放开前必须知道）：qwen 的 adaptive precision 是 C2-ignored（loader 从不装 islands）
  → **qwen 是唯一「允许 fp16 且完全无 fp32 岛」的族**。
- 最小改动面（只评估，不实施）：①`attention_backend.py:252` 改成
  `dtype in {fp16, bf16} and flash_attn_available_for_dtype(dtype)`；②`compat_matrix.py:893-896` 的 forbid
  改按 provider 能力判定；③mixed-NF4 + fp16 不改外层 dtype，改走 dir1（qwen 是唯一 all-linear NF4 policy
  族 → **dir1 宿主最完整**）。
- **是否可放开：是（最有希望），但零数值证据 + fused projections fp16 完全未测 + dir1 只在 sm<80 生效。**

**`moss_image` → 死因1（前向越界风险；跨实现证据，本仓未复现）**

- **零拦截**：`moss_compat.py:45` 显式接受 `mixed_precision ∈ {bf16, fp16, no}`，只把 `base_compute`
  锁 bf16；TE 侧 `integration.py:10` 硬钉 `dtype=torch.bfloat16`，因此 `text_encoder_runtime.py:24/56`
  的「requires BF16」raise 在默认接线下**永不触发**（TE 与 DiT 精度已解耦）。**所以 MOSS 不可能是死因3**。
- 判别依据：同架构 B 侧 FP16_FIX 四模式全败（`online` 14 步 10 步 NaN、`fp32_residual` 13/20 步 NaN），
  结论「越界发生在算子内部，改存储救不了」；B 文档自述 base stream 峰值 **6.2e4**（距 65504 仅 ~5% 余量）。
- 本仓反证与边界：sm75、512²、16–32 步 loss 全 finite，但原文明确「不构成 fp16 稳定」；1024² 未测。
- 无 fp32 残差守卫（已核实 `math.py:107-112` 残差全程在计算 dtype；只有 `_RMSNorm.forward` 内部 upcast）。
- dir1 不适用：无 DiT NF4 → 没有 `Linear4bit` 宿主。
- **是否可放开：不是放开问题，是补守卫**；建议先做 1024² 的 fp16 vs bf16 对照探针。

### 4.4 两条判据（请连同死因表一起引用）

1. **dir1 的适用性是「前向 amax」属性，不是「族」属性**：前向越界的族（死因1）dir1 天然免疫；
   前向正常、只有反向下溢的族（死因2）dir1 才能曲线救国。
2. **「代码里写了 raise」≠「物理上不行」**：8 类精度 raise 里 7 类检查的是 dtype 本身；唯一有前向实测
   支撑的是 z_image（且证据只指向个别层）与 moss（跨实现）。

### 4.5 象限三内部可嫁接性排序

**qwen（有 all-linear mixed-NF4 宿主、前向无越界证据）> z_image（有 NF4 宿主但越界层不可排除）>
moss（无 NF4 宿主，0 可嫁接）**。krea2 自身不在象限三的可拓展面内——它已经是基准族。

## 5. 跨族系统级缺陷

### 5.1 P0-1：compat 在 precision 改写之前执行 → sm<8 绕过全部 fp16 fail-closed

- 顺序：`train_session.py:209` `assert_extra_args`（→ `extra_args.py:87` `check_training_compat`，
  另有 `bootstrap.py:189` 的第二次调用在数据集生成期）→ `train_session.py:217`
  `resolve_mixed_precision`。全仓只有这两处 `check_training_compat`，**都在改写之前**。
- 后果 1（z_image）：配置写 bf16 → 过 compat → 被改写成 fp16 → `z_image_bf16_only` **从未生效**
  → Z-Image 在 sm<8 上以未经 islands 的结构跑 fp16。
- 后果 2（qwen）：bf16+flash 过 compat → 改写成 fp16 → `qwen_image_2_1_mixed_precision_flash`
  （本该在启动期报错的硬 code）失效，改为首次 forward 时由 attention backend 抛 RuntimeError。
- 后果 3（krea2）：即 §2.1 的「无守卫 fp16 残差流的实际触发方式」。
- 修法：`resolve` 之后重跑一次 compat；或让 resolve 只做族感知改写（对 z_image / qwen 直接报错而不是改写）。

### 5.2 P1：precision_policy 的改写与所有预量化 mixed-NF4 族在 sm<80 上互斥

- 三条约束（`precision_policy.py:36-46/57-58`、`train_session.py:217/440`、四个 loader 的 BF16 断言）
  串起来把「预量化 mixed-NF4 权重」的可达集合压到只有 Krea-2。
- 后果：z_image / qwen / anima 的预量化 NF4 在 V100 / 2080Ti / T4 上会在 **model load** 处炸，
  错误信息是 "Mixed NF4 requires BF16 compute"，用户几乎不可能自行定位到 `precision_policy`；
  preflight 因为判定在改写**之前**所以放行。
- 现成修法：把 `compat_matrix._loads_mixed_nf4_weights` 接进 `resolve_mixed_precision` 的豁免条件
  （1 个函数 + 定向测试），**小改，不是零成本**。价值不在 dir1，而在「让这些族的预量化 NF4 在
  老卡上能起来」。

### 5.3 P0-3：drop-after-unscale 残留 UNSCALED stage

见 §2.5。

### 5.4 文档 / 文案口径缺陷（低优先，但会误导排障）

- `compat_matrix.py:227` 把 `z_image_bf16_only` 的理由写成「flash+fp16 raises」，实际条件更宽
  （torch+fp16 无 islands 也拒）。
- `krea2_attn_fp16_ignored`（`compat_matrix.py:783-792`）与 `nf4_dir1_ignored`（:775-781）按
  `base_compute` / family 判定，会对「anima + 预量化 NF4」给出与运行时**相反**的提示。
- `AGENTS.md` 的族清单只写 4 族，与 `MODEL_FAMILY_REGISTRY` 的 5 族属口径差异；MOSS 的精度契约位置
  （`moss_compat.py` 而非 `compat_matrix.py`）也应写明。

## 6. 未测与证据缺口（必须与结论一起读）

| 缺口 | 现状 | 补齐方式 |
| --- | --- | --- |
| z_image / qwen / moss 的逐层 dY 下溢测量 | **未测**（只有 krea2 与 anima 有；机制可类推、数字不可） | 按 `bench/krea2_nf4_fp16_grad` 口径做只读离线探针 |
| krea2 基座（`mixed_precision=fp16` 非 NF4）的梯度偏差 | **未测**（现有数字全是 NF4 fp16 内层 GEMM；`capture_real.set_compute_dtype` 只作用于 `Linear4bit`，不适用 bf16 基座） | 需新探针：25 GB bf16 基座 + `enable_block_swap`，成本远超 3~5 分钟 |
| **dir1 单开耗时** | ✅ **已补测（L2，2026-11-10）** | 见下方 2×2 消融表 |
| 真实权重下的 dir1 → Anima | **未测**（t5 用合成随机权重、单/双层；只证明 dtype 语义与量级） | 真实 Anima + 真实 mixed-NF4 权重 |
| moss 1024² fp16 行为 | **未测** | 1024² fp16 vs bf16 对照 |
| z_image / qwen 在 sm80+ 的 fp16 表现 | **未测** | 3080 真机 |
| `_cast_qkv_for_flash` 在 fp16 训练下的 backward dtype 一致性 | **未测**（只做静态读码） | 定向探针 |
| Anima `compile_seq_bands` / native flatten 与 fp16 的交互 | **未测**（`native_family.py:36-40` 只校验 block 数，未校验 band） | shape/invariant 测试 |

### 6.1 dir1 单开耗时：2×2 正交消融（L2，2026-11-10 补测）

此前「44.8 → 11.06 s/step（4.05×）」是 **dir1 + `--krea2_attn_fp16` 双开**且跨修复版本
（44.8s 测于 autocast 截胡未修复前）的数字，无法分离出 dir1 单独的贡献。本次用
`/tmp/probe_2x2.py` 在 256² 上跑完整四格，**同一版本、交替运行取中位数**：

| arm | dir1 | attn_fp16 | s/step | vs baseline | peak GiB |
| --- | --- | --- | ---: | ---: | ---: |
| baseline | ✗ | ✗ | 8.918 | 1.00× | 8.86 |
| **dir1** | **✓** | ✗ | **2.735** | **3.26×** | 8.96 |
| attn | ✗ | ✓ | 8.761 | 1.02× | 8.57 |
| both | ✓ | ✓ | 2.579 | 3.46× | 8.67 |

正交分解（两项共享同一份端到端预算，**乘积会高估，不能相乘**）：

| 组合 | 加速 | 说明 |
| --- | ---: | --- |
| dir1 单独（attn 关） | **3.26×** | 8.92s → 2.74s，**这是 dir1 的纯净贡献** |
| dir1 叠在 attn 之上 | 3.40× | 8.76s → 2.58s |
| attn 单独（dir1 关） | **1.02×** | 8.92s → 8.76s，**256² 上几乎无收益** |
| attn 叠在 dir1 之上 | 1.06× | 2.74s → 2.58s |

**两条结论**：

1. **dir1 是绝对主力**：单独就是 3.26×，与 256² 历史 `dir1_e2e` 的 3.31~3.33× 一致。
2. **attn_fp16 的收益强烈依赖序列长度**：256²（1024 tokens）下只有 1.02×，因为
   attention 在总预算里占比很小；它的价值要到 1024²（4864 tokens，attention 吃
   `2.4 GiB/层` 的 MATH）才显现。这与 §3.2「`_fp16_attention` 是 bf16 回退守卫」的定位
   一致——它是**长序列的止损**，不是短序列的提速。

峰值显存四格都在 8.57~8.96 GiB，差异在噪声范围内；256² 下两种优化都不改变显存曲线。

另有两项口径提醒：

- t5 的 11.7 ms vs 39.4 ms kernel self-time 是**单次 profiler 采样**，不是稳定 benchmark，
  **不得当作性能结论**；能作结论的是 kernel 名（`fmha_cutlassF_f16_..._sm70` vs `magma_sgemmEx_kernel`）
  与峰值增量（16.0 MiB vs 2408~2449 MiB）的量级差。
- 本报告未覆盖：`full_fp16`（optimizer 持 fp16 参数，`accelerator.py:281-297` patch）、PP v0 fp16
  （`library/training/pp/contract.py` 有独立 fp16 契约）、多进程 fp16 窗口事务（另有专文）。

## 7. 附录：Krea-2 提炼的「四大 FP16 破局武器」跨模型复用清单

| 武器 | 一句话定位 | 代码位置 | 证据等级 |
| --- | --- | --- | --- |
| 1 dir1 指数平移 | 破**反向** 1e-9 下溢；免 GradScaler 全局 scale；不改 BF16 契约 | `library/quantization/nf4.py::_Dir1MatMul4Bit` | L1（Krea-2 端到端 4.05×）+ L2（数值 A/B） |
| 2 fp16 注意力三明治 | **bf16 回退守卫**：把 MATH（2.4 GiB/层）拉回 EFFICIENT（16 MiB） | `library/models/krea2_raw/attention_backend.py::_fp16_attention` | L2（sm70 后端判定）+ L1（Krea-2 双开） |
| 3 `autocast(enabled=False)` 内层护盾 | **唯一必需件**：阻断外层 autocast(bf16) 越权截胡 | attention 与 dir1 两处 | L2（唯一翻转后端的变量） |
| 4 FP32 loss 种子 + `copy.copy(quant_state)` | 破 loss 侧 fp32 不可用 + 免污染共享 qstate | `library/training/losses.py:310-330`、`library/quantization/nf4.py:dequant_fp16` | L0/L1（全族共享，见 §2.4） |

### 武器1：dir1 算子内 frexp/ldexp 指数平移

- **解决什么物理问题**：fp16 的最小正规数 6.1e-5，而中深层 dY 是 6.5e-9~1.7e-7（264 个 Linear 的
  grad_output 除 2 个外全部低于该下限；512² 的 latent grad 100% < 6.1e-5、69.3% 精确归零）。
  全局 GradScaler 能救但会在高分辨率撞标量上限（k 上限 15，
  1024² 需 k≥16~18）。dir1 把「指数搬运」下沉到算子内部：前向 fp16 GEMM、反向用 `frexp` 取指数、
  `ldexp` 搬回，`clamp [-126, 126]` 双向兜底，**外层 `compute_dtype` 仍是 bfloat16**，
  于是用 bf16 的 8 位指数承载跨层梯度量级、用 fp16 的 tensor core 做内层矩阵乘。
- **代码在哪**：`library/quantization/nf4.py::_Dir1MatMul4Bit`（:72-129）、
  `BF16AutocastLinear4bit._forward_dir1`（:104-109）、开关 `set_dir1_enabled`（:31-47，sm≥80 抛
  ValueError）、全局标志 `_DIR1_ENABLED`（:17）。
- **已被哪些实测验证**：L1 —— 双开 44.8→11.06 s/step（4.05×）、150 步无 NaN；算子级 bf16/dir1 从
  M=256 的 4.95× 升到 M≥4096 的 6.4–6.6×；256² fp32 裁判下 dir1 梯度 40.974%（cos 0.92553）≈
  bf16 的 40.614%（cos 0.92764），即 **dir1 保住精度但没提升精度**。L2 —— 合成权重 dir1 开/关
  loss 0.398795 vs 0.398803（rel 1.8e-5）、dX rel_l2 4.8e-3，全 finite。
- **移植到别族的前提**：①sm<80；②宿主必须是 NF4 packed `Linear4bit`（普通 `nn.Linear` 不吃）；
  ③外层保持 bf16 autocast；④M 是真实 token 数（M=1 反而慢一倍）；⑤外层 autocast 必须被
  `enabled=False` 屏蔽；⑥bias 分支假定 2D 输入（`bias=False` 的 MLP 不触发）；
  ⑦`{anima, mixed-NF4, dir1, sm<80}` 在运行时不可达（§3.1），**别把 Krea-2 的 `base_compute=nf4`
  豁免误当成 Anima 可用**。

### 武器2：SDPA 的 `_PreFp16SDPA` / `_PostFp16SDPA` 三明治（含 GQA 展开）

- **定位修正**：**不是「提速武器」，是 bf16 回退守卫**。在 fp16 生产路径下收益为 0；它的真实价值是
  在 `autocast(bf16)` 环境里把 MATH（2408~2449 MiB/层）拉回 EFFICIENT（16.0 MiB/层）。
- **解决什么物理问题**：sm70 上 bf16 没有 EFFICIENT/FLASH/CUDNN kernel，只剩 MATH，而 MATH 要物化
  `2 × H × N² × 4` 的分数矩阵（Krea-2 单层 10 GiB 量级；Anima N=4032/H=16 实测 2.4 GiB）。
  Krea-2 另有 GQA 48:12，需要用 `repeat_interleave` 把 kv 头展开成 MHA 才能吃到 fp16 kernel。
  同时反向下溢仍在（dO 到 1e-6 量级），所以 frexp/ldexp 不能省。
- **代码在哪**：`library/models/krea2_raw/attention_backend.py::_fp16_attention`（:263-290）、
  `_PreFp16SDPA` / `_PostFp16SDPA`（:78-125）、GQA 展开段（:257-261）。
- **已被哪些实测验证**：L2 —— fp16 → EFFICIENT（`fmha_cutlassF_f16_aligned_32x128_rf_sm70`，16.0 MiB）；
  `autocast(bf16)` → MATH（`magma_sgemmEx_kernel` 14.1 ms + `volta_sgemm_64x64_nn/tn`，2408~2449 MiB）；
  内套 `enabled=False` → 回到 EFFICIENT；mask / dropout / scale / 非连续 v 都不翻转。
  真实 `Attention` 路径 128.0 / 152.2 / 2440.6 MiB。L1 —— Krea-2 双开 4.05×。
- **移植到别族的前提**：①注入点在 attention 分派层（`networks/attention_dispatch.py` 的 `torch` 分支），
  不要照搬绑定 `dit._mask` 与 `rearrange` 输出契约的 `_cudnn_attention`；②Anima 是 MHA → GQA 展开段是
  **死代码**，直接省掉；③frexp/ldexp 仍必要（`dX.amax=8.98e-7` = 0.0147× fp16 正规下限；上游 1e-6 时
  layer1 有 37.6% 落 subnormal、裸 `.half()` 归零 24.5%）；④`contiguous()` 在 torch 2.10 上非必要，
  保留只作保险；⑤**收益预期要按「守卫」写**：fp16 生产路径 ≈ 0，bf16 路径才是救命的。

### 武器3：`with torch.autocast(device_type="cuda", enabled=False):` 内层护盾

- **解决什么物理问题**：外层 `autocast(bf16)` 会把内层手写的 fp16 计算**静默**抬回 bf16——不报错、
  不告警，只是慢内核 + 精度反降。这是「内层 fp16 岛」存在的唯一前提，也是 t5 探针里**唯一**能翻转
  SDPA 后端的变量。
- **代码在哪**：`library/quantization/nf4.py:104`（dir1）、
  `library/models/krea2_raw/attention_backend.py:283`（attention）、
  `library/models/krea2_raw/native_precision.py:11/85`（fp32 岛）。
- **已被哪些实测验证**：L2 —— 同一份 fp16 q/k/v：无 autocast = EFFICIENT(16 MiB)；
  `autocast(bf16)` = MATH(2408~2449 MiB)；`autocast(bf16)` + `enabled=False` = EFFICIENT(16 MiB)。
  L1 —— [inner_path](krea2_nf4_fp16_inner_path_20261109.md) 记录的两次实例，未修 44.8 s/step
  与修好后 11.06 s/step 的差异来源就是这两处护盾。
- **移植到别族的前提**：①这是**唯一必需件**，其他三件可视场景省略；②护盾只挡「外层 autocast」，
  挡不住「调用方自己 `.to(dtype)`」；③`_Dir1MatMul4Bit.backward`（`nf4.py:124` 附近）当前**没有**这层
  护盾——现在 backward 在 autocast 区外无实害，但一旦被放进 autocast 就会静默失效，建议补上。

### 武器4：FP32 loss 种子 + `copy.copy(quant_state)` 无状态浅拷贝

- **解决什么物理问题**：①loss 侧——`target` 与 `model_pred` 双 `.float()` 后才算 MSE，避免 fp16 下
  `(v-t)²` 在 int16 级别分辨率上抖动；②量化状态侧——`dequant_fp16` 需要把 `QuantState.dtype` 改成 fp16
  再反量化，直接改会污染共享对象，浅拷贝后改才安全（多副本/多模态共用同一 `quant_state` 时尤其关键）。
- **代码在哪**：`library/training/losses.py:310-330`（`_flow_match_loss` 双 `.float()`）、
  `library/quantization/nf4.py::dequant_fp16`（:50-69，`copy.copy(QuantState)` 后改 dtype）。
- **已被哪些实测验证**：L0/L1 —— loss 侧全族共享（anima/krea2/z_image/qwen/moss 都只产出
  `(model_pred, target)`）；quant_state 侧由 5 处 fail-closed（`weights.py:79`、`weights.py:244`、
  `mixed_nf4.py:57`、`nf4.py:196`、`training.py:18-20`）共同约束。
- **必须补的一句**：**fp32 loss ≠ fp32 梯度**。`.float()` 是带 backward 的 cast，梯度会舍回
  `model_pred` 的 fp16；Krea-2 文档 §5.1 实测该缺口（`2*(v-t)/N ≈ 2e-5`，在 fp16 里只剩 0~2 位）。
  所以「loss 用 fp32 算」不能作为精度论证，真正的精度资产是武器1 的算子内指数平移。
- **移植到别族的前提**：loss 侧零前提（已全族共享）；`quant_state` 侧只对 NF4 相关族适用，
  MOSS 没有 DiT NF4 故不涉及。

## 8. 复现入口（只读，不含训练）

- 静态普查：`git log -1 --oneline` 对齐 `9d530588`；按本文 `file:line` 逐条核对源码。
- sm70 后端探针（t5 用过的口径，脚本留在 `/tmp`、未入库）：`torch.backends.cuda.can_use_efficient_attention`
  + `torch.profiler` 记 kernel 名与 `max_memory_allocated` 增量；真实模型路径用
  `library/anima/models.py::Attention` + `dispatch_attention(attn_mode="torch")`。
- 已有的梯度裁判口径：`bench/krea2_nf4_fp16_grad/`（fp32 臂为唯一裁判）。

本报告未改动任何源码、未运行训练、未触碰数据集 / models / output / logs。
