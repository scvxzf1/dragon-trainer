# 个性化训练：区域课程、整图回归与自适应正则化

状态：半活跃提案（foreground_mean、观测 controller、动态 weighting/affine 已落地；真实图像质量实验待数据）  
适用版本：当前 main
相关代码：`library/training/stage_schedule.py`、`library/training/losses.py`、`library/datasets/subsets.py`

## 结论

**方向可行，但当前 `masked_loss` 不等于“区域强化加速拟合”。**

仓库已经能将同一批图片拆成有前景 mask 的区域阶段、无 mask 的整图阶段，并按全局 step 百分比切换。区域阶段可用 `inverted_mask_prior_weight` 将 mask 外部锚定到关闭 adapter 的 base 预测；整图阶段可用 `prior_preservation_weight` 约束整张图的 base 行为。因此“先隔离主体、后恢复整图”的消融可以立即开展。

但现有 masked loss 是 `loss * mask` 后对**整张** latent 平均。小物体的总梯度会随 mask 面积缩小：它实现的是“忽略背景”，不是保持区域内梯度尺度的“区域强化”。要验证“前期加速拟合”，最小新增项是**按 mask 面积归一化的区域 loss**；自适应仿射增广和按扩散时间桶的 loss 权重是独立的第二阶段。

## 本地论文证据与边界

| 本地资料 | 可复用结论 | 本提案的边界 |
| --- | --- | --- |
| APT，CVPR 2025，§3.1、§3.4、§4.5–4.6：`papers/personalization/01_APT/APT_CVPR2025.txt` | 用每个扩散时间桶中 base/微调模型去噪损失的 EMA 构造过拟合指标，以它调节仿射增广概率和 `(1-γ_t)` 去噪损失权重；低噪声桶更早记忆化。 | 自适应信号必须按 timestep bin，不能把固定后半程降权称作 APT 复现。APT 还含表示稳定化和 attention 对齐，ATA 单项必须消融。 |
| Break-A-Scene，2023，§3–4、Eq. (1)：`papers/personalization/03_Break-A-Scene/Break-A-Scene_arXiv_2305.16311.txt` | 掩码扩散重建能将身份信号限制在目标区域；先受限、后低学习率放开参数，可折中重建与可编辑性。 | 支持“先局部、后放开”的课程思想；其第一阶段是 token-only，不是本仓 LoRA 路径，不能外推固定切换比例。 |
| DreamBooth，2022，§3.3：`papers/personalization/04_DreamBooth/DreamBooth_arXiv_2208.12242.txt` | 少样本微调会语言漂移、丢失姿态/视角多样性；类先验保持可缓解。 | 整图背景重建不足以保护通用先验，必须同时评估不含 trigger 的通用 prompt。 |
| SID，CVPR 2024，§4、§7.1：`papers/personalization/02_SID_Selectively_Informative_Description/SID_CVPR2024.txt` | 背景、邻物、姿态会与主体缠结；分割有帮助但有局限，caption 信息选择同样关键。 | 区域课程不能替代干净 caption、正确类别词和多视角；错误 mask 会被更快强化。 |

## 现有能力与限制

| 需求 | 当前事实 | 结论 |
| --- | --- | --- |
| 分阶段数据 | `stage_schedule` 覆盖 `[start_pct,end_pct)`，切换时重建 bucket/DataLoader；首次过滤前保存全量数据快照。 | 两阶段课程可立即使用；所有阶段 cache 必须在开训前完成。 |
| 前景 mask | `mask_dir` 读 `{stem}_mask.png` 并产生 `alpha_masks`；未指定目录会自动探测 `post_image_dataset/masks`。 | 可作区域阶段输入。 |
| 普通 mask loss | 默认 `mask_loss_normalize="none"` 保留“相乘后对全空间平均”；可显式切到 `foreground_mean`。 | 旧配置数值不变；区域强化必须显式启用面积归一化。 |
| 先验保持 | `inverted_mask_prior_weight` 在 mask 外比较 adapter/base；`prior_preservation_weight` 在整图比较 adapter/base。 | 可作低成本正则；后者不是 DreamBooth 的独立生成类图像池。 |
| 增广与权重 | subset 有静态增广；新增 controller 可按 bin 记录 `gamma_b`，并以独立开关驱动动态 denoise weighting 与同步 affine。 | 默认仍关闭策略；先用 JSONL 离线核对，再做 on/off 配对。 |

### 两个关键陷阱

1. `masked_loss = false` **不能**让仍携带 `alpha_masks` 的 batch 变整图 loss：当前条件是“`masked_loss` 为真 **或** batch 有 alpha mask”。整图阶段必须不提供 alpha mask。
2. `mask_dir = None` 会自动探测全局 mask 目录。整图数据行应显式写 `mask_dir = ""` 并设 `alpha_mask = false`。两行必须使用不同的 `cache_dir`，否则区域 cache 中的 alpha sidecar 可能被整图阶段读回。

现有[训练指南的 masked loss 小节](../guidelines/training.md#masked-loss-sam--mit)已在同次修改中更正这项行为。

## 不改代码即可运行的基线

下面是独立用户 TOML 的骨架（包含 `[[stage_schedule]]` 与两段 `[[datasets]]` 定义）；不要覆盖 `configs/base.toml`。数值只是第一个实验点，必须消融。

```toml
masked_loss = true
inverted_mask_prior_weight = 0.05  # 区域阶段：仅 mask 外的 base 锚点
prior_preservation_weight = 0.05  # 整图阶段：扫描 0/.02/.05/.1

stage_schedule_enabled = true
[[stage_schedule]]
name = "区域归因"
subset_index = 0
start_pct = 0.0
end_pct = 0.35

[[stage_schedule]]
name = "整图回归"
subset_index = 1
start_pct = 0.35
end_pct = 1.0

[general]
caption_extension = ".txt"

# 行 0：同图同 caption，带 mask，独立缓存
[[datasets]]
batch_size = 1
  [[datasets.subsets]]
  image_dir = "post_image_dataset/resized"
  cache_dir = "post_image_dataset/lora_region"
  mask_dir = "post_image_dataset/masks"
  alpha_mask = true
  recursive = true

# 行 1：同图同 caption，明确没有 mask，独立缓存
[[datasets]]
batch_size = 1
  [[datasets.subsets]]
  image_dir = "post_image_dataset/resized"
  cache_dir = "post_image_dataset/lora_full"
  mask_dir = ""
  alpha_mask = false
  recursive = true
```

运行前：

1. 抽查空 mask、近满幅 mask、遮挡、细发丝、手持物、粘连背景；区域阶段会放大标注错误。
2. 两个 `cache_dir` 都须完成 VAE/text cache；stage 调度不会在运行中预处理。
3. 用短 smoke run 确认日志有两个 `[stage]` 事件，切换后整图 batch 不带 `alpha_masks`；切换点保存 checkpoint 和同一套 sample prompt。
4. 复制并按本机路径调整 [`docs/examples/personalization_region_to_full.toml`](../examples/personalization_region_to_full.toml)，例如：

```bash
.venv/bin/python tasks.py lora METHOD=lora PRESET=default \
  --dataset_config docs/examples/personalization_region_to_full.toml
```

   若完整训练 TOML 已提供数据块，不要再以另一份 `--dataset_config` 重复定义；开训前用 `print-config` 审计。

以 `35% → 65%` 为首个点而非默认值，至少扫描 `20/35/50%`。单参考图宜缩短区域阶段，并先改善 caption。

## 已落地：真正的区域强化

已新增 `mask_loss_normalize = "none" | "foreground_mean"` 和 `foreground_loss_weight`，默认 `"none"` / `1.0`，保证旧配置数值不变。对单样本空间损失 (ℓ)、latent 尺度软 mask (M)：

    L_fg = sum(M * ℓ) / max(sum(M), eps)
    L_bg = sum((1-M) * ℓ) / max(sum(1-M), eps)
    L_data = w_fg * L_fg + w_bg * L_bg

区域阶段用 `w_fg=1,w_bg=0`；整图阶段不传 mask。区域平均后才乘 `ctx.loss_weights`，不改变 timestep weighting 语义。面积必须在 latent 分辨率计算；空/近空 mask 要带 stem 报错并跳过或回退整图，不能除以极小数。

- 实现位于 `library/training/mask_loss.py`，由 flow-match、VR、velocity-direction 共用；没有向 `train.py` 堆分支。
- 已同步 CLI、config schema 自动发现和 checkpoint metadata 字段。
- 已覆盖全 1/半 mask、面积不变性、空 mask、零权重、CLI 默认值、stage/cache 与 `mask_dir = ""` 回归。

## 第二阶段：APT 风格自适应增广与权重

已新增运行时状态容器和可观测的 `AdaptivePersonalizationController`，而不是让静态 `color_aug` 随 epoch 线性变化。

1. **观察：** 对同一 `x_t,t` 低频无梯度运行 adapter-disabled base forward；按 `timestep_bins` 记录 `EMA(L_base-L_adapter)`、样本数与 `γ_b∈[0,1]`。复用 `run_prior_preservation_forward()` 的 multiplier/块交换保护。
2. **行动：** 满足最小样本数后，`p_affine,b=clamp(k_aug*γ_b,0,p_max)`，`w_denoise,b=clamp(1-k_loss*γ_b,w_min,1)`。配置进入 TOML/metadata，运行指标进入 progress JSONL；冷启动为 `p=0,w=1`。controller EMA 当前是进程内状态，尚未承诺跨 resume 恢复。
3. **增广位置：** 对 cached latent、noise、noisy input、mask 和 conditioning image 使用同一 affine grid；由于 affine 是线性的，这与先同步变换 latent/noise 再混合等价。不得只移动 mask 或 conditioning image。
4. **关系与安全：** stage 是全局进度课程，`γ_b` 是扩散噪声层级闭环；小 mask 会扭曲 reference 差，首版 controller 应只在整图阶段启用或剔除低覆盖率样本。始终 `w_min>0`、限制 `p_max`；NaN、空 mask、reference 失败回退 `p=0,w=1` 并告警。

以下字段已可识别；策略默认关闭，便于严格 on/off 配对：

    adaptive_personalization_observe = true
    adaptive_personalization_timestep_bins = 10
    adaptive_personalization_ema_decay = 0.95
    adaptive_personalization_probe_every_n_steps = 4
    adaptive_personalization_loss_weighting = false
    adaptive_personalization_affine = false
    adaptive_personalization_min_bin_samples = 16
    adaptive_personalization_affine_probability_max = 0.5
    adaptive_personalization_denoise_weight_min = 0.25

不要直接复制 APT 的表示统计/attention 对齐：Anima DiT 的 attention layout、native token bucket、compile 边界不同，应先证明 ATA 与区域课程的增益。

## 四组基线与当前证据

先运行 CPU 合约基线：

```bash
timeout 180 .venv/bin/python -m pytest tests/test_personalization_baselines.py -q
```

结果为 5 passed，覆盖全程整图、区域 foreground_mean、区域→整图、区域→整图+prior 的有限 loss/stage/cache 合约。

随后用临时生成的两张 1008×1024 合成图、外部 PNG mask、独立 region/full VAE+TE cache，在 RTX 3080、`low_vram_blockswap`、plain LoRA rank 4 上完成真实训练链路 smoke。为保持每组极短，四组只选一张图；这些 loss 不可横向解释为质量优劣：

| 运行 | optimizer steps | JSONL 证据 | 峰值显存 |
| --- | ---: | --- | ---: |
| 全程整图 | 1 | `run_end=ok`，loss `0.00477150` | 4.07 GiB |
| 全程区域 `foreground_mean` | 1 | `run_end=ok`，loss `0.00077716` | 4.07 GiB |
| 区域→整图 | 2 | step 1 `stage=region`，step 2 `stage=full`，loss `0.00199477 → 0.00426942` | 4.11 GiB |
| 区域→整图 + inverted-mask prior 0.05 | 2 | 两阶段均完成，loss `0.00560414 → 0.00553087` | 4.41 GiB |

这次 smoke 还发现并修复两个真实接线缺口：外部 `mask_dir` 已预加载 mask 时不再强制 latent NPZ 内嵌 `alpha_mask`；用户 dataset TOML 的 `stage_schedule` 会在蓝图清洗前复制到运行参数，不再被“忽略”后静默失效。

controller 配对也已在真实训练链路完成：observer 两步记录 `observer_calls=2`、`count_bin_00=2` 和非零 `γ`；动态 weighting 在 `γ=0` 时保持 `denoise_weight=1`；四步 affine 强制敏感度 smoke 在后两步记录 `gamma=1`、`affine_fraction=1` 并正常结束。它只证明策略接线与回退，不证明 APT 收益。

仍须用真实个性化图片，在固定 seed/steps/rank/caption/prompt 下重跑质量四组并保存区域结束、整图中点、最终 checkpoint；完成盲评/身份与可编辑性指标前，不得把本文迁入 `docs/experimental/` 或 `docs/methods/`。

## 评估与准入

固定基础模型、seed、总 optimizer steps、LoRA rank、caption、sample prompt，并保存区域结束、整图中点、最终 checkpoint。比较全程整图、全程区域、区域→整图、区域→整图+prior；再逐项加入归一化、动态权重、动态增广。

不要只看 FM-MSE。至少评估：主体裁剪 DINO/CLIP-I 或盲评；换背景、远景、多人/多物、姿势 prompt 的 CLIP-T/盲评；不含 trigger 的同类和通用 prompt 与 base 对照；SID 的主体一致、非主体解缠、文本对齐三轴（或可复现近似）；以及每 bin `γ_b`、实际增广比例、前景面积、stage index、梯度范数。

只有在“身份不下降、通用/构图不显著退化”时接受方案。仅提高训练图重建或主体 close-up、却丢失背景/远景服从性，是过拟合。以固定预算下达到同等身份分数所需 step 数判断“加速”，不能只看区域阶段 loss 更低。

## 实施顺序

1. 已完成四组 CPU 合约基线和合成图 GPU 链路 smoke；真实个性化图片质量四组仍待执行。
2. 已落地 `foreground_mean` 和数值/缓存/阶段回归。
3. 已落地 controller 观测和 JSONL 指标；先不驱动策略，离线核对 `γ_b`。
4. 动态 loss weighting 与同步仿真 affine 已有独立开关和配对测试；真实图像实验中必须先开 weighting，再开 affine。
5. 真实配方稳定后再转入 `docs/experimental/` 或 `docs/methods/`；本文件继续保留研究风险和依据。
