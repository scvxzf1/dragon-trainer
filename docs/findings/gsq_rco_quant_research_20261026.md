# GSQ-RCO 量化主张对当前训练器量化栈的静态核查与研究记录

状态：研究记录（静态分析，未实现，不改默认）
适用版本：当前 main（dev @ 3b85e43）
日期：2026-10-26
入口命令：无（只读分析）
相关代码：library/runtime/convrot/、library/quantization/、library/inference/corrections/cns.py

> 本文只做**静态分析**：不改训练默认、不改量化实现、不启动训练、不动用户数据，仓库内不新增任何分支。
> 全文结论一律按三类标注分离，禁止混写：
> **[论文]** 论文原文主张（可 quote）；**[仓库]** 当前代码事实（file:line）或已有实测数值（须给口径）；**[推论]** 未测假设，只能记为 `H`。
>
> 证据强度分级：**实测级**（有可复算原始数据且口径完整）> **代码级**（读源码即证）> **文档记载**（文档称已测但原始数据缺失，不可复算）> **未测**（无任何证据）> **不可引用**（文档与 TODO / 缺失产物自相矛盾）。
>
> 本文**不宣称任何数值已验证**；所有数字都是转述既有证据或论文原文，并保留其原始口径与可复算性标记。

---

## 0. 结论速览

| 用户断言 | 判定 | 一句话依据 |
| --- | --- | --- |
| Q0 GSQ-RCO 落地后比 w4a16 强很多 | **不成立** | `--base_compute` 枚举无 `w4a16`；唯一 4-bit 权重路径是**未接线的静态 opt-in 实验** `library/runtime/nm_w4a16/` + `bench/nm_w4a16/probe.py` + CLI `exp-nm-w4a16`（2:4 INT4 / BF16 激活，无 checkpoint/compile/merge 集成、无质量证据）；GSQ/RCO 零实现；论文侧 4-bit 附近收益递减（GSQ 实验点只到 3-bit，无 4-bit 数据点） |
| Q1 相当于重新分配了各层权重 | **部分成立** | 读作"重新分配**位宽预算**"成立；读作"权重被重排/旋转"不成立 |
| Q2 显存 −10%、速度不变 | **部分成立** | 显存方向成立但**不是 −10%**（可复算 −16.216%，分母 BF16）；速度"不变"仅 90HX swap=12 的 +1.16% 单点近似 |
| Q3 修好交叉投影层 / AdaLN+FFN 被量化 / 4→2.5bpw | **不成立** | AdaLN 被**实测级反证**；FFN(=mlp) 被量化成立；4bpw/2.5bpw 量纲错误且来源不可追溯 |
| Q4 质量改进主要来自 CNS，不是 GSQ+RCO | **无实测路径（不可判定）** | 单 run 内 CNS 与量化不可达；GSQ+RCO 项恒未实现 → 该句是空真 |
| Q5 CNS 不够稳定 / GSQ-RCO 让 CNS 更稳定 | 前半**无实测证据**；后半**不成立** | CNS 画质/稳定性实测证据强度 = 0；交互项无定义域 |

一句话总结：**arXiv 2411.17525 不是 GSQ-RCO 论文**（是 HIGGS / Linearity Theorem），而"GSQ-RCO"这两个名字对应另外两篇独立 LLM 量化论文（GSQ=arXiv:2604.18556、RCO=arXiv:2605.00649），在本仓库全量 0 命中、零实现；因此 Q0–Q5 中凡涉及 GSQ-RCO 的效果主张，均**没有本仓可落地的对照物**，只能作为方向性参考，不能当作既成事实。

---

## 1. 论文身份核验

**[论文] arXiv 2411.17525 的真实身份**

- 标题（quote）："Pushing the Limits of Large Language Model Quantization via the Linearity Theorem"
- 作者：Vladimir Malinovskii、Andrei Panferov、Ivan Ilin、Han Guo、Peter Richtárik、Dan Alistarh（Yandex / HSE / ISTA / KAUST / MIT / NeuralMagic）
- 提交：`[Submitted on 26 Nov 2024]`，仅 v1；`cs.LG`；DOI `10.48550/arXiv.2411.17525`
- 方法代号（quote）：**HIGGS = "Hadamard Incoherence and Gaussian MSE-Optimal GridS"**
- 正文核心：Linearity Theorem（逐层 ℓ2 相对误差 `t_l²` → perplexity 增量的线性近似 `E[PPL(Ŵ)] ≈ PPL(W*) + Σ_l α_l t_l²`）+ 两个应用：(1) data-free 的 RHT + MSE-optimal grid 量化（HIGGS）；(2) 中位宽区间的 per-layer 非均匀 bit 分配。
- 方法边界：全文是 **LLM 权重-only 训练后量化（PTQ）**，与扩散模型 / DiT / 训练器显存吞吐工程无交集。
  - **词频口径（必须用词边界，否则必错）**：`rg -w -i 'DiT' /tmp/p2411/paper.txt /tmp/p2411/full.txt` → **0 命中**（`-w` 词边界；PDF/HTML 双通道均 0）。
  - 不用词边界时 `rg -i -o 'dit' /tmp/p2411/paper.txt | wc -l` → **11**；逐条核对上下文后，11 处**全部是 `additional` / `additive` / `bitwdiths`（以及 PDF 断行产生的 `ditionall`）的子串**，无一处是独立的 `DiT`。HTML 通道（`/tmp/p2411/full.txt`）无词边界命中 13 处，同样是 `Additional` / `Additionally` / `additive` / `bitwdiths` 子串。
  - `diffusion`、`activation quantization`、`W4A16`、`VRAM` 双通道命中同样为 0。

**[论文] "GSQ-RCO" 这个名字与 2411.17525 不匹配**

- GSQ 与 RCO 是**可独立检索到的另外两篇论文**（GSQ=arXiv:2604.18556，RCO=arXiv:2605.00649），不是 2411.17525 的内容。
- 引用关系不对称：GSQ 引用 RCO，而 **RCO v3 全文不引 GSQ**（`rg "GSQ"` = 0）→ 把两者当成一篇论文的"联合方法"没有文献依据；"GSQ+RCO 联合"在本仓唯一定量锚点只能是 GSQ Table 1 的 NU（非均匀）行。

**[仓库] 本仓库对这三个名字的命中情况**

| 关键词 | 仓库源码命中 | 结论 |
| --- | --- | --- |
| `GSQ` | 0（仅依赖目录噪声） | 零实现 |
| `RCO` | 0（仅依赖目录噪声） | 零实现 |
| `2411.17525` | 0（本次核查前 `docs/` 全树 0 命中） | 无既有引用 |
| `HIGGS` / `Linearity` | 0（实现层面） | 未移植 |

→ **判定**：把 "GSQ-RCO" 当作 2411.17525 的简称是**名称错配**；本文统一按「**待核别名**」处理：引用时必须拆成 GSQ(2604.18556) / RCO(2605.00649) / HIGGS(2411.17525) 三篇分别标注，禁止合并引用。

---

## 2. 可迁移组件与不可迁移部分

### 2.1 三篇论文的可迁移性分级

| 组件 | 来源 | 迁移到本仓（DiT 训练器） | 依据与阻塞点 |
| --- | --- | --- | --- |
| Hadamard / RHT 旋转（权重侧） | HIGGS §4.1 | **已部分存在，可直接复用** | 本仓 `library/runtime/convrot/rht.py` 已有 RHT 实现（默认 `dense`，rht.py:170-175），语义上是"权重旋转后再量化"；无需新写旋转 |
| MSE-optimal grid（Gaussian、vector quant） | HIGGS §4.2 | **理论可迁移，工程成本高** | 需要 Pagès–Printems 网格优化 + 预计算网格表（`n` 须为 2 的幂，网格点数 ≤ ≈2^10），且要配套 FLUTE 类查表 kernel；与现有 int8 GEMM 路径不兼容 |
| linearity theorem 的 α_l 标定（Gaussian noise insertion, J=15） | HIGGS §5 / Alg.3 | **概念可迁移（作为敏感度度量）** | 本仓无 per-layer 敏感度标定通路；但该思路与现有 `cmmd` / 逐层误差分析可对接 |
| per-layer 非均匀 bit 分配 | HIGGS §5（实际是 CP-SAT/ILP） | **不可直接迁移** | 本仓**没有 mixed-bit / per-layer 位宽选择面**：`base_compute` 只有 `bf16 / w8a16_convrot / w8a8_convrot / nf4`（cli_args.py:1103-1106；checks.py:68-89），CLI/config 面均无 bit-width 配置项。注意 `library/runtime/nm_w4a16/` 是**单一均匀 INT4 + 2:4 结构化稀疏**的静态实验（见 §2.2 第 5 条），不是 per-layer 混合位宽，不能拿它当 mixed-bit 通路 |
| Gumbel-Softmax 网格点选择（学 logits） | GSQ | **不可迁移（短期内）** | 需要可微的"选网格点"松弛 + 逐 block 三阶段重建 + 校准数据管线；论文在 8×H200 下 8B 需 10h 校准；且 GSQ 是 **非 data-free、非 QAT**，需要 calibration 前向，本仓无对应管线 |
| per-group bit 档位 logits + 等式预算约束 | RCO | **不可迁移** | 需要"同一套 GPTQ 候选库 + 多 bit 档位候选权重"作为前提；本仓不存在多档位候选权重库，也不存在 GPTQ 实现 |
| 精确 DP 解 MCKP + tangent projection 反向 | RCO | **不可迁移** | 依赖 additive cost 结构与 proxy objective；属于论文侧最重的实现部分 |

### 2.2 与本仓结构的硬冲突（代码级）

1. **量化粒度冲突**：本仓 ConvRot 是"**哪些模块被量化**"（`convrot_scope` 白名单：mlp / attention，`library/runtime/int8_linear.py:18-49`）+ 组内 int8（`group_size` 默认 256）。**没有 per-layer / per-group 位宽选择面**，HIGGS/RCO 的优化变量在本仓不存在。
2. **compute 精度冲突**：`base_compute` 与 `block_swap_transfer_dtype=int8` **硬互斥**（checks.py:101-113），说明现有量化栈的设计前提是"训练期 int8 GEMM + bf16 其余"，不是"权重 2.5bpw + 查表 kernel"。
3. **NF4 是加载期 4-bit**：`library/quantization/nf4.py` / `mixed_nf4.py` 只在 Krea-2 权重加载路径生效（cli_args.py:1114），且只允许 BF16 compute；它与 GSQ/RCO 的"训练后重分配位宽"是不同层次的事，不可互相替代。
4. **扩散侧没有校准集语义**：GSQ/HIGGS 的校准依赖"文本 token 上的 PPL/KL"，DiT 的对应物是 latent 重建误差或 CMMD，两者的量纲与可加性都未被论证 → 迁移必须重新定义目标函数，不能照抄 α_l。
5. **4-bit 权重路径存在，但不是 W4A16 训练基线（重要纠正）**：`--base_compute` 枚举里**没有 `w4a16`**（cli_args.py:1103-1106），但仓库**确实存在一个 tracked 的 opt-in 4-bit 实验实现**：
   - 代码：`library/runtime/nm_w4a16/`（`apply.py` / `linear.py` / `mask.py` / `packing.py` / `cost.py` / `config.py`）+ `bench/nm_w4a16/probe.py`；
   - 入口：`python tasks.py exp-nm-w4a16`（`tasks.py:28` 导入、`tasks.py:323-324` 注册），实现在 `scripts/experimental_tasks/nm_w4a16.py`；
   - 语义：**2:4 结构化稀疏 + uniform INT4 权重 + BF16 激活**，首期对手是 dense BF16 而不是 dense W4A16（`docs/experimental/nm-w4a16.md:1-12`）；
   - 边界：**未接入训练 bootstrap / WebUI / 配置面**，**无 checkpoint / compile / merge 集成**，`docs/experimental/nm-w4a16.md:3` 自述「静态实现，未进行真机验收」，`tests/test_nm_w4a16_static.py` 只有静态用例 → **无任何质量或性能证据**。
   → 因此准确的表述是「`--base_compute` 枚举无 w4a16；4-bit 只有这条未接线的静态 opt-in 路径」，而不是「仓库没有 W4A16 实现」；它仍**不能**充当 Q0 的可跑对照基线。

---

## 3. 因子分解、归因框架与 CNS 归因矩阵

### 3.1 因子分解（这是判断 Q4/Q5 的唯一合法框架）

任何"画质变好/变差"的主张，在当前系统里必须分解为三个可分离因子：

```
ΔQ = ΔQ(权重量化) + ΔQ(CNS) + ΔQ(交互项) + ΔQ(其余混杂)
```

- **因子 A — 权重量化**：发生在**训练期 / 加载期**。本仓今日只有两条可跑路：训练 `--base_compute w8a16_convrot|w8a8_convrot`（`convrot_scope` 默认 `mlp`，apply.py:355；bootstrap.py:316）、Krea-2 的 `nf4` 加载。（另有 opt-in 静态实验 `library/runtime/nm_w4a16/` + `exp-nm-w4a16`，未接入训练 bootstrap，**不产生可训练/可评测权重**，见 §2.2 第 5 条。）
- **因子 B — CNS（Colored Noise Sampling）**：发生在**推理采样期**，唯一 seam 是 `ERSDESampler._sample_noise`（library/inference/sampling.py:94-100），注入点 sampling.py:167-178，重着色在 `library/inference/corrections/cns.py:153-161`（`scale(f)=sqrt(clip(1-γ,0,1))` → fft2 × scale → ifft2().real → 逐 channel std 归一）。
- **交互项 A×B**：机制上**不能假定为零**。CNS 的 γ 是由**具体前向轨迹**测出的（bench/cns/gamma_probe.py:107-125），而权重量化会改变轨迹 → γ 有效性在量化权重下**未测**（不可断言正交）。
- **混杂项**：LoRA on/off（已知 γ 近似 LoRA-transparent，仅差 ~0.001）、cfg、aspect、step 数、flow_shift、VAE 解码路径。

### 3.2 CNS 归因矩阵

| 因子 | 实现位置 | 生效期 | 今日可达组合 | 画质/稳定性证据强度 | 归因可否判定 |
| --- | --- | --- | --- | --- | --- |
| A0 权重 bf16（默认） | cli_args.py:1103-1106 | 训练/推理全程 | 任意 | 基线 | — |
| A1 convrot int8（W8A16/W8A8） | library/runtime/convrot/* | 训练期（int8 训练 live） | × er_sde 白噪声（Anima 训练预览）**可达**；× CNS **不可达**（差三处接线） | 显存/速度为**实测级**（90HX 原始 JSON 可复算）；**画质=CMMD 未测** | 显存/速度：可；画质：不可 |
| A2 Krea-2 NF4 | library/models/krea2_raw/weights.py、library/quantization/nf4.py | 加载期（仅 Krea-2） | 推理侧**永久不可达** × CNS（krea2_raw/inference_runner.py:102-109 显式拒绝 `--cns`；Qwen 同 qwen_image_2_1/inference_runner.py:50） | 有各自的性能/显存记录；**CNS 交叉=0** | 交叉项：不可 |
| A3 nm_w4a16（2:4 INT4 + BF16 激活，opt-in 实验） | library/runtime/nm_w4a16/、bench/nm_w4a16/probe.py、CLI `exp-nm-w4a16`（tasks.py:28,323-324） | 仅静态 plan / GPU probe，**未接入训练 bootstrap** | **不可达**（无训练/推理接线，未接 WebUI/配置面） | **0**：无 checkpoint/compile/merge 集成，`docs/experimental/nm-w4a16.md:3` 自述未真机验收 | 不可判定 |
| B0 er_sde 白噪声 | sampling.py（默认噪声） | 推理采样期 | 全组合 | 基线（**唯一合法 CNS 基线**） | — |
| B1 er_sde + CNS | corrections/cns.py + sampling.py:94-100 | 推理采样期 | 仅 Anima bf16 推理（inference.py **零量化开关**） | **= 0**：`bench/cns/results` 不存在（且 .gitignore:36 忽略）、plan.md:108-111 仍是 CMMD+读网格 TODO、无 CNSRecolorer 数学用例 | 不可判定 |
| 交互 A×B | 无代码承载 | — | **不可达** | **无定义域** | 不可判定 |
| GSQ / RCO 项 | 无 | — | 无 | 零实现 | 恒不存在（空真） |

**矩阵读法（三条硬结论）**：

1. **单 run 内"量化 × CNS"不可达** —— 训练侧量化开关只有 `--base_compute`（cli_args.py:1106+）与 Krea-2 的 `nf4`（另有未接线的静态实验入口 `exp-nm-w4a16`，不产生可训练权重）；`inference.py` 对 `convrot|int8|nf4|w4a16` 全 0 命中，**零量化开关**；而 Krea-2/Qwen 推理**显式拒绝 `--cns`**。所以任何"量化让 CNS 更稳/更不稳"的说法，今日都**没有可执行的对照**。
2. **训练预览也不带 CNS** —— `library/anima/training.py:893` 归一化 sampler，`:901-906` 在 `sampler=="er_sde"` 时构造 `ERSDESampler(sigmas, seed, device)`，**未传 `cns=`**；`--cns/--cns_strength` 只存在于 `library/inference/args.py:508,516`，`library/training/cli_args.py` 对 cns 零命中。故今日可达 = 「convrot int8 × er_sde × **白噪声**」。
3. **CNS 画质证据强度 = 0** —— 见 §3.4 引用禁用清单。

### 3.3 2×2 消融设计（预注册，未执行）

| | B0：er_sde 白噪声 | B1：er_sde + CNS（同一 γ 文件） |
| --- | --- | --- |
| **A0：bf16（默认）** | 基线格 | 主效应 B 的来源 |
| **A1：`--base_compute w8a16_convrot --convrot_scope mlp`** | 主效应 A 的来源 | 交互项格 |

- **Estimand（必须在跑之前写死）**：本设计估计的是**整链效应**——「训练期量化 → 训练轨迹 → 预览采样」这条链的端到端差异，**不是**"量化对同一组权重"的纯推理效应。原因（代码级）：A1 的量化只在训练期 `apply` 生效，量化后的权重随后会参与训练，所以 **A0 与 A1 是两次独立训练**，权重本身不同；`inference.py` 零量化开关 → 今日**不存在**"仅预览期量化同一 LoRA / 同一 checkpoint"的对照臂。
- **因此交互项的解释边界**：A×B 除"量化权重 × 噪声重着色"外，还含"训练轨迹差异 × CNS"的耦合成分，**不能**解释为单纯的权重-噪声交互；报告时必须同时给出 A0/A1 各自的训练损失曲线与步数，否则交互项不可归因。
- A1 可选 `--convrot_scope all` 另跑一组作为敏感性臂。
- 主效应 A = M(A1B0) − M(A0B0)；主效应 B = M(A0B1) − M(A0B0)；交互 A×B = M(A1B1) − M(A1B0) − M(A0B1) + M(A0B0)。
- **基线纪律（不可妥协）**：
  - 基线**必须是 er_sde 白噪声**。`euler` 无噪声（sampling.py:37-39，确定性 ODE，CNS 为 no-op）；`lcm` 亦 no-op（LCMSampler 无 cns 形参，sampling.py:205-233）。用 euler/lcm 当基线会把 CNS 效应测成"零"，是伪结论。
  - **A 臂内部（同一 A 的 B0 vs B1）** 固定：同 model、同 LoRA/checkpoint、同 seed 集（≥4）、同 prompt 集（≥8）、同分辨率、同步数、同 cfg、同 flow_shift、同 VAE 解码、同评测脚本 —— 这一层是干净的单因子对比。
  - **跨 A 臂（A0 vs A1）不做"同权重"约束**（做不到）：只要求同 model 家族、同数据、同超参、同训练步数、同 seed，并按上一条同时报告训练轨迹指标，承认两次独立训练的训练噪声进入主效应 A。
  - 每次必须**记录 cfg / aspect / step 数 / flow_shift / sampler / γ 来源**。原因：shipped `aspects=[[0,0]]` 使"最近 aspect 选择"事实空转；γ 仅在 cfg=4.0 / 28 步 / flow_shift=3 / bf16 下标定（calibrate.py:177-178,196-199 硬编码 bf16）。
- **指标**：CMMD（`library/training/cmmd.py`）+ 高频纹理能量 + 盲评。**禁止用任何 LLM 指标**（zero-shot accuracy / WikiText-2 PPL / FW-C4 PPL）替代生图质量。
- **判读（预注册，避免事后叙事）**：Q4 成立需 `|B| > |A|` 且 A、B 同号为正；Q5 前半成立需 `Var_seed(A0B1) > Var_seed(A0B0)`，后半成立需 `Var_seed(A1B1) < Var_seed(A0B1)` 且交互项同向。
- **位置约束**：只能在 **Anima** 上做。Krea-2 训练预览是 **euler-only 硬拒**（`library/models/krea2_raw/training_preview.py:96-100`：`if sampler != "euler": raise ValueError(...)`），CNS 在其上无面。
- **采样器前提（极易踩空）**：训练预览的 `--sample_sampler` **默认就是 `euler`**（`library/training/cli_args.py:860-861` `default="euler"`；`library/anima/training.py:79-90::normalize_training_sample_sampler` 亦回退 euler），而 euler 下 CNS 是 no-op → **必须显式 `--sample_sampler er_sde`** 才可能到达 `ERSDESampler` 分支（`library/anima/training.py:901-906`）。缺这一步会把 B1 格测成与 B0 完全相同。
- **前置判定（必须先做）**：γ 是 bf16 轨迹下标定的；量化权重会改轨迹 → **先判"是否需要重标定 γ"，再比画质**。重标定通路当前不存在（calibrate.py 硬编码 bf16，`--extra` 只透传 inference flags）。

### 3.4 引用禁用 / 口径清单（后续任何文档必须遵守）

1. **禁用** `docs/methods/cns.md:39` 的 "A/B-validated as well-working"：它与 `bench/cns/plan.md:108-111` 的 A/B TODO 及缺失的 `bench/cns/results/` 直接矛盾。
2. **禁用**任何"恰好 −10% 显存"表述：没有任何配置对应。
3. **必须带分母**：−16.216%（分母 BF16 5.61140 GiB）与 −24.813%（分母**修复前 W8A16 swap0** 6.25306 GiB）都对，但**不可互换**。
4. **必须标不可复算**：3080 期 `output/tests/convrot_*.json` 0 命中 → 一律写「文档记载为已测，原始 JSON 缺失，不可本地复算」。
5. **禁止并表**三类 LLM 指标：GSQ 只有 zero-shot accuracy（全文 perplexity 命中 = 0）、HIGGS 用 WikiText-2 PPL、RCO 用 FW/C4 PPL 且候选是共享 GPTQ 库。
6. **RCO 候选库是共享 GPTQ**（"Same GPTQ-quantized weights per layer/bitwidth"），RCO **不重新量化**；"GSQ+RCO 联合"唯一定量锚点是 GSQ Table 1 的 NU 行（2.37 / 2.62 avg bits）。
7. **4bpw / 2.5bpw** 一律写「来源不可追溯」；**bpw 是位宽口径，不是误差**。
8. AdaLN 被量化 = **实测级反证**（正面结论）；FFN(=mlp) 被量化 = 成立（但术语要改成 mlp）。

---

## 4. Q0–Q5 判定表（结论 + 依据 + 缺口）

### Q0『GSQ-RCO 落地后比 w4a16 强很多』

- [论文·陈述 A｜GSQ 的实验点覆盖] GSQ 正文明确「we focus on **2-bit and 3-bit weight-only quantization with a group size of 128**」，其自身实验点是 ternary(1.71) / 2.13 / 2.37 / 2.62 / 3.13（平均 bits）→ **只覆盖 2/3-bit、group size=128，没有任何 4-bit 数据点**（`rg -i "4-bit"` 仅出现在背景/引用句）。因此"GSQ 在 4-bit（W4A16）档更强"在原文**零数据**。
- [论文·陈述 B｜HIGGS 在 4.25bpw 的边际优势] HIGGS Table 11（Qwen2.5-7B，WikiText-2 PPL）：4.25 bpw 上 `NF 6.340` vs `HIGGS(p=3) 6.278` / `HIGGS(dyn) 6.255` → 相对 NF4 只剩 **0.06–0.09 PPL**；同一表 3.25 bpw 上优势才明显（`NF 7.588` vs `HIGGS(dyn) 6.566`）。→ **4-bit 附近收益递减**。
- [论文·陈述 B 的旁证｜RCO 也在大位宽收敛] RCO Table 2（Qwen3-8B，252 个 GPTQ 线性层，FW/C4 PPL，FP16 FW=10.96）：3.5 bits `RCO 11.41–11.46` vs `dynamic HIGGS 11.65`；4.0 bits RCO 与 HIGGS/EvoPress 基本重合 → RCO 的收益同理集中在低 bpw，4-bit 收敛。
- [论文·反方向证据（须同时给出，不得只挑支持方）] GSQ Table 1（Llama-3.1-8B，zero-shot 平均分）：`GSQ 3.13 bpw = 72.32` vs `GPTQ 3.25 = 59.18`，即在 3-bit 档 GSQ 对 GPTQ 有极大优势。这条**部分支持**用户"低 bit 更强"的直觉，但档位是 3-bit、对手是 GPTQ 而非 w4a16。
- **[口径红线]** 上述各条**不可并表**：GSQ 用 **zero-shot accuracy**（全文 perplexity 命中 = 0），HIGGS 用 **WikiText-2 PPL**，RCO 用 **FW/C4 PPL**（且候选是共享 GPTQ 库）；三者模型域、指标、可比性都不同，只能分列陈述。
- [仓库] `--base_compute` 的合法值是 `bf16 / w8a16_convrot / w8a8_convrot / nf4`（cli_args.py:1103-1106；checks.py:68-89）→ **枚举里没有 `w4a16`**；4-bit 权重只有一条**未接线的静态 opt-in 实验路径**：`library/runtime/nm_w4a16/` + `bench/nm_w4a16/probe.py` + CLI `exp-nm-w4a16`（tasks.py:28,323-324；语义是 2:4 结构化稀疏 + uniform INT4 权重 + BF16 激活），它**未接入训练 bootstrap / WebUI / 配置面，无 checkpoint / compile / merge 集成，无质量证据**（`docs/experimental/nm-w4a16.md:3` 自述「静态实现，未进行真机验收」）→ 仍**不能充当可跑的 W4A16 对照基线**；GSQ/RCO 则是零实现。
- **判定：不成立**。缺口：(i) 一个同轴、同 bit、真机可跑的 w4a16 基线（现有 `nm_w4a16` 只是静态实验，不能当真机对照）；(ii) GSQ/RCO 的 DiT 可移植实现 + 校准管线；(iii) 生图质量度量协议；(iv) 可比的 bpw 定义。

### Q1『相当于重新分配了各层权重』

- 两种读法必须分开：
  - **(1a) 重新分配各层/各组"位宽预算"** → **成立**（[论文] HIGGS §5 用 ILP/CP-SAT 做 per-layer 非均匀分配，α_l 由 Gaussian noise insertion 标定；RCO 变量是 per-group "选哪个 bit 档位"的 logits，约束 `C(𝛂)=B`，原文（`rco_full.txt:179`）为 "RCO provides first-order optimization of the actual loss **with the expected budget enforced exactly at every iterate**"；GSQ NU 2.37 / 2.62 即非均匀分配产物）。
  - **(1b) 权重被重排/旋转** → **不成立**（两篇 `hadamard` 命中仅出现在**参考文献标题**；RCO 复用同一套 GPTQ 候选库、不重新量化；RHT 是 HIGGS 组件，且旋转在后量化前，不是"重分配"）。
- [仓库] 本仓只有"哪些模块被量化"的白名单 + 组内 int8，**无任何 mixed-bit / per-layer 位宽分配通路** → (1a) 在本仓尚无实现面。
- **判定：部分成立**（成立当且仅当读作"位宽预算分配"）。

### Q2『显存 −10%、速度不变』

- [仓库实测·90HX，可复算] W8A16 swap0 峰值 **4.70146 GiB** vs BF16 **5.61140 GiB** → **−16.216%**（分母 = BF16；原始量 `output/convrot-lifecycle-fix-20261003/production-audit.json`）。
- [口径警告] **−24.813%** 的分母是**修复前 W8A16 swap0 = 6.25306 GiB**（`output/convrot-90hx-20261002-210307/s4-production-audit.json`）。两个百分比都正确但**分母不同、不可互换**。
- [仓库实测] **无任何配置给出恰好 −10%**。
- [仓库实测·速度（只列可复算点）] 90HX 上**仅三个点**有原始量支撑：① `swap=0` W8A16 比 BF16 慢 **+10.59%**（×1.10593；原始 `wall_sec_per_step` 中位数 1.738801 vs 1.572258，见 `output/convrot-90hx-20261002-210307/s4-minswap-performance-r{0,1,2}-{bf16,w8a16}/result.json`）；② `swap=12` W8A16 仅 **+1.16%**（×1.01155；`s1-performance-summary.json` 三值中位数 2.753264 vs 2.721824）；③ `swap=12` W8A8 比 BF16 **+59.4%**（×1.59364；4.337610 vs 2.721824）。机制解释（代码级）：`base_compute` 与 `block_swap_transfer_dtype=int8` 硬互斥（checks.py:101-113）→ swap 传 int8 载荷，省下 H2D 字节，故 swap≥12 时差距收窄甚至反超。**本记录不给出跨 swap / 跨硬件的加速比区间**：早期版本引用过两个比值区间，但其每个端点的配置 / 文件 / 行号来源未逐条溯源，已按复核意见整段删除；需要区间结论时应回到 `convrot_longrun_bf16_w8a8_w8a16_20260727.md` 与 90HX 原始目录逐点复核，**不得跨硬件或跨 swap 外推**。
- [文档记载·不可复算] 3080 期数字（如 3.43→3.17 GB、4.95→4.11 GB）一律附：「文档记载为已测，原始 JSON 缺失（`output/tests/convrot_*.json` 0 命中），不可本地复算」。
- **判定：部分成立**（显存方向成立、数值不是 −10%；速度"不变"仅单点近似）。缺口：要字面 −10% 需指定分母 + 配置组合；要"速度不变"需解释 swap=0 的 dequant 税，或改 swap≥12 并明示这是 block-swap 收益而非 convrot 收益。

### Q3『修好交叉投影层问题；之前 convrot 把 AdaLN 和 FFN 都量化了；误差 4bpw，GSQ-RCO 压到 2.5bpw』

| 子命题 | 判定 | 依据 |
| --- | --- | --- |
| Q3a AdaLN 被量化 | **不成立（实测级反证）** | 用 90HX 捕获的 **314 条 runtime Linear 形状**跑真实分类器：`scope=mlp`→**56**、`attention`→**140**、`all`→**196** 全部对上；block 内未命中的模块真名是 `adaln_up_self_attn` / `adaln_up_cross_attn` / `adaln_up_mlp` / `adaln_fused_down.1`；block 外排除 `final_layer.adaln_modulation.1/2`、`final_layer.linear`、`x_embedder.proj.1`、`t_embedder.1.linear_1/2`。代码级佐证：结构性五重过滤 —— 路径正则 `^(?:net\.)?blocks\.\d+\.<name>$`（int8_linear.py:16）+ 白名单不含 `adaln_*`/`modulation*`（:18-49）+ bias≠None 跳过（checks.py:48-49）+ requires_grad 跳过（checks.py:50-51）+ 仅 LoRA 已包裹模块（apply.py:445-451） |
| Q3b FFN 被量化 | **成立（术语须改）** | `convrot_scope` 默认 `mlp`（apply.py:355；bootstrap.py:316）；MLP 白名单为 `mlp.layer1` / `mlp.layer2` / `img_mlp.gate_layer` / `img_mlp.proj` + MOSS-Image DiT 的 `proj.ffn_up_gate` / `proj.ffn_down`（`library/runtime/int8_linear.py:18-26`）。仓库术语是 **mlp**，不是 FFN |
| Q3c "修好了交叉投影层问题" | **不可判定（来源不可追溯）** | 本仓无"交叉投影层"术语；最接近的是 attention 白名单里的 `cross_attn_q/kv/out`（int8_linear.py:28-49）。无 issue、无 bug 复现、无 before/after 数值、无对应 commit |
| Q3d 误差 4bpw / 压到 2.5bpw | **不成立（量纲错误 + 来源不可追溯）** | bpw = bits per weight 是**位宽**不是误差；"2.5bpw 比 4bpw 误差更小"在原文表述下自相矛盾。本仓无 mixed-bit 通路、无 2.5bpw 配置 |
| Q3e GSQ-RCO 修复了它 | **不成立** | 无实现、无证据（见 Q0） |

- **整体判定：不成立**（分项：FFN 成立 / AdaLN 被反证 / bpw 量纲错误 / "交叉投影层问题"不可追溯 / GSQ-RCO 无实现）。

### Q4『质量改进主要是 CNS 贡献的，不是 GSQ+RCO』

- [仓库·不可达] CNS 只在 Anima bf16 推理生效（`inference.py` 对 `convrot|int8|nf4|w4a16` 全 0 命中，**零量化开关**）；量化只在训练 `--base_compute` 或 Krea-2 nf4（`nm_w4a16` 为未接线静态实验，不产生可评测权重）；Krea-2 / Qwen 推理显式拒绝 `--cns` → **单 run 不可达**。
- [仓库·预览面] 训练预览不走 `generation.py`：`library/anima/training.py:893` / `:901-906` 构造 `ERSDESampler` **未传 `cns=`**；`--cns` 只在 `library/inference/args.py:508,516`。
- [证据强度] CNS 画质证据 = 0（见 §3.4 第 1 条）。
- [空真] GSQ+RCO 零实现 → "不是 GSQ-RCO 贡献"在当前系统里是**空真**（该项从未被启用）；真正可比的是 convrot int8 / nf4。
- **判定：无实测路径（不可判定）**。缺口：§3.3 的 2×2（差三处接线）。

### Q5『CNS 不够稳定；GSQ+RCO 让 CNS 对纹理的改进更稳定』

- 前半「CNS 不够稳定」= **无实测证据（不可判定）**：`bench/cns/results` 不存在、plan.md:108-111 仍是 TODO、cns.md:39 不可引用、tests 只有 Krea-2 拒绝契约（tests/test_krea2_inference_capabilities.py:20-21,51-52）、无 CNSRecolorer 数学用例。
- 后半「GSQ-RCO 让 CNS 更稳」= **不成立**：前提未实现 + 无交互通路 + "稳定"未定义（缺方差定义、样本量、失败判据）。
- [仓库·静态风险（未测，仅假设 H）] late-σ 频带 scale 动态范围 **11.6× / 39.7×**；shipped `aspects=[[0,0]]` 使"最近 aspect 选择"事实空转；γ 仅在 cfg=4.0 / 28 步 / flow_shift=3 / bf16 下标定；且 γ 对轨迹场敏感（cfg1→cfg4+aspect spread：+0.410 → +0.521，LoRA on/off 仅差 0.001）。
  - **11.6× / 39.7× 的复现配方（可自查）**：取 shipped `networks/calibration/cns_gamma.npz`（`gamma` shape `(1,28,32)` = 28 步 × 32 radial bins），按 `library/inference/corrections/cns.py:153-161` 的 `scale = sqrt(clip(1−γ, 0, 1))` 计算，再取**每一步内** 32 个 bin 的 `max(scale)/min(scale)`：
    `.venv/bin/python -c "import numpy as np; g=np.load('networks/calibration/cns_gamma.npz')['gamma'][0]; s=np.sqrt(np.clip(1-g,0,1)); print((s.max(1)/s.min(1))[-2:])"` → `[11.554576 39.70726 ]`（≈ 11.6× / 39.7×，即倒数第二步、末步）。这是**末两步内**的频带动态范围，不是全序列最大值，口径必须随数字一起写。
- **判定：前半无实测证据；后半不成立。**

---

## 5. 与现有量化 / CNS 工作的关系与冲突点

本节**只引用不重复**，细节以被引文档为准。

### 5.1 量化侧既有记录

| 既有文档 | 与本记录的关系 |
| --- | --- |
| [convrot_longrun_bf16_w8a8_w8a16_20260727.md](convrot_longrun_bf16_w8a8_w8a16_20260727.md) | W8A8 / W8A16 长跑对照 → 本记录 Q2 的显存/速度口径来源之一 |
| [anima_int8_base_linear_audit.md](anima_int8_base_linear_audit.md) | base Linear 的 int8 覆盖审计 → 与 §2.2 的"量化粒度只有模块白名单"一致 |
| [diffusion_mixed_quant_post_training_survey_20261002.md](diffusion_mixed_quant_post_training_survey_20261002.md) | 扩散侧 mixed-quant 后训练调研 → 本记录 §2 可迁移性的方法论参照；注意其结论面向 DiT，不可与 LLM 论文数字并表 |
| [int8_training_local_vs_musubi_20261002.md](int8_training_local_vs_musubi_20261002.md) | 训练期 int8 的本地实现 vs musubi 对照 |
| [int8_encoding_floor_50hx_20261019.md](int8_encoding_floor_50hx_20261019.md) | int8 编码下限（50HX）→ 本记录把 int8 视为"现有可实现下限"的依据 |
| [krea2_nf4_ablation_findings.md](krea2_nf4_ablation_findings.md) / [krea2_nf4_correction_pg199.md](krea2_nf4_correction_pg199.md) / [krea2_nf4_h2d_bottleneck_findings.md](krea2_nf4_h2d_bottleneck_findings.md) 等 | Krea-2 NF4 加载期 4-bit 的唯一既有通道 → Q0 的"最接近 w4a16 的对照物"，但 **不叫 W4A16、且只允许 BF16 compute** |
| [qwen_image_2_1_int8_swap_90hx_20260927.md](qwen_image_2_1_int8_swap_90hx_20260927.md) | 另一族的 int8 + swap 记录 → 说明 `base_compute` × swap 的互斥约束是跨族契约 |
| [qwen_nf4_compile_90hx_20261005.md](qwen_nf4_compile_90hx_20261005.md) | Qwen-Image 2.1 NF4 + compile → 与 Krea-2 同理，推理侧拒绝 `--cns`（qwen_image_2_1/inference_runner.py:50） |
| [docs/experimental/nm-w4a16.md](../experimental/nm-w4a16.md) | 本仓**唯一** 4-bit 权重代码路径：2:4 结构化稀疏 + uniform INT4 + BF16 激活，opt-in 实验入口 `python tasks.py exp-nm-w4a16`（tasks.py:28,323-324）。**只引用不重复**：其自述为「静态实现，未进行真机验收」（:3），未接入 bootstrap/WebUI/配置面、无 checkpoint/compile/merge 集成 → 本记录 §2.2 第 5 条与 §4 Q0 的缺口 (i) 即以此为准 |

### 5.2 CNS 侧既有记录与**组合张力**

来源：[docs/methods/cns.md](../methods/cns.md)、[bench/cns/plan.md](../../bench/cns/plan.md)。

**读法**：下表「张力原文」列**保留英文原文 quote**（我方中文转述只在括号里以「译：」标出，属译文而非原文，引用时必须以英文原文为准）。

| 组合 | 张力原文（quote）+ 译 | 本记录的判定影响 |
| --- | --- | --- |
| **Spectrum** | "structurally clean (different seam) but a **real tension**: Spectrum forecasts a *smooth* feature trajectory, and CNS concentrates energy in high-freq *late* — exactly where the Chebyshev forecaster is weakest. … the cached-step error must be A/B-checked, not assumed"（cns.md:92）<br>译：结构上干净（不同 seam），但存在真实张力 —— Spectrum 预报平滑特征轨迹，而重着色噪声把能量集中在 high-freq 的**晚期**，正是 Chebyshev 预报最弱处……cached-step 误差必须做 A/B，不得假定 | 任何 CNS A/B 必须**同时**验证 Spectrum 的 cached-step 误差不增长，否则 B1 格不可解释 |
| **DCW** | "Watch, don't block. CNS already injects *less* energy into the resolved LL band; DCW then bias-corrects LL post-step. **Possible partial double-count** — no conflict observed, but worth an explicit ablation if stacked."（cns.md:95）<br>译：观察，不阻塞。CNS 已给 resolved LL 带注入更少能量，DCW 随后在 step 后对 LL 做 bias 修正 → 可能部分重复计数；未观察到冲突，但若叠加值得做显式消融 | 若基线启用 DCW，CNS 效应可能被部分抵消 → 2×2 必须固定 DCW 开关状态并记录 |
| **SMC-CFG** | "none — drift seam, CNS is the noise term. Genuinely independent."（plan.md:52）<br>译：无冲突 —— drift seam，CNS 是噪声项，二者确实独立 | 不构成阻塞，但同样要固定 |
| **mod-guidance** | 正交 seam，但会改变 v 轨迹 → γ 应在 mod-guidance ON 下重标定（2nd-order）（**本条为转述，非逐字 quote**） | 与 §3.3 的"γ 重标定前置判定"同源 |

**代码侧事实**：上述张力**没有任何互斥检查**兜底 —— 组合安全完全依赖人工纪律，不是运行时保证。

**跨因子冲突（本记录新增识别）**：CNS 与量化的冲突不在"能量守恒"层（CNS 只换噪声 draw，外层 `alpha_t · s_noise · noise_coeff` 不动），而在 **γ 的有效性**层 —— γ 从 bf16 轨迹测出，量化改轨迹 → γ 可能失配。这条今日**无任何代码或实验覆盖**。

---

## 6. 若要实机验证的最小实测计划（**未执行**）

> 以下全部**未执行**，仅作为后续 opt-in 方案。硬约束：**显式 opt-in、不改训练默认、不动用户数据、不占用户正在训练的 GPU**（默认用闲置卡或用户授权的窗口）、不写进任何默认配置。

**Step 0（前置判定，必须先做）—— γ 是否需要重标定**
纯推理、低成本：在 bf16 与 `w8a16_convrot` 两种权重下各跑 `bench/cns/gamma_probe.py`，比较 γ 曲线的 spread / agg σ50 / linear MAE（同 prompt、同 cfg、同 aspect、同 step 数）。
判据：若两条 γ 曲线的 spread 差异大于 Phase-0 内 LoRA on/off 的 ~0.001 量级，则"γ 可跨精度复用"不成立 → 必须先补一条量化标定通路（当前 `bench/cns/calibrate.py` 硬编码 bf16），否则后续 A/B 不同源。

**Step 1 —— 三处最小接线（代码改动，仅在做实测时提交，且默认关闭）**
1. `library/anima/training.py` 预览采样辅助（:855-910 区间）新增 `cns` 形参，构造 `ERSDESampler(..., cns=cns)`（`sampling.py:42-100` 已有 `cns` 形参与 `_sample_noise` 门控，无需改采样器本体）。
2. 由 `networks/calibration/cns_gamma.npz` 构造 `CNSRecolorer.from_path(spec, strength=...)`，复用 `library/inference/generation.py:38-56::_build_cns_recolorer` 的模式。
3. 训练预览 CLI/config 面新增 `cns` / `cns_strength` 并透传（现仅 `library/inference/args.py:508,516`）；同步 compat 拒绝矩阵（非 Anima family fail closed）与 WebUI 表单/校验。

**Step 2 —— 跑 §3.3 的 2×2**（A0/A1 × B0/B1，每格 ≥8 prompt × ≥4 seed），外加 A1+重标定 γ 的敏感性臂。产物落 `bench/cns/results/`（注意该目录被 .gitignore:36 忽略，需显式归档证据）。

**Step 3 —— 判读与归档**：按 §3.3 预注册判据出结论；无论正负都写进 `docs/findings/`，并同时归档原始 CMMD / 高频能量 / 盲评表格。

**不建议做的**：把 GSQ/RCO 移植排进本轮验证 —— 它需要校准管线 + 多档位候选权重库，与本仓"训练期 int8 GEMM"的架构正交，属独立研究项，不是 Q0 的快捷路径。

---

## 7. 未解决疑点与需用户拍板的选项

| # | 疑点 | 需要用户决定 |
| --- | --- | --- |
| 1 | "GSQ-RCO" 到底指什么 | 是否指 GSQ(2604.18556) + RCO(2605.00649) 两篇的组合？若是，本文按"待核别名"处理即可；若指向 2411.17525，则**名称错配需更正** |
| 2 | Q2 的 −10% 来源 | 是否有特定配置（swap / scope / 分辨率）对应"恰好 −10%"？当前可复算只有 −16.216% / −24.813%；若原始出处不可考，建议改用"−16.2%（分母 BF16）" |
| 3 | Q3 的"交叉投影层问题" | 是否有对应 issue / commit / 复现脚本？无则该项只能记为"不可追溯主张" |
| 4 | Q3 的 4bpw / 2.5bpw | 这两个数字的出处（哪次实验、哪个配置、报告在哪）？**bpw 是位宽口径，不是误差**，需重新表述 |
| 5 | Q4/Q5 是否值得投入实测 | 是否授权做 §6 的三处接线 + 2×2（需少量代码改动，可保持默认关闭）？若不授权，Q4/Q5 只能维持"不可判定" |
| 6 | γ 量化后是否重标定 | 若做实测，先做 Step 0；若 γ 失配，是否接受新增"量化感知标定通路"（新代码面） |
| 7 | GSQ/RCO 是否立项移植 | 若要，建议作为独立研究项立项（校准管线 + 候选权重库 + DiT 目标函数重定义），**不并入本轮量化栈工作** |

---

## 8. 本记录的自限声明

- 本文**未运行任何 GPU 任务、未修改任何源码或配置、未改训练默认、未开 git 分支**。
- 文中所有数值都是既有证据的**转述**，并保留了"可复算 / 半复算 / 不可复算"标签；**本文不宣称任何数值经本轮验证**。
- 文中判定全部基于静态分析（源码 + 既有文档 + 论文原文）；凡标 `H` 的均为未测假设。
- 相关依赖证据：训练器量化栈静态审计、CNS 实现与证据审计、Q0–Q5 判定依据链、Q2 类数字独立复算、GSQ/RCO 原文定量核验（同批只读任务，均未改动仓库）。
