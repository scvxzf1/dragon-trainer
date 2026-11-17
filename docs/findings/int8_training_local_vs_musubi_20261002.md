# INT8 训练：本地 ConvRot 与 musubi 实现对比

状态：调研记录 / 阶段快照
查阅日期：2026-10-02。范围：当前源码、历史实验文档、固定提交的线上源码与讨论。本轮没有运行 GPU、启动训练或修改训练实现。

## 结论

用户记忆正确：历史 RTX 3080 实验记录中，W8A8 完整训练慢于 W8A16。本地早已有真实 INT8 前向、ConvRot、冻结基座 LoRA、scope/尺寸阈值混精、prequant 和换层集成。不能把线上 ConvRot 当成本地尚未实现的新路线。

musubi 的主要差异是 **Triton 逐行激活量化、融合 GEMM 后处理，以及默认较低精度的反向计算**。后者同时改变数值策略；本地曾否决类似降精尝试。因此，线上速度收益不能全部归因于更好的 kernel，也不能直接替代本地实现。

值得研究的是在保持本地数值契约的前提下，减少量化和 GEMM 后处理的中间张量与 kernel launch。暂不支持重新实现整套 ConvRot、默认切换 W8A8，或宣称 musubi 可让 Anima/3080 超过现有 W8A16。

证据分级：本轮直接核对了实时源码与固定线上提交；历史数值来自仓库实验报告。对应长训原始目录及本报告引用的 `output/tests/convrot_*.json` 在当前 checkout 未找到，不能称为本轮独立复算或重新实测。

## 1. 3080 历史结论及其适用范围

### 1710-step 长训

来源：[2026-07-27 长训审计](convrot_longrun_bf16_w8a8_w8a16_20260727.md)，尤其 44–82 行。

文档记录 RTX 3080 10GB/SM86、LoRA rank32、batch1、BF16、Flash、gradient checkpointing、compile/dynamic sequence、scope=all、swap26。三组均完成 6 epoch / 1710 steps；稳态时间排除了采样、保存的大间隔。

| 模式 | 稳态 s/step | 相对 BF16 时间 | 相对 W8A16 时间 |
|---|---:|---:|---:|
| BF16 | 1.2662 | 1.000× | — |
| W8A16 | 1.3708 | 1.083× | 1.000× |
| W8A8 | 1.4626 | 1.155× | **1.067×** |

这支持“W8A8 比 W8A16 慢约 6.7%”。不支持严格质量等价：没有固定训练全局 seed；57 张图片没有独立验证集；两个 sample prompt 实际重复。

该轮 ConvRot 约 4.1 GB 的高显存包含历史 residency 缺陷：量化 payload 没有随 DiT block 换出。2026-08-04 已改为 block-owned carrier；不能将旧峰值当作当前实现固有显存成本。

### Residency 修复后的独立短测

来源：[实验记录 §G.23](../experimental/convrot_int8_training.md)（按标题搜索 `G.23`；正文 833–868 行）。文档记录 RTX 3080、compile、GC、scope=all、rank32、每组 8 steps。

| swap | BF16 s/step | W8A16 s/step | W8A8 s/step | W8A8 比 W8A16 慢 |
|---:|---:|---:|---:|---:|
| 0 | 1.954 | 2.039 | 3.019 | 48.1% |
| 12 | 2.849 | 2.748 | 3.988 | 45.1% |
| 26 | 3.613 | 3.148 | 4.884 | 55.1% |

例如 swap12 的报告峰值：BF16 3.669、W8A16 2.776、W8A8 3.093，原表单位写 GB。这轮支持“修复驻留后仍有 W8A8 性能倒退”，同时表明 W8A16 在某些交换配置下可以省显存并改善步时。

但原始 JSON 缺失，不能独立核对 warmup、计时窗口及 bytes→GB/GiB 换算。此表与 1710-step 表是两轮实验，不能混合绝对时间得出版本提速/退化幅度。

## 2. 本地的几种“INT8”并不是同一条路径

| 路径 | INT8 用在哪里 | Linear 运算/训练语义 |
|---|---|---|
| 旧 `Int8FrozenLinear` | 未旋转基座权重存储 | 反量化后浮点 `F.linear`；无 INT8 激活 GEMM |
| `block_swap_transfer_dtype=int8` | CPU master 与传输 | 恢复到 GPU dense 浮点权重后计算 |
| `w8a16_convrot` | 旋转后的权重 | 在线旋转输入，反量化权重，浮点 GEMM；通常 BF16 |
| `w8a8_convrot` | 旋转权重及当前 Linear 输入 | 动态量化输入，真实 INT8 前向；默认 FP32 输入梯度 GEMM |
| 8-bit optimizer | 可训练参数的优化器状态 | 不决定冻结基座或激活的 GEMM 精度 |

代码入口：`library/runtime/int8_linear.py:188`、`block_swap_masters.py:249,406`、`convrot/fused.py:139,185,209,256`、`library/training/optimizers.py:82`。

W8A8 的 A8 指所选 Linear 的 GEMM 操作数。block 边界、attention、LoRA、残差和 checkpoint 保存的张量并未因此全部变成 INT8。两条 ConvRot frozen-base backward 都不保存完整输入激活；不能将 W8A8 理解为在 W8A16 基础上又把所有训练激活减半。

基座冻结仍需传播 dX：前面层的 LoRA 更新要通过后续冻结基座反传。以 `R` 为分组正交对称 Hadamard，`Ŵ_R = diag(s_w)Q_w`：

```text
W8A16 forward: y = (xR) Ŵ_Rᵀ
W8A8  forward: y ≈ dequant(Q(xR)) Ŵ_Rᵀ
STE backward:  dx ≈ (dy Ŵ_R) R
```

W8A8 backward 忽略激活 round/clamp/scale 的导数，采用 STE 型近似；没有训练 INT8 基座权重。两者前向不同，因此即使采用形式相同的 dX 公式，实际收到的 dy、adapter 梯度和训练轨迹也会不同。

## 3. 当前本地与线上 musubi 的逐项差别

线上固定版本：musubi-tuner `f8a1b03794a49239a3539015075f5123d6c07d66`，提交日期 2026-09-27。本轮联网确认缓存中的 kernels/utils/Krea2 loader 与固定版本逐字一致。

| 项目 | 本地 W8A8 默认 | musubi ConvRot CUDA + Triton |
|---|---|---|
| 权重 | FP32 旋转，逐输出通道 INT8 + FP32 scale | 同类离线旋转/量化策略 |
| Hadamard | Sylvester@256；已有 regular 选项 | regular@256；同组大小的 regular 构造与本地一致 |
| 在线旋转 | dense `torch.matmul` | dense `torch.matmul`，没有融合进量化/GEMM |
| 激活量化 | PyTorch absmax、FP32 除法取码 | 独立 Triton rowwise kernel；BF16 输入时 scale/商有 BF16 舍入 |
| 前向 GEMM | `torch._int_mm` → 完整 INT32 输出 → FP32 双 scale → BF16 | Triton INT8 dot/INT32 累加，kernel 内 FP32 scale/bias，直接输出目标 dtype |
| 默认 backward | FP32 `dy/scale/Wq` GEMM，再转 compute dtype 逆旋转 | 模式名 `bf16`；实际随 `grad_out.dtype` 反量化和 matmul，通常 BF16 |
| 可选低精反向 | TF32 opt-in；当前没有 BF16 STE 产品开关 | 可选梯度量化的 INT8 backward；新增误差与转置临时张量 |
| fallback | auto 失败可用已量化的 A/W 转 FP32 GEMM；仍保留 A8 数值 | 无 Triton 时不量化 A，旋转+反量化 `F.linear`，近似 W8A16 路径 |
| 量化范围 | LoRA org_forward，scope、尺寸筛选及大层单独 mode | Krea2 `blocks.`，排除 `mod.`、`norm`、`txtfusion` |

### 工程融合与数值策略必须分开

本地 `fused.py` 合并的是 `autograd.Function`，当前模块没有专用 Triton 实现。`torch.compile` 可优化部分操作，但仅凭 fused 命名不能声称旋转、量化、GEMM 已成为单个 GPU kernel。

线上也不是整个流程一核完成。真正可借鉴的是：

1. rowwise kernel 在一次启动内做 absmax、scale、取码，避免 PyTorch eager 下完整 FP32 激活中间张量的多次读写。
2. GEMM 的反量化/bias 后处理放进同一个 kernel，避免将完整 INT32 和 FP32 输出先后写入显存。

局部 FP32 运算可以在寄存器里完成；“保留 FP32 取码”并不必然要求把整个 FP32 activation tensor 写入显存。但新 kernel 的舍入、scale 下限、乘法顺序、编译器行为仍要单独对照。

musubi BF16 激活路径在取码时会把除数 scale 和商舍入为 BF16，但输出 scale 仍存 FP32，clamp 范围 `[-128,127]`；本地使用 FP32 除法、`[-127,127]`。两边 scale 下限规则也不同。原封不动复制线上 quant kernel，会改变本地量化码。

反向还存在不同的浮点运算顺序：本地计算 `(dy * scale) @ cast(Wq)`；musubi 默认计算 `dy @ (cast(Wq) * scale)`。精确实数中等价，有限精度下不等价。本地曾失败的 BF16 backward 尝试不能直接证明 musubi 当前公式必然失败，但足以说明这不是无需验收的等价加速。

源码依据：本地 `convrot/gemm.py:60,146`、`fused.py:185,256`、`apply.py:629`；线上 [量化取码](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/modules/convrot_int8_kernels.py#L215)、[融合后处理](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/modules/convrot_int8_kernels.py#L420)、[多步骤前向](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/modules/convrot_int8_kernels.py#L479)、[反向](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/modules/convrot_int8_utils.py#L349)。

### 本地已有的优化与边界

- apply 阶段缓存 W8A8 `[K,N]` 连续权重，避免每步转置；`gemm.py` 将 M 补到 32 倍数并裁回，处理 SM86 的 4200-token 兼容问题。
- 量化 weight/scale 挂在实际 DiT Linear 下的 runtime-only carrier，随 block swap；dense BF16 权重逐模块释放到 meta，避免始终保留两套基座。
- `apply_to/load_weights → ConvRot patch → compile`；adapter 仍在原始特征空间。checkpoint 重算会再次执行旋转、激活量化和 GEMM。
- scope/尺寸阈值已支持 W8A16 与 W8A8 混合；尚非按任意模块名称或敏感度自动分配的 precision map。
- W8A8 `auto` 会在部分不支持/失败条件下浮点 fallback；配置名称本身不能证明一次运行已使用真实 INT8 kernel。诊断应记录后端或显式要求 `int_mm`。
- 默认 W8A8 不主动启用 TF32，但也不强制覆盖其他代码设置的全局 matmul precision；新对照需记录该状态。

源码：`apply.py:231,371,631,637,675`、`gemm.py:135`、`block_swap_payload.py:15`、`library/training/bootstrap.py:622`。

## 4. 为什么 W8A8 倒退：已经知道什么

源码支持的候选成本包括：W8A8 专有的激活量化、INT32/FP32 输出中间张量、FP32 backward、对齐和 fallback；两条路径共有在线旋转，checkpoint 会重复前向。不能把共有的 RHT 单独当成 W8A8 相对 W8A16 变慢的全部原因。

尤其是默认 W8A16 backward 已使用 BF16 compute；W8A8 backward 则为 FP32。比较两者不是单纯比较同一 GEMM 的 A16/A8 两种输入格式。Ampere 支持 INT8 Tensor Core，也不保证具体形状的 INT8 kernel 比 BF16 库实现更快。

历史 [§G.5 单算子实验](../experimental/convrot_int8_training.md)（307–357 行）使用 M=4032、Anima MLP `[8192,2048]`/`[2048,8192]`：`_int_mm only` 约为 BF16 用时的 1.31×，带 post-scale 约 1.60×。所以当时即使免费消掉后处理，也不足以让**同一个 `_int_mm` 主体**超过 BF16。

这解释了当时否决立即开发 Triton 的依据，但不是所有 Triton INT8 kernel 的理论下界：更换 GEMM 实现后主体性能也可能变化。早期 profiler 的 ConvRot self-time 占比也不能直接换算为当前完整 step 的提速上限；代码、scope、swap 和测量阶段已经变化。

### 降精曾有收益，但部分数值检查失败

来源：[实验记录 §G.15](../experimental/convrot_int8_training.md)，582–601 行。

| 历史尝试 | compile 步时记录 | 三个 seed 的检查结果 |
|---|---:|---|
| half activation quant | 约 1.27 s | 1/3；seed2 grad_rel 约 1.4 |
| half quant + 保留 FP32 STE | 约 1.58 s | 1/3；仅恢复反向精度仍未解决 |
| BF16 STE backward | 约 1.27 s 档 | 1/3 |
| TF32 STE backward | 约 1.42 s | 2/3；seed0 grad_rel 9.9% |
| 默认 FP32 codes + FP32 STE | 约 1.58 s | 3/3 |

这里的 gate 为输出 relative L2 ≤3%、loss 相对差 ≤5%、adapter **总梯度范数的相对差** ≤5%。当前 `scripts/experiments/convrot_checkpoint_probe.py:78,313` 明确用两个梯度范数的标量差，不是梯度向量 relative L2；“3/3”不能扩大成每层梯度方向或长期生成质量已经等价。

W8A8 scale 降到 BF16 也有超过 5% gate 的记录，当前仍保留 FP32（`apply.py:638`）。这些证据说明保留 FP32 有历史原因；不证明所有更低精度策略都不可用，也不应直接丢掉既有门槛去追逐线上速度。

## 5. 线上性能和测试真正证明了什么

[musubi PR #1008](https://github.com/kohya-ss/musubi-tuner/pull/1008) 作者记录：Krea2、1024²、batch1、Flash、gradient checkpointing。

| GPU | BF16 | FP8 scaled | ConvRot INT8 |
|---|---:|---:|---:|
| RTX 3090 | 该配置未装入显存 | 7.1 s/step | 5.3 s/step |
| RTX PRO 6000 Blackwell Max-Q | 2.0 | 2.3 | 1.9；compile 1.8 |

3090 的约 25% 时间下降是相对它的 FP8 路径。该 Krea2 loader 显式 `use_scaled_mm=False`，没有本地 W8A16 同条件对照；Blackwell 的未编译 INT8 相对 BF16 为约 5% 时间改善。模型、硬件、数值规则与训练栈不同，不能据此推断 Anima/3080 收益。

[线上测试](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/tests/test_krea2_convrot_int8.py#L130) 中，前向/输入梯度主要对照量化权重的 dequant reference：forward <3%，默认 grad_x <2%，INT8 backward <5%。另有原始权重 roundtrip 误差测试。固定相同输入及外部 grad_out 的 LoRA A/B 梯度比较，只验证独立 additive 分支；真实训练的 grad_out 会受量化前向影响。

本地完整模型单步 gate 与线上局部随机矩阵测试测量对象不同，而且本地总梯度范数也有盲区。不能直接比较“5% 对 2%”后宣布哪边更严格，更不能将线上局部测试视为通过本地端到端检查。

其他常见线上路径也要区分：bitsandbytes [MatMul8bitLt.backward](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/autograd/_functions.py#L234) 的输入梯度路径会将量化权重恢复到输入 dtype 后做浮点 matmul；可训练权重是另一分支。Diffusers [TorchAO trainable 白名单](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/src/diffusers/quantizers/torchao/torchao_quantizer.py#L493) 表示接口允许训练，不证明 forward/backward 全部 INT8。QAT、8-bit optimizer、推理加载 LoRA 同样不能替代这里的训练实现核验。

## 6. 后续研究应只比较具体差异

本轮结论是保留现有 BF16/W8A16 基线，先不启动实现或新训练。若后续进入优化验证，建议按以下顺序隔离变量：

1. **先核后端与成本。** 固定真实 Anima M/K/N、相同精度/旋转/scope/compile/swap，分别测 RHT、量化、GEMM、后处理、dX、checkpoint 重算和换层；记录实际是否 fallback、全局 TF32、稳态和峰值时刻。
2. **只换工程内核。** rowwise quant 保留本地 FP32 取码、clamp/scale 规则；GEMM 融合缩放而保留 FP32 backward。分别验量化码、输出、梯度，再评完整 step；不把一次全面切换的收益归因于某个优化。
3. **另列数值实验。** regular、BF16 backward、INT8 backward 每次单独改变。除总梯度范数外增加 adapter 梯度向量 relative L2/cosine、分层异常和参数更新对照；使用相同初始化/噪声/时间步，再评采样质量。
4. **最终按相同显存预算评完整配置。** 相同 swap 便于隔离算子，但实用价值还取决于各自可减少多少 swap、能否增加 batch，以及含采样峰值。内核微基准过关不等于应切换产品默认。

这些是待验证方向，没有本轮速度或显存承诺。与重新引入 ConvRot 相比，复用本地已存在的训练、换层和数值检查，把差异缩小到几个 kernel/反向策略，更符合已有证据。

## 证据入口

- [本地 ConvRot 架构/规格（已归档）](../../_archive/docs/proposal/convrot_w8a_training_plan.md)
- [优化路线与历史否决条件](../proposal/convrot_w8a_optimization_roadmap.md)
- [实验记录](../experimental/convrot_int8_training.md)、[1710-step 长训](convrot_longrun_bf16_w8a8_w8a16_20260727.md)
- [上一份扩散量化线上调研](diffusion_mixed_quant_post_training_survey_20261002.md)
- 固定线上源码与 PR 响应缓存：`output/online-diffusion-mixed-quant-20261002/`；musubi 提交与正文链接一致。

本文只更新研究文档，不改变已有配置、权重、训练进程或运行数据。
