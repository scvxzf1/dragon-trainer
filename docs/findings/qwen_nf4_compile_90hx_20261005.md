# Qwen Image 2.1 全 Linear NF4 编译验收

状态：实验，2026-10-05，10 月 6 日追加定位与受限入口验收。
`qwen_nf4_compile_experiment` 已接入真实 `train.py` 训练入口，默认关闭。
CMP 90HX 的完整 NF4、真实 Edit 数据两步训练完成 AdamW 更新、约 67 MB
checkpoint 保存并正常退出；追加两次训练预览、完整状态保存与独立进程
step2→3 续训也均正常退出。所需配置与当前边界见
[实验说明](../experimental/qwen-nf4-compile.md)；默认 NF4 compile 拒绝仍保留。

默认动态编译曾在完整权重验收中失败；静态编译、原生 image LayerNorm、
非 persistent reduction 与精度 cast 保留仍不足以通过真实 flow loss 对照。
定位到 eager 反向波动与真实 MLP SiLU backward 的编译边界微差后，联合严格
确定性和原生 SiLU 的完整 32-block 真实两步对照达到 2246/2246 张量逐位一致。
生产 attention-only、zero-up 初始化的两个真实 Edit bucket 又达到 1286/1286
逐位一致，均有不同代理正常退出的独立重验。原失败、中断及 reporter 错误
记录保留于正文。通过范围限该联合策略，具体浮点指令的因果归属尚未证明。

相关代码：`library/models/qwen_image_2_1/compile.py`、
`scripts/quantization/qwen_nf4_compile_acceptance.py`、
`scripts/quantization/probe_qwen_nf4_compile.py`、
`tests/test_qwen_nf4_compile_training.py`、
`library/quantization/qwen_nf4_compile_policy.py`、
`library/models/qwen_image_2_1/nf4_compile_runtime.py`、
`tests/test_qwen_nf4_compile_experiment.py`。
追加诊断工具：`probe_qwen_nf4_attention_core.py`、`probe_qwen_nf4_projection.py`
与 `probe_qwen_nf4_joint_attention.py`，均位于 `scripts/quantization/`，不接入生产 gate。

## 环境与验收方法

CMP 90HX 10 GiB，PyTorch 2.12.0、bitsandbytes 0.49.2、Triton 3.7.0。
使用既有 `all-linear-nf4-v1` 权重，不加载 text encoder/VAE、不下载权重。
同一 NF4 checkpoint 顺序执行 eager/Inductor，初始 LoRA state 逐位相同。
生产 plain LoRA factory 覆盖 attention 与 MLP；这一覆盖强于当前默认
attention-only 配置，不能把结果直接当作原 1024 数据集任务的测量。

两步 SGD、BF16 autocast、full checkpoint、真实 ModelOffloader；
target latent 从 `[1,64,2,2]` 变为 `[1,64,2,4]`，text 从 3 变为 5 个位置，
reference latent 为 `[1,64,2,2]`。第一步更新后做一次无梯度编辑预测，再恢复训练。
要求每个 LoRA 参数和输入梯度有限、非零，optimizer 实际更新参数；
packed 权重冻结、对象与 QuantState 身份、payload 保持不变。

预置门槛保持不变：输出/输入梯度 `rtol=0.02, atol=0.002`，
FP32 LoRA 梯度/参数 `rtol=0.02, atol=0.0002`，逐张量 relative-L2 不超过 1%。
验收记录实际 Dynamo/Inductor 图、目标块调用及 graph breaks。

## 小模型与 BF16 舍入

4-block 真实 Qwen、全 Linear NF4、rank4。原默认 Inductor 虽能执行，
有 29 项 LoRA 梯度 relative-L2 超限，最大 1.7844%；各项 `assert_close` 通过。
输出最大 relative-L2 0.1324%，输入梯度 0.3940%。

PyTorch Inductor 的 `emulate_precision_casts=True` 保留融合时的 eager BF16
舍入边界。将此选项限定在 Qwen NF4 compiler，不改变普通 BF16 compiler 选项。
采用该选项后，swap1/2 × foreach/slab × all/resident 共 8 组通过，
比较的输出、梯度、更新后参数及预测均逐位相同；总耗时 233.19 秒。
这证明小模型基础路径可编译，不能外推完整权重数值一致。

## 完整权重默认动态编译的独立验收

既有约 3.42 GiB 的 32-block NF4 checkpoint，rank16、swap24、foreach、
scope=all、dynamic_seq=true、Inductor default mode，并保留 BF16 舍入选项。
232 个 packed NF4 模块、224 个 LoRA 模块；32 个 compiled inner forward 各实际调用 5 次，
2 个 unique graphs，`graph_breaks={}`。两步更新、预测后恢复训练及 packed 检查通过。

| 对比量 | 最大 relative-L2 |
| --- | ---: |
| 输出 | 0.7626% |
| loss | 0.0940% |
| latent 梯度（第一步 / 第二步） | 2.4609% / 10.9111% |
| 单个 LoRA 梯度 | 61.3168% |
| 更新后 adapter 参数 | 0.002331% |
| 无梯度编辑预测 | 1.3256% |

1074 个检查失败，其中 175 个 `assert_close` 失败，其余为 relative-L2 超限。
因此失败不只是辅助相对误差门槛。最坏单项 LoRA 梯度是第二步 block5
`img_mlp_gate_layer.lora_down.weight`，最大绝对误差 0.00026727；
相对误差的逐参数最大值不能当作全部梯度的整体误差。
保留 eager 舍入不足以修复完整模型差异。下文已定位部分 LayerNorm/cast 边界，
剩余整网差异尚未全部解释。

| 路径 | 单次验收耗时 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: |
| eager | 26.959 s | 3.752 GiB | 4.162 GiB |
| compiled | 77.344 s | 5.023 GiB | 5.666 GiB |

compiled 耗时包含冷编译；此表不是稳态训练速度结论。未测 1024 实际数据、
真实 Euler/VAE 采样、长期训练或图片质量。所有预测检查均为单次 family forward。

## 复现与证据

### 换层调度隔离（追加验收）

使用同一 32-block、约 3.42 GiB 的完整 NF4 权重，将 `blocks_to_swap` 设为 `0`，其余
参数保持 rank16、all-scope、dynamic sequence、BF16 舍入选项不变。编译仍得到 32 个
compiled blocks、2 个 Inductor graphs、无 graph break，数值失败与 swap24 基本逐项相同：

| 对比量 | no-swap 最大 relative-L2 |
| --- | ---: |
| step0 输出 | 0.7251% |
| step0 latent 梯度 | 2.4609% |
| step1 单个 LoRA 梯度 | 61.3168% |

运行日志：`/tmp/qwen_nf4_compile_count0_20261005.log`；完整比较文件：
`/tmp/qwen_nf4_compile_count0/20261005-202028-574077/comparison.json`。因此 block swap
调度不是主要差异来源。已有 tiny 4-block 对照还显示：无 LoRA 的 base-only、LoRA+gradient
checkpoint、以及 8 组 swap/scope 组合均通过；这些小宽度结果不能外推完整权重。
后续使用下文的完整尺寸 block VJP 隔离具体阶段和算子，compile 数值 gate 继续保留。

尝试关闭完整模型的 gradient checkpointing（仍为 no-swap、all-scope）时，Inductor 首个
完整 forward 在 90HX 10 GiB 上 OOM：约 9.53 GiB 已占用，仅剩 95.5 MiB，额外申请
96 MiB 失败（日志：`/tmp/qwen_nf4_compile_no_ckpt_20261005.log`）。因此当前硬件无法用
“full no-checkpoint”直接完成这条二分；这不是数值通过证据，也不改变生产侧必须使用 full
checkpoint 的兼容边界。

### 完整尺寸 block 与子阶段定位

新增 `scripts/quantization/probe_qwen_nf4_block_precision.py`：用完整 NF4 基座的无梯度
Edit-like forward 捕获 block 输入，再逐块比较 eager、重复 eager、Inductor 的固定 cotangent
VJP。只保留单个 block 的反向图，完整 32-block 基座仍 resident；没有 optimizer 更新。
这使用验收脚本的小 token 输入，**不是实际 1024 图片分辨率的训练验收**。adapter seed=41、
输入 seed=171、cotangent seed=172；同时比较 packed payload、递归 QuantState 和 adapter
raw-byte SHA256。所查四块的重复 eager 全部逐位一致，packed 与 adapter 保持不变。

| block | 输出不同元素 / 40960 | hidden 梯度 relative-L2 | 最坏单个 LoRA 梯度 relative-L2 |
| --- | ---: | ---: | ---: |
| 0 | 0 | 0.003286% | 0.002899% |
| 5 | 146 | 0.154744% | 0.401240% |
| 19 | 0 | 0% | 0% |
| 31 | 10 | 0.066001% | 0.212072% |

block5/31 虽然上述相对误差小于 1%，仍有 `assert_close` 失败，不能视为通过。
关闭 fusion 的 `unfused` 和确认生成 `div_rn` 的 `division` 对照均没有消除 block5 的主要
偏差。`emulate_precision_casts` 已关闭 FP fusion，因此“再关闭 FMA”不是新方案。
关闭冻结 modulation 的额外 VJP 后，block5 主要偏差仍相同。

`qwen_nf4_block_stages.py` 在 compile 外准备共同的 eager 中间输入，避免将上游误差混入
子阶段。block5 的 MLP 输出、hidden VJP、全部 MLP LoRA VJP 逐位一致；attention 输出
逐位一致，但 hidden VJP 的 54 个元素和少量 K LoRA 梯度不同，仍有两项 `assert_close`
失败。独立 norm1/norm2 的 FP32 输出均通过，最大绝对误差为 `3.814697e-6`，relative-L2
约 `6.5e-8` / `6.3e-8`。norm1 经 modulation 转 BF16 后没有差异；**norm2 的微小 FP32
差异经 modulation 转 BF16，改变了两个 MLP 输入元素**，其中一项由 `0.00787353515625`
变为 `0.0078125`。这确认了一个可触发后续差异的精度边界，不证明它是整网唯一根因。

### 保留原生 LayerNorm 的因果对照

实验 helper `qwen_nf4_native_layer_norm.py` 通过 opaque custom ops 保留原生 LayerNorm
forward/backward，只 patch 非 affine `img_norm1/img_norm2`，默认不开启、不接生产。
CUDA AMP 显式使用 FP32；仅支持一阶 VJP。两处 norm 都 patch 时，block5 输出逐位一致、
hidden VJP 仅五个元素不同（relative-L2 `2.765826e-5`），所有检查通过；block19/31 的
输出和梯度逐位一致。但 block0 新出现 1728 个输出元素不同和 16 项检查失败，因此整组
实验仍失败，不能全局采用该补丁。

不同代理独立加载 `.pt` 逐 tensor 核验：上述四块的输入、cotangent、初始 adapter，以及
patch 前后的共同 eager 输出/梯度全部逐位一致。10 月 6 日复核旧结果的实际 arguments，
nativeboth 和仅 native norm2 的 block0 记录都为 `modulation_gradient=false`；旧文称
nativeboth 启用了额外 modulation 求导的描述有误。无 modulation 求导的原 norm
对照仍通过（输出逐位一致、hidden relative-L2 `3.285612e-5`）。独立 opaque
norm 在真实 block0 shape 的 fullgraph VJP 中逐位一致；剩余问题仍需隔离跨阶段融合、cast
和残差运算，不能把 custom op 对照当作生产修复。

独立验收发现 helper 的 fake 统计量曾错误继承 CUDA BF16 dtype；现已修正为 CUDA
FP16/BF16 的 mean/rstd 使用 FP32，CPU 保留输入 dtype。修复后 CPU/CUDA ×
FP32/FP16/BF16 六组 `torch.library.opcheck` 的 schema、autograd、fake tensor、动态 AOT
检查全部通过。该问题不直接解释上述 AMP 结果，因为 AMP 路径输入已为 FP32。

进一步将干预范围缩到 **只 patch `img_norm2`**，block0/5/19/31 全部通过原门槛，
四块输出逐位一致；block0/5 的 hidden VJP relative-L2 分别为 `3.285612e-5` /
`2.765826e-5`，最坏单个 LoRA VJP 分别为 `2.898553e-5` / `5.305578e-6`；
block19/31 的 VJP 逐位一致。每块产生一个实际 fullgraph、无 graph break，packed 和
adapter SHA256 不变。CPU 再次逐 tensor 核验与控制组的输入、cotangent 和共同 eager
结果完全一致，记录在该目录的 `intervention_fairness.json`。

### 仅原生 norm2 的完整两步复查

用临时 wrapper `/tmp/probe_qwen_nf4_native_norm2_full_20261005.py` 复用原 `run_case` 和
`compare_runs`，保持 32-block、rank16、full checkpoint、swap24、foreach、all scope、
dynamic sequence 和原阈值，只在 loader 中 patch `img_norm2`。32 个 compiled inner
forward 各实际调用五次，2 个 unique graphs，无 graph break；packed 原检查通过。
eager 和 compiled 的 672 个初始 adapter state tensors 逐位一致。追加 CPU 核验发现，
patch 后 eager 的两步输出/loss/latent 梯度、每步 448 个 LoRA 梯度、672 个更新后 state
tensors 和中间无梯度预测，与先前未 patch 的 no-swap eager 控制全部逐位一致。

| 对比量 | 原完整模型 | 仅原生 norm2 |
| --- | ---: | ---: |
| 输出最大 relative-L2 | 0.7626% | 0.6304% |
| 第一步 latent 梯度 relative-L2 | 2.4609% | 2.2303% |
| 第二步 latent 梯度 relative-L2 | 10.9111% | 4.8675% |
| 第一步最坏单个 LoRA 梯度 relative-L2 | 15.1079% | 13.7633% |
| 第二步最坏单个 LoRA 梯度 relative-L2 | 61.3168% | 42.6115% |
| 无梯度编辑预测 relative-L2 | 1.3256% | 0.7452% |

1033 个检查仍失败，涉及 901 个不同键，其中 135 个 `assert_close` 失败。
最坏 LoRA 梯度变为第二步 block0 `attn_to_q.lora_up.weight`。eager/compiled 单次耗时
28.89 / 70.21 秒，compiled 包含冷编译，不能当作稳态性能比较。实验退出码 `1`，
说明局部四块通过不能外推整网；保留 norm2 改善了部分指标，尚未解释全部差异。
下一步应扩展逐块覆盖并比较两种序列长度，追踪其余 norm、attention 与跨阶段的 cast
边界。生产 gate 未更改，未将 custom op helper 接入训练。

完整结果：`/tmp/qwen_nf4_native_norm2_full_20261005/20261005-225141-093992/result.json`，
同目录保存 `comparison.json`、`eager.pt`、`compiled.pt`；日志：
`/tmp/qwen_nf4_native_norm2_full_20261005.log`。

### 扩展 block 和 norm 入口（10 月 5 日追加）

固定初始 adapter、输入索引 0、随机 cotangent、关闭额外 modulation 求导，仅保留原生
`img_norm2`。block1/2/3/9 通过；block4 输出逐位一致，但 K projection 的
`lora_down.weight` 有 12/65536 个元素不满足 `assert_close`，失败元素中的最大绝对误差
`0.0005607605`；全 tensor 最大绝对误差 `0.0078125`，869 个元素字节不同，
relative-L2 `0.000528305`。block6/7/8 的输出分别有
10028/11791/8585 个不同元素，整组失败。

两处 image norm 都保留原生后，block6/7/8 的输出、hidden VJP 和全部 LoRA VJP
逐位一致，block4 的 K 梯度偏差仍存在。结合此前 block0 的回归，这个补丁仍不适合
全局采用。证据：
`/tmp/qwen_nf4_native_norm2_unsampled_20261005/20261005-225816-355232/result.json`、
`/tmp/qwen_nf4_native_both_unsampled_20261005/20261005-230538-016704/result.json`。

新增 `norm1-entry` / `norm2-entry` 阶段，比较真实 norm → 冻结 modulation → BF16
cast 的入口；norm2 的共同残差输入在 compile 外 eager 准备，norm 阶段没有 adapter
求导目标。原 norm1-entry 的 block0/6 分别有 1/2 个 BF16 元素不同，最大绝对误差
`1.396984e-9` / `3.814697e-6`，hidden VJP 逐位一致且门槛通过；保留原生 norm1
后二者输出和 VJP 都逐位一致。这说明微小入口差异确实存在，不能仅凭入口通过
外推下游矩阵和整网通过。证据：
`/tmp/qwen_nf4_norm1_entry_original_20261005/20261005-230952-676168/result.json`、
`/tmp/qwen_nf4_norm1_entry_native_20261005/20261005-233004-649564/result.json`。

### attention 核心和真实合成损失 cotangent

独立核心工具分别隔离 Q/K RMSNorm、complex RoPE、真实 segmented SDPA，以及
norm+RoPE、norm+RoPE+SDPA 组合。Q/K[/V] 是保持原 stride 的独立叶子，投影在
compile 外完成，故这些阶段没有 NF4 Linear 或 LoRA 梯度覆盖。block4 的随机
cotangent 对照中，五阶段输出和全部输入 VJP 逐位一致。

随后从完整 32-block、full non-reentrant checkpoint、swap24 的首次前向 tensor
捕获真实反向 cotangent；目标是合成 Edit fixture 的
`output.float().square().mean()`，不是训练器的 flow-matching target loss。没有 optimizer
更新，损失 `3.3527064323425293`，与原完整模型 eager step0 一致。五阶段均通过原门槛；
单独 qk-norm 的 dK 有一个元素不同，最大绝对误差 `4.76837e-7`、relative-L2
`2.50191e-5`，其余阶段（包括完整 core）输出和 dQ/dK/dV 逐位一致。

不同代理独立核对保存的 tensor bytes、真实公式、首前向 hooks、清理和状态 hashes；
另确认 SDPA VJP 等于捕获的 RoPE cotangent，RoPE VJP 等于 norm cotangent，core
Q/K VJP 等于 qk-rope Q/K VJP。原 source 比较曾用 `torch.equal`，不能区分正负零；
现已改为 uint8 bytes，已有输出也独立按字节复核通过。这支持当前捕获链路，不证明
完整 attention 或其它长度已通过。证据：
`/tmp/qwen_nf4_attention_full_loss_block4_20261005/20261005-233839-289714/result.json`。

### Q/K/V/out 投影对照（独立真机验收）

使用上述合成损失的真实投影输入/输出/cotangent，比较实际 NF4 + 生产 LoRA forward，
梯度目标是现有 network 的 FP32 down/up 参数。原始 Q/K/V 输入是 FP32，out 输入
是 BF16；额外对照先将输入转 BF16。两组的四个投影全部逐位通过：每个输出和输入
梯度均为 40960 个元素、每个 down/up 梯度为 65536 个元素，compiled 与 eager 的
不同元素和 relative-L2 均为 0。source output 与重复 eager 均按字节一致；每个投影
实际产生一个 fullgraph，无 graph break，adapter/packed hashes 前后不变。

独立验收保留原门槛，两组 exit 0。因为 original+lora 没有失败，未运行 base-only
反事实对照。只能排除这些固定输入/状态下的独立投影，仍须查共享入口、梯度汇合、
投影→norm→attention 连接图和其它长度。证据：
`/tmp/qwen_nf4_projection_original_lora_20261005/20261005-235513-313246/result.json`、
`/tmp/qwen_nf4_projection_bf16_lora_20261005/20261005-235652-230273/result.json`。

### QKV 和 attention 连接图（10 月 6 日独立真机验收）

新增连接图工具，以同一个实际 FP32 投影输入叶子同时驱动 Q/K/V，分别编译
QKV、QKV→RMSNorm→complex RoPE→segmented SDPA、完整 attention（追加 out
projection/dropout）。三阶段实际 LoRA 参数覆盖为 6/6/8 个。没有在 compile 内
重新计算 norm1/modulation，也没有包含 gate/residual、norm2 或 MLP。

block4 和 block0 的 shared/original 三阶段共 56 项输出、输入梯度及 FP32 LoRA
梯度全部逐位一致，eager 重复稳定、source bytes 一致，实际 graph 覆盖和状态
hashes 通过，两组 exit0。因全部通过，未运行 independent/BF16 连接图矩阵。
该结果在两个 block 的固定输入下没有复现共享 QKV 汇合误差；剩余定位应检查
norm/modulation→attention 和 gate/residual→norm2/MLP 的连接边界。

独立静态/CPU 审查确认公式和真实参数目标，同时指出 shared 叶子 VJP 不能要求
逐位等于三个 independent VJP 在事后 FP32 相加：BF16 autocast 的 cast 和累加位置
不同会改变舍入。诊断工具各布局分别比较自身 eager/compiled，跨布局差异本身
不能当作 compiler 错误。证据：
`/tmp/qwen_nf4_joint_shared_original_20261006/20261006-001605-451486/result.json`、
`/tmp/qwen_nf4_joint_block0_shared_original_20261006/20261006-001822-484057/result.json`；
字节独立复核：`/tmp/qwen_nf4_joint_gpu_summary_20261006.json`。

### 冻结 gate 和残差小算子（10 月 6 日追加）

用三份 block0 的真实 BF16 hidden/modulation/mask 输入隔离 modulation row 选择、
`1+scale`、两组 gate 的 `tanh`、乘权和单/双残差。gate/modulation 冻结，hidden
保持原 BF16 dtype 作为唯一叶子。attention 固定输出借用 block5，长度 16 作平铺
截取，MLP 用等 RMS 的随机张量；因此这是反事实局部算子对照，不是真实 block0
attention/MLP 图。

共 72 项比较全部满足原数值门槛，51 项前向输出逐字节一致，gate tanh 没有输出
差异。12 项 BF16 VJP 有共 700 处正零/负零字节差异，数值 `max_abs=0`、relative-L2=0；
不能称全部严格逐位一致，也不能用这些零符号解释此前非零的输出/LoRA 梯度误差。
eager/compiled 各自重复结果逐位稳定。探针 exit0，证据：
`/tmp/qwen_nf4_gate_rounding_20261006/20261006-002953/result.json`；同目录保留图证据
和输入/输出/VJP。下一步用 `attention-prefix`、`attention-residual`、`mlp-suffix`
检验包含真实 NF4/LoRA 的 norm/gate/residual 连接边界。

### 第二种序列长度（10 月 6 日追加）

`--edit-input-index 1` 使用固定初始 adapter，hidden 从 `[1,10,4096]` 变为
`[1,16,4096]`，不是原两步验收中更新后的 step1。仅原生 norm2、相同 seeds、随机
cotangent、关闭 modulation 求导，结果如下。所有 relative-L2 虽小于 1%，失败项仍
触发原 `assert_close`，不应以辅助相对误差通过代替完整门槛。

| block | 输出不同元素 / relative-L2 | hidden 梯度 relative-L2 | 最大单 tensor relative-L2 | 原门槛 |
| --- | ---: | ---: | ---: | --- |
| 0 | 16770 / 0.206804% | 0.470036% | 0.554082% | FAIL |
| 4 | 0 / 0% | 0% | 0% | 逐位 PASS |
| 5 | 7060 / 0.016334% | 0.254706% | 0.637621% | FAIL |
| 19 | 0 / 0% | 0% | 0.0000911% | PASS，K-up 一个元素不同 |
| 31 | 1800 / 0.011721% | 0.157499% | 0.622358% | FAIL |

block0/5/31 的输出、hidden 梯度和 14 个 LoRA 梯度全部触发 `assert_close`。
全部重复 eager 逐位稳定、finite，每块一个实际 graph、无 break，adapter/packed
前后不变且与输入索引 0 的 hashes 一致。未 patch 对照确认输入、modulation、rotary、
mask 和 cotangent 相同；block0/4/5 失败，block19 通过。该对照 timeout175、exit124，
block31 未完成，block19 的已保存 tensor 使用原 `_compare` 在 CPU 恢复比较，不改写
原 partial result。不能将未完成对照描述为完整验收。

norm2 完整结果（exit1）：
`/tmp/qwen_nf4_second_length_norm2_20261006/20261006-000330-706602/result.json`；
未 patch 部分结果：
`/tmp/qwen_nf4_second_length_original_20261006/20261006-000736-027724/result.json`。
各父目录保留日志、退出码和汇总。交接时一个旧模型文件名不存在而 exit1，随后使用
同目录现有 `qwen_image_2.1_nf4.safetensors`，未下载或替换模型。该结果确认长度会改变
局部偏差，不能把 index0 的通过外推其它形状，更不能解除完整模型 compile gate。

### norm1 连接边界的因果对照（10 月 6 日追加）

按生产公式新增三个连接阶段，输出保持真实计算 dtype，不额外加 BF16 cast：
`attention-prefix` 包含 norm1/mod1/attention；`attention-residual` 追加真实 gate1
残差；`mlp-suffix` 从共同 eager post-attention hidden 开始，包含 norm2/mod2/MLP
和 gate2 残差。分别覆盖实际 8/8/6 个 FP32 LoRA 参数及 4/4/3 个 NF4 Linear。

同一 block0、输入索引 1、随机 cotangent、冻结 modulation 的独立真机结果：

| 阶段 / 原生 norm | 输出 relative-L2 | hidden VJP relative-L2 | LoRA `assert_close` | exit |
| --- | ---: | ---: | --- | ---: |
| attention-prefix / norm2 | 0.054557% | 0.181390% | 8/8 失败 | 1 |
| attention-residual / norm2 | 0.025977% | 0.148654% | 8/8 失败 | 1 |
| mlp-suffix / norm2 | 0% | 0% | 6/6 逐位通过 | 0 |
| attention-prefix / norm1+norm2 | 0% | 0.001994% | 8/8 通过 | 0 |

最后一项只增加原生 norm1 干预（prefix 不执行 norm2），source-bound/cotangent
hashes、packed 和初始 adapter 都与控制相同。主代理另逐 tensor 比较两组共同
eager 的输出、hidden VJP 和 8 个 LoRA VJP，10 个 tensor 全部 raw bytes 相等。
这排除了改变 eager 基准导致通过的解释，确认 `img_norm1` 连接边界是当前形状
的一个数值偏差来源。

nativeboth 的输出和六个参数 VJP 逐位一致；hidden、K-down、K-up VJP 分别有
4/70/1 个元素不同，relative-L2 为 `0.00199395%` / `0.00388230%` / `0.000116165%`，
均满足原门槛。四次运行重复 eager 逐位稳定、全部 finite，每次实际一个 graph、
无 graph break，packed/adapter 前后未改变。仍不能外推整个 block 或整网通过，
尤其两处 norm 都 native 在输入索引 0/block0 的回归仍未修复；追加复现见下文。

汇总：`/tmp/qwen_nf4_connected_stages_summary_20261006.json`。原始结果：

- prefix：`/tmp/qwen_nf4_connected_stages_attention-prefix_20261006/20261006-003538-073244/result.json`。
- residual：`/tmp/qwen_nf4_connected_stages_attention-residual_20261006/20261006-003723-857270/result.json`。
- suffix：`/tmp/qwen_nf4_connected_stages_mlp-suffix_20261006/20261006-003909-058117/result.json`。
- prefix nativeboth：`/tmp/qwen_nf4_connected_stages_attention-prefix_nativeboth_20261006/20261006-004044-129953/result.json`。

### 输入索引 0 的 nativeboth 回归复现

不同代理用冻结 modulation、rank16、native attention、fullgraph、无 checkpoint 的
相同条件重跑 block0。完整 block 输出有 1728 个元素不同、最大绝对误差 2；hidden
VJP 有 5278 个元素不同、最大绝对误差 4，输出/hidden/14 个 LoRA 的 `assert_close`
全部失败。只保留 norm1/mod1/attention 的 `attention-prefix` 已复现失败：输出
484 个元素不同、最大绝对误差 0.25；hidden VJP 384 个元素不同、最大绝对误差 0.5。
Q-down、K-down/up、out-down/up 的门槛失败，V-down/up 逐位一致。
`attention-residual` 亦失败；从共同 post-attention hidden 开始的 `mlp-suffix`
输出/hidden/6 个 LoRA 全部逐位一致并通过。

因此输入索引 0 的回归在加入 gate1 残差、norm2 或 MLP 前已经出现。各项重复 eager
逐位稳定，实际一个 fullgraph、无 graph break，packed/adapter 未改变；差异不是
signed-zero。CPU 复核旧 nativeboth 与旧 norm2-only 的输入 31 个叶子、初始 adapter
672 个张量和共同 eager 16 个目标均逐字节相同；新旧 nativeboth 的 compiled 16 个
目标也逐字节相同。该复现排除了输入、初始权重或 modulation 求导集合不同的解释，
尚未定位连接图内唯一算子。

证据：`/tmp/qwen_nf4_native_both_block0_review_20261006.json`、
`/tmp/qwen_nf4_native_both_block0_cpu_comparison_20261006.json`；四项原始目录为
`/tmp/qwen_nf4_native_both_block0_{full,attention_prefix,attention_residual,mlp_suffix}_20261006/`。
probe 的 VJP 描述已按 `modulation_gradient` 实值生成，避免 manifest 错列求导目标。

### 连接前缀的编译选项对照

对上述 input0/block0/nativeboth `attention-prefix` 做三项单变量对照，全部保留
`emulate_precision_casts=True` 和原数值门槛。unfused（关闭 epilogue fusion、
`max_fusion_size=1`）与 division（`eager_numerics.division_rounding=True`）仍使用
dynamic shape；其 compiled 输出/全部梯度和指标与原 casts 控制逐字节一致，均无改善。
静态 shape（`dynamic=False`）则明显改善，但仍未通过：

| 对比量 | dynamic casts / unfused / division | static casts |
| --- | --- | --- |
| 输出：不同元素 / max abs | 484 / 0.25 | 0 / 0 |
| hidden VJP：不同元素 / max abs | 384 / 0.5 | 66 / 0.0625 |
| hidden VJP relative-L2 | 0.04614% | 0.005875% |
| LoRA `assert_close` 失败 | 5/8 | 1/8（K-up） |
| K-up VJP max abs | 0.03125 | 0.00390625 |
| exit | 1 | 1 |

三项 initial adapter、captured inputs/cotangent、eager/repeat-eager 都与控制逐字节
一致，packed/QuantState 未变；各有一个实际 graph、无 break、全部 finite，未超时。
静态形状只改善本次 shape，hidden 和 K-up 仍有 `assert_close` 失败，不能据此解锁整网。
static counters 出现 `pad_mm_bench=9`、`pad_rewritten_as_cat=3`，dynamic 没有；这是
代码生成差异线索，尚未归因到具体误差源。“unfused”也不代表所有算子完全未融合，
生成 trace 仍包含 norm/modulation/attention 内部的融合运算。

汇总与 CPU 字节复核：`/tmp/qwen_nf4_norm1_options_summary_20261006.json`；
trace 关键行：`/tmp/qwen_nf4_norm1_options_trace_20261006.txt`；原始结果在
`/tmp/qwen_nf4_norm1_options_{unfused,division,static}_20261006/`。
后续单变量 codegen 干预和完整模型候选验收见下文；生产 gate 保持不变。

### norm1 到单投影的连接对照与独立审查

新增 `norm1-to-q/k/v`，从原始 block hidden 开始，按真实 norm1/mod1 公式调用 patched
NF4 投影，不提前加入 BF16 cast。FP32 指 adapter 参数存储，rank 矩阵运算按真实
BF16 autocast/base-output dtype 执行。只求对应 down/up 两参数；whole-block 7 个
packed 模块保持检查，实际 phase 执行 1 个。cotangent 为 seed172 独立阶段 VJP，
不是真实 attention 上游梯度或 flow loss。

input0 的 K dynamic/static、Q dynamic、V dynamic，以及 input1 的 K dynamic/static
六项均 exit `0`：输出、hidden/down/up VJP 全部 raw bytes 相同，max abs/relative-L2
为 0，每次实际一个 fullgraph、无 break、VJP calls=1。新阶段用于推导输出 shape 的
共同 no-grad pre-call 不计入 calls；旧阶段的 cotangent 和调用行为保持原样。

不同代理用真实小 Qwen + 生产 LoRA 对照 block hook 捕获投影，输出和三类 VJP 逐位
一致；22 项 CPU 测试通过。六项 artifact 直接 CPU uint8 比较，原 bound 中 hidden、
modulation、rotary、target mask、key-valid、cotangent 和初始 adapter 均与旧控制
相同。这排除了该独立 cotangent 下 norm1→单投影连接，不能排除真实 attention
cotangent 下的响应。证据：`/tmp/qwen_nf4_norm1_projection_20261006/summary.json`。

### codegen 核验与非 persistent reduction 干预

CPU trace 审查以缓存 `source_code`/磁盘源码字节、AOT 方向、时间、shape 和 post-grad
File/code annotations 对应 forward/backward；显式 provenance stack 字段为空，
不将候选 cache 当作唯一执行或根因证明。static 实际矩阵乘仍为 M=10 / gradient K=10，
没有选用补长 GEMM；`pad_mm_bench=9` 仅为 benchmark 尝试。
`pad_rewritten_as_cat=3` 来自三段 SDPA mask 的右 pad，不能解释为矩阵 padding。
两组都保留 opaque native norm1 和 efficient SDPA，Q/K RMS forward/backward 为
128-wide persistent reduction。审查原文与 cache 索引：
`/tmp/qwen_static_dynamic_trace_audit_report_20261006.md` 和同名 `index` JSON。

保持其他条件，分别只改一项：

| 配置 | 输出 / hidden VJP 不同元素 | LoRA close 失败 | exit |
| --- | --- | --- | ---: |
| dynamic + specialize_float | 484 / 384 | 5/8 | 1 |
| dynamic + force_pointwise_cat | 0 / 64 | 1/8（K-up） | 1 |
| static + persistent reductions=false，input0 prefix | 0 / 0 | 0/8 | 0 |
| 同配置，input1 prefix | 0 / 0 | 0/8 | 0 |
| 同配置，input0 完整 block0 | 0 / 0 | 0/14 | 0 |

specialize_float 确实消除了 CPU f64 scalar primals，却与原动态失败的输出/梯度字节
相同。pointwise cat 的数值虽改善，前向可执行源码与原 casts 控制相同，目标 cat
未确认发生变化，不能据此作因果归因。nonpersistent 对照则在真实 Q/K forward/
backward 从 persistent 切到普通 reduction 后，三个局部 case 全部逐位一致。
各项 inputs/cotangent/initial-adapter/eager/repeat-eager 与控制逐字节相同，packed
和 adapter 未变，原门槛保留。证据：
`/tmp/qwen_nf4_prefix_codegen_interventions_20261006/{report.md,summary.json}`。
该局部 A/B 支持 reduction 选择是当前组合的数值因素，不是全模型唯一根因结论。

### 静态 + 原生 norm + 非 persistent 的完整两步验收

完整 32-block、232 packed、rank16、swap24/foreach/all scope、full checkpoint，
两种原 Edit fixture 和两步 SGD + 中间预测，eager/compiled 同时 patch 64 个
`img_norm1/img_norm2`。编译使用 `dynamic_seq=False`、
`emulate_precision_casts=True`、`triton.persistent_reductions=False`。
原 compare_runs 的 2247 项比较 0 失败，672 个初始 adapter 张量 exact 相同。
32 个 compiled inner 各调用 5 次，3 个 graphs、无 break。

| 对比量 | 结果 |
| --- | --- |
| 两步 output/loss/latent gradient、中间预测 | 全部逐位相同 |
| step0 的 448 LoRA gradients / 672 更新 state | 全部逐位相同 |
| step1 LoRA gradients | 106/448 非 bitwise；max abs `3.051758e-5`，最差 relative-L2 `0.501970%` |
| step1 更新 state | 106/672 非 bitwise；max abs `3.166497e-8`，最差 relative-L2 `0.000016001%` |

上述残差全部通过原 close 与 relative-L2≤1% 门槛。每个 run 核对 packed 与 QuantState
对象 identity、frozen/no-grad，另 CPU 字节核对 3,557,556,224 packed bytes 和
113,164,064 QuantState bytes 不变。共同 patch eager 与旧 no-patch count0、旧
no-patch swap24、旧 norm2-only swap24 的初始权重/所有记录逐字节相同，未偷换基准。
同配置 compiled repeat 又通过原门槛，并与首轮 compiled 全部张量逐字节相同。
耗时含冷编译，不能作为稳态速度结论。

初始 wrapper 证据：`/tmp/qwen_nf4_full_static_nonpersistent_20261006/`，包含
`eager.pt`、`compiled.pt`、`comparison.json`、`result.json`、复跑与 raw 对照。
随后新增显式诊断 CLI 参数，默认 dynamic/persistent/native=`true/true/false` 保持
原行为。由不同代理直接执行原生 CLI，无临时 torch/compiler 注入，exit `0`，约 89 秒。
新旧 eager/compiled 各 2919 个张量、838,869,768 字节全相同；native patch 每 run
64 个、32 compiled blocks/3 graphs/无 break，最差梯度 relative-L2 同为 0.501970%。
新 CLI 本次 packed payload/object 检查通过，没有额外保存全 QuantState raw snapshot。

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU的UUID> timeout 300 .venv/bin/python \
  scripts/quantization/probe_qwen_nf4_compile.py --model /path/qwen-nf4.safetensors \
  --output-dir /tmp/qwen-nf4-static-diagnostic --count 24 --restore-mode foreach --scope all \
  --no-dynamic --no-persistent-reductions --native-layer-norm
```

独立原生 CLI 证据：`/tmp/qwen_nf4_diagnostic_cli_acceptance_20261006/report.md`、
`20261006-023058-997722/result.json`、`raw_comparison.json`。默认兼容、配置与失败留档
CPU 检查以及 5 项 compile 契约测试通过；诊断失败会保存原 error 和已有 `.pt`/comparison。
所有新增入口仅用于诊断。此处的通过结论限于合成输入；真实缓存和训练 loss 的
追加失败见下节。长期/质量和 resume 未验收，NF4 compile 生产 gate 保持。

### 真实 Edit 缓存与 flow loss 两步对照：失败

同一静态/native norm/非 persistent reduction 组合，在既有真实 pair 上完成
两步 SGD，exit `1`，138.15 秒，无 OOM 或 timeout。target 为 FP32
`[1,64,74,56]`（896×1184 bucket），reference 为 FP32 `[1,64,76,54]`，
Qwen3-VL cache 为 BF16 `[1,1070,4096]`，reference image slots 为 1026。
caption、参考图与 bucket fingerprint 匹配；不重缓存、裁剪或缩小 shape。

沿用实际 `family.forward_for_loss` 与 `_flow_match_loss`：固定 sigma=0.5、
noise seed=171，`noisy=(1-sigma)*latent+sigma*noise`，`target=noise-latent`；
FP32 prediction/target 的 MSE，weighting/sampleweight 为 1，无 mask，最后 mean。
两侧初始 672 个 adapter state 张量逐位相同，每步均更新全部 448 个参数。
仍为完整 32 blocks、rank16 attention+MLP 诊断 LoRA、full checkpoint、
swap24/foreach/all；该 adapter 覆盖区别于生产 attention-only。

| 步骤 | eager loss | compiled loss | output relative-L2 | latent gradient relative-L2 | 最差 LoRA gradient relative-L2 | 超过梯度门槛 |
| --- | --- | --- | --- | --- | --- | --- |
| step0 | 0.2706372738 | 0.2706372738 | 0，逐位相同 | 1.8340% | 1.6666% | 24/448 |
| step1 | 0.2701410055 | 0.2704493105 | 0.8906% | 66.5417% | 43.8067% | 446/448 |

2246 项 tensor 比较产生 482 次失败检查；同一 tensor 的 close 和 relative-L2
可分别失败。step1 output 虽 relative-L2 小于 1%，仍不通过原 close，
max abs 为 0.3984375。所有张量有限，更新后的 adapter state 比较仍通过，
不能以 state 通过代替梯度门槛。32 个 compiled inner 各调用 4 次，
1 个 graph、无 break；232 个 NF4 模块的冻结、对象身份、packed payload
与 QuantState tensor bytes 不变。未独立快照 QuantState 的 scalar attributes。

完整证据：`/tmp/qwen_nf4_real_edit_flow_20261006/`，包括 `report.md`、
`comparison.json`、`metrics_summary.json`、真实 fixture、eager/compiled 张量、
运行脚本与日志。这里只覆盖一个真实 pair 和固定 sigma，没有长期或质量结论。

### 真实 step0 逐层反向捕获：分叉位置与复跑局限

追加只做 step0、零 optimizer 更新的捕获，wrapper 位于 compiled inner 外；
各层首次 forward 与 checkpoint 重算分开计数。完整真实输入保留，不缩 shape。
该次 32 层的 hidden 输入与 forward 输出均逐位相同，block31 的 output
cotangent 也逐位相同；其 input VJP relative-L2 为 0.005239508%，
max abs 为 `1.49011611938e-8`，低于原门槛。逆序首次超过 1% 的
input VJP 位于 block11；最终 latent gradient 为 1.65936295%，
17/448 个 LoRA gradient 失败，最差为 1.39085306%。

这不是唯一算子归因：捕获侧 eager latent VJP 与原两步实验 step0 相差
1.54656777%，compiled 相差 1.68656595%；捕获前后两轮的 eager 梯度也不同。
输出/loss 和初始化摘要相同，但 hooks 引入图外同步，修复复跑还将 CPU
线程数设为 8，尚未建立稳定的 eager repeat 对照。需要相同 input/cotangent
的局部 checkpoint VJP 回放和复跑控制，才能判定具体 backward 差异。

产物：`/tmp/qwen_nf4_real_flow_backward_capture_20261006/`，含全层指标
`result.json` / `block_metrics.csv`、block31 的真实输入/输出/cotangent/VJP
以及 step0 张量。首轮捕获完成后因 scalar byte-view 统计错误 exit `1`；
修正临时脚本后捕获 exit `0`（181.60 秒），1 个 graph、无 break。
该目录的局部 `replay.py` 未执行，`skipped_time_budget` 不代表数值通过；
另一个代理在新目录完成的局部回放见下节。
不同代理的 CPU 审查独立重算 block31、最终梯度、448 个 LoRA gradients，
并核验六项 manifest hashes、完整 bound 参数与小型 checkpoint hook 机制，
结果吻合。全32层 raw equality 仅由脚本和统计支持，独立原始张量复核限于
已保存的 block31。审查对捕获机制给予有限支持，对数值 gate 仍判失败；
报告为 `/tmp/qwen_real_capture_independent_review_20261006/report.md`。

### block31 真实 cotangent 的局部回放

不同代理从捕获产物精确恢复 rank16 adapter，仅将 block31 与对应 adapter
放 CUDA，其余模型留 CPU；保留真实输入/常量/cotangent 的 stride、native norms、
static/nonpersistent 选项与 non-reentrant checkpoint，不缩 shape。

| 对照 | hidden VJP relative-L2 | 最大 adapter VJP relative-L2 |
| --- | --- | --- |
| full capture → local eager（hidden/output） | 0.002732% | 此轮未比较 |
| local eager → repeat eager | 0.002716% | 0.047256% |
| local eager → compiled | 0.005332% | 0.052681% |
| MLP-suffix eager → compiled | 0.00004405% | 0.008426% |

输出均逐位相同，所有局部对照通过原 close 和 relative-L2≤1% 门槛。
whole-block eager repeat 的梯度不逐位相同，所以 compiled 微差仍不能唯一归因。
MLP-suffix 使用真实 full VJP 的 after-attention 输入、梯度和原 output cotangent；
其 eager 与 full VJP 对照、eager repeat 均逐位一致。compiled suffix 的 hidden
83 个元素、gate_layer down/up 的 84/2 个元素有微差，全部低于原门槛。
这只隔离了一个连接阶段，尚未覆盖其余层或解决全模型累积失败。

两个 compiled case 各为 1 个实际 graph、无 break、2 次调用；7 个 NF4 模块
与14个 adapter 参数的字节、对象/QuantState/data pointer 身份不变，`.grad=None`。
GPU 命令分别 exit `0`（23.68 / 27.25 秒），90HX 退出为空闲。
产物与精确命令见 `/tmp/qwen_nf4_real_block31_replay_20261006/{report.md,summary.json}`。
生产 gate 保持；完整 eager repeat 与严格确定性两步对照的追加结果见下节。

### 完整 eager 复跑基线与严格确定性控制

移除新增逐层诊断 hooks，保留 block swap 原有 backward hooks，固定 CPU
threads=8、同一真实 fixture 与原始 adapter 初始化；零 optimizer 更新。
两个模型实例各重复一次相同 forward/backward：同实例 latent VJP
relative-L2 为 1.442418% / 1.513365%，跨实例为 1.529111% / 1.660922%。
输出与 loss 逐位一致，但原 1% 梯度门槛失败；adapter gradient 超限数量分别
为 17/15/21/14。该基线说明原 eager/compiled 差异包含 eager 自身波动。

独立进程联合启用 `torch.use_deterministic_algorithms(True, warn_only=False)`
与 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 后，单实例两次 latent VJP 逐位相同；
448 个 adapter VJP 及 output/loss 的逐位一致由代码、JSON 和日志支持，
这轮未保存全部原始张量。strict 数值记录及 packed 检查已完成，但外层
`timeout 70` 在 RESULT 后 exit `124`，不能把整条命令报告为 PASS。

CPU 独立审查重算六份 latent、fixture hashes 和原 eager 初始化 digest，
核验上述结果；完整梯度结论的独立原始复算范围仍有限。产物为
`/tmp/qwen_nf4_real_eager_repeat_control_20261006/`，独立报告为
`/tmp/qwen_eager_repeat_control_review_20261006/review.md`。

Torch 2.12 精确 commit 的源码审查表明 native SDPA 会自动选子后端，显式
mask 排除 native Flash；memory-efficient backward 的 split-key 模式可能
非确定，strict 会强制 split=1。此次尚未记录实际 ATen 子算子，且 strict
也影响 Inductor 实现选择，不能把联合控制当作 split-key 单因素因果证明。
审查原文与官方源码 URL 保存在
`/tmp/qwen_native_sdpa_determinism_audit_20261006.md`。

### 严格确定性下的真实两步训练对照：仍失败

追加完整自然终态的 strict eager / compiled / fresh eager，合计六个短步，
GPU 命令 exit `1`（非 timeout），200.099 秒。保持原真实 pair、flow MSE、
sigma=0.5、noise seed171、rank16 attention+MLP、full checkpoint、swap24，
native norms/static/nonpersistent；每步 448 个参数均有效 SGD(lr=0.001) 更新。
strict 在 trace 前设置，两侧进程级 CUBLAS workspace 一致，CPU/interop
threads=8，native attention context 包围 forward、backward 和 checkpoint 重算。

fresh eager 与首轮 eager 的两步 2246 个比较张量全部逐位一致，原门槛零失败，
涵盖 output、loss、latent VJP、448 gradients 和 672 更新 state。编译对照仍
有 488 次失败检查（465 个 tensor keys；23 close、465 relative-L2 失败）。

| 步骤 | eager loss | compiled loss | output relative-L2 | latent VJP relative-L2 | 最差 adapter VJP relative-L2 | 超限 adapter VJP |
| --- | --- | --- | --- | --- | --- | --- |
| step0 | 0.2706372738 | 0.2706372738 | 0，逐位相同 | 1.626380% | 1.333123% | 16/448 |
| step1 | 0.2701036632 | 0.2705284059 | 1.028889% | 80.152309% | 46.005902% | 446/448 |

step1 output max abs=0.4921875，latent VJP max abs=0.00401306152。
所有 tensor 有限，更新 state 仍过原门槛；这不替代 gradient 验收。
32 个 compiled blocks 各4次调用，1个实际 graph、无 break；三轮232个 NF4
模块的 packed/QuantState tensor bytes、冻结与对象身份检查通过，66个非packed
冻结参数/缓冲在各轮前后与跨轮均相同。没有改变原 gate 或降门槛。

完整 eager/compiled 张量、精确命令与退出码、repeat raw hashes 和各项比较见
`output/tests/qwen_nf4_real_flow_strict_20261006/`；不同代理独立重算全部2246项，
指标与488次失败完全吻合，并逐参数重建四步SGD更新，全部逐位一致。
独立报告为 `/tmp/qwen_strict_two_step_independent_review_20261006/review.md`。
该结果建立了稳定的两步 eager 基线；剩余差异的局部算子隔离见下节。
仍限一个 pair、固定 sigma 与诊断 adapter，不能放行生产 compile。

### 严格真实 block31 的 SiLU backward 因果对照

固定既有真实 block31 输入与 flow loss cotangent，strict eager repeat 的
output、hidden VJP 和14个 adapter VJP（16张量）全部逐位一致。
attention-residual、shared QKV 和真实 Q/K RMSNorm adjoint 的 eager/compiled
也逐位相同；MLP suffix 的 hidden 83元素、gate-down 84元素、gate-up 2元素
出现微差。进一步使用实际 `[1,8292,12288]` BF16 SiLU 输入和真实 cotangent，
compiled 前向逐位相同，反向9个元素不同，relative-L2为 `1.6602951e-6`。

仅将本地 SiLU forward/backward 保留为 opaque `aten.silu` / `aten.silu_backward`
后，整个 block31 的16张量全部与 eager 逐位相同。该单变量干预支持 SiLU
编译边界是此 fixture 的充分修复点，尚未证明具体浮点指令或舍入边界。
独立 CPU profiler 记录实际 memory-efficient SDPA 的 ATen forward/backward；
CUPTI 在该卡不可用，不能声称 GPU kernel trace 验收。

局部报告为 `/tmp/qwen_nf4_real_strict_backward_stages_20261006/report.md`，
独立审查为 `/tmp/qwen_strict_silu_causal_review_20261006/review.md`。
局部大快照未持久化，独立复核限于源码、指标与小型 autograd/hook 契约；
source cotangent 来自旧 non-strict 全网，不能当作新 strict 全网 adjoint。

### 全32层真实两步 + 原生 SiLU：保存结果原门槛通过

新 helper `scripts/quantization/qwen_nf4_native_silu.py` 将32个 image MLP
SiLU 保留原生一阶 forward/backward；先验证所有目标再 patch，保持模块、
参数和 state_dict。它仅供诊断，不支持 double backward。两侧都 patch
原生 SiLU，保留64个 native image norms、strict、static、非 persistent
及前述真实 pair/flow loss/两步 SGD 条件；未改变阈值。

patch 后 eager 的2246张量与原 stable strict eager 全部逐位相同，初始672个
adapter与66个非packed冻结状态相同。新 compiled 的2246张量也与原 eager
全部逐位相同，原 close/relative-L2 门槛零失败，max abs/relative-L2全0。
每步448参数实际更新，loss为 `0.2706372738 → 0.2701036632`。
32个 compiled blocks各4次调用，观测到1个 graph、无 break；此整网路径
未传 `fullgraph=True`，不能把观测到无 break 描述为强制 fullgraph 配置。
两轮232个 packed/QuantState tensor bytes、对象身份和冻结检查通过。

原GPU脚本138.25秒完成两轮保存与数值比较后，reporter将布尔值误当字典，
exit `1`；原执行源码、日志和错误保留。修正版临时脚本未据此冒称运行成功。
CPU保存张量校验exit `0`，不同代理又独立重算完整2246张量、初始adapter及
每步448 SGD更新，均逐位一致。原始证据在
`output/tests/qwen_nf4_real_flow_native_silu_20261006/`，独立复算在
`/tmp/qwen_native_silu_full_independent_review_20261006/`。

原生 synthetic CLI 增加默认关闭的 `--native-silu` 和 `--deterministic`，
与原 `--native-layer-norm --no-dynamic --no-persistent-reductions` 组合使用；
该 CLI 仍使用合成输入/平方损失。入口恢复原确定性 flags及自动设置的
CUBLAS环境变量；workspace设置需要新进程，已初始化handle不保证受新值影响。
最终14项CPU契约测试、Ruff和Pyright通过。不同代理独立执行真实两步复跑，
自然exit `0`（158.04秒），新eager/compiled对旧stable eager/已保存compiled
及彼此的五组比较均2246/2246逐位相同，初始672项一致。32 blocks各4次调用，
1 graph、无break，原threshold全部误差0；每步448更新，66项冻结状态及232个
packed/QuantState字节与身份检查通过。

原生synthetic CLI也自然exit `0`：2247项及初始adapter全部逐位相同，32块
各5次调用、3graphs、无break。成功/失败、workspace原缺失/已有与提前import
failure共5项独立CPU环境恢复检查通过。原exit `1`证据保留，未改写其结果。
最终独立验收为 `/tmp/qwen_native_silu_clean_exit_final_acceptance_20261006/`，
`commands.json`、`real_gpu_verification.json`、`cli_verification.json`分别保存
精确命令、真实训练及合成CLI的结果。实际CLI两份`.pt`分别约839.8MB，不能
把逻辑tensor bytes估计当作文件实际占用；所有既有证据保留。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=<空闲GPU的UUID> \
  timeout 180 .venv/bin/python scripts/quantization/probe_qwen_nf4_compile.py \
  --model /path/qwen-nf4.safetensors --output-dir /tmp/qwen-nf4-native-silu \
  --count 24 --restore-mode foreach --scope all --no-dynamic \
  --no-persistent-reductions --native-layer-norm --native-silu --deterministic
```

生产gate保持，未将单pair/固定sigma/诊断adapter的一阶数值验收扩大为通用
训练支持。下一节追加生产attention-only与两个bucket的证据；其他optimizer、
resume、预览恢复及性能仍待验收。

### 生产 attention-only 与两个真实 bucket：独立正常退出通过

使用生产 `qwen_image_2_1_target_kwargs()`：128个 attention adapter、256个
trainable 参数、384项初始 state；排除 MLP/modulation，rank/alpha=16、
seed=41、dropout关闭。保留 factory 的 kaiming down / zero up 初始化，
未使用诊断随机 up。输入来自既有 Edit 缓存，不重建或缩放数据：

| 步骤 | bucket | target latent | reference latent | text | sigma / noise seed |
| --- | --- | --- | --- | --- | --- |
| 0 | 896×1184 | `[1,64,74,56]` | `[1,64,76,54]` | `[1,1070,4096]` | 0.5 / 171 |
| 1 | 1184×896 | `[1,64,56,74]` | `[1,64,52,78]` | `[1,1058,4096]` | 0.25 / 172 |

保留 strict deterministic、CUBLAS workspace、static、非 persistent reduction、
eager精度cast、native image LN/SiLU、full checkpoint、swap24/foreach、
scope=all 和真实 flow loss。两步 eager/compiled 的1286项全部逐位一致：
output/loss/input gradient各2项、adapter gradient512项、更新后state768项。
原 close/relative-L2 门槛未变，最大绝对误差与relative-L2均0。32 blocks各
4次调用，观测到2个graph、无break；未强制 `fullgraph=True`。

首步128个 down梯度为0，128个 up梯度非零且实际更新；第二步256个梯度均
finite非零，251个参数张量字节变化。其余5个down的实际FP32更新因舍入
没有字节变化；每个参数的实际结果都与 `before.add(grad, alpha=-.001)`
逐位相同。非零梯度不保证FP32参数字节必变，未以此放松数值比较门槛。
两轮232个packed权重/QuantState字节与身份、66个非packed冻结状态均通过。

原主对照完成后，可选compiled repeat在第二步被本任务时间预算中断，原命令
仍为exit `1` / failed，不能称完整PASS。不同代理的独立副本仅移除可选repeat
并更换输出目录，主对照自然exit `0`，GPU段148.10秒。该耗时复用编译缓存，
不作为冷编译或稳态性能结论，也不宣称可选repeat稳定性通过。

独立复算10个输入文件、14个输入tensor及两个fixture指纹；旧/新运行两侧
2572个结果hash一致，两目录各10对保存的小型张量均raw exact。完整张量在
CPU内存比较并留hash，只持久化selected小型张量；不能仅凭hash逆推旧运行
的全部SGD算术，正常退出的新运行逐参数检查了actual/expected原始字节。
独立报告为 `output/tests/qwen_nf4_attention_only_buckets_independent_20261006/report.md`，
JSON、命令、退出码、源码差异和指纹复算证据在同目录。原失败/中断证据保留于
`output/tests/qwen_nf4_attention_only_buckets_20261006/`。

本节证明所测两个真实bucket与生产初始化的诊断组合数值一致。当时生产训练
入口尚未接入联合策略；后续受限入口验收见下一节。默认NF4 compile gate保留。

### 默认关闭的实验入口：真实 train.py 两步通过

联合策略已通过 `qwen_nf4_compile_experiment=true` 接入 CLI、直接 training session、
bootstrap 和 WebUI 预检；Next 工作台在 Qwen Edit 的资源/编译分组显示默认关闭的
checkbox。配置错误为不可豁免错误，合法 NF4 header 与配置只提示实验 warning。
该开关与 `allow_unverified_nf4_compile` 不能同时启用。合同和启动例见
[实验说明](../experimental/qwen-nf4-compile.md)。

session 在 CUDA 初始化前设置 CUBLAS `:4096:8` 与严格确定性，退出恢复原状态；
已有不同 workspace、未设置 workspace 的已初始化 CUDA 或多进程会拒绝。
编译仍位于 adapter apply/load 和 checkpoint/swap 安装之后；先核验真实 232 个
packed BF16 NF4 Linear、32 blocks、128 个已 apply 的 attention-only plain LoRA
和 swap24/foreach/BF16，再共同验证并安装原生 image LayerNorm/SiLU。
static、scope=all、Inductor default，保留 eager cast，关闭 persistent reductions。
非实验 BF16 编译路径保留原行为，原诊断脚本通过兼容 shim 复用迁移后的算子。

不同代理以 `runpy` 执行真实 `train.py __main__`、真实 sys.argv/TOML/schema/bootstrap，
未手工安装诊断 patch、替换 optimizer 或绕过 guard。父 shell 显式提供 workspace；
自动设置 workspace 的路径有 CPU 顺序/恢复测试，尚无独立 fresh GPU launcher 实测。
使用原导入配置的临时副本，50 对 Edit 数据及既有缓存，数据集包含三个 bucket：
800×1344（1 对）、896×1184（48 对）、1184×896（1 对）；本次只训练两步。

| 验收项 | 结果 |
| --- | --- |
| 实际模型合同 | 232 NF4、32 blocks、swap24/foreach/BF16、128 个 rank/alpha16 plain attention adapters |
| 实际编译 | 32 blocks，128 次 compiled callable 调用；Dynamo captured calls 223、unique graphs 1，无 graph-break counter |
| 优化器 | 实际 AdamW，fused=true、lr=1e-4；每步 256 个 finite 梯度张量、256 个参数张量变化 |
| 首步/第二步非零梯度 | 128 / 256，保持生产 zero-up 初始化 |
| 保存 | 67,162,848 bytes，384 个 finite tensors，其中 256 个相对初始化变化，metadata steps=2 |
| 进程 | exit 0，80.91 秒，150 秒 timeout；退出后 90HX 17 MiB、无 compute 进程 |

这次真实入口 smoke 与前节逐张量 eager/compiled 对照是不同验收；本次没有另做
fresh 完整张量 A/B，也没有强制 `fullgraph=True`。耗时复用编译缓存，不作速度结论。
该轮未覆盖完整 optimizer/RNG resume 或训练预览；追加验收见下一节。
其他 optimizer、长训、画质、性能和跨硬件仍未验收。
checkpoint 的 `ss_model_family`/`ss_network_spec` 正确；通用 `modelspec.architecture`
仍沿用既有 `anima-preview/lora`。本仓 Qwen loader 使用前两项，外部兼容性尚未验收。

失败尝试保留：首次 launcher 有事件命名冲突；随后临时副本误设
`save_every_n_steps=0`，首步已更新但保存分支取模除零、exit 1。频率恢复原值 200
后第二次正常保存；没有修改生产保存代码，也没有覆盖原导入配置。
原比较门槛未放宽。CPU 合同测试 35 项通过；不同代理独立复测零 dropout 字符串与
非法 network_args 的 10 项边界，Ruff/Pyright/diff-check 通过。

证据（本地未入库产物）：真实入口报告 `output/tests/qwen_nf4_compile_train_entry_20261006/report.md`、
同目录 `attempt2-result.json`、日志、退出码和 checkpoint；
前端构建报告 `output/tests/qwen_nf4_compile_frontend_finish_20261006/report.md`
记录当时的摘要检查、TypeScript 与隔离 Vite 构建。随后
前端正式构建报告 `output/tests/qwen_nf4_compile_frontend_acceptance_20261006/report.md`
记录 checkbox 定向检查、273 项摘要、TypeScript 和本地正式构建通过，
`web/static/dragon-next` 已更新，旧资源与备份保留。未做浏览器验收或线上发布。

### 训练预览、完整状态保存与独立进程续训

不同代理在新的 `output/tests/qwen_nf4_compile_preview_resume_clean_exit_20261006/`
执行真实 `train.py` CLI。fresh 训练两步、每步一次 512×512 Euler1 预览、
完整 Accelerate 状态保存均完成，自然 exit 0，耗时 118.84 秒；新进程加载
该状态，从 step2 更新到 step3，再保存最终 adapter，自然 exit 0，64.86 秒。
两个父 shell 均未提供 workspace，CLI 自动设置 `:4096:8` 和严格确定性，
finally 恢复原环境；fresh 入口设置时 CUDA 尚未初始化。

| 验收项 | 结果 |
| --- | --- |
| 实际合同与编译 | packed NF4 232、blocks 32、plain attention LoRA 128、swap24/foreach/BF16；fresh compiled 调用 192 次 |
| 更新与预览 | 每步 256 个有限梯度张量、256 个参数张量变化；两次 preview latent `[1,64,32,32]` finite |
| 预览后恢复 | model/network training、checkpoint、swap、forward_only 与 32 个 compiled inner identity 全部恢复 |
| 图片 | 两张 PNG Pillow 完整解码；1024×540 RGB 参考图/标题拼图，生成区 512×512 |
| 完整状态 | 仅 5 个文件，201,608,480 bytes，包含 adapter/AdamW/scheduler/RNG，不保存冻结基座 |
| 新进程恢复 | adapter、完整 AdamW moments/step2、scheduler、Torch CPU/CUDA 与 NumPy RNG hash 匹配；step3 实际更新 |
| 最终权重与资源 | steps2/3 两份 checkpoint，各 384 个 finite tensors；结束后 90HX 17 MiB、0%、无 compute 进程 |

Python RNG 在 `random.setstate` 原函数返回时，原始 tuple 与保存值逐位相同
（hash 前缀 `a8c24890`）；`Accelerator.load_state` 返回时已为 `eee9bab7`。
独立 CPU 因果审查用同一状态文件精确复现：恢复后的两条 Rich 日志渲染
源文件超链接，`Style.copy` 调用全局 `random.randint`，14 次 randint 共消费
15 次 getrandbits(20)，MT index 440→455。禁日志或仅关闭 `enable_link_path`
均使返回 tuple 精确一致；状态快照与 digest 不消费 RNG。该日志消费是真实
状态变化，可能影响后续 Python 随机逻辑。生产恢复算法和全局日志均未修改。

此验收证明状态加载与续训生命周期；尚未做连续训练与恢复后的完整轨迹
逐位 A/B，不外推长训、画质、速度、其他 optimizer/adapter/shape 或硬件。
最终实际生产源码 SHA256 16/16 与本轮完成审计时一致。

首轮证据目录 `output/tests/qwen_nf4_compile_preview_resume_20261006/` 保留。
该轮真实预览和状态保存成功后，临时 observer 对 0D optimizer step 直接
`view(uint8)` 报错，训练进程 exit 1；修复临时哈希的 `reshape(-1)` 后，在新
目录完成正常退出复验，没有因此修改生产源码。

原始证据（本地未入库产物）：生命周期报告 `output/tests/qwen_nf4_compile_preview_resume_clean_exit_20261006/report.json`、
同目录 `preview-result.json`、`resume-result.json`、`artifacts-verification.json`、
`commands.txt`、`final-source-hashes.json`；
Python RNG 因果报告 `output/tests/qwen_nf4_compile_rng_review_20261006/report.md`。

主要原始证据（均只读模型、输出到新时间戳目录）：

- block0/5：`/tmp/qwen_nf4_block_precision_casts_20261005/20261005-213406-489658/result.json`。
- block19/31：`/tmp/qwen_nf4_block_precision_depth_20261005/20261005-214301-880664/result.json`。
- attention / MLP：`/tmp/qwen_nf4_stage_attention_20261005/20261005-220240-120351/result.json`、
  `/tmp/qwen_nf4_stage_mlp_20261005/20261005-220342-864432/result.json`。
- norm1 / norm2：`/tmp/qwen_nf4_stage_norm1_fp32_20261005/20261005-220717-752576/result.json`、
  `/tmp/qwen_nf4_stage_norm2_fp32_20261005/20261005-221353-293967/result.json`。
- BF16 cast 边界：`/tmp/qwen_nf4_norm_cast_boundary_20261005.json`、
  `/tmp/qwen_nf4_norm2_cast_boundary_20261005.json`。
- 两处原生 norm：`/tmp/qwen_nf4_block_native_norm_20261005/20261005-222456-268809/result.json`。
- 同求导条件 block0：`/tmp/qwen_nf4_block_frozen_mod_control_20261005/20261005-223512-049307/result.json`。
- 仅原生 norm2：`/tmp/qwen_nf4_block_native_norm2_20261005/20261005-224616-025342/result.json`。

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU的UUID> timeout 300 .venv/bin/python -m pytest -q -s \
  tests/test_qwen_nf4_compile_training.py

CUDA_VISIBLE_DEVICES=<空闲GPU的UUID> timeout 420 .venv/bin/python \
  scripts/quantization/probe_qwen_nf4_compile.py --model /path/qwen-nf4.safetensors \
  --count 24 --restore-mode foreach --scope all
```

probe 只读已有模型，在新的时间戳目录保存 `eager.pt`、`compiled.pt`、
`comparison.json`；成功与失败均保存 `result.json`，失败记录原 error 与已有产物。
普通 BF16 文件或不存在 packed NF4 模块会被明确拒绝。

本机忽略目录中的原始证据：

- 首轮默认编译：`output/tests/qwen_nf4_compile_20261005/tiny-20261005-163345/comparison.json`。
- 舍入选项单组：`output/tests/qwen_nf4_compile_20261005/casts-khxudb/pytest.log`。
- 8 组矩阵：`output/tests/qwen_nf4_compile_20261005/matrix-CGPI1W/pytest.log`。
- 完整模型：`output/tests/qwen_nf4_compile_20261005/20261005-164618-255076/comparison.json`。
- 独立验收日志：`output/tests/qwen_nf4_compile_20261005/independent_20261005_full/full.log`。

补强 NF4 身份、初始 LoRA 一致性、编译块调用计数及初始化清理检查后，
tiny swap2/slab/resident 定向重验通过（45.68秒）。共享 NF4 拒绝契约与既有
Qwen block swap/compile 定向回归24项通过；Ruff与Python语法检查通过。

后续应先隔离完整权重的前向/反向差异，再进行编译范围、attention 和
量化算子的定向对照；通过既定数值验收后再放行共享训练校验。
