# 扩散模型混合量化基座：实现、后训练与公开讨论

状态：调研记录 / 阶段快照
查阅日期：2026-10-02。依据论文、官方代码、维护文档和 GitHub 讨论；本轮未安装框架、下载模型或启动训练。性能数字均注明原始条件，不视为本地复现。

## 结论

扩散模型的量化基座微调、混合精度分配和量化误差补偿都有公开实现或论文依据。当前更常见的工程形式是大矩阵低位、敏感模块高精度，以及冻结量化基座训练浮点 LoRA。本轮未核到一个已经完整验收、可直接给 Anima 使用的“按重要性自动分配 BF16/FP8/NF4、兼容任意换层和反向”的训练器入口。

本地实验支持的结论是：当前 Anima MLP-only NF4/BF16、BF16 compute、swap12 的训练峰值收益很小，含采样的全程峰值没有收益。它不足以否定混合量化的整体价值，也没有比较每种精度各自优化后的 swap 数量与吞吐。

最有参考价值的路线有三条：MixDQ/ViDiT-Q 的实际量化敏感性测量；EfficientDM 的量化感知 adapter 与蒸馏；musubi ConvRot INT8 的真实低位前向加可微反向。三者分别解决“哪里量化”“怎样补偿误差”“怎样获得硬件收益”，不能互相替代。

补充核对：本地已经实现 ConvRot W8A16/W8A8，并有 RTX 3080 上 W8A8 慢于 W8A16 的实验记录。musubi 的参考价值在于具体 Triton kernel 与反向数值策略，不能视为本地缺失的新算法；差异、历史否决路径与证据限制见 [INT8 本地/线上专项对比](int8_training_local_vs_musubi_20261002.md)。

## 先区分四个概念

| 路线 | 实际行为 | 能否直接说明量化基座 LoRA 训练已支持 |
|---|---|---|
| PTQ，训练后量化 | 完整基座训练结束后进行校准、分解、量化参数重建 | 不能；通常目标是推理部署 |
| 冻结量化基座 + LoRA | 冻结压缩权重，训练浮点 adapter；仍需要基座输入梯度 dX | 可以，但要核对具体模型、后端、反向与换层路径 |
| QAT / 量化误差补偿后训练 | 训练过程中模拟量化，调整权重、scale 或 adapter，可带教师蒸馏 | 可以证明特定算法的训练流程；未必使用真实 packed 低位训练内核 |
| 量化模型加载已有 LoRA | 将已有 adapter 加到量化推理模块，或拼接到高精度低秩分支 | 不能；LoRA inference support 不等于 backward support |

PTQ 也可能包含梯度优化，例如 AdaRound 或 block reconstruction。是否调用 backward 不是唯一分类标准；优化的是量化参数、教师重建目标还是用户任务 adapter，需要分别说明。

## 当前训练器的实际实现

| 项目 | 本次核到的能力 | 混合粒度与限制 |
|---|---|---|
| SimpleTuner | NF4、FP8、Quanto/TorchAO 等量化基座训练路径；LoRA 时冻结基座 | 组件 precision 分别配置；Quanto 按模型排除敏感模块。组件混合不等于一个 DiT 内任意 BF16/FP8/NF4 映射。RamTorch 权重卸载与手动 Quanto/TorchAO 量化有兼容限制 |
| musubi-tuner Krea2 | scaled FP8 或 ConvRot INT8 基座 + LoRA；checkpoint、block swap | 固定 target/exclude：量化 `blocks.`，排除 modulation、norm、txtfusion。当前 Krea2 FP8 patch 显式 `use_scaled_mm=False` |
| AI Toolkit | 量化配置 include/exclude；可保留逐模块预量化 checkpoint 的混合后端 | loader 条件匹配才保留，否则可能反量化/重新量化。源码混合后端示例包含文本编码器；不能据此声称 Anima DiT NF4 三路混合已验证 |
| Diffusers + bitsandbytes | `Params4bit.from_prequantized`、NF4 加载、冻结基础权重和可训练 adapter 基础设施 | 量化文档的大量例子是推理；具体 LoRA 脚本是否接入、PEFT/Accelerate 是否支持必须分别查。8-bit optimizer 不等于量化基座 |

### SimpleTuner：低位权重与高精度岛

NF4 配置明确使用 `load_in_4bit=True`、double quant、`quant_type="nf4"`，compute dtype 另外指定。Quanto 分支对 Flux 等保留 norm、`proj_out`、`x_embedder`、`context_embedder`，再冻结基座。这是静态模块选择，不是根据训练更新热图自动决定位宽。

- [NF4 配置源码](https://github.com/bghira/SimpleTuner/blob/d6b5939104d7d09a08e6fe8a63a0f34407004d5c/simpletuner/helpers/training/quantisation/__init__.py#L127)
- [模型专属排除规则](https://github.com/bghira/SimpleTuner/blob/d6b5939104d7d09a08e6fe8a63a0f34407004d5c/simpletuner/helpers/training/quantisation/__init__.py#L475)
- [组件级 quantization_config](https://github.com/bghira/SimpleTuner/blob/d6b5939104d7d09a08e6fe8a63a0f34407004d5c/documentation/OPTIONS.md#L572)

### AI Toolkit：已有逐模块预量化混合保留

当前 loader 注释明确说明 checkpoint 可以 “deliberately mix backends”。请求的 backend 已在 shipped 集合且无额外转换要求时，保留原来的模块组合；否则按请求反量化或重新量化。这证明静态混合格式并非只能自己手写，但不证明所有量化类型和训练组合可互换。

- [预量化混合恢复与分支条件](https://github.com/ostris/ai-toolkit/blob/ecee894ed2b1f3716d9d7326693061ec1a3105bb/toolkit/models/v2/_mixin.py#L287)
- [冻结权重、输入梯度与反量化执行](https://github.com/ostris/ai-toolkit/blob/ecee894ed2b1f3716d9d7326693061ec1a3105bb/toolkit/util/ostris_quant.py#L66)

### 框架“支持量化”到“支持训练”中间仍有接口边界

TorchAO 的 `FqnToConfig` 已支持按完整模块名、正则与默认规则分配不同配置或 `None`。但所查 Diffusers TorchAO `is_trainable` 白名单仅包含若干 INT8/FP8 配置，INT4、NF4 和 `FqnToConfig` 本身不在名单中；PEFT TorchAO 的 `Int4Tensor` LoRA 路径还明确报“不支持，当前只支持 int8 weights”。这是包装层边界，不能扩大成“底层数学上不能训练”。

Quanto 有真实 autograd、include/exclude、QAT 和 freeze 路径；未 freeze 的 QAT 仍保留浮点底权，与只存 packed 权重的冻结基座训练不同。当前 Diffusers Quanto 文档已标记 deprecated、计划于1.0移除，Quanto 处于维护模式，新的长期架构不宜只依赖它。

- [TorchAO FqnToConfig](https://github.com/pytorch/ao/blob/a701b6a6058720c21f95908b7ae4a24bf0cae1b6/torchao/quantization/quant_api.py#L1565)
- [Diffusers TorchAO trainable 白名单](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/src/diffusers/quantizers/torchao/torchao_quantizer.py#L493)
- [PEFT INT4 类型限制](https://github.com/huggingface/peft/blob/532a05dd505c28993119b7715ee286f4234bf51b/src/peft/tuners/lora/torchao.py#L48)
- [Quanto 训练、freeze 与维护状态](https://github.com/huggingface/optimum-quanto/blob/5b94afb31bcdaf7e6acde1f4d88555d8e10bd85c/README.md)
- [Diffusers Quanto deprecated 文档](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/docs/source/en/quantization/quanto.md)

## 真正的扩散模型量化后训练

### EfficientDM：公开量化感知 LoRA + 蒸馏代码

[EfficientDM，ICLR 2024](https://arxiv.org/abs/2310.03270) 提出 QALoRA：量化感知的低秩适配，并将完整精度模型的去噪能力蒸馏给量化模型。官方代码真实包含训练循环。

普通冻结量化基座 LoRA 可概括为 `dequant(Q(W))x + BAx`，adapter 独立保持浮点。EfficientDM 的 `QuantModule_intnlora` 则先恢复基座、构造 `BA`，再将二者相加后联合假量化；部署目标是低秩更新可以合并并一起量化。训练还优化 weight delta 和时间步相关 activation delta/zero-point。教师和学生使用同一 `x,t,c`，对噪声预测做蒸馏 MSE。

它直接说明“补偿量化误差”可以是一项明确的后训练任务。普通画风/人物 LoRA 的目标是拟合用户数据，可能顺带补偿部分量化误差，但不能当作专门的量化补偿算法。

公开入口主要针对旧 LDM-4、ImageNet256、DDIM，固定rank32；不是现代 Anima DiT 即插即用代码。其假量化训练也不意味着真实 INT4 GEMM 或训练显存按4bit比例下降。

- [官方仓库与训练入口](https://github.com/ThisisBillhe/EfficientDM)
- [训练参数选择与首尾8bit](https://github.com/ThisisBillhe/EfficientDM/blob/main/quant_scripts/train_efficientdm.py#L98)
- [量化感知低秩层](https://github.com/ThisisBillhe/EfficientDM/blob/main/quant_scripts/quant_layer.py#L446)
- [教师/学生去噪训练循环](https://github.com/ThisisBillhe/EfficientDM/blob/main/ldm/models/diffusion/ddim.py#L383)

### BitsFusion：混合位宽后的两阶段训练，但没有公开实现

[BitsFusion，NeurIPS 2024](https://arxiv.org/abs/2406.04333) 在 SD1.5 UNet 上按层分配低位权重，平均约1.99bit。先通过完整精度教师做 CFG-aware 输出及中间特征蒸馏，再使用正常噪声预测目标微调；论文明确同时优化权重与 scaling factors。这是直接相关的“混合量化后再训练”，并非只做无训练校准。

资源条件也很重要：论文描述第一阶段8×A100、20K iterations、总batch256；第二阶段32×A100、50K iterations、总batch1024，使用内部数据。它证明方法存在，不能作为10GB单卡小规模画风LoRA的直接成本参考。

截至查阅日，[官方仓库](https://github.com/snap-research/BitsFusion) 仍写 “The code is under company review.”，不能列为已公开可复现的训练实现。

## 混合精度究竟如何分配：论文与实现

| 方法 | 真正混合/优化的单位 | 对 Anima 有用的启示 | 不应误读为 |
|---|---|---|---|
| [MixDQ，ECCV2024](https://arxiv.org/abs/2405.17873) | 逐Linear/Conv的W/A位宽；论文候选W2/4/8、A4/8；整数规划预算 | 分开衡量内容相关和质量相关层，单层量化敏感性→候选组合→实际生成筛选 | LoRA更新越小就越可量化；所有位宽都有对应实用kernel |
| [ViDiT-Q，ICLR2025](https://arxiv.org/abs/2406.02540) | layer/block、spatial/cross/temporal模块、timestep范围；动态细粒度activation量化 | 对不同模块使用适合的质量指标；论文把去噪分4段研究敏感性 | 一套固定早层/晚层名单适合所有DiT；NF4编码等同论文INT4 |
| [Q-DiT，CVPR2025](https://arxiv.org/abs/2406.17343) | 逐层量化group size/粒度搜索，主配置W4A8/W6A8 | 高敏感层也可缩小group，未必只能提高bit；用生成质量约束搜索 | 所有“mixed”都是bitwidth搜索 |
| [PTQD，NeurIPS2023](https://arxiv.org/abs/2305.10657) | 权重位宽固定，按timestep选activation位宽，配合误差修正 | 时间步敏感度需要覆盖；可研究activation精度而不复制多份权重 | 按timestep将NF4权重无损变回BF16 |
| [PTQ4DiT，NeurIPS2024](https://arxiv.org/abs/2405.16005) | channel salience balancing、跨timestep校准、layer/block重建 | 校准应覆盖扩散轨迹，PTQ允许量化参数重建训练 | 一个现成按层混合bit、可直接加速训练的内核包 |
| [Q-Diffusion，ICCV2023](https://arxiv.org/abs/2302.04304) | 跨timestep校准、UNet shortcut分拆、adaptive rounding | 不能仅用单个时间步或少量静态权重统计评估量化 | 原始模拟checkpoint等于真实INT4 GPU部署 |

MixDQ 官方搜索文档明确说：某一层单独量化后测 SQNR/SSIM 等指标；整数规划给出候选；并非每个规划配置都好，仍需实际生成筛选。其原方法不使用代码中可选的 LSQ-like training。ViDiT-Q 论文也强调仅 MSE 不能准确反映生成质量。

LoRA 热图衡量任务更新 `ΔW`；量化误差是 `E = dequant(Q(W)) - W`，对一层输出的直接影响是 `Ex`，还取决于输入激活、时间步、条件和下游传播。两者不是同一个测量对象。热图可缩小候选范围，不足以独自决定BF16/NF4分配。

当前 ViDiT-Q 默认分支是 `viditq`，已于2025/01公开 OpenSORA v1.2/PixArt-Sigma 的 CUDA kernel 与推理流程。旧main/旧README不代表当前发布状态。官方清楚区分 software simulation（仍在FP16计算，可能更慢）和hardware真实整数kernel。

- [MixDQ 搜索流程，固定源码快照](https://github.com/thu-nics/MixDQ/blob/4f6b32ad20d980494bcfecae50bfde5d17ed805c/mixed_precision_scripts/mixed_precision_search.md)
- [ViDiT-Q 当前实现与kernel](https://github.com/thu-nics/ViDiT-Q/tree/43657d11b32192ec0f50f0dac8d3a16c3d841196)
- [Q-DiT 实现](https://github.com/Juanerx/Q-DiT)
- [PTQ4DiT 实现](https://github.com/adreamwu/PTQ4DiT)
- [PTQD 实现](https://github.com/ziplab/PTQD)
- [Q-Diffusion 模拟量化说明](https://github.com/Xiuyu-Li/q-diffusion)

## Nunchaku/SVDQuant：高精度低秩旁路不是训练能力证明

SVDQuant 使用4bit权重与4bit激活残差主路径，并用16bit低秩分支承接难以量化的分量。它与本地NF4权重、BF16激活/运算的W4A16路径不同。

官方 Nunchaku LoRA 文档写的是把已有用户 LoRA 权重直接拼接到高精度低秩分支，供推理使用。SVD补偿分支、用户任务LoRA和量化误差补偿训练是三件相关但不同的事情。

查到的公开 issue #609（2025-08-15）请求 backward pass，#857（2025-12-31）请求 QeRL/backward；本轮未找到正式完成的官方训练反向流程。issue关闭不能解释为功能实现。精确搜索的官方 `svdq-finetune` 来源没有核实成功，不能将其当作已发布产品。

Nunchaku安装文档列出sm86支持，但那是对应INT4推理kernel的支持，不代表NVFP4或训练backward可用。权重格式、packing、GPU指令和kernel必须共同匹配。

- [Nunchaku方法与定位](https://github.com/nunchux-ai/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/README.md)
- [LoRA直接拼接文档](https://github.com/nunchux-ai/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/docs/source/usage/lora.rst)
- [安装/GPU支持](https://github.com/nunchux-ai/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/docs/source/installation/installation.rst)
- [Backward请求 #609](https://github.com/nunchux-ai/nunchaku/issues/609)、[RFC #857](https://github.com/nunchux-ai/nunchaku/issues/857)
- [DeepCompressor PTQ流程](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/README.md)

## 更贴近sm86训练的实现：ConvRot INT8

musubi [PR #1008](https://github.com/kohya-ss/musubi-tuner/pull/1008) 于2026-07-21创建、08-06合并，明确针对 Krea2 冻结基座 LoRA。它可供对比本地已有 ConvRot 路径的算子与数值策略：

- 离线旋转权重并按输出通道量化为INT8，保存scale；前向在线旋转激活、动态逐行量化，执行INT8 GEMM。
- 默认反向临时恢复权重到梯度dtype，计算dX；也提供INT8反向选项，但增加梯度量化误差与临时转置开销。
- LoRA保持独立浮点训练分支；基座与scale冻结。它不是EfficientDM的联合量化adapter，也没有自动教师补偿目标。
- 指定源码的旋转、激活量化、GEMM是多个步骤；融合的是GEMM的反量化/bias epilogue。PR文字描述比代码宽，不能说整条路径只有一个融合kernel。
- 没有Triton则退回临时反量化浮点矩阵乘，只保留存储节省；INT8反向要求CUDA+Triton。

PR作者给出的1024²、batch1、Flash、checkpoint训练数据：RTX3090为FP8 7.1秒/步、ConvRot INT8 5.3秒/步；Blackwell为BF16 2.0、FP8 2.3、INT8 1.9秒/步。这是特定Krea2实现的作者报告，不是本地独立复现或质量认证。

前向/输入梯度测试对照主要是“量化权重的反量化参考”，并非原始BF16模型；另有原始权重roundtrip测试。相关阈值及外部固定grad_out的LoRA梯度测试只证明局部契约；将其kernel或反向策略接入本地Anima时，仍需重新验收完整训练轨迹与成图质量。

CMP90HX为sm86，不应以原生FP8 Tensor Core作为方案前提。真实INT8计算更契合该代硬件，但仍应分别评估旋转、量化、反向反量化和换层的总成本。加入FP8存储格式本身没有这样的算力收益保证。

- [Krea2目标层/排除层和FP8 use_scaled_mm=False](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/krea2/krea2_utils.py#L23)
- [ConvRot训练autograd](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/modules/convrot_int8_utils.py#L315)
- [实际多步骤kernel调用](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/modules/convrot_int8_kernels.py#L447)
- [局部数值测试](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/tests/test_krea2_convrot_int8.py#L130)

## 公开讨论揭示的工程问题

| 来源 | 原讨论内容 | 解读边界 |
|---|---|---|
| [SimpleTuner #737，2024-08](https://github.com/bghira/SimpleTuner/issues/737) | NF4支持需训练器依赖链共同兼容，推理支持不能代替训练支持 | 是历史状态，不能用来否定今日源码已有NF4路径 |
| [SimpleTuner #1149，2024-11](https://github.com/bghira/SimpleTuner/issues/1149) | CPU量化更慢，但可避免GPU加载量化时OOM | 加载峰值、训练峰值、量化文件大小必须分别计量 |
| [musubi #715，2025-11](https://github.com/kohya-ss/musubi-tuner/issues/715) | adapter参数dtype、优化器会造成不同训练器的显存差；帖子中关于OneTrainer的猜测被参与者纠正 | 不能把跨框架显存差全归因于FP8/NF4 |
| [Nunchaku #609，2025-08](https://github.com/nunchux-ai/nunchaku/issues/609) | 发帖者正是因现有量化训练每次转回高精度，希望有原生低位backward | 直接呼应当前瓶颈，但feature request不是实现承诺 |
| [musubi #1008，2026-07/08](https://github.com/kohya-ss/musubi-tuner/pull/1008) | 提供真实INT8 forward与可选低位backward的LoRA训练路径 | 已合并源码强于提案，但性能/质量仍限原验证条件 |

## 为什么本地实验没有达到论文中的显存收益

训练峰值近似由时间轴上的最大同时存活集合决定：

`peak = max_t(resident base weights + live activations + adapter/gradient/optimizer + operator temporaries + swap staging)`。

文件大小只对应基座存储。换层已将部分权重移到CPU，量化和offload的收益可能重叠；训练输入梯度、checkpoint重算、反量化与稳定换层缓冲仍存在。混合槽需要容纳该槽最大的表示，不能按压缩文件大小预测buffer体积。

本地修复后的短测：步末2.2727GiB，对照BF16为2.8464GiB；首次采样前backward峰值4.0231GiB，对照4.0186GiB；含采样峰值5.0513GiB，对照4.6998GiB。它说明当前实现真正省了部分驻留权重，但未改变峰值主导项。

MixDQ/ViDiT-Q/Nunchaku的许多优势来自激活量化、专用kernel、融合操作和推理没有训练反向；不能用其W4A4/W4A8推理数字推断本地NF4+BFloat16训练收益。例如MixDQ官方SDXL-Turbo UNet W8A8在RTX4080报告1.45×推理速度、约1.99×峰值内存改善，既不是Anima，也不是训练。

本地证据：[异型换层与显存报告](anima_mixed_nf4_swap_memory_90hx_20261002.md)。

## 对 Anima 的下一步建议

1. **先以相同显存预算比较完整配置。** 当前固定swap12对照只回答该配置下的差异。应比较BF16与量化各自可用的swap数量、吞吐、训练峰值和采样峰值；更低swap若无益，则不继续扩大实现复杂度。需要独立短测，不是线上论文已经证明。
2. **量化策略按实测扰动和实际驻留收益分配。** LoRA热图先选候选；对具体Linear进行单模块量化，跨多个timestep/条件收集预测扰动，再测少量组合。预算应包含换层槽的最大表示与峰值时刻，而不是仅最小化文件字节数。
3. **sm86上的INT8研究应从本地已有ConvRot及失败记录出发。** 本地W8A8已有真实INT8前向，默认保留FP32取码与STE反向。musubi的Triton量化/融合后处理可作为差异研究；默认BF16反向及不同取码规则则是新的数值实验。先隔离kernel和精度变量，再验证输入梯度、adapter梯度、checkpoint、swap与真实采样质量，不应重新实现整套ConvRot或承诺超过W8A16。详见[专项对比](int8_training_local_vs_musubi_20261002.md)。
4. **质量保持确有需求时再做量化补偿训练。** 借鉴EfficientDM，定义完整精度teacher到量化student的预测/轨迹目标，区分补偿adapter与任务adapter；BitsFusion的大规模成本不能直接压缩成“再训几百步就能恢复”。
5. **保留BF16作为当前默认与teacher。** 当前实验尚未证明量化能改善用户实际受限的峰值，也未进行长期质量验收。保留实验artifact和可重复证据，比把混合量化判为普遍无用或直接默认推广更符合现有数据。

## 证据快照

SimpleTuner `d6b5939`（2026-10-01）、musubi `f8a1b03`（2026-09-27）、AI Toolkit `ecee894`（2026-09-27）、Diffusers `578c9b2`（2026-10-01）；其他关键源码链接也尽可能固定到所读commit。EfficientDM等使用主分支链接的项目以2026-10-02读取状态为准，未来可能变化。

关键原文与GitHub API响应存于本机 `output/online-diffusion-mixed-quant-20261002/`，包含来源URL、HTTP读取结果、行号摘录、Nunchaku讨论、musubi合并状态与论文代码分支SHA。讨论中的参与者意见、PR作者测量和论文主张分别保留其证据等级，没有作为本机热测结论。
