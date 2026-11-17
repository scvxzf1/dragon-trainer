# 公开训练器 DDP / TP / PP 实现调研

状态：2026-10-04 固定版本源码调研；未运行外部训练器或多卡性能测试。
范围：图像/视频 LoRA 训练器，以及具备真实 TP/PP 训练路径的通用框架。

## 结论

本项目可以采用共同的多卡拓扑、数据协议和保存协议，面向 Anima、Krea-2、Z-Image、
Qwen-Image 2.1 全模型开发；模型族分别提供自己的 TP 分片计划与 PP 阶段接口。

- DDP：sd-scripts / Musubi 与本项目的 Accelerate + adapter 手动梯度平均最接近；
  OneTrainer 的初始化广播和全局 bucket 后按 rank 分样本值得借鉴。
- PP：diffusion-pipe 有实际接入训练循环的 DeepSpeed PipelineEngine、阶段参数归属、
  1F1B、动态 shape 和 adapter 聚合保存，是最接近本项目的参考。
  其当前源码已有 Krea-2、Z-Image、Qwen-Image 2.1 分层适配。
- TP：TorchTitan 的同步 TP + LoRA 分片，以及 Megatron 的行/列并行、head 切分和
  process groups 是主要参考；不能直接把它们的 LLM 模型实现套到 DiT。
- 推荐共同设计 DDP/TP/PP 接口，先闭环 PP 生产训练，同时用同步 TP + plain LoRA
  建立算子等价性验证。TP 的通信重叠和混合并行建立在这两条路径验证之后。

这里的“未找到”仅表示本次固定源码及调用链中未找到实际训练接入，不代表作者永远不支持。

## 固定源码版本

| 项目 | 默认分支快照 SHA | commit UTC 日期 |
| --- | --- | --- |
| kohya-ss/sd-scripts | `690ea7f96c23182352ec63def76d431c6120bd2f` | 2026-09-24 |
| kohya-ss/musubi-tuner | `f8a1b03794a49239a3539015075f5123d6c07d66` | 2026-09-27 |
| Nerogar/OneTrainer | `23df3832e8f213d64c337e39f9365ba46e99fed0` | 2026-08-19 |
| tdrussell/diffusion-pipe | `334106c2d0e29131b53d504b80e211942dac147e` | 2026-09-28 |
| pytorch/torchtitan | `e3929368edd2f2457456913890ba2b4c0f27da31` | 2026-10-04 |
| NVIDIA/Megatron-LM | `ef8cd10b8e34a94c9ad7452e7851f58337f353a0` | 2026-10-03 |

diffusion-pipe 的 `requirements.txt` 锁定 DeepSpeed 0.18.4；下文 DeepSpeed 源码来自
对应 tag 的 commit `b35d9eb01bc04e774cc05dc43713f2a41423da5c`。
证据链接均使用完整 SHA，避免默认分支更新使行号失效。

## 实际能力对照

| 项目 | 数据并行 | 训练 TP | 训练 PP | 其他分片能力 |
| --- | --- | --- | --- | --- |
| sd-scripts | Accelerate；adapter 显式 grad mean | 未找到 | 未找到 | 部分脚本有 DeepSpeed ZeRO 0–3；不可外推全部模型族 |
| Musubi | Accelerate；adapter 显式 grad mean | 未找到训练接入 | 未找到 | 源码说明当前不支持 DeepSpeed；未找到专用插件接入；FSDP 注释不足以证明接入 |
| OneTrainer | 自管进程组和 grad mean | 未找到 | 未找到 | 未找到实际 FSDP wrapper；配置字段不能作为证据 |
| diffusion-pipe | 同一 PP stage 的 DP 副本同步 | 未接入 model/TP 轴 | DeepSpeed，真实 DP+PP | 默认无 ZeRO；此 PipelineEngine 拒绝 ZeRO-2/3 |
| TorchTitan | 复制 DP / FSDP 分片 DP | 同步行/列 TP、LoRA 参数分片 | PyTorch PipelineStage + schedule | FSDP、SP、CP 等；不同模型有不同 sharding plan |
| Megatron-LM | 独立 DP process groups | 行/列 TP、head 切分、SP、通信重叠 | 1F1B、交错 PP | distributed optimizer / 参数分片等独立能力 |

FSDP/ZeRO 是分片数据并行：执行一个 unit 时通常仍要集齐该 unit 的参数；TP 则直接执行
分片矩阵乘。CP/SP 是 token/激活布局相关机制，不能由“sequence”字段推断成 TP。
Musubi 的 generation helper 出现 `tp_mesh`，也不能据此宣称训练 TP 已接入。

## DDP：adapter 梯度与数据协议

sd-scripts 在 `train_network.py::all_reduce_network()` 中逐参数执行
`accelerator.reduce(param.grad, reduction="mean")`；`accelerator.sync_gradients` 为真时
调用它。Musubi 的 `trainer_base.py` 采用相同更新边界，并检查 distributed type。
两者 LoRA 通过替换基座 Linear 的 forward 生效。[梯度函数][sd-grad]、[更新边界][sd-loop]、
[Musubi 更新边界][musubi-grad]。

因此“用了 accelerator.prepare”不意味着 monkey-patched adapter 的归约 ownership 已解决。
本项目未来增加 TP/PP/FSDP 后，必须明确每个参数由哪个 group、在哪个累积边界归约一次。
尤其需要核验 `grad=None` 的 rank 分歧，防止某个 rank 跳过 collective 而其他 rank 等待。

OneTrainer 由 `MultiTrainer` spawn worker 并初始化 process group，更新边界进行
SUM/world-size 梯度平均，支持可选异步归约。它在创建 optimizer 前广播参数，注释明确说明
随机 LoRA 初始化可能不同；数据先按 `batch_size × world_size` 整理同 aspect bucket，
再通过 DistributedSampler 分配。[进程启动][ot-launch]、[初始化广播][ot-init]、
[全局 bucket][ot-bucket]、[归约][ot-grad]。

DDP 可以让不同 rank 处理不同样本；TP/PP rank 共同处理同一份模型副本的样本。
将后者接入时，noise、timestep、caption dropout、样本顺序和 bucket 必须在对应模型并行组
保持一致。不能继续把所有 world ranks 当作独立的数据 rank。

sd-scripts 还有 DeepSpeedPlugin 的实际 prepare 路径，支持配置 ZeRO stage 0–3；
Musubi 的 `prepare_accelerator()` 文档字符串写明
`DeepSpeed is not supported in this script currently.`；本次未找到专用 DeepSpeed 插件训练接入，
该说明并非运行时拒绝检查。
[sd-scripts ZeRO][sd-zero]、[Musubi 边界][musubi-zero]。ZeRO-1/2 主要节省 optimizer/gradient；
冻结基座 + LoRA 时这两部分较小，不能把它们当成解决巨大 DiT 权重容量的主要办法。

## PP：diffusion-pipe 的完整训练路径

### 拓扑与参数归属

`train.py` 加载 diffusion model、注入/加载 adapter 后调用 `to_layers()`，再构造
`ManualPipelineModule`，传入 `num_stages`、分层方式、loss、`dynamic_shape=True`。
它通过 `deepspeed.initialize()` 和 `model_engine.train_batch()` 真正运行训练。
只提供 PP stage 数时，DeepSpeed 自动构建 DP+PP 拓扑：`world_size = DP × PP`。
[构造与启动][dp-engine]、[调用主循环][dp-loop]、[拓扑][ds-topology]。

DeepSpeed 只注册和搬到 GPU 当前 stage 的模块；optimizer 从当前 pipeline module 的
trainable parameters 创建。没有 trainable 参数的 stage 有 DummyOptimizer 路径。
这与借用整模型 block 的 probe wrapper 有实质区别：生产 PP 必须分离参数与 optimizer 归属。
[stage 构建][ds-stage]。

diffusion-pipe 每个 rank 最初仍构造完整模型及 adapter，再筛选 stage-local GPU 模块；
它没有因此获得分片磁盘加载或低 CPU 内存初始化。全模型初始化方式不宜照搬。

### 各模型族的阶段接口

Krea-2 使用 `InitialLayer → TransformerLayer × blocks → FinalLayer`；中间传递 hidden、
timestep/modulation、freqs、mask、sizes 的 Tensor tuple。Qwen-Image 2.1 的 tuple 还包含
编辑输入所需的 segment/conditioning 信息；Z-Image 按 `diffusion_model.layers` 分层。
本次未找到其 Anima adapter。[Krea-2][dp-krea]、[Qwen-Image 2.1][dp-qwen]、[Z-Image][dp-z]。

初层负责 embedding/conditioning，FinalLayer 负责输出投影与还原空间结构；
末 stage 的 PipelineEngine 再调用独立的 `loss_fn` 计算 loss。
不是只把 block 列表平均分成两段就结束；各阶段接口要包含 backward 和 loss 所需信息。

### 分层、调度与动态 shape

默认按全部参数量分层。项目专门修改 DeepSpeed 参数计数，包含冻结基座，避免只数 LoRA
参数造成错误平衡；`partition_method="manual"` 与 `partition_split` 可手工指定非均匀边界，
用于显存容量不同的卡。[全部参数计数][dp-count]、[手工分层][dp-partition]。
代码提供手工控制；本次未找到自动显存/吞吐 profile 或异构性能验证。

标准 TrainSchedule 交错 forward/backward，最后归约 DP 梯度并更新 optimizer；
microbatch 数来自 gradient accumulation steps。[调度][ds-schedule]。

```text
global_batch = micro_batch_size × accumulation_steps × DP_degree
```

PP_degree 和 TP_degree 不能再次乘进有效 batch。microbatch=1 时两阶段只能先后执行，
主要收益是容纳模型；增加 microbatches 才能改善流水线吞吐，也会改变有效 batch。

数据层先组成同 bucket 的 `micro_batch × GAS × DP` 大批次，再分 DP 和 microbatches。
下一步可换 bucket；每步 `reset_activation_shape()`，配合动态 tensor 元数据通信。
依赖 noise 的 target 通过首末 stage 的 send/recv 同步，并在 schedule 前预加载，
避免与 PP P2P 顺序冲突。
[bucket][dp-bucket]、[target 传递][dp-target]、[shape reset][dp-loop]。

借鉴动态 batch 组织时必须保留本项目的模型族不变量：特别是 Anima 的 max-padded text
不能改成按文本长度裁尾；Qwen 专属 latent/编辑 conditioning 也不能借用其他族 schema。

### 导出与恢复

LoRA 导出与训练恢复是两条独立路径。导出使用 DP rank0 的各 PP stage，按 `original_name`
写各自 trainable state，barrier 后由 stage0 CPU 合并并输出完整 adapter。
训练 checkpoint 调 `engine.save_checkpoint(exclude_frozen_parameters=True)`，保存训练参数、
optimizer/scheduler 和 dataloader 游标等；恢复先重建基座与 adapter，再 load checkpoint。
[保存][dp-save]、[恢复][dp-resume]。

本次未找到“更换 PP/DP degree 后 optimizer/RNG/数据位置完全等价恢复”的实测证据，
不能由存在 load_checkpoint 推导该保证。

DeepSpeed 0.18.4 此 PipelineEngine 明确拒绝 ZeRO-2/3；diffusion-pipe block swap 要求
`pipeline_stages==1`。这说明组合需要实现契约，单个引擎支持 PP 不等于支持任意组合。
[ZeRO 边界][ds-zero]、[swap 边界][dp-swap]。
其 `gradient_release` 还有作者明确指出不满足标准梯度累积数学语义的特殊路径，
不应作为标准 PP 的参考。[gradient release][dp-release]。

## TP：TorchTitan / Megatron 的分片与 LoRA

TorchTitan 构建 DeviceMesh，将 DP replicate、DP shard、CP、TP、PP 分成明确轴。
Megatron 使用对应独立 process groups。移植基础 TP/PP 时可先取 `world=DP×TP×PP`，
需要 FSDP/CP 时再扩展轴；不能让现有 world-wide adapter reduce 混合这些轴。
[TorchTitan mesh][tt-mesh]、[Megatron groups][mg-groups]。

Attention 的 QKV 是输出维度/head 分片，output projection 是输入维度分片；
MLP 的扩张层分输出，收缩层分输入。PyTorch 权重 `[out,in]` 对应 column 的 axis0、
row 的 axis1。必须同时修改 local head count、reshape、fused QKV 布局和 attention 后端。
[TorchTitan 分片计划][tt-plan]、[Megatron 层][mg-tp]。

非 SP row projection 对 partial output 做 all-reduce；SP 路径在 column 输入 all-gather，
row 输出 reduce-scatter，并在 norm 等位置保存 token 分片。Megatron 还提供 input-gradient
collective 与 weight-gradient GEMM 的通信重叠。
[column 通信][tt-collectives]、[row 通信][tt-row-collectives]、[梯度通信重叠][mg-overlap]。
SP 的布局重分配不能被实现成删除 Anima 文本 padding。

TorchTitan 的 LoRA `_adapter_sharding()` 给出了可直接参考的参数规则：

| 基座 Linear | LoRA A `[r,in]` | LoRA B `[out,r]` |
| --- | --- | --- |
| column/output 分片 | 复制 | output 分片 |
| row/input 分片 | input 分片 | 复制 |

adapter 更新先加入本地 Linear 输出，TP collective 仍由外层基座 projection 负责。
[LoRA 实现][tt-lora]。

以下是接入本项目时的数学推导：复制 A/B 的梯度需要聚合不同 TP shard 的贡献；
按上述布局通常是 TP SUM，而数据副本之间才按 DP mean。同一梯度不得被 adapter hook、
DTensor/FSDP 和手工 reduce 重复归约。冻结基座依然需要传播 input gradient。

TorchTitan 明确声明 async TP transform 与 LoRA transform 冲突，因为融合 kernel 绕过
`_linear` 会漏算 adapter。[明确冲突][tt-async]。
所以优先参考同步 TP；编译、量化或融合路径必须验证 adapter 仍在实际计算图中。

TorchTitan 的 Flux sharding 文件当前显式使用 DP/CP，不足以证明其 Flux 或本项目四族已有
TP/PP 验证。[Flux 布局][tt-flux]。LLM 的 TP/PP runtime 是框架参考，模型适配仍需自己实现。

PP optimizer 的参数组在不同 rank 指向不同层，TorchTitan DCP 因此按完整参数名 flatten
optimizer state；Megatron checkpoint 携带 shard 轴/全局 tensor 区域元数据。
供单卡推理使用的 adapter 要重新拼接 TP shards，训练恢复则需要布局与 optimizer 状态。
[DCP 参数名][tt-dcp]。

## 本项目全模型接入建议

1. 公共拓扑：建立 DP/TP/PP groups，统一 GPU 白名单、world-size、数据 rank 与 batch 计算。
   单卡等价于所有 degree=1，不为不同模型族复制 launch 主循环。
2. 模型适配：四个模型族都注册 PP 初层/中间层/末层的 Tensor schema，以及 TP
   attention/MLP/head/normalization 计划。模型范围面向全部已注册族，逐族补齐并验收。
3. 参数归属：apply/load adapter 在分片执行前完成；每 rank 的基座、adapter、optimizer
   只包含实际负责部分。DDP、TP、PP 采用各自的梯度同步规则。
4. 数据和条件：同一个 TP/PP 模型副本共享样本、noise、timestep、bucket。
   1F1B 同时保留多个 microbatch，FEI/router 等共享可变状态必须有 microbatch 对应的
   生命周期，不能由下一次 forward 覆盖尚未完成 backward 的条件。此项是设计风险，
   尚未通过本次调研证明本项目存在具体 bug。
5. 保存：保持全局参数名，区分逐 rank 的完整训练恢复与合并 adapter 导出，记录 topology、
   shard/range、optimizer、RNG 和数据位置。首先验证同 topology 恢复，再研究重分片恢复。
6. 验收：四族分别对照单卡 forward、loss、adapter gradient 和 optimizer update；再测
   双卡 P2P/collectives、不同 bucket、保存重载、resume 首步、真实显存峰值与持续吞吐。
   adapter 变体通过同一接口扩展，但 plain LoRA 的通过不能外推所有变体。

选择 PP 优先闭环的依据是其跨 stage 边界通信方式和 diffusion-pipe 的完整扩散实现。
TP 每 block 都会同步激活，是否能更快必须在目标互联上实测；高速互联是优势而非硬性
型号要求。多个 microbatch 能否接受、单层是否能放入一张卡，也会改变选择。

## 本项目历史证据

此前“只有 DDP”的准确范围是生产入口。本项目已有独立 Anima PP2/TP2 探针，
见 [2026-09-04 双卡报告](anima_dual_gpu_parallel_probe_20260904.md) 和
[PP 阶段规划（已归档）](../../_archive/docs/proposal/krea2_pipeline_parallel.md)。历史探针使用 CMP 170HX 64GB +
RTX 3080 10GB、PHB、无 CUDA P2P：PP2 为单 microbatch fill-drain，并非真实 1F1B；
TP2 有大量激活通信；INT8 传输压缩未带来实质端到端加速。
这些是特定硬件、八步 smoke 的结果，不能作为当前 GTX 1050/CMP 90HX 或同构双卡的性能预测。

本次只修改调研文档，不放行生产 TP/PP，也未执行训练或修改 GPU 设置。

## 固定版本源码证据

[sd-grad]: https://github.com/kohya-ss/sd-scripts/blob/690ea7f96c23182352ec63def76d431c6120bd2f/train_network.py#L225-L228
[sd-loop]: https://github.com/kohya-ss/sd-scripts/blob/690ea7f96c23182352ec63def76d431c6120bd2f/train_network.py#L1617-L1622
[sd-zero]: https://github.com/kohya-ss/sd-scripts/blob/690ea7f96c23182352ec63def76d431c6120bd2f/library/deepspeed_utils.py#L19-L84
[musubi-grad]: https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/training/trainer_base.py#L2201-L2207
[musubi-zero]: https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/training/accelerator_setup.py#L72
[ot-launch]: https://github.com/Nerogar/OneTrainer/blob/23df3832e8f213d64c337e39f9365ba46e99fed0/modules/trainer/MultiTrainer.py#L38-L89
[ot-init]: https://github.com/Nerogar/OneTrainer/blob/23df3832e8f213d64c337e39f9365ba46e99fed0/modules/util/optimizer_util.py#L57
[ot-bucket]: https://github.com/Nerogar/OneTrainer/blob/23df3832e8f213d64c337e39f9365ba46e99fed0/modules/dataLoader/mixin/DataLoaderText2ImageMixin.py#L300
[ot-grad]: https://github.com/Nerogar/OneTrainer/blob/23df3832e8f213d64c337e39f9365ba46e99fed0/modules/util/multi_gpu_util.py#L65-L87
[dp-engine]: https://github.com/tdrussell/diffusion-pipe/blob/334106c2d0e29131b53d504b80e211942dac147e/train.py#L608-L635
[dp-loop]: https://github.com/tdrussell/diffusion-pipe/blob/334106c2d0e29131b53d504b80e211942dac147e/train.py#L917-L924
[ds-topology]: https://github.com/deepspeedai/DeepSpeed/blob/b35d9eb01bc04e774cc05dc43713f2a41423da5c/deepspeed/runtime/pipe/module.py#L169-L192
[ds-stage]: https://github.com/deepspeedai/DeepSpeed/blob/b35d9eb01bc04e774cc05dc43713f2a41423da5c/deepspeed/runtime/pipe/module.py#L232-L289
[dp-krea]: https://github.com/tdrussell/diffusion-pipe/blob/334106c2d0e29131b53d504b80e211942dac147e/models/krea2.py#L52-L58
[dp-qwen]: https://github.com/tdrussell/diffusion-pipe/blob/334106c2d0e29131b53d504b80e211942dac147e/models/qwen_image21.py#L51-L62
[dp-z]: https://github.com/tdrussell/diffusion-pipe/blob/334106c2d0e29131b53d504b80e211942dac147e/models/z_image.py#L30-L36
[dp-count]: https://github.com/tdrussell/diffusion-pipe/blob/334106c2d0e29131b53d504b80e211942dac147e/train.py#L79-L90
[dp-partition]: https://github.com/tdrussell/diffusion-pipe/blob/334106c2d0e29131b53d504b80e211942dac147e/utils/pipeline.py#L7-L53
[ds-schedule]: https://github.com/deepspeedai/DeepSpeed/blob/b35d9eb01bc04e774cc05dc43713f2a41423da5c/deepspeed/runtime/pipe/schedule.py#L189-L256
[dp-bucket]: https://github.com/tdrussell/diffusion-pipe/blob/334106c2d0e29131b53d504b80e211942dac147e/utils/dataset.py#L347-L389
[dp-target]: https://github.com/tdrussell/diffusion-pipe/blob/334106c2d0e29131b53d504b80e211942dac147e/utils/dataset.py#L1387-L1405
[dp-save]: https://github.com/tdrussell/diffusion-pipe/blob/334106c2d0e29131b53d504b80e211942dac147e/utils/saver.py#L58-L128
[dp-resume]: https://github.com/tdrussell/diffusion-pipe/blob/334106c2d0e29131b53d504b80e211942dac147e/train.py#L871-L893
[ds-zero]: https://github.com/deepspeedai/DeepSpeed/blob/b35d9eb01bc04e774cc05dc43713f2a41423da5c/deepspeed/runtime/pipe/engine.py#L72-L80
[dp-swap]: https://github.com/tdrussell/diffusion-pipe/blob/334106c2d0e29131b53d504b80e211942dac147e/train.py#L578-L586
[dp-release]: https://github.com/tdrussell/diffusion-pipe/blob/334106c2d0e29131b53d504b80e211942dac147e/train.py#L713-L759
[tt-mesh]: https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/torchtitan/distributed/parallelism_context.py#L145-L178
[mg-groups]: https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/megatron/core/parallel_state.py#L849-L859
[tt-plan]: https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/torchtitan/models/common/decoder_sharding.py#L243-L298
[mg-tp]: https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/megatron/core/tensor_parallel/layers.py#L1044-L1090
[tt-collectives]: https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/torchtitan/models/common/linear.py#L154-L178
[tt-row-collectives]: https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/torchtitan/models/common/linear.py#L229-L277
[mg-overlap]: https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/megatron/core/tensor_parallel/layers.py#L795-L825
[tt-lora]: https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/torchtitan/models/common/lora.py#L93-L129
[tt-async]: https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/torchtitan/config/transform/async_tensor_parallel.py#L83-L87
[tt-flux]: https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/torchtitan/models/flux/sharding.py#L20-L71
[tt-dcp]: https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/torchtitan/components/checkpointer/dcp.py#L96-L117
