# Qwen Image 2.1 / CMP 90HX 结构性优化审阅

日期：2026-09-27
状态：半活跃（2026-09-28 更新）——方案 1（INT8 packed block-swap）已实现并完成 INT8/BF16 各 60-step 首轮热测；观察到约 3.69% 单轮步时改善，但未达到推荐门槛，A/B/A、模型质量和续训验收仍待完成。方案 3 的 MLP/QKV/重算策略已集成，独立小模型验证与完整模型短测通过；续训首步梯度差异在确定性诊断中消失，尚未做性能验收，详见[阶段 3 开发与验收（已归档）](../../_archive/docs/proposal/qwen_image_2_1_training_block_stage3_20260927.md)。方案 2、4 尚未实施。
范围：Qwen 2.1 T2I、BF16、full checkpoint、swap24、all-block dynamic compile；允许研究高风险方案
代码基线：HEAD `4df001242` 加当前工作区改动；依赖实现以本机已安装 Diffusers / FlashAttention 源码为准

## 结论与证据修正

仍优先研究减少权重传输量。方案 1 已完成首次真机热测，但约 3.69% 的单轮改善不足以确认收益；在温频受控的交替复测和质量验收完成前保持显式 opt-in，不改默认值。其次是低比特驻留，以及专用训练 block 的 GEMM / 重算组织；Flash metadata 小修继续后置。热测明细见 [INT8 packed block-swap findings](../findings/qwen_image_2_1_int8_swap_90hx_20260927.md)。

此次审阅发现，先前判断需要补上三项限制：

1. **真实模型为 32 heads、hidden=4096。** `weights.py:217` 使用默认 `QwenImage21Transformer2DModel()`；本机 Diffusers 的 `transformer_qwenimage21.py:761` 默认 heads=32、head_dim=128、mlp_ratio=3。真实 safetensors header 的 `attn.to_q.weight` 为 `[4096,4096]`，`img_mlp.gate_up.weight` 为 `[24576,4096]`。先前 `phase2b_attention_microbench.py:212` 写死 heads=24，其 SDPA 分派和局部速度仅说明这个探针形状，不能确认 32-head 真实训练的 kernel。实际缓存仍是 75 个文本 token、4032 个图像 token。
2. **`gpu_wait_ms=0` 不是已测得 GPU 没等待。** `offloading.py:351` 的 `ANIMA_BLOCK_SWAP_PROFILE_GPU_WAIT` 默认关闭；`:1637` 仅在开启时创建 wait timing events，`:964` 对缺失数据填 0。现有 profile 无法用这些 0 排除传输瓶颈。另外该轮所有事件 `step=1`，因为计数在 prepare 时递增（`:1828`），不能按该字段划出 optimizer 稳态步。
3. **硬件链路是 PCIe 1.0 ×16。** 本轮只读查询 `nvidia-smi` 的 current/max link gen 均为 1、width 均为 16；`/sys/bus/pci/devices/0000:04:00.0/{current,max}_link_speed` 均为 `2.5 GT/s PCIe`，width 均为 16。没有修改驱动、时钟或链路设置。

这些修正不撤销真实训练 A/B/A 的 `7.95 → 7.38 s/step` 记录，但会改变瓶颈排序。原缓存实验仅有跨版本、不同 profile 条件的短测，不能证明该方向不存在任何收益；撤下缓存的理由仍是缺乏足够的端到端采用证据。

## 为什么换块应当排第一

当前路径已经使用 pinned CPU masters、整块 slab H2D、独立 copy streams，并去掉了 frozen weights 的 D2H。相关实现为 `library/runtime/offloading.py:1271`、`:1286`、`:1309`，所以再次提出“异步复制”“固定页内存”不会构成新的大优化。

`qwen21-phase3-flash-cache-r2-20260927/block_swap.jsonl` 记录：

| 项目 | 当前值 |
| --- | --- |
| Transformer blocks | 32 |
| 每块 frozen BF16 数据 | 436,208,128 bytes，约 416 MiB |
| 全部 block 权重 | 13.00 GiB |
| GPU 物理 slots | 8 = 32−24 |
| 20 步 H2D 记录 | forward 480 + backward 480 |
| 每步交换量 | 48 块，约 19.50 GiB |
| 单条 H2D event 中位数 | forward 142.06 ms、backward 142.18 ms |

8 个 slots 内的逻辑 block 会轮换，不是固定 8 个 block 永不下卡。`submit_move_blocks():1868` 虽使用 `depth=1`，实际把刚退休的 block i 的槽位用于 i+8；已有多块距离的流水，不能误解为只提前一个 block。

PCIe 1.0 ×16 在 8b/10b 编码后、忽略协议开销的单向上限为 4 GB/s。19.50 GiB 对应 **5.23 秒/步的理想传输下限**。它不是可以直接从 7.38 秒中减掉的独立耗时；传输和计算会重叠。但足以说明：即便把部分计算优化很多，BF16 交换量不变时仍受这个下限约束。多个 copy stream 不能突破物理链路总带宽。

## 方案排序

| 顺序 | 方案 | 改变的主要成本 | 风险与上限 |
| --- | --- | --- | --- |
| 1 | 整块 INT8 传输 + 融合反量化到 BF16 slot | H2D 字节量 | 接近减半 payload；改变 frozen base 数值，需要单独质量验收 |
| 2 | Qwen 专用低比特权重驻留 | 交换次数与 GPU 权重占用 | 有机会减少或取消 swap；反量化可能抵消收益，显存和质量未验证 |
| 3 | 专用训练 block：融合大投影 + 有预算的重算 | 大 GEMM 组织、重复 forward、临时激活 | 可保持算法等价；受传输下限约束，须与显存/slots 联合评估 |
| 4 | 统一 attention 计算图 | Python graph breaks、gather 和多段 attention 调用 | 复用现有 FA2 custom ops 或编译 Flex；尚无真实形状分项收益证据 |
| 可选激进分支 | 仅训练后 N 层、冻结前缀无梯度执行 | 前缀 backward / recompute / 反向换块 | 改变 adapter 容量，不能当作等价实现优化 |

### 1. 整块 INT8 传输，保持 BF16 GEMM

现成入口：Qwen adapter 已把 `transfer_dtype` 传给共享 `ModelOffloader`（`library/models/qwen_image_2_1/block_swap.py:42`）；共享 `Int8BlockSwapCpuMaster` 保存 int8 权重及 per-row scale（`library/runtime/block_swap_masters.py:73`）。这可以作为原型基础。

实现目标是：初始化时量化 frozen weights → 每块打包 int8 slab 与 scales → 单次/少量 H2D → GPU 分块反量化写入固定 BF16 slot → compute stream 等待该 slot ready。LoRA 权重与梯度保留当前精度。当前工作树已实现 packed CPU master、scale slab、BF16 slot 分块恢复及 resident block 的一致 INT8 初始化；没有单独的融合反量化 kernel。90HX profile 实测每块 payload 为 268,599,808 bytes（BF16 为 436,208,128 bytes，减少约 38.4%），应以该测量而非理论减半估算当前流量。该单轮对应约 3.69% 步时改善，不能由理论 PCIe 下限外推更多端到端收益。

这不是切换一个配置键：

- 已实现 INT8 专用 packed slab / restore plan：CPU 侧打包 INT8 payload 与 scale，GPU 侧复用 BF16 slot 暂存并分块恢复；90HX 热测使用 `restore_mode="slab"`。这替代了原先 `_get_cached_restore_slab()` 排除 INT8 master、回退逐权重恢复的状态。
- 已实现 resident blocks 从同一 INT8 master 物化；方案 1 的训练与 checkpoint recompute 经 90HX full-checkpoint + swap24 60-step 测试，避免首次 forward 使用原始 BF16、后续重算却使用反量化权重的生命周期不一致。
- 仍需明确不同 restore mode 的边界。90HX 热测只覆盖 `slab` 与 `int8_restore_mode="copy"`；`direct_bind`、`reuse_storage` 的舍入路径没有由本轮热测覆盖，不据此宣称全部恢复模式数值等价。
- 90HX 没有原生 FP8 Tensor Core 算术，但 FP8 **传输格式** 后恢复 BF16 不依赖它。不能把 FP8 compute 的硬件限制误当作 FP8 压缩传输不可能；优先 INT8 是因为可控的 scales 和现成 CPU master 结构，仍需与 FP8 数值/速度对照。

后续验收：本轮已记录每块 payload、反量化误差、峰值显存并完成训练保存；profile 尚不能提供 60 个 optimizer step 的传输事件分桶，且 GPU wait 计时未开启。仍需完成温频受控的交替性能复测、固定量化参考下的 forward/recompute 梯度对照、相对 BF16 的多提示词质量对照，以及 checkpoint reload/续训；只有重复整步收益和质量均通过后才考虑推荐。

### 2. 低比特驻留，争取消除换块

32 个 block 约 6.979B frozen 参数。只计算权重 payload，INT8 约 6.50 GiB，4-bit 约 3.25 GiB；还需加 scales、非 block 权重、LoRA/optimizer、activations、workspace 和 allocator 余量。因此不能宣称 INT8 必定能在 10GB 上全驻留，4-bit 的空间更宽裕。

这是 Qwen 专属量化加载/Linear 路径：当前 `weights.py:211` 严格加载 Diffusers 模型，`compat_matrix.py::_check_qwen_image_2_1_contract` 要求 BF16 并拒绝非 BF16 base compute。可参考 Krea 的预量化磁盘格式与 `Linear4bit` 支持，但不能直接套用 Krea 的吞吐或数值结论。

先支持冻结底模低比特存储、BF16 计算，确保 backward 仍正确计算 dX 和 LoRA 梯度。LoRA 与 optimizer 不量化。实验以“减少 swap 数/完全驻留后的整步时间”为指标，同时记录反量化开销；4-bit payload 小不等于大 GEMM 自动获得 4 倍加速。

与方案 1 的区别：方案 1 保持 GPU resident 权重为 BF16，主要省总线；本方案把 GPU 权重也压缩，目标是改变驻留布局。数值验收必须区分相对量化参考的实现正确性，以及相对 BF16 的模型质量变化。

### 3. 专用训练 block，联合优化 GEMM 和重算预算

当前 Qwen 是 single-stream，MLP 未挂 LoRA（`lora_targets.py:9`）。Diffusers 的 Q/K/V 分别做 Linear（`transformer_qwenimage21.py:336`），SwiGLU 的 gate/up 也独立计算（`:203`）；加载器把磁盘 packed `gate_up` 拆回两个运行时权重（`weights.py:222`）。

建议先为 Qwen 新建训练专用 block/MLP 模块，复用权重映射，优先：

1. 将 frozen SwiGLU gate/up 合成一次大投影，融合后续 SiLU×乘法，减少中间写回；不是每步 `cat` 权重。
2. 将 base QKV 合并为一次投影，同时保留 q/k/v 三套独立 LoRA 参数和保存命名。改成一个共享 rank16 QKV LoRA 会改变参数化，不是等价融合。
3. 在同一模块中实现明确的 activation 保存/重算策略；把省下的显存用于少量保存昂贵结果或增加权重 slots，实测二者谁更划算。

量级依据：每层 Linear frozen weights 为 `(4+9)×4096²`，MLP 占其 `9/13≈69%`。4107 tokens 的 Linear forward 约 1.79 TFLOPs/block，32 层约 57.3 TFLOPs/pass。full checkpoint 增加大量 forward 重算；non-reentrant early-stop/AOT 可能免去部分尾部，不能未经计数就认定每个算子完整执行两遍。大矩阵融合不减少矩阵乘法 FLOPs，收益来自 kernel 选择、输入复用及中间写回。

显存也必须按真实维度算：单份 hidden 32.09 MiB、QKV 96.26 MiB、SwiGLU 两路中间值 192.52 MiB/层。仅把两路中间值保留 32 层就约 6 GiB，不能直接取消全模型 checkpoint。只保存 attention output 也不能免掉 backward 所需的 Q/K/V；单纯把整块 checkpoint 拆成 attention/MLP 两个 checkpoint，不自动减少大 GEMM 重算。

当前 `compile.py:22` 已把 swap scheduler 留在图外，`:36` 编译 `adapter.inner_forwards`；“把换块放到编译外”不是待完成优化。`scope=resident` 当前只是编译前 `32−swap` 个逻辑 block，不能在 swap24 的轮换布局中解释为编译永驻块。若做固定 shape specialization，应按 image tokens、文本长度、batch/segment 布局管理图数量，不能只覆盖单图 benchmark。

验收重点：完整 block forward、输入梯度、每组 LoRA 梯度、保存/加载映射，checkpoint+swap 组合的 weight lifetime；测 GEMM 调用次数、实际重算次数、Dynamo graph breaks、peak 和稳态。不要仅通过放开兼容层的 selective flag 宣称已实现。

### 4. Attention 计算图重构

当前 wrapper 整体 `@torch.compiler.disable(recursive=True)`（`attention_backend.py:186`），mask 同步和 K/V boolean gather 仍在每次调用执行。FA2 2.8.3 provider **已经**注册 varlen forward/backward custom ops 与 fake implementations（本机 `flash_attn_interface.py:142,193,329,393`），所以不需要先重写一套 FA2 kernel 包装。

两个可比较原型：

- 将 shape/layout/有效索引准备移到 block 外，让真实 FA2 custom op 留在可编译的计算段；对于 padding 场景若采用整个训练序列的 packed layout，需要一起处理 RoPE、segment boundaries、residual 和输出映射。只缓存 metadata 的已撤下原型不等于这一方案已经测过。
- 接入 Diffusers 的 `QwenImage21FlexAttnProcessor`，用一次 block-causal attention 替代文本/图像多次调用。现成 processor 在 `transformer_qwenimage21.py:366`，当前 family 不允许 `flex`。必须验证编译前置条件，防止其未编译 dense FP32 attention 路径 OOM，并完整验证 backward、left padding 和 Edit 多 segment 语义。

统一 causal FA2 不能直接表示整段 Qwen block-causal mask；T2I 文本段很短，单独替换文本 SDPA 优先级较低。前缀 KV 也不能跨 optimizer step 按推理缓存直接复用，LoRA 更新会改变它，detach 还会截断训练梯度。

### 可选：改变可训练层范围

只训练末 8/16 层 attention LoRA，并把前缀显式放入 no-grad forward，可以免去前缀反向和 checkpoint 重算，还可能改为单向交换前缀。当前 `compat_matrix.py::_check_qwen_image_2_1_contract` 拒绝 layer-range targeting，现有 full-backward swap hooks 也不能直接复用。

这会减少 adapter 的表达能力，必须与全层 LoRA 做学习曲线/质量对照；冻结前缀也不允许把随噪声/timestep 变化的 DiT 激活做永久数据缓存。它是用户接受训练方法变化时的高风险支线，不作为等价性能优化混入基线。

## 执行与验收顺序

1. 从实际 forward 抓取 32-head Q/K/V shape/stride/dtype/mask；用 CPU operator trace 或强制 SDPA 对照确认 kernel。开启 GPU wait timing，按 optimizer window 关联 events；保存运行时源码 hash、配置、启动环境和温频遥测。这里的测量服务于大改方向选择。
2. 主线先验证 INT8 master 一致性，再实现 packed transport+fused restore；独立支线验证 4-bit/INT8 驻留的数值与显存可行性。
3. 若仍受大 GEMM/重算约束，再上专用训练 block；若 attention 占比与 graph breaks 支持，再上 attention 图重构。每次保留单因素对照，后续才组合。
4. 大改采用门槛建议为同代码/环境下交替对照稳态至少约 5% 收益；这是投入回报目标，不是对结果的承诺。至少 3 轮交替、60-step 以上持续窗口，无异常、无持续漂移，保存/重载/optimizer resume 通过。
5. 等价方案要求输出及 LoRA/input gradients 接近固定 BF16 参考；量化方案先通过固定量化参考正确性，再单独验收相对 BF16 的训练质量。若质量/速度不通过，保留原 BF16 torch/Flash 回退路径。

上述结构性审阅最初仅检查模型 header、缓存形状、源码、已有 profile 与 PCIe 信息；之后方案 1 已在当前工作树实现并完成首轮真机热测。该更新不改变 INT8 默认关闭的结论。
