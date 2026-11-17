# Adapter-aware Checkpoint 可行性探索文档

状态：半活跃提案（P0 size-based 策略已接入 Anima；P1–P3 仍为提案）
适用版本：当前 `dev`；默认关闭（`selective_checkpoint = "off"`）

## 核心结论

Adapter-aware checkpoint 在本项目中**可行，且 P0（基于张量尺寸的 SAC 策略）已在 Anima 族落地为显式实验开关**（`--selective_checkpoint adapter_aware` / `peak_blocks_adapter_aware`，配 `--selective_checkpoint_blocks`）；默认仍保持关闭，P1–P3（显式 adapter-tag 标注、routing sparse、激活压缩）继续作为后续探索提案。

## 1. 目标定义

该想法可以归纳为：

- **Base DiT 大 activation**：优先 checkpoint / recompute，例如 attention Q/K/V、attention 输出、MLP 中间激活。
- **Adapter 小 activation**：尽量保留，例如 LoRA down/up projection、router logits、routing weights、正交约束相关小矩阵。
- **Routing-aware**：保留 routing 小张量，只对大分支做重算。
- **Constraint-aware**：对 OrthoLoRA / Hydra / FeRA 等需要特定中间值的 adapter，避免错误重算或丢失必要梯度路径。

这属于 PEFT-aware selective activation checkpointing / selective rematerialization，即“知道哪些张量属于 base，哪些属于 adapter”的选择性激活保存策略。

## 2. 外部研究与实现核验

### 2.1 PyTorch 官方能力：直接可用

PyTorch 已经提供 selective activation checkpointing 机制：

- `torch.utils.checkpoint.create_selective_checkpoint_contexts`
- `CheckpointPolicy`
- non-reentrant checkpoint

官方文档：<https://docs.pytorch.org/docs/stable/checkpoint.html>

这为本项目实现 adapter-aware checkpoint 提供了直接工程基础。

可行路径是：

```text
Block.forward
  -> torch.utils.checkpoint.checkpoint(..., use_reentrant=False, context_fn=...)
  -> selective policy 决定哪些 op 输出 MUST_SAVE / PREFER_RECOMPUTE
```

### 2.2 TorchTitan 相关工作：概念相关，但需纠偏

TorchTitan issue：<https://github.com/pytorch/torchtitan/issues/2515>

它确实涉及 LoRA-aware checkpointing / adapter-aware 保存逻辑，但更偏向：

- 训练状态 checkpoint
- adapter-only 保存/加载
- base model 与 adapter 参数管理

它**不是完全等价于 activation checkpointing**。

所以它能证明“adapter/base 分离管理”是生产框架正在做的方向，但不能直接当作“LoRA activation checkpoint 策略”的完整实现依据。

### 2.3 HyC-LoRA：高度相关

HyC-LoRA 关注 LoRA fine-tuning 中 activation 内存瓶颈，结合：

- activation compression
- selective recomputation
- LoRA-aware error compensation

资料：

- OpenReview: <https://openreview.net/forum?id=WU86J4uM9V>
- PDF: <https://openreview.net/pdf?id=WU86J4uM9V>
- GitHub: <https://github.com/thu-ee-acts-lab/HyC-LoRA-release>

它与本项目想法高度相关，但实现复杂度高于第一阶段需要。

建议作为中长期参考，不建议第一版直接上 activation compression。

### 2.4 VeLoRA / mLoRA / HydraLoRA：相关但不是直接实现模板

相关资料：

- VeLoRA OpenReview: <https://openreview.net/forum?id=bFoQXD7Uls>
- VeLoRA PDF: <https://proceedings.neurips.cc/paper_files/paper/2024/file/4a9eaf6dff3fdac9ab1aaf4c0fe2d563-Paper-Conference.pdf>
- mLoRA arXiv: <https://arxiv.org/abs/2312.02515>
- mLoRA PVLDB: <https://www.vldb.org/pvldb/vol18/p1948-tang.pdf>
- HydraLoRA OpenReview: <https://openreview.net/forum?id=qEpi8uWX3N>
- HydraLoRA HTML: <https://arxiv.org/html/2404.19245v1>

它们证明：

- adapter 场景确实有独特内存结构；
- 多 adapter / routed adapter 的 activation 优化有研究价值；
- routing 与 checkpoint 可以结合。

但这些工作不等价于“本项目 DiT + LoRA/Hydra/FeRA 的现成方案”。

### 2.5 经典基础

Activation checkpointing 的基础论文：

- Chen et al. “Training Deep Nets with Sublinear Memory Cost”：<https://arxiv.org/abs/1604.06174>

NVIDIA / Megatron / NeMo 的 recomputation 文档也说明 selective checkpointing 已是成熟工程方向：

- NeMo DiT 文档：<https://docs.nvidia.com/nemo-framework/user-guide/25.02/vision/diffusionmodels/dit.html>
- Megatron Bridge activation recomputation：<https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/training/activation-recomputation.html>

## 3. 当前项目适配性评估

### 3.1 架构上适合做

当前项目中，DiT block 本身就是自然 checkpoint 边界：

```text
Anima
  -> Block[]
      -> attention
      -> MLP
      -> LoRA / Hydra / FeRA monkey-patched Linear
```

关键点：

- adapter 是 monkey-patch 到 DiT Linear 上的；
- `compile_blocks()` 必须在 `network.apply_to()` 后执行；
- checkpoint 放在 `Block.forward` 层是合理位置；
- selective checkpoint 可以针对 block 内 op 做策略选择。

这和 adapter-aware checkpoint 的需求基本匹配。

### 3.2 LoRA family 适配性较高

LoRA adapter activation 通常远小于 base activation：

```text
base hidden dim: 2048
LoRA rank: 4 - 128
```

例如：

```text
base activation: [B, tokens, 2048]
LoRA down activation: [B, tokens, rank]
```

因此策略上合理：

- base projection / MLP 大 tensor：重算；
- LoRA down / router / gate 小 tensor：保存。

### 3.3 Hydra / FeRA routing 需要谨慎

当前 routed adapter 不一定是 hard routing。

如果是 soft routing / all-expert einsum：

```text
所有专家都有参与计算，只是权重不同
```

那么“只 checkpoint 被 route 到的 branch”并不严格成立。

可行性分两层：

- **当前 soft routing**：保留 router logits / weights，小 tensor 可保存；专家计算仍按现有路径处理。
- **未来 hard top-k routing**：才适合真正做“只重算被选中的 branch”。

因此该点是**中长期优化**，不建议作为第一版目标。

### 3.4 OrthoLoRA / 正交约束可支持，但需要白名单

OrthoLoRA 这类 adapter 可能依赖：

- 小矩阵变换；
- Cayley / solve 类中间值；
- 正交约束相关统计量。

这些 tensor 通常较小，适合保存。

风险点是：

- 如果 selective policy 只按 op 类型重算，可能误重算某些数值敏感路径；
- 如果约束项依赖 forward 时缓存，需要确认缓存生命周期覆盖 backward。

建议第一阶段给这些 adapter 小 op / 小 tensor 默认 `MUST_SAVE`。

## 4. 推荐实现分层

### P0：size-based selective checkpoint，最小可行

策略：

```text
小 tensor: MUST_SAVE
大 tensor: PREFER_RECOMPUTE
view / reshape / detach / lightweight op: 默认保存或交给 PyTorch
matmul / addmm / bmm / attention / MLP 大输出: 倾向重算
```

优点：

- 侵入小；
- 和 adapter 小 activation 天然匹配；
- 不需要改 LoRA module forward；
- 可作为实验开关。

缺点：

- 不是真正语义级 adapter-aware；
- 可能把某些 base 小 tensor 也保存；
- 可能把某些 adapter 大输出也重算。

结论：**推荐作为第一阶段。**

### P1：adapter-tag aware policy，更精确

在 adapter forward 前后设置上下文标记：

```text
inside_adapter = True
```

然后 selective checkpoint policy 根据上下文判断：

```text
adapter op -> MUST_SAVE
base op -> PREFER_RECOMPUTE
```

优点：

- 更符合“adapter-aware”定义；
- 可以对 LoRA / Hydra / OrthoLoRA 分别定制；
- 后续可扩展到 FeRA、ReFT、GLoRA。

缺点：

- 需要修改 adapter module 或加入上下文管理；
- 与 torch.compile / TorchDispatchMode 交互要验证；
- monkey-patch Linear 的边界要处理干净。

结论：**适合作为第二阶段。**

### P2：routing-aware sparse checkpoint

仅当 routed adapter 变成 hard routing / top-k sparse routing 时成立：

```text
router logits 保存
selected expert branch 保存或重算
unselected branch 不计算 / 不保存
```

当前 soft routing 下收益有限。

结论：**不建议第一阶段实现。**

### P3：HyC-LoRA / VeLoRA 式 activation compression

进一步做：

- adapter-aware compression；
- 低 bit activation；
- error compensation；
- offload + rematerialization 联合调度。

优点是潜在收益大。

缺点是：

- 数值风险高；
- 工程复杂；
- 需要完整 benchmark；
- 对 DiT 图和 adapter 变体侵入较深。

结论：**适合研究分支，不适合直接进主训练路径。**

## 5. 主要风险

### 5.1 torch.compile 兼容性

本项目有 `compile_blocks()` 不变量：

```text
apply adapter -> load weights -> compile blocks
```

checkpoint 必须在这个顺序之后正确生效。

需要重点验证：

- eager + checkpoint；
- torch.compile + checkpoint；
- dynamic native flatten shape；
- block swap / low VRAM preset。

### 5.2 non-reentrant checkpoint 语义

建议只考虑：

```python
use_reentrant=False
```

原因：

- PyTorch 官方更推荐；
- 支持更复杂 autograd graph；
- 对 adapter 内部参数梯度更友好。

但仍要验证：

- 输入 latent 不 require grad 时，adapter 参数是否正常反传；
- LoRA custom autograd 是否与 checkpoint 重算兼容。

### 5.3 routing buffer 生命周期

Hydra / FeRA 可能有：

```text
_last_gate
_routing_weights
_freq_routing_weights
_content_routing_weights
```

这些必须至少活到 backward 完成。

如果训练 loop 在 backward 前清理，会出问题。当前项目看起来是在 step 前清理上一轮缓存，方向是合理的，但实现时仍需加测试锁住。

### 5.4 soft routing 与“只 route branch”冲突

如果当前实现是 soft MoE：

```text
所有专家输出 * routing weights 后求和
```

则不能声称只 checkpoint 被 route branch。

应改成更准确表述：

```text
保留 routing 小 tensor；
专家大 activation 按现有 dense/soft 路径参与 checkpoint；
未来 sparse routing 才做 selected-branch checkpoint。
```

### 5.5 显存收益可能受瓶颈限制

如果当前显存主要来自：

- optimizer state；
- frozen base 参数；
- attention backend workspace；
- activation 外的缓存；
- compiled graph cache；

那么 adapter-aware checkpoint 的收益会被稀释。

所以必须用 memory probe / profiler 实测，而不是只靠理论估算。

## 6. 建议验证矩阵

### 6.1 正确性测试

建议覆盖：

- LoRA
- OrthoLoRA
- HydraLoRA
- FeRA / global router
- T-LoRA mask
- custom down autograd
- torch.compile
- native flatten bucket shape

验证指标：

```text
loss 是否有限
adapter grad 是否存在
grad norm 是否接近 baseline
router grad 是否存在
没有 backward double-free / stale cache
```

### 6.2 性能测试

对比模式：

```text
off
mlp_only
every_other
adapter_aware
peak_blocks_adapter_aware
```

指标：

```text
torch.cuda.max_memory_allocated
torch.cuda.max_memory_reserved
step time
forward time
backward time
recompute 次数
loss 曲线
```

建议先跑 2-5 step smoke，不启动长训练。

### 6.3 预期收益

理论上，收益排序大致可能是：

```text
full block checkpoint > adapter-aware checkpoint > mlp-only checkpoint > off
```

但 adapter-aware 的优势在于：

- 比 full checkpoint 少重算 adapter 小路径；
- 对 PEFT 更友好；
- routing / Ortho 小 tensor 不容易被误处理；
- 可能在显存与速度之间取得更好折中。

## 7. 推荐落地策略

建议未来实现时采用：

```text
实验开关，不默认开启
```

例如：

```text
selective_checkpoint = "adapter_aware"
selective_checkpoint = "peak_blocks_adapter_aware"
```

推荐优先级与当前状态：

1. P0 size-based SAC policy（**已在 Anima 落地**：`--selective_checkpoint adapter_aware` / `peak_blocks_adapter_aware`）
2. 单元测试和短步 smoke 对照（持续补充多分辨率/多变体数据）
3. 若后续收益明确，再推进 P1 adapter-tag aware（仍为提案）
4. P2 routing sparse 和 P3 activation compression 放后续研究分支（仍为提案）

## 8. 最终判断

Adapter-aware checkpoint 对本项目是：

- **技术上可行**
- **和 DiT + LoRA family 架构匹配**
- **有 PyTorch 官方 selective checkpoint API 支撑**
- **有 HyC-LoRA / VeLoRA / mLoRA 等相关研究支撑**
- **适合作为实验型显存优化功能推进**

但第一版不应承诺：

- 完全只 checkpoint base；
- soft routing 下只重算 route branch；
- 所有 adapter 变体零风险兼容；
- 一定优于现有 checkpoint。

更稳妥的定位是：

> 一个 PEFT-aware 的 selective activation checkpoint 实验模式，用更低侵入的方式优先重算 base 大 activation，同时尽量保留 adapter / router / constraint 小 tensor。
