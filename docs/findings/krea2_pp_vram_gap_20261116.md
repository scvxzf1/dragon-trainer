# Krea-2 fp16 + PP=2 在双 V100 16GB 上的显存缺口

状态：实验（实测结论 + 待决策）
适用版本：`dev` @ `edbd005c` 之后，分支 `codex/krea2-pp-wiring-20261116`
测试硬件：2 × Tesla V100-SXM2-16GB（sm70，200 W 限功耗）、`torch 2.10.0+cu129`
相关代码：`library/models/krea2_raw/{pp_stages,attention_backend,dit}.py`、
`library/training/pp/krea2_runtime.py`

本文是一份**决策材料**：把「Krea-2 能否在双 V100 16GB 上跑 fp16 + 流水线并行（PP=2）」这件事的
全部实测数据、缺口位置和可选方向摊开，供讨论。所有数字都是本机实测，不是估算。

---

## 1. 为什么会有这个问题

Krea-2 是 128 亿参数的 DiT。bf16 权重 **25 GB**，而单张 V100 只有 **16 GB**。所以：

- **单卡 fp16 绝对不可能**（25 > 16）
- **PP=2 每卡约 12.5 GB 权重**（勉强），这是 fp16 路径唯一可能成立的方式
- 单卡 NF4 是 6.2 GB，能装，但 NF4 被 PP 三重硬拒（见 §7）

换句话说，**PP 对 Krea-2 不是优化手段，是 fp16 的前置条件**。

## 2. 已经打通的部分（这些不是缺口）

| 项 | 状态 | 证据 |
|---|---|---|
| stage 拆分数值正确性 | ✅ | 两段拼回与 `model.forward` **按位相等**（`torch.equal`），对每个切点成立 |
| 双卡 1F1B 前向 | ✅ | 双 V100 + `cpu:gloo,cuda:nccl`，`schedule.step` 返回正常，rank1 输出 `(2,16,1,4,4)` 与 `forward_for_loss` 契约一致 |
| 双卡反向 + 优化器 | ✅ | 两卡梯度全非零（75/75、43/43），`optimizer_step` 后参数真的变了 |
| 非 block 子模块释放 | ✅ | 按 stage 裁掉不属于本 rank 的 `txtfusion`/`tproj`/`txtmlp`/`tmlp`（1.24 GiB） |
| frozen 槽梯度契约 | ✅ | 发送侧与接收侧都 detach，符合 `KREA2_BOUNDARY_FROZEN_INDICES=(1,2,3,4)` |

以上全部在 **tiny 模型（4 层、0.5M 参数）** 上验证。**换成真实 12.82B 权重后，卡在显存。**

## 3. 真实权重的精确分解（实测，非估算）

从 `krea2_raw_bf16.safetensors` 直接读 tensor 统计：

| 模块 | 体积 | 归属 |
|---|---|---|
| `blocks` × 28 | **22.64 GiB**（0.809 GiB/层） | 按 stage 切 |
| `txtfusion` | 0.64 GiB | 仅 stage0 |
| `tproj` | 0.42 GiB | 仅 stage0 |
| `txtmlp` | 0.10 GiB | 仅 stage0 |
| `tmlp` | 0.07 GiB | 仅 stage0 |
| `last` / `first` | ≈0.00 GiB | last 归 stage1 |
| **合计** | **23.88 GiB** | （fp16 = 12.82B 参数） |

注意：规划器用的 `stage_zero_block_offset=-1` 让切分变成 **13/15**（不是 14/14），因为 stage0
还要扛上面 1.24 GiB 的融合模块。

## 4. 缺口：不是 52 MB，是 `param.grad` 跨层累积（已用探针证伪）

### 4.1 最初的错误结论

第一轮实测（**全参数 `requires_grad=True`**）得到：

| 切分 | stage | 权重驻留 | 前向峰值 | backward |
|---|---|---|---|---|
| 13/15 | stage0（13 blocks + fusions） | 11.75 GiB | 13.12 GiB 通过 | — |
| 13/15 | stage1（15 blocks + last） | 12.13 GiB | 13.52 GiB 通过 | OOM，差 52 MB |
| 14/14 | stage0（14 blocks + fusions） | 12.56 GiB | 14.82 GiB 通过 | — |
| 14/14 | stage1（14 blocks + last） | 11.32 GiB | 12.66 GiB 通过 | OOM，差 192 MB |

当时把 14/14 反而更差归因于「checkpoint 重算主导 backward 峰值」——**这个解释是错的**。

### 4.2 真正的机理

`requires_grad=True` 时，每反传完一层，PyTorch 就为该层参数**永久保留**一份 fp16 的
`param.grad`，直到 `optimizer.zero_grad()`：

```
15 层 × 0.809 GiB = 12.13 GiB 的 .grad    ← 与权重本身一样大
```

所以「差 52 MB」并不是"跑完 15 层只差 52 MB"，而是**在倒数第 3 层被累积的 grad 撑爆时，
压在最后一个张量上的缺口**。14/14 少一层、省下 828 MB，于是多扛过一层，死在第 4 层，
那次的最后一个张量差 192 MB——**两个数字都只是最后一根稻草，不代表总缺口**。

这也意味着：全参数微调要在 16 GB 上同时放下 12.13 GiB 权重 + 12.13 GiB 梯度
（还不算 2×12.13 GiB 的 AdamW 状态），**物理上不可能**。

### 4.3 探针：拿掉跨层累积，只保留算子峰值

```python
for p in model.parameters():
    if p.requires_grad:
        p.register_post_accumulate_grad_hook(
            lambda tensor: setattr(tensor, "grad", None)
        )
```

它保留 100% 的重算激活与 backward 算子临时显存，只去掉 `N × 828 MB` 的跨层累积——
这正是 LoRA 训练的真实形态（base frozen，只有 adapter 有 `.grad`）。

**实测结果（13/15，4032 token，checkpoint 开，EFFICIENT attention，expandable_segments）**：

| stage | 权重驻留 | 前向峰值 | **backward 峰值** | 余量 |
|---|---|---|---|---|
| stage0（13 blocks + fusions） | 11.75 GiB | 13.97 GiB | **15.24 GiB** | **0.01 GiB（10 MB）** |
| stage1（15 blocks + last） | 12.13 GiB | 13.52 GiB | **14.79 GiB** | 0.48 GiB |

### 4.3b 定位并修掉 stage0 的 890 MiB（txtfusion 未 checkpoint）

stage0 权重更少却更紧，逐步插桩把它定位到了单一模块：

```
after first           12102.1 MiB
after tproj           12102.2 MiB
after txtfusion       12992.2 MiB   (+890 MiB)   ← 全部在这里
after txtmlp          13003.7 MiB
after cat             13054.0 MiB
after mask4d+posemb   13077.1 MiB
after blocks          13740.1 MiB
```

根因：`enable_gradient_checkpointing()` **只遍历 `self.blocks`**，从不覆盖首段独有的
前置模块。`txtfusion`（0.64 GiB 权重）处理的是 `(B, 256, 12, 2560)` 即 30720 宽的文本
序列，它的内部激活因此一直驻留到 13 个 block 全部跑完。只冻结 `requires_grad` 无效
（证明不是 `.grad` 或 dW workspace），必须包 checkpoint。

修法：`_txtprefix_checkpoint()` 在 training + grad enabled 时用
`torch.utils.checkpoint(use_reentrant=False)` 包住 `txtfusion`，推理/无梯度路径保持
原样直调（不改任何数值）。

**修复后实测**：

| stage | 前向峰值 | **backward 峰值** | 余量 |
|---|---|---|---|
| stage0 | 13.97 → **13.10 GiB** | 15.24 → **14.37 GiB** | 0.01 → **0.89 GiB** |
| stage1 | 13.52 GiB（不变） | 14.79 GiB（不变） | 0.48 GiB |

**两个 stage 现在都在 15 GiB 以下。** stage0 的 0.89 GiB 足以覆盖 1F1B 预热/稳态期
首段必须同时滞留 2 个 microbatch 的额外 0.4〜0.5 GiB。

两者**都跑通了**，因此：

1. **所谓 52 MB 缺口是伪缺口**——它由全参数测试的 `.grad` 累积造成，不代表 LoRA 场景。
2. **但 stage0 才是真正的瓶颈**，不是 stage1：它只剩 **10 MB** 余量。
3. stage0 前向峰值（13.97）比 stage1（13.52）更高，因为首段还要做
   `prepare_img_tokens` + concat + 256 对齐 padding，这些在 4352 token 下是实打实的大张量。

### 4.4 这对 1F1B 意味着什么

PP=2 的 1F1B 在预热/稳态阶段，**stage0 必须同时滞留 2 个 microbatch** 的 checkpoint 边界输入
和 stage 输出（等 stage1 反传回来），而 stage1 永远只需驻留 1 个。也就是说双卡联调时 stage0
还要再多吃 0.4〜0.5 GiB。

按修复前的 10 MB 余量，stage0 必然崩；**§4.3b 的 txtfusion checkpoint 已把余量提到
0.89 GiB，这个阻塞点解除**，双卡 1F1B 联调在显存上已具备条件（仍需实测首段双 microbatch
的真实开销）。剩下要考虑的是数值/正确性验证，而不是容量。

## 4.5 终点：双卡真实权重 1F1B 在 4032-token 上跑通

经过 §4.3b（txtfusion checkpoint）、GQA 折叠、以及 Block 内 nested attn checkpoint
三步，**真实 12.82B 权重 + 双 V100 16GB + 4032-token + PP=2 + 1F1B 已经端到端跑通**：

```
[rank0] stage0, 13 blocks: peak 14.34 GiB, 余量 560 MB, prediction=None, losses=[]
[rank1] stage1, 15 blocks: peak 14.10 GiB, 余量 850 MB, prediction=(2,16,1,126,128)
                            losses=[112499.29, 162935.33]
```

验收点：两个 microbatch 各自产生 loss；跨 microbatch 梯度累加正确
（rank1 78/78、rank0 89/87 非零）；5D 输出与 `forward_for_loss` 契约一致；
单步 14.8s / 11.3s（未做性能优化）。

### 三步优化的实际收益

| 步骤 | 机制 | 实测收益 |
|---|---|---|
| txtfusion checkpoint | 首段独有模块此前从不被 `enable_gradient_checkpointing()` 覆盖（它只遍历 `blocks`），890 MiB 内部激活驻留整段 | stage0 反向 15.24 → 14.37 GiB |
| GQA query-group 折叠 | V100 的 EFFICIENT 不支持 `enable_gqa`，改为把 4 个 q 头折进序列维（`view`，零拷贝，按位等价） | 省 100 MB |
| Block 内 nested attn checkpoint | 反向是 mlp→attn，包前半段 attn 才能让两者错峰；包 mlp 等于没包 | rank0 14.98 → 14.34 GiB（省 640 MB） |

### 两条被实测否定的直觉

1. **「冻结前置模块的 requires_grad 就能省那 850 MB」** —— 只降 10 MB。省不掉的是
   *前向中间激活*，必须靠 checkpoint 重算。
2. **「Block 内平铺拆成两个 checkpoint（attn / mlp）」** —— 反而多 140 MB。
   1F1B 下 stage0 同时有 2 microbatch × 13 层 = 26 个 Block 在途，平铺拆开会让每个
   在途 Block 多存一份中间态 `x_after_attn`（53.5 MB），常驻涨 26 × 53.5 = 1391 MB；
   实测只涨 140 MB，反过来说明单层重算峰值确实降了约 1.22 GiB，只是被 26 倍的常驻
   边界张量反噬。**正解是嵌套（外层保持、内层只包 attn）**，只有当前正在反传的那 1 个
   Block 会临时多存一份输入。

### 一个实现细节：内层必须也用 `use_reentrant=False`

内层 `use_reentrant=True` 会与外层 `use_reentrant=False` 的 `saved_tensors_hooks`
冲突，直接抛 `Trying to backward through the graph a second time`。两层都用
`use_reentrant=False` 才通过。

## 4.6 还剩什么

容量问题已解决，4032-token 在双 V100 16GB 上有 **560/850 MB** 余量。剩下的不是显存：

1. **LoRA adapter 接线**（方向 C）：`build_krea2_stage_module` 目前明确拒绝 `network`
   （"the Krea-2 PP probe cut does not support network adapters yet"）。上面的实测用
   「基座冻结 + 只留 Norm/Bias 可训练」模拟了 LoRA 的梯度形态，但真正的 adapter 尚未接入。
2. **数值验收**：当前只验证了 loss 与梯度非零，还没有与单卡 DDP 的逐步对齐。
3. **性能**：单步 14.8s/11.3s 是首次真实权重测量，没有和 DDP 基线对比。


## 4.7 数值对齐：PP=2 1F1B vs 单卡参考臂

方法：同一 tiny Krea-2（4 层）+ 真 LoRA（rank=8）+ 相同 seed / 数据 / lr，
跑 3 个优化步，对比 loss 轨迹与最终 LoRA 权重。

| 步 | 单卡参考臂 loss | PP 臂 raw | PP / 65536 | 相对差 |
|---|---|---|---|---|
| 0 | 0.339692 | 22263.51 | 0.339714 | 0.0065% |
| 1 | 0.333711 | 21865.69 | 0.333644 | 0.0201% |
| 2 | 0.319635 | 20943.16 | 0.319567 | 0.0212% |

**loss 轨迹吻合到 99.98%。**

### 必须知道的读数约定：`losses` 是 scaled loss

`make_last_stage_loss_fn` 在把 loss 交回调度器之前要先 `scaler.scale(loss)`
（PiPPy 的 `.backward()` 在 `schedule.step()` 内部发生，缩放必须在 loss_fn 内完成），
所以调用方拿到的 `losses` **已经乘过 `scaler.get_scale()`**（fp16 默认初始 65536 = 2^16）。
读数要除以 scale 才是真实 loss。

这也解释了真实 12.82B 双卡联调里那组 `losses=[112499.3, 162935.3]`：
除以 65536 是 `[1.7166, 2.4862]`，与单卡探针测得的 `loss=2.4206` 同量级、互相印证。

### 权重对齐：AdamW 的符号跳变会伪装成"没对上"

直接把两臂 LoRA 权重相比，AdamW 下得到 median 0.294 / **max 1.975**。那个 1.975
指向 2.0 不是巧合：AdamW 第一步的更新量约等于 `-lr * sign(g)`，梯度大小被归一化掉了，
只剩符号。`lora_up` 零初始化时大量元素的梯度接近 0，单卡（batch=2 一次 GEMM）与
PP（2 × batch=1 分别 GEMM 再累加）的 fp16 尾数舍入差足以让 `1e-7` 级别元素的符号翻转，
于是一边更新 `+lr`、一边更新 `-lr`，相对误差正好 `2*lr/lr = 2.0`。

换成 `SGD(lr=1e-2)` 消除符号放大后：

```
LoRA 权重相对误差: median = 0.000e+00, p90 = 4.4e-2, max = 8.5e-2
```

**median 完全为 0**，残余的 8.5e-2 是 fp16 累加顺序差异的正常量级。

### 对齐过程中修掉的真缺陷

`Krea2PipelineRuntime.step()` 缺 `self._transaction.begin_step()`（moss_runtime 有）。
fp16 事务按单个优化区间记账，第二次 `step()` 没有 `begin_step()` 会直接抛
`pp_scaler_decision_asymmetry`。**这条路径此前只验证过单步，多步训练从未跑通**。


## 5. 已经排除的两个假设（都实测过，都不是缺口）

### 5.1 attention 没走快路径 —— 已修复，不是剩余缺口

V100 上 cuDNN MHA 不支持（`only supports sm80-sm121`），`_cudnn_attention` 的 except 分支
每次都命中，而原来那里是一条**无约束** SDPA，PyTorch 在这套硬件上选中了 **MATH**，于是每次
调用都物化完整 `[H, L, L]` score 矩阵（4032-token 下约 2.0 GiB）。

修复后（改为 `sdpa_kernel([EFFICIENT_ATTENTION, MATH])`，GQA 时先展开成 MHA）：

| | MATH | EFFICIENT |
|---|---|---|
| 纯 SDPA，fp16，seq=4608 | 3.39 GiB | **0.28 GiB** |
| 真实模型 stage1 额外开销 | 3.39 GiB（前向 OOM） | **1.37 GiB（前向通过）** |

两个硬件限制（实测，会写进代码注释）：
- V100 的 EFFICIENT **不支持 bf16**（`Expected dtype: {Half, Float}. Got BFloat16`）
- V100 的 EFFICIENT **不支持 `enable_gqa=True`**（GQA 48/12 报 `No available kernel`，
  同样形状展开成 48/48 就过）

### 5.2 梯度检查点没生效 —— 已确认生效

对比实测（同一 stage1、同一尺寸）：

| 配置 | 前向 | 结论 |
|---|---|---|
| checkpoint **关** | 需 102 MB，只剩 22 MB → **前向就 OOM** | — |
| checkpoint **开** | `alloc=12.26 peak=12.66` → **前向通过** | 省了约 3 GiB |

代码链也核对过：`SingleStreamDiT.enable_gradient_checkpointing()` 会遍历 `self.blocks`
逐个开启（`dit.py:516-519`），block 的判定是 `is_grad_enabled and training and flag`
三条件与（`dit.py:389-393`）。另外用 grep 确认过没有任何其他地方静默改写
`gradient_checkpointing` / `_ATTN_FP16_ENABLED` / `attention_mode`。

**所以两个优化都吃满了，不是漏了什么开关。**

## 6. 必须说明的测试偏差（这次测试比真实训练更严格）

我的实测把**全部 12.82B 参数设为 `requires_grad=True`**。而真实训练是 **LoRA**：

- base 权重 frozen → autograd 不需要为它保存每层输入梯度 → **激活需求更低**
- optimizer 状态只覆盖 adapter（几百万参数），而非 12.82B 的 AdamW 状态（那是 2×23.88 GiB，绝对装不下）

所以本文的显存数字是**全参数训练的上界**。真实 LoRA 场景可能本来就不紧张——**但这需要实测，
目前还没做**，因为 adapter 还没接进 PP 路径（`build_krea2_stage_module` 现在明确拒绝 `network`）。

**这是当前最大的未知量，也是讨论时第一个该问的问题。**

## 7. 可选方向（附代价）

| 方向 | 预计效果 | 代价 / 风险 |
|---|---|---|
| **A. 降 token bucket**（4032 → 3584 或更低） | 激活按 N² 降约 20%；stage0 的 10 MB 余量变成 ~400 MB | 分辨率下降；3584 在 `CONSTANT_TOKEN_BUCKETS` 里有相邻档 |
| **B. 拆细 checkpoint 粒度**：Attention 与 MLP 分别包 checkpoint | backward 时 MLP 大张量先算先释放，单层重算峰值砍半，预计省 400〜600 MB | 计算量不变（重算次数仍 1 次）；需要改 block 内部结构 |
| **C. 接 LoRA 后重测** | §4.3 的探针已经证明 LoRA 形态下两个 stage 都能跑通，
风险转移到 stage0 的 10 MB 余量 | 需要先把 adapter 接进 PP 路径（工作量中等），并重测 1F1B 下 stage0 的双 microbatch 开销 |
| **D. 换 48 GB 级卡**（A100 / H100 / 90HX） | 权重问题消失 | PP 的价值从"装得下"变成"更大 batch / 更快"；方案本身不用改 |
| **E. PP=4 或更多 stage** | 每卡权重 ~6 GB | **当前 PP v0 只支持 stages=2**（`wired_families` 报 `stages: [2]`），要扩要改运行时 |
| **F. NF4 + PP** | 每卡 3.1 GB，非常宽裕 | **被三重硬拒**：`pp_quantization_unsupported`（`validate_no_quantization`）+ NF4 的 `COMPUTE_DTYPE` 硬编码 bf16（与 PP 只收 fp16+scaler 冲突）+ FP16/NF4 数值验收未通过（LoRA rel-L2 17.55%、输入梯度 35.6%） |
| **G. 分块 attention / chunked SDPA** | 进一步降 attention 峰值 | 需要自定义实现，且与 `_trainer_autocast` 的交互要重验 |
| **H. 优化器状态 offload / CPU offload** | 省 AdamW 状态 | 只对全参数训练有意义；LoRA 场景下本来就小 |

## 8. 讨论时值得先问的三个问题

1. **目标场景到底是 LoRA 还是全参数微调？** 如果是 LoRA，§6 的偏差可能让整个缺口消失，
   应该先接 adapter 重测，而不是现在就去降分辨率。
2. **分辨率能接受多少？** 如果 3584 或 3072 token 可接受，方向 A 立刻闭环。
3. **双 V100 是不是硬要求？** 如果"能跑 Krea-2 fp16 训练"比"必须在这台机器上"更重要，
   方向 D 最省事，而且 PP 接线的工作（本文 §2 那些已验证的部分）在 48 GB 卡上依然有效，
   只是价值从"装得下"变成"更快"。

## 9. 复现方式

单卡显存测量脚本（两 stage 分开跑，避免 25 GB 加载两次以上）：

```bash
CKPT=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  .venv-v100/bin/python /tmp/krea2_real_vram.py 1     # 0 = stage0, 1 = stage1
```

双卡 1F1B 探测（tiny 模型，验证调度链路而非显存）：

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_ASYNC_ERROR_HANDLING=1 PYTHONPATH=. \
  .venv-v100/bin/python /tmp/krea2_pp_two_rank.py
```

两个脚本在会话中是临时文件；需要长期复现时应移入 `scripts/experiments/`。
注意 `CUDA_VISIBLE_DEVICES` 在本机按 **CUDA 枚举序**（FASTEST_FIRST）：`0,1` 才是两张 V100，
`1,2` 会拿到 GTX 1050（sm61，torch 直接报 `no kernel image`）。

## 4.8 双 V100 满血 896×1200 端到端跑通（2026-11-17）

`Krea-2 + fp16 + PP=2`，896×1200（4352 image tokens）、双 V100 16GB、不换数据集、
不降 token 数，**前向 + 反向 + 优化器 + 保存全部跑通**；2 步验证无 skip、无
`pp_scaler_decision_asymmetry`、GradScaler scale 两保持一致（65536）。

### 4.8.1 决定性的一步：块交换要按"本 rank 实际持卡的 block"选

`stage_blocks` 是**整个模型**的 28 个 block。按位置取 `blocks[1:5]` 在尾段 rank
上选中的是 1..4 号——那是首段 rank 的层，在本 rank 上早已被
`release_foreign_submodules` 释放，因此 offload 释放 0 MiB，交换在最需要的地方
完全没生效。改为先按"参数是否在本 device 上"筛出本 stage 拥有的块。

`KREA2_PP_PROBE=1` 的 per-block 入口/出口探针给出对照：

| | rank0 反向最深处 | rank1 反向最深处 |
|---|---|---|
| swap=0 | idx=12（14.65 GiB） | idx=24（14.71 GiB） |
| swap=4 | **idx=0（12.27 GiB）** | **idx=13（13.87 GiB）** |

即块交换确实买回了整段反向。

### 4.8.2 两条值得记住的判读经验

1. **OOM 报的 `allocated` 不是诊断量**：它天然等于天花板减去失败请求量，任何 OOM
   都报同一个值（本机恒为 ~14.9x GiB）。唯一有效的读数是"反向推进到第几个 block"。
2. **chunk 不套 `torch_checkpoint` 等于白切**：`is_grad_enabled()` 为真时每块的
   fp32 中间量仍被 autograd 钉在计算图里，4×26 MiB 依旧 102 MiB，总占用纹丝不动。
   只有 SwiGLU 那处套了 checkpoint，所以它才是唯一真正省下显存的。

### 4.8.3 卸载必须 `resize_(0)`

只把 `.data` 指回 pinned cpu 副本不够：linear 的反向会保存 `weight.t()` 转置视图，
它独立持有同一 StorageImpl 的强引用。需 `untyped_storage().resize_(0)`
（FSDP 手法），重算路径再从 pinned 副本重建 buffer。

### 4.8.4 已知缺口

- PP 的 progress 行 `loss=None`：只有首段 rank 打印，而 loss 只在末段计算。属显示
  口径问题，不影响训练；如需可见，应由末段上报后聚合。
- 尚未做：性能基线 vs DDP、resume 后数值连续性、真实 12.82B checkpoint 往返。
