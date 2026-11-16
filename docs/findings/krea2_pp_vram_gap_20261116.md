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

## 4. 缺口：每张卡都差最后一点

测试配置：4032-token bucket（latent `(1,16,1,126,128)` → 63×64 = 4032 图像 token，256 文本 token），
**梯度检查点开启**，**attention 走 memory-efficient**，`PYTORCH_ALLOC_CONF=expandable_segments:True`。
单卡 V100、真实权重、fp16、**全参数可训练**（见 §6 的偏差说明）。

| 切分 | stage | 权重驻留 | 前向峰值 | backward |
|---|---|---|---|---|
| **13/15** | stage0（13 blocks + fusions） | 11.75 GiB | 13.12 GiB 通过 | 未测（预计更紧） |
| **13/15** | stage1（15 blocks + last） | 12.13 GiB | **13.52 GiB 通过** | **OOM，差 52 MB** |
| 14/14 | stage0（14 blocks + fusions） | 12.56 GiB | 14.82 GiB 通过 | 未测（仅剩 0.39 GiB） |
| 14/14 | stage1（14 blocks + last） | 11.32 GiB | 12.66 GiB 通过 | **OOM，差 192 MB** |

**结论：最优配置（13/15）的 stage1 在 backward 阶段差 52 MB。**

一个反直觉但很重要的观察：14/14 的前向峰值更低（12.66 < 13.52），但 **backward 缺口更大**
（192 MB > 52 MB）。原因是 `torch_checkpoint(use_reentrant=False)` 在重算每个 block 时会把该
block 的全部中间态重新分配，backward 峰值由「重算瞬时峰值」主导，而不是前向残留。
**所以切分必须按 backward 峰值挑，不能按前向挑。**

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
| **A. 降 token bucket**（4032 → 3584 或更低） | 激活按 N² 降约 20%（远超 52 MB） | 分辨率下降；3584 在 `CONSTANT_TOKEN_BUCKETS` 里有相邻档。**风险最低** |
| **B. 硬挤 52 MB** | 可能过 | 脆弱：换 torch 版本 / 换 bucket 就会崩；不是工程余量 |
| **C. 接 LoRA 后重测**（§6 的偏差） | 可能直接不需要挤 | 需要先把 adapter 接进 PP 路径（工作量中等） |
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
