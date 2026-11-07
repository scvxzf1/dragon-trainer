# Krea-2-Raw 梯度检查点落地方案

状态：已归档\
归档日期：2026-10-04\
原位置：`docs/proposal/krea2_raw_gradient_checkpointing.md`\
当前说明：[当前能力与维护入口](../../../docs/multi_model_support.md)

以下正文保留原始历史记录；代码路径、命令、实现状态和验收要求按原记录日期解释。
当前开发与验收遵循 [AGENTS.md](../../../AGENTS.md)。

---

状态：已实现（配套 [`krea2_raw_migration.md`](krea2_raw_migration.md) 阶段 4 训练串通；落地见 `library/models/krea2_raw/dit.py`，1024×1024 真实训练实测通过，见 [`../findings/krea2_raw_migration_stage6_findings.md`](../../../docs/findings/krea2_raw_migration_stage6_findings.md)）
适用版本：当前 main（grad-ckpt 已落地，提交 `0f8f934c`；本文档作设计背景，部分假设已与实现分叉——见 §11 落地核对）
日期：2026-08-08（原调查稿）；2026-08-09 补落地核对 §11
入口命令：`python tasks.py lora model_family=krea2_raw` + `gradient_checkpointing=true`（落地后训练入口）

相关代码：

- 迁移目标 DiT 移植源：`krea-ai/krea-2` GitHub `mmdit.py`（本地探查副本 `/tmp/krea2_mmdit.py`，官方仓库无训练代码）
- anima 现有 grad-ckpt 机制（迁移基线）：`library/anima/models.py`、`library/runtime/harness.py`、`library/runtime/offloading.py`、`library/training/bootstrap.py`
- compile × grad-ckpt × block swap 测试矩阵：`tests/test_compile_checkpoint_block_swap_hot.py`

> **一句话：** 官方 `krea-ai/krea-2` 是纯推理仓库，没有训练/grad-ckpt 代码；但 diffusers 主干 `Krea2Transformer2DModel` 已有标准 block 级 grad-ckpt（`use_reentrant=False`），社区实现（multimodalart Space、ai-toolkit）同模式。anima 训练器自研的 grad-ckpt 机制（手动 bool 标志 + block 级 + `use_reentrant=False` + UnslothOffloadedGradientCheckpointer + adapter_aware selective + compile 编译 `_forward`）与 Krea-2 block 结构高度兼容，**直接复用大半套，只重写 block 内层 `_forward`**。block swap × grad-ckpt 调度问题在 anima 现有架构中已正确处理（backward hook 在 recompute 完成后才 swap out），Krea-2 直接继承。24GB 4090 训练 13B LoRA 的保底路径是 NF4 量化 + LoRA + grad-ckpt + 8bit Adam，anima 独有的 block swap 叠加是更激进路径。

---

## 1. 调查证据：三方对照

| 维度 | 官方 krea-ai/krea-2 | diffusers 主干 `Krea2Transformer2DModel` | anima 训练器（现状） |
|---|---|---|---|
| 有 grad-ckpt | ❌ 无（纯推理，`mmdit.py` 全文无 checkpoint/use_reentrant/requires_grad） | ✅ 有 | ✅ 有（自研多档位） |
| API 形态 | — | `_supports_gradient_checkpointing=True` + `enable_gradient_checkpointing()` | 手动 bool 标志 + `Block.enable_*_checkpointing` |
| 粒度 | — | block 级（整个 `Krea2TransformerBlock`） | block 级（主力）+ MLP 级 + selective |
| `use_reentrant` | — | `False`（ModelMixin 默认） | `False`（全部分支） |
| 与 compile 顺序 | — | grad-ckpt → compile | grad-ckpt → compile（不变量） |
| 训练脚本 | ❌ 无 | ✅ `train_dreambooth_lora_krea2.py`（`--gradient_checkpointing` flag） | ✅ 本仓库 |
| block swap | — | ❌ 无 | ✅ 有（`ANIMA_BLOCK_SWAP_*`，社区无此路径） |
| 24GB 4090 路径 | — | NF4 QLoRA + LoRA + grad-ckpt + 8bit Adam（官方验证） | block swap 叠加 grad-ckpt（anima 独有，待验证） |

**关键收敛点：** 三方全部用 block 级 + `use_reentrant=False`。`use_reentrant=False` 是与 `torch.compile` 共存的**强制前提**（PyTorch issue #106555：reentrant + compile 触发 `COMPILED_AUTOGRAD_POISON` 崩溃）。

## 2. SingleStreamBlock 的 checkpoint 边界

官方 `mmdit.py`（探查副本 `:309-318`）`SingleStreamBlock.forward`：

```python
def forward(self, x, vec, freqs, mask=None):
    prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
    x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs, mask)
    x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
    return x
```

**checkpoint 边界设计：**

- **输入**：`x`（image+text 联合序列，已 cat）+ `vec`（temb）+ `freqs`（RoPE）+ `mask` —— 全部作为 checkpoint 位置参数传入。
- **输出**：`x`（同形状）。
- **modulation**：`self.mod(vec)` 产出 6 chunk，**在 block 内部计算**，recompute 时重算——无副作用，因为 `vec`（temb）是 checkpoint 输入、稳定。无需把 modulation 预算到 checkpoint 外。
- **RoPE**：`freqs` 作为位置参数传入，recompute 时复用。
- **saved 量**：block 输入 `x`（联合序列）+ temb + freqs + mask。recomputation 重算 modulation + attn + mlp。

**与 anima Block 的结构差异：**

| | anima `Block` | Krea-2 `SingleStreamBlock` |
|---|---|---|
| 结构 | self-attn + cross-attn + MLP（三段，dual-stream） | attn + mlp（两段，single-stream，text/image 已 cat） |
| 调制 | AdaLN-LoRA（256 bottleneck + 3 路 up + `adaln_lora_B_T_3D`） | DoubleSharedModulation（6 chunk，轻） |
| 联合序列 | 无（视觉 token 单独，cross-attn 注入文本） | 有（image+text cat 成一条序列，更长） |
| saved 输入大小 | `(B,T,H,W,D)` 视觉序列 | `(B, T_img+T_txt, D)` 联合序列（更长） |

**判断：** anima 的 Block 级 checkpoint 把 self-attn+cross-attn+MLP 一起包。Krea-2 single-stream **没有独立 cross-attn**，block 边界天然就是 attn+mlp 一起——**直接套 anima 的 Block 级 checkpoint 路径，粒度合适**，无需 sub-block 切分。diffusers Krea2 / Flux / SD3 社区实现均未拆 attn/mlp，印证此判断。

## 3. anima 现有机制（迁移基线，已逐行核实）

### 3.1 核心组件

| 组件 | 位置 | 说明 |
|---|---|---|
| `UnslothOffloadedGradientCheckpointer` | `library/anima/models.py:88`（`forward` 95-109，`backward` 111-134） | CPU offload 档位的 autograd.Function |
| `unsloth_checkpoint` | `library/anima/models.py:137-140`（`@torch._disable_dynamo` 在 `:137`） | 包装函数，**带 `@torch._disable_dynamo`**——这是 compile 必须编译 `_forward` 而非 `forward` 的根因 |
| `ADAPTER_AWARE_CHECKPOINT_MAX_SAVE_NUMEL` | `library/anima/models.py:204` | `1_048_576`（1 MiB elements），adapter_aware 档位的小张量保存阈值 |
| `_adapter_aware_checkpoint_policy` | `library/anima/models.py:381-394` | `MUST_SAVE`（小且可训练）/ `PREFER_RECOMPUTE`（默认/无梯度）判定 |
| `Block._forward`（被 checkpoint 包装的内层） | `library/anima/models.py:1432` 起 | 注意 compile 后被替换，原始存于 `block._anima_compile_base_forward`（`:2175`） |
| `Block.forward` checkpoint 分支 | `library/anima/models.py:1662` 起 | 4 分支：adapter_aware(`:1677`) / unsloth(`:1694`) / cpu_offload(`:1718`) / 普通(`:1731`)，全部 `use_reentrant=False`（`:1686,1727,1740`） |
| `Block.enable_gradient_checkpointing` | `library/anima/models.py:1342-1350` | 接受 `cpu_offload` / `unsloth_offload` 标志 |
| `Block.enable_adapter_aware_checkpointing` | `library/anima/models.py:1376-1387` | adapter_aware 档位入口 |
| `Block.enable_mlp_checkpointing` / `enable_mlp_layer1_checkpointing` | `library/anima/models.py:1360-1374` | MLP 子粒度档位（针对 GPT2 FeedForward，Krea-2 SwiGLU 需重写） |
| `Anima.enable_gradient_checkpointing` | `library/anima/models.py:1942-1948` | 聚合到所有 block |
| `Anima.enable_selective_checkpointing` | `library/anima/models.py:2001-2046` | 8 种 mode 分发（off/adapter_aware/every_other/mlp_only/mlp_layer1_only/peak_blocks_*） |
| `Anima.compile_blocks` | `library/anima/models.py:2048` 起（方法体延续至 ~2190+） | **编译 `_forward` 不编译 `forward`**（注释 `:2071-2075`）；block-swap coexistence 注释 `:2077-2084`（默认 `compile_block_scope="resident"`，swap 出去的 tail block 保持 eager） |

### 3.2 加载顺序不变量

`library/runtime/harness.py::build_anima` 6 步顺序（`:75-86` 注释、`:111-200` 实现）：

1. `load_anima_model` → DiT 上设备（`:128-135`）
2. `requires_grad_(False)` + `reset_mod_guidance`（`:136-137`）
3. adapter `apply_to` → `load_weights`（`:153-154`）
4. **grad-ckpt**（`:179-181`，gated on `anima.training`）
5. `train()` / `eval()`（`:183-190`）
6. **compile last**（`:194-200`，注释 `:192-193` "Adapter monkey-patches must be installed first or torch.compile traces the wrong forward"）

`library/training/bootstrap.py` 同序：apply_to(`:561`) → load(`:564`) → grad-ckpt(`:569-581`) → fp32_residual(`:587`) → convrot(`:599`) → compile(`:628`)。注意 `:569-573` 的 `cpu_offload_checkpointing` 分支调 `unet.enable_gradient_checkpointing(cpu_offload=True)`，Krea-2 的 `enable_gradient_checkpointing(self)` 不接受该参数——见 §11.2 兼容缺口。

`enable_training_grad_ckpt`（`harness.py:1041-1053`）**只走 unsloth_offload 路径**（`:1050`），不分支 cpu_offload/普通——三路分支实际在 `Block.forward`（`:1691-1741`）按 block 上的标志自洽选择。

### 3.3 compile 编译 `_forward` 不编译 `forward`

`models.py:2071-2075` 注释原文：

> Compiles `_forward` (the actual attention/MLP computation) rather than `forward` (the checkpointing wrapper). This is critical because `unsloth_checkpoint` has `@torch._disable_dynamo`, which causes an immediate graph break if `forward` itself is compiled — dynamo compiles nothing useful but still checks shape guards, causing recompile storms.

Krea-2 DiT 移植后**必须沿用此结构**：`forward` 做 checkpoint 包装，`_forward` 做实际计算，compile 编译 `_forward`。

### 3.4 compile × block swap coexistence（Krea-2 直接继承）

`models.py:2077-2084` 注释：block swap 开启时，默认 `compile_block_scope="resident"`，**只编译常驻 head blocks，swap 出去的 tail blocks 保持 eager `_forward`**——因为 offloader 通过 `.weight.data` reassignment 换权重，Dynamo 对每个 Parameter 的 dispatch key 做 guard，编译 swapped blocks 会每步 recompile。`compile_block_scope="all"` 可恢复全 block 编译（用于显存极紧、Inductor activation planning 值得 recompile 开销的场景）。

## 4. block swap × grad-ckpt 调度（anima 已正确处理，Krea-2 继承）

这是迁移前最担心的点：被 checkpoint 包裹的 block，backward 需重算它时，其权重是否已被 swap 出 GPU？逐行核实后结论：**anima 现有架构已正确处理，Krea-2 直接继承调度即可**。

### 调度时序

**forward 主循环**（`library/anima/models.py` `_run_blocks`，def 在 `:2389`，循环主体 `:2419-2458`）：

- `:2421` `self.offloader.wait_for_block(block_idx)` —— block 级 forward 前**等待该 block 权重 H2D 完成**
- `:2439-2445` `x = block(x, ...)` —— 实际 forward，checkpoint 在此触发（保存 input 到 CPU / 重算在 backward）
- `:2458` `self.offloader.submit_move_blocks(...)` —— forward 后**提交换出 + 预取下一个**

**backward hook**（`library/runtime/offloading.py:1617-1674`）：

- `:1617-1619` `block.register_full_backward_hook(hook)` —— `register_full_backward_hook` 在该 block 反向**全部结束后**触发，即 recompute 已完成
- `:1665-1671` `backward_prefetch` —— 本 block 反向完成 → 把它换出、把对应早期 block 换入
- `:1672-1673` `backward_wait` —— 等待前一个 block 的 transfer 完成

**预取深度恒为 1**（`offloading.py:1750` `depth = 1`，`:1754` 训练时 forward 不把已被 checkpoint 保护的"未来"block 提前换出）。

**只换 frozen weight，trainable 留 GPU**（`offloading.py:1714-1716`，`include_trainable=False`，注释 "keep adapter/trainable weights on the training device"）。

### 结论

swap out **不会抢先于 recompute**：被 checkpoint 的 block 重算时，其 frozen base 权重仍在 GPU（forward 主循环 `:2421` 已 `wait_for_block` 换入，backward hook `:1665` 在该 block 反向**完成后**才换出）。trainable adapter 权重全程在 GPU。Krea-2 block 结构不同但调度边界相同（block 级 forward → checkpoint → backward recompute → hook swap out），**`offloading.py` 的调度逻辑无需改**，只需让 Krea-2 DiT 复用 `_run_blocks` 主循环或等价编排。

### 测试守护

`tests/test_compile_checkpoint_block_swap_hot.py`：三因子（compile × grad-ckpt × block swap）全组合 8 scenario（`:35-49`，含 `compile_full_checkpoint_block_swap` 三因子全开 `:44-48`），每个跑 `loss.backward()`（`:209`），校验 block_swap profile 含 `phase == "backward_wait"` + `submit_phase == "backward_prefetch"`（`:239-244`）。Krea-2 落地后需补等价矩阵测试。

## 5. 迁移落地清单

### 5.1 直接复用（不用改）

1. **`torch.utils.checkpoint.checkpoint(use_reentrant=False)`** —— Krea-2 同样适用，与 compile 共存前提。
2. **compile-after-grad-ckpt 顺序**（`harness.py:64` 6 步不变量）—— Krea-2 加载器照搬，建议补运行时断言防乱序（见 [`krea2_raw_migration_notes.md`](krea2_raw_migration_notes.md) R3）。
3. **compile 编译 `_forward` 不编译 `forward`** —— Krea-2 DiT 必须沿用 `forward`(checkpoint 包装)/`_forward`(实际计算)分离结构。
4. **`UnslothOffloadedGradientCheckpointer`（CPU offload 档位）** —— 13B 下显存更吃紧，offload 档位价值更高，直接复用 `models.py:88-140` 的 autograd.Function。
5. **block swap 调度**（`offloading.py`）—— `_run_blocks` 主循环 + backward hook + 预取深度 1 + `include_trainable=False`，Krea-2 复用编排，调度逻辑不动。
6. **测试矩阵模式**（`tests/test_compile_checkpoint_block_swap_hot.py` 8 scenario）—— 套用到 Krea-2 block。

### 5.2 要改（适配 Krea-2 block 结构）

1. **`SingleStreamBlock._forward` 重写**：attn + mlp 两段（替代 anima 的 self + cross + mlp 三段）。checkpoint 包装层结构不变。modulation（`self.mod(vec)`）在 `_forward` 内部算，无需 anima 的 `adaln_lora_B_T_3D` 注入参数。
2. **checkpoint 调用签名**：`torch.utils.checkpoint.checkpoint(self._forward, x, vec, freqs, mask, use_reentrant=False)`。`vec`/`freqs`/`mask` 作为位置参数传入，recompute 时复用。
3. **MLP 子粒度档位重写或删除**：anima 的 `mlp_only` / `mlp_layer1_only` 针对 GPT2 FeedForward（layer1+gelu+layer2）。Krea-2 用 SwiGLU（`gate`/`up`/`down` 三 Linear + silu 门控），子粒度档位语义不同。**首日只用 block 级即可**，子档位留待按需加。
4. **`adapter_aware` selective policy 重新评估**：anima 的 `MUST_SAVE` 列表针对 AdaLN-LoRA + Hydra 路由设计的。Krea-2 light bias modulation 中间量更小更少，`ADAPTER_AWARE_CHECKPOINT_MAX_SAVE_NUMEL=1_048_576`（`models.py:204`）阈值可能要调。**首日只 plain LoRA，整个 `adapter_aware` 档位可省**，后续按需启用。
5. **3D RoPE → Krea-2 3-axis Axial RoPE**：`freqs` 作为位置参数传 checkpoint 边界，比 anima 的 `rope_cos_sin` tuple 更简单（直接 tensor）。anima 的 `_make_dynamic_seq_forward`（`models.py:154`，recompute-safe `mark_dynamic`）若 Krea-2 仍用动态 seq 则复用，固定 bucket 则删。
6. **`compile_block_scope` 默认值**：Krea-2 13B 更可能开 block swap，默认 `"resident"`（只编译常驻 block）语义沿用；若 Krea-2 bucket 收敛到更少 token-count family，可考虑 `"all"`。

### 5.3 三方社区都没做、anima 独有的优势

**block swap + grad-ckpt 叠加** —— diffusers `train_dreambooth_lora_krea2.py` 和 ai-toolkit 都**没有 block swap**（模型权重整体驻留 GPU 或 NF4 量化）。anima 训练器有成熟 block swap 体系（`ANIMA_BLOCK_SWAP_*`、`restore_mode=slab`、prefetch depth=1、backward hook 调度）。

- grad-ckpt 压 **activation** 显存。
- block swap 压 **模型权重** 显存。
- 二者叠加是社区独创路径，**比 diffusers 官方更激进**，且 §4 已确认调度兼容。

## 6. 24GB 4090 训练可行性

13B bf16 权重 ~26GB，**裸放就超 24GB**。可行路径：

| 方案 | 模型权重显存 | 可行性 | 来源 |
|---|---|---|---|
| **NF4（bnb）+ LoRA + grad-ckpt + 8bit Adam + offload VAE/TE** | ~6.5GB | ✅ **diffusers 官方验证可行**，24GB mid-range GPU | Krea-2 README |
| FP8 训练（需 SM≥8.9）+ LoRA + grad-ckpt | ~13GB | ⚠️ 可能可行，README 未给确切数字 | diffusers |
| bf16 + grad-ckpt + LoRA + block swap | 26GB（可 swap 出去） | ⚠️ 需验证 block swap 与 grad-ckpt 调度兼容性（§4 已理论确认，待实测） | anima 独有路径 |

**建议：** anima 训练器走第三条路（block swap 是其强项），但首日优先实现 NF4 路径作为保底（diffusers 官方验证过）。block swap 叠加 grad-ckpt 作为优化目标，阶段 4 用 `tests/test_compile_checkpoint_block_swap_hot.py` 等价矩阵 + `peak_probe.py` 实测验证。是否新增 `low_vram_krea2` preset 见 [`krea2_raw_migration_notes.md`](krea2_raw_migration_notes.md) R7。

## 7. 阶段 4 验证项

落地后必须验证（对应 [`krea2_raw_migration.md`](krea2_raw_migration.md) 阶段 4 退出条件）：

1. **block 级 grad-ckpt 单 forward 通过**：`SingleStreamBlock` 加 `enable_gradient_checkpointing`，`torch.utils.checkpoint.checkpoint(self._forward, x, vec, freqs, mask, use_reentrant=False)` 跑通 forward + backward。
2. **compile × grad-ckpt 共存**：compile 编译 `_forward`，grad-ckpt 开启，无 `COMPILED_AUTOGRAD_POISON` 崩溃，无 recompile storm。
3. **block swap × grad-ckpt × compile 三因子矩阵**：套 `tests/test_compile_checkpoint_block_swap_hot.py` 8 scenario，校验 `backward_wait` / `backward_prefetch` 事件。
4. **recompute 时权重驻留实测**：开 block swap + grad-ckpt，确认被 checkpoint 的 block 重算时其 frozen 权重在 GPU（§4 理论已确认，实测兜底）。
5. **显存峰值实测**：`library/runtime/peak_probe.py` 跑 NF4 路径 vs block swap 路径，确认 24GB 4090 可承载。
6. **单 prompt 过拟合 loss 下降**：grad-ckpt 开启下 loss 应与 eager 同样单调下降（grad-ckpt 不改数学，只改显存）。

## 8. 风险与失败模式

### F1: `use_reentrant=True` 误用 → compile 崩溃

**触发：** 移植时误把 `use_reentrant=True` 或漏写 `use_reentrant=False`。
**现象：** `COMPILED_AUTOGRAD_POISON` 崩溃（PyTorch issue #106555）。
**缓解：** 所有 `torch.utils.checkpoint.checkpoint` 调用显式 `use_reentrant=False`；`Block.forward` 4 分支（`models.py:1686,1727,1740`）逐一对齐。

### F2: compile 编译 `forward` 而非 `_forward` → recompile storm

**触发：** Krea-2 DiT 移植时未保留 `forward`/`_forward` 分离。
**现象：** dynamo 在 `unsloth_checkpoint` 的 `@torch._disable_dynamo` 处 graph break，但仍检查 shape guards → recompile storm，编译无收益。
**缓解：** 严格沿用 anima 的 `forward`(checkpoint 包装)/`_forward`(实际计算)分离，compile 编译 `_forward`（`models.py:2071-2075` 注释）。

### F3: block swap 与 grad-ckpt 调度竞态（理论已排除，实测兜底）

**触发：** block swap 调度逻辑改动，或 Krea-2 DiT 不复用 `_run_blocks` 主循环。
**现象：** recompute 时 block 权重已被 swap 出 GPU → 假权重或 NaN。
**缓解：** §4 调度时序已确认 swap out 不会抢先于 recompute；Krea-2 复用 `offloading.py` 调度，不动 `_run_blocks`；`tests/test_compile_checkpoint_block_swap_hot.py` 矩阵 + `peak_probe.py` 实测兜底。

### F4: 24GB 4090 即便 NF4 + block swap 仍 OOM

**触发：** 13B 量化后 + LoRA + optimizer state + activation + VAE/TE 残留超 24GB。
**现象：** OOM。
**缓解：** 缩小 LoRA rank / 限制可训层 / offload VAE/TE 到 CPU / 评估是否需 int8 base（Krea-2 官方无 int8 base，自建工程量大，见 [`krea2_raw_migration_notes.md`](krea2_raw_migration_notes.md) R7）。

### F5: `adapter_aware` 档位阈值不适配 Krea-2 modulation

**触发：** 启用 `adapter_aware` selective 后，Krea-2 light bias modulation 的中间量保存策略错误。
**现象：** 显存未省或重算过多导致训练变慢。
**缓解：** 首日只 plain LoRA + block 级 standard checkpoint，`adapter_aware` 档位首日不启用；后续按需调 `ADAPTER_AWARE_CHECKPOINT_MAX_SAVE_NUMEL` 阈值。

## 9. 与其他提案/文档的关系

- **[`krea2_raw_migration.md`](krea2_raw_migration.md)**：本文件是其阶段 4（训练串通）的 grad-ckpt 子设计。阶段 4 退出条件"单 prompt 过拟合 loss 下降"依赖本文件的 §5 落地清单和 §7 验证项。
- **[`krea2_raw_migration_notes.md`](krea2_raw_migration_notes.md)**：
  - R3（compile after apply 无运行时防呆）—— 本文件 §5.1 第 2 条建议补 `_applied` 断言，与 R3 同步受益。
  - R7（13B 显存可行性）—— 本文件 §6 的三条显存路径是 R7 的具体化。
- **[`../multi_model_support.md`](../../../docs/multi_model_support.md)**：`forward_for_loss` 承重接口由 family 实现，grad-ckpt 是 `forward_for_loss` 内部细节，不跨 family 边界。
- **`docs/optimizations/for_compile.md`**：compile × grad-ckpt 的通用结构（编译 `_forward` 不编译 `forward`、`use_reentrant=False`）在此文档记录，本文件是其在 Krea-2 上的具体化。

## 10. 不在本文件范围

- Krea-2 DiT 架构本体移植细节（见 [`krea2_raw_migration.md`](krea2_raw_migration.md) 阶段 2 + [`krea2_raw_migration_notes.md`](krea2_raw_migration_notes.md) R4 LoRA 注入点）。
- NF4 量化的具体实现（diffusers `prepare_krea2_model_for_4bit` 路径，落地时参照 diffusers `train_dreambooth_lora_krea2.py`，不在本文件展开）。
- text encoder（Qwen3-VL）侧的 grad-ckpt（anima 在 `bootstrap.py:566-571`/`:911-923` 有 text-encoder grad-ckpt workaround，Krea-2 文本链路见 [`krea2_raw_migration.md`](krea2_raw_migration.md) 阶段 1，本文件只覆盖 DiT block 级）。
- ConvRot / Turbo / DCW 等与 grad-ckpt 的交互（Krea-2 首日不做这些，见 [`krea2_raw_migration.md`](krea2_raw_migration.md) §1 非目标）。

## 11. 落地核对（2026-08-09 补）

本节对照 §5 落地清单与实际代码，记录分叉。原文档 §1-§10 是落地前的调查设计稿；实际落地走了更保守的策略，下列条目以实际代码为准。

### 11.1 已落地（与 §5.2 一致）

- **`SingleStreamBlock._forward` 重写**（§5.2.1）：`library/models/krea2_raw/dit.py:367-375`，attn+mlp 两段，`self.mod(vec)` 在 `_forward` 内部计算，无 anima 的 `adaln_lora_B_T_3D` 注入参数。
- **checkpoint 调用签名**（§5.2.2）：`dit.py:385-392`，`torch_checkpoint(self._forward, x, vec, freqs, mask, use_reentrant=False)`，`vec`/`freqs`/`mask` 作位置参数。全文件唯一 checkpoint 调用，`use_reentrant=False` 贯穿。
- **RoPE/freqs 边界**（§5.2.5）：`freqs` 作位置参数传入（`dit.py:389`），3-axis Axial RoPE（`PositionalEncoding` `:166-180`，axes `[96,16,16]` @ headdim=128）。固定 bucket，`_make_dynamic_seq_forward`/`mark_dynamic` 未移植——符合 §5.2.5 "固定 bucket 则删"。
- **MLP 子粒度省略**（§5.2.3）+ **adapter_aware 省略**（§5.2.4）：`dit.py:357-358` 注释自述"首日只标准 grad-ckpt（无 cpu_offload/unsloth/adapter-aware 变体）"，符合 §5.2.3/§5.2.4 "首日省略"建议。
- **block swap 调度复用**（§5.1.5）：`dit.py:506-527` `enable_block_swap` 复用 `library.runtime.offloading.ModelOffloader`；`:561-580` `_run_blocks` 移植自 anima，调度时序与 §4 一致（`wait_for_block` → block forward → `submit_move_blocks`）。

### 11.2 与设计的分叉

- **compile 路径全程 eager**（§5.1.3/§5.2.6/§7.2/§7.3 不适用）：`SingleStreamDiT` **没有 `compile_blocks` 方法**（`dit.py:5-6` 注释说"anima 的 compile_blocks() 统一编译"是设计意图，但未实现），`library/runtime/harness.py:766` 用鸭子类型 `if not hasattr(unet, "compile_blocks"): return` 跳过整个 compile 序列（commit `b7afaba5` 固化）。`_forward` 仅作 grad-ckpt 的 recompute 函数，不被 compile。因此 §7.2 "compile × grad-ckpt 共存"、§7.3 "compile × grad-ckpt × block swap 三因子矩阵"对 Krea-2 **不适用**，三因子退化为 grad-ckpt × block swap 二因子。`compile_block_scope="resident"` 默认值在 `harness.py:721`/`bootstrap.py:650-651` 保留，但对 Krea-2 不生效。
- **`UnslothOffloadedGradientCheckpointer` 未复用**（§5.1.4）：Krea-2 走标准 `torch.utils.checkpoint`，未移植 anima 的 CPU offload 档位 autograd.Function。
- **compile-after-grad-ckpt 运行时断言未实现**（§5.1.2/R3）：顺序正确（`bootstrap.py:569` grad-ckpt → `:628` compile），但只有注释，无 `assert network._applied` 之类防呆。
- **`cpu_offload` 签名缺口**（§8 风险清单未覆盖）：`bootstrap.py:570-571` 通用分支 `if args.cpu_offload_checkpointing: unet.enable_gradient_checkpointing(cpu_offload=True)` 对 Krea-2 不兼容——`SingleStreamDiT.enable_gradient_checkpointing(self)`（`dit.py:491`）只收 `self`。用户若对 Krea-2 同时传 `--gradient_checkpointing --cpu_offload_checkpointing` 会 `TypeError: unexpected keyword argument 'cpu_offload'`。默认路径（`:573` 无参调用）正常。缓解：Krea-2 preset/文档应明确禁用 `cpu_offload_checkpointing`，或在 DiT 侧加 `**kwargs` 兜底吸收。

### 11.3 实测结论

阶段 6 配置收口里程碑：`forward_for_loss` + `model_family` 正式串通 `train.py`，1024×1024 grad-ckpt 真实 flow-matching LoRA 训练实测通过——loss 0.465→0.198（50 步），显存 peak 27.9GB（PG199 32GB bf16，稳态 24.5GB，余量 ~4GB），step 3.47s/it，checkpoint 92MB。本次为 **grad-ckpt on, swap off**，未叠加 block swap 实测（二者正交，findings 文档明确未叠加）；未测 NF4 路径（仅 §6 作为 24GB 4090 保底方案提及）。详见 [`../findings/krea2_raw_migration_stage6_findings.md`](../../../docs/findings/krea2_raw_migration_stage6_findings.md)。

### 11.4 行号订正

本文档 anima 侧引用基于 2026-08-08 main，落地后部分文件局部增长导致系统性偏移，已订正：`bootstrap.py` 引用组整体 +9 行（apply_to `:561`/load `:564`/grad-ckpt `:569-581`/fp32_residual `:587`/convrot `:599`/compile `:628`）；`harness.py::enable_training_grad_ckpt` +16 行（`:1041-1053`）；另 `_adapter_aware_checkpoint_policy` 末行 `:394`、`_run_blocks` def 起点 `:2389`、测试 backward 校验段 `:239-244`。结构性/顺序性断言全部仍然正确。
