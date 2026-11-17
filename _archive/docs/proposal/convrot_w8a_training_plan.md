# ConvRot W8A 训练路径正式规格书（战略 C）

状态：提案 / **核心已实现（实验默认关闭）**
适用版本：当前 main（设计冻结 + 2026-07-24 落地）
日期：2026-07-24
入口命令：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_convrot_*.py -q
.venv/bin/python scripts/experiments/convrot_equivalence_probe.py --mode w8a16
python tasks.py lora ... --base_compute w8a16_convrot   # 显式开启
```

相关代码：

- `library/runtime/convrot/`（**已实现**）
- `library/training/bootstrap.py::maybe_apply_convrot_base`
- `library/training/cli_args.py`（`base_compute` / `convrot_*`）
- 对照（非 ConvRot）：`library/runtime/int8_linear.py`、block-swap int8 传输

相关文档：

- 研究记录：[`../experimental/convrot_int8_training.md`](../experimental/convrot_int8_training.md)
- **后续优化路线图：** [`convrot_w8a_optimization_roadmap.md`](convrot_w8a_optimization_roadmap.md)
- 旧 int8 审计：[`../findings/anima_int8_base_linear_audit.md`](../findings/anima_int8_base_linear_audit.md)
- adapter 侧预缩放：[`../methods/channel_scaling.md`](../methods/channel_scaling.md)
- outlier 证据：`bench/channel_stats/`

> **一句话：** 在 DiT 冻结 base 上自建 **group-wise Regular Hadamard（ConvRot）+ W8A16 / W8A8** 训练路径，挂在现有 LoRA `org_forward` 补丁缝上。
> **默认 `base_compute=bf16`**；开启后不要与 block-swap int8 传输混淆。

---

## 0. 一句话目标

用户（或实验者）显式打开 **`base_compute=w8a16_convrot`（或后续 `w8a8_convrot`）** 后，系统在 **LoRA 已 apply、compile 之前** 将选定冻结 base Linear 的 `org_forward` 替换为 **旋转域 int8 权重路径**，使：

```text
y = base_w8a*(x) + lora_delta(x)
```

其中 base 路径处理 DiT outlier（group RHT），adapter 在**原输出空间**残差相加、正常反传；默认配置与 WebUI 主路径保持 **bf16**，不改变未开启用户的数值与行为。

---

## 1. 非目标（全期明确不做）

1. **不把** `Int8FrozenLinear` / 现有 `int8_linear` **贴牌**成 ConvRot。
2. **不把** `block_swap_transfer_dtype=int8` 当训练质量或 W8A* 方案。
3. **不默认**写入 `configs/base.toml`；默认始终 `base_compute=bf16`。
4. **不**对 AdaLN / modulation / final_layer / embedder / TE / router 做 int8。
5. **不**在第一期做真 int8 Tensor Core kernel 或 Triton 融合（可用 fake-quant + STE + dequant×bf16）。
6. **不**把 Comfy 推理 quant 权重格式直接当训练计算图默认。
7. **不**在热点文件（`train.py`、`networks/lora_anima/network.py` 等）堆大块业务；新逻辑进 `library/runtime/convrot/`。
8. **不**默认长训、下载大模型、推送；实现与验证需用户明确授权时再做。
9. **不**承诺跨机/跨 GPU 速度收益；第一期成功标准是**数值与短训 sample**，不是 FPS。
10. **不**与路线 B（ARA：冻结任意 quant base + 小 LoRA 补误差）混成同一开关。

---

## 2. 术语

| 术语 | 含义 |
| --- | --- |
| **ConvRot** | Group-wise Regular Hadamard Transform（RHT）后再 quant；名称中的 Conv **不是**卷积层量化 |
| **group size** | RHT 分组长度；候选 `64 / 256 / 1024`，须整除 `in_features` |
| **W8A16** | 权重 int8（旋转后）；激活 bf16/fp16 |
| **W8A8** | 权重 int8；激活**动态** int8；目标 int8×int8 GEMM |
| **W4A4** | 论文主推推理格式；**本规格训练路径不做** |
| **org_forward** | LoRA monkey-patch 捕获的 base Linear 原始 forward |
| **online_from_bf16** | 从当前冻结 bf16 权重在线 RHT+quant，不依赖社区 prequant 文件 |
| **prequant_checkpoint** | 加载社区/自建旋转 int8 权重（Phase 0/可选） |
| **fake path** | 用高精度算子模拟 quant 截断 + STE，不调用 int8 GEMM |
| **战略 A/B/C** | A=仅推理对接；B=ARA 冻结 quant base；**C=本规格真 W8A* 训练路径** |
| **路径 A/B（现状）** | A=`int8_linear` org_forward 补丁（probe only）；B=block-swap int8 传输（训练可开） |

补充：

- “真 W8A8 训练”= forward 中存在 act quant + int8 GEMM（或等价 fake 路径且接口预留），**不是** dequant 后永远 `F.linear` 却命名为 W8A8。
- “存储 int8”≠“计算 int8”。

---

## 3. 背景与问题陈述

### 3.1 旧 int8 失败根因（本仓证据）

| 证据 | 结论 |
| --- | --- |
| `bench/channel_stats/`：base 输入通道 dominance mean 16.9 / max **96×** | DiT 存在强 **DC-bias outlier feature**（非 attention sink） |
| 主导通道在 base / LoRA r=32 / r=64 几乎同分布 | **frozen base 属性**，adapter 推不动输入分布 |
| 社区 Anima latent 质量序 | `GGUF Q8 > INT8 ConvRot > … ≥ INT8 Row > INT8 Tensorwise` |
| `Int8FrozenLinear.forward` | **dequant → `F.linear`**，高精度 GEMM |
| `anima_int8_base_linear_audit.md` | 权重 L2 偶发 PASS ≠ adapter grad / 多 seed / 短训可用 |

**归纳：** 以前效果差，很大概率不是“int8 本身不行”，而是 **裸 rowwise 未处理 DiT outlier**，且现有路径根本不是低比特 GEMM 训练。

### 3.2 为何必须战略 C

| 路线 | 内容 | 是否解决训练 W8A* |
| --- | --- | --- |
| A | 文档 + 可选加载社区 int8-ConvRot 推理 | 否 |
| B | 冻结 quant base + LoRA 补误差（ARA 类） | 部分；可能只是补 quant 误差 |
| **C** | group RHT + W8A16→W8A8 + 训练 hook | **是（本规格）** |

公开生态（Comfy ≥0.27 native int8 ConvRot、convert_to_quant、公开 Anima INT8-ConvRot 权重）成熟于 **load / forward / serialize**，不是 `train.py`。本机 Comfy 副本若仍为 0.22.x，则**连推理 native ConvRot 都没有**，更不能依赖本地抄实现。

### 3.3 Anima 维度友好性（已核实源码默认）

| 量 | 默认 | group 64/256/1024 |
| --- | --- | --- |
| `model_channels` | 2048 | 均可整除 |
| `mlp_ratio` → `d_ff` | 4 → 8192 | 均可整除 |
| `self_attn.qkv_proj` out | 6144 | 均可整除 |
| `cross_attn.kv_proj` | 1024→4096 | 均可整除 |

---

## 4. 成功标准与 Gate

### 4.1 产品级成功标准（放行）

| 阶段 | 必须满足 |
| --- | --- |
| **Phase 1 放行** | mlp-only、`w8a16_convrot`：adapter grad 多 seed 稳定；短训 sample **接近**同超参 bf16 LoRA；默认 bf16 路径零回归 |
| **Phase 2 放行** | `w8a8_convrot` 独立开关默认 off；上述质量门 + 可选显存/速度 KPI 文档化（不阻塞 Phase 1） |
| **稳定方法** | 迁 `docs/methods/` 前需：用户文档、metadata、merge 策略、与 block-swap/compile 兼容说明 |

**禁止**仅用权重反量化 L2 或单 batch probe PASS 作为放行依据。

### 4.2 工程 smoke 阈值（可复用旧习惯，不足作放行）

| 工具类 | 建议 smoke |
| --- | --- |
| 权重 / 旋转 round-trip | RHT 正交误差 &lt; 数值容差；quant 后 dequant 相对误差可记录 |
| toy / tiny 图 | output rel L2 ≤ **3%**（对齐旧 int8 probe） |
| adapter grad | grad norm rel ≤ **5%**（smoke）；多 seed 不得频繁爆阈值 |
| 旧 audit | 权重 rel L2 p95 &lt; **2%** 仅作存储审计，**不**等于训练可用 |

### 4.3 回归不变量（任何 MR 必须）

1. 未设置 `base_compute` / 默认 bf16：行为与现 main 一致。
2. AdaLN / final / embedder 从未被 patch。
3. `compile_blocks` 仅在 patch 之后执行。
4. 保存的 adapter safetensors 不含 base quant payload（除非显式设计导出工具）。
5. merge 对 ConvRot base 默认拒绝或要求 dequant 路径。

---

## 5. 用户面配置

### 5.1 配置项（已接入 CLI / schema 自动发现；默认 bf16）

| Key | 类型 | 默认 | 必显 | 说明 |
| --- | --- | --- | --- | --- |
| `base_compute` | enum | `"bf16"` | 是（实验区） | `"bf16"` \| `"w8a16_convrot"` \| `"w8a8_convrot"` |
| `convrot_group_size` | int | `256` | 否（高级） | `64` \| `256` \| `1024` |
| `convrot_scope` | str | `"mlp"` | 否（高级） | 同 `int8_linear` scope 语法：`mlp` / `attention` / `all` / 组合 |
| `convrot_weight_source` | enum | `"online_from_bf16"` | 否（高级） | `"online_from_bf16"` \| `"prequant_checkpoint"` |
| `convrot_prequant_path` | path | 空 | 否 | 仅 `prequant_checkpoint` 时使用 |

### 5.2 CLI

- 挂在 `library/training/cli_args.py` 实验参数组。
- 用户文档写：`python tasks.py lora ... --base_compute w8a16_convrot`。
- **禁止**与 `--block_swap_transfer_dtype int8` 共用别名或互相覆盖。

### 5.3 WebUI（MVP 已落地）

- 位置：**配置** Tab → 分类「优化」→ 区块「显存与速度优化」→ block swap 簇下方。
- 字段：`base_compute`（始终可见，默认 `bf16`）、`convrot_group_size` / `convrot_scope`（仅非 bf16 时展开；scope 固定 `mlp`）。
- 文案标明实验、非默认；live 兼容提示与 `block_swap_transfer_dtype=int8` 互斥。
- 未暴露：prequant 路径、Hadamard 种类、kernel env（仍用 CLI / 环境变量）。

### 5.4 UI / 日志文案（中文）

- 开关标题：`ConvRot 冻结 base（W8A16）`
- 帮助：在冻结 DiT Linear 上对权重做分组 Hadamard 旋转后再 int8 存储/计算，用于缓解裸 int8 在 DiT 上的 outlier 问题；默认关闭；不等于 block-swap 传输压缩。
- 启动日志：`[convrot] mode=w8a16 scope=mlp group=256 patched=N skipped=M`

---

## 6. 架构与模块边界

### 6.1 新建目录（强制）

```text
library/runtime/convrot/
  __init__.py
  rht.py                 # group Regular Hadamard
  quant.py               # weight/act int8 + scale；STE 接口
  linear_w8a16.py        # 旋转域 int8 存；W8A16 forward
  linear_w8a8.py         # online act RHT+quant + int8 path（二期）
  apply.py               # apply_convrot_to_lora_network
  scope.py               # re-export / thin wrap int8_linear scope helpers
  metadata.py            # ss_* stamp、merge 拒绝辅助
  checks.py              # compile 后调用 raise、group 整除 assert
```

**禁止**把大块逻辑写入：`train.py`、`library/training/bootstrap.py`（仅薄钩子）、`networks/lora_anima/network.py`、`library/inference/generation.py`。

### 6.2 复用（不要重写）

| 资产 | 用途 |
| --- | --- |
| `classify_frozen_linear_module` / `MLP_*` / `ATTENTION_*` | scope 白名单 |
| `patch_lora_frozen_base_forwards_with_int8` 控制流 | 遍历 `unet_loras`、改 `org_forward`、挂 buffer |
| `create_and_apply_network` 时序 | apply → load → grad-ckpt → fp32 residual → **patch** → compile |
| `int8_*_probe` / gate 习惯 | equivalence 脚本分层 |
| `bench/channel_stats/` | outlier 动机证据 |

### 6.3 必须新建

| 组件 | 原因 |
| --- | --- |
| group RHT | 本仓零实现 |
| 旋转域 quant 存储 | 不同于裸 weight absmax |
| W8A16 / W8A8 forward + STE | 现路径永远 dequant 后 `F.linear` 且无旋转 |
| bootstrap / CLI 接线 | 主路径零调用 |
| 全变体 base 句柄策略 | 见 §8 兼容债 |
| ConvRot 专用测试与 probe | 旧 audit 不够 |

---

## 7. 集成时序（唯一安全挂点）

### 7.1 训练主路径

```text
load_target_model
  └─ enable_block_swap(...)          # 可选；与 ConvRot 语义分离

create_and_apply_network  (library/training/bootstrap.py)
  ├─ create_network*
  ├─ network.apply_to(...)           # ★ org_forward 捕获
  ├─ network.load_weights(...)       # 可选；仅 adapter
  ├─ gradient_checkpointing
  ├─ maybe_enable_fp32_residual      # 若 fp16
  ├─ ★ apply_convrot_to_lora_network(...)   # 仅当 base_compute ∈ {w8a16,w8a8}_convrot
  └─ compile_blocks_for_training(...)       # COMPILE LAST

setup_optimizer → accelerator.prepare → train loop
  └─ apply_router_conditioning / set_fei   # 与 base act rotate 默认正交
```

**精确缝：** `bootstrap.py` 中 `load_weights` + grad-ckpt + fp32 residual **之后**、`compile_blocks_for_training` **之前**（约现 428 行附近）。

### 7.2 推理 / harness

`library/runtime/harness.py::build_anima` 编码同一 compile-last 不变量。若推理侧需要加载 W8A* 训练出的 adapter：

- 默认 **不需要** base ConvRot（adapter 在原空间）；
- 若要做 Phase 0 社区权重对齐，用独立 loader，**不要**塞进默认 `build_anima`。

### 7.3 monkey-patch 语义（硬约束）

```text
apply_to:
  lora.org_forward = Linear.forward
  Linear.forward = lora.forward

lora.forward(x):
  return lora.org_forward(x) + lora_delta(x)
```

ConvRot **只替换 `lora.org_forward`**。
**禁止**在 `apply_to` 之后 `setattr` 替换 child `nn.Linear`（会绕过 LoRA 链）。
`replace_frozen_base_linears_with_int8` 仅允许无 adapter 的探针场景。

---

## 8. 兼容债与变体矩阵

### 8.1 `org_module_ref` 覆盖（开工前必须认清）

| 模块族 | `org_module_ref` | Phase 1 |
| --- | --- | --- |
| `LoRAModule` / DoRA / plugins（LoHa/LoKr/VeRA/GLoRA） | 有 | **支持**（DoRA 见下） |
| Hydra / Ortho / StackedExperts / FeRA 等 | 常无（`apply_to` 后 `org_module` 已 del） | **默认跳过或后续工单** |
| ReFT | 包 Block.forward，非 Linear | **不在 scope** |

**Phase 1 默认支持面：** plain `LoRAModule`（`configs/methods/lora.toml` / gui `lora`）。
**后续：** 统一在 `apply_to` 时保留 `org_module_ref`，或从 `org_forward.__self__` 安全取 Linear。

### 8.2 DoRA

DoRA 直接读 `org_module_ref[0].weight`。仅改 `org_forward` 可能导致 magnitude 路径与 quant base 不一致。

- Phase 1：**排除 DoRA** 或检测到 DoRA 时拒绝启用 `base_compute=*_convrot` 并给出明确错误。
- 后续：DoRA 读 dequant 权重或禁用 quant 路径上的 DoRA。

### 8.3 fuse / merge / bake

- `fuse_weight` / `merge_to` / `tasks.py merge` 默认假设高精度可写 `weight`。
- ConvRot 开启时：merge **拒绝**或要求先 dequant 回 bf16 再折叠。
- 文档与 `metadata.py` 必须同步。

### 8.4 block-swap int8

| 项 | 现状 | 与 ConvRot |
| --- | --- | --- |
| CLI | `block_swap_transfer_dtype` | 保留；**禁止别名** |
| scope / restore | env `ANIMA_BLOCK_SWAP_INT8_*` | 与 `convrot_scope` 无关 |
| 叠用 | 双 dequant 语义混乱 | Phase 1：**互斥**（开启 convrot 时拒绝 transfer_dtype=int8，或文档强制 bf16 transfer） |

### 8.5 FEI / Hydra router

| 组件 | 与 online act rotate |
| --- | --- |
| `set_fei` / FEI | 读 4D latent；与 token Linear 旋转 **正交** |
| `router_source=fei/sigma` | 与 act rotate **默认不冲突** |
| `router_source=input` | W8A8 时 router 仍应读**未旋转** `x`；禁止误改 |

Phase 1 W8A16 可与 FEI Hydra 共存（若该变体能拿到 base 句柄）；W8A8 + `router_source=input` 需专项测试。

### 8.6 channel_scaling

- 只缩放 **adapter 输入**（`inv_scale` + `lora_down`），**不是** base int8 GEMM 预处理。
- 允许与 ConvRot 共存，但文档写清：二者坐标系不同，**禁止**实现成“双重旋转”。

### 8.7 5D latent

- DiT 边界：`unsqueeze(2)` / `squeeze(2)`；Linear 只看最后一维 D。
- ConvRot 不改变 `in_features`；FEI 仍在 unsqueeze 前 4D 计算。

### 8.8 torch.compile

| 风险 | 缓解 |
| --- | --- |
| Trace 时机 | 只在 compile 前 patch |
| graph break | Phase 1 纯 tensor + `F.linear` |
| CUDAGraph | W8A8 初期关或 skip dynamic |
| block-swap + compile | 仅 resident 或与 convrot 互斥 |

---

## 9. 算法规格

### 9.1 Group RHT

- 输入：最后一维可被 `group_size` 整除的向量/矩阵维。
- 对 **in-feature** 维按 group 做 Regular Hadamard（正交、可逆、`R^{-1}=R^T` 在适当归一化下）。
- 复杂度：group-wise \(\mathcal{O}(K)\) 宣称（相对全维 \(\mathcal{O}(K^2)\)）。
- 实现：`rht.py`；单测覆盖正交、可逆、batch shape、非整数除报错。

### 9.2 权重量化（旋转后）

```text
W_rot = RHT_in(W)          # 按 in_features 分组旋转
scale_i = amax(|W_rot|_{i,:}) / 127
W_q = round(W_rot / scale).clamp(-127, 127).to(int8)
```

- per-**output-channel** absmax（与现 `quantize_weight_per_channel` 同形，但输入是旋转后权重）。
- bias 必须为 `None`；`requires_grad` 必须为 `False`。

### 9.3 W8A16 forward（Phase 1）

社区定义对齐：

```text
INT8 ConvRot ≈ row-wise INT8，但 quant 前对 weight/activation 做同 group 旋转
```

推荐数值路径（可 fake）：

```text
x_rot = RHT_in(x)                    # act 在线旋转，仍 bf16
W_hat = dequant(W_q, scale)          # bf16/fp32
y = F.linear(x_rot, W_hat)           # 或等价 matmul
# y 处于原输出空间（Hadamard 正交性保证与「W 旋转 + x 旋转」一致）
```

可选 baseline（仅调试）：dequant 后乘 `R^T` 回原域再 `F.linear(x, W_orig)`——必须与上式单测等价。

**STE：** 反传穿过 quant 时对 weight 用 STE 或 stop-grad（base 冻结时 stop-grad 即可）；adapter 路径不受影响。

### 9.4 W8A8 forward（Phase 2）

```text
x → RHT → dynamic absmax quant → int8×int8 GEMM(W_q) → dequant → y
```

- 默认 off。
- 第一实现允许 fake int8 matmul；接口与 buffer 布局按真 GEMM 预留。
- 独立 gate；初期建议 `torch_compile=false` 或限制 backend。

### 9.5 Scope 白名单 / 黑名单

**可候选（与 `int8_linear` 一致）：**

- MLP：`mlp.layer1`、`mlp.layer2`
- Attention：`self_attn.{qkv,q,k,v,kv,output}_proj`、`cross_attn.{q,k,v,kv,output}_proj`

**硬排除：**

- 一切 `adaln_*` / `*_modulation*`
- `final_layer.*`
- `x_embedder` / `t_embedder` / `pooled_text_proj` / `llm_adapter`
- 任意 trainable / `bias is not None`
- LoRA / router / FEI / guidance 参数

**Phase 1 默认 scope：`mlp`。**

### 9.6 残差相加域

```text
y = base_convrot(x) + lora_delta(x)
```

- `lora_delta` 始终在**原特征空间**计算（`lora_down(x)` 等）。
- **不要**对 `lora_down` 输入强制 RHT，除非整条 adapter 路径重新定义（本规格禁止）。

---

## 10. `apply_convrot_to_lora_network` 规格

```python
def apply_convrot_to_lora_network(
    network: nn.Module,
    *,
    mode: Literal["w8a16", "w8a8"],
    scope: str = "mlp",
    group_size: int = 256,
    weight_source: str = "online_from_bf16",
    prequant_path: str | None = None,
    dry_run: bool = False,
) -> list[ConvRotLoRABaseForwardPatch]:
    ...
```

行为：

1. 若检测到已 `compile_blocks`，**raise**。
2. 遍历 `network.unet_loras`（及明确支持的变体列表）。
3. 用 `original_name` + `classify_frozen_linear_module` 过滤。
4. 解析 base Linear：优先 `org_module_ref[0]`；无 ref 则按策略跳过并计入 `skipped`（Phase 1 不静默成功）。
5. 校验 `in_features % group_size == 0`、bias=None、frozen。
6. `online_from_bf16`：RHT → quant → `register_buffer` 到 lora（名 `_convrot_*`，避免与 `_int8_base_*` 冲突）。
7. 设置 `lora.org_forward = <closure>`。
8. 返回 patch 清单；调用方打日志。

**与 `patch_lora_frozen_base_forwards_with_int8` 的关系：** 控制流可镜像；**算子与 buffer 命名必须独立**；禁止内部调用旧 int8 quant 后假装已旋转。

---

## 11. 保存、加载与 Metadata

| 项 | 规格 |
| --- | --- |
| 默认 checkpoint | **只存 adapter**（与现 LoRA 一致） |
| metadata 建议 | `ss_base_compute`、`ss_convrot_group_size`、`ss_convrot_scope`、`ss_convrot_weight_source` |
| 加载 adapter | 不依赖 base 仍为 ConvRot；用户换 bf16 base 应可加载（质量另论） |
| 导出 base quant | 独立工具（非默认）；layout 若要对齐 Comfy 另开工单 |
| merge | 默认拒绝；错误信息指向 dequant / 关闭 convrot 再 merge |

---

## 12. 测试计划

### 12.1 单元测试（CI 友好，无大模型）

| 文件 | 内容 |
| --- | --- |
| `tests/test_convrot_rht.py` | 正交、可逆、group 整除、shape |
| `tests/test_convrot_quant.py` | 旋转后 quant/dequant、scale shape |
| `tests/test_convrot_w8a16_linear.py` | toy bf16 vs W8A16；STE/stop-grad |
| `tests/test_convrot_apply.py` | scope、skip AdaLN、dry_run、无 ref 变体、compile 后 raise、与 int8 buffer 不冲突 |

### 12.2 探针脚本

| 脚本 | 内容 |
| --- | --- |
| `scripts/experiments/convrot_equivalence_probe.py` | 镜像 int8 probe：output L2 / loss / **adapter grad**；mlp 默认；多 seed 选项 |
| 可选 | 公开权重推理对齐（需下载，默认不进 CI） |

### 12.3 手动 / 授权后

- 短训 sample：同 seed / 同 prompt / bf16 vs W8A16。
- 不默认跑全量训练或下载。

### 12.4 建议 pytest 入口（实现后）

```bash
timeout 60 .venv/bin/python -m pytest tests/test_convrot_rht.py tests/test_convrot_quant.py tests/test_convrot_w8a16_linear.py tests/test_convrot_apply.py -q
```

---

## 13. 里程碑与工单顺序

### Milestone 0 — 文档与对齐（可选）

- [x] 研究记录：`docs/experimental/convrot_int8_training.md`
- [x] 本规格书：`docs/proposal/convrot_w8a_training_plan.md`
- [ ] （可选）Comfy ≥0.27 或 `convert_to_quant` 推理对齐 — **需用户授权下载/升级**

### Milestone 1 — 算法核

1. [x] `library/runtime/convrot/{rht,quant,linear_w8a16}.py`
2. [x] unit tests（RHT / quant / toy linear）
3. [x] 默认配置保持 `base_compute=bf16`

### Milestone 2 — 训练安全 apply

1. [x] `apply.py` + `checks.py` + `metadata.py`
2. [x] `bootstrap.py` 薄钩子
3. [x] `cli_args` + compat mutex
4. [x] `test_convrot_apply` + equivalence probe 脚本
5. [x] 与 `block_swap_transfer_dtype=int8` 互斥校验

### Milestone 3 — 质量 gate

1. [x] toy multi-seed probe（CI）
2. [x] 完整 checkpoint multi-seed probe + 20-step short-train sample 对照（W8A16 + W8A8；2026-07-24）
3. [x] 更新 experimental 文档为「可运行实验」

### Milestone 4 — W8A8

1. [x] `linear_w8a8.py` + online act quant + STE
2. [x] 默认 off（`base_compute` 默认 bf16）
3. [ ] compile / cudagraph / `router_source=input` 专项
4. [x] 真 int8 GEMM（`torch._int_mm` + float 回退；Triton 融合仍未做）
5. [x] quant 后默认释放 base bf16 权重（meta 占位）
6. [x] 融合 RHT+quant+GEMM（单 autograd.Function；默认 dense RHT + dequant；FWHT/int8pack opt-in）

### Milestone 5 — 变体与产品化（按需）

1. [x] 无 ref 跳过计入日志；0 patch raise
2. [x] DoRA 拒绝
3. [x] WebUI 实验项 MVP：`base_compute` / `convrot_group_size` / `convrot_scope=mlp`（配置页 → 显存与速度优化；默认 bf16）
4. [x] merge 拒绝 + metadata stamp
5. [ ] 稳定后迁 `docs/methods/`

---

## 14. 风险登记

| ID | 风险 | 等级 | 缓解 |
| --- | --- | --- | --- |
| R1 | RHT 坐标系与 LoRA 残差不一致 | 高 | 单测锁等价；delta 永不强制旋转 |
| R2 | 无 `org_module_ref` 变体被静默跳过 | 高 | Phase 1 限定 plain LoRA；跳过必须计入日志；禁止“0 patch 仍成功” |
| R3 | DoRA / merge 读 weight | 中 | Phase 1 拒绝 DoRA；merge 拒绝 |
| R4 | 与 block-swap int8 双 dequant | 中 | 互斥校验 |
| R5 | compile graph break | 中 | W8A16 先 `F.linear`；W8A8 限制 compile |
| R6 | 社区权重 layout 不明 | 中 | 默认 `online_from_bf16` |
| R7 | 本机无外部实现可抄 | 低 | 按论文+自建；不依赖 Comfy 0.22 |
| R8 | 评估误用旧 audit PASS | 高 | §4 写死放行标准 |
| R9 | 热点文件膨胀 | 中 | 强制 `library/runtime/convrot/` |
| R10 | 用户把开关当省显存银弹 | 中 | 文案强调实验；速度非 Phase 1 KPI |

---

## 15. 开放问题与冻结默认

| # | 问题 | **本规格冻结默认** |
| --- | --- | --- |
| 1 | 是否只实现 W8A16？ | 接口预留 W8A8；**实现先 W8A16** |
| 2 | 权重来源？ | **`online_from_bf16`** |
| 3 | 成功标准？ | **adapter grad + 短训 sample** |
| 4 | 范围？ | **plain LoRA + scope=mlp** |
| 5 | group size？ | **256** |
| 6 | 默认配置？ | **`base_compute=bf16`** |
| 7 | block-swap int8？ | **与 convrot 互斥** |
| 8 | WebUI？ | Phase 1 可不做 |
| 9 | 真 int8 GEMM？ | Phase 2+ |
| 10 | 开源训练参考？ | 按无现成产品路径自建 |

变更以上默认须更新本文件版本说明，并同步 experimental 摘要。

---

## 16. 分期产品语义（防混开关）

| 开关/路径 | 实际在算什么 | 是否本规格 |
| --- | --- | --- |
| `base_compute=bf16` | 现状 | 默认 |
| `block_swap_transfer_dtype=int8` | CPU master 压缩 + H2D dequant | **否** |
| `int8_linear` probe | 裸 rowwise 存 + dequant GEMM | **否** |
| `w8a16_convrot` | RHT + int8 W + bf16 A | **Phase 1** |
| `w8a8_convrot` | + 动态 int8 A + int8 GEMM | **Phase 2** |
| Comfy load int8-ConvRot | 推理 | 战略 A，非训练 |

---

## 17. 反上帝代码与审查清单

实现 MR 自检：

- [ ] 新逻辑在 `library/runtime/convrot/`，bootstrap 仅薄调用
- [ ] 单函数 &lt;100 行优先；超限拆 helper
- [ ] 未改 `configs/base.toml` 默认
- [ ] 未把 int8_linear 改名贴牌
- [ ] 有 unit tests；有 dry_run
- [ ] 文档：本规格 + experimental 入口/状态同步
- [ ] `git diff --check` 干净
- [ ] 未触碰用户数据目录 / 未擅自下载大模型

贡献等级：**Tier 1.5**（效率/数值/现有算法扩展）；W8A8 真 kernel 可升 **Tier 2**。

---

## 18. 主要来源

- [arXiv:2512.03673 ConvRot](https://arxiv.org/abs/2512.03673)
- [ComfyUI v0.27.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.27.0) / [PR #14859](https://github.com/Comfy-Org/ComfyUI/pull/14859)
- [Anima INT8 ConvRot 权重](https://huggingface.co/obsxrver/ComfyUI-Native-INT8_ConvRot)
- [INT8 质量基准（含 Anima）](https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/Metrics.md)
- [convert_to_quant](https://github.com/silveroxides/convert_to_quant)
- [QLoRA](https://arxiv.org/abs/2305.14314) / [SmoothQuant](https://arxiv.org/abs/2211.10438) / [QuaRot](https://arxiv.org/abs/2404.00456)
- 本仓：`docs/experimental/convrot_int8_training.md`、`docs/findings/anima_int8_base_linear_audit.md`、`library/runtime/int8_linear.py`、`library/training/bootstrap.py`、`bench/channel_stats/`

---

## 19. 修订记录

| 日期 | 变更 |
| --- | --- |
| 2026-07-24 | 初版：战略 C 完整规格；基于四路代码探索冻结默认与里程碑 |
| 2026-07-24 | 落地 M1–M5 最小集：`library/runtime/convrot/`、bootstrap 钩子、CLI、compat 互斥、merge 拒绝、toy multi-seed probe PASS |
| 2026-07-24 | 授权后跑 full-checkpoint probe（2/3 严格 gate；grad seed0 5.9%）+ 20-step short-train sample 对照（loss rel <0.1%） |
| 2026-07-24 | 补跑 full-checkpoint + short-train 的 **W8A8**；experimental 文档合并 W8A16/W8A8 验证矩阵 |
| 2026-07-24 | 任务1+2：默认 free base bf16 权重（meta）；W8A8 真 int8 GEMM（`_int_mm`/float 回退）；mem/speed：free 后 peak 低于 bf16，端到端仍慢于 bf16 |
| 2026-07-24 | 任务3：`fused.py` + `group_fwht`；单 autograd 融合 RHT+quant+GEMM。本机 3080 实测 FWHT/`_weight_int8pack_mm` 慢于 dense+dequant（W8A16 曾到 47s/step），默认改回 dense RHT + dequant；FWHT/int8pack 仅 env opt-in |
| 2026-07-24 | 新增后续优化提案 [`convrot_w8a_optimization_roadmap.md`](convrot_w8a_optimization_roadmap.md)：P0 profile/prequant/口径；P1 混精/缩小 scope；P2 真 Triton 融合门槛 |
| 2026-07-24 | P0-A 落地：`scripts/experiments/convrot_step_profile_probe.py`；本机饼图否决立刻 Triton（convrot tax~14%）；下一步 W8A16 bf16 计算 dtype |
| 2026-07-24 | P0-A2：W8A16 dequant/`F.linear` 改 bf16 计算；mem_speed ~3.0→~2.0 s/step |
| 2026-07-25 | P0-C：`prequant_checkpoint` 原生 v1 加载/导出（`library/runtime/convrot/prequant.py`）；act RHT 仍在线 |
| 2026-07-25 | P0-D：regular Hadamard（Kronecker \(4^k\)）+ multi-seed 对照；默认仍 sylvester；opt-in `ANIMA_CONVROT_HADAMARD=regular` |
