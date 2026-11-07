# Design: 合并 MonadForge 训练侧 fp16 防护链到 anima_lora

状态：设计已确认，待实现计划  
适用版本：当前 `anima_lora` main  
来源对照：`/home/scv/nvme0n1p1/训练器测试/MonadForge`  
日期：2026-07-12

## 1. 目标

把 MonadForge 已经跑通的**训练侧 fp16 完整防护链**合进 `anima_lora`，让 pre-Ampere GPU（尤其 V100/T4）在默认 `bf16` 配置下也能稳定进入 fp16 训练，并且不炸 residual、不花 VAE 预览图。

成功标准：

1. `mixed_precision="bf16"` + GPU `sm < 8` 时，在 `Accelerator` 构造前自动切到 `fp16`。
2. `mixed_precision=="fp16"` 时，DiT residual 流启用 fp32 安全累加，且发生在 `compile_blocks` 之前。
3. `fp16 + sm < 8` 时，VAE 默认强制 fp32；`--half_vae` 可覆盖；`--no_half_vae` 永远 fp32。
4. 现有 V100 自动 `lora_fp32_compute` 逻辑保持可用且不被破坏。
5. 推理侧 `runtime_dtype` 保持现状，不回退成 hardcode bf16。
6. 相关单测通过；默认不跑真实大模型训练。

## 2. 非目标

- 不改推理默认精度策略，不删除 `library/inference/precision.py` 的可配置 `runtime_dtype`。
- 不把 MonadForge 的 `train.py` 巨石结构原样搬回 `anima_lora`。
- 不改默认配置值：`configs/base.toml` 仍保持 `mixed_precision = "bf16"`。
- 不在本轮重做 optimizer / GradScaler 之外的数值栈大重构。
- 不默认执行真实长训练、大模型下载或线上发布。

## 3. 背景与现状

### 3.1 MonadForge 已有能力

- `_resolve_mixed_precision(args)`：默认/配置为 `bf16` 且 `sm < 8` 时就地改为 `fp16`。
- `_resolve_vae_dtype(args, weight_dtype)`：pre-Ampere + fp16 时强制 VAE fp32。
- `Anima.enable_fp32_residual()` + `Block/_residual_add/_gated_residual_add`：防止 residual / gate*branch 超过 fp16 上限。
- V100 + fp16 自动 `lora_fp32_compute`。
- 配套测试：mixed precision / VAE dtype / residual safe / lora dtype policy。

### 3.2 anima_lora 现状

已有：

- `mixed_precision` / `full_fp16` / `full_bf16` 开关
- `prepare_dtype` / `patch_accelerator_for_fp16_training`
- V100 自动 `lora_fp32_compute`（`library/training/bootstrap.py`）
- LoRA rank fp32 计算策略（`networks/lora_modules/base.py`）
- 推理 `runtime_dtype`
- V100 预设 `configs/gui-methods/lora-v100-stable.toml`（手动写 `fp16`）

缺失：

- 自动 `bf16 → fp16`
- DiT residual 防溢出
- VAE 半精度自动保护
- `--half_vae`
- 对应单测 / residual bench 对齐

## 4. 已确认决策

| 决策点 | 选择 |
|---|---|
| 合并范围 | 训练侧完整防护链 |
| 自动切精度规则 | 与 MonadForge 一致：`mixed_precision=="bf16"` 且 `sm < 8` 就地改 `fp16` |
| 落地方案 | 行为对齐 + 模块化落位（方案 A） |
| 推理 | 不动 `runtime_dtype` |
| 显式 bf16 与默认 bf16 | 不区分；只要值是 `bf16` 就会在 pre-Ampere 上切换 |

## 5. 架构

### 5.1 运行时序

```text
读配置 mixed_precision
  -> resolve_mixed_precision(args)          # prepare_accelerator 之前
  -> prepare_accelerator(args)
  -> prepare_dtype(args)
  -> resolve_vae_dtype(args, weight_dtype)
  -> 加载模型 / 建 network / apply_to / load_weights
  -> 已有：V100+fp16 自动 lora_fp32_compute
  -> if mixed_precision=="fp16" and Anima: enable_fp32_residual()
  -> compile_blocks(...)                    # 必须在 residual 之后
  -> 训练 loop
```

### 5.2 模块边界

| 单元 | 职责 | 依赖 |
|---|---|---|
| `library/training/precision_policy.py`（新建） | 纯函数：mixed precision / VAE dtype 解析 | `torch`、可注入 capability 探测 |
| `library/training/train_session.py` | 启动接线：切精度、定 VAE dtype | precision_policy、accelerator |
| `library/training/bootstrap.py` | residual 开启点（compile 前） | Anima 模型 API |
| `library/anima/models.py` | residual 数值护栏 | 无训练依赖 |
| `library/training/cli_args.py` | `--half_vae` 与文案 | argparse |

原则：

- 热点文件只做 facade / 小范围接线，不把整套策略逻辑堆回去。
- 精度策略可单测、可 mock，不依赖真实 GPU 型号。
- residual 逻辑放模型层，启动层只负责“何时打开”。

## 6. 接口设计

### 6.1 `resolve_mixed_precision`

```python
def resolve_mixed_precision(
    args,
    *,
    get_capability: Callable[[], tuple[int, int]] | None = None,
) -> None:
    ...
```

规则：

1. 仅当 `args.mixed_precision == "bf16"` 才处理。
2. CUDA 不可用：直接返回。
3. capability 探测失败：warning，保持 `bf16`。
4. `major < 8`：`args.mixed_precision = "fp16"`，warning 说明原因。
5. Ampere+：不变。

### 6.2 `resolve_vae_dtype`

```python
def resolve_vae_dtype(
    args,
    weight_dtype: torch.dtype,
    *,
    get_capability: Callable[[], tuple[int, int]] | None = None,
) -> torch.dtype:
    ...
```

优先级：

1. `no_half_vae` → `torch.float32`
2. `half_vae` → `weight_dtype`
3. `mixed_precision != "fp16"` → `weight_dtype`
4. CUDA 不可用 / 探测失败 → `weight_dtype`（探测失败打 warning）
5. `major < 8` → `torch.float32`
6. 其它 → `weight_dtype`

### 6.3 residual 护栏

在 `library/anima/models.py`：

- `Block.fp32_residual: bool = False`
- `FinalLayer.fp32_residual: bool = False`
- `Block._residual_add(a, b)`
- `Block._gated_residual_add(residual, gate, branch)`
- `Anima.enable_fp32_residual()`：所有 block + final_layer 置 True

`Block._forward` 中三处 residual 更新改为走 helper：

- self-attn residual
- cross-attn residual
- mlp residual

`FinalLayer` 在 `fp32_residual=True` 时对 AdaLN modulate 相关敏感路径做 fp32 安全计算，行为对齐 MonadForge。

约束：

- 默认 False，bf16/fp32 路径保持现有行为。
- 分支必须是 plain bool，避免 compile 时 data-dependent 控制流。
- 必须在 `compile_blocks_for_training` 前启用。

### 6.4 CLI

在 `library/training/cli_args.py`：

- 新增 `--half_vae`
- 收紧 `--no_half_vae` 文案，明确“永远 fp32”

不新增额外环境变量。

## 7. 接线细节

### 7.1 `train_session.py`

当前：

```python
accelerator = prepare_accelerator(args)
weight_dtype, save_dtype = prepare_dtype(args)
vae_dtype = (torch.float32 if args.no_half_vae else weight_dtype) if trainer.cast_vae(args) else None
```

改为：

```python
resolve_mixed_precision(args)
accelerator = prepare_accelerator(args)
weight_dtype, save_dtype = prepare_dtype(args)
vae_dtype = (
    resolve_vae_dtype(args, weight_dtype)
    if trainer.cast_vae(args)
    else None
)
```

### 7.2 `bootstrap.py`

在 network apply/load、gradient checkpointing 设置之后，`if args.torch_compile:` 之前：

```python
if args.mixed_precision == "fp16" and isinstance(unet, anima_models.Anima):
    unet.enable_fp32_residual()
    logger.info("fp16 mixed precision: enabled fp32 residual accumulation ...")
```

说明：

- 即使 `torch_compile=false`，也要启用 residual 护栏。
- 已有 `should_auto_enable_lora_fp32_compute` 不动。

## 8. 错误处理

| 情况 | 行为 |
|---|---|
| CUDA 不可用 | 不自动切 fp16；VAE 不强制改 |
| capability 读失败 | fail-closed：保持现状 + warning |
| 用户显式 `fp16` | 不回调；继续 residual / VAE 保护 |
| 用户 `mixed_precision="no"` | 不介入 |
| `no_half_vae` + `half_vae` 同时开 | `no_half_vae` 赢 |
| residual 在 compile 后才开 | 禁止；接线保证顺序 |

## 9. 测试计划

TDD：先写失败测试，再实现。

### 9.1 新建

1. `tests/test_mixed_precision_resolver.py`
   - bf16 + sm70/sm75 → fp16
   - bf16 + sm80 → 保持 bf16
   - 已是 fp16 / no → 不动
   - capability 失败 → 保持 bf16
2. `tests/test_vae_dtype_resolver.py`
   - `no_half_vae` 永远 fp32
   - `half_vae` 覆盖自动保护
   - fp16 + pre-Ampere → fp32
   - bf16 / Ampere → 跟随 weight_dtype
3. `tests/test_fp16_residual_safe.py`
   - residual 溢出时 `fp32_residual=True` 有限
   - 默认 False 惰性 / 与旧路径一致
   - `enable_fp32_residual` 传播到全部 block 与 final layer

### 9.2 回归

- `tests/test_training_bootstrap.py` 中 `lora_fp32_compute` 相关用例
- 如 CLI 注册有测：确认 `--half_vae` 存在

### 9.3 不默认跑

- 真实 V100 长训练
- 大模型下载
- 完整 e2e 出图训练

## 10. 文档与配置

最小文档动作：

- 本 design 文档
- 实现时如有用户可见行为变化，在 `docs/configuration/` 或 V100 预设注释补一句：
  - 默认 `bf16` 在 pre-Ampere 会自动切 `fp16`
  - VAE 在该路径默认 fp32

`configs/gui-methods/lora-v100-stable.toml` 可继续显式 `mixed_precision="fp16"`，作为兼容预设，不强制删除。

## 11. 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| residual 改动影响 bf16 数值 | High | 默认 flag=False；补 inert 测试 |
| compile 后改 flag 导致重编译/行为错 | High | 强制 residual 在 compile 前；测试钉顺序约束 |
| 自动切精度让“显式 bf16”也被改掉 | Medium | 已确认与 MonadForge 一致；日志明确提示 |
| VAE 强制 fp32 增加显存 | Medium | 仅 pre-Ampere + fp16；提供 `--half_vae` |
| 热点文件继续膨胀 | Medium | 策略进新模块；models 只加必要护栏 |

## 12. 实现顺序建议

1. 新建 `precision_policy.py` + mixed/VAE resolver 测试
2. 接 `train_session.py` + `--half_vae`
3. 实现 residual 护栏 + residual 测试
4. 在 `bootstrap.py` 打开 residual
5. 跑定向测试，修回归
6. 补最小文档注释

## 13. 明确不做的回退

- 不把推理 hardcode 回 `torch.bfloat16`
- 不恢复旧 metadata fallback
- 不改 Text Encoder padding / constant token buckets 等无关不变量
