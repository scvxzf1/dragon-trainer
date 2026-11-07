状态：已完成（方向 A 端到端验证通过 + 落盘/小卡链路 + compat_matrix 放开 + 消融矩阵六维基准）
适用版本：当前 krea2-migration 分支
入口命令：
- `.venv/bin/python scripts/krea2/probe_nf4_blockswap_compat.py`（命门验证，已过）
- `.venv/bin/python scripts/krea2/probe_nf4_blockswap.py`（端到端，已过）
- `.venv/bin/python scripts/krea2/probe_nf4_save.py`（NF4 权重落盘 round-trip，已过）
- `bash scripts/krea2/run_nf4_ablation.sh`（NF4 × {完整检查点, 块交换} 5 格消融矩阵，已过；`SMOKE=1` 冒烟）
- `.venv/bin/python scripts/krea2/probe_nf4_ablation_1024.py`（3080 1024 swap 维度步进曲线测绘，已过；OOM 自动 +4 逼近临界、过点 +2 细扫）
相关代码：`library/runtime/offloading.py`、`library/runtime/block_swap_masters.py`、`library/models/krea2_raw/quantize.py`、`library/models/krea2_raw/weights.py`、`library/training/compat_matrix.py`、`library/training/model_loading.py`、`library/training/cli_args.py`、`scripts/krea2/probe_nf4_ablation.py`、`scripts/krea2/probe_nf4_ablation_1024.py`、`scripts/krea2/run_nf4_ablation.sh`
相关 findings：[krea2_nf4_ablation_findings.md](../../../docs/findings/krea2_nf4_ablation_findings.md)（5 格六维消融矩阵：显存/内存/速度/loss/数学实现/数学偏移）

# NF4 × block swap 落地方案

## 背景

`compat_matrix.py` 曾拒绝 `base_compute=nf4` × `blocks_to_swap>0`，标记
`nf4_block_swap_unverified`，理由是「block-swap offloader 遍历 named_modules 取
.weight；Linear4bit.weight 是 Params4bit，offloader 兼容性未经探针验证」。方向 A
端到端探针通过后，已降级为 `nf4_block_swap_host_ram` warning（不再硬拒，提醒主战场
是 host RAM）。

### 触发动机：bf16 block swap 在 62GB RAM 机器上宕机

bf16 路径下 block swap 的 CPU masters 是**全量 28 块**的 bf16 权重副本且
`pin_memory=True`（`offloading.py:372`），实测 `Block swap frozen CPU masters prepared:
22.64 GiB across 28 blocks`。叠加 DiT 原权重 26GB（`load_krea2_dit(device="cpu")`）
= 48.6GB，本机 62GB RAM + swap 已满（7.3/8.0Gi）→ pinned 22.64GB 换不出 → thrash
→ 宕机。

**NF4 × block swap 的真正价值不在 GPU 显存，在 host RAM**：NF4 master 仅 6.7GB
（4-bit 码 + quant_state，实测为 bf16 master 的 25.8%），解决宕机。

## 实测事实（探针 + bnb 源码核实）

### bnb 0.49.2 Params4bit 契约（已点验）

- `Params4bit(torch.nn.Parameter)`：`.data` 存 4-bit 打包码（uint8，2 值/byte），
  `.quant_state` 是 `QuantState`（absmax + code + shape + dtype + state2 二级量化）。
- `Params4bit.to(device)`（`modules.py:341`）：`bnb_quantized=True` 时走 else 分支，
  同步搬 4-bit 码（`super().to`）和 quant_state（`QuantState.to` 搬 code/absmax/
  state2.code/state2.absmax/offset），**原地操作**（同 `nn.Module.to`），`bnb_quantized`
  保持 True。
- `__deepcopy__`（`modules.py:258`）：对 `quant_state` 和 `data` 都 `copy.deepcopy`，
  完整复制 quant_state（含 state2），但 `module` 字段不深拷（引用原 Linear4bit）。
- `Linear4bit.forward`（`modules.py:528`）：直接传 `self.weight`（Params4bit）给
  `bnb.matmul_4bit`，底层 `MatMul4Bit.forward`（`_functions.py:299-315`）每步调
  `F.dequantize_4bit(B, quant_state).to(A.dtype).t()` 后 `F.linear` —— **反量化在
  每次 forward 实时发生，产出完整 bf16 权重临时 tensor**。

### NF4 forward 开销（实测 PG199 bf16）

| Linear 形状 | NF4 fwd | bf16 fwd | NF4/bf16 | 反量化临时显存 |
| --- | --- | --- | --- | --- |
| 5120×5120 | 0.389ms | 0.054ms | 7.15× | 54.1MB（≈bf16 权重） |
| 4096×12288 | 0.408ms | 0.089ms | 4.61× | 103.8MB（≈bf16 权重） |

NF4 forward 慢 4.6-7.2×（反量化是每步真开销），反量化产出完整 bf16 权重（临时显存
≈ bf16 权重体积）。

### Params4bit master 体积（实测）

4-bit 码 = bf16 的 25%，quant_state ≈ 0.8%（absmax + code + state2），合计 ≈ bf16
的 25.8%。对 26GB Krea-2 DiT：**NF4 master ≈ 6.7GB vs bf16 master 22.64GB，省 3.4×**。

## 方向 A（初版，推荐）：deepcopy master + Params4bit.to() 整体搬运

### 命门已验证

`scripts/krea2/probe_nf4_blockswap_compat.py` 全绿：
- deepcopy 完整性（含 state2 双重量化）OK
- CPU 持有合法 OK
- **整体搬运 forward delta=0**（deepcopy→CPU→cuda 不改 forward 语义）
- 5 轮交替搬运（master 独立副本）delta=0，master 不被搬运污染
- 量化状态保持 OK

关键探针发现：`Linear4bit.to()` 是原地操作，offloader 必须持有**独立 CPU master
副本**，搬运时从 master deepcopy 重建 GPU 副本——契合现有 `_cpu_weight_masters`
设计（master 独立、搬运不污染）。

### 端到端验证（方向 A 已落地，PG199, 1024×1024, swap=4, 30 步）

`probe_nf4_blockswap.py` 全绿（exit 0）：

| 验证项 | 结果 | bf16 baseline 对比 |
| --- | --- | --- |
| forward+backward+opt 跑通 | ✅ 30 步全跑完 | — |
| loss 单调下降 | ✅ first5=0.0085 → last5=0.0017 | NF4-only 量级可比 |
| LoRA grad 非零 | ✅ [0.0005, 0.0104] 全非零 | 搬运不阻断梯度流 |
| DiT Linear4bit frozen | ✅ bnb_quantized=True, device=cuda, 有限 | 搬运不改 frozen 权重 |
| **host RAM（主战场）** | ✅ **18.18GB**（训练前后持平） | bf16 master 22.64GB 单项超它 → 62GB 机宕机 |
| **GPU peak** | ✅ **10.26GB** | NF4-only 10.49GB / bf16+swap 32.62GB |
| block swap master 体积 | 5.66GB / 28 blocks | bf16 路径 22.64GB（NF4 是其 25%） |
| avg step | 3.88s | NF4-only 反量化慢 ~5×（同量级） |

**两个核心收益都拿到**：主战场 host RAM 18.18GB（bf16 路径 22.64GB master 单项就超
它，曾在 62GB 机宕机），GPU 10.26GB（比 NF4-only 还低 ~0.2GB，block swap 移出 4 块）。

### NF4 权重磁盘落盘/加载（小卡链路）

在线量化（`quantize_dit_to_nf4`）的 `.to(device)` 要把整个 bf16 DiT（26GB）一次性
搬到 GPU 触发量化，需一张能放 26GB bf16 的卡（PG199 32GB）。3080 等 8-12GB 卡无法
在线量化。`quantize.py` 加 `save_nf4_dit` / `load_nf4_dit_into` 走 bnb 0.49.2 官方
`QuantState.as_dict(packed=True)` + `from_prequantized` 契约 round-trip，落盘后小卡
直接 `load_krea2_dit(nf4_path=...)` 加载 6.6GB，绕过硬约束。

`probe_nf4_save.py` 全绿：
- 阶段 1（小规模冒烟）：2 个 Linear4bit round-trip，4-bit 码逐字节一致，forward
  delta=0.00e+00（含 state2 双重量化）。
- 阶段 2（真实 DiT 落地，PG199 在线量化 → 落盘 → 磁盘加载对比）：264 Linear4bit
  落盘 **6.61GB / 14.2s**，磁盘加载 89.2s（不重新量化，`bnb_quantized=True` 直来自
  存盘），同 path 4-bit 码逐字节一致，forward delta=0.00e+00。
- 落盘文件：`models/diffusion_models/krea2_raw_nf4.safetensors`（6.61GB）。

2026-08-09 增加自包含 NF4 v2：在不重新量化的前提下，将旧 NF4 的
264 个 Linear4bit 与 BF16 底模中缺失的 159 个模型 tensor（5.44MiB）合并。
`load_krea2_dit` 依据 safetensors metadata 自动识别 v2，可直接将它作为
`pretrained_model_name_or_path`，不再读取 25GB BF16 payload。旧 v1 overlay 路径保持兼容。

```bash
.venv/bin/python scripts/krea2/build_self_contained_nf4.py
```

实现和 PG199/RTX 3080 验证记录见
`docs/findings/krea2_nf4_self_contained.md`。

### 3080 小卡实测（8.6GB 可用，磁盘 NF4 + TE-CPU）

3080（10GB，gnome-remote 残留后可用 ~8.6GB）用磁盘 NF4 + TE-CPU encode 链路实测，
证明方向 A 在 8GB 级小卡上训练可行（`probe_nf4_blockswap.py`）：

| 分辨率 | swap | GPU peak | host RSS | avg step | loss 首末 | 训练 |
| --- | --- | --- | --- | --- | --- | --- |
| 512×512 | 20 | 5.42GB | 20.58GB | 4.88s | 0.0101→0.0017 | ✅ |
| 1024×1024 | 20 | 7.65GB | 20.47GB | 12.14s | 0.0087→0.0039 | ✅ |

**1024×1024 swap 维度完整曲线**（`probe_nf4_ablation_1024.py` sweep 模式，每轮 10 步）：

| swap | GPU peak | host RSS | avg step | 训练 |
| --- | --- | --- | --- | --- |
| 8 | OOM | — | — | ❌ |
| 12 | OOM | — | — | ❌ |
| 16 | OOM | — | — | ❌ |
| **20** | **7.65GB** | 20.47GB | 12.14s | ✅ 临界点 |
| 22 | 6.78GB | 21.07GB | 12.27s | ✅ |
| 24 | 5.91GB | 21.66GB | 12.37s | ✅ |
| 26 | 5.04GB | 22.25GB | 12.48s | ✅ 上限 |
| 28 | — | — | — | ❌ 断言拒（见下） |

三个规律：
1. **GPU peak 随 swap 线性降**：每 +2 swap 省 ~0.87GB（移出 NF4 权重 + 反量化窗口）。
2. **host RSS 随 swap 线性升**：每 +2 swap 升 ~0.6GB（CPU master + 搬运缓冲）——swap
   省 GPU 显存换 host RAM。
3. **速度几乎不变**（12.14→12.48s，仅慢 3%）：反量化主导开销（NF4 forward 慢 ~7×），
   H2D 搬运藏在反量化窗口里不构成瓶颈。**这印证方向 B（slab+重叠）在 1024 下优化收益
   不值得其风险**——H2D 串行不是瓶颈，方向 B 的存在前提（大分辨率 H2D 串行成瓶颈）
   在 1024 不成立。

swap=28 全交换失败不是 OOM 也不是 bug：`dit.py:516` 的 `enable_block_swap` 断言
`blocks_to_swap <= num_blocks - 2`（DiT 28 块，最多 swap 26，留 2 块常驻给 recompute），
swap=28 被显式拒。这是 offloader 的设计内硬上限，非物理限制。

**3080 工作点选择**：swap=20 是临界（GPU 7.65GB，可用 8.6GB 富余 ~1GB，最省显存安全
余量小）；swap=24（5.91GB）或 26（5.04GB）富余更大，速度代价仅 ~2%。生产建议
swap=24 兼顾安全余量与速度。

### 适配点

1. `_capture_cpu_master`（`block_swap_masters.py:100`）：加 `isinstance(weight,
   Params4bit)` 分支，用 `copy.deepcopy(weight)` 建 master（不 detach、不
   transfer_dtype、不 slab 打包），stats 报 4-bit 实际字节数。
2. `_parked_cpu_master_tensor`（`:172`）+ `_restore_cpu_master_tensor`（`:178`）：
   Params4bit 分支用 `master.to(device)` 整体搬（同步搬 quant_state），返回
   Params4bit。
3. `offloading.py` 所有 `module.weight.data =` / `.copy_(...)` 赋值点（:158,179,192,
   1210,1232,1312,1349 等）：Params4bit 分支改成 `module.weight = restored_params4bit`
   整体赋回（不是 `.data`）。需注意 `module` 引用：deepcopy 副本的 `weight.module`
   指向原 module，挂回时若 `fix_4bit_weight_quant_state_from_module` 依赖它需手动设
   `new.module = target`。
4. `compat_matrix.py:251`：`nf4_block_swap_unverified` 已降级为 `nf4_block_swap_host_ram`
   warning（不再硬拒），保留可见提示提醒 host RAM 主战场。

### 取舍依据

- **host RAM**：6.7GB（解决宕机，比 bf16 省 3.4×）。
- **GPU 显存**：NF4 + swap 4 ≈ 10-12GB，比 NF4-only（10.49GB）只省 ~1GB —— GPU
  不是主战场。
- **速度**：`Params4bit.to()` 走默认 stream、non_blocking 不可控，H2D 串行，无法
  和 forward 重叠。小分辨率下 ①② 接近；大分辨率下 ① 比 ② 慢 20-40%（② 可重叠，
  见进阶方向）。
- **实现风险低**：用 bnb 官方 `deepcopy` + `.to()`，探针已证 delta=0，bnb 升级
  鲁棒。

## 方向 B（进阶开发存储，未落地）：slab master + 手动重建 Params4bit

> 速度优化方向，待 ① 落地后大分辨率探针证明 H2D 串行成瓶颈时再做。**显存上限与
> ① 几乎相同，速度上限更高，但实现风险显著**。
>
> **2026-08-09 前置诊断结论：NOT_WORTH，归档。** PG199 1024×1024 双口径实测
> H2D 占比都 0.0%（swap=4 vs swap=0 单步差 1ms，forward/backward 事件差 4ms/-3ms
> 噪声级），H2D 完全藏在 NF4 反量化窗口；理论上界 4 块 ~1.4GB ÷ 20GB/s ≈ 70ms =
> 整步 2%。方向 B 重叠收益上界 2%，不值得其实现风险。落地前提 2「H2D 串行占
> 显著比例」**不成立**。详见
> [krea2_nf4_h2d_bottleneck_findings.md](../../../docs/findings/krea2_nf4_h2d_bottleneck_findings.md)。
> 8 字段清单 + bnb 契约保留在此段，供未来大分辨率（>1024）或带宽受限场景复用。

### 数学模型

- **存什么**：4-bit 码打包成连续 slab（复用现有 bf16 slab 路径），quant_state
  单独列管（按 block 存 absmax/code/state2 的独立副本或引用）。
- **搬什么**：4-bit 码走 slab 的 `gpu_slab.copy_(cpu_slab, non_blocking=True)`
  （现有 foreach + per-slot stream 机制，可异步重叠）；quant_state 跟着搬或从 GPU
  常驻池取。
- **重建什么**：搬运后**手动重建 Params4bit** —— `Params4bit.__new__()` + 设 8 个
  字段（`data`/`quant_state`/`bnb_quantized`/`quant_type`/`blocksize`/
  `compress_statistics`/`quant_storage`/`module`），挂回 `module.weight`。
- **数学不变量**：理论上 delta=0（同样的 4-bit 码 + quant_state），但依赖 bnb
  内部字段，漏一个静默算错或 forward 崩。

### 与方向 A 的区别

| 维度 | A deepcopy | B slab+手动重建 |
| --- | --- | --- |
| 实现风险 | 低（官方 API，探针已证） | 高（手动拼 8 字段，依赖 bnb 内部，易碎） |
| host RAM | 6.7GB | ~6.6GB（平手） |
| GPU 显存 | ~10-12GB | ~10-12GB（平手） |
| 小分辨率速度 | 略慢（H2D 串行，窗口小） | 略快（可重叠，窗口小） |
| 大分辨率速度 | 慢（H2D 串行） | 快 20-40%（H2D 藏在反量化窗口） |
| bnb 升级鲁棒性 | 强 | 弱 |

**速度差异根因**：NF4 forward 反量化慢（7×），大分辨率下 forward 计算时间长，
B 的 per-slot stream 能把 H2D（0.96GB/步 ÷ 20GB/s ≈ 48ms）藏在反量化窗口里，A
的默认 stream 串行做不到。**NF4 的反量化慢反而放大了 B 的重叠价值**。

### 落地前提

1. 方向 A 端到端探针（`probe_nf4_blockswap.py`）先跑通，提供大分辨率 H2D 串行
   开销的对照基线。
2. 验证大分辨率（512×512 / 1024×1024）下 H2D 串行确实占 step 时间显著比例（若
   反量化主导、H2D 占比小，B 的优化收益不值得其风险）。
3. 手动重建 Params4bit 需写专门的单测覆盖 8 字段完整性 + state2 嵌套 + bnb 版本
   兼容。

## 落地步骤（方向 A）

1. ✅ 兼容性探针 `probe_nf4_blockswap_compat.py`（命门验证，已过）。
2. ✅ offloader 加 Params4bit 分支（`offloading.py` + `block_swap_masters.py`，
   isinstance 分流不碰 bf16/int8/fp8 路径，`tests/test_block_swapping.py` 64 测试
   全绿）。
3. ✅ 端到端探针 `probe_nf4_blockswap.py`（NF4 + block_swap + LoRA + grad-ckpt，
   host RAM 18.18GB、GPU 10.26GB、loss 0.0084→0.0017、grad 非零、frozen 不变，
   全绿）。
4. ✅ NF4 权重落盘/加载 `probe_nf4_save.py`（round-trip 保真，落盘 6.61GB，
   `load_krea2_dit(nf4_path=...)` 小卡链路）。
5. ✅ 放开 `compat_matrix.py`（`nf4_block_swap_unverified` error →
   `nf4_block_swap_host_ram` warning，`tests/test_training_compat_matrix.py`
   14 测试全绿，含 `nf4_path_ignored` 校验）。
6. ✅ train.py 生产链路接线 `--nf4_prequantized_path`（`cli_args.py` 加参数 +
   `model_loading.py::_load_krea2_dit` 优先读 `args.nf4_prequantized_path` 传
   `nf4_path=` 给 `load_krea2_dit`，跳过在线量化；`compat_matrix.py` 加
   `nf4_path_ignored` warning 提醒配错）。生产命令：
   `train.py --base_compute nf4 --nf4_prequantized_path
   models/diffusion_models/krea2_raw_nf4.safetensors --blocks_to_swap 4`
   ——小卡直接加载 6.61GB NF4 文件训练，绕过 26GB bf16 在线量化硬约束。

## 消融矩阵六维基准（NF4 × {完整检查点, 块交换}）

`scripts/krea2/probe_nf4_ablation.py` + `run_nf4_ablation.sh` 跑 5 格消融矩阵
（PG199 32GB, 1024×1024, 30 步, LoRA dim=16），六维指标全记录到
`docs/findings/krea2_nf4_ablation.jsonl`。完整矩阵表 + 判断见
[krea2_nf4_ablation_findings.md](../../../docs/findings/krea2_nf4_ablation_findings.md)。

核心结论：

| 维度 | bf16 基线 | NF4-only | NF4+swap4 | NF4+ckpt | NF4+swap4+ckpt |
| --- | --- | --- | --- | --- | --- |
| GPU peak | 29.45GB | 10.49GB | 10.26GB | 10.49GB | 10.45GB |
| host RSS | 26.10GB | 18.17GB | 18.18GB | 18.18GB | 18.18GB |
| avg step | 3.29s | 3.41s | 3.88s | 3.41s | 3.88s |
| loss first5→last5 | 0.0020→0.0005 | 0.0085→0.0018 | 0.0083→0.0017 | 0.0083→0.0017 | 0.0084→0.0017 |
| 数学偏移 cos | (ref) | 0.9972 | 0.9972 | 0.9972 | 0.9972 |
| 数学偏移 rel_l2 | (ref) | 7.55% | 7.55% | 7.55% | 7.55% |
| ckpt round-trip | — | — | — | delta=0 | delta=0 |

1. **NF4 显存省 72%（29.45→10.49GB），速度仅慢 3.6%（3.29→3.41s）**，forward 方向
   高度保真（cos=0.9972, rel_l2=7.55%）。反量化慢 4.6-7.2× 但被 grad-ckpt + LoRA +
   attention 摊薄。
2. **块交换主战场在 host RAM 不在 GPU**：NF4+swap4 GPU 仅省 0.23GB（10.49→10.26），
   但 host RAM 18.18GB 远低于 bf16 路径 22.64GB pinned master（曾在 62GB 机宕机）。
   数学偏移与 NF4-only **逐位一致** → 块交换 deepcopy master + Params4bit.to()
   整体搬运零语义偏移，delta 纯量化误差。
3. **完整检查点续训无损**：LoRA+opt state 存盘 reload 后 round-trip delta=0，
   loss 跳变 0.0001（数值噪声级）。DiT NF4 master 保留不重载，省去 del+reload 26GB
   DiT 开销。检查点 289.4MB（LoRA 96.4MB + opt 193MB），存盘 1.3-1.4s。
4. **三轴全开无相互破坏**：10.45GB GPU + 18.18GB host RAM + 无损续训，是 8-12GB
   卡训练 Krea-2 的可用配置。

**关键修正**：block swap 模式下 round-trip 验证 forward 必须先调
`prepare_block_swap_before_forward`，否则 swap block 还在 CPU，forward 访问
Linear4bit 时 device 错乱 → `RuntimeError: invalid argument to getCurrentStream`。
swap=0 不受影响，swap>0 的三轴全开格首次因此崩溃，补 prepare 后通过。

## 热点文件守则交代

改 `offloading.py`（热点）的交代：
- **为什么必须改热点**：block swap 搬运逻辑全在 offloading.py，无法外移。
- **为什么不放新模块**：Params4bit 分支与现有 bf16/int8/fp8 路径共享 slab/stream/
  foreach 机制，独立模块丢这些优化且重复维护。
- **如何控制侵入**：`isinstance(Params4bit)` 早返回分支，不碰现有 bf16 路径；
  改动集中在 master 捕获/停放/恢复 + 赋值点。
- **测试**：`probe_nf4_blockswap.py` 端到端 + 现有 anima bf16 block swap 探针回归。

## 3080 速度扩展研究（阶段 1）

2026-08-09 同机 PG199/RTX 3080 消融已定位 `12s/it` 的主因：3080 代表性
BF16 大矩阵比 PG199 慢 4.4-5.2×，NF4 大矩阵慢 4.1-4.6×，attention 慢
2.83×；3080 已满频、满功耗、100% utilization，不是运行时降频。

探针口径修正：`prepare_block_swap_before_forward(free_cache=False)` 若每步强制
调用会多计约 196ms，但生产训练只在 accelerator prepare/验证恢复时调用，
所以这不是可变现的生产加速。文本 padding 尾裁剪将序列 4608→4107，但
PG199/3080 都无可测收益，行为改动已回退。完整数据与下阶段见
[krea2_3080_speed_stage1.md](../../../docs/findings/krea2_3080_speed_stage1.md)。

阶段 2 选择性 checkpoint 结果：PG199 `every_other` 将稳态步时
3.370→2.901s（-13.9%），峰值 10.49→28.46GB，适合 32GB opt-in。RTX 3080
swap20 即使只放开 1 个 block 也 OOM，故该路径无法优化 10GB 工作点。见
[krea2_3080_speed_stage2.md](../../../docs/findings/krea2_3080_speed_stage2.md)。

阶段 3 per-block compile 实现编译 block `_forward`，block swap 默认只编译 resident
块。PG199 NF4 full-ckpt 3.370→2.726s（-19.1%）。RTX 3080 swap20 的
12.140→11.744s 只是冷态短窗口，阶段 5 长窗口已修正为 12.06→12.65s，不再宣称
持续 3.3% 或“约 60 步回本”；其可靠价值是显存余量和避免 eager backward OOM。
multi-bucket 与默认 fixed compile 的最终状态见阶段 9。见
[krea2_3080_speed_stage3.md](../../../docs/findings/krea2_3080_speed_stage3.md) 和
[krea2_3080_speed_stage5.md](../../../docs/findings/krea2_3080_speed_stage5.md)。

阶段 4 叠加结果：compile + every-other（14/28 checkpoint）在 PG199 首步 OOM。
改为 16/28 checkpoint 后 20 步稳态 `2.408s/it`，比 3.370s 快 28.5%，loss
正常下降；但报告峰值 31.55GB 无安全余量，仅作 PG199 实验点。安全档仍是
full checkpoint + compile（2.726s/it / 11.06GB）。见
[krea2_3080_speed_stage4.md](../../../docs/findings/krea2_3080_speed_stage4.md)。

阶段 5 修正 3080 compile 结论：20 步从 12.06s 漂到 12.65s，120 秒纯 BF16
GEMM 在 84°C/约 1800MHz 也退化 7-9%，因此 3.3% 只是冷态短窗口，不是
持续加速。当前 eager swap20 在首个 backward OOM，compiled 则完成 20 步，其可靠
价值是显存余量。长训速度的下一主因是 3080 散热/频率曲线。见
[krea2_3080_speed_stage5.md](../../../docs/findings/krea2_3080_speed_stage5.md)。

阶段 6 FP16 compute 候选已拒绝：3080 代表性 FP16 Linear 比 BF16 快 13.7%，
NF4 层 backward 27.07→10.02ms，但同权重/同输入的输入梯度 rel-L2 达 35.6%、
cos=0.943。不值得以训练轨迹换速度，保留 `weights.py` 强制 BF16 契约。见
[krea2_3080_speed_stage6.md](../../../docs/findings/krea2_3080_speed_stage6.md)。

阶段 7 拒绝 `compile_inductor_mode="reduce-overhead"`：CUDA Graph 输出在 non-reentrant
checkpoint backward recompute 的第二次调用中被覆盖，直接 RuntimeError。Krea-2
`compile_blocks` 现仅接受 None/default mode，其他未验证 preset 同样显式拒绝。见
[krea2_3080_speed_stage7.md](../../../docs/findings/krea2_3080_speed_stage7.md)。

阶段 8 rank 消融：PG199 compile 安全档中 rank16→8 将可训参数 48.17M→
24.08M，但步时 2.726s→2.73s 持平，峰值仅省 145MB。rank 不是速度旋钮，
不应以 adapter 容量换不存在的加速。见
[krea2_3080_speed_stage8.md](../../../docs/findings/krea2_3080_speed_stage8.md)。

阶段 9 补齐 multi-bucket：24 个 constant-token buckets 在固定 512 文本后只产生
4608/4864 两张 block 图；同 family 不同宽高比首次即稳态 2.731/2.956s，
回访无重编译，峰值 <=11.35GB。Krea 方法默认开 fixed/default/resident compile，
dynamic_seq 仍关闭。见 [krea2_3080_speed_stage9.md](../../../docs/findings/krea2_3080_speed_stage9.md)。

阶段 10 验证 compile × 完整 checkpoint：LoRA/forward round-trip max delta 均为 0，
optimizer reload 后 loss jump 0.000214，续训仍 2.728-2.730s，无重编译或峰值退化。
默认 compile 可用于正常保存/续训。见
[krea2_3080_speed_stage10.md](../../../docs/findings/krea2_3080_speed_stage10.md)。

阶段 11 CUDA profile 定位 compile 收益：GEMM 1593ms 与 attention 847ms 前后不变；
eager 的 mul/copy/add 约 601ms 被 Triton 融合，可见 dequant 706→486 次、
102→68ms。compiled 后 GEMM+attention 约占 89%，是后续软件优化天花板。见
[krea2_3080_speed_stage11.md](../../../docs/findings/krea2_3080_speed_stage11.md)。

阶段 12 packed varlen FlashAttention 候选：只打包有效 token 并使用 native GQA，
attention micro 在 PG199/3080 分别快 47%/34%。PG199 全模型与两类 token family
稳态快 11-13%；3080 swap20 的 20 步热稳态从历史 12.65 降到末五步均值 12.145s
（约 4%），GPU peak 6.09GB。checkpoint reload 的 LoRA/forward delta 均为 0。
由于仍缺显式 backend、可选依赖/V100 fallback、batch/CFG 和 Dynamo 配置边界，当前
仅保留实验包装器，不改生产默认。见
[krea2_3080_speed_stage12.md](../../../docs/findings/krea2_3080_speed_stage12.md)。
