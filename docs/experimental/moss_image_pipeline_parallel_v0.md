# MOSS-Image 流水线并行 v0

状态：实验（库级 runtime 与主训练专用 PP 会话均已合入 `dev` 并通过 CPU/Gloo 回归及双 V100 CUDA tiny 独立验收；真实大模型权重长训练未验收）
适用版本：当前 `dev`（含 `anima` / `moss_image` / `krea2_raw` 三族 PP 接线）
入口命令：主训练入口见 [moss_image_pp_trainer_wiring.md](moss_image_pp_trainer_wiring.md) 与 [distributed-training-card.md](distributed-training-card.md)；未接线的 `z_image` / `qwen_image_2_1` 维持 `pp_runtime_unavailable` / `pp_trainer_path_unavailable` 具名拒绝（fail-closed）
相关代码：`library/training/pp/moss_runtime.py`、
`library/models/moss_image/pipeline_parallel.py`、
`library/models/family_registry.py`、`library/training/pp/`
需求契约：[../../_archive/docs/proposal/moss_image_pipeline_parallel_v0.md](../../_archive/docs/proposal/moss_image_pipeline_parallel_v0.md)（需求冻结归档原文）

一句话：MOSS-Image 复用 Anima PP v0 的调度/事务/checkpoint 骨架，落地了族专属的
双 stage 切分、真实文本长度 conditioning 与分 rank checkpoint；`anima`、`moss_image`、`krea2_raw`
三族发布同一个 stage-local runtime backend 常量 `torch_pipelining_1f1b`，由 `library.training.pp` 按族分发组件。
MOSS 的主训练循环已通过专用会话派发接线（绝不静默回退 DDP）；未接线族与任何误入 DDP 内部的 PP 运行仍具名拒绝。

## 支持范围

| 维度 | v0 支持 | 拒绝方式 |
| --- | --- | --- |
| 模型族 | 本文聚焦 `moss_image`（跨族已接线 `anima` / `moss_image` / `krea2_raw`） | 未接线族（`z_image` / `qwen_image_2_1`）具名拒绝 |
| stages / schedule / split | 2 / `1f1b` / `balanced`（registry 计划：42 层 → 0:21 / 21:42） | `pp_config_invalid` 族 |
| latent | 4D `[B,128,H,W]` packed（packed 边无奇偶要求，偶数约束仅 pre-pack 32ch） | `pp_shape_contract_violation` |
| 精度 | 纯 fp16 + `GradScaler`，fp32 LoRA master weights | `pp_mixed_precision_unsupported` / `pp_fp16_master_weights_required` |
| 文本 | per-sample 真实长度 mask/RoPE；wire 容量 `token_cap = H*W + max_text_tokens` | 非二进制 mask、空样本、超容均具名拒绝 |
| block swap / compile | 普通 `blocks_to_swap` 与 `torch_compile` 均与 PP 互斥（`pipeline_parallel_blocks_to_swap` 仅 `krea2_raw` 支持，MOSS 须为 `0`） | `pp_config_invalid` |
| token 桶 | 一个 run 一个 `(batch, H, W)` wire 形状；`batch` 必须整除 `microbatches` | `pp_token_bucket_mismatch` / `pp_shape_contract_violation` |

## 族边界（MOSS 专属组件，不复制 Anima）

* stage 0：`video_embedder`、`text_embedder`（refinement，真实长度 RoPE）、
  `shared_block_modulation`、`layers[0:21)`；stage 1：`layers[21:42)`、
  `final_norm_video`、head 门控、`final_linear_video`、unflatten。
* boundary 三槽：joint hidden（可微，grad 回到 stage 0 的 text 路径）、
  per-sample 真实长度（float32，torch pipelining 对收到的张量强制
  `requires_grad`，int 不可用）、共享 modulation（冻结 detach）。frozen slots
  = `(1, 2)`，由 `FrozenBoundaryPipelineStage` 成对覆盖保证不挂起。
* transport padding 永不进 attention：接收 stage 按真实长度切片重建 per-sample
  序列；RoPE 由各 stage 从 `(H, W, S_i)` 本地重建，与生产 forward 同源。
* adapter 归属按真实 `lora_unet_layers_<N>_proj_[qkvo]` 名解析，并对已 apply
  网络做 target 模块几何交叉校验；Anima 的 `lora_unet_blocks_` 正则不适用。
* checkpoint：`moss-image-pp-stage-checkpoint` 新格式（与 Anima 格式双向具名拒绝，
  读前校验），manifest 记录 family/层计划/wire 形状/adapter 归属表/max_text_tokens
  /dtype。fp16 增长策略（factor/backoff/interval）经 last-stage GradScaler state
  持久化，load 后向所有 stage 重新协商（restore 端构造参数不同也不会漂移）。
* step 前各 rank 本地校验 + 一次 CPU `all_gather_object` 协商，任一失败两 rank
  以同一 `pp_*` code 在任何通信前对称拒绝。

## 复用与共享面

`pp.contract` / `pp.transactions` / `pp.schedule`（调度与 loss）/ `pp.stages`
（frozen-boundary wire）/ `pp.checkpoint*`（run root/manifest/协商/apply）原样或
最小扩展复用；`pp.layout`（Anima 5D）不复用。共享扩展保持最小：最初两处
（`PpFp16Transaction.restore_policy` 策略恢复与 checkpoint 格式按 family 分流的
参数化）加上后续修复轮新增的 `pp/checkpoint_validation.py` 纯校验 helper 与
`validate_master_weight_dtype` 的可选 device 检查（均为纯校验/可选参数，不扩
架构、不加 optimizer 注册框架）。MOSS 侧计算 dtype 政策：boundary dtype =
activation dtype（`_Block`
不提升 residual stream）；`PpComputePolicy.boundary_dtype` 是 Anima 专属语义。

## 验收与当前边界

* **CPU / Gloo 套件**：`tests/test_moss_image_pp_runtime.py`、`tests/test_moss_image_pp_checkpoint.py`、
  `tests/test_moss_image_pp_checkpoint_validation.py`、`tests/test_moss_image_pp_devices.py`、
  `tests/test_moss_image_pp_harness.py`、`tests/test_moss_image_pp_wiring.py`（`-m 'not hardware'`）覆盖：
  - staged forward 与生产单卡 forward 预测逐位相等（fp32、非前缀 mask、奇数 packed 边、非零 LoRA）；
  - transport padding NaN/inf 双层不变性、空样本与坏 timesteps/target 对称拒绝；
  - 双 rank PP loss / 梯度相对同切分非 PP 参考在预登记容差内一致（梯度按容差验收，不混称逐位相等）；
  - 单侧 overflow 双 rank 对称 skip、四段式纯本地校验 + all-gather 协商恢复（坏 payload 零状态变异）、跨增长边界 fresh-pair replay。
* **双 V100 CUDA tiny 独立验收**：硬件节点 `tests/test_moss_image_pp_runtime.py::test_two_rank_cuda_fp16_runtime_and_resume`
  实跑通过（峰值 allocated 约 23.2 / 21.4 MiB，双 rank overflow 对称 skip，scale / growth tracker / interval 跨进程恢复一致，同窗口未缩放 loss 在预登记 `RTOL=1e-2` 内对齐）。多轮独立验证过程与原始归档证据详见 [moss_pp_v0_validation_20261112.md](../findings/moss_pp_v0_validation_20261112.md)。
* **主训练会话接线**：库级 runtime 仅负责 stage 步进与 stage checkpoint；包含 RNG、数据游标、LR scheduler 与最终 LoRA 合并落盘的完整主训练路径由 [moss_image_pp_trainer_wiring.md](moss_image_pp_trainer_wiring.md) 承载（独立验证见 [moss_pp_trainer_wiring_validation_20261114.md](../findings/moss_pp_trainer_wiring_validation_20261114.md)）。
* **未验证边界**：真实大模型权重的端到端长训练、收敛质量与通用独立推理尚未纳入本模块验收范围。
