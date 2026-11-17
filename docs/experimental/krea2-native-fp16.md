# Krea-2 原生 FP16

状态：实验性显式能力；当前仅有 CMP 90HX 三步 smoke，不是质量或长期稳定性认证。

## 范围

该模式用于官方、非量化 Krea-2-Raw DiT 的显式 native AMP FP16 训练。它是 Krea family policy，不是可套用于任意算子的通用混合精度策略。DiT base 参数以 FP16 加载并冻结，LoRA 参数保持 FP32；模型 residual 路径和 final projection 按族策略保留 FP32，梯度缩放器由 Accelerate 管理。训练契约会在配置阶段拒绝不支持的组合，并在执行时观察 guard 属性。

当前允许的窄配置：

- `model_family=krea2_raw`，`adaptive_precision=native_fp16`（必须显式选择，不支持 `auto`）。
- 官方非量化 Krea-2 checkpoint；`base_compute` 不得为 NF4。
- `network_module=networks.lora_anima` 的 plain LoRA，仅训练 DiT；不支持其他 adapter/路由扩展。
- `mixed_precision=fp16`、torch attention、full gradient checkpointing、单进程。
- `torch_compile=false`；可按显存设置 block swap。
- startup OOM retry、pipeline parallel、NF4、selective/offloaded checkpoint、compile 均不支持。

源码契约位于 `library/training/adaptive_runtime/native_precision.py` 和 `native_family.py`；Krea 专属 guard 位于 `library/models/krea2_raw/native_precision.py`。其 guard 要求官方 `SingleStreamDiT`、非量化 FP16 base 权重，并验证 DiT/text/final residual 与 final linear 的 FP32 保护。不要把 Anima 的 guard 策略或质量证据外推到 Krea。

CMP 90HX 初始 smoke 数据、CPU/独立测试和当前未完成项见 [Krea-2 native FP16 smoke findings](../findings/krea2_native_fp16_smoke_90hx_20261002.md)。

## 当前结论与下一步

当前证据只支持“所述三步配置能够完成有效更新并保存”。在进入更长训练或生产建议前，独立 checkpoint/state 保存审计已通过；应继续做受控的 20-step save/resume 验证，核对 scaler、optimizer、scheduler、sampler、随机状态及恢复后的有效更新，再定位性能差异。三步 smoke 的耗时包含启动阶段，不可当作稳态步速；Anima 的 200-step 质量证据不适用于 Krea。

## 相关 Krea-2 低精度路径分流

不要把本文档的 `adaptive_precision = "native_fp16"` 与另外两条 Krea-2 低精度路径混用：

- **双卡 Krea-2 流水线并行（PP v0）**：使用 `pipeline_parallel = true` + `mixed_precision = "fp16"`（关闭 `adaptive_precision`、关闭普通 `blocks_to_swap`，按需设每阶段 `pipeline_parallel_blocks_to_swap`，且要求 `train_batch_size % pipeline_parallel_microbatches == 0`）。详见 [分布式训练卡片](distributed-training-card.md) 与 [显存实验记录](../findings/krea2_pp_vram_gap_20261116.md)。
- **单卡 sm70/sm75（V100 / 2080Ti 等无原生 BF16 Tensor Core）NF4 内层 FP16**：外层保持 `mixed_precision = "bf16"` + `base_compute = "nf4"`，通过模板 `configs/methods/krea2_lora_nf4_fp16.toml` 同时开启 `nf4_dir1 = true` 与 `krea2_attn_fp16 = true`（配 `attn_mode = "torch"`），由内部逐层 `frexp`/`ldexp` 缩放把内层 GEMM/SDPA 切到 FP16 Tensor Core，避免整网改 `mixed_precision = "fp16"` 导致的梯度下溢偏差。详见 [内层 FP16 实验结论](../findings/krea2_nf4_fp16_inner_path_20261109.md) 与 [逐层缩放提案](../proposal/krea2_nf4_fp16_layer_scaling.md)。
