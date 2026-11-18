"""Chinese WebUI preflight translations for ``library.training.compat_matrix`` codes."""

from __future__ import annotations

from typing import Any


STATIC_COMPAT_MESSAGES_ZH: dict[str, str] = {
    "invalid_selective_checkpoint": (
        "selective_checkpoint 取值不受支持；请改为 off、adapter_aware、"
        "mlp_only、peak_blocks_mlp 等已知模式。"
    ),
    "invalid_compat_unverified_policy": (
        "compat_unverified_policy 取值不受支持；请改为 strict、warn 或 allow。"
    ),
    "invalid_base_compute": (
        "base_compute 取值不受支持；允许取值：bf16、fp16、none、off、"
        "w8a16_convrot、w8a8_convrot、w8a16、w8a8、nf4。"
    ),
    "loss_type_huber_schedule_unsupported": (
        "huber/smooth_l1 损失不支持 huber_schedule=snr：SNR 调度需要 "
        "noise_scheduler.alphas_cumprod，而 flow-matching 族只有 "
        "FlowMatchEulerDiscreteScheduler，训练第一步就会 NotImplementedError。"
        "请改用 huber_schedule=constant 或 exponential。"
    ),
    "convrot_block_swap_int8_mutex": (
        "base_compute 的 ConvRot 路径与 block_swap_transfer_dtype=int8 互斥"
        "（会导致双重反量化/语义冲突）；启用 ConvRot 时请使用 block_swap_transfer_dtype=bf16。"
    ),
    "nf4_block_swap_host_ram": (
        "base_compute=nf4 + blocks_to_swap 已验证可用（Params4bit 整块搬运）；"
        "主要资源约束在主机内存（pinned NF4 master 约 5.7GB），低内存机器请留意主机 RAM 预算。"
    ),
    "nf4_path_ignored": (
        "nf4_prequantized_path 仅在 base_compute=nf4 时生效，当前配置下会被忽略；"
        "请移除该路径或切换 base_compute=nf4。"
    ),
    "convrot_attn_flash_compile_dtype": (
        "ConvRot 覆盖注意力层且配合 attn_mode=flash + torch_compile 时可能产生 float32 Q/K/V"
        "（Flash 入口会自动强转兜底）；若无需量化注意力层，建议改用 convrot_scope=mlp 以省去强转开销。"
    ),
    # Audited C1 codes (docs/findings/compat_unverified_audit_20261109.md).
    "qwen_image_2_1_mixed_precision": (
        "Qwen Image 2.1 仅验收过 bf16；该 mixed_precision 在 torch/sdpa 路径会生效，"
        "但缺少真机验收（未验证）。compat_unverified_policy=strict 会拒绝该组合。"
    ),
    "qwen_image_2_1_masked_loss": (
        "Qwen Image 2.1 掩码损失已接线但缺少数据与质量验收（未验证）；"
        "compat_unverified_policy=strict 会拒绝该组合。"
    ),
    "qwen_image_2_1_loss_type": (
        "Qwen Image 2.1 官方配方仅使用 L2；其他 loss_type 会生效但未验收（未验证）。"
        "compat_unverified_policy=strict 会拒绝该组合。"
    ),
    "qwen_image_2_1_weighting_scheme": (
        "Qwen Image 2.1 官方配方使用 weighting_scheme=none；其他取值会生效但未验收"
        "（未验证）。compat_unverified_policy=strict 会拒绝该组合。"
    ),
    "z_image_loss_type": (
        "Z-Image 官方配方仅使用 L2；其他 loss_type 会生效但未验收（未验证）。"
        "compat_unverified_policy=strict 会拒绝该组合。"
    ),
    "z_image_flow_shift": (
        "Z-Image 官方配方使用 discrete_flow_shift=6.0；其他取值会生效但未验收"
        "（未验证）。compat_unverified_policy=strict 会拒绝该组合。"
    ),
    "z_image_weighting_scheme": (
        "Z-Image 官方配方使用 weighting_scheme=none；其他取值会生效但未验收"
        "（未验证）。compat_unverified_policy=strict 会拒绝该组合。"
    ),
    # Qwen Image 2.1 hard & C2 codes
    "qwen_image_2_1_mixed_precision_flash": (
        "Qwen Image 2.1 的 attn_mode=flash 只验证过 bf16 计算；fp16 + flash 会在"
        "注意力后端直接 RuntimeError，compat_unverified_policy 任何档位都不会放行。"
    ),
    "qwen_image_2_1_mixed_precision_nf4": (
        "预量化 mixed-NF4 权重要求 BF16 计算；fp16 + mixed-NF4 会在 NF4 加载器里"
        "直接报错，compat_unverified_policy 任何档位都不会放行。"
    ),
    "qwen_image_2_1_base_compute": (
        "Qwen Image 2.1 无运行期 NF4/ConvRot 在线量化；请保持 base_compute=bf16"
        "（预量化 mixed-NF4 权重同样在 bf16 下加载）。"
    ),
    "qwen_image_2_1_adaptive_precision": (
        "Qwen Image 2.1 不支持 adaptive_precision 自适应精度，请保持 off。"
    ),
    "qwen_image_2_1_attn_mode": (
        "Qwen Image 2.1 注意力仅支持 torch、sdpa、native（均为 torch 别名）以及 "
        "bf16 下的 flash，且必须关闭 xformers。"
    ),
    "qwen_image_2_1_blocks_to_swap": (
        "Qwen Image 2.1 共 32 层且至少保留 2 层常驻显存，blocks_to_swap 必须在 0 到 30 之间。"
    ),
    "qwen_image_2_1_block_swap_checkpoint": (
        "Qwen Image 2.1 开启分块交换（blocks_to_swap > 0）时必须同时开启完整 gradient_checkpointing。"
    ),
    "qwen_image_2_1_compile_backend": (
        "Qwen Image 2.1 分块编译不支持 dynamo_backend=cudagraphs。"
    ),
    "qwen_image_2_1_compile_mode": (
        "Qwen Image 2.1 分块编译仅支持默认 compile_inductor_mode=default。"
    ),
    "qwen_image_2_1_auto_block_swap": (
        "Qwen Image 2.1 尚未实现 auto_block_swap 自动分块交换。"
    ),
    "qwen_image_2_1_selective_checkpoint": (
        "Qwen Image 2.1 仅支持完整 gradient_checkpointing，请保持 selective_checkpoint=off。"
    ),
    "qwen_image_2_1_cpu_offload_checkpointing": (
        "Qwen Image 2.1 未实现 CPU/Unsloth 激活检查点卸载。"
    ),
    "qwen_image_2_1_network_train_unet_only": (
        "Qwen Image 2.1 不支持训练文本编码器，必须保持 network_train_unet_only=true。"
    ),
    "qwen_image_2_1_weighted_captions": (
        "Qwen Image 2.1 尚未实现 weighted_captions 加权标注。"
    ),
    "qwen_image_2_1_caption_dropout_rate": (
        "Qwen Image 2.1 要求 caption_dropout_rate=0（含子集配置）。"
    ),
    "qwen_image_2_1_timestep_sampling": (
        "Qwen Image 2.1 固定使用 uniform 动态偏移 sigma 采样，请保持 timestep_sampling=uniform。"
    ),
    "qwen_image_2_1_preview_text_cache": (
        "Qwen Image 2.1 训练预览要求开启文本编码器缓存（use_text_cache=true）。"
    ),
    "qwen_image_2_1_layer_start": (
        "Qwen Image 2.1 未实现 layer_start/layer_end 层范围限定。"
    ),
    "qwen_image_2_1_t_min": (
        "Qwen Image 2.1 使用完整 sigma 区间，不支持 t_min/t_max。"
    ),
    "invalid_qwen_saved_projection_blocks": "qwen_saved_projection_blocks 必须是非负整数。",
    "invalid_qwen_projection_budget_mib": "qwen_projection_budget_mib 必须是非负整数。",
    "qwen_saved_projection_contract": (
        "保存投影激活需要同时设置 qwen_fused_projections=all，且 block 数与 MiB 预算均为正整数。"
    ),
    "invalid_qwen_fused_projections": (
        "qwen_fused_projections 必须为 off、mlp、qkv 或 all。"
    ),
    "qwen_fused_projections_family": (
        "qwen_fused_projections 投影打包仅支持 qwen_image_2_1 模型族。"
    ),
    "qwen_fused_projections_plain_lora": (
        "qwen_fused_projections 投影打包仅支持标准 plain LoRA，不支持 LoKr 与 LoHa。"
    ),
    "qwen_fused_projections_mixed_precision": "实验性投影打包要求 mixed_precision=bf16。",
    "qwen_fused_projections_base_compute": "实验性投影打包要求 base_compute=bf16。",
    "qwen_fused_projections_block_swap_transfer_dtype": (
        "实验性投影打包要求 block_swap_transfer_dtype=bf16。"
    ),
    "qwen_fused_projections_gradient_checkpointing": (
        "实验性投影打包要求开启完整 gradient_checkpointing。"
    ),
    "qwen_fused_projections_selective_checkpoint": (
        "实验性投影打包要求 selective_checkpoint=off。"
    ),
    "qwen_fused_projections_cpu_offload_checkpointing": (
        "实验性投影打包要求关闭 CPU/Unsloth 检查点卸载。"
    ),
    # Krea-2 codes
    "krea2_invalid_attn_mode": (
        "Krea-2 注意力后端仅支持 torch、flash，以及 sdpa/cudnn/cudnn_sdpa"
        "（这三个都是 torch 的别名）；其余写法会被启动预检拒绝，加载阶段也会报错。"
    ),
    "krea2_compile_dynamic_seq": (
        "Krea-2 使用两张固定 token-family 编译图；训练启动时会自动关闭 compile_dynamic_seq。"
    ),
    "compile_seq_bands_requires_dynamic_seq": (
        "compile_seq_bands 只在 compile_dynamic_seq=true 时生效；训练启动时会自动关闭。"
    ),
    "krea2_compile_seq_bands": (
        "Krea-2 使用固定 token-family 编译图；训练启动时会自动关闭 compile_seq_bands。"
    ),
    "z_image_compile_seq_bands": (
        "compile_seq_bands 仅适用于 Anima native-flatten 路径；Z-Image 训练启动时会自动关闭。"
    ),
    "krea2_compile_inductor_mode": "Krea-2 仅支持 compile_inductor_mode=default。",
    "krea2_selective_checkpoint": "Krea-2 选择性检查点仅支持 off 或 every_other。",
    "krea2_v100_flash_stability": (
        "v100_flash_stability 是 Anima 专用诊断项，Krea-2 下必须为 off。"
    ),
    # Z-Image codes
    "z_image_attention_mode": (
        "Z-Image 注意力仅支持 torch、sdpa、native（均为 torch 别名）以及 flash，且必须关闭 xformers。"
    ),
    "z_image_bf16_only": (
        "Z-Image 默认要求 mixed_precision=bf16；自适应 FP16/FP32 或纯 FP32 路径需配合 torch/sdpa 注意力。"
    ),
    "z_image_base_compute": (
        "Z-Image 无运行期 NF4/ConvRot 在线量化；预量化 mixed-NF4 权重同样要求 base_compute=bf16。"
    ),
    "z_image_torch_compile": "Z-Image 暂不支持 torch_compile，请保持 torch_compile=false。",
    "z_image_block_swap_range": (
        "Z-Image 共 30 层且至少保留 2 层常驻显存，blocks_to_swap 必须在 0 到 28 之间。"
    ),
    "z_image_selective_checkpoint": (
        "Z-Image 仅支持完整 gradient_checkpointing，请保持 selective_checkpoint=off。"
    ),
    "z_image_checkpoint_offload": "Z-Image 不支持 CPU/Unsloth 激活检查点卸载。",
    "z_image_layer_range": "Z-Image 尚未实现 layer_start/layer_end 层范围限定，请清空该设置。",
    "z_image_weighted_captions": "Z-Image 尚未实现 weighted_captions 加权标注。",
    "z_image_timestep_range": "Z-Image 使用完整官方调度网格，不支持 t_min/t_max。",
    "z_image_training_sampler": "Z-Image 仅支持默认训练采样器 sampler=default。",
    "z_image_v100_flash_stability": (
        "v100_flash_stability 是 Anima 专用诊断项，Z-Image 下必须为 off。"
    ),
    "z_image_timestep_sampling": (
        "Z-Image 使用官方 shifted-uniform 调度，请保持 timestep_sampling=uniform。"
    ),
    "z_image_caption_dropout": (
        "Z-Image 暂不支持 caption_dropout_rate > 0（含子集配置），请设为 0。"
    ),
    # MOSS-Image warning
    "moss_image_int8_experimental": (
        "MOSS-Image 的 int8 分块交换传输属于实验特性：单步 loss 相对未量化路径约有 "
        "0.3%–2.0% 偏移，正式训练前请先核对生成质量。"
    ),
    # Pipeline Parallel & Checkpointing / Block Swap codes
    "pipeline_parallel_runtime_unavailable": (
        "当前模型族的流水线分层规划已接入，但主训练 loop 的 1F1B 调度尚未接入；"
        "当前会阻止启动，不会静默回退到 DDP。"
    ),
    "pipeline_parallel_experimental_runtime": (
        "流水线并行已解析到实验性的 stage-local PP v0 运行时"
        "（2 阶段 / 1F1B / fp16 事务），由 library.training.pp 驱动。"
        "Anima、MOSS-Image 与 Krea-2 已接线专用 PP 训练会话（2 阶段 / 1F1B / fp16 / "
        "plain LoRA / 完整缓存范围）；其余模型族遇到 pipeline_parallel 仍以 "
        "pp_trainer_path_unavailable 具名拒绝，绝不静默回退 DDP。"
    ),
    "pipeline_parallel_microbatches_below_stages": (
        "pipeline_parallel_microbatches 必须 >= pipeline_parallel_stages；"
        "torch 2.10 的 Schedule1F1B 与 ScheduleGPipe 都会在构造时拒绝更小的值，"
        "所以 microbatches=1 不是 2 阶段流水线的 GPipe 替代方案。"
    ),
    "negative_blocks_to_swap": "blocks_to_swap 不能小于 0。",
    "selective_full_gradient_checkpointing": (
        "selective_checkpoint 是 DiT 选择性检查点模式，不能同时开启完整 "
        "gradient_checkpointing；请关闭完整检查点，保留 selective_checkpoint。"
    ),
    "selective_cpu_offload": (
        "selective_checkpoint 不支持 CPU activation offload；请关闭 cpu_offload_checkpointing。"
    ),
    "selective_unsloth_offload": (
        "selective_checkpoint 不能和 unsloth_offload_checkpointing 同时开启。"
    ),
    "block_swap_cpu_offload": (
        "blocks_to_swap 可以和普通 gradient_checkpointing 同用，但不能和 "
        "cpu_offload_checkpointing 同时开启。"
    ),
    "block_swap_unsloth_offload": (
        "blocks_to_swap 可以和普通 gradient_checkpointing 同用，但不能和 "
        "unsloth_offload_checkpointing 同时开启。"
    ),
    "unsloth_cpu_offload": (
        "unsloth_offload_checkpointing 不能和 cpu_offload_checkpointing 同时开启。"
    ),
    "unsloth_enables_gradient_checkpointing": (
        "unsloth_offload_checkpointing 已开启，训练启动时会自动开启 gradient_checkpointing。"
    ),
    "cpu_offload_without_gradient_checkpointing": (
        "cpu_offload_checkpointing 只有配合 gradient_checkpointing 才会生效；"
        "请开启 gradient_checkpointing 或关闭 cpu_offload_checkpointing。"
    ),
    "lokr_full_checkpoint_compile": (
        "LoKr + 完整 gradient_checkpointing + torch_compile 属于实验性叠加；"
        "启动编译时会提高 Dynamo graph/accumulated 预算并稳定 graph 查找顺序，"
        "blocks_to_swap 可继续保留。"
    ),
    "block_swap_cudagraphs_disable_compile": (
        "blocks_to_swap 会在 CPU/GPU 间移动 DiT block 权重，"
        "dynamo_backend='cudagraphs' 不安全；训练启动时会关闭 torch_compile。"
    ),
    "block_swap_soft_tokens": (
        "blocks_to_swap 不支持 Soft Tokens 这类 multi-forward 方法；请保持 blocks_to_swap=0。"
    ),
    "block_swap_functional_loss": (
        "blocks_to_swap 不支持 functional_loss_weight > 0 的 multi-forward 训练；请关闭 block swap。"
    ),
    # Pre-Ampere inner FP16 switches
    "nf4_dir1_ignored": (
        "nf4_dir1 只在 base_compute=nf4 时生效；当前 base_compute 没有 NF4 Linear "
        "可加速，开关会被忽略。请关闭该开关，或切换 base_compute=nf4。"
    ),
    "nf4_dir1_requires_cuda": (
        "nf4_dir1 仅面向无原生 BF16 Tensor Core 的 sm70/sm75（V100 / 2080Ti / T4），"
        "且需要 CUDA 设备：set_dir1_enabled(True) 在无 CUDA 时会直接抛 RuntimeError。"
        "当前预检未检测到可用 CUDA，会阻止启动，不会静默回退；请保持 nf4_dir1=false。"
    ),
    "nf4_dir1_requires_pre_ampere": (
        "nf4_dir1 仅面向无原生 BF16 Tensor Core 的 sm70/sm75（V100 / 2080Ti / T4）；"
        "训练设备上 bf16 已走 tensor core，set_dir1_enabled(True) 会直接抛 ValueError。"
        "sm80+ 请保持 nf4_dir1=false 以使用原生 BF16；当前会阻止启动，不会静默回退。"
    ),
    "nf4_dir1_architecture_unknown": (
        "无法解析训练设备的 compute capability（白名单里的 GPU 查不到架构，或当前 CUDA 设备读不到算力），"
        "不能确认属于 sm70/sm75；预检降级为警告、不阻断启动，"
        "运行时 set_dir1_enabled(True) 仍会在 sm80+ 上抛错。请确认显卡架构，或关闭 nf4_dir1。"
    ),
    "krea2_attn_fp16_requires_cuda": (
        "krea2_attn_fp16 仅面向无原生 BF16 Tensor Core 的 sm70/sm75"
        "（V100 / 2080Ti / T4），需要 CUDA 设备才能拿到它使用的 SDPA 内核。"
        "当前预检看不到可用 CUDA：运行时只是把全局 flag 置真后空转、不会抛错，"
        "所以这里按「开关不生效」给出警告而不阻断启动。请保持 krea2_attn_fp16=false。"
    ),
    "krea2_attn_fp16_requires_pre_ampere": (
        "krea2_attn_fp16 仅面向无原生 BF16 Tensor Core 的 sm70/sm75"
        "（V100 / 2080Ti / T4）；训练设备已有原生 bf16 SDPA 内核，"
        "set_krea2_attn_fp16(True) 会直接抛 ValueError。sm80+ 请保持 "
        "krea2_attn_fp16=false 以使用原生 BF16；当前会阻止启动，不会静默回退。"
    ),
    "krea2_attn_fp16_architecture_unknown": (
        "无法解析训练设备的 compute capability（白名单里的 GPU 查不到架构，或当前 CUDA 设备读不到算力），"
        "不能确认属于 sm70/sm75；预检降级为警告、不阻断启动，"
        "运行时 set_krea2_attn_fp16(True) 仍会在 sm80+ 上抛错。请确认显卡架构，或关闭 krea2_attn_fp16。"
    ),
    "krea2_attn_fp16_requires_torch_attention": (
        "krea2_attn_fp16 只在 attn_mode=torch/sdpa（cudnn、cudnn_sdpa 是同义别名）"
        "且 mixed_precision=bf16 时生效：运行时代码只在 cuDNN SDPA 分支里对 bf16 "
        "输入读取该开关。attn_mode=flash 会直接走 Flash varlen 并以自身 dtype 运行，"
        "compute 不是 bf16 时也不会进入该分支，两种情况下开关都是空转。"
        "要获得 attention 内层 fp16 + GQA 展开，请把 attn_mode 切到 torch"
        "（sdpa/cudnn/cudnn_sdpa 等价）并保持 mixed_precision=bf16；否则请关闭 krea2_attn_fp16。"
    ),
    "krea2_attn_fp16_ignored": (
        "krea2_attn_fp16 只在 Krea-2（model_family=krea2_raw）的 attention 路径生效；"
        "当前模型族不会走 Krea-2 attention，开关会被忽略。请关闭该开关，或切换模型族。"
    ),
    "nf4_fp16_inner_pairing": (
        "当前设备无原生 BF16 Tensor Core（sm70/sm75）：Krea-2 + NF4 下 nf4_dir1 负责 "
        "NF4 Linear 内层 fp16 GEMM，krea2_attn_fp16 负责 attention 内层 fp16 SDPA"
        "（GQA 展开为 MHA）。当前只开启其中一个，建议两个同时开启，以获得完整的内层 fp16 收益。"
    ),
    "nf4_fp16_inner_available": (
        "当前设备无原生 BF16 Tensor Core（sm70/sm75），bf16 GEMM 会落到软件模拟路径"
        "（V100 实测 44.8 vs 11.06 s/step）。两个内层 fp16 开关都未开启。"
        "开启 nf4_dir1 可把 NF4 Linear 内层 GEMM 切到 fp16（该类 GEMM 约 4×）；"
        "若注意力后端是 torch/sdpa，krea2_attn_fp16 还可把内层 SDPA 也切到 fp16"
        "（attn_mode=flash 下该开关不生效）。两者都保持 compute_dtype=bfloat16，"
        "不要把 mixed_precision 改成 fp16——那条路径因梯度下溢已被否决。"
    ),
}

DYNAMIC_COMPAT_PREFIXES_ZH: dict[str, str] = {
    "pipeline_parallel_config": "流水线配置无效",
    "invalid_model_family": "模型族不受支持",
    "family_plain_lora_only": "当前模型族适配器类型受限",
    "qwen_nf4_compile_experiment": "Qwen NF4 编译实验配置不满足要求",
    "auto_block_swap_contract": "自动分块交换（auto_block_swap）配置约束未满足",
    "adaptive_precision_contract": "自适应精度（adaptive_precision）配置约束未满足",
    "sample_vae_decode_contract": "训练预览 VAE 解码配置不兼容",
    "qwen_preview_memory_contract": "Qwen Image 2.1 训练预览显存契约未满足",
}


def format_compat_web_message(item: Any, *, device: Any = None) -> str:
    """Translate one ``TrainingCompatIssue`` / ``TrainingCompatMutation`` for WebUI."""
    code = str(getattr(item, "code", "") or "")
    raw_message = str(getattr(item, "message", "") or "")

    if code == "block_swap_compile_mode_cudagraphs":
        value = getattr(item, "value", None)
        target = value or "默认 Inductor mode"
        return f"blocks_to_swap 不兼容 Inductor CUDAGraph mode；训练启动时会改用 {target}。"

    if code in DYNAMIC_COMPAT_PREFIXES_ZH:
        message = f"{DYNAMIC_COMPAT_PREFIXES_ZH[code]}：{raw_message}"
    elif code in STATIC_COMPAT_MESSAGES_ZH:
        message = STATIC_COMPAT_MESSAGES_ZH[code]
    elif code.startswith("moss_image_"):
        key = str(getattr(item, "key", "") or code.removeprefix("moss_image_"))
        message = f"MOSS-Image 训练契约限制（{key}）：{raw_message}"
    elif code.startswith("pipeline_parallel_"):
        message = f"流水线并行准入未通过：{raw_message}"
    else:
        message = raw_message

    if "[allow]" in raw_message and "compat_unverified_policy=allow" not in message:
        message = f"{message}（compat_unverified_policy=allow 已降级为警告：该键在本族未接线，会被忽略）"

    if device is not None and code.endswith("_requires_pre_ampere"):
        if getattr(device, "selected_capabilities", ()):
            detail = getattr(device, "device_label", "") or ", ".join(
                f"sm_{major}{minor}" for major, minor in device.selected_capabilities
            )
            message = f"{message}（被选 GPU: {detail}）"
        elif getattr(device, "capability", None) is not None:
            major, minor = device.capability
            message = f"{message}（当前预检设备 sm_{major}{minor}）"
    return message
