# 当前优化配置梳理

本文只记录当前项目中已经存在的优化相关配置事实，不评价配置好坏。整理范围限定为显存、速度、稳定性、质量、易用性和配置复用。

## 采集范围

- 配置链：`configs/base.toml`、`configs/presets.toml`、`configs/methods/*.toml`、`configs/gui-methods/*.toml`。
- 任务入口：`tasks.py`、`scripts/tasks/`、`scripts/experimental_tasks/`。
- WebUI：`web/frontend-next/src/features/training-config/domain/*`、`web/frontend-next/src/features/training-config/`。
- 维护说明：`AGENTS.md`、`docs/multi_model_support.md`、`docs/structure/anima-optimizations.md`、`docs/findings/`。
- 排除范围：`configs/imported/`、`configs/web-training-history/`、`configs/web-training-queue/`、`output/`、`models/`、`post_image_dataset/` 等用户数据或运行产物。

## 配置来源和合并链

训练配置按以下顺序合并，后者覆盖前者：

```text
configs/base.toml
  -> configs/presets.toml[<preset>]
  -> configs/<methods_subdir>/<method_or_variant>.toml
  -> CLI args
```

| 配置名 | 所在位置 | 作用 | 默认值/候选值 | 适用场景 | 风险或副作用 | UI 暴露 |
| --- | --- | --- | --- | --- | --- | --- |
| `preset` | `configs/presets.toml`、`train.py --preset`、`tasks.py PRESET=...` | 选择硬件/采样覆盖项 | `default`、`low_vram`、`low_vram_blockswap`、`balanced_16g`、`graft`、`half`、`quarter`、`tenth`、`debug` | 快速切换显存档或试跑采样比例 | 方法配置会覆盖同名键，需看最终 merge 结果 | 是，预设下拉和 guide |
| `method` | `configs/methods/*.toml`、`train.py --method` | 选择算法或模型族配置 | `lora`、`krea2_lora`、`krea2_lora_nf4_fp16`、`z_image_lora`、`qwen_image_2_1_lora`、`moss_image_lora`、`moss_image_lora_t768`、`moss_image_lora_t512`、`moss_image_lora_tecap1024`、`register`、`chimera`、`ip_adapter`、`easycontrol`、`soft_tokens`、`turbo`、`spd`、`byg`、`colorize` | CLI/任务入口训练 | family 文件可能包含多组内部开关，非自包含变体 | 部分 |
| `methods_subdir` | `train.py --methods_subdir`、`scripts/tasks/training.py` | 在 `methods` 与 `gui-methods` 间切换 | 默认 `methods`；WebUI 训练用历史命名目录 `gui-methods` | WebUI 自包含变体训练 | 选错目录会找不到配置或跑错变体 | 间接暴露 |
| `config_file` | `train.py --config_file`、Web runtime config | 直接加载 TOML 配置 | 任意 TOML 路径 | WebUI 生成 runtime 配置、CLI 复现 | 路径或内容错误会绕过预期方法/预设选择 | WebUI 内部 |
| `print-config` | `tasks.py print-config`、`train.py --print-config` | 输出最终合并配置和来源 | `METHOD=<name>`、`PRESET=<name>` | 排查覆盖关系 | 只打印不训练 | CLI |
| `base_config` | `library/config/schema.py` | TOML 父配置递归合并 | 默认无 | 复用大块配置 | 嵌套过深会增加来源理解成本 | 否 |
| `dataset_config` | 方法配置、WebUI 数据集选择 | 指向独立数据集蓝图 | 如 `configs/datasets/ip_adapter.toml`、`configs/datasets/easycontrol.toml` | 多数据集/方法专用数据布局 | 改图像、caption、分桶后需重建缓存 | 是 |

## 显存优化配置

| 配置名 | 所在位置 | 作用 | 默认值/候选值 | 适用场景 | 风险或副作用 | UI 暴露 |
| --- | --- | --- | --- | --- | --- | --- |
| `blocks_to_swap` | `configs/base.toml`、`configs/presets.toml`、方法/GUI 变体、`train.py` | 把 DiT frozen block 在 CPU/GPU 间交换，降低 GPU 驻留显存 | base 未设显式值；`default=0`；`low_vram_blockswap=8`；`balanced_16g=12`；`graft=20`；LoKr 快捷为 `23` | 16GB/低显存训练、block swap 实验 | 训练变慢；不能与 `cpu_offload_checkpointing`、`unsloth_offload_checkpointing`、Soft Tokens、`functional_loss_weight>0` 同用 | 是，含快捷按钮 |
| `block_swap_transfer_dtype` | `configs/base.toml`、`train.py`、`library/training/moss_compat.py`、WebUI catalog | block swap frozen base 权重的 CPU master/传输精度 | 默认 `bf16`；候选 `bf16`、`fp8_e4m3`、`int8` | block swap 传输带宽实验、显存极度紧张时减传输量 | `fp8_e4m3`/`int8` 会量化 frozen base 权重；MOSS 上 int8 现在可通过兼容门禁，但会打印实验性 WARNING；CMP 50HX 实测逐 step loss 偏移 0.3%~2.0%，**不建议默认开**（见 `docs/findings/moss_image_int8_blockswap_diagnosis_20261018.md`） | 是 |
| `block_swap_restore_mode` | `configs/base.toml`、`train.py`、WebUI catalog | block swap restore 阶段的 H2D 拷贝布局 | 默认 `slab`；候选 `foreach`、`slab` | 高带宽卡上减少小拷贝调度 | slab 把同 slot 多个 weight 合并成一段连续 H2D；int8 传输或混合 dtype 时自动回退 foreach；RTX 3080 基线每块约快 0.7ms | 是 |
| `block_swap_int8_restore_mode` | `library/training/cli_args.py`、`library/runtime/block_swap_config.py`、`library/runtime/offloading.py`、WebUI catalog | int8 master 的恢复方式（仅 `block_swap_transfer_dtype=int8` 时生效） | 默认 `fused`；候选 `fused`、`copy`、`reuse_storage`、`direct_bind`（别名 `single_kernel`） | 带宽受限链路（PCIe Gen1 等）上降低反量化开销与临时显存 | `fused` 用单条 `torch.mul(int8, fp32, out=half)` 直写 slot，无 fp32/fp16 临时量，目标张量非连续时直接报错而非静默写副本；CMP 50HX 实测 2.3986→1.9740 s/步、reserved 9.16→7.21 GiB；算子级逐位等价已验证，但 MOSS int8 相对未量化路径的逐 step loss 偏移为 0.3%~2.0%（见 `docs/findings/moss_image_int8_blockswap_diagnosis_20261018.md`） | 是 |
| `block_swap_int8_staging_ring` | `library/training/cli_args.py`、`library/runtime/block_swap_config.py`、WebUI catalog | `fused` 恢复使用的 staging 缓冲数量，即显存↔速度置换器 | 默认 `2`；`0`=每换出块一份（swap × 48 MiB，最快）；`1`=全局一份（48 MiB，最省）；`N`=N 份共享 | 显存卡在刀刃时压占用、显存富裕时换速度 | `0` 在 swap=10 实测 1.1204→1.0818 s/步（+3.5%）但占 swap × 48 MiB；`1`/2 是 swap=6/7 能跑起来的准入条件（2 份 9732 MiB vs 每块独立约 9932 MiB 超 9788 MiB 上限）。env 回退 `ANIMA_BLOCK_SWAP_INT8_STAGING_RING` | 是 |
| `block_swap_int8_dual_stream` | `library/training/cli_args.py`、`library/runtime/block_swap_config.py`、WebUI catalog | 把 `fused` 恢复拆成纯 DMA 流 + `priority=-1` 反量化流 | 默认 `false` | 需要稳定长训 ETA 或机器有后台负载 | 吞吐不变（swap=20 中位 2.0116 vs 单流 2.0243 s），但逐 step 极差从最高 1.59 s 收敛到约 0.015 s；双流自带 2 份 48 MiB 缓冲并忽略 `staging_ring`。env 回退 `ANIMA_BLOCK_SWAP_INT8_DUAL_STREAM` | 是 |
| `block_swap_int8_restore_chunk_rows` | `library/training/cli_args.py`、`library/runtime/block_swap_config.py`、WebUI catalog | `reuse_storage` 模式每次处理的行数 | 默认 `0`（整块） | 压低非融合模式的瞬时显存峰值 | 行数过小会增加 kernel 次数并变慢。env 回退 `ANIMA_BLOCK_SWAP_INT8_RESTORE_CHUNK_ROWS` | 是 |
| `auto_block_swap` / `block_swap_auto` | `library/training/auto_block_swap/`、`library/runtime/block_swap_auto.py`、WebUI catalog | 启动前独立校准块交换数（`auto_block_swap`）或首步优化器边界一次性晋升常驻块（`block_swap_auto`） | 默认关闭 | 未知硬件初始标定或按显存余量减少换块 | 实验功能；不支持 PP 等拓扑，详见 `docs/configuration/auto-block-swap.md` | `auto_block_swap` 是；`block_swap_auto` CLI |
| `base_compute` / `nf4_prequantized_path` / `nf4_dir1` / `krea2_attn_fp16` | `configs/base.toml`、`configs/methods/krea2_lora_nf4_fp16.toml`、`library/quantization/`、WebUI catalog | 基座量化精度（`bf16`/`nf4`/`w8a16_convrot`/`w8a8_convrot`）及 Krea-2 `sm<80` 内层 FP16 路径 | `base_compute` 默认 `bf16`；`nf4_dir1`/`krea2_attn_fp16` 默认 `false` | 低显存量化训练或 V100/Turing Krea-2 NF4 训练 | NF4 外层 compute 恒为 BF16；`nf4_dir1`/`krea2_attn_fp16` 仅限 `sm<80` Krea-2 | 是 |
| `adaptive_precision` / `adaptive_oom_retry*` | `configs/base.toml`、`library/training/adaptive_runtime/` | 实验性 `native_fp16` / `fp16_fp32` 岛屿精度与受限 OOM 重试恢复 | 默认 `off` / `false` | Anima/Krea-2/Z-Image FP16 与 OOM 恢复实验 | Qwen 与 MOSS 显式拒绝；受限契约见 `docs/experimental/adaptive-runtime.md` | CLI/TOML |
| `pipeline_parallel` / `pipeline_parallel_blocks_to_swap` | `library/models/pipeline_parallel.py`、`library/training/pp/`、WebUI 分布式卡片 | 双卡 2-stage 1F1B 流水线并行训练与 Krea-2 PP 每阶段换块数 | 默认关闭 / `0` | Anima、MOSS-Image、Krea-2 双卡 FP16 plain LoRA 训练 | 仅支持 `mixed_precision=fp16`、禁用普通 `blocks_to_swap` 与量化；`pipeline_parallel_blocks_to_swap` 仅 Krea-2 可设非零 | 是 |
| `compat_unverified_policy` | `library/training/compat_matrix.py`、`cli_args.py` | 控制未真机验收组合的预检档位 | 默认 `warn`；候选 `strict`、`warn`、`allow` | 尝试 C1 未验证组合或排障回退 | `C2-raise`/`C3`/输入校验与真矛盾在三档恒为 error | CLI/TOML（WebUI 沿用默认 `warn`） |
| `ANIMA_BLOCK_SWAP_PREFETCH_DEPTH` | `library/runtime/block_swap_config.py`、`library/runtime/offloading.py` | 块交换预取深度 K（env 变量，兼容旋钮） | 默认 `1`；`>1` 被忽略（训练/推理统一钳 `1`） | 无（已停用） | K≥2 与该设计「resident slot 数恒定、`to_cuda` 由 `to_cpu` 唯一决定」根本冲突：更深的 lead 会把尚未运行的块当退役块 park 到 CPU，forward 读到 CPU 权重（`mat2 is on cpu`，真实训练 fwd+bwd 必崩）。零显存代价下无正确 K≥2，已钳 `1`（见 `docs/findings/blockswap_baseline_20260806.md` 方向1） | CLI/env |
| `block_swap_profile_jsonl` | `configs/base.toml`、`balanced_16g`、Web runtime | 记录 block swap transfer/wait profile | 默认 `off`；候选 `off`、`auto`、显式路径；`balanced_16g=off` | 判断 H2D/D2H、prefetch runway 和 slot reuse | 诊断探针会扰动吞吐；正式长训/热测对比应关闭；完整 GPU wait timing 需显式 `ANIMA_BLOCK_SWAP_PROFILE_GPU_WAIT=1` | 是 |
| `disable_block_swap_for_eval` | `configs/base.toml`、`train.py` | sample/validation 阶段临时暂停 block swap | 默认 `false` | 训练需要 swap，但评估完整 DiT 能放进显存 | 评估阶段可能 OOM | 是 |
| `gradient_checkpointing` | `configs/base.toml`、`presets.toml`、8GB GUI 变体、`train.py` | 反向传播重算中间激活以降显存 | base `false`；`low_vram=true`；`lora-8gb/tlora-8gb/hydralora-8gb=true` | 8GB/低显存、LoKr 兜底 | 训练变慢；不能与 `selective_checkpoint` 同时开 | 是 |
| `unsloth_offload_checkpointing` | `configs/base.toml`、`low_vram`、8GB GUI 变体、`train.py` | 将 checkpoint 激活卸载到 CPU RAM | base `false`；`low_vram=true`；8GB GUI 变体 `true` | 极低显存保命 | 需要 `gradient_checkpointing=true`；不能与 block swap 或 CPU offload 同用；CPU/PCIe 压力上升 | 是 |
| `cpu_offload_checkpointing` | `train.py` | 标准 CPU activation offload | 默认 `false` | 实验性低显存路径 | 不能与 block swap、Unsloth offload 同用 | CLI |
| `selective_checkpoint` | `configs/base.toml`、`balanced_16g`、WebUI catalog、`train.py` | 选择性 DiT activation checkpoint | 默认 `off`；候选 `off`、`adapter_aware`、`peak_blocks_adapter_aware`、`every_other`、`mlp_only`、`mlp_layer1_only`、`peak_blocks_mlp`、`peak_blocks_mlp_layer1` | block swap 仍接近 OOM 时定点重算；`adapter_aware` 保留 LoRA/router 小中间值并重算 DiT 大激活 | 不能与全量 `gradient_checkpointing`、CPU offload、Unsloth offload 同用；会增加重算成本 | 是 |
| `selective_checkpoint_blocks` | `configs/base.toml`、WebUI catalog、`train.py` | 指定 peak block 重算编号 | 默认空；示例 `25-27`、`24,25,26,27` | 定位高峰 block 后微调 | 编号错误会启动失败；仅 `peak_blocks_*` 模式生效 | 是 |
| `memory_probe_jsonl` | `configs/base.toml`、Web runtime、`train.py` | 记录训练阶段 CUDA 显存、adapter、optimizer 摘要 | 默认 `off`；候选 `off`、`auto`、显式路径 | OOM 排查、短跑诊断 | 观测工具，不降显存；每步记录会放大日志 | 是 |
| `memory_probe_max_steps` | `configs/base.toml`、WebUI catalog、`train.py` | 控制 memory probe 详细 step 数 | 默认 `2`；候选 `1`、`2`、`3`、`5`、`0` | OOM 前几步定位 | `0` 表示每步记录，不适合长训 | 是 |
| `peak_probe_jsonl` | `configs/base.toml`、Web runtime、`train.py` | 记录更细粒度 DiT/LoKr CUDA 峰值事件 | 默认 `off`；候选 `off`、`auto`、显式路径 | LoKr/MLP 峰值定位 | `ops/lokr/full` 会扰动 compiled graph，只适合短跑 | 是 |
| `peak_probe_max_steps` | `configs/base.toml`、WebUI catalog、`train.py` | 控制 peak probe step 数 | 默认 `2`；候选 `1`、`2`、`5`、`0` | 短跑峰值定位 | 过大影响速度统计和日志体积 | 是 |
| `peak_probe_level` | `configs/base.toml`、WebUI catalog、`train.py` | peak probe 粒度 | 默认 `block`；候选 `block`、`ops`、`lokr`、`full` | 从 block 边界逐步深入到 LoKr delta | 粒度越高越扰动性能 | 是 |
| `mixed_precision` | `configs/base.toml`、`train.py`、WebUI catalog | 训练混合精度 | base `bf16`；候选 `bf16`、`fp16`、`no` | NVIDIA 新卡优先 bf16 | 旧卡不支持 bf16；fp16 稳定性较弱 | 是 |
| `save_precision` | `configs/base.toml`、`train.py`、WebUI catalog | 保存权重精度 | base `bf16`；候选 `bf16`、`fp16`、`float` | 减少权重体积/保持训练精度 | 低精度保存可能影响后处理兼容性 | 是 |
| `use_vae_cache` | `configs/base.toml`、`train.py`、WebUI catalog | 使用预处理 VAE latent 缓存 | 默认 `true` | 避免训练时重复 VAE encode | 图像或 VAE 改动后需重建缓存 | 是 |
| `use_text_cache` | `configs/base.toml`、`train.py`、WebUI catalog | 使用文本编码器输出缓存 | 默认 `true` | cache 后释放 TE，给 DiT 腾显存 | caption/tokenizer 改动后需重建缓存；训练 TE 时不能同时缓存 | 是 |
| `skip_cache_check` | `configs/base.toml`、`train.py`、WebUI catalog | 跳过缓存完整性检查 | 默认 `true` | 稳定复训、减少启动检查 | 可能训练中途才发现缓存缺失或过期 | 是 |
| `cache_llm_adapter_outputs` | `configs/methods/lora.toml`、`chimera.toml`、GUI LoRA 系变体 | 缓存 LLM adapter 输出 | LoRA/Hydra/Chimera 多为 `true` | 路由/文本特征重复使用 | 依赖文本缓存；训练 LLM adapter 时不兼容 | 是 |
| `vae_chunk_size` | `configs/base.toml`、WebUI catalog | VAE 编码/解码分块大小 | 默认 `64`；候选 `16`、`32`、`64`、`128` | 预处理、采样阶段显存/速度平衡 | 越大越快但峰值越高 | 是 |
| `vae_disable_cache` | `configs/base.toml`、WebUI catalog | 禁用 VAE 内部缓存 | 默认 `true` | 显存紧张 | 可能牺牲少量速度 | 是 |

## 训练效率配置

| 配置名 | 所在位置 | 作用 | 默认值/候选值 | 适用场景 | 风险或副作用 | UI 暴露 |
| --- | --- | --- | --- | --- | --- | --- |
| `optimizer_type` | `configs/base.toml`、方法配置、`library/training/optimizers.py`、WebUI catalog | 选择优化器 | base `AdamW`；候选含 `AdamW`、`CAME`、`AdamW8bit`、`Lion`、`Prodigy`、`ProdigyPlusScheduleFree`、`Automagic` | 收敛速度、显存和优化器状态权衡 | 切换优化器需要重新理解学习率和 scheduler；部分依赖额外包 | 是 |
| `optimizer_args` | `configs/methods/*`、WebUI catalog | 传优化器额外参数 | 例如 `weight_decay=1e-2`、`weight_decay=1e-4` | fused、weight decay、方法实验 | 不支持的参数会启动失败 | 是 |
| `lr_scheduler` | `configs/base.toml`、WebUI catalog | 学习率调度器 | base `cosine`；候选含 `constant`、`constant_with_warmup`、`cosine`、`polynomial`、`lulu_loss_gated_cosine` | 控制训练中后期学习率 | 与 optimizer 不匹配会让训练经验失效 | 是 |
| `lr_warmup_steps` | `configs/base.toml`、WebUI catalog | 学习率 warmup 步数或比例 | base `0.05`；可填整数或 `<1` 比例 | 开头稳定性 | 短跑中过长会只看到 warmup | 是 |
| `train_batch_size` | `train.py`、WebUI 表单、dataset blueprint merge | 训练 batch size | CLI 默认 `1`；WebUI 默认 `1` | 显存充足时提高吞吐 | 最容易 OOM；改变每轮 step 数 | 是 |
| `gradient_accumulation_steps` | `train.py`、WebUI 表单 | 梯度累积 | 默认 `1` | 用低显存模拟更大有效 batch | 更新频率下降，训练变慢 | 是 |
| `sample_ratio` | `configs/presets.toml`、`train.py` | 每轮采样数据比例 | `half=0.5`、`quarter=0.25`、`tenth=0.1`、`debug=0.001` | 快速冒烟、短实验 | 结果不能代表完整训练 | 是 |
| `max_data_loader_n_workers` | `train.py`、WebUI catalog、部分方法配置 | DataLoader worker 数 | CLI 默认 `1`；WebUI 候选 `0/2/4/8`；Chimera/Soft Tokens 常设 `1` | 数据加载速度 | 多进程占主机内存；调试时更复杂 | 是 |
| `dataloader_pin_memory` | `configs/base.toml`、`train.py`、WebUI catalog | DataLoader pinned memory | base `true` | 加速 CPU 到 GPU 数据搬运 | 占主机内存 | 是 |
| `persistent_data_loader_workers` | `configs/base.toml`、`train.py`、WebUI catalog | worker 跨 epoch 常驻 | base `true` | 多轮训练减少 worker 重启 | 持续占进程/内存，调试不便 | 是 |
| `dataloader_prefetch_factor` | `train.py` | DataLoader prefetch factor | CLI 默认 `1` | worker>0 时调整预取 | 当前未写入 base，也未进入 Web 表单 | CLI |
| `log_every_n_steps` | `configs/base.toml`、WebUI catalog | step 日志频率 | base `2` | 平衡日志细节和 I/O | 太密会增加 I/O/前端刷新压力 | 是 |
| `sample_every_n_epochs` | `train.py`、WebUI 表单 | 按 epoch 生成样张 | 默认空/关闭 | 训练中观察质量 | 样张越频繁训练越慢 | 是 |
| `sample_every_n_steps` | `train.py`、WebUI 表单 | 按 step 生成样张 | 默认空/关闭 | 长 epoch 数据集提前观察 | 设太小会频繁打断训练 | 是 |
| `sample_at_first` | `train.py`、WebUI 表单 | 训练前先生成样张 | 默认 `false` | 验证采样链路 | 启动变慢；显存紧张时也可能 OOM | 是 |
| `sample_sampler` | `train.py`、WebUI catalog | 训练预览采样器 | 默认 `euler`；Web 候选 `euler`、`er_sde`、`lcm` | 预览速度/观感对照 | 与最终推理采样器不同会影响判断 | 是 |
| `save_every_n_epochs` | 方法/GUI 配置、`train.py` | 普通权重保存间隔 | LoRA GUI 多为 `2`；IP/Easy/Soft 多为 `4/12` | 挑训练轮次 | 保存频繁占磁盘 | 是 |
| `checkpointing_epochs` | 方法/GUI 配置、`train.py` | 可恢复状态保存间隔 | LoRA GUI 多为 `2`；base 无显式 | 中断恢复 | checkpoint-state 体积大；只保留最近状态时需理解覆盖行为 | 是 |
| `torch_compile` | `configs/base.toml`、`presets.toml`、WebUI catalog | 启用 PyTorch compile blocks | base `true`；`low_vram_blockswap=false`；`balanced_16g=true` | 长训提速、Anima native bucket 编译路径 | 首次启动慢；与 CUDAGraph block swap 模式存在兼容限制 | 是 |
| `compile_inductor_mode` | `train.py`、WebUI catalog | Inductor mode | 默认 `None/default`；候选 `default`、`reduce-overhead`、`max-autotune`、`max-autotune-no-cudagraphs` | 编译性能消融 | block swap 会避开 CUDAGraph 模式；收益依环境 | 是 |
| `attn_mode` | `configs/base.toml`、方法配置、WebUI catalog | 注意力后端 | base `flash`；候选 `torch`、`xformers`、`flash`、`sageattn`、`flex`、`sdpa` | 性能/兼容性切换 | 高性能后端依赖 CUDA/PyTorch/显卡支持 | 是 |
| `use_custom_down_autograd` | `configs/base.toml`、WebUI catalog、LoKr playbook | 自定义 LoRA/LoKr 反向路径 | base `false` | 降低 LoRA/LoKr 临时峰值或提升吞吐的显式实验开关 | 底层 autograd 优化，异常时需回退到 `false` 排查 | 是 |

## 训练质量和稳定性配置

| 配置名 | 所在位置 | 作用 | 默认值/候选值 | 适用场景 | 风险或副作用 | UI 暴露 |
| --- | --- | --- | --- | --- | --- | --- |
| `timestep_sampling` | `configs/base.toml`、WebUI catalog | 时间步采样策略 | base `sigmoid`；候选 `sigma`、`uniform`、`sigmoid`、`shift`、`flux_shift` | flow matching 训练分布 | 改变噪声阶段覆盖，结果不可直接与 baseline 对比 | 是 |
| `sigmoid_bias` | `configs/base.toml`、WebUI catalog、实验配置 | sigmoid/logit-normal 偏置 | base `0.0` | 结构/细节阶段分布消融 | 偏置过大会训练分布过窄 | 是 |
| `discrete_flow_shift` | `configs/base.toml`、WebUI catalog | flow matching shift | base `1.0` | 调整噪声调度 | 反馈不直观，可能偏离推理预期 | 是 |
| `weighting_scheme` | `train.py`、WebUI catalog | loss 时间步权重 | 默认 `uniform`；候选 `min_snr`、`p2`、`sigma_sqrt`、`cosmap` 等 | SNR/时间步权重实验 | loss 数值不可直接对比 | 是 |
| `min_snr_gamma` | WebUI catalog、训练参数 | Min-SNR gamma | Web 默认 `5.0` | `weighting_scheme=min_snr` | 过强会影响细节阶段 | 是 |
| `p2_gamma` / `p2_k` | WebUI catalog、训练参数 | P2 loss 权重形状 | Web 默认 `1.0` / `1.0` | `weighting_scheme=p2` | 增加实验维度 | 是 |
| `velocity_direction_loss_weight` | WebUI catalog、训练参数 | FasterDiT 风格速度方向辅助损失 | Web 默认 `0.0` | 收敛信号实验 | 权重过大可能压过 MSE 幅值学习 | 是 |
| `masked_loss` | `configs/base.toml`、方法配置、WebUI catalog | mask 区域损失控制 | base `true`；EasyControl/Colorize 有覆盖 | 带字/漫画/需要忽略区域的数据 | mask 错误会忽略应学习区域 | 是 |
| `caption_dropout_rate` | 方法/GUI 配置、`train.py`、WebUI catalog | caption dropout | LoRA/Chimera/IP 多为 `0.1`；GUI IP/Easy `0.2` | 风格鲁棒性、图像条件训练 | 太高会降低提示词服从性 | 是 |
| `use_shuffled_caption_variants` | LoRA/Chimera/GUI 变体、WebUI catalog | 使用 caption 打乱变体 | 多数 LoRA 系变体 `true` | 降低 caption 顺序依赖 | caption 噪声会被放大 | 是 |
| `validation_split_num` | `configs/base.toml`、方法配置、WebUI 数据集字段 | 固定验证数量 | base `0`；LoRA/Chimera 方法常 `8` | 留出验证/CMMD | 小数据集会减少训练样本 | 是 |
| `use_cmmd` | `train.py`、GUI 变体、WebUI catalog | CMMD validation 信号 | CLI 默认 `true`；多数 GUI 变体写 `false` | 质量验证 | 采样和 PE encoder 路径增加显存/时间 | 是 |
| `validation_baselines` | `ip_adapter.toml`、`train.py`、WebUI catalog | 方法 adapter 验证基线 | IP 方法 `false`；CLI 默认 `true` | IP-Adapter delta 诊断 | 每个 baseline 增加额外 val forward | 是 |

## 模型和 adapter 配置

| 配置名 | 所在位置 | 作用 | 默认值/候选值 | 适用场景 | 风险或副作用 | UI 暴露 |
| --- | --- | --- | --- | --- | --- | --- |
| `network_module` | `configs/base.toml`、方法配置、WebUI catalog | 选择 adapter/network 实现 | base `networks.lora_anima`；候选含 IP/Easy/Soft Tokens | 切换方法实现 | 选错模块会导致权重格式/参数不匹配 | 是 |
| `network_dim` / `network_alpha` | 方法/GUI 配置、WebUI 表单 | LoRA/adapter rank 与缩放 | plain LoRA GUI `32/32`；IP `16`；Easy GUI `16/16`；VeRA `256/256` | 容量/速度/显存平衡 | rank 越大显存和过拟合风险越高 | 是 |
| `lora_adapter_kind` | WebUI catalog | 在表单中选择 LoRA/LoHa/LoKr/GLoRA/VeRA 类别 | 默认 `lora`；候选 `lora`、`loha`、`lokr`、`glora`、`vera` | 同一表单切换 adapter 类型 | 需要同步清理互斥 flags | 是 |
| `use_ortho` | `lora.toml`、`gui-methods/ortholora.toml`、`tlora*.toml`、WebUI catalog | 启用 OrthoLoRA 正交参数化 | LoRA family 当前 `true`；plain GUI LoRA 无 | 减少更新污染、T-LoRA 组合 | 训练机制更复杂，合并/兼容需确认 | 是 |
| `use_timestep_mask` | `lora.toml`、`tlora*.toml`、VeRA、WebUI catalog | 启用 T-LoRA 时间步 rank mask | T-LoRA/VeRA `true` | 不同去噪阶段分配容量 | 增加 `min_rank` 等超参 | 是 |
| `min_rank` / `alpha_rank_scale` | T-LoRA/Hydra/Chimera/VeRA 配置 | 控制时间步 mask 最低秩与 power-law 日程指数 | T-LoRA GUI `min_rank=8`；Hydra 8GB `1`；VeRA `64` | 时间步容量控制 | 过低可能欠拟合局部阶段 | 是 |
| `channel_scaling_alpha` | `lora.toml`、Chimera、Turbo/SPD network | 通道缩放强度 | 常见 `0.5` | LoRA/实验方法通道缩放 | 与方法实现耦合 | 是 |
| `use_moe_style` | `lora.toml`、Hydra GUI | Hydra/FeRA MoE 结构 | 候选 `false`、`shared_A`、`independent_A`；Hydra 多为 `shared_A` | 专家路由类 adapter | 显存、速度、推理兼容性复杂度上升 | 是 |
| `route_per_layer` | `lora.toml`、Hydra GUI、WebUI catalog | 是否逐层路由 | LoRA method 当前 `true`；Hydra GUI `true` | MoE 路由细粒度 | 路由统计和调参复杂 | 是 |
| `router_source` | `lora.toml`、Hydra GUI、WebUI catalog | 路由信号来源 | 候选 `none`、`input`、`sigma`、`fei`、`crossattn_emb`；LoRA method `input`；Hydra GUI `sigma` | Hydra/FeRA/文本全局路由消融 | `crossattn_emb` 只适用于 network-level 全局路由；信号来源不同，结果不可直接混比 | 是 |
| `num_experts` | `lora.toml`、Hydra GUI、WebUI catalog | 专家数 | LoRA method `4`；Hydra GUI `6` | MoE 容量 | 专家越多显存/速度成本越高，可能利用不均 | 是 |
| `balance_loss_weight` / `balance_loss_warmup_ratio` | LoRA/Hydra/Chimera 配置 | 专家均衡 loss | LoRA `1e-7`/`0.4`；Chimera `1.0`/`0.1` | 防止专家坍缩 | loss 过强可能干扰主任务 | 是 |
| `router_targets` | LoRA/Hydra/Chimera 配置 | 路由作用层正则 | 常见 `cross_attn.output_proj` 与 `mlp.layer[12]`；Hydra 8GB 仅 MLP | 控制 MoE 插入范围 | regex 错误会影响训练层覆盖 | 是 |
| `use_lokr` / `lokr_factor` | `gui-methods/lokr.toml`、WebUI catalog | 启用 LoKr Kronecker adapter | `use_lokr=true`；`lokr_factor=8` | LyCORIS LoKr 格式、复杂风格 | 16GB 下峰值紧张；推理需 LoKr 支持 | 是 |
| `lokr_grouped_delta_backend` | `gui-methods/lokr.toml`、LoKr plugin | LoKr delta forward 后端 | 默认 `triton`；兼容/对照可设 `eager` | 大 token/大矩阵 LoKr 加速 | 不兼容设备/布局回退 eager；首次 Triton 编译有额外开销 | 是 |
| `lokr_grouped_delta_backward_backend` | `gui-methods/lokr.toml`、LoKr plugin | LoKr grouped-delta backward 后端 | 默认 `triton_grad_w1_w2_grad_x`；可切换分阶段 Triton 或 `eager` | full `w2`、标量 gate、连续 CUDA tensor | full-gradient 组合已通过 RTX 3080 组件验证和 CMP 170HX 3-seed 端到端热测；长训质量和 Triton 编译异常降级仍是残余风险 | 是 |
| `lokr_factor_group_size` | `gui-methods/lokr.toml`、LoKr plugin、WebUI catalog | LoKr grouped projection 分组 | 默认 `8`；候选 `1`、`2`、`4`、`8` | LoKr 速度/显存平衡 | 值越大越快但临时激活越大；OOM 时退到 `4/2/1` | 是 |
| `lokr_project_chunk_bytes` | `gui-methods/lokr.toml`、WebUI catalog | LoKr row chunk 字节阈值 | 默认 `4194304`；候选 `1MiB` 到 `16MiB` | 细化 LoKr delta apply 峰值 | 小值更慢，大值更容易 OOM | 是 |
| `use_loha` | `gui-methods/loha.toml`、WebUI catalog | 启用 LoHa | GUI `loha` 变体 `true` | 需要 LoHa/LyCORIS 格式 | 与 LoKr/GLoRA/VeRA 等互斥 | 是 |
| `use_glora` | `gui-methods/glora.toml`、WebUI catalog | 启用 GLoRA | GUI `glora` 变体 `true` | GLoRA 兼容实验 | 推理/继续训练需识别 GLoRA 权重 | 是 |
| `use_vera` / VeRA 参数 | `gui-methods/vera.toml`、WebUI catalog | 启用 VeRA，控制随机投影 | `use_vera=true`；`projection_prng_key=0`；`d_initial=0.1`；`save_projection=false` | 极低参数 adapter 消融 | 加载端需按种子重建或保存投影 | 是 |
| `add_reft` / `reft_*` | `gui-methods/reft.toml`、`tlora_ortho_reft.toml`、WebUI catalog | 残差流干预 | ReFT GUI `reft_dim=64`、`layers=last_8`；组合变体 `32` | 语义干预增强 | 合并/推理兼容性弱于普通 LoRA | 是 |
| `use_chimera_hydra` / Chimera 参数 | `configs/methods/chimera.toml`、`gui-methods/chimera_hydra.toml`、WebUI catalog | 内容池+频率池双路由 Hydra | GUI `content=4`、`freq=2`；method `content=6`、`freq=2` | 方法实验、内容/频率专家分工 | 字段多，显存和解释成本高 | 是 |
| `use_ip_adapter` / IP 参数 | `configs/methods/ip_adapter.toml`、`gui-methods/ip_adapter.toml`、WebUI catalog | 图像 cross-attention adapter | `network_dim=16`；`resampler_layers=2`；`heads=8`；`ip_scale=1.0` | 参考图条件训练 | 需要 PE 特征/identity pair 数据准备；验证 baseline 额外耗时 | 是 |
| `use_easycontrol` / Easy 参数 | `configs/methods/easycontrol.toml`、`gui-methods/easycontrol.toml`、WebUI catalog | 图像 self-attn/FFN 条件控制 | GUI `network_dim=16`；`cond_token_count=4096`；`b_cond_init=-10` | 图像控制训练 | 专用数据集和缓存目录；条件 token 多会占显存 | 是 |
| `network_args` | IP/Easy/Soft Tokens/LoKr 等方法、WebUI network arg specs | 方法额外参数透传 | `key=value` 字符串数组 | 方法内部调参 | 格式错误或未知参数会启动失败 | 部分 |

## WebUI 暴露面

| 配置名 | 所在位置 | 作用 | 默认值/候选值 | 适用场景 | 风险或副作用 | UI 暴露 |
| --- | --- | --- | --- | --- | --- | --- |
| 自包含变体文件 | `configs/gui-methods/*.toml` | 每个变体一个自包含训练配置；目录名为历史兼容名称 | 当前有 `chimera_hydra`、`easycontrol`、`glora`、`hydralora`、`hydralora-8gb`、`ip_adapter`、`loha`、`lokr`、`lora`、`lora-8gb`、`lora-convrot-vram`、`lora-v100-stable`、`lora_signal_probe`、`ortholora`、`reft`、`soft_tokens`、`tlora`、`tlora-8gb`、`tlora_ortho_reft`、`vera` | WebUI 选择训练方法 | 变体文件会覆盖 preset 同名键；`lora_signal_probe` 为维护/诊断变体，不是新手默认入口 | 是 |
| 表单分类 `optimization` | `web/frontend-next/src/features/training-config/stageGroups.ts`、`domain/config-field-catalog.js` | 将优化字段集中到资源/训练计划分组 | 包含显存与速度、LoKr 专用、数据加载与 VAE、实验性功能 | 用户查找关键开关 | 表单默认值与最终 merge 值需要看当前配置 | 是 |
| 资源快捷按钮 `全 GPU` | `web/frontend-next/src/features/training-config/resourceGroups.ts`、`trainingForm.ts` | 关闭 block swap/probe/offload，保持 compile | `blocks_to_swap=0`、`torch_compile=true` | 显存充足、最快路径 | 显存不足会 OOM | 是 |
| 资源快捷按钮 `Balanced 16G` | `web/frontend-next/src/features/training-config/resourceGroups.ts`、`configs/presets.toml[balanced_16g]` | 普通 LoRA 16GB block swap 档 | `blocks_to_swap=12`、`bf16`、`profile=off`、`gradient_checkpointing=false` | 16GB 普通 LoRA 优先档 | 不等同 LoKr 稳定档；需要诊断时手动打开 profile | 是 |
| 资源快捷按钮 `FP8 测试` | `web/frontend-next/src/features/training-config/resourceGroups.ts` | FP8 block swap 传输消融 | `blocks_to_swap=12`、`block_swap_transfer_dtype=fp8_e4m3`、probe auto | 传输实验 | 不建议默认训练，存在量化误差 | 是 |
| 资源快捷按钮 `更省显存` | `web/frontend-next/src/features/training-config/resourceGroups.ts` | 增加 block swap 数 | `blocks_to_swap=16`、`profile=off` | 16GB 普通 LoRA 更省显存 | 比 Balanced 更慢 | 是 |
| 资源快捷按钮 `LoKr 16G` | `web/frontend-next/src/features/training-config/resourceGroups.ts`、LoKr playbook | LoKr 专用救场 | `blocks_to_swap=23`、`lokr_factor_group_size=8`、`memory_probe=auto`、`profile=off` | LoKr 16GB 首次试跑 | 余量很薄；仍可能需 allocator 环境变量；需要 block swap 归因时手动开 profile | 是 |
| 资源快捷按钮 `OOM 兜底` | `web/frontend-next/src/features/training-config/resourceGroups.ts` | block swap + selective checkpoint | `blocks_to_swap=12`、`selective_checkpoint=mlp_only`、`profile=off` | 普通路径仍 OOM | 会变慢；不适合作为 LoKr 首选 fallback | 是 |
| Web 全局设置 | `configs/web-ui-settings.toml`、`web/services/settings_service.py` | 输出根目录、配置根和界面比例 | `output_root=output/runs` | WebUI runtime 输出和界面默认值 | 文档不应固化本机绝对路径 | 是 |
| 全局模型配置库 | `configs/web-ui-settings.toml`、`web/services/model_config_service.py` | 有序五模型族（Anima/Krea-2/Z-Image/Qwen Image 2.1/MOSS-Image）路径组合与只读能力标签 | `model_config_library.items`；默认项镜像 `[global]` 路径与 `model_family` | 新建预设、配置页选择和旧消费者兼容 | 默认项切换会影响空路径配置的回退值 | 是 |

## 实验方法配置

| 配置名 | 所在位置 | 作用 | 默认值/候选值 | 适用场景 | 风险或副作用 | UI 暴露 |
| --- | --- | --- | --- | --- | --- | --- |
| Turbo / DP-DMD | `configs/methods/turbo.toml`、`tasks.py exp-turbo`、`scripts/experimental_tasks/training.py` | 2-step LoRA student 蒸馏 | `iterations=2000`、`batch_size=1`、`student_rank=64`、`teacher_cfg=4`、`student_steps=2` | 快速推理 student 实验 | bespoke loop，绕过普通 `train.py/accelerate`；结果和普通 LoRA 训练不可直接比较 | CLI 实验入口 |
| SPD | `configs/methods/spd.toml`、`tasks.py exp-spd`、WebUI SPD 配置区 | 多分辨率轨迹 adapter | `iterations=4000`、`rank=32`、`stages=[0.5,1.0]`、`transition_sigmas=[0.5]` | SPD sampler 轨迹训练 | 专用 `scripts/distill_spd.py`，普通 Web 训练不应直接启动 | Web 可编辑/CLI 启动 |
| ChimeraHydra | `configs/methods/chimera.toml`、`gui-methods/chimera_hydra.toml`、`tasks.py exp-chimera` | 内容池+频率池双路由 OrthoHydra | method `network_dim=64`、`blocks_to_swap=8`；GUI `network_dim=32`、`blocks_to_swap=0` | 路由分工实验 | 参数多、显存/解释成本高 | 是 |
| IP-Adapter | `configs/methods/ip_adapter.toml`、`gui-methods/ip_adapter.toml`、`tasks.py exp-ip-adapter` | 图像条件 cross-attention | method `blocks_to_swap=2`、`ip_features_cache_to_disk=true`；GUI `blocks_to_swap=0` | 参考图控制 | 需要 `preprocess-pe`/PE feature cache；identity pair 配置依赖 caption index | 是 |
| EasyControl | `configs/methods/easycontrol.toml`、`gui-methods/easycontrol.toml`、`tasks.py exp-easycontrol` | 图像条件 self-attn/FFN 控制 | method `network_dim=32`、`gradient_checkpointing=true`、`unsloth=true`；GUI `network_dim=16`、`blocks_to_swap=0` | 图像控制训练 | 专用数据集/缓存；条件 token 数影响显存 | 是 |
| Soft Tokens | `configs/methods/soft_tokens.toml`、`gui-methods/soft_tokens.toml`、`tasks.py exp-soft-tokens` / `exp-test-soft` | 每层/每时间桶软文本 token | method `n_layers=6`、`n_t_buckets=14`、contrastive/softrank 参数；GUI `n_layers=10`、`n_t_buckets=100` | 文本条件容量实验 | 不支持 block swap；推理需经 `exp-test-soft` / `--soft_tokens_weight` 或 `AnimaSoftTokensLoader` | 是，实验 |
| Qwen NF4 编译实验 | `qwen_nf4_compile_experiment`、`docs/experimental/qwen-nf4-compile.md` | Qwen Image 2.1 Edit 在指定受限 NF4 组合下启用编译 | 默认 `false` | CMP 90HX 受限 Edit NF4 编译实验 | 仅限 BF16 Edit、plain LoRA rank/alpha16、torch attention、full checkpoint、swap24/foreach；普通 NF4 compile 仍拒绝 | 是，实验 |
| BYG | `configs/methods/byg.toml`、`tasks.py exp-byg` | unpaired instruction-editing 训练 | `network_dim=64`、`max_train_epochs=2`、`gradient_checkpointing=true`、`blocks_to_swap=0` | 编辑类实验 | 明确保持 `unsloth_offload_checkpointing=false`；需先构建 BYG sidecars | CLI 实验入口 |
| Colorize EasyAdapter | `configs/methods/colorize.toml`、`tasks.py exp-easycontrol EASYADAPTER=colorize` | EasyControl colorize adapter | `network_dim=32`、`gradient_checkpointing=true`、`unsloth=true`、`masked_loss=false` | Sketch2Manga/colorize 实验 | 需要额外 adapter 权重和专用数据集 | CLI 实验入口 |

### int8 块交换的三套玩家档位（CMP 50HX / PCIe Gen1 实测）

以下组合都已在 10 步单图短程验证，参数可直接写进 TOML（与 `ANIMA_*` 环境变量并存，TOML/CLI 优先）：

| 档位 | blocks_to_swap | transfer_dtype | int8_restore_mode | int8_staging_ring | int8_dual_stream | 实测 |
| --- | ---: | --- | --- | ---: | --- | --- |
| A 极限榨干 | 7 | int8 | fused | 2（触顶时改 1） | true | 0.8793 s/步（相对 bf16/swap20 3.33×），整卡 9512 MiB |
| B 显存富裕提速 | 10~20 | int8 | fused | 0 | false | swap=10 1.0818 s/步（比 ring=2 快 3.5%） |
| C 零量化误差 | 7~10 | bf16 | — | — | — | swap=10 1.4810 s/步；无 int8 的 0.3%~2.0% loss 偏移 |

训练启动时会打印一行生效组合，便于贴日志求助与对照 benchmark：

```text
[BlockSwap] swap=7, transfer_dtype=int8, restore_mode=slab, int8_restore_mode=fused, int8_scope=all, staging_ring=2 (2 x 48 MiB), dual_stream=True
```

该行同时写入 `block_swap_profile_jsonl` 的 `block_swap_config` 事件（字段 `int8_staging_ring`、`int8_dual_stream`）。

## 当前事实性注意点

- `balanced_16g` 当前是普通 LoRA 的 16GB block swap 优先档：`blocks_to_swap=12`、`torch_compile=true`、`selective_checkpoint=off`、`block_swap_profile_jsonl=off`。
- LoKr 16GB 路径单独维护：当前快捷按钮使用 `blocks_to_swap=23`、`lokr_factor_group_size=8`、`memory_probe_jsonl=auto`，不是 `balanced_16g` 的简单延伸。
- `block_swap_transfer_dtype=fp8_e4m3` 已在 WebUI 暴露为实验项，但维护报告结论是保留实验开关、默认仍用 `bf16`。
- `block_swap_profile_jsonl=auto` 用于定位问题，不是速度基线；要比较版本吞吐时应改为 `off`，否则 observer 线程、JSONL 写入和可选 GPU timing event 都会进入测量口径。
- `selective_checkpoint`、full `gradient_checkpointing`、Unsloth offload、block swap 之间有硬兼容边界，配置矩阵需要显式标注。
- `configs/methods/` 是 family 配置；`configs/gui-methods/` 是用户可见自包含变体。整理 UI 暴露状态时应优先看 `gui-methods` 和 WebUI catalog。
