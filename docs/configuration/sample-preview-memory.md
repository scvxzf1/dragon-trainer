# 训练预览 VAE 解码与块交换策略

状态：稳定

适用于全部 5 个已注册模型族（Anima、Krea-2、Z-Image、Qwen Image 2.1 与 MOSS-Image / `moss_image`）的训练内预览和训练停止后的延迟解码。
前端 `/next/training` 的「训练计划 / 训练预览」提供通用字段「预览 VAE 解码」；Qwen Image 2.1 另提供专属预览块交换覆盖。
通用 `sample_vae_decode` 仅控制 VAE 解码阶段；各模型族的 DiT 块交换默认沿用训练期配置，仅 Qwen 支持见下文的采样期覆盖。

```toml
sample_vae_decode = "tiled"
```

| 值 | 行为 |
| --- | --- |
| `tiled`（默认） | 主动在目标设备分块解码；不自动转 CPU |
| `auto` | 沿用 VAE 原 tiling 状态进行目标设备解码；仅 CUDA OOM 时尝试分块 GPU，再尝试分块 CPU |
| `cpu` | 分块 CPU 解码，保留 VAE 原精度 |

正常训练目标设备为 GPU；CPU 验证环境仍可调用相同接口。
分块可能增加耗时并改变局部像素。默认值用于减少预览解码的显存峰值，不保证峰值低于训练。
策略不改变训练 latent 缓存、模型权重或预处理编码策略。

## 模型差异

- Anima / Krea-2 共用本地 Qwen VAE；保留 `decode_to_pixels()` 内部的 mean/std 还原和 4D/5D 约定。
- Z-Image 使用 Diffusers Flux `AutoencoderKL`；保留 `latent / scaling_factor + shift_factor`。
  其默认 tiling 阈值等于配置中的 1024，开启 tiling 后解码 1024 图仍可能走整图。
  预览策略因此临时将像素 tile 上限设为 512，并按 VAE 空间倍率同步 latent tile，退出恢复原阈值。
- Qwen Image 2.1 保留专属 64 通道 latent、mean/std 和编辑参考图结果保存逻辑。
- MOSS-Image（`moss_image`）使用专属 Flux.2 VAE 解码 `[B, 128, H, W]` 归一化 latent（`library/models/moss_image/latent.py::decode_moss_latents`），复用同一套放置、分块、OOM 回退与恢复策略。

## Qwen 预览块交换覆盖

训练内采样默认继承当前训练 DiT 的块交换数量、传输精度与恢复方式。针对 Qwen Image 2.1，`/next/training` 的「训练计划 / 训练预览」及 CLI/TOML 额外提供独立的采样期 DiT 块交换覆盖（其他模型族隐藏并拒绝这两个 Qwen 专属键）：

| 配置键 | 默认值 | 行为 |
| --- | --- | --- |
| `qwen_sample_swap_mode` | `"inherit"` | `"inherit"` 继承训练 `blocks_to_swap`；`"custom"` 启用自定义采样交换数量 |
| `qwen_sample_blocks_to_swap` | `0` | 自定义交换块数，合法整数范围 `0..30`（`0` 表示采样时全部块驻留 GPU） |

切回 `"inherit"` 会忽略已保存的合法 `qwen_sample_blocks_to_swap`，便于后续再次切回 `"custom"`。TOML、CLI 与 Web 预检共用同一套校验（`library/models/qwen_image_2_1/preview_memory.py`）：

```toml
blocks_to_swap = 24
block_swap_transfer_dtype = "bf16"
qwen_sample_swap_mode = "custom"
qwen_sample_blocks_to_swap = 28
sample_vae_decode = "tiled"
```

核心运行契约：

- **BF16 无损传输要求**：当 `"custom"` 目标块数与训练 `blocks_to_swap` 不同时，仅允许 `block_swap_transfer_dtype = "bf16"` 无损传输，并要求当前训练更新已提交、梯度已清空；它不改写训练配置文件，也不替换 adapter / optimizer 对象。
- **`count=0` 调度入口预装**：当训练关闭块交换（`blocks_to_swap = 0`）而预览配置为 `"custom"` 且 `qwen_sample_blocks_to_swap > 0` 时，模型加载阶段会在 adapter 挂载和 `torch.compile` 之前预装 `count=0` 的块交换调度入口，避免采样阶段动态重包装已编译 forward。
- **退出恢复与跨 rank 协调**：采样结束后必须恢复原训练交换块数、训练模式与 RNG 状态；入口切换、采样或退出恢复发生错误均视为致命错误并汇总同步到所有 rank，绝不带着不确定的块驻留状态继续训练或让其他 rank 卡在尾部 barrier。
- **不做去噪 OOM 自动增块**：Qwen 当前不设去噪 OOM 自动上调交换块数的动态契约；采样默认保留 B 路径，不启用 DiT K/V 缓存。
- **独立基准脚本**：`scripts/qwen_image_2_1/sampling_benchmark.py` 读取训练配置的 `blocks_to_swap`、传输精度和恢复方式，并复用上述 `qwen_sample_swap_*` 覆盖与 `sample_vae_decode` 策略，记录训练/采样有效块数及实际解码分支（逐图 `decode_s` 包含策略切换与设备迁移耗时；固定条件对照脚本 `sampling_prepare_ab.py` 仍保持原 `swap24/BF16/slab` 契约不变）。

## 恢复与失败

解码前将 latent 保持在 CPU；具体解码回调按 VAE 当前设备搬运，CPU 分支不会提前搬 latent 到 GPU。
退出恢复 VAE 原设备、精度、tiling 开关和临时 tile 阈值。恢复失败为致命错误，各 rank 在尾部 barrier
前协调失败，不能让其他 rank 无限等待或继续使用不确定的训练状态。

CUDA OOM 回退会先清理异常帧与 VAE feature cache；普通错误不触发自动回退。
保存 PNG（以及已有的元数据/日志）成功后才删除对应 latent；普通单图失败保留 latent 供重试。

`torch.OutOfMemoryError` 与 `torch.cuda.OutOfMemoryError` 在当前 PyTorch 中可能是同一类型。
因此回退还要求错误明确包含 CUDA 与 out-of-memory 信息；CPU/host OOM、来源不明的 OOM、
NaN、超时和非法内存访问都不会降级重试。预览失败不发送训练 OOM 通知，也不修改训练精度、
batch、optimizer 或 `adaptive_oom_retry`。

进入 CPU 阶段前，先检查主机可用 RAM（尊重 cgroup v2 限制，不把 swap 当作 RAM）。
额外内存估算包含需要从设备复制的 VAE 参数/buffer、latent 转换、分块工作区及整图像素转换；
再保留至少 2 GiB 或可见总 RAM 的 10%。估算超出余量时，拒绝 CPU 迁移和解码，保留 latent。
分块工作区使用保守启发式，检查是一次性预算而非操作系统硬配额；其他进程、模型内部临时分配
和实际分块实现仍可能导致 CPU OOM。CPU OOM 只结束该图解码并保留 latent，不触发训练自适应重试。

预览事件独立追加到 `output_dir/sample/vae-decode-events.jsonl`，记录 latent 文件名、策略阶段、
attempt、明确 OOM、fallback 的下一阶段、CPU estimated/available/reserve 字节数、success、
reject 和 restoration_fatal。事件日志写入失败仅 warning，不把解码或训练变成失败。
success 表示解码返回图像；若随后的状态恢复失败，还会记录 restoration_fatal 并停止训练。

## 旧配置

`qwen_sample_vae_decode` 是 `sample_vae_decode` 的兼容别名；旧 TOML/CLI 仍可使用。
同一配置层同时包含新旧键时，新键优先，与 TOML 顺序无关。配置层间仍遵守正常合并顺序，
因此方法配置中的旧键会覆盖基础配置默认值，显式 `auto` / `cpu` 不会被新默认改写。
新界面只显示和写入通用键；读取旧配置时也能显示原策略。

## 验证边界

单元测试覆盖各注册模型族的策略校验、`qwen_sample_swap_*` 切换/恢复、别名合并、新键优先、latent 归一化、PNG 成功后删除、失败保留和跨 rank 恢复错误传播。

CMP 90HX 上的 1024×1024 BF16 真实 VAE 解码测量（每策略预热 1 次、正式 3 次中位数；计时含设备迁移与状态恢复，`auto` 未触发 OOM 即走整图 GPU）：

| 模型族 | 整图 → 分块耗时 | 整图 → 分块峰值 GiB |
| --- | --- | --- |
| Anima | 0.71 → 1.37 秒 | 4.37 → 0.51 |
| Krea-2 | 0.70 → 1.36 秒 | 4.37 → 0.51 |
| Z-Image | 0.63 → 0.87 秒 | 2.55 → 0.77 |
| Qwen Image 2.1 | 1.68 → 2.40 秒 | 7.21 → 1.07 |

分块解码验证了真实 tiled 分支进入及设备/精度/tiling/tile 阈值恢复；需要严格保持整图解码像素时可显式选 `auto`。另有 Anima 真实 Qwen VAE 的小图容量故障注入验证（`auto` 完整走通 GPU → tiled GPU → CPU BF16 回退链、保存 PNG 后删除 latent 并恢复原状态；CPU BF16 仅作保底路径，不承诺大图吞吐）。本地归档路径为 `output/tests/qwen-vae-tiled-ab-20261004-r1/REPORT.md`、`output/tests/all-family-vae-preview-20261004-r2/REPORT.md` 与 `output/tests/preview-cpu-fallback-90hx-20261004-r2/`；完整恢复与预览证据见 [交付报告](../findings/anima_committed_oom_preview_delivery_20261004.md)。
