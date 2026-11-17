# Qwen Image 2.1 Edit 低显存预处理

状态：稳定

Qwen Image 2.1 Edit 的 VAE latent、reference latent 和文本条件需要成对生成。使用完整 `preprocess` 任务，它会按数据集配置 resize、复制对应 caption，再生成 Edit 缓存；不要把 Edit 当作普通 TE 或 VAE 缓存任务运行。

## 使用方式

WebUI 启动的完整训练预处理会读取该次运行的 `ANIMA_RUNTIME_CONFIG`。直接从命令行运行时，也可使用常规配置合并链：`configs/base.toml` → preset → 方法文件。通过 `METHOD` 选择方法，`METHODS_SUBDIR` 选择方法目录（缺省为 `methods`，WebUI 变体通常为 `gui-methods`），`PRESET` 选择预设。例如：

```bash
METHOD=qwen_image_2_1_lora METHODS_SUBDIR=methods PRESET=default .venv/bin/python tasks.py preprocess
```

如果要让完整任务使用特定运行时 TOML：

```bash
ANIMA_RUNTIME_CONFIG=path/to/runtime.toml .venv/bin/python tasks.py preprocess
```

Edit 独立缓存命令适用于已完成 resize、只需构建配对缓存的场景：

```bash
.venv/bin/python -m scripts.qwen_image_2_1.preprocess_edit_cache \
  --dataset_config path/to/dataset.toml \
  --vae /path/to/qwen_image_2.1_vae_bf16.safetensors \
  --qwen3 /path/to/qwen3vl_8b_bf16.safetensors
```

这里的 `--dataset_config` 必须是含 `[[datasets.subsets]]` 的**数据集 TOML**，而非训练方法 TOML。每个 subset 都要设置 `reference_image_dir`，并与目标图片目录及 caption 配对。图片与 reference 按相对路径/stem 配对，caption 必须是对应图片的同 stem sidecar，且包含非空指令。Edit 不支持 caption 多行变体、caption dropout、随机裁剪/翻转/颜色增强、wildcard 或 regularization subset。

## 低显存 TE 缓存

文本编码器以 BF16 加载。默认 `--device auto --offload auto`：设备自动选 CUDA（可用时）或 CPU；在 CUDA 上，运行时比较 Qwen3-VL core 参数字节数与当前 free VRAM 的 60%，权重超过该预留阈值时自动启用 Accelerate CPU offload，否则将模型放到设备上。查询显存失败时自动选择 offload。CPU offload 仅作用于 `model.model` 的 vision/language core；缓存路径不使用 `lm_head`，因此它不会被 offload hook 搬到 GPU。

可通过独立 CLI 覆盖设备及 offload 策略：

```text
--device auto|cpu|cuda
--offload auto|on|off
```

`--offload on` 要求执行设备为 CUDA；CPU 设备不能启用 CPU offload。`--offload off` 禁用 offload。

WebUI 的「训练配置 → 输入准备 → 预处理与数据加载」提供 **文本编码器缓存策略**。选择 Qwen Image 2.1 后可见，保存为训练配置键 `qwen_text_encoder_cache_policy`：

| 界面选项 | 配置值 | 实际执行 |
| --- | --- | --- |
| 自动（推荐） | `auto` | CUDA 可用时使用 GPU，按当前 60% 阈值决定是否卸载；无 CUDA 则 CPU |
| GPU 计算＋CPU 卸载 | `cpu_offload` | 强制 CUDA 计算，core 权重按模块从 CPU 搬运；无 CUDA 时拒绝 |
| 全部驻留 GPU | `gpu` | core 驻留 CUDA，关闭卸载；显存不足可能 OOM |
| 仅 CPU | `cpu` | Qwen3-VL 在 CPU 上运行，不改变 VAE 或 DiT 的设备选择 |

Edit 和 T2I 的预处理任务都会读取这个配置。独立缓存 CLI 对应 `--cache_policy auto|cpu_offload|gpu|cpu`。旧 `--device/--offload` 参数仍可在 auto 下使用；与非 auto 策略矛盾的参数会明确拒绝，不静默覆盖。

日志会显示 `Qwen3-VL cache policy: requested=..., execution_device=..., cpu_offload=...`。有效缓存可以直接复用，改变策略不会强制重新编码；缓存全命中时日志会说明 `encoder not loaded`。要重新编码应使用独立缓存副本或明确开启重建。`lm_head` 在所有策略下都不参与缓存，GPU 常驻策略也只移动实际使用的 core。

自动策略是经验阈值，不是对任意图片尺寸或文本长度的显存保证，当前不提供 OOM 后自动切换策略重试。10 GiB 上建议保留自动，或选择 GPU 计算＋CPU 卸载。

完整 `tasks.py preprocess` 任务可用 `--overwrite` 重建 Edit 的文本和 latent 缓存。Edit 不允许仅重建其中一种缓存；`preprocess-te` 与 `preprocess-vae` 会明确拒绝 Edit 数据集，应使用完整任务。caption 中写入的 edit 指令只是同 stem 配对输入，本功能不覆盖多图采样，也不表示该数据集已完成训练。

## 分辨率与编译缓存

Qwen Image 2.1 的 VAE 下采样 16 倍，DiT 再按 2×2 latent 分组，因此训练目标尺寸必须对齐 32 像素。预处理、训练分桶和缓存预检共同使用 Qwen 专属对齐规则，例如 `896×1200` 向下对齐为 `896×1184`；其他模型的通用 bucket 表不变。关闭放大时，按原图尺寸向下对齐，例如 `624×900` 变为 `608×896`。旧尺寸的 latent 缓存和不匹配的编辑文本缓存不会被当成有效缓存。

WebUI 启动训练时会把 `TORCHINDUCTOR_CACHE_DIR` 和 `TRITON_CACHE_DIR` 转成绝对路径，以便编译子进程切换工作目录后仍可找到生成文件。页面和任务元数据仍可显示相对路径。低显存预处理不要求关闭 DiT 的块交换或编译；它们作用于不同阶段。

## 验证边界

四个缓存策略枚举值（`auto`、`cpu_offload`、`gpu`、`cpu`）的配置校验、API 读写与阈值分支均由单元测试覆盖。CMP 90HX 10 GiB 真机已完成 `cpu_offload` / `cpu` 完整配对预处理对照，以及 50 对样本缓存与 1 轮 / 50 步 BF16 编辑训练（含 `blocks_to_swap=24`、编译、checkpoint 保存与 CPU 严格重载）；验证详情与适用边界见[热训练验收报告](../findings/qwen_edit_hot_training_20260928.md)。
