# MOSS-Image BF16 配置：CMP 50HX 十步基准

状态：2026-10-17 真机短程实测完成。10/10 次 optimizer 更新有效，零跳步，进程退出码 0，最终 LoRA 已保存。

后 9 步平均 **11.3386 秒/步**；训练 update 的 PyTorch 峰值 allocated/reserved 为
**5.8498/5.9277 GiB**，
整卡显存观测峰值 **6293 MiB（6.1455 GiB）**。

## 固定条件

| 项目 | 本轮值 |
| --- | --- |
| GPU | NVIDIA CMP 50HX，物理 GPU 1，sm_75；按 UUID 隔离 |
| 代码版本 | `975e14a8da818b993105b47c669896c769001de7` |
| 环境 | Python 3.13.11；PyTorch 2.12.0+cu130；CUDA 13.0 |
| 模型 | 官方 MOSS-Image-SFT 完整 DiT，5,082,308,608 个参数 |
| 精度 | DiT 参数 BF16；autocast BF16；可训练 LoRA 参数 FP32；自适应精度关闭；无 GradScaler |
| Adapter | plain LoRA，rank/alpha=32/32，33,030,144 个可训练参数 |
| 数据 | 与既有 smoke 相同的单张缓存图，256×256，batch size=1，accumulation=1 |
| 文本缓存 | NF4 TE 已建缓存；hiddens=[22,4096]，有效 token=22；训练时不加载 TE |
| Latent 缓存 | 已建磁盘缓存，latents=[128,16,16]；训练时不加载 VAE |
| 训练 | full gradient checkpointing，swap20，BF16 transfer，foreach restore，torch attention，compile=false |
| 优化器 | AdamW fused=true，learning_rate=5e-5，constant scheduler，max_grad_norm=1 |
| 随机性 | seed=42，从全新 LoRA 初始化开始，不 resume |
| 保存与预览 | 中途 LoRA/状态保存关闭；最终保存一次 LoRA；采样与验证关闭 |
| 训练 update 时间窗口 | 2026-10-17 08:57:01–08:58:55，Asia/Shanghai |

50HX 没有原生 BF16 Tensor Core 加速。本报告的 BF16 指参数与 autocast 配置，
属于当前 sm_75 路径的基准；本轮未切换到 FP16＋精度保护。

## 速度和显存

| 指标 | 结果 |
| --- | ---: |
| 首步 | 11.819823 秒 |
| 全 10 步纯 update 总耗时 | 113.867213 秒 |
| 全 10 步平均 | 11.386721 秒/步 |
| 后 9 步平均 | 11.338599 秒/步 |
| 后 9 步中位数 | 11.337132 秒/步 |
| 后 9 步最小–最大 | 11.330103–11.350107 秒/步 |
| 后 9 步吞吐 | 5.2917 updates/min |
| PyTorch 训练峰值 allocated | 5.849788 GiB |
| PyTorch 训练峰值 reserved | 5.927734 GiB |
| nvidia-smi 整卡显存观测峰值 | 6293 MiB |
| nvidia-smi 训练进程显存观测峰值 | 6262 MiB |
| 训练窗口温度 | 48–59 °C |
| 训练窗口采样功耗最大值 | 93.21 W |
| 子进程总墙钟时间，含加载和最终保存 | 162.195285 秒 |

计时在正常训练入口的 `library.training.loop._run_step` 两端执行 CUDA synchronize，
覆盖 forward、backward、block swap、梯度裁剪、optimizer 和 scheduler；
数据取 batch、外层日志、epoch 切换和最终保存位于计时区间外。
10 步全部保留，首步单列，后 9 步另作统计，不额外运行 warmup。

每次 update 开始前重置 CUDA allocator 峰值，先同步结束再读取当前与峰值显存。
上表 PyTorch 训练峰值取 10 个 update 峰值的最大值；加载峰值单独保存在 `result.json`。
`nvidia-smi` 通过训练进程内的现有 Telemetry 约每秒采样一次，
训练窗口共 104 条记录，错误 0；其峰值为采样观测值。
PyTorch allocator 数值与整卡/进程占用采用不同统计范围，分别保留。

## 逐步原始结果

| Step | 秒/update | Loss | 峰值 allocated（GiB） | 峰值 reserved（GiB） |
| --- | ---: | ---: | ---: | ---: |
| 1 | 11.819823 | 0.047247 | 5.7565 | 5.9277 |
| 2 | 11.330695 | 0.597476 | 5.8498 | 5.9277 |
| 3 | 11.330103 | 0.048278 | 5.8498 | 5.9277 |
| 4 | 11.334953 | 0.053844 | 5.8498 | 5.9277 |
| 5 | 11.337132 | 0.051721 | 5.8498 | 5.9277 |
| 6 | 11.350107 | 0.061661 | 5.8498 | 5.9277 |
| 7 | 11.338195 | 0.057979 | 5.8498 | 5.9277 |
| 8 | 11.344136 | 0.041376 | 5.8498 | 5.9277 |
| 9 | 11.335514 | 0.055587 | 5.8498 | 5.9277 |
| 10 | 11.346555 | 0.050850 | 5.8498 | 5.9277 |

该基准固定为单图、256×256、22-token 条件。后续精度 A/B 对照应保持相同 seed、缓存、
swap、attention、optimizer 和计时范围；1024 分辨率与长提示词有各自的计算和显存预算。

## 复查与复测输入

运行目录：`output/runs/moss-image-bf16-baseline10-20261017-085341`（以下均为本地未入库产物）。

- 汇总与逐步记录：`output/runs/moss-image-bf16-baseline10-20261017-085341/result.json`、`output/runs/moss-image-bf16-baseline10-20261017-085341/steps.jsonl`、`output/runs/moss-image-bf16-baseline10-20261017-085341/steps.csv`。
- 日志与遥测：`output/runs/moss-image-bf16-baseline10-20261017-085341/train.log`、`output/runs/moss-image-bf16-baseline10-20261017-085341/progress.jsonl`、`output/runs/moss-image-bf16-baseline10-20261017-085341/telemetry.jsonl`、`output/runs/moss-image-bf16-baseline10-20261017-085341/exit.json`。
- 精度、环境与缓存形状：`output/runs/moss-image-bf16-baseline10-20261017-085341/adapter/precision_report.json`、`output/runs/moss-image-bf16-baseline10-20261017-085341/environment.json`、`output/runs/moss-image-bf16-baseline10-20261017-085341/cache-shapes.json`。
- 命令与配置快照：`output/runs/moss-image-bf16-baseline10-20261017-085341/command.json`、`output/runs/moss-image-bf16-baseline10-20261017-085341/launch.py`、`output/runs/moss-image-bf16-baseline10-20261017-085341/config.toml`、`output/runs/moss-image-bf16-baseline10-20261017-085341/dataset.toml`、`output/runs/moss-image-bf16-baseline10-20261017-085341/source-method.toml`。

复测使用新的运行目录，复制 launcher 与配置，更新 `dataset_config`、`output_dir` 和
`progress_jsonl` 到新目录。配置中的模型与缓存路径指向本机既有资源；
运行目录和原始日志属于本地产物，不随业务代码提交。
