# FasterDiT-style 收敛信号加密方案与验证记录

状态：研究记录 / 阶段快照
适用版本：以文中日期、提交和运行环境为准；不作为当前 main 操作说明

日期：2026-06-05

## 目标

在 Anima LoRA 的 DiT / flow-matching 训练路径中，低风险验证“类似 FasterDiT 通过扩展/加密训练信号加快收敛”的方向。

本轮只做不改变默认行为的实验开关、WebUI 暴露和短跑验证；不把实验项默认打开。

## 资料线索

- FasterDiT：不改 DiT 架构，通过 SNR / timestep 分布调整和 velocity direction auxiliary loss 加快 DiT 训练。
  - https://arxiv.org/abs/2410.10356
- REPA：用外部视觉表征对齐中间层，给生成模型增加表征监督。
  - https://arxiv.org/abs/2410.06940
- MDT / MaskDiT：通过 masked latent / masked patch reconstruction 增加 token 级辅助监督。
  - https://arxiv.org/abs/2303.14389
  - https://arxiv.org/abs/2306.09305
- SD-DiT：通过 self-supervised discrimination 增加判别式辅助监督。
  - https://arxiv.org/abs/2403.17004
- Min-SNR / P2：按 SNR 调整 loss weighting，缓解时间步梯度冲突。
  - https://arxiv.org/abs/2303.09556
  - https://arxiv.org/abs/2204.00227

## 当前落地范围

### 1. Velocity direction auxiliary loss

训练侧支持 `--velocity_direction_loss_weight`：

- 默认 `0.0`，不改变现有训练行为。
- 只在训练时生效，验证 FM-MSE 保持干净。
- 当前实现按 Anima latent `[B,C,H,W]` 在通道维做 per-pixel cosine direction loss。

建议短跑 sweep：

```text
0.01 / 0.03 / 0.05
```

### 2. SNR / timestep 相关实验开关

已支持：

- `--sigmoid_scale`
- `--sigmoid_bias`
- `--weighting_scheme min_snr --min_snr_gamma <gamma>`
- `--weighting_scheme p2 --p2_gamma <gamma> --p2_k <k>`

建议短跑 sweep：

```text
baseline
sigmoid_bias = -0.5 / 0.5
velocity_direction_loss_weight = 0.03
min_snr_gamma = 5
p2_gamma = 0.5
```

### 3. WebUI 暴露

WebUI 配置 catalog 新增折叠组：

```text
收敛信号实验
```

包含：

- `sigmoid_scale`
- `sigmoid_bias`
- `weighting_scheme`
- `min_snr_gamma`
- `p2_gamma`
- `p2_k`
- `velocity_direction_loss_weight`

这些字段默认折叠，避免新手误调。

### 4. Bench runner

短跑入口：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m bench.signal_probe.run_training \
  --steps 40 \
  --arms baseline dir003 minsnr5 bias_p05 \
  --seeds 42 \
  --sample-every-n-steps 999
```

本机显卡约束：

- 物理 GPU 0 = GTX 1050 4G，不用于训练。
- 物理 GPU 1 = RTX 3080 Ti Laptop 16G，用于训练。
- runner 默认 `--gpu-index 1`，并在启动前检查 torch 子进程只看到一张卡，且该卡不是 GTX 1050。

## 已完成验证

### 单测 / 静态钩子

命令：

```bash
timeout 60 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m pytest \
  tests/test_velocity_direction_loss.py \
  tests/test_anima_loss_weighting.py \
  tests/test_loss_registry.py \
  tests/test_signal_probe_runner.py \
  tests/test_gui_variants.py \
  tests/test_config.py \
  tests/test_training_frontend_state.py -q
```

结果：

```text
127 passed in 10.59s
```

### 真实训练 smoke

命令：

```bash
timeout 900 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m bench.signal_probe.run_training \
  --steps 4 \
  --arms baseline dir003 \
  --seeds 42 \
  --sample-every-n-steps 999
```

GPU 映射验证：

```json
{"count": 1, "name": "NVIDIA GeForce RTX 3080 Ti Laptop GPU", "memory_total_mb": 15982}
```

结果：

| arm | steps | returncode | avr_loss | max_memory_allocated_gb |
| --- | ---: | ---: | ---: | ---: |
| baseline | 4 | 0 | 0.12301489524543285 | 3.963097095489502 |
| dir003 | 4 | 0 | 0.12459021061658859 | 3.963097095489502 |

输出：

- `output/bench/fasterdit_signal/baseline_s42_4step/summary.json`
- `output/bench/fasterdit_signal/dir003_s42_4step/summary.json`
- `output/bench/fasterdit_signal/runs.csv`

## 后续建议

4 step 只能证明路径可运行，不能证明收敛收益。若要比较趋势，下一步至少跑：

```text
baseline / dir003 / minsnr5 / bias_p05
80-200 steps × 2-3 seeds
```

判断依据不要只看训练 loss，应同时看：

- 固定 prompt 样张
- validation FM-MSE / CMMD（如果成本可接受）
- 是否出现提示词服从性下降或低 sigma 细节不足

## 80-step 阶段记录

日期：2026-06-05

命令：

```bash
timeout 3600 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m bench.signal_probe.run_training \
  --steps 80 \
  --arms baseline dir003 minsnr5 bias_p05 \
  --seeds 42 \
  --sample-every-n-steps 999
```

GPU 映射：

```json
{"count": 1, "name": "NVIDIA GeForce RTX 3080 Ti Laptop GPU", "memory_total_mb": 15982}
```

结果：

| arm | final avr_loss | vs baseline | last10 mean | first avr | steps | elapsed_s | max_mem_gb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.116094688 | +0.000000000 | 0.116176570 | 0.101751074 | 80 | 232.7 | 3.963 |
| dir003 | 0.117563055 | +0.001468367 | 0.117643211 | 0.103125125 | 80 | 236.8 | 3.963 |
| minsnr5 | 0.094104527 | -0.021990162 | 0.095837707 | 0.101751074 | 80 | 238.5 | 3.963 |
| bias_p05 | 0.106928603 | -0.009166085 | 0.106438610 | 0.097405896 | 80 | 235.6 | 3.963 |

输出：

- `output/bench/fasterdit_signal/baseline_s42_80step/summary.json`
- `output/bench/fasterdit_signal/dir003_s42_80step/summary.json`
- `output/bench/fasterdit_signal/minsnr5_s42_80step/summary.json`
- `output/bench/fasterdit_signal/bias_p05_s42_80step/summary.json`
- `output/bench/fasterdit_signal/runs.csv`

初步观察：

- 4 组均成功跑满 80 step，且 runner 记录确认使用物理 GPU 1。
- `minsnr5` 的 final / last10 loss 明显低于 baseline，但它改变了 loss weighting，不能直接把数值优势等同于生成质量优势。
- `bias_p05` 的训练 loss 也低于 baseline，说明偏向高 sigma 结构阶段可能值得继续跑更长对照。
- `dir003` 在 80 step 的 avr_loss 略高于 baseline；方向损失会改变优化目标，不能仅凭基础日志中的 avr_loss 判死，需要样张/验证指标辅助判断。

下一步建议：

1. 对 `baseline / minsnr5 / bias_p05` 跑 200 step；`dir003` 可保留但不优先。
2. 开启固定 prompt 低频采样，或在 80-step checkpoint 上补一次相同 prompt 推理对比。
3. 若继续多 seed，优先 `42,43,44`，避免单 seed 误判。

## 200-step 阶段记录

日期：2026-06-05

命令：

```bash
timeout 7200 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m bench.signal_probe.run_training \
  --steps 200 \
  --arms baseline minsnr5 bias_p05 \
  --seeds 42 \
  --sample-every-n-steps 999
```

GPU 映射：

```json
{"count": 1, "name": "NVIDIA GeForce RTX 3080 Ti Laptop GPU", "memory_total_mb": 15982}
```

结果：

| arm | final avr_loss | vs baseline | last10 mean | last50 mean | first avr | steps | elapsed_s | max_mem_gb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.106505378 | +0.000000000 | 0.107187056 | 0.109953153 | 0.101751074 | 200 | 514.8 | 3.963 |
| minsnr5 | 0.091837470 | -0.014667908 | 0.093421917 | 0.092810887 | 0.101751074 | 200 | 515.4 | 3.963 |
| bias_p05 | 0.095798569 | -0.010706808 | 0.097461726 | 0.100163100 | 0.097405896 | 200 | 514.1 | 3.963 |

输出：

- `output/bench/fasterdit_signal/baseline_s42_200step/summary.json`
- `output/bench/fasterdit_signal/minsnr5_s42_200step/summary.json`
- `output/bench/fasterdit_signal/bias_p05_s42_200step/summary.json`
- `output/bench/fasterdit_signal/runs.csv`

初步观察：

- 3 组均成功跑满 200 step，且 runner 记录确认使用物理 GPU 1。
- `minsnr5` 继续保持最低训练 loss，但因为 Min-SNR 改变了 loss weighting，仍不能直接等价为画质更好。
- `bias_p05` 也低于 baseline，且未改变 loss weighting，只改变时间步采样分布；作为下一阶段候选更容易解释。
- baseline 从 80→200 的 avr_loss 下降明显，说明这个小数据集短跑仍在继续学习；只看 80 step 容易过早下结论。

下一步建议：

1. 对 200-step checkpoint 做固定 prompt 样张对比，优先看 `baseline / minsnr5 / bias_p05`。
2. 如果只继续一个训练分支，优先 `bias_p05`；如果继续实验分支，`minsnr5` 也保留，但需要验证未牺牲低 sigma 细节。
3. 多 seed 前先补样张，否则 loss 指标解释力不足。

## 200-step 固定 prompt 出图对比

日期：2026-06-05

命令要点：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python inference.py \
  --dit models/diffusion_models/anima-preview3-base.safetensors \
  --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
  --vae models/vae/qwen_image_vae.safetensors \
  --vae_chunk_size 64 --vae_disable_cache --attn_mode flash \
  --lora_weight <200-step checkpoint> --lora_multiplier 1.0 \
  --negative_prompt "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia" \
  --flow_shift 1.0 --sampler er_sde \
  --from_file configs/bench/signal_probe_prompts.txt \
  --save_path <per-arm output dir>
```

GPU 映射确认：

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
visible_device0 = NVIDIA GeForce RTX 3080 Ti Laptop GPU
```

注意：推理时显式使用 `models/diffusion_models/anima-preview3-base.safetensors`，与 checkpoint snapshot 中的训练基座一致；不要让 `inference.py` 走 `configs/base.toml` 默认的 `anima-base-v1.0.safetensors`。

输出：

- 对比目录：`output/bench/fasterdit_signal/samples_200step_fixed_prompts_20260605-153157/`
- 并排图：`output/bench/fasterdit_signal/samples_200step_fixed_prompts_20260605-153157/contact_sheet.png`
- 详细报告：`output/bench/fasterdit_signal/samples_200step_fixed_prompts_20260605-153157/report.md`
- 样张索引：`output/bench/fasterdit_signal/samples_200step_fixed_prompts_20260605-153157/samples.csv`

样张设置：

| prompt | seed | size | steps | CFG |
| --- | ---: | ---: | ---: | ---: |
| portrait | 42001 | 768x768 | 20 | 4.0 |
| full body dynamic | 42002 | 768x768 | 20 | 4.0 |

初步视觉观察：

- 两条 prompt 下三组均正常出图，无黑图、崩坏或明显加载错误。
- prompt01 头像：baseline 与 minsnr5 很接近；bias_p05 色调和面部细节略有变化，但差距不大。
- prompt02 全身动态：baseline 与 minsnr5 构图接近；bias_p05 构图/姿态变化更明显，更贴近“dynamic pose”的方向，但单 seed 不足以证明稳定优势。
- `minsnr5` 的训练 loss 优势没有在这组样张里转化为肉眼显著优势；仍需更多 prompt/seed 或验证指标。
- `bias_p05` 没有明显负面迹象，且只改 timestep 采样，仍是更容易解释的下一阶段候选。

下一步建议：

1. 扩充固定 prompt 到 8-12 条，覆盖头像、半身、全身、复杂手部、室内外和高细节服饰。
2. 对 `baseline / bias_p05 / minsnr5` 至少跑 `seed=42,43,44` 的 200-step 或 400-step 对照。
3. 如果只继续一个分支，优先 `bias_p05`；如果保留指标探索，再并行保留 `minsnr5`，但不要只用 loss 做最终判断。
