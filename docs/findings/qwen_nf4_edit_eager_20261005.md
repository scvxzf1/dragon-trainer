# Qwen Image 2.1 NF4 eager Edit 真机 smoke（2026-10-05）

状态：研究记录 / 阶段快照（单步 smoke，非完整收敛或 compile 验收）

## 结论

真实 50 对 Edit 数据在 CMP 90HX 10 GiB 上完成了 Qwen Image 2.1 全量 NF4、BF16 compute、plain LoRA 的单步 eager 训练，并成功保存 checkpoint。这个结果确认 Edit eager 训练链路可行；它不等于 50 步收敛、图片质量或 `torch.compile` 数值一致性验收。

2026-10-06 完成审计：下文 `/tmp` 权重及原始日志当前已不存在，持久化输出目录
中也未找到对应副本。本次只能追溯历史报告，无法再次读取这份 eager checkpoint；
未重新执行该 GPU 训练。后续 NF4 compile 的训练/预览/状态续训证据保存在
`output/tests/`，对应验收范围见[编译报告](qwen_nf4_compile_90hx_20261005.md)。

## 配置与结果

| 项目 | 实测值 |
| --- | --- |
| 配置 | `configs/imported/10-5-50x-nf4-qwen21-edit.toml` 的临时副本，唯一覆盖 `torch_compile=false`、`max_train_steps=1` |
| 数据 | `configs/datasets/9-27-50x-edit.toml`，a-2/b-2 共 50 对，batch=1 |
| 模型 | `qwen_image_2.1_nf4.safetensors`、Qwen3-VL 8B BF16、Qwen Image 2.1 VAE BF16 |
| GPU | CMP 90HX，10 GiB；CUDA UUID `GPU-806debd9-d254-da57-d515-ccae15b4ac6a` |
| 训练 | BF16，torch attention，full checkpoint，24/32 block swap，plain LoRA rank/alpha=16 |
| 结果 | 1/1 step，正常退出（return code 0），`loss=0.235` |
| 显存 | `max_memory_allocated=5.99 GiB`，`max_memory_reserved=6.30 GiB` |
| 权重 | `/tmp/qwen_nf4_edit_eager/nf4-edit-eager-smoke.safetensors`，约 65 MB |

训练日志：`/tmp/qwen_nf4_edit_eager_smoke_20261005_retry.log`。训练前发现既有缓存使用旧分桶（例如 `896x1200`），而当前代码建桶为 `896x1184`；使用项目自带预处理器按当前蓝图补建缓存后，50/50 Qwen3-VL 条件缓存和 50/50 target/reference latent 均通过完整性检查。预处理日志：`/tmp/qwen_nf4_edit_preprocess_20261005.log`。

## 边界

- 本次明确关闭 `torch.compile`，不能作为 NF4 compile 放行证据；完整模型 compile 数值差异仍见 `qwen_nf4_compile_90hx_20261005.md`。
- 本次训练仅做单步链路 smoke，未测长期稳定性、续训或 50 步质量。10 月 6 日追加使用该权重完成独立单参考出图，以及四参考/CFG2/两步 Edit 出图和 latent 再解码像素一致，见[独立推理报告](qwen_standalone_inference_20261005.md)。
- 预处理生成的新版 sidecar 写入既有 `post_image_dataset/9-27-50x-edit-cache`，旧文件未删除；训练输出和临时配置均在 `/tmp`。
