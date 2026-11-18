# 预处理 Auto Batch

状态：当前实现；真实 GPU 验证范围见 [消融记录](../findings/preprocess_auto_batch_20260906.md)。

## 配置与范围

```toml
preprocess_memory_profile = "auto"
preprocess_vae_cache_batch_size = "auto"
preprocess_text_cache_batch_size = "auto"
```

三个字段都为 `auto` 时，WebUI / task 预处理不再把它们展开成固定 `(2,16)`。
VAE 和文本编码各自从 batch 1 开始探测。自动结果只属于当前进程，日志记录批次变化，
不写回 TOML、不修改模型精度、padding、VAE 分块或训练 batch size。

批大小字段的 `auto` 跟随显存模式。其他模式继续保持固定预设：

| 模式 | VAE | 文本 |
| --- | ---: | ---: |
| auto | 自适应 | 自适应 |
| low_vram | 1 | 4 |
| balanced | 2 | 8 |
| speed | 4 | 16 |

显式正整数覆盖对应预设，**不自动退避或重试**。非法批大小明确报错。
现有配置里显式保存的整数不会被新默认覆盖。CPU auto 保守地固定为 1。

支持 `scripts.preprocess.cache_latents`、`scripts.preprocess.cache_text_embeddings`、
`scripts.krea2.preprocess_te_cache`、`scripts.z_image.preprocess_te_cache`、
`scripts.qwen_image_2_1.preprocess_te_cache` 与 `scripts.moss_image.preprocess_te_cache` 的
`--batch_size auto`（其中 `moss_image` 文本缓存因保存真实变长序列而将 `auto` 固定解析为 `1`，拒绝大于 `1` 的显式批大小；`qwen_image_2_1` Edit 配对预处理走 `scripts.qwen_image_2_1.preprocess_edit_cache` 逐对编码）；这些 CLI 未指定参数时默认使用 `auto`，共享的
`cache_latents()` / `cache_text_embeddings()` API 同样默认使用 `auto`。
训练内部的 `vae_batch_size` / `text_encoder_batch_size` 属于另外一条路径，没有改变。

## 搜索与恢复

1. 从 1 起步，排除第一次冷启动耗时，每档收集两次有效测量，再探测 2、4。
2. 统计单批峰值减去批前常驻显存，使用保守单样本增量估算容量。
   可用预算计入可复用的 PyTorch reserved 内存，受进程 allocator 限额约束，
   留出 `max(512MiB, 总显存的10%)` 安全余量，并对增量加 10% 预测余量。
3. 到 4 后，若预算允许，最多一次上拉 4 倍；实现有 batch 32 的资源上限。
   小数据集、尾批或显存不足时可能未到 4 就结束或收敛，不会重复处理图片来凑探针。
4. 较大批次吞吐提升不足 5% 时，选择已测档位中距最佳吞吐 5% 内的最小批大小。
   稳定后不反复撞上限；至少 16 个成功批次后，预算增长超过 25% 才重新上探。
5. CUDA OOM 时退出失败 encode 的异常栈，清理模型缓存和对应设备 allocator，再把失败批次
   减半重试。失败档位形成上界；成功后在边界内折半探测，避免重复尝试已知失败档位。

VAE（覆盖全部 5 个模型族的 `cache_latents`）按精确 `(W,H)` 分别学习，保留有限 CPU decode/save 流水线。
文本在 caption 变体展开后控制真正的 forward 批大小；caption 变体在重试前已生成，
不因 OOM 重新洗牌。Anima、Krea-2、Z-Image 与 Qwen Image 2.1（`qwen_image_2_1` T2I）的文本编码使用固定 padding，控制器不裁剪 token；MOSS-Image（`moss_image`）则保存按 `max_token_length` 截断的真实长度序列而非固定 max-padding，因此文本缓存固定以单条（`batch_size=1`）执行。

只有编码和 GPU 到 CPU 的结果传输属于重试边界；磁盘写入在成功编码之后执行。
非 OOM 错误不重试，模型清理失败也立即停止。batch 1 仍 OOM 或模型加载阶段 OOM 时，
需要释放显存或调整模型部署；auto 不能让放不下的模型凭空装入 GPU。

## 限制

- 这是在线测量与有界搜索，不是最优吞吐保证。桌面渲染、其他进程、温度和形状变化会影响测量。
- 探索阶段峰值可能高于最终稳定批次；显存最省的配置仍是显式 batch 1 / low_vram。
- BF16 算子会因批大小选择不同计算路径。输出格式、顺序、mask 和 padding 不变，
  但浮点张量不保证逐 bit 相同；固定整数模式可用于要求固定执行形状的复现实验。
- 不自动处理 CPU 内存不足、模型加载失败、跨进程共享调优或多设备 offload。
- 本次没有改变 ComfyUI vendor 树；发布节点若要包含这些共享源码，需按仓库规则 vendor-sync。

代码：`library/preprocess/{batch_policy,adaptive_batch,batch_config}.py`。
定向测试：`tests/test_preprocess_auto_batch.py`、`tests/test_preprocess_auto_cache.py`。
