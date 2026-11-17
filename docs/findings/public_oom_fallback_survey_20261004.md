# 公开训练器 OOM 回退源码调研

状态：调研记录 / 阶段快照
日期：2026-10-04。这是公开源码研究，没有执行这些项目的 GPU 测试，也没有修改本项目的训练行为。来源固定到调研时默认分支的 commit；当前开发 HEAD 的实现可能不同于用户安装的发布版本。

## 主要结论

没有在本次核查的路径中找到可以直接合入本项目、同时保证模型/optimizer/数据游标/RNG 事务一致性的通用 OOM 续训实现。

常见做法是缩 batch 后同进程重调函数、启动短校准后恢复检查点，或诊断并终止。与本项目接近的扩散训练器多使用显式 swap 和显式 checkpoint resume。CPU 预览解码回退、启动校准和训练中途恢复是不同能力，不能合并成“支持自动 OOM 恢复”。

本项目的启动期新进程 swap 重试值得保留。优先补齐 CUDA/host OOM 分类及真实启动 OOM 验收；中途恢复应先补完整、可验证的已提交 checkpoint 和数据状态，再接自动重启。

## 项目对照

下表中“未见自动恢复”仅指本次列出的已读路径，不代表穷举了项目所有模型、插件和分布式分支。

| 项目 | 已核查的处理 | 重试和状态边界 | 对本项目的价值 |
| --- | --- | --- | --- |
| Accelerate | `find_executable_batch_size` 捕获匹配异常、清缓存、缩 batch | 同进程重新调用整个函数；不负责回滚参数/optimizer/RNG | 借鉴隔离的启动 batch 搜索；不能充当中途事务恢复 |
| Transformers Trainer | 显式 `auto_find_batch_size` 包装完整 inner training loop | 同进程重入；普通路径不能保证权重、optimizer、进度共同回滚 | 分开训练/eval OOM；谨慎处理有效 batch 变化 |
| Lightning | fit 开始时保存临时 checkpoint，短 trial 搜索 batch，最终恢复 | 恢复已保存训练状态；trial 间不是逐次完整回滚，无通用 RNG 精确恢复保证 | 启动探针结束后恢复正式训练状态 |
| musubi-tuner | 显式 blocks-to-swap；Kandinsky5 预览 VAE 解码失败后 CPU 重做 | 已读训练主 loop 没有 OOM retry；CPU fallback 只在 sampling | 单独设计预览回退，保留训练配置边界 |
| sd-scripts | Anima ControlNet backward OOM 打印形状/显存并重新抛出 | 诊断后停止；普通 LoRA 主 loop 未见自动 retry | 完整记录失败阶段、shape 和显存 |
| SimpleTuner | VAE cache OOM 退出，训练异常上报后抛出；支持显式 resume | checkpoint 保存 sampler 状态；恢复后重新应用 block-swap 放置 | 完整检查点、sampler 和恢复后的设备布局 |
| OneTrainer | 已读 GenericTrainer/CLI 未见自动训练 OOM 重试 | fused back pass 可在 backward 内更新单个参数 | 不以“optimizer.step 尚未调用”泛化判断没有更新 |
| Axolotl | 显式将 `auto_find_batch_size` 传给 HF Trainer | 委托框架，不是自有 OOM checkpoint 回退 | 用配置清楚表达能力来源 |
| LLaMA-Factory | 已读 SFT 使用 HF Trainer，并显式传 resume checkpoint | 本地已读路径未见主动开启 batch 搜索或 OOM catch/retry | 显式 resume 不等于失败后自动恢复 |
| torchtune | 单卡 full-finetune recipe 直接训练，epoch 末保存状态 | 已读路径未见自动 OOM retry；其说明不支持 mid-epoch checkpoint | 明确恢复粒度和会丢弃的进度 |

## 框架源码依据

### Accelerate

固定 commit：`01c73fbdb9a7cdbf3c750c22160e7568f2c339c0`。

- [OOM 分类](https://github.com/huggingface/accelerate/blob/01c73fbdb9a7cdbf3c750c22160e7568f2c339c0/src/accelerate/utils/memory.py#L100-L117) 使用 RuntimeError + 字符串，覆盖 GPU/MPS/CPU allocator/HPU 和 cuDNN unsupported；不是仅 CUDA typed OOM 的窄分类。
- [缩 batch 与重调用](https://github.com/huggingface/accelerate/blob/01c73fbdb9a7cdbf3c750c22160e7568f2c339c0/src/accelerate/utils/memory.py#L162-L192)：本 commit 默认 `int(batch_size * 0.9)`，失败时 GC/device cache cleanup 后再次调用函数，最终到 0 停止。不要把旧版“减半”描述用于这个版本。
- [free_memory](https://github.com/huggingface/accelerate/blob/01c73fbdb9a7cdbf3c750c22160e7568f2c339c0/src/accelerate/accelerator.py#L3929-L3956) 清内部对象引用与 step；不是权重/optimizer 回滚。包装器也不自动重建模型、optimizer、scheduler、RNG 或 dataloader。

函数如果已执行更新，再次调用能否从初始状态开始取决于函数自身。装饰器不能提供这项保证。SIGKILL、普通 MemoryError、NaN 和 timeout 不属于这里的匹配缩 batch 路径。

### Transformers Trainer

固定 commit：`469230357aab0f2b303b0d638c1f8d06edb14184`。

- [train 包装入口](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/trainer.py#L1540-L1599) 与 [委托 Accelerate](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/trainer_utils.py#L938-L966)：`auto_find_batch_size` 包装 `_inner_training_loop`，不是一个独立新进程校准器。
- [model_init 在外层 train](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/trainer.py#L1478-L1489)，不是每次 inner retry 都重新执行；[optimizer 创建有 None 条件](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/trainer.py#L1255-L1269)。
- [prepare 与 restore](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/trainer.py#L1697-L1786)：重入会清内部引用、重建 dataloader/进度；普通路径的旧 optimizer 可以保留，Trainer 自建 scheduler 可重新创建。普通模型权重的 resume 加载位于外层 train。DeepSpeed/FSDP 有不同逻辑，本报告不外推它们。
- [effective batch 公式](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/trainer.py#L2531-L2547) 是 micro batch × accumulation × data-parallel world size；缩 micro batch 没有自动补偿 accumulation。
- [eval OOM 阻断](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/trainer.py#L3216-L3235)：避免评估 OOM 被错误地用于缩小训练 batch。

这些代码不能证明任意中途 OOM 的精确 rollback。普通路径可能保留已改权重/optimizer，却重置 loop 进度及重建 scheduler；没有 resume 时也没有对应的 RNG 回滚。显式 resume 时，inner 重入会重新加载 optimizer/scheduler/scaler，但普通模型权重仅在外层加载，仍不能据此保证一致回滚。不要直接复制为本项目的续训语义。

### Lightning

固定 commit：`84df182f50ab34301aabb3c0eb4031815bfb413d`。

- [fit 开始触发](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/src/lightning/pytorch/callbacks/batch_size_finder.py#L187-L205)，默认每 trial 3 步。power 策略成功扩大、失败减半；binsearch 可设置余量。
- [trial 前保存与 finally 恢复](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/src/lightning/pytorch/tuner/batch_size_scaling.py#L76-L111)。
- [trial loop 和 dataloader reset](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/src/lightning/pytorch/tuner/batch_size_scaling.py#L389-L418)：trial 会真实更新；trial 间恢复 loop progress，不是每次加载完整 checkpoint。
- [最终 restore 链](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/src/lightning/pytorch/trainer/connectors/checkpoint_connector.py#L233-L258) 与 [训练状态 restore](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/src/lightning/pytorch/trainer/connectors/checkpoint_connector.py#L291-L309) 恢复已保存的模型、optimizer、scheduler、precision/scaler、loop 等。
- [OOM 分类](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/src/lightning/pytorch/utilities/memory.py#L49-L80) 同样包含 CPU allocator 和 cuDNN unsupported；[恢复代码 TODO](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/src/lightning/pytorch/tuner/batch_size_scaling.py#L154-L177) 说明仍有额外状态待 reset。

这是有价值的启动校准范式，但不提供新进程隔离，也不能承诺通用 RNG、worker 预取、外部 IO 或自定义状态完全回滚。SIGKILL 不能执行 finally。找到 batch 后也不自动调整 accumulation。

## 扩散训练器源码依据

### musubi-tuner / sd-scripts

- musubi commit `f8a1b03794a49239a3539015075f5123d6c07d66`：[显式 swap 初始化](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/training/trainer_base.py#L1645-L1678)、[训练主 loop](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/training/trainer_base.py#L2176-L2220)。所读路径没有 OOM 后自动增加 swap 或重放 optimizer 事务。
- [Kandinsky5 sampling CPU fallback](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/kandinsky5_train_network.py#L370-L392) 对 VAE decode 的所有 Exception 清显存并转 CPU 重做。这是预览解码，不是训练重试；宽泛 catch 也不能直接作为 OOM 分类模板。
- sd-scripts commit `690ea7f96c23182352ec63def76d431c6120bd2f`：[Anima ControlNet backward OOM 诊断后 raise](https://github.com/kohya-ss/sd-scripts/blob/690ea7f96c23182352ec63def76d431c6120bd2f/anima_train_control_net_lllite.py#L835-L857)，记录 step、epoch、latent/prompt/conditioning shape 和 memory summary；[普通 LoRA 更新 loop](https://github.com/kohya-ss/sd-scripts/blob/690ea7f96c23182352ec63def76d431c6120bd2f/train_network.py#L1617-L1631) 未见本地 OOM retry wrapper。

### SimpleTuner

固定 commit：`d6b5939104d7d09a08e6fe8a63a0f34407004d5c`。

- [VAE 缓存 OOM 退出](https://github.com/bghira/SimpleTuner/blob/d6b5939104d7d09a08e6fe8a63a0f34407004d5c/simpletuner/helpers/caching/vae.py#L1914-L1931)；[训练失败上报/抛出/finally cleanup](https://github.com/bghira/SimpleTuner/blob/d6b5939104d7d09a08e6fe8a63a0f34407004d5c/simpletuner/helpers/training/trainer.py#L2037-L2080)。这里不是自动缩 batch 或新进程重试。
- [checkpoint 保存](https://github.com/bghira/SimpleTuner/blob/d6b5939104d7d09a08e6fe8a63a0f34407004d5c/simpletuner/helpers/training/trainer.py#L6682-L6737) 包含 accelerator state、sampler state、临时保存路径和 manifest。
- [resume](https://github.com/bghira/SimpleTuner/blob/d6b5939104d7d09a08e6fe8a63a0f34407004d5c/simpletuner/helpers/training/trainer.py#L5113-L5157) 恢复 accelerator 后重新施加 block-swap 设备放置，并加载 sampler 状态。保存/加载调用存在，不等于已证明本项目 dataset/prefetch/RNG 可以精确重放。
- [GPU health 分类](https://github.com/bghira/SimpleTuner/blob/d6b5939104d7d09a08e6fe8a63a0f34407004d5c/simpletuner/helpers/training/gpu_circuit_breaker.py#L716-L751) 覆盖 CUDA/cuDNN/cuBLAS/NCCL/OOM/illegal access；它不能用于决定所有这些异常都可安全重试。

不要复制其删除无效 checkpoint 的选项；本项目应保留失败产物和用户训练历史。

### OneTrainer

固定 commit：`23df3832e8f213d64c337e39f9365ba46e99fed0`。

- [GenericTrainer 更新主路径](https://github.com/Nerogar/OneTrainer/blob/23df3832e8f213d64c337e39f9365ba46e99fed0/modules/trainer/GenericTrainer.py#L753-L793) 与 [CLI](https://github.com/Nerogar/OneTrainer/blob/23df3832e8f213d64c337e39f9365ba46e99fed0/scripts/train.py#L47-L58) 的已读路径未见自动 OOM 恢复。
- [fused back pass](https://github.com/Nerogar/OneTrainer/blob/23df3832e8f213d64c337e39f9365ba46e99fed0/modules/trainer/GenericTrainer.py#L560-L605)：`register_post_accumulate_grad_hook` 可调用 `optimizer.step_parameter`。启用该选项时，backward 中途失败前就可能已修改部分参数/optimizer。

因此“尚未调用常规 optimizer.step，所以 backward OOM 可以直接重跑”的假设不能推广到全部训练器。`zero_grad()` 和 `empty_cache()` 都不会撤销参数或 optimizer moments。

## LLM 训练器源码依据

- Axolotl commit `19b890c537490f38efb1b121b7b37f7234e715f2`：[传递 auto_find_batch_size](https://github.com/axolotl-ai-cloud/axolotl/blob/19b890c537490f38efb1b121b7b37f7234e715f2/src/axolotl/core/builders/causal.py#L228-L238)。这是 HF 委托，不是 Axolotl 自有事务回滚。其周期 empty-cache 配置也不等于 OOM recovery。
- LLaMA-Factory commit `ce9dc9e072f80fa3abe0989d4ab90da25f083438`：[SFT 构建和显式 resume](https://github.com/hiyouga/LLaMA-Factory/blob/ce9dc9e072f80fa3abe0989d4ab90da25f083438/src/llamafactory/train/sft/workflow.py#L105-L121)、[Trainer 继承](https://github.com/hiyouga/LLaMA-Factory/blob/ce9dc9e072f80fa3abe0989d4ab90da25f083438/src/llamafactory/train/sft/trainer.py#L47-L48)。本次没有证明其底层参数是否可在所有模式开启 batch 搜索，不将 HF 能力自动认作本地已启用。
- torchtune commit `bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1`：[单卡 loop](https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/recipes/full_finetune_single_device.py#L571-L619)、[checkpoint 粒度说明](https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/recipes/full_finetune_single_device.py#L93-L98)。已查 recipe 的 optimizer/recipe state 按 epoch 保存，不支持 mid-epoch checkpoint。未穷举 FSDP/distributed recipe。

## 建议落地顺序

1. 统一明确 CUDA OOM、host OOM、模糊分配错误、非有限值、timeout/cancel 的分类；记录 phase、shape、dtype、swap、有效更新数和资源峰值。illegal access、NCCL 故障等不能合并为容量不足重试。
2. 保留启动期新进程重试，固定精度和有效 batch，仅增加 swap；完成 Anima native FP16 真实 CUDA OOM -> 新进程 -> 有效更新与保存的验收后，再单独扩展 Krea。batch=1 的扩散训练无法依赖继续缩 batch 解困。
3. 将预览 VAE 的低显存回退作为独立功能；仅对明确 OOM 使用 tiled/CPU 解码，记录与训练主 loop 无关的回退事件。
4. 中途恢复先实现已提交 checkpoint 的原子发布、完整 optimizer/scheduler/scaler/RNG/sampler/bucket 数据状态和预取边界协议；新进程加载后重新施加 swap 设备布局，验证恢复游标，再连接自动重试。
5. 验收必须覆盖“optimizer 已部分修改后 OOM”的故障注入，并明确回滚到最近提交点可能重跑多少更新。若不能验证这些状态，保持停止行为。

这是研究建议，未将任何外部重试逻辑合入当前项目。当前本地实现边界见 [Anima native 启动 OOM 静态验收](anima_native_fp16_startup_oom_static_20261002.md)及 [Krea native smoke](krea2_native_fp16_smoke_90hx_20261002.md)。
