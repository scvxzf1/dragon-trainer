# 多模型采样批次与 tiled 块交换生命周期验收

状态：验收记录 / 阶段快照
日期：2026-10-02。范围：Anima、Krea-2、Z-Image 训练预览，以及 Anima tiled 推理。
Qwen Image 2.1 继续默认方案 B；本轮没有扩展其他模型族的 DiT K/V 缓存。

## 实现结论

训练预览通过 `library/training/preview_lifecycle.py` 统一管理一批样张的状态。
在改变 adapter 模式、清理 timestep mask 或切换块交换之前保存 CPU/当前 CUDA RNG
和 adapter 原始 training 状态；目录创建、全部 prompt 预测及退出清理处于保护区内。
入口失败、预测失败和恢复失败都会进入原有跨进程失败汇总，避免某个 rank 提前进入 barrier。
各项恢复独立尝试，RNG 最后恢复，覆盖恢复 hook 自身消耗随机数的情况。

删除三族每个 prompt 前的重复 prepare。启用块交换且走正常 inference/training switch
时，P 个 prompt 的批次原有 P+2 次初始化减为进入/退出各一次，共 2 次。
采样禁用块交换时仍走模型自己的 pause/resume；解码阶段因移动模型而需要的重新初始化保留。
Anima sample compile-range 准备仍先于 pause/switch，不改变编译和 adapter apply 顺序。

Anima tiled 推理通过 `library/inference/block_swap_lifecycle.py` 包住连续去噪预测，
删除每个 tile 的正/负分支重复 prepare。退出时按入口 `forward_only` 模式恢复合法的
canonical 驻留布局，不尝试恢复中断时旋转的物理位置。无 swap 时完全不访问 offloader；
非法模型契约以及不支持 forward-only 的 sparse CPU master 继续拒绝。
原 router、FEI 和 P-GRAFT 清理路径保留，helper 不改变模型或 adapter 的 train/eval 状态。

若恢复本身失败，会传播错误；有原始异常时通过 exception note 附加恢复错误。
这表示“失败也尝试全部恢复”，不保证损坏的 offloader 或设备错误后一定能继续训练。
热点采样入口只保留编排，新增状态管理放在上述两个独立 helper 中。

## 验收证据

定向 CPU 联合测试：140 passed、8 skipped，退出状态 0（31.70 秒）。
覆盖生命周期、三族训练预览、真实双进程 Gloo 正常分片及入口/采样失败、
Anima compile 顺序、Z-Image swap、Krea-2 compile、Qwen 预览和 B 默认开关。

独立审阅未发现实现阻塞，但发现初版 GPU fixture 的标量 identity 权重可能因可交换性
和 BF16 舍入漏检权重错位，且 sample stub 没主动消耗 RNG。本轮补强了这两项证据：

- 每块使用不同的非交换通道混合矩阵和通道 adapter；CPU 检查 30 块相邻矩阵不同且不交换。
- 每次 `wait_for_block` 后直接核对当前 payload 对应独立 CPU 权重快照，退出再核对驻留设备和权重。
- 每个 sample 主动消耗 CPU/CUDA 随机数，再检查正常/失败退出后的 RNG 快照相等。

补强后的专项测试：34 passed，退出状态 0（13.82 秒），其中 8 项真实 CUDA/slab 检查：

| 路径 | 检查内容 | 结果 |
| --- | --- | --- |
| tiled，6 blocks / swap2 | 16 次连续预测与全 GPU 基线相等；每块 payload 正确；中间块失败恢复；恢复后 adapter backward | 通过 |
| tiled，30 blocks / swap26 | 非整除槽位的 16 次连续预测、全 GPU 基线、权重归属、中间失败恢复 | 通过 |
| 三族实际 sample_images 入口，各正常/第二 prompt 失败 | 两种尺寸、模拟 12 次 CFG 正负预测调用、prepare=2、adapter 原 eval 状态、CPU/CUDA RNG 恢复和排空传输 | 6 项通过 |

硬件仅使用显式 UUID 指定的 CMP 90HX；测试模块为小型 BF16 模块，offloader 使用真实
`ModelOffloader`。三族测试替换内层样图生成函数以驱动这些模块，没有加载三族完整权重。
没有做三族完整模型长期热训练、真实样图质量或吞吐 A/B，因此不报告速度提升比例，
也不把 Qwen 已有 A/B 耗时收益外推到其他模型族。

专项复现命令：

```bash
ANIMA_RUN_BLOCK_SWAP_CUDA_TESTS=1 \
CUDA_VISIBLE_DEVICES=<单张空闲 GPU 的 UUID> \
timeout 60 .venv/bin/python -m pytest -q tests/test_tiled_block_swap_lifecycle.py
```

CPU 联合测试入口：

```bash
CUDA_VISIBLE_DEVICES='' timeout 90 .venv/bin/python -m pytest -q \
  tests/test_preview_lifecycle.py tests/test_family_preview_lifecycle.py \
  tests/test_distributed_sampling.py tests/test_krea2_training_sample_preview.py \
  tests/test_z_image_training_sample_preview.py tests/test_sample_token_budget.py \
  tests/test_tiled_block_swap_lifecycle.py tests/test_inference_hydra_sigma.py \
  tests/test_z_image_block_swap.py tests/test_krea2_compile_blocks.py \
  tests/test_qwen_image_2_1_preview.py tests/test_qwen_preview_kv_opt_in.py
```

CPU 联合数字来自 fixture 补强前；补强新增 1 项 CPU fixture 检查，并已通过上述专项重跑。
验收结束时本轮 CUDA 测试进程已退出；20203 状态 idle、训练队列 0。
本轮没有重启 WebUI、启动长训练或终止其他进程。

相关 Qwen 证据：[方案 B](qwen_sampling_prepare_ab_90hx_20261002.md)、
[K/V A/B/C](qwen_sampling_kv_abc_90hx_20261002.md)。
