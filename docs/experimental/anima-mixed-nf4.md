# Anima 混合 NF4 checkpoint

状态：实验。仅接受标准 28 block、hidden width 2048、全 BF16 Anima safetensors。

固定策略保留 `blocks.14..20.mlp.layer1` 为 BF16，其余 21 个 layer1 和全部
28 个 layer2 使用 NF4（blocksize 64、double quant、uint8 packed），共 49 个权重。
所有其他 tensor 的 key、dtype、shape、字节值及源 metadata 原样保留。
`diffusion.mixed_nf4.manifest` 是当前 canonical manifest，保存 family、版本、
prefix、精确策略、完整源张量清单、源 SHA256、qstate 元数据和 bnb 版本。
旧版 `anima.mixed_nf4.manifest` 只读兼容，不再由新转换器写出。

```bash
.venv/bin/python scripts/anima/quantize_mixed_nf4.py --source /path/base.safetensors --output /path/mixed.safetensors --device cpu --threads 4
.venv/bin/python scripts/anima/quantize_mixed_nf4.py --source /path/base.safetensors --output /path/mixed.safetensors --verify-only
```

转换逐矩阵执行，日志输出进度；CPU 转换需要足够内存容纳保留的模型状态及保存缓冲。
源文件不会覆盖；目标已存在时拒绝写入，并用同目录临时文件和原子无覆盖发布。
发布前校验源 SHA256、量化状态和全部非目标 tensor 字节一致。

Anima loader 优先识别 canonical manifest，必要时回退旧 manifest，直接恢复
`Linear4bit`，不重复量化；canonical manifest 的 qstate 内容也会被校验，
不是只检查 safetensors header。
只允许 BF16 compute，拒绝加载时 LoRA merge；普通 BF16 路径保持原行为。
CPU 小夹具验证 bnb 状态 roundtrip、BF16 前后向及 plain LoRA 挂载。
2026-10-02 已完成 CMP 90HX 上 plain LoRA、BF16 compute、1024×1024、swap12、
full checkpoint、无 compile 的 BF16/混合基座各500步训练；这不代表所有 adapter
variants、其他硬件或完整 Anima compile 已验收。

## 异型权重换层与短程显存诊断

异型 BF16/NF4 按逻辑矩阵尺寸校验，训练时共享提前分配的最大容量缓冲，保留
bitsandbytes 反向保存的 packed storage alias；采样复用训练缓冲池。
动态调整 swap 数量在新槽及计划校验失败时恢复旧拓扑，纯 BF16 路径保留计划缓存。

真实8步对照包含第4/8步1024采样与恢复训练。首次采样前反向峰值约
4.0621→4.0231 GiB，普通训练步结束约2.3117→2.2727 GiB；含采样进程峰值反而
4.9341→5.0513 GiB。修复改善训练阶段，但不能宣称全程峰值下降。
证据和数值限制见[换层显存报告](../findings/anima_mixed_nf4_swap_memory_90hx_20261002.md)。

2026-10-03 新增专属 Linear autocast 修复：CUDA BF16 autocast 下遵守普通 Linear
的 BF16 输出语义，避免 bnb 将 FP32 输入的结果转回 FP32，导致 LoRA、残差和
checkpoint 激活膨胀。没有新增配置项，也不改量化策略或权重格式；无 autocast
时保留原 bnb 行为。旧路径的舍入结果会改变，不能宣称训练权重逐位等价。
验收与同参数对照见 [autocast 显存报告](../findings/anima_mixed_nf4_autocast_90hx_20261003.md)。

该Linear及qstate校验已抽到 `library/quantization/` 供其他模型族复用；Anima旧格式、
热点层策略和import入口保持兼容。三族扩展的独立策略及验收范围见
[多模型族混合NF4](multi-family-mixed-nf4.md)。

诊断命令（训练时长仍由独立配置的 `max_train_steps` 决定）：

```bash
python scripts/anima/probe_mixed_nf4_memory.py --config_file /path/short-run.toml --phase-jsonl /path/phases.jsonl --trace-steps 8
```

探针同步 forward/backward/optimizer 边界并记录各阶段独立峰值，公共 CUDA max API
维持累计口径；不用于比较训练吞吐。额外的 `--offloader-source` 只在隔离诊断进程加载
已保存旧版代码用于受控回放，不修改项目文件，也不作为训练产品入口。
