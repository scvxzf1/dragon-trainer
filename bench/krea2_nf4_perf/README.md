# Krea-2 NF4 性能口径研究

状态：实验。含优化器标量设备审计、nsys 只读分析、scaled-cast 算子验证，以及复用生产入口的计时包装。
结果见 [测量口径审计](../../docs/findings/krea2_nf4_profile_audit_20261112.md)和
[真实单步与融合候选](../../docs/findings/krea2_nf4_scaled_cast_20261112.md)。

## 单热点优化验证

- `analyze_nsys.py <report.sqlite> --output <new.json>`：只读导出的 nsys SQLite，必须有非空 CUPTI kernel 表；输出 union 忙时、NVTX 窗口和热点。kernel 忙不等于 SM occupancy。
- `scaled_cast_probe.py --output <new.json>`：先过 GPU 闸门；全 16-bit 位模式与13档缩放对照，再测4608×6144算子；不代表整步提速。
- `production_probe.py --arm reference|candidate --result <new.json> [--capture-final <new.safetensors>] --config_file <isolated.toml>`：复用 `train.py`，只包装 `_run_microbatch` 计时。reference 在进程内还原 Torch 缩放；数值张量回传只发生在最终计时完成后。配置必须关闭 sample/validation，使用隔离输出和缓存，不复用用户队列。
- GPU 采样上限：本轮用户已明确批准单进程最多600秒；每次仍需重新过 gpu-gate。闸门不通过即停止，不自动挤占或循环等待。
- 热测终局（2026-11-14）：四组六步 ABBA 归因（`compare_abba.py` → `abba-attribution.json`）证明轨迹级梯度发散是基线固有噪声（A/A 27.8%）；`shadow_verify.py` 在真实训练输入上 904/904 次调用逐位等价。候选数学等价成立，收益约 -2.29% 延迟 / -430MiB 显存（短窗口口径）。候选补丁当前**应用在研究分支**，是否合入 dev 由用户决定。
- `scaled_cast_candidate.patch` 保留候选源码和测试；production_probe 在补丁未应用时拒绝 candidate 臂，防止误测成相同实现。
- 影子校验用法：候选补丁应用后，用 `shadow_verify.py --result <new.json> --config_file <隔离配置>` 跑 2 步真实训练；verdict=PASS 要求 0 失配。


## optimizer_scalar_probe.py

- 不加载模型或数据，不操作缓存、训练历史和队列。
- 392 个两元素 FP32 参数，分别检查默认 / foreach / fused（CUDA）AdamW。
- warmup 后记录一次 CPU profiler 计数，再独立运行一步，以 dispatch 输入设备交叉核对。
- CPU 模式不创建 CUDA 张量；CUDA 必须显式指定 `--device cuda`，不可用时不回退 CPU。
- `--output` 必填，只允许新文件；重复路径直接报错，不覆盖既有结果。
- 断言 profiler 与独立 dispatch 总计数一致、参数更新后有限；不比较训练梯度精度。

在有项目环境的工作区：

```bash
CUDA_VISIBLE_DEVICES='' timeout 60 .venv/bin/python -B \
  bench/krea2_nf4_perf/optimizer_scalar_probe.py --device cpu \
  --output /tmp/krea2-scalar-cpu-<unique>.json
```

CUDA 运行前必须执行本机 gpu-gate，并使用它返回的设备（此处设备值为占位符）：

```bash
/mnt/SD_Dataset/训练器相关/dsh-trainer-preset/gpu-gate.sh --need 512 --count 1 --json
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gate-selected-device> \
  timeout 60 .venv/bin/python -B bench/krea2_nf4_perf/optimizer_scalar_probe.py \
  --device cuda --output /tmp/krea2-scalar-cuda-<unique>.json
```

本轮隔离工作树没有另装环境，实际使用相邻主仓已存在的
`/mnt/SD_Dataset/训练器相关/krea2-webui/.venv/bin/python`；没有复制或改动环境。
本轮结果写入 findings 的两个新 JSON，复现时须换新输出文件名。

**解释边界**：`_local_scalar_dense` 的输入在 CPU 时不构成 GPU→CPU 同步。
事件计数不能证明 GPU 利用率、训练瓶颈或性能收益；此脚本没有 CUPTI kernel 时间线。
