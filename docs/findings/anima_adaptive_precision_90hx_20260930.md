# Anima 自适应精度首阶段 CMP 90HX 探针（2026-09-30）

状态：实验 / 阶段探针记录（非生产精度认证）

## 结论

本轮只完成实验 CLI 与首模型 disposable probe，不是 production precision 认证，
也没有把 auto precision 或 OOM 策略接入 production。CMP 90HX 上全部 FP16-islands
与原生 BF16 候选均完成 3 次 finite optimizer update，但数值比较因输入/adapter
梯度偏差而拒绝；唯一通过的是显式全 FP32 线性层控制组。它只是诊断基线，不是
“FP16 成功”或低精度候选。当前默认精度、生产输入舍入、自动选择与 OOM 扩展均不变。

## 实验边界

- GPU：CMP 90HX 10 GiB，SM 8.6，UUID `GPU-806debd9-d254-da57-d515-ccae15b4ac6a`。
- 基座来源：包含 685 个 BF16 tensor 的 checkpoint，而非原始 FP32 权重；基座加载 dtype 为 BF16。
- 设置：AdamW，lr `1e-4`，LoRA rank/alpha `4/4`，full checkpoint，swap 20，
  torch attention，关闭 compile 与 TF32，microbatch 1，3 updates；sigma 为
  `0.2/0.5/0.8`。缓存输入 shape 为 latent/noise `[1,16,1,150,112]`、prompt
  `[1,512,1024]`，prompt 保留 full padding，latent mask 为全零。
- 各参考与候选固定 FP32 adapter state、输入和 RNG，并经 capture/replay 对齐。
  BF16 原生候选仅在模型边界 cast sigma 为 BF16；内部 replay 仍为 FP32，这不同于
  production input rounding。
- 数据来自现有单图缓存 fixture 和单 bucket；不能代表数据集、分辨率或 sigma 分布。
  权重文件 SHA-256 为 `bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e`；
  probe 输入 SHA-256 为 `0c2d6f12569596357efd5a1b0b652150f176ef37e2f475bda0bcd533ccc17d78`。

## 数值结果

下表误差是最大 prediction / input-gradient / adapter-gradient relative-L2；百分数列
已将原始比值乘以 100。每个候选均为 finite 3-step worker，但门槛不通过即 fail-closed。

| 组别 | Prediction rel-L2 | Input grad rel-L2 | Adapter grad rel-L2 | 峰值 GPU | 峰值 RSS | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 seed | 0.0775% | 90.1546% | 31.6477% | 3998 MiB | 7.78 GB | 拒绝 |
| FP16 conditioning | 0.0426% | 120.558% | 32.0214% | 4120 MiB | 8.62 GB | 拒绝 |
| FP16 conditioning + attention FP32 | 0.0327% | 41.767% | 14.135% | 4518 MiB | 10.75 GB | 拒绝 |
| FP32 control（全部 Linear FP32） | 0% | 0.000357% | 0.000178% | 5236 MiB | 19.69 GB | 通过控制对照 |
| 原生 BF16 | 1.28368% | 281.103% | 68.328% | 3154 MiB | 10.21 GB | 拒绝 |

CPU RSS 可超过 GPU 显存容量；峰值本身不是可部署性证明。最初 `first` 轮 reference
因 forward mask 为 `None` 在 0 update 失败；修正输入后，`controls` 轮全 FP32 reference
因 `nvidia-smi` 2 秒查询超时而停止，监控 fail-closed；复测 reference 与 FP32
control 完成。BF16 首次遇到 sigma dtype mismatch，修正后复测完成但仍因梯度误差
拒绝。每个 worker 启动前检查 GPU idle 与 system/cgroup reserve，采集 1 秒 GPU、RSS、
host swap telemetry；单次查询限时 5 秒，临时查询失败时记录 `gpu:null` 与错误，连续 3 次错误或 10 秒遥测空窗
会停止。结束时残留显存为 17 MiB。没有自动改 source/candidate 的 OOM、swap 或精度。

跨两次独立运行的参考记录中第 1 步 digest 一致；第 2/3 步输入和 RNG 对齐，但 state/snapshot 不同，
故 capture 仍在比较容差内，不能声称跨进程逐位确定性。全部结果保存在
`output/runs/anima-adaptive-precision-90hx-20260930-{first,retest,controls,controls-retest,bf16-control,bf16-retest}/`；
其中 `environment.json`、`reference/`、`attempt-*/comparison.json`、`worker.log`、
`telemetry.jsonl` 与 `summary.json` 保留逐轮证据；最终控制对照选择记录位于
`controls-retest/selected-plan.json`。

## 运行方式

命令要求已存在 BF16 权重和准备好的输入包；下例的绝对路径均为调用者占位，不是
仓库默认值。命令只接受 `swap 0..26`、`steps 1..5`，每个 worker 有超时和尝试上限。

```bash
.venv/bin/python -m bench.adaptive_runtime.search_anima_precision \
  --weights /path/to/anima-bf16.safetensors \
  --inputs /path/to/prepared-anima-inputs.safetensors \
  --output-dir output/runs/anima-precision-new \
  --gpu-uuid GPU-... --swap 20 --steps 3 \
  --worker-timeout 150 --max-attempts 2 \
  --profile seed --profile conditioning
```

指定候选全部被拒绝时 CLI 以 exit 1 结束；若只运行 `--profile fp32-control`，成功控制
对照可 exit 0，但其 scope 仅是 FP32 诊断对照，绝不表示 FP16/BF16 获准。

## 验证状态

最终独立 Sol 复核的 `tests/test_adaptive*.py` 为 601 passed、2 skipped、3 xfailed，
退出码 0；Ruff 与本任务空白检查通过。本轮主定向回归为 114 passed。
剩余 3 项 xfail 是既有共享参数边界缺口，不计作通过。
CPU 数学回归移除 tiny-cosine 两个 strict xfail 后的修复状态见
[精度数值说明第 8.2 节](../structure/adaptive-precision-numerics.md)。

## 后续与限制

后续分支与 scale 对照已完成，见[梯度误差定位报告](anima_adaptive_precision_diagnostics_90hx_20260930.md)。
六个新增诊断候选仍未通过，不改变本报告的首阶段结论。

先按 392 项梯度 inventory 定位误差，优先诊断 adapter update、MLP 与 attention 分支，
再增加 bucket/timestep 覆盖并重复 reference/candidate。当前没有开展真实 production train
桥接、长训、图像质量评估或 Turing 验证；只有 Anima 首阶段，不代表通用多模型精度
轮子完成。不得据此放宽梯度门槛、改变生产精度默认值或推导稳态速度收益。

热测后增加了源 BF16 tensor 头部校验及每次 worker 启动前的 GPU 空闲复查，已经独立
CPU 验证，未再做 GPU 复跑；不修改前述热测结果。空闲复查不能保证设备独占，仍有
检查与启动之间的竞态。最新 worker 签名含 probe 源码 SHA-256，旧参考不会静默用于
修改后的 worker。后续复测必须重新生成同版本参考。
