# Anima 原生 FP16 入口与保存/续训验证

日期：2026-10-02。状态：代码实现、生产短训和独立收尾验收通过。
范围：Anima 普通 LoRA、CMP 90HX 10GB（SM86）。默认精度未改变。

## 交付

新增 `adaptive_precision="native_fp16"`：FP16 基础模型、原生 AMP、FP32
LoRA 和既有 residual/final 保护，正式使用 `adaptive_loss_scale` 配置初始值。
Anima 的 `auto` FP16 候选进入此模式；安培继续优先 BF16，其他族原路由保留。
显式/冻结的 `fp16_fp32` 仍使用实验岛屿合同。CLI、配置 schema 和 Next
选项/帮助同步，使用方法见[配置说明](../experimental/anima-native-fp16.md)。

普通训练输出新增主进程原子写入的精度报告；同目录多次运行保留各自报告。
报告读取实际参数 dtype、guards、scaler、继承步数和本次尝试/有效更新/跳步，
不复制参数或完整优化器状态，不逐步同步 GPU 获取资源。
正常完成必须经过最终保存后的完成事件；提前返回不会冒充完成。

## 生产热测试

Luna 使用 `train.py --config_file` 直接执行生产路径，没有覆盖 accelerator、
flow/noise 或 optimizer 数学。复用既有 16 图缓存、4032/4200 native bucket、
rank/alpha4、AdamW lr1e-4、torch attention、full checkpoint、swap20，
本次 gradient accumulation 为 2，关闭采样、compile、自动 swap 和 OOM 重试。

| 观测 | 首段 | 续训 |
| --- | ---: | ---: |
| 进程退出码 | 0 | 0 |
| 继承 global step | 0 | 3 |
| 本次尝试 / 有效 / 跳步 | 3 / 3 / 0 | 2 / 2 / 0 |
| 最终 global step | 3 | 5 |
| 配置初始 scale | 1024 | 2048 |
| 实际恢复 / 最终 scale | 1024 / 1024 | 1024 / 1024 |
| 392 组真实 AdamW state step | 全部 3 | 全部 5 |

实际模型参数为 FP16，392 个可训练 LoRA 参数为 FP32；28 个 block residual
guard 和 final guard 均启用。保存权重有限，续训后 392 个浮点 LoRA 张量改变。
日志所见 loss 有限，证明短保存/恢复链路可用，不是另一轮训练质量评估。

逐秒外部监控有 93 个成功 GPU 采样、零查询失败，已观测整卡显存峰值
**3488 MiB**，总容量 **10240 MiB**；二者不是两阶段峰值。
训练进程树 RSS 已观测峰值 11,243,552,768 字节（约 10.47 GiB）。
结束目标 GPU 回到 17 MiB、0% 利用率。未改变功耗、风扇或另一张显卡。

结果根为 `output/runs/anima-native-fp16-production-resume-90hx-20261002-hot1/`。
两段命令都传 `native.toml`，阶段之间更新其内容；`stage1.toml` 和
`stage2.toml` 保留各段配置快照。续训读取第一段 `native-state`，该目录随后
被 step5 收尾保存更新；独立 step3/step5 检查点均保留。

- 机器汇总：`output/runs/anima-native-fp16-production-resume-90hx-20261002-hot1/summary.json`
- 原始监控：`output/runs/anima-native-fp16-production-resume-90hx-20261002-hot1/monitor.jsonl`
- 首段精度报告：`output/runs/anima-native-fp16-production-resume-90hx-20261002-hot1/precision_report.json`
- 续训精度报告：`output/runs/anima-native-fp16-production-resume-90hx-20261002-hot1/precision_report.89d91f00d109415ab140124e27bf1903.json`
- 最终独立 CPU 验收：`output/runs/anima-native-fp16-production-resume-90hx-20261002-hot1/final-independent-audit.json`

## 验收与边界

不同 Sol 代理独立检查入口和报告，发现的 checkpoint adapter 绕过、旧 worker
错误分类、提前返回误报完成均已修正。最终入口/合同/loop/replay/supervisor/Web
预检的 12 文件定向集成回归为 **326 passed、1 skipped、3 xfailed**，退出 0。
独立收尾代理复跑报告 24 个用例，Ruff 通过，并直接读取原始检查点和监控核验
真实 AdamW/scaler/参数改变、退出码、更新数和资源峰值，结论 pass。
Next TypeScript 和 trainingForm 20 个用例通过；没有构建替换已有前端发布资源。

全局文档完整性测试仍有 3 个既有失败：旧文档索引遗漏、不可达提案及状态标签
缺失；本次新增说明及报告均已纳入索引，不顺手改写无关历史文档。
默认 `off` 保持兼容。Turing 真机、长期质量、自动敏感层搜索和 native OOM
重试未交付；更新后 OOM 不自动恢复。旧严格 0/12 失败保留，
[200步 A/B](anima_fp16_bf16_ab200_90hx_20261002.md) 也不认证所有任务质量。
