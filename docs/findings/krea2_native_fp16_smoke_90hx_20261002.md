# Krea-2 native FP16 CMP 90HX 三步 smoke

状态：实验 / smoke 记录（非训练质量或长期稳定性认证）
日期：2026-10-02。范围：官方非量化 Krea-2-Raw、plain LoRA、单进程、torch attention、full checkpoint、compile 关闭。此记录是运行契约和 smoke 交付，不是训练质量或长期稳定性认证。

## 软件与测试证据

- 实现边界：`library/training/adaptive_runtime/native_precision.py` 的配置拒绝规则、`native_family.py` 的族级 guard 安装/执行检查，以及 `library/models/krea2_raw/native_precision.py` 的 Krea 专属 FP32 residual/final-linear 策略。
- 支持范围严格限于显式 `adaptive_precision=native_fp16`、官方非量化模型、FP16 base、FP32 plain-LoRA、torch attention、full checkpoint、compile=false、单进程。按契约拒绝 auto、startup OOM retry、NF4、pipeline parallel、compile、selective/offloaded checkpoint 和非 plain-LoRA 组合。
- CPU 定向测试为 218+31 通过；独立验收为 259 通过、1 跳过。最终 guard 修正后的回归为 93 通过，另一组独立定向验证为 9+13 通过。它们验证的软件契约不等于硬件稳定性或质量认证。

## CMP 90HX 三步配对 smoke

证据目录：`output/runs/krea2-native-fp16-90hx-20261002-prep/attempts/{bf16-v2,native-fp16-v2}/`。两组均继承相同初始 LoRA 状态，588 个初始 hash 全部匹配；各完成 3/3 attempted updates、0 skip，report 完成并成功保存模型与训练 state。FP16 scaler 初始和最终均为 1024，终态 growth_tracker=3；本轮没有执行 resume。运行时观察到 Krea guard 的 28 个 DiT blocks、text/refiner residual、final residual 和 final linear 保护均启用。

| 项目 | BF16 对照 | native FP16 |
|---|---:|---:|
| 实际 DiT base 参数 | BF16 | FP16 |
| LoRA 参数 | FP32 | FP32 |
| GPU used peak（外部 monitor） | 7244 MiB | 8542 MiB |
| RSS tree peak（外部 monitor） | 58,205,102,080 bytes（约 54.21 GiB） | 50,230,943,744 bytes（约 46.78 GiB） |
| 总耗时（含启动） | 190.757 s | 226.178 s |
| 有效更新 / skip | 3 / 0 | 3 / 0 |

RSS 来自 `supervisor.json` 的 `monitor.peak_rss_tree_bytes`，是进程树监控峰值；不要与 `precision_report.json` 中 `resources.process_rss_max_bytes`（单进程生命周期口径）混为一谈。GPU used peak 也不同于 worker log 的 PyTorch allocated/reserved 峰值。短运行耗时包含模型加载、初始化等启动成本，不能据此推断稳态步速。

### 失败路径和边界

早期 BF16 尝试因监控时限包含加载而在两次有效更新后中断，保留为超时记录；本轮没有真实 CUDA OOM 验收，当前 family 仍拒绝自动 OOM 重试。当前有效 FP16 v2 运行使用了 26 个 block swap。Native FP16 的 GPU used peak 高于 BF16，虽 RSS tree 较低，但三步结果不能说明完整作业资源趋势。

NF4 历史对照来自不同设备/设置：PG199 1024×1024、LoRA rank 16、30 步记录中，BF16 为 GPU 29.45 GB / RSS 26.10 GB / 3.29 s 平均步时，NF4 为 GPU 10.49 GB / RSS 18.17 GB / 3.41 s。该数据见 [NF4 消融](krea2_nf4_ablation_findings.md)，只说明该历史实验本身；不得与 90HX native FP16 smoke 计算跨硬件收益百分比。RTX 3080 边界见[速度研究最终结论](krea2_3080_speed_final.md)及[扩展对照](krea2_3080_speed_comparison_extended.md)。[90HX NF4 downstream correction](krea2_nf4_downstream_correction_90hx.md) 是 teacher/correction 校准实验，不是 native FP16 训练证据。

Anima 既有 200-step 质量证据仅适用于其自身 family、模型和实验，不可外推到 Krea。三步 smoke 没有质量结论，也没有长期数值稳定性结论。

## 后续验收

1. 独立 CPU state/checkpoint 审计已通过：两组各 392 个 FP32 trainable 全部有限且改变，392 个 AdamW 状态均为 step 3，moments 有限；导出 BF16 checkpoint 与 FP32 state 按保存 dtype 转换后一致。证据为 run 目录 `audit/checkpoint_audit_v2_cpu.json`。这只证明保存完整性，尚未证明恢复语义。

   审计发现旧遥测 completed_steps 将更新字典字段数记为 5；监控源码已改用 effective 计数，旧遥测原样保留。资源统计经独立重算一致。日志更新累计耗时为 BF16 16/32/47 秒、FP16 18/34/49 秒，只有整数秒精度，不能作为精确稳态速度。
2. 进行一次受控 20-step save/resume：记录恢复前后有效 step、skip、loss/梯度 finite 状态、scaler 与各 state hash；单独确认续训不是重新开始或只载入 LoRA 权重。
3. 性能定位采用同机配对重复并剔除启动段，报告 steady-state step time、GPU used/allocated/reserved 和 RSS tree；控制温度/时钟、swap、bucket 与数据输入。没有这些证据前不宣称 FP16 更快或更省显存。
4. 若要评估训练质量，另立 Krea 专属、预先登记的长程配对质量方案；本 smoke 不替代该方案，也不执行 200-step 外推。
