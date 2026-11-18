# Qwen Image 2.1 LoKr / LoHa 90HX 热测

状态：研究记录 / 阶段快照

2026-09-29，CMP 90HX 10 GiB，Qwen Image 2.1 T2I，同一单图数据、缓存和种子 1234。配置为 BF16 base/训练、full checkpoint、swap24、`attn_mode="torch"`、Inductor default compile、关闭阶段 3 投影融合。每组独立进程；热测 20 步，统计第 6--20 步的相邻完成时间。配置快照、命令、GPU 遥测、训练日志、进度和 checkpoint 均保存在下列结果目录。

| Adapter | smoke | 20 步 | 稳态中位数 | 首/末 5 步均值 | PyTorch 分配峰值 | checkpoint |
| --- | --- | --- | ---: | ---: | ---: | --- |
| plain LoRA | 历史同配置 smoke 通过 | PASS | 7.938 s | 7.941 / 7.936 s | 5.44 GiB | 384 tensors, 33.6 MB |
| LoHa | PASS | PASS | 9.090 s | 9.095 / 9.091 s | 5.97 GiB | 640 tensors, 67.2 MB |
| LoKr | PASS | PASS | 11.373 s | 11.372 / 11.372 s | 5.69 GiB | 384 tensors, 67.2 MB |

相对本轮 plain LoRA，LoHa 慢约 14.5%，LoKr 慢约 43.3%。各运行记录的训练源码哈希、模型和输入指纹一致；LoKr 与后续两组的验收脚本哈希仅因 CLI 帮助文案修订而不同。这些是单轮、顺序执行的短窗口结果，不是多轮交替性能结论；三个变体的参数化与训练开销也不同，不应仅按步时选择方法。

## 失败与修复

首次 LoKr smoke 在编译后首步反向遇到冻结 base Linear 权重的 AOT/Inductor stride 断言；此前日志还显示 Dynamo `recompile_limit=8` 被触及。Qwen 的编译路径当时绕过了 Anima 已有的 LoKr checkpoint 编译预算逻辑。现在 Qwen 在编译 block 前复用该预算设置：128 个 LoKr 模块、1 个签名对应 `recompile_limit=296`、`accumulated_recompile_limit=1184`，并关闭 Dynamo LRU 图排序。修复后两步 smoke 和 20 步热测通过。这个 A/B 支持该接线有效，但未独立证明 stride 错配的唯一根因。

失败原始记录：`output/runs/qwen21-variant-hot-acceptance-20260929-retest/lokr-smoke/`。通过的结果目录：

- `output/runs/qwen21-variant-lokr-compile-budget-smoke-20260929/`
- `output/runs/qwen21-variant-lokr-compile-budget-hot-20260929/`
- `output/runs/qwen21-variant-loha-hot-20260929/`
- `output/runs/qwen21-variant-lora-current-baseline-20260929/`

验收脚本对每步有限 loss、完整 step/run_end、GPU 连续遥测和 checkpoint 的 variant metadata、键型、非零有限张量做了检查。当前证据限于 T2I、单图、torch attention、BF16、full checkpoint、swap24、compile；没有验证 60 步以上稳定性、LoKr/LoHa 实卡重载续训、样图质量、Edit、Flash、其他 swap 数或其他硬件。DoRA、VeRA 及其他方法仍由兼容层拒绝。

2026-10-05 更新：Next WebUI 已在 T2I 中提供 BF16 LoKr/LoHa 实验选项，Edit 仍只提供 plain LoRA。
该入口不扩大以上真机证据范围，也不代表 NF4 + LoKr/LoHa 或独立 LoKr/LoHa 推理已验收。
