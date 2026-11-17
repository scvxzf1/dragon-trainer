# ConvRot INT8：CMP 90HX 重新验证结论

状态：验证记录 / 阶段快照

本轮已完成 S0–S4 的工具修正、基线、优化筛选和生产入口功能验证；S5 按原计划的条件门控跳过，状态为 `SKIPPED_NO_ACCEPTED_CANDIDATE`。**当前 Anima 1024、rank32 场景下，BF16 无换层仍最快；W8A16 降低训练窗口显存，但采样阶段的生命周期问题使全流程峰值反而更高。** 这不是量化普遍无用的证明。

原协议：[六阶段计划（已归档）](../../_archive/docs/proposal/convrot_90hx_development_validation_plan_20261002.md)。后续：[生命周期修复与确定性验证计划](../proposal/convrot_90hx_followup_development_plan_20261002.md)。实验根目录原始证据：`output/convrot-90hx-20261002-210307/summary.md`，所有性能数字来自新进程热测，不沿用 3080 历史数据。

## 范围与固定条件

- CMP 90HX，SM86，UUID `GPU-806debd9-d254-da57-d515-ccae15b4ac6a`；Torch 可见显存 9.6458 GiB，统一预算扣除 512 MiB 后约 9.1458 GiB。没有调整功耗、风扇或时钟，也没有使用桌面 GTX1050。
- Torch 2.12.0 / CUDA13 / driver580.178.04 / Triton3.7 / bitsandbytes0.49.2 / Flash2.8.3。
- Anima `anima-base-v1.0-sha.safetensors`；现有 60 张 1024 图片与 cache。数值矩阵为两份 cache × 三个 seed × 三个 sigma，共 18 个独立单步更新样本。使用训练集诊断，不是 held-out 质量评估。
- BF16 compute、FP32 plain LoRA rank/alpha32、AdamW LR2e-5/WD0.01、Flash、full checkpoint、compile off、foreach BF16 transfer。ConvRot 为 scope all、dense Sylvester@256；FP32 matmul highest、TF32 off。
- 相同初始化，392 个 FP32 可训练参数张量、196 个 fused Linear adapter 目标；导出 560 个 split 权重张量来自 QKV/KV 拆分，不能当作新增训练目标。初始文件缺少的 196 项只有 alpha buffer，运行时 alpha 均为32。
- 输入 latent 为 `[1,16,1,128,128]`，context 为 `[1,512,1024]`；保持 max-padding attention sinks。314 个运行时 Linear hook 输入均为 BF16。

## S1：固定 swap12 重建基线

每模式每轮 10 步预热 + 30 步测量，三个新进程轮次，轮换顺序。计时窗口前后等待 offloader futures 和 CUDA；窗口内不做逐参数 CPU 标量回传。

| 模式 | 三轮秒/步 | 中位数秒/步 | 训练窗口峰值 allocated GiB |
|---|---|---:|---:|
| B0 BF16 | 2.7218 / 2.7327 / 2.7016 | 2.7218 | 4.0072 |
| B1 W8A16 | 2.7379 / 2.7533 / 2.7542 | 2.7533 | 3.2249 |
| B2 W8A8、FP32 dX | 4.3150 / 4.3378 / 4.3376 | 4.3376 | 3.4907 |

W8A8 比 W8A16 慢约 58%。这里的 PASS 只表示执行完整、有限值和有效计时，不能表示训练质量等价。性能汇总：`output/convrot-90hx-20261002-210307/s1-performance-summary.json`。

## S2：保数值 Triton 工程优化

E1 只替换 rowwise quant，E2 只替换 INT8 GEMM/缩放，E3 合并二者；反向仍为 FP32。算子测试发现并修复了 scale 舍入差异：PyTorch 标量 `/127` 的取码参考需要匹配 FP32 reciprocal multiplication。没有接入生产默认 dispatch。

| 候选 | 热测秒/步 | 训练窗口峰值 allocated GiB |
|---|---:|---:|
| E1 Triton quant | 4.1307 | 3.2738 |
| E2 Triton GEMM | 4.2022 | 3.3662 |
| E3 两者合并，三轮中位数 | 3.9670 | 3.2738 |

E3 三轮为 3.96695 / 3.96386 / 3.97045 秒/步，比 B2 快约 8.5%，但仍比固定 swap12 的 W8A16 慢约 44%，训练峰值也没有优于 W8A16。属于局部工程收益，没有成为推荐训练配置。

七个实际/合成形状微测的 code、scale、output 精确对齐；量化耗时改善约 2.6–13.9 倍，GEMM 约 1.07–1.86 倍。最大 `(4096,2048,8192)` GEMM 的额外分配由 320 MiB 降到 64 MiB。此局部 256 MiB 差值不能直接当作整步峰值下降。微测与分配证据：`output/convrot-90hx-20261002-210307/s2-microbench.json`、`output/convrot-90hx-20261002-210307/s2-microbench-memory.json`。

## S1/S3：数值门槛与参考噪声

预登记门槛保持不变：output relL2≤0.03、loss relative≤0.05、gradient norm/vector relative≤0.05、gradient cosine≥0.995、update relative≤0.05；逐层 relative≤0.10、cosine≥0.99，零范数容差1e-12和绝对L2容差1e-8。完整18样本、参数名称、逐层梯度和更新缺一不能通过。

与 B0 比较，B1/B2 的 output 门槛全部通过，但全18样本梯度门槛失败：B1 梯度 relL2 为 0.2291–0.9100，B2 为 0.1885–1.7231。有限值和前向接近不能推出 LoRA 更新接近。

必须同时看参考重复：默认 Flash backward 非确定性。BF16 自身重跑的 output/loss 完全相同，梯度误差约 0.8–1.5%，但单步 AdamW 更新误差约 6–8%，全18样本触发5%更新门槛。B2 自身重跑最大梯度误差0.02085、最大更新误差0.10417，也全部触发更新门槛。不能把这类失败全部归罪于优化内核。

E1/E2/E3 对 B2 的 output 精确相同，梯度门槛全部通过，最大梯度误差分别约0.02675/0.02089/0.02023；更新仍失败，与参考自身噪声量级相近。因此局部算子契约通过，完整模型更新等价仍未获正式认证。

| 反向候选 | 对 B2 的梯度 relL2 | 梯度判定 | 完整更新判定 |
|---|---|---|---|
| N1 BF16 dX | 0.01025–0.02310，min cosine0.999791 | 整体及逐层通过 | 更新/逐层更新18/18失败；受参考噪声影响，INCONCLUSIVE，不晋级 |
| N2 INT8 dX | 0.04443–0.10678，min cosine0.994913 | 整体14/18、逐层18/18失败 | 18/18失败；附加梯度偏离超出自身重复控制，拒绝 |

N1/N2 对 B0 的梯度也均未通过。N2 的实现是 FP32 `dy*scale` fold 后量化、逆布局 INT8 GEMM、BF16 RHT；不能说它等同于 musubi 的 BF16 fold。

两独立进程的 **单样本 deterministic Flash backward 控制**得到 output/gradient/update 差值全部为0。它证明确定性诊断路径可行，尚不替代完整18样本验收；默认吞吐/生产 Flash 未改。单样本控制：`output/convrot-90hx-20261002-210307/s0-flash-deterministic-control-assessment.json`。

10组数值运行的360个完整 gradient/update 文件经 header 审计，均包含392个FP32参数，shape与初始manifest匹配。审计：`output/convrot-90hx-20261002-210307/numerical-tensor-header-audit.json`。3 seeds ×20连续轨迹按门控未执行，不声称长期更新一致。

## S4：同预算性能、采样、保存和续训

BF16/W8A16 都能在统一9.1458 GiB预算下 swap0，因此最低换层均为0。三个新进程轮次各10warm+30measure，交替运行。

| 模式 | 三轮秒/步 | 中位数秒/步 | 训练窗口峰值 allocated GiB |
|---|---|---:|---:|
| BF16 swap0 | 1.57578 / 1.57153 / 1.57226 | 1.57226 | 5.57264 |
| W8A16 swap0 | 1.73796 / 1.73880 / 1.73898 | 1.73880 | 3.98075 |

W8A16 在这个同预算比较中慢10.6%，虽然训练窗口显存少约1.59 GiB。6/6状态：`output/convrot-90hx-20261002-210307/s4-minswap-performance-status.json`。不要将其与另一个栈的8步生产入口峰值混算。

生产入口第4/8步采样1024图，保存adapter和optimizer state。预算用例覆盖到第8步采样后，结果如下；峰值是全训练+采样窗口，不是 `.to(cpu)` 后的瞬时驻留。

| 模式 | 全流程 peak allocated GiB | peak reserved GiB |
|---|---:|---:|
| BF16 swap0 | 5.56554 | 5.82227 |
| W8A16 swap0 | 6.25306 | 7.25781 |
| BF16 swap26 | 4.69976 | 5.11914 |
| W8A16 swap26 | 4.85806 | 5.31250 |

固定 swap12 的采样全流程峰值为 BF16 4.69976 GiB，W8A16/W8A8均5.67907 GiB；量化组的高峰在sample_after。预算用例两张PNG均为1024，state.current_step和392个optimizer step均为8，最终权重有限且保存hash。生产审计：`output/convrot-90hx-20261002-210307/s4-production-audit.json`。初始8步配置最后一次采样缺少phase标签，后来预算配置补齐，不用缺失标签推定峰值位置。

B0/B1生产续训均由5恢复到10，392个optimizer状态的step均到10。**生产dataloader精确游标恢复尚未建立，不能宣称生产连续10步等于5+5。** 另一个固定输入游标、恢复optimizer/RNG、开启deterministic Flash的受控replay，B0/B1连续10与5+恢复5最终参数逐元素相等；这是受控恢复契约证据。

采样拼图（`output/convrot-90hx-20261002-210307/s4-smoke-contact-sheet.png`）仅为8步功能预览，不是500步质量验收。候选compile/生产精度policy接入按筛选门控未执行。

## 两个已证实的内存生命周期问题

1. **换层所有权与普通CPU迁移不一致。** `BlockSwapManagedTensor._apply` 在placement active时跳过普通Module.to，由offloader独占placement。训练预览decode调用 `dit.to("cpu")` 后加载VAE，却没有显式迁走offloader管理的量化载荷。实际swap12探针：CPU park前后managed GPU payload均为1,008,336,896 bytes（约0.9391 GiB），删除训练图后仍驻留。BF16相应allocated可下降到约0.33 GiB，量化仍约1.27 GiB。
2. **无换层时训练图仍引用冻结量化常量。** ConvRot autograd Function把`w_q/w_scale/hadamard`放在ctx属性中。swap0时载荷owner已经CPU park、managed GPU payload归零，但output graph还在时allocated约1.90 GiB；删除已完成训练图后降到0.3321 GiB，额外释放约1.57 GiB。表明仅迁移model owner不足以释放graph引用。

源码入口：`library/runtime/block_swap_payload.py:27`、`library/anima/training.py:1349`、`library/runtime/convrot/fused.py:177,246`。证据文件为实验根目录的`s4-lifetime-{bf16,w8a16,w8a8}.json`及量化组`-swap0.json`。

探针的`live_backward_ctx_cuda_storage_bytes`实际是对反向中观察到的tensor的弱引用，不是ctx自身弱引用。swap12下owner也会保留这些tensor，不能以该字段断言删图后ctx仍活着；swap0的owner归零和删除graph后的allocated变化才构成该因果证据。

2026-10-02本轮只诊断，未修复生产生命周期。2026-10-03后续已实现修复并完成真实短测，见[生命周期修复报告](convrot_lifecycle_fix_90hx_20261003.md)；下述建议保留为当时结论。后续需要显式完成step后的graph释放边界，以及offloader-owned park/restore API；不能直接取消placement guard或无条件清空ctx属性，避免破坏retain_graph/create_graph/checkpoint/compile语义。

## 工具验证与限制

CPU定向测试49通过。primitive suite64通过，其中15CPU+49真实CUDA；补充policy/scale19真实CUDA通过，合计68真实CUDA检查。最初scale舍入和CUDA int32 addmm参考测试失败日志保留，修正后重验通过。收尾再次执行49项CPU测试全通过，Ruff通过，Pyright为0 errors/0 warnings；文档链接与最终JSON解析通过。最终检查：`output/convrot-90hx-20261002-210307/final-checks.json`。

初次独立审阅发现的三项假PASS问题（缺失同名梯度、小更新/逐层更新漏验、不完整18矩阵）已修复并补回归。由于代理会话额度耗尽，修复后没有新的独立最终复审，不能标为独立验收通过。

CMP的CUPTI返回`CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED(42)`，Chrome trace只有CPU事件；另外用CUDA Event进行局部诊断。同步诊断/p50/p95不用于吞吐排名；1秒NVML也不能代表瞬时精确峰值。

## S5与当前建议

没有候选同时通过数值和实用性门槛。S5配置虽已准备（500步、250步采样、1024、数据集caption衍生提示词），实际未运行；没有500步质量结论，也没有隐式放宽门槛。当前90HX的本场景使用BF16 swap0更合理。E3继续保留实验实现；N1待完整确定性复核；N2按现门槛拒绝。优先修复全流程内存管理后，再决定量化是否能够在更受限预算或更大模型上转化为实际收益。
