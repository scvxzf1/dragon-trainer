# Qwen Image 2.1 专用训练 block：阶段 3 开发与验收

日期：2026-09-27
状态（更新于 2026-09-28）：3A/3B/3C 已实现并集成，默认关闭。数学/梯度、真实 swap/compile、生产保存加载、完整模型确定性续训、compiled 重算计数及三轮 400-step 热测验收完成。性能采用门槛未达到：torch 约 2.2%、Flash 约 0.6–1.2%，保留实验 opt-in，不推荐默认开启。
范围：BF16 frozen base、attention-only plain LoRA、90HX、swap24、compile；不叠加 INT8 传输，不增加 MLP LoRA，不改变默认配置。

## 交付顺序

| 阶段 | 开发交付 | 独立验收门槛 | 当前状态 |
| --- | --- | --- | --- |
| 3A | 冻结 MLP 的 gate/up 初始化打包，一次投影；保留 out | forward、dX、checkpoint 对照；无重复权重；offloader 可识别 | 已集成，CPU/90HX 小模型独立验证通过 |
| 3B | 冻结 QKV 一次投影，三个独立 LoRA 增量 | 非零 LoRA forward/dX/dA/dB、旧保存键与 round-trip、swap 后权重更新 | 已集成，非零梯度、RoPE/mask、磁盘 round-trip 验证通过 |
| 3C | 预算选块，保存大投影输出，分段 checkpoint/compile | 实测重算 GEMM 数下降、梯度一致、swap 生命周期正确、显存预算生效 | 已集成，真实 swap2+Inductor 数值通过；完整 compiled trace 确认 saved4 的 QKV/gate-up 重算各 32→28，dX 不变 |
| 集成 | 明确 opt-in 入口、启动顺序与拒绝边界 | 原路径回归；三项分别与组合均可执行，保存重载/续训 | 240 项 CPU 独立回归及新快照 86 项通过；生产 writer/reader fresh split/packed 输出/梯度独立验收通过；确定性完整 6 步最终模型/optimizer/scheduler/loss 精确一致 |
| 热测 | 固定配置、源码指纹、日志、温频与显存遥测 | 交替持续测试，无 OOM/NaN/梯度异常；约 5% 整步收益作为采用目标 | 12 组/400 步完成，含四组 60 步；torch 缩短 2.22–2.24%，Flash 缩短 0.56–1.22%，采用目标未达，保持默认关闭 |

不将开发自测作为独立验收，不将单算子提速作为整步收益，不以“能训练”替代梯度验证。每阶段保留变更、验证命令和结果；无收益时记录否定结果，不宣称默认推荐。

## 数学契约

采用 PyTorch Linear 的行向量约定。冻结 MLP：

```text
G = X Wg^T, U = X Wu^T, H = silu(G) * U, Y = H Wo^T
Wgu = concat_rows(Wg, Wu), [G, U] = X Wgu^T
dH = dY Wo
dG = dH * U * silu'(G), dU = dH * silu(G)
dX = concat_cols(dG, dU) Wgu
```

冻结 QKV 与独立 LoRA（i 属于 q/k/v）：

```text
Wi 不训练；Ai、Bi 分别保留原参数、rank、scale 和保存键
Pi = X Wi^T + si (X Ai^T) Bi^T
dX = concat_cols(dPq, dPk, dPv) Wqkv + sum_i si (dPi Bi) Ai
dBi = si dPi^T (X Ai^T)
dAi = si (dPi Bi)^T X
```

以上梯度式描述无 dropout/mask 的核心；存在 dropout、timestep mask 或 channel scale 时必须复用原 LoRA 的实际分支语义，不能用简化公式替代。BF16 GEMM 形状变化可改变舍入，数学等价不要求 BF16 位级一致；用 FP64/FP32 建立严格参考，再记录真实 BF16 误差。

## 存储、加载与执行契约

- 模型严格加载完成后才转换，不改原底模文件格式。MLP 默认拒绝训练中或已挂 adapter 的源模块，调用方显式确认 frozen-base 语义。
- QKV 在 LoRA apply/load 后转换，保留 adapter 参数对象与名称；转换完成后再建立 CPU masters、首次 prepare 和 compile。
- packed 底模权重只注册一份，不能保留旧 bound forward 导致 split 权重长期存活。不能缓存跨 swap 的旧 Tensor/view，反向必须读取该 block 当前有效权重。
- 不修改 Diffusers 安装目录或全局 helper。专用 processor 共享投影结果后，保持 norm、RoPE、文本因果/左填充、Edit segment 语义。
- 变换若不支持某个 adapter、processor、恢复模式或调用阶段，必须在执行前明确拒绝；不能绕过 LoRA 或静默丢失配置。
- 保存验收覆盖网络参数键、tensor 值，以及新旧运行态加载同一 adapter 的输出/梯度；训练保存成功不等于 resume 完成。

## 3C 激活与重算预算

B=1、4107 tokens、hidden=4096、BF16 时，hidden 约 32.09 MiB，QKV 约 96.26 MiB，gate/up 约 192.52 MiB。每块保留两项投影结果的原始 payload 约 288.78 MiB；这是部分激活的大小，不能作为整块增量显存上界，还须计入额外输入、LoRA 激活、autograd、workspace 和 allocator。

候选执行：选中 block 的 base QKV 和 gate/up 放在内层 checkpoint 外；attention 后段与 MLP 后段分段 checkpoint。未选中块维持 full checkpoint。选中块外层不得继续整块 checkpoint，否则不能保证免除投影重算。保留整块 swap backward hooks；内层重算及 base dX 完成前不得释放该块权重。

每个选中块理论免除一次 packed QKV、一次 packed gate/up 的前向重算；两项 dX GEMM 仍需执行。需通过真实执行计数验证 early-stop/AOT 下的差异。compile 以计算段为边界，swap/预算编排在图外；仅给已编译整块外套 selective policy 不足以证明内部 GEMM 被保留。

先选单块，再逐步增大预算；同时比较将同等显存用于增加权重 slots 的收益。禁止从 PyTorch allocated 直接推断可安全使用的全部显存余量。

## 验收分工与证据

实现由 `gpt-6-sol` 代理负责；另一名允许模型的代理独立核验。主代理审阅关键差异、集成及最终证据。所有子代理使用清洁上下文；多代理写入使用隔离工作树。

开发工作树：`codex/qwen-stage3-20260927`，基于 HEAD `4df001242` 和开始开发时的当前源码改动快照。集成只带回本任务白名单文件，保留主工作树中其他正在进行的修改。

关联：[结构性优化审阅](qwen_image_2_1_90hx_optimization_20260927.md)、[INT8 首轮热测](../findings/qwen_image_2_1_int8_swap_90hx_20260927.md)。

开发验收、完整模型显存与续训差异的最新证据见
[阶段 3 验收记录](../findings/qwen_image_2_1_stage3_acceptance_20260928.md)。
