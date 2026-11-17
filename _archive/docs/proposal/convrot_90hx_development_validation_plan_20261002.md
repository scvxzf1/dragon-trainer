# ConvRot INT8：CMP 90HX 开发与重新验证计划

状态：S0–S4 工具、基线、优化筛选与生产功能验证已完成；S5 因没有通过门槛的候选而跳过（不是500步完成）。原协议与门槛保留。结论见[90HX验证报告](../findings/convrot_90hx_validation_20261002.md)，后续见[开发计划](convrot_90hx_followup_development_plan_20261002.md)。日期：2026-10-02。

目标：在当前 CMP 90HX 上重新比较 BF16、现有 W8A16、现有 W8A8；再分别验证保持数值规则的 Triton 优化与改变反向精度的方案，决定是否值得用于真实 Anima LoRA 训练。

依据：[本地与 musubi 专项对比](../findings/int8_training_local_vs_musubi_20261002.md)、[已有架构规格](convrot_w8a_training_plan.md)、[3080 优化路线](convrot_w8a_optimization_roadmap.md)。本计划补充新的硬件与验证协议，不覆盖历史结果，不把旧 3080 数字当成当前基线。

## 1. 当前资源和固定输入

2026-10-02 20:42 +0800 的只读快照：

- GPU 1：NVIDIA CMP 90HX，10240 MiB；UUID `GPU-806debd9-d254-da57-d515-ccae15b4ac6a`。查询时利用率 0%，未列出该卡 compute 进程。空闲状态只是快照，每次执行前重新检查。
- 驱动 `580.178.04`；`.venv` 分发元数据为 torch `2.12.0`、Triton `3.7.0`、bitsandbytes `0.49.2`、flash_attn `2.8.3+cu130torch2.12`。本轮没有 import CUDA 验证 ABI；S0 记录实际 CUDA runtime、SM、设备身份和算子可用性。
- BF16 基座：`/home/scv/pac/models/diffusion_models/anima/anima-base-v1.0-sha.safetensors`。
- 可复用输入：[1024 数据集配置](../../configs/datasets/anima-ab-1024-20261002.toml)，60 张图片、独立 cache、关闭 bucket。文件存在性已核，cache 内容/schema 由 S0 核验。
- 可复用 profile：[上轮 BF16 配置](../../output/anima-mixed-swap-fix-20261002-163852/bf16.toml)。只提取公共参数，重新生成独立输出路径和训练时长，不直接启动旧配置。

首轮固定 Anima、1024×1024、batch1、accumulation1、plain LoRA rank/alpha32、AdamW、LR 2e-5、BF16 compute、full checkpoint、Flash、swap12、BF16 transfer、foreach restore、compile off。禁用 adaptive precision、自动 swap、OOM 自动降档和 caption dropout，避免对照过程中配置漂移。

ConvRot 首轮使用 `scope=all`、Sylvester@256、dense RHT、释放 dense 基座；三组均从同一 BF16 文件派生。NF4 权重不进入此次主矩阵。LoRA 使用相同保存的 FP32 初始 state，核验逐 tensor dtype/hash；固定 optimizer、scheduler 和样本/噪声/时间步流。

设备用 `CUDA_VISIBLE_DEVICES=<上述 UUID>` 锁定，进程内统一 `cuda:0`。不能依赖物理 index 永久为 1，或旧探针的“自动选择最大 GPU”。不修改功耗、频率、风扇；记录温度、功耗、时钟、利用率和其他负载。

## 2. 对照编号与开发顺序

| 编号 | 内容 | 用于回答 |
|---|---|---|
| B0 | 当前 BF16 基座 | 原始训练速度、显存与数值参考 |
| B1 | 当前 W8A16 ConvRot | 现有压缩基座方案是否已经足够好 |
| B2 | 当前 W8A8 ConvRot，强制真实 INT8 前向，FP32 backward | 当前 90HX 上是否重现性能倒退 |
| E1 | B2 只换 Triton rowwise quant | 保留取码规则能否减少量化成本 |
| E2 | B2 只换 INT8 GEMM + 融合缩放 | 减少 INT32/FP32 中间输出是否有效 |
| E3 | 合并通过验证的 E1、E2 | 纯工程优化的完整训练收益 |
| N1 | 胜出工程方案，只换 BF16 backward | 改变反向精度的独立收益与误差 |
| N2 | 胜出工程方案，只换 INT8 backward | 进一步量化输入梯度计算是否值得 |

N1 与 N2 都对照同一个 FP32-backward 父方案，不把 INT8 backward 与 N1 的额外变化混在一起。若没有 E 类胜出方案，其父方案就是 B2。regular@256 作为另一个单变量数值实验，先不与 N1/N2 叠加；只有证据支持再组合。scope/尺寸阈值混精也只在实测显示某类矩阵有收益后启用。

## 3. S0：先让验证工具可信

这是第一批开发交付，不先修改生产数值策略。现有探针不能原样作为此次验收器：

| 已核对的问题 | 处理方式 |
|---|---|
| mem-speed 每步逐参数 `.item()` 求梯度范数，影响同步和步时 | 数值采集与吞吐分开；计时区不逐 tensor 回传 CPU |
| `x == x` 只排 NaN，Inf 会通过 | 对 loss、梯度、更新和最终权重做明确 finite 检查 |
| microbench 的 `int_mm_only` 包含每次转置，且缺生产 M padding | 单独记录 setup；预生成 KN 权重；与生产对齐/crop 一致 |
| microbench backward dtype 与生产 W8A8 不同，含旧硬编码端到端锚点 | 使用实际 FP32 backward；旧锚点标为历史，不进入新汇总 |
| checkpoint probe 只比较梯度总范数 | 增加命名参数逐层及整体梯度向量、更新向量对照 |
| step-profile 固定 rank4/torch attention，mem-speed 也固定 torch attention | 新 runner 显式复用 rank32/Flash 公共配置，不混用不同栈的耗时 |
| short-train 的 skip-sample 未控制实际采样，部分 gate 恒真 | 新真实训练 runner 走已有训练/采样入口；修正或明确停用旧误导性字段 |
| W8A8 auto 可静默浮点 fallback | 严格验证模式不允许静默回退；记录实际 backend/失败原因 |

位置：`scripts/experiments/convrot_mem_speed_probe.py:286,397,461`、`convrot_fusion_microbench.py:193,229,547`、`convrot_checkpoint_probe.py:313`、`convrot_step_profile_probe.py:193`、`convrot_short_train_probe.py:188,371,433`。实现时重新读取实时代码，不按行号盲改。

S0 产物：设备/软件/source manifest；三份独立 baseline 配置；相同初始化；可重放输入 manifest；分离的数值、计时、显存 runner。记录有效配置而非只记录 TOML 差异。

数据核验必须保留 Anima text max-padding，不裁剪 attention sinks。1024 的实际 M/K/N 从运行时捕获，不能把 latent H×W 当作所有 Linear 的 token 数；text cross-attention 也有自己的 M。缓存缺失只补新实验所需文件，不覆盖旧实验数据。

**验收：** CPU 测试和目标卡 CUDA smoke 通过；NaN/Inf、错误设备、fallback、无效配置均能明确失败；一次无 profiler 的计时流程可产生完整 manifest、状态码和独立输出。

## 4. S1：重建当前 90HX 基线

按下面顺序推进，先解决可运行性，再扩展统计：

1. B0/B1/B2 各 3 个有效 optimizer steps，验证有限值、payload placement、保存；B2 强制 INT8 前向，不能用浮点 fallback 掩盖兼容失败。
2. 在实际模型上生成两份代表性缓存输入，固定三个 seed、低/中/高三个实际 timestep（记录采样器映射后的值）；共 18 个重放样本/模式，计算输出、loss、完整 adapter 梯度和一次 AdamW 更新。每个单次更新样本分别恢复相同的 adapter、optimizer、scheduler 和 RNG 状态，不承接前一样本的更新；这与后续连续 20 步轨迹实验分开。先只跑一个样本调通，再展开。
3. 性能测量每模式每轮 10 步预热 + 30 步测量，3 轮独立进程；顺序轮换 B0/B1/B2、B1/B2/B0、B2/B0/B1，共 360 步。预热不足或温度持续漂移时将该轮标为无效，稳定后重测，不混进统计。
4. 另开诊断进程，每模式捕获 1–3 个稳态 step 的 profiler；另做 8 步显存阶段追踪，第 4/8 步采样一张 1024 图，验证训练→采样→训练。诊断同步和 reset peak 的结果不用于吞吐排名。

固定 swap12 是主对照。初始 smoke 装不下时，保存失败记录，给三组统一增加 swap 后重建主基线；不得只给某一组隐式降档。swap0 若 OOM，仅记不可运行；不持续盲试。

输出必须包含：

- 稳态完整 step wall time、窗口吞吐、p50/p95。吞吐窗口的开始/结束等待相关 CUDA/offload streams 及待完成的 CPU offload 工作完成，计时区不逐步同步；逐步同步延迟另开诊断测量并明确标注，不用其 p50/p95 冒充原流水线吞吐。CUDA Event 算子时间单列，不能代替 CPU offload/optimizer 等完整耗时。
- allocated/reserved（统一 bytes + GiB）、NVML 进程显存、apply/loading 峰值、forward/backward/optimizer 峰值、采样和回训峰值。NVML 采样峰值注明采样间隔，不伪装成精确瞬时峰值。
- RHT、取码、INT8 GEMM、缩放、dX、checkpoint 重算、H2D/D2H 与等待；profiler self-time 只做定位，不直接加成速度收益。
- 每组实际 patched module/shape、权重/scale dtype、量化 payload CPU/GPU bytes、后端、compile/TF32 状态。严格验证期显式设置相同 FP32 matmul 策略并记录；与正常默认策略有差异时另列配置。

**阶段判断：** S1 要回答当前 W8A8 相对 W8A16 慢在哪里；不强求复现旧 3080 的百分比。若当前软件使基线质量检查失败，先记录并定位，不降低阈值制造通过。

## 5. S2：保持数值规则的工程优化

先由 S1 真实 shape/call count 建立矩阵清单，开发 microbench 覆盖 1024 实际 M，以及 4032/4200 兼容形状；后两者无实际数据时明确标为合成算子测试。

**E1：** 一个 Triton rowwise kernel 完成 absmax、scale、FP32 除法、round、clamp。保持本地 epsilon 位置、`[-127,127]`、ties 规则和输出 scale；不采用 musubi BF16 商舍入。FP32 中间值尽量留在寄存器，不物化完整 FP32 activation。

**E2：** INT8 GEMM 内执行 INT32 accumulator→FP32→依次乘 activation scale、weight scale→目标 dtype；使用生产布局并正确 mask M/N/K 尾块。保持本地缩放乘法顺序，防止编译器重关联改变 BF16 舍入。没有 Triton或形状不支持时显式标记，性能测试禁止静默回退。

E1/E2 分别对照 B2；只有通过的组件合并为 E3。RHT、FP32 backward、scope、权重量化与 LoRA apply 顺序不同时改变。现有 `gemm.py`/`fused.py` 仅做薄 dispatch，新 kernel 放独立模块。

**严格局部契约：** 固定相同输入和 payload，INT8 codes、scale、整数累加结果要求一致；保持运算顺序的 BF16 输出以逐位一致为目标。若出现不一致，先定位，不能自动归入“允许量化误差”。FP32 局部结果需要容差时预先固化公式/容差和原因，并转入独立数值差异记录。

边界覆盖：全零/极小值/outlier/负值、round ties、饱和、非连续输入、leading dims、bias与dtype、M=1/16/17/4032/4200、尾块 mask、非法 K/N、NaN/Inf 输入策略。基座冻结情况下用固定外部 grad_out 检验 dX，再用真实 loss 链检验 adapter 梯度；不能只验证 additive LoRA 分支。

**推进门槛（工程目标，不是已有结果）：** 热点算子改善至少 10%，并由调用次数估算完整 step 有机会改善至少 5%；或明确减少至少 256 MiB 的峰值主导临时内存。未达到时停止扩大该组件，仅保留诊断结果。旧 3080 的“self-time >50% 才开 Triton”不作为硬门槛；新判断基于 90HX 同条件测量和有限范围原型。

## 6. S3：将低精度反向作为单独数值实验

只在 S1/S2 已能解释成本后进入。N1 首先实现 musubi 式 `dy @ (cast(Wq) * scale)` 的 BF16 backward；必要时增加本地式 `(dy * scale) @ cast(Wq)` 对照，明确区分舍入顺序。N2 单独量化经过 scale 的上游梯度并执行 INT8 GEMM，记录转置/缓存的额外驻留和时间，不能只比较 matmul 主体。

默认继续 FP32 backward。新策略必须显式 opt-in、写入有效配置与 checkpoint metadata，resume 时恢复或拒绝不一致；不使用全局 monkey patch 暗中改变其他训练。

数值验收分两层：

| 比较 | 必须记录/检查 |
|---|---|
| 工程候选 vs 同数值策略父方案 | codes/scale、输出、dX、完整 adapter 梯度、一次及 20 步更新；同一初始化/输入流；参考自身重复运行用于识别非确定性 |
| 数值候选 vs FP32-backward 父方案及 B0 | 输出 relative L2、loss 相对差、整体/逐层梯度向量 relative L2/cosine/max_abs、总范数、missing/zero/nonfinite、AdamW 更新与最终参数 |

初始筛选阈值在跑候选前冻结：沿用输出 relative L2 ≤3%、loss 相对差 ≤5%、总梯度范数差 ≤5%，新增非零梯度整体向量 relative L2 ≤5%、cosine ≥0.995、一步更新向量 relative L2 ≤5%。这些是项目拟定的保守筛选值，不是理论质量保证。父方案自身先测；若父方案也不通过，记录限制并阻止质量等价声明，另行论证改标准，不能看完候选后悄悄放宽。

近零梯度不使用不稳定的相对误差/cosine强行评分：记录绝对误差与范数，S0 依据父方案梯度尺度冻结零阈值。逐层报告保存最差层，不能用大层整体范数掩盖小层异常。LoRA 零初始化造成某分支第一步零梯度，必须在后续非零状态再测。

逐层规则同样在 S0、候选结果揭晓前冻结：参考非零分支出现 missing gradient、nonfinite 或异常归零立即失败；非近零层初始采用梯度向量 relative L2 ≤10%、cosine ≥0.99 的局部上限。近零层的绝对误差预算依据参考自身重复误差与尺度明确写入 manifest，未定义的层标为未验收；超过局部上限不能仅因整体指标通过而放行。

单步筛选过关后用 3 seeds × 20 步配对轨迹；记录每步梯度/更新和固定验证输入输出，长期参数漂移不要求逐位一致。任何 seed 出现 nonfinite、持续爆炸或未解释的明显偏离，停止该候选。只有局部通过不能进入默认配置。

## 7. S4：完整训练集成、换层与实用预算

仅扩展胜出候选，避免全矩阵笛卡尔积。先 B0/B1/候选主 profile 的无 compile 测试，再验证同样的 compile 配置；compile 必须在 adapter apply/load 后，保留 Anima native shapes，不套用 Krea2 的编译方案。

- swap12 为主；增加 swap0/26 的独立 smoke/显存点，验证 carrier 生命周期。若不可运行或交换范围不合法，明确记录，不用自动 fallback。
- 对实际可用内存预算，按 swap0→4→8→12…进行受控搜索；OOM 后退出该子进程，失败配置原样存档。每种精度选择满足相同预算/余量的最小可行 swap，再比较完整吞吐。
- 统一预算由 S0 当前可用显存确定，预留至少 512 MiB 运行余量；保留训练+采样峰值，不以步末 allocated 代替容量判断。
- 检验采样前后 payload placement、逐步显存趋势、换层临时缓冲是否增长；训练→采样→回训→保存→新进程 resume 必须成功。
- resume 小测建议连续 10 步对照 5+保存+恢复+5 步，固定 RNG/optimizer/scheduler；核对有效更新次数、参数与结果，在参考自身非确定性范围内比较。
- 当前单卡任务无需铺开 distributed 或所有 adapter 变体。保持 plain LoRA，独立验收通过后再扩范围。

最终实用门槛：在相同质量筛选与显存预算下，候选相比最佳 B0/B1 的完整 step 用时降低至少 5%，且三轮方向一致；或全程峰值下降至少 10% 且至少 256 MiB，步时退化不超过 5%。3% 以内差异默认记作无明确收益，3–5% 为待确认；记录原始波动，不凭单轮最快值晋级。

仅比旧 B2 快而仍输给 B1，可以交付为实验内核，不标为推荐训练档。若优势只出现在减少 swap 的配置，明确标注收益来自整体配置，而不是全部归因于 INT8 GEMM。

## 8. S5：500 步真实训练验收

沿用用户此前指定的真实验证尺度：每组 500 个有效 optimizer steps、训练/采样 1024×1024，第 250/500 步采样。500 步是一次实验轮次，不将其误写成数据集单个 epoch。

最多三组：B0、B1、最终 INT8 候选；若没有候选达到前述门槛，则不开展无目的的三组长训，仅交付否决依据，必要时验证 B0/B1 的实际价值。500 步阶段先跑一个共同 seed，结论仅限该任务/seed；需要推广时再扩种子和数据集。

沿用现有 60 张图、标签来源和共享 cache。若质量目标包含泛化，S0 在实验副本中用内容 hash 分组固定 48 train/12 held-out，三组共用划分，显式绕过小样本自动关闭验证的旧路径；检查副本/cache sidecar可读取且无泄漏。若只做完整 60 图训练，固定验证只能称为训练集诊断，不声称 held-out 泛化。

采样选 4 条不重复的代表性 caption 衍生提示词 × 2 个固定 seed，共 8 张/采样事件；内容在 S0 固化，三组共用，不在跑完后挑有利图片。Euler、相同 CFG/采样步数/负面词/VAE，建议正式观察用 20 步；第 0 步样图在独立评估进程生成，避免污染计时窗口。

两种评估分别回答不同问题：

1. 原生训练基座 + 所得 LoRA：真实交付组合的采样质量、速度和采样峰值。
2. 公共 BF16 基座 + 各组 LoRA：adapter 是否因量化补偿而影响迁移使用。独立评估进程执行，不混入原生训练峰值。

质量报告包括：固定验证样本/噪声/时间步的预测 loss、轨迹异常、同排列样图、人物/风格保真和明显结构/色彩缺陷。不能以训练 loss 更低、图片像素接近或单张主观观感宣布整体质量等价。

保存 250/500-step adapter、训练状态、有效配置、初始与最终 hash、原始进度和采样元数据；明确最终 500 步保存已发生，避免旧实验“跑满但末次权重未保存”的缺陷。

## 9. 代码边界、交付与执行规模

| 工作包 | 建议责任范围 | 验收者 |
|---|---|---|
| 验证基础设施 | 新 `bench/convrot_90hx/` 的 manifest/input/numerics/performance/report 模块；旧探针只必要修正或薄转发 | 非实现者审查数值公式、计时边界和失败处理 |
| 前向内核 | `library/runtime/convrot/` 下独立 Triton quant/GEMM 模块与薄 dispatch；对应定向测试 | 独立检查边界 mask、取码、dtype、fallback、梯度与 compile |
| 反向策略 | ConvRot 专属 autograd/配置/metadata，小范围接线，默认不变 | 独立验收 N1/N2 的真实梯度、保存恢复及差异归因 |
| 真实训练与报告 | 新实验配置/输出；复用训练与采样入口 | 主代理核验配置、原始日志、性能和样图证据 |

上述新模块是计划接口，不是当前已存在的命令。执行实现时先确认对应目录未被他人使用。当前工作树很脏；冻结此次依赖文件 hash/diff，保留用户现场。需要并行修改时用独立工作树，明确带入的当前依赖版本，不能悄悄改用线上 dev 而改变基线，不 stash/revert 用户改动。

每次实验新建 `output/convrot-90hx-<timestamp>/`，保留 `manifest.json`、配置、输入/初始化 hash、逐步 JSONL、backend、原始 profiler、采样元数据、失败记录和 `summary.md`。成功/失败都保留；源数据与旧报告不覆盖。

计划内工作量：S1 主性能矩阵 360 步加数值/短测；E/N 候选逐级淘汰；S5 最多 1500 步加采样。运行预算在 S1 后用实际秒/步估算，加上加载、编译、采样和重试，当前不承诺固定小时数。

执行前先跑相关 CPU 定向测试，随后将 CUDA 测试锁到目标 UUID；不把 CUDA skipped 当作通过。独立验收只复核目标变更与证据，不重跑无关全套训练。

每阶段交付后标注 PASS / FAIL / INCONCLUSIVE 和原因。失败优先缩小到具体 shape/seed/backend，不直接增加训练步数。最终报告分别回答“数值可用”“速度有益”“全程省显存”“值得默认推荐”，四项不得互相代替。
