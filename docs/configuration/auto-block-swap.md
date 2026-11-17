# AUTO 块交换

状态：实验，默认关闭。Krea-2 NF4 已完成启动校准及全程动态单卡热测；Anima
已完成一次真实 startup AUTO 16 步验收。Z-Image 的 GPU 探针能运行，但本机候选
触发系统换页保护而拒绝正式训练，因此两族都还不能宣称 dynamic 或小 RAM 泛化。

## 开启

Next 工作台 `/next/training` 的「资源与预检 / 模型驻留」提供 `AUTO 块交换（实验）`。
也可在训练 TOML 中设置：

```toml
auto_block_swap = true
auto_block_swap_mode = "startup"
auto_block_swap_max_trials = 6
auto_block_swap_timeout = 1800
auto_block_swap_swap_io_limit_mb = 1024.0
auto_block_swap_vram_reserve_percent = 10.0
auto_block_swap_preference = "balanced"
```

CLI 对应 `--auto_block_swap`，关闭用 `--no-auto_block_swap`。
`blocks_to_swap` 仍为整数，AUTO 只覆盖当前进程中的有效值，不改写用户配置。
旧的手动配置和默认值不变。不是 `blocks_to_swap = "auto"`。

## 显存余量与资源倾向

两个参数同时用于启动校准和全程动态模式，关闭 AUTO 后不生效：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `auto_block_swap_vram_reserve_percent` | `10.0` | 范围 `0..90`，按显卡**总显存**计算保留目标，支持小数。有效余量为 `max(1 GiB, 总显存 × 比例 / 100)`。 |
| `auto_block_swap_preference` | `"balanced"` | `balanced` 均衡，`vram` 优先节省显存，`ram` 优先节省主机内存。 |
| `auto_block_swap_swap_io_limit_mb` | `1024.0` | 系统 SWAP 累计 IO 上限（MiB）；达到上限或物理 RAM 保留线时停止。`0` 表示不设 IO 上限。 |

例如 16 GiB 显卡设为 `25`，目标余量是 4 GiB，而不是当前空闲显存的 25%。
这是峰值预测与候选验收目标，不是驱动层硬配额；设置 `0` 也不会取消 1 GiB 安全底线。

- `balanced` 保留原策略：启动时在最快候选 3% 范围内优先更多交换；动态接受需超过
  3% 收益并能摊销迁移、编译预热成本。
- `vram` 在预算内倾向更多交换，允许约 10% 的计时取舍，不降低主机 RAM 安全线。
- `ram` 在预算内倾向更少交换，并实际释放不参与训练交换的 CPU 权重副本。
  当前仅支持 Krea-2；物理内存较小、显存有余量时适用。
- 动态资源倾向也必须通过负载匹配与漂移检查，预计剩余训练时间（含切换和预热）
  最多比参考增加 10%；反倾向候选只有明显更快时才接受。短时测量不保证实际长训
  损失一定低于 10%。任何倾向都不能覆盖显存余量、物理内存或换页保护。

训练交换参与集合为首 `n` 块与末 `n` 块的并集，不仅是末尾。例如 Krea-2 的 28 块
在 `ram`、交换 4 块时只保留首尾 8 块的 CPU masters；交换数达到 14 时仍覆盖全部
28 块，继续小幅降低交换数未必立即省 RAM。动态增大交换数前检查新增 CPU 副本
和临时缓冲预算，减少后清理相关传输缓存与非参与块副本。

这不消除初次模型加载峰值，也不能让 RAM、VRAM 都不足的机器强行运行。主机压力
物理 RAM 保留线越界仍明确停止；系统 SWAP 只在配置的 IO 上限内作为兜底，不能把磁盘虚拟内存当作等价 RAM；需要保留百分比过高时应降低目标
或调整训练负载，不能把“优先节省内存”理解为忽略显存限制。

## 实验边界

- 单 CUDA 训练进程，Anima / Krea-2 / Z-Image 的普通 LoRA。
- 完整磁盘 latent/text 缓存，`max_data_loader_n_workers=0`。
- 无训练采样、验证集、阶段调度和续训；不支持自定义 network_args 或其他 adapter 分支。
- 不支持 `artist_filter` 和 `debug_dataset` 入口；关闭预览后仍保留样张分辨率的 compile 预算。
- 不支持额外 prior preservation forward（`prior_preservation_weight`、
  `inverted_mask_prior_weight` 必须为 0）。Anima caption dropout 必须预先生成
  `_anima_uncond_te.safetensors`，AUTO 不会临时加载编码器补写该缓存。
- BF16 传输；底模支持 BF16 或 Krea 的已量化 NF4，不进行在线量化。
- 不改变 batch、梯度累积、精度、梯度检查点、学习率或 compile 设置。
- Krea resident compile 在候选数量确定后按原生产顺序执行。

## 全程动态模式

Krea-2 可显式开启动态实验模式，旧配置仍保持启动校准语义：

```toml
auto_block_swap = true
auto_block_swap_mode = "dynamic"
auto_block_swap_interval = 8
```

CLI 使用 `--auto_block_swap --auto_block_swap_mode dynamic`。Next 工作台 `/next/training` 在「资源与预检 /
模型驻留」提供模式选择与评估窗口。动态模式不使用 `auto_block_swap_max_trials`，
`auto_block_swap_timeout` 只约束独立模型盘点进程。

动态模式保留上述实验限制，当前仅支持 Krea-2 与 resident compile 范围。它不是
自动修改 batch、精度、checkpoint 或学习率的功能，也不保证比每一种固定配置快。

- 启动只做实际块数和主机预算盘点，正式训练从预算允许的最大交换数开始。
- 每次完整 optimizer 更新前检查物理内存、换页 IO、可用显存与预测峰值；显存受压
  优先增加交换数。均衡/显存倾向仍保留全块 CPU masters；内存倾向按实际交换集合
  保留副本。物理 RAM 保留线越界仍终止训练；允许的系统 swap IO 达到上限也终止，
  但在上限内可继续使用磁盘 swap 完成块搬运。
- 性能策略逐步探索更少交换数，步幅 2/4/8；短距离收益不足可扩大探测距离。
  也会探测更多交换数。压力解除或冷却结束后继续评估，不永久锁定启动结果。
- 使用按负载匹配的 A/B/A 窗口；每种负载先排除两次预热，默认每窗口至少八次
  有效更新且各负载至少两次。基线漂移超过 10% 时不接受；均衡档的收益需超过 3%，
  且首次编译预热超额耗时、迁移耗时能由剩余收益摊销。资源倾向的取舍见上文。
- A/B/A 评分按基线阶段实际观测的完整梯度累积 shape 频率加权；每轮试验开始时
  冻结权重，候选及复测窗口的频率变化不改变该轮评分权重。各 shape 的安全覆盖和
  漂移门槛仍独立检查，报告记录冻结权重与加权评分。
- 梯度累积按完整更新的 shape 组合及数量匹配，不混比不同更新负载。已知数据集的
  各 bucket 最满 batch 形状都见到之前不晋升；运行中新形状在更新边界回到最大
  交换数。未知形状若出现在非最大交换数的累积中段则明确停止，不带着半批梯度迁移。
- 冷却至少 32 次更新，资源检查不受性能窗口/冷却限制。性能试探迁移 OOM
  回滚迁移并设置临时拒绝边界；显存压力或新形状触发的安全迁移失败则停止训练，
  不以旧驻留配置继续当前 batch。已部分执行的 forward/backward/optimizer OOM 不重试。
- 保留 adapter、optimizer、冻结 NF4 Parameter 身份和 CPU masters；在边界排空
  搬运任务并重建 backward hooks，切换常驻编译范围，复用已编译 callable。
  inactive NF4 块只把打包码停放到 CPU，小型只读量化状态仍可驻留 GPU。

GPU 预测保守使用本进程累计 allocator 水位与常驻权重差值；晋升额外保留至少
512 MiB 或两块大小。驱动余量只是边界时刻的快照，外部进程随后突发占用、未知
负载或编译临时峰值仍可能 OOM。非常不均匀的 bucket/累积组合可能延迟性能探索，
但不会关闭资源检查。当前不支持自动保存后重启或动态模式续训。

报告在盘点 `summary.json` 同目录的 `runtime.jsonl`，记录更新耗时、块数、余量、
切换原因、A/B/A 判定、迁移和控制器耗时及结束状态。结束状态只标记训练循环，
最终 checkpoint 保存是否成功仍以正式训练结果为准。

## 启动校准过程

1. 父进程在加载模型前启动独立盘点进程。读取实际 `blocks` / `layers`，
   当前三族都至少保留两块常驻，且未知模型族明确拒绝。
2. CPU 模型加载后估计 masters、两块 scratch、tail 副本和物理内存余量。
   均衡/显存倾向按全块 masters 预算；内存倾向按首尾交换集合预算，不能只算 tail。
3. 从模型上限与主机预算共同允许的最大交换数开始。每个候选重新启动进程，
   走正式模型加载、adapter 应用、compile、optimizer 准备顺序。
4. 每个数据集的每个有效分辨率取一个最满的真实 batch，大负载先测；
   每种情况执行三次完整 optimizer 更新，每次覆盖用户设置的梯度累积。
   首次更新用于冷启动与 optimizer 状态分配，后两次用于计时。
   报告为每个代表情况记录该数据集、该分辨率的实际 batch 次数；候选 `seconds`
   是各情况热更新中位时间按这些次数加权的估计，另保留等权
   `unweighted_seconds` 供诊断。梯度累积仍由每次探针更新完整覆盖，权重按
   dataset batch 次数估计，不模拟跨 bucket 的真实累积组合；稀有尾批仍由最满
   batch 做安全覆盖，不能把加权估计当作真实长训加速证据。
5. 逐渐减少交换数，步幅 2、4、8。CUDA OOM 或显存余量不足时在成功/失败区间二分。
   非 CUDA OOM、主机资源压力、超时、非有限 loss 或跳过更新都不能当作可恢复训练失败。
6. 只从通过的候选选择；均衡档在最快值 3% 内优先更多交换、更大显存余量。
   显存/内存倾向在最快值 10% 内分别选择更多/更少交换。
   最后另启进程确认；确认资源失败、非有限/零耗时，或相对选定候选的加权
   `seconds` 漂移绝对值超过 10%，均不启动正式训练。`summary.json` 保留确认
   时间、候选时间、漂移比例、门槛及稳定判定；阈值暂不可配置。

搜索默认最多六个候选，另加盘点和最终确认。不是全范围穷举，也不保证全局最优。
多 bucket 和冷编译可能令校准耗时很长；超时包含加载、编译和全部更新。

## 内存与安全

主机预算只计算可用物理 RAM，并在 Linux cgroup v2 可读时取容器限制。
保留 `max(2 GiB, 主机容量的 10%)`；系统 swap/pagefile 不计入物理容量，但可在
`auto_block_swap_swap_io_limit_mb` 上限内作为块交换兜底，并单独记录累计 IO。
监控每 50ms 采集主机可用内存和进程树 RSS，触及余量停止自己的候选进程。
超过配置的系统换页 IO 上限会拒绝成功候选，可能受到其他进程干扰；物理可用 RAM
触及保留线时始终拒绝。
这不是操作系统硬配额，不能保证在极快的主机内存分配突发前阻止系统 OOM。

GPU 保留 `max(1 GiB, 总显存 × auto_block_swap_vram_reserve_percent / 100)`，默认 10%。使用完整运行的 allocator 峰值和
驱动可用显存预算，包含首个 backward、optimizer 状态以及编译/加载分配。
短探针不能证明后续所有文本长度、数据顺序或外部显存竞争都安全。

探针不进入正式 epoch 循环的采样/保存阶段，也不写训练 checkpoint。
临时请求文件权限为 0600，进程结束删除；无 pickle、无 tracker/HF 上传。
取消和超时只终止该候选创建的进程组，并等待退出。

报告位于 `<output_dir>/auto-block-swap/calibration-*/`，包含 summary、候选 JSON、
日志和 block-swap profile。保留失败证据，不清理用户目录。
`startup` 选定值在训练期间固定；低余量时低频警告，不热切换 offloader。
`dynamic` 的边界和行为见上文。自动 checkpoint 重启、校准结果跨运行复用尚未实现。

## 验证

```bash
python -m pytest tests/test_auto_block_swap*.py
python -m pytest tests/test_dynamic_block_swap*.py
python -m pytest tests/test_sparse_block_swap_masters.py
```

测试包含真实 CPU autograd/AdamW/Accelerate 累积、纯策略边界、故障注入、超时和取消、
配置隔离与前端类型。资源故障通过 fixture 注入；另有 CUDA 小模型测试验证 NF4
前向/梯度、存储身份和 compiled callable 复用。这些单元测试不是大模型性能证据。
资源参数测试额外覆盖总显存百分比在探针、动态控制器与固定监控中的一致性、模拟小
RAM 预算、资源倾向选择，以及真实 CUDA 小模型的稀疏 CPU masters 增减、slab、
NF4 eager/compiled 零容差前向与梯度、迁移 OOM 回滚。尚未新增受限物理 RAM 的
大模型热测，不能据此承诺所有小内存机器都可训练。

资源参数回归中，既有 BF16 forward-only slab 回环零容差测试再次出现偶发失败；
相同测试在隔离组合和逐层诊断中也有通过记录，根因尚未确认，未放宽容差。
该测试是推理路径，稀疏 CPU masters 当前仅用于训练，但不能据此排除共享 offloader
风险。实验状态保留，不能将本轮定向测试通过解释为整个 block-swap 测试集无故障。

纯策略合成消融（`tests/test_auto_block_swap_ablation.py`）结果如下，探针数不含盘点
和最终确认；它验证搜索机制，不能用于宣称实际 GPU 速度或显存收益：

| 策略 | 选定交换数 | 候选探针数 |
| --- | ---: | ---: |
| 完整策略 | 18 | 7 |
| 固定步幅 1 | 18 | 10 |
| 不做失败二分 | 20 | 4 |
| 不保留显存余量 | 16 | 7 |

该合成场景使用独立的试验预算，不等同于生产默认六次候选上限。
无显存余量一项的结果越过了场景设定的安全边界。

真实 GPU 测试与消融见
[Krea-2 单卡热测报告](../findings/auto_block_swap_hot_20260908.md)：完成普通/受压 AUTO、
固定步幅、无失败二分、无安全余量、72 次 A/B/A 长窗口更新和 16 步正式保存。
部分消融复用公共硬件候选，但确认均独立新测，报告明确区分计数与耗时。

本次 4 块在显存受压时实际 OOM，8 块虽然完成更新仍因余量不足被完整策略拒绝。
普通 AUTO 选 4 块，受压 AUTO 选 26 块；两次 26 块长窗口基线相差约 22%，
不能宣称 AUTO 稳定加速。当前最终确认新增 10% 时间漂移拒绝，但不会重新排序；
历史试验结果不因此改写。更多模型/负载验证之前，继续保持显式开启，
不自动推广为默认配置，也不能宣称所有 GPU 安全。

运行中调整的真实事件、固定对照、NF4 存储修复与单独策略消融见
[全程动态验收报告](../findings/auto_block_swap_dynamic_20260908.md)。动态模式使用
A/B/A 复测和成本门槛；实测23块只比25块快约0.9%，正确拒绝了低于3%门槛的候选。

## blocks_to_swap 的合法区间（按族）

`blocks_to_swap` 是整数，**合法上界按模型族不同**，不是全局统一值。各族都要求至少保留
两块常驻（换出集合是首 `n` 块与末 `n` 块的并集），所以上界为「换出块总数 − 2」：

| 模型族 | 可换出块总数 | `blocks_to_swap` 合法区间 | 单块载荷 | 上界处的每步搬运量 |
| --- | ---: | --- | ---: | ---: |
| Anima | 28（`net.blocks`） | `0`（关闭）或 `1..26` | 132.0 MiB | 7197 MB |
| Krea-2-Raw | 28（`blocks`） | `0`（关闭）或 `1..26` | 828.1 MiB | 45153 MB |
| Z-Image | 30（`layers`） | `0`（关闭）或 `1..28` | 345.1 MiB | 20262 MB |
| Qwen-Image 2.1 | 32（`transformer_blocks`） | `0`（关闭）或 `1..30` | 416.0 MiB | 26172 MB |
| MOSS-Image（`moss_image`） | 42（`layers`） | `0`（关闭）或 `1..40` | 216.0 MiB | 18120 MB |

每步搬运量 = `2 × blocks_to_swap × 单块载荷`（每个换出块一步恢复两次）。给出上界是因为
超限会被断言或校验直接拒绝，典型报错：

- Anima：`AssertionError: Cannot swap more than 26 blocks. Requested: N blocks.`（`library/anima/models.py`）；
- Krea-2-Raw：同样的 `Cannot swap more than 26 blocks` 断言（`library/models/krea2_raw/dit.py`）；
- Qwen-Image 2.1：`blocks_to_swap must be between 0 and 30`（`library/training/compat_matrix.py`）；
- Z-Image：`Z-Image has 30 main layers and requires at least two resident layers; use blocks_to_swap between 0 and 28.`；
- MOSS-Image：`ValueError: MOSS blocks_to_swap must be 0..40`（`library/models/moss_image/block_swap.py`）。

`0` 表示完全关闭块交换。这些数字来自只读权重头部的真实字节量并经真机训练核对；推算方法见
[各族分块载荷与晋升算术](../findings/block_swap_family_payload_50hx_20261019.md) 与
[MOSS 块交换字节规律](../findings/moss_block_swap_law_20261021.md)。
单块载荷决定「省一块」的价值：Krea-2 上少换出一块约省 0.55 s/步，是 Anima（0.088 s/步）的
6 倍以上，所以大块族更应该开启自动水位。

## 各族运行时键约束（训练前校验）

训练启动时 `library/training/compat_matrix.py::check_training_compat` 会结合
[`--compat_unverified_policy`](model-training-capabilities.md#兼容预检策略--compat_unverified_policy)
（`strict` / 默认 `warn` / `allow`）对各族运行时键做分级校验，不再一律按单档报错：

| 类别 | 在 `strict` / 默认 `warn` / `allow` 下的行为 | 典型键（Z-Image / Qwen-Image 2.1 / MOSS） |
| --- | --- | --- |
| **C1（实现已生效、仅缺真机验收）** | `strict` 报 error；**默认 `warn` 与 `allow` 降级为带 `[unverified 未验证]` 标记的 warning 放行** | Z-Image / Qwen-Image 2.1 的 `loss_type`、`weighting_scheme`，Z-Image 的 `discrete_flow_shift`，Qwen-Image 2.1 的 `masked_loss` 与 `torch`/`sdpa` 下的 `mixed_precision` |
| **C2-ignored（族内不读取的无效果开关）** | `strict` 与默认 `warn` 仍报 error；**仅在显式 `--compat_unverified_policy allow` 时降级为 warning 放行** | Z-Image / Qwen-Image 2.1 的 `timestep_sampling`（不为 `"uniform"` 时）、`base_compute`、`selective_checkpoint` 等 |
| **硬约束（C2-raise / C3 / A / B 与族专属门禁）** | **三档恒为 error，任何档位都不会放行** | Z-Image / Qwen-Image 2.1 的 `caption_dropout_rate != 0`、`weighted_captions`、非法 `attn_mode`（仅 `torch` / `sdpa` / `flash`）与越界 `blocks_to_swap`；MOSS（`moss_compat.py`）的全部契约项 |

手写运行时 TOML 或拼装 CLI 时最容易踩到的仍是 **`timestep_sampling`**（属于 C2-ignored，默认 `warn` 下依旧报错）：
`configs/base.toml` 默认 `timestep_sampling = "sigmoid"`，走标准
`base.toml → presets.toml → methods/*.toml` 合并链时方法配置会将其覆盖为 `"uniform"`；
若绕开方法配置手写 runtime TOML 或临时 CLI，未覆盖的 `sigmoid` 在默认 `warn` 档位下仍会被预检拦截。
标准修法是在配置或 CLI 中显式写 `timestep_sampling = "uniform"`（完整分级列表与审计依据见
[兼容预检策略说明](model-training-capabilities.md#兼容预检策略--compat_unverified_policy)）。

## 进程内水位调优（--block_swap_auto，2026-10-19 新增）

上面的 AUTO 是独立进程启动校准。本节是另一个更轻的机制：在当前训练进程内，跑完第一个优化器步后
量一次自己的空闲显存，把余量换算成「还能让多少换出块常驻」，在优化器边界用现成的
ModelOffloader.reconfigure() 事务一次性晋升。两者可以并存，但不共享策略。

```toml
block_swap_auto = true
block_swap_vram_margin_mib = 400     # 默认 400 MiB
block_swap_auto_stop_ratio = 0.0     # 默认 0.0（不设下限，按显存余量尽量晋升）
```

CLI 对应 `--block_swap_auto` / `--no-block_swap_auto`、`--block_swap_vram_margin_mib` 与
`--block_swap_auto_stop_ratio`（环境变量 `ANIMA_BLOCK_SWAP_AUTO_DMA_RATIO_FLOOR`）：`0.0`（默认）表示只受空闲显存余量约束尽量晋升常驻块以最大化吞吐；设为正比例（如 `0.5`）时，一旦预计剩余 DMA 搬运耗时低于实测步时的该比例便提前停止晋升，以极小速度代价保留更多换出块和显存缓冲。

| 维度 | 启动校准 AUTO | 进程内 --block_swap_auto |
| --- | --- | --- |
| 生效时机 | 训练前，多次独立候选进程 | 第一个优化器步之后，一次边界事务 |
| 决策依据 | 候选实测耗时 + 显存/内存策略 | 本进程 mem_get_info() 空闲显存与 per-block 字节 |
| 是否改 blocks_to_swap | 只覆盖本次运行的有效值 | 只下调（晋升常驻），不会上调 |
| 失败行为 | 候选被拒绝 | 事务自动回滚，保留原水位；异常只告警不中断训练 |
| 典型用途 | 跨分辨率/卡型首次定档 | 已知保守水位时的自动收敛 |

实测（CMP 50HX，Anima 512×512，保守起点 blocks_to_swap=24，margin 400 MiB）：
日志 promoting 22 block(s) to resident, blocks_to_swap 24 -> 2 (free=7.57 GiB)；
每步任务数 48 → 4、H2D 字节 6644 MB → 554 MB、DMA 2322 ms → 178 ms。
晋升上限受 offloader 契约约束（至少保留 2 块换出），因此该卡在 512 下收敛到 swap=2。
证据与边界见 [Block Swap 进程内水位调优（CMP 50HX 第 3 轮）](../findings/block_swap_auto_tuner_50hx_20261019.md)。
