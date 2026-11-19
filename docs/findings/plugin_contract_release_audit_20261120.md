# 通用层与族插件修复及复审表

状态：修复中

基线为 `dev` 的 `7901d68a`。P4 原工作树以 `93aaeddc` 为基线，P5 原工作树以 `24efcc36` 为基线。开发和集成在独立工作树执行。

| 编号 | 修复项 | 合入验收 | 状态 |
| --- | --- | --- | --- |
| P0-1 | 集成 P4/P5 声明、实现和测试 | 保留 P4 Flow/cache 和 P5 独立推理/预览；同一树跑回归 | 进行中 |
| P1-1 | PP 精度报告阶段、更新计数与退出状态 | 共享事务事件映射到报告，显式区分尝试、提交和跳步 | 进行中 |
| P1-2 | PP 续训出现零步成功 | 核查实际数据游标和目标步数，同数据集从保存点继续，非法零步具名拒绝 | 进行中 |
| P1-3 | WebUI PP 被记录为 DDP | 队列和历史消费同一有效模式，保留旧记录兼容 | 进行中 |
| P2-1 | Anima 正式训练绕过共享 Flow | 通用层持有主流程，Anima 保留辅助目标、max-padded 文本及路由顺序 | 待修复 |
| P2-2 | sigma、模型 timestep、训练张量边界不完整 | 区分 sigma 与 timestep，检测 GPU 非法值，按 registry 检查 latent 通道与形状 | 待修复 |
| P2-3 | Latent cache 几何和 flip 未完整验证 | 验证 batch、通道、步长、空间尺寸、普通及翻转形状，保留旧缓存兼容 | 待修复 |
| P2-4 | 文档与已合入版本不一致 | 标记固定代码与执行证据，记录已验证边界和完整提交清单 | 待复审 |
| P2-5 | Anima 独立推理以空资源包装 legacy 整段 | 保留 Anima adapter/采样语义，生产入口显式分离 TE 编码、DiT 条件化/采样、VAE 解码，分别验证释放与异常 | 已定位 |
| P3-1 | TE 会话恢复和生命周期边界 | eager 与族管理的 offload 分工明确，验证惰性加载成功、异常和卸载 | 待修复 |
| P3-2 | 通用策略基类位于 Anima | 中立基类单一来源，Anima 旧路径保留兼容导出 | 待修复 |
| P3-3 | 插件协议粒度过粗 | 能力分开验证，缺失能力或返回不满足边界时具名拒绝 | 待修复 |

## 复合审计

当前判定为部分符合，发布门禁未通过。注册表惰性分派、未知族拒绝和各族缓存布局属于已接入的契约。共享流程的实际所有权仍有下列缺口。

| 路径 | 判定与证据 | 开发及验证意见 |
| --- | --- | --- |
| P4 正式训练 | **违反共享编排契约**。`library/training/batch_step.py:59` 经 `library/anima/plugin.py:38` 直接进 `library/anima/noise_target.py:37`。Anima 自行执行噪声、采样、路由与前向，`tests/test_flow_step.py:329` 和 `:373` 分别直调 sampler/forward hook | 从真实 batch 入口做固定 noise/sigma 的旧新路径对照，核对 prediction、target、梯度、aux、router 顺序、padding、train/eval。公共层持有步骤，族钩子保留扩展 |
| P4 Flow 边界 | **违反输入契约**。`library/training/flow_step.py:82` 检查 rank，缺 registry 通道/非空轴。`:114` 仅 CPU 检查 sigma 的有限性和范围 | 用 registry 验证 shape。分别校验 sigma 和模型 timestep，V100 加坏值回归并检查无逐步 D2H。保留 MOSS DiT 内部时间尺度和 loss 端消费语义 |
| P4/P5 latent | **只接入部分路径**。P5 后续差异的 `library/strategies/latent_cache.py:130` 检查策略编码输出，但接受任意 `ndim >= 4`。`:248` 读盘不校验数组形状。`scripts/preprocess/cache_latents.py:89` 使用另一个写入引擎 | CLI、策略写入与训练读盘使用同一几何校验。覆盖 4D、合法旧 singleton 5D、错误 batch/channel/stride/HW、坏 flip 和旧 sidecar。保留 Anima/Krea-2 的共享 latent 空间 |
| P4 策略和 TE | **违反依赖方向，生命周期待补证**。通用 Strategy 仍位于 `library/anima/text_strategies.py`，Qwen/Z-Image/Krea-2/MOSS 消费该路径。`library/strategies/text_cache.py:447` 现有 eager 会话刻意排除 Qwen meta/offload | 将基类移至中立模块，保留旧导出。通用层只编排会话，Qwen 插件拥有 meta hook 安装和卸载。补实际缓存入口的 lazy/eager、异常恢复及 MOSS 真实长度与量化 identity 回归 |
| P5 独立推理 | **Anima 违反阶段契约**。`library/inference/anima_legacy_runner.py:42` 向 `run_stage` 提供 `None`。`library/inference/generation.py:914` 先将 DiT 加载至目标设备，后在 `library/inference/text.py:95` 加载默认同设备的 TE。`tests/test_phase5_closure.py:21` 检查源码字符串，`:63` 替换 `generate` | 从单图生产入口用可追踪资源验证各 stage 的加载、运行和释放。以真实 Anima 权重复验驻留峰值与输出。批量、交互及需 DiT 预处理的 adapter 路径分开验收 |
| P5 插件 | **基本 fail-closed，能力隔离未达标**。`library/models/family_plugins.py:125` 把推理、缓存、预览、训练钩子绑在首次加载的全能力校验 | 按调用场景验证能力及返回值，注册表保留唯一身份与能力声明。覆盖缺少受支持能力、错误 runner identity、未声明能力及无关可选能力缺失 |
| WebUI/PP | **属于运行时与验收阻断**。队列存档把 9 个 PP 用例均标为 `ddp`，历史命令使用 2 个进程。14 个标名 DDP 的用例仅允许单卡。8 个队列 `done` 的 PP 报告均停在 `incomplete/setup`。`M-PP-01-corrected` 队列成功但执行 0 步 | 使用共同有效模式记录队列、历史和启动。报告接收共享事务的尝试、成功和跳步事件。续训断言每 rank 游标和实际步增量，身份不匹配保持拒绝。WebUI 补双卡 DDP 与 full-checkpoint 开关对照 |

实跑 CPU 定向套件均退出 `0`。P4 原树 67 通过、4 跳过（CUDA 或官方 MOSS 对照），P5 原树及后续差异 69 通过、10 跳过（双 V100 runner 原语）。这两组分别执行。P5 runner 硬件套件使用小型 CUDA 资源桩，真实模型驻留需要单独验证。

历史 WebUI 证据取自 `/tmp/p4-fullflow-summary.json`、`/tmp/p4-fullflow-evidence.jsonl` 及队列和训练历史元数据。23 项包含 21 项有实际训练步、1 项 checkpoint identity 正确拒绝、1 项零步误报成功。同一配置的 `M-PP-01b-resume` 有 5 步记录。损失和速度只报告运行观测，拟合结论需固定数据和随机流的对照。

## 复审与发布

- 在集成版本核对实际执行入口和依赖方向，并处理独立复审发现。
- 按设备记录验证。GPU 验收保留 FP16 训练和 MOSS NF4/BF16 TE 契约，经 WebUI 启动并关联任务证据。
- 发布前登记提交、目标分支与验证结果，核对远端和本地 `dev` 指向集成结果。
