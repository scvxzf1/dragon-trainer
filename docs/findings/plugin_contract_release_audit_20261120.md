# 通用层与族插件修复及复审表

状态：修复中，复审更新于 2026-11-20（L0 静态 + L1 CPU + P2-2 GPU L2 坏值冒烟；其余 GPU 验收未做）

基线为 `dev` 的 `7901d68a`。P4 原工作树以 `93aaeddc` 为基线，P5 原工作树以 `24efcc36` 为基线。开发和集成在独立工作树执行，当前集成分支 `codex/plugin-contract-integration`，HEAD 为 `b91dc479`。

本表在 2026-11-20 复核过一轮：原表把 P1-1/P1-2/P1-3 记为「进行中」，但对应修复已在 `c666ac47`、`b91dc479` 两个提交落地。下表的「状态」写的是**集成分支 `codex/plugin-contract-integration`** 的实际提交状态：P2/P3 七项已按依赖序落成 4 个原子提交（`05fe7375` P3-2/P3-3 → `b2093f66` P2-3/P3-1 → `a6b9c81a` P2-1/P2-2 → `d10f489a` P2-5，含 6 个未跟踪文件入库），本文件随第 5 条 docs 提交入库。

| 编号 | 修复项 | 合入验收 | 状态 |
| --- | --- | --- | --- |
| P0-1 | 集成 P4/P5 声明、实现和测试 | 保留 P4 Flow/cache 和 P5 独立推理/预览；同一树跑回归 | 已提交（`baf13d29`、`c2fd7909`、`2095158b`），待复审 |
| P1-1 | PP 精度报告阶段、更新计数与退出状态 | 共享事务事件映射到报告，显式区分尝试、提交和跳步 | 已提交（`c666ac47`），待复审 |
| P1-2 | PP 续训出现零步成功 | 核查实际数据游标和目标步数，同数据集从保存点继续，非法零步具名拒绝 | 已提交（`c666ac47`），待复审 |
| P1-3 | WebUI PP 被记录为 DDP | 队列和历史消费同一有效模式，保留旧记录兼容 | 已提交（`b91dc479`），待复审 |
| P2-1 | Anima 正式训练绕过共享 Flow | 通用层持有主流程，Anima 保留辅助目标、max-padded 文本及路由顺序 | 已提交（`a6b9c81a`），待复审 |
| P2-2 | sigma、模型 timestep、训练张量边界不完整 | 区分 sigma 与 timestep，检测 GPU 非法值，按 registry 检查 latent 通道与形状 | 已提交（`a6b9c81a`），待复审；GPU L2 坏值冒烟已执行（见复核记录） |
| P2-3 | Latent cache 几何和 flip 未完整验证 | 验证 batch、通道、步长、空间尺寸、普通及翻转形状，保留旧缓存兼容 | 已提交（`b2093f66`），待复审 |
| P2-4 | 文档与已合入版本不一致 | 标记固定代码与执行证据，记录已验证边界和完整提交清单 | 本文件即该项的修订产物，待复审；SHA 回填与复核记录见本文件（随本提交入库） |
| P2-5 | Anima 独立推理以空资源包装 legacy 整段 | 保留 Anima adapter/采样语义，生产入口显式分离 TE 编码、DiT 条件化/采样、VAE 解码，分别验证释放与异常 | 已提交（`d10f489a`），待复审 |
| P3-1 | TE 会话恢复和生命周期边界 | eager 与族管理的 offload 分工明确，验证惰性加载成功、异常和卸载 | 已提交（`b2093f66`），待复审 |
| P3-2 | 通用策略基类位于 Anima | 中立基类单一来源，Anima 旧路径保留兼容导出 | 已提交（`05fe7375`），待复审 |
| P3-3 | 插件协议粒度过粗 | 能力分开验证，缺失能力或返回不满足边界时具名拒绝 | 已提交（`05fe7375`），待复审 |

## 复合审计

当前判定为部分符合，发布门禁未通过。注册表惰性分派、未知族拒绝和各族缓存布局属于已接入的契约。下表「原判定」取自 `7901d68a` 基线，保留作为审计痕迹；「当前复核」为 2026-11-20 工作区事实。

| 路径 | 原判定与证据 | 当前复核（2026-11-20） |
| --- | --- | --- |
| P4 正式训练 | **违反共享编排契约**。`library/training/batch_step.py:59` 经 `library/anima/plugin.py:38` 直接进 `library/anima/noise_target.py:37`。Anima 自行执行噪声、采样、路由与前向，`tests/test_flow_step.py:329` 和 `:373` 分别直调 sampler/forward hook | **工作区已改为通用层持步**。`batch_step.py:61` 以 `capability="flow"` 取插件；`library/anima/noise_target.py:45-51` 直接调 `run_family_flow_step`，`noise_target.py` 只留钩子实现（`sample_anima_sigmas`、`prepare_anima_flow_inputs`、`prepare_anima_conditions`、`forward_anima_velocity`、`observe_anima_flow`、`adjust_anima_weighting`）。辅助目标（prior preservation `noise_target.py:256/282/297/319`、VR reference `:359`）仍在族侧，经 `library/anima/plugin.py:65` 的 `observe_flow_step` 挂进公共步。路由顺序由 `route_before_conditions = True`（`plugin.py:43`）+ `library/training/flow_step.py:171-184` 的早/晚路由分支表达。新测试 `tests/test_flow_step.py:408` 从真实 batch 入口 `_get_noise_pred_and_target` 固定 noise/sigma 断言事件序 `shared_flow → sampler → prime → sigma_router → text_router → forward → extras`，train/eval 双参数各一遍 |
| P4 Flow 边界 | **违反输入契约**。`library/training/flow_step.py:82` 检查 rank，缺 registry 通道/非空轴。`:114` 仅 CPU 检查 sigma 的有限性和范围 | **工作区已补齐**。`flow_step.py:82-102` 增非空轴与按 registry 的 latent 通道校验；`_validate_flow_inputs`（`:230-252`）校验 latents/noise/noisy 三者的 shape 与 device 不被钩子改动；sigma 与模型 timestep 分开校验（`:116-140`）；`_check_flow_values`（`:223-227`）在 CUDA 上走 `torch._assert_async`，不再隐式逐步 D2H；loss weight 形状/设备也显式拒绝（`:208-221`） |
| P4/P5 latent | **只接入部分路径**。P5 后续差异的 `library/strategies/latent_cache.py:130` 检查策略编码输出，但接受任意 `ndim >= 4`。`:248` 读盘不校验数组形状。`scripts/preprocess/cache_latents.py:89` 使用另一个写入引擎 | **工作区已统一到单一校验**。新增 `library/strategies/latent_geometry.py`（`validate_latent_shape` 按 `LatentSpaceSpec` 的 `vae_spatial_compression`/`latent_channels` 校验，接受 4D 与合法 singleton-time 5D；`validate_latent_batch_geometry` 校验 batch 与逐 bucket 几何）。三条路径都接上：策略编码 `latent_cache.py` 的 `_validate_encoded_latents`（原 `ndim >= 4` 放宽判断已删除）、读盘侧对普通与 `latents_flipped` 分别校验并要求两者同形、预处理写入 `library/preprocess/latents.py:361-367` 由 `scripts/preprocess/cache_latents.py:105` 传入 `latent_spec`。旧缓存不硬失败：读盘侧几何校验放在缓存有效性判定里（`library/strategies/latent_cache.py:106-161`），不合规时 `logger.warning` 记 `Invalid latent cache geometry` 并返回 `False`，该条按缓存缺失回落到重新编码，不会中断训练；`flip_aug` 下翻转形状与普通形状不一致同样判无效（`:153-154`）。回归 `tests/test_latent_cache_geometry.py` 17 项 |
| P4 策略和 TE | **违反依赖方向，生命周期待补证**。通用 Strategy 仍位于 `library/anima/text_strategies.py`，Qwen/Z-Image/Krea-2/MOSS 消费该路径。`library/strategies/text_cache.py:447` 现有 eager 会话刻意排除 Qwen meta/offload | **工作区已修依赖方向，生命周期待真机**。中立基类落到 `library/strategies/text.py`（`TokenizeStrategy`/`TextEncodingStrategy`/`TextEncoderOutputsCachingStrategy`），`library/anima/text_strategies.py` 缩为 15 行兼容再导出，`library/strategies/__init__.py` 惰性导出三类型。Qwen 的 meta hook 改由插件拥有：`library/models/qwen_image_2_1/plugin.py` 新增 `cache_preview_text`，用 `text_encoder_cache_session(..., manage=loaded_text_encoder_for_cache)` 把安装/卸载交给族；会话本身新增 `load=`（惰性加载，失败时清 CUDA 缓存再抛）与 `manage=`（族托管，`library/strategies/text_cache.py:447-472`）。Qwen 专用分支从 `library/training/text_encoder_cache.py` 的调用点移除，改走 `capability="qwen_preview_text"` |
| P5 独立推理 | **Anima 违反阶段契约**。`library/inference/anima_legacy_runner.py:42` 向 `run_stage` 提供 `None`。`library/inference/generation.py:914` 先将 DiT 加载至目标设备，后在 `library/inference/text.py:95` 加载默认同设备的 TE。`tests/test_phase5_closure.py:21` 检查源码字符串，`:63` 替换 `generate` | **工作区已按阶段拆分**。`anima_legacy_runner.py` 改为显式三段：TE 编码走独立 `FamilyInferenceRunner.run_text`（`text_encoder_cpu` 时落 CPU）、EasyControl 参考在独立 `reference_vae` 阶段编码后 `.cpu()` 持有、DiT 在 `dit` 阶段加载并在 `generate(..., shared_models=..., precomputed_text_data=...)` 中复用。参考条件封装到新文件 `library/anima/inference_references.py`。`supports_reference_conditioning` 由 `False` 改 `True` 并接 `validate_family_inference_options`。测试仍是 `tests/test_phase5_closure.py`（该文件在本工作树为**未跟踪新文件**，352 行 7 项）：旧 `:63 test_anima_legacy_generation_is_wrapped_by_shared_stage`（monkeypatch 掉 `generate` 后只断言 `["generate", "allocator"]`）已换成事件序行为断言 —— `test_anima_single_prompt_releases_text_encoder_before_dit`（`:63`）、`test_anima_easycontrol_reference_vae_finishes_before_dit_load`（`:122`，断言 `vae:release` 先于 `dit:load`、参考 latent 停在 CPU）、`test_z_image_runner_uses_shared_text_then_dit_stages`（`:188`）、`test_z_image_runner_uses_shared_vae_stage`（`:273`）。`:21` 的源码字符串守卫（断言每个 runner 源码含 `FamilyInferenceRunner`）保留未动，属静态检查，不能替代阶段释放验证 |
| P5 插件 | **基本 fail-closed，能力隔离未达标**。`library/models/family_plugins.py:125` 把推理、缓存、预览、训练钩子绑在首次加载的全能力校验 | **工作区已按能力分组**。`family_plugins.py:146` 定义 `_CAPABILITY_METHODS`：`cache`（6 个构建/编码方法）、`flow`（4 个）、`inference`（`build_inference_runner`）、`preview`（2 个）、`qwen_preview_text`（1 个）；`get_family_plugin(family, capability=...)` 只校验该组，缺方法抛 `FamilyPluginContractError`，未知能力抛 `ValueError`。全部入口已按场景传参：`batch_step.py:61`/`trainer_network_mixin.py:181`/各族 `family.py` → `flow`，`scripts/tasks/preprocess.py:681`、`anima_strategies.py:63/94/98/111` → `cache`，`library/inference/family_runtime.py:129` → `inference`，`preview_live_decode.py:123/138` → `preview`，`text_encoder_cache.py:63-67` → Qwen 走 `qwen_preview_text`、其余族走 `cache`。缺能力路径的回归在 `tests/test_family_inference_plugins.py`、`tests/test_model_family_registry.py` |
| WebUI/PP | **属于运行时与验收阻断**。队列存档把 9 个 PP 用例均标为 `ddp`，历史命令使用 2 个进程。14 个标名 DDP 的用例仅允许单卡。8 个队列 `done` 的 PP 报告均停在 `incomplete/setup`。`M-PP-01-corrected` 队列成功但执行 0 步 | **已提交，CPU 侧验证，运行证据待重采**。`b91dc479` 改 `web/services/training/{distributed,queue_enqueue,queue_state,launcher_start,launcher_job,launcher_runtime,history_store,history_ops,history_config_chips,runtime_common}.py`，新增 `tests/test_web_training_distributed_mode.py`（7 项，本机 CPU 通过）。PP 报告侧：`c666ac47` 让 `library/training/precision_report.py` 区分 `attempted`/`effective`/`skipped` 并接收 PP 事务的 scale（`pp_runtime_ready`/`pp_optimizer`/`resumed(pp_scale=)`）；续训逻辑落到新文件 `library/training/pp/trainer_resume.py`（分 rank `trainer_state_rank{r}.pt`、全 rank 一致才 apply、旧 checkpoint 缺 trainer-state 默认具名拒绝，`MOSS_PP_RESUME_WEIGHTS_ONLY=1` 才允许重量续训）。**原表引用的队列/历史数字本轮无法复核**（见下） |

## 实跑证据（2026-11-20，L1 CPU）

在集成工作树 `codex/plugin-contract-integration`（HEAD `b91dc479` + 未提交工作区改动）用 `.venv/bin/python -m pytest` 逐文件跑定向套件。除 `test_moss_image_pp_trainer_checkpoint` 首次 60 s 超时（`rc=124`）后以 300 s 预算重跑通过外，其余每项均在 60 s 内退出 `0`；合计 **197 通过、0 失败、0 跳过**：

| 套件 | 结果 |
| --- | --- |
| `tests/test_flow_step.py` | 18 passed |
| `tests/test_latent_cache_geometry.py` | 17 passed |
| `tests/test_phase5_closure.py` | 7 passed |
| `tests/test_family_inference_contracts.py` | 37 passed |
| `tests/test_family_inference_plugins.py` | 11 passed |
| `tests/test_model_family_registry.py` | 17 passed |
| `tests/test_text_cache_session.py` | 6 passed |
| `tests/test_qwen_text_encoder_runtime.py` | 15 passed |
| `tests/test_cache_identity.py` | 20 passed |
| `tests/test_moss_image_inference_contract.py` | 8 passed |
| `tests/test_web_training_distributed_mode.py` | 7 passed |
| `tests/test_moss_image_pp_trainer.py` | 15 passed |
| `tests/test_moss_image_pp_trainer_checkpoint.py` | 6 passed（103.67s，超 60s 预算，需放宽超时再跑） |
| `tests/test_documentation_integrity.py` | 13 passed |

## 未验证边界（诚实清单）

- **无 GPU 验收（除 P2-2 L2 冒烟）**。本表其余结论为 L0 静态 + L1 CPU；P2-2 的 CUDA `_assert_async` 分支已完成 L2 坏值冒烟（见复核记录），涉及精度/dtype、显存、fp16 GradScaler 实跑、PP 两卡通信的改动（P1-1/P1-2、MOSS NF4/BF16 TE 契约）**未做 GPU 数值验证**。
- **P4 原树 67 通过 / 4 跳过、P5 原树 69 通过 / 10 跳过** 这两组数字本轮未重跑，只复核了集成树。
- **历史 WebUI 证据已丢失**：原表引用的 `/tmp/p4-fullflow-summary.json`、`/tmp/p4-fullflow-evidence.jsonl` 在本机已不存在（`ls` 退出码 2）；仓库内也没有 `configs/web-training-queue`。「9 个 PP 用例标 `ddp`」「14 个 DDP 用例仅单卡」「8 个 PP 报告停在 `incomplete/setup`」「`M-PP-01-corrected` 0 步」「`M-PP-01b-resume` 5 步」「23 项含 21 项有实际训练步」等结论**当前无证据支撑**，需重新采集后才能作为发布依据。
- **P2-5 只到单元/CPU 级**：分段释放的行为断言通过，但真实 Anima 权重的驻留峰值、批量与交互路径、需 DiT 预处理的 adapter 路径均未实机验证。
- **P3-1 只到 CPU 级**：`manage=` 托管路径在 Qwen 分支上以 mock 验证，真实 Qwen meta/offload 安装与卸载、异常恢复未实机验证。
- **工作区改动未提交**：P2-1/P2-2/P2-3/P2-5/P3-1/P3-2/P3-3 的实现与测试差异共 51 个源码/测试文件（另有 5 个 `docs/` 文件，含本文件）+ 6 个未跟踪文件（`library/anima/inference_references.py`、`library/strategies/latent_geometry.py`、`library/strategies/text.py`、`tests/test_latent_cache_geometry.py`、`tests/test_moss_image_inference_contract.py`、`tests/test_phase5_closure.py`）。未提交即等于未入基线，未跟踪文件尤其容易在切分支/清理时丢失。
- **GPU 数值验证**：P2-2 的 CUDA `_assert_async` 分支已完成 L2 坏值冒烟（独立子进程、gpu-gate 放行），见下方「复核记录」；涉及精度/dtype、真实显存与 fp16 GradScaler、PP 两卡通信的其余改动（P1-1/P1-2、MOSS NF4/BF16 TE 契约）仍**未做 GPU 数值验证**。

## 复核记录（2026-11-20）

### `_check_flow_values` 作用域核验（只读审计，A1）

结论：不误拦 fp16 模型输出/梯度。`_check_flow_values` 仅作用于**输入侧**张量；模型输出与 loss weighting 均不做 isfinite 检查，fp16 溢出（含 GradScaler 溢出跳步）不会被误拦。

| 行号 | 张量 | 输入侧 / 输出侧 | 有限性检查 | 是否误拦 fp16 输出/梯度 |
| --- | --- | --- | --- | --- |
| `library/training/flow_step.py:131` | sigmas（钩子返回） | 输入侧 | isfinite + [0,1] 范围 | 否（合法 fp16 sigma 不受影响） |
| `library/training/flow_step.py:143` / `:167` | timesteps（模型 timestep） | 输入侧 | isfinite（逐分支） | 否 |
| `library/training/flow_step.py:186-193` | prediction | 输出侧 | 仅 shape，无 isfinite | 否 |
| `library/training/flow_step.py:208-219` | loss weighting（由 sigma 派生） | 输入侧 | 仅 shape/device，无 isfinite | 否（不拦 GradScaler 溢出跳步） |
| `library/training/flow_step.py:225` | CUDA 分支 | 输入侧 | `torch._assert_async`（设备端） | 否；注意延迟显形与进程级毒化语义（见下） |

CUDA 语义备注：CUDA 分支走 `torch._assert_async`，坏值不立即抛 Python 异常，而是在后续 synchronize/CUDA 调用时以 `AcceleratorError`/device-side assert 显形，且会毒化整个 CUDA 上下文 —— 因此 GPU 用例必须独立子进程隔离并在用例末显式 synchronize（冒烟已按此执行）。

### GPU L2 坏值冒烟（P2-2，独立子进程隔离）

- **gpu-gate 放行**：decision=run、`CUDA_VISIBLE_DEVICES="1"`（V100），未触碰 GTX 1050 与 card0，gate 退出码 0。
- **执行方式**：6 个用例各自独立子进程，真实走 `run_family_flow_step` → `_check_flow_values` CUDA 分支（`torch._assert_async`）；脚本为临时文件（`/tmp/t3_smoke.py`，会话产物，不入库），仓库零改动。
- **结果**：

| 用例 | 注入值 | 结果 |
| --- | --- | --- |
| sigma nan | sigma=nan | rc=1，device-side assert：`family sigma hook must return finite sigmas in [0,1]` |
| sigma inf | sigma=inf | rc=1，同上 |
| sigma 越界 | sigma>1 | rc=1，同上 |
| timestep inf | timestep=inf | rc=1，device-side assert：`family model timestep must be finite` |
| good（正例） | 全合法 | rc=0，峰值显存 0.01 MiB |
| good_pred_fp16_overflow（关键旁证） | prediction 含 inf（模拟 fp16 溢出/GradScaler） | rc=0 正常通过，实证 `_check_flow_values` 不误拦模型输出 |

- **延迟显形与进程毒化**：4 个坏值用例的 `AcceleratorError` 均由用例末 synchronize 触发，验证 A1 描述的设备端断言延迟显形与进程级毒化语义。
- **未覆盖边界**：本冒烟只覆盖 P2-2 的 `_check_flow_values` CUDA 分支；真实权重训练、fp16 GradScaler 实际溢出跳步、PP 两卡通信仍未做 GPU 验收。

## 复审与发布

- 在集成版本核对实际执行入口和依赖方向，并处理独立复审发现。
- 按设备记录验证。GPU 验收保留 FP16 训练和 MOSS NF4/BF16 TE 契约，经 WebUI 启动并关联任务证据。
- 发布前登记提交、目标分支与验证结果，核对远端和本地 `dev` 指向集成结果。

发布门禁当前**未通过**，剩余阻塞项按顺序为：① ~~提交工作区差异（含 6 个未跟踪文件）~~ 已完成（`05fe7375`/`b2093f66`/`a6b9c81a`/`d10f489a` + 本 docs 提交，未 push）；② 重采 WebUI/PP 运行证据；③ 真实 Anima 权重 GPU 验收与 PP 两卡通信验证（P2-2 L2 坏值冒烟已过，不在此列）。
