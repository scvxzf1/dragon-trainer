# 通用层与族插件修复及复审表

状态：修复中，复审更新于 2026-11-20（L0 静态 + L1 CPU + GPU 真机验证：P2-2 L2 冒烟、P2-5 非 Anima 权重分段释放、P3-1 部分、门禁②③ MOSS PP 双卡 fp16 实跑）

基线为 `dev` 的 `7901d68a`。P4 原工作树以 `93aaeddc` 为基线，P5 原工作树以 `24efcc36` 为基线。开发和集成在独立工作树执行，当前集成分支 `codex/plugin-contract-integration`，HEAD 为 `b91dc479`。

本表在 2026-11-20 复核过一轮：原表把 P1-1/P1-2/P1-3 记为「进行中」，但对应修复已在 `c666ac47`、`b91dc479` 两个提交落地。下表的「状态」写的是**集成分支 `codex/plugin-contract-integration`** 的实际提交状态：P2/P3 七项已按依赖序落成 4 个原子提交（`05fe7375` P3-2/P3-3 → `b2093f66` P2-3/P3-1 → `a6b9c81a` P2-1/P2-2 → `d10f489a` P2-5，含 6 个未跟踪文件入库），本文件随第 5 条 docs 提交入库。

| 编号 | 修复项 | 合入验收 | 状态 |
| --- | --- | --- | --- |
| P0-1 | 集成 P4/P5 声明、实现和测试 | 保留 P4 Flow/cache 和 P5 独立推理/预览；同一树跑回归 | 已提交（`baf13d29`、`c2fd7909`、`2095158b`），待复审 |
| P1-1 | PP 精度报告阶段、更新计数与退出状态 | 共享事务事件映射到报告，显式区分尝试、提交和跳步 | 已提交（`c666ac47`），待复审；L3 运行证据已有（门禁②③通过，持久证据见 `docs/findings/pp-runtime-evidence-20261120.md`） |
| P1-2 | PP 续训出现零步成功 | 核查实际数据游标和目标步数，同数据集从保存点继续，非法零步具名拒绝 | 已提交（`c666ac47`），待复审；L3 运行证据已有（精确续训 step2→step3 + 两种具名拒绝实测，见 `docs/findings/pp-runtime-evidence-20261120.md`） |
| P1-3 | WebUI PP 被记录为 DDP | 队列和历史消费同一有效模式，保留旧记录兼容 | 已提交（`b91dc479`），待复审；L3 运行证据已有（5 队列项 / 2 条历史 `distributed_mode="pp"`，旧记录零改写，见 `docs/findings/pp-runtime-evidence-20261120.md`） |
| P2-1 | Anima 正式训练绕过共享 Flow | 通用层持有主流程，Anima 保留辅助目标、max-padded 文本及路由顺序 | 已提交（`a6b9c81a`），待复审 |
| P2-2 | sigma、模型 timestep、训练张量边界不完整 | 区分 sigma 与 timestep，检测 GPU 非法值，按 registry 检查 latent 通道与形状 | 已提交（`a6b9c81a`），待复审；GPU L2 坏值冒烟已执行（见复核记录） |
| P2-3 | Latent cache 几何和 flip 未完整验证 | 验证 batch、通道、步长、空间尺寸、普通及翻转形状，保留旧缓存兼容 | 已提交（`b2093f66`），待复审 |
| P2-4 | 文档与已合入版本不一致 | 标记固定代码与执行证据，记录已验证边界和完整提交清单 | 本文件即该项的修订产物，待复审；SHA 回填与复核记录见本文件（随本提交入库） |
| P2-5 | Anima 独立推理以空资源包装 legacy 整段 | 保留 Anima adapter/采样语义，生产入口显式分离 TE 编码、DiT 条件化/采样、VAE 解码，分别验证释放与异常 | 已提交（`d10f489a`），待复审；L2（非 Anima 权重）：krea2_raw 族真机分段释放已验（见复核记录），Anima 真实权重驻留仍未验证（`models/diffusion_models` 缺 Anima DiT safetensors） |
| P3-1 | TE 会话恢复和生命周期边界 | eager 与族管理的 offload 分工明确，验证惰性加载成功、异常和卸载 | 已提交（`b2093f66`），待复审；L2（部分）：会话与策略语义已到真机（异常原样传播 + empty_cache 1 次；cpu/auto 两种 policy PASS，第 4 项经 stand-in 模块走真实 `loaded_text_encoder_for_cache`，不等于真 4B 显存数字）；真 8B 权重加载缺资产 BLOCKED（预期拒绝，见复核记录），不在本审计范围修 loader |
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
| WebUI/PP | **属于运行时与验收阻断**。队列存档把 9 个 PP 用例均标为 `ddp`，历史命令使用 2 个进程。14 个标名 DDP 的用例仅允许单卡。8 个队列 `done` 的 PP 报告均停在 `incomplete/setup`。`M-PP-01-corrected` 队列成功但执行 0 步 | **已提交，CPU 侧验证，运行证据待重采**。`b91dc479` 改 `web/services/training/{distributed,queue_enqueue,queue_state,launcher_start,launcher_job,launcher_runtime,history_store,history_ops,history_config_chips,runtime_common}.py`，新增 `tests/test_web_training_distributed_mode.py`（7 项，本机 CPU 通过）。PP 报告侧：`c666ac47` 让 `library/training/precision_report.py` 区分 `attempted`/`effective`/`skipped` 并接收 PP 事务的 scale（`pp_runtime_ready`/`pp_optimizer`/`resumed(pp_scale=)`）；续训逻辑落到新文件 `library/training/pp/trainer_resume.py`（分 rank `trainer_state_rank{r}.pt`、全 rank 一致才 apply、旧 checkpoint 缺 trainer-state 默认具名拒绝，`MOSS_PP_RESUME_WEIGHTS_ONLY=1` 才允许重量续训）。**原表引用的队列/历史数字为历史环境口径且证据已丢失，本轮已以新实跑替代**（见下） |

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

- **GPU 验收覆盖面（更新后）**。P2-2 CUDA `_assert_async` 分支已完成 L2 坏值冒烟（见复核记录）；门禁②③已以真实 MOSS-Image PP 双 V100 fp16 实跑通过（队列/历史模式、报告计数、精确续训、具名拒绝，见 `docs/findings/pp-runtime-evidence-20261120.md`）。**仍未验证**：fp16 GradScaler 真实溢出跳步（本轮全 effective，未观察到一次 skip）、真实 Anima 权重 GPU 验收、MOSS NF4/BF16 TE 契约真机、收敛性/长训练。
- ~~P4 原树 67 通过 / 4 跳过、P5 原树 69 通过 / 10 跳过 未重跑~~ **已关闭（t6）**：现 HEAD 实测 P4 树 `93aaeddc` 102 passed / 0 skipped / 0 failed、P5 树 `24efcc36` 147 passed / 0 skipped / 0 failed。历史两组数字属**历史环境口径，与现 HEAD 不可直接对比**（两树是更早提交状态、文件集与依赖不同，当时依赖缺失导致 skip），不是"数字错误"，现值也不"优于历史"（口径不同）。
- ~~历史 WebUI 证据已丢失~~ **已关闭（t10）**：门禁②③以极小 MOSS PP 队列 + 续训重采，持久证据落 `docs/findings/pp-runtime-evidence-20261120.md`；原表引用的 `/tmp/p4-fullflow-*` 旧证据仍不存在，其结论作废，以新证据为准。
- **P2-5 部分（非 Anima 权重已验，Anima 缺口保留）**：CPU 行为断言 + 非 Anima 权重（krea2_raw 族）真机 TE→DiT→VAE 分段释放已验（见复核记录）；**真实 Anima 权重的驻留峰值仍未验证**（`models/diffusion_models` 只有 krea2_raw 三件，缺 Anima DiT safetensors），批量与交互路径、需 DiT 预处理的 adapter 路径同样未实机验证。
- **P3-1 部分（语义已到真机，权重资产缺）**：会话与策略语义真机验证 PASS（异常原样传播 + empty_cache 调 1 次；cpu/auto 两种 policy 的设备放置、bf16 保持、退出零残留、peak_reserved 回落基线；第 4 项用 stand-in 模块走真实 `loaded_text_encoder_for_cache`，**不等于真 4B 显存数字**）。真 8B 权重加载 **BLOCKED 缺资产**：本机无 Qwen3-VL **8B** ComfyUI 风格单文件 TE 与匹配 tokenizer（`docs/multi_model_support.md:77` 要求口径）；现有 `qwen3vl_4b_bf16.safetensors` 型号（4B vs 8B）与键风格（HF 原生 `model.language_model.*` vs 期望 ComfyUI `model.*` 再映射，`library/models/qwen_image_2_1/weights.py:105-108`）双重不符，`weights.py:128` 的 `QwenImage21CheckpointError` 属**预期拒绝**（严格集合校验 + `text_encoder_runtime.py:82-88` meta-guard 是既有契约）。需单独立项补齐权重，不在本审计范围内修 loader。
- ~~逐提交隔离回放未做~~ **已关闭（t7）**：C2 `b2093f66`/C3 `a6b9c81a`/C4 `d10f489a` 可独立回放绿（86/104/156 passed）；C1 `05fe7375` **非 bisect-safe**（见复核记录）。
- ~~L2 冒烟脚本未入库~~ **已关闭（t5，`4fa96cb5`）**：`tests/test_flow_step_assert_async.py`（FLOW_STEP_CUDA 门控 + 子进程隔离）已入库。
- **fp16 GradScaler 实跑未验**：本仓 fp16 训练走 GradScaler 跳步，P2-2 只证明"坏值被设备端断言拦住、合法 inf 输出不被误拦"，门禁②③实跑全 effective，未在真实训练步中观察一次真实的溢出跳步。
- **PP 语义残余分支未验（t10 保留清单）**：`weights_only` 放行路径、`pp_resume_no_work` 分支、真实 overflow skip、收敛性/长训练均未验证。
- **待复核改进项（本轮新发现，未修）**：`library/training/train_session.py:143-144` 在 `seed=None` 时静默 `random.randint`，而 PP run identity = `sha256(family|seed|output_name)` → 跨进程续训要求显式固定 seed，未钉 seed 会静默随机化并导致 identity 不匹配（本轮实测撞到 `pp_checkpoint_identity_mismatch`）。属可用性缺陷，本审计未修，列为待复核改进项。

缺漏汇总（2026-11-20 第二轮更新，不含已关闭项）：影响交付 1 项（真实 Anima 权重 GPU 验收与显存驻留——含 fp16 溢出跳步真机观察，缺 Anima DiT safetensors 资产）；证据完整性 2 项（P3-1 缺 Qwen3-VL 8B ComfyUI 风格权重资产，需单独立项补齐；P2-5 批量与 adapter 路径、收敛性/长训练与 PP 残余分支）；工程卫生 2 项（vLLM 恢复、旧工作树 P5 遗留 13 个未提交改动按用户选择 A 暂不动）；待复核改进 1 项（train_session seed=None 静默随机化 vs PP identity）。

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
- **未覆盖边界**：本冒烟只覆盖 P2-2 的 `_check_flow_values` CUDA 分支；真实权重训练、fp16 GradScaler 实际溢出跳步、PP 两卡通信（后者已由 t10 门禁②③覆盖）见未验证边界。

### gpu-gate 与 CUDA 枚序（工程事实，避免误诊）

- `gpu-gate.sh` **存在且可用**（`/mnt/SD_Dataset/训练器相关/dsh-trainer-preset/gpu-gate.sh`，权限 `-rwx--x--x`，不在 PATH）。裸命令名调用会报 "command not found"，正确姿势是 `bash <绝对路径>` —— 某成员首跑时的 "gate 不可用" 属误诊；其用例本身不依赖 gate（门控是 `FLOW_STEP_CUDA=1` + `cuda.is_available()`），**GPU 实跑的 gate 通路由 captain 在同一时间窗补跑确认**（`bash …/gpu-gate.sh --count 1 --need 8000 --json` → decision=run、CVD="1"、rc=0），成员首跑时以 nvidia-smi 判空闲（当时两卡各 11–14 MiB）。
- **CUDA 枚序 ≠ nvidia-smi 索引**：自证命令 `CUDA_VISIBLE_DEVICES=2 python -c 'import torch;print(torch.cuda.get_device_name(0))'` → **NVIDIA GeForce GTX 1050**。默认 FASTEST_FIRST 下 0/1=两张 V100、2=1050；`CUDA_DEVICE_ORDER=PCI_BUS_ID` 才与 nvidia-smi 对齐。`gpu-gate.sh` **未设置 `CUDA_DEVICE_ORDER`**，其 `cuda_visible_devices` 是 nvidia-smi 索引口径：单卡 "1" 碰巧正确，多卡 "1,2" 在默认枚序下会把 1050 带进来（只记事实，preset 脚本属跨仓资产不在本审计范围修改）。WebUI 训练子进程自身走 `CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=<nvidia-smi 索引>`（`web/services/training/gpu.py:164-165`），不受此坑影响。

### P4/P5 原树套件复跑（t6）

现 HEAD 实测（解释器 = 集成树 .venv，Python 3.13.9 / pytest 9.1.1，cwd 指向被测树，`PYTHONDONTWRITEBYTECODE=1`，零新增 `__pycache__`）：P4 树 `93aaeddc` 8 文件全 rc=0 → **102 passed / 0 skipped / 0 failed**；P5 树 `24efcc36`（13 个未提交改动按用户决定 A 未动）13 文件全 rc=0 → **147 passed / 0 skipped / 0 failed**。历史 "P4 67/4、P5 69/10" 两组均**未复现**，归因为历史环境口径（更早提交状态、文件集不同、当时依赖缺失 skip），与现 HEAD 不可直接对比。

### 逐提交隔离回放（t7，bisect 可用性）

- C1 `05fe7375`：**非 bisect-safe**。收集期 `tests/test_family_inference_contracts.py:9` `ImportError: cannot import name 'validate_family_inference_options'`（该符号由 C4 `d10f489a` 引入）；排除该文件后 `test_family_inference_plugins` 27 passed + 1 failed（`test_qwen_preview_hook_is_required_only_when_requested`，C1 版 qwen plugin 尚无 `cache_preview_text`，该方法由 C2 `b2093f66` 引入）。
- C2 `b2093f66` 86 passed、C3 `a6b9c81a` 104 passed、C4 `d10f489a` 156 passed（C2/C3 回放时 contracts 文件继承 C1 坏版本无法收集，属传导非自身问题）。
- **bisect 请从 `b2093f66` 起**，不要用 `05fe7375` 作为可判定边界。根因：C1 把"验收测试文件整文件随组入库"，其中混入了依赖后续提交 API 的新断言（跨提交测试耦合）。

### P2-2 L2 冒烟落成正式用例（t5，`4fa96cb5`）

新增 `tests/test_flow_step_assert_async.py`（FLOW_STEP_CUDA 门控 + 子进程隔离，6 用例）：无门控 6 skipped rc=0；CVD=1 实跑 **6 passed（44.77s）**，`test_flow_step.py` 仍 18 passed。gate 通路由 captain 同一时间窗补跑确认（见上），仓库从此有持久复现入口。

### P2-5 真机分段释放（t9，非 Anima 权重）

结论：TE → DiT → VAE 分段加载与释放顺序**真机成立**，链路 `family_runtime.prepare_inference_family` → `krea2_raw/inference_runner.{prepare_krea2_text, load_krea2_dit_for_inference, generate_krea2}` → `qwen_vae.load_vae/decode_latent`；权重族 = krea2_raw（nf4_self_contained DiT + qwen3vl_4b TE + qwen VAE）。逐阶段（MiB，allocated 驻留 / 峰值 / reserved）：①TE 峰值 8947.9、释放后 reserved 回落 98.0；②DiT NF4 驻留 6394.8、采样 2 步峰值 7608.2、释放后 reserved 回落 102.0；③VAE 驻留 316.6、decode 512×512 峰值 2478.5、allocated 回 73.3。产出 sanity：latent `[1,16,1,64,64]` → image `[3,512,512]`；跑后两卡 11 MiB 零残留。如实记录：MOSS 推理在 `family_runtime` fail-closed（仅训练预览），无法做 MOSS 版；sm70 BF16 外层门禁明确拒绝 flash（"FP16 provider bridge 可能产生非有限 Q/K 梯度"），本次只验 torch 路径——门禁拒绝本身是正确行为，不是缺陷。

### 门禁②③ 真机实跑（t10）

摘要（完整命令、退出码与逐字段数字见 `docs/findings/pp-runtime-evidence-20261120.md`，随本提交入库）：

- **门禁②（模式记录与报告计数）**：队列 item 与 history meta 的 `distributed_mode` 全部 = `"pp"`（5 队列项 / 2 条成功历史），旧记录零改写。首轮 2 步 `updates={attempted:2, effective:2, skipped:0, total_global_step:2}`、status=completed；续训 1 步 `{attempted:1, effective:1, skipped:0, total_global_step:3}`（含继承的 2）。fp16 GradScaler 初值/终值/跨进程恢复均 65536.0。MOSS NF4 缓存 + 训练只读缓存不加载 TE，BF16-only/frozen 契约未触碰。双卡峰值 7.5 / 6.5 GiB。
- **门禁③（具名拒绝与精确续训）**：具名拒绝实测两种 —— stateless bundle 默认路径双 rank 对称 `pp_checkpoint_incomplete`（权重复用必须显式 `pipeline_parallel_resume_weights_only`）与 `pp_checkpoint_identity_mismatch`；钉 `seed=3747612563` 后精确续训成功：restore step2 → 恰 +1 步（step3/3，effective 3），per-rank 游标 `epoch1/batch1 → epoch2/batch1` 一致。
- **保留未验证**：`weights_only` 放行路径、`pp_resume_no_work` 分支、真实 overflow skip（本轮全 effective）、收敛性/长训练。
- **新发现（待复核改进项，未修）**：`library/training/train_session.py:143-144` `seed=None` 静默随机化 vs PP identity `sha256(family|seed|output_name)`，见未验证边界。

## 复审与发布

- 在集成版本核对实际执行入口和依赖方向，并处理独立复审发现。
- 按设备记录验证。GPU 验收保留 FP16 训练和 MOSS NF4/BF16 TE 契约，经 WebUI 启动并关联任务证据。
- 发布前登记提交、目标分支与验证结果，核对远端和本地 `dev` 指向集成结果。
- **测试改动必须按依赖拆到对应提交**（t7 实践约束），不得整文件随最早一组入库，否则后续任何 bisect 都会在收集期失败（C1 `05fe7375` 即此问题的实例）。

发布门禁当前**未通过**，剩余阻塞项：① ~~提交工作区差异（含 6 个未跟踪文件）~~ 已完成（`05fe7375`/`b2093f66`/`a6b9c81a`/`d10f489a` + docs 提交，未 push）；② ~~重采 WebUI/PP 运行证据~~ 已完成（t10 门禁②③真机实跑，持久证据 `docs/findings/pp-runtime-evidence-20261120.md`，含保留未验证清单：`weights_only` 放行路径、`pp_resume_no_work`、真实 overflow skip、收敛性/长训练）；③ 部分完成——PP 两卡通信与 fp16 训练已由 t10 覆盖（真实 MOSS PP 双 V100），**真实 Anima 权重 GPU 验收仍未完成**（缺 Anima DiT safetensors 资产，含 fp16 溢出跳步真机观察）。
