# 后端多模型兼容审计

状态：历史审计快照（后续加固已记录）

日期：2026-08-10

适用版本：2026-08-10 的 `krea2-migration` 工作树（审计时 `HEAD=df8ff775`，工作树包含用户未提交改动）；当前能力以 [`../multi_model_support.md`](../multi_model_support.md) 和实时源码为准

范围：基础模型 family 的配置、预处理、训练、LoRA/checkpoint、推理、Web/API 和测试边界。
这里的“多模型”指 `model_family`，不等同于 LoRA / LoKr / Hydra 等 adapter family。

## 结论

2026-08-24 后续加固：P2-1/P2-2 已完成。当前已有
`library.models.family_registry.ModelFamilySpec` 作为 canonical family/capability/cache
契约，核心训练、预处理、推理和 Web image-test 均使用覆盖全部已注册
family 的显式 handler 表。新 family 被注册但漏接任一 handler 时会 fail-closed，
不再回退 Anima。TE cache 新文件也已写入 `model_family`/`cache_schema`
metadata，Krea cache 额外校验 hidden/mask shape、dtype 和新格式 hidden width；
旧无 metadata cache 仍兼容。

审计时后端是**已贯通的双模型实现**；2026-08-24 后续加固又注册了 Z-Image v1，
但系统仍是显式静态 family 集合，不是动态插件框架：

- Anima 是默认且完整的 family。
- Krea-2 Raw 已贯通 Qwen3-VL 文本链、共享 Qwen VAE、SingleStreamDiT、plain LoRA
  训练、TE cache、NF4、block swap、checkpoint、fixed compile、`torch` / `flash`
  attention，以及标准单提示词 Euler + CFG 推理。
- Z-Image 已作为第三个 family 接入训练与训练预览；通用推理仍未注册，并由空的
  inference mode/sampler 集合 fail closed。
- 当前已有统一 `ModelFamilySpec` registry，但没有动态 `ModelFamily` plugin Protocol；
  每项 operation 仍通过覆盖全部已注册 family 的显式 handler 表分派。

审计未发现必须立即停用现有 Anima 或 Krea plain-LoRA 主链的 P0 问题，初始发现
**6 项 P1**。这些 P1 已于 2026-08-10 在当前工作树完成 fail-closed 修复；不支持的能力
现在会在加载大模型前拒绝，不再进入错误路径或静默忽略配置。

## P1 修复状态

| Finding | 状态 | 落地 |
| --- | --- | --- |
| P1-1 Krea batch/interactive 误走 Anima | 已修复 | `validate_krea2_inference_args()` 在安装 Anima strategy 前拒绝 |
| P1-2 Web family/path 分裂 | 已修复 | 当前配置 family 优先，全局/env 仅 fallback；子进程始终显式传 family |
| P1-3 Krea 参数静默忽略 | 已修复 | 仅放行 Euler/官方自动 mu shift；SMC、CNS、Soft Tokens 等显式拒绝 |
| P1-4 family/metadata fail-open | 已修复 | 公共严格 normalizer；未知 args/env/Web/preprocess/checkpoint 均拒绝 |
| P1-5 runtime/network family 分裂 | 已修复 | 同值冗余可兼容，冲突 `network_args model_family` 启动前失败 |
| P1-6 Krea 高级 adapter fail-open | 已修复 | compat allowlist + factory spec + adapter resolver 三层防线 |

旧 Anima checkpoint 兼容规则保持不变：`ss_model_family` **字段缺失**仍解释为 Anima；
字段一旦存在就必须是合法 canonical family。Krea 推理要求 checkpoint 明确盖
`ss_model_family=krea2_raw`。

## 审计时能力矩阵（2026-08-10）

下表是双模型审计时的历史快照。Z-Image 于 2026-08-24 后续接入训练和训练预览，
不应从本表的“第三个 family”行推断当前 registry 状态。

| 能力 | Anima | Krea-2 Raw | 审计结论 |
| --- | --- | --- | --- |
| CLI/TOML family 选择 | 完整 | 完整 | 审计时 args/env/内部入口统一校验，仅允许两个 canonical family |
| 文本编码 | Qwen3 + LLM adapter | Qwen3-VL + MFA | 训练和单提示词推理已分流 |
| VAE / latent | Qwen Image VAE | 共享同一 VAE | 4D cache、5D DiT 边界可复用 |
| TE cache | `_anima_te` | `_krea2_te` | 后缀和 preprocess fingerprint 已隔离 |
| 训练前向 | 完整能力面 | rectified-flow 主链 | Krea 不支持 Anima extra forwards |
| adapter 训练 | 完整/实验目录 | plain LoRA 为正式支持面 | 兼容矩阵、factory 和 adapter resolver 均拒绝高级 adapter |
| NF4 / block swap / checkpoint | 支持 | 支持 | Krea 有探针和定向测试证据 |
| compile / checkpoint mode | 多种模式 | fixed resident；full 或 every-other | 兼容矩阵已做部分 fail-fast |
| Attention | 多后端 | `torch` / `flash` | 训练和 Web 生图有前置校验 |
| 单提示词推理 | 完整 | Euler + CFG + LoRA | 已 family dispatch |
| `--from_file` 批量推理 | 完整 | **显式拒绝** | 未实现，不再进入 Anima loader/strategy |
| interactive 推理 | 完整 | **显式拒绝** | 未实现，不再调用 Anima `generate()` |
| 非 Euler sampler / SMC / CNS | 支持或按 Anima 约束 | **显式拒绝** | 未接线参数不再静默忽略 |
| 编辑、蒸馏、DCW、Spectrum | 支持或实验支持 | 不支持 | 部分入口已显式拒绝 |
| Web 模型配置库 | 支持 | 支持 | 原子写入、revision 冲突保护已完成 |
| 第三个基础模型 family | 无（审计时） | 无（审计时） | 审计时未注册；后续已加入 Z-Image v1（训练/预览），通用推理仍为空 |

## 已建立的正确边界

### 训练主链

- `library/training/model_loading.py:29-86` 按 `resolve_model_family(args)` 分流文本编码器
  和 DiT，VAE 明确共享。
- `library/training/anima_strategies.py:17-80` 分流 tokenize、text encoding 和 TE cache。
- `library/training/batch_step.py:19-32` 将 Krea 训练转给
  `library/models/krea2_raw/family.py::compute_noise_pred_and_target`，Anima 保持原路径。
- Krea `forward_for_loss` 保持 `(B,C,T=1,H,W)` 边界，下游继续消费统一的
  `(prediction, target, timesteps, weighting)` 契约。
- `library/cache_pool/fingerprint.py` 已把 `model_family` 纳入预处理指纹；Krea TE 使用
  `_krea2_te.safetensors`，避免与 Anima 文本缓存直接碰撞。

### 推理主链

- `inference.py:1004-1029` 为 Krea 单提示词提供薄 dispatch。
- `library/models/krea2_raw/inference_runner.py` 独立承担 TE lazy load、strict DiT load、
  LoRA attach 和 Krea 官方 Euler flow-matching 采样，避免把逻辑继续堆入热点文件。
- `inference_runner.py::validate_krea2_inference_args` 已显式拒绝 batch/interactive、非 Euler
  sampler、自定义 Flow Shift、SMC、CNS、Soft Tokens，以及 P-GRAFT、Spectrum、SPD、
  tiled diffusion、DCW、mod-guidance、IP-Adapter 和 EasyControl。
- TE 在 DiT 前编码并释放，保持 `TE -> free -> DiT` 的显存不变量。

### Web/API

- `web/services/model_config_service.py:52-93` 同一次原子写入模型配置库和默认项的
  family/三条路径，并使用 revision 防止多页面覆盖。
- `web/services/image_test_service.py:218-224` 在启动大模型前限制 Krea attention 和 dtype。
- `library/training/compat_matrix.py` 已覆盖 Krea plain-LoRA allowlist、attention、dynamic
  compile、Inductor mode、selective checkpoint 和 V100 Flash 的关键组合。

## P1 Findings

以下各节保留审计时的故障证据和决策理由；当前落地状态以“P1 修复状态”表为准。

### P1-1 Krea 批量和 interactive 推理进入 Anima 路径

**证据**

- `inference.py:1107-1116` 在判断 `--from_file` / `--interactive` 前无条件安装
  `AnimaTokenizeStrategy` 和 `AnimaTextEncodingStrategy`。
- `inference.py:1118-1127` 先进入 batch/interactive；family dispatch 只存在于
  `inference.py:1135-1138` 的单提示词分支。
- `process_batch_prompts()` 在 `inference.py:782-815` 调用 Anima 的
  `load_dit_model()`、`load_text_encoder()` 和 `prepare_text_inputs()`。
- `process_interactive()` 在 `inference.py:983-988` 直接调用 Anima `generate()`。

**影响**

`--model_family krea2_raw --from_file ...` 和 Krea interactive 不是降级功能，而是会使用
错误架构的加载/文本/forward 假设，可能在加载时失败，也可能产生难以定位的参数错配。

**建议**

短期在参数校验阶段明确拒绝 Krea 的 batch/interactive。完成 Krea model reuse、批量 TE
预编码和交互共享模型生命周期后，再开放这两个模式。

### P1-2 Web 生图的 family 与当前配置路径可以分裂

**证据**

- 前端把当前训练配置完整放入 image-test payload，且配置本身包含 `model_family`：
  `web/static/js/features/image-test/index.js:263-282`。
- 后端却忽略 `payload.config.model_family`，只从全局设置读取 family：
  `web/services/image_test_service.py:203-208`。
- 同一请求的 DiT/Qwen3/VAE 路径来自当前配置：
  `web/services/image_test_service.py:264-284`。
- `web/services/config/preflight_paths.py:29-43` 只回填三条模型路径，不同步 family。

**影响**

当前配置是 Krea、全局默认是 Anima（或反向）时，Web 生图可能把一个 family 的路径交给
另一个 family 的 loader。模型配置库默认项的原子镜像解决了“保存默认项时分裂”，但没有
解决“当前显式配置不同于全局默认”的情况。

**建议**

image-test 应优先使用当前配置的 `model_family`，全局 family 只作为字段缺失时的 fallback；
后端应把 `{family, dit, text_encoder, vae}` 作为一个不可拆的 model selection 校验。

### P1-3 Krea 推理暴露但忽略部分采样和扩展参数

**证据**

- CLI 暴露 `sampler=euler|er_sde|lcm` 和 `flow_shift`：`inference.py:253-265`。
- Web 会提交并展示 sampler/flow shift：
  `web/services/image_test_service.py:203-215,274-284,685-708`。
- Krea `generate_krea2()` 没有读取 `args.sampler` 或 `args.flow_shift`，始终调用
  `library/models/krea2_raw/sampling.py::sample` 的官方 Euler + mu-shift 路径：
  `library/models/krea2_raw/inference_runner.py:219-274`。
- CLI 的 SMC-CFG、CNS 和 `soft_tokens_weight` 也没有进入 Krea runner；实际拒绝表只有
  `inference_runner.py:48-57` 的八项，和文件顶部“直接拒绝 soft-tokens/Hydra”的注释不一致。

**影响**

用户可以选择 `er_sde`、`lcm`、自定义 Flow Shift、SMC-CFG、CNS 或 Soft Tokens，命令和
历史记录看起来保留了配置，但 Krea 实际仍跑官方 Euler 默认语义。这会污染效果对比和实验记录。

**建议**

建立 Krea inference capability 校验：当前仅接受 `sampler=euler`，并拒绝所有未接线参数。
如果保留 Krea 官方自动 mu shift，应在 Web/CLI 明确把 Anima `flow_shift` 标为不适用，而不是
接受后忽略。

### P1-4 family 解析和 checkpoint metadata 采用 fail-open 回退

**证据**

- `library/env.py:257-273` 只做 strip/lower，不验证 env/TOML family。
- 多数分发写成 `family == "krea2_raw"`，否则进入 Anima，例如
  `model_loading.py:40-50,80-86` 和 `batch_step.py:22-31`。
- `networks/lora_anima/config.py:501-511` 将未知 network family 静默改成 Anima。
- `networks/lora_anima/factory.py:833-847` 将未知 checkpoint `ss_model_family` warning 后
  回退 Anima。
- Web 模型配置 API 严格拒绝未知值，但全局设置把未知值清空为默认 Anima，行为不一致。

**影响**

CLI choices 能挡住直接输入，但 env、合并配置、旧 checkpoint 或内部调用仍可能把拼写错误
变成 Anima 执行。对模型架构选择而言，fail-open 比提前终止风险更高。

**建议**

新增唯一的 `normalize_model_family(value, *, allow_empty)`；运行入口和 checkpoint attach
必须对未知值 fail-closed。只有“metadata 完全缺失的历史 Anima checkpoint”可以保留兼容默认，
且应同时核验 target container/key 形状，避免把旧 Krea 文件误判为 Anima。

### P1-5 args family 与 network family 可以互相矛盾

**证据**

- `library/training/bootstrap.py:133-157` 允许显式 `--network_args model_family=...` 优先于
  `resolve_model_family(args)`。
- 同一函数在 `bootstrap.py:159-183` 又只按 `resolve_model_family(args)` 决定是否注入
  Krea `SingleStreamBlock` target spec。
- 两个值之间没有一致性断言。

**影响**

可以构造“加载 Krea DiT + 注入 Krea target，但 checkpoint stamp 为 Anima”，或“加载 Anima
DiT + checkpoint stamp 为 Krea，但仍使用 Anima target”的组合。错误可能直到加载 checkpoint
或推理 attach 才暴露。

**建议**

`model_family` 不应作为可独立覆盖的普通 network arg。network cfg 必须从已解析的运行 family
单向注入；发现重复配置且值不同应在加载大模型前报错。

### P1-6 Krea 与高级 adapter 缺少统一的训练拒绝矩阵

**证据**

- `library/models/krea2_raw/family.py:158-169` 明确只实现最简 rectified-flow，不实现
  crossattn、postfix、method-adapter extra forwards、VR loss、affine 和 observer。
- `library/training/adapter_resolver.py:13-39` 只按方法 flag 注册 IP-Adapter、EasyControl、
  BYG、Soft Tokens，没有读取 `model_family`。
- `library/training/batch_step.py:148-160` 允许拥有训练步的 adapter 绕过 family 的标准
  `compute_noise_pred_and_target`。
- `library/training/compat_matrix.py:143-188` 的 Krea 专属规则只覆盖运行时优化组合，
  没有 adapter capability 校验。

**影响**

用户把 Anima 方法配置与 Krea family 组合时，可能晚到 network build/首个 batch 才失败，
也可能绕过 Krea family 的约束进入未经验证的 adapter forward。

**建议**

建立 `family x adapter x operation` 能力表，至少区分 `train`、`infer_attach`、`merge`、
`resume`。当前 Krea 应只放行经过验证的 plain LoRA/NF4 组合，其他组合显式拒绝或标记实验。

## P2 Findings

### P2-1 缓存内容仍是位置协议（已修复，2026-08-24）

Krea TE cache 通过 `[hiddens, mask, caption_dropout_rate]` 与 Anima 的“末位 rate”约定兼容，
`library/training/batch_preprocess.py` 会拆掉最后一项。后缀隔离已经降低碰撞概率，但缓存内容
没有 family/schema/version 字段；损坏文件或错误 sidecar 可能被按位置误读。

已为新写入的 Anima/Krea TE cache 增加 `model_family` 和 `cache_schema`
safetensors metadata。加载显式 metadata 冲突的 cache 会拒绝；Krea 会通过
safetensors slice header 校验 hidden/mask 的维度、dtype、序列长度以及新格式
2560 hidden width，不需物化整份 cache。旧无 metadata cache 保持可读。

### P2-2 family 扩展点是分散的二分支（已修复，2026-08-24）

当前核心结构是多处 `if family == "krea2_raw" else anima`，没有注册式 family API。
新增 family 至少要同步：

1. env/CLI/Web 枚举与 alias；
2. model/text/VAE loader；
3. tokenize、encode、cache suffix/schema；
4. training forward/loss/sampler；
5. LoRA target 和 checkpoint metadata；
6. inference runner 与支持模式；
7. Web preflight 和 capability UI；
8. 测试矩阵和文档。

已落地 `ModelFamilySpec` 注册表，集中 canonical name/alias、TE cache 契约、
network/method-adapter capability、inference mode/sampler/attention/Flow Shift 约束。
核心 call site 改为 `dispatch_model_family()` 显式 handler 表；该 helper 每次检查
是否覆盖全部已注册 family，因此新增注册项不会被任一 `else anima`
静默吸收。Anima 大目录未搬迁，训练和推理公式未改变。

### P2-3 历史文档状态落后于代码

审计时 `docs/multi_model_support.md` 仍写“Nothing here has been implemented”，
`_archive/docs/proposal/krea2_raw_migration.md` 和阶段 6 findings 的部分段落仍写推理 dispatch 未闭合。
本轮已更新前两份文档的状态说明；阶段 findings 保留历史原文。本审计是 2026-08-10 的
当前事实入口。

## 测试覆盖审计

### 已覆盖

- Krea attention backend、compile、selective checkpoint、NF4/self-contained NF4。
- Krea TE cache、preprocess dispatch、cache fingerprint。
- Krea training sample preview 和部分 model-loading integration。
- `ss_model_family` stamp 保存/读取。
- Web global family、模型配置库、image-test attention 约束。

### 本轮补齐

1. Krea `--from_file` / `--interactive` 在安装 Anima strategy 前拒绝。
2. Krea 对 `er_sde`、`lcm`、Flow Shift、SMC、CNS、Soft Tokens 的 fail-fast。
3. image-test 当前配置、全局设置和 env/default 的 family 优先级。
4. args/env、`network_args model_family` 和 checkpoint metadata 的未知/冲突行为。
5. Krea plain-LoRA allowlist 与高级 adapter 配置拒绝矩阵。
6. preprocess 和 legacy model-config 入口的未知 family 拒绝。

### 仍缺或不足

1. 从完整 Krea TE/latent cache 进入 `train_session` 的最小端到端测试。
2. malformed/wrong-family TE cache schema 测试。
3. 第一个 Krea LoRA 的 Web queue -> preprocess -> train -> save -> image-test 闭环测试。

真实 GPU 探针和历史 findings 已证明 Krea plain-LoRA、NF4、checkpoint、compile/Flash 的机制，
但不能替代上述控制流和拒绝边界测试。

## 建议落地顺序

### 第一阶段：先 fail-closed（已完成）

1. 已统一 family normalize/validate，未知值不再进入 Anima。
2. 已拒绝 Krea batch/interactive 和本轮确认的未接线推理参数。
3. image-test 已改为当前配置 family 优先，并始终向子进程显式传 family。
4. 已禁止 `network_args` 单独覆盖 family，并校验 runtime/network/checkpoint 一致性。
5. 已在兼容矩阵、factory 和 adapter resolver 加入 Krea plain-LoRA allowlist。

这一阶段改动小，能消除大部分“看似成功但语义错误”的风险。

### 第二阶段：形成 capability registry

建议的最小结构：

```python
@dataclass(frozen=True)
class ModelFamilySpec:
    name: str
    load_target_model: Callable
    tokenize_strategy: Callable
    text_encoding_strategy: Callable
    text_cache_strategy: Callable
    lora_target_spec: Callable
    training_step: Callable
    inference_runner: Callable
    supported_inference_modes: frozenset[str]
    supported_samplers: frozenset[str]
    supported_adapters: frozenset[str]
```

先注册 `anima` 和 `krea2_raw`，逐步替换散落白名单；不要在同一轮搬迁 Anima 大文件或改变
训练行为。

### 第三阶段：补能力而不是扩大参数表

- 为 Krea 实现 batch/interactive model reuse。
- 只有完成公式、数值和效果验证后才开放新的 Krea sampler/adapter。
- Web 依据 capability registry 过滤 UI，同时后端继续独立校验，不能只依赖前端隐藏字段。

## 验证

P1 运行时代码和审计文档落地后，按单文件或小组执行以下定向测试类别：

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_model_family_fail_closed.py \
  tests/test_krea2_inference_capabilities.py \
  tests/test_factory_metadata_flow.py \
  tests/test_settings_model_family.py \
  tests/test_training_compat_matrix.py \
  tests/test_image_test_service.py \
  tests/test_krea2_attention_webui.py \
  tests/test_model_config_service.py \
  tests/test_preprocess_reuse_flags.py -q

timeout 60 .venv/bin/python -m pytest tests/test_documentation_integrity.py -q
```

另对 training bootstrap/runtime config、Web HTTP/preflight/global settings、generation request、
推理 adapter/PNG metadata 和相关前端契约执行扩展回归。

结果：23 个定向与关联测试文件合计 `295 passed`；文档完整性测试 `8 passed`；
Ruff、4 个修改 JS 文件的 `node --check` 和 `git diff --check` 均通过。

本轮已修改运行时 fail-closed 边界，但未启动真实训练、GPU 探针或大模型推理；这些结果不应
外推为新增 Krea batch/interactive/sampler/高级 adapter 能力。
