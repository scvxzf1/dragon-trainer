# 多模型支持现状

状态：稳定架构说明
适用版本：当前 `main` 工作树
相关代码：`library/models/family_registry.py`、`library/training/`、`library/models/<family>/`

本文记录当前已经落地的多模型边界及仍然存在的 Anima 耦合。它不是新增模型的未来方案，
也不把训练预览等同于通用推理能力。2026-08-10 的阶段性风险快照仍保留在
[后端多模型审计](findings/backend_multi_model_audit_20260810.md)；其中部分问题已经由后续
registry 和 fail-closed dispatch 解决，应以本文和实时源码为准。

## 当前模型族

`library/models/family_registry.py::MODEL_FAMILY_REGISTRY` 是模型族名称、别名和能力边界的
单一事实源。所有 family operation 必须为已注册模型族提供完整 handler；缺失 handler、未知
family、未注册推理模式或 sampler 都应失败，不得静默回退 Anima。

| 模型族 | 当前定位 | Adapter | 通用推理 | Attention |
| --- | --- | --- | --- | --- |
| `anima` | 默认生产模型族 | 完整方法面 | `single` / `batch` / `interactive`；`euler` / `er_sde` / `lcm` | registry 中声明的 Anima 后端 |
| `krea2_raw` | 第二生产模型族 | 开放所有网络变体选择 | `single` + `euler` | `torch` / `sdpa` / `flash` |
| `z_image` | 训练与训练预览 v1 | 开放所有网络变体选择 | 尚未注册；mode/sampler 集合为空 | `torch` / `sdpa` / `flash` (`flash_varlen`) |
| `qwen_image_2_1` | T2I/Edit 训练与训练预览；CMP 90HX torch/Flash 指定组合各 60 步短测通过 | T2I plain LoRA；T2I 实验 LoKr/LoHa；Edit 仅 plain LoRA | `single` + `euler`（T2I/Edit、plain LoRA） | `torch` / `sdpa` / `flash`（BF16） |
| `moss_image` | 普通文生图 plain LoRA 首版；真机质量与性能仍需按硬件验收 | plain LoRA | 尚未注册；mode/sampler 集合为空 | `torch` / `sdpa` |

实验性 pipeline_parallel：`moss_image` 与 `anima` 已发布共用 stage-local PP v0
backend（见 `docs/experimental/moss_image_pipeline_parallel_v0.md`、
[anima_pipeline_parallel_v0.md](experimental/anima_pipeline_parallel_v0.md) 与
[moss_image_pp_trainer_wiring.md](experimental/moss_image_pp_trainer_wiring.md)）；
两个族均已接线专用 PP 训练会话（PP=2/1F1B/FP16/plainLoRA/完整缓存范围；Anima 侧经
族适配器复用 `build_anima_pipeline_runtime`，WebUI/队列的 `pipeline_parallel` 与
`model_family` 键必须成对进入 runtime config；gpu_whitelist 守卫已放开为解析级
fail-closed：GPU 全卡可选，唯一具名拒绝 `gpu_whitelist_unauthorized_gpu` 的条件是
显式选卡索引在 nvidia-smi 中解析不到物理卡；不适配卡（如 GTX 1050）改为分布式
诊断端点的建议级警告，不再阻断 enqueue/start），
其余族与 DDP 内部路径维持 `pp_trainer_path_unavailable` 具名拒绝，不构成通用
PP 训练支持。CPU/Gloo 与双 V100 CUDA tiny 热测均已独立验收（rc=0）；真实
大模型权重训练未验收。

WebUI 能选择五个模型族，但这不扩大 registry 的能力集合。尤其是 Z-Image 的训练 sample
preview 只服务训练流程，不代表生图测试、独立 CLI 推理、batch 或 interactive 已可用。

Anima、Krea-2 与 Z-Image 开放网络变体选择，配置预检与 network 创建/加载不再套用
“非 Anima 仅 plain LoRA”的统一白名单。`supported_network_specs=None`、
`plain_lora_only=False` 表示允许尝试，不代表所有变体组合均已通过训练验证。
不兼容的权重、精度、目标层结构与 forward 契约仍在对应实现处报错，不静默替换成 LoRA。
Krea-2 与 Z-Image 的 `supports_method_adapters=False` 保留：IP-Adapter、EasyControl、BYG
等依赖独立训练 forward hooks 的方法不因开放网络变体而自动适配。
Qwen Image 2.1 的 `plain_lora_only=True`，不适用上述开放策略。

## Pipeline Parallel 规划能力

Anima、Krea-2 与 Z-Image 声明了双卡 PP 拓扑，并通过公共
`library/models/pipeline_parallel.py` 完成配置归一化、互斥校验和连续 block 范围规划。
Anima 使用 `blocks` 并均分 28/40 block；Krea-2 使用 `blocks` 并保留 28 block
的 13/15 启发式；Z-Image 只分配 `layers` 中的 30 个 main layers，不混入 refiner。

这些是 planner/probe 能力，不是生产 runtime。当前三族的 registry capability 都显式标记
`runtime_available=false`；CLI、WebUI、续训和队列均在合法规划后 fail closed，不会回退到
普通 DDP。完整设计和当前边界见 [双卡 Pipeline Parallel 阶段 1](proposal/krea2_pipeline_parallel.md)。

## 当前路由

### 训练

- `train.py` 仍保留 `AnimaTrainer` 类名作为兼容 facade；统一训练编排位于
  `library/training/`。
- `library/training/model_loading.py` 按 `model_family` 分派 text encoder、VAE 和 DiT 加载。
- `library/training/batch_step.py::_get_noise_pred_and_target` 使用完整 handler 表分派每个 batch；
  Anima、Krea-2 和 Z-Image 各自拥有训练 forward/target 契约。
- 非 Anima family 不通过 metadata fallback 猜测行为；配置、network metadata 和运行 family
  发生冲突时应由 compatibility/registry 层拒绝。

### 缓存和模型实现

- Anima 实现在 `library/anima/`，Krea-2 和 Z-Image 分别位于
  `library/models/krea2_raw/`、`library/models/z_image/`。
- text cache 通过 registry 隔离：Anima 使用 `_anima_te.safetensors`，Krea-2 使用
  `_krea2_te.safetensors`，Z-Image 使用 `_z_image_te.safetensors`。
- latent/cache schema 由各 family 的 strategy/latent 模块维护；不要通过改名让不同模型族
  共享不兼容 sidecar。
- 所有 family 继续遵守 lazy loading 顺序：text encoder 缓存并释放，VAE 缓存并释放，最后
  加载 DiT、apply adapter 并进入训练。

### 推理

- Anima 推理主链位于 `library/inference/`。
- Krea-2 通过 `library/models/krea2_raw/inference_runner.py` 走独立 single/euler 路径；
  Anima-only extras 必须显式拒绝。
- Z-Image 目前只有 `library/models/z_image/training_preview.py`。通用 image-test 和独立推理
  没有注册 handler，请勿从 WebUI family 下拉框推断其已支持。
- Qwen Image 2.1 的独立入口支持 BF16 `single` + `euler` 的 T2I/Edit 和最多一个 plain LoRA；Edit 接受 1–4 张有序参考图，完整 NF4 权重的单参考单步与四参考/CFG2/两步出图及 latent 再解码已通过。image-test、batch 和 interactive 仍显式拒绝。

### 采样批次的块交换生命周期

Anima、Krea-2 与 Z-Image 训练预览在一批样张进入/退出时统一切换块交换，
不再在每个 prompt 前重复初始化。入口、预测和清理异常进入既有跨进程失败汇总；
退出独立尝试恢复 adapter 原始 training 状态、合法驻留布局以及 CPU/当前 CUDA RNG。
恢复本身失败仍会传播错误，不保证设备错误后可以继续训练。解码移动模型后的必要 prepare 保留。

Anima tiled 推理同样复用一次 forward-only scope，退出恢复原 swap 模式。
CPU/Gloo 和小型真实 CUDA/slab 验收已通过，完整三族权重的吞吐与样图质量未在本轮测量。
本轮没有推广跨族 DiT K/V 缓存；Qwen 继续默认 B。详见
[采样生命周期验收](findings/sampling_batch_lifecycle_20261002.md)。

## Qwen Image 2.1 首阶段边界

2026-10-03 增补：Krea-2、Z-Image、Qwen Image 2.1 新增专用主干MLP混合BF16/NF4
文件的实验加载路径，通过现有模型路径自动识别，不新增训练配置项。
三族格式、官方小模型plain LoRA/checkpoint及CUDA换层验收已通过；没有完整权重热测或
质量结论。新格式限BF16/eager/full checkpoint，拒绝compile和Qwen融合投影等组合。
`base_compute=nf4`在Z-Image/Qwen仍不可用；下文旧NF4禁用声明指该通用开关，
专用混合文件的完整边界见[多模型族混合NF4](experimental/multi-family-mixed-nf4.md)。

`configs/methods/qwen_image_2_1_lora.toml` 是 T2I plain LoRA 起始配置。三个组件使用
ComfyUI 单文件 safetensors：32-block DiT、Qwen3-VL 8B 和独立的 RGBA/64-channel VAE。
加载器在装载权重前核对完整参数名与形状；文本编码器的 ComfyUI `model.*` 命名映射到
Transformers `model.language_model.*`，VAE 转换卷积的 singleton 时间核，DiT 拆分
`img_mlp.gate_up`。

**Tokenizer 是独立依赖，必须与 TE 配套。** `qwen3` 指向单个 `.safetensors` 时，
`library/models/qwen_image_2_1/strategy.py::resolve_tokenizer_path` 会从该文件所在目录
逐级向上查找带 `tokenizer_config.json` 的 `tokenizer/` 目录，或 ComfyUI 安装里的
`comfy/text_encoders/qwen25_tokenizer`；两者都找不到就报
`FileNotFoundError: Qwen Image 2.1 needs a local Qwen3-VL tokenizer`。
因此「单文件 TE」不等于「自带 tokenizer」，需要满足以下之一：

- 在 TE 文件同级放一个 `tokenizer/` 目录（Qwen3-VL 的 `tokenizer_config.json` /
  `tokenizer.json` / `vocab.json` / `merges.txt`）；
- 或在同一目录树内存在 ComfyUI 的 `comfy/text_encoders/qwen25_tokenizer`。

注意该族要求 Qwen3-VL 词表（`<|im_start|>`=151644、pad=151643），加载时校验 pad token；
ComfyUI 的 `qwen25_tokenizer` 虽能通过该校验（pad 同为 151643），但其词表对齐 Qwen2.5
而非 Qwen3-VL，**不保证与本族 TE 训练时一致**，仅作兼容回退。
若使用官方 Diffusers 目录或本机 HF 快照（如 `Qwen3-VL-4B-Instruct`），
目录内已带 tokenizer，无需额外配置。文本以官方 T2I 模板左填充，取最终 decoder 层 RMSNorm 前 hidden，
使用独立 `_qwen_image_2_1_te.safetensors` 缓存。RGB 数据在 VAE 前补不透明 alpha，
64-channel latent 使用 VAE 的逐通道 mean/std，并隔离到 `_qwen_image_2_1.npz`。

训练 forward 使用未 patch 的 latent token、2x2 image slot mask、动态指数 shift 的
flow-matching sigma 和 attention-only LoRA。当前仅允许 BF16、`torch`/`sdpa` 原生 attention
或 opt-in `flash`，以及 full checkpoint。Qwen 专属 `transformer_blocks` block swap 和 per-block `torch.compile`
已接入，方法配置初始启用 `blocks_to_swap=24`、`torch_compile=true`、dynamic sequence；
编译发生在 LoRA apply/load 后，换块调度留在编译图外。Next 数据集预设现支持 Qwen Image 2.1
编辑 LoRA：每组使用同一配对名称的“编辑前”和“编辑后”两个 UI 子集，可有多组；保存时只有
“编辑后”进入训练蓝图，其 `reference_image_dir` 指向对应“编辑前”的图片目录。图片仍按
相对路径与 case-fold 文件名 stem 配对，缓存 Qwen3-VL image-slot 条件及参考图 VAE latent；
首版要求 batch size 1，不支持混用普通/正则子集或分阶段调度，
随机裁剪、翻转和颜色增强。Apply 会校验 Qwen family、plain LoRA、缓存开关及图像配对，并写入
`qwen_image_2_1_task="edit"`，不会启动训练。

Edit 已通过配对、缓存 schema/网格、DiT 序列形状、Apply 和 UI 单元测试，并在 CMP 90HX 上用 50 对真实
编辑数据完成 plain LoRA 的 50 步 BF16/full-checkpoint/swap/compile 链路验收；训练预览也已通过单、双、
四参考图及 CFG 编辑短测（见[Edit 热训练](findings/qwen_edit_hot_training_20260928.md)和[编辑采样](findings/qwen_edit_sampling_hot_20260928.md)）。该证据覆盖训练链路和采样后恢复，不代表编辑质量、optimizer 续训、其他 adapter
或其他硬件组合已验收。`auto_block_swap`、CUDA Graph compile preset、全量NF4开关、
文本编码器训练、caption dropout 和 PP 仍未接入。T2I/Edit 独立推理通过
`library/models/qwen_image_2_1/inference_runner.py` 复用专属 Euler、64-channel/16x VAE decode
和严格 plain LoRA loader，并支持专属 `--latent_path` 解码。完整 NF4 的单参考 Edit 已出图；
四参考 CFG2/两步在接入自动 TE offload 后出图并完成 latent 再解码，PNG 像素逐字节相同。
边界见[独立推理验收](findings/qwen_standalone_inference_20261005.md)。仍不支持 image-test、
batch/interactive、LoKr/LoHa 推理或独立 compile。NF4 训练 compile 的静态 + 原生 norm +
非 persistent reduction 诊断组合已通过完整两步合成 Edit 数值对照，但真实缓存、
896×1184 bucket 与 flow loss 两步对照失败；strict eager 跨实例两步已逐位复现，
追加原生SiLU诊断后，完整真实两步保存张量通过原梯度门槛并逐位一致，已独立
复算；随后真实两步与原生合成CLI独立重验均正常exit0。生产attention-only、
zero-up初始化的两个真实Edit bucket也完成1286项逐位一致的独立重验、exit0。
联合策略已接入默认关闭的 `qwen_nf4_compile_experiment` 受限入口；真实 `train.py`
完成两步 AdamW 更新、约 67 MB checkpoint 保存并正常退出；追加两次 512×512 训练预览、
完整状态保存及新进程 step2→3 续训均自然退出。Next 工作台开关与本地正式构建通过。
仅限 BF16 Edit、plain attention LoRA rank/alpha16、torch attention、full checkpoint、
swap24/foreach/BF16、static/all/Inductor default。普通 NF4 compile 拒绝保留；
连续训练与续训的完整轨迹逐位复现、其他 optimizer、长训、性能、画质和跨硬件尚未验收。
Python RNG 恢复即刻与保存值一致，随后 Rich 日志会消费全局 Python RNG，详见定位报告。配置见
[NF4 编译实验](experimental/qwen-nf4-compile.md)，历史失败与验收见
[NF4 编译定位](findings/qwen_nf4_compile_90hx_20261005.md)。
依赖 Diffusers 尚未发版的 Qwen 2.1 实现，
`pyproject.toml` 暂固定到包含官方 PR 的上游提交。Flash 使用 Qwen 专属 packed
FlashAttention 2 varlen：按真实 key-valid mask gather K/V，避免把 tokenizer 左填充误作有效前缀；
带因果三角 mask 的文本段仍走原生 SDPA，图像查询走 Flash varlen。`configs/methods/`
继续默认 `attn_mode="torch"`。CPU 小模型 forward/backward、外置 adapter 梯度与路由契约测试
通过；CMP 90HX 的 torch 和 Flash 各完成 60-step 单图运行及 LoRA 保存，报告分别见
[torch 热测](findings/qwen_image_2_1_90hx_hot_test_20260926.md)和
[Flash 热测](findings/qwen_image_2_1_flash_90hx_20260927.md)。当前 90HX 单次对照中 Flash
中位步时约 10.16 秒，torch 约 7.94 秒，因此 Flash 保持 opt-in，不作为速度推荐。
这些短测不代表训练收敛、样图质量、保存后重载/续训、其他 swap 数或其他硬件；`24` 是通过
测试的点位，不是跨硬件默认建议。

LoKr/LoHa 仅在 T2I 的指定 BF16 + full checkpoint + swap24 + torch attention + compile
组合上完成 90HX 两步 smoke 与 20 步热测；本轮 plain LoRA、LoHa、LoKr 的稳态中位步时分别为
7.938、9.090、11.373 秒。LoKr 需要在 Qwen block 编译前启用 checkpoint 编译预算；阶段 3
投影融合与 LoKr/LoHa 组合仍拒绝。DoRA、VeRA 和其他变体未开放。Next WebUI 仅在 T2I 任务中展示 LoKr/LoHa
实验选项，Edit 任务仍只展示 plain LoRA；这不扩大后端兼容矩阵或真机验收范围。
见 [变体热测](findings/qwen_image_2_1_adapter_variants_90hx_20260929.md)。

### 训练预览的 DiT 前缀 K/V 缓存

训练预览的块交换默认继承训练状态，可通过 `qwen_sample_swap_mode="custom"` 和
`qwen_sample_blocks_to_swap` 临时覆盖数量并在退出恢复。所有模型族的训练预览 VAE 独立使用
`sample_vae_decode="tiled"/"auto"/"cpu"`，默认分块 GPU 解码；自动模式仅在 CUDA OOM 时逐级降级。
这些配置不控制 DiT K/V 缓存；边界、界面入口与验收见
[Qwen 预览显存策略](configuration/qwen-preview-memory.md)。

Qwen 训练预览支持官方文本/参考图前缀 K/V 缓存。JSON 或 TOML 样张条目设置
`use_kv_cache=true` 显式启用；省略或设为 `false` 保留 B 路径（无逐步 swap 初始化、无 DiT K/V 缓存）。
直接 Python 采样 API 同样默认 B，可通过 `use_kv_cache=True` 显式启用 C；
开关不改变训练 forward 或磁盘 TE/VAE 缓存。
缓存只在当前样张内复用，CFG 正负分支独立，下一样张、下一采样轮次或权重更新后重新建立。
仅对官方 causal/eval 模型及静态 LoRA/LoHa/LoKr 启用；动态 hook、未知 attention processor
和自定义 forward 回退无缓存。当前前端没有增加缓存控件。
该路径增加显存，BF16 输出也可能不同；CMP 90HX 上的收益与数值边界见
[A/B/C 验收](findings/qwen_sampling_kv_abc_90hx_20261002.md)，不作为默认速度优化。

## Z-Image v1 边界

Z-Image 使用官方 Diffusers transformer、Qwen3 text encoder 和 Flux VAE 组件。当前加载器支持
Diffusers 目录及经过组件校验的单文件路径，并在训练前校验 latent geometry、affine factor、
transformer input/caption width 和 Qwen3 hidden width。

已经接通的契约：

- Qwen3 ChatML、倒数第二层 hidden state、固定 512-token hidden/mask cache；
- 独立 `_z_image_te.safetensors` 和 `_z_image.npz` sidecar；
- `(latent - 0.1159) * 0.3611` 归一化和 shift `6` 的 1000-step flow grid；
- BF16 SDPA/FlashAttention varlen、full gradient checkpointing、attention-only plain LoRA；
- 训练 sample preview；
- `library/models/z_image/block_swap.py` 对 30 个 main layers 的 block swap，refiner 与
  input/output 模块常驻。

`blocks_to_swap=0` 表示关闭；启用时实现要求
`1 <= blocks_to_swap <= len(model.layers) - 2`，当前 30-layer 模型即 `1..28`。adapter 复用
共享 `ModelOffloader`，并保留 checkpoint、training/inference switch、pause/resume 和 profile
协议；不要把它当作 Krea-2 裸 DiT 的复制实现。

以下能力在 v1 明确不可用或由兼容层关闭：

- 通用推理/image-test、batch/interactive mode 和任意推理 sampler；
- 全量NF4开关、`torch.compile`、per-band dynamic-seq 和 selective/offloaded checkpointing；
  专用混合NF4文件作为实验例外，见上方增补与实验文档；
- 依赖独立训练 forward hooks 的 MethodAdapter（如 IP-Adapter、EasyControl、BYG）；
- weighted caption、layer range、alternate loss/timestep scheme；
- 任意大于 0 的 dataset/subset caption dropout。

起始配置为 `configs/methods/z_image_lora.toml`。不要把“理论上可移植”的 adapter 写成当前
支持；能力扩展必须先更新 registry、compatibility、训练/推理 handler 和定向测试。

## 仍存在的耦合

多模型主链已经穿过 trainer forward，但尚未变成动态插件系统，以下耦合仍是维护边界：

| 区域 | 当前事实 | 维护要求 |
| --- | --- | --- |
| family registry | 显式静态注册，handler 必须覆盖全部 family | 新 family 同步补全所有 operation handler 和 fail-closed 测试 |
| 训练加载 | `model_loading.py` 已分派，但仍直接导入 Anima loader/compat | 不要把 Anima fallback 当成通用协议 |
| 训练辅助 | `noise_target.py`、`train_session.py`、`trainer_network_mixin.py` 仍有 Anima import | 新 family 先走显式分支，不要伪装成 Anima tensor/layout |
| Adapter | 部分高级方法仍绑定具体 block 名称、cross-attention 和 monkey patch | 所有模型开放网络变体选择，具体实现校验兼容性；独立 forward hooks 仍需逐项适配 |
| 推理 | Anima 与 Krea-2 有独立路径，Z-Image 尚无通用路径 | registry 集合为空时必须拒绝，不得借用 Anima sampler |
| Compile/bucket | Anima 支持 dynamic-seq；Krea-2 使用固定 token family；Z-Image 未验证 compile | family compatibility 必须关闭不适用参数 |

## 跨 family 隔离契约

不同 family 的模型、text cache、latent cache 和 adapter 之间没有兼容面，任何“看起来能读”
的组合都按错误处理，不做静默降级：

- **Adapter**：LoRA checkpoint 的 `ss_model_family` stamp 必须与目标模型 family 一致
  （未 stamp 等同于 `anima`）。训练 resume / `--base_weights` / `--dim_from_weights`、
  Anima harness 和推理加载都会在读取 tensor、构建 target container 之前 fail closed，
  由 `networks/lora_anima/factory.py::require_adapter_family` 统一判定。跨 family 加载会
  命中零个模块并伪装成“已加载但无效果”，因此不能放行。
- **Text cache**：sidecar 后缀和 metadata schema 均按 family 区分；MOSS-Image 另有
  `nf4` / `bf16` 两套身份，切换量化方式必须换目录或重新预处理。
- **Latent cache**：后缀按 latent space 区分（`_anima.npz` / `_moss_image.npz` 等），
  MOSS 读取使用 stride 16 的 `{H}x{W}` key，Anima 的 stride 8 key 不会被命中。
- **模型**：`dispatch_model_family` 对未知 family、缺失 handler、未注册推理 mode/sampler
  一律拒绝；MOSS-Image 目前不注册通用推理 mode/sampler。

回归入口：`tests/test_adapter_family_isolation.py`（CPU-only，覆盖 adapter / text cache /
latent cache 三个面）。

## 修改检查清单

新增或扩展模型族时至少同步检查：

1. `MODEL_FAMILY_REGISTRY` 的别名、cache、adapter、推理和 attention 能力。
2. `library/training/model_loading.py`、`batch_step.py` 及 preprocessing strategy 的完整 handler。
3. network target、metadata stamp、保存/加载和 resume round-trip。
4. WebUI family 过滤、preflight 文案、训练预览和 image-test 的真实能力差异。
5. compatibility matrix 是否对无效配置 fail closed，且没有“接受但忽略”。
6. family-specific cache suffix、latent normalization、text mask 和 4D/5D latent 边界测试。

相关回归入口包括 `tests/test_model_family_registry.py`、`tests/test_z_image_family.py`、
`tests/test_z_image_block_swap.py`、`tests/test_krea2_*` 和 WebUI family/compatibility 测试；实际文件名
以当前 `tests/` 为准。
