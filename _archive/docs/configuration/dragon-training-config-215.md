# 已退役 Dragon 训练器 215 项配置快照

重组探索：[Dragon 训练配置渐进披露重组提案](../proposal/dragon_config_progressive_disclosure.md)

> 范围说明：下文“当前”“可用”等表述仅指快照采集时该配置的页面状态，不代表现在或所有模型族的能力。当前配置工作台请参阅 [Next README](../../web/frontend-next/README.md) 和[多模型支持说明](../multi_model_support.md)；各模型族参数兼容性以当前入口为准。

> 快照日期：2026-09-03（Asia/Shanghai）
>
> 页面：历史 Dragon 页面；当前配置入口为 `/next/training`
>
> 当前训练配置：`configs/imported/8-8-测试.toml`
>
> 运行覆盖：`default`
>
> 当前模型族：`krea2_raw`
>
> 数据集配置：`configs/datasets/8-8-k2.toml`

## 统计口径

本清单按 Dragon UI 的“全部参数 / 积木流 Beta”在上述配置与运行覆盖下实际渲染的 `215` 个配置块收集，包含当前不可用但仍显示的字段。它不是后端所有候选参数的静态全集。

同一代码库还存在其他统计口径：表单布局去重字段为 `210`，中文帮助索引为 `246`，动态后端 schema 当前为 `425`。它们分别代表布局目录、帮助覆盖面和训练 CLI/schema 候选面，不能与页面运行时的 215 项混用。

当前 215 项中：可用 `148`，当前不可用 `67`；新人层级 `59`，初学者层级 `12`，高级层级 `144`；必填 `4`，实验性标记 `11`。

## 当前分类情况

### 页面导航五分类

五分类控制分组编辑入口；下表按字段第一次出现的位置归属，避免跨分类复用字段重复计数，因此合计正好 215。

- **必填**：7 项
- **常用**：38 项
- **预览**：6 项
- **优化**：60 项
- **高级**：104 项

其中 `model_family`、`compile_dynamic_seq`、`compile_seq_bands`、`activation_memory_budget`、`train_adaln`、`prior_loss_weight` 是当前配置运行时补入“高级 / 其他可用参数”的字段；`dataset_config` 是页面的数据集工具卡，计入“必填 / 数据集设置”。少数字段在布局定义中跨分类复用；完整逐项表的“导航分类”列会同时列出这些归属。

### 全部参数八章节

八章节是“全部参数”画布当前实际使用的分类，也是下方 215 项清单的主分组。

| 章节 | 数量 | 占比 |
| --- | ---: | ---: |
| 模型与数据 | 7 | 3.3% |
| 核心训练 | 35 | 16.3% |
| 训练预览 | 6 | 2.8% |
| 显存与速度 | 43 | 20.0% |
| 数据与缓存 | 13 | 6.0% |
| 正则化 | 5 | 2.3% |
| 方法架构 | 60 | 27.9% |
| 实验与高级 | 46 | 21.4% |
| **合计** | **215** | **100.0%** |

八章节由更细的标签聚合而成：模型、数据、核心训练、训练量、输出、预览、显存与速度、数据与 VAE、缓存与预处理、数据高级、正则化、方法架构、LoKr、实验性、SPD、高级。

## 补充：旧分组页的默认暴露基线

2026-09-04 用户提供的三张旧分组页截图，用于约束四阶段页面的**默认显示范围**，不是新的字段总数快照。截图中的“显示高级配置”均未开启，因此这些字段不应依赖“全部高级选项”才能看到。

### 必填截图

截图顶部写作“显示 6 / 6 项”，但实际画面和当前配置契约可识别出 7 个配置键；数据集工具卡的历史计数口径没有把所有下方控件一致计入。默认面应保留：

`pretrained_model_name_or_path`、`qwen3`、`vae`、`dataset_config`、`use_shuffled_caption_variants`、`masked_loss`、`caption_dropout_rate`。

### 常用截图

截图明确显示 20 / 20 项：

`output_name`、`max_train_epochs`、`learning_rate`、`save_every_n_epochs`、`save_last_n_epochs`、`checkpointing_epochs`、`checkpointing_last_n_epochs`、`network_train_unet_only`、`network_dim`、`network_alpha`、`lora_adapter_kind`、`dora_wd`、`optimizer_type`、`lr_scheduler`、`timestep_sampling`、`discrete_flow_shift`、`log_every_n_steps`、`max_train_steps`、`train_batch_size`、`gradient_accumulation_steps`。

### 优化截图

顶部计数为 52 / 52；当前截图完整拍到其中“显存与速度优化”小节的 35 项，其余小节位于截图下方，不能据图补写。可确认的 35 项是：

`pipeline_parallel`、`pipeline_parallel_stages`、`pipeline_parallel_microbatches`、`pipeline_parallel_schedule`、`pipeline_parallel_split`、`blocks_to_swap`、`block_swap_transfer_dtype`、`block_swap_restore_mode`、`selective_checkpoint`、`selective_checkpoint_blocks`、`base_compute`、`block_swap_profile_jsonl`、`memory_probe_jsonl`、`memory_probe_max_steps`、`peak_probe_jsonl`、`peak_probe_max_steps`、`peak_probe_level`、`preprocess_vae_cache_batch_size`、`preprocess_text_cache_batch_size`、`preprocess_memory_profile`、`preprocess_precision_preference`、`reuse_dataset_cache_copy`、`reuse_vae_latents`、`reuse_text_encoder_cache`、`cache_fingerprint_mode`、`force_rebuild_preprocess_cache`、`gradient_checkpointing`、`precision_preference`、`lr_warmup_steps`、`unsloth_offload_checkpointing`、`disable_block_swap_for_eval`、`attn_mode`、`torch_compile`、`compile_block_scope`、`use_custom_down_autograd`。

截图还明确了默认呈现语义：

- 流水线并行、块交换、选择性重算、探针、预处理、缓存复用和 compile 属于通用训练/资源工作面，默认长期渲染。
- 父功能关闭时，通用子项保留位置并禁用，方便用户看见完整设置关系；不因关闭而从画布消失。
- Adapter 变体选择器常态渲染；Anima 可直接切换，Krea-2 / Z-Image 显示固定的 plain LoRA 及不可用原因。
- 模型族、方法或 Adapter 真正不相关的**专属参数分支**仍隐藏，例如普通 LoRA 下的 LoKr / VeRA 子项，Krea-2 / Z-Image 下的非 plain-LoRA 分支。
- 模式专属的成组参数仍按条件披露，例如 `base_compute=bf16` 时不显示 ConvRot 子项。

当前实现把旧存储 id `newcomer / beginner / all` 作为迁移兼容保留，但界面语义改为“精简 / 默认 / 全部适用”；无历史偏好时使用“默认”，并覆盖上述 35 个已确认优化字段。长期目标仍是由模型族、方法和能力规则决定适用集合，而不是按用户熟练度分级。

## 字段清单

说明：

- “当前值”来自页面表单快照，`（空）` 表示当前合并配置未提供有效显示值。
- “当前不可用”表示该字段仍被渲染，但在当前 `krea2_raw + lora` 条件下控件被禁用。
- “层级”是页面“隐藏高级配置项”控件使用的新人 / 初学者 / 高级层级，不等于五个导航分类中的“高级”。
- “默认值”是前端表单 UI 默认值，不一定等同于训练 CLI 默认值；仅在与当前值不同或有助核对时列入备注。

### 模型与数据（7 项）

| # | 中文名 / 配置键 | 当前值 | 细分类 | 导航分类 | 层级 | 状态 | 控件 | 说明 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 数据集配置<br>`dataset_config` | `configs/datasets/8-8-k2.toml` | 数据 | 必填 / 数据集设置 | 新人 | 可用、必填 | `dataset` | 当前训练使用的数据集配置 TOML。 |
| 2 | 基础模型路径<br>`pretrained_model_name_or_path` | `models/diffusion_models/krea2_raw_nf4_self_contained.safetensors` | 模型 | 必填 / 基础模型路径 | 新人 | 可用、必填 | `text` | 基础 DiT 模型权重路径，也就是 LoRA 要挂在哪个底模上训练。 |
| 3 | Qwen3 文本编码器路径<br>`qwen3` | `models/text_encoders/qwen3vl_4b_bf16.safetensors` | 模型 | 必填 / 基础模型路径 | 新人 | 可用、必填 | `text` | Qwen3 文本编码器路径，用来把 caption 和提示词变成模型能理解的条件。 |
| 4 | VAE 路径<br>`vae` | `models/vae/qwen_image_vae.safetensors` | 模型 | 必填 / 基础模型路径 | 新人 | 可用、必填 | `text` | VAE 模型路径，负责把图片和训练用 latent 互相转换。 |
| 5 | 使用打乱标题变体<br>`use_shuffled_caption_variants` | `true` | 数据 | 必填 / 数据集设置 | 新人 | 可用 | `toggle` | 训练时使用预处理生成的 caption 打乱变体。 UI 默认：`false`。 |
| 6 | 遮罩损失<br>`masked_loss` | `true` | 数据 | 必填 / 数据集设置 | 新人 | 可用 | `toggle` | 只在非遮罩区域计算损失。 UI 默认：`false`。 |
| 7 | 标题丢弃率<br>`caption_dropout_rate` | `0.1` | 数据 | 必填 / 数据集设置 | 新人 | 可用 | `number` | 每个样本丢弃 caption 的概率。 UI 默认：`0`。 |

### 核心训练（35 项）

| # | 中文名 / 配置键 | 当前值 | 细分类 | 导航分类 | 层级 | 状态 | 控件 | 说明 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 输出名称<br>`output_name` | `anima` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `text` | 保存权重文件时使用的文件名前缀。 |
| 9 | 最大训练轮数<br>`max_train_epochs` | `5` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `number` | 最大训练轮数，也就是数据集会被完整看多少遍。 |
| 10 | 学习率<br>`learning_rate` | `0.00002` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `number` | 学习率，决定每一步参数改动有多大。 |
| 11 | 模型保存间隔<br>`save_every_n_epochs` | `2` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `number` | 每隔多少轮保存一次普通模型权重。 |
| 12 | 权重保留数量<br>`save_last_n_epochs` | `-1` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `number` | 普通模型权重最多保留多少份。 |
| 13 | 训练状态保存间隔<br>`checkpointing_epochs` | `2` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `number` | 每隔多少轮保存一次可恢复训练状态。 |
| 14 | 续训点保留数量<br>`checkpointing_last_n_epochs` | `1` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `number` | 自动续训点最多保留多少份。 |
| 15 | 仅训练 DiT<br>`network_train_unet_only` | `true` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `toggle` | 只训练 DiT 主模型侧的适配器，冻结文本编码器。 |
| 16 | LoRA 秩<br>`network_dim` | `32` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `number` | LoRA 的容量大小，也叫 rank 或秩。 |
| 17 | LoRA Alpha<br>`network_alpha` | `32` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `number` | LoRA 的缩放强度，影响训练结果最终作用有多猛。 |
| 18 | LoRA 结构<br>`lora_adapter_kind` | `lora` | 核心训练 | 常用 / 常用训练设置 | 新人 | 当前不可用 | `select` | 常态显示 Adapter 变体；当前 Krea-2 固定为 plain LoRA，Anima 下可切换 LoRA / LoHa / LoKr / GLoRA / VeRA。 |
| 19 | 启用 DoRA<br>`dora_wd` | `false` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `toggle` | 在普通 LoRA 上启用 DoRA（Weight-Decomposed LoRA）。 |
| 20 | LoKr Factor<br>`lokr_factor` | `8` | 核心训练 | 常用 / 常用训练设置 | 新人 | 当前不可用 | `select` | LoKr 的 Kronecker 分解因子。 |
| 21 | VeRA 投影随机种子<br>`vera_projection_prng_key` | `0` | 核心训练 | 常用 / 常用训练设置 | 高级 | 当前不可用 | `select` | VeRA 冻结随机投影 A/B 的生成种子。 |
| 22 | VeRA d 初始值<br>`vera_d_initial` | `0.1` | 核心训练 | 常用 / 常用训练设置 | 高级 | 当前不可用 | `select` | VeRA 中 lambda_d 缩放向量的初始值。 |
| 23 | 保存 VeRA 投影矩阵<br>`vera_save_projection` | `false` | 核心训练 | 常用 / 常用训练设置 | 高级 | 当前不可用 | `toggle` | 是否把 VeRA 冻结随机投影矩阵也写进 checkpoint。 |
| 24 | 继续训练权重路径<br>`network_weights` | （空） | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `text` | 从已有适配器检查点热启动训练。 |
| 25 | 从权重读取秩<br>`dim_from_weights` | （空） | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `text` | 从热启动检查点读取 rank，而不是使用表单里的 network_dim。 |
| 26 | 优化器<br>`optimizer_type` | `AdamW` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `select` | 优化器算法。 |
| 27 | 优化器参数<br>`optimizer_args` | （空） | 核心训练 | 常用 / 常用训练设置 | 高级 | 可用 | `textarea` | 传给优化器的额外参数。 |
| 28 | 学习率调度<br>`lr_scheduler` | `cosine` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `select` | 学习率调度策略。 |
| 29 | 时间步采样<br>`timestep_sampling` | `sigmoid` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `select` | 训练时如何采样去噪时间步。 |
| 30 | Flow 偏移<br>`discrete_flow_shift` | `1` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `number` | flow matching 噪声调度偏移参数。 |
| 31 | 日志记录间隔<br>`log_every_n_steps` | `2` | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `number` | 每多少训练步记录一次日志。 |
| 32 | 日志目录<br>`logging_dir` | （空） | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `text` | 训练日志目录，主要给 TensorBoard 和历史曲线使用。 |
| 33 | 日志后端<br>`log_with` | （空） | 核心训练 | 常用 / 常用训练设置 | 新人 | 可用 | `select` | 训练日志后端。 |
| 34 | 最大训练步数<br>`max_train_steps` | `0` | 训练量 | 常用 / 步数与训练量 | 新人 | 可用 | `number` | 固定训练总步数，用 step 而不是轮数来控制训练多久。 |
| 35 | 批大小<br>`train_batch_size` | `1` | 训练量 | 常用 / 步数与训练量 | 新人 | 可用 | `number` | 每个训练 step 同时送进 GPU 的图片数量。 |
| 36 | 梯度累积步数<br>`gradient_accumulation_steps` | `1` | 训练量 | 常用 / 步数与训练量 | 新人 | 可用 | `number` | 累积多少个小批次后，再真正更新一次参数。 |
| 37 | 数据采样比例<br>`sample_ratio` | （空） | 训练量 | 常用 / 步数与训练量 | 新人 | 可用 | `select` | 每轮使用的数据比例。 |
| 38 | 模型保存格式<br>`save_model_as` | `safetensors` | 输出 | 高级 / 输出格式与训练范围 | 新人 | 可用 | `select` | 模型保存格式。 |
| 39 | 保存精度<br>`save_precision` | `bf16` | 输出 | 高级 / 输出格式与训练范围 | 新人 | 可用 | `select` | 保存权重时使用的精度。 |
| 40 | 权重衰减<br>`weight_decay` | （空） | 输出 | 高级 / 输出格式与训练范围 | 新人 | 当前不可用 | `text` | 设置优化器对可训练权重的衰减强度。 |
| 41 | 启用 CMMD 验证<br>`use_cmmd` | `false` | 输出 | 高级 / 输出格式与训练范围 | 新人 | 可用 | `toggle` | 在验证阶段启用 CMMD 生成质量指标。 |
| 42 | IP 诊断间隔<br>`ip_diagnostics_epochs` | `999` | 输出 | 高级 / 输出格式与训练范围 | 高级 | 当前不可用 | `number` | 设置 IP-Adapter 路由与条件诊断的轮次间隔。 |

### 训练预览（6 项）

| # | 中文名 / 配置键 | 当前值 | 细分类 | 导航分类 | 层级 | 状态 | 控件 | 说明 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 43 | 样张提示词文件<br>`sample_prompts` | （空） | 预览 | 预览 / 训练中预览图 | 新人 | 可用 | `textarea` | 训练过程中用来生成预览图的提示词。 |
| 44 | 按轮生成样张<br>`sample_every_n_epochs` | （空） | 预览 | 预览 / 训练中预览图 | 新人 | 可用 | `text` | 每隔多少轮生成一次预览图。 |
| 45 | 按步生成样张<br>`sample_every_n_steps` | （空） | 预览 | 预览 / 训练中预览图 | 新人 | 可用 | `text` | 每隔多少训练步生成一次预览图。 |
| 46 | 开始前生成样张<br>`sample_at_first` | `false` | 预览 | 预览 / 训练中预览图 | 新人 | 可用 | `toggle` | 训练开始前先生成一组初始样张。 |
| 47 | 样张采样器<br>`sample_sampler` | `euler` | 预览 | 预览 / 训练中预览图 | 新人 | 可用 | `select` | 训练中样张使用的采样器。 |
| 48 | 随机种子<br>`seed` | `42` | 预览 | 预览 / 训练中预览图；高级 / SPD CLI 实验 | 新人 | 当前不可用 | `number` | 设置数据打乱、噪声采样等随机过程的种子。 |

### 显存与速度（43 项）

| # | 中文名 / 配置键 | 当前值 | 细分类 | 导航分类 | 层级 | 状态 | 控件 | 说明 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 49 | 启用模型流水线并行<br>`pipeline_parallel` | `false` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `toggle` | 为两张 GPU 规划所选模型族的分层范围。 |
| 50 | 流水线阶段数<br>`pipeline_parallel_stages` | `2` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 当前不可用 | `select` | 决定把所选模型分到几张显卡上。 |
| 51 | 流水线微批数<br>`pipeline_parallel_microbatches` | `4` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 当前不可用 | `select` | 决定两张显卡每轮交替处理几份小批量。 |
| 52 | 流水线调度<br>`pipeline_parallel_schedule` | `1f1b` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 当前不可用 | `select` | 决定两张显卡怎样轮流工作。 |
| 53 | 流水线分层策略<br>`pipeline_parallel_split` | `balanced` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 当前不可用 | `select` | 选择当前模型的主块如何分配到两张卡。 |
| 54 | CPU/GPU 交换块数<br>`blocks_to_swap` | `26` | 显存与速度 | 优化 / 显存与速度优化 | 新人 | 可用 | `number` | 显存不够时，把一部分模型暂时放到电脑内存里。 |
| 55 | 块交换传输精度<br>`block_swap_transfer_dtype` | `bf16` | 显存与速度 | 优化 / 显存与速度优化 | 新人 | 可用 | `select` | 块交换 frozen base 权重在 CPU 侧保存和传输时使用的精度。 |
| 56 | 块交换恢复路径<br>`block_swap_restore_mode` | `slab` | 显存与速度 | 优化 / 显存与速度优化 | 新人 | 可用 | `select` | 块交换 restore 阶段如何把 frozen base 权重恢复回 GPU。 |
| 57 | 选择性重算<br>`selective_checkpoint` | `off` | 显存与速度 | 优化 / 显存与速度优化 | 新人 | 可用 | `select` | 只对部分 DiT 计算做 activation 重算。 |
| 58 | 定点重算块<br>`selective_checkpoint_blocks` | （空） | 显存与速度 | 优化 / 显存与速度优化 | 初学者 | 可用 | `text` | 定点重算的 DiT block 编号列表。 |
| 59 | 底模计算路径<br>`base_compute` | `nf4` | 显存与速度 | 优化 / 显存与速度优化 | 新人 | 可用 | `select` | 决定底模保持原精度，还是压缩后再计算。 UI 默认：`bf16`。 |
| 60 | ConvRot 组大小<br>`convrot_group_size` | `256` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 当前不可用 | `select` | ConvRot 分组大小（RHT 的 group size）。 |
| 61 | ConvRot 作用范围<br>`convrot_scope` | `mlp` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 当前不可用 | `select` | ConvRot 作用到哪些 Linear 模块。 |
| 62 | ConvRot Hadamard<br>`convrot_hadamard` | `sylvester` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 当前不可用 | `select` | Group RHT 的 Hadamard 构造（P0-D / 质量 opt-in）。 |
| 63 | ConvRot 最小 in_features<br>`convrot_min_in_features` | `0` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 当前不可用 | `number` | 按 in_features 下限过滤 ConvRot patch（P1-G）。 |
| 64 | 仅最大 in_features<br>`convrot_largest_in_features_only` | `false` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 当前不可用 | `toggle` | 在 scope 命中层里只 patch 最大 in_features 的 Linear（P1-G）。 |
| 65 | 大层 ConvRot 模式<br>`convrot_large_layer_mode` | （空） | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 当前不可用 | `select` | 大 in_features 层覆盖计算模式（P1-F 混精）。 |
| 66 | 大层 in_features 阈值<br>`convrot_large_min_in_features` | （空） | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 当前不可用 | `text` | 触发 large_layer_mode 的 in_features 阈值（P1-F）。 |
| 67 | 块交换 Profile<br>`block_swap_profile_jsonl` | `off` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `select` | 记录每个交换块的搬运和等待耗时。 |
| 68 | 显存探针<br>`memory_probe_jsonl` | `off` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `select` | 记录训练级 CUDA 显存和 adapter/optimizer 摘要。 |
| 69 | 探针步数<br>`memory_probe_max_steps` | `2` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `select` | 显存探针记录详细 step 快照的步数上限。 |
| 70 | 峰值探针<br>`peak_probe_jsonl` | `off` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `select` | 记录更细粒度的 DiT block / LoKr 峰值显存事件。 |
| 71 | 峰值探针步数<br>`peak_probe_max_steps` | `2` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `select` | 峰值探针记录详细事件的步数上限。 |
| 72 | 峰值探针粒度<br>`peak_probe_level` | `block` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `select` | 峰值探针的事件粒度。 |
| 73 | VAE 预处理批大小<br>`preprocess_vae_cache_batch_size` | `auto` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `select` | VAE latent cache 的批大小。 |
| 74 | 文本缓存批大小<br>`preprocess_text_cache_batch_size` | `auto` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `select` | 文本编码缓存的批大小。 |
| 75 | 预处理显存模式<br>`preprocess_memory_profile` | `auto` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `select` | 预处理阶段的显存/速度预设。 |
| 76 | 预处理精度<br>`preprocess_precision_preference` | `bf16` | 显存与速度 | 优化 / 显存与速度优化 | 初学者 | 可用 | `select` | 预处理阶段优先采用哪种计算精度。 |
| 77 | 复用数据集缓存拷贝<br>`reuse_dataset_cache_copy` | `false` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `toggle` | 允许复用数据集预处理阶段的已有拷贝。 UI 默认：`true`。 |
| 78 | 复用 VAE Latent 缓存<br>`reuse_vae_latents` | `false` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `toggle` | 复用已生成的 VAE latent 缓存。 UI 默认：`true`。 |
| 79 | 复用文本编码缓存<br>`reuse_text_encoder_cache` | `false` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `toggle` | 复用已生成的文本编码器特征缓存。 UI 默认：`true`。 |
| 80 | 缓存指纹模式<br>`cache_fingerprint_mode` | `light` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `select` | 决定预处理缓存如何判定源文件已变更。 |
| 81 | 强制重建预处理缓存<br>`force_rebuild_preprocess_cache` | `false` | 显存与速度 | 优化 / 显存与速度优化 | 新人 | 可用 | `toggle` | 强制忽略现有预处理缓存并全量重建。 |
| 82 | 梯度检查点<br>`gradient_checkpointing` | `true` | 显存与速度 | 优化 / 显存与速度优化 | 新人 | 可用 | `toggle` | 用更多计算换更低显存的训练开关。 |
| 83 | 精度倾向<br>`precision_preference` | `bf16` | 显存与速度 | 优化 / 显存与速度优化 | 新人 | 可用 | `select` | 训练时优先采用哪种数值精度方案。 |
| 84 | 预热步数<br>`lr_warmup_steps` | `0.05` | 显存与速度 | 优化 / 显存与速度优化 | 新人 | 可用 | `number` | 学习率预热步数。 |
| 85 | Unsloth 检查点卸载<br>`unsloth_offload_checkpointing` | `false` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `toggle` | 把梯度检查点卸载到 CPU 内存。 |
| 86 | 评估时暂停交换块<br>`disable_block_swap_for_eval` | `false` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `toggle` | 验证和训练中预览图阶段临时暂停块交换。 |
| 87 | 注意力后端<br>`attn_mode` | `flash` | 显存与速度 | 优化 / 显存与速度优化 | 新人 | 可用 | `select` | 注意力计算使用的后端实现。 |
| 88 | 启用 torch.compile<br>`torch_compile` | `true` | 显存与速度 | 优化 / 显存与速度优化 | 初学者 | 可用 | `toggle` | 是否让 PyTorch 先编译模型计算图再训练。 |
| 89 | 编译块范围<br>`compile_block_scope` | `resident` | 显存与速度 | 优化 / 显存与速度优化 | 初学者 | 可用 | `select` | block swap 开启时，哪些 DiT block 参与 torch.compile。 |
| 90 | Inductor 编译模式<br>`compile_inductor_mode` | （空） | 显存与速度 | 优化 / 显存与速度优化 | 初学者 | 可用 | `select` | Inductor 编译器优化模式。 |
| 91 | 自定义 Down 反向<br>`use_custom_down_autograd` | `false` | 显存与速度 | 优化 / 显存与速度优化 | 高级 | 可用 | `toggle` | 使用自定义 LoRA down 矩阵反向实现。 |

### 数据与缓存（13 项）

| # | 中文名 / 配置键 | 当前值 | 细分类 | 导航分类 | 层级 | 状态 | 控件 | 说明 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 92 | DataLoader 进程数<br>`max_data_loader_n_workers` | `0` | 数据与 VAE | 优化 / 数据加载与 VAE 资源 | 新人 | 可用 | `select` | 设置 DataLoader 并行工作进程上限。 |
| 93 | VAE 分块大小<br>`vae_chunk_size` | `64` | 数据与 VAE | 优化 / 数据加载与 VAE 资源 | 新人 | 可用 | `select` | VAE 解码/编码时的分块大小。 |
| 94 | 禁用 VAE 缓存<br>`vae_disable_cache` | `true` | 数据与 VAE | 优化 / 数据加载与 VAE 资源 | 高级 | 可用 | `toggle` | 禁用 VAE 内部缓存。 |
| 95 | 固定内存加载<br>`dataloader_pin_memory` | `true` | 数据与 VAE | 优化 / 数据加载与 VAE 资源 | 新人 | 可用 | `toggle` | DataLoader 是否使用 pinned memory。 |
| 96 | 常驻数据加载进程<br>`persistent_data_loader_workers` | `true` | 数据与 VAE | 优化 / 数据加载与 VAE 资源 | 新人 | 可用 | `toggle` | DataLoader worker 是否跨 epoch 常驻。 |
| 97 | 使用 VAE 缓存<br>`use_vae_cache` | `true` | 缓存与预处理 | 高级 / 缓存与预处理 | 新人 | 可用 | `toggle` | 使用 VAE latent 缓存。 |
| 98 | 使用文本缓存<br>`use_text_cache` | `true` | 缓存与预处理 | 高级 / 缓存与预处理 | 新人 | 可用 | `toggle` | 使用文本编码器输出缓存。 |
| 99 | 缓存 LLM 适配器输出<br>`cache_llm_adapter_outputs` | `true` | 缓存与预处理 | 高级 / 缓存与预处理 | 高级 | 可用 | `toggle` | 把 LLM adapter 输出缓存到磁盘。 UI 默认：`false`。 |
| 100 | 跳过缓存检查<br>`skip_cache_check` | `true` | 缓存与预处理 | 高级 / 缓存与预处理 | 高级 | 可用 | `toggle` | 启动时跳过缓存完整性检查。 UI 默认：`false`。 |
| 101 | IP 特征写入磁盘<br>`ip_features_cache_to_disk` | `false` | 缓存与预处理 | 高级 / 缓存与预处理 | 高级 | 可用 | `toggle` | 是否把 IP-Adapter 图像特征缓存到磁盘。 |
| 102 | 数据路径匹配<br>`path_pattern` | `*` | 数据高级 | 高级 / 更多数据集配置 | 高级 | 可用 | `text` | 限制数据集扫描时接受的图像路径模式。 |
| 103 | 过滤低分辨率图<br>`drop_lowres_images` | `true` | 数据高级 | 高级 / 更多数据集配置 | 高级 | 可用 | `toggle` | 是否过滤分辨率过低的训练图像。 |
| 104 | 最低像素数<br>`min_pixels` | `500000` | 数据高级 | 高级 / 更多数据集配置 | 高级 | 可用 | `select` | 定义低分辨率过滤的最小像素总数。 |

### 正则化（5 项）

| # | 中文名 / 配置键 | 当前值 | 细分类 | 导航分类 | 层级 | 状态 | 控件 | 说明 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 105 | 无数据集先验保留权重<br>`prior_preservation_weight` | `0` | 正则化 | 优化 / 无数据集正则化 | 高级 | 可用 | `number` | 无额外数据集的先验保留辅助损失权重。 |
| 106 | 空提示先验保留<br>`blank_prompt_preservation` | `false` | 正则化 | 优化 / 无数据集正则化 | 高级 | 可用 | `toggle` | 使用空提示 T5("") 作为先验保留条件。 |
| 107 | DOP 触发词<br>`diff_output_preservation_trigger` | （空） | 正则化 | 优化 / 无数据集正则化 | 高级 | 可用 | `text` | DOP/class prompt 先验保留的触发词。 |
| 108 | DOP 类提示<br>`diff_output_preservation_class` | （空） | 正则化 | 优化 / 无数据集正则化 | 高级 | 可用 | `text` | DOP/class prompt 先验保留的类提示。 |
| 109 | 反转遮罩先验权重<br>`inverted_mask_prior_weight` | `0` | 正则化 | 优化 / 无数据集正则化 | 高级 | 可用 | `number` | 只在遮罩外区域做先验保留的辅助损失权重。 |

### 方法架构（60 项）

| # | 中文名 / 配置键 | 当前值 | 细分类 | 导航分类 | 层级 | 状态 | 控件 | 说明 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 110 | LoKr 结构化 einsum<br>`lokr_use_einsum` | `true` | LoKr | 常用 / LoKr 专用优化 | 高级 | 当前不可用 | `toggle` | LoKr 是否使用结构化 einsum 计算路径。 |
| 111 | LoKr 轻量分解 W2<br>`lokr_decompose_w2` | `false` | LoKr | 常用 / LoKr 专用优化 | 高级 | 当前不可用 | `toggle` | 是否把大的 lokr_w2 再拆成 lokr_w2_a/lokr_w2_b。 |
| 112 | LoKr 全因子模式<br>`lokr_full_factor` | `true` | LoKr | 常用 / LoKr 专用优化 | 高级 | 当前不可用 | `toggle` | 显式声明 LoKr 全因子布局，并替代旧 network_dim=114514 哨兵。 |
| 113 | 允许旧 LoKr 哨兵 dim<br>`lokr_allow_legacy_dim` | `false` | LoKr | 常用 / LoKr 专用优化 | 高级 | 当前不可用 | `toggle` | 是否允许旧版 network_dim=114514 全因子哨兵。 |
| 114 | LoKr 分组<br>`lokr_factor_group_size` | `8` | LoKr | 常用 / LoKr 专用优化 | 高级 | 当前不可用 | `select` | LoKr 自定义反向一次计算的输出 factor 组数。 |
| 115 | LoKr 张量切块阈值<br>`lokr_project_chunk_bytes` | `4194304` | LoKr | 常用 / LoKr 专用优化 | 高级 | 当前不可用 | `select` | LoKr 投影内部 row chunk 的字节阈值。 |
| 116 | LoKr 分组 Delta 后端<br>`lokr_grouped_delta_backend` | `triton` | LoKr | 常用 / LoKr 专用优化 | 高级 | 当前不可用 | `select` | 选择 LoKr 分组 Delta 权重的计算后端。 |
| 117 | LoKr 分组 Delta 反向后端<br>`lokr_grouped_delta_backward_backend` | `triton_grad_w1_w2_grad_x` | LoKr | 常用 / LoKr 专用优化 | 高级 | 当前不可用 | `select` | 选择 LoKr 分组 Delta 的反向计算后端。 |
| 118 | 网络模块<br>`network_module` | `networks.lora_anima` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `select` | 训练时加载的网络实现模块。 |
| 119 | 网络额外参数<br>`network_args` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `textarea` | 传给 network_module 的额外参数列表。 |
| 120 | 启用 OrthoLoRA<br>`use_ortho` | `false` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `toggle` | 启用 OrthoLoRA，用正交参数化约束 LoRA 更新。 |
| 121 | 启用 T-LoRA<br>`use_timestep_mask` | `false` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `toggle` | 启用 T-LoRA，让有效 rank 随去噪时间步变化。 |
| 122 | 最小秩<br>`min_rank` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `select` | T-LoRA 在低噪声时间步保留的最小活跃 rank。 |
| 123 | 秩日程幂指数<br>`alpha_rank_scale` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | T-LoRA 有效 rank 日程的幂指数 α。 |
| 124 | 起始层<br>`layer_start` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | 从第几层开始应用 LoRA。 |
| 125 | 启用 ReFT<br>`add_reft` | `false` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `toggle` | 启用 ReFT，在 DiT 块残差流上添加可训练干预。 |
| 126 | ReFT 秩<br>`reft_dim` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `select` | ReFT 干预秩。 |
| 127 | ReFT Alpha<br>`reft_alpha` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `select` | ReFT 缩放因子。 |
| 128 | ReFT 层范围<br>`reft_layers` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `select` | 哪些 DiT 块启用 ReFT。 |
| 129 | MoE 结构<br>`use_moe_style` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `select` | 选择 MoE 专家结构，false 表示不用专家路由。 |
| 130 | 逐层路由<br>`route_per_layer` | `false` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `toggle` | 控制路由器是每层独立，还是全模型共享。 |
| 131 | 路由信号来源<br>`router_source` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `select` | 专家路由使用的信号来源。 |
| 132 | 专家数量<br>`num_experts` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `select` | MoE/Hydra/FeRA 的专家数量。 |
| 133 | 均衡损失权重<br>`balance_loss_weight` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | 专家负载均衡损失权重。 |
| 134 | 均衡损失预热比例<br>`balance_loss_warmup_ratio` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | 训练前多少比例的步数暂不启用均衡损失。 |
| 135 | 路由器学习率倍率<br>`network_router_lr_scale` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | 路由器学习率相对主学习率的倍率。 |
| 136 | 路由目标层<br>`router_targets` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | 限制哪些线性层参与路由适配的正则表达式。 |
| 137 | Sigma 特征维度<br>`sigma_feature_dim` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | sigma 路由器的时间步特征维度。 |
| 138 | 分桶均衡权重<br>`per_bucket_balance_weight` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | 每个 sigma 桶内部的额外负载均衡权重。 |
| 139 | Sigma 桶数量<br>`num_sigma_buckets` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `select` | 把时间步划分成多少个 sigma 桶。 |
| 140 | 按 Sigma 桶专门化专家<br>`specialize_experts_by_sigma_buckets` | `false` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `toggle` | 是否把专家硬分配给不同 sigma 桶。 |
| 141 | Sigma 桶边界<br>`sigma_bucket_boundaries` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | 自定义 sigma 桶边界。 |
| 142 | 启用 IP-Adapter<br>`use_ip_adapter` | `false` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `toggle` | 启用 IP-Adapter 图像条件训练。 |
| 143 | IP 图像条件丢弃率<br>`ip_image_drop_p` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | 训练时丢弃图像条件的概率。 |
| 144 | 验证基线对照<br>`validation_baselines` | `false` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `toggle` | 验证时额外运行方法专属的基线对照。 |
| 145 | 身份对采样模式<br>`ip_pair_mode` | `identity` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `select` | 选择 IP-Adapter 参考图与训练目标图的配对方式。 |
| 146 | 身份对采样概率<br>`ip_pair_prob` | `0.8` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `number` | 每个训练步使用异图身份对的概率。 |
| 147 | 身份对最低层级<br>`ip_pair_min_level` | `artist` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `select` | 身份对查找失败时允许回退到的最宽松层级。 |
| 148 | 身份对 Caption 去标概率<br>`ip_pair_caption_strip_p` | `0` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `number` | 异图配对步中移除目标 caption 里角色/作品标签的概率。 |
| 149 | 启用 EasyControl<br>`use_easycontrol` | `false` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `toggle` | 启用 EasyControl 图像条件方法。 |
| 150 | EasyControl 条件丢弃率<br>`easycontrol_drop_p` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | 训练时丢弃 EasyControl 条件的概率。 |
| 151 | EasyControl 条件噪声上限<br>`easycontrol_cond_noise_max` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | 给 EasyControl 条件图加入噪声的最大强度。 |
| 152 | FeRA 分带数<br>`fera_num_bands` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | FeRA 将 sigma/FEI 空间划分的带数。 |
| 153 | FEI 特征维度<br>`fei_feature_dim` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | FEI 路由特征维度。 |
| 154 | FEI 低 Sigma 除数<br>`fei_sigma_low_div` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | FEI 中低 sigma 区域的缩放除数。 |
| 155 | 路由器隐藏维度<br>`router_hidden_dim` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | 路由器隐藏层宽度。 |
| 156 | 路由温度<br>`router_tau` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | 路由温度，控制专家分配尖锐程度。 |
| 157 | FeRA FECL 权重<br>`fera_fecl_weight` | （空） | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `text` | FeRA 的 FECL 辅助损失权重。 |
| 158 | 启用 ChimeraHydra<br>`use_chimera_hydra` | `false` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `toggle` | 启用 ChimeraHydra 的内容专家池与频率专家池。 |
| 159 | 内容专家数<br>`num_experts_content` | `4` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 当前不可用 | `number` | 设置 ChimeraHydra 内容专家池的专家数。 |
| 160 | 频率专家数<br>`num_experts_freq` | `2` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 当前不可用 | `number` | 设置 ChimeraHydra 频率/FEI 专家池的专家数。 |
| 161 | 内容池均衡权重<br>`balance_w_content` | `0.000002` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 当前不可用 | `number` | 设置 ChimeraHydra 内容专家池的路由均衡权重。 |
| 162 | 频率池均衡权重<br>`balance_w_freq` | `0.000005` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 当前不可用 | `number` | 设置 ChimeraHydra 频率专家池的路由均衡权重。 |
| 163 | 内容路由学习率倍率<br>`network_content_router_lr_scale` | `10` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 当前不可用 | `number` | 缩放 ChimeraHydra 内容路由器相对于主学习率的更新速度。 |
| 164 | 频率路由学习率倍率<br>`network_freq_router_lr_scale` | `2` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 当前不可用 | `number` | 缩放 ChimeraHydra 频率路由器相对于主学习率的更新速度。 |
| 165 | 内容路由信号来源<br>`content_router_source` | `crossattn_emb` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `select` | 选择 ChimeraHydra 内容专家路由器的输入信号。 |
| 166 | 内容路由初始化标准差<br>`content_router_init_std` | `0.001` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `number` | ChimeraHydra 内容路由器权重的初始化标准差。 |
| 167 | 内容路由 LayerNorm<br>`content_router_layer_norm` | `false` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 可用 | `toggle` | 在送入 ChimeraHydra 内容路由器前对池化特征应用无参数 LayerNorm。 UI 默认：`true`。 |
| 168 | 频率路由初始化标准差<br>`freq_router_init_std` | `0.02` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 当前不可用 | `number` | 设置 ChimeraHydra 频率路由器初始权重的标准差。 |
| 169 | 频率路由 LayerNorm<br>`freq_router_layer_norm` | `false` | 方法架构 | 高级 / 方法内部与实验架构 | 高级 | 当前不可用 | `toggle` | 控制频率路由器在决策前是否对输入特征做 LayerNorm。 UI 默认：`true`。 |

### 实验与高级（46 项）

| # | 中文名 / 配置键 | 当前值 | 细分类 | 导航分类 | 层级 | 状态 | 控件 | 说明 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 170 | Sigmoid 缩放<br>`sigmoid_scale` | `1` | 实验性 | 优化 / 实验性功能 | 初学者 | 可用、实验性 | `number` | sigmoid/logit-normal 时间步采样的缩放系数。 |
| 171 | Sigmoid 偏置<br>`sigmoid_bias` | `0` | 实验性 | 优化 / 实验性功能 | 初学者 | 可用、实验性 | `number` | sigmoid/logit-normal 时间步采样的 logit 偏置。 |
| 172 | 损失权重方案<br>`weighting_scheme` | `uniform` | 实验性 | 优化 / 实验性功能 | 初学者 | 可用、实验性 | `select` | 按 sigma/SNR 给基础 flow-matching loss 加权。 |
| 173 | Min-SNR Gamma<br>`min_snr_gamma` | `5` | 实验性 | 优化 / 实验性功能 | 初学者 | 可用、实验性 | `number` | Min-SNR 权重方案的 gamma 上限。 |
| 174 | P2 Gamma<br>`p2_gamma` | `1` | 实验性 | 优化 / 实验性功能 | 初学者 | 可用、实验性 | `number` | P2 权重方案的指数强度。 |
| 175 | P2 k<br>`p2_k` | `1` | 实验性 | 优化 / 实验性功能 | 初学者 | 可用、实验性 | `number` | P2 权重方案的 SNR 偏移项。 |
| 176 | 速度方向损失权重<br>`velocity_direction_loss_weight` | `0` | 实验性 | 优化 / 实验性功能 | 初学者 | 可用、实验性 | `number` | FasterDiT 风格的速度方向辅助损失权重。 |
| 177 | SPD DiT 模型路径<br>`dit_path` | `models/diffusion_models/anima-base-v1.0.safetensors` | SPD | 高级 / SPD CLI 实验 | 高级 | 当前不可用、实验性 | `text` | 指定 SPD 实验使用的 DiT 底模权重路径。 |
| 178 | SPD 数据目录<br>`data_dir` | `post_image_dataset/lora` | SPD | 高级 / SPD CLI 实验 | 高级 | 当前不可用、实验性 | `text` | 指定 SPD 实验使用的数据目录。 |
| 179 | SPD 迭代步数<br>`iterations` | `2000` | SPD | 高级 / SPD CLI 实验 | 高级 | 当前不可用、实验性 | `number` | 设置 SPD 实验内部优化或迭代次数。 |
| 180 | 通道缩放 Alpha<br>`channel_scaling_alpha` | `0.5` | SPD | 高级 / SPD CLI 实验；高级 / 方法内部与实验架构 | 高级 | 当前不可用、实验性 | `number` | 设置 EasyControl 条件通道缩放的 Alpha。 |
| 181 | Soft Tokens 层数<br>`n_layers` | `10` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `number` | 设置 Soft Tokens 注入或调制的模型层数。 |
| 182 | 时间桶数量<br>`n_t_buckets` | `100` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `number` | 设置 Soft Tokens 按时间步/sigma 分组的桶数。 |
| 183 | Soft Tokens 初始化标准差<br>`init_std` | `0.02` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `number` | 设置 Soft Tokens 可学习向量的初始化标准差。 |
| 184 | Soft Tokens 拼接位置<br>`splice_position` | `end_of_sequence` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `select` | 选择 Soft Tokens 在文本 Token 序列中的拼接位置。 |
| 185 | 对比损失权重<br>`contrastive_weight` | `0` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `number` | 设置对比损失加入总损失的权重。 |
| 186 | 对比负样本数<br>`contrastive_k` | `1` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `number` | 设置每个样本用于对比学习的负样本数量。 |
| 187 | 对比损失触发间隔<br>`contrastive_every_n` | `1` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `number` | 设置每隔多少个训练 step 计算一次对比损失。 |
| 188 | 对比负样本模式<br>`contrastive_negative_mode` | `shuffled` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `select` | 选择对比学习如何构造负样本。 |
| 189 | 对比目标函数<br>`contrastive_objective` | `infonce` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `select` | 选择对比分支使用的损失目标。 |
| 190 | Jaccard 负样本惩罚<br>`contrastive_jaccard_alpha` | `1` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `number` | 设置根据标签 Jaccard 重叠度调整负样本惩罚的强度。 |
| 191 | InfoNCE 温度<br>`contrastive_tau` | `0.5` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `number` | 设置 InfoNCE 相似度 softmax 的温度。 |
| 192 | 对比损失预热比例<br>`contrastive_warmup_ratio` | `0.1` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `number` | 设置对比损失从弱到完整权重的预热比例。 |
| 193 | SoftRank softness<br>`softrank_softness` | `0.1` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `number` | 设置 SoftRank 排序近似的平滑程度。 |
| 194 | SoftRank 方法<br>`softrank_method` | `neuralsort` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `select` | 选择 SoftRank 对候选相似度排名的可微近似方法。 |
| 195 | 双 Soft Tokens Bank<br>`dual_bank` | `false` | 高级 | 高级 / Soft Tokens 参数 | 高级 | 当前不可用 | `toggle` | 控制 Soft Tokens 是否使用两组可学习 Token Bank。 |
| 196 | IP 视觉编码器<br>`encoder` | `pe` | 高级 | 高级 / IP-Adapter 高级参数 | 高级 | 当前不可用 | `select` | 选择 IP-Adapter 提取图像特征的视觉编码器。 |
| 197 | IP 编码器维度<br>`encoder_dim` | `1024` | 高级 | 高级 / IP-Adapter 高级参数 | 高级 | 当前不可用 | `number` | 设置 IP-Adapter 视觉编码器输出特征维度。 |
| 198 | IP Resampler 层数<br>`resampler_layers` | `2` | 高级 | 高级 / IP-Adapter 高级参数 | 高级 | 当前不可用 | `number` | 设置 IP-Adapter Resampler 的变换层数。 |
| 199 | IP Resampler 头数<br>`resampler_heads` | `8` | 高级 | 高级 / IP-Adapter 高级参数 | 高级 | 当前不可用 | `number` | 设置 IP-Adapter Resampler 的注意力头数。 |
| 200 | IP 条件强度<br>`ip_scale` | `1` | 高级 | 高级 / IP-Adapter 高级参数 | 高级 | 当前不可用 | `number` | 设置 IP-Adapter 图像条件特征的总体强度。 |
| 201 | IP Gate 学习率<br>`gate_lr` | `0.001` | 高级 | 高级 / IP-Adapter 高级参数 | 高级 | 当前不可用 | `number` | 设置 IP-Adapter 条件 gate 参数的独立学习率。 |
| 202 | 启用 PE-LoRA<br>`pe_lora_enabled` | `false` | 高级 | 高级 / IP-Adapter 高级参数 | 高级 | 当前不可用 | `toggle` | 控制是否对 PE-Core/位置编码链路注入 LoRA。 |
| 203 | PE-LoRA 秩<br>`pe_lora_rank` | `16` | 高级 | 高级 / IP-Adapter 高级参数 | 高级 | 当前不可用 | `number` | 设置 PE-LoRA 分支的秩。 |
| 204 | PE-LoRA Alpha<br>`pe_lora_alpha` | `16` | 高级 | 高级 / IP-Adapter 高级参数 | 高级 | 当前不可用 | `number` | 设置 PE-LoRA 分支的输出缩放 Alpha。 |
| 205 | PE-LoRA 起始层<br>`pe_lora_layer_from` | `8` | 高级 | 高级 / IP-Adapter 高级参数 | 高级 | 当前不可用 | `number` | 设置从 PE-Core 的哪一层开始注入 LoRA。 |
| 206 | 条件注意力初始门控<br>`b_cond_init` | `-10` | 高级 | 高级 / EasyControl 高级参数 | 高级 | 当前不可用 | `number` | 设置 EasyControl 条件注意力门控偏置的初始值。 |
| 207 | EasyControl 条件强度<br>`cond_scale` | `1` | 高级 | 高级 / EasyControl 高级参数 | 高级 | 当前不可用 | `number` | 设置 EasyControl 条件分支注入主模型的强度。 |
| 208 | EasyControl FFN LoRA<br>`apply_ffn_lora` | `true` | 高级 | 高级 / EasyControl 高级参数 | 高级 | 当前不可用 | `toggle` | 控制 EasyControl 是否同时对 FFN 层注入 LoRA。 |
| 209 | EasyControl 条件 Token 数<br>`cond_token_count` | `4096` | 高级 | 高级 / EasyControl 高级参数 | 高级 | 当前不可用 | `number` | 设置 EasyControl 条件分支产生的 Token 数量。 |
| 210 | 模型家族<br>`model_family` | `krea2_raw` | 高级 | 高级 / 其他可用参数 | 高级 | 可用 | `text` | 选择当前配置所训练的底模家族和对应运行时路由。 |
| 211 | 动态令牌序列编译<br>`compile_dynamic_seq` | `true` | 高级 | 高级 / 其他可用参数 | 高级 | 可用 | `toggle` | 让 Anima 的多种图片尺寸共用一套编译结果。 UI 默认：`false`。 |
| 212 | 分带动态序列编译<br>`compile_seq_bands` | `false` | 高级 | 高级 / 其他可用参数 | 高级 | 可用 | `toggle` | 把 Anima 的动态 token 长度拆成多个紧凑分带编译。 |
| 213 | 激活内存预算<br>`activation_memory_budget` | `1` | 高级 | 高级 / 其他可用参数 | 高级 | 可用 | `number` | 限制 torch.compile AOT 分割器为反向传播保存的激活预算。 |
| 214 | 训练 AdaLN 调制层<br>`train_adaln` | `false` | 高级 | 高级 / 其他可用参数 | 高级 | 可用 | `toggle` | 除常规 attention/MLP 外，也训练 DiT block 的 AdaLN 调制投影适配器。 |
| 215 | 正则化损失权重<br>`prior_loss_weight` | `1` | 高级 | 高级 / 其他可用参数 | 高级 | 可用 | `number` | 正则化图像的损失权重系数。 |

## 分类与维护来源

- Next 阶段名称及字段分组：`web/frontend-next/src/features/training-config/stageGroups.ts`、`resourceGroups.ts`
- 字段键、默认值与输入类型：`web/frontend-next/src/features/training-config/domain/config-field-catalog.js`、`defaults.js`、`config-field-types.js`
- 当前模型/方法可用性：`web/frontend-next/src/features/training-config/domain/config-field-availability.js`
- 中文名称与短说明：`web/frontend-next/src/features/training-config/domain/field-help-summary.js`
- 页面运行时字段筛选、适用性与编辑：`web/frontend-next/src/features/training-config/fieldCatalog.ts`、`TrainingFieldEditor.tsx`

当上述目录、当前配置文件、preset 或模型族发生变化时，应重新生成本快照，并重新核对 215 与各章节计数；不要把 215 写成永久不变量。
