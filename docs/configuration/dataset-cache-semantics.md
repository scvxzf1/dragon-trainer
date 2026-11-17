# 数据集缓存语义

状态：稳定

Next `/next/datasets` 数据集编辑器和训练 runtime 共享以下契约：

- `mask_mode` 是唯一的遮罩模式字段：`none`、`external`、`embedded`、`auto`。
- `external` 只表示外部遮罩目录；缺少某张图的遮罩时，训练按全 1 mask 继续，latent NPZ 不要求 `alpha_mask`。
- `embedded` 要求 latent NPZ 包含当前桶尺寸的 `alpha_mask_<HxW>`；`none` 不读取遮罩；`auto` 仅为旧配置保留默认目录发现。
- latent 和文本缓存后缀来自 `library/models/family_registry.py`（及各族 strategy）：Anima 使用 `_anima.npz` + `_anima_te.safetensors`，Krea-2 使用 `_anima.npz` + `_krea2_te.safetensors`，Z-Image 使用 `_z_image.npz` + `_z_image_te.safetensors`，Qwen Image 2.1 使用 `_qwen_image_2_1.npz` + `_qwen_image_2_1_te.safetensors`，MOSS-Image 使用 `_moss_image.npz` + `_moss_image_te_nf4.safetensors`（切换 `moss_te_quantization="bf16"` 时为 `_moss_image_te_bf16.safetensors`）。
- runtime preflight 按每张训练图解析镜像缓存路径并复用训练策略 validator；部分缓存或无效缓存直接阻止 runtime 启动。
- `ip_features_cache_to_disk=false` 时不检查 PE sidecar；启用磁盘缓存时按 `ip_encoder` 检查 `{stem}_anima_{encoder}.safetensors`。
- 普通训练配置由 Web 编排先启动预处理，完成后自动进入训练；训练进程不在缺缓存时临时加载编码器补缓存，以保持懒加载顺序。

编辑器保存会保留未展示的 dataset/subset/custom attribute 字段，避免只编辑路径或分桶设置就静默删除训练参数。

## 模型身份与强制重建

预处理 sidecar 会写入一份「模型身份」记录（`library/cache_identity.py`），用于判断缓存是否仍由当前权重产生：

- 文本缓存 `{stem}_*_te.safetensors`：文本编码器路径身份、tokenizer、T5 tokenizer、vocab pack、DOP trigger/class；启用 `cache_llm_adapter_outputs` 时还包括提供 LLM Adapter 的 DiT 路径身份。
- latent 缓存 `{stem}_{W}x{H}_*.npz`：VAE 路径身份和模型族（模型族与 latent space 一一对应，不跨族比较）。

判定规则：

| 记录状态 | 行为 |
| --- | --- |
| 无身份记录（旧缓存） | 默认继续复用，每个缓存类型只警告一次；开启严格检查时按失效处理 |
| 身份与当前权重一致 | 复用 |
| 身份与当前权重不一致 | 失效：预处理重新编码该 sidecar，训练侧直接拒绝它 |

- 训练入口 `--require_cache_identity`：连未打戳的旧缓存也视为失效，避免继承来源不明的缓存。
- 预处理子命令（TE / latent）同样接受 `--require_cache_identity`；在配置里设 `require_cache_identity = true` 时，preprocess 任务会把它转发给对应子进程。
- 配置了 `vocab_pack_path` 时，预处理和训练都会自动按严格模式处理：vocab pack 变化不会改变 caption mtime，必须靠身份记录发现。
- 需要无条件重建时使用 preprocess 的 `--overwrite`，或直接使用新的缓存目录。
- `--skip_cache_check` 仍是内容/schema 校验的逃生口；但配置了 `require_cache_identity`（或 `vocab_pack_path`，它会自动开启严格模式）时身份校验优先生效，避免换权重后静默复用旧 sidecar。

路径身份使用 size + mtime；目录身份使用文件清单（相对路径、大小、mtime），不读取权重内容。vocab pack 额外做内容 sha256，因此可以发现同尺寸、同 mtime 的就地修改；其他权重文件若被同尺寸同 mtime 的就地替换，身份仍然无法识别，这类来源不确定的 checkpoint 应换缓存目录或走 `--overwrite`。

更换文本编码器、VAE、tokenizer 或 vocab pack 后不需要手动删除缓存：重跑预处理即可让失效 sidecar 重新编码，训练侧也会拒绝旧的 sidecar 而不是静默使用。

## Anima vocab pack 配置选择

当前配置键为 `vocab_pack_path`，可写在 `configs/base.toml`、方法配置或 WebUI 训练草稿中。默认值保持空字符串：

```toml
vocab_pack_path = ""
```

空值表示不选择额外 pack，不改变既有 Anima tokenizer/runtime 行为；它不是 `jp_v1` 的默认启用。只有在实际 pack runtime 已安装并且预处理、训练都能访问同一资源时，才应填写例如：

```toml
vocab_pack_path = "models/vocab_packs/anima_cjk_vocab_pack_jp_v1"
```

配置后，`preprocess` / `preprocess-te` 会把该路径转发给 Anima TE 缓存脚本，训练入口也会将其写入缓存身份。WebUI 只在 `model_family = "anima"` 时显示该字段；切换到其他模型族时隐藏。当前字段完成的是缓存身份与强制失效接线，**不代表 jp_v1 的 tokenizer/glyph runtime 已完成接入或热验收**。

选择 pack 后建议先运行：

```bash
python tasks.py preprocess-te
```

不要把本机绝对路径或尚未存在的 pack 路径写入版本控制的默认方法配置；本机路径应通过外置配置、方法覆盖或 WebUI 草稿提供。
