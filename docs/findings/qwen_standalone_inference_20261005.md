# Qwen Image 2.1 独立 T2I / Edit 推理接线（2026-10-05，10 月 6 日追加）

状态：验收记录 / 阶段快照

Qwen Image 2.1 已接入独立 `inference.py` 单图入口：`single`、T2I / Edit、Euler、BF16、
最多一个 plain LoRA。Edit 接受 1–4 张有序参考图。runner 依次加载和释放 Qwen3-VL、
参考图 VAE、DiT + LoRA，最后使用专属 64-channel/16x VAE 和 mean/std 反归一化保存图片。
完整权重真机证据包含 T2I、单参考 Edit，以及四参考/CFG2/两步 Edit 和保存 latent
后的专属再解码。早期 CPU 文本编码超时在接入已有自动 GPU offload 后完成。
image-test、batch/interactive、LoKr/LoHa 推理及 compile 仍显式拒绝。

2026-10-06 完成审计：下文 `/tmp` 出图、latent、manifest 及原始日志当前已不存在，
持久化输出目录中未找到对应副本。源码的独立入口、single/Euler、plain LoRA 和
1–4 参考图边界已重新核对；历史真机结果只能追溯本报告，当前无法重新核对
图片、latent 或像素 hash。本次未重新执行 GPU 推理，也未将历史出图记为新验收。

实现入口：

- `library/models/qwen_image_2_1/inference_runner.py`
- `library/models/qwen_image_2_1/inference_conditions.py`
- `library/models/qwen_image_2_1/inference_latents.py`
- `library/models/qwen_image_2_1/inference_text_encoder.py`
- `library/inference/family_runtime.py`
- `library/models/family_registry.py`
- `inference.py`

验证结果：Qwen runner、family registry/fail-closed、adapter capability 定向测试共 16 项通过；
相关 Python 文件 `py_compile`、Ruff 和 `git diff --check` 通过。像素尺寸要求 `%32==0`（16x
VAE stride 加 2x2 token patch），采样步数必须为正，避免把非法尺寸或 `1 / steps` 错误推迟到
采样内核。

## 真实单图 smoke

完整 32-block NF4 权重在 CMP 90HX 10 GiB 上完成独立 T2I 的真实单图链路：CPU Qwen3-VL
编码、GPU DiT Euler 采样、采样后释放 DiT、GPU Qwen VAE 解码并保存 PNG，退出码为 `0`。

| 项目 | 本次结果 |
| --- | --- |
| DiT / compute | 全 Linear NF4 / BF16 |
| 文本编码器 / VAE | Qwen3-VL 8B BF16（CPU）/ Qwen Image 2.1 VAE BF16（GPU） |
| 输入 | `a small red bird on a branch`，空 negative prompt，seed=7 |
| 采样 | 512×512，1 Euler step，CFG=1.0，torch attention，无 adapter |
| 图片 | `/tmp/qwen_standalone_t2i_smoke/20261005-204235-098_7.png` |
| 文件核验 | Pillow `verify()` 通过，512×512、RGB、294592 bytes |
| 日志 | `/tmp/qwen_standalone_t2i_smoke_gpu_decode_20261005.log` |

初次 smoke 因 runner 将 VAE 设备硬编码为 CPU，在解码阶段达到 600 秒时限，退出码 `124`，
未生成图片；原日志保留在 `/tmp/qwen_standalone_t2i_smoke_20261005.log`。现在 VAE 使用
传入的运行设备，仍保持 DiT 释放后才加载 VAE 的顺序；解码在 `torch.inference_mode()` 内
执行。设备修复后的 runner 定向测试 `6 passed`，包含 CPU 和非默认 `cuda:2` 的设备选择
契约；Ruff、Python 语法检查通过。这里的 `cuda:2` 用例使用 mock，不是真实第三张卡验收。

## 真实 plain LoRA 加载与采样

追加使用既有 T2I adapter
`output/runs/qwen21-variant-hot-acceptance/lora-smoke/checkpoints/qwen21_lora_smoke.safetensors`，
其 metadata 声明 `ss_model_family=qwen_image_2_1`、`ss_network_spec=lora`、rank/alpha=16。
NF4 基座、文本编码器/VAE、prompt、seed、分辨率和采样参数与无 adapter smoke 相同，
multiplier=1.0。加载 384 个 checkpoint 张量、挂载 128 个 plain LoRA 模块，真实采样、
GPU VAE 解码和保存全部完成，退出码 `0`。

- 图片：`/tmp/qwen_standalone_t2i_lora_smoke/20261005-224510-112_7.png`。
- Pillow `verify()` 通过，512×512、RGB、294647 bytes。
- SHA256：`6964c76afa72c7d01b82603f5e880f900378235a21ac543bd32415a5a5d0e08d`。
- 日志：`/tmp/qwen_standalone_t2i_lora_smoke_20261005.log`，包含 plain LoRA attached、
  sampling、VAE decoding on cuda、image saved、Done。
- 与相同 seed 的无 adapter 图片比较，262144 个像素中 164748 个发生改变，通道平均绝对
  差为 0.29342（8-bit）、最大差为 7。该差异支持加载路径生效，不是图片质量结论。

验收前修复 loader 的静默缺键问题：原先 `strict=False` 且仅 debug unexpected keys。
现在检查 checkpoint 的 DiT-only 键、必需 alpha，以及 factory 生成的未来 state_dict 的
完整键和 shape，在 `apply_to` 前拒绝破损 checkpoint，随后 `strict=True` 载入。
真实小 Qwen factory 的定向 CPU 测试含完整加载、缺 alpha/up/down、多余键、非 DiT 键、
shape 不符、错误/缺失 family 及 unsupported spec；连同既有 runner 测试共 `18 passed`。
Ruff、Python 语法检查和指定路径 diff-check 通过。不同代理独立重跑同组测试，
`18 passed`（10.70 秒），审阅未来 state_dict 与 apply 时真实注册一致、破损文件不修改
原 forward，未发现实际缺陷。真实 T2I adapter 在该校验下出图通过。

以上 T2I 单步 smoke 仅确认真实加载与出图链路，未验收 50 步图片质量、较大分辨率的
解码显存或长期稳定性。Edit 的独立验收见下文。

## 独立 Edit 接线与真实单参考图

新增 `--qwen_image_2_1_task edit` 和互斥的 `--reference_image` / `--reference_images`，
参考图数量限制为 1–4，保留传入顺序。图像采用受限读取、EXIF 方向修正和 RGB 转换，
一次 BICUBIC 缩放后的相同 PIL 对象用于正负 Qwen3-VL 条件和参考图 VAE。
条件构建检查 image slots、正负 mask、64-channel latent 和偶数 latent 网格；
其他模型族拒绝 Qwen task/reference flags。旧 T2I 调用继续兼容。

独立 CPU 审查重跑推理/Edit 定向测试 `35 passed`，family/preview 补充 `7 passed`；
真实小 Diffusers Qwen 2.1 在 inference mode 下完成 1–4 参考、BF16、两步 CFG/Euler
有限值检查，真实 Qwen2VL processor 的 image slots 匹配。小模型不能替代完整权重验收。

CMP 90HX 的完整 32-block NF4、BF16、torch attention，使用真实 eager Edit 单步权重
`/tmp/qwen_nf4_edit_eager/nf4-edit-eager-smoke.safetensors` 和一个真实参考图，完成
512×512、Euler 1 step、CFG=1、seed=7 的真实 CLI，exit `0`，耗时 `119.01` 秒。
参考 latent 为 `[1,64,38,28]`，image slots 为 266；生成 latent 为 `[1,64,32,32]`，
全部 finite，范围 `[-4.1875,4.0625]`。输出为：

- `/tmp/qwen_standalone_edit_smoke_20261006/20261006-010547-041_7.png`。
- Pillow `verify()` 通过，512×512、RGB，SHA256 为
  `e48f9e707274b8de1ef1c03501eba5610ace8b115170dbf93017256aabb70c6f`。
- 同目录 `manifest.json`、`events.jsonl`、`stdout.log`、`stderr.log` 保存完整命令与阶段证据。

weakref 监控确认 TE/free → reference VAE/free → DiT + LoRA → sampling → DiT/free
→ output VAE/free；输出 VAE 加载前 CUDA allocated 约 8.65 MB。参考图 hash/尺寸/mode
保持不变，90HX 结束时为 17 MiB/0%。这只证明单参考单步链路，不是 Edit 画质验收。

追加四张有序真实参考图、512×512、Euler 2 steps、CFG=2、`latent_images` 的 CLI，
在 CPU Qwen3-VL 编码阶段达到 300 秒时限，exit `124`，耗时 `300.51` 秒。
未到参考 VAE/DiT 阶段，未生成 latent/PNG，计划的保存后再解码与像素比较未执行。
未降条件或重跑；源图和源码 hash 不变，90HX 恢复 17 MiB、无 probe 进程。
证据：`/tmp/qwen_edit_multi_ref_roundtrip_20261006/manifest.json` 及同目录日志。
该超时不构成多参考 Edit 不可行的结论，也不构成其完整权重验收通过。

## Qwen latent 专属解码与独立审查

审查发现旧 `--latent_path` 分支绕过 Qwen validator 并调用共享 Anima VAE。现已在
`main()` 中分派到 Qwen 专属 helper，仅读取 latent 并加载 Qwen 输出 VAE；不会加载
TE、DiT 或 adapter。全部文件先检查浮点/finite `[1,64,H,W]`、16x 尺寸 metadata，
然后逐文件恢复 seed/prompt/negative prompt，使用独立 args 副本防止元数据污染。
safetensors 要求 `latent` 键，其他 tensor 文件使用 `weights_only=True`；
Edit/reference、LoRA、prompt-file/interactive 和不支持的精度/sampler/compile 参数
在 VAE 加载前拒绝。PNG 名称含输入 stem，输出类型保持 images/latent/latent_images。

不同于实现者的代理重跑三份定向测试 `56 passed`，Ruff、Pyright、指定路径
diff-check 全部 exit `0`。独立 `/tmp/qwen_latent_review_probe.py` 还确认非法 seed、Inf、
字典或不安全对象 `.pt`、缺失文件均在加载 VAE 前拒绝；解码/保存异常只清理一次。
保存采样 latent 后的完整权重 CLI 再解码像素对照已在下一节完成；CPU 证据本身不代替
GPU 数值、显存或图像质量验收。

## 自动 TE offload 与四参考图再解码

独立推理原先直接把完整 TE 放 CPU 或 GPU，遗漏训练缓存已有的低显存 runtime。
现在 T2I/Edit 统一严格 CPU load，再复用 `loaded_text_encoder_for_cache`：默认 CUDA
按空闲显存预算选择 core GPU 或 Accelerate CPU offload，unused lm_head 留 CPU。
显式 `--text_encoder_cpu` 和 CPU device 继续只在 CPU 编码；正负条件单独执行，
退出 encoder stage 后才加载参考 VAE/DiT。12 项初始设备/异常测试和相关入口共 78 项
通过；后续独立审查与异常清理修复另见下段。

保持此前四参考图 CLI 的有序输入、512×512、Euler 2 steps、CFG=2、seed=7、同 NF4
基座和真实 Edit plain LoRA、`latent_images`，只改用默认 CUDA auto offload。真正
`inference.py __main__` Edit exit `0`，45.203 秒；专属 `--latent_path` 再解码 exit `0`，
13.398 秒；总计 58.924 秒。此次编码不再因 300 秒 CPU 时限中止。

- TE strict CPU load；execution device=`cuda`、auto offload=`true`、lm_head=`cpu`；
  Accelerate hooks 577→0，TE weakref 释放。
- 正负条件 image slots 均为 1007；四参考 latent 分别 `[1,64,38,28]` 和三个
  `[1,64,38,26]`，均 finite。
- TE/free → reference VAE/free → DiT + LoRA → sampling → DiT/free → output VAE/free
  顺序通过；decode CLI 仅读取 latent 和加载输出 VAE。
- 保存 latent `[1,64,32,32]`，512×512 metadata；两张 RGB PNG 均通过 Pillow verify，
  像素逐字节相同，pixel SHA256 为
  `9b2a163348965044d55fe69253ed343ddbd6436f78ea06f217eb84cad2cfe857`。
- 90HX 结束为 17 MiB、无 compute 进程；源码和四张参考图 hash 不变。

产物根 `/tmp/qwen_edit_offload_multi_ref_roundtrip_20261006/`：
`manifest.json` 和 `events.jsonl` 保存完整命令/阶段/hook/释放证据，
`edit/20261006-021749-406_7.png` 为原图，`decode/` 内 PNG 为再解码结果。
这里只验收该四参考/512²/两步组合的功能，不评价画质或外推长测、分辨率及其他硬件。

独立 CPU 审查还发现 Accelerate 部分安装失败可留下无 hook 的 meta 参数，清理时
`model.to('cpu')` 会遮盖原始错误。共享 runtime 现逐项清理；残留 meta 不盲目复制，
初始化/编码异常保留 identity 并附 secondary notes，正常清理失败仍明确报错。
真实 CPU Accelerate partial init、持续 child detach failure、正常和 encode error
独立 probe 通过；117 项 runtime/inference/cache-policy/preview 定向检查通过。
调用层 allocator/GC 清理同样保留原异常，且不长期存储异常对象以免形成 encoder
引用环；对应回归覆盖正常错误报告、两步清理都尝试及编码失败的弱引用释放。
最终修复经不同代理独立验收：关闭自动 GC、没有外部 exception 引用时，真实
shared CPU runtime 与 Accelerate CPU hooks 的 encoder 在函数退出后立即释放，
未依赖事后 collection；原始异常 identity 与 traceback 保留，secondary failures
进入 notes。runtime/调用层共 33 项测试以及三份独立 probe、Ruff/Pyright 均通过。
报告为 `/tmp/qwen_encoder_cleanup_acceptance_v2_20261006.md`。这里不覆盖完整
loader 返回前失败或真实 CUDA allocator 故障；正常 CUDA offload 的出图证据见上文。
