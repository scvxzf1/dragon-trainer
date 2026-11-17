# 多模型族 BF16/NF4 混合基座

状态：实验，2026-10-03。三个新增模型族的格式、加载与 CUDA 小模型训练验收通过；
尚未转换这三族的完整权重，也未做真实数据集热训练、采样质量或显存收益对照。

相关代码：`library/quantization/`、三个模型族的 `weights.py`、
`scripts/quantization/convert_mixed_nf4.py`。

## 策略与现有配置

混合格式通过现有 `pretrained_model_name_or_path` 自动识别，无新增训练配置项，
不修改已有用户配置。计算仍为 BF16，LoRA 可训练参数保持既有精度。
这与 `base_compute=nf4` 的 Krea 全 Linear 量化入口不同：Z-Image/Qwen 的该配置值
仍拒绝，不能用它启用此格式。

| 模型族 | NF4 模块 | 保留 BF16 |
| --- | --- | --- |
| Anima | 原有49个MLP矩阵固定策略 | 原热点层与其他模块，文件格式不变 |
| Krea-2 Raw | `blocks.*.mlp.{gate,up,down}` | attention、embedding、调制、norm、输入输出等 |
| Z-Image | `layers.*.feed_forward.{w1,w2,w3}` | attention、全部refiners、embedding、norm、输入输出等 |
| Qwen Image 2.1 | `transformer_blocks.*.img_mlp.{gate_layer,proj,out}` | attention、文本处理、调制、norm、输入输出等 |

新增三族使用 `main-mlp-nf4-v1`，NF4 blocksize64、double quant、BF16 compute。
这是按模块划分的首版策略，没有照搬Anima热点层号，也不宣称三族全部MLP均不敏感。
各族后续的层敏感度、量化误差与长期质量需要分别评估。

> **第 5 模型族 `moss_image` 与 `--nf4_dir1` 边界说明：**
> - **`moss_image` 不在主干 mixed-NF4 范围内**：`convert_mixed_nf4.py` 仅转换 `krea2_raw`、`z_image`、`qwen_image_2_1`（Anima 沿用独立固定策略）；MOSS-Image 仅在文本编码器离线缓存阶段支持 NF4（`_moss_image_te_nf4.safetensors`），主干 DiT 不提供 mixed-NF4 转换或加载。
> - **`nf4_dir1` 与 mixed-NF4 的边界**：`--nf4_dir1`（搭配 `krea2_attn_fp16`，见 `configs/methods/krea2_lora_nf4_fp16.toml`）仅对 `model_family = "krea2_raw"` 且 `base_compute = "nf4"` 的全 Linear NF4 路径生效（面向 sm70/sm75 卡的内层 FP16 Tensor Core 恢复）；通过 `pretrained_model_name_or_path` 自动识别的主干 MLP mixed-NF4 保持 `base_compute = "bf16"`，不消费 `--nf4_dir1`。

`BF16AutocastLinear4bit` 共享Anima已验证的autocast修复，避免BF16运算后的输出
被恢复成FP32并放大LoRA及checkpoint激活。Krea旧全NF4在线构造和预量化加载也
接入该类；旧v1/v2格式的结构完整性检查保留，无autocast时仍使用bnb原输出语义。
Anima保存格式与旧类import兼容。普通文件头检查不强制导入bitsandbytes。

## 离线转换与加载

只接受本地单文件普通BF16 safetensors。Krea使用既有原生命名；Z-Image使用原loader
支持的单文件格式；Qwen先完成普通BF16 `gate_up` 拆分再量化。输出统一使用runtime
canonical keys，packed字节不会进入普通dense拆分逻辑。

```bash
.venv/bin/python scripts/quantization/convert_mixed_nf4.py \
  --family krea2_raw --source /path/source-bf16.safetensors \
  --output /path/krea2-mixed-nf4.safetensors --threads 4
```

`--family` 还可取 `z_image`、`qwen_image_2_1`。转换在CPU加载完整普通模型，逐矩阵
量化并保存新文件，需要足够主机内存同时容纳源模型和输出payload；不是流式低内存
转换器。不会下载模型，不覆盖源文件或已有目标，发布采用同目录临时文件与原子无覆盖链接。

新metadata `diffusion.mixed_nf4.manifest` 包含family、版本、固定策略、完整源张量
库存、源SHA256、计算精度及bnb版本。加载器以实际模型结构校验全部key、shape、
保留张量dtype与qstate，拒绝未知族、缺失矩阵、多余tensor及损坏量化状态。
double-quant二级scale decode dtype必须是FP32。直接恢复packed权重，不重复量化。
Z-Image的refiner保持原实现，Qwen保留派生频率恢复步骤。

## 首版训练边界

新增格式要求 BF16、plain LoRA、full gradient checkpoint、eager训练；交换使用BF16
transfer，foreach和slab已过小模型验证。已有不兼容配置会明确报错，不静默改参数。

新格式暂拒绝 `torch_compile`、FP16/ConvRot、自适应精度、checkpoint offload、
selective checkpoint及其他adapter变体；Qwen还拒绝融合MLP/QKV、saved projections
和segmented training组合。Qwen现有默认训练配置若开启compile，不能直接据此推断
它能原样使用新混合文件。本轮没有修改这些配置默认值，也没有重跑完整Qwen Edit训练。
这些限制针对新三族混合格式，不改已有BF16或Krea旧全NF4能力。

WebUI预检现在只读模型文件头，提前执行同一训练契约。新Krea混合NF4和Qwen的
compile默认拒绝可由单次风险确认放行；命令行必须显式传入
`--allow_unverified_nf4_compile`，TOML同名值不会启用。其余兼容性和格式检查不变，
Z-Image不提供编译放行。使用步骤见[训练预检与风险确认](../features/config-workbench.md#训练预检与风险确认)。

## 验收与限制

定向CPU联合验收101 passed：三族实际小模型loader往返、原BF16及Krea旧NF4回归、
qstate损坏/无mutation拒绝，以及没有bitsandbytes时普通header/training模块可导入。
另有95项既有族路由、兼容矩阵、Z/Qwen换层及Krea NF4 adapter回归通过。

CUDA联合验收93 passed / 1 skipped，包括：

- 新三族各4个block的真实模型，CPU转换→meta恢复、FP32 plain LoRA参数、BF16 autocast、
  full checkpoint、真实`ModelOffloader`；swap1/2 × foreach/slab共12项。
- 输出、输入梯度及全部LoRA梯度与相同packed基座的无换层对照逐位一致，且非零、finite。
- 训练反向→无梯度预测→恢复换层→再反向；packed payload、Parameter/QuantState身份及
  设备检查通过。无VAE解码或真实图片质量比较。
- Anima autocast及异型换层回归81通过、1项已有重复组合跳过。

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU的UUID> timeout 90 .venv/bin/python -m pytest -q \
  tests/test_mixed_nf4_family_training.py tests/test_anima_mixed_nf4_autocast.py \
  tests/test_block_swap_mixed_nf4.py
```

GPU验收在CMP90HX完成，未覆盖完整模型尺寸、Flash attention或编译。小模型换层等价
证明实现路径正确，不证明NF4与原BF16模型数值等价，更不证明训练收敛或样图质量。
Anima实测减少638MiB训练峰值的数字不能直接外推给这三族。

最终独立复核通过，Pyright 0 errors / 0 warnings。额外64×64共享NF4 Linear的
默认 `torch.compile` + BF16 autocast前后向与eager/stock显式BF16参照逐位一致，
用于检查Krea旧全NF4构造器迁移的基础兼容性；不等同于三族完整模型编译验收，
新混合格式的compile拒绝保持不变。

2026-10-05补充：Qwen全Linear NF4的8组小模型Inductor编译验收通过，但既有
32-block完整权重的独立验收出现latent及LoRA梯度超限，生产compile拒绝继续保留。
保留eager BF16舍入可消除本次小模型差异，尚不足以修复完整模型。复现命令、
数值门槛和原始证据见[Qwen NF4编译验收](../findings/qwen_nf4_compile_90hx_20261005.md)。

2026-10-06追加：严格确定性、static、原生 image LayerNorm/SiLU、保留精度 cast
与关闭 persistent reductions 的联合策略通过完整真实两步数值对照。已接入默认
关闭的 `qwen_nf4_compile_experiment`，真实 `train.py` 完成两步 AdamW 与保存。
这项例外仅覆盖 Qwen Edit 的指定全 Linear NF4/plain LoRA 配置；其他默认 compile
拒绝和上文历史失败保持。合同及未验收范围见[受限实验说明](qwen-nf4-compile.md)。

下一阶段应逐族使用同一初始LoRA、数据与采样参数，分别测加载、前向、反向、采样和恢复
训练，评估权重体积、训练峰值、VAE主导的全程峰值及长期质量；之后再扩展compile和
Qwen融合路径。当前未自动转换真实模型文件、注册模型组或启动长训练。
