# Qwen Image 2.1 Edit NF4 编译实验

状态：可运行实验，默认关闭；2026-10-06。

`qwen_nf4_compile_experiment=true` 启用已通过短测的联合编译策略。CMP 90HX 10 GiB
上，真实 `train.py` 的全 Linear NF4 Edit 两步训练完成 AdamW 更新、checkpoint 保存
并正常退出。此前完整 32-block 真实两步数值对照为 2246/2246 张量逐位一致；
生产 attention-only 初始化与两个真实 bucket 的独立对照为 1286/1286 逐位一致。
原数值门槛未放宽，证据见[编译定位与验收](../findings/qwen_nf4_compile_90hx_20261005.md)。

追加 fresh 真实入口两步训练、两次 512×512 Euler 预览和完整 Accelerate 状态保存，
新进程恢复 step2 后更新到 step3，两个进程均正常退出。普通 NF4 compile 的默认拒绝
仍保留。该开关仅支持下面的受限配置；连续训练与续训的完整轨迹逐位复现、其他
optimizer、长训、速度、画质和其他硬件尚未验收。BF16 LoKr/LoHa 的 T2I 实验
与此入口分别维护，NF4 Edit 仍限 plain LoRA。

## 配置

在既有、可运行的 NF4 Edit 配置副本中设置以下值，保留自己的模型路径和专属
数据缓存。不要把这段当作包含模型路径与数据集蓝图的完整配置。

```toml
qwen_nf4_compile_experiment = true
allow_unverified_nf4_compile = false
torch_compile = true
model_family = "qwen_image_2_1"
qwen_image_2_1_task = "edit"
base_compute = "bf16"
mixed_precision = "bf16"
adaptive_precision = "off"
attn_mode = "torch"
gradient_checkpointing = true
selective_checkpoint = "off"
network_module = "networks.lora_anima"
network_dim = 16
network_alpha = 16
blocks_to_swap = 24
block_swap_transfer_dtype = "bf16"
block_swap_restore_mode = "foreach"
compile_dynamic_seq = false
compile_seq_bands = false
compile_block_scope = "all"
dynamo_backend = "inductor"
compile_inductor_mode = "default"
max_train_steps = 2
```

普通 LoRA 使用 attention-only 的生产目标、kaiming down/zero up 初始化、无路由、
无 dropout、multiplier=1。必须关闭动态/自动 block swap、自适应 OOM 重试、
FP16/FP32 精度岛屿、文本编码器训练、CPU/Unsloth offloaded checkpoint、投影融合、
定额投影保留、segmented training、分区重算实验及额外基座权重。
具体合法键由 `library/quantization/qwen_nf4_compile_policy.py` 与共享 NF4 合同校验；
运行时再次检查实际模型、packed 权重、adapter 和 offloader。

`save_every_n_steps` 保持正数；本次验收保留原值 200，并由训练结束保存最终权重。
设置为 0 会命中既有保存分支取模错误。`network_weights` 仅是 adapter 热启动，
不能代替包含 optimizer/scheduler/RNG 的完整 resume。

## CLI 和 WebUI

两步启动例（模型、缓存及输出路径从实验配置副本读取）：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="<空闲 GPU UUID>" \
  .venv/bin/python train.py --config_file "<实验配置副本.toml>"
```

session 在 CUDA 初始化前启用严格确定性，退出恢复原确定性 flags 和 workspace。
缺省 workspace 会由入口设置；如果 CUDA 已初始化且 workspace 缺失，或环境已有
不同 workspace，会拒绝启动。仅支持单进程。最初入口验收由父 shell 显式提供
workspace；追加的 fresh 训练和独立 resume 进程均由入口自动设置，并在 finally
恢复原环境，fresh 入口设置时 CUDA 尚未初始化。

Next 工作台的「资源 / 编译」分组显示「Qwen NF4 编译实验」，默认不勾选。
选择 Qwen Edit 后可见；已有开启值在其他上下文仍保留显示，方便关闭。
开关不会自动替用户改写上述配置。合法 header 与合同提示实验 warning，
不匹配配置直接拒绝。此开关与「未验证 NF4 编译」风险豁免不能同时启用。
前端类型检查、摘要检查和本地正式构建通过，`web/static/dragon-next` 静态入口
已更新，旧哈希资源与备份保留。尚未做浏览器验收、服务重启或线上发布。

## 实现与证据

adapter apply/load、checkpoint 和 swap 安装后才编译 block inner forward，
swap dispatch 留在图外。联合策略使用原生 image LayerNorm/SiLU、static/all、
Inductor default，保留 eager 精度 cast 并关闭 persistent reductions；不强制 fullgraph。
原生算子限一阶梯度。默认 BF16 路径与原 NF4 compile 风险确认行为保留。

| 验收 | 证据与范围 |
| --- | --- |
| 完整真实两步数值 | 2246/2246 逐位一致，独立复验正常退出；诊断覆盖 attention 与 MLP adapter |
| 生产 attention-only 与双 bucket | 1286/1286 逐位一致，原 zero-up 初始化，独立 exit 0 |
| 真实训练入口 | 50 对数据集中的两步 AdamW，256 参数张量每步变化，384 finite tensors 的 67,162,848-byte checkpoint，exit 0 |
| 训练预览与状态保存 | fresh 两步训练、两次 512×512 Euler1 预览、约 201.6 MB 完整状态，exit 0；预览后训练状态与 32 个 compiled inner identity 恢复 |
| 独立进程完整状态恢复 | step2→3，adapter、AdamW moments/step、scheduler、Torch CPU/CUDA 与 NumPy RNG 匹配，exit 0；Python RNG 观察边界见下文 |
| CPU 合同 | 35 项通过；独立零 dropout 字符串与 malformed network_args 边界 10 项通过 |
| 前端 | checkbox 定向用例、TypeScript、273 项帮助摘要检查、本地正式 Vite 构建通过 |

真实入口原始结果与首次失败保留在
`output/tests/qwen_nf4_compile_train_entry_20261006/`；数值双 bucket 独立报告位于
`output/tests/qwen_nf4_attention_only_buckets_independent_20261006/`。
预览/续训报告位于 `output/tests/qwen_nf4_compile_preview_resume_clean_exit_20261006/`，
前端正式构建记录位于 `output/tests/qwen_nf4_compile_frontend_acceptance_20261006/`。
首次临时配置的保存频率错误修正后完成训练，生产保存代码与原导入配置均保留。

完整状态恢复时，`random.setstate` 返回即刻的 Python RNG 原始 tuple 与保存值
完全相同；随后 Accelerate 的两条 Rich 日志生成源文件超链接，真实消费全局
Python RNG。独立 CPU 因果复现与新 GPU 观察均记录此边界，见日志随机状态诊断
`output/tests/qwen_nf4_compile_rng_review_20261006/report.md`。
因此本次只确认状态加载与续训生命周期，完整连续轨迹逐位等价仍未验收；
没有给恢复算法补偿性设置 RNG，也未修改全局日志行为。

预览 PNG 为 1024×540 的参考图/标题拼图，生成区是 512×512；完整 Pillow 解码
通过。首轮保存后的临时观察器因 0D optimizer step 哈希转换报错退出，旧失败
目录 `output/tests/qwen_nf4_compile_preview_resume_20261006/` 保留；修复观察器后
在新的 clean-exit 目录完成自然退出验收，未修改生产训练源码。

真实入口 smoke 没有另外执行 fresh 完整张量 eager/compiled A/B，不能把它等同于
前面的数值诊断。编译缓存已复用，耗时不代表冷编译或持续性能。保存的
`ss_model_family`/`ss_network_spec` 正确，通用 `modelspec.architecture` 仍沿用既有
`anima-preview/lora`；本仓 Qwen loader 使用前两项，外部工具兼容性未验收。
