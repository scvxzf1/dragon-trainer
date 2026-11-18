# Qwen Image 2.1 Edit LoRA 真机验收（2026-09-28）

状态：验收记录 / 阶段快照（50 步链路验收，非完整收敛结论）

## 结论与范围

50 对真实编辑数据通过 WebUI 完成 1 轮、50 个优化步，正常保存 LoRA。Qwen3-VL 低显存预处理、编辑缓存预检、DiT forward/backward、BF16 块交换与编译、最终保存和 CPU 严格重载均有实际证据。

本轮关闭采样，不涵盖编辑效果质量、保存后 GPU 推理、optimizer 续训、多参考图训练或其他硬件组合。50 步是链路验收，不是充分训练的质量结论。

## 配置与结果

| 项目 | 实测值 |
| --- | --- |
| 配置 | `configs/imported/9-27-50x-qwen21-edit.toml` |
| 数据集 | `configs/datasets/9-27-50x-edit.toml`；a-2 原图、b-2 编辑后目标，50 对 |
| GPU | CMP 90HX，10 GiB；WebUI GPU 1 |
| 方法 | plain LoRA，rank/alpha=16，batch=1，初始 LR=1e-4 |
| 执行 | BF16，torch attention，full checkpoint，swap24/32，compile all/dynamic=True/default |
| 目标 bucket | 48 张 896×1184，1 张 800×1344，1 张 1184×896 |
| 完成 | 50/50 steps，epoch=1，正常退出 |
| 训练循环 | 13 分 34 秒；含首步冷编译约 98.8 秒，不含预处理/模型加载 |
| 末步 recent_s_per_step | 14.5906 秒；本轮记录值最大 14.6152 秒 |
| loss | 最后一条 current=0.364145；epoch average=0.362767 |
| CUDA max allocated | 7.28220 GiB |
| CUDA max reserved | 8.56055 GiB |
| nvidia-smi 观测 | 2 秒采样最高 9062 MiB，温度最高 54°C；非瞬时峰值保证 |

TensorBoard 每 2 步记录一次：25 条 `loss/current`、`loss/average` 以及显存和步耗时记录，全部有限。冷编译时间不计入上述 recent 窗口；不能用稳态步耗时推算首次启动时间。

## 修复的真实阻塞

1. **Qwen3-VL 8B BF16 整体搬入 10 GiB GPU 溢出。** 加载在 CPU，自动按显存预算对 vision/language core 安装 Accelerate CPU offload；缓存不经过 `lm_head`，不保留全部层 hidden states，保持原来的 pre-norm 输出语义。真实 50 张图文缓存成功。TE 退出并释放后才加载 VAE，随后再释放 VAE、加载 DiT。
2. **运行时数据集丢失 reference 字段、resize 缺失 caption。** 保留 `reference_image_dir`，将同 stem 指令 sidecar 镜像到运行时 resized 目录；严格校验编辑数据条件。
3. **预检沿用 T2I cache 命名/规则。** 使用 Edit 指纹及参考图 latent 校验，核对运行时 caption、bucket、slot 和 metadata；预检保持只读。
4. **896×1200 产生奇数 latent 维度，首个 forward 失败。** Qwen 专属 bucket 向下对齐 32 像素，resize、训练、audit 共用规则；no_upscale 非 canonical 原图也一致。Anima 的 24 个 canonical bucket 未改，旧几何缓存不误用。
5. **Inductor 相对缓存目录导致 g++ 找不到生成的 `.main.cpp`。** WebUI 传入编译器的 Inductor/Triton 缓存环境值解析为绝对路径，任务元数据仍保留可读的相对路径。

最终测试保留 swap24 和编译，没有通过关闭它们绕开故障。原浏览器未保存的 swap26 草稿未覆盖。原图、模型与历史失败任务均保留。

## 产物与复核证据

运行目录：`output/runs/9-27-50x-qwen21-edit-20260928-094640`。

最终权重：`training_output/9-27-50x-qwen21-edit.safetensors`，33,608,480 bytes。

SHA256：`8b5cb8eddab46909efcd6c24cb5f39c0d039d66bfa48a09c5bfe4e9ef79e5dc9`。

独立 CPU 验收通过：

- metadata：`ss_steps=50`、`ss_epoch=1`、`ss_model_family=qwen_image_2_1`。
- 384/384 张量有限，128/128 `lora_up.weight` 非零，无 `_orig_mod` 包装键。
- 通过真实 `create_network_from_weights` 构造 128 个 LoRA 模块、挂到 meta Qwen 基模，再用 `load_state_dict(strict=True)` 在 CPU 加载；384/384 键与形状匹配，回读逐张量相等。未运行基模 forward。

运行目录另存 `hot-test-console.log`、`hot-test-observations.jsonl`、`hot-test-scalars.json`、`hot-test-final-status.json`、`verify_checkpoint.py`；模型配置、数据集快照、TensorBoard events 仍在原运行目录。

## 定向测试与独立审核

各组存在重叠，不相加为总测试数：

- 几何、resize、cache audit、canonical bucket：88 passed。
- 编码器、Edit 预处理、runtime config、Edit data：37 passed。
- 补充编译缓存 cwd 回归后，runtime config：11 passed。
- 独立几何验收：38 passed，并检查 Ruff、Pyright、diff whitespace。
- 独立 Qwen3-VL 语义/清理验收：17 passed，Ruff/语法检查通过。
- 编译路径独立审计及最终权重独立验收通过。

启动时存在 Inductor 复杂算子和 SM 数量提示，最终训练正常完成；未将这些性能提示视作数值正确性或编辑质量的证据。未运行全仓测试、未提交或推送代码。
