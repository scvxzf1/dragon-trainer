# Methods 索引

一句话：这里放稳定或已接入能力的使用说明，偏“怎么用、怎么配置”。

状态：稳定
适用版本：当前 main
相关入口：`docs/README.md`

## 当前文档

| 文档 | 说明 |
| --- | --- |
| [lokr.md](lokr.md) | LoKr Kronecker 分解、全因子模式与旧哨兵迁移 |
| [loha.md](loha.md) | LoHa Hadamard 积分解：兼容可用、非主力，PEFT/LyCORIS 键布局 |
| [dora.md](dora.md) | DoRA magnitude、完整 edit 强度语义、NF4 只读基座和 merge/fuse 边界 |
| [hydra-lora.md](hydra-lora.md) | HydraLoRA 多专家路由，配合 [../structure/hydralora.md](../structure/hydralora.md) 阅读 |
| [psoft-integrated-ortholora.md](psoft-integrated-ortholora.md) | OrthoLoRA / Cayley 正交参数化，配合 [../structure/ortholora.md](../structure/ortholora.md) 阅读 |
| [timestep_mask.md](timestep_mask.md) | T-LoRA 时间步 rank mask，配合 [../structure/timestep-mask.md](../structure/timestep-mask.md) 阅读 |
| [reft.md](reft.md) | ReFT 残差流表示编辑，配合 [../structure/reft.md](../structure/reft.md) 阅读 |
| [adaln.md](adaln.md) | AdaLN LoRA：`train_adaln` 旋钮与 runtime↔ComfyUI 键布局互转 |
| [mod-guidance.md](mod-guidance.md) | Modulation guidance，基于 pooled-text AdaLN steering |
| [../../_archive/docs/methods/invert.md](../../_archive/docs/methods/invert.md) | 已归档的历史 inversion 说明；现行可运行实验探针见 [../experimental/directedit_editing_v3.md](../experimental/directedit_editing_v3.md)（`python tasks.py exp-invert-directedit`） |
| [spectrum.md](spectrum.md) | Spectrum 推理加速，配合 [../structure/spectrum.md](../structure/spectrum.md) 阅读 |
| [dcw.md](dcw.md) | DCW：post-step SNR-t bias correction |
| [smc_cfg.md](smc_cfg.md) | SMC-CFG / CFG-Ctrl 风格控制器 |
| [cns.md](cns.md) | Colored Noise Sampling |
| [channel_scaling.md](channel_scaling.md) | 通道缩放相关方法记录 |

此外，仓库还提供以下已接入配置入口（尚无独立 methods 单页）：

- **VeRA / GLoRA**：自包含 GUI 变体位于 `configs/gui-methods/vera.toml` 与 `configs/gui-methods/glora.toml`，可通过 `python tasks.py lora-gui vera` / `python tasks.py lora-gui glora` 或 WebUI 变体选择启动（VeRA 超参消融计划见 [../experimental/vera_ablation.md](../experimental/vera_ablation.md)）。
- **Register-token adapter**：配置位于 `configs/methods/register.toml`（`networks.methods.register`），CLI 入口为 `python tasks.py register`。

Postfix 当前用户入口在 [../guidelines/training.md#postfix](../guidelines/training.md#postfix)，[../experimental/postfix.md](../experimental/postfix.md) 只保留兼容跳转。

## 维护规则

- 方法已经稳定或已接入主路径时，优先放本目录。
- 仍在实验、调参或占位阶段的能力，放到 `docs/experimental/`。
- 原理、数学和架构细节放到 `docs/structure/`，本目录只保留使用与配置说明。
- 新增文档后，同步更新本索引和 [../README.md](../README.md)。
