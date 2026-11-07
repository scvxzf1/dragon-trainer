# Guidelines 索引

一句话：这里是用户和维护者最常用的安装、训练、推理操作说明。

状态：稳定
适用版本：当前 main
相关入口：`docs/README.md`

## 当前文档

| 文档 | 说明 |
| --- | --- |
| [根 README](../../README.md)、[Next 前端 README](../../web/frontend-next/README.md) | 当前安装与 WebUI 使用入口 |
| [linux-deployment.zh.md](linux-deployment.zh.md) | Linux 部署与启动指南 |
| [git-sync-policy.md](git-sync-policy.md) | 按目标仓库 URL 区分 `dev` 日常集成与 `main` 发布的同步规则 |
| [training.md](training.md) | 训练参考：LoRA 变体、caption shuffle、masked loss、数据集配置 |
| [inference.md](inference.md) | 推理参考：推理命令、DCW、Spectrum、prompt 文件 |
| [difference_between_comfy.md](difference_between_comfy.md) | anima_lora 与 ComfyUI 核心实现差异 |
| [authorization_tiers.md](authorization_tiers.md) | 授权分级：哪些动作 agent 可直接做、哪些必须等用户 |

## 历史参考

以下综合指南保留已退役 PySide6 桌面 GUI 的流程；安装和 WebUI 操作请使用上面的当前入口。

| 文档 | 说明 |
| --- | --- |
| [指南书.md](指南书.md) | 中文历史综合指南 |
| [guidebook.md](guidebook.md) | 英文历史综合指南 |
| [ガイドブック.md](ガイドブック.md) | 日文历史综合指南 |
| [가이드북.md](가이드북.md) | 韩文历史综合指南 |

## 维护规则

- 中文主路径优先；翻译文档可滞后，但不能成为唯一事实来源。
- 用户安装、训练、推理入口变更时，优先更新本目录。
- 方法细节仍应链接到 `docs/methods/` 或 `docs/experimental/`，避免把算法正文堆进指南。
- 新增文档后，同步更新本索引和 [../README.md](../README.md)。
