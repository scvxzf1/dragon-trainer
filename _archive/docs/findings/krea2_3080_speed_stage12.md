状态：阶段 12 已完成（FlashAttention varlen 已生产化为显式 opt-in，默认仍为 cuDNN SDPA）
日期：2026-08-09
原始摘要：`krea2_3080_speed_stage12.json`
探针：`scripts/krea2/probe_nf4_flash_varlen.py`、`scripts/krea2/probe_nf4_ablation.py`

# Krea-2 packed varlen FlashAttention 消融

## 动机和实现

阶段 11 证明 compiled 单步约 89% 已落在 GEMM 和 attention，其中 cuDNN dense
attention forward+backward 约 847ms。Krea-2 的 `(B,1,L,L)` mask 是有效 token
mask 的外积；实验路径从对角线恢复 `(B,L)` 有效位，只打包有效 Q/K/V，调用
FlashAttention 2.8.3 `flash_attn_varlen_func` 的 native GQA，再 scatter 回 padded
layout。DiT 外部张量契约和 padding attention sink 之外的有效 token 语义不变。

首次 compile 因 `aten.nonzero` 动态输出失败；实验包装器显式开启
`torch._dynamo.config.capture_dynamic_output_shape_ops=True` 后全模型编译通过。

## 数值和内核级结果

在 `L=4608`、BF16、forward+backward 口径下：

| GPU | cuDNN dense | Flash varlen | 变化 |
| --- | ---: | ---: | ---: |
| PG199 | 27.93ms | 14.80ms | -47.0% |
| RTX 3080 | 47.12ms | 30.94ms | -34.3% |

输出 rel-L2 `0.00303`、cos `0.9999955`；Q/K/V 梯度 rel-L2 分别为
`0.00361/0.00197/0.000703`，cos 均不低于 `0.9999934`。无效 padding 梯度最大值
严格为 0，峰值只增加约 77MB。这是不同 attention kernel 的 BF16 舍入差异，未发现
mask 泄漏或梯度断路。

## PG199 全模型

1024² NF4、rank16 LoRA、full gradient checkpoint、28 个 resident compiled blocks：

| 路径 | 稳态 step | GPU peak |
| --- | ---: | ---: |
| 历史 cuDNN compile | 2.726s | 约 11.06GB |
| Flash varlen compile | 2.418-2.424s | 10.87GB |

中位 `2.421s/it`，快 11.2%。中途保存/reload LoRA 96.4MB + optimizer 193.0MB
后，LoRA delta=0、forward delta=0，续训仍为 `2.424s/it`。

最终 50 步长稳态复核同样通过：首步 compile `17.013s`，其余步通常为
`2.417-2.439s`（单个 `2.804s` 抖动），末步 `2.429s`；loss
`0.008386→0.000784`，梯度始终非零且有限，peak 仍为 `10.867GB`。

multi-bucket 探针也通过：4032-token family 复访 `2.387-2.389s`，比历史
`2.731s` 快 12.5%；4200-token family 代表稳态 `2.569s`，比历史 `2.956s` 快
13.1%。最后一个 4200 复访为 `3.011s` 单点抖动，因此不把该 family 的收益写得更高。

## RTX 3080 长窗口

1024² NF4 + swap20 + full checkpoint + 8 个 resident compiled blocks 跑 20 步：

- 首步含 compile：`24.692s`。
- 第 1-5 个稳态步：`11.735-11.847s`。
- 第 15-19 步：`12.118-12.202s`，末五步均值 `12.145s`。
- 历史 cuDNN compile 长窗口末步 `12.65s`，故热稳态约快 4.0%。
- GPU peak `6.094GB`，host RSS `20.463GB`；loss `0.008472→0.002429`，梯度非零且有限。

Flash 路径没有消除 3080 的热漂移，也没有改变 GEMM 主瓶颈，所以目标卡收益显著低于
PG199。它把 `12.65s` 降到约 `12.15s`，是目前首个在 3080 长窗口仍有正收益的软件
attention 候选，但不是数量级变化。

## 生产化更新

后续更新已将结论升级为 **PRODUCTION_READY_EXPLICIT_OPT_IN**：

1. `library/models/krea2_raw/attention_backend.py` 承担 cuDNN SDPA 与 packed
   FlashAttention varlen 实现；`dit.py` 只保留薄 dispatch。
2. `attn_mode="torch"` 是默认和历史兼容路径；`attn_mode="flash"` 才会启用
   varlen backend 与 Dynamo 动态输出捕获。缺失 provider、FP32 或 V100 BF16
   会在加载 25GB DiT 前明确拒绝，不静默切换数值路径。
3. 共享兼容矩阵会前置拒绝 Krea-2 的非法 attention、非 default
   Inductor mode、Anima-only V100 诊断和非 `off/every_other` selective
   checkpoint；从 `base.toml` 继承的 `compile_dynamic_seq=true` 会在启动时自动
   归一为 `false`。WebUI 训练表单与图像测试下拉同步按 family 过滤。
4. CPU 契约测试覆盖 batch>1、native GQA、变长打包、padding 输出/梯度为
   0、provider/dtype 拒绝以及训练/推理接线。PG199 真实 batch=2 BF16
   `torch.compile + checkpoint` 中，padding 输出与 Q/K/V 梯度最大值均为 0。
5. 正式 backend 的 PG199 1024² NF4/full-checkpoint/28-block compile 六步探针：
   首步含编译 `16.01s`，第 6 步 `2.42s`，GPU peak `10.86GB`，loss
   `0.0083→0.0068`，与本文历史 monkeypatch 结果 `2.421s/it / 10.87GB`
   一致。

因此 Flash varlen 已是受支持的 Krea-2 可选编译方案，但不是新默认。
`configs/methods/krea2_lora.toml` 仍显式保留 `attn_mode="torch"`，升级不改变
旧任务的 cuDNN 数值和依赖行为。
