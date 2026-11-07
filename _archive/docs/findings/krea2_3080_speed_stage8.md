状态：阶段 8 已完成（LoRA rank 消融 → NOT_WORTH_FOR_SPEED）
日期：2026-08-09
原始摘要：`krea2_3080_speed_stage8.json`

# Krea-2 速度研究：LoRA rank 消融

## 假设

当前 Krea-2 LoRA dim16 有 196 个注入模块、48.17M 可训参数。将 rank 降为 8
会减半 adapter GEMM 和 optimizer state，可能降低步时；但它也改变模型容量，
只有收益明确时才能当作性能建议。

## 实测

PG199 1024²、NF4、swap0、full checkpoint、default compile：

| rank / alpha | 可训参数 | 稳态 step | GPU peak |
| --- | ---: | ---: | ---: |
| 16 / 8 | 48.17M | 2.726s | 11.057GB |
| 8 / 4 | 24.08M | 2.728-2.737s | 10.912GB |

rank8 的 5 个稳态步为 `2.729/2.737/2.728/2.728/2.729s`，与 rank16 持平。
峰值仅减少 145MB。loss 有限且下降，但 6 步探针不证明 rank8 的最终质量。

## 判定

**NOT_WORTH_FOR_SPEED**。Krea-2 步时由冻结 DiT 大矩阵、NF4 反量化和 checkpoint
重算主导，adapter rank8/16 的差异被完全淹没。不应为了速度降 rank；rank
应根据 adapter 容量/质量需求选择。

探针新增 `K2_ABL_LORA_DIM` / `K2_ABL_LORA_ALPHA`，供未来质量实验复用，不修改
生产默认 rank。
