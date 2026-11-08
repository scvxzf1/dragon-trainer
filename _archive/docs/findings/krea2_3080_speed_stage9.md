状态：阶段 9 已完成（multi-bucket compile 图复用）
日期：2026-08-09
原始摘要：`krea2_3080_speed_stage9.json`
探针：`scripts/krea2/probe_nf4_compile_buckets.py`

# Krea-2 multi-bucket compile 消融：阶段 9

## 问题

阶段 3 只验证固定 1024²。生产数据集使用 24 个 `CONSTANT_TOKEN_BUCKETS`，
如果 Inductor 按每个宽高比编译一张图，首轮时间、cache budget 和显存都不可控。

Krea-2 block compile 位于 patchify/position 生成之后，理论上只看 combined sequence
形状。文本固定 512 token 后：

- image 4032 family：`4032 + 512 = 4544`，pad 到 `4608`
- image 4200 family：`4200 + 512 = 4712`，pad 到 `4864`

因此预期仅两张 block 图。

## 方法

为避免每个格重复加载 8.3GB TE 和 VAE，探针用形状真实的合成 latent/
text embedding，只加载一次 NF4 DiT + LoRA。交替执行：

1. 4032-a：1008×1024
2. 4200-a：960×1120
3. 4032-b：896×1152（同 family 不同 aspect）
4. 4200-b：1120×960（同 family 不同 aspect）
5. 回访 4032-a
6. 回访 4200-a

## 结果

| visit | family / size | step | peak |
| ---: | --- | ---: | ---: |
| 0 | 4032-a / 1008×1024 | 14.871s | 10.964GB |
| 1 | 4200-a / 960×1120 | 11.064s | 11.350GB |
| 2 | 4032-b / 896×1152 | **2.731s** | 11.156GB |
| 3 | 4200-b / 1120×960 | **2.956s** | 11.350GB |
| 4 | 4032-a 回访 | **2.730s** | 11.156GB |
| 5 | 4200-a 回访 | **2.955s** | 11.350GB |

两个 family 的首次有编译开销；同 family 换宽高比后立即进入稳态，回访也稳定。
峰值不超过 11.35GB。这证明 24 buckets 不是 24 张 block 图，而是 2 张。

## 判定与默认值

**PASS**。`configs/methods/krea2_lora.toml` 现默认：

```toml
torch_compile = true
compile_dynamic_seq = false
compile_block_scope = "resident"
compile_inductor_mode = "default"
```

PG199 获得阶段 3 已证的 19.1% 稳态收益；RTX 3080 的主收益是稳定的显存
余量，不再宣称热稳态加速。block swap 仍只编译 resident 块，不启用
dynamic sequence 或其他 Inductor preset。
