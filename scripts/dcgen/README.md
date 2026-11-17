# scripts/dcgen — DC-Gen Anima f32c32 工作目录

状态：实验 / POC 探针
方案文档见 `docs/proposal/dcgen_anima_f32c32.md`。

本目录存放 DC-Gen Anima 锻造（f32c32 + patch1 + 1024²）的探针与后续训练脚本。

## 进度

| 日期 | 内容 | 状态 |
|---|---|---|
| 2026-08-19 | 双 latent 缓存探针（probe 0） | 通过 |
| 2026-08-19 | Patch embedding alignment dry-run（probe 1） | 通过 |
| 2026-08-19 | Output-head alignment dry-run（probe 2，阶段 1） | 通过 |
| 2026-08-19 | LoRA E2E dry-run（probe 3，阶段 2） | 通过 |
| 2026-08-19 | Forged base save/load roundtrip（probe 4） | 通过 |

## 运行

> **模型路径说明**：运行前需指定本机 Anima 基座 DiT 与 VAE 权重路径——若脚本已开放 CLI 参数（可先用 `--help` 查看，如 `--dit <path-to-anima-dit>` / `--vae <path-to-anima-vae>`），请显式通过 CLI 参数传入；若当前检出版本仍使用脚本顶部的 `DIT_PATH` / `ANIMA_VAE_PATH` 常量，请先指向本机有效权重路径后再执行，勿依赖开发机默认路径。

```bash
# 双 latent 缓存（生成 _out/dual_latent_cache/*.npz）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/dcgen/probe_dual_latent_cache.py

# patch alignment dry-run（依赖 probe 0 产物）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/dcgen/probe_patch_align.py

# output-head alignment dry-run（阶段 1，需加载旧 DiT）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/dcgen/probe_output_head_align.py

# LoRA E2E dry-run（阶段 2，需加载旧 DiT）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/dcgen/probe_lora_e2e.py

# 锻造基座 save/load roundtrip（probe 4，验证 library/models/dcgen_forge.py 无损往返）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/dcgen/probe_forge_roundtrip.py
```

## 目录

- `probe_dual_latent_cache.py`：probe 0，双 latent 形状/命名/读回契约。
- `probe_patch_align.py`：probe 1，新 `x_embedder` 对齐旧下采样特征（阶段 0）。
- `probe_output_head_align.py`：probe 2，冻结主干联合训输入/输出层（阶段 1）。
- `probe_lora_e2e.py`：probe 3，LoRA 端到端训练链路（阶段 2）。
- `probe_forge_roundtrip.py`：probe 4，锻造基座 `forge_new_dit` → `save_forged_dit` → `load_forged_dit` 元数据与前向无损往返验证。
- `_out/`：探针产物，不提交。
