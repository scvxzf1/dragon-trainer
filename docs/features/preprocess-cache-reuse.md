# 预处理缓存复用（共享池）

状态：稳定
适用版本：当前 WebUI Next 工作台 `/next/training`
入口：`/next/training` 输入准备 → 缓存分组
相关代码：`library/cache_pool/`、`web/services/training/runtime_prepare.py`、`web/services/training/history_ops.py`

---

## 1. 这是干什么的

一句话：同一数据集反复测 LoRA/LoKr 时，复用 **dataset_cache 拷贝 / VAE / TE**，少拷贝、少重算；删历史默认不误删共享池。

共享内容放在：

```text
output/cache_pool/<fingerprint>/
  manifest.json
  resized/
  lora/
  refs.json
```

每次 run 仍有自己的：

```text
output/runs/<run>/
  dataset_cache/dataset-xx/{resized,lora}   # 默认挂到共享池
  training_output/                          # 永远独立
  model_cache/
```

---

## 2. 配置开关（按训练配置保存）

| 键 | 默认 | 含义 |
|---|---|---|
| `reuse_dataset_cache_copy` | true | **A**：挂共享池，避免整包 copytree |
| `reuse_vae_latents` | true | **B**：有匹配 VAE sidecar 则 skip |
| `reuse_text_encoder_cache` | true | **C**：有匹配 TE sidecar 则 skip |
| `cache_fingerprint_mode` | `light` | `light` 或 `content` |
| `force_rebuild_preprocess_cache` | false | 本次强制重建（写本 run 私有，不静默覆盖池） |

### 指纹模式

- **light（默认）**：源路径 + 预处理签名 + 文件 name/size/mtime  
- **content**：再对**输入图字节 + caption 文本**做 SHA256（不 hash npz/te）

改图、改 caption、改分辨率/过滤规则后：

1. 轻量模式通常会因 mtime/size 变化自动换指纹  
2. 不放心就切 `content`，或临时打开「强制重建」

---

## 3. 删除语义

| 对象 | 默认 |
|---|---|
| 历史 meta | 可删 |
| 运行目录 training_output / model_cache | 彻底删除运行目录时删除 |
| dataset_cache 挂载点（symlink） | 只拆链接，不跟删池 |
| `output/cache_pool/<fp>` | **默认不删** |

引用计数：`refs.json` 记录 run_id。删 run 时会 release ref。  
无引用池条目可用 `cleanup_orphan_cache_pool()` 清理（第一期 API/工具级，不做后台自动 GC）。

---

## 4. 注意

- 触发词克隆、`nl_tag_mix` 会对**改写后的 source**单独算指纹，不与原集混池。  
- A/B/C 全关可回到偏隔离行为。  
- 训练权重、日志、样张仍 per-run 隔离。  
- CLI 默认 `post_image_dataset/*` 本就共享；本能力主要服务 WebUI runtime。

---

## 5. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_cache_pool_fingerprint.py \
  tests/test_cache_pool_store.py \
  tests/test_cache_pool_policy.py \
  tests/test_cache_pool_gc.py \
  tests/test_training_runtime_config_start.py \
  tests/test_preprocess_reuse_flags.py \
  tests/test_training_history_delete.py \
  -q
```
