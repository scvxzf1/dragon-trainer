# library/preprocess/ 健壮性增强分析

状态：已归档
对比分支：`main` vs `upstream-snapshot/main`
分析重点：健壮性修复、fail-fast 检查、路径约束、避免重复操作的优化

---

## 1. library/preprocess/__init__.py

**修复类型**：优化（lazy import）

**改动描述**：
- 引入 PEP 562 lazy import 机制（`__getattr__`）
- 避免 GUI 启动时通过 `import library.preprocess.resize_preview` 间接拖入 torch/cv2
- 将所有公开接口改为按需加载，仅在首次访问时解析真实模块

**风险评估**：低
- 纯导入机制改动，不改变运行时行为
- 缓存机制（`globals()[name] = value`）确保每个符号仅解析一次

**是否依赖其他文件**：是
- 新增 `library/preprocess/reconcile.py`（cache reconcile 接口）
- 新增 `library/preprocess/caption_variants.py`（caption sidecar 接口）
- `library/preprocess/captions.py` 被删除（功能迁移到 caption_variants）

**建议行动**：合并
- 显著改善 GUI 启动性能（避免加载重型依赖）
- lazy import 是标准 Python 模式，风险可控

---

## 2. library/preprocess/latents.py

### 2.1 corrupt file isolation（健壮性修复）

**修复类型**：健壮性 + fail-fast

**改动描述**：
- 新增 `_decode_batch` CPU 阶段，对每张图片独立 try-except
- 图片解码失败（truncated/corrupt）时隔离为 `failed` 列表，不中止整个批次
- 返回 `(skipped, failed, kept, img_batch)`，区分三种状态
- 批处理完成后统一打印失败清单：
  ```python
  if failed_all:
      print(f"\n⚠ {len(failed_all)} image(s) could not be decoded and were skipped "
            f"(no latent cached — re-stage these, e.g. delete + re-run mangafy):")
      for p, reason in failed_all:
          print(f"  {p}  ({reason})")
  ```

**风险评估**：低
- 一个损坏的 PNG 不再导致整个 latent cache 进程崩溃
- 错误信息清晰（记录路径 + 异常类型）

**是否依赖其他文件**：否

**建议行动**：合并
- 关键健壮性修复，大幅提升容错能力

---

### 2.2 per-resolution skip check（性能优化）

**修复类型**：优化

**改动描述**：
- 新增 `_latent_cached(npz_path, w, h)` 函数，检查特定分辨率的 latent key 是否存在
- 新增 `count_pending_latents` 函数，**无需加载 VAE** 即可统计待编码数量
- 通过只读 NPZ header 判断 `latents_{H}x{W}` key 是否存在（避免完整解码）

**风险评估**：低
- 纯查询优化，不改变编码逻辑
- try-except 包裹 `np.load`，损坏的 NPZ 视为未缓存（会重新编码）

**是否依赖其他文件**：否

**建议行动**：合并
- 允许 webui/CLI 在 VAE 加载前快速判断是否需要跳过整个步骤

---

### 2.3 CPU/IO 并行化（性能优化）

**修复类型**：优化

**改动描述**：
- 引入 `ThreadPoolExecutor` 双线程池：`decode_ex`（图片解码）+ `save_ex`（NPZ 写入）
- GPU VAE forward 保持串行，CPU/IO 阶段与 GPU 重叠执行
- 新增 `_save_batch` 函数，在独立线程中执行 NPZ 读-改-写
- 引入 backpressure 机制（`max_saves`），防止内存膨胀

**风险评估**：低
- 输出与串行路径字节级一致（docstring 明确说明）
- 每个 npz_path 在单次 `cache_latents` 调用中仅写入一次，无 race condition

**是否依赖其他文件**：否

**建议行动**：合并
- 性能提升显著（GPU 不再空闲等待 IO）
- 代码重构良好，错误处理完善

---

## 3. library/preprocess/pe.py

### 3.1 count_pending_pe（性能优化）

**修复类型**：优化

**改动描述**：
- 新增 `count_pending_pe(data_dir, encoder, ...)` 函数
- **无需加载 vision encoder** 即可统计待编码的 PE sidecar 数量
- 纯文件系统存在性检查，镜像 `cache_pe_features` 的跳过逻辑

**风险评估**：低
- 纯查询函数，不改变编码行为

**是否依赖其他文件**：是
- `library/io/cache_names.py` 的 `pe_cache_suffix(encoder)` 函数

**建议行动**：合并
- 与 `count_pending_latents` 对称，允许 webui 快速决策是否需要加载编码器

---

### 3.2 DataLoader 合并（性能优化）

**修复类型**：优化

**改动描述**：
- 将所有 `(W, H)` 分组合并为单一 DataLoader + `batch_sampler`
- 避免 Windows spawn() 多次重新导入 torch/library（每个 `(W, H)` group 都会 spawn 一次）
- 保持 batch 内形状同质性（通过 `batch_sampler` 控制）

**风险评估**：低
- 仅改变 DataLoader 组织方式，batch 内部仍为单一形状
- Windows 上性能提升显著

**是否依赖其他文件**：否

**建议行动**：合并
- Windows 平台友好改进，Linux 上无负面影响

---

### 3.3 grid_h/grid_w metadata 注入

**修复类型**：健壮性 + 功能增强

**改动描述**：
- 在 PE sidecar 的 safetensors metadata 中记录 `grid_h` 和 `grid_w`
- 消费者（REPA v2）可直接读取网格尺寸，无需重新推导 aspect bucket

**风险评估**：低
- 向前兼容（缺失 metadata 的旧 sidecar 仍可工作）
- metadata 不影响 tensor 内容

**是否依赖其他文件**：是
- `library/vision/buckets.pick_bucket` 函数

**建议行动**：合并
- 减少重复计算，提升 REPA 加载效率

---

## 4. library/preprocess/text.py

### 4.1 避免重复 Image.open（优化）

**修复类型**：优化

**改动描述**：
- 提取 commit `6632da48`："skip redundant TE image opens"
- TE 仅需 caption `.txt`，不需图片像素
- 引入 `already_filtered` 标志：当 `keep_stems` 或 `keep_rel_stems` 已过滤时，跳过 `min_pixels` 检查
- `check_pixels = min_pixels > 0 and not already_filtered`

**风险评估**：低
- 逻辑明确：已通过 stem filter 的图片无需再读取像素判断尺寸
- TE 流程完全不需要图片内容（caption-only）

**是否依赖其他文件**：否

**建议行动**：合并
- 显著减少冗余文件 IO，尤其在大数据集上

---

### 4.2 mtime-based cache invalidation（健壮性修复）

**修复类型**：健壮性

**改动描述**：
- 新增 `_cache_is_current(image_path, cache_path)` 函数
- cache mtime 必须 ≥ 所有源文件 mtime：
  - `{stem}.txt`（caption）
  - `{stem}.variants.txt`（variant sidecar，如果存在）
- OSError 安全处理（stat 失败视为 cache 无效）

**风险评估**：低
- 标准 mtime 比较模式
- 防御式编程（try-except OSError）

**是否依赖其他文件**：是
- `library/preprocess/caption_variants.variants_sidecar_path`

**建议行动**：合并
- 确保 caption 编辑后 TE cache 正确失效

---

### 4.3 randomized caption family 升级检查

**修复类型**：健壮性

**改动描述**：
- 新增 `_cache_has_randomized(cache_path)` 函数
- 检查 TE cache 是否包含 `num_randomized` marker（randomized `r`-family）
- 允许旧 cache 在启用 `--caption_tag_randomize_rate` 后自动升级

**风险评估**：低
- 只读 safetensors header（不加载 tensor）
- unreadable/partial cache → 视为缺失 randomized family（重新编码）

**是否依赖其他文件**：否

**建议行动**：合并
- 平滑升级路径，避免手动删除旧 cache

---

## 5. library/preprocess/reconcile.py（新增文件）

**修复类型**：健壮性 + 路径约束

**改动描述**：
- 新增 cache reconcile 模块，用于清理 stale caches
- 根据 `target_res` 重新计算每张图片的正确 bucket
- 识别四类 stale cache：
  - `latent npz`：WxH != correct bucket
  - `resized png`：on-disk size != correct bucket
  - `PE sidecar`：bucket 变化时删除（文件名不含分辨率）
  - `mask`：bucket 变化时删除
- TE cache（`_anima_te.safetensors`）永不删除（纯文本，与分辨率无关）

**核心函数**：
- `find_stale_caches()` — 扫描并返回 `StaleCaches` dataclass
- `delete_stale()` — 执行实际删除
- `reconcile_caches()` — 组合函数，支持 dry-run

**风险评估**：中
- 删除操作不可逆
- 但逻辑明确：仅删除分辨率不匹配的 cache，TE cache 受保护

**是否依赖其他文件**：是
- `library/datasets/buckets`（`choose_edge`, `freefit_bucket` 等）
- `library/preprocess/caption_variants.VARIANTS_SIDECAR_SUFFIX`

**建议行动**：合并
- 必需功能：free-fit 迁移后需要清理 constant-token 时代的旧 cache
- 建议首次运行时使用 dry-run 验证

---

## 6. library/preprocess/caption_variants.py（新增文件）

**修复类型**：重构（torch-free 提取）

**改动描述**：
- 从 `text.py` 提取 caption variant 生成逻辑，移除所有 torch 依赖
- 核心功能：
  - `generate_caption_variants()` — shuffle + tag-dropout + identity-randomize
  - `build_erasure_token_pool()` — dual-single token pool（Qwen3 + T5 共同单 token）
  - `variants_sidecar_path()` / `write_variants_sidecar()` / `read_variants_sidecar()` — sidecar 读写
- **torch-free 保证**：GUI 和 caption 预处理步骤无需加载模型即可生成 variant

**关键改进**：
- `protect_fn` 参数：标记受保护 tag（colorize prep 需保留 copyright tags）
- `tag_randomize_rate` + `erasure_pool`：identity erasure 正则化（lexinvariant，arXiv:2305.16349）
- `erasure_pool` 强制非空检查（`tag_randomize_rate > 0` 时 raise ValueError）

**风险评估**：低
- 纯逻辑提取，`text.py` 仍 re-export 这些函数（向后兼容）
- 无模型依赖，单元测试容易

**是否依赖其他文件**：是
- `library/anima/training`（`NO_ARTIST_SENTINEL`, `find_anima_prefix_end` 等）

**建议行动**：合并
- 关键架构改进：解耦 caption 生成和 TE 编码
- GUI 性能优化必需（避免启动时加载 torch）

---

## 7. library/preprocess/captions.py（已删除）

**修复类型**：重构

**改动描述**：
- 文件被完全删除，功能迁移到 `caption_variants.py`
- 移除了 `CaptionSource`, `StructuredCaption`, `captions.json` 等复杂抽象
- 简化为单一文本 sidecar 模式（`.txt` + `.variants.txt`）

**风险评估**：高（如果 main 分支仍在使用这些接口）

**是否依赖其他文件**：否（被删除方）

**建议行动**：暂缓
- **需要审查 main 分支是否有代码依赖 `library.preprocess.captions` 的导入**
- 如果 webui dataset editor 依赖 `CaptionSource` / `StructuredCaption`，则不能直接合并
- 建议先 grep 检查依赖：
  ```bash
  rg "from library.preprocess.captions import" --type py
  rg "import library.preprocess.captions" --type py
  ```

---

## 总结：建议合并策略

### 立即合并（低风险）：
1. `latents.py` 的 corrupt file isolation + per-resolution skip + 并行化
2. `pe.py` 的 `count_pending_pe` + DataLoader 合并 + grid metadata
3. `text.py` 的避免重复 Image.open + mtime 检查 + randomized family 升级
4. `caption_variants.py` 新增（torch-free 提取）
5. `reconcile.py` 新增（cache reconcile 功能）
6. `__init__.py` 的 lazy import

### 需要验证后合并：
7. `captions.py` 删除 — **必须先确认 main 分支无依赖**

### 拒绝合并（非健壮性改动）：
- `images.py` 的 424 行 diff：包含 free-fit 替换 constant-token 的大型策略改动，超出本次健壮性审计范围

---

## 风险矩阵

| 文件 | 健壮性修复 | 性能优化 | 架构重构 | 风险等级 | 依赖其他文件 |
|------|-----------|---------|---------|---------|------------|
| `latents.py` | ✓ (corrupt isolation) | ✓ (并行化) | ✗ | 低 | 否 |
| `pe.py` | ✓ (metadata) | ✓ (DataLoader) | ✗ | 低 | 是 (cache_names) |
| `text.py` | ✓ (mtime check) | ✓ (skip Image.open) | ✗ | 低 | 是 (caption_variants) |
| `caption_variants.py` | ✓ (error check) | ✓ (torch-free) | ✓ | 低 | 是 (anima.training) |
| `reconcile.py` | ✓ (stale cleanup) | ✗ | ✗ | 中 (删除操作) | 是 (buckets) |
| `__init__.py` | ✗ | ✓ (lazy import) | ✓ | 低 | 是 (新增模块) |
| `captions.py` 删除 | ✗ | ✗ | ✓ | **高** | **需审查依赖** |
| `images.py` | 排除（大型策略替换） | — | — | — | — |

---

## 后续行动

1. **立即**：grep 检查 `library.preprocess.captions` 依赖
2. **合并前**：运行完整测试套件，重点关注：
   - `tests/test_preprocess_dataset.py`
   - webui dataset editor caption 加载逻辑
3. **合并后**：在真实数据集上测试 cache reconcile（dry-run）
