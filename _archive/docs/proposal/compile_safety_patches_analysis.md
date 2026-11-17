# Compile/Dynamic-Seq 安全补丁分析报告

状态：已归档
**生成时间**: 2026-06-24
**对比范围**: main vs upstream-snapshot/main
**聚焦范围**: compile cache/wrapper 检测、token budget 自动扩展、EasyControl compile 保护

---

## 1. library/anima/training.py - Sample Preview Token Range 检查

### 修改类型
**安全补丁** - 防止运行时 ConstraintViolationError

### 关键代码块
```python
# Train-time sampling runs through the same compiled blocks as training. The
# compile token budget already covers the prompts present at startup
# (train.py::_sample_prompt_token_counts), but prompts are re-read from disk
# at every sample event — a resolution added mid-run can fall outside the
# dynamic-seq mark_dynamic range and would crash the run with a
# ConstraintViolationError (#42). Skip it instead.
seq_len = (width // 16) * (height // 16)
seq_range = getattr(dit, "_dynamic_seq_range", None)
if (
    getattr(dit, "_dynamic_seq", False)
    and seq_range is not None
    and not (seq_range[0] <= seq_len <= seq_range[1])
):
    logger.warning(
        f"Skipping sample prompt at {width}x{height} ({seq_len} tokens): outside "
        f"the compiled dynamic-seq token range {seq_range}. The compile budget "
        "covers the training buckets plus the sample prompts present at startup; "
        "to sample at this resolution, restart training with it in the prompt "
        "file, lower --w/--h, or disable torch_compile."
    )
    return
```

### 风险评估
- **问题根源**: 训练时 sample preview 使用与训练相同的 compiled blocks，但 prompt 文件在每次 sample event 时重新读取
- **触发场景**: 运行时添加的新分辨率超出启动时编译的 dynamic-seq mark_dynamic range
- **影响**: ConstraintViolationError 导致训练中断（issue #42）
- **修复策略**: 超范围 prompt 主动跳过，记录 warning

### 依赖关系
- 依赖 `dit._dynamic_seq` 和 `dit._dynamic_seq_range` 属性（由 `compile_blocks` 设置）
- 上游提交: `e0b1c71d` (sampler compilation budget)

### 建议行动
**建议合并** - 这是已验证的安全补丁，防止运行时崩溃
- 无本地冲突风险（纯新增检查）
- 可独立工作，不需要其他改动

---

## 2. library/runtime/dynamo.py - pin_dynamo_limit 工具函数（新文件）

### 修改类型
**基础设施** - 修复 ContextVar 导致的 backward compile context 预算回退

### 关键代码块
```python
def pin_dynamo_limit(name: str, value: int) -> int:
    """Raise a dynamo recompile budget so it holds in EVERY execution context.

    torch._dynamo.config.<name> is backed by a ContextVar (user_override),
    so a plain config.<name> = value assignment only takes effect in the thread
    /context that ran it. Dynamo compiles the grad-bearing block _forward in a
    *different* context (the AOTAutograd / backward compile path), where the override
    is absent and the read falls back to the config entry's default (8) — so the
    budget silently reverts and the loop spills to eager at the first grad forward.
    """
    import torch._dynamo as _dynamo

    cfg_mod = _dynamo.config
    target = max(getattr(cfg_mod, name), value)
    setattr(cfg_mod, name, target)  # context-local override (main thread + logs)
    try:
        entry = cfg_mod._config[name]
        canon = (entry.alias or name).rsplit(".", 1)[-1]
        cfg_mod._config[canon].default = target  # global fallback (all contexts)
    except Exception as e:
        logger.warning(
            f"could not pin dynamo {name} default ({e}); budget may revert to 8 "
            "in the backward-compile context and spill to eager"
        )
    return target
```

### 风险评估
- **问题根源**: `torch._dynamo.config` 使用 ContextVar，forward 和 backward 编译运行在不同上下文
- **影响**: backward 编译时预算回退到默认 8，导致多分辨率训练 spill to eager
- **修复策略**: 同时设置 context-local override 和 canonical entry default

### 依赖关系
- 新增独立模块 `library/runtime/dynamo.py`
- 被 `library/anima/models.py`、`library/runtime/harness.py`、`networks/methods/easycontrol.py` 依赖
- 上游提交: `9bb6808f` (easycontrol recompilation issue fix)

### 建议行动
**建议合并** - 核心基础设施修复
- 解决静默性能退化（spill to eager）
- 多处调用点需要协同修改

---

## 3. library/anima/models.py - compile_blocks 使用 pin_dynamo_limit

### 修改类型
**扩展检测** - 修复 backward context 预算回退 + max() 保护

### 关键代码块
```python
def compile_blocks(self, ...):
    """
    Also raises the dynamo cache-size budget to fit those token-count
    families. 2 * n + 8: the 2 * covers fwd+bwd sharing the one
    _forward bytecode, the + 8 covers requires_grad / stride
    specializations (the live path traces ~5 graphs, not 2). max() is
    load-bearing — a caller that knows it has *more* distinct shapes (e.g.
    the multi-resolution SPD distill) raises the limit higher beforehand and
    this must not clobber it back down. This call's own budget only ever
    covers the two full-res families.
    """
    from library.runtime.dynamo import pin_dynamo_limit

    n = int(n_token_families) if n_token_families is not None else len(counts)
    limit = pin_dynamo_limit("recompile_limit", 2 * n + 8)
```

### 风险评估
- **原实现问题**:
  - 直接赋值 `_dynamo.config.cache_size_limit = 2 * n + 8`（无 max 保护）
  - backward context 回退到 8
- **修复**:
  - `pin_dynamo_limit` 同时设置 override 和 default
  - ~~内部有 `max()` 保护（实际 pin_dynamo_limit 内部有 max）~~

### 依赖关系
- 依赖 `library/runtime/dynamo.py::pin_dynamo_limit`
- 上游提交: `9bb6808f` (easycontrol recompilation issue fix)

### 建议行动
**建议合并** - 与 dynamo.py 配套修复
- 确保多分辨率训练（如 SPD distill）不被覆盖

---

## 4. library/runtime/harness.py - compile_dit_blocks + isolate_compile_cache

### 修改类型
**保护逻辑** - compile cache 隔离防止 seq_range guard 污染

### 关键代码块 A: compile_dit_blocks
```python
def compile_dit_blocks(
    anima,
    backend: str = "inductor",
    mode: Optional[str] = None,
    dynamic_seq: bool = False,
    n_token_families: Optional[int] = None,
    seq_range: Optional[tuple] = None,
) -> None:
    """torch.compile each Block._forward for a distillation/training run.

    Pin the canonical .default (not a context-local override) so the wider
    distillation-pool budget survives into the backward compile context.
    """
    from library.runtime.dynamo import pin_dynamo_limit

    pin_dynamo_limit("recompile_limit", cache_size_limit)
    anima.compile_blocks(backend, mode, n_token_families=n_token_families,
                         dynamic_seq=dynamic_seq, seq_range=seq_range)
```

### 关键代码块 B: isolate_compile_cache
```python
def isolate_compile_cache(signature: str) -> str:
    """Route this run's torch.compile caches to a per-signature directory.

    The persistent compile caches (FxGraphCache AND AOTAutogradCache,
    both rooted at TORCHINDUCTOR_CACHE_DIR) key on the FX graph but NOT on
    the mark_dynamic value range, so processes compiled with different
    seq-range bounds poison each other through the shared default cache dir.
    Concretely: inference/bench runs compile the block graph with the canonical
    1024-table default range and deposit entries whose stored guards are floored
    at seq >= 4032; a later multi-tier training run marks [3000, 4200],
    and if its first compile's example batch happens to be ≥4032 tokens, the
    stale entry's guard evaluates TRUE at that hint — AOTAutogradCache accepts
    the hit and re-asserts the narrow guard into the fresh ShapeEnv
    (autograd_cache.py::evaluate_guards), which then contradicts the wider
    mark constraint → ConstraintViolationError (instead of a cache miss).
    """
    global _compile_cache_base
    import hashlib
    import os

    if _compile_cache_base is None:
        base = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or default_cache_dir()
        _compile_cache_base = base

    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    target = os.path.join(_compile_cache_base, f"anima-sig-{digest}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = target
    return target
```

### 风险评估
- **问题根源**: FxGraphCache/AOTAutogradCache 不对 mark_dynamic range 做区分
- **触发场景**:
  - inference 编译 default range (seq >= 4032)
  - training 编译 wider range [3000, 4200]
  - 首批 batch 恰好 >= 4032 → 命中 stale entry → guard 冲突
- **影响**: ConstraintViolationError（依赖首批 batch token count，间歇性触发）
- **修复策略**: 按 compile signature 隔离 TORCHINDUCTOR_CACHE_DIR

### 依赖关系
- 新增 `compile_signature()` 工具函数（规范化签名字符串）
- 上游提交: `0ce87f59` (script compilation error fixes, colorize harness)
- 被所有 compile 入口点（train.py, distill scripts）依赖

### 建议行动
**建议合并** - 关键安全修复
- 防止跨进程 compile cache 污染
- 需配套 train.py 调用 `isolate_compile_cache(compile_signature(...))`

---

## 5. networks/methods/easycontrol.py - compile_cond_stream + dynamic_seq 包装

### 修改类型
**保护逻辑** - 保留原始 inner source，防止重复编译 wrapper

### 关键代码块 A: compile_cond_stream
```python
def compile_cond_stream(
    self,
    backend: str = "inductor",
    mode: Optional[str] = None,
    n_token_families: Optional[int] = None,
    dynamic_seq: bool = False,
    seq_range: Optional[tuple] = None,
):
    """torch.compile each block's two-stream cond forward.

    compile_blocks() only reaches the DiT's own block._forward; the
    active (cond-on) training path routes through _two_stream_inner
    instead (see _make_patched_block_forward), so without this the entire
    cond stream — every cond LoRA projection — runs eager and
    torch_compile is a no-op for EasyControl training.
    """
    from library.runtime.dynamo import pin_dynamo_limit

    n = n_token_families if n_token_families is not None else 2
    per_obj = 4 * n + 16
    pin_dynamo_limit("recompile_limit", per_obj)
    pin_dynamo_limit("accumulated_recompile_limit", len(self._block_modules) * per_obj)

    # ... compile each block._easycontrol_two_stream_inner
    for block in self._block_modules:
        block._easycontrol_two_stream_inner = torch.compile(
            block._easycontrol_two_stream_inner, **compile_kwargs
        )
```

### 关键代码块 B: dynamic_seq wrapper 保护原始 inner
```python
def patched_forward(...):
    inner = block._easycontrol_two_stream_inner

    # compile_dynamic_seq: mark the varying seq axes dynamic INSIDE the
    # checkpointed callable. The checkpoint recomputes in BACKWARD via
    # detach_variable, which detaches the tensor args (x / cond_x) into fresh
    # tensors that LOSE the mark while the RoPE tuples are passed through and
    # KEEP it — that asymmetry is the ConstraintViolationError. Marking inside
    # re-applies on each recompute so forward and backward agree.
    if ec_net._dynamic_seq:
        _compiled_inner = inner
        _lo, _hi = ec_net._dynamic_seq_range

        def inner(x_, emb_, ..., _ci=_compiled_inner, _lo=_lo, _hi=_hi):
            torch._dynamo.mark_dynamic(x_, 2, min=_lo, max=_hi)
            torch._dynamo.mark_dynamic(cond_x_, 1, min=_lo, max=_hi)
            for _r in (rope_, cond_rope_):
                if _r is not None:
                    torch._dynamo.mark_dynamic(_r[0], 0, min=_lo, max=_hi)
                    torch._dynamo.mark_dynamic(_r[1], 0, min=_lo, max=_hi)
            return _ci(x_, emb_, ..., cond_x_, cond_emb_, ...)
```

### 风险评估
- **原问题**:
  - `compile_blocks()` 不覆盖 EasyControl 的 two-stream path
  - cond LoRA projections 运行 eager → torch_compile 无效
- **dynamic_seq 问题**:
  - checkpoint backward 通过 detach_variable recompute
  - detach 后 tensor 丢失 dynamic mark，RoPE tuple 保留
  - 不对称导致 ConstraintViolationError
- **修复策略**:
  - wrapper function 捕获 compiled inner + range 参数
  - 每次调用重新 mark_dynamic（forward 和 backward recompute 都生效）

### 依赖关系
- 依赖 `library/runtime/dynamo.py::pin_dynamo_limit`
- 依赖 `_dynamic_seq` / `_dynamic_seq_range` 属性
- Mirrors `library/anima/models.py::_run_blocks` 设计
- 上游提交: `9bb6808f` (easycontrol recompilation issue fix)

### 建议行动
**建议合并** - EasyControl 训练必需修复
- 否则 torch_compile 对 EasyControl 无效
- 需配套调用 `network.compile_cond_stream(...)` after `apply_to()`

---

## 总结与合并建议

### 最小安全补丁集合（按优先级）

#### P0 - 必需合并（运行时崩溃防护）
1. **library/runtime/dynamo.py** - `pin_dynamo_limit` 基础设施
2. **library/anima/training.py** - sample preview token range 检查
3. **library/runtime/harness.py** - `isolate_compile_cache` 防止 cache 污染

#### P1 - 高价值修复（性能/正确性）
4. **library/anima/models.py** - `compile_blocks` 使用 `pin_dynamo_limit`
5. **networks/methods/easycontrol.py** - `compile_cond_stream` + dynamic_seq wrapper

### 风险矩阵

| 组件 | 本地冲突风险 | 功能完整性 | 独立性 |
|------|------------|----------|--------|
| dynamo.py | 无（新文件） | 核心基础设施 | 依赖项 |
| training.py (sample check) | 低 | 独立防护 | 完全独立 |
| harness.py (isolate_cache) | 低 | 核心防护 | 需 train.py 调用 |
| models.py (pin_dynamo) | 中 | 依赖 dynamo.py | 配套修改 |
| easycontrol.py | 中 | 依赖 dynamo.py | EasyControl 专用 |

### 合并路径建议

**阶段 1: 基础设施**
```bash
# 提取 dynamo.py（新文件，零冲突）
git show upstream-snapshot/main:library/runtime/dynamo.py > library/runtime/dynamo.py
```

**阶段 2: 独立安全补丁**
```bash
# training.py sample preview check（10行纯新增）
git show e0b1c71d -- library/anima/training.py | git apply --3way
```

**阶段 3: compile cache 隔离**
```bash
# harness.py isolate_compile_cache + compile_signature
git diff upstream-snapshot/main^..upstream-snapshot/main -- library/runtime/harness.py \
  | grep -A 100 "isolate_compile_cache\|compile_signature" | git apply --3way
```

**阶段 4: models.py + easycontrol.py（需测试验证）**
- 手动迁移 `pin_dynamo_limit` 调用
- 测试多分辨率训练场景
- 测试 EasyControl 编译效果

### 测试验证点

1. **Sample preview 防护**
   - 启动训练后动态添加超范围分辨率 prompt
   - 验证 warning 输出，不崩溃

2. **Cache 隔离**
   - 先运行 inference（default range）
   - 再运行 training（wider range）
   - 验证无 ConstraintViolationError

3. **Backward context 预算**
   - 多分辨率训练（>8 graphs）
   - 监控无 "skipping graph" eager fallback

4. **EasyControl compile**
   - 对比 compile_cond_stream 前后训练速度
   - 验证 cond LoRA projections 被编译

---

**文档版本**: v1.0
**审核状态**: 待验证
**下一步**: 逐阶段提取 + 单元测试验证
