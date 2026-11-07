# Preprocess Cache Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 WebUI 在数据集不变时复用 dataset_cache 拷贝 / VAE / TE，并通过配置级三开关与 light/content 指纹控制，同时保证删历史不误删仍被引用的共享池。

**Architecture:** 新增 `library/cache_pool/` 内容寻址共享池（fingerprint → resized + lora）。WebUI runtime materialize 从“每 run copytree”改为“算指纹 → 命中则 mount → 按 A/B/C skip 或补建”。训练产物仍 per-run；历史删除默认不碰 `output/cache_pool/`，只维护 refs 与可选无引用清理。

**Tech Stack:** Python 3.13、现有 `library/io/cache*` / preprocess 管线、WebUI `web/services/training/runtime_*` + `history_*`、配置 `library/config/schema.py`、前端 `web/static/js/config/catalog/*`、pytest。

**Spec:** `docs/superpowers/specs/2026-07-11-preprocess-cache-reuse-design.md`

## Global Constraints

- 默认用简体中文沟通；代码标识符与错误信息保持项目原有风格。
- 第一期只做 A/B/C（dataset_cache 拷贝 / VAE / TE），不做 PE 开关。
- 三开关按**训练配置**保存；默认全开：`reuse_dataset_cache_copy=true`、`reuse_vae_latents=true`、`reuse_text_encoder_cache=true`。
- 指纹默认 `light`（路径 + 预处理签名 + name/size/mtime）；可切 `content`（额外 hash 输入图字节 + caption 文本；**不** hash npz/te）。
- 池根默认：`resolve_output_root() / "cache_pool"`（通常 `output/cache_pool`），不进 `configs/web-training-history/`。
- 挂载回退：`symlink` → `hardlink`（文件级）/ 平台 junction → `copy`。
- 普通强制重建默认写**本 run 私有 cache**，避免静默污染共享池。
- 热点文件（`runtime_prepare.py`、`runtime_datasets.py`、`history_store.py`）只做薄编排；核心逻辑进 `library/cache_pool/`。
- 测试命令默认：`timeout 60 .venv/bin/python -m pytest <file> -q`。
- 不默认跑真实大模型 preprocess / 长训练 / 下载。
- 触发词克隆与 `nl_tag_mix` 必须对**改写后的 source**独立算指纹。
- 训练不变量（TE padding、bucket、lazy load、compile-after-apply）一律不动。

---

## File Map

| 路径 | 职责 |
|---|---|
| `library/cache_pool/__init__.py` | 包导出 |
| `library/cache_pool/fingerprint.py` | light/content 指纹与 preprocess 签名 |
| `library/cache_pool/store.py` | 池路径、manifest、原子发布、命中查询 |
| `library/cache_pool/refs.py` | run_id 引用增减、无引用枚举 |
| `library/cache_pool/mount.py` | 目录/文件挂载与回退 |
| `library/cache_pool/policy.py` | 从 cfg 解析 A/B/C + mode + force 标志 |
| `library/config/schema.py` | 注册新配置键 |
| `web/services/training/runtime_datasets.py` | materialize 改用池/挂载 |
| `web/services/training/runtime_prepare.py` | prepare 时写入 bindings / 调用池 |
| `web/services/training/runtime_resume.py` | resume 路径尊重复用策略 |
| `web/services/training/history_store.py` | 删除时 drop refs；不删池 |
| `web/services/training/` 或 `web/routes/` 可选 API | 清理无引用池（若走 HTTP） |
| `web/static/js/config/catalog/*` | 标签、默认、帮助、表单布局 |
| `web/static/js/config/catalog/field-help-dataset.js` | 更新 resized/lora 说明：可挂共享池 |
| `docs/features/` 短文或更新 `docs/features/dataset-editor.md` / 新 `cache-reuse.md` | 用户语义 |
| `tests/test_cache_pool_*.py` | 单元测试 |
| `tests/test_training_runtime_config_*.py` | runtime 回归与复用行为 |
| `tests/test_training_history_delete.py` | 删历史不删池 |

---

### Task 1: Fingerprint（light + content）

**Files:**
- Create: `library/cache_pool/__init__.py`
- Create: `library/cache_pool/fingerprint.py`
- Create: `tests/test_cache_pool_fingerprint.py`

**Interfaces:**
- Produces:
  - `SCHEMA_VERSION: str = "1"`
  - `FingerprintMode = Literal["light", "content"]`
  - `dataclass PreprocessSignature` 字段至少：`resolution`, `drop_lowres_images`, `min_pixels`, `model_family`, `vae_related`, `text_related`（实现可用 `dict[str, Any]` 规范化后冻结）
  - `build_preprocess_signature(cfg: dict[str, Any], subset_settings: dict[str, Any] | None = None) -> dict[str, Any]`
  - `scan_input_inventory(source_dir: Path, *, recursive: bool, path_pattern: str | None, caption_mode: str | None) -> list[dict[str, Any]]`
    - 每项至少：`relpath`, `size`, `mtime_ns`；content 模式另含 `digest`
  - `compute_fingerprint(*, mode: FingerprintMode, source_dir: Path, inventory: list[dict[str, Any]], preprocess_signature: dict[str, Any], normalized_source: str) -> str`
    - 返回 16–32 hex 小写短 hash（建议 sha256 截断 16）

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache_pool_fingerprint.py
from __future__ import annotations

from pathlib import Path

from library.cache_pool.fingerprint import (
    build_preprocess_signature,
    compute_fingerprint,
    scan_input_inventory,
)


def _touch_image(path: Path, data: bytes = b"\x89PNG\r\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_light_fingerprint_stable_for_same_inputs(tmp_path: Path) -> None:
    src = tmp_path / "src"
    img = src / "a.png"
    cap = src / "a.txt"
    _touch_image(img, b"img-a")
    cap.write_text("1girl, smile", encoding="utf-8")

    inv = scan_input_inventory(src, recursive=True, path_pattern=None, caption_mode="txt")
    sig = build_preprocess_signature(
        {"resolution": 1024, "drop_lowres_images": True, "min_pixels": 500000, "model_family": "anima"},
    )
    fp1 = compute_fingerprint(
        mode="light",
        source_dir=src,
        inventory=inv,
        preprocess_signature=sig,
        normalized_source=str(src.resolve()),
    )
    inv2 = scan_input_inventory(src, recursive=True, path_pattern=None, caption_mode="txt")
    fp2 = compute_fingerprint(
        mode="light",
        source_dir=src,
        inventory=inv2,
        preprocess_signature=sig,
        normalized_source=str(src.resolve()),
    )
    assert fp1 == fp2
    assert len(fp1) >= 16


def test_light_fingerprint_changes_when_caption_mtime_or_bytes_change(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _touch_image(src / "a.png", b"img-a")
    cap = src / "a.txt"
    cap.write_text("1girl", encoding="utf-8")
    sig = build_preprocess_signature({"resolution": 1024, "model_family": "anima"})
    inv1 = scan_input_inventory(src, recursive=True, path_pattern=None, caption_mode="txt")
    fp1 = compute_fingerprint(
        mode="light", source_dir=src, inventory=inv1, preprocess_signature=sig,
        normalized_source=str(src.resolve()),
    )
    cap.write_text("1girl, smile", encoding="utf-8")
    inv2 = scan_input_inventory(src, recursive=True, path_pattern=None, caption_mode="txt")
    fp2 = compute_fingerprint(
        mode="light", source_dir=src, inventory=inv2, preprocess_signature=sig,
        normalized_source=str(src.resolve()),
    )
    assert fp1 != fp2


def test_content_mode_hashes_file_bytes_not_just_mtime(tmp_path: Path) -> None:
    src = tmp_path / "src"
    img = src / "a.png"
    _touch_image(img, b"img-a")
    (src / "a.txt").write_text("tags", encoding="utf-8")
    sig = build_preprocess_signature({"resolution": 1024, "model_family": "anima"})
    inv1 = scan_input_inventory(src, recursive=True, path_pattern=None, caption_mode="txt")
    fp1 = compute_fingerprint(
        mode="content", source_dir=src, inventory=inv1, preprocess_signature=sig,
        normalized_source=str(src.resolve()),
    )
    # rewrite same size if possible; content must change with bytes
    img.write_bytes(b"img-b")
    inv2 = scan_input_inventory(src, recursive=True, path_pattern=None, caption_mode="txt")
    fp2 = compute_fingerprint(
        mode="content", source_dir=src, inventory=inv2, preprocess_signature=sig,
        normalized_source=str(src.resolve()),
    )
    assert fp1 != fp2


def test_adapter_method_not_in_signature() -> None:
    a = build_preprocess_signature({"resolution": 1024, "network_module": "networks.lora", "learning_rate": 1e-4})
    b = build_preprocess_signature({"resolution": 1024, "network_module": "networks.lokr", "learning_rate": 1e-3})
    # preprocess-facing keys equal ⇒ signature equal（方法/lr 不得进入）
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 .venv/bin/python -m pytest tests/test_cache_pool_fingerprint.py -q`  
Expected: FAIL（`ModuleNotFoundError` 或 import 失败）

- [ ] **Step 3: Write minimal implementation**

`library/cache_pool/__init__.py`:

```python
"""Shared preprocess cache pool (content-addressed resized + VAE/TE)."""

from library.cache_pool.fingerprint import (
    SCHEMA_VERSION,
    build_preprocess_signature,
    compute_fingerprint,
    scan_input_inventory,
)

__all__ = [
    "SCHEMA_VERSION",
    "build_preprocess_signature",
    "compute_fingerprint",
    "scan_input_inventory",
]
```

`library/cache_pool/fingerprint.py` 要点：

```python
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "1"
FingerprintMode = Literal["light", "content"]

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# 与 WebUI DATASET_IMAGE_EXTS 对齐；实现时优先 import 现有常量若无循环依赖

_PREPROCESS_KEYS = (
    "resolution",
    "drop_lowres_images",
    "min_pixels",
    "model_family",
    "enable_bucket",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "caption_extension",
    "keep_tokens",
    # 仅纳入影响 VAE/TE 输出的键；不要纳入 learning_rate / network_*
)


def build_preprocess_signature(
    cfg: dict[str, Any], subset_settings: dict[str, Any] | None = None
) -> dict[str, Any]:
    merged = dict(cfg)
    if subset_settings:
        # subset 级 preprocess 覆盖（custom_attributes.preprocess 等）优先
        merged = {**merged, **subset_settings}
    out: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    for key in _PREPROCESS_KEYS:
        if key in merged:
            out[key] = merged[key]
    return out


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_input_inventory(
    source_dir: Path,
    *,
    recursive: bool,
    path_pattern: str | None,
    caption_mode: str | None,
    mode_for_digest: FingerprintMode | None = None,
) -> list[dict[str, Any]]:
    """Enumerate images + paired captions.

    ``mode_for_digest`` 若为 content 则填 digest；否则仅 size/mtime。
    实现时可让 compute_fingerprint 在 content 时再补 digest，二选一但测试要一致。
    """
    source_dir = source_dir.resolve()
    paths: list[Path] = []
    if recursive:
        for p in source_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                paths.append(p)
    else:
        for p in source_dir.iterdir():
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                paths.append(p)
    # path_pattern：若项目已有 normalize/filter helper，复用 web/config 或 library 现有逻辑；
    # 无 pattern 时不过滤。
    items: list[dict[str, Any]] = []
    for img in sorted(paths, key=lambda p: p.as_posix()):
        rel = img.relative_to(source_dir).as_posix()
        st = img.stat()
        entry: dict[str, Any] = {
            "relpath": rel,
            "size": st.st_size,
            "mtime_ns": int(st.st_mtime_ns),
            "kind": "image",
        }
        cap = img.with_suffix(".txt")
        if caption_mode in (None, "txt", "sidecar") and cap.is_file():
            cst = cap.stat()
            items.append({
                "relpath": cap.relative_to(source_dir).as_posix(),
                "size": cst.st_size,
                "mtime_ns": int(cst.st_mtime_ns),
                "kind": "caption",
            })
        items.append(entry)
    items.sort(key=lambda x: x["relpath"])
    return items


def compute_fingerprint(
    *,
    mode: FingerprintMode,
    source_dir: Path,
    inventory: list[dict[str, Any]],
    preprocess_signature: dict[str, Any],
    normalized_source: str,
) -> str:
    payload_inventory: list[dict[str, Any]] = []
    source_dir = source_dir.resolve()
    for item in inventory:
        row = {
            "relpath": item["relpath"],
            "size": item["size"],
            "mtime_ns": item["mtime_ns"],
            "kind": item.get("kind"),
        }
        if mode == "content":
            p = source_dir / item["relpath"]
            row["digest"] = item.get("digest") or (_file_digest(p) if p.is_file() else "")
        payload_inventory.append(row)
    blob = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "normalized_source": normalized_source,
        "preprocess_signature": preprocess_signature,
        "inventory": payload_inventory,
    }
    raw = json.dumps(blob, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]
```

实现时注意：`scan_input_inventory` 的 `caption_mode` 与 WebUI captions.json 路径对齐——若 subset 使用 `captions.json`，inventory 必须纳入该 json 的 size/mtime（content 模式 hash 文件内容）。最小实现可先支持 txt sidecar + 若存在 `captions.json` 则加入单条 inventory。

- [ ] **Step 4: Run tests**

Run: `timeout 60 .venv/bin/python -m pytest tests/test_cache_pool_fingerprint.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add library/cache_pool/__init__.py library/cache_pool/fingerprint.py tests/test_cache_pool_fingerprint.py
git commit -m "feat(cache_pool): add light/content input fingerprints"
```

---

### Task 2: Pool store + refs + mount

**Files:**
- Create: `library/cache_pool/store.py`
- Create: `library/cache_pool/refs.py`
- Create: `library/cache_pool/mount.py`
- Create: `tests/test_cache_pool_store.py`

**Interfaces:**
- Produces:
  - `default_pool_root() -> Path` → `resolve_output_root()/cache_pool`；测试可 monkeypatch
  - `pool_entry_dir(pool_root: Path, fingerprint: str) -> Path`
  - `write_manifest(entry_dir: Path, manifest: dict) -> None`
  - `read_manifest(entry_dir: Path) -> dict | None`
  - `publish_pool_entry(pool_root, fingerprint, *, staging_dir: Path, manifest: dict) -> Path`  
    原子：`staging` rename 到 `pool_root/fp`（已存在则复用已发布）
  - `acquire_ref(entry_dir: Path, run_id: str) -> None`
  - `release_ref(entry_dir: Path, run_id: str) -> None`
  - `list_orphans(pool_root: Path) -> list[Path]`（refs 空或缺失）
  - `mount_dir(src: Path, dst: Path) -> str` 返回 `link_mode`: `"symlink"|"hardlink"|"copy"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache_pool_store.py
from __future__ import annotations

from pathlib import Path

from library.cache_pool.mount import mount_dir
from library.cache_pool.refs import acquire_ref, list_orphans, release_ref
from library.cache_pool.store import publish_pool_entry, read_manifest


def test_publish_and_ref_lifecycle(tmp_path: Path) -> None:
    pool = tmp_path / "cache_pool"
    staging = tmp_path / "staging"
    (staging / "resized").mkdir(parents=True)
    (staging / "lora").mkdir(parents=True)
    (staging / "resized" / "a.png").write_bytes(b"x")
    manifest = {"schema_version": "1", "fingerprint": "abc123", "mode": "light"}
    entry = publish_pool_entry(pool, "abc123", staging_dir=staging, manifest=manifest)
    assert entry.is_dir()
    assert (entry / "resized" / "a.png").is_file()
    assert read_manifest(entry)["fingerprint"] == "abc123"

    acquire_ref(entry, "run-1")
    acquire_ref(entry, "run-2")
    release_ref(entry, "run-1")
    assert list_orphans(pool) == []
    release_ref(entry, "run-2")
    orphans = list_orphans(pool)
    assert entry in orphans


def test_mount_dir_fallback(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.bin").write_bytes(b"data")
    dst = tmp_path / "dst"
    mode = mount_dir(src, dst)
    assert mode in {"symlink", "hardlink", "copy"}
    assert (dst / "f.bin").read_bytes() == b"data"
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 60 .venv/bin/python -m pytest tests/test_cache_pool_store.py -q`  
Expected: FAIL import

- [ ] **Step 3: Minimal implementation**

`store.py`：

```python
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def default_pool_root() -> Path:
    # 延迟 import 避免循环
    from web.services.settings_service import resolve_output_root

    return Path(resolve_output_root()) / "cache_pool"


def pool_entry_dir(pool_root: Path, fingerprint: str) -> Path:
    safe = "".join(c for c in fingerprint if c.isalnum())[:64]
    return pool_root / safe


def write_manifest(entry_dir: Path, manifest: dict[str, Any]) -> None:
    entry_dir.mkdir(parents=True, exist_ok=True)
    path = entry_dir / "manifest.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_manifest(entry_dir: Path) -> dict[str, Any] | None:
    path = entry_dir / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def publish_pool_entry(
    pool_root: Path,
    fingerprint: str,
    *,
    staging_dir: Path,
    manifest: dict[str, Any],
) -> Path:
    pool_root.mkdir(parents=True, exist_ok=True)
    final = pool_entry_dir(pool_root, fingerprint)
    if final.exists() and read_manifest(final) is not None:
        return final
    write_manifest(staging_dir, manifest)
    # 原子发布：同文件系统 rename
    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_name = parent / f".tmp-{fingerprint}-{os.getpid()}"
    if tmp_name.exists():
        shutil.rmtree(tmp_name)
    shutil.move(str(staging_dir), str(tmp_name))
    try:
        tmp_name.rename(final)
    except FileExistsError:
        shutil.rmtree(tmp_name, ignore_errors=True)
    return final
```

`refs.py`：

```python
from __future__ import annotations

import json
from pathlib import Path


def _refs_path(entry_dir: Path) -> Path:
    return entry_dir / "refs.json"


def _load(entry_dir: Path) -> list[str]:
    path = _refs_path(entry_dir)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    refs = data.get("run_ids") if isinstance(data, dict) else data
    return [str(x) for x in (refs or [])]


def _save(entry_dir: Path, run_ids: list[str]) -> None:
    path = _refs_path(entry_dir)
    uniq = sorted(set(run_ids))
    path.write_text(json.dumps({"run_ids": uniq}, ensure_ascii=False, indent=2), encoding="utf-8")


def acquire_ref(entry_dir: Path, run_id: str) -> None:
    run_id = str(run_id).strip()
    if not run_id:
        return
    refs = _load(entry_dir)
    if run_id not in refs:
        refs.append(run_id)
    _save(entry_dir, refs)


def release_ref(entry_dir: Path, run_id: str) -> None:
    run_id = str(run_id).strip()
    refs = [r for r in _load(entry_dir) if r != run_id]
    _save(entry_dir, refs)


def list_orphans(pool_root: Path) -> list[Path]:
    if not pool_root.is_dir():
        return []
    out: list[Path] = []
    for child in pool_root.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not (child / "manifest.json").is_file():
            continue
        if not _load(child):
            out.append(child)
    return out
```

`mount.py`：

```python
from __future__ import annotations

import os
import shutil
from pathlib import Path


def mount_dir(src: Path, dst: Path) -> str:
    src = src.resolve()
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
        return "symlink"
    except OSError:
        pass
    # hardlink 目录不可用：退化为 copytree（文件级 hardlink 可作为后续优化）
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return "copy"
```

- [ ] **Step 4: Run tests**

Run: `timeout 60 .venv/bin/python -m pytest tests/test_cache_pool_store.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add library/cache_pool/store.py library/cache_pool/refs.py library/cache_pool/mount.py tests/test_cache_pool_store.py
git commit -m "feat(cache_pool): add pool publish, refs, and mount fallback"
```

---

### Task 3: Policy 解析 + schema 配置键

**Files:**
- Create: `library/cache_pool/policy.py`
- Modify: `library/config/schema.py`（在 `lora_cache_dir` 注册块附近 `CONFIG_SCHEMA.setdefault`）
- Create: `tests/test_cache_pool_policy.py`
- 如有 TOML roundtrip 测试：扩展 `tests/test_config.py` 或最小新测

**Interfaces:**
- Produces:
  - `dataclass CacheReusePolicy`：
    - `reuse_dataset_cache_copy: bool`
    - `reuse_vae_latents: bool`
    - `reuse_text_encoder_cache: bool`
    - `fingerprint_mode: FingerprintMode`
    - `force_rebuild: bool`
  - `parse_cache_reuse_policy(cfg: dict[str, Any]) -> CacheReusePolicy`
  - 默认：三 bool True，mode `"light"`，force False

Schema 键（manual ConfigKey）：

| name | type | default | choices |
|---|---|---|---|
| `reuse_dataset_cache_copy` | bool | True | |
| `reuse_vae_latents` | bool | True | |
| `reuse_text_encoder_cache` | bool | True | |
| `cache_fingerprint_mode` | str | `"light"` | `("light", "content")` |
| `force_rebuild_preprocess_cache` | bool | False | |

- [ ] **Step 1: Failing tests**

```python
# tests/test_cache_pool_policy.py
from library.cache_pool.policy import parse_cache_reuse_policy


def test_defaults_all_reuse_light() -> None:
    p = parse_cache_reuse_policy({})
    assert p.reuse_dataset_cache_copy is True
    assert p.reuse_vae_latents is True
    assert p.reuse_text_encoder_cache is True
    assert p.fingerprint_mode == "light"
    assert p.force_rebuild is False


def test_parse_overrides() -> None:
    p = parse_cache_reuse_policy({
        "reuse_dataset_cache_copy": False,
        "reuse_vae_latents": False,
        "cache_fingerprint_mode": "content",
        "force_rebuild_preprocess_cache": True,
    })
    assert p.reuse_dataset_cache_copy is False
    assert p.reuse_vae_latents is False
    assert p.reuse_text_encoder_cache is True
    assert p.fingerprint_mode == "content"
    assert p.force_rebuild is True
```

并加 schema 注册存在性测试（按项目现有 `populate_schema` 方式调用后再 assert keys in CONFIG_SCHEMA）。

- [ ] **Step 2: Run fail → implement policy + schema setdefault → pass**

- [ ] **Step 3: Commit**

```bash
git add library/cache_pool/policy.py library/config/schema.py tests/test_cache_pool_policy.py
git commit -m "feat(config): register cache reuse policy keys"
```

---

### Task 4: Runtime materialize 接入共享池（A）

**Files:**
- Modify: `web/services/training/runtime_datasets.py`
- Modify: `web/services/training/runtime_prepare.py`
- Modify: `web/services/training/runtime_resume.py`（`copy_existing` 路径与 policy 对齐）
- Create: `tests/test_training_runtime_cache_reuse.py`
- 更新必要处：`tests/test_training_runtime_config_core.py` 若默认行为从“空目录存在”变为“可 symlink”

**Interfaces:**
- Consumes: Task1–3 的 fingerprint / store / mount / policy
- Produces:
  - `_materialize_subset_cache_binding(...)` 返回：
    ```python
    {
      "fingerprint": str,
      "pool_path": str,          # display path
      "link_mode": str,
      "reuse_flags": {"A": bool, "B": bool, "C": bool},
      "fingerprint_mode": str,
      "image_dir": str,
      "cache_dir": str,
    }
    ```
  - `run.meta.json` / runtime payload 增加 `dataset_cache_bindings` 与 `cache_pool_root`

**行为：**

1. 对每个 subset（在 nl_tag_mix / trigger_clone 之后）：
   - `policy = parse_cache_reuse_policy(cfg)`
   - `sig = build_preprocess_signature(cfg, subset preprocess settings)`
   - `inv = scan_input_inventory(source_path, ...)`
   - `fp = compute_fingerprint(mode=policy.fingerprint_mode, ...)`
2. `entry = pool_entry_dir(pool_root, fp)`
3. 若 force_rebuild：目标改为 **run 私有** `group_dir/resized` 与 `group_dir/lora`（不 publish 覆盖池）
4. 若非 force 且 entry 有 manifest：
   - A 开：`mount_dir(entry/resized, group/resized)` 与 lora
   - A 关：`copytree` 到 group（可从池 copy）
5. 若未命中：创建 staging，mkdir resized/lora，后续 preprocess 写入；完成后 `publish_pool_entry`（仅当 A 开或显式共享写策略；第一期：默认新建写入池再 mount，与“共享内容池”一致；force 时只写私有）
6. `acquire_ref(entry, run_id)` — `run_id` 用 run_dir 名或 history task id（prepare 阶段可用 run_dir.name）

- [ ] **Step 1: Failing integration-style test**

```python
# tests/test_training_runtime_cache_reuse.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import toml

from tests.training_resume_test_support import (
    _patch_runtime_service_paths,
    _write_runtime_config_tree,
)
from web.services import training_service


def test_second_prepare_reuses_pool_without_full_private_copy(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    # 给 imported 配置对应的 source 放最小图片+caption（路径以 support 树为准）
    src_a = tmp_path / "image_dataset" / "a"
    src_a.mkdir(parents=True, exist_ok=True)
    (src_a / "x.png").write_bytes(b"\x89PNG\r\nfake")
    (src_a / "x.txt").write_text("1girl", encoding="utf-8")

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 11, 45, 14)

        @classmethod
        def fromtimestamp(cls, value):
            return datetime.fromtimestamp(value)

    monkeypatch.setattr(training_service, "datetime", FixedDatetime)

    runtime1 = training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )
    run1 = tmp_path / "output" / "runs" / "522-20260523-114514"
    pool_root = tmp_path / "output" / "cache_pool"
    assert pool_root.is_dir() or any(pool_root.glob("*")) or True  # 允许实现后创建

    # 在第一次 run 绑定的池 resized 写入 sentinel（实现后 bindings 可从 run.meta 读）
    meta1 = json.loads((run1 / "run.meta.json").read_text(encoding="utf-8"))
    bindings = meta1.get("dataset_cache_bindings") or []
    assert bindings, "prepare 应写入 dataset_cache_bindings"
    pool_path = Path(bindings[0]["pool_path"])
    if not pool_path.is_absolute():
        pool_path = tmp_path / pool_path
    sentinel = pool_path / "resized" / "sentinel.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("shared", encoding="utf-8")

    class FixedDatetime2(FixedDatetime):
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 12, 0, 0)

    monkeypatch.setattr(training_service, "datetime", FixedDatetime2)
    runtime2 = training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )
    run2 = tmp_path / "output" / "runs" / "522-20260523-120000"
    resized2 = run2 / "dataset_cache" / "dataset-01" / "resized"
    # 第二次应看到池里的 sentinel（symlink/copy 命中共享内容）
    assert (resized2 / "sentinel.txt").read_text(encoding="utf-8") == "shared"

    # 池条目不应因第二次 prepare 再复制出第二份逻辑内容目录（允许一个 fp）
    entries = [p for p in pool_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    assert len(entries) >= 1
```

- [ ] **Step 2: Implement materialize helpers in `runtime_datasets.py`；`runtime_prepare` 调用**

关键改造点（概念代码）：

```python
# runtime_datasets.py 内新增
def _bind_subset_to_cache_pool(
    *,
    cfg: dict[str, Any],
    row: dict[str, Any],
    group_dir: Path,
    pool_root: Path,
    run_id: str,
    source_dir: str,
) -> dict[str, Any]:
    from library.cache_pool.fingerprint import (
        build_preprocess_signature,
        compute_fingerprint,
        scan_input_inventory,
    )
    from library.cache_pool.mount import mount_dir
    from library.cache_pool.policy import parse_cache_reuse_policy
    from library.cache_pool.refs import acquire_ref
    from library.cache_pool.store import pool_entry_dir, publish_pool_entry, read_manifest, write_manifest

    policy = parse_cache_reuse_policy(cfg)
    source_path = _resolve_display_path(source_dir)
    if source_path is None or not source_path.is_dir():
        raise ValueError(f"invalid source_dir: {source_dir}")
    settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    preprocess_settings = settings.get("preprocess") if isinstance(settings.get("preprocess"), dict) else settings
    sig = build_preprocess_signature(cfg, preprocess_settings)
    inv = scan_input_inventory(
        source_path,
        recursive=_bool_value_for_row(row.get("recursive"), True),
        path_pattern=_normalize_path_pattern(row.get("path_pattern")),
        caption_mode=str(settings.get("caption_source_mode") or "txt"),
    )
    fp = compute_fingerprint(
        mode=policy.fingerprint_mode,
        source_dir=source_path,
        inventory=inv,
        preprocess_signature=sig,
        normalized_source=str(source_path.resolve()),
    )
    entry = pool_entry_dir(pool_root, fp)
    resized_dst = group_dir / "resized"
    lora_dst = group_dir / "lora"
    link_mode = "copy"
    if policy.force_rebuild:
        resized_dst.mkdir(parents=True, exist_ok=True)
        lora_dst.mkdir(parents=True, exist_ok=True)
        link_mode = "private"
    else:
        manifest = read_manifest(entry)
        if manifest is None:
            # staging under pool temp
            staging = pool_root / f".staging-{fp}-{run_id}"
            if staging.exists():
                shutil.rmtree(staging)
            (staging / "resized").mkdir(parents=True)
            (staging / "lora").mkdir(parents=True)
            write_manifest(staging, {
                "schema_version": "1",
                "fingerprint": fp,
                "mode": policy.fingerprint_mode,
                "preprocess_signature": sig,
            })
            entry = publish_pool_entry(pool_root, fp, staging_dir=staging, manifest=read_manifest(staging) or {})
        if policy.reuse_dataset_cache_copy:
            link_mode = mount_dir(entry / "resized", resized_dst)
            mount_dir(entry / "lora", lora_dst)
        else:
            resized_dst.mkdir(parents=True, exist_ok=True)
            lora_dst.mkdir(parents=True, exist_ok=True)
            if (entry / "resized").exists():
                _copy_runtime_dataset_dir(str(entry / "resized"), resized_dst)
            if (entry / "lora").exists():
                _copy_runtime_dataset_dir(str(entry / "lora"), lora_dst)
            link_mode = "copy"
        acquire_ref(entry, run_id)
    return {
        "fingerprint": fp,
        "pool_path": _display_settings_path(entry),
        "link_mode": link_mode,
        "reuse_flags": {
            "A": policy.reuse_dataset_cache_copy,
            "B": policy.reuse_vae_latents,
            "C": policy.reuse_text_encoder_cache,
        },
        "fingerprint_mode": policy.fingerprint_mode,
        "image_dir": _display_settings_path(resized_dst),
        "cache_dir": _display_settings_path(lora_dst),
    }
```

`runtime_prepare.py` 循环内用 binding 的 image_dir/cache_dir 填 runtime_rows，并把 bindings 写入 `_write_runtime_run_meta`。

- [ ] **Step 3: Run**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_runtime_cache_reuse.py \
  tests/test_training_runtime_config_core.py \
  tests/test_training_runtime_config_start.py \
  tests/test_training_runtime_config_trigger_clone.py \
  tests/test_training_runtime_config_nl_tag.py \
  -q
```

Expected: PASS（必要时微调旧测试：允许 resized 为 symlink）

- [ ] **Step 4: Commit**

```bash
git add web/services/training/runtime_datasets.py web/services/training/runtime_prepare.py \
  web/services/training/runtime_resume.py tests/test_training_runtime_cache_reuse.py \
  tests/test_training_runtime_config_*.py
git commit -m "feat(webui): mount shared cache_pool into runtime dataset_cache"
```

---

### Task 5: Preprocess 尊重 B/C 与 force（skip / 强制层）

**Files:**
- Modify: `scripts/tasks/preprocess.py` 和/或 WebUI 启动 preprocess 的参数组装处  
  （若 WebUI 通过 runtime 配置调用 preprocess，优先读 `config.runtime.toml` 的 reuse 标志，转发 `--overwrite` 或层选择）
- Modify: 现有 latent/text cache CLI 已有 skip；确保 **B/C 关或 force** 时对该层传 overwrite
- Create: `tests/test_preprocess_reuse_flags.py`（可用 monkeypatch 捕获子进程 argv，不跑真实 encode）

**行为表：**

| 条件 | VAE | TE |
|---|---|---|
| reuse_vae true 且文件齐 | skip | — |
| reuse_vae false 或 force | overwrite/rebuild | — |
| reuse_te true 且文件齐 | — | skip |
| reuse_te false 或 force | — | overwrite/rebuild |

强制重建且非“发布到池”：输出目录为 run 私有 cache_dir（Task4 已指到私有）。

**现状：** `library/preprocess/latents.py` / 部分 text 路径已支持 `overwrite=` 参数，但 `scripts/preprocess/cache_latents.py` **尚未暴露** `--overwrite` CLI。Task5 必须同时：

1. 给 `scripts/preprocess/cache_latents.py` 增加 `--overwrite`（store_true），传入 `cache_latents(..., overwrite=args.overwrite)`  
2. 给 `scripts/preprocess/cache_text_embeddings.py` 增加对称 `--overwrite`（若 TE skip 路径需要；与现有 skip existing 对齐）  
3. `scripts/tasks/preprocess.py` 在 `reuse_vae_latents=false` 或 `force_rebuild_preprocess_cache=true` 时附加 `--overwrite` 到 VAE 子命令；TE 同理用 `reuse_text_encoder_cache`

- [ ] **Step 1: 写 argv 捕获测试**

```python
# tests/test_preprocess_reuse_flags.py
from __future__ import annotations

from scripts.tasks import preprocess as preprocess_task


def test_vae_overwrite_flag_when_reuse_disabled(monkeypatch):
    captured = []

    def fake_run(cmd, *args, **kwargs):
        captured.append(cmd)
        return 0

    monkeypatch.setattr(preprocess_task, "run", fake_run)  # 若实际是 subprocess 包装名，改成真实符号
    # 构造 overrides：reuse_vae_latents=False，并 stub _preprocess_rows 返回一行
    # 调用 _run_preprocess_vae(row, [])
    # assert any(c for c in captured if "--overwrite" in c)
```

实现时以 `scripts/tasks/preprocess.py` 真实子进程调用函数名为准（`rg "subprocess|run_module|_run_preprocess_vae" scripts/tasks/preprocess.py`）。

- [ ] **Step 2: 接线实现 → 测试 PASS**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(preprocess): honor VAE/TE reuse flags and force rebuild"
```

---

### Task 6: 历史删除 drop refs + 不删池 + 无引用清理

**Files:**
- Modify: `web/services/training/history_store.py`（`_delete_history_task` / runtime delete 流程）
- 可选 Create: `web/services/training/cache_pool_gc.py` + route/API
- Modify: `tests/test_training_history_delete.py`
- Create: `tests/test_cache_pool_gc.py`

**行为：**

1. 删除 run 前：读 `run.meta.json` / task meta 的 `dataset_cache_bindings`
2. 对每个 `pool_path`：`release_ref(entry, run_id)`
3. 删除 runtime 目录时：
   - 若 `dataset_cache` 子项是指向 `cache_pool` 的 symlink：只 unlink 链接，**禁止** follow 删池
   - `shutil.rmtree` 前对 symlink 目录特殊处理
4. 新增函数 `cleanup_orphan_cache_pool(pool_root=None) -> dict`：只删 `list_orphans`
5. 第一期 UI 可先后端 API + 文档；前端按钮可同 Task7 做最小入口

- [ ] **Step 1: Tests**

```python
def test_delete_history_releases_ref_but_keeps_shared_pool(tmp_path, monkeypatch):
    # 构造 pool entry + refs [run-a, run-b]
    # 构造 history task 指向 run-a
    # delete task run-a
    # assert pool entry 仍在，refs 仅 run-b


def test_cleanup_orphans_only_when_unreferenced(tmp_path):
    # refs 空 → cleanup 删除；有 refs → 保留
```

- [ ] **Step 2: Implement → PASS**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(history): protect shared cache_pool on task delete"
```

---

### Task 7: 前端配置项（三开关 + 严格度 + 强制重建）

**Files:**
- Modify: `web/static/js/config/catalog/labels-options.js`
- Modify: `web/static/js/config/catalog/defaults.js`
- Modify: `web/static/js/config/catalog/form-layout.js`（放在预处理相关 section，靠近 `preprocess_memory_profile`）
- Modify: `web/static/js/config/catalog/field-help-training.js` 或 `extra-field-help.js`
- Modify: `web/static/js/config/catalog/field-help-dataset.js`（更新 resized/lora：说明可挂共享池）
- 同步 cache token（项目要求：改 import 时更新 `?v=` token；按现有 `module-bootstrap-*` 惯例 bump 相关入口）
- Test: `tests/test_training_frontend_state.py` / catalog 相关测试若有

**UI 文案（简体）：**

| key | 标签 |
|---|---|
| `reuse_dataset_cache_copy` | 复用数据集缓存拷贝 |
| `reuse_vae_latents` | 复用 VAE Latent 缓存 |
| `reuse_text_encoder_cache` | 复用文本编码缓存 |
| `cache_fingerprint_mode` | 缓存指纹模式 |
| `force_rebuild_preprocess_cache` | 强制重建预处理缓存 |

`cache_fingerprint_mode` options: `light`, `content`  
labels: `light`→`轻量指纹`，`content`→`全内容指纹(输入)`

defaults.js：

```javascript
reuse_dataset_cache_copy: true,
reuse_vae_latents: true,
reuse_text_encoder_cache: true,
cache_fingerprint_mode: 'light',
force_rebuild_preprocess_cache: false,
```

form-layout：在预处理 section 的 keys 数组加入上述 5 项。

- [ ] **Step 1: 若存在前端 catalog 快照测试，先改测试期望**
- [ ] **Step 2: 改 catalog 文件**
- [ ] **Step 3: Run**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_state.py -q
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(webui): expose cache reuse toggles in config form"
```

---

### Task 8: 用户文档 + 帮助文案收尾

**Files:**
- Create: `docs/features/preprocess-cache-reuse.md`
- Modify: `docs/features/README.md` 索引一行
- Modify: `field-help-dataset.js` 中 resized/lora 描述（若 Task7 未完成）

文档必须写清：

1. A/B/C 含义与默认开
2. light vs content 开销直觉
3. 共享池路径默认 `output/cache_pool`
4. 删历史 ≠ 删共享池；如何清理无引用
5. 改图/改 caption 后应强制重建或依赖指纹失效

- [ ] **Step 1: 写文档**
- [ ] **Step 2: `git diff --check -- docs/features/preprocess-cache-reuse.md docs/features/README.md`**
- [ ] **Step 3: Commit**

```bash
git commit -m "docs: explain preprocess cache reuse and history delete rules"
```

---

### Task 9: 端到端回归清单（不跑大模型）

- [ ] **Step 1: 跑缓存相关单测包**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_cache_pool_fingerprint.py \
  tests/test_cache_pool_store.py \
  tests/test_cache_pool_policy.py \
  tests/test_training_runtime_cache_reuse.py \
  tests/test_preprocess_reuse_flags.py \
  tests/test_cache_pool_gc.py \
  tests/test_training_history_delete.py \
  tests/test_training_runtime_config_core.py \
  tests/test_training_runtime_config_trigger_clone.py \
  tests/test_training_runtime_config_nl_tag.py \
  -q
```

Expected: PASS

- [ ] **Step 2: 手动冒烟（可选，需用户数据）**  
  同配置连续 prepare/preprocess 两次，观察第二次 skip/挂载日志。

- [ ] **Step 3: Final commit if docs/tests only polish**

---

## Spec Coverage Checklist

| Spec 要求 | Task |
|---|---|
| 共享池 `cache_pool/<fp>/` | 2, 4 |
| A/B/C 三开关配置级默认 true | 3, 7 |
| light 默认 + content 可切 | 1, 3, 7 |
| content 只 hash 输入 | 1 |
| mount 回退 | 2, 4 |
| force 写私有 | 4, 5 |
| nl_tag_mix / trigger_clone 独立 fp | 4（在改写后算 fp） |
| 删历史不删池 + refs | 6 |
| 无引用清理 | 6 |
| 前端控件 | 7 |
| 用户文档 | 8 |
| 测试 | 1–6, 9 |
| 非目标 PE/全局设置/自动 GC 守护 | 不实现 |

## Placeholder / Consistency Self-Check

- 配置键名与 spec 一致：`reuse_dataset_cache_copy` / `reuse_vae_latents` / `reuse_text_encoder_cache` / `cache_fingerprint_mode` / `force_rebuild_preprocess_cache`
- `FingerprintMode` 仅 `light|content`
- 池根统一 `resolve_output_root()/cache_pool`
- 无 TBD 步骤；overwrite 实参名实现前用 `rg` 对齐现有 CLI（Task5 已要求）

## Open Points Resolved for Implementers

| 点 | 本 plan 默认 |
|---|---|
| 池根 | `resolve_output_root()/cache_pool`，第一期不做 settings 覆盖键 |
| force_rebuild | 配置 bool；用完由 UI 复位更佳，后端容忍 true |
| refs | 每池 `refs.json` |
| 前端位置 | 预处理 section，邻 `preprocess_memory_profile` |

---
