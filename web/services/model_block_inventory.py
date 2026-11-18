"""Inspect local DiT block keys without loading tensors or importing torch."""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re

from library.env import anima_home
from library.models.family_registry import get_model_family_spec
from web.services import path_safety

ROOT = anima_home()
_MAX_HEADER_BYTES = 64 * 1024 * 1024


@lru_cache(maxsize=32)
def _header_keys(path: str, mtime_ns: int, size: int, inode: int) -> tuple[str, ...]:
    # The stat signature invalidates hot results when a checkpoint is replaced.
    with Path(path).open("rb") as handle:
        length_bytes = handle.read(8)
        length = int.from_bytes(length_bytes, "little")
        if len(length_bytes) != 8 or not 0 < length <= min(size - 8, _MAX_HEADER_BYTES):
            raise ValueError("safetensors header 长度不合法")
        header = handle.read(length)
    _, keys = path_safety.read_safetensors_header_bytes(length_bytes + header)
    return tuple(keys)


def _keys(path: Path) -> tuple[str, ...]:
    stat = path.stat()
    return _header_keys(str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino)


def _weight_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".safetensors":
            raise ValueError("只支持 .safetensors 权重或本地模型目录")
        return [path]
    if not path.is_dir():
        raise ValueError("主模型路径不存在")
    directory = path / "transformer" if (path / "transformer").is_dir() else path
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if indexes:
        if len(indexes) != 1:
            raise ValueError("模型目录中有多个权重索引，请指定 transformer 目录")
        index = indexes[0]
        if index.stat().st_size > _MAX_HEADER_BYTES:
            raise ValueError("权重索引过大")
        data = json.loads(index.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("权重索引缺少 weight_map")
        files = []
        for name in sorted(set(weight_map.values())):
            if not isinstance(name, str) or Path(name).is_absolute() or path_safety.contains_parent_ref(name):
                raise ValueError("权重分片路径不合法")
            file = (directory / name).resolve()
            if not file.is_relative_to(directory.resolve()) or file.suffix.lower() != ".safetensors" or not file.is_file():
                raise ValueError("权重分片不存在或超出模型目录")
            files.append(file)
        return files
    files = sorted(directory.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError("模型目录需包含单个 safetensors 文件或分片索引")
    return files


def inspect_model_blocks(model_family: str, model_path: str) -> dict[str, object]:
    spec = get_model_family_spec(model_family)
    family = spec.name
    raw = path_safety.normalize_user_path_value(os.path.expandvars(model_path)).strip()
    if not raw:
        raise ValueError("请先填写基础模型路径")
    if path_safety.contains_parent_ref(raw):
        raise ValueError("主模型路径不允许包含 ..")
    path = path_safety.resolve_display_path(str(Path(raw).expanduser()), root=ROOT)
    assert path is not None
    files = _weight_files(path)
    prefix = spec.topology.block_container
    # Comfy checkpoints may wrap the same DiT keys in model.diffusion_model.
    pattern = re.compile(rf"^(?:net\.|model\.diffusion_model\.)?{prefix}\.(\d+)\.")
    indices = {int(match.group(1)) for file in files for key in _keys(file)
               if (match := pattern.match(key))}
    if not indices:
        raise ValueError(f"权重中未找到 {family} 的 {prefix}.N block 键")
    count = max(indices) + 1
    if len(indices) != count:
        raise ValueError("权重 block 索引必须从 0 开始且连续，无法确定实际层数")
    return {"ok": True, "model_family": family, "block_count": count,
            "max_swap": spec.topology.swap_limit_for(count), "source": "safetensors_header",
            "file_count": len(files)}
