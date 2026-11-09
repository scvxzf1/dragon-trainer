"""数据集图片目录预检：拦住 config_file 覆盖启动时静默继承空 image_dataset。

背景：WebUI 用 ``config_file`` 覆盖启动时，覆盖文件通常不含数据目录，
会继承 ``base.toml`` 的 ``source_image_dir = "image_dataset"``。该目录若是
空的，预处理生成 0 张缓存，直到 train_session 才报 ``No data found``，
错误信息指向训练侧、排查成本高。

这里直接单测检查函数本身，不走完整 preflight 的模型/依赖链。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from web.services.config.preflight import _check_dataset_image_dirs


def _run(cfg: dict) -> list[dict]:
    checks: list[dict] = []

    def add(level: str, key: str, message: str, path: Path | None = None, **flags):
        item = {"level": level, "key": key, "message": message}
        if path is not None:
            item["path"] = str(path)
        item.update(flags)
        checks.append(item)

    _check_dataset_image_dirs(cfg, add)
    return checks


def _levels(checks: list[dict]) -> list[str]:
    return [item["level"] for item in checks]


def test_reports_missing_dataset_dir(tmp_path: Path) -> None:
    cfg = {"datasets": [{"subsets": [{"image_dir": str(tmp_path / "nope")}]}]}

    checks = _run(cfg)

    assert _levels(checks) == ["error"]
    assert "不存在" in checks[0]["message"]


def test_reports_empty_dataset_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty_dataset"
    empty.mkdir()
    cfg = {"datasets": [{"subsets": [{"image_dir": str(empty)}]}]}

    checks = _run(cfg)

    assert _levels(checks) == ["error"]
    assert "没有图片" in checks[0]["message"]


def test_accepts_dataset_dir_with_images(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(dataset / "a.png")
    cfg = {"datasets": [{"subsets": [{"image_dir": str(dataset)}]}]}

    checks = _run(cfg)

    assert _levels(checks) == ["ok"]
    assert "可用" in checks[0]["message"]


def test_falls_back_to_source_image_dir_when_no_datasets(tmp_path: Path) -> None:
    """没有内联 datasets 时检查 source_image_dir（预处理源）。"""
    dataset = tmp_path / "src"
    dataset.mkdir()
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(dataset / "b.jpg")
    cfg = {"source_image_dir": str(dataset)}

    checks = _run(cfg)

    assert _levels(checks) == ["ok"]


def test_handles_dataset_level_image_dir(tmp_path: Path) -> None:
    """dataset 本身带 image_dir、没有 subsets 时也要检查。"""
    empty = tmp_path / "dataset_level"
    empty.mkdir()
    cfg = {"datasets": [{"image_dir": str(empty)}]}

    checks = _run(cfg)

    assert _levels(checks) == ["error"]
    assert "没有图片" in checks[0]["message"]


def test_ignores_non_image_files(tmp_path: Path) -> None:
    """只有 caption 没有图片，等于没有数据。"""
    dataset = tmp_path / "captions_only"
    dataset.mkdir()
    (dataset / "a.txt").write_text("a caption", encoding="utf-8")
    cfg = {"datasets": [{"subsets": [{"image_dir": str(dataset)}]}]}

    checks = _run(cfg)

    assert _levels(checks) == ["error"]
    assert "没有图片" in checks[0]["message"]


def test_duplicate_dirs_reported_once(tmp_path: Path) -> None:
    """同一目录在多个 subset 里出现时只报一次。"""
    empty = tmp_path / "dup"
    empty.mkdir()
    cfg = {
        "datasets": [
            {"subsets": [{"image_dir": str(empty)}]},
            {"subsets": [{"image_dir": str(empty)}]},
        ]
    }

    checks = _run(cfg)

    assert len([c for c in checks if c["level"] == "error"]) == 1


def test_silent_when_no_dataset_configured() -> None:
    """完全没有数据集配置时不发声，交给别的检查处理。"""
    assert _run({}) == []
