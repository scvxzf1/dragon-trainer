"""Distributed training backend tests: diagnosis endpoint + hardware + guard + settings.

沿用现有 fake 模式（monkeypatch nvidia-smi runner / 临时 settings 文件），
不发真实 GPU 调用，不启动训练。
"""

from __future__ import annotations

import pytest

from web.services.settings_service import (
    DistributedSettingsError,
    get_distributed_settings,
    save_distributed_settings,
)
from web.services.training import distributed as dist
from web.services.training import gpu as gpu_helpers
from web.services.training.gpu import (
    classify_pp_hardware_risks,
    parse_gpu_hardware_rows,
    query_gpu_hardware_snapshot,
)


def _runner(rows: str):
    def run(_command):
        return rows

    return run


_HOST_HARDWARE_ROWS = "\n".join(
    [
        "0, GPU-410a51d5-c1e2-6616-b56d-ba2e3ee2aa1b, NVIDIA GeForce GTX 1050, 4096, 6.1",
        "1, GPU-b4b014b2-953b-28e8-b8c1-425c4a401165, Tesla V100-SXM2-16GB, 16384, 7.0",
        "2, GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061, Tesla V100-SXM2-16GB, 16384, 7.0",
    ]
)


# ---------------------------------------------------------------------------
# 模式归一：旧队列/旧设置无模式字段缺省 DDP
# ---------------------------------------------------------------------------


def test_normalize_distributed_mode_defaults_to_ddp():
    assert dist.normalize_distributed_mode(None) == "ddp"
    assert dist.normalize_distributed_mode("") == "ddp"
    assert dist.normalize_distributed_mode("PP") == "pp"
    assert dist.normalize_distributed_mode("single") == "single"
    assert dist.normalize_distributed_mode("bogus") == "ddp"
    assert dist.normalize_distributed_mode(None, default="single") == "single"


def test_item_distributed_mode_reads_old_queue_items():
    assert dist.item_distributed_mode({}) == "ddp"
    assert dist.item_distributed_mode({"distributed_mode": "pp"}) == "pp"
    assert dist.item_distributed_mode(None) == "ddp"


# ---------------------------------------------------------------------------
# 硬件快照：一次 nvidia-smi 查询 + 解析（compute_cap 缺失回退型号表）
# ---------------------------------------------------------------------------


def test_parse_gpu_hardware_rows():
    rows = parse_gpu_hardware_rows(_HOST_HARDWARE_ROWS)
    assert [row["index"] for row in rows] == [0, 1, 2]
    assert rows[0]["memory_total_mb"] == 4096
    assert rows[0]["capability"] == (6, 1)
    assert rows[1]["capability"] == (7, 0)
    # 老驱动没有 compute_cap 列：型号表兜底
    legacy = parse_gpu_hardware_rows("0, GPU-x, Tesla V100-SXM2-16GB, 16384")
    assert legacy[0]["capability"] == (7, 0)
    assert legacy[0]["memory_total_mb"] == 16384


def test_query_gpu_hardware_snapshot_uses_single_query():
    rows = query_gpu_hardware_snapshot(runner=_runner(_HOST_HARDWARE_ROWS))
    assert len(rows) == 3


def test_classify_pp_hardware_risks_three_layers():
    risks = {risk["key"] for risk in classify_pp_hardware_risks(
        parse_gpu_hardware_rows(_HOST_HARDWARE_ROWS)
    )}
    assert risks == {"pp_low_vram", "pp_vram_heterogeneous", "pp_mixed_generation"}


def test_classify_pp_hardware_risks_clean_pair_is_quiet():
    rows = parse_gpu_hardware_rows(
        "0, GPU-a, Tesla V100-SXM2-16GB, 16384, 7.0\n"
        "1, GPU-b, Tesla V100-SXM2-16GB, 16384, 7.0"
    )
    assert classify_pp_hardware_risks(rows) == []


def test_hardware_diagnosis_is_pp_only():
    pp = dist.hardware_diagnosis("pp", [0, 1], runner=_runner(_HOST_HARDWARE_ROWS))
    ddp = dist.hardware_diagnosis("ddp", [0, 1], runner=_runner(_HOST_HARDWARE_ROWS))
    assert pp["risks"]
    assert ddp["risks"] == []
    assert [row["index"] for row in ddp["gpus"]] == [0, 1]


# ---------------------------------------------------------------------------
# wired 族表
# ---------------------------------------------------------------------------


def test_wired_pp_families_marks_wired_and_unwired():
    families = {item["family"]: item for item in dist.wired_pp_families()}
    assert families["moss_image"]["wired"] is True
    assert families["krea2_raw"]["wired"] is False
    assert families["moss_image"]["stages"]


# ---------------------------------------------------------------------------
# settings [distributed] 节：读写 + 校验
# ---------------------------------------------------------------------------


def test_distributed_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "web.services.settings_service.SETTINGS_FILE", tmp_path / "web-ui-settings.toml"
    )
    default_payload = get_distributed_settings()
    assert default_payload["mode"] == "ddp"
    assert default_payload["gpu_whitelist"] == []

    saved = save_distributed_settings({
        "mode": "pp",
        "gpu_whitelist": ["2", 1, "bad", -1, 1],
        "pipeline_parallel_stages": 2,
        "pipeline_parallel_microbatches": 4,
    })
    assert saved["mode"] == "pp"
    assert saved["gpu_whitelist"] == [2, 1]
    assert get_distributed_settings()["mode"] == "pp"

    # 其他节不受影响（设置通道保持原样合并）
    import toml

    raw = toml.loads((tmp_path / "web-ui-settings.toml").read_text(encoding="utf-8"))
    assert "distributed" in raw


def test_distributed_settings_validation_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "web.services.settings_service.SETTINGS_FILE", tmp_path / "web-ui-settings.toml"
    )
    with pytest.raises(DistributedSettingsError):
        save_distributed_settings({"mode": "tp"})
    with pytest.raises(DistributedSettingsError):
        save_distributed_settings({"pipeline_parallel_stages": 0})
    with pytest.raises(DistributedSettingsError):
        save_distributed_settings({"pipeline_parallel_schedule": "gpipe"})


# ---------------------------------------------------------------------------
# 数据集就绪诊断（PP 模式，fake 目录 + census/cache 复用）
# ---------------------------------------------------------------------------


def _write_dataset_toml(tmp_path, subsets):
    lines = ["[[datasets]]", ""]
    for subset in subsets:
        lines.append("[[datasets.subsets]]")
        for key, value in subset.items():
            lines.append(f"{key} = {value!r}")
        lines.append("")
    return "\n".join(lines)


def test_diagnose_pp_dataset_readiness_bucket_mismatch(tmp_path, monkeypatch):
    import toml

    from web.services.config import datasets as datasets_mod

    subset_a = tmp_path / "subset-a"
    subset_b = tmp_path / "subset-b"
    subset_a.mkdir()
    subset_b.mkdir()
    # A: 1024x1024（4200 族）; B: 512x768（不同桶）→ 多子集桶不一致
    from PIL import Image

    Image.new("RGB", (1024, 1024)).save(subset_a / "a.png")
    Image.new("RGB", (512, 768)).save(subset_b / "b.png")

    dataset_toml = tmp_path / "dataset.toml"
    dataset_toml.write_text(_write_dataset_toml(tmp_path, [
        {"image_dir": str(subset_a), "num_repeats": 1},
        {"image_dir": str(subset_b), "num_repeats": 1},
    ]), encoding="utf-8")

    cfg = toml.loads(dataset_toml.read_text(encoding="utf-8"))
    cfg["dataset_config"] = str(dataset_toml)
    monkeypatch.setattr(
        "web.services.config.datasets._dataset_config_path_from_cfg",
        lambda _cfg: dataset_toml,
    )
    checks = dist.diagnose_pp_dataset_readiness(cfg)
    keys = {check["key"] for check in checks}
    assert "dataset_bucket_mismatch" in keys
    mismatch = next(c for c in checks if c["key"] == "dataset_bucket_mismatch")
    assert mismatch["level"] == "error"


def test_diagnose_pp_dataset_readiness_uniform_single_bucket(tmp_path, monkeypatch):
    import toml
    from PIL import Image

    from web.services.config import datasets as datasets_mod

    subset = tmp_path / "subset-a"
    subset.mkdir()
    Image.new("RGB", (1024, 1024)).save(subset / "a.png")
    Image.new("RGB", (1024, 1024)).save(subset / "b.png")

    dataset_toml = tmp_path / "dataset.toml"
    dataset_toml.write_text(_write_dataset_toml(tmp_path, [
        {"image_dir": str(subset), "num_repeats": 1},
    ]), encoding="utf-8")

    cfg = toml.loads(dataset_toml.read_text(encoding="utf-8"))
    cfg["dataset_config"] = str(dataset_toml)
    monkeypatch.setattr(
        "web.services.config.datasets._dataset_config_path_from_cfg",
        lambda _cfg: dataset_toml,
    )
    checks = dist.diagnose_pp_dataset_readiness(cfg)
    keys = {check["key"] for check in checks}
    assert "dataset_bucket_mismatch" not in keys
    assert "dataset_single_subset" in keys
    assert "dataset_bucket_uniform" in keys


# ---------------------------------------------------------------------------
# 顶层诊断入口：单卡/DDP 模式不产出 PP 检查；配置装载失败降级不抛
# ---------------------------------------------------------------------------


def test_diagnose_distributed_training_ddp_mode_has_no_pp_checks(monkeypatch):
    import web.services.config_service as config_service

    monkeypatch.setattr(
        config_service,
        "_load_training_config_for_web_run",
        lambda *a, **k: {"model_family": "moss_image"},
    )
    payload = dist.diagnose_distributed_training(
        "lora", "default", "gui-methods", mode="ddp", gpu_whitelist=[],
    )
    assert payload["ok"] is True
    assert payload["mode"] == "ddp"
    assert payload["checks"] == []
    assert payload["wired_families"]


def test_diagnose_distributed_training_config_unreadable_degrades(monkeypatch):
    import web.services.config_service as config_service

    def _boom(*_a, **_k):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(config_service, "_load_training_config_for_web_run", _boom)
    payload = dist.diagnose_distributed_training(
        "lora", "default", "gui-methods", mode="pp", gpu_whitelist=[],
    )
    assert payload["ok"] is True
    assert any(check["key"] == "config_unreadable" for check in payload["checks"])


def test_gpu_identity_guard_still_fails_closed_on_unresolvable_index():
    from web.services.training.gpu import ensure_gpu_whitelist_authorized

    with pytest.raises(ValueError) as excinfo:
        ensure_gpu_whitelist_authorized([7], runner=_runner(_HOST_HARDWARE_ROWS))
    assert gpu_helpers.GPU_WHITELIST_UNAUTHORIZED in str(excinfo.value)


# ---------------------------------------------------------------------------
# PP admission diagnosis: one entry (family-aware), human wording + raw detail
# ---------------------------------------------------------------------------

_PP_CFG = {
    "model_family": "anima",
    "pipeline_parallel": True,
    "pipeline_parallel_stages": 2,
    "pipeline_parallel_microbatches": 2,
    "pipeline_parallel_schedule": "1f1b",
    "pipeline_parallel_split": "balanced",
    "mixed_precision": "fp16",
    "gradient_accumulation_steps": 1,
    # The DiT-only LoRA knob Anima's admission requires; without it the run is
    # refused with pp_config_invalid, which is exactly what we assert below.
}


def test_diagnose_pp_config_uses_the_family_aware_admission():
    """Regression: this used to import a name trainer_admission never exported."""

    checks = dist.diagnose_pp_config(dict(_PP_CFG), 2)
    keys = {check["key"] for check in checks}
    assert "diagnosis_unavailable" not in keys
    refused = [check for check in checks if check["key"] == "config_invalid"]
    assert refused, checks
    # Human wording on the primary line, the raw refusal kept as detail.
    assert "PP 参数" in refused[0]["message"]
    assert "pp_config_invalid" in refused[0]["detail"]


def test_diagnose_pp_config_explains_an_unwired_family_in_plain_words():
    cfg = dict(_PP_CFG, model_family="krea2_raw")
    checks = dist.diagnose_pp_config(cfg, 2)
    unwired = [check for check in checks if check["key"] == "trainer_path_unavailable"]
    assert unwired, checks
    assert "还没有接入流水线并行训练" in unwired[0]["message"]
    assert "Anima" in unwired[0]["message"]
    assert "pp_trainer_path_unavailable" in unwired[0]["detail"]


@pytest.mark.parametrize(
    "exc",
    [
        ImportError("cannot import name 'x' from 'y'"),
        ModuleNotFoundError("No module named 'z'"),
        AttributeError("module has no attribute 'q'"),
        NameError("name 'w' is not defined"),
    ],
)
def test_diagnosis_self_faults_degrade_instead_of_blaming_the_config(exc):
    """A broken diagnosis import must not read as 'your PP config is wrong'."""

    level, key, message, detail = dist._classify_admission_failure(exc)
    assert level == "warning"
    assert key == "diagnosis_unavailable"
    assert "与你的配置无关" in message
    assert type(exc).__name__ in detail


# ---------------------------------------------------------------------------
# PP 通信路径诊断（info only，永不拦截）
# ---------------------------------------------------------------------------


def test_comm_path_check_reports_nvlink(monkeypatch):
    from web.services.training import distributed as dist_mod

    class _Proc:
        returncode = 0
        stdout = (
            "        GPU0    GPU1    CPU Affinity\n"
            "GPU0     X      NV4     0-35\n"
            "GPU1    NV4      X      0-35\n"
        )

    monkeypatch.setattr(dist_mod.subprocess, "run", lambda *a, **k: _Proc())
    check = dist_mod._pp_comm_path_check()
    assert check["key"] == "comm_path_nvlink"
    assert check["level"] == "info"
    assert check["code"] == "pp_comm_path_nvlink"
    assert "NV4" in check["message"]


def test_comm_path_check_reports_pcie_without_nvlink(monkeypatch):
    from web.services.training import distributed as dist_mod

    class _Proc:
        returncode = 0
        stdout = (
            "        GPU0    GPU1    CPU Affinity\n"
            "GPU0     X      PHB     0-35\n"
            "GPU1    PHB      X      0-35\n"
        )

    monkeypatch.setattr(dist_mod.subprocess, "run", lambda *a, **k: _Proc())
    check = dist_mod._pp_comm_path_check()
    assert check["key"] == "comm_path_pcie"
    assert check["level"] == "info"
    assert "PHB" in check["message"]


def test_comm_path_check_returns_none_when_topology_unavailable(monkeypatch):
    from web.services.training import distributed as dist_mod

    class _Proc:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(dist_mod.subprocess, "run", lambda *a, **k: _Proc())
    assert dist_mod._pp_comm_path_check() is None

    def _raise(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(dist_mod.subprocess, "run", _raise)
    assert dist_mod._pp_comm_path_check() is None


def test_topology_parser_dedupes_pairs_and_ignores_diagonal():
    from web.services.training.distributed import _parse_topology_pairs

    text = (
        "        GPU0    GPU1    GPU2    CPU Affinity\n"
        "GPU0     X      PHB     SYS     0-35\n"
        "GPU1    PHB      X      NV4     0-35\n"
        "GPU2    SYS     NV4     X       0-35\n"
    )
    pairs = _parse_topology_pairs(text)
    assert pairs == [(0, 1, "PHB"), (0, 2, "SYS"), (1, 2, "NV4")]
