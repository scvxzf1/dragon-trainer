"""Krea-2 fp16 两个实验开关的端到端接线审计。

四条链路各自焊死：

1. **一致性**：``nf4_dir1`` / ``krea2_attn_fp16`` 同时存在于
   ``configs/base.toml``（默认 False）、``load_merged_config``、``FIELD_HELP``、
   ``FORM_GROUPS["Performance"]``、``get_groups()`` 输出和前端
   ``config-field-catalog.js`` / Label / Tooltip；并守卫两个键不会被
   ``RETIRED_TOP_LEVEL_CONFIG_FIELDS`` 或 ``UI_ONLY_CONFIG_FIELDS`` 静默删除。
2. **往返**：``PATCH /api/config/raw`` → 合并回读 → Web runtime TOML →
   ``train.py`` 的 argparse（``read_config_from_file``）每一步都存活为 True。
3. **Preflight 矩阵**：sm70(7,0) / sm86(8,6) × NF4 / 非 NF4 × Krea-2 / 非 Krea-2
   覆盖四类设计分支，并额外校验 Web 预检的探针接线与中文文案；
   ``info`` 级联动提示还要能一路显示到前端 preflight 面板，
   不能被并入 error/warning 或丢弃。
4. **autocast 防截胡**：bf16 autocast 下 dir1 内层 GEMM 与 attention 内层 SDPA
   必须仍以 fp16 执行（前反向），并守卫既有 GPU 回归用例还在。

测试只使用合成权重、monkeypatch 的 GPU capability 与隔离临时配置根，
不依赖真实模型文件，也不落任何仓库内运行数据。
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import re
from pathlib import Path
from unittest import mock

import pytest
import toml
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SWITCHES = ("nf4_dir1", "krea2_attn_fp16")
FRONTEND_DOMAIN = (
    REPO_ROOT
    / "web"
    / "frontend-next"
    / "src"
    / "features"
    / "training-config"
    / "domain"
)
TRAINING_CONFIG_UI = (
    REPO_ROOT / "web" / "frontend-next" / "src" / "features" / "training-config"
)
SM70 = (7, 0)
SM86 = (8, 6)
KREA2_NF4 = {"model_family": "krea2_raw", "base_compute": "nf4"}
CUDA_AVAILABLE = torch.cuda.is_available()
DIR1_DEVICE_SUPPORTED = CUDA_AVAILABLE and torch.cuda.get_device_capability()[0] < 8


@pytest.fixture(autouse=True)
def _preflight_facade_ready():
    """preflight_compat 的 ``_bool_value`` 等 helper 由 facade 注入。

    直接 import ``preflight_compat`` 时这些名字还是「未同步」占位符，
    先跑一次 ``_sync_from_facade`` 让单独运行本文件也成立。
    """
    from web.services.config.preflight_runtime import _sync_from_facade

    _sync_from_facade()


def _read_frontend(rel_path: str) -> str:
    path = FRONTEND_DOMAIN / rel_path
    assert path.is_file(), f"前端文件缺失: {path}"
    return path.read_text(encoding="utf-8")


def _read_training_config_ui(rel_path: str) -> str:
    path = TRAINING_CONFIG_UI / rel_path
    assert path.is_file(), f"前端文件缺失: {path}"
    return path.read_text(encoding="utf-8")


def _levels(checks: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for level, key, _message in checks:
        grouped.setdefault(level, []).append(key)
    return grouped


# ---------------------------------------------------------------------------
# ① 一致性审计
# ---------------------------------------------------------------------------


def test_switch_surfaces_stay_aligned():
    """同一对键必须在配置、合并、表单字典和前端目录里同时出现。"""
    from web.services import config_service
    from web.services.config import metadata

    base = toml.loads((REPO_ROOT / "configs" / "base.toml").read_text(encoding="utf-8"))
    merged = config_service.load_merged_config("lora", "default", "gui-methods")
    groups = metadata.get_groups()["groups"]
    catalog = _read_frontend("config-field-catalog.js")
    labels = _read_frontend("labels-options.js")
    summaries = _read_frontend("field-help-summary.js")
    help_resources = _read_frontend("help/resources.js")

    for key in SWITCHES:
        assert base.get(key) is False, (
            f"configs/base.toml 必须声明 {key} = false（默认关闭的实验开关）"
        )
        assert merged.get(key) is False, (
            f"load_merged_config 默认必须带出 {key}=False，当前 {merged.get(key)!r}"
        )
        help_entry = metadata.FIELD_HELP.get(key)
        assert help_entry, f"FIELD_HELP 缺少 {key}，WebUI 会渲染裸露 key"
        assert help_entry.get("en", "").strip(), f"FIELD_HELP[{key}]['en'] 不能为空"
        assert help_entry.get("ko", "").strip(), f"FIELD_HELP[{key}]['ko'] 不能为空"
        assert key in metadata.FORM_GROUPS["Performance"], (
            f"{key} 不在 FORM_GROUPS['Performance']，表单不会渲染它"
        )
        assert key in groups["Performance"], (
            f"get_groups() 的 Performance 分组丢失 {key}"
        )
        assert re.search(rf"^\s*'{re.escape(key)}',?\s*$", catalog, re.MULTILINE), (
            f"前端 config-field-catalog.js 缺少 {key} 的数组条目"
        )
        assert re.search(rf"^\s*{re.escape(key)}\s*:\s*['\"]", labels, re.MULTILINE), (
            f"前端 labels-options.js 缺少 {key} 的中文 Label 赋值项"
        )
        assert re.search(
            rf'^\s*"{re.escape(key)}"\s*:\s*["\']', summaries, re.MULTILINE
        ), f"前端 field-help-summary.js 缺少 {key} 的摘要赋值项"
        assert re.search(rf"\b{re.escape(key)}:\s*\{{", help_resources), (
            f"前端 help/resources.js 缺少 {key} 的帮助条目对象"
        )


def test_experimental_switches_are_not_retired_or_ui_only():
    """F2 守卫：两键一旦落进这两个集合，PATCH 保存时会被静默删除。

    ``raw_files._prepare_raw_file_patch`` 先按 ``UI_ONLY_CONFIG_FIELDS`` 和
    ``RETIRED_TOP_LEVEL_CONFIG_FIELDS`` 过滤 values，再
    ``_remove_retired_top_level_fields`` 从现有 TOML 里删键。
    """
    from web.services.config import metadata

    for key in SWITCHES:
        assert key not in metadata.RETIRED_TOP_LEVEL_CONFIG_FIELDS, (
            f"{key} 进了 RETIRED_TOP_LEVEL_CONFIG_FIELDS，保存时会被静默删除"
        )
        assert key not in metadata.UI_ONLY_CONFIG_FIELDS, (
            f"{key} 进了 UI_ONLY_CONFIG_FIELDS，PATCH 会直接丢弃这个值"
        )


# ---------------------------------------------------------------------------
# ② 保存 / 合并 / 启动往返审计
# ---------------------------------------------------------------------------


def test_patch_save_merge_runtime_and_argparse_roundtrip(tmp_path, monkeypatch):
    """配置置 True 后，保存→合并→runtime TOML→argparse 每一跳都不能丢。"""
    from tests.training_resume_test_support import (
        _FakeJsonRequest,
        _patch_runtime_service_paths,
        _write_runtime_config_tree,
    )
    from web.routes import config as config_routes
    from web.services import config_service, training_service

    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    source_rel = "configs/imported/522.toml"

    response = asyncio.run(
        config_routes.handle_raw_patch(
            _FakeJsonRequest(
                {
                    "file": source_rel,
                    "values": {key: True for key in SWITCHES},
                }
            )
        )
    )
    body = json.loads(response.text or "{}")
    assert response.status == 200, body
    assert set(body["changed"]) == set(SWITCHES), body

    saved = toml.loads((tmp_path / source_rel).read_text(encoding="utf-8"))
    merged = config_service.load_merged_config("522", "default", "imported")
    runtime = training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file=source_rel,
    )
    runtime_path = tmp_path / runtime["runtime_config_file"]
    runtime_cfg = toml.loads(runtime_path.read_text(encoding="utf-8"))

    for key in SWITCHES:
        assert saved.get(key) is True, f"PATCH 落盘后 {key} 丢失: {saved}"
        assert merged.get(key) is True, f"合并回读后 {key} 丢失: {merged.get(key)!r}"
        assert runtime_cfg.get(key) is True, (
            f"冻结进 runtime TOML 后 {key} 丢失: {runtime_cfg.get(key)!r}"
        )

    # 启动链路：launcher 只把 runtime TOML 作为 --config_file 交给 train.py，
    # 这里复用同一个 argparse 合并入口，确认两键最终解析为 True。
    from library.config.io import read_config_from_file
    from library.training.cli_args import setup_parser

    parser = setup_parser()
    argv = [
        "--method",
        "522",
        "--preset",
        "default",
        "--methods_subdir",
        "imported",
        "--config_file",
        str(runtime_path),
    ]
    parsed = read_config_from_file(parser.parse_args(argv), parser, argv=argv)
    assert parsed.nf4_dir1 is True, "runtime TOML 里的 nf4_dir1 没有进入 argparse"
    assert parsed.krea2_attn_fp16 is True, (
        "runtime TOML 里的 krea2_attn_fp16 没有进入 argparse"
    )


# ---------------------------------------------------------------------------
# ③ Preflight 规则矩阵
# ---------------------------------------------------------------------------


def _compat_result(config: dict, **kwargs):
    from library.training.compat_matrix import check_training_compat

    return check_training_compat(config, **kwargs)


def _codes(issues) -> set[str]:
    return {issue.code for issue in issues}


def _keys(issues) -> set[str]:
    return {issue.key for issue in issues}


def test_preflight_matrix_matches_design():
    """四类设计分支：架构 blocker / 忽略 warning / 双开 info / 全关静默。"""
    # 场景 0：两个开关都关，任何硬件与模型族都不产生风险项。
    assert _compat_result(dict(KREA2_NF4)).errors == ()
    assert _compat_result(dict(KREA2_NF4), cuda_available=False).warnings == ()
    assert _compat_result(dict(KREA2_NF4), cuda_available=True, device_capability=SM86).warnings == ()

    # 场景 1a：sm70 + Krea-2 + NF4 双开 → 放行，且不再提示双开。
    both_on = {**KREA2_NF4, "nf4_dir1": True, "krea2_attn_fp16": True}
    result = _compat_result(both_on, cuda_available=True, device_capability=SM70)
    assert result.errors == ()
    assert result.warnings == ()
    assert result.infos == ()

    # 场景 4（联动提示）：sm70 + Krea-2 + NF4 只开一个 → info 指向缺失的那个。
    for enabled, missing in (("nf4_dir1", "krea2_attn_fp16"), ("krea2_attn_fp16", "nf4_dir1")):
        result = _compat_result(
            {**KREA2_NF4, enabled: True},
            cuda_available=True,
            device_capability=SM70,
        )
        assert result.errors == ()
        pairing = [item for item in result.infos if item.code == "nf4_fp16_inner_pairing"]
        assert pairing, f"只开 {enabled} 时应提示补齐 {missing}"
        assert _keys(pairing) == {missing}
        assert "sm70" in pairing[0].message

    # 场景 1b：sm86（原生 BF16 tensor core）双开 → 两个 pre-Ampere blocker。
    result = _compat_result(both_on, cuda_available=True, device_capability=SM86)
    assert _codes(result.errors) == {
        "nf4_dir1_requires_pre_ampere",
        "krea2_attn_fp16_requires_pre_ampere",
    }
    assert _keys(result.errors) == set(SWITCHES)
    assert all("sm86" in item.message for item in result.errors)

    # 场景 1c：无 CUDA / 架构未知 → 各自 blocker，不能假装 sm70 放行。
    no_cuda = _compat_result(both_on, cuda_available=False)
    # 场景 1c：无 CUDA 时 nf4_dir1 仍是 blocker（运行时 set_dir1_enabled 抛
    # RuntimeError），krea2_attn_fp16 的运行时空转不抛错，只降级为 warning。
    no_cuda = _compat_result(both_on, cuda_available=False)
    assert _codes(no_cuda.errors) == {"nf4_dir1_requires_cuda"}
    assert _codes(no_cuda.warnings) == {"krea2_attn_fp16_requires_cuda"}
    # 架构读不到：两者都无法证明 sm70/sm75，统一降级为 warning（不阻断）。
    unknown_arch = _compat_result(both_on, cuda_available=True, device_capability=None)
    assert unknown_arch.errors == ()
    assert _codes(unknown_arch.warnings) == {
        "nf4_dir1_architecture_unknown",
        "krea2_attn_fp16_architecture_unknown",
    }
    # 没有硬件上下文时不臆造结论，交给运行时 set_* 门禁 fail closed。
    assert _compat_result(both_on).errors == ()

    # 场景 2：非 NF4 开 nf4_dir1 → warning（sm70 下没有 NF4 Linear 可加速）。
    result = _compat_result(
        {"model_family": "krea2_raw", "base_compute": "bf16", "nf4_dir1": True},
        cuda_available=True,
        device_capability=SM70,
    )
    assert result.errors == ()
    assert _codes(result.warnings) == {"nf4_dir1_ignored"}
    assert _keys(result.warnings) == {"nf4_dir1"}

    # 场景 3：非 Krea-2 开 krea2_attn_fp16 → warning（该族从不走 Krea-2 attention）。
    result = _compat_result(
        {"model_family": "anima", "base_compute": "nf4", "krea2_attn_fp16": True},
        cuda_available=True,
        device_capability=SM70,
    )
    assert result.errors == ()
    assert _codes(result.warnings) == {"krea2_attn_fp16_ignored"}
    assert _keys(result.warnings) == {"krea2_attn_fp16"}

    # 已 blocker 时不再补一条"被忽略"的噪声 warning。
    result = _compat_result(
        {"model_family": "krea2_raw", "base_compute": "bf16", "nf4_dir1": True},
        cuda_available=True,
        device_capability=SM86,
    )
    assert _codes(result.errors) == {"nf4_dir1_requires_pre_ampere"}
    assert result.warnings == ()


def _web_compat_checks(
    cfg: dict, probe=None, *, gpu_whitelist=None
) -> list[tuple[str, str, str]]:
    from web.services.config import preflight_compat

    checks: list[tuple[str, str, str]] = []

    def add(level, key, message, **_flags):
        checks.append((level, key, message))

    context = (
        mock.patch.object(
            preflight_compat, "_probe_cuda_device_for", lambda _cfg: probe
        )
        if probe is not None
        else contextlib.nullcontext()
    )
    with context:
        preflight_compat._check_checkpointing_config(
            cfg, add, gpu_whitelist=gpu_whitelist
        )
    return checks


def test_web_preflight_probe_and_messages_follow_the_same_matrix():
    """Web 预检必须把探针结果喂给 compat，并输出可执行的中文提示。"""
    from web.services.config import preflight_compat

    both_on = {**KREA2_NF4, "nf4_dir1": True, "krea2_attn_fp16": True}

    # 两个开关都关时不触碰 CUDA：探针本身不允许被调用。
    with mock.patch.object(
        preflight_compat,
        "_probe_cuda_device",
        side_effect=AssertionError("开关全关时不应探测 CUDA"),
    ):
        assert preflight_compat._probe_cuda_device_for(dict(KREA2_NF4)) is None

    # sm70 双开：完全放行。
    assert _web_compat_checks(both_on, preflight_compat._CudaDeviceProbe(True, SM70)) == []

    # sm70 只开一个：info 建议双开，key 指向缺失项。
    for enabled, missing in (("nf4_dir1", "krea2_attn_fp16"), ("krea2_attn_fp16", "nf4_dir1")):
        checks = _web_compat_checks(
            {**KREA2_NF4, enabled: True},
            preflight_compat._CudaDeviceProbe(True, SM70),
        )
        levels = _levels(checks)
        assert set(levels) == {"info"}, checks
        assert levels["info"] == [missing]
        assert "建议两个同时开启" in checks[0][2]

    # sm86 双开：两条 error，并带上当前设备号。
    checks = _web_compat_checks(
        both_on, preflight_compat._CudaDeviceProbe(True, SM86)
    )
    levels = _levels(checks)
    assert set(levels) == {"error"}, checks
    assert levels["error"] == list(SWITCHES)
    assert all("sm_86" in message for _level, _key, message in checks)
    assert any("set_dir1_enabled(True)" in message for _l, _k, message in checks)
    assert any("set_krea2_attn_fp16(True)" in message for _l, _k, message in checks)

    # 无 CUDA：nf4_dir1 的运行时门禁会抛 RuntimeError → error；
    # krea2_attn_fp16 在无 CUDA 时只是空转 → 只给 warning。
    checks = _web_compat_checks(
        both_on, preflight_compat._CudaDeviceProbe(False, None)
    )
    levels = _levels(checks)
    assert levels == {"error": ["nf4_dir1"], "warning": ["krea2_attn_fp16"]}, checks
    assert all("CUDA" in message for _l, _k, message in checks)
    assert "阻止启动" in checks[0][2]

    # 架构未知：无法确认 sm70/sm75，两者都降级为 warning（不阻断）。
    checks = _web_compat_checks(
        both_on, preflight_compat._CudaDeviceProbe(True, None)
    )
    levels = _levels(checks)
    assert levels == {
        "warning": ["nf4_dir1", "krea2_attn_fp16"],
    }, checks
    assert all("compute capability" in message for _l, _k, message in checks)


def test_web_preflight_flags_attention_switch_dead_under_flash_or_fp16():
    """F1：attn_mode=flash / 非 bf16 compute 下 krea2_attn_fp16 是死开关。"""
    from web.services.config import preflight_compat

    probe = preflight_compat._CudaDeviceProbe(True, SM70)

    # flash + 双开：只有该 warning，不再产生 pairing info。
    checks = _web_compat_checks(
        {**KREA2_NF4, "attn_mode": "flash", "nf4_dir1": True, "krea2_attn_fp16": True},
        probe,
    )
    levels = _levels(checks)
    assert levels == {"warning": ["krea2_attn_fp16"]}, checks
    assert "attn_mode=torch" in checks[0][2]

    # flash + 只开 nf4_dir1：不能建议打开一个在 flash 下不生效的开关。
    assert (
        _web_compat_checks(
            {**KREA2_NF4, "attn_mode": "flash", "nf4_dir1": True}, probe
        )
        == []
    )

    # torch/sdpa 恢复 pairing info。
    for attn_mode in ("torch", "sdpa"):
        checks = _web_compat_checks(
            {**KREA2_NF4, "attn_mode": attn_mode, "nf4_dir1": True}, probe
        )
        assert _levels(checks) == {"info": ["krea2_attn_fp16"]}, checks

    # 非 bf16 compute 同样让开关空转。
    checks = _web_compat_checks(
        {
            **KREA2_NF4,
            "attn_mode": "torch",
            "mixed_precision": "fp16",
            "krea2_attn_fp16": True,
        },
        probe,
    )
    assert _levels(checks) == {"warning": ["krea2_attn_fp16"]}, checks


def test_web_preflight_whitelist_uses_physical_gpu_not_device0_probe(monkeypatch):
    """F2：whitelist 非空时按物理 index 解析，不能回退 Web 进程 device-0。"""
    from web.services.config import preflight_compat
    from web.services.training import gpu as gpu_module

    monkeypatch.setattr(
        gpu_module,
        "query_gpu_compute_capabilities",
        lambda indices=None, **_kwargs: {0: SM86, 1: SM70},
    )

    def _no_device0_probe(_cfg):  # pragma: no cover - 断言用
        raise AssertionError("whitelist 非空时不应回退 device-0 探针")

    monkeypatch.setattr(preflight_compat, "_probe_cuda_device_for", _no_device0_probe)

    # 只选物理 index 1（V100 sm70）：放行，info 提到 sm70 而不是 Web 看到的 sm86。
    checks = _web_compat_checks(
        {**KREA2_NF4, "attn_mode": "torch", "nf4_dir1": True},
        gpu_whitelist=[1],
    )
    assert _levels(checks) == {"info": ["krea2_attn_fp16"]}, checks
    assert "sm70" in checks[0][2]
    assert "sm86" not in checks[0][2]

    # 同时选中 3080(sm86) + V100(sm70)：任一 sm80+ 就阻断。
    checks = _web_compat_checks(
        {**KREA2_NF4, "attn_mode": "torch", "nf4_dir1": True},
        gpu_whitelist=[0, 1],
    )
    assert _levels(checks) == {"error": ["nf4_dir1"]}, checks
    assert "sm_86" in checks[0][2] and "sm_70" in checks[0][2]

    # 被选设备解析不到 capability：降级为 warning，不阻断。
    monkeypatch.setattr(
        gpu_module,
        "query_gpu_compute_capabilities",
        lambda indices=None, **_kwargs: {},
    )
    checks = _web_compat_checks(
        {**KREA2_NF4, "attn_mode": "torch", "nf4_dir1": True},
        gpu_whitelist=[1],
    )
    assert _levels(checks) == {"warning": ["nf4_dir1"]}, checks
    assert "compute capability" in checks[0][2]


def test_routes_and_preflight_forward_gpu_whitelist():
    """F2 接线：每个 preflight 调用点都必须透传 gpu_whitelist。"""
    routes_source = (REPO_ROOT / "web" / "routes" / "training.py").read_text(
        encoding="utf-8"
    )
    calls = re.findall(
        r"preflight_training_config\((.*?)\n\s*\)", routes_source, re.DOTALL
    )
    assert len(calls) >= 3, f"routes 应至少有 3 个 preflight 调用点，实际 {len(calls)}"
    missing = [call for call in calls if "gpu_whitelist=" not in call]
    assert not missing, f"有 preflight 调用点没有透传 gpu_whitelist: {missing}"

    preflight_source = (REPO_ROOT / "web" / "services" / "config" / "preflight.py").read_text(
        encoding="utf-8"
    )
    assert "gpu_whitelist: Any = None" in preflight_source, (
        "preflight_training_config 必须接受 gpu_whitelist 关键字（默认 None）"
    )
    assert re.search(
        r"_check_checkpointing_config\([^)]*gpu_whitelist=gpu_whitelist",
        preflight_source,
        re.DOTALL,
    ), "preflight() 必须把 gpu_whitelist 传给 _check_checkpointing_config"

    # 预处理完成后的自动启动复检：这一跳手里有 pending 的物理 index 白名单，
    # 若不带下去就会退回 Web 进程 device-0，异构机上按错卡判定架构。
    launcher_source = (
        REPO_ROOT / "web" / "services" / "training" / "launcher_start.py"
    ).read_text(encoding="utf-8")
    assert "preflight_training_config(" in launcher_source, (
        "launcher_start 应有 preflight 调用"
    )
    assert 'gpu_whitelist=pending.get("gpu_whitelist")' in launcher_source, (
        "launcher_start 的复检必须透传 pending 的 gpu_whitelist，"
        "否则会退回 Web 进程 device-0"
    )


def test_info_pairing_hint_reaches_the_frontend_preflight_panel():
    """info 级联动提示必须一路显示到前端 preflight 面板，不能混同或丢弃。

    后端：Web 预检的 ``add()`` 只有 error/warning 两个收集分支，
    info 仍会 append 进 ``checks``（``summary.checks`` 计入），因此不会被
    ``errors``/``warnings`` 吞掉，也不会被误算成警告。
    前端：``TrainingPreflightPanel`` 为 info 单独渲染「提示」标签并保留
    ``data-level``，排序 rank 是 error < warning < info < ok；既有前端单测
    ``TrainingLaunchDialog.test.tsx`` 已覆盖 info 行的渲染与排序。
    """
    from web.services.config import preflight as preflight_module
    from web.services.config import preflight_compat

    add_source = inspect.getsource(preflight_module)
    assert 'if level == "error":\n            errors.append(item)' in add_source, (
        "preflight 的 add() 必须只把 error 收进 errors"
    )
    assert 'elif level == "warning":\n            warnings.append(item)' in add_source, (
        "preflight 的 add() 必须只把 warning 收进 warnings（info 不得混入）"
    )
    assert add_source.index("checks.append(item)") < add_source.index(
        'if level == "error"'
    ), "checks 必须先于 level 分流 append，info 才会保留在 checks 里"

    # Web 预检确实产出 info 级 check（不是 warning，也不是 error）。
    checks = _web_compat_checks(
        {**KREA2_NF4, "nf4_dir1": True},
        preflight_compat._CudaDeviceProbe(True, SM70),
    )
    assert [item[:2] for item in checks] == [("info", "krea2_attn_fp16")], checks
    mapped = preflight_compat._compat_web_message(compat_info_pairing_item())
    assert mapped != "x", "info 级 code 缺少中文映射，前端会显示英文原始 message"
    assert "建议两个同时开启" in mapped

    panel_source = _read_training_config_ui("TrainingPreflightPanel.tsx")
    assert re.search(r'check\.level === "info"\s*\?\s*"提示"', panel_source, re.DOTALL), (
        "TrainingPreflightPanel 必须单独识别 info 级并渲染「提示」标签"
    )
    assert re.search(r"data-level=\{check\.level\}", panel_source), (
        "info 行必须带 data-level，样式与测试才能区分级别"
    )
    rank_block = re.search(r"const rank = \{(?P<body>[^}]*)\}", panel_source, re.DOTALL)
    assert rank_block, "TrainingPreflightPanel 必须声明 preflight 排序 rank"
    rank_body = rank_block.group("body")
    for level, order in (("error", 0), ("warning", 1), ("info", 2), ("ok", 3)):
        assert re.search(rf"\b{level}:\s*{order}\b", rank_body), (
            f"rank 定义缺少 {level}: {order}（info 必须排在 warning 之后、ok 之前）"
        )

    dialog_test = _read_training_config_ui("TrainingLaunchDialog.test.tsx")
    assert '{ level: "info"' in dialog_test, (
        "前端缺少 info 级 check 的渲染回归用例"
    )
    assert '["error", "warning", "info", "ok", "ok"]' in dialog_test, (
        "前端用例必须断言 info 不被合并/丢弃且排序正确"
    )


def compat_info_pairing_item():
    """构造一条 info 级 ``nf4_fp16_inner_pairing`` 用于消息映射断言。"""
    from library.training.compat_matrix import TrainingCompatIssue

    return TrainingCompatIssue("nf4_fp16_inner_pairing", "krea2_attn_fp16", "x")


# ---------------------------------------------------------------------------
# ④ autocast 防截胡审计
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _recording_autocast(record: list[tuple[str, bool]], device_type="cuda", enabled=True, **_kwargs):
    record.append((device_type, bool(enabled)))
    yield


def test_attention_fp16_inner_sdpa_runs_outside_bf16_autocast():
    """attention 内层必须显式关掉外层 autocast，前反向都保持 fp16 内层。

    用 CPU 张量 + 记录型 autocast 复现训练时的 ``autocast(bf16)`` 场景：
    SDPA 被调用时，最近的 autocast 上下文必须是 ``("cuda", False)``，
    入参与出参都必须是 fp16；对外则回到 bf16 契约。
    """
    from torch.nn import functional

    from library.models.krea2_raw import attention_backend

    autocast_calls: list[tuple[str, bool]] = []
    sdpa_calls: list[dict] = []
    real_sdpa = functional.scaled_dot_product_attention

    def spy_sdpa(query, key, value, **kwargs):
        out = real_sdpa(query, key, value, **kwargs)
        sdpa_calls.append(
            {
                "autocast": autocast_calls[-1] if autocast_calls else None,
                "q": query.dtype,
                "k": key.dtype,
                "v": value.dtype,
                "out": out.dtype,
                "kv_heads": key.shape[1],
                "q_heads": query.shape[1],
            }
        )
        return out

    with mock.patch.object(torch, "autocast", lambda **kw: _recording_autocast(autocast_calls, **kw)), \
            mock.patch.object(attention_backend, "sdpa_kernel", lambda *a, **k: contextlib.nullcontext()), \
            mock.patch.object(functional, "scaled_dot_product_attention", spy_sdpa):
        q = torch.randn(1, 4, 8, 16, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 2, 8, 16, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, 2, 8, 16, dtype=torch.bfloat16, requires_grad=True)
        out = attention_backend._fp16_attention(q, k, v, mask=None, scale=None, gqa=True)
        assert out.dtype == torch.bfloat16, "对外 compute_dtype 契约必须保持 bf16"

        # GQA 展开：SDPA 里 kv head 数必须已被展开成 q head 数。
        assert sdpa_calls, "SDPA 未被调用，fp16 attention 路径没生效"
        call = sdpa_calls[0]
        assert call["q_heads"] == call["kv_heads"] == 4
        assert call["autocast"] == ("cuda", False), (
            f"SDPA 在 autocast={call['autocast']!r} 下执行；"
            "bf16 autocast 会把 fp16 内核顶回 MATH（O(N²) 显存）"
        )
        assert (call["q"], call["k"], call["v"], call["out"]) == (
            torch.float16,
            torch.float16,
            torch.float16,
            torch.float16,
        )

        # 反向：深层 dO（1e-8 量级）必须仍能回到 bf16 且非零。
        out.backward(torch.full_like(out, 1e-8))
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            grad = tensor.grad
            assert grad is not None, f"{name} 没有梯度"
            assert grad.dtype == torch.bfloat16, f"{name}.grad dtype={grad.dtype}"
            assert torch.isfinite(grad).all(), f"{name}.grad 出现 inf/nan"
            assert (grad != 0).any(), (
                f"{name}.grad 全零：反向的 frexp/ldexp 平移没有生效，"
                "1e-8 量级 dO 被 fp16 subnormal 吃掉"
            )
        assert k.grad.shape == k.shape, "GQA 展开后 kv 梯度必须 sum 回原 head 数"


@pytest.mark.skipif(
    not DIR1_DEVICE_SUPPORTED,
    reason="dir1 只对 sm<80（无原生 bf16 tensor core）有意义",
)
def test_dir1_inner_gemm_runs_outside_bf16_autocast_and_backward_survives():
    """训练在 autocast(bf16) 里跑：dir1 内层 GEMM 必须仍是 fp16，反向不归零。"""
    from bitsandbytes.functional import quantize_4bit

    from library.quantization.nf4 import _Dir1MatMul4Bit

    torch.manual_seed(0)
    dim = 128
    weight = torch.randn(dim, dim, device="cuda", dtype=torch.float32) * 0.02
    packed, quant_state = quantize_4bit(
        weight, quant_type="nf4", blocksize=64, compress_statistics=True
    )
    activation = (
        torch.randn(8, dim, device="cuda", dtype=torch.bfloat16) * 1e-3
    ).requires_grad_(True)

    seen: list[tuple[torch.dtype, torch.dtype, torch.dtype]] = []
    real_linear = torch.nn.functional.linear

    def spy_linear(a, b, bias=None):
        out = real_linear(a, b, bias)
        seen.append((a.dtype, b.dtype, out.dtype))
        return out

    with mock.patch.object(torch.nn.functional, "linear", spy_linear), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        out = _Dir1MatMul4Bit.apply(activation, packed.t(), None, None, quant_state)

    assert out.dtype == torch.bfloat16, "对外 compute_dtype 契约必须保持 bf16"
    assert seen, "F.linear 未被调用，dir1 路径没生效"
    act_dtype, weight_dtype, out_dtype = seen[0]
    assert act_dtype == torch.float16, f"激活 dtype {act_dtype}，应为 fp16"
    assert weight_dtype == torch.float16, f"权重 dtype {weight_dtype}，应为 fp16"
    assert out_dtype == torch.float16, (
        f"GEMM 输出被 autocast(bf16) 转成了 {out_dtype}；"
        "dir1 的 F.linear 必须包在 torch.autocast(device_type='cuda', enabled=False) 里"
    )

    out.backward(torch.full_like(out, 1e-8))
    grad = activation.grad
    assert grad is not None and grad.dtype == torch.bfloat16
    assert torch.isfinite(grad).all(), "深层 dO 反向出现 inf/nan"
    assert (grad != 0).any(), "深层 dO 反向全零：frexp/ldexp 平移没有生效"


def test_existing_autocast_regressions_stay_registered():
    """守卫：两条 GPU 回归用例仍在，删除它们等于放开 autocast 防截胡覆盖。"""
    from tests.test_krea2_attention_backend import (
        test_fp16_attention_sdpa_survives_bf16_autocast,
    )
    from tests.test_krea2_nf4_dir1 import TestDir1MatMul4Bit

    assert callable(test_fp16_attention_sdpa_survives_bf16_autocast)
    assert callable(
        getattr(TestDir1MatMul4Bit, "test_runs_fp16_under_bf16_autocast", None)
    )
    # 生产侧包装也必须在位：autocast 的强转发生在 aten 的 C++ 分发层。
    from library.models.krea2_raw import attention_backend
    from library.quantization import nf4

    dir1_source = inspect.getsource(nf4._Dir1MatMul4Bit.forward)
    attention_source = inspect.getsource(attention_backend._fp16_attention)
    assert "enabled=False" in dir1_source, "dir1 内层 GEMM 丢失 autocast 关闭包裹"
    assert "enabled=False" in attention_source, "attention 内层 SDPA 丢失 autocast 关闭包裹"


def test_attention_help_entry_documents_flash_precondition():
    """F1 文案：help/resources.js 必须写明 flash 下开关不生效、要 torch+bf16。"""
    help_source = _read_frontend("help/resources.js")
    match = re.search(r"\bkrea2_attn_fp16:\s*\{(?P<body>.*?)\n  \},", help_source, re.DOTALL)
    assert match, "help/resources.js 缺少 krea2_attn_fp16 帮助条目"
    body = match.group("body")
    assert "attn_mode=flash" in body, "帮助条目必须点名 attn_mode=flash 下不生效"
    assert "不生效" in body
    assert "BF16" in body or "bf16" in body, "帮助条目必须写明 bf16 计算前置条件"
