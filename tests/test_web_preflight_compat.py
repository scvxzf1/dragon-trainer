"""nf4_dir1 / krea2_attn_fp16 preflight-compat regressions.

The two experimental switches only make sense on GPUs without native BF16
tensor cores (sm70/sm75). These tests pin every rule branch to the runtime
gates ``set_dir1_enabled`` / ``set_krea2_attn_fp16`` and to
``run_krea2_attention`` (flash never reads the attention switch):

1. blocker: either switch on with a sm80+ device; ``nf4_dir1`` without CUDA
   stays a blocker because its runtime gate raises;
2. warning: ``nf4_dir1`` without ``base_compute=nf4``;
3. warning: ``krea2_attn_fp16`` outside ``model_family=krea2_raw``;
4. info: sm70/sm75 + Krea-2 + NF4 with only one of the two switches on;
5. warning: ``krea2_attn_fp16`` under ``attn_mode=flash`` or non-bf16
   compute, where the runtime never consumes the switch (dead switch).

Web-level expectations (hardware probe / level mapping) are covered at the
bottom: ``krea2_attn_fp16`` without CUDA or with unreadable capability only
warns, and a GPU whitelist is resolved by physical nvidia-smi index.
"""

from __future__ import annotations

import pytest

from library.training.compat_matrix import check_training_compat
from web.services.config import preflight_compat
from web.services.training import gpu as gpu_module

BASE_CONFIG = {
    "model_family": "krea2_raw",
    "base_compute": "nf4",
    "attn_mode": "flash",
    "compile_dynamic_seq": False,
    "compile_inductor_mode": "default",
    "selective_checkpoint": "off",
    "v100_flash_stability": "off",
}
SM70 = (7, 0)
SM86 = (8, 6)
F1_CODE = "krea2_attn_fp16_requires_torch_attention"


@pytest.fixture(autouse=True)
def _preflight_facade_ready():
    """``preflight_compat._bool_value`` 等 helper 由 facade 注入，先同步一次。"""
    from web.services.config.preflight_runtime import _sync_from_facade

    _sync_from_facade()


def _compat(overrides: dict, *, cuda_available=True, capability=SM70):
    config = dict(BASE_CONFIG)
    config.update(overrides)
    return check_training_compat(
        config,
        cuda_available=cuda_available,
        device_capability=capability,
    )


def _codes(items) -> set[str]:
    return {item.code for item in items}


def _web_checks(cfg: dict, monkeypatch, *, probe=None, whitelist=None, capabilities=None):
    """Run ``_check_checkpointing_config`` with an injected hardware context."""
    if probe is not None:
        monkeypatch.setattr(
            preflight_compat, "_probe_cuda_device_for", lambda _cfg: probe
        )
    if capabilities is not None:
        monkeypatch.setattr(
            gpu_module,
            "query_gpu_compute_capabilities",
            lambda indices=None, **_kwargs: dict(capabilities),
        )
    checks: list[tuple[str, str, str]] = []

    def add(level, key, message, path=None, **flags) -> None:
        checks.append((level, key, message))

    preflight_compat._check_checkpointing_config(
        cfg, add, gpu_whitelist=whitelist
    )
    return checks


# ---------------------------------------------------------------------------
# 规则层：硬件 / 模型族 / base_compute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["nf4_dir1", "krea2_attn_fp16"])
def test_sm80_plus_blocks_either_switch(flag: str) -> None:
    result = _compat({flag: True}, capability=SM86)

    assert not result.ok
    assert f"{flag}_requires_pre_ampere" in _codes(result.errors)
    assert flag not in _codes(result.warnings)


def test_missing_cuda_blocks_nf4_dir1_but_only_warns_attention() -> None:
    nf4 = _compat({"nf4_dir1": True}, cuda_available=False, capability=None)
    assert not nf4.ok
    assert _codes(nf4.errors) == {"nf4_dir1_requires_cuda"}

    attention = _compat(
        {"krea2_attn_fp16": True, "attn_mode": "torch"},
        cuda_available=False,
        capability=None,
    )
    assert attention.ok
    assert attention.errors == ()
    assert "krea2_attn_fp16_requires_cuda" in _codes(attention.warnings)


def test_unreadable_capability_warns_instead_of_blocking() -> None:
    nf4 = _compat({"nf4_dir1": True}, capability=None)
    assert nf4.ok
    assert "nf4_dir1_architecture_unknown" in _codes(nf4.warnings)

    attention = _compat({"krea2_attn_fp16": True, "attn_mode": "torch"}, capability=None)
    assert attention.ok
    assert "krea2_attn_fp16_architecture_unknown" in _codes(attention.warnings)


def test_nf4_dir1_without_nf4_base_only_warns() -> None:
    result = _compat({"base_compute": "bf16", "nf4_dir1": True})

    assert result.ok
    assert "nf4_dir1_ignored" in _codes(result.warnings)


def test_krea2_attn_fp16_outside_krea2_only_warns() -> None:
    result = _compat({"model_family": "anima", "krea2_attn_fp16": True})

    assert result.ok
    assert "krea2_attn_fp16_ignored" in _codes(result.warnings)


# ---------------------------------------------------------------------------
# 第 5 类：flash / 非 bf16 compute 下 attention 开关不生效
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attn_mode", ["flash", "FLASH"])
def test_krea2_attn_fp16_warns_when_flash_never_reads_it(attn_mode: str) -> None:
    result = _compat(
        {"nf4_dir1": True, "krea2_attn_fp16": True, "attn_mode": attn_mode}
    )

    assert result.ok
    assert F1_CODE in _codes(result.warnings)
    assert "nf4_fp16_inner_pairing" not in _codes(result.infos)


def test_flash_suppresses_pairing_hint_even_when_only_nf4_dir1_is_on() -> None:
    result = _compat({"nf4_dir1": True})

    assert result.ok
    assert result.infos == ()
    assert F1_CODE not in _codes(result.warnings)


@pytest.mark.parametrize("attn_mode", ["torch", "sdpa", "cudnn", "cudnn_sdpa"])
def test_torch_attention_aliases_restore_pairing_hint(attn_mode: str) -> None:
    result = _compat({"nf4_dir1": True, "attn_mode": attn_mode})

    assert result.ok
    assert [item.key for item in result.infos] == ["krea2_attn_fp16"]
    assert result.warnings == ()


def test_unset_attn_mode_is_not_treated_as_flash() -> None:
    # 真实 WebUI 配置一定带 base.toml 的 flash；显式缺省不臆造死开关结论。
    result = _compat({"nf4_dir1": True, "attn_mode": ""})

    assert [item.key for item in result.infos] == ["krea2_attn_fp16"]


@pytest.mark.parametrize("precision", ["fp16", "no", "fp32"])
def test_non_bf16_compute_makes_attention_switch_inert(precision: str) -> None:
    result = _compat(
        {
            "krea2_attn_fp16": True,
            "attn_mode": "torch",
            "mixed_precision": precision,
        }
    )

    assert result.ok
    assert F1_CODE in _codes(result.warnings)
    assert result.infos == ()


def test_bf16_compute_keeps_attention_switch_effective() -> None:
    result = _compat(
        {"krea2_attn_fp16": True, "attn_mode": "torch", "mixed_precision": "bf16"}
    )

    assert result.ok
    assert F1_CODE not in _codes(result.warnings)


# ---------------------------------------------------------------------------
# 联动提示（分支 4）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capability", [SM70, (7, 5)])
@pytest.mark.parametrize(
    ("enabled", "missing"),
    [("nf4_dir1", "krea2_attn_fp16"), ("krea2_attn_fp16", "nf4_dir1")],
)
def test_sm70_sm75_single_switch_suggests_both(
    capability: tuple[int, int], enabled: str, missing: str
) -> None:
    result = _compat({"attn_mode": "torch", enabled: True}, capability=capability)

    assert result.ok
    assert [item.key for item in result.infos] == [missing]
    assert "nf4_fp16_inner_pairing" in _codes(result.infos)


def test_no_pairing_hint_when_both_or_neither_switch_is_on() -> None:
    both = _compat({"attn_mode": "torch", "nf4_dir1": True, "krea2_attn_fp16": True})
    neither = _compat({"attn_mode": "torch"})

    # 双开：内层 fp16 已完整启用，无需任何建议。
    assert not both.infos
    # 双关：不再是「再开一个」的 pairing 建议，而是提示存在可开的内层 fp16 路径
    # —— sm<80 上默认 bf16 的每个 GEMM 都落到软件模拟（V100 44.8 vs 11.06 s/step）。
    assert {item.code for item in neither.infos} == {"nf4_fp16_inner_available"}


def test_inner_available_hint_only_for_krea2_nf4_on_pre_ampere() -> None:
    """双关提示只在 Krea-2 + NF4 + sm<80 + bf16 下出现，且绝不建议改用 fp16。"""
    def codes(**overrides):
        return {item.code for item in _compat(overrides).infos}

    base = {"attn_mode": "torch"}
    assert codes(**base) == {"nf4_fp16_inner_available"}

    # 任一开关已开 → 不再是「双关」场景
    assert "nf4_fp16_inner_available" not in codes(nf4_dir1=True, **base)
    assert "nf4_fp16_inner_available" not in codes(krea2_attn_fp16=True, **base)

    # 非 bf16 compute：两个开关都要求 bf16，建议开它们是误导
    for precision in ("fp16", "fp32", "no"):
        assert "nf4_fp16_inner_available" not in codes(
            mixed_precision=precision, **base
        )

    # sm80+ 有原生 bf16 tensor core，不存在这个问题
    assert not {
        item.code
        for item in check_training_compat(
            dict(BASE_CONFIG, **base), cuda_available=True, device_capability=SM86
        ).infos
    }

    # 非 Krea-2 / 非 NF4 不适用（两个开关都不会被消费）
    assert "nf4_fp16_inner_available" not in codes(
        model_family="anima", **base
    )
    assert "nf4_fp16_inner_available" not in codes(
        base_compute="bf16", **base
    )

    # 文案必须点明 bf16 GEMM 走软件模拟，并明确禁止改用 fp16 compute
    message = next(
        item.message
        for item in _compat(base).infos
        if item.code == "nf4_fp16_inner_available"
    )
    assert "software emulation" in message
    assert "do not switch mixed_precision to fp16" in message

    # Web 层必须有中文文案，不能回退英文原文
    from web.services.config.preflight_compat import _compat_web_message

    web_message = _compat_web_message(
        next(
            item
            for item in _compat(base).infos
            if item.code == "nf4_fp16_inner_available"
        )
    )
    assert "软件模拟" in web_message
    assert "不要把 mixed_precision 改成 fp16" in web_message


# ---------------------------------------------------------------------------
# Web 层：级别映射 / 探针接线 / whitelist 物理 index
# ---------------------------------------------------------------------------


def test_web_preflight_maps_blockers_to_errors(monkeypatch) -> None:
    checks = _web_checks(
        {**BASE_CONFIG, "nf4_dir1": True},
        monkeypatch,
        probe=preflight_compat._CudaDeviceProbe(True, SM86),
    )

    blocking = [item for item in checks if item[1] == "nf4_dir1"]
    assert [item[0] for item in blocking] == ["error"]
    assert "sm80+" in blocking[0][2]


def test_web_preflight_maps_pairing_hint_to_info(monkeypatch) -> None:
    checks = _web_checks(
        {**BASE_CONFIG, "attn_mode": "torch", "nf4_dir1": True},
        monkeypatch,
        probe=preflight_compat._CudaDeviceProbe(True, SM70),
    )

    assert [item[:2] for item in checks] == [("info", "krea2_attn_fp16")]
    assert "建议两个同时开启" in checks[0][2]


def test_web_preflight_warns_instead_of_blocking_attention_without_cuda(monkeypatch) -> None:
    checks = _web_checks(
        {
            **BASE_CONFIG,
            "attn_mode": "torch",
            "nf4_dir1": True,
            "krea2_attn_fp16": True,
        },
        monkeypatch,
        probe=preflight_compat._CudaDeviceProbe(False, None),
    )

    levels = {key: level for level, key, _message in checks}
    assert levels == {"nf4_dir1": "error", "krea2_attn_fp16": "warning"}
    assert all("CUDA" in message for _level, _key, message in checks)


def test_web_preflight_warns_when_capability_is_unknown(monkeypatch) -> None:
    checks = _web_checks(
        {
            **BASE_CONFIG,
            "attn_mode": "torch",
            "nf4_dir1": True,
            "krea2_attn_fp16": True,
        },
        monkeypatch,
        probe=preflight_compat._CudaDeviceProbe(True, None),
    )

    levels = {key: level for level, key, _message in checks}
    assert levels == {"nf4_dir1": "warning", "krea2_attn_fp16": "warning"}
    assert all("compute capability" in message for _level, _key, message in checks)


def test_web_preflight_flags_attention_switch_dead_under_flash(monkeypatch) -> None:
    checks = _web_checks(
        {**BASE_CONFIG, "krea2_attn_fp16": True},
        monkeypatch,
        probe=preflight_compat._CudaDeviceProbe(True, SM70),
    )

    assert [item[:2] for item in checks] == [("warning", "krea2_attn_fp16")]
    assert "attn_mode=torch" in checks[0][2]


def test_web_preflight_resolves_whitelist_by_physical_gpu_index(monkeypatch) -> None:
    """Mock 两张不同 capability：只选 V100(sm70) 时必须放行且不报 sm86。"""
    capabilities = {0: SM86, 1: SM70}

    # 只选物理 index 1（V100）：放行；info 用被选设备 sm70。
    checks = _web_checks(
        {**BASE_CONFIG, "attn_mode": "torch", "nf4_dir1": True},
        monkeypatch,
        whitelist=[1],
        capabilities=capabilities,
    )
    assert [item[:2] for item in checks] == [("info", "krea2_attn_fp16")]
    assert "sm70" in checks[0][2]
    assert "sm86" not in checks[0][2]

    # 同时选中 sm86 + sm70：任一 sm80+ 就阻断，文案点名被选设备。
    checks = _web_checks(
        {**BASE_CONFIG, "attn_mode": "torch", "nf4_dir1": True},
        monkeypatch,
        whitelist=[0, 1],
        capabilities=capabilities,
    )
    assert [item[:2] for item in checks] == [("error", "nf4_dir1")]
    assert "sm_86" in checks[0][2] and "sm_70" in checks[0][2]

    # 被选设备解析不到 capability：降级为 warning，不阻断。
    checks = _web_checks(
        {**BASE_CONFIG, "attn_mode": "torch", "nf4_dir1": True},
        monkeypatch,
        whitelist=[1],
        capabilities={},
    )
    assert [item[:2] for item in checks] == [("warning", "nf4_dir1")]
    assert "compute capability" in checks[0][2]


def test_web_process_without_cuda_does_not_block_when_whitelist_resolves(monkeypatch) -> None:
    """F3：Web 进程看不到 CUDA、但任务白名单能解析出 sm70 时不得误报 blocker。"""
    monkeypatch.setattr(
        preflight_compat,
        "_probe_cuda_device_for",
        lambda _cfg: (_ for _ in ()).throw(
            AssertionError("whitelist 非空时不应回退 device-0 探针")
        ),
    )

    checks = _web_checks(
        {
            **BASE_CONFIG,
            "attn_mode": "torch",
            "nf4_dir1": True,
            "krea2_attn_fp16": True,
        },
        monkeypatch,
        whitelist=[0, 1],
        capabilities={0: SM70, 1: (7, 5)},
    )

    # 两张被选设备都是 sm<80 → 双开完全放行，不因 Web 进程无 CUDA 而阻断。
    assert checks == []


def test_web_preflight_partial_whitelist_resolution_stays_a_warning(monkeypatch) -> None:
    """F2：白名单里有设备查不到 capability 时不假装放行（已知的 sm<80 也不够）。"""
    checks = _web_checks(
        {**BASE_CONFIG, "attn_mode": "torch", "nf4_dir1": True},
        monkeypatch,
        whitelist=[0, 1],
        capabilities={0: SM70},
    )

    assert [item[:2] for item in checks] == [("warning", "nf4_dir1")]
    assert "compute capability" in checks[0][2]
