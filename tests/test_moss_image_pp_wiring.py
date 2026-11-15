"""MOSS-Image PP v0 wiring acceptance (task t2).

Frozen entrypoint.  What this file proves:

* the ``moss_image`` family publishes the *shared* stage-local PP v0 backend
  constant (no separate availability flag exists), the balanced plan over 42
  layers comes from the existing registry planner, and the compat matrix
  releases a legal MOSS PP config to the experimental-runtime warning;
* families without a stage-local runtime (krea2_raw / z_image / qwen_image_2_1)
  keep their named rejections; unknown PP knobs for MOSS (block swap, compile)
  are refused by the shared family contract;
* the production trainer entry admission admits only the wired MOSS family to
  the dedicated PP trainer path (dispatch in train_session), every other family
  and every DDP-interior code path keep the ``pp_trainer_path_unavailable``
  refusal (fail-closed), and
  ``moss_compat`` no longer blanket-rejects the ``pipeline_parallel`` key --
  the gate, not the family shim, decides whether the config is servable.
"""

from __future__ import annotations

import dataclasses

import pytest

from library.models.family_registry import (
    MODEL_FAMILY_REGISTRY,
    PIPELINE_PARALLEL_RUNTIME_BACKEND_1F1B,
)
from library.models.pipeline_parallel import make_pipeline_plan
from library.training.compat_matrix import check_training_compat
from library.training.moss_compat import check_moss_training_contract
from library.training.pp.errors import PipelineParallelError
from library.training.pp.gate import (
    pipeline_parallel_runtime_issue,
    require_accelerate_checkpoint,
    require_pipeline_runtime,
    require_trainer_topology,
)

from tests.moss_pp_test_support import HEIGHT, MAX_TEXT_TOKENS, WIDTH


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model_family": "moss_image",
        "pipeline_parallel": True,
        "pipeline_parallel_stages": 2,
        "pipeline_parallel_microbatches": 2,
        "pipeline_parallel_schedule": "1f1b",
        "pipeline_parallel_split": "balanced",
        "blocks_to_swap": 0,
        "torch_compile": False,
        "selective_checkpoint": "off",
        "cpu_offload_checkpointing": False,
        "unsloth_offload_checkpointing": False,
        "network_train_unet_only": True,
        "mixed_precision": "fp16",
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# registry and planner
# ---------------------------------------------------------------------------


def test_moss_family_publishes_the_shared_runtime_backend():
    spec = MODEL_FAMILY_REGISTRY["moss_image"].pipeline_parallel
    assert spec is not None
    assert spec.block_container == "layers"
    assert spec.default_num_blocks == 42
    assert spec.known_num_blocks == frozenset({42})
    assert spec.supported_stages == frozenset({2})
    assert spec.supported_schedules == frozenset({"1f1b"})
    assert spec.supported_splits == frozenset({"balanced"})
    # One shared backend identifier; ``runtime_available`` derives from it.
    assert spec.runtime_backend == PIPELINE_PARALLEL_RUNTIME_BACKEND_1F1B
    assert spec.runtime_available is True


def test_families_without_a_runtime_stay_fail_closed():
    for family in ("krea2_raw", "z_image", "qwen_image_2_1"):
        spec = MODEL_FAMILY_REGISTRY[family].pipeline_parallel
        if spec is None:
            continue
        assert spec.runtime_backend is None
        assert spec.runtime_available is False


def test_moss_plan_is_registry_backed():
    plan = make_pipeline_plan(family="moss_image", stages=2, num_blocks=42)
    assert plan.ranges == ((0, 21), (21, 42))
    assert plan.family == "moss_image"
    assert plan.block_container == "layers"


# ---------------------------------------------------------------------------
# gate and compat matrix
# ---------------------------------------------------------------------------


def test_gate_releases_a_legal_moss_pp_config():
    runtime = require_pipeline_runtime(_config(), world_size=2)
    assert runtime.stages == 2 and runtime.schedule == "1f1b"
    assert pipeline_parallel_runtime_issue(_config(), world_size=2) is None


def test_compat_matrix_reports_the_experimental_runtime_for_moss():
    result = check_training_compat(_config(), world_size=2)
    errors = [item for item in result.errors if item.key == "pipeline_parallel"]
    assert errors == []
    warnings = [item for item in result.warnings if item.key == "pipeline_parallel"]
    assert [item.code for item in warnings] == ["pipeline_parallel_experimental_runtime"]
    assert "MOSS-Image" in warnings[0].message


def test_compat_matrix_still_names_the_microbatch_floor_for_moss():
    result = check_training_compat(_config(pipeline_parallel_microbatches=1), world_size=2)
    issues = [item for item in result.errors if item.key == "pipeline_parallel"]
    assert [item.code for item in issues] == ["pipeline_parallel_microbatches_below_stages"]


@pytest.mark.parametrize("family", ["krea2_raw", "z_image"])
def test_families_without_a_runtime_keep_the_named_refusal(family):
    result = check_training_compat(_config(model_family=family), world_size=2)
    issues = [item for item in result.errors if item.key == "pipeline_parallel"]
    assert [item.code for item in issues] == ["pipeline_parallel_runtime_unavailable"]


def test_family_without_a_planner_is_refused_by_name():
    result = check_training_compat(
        _config(model_family="qwen_image_2_1"), world_size=2
    )
    issues = [item for item in result.errors if item.key == "pipeline_parallel"]
    assert [item.code for item in issues] == ["pipeline_parallel_config"]


def test_block_swap_and_compile_stay_refused_for_moss_pp():
    # The family contract raises a bare ValueError; the gate re-raises it with
    # the stable ``pp_config_invalid`` code.
    with pytest.raises(PipelineParallelError, match="blocks_to_swap"):
        require_pipeline_runtime(_config(blocks_to_swap=2), world_size=2)
    with pytest.raises(PipelineParallelError, match="torch_compile"):
        require_pipeline_runtime(_config(torch_compile=True), world_size=2)


# ---------------------------------------------------------------------------
# main training entries stay fail-closed
# ---------------------------------------------------------------------------


def test_wired_moss_passes_entry_admission_but_ddp_interior_stays_refused():
    """The wired MOSS family passes the entry preflight; the DDP interior does not.

    The entry admission (cli_entry / train_session dispatch) may admit the wired
    moss_image family to the dedicated PP trainer path; every DDP-interior guard
    (bootstrap / loop hotspots) keeps refusing it, so a mis-routed run can never
    silently become data parallel. Unwired families keep the refusal everywhere.
    """

    from library.training.pp.gate import require_trainer_pp_admission

    require_trainer_pp_admission(_config(), where="unit-test")
    require_trainer_pp_admission(_config(), where="cli_entry")
    require_trainer_pp_admission(_config(), where="train_session.pp_dispatch")
    for where in (
        "bootstrap.setup_optimizer_and_dataloader",
        "bootstrap.prepare_with_accelerator",
        "loop.run_training_loop",
        "loop._run_microbatch",
    ):
        with pytest.raises(PipelineParallelError) as excinfo:
            require_trainer_topology(_config(), where=where)
        assert excinfo.value.code == "pp_trainer_path_unavailable"


def test_unwired_families_keep_the_entry_refusal():
    from library.training.pp.gate import require_trainer_pp_admission

    # anima is wired since the Anima trainer-level PP adapter landed
    # (docs/proposal/anima_pp_trainer_wiring.md §3): admission accepts it.
    # Families without any runtime are refused one step earlier by the shared
    # gate. Either way nothing reaches a DDP fallback.
    require_trainer_pp_admission(_config(model_family="anima"), where="unit-test")
    for family in ("krea2_raw", "z_image"):
        with pytest.raises(PipelineParallelError) as excinfo:
            require_trainer_pp_admission(_config(model_family=family), where="unit-test")
        assert excinfo.value.code == "pp_runtime_unavailable"


def test_accelerate_checkpoint_refusal_covers_moss():
    with pytest.raises(PipelineParallelError) as excinfo:
        require_accelerate_checkpoint(_config(), where="unit-test")
    assert excinfo.value.code == "pp_accelerate_checkpoint_unsupported"


def test_moss_compat_releases_the_pipeline_parallel_key():
    class _Out:
        def __init__(self):
            self.errors = []
            self.warnings = []

        def error(self, code, key, message):
            self.errors.append((code, key, message))

        def warning(self, code, key, message):
            self.warnings.append((code, key, message))

    out = _Out()
    check_moss_training_contract(out, _config())
    assert not [item for item in out.errors if item[0] == "moss_image_pipeline_parallel"]
    # A sibling unsupported knob is still rejected: the shim's other contracts
    # did not loosen.
    check_moss_training_contract(out, _config(torch_compile=True))
    assert any(item[0] == "moss_image_torch_compile" for item in out.errors)


# ---------------------------------------------------------------------------
# wire contract stays recordable for preflight surfaces
# ---------------------------------------------------------------------------


def test_wire_capacity_is_derivable_for_preflight():
    from library.models.moss_image.pipeline_parallel import resolve_wire_capacity

    assert resolve_wire_capacity(HEIGHT, WIDTH, MAX_TEXT_TOKENS) == (
        HEIGHT * WIDTH + MAX_TEXT_TOKENS
    )
