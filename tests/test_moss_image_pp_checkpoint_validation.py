"""Pure checkpoint validation regressions; no GPU/model/distributed launch."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from library.training.pp.checkpoint_validation import (
    validate_checkpoint_content,
    validate_optimizer_payload,
)
from library.training.pp.checkpoint_state import content_hash_object
from library.training.pp.errors import CODE_CHECKPOINT_STATE_MISMATCH, PipelineParallelError


def _state():
    scaler = torch.amp.GradScaler("cpu").state_dict()
    return SimpleNamespace(scale=scaler["scale"], growth_tracker=2, scaler=scaler)


def test_scaler_internal_tracker_is_required_before_apply():
    state = _state()
    state.scaler.pop("_growth_tracker")
    before = deepcopy(state.scaler)
    with pytest.raises(PipelineParallelError, match="_growth_tracker") as excinfo:
        validate_checkpoint_content(state)
    assert excinfo.value.code == CODE_CHECKPOINT_STATE_MISMATCH
    assert state.scaler == before


@pytest.mark.parametrize("tracker", [-1, 0.5, float("nan"), True, torch.tensor([1, 2])])
def test_scaler_internal_tracker_must_be_a_valid_counter(tracker):
    state = _state()
    state.scaler["_growth_tracker"] = tracker
    with pytest.raises(PipelineParallelError) as excinfo:
        validate_checkpoint_content(state)
    assert excinfo.value.code == CODE_CHECKPOINT_STATE_MISMATCH


@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_positive_but_inconsistent_scales_are_rejected(factor):
    state = _state()
    state.scale *= factor
    before = deepcopy(state.scaler)
    with pytest.raises(PipelineParallelError, match="does not match") as excinfo:
        validate_checkpoint_content(state)
    assert excinfo.value.code == CODE_CHECKPOINT_STATE_MISMATCH
    assert state.scaler == before


def test_valid_scaler_and_independent_trackers_still_restore():
    state = _state()
    assert state.growth_tracker != state.scaler["_growth_tracker"]
    before = deepcopy(state.scaler)
    validate_checkpoint_content(state)
    restored = torch.amp.GradScaler("cpu")
    restored.load_state_dict(state.scaler)
    assert restored.state_dict() == before == state.scaler
    assert restored.get_scale() == state.scale


def _optimizer_state(*, remap=True, amsgrad=False):
    parameters = [torch.nn.Parameter(torch.zeros(size)) for size in (2, 3)]
    optimizer = torch.optim.AdamW(
        [{"params": [parameter]} for parameter in parameters],
        lr=1e-3, amsgrad=amsgrad,
    )
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    payload = deepcopy(optimizer.state_dict())
    if remap:
        ids = [91, 17]  # saved IDs are not live parameter offsets
        payload["state"] = {
            new_id: payload["state"][group["params"][0]]
            for new_id, group in zip(ids, payload["param_groups"])
        }
        for new_id, group in zip(ids, payload["param_groups"]):
            group["params"] = [new_id]
    return optimizer, payload


def test_noncontiguous_saved_ids_follow_group_order_and_remain_loadable():
    optimizer, payload = _optimizer_state(amsgrad=True)
    before = content_hash_object(payload), content_hash_object(optimizer.state_dict())
    validate_optimizer_payload(payload, optimizer)
    assert (content_hash_object(payload), content_hash_object(optimizer.state_dict())) == before
    optimizer.load_state_dict(payload)
    optimizer.step()
    assert all(
        torch.isfinite(parameter).all()
        for group in optimizer.param_groups for parameter in group["params"]
    )


@pytest.mark.parametrize("moment", ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"])
def test_remapped_parameter_moment_shape_is_checked_before_mutation(moment):
    optimizer, payload = _optimizer_state(amsgrad=True)
    payload["state"][91][moment] = torch.ones(3)  # ID 91 belongs to shape (2,)
    before = content_hash_object(optimizer.state_dict())
    with pytest.raises(PipelineParallelError, match="shape") as excinfo:
        validate_optimizer_payload(payload, optimizer)
    assert excinfo.value.code == CODE_CHECKPOINT_STATE_MISMATCH
    assert content_hash_object(optimizer.state_dict()) == before


@pytest.mark.parametrize("step", [
    float("nan"), float("inf"), -1, True, None,
    torch.tensor(float("nan")), torch.tensor(float("-inf")),
    torch.tensor(-1.0), torch.tensor(True),
])
def test_optimizer_step_is_finite_nonnegative_real_scalar(step):
    optimizer, payload = _optimizer_state(remap=False)
    payload["state"][0]["step"] = step
    before = content_hash_object(optimizer.state_dict())
    with pytest.raises(PipelineParallelError, match="step") as excinfo:
        validate_optimizer_payload(payload, optimizer)
    assert excinfo.value.code == CODE_CHECKPOINT_STATE_MISMATCH
    assert content_hash_object(optimizer.state_dict()) == before


@pytest.mark.parametrize("step", [0, 1.0, torch.tensor(0.0), torch.tensor(3.0)])
def test_legal_optimizer_step_scalar_and_tensor_are_accepted(step):
    optimizer, payload = _optimizer_state()
    payload["state"][91]["step"] = step
    validate_optimizer_payload(payload, optimizer)


@pytest.mark.parametrize("key", ["step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"])
def test_initialized_adam_state_requires_its_restore_fields(key):
    optimizer, payload = _optimizer_state(amsgrad=True)
    payload["state"][91].pop(key)
    with pytest.raises(PipelineParallelError, match="missing"):
        validate_optimizer_payload(payload, optimizer)


def test_empty_state_before_first_update_is_legal():
    optimizer, payload = _optimizer_state()
    payload["state"] = {}
    validate_optimizer_payload(payload, optimizer)


def test_other_optimizers_factorized_fields_are_not_blanket_rejected():
    parameter = torch.nn.Parameter(torch.zeros(2, 3))
    optimizer = torch.optim.SGD([parameter], lr=1e-3, momentum=0.9)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    payload = deepcopy(optimizer.state_dict())
    payload["state"][0]["exp_avg_sq_row"] = torch.ones(2)
    payload["state"][0]["exp_avg_sq_col"] = torch.ones(3)
    validate_optimizer_payload(payload, optimizer)


@pytest.mark.parametrize("bad_ids", ["duplicate", "unreferenced"])
def test_ambiguous_parameter_state_ids_are_refused(bad_ids):
    optimizer, payload = _optimizer_state()
    if bad_ids == "duplicate":
        payload["param_groups"][1]["params"] = [91]
    else:
        payload["state"][999] = deepcopy(payload["state"][91])
    with pytest.raises(PipelineParallelError) as excinfo:
        validate_optimizer_payload(payload, optimizer)
    assert excinfo.value.code == CODE_CHECKPOINT_STATE_MISMATCH
