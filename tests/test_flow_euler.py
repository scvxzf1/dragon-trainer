from __future__ import annotations

import torch
import pytest

from library.inference.flow_euler import combine_cfg, euler_step, run_flow_euler_loop


def test_euler_step_preserves_flow_direction_and_dtype():
    latent = torch.tensor([1.0, 2.0], dtype=torch.float64)
    velocity = torch.tensor([3.0, -4.0], dtype=torch.float64)
    result = euler_step(latent, velocity, torch.tensor(0.8), torch.tensor(0.3))
    torch.testing.assert_close(result, torch.tensor([-0.5, 4.0], dtype=torch.float64))
    assert result.dtype is torch.float64


@pytest.mark.parametrize("scale, expected", [(0.0, 3.0), (1.0, 4.0), (2.0, 5.0)])
def test_combine_cfg(scale, expected):
    result = combine_cfg(torch.tensor([3.0]), torch.tensor([2.0]), scale)
    torch.testing.assert_close(result, torch.tensor([expected]))


def test_loop_does_not_call_negative_predictor_without_cfg():
    seen = []

    def positive(latent, sigma):
        seen.append(float(sigma.item()))
        return torch.ones_like(latent)

    def negative(_latent, _sigma):
        raise AssertionError("negative predictor must stay unused")

    result = run_flow_euler_loop(
        torch.zeros(1), [1.0, 0.5, 0.0], positive,
        guidance_scale=0.0, predict_negative=negative,
    )
    torch.testing.assert_close(result, torch.tensor([-1.0]))
    assert seen == [1.0, 0.5]


def test_loop_applies_cfg_before_each_euler_step():
    def positive(latent, _sigma):
        return torch.ones_like(latent) * 3

    def negative(latent, _sigma):
        return torch.ones_like(latent) * 2

    result = run_flow_euler_loop(
        torch.zeros(1), [1.0, 0.5, 0.0], positive,
        guidance_scale=2.0, predict_negative=negative,
    )
    torch.testing.assert_close(result, torch.tensor([-5.0]))


def test_loop_calls_branch_callback_before_each_prediction():
    events = []

    def branch(index, branch_index, latent, sigma):
        events.append((index, branch_index, float(latent.item()), float(sigma.item())))

    run_flow_euler_loop(
        torch.zeros(1), [1.0, 0.5, 0.0], lambda latent, sigma: latent + sigma,
        guidance_scale=1.0,
        predict_negative=lambda latent, sigma: latent - sigma,
        branch_start_callback=branch,
    )

    assert [event[:2] for event in events] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [event[3] for event in events] == [1.0, 1.0, 0.5, 0.5]


def test_loop_rejects_missing_negative_predictor():
    with pytest.raises(ValueError, match="predict_negative"):
        run_flow_euler_loop(torch.zeros(1), [1.0, 0.0], lambda x, _: x, guidance_scale=1.0)
