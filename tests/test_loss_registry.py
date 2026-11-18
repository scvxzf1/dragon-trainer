"""Smoke tests for the M1 loss registry / composer.

Asserts the set of active losses built from each method config matches the
intended pre-refactor behavior. No forward pass — these run with a fake
network object because the composer's activation logic only inspects
attribute presence.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch

from library.training import (
    LOSS_REGISTRY,
    LossComposer,
    build_loss_composer,
)
from library.training.losses import LossContext, _multiscale_loss


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        method="lora",
        loss_type="l2",
        masked_loss=False,
        multiscale_loss_weight=0.0,
        functional_loss_weight=0.0,
        velocity_direction_loss_weight=0.0,
        prior_preservation_weight=0.0,
        inverted_mask_prior_weight=0.0,
        model_family="anima",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _net(**attrs) -> SimpleNamespace:
    # Composer reads `_ortho_reg_weight` and `_balance_loss_weight`.
    defaults = {"_ortho_reg_weight": 0.0, "_balance_loss_weight": 0.0}
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


def test_registry_contains_expected_keys():
    assert {
        "flow_match",
        "ortho_reg",
        "hydra_balance",
        "functional",
        "multiscale",
        "velocity_direction",
    }.issubset(set(LOSS_REGISTRY.keys()))


def test_lora_composer_is_flow_match_only():
    args = _make_args(method="lora")
    composer = build_loss_composer(args, _net())
    assert isinstance(composer, LossComposer)
    assert composer.active_losses == ["flow_match"]


def test_hydra_balance_activates_when_network_requests_it():
    args = _make_args(method="lora")
    composer = build_loss_composer(args, _net(_balance_loss_weight=0.01))
    assert "hydra_balance" in composer.active_losses
    assert "flow_match" in composer.active_losses


def test_functional_activates_when_weight_set():
    args = _make_args(method="lora", functional_loss_weight=1.0)
    composer = build_loss_composer(args, _net())
    assert "functional" in composer.active_losses


def test_multiscale_activates_when_weight_set():
    args = _make_args(method="lora", multiscale_loss_weight=0.5)
    composer = build_loss_composer(args, _net())
    assert "multiscale" in composer.active_losses


@pytest.mark.parametrize(
    ("model_family", "latent_hw", "expected_nonzero"),
    [
        ("anima", (64, 64), False),
        ("qwen_image_2_1", (64, 64), True),
        ("moss_image", (64, 64), True),
    ],
)
def test_multiscale_uses_family_vae_compression(
    model_family, latent_hw, expected_nonzero
):
    args = _make_args(model_family=model_family, multiscale_loss_weight=0.5)
    prediction = torch.ones(1, 2, *latent_hw)
    context = LossContext(
        args=args,
        batch={},
        model_pred=prediction,
        target=torch.zeros_like(prediction),
        timesteps=torch.zeros(1),
        weighting=None,
        huber_c=None,
        loss_weights=torch.ones(1),
        network=SimpleNamespace(),
    )
    result = _multiscale_loss(context)
    assert bool(torch.count_nonzero(result)) is expected_nonzero


def test_velocity_direction_activates_when_weight_set():
    args = _make_args(method="lora", velocity_direction_loss_weight=0.05)
    composer = build_loss_composer(args, _net())
    assert composer.active_losses == ["flow_match", "velocity_direction"]


def test_prior_preservation_activates_when_weight_set():
    args = _make_args(method="lora", prior_preservation_weight=0.2)
    composer = build_loss_composer(args, _net())
    assert composer.active_losses == ["flow_match", "prior_preservation"]


def test_inverted_mask_prior_activates_when_weight_set():
    args = _make_args(method="lora", inverted_mask_prior_weight=0.2)
    composer = build_loss_composer(args, _net())
    assert composer.active_losses == ["flow_match", "inverted_mask_prior"]
