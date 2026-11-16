"""Krea-2 PP probe cut: the two halves must reproduce the monolith exactly."""

from __future__ import annotations

import torch

from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT
from library.models.krea2_raw.pp_stages import (
    krea2_forward_first,
    krea2_forward_last,
    krea2_token_cap,
)

LAYERS = 4
TXT_TOKENS = 5
IMG_TOKENS = 4  # 2x2 patch grid


def _tiny_model(seed: int = 7) -> SingleStreamDiT:
    torch.manual_seed(seed)
    config = SingleMMDiTConfig(
        features=128,
        tdim=32,
        txtdim=64,
        heads=4,
        multiplier=2,
        layers=LAYERS,
        patch=2,
        channels=16,
        kvheads=2,
        txtlayers=2,
        txtheads=4,
        txtkvheads=4,
    )
    model = SingleStreamDiT(config)
    model.eval()
    return model


def _inputs(model: SingleStreamDiT, batch: int = 2, seed: int = 11):
    generator = torch.Generator().manual_seed(seed)
    total = TXT_TOKENS + IMG_TOKENS
    img = torch.randn(
        batch, IMG_TOKENS, model.PATCH_SIZE**2 * model.LATENT_CHANNELS, generator=generator
    )
    context = torch.randn(
        batch, TXT_TOKENS, model.config.txtlayers, model.config.txtdim, generator=generator
    )
    t = torch.rand(batch, generator=generator)
    pos = torch.randn(batch, total, 3, generator=generator)
    mask = torch.ones(batch, total, dtype=torch.bool)
    return img, context, t, pos, mask


def test_token_cap_matches_the_monolith_padding() -> None:
    assert krea2_token_cap(5, 4) == 256  # 9 -> next multiple of 256
    assert krea2_token_cap(128, 128) == 256
    assert krea2_token_cap(200, 100) == 512  # 300 -> 512


def test_split_reproduces_the_monolith_bit_exactly() -> None:
    model = _tiny_model()
    img, context, t, pos, mask = _inputs(model)

    with torch.no_grad():
        full = model(img, context, t, pos, mask)
        state = krea2_forward_first(
            model,
            model.blocks[:2],
            img,
            timesteps=t,
            context=context,
            pos=pos,
            mask=mask,
        )
        split = krea2_forward_last(model, model.blocks[2:], state)

    assert torch.equal(full, split), (full - split).abs().max().item()


def test_split_is_independent_of_the_cut_point() -> None:
    model = _tiny_model()
    img, context, t, pos, mask = _inputs(model)
    with torch.no_grad():
        reference = model(img, context, t, pos, mask)
        for cut in range(0, LAYERS + 1):
            state = krea2_forward_first(
                model,
                model.blocks[:cut],
                img,
                timesteps=t,
                context=context,
                pos=pos,
                mask=mask,
            )
            split = krea2_forward_last(model, model.blocks[cut:], state)
            assert torch.equal(reference, split), f"cut={cut}"


def test_state_shapes_are_fixed_for_one_geometry() -> None:
    model = _tiny_model()
    img, context, t, pos, mask = _inputs(model)
    with torch.no_grad():
        state = krea2_forward_first(
            model,
            model.blocks[:1],
            img,
            timesteps=t,
            context=context,
            pos=pos,
            mask=mask,
        )
    cap = krea2_token_cap(TXT_TOKENS, IMG_TOKENS)
    assert state.hidden.shape == (2, cap, model.config.features)
    # tproj emits the six-way modulation vector: 6 * features per sample.
    assert state.tvec.shape == (2, 1, 6 * model.config.features)
    assert state.mask.shape == (2, cap)
    assert state.tembed.shape == (2, 1, model.config.features)
    assert (state.txtlen, state.imglen) == (TXT_TOKENS, IMG_TOKENS)


def test_gradients_flow_across_the_cut() -> None:
    model = _tiny_model()
    img, context, t, pos, mask = _inputs(model)
    img.requires_grad_(True)
    state = krea2_forward_first(
        model, model.blocks[:2], img, timesteps=t, context=context, pos=pos, mask=mask
    )
    out = krea2_forward_last(model, model.blocks[2:], state)
    out.sum().backward()
    assert img.grad is not None and img.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Stage modules + non-block submodule release
# ---------------------------------------------------------------------------


def _stage_pair(model: SingleStreamDiT, cut: int = 2):
    from library.models.krea2_raw.pp_stages import Krea2FirstStage, Krea2LastStage

    first = Krea2FirstStage(
        model=model,
        blocks=model.blocks[:cut],
        stage_index=0,
        stages=2,
        block_range=(0, cut),
        max_text_tokens=TXT_TOKENS,
        img_tokens=IMG_TOKENS,
        activation_dtype=torch.float32,
    )
    last = Krea2LastStage(
        model=model,
        blocks=model.blocks[cut:],
        stage_index=1,
        stages=2,
        block_range=(cut, LAYERS),
        max_text_tokens=TXT_TOKENS,
        img_tokens=IMG_TOKENS,
        activation_dtype=torch.float32,
    )
    return first, last


def test_stage_modules_reproduce_the_monolith() -> None:
    model = _tiny_model()
    img, context, t, pos, mask = _inputs(model)
    first, last = _stage_pair(model)
    with torch.no_grad():
        reference = model(img, context, t, pos, mask)
        wire = first(img, timesteps=t, context=context, pos=pos, mask=mask)
        out = last(*wire)
    assert torch.equal(reference, out), (reference - out).abs().max().item()


def test_stage_wire_has_five_fixed_slots() -> None:
    model = _tiny_model()
    img, context, t, pos, mask = _inputs(model)
    first, _ = _stage_pair(model)
    with torch.no_grad():
        wire = first(img, timesteps=t, context=context, pos=pos, mask=mask)
    cap = krea2_token_cap(TXT_TOKENS, IMG_TOKENS)
    hidden, tvec, freqs, mask_f, tembed = wire
    assert hidden.shape == (2, cap, model.config.features)
    assert tvec.shape == (2, 1, 6 * model.config.features)
    assert mask_f.dtype == torch.float32 and mask_f.shape == (2, cap)
    assert tembed.shape == (2, 1, model.config.features)
    assert first.token_cap == cap


def test_release_foreign_submodules_trims_the_right_side() -> None:
    from library.models.krea2_raw.pp_stages import release_foreign_submodules

    head = _tiny_model()
    dropped = release_foreign_submodules(head, is_first=True)
    assert dropped == ("last",)
    assert head.last is None
    assert head.txtfusion is not None and head.first is not None

    tail = _tiny_model()
    dropped = release_foreign_submodules(tail, is_first=False)
    assert set(dropped) == {"first", "tmlp", "tproj", "txtfusion", "txtmlp"}
    for name in dropped:
        assert getattr(tail, name) is None
    assert tail.last is not None


def test_stage_modules_survive_after_the_other_side_is_released() -> None:
    """The trailing stage must still run once the leading-only parts are gone."""

    model = _tiny_model()
    img, context, t, pos, mask = _inputs(model)
    first, last = _stage_pair(model)
    with torch.no_grad():
        wire = first(img, timesteps=t, context=context, pos=pos, mask=mask)
    from library.models.krea2_raw.pp_stages import release_foreign_submodules

    release_foreign_submodules(model, is_first=False)
    with torch.no_grad():
        out = last(*wire)
        reference = _tiny_model()(img, context, t, pos, mask)
    assert torch.equal(reference, out)


# ---------------------------------------------------------------------------
# Wire geometry + boundary metas (runtime step 3, geometry half)
# ---------------------------------------------------------------------------


def _tiny_wire(model, *, microbatches: int = 2):
    from library.training.pp.krea2_runtime import resolve_krea2_wire_spec

    # 4x4 latent with patch 2 -> 2x2 = 4 image tokens
    return resolve_krea2_wire_spec(
        model,
        latent_shape=(2, 16, 1, 4, 4),
        max_text_tokens=TXT_TOKENS,
        microbatches=microbatches,
    )


def test_wire_spec_freezes_geometry() -> None:
    model = _tiny_model()
    wire = _tiny_wire(model)
    assert wire.img_tokens == 4
    assert wire.token_cap == krea2_token_cap(TXT_TOKENS, 4) == 256
    assert wire.dim == model.config.features
    assert wire.microbatch_batch_size == 1
    assert wire.freqs_shape == (1, 256, model.config.features // model.config.heads // 2, 2, 2)


def test_boundary_metas_match_the_real_wire_shapes() -> None:
    from library.training.pp.contract import PpComputePolicy
    from library.training.pp.krea2_runtime import (
        KREA2_BOUNDARY_FROZEN_INDICES,
        krea2_boundary_metas,
    )

    model = _tiny_model()
    wire = _tiny_wire(model)
    policy = PpComputePolicy.from_dtype(torch.float32)
    first, last = _stage_pair(model)
    first_in, first_out = krea2_boundary_metas(
        first, wire=wire, policy=policy, latent_dtype=torch.float32
    )
    last_in, last_out = krea2_boundary_metas(
        last, wire=wire, policy=policy, latent_dtype=torch.float32
    )
    assert first_in[0].shape == wire.microbatch_latent_shape
    assert tuple(t.shape for t in first_out) == tuple(t.shape for t in last_in)
    assert last_out[0].shape == (1, 256, 16)
    assert KREA2_BOUNDARY_FROZEN_INDICES == (1, 2, 3, 4)

    # The frozen slots must be exactly the non-hidden ones.
    img, context, t, pos, mask = _inputs(model)
    with torch.no_grad():
        produced = first(img, timesteps=t, context=context, pos=pos, mask=mask)
    # metas are frozen per *microbatch*; the probe batch is larger, so only the
    # non-batch axes must agree.
    for index, meta in enumerate(first_out):
        assert meta.shape[1:] == produced[index].shape[1:], (
            index,
            meta.shape,
            produced[index].shape,
        )


def test_wire_spec_rejects_a_geometry_it_cannot_freeze() -> None:
    import pytest as _pytest

    from library.training.pp.errors import PipelineParallelError
    from library.training.pp.krea2_runtime import resolve_krea2_wire_spec

    model = _tiny_model()
    with _pytest.raises(PipelineParallelError):
        resolve_krea2_wire_spec(model, latent_shape=(3, 16, 1, 4, 4), max_text_tokens=5, microbatches=2)
    with _pytest.raises(PipelineParallelError):
        resolve_krea2_wire_spec(model, latent_shape=(2, 8, 1, 4, 4), max_text_tokens=5, microbatches=2)


def test_build_krea2_stage_module_splits_the_plan_contiguously() -> None:
    from library.models.krea2_raw.pp_stages import Krea2FirstStage, Krea2LastStage
    from library.training.pp.krea2_runtime import build_krea2_stage_module

    model = _tiny_model()
    wire = _tiny_wire(model, microbatches=1)
    head, head_module = build_krea2_stage_module(model, stage_index=0, stages=2, wire=wire)
    tail, tail_module = build_krea2_stage_module(model, stage_index=1, stages=2, wire=wire)

    assert isinstance(head_module, Krea2FirstStage)
    assert isinstance(tail_module, Krea2LastStage)
    assert head.is_first and not head.is_last
    assert tail.is_last and not tail.is_first
    assert head.block_range[0] == 0
    assert head.block_range[1] == tail.block_range[0]
    assert tail.block_range[1] == LAYERS
    assert head.local_blocks + tail.local_blocks == LAYERS

    img, context, t, pos, mask = _inputs(model)
    with torch.no_grad():
        reference = model(img, context, t, pos, mask)
        wire_tensors = head_module(
            img, timesteps=t, context=context, pos=pos, mask=mask
        )
        out = tail_module(*wire_tensors)
    assert torch.equal(reference, out)


def test_build_krea2_stage_module_rejects_adapters_for_now() -> None:
    import pytest as _pytest

    from library.training.pp.errors import PipelineParallelError
    from library.training.pp.krea2_runtime import build_krea2_stage_module

    model = _tiny_model()
    wire = _tiny_wire(model)
    with _pytest.raises(PipelineParallelError):
        build_krea2_stage_module(model, stage_index=0, stages=2, wire=wire, network=object())
