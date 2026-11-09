"""Acceptance for the Anima PP mini-forward semantics (task t9).

Covers the six divergences between
``scripts/experiments/anima_parallel/pipeline_parallel.py`` and the
authoritative ``library.anima.models.Anima.forward_mini_train_dit``:

1. ``use_fp32`` was hard-coded ``False``
2. the RoPE dtype did not follow the block compute dtype
3. the modulation-guidance final-weight term was missing
4. the ``_dynamic_seq_strict_marks`` SDPA scope was bypassed
5. pooled text modulation was silently ignored -> fail-closed
6. flex attention was silently ignored -> fail-closed

(1) and (2) are pure dtype rules, so they are pinned by function-level
assertions plus a call-level check that the rules actually reach the block stack
and the final layer.  (3) and (4) are pinned numerically on CPU/fp32 (the dtype
rules are inactive there, which is exactly why (1)/(2) need their own tests) and
(4) additionally by asserting the SDPA scope is really entered.  (5) and (6) are
pinned as named fail-closed errors at every entry point.

Equality policy: the mini forward must be **bit-identical** to the authority on
the same device, not merely close.  Any non-zero ``max_abs_diff`` means an
operation, its order, or its dtype drifted, and the assertion message reports the
magnitude so the drift is diagnosable instead of being hidden behind a tolerance.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from library.anima.models import Anima
from scripts.experiments.anima_parallel import pipeline_parallel as pp

MICRO_BLOCKS = 4
MICRO_CHANNELS = 24


def _micro_model(*, seed: int = 123, attn_mode: str = "torch",
                 strict_marks: bool = True, dezero_adaln: bool = True,
                 num_blocks: int = MICRO_BLOCKS) -> Anima:
    """A 4-block Anima small enough to run the full forward on CPU.

    ``dezero_adaln`` is required for meaningful numeric comparisons: both
    ``Block`` and ``FinalLayer`` zero-initialise the last AdaLN projection, so on
    a fresh model the timestep-embedding path (and therefore the
    modulation-guidance final term) is a no-op and a regression would be
    invisible.
    """

    torch.manual_seed(seed)
    model = Anima(
        max_img_h=4,
        max_img_w=4,
        max_frames=1,
        in_channels=4,
        out_channels=4,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=MICRO_CHANNELS,
        num_blocks=num_blocks,
        num_heads=2,
        mlp_ratio=2.0,
        crossattn_emb_channels=MICRO_CHANNELS,
        pos_emb_learnable=True,
        use_adaln_lora=True,
        adaln_lora_dim=8,
        use_llm_adapter=False,
        attn_mode=attn_mode,
    )
    model.eval()
    model.requires_grad_(False)
    # ``compile_blocks()`` is what normally raises this flag; set it directly to
    # exercise the strict-band SDPA path without compiling on CPU.
    model._dynamic_seq_strict_marks = strict_marks
    if dezero_adaln:
        generator = torch.Generator().manual_seed(seed + 7)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "adaln_modulation" in name and parameter.dim() >= 2:
                    parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return model


def _micro_batch(count: int = 2):
    generator = torch.Generator().manual_seed(999)
    return (
        torch.randn(count, 4, 1, 4, 4, generator=generator),
        torch.rand(count, generator=generator),
        torch.randn(count, 3, MICRO_CHANNELS, generator=generator),
        torch.zeros(count, 1, 4, 4),
    )


def _authoritative(model, inputs) -> torch.Tensor:
    x, timesteps, context, padding_mask = inputs
    with torch.no_grad():
        return model.forward_mini_train_dit(
            x, timesteps, context, padding_mask=padding_mask
        )


def _mini_forward(model, inputs) -> torch.Tensor:
    x, timesteps, context, padding_mask = inputs
    with torch.no_grad():
        hidden, embedding, params, kwargs = pp.prepare_block_inputs(
            model, x, timesteps, context, padding_mask
        )
        blocks = pp.run_local_blocks(model, hidden, embedding, context, params, kwargs)
        return pp.finish_model(model, blocks, embedding, kwargs["adaln_lora_B_T_3D"])


# --- (1)(2) dtype rules ------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "use_fp32", "compute_dtype"),
    [
        (torch.float16, True, torch.float32),
        (torch.bfloat16, False, torch.bfloat16),
        (torch.float32, False, torch.float32),
    ],
)
def test_use_fp32_and_rope_dtype_follow_input_dtype(dtype, use_fp32, compute_dtype):
    """models.py:2665 / 2668-2674 — both are functions of the compute dtype."""

    assert pp.resolve_use_fp32(dtype) is use_fp32
    assert pp.resolve_block_compute_dtype(dtype) is compute_dtype

    cos = torch.zeros(4, 1, 1, 8, dtype=torch.bfloat16)
    sin = torch.zeros(4, 1, 1, 8, dtype=torch.bfloat16)
    cast_cos, cast_sin = pp.cast_rope_for_compute((cos, sin), dtype)
    assert cast_cos.dtype is compute_dtype
    assert cast_sin.dtype is compute_dtype


def test_cast_rope_for_compute_passes_none_through():
    assert pp.cast_rope_for_compute(None, torch.float16) is None


@pytest.mark.parametrize(
    ("dtype", "use_fp32", "rope_dtype"),
    [
        (torch.float16, True, torch.float32),
        (torch.bfloat16, False, torch.bfloat16),
        (torch.float32, False, torch.float32),
    ],
)
def test_normalize_block_kwargs_overrides_stale_use_fp32(dtype, use_fp32, rope_dtype):
    """A caller cannot pin the wrong precision.

    ``AnimaOwnedStage._blocks`` builds its kwargs with ``"use_fp32": False``.
    Normalisation at the single ``_run_blocks`` call site makes that stale value
    harmless instead of silently wrong under fp16.
    """

    stale = {
        "rope_cos_sin": (
            torch.zeros(4, 1, 1, 8, dtype=torch.bfloat16),
            torch.zeros(4, 1, 1, 8, dtype=torch.bfloat16),
        ),
        "adaln_lora_B_T_3D": None,
        "use_fp32": False,
    }
    normalized = pp.normalize_block_kwargs(dtype, stale)
    assert normalized["use_fp32"] is use_fp32
    assert normalized["rope_cos_sin"][0].dtype is rope_dtype
    assert normalized["rope_cos_sin"][1].dtype is rope_dtype
    # The caller's mapping must not be mutated in place.
    assert stale["use_fp32"] is False


def test_run_local_blocks_forces_dtype_derived_kwargs(monkeypatch):
    """The dtype rule reaches the real block-stack call site."""

    model = _micro_model()
    seen: dict[str, object] = {}

    def fake_run_blocks(hidden, embedding, crossattn, params, **kwargs):
        seen.update(kwargs)
        return hidden

    monkeypatch.setattr(model, "_run_blocks", fake_run_blocks)
    hidden = torch.zeros(1, 1, 2, 2, MICRO_CHANNELS, dtype=torch.float16)
    rope = (
        torch.zeros(4, 1, 1, 8, dtype=torch.bfloat16),
        torch.zeros(4, 1, 1, 8, dtype=torch.bfloat16),
    )
    pp.run_local_blocks(
        model,
        hidden,
        None,
        None,
        None,
        {"rope_cos_sin": rope, "adaln_lora_B_T_3D": None, "use_fp32": False},
    )
    assert seen["use_fp32"] is True
    assert seen["rope_cos_sin"][0].dtype is torch.float32


def test_finish_model_passes_dtype_derived_use_fp32(monkeypatch):
    """``finish_model`` derives ``use_fp32`` from the hidden dtype (models.py:2665)."""

    model = _micro_model()
    recorded: dict[str, object] = {}

    class _RecordingFinalLayer(torch.nn.Module):
        def forward(self, x, emb, adaln_lora_B_T_3D=None, use_fp32=False):
            recorded["use_fp32"] = use_fp32
            recorded["emb"] = emb
            # Shape expected by unpatchify: M = p1 * p2 * pt * out_channels.
            return torch.zeros(*x.shape[:-1], 16, dtype=x.dtype)

    monkeypatch.setattr(model, "final_layer", _RecordingFinalLayer())

    hidden_fp16 = torch.zeros(1, 1, 2, 2, MICRO_CHANNELS, dtype=torch.float16)
    pp.finish_model(model, hidden_fp16, torch.zeros(1, 1, MICRO_CHANNELS), None)
    assert recorded["use_fp32"] is True

    hidden_fp32 = torch.zeros(1, 1, 2, 2, MICRO_CHANNELS, dtype=torch.float32)
    pp.finish_model(model, hidden_fp32, torch.zeros(1, 1, MICRO_CHANNELS), None)
    assert recorded["use_fp32"] is False


# --- (3)(4) numeric equivalence on CPU/fp32 ----------------------------------


@pytest.mark.parametrize("strict_marks", [True, False])
def test_mini_forward_matches_authoritative_on_cpu_fp32(strict_marks):
    """The split forward is bit-identical to ``forward_mini_train_dit``."""

    model = _micro_model(strict_marks=strict_marks)
    inputs = _micro_batch()
    expected = _authoritative(model, inputs)
    actual = _mini_forward(model, inputs)
    max_abs = float((expected - actual).abs().max())
    assert torch.equal(expected, actual), (
        "PP mini forward drifted from forward_mini_train_dit; "
        f"max_abs_diff={max_abs} (expected 0.0 — a non-zero value means op order "
        "or dtype changed, not an acceptable tolerance)"
    )


def test_mod_guidance_final_term_matches_authoritative():
    """models.py:2778-2780 — the final-weight term must reach ``final_layer``."""

    model = _micro_model()
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        model._mod_guidance_delta.copy_(
            torch.randn(1, MICRO_CHANNELS, generator=generator)
        )
        model._mod_guidance_schedule.copy_(
            torch.randn(MICRO_BLOCKS, generator=torch.Generator().manual_seed(6))
        )
        model._mod_guidance_final_w.fill_(0.37)

    inputs = _micro_batch()
    expected = _authoritative(model, inputs)
    actual = _mini_forward(model, inputs)
    assert torch.equal(expected, actual), (
        "modulation-guidance final term missing or misapplied; "
        f"max_abs_diff={float((expected - actual).abs().max())}"
    )

    # Guard against a vacuous fixture: without the term the output must differ
    # measurably, otherwise this test could not catch the regression at all.
    with torch.no_grad():
        hidden, embedding, params, kwargs = pp.prepare_block_inputs(model, *inputs)
        blocks = pp.run_local_blocks(model, hidden, embedding, inputs[2], params, kwargs)
        without_term = model.unpatchify(
            model.final_layer(
                blocks,
                embedding,
                adaln_lora_B_T_3D=kwargs["adaln_lora_B_T_3D"],
                use_fp32=False,
            )
        )
    assert float((expected - without_term).abs().max()) > 1e-3


@pytest.mark.parametrize("strict_marks", [True, False])
def test_block_sdpa_scope_applied_only_on_strict_marks(strict_marks):
    """models.py:2745-2765 — the strict-band SDPA scope must be reproduced."""

    model = _micro_model(strict_marks=strict_marks)
    scope = pp.block_sdpa_scope(model)
    assert isinstance(scope, contextlib.nullcontext) is (not strict_marks)


def test_run_local_blocks_enters_strict_sdpa_scope(monkeypatch):
    """The scope is actually entered on the block path, not just constructible."""

    import torch.nn.attention as torch_attention

    calls: list[list[object]] = []
    real_sdpa_kernel = torch_attention.sdpa_kernel

    def spy(backends):
        calls.append(list(backends))
        return real_sdpa_kernel(backends)

    monkeypatch.setattr(torch_attention, "sdpa_kernel", spy)
    model = _micro_model(strict_marks=True)
    inputs = _micro_batch(count=1)
    _mini_forward(model, inputs)
    assert len(calls) == 1
    assert [backend.name for backend in calls[0]] == [
        "CUDNN_ATTENTION",
        "EFFICIENT_ATTENTION",
        "MATH",
    ]


def test_block_sdpa_scope_ignores_non_torch_attention():
    """flash attention does not go through the SDPA dispatcher."""

    model = _micro_model(attn_mode="flash", strict_marks=True)
    assert isinstance(pp.block_sdpa_scope(model), contextlib.nullcontext)


# --- (5)(6) fail-closed capabilities -----------------------------------------


def _enable_pooled(model) -> None:
    model.enable_pooled_text_modulation = True


def _enable_flex(model) -> None:
    model.attn_mode = "flex"


def _enable_native_flatten(model) -> None:
    model._native_flatten = True


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (_enable_pooled, pp.PP_POOLED_TEXT_MODULATION_UNSUPPORTED),
        (_enable_flex, pp.PP_FLEX_ATTENTION_UNSUPPORTED),
        (_enable_native_flatten, pp.PP_NATIVE_FLATTEN_UNSUPPORTED),
    ],
    ids=["pooled_text_modulation", "flex_attention", "native_flatten"],
)
@pytest.mark.parametrize("entry", ["prepare_block_inputs", "run_local_blocks", "finish_model"])
def test_unsupported_capability_fails_closed_at_every_entry(entry, mutate, code):
    """Never silently ignore a capability the split forward cannot reproduce."""

    model = _micro_model()
    mutate(model)
    calls = {
        "prepare_block_inputs": lambda: pp.prepare_block_inputs(
            model, None, None, None, None
        ),
        "run_local_blocks": lambda: pp.run_local_blocks(
            model, None, None, None, None, {}
        ),
        "finish_model": lambda: pp.finish_model(model, None, None, None),
    }
    with pytest.raises(pp.AnimaPipelineUnsupported) as excinfo:
        calls[entry]()
    assert excinfo.value.code == code
    # The code must be visible in the message too: preflight output and logs
    # surface str(exc), not the attribute.
    assert code in str(excinfo.value)


def test_supported_contract_does_not_raise():
    """The guards must not reject the ordinary Anima PP configuration."""

    model = _micro_model()
    pp.assert_mini_forward_supported(model)
    pp.assert_mod_guidance_indexable(model)


def test_mod_guidance_on_sliced_stage_fails_closed():
    """Per-block guidance indexes global block ids; a slice would misapply it."""

    model = _micro_model()
    with torch.no_grad():
        model._mod_guidance_delta.fill_(0.1)
        model._mod_guidance_final_w.fill_(0.5)
    model.blocks = torch.nn.ModuleList(list(model.blocks)[2:])
    with pytest.raises(pp.AnimaPipelineUnsupported) as excinfo:
        pp.assert_mod_guidance_indexable(model)
    assert excinfo.value.code == pp.PP_MOD_GUIDANCE_UNSUPPORTED


def test_finish_model_requires_mod_guidance_contract():
    """A model without the guidance buffers must not be finished silently."""

    class _NoGuidance:
        enable_pooled_text_modulation = False
        attn_mode = "torch"
        _native_flatten = False

    with pytest.raises(pp.AnimaPipelineUnsupported) as excinfo:
        pp.finish_model(_NoGuidance(), None, None, None)
    assert excinfo.value.code == pp.PP_MOD_GUIDANCE_CONTRACT_UNSUPPORTED
