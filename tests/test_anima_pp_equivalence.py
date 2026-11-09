"""CPU + Gloo equivalence evidence for Anima PP=2 (task t3).

What this file proves, and what it explicitly does *not* claim:

* ``microbatches == 1`` — the stage boundary, the prediction, the loss and the
  adapter gradients are **bit-identical** (``torch.equal``) to the single-device
  reference.  A single microbatch is the only case where the split changes
  neither the batch shapes nor the gradient accumulation order, so it is the
  only case where bit-equality is claimed.
* ``microbatches >= 2`` — only **numerical** equivalence is claimed.  Splitting
  the batch changes the shapes each GEMM sees and the order in which
  per-microbatch contributions are summed, so the multi-microbatch cases report
  ``max_abs_diff`` / ``max_rel_diff`` and are deliberately not labelled
  bit-exact (``CaseConfig.bit_exact_claimed`` is asserted ``False``).  Each of
  them also reports a comparison against a *microbatched* single-device
  reference, which separates "the pipeline added drift" from "microbatching
  added drift".
* The last stage is the only stage that computes a loss.
* Both stages agree on which boundary tensors exchange gradients (the "skip"
  decision), and the executed p2p traffic is symmetric.

Everything runs on CPU with the Gloo backend; no test here needs CUDA.

torch 2.10 constraints the harness works around are documented in
``tests/pp_equivalence_test_support.py`` and pinned by the tests below:
``n_microbatches >= num_stages`` for *every* schedule, ``scale_grads`` not
seeing monkey-patched LoRA parameters, and the zero-initialised AdaLN
projections that would otherwise make every comparison vacuous.
"""

from __future__ import annotations

import pytest
import torch

from tests import pp_equivalence_dist_test_support as dist
from tests import pp_equivalence_test_support as h

#: Numerical tolerances for the ``microbatches >= 2`` cases.  Measured drift on
#: this fixture is ~1e-6 (predictions/boundary) and ~3e-8 (gradients), i.e. four
#: orders of magnitude below these bounds; the bounds are loose enough not to
#: flake and tight enough to catch a real defect (for example the un-scaled
#: ``scale_grads`` trap shows up as a ~4e-2 gradient error).
NUMERIC_ATOL = 1e-5
NUMERIC_RTOL = 1e-3


# --- shared assertions -------------------------------------------------------


def _assert_loss_only_on_last_stage(report: dict, microbatches: int) -> None:
    first = dist.rank_report(report, 0)
    last = dist.rank_report(report, 1)
    assert first["loss_count"] == 0, "the first stage must not compute a loss"
    assert last["loss_count"] == microbatches, (
        "the last stage must compute exactly one loss per microbatch; got "
        f"{last['loss_count']} for {microbatches} microbatches"
    )
    assert not any("final_layer" in name for name in first["structure"]["stage_modules"])
    assert "final_layer" in last["structure"]["stage_modules"]


def _assert_boundary_bits_survive_the_wire(report: dict) -> None:
    sent = dist.rank_report(report, 0)["boundary_sent"]
    received = dist.rank_report(report, 1)["boundary_received"]
    assert sent == received, (
        "the boundary activation changed while crossing the Gloo wire; a "
        "float32 send/recv must be bit-preserving"
    )
    assert sent == dist.rank_report(report, 0)["reference_boundary"], (
        "the pre stage emitted a boundary that differs from the single-device "
        "reference"
    )


def _assert_skip_decision_agrees(report: dict, microbatches: int) -> None:
    """Both stages must agree on the gradient exchange plan for every entry."""

    first = dist.rank_report(report, 0)["boundary"]
    last = dist.rank_report(report, 1)["boundary"]
    assert first["spec"] == last["spec"]
    assert first["on_wire"] == last["on_wire"]
    assert first["skipped"] == last["skipped"]
    assert first["plan"] == last["plan"], (
        "the two stages disagree on which boundary entries exchange gradients"
    )
    assert first["plan"] == [
        name in first["on_wire"] for name in first["spec"]
    ], (
        "every tensor on the wire must exchange gradients on torch 2.10; the "
        "receiver mirrors the sender's activation send info"
    )
    # The sender emits exactly the activations the receiver expects gradients for.
    assert first["stage_output_tensors"] == len(first["on_wire"])
    assert last["stage_input_tensors"] == len(first["on_wire"])
    assert first["stage_output_tensors"] == last["stage_input_tensors"]
    assert last["grad_send_slots"] == first["stage_output_tensors"]
    assert first["grad_recv_slots"] == [len(first["on_wire"])]
    assert last["grad_recv_slots"] == [0]
    # The executed traffic is symmetric, so no rank is left waiting.  The
    # per-microbatch op count equals the number of tensors on the wire.
    first_comm = dist.rank_report(report, 0)["communication"]
    last_comm = dist.rank_report(report, 1)["communication"]
    per_microbatch = len(first["on_wire"])
    expected_calls = microbatches * per_microbatch
    assert first_comm["fwd_send_calls"] == last_comm["fwd_recv_calls"] == expected_calls
    assert last_comm["bwd_send_calls"] == first_comm["bwd_recv_calls"] == expected_calls
    assert first_comm["fwd_send_bytes"] == last_comm["fwd_recv_bytes"] > 0
    assert last_comm["bwd_send_bytes"] == first_comm["bwd_recv_bytes"] > 0
    assert first_comm["fwd_recv_calls"] == 0 and first_comm["bwd_send_calls"] == 0
    assert last_comm["fwd_send_calls"] == 0 and last_comm["bwd_recv_calls"] == 0


def _assert_bit_exact(report: dict, rank: int, name: str) -> None:
    data = dist.comparison(report, rank, name)
    assert data["equal"], (
        f"{name} on rank {rank} is not bit-exact: "
        f"max_abs_diff={data['max_abs_diff']:.3e} "
        f"max_rel_diff={data['max_rel_diff']:.3e}\n"
        + dist.report_summary(report)
    )
    assert data["max_abs_diff"] == 0.0


def _assert_numerically_equal(report: dict, rank: int, name: str) -> dict:
    data = dist.comparison(report, rank, name)
    assert data["max_abs_diff"] <= NUMERIC_ATOL, (
        f"{name} on rank {rank} exceeded atol={NUMERIC_ATOL}: "
        f"max_abs_diff={data['max_abs_diff']:.3e} "
        f"rel_to_scale={data['rel_to_scale']:.3e}\n"
        + dist.report_summary(report)
    )
    assert data["rel_to_scale"] <= NUMERIC_RTOL, (
        f"{name} on rank {rank} exceeded rtol={NUMERIC_RTOL}: "
        f"max_abs_diff={data['max_abs_diff']:.3e} "
        f"rel_to_scale={data['rel_to_scale']:.3e}\n"
        + dist.report_summary(report)
    )
    return data


def _assert_gradient_cosine(report: dict, rank: int, name: str) -> dict:
    """Track ②: the gradient *direction* must match the single-device reference.

    Deliberately kept next to the element-wise assertions instead of replacing
    them.  Cosine similarity is scale invariant, so it answers "does the update
    point the same way as the full-batch step?" while ``atol``/``rtol`` answer
    "and is its magnitude right?"; the torch 2.10 ``scale_grads`` trap (adapter
    gradients ``n`` times too large) is invisible to cosine but caught by the
    element-wise bound, and a genuine direction drift is the other way round.
    """

    data = dist.cosine(report, rank, name)
    detail = ", ".join(
        f"{key}={value:.9f}" for key, value in sorted(data["per_tensor"].items())
    )
    assert data["aggregate"] > h.COSINE_THRESHOLD, (
        f"{name} on rank {rank}: aggregate gradient cosine "
        f"{data['aggregate']:.9f} is below {h.COSINE_THRESHOLD}; "
        f"per-tensor: {detail}\n" + dist.report_summary(report)
    )
    assert data["min_per_tensor"] > h.COSINE_THRESHOLD, (
        f"{name} on rank {rank}: at least one adapter gradient points away from "
        f"the reference (min {data['min_per_tensor']:.9f}); "
        f"per-tensor: {detail}\n" + dist.report_summary(report)
    )
    return data


# --- fixture validity (no distributed machinery) -----------------------------


def test_timestep_conditioning_matches_prepare_block_inputs():
    """The post stage's conditioning helper must equal the authoritative region."""

    from scripts.experiments.anima_parallel import pipeline_parallel as pp

    model, _ = h.build_micro_anima()
    inputs, _ = h.build_micro_batch(2)
    with torch.no_grad():
        _, embedding, _, kwargs = pp.prepare_block_inputs(model, *inputs)
        helper_embedding, helper_adaln = h.timestep_conditioning(model, inputs[1])
    assert torch.equal(helper_embedding, embedding)
    assert torch.equal(helper_adaln, kwargs["adaln_lora_B_T_3D"])


def test_fixture_conditioning_paths_are_live():
    """A stock Anima model would make the comparisons below vacuous.

    ``Block``/``FinalLayer`` zero-initialise their AdaLN output projections, so
    on a fresh model the final layer ignores ``t_embedding`` completely.  The
    fixture de-zeroes them; this test proves the fixture is not vacuous *and*
    documents the trap by showing the stock model cannot see the difference.
    """

    from scripts.experiments.anima_parallel import pipeline_parallel as pp

    model, _ = h.build_micro_anima()
    assert model.final_layer.adaln_modulation[2].weight.count_nonzero() > 0
    for index, block in enumerate(model.blocks):
        assert block.adaln_up_mlp.weight.count_nonzero() > 0, index
        assert block.adaln_up_self_attn.weight.count_nonzero() > 0, index

    inputs, _ = h.build_micro_batch(1)

    def final_outputs(current_model):
        with torch.no_grad():
            hidden, embedding, params, kwargs = pp.prepare_block_inputs(
                current_model, *inputs
            )
            blocks = pp.run_local_blocks(
                current_model, hidden, embedding, inputs[2], params, kwargs
            )
            adaln = kwargs["adaln_lora_B_T_3D"]
            with_embedding = pp.finish_model(current_model, blocks, embedding, adaln)
            without_embedding = pp.finish_model(
                current_model, blocks, torch.zeros_like(embedding), adaln
            )
        return with_embedding, without_embedding

    live_with, live_without = final_outputs(model)
    assert float((live_with - live_without).abs().max()) > 1e-3, (
        "the fixture's final-layer timestep path is dead; a wrong-microbatch "
        "conditioning bug would be invisible"
    )

    stock_model, _ = h.build_micro_anima(dezero_adaln=False)
    stock_with, stock_without = final_outputs(stock_model)
    assert torch.equal(stock_with, stock_without), (
        "a stock model is expected to be blind to the final-layer timestep term; "
        "if this changes, revisit the de-zero requirement"
    )


def test_lora_adapters_are_not_in_the_model_parameter_tree():
    """Why the stage module has to register its adapters (docstring, fact 2).

    ``Schedule*.scale_grads`` divides ``submod.parameters()``.  A LoRA adapter
    monkey-patched onto frozen ``nn.Linear`` modules is not reachable from the
    model, so without explicit registration the microbatch gradients are never
    averaged and come out ``n`` times too large.
    """

    model, network = h.build_micro_anima()
    model_ids = {id(parameter) for parameter in model.parameters()}
    adapter_ids = {
        id(parameter) for parameter in network.parameters() if parameter.requires_grad
    }
    assert adapter_ids, "the fixture must expose trainable adapter parameters"
    assert not (model_ids & adapter_ids), (
        "adapter parameters unexpectedly live inside the model tree; the "
        "scale_grads reasoning in pp_equivalence_test_support.py needs review"
    )


def test_stage_ownership_is_local_and_adapters_are_registered():
    model, network = h.build_micro_anima()
    inputs, _ = h.build_micro_batch(1)
    examples = h.stage_examples(model, inputs)

    first = h.AnimaPPStageModule(
        model,
        network,
        rank=0,
        split=h.BLOCK_SPLIT,
        rope=examples.rope,
        batch_inputs=inputs,
    )
    assert set(first.model._modules) == {
        "blocks",
        "x_embedder",
        "pos_embedder",
        "t_embedder",
        "t_embedding_norm",
    }
    assert not hasattr(first.model, "final_layer")
    assert sorted(first.adapters) == [
        f"lora_unet_blocks_{block}_mlp_layer{layer}"
        for block in range(h.BLOCK_SPLIT)
        for layer in (1, 2)
    ]
    # Every owned adapter is visible to ``submod.parameters()`` (scale_grads).
    stage_parameters = list(first.parameters())
    for parameter in first.adapters.parameters():
        assert any(parameter is candidate for candidate in stage_parameters)
    assert sum(1 for p in first.parameters() if p.requires_grad) == 8

    second_model, second_network = h.build_micro_anima()
    second = h.AnimaPPStageModule(
        second_model,
        second_network,
        rank=1,
        split=h.BLOCK_SPLIT,
        rope=examples.rope,
        batch_inputs=inputs,
    )
    assert set(second.model._modules) == {
        "blocks",
        "t_embedder",
        "t_embedding_norm",
        "final_layer",
    }
    assert not hasattr(second.model, "x_embedder")
    assert sorted(second.adapters) == [
        f"lora_unet_blocks_{block}_mlp_layer{layer}"
        for block in range(h.BLOCK_SPLIT, h.MICRO_BLOCKS)
        for layer in (1, 2)
    ]
    assert sum(1 for p in second.parameters() if p.requires_grad) == 8


def test_split_forward_matches_authoritative_forward():
    """Splitting the block stack must not change a single bit on CPU/fp32."""

    inputs, _ = h.build_micro_batch(1)
    split = h.local_pp_forward(inputs)

    model, _ = h.build_micro_anima()
    with torch.no_grad():
        expected = model(*inputs[:3], padding_mask=inputs[3])
    assert torch.equal(split.prediction, expected), (
        "the split forward drifted from forward_mini_train_dit: "
        f"max_abs_diff={float((split.prediction - expected).abs().max()):.3e}"
    )

    reference_model, _ = h.build_micro_anima()
    reference = h.reference_boundary(reference_model, inputs)
    assert torch.equal(torch.cat(split.boundary), torch.cat(reference))


def test_split_forward_multi_microbatch_is_numerically_equal():
    """With two microbatches the split forward is numerically, not bit, equal."""

    inputs, _ = h.build_micro_batch(2)
    split = h.local_pp_forward(inputs)

    model, _ = h.build_micro_anima()
    with torch.no_grad():
        expected = model(*inputs[:3], padding_mask=inputs[3])
    difference = float((split.prediction - expected).abs().max())
    assert difference <= NUMERIC_ATOL, (
        "the split forward diverged beyond the numerical tolerance: "
        f"max_abs_diff={difference:.3e}"
    )
    assert not torch.equal(split.prediction, expected), (
        "batching two samples in one forward is expected to differ in the low "
        "bits from two single-sample forwards; if that stopped being true the "
        "multi-microbatch equality policy could be revisited"
    )

    reference_model, _ = h.build_micro_anima()
    reference = h.reference_boundary(reference_model, inputs)
    assert torch.equal(torch.cat(split.boundary), torch.cat(reference))


def test_boundary_skip_plan_is_identical_for_both_stages():
    """Both stages derive the same gradient exchange plan for the same boundary.

    torch 2.10 mirrors the sender's activation send info into the receiver's
    gradient receive info, so a per-tensor "skip" on one side only would
    deadlock.  The plan is therefore a pure function of the boundary spec, and
    both stages must also agree on what actually crossed the boundary.
    """

    expected_skipped = {
        False: ("t_embedding", "adaln_lora", "rope_cos_sin", "crossattn_emb"),
        True: ("rope_cos_sin", "crossattn_emb"),
    }
    for wire_conditioning in (False, True):
        spec = h.boundary_spec(wire_conditioning=wire_conditioning)
        plan = h.grad_exchange_plan(spec)
        assert plan == tuple(entry.on_wire for entry in spec)
        assert h.skipped_entries(spec) == expected_skipped[wire_conditioning]
        assert spec[0].name == "hidden" and spec[0].requires_grad
        assert all(plan[index] for index, entry in enumerate(spec) if entry.on_wire)
        on_wire = [entry.name for entry in spec if entry.on_wire]
        assert on_wire == (
            ["hidden", "t_embedding", "adaln_lora"] if wire_conditioning else ["hidden"]
        )

    # Two real stages, in process: what the pre stage emitted is exactly what the
    # post stage received, for every microbatch and for both boundary modes.
    inputs, _ = h.build_micro_batch(2)
    for wire_conditioning in (False, True):
        split = h.local_pp_forward(inputs, wire_conditioning=wire_conditioning)
        pre, post = split.modules
        assert len(pre.boundary_out) == len(post.boundary_in) == inputs[0].shape[0]
        for sent, received in zip(pre.boundary_out, post.boundary_in):
            assert torch.equal(sent, received)


def test_stock_schedules_reject_single_microbatch():
    """torch 2.10 rejects ``n_microbatches < num_stages`` for every schedule.

    This is why the ``microbatches == 1`` case is driven through the per-chunk
    stage API instead of ``ScheduleGPipe``: it is *not* a 1F1B claim, and it is
    not a GPipe-schedule claim either.
    """

    from torch.distributed.pipelining import Schedule1F1B, ScheduleGPipe

    class _StubStage:
        num_stages = 2

    for schedule_type in (ScheduleGPipe, Schedule1F1B):
        with pytest.raises(ValueError, match="Number of microbatches"):
            schedule_type(_StubStage(), n_microbatches=1)

    with pytest.raises(ValueError, match="n_microbatches >= num_stages"):
        h.build_schedule(_StubStage(), microbatches=1, schedule="gpipe")


def test_mod_guidance_on_a_sliced_stage_fails_closed():
    """Per-block guidance indexes global block ids, so a stage must refuse it."""

    from scripts.experiments.anima_parallel import pipeline_parallel as pp

    model, network = h.build_micro_anima()
    with torch.no_grad():
        model._mod_guidance_delta.fill_(0.1)
        model._mod_guidance_final_w.fill_(0.5)
    inputs, _ = h.build_micro_batch(1)
    with torch.no_grad():
        _, _, _, kwargs = pp.prepare_block_inputs(model, *tuple(v[:1] for v in inputs))
    rope = kwargs["rope_cos_sin"]

    for rank in (0, 1):
        # A fresh model per rank: constructing a stage module consumes it.
        stage_model, stage_network = h.build_micro_anima()
        with torch.no_grad():
            stage_model._mod_guidance_delta.fill_(0.1)
            stage_model._mod_guidance_final_w.fill_(0.5)
        with pytest.raises(pp.AnimaPipelineUnsupported) as excinfo:
            h.AnimaPPStageModule(
                stage_model,
                stage_network,
                rank=rank,
                split=h.BLOCK_SPLIT,
                rope=rope,
                batch_inputs=inputs,
            )
        assert excinfo.value.code == pp.PP_MOD_GUIDANCE_UNSUPPORTED


def test_harness_detects_wrong_microbatch_conditioning():
    """Negative control: the harness must fail when conditioning is mismatched.

    The post stage recomputes the frozen timestep conditioning from its own copy
    of the microbatch.  Feeding it the wrong microbatch's conditioning must move
    the prediction well beyond the numerical tolerance, otherwise every
    comparison in this file would be worthless.
    """

    inputs, _ = h.build_micro_batch(2)
    good = h.local_pp_forward(inputs)
    swapped = h.local_pp_forward(inputs, post_microbatch_order=(1, 0))

    model, _ = h.build_micro_anima()
    with torch.no_grad():
        expected = torch.cat(
            [
                model(
                    *(value[index : index + 1] for value in inputs[:3]),
                    padding_mask=inputs[3][index : index + 1],
                )
                for index in range(2)
            ]
        )

    assert torch.equal(good.prediction, expected), (
        "the correctly wired split must be bit-exact against the per-sample "
        f"reference (max_abs_diff="
        f"{float((good.prediction - expected).abs().max()):.3e})"
    )
    gap = float((swapped.prediction - expected).abs().max())
    assert gap > 1e-3, (
        "the fixture cannot detect a wrong-microbatch conditioning bug; the "
        f"conditioning path is too weak (gap={gap:.3e})"
    )


# --- 2-rank Gloo equivalence cases ------------------------------------------


@pytest.mark.integration
@pytest.mark.probe
def test_pp_cpu_gloo_microbatch1_is_bit_exact(tmp_path):
    """``microbatches == 1``: boundary, prediction, loss and gradients bit-equal.

    Driven through the per-chunk stage API because torch 2.10 refuses
    ``n_microbatches=1`` for a 2-stage pipeline in *every* stock schedule.  This
    case therefore makes no 1F1B claim.
    """

    case = dist.CaseConfig(microbatches=1, schedule="single", wire_conditioning=False)
    report = dist.run_case(tmp_path / "case", case)

    _assert_loss_only_on_last_stage(report, 1)
    _assert_boundary_bits_survive_the_wire(report)
    _assert_skip_decision_agrees(report, 1)

    _assert_bit_exact(report, 0, "boundary_vs_full_batch")
    _assert_bit_exact(report, 0, "boundary_vs_microbatched")
    _assert_bit_exact(report, 0, "grads_vs_full_batch")
    _assert_bit_exact(report, 0, "grads_vs_microbatched")
    _assert_gradient_cosine(report, 0, "grads_vs_full_batch")
    _assert_gradient_cosine(report, 0, "grads_vs_microbatched")
    _assert_bit_exact(report, 1, "boundary_vs_full_batch")
    _assert_bit_exact(report, 1, "prediction_vs_full_batch")
    _assert_bit_exact(report, 1, "prediction_vs_microbatched")
    _assert_bit_exact(report, 1, "losses_vs_microbatched")
    _assert_bit_exact(report, 1, "mean_loss_vs_full_batch")
    _assert_bit_exact(report, 1, "grads_vs_full_batch")
    _assert_bit_exact(report, 1, "grads_vs_microbatched")
    _assert_gradient_cosine(report, 1, "grads_vs_full_batch")
    _assert_gradient_cosine(report, 1, "grads_vs_microbatched")


@pytest.mark.integration
@pytest.mark.probe
def test_pp_cpu_gloo_microbatch1_wire_conditioning_is_bit_exact(tmp_path):
    """Same as above with the frozen conditioning on the wire.

    Both stages must still exchange gradients for those frozen tensors (the
    receiver mirrors the sender's send info), so this pins the "skip" decision
    for a boundary that carries non-differentiable tensors.
    """

    case = dist.CaseConfig(microbatches=1, schedule="single", wire_conditioning=True)
    report = dist.run_case(tmp_path / "case", case)

    _assert_loss_only_on_last_stage(report, 1)
    _assert_boundary_bits_survive_the_wire(report)
    _assert_skip_decision_agrees(report, 1)
    assert dist.rank_report(report, 0)["boundary"]["on_wire"] == [
        "hidden",
        "t_embedding",
        "adaln_lora",
    ]

    _assert_bit_exact(report, 0, "boundary_vs_full_batch")
    _assert_bit_exact(report, 0, "grads_vs_full_batch")
    _assert_bit_exact(report, 1, "prediction_vs_full_batch")
    _assert_bit_exact(report, 1, "mean_loss_vs_full_batch")
    _assert_bit_exact(report, 1, "grads_vs_full_batch")
    _assert_gradient_cosine(report, 0, "grads_vs_full_batch")
    _assert_gradient_cosine(report, 1, "grads_vs_full_batch")


@pytest.mark.integration
@pytest.mark.probe
def test_pp_cpu_gloo_microbatch2_is_numerically_equivalent(tmp_path):
    """``microbatches == 2`` with GPipe: numeric equivalence only.

    This case is **not** called bit-exact: splitting the batch changes both the
    GEMM shapes and the order in which per-microbatch contributions are summed.
    The numbers are reported, and the comparison against the microbatched
    single-device reference separates pipeline drift from microbatch drift.
    """

    case = dist.CaseConfig(microbatches=2, schedule="gpipe")
    assert case.bit_exact_claimed is False
    report = dist.run_case(tmp_path / "case", case)

    _assert_loss_only_on_last_stage(report, 2)
    _assert_boundary_bits_survive_the_wire(report)
    _assert_skip_decision_agrees(report, 2)

    # The split itself is bit-exact: per-microbatch work is unchanged.
    _assert_bit_exact(report, 0, "boundary_vs_microbatched")
    _assert_bit_exact(report, 1, "boundary_vs_microbatched")
    _assert_bit_exact(report, 1, "prediction_vs_microbatched")
    _assert_bit_exact(report, 1, "losses_vs_microbatched")

    # Against the single-device full-batch reference only numerics are claimed.
    _assert_numerically_equal(report, 0, "boundary_vs_full_batch")
    _assert_numerically_equal(report, 0, "grads_vs_full_batch")
    _assert_numerically_equal(report, 1, "prediction_vs_full_batch")
    _assert_numerically_equal(report, 1, "mean_loss_vs_full_batch")
    _assert_numerically_equal(report, 1, "grads_vs_full_batch")
    _assert_numerically_equal(report, 1, "grads_vs_microbatched")
    # Track ② — direction agreement with the full-batch step, alongside (not
    # instead of) the element-wise bounds above.
    _assert_gradient_cosine(report, 0, "grads_vs_full_batch")
    _assert_gradient_cosine(report, 1, "grads_vs_full_batch")
    _assert_gradient_cosine(report, 1, "grads_vs_microbatched")


@pytest.mark.integration
@pytest.mark.probe
def test_pp_cpu_gloo_microbatch3_1f1b_is_numerically_equivalent(tmp_path):
    """``microbatches == 3`` with Schedule1F1B: numeric equivalence only.

    3 is the smallest microbatch count where 1F1B genuinely departs from GPipe
    ordering on a 2-stage pipeline (``microbatches == 2`` is GPipe-equivalent and
    ``microbatches == 1`` is rejected outright by torch 2.10).
    """

    case = dist.CaseConfig(microbatches=3, schedule="1f1b")
    assert case.bit_exact_claimed is False
    report = dist.run_case(tmp_path / "case", case)

    _assert_loss_only_on_last_stage(report, 3)
    _assert_boundary_bits_survive_the_wire(report)
    _assert_skip_decision_agrees(report, 3)

    _assert_bit_exact(report, 0, "boundary_vs_microbatched")
    _assert_bit_exact(report, 1, "boundary_vs_microbatched")
    _assert_bit_exact(report, 1, "prediction_vs_microbatched")
    _assert_bit_exact(report, 1, "losses_vs_microbatched")

    _assert_numerically_equal(report, 0, "boundary_vs_full_batch")
    _assert_numerically_equal(report, 0, "grads_vs_full_batch")
    _assert_numerically_equal(report, 1, "prediction_vs_full_batch")
    _assert_numerically_equal(report, 1, "mean_loss_vs_full_batch")
    _assert_numerically_equal(report, 1, "grads_vs_full_batch")
    # Track ② — 1F1B changes the accumulation order, so the low bits differ; the
    # cosine says the update still points the same way as the full-batch step.
    _assert_gradient_cosine(report, 0, "grads_vs_full_batch")
    _assert_gradient_cosine(report, 1, "grads_vs_full_batch")
    # 1F1B backward order differs from the naive per-microbatch loop, so the
    # gradient accumulation order (and therefore the low bits) differ here.
    _assert_numerically_equal(report, 1, "grads_vs_microbatched")
    _assert_gradient_cosine(report, 1, "grads_vs_microbatched")
