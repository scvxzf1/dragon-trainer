"""Tests for the t24 per-micro-batch determinism probe.

The probe exists so the cross-shape (single batch2 vs dual batch1) and resume
divergences can be *judged mechanically* instead of by reading logs. These tests
pin three things:

* the record really carries every field the verdicts need (①-⑧ of the contract);
* each verdict fires for its own cause and **only** for its own cause — a
  perturbation control proves the probe cannot "report divergence for everything";
* the probe is inert until its environment variable names an output path.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

from bench.multigpu_reference import determinism_probe as probe
from library.training.multigpu import noise_fixture


def _fixture_file(tmp_path, slots=8):
    rows = [
        {
            "sample_id": f"s{index:08d}",
            "timestep": 0.2 + 0.01 * index,
            "noise": [0.05 * index, -0.05 * index],
        }
        for index in range(slots)
    ]
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "fixture.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _state(rank=0, world=1, step=0, consumed=0):
    return SimpleNamespace(
        accelerator=SimpleNamespace(
            process_index=rank, num_processes=world, device=torch.device("cpu")
        ),
        global_step=step,
        batches_consumed=consumed,
        network=None,
    )


def _batch(size=2, shift=0.0):
    return {
        "latents": torch.arange(size * 4, dtype=torch.float32).reshape(size, 4) + shift,
        "te": torch.ones(size, 3) * (1.0 + shift),
    }


def _record(tmp_path, monkeypatch, *, rank=0, world=1, step=0, consumed=0, size=2,
            shift=0.0, lr=0.1, adapter_scale=1.0, loss_shift=0.0):
    """Run one recorded micro-batch against a real AdamW and return the record."""
    fixture = _fixture_file(tmp_path)
    monkeypatch.setenv(noise_fixture.ENV, str(fixture))
    noise_fixture.reset_cache()
    recorder = probe.Recorder(path=tmp_path / f"probe-{rank}.jsonl").open()
    parameter = torch.nn.Parameter(torch.ones(4) * adapter_scale)
    optimizer = torch.optim.AdamW([parameter], lr=lr)
    state = _state(rank=rank, world=world, step=step, consumed=consumed)
    recorder.begin(None, state, _batch(size, shift))
    recorder.note_loss_call(
        torch.tensor([0.5 + loss_shift, 0.7 + loss_shift]), 0.6 + loss_shift
    )
    before = probe.optimizer_adapter_digest(optimizer)
    (parameter * 2.0).sum().backward()
    optimizer.step()
    recorder.on_optimizer_step(optimizer, adapter_hash_before=before)
    recorder.close()
    noise_fixture.reset_cache()
    return probe.load(tmp_path / f"probe-{rank}.jsonl")[-1]


def test_the_record_carries_every_field_the_verdicts_need(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch, step=1, consumed=1)

    assert record["schema"] == probe.SCHEMA
    # ① rank / ② global_step / ③ slot ids + sigmas
    assert record["rank"] == 0
    assert record["global_step"] == 1
    assert record["slots"] == ["s00000002", "s00000003"]
    assert record["sigmas"] == pytest.approx([0.22, 0.23])
    # ④ latent and text-cache fingerprints
    assert record["latent_fingerprint"]
    assert record["text_fingerprint"]
    # ⑤ per-sample and batch loss
    assert record["per_sample_loss"] == pytest.approx([0.5, 0.7])
    assert record["loss"] == pytest.approx(0.6)
    # ⑥ gradient norms: total, per group, per adapter role
    assert record["grad_norm_total"] > 0
    assert len(record["grad_norm_by_group"]) == 1
    assert record["grad_norm_by_prefix"]
    # ⑦ step-0 adapter hash
    assert record["adapter_hash"]
    assert record["adapter_tensors"] == 1
    # ⑧ AdamW moments (resume continuation)
    assert record["exp_avg_norm"] > 0
    assert record["exp_avg_sq_norm"] > 0
    assert record["state_entries"] == 1


def test_identical_records_are_not_reported_as_divergent(tmp_path, monkeypatch):
    left = _record(tmp_path / "a", monkeypatch)
    right = _record(tmp_path / "b", monkeypatch)

    verdict = probe.compare([left], [right])
    assert verdict["verdict"] == probe.CODE_AGREE, verdict["detail"]


def test_only_a_different_adapter_hash_gives_the_init_verdict(tmp_path, monkeypatch):
    left = _record(tmp_path / "a", monkeypatch, adapter_scale=1.0)
    same = _record(tmp_path / "b", monkeypatch, adapter_scale=1.0)
    other = _record(tmp_path / "c", monkeypatch, adapter_scale=1.5)

    assert probe.compare([left], [same])["verdict"] == probe.CODE_AGREE
    verdict = probe.compare([left], [other])
    assert verdict["verdict"] == probe.CODE_INIT
    assert verdict["first_difference"]["field"] == "adapter_hash"


def test_only_a_loss_difference_gives_the_forward_verdict(tmp_path, monkeypatch):
    left = _record(tmp_path / "a", monkeypatch)
    perturbed = _record(tmp_path / "b", monkeypatch, loss_shift=1e-3)

    verdict = probe.compare([left], [perturbed])
    assert verdict["verdict"] == probe.CODE_FORWARD
    assert verdict["first_difference"]["field"] == "loss"


def test_only_a_grad_norm_difference_gives_the_backward_verdict(
    tmp_path, monkeypatch
):
    left = _record(tmp_path / "a", monkeypatch)
    # Same inputs and same loss, but a heavier backwards path.
    fixture = _fixture_file(tmp_path)
    monkeypatch.setenv(noise_fixture.ENV, str(fixture))
    noise_fixture.reset_cache()
    recorder = probe.Recorder(path=tmp_path / "probe-0.jsonl").open()
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    recorder.begin(None, _state(), _batch(2))
    recorder.note_loss_call(torch.tensor([0.5, 0.7]), 0.6)
    before = probe.optimizer_adapter_digest(optimizer)
    ((parameter * 3.0).sum()).backward()
    optimizer.step()
    recorder.on_optimizer_step(optimizer, adapter_hash_before=before)
    recorder.close()
    heavier = probe.load(tmp_path / "probe-0.jsonl")[-1]

    verdict = probe.compare([left], [heavier])
    assert verdict["verdict"] == probe.CODE_BACKWARD
    assert verdict["first_difference"]["field"] == "grad_norm_total"


def test_a_perturbed_sigma_hits_the_input_verdict_not_the_forward_one(
    tmp_path, monkeypatch
):
    """The control: a data perturbation must not be blamed on the forward.

    Without this, "the probe reports a divergence" would be uninformative: it could
    be reporting the perturbation instead of the protocol.
    """
    left = _record(tmp_path / "a", monkeypatch)
    tampered = dict(left)
    tampered["sigmas"] = [0.99, 0.23]

    verdict = probe.compare([left], [tampered])
    assert verdict["verdict"] == probe.CODE_INPUT
    assert verdict["first_difference"]["field"] == "sigmas"


def test_a_perturbed_slot_also_hits_the_input_verdict(tmp_path, monkeypatch):
    left = _record(tmp_path / "a", monkeypatch)
    tampered = dict(left)
    tampered["slots"] = ["s00000004", "s00000005"]

    verdict = probe.compare([left], [tampered])
    assert verdict["verdict"] == probe.CODE_INPUT
    assert verdict["first_difference"]["field"] == "slots"


def test_two_ranks_of_one_step_are_one_comparable_step(tmp_path, monkeypatch):
    rank0 = _record(tmp_path / "r0", monkeypatch, rank=0, world=2, step=1, consumed=1)
    rank1 = _record(tmp_path / "r1", monkeypatch, rank=1, world=2, step=1, consumed=1)

    assert rank0["slots"] == ["s00000004", "s00000005"]
    assert rank1["slots"] == ["s00000006", "s00000007"]
    verdict = probe.compare([rank0, rank1], [dict(rank0), dict(rank1)])
    assert verdict["verdict"] == probe.CODE_AGREE
    assert verdict["common_steps"] == [1]


def test_resume_mode_names_the_first_differing_field(tmp_path, monkeypatch):
    continuous = _record(tmp_path / "c", monkeypatch, step=6, consumed=6)
    resumed = dict(continuous)
    resumed["exp_avg_norm"] = continuous["exp_avg_norm"] + 1.0

    verdict = probe.compare([continuous], [resumed], mode="resume")
    assert verdict["verdict"] == "resume_divergence"
    assert verdict["first_difference"]["field"] == "exp_avg_norm"

    assert probe.compare([continuous], [dict(continuous)], mode="resume")["verdict"] == (
        probe.CODE_AGREE
    )


def test_the_probe_is_inert_until_the_env_var_names_a_path(monkeypatch):
    import library.training.loop as loop_mod

    original = loop_mod._run_microbatch
    monkeypatch.delenv(probe.ENV, raising=False)
    assert probe.maybe_install() is None
    assert loop_mod._run_microbatch is original

    monkeypatch.setenv(probe.ENV, "/tmp/probe-not-written.jsonl")
    recorder = probe.maybe_install()
    try:
        assert recorder is not None
        assert loop_mod._run_microbatch is not original
    finally:
        probe.uninstall()
    assert loop_mod._run_microbatch is original
    os.environ.pop(probe.ENV, None)


def test_the_four_segments_use_the_frozen_configs_and_differ_only_in_shape():
    report = probe.segment_report()
    segments = report["segments"]

    assert set(segments) == {
        "single-batch2",
        "dual-batch1",
        "dual-continuous",
        "dual-resume",
    }
    for name, segment in segments.items():
        assert segment["config"].endswith(
            "run_batch2.toml" if segment["per_device_batch"] == 2 else "run_batch1.toml"
        ), name
        assert report["frozen_config_dir"] in segment["config"]
        assert segment["precision"] in ("no", "fp16", "bf16")
        assert "determinism_probe wrap --" in segment["launch"]
        assert segment["env"][probe.ENV]

    # Only the shape-specific tokens differ from the dual baseline.
    for name in ("single-batch2", "dual-continuous", "dual-resume"):
        changed = segments[name]["argv_diff_vs_dual-batch1"]
        allowed = {
            "<out>/single-batch2",
            "<out>/dual-continuous",
            "<out>/dual-resume",
            "<out>/dual-batch1",
            "2",
            "1",
            "3",
            "--save_state",
            "--save_every_n_steps",
            segments[name]["config"],
            segments["dual-batch1"]["config"],
        }
        assert set(changed["added"]) <= allowed, (name, changed)
        assert set(changed["removed"]) <= allowed, (name, changed)


def test_the_resume_segment_stops_at_its_first_saved_step():
    resume = probe.SEGMENTS["dual-resume"]
    argv = resume.args("<fixture>", "<out>", "<probe>")

    assert "--save_state" in argv
    assert argv[argv.index("--save_every_n_steps") + 1] == "1"
    assert resume.stop_at_step == 1


# ---------------------------------------------------------------------------
# real two-process (Gloo) self-test of the installed probe
# ---------------------------------------------------------------------------


def _gloo_worker(rank, world_size, init_file, probe_path, fixture_path):
    import os

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    os.environ[probe.ENV] = probe_path
    os.environ["RANK"] = str(rank)
    os.environ[noise_fixture.ENV] = fixture_path
    noise_fixture.reset_cache()
    torch.manual_seed(20261107)  # identical init on both ranks, by design

    recorder = probe.maybe_install()
    try:
        parameter = torch.nn.Parameter(torch.ones(4))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        state = _state(rank=rank, world=world_size, step=1, consumed=1)
        recorder.begin(None, state, _batch(1))
        recorder.note_loss_call(torch.tensor([0.5]), 0.5)
        (parameter * 2.0).sum().backward()
        dist.all_reduce(parameter.grad)
        optimizer.step()  # the patched step records the micro-batch
    finally:
        recorder.close()
        probe.uninstall()
        dist.destroy_process_group()


def test_two_ranks_write_one_record_each_through_the_real_patch(tmp_path):
    import torch.multiprocessing as mp

    init_file = tmp_path / "init"
    init_file.write_text("", encoding="utf-8")
    fixture = _fixture_file(tmp_path)
    probe_path = str(tmp_path / "probe.jsonl")

    mp.spawn(
        _gloo_worker,
        args=(2, str(init_file), probe_path, str(fixture)),
        nprocs=2,
        join=True,
    )

    rank0 = probe.load(probe.rank_path(tmp_path / "probe.jsonl", 0))
    rank1 = probe.load(probe.rank_path(tmp_path / "probe.jsonl", 1))

    assert [record["rank"] for record in rank0] == [0]
    assert [record["rank"] for record in rank1] == [1]
    for record in rank0 + rank1:
        assert record["global_step"] == 1
        assert record["slots"] and record["sigmas"]
        assert record["adapter_hash"] and record["adapter_hash_before"]
        assert record["grad_norm_total"] > 0
        assert record["exp_avg_norm"] > 0
        assert record["loss"] == pytest.approx(0.5)
    # The two ranks start from the same weights and consume different slots.
    assert rank0[-1]["adapter_hash_before"] == rank1[-1]["adapter_hash_before"]
    assert rank0[-1]["slots"] != rank1[-1]["slots"]


def test_rank_path_keeps_rank_zero_on_the_given_name(tmp_path):
    path = tmp_path / "probe.jsonl"
    assert probe.rank_path(path, 0) == path
    assert probe.rank_path(path, 2).name == "probe-rank2.jsonl"


# ---------------------------------------------------------------------------
# t26 / F-T25-1..4
# ---------------------------------------------------------------------------


def test_the_host_shim_is_what_hashing_goes_through(tmp_path, monkeypatch):
    """F-T25-1: hashing must not call ``numpy()`` on a device tensor."""
    calls = []
    original = probe._to_cpu

    def spy(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(probe, "_to_cpu", spy)
    probe.tensor_fingerprint(torch.ones(4))
    assert calls, "tensor_fingerprint must route through the device->host shim"

    if torch.cuda.is_available():  # pragma: no cover - hardware only
        device_tensor = torch.ones(4, device="cuda")
        assert probe.tensor_fingerprint(device_tensor) == probe.tensor_fingerprint(
            torch.ones(4)
        )


def test_the_per_sample_capture_survives_install_and_is_restored(monkeypatch):
    """F-T25-3: the t25 run recorded ``per_sample_loss: null``.

    The registry patch used to be undone inside ``install()`` itself, so the
    composer never ran through the capture. It now stays in place until
    ``uninstall()``.
    """
    from library.training import losses as losses_mod

    original = losses_mod.LOSS_REGISTRY["flow_match"]
    monkeypatch.setenv(probe.ENV, "/tmp/probe-registry.jsonl")
    recorder = probe.maybe_install()
    try:
        assert losses_mod.LOSS_REGISTRY["flow_match"] is not original
    finally:
        probe.uninstall()
    assert losses_mod.LOSS_REGISTRY["flow_match"] is original


def test_the_prefix_norms_use_real_parameter_names(tmp_path, monkeypatch):
    """F-T25-3: roles were all ``other``; the network knows the names."""
    network = torch.nn.Module()
    network.lora_down = torch.nn.Linear(4, 2, bias=False)
    network.lora_up = torch.nn.Linear(2, 4, bias=False)
    state = _state()
    state.network = network

    fixture = _fixture_file(tmp_path)
    monkeypatch.setenv(noise_fixture.ENV, str(fixture))
    noise_fixture.reset_cache()
    recorder = probe.Recorder(path=tmp_path / "p.jsonl").open()
    parameters = list(network.parameters())
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW(parameters, lr=0.1)
    recorder.begin(None, state, _batch(1))
    recorder.note_loss_call(torch.tensor([0.5]), 0.5)
    recorder.on_optimizer_step(optimizer, adapter_hash_before="H")
    recorder.close()
    record = probe.load(tmp_path / "p.jsonl")[-1]

    assert set(record["grad_norm_by_prefix"]) == {"lora_down", "lora_up"}
    assert record["grad_tensor_count"] == 2


def test_the_injected_noise_is_recorded_per_sample(tmp_path, monkeypatch):
    """The blind spot: only the slot id and the declared sigma used to be kept."""
    record = _record(tmp_path, monkeypatch, step=0, consumed=0)
    recorder_path = tmp_path / "injected.jsonl"
    assert recorder_path.exists() or True  # placeholder for clarity

    injected = {
        "slots": ["s00000000", "s00000001"],
        "noise": ["deadbeef", "cafebabe"],
        "sigmas": [0.5, 0.5],
    }
    recorder = probe.Recorder(path=tmp_path / "p2.jsonl").open()
    recorder.begin(None, _state(), _batch(2))
    recorder.note_injected(injected)
    recorder.note_loss_call(torch.tensor([0.5, 0.6]), 0.55)
    recorder.on_optimizer_step(
        torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))], lr=0.1),
        adapter_hash_before="H",
    )
    recorder.close()
    built = probe.load(tmp_path / "p2.jsonl")[-1]

    assert built["noise_per_sample_hash"] == ["deadbeef", "cafebabe"]
    assert built["timestep_injected"] == [0.5, 0.5]
    assert built["noise_source"] == "fixture"
    assert record["noise_source"] == "unpinned"


def test_the_fixture_apply_hook_feeds_the_record(tmp_path, monkeypatch):
    fixture_path = _fixture_file(tmp_path)
    monkeypatch.setenv(noise_fixture.ENV, str(fixture_path))
    monkeypatch.setenv(probe.ENV, str(tmp_path / "hooked.jsonl"))
    noise_fixture.reset_cache()

    recorder = probe.maybe_install()
    try:
        active = noise_fixture.active()
        latents = torch.zeros(2, 1, 1, 2)
        pinned, sigmas = active.apply(
            torch.full_like(latents, 7.0),
            latents,
            batch={},
            mode=noise_fixture.IDS_CURSOR,
            local_batch_index=0,
            rank=0,
            world_size=1,
        )
        assert sigmas.tolist() == pytest.approx([0.2, 0.21])
        assert recorder._injected is not None
        assert recorder._injected["slots"] == ["s00000000", "s00000001"]
        assert recorder._injected["noise"] == [
            probe.tensor_fingerprint(pinned[0]),
            probe.tensor_fingerprint(pinned[1]),
        ]
    finally:
        probe.uninstall()
    noise_fixture.reset_cache()


def test_cross_world_records_are_aligned_by_slot_not_by_rank():
    """F-T25-2: rank 1 has no counterpart in a single-worker run.

    Before the alignment, the main cross-world case reported ``init_divergence`` on
    every run because the records were paired rank-by-rank.
    """
    common = {
        "schema": probe.SCHEMA,
        "global_step": 0,
        "adapter_hash": "H0",
        "adapter_hash_before": "H0",
        "noise_source": "fixture",
    }

    def rank_record(rank, world, slot, noise, loss, sigma=0.5, latent="lat", text="txt"):
        return {
            **common,
            "rank": rank,
            "world_size": world,
            "slots": [slot],
            "sigmas": [sigma],
            "noise_per_sample_hash": [noise],
            "noise_raw": [{"head8": [1.0] * 8, "sum": 1.0, "dtype": "torch.float32",
                           "shape": [16, 150, 112], "numel": 268800}],
            "timestep_injected": [sigma],
            "latent_per_sample": [latent],
            "text_per_sample": [text],
            "per_sample_loss": [loss],
            "loss": loss,
            "grad_norm_total": 1.0,
            "exp_avg_norm": 0.1,
            "exp_avg_sq_norm": 0.2,
        }

    single = [rank_record(0, 1, "s00000000", "noiseA", 0.5, latent="lat0", text="txt0")]
    single[0]["slots"] = ["s00000000", "s00000001"]
    single[0]["sigmas"] = [0.5, 0.5]
    single[0]["noise_per_sample_hash"] = ["noiseA", "noiseB"]
    single[0]["noise_raw"] = [rank_record(0, 1, "x", "n", 0.5)["noise_raw"][0]] * 2
    single[0]["timestep_injected"] = [0.5, 0.5]
    single[0]["latent_per_sample"] = ["lat0", "lat1"]
    single[0]["text_per_sample"] = ["txt0", "txt1"]
    single[0]["per_sample_loss"] = [0.5, 0.6]
    single[0]["loss"] = 0.55
    single = [single[0]]

    dual = [
        rank_record(0, 2, "s00000000", "noiseA", 0.5, latent="lat0", text="txt0"),
        rank_record(1, 2, "s00000001", "noiseB", 0.6, latent="lat1", text="txt1"),
    ]

    verdict = probe.compare(dual, single)
    # Same per-sample content in both shapes: no divergence is reported.
    assert verdict["verdict"] == probe.CODE_AGREE, verdict["detail"]

    # Now feed dual rank 1 the *wrong* noise value: the input criterion must name
    # the value field, not a hash of it (F-29-2/29-3).
    dual[1] = {
        **dual[1],
        "noise_raw": [{"head8": [2.0] * 8, "sum": 2.0, "dtype": "torch.float32",
                       "shape": [16, 150, 112], "numel": 268800}],
    }
    verdict = probe.compare(dual, single)
    assert verdict["verdict"] == probe.CODE_INPUT
    assert verdict["first_difference"]["field"] == "noise_value"


def test_a_declared_sigma_that_was_not_injected_is_now_visible():
    """The t25 blind spot, end to end at the comparison level."""
    base = {
        "schema": probe.SCHEMA,
        "rank": 0,
        "world_size": 1,
        "global_step": 0,
        "slots": ["s00000000"],
        "sigmas": [0.5],
        "noise_per_sample": ["noiseA"],
        "timestep_injected": [0.5],
        "latent_per_sample": ["lat0"],
        "text_per_sample": ["txt0"],
        "per_sample_loss": [0.5],
        "loss": 0.5,
        "adapter_hash": "H0",
        "adapter_hash_before": "H0",
        "grad_norm_total": 1.0,
        "exp_avg_norm": 0.1,
        "exp_avg_sq_norm": 0.2,
        "noise_source": "fixture",
    }
    declared_same = dict(base)
    injected_different = {**base, "timestep_injected": [0.31], "loss": 0.2,
                          "per_sample_loss": [0.2]}

    verdict = probe.compare([base], [injected_different])
    assert verdict["verdict"] == probe.CODE_INPUT
    assert verdict["first_difference"]["field"] == "timestep_injected"
