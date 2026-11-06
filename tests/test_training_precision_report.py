from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from library.training import precision_report as report


def args_for(path, **kwargs):
    return SimpleNamespace(output_dir=str(path), adaptive_precision="off", mixed_precision="no", **kwargs)


def read(path):
    return json.loads((path / "precision_report.json").read_text())


@pytest.mark.parametrize("recovery,scope", [(False, "startup_only"), (True, "committed_resume")])
def test_supervisor_reports_actual_recovery_scope(tmp_path, recovery, scope):
    observed = report.PrecisionReport(args_for(tmp_path, adaptive_oom_retry_resume=recovery))
    report.report_supervisor(observed, {"status": "ok", "attempts": []}, tmp_path / "summary.json")
    assert observed.data["supervisor"]["native_scope"] == scope


@pytest.fixture(autouse=True)
def cpu_rank(monkeypatch):
    monkeypatch.setattr(report, "_main_rank", lambda: True)


@pytest.mark.parametrize("stage,error,status", [
    ("initialization", ValueError("contract"), "rejected"),
    ("model_load", RuntimeError("load"), "failed"),
    ("final_save", OSError("save"), "failed"),
    ("forward", KeyboardInterrupt(), "cancelled"),
])
def test_lifecycle_preserves_original_exception(tmp_path, stage, error, status):
    args = args_for(tmp_path)

    def body(trainer, args):
        report.notify(stage)
        raise error

    with pytest.raises(type(error)) as captured:
        report.reported_session(object(), args, body)
    assert captured.value is error
    data = read(tmp_path)
    assert data["status"] == status and data["end_time"]
    assert data["exception"]["type"] == type(error).__name__
    assert report._current.get() is None
    assert all(not isinstance(value, report.PrecisionReport) for value in vars(args).values())


def test_completed_off_and_repeated_runs_never_overwrite(tmp_path):
    args = args_for(tmp_path)
    def complete(*unused):
        report.notify("finish")
        return 42

    assert report.reported_session(object(), args, complete) == 42
    old = (tmp_path / "precision_report.json").read_bytes()
    report.reported_session(object(), args, complete)
    assert (tmp_path / "precision_report.json").read_bytes() == old
    unique = list(tmp_path.glob("precision_report.*.json"))
    assert len(unique) == 1
    assert json.loads(unique[0].read_text())["status"] == "completed"
    data = read(tmp_path)
    assert data["precision"]["requested"] == "off"
    assert data["updates"]["attempted"] == 0
    assert data["resources"]["process_rss_max_bytes"] > 0
    assert data["validation_scope"]["current_run_quality_certified"] is False
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("event,status", [
    ("no_data", "rejected"), ("network_not_created", "rejected"),
    ("debug_dataset", "debug_completed"), ("setup", "incomplete"),
])
def test_early_return_cannot_claim_training_completed(tmp_path, event, status):
    report.reported_session(object(), args_for(tmp_path), lambda *a: report.notify(event))
    data = read(tmp_path)
    assert data["status"] == status
    assert data["termination_reason"] == event
    assert data["updates"]["effective"] == 0


def test_existing_user_report_is_preserved(tmp_path):
    (tmp_path / "precision_report.json").write_text("user-owned")
    report.reported_session(object(), args_for(tmp_path), lambda *a: None)
    assert (tmp_path / "precision_report.json").read_text() == "user-owned"
    assert len(list(tmp_path.glob("precision_report.*.json"))) == 1


def test_non_main_rank_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_main_rank", lambda: False)
    report.reported_session(object(), args_for(tmp_path), lambda *a: None)
    assert not list(tmp_path.iterdir())


def test_rank_detection_uses_global_rank(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setenv("RANK", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    assert report._main_rank() is False


def test_atomic_reader_only_sees_complete_json(tmp_path, monkeypatch):
    original_link, original_replace = report.os.link, report.os.replace
    seen = []

    def link(source, target):
        seen.append(json.loads(open(source).read())["status"])
        original_link(source, target)

    def replace(source, target):
        seen.append(json.loads(open(source).read())["status"])
        assert json.loads(open(target).read())["status"] == "running"
        original_replace(source, target)

    monkeypatch.setattr(report.os, "link", link)
    monkeypatch.setattr(report.os, "replace", replace)
    report.reported_session(object(), args_for(tmp_path), lambda *a: report.notify("finish"))
    assert seen == ["running", "running", "completed"]


def fake_accelerator(scale):
    return SimpleNamespace(is_main_process=True, device="cpu", native_amp=True,
                           mixed_precision="fp16", scaler=SimpleNamespace(get_scale=lambda: scale[0]),
                           unwrap_model=lambda model: model)


def test_actual_dtypes_guards_resume_scale_and_inherited_step(tmp_path):
    args = args_for(tmp_path, resume="checkpoint")
    args.adaptive_precision = "auto"
    scale = [1024.0]
    accelerator = fake_accelerator(scale)
    model = torch.nn.Linear(2, 2).half()
    model.blocks = [SimpleNamespace(fp32_residual=True), SimpleNamespace(fp32_residual=False)]
    model.final_layer = SimpleNamespace(fp32_residual=True)
    network = torch.nn.Linear(2, 2)

    def body(trainer, args):
        args.adaptive_resolved_mode = "native_fp16"
        args.mixed_precision = "fp16"
        args.adaptive_resolution_reason = "test hardware choice"
        report.observe("accelerator_ready", accelerator)
        report.observe("models_prepared", model, network)
        scale[0] = 256.0
        report.observe("resumed", 17)
        state = SimpleNamespace(global_step=19, accelerator=accelerator)
        report.notify("final_save", state)
        scale[0] = 128.0

    report.reported_session(object(), args, body)
    data = read(tmp_path)
    assert data["precision"] == {"requested": "auto", "resolved": "native_fp16", "mixed_precision": "fp16",
                                  "selection_reason": "test hardware choice"}
    assert data["actual_parameters"]["model"]["torch.float16"]["elements"] == 6
    assert data["actual_parameters"]["trainable_adapter"]["torch.float32"]["elements"] == 6
    assert data["fp32_guards"]["blocks_enabled"] == 1
    assert data["fp32_guards"]["final_layer_enabled"] is True
    assert data["amp"]["scaler_initial"] == 1024.0
    assert data["amp"]["scaler_resumed"] == 256.0
    assert data["amp"]["scaler_final"] == 128.0
    assert data["resume"] == {"requested": True, "inherited_global_step": 17}
    assert data["updates"]["total_global_step"] == 19
    assert scale == [128.0]


def test_report_failure_warns_and_never_masks_training_error(tmp_path, monkeypatch, caplog):
    error = RuntimeError("original training failure")
    monkeypatch.setattr(report.PrecisionReport, "write", lambda *a: (_ for _ in ()).throw(OSError("read-only")))
    with pytest.raises(RuntimeError) as caught:
        report.reported_session(object(), args_for(tmp_path), lambda *a: (_ for _ in ()).throw(error))
    assert caught.value is error
    assert "precision report" in caplog.text and "read-only" in caplog.text
    assert report.reported_session(object(), args_for(tmp_path), lambda *a: 7) == 7


def test_unknown_output_does_not_invent_path(caplog):
    report.report_entry_rejection(SimpleNamespace(), ValueError("reject"))
    assert "actual output_dir is unknown" in caplog.text


@pytest.mark.parametrize("scale", [float("nan"), float("inf")])
def test_invalid_nonfinite_configuration_still_has_rejection_report(tmp_path, scale):
    report.report_entry_rejection(args_for(tmp_path, adaptive_loss_scale=scale), ValueError("invalid scale"))
    data = read(tmp_path)
    assert data["status"] == "rejected"
    assert data["contract"]["adaptive_loss_scale"] == str(scale)


def loop_fixture(monkeypatch):
    from library.training import loop
    from library.training.auto_block_swap import online

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accelerator = SimpleNamespace(sync_gradients=False, optimizer_step_was_skipped=False,
                                  accumulate=lambda *a: nullcontext(), unwrap_model=lambda model: model,
                                  backward=lambda loss: loss.backward())
    trainer = SimpleNamespace(_cudagraph_mark_step=False, _adapters=[],
                              _state=SimpleNamespace(personalization_observer={}),
                              on_step_start=lambda *a, **k: None, run_after_backward=lambda *a: None,
                              process_batch=lambda *a, **k: model(torch.ones(1, 2)).square().mean())
    state = SimpleNamespace(args=SimpleNamespace(max_grad_norm=0.0), accelerator=accelerator,
                            global_step=10, network=model, training_model=model, optimizer=optimizer,
                            lr_scheduler=SimpleNamespace(step=lambda: None), train_ctx=None,
                            on_step_start_for_network=lambda *a: None, text_encoder=None, unet=None,
                            profile_started=False, is_tracking=False)
    monkeypatch.setattr(online, "before_microstep", lambda *a: None)
    monkeypatch.setattr(online, "after_microstep", lambda *a: None)
    monkeypatch.setattr(loop, "synchronize_optimizer_gradients", lambda *a: None)
    return loop, trainer, state


@pytest.mark.parametrize("failure", [False, True])
def test_real_loop_accumulation_skips_and_optimizer_exception(tmp_path, monkeypatch, failure):
    loop, trainer, state = loop_fixture(monkeypatch)

    def body(trainer, args):
        report.observe("resumed", 10)
        loop._run_step(trainer, state, object())  # accumulation microstep
        state.accelerator.sync_gradients = True
        if failure:
            monkeypatch.setattr(state.optimizer, "step", lambda: (_ for _ in ()).throw(RuntimeError("update")))
        loop._run_step(trainer, state, object())
        state.global_step += 1
        state.accelerator.optimizer_step_was_skipped = True
        loop._run_step(trainer, state, object())
        state.global_step += 1

    if failure:
        with pytest.raises(RuntimeError, match="update"):
            report.reported_session(trainer, args_for(tmp_path), body)
    else:
        report.reported_session(trainer, args_for(tmp_path), body)
    updates = read(tmp_path)["updates"]
    # E3 语义（2026-11-06 用户裁定）：没有任何进度副作用的边界不提交，因此
    # ``optimizer_step_was_skipped`` 的第三次调用不再记为 attempted/skipped；
    # 只有真正应用了更新的边界进入账目（失败用例里 optimizer.step 抛错，同样不提交）。
    # 丢窗/跳步的度量改由窗口账本（window_discards / window_replays）承载。
    assert (updates["attempted"], updates["effective"], updates["skipped"]) == (
        (0, 0, 0) if failure else (1, 1, 0))
    assert updates["total_global_step"] == (10 if failure else 12)


def test_public_session_wrapper_covers_entry_rejection(tmp_path, monkeypatch):
    from library.training import train_session

    error = ValueError("native contract")
    monkeypatch.setattr(train_session, "_run_training_session", lambda *a: (_ for _ in ()).throw(error))
    with pytest.raises(ValueError):
        train_session.run_training_session(object(), args_for(tmp_path))
    assert read(tmp_path)["status"] == "rejected"


def test_failed_native_worker_preserves_child_report_path_for_parent(tmp_path):
    runtime = SimpleNamespace(precision_report_path=None)
    trainer = SimpleNamespace(_adaptive_training_runtime=runtime)
    error = RuntimeError("model load failed")
    def body(*args):
        report.notify("model_load")
        raise error

    with pytest.raises(RuntimeError) as caught:
        report.reported_session(trainer, args_for(tmp_path), body)
    assert caught.value is error
    assert runtime.precision_report_path == str(tmp_path / "precision_report.json")
    parent = report.PrecisionReport(args_for(tmp_path))
    report.report_supervisor(parent, {"status": "failed", "attempts": [{"attempt": 0,
                            "result": {"precision_report_path": runtime.precision_report_path}}]},
                            tmp_path / ".adaptive-recovery" / "summary.json")
    assert parent.data["supervisor"]["child_reports"] == [runtime.precision_report_path]


def test_mock_cuda_capability_once_peaks_do_not_reset_or_synchronize(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: (
        calls.append(str(device)) or SimpleNamespace(name="fake SM75", major=7, minor=5, total_memory=12345)))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 456)
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: pytest.fail("reporter must not synchronize or reset"))
    accelerator = fake_accelerator([1024.0])
    accelerator.device = "cuda:0"

    def body(*args):
        report.observe("accelerator_ready", accelerator)
        report.notify("model_load")
        report.notify("final_save")

    report.reported_session(object(), args_for(tmp_path), body)
    data = read(tmp_path)
    assert calls == ["cuda:0"]
    assert data["gpu"]["compute_capability"] == [7, 5]
    assert data["resources"]["cuda_peak_allocated_bytes"] == 123
    assert data["resources"]["cuda_peak_reserved_bytes"] == 456
    assert "external reset" in data["resources"]["cuda_peak_scope"]


def test_real_cli_precision_rejection_uses_final_artist_destination(tmp_path, monkeypatch):
    from library.training import cli_entry
    from library.training.adaptive_runtime import precision

    monkeypatch.chdir(tmp_path)
    args = args_for(tmp_path / "unused", artist_filter="@artist")
    parser = SimpleNamespace(parse_args=lambda *a: args)
    monkeypatch.setattr(cli_entry._config_schema, "populate_schema", lambda *a, **k: None)
    monkeypatch.setattr(cli_entry, "verify_command_line_training_args", lambda *a: None)
    monkeypatch.setattr(cli_entry, "read_config_from_file", lambda *a: args)
    error = ValueError("precision decision rejected")
    monkeypatch.setattr(precision, "resolve_adaptive_precision", lambda *a: (_ for _ in ()).throw(error))
    with pytest.raises(ValueError) as caught:
        cli_entry.run_training_cli(setup_parser=lambda: parser, trainer_factory=lambda: pytest.fail("trainer created"),
                                   install_stop_signal_handlers=lambda: None, install_crash_reporter=lambda *a: None,
                                   argv=["train.py"])
    assert caught.value is error
    assert read(tmp_path / "output/ckpt-artist")["status"] == "rejected"
    assert not (tmp_path / "unused").exists()


def test_cli_supervisor_rejection_preserves_error_and_report(tmp_path, monkeypatch):
    from library.training import cli_entry
    from library.training.adaptive_runtime import supervisor

    args = args_for(tmp_path, adaptive_oom_retry=True)
    args.adaptive_precision = "native_fp16"
    parser = SimpleNamespace(parse_args=lambda *a: args)
    monkeypatch.setattr(cli_entry._config_schema, "populate_schema", lambda *a, **k: None)
    monkeypatch.setattr(cli_entry, "verify_command_line_training_args", lambda *a: None)
    monkeypatch.setattr(cli_entry, "read_config_from_file", lambda *a: args)
    error = ValueError("native OOM retry is unsupported")
    monkeypatch.setattr(supervisor, "run_supervised", lambda *a: (_ for _ in ()).throw(error))
    with pytest.raises(ValueError) as caught:
        cli_entry.run_training_cli(setup_parser=lambda: parser, trainer_factory=lambda: pytest.fail("trainer created"),
                                   install_stop_signal_handlers=lambda: None, install_crash_reporter=lambda *a: None,
                                   argv=["train.py"])
    assert caught.value is error
    assert read(tmp_path)["status"] == "rejected"
