"""Start / preprocess launch wiring tests split from test_training_resume.py."""

from __future__ import annotations

from tests.nf4_preflight_test_support import write_ordinary_checkpoint

from tests import training_resume_test_support as _training_resume_support

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

# Split: start_preprocess

def test_handle_start_converts_plain_config_to_preprocess_train_after(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    image_a = tmp_path / "image_dataset" / "a"
    image_b = tmp_path / "image_dataset" / "b"
    image_a.mkdir(parents=True)
    image_b.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_a / "a.png")
    Image.new("RGB", (8, 8), color=(30, 20, 10)).save(image_b / "b.png")
    (tmp_path / "models").mkdir()
    write_ordinary_checkpoint(tmp_path / "models" / "anima.safetensors")
    (tmp_path / "models" / "qwen.safetensors").write_bytes(b"qwen")
    (tmp_path / "models" / "vae.safetensors").write_bytes(b"vae")
    (tmp_path / "configs" / "base.toml").write_text(
        "\n".join(
            [
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
                'dataset_config = "configs/datasets/522.toml"',
                'source_image_dir = "image_dataset/a"',
                'max_train_steps = 100',
            ]
        ),
        encoding="utf-8",
    )

    class FakeService:
        def __init__(self):
            self.preprocess_calls = []
            self.start_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

        async def start(self, *args, **kwargs):
            self.start_calls.append((args, kwargs))

    svc = FakeService()
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "configs/imported/522.toml",
            "extra_args": ["--seed", "0"],
            "gpu_whitelist": [0],
            "confirmed": True,
            "confirm_preprocess": True,
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["job"] == "preprocess"
    assert payload["train_after"] is True
    assert "自动开始训练" in payload["message"]
    assert len(svc.preprocess_calls) == 1
    assert svc.start_calls == []
    args, kwargs = svc.preprocess_calls[0]
    assert args[:5] == ("522", "default", "imported", ["--seed", "0"], True)
    assert kwargs["config_file"] == "configs/imported/522.toml"
    assert kwargs["gpu_whitelist"] == [0]

def test_handle_start_requires_explicit_confirmation_before_preprocess_train_after(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    image_a = tmp_path / "image_dataset" / "a"
    image_b = tmp_path / "image_dataset" / "b"
    image_a.mkdir(parents=True)
    image_b.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_a / "a.png")
    Image.new("RGB", (8, 8), color=(30, 20, 10)).save(image_b / "b.png")
    (tmp_path / "models").mkdir()
    write_ordinary_checkpoint(tmp_path / "models" / "anima.safetensors")
    (tmp_path / "models" / "qwen.safetensors").write_bytes(b"qwen")
    (tmp_path / "models" / "vae.safetensors").write_bytes(b"vae")
    (tmp_path / "configs" / "base.toml").write_text(
        "\n".join(
            [
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
                'dataset_config = "configs/datasets/522.toml"',
                'source_image_dir = "image_dataset/a"',
                'max_train_steps = 100',
            ]
        ),
        encoding="utf-8",
    )

    class FakeService:
        def __init__(self):
            self.preprocess_calls = []
            self.start_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

        async def start(self, *args, **kwargs):
            self.start_calls.append((args, kwargs))

    svc = FakeService()
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "configs/imported/522.toml",
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 409
    payload = json.loads(response.text)
    assert payload["requires_confirmation"] is True
    assert payload["requires_preprocess_confirmation"] is True
    assert payload["preflight"]["ok"] is True
    assert svc.preprocess_calls == []
    assert svc.start_calls == []

def test_start_preprocess_preserves_extra_args_for_pending_training(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        training_service,
        "_prepare_web_runtime_config",
        lambda *args, **kwargs: {
            "runtime_config_file": "output/runs/522-20260523-114514/config.runtime.toml",
            "output_dir": "output/runs/522-20260523-114514/training_output",
            "sample_dir": "output/runs/522-20260523-114514/training_output/sample",
            "sample_config": {},
            "data_dirs": {},
            "run_dir": "output/runs/522-20260523-114514",
        },
    )

    svc = TrainingService(web.Application())

    async def fake_launch(*args, **kwargs):
        return None

    svc._launch_job = fake_launch
    asyncio.run(
        svc.start_preprocess(
            "522",
            "default",
            "imported",
            ["--sample_every_n_steps", "5"],
            train_after=True,
        )
    )

    assert svc._pending_train_after_preprocess["extra_args"] == ["--sample_every_n_steps", "5"]

def test_start_training_appends_network_weights_and_history_meta(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", tmp_path / "history")
    weight = _write_continue_lora_weight(tmp_path / "weights" / "demo.safetensors", kind="LoRA")

    captured = {}
    svc = TrainingService(web.Application())

    async def fake_launch(cmd, env, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        svc.current_task_dir = tmp_path / "history" / "fake-task"
        svc.current_task_dir.mkdir(parents=True)
        svc.current_task_id = "fake-task"
        history_kwargs = {
            key: kwargs[key]
            for key in (
                "job",
                "variant",
                "preset",
                "methods_subdir",
                "output_dir",
                "sample_dir",
                "data_dirs",
                "sample_config",
                "config_file",
                "resume_info",
                "continue_info",
                "gpu_whitelist",
                "runtime_info",
            )
            if key in kwargs
        }
        svc._start_history_task(command=cmd, **history_kwargs)

    svc._launch_job = fake_launch
    asyncio.run(
        svc.start(
            "lora",
            "default",
            [],
            "gui-methods",
            config_file="configs/gui-methods/lora.toml",
            use_runtime_dir=False,
            continue_info={"continue_from_weight_abs_path": str(weight)},
        )
    )

    meta = json.loads((tmp_path / "history" / "fake-task" / "meta.json").read_text(encoding="utf-8"))
    assert captured["cmd"][1] == str(tmp_path / "train.py")
    assert "accelerate.commands.accelerate_cli" not in captured["cmd"]
    assert "--network_weights" in captured["cmd"]
    assert str(weight.resolve()) in captured["cmd"]
    assert "--dim_from_weights" in captured["cmd"]
    assert meta["training_mode"] == "continue_lora"
    assert meta["continue_from_weight_abs_path"] == str(weight.resolve())
    assert meta["continue_from_weight_name"] == "demo.safetensors"
    assert meta["continue_from_weight_kind"] == "LoRA"
    snapshot = (tmp_path / "history" / "fake-task" / "config.snapshot.toml").read_text(encoding="utf-8")
    assert '# training_mode = "continue_lora"' in snapshot
    path_line = next(
        line
        for line in snapshot.splitlines()
        if line.startswith("# continue_from_weight_abs_path = ")
    )
    path_literal = path_line.split("=", 1)[1].strip()
    assert toml.loads(f"value = {path_literal}")["value"] == str(weight.resolve())

def test_start_training_aligns_accelerate_mixed_precision_from_config(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    monkeypatch.setenv(ACCELERATE_LAUNCH_ENV, "1")
    monkeypatch.setenv("ANIMA_ACCELERATE_NUM_PROCESSES", "2")
    runtime_config = tmp_path / "configs" / "imported" / "522.toml"
    runtime_config.write_text(
        runtime_config.read_text(encoding="utf-8") + '\nmixed_precision = "fp16"\n',
        encoding="utf-8",
    )
    captured = {}
    svc = TrainingService(web.Application())

    async def fake_launch(cmd, env, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["kwargs"] = kwargs

    svc._launch_job = fake_launch

    asyncio.run(
        svc.start(
            "522",
            "default",
            [],
            "imported",
            config_file="configs/imported/522.toml",
            use_runtime_dir=False,
        )
    )

    assert captured["env"][ACCELERATE_MIXED_PRECISION_ENV] == "fp16"
    assert captured["cmd"][captured["cmd"].index("--mixed_precision") + 1] == "fp16"

def test_start_training_extra_args_override_accelerate_mixed_precision(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    monkeypatch.setenv(ACCELERATE_LAUNCH_ENV, "1")
    monkeypatch.setenv("ANIMA_ACCELERATE_NUM_PROCESSES", "2")
    runtime_config = tmp_path / "configs" / "imported" / "522.toml"
    runtime_config.write_text(
        runtime_config.read_text(encoding="utf-8") + '\nmixed_precision = "bf16"\n',
        encoding="utf-8",
    )
    captured = {}
    svc = TrainingService(web.Application())

    async def fake_launch(cmd, env, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["kwargs"] = kwargs

    svc._launch_job = fake_launch

    asyncio.run(
        svc.start(
            "522",
            "default",
            ["--mixed_precision", "no"],
            "imported",
            config_file="configs/imported/522.toml",
            use_runtime_dir=False,
        )
    )

    assert captured["env"][ACCELERATE_MIXED_PRECISION_ENV] == "no"
    assert captured["cmd"][captured["cmd"].index("--mixed_precision") + 1] == "no"

def test_start_preprocess_keeps_continue_info_for_pending_training(tmp_path, monkeypatch):
    monkeypatch.delenv("ANIMA_ACCELERATE_LAUNCH", raising=False)
    monkeypatch.delenv("ANIMA_ACCELERATE_NUM_PROCESSES", raising=False)
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    runtime_config = tmp_path / "output" / "runs" / "522-20260523-114514" / "config.runtime.toml"
    runtime_config.parent.mkdir(parents=True)
    runtime_config.write_text('network_module = "networks.lora_anima"\n', encoding="utf-8")
    weight = _write_continue_lora_weight(tmp_path / "weights" / "demo.safetensors", kind="LoRA")
    monkeypatch.setattr(
        training_service,
        "_prepare_web_runtime_config",
        lambda *args, **kwargs: {
            "runtime_config_file": str(runtime_config),
            "output_dir": "output/runs/522-20260523-114514/training_output",
            "sample_dir": "output/runs/522-20260523-114514/training_output/sample",
            "sample_config": {},
            "data_dirs": {},
            "run_dir": "output/runs/522-20260523-114514",
        },
    )

    svc = TrainingService(web.Application())

    captured = {}

    async def fake_launch(cmd, env, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = env
        return None

    svc._launch_job = fake_launch
    asyncio.run(
        svc.start_preprocess(
            "lora",
            "default",
            "gui-methods",
            train_after=True,
            gpu_whitelist=[0, 1],
            continue_info={"continue_from_weight_abs_path": str(weight)},
        )
    )

    assert svc._pending_train_after_preprocess["continue_info"]["continue_from_weight_abs_path"] == str(weight.resolve())
    assert svc._pending_train_after_preprocess["gpu_whitelist"] == [0, 1]
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "accelerate.commands.accelerate_cli" not in captured["cmd"]

def test_handle_start_returns_400_for_missing_config_file(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    class FakeService:
        async def start_preprocess(self, *args, **kwargs):
            raise AssertionError("不应启动预处理")

        async def start(self, *args, **kwargs):
            raise AssertionError("不应启动训练")

    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "configs/imported/missing.toml",
        },
        {"training_service": FakeService()},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 400

def test_handle_start_blocks_preprocess_environment_error(monkeypatch):
    class FakeService:
        def __init__(self):
            self.preprocess_calls = []
            self.start_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

        async def start(self, *args, **kwargs):
            self.start_calls.append((args, kwargs))

    failure = {
        "ok": False,
        "summary": {"errors": 1, "warnings": 0, "checks": 1},
        "checks": [{
            "level": "error",
            "key": "preprocess_environment",
            "message": "预处理启动环境异常: ModuleNotFoundError",
        }],
        "errors": [{
            "level": "error",
            "key": "preprocess_environment",
            "message": "预处理启动环境异常: ModuleNotFoundError",
        }],
        "warnings": [],
    }
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: failure)
    svc = FakeService()
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "configs/imported/522.toml",
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 400
    payload = json.loads(response.text)
    assert payload["preflight"]["errors"][0]["key"] == "preprocess_environment"
    assert svc.preprocess_calls == []
    assert svc.start_calls == []

def test_handle_preprocess_blocks_preprocess_environment_error(monkeypatch):
    class FakeService:
        def __init__(self):
            self.preprocess_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

    failure = {
        "ok": False,
        "summary": {"errors": 1, "warnings": 0, "checks": 2},
        "checks": [
            {"level": "ok", "key": "source_image_dir", "message": "源图像目录 存在"},
            {
                "level": "error",
                "key": "preprocess_environment",
                "message": "预处理启动环境异常: ModuleNotFoundError",
            },
        ],
        "errors": [{
            "level": "error",
            "key": "preprocess_environment",
            "message": "预处理启动环境异常: ModuleNotFoundError",
        }],
        "warnings": [],
    }
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: failure)
    svc = FakeService()
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "configs/imported/522.toml",
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_preprocess(req))

    assert response.status == 400
    payload = json.loads(response.text)
    assert payload["preflight"]["errors"][0]["key"] == "preprocess_environment"
    assert svc.preprocess_calls == []

def test_handle_preprocess_requires_confirmation_before_train_after(monkeypatch):
    class FakeService:
        def __init__(self):
            self.preprocess_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

    preflight = {
        "ok": True,
        "summary": {"errors": 0, "warnings": 0, "checks": 1},
        "checks": [{"level": "ok", "key": "source_image_dir", "message": "源图像目录存在"}],
        "errors": [],
        "warnings": [],
    }
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: preflight)
    svc = FakeService()
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "configs/imported/522.toml",
            "train_after": True,
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_preprocess(req))

    assert response.status == 409
    payload = json.loads(response.text)
    assert payload["requires_confirmation"] is True
    assert payload["requires_train_after_confirmation"] is True
    assert svc.preprocess_calls == []

def test_handle_preprocess_allows_confirmed_train_after(monkeypatch):
    class FakeService:
        def __init__(self):
            self.preprocess_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

    preflight = {
        "ok": True,
        "summary": {"errors": 0, "warnings": 0, "checks": 1},
        "checks": [{"level": "ok", "key": "source_image_dir", "message": "源图像目录存在"}],
        "errors": [],
        "warnings": [],
    }
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: preflight)
    svc = FakeService()
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "configs/imported/522.toml",
            "extra_args": ["--foo"],
            "gpu_whitelist": [0],
            "train_after": True,
            "confirmed": True,
            "confirm_train_after": True,
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_preprocess(req))

    assert response.status == 200
    payload = json.loads(response.text)
    assert "自动开始训练" in payload["message"]
    assert len(svc.preprocess_calls) == 1
    args, kwargs = svc.preprocess_calls[0]
    assert args[:5] == ("522", "default", "imported", ["--foo"], True)
    assert kwargs["config_file"] == "configs/imported/522.toml"
    assert kwargs["gpu_whitelist"] == [0]

def test_handle_start_blocks_spd_cli_only_variant(monkeypatch):
    class FakeService:
        async def start_preprocess(self, *args, **kwargs):
            raise AssertionError("不应启动预处理")

        async def start(self, *args, **kwargs):
            raise AssertionError("不应启动训练")

    monkeypatch.setattr(
        training_routes,
        "preflight_training_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应预检测 SPD")),
    )
    req = _FakeJsonRequest(
        {
            "variant": "spd",
            "preset": "default",
            "methods_subdir": "methods",
            "config_file": "configs/methods/spd.toml",
        },
        {"training_service": FakeService()},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 400
    payload = json.loads(response.text)
    assert payload["ok"] is False
    assert "CLI" in payload["error"]
