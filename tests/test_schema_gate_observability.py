"""C-R4/C-R5: schema gate nested boundary + load-failure observability."""

from __future__ import annotations

from library.config import schema as config_schema
from web.services.config import schema_gate


def test_schema_load_failure_is_observable(monkeypatch):
    """When populate fails, validation must not silently pretend schema is healthy."""
    monkeypatch.setattr(config_schema, "CONFIG_SCHEMA", {})
    monkeypatch.setattr(config_schema, "get_schema", lambda: config_schema.CONFIG_SCHEMA)

    def _broken_inputs():
        raise ImportError("schema parser unavailable for test")

    monkeypatch.setattr(schema_gate, "_build_schema_inputs", _broken_inputs)
    schema_gate.reset_schema_load_state_for_tests()

    status = schema_gate.get_schema_load_status()
    assert status["loaded"] is False
    assert status["key_count"] == 0

    errors, warnings = schema_gate.validate_patch_values({"network_dim": 16})
    assert errors == []
    assert any("schema" in w.lower() or "unavailable" in w.lower() for w in warnings)

    status_after = schema_gate.get_schema_load_status()
    assert status_after["ok"] is False
    assert status_after["error"]
    assert "unavailable" in status_after["error"].lower()


def test_ensure_schema_populated_does_not_import_train() -> None:
    import subprocess
    import sys

    code = (
        "import sys; "
        "from web.services.config import schema_gate; "
        "schema = schema_gate.ensure_schema_populated(); "
        "assert 'network_dim' in schema and 'use_lokr' in schema; "
        "assert 'train' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_nested_dict_and_list_values_are_not_false_errors(monkeypatch):
    """Nested tables/lists are intentionally out of top-level schema gate scope."""

    class _Spec:
        choices = None

    fake_schema = {
        "network_dim": _Spec(),
        "mixed_precision": type("S", (), {"choices": ["no", "fp16", "bf16"]})(),
    }
    monkeypatch.setattr(config_schema, "get_schema", lambda: fake_schema)
    monkeypatch.setattr(config_schema, "resolve_alias", lambda key: key)
    monkeypatch.setattr(
        config_schema,
        "_coerce_value",
        lambda spec, value: value,
    )
    monkeypatch.setattr(schema_gate, "ensure_schema_populated", lambda: fake_schema)
    schema_gate.reset_schema_load_state_for_tests()
    schema_gate._SCHEMA_LOAD_OK = True
    schema_gate._SCHEMA_LOAD_ERROR = ""
    schema_gate._SCHEMA_KEY_COUNT = len(fake_schema)

    errors, warnings = schema_gate.validate_config_mapping(
        {
            "network_dim": 32,
            "datasets": [{"image_dir": "/tmp"}],
            "network": {"dim": 8},
            "mixed_precision": "fp16",
            "sample_prompts": ["a", "b"],
        }
    )
    assert errors == []
    assert warnings == []


def test_nested_tables_do_not_validate_inner_unknown_keys_as_top_level(monkeypatch):
    class _Spec:
        choices = None

    fake_schema = {"network_dim": _Spec()}
    monkeypatch.setattr(config_schema, "get_schema", lambda: fake_schema)
    monkeypatch.setattr(config_schema, "resolve_alias", lambda key: key)
    monkeypatch.setattr(config_schema, "_coerce_value", lambda spec, value: value)
    monkeypatch.setattr(schema_gate, "ensure_schema_populated", lambda: fake_schema)
    schema_gate.reset_schema_load_state_for_tests()
    schema_gate._SCHEMA_LOAD_OK = True
    schema_gate._SCHEMA_LOAD_ERROR = ""
    schema_gate._SCHEMA_KEY_COUNT = 1

    errors, warnings = schema_gate.validate_config_mapping(
        {
            "network": {"totally_unknown_nested": 1},
            "network_dim": 8,
        }
    )
    assert errors == []
    # Nested unknown keys stay nested; top-level gate must not invent warnings for them.
    assert warnings == []
