"""Schema gate helpers for WebUI raw/preflight config validation."""

from __future__ import annotations

import logging
from typing import Any

from library.config import schema as config_schema

LOGGER = logging.getLogger("web.services.config.schema_gate")

# Observability for C-R5: schema load must not be a silent no-op.
_SCHEMA_LOAD_ATTEMPTED = False
_SCHEMA_LOAD_OK = False
_SCHEMA_LOAD_ERROR = ""
_SCHEMA_KEY_COUNT = 0

# Keys whose empty/None patch value is normalized to "delete the TOML key"
# before the raw-file write. They must pass the choice gate even though an
# empty string is not a legal enum value for the trainer (raw_files
# converts them to deletion).
_DELETE_ON_EMPTY_PATCH_KEYS = frozenset(
    {
        "sample_every_n_epochs",
        "sample_every_n_steps",
        "max_train_epochs",
        "convrot_large_layer_mode",
    }
)


def reset_schema_load_state_for_tests() -> None:
    """Reset module-level schema load telemetry (tests only)."""
    global _SCHEMA_LOAD_ATTEMPTED, _SCHEMA_LOAD_OK, _SCHEMA_LOAD_ERROR, _SCHEMA_KEY_COUNT
    _SCHEMA_LOAD_ATTEMPTED = False
    _SCHEMA_LOAD_OK = False
    _SCHEMA_LOAD_ERROR = ""
    _SCHEMA_KEY_COUNT = 0


def get_schema_load_status() -> dict[str, Any]:
    """Return last schema-load status for diagnostics and tests."""
    schema = config_schema.get_schema()
    key_count = len(schema) if schema else int(_SCHEMA_KEY_COUNT or 0)
    loaded = bool(schema)
    ok = loaded and (_SCHEMA_LOAD_OK or not _SCHEMA_LOAD_ATTEMPTED)
    if loaded and not _SCHEMA_LOAD_ATTEMPTED:
        # Schema already populated by another entrypoint (e.g. tests/train).
        ok = True
    return {
        "ok": bool(ok and not _SCHEMA_LOAD_ERROR) if loaded else bool(_SCHEMA_LOAD_OK),
        "loaded": loaded,
        "attempted": bool(_SCHEMA_LOAD_ATTEMPTED),
        "key_count": key_count,
        "error": str(_SCHEMA_LOAD_ERROR or ""),
    }


def _build_schema_inputs() -> tuple[Any, dict[str, config_schema.ConfigKey]]:
    """Build CLI parser + network extras without importing ``train``."""
    from library.training.cli_args import setup_parser
    from networks.registry import all_network_kwargs

    parser = setup_parser()
    extras = {
        key: config_schema.ConfigKey(name=key, type="str", source="network_module")
        for key in all_network_kwargs()
    }
    return parser, extras


def ensure_schema_populated() -> dict[str, config_schema.ConfigKey]:
    """Populate CONFIG_SCHEMA if empty; record load failures for observability."""
    global _SCHEMA_LOAD_ATTEMPTED, _SCHEMA_LOAD_OK, _SCHEMA_LOAD_ERROR, _SCHEMA_KEY_COUNT

    schema = config_schema.get_schema()
    if schema:
        _SCHEMA_LOAD_OK = True
        _SCHEMA_LOAD_ERROR = ""
        _SCHEMA_KEY_COUNT = len(schema)
        return schema

    _SCHEMA_LOAD_ATTEMPTED = True
    try:
        parser, extras = _build_schema_inputs()
        config_schema.populate_schema(parser, extras=extras)
    except Exception as exc:
        _SCHEMA_LOAD_OK = False
        _SCHEMA_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        _SCHEMA_KEY_COUNT = 0
        LOGGER.warning("schema population failed; validation degraded: %s", _SCHEMA_LOAD_ERROR)
        return config_schema.get_schema()

    schema = config_schema.get_schema()
    if schema:
        _SCHEMA_LOAD_OK = True
        _SCHEMA_LOAD_ERROR = ""
        _SCHEMA_KEY_COUNT = len(schema)
    else:
        _SCHEMA_LOAD_OK = False
        _SCHEMA_LOAD_ERROR = _SCHEMA_LOAD_ERROR or "schema empty after populate"
        _SCHEMA_KEY_COUNT = 0
        LOGGER.warning("schema population produced empty schema")
    return schema


def _schema_unavailable_warning() -> str:
    status = get_schema_load_status()
    detail = status.get("error") or "schema unavailable"
    return f"schema unavailable; validation skipped ({detail})"


def normalize_patch_values(
    values: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Normalize known scalar values and return ``(values, errors, warnings)``.

    Policy:
    - unknown key -> warning
    - known scalar -> schema-declared type coercion
    - choices mismatch / conversion failure -> error
    - nested dict/list values are intentionally out of scope for this gate
    - schema load failure -> warning (never silent no-op)
    """
    ensure_schema_populated()
    schema = config_schema.get_schema()
    if not schema:
        return dict(values), [], [_schema_unavailable_warning()]

    normalized = dict(values)
    errors: list[str] = []
    warnings: list[str] = []
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            continue
        # Nested tables/lists are not top-level argparse schema keys.
        if isinstance(value, (dict, list)):
            continue
        if key in _DELETE_ON_EMPTY_PATCH_KEYS and value in ("", None):
            continue
        resolved = config_schema.resolve_alias(key)
        if resolved not in schema:
            warnings.append(f"unknown key: {key}")
            continue
        spec = schema[resolved]
        try:
            coerced = config_schema._coerce_value(spec, value)
        except (TypeError, ValueError) as exc:
            errors.append(f"{key}: 类型无法转换 ({exc})")
            continue
        normalized[key] = coerced
        if spec.choices and coerced not in spec.choices:
            if not (coerced is None and None in spec.choices):
                errors.append(
                    f"{key}={coerced!r} 不在允许取值 {list(spec.choices)} 中"
                )
    return normalized, errors, warnings


def validate_patch_values(values: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return schema validation errors and warnings for a top-level patch."""
    _normalized, errors, warnings = normalize_patch_values(values)
    return errors, warnings


def validate_config_mapping(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validate a merged/config dict with the same warning/error policy.

    Boundary (C-R4): only top-level scalar-ish keys are checked. Nested
    dict/list tables (e.g. ``datasets``, ``network`` tables) are skipped so
    the gate does not invent false errors for structured TOML sections.
    """
    if not isinstance(cfg, dict):
        return [], []
    values = {
        key: value
        for key, value in cfg.items()
        if isinstance(key, str) and not isinstance(value, (dict, list))
    }
    return validate_patch_values(values)
