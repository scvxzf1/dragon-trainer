"""Helpers for turning config values into plain Python containers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSY_STRINGS = frozenset({"0", "false", "no", "off"})
_DISABLED_STRINGS = _FALSY_STRINGS | frozenset({"", "none"})


def parse_bool(
    value: Any,
    default: bool = False,
    *,
    strict: bool = False,
    field: str = "value",
) -> bool:
    """Canonical boolean coercion across CLI kwargs, TOML, and WebUI patches."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        if strict:
            raise ValueError(f"{field} must be a boolean, got {value!r}")
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUTHY_STRINGS:
            return True
        if normalized in _FALSY_STRINGS:
            return False
        if not strict and normalized in {"", "none"}:
            return default
        if strict:
            raise ValueError(f"{field} must be a boolean, got {value!r}")
        return False
    if strict:
        raise ValueError(f"{field} must be a boolean, got {value!r}")
    return bool(value)


def is_flag_enabled(value: Any) -> bool:
    """Fail-closed non-disabled predicate (treats non-empty enum strings as active)."""
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and value == 0:
        return False
    return str(value).strip().lower() not in _DISABLED_STRINGS


def to_plain_config(value: Any) -> Any:
    """Recursively convert config mapping/list containers to pickle-safe types."""
    if isinstance(value, Mapping):
        return {key: to_plain_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_plain_config(item) for item in value)
    return value
