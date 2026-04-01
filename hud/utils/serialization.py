from __future__ import annotations

import json
from typing import Any

import pydantic_core


def _unserializable_placeholder(value: Any) -> str:
    return f"<{type(value).__name__}: not serializable>"


def json_safe_value(value: Any) -> Any:
    """Serialize a value into JSON-compatible data when possible."""
    if isinstance(value, str | int | float | bool | type(None)):
        return value

    try:
        return json.loads(pydantic_core.to_json(value, fallback=_unserializable_placeholder))
    except Exception:
        return _unserializable_placeholder(value)


def json_safe_dict(values: dict[str, Any]) -> dict[str, Any]:
    """Serialize a mapping into JSON-compatible data."""
    return {key: json_safe_value(value) for key, value in values.items()}
