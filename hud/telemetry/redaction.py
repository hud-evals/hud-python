"""Redact credentials injected into environment setup telemetry."""

from __future__ import annotations

import json
from typing import Any

_REDACTED = "[REDACTED]"


def redact_runtime_credentials(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy telemetry without runtime credential fields or their echoed values."""
    secrets: set[str] = set()
    redacted = _redact_fields(payload, secrets)
    if not secrets:
        return redacted
    return _replace_secrets(redacted, sorted(secrets, key=len, reverse=True))


def _redact_fields(value: Any, secrets: set[str]) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key.lower() == "hud_api_key":
                if isinstance(item, str) and item and item != _REDACTED:
                    secrets.add(item)
                result[key] = _REDACTED
            else:
                result[key] = _redact_fields(item, secrets)
        return result
    if isinstance(value, list):
        return [_redact_fields(item, secrets) for item in value]
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        redacted = _redact_fields(decoded, secrets)
        return json.dumps(redacted, ensure_ascii=False) if redacted != decoded else value
    return value


def _replace_secrets(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_secrets(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_secrets(item, secrets) for item in value]
    if isinstance(value, str):
        if value.lstrip().startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                pass
            else:
                redacted = _replace_secrets(decoded, secrets)
                return json.dumps(redacted, ensure_ascii=False) if redacted != decoded else value
        for secret in secrets:
            value = value.replace(secret, _REDACTED)
    return value
