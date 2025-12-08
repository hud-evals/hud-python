"""Environment variable resolution utilities."""

from __future__ import annotations

import contextlib
import os
from collections import defaultdict
from string import Template
from typing import TYPE_CHECKING, Any

from hud.settings import settings

if TYPE_CHECKING:
    from collections.abc import Mapping


def resolve_env_vars(obj: Any, extra_mapping: Mapping[str, Any] | None = None) -> Any:
    """Recursively resolve ${VAR_NAME} placeholders in strings.

    Uses Python's string.Template for substitution. Sources values from:
    1. os.environ
    2. hud.settings (loads from project .env and ~/.hud/.env)
    3. Optional extra_mapping parameter

    Uppercase aliases are automatically added for settings keys,
    so both ${api_key} and ${API_KEY} work.

    Missing variables resolve to empty strings.

    Args:
        obj: The object to resolve (string, dict, list, or other).
        extra_mapping: Optional additional key-value pairs to include.

    Returns:
        The object with all ${VAR_NAME} placeholders resolved.

    Example:
        >>> resolve_env_vars({"key": "${MY_VAR}"})
        {'key': 'resolved_value'}
    """
    # Build mapping from environment and settings
    mapping: dict[str, Any] = dict(os.environ)
    settings_dict = settings.model_dump()
    mapping.update(settings_dict)

    # Add UPPERCASE aliases for settings keys
    for key, val in settings_dict.items():
        with contextlib.suppress(Exception):
            mapping[key.upper()] = val

    if settings.api_key:
        mapping["HUD_API_KEY"] = settings.api_key

    if extra_mapping:
        mapping.update(extra_mapping)

    def substitute(value: Any) -> Any:
        if isinstance(value, str):
            safe_mapping = defaultdict(str, mapping)
            return Template(value).substitute(safe_mapping)
        elif isinstance(value, dict):
            return {k: substitute(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute(item) for item in value]
        return value

    return substitute(obj)
