"""Retired path of ``hud.runtime``, served with a warning until hud v0.7.0."""

from __future__ import annotations

from typing import Any

from hud.utils.deprecation import RUNTIME_NAMES, moved_to, resolve_retired

_RETIRED = moved_to("hud.runtime", *RUNTIME_NAMES)


def __getattr__(name: str) -> Any:
    return resolve_retired(__name__, name, _RETIRED)
