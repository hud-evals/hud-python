"""HUD control-channel client: ``connect``, ``HudClient``, ``Manifest`` and ``HudProtocolError``."""

from __future__ import annotations

from typing import Any

from hud.utils.deprecation import moved_to, resolve_retired

from .client import HudClient, HudProtocolError, Manifest

__all__ = [
    "HudClient",
    "HudProtocolError",
    "Manifest",
]

_RETIRED = moved_to("hud.capabilities", "register_client")


def __getattr__(name: str) -> Any:
    return resolve_retired(__name__, name, _RETIRED)
