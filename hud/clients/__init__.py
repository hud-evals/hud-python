"""HUD control-channel client: ``connect``, ``HudClient``, ``Manifest``, ``HudProtocolError``, and
``register_client`` for capability clients of new protocols."""

from __future__ import annotations

from .client import HudClient, HudProtocolError, Manifest, connect, register_client

__all__ = [
    "HudClient",
    "HudProtocolError",
    "Manifest",
    "connect",
    "register_client",
]
