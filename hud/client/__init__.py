"""HUD wire client: ``Manifest``, ``ServerInfo``, ``HudClient``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hud.capabilities import Capability


@dataclass(frozen=True, slots=True)
class ServerInfo:
    """Identity of the env serving this session (for compatibility / observability)."""

    name: str
    version: str


@dataclass(frozen=True, slots=True)
class Manifest:
    """Env welcome frame returned by ``HudClient.hello()``."""

    session_id: str
    protocol_version: str  # e.g. "hud/1.0"
    server_info: ServerInfo
    bindings: list[Capability]


from .client import HudClient, HudProtocolError, connect  # noqa: E402
from .launch import Variant, launch, variant  # noqa: E402
from .rollout import Rollout  # noqa: E402

__all__ = [
    "HudClient",
    "HudProtocolError",
    "Manifest",
    "Rollout",
    "ServerInfo",
    "Variant",
    "connect",
    "launch",
    "variant",
]
