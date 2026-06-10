"""Capability declarations + clients.

Only :class:`Capability` / :class:`CapabilityClient` (the declaration base) are
imported eagerly. The concrete clients are loaded lazily on first attribute
access (PEP 562) so that importing this package — e.g. the server side bringing
up an :class:`~hud.environment.Environment` with a robot capability — does not
pull heavy/optional client dependencies (``fastmcp`` for MCP, ``websockets``'s
asyncio client for CDP). This lets an env server run in a minimal environment
(e.g. an Isaac Sim conda env pinned to an older ``websockets``).

The *env-side* robot runtime (the ``robot/1`` bridges, action providers, and sim
runners) lives in :mod:`hud.environment.robots`; only the agent-side
:class:`~hud.capabilities.robot.RobotClient` is a capability client and stays here.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .base import Capability, CapabilityClient

#: Public name -> (submodule, attribute). Loaded on demand by ``__getattr__``.
_LAZY: dict[str, tuple[str, str]] = {
    "CDPClient": ("cdp", "CDPClient"),
    "MCPClient": ("mcp", "MCPClient"),
    "RFBClient": ("rfb", "RFBClient"),
    "RobotClient": ("robot", "RobotClient"),
    "SSHClient": ("ssh", "SSHClient"),
}

if TYPE_CHECKING:  # static analysers still see the real symbols
    from .cdp import CDPClient
    from .mcp import MCPClient
    from .rfb import RFBClient
    from .robot import RobotClient
    from .ssh import SSHClient


def __getattr__(name: str) -> object:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = target
    value = getattr(import_module(f".{module}", __name__), attr)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "CDPClient",
    "Capability",
    "CapabilityClient",
    "MCPClient",
    "RFBClient",
    "RobotClient",
    "SSHClient",
]
