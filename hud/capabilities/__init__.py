"""Capability declarations + clients.

The env-side robot runtime (bridges, action providers, sim runners) lives in
:mod:`hud.environment.robots`; only the agent-side
:class:`~hud.capabilities.robot.RobotClient` is a capability client and stays here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Capability, CapabilityClient
from .cdp import CDPClient
from .mcp import MCPClient
from .rfb import RFBClient
from .ssh import SSHClient

if TYPE_CHECKING:
    from .robot import RobotClient


def __getattr__(name: str) -> object:
    # RobotClient pulls optional dependencies (numpy/msgpack — the ``robot``
    # extra), so unlike the core clients above it is imported on first access.
    if name == "RobotClient":
        from .robot import RobotClient

        return RobotClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CDPClient",
    "Capability",
    "CapabilityClient",
    "MCPClient",
    "RFBClient",
    "RobotClient",
    "SSHClient",
]
