"""Capability declarations + clients."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Capability, CapabilityClient
from .cdp import CDPClient
from .filetracking import FileTrackingClient
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
    "FileTrackingClient",
    "MCPClient",
    "RFBClient",
    "RobotClient",
    "SSHClient",
]
