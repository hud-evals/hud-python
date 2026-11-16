"""Capability declarations + clients."""

from __future__ import annotations

from .base import Capability, CapabilityClient
from .connection import Connection
from .mcp import MCPClient
from .rfb import RFBClient
from .ssh import SSHClient

__all__ = [
    "Capability",
    "CapabilityClient",
    "Connection",
    "MCPClient",
    "RFBClient",
    "SSHClient",
]
