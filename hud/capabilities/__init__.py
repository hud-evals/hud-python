"""Capability declarations + clients."""

from __future__ import annotations

from .base import Capability, CapabilityClient
from .cdp import CDPClient
from .connection import Connection
from .mcp import MCPClient
from .rfb import RFBClient
from .ssh import SSHClient

__all__ = [
    "CDPClient",
    "Capability",
    "CapabilityClient",
    "Connection",
    "MCPClient",
    "RFBClient",
    "SSHClient",
]
