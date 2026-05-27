"""Capability declarations + clients."""

from .base import Capability, CapabilityClient
from .mcp import MCPClient
from .ssh import SSHClient

__all__ = ["Capability", "CapabilityClient", "MCPClient", "SSHClient"]
