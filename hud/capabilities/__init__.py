"""Capability declarations + clients."""

from .base import Capability, CapabilityClient
from .mcp import MCPClient
from .rfb import RFBClient
from .ssh import SSHClient

__all__ = ["Capability", "CapabilityClient", "MCPClient", "RFBClient", "SSHClient"]
