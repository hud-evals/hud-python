"""Capability declarations + clients."""

from .base import Capability, CapabilityClient
from .cdp import CDPClient
from .mcp import MCPClient
from .rfb import RFBClient
from .ssh import SSHClient

__all__ = ["CDPClient", "Capability", "CapabilityClient", "MCPClient", "RFBClient", "SSHClient"]
