"""MCP Client wrapper - backward compatibility shim.

This module provides backward compatibility for existing code that imports
MCPClient from hud.client. New code should import from hud.clients instead.
"""

# Import the default client implementation for backward compatibility
from __future__ import annotations

from .clients import MCPClient

# Re-export for backward compatibility
__all__ = ["MCPClient"]
