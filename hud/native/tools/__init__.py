"""Standalone HUD tools.

``BaseTool``s you register ad-hoc on your own :class:`hud.server.MCPServer`, which
the new :class:`hud.environment.Environment` then exposes as an ``mcp`` capability.
These are the tools the provider agents don't drive natively (jupyter, memory,
playwright, plus the bash/edit coding tools memory builds on).
"""

from .base import BaseHub, BaseTool
from .coding import BashTool, EditTool
from .jupyter import JupyterTool
from .memory import MemoryTool
from .playwright import PlaywrightTool

__all__ = [
    "BaseHub",
    "BaseTool",
    "BashTool",
    "EditTool",
    "JupyterTool",
    "MemoryTool",
    "PlaywrightTool",
]
