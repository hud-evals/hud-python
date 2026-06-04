"""Standalone HUD tools.

``BaseTool``s you register ad-hoc on your own :class:`hud.server.MCPServer`, which
the new :class:`hud.environment.Environment` then exposes as an ``mcp`` capability.
These are the tools the provider agents don't drive natively (jupyter, memory,
playwright, plus the bash/edit coding tools memory builds on), and ``AgentTool``
for exposing a task as a sub-agent tool.

Exports are resolved lazily so importing one tool never pulls another's optional
dependency (e.g. importing ``AgentTool`` won't import playwright).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import AgentTool as AgentTool
    from .base import BaseTool as BaseTool
    from .coding import BashTool as BashTool
    from .coding import EditTool as EditTool
    from .jupyter import JupyterTool as JupyterTool
    from .memory import MemoryTool as MemoryTool
    from .playwright import PlaywrightTool as PlaywrightTool

_LAZY: dict[str, str] = {
    "AgentTool": ".agent",
    "BaseTool": ".base",
    "BashTool": ".coding",
    "EditTool": ".coding",
    "JupyterTool": ".jupyter",
    "MemoryTool": ".memory",
    "PlaywrightTool": ".playwright",
}

__all__ = [
    "AgentTool",
    "BaseTool",
    "BashTool",
    "EditTool",
    "JupyterTool",
    "MemoryTool",
    "PlaywrightTool",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)
