"""Standalone HUD tools.

``BaseTool``s you register ad-hoc on your own :class:`hud.server.MCPServer`, which
the new :class:`hud.environment.Environment` then exposes as an ``mcp``
capability, and ``AgentTool`` for exposing a task as a sub-agent tool.

Shell, file editing, computer use, and browsing are capabilities, not tools:
declare ``ssh`` / ``rfb`` / ``cdp`` (e.g. via
:class:`hud.environment.Workspace`) and the agent harness drives them with
provider-native tools.

Symbols and submodules removed in the v6 teardown (computer/shell tools,
``jupyter``, ``playwright``, ``types``, ``filesystem``, …) still resolve for
deployed v5 envs via :mod:`hud._legacy`.
"""

from __future__ import annotations

from typing import Any

from hud._legacy import resolve_legacy_name

from .agent import AgentTool
from .base import BaseTool

__all__ = [
    "AgentTool",
    "BaseTool",
]


def __getattr__(name: str) -> Any:
    return resolve_legacy_name(__name__, name)
