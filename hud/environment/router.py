"""Tool routing for Environment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import mcp.types as mcp_types

if TYPE_CHECKING:
    from hud.environment.connection import Connector

__all__ = ["ConflictResolution", "ToolRouter", "LOCAL_CONNECTION"]

logger = logging.getLogger(__name__)

LOCAL_CONNECTION = "__local__"


class ConflictResolution(str, Enum):
    """Strategy for resolving tool name conflicts."""
    PREFIX = "prefix"        # Add connection name as prefix
    FIRST_WINS = "first_wins"  # First connection wins
    LAST_WINS = "last_wins"   # Last connection wins
    ERROR = "error"          # Raise error on conflict


@dataclass
class ToolRouter:
    """Routes tool calls to local or remote handlers with conflict resolution."""

    conflict_resolution: ConflictResolution = ConflictResolution.PREFIX
    _tools: list[mcp_types.Tool] = field(default_factory=list)
    _routing: dict[str, str] = field(default_factory=dict)  # name -> connection
    _local_names: set[str] = field(default_factory=set)

    @property
    def tools(self) -> list[mcp_types.Tool]:
        return self._tools

    def is_local(self, name: str) -> bool:
        return name in self._local_names

    def get_connection(self, name: str) -> str | None:
        """Get connection name for tool, None if local or not found."""
        conn = self._routing.get(name)
        return None if conn == LOCAL_CONNECTION else conn

    def clear(self) -> None:
        self._tools.clear()
        self._routing.clear()
        self._local_names.clear()

    def build(
        self,
        local_tools: list[mcp_types.Tool],
        connections: dict[str, Connector],
        connection_order: list[str],
    ) -> None:
        """Build routing from local tools and connection caches.
        
        Local tools always have priority over remote tools.
        """
        self.clear()
        seen: dict[str, str] = {}

        # Local tools first (always priority)
        for tool in local_tools:
            seen[tool.name] = LOCAL_CONNECTION
            self._routing[tool.name] = LOCAL_CONNECTION
            self._local_names.add(tool.name)
            self._tools.append(tool)

        # Remote connections in order
        for conn_name in connection_order:
            if conn_name not in connections:
                continue
            for tool in connections[conn_name].cached_tools:
                name = tool.name
                if name in seen:
                    existing = seen[name]
                    if existing == LOCAL_CONNECTION:
                        continue  # Local always wins
                    if not self._handle_conflict(name, existing, conn_name):
                        continue
                    self._tools = [t for t in self._tools if t.name != name]

                seen[name] = conn_name
                self._routing[name] = conn_name
                self._tools.append(tool)

        logger.debug("Router: %d tools (%d local)", len(self._tools), len(self._local_names))

    def _handle_conflict(self, name: str, existing: str, new: str) -> bool:
        """Handle remote-to-remote conflict. Returns True to replace existing."""
        if self.conflict_resolution == ConflictResolution.ERROR:
            raise ValueError(f"Tool conflict: '{name}' in '{existing}' and '{new}'")
        if self.conflict_resolution == ConflictResolution.FIRST_WINS:
            return False
        if self.conflict_resolution == ConflictResolution.LAST_WINS:
            return True
        # PREFIX - shouldn't conflict if prefixes set correctly
        return False
