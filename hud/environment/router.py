"""MCP routing for Environment - tools, prompts, and resources."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mcp.types as mcp_types

    from hud.environment.connection import Connector

__all__ = ["LOCAL_CONNECTION", "ConflictResolution", "MCPRouter", "ToolRouter"]

logger = logging.getLogger(__name__)

LOCAL_CONNECTION = "__local__"


class ConflictResolution(str, Enum):
    """Strategy for resolving name conflicts."""

    PREFIX = "prefix"  # Add connection name as prefix
    FIRST_WINS = "first_wins"  # First connection wins
    LAST_WINS = "last_wins"  # Last connection wins
    ERROR = "error"  # Raise error on conflict


@dataclass
class MCPRouter:
    """Routes tools, prompts, and resources to local or remote handlers.

    Builds routing tables during Environment.__aenter__ from local registrations
    and connection caches. Provides get_*_connection() methods to find which
    connection serves a given tool/prompt/resource.
    """

    conflict_resolution: ConflictResolution = ConflictResolution.PREFIX

    # Tool routing
    _tools: list[mcp_types.Tool] = field(default_factory=list)
    _tool_routing: dict[str, str] = field(default_factory=dict)  # name -> connection
    _local_tool_names: set[str] = field(default_factory=set)

    # Prompt routing
    _prompts: list[mcp_types.Prompt] = field(default_factory=list)
    _prompt_routing: dict[str, str] = field(default_factory=dict)  # name -> connection

    # Resource routing
    _resources: list[mcp_types.Resource] = field(default_factory=list)
    _resource_routing: dict[str, str] = field(default_factory=dict)  # uri -> connection

    # =========================================================================
    # Tool routing (backwards compatible)
    # =========================================================================

    @property
    def tools(self) -> list[mcp_types.Tool]:
        return self._tools

    def is_local(self, name: str) -> bool:
        """Check if tool is local (backwards compat)."""
        return name in self._local_tool_names

    def get_connection(self, name: str) -> str | None:
        """Get connection name for tool, None if local or not found (backwards compat)."""
        return self.get_tool_connection(name)

    def get_tool_connection(self, name: str) -> str | None:
        """Get connection name for tool, None if local or not found."""
        conn = self._tool_routing.get(name)
        return None if conn == LOCAL_CONNECTION else conn

    # =========================================================================
    # Prompt routing
    # =========================================================================

    @property
    def prompts(self) -> list[mcp_types.Prompt]:
        return self._prompts

    def get_prompt_connection(self, name: str) -> str | None:
        """Get connection name for prompt, None if local or not found."""
        conn = self._prompt_routing.get(name)
        return None if conn == LOCAL_CONNECTION else conn

    # =========================================================================
    # Resource routing
    # =========================================================================

    @property
    def resources(self) -> list[mcp_types.Resource]:
        return self._resources

    def get_resource_connection(self, uri: str) -> str | None:
        """Get connection name for resource, None if local or not found."""
        conn = self._resource_routing.get(uri)
        return None if conn == LOCAL_CONNECTION else conn

    # =========================================================================
    # Building routes
    # =========================================================================

    def clear(self) -> None:
        """Clear all routing tables."""
        self._tools.clear()
        self._tool_routing.clear()
        self._local_tool_names.clear()
        self._prompts.clear()
        self._prompt_routing.clear()
        self._resources.clear()
        self._resource_routing.clear()

    def build(
        self,
        local_tools: list[mcp_types.Tool],
        connections: dict[str, Connector],
        connection_order: list[str],
    ) -> None:
        """Build tool routing from local tools and connection caches.

        Local tools always have priority over remote tools.
        Tools starting with '_' are internal and hidden from listing
        (but still callable directly).
        """
        # Clear tool routing only (prompts/resources built separately)
        self._tools.clear()
        self._tool_routing.clear()
        self._local_tool_names.clear()

        seen: dict[str, str] = {}

        # Local tools first (always priority)
        for tool in local_tools:
            seen[tool.name] = LOCAL_CONNECTION
            self._tool_routing[tool.name] = LOCAL_CONNECTION
            self._local_tool_names.add(tool.name)
            if not tool.name.startswith("_"):
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
                        continue
                    if not self._handle_conflict(name, existing, conn_name):
                        continue
                    self._tools = [t for t in self._tools if t.name != name]

                seen[name] = conn_name
                self._tool_routing[name] = conn_name
                if not name.startswith("_"):
                    self._tools.append(tool)

        logger.debug("Router: %d tools (%d local)", len(self._tools), len(self._local_tool_names))

    def build_prompts(
        self,
        local_prompts: list[mcp_types.Prompt],
        connections: dict[str, Connector],
    ) -> None:
        """Build prompt routing from local prompts and connections.

        Uses cached prompts from connections (populated during __aenter__).
        """
        self._prompts.clear()
        self._prompt_routing.clear()

        seen: dict[str, str] = {}

        # Local prompts first (always priority)
        for prompt in local_prompts:
            seen[prompt.name] = LOCAL_CONNECTION
            self._prompt_routing[prompt.name] = LOCAL_CONNECTION
            self._prompts.append(prompt)

        # Use cached prompts from each connection (populated during __aenter__)
        results: list[tuple[str, list[mcp_types.Prompt]]] = [
            (conn_name, conn.cached_prompts) for conn_name, conn in connections.items()
        ]

        # Process results in connection order (dict preserves insertion order)
        for conn_name, remote_prompts in results:
            for prompt in remote_prompts:
                name = prompt.name
                if name in seen:
                    existing = seen[name]
                    if existing == LOCAL_CONNECTION:
                        continue  # Local always wins
                    if not self._handle_conflict(name, existing, conn_name):
                        continue
                    # Remove old prompt from list
                    self._prompts = [p for p in self._prompts if p.name != name]

                seen[name] = conn_name
                self._prompt_routing[name] = conn_name
                self._prompts.append(prompt)

        logger.debug("Router: %d prompts", len(self._prompts))

    def build_resources(
        self,
        local_resources: list[mcp_types.Resource],
        connections: dict[str, Connector],
    ) -> None:
        """Build resource routing from local resources and connections.

        Uses cached resources from connections (populated during __aenter__).
        """
        self._resources.clear()
        self._resource_routing.clear()

        seen: dict[str, str] = {}

        # Local resources first (always priority)
        for resource in local_resources:
            uri = str(resource.uri)
            seen[uri] = LOCAL_CONNECTION
            self._resource_routing[uri] = LOCAL_CONNECTION
            self._resources.append(resource)

        # Use cached resources from each connection (populated during __aenter__)
        results: list[tuple[str, list[mcp_types.Resource]]] = [
            (conn_name, conn.cached_resources) for conn_name, conn in connections.items()
        ]

        # Process results in connection order (dict preserves insertion order)
        for conn_name, remote_resources in results:
            for resource in remote_resources:
                uri = str(resource.uri)
                if uri in seen:
                    existing = seen[uri]
                    if existing == LOCAL_CONNECTION:
                        continue  # Local always wins
                    if not self._handle_conflict(uri, existing, conn_name):
                        continue
                    # Remove old resource from list
                    self._resources = [r for r in self._resources if str(r.uri) != uri]

                seen[uri] = conn_name
                self._resource_routing[uri] = conn_name
                self._resources.append(resource)

        logger.debug("Router: %d resources", len(self._resources))

    def _handle_conflict(self, name: str, existing: str, new: str) -> bool:
        """Handle remote-to-remote conflict. Returns True to replace existing."""
        if self.conflict_resolution == ConflictResolution.ERROR:
            raise ValueError(f"Conflict: '{name}' in '{existing}' and '{new}'")
        if self.conflict_resolution == ConflictResolution.FIRST_WINS:
            return False
        return self.conflict_resolution == ConflictResolution.LAST_WINS


# Backwards compatibility alias
ToolRouter = MCPRouter
