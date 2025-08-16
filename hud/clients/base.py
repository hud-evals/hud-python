"""Base protocol and implementation for HUD MCP clients."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import mcp.types as types

    from hud.types import MCPToolResult

else:
    pass


logger = logging.getLogger(__name__)


@runtime_checkable
class AgentMCPClient(Protocol):
    """Minimal interface for MCP clients used by agents."""

    async def initialize(self) -> None:
        """Initialize the client - connect and fetch telemetry."""
        ...

    async def list_tools(self) -> list[types.Tool]:
        """List all available tools."""
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        """Execute a tool by name."""
        ...


class BaseHUDClient(ABC):
    """Base class with common HUD functionality."""

    def __init__(
        self,
        mcp_config: dict[str, dict[str, Any]],
        verbose: bool = False,
        strict_validation: bool = False,
    ) -> None:
        """
        Initialize base client.

        Args:
            mcp_config: MCP server configuration dict
            verbose: Enable verbose logging
            strict_validation: Enable strict tool output validation
        """
        self.mcp_config = mcp_config
        self.verbose = verbose
        self.strict_validation = strict_validation
        self._telemetry_data: dict[str, Any] = {}
        self._initialized = False
        self._tools: list[types.Tool] = []

        if self.verbose:
            self._setup_verbose_logging()

    def _setup_verbose_logging(self) -> None:
        """Configure verbose logging for debugging."""
        logging.getLogger("mcp").setLevel(logging.DEBUG)
        logging.getLogger("fastmcp").setLevel(logging.DEBUG)

        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

    async def initialize(self) -> None:
        """Initialize connection and fetch telemetry."""
        if self._initialized:
            return

        logger.debug("Initializing MCP client...")

        # Subclasses implement connection
        await self._connect()

        # Discover tools
        self._tools = await self.list_tools()
        logger.debug("Client discovered %d tools", len(self._tools))

        # Common HUD behavior - fetch telemetry
        await self._fetch_telemetry()

        self._initialized = True
        logger.info("Client initialized with %d tools", len(self._tools))

    @abstractmethod
    async def _connect(self) -> None:
        """Subclasses implement their connection logic."""
        raise NotImplementedError

    async def _fetch_telemetry(self) -> None:
        """Common telemetry fetching for all HUD clients."""
        try:
            # Try to read telemetry resource directly
            result = await self._read_resource_internal("telemetry://live")
            if result and result.contents:
                # Parse telemetry data
                telemetry_data = json.loads(result.contents[0].text)  # type: ignore
                self._telemetry_data = telemetry_data

                logger.info("📡 Telemetry data fetched:")
                if "live_url" in telemetry_data:
                    logger.info("   🖥️  Live URL: %s", telemetry_data["live_url"])
                if "status" in telemetry_data:
                    logger.info("   📊 Status: %s", telemetry_data["status"])
                if "services" in telemetry_data:
                    logger.debug("   📋 Services:")
                    for service, status in telemetry_data["services"].items():
                        status_icon = "✅" if status == "running" else "❌"
                        logger.debug("      %s %s: %s", status_icon, service, status)

                if self.verbose:
                    logger.debug("Full telemetry data:\n%s", json.dumps(telemetry_data, indent=2))
        except Exception as e:
            # Telemetry is optional
            if self.verbose:
                logger.debug("No telemetry available: %s", e)

    @abstractmethod
    async def _read_resource_internal(self, uri: str) -> types.ReadResourceResult | None:
        """Internal method to read resources - subclasses implement."""
        raise NotImplementedError

    def get_telemetry_data(self) -> dict[str, Any]:
        """Get collected telemetry data."""
        return self._telemetry_data

    def get_available_tools(self) -> list[types.Tool]:
        """Get list of discovered tools."""
        return self._tools

    @property
    def is_connected(self) -> bool:
        """Check if client is connected and initialized."""
        return self._initialized

    @abstractmethod
    async def list_tools(self) -> list[types.Tool]:
        """List all available tools."""
        raise NotImplementedError

    @abstractmethod
    async def list_resources(self) -> list[types.Resource]:
        """List all available resources."""
        raise NotImplementedError

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        """Execute a tool by name."""
        raise NotImplementedError

    async def get_hub_tools(self, hub_name: str) -> list[str]:
        """Get all subtools for a specific hub (setup/evaluate).

        Args:
            hub_name: Name of the hub (e.g., "setup", "evaluate")

        Returns:
            List of available function names for the hub
        """
        try:
            # Read the hub's functions catalogue resource
            result = await self._read_resource_internal(f"file:///{hub_name}/functions")
            if result and result.contents:
                # Parse the JSON list of function names
                import json

                functions = json.loads(result.contents[0].text)  # type: ignore
                return functions
        except Exception as e:
            if self.verbose:
                logger.debug("Could not read hub functions for '%s': %s", hub_name, e)
        return []

    async def analyze_environment(self) -> dict[str, Any]:
        """Complete analysis of the MCP environment.

        Returns:
            Dictionary containing:
            - tools: All tools with full schemas
            - hub_tools: Hub structures with subtools
            - telemetry: Telemetry resources and data
            - resources: All available resources
            - metadata: Environment metadata
        """
        analysis: dict[str, Any] = {
            "tools": [],
            "hub_tools": {},
            "telemetry": self._telemetry_data,
            "resources": [],
            "metadata": {
                "servers": list(self.mcp_config.keys()),
                "initialized": self._initialized,
            },
        }

        # Get all tools with schemas
        tools = await self.list_tools()
        for tool in tools:
            tool_info = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            analysis["tools"].append(tool_info)

            # Check if this is a hub tool (like setup, evaluate)
            if (
                tool.description
                and "internal" in tool.description.lower()
                and "functions" in tool.description.lower()
            ):
                # This is likely a hub dispatcher tool
                hub_functions = await self.get_hub_tools(tool.name)
                if hub_functions:
                    analysis["hub_tools"][tool.name] = hub_functions

        # Get all resources
        try:
            resources = await self.list_resources()
            for resource in resources:
                resource_info = {
                    "uri": str(resource.uri),
                    "name": resource.name,
                    "description": resource.description,
                    "mime_type": getattr(resource, "mimeType", None),
                }
                analysis["resources"].append(resource_info)
        except Exception as e:
            if self.verbose:
                logger.debug("Could not list resources: %s", e)

        return analysis
