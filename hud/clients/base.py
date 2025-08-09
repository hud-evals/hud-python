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

    def __init__(self, mcp_config: dict[str, dict[str, Any]], verbose: bool = False) -> None:
        """
        Initialize base client.

        Args:
            mcp_config: MCP server configuration dict
            verbose: Enable verbose logging
        """
        self.mcp_config = mcp_config
        self.verbose = verbose
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

        logger.info("Initializing MCP client...")

        # Subclasses implement connection
        await self._connect()

        # Discover tools
        self._tools = await self.list_tools()
        logger.info("Discovered %d tools", len(self._tools))

        # Common HUD behavior - fetch telemetry
        await self._fetch_telemetry()

        self._initialized = True
        logger.info("Client initialization complete")

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
                    logger.info("   📋 Services:")
                    for service, status in telemetry_data["services"].items():
                        status_icon = "✅" if status == "running" else "❌"
                        logger.info("      %s %s: %s", status_icon, service, status)

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
