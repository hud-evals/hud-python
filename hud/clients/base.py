"""Base protocol and implementation for HUD MCP clients."""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

from mcp.types import Implementation

from hud.shared.exceptions import HudAuthenticationError, HudException
from hud.types import MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole
from hud.version import __version__ as hud_version

if TYPE_CHECKING:
    import mcp.types as types
logger = logging.getLogger(__name__)

hud_console = HUDConsole(logger=logger)


@runtime_checkable
class AgentMCPClient(Protocol):
    """Minimal interface for MCP clients used by agents.

    Any custom client must implement this interface.

    Any custom agent can assume that this will be the interaction protocol.
    """

    _initialized: bool

    @property
    def mcp_config(self) -> dict[str, dict[str, Any]]:
        """Get the MCP config."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if client is connected and initialized."""
        ...

    async def initialize(self, mcp_config: dict[str, dict[str, Any]] | None = None) -> None:
        """Initialize the client."""
        ...

    async def list_tools(self) -> list[types.Tool]:
        """List all available tools."""
        ...

    async def call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """Execute a tool by name."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the client."""
        ...


class BaseHUDClient(AgentMCPClient):
    """Base class with common HUD functionality that adds:
    - Connection management
    - Tool discovery
    - Telemetry fetching (hud environment-specific)
    - Logging
    - Strict tool output validation (optional)
    - Environment analysis (optional)

    Any custom client should inherit from this class, and implement:
    - _connect: Connect to the MCP server
    - list_tools: List all available tools
    - list_resources: List all available resources
    - call_tool: Execute a tool by name
    - read_resource: Read a resource by URI
    - _disconnect: Disconnect from the MCP server
    - any other MCP client methods
    """

    client_info = Implementation(name="hud-mcp", title="hud MCP Client", version=hud_version)

    def __init__(
        self,
        mcp_config: dict[str, dict[str, Any]] | None = None,
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
        self.verbose = verbose
        self._mcp_config = mcp_config
        self._strict_validation = strict_validation

        self._initialized = False
        self._telemetry_data = {}  # Initialize telemetry data
        self._cached_resources: list[types.Resource] = []  # Cache for resources
        self._cached_prompts: list[types.Prompt] = []  # Cache for prompts

        if self.verbose:
            self._setup_verbose_logging()

    async def initialize(self, mcp_config: dict[str, dict[str, Any]] | None = None) -> None:
        """Initialize connection and fetch tools."""
        if self._initialized:
            hud_console.warning(
                "Client already connected, if you want to reconnect or change the configuration, "
                "call shutdown() first. This is especially important if you are using an agent."
            )
            return

        self._mcp_config = mcp_config or self._mcp_config
        if self._mcp_config is None:
            from hud.shared.exceptions import HudConfigError

            raise HudConfigError(
                "An MCP server configuration is required. "
                "Either pass it to the constructor or call initialize with a configuration"
            )

        hud_console.debug("Initializing MCP client...")

        try:
            # Check if API key is set for HUD API
            for server_config in self._mcp_config.values():
                url = server_config.get("url", "")
                headers = server_config.get("headers", {})
                if "mcp.hud.ai" in url and len(headers.get("Authorization", "")) < 10:
                    raise HudAuthenticationError(
                        f'Sending authorization "{headers.get("Authorization", "")}", which may'
                        " be incomplete. Ensure HUD_API_KEY environment variable is set or send it"
                        " as a header. You can get an API key at https://hud.ai"
                    )
            # Subclasses implement connection
            await self._connect(self._mcp_config)
        except HudException:
            raise
        except Exception as e:
            hud_console.error(f"Failed to initialize MCP client: {e}")
            raise HudException from e

        # Common hud behavior - fetch telemetry
        await self._fetch_telemetry()

        self._initialized = True

    async def shutdown(self) -> None:
        """Disconnect from the MCP server."""
        if self._initialized:
            await self._disconnect()
            self._initialized = False
            self._cached_resources.clear()
            self._cached_prompts.clear()
            hud_console.info("Environment Shutdown completed")
        else:
            hud_console.debug("Client was not initialized, skipping disconnect")

    @overload
    async def call_tool(self, tool_call: MCPToolCall, /) -> MCPToolResult: ...
    @overload
    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult: ...

    async def call_tool(
        self,
        tool_call: MCPToolCall | None = None,
        *,
        name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        if tool_call is not None:
            return await self._call_tool(tool_call)
        elif name is not None:
            return await self._call_tool(MCPToolCall(name=name, arguments=arguments))
        else:
            raise TypeError(
                "call_tool() requires either an MCPToolCall positional arg "
                "or keyword 'name' (and optional 'arguments')."
            )

    @abstractmethod
    async def _connect(self, mcp_config: dict[str, dict[str, Any]]) -> None:
        """Subclasses implement their connection logic."""
        raise NotImplementedError

    @abstractmethod
    async def list_tools(self) -> list[types.Tool]:
        """List all available tools."""
        raise NotImplementedError

    async def list_resources(self) -> list[types.Resource]:
        """List all available resources.

        Uses cached resources if available, otherwise fetches from the server.

        Returns:
            List of available resources.
        """
        # If cache is empty, populate it
        if not self._cached_resources:
            self._cached_resources = await self._list_resources_impl()
        return self._cached_resources

    @abstractmethod
    async def _list_resources_impl(self) -> list[types.Resource]:
        """Implementation-specific resource listing. Subclasses must implement this."""
        raise NotImplementedError

    async def list_prompts(self) -> list[types.Prompt]:
        """List all available prompts.

        Uses cached prompts if available, otherwise fetches from the server.
        Prompts are optional in MCP; default implementation returns an empty list.
        """
        if not self._cached_prompts:
            self._cached_prompts = await self._list_prompts_impl()
        return self._cached_prompts

    async def _list_prompts_impl(self) -> list[types.Prompt]:
        """Implementation-specific prompt listing (optional).

        Subclasses can override to support prompt discovery.
        """
        return []

    @abstractmethod
    async def _call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """Execute a tool by name."""
        raise NotImplementedError

    @abstractmethod
    async def read_resource(self, uri: str) -> types.ReadResourceResult | None:
        """Read a resource by URI."""
        raise NotImplementedError

    @abstractmethod
    async def _disconnect(self) -> None:
        """Subclasses implement their disconnection logic."""
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        """Check if client is connected and initialized."""
        return self._initialized

    @property
    def mcp_config(self) -> dict[str, dict[str, Any]]:
        """Get the MCP config."""
        if self._mcp_config is None:
            from hud.shared.exceptions import HudConfigError

            raise HudConfigError("Please initialize the client with a valid MCP config")
        return self._mcp_config

    async def __aenter__(self: Any) -> Any:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Async context manager exit."""
        await self.shutdown()

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

    async def _fetch_telemetry(self) -> None:
        """Common telemetry fetching for all hud clients."""
        try:
            # Get resources (will use cache if available, otherwise fetch)
            resources = await self.list_resources()
            telemetry_available = any(
                str(resource.uri) == "telemetry://live" for resource in resources
            )

            if not telemetry_available:
                if self.verbose:
                    hud_console.debug("Telemetry resource not available from server")
                return

            # Try to read telemetry resource directly
            result = await self.read_resource("telemetry://live")
            if result and result.contents:
                # Parse telemetry data
                telemetry_data = json.loads(result.contents[0].text)  # type: ignore
                self._telemetry_data = telemetry_data

                if "live_url" in telemetry_data:
                    hud_console.info(f"   🖥️  Live URL: {telemetry_data['live_url']}")
                if "vnc_url" in telemetry_data:
                    hud_console.info(f"   🖥️  VNC URL: {telemetry_data['vnc_url']}")
                if "cdp_url" in telemetry_data:
                    hud_console.info(f"   🦾  CDP URL: {telemetry_data['cdp_url']}")
                if "status" in telemetry_data:
                    hud_console.debug(f"   📊 Status: {telemetry_data['status']}")
                if "services" in telemetry_data:
                    hud_console.debug("   📋 Services:")
                    for service, status in telemetry_data["services"].items():
                        status_icon = "✅" if status == "running" else "❌"
                        hud_console.debug(f"      {status_icon} {service}: {status}")

                if self.verbose:
                    hud_console.debug(
                        f"Full telemetry data:\n{json.dumps(telemetry_data, indent=2)}"
                    )
        except Exception as e:
            # Telemetry is optional
            if self.verbose:
                hud_console.debug(f"No telemetry available: {e}")

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
        if not self._initialized:
            from hud.shared.exceptions import HudClientError

            raise HudClientError("Client must be initialized before analyzing the environment")

        analysis: dict[str, Any] = {
            "tools": [],
            "hub_tools": {},
            "telemetry": self._telemetry_data,
            "resources": [],
            "prompts": [],
            "scenarios": [],
            "verbose": self.verbose,
            "metadata": {
                "servers": list(self._mcp_config.keys()),  # type: ignore
                "initialized": self._initialized,
            },
        }

        # Get all tools with schemas
        tools = await self.list_tools()
        for tool in tools:
            tool_info = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
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
                resource_info: dict[str, Any] = {
                    "uri": str(resource.uri),
                    "name": resource.name,
                    "description": resource.description,
                    "mime_type": getattr(resource, "mimeType", None),
                }
                # Include meta field if present (contains scenario source code)
                meta = getattr(resource, "meta", None)
                if meta:
                    resource_info["meta"] = meta
                analysis["resources"].append(resource_info)
        except Exception as e:
            if self.verbose:
                hud_console.debug("Could not list resources: " + str(e))

        # Get all prompts (optional)
        try:
            prompts = await self.list_prompts()
            for prompt in prompts:
                raw_args = getattr(prompt, "arguments", []) or []
                args: list[dict[str, Any]] = [
                    {
                        "name": getattr(a, "name", None),
                        "required": getattr(a, "required", None),
                        "description": getattr(a, "description", None),
                    }
                    for a in raw_args
                ]

                prompt_info: dict[str, Any] = {
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": args,
                }
                # Include meta field if present
                meta = getattr(prompt, "meta", None)
                if meta:
                    prompt_info["meta"] = meta
                    # Merge type/default info from meta.arguments into the arguments array
                    if isinstance(meta, dict) and "arguments" in meta:
                        meta_args = {a["name"]: a for a in meta["arguments"] if "name" in a}
                        for arg in args:
                            arg_name = arg.get("name")
                            if arg_name and arg_name in meta_args:
                                meta_arg = meta_args[arg_name]
                                if "default" in meta_arg:
                                    arg["default"] = meta_arg["default"]
                                if "type" in meta_arg:
                                    arg["type"] = meta_arg["type"]
                                if "inputSchema" in meta_arg:
                                    arg["inputSchema"] = meta_arg["inputSchema"]
                analysis["prompts"].append(prompt_info)
        except Exception as e:
            if self.verbose:
                hud_console.debug("Could not list prompts: " + str(e))

        # Derive "scenarios" from Environment.@scenario prompts/resources.
        # A scenario is exposed as:
        # - Prompt: name "{env}:{scenario}" with description prefix "[Setup]"
        # - Resource: uri "{env}:{scenario}" with description prefix "[Evaluate]"
        # Both prompt and resource contain meta.code with the scenario source code
        scenarios_by_id: dict[str, dict[str, Any]] = {}

        for p in analysis.get("prompts", []):
            desc = (p.get("description") or "").strip()
            if not desc.startswith("[Setup]"):
                continue
            scenario_id = p.get("name")
            if not scenario_id:
                continue
            env_name, scenario_name = ([*scenario_id.split(":", 1), ""])[:2]
            scenario_info: dict[str, Any] = {
                "id": scenario_id,
                "env": env_name,
                "name": scenario_name or scenario_id,
                "setup_description": desc,
                "arguments": p.get("arguments") or [],
                "has_setup_prompt": True,
                "has_evaluate_resource": False,
            }
            # Extract code from meta field if present
            meta = p.get("meta")
            if meta and isinstance(meta, dict) and "code" in meta:
                scenario_info["code"] = meta["code"]
            scenarios_by_id[scenario_id] = scenario_info

        for r in analysis.get("resources", []):
            desc = (r.get("description") or "").strip()
            if not desc.startswith("[Evaluate]"):
                continue
            scenario_id = r.get("uri")
            if not scenario_id:
                continue
            env_name, scenario_name = ([*scenario_id.split(":", 1), ""])[:2]
            if scenario_id not in scenarios_by_id:
                scenarios_by_id[scenario_id] = {
                    "id": scenario_id,
                    "env": env_name,
                    "name": scenario_name or scenario_id,
                    "arguments": [],
                    "has_setup_prompt": False,
                    "has_evaluate_resource": True,
                }
            scenarios_by_id[scenario_id]["evaluate_description"] = desc
            scenarios_by_id[scenario_id]["has_evaluate_resource"] = True
            # Extract code from meta field if not already present (from prompt)
            meta = r.get("meta")
            if (
                meta
                and isinstance(meta, dict)
                and "code" in meta
                and "code" not in scenarios_by_id[scenario_id]
            ):
                scenarios_by_id[scenario_id]["code"] = meta["code"]

        analysis["scenarios"] = sorted(
            scenarios_by_id.values(),
            key=lambda s: (str(s.get("env") or ""), str(s.get("name") or "")),
        )

        return analysis

    async def get_hub_tools(self, hub_name: str) -> list[str]:
        """Get all subtools for a specific hub (setup/evaluate).

        Args:
            hub_name: Name of the hub (e.g., "setup", "evaluate")

        Returns:
            List of available function names for the hub
        """
        try:
            # Read the hub's functions catalogue resource
            result = await self.read_resource(f"file:///{hub_name}/functions")
            if result and result.contents:
                # Parse the JSON list of function names
                import json

                functions = json.loads(result.contents[0].text)  # type: ignore
                return functions
        except Exception as e:
            if self.verbose:
                hud_console.debug(f"Could not read hub functions for '{hub_name}': {e}")
        return []
