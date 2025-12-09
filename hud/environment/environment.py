"""Environment class - unified MCP server and client."""

from __future__ import annotations

import asyncio
import logging
import types
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import mcp.types as mcp_types

from hud.environment.connection import Connector
from hud.environment.connectors import ConnectorsMixin
from hud.environment.integrations import IntegrationsMixin
from hud.environment.mock import MockMixin
from hud.environment.router import ConflictResolution, ToolRouter
from hud.server.server import MCPServer
from hud.trace.mixin import TraceMixin
from hud.types import MCPToolResult

__all__ = ["Environment"]

logger = logging.getLogger(__name__)

# Type alias for async callables (no-arg functions that return awaitable)
AsyncCallable = Callable[[], Awaitable[Any]]


class Environment(
    ConnectorsMixin,
    IntegrationsMixin,
    MockMixin,
    TraceMixin,
    MCPServer,
):
    """Unified MCP environment that acts as both server and client.

    Features:
        - Define local tools with @env.tool decorator
        - Connect to HUD Hub, URLs, or mcp_config dicts
        - Automatic tool routing (local vs remote)
        - Format tools for any LLM provider
        - Integrate with popular agent frameworks
        - Mock mode for testing without real connections

    Connector methods (connect to sources):
        connect_hub(name) - HUD Hub environment
        connect_url(url) - MCP server via URL
        connect_mcp(config) - Single mcp_config server
        connect_mcp_config(mcp_config) - Multiple mcp_config servers
        connect_task(slug) - Load task from platform by slug
        connect_image(image) - Docker image via stdio
        connect_fastapi(app) - Mount FastAPI app as MCP server
        connect_openapi(spec) - Mount OpenAPI spec as MCP server
        connect_server(server) - Mount MCPServer/FastMCP directly

    Mock methods (for testing):
        mock() - Enable mock mode, all tools return mock values
        unmock() - Disable mock mode
        mock_tool(name, output) - Set specific mock output for a tool
        is_mock - Check if mock mode is enabled

    OpenAI integrations:
        as_openai_chat_tools() - Chat Completions format
        as_openai_responses_tools() - Responses API format
        as_openai_agent_tools() - Agents SDK (requires openai-agents)

    Anthropic/Claude integrations:
        as_claude_tools() - Claude API format
        as_claude_programmatic_tools() - Programmatic tool use
        as_anthropic_runner() - Tool runner (requires anthropic)

    Google/Gemini integrations:
        as_gemini_tools() - Gemini format
        as_gemini_tool_config() - Tool execution config

    LangChain integrations:
        as_langchain_tools() - StructuredTools (requires langchain-core)

    Example:
        ```python
        env = Environment("my-env")


        @env.tool
        def greet(name: str) -> str:
            return f"Hello, {name}!"


        env.connect_hub("browser", prefix="browser")

        async with env:
            # Get tools in any format
            openai_tools = env.as_openai_chat_tools()
            claude_tools = env.as_claude_tools()

            # Call tools - automatically routed
            result = await env.call_tool("greet", name="World")

            # Or pass provider-specific format - auto-detected
            result = await env.call_tool(response.choices[0].message.tool_calls[0])

        # Mock mode for testing
        env.mock()
        env.mock_tool("browser_navigate", "Navigation successful")
        async with env:
            result = await env.call_tool("browser_navigate", url="https://example.com")
            # Returns mock value instead of actually navigating
        ```
    """

    MAX_CONCURRENT_CONNECTIONS = 10

    def __init__(
        self,
        name: str = "environment",
        instructions: str | None = None,
        conflict_resolution: ConflictResolution = ConflictResolution.PREFIX,
        **fastmcp_kwargs: Any,
    ) -> None:
        super().__init__(name=name, instructions=instructions, **fastmcp_kwargs)
        self._connections: dict[str, Connector] = {}
        self._router = ToolRouter(conflict_resolution=conflict_resolution)
        self._in_context = False

        # Tool call queues - run after connections established
        self._setup_calls: list[tuple[str, dict[str, Any]]] = []
        self._evaluate_calls: list[tuple[str, dict[str, Any]]] = []

        # Task prompt - set by connect_task or manually
        self.prompt: str | None = None

        # Track which lifecycle tools we've warned about (only warn once per tool)
        self._warned_lifecycle_tools: set[str] = set()

        # Initialize mock state
        self._init_mock()

    # =========================================================================
    # Core Methods
    # =========================================================================

    def as_tools(self) -> list[mcp_types.Tool]:
        """Return tools in MCP format (base format)."""
        return self._router.tools

    async def call_tool(self, call: Any, /, **kwargs: Any) -> Any:
        """Call a tool, auto-detecting format and returning matching result format.

        Accepts any format:
            - String with kwargs: call_tool("navigate", url="...")
            - Tuple: call_tool(("navigate", {"url": "..."}))
            - MCPToolCall: call_tool(MCPToolCall(name="navigate", ...))
            - OpenAI: call_tool(response.choices[0].message.tool_calls[0])
            - Claude: call_tool(response.content[0])  # tool_use block
            - Gemini: call_tool(response.candidates[0].content.parts[0])

        Returns:
            Result formatted to match input format (OpenAI -> OpenAI tool message, etc.)
        """
        from hud.environment.utils import format_result, parse_tool_call

        # Parse the tool call (kwargs merged when call is string)
        parsed, fmt = parse_tool_call(call, **kwargs)
        self._check_lifecycle_warning(parsed.name)
        result = await self._execute_tool(parsed.name, parsed.arguments or {})
        return format_result(result, parsed, fmt)

    def _check_lifecycle_warning(self, name: str) -> None:
        """Warn once if calling a setup/evaluate tool manually."""
        if name in self._warned_lifecycle_tools:
            return
        setup = {n for n, _ in self._setup_calls}
        evaluate = {n for n, _ in self._evaluate_calls}
        if name not in setup and name not in evaluate:
            return
        self._warned_lifecycle_tools.add(name)
        phase = "setup" if name in setup else "evaluate"
        logger.warning(
            "Tool '%s' is a %s tool (runs automatically). Manual call may duplicate.",
            name,
            phase,
        )

    async def call_tools(self, calls: Any) -> list[Any]:
        """Call multiple tools, returning results in matching formats."""
        if calls is None:
            return []
        if not isinstance(calls, list):
            return [await self.call_tool(calls)]

        # Filter to tool calls only (skip text blocks, etc.)
        tool_calls = []
        for call in calls:
            t = call.get("type") if isinstance(call, dict) else getattr(call, "type", None)
            if t is None or t in ("tool_use", "function"):
                tool_calls.append(call)

        return await asyncio.gather(*[self.call_tool(c) for c in tool_calls])

    # =========================================================================
    # Lifecycle Configuration
    # =========================================================================

    def setup_tool(self, call: Any, /, **kwargs: Any) -> Environment:
        """Add a tool call to execute after connections are established."""
        from hud.environment.utils import parse_tool_call

        if isinstance(call, str) and kwargs:
            self._setup_calls.append((call, kwargs))
        else:
            parsed, _ = parse_tool_call(call)
            self._setup_calls.append((parsed.name, parsed.arguments or {}))
        return self

    def evaluate_tool(self, call: Any, /, **kwargs: Any) -> Environment:
        """Add a tool call to execute before disconnecting."""
        from hud.environment.utils import parse_tool_call

        if isinstance(call, str) and kwargs:
            self._evaluate_calls.append((call, kwargs))
        else:
            parsed, _ = parse_tool_call(call)
            self._evaluate_calls.append((parsed.name, parsed.arguments or {}))
        return self

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> Environment:
        """Connect all connectors, build routing, run setup tools."""
        self._in_context = True

        # Connect to all servers (on_connect callbacks run first within connect())
        sem = asyncio.Semaphore(self.MAX_CONCURRENT_CONNECTIONS)
        errors: list[tuple[str, Exception]] = []

        async def connect_one(name: str, conn: Connector) -> None:
            async with sem:
                try:
                    await conn.connect()
                    await conn.list_tools()
                except Exception as e:
                    errors.append((name, e))

        if self._connections:
            await asyncio.gather(*[connect_one(n, c) for n, c in self._connections.items()])
            if errors:
                for conn in self._connections.values():
                    if conn.is_connected:
                        await conn.disconnect()
                name, err = errors[0]
                raise ConnectionError(f"Failed to connect to {name}") from err

        await self._build_routing()

        # Setup tool calls (after connections)
        for name, args in self._setup_calls:
            await self._execute_tool(name, args)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Run evaluate tools, exit queue, then disconnect."""
        from hud.agents.base import find_reward

        # Evaluate tool calls and collect rewards
        rewards: list[float] = []
        for name, args in self._evaluate_calls:
            try:
                result = await self._execute_tool(name, args)
                rewards.append(find_reward(result))
            except Exception as e:
                logger.warning("Evaluate tool %s failed: %s", name, e)

        # Store average reward from evaluate tools
        self._evaluate_reward: float | None = None
        if rewards:
            self._evaluate_reward = sum(rewards) / len(rewards)

        self._in_context = False
        if self._connections:
            await asyncio.gather(*[c.disconnect() for c in self._connections.values()])
        self._router.clear()

    async def _build_routing(self) -> None:
        """Build tool routing from local tools and connection caches."""
        local_tools = await self._tool_manager.list_tools()
        self._router.build(
            local_tools=[t.to_mcp_tool() for t in local_tools],
            connections=self._connections,
            connection_order=list(self._connections.keys()),
        )
        # Populate mock schemas for auto-generated mock values
        self._populate_mock_schemas()

    # =========================================================================
    # Tool Operations
    # =========================================================================

    async def list_tools(self) -> list[mcp_types.Tool]:
        """Refresh tools from all connections and rebuild routing."""
        if self._connections:
            await asyncio.gather(*[c.list_tools() for c in self._connections.values()])
        await self._build_routing()
        return self._router.tools

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Execute a tool by name. Routes to local or remote handler.

        If mock mode is enabled, returns a mock result instead of executing.
        """
        # Check mock mode first
        if self._mock_mode:
            logger.debug("Mock mode: returning mock result for tool %s", name)
            return self._get_mock_result(name, arguments)

        if self._router.is_local(name):
            # Call tool manager directly to avoid FastMCP context requirement
            result = await self._tool_manager.call_tool(name, arguments)
            return MCPToolResult(content=result.content, isError=False)

        connection_name = self._router.get_connection(name)
        if connection_name:
            conn = self._connections[connection_name]
            result = await conn.call_tool(name, arguments)
            return MCPToolResult(content=result.content, isError=result.isError)

        raise ValueError(f"Tool not found: {name}")

    # =========================================================================
    # Resource Operations
    # =========================================================================

    async def list_resources(self) -> list[mcp_types.Resource]:
        """List all resources (local + remote)."""
        local = await self._resource_manager.list_resources()
        resources: list[mcp_types.Resource] = [r.to_mcp_resource() for r in local]

        if self._connections:
            results = await asyncio.gather(
                *[c.list_resources() for c in self._connections.values()], return_exceptions=True
            )
            for r in results:
                if isinstance(r, list):
                    resources.extend(r)

        return resources

    async def read_resource(
        self, uri: str
    ) -> list[mcp_types.TextResourceContents | mcp_types.BlobResourceContents]:
        """Read a resource by URI (tries local first, then remote)."""
        from pydantic import AnyUrl

        try:
            result = await self._resource_manager.read_resource(uri)
            resource_uri = AnyUrl(uri)
            if isinstance(result, str):
                return [mcp_types.TextResourceContents(uri=resource_uri, text=result)]
            import base64

            return [
                mcp_types.BlobResourceContents(
                    uri=resource_uri, blob=base64.b64encode(result).decode()
                )
            ]
        except Exception:
            pass

        for conn in self._connections.values():
            try:
                return await conn.read_resource(uri)
            except Exception:
                continue

        raise ValueError(f"Resource not found: {uri}")

    # =========================================================================
    # Prompt Operations
    # =========================================================================

    async def list_prompts(self) -> list[mcp_types.Prompt]:
        """List all prompts (local + remote)."""
        local = await self._prompt_manager.list_prompts()
        prompts: list[mcp_types.Prompt] = [p.to_mcp_prompt() for p in local]

        if self._connections:
            results = await asyncio.gather(
                *[c.list_prompts() for c in self._connections.values()], return_exceptions=True
            )
            for r in results:
                if isinstance(r, list):
                    prompts.extend(r)

        return prompts

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> mcp_types.GetPromptResult:
        """Get a prompt by name (tries local first, then remote)."""
        try:
            return await self._prompt_manager.render_prompt(name, arguments or {})
        except Exception:
            pass

        for conn in self._connections.values():
            try:
                return await conn.get_prompt(name, arguments)
            except Exception:
                continue

        raise ValueError(f"Prompt not found: {name}")

    # =========================================================================
    # Server Methods
    # =========================================================================

    def serve(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "streamable-http",
        host: str = "0.0.0.0",  # noqa: S104
        port: int = 8000,
        **kwargs: Any,
    ) -> None:
        """Start serving as an MCP server."""
        self.run(transport=transport, host=host, port=port, **kwargs)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def connections(self) -> dict[str, Connector]:
        return self._connections

    @property
    def is_connected(self) -> bool:
        return self._in_context

    @property
    def is_parallelizable(self) -> bool:
        """True if all connections are remote (can spawn multiple instances)."""
        if not self._connections:
            return True  # No connections = can parallelize (local tools only)
        return all(conn.is_remote for conn in self._connections.values())

    @property
    def local_connections(self) -> list[str]:
        """Names of local (non-parallelizable) connections."""
        return [name for name, conn in self._connections.items() if conn.is_local]

    def _get_env_config(self) -> dict[str, Any] | None:
        """Get serializable environment configuration for trace storage.

        Returns EnvConfig-compatible dict with:
        - name: Environment name
        - hubs: List of hub configs (connect_hub calls)
        - setup_tools: Tools to run after connection (MCPToolCall format)
        - evaluate_tools: Tools to run before disconnection (MCPToolCall format)
        """
        hub_configs = getattr(self, "_hub_configs", [])

        # Convert setup/evaluate calls to MCPToolCall format
        setup_tools = [{"name": name, "arguments": args} for name, args in self._setup_calls]
        evaluate_tools = [{"name": name, "arguments": args} for name, args in self._evaluate_calls]

        # Only return config if there's something to store
        if not hub_configs and not setup_tools and not evaluate_tools:
            return None

        return {
            "name": self.name,
            "hubs": hub_configs,
            "setup_tools": setup_tools,
            "evaluate_tools": evaluate_tools,
        }

    @property
    def _all_hubs(self) -> bool:
        """True if all tools came from connect_hub (fully reproducible).

        Returns False if there are:
        - Local tools (@env.tool, connect_fastapi, connect_openapi, connect_server)
        - Non-hub connections (connect_url, connect_mcp, connect_image, etc.)
        """
        hub_configs = getattr(self, "_hub_configs", [])

        # Check for local tools (mounted servers, @env.tool)
        # _tool_manager comes from MCPServer base class
        local_tool_count = len(self._tool_manager._tools) if hasattr(self, "_tool_manager") else 0
        if local_tool_count > 0:
            return False

        # No hubs and no connections = trivially all hubs (empty env)
        if not hub_configs and not self._connections:
            return True

        # Has connections but no hubs = not all hubs
        if not hub_configs:
            return False

        # Compare hub count to connection count
        return len(hub_configs) >= len(self._connections)

    def __repr__(self) -> str:
        return f"Environment({self.name!r}, connections={list(self._connections.keys())})"
