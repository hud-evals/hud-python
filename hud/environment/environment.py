"""Environment class - unified MCP server and client."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Self

import mcp.types as mcp_types
from pydantic import AnyUrl  # noqa: TC002 - used at runtime in handler

from hud.environment.connectors import ConnectorsMixin
from hud.environment.integrations import IntegrationsMixin
from hud.environment.mock import MockMixin
from hud.environment.router import ConflictResolution, ToolRouter
from hud.environment.scenarios import ScenarioMixin, _safe_session_id
from hud.server.server import MCPServer
from hud.types import MCPToolResult

if TYPE_CHECKING:
    import types

    from hud.environment.connection import Connector
    from hud.eval.task import Task

__all__ = ["Environment"]

logger = logging.getLogger(__name__)

# Suppress verbose fastmcp logging
logging.getLogger("fastmcp.server.server").setLevel(logging.WARNING)
logging.getLogger("fastmcp.server.openapi").setLevel(logging.WARNING)

# Type alias for async callables (no-arg functions that return awaitable)
AsyncCallable = Callable[[], Awaitable[Any]]


class Environment(
    ConnectorsMixin,
    IntegrationsMixin,
    MockMixin,
    ScenarioMixin,
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

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize environment name to lowercase with hyphens.

        - Strips whitespace
        - Replaces spaces and underscores with hyphens
        - Lowercases the result
        - Removes any non-alphanumeric characters except hyphens
        """
        import re

        normalized = name.strip().lower()
        normalized = normalized.replace(" ", "-").replace("_", "-")
        # Keep only alphanumeric and hyphens
        normalized = re.sub(r"[^a-z0-9-]", "", normalized)
        # Collapse multiple hyphens
        normalized = re.sub(r"-+", "-", normalized)
        # Strip leading/trailing hyphens
        return normalized.strip("-") or "environment"

    def __init__(
        self,
        name: str = "environment",
        instructions: str | None = None,
        conflict_resolution: ConflictResolution = ConflictResolution.PREFIX,
        **fastmcp_kwargs: Any,
    ) -> None:
        # Normalize name to prevent casing/spacing issues
        name = self._normalize_name(name)
        super().__init__(name=name, instructions=instructions, **fastmcp_kwargs)
        self._connections: dict[str, Connector] = {}
        self._router = ToolRouter(conflict_resolution=conflict_resolution)
        # Granular routing flags - only rebuild what's invalidated
        self._tool_routing_built = False
        self._prompt_routing_built = False
        self._resource_routing_built = False
        self._in_context = False

        # Tool call queues - run after connections established
        self._setup_calls: list[tuple[str, dict[str, Any]]] = []
        self._evaluate_calls: list[tuple[str, dict[str, Any]]] = []
        self._integration_test_calls: list[tuple[str, dict[str, Any]]] = []
        # Store setup tool results for append_setup_output feature
        self._setup_results: list[MCPToolResult] = []

        # Default prompt (EvalContext has per-run prompt)
        self.prompt: str | None = None

        # Serialization support
        # _hub_config: set by connect_hub() for v5 format {"name": "hub", "include": [...]}
        # _mcp_config: set by connect_mcp_config() for v4 format {"server_name": {...}}
        self._hub_config: dict[str, Any] | None = None
        self._mcp_config: dict[str, dict[str, Any]] | None = None

        # Agent-level tool filtering (applied in as_tools(), not at connection level)
        # This allows Environment to call all tools while limiting agent visibility
        self._agent_include: list[str] | None = None
        self._agent_exclude: list[str] | None = None

        # Stable session identifier for multi-turn reuse (set by Chat).
        # When set, Connector.copy() reuses this as Environment-Id instead
        # of generating a fresh UUID, so the remote server treats all turns
        # as one session.
        self._stable_environment_id: str | None = None

        # Initialize mock state
        self._init_mock()

        # Initialize scenario state
        self._init_scenarios()

    # =========================================================================
    # Core Methods
    # =========================================================================

    def as_tools(self) -> list[mcp_types.Tool]:
        """Return tools in MCP format (base format).

        Applies scenario-level and agent-level filtering in order:
        1. Scenario-level: exclude_sources and exclude_tools remove tools
        2. Scenario-level: allowed_tools rescues specific tools back from exclusions
        3. Agent-level: _agent_include/_agent_exclude (fnmatch)

        Supports fnmatch-style wildcards (e.g., "*setup*", "browser_*").
        """
        import fnmatch

        tools = self._router.tools

        # Scenario-level exclusion (from @env.scenario(exclude_tools/exclude_sources))
        session = self._active_session
        if session:
            excluded_sources = set(session.exclude_sources) if session.exclude_sources else None
            excluded_patterns = session.exclude_tools

            if excluded_sources or excluded_patterns:
                filtered = []
                for tool in tools:
                    if excluded_sources:
                        source = self._router._tool_routing.get(tool.name, "")
                        if source in excluded_sources:
                            continue
                    if excluded_patterns and any(
                        fnmatch.fnmatch(tool.name, pat) for pat in excluded_patterns
                    ):
                        continue
                    filtered.append(tool)
                tools = filtered

            # Rescue: add back tools matching allowed_tools patterns
            allowed_patterns = session.allowed_tools
            if allowed_patterns:
                visible_names = {t.name for t in tools}
                for tool in self._router.tools:
                    if tool.name not in visible_names and any(
                        fnmatch.fnmatch(tool.name, pat) for pat in allowed_patterns
                    ):
                        tools.append(tool)

        # Apply agent-level filtering (from v4 allowed_tools/disallowed_tools)
        if self._agent_include is not None or self._agent_exclude is not None:
            filtered = []
            for tool in tools:
                # Include filter: None means include all, check if matches any pattern
                if self._agent_include is not None and not any(
                    fnmatch.fnmatch(tool.name, pattern) for pattern in self._agent_include
                ):
                    continue
                # Exclude filter: skip if tool matches any exclude pattern
                if self._agent_exclude is not None and any(
                    fnmatch.fnmatch(tool.name, pattern) for pattern in self._agent_exclude
                ):
                    continue
                filtered.append(tool)
            return filtered

        return tools

    def add_tool(self, obj: Any, **kwargs: Any) -> None:
        super().add_tool(obj, **kwargs)
        self._tool_routing_built = False  # Only invalidate tool routing

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
        result = await self._execute_tool(parsed.name, parsed.arguments or {})
        return format_result(result, parsed, fmt)

    def _connections_with_tool(self, tool_name: str) -> set[str]:
        """Get connection names that have a specific tool.

        Uses cached_tools from each Connector to check availability.
        """
        result = set()
        for name, connector in self._connections.items():
            tool_names = {t.name for t in connector.cached_tools}
            if tool_name in tool_names:
                result.add(name)
        return result

    async def _broadcast_tool(
        self,
        tool_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Broadcast a tool call to all connections that have the tool.

        Automatically filters to only connections where the tool exists
        (based on cached_tools from initial discovery).

        For internal tools (starting with _), tries ALL connections since
        internal tools are hidden from list_tools() and won't be in cached_tools.

        Args:
            tool_name: Name of the tool to call
            **kwargs: Arguments to pass to the tool

        Returns:
            Dict mapping connection name to result (or exception)
        """
        import asyncio

        # For internal tools (underscore prefix), try ALL connections since
        # they're hidden from list_tools() and won't appear in cached_tools.
        # For regular tools, only try connections that advertise the tool.
        if tool_name.startswith("_"):
            targets = set(self._connections.keys())
        else:
            targets = self._connections_with_tool(tool_name)

        results: dict[str, Any] = {}

        async def call_one(name: str) -> None:
            connector = self._connections.get(name)
            if not connector or not connector.client:
                return
            try:
                # Use connector.call_tool which expects arguments as a dict
                results[name] = await connector.call_tool(tool_name, kwargs)
                logger.debug("Broadcast '%s' to '%s' succeeded", tool_name, name)
            except Exception as e:
                results[name] = e
                logger.debug("Broadcast '%s' to '%s' failed: %s", tool_name, name, e)

        await asyncio.gather(*[call_one(n) for n in targets], return_exceptions=True)
        return results

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

    async def __aenter__(self) -> Self:
        """Connect all connectors, build routing, run setup tools."""
        self._in_context = True

        # Connect to all servers and fetch tools/prompts/resources in parallel
        sem = asyncio.Semaphore(self.MAX_CONCURRENT_CONNECTIONS)
        errors: list[tuple[str, Exception]] = []

        async def connect_one(name: str, conn: Connector) -> None:
            async with sem:
                try:
                    await conn.connect()
                    # Batch fetch all MCP primitives in parallel for performance
                    await asyncio.gather(
                        conn.list_tools(),
                        conn.list_prompts(),
                        conn.list_resources(),
                    )
                except Exception as e:
                    errors.append((name, e))

        if self._connections:
            await asyncio.gather(*[connect_one(n, c) for n, c in self._connections.items()])
            if errors:
                for conn in self._connections.values():
                    if conn.is_connected:
                        await conn.disconnect()
                name, err = errors[0]
                str_err = str(err).replace("Client failed to connect: ", "")  # Strip from FastMCP
                raise ConnectionError(f"Failed to connect to {name}: {str_err}") from err

        await self._build_routing()

        # Setup tool calls (after connections) - abort if any setup tool fails
        # Store results for append_setup_output feature
        self._setup_results = []
        for name, args in self._setup_calls:
            result = await self._execute_tool(name, args)
            self._setup_results.append(result)
            if result.isError:
                # Extract error message from result content
                error_msg = "Setup tool failed"
                if result.content:
                    for block in result.content:
                        if isinstance(block, mcp_types.TextContent):
                            error_msg = block.text
                            break
                # Clean up connections before raising (since __aexit__ won't be called)
                for conn in self._connections.values():
                    if conn.is_connected:
                        await conn.disconnect()
                raise RuntimeError(f"Setup tool '{name}' failed: {error_msg}")

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
                # Record 0.0 for failed evaluate tools so they affect the average
                rewards.append(0.0)

        # Store average reward from evaluate tools
        self._evaluate_reward: float | None = None
        if rewards:
            self._evaluate_reward = sum(rewards) / len(rewards)

        self._in_context = False
        if self._connections:
            await asyncio.gather(*[c.disconnect() for c in self._connections.values()])
        self._router.clear()
        self._tool_routing_built = False
        self._prompt_routing_built = False
        self._resource_routing_built = False
        self._scenario_sessions = {}  # Clear stale scenario state

    async def run_async(
        self,
        transport: Literal["stdio", "http", "sse"] | None = None,
        show_banner: bool = True,
        **transport_kwargs: Any,
    ) -> None:
        """Run the MCP server, auto-connecting all connectors first.

        This ensures that tools from external MCP servers (via connect_mcp_config)
        are discovered and available when the server starts.
        """
        async with self:  # Connect all connectors via __aenter__
            await super().run_async(
                transport=transport, show_banner=show_banner, **transport_kwargs
            )

    async def _build_routing(self) -> None:
        """Build routing for tools, prompts, and resources in parallel.

        Only rebuilds what's actually invalidated for performance.
        """
        tasks = []
        if not self._tool_routing_built:
            tasks.append(self._build_tool_routing())
        if not self._prompt_routing_built:
            tasks.append(self._build_prompt_routing())
        if not self._resource_routing_built:
            tasks.append(self._build_resource_routing())
        if tasks:
            await asyncio.gather(*tasks)

    async def _build_tool_routing(self) -> None:
        """Build tool routing from local tools and connection caches."""
        local_tools_list = await self._local_provider.list_tools()
        local_tools = list(local_tools_list)
        self._router.build(
            local_tools=[t.to_mcp_tool() for t in local_tools],
            connections=self._connections,
            connection_order=list(self._connections.keys()),
        )
        # Populate mock schemas for auto-generated mock values
        self._populate_mock_schemas()
        self._tool_routing_built = True

    async def _build_prompt_routing(self) -> None:
        """Build prompt routing from local prompts and connections."""
        local_prompts_list = await self._local_provider.list_prompts()
        local_prompts = [p.to_mcp_prompt() for p in local_prompts_list]
        self._router.build_prompts(local_prompts, self._connections)
        self._prompt_routing_built = True

    # FastMCP server internals expect list_prompts() to return FastMCP prompt
    # objects with a .version attribute. HUD's router, however, builds routing
    # from mcp.types.Prompt definitions. If we return the router's MCP prompt
    # objects directly from list_prompts(), FastMCP 3.x crashes while handling
    # prompts/list with: "'Prompt' object has no attribute 'version'".
    # Keep the router path and server path split so each layer gets the prompt
    # shape it expects.
    async def _list_mcp_prompts(self) -> list[mcp_types.Prompt]:
        """Return MCP prompt definitions for HUD's internal routing logic."""
        if self._connections:
            await asyncio.gather(*[c.list_prompts() for c in self._connections.values()])
        await self._build_prompt_routing()
        return self._router.prompts

    @staticmethod
    def _to_fastmcp_prompt(prompt: mcp_types.Prompt) -> Any:
        """Convert an MCP prompt definition into a FastMCP prompt component."""
        from fastmcp.prompts.prompt import Prompt, PromptArgument

        arguments = [
            PromptArgument(
                name=arg.name,
                description=arg.description,
                required=bool(arg.required),
            )
            for arg in (prompt.arguments or [])
        ]
        return Prompt(
            name=prompt.name,
            version=None,
            title=prompt.title,
            description=prompt.description,
            icons=prompt.icons,
            arguments=arguments or None,
            meta=getattr(prompt, "meta", None),
        )

    async def _build_resource_routing(self) -> None:
        """Build resource routing from local resources and connections."""
        local_resources_list = await self._local_provider.list_resources()
        local_resources = [r.to_mcp_resource() for r in local_resources_list]
        self._router.build_resources(local_resources, self._connections)
        self._resource_routing_built = True

    # =========================================================================
    # MCP Protocol Overrides - Include connector tools in MCP responses
    # =========================================================================

    def _setup_handlers(self) -> None:
        """Override FastMCP to register our custom handlers for tools and prompts.

        FastMCP 3.x handlers expect (self, request) -> Result signatures.
        We wrap our handlers to match.
        """
        super()._setup_handlers()

        # Re-register with correct FastMCP 3.x signatures
        @self._mcp_server.list_tools()
        async def _list_tools_handler(
            request: Any = None,
        ) -> mcp_types.ListToolsResult:
            tools = await self._env_list_tools()
            return mcp_types.ListToolsResult(tools=tools)

        @self._mcp_server.call_tool()
        async def _call_tool_handler(
            name: str, arguments: dict[str, Any] | None = None
        ) -> list[Any]:
            return await self._env_call_tool(name, arguments)

        @self._mcp_server.get_prompt()
        async def _get_prompt_handler(
            name: str, arguments: dict[str, str] | None = None
        ) -> mcp_types.GetPromptResult:
            return await self._env_get_prompt(name, arguments)

        @self._mcp_server.list_prompts()
        async def _list_prompts_handler(
            request: Any = None,
        ) -> mcp_types.ListPromptsResult:
            # This handler must return MCP prompt definitions. Returning FastMCP
            # prompt components here causes ListPromptsResult validation errors.
            prompts = await self._env_list_prompts()
            return mcp_types.ListPromptsResult(prompts=prompts)

        @self._mcp_server.list_resources()
        async def _list_resources_handler(
            request: Any = None,
        ) -> mcp_types.ListResourcesResult:
            resources = await self._env_list_resources()
            return mcp_types.ListResourcesResult(resources=resources)

        @self._mcp_server.read_resource()
        async def _read_resource_handler(
            uri: AnyUrl, **kwargs: Any
        ) -> mcp_types.ReadResourceResult:
            contents = await self.read_resource(str(uri), **kwargs)
            return mcp_types.ReadResourceResult(contents=contents)

    async def _env_list_tools(self) -> list[mcp_types.Tool]:
        """Return all tools including those from connectors."""
        if not self._tool_routing_built:
            await self._build_tool_routing()
        return self._router.tools

    async def _env_list_prompts(self) -> list[mcp_types.Prompt]:
        """Return all prompts including those from connectors."""
        return await self._list_mcp_prompts()

    async def _env_list_resources(self) -> list[mcp_types.Resource]:
        """Return all resources including those from connectors."""
        if not self._resource_routing_built:
            await self._build_resource_routing()
        return self._router.resources

    async def _env_call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Any]:
        """Route tool calls through our router (handles both local and connector tools)."""
        args = dict(arguments or {})

        # Extract trace context propagated via MCP request (meta or arguments)
        trace_id = args.pop("_hud_trace_id", None)
        meta = kwargs.get("_meta") or kwargs.get("meta")
        if not trace_id and isinstance(meta, dict):
            trace_id = meta.get("_hud_trace_id") or meta.get("trace_id")

        # FastMCP does not forward request meta as call_tool kwargs.
        # Read request_ctx directly to extract _hud_trace_id from MCP metadata.
        if not trace_id:
            try:
                from mcp.server.lowlevel.server import request_ctx

                req_meta = getattr(request_ctx.get(), "meta", None)
                if req_meta is not None:
                    extra = getattr(req_meta, "model_extra", None) or {}
                    trace_id = extra.get("_hud_trace_id") or extra.get("trace_id")
            except (ImportError, LookupError):
                pass

        if trace_id:
            from hud.eval.context import set_trace_context

            with set_trace_context(trace_id):
                result = await self._execute_tool(name, args)
        else:
            result = await self._execute_tool(name, args)

        return result.content or []

    async def _env_get_prompt(
        self, name: str, arguments: dict[str, str] | None = None, **kwargs: Any
    ) -> mcp_types.GetPromptResult:
        """Handle get_prompt requests, routing scenario prompts through run_scenario_setup.

        FastMCP 3.x's FunctionPrompt.render() filters kwargs to only those
        explicitly named in the handler's signature, which strips scenario
        args (user_id, items, etc.) because our handler uses **kwargs.
        Bypass that by calling run_scenario_setup directly for scenario
        prompts (those containing ':').
        """
        if ":" in name and name.split(":")[0] in (self.name, getattr(self, "_source_env_name", "")):
            # Local scenario prompt — run setup directly
            scenario_name = name.split(":", 1)[1]
            str_args = {k: v for k, v in (arguments or {}).items()}

            # Extract MCP session ID for multi-client isolation using the same
            # helper as scenario prompt/resource handlers.
            session_id = _safe_session_id(None)

            prompt_text = await self.run_scenario_setup(
                scenario_name, str_args, session_id=session_id
            )
            if not prompt_text:
                raise ValueError(f"Scenario '{name}' returned empty prompt")
            return mcp_types.GetPromptResult(
                messages=[
                    mcp_types.PromptMessage(
                        role="user",
                        content=mcp_types.TextContent(type="text", text=prompt_text),
                    )
                ]
            )

        # Non-scenario prompt or remote — delegate to parent
        return await self.get_prompt(name, arguments)

    # =========================================================================
    # Tool Operations
    # =========================================================================

    async def list_tools(self, **kwargs: Any) -> list[mcp_types.Tool]:
        """Refresh tools from all connections and rebuild tool routing."""
        if self._connections:
            await asyncio.gather(*[c.list_tools() for c in self._connections.values()])
        await self._build_tool_routing()
        return self._router.tools

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Execute a tool by name. Routes to local or remote handler.

        If mock mode is enabled, returns a mock result instead of executing.
        """
        # Check mock mode first
        if self._mock_mode:
            logger.debug("Mock mode: returning mock result for tool %s", name)
            return self._get_mock_result(name, arguments)

        # Rebuild tool routing if invalidated (e.g., after add_tool)
        if not self._tool_routing_built:
            await self._build_tool_routing()

        if self._router.is_local(name):
            # Call via FastMCP's call_tool (parent class) which handles
            # context injection for elicitation, set_state, etc.
            # run_middleware=False because this is an internal call, not an
            # MCP protocol message.  The middleware chain's call_next lambda
            # resolves to self.call_tool which has a different (multi-format)
            # signature and would TypeError with positional (name, arguments).
            from fastmcp import FastMCP
            result = await FastMCP.call_tool(self, name, arguments, run_middleware=False)
            return MCPToolResult(
                content=result.content, structuredContent=result.structured_content
            )

        connection_name = self._router.get_connection(name)
        if connection_name:
            conn = self._connections[connection_name]
            result = await conn.call_tool(name, arguments)
            return MCPToolResult(
                content=result.content,
                isError=result.isError,
                structuredContent=result.structuredContent,
            )

        raise ValueError(f"Tool not found: {name}")

    # =========================================================================
    # Resource Operations
    # =========================================================================

    async def list_resources(self) -> list[mcp_types.Resource]:
        """Refresh resources from all connections and rebuild resource routing."""
        if self._connections:
            await asyncio.gather(*[c.list_resources() for c in self._connections.values()])
        await self._build_resource_routing()
        return self._router.resources

    async def read_resource(
        self, uri: str, **kwargs: Any
    ) -> list[mcp_types.TextResourceContents | mcp_types.BlobResourceContents]:
        """Read a resource by URI using router for connection lookup."""
        from pydantic import AnyUrl

        # Ensure resource routing is built
        if not self._resource_routing_built:
            await self._build_resource_routing()

        # Use router to find which connection has this resource
        conn_name = self._router.get_resource_connection(uri)

        if conn_name is None:
            # Local resource -- read via local provider
            try:
                resource = await self._local_provider.get_resource(uri)
                if resource is None:
                    raise ValueError(f"Resource not found: {uri}")
                result = await resource.read()
                resource_uri = AnyUrl(uri)

                content = getattr(result, "content", result)
                if isinstance(content, str):
                    return [mcp_types.TextResourceContents(uri=resource_uri, text=content)]
                if hasattr(content, "text"):
                    return [mcp_types.TextResourceContents(uri=resource_uri, text=content.text)]  # type: ignore[union-attr]
                import base64

                raw = content if isinstance(content, bytes) else str(content).encode()
                return [
                    mcp_types.BlobResourceContents(
                        uri=resource_uri, blob=base64.b64encode(raw).decode()
                    )
                ]
            except Exception as e:
                logger.debug("Local resource read failed for %s: %s", uri, e)
                raise ValueError(f"Resource not found: {uri}") from e
        else:
            # Remote resource
            conn = self._connections.get(conn_name)
            if conn is None:
                raise ValueError(f"Connection '{conn_name}' not found for resource '{uri}'")
            return await conn.read_resource(uri)

    # =========================================================================
    # Prompt Operations
    # =========================================================================

    async def list_prompts(self) -> list[Any]:
        """List prompts as FastMCP prompt components for server-side MCP operations."""
        prompts = await self._list_mcp_prompts()
        return [self._to_fastmcp_prompt(prompt) for prompt in prompts]

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> mcp_types.GetPromptResult:
        """Get a prompt by name using router for connection lookup."""
        # Ensure prompt routing is built
        if not self._prompt_routing_built:
            await self._build_prompt_routing()

        # Use router to find which connection has this prompt
        conn_name = self._router.get_prompt_connection(name)

        if conn_name is None:
            # Local prompt -- render via FastMCP's render_prompt (parent class)
            try:
                from fastmcp import FastMCP

                return await FastMCP.render_prompt(self, name, arguments or {})  # type: ignore[return-value]
            except Exception as e:
                raise ValueError(f"Prompt not found: {name}") from e
        else:
            # Remote prompt
            conn = self._connections.get(conn_name)
            if conn is None:
                raise ValueError(f"Connection '{conn_name}' not found for prompt '{name}'")
            return await conn.get_prompt(name, arguments)

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

    # =========================================================================
    # Serialization
    # =========================================================================

    @property
    def is_serializable(self) -> bool:
        """True if environment can be serialized (no local tools/scenarios).

        For v5 format: requires hub config from connect_hub()
        For v4 format: requires mcp_config, prompt, AND evaluate_tool
        """
        # Check for local tools (registered via @env.tool)
        if self._router._local_tool_names:
            return False
        # Check for local scenarios (registered via @env.scenario)
        if getattr(self, "_scenarios", {}):
            return False
        # v5 hub format
        if self._hub_config is not None:
            return True
        # v4 format requires mcp_config + prompt + evaluate_tool
        if self._mcp_config is not None:
            return bool(self.prompt and self._evaluate_calls)
        return False

    def to_config(self) -> dict[str, Any]:
        """Serialize environment config for remote submission.

        Returns the config in either v5 format (hub-based) or v4 format (legacy).
        For v4 format, automatically includes prompt, setup_tool, and evaluate_tool
        from the Environment's state.

        Returns:
            dict: Serializable config

        Raises:
            ValueError: If environment has local tools/scenarios that can't be serialized

        Example:
            ```python
            # v5 hub-based
            env = Environment("my").connect_hub("browser", include=["navigate"])
            env.to_config()  # {"name": "browser", "include": ["navigate"]}

            # v4 legacy (from Task.from_v4())
            task = Task.from_v4(legacy_task)
            task.env.to_config()  # {"prompt": "...", "mcp_config": {...}, ...}
            ```
        """
        if self._router._local_tool_names:
            raise ValueError(
                f"Cannot serialize Environment with local tools: "
                f"{list(self._router._local_tool_names)}. "
                "Local tools require local execution. For remote submission, "
                "use dict config or connect to a remote hub."
            )
        if getattr(self, "_scenarios", {}):
            raise ValueError(
                f"Cannot serialize Environment with local scenarios: "
                f"{list(self._scenarios.keys())}. "
                "Local scenarios require local execution. For remote submission, "
                "define scenarios on the remote environment."
            )

        # v5 hub-based format
        if self._hub_config is not None:
            return self._hub_config.copy()

        # v4 legacy format - requires mcp_config, prompt, AND evaluate_tool
        if self._mcp_config is not None:
            # Validate required fields for v4 format
            if not self.prompt:
                raise ValueError(
                    "Cannot serialize v4 Environment without prompt. "
                    "Set env.prompt before serializing."
                )
            if not self._evaluate_calls:
                raise ValueError(
                    "Cannot serialize v4 Environment without evaluate_tool. "
                    "Use env.evaluate_tool() to define evaluation criteria."
                )

            config: dict[str, Any] = {
                "prompt": self.prompt,
                "mcp_config": self._mcp_config,
                "evaluate_tool": [
                    {"name": name, "arguments": args} for name, args in self._evaluate_calls
                ],
            }
            if self._setup_calls:
                config["setup_tool"] = [
                    {"name": name, "arguments": args} for name, args in self._setup_calls
                ]
            return config

        raise ValueError(
            "Cannot serialize Environment without config. "
            "Use connect_hub() for v5 tasks or connect_mcp_config() for legacy tasks."
        )

    def __repr__(self) -> str:
        return f"Environment({self.name!r}, connections={list(self._connections.keys())})"

    # =========================================================================
    # Task Creation
    # =========================================================================

    def __call__(
        self,
        scenario: str | None = None,
        **args: Any,
    ) -> Task:
        """Create a Task from this environment.

        Returns a Task that can be passed to hud.eval() for orchestration.

        Args:
            scenario: Scenario name to run (from @env.scenario). Optional for v4 legacy.
            **args: Arguments for the scenario

        Returns:
            Task: A runnable evaluation unit

        Example:
            ```python
            env = Environment("my-env").connect_hub("browser")


            @env.scenario()
            async def checkout(user_id: str):
                yield "Complete checkout"
                yield 1.0


            # Single task via hud.eval
            async with hud.eval(env("checkout", user_id="alice")) as ctx:
                await agent.run(ctx.prompt)

            # Multiple tasks with variants
            tasks = [env("checkout", user_id="alice"), env("checkout", user_id="bob")]
            async with hud.eval(tasks, variants={"model": ["gpt-4o"]}, group=4) as ctx:
                ...
            ```
        """
        from hud.eval.task import Task

        return Task(
            env=self,
            scenario=scenario,
            args=args,
        )
