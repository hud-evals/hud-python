"""Tests for Environment class - context manager, resources, prompts, prompt feature."""

from __future__ import annotations

import pytest


class TestEnvironmentPrompt:
    """Tests for Environment.prompt feature."""

    def test_prompt_defaults_to_none(self) -> None:
        """Environment.prompt defaults to None."""
        from hud.environment import Environment

        env = Environment("test")
        assert env.prompt is None

    def test_prompt_can_be_set(self) -> None:
        """Environment.prompt can be set manually."""
        from hud.environment import Environment

        env = Environment("test")
        env.prompt = "Navigate to google.com"
        assert env.prompt == "Navigate to google.com"


class TestEnvironmentContextManager:
    """Tests for Environment async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_sets_in_context_flag(self) -> None:
        """Context manager sets _in_context flag."""
        from hud.environment import Environment

        env = Environment("test")

        assert env._in_context is False

        async with env:
            assert env._in_context is True

        assert env._in_context is False

    @pytest.mark.asyncio
    async def test_context_manager_no_connections(self) -> None:
        """Context manager works with no connections."""
        from hud.environment import Environment

        env = Environment("test")

        async with env:
            # Should work without connections
            pass


class TestEnvironmentResources:
    """Tests for Environment resource operations."""

    @pytest.mark.asyncio
    async def test_list_resources_empty(self) -> None:
        """list_resources returns empty list when no resources."""
        from hud.environment import Environment

        env = Environment("test")

        async with env:
            resources = await env.list_resources()

        assert resources == []

    @pytest.mark.asyncio
    async def test_read_resource_not_found(self) -> None:
        """read_resource raises when resource not found."""
        from hud.environment import Environment

        env = Environment("test")

        async with env:
            with pytest.raises(ValueError, match="Resource not found"):
                await env.read_resource("file://nonexistent.txt")


class TestEnvironmentPrompts:
    """Tests for Environment prompt operations (MCP prompts, not task prompt)."""

    @pytest.mark.asyncio
    async def test_list_prompts_empty(self) -> None:
        """list_prompts returns empty list when no prompts."""
        from hud.environment import Environment

        env = Environment("test")

        async with env:
            prompts = await env.list_prompts()

        assert prompts == []

    @pytest.mark.asyncio
    async def test_list_prompts_returns_fastmcp_prompt_components(self) -> None:
        """list_prompts returns FastMCP prompt objects with version attr."""
        import mcp.types as mcp_types

        from hud.environment import Environment

        env = Environment("test")

        async def fake_list_mcp_prompts() -> list[mcp_types.Prompt]:
            return [
                mcp_types.Prompt(
                    name="test:prompt",
                    description="Prompt description",
                    arguments=[
                        mcp_types.PromptArgument(
                            name="foo",
                            description="Foo arg",
                            required=True,
                        )
                    ],
                )
            ]

        env._list_mcp_prompts = fake_list_mcp_prompts  # type: ignore[method-assign]

        prompts = await env.list_prompts()

        assert len(prompts) == 1
        assert prompts[0].name == "test:prompt"
        assert hasattr(prompts[0], "version")
        assert prompts[0].version is None

    @pytest.mark.asyncio
    async def test_get_prompt_not_found(self) -> None:
        """get_prompt raises when prompt not found."""
        from hud.environment import Environment

        env = Environment("test")

        async with env:
            with pytest.raises(ValueError, match="Prompt not found"):
                await env.get_prompt("nonexistent")


class TestEnvironmentSetupEvaluate:
    """Tests for setup_tool and evaluate_tool methods."""

    def test_setup_tool_with_name_and_kwargs(self) -> None:
        """setup_tool accepts name and kwargs."""
        from hud.environment import Environment

        env = Environment("test")
        env.setup_tool("navigate", url="https://example.com")

        assert len(env._setup_calls) == 1
        assert env._setup_calls[0] == ("navigate", {"url": "https://example.com"})

    def test_setup_tool_returns_self(self) -> None:
        """setup_tool returns self for chaining."""
        from hud.environment import Environment

        env = Environment("test")
        result = env.setup_tool("navigate", url="https://example.com")

        assert result is env

    def test_evaluate_tool_with_name_and_kwargs(self) -> None:
        """evaluate_tool accepts name and kwargs."""
        from hud.environment import Environment

        env = Environment("test")
        env.evaluate_tool("check_text", contains="success")

        assert len(env._evaluate_calls) == 1
        assert env._evaluate_calls[0] == ("check_text", {"contains": "success"})

    def test_evaluate_tool_returns_self(self) -> None:
        """evaluate_tool returns self for chaining."""
        from hud.environment import Environment

        env = Environment("test")
        result = env.evaluate_tool("check_text", contains="success")

        assert result is env

    def test_chaining_multiple_setup_calls(self) -> None:
        """Multiple setup_tool calls can be chained."""
        from hud.environment import Environment

        env = (
            Environment("test")
            .setup_tool("navigate", url="https://example.com")
            .setup_tool("wait", seconds=2)
        )

        assert len(env._setup_calls) == 2


class TestEnvironmentMCPProtocol:
    """Tests for MCP protocol overrides - Environment._env_list_tools and _env_call_tool.

    These test that Environment properly exposes connector tools via MCP handlers.
    """

    @pytest.mark.asyncio
    async def test_env_list_tools_includes_local_tools(self) -> None:
        """_env_list_tools returns local tools after routing is built."""
        from hud.environment import Environment

        env = Environment("test")

        @env.tool()
        def my_tool(x: int) -> int:
            """A test tool."""
            return x * 2

        # Build routing (simulates what __aenter__ does)
        await env._build_routing()

        # Call the handler that MCP will call
        tools = await env._env_list_tools()

        assert len(tools) == 1
        assert tools[0].name == "my_tool"

    @pytest.mark.asyncio
    async def test_env_list_tools_includes_connector_tools(self) -> None:
        """_env_list_tools returns tools from connectors (the key feature)."""
        import mcp.types as mcp_types

        from hud.environment import Environment

        env = Environment("test")

        # Create a mock connector with cached tools
        mock_tools = [
            mcp_types.Tool(
                name="remote_tool",
                description="A remote tool",
                inputSchema={"type": "object"},
            )
        ]

        class MockConnector:
            is_connected = True
            _tools_cache = mock_tools

            @property
            def cached_tools(self) -> list[mcp_types.Tool]:
                return self._tools_cache

            @property
            def cached_prompts(self) -> list[mcp_types.Prompt]:
                return []

            @property
            def cached_resources(self) -> list[mcp_types.Resource]:
                return []

            async def connect(self) -> None:
                pass

            async def disconnect(self) -> None:
                pass

            async def list_tools(self) -> list[mcp_types.Tool]:
                return self._tools_cache

        # Add the mock connector
        env._connections["mock"] = MockConnector()  # type: ignore

        # Build routing
        await env._build_routing()

        # Call the handler that MCP will call
        tools = await env._env_list_tools()

        # Should include the remote tool
        tool_names = [t.name for t in tools]
        assert "remote_tool" in tool_names

    @pytest.mark.asyncio
    async def test_env_call_tool_routes_to_local(self) -> None:
        """_env_call_tool routes local tool calls correctly."""
        from hud.environment import Environment

        env = Environment("test")
        called_with: list[int] = []

        @env.tool()
        def my_tool(x: int) -> str:
            """A test tool."""
            called_with.append(x)
            return f"result: {x}"

        # Build routing
        await env._build_routing()

        # Call the handler that MCP will call
        result = await env._env_call_tool("my_tool", {"x": 42})

        assert called_with == [42]
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_env_call_tool_routes_to_connector(self) -> None:
        """_env_call_tool routes connector tool calls correctly."""
        from unittest.mock import AsyncMock

        import mcp.types as mcp_types

        from hud.environment import Environment
        from hud.types import MCPToolResult

        env = Environment("test")

        # Create a mock connector
        mock_tools = [
            mcp_types.Tool(
                name="remote_tool",
                description="A remote tool",
                inputSchema={"type": "object"},
            )
        ]

        class MockConnector:
            is_connected = True
            _tools_cache = mock_tools
            call_tool = AsyncMock(
                return_value=MCPToolResult(
                    content=[mcp_types.TextContent(type="text", text="remote result")],
                    isError=False,
                )
            )

            @property
            def cached_tools(self) -> list[mcp_types.Tool]:
                return self._tools_cache

            @property
            def cached_prompts(self) -> list[mcp_types.Prompt]:
                return []

            @property
            def cached_resources(self) -> list[mcp_types.Resource]:
                return []

            async def connect(self) -> None:
                pass

            async def disconnect(self) -> None:
                pass

            async def list_tools(self) -> list[mcp_types.Tool]:
                return self._tools_cache

        mock_conn = MockConnector()
        env._connections["mock"] = mock_conn  # type: ignore

        # Build routing
        await env._build_routing()

        # Call the handler that MCP will call
        result = await env._env_call_tool("remote_tool", {"arg": "value"})

        # Verify the connector was called
        mock_conn.call_tool.assert_called_once_with("remote_tool", {"arg": "value"})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_env_call_tool_propagates_trace_from_request_ctx_to_agent_tool(self) -> None:
        """_env_call_tool reads trace_id from request_ctx for AgentTool calls."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from mcp.server.lowlevel.server import request_ctx
        from mcp.shared.context import RequestContext
        from mcp.types import RequestParams

        from hud.environment import Environment
        from hud.tools import AgentTool

        env = Environment("test")

        @env.scenario()
        async def investigate(issue: str):
            yield {"task": f"Investigate {issue}"}

        agent_tool = AgentTool(env("investigate"), model="claude", trace=True)
        env.add_tool(agent_tool.mcp)
        await env._build_routing()

        with (
            patch("hud.eval.manager.run_eval") as mock_run_eval,
            patch("hud.agents.create_agent") as mock_create_agent,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_run_eval.return_value = mock_ctx

            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(return_value=MagicMock(content="subagent output"))
            mock_create_agent.return_value = mock_agent
            req_meta = RequestParams.Meta.model_validate({"_hud_trace_id": "trace-from-meta"})
            req_context = RequestContext(
                request_id="test-req",
                meta=req_meta,
                session=MagicMock(),
                lifespan_context=None,
            )
            token = request_ctx.set(req_context)  # type: ignore[arg-type]
            try:
                result = await env._env_call_tool("investigate", {"issue": "order decline"})
            finally:
                request_ctx.reset(token)

        assert len(result) == 1
        assert mock_run_eval.call_args.kwargs["trace_id"] == "trace-from-meta"

    def test_setup_handlers_registers_custom_handlers(self) -> None:
        """Verify _setup_handlers registers our _env_list_tools and _env_call_tool."""
        from hud.environment import Environment

        env = Environment("test")

        # Verify the custom handlers exist
        assert hasattr(env, "_env_list_tools")
        assert hasattr(env, "_env_list_prompts")
        assert hasattr(env, "_env_call_tool")
        assert callable(env._env_list_tools)
        assert callable(env._env_list_prompts)
        assert callable(env._env_call_tool)

    @pytest.mark.asyncio
    async def test_list_prompts_handler_returns_list_prompts_result(self) -> None:
        """list_prompts handler should wrap prompts in ListPromptsResult."""
        import mcp.types as mcp_types

        from hud.environment import Environment

        env = Environment("test")

        async def fake_list_mcp_prompts() -> list[mcp_types.Prompt]:
            return [
                mcp_types.Prompt(
                    name="test:prompt",
                    description="Prompt description",
                    arguments=[],
                )
            ]

        env._list_mcp_prompts = fake_list_mcp_prompts  # type: ignore[method-assign]
        handler = env._mcp_server.request_handlers[mcp_types.ListPromptsRequest]
        request = mcp_types.ListPromptsRequest(method="prompts/list")

        result = await handler(request)

        assert isinstance(result.root, mcp_types.ListPromptsResult)
        assert len(result.root.prompts) == 1
        assert result.root.prompts[0].name == "test:prompt"
        assert isinstance(result.root.prompts[0], mcp_types.Prompt)
        assert not hasattr(result.root.prompts[0], "version")

    @pytest.mark.asyncio
    async def test_read_resource_handler_returns_read_resource_result(self) -> None:
        """read_resource handler should wrap contents in ReadResourceResult."""
        from typing import Any

        import mcp.types as mcp_types
        from pydantic import AnyUrl

        from hud.environment import Environment

        env = Environment("test")

        async def fake_read_resource(
            _uri: str, **_kwargs: Any
        ) -> list[mcp_types.TextResourceContents]:
            return [
                mcp_types.TextResourceContents(
                    uri=AnyUrl("test://resource"),
                    text='{"reward": 1.0, "done": true}',
                )
            ]

        env.read_resource = fake_read_resource  # type: ignore[method-assign]
        handler = env._mcp_server.request_handlers[mcp_types.ReadResourceRequest]
        request = mcp_types.ReadResourceRequest(
            method="resources/read",
            params=mcp_types.ReadResourceRequestParams(uri=AnyUrl("test://resource")),
        )

        result = await handler(request)

        assert isinstance(result.root, mcp_types.ReadResourceResult)
        assert len(result.root.contents) == 1


class TestEnvironmentToolFiltering:
    """Tests for agent-level tool filtering with wildcard support (v4 backwards compat)."""

    @pytest.mark.asyncio
    async def test_as_tools_no_filter(self) -> None:
        """as_tools returns all tools when no filter is set."""
        from hud.environment import Environment

        env = Environment("test")

        @env.tool()
        def tool_a() -> str:
            """Tool A."""
            return "a"

        @env.tool()
        def tool_b() -> str:
            """Tool B."""
            return "b"

        await env._build_routing()

        tools = env.as_tools()
        tool_names = [t.name for t in tools]

        assert "tool_a" in tool_names
        assert "tool_b" in tool_names

    @pytest.mark.asyncio
    async def test_as_tools_exact_include(self) -> None:
        """as_tools filters with exact include list."""
        from hud.environment import Environment

        env = Environment("test")

        @env.tool()
        def tool_a() -> str:
            """Tool A."""
            return "a"

        @env.tool()
        def tool_b() -> str:
            """Tool B."""
            return "b"

        env._agent_include = ["tool_a"]
        await env._build_routing()

        tools = env.as_tools()
        tool_names = [t.name for t in tools]

        assert "tool_a" in tool_names
        assert "tool_b" not in tool_names

    @pytest.mark.asyncio
    async def test_as_tools_exact_exclude(self) -> None:
        """as_tools filters with exact exclude list."""
        from hud.environment import Environment

        env = Environment("test")

        @env.tool()
        def tool_a() -> str:
            """Tool A."""
            return "a"

        @env.tool()
        def tool_b() -> str:
            """Tool B."""
            return "b"

        env._agent_exclude = ["tool_a"]
        await env._build_routing()

        tools = env.as_tools()
        tool_names = [t.name for t in tools]

        assert "tool_a" not in tool_names
        assert "tool_b" in tool_names

    @pytest.mark.asyncio
    async def test_as_tools_wildcard_exclude_prefix(self) -> None:
        """as_tools filters with wildcard prefix pattern (e.g., 'setup_*')."""
        from hud.environment import Environment

        env = Environment("test")

        @env.tool()
        def setup_database() -> str:
            """Setup tool."""
            return "setup"

        @env.tool()
        def setup_user() -> str:
            """Another setup tool."""
            return "setup"

        @env.tool()
        def run_query() -> str:
            """Regular tool."""
            return "query"

        env._agent_exclude = ["setup_*"]
        await env._build_routing()

        tools = env.as_tools()
        tool_names = [t.name for t in tools]

        assert "setup_database" not in tool_names
        assert "setup_user" not in tool_names
        assert "run_query" in tool_names

    @pytest.mark.asyncio
    async def test_as_tools_wildcard_exclude_contains(self) -> None:
        """as_tools filters with wildcard contains pattern (e.g., '*setup*')."""
        from hud.environment import Environment

        env = Environment("test")

        @env.tool()
        def hud_setup() -> str:
            """Contains setup."""
            return "setup"

        @env.tool()
        def setup_env() -> str:
            """Starts with setup."""
            return "setup"

        @env.tool()
        def my_setup_tool() -> str:
            """Contains setup in middle."""
            return "setup"

        @env.tool()
        def run_query() -> str:
            """No setup in name."""
            return "query"

        env._agent_exclude = ["*setup*"]
        await env._build_routing()

        tools = env.as_tools()
        tool_names = [t.name for t in tools]

        assert "hud_setup" not in tool_names
        assert "setup_env" not in tool_names
        assert "my_setup_tool" not in tool_names
        assert "run_query" in tool_names

    @pytest.mark.asyncio
    async def test_as_tools_multiple_wildcard_patterns(self) -> None:
        """as_tools filters with multiple wildcard patterns."""
        from hud.environment import Environment

        env = Environment("test")

        @env.tool()
        def setup_db() -> str:
            """Setup tool."""
            return "setup"

        @env.tool()
        def evaluate_result() -> str:
            """Evaluate tool."""
            return "evaluate"

        @env.tool()
        def checkout_branch() -> str:
            """Checkout tool."""
            return "checkout"

        @env.tool()
        def run_query() -> str:
            """Regular tool."""
            return "query"

        env._agent_exclude = ["*setup*", "*evaluate*", "checkout_branch"]
        await env._build_routing()

        tools = env.as_tools()
        tool_names = [t.name for t in tools]

        assert "setup_db" not in tool_names
        assert "evaluate_result" not in tool_names
        assert "checkout_branch" not in tool_names
        assert "run_query" in tool_names

    @pytest.mark.asyncio
    async def test_as_tools_wildcard_include_all(self) -> None:
        """as_tools with ['*'] include pattern matches all tools."""
        from hud.environment import Environment

        env = Environment("test")

        @env.tool()
        def tool_a() -> str:
            """Tool A."""
            return "a"

        @env.tool()
        def tool_b() -> str:
            """Tool B."""
            return "b"

        env._agent_include = ["*"]
        await env._build_routing()

        tools = env.as_tools()
        tool_names = [t.name for t in tools]

        assert "tool_a" in tool_names
        assert "tool_b" in tool_names

    @pytest.mark.asyncio
    async def test_as_tools_include_and_exclude_combined(self) -> None:
        """as_tools applies both include and exclude filters."""
        from hud.environment import Environment

        env = Environment("test")

        @env.tool()
        def browser_navigate() -> str:
            """Browser tool."""
            return "nav"

        @env.tool()
        def browser_setup() -> str:
            """Browser setup - should be excluded."""
            return "setup"

        @env.tool()
        def file_read() -> str:
            """File tool - not included."""
            return "read"

        env._agent_include = ["browser_*"]
        env._agent_exclude = ["*setup*"]
        await env._build_routing()

        tools = env.as_tools()
        tool_names = [t.name for t in tools]

        assert "browser_navigate" in tool_names
        assert "browser_setup" not in tool_names  # Excluded by *setup*
        assert "file_read" not in tool_names  # Not included by browser_*


class TestMCPServerToolExclusion:
    """Tests that scenario exclude_tools/exclude_sources/allowed_tools
    are enforced on the MCP server path (_env_list_tools, _env_call_tool).
    """

    @pytest.mark.asyncio
    async def test_env_list_tools_applies_scenario_filtering(self) -> None:
        """_env_list_tools resolves the MCP session and applies scenario filtering.

        The filtering logic itself (exclude_tools, exclude_sources, allowed_tools)
        is tested thoroughly in test_scenarios.py::TestScenarioToolExclusion.
        This test verifies the MCP server path wires up session lookup correctly.
        """
        from types import SimpleNamespace

        import mcp.types as mcp_types
        from mcp.server.lowlevel.server import request_ctx

        from hud.environment import Environment
        from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

        env = Environment("test-env")

        @env.tool()
        def browser_navigate(url: str) -> str:
            """Navigate."""
            return url

        @env.tool()
        def browser_screenshot() -> str:
            """Screenshot."""
            return "img"

        @env.tool()
        def bash(cmd: str) -> str:
            """Run command."""
            return cmd

        connector = Connector(
            transport={},
            config=ConnectionConfig(),
            name="remote-hub",
            connection_type=ConnectionType.REMOTE,
        )
        connector._tools_cache = [
            mcp_types.Tool(name="remote_a", inputSchema={"type": "object"}),
        ]
        env._connections["remote-hub"] = connector

        @env.scenario(
            "filtered",
            exclude_tools=["browser_*"],
            exclude_sources=["remote-hub"],
            allowed_tools=["browser_navigate"],
        )
        async def filtered():
            yield "Do it"
            yield 1.0

        await env._build_routing()

        req = SimpleNamespace(
            session=SimpleNamespace(),
            request=SimpleNamespace(headers={"mcp-session-id": "test-session"}),
        )
        token = request_ctx.set(req)  # type: ignore[arg-type]
        try:
            await env._env_get_prompt("test-env:filtered", {})
            tools = await env._env_list_tools()
        finally:
            request_ctx.reset(token)

        tool_names = [t.name for t in tools]
        assert "bash" in tool_names
        assert "browser_navigate" in tool_names  # Rescued by allowed_tools
        assert "browser_screenshot" not in tool_names  # Excluded by pattern
        assert "remote_a" not in tool_names  # Excluded by source

    @pytest.mark.asyncio
    async def test_env_call_tool_rejects_excluded_tool(self) -> None:
        """_env_call_tool raises ValueError for excluded tools."""
        from types import SimpleNamespace

        from mcp.server.lowlevel.server import request_ctx

        from hud.environment import Environment

        env = Environment("test-env")

        @env.tool()
        def browser_navigate(url: str) -> str:
            """Navigate."""
            return url

        @env.tool()
        def bash(cmd: str) -> str:
            """Run command."""
            return cmd

        @env.scenario("headless", exclude_tools=["browser_*"])
        async def headless():
            yield "Do it"
            yield 1.0

        await env._build_routing()

        req = SimpleNamespace(
            session=SimpleNamespace(),
            request=SimpleNamespace(headers={"mcp-session-id": "test-session-4"}),
        )
        token = request_ctx.set(req)  # type: ignore[arg-type]
        try:
            await env._env_get_prompt("test-env:headless", {})
            with pytest.raises(ValueError, match="not available"):
                await env._env_call_tool("browser_navigate", {"url": "http://example.com"})
        finally:
            request_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_env_call_tool_allows_non_excluded_tool(self) -> None:
        """_env_call_tool succeeds for non-excluded tools."""
        from types import SimpleNamespace

        from mcp.server.lowlevel.server import request_ctx

        from hud.environment import Environment

        env = Environment("test-env")

        @env.tool()
        def browser_navigate(url: str) -> str:
            """Navigate."""
            return url

        @env.tool()
        def bash(cmd: str) -> str:
            """Run command."""
            return cmd

        @env.scenario("headless", exclude_tools=["browser_*"])
        async def headless():
            yield "Do it"
            yield 1.0

        await env._build_routing()

        req = SimpleNamespace(
            session=SimpleNamespace(),
            request=SimpleNamespace(headers={"mcp-session-id": "test-session-5"}, scope={}),
        )
        token = request_ctx.set(req)  # type: ignore[arg-type]
        try:
            await env._env_get_prompt("test-env:headless", {})
            # Should not raise - bash is not excluded
            result = await env._env_call_tool("bash", {"cmd": "echo hi"})
            assert result is not None
        finally:
            request_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_env_call_tool_allows_internal_tools(self) -> None:
        """_env_call_tool always allows underscore-prefixed internal tools."""
        from types import SimpleNamespace

        from mcp.server.lowlevel.server import request_ctx

        from hud.environment import Environment

        env = Environment("test-env")

        @env.tool()
        def browser_navigate(url: str) -> str:
            """Navigate."""
            return url

        @env.scenario("headless", exclude_tools=["*"])
        async def headless():
            answer = yield "Do it"
            yield 1.0 if answer == "ok" else 0.0

        await env._build_routing()

        req = SimpleNamespace(
            session=SimpleNamespace(),
            request=SimpleNamespace(headers={"mcp-session-id": "test-session-6"}, scope={}),
        )
        token = request_ctx.set(req)  # type: ignore[arg-type]
        try:
            await env._env_get_prompt("test-env:headless", {})
            # _hud_submit should always work even with exclude_tools=["*"]
            result = await env._env_call_tool(
                "_hud_submit", {"scenario": "headless", "answer": "ok"}
            )
            assert result is not None
        finally:
            request_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_env_list_tools_no_session_returns_all(self) -> None:
        """_env_list_tools returns all tools when no scenario session is active."""
        from hud.environment import Environment

        env = Environment("test-env")

        @env.tool()
        def browser_navigate(url: str) -> str:
            """Navigate."""
            return url

        @env.tool()
        def bash(cmd: str) -> str:
            """Run command."""
            return cmd

        @env.scenario("headless", exclude_tools=["browser_*"])
        async def headless():
            yield "Do it"
            yield 1.0

        await env._build_routing()

        # No scenario setup, no request_ctx - should return all tools
        tools = await env._env_list_tools()
        tool_names = [t.name for t in tools]
        assert "browser_navigate" in tool_names
        assert "bash" in tool_names

    @pytest.mark.asyncio
    async def test_env_call_tool_no_session_allows_all(self) -> None:
        """_env_call_tool allows any tool when no scenario session is active."""
        from hud.environment import Environment

        env = Environment("test-env")

        @env.tool()
        def browser_navigate(url: str) -> str:
            """Navigate."""
            return url

        @env.scenario("headless", exclude_tools=["browser_*"])
        async def headless():
            yield "Do it"
            yield 1.0

        await env._build_routing()

        # No scenario setup - should allow any tool
        result = await env._env_call_tool("browser_navigate", {"url": "http://example.com"})
        assert result is not None
