"""Tests for @env.tool() decorator and tool operations."""

from __future__ import annotations

import pytest

from hud.environment import Environment


class TestToolDecorator:
    """Tests for @env.tool() decorator."""

    def test_tool_registers_function(self) -> None:
        """@env.tool registers the function in tool manager."""
        env = Environment("test-env")

        @env.tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        # Check tool was registered
        tool_names = list(env._tool_manager._tools.keys())
        assert "add" in tool_names

    def test_tool_with_custom_name(self) -> None:
        """@env.tool(name=...) uses custom name."""
        env = Environment("test-env")

        @env.tool(name="custom_add")
        def add(a: int, b: int) -> int:
            return a + b

        tool_names = list(env._tool_manager._tools.keys())
        assert "custom_add" in tool_names
        assert "add" not in tool_names

    def test_tool_preserves_docstring(self) -> None:
        """@env.tool preserves function docstring as description."""
        env = Environment("test-env")

        @env.tool()
        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}!"

        tool = env._tool_manager._tools.get("greet")
        assert tool is not None
        assert "Greet someone by name" in (tool.description or "")

    def test_tool_async_function(self) -> None:
        """@env.tool works with async functions."""
        env = Environment("test-env")

        @env.tool()
        async def fetch_data(url: str) -> str:
            """Fetch data from URL."""
            return f"Data from {url}"

        tool_names = list(env._tool_manager._tools.keys())
        assert "fetch_data" in tool_names

    def test_tool_returns_function(self) -> None:
        """@env.tool returns the original function."""
        env = Environment("test-env")

        @env.tool()
        def add(a: int, b: int) -> int:
            return a + b

        # Should be able to call it directly
        assert add(2, 3) == 5


class TestListTools:
    """Tests for list_tools and as_tools."""

    @pytest.mark.asyncio
    async def test_as_tools_returns_registered_tools(self) -> None:
        """as_tools returns list of registered MCP tools."""
        env = Environment("test-env")

        @env.tool()
        def tool1() -> str:
            return "1"

        @env.tool()
        def tool2() -> str:
            return "2"

        async with env:
            tools = env.as_tools()
            tool_names = [t.name for t in tools]
            assert "tool1" in tool_names
            assert "tool2" in tool_names

    @pytest.mark.asyncio
    async def test_as_tools_empty_when_no_tools(self) -> None:
        """as_tools returns empty list when no tools registered."""
        env = Environment("test-env")
        async with env:
            tools = env.as_tools()
            # May have built-in _hud_submit tool
            user_tools = [t for t in tools if not t.name.startswith("_")]
            assert len(user_tools) == 0


class TestCallTool:
    """Tests for call_tool method."""

    @pytest.mark.asyncio
    async def test_call_tool_executes_function(self) -> None:
        """call_tool executes registered tool function."""
        env = Environment("test-env")
        executed = []

        @env.tool()
        def greet(name: str) -> str:
            executed.append(name)
            return f"Hello, {name}!"

        async with env:
            result = await env.call_tool("greet", name="Alice")

        assert executed == ["Alice"]
        assert result is not None

    @pytest.mark.asyncio
    async def test_call_tool_async_function(self) -> None:
        """call_tool works with async tool functions."""
        env = Environment("test-env")

        @env.tool()
        async def async_greet(name: str) -> str:
            return f"Hello, {name}!"

        async with env:
            result = await env.call_tool("async_greet", name="Bob")

        assert result is not None

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self) -> None:
        """call_tool raises for unknown tool."""
        env = Environment("test-env")

        async with env:
            with pytest.raises(ValueError, match="Tool not found"):
                await env.call_tool("nonexistent")


class TestMockMode:
    """Tests for mock mode."""

    def test_mock_mode_default_false(self) -> None:
        """Mock mode is False by default."""
        env = Environment("test-env")
        assert env._mock_mode is False
        assert env.is_mock is False

    def test_mock_enables_mock_mode(self) -> None:
        """mock() enables mock mode."""
        env = Environment("test-env")
        env.mock()
        assert env._mock_mode is True
        assert env.is_mock is True

    def test_unmock_disables_mock_mode(self) -> None:
        """unmock() disables mock mode."""
        env = Environment("test-env")
        env.mock()
        env.unmock()
        assert env._mock_mode is False

    def test_mock_returns_self_for_chaining(self) -> None:
        """mock() returns self for chaining."""
        env = Environment("test-env")
        result = env.mock()
        assert result is env

    def test_mock_tool_sets_custom_output(self) -> None:
        """mock_tool() sets custom output for a tool."""
        env = Environment("test-env")
        env.mock_tool("navigate", "Custom result")
        assert env._mock_outputs["navigate"] == "Custom result"

    @pytest.mark.asyncio
    async def test_mock_mode_returns_mock_response(self) -> None:
        """Mock mode returns mock response instead of executing tool."""
        env = Environment("test-env")
        call_count = 0

        @env.tool()
        def real_tool() -> str:
            nonlocal call_count
            call_count += 1
            return "real result"

        env.mock()
        env.mock_tool("real_tool", "mocked result")

        async with env:
            result = await env.call_tool("real_tool")

        # Tool should not be called in mock mode
        assert call_count == 0
        # Should get the mock result
        assert result is not None
