"""Tests for base tool classes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import FastMCP
from mcp.types import ContentBlock, TextContent

from hud.tools.base import _INTERNAL_PREFIX, BaseHub, BaseTool


class MockTool(BaseTool):
    """Mock tool for testing."""

    async def __call__(self, param1: Any = None, param2: Any = None) -> list[ContentBlock]:
        """Execute the mock tool."""
        kwargs = {"param1": param1, "param2": param2}
        return [TextContent(type="text", text=f"Mock result: {kwargs}")]


class TestBaseTool:
    """Test BaseTool class."""

    def test_init_with_defaults(self):
        """Test BaseTool initialization with default values."""

        class TestTool(BaseTool):
            """A test tool."""

            async def __call__(self, **kwargs: Any) -> list[ContentBlock]:
                return []

        tool = TestTool()

        # Check auto-generated values
        assert tool.name == "test"
        assert tool.title == "Test"
        assert tool.description == "A test tool."
        assert tool.env is None
        assert tool.__name__ == "test"
        assert tool.__doc__ == "A test tool."

    def test_init_with_custom_values(self):
        """Test BaseTool initialization with custom values."""

        env = {"key": "value"}
        tool = MockTool(
            env=env, name="custom_tool", title="Custom Tool", description="Custom description"
        )

        assert tool.env == env
        assert tool.name == "custom_tool"
        assert tool.title == "Custom Tool"
        assert tool.description == "Custom description"
        assert tool.__name__ == "custom_tool"
        assert tool.__doc__ == "Custom description"

    def test_init_no_docstring(self):
        """Test BaseTool with no docstring."""

        class NoDocTool(BaseTool):
            async def __call__(self, **kwargs: Any) -> list[ContentBlock]:
                return []

        tool = NoDocTool()
        assert tool.description is None
        assert not hasattr(tool, "__doc__") or tool.__doc__ is None

    def test_register(self):
        """Test registering tool with FastMCP server."""

        server = MagicMock(spec=FastMCP)
        tool = MockTool(name="test_tool")

        # Test register returns self for chaining
        result = tool.register(server, tag="test")

        assert result is tool
        server.add_tool.assert_called_once()

        # Check the tool passed has correct name
        call_args = server.add_tool.call_args
        assert call_args[1]["tag"] == "test"

    def test_mcp_property_cached(self):
        """Test mcp property returns cached FunctionTool."""

        tool = MockTool(name="cached_tool", title="Cached Tool", description="Test caching")

        # First access creates the tool
        mcp_tool1 = tool.mcp
        assert hasattr(tool, "_mcp_tool")

        # Second access returns cached
        mcp_tool2 = tool.mcp
        assert mcp_tool1 is mcp_tool2

    def test_mcp_property_attributes(self):
        """Test mcp property creates FunctionTool with correct attributes."""

        tool = MockTool(
            name="mcp_test", title="MCP Test Tool", description="Testing MCP conversion"
        )

        with patch("fastmcp.tools.FunctionTool") as MockFunctionTool:
            mock_ft = MagicMock()
            MockFunctionTool.from_function.return_value = mock_ft

            result = tool.mcp

            # The wrapper function is passed, not the tool itself
            MockFunctionTool.from_function.assert_called_once()
            call_args = MockFunctionTool.from_function.call_args

            # Check that the correct parameters were passed
            assert call_args[1]["name"] == "mcp_test"
            assert call_args[1]["title"] == "MCP Test Tool"
            assert call_args[1]["description"] == "Testing MCP conversion"
            assert result is mock_ft


class TestBaseHub:
    """Test BaseHub class."""

    def test_init_basic(self):
        """Test BaseHub basic initialization."""

        hub = BaseHub("test_hub")

        assert hub._prefix_fn("tool") == f"{_INTERNAL_PREFIX}tool"
        assert hasattr(hub, "_tool_manager")
        assert hasattr(hub, "_resource_manager")

    def test_init_with_env(self):
        """Test BaseHub initialization with environment."""

        env = {"test": "env"}
        hub = BaseHub("test_hub", env=env, title="Test Hub", description="A test hub")

        assert hub.env == env

    @pytest.mark.asyncio
    async def test_dispatcher_tool_registered(self):
        """Test that dispatcher tool is registered on init."""

        hub = BaseHub("dispatcher_test")

        # Check dispatcher tool exists
        tools = hub._tool_manager._tools
        assert "dispatcher_test" in tools

        # Test calling dispatcher with internal tool
        @hub.tool("internal_func")
        async def internal_func(value: int) -> Any:
            return [TextContent(type="text", text=f"Internal: {value}")]

        # Call dispatcher
        result = await hub._tool_manager.call_tool(
            "dispatcher_test", {"name": "internal_func", "arguments": {"value": 42}}
        )

        # ToolResult has content attribute
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "Internal: 42"

    @pytest.mark.asyncio
    async def test_functions_catalogue_resource(self):
        """Test functions catalogue resource lists internal tools."""

        hub = BaseHub("catalogue_test")

        # Add some internal tools
        @hub.tool("func1")
        async def func1() -> Any:
            return []

        @hub.tool("func2")
        async def func2() -> Any:
            return []

        # Get the catalogue resource
        resources = hub._resource_manager._resources
        catalogue_uri = "file:///catalogue_test/functions"
        assert catalogue_uri in resources

        # Call the resource
        resource = resources[catalogue_uri]
        content = await resource.read()
        # The resource returns JSON content, parse it
        import json

        funcs = json.loads(content)

        assert sorted(funcs) == ["func1", "func2"]

    def test_tool_decorator_with_name(self):
        """Test tool decorator with explicit name."""

        hub = BaseHub("decorator_test")

        # Test positional name
        decorator = hub.tool("my_tool")
        assert callable(decorator)

        # Test keyword name
        decorator2 = hub.tool(name="my_tool2", tags={"test"})
        assert callable(decorator2)

    def test_tool_decorator_without_name(self):
        """Test tool decorator without name."""

        hub = BaseHub("decorator_test")

        # Test bare decorator
        decorator = hub.tool()
        assert callable(decorator)

        # Test decorator with only kwargs
        decorator2 = hub.tool(tags={"test"})
        assert callable(decorator2)

    def test_tool_decorator_phase2(self):
        """Test tool decorator phase 2 (when function is passed)."""

        hub = BaseHub("phase2_test")

        async def my_func() -> Any:
            return []

        # Simulate phase 2 of decorator application
        with patch.object(FastMCP, "tool") as mock_super_tool:
            mock_super_tool.return_value = my_func

            # Call with function directly (phase 2)
            result = hub.tool(my_func, tags={"test"})

            assert result is my_func
            mock_super_tool.assert_called_once_with(my_func, tags={"test"})

    @pytest.mark.asyncio
    async def test_list_tools_hides_internal(self):
        """Test _list_tools hides internal tools."""

        hub = BaseHub("list_test")

        # Add public tool (use @hub.tool() without prefix for public tools in FastMCP)
        from fastmcp.tools import Tool

        async def public_tool() -> Any:
            return []

        public_tool_obj = Tool.from_function(public_tool)
        hub.add_tool(public_tool_obj)

        # Add internal tool
        @hub.tool("internal_tool")
        async def internal_tool() -> Any:
            return []

        # List tools should only show public
        tools = await hub._list_tools()
        tool_names = [t.name for t in tools]

        assert "public_tool" in tool_names
        assert "internal_tool" not in tool_names
        assert f"{_INTERNAL_PREFIX}internal_tool" not in tool_names

    def test_resource_and_prompt_passthrough(self):
        """Test that resource and prompt decorators pass through."""

        hub = BaseHub("passthrough_test")

        # These should be inherited from FastMCP
        assert hasattr(hub, "resource")
        assert hasattr(hub, "prompt")
        # Check they're the same methods (by name)
        assert hub.resource.__name__ == FastMCP.resource.__name__
        assert hub.prompt.__name__ == FastMCP.prompt.__name__
