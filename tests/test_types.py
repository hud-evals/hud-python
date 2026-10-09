from __future__ import annotations

from unittest.mock import patch

from mcp.types import ImageContent, TextContent

from hud.types import MCPToolCall, MCPToolResult


def test_mcp_tool_call_str_long_args():
    """Test MCPToolCall __str__ truncates long arguments."""
    tool_call = MCPToolCall(
        name="test_tool",
        arguments={"very": "long" * 30 + " argument string that should be truncated"},
    )
    result = str(tool_call)
    assert "..." in result
    assert len(result) < 100


def test_mcp_tool_call_str_invalid_json_args():
    """Test MCPToolCall __str__ handles non-JSON-serializable arguments."""
    tool_call = MCPToolCall(name="test_tool", arguments={"func": lambda x: x})
    result = str(tool_call)
    assert "test_tool" in result


def test_mcp_tool_call_rich():
    """Test MCPToolCall __rich__ calls hud_console."""
    with patch("hud.utils.hud_console.hud_console") as mock_console:
        mock_console.format_tool_call.return_value = "formatted"
        tool_call = MCPToolCall(name="test", arguments={})
        result = tool_call.__rich__()
        assert result == "formatted"
        mock_console.format_tool_call.assert_called_once()


def test_mcp_tool_call_annotation_in_model_dump():
    """model_dump() includes annotation when set."""
    tool_call = MCPToolCall(name="click", arguments={"x": 100}, annotation="Navigate to login page")
    data = tool_call.model_dump()
    assert data["annotation"] == "Navigate to login page"


def test_mcp_tool_call_annotation_roundtrip():
    """Annotation survives serialize -> deserialize roundtrip."""
    original = MCPToolCall(name="click", arguments={"x": 100}, annotation="Step 1: open menu")
    data = original.model_dump(mode="json")
    restored = MCPToolCall(**data)
    assert restored.annotation == "Step 1: open menu"
    assert restored.name == original.name
    assert restored.arguments == original.arguments


def test_mcp_tool_call_annotation_none_excluded():
    """model_dump(exclude_none=True) omits annotation when None."""
    tool_call = MCPToolCall(name="click", arguments={})
    data = tool_call.model_dump(exclude_none=True)
    assert "annotation" not in data


def test_mcp_tool_call_annotation_defaults_to_none():
    """MCPToolCall without explicit annotation defaults to None."""
    tool_call = MCPToolCall(name="click", arguments={"x": 1})
    assert tool_call.annotation is None


def test_mcp_tool_call_str_with_annotation():
    """__str__ appends annotation comment when set."""
    tool_call = MCPToolCall(name="click", arguments={"x": 1}, annotation="Open the sidebar")
    result = str(tool_call)
    assert result.endswith("  # Open the sidebar")
    assert "click" in result


def test_mcp_tool_call_str_without_annotation():
    """__str__ has no annotation comment when annotation is None."""
    tool_call = MCPToolCall(name="click", arguments={"x": 1})
    result = str(tool_call)
    assert "#" not in result


def test_mcp_tool_call_rich_with_annotation():
    """__rich__ includes escaped annotation in bright_black markup."""
    with patch("hud.utils.hud_console.hud_console") as mock_console:
        mock_console.format_tool_call.return_value = "formatted"
        tool_call = MCPToolCall(name="test", arguments={}, annotation="has [brackets] & stuff")
        result = tool_call.__rich__()
        assert "[bright_black]" in result
        assert "has \\[brackets] & stuff" in result


def test_mcp_tool_result_text_content():
    """Test MCPToolResult with text content."""
    result = MCPToolResult(
        content=[TextContent(text="Test output", type="text")],
        isError=False,
    )
    assert "Test output" in str(result)
    assert "✓" in str(result)


def test_mcp_tool_result_multiline_text():
    """Test MCPToolResult with multiline text uses first line."""
    result = MCPToolResult(
        content=[TextContent(text="First line\nSecond line\nThird line", type="text")],
        isError=False,
    )
    assert "First line" in result._get_content_summary()
    assert "Second line" not in result._get_content_summary()


def test_mcp_tool_result_image_content():
    """Test MCPToolResult with image content."""
    result = MCPToolResult(
        content=[ImageContent(data="base64data", mimeType="image/png", type="image")],
        isError=False,
    )
    summary = result._get_content_summary()
    assert "Image" in summary or "📷" in summary


def test_mcp_tool_result_structured_content():
    """Test MCPToolResult with structured content."""
    result = MCPToolResult(
        content=[],
        structuredContent={"key": "value", "nested": {"data": 123}},
        isError=False,
    )
    summary = result._get_content_summary()
    assert "key" in summary


def test_mcp_tool_result_structured_content_non_serializable():
    """Test MCPToolResult with non-JSON-serializable structured content."""
    result = MCPToolResult(
        content=[],
        structuredContent={"func": lambda x: x},
        isError=False,
    )
    summary = result._get_content_summary()
    assert summary  # Should have some string representation


def test_mcp_tool_result_error():
    """Test MCPToolResult when isError is True."""
    result = MCPToolResult(
        content=[TextContent(text="Error message", type="text")],
        isError=True,
    )
    assert "✗" in str(result)


def test_mcp_tool_result_rich():
    """Test MCPToolResult __rich__ calls hud_console."""
    with patch("hud.utils.hud_console.hud_console") as mock_console:
        mock_console.format_tool_result.return_value = "formatted"
        result = MCPToolResult(
            content=[TextContent(text="Test", type="text")],
            isError=False,
        )
        rich_output = result.__rich__()
        assert rich_output == "formatted"
        mock_console.format_tool_result.assert_called_once()
