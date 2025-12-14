from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from mcp.types import ImageContent, TextContent

from hud.types import AgentResponse, LegacyTask, MCPToolCall, MCPToolResult, Trace, TraceStep


def test_task_with_json_strings():
    """Test LegacyTask with JSON strings for config fields."""
    task = LegacyTask(
        prompt="test",
        mcp_config='{"test": "config"}',  # type: ignore
        metadata='{"key": "value"}',  # type: ignore
        agent_config='{"system_prompt": "test"}',  # type: ignore
    )
    assert task.mcp_config == {"test": "config"}
    assert task.metadata == {"key": "value"}
    assert task.agent_config is not None
    assert task.agent_config.system_prompt == "test"


def test_task_json_parse_error():
    """Test LegacyTask raises error on invalid JSON."""
    from hud.shared.exceptions import HudConfigError

    with pytest.raises(HudConfigError, match="Invalid JSON string"):
        LegacyTask(prompt="test", mcp_config="{invalid json}")  # type: ignore


def test_task_agent_config_rejects_extra_fields():
    """Test LegacyTask agent_config rejects unknown fields."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LegacyTask(
            prompt="test",
            mcp_config={},
            agent_config={"model": "test", "unknown_field": "value"},  # type: ignore
        )


def test_task_setup_tool_from_json_string():
    """Test LegacyTask converts JSON string to tool call."""
    task = LegacyTask(
        prompt="test",
        mcp_config={},
        setup_tool='{"name": "test_tool", "arguments": {"x": 1}}',  # type: ignore
    )
    assert isinstance(task.setup_tool, MCPToolCall)
    assert task.setup_tool.name == "test_tool"


def test_task_setup_tool_json_error():
    """Test LegacyTask raises error on invalid tool JSON."""
    from hud.shared.exceptions import HudConfigError

    with pytest.raises(HudConfigError, match="Invalid JSON string"):
        LegacyTask(prompt="test", mcp_config={}, setup_tool="{invalid}")  # type: ignore


def test_task_setup_tool_from_list():
    """Test LegacyTask converts list of dicts to list of tool calls."""
    task = LegacyTask(
        prompt="test",
        mcp_config={},
        setup_tool=[
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
        ],  # type: ignore
    )
    assert isinstance(task.setup_tool, list)
    assert len(task.setup_tool) == 2
    assert all(isinstance(t, MCPToolCall) for t in task.setup_tool)


def test_task_env_var_substitution():
    """Test LegacyTask resolves environment variables."""
    with patch.dict("os.environ", {"TEST_VAR": "test_value"}):
        task = LegacyTask(
            prompt="test",
            mcp_config={"url": "${TEST_VAR}"},
        )
        assert task.mcp_config["url"] == "test_value"


def test_task_env_var_nested():
    """Test LegacyTask resolves env vars in nested structures."""
    with patch.dict("os.environ", {"NESTED_VAR": "nested_value"}):
        task = LegacyTask(
            prompt="test",
            mcp_config={"level1": {"level2": {"url": "${NESTED_VAR}"}}},
        )
        assert task.mcp_config["level1"]["level2"]["url"] == "nested_value"


def test_task_env_var_in_list():
    """Test LegacyTask resolves env vars in lists."""
    with patch.dict("os.environ", {"LIST_VAR": "list_value"}):
        task = LegacyTask(
            prompt="test",
            mcp_config={"items": ["${LIST_VAR}", "static"]},
        )
        assert task.mcp_config["items"][0] == "list_value"


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


def test_agent_response_str_with_reasoning():
    """Test AgentResponse __str__ includes reasoning."""
    response = AgentResponse(reasoning="Test reasoning", content="Test content")
    output = str(response)
    assert "Reasoning: Test reasoning" in output
    assert "Content: Test content" in output


def test_agent_response_str_with_tool_calls():
    """Test AgentResponse __str__ includes tool calls."""
    response = AgentResponse(
        tool_calls=[
            MCPToolCall(name="tool1", arguments={"a": 1}),
            MCPToolCall(name="tool2", arguments={"b": 2}),
        ]
    )
    output = str(response)
    assert "Tool Calls:" in output
    assert "tool1" in output
    assert "tool2" in output


def test_agent_response_str_with_raw():
    """Test AgentResponse __str__ includes raw."""
    response = AgentResponse(raw={"raw_data": "value"})
    output = str(response)
    assert "Raw:" in output


def test_trace_len():
    """Test Trace __len__ returns number of steps."""
    trace = Trace()
    trace.append(TraceStep(category="mcp"))
    trace.append(TraceStep(category="agent"))
    assert len(trace) == 2


def test_trace_num_messages():
    """Test Trace num_messages property."""
    trace = Trace(messages=[{"role": "user"}, {"role": "assistant"}])
    assert trace.num_messages == 2
