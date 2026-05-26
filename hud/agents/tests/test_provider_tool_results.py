"""Provider continuation contracts for environment tool results."""

from __future__ import annotations

from typing import Any, cast

from mcp import types

from hud.agents.claude.tools.base import ClaudeFunctionTool
from hud.agents.gemini.tools.base import GeminiFunctionTool
from hud.agents.openai.tools.base import OpenAIFunctionTool
from hud.agents.openai_compatible.tools.base import OpenAICompatibleFunctionTool
from hud.agents.tests.conftest import mcp_tool
from hud.types import MCPToolCall, MCPToolResult


def _text_image_result() -> MCPToolResult:
    return MCPToolResult(
        content=[
            types.TextContent(type="text", text="text output"),
            types.ImageContent(type="image", data="image-bytes", mimeType="image/png"),
        ],
        isError=False,
    )


def test_openai_formats_text_image_structured_and_error_results() -> None:
    tool = OpenAIFunctionTool.from_tool(mcp_tool("lookup", description="Lookup things"))
    assert tool is not None

    output = tool.format_result(
        MCPToolCall(name="lookup", id="call_1", arguments={}),
        MCPToolResult(
            content=[
                types.TextContent(type="text", text="failed"),
                types.ImageContent(type="image", data="image-bytes", mimeType="image/png"),
            ],
            isError=True,
            structuredContent={"code": 500},
        ),
    )

    assert output is not None
    output_dict = cast("dict[str, Any]", output)
    assert output_dict["type"] == "function_call_output"
    assert output_dict["call_id"] == "call_1"
    blocks = cast("list[dict[str, Any]]", output_dict["output"])
    assert {"type": "input_text", "text": "[tool_error] true"} in blocks
    assert {"type": "input_text", "text": '{"code": 500}'} in blocks
    assert {"type": "input_text", "text": "failed"} in blocks
    assert {
        "type": "input_image",
        "image_url": "data:image/png;base64,image-bytes",
    } in blocks


def test_openai_formats_empty_result_as_empty_function_output() -> None:
    tool = OpenAIFunctionTool.from_tool(mcp_tool("lookup", description="Lookup things"))
    assert tool is not None

    output = tool.format_result(
        MCPToolCall(name="lookup", id="call_1", arguments={}),
        MCPToolResult(content=[], isError=False),
    )

    assert output is not None
    blocks = cast("list[dict[str, Any]]", cast("dict[str, Any]", output)["output"])
    assert blocks == [{"type": "input_text", "text": ""}]


def test_claude_formats_result_blocks_and_citation_documents() -> None:
    tool = ClaudeFunctionTool.from_tool(mcp_tool("lookup", description="Lookup things"))

    message = tool.format_result(
        MCPToolCall(
            name="lookup",
            id="call_1",
            arguments={},
            _meta=types.RequestParams.Meta.model_validate({"enable_citations": True}),
        ),
        _text_image_result(),
    )

    assert message is not None
    assert message["role"] == "user"
    content = cast("list[dict[str, Any]]", message["content"])
    tool_result = content[0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_1"
    assert cast("list[dict[str, Any]]", tool_result["content"]) == [
        {"type": "text", "text": "text output"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "image-bytes",
            },
        },
    ]
    assert content[1]["type"] == "document"
    assert content[1]["citations"] == {"enabled": True}


def test_claude_formats_errors_as_tool_result_text() -> None:
    tool = ClaudeFunctionTool.from_tool(mcp_tool("lookup", description="Lookup things"))

    message = tool.format_result(
        MCPToolCall(name="lookup", id="call_1", arguments={}),
        MCPToolResult(
            content=[types.TextContent(type="text", text="boom")],
            isError=True,
        ),
    )

    assert message is not None
    tool_result = cast("list[dict[str, Any]]", message["content"])[0]
    assert tool_result["content"] == [{"type": "text", "text": "Error: boom"}]


def test_gemini_formats_success_and_error_function_responses() -> None:
    tool = GeminiFunctionTool.from_tool(mcp_tool("lookup", description="Lookup things"))

    success = tool.format_result(
        MCPToolCall(name="lookup", provider_name="provider_lookup", arguments={}),
        MCPToolResult(
            content=[types.TextContent(type="text", text="found")],
            isError=False,
        ),
    )
    error = tool.format_result(
        MCPToolCall(name="lookup", arguments={}),
        MCPToolResult(
            content=[types.TextContent(type="text", text="failed")],
            isError=True,
        ),
    )

    success_parts = success.parts or []
    error_parts = error.parts or []
    success_response = success_parts[0].function_response
    error_response = error_parts[0].function_response
    assert success_response is not None
    assert success_response.name == "provider_lookup"
    assert success_response.response == {"success": True, "output": "found"}
    assert error_response is not None
    assert error_response.response == {"error": "failed"}


def test_openai_compatible_formats_text_image_and_structured_results() -> None:
    tool = OpenAICompatibleFunctionTool.from_tool(mcp_tool("lookup", description="Lookup things"))

    image_output = tool.format_result(
        MCPToolCall(name="lookup", id="call_1", arguments={}),
        _text_image_result(),
    )
    structured_output = tool.format_result(
        MCPToolCall(name="lookup", id="call_2", arguments={}),
        MCPToolResult(
            content=[], isError=False, structuredContent={"result": {"type": "text", "text": "ok"}}
        ),
    )

    assert image_output == [
        {"role": "tool", "tool_call_id": "call_1", "content": "text output"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tool returned the following:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,image-bytes"}},
            ],
        },
    ]
    assert structured_output == {"role": "tool", "tool_call_id": "call_2", "content": "ok"}
