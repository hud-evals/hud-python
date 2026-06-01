"""Computer tool contracts shared across provider adapters."""

from __future__ import annotations

from typing import Any, cast

import pytest
from mcp import types

from hud.agents.gemini.tools.computer import (
    GEMINI_COMPUTER_SPEC,
    GEMINI_SAFETY_BLOCKED_PREFIX,
    GEMINI_URL_PREFIX,
    GeminiComputerTool,
)
from hud.agents.openai.tools.computer import OpenAIComputerTool
from hud.agents.openai_compatible.tools import OpenAICompatibleAgentTools
from hud.agents.openai_compatible.tools.glm_computer import GLM_COMPUTER_SPEC, GLMComputerTool
from hud.agents.openai_compatible.tools.qwen_computer import (
    QWEN_COMPUTER_SPEC,
    QwenComputerTool,
)
from hud.agents.tests.conftest import RecordingToolEnvironment, mcp_tool, text_result
from hud.agents.tools.computer import execute_computer_calls
from hud.types import MCPToolCall, MCPToolResult


def _image_result(data: str = "screenshot") -> MCPToolResult:
    return MCPToolResult(
        content=[types.ImageContent(type="image", data=data, mimeType="image/png")],
        isError=False,
    )


@pytest.mark.asyncio
async def test_shared_computer_execution_appends_screenshot_when_required() -> None:
    calls: list[MCPToolCall] = []

    async def call_tool(call: MCPToolCall) -> MCPToolResult:
        calls.append(call)
        if (call.arguments or {}).get("action") == "screenshot":
            return _image_result("after")
        return text_result("clicked")

    result = await execute_computer_calls(
        call_tool,
        env_tool_name="computer",
        calls=[{"action": "click", "x": 1, "y": 2}],
        ensure_screenshot=True,
    )

    assert [(call.name, call.arguments) for call in calls] == [
        ("computer", {"action": "click", "x": 1, "y": 2}),
        ("computer", {"action": "screenshot"}),
    ]
    assert [type(block).__name__ for block in result.content] == ["TextContent", "ImageContent"]


@pytest.mark.asyncio
async def test_openai_computer_skips_extra_screenshot_when_action_returns_image() -> None:
    spec = OpenAIComputerTool.default_spec("gpt-5.4")
    assert spec is not None
    tool = OpenAIComputerTool(env_tool_name="computer", spec=spec)
    calls: list[MCPToolCall] = []

    async def call_tool(call: MCPToolCall) -> MCPToolResult:
        calls.append(call)
        return _image_result("already")

    result = await tool.execute(
        call_tool,
        {"type": "click", "x": 1, "y": 2},
    )

    assert [(call.name, call.arguments) for call in calls] == [
        ("computer", {"action": "click", "x": 1, "y": 2, "button": "left", "hold_keys": None})
    ]
    assert result == _image_result("already")


@pytest.mark.asyncio
async def test_openai_compatible_registry_routes_native_computer_tools_by_model() -> None:
    computer = mcp_tool("computer", meta={"capability": "computer"})
    glm_environment = RecordingToolEnvironment(results={"computer": text_result("clicked")})
    qwen_environment = RecordingToolEnvironment(results={"computer": text_result("waited")})

    glm_tools = OpenAICompatibleAgentTools()
    glm_tools.prepare(model="glm-4.5v", tools=[computer])
    qwen_tools = OpenAICompatibleAgentTools()
    qwen_tools.prepare(model="qwen-vl-max", tools=[computer])

    await glm_tools.execute(
        glm_environment.call_tool,
        MCPToolCall(name="computer", arguments={"action": "left_click", "start_box": "[10,20]"}),
    )
    await qwen_tools.execute(
        qwen_environment.call_tool,
        MCPToolCall(name="computer_use", arguments={"action": "wait", "time": 1.5}),
    )

    glm_params = [cast("dict[str, Any]", param) for param in glm_tools.params]
    qwen_params = [cast("dict[str, Any]", param) for param in qwen_tools.params]
    assert any(param.get("function", {}).get("name") == "computer" for param in glm_params)
    assert any(param.get("type") == "computer_use" for param in qwen_params)
    assert [(call.name, call.arguments) for call in glm_environment.calls] == [
        ("computer", {"action": "click", "x": 10, "y": 15, "button": "left"}),
        ("computer", {"action": "screenshot"}),
    ]
    assert [(call.name, call.arguments) for call in qwen_environment.calls] == [
        ("computer", {"action": "wait", "time": 1500})
    ]


@pytest.mark.asyncio
async def test_openai_computer_translates_actions_and_requires_final_screenshot() -> None:
    spec = OpenAIComputerTool.default_spec("gpt-5.4")
    assert spec is not None
    tool = OpenAIComputerTool(env_tool_name="computer", spec=spec)
    calls: list[MCPToolCall] = []

    async def call_tool(call: MCPToolCall) -> MCPToolResult:
        calls.append(call)
        if (call.arguments or {}).get("action") == "screenshot":
            return _image_result("after")
        return text_result("acted")

    result = await tool.execute(
        call_tool,
        {"type": "click", "x": 10, "y": 20, "button": "wheel", "keys": ["ctrl"]},
    )

    assert result.content == [
        types.TextContent(type="text", text="acted"),
        types.ImageContent(type="image", data="after", mimeType="image/png"),
    ]
    assert [(call.name, call.arguments) for call in calls] == [
        (
            "computer",
            {
                "action": "click",
                "x": 10,
                "y": 20,
                "button": "middle",
                "hold_keys": ["ctrl"],
            },
        ),
        ("computer", {"action": "screenshot"}),
    ]


def test_openai_computer_formats_screenshot_for_provider_continuation() -> None:
    spec = OpenAIComputerTool.default_spec("gpt-5.4")
    assert spec is not None
    tool = OpenAIComputerTool(env_tool_name="computer", spec=spec)

    formatted = tool.format_result(
        MCPToolCall(name="computer", id="call_1", arguments={}),
        _image_result("after"),
    )

    output = cast("dict[str, Any]", formatted)
    assert output["type"] == "computer_call_output"
    assert output["call_id"] == "call_1"
    assert output["output"] == {
        "type": "computer_screenshot",
        "image_url": "data:image/png;base64,after",
        "detail": "original",
    }


def test_openai_computer_rejects_provider_continuation_without_screenshot() -> None:
    spec = OpenAIComputerTool.default_spec("gpt-5.4")
    assert spec is not None
    tool = OpenAIComputerTool(env_tool_name="computer", spec=spec)

    with pytest.raises(ValueError, match="missing screenshot"):
        tool.format_result(
            MCPToolCall(name="computer", id="call_1", arguments={}),
            text_result("no screenshot"),
        )


@pytest.mark.asyncio
async def test_gemini_computer_blocks_unconfirmed_safety_decision_without_environment_call() -> (
    None
):
    tool = GeminiComputerTool(env_tool_name="computer", spec=GEMINI_COMPUTER_SPEC)
    environment = RecordingToolEnvironment()

    result = await tool.execute(
        environment.call_tool,
        {
            "action": "click_at",
            "safety_decision": {"decision": "require_confirmation"},
        },
    )

    assert environment.calls == []
    assert result.isError is False
    assert result.content == [
        types.TextContent(
            type="text",
            text=(
                f"{GEMINI_SAFETY_BLOCKED_PREFIX}"
                "Gemini Computer Use action requires user confirmation before execution."
            ),
        )
    ]


def test_gemini_computer_formats_url_safety_and_inline_screenshot_parts() -> None:
    tool = GeminiComputerTool(env_tool_name="computer", spec=GEMINI_COMPUTER_SPEC)

    content = tool.format_result(
        MCPToolCall(
            name="computer_use",
            provider_name="click_at",
            arguments={"safety_decision": {"decision": "allow"}},
        ),
        MCPToolResult(
            content=[
                types.TextContent(type="text", text="clicked"),
                types.TextContent(type="text", text=f"{GEMINI_URL_PREFIX}https://example.com"),
                types.ImageContent(type="image", data="YWJj", mimeType="image/png"),
            ],
            isError=False,
        ),
    )

    parts = content.parts or []
    response = parts[0].function_response
    assert response is not None
    assert response.name == "click_at"
    assert response.response == {
        "success": True,
        "output": "clicked",
        "url": "https://example.com",
        "safety_acknowledgement": True,
    }
    response_parts = response.parts or []
    assert response_parts[0].inline_data is not None
    assert response_parts[0].inline_data.data == b"abc"


@pytest.mark.asyncio
async def test_glm_computer_scales_normalized_click_coordinates() -> None:
    tool = GLMComputerTool(
        env_tool_name="computer",
        spec=GLM_COMPUTER_SPEC,
        display_width=1000,
        display_height=500,
        coordinate_space=None,
    )
    environment = RecordingToolEnvironment(results={"computer": text_result("ok")})

    await tool.execute(
        environment.call_tool,
        {"action": "left_click", "start_box": "[999,999]"},
    )

    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("computer", {"action": "click", "x": 999, "y": 499, "button": "left"}),
        ("computer", {"action": "screenshot"}),
    ]


@pytest.mark.asyncio
async def test_glm_computer_repairs_xml_encoded_arguments() -> None:
    tool = GLMComputerTool(
        env_tool_name="computer",
        spec=GLM_COMPUTER_SPEC,
        display_width=1000,
        display_height=500,
        coordinate_space=None,
    )
    environment = RecordingToolEnvironment(results={"computer": text_result("ok")})

    await tool.execute(
        environment.call_tool,
        {"action": ("left_click<arg_key>start_box</arg_key><arg_value>[500,500]</arg_value>")},
    )

    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("computer", {"action": "click", "x": 500, "y": 250, "button": "left"}),
        ("computer", {"action": "screenshot"}),
    ]


@pytest.mark.asyncio
async def test_qwen_computer_translates_wait_seconds_to_milliseconds() -> None:
    tool = QwenComputerTool(
        env_tool_name="computer",
        spec=QWEN_COMPUTER_SPEC,
        display_width=1000,
        display_height=500,
        description="computer",
    )
    environment = RecordingToolEnvironment(results={"computer": text_result("waited")})

    await tool.execute(environment.call_tool, {"action": "wait", "time": 1.5})

    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("computer", {"action": "wait", "time": 1500})
    ]


@pytest.mark.asyncio
async def test_qwen_computer_translates_drag_sequence() -> None:
    tool = QwenComputerTool(
        env_tool_name="computer",
        spec=QWEN_COMPUTER_SPEC,
        display_width=1000,
        display_height=500,
        description="computer",
    )
    environment = RecordingToolEnvironment(results={"computer": text_result("dragged")})

    await tool.execute(
        environment.call_tool,
        {"action": "left_click_drag", "coordinate": [10, 20]},
    )

    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("computer", {"action": "mouse_down", "button": "left"}),
        ("computer", {"action": "move", "x": 10, "y": 20}),
        ("computer", {"action": "mouse_up", "button": "left"}),
        ("computer", {"action": "screenshot"}),
    ]
