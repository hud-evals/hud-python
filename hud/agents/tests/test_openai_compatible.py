from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import mcp.types as types
import pytest

from hud.agents.openai_chat import OpenAIChatAgent
from hud.agents.openai_compatible.tools import openai_compatible_tools
from hud.agents.openai_compatible.tools.computer import (
    GLMComputerTool,
    QwenComputerTool,
    _fix_glm_xml_args,
    _parse_glm_box,
)
from hud.agents.openai_compatible.tools.filesystem import ReadTool
from hud.agents.tools import EnvironmentCapability
from hud.types import MCPToolCall, MCPToolResult

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam


def computer_tool(name: str = "computer") -> types.Tool:
    return types.Tool(
        name=name,
        description="Control computer with mouse, keyboard, and screenshots",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["action"],
        },
        _meta={"resolution": {"width": 1024, "height": 768}},
    )


def capability(tool: types.Tool) -> EnvironmentCapability:
    return EnvironmentCapability(name="computer", tool_name=tool.name, tool=tool)


def filesystem_tool(name: str) -> types.Tool:
    return types.Tool(
        name=name,
        description=f"{name} environment tool",
        inputSchema={"type": "object", "properties": {}},
    )


def filesystem_capability(tool_name: str = "read") -> EnvironmentCapability:
    tool = filesystem_tool(tool_name)
    return EnvironmentCapability(
        name="filesystem",
        tool_name=tool.name,
        tool=tool,
        metadata={"tools": {"read": "read", "grep": "grep", "glob": "glob", "list": "list"}},
    )


def test_openai_compatible_agent_uses_glm_computer_tool() -> None:
    agent = OpenAIChatAgent.create(
        model="glm-4.6v",
        api_key="test-key",
        base_url="http://example.com/v1",
    )
    tool = computer_tool()
    agent._available_tools = [tool]
    agent._categorized_tools = agent.categorize_tools([tool])
    agent._initialized = True
    agent._on_tools_ready()

    schemas = agent.get_tool_schemas()
    schema = cast("dict[str, Any]", schemas[0])

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "computer"
    assert len(schemas) == 1
    assert "computer" in agent._openai_compatible_native_tools
    actions = schema["function"]["parameters"]["properties"]["action"]["enum"]
    assert "DONE" not in actions
    assert "FAIL" not in actions


def test_openai_compatible_agent_uses_qwen_computer_tool() -> None:
    agent = OpenAIChatAgent.create(
        model="qwen2.5-vl",
        api_key="test-key",
        base_url="http://example.com/v1",
    )
    tool = computer_tool()
    agent._available_tools = [tool]
    agent._categorized_tools = agent.categorize_tools([tool])
    agent._initialized = True
    agent._on_tools_ready()

    schemas = agent.get_tool_schemas()
    schema = cast("dict[str, Any]", schemas[0])

    assert schema["type"] == "computer_use"
    assert schema["name"] == "computer_use"
    assert len(schemas) == 1
    assert "computer_use" in agent._openai_compatible_native_tools
    actions = schema["parameters"]["properties"]["action"]["enum"]
    assert "terminate" not in actions
    assert "answer" not in actions


def test_openai_compatible_registry_ignores_legacy_native_metadata() -> None:
    tool = types.Tool(
        name="glm_computer",
        description="legacy GLM computer",
        inputSchema={"type": "object", "properties": {}},
        _meta={
            "native_tools": {
                "openai_compatible": {
                    "api_type": "gui_agent_glm45v",
                    "api_name": "computer",
                    "role": "computer",
                }
            }
        },
    )
    agent = OpenAIChatAgent.create(
        model="glm-4.6v",
        api_key="test-key",
        base_url="http://example.com/v1",
    )

    categorized = agent.categorize_tools([tool])

    assert categorized.generic == [tool]
    assert categorized.skipped == []


def test_openai_compatible_agent_uses_filesystem_tool_shapes() -> None:
    agent = OpenAIChatAgent.create(
        model="gpt-4o",
        api_key="test-key",
        base_url="http://example.com/v1",
    )
    tools = [filesystem_tool(name) for name in ("read", "grep", "glob", "list")]
    agent._available_tools = tools
    agent._categorized_tools = agent.categorize_tools(tools)
    agent._initialized = True
    agent._on_tools_ready()

    schemas = agent.get_tool_schemas()
    function_schemas = [cast("ChatCompletionToolParam", schema) for schema in schemas]

    assert [schema["function"]["name"] for schema in function_schemas] == [
        "read",
        "grep",
        "glob",
        "list",
    ]
    assert len(schemas) == 4
    assert set(agent._openai_compatible_backing_tools) == {"read", "grep", "glob", "list"}
    filesystem = agent._environment_capabilities["filesystem"]
    assert filesystem.metadata["tools"] == {
        "read": "read",
        "grep": "grep",
        "glob": "glob",
        "list": "list",
    }


def test_openai_compatible_registry_maps_filesystem_capability_to_read_tool() -> None:
    tool = openai_compatible_tools.tool_for_capability(
        filesystem_capability(),
        "gpt-4o",
    )

    assert isinstance(tool, ReadTool)
    assert tool.to_params()["function"]["name"] == "read"


def test_parse_glm_box() -> None:
    assert _parse_glm_box("[513,438]") == (513, 438)
    assert _parse_glm_box("513, 438") == (513, 438)
    assert _parse_glm_box([513, 438]) == (513, 438)
    assert _parse_glm_box([[513, 438]]) == (513, 438)
    assert _parse_glm_box("bad") is None


def test_fix_glm_xml_args() -> None:
    result = _fix_glm_xml_args(
        {"action": "left_click\n<arg_key>start_box</arg_key>\n<arg_value>[114, 167]"}
    )

    assert result == {"action": "left_click", "start_box": "[114, 167]"}


@pytest.mark.asyncio
async def test_glm_computer_translates_to_environment_calls() -> None:
    tool = GLMComputerTool.from_capability(
        capability(computer_tool()),
        GLMComputerTool.default_spec("glm-4.6v"),  # type: ignore[arg-type]
        "glm-4.6v",
    )
    calls: list[MCPToolCall] = []

    async def caller(call: MCPToolCall) -> MCPToolResult:
        calls.append(call)
        return MCPToolResult(content=[], isError=False)

    await tool.execute(caller, {"action": "left_click", "start_box": "[500,300]"})

    assert calls[0].name == "computer"
    assert calls[0].arguments == {
        "action": "click",
        "x": 512,
        "y": 230,
        "button": "left",
    }
    assert calls[1].arguments == {"action": "screenshot"}


@pytest.mark.asyncio
async def test_qwen_computer_translates_to_environment_calls() -> None:
    tool = QwenComputerTool.from_capability(
        capability(computer_tool()),
        QwenComputerTool.default_spec("qwen2.5-vl"),  # type: ignore[arg-type]
        "qwen2.5-vl",
    )
    calls: list[MCPToolCall] = []

    async def caller(call: MCPToolCall) -> MCPToolResult:
        calls.append(call)
        return MCPToolResult(content=[], isError=False)

    await tool.execute(caller, {"action": "scroll", "coordinate": [100, 200], "pixels": 50})

    assert calls[0].name == "computer"
    assert calls[0].arguments == {
        "action": "scroll",
        "x": 100,
        "y": 200,
        "scroll_y": -50,
    }
    assert calls[1].arguments == {"action": "screenshot"}


@pytest.mark.asyncio
async def test_qwen_left_click_drag_uses_mouse_drag_sequence() -> None:
    tool = QwenComputerTool.from_capability(
        capability(computer_tool()),
        QwenComputerTool.default_spec("qwen2.5-vl"),  # type: ignore[arg-type]
        "qwen2.5-vl",
    )
    calls: list[MCPToolCall] = []

    async def caller(call: MCPToolCall) -> MCPToolResult:
        calls.append(call)
        return MCPToolResult(content=[], isError=False)

    await tool.execute(caller, {"action": "left_click_drag", "coordinate": [300, 400]})

    assert [call.name for call in calls] == ["computer", "computer", "computer", "computer"]
    assert [call.arguments for call in calls] == [
        {"action": "mouse_down", "button": "left"},
        {"action": "move", "x": 300, "y": 400},
        {"action": "mouse_up", "button": "left"},
        {"action": "screenshot"},
    ]


@pytest.mark.asyncio
async def test_openai_compatible_filesystem_tool_forwards_to_environment_tool() -> None:
    tool = ReadTool.from_capability(
        filesystem_capability(),
        ReadTool.default_spec("gpt-4o"),
        "gpt-4o",
    )
    calls: list[MCPToolCall] = []

    async def caller(call: MCPToolCall) -> MCPToolResult:
        calls.append(call)
        return MCPToolResult(content=[], isError=False)

    await tool.execute(caller, {"filePath": "/workspace/app.py", "offset": 10, "limit": 5})

    assert len(calls) == 1
    assert calls[0].name == "read"
    assert calls[0].arguments == {"filePath": "/workspace/app.py", "offset": 10, "limit": 5}


def test_openai_compatible_tool_registry_selects_model_specific_tool() -> None:
    tool = computer_tool()
    cap = capability(tool)

    glm_tool = openai_compatible_tools.tool_for_capability(cap, "glm-4.6v")
    qwen_tool = openai_compatible_tools.tool_for_capability(cap, "qwen2.5-vl")
    unsupported = openai_compatible_tools.tool_for_capability(cap, "llama")

    assert isinstance(glm_tool, GLMComputerTool)
    assert isinstance(qwen_tool, QwenComputerTool)
    assert unsupported is None
