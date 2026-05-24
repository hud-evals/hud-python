from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from hud.agents.tests.conftest import (
    RecordingToolEnvironment,
    RoutingHarnessTools,
    mcp_tool,
    text_result,
)
from hud.agents.tools.capabilities import discover_environment_capabilities
from hud.types import MCPToolCall

if TYPE_CHECKING:
    from hud.agents.tools import ToolMetadata


@pytest.mark.asyncio
async def test_generic_tool_call_routes_to_matching_environment_tool() -> None:
    environment = RecordingToolEnvironment(
        [mcp_tool("lookup")],
        results={"lookup": text_result("found")},
    )
    agent_tools = RoutingHarnessTools()
    agent_tools.prepare(model="test-model", tools=environment.tools)

    outputs = await agent_tools.execute(
        environment.call_tool,
        MCPToolCall(name="lookup", arguments={"query": "hud"}),
    )

    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("lookup", {"query": "hud"})
    ]
    assert outputs == [{"role": "tool", "name": "lookup", "content": "found", "is_error": False}]


@pytest.mark.asyncio
async def test_capability_metadata_routes_provider_tool_to_environment_tool() -> None:
    environment = RecordingToolEnvironment([mcp_tool("run_shell")])
    agent_tools = RoutingHarnessTools()
    agent_tools.prepare(
        model="test-model",
        tools=environment.tools,
        tool_metadata={"capabilities": {"shell": "run_shell"}},
    )

    await agent_tools.execute(
        environment.call_tool,
        MCPToolCall(name="shell", arguments={"command": "pwd"}),
    )

    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("run_shell", {"command": "pwd"})
    ]


@pytest.mark.asyncio
async def test_name_fallback_routes_native_tool_when_metadata_is_absent() -> None:
    environment = RecordingToolEnvironment([mcp_tool("bash")])
    agent_tools = RoutingHarnessTools()
    agent_tools.prepare(model="test-model", tools=environment.tools)

    await agent_tools.execute(
        environment.call_tool,
        MCPToolCall(name="shell", arguments={"command": "echo hi"}),
    )

    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("bash", {"command": "echo hi"})
    ]


@pytest.mark.asyncio
async def test_grouped_capability_metadata_routes_to_the_selected_environment_tool() -> None:
    environment = RecordingToolEnvironment([mcp_tool("read"), mcp_tool("grep")])
    agent_tools = RoutingHarnessTools()
    agent_tools.prepare(
        model="test-model",
        tools=environment.tools,
        tool_metadata={"capabilities": {"filesystem": {"tools": {"read": "read", "grep": "grep"}}}},
    )

    await agent_tools.execute(
        environment.call_tool,
        MCPToolCall(name="read_file", arguments={"path": "README.md"}),
    )

    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("read", {"path": "README.md"})
    ]


@pytest.mark.asyncio
async def test_native_tool_takes_precedence_over_generic_tool_with_same_environment_name() -> None:
    environment = RecordingToolEnvironment([mcp_tool("bash"), mcp_tool("lookup")])
    agent_tools = RoutingHarnessTools()
    agent_tools.prepare(model="test-model", tools=environment.tools)

    await agent_tools.execute(
        environment.call_tool,
        MCPToolCall(name="shell", arguments={"command": "whoami"}),
    )

    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("bash", {"command": "whoami"})
    ]
    with pytest.raises(KeyError):
        await agent_tools.execute(
            environment.call_tool,
            MCPToolCall(name="bash", arguments={"command": "whoami"}),
        )


@pytest.mark.asyncio
async def test_unknown_provider_tool_fails_before_environment_execution() -> None:
    environment = RecordingToolEnvironment([mcp_tool("lookup")])
    agent_tools = RoutingHarnessTools()
    agent_tools.prepare(model="test-model", tools=environment.tools)

    with pytest.raises(KeyError):
        await agent_tools.execute(
            environment.call_tool,
            MCPToolCall(name="missing", arguments={}),
        )

    assert environment.calls == []


@pytest.mark.asyncio
async def test_timeout_error_propagates_to_run_loop_boundary() -> None:
    environment = RecordingToolEnvironment(
        [mcp_tool("lookup")],
        results={"lookup": TimeoutError("tool timed out")},
    )
    agent_tools = RoutingHarnessTools()
    agent_tools.prepare(model="test-model", tools=environment.tools)

    with pytest.raises(TimeoutError, match="tool timed out"):
        await agent_tools.execute(
            environment.call_tool,
            MCPToolCall(name="lookup", arguments={}),
        )


def test_invalid_capability_metadata_fails_at_the_boundary() -> None:
    with pytest.raises(ValueError, match="Invalid capability metadata"):
        discover_environment_capabilities(
            [mcp_tool("lookup")],
            tool_metadata=cast(
                "ToolMetadata",
                {"capabilities": {"lookup": {"unexpected": "shape"}}},
            ),
        )


@pytest.mark.asyncio
async def test_stale_capability_metadata_falls_back_to_available_tool_names() -> None:
    environment = RecordingToolEnvironment([mcp_tool("bash")])
    agent_tools = RoutingHarnessTools()
    agent_tools.prepare(
        model="test-model",
        tools=environment.tools,
        tool_metadata={"capabilities": {"shell": "missing_shell"}},
    )

    await agent_tools.execute(
        environment.call_tool,
        MCPToolCall(name="shell", arguments={"command": "pwd"}),
    )

    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("bash", {"command": "pwd"})
    ]
