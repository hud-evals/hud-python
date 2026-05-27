from __future__ import annotations

import pytest

from hud.agents.tests.conftest import (
    RecordingToolEnvironment,
    RoutingHarnessTools,
    mcp_tool,
    text_result,
)
from hud.types import MCPToolCall


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
async def test_tool_capability_metadata_routes_native_tool() -> None:
    environment = RecordingToolEnvironment([mcp_tool("bash", meta={"capability": "shell"})])
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
async def test_native_tool_takes_precedence_over_generic_tool_with_same_environment_name() -> None:
    environment = RecordingToolEnvironment(
        [mcp_tool("bash", meta={"capability": "shell"}), mcp_tool("lookup")]
    )
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


@pytest.mark.asyncio
async def test_tool_name_does_not_imply_native_capability() -> None:
    environment = RecordingToolEnvironment([mcp_tool("bash")])
    agent_tools = RoutingHarnessTools()
    agent_tools.prepare(model="test-model", tools=environment.tools)

    with pytest.raises(KeyError):
        await agent_tools.execute(
            environment.call_tool,
            MCPToolCall(name="shell", arguments={"command": "pwd"}),
        )

    assert environment.calls == []
