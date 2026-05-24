from __future__ import annotations

import asyncio

import pytest

from hud.agents.base import AgentContext
from hud.agents.tests.conftest import (
    HarnessConfig,
    RecordingToolEnvironment,
    ScriptedAgent,
    mcp_tool,
    text_prompt,
    text_result,
)
from hud.types import AgentResponse, MCPToolCall


@pytest.mark.asyncio
async def test_run_returns_final_response_without_tools() -> None:
    agent = ScriptedAgent([AgentResponse(content="done", done=True)])

    result = await agent.run(AgentContext(messages=[text_prompt("do it")]))

    assert result.done is True
    assert result.isError is False
    assert result.content == "done"
    assert agent.seen_messages == [[{"role": "user", "content": "do it"}]]


@pytest.mark.asyncio
async def test_run_executes_tool_call_and_continues_with_tool_result() -> None:
    environment = RecordingToolEnvironment(
        [mcp_tool("lookup")],
        results={"lookup": text_result("found it")},
    )
    agent = ScriptedAgent(
        [
            AgentResponse(
                tool_calls=[MCPToolCall(name="lookup", arguments={"query": "thing"})],
                done=False,
            ),
            AgentResponse(content="answer", done=True),
        ]
    )

    result = await agent.run(
        AgentContext(messages=[text_prompt("find thing")], tool_client=environment.client)
    )

    assert result.content == "answer"
    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("lookup", {"query": "thing"})
    ]
    assert agent.seen_messages[1][-1] == {
        "role": "tool",
        "name": "lookup",
        "content": "found it",
        "is_error": False,
    }


@pytest.mark.asyncio
async def test_run_supports_multiple_tool_steps_before_final_answer() -> None:
    environment = RecordingToolEnvironment(
        [mcp_tool("first"), mcp_tool("second")],
        results={"first": text_result("one"), "second": text_result("two")},
    )
    agent = ScriptedAgent(
        [
            AgentResponse(tool_calls=[MCPToolCall(name="first", arguments={})]),
            AgentResponse(tool_calls=[MCPToolCall(name="second", arguments={"n": 2})]),
            AgentResponse(content="finished", done=True),
        ]
    )

    result = await agent.run(
        AgentContext(messages=[text_prompt("go")], tool_client=environment.client)
    )

    assert result.content == "finished"
    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("first", {}),
        ("second", {"n": 2}),
    ]
    assert len(agent.seen_messages) == 3


@pytest.mark.asyncio
async def test_run_preserves_same_turn_tool_call_order() -> None:
    environment = RecordingToolEnvironment(
        [mcp_tool("first"), mcp_tool("second")],
        results={"first": text_result("one"), "second": text_result("two")},
    )
    agent = ScriptedAgent(
        [
            AgentResponse(
                tool_calls=[
                    MCPToolCall(name="first", arguments={"order": 1}),
                    MCPToolCall(name="second", arguments={"order": 2}),
                ]
            ),
            AgentResponse(content="finished", done=True),
        ]
    )

    result = await agent.run(
        AgentContext(messages=[text_prompt("call both")], tool_client=environment.client)
    )

    assert result.content == "finished"
    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("first", {"order": 1}),
        ("second", {"order": 2}),
    ]
    assert agent.seen_messages[1][-2:] == [
        {"role": "tool", "name": "first", "content": "one", "is_error": False},
        {"role": "tool", "name": "second", "content": "two", "is_error": False},
    ]


@pytest.mark.asyncio
async def test_unlimited_max_steps_runs_until_final_answer() -> None:
    environment = RecordingToolEnvironment([mcp_tool("loop")])
    agent = ScriptedAgent(
        [
            AgentResponse(tool_calls=[MCPToolCall(name="loop", arguments={"step": 1})]),
            AgentResponse(tool_calls=[MCPToolCall(name="loop", arguments={"step": 2})]),
            AgentResponse(content="done", done=True),
        ]
    )

    result = await agent.run(
        AgentContext(messages=[text_prompt("loop")], tool_client=environment.client),
        max_steps=-1,
    )

    assert result.content == "done"
    assert [call.arguments for call in environment.calls] == [{"step": 1}, {"step": 2}]


@pytest.mark.asyncio
async def test_tool_timeout_stops_run_with_error_trace() -> None:
    environment = RecordingToolEnvironment(
        [mcp_tool("slow")],
        results={"slow": TimeoutError("too slow")},
    )
    agent = ScriptedAgent([AgentResponse(tool_calls=[MCPToolCall(name="slow", arguments={})])])

    result = await agent.run(
        AgentContext(messages=[text_prompt("try slow")], tool_client=environment.client)
    )

    assert result.isError is True
    assert result.info["error"] == "too slow"
    assert [(call.name, call.arguments) for call in environment.calls] == [("slow", {})]


@pytest.mark.asyncio
async def test_tool_errors_are_returned_to_the_model_as_error_results() -> None:
    environment = RecordingToolEnvironment(
        [mcp_tool("lookup")],
        results={"lookup": RuntimeError("backend exploded")},
    )
    agent = ScriptedAgent(
        [
            AgentResponse(tool_calls=[MCPToolCall(name="lookup", arguments={})]),
            AgentResponse(content="recovered", done=True),
        ]
    )

    result = await agent.run(
        AgentContext(messages=[text_prompt("try")], tool_client=environment.client)
    )

    assert result.content == "recovered"
    assert agent.seen_messages[1][-1]["is_error"] is True
    assert agent.seen_messages[1][-1]["content"] == "backend exploded"


@pytest.mark.asyncio
async def test_missing_tool_client_turns_tool_call_into_error_trace() -> None:
    agent = ScriptedAgent([AgentResponse(tool_calls=[MCPToolCall(name="lookup", arguments={})])])

    result = await agent.run(AgentContext(messages=[text_prompt("call lookup")]))

    assert result.isError is True
    assert result.info["error"] == "call_tool callback is required to execute tool calls"


@pytest.mark.asyncio
async def test_max_steps_caps_tool_loop() -> None:
    environment = RecordingToolEnvironment([mcp_tool("lookup")])
    agent = ScriptedAgent(
        [
            AgentResponse(tool_calls=[MCPToolCall(name="lookup", arguments={})]),
            AgentResponse(content="should not be reached", done=True),
        ]
    )

    result = await agent.run(
        AgentContext(messages=[text_prompt("loop")], tool_client=environment.client),
        max_steps=1,
    )

    assert result.done is True
    assert result.content is None
    assert len(environment.calls) == 1
    assert len(agent.seen_messages) == 1


@pytest.mark.asyncio
async def test_auto_respond_can_continue_after_a_done_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str | None] = []

    async def continue_once(content: str | None, *, enabled: bool) -> object:
        calls.append(content)
        assert enabled is True
        if len(calls) > 1:
            return None
        return text_prompt("continue")

    monkeypatch.setattr("hud.agents.base.auto_respond", continue_once)
    agent = ScriptedAgent(
        [
            AgentResponse(content="need input", done=True),
            AgentResponse(content="final", done=True),
        ],
        config=HarnessConfig(auto_respond=True),
    )

    result = await agent.run(AgentContext(messages=[text_prompt("start")]))

    assert result.content == "final"
    assert calls == ["need input", "final"]
    assert agent.seen_messages[1][-1] == {"role": "user", "content": "continue"}


@pytest.mark.asyncio
async def test_model_step_exception_returns_error_trace() -> None:
    agent = ScriptedAgent([RuntimeError("model failed")])

    result = await agent.run(AgentContext(messages=[text_prompt("start")]))

    assert result.done is True
    assert result.isError is True
    assert result.content == "model failed"


@pytest.mark.asyncio
async def test_keyboard_interrupt_returns_interrupted_trace() -> None:
    agent = ScriptedAgent([KeyboardInterrupt()])

    result = await agent.run(AgentContext(messages=[text_prompt("start")]))

    assert result.isError is True
    assert result.content == "Interrupted by user"
    assert result.info["error"] == "Interrupted by user"


@pytest.mark.asyncio
async def test_cancelled_run_returns_cancelled_trace() -> None:
    agent = ScriptedAgent([asyncio.CancelledError()])

    result = await agent.run(AgentContext(messages=[text_prompt("start")]))

    assert result.isError is True
    assert result.content == "Cancelled"
    assert result.info["error"] == "Cancelled"


@pytest.mark.asyncio
async def test_trace_messages_include_provider_history_before_stop() -> None:
    environment = RecordingToolEnvironment(
        [mcp_tool("lookup")],
        results={"lookup": text_result("found")},
    )
    agent = ScriptedAgent(
        [
            AgentResponse(tool_calls=[MCPToolCall(name="lookup", arguments={})]),
            AgentResponse(content="done", done=True),
        ]
    )

    result = await agent.run(
        AgentContext(messages=[text_prompt("start")], tool_client=environment.client)
    )

    assert result.content == "done"
    assert result.messages == [
        {"role": "user", "content": "start"},
        {"role": "tool", "name": "lookup", "content": "found", "is_error": False},
    ]
