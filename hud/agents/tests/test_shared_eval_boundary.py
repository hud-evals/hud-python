from __future__ import annotations

import pytest
from mcp import types

from hud.agents.tests.conftest import (
    HarnessEvalContext,
    RoutingHarnessTools,
    ScriptedAgent,
    mcp_tool,
    text_prompt,
    text_result,
)
from hud.types import AgentResponse, MCPToolCall, Trace


@pytest.mark.asyncio
async def test_eval_run_submits_final_content() -> None:
    ctx = HarnessEvalContext(prompt="Do the task")
    agent = ScriptedAgent([AgentResponse(content="answer", done=True)])

    result = await ctx.run_agent(agent)

    assert result.content == "answer"
    assert ctx.submitted == "answer"


@pytest.mark.asyncio
async def test_eval_run_submits_citations_with_content() -> None:
    citations = [{"type": "url", "source": "https://example.com"}]
    ctx = HarnessEvalContext(prompt="Do the task")
    agent = ScriptedAgent(
        [AgentResponse(content="answer with sources", citations=citations, done=True)]
    )

    result = await ctx.run_agent(agent)

    assert result.citations == citations
    assert ctx.submitted == {"content": "answer with sources", "citations": citations}


@pytest.mark.asyncio
async def test_eval_run_does_not_submit_empty_content() -> None:
    ctx = HarnessEvalContext(prompt="Do the task")
    agent = ScriptedAgent([AgentResponse(content="", done=True)])

    result = await ctx.run_agent(agent)

    assert result.content == ""
    assert ctx.submitted is None


@pytest.mark.asyncio
async def test_eval_run_records_error_without_submission() -> None:
    ctx = HarnessEvalContext(prompt="Do the task")
    agent = ScriptedAgent([AgentResponse(content="bad", isError=True, done=True)])

    result = await ctx.run_agent(agent)

    assert result.isError is True
    assert isinstance(ctx.error, Exception)
    assert str(ctx.error) == "bad"
    assert ctx.submitted is None


@pytest.mark.asyncio
async def test_eval_run_requires_prompt_when_no_conversation_or_scenario_messages() -> None:
    ctx = HarnessEvalContext(prompt="")
    agent = ScriptedAgent([AgentResponse(content="unused", done=True)])

    with pytest.raises(ValueError, match=r"ctx\.prompt is not set"):
        await ctx.run_agent(agent)


@pytest.mark.asyncio
async def test_prompt_messages_prefer_scenario_messages_over_conversation_and_prompt() -> None:
    scenario_message = text_prompt("scenario message", role="assistant")
    ctx = HarnessEvalContext(prompt="fallback prompt")
    ctx.conversation = [{"role": "user", "content": "conversation message"}]
    ctx.set_scenario_messages([scenario_message])
    agent = ScriptedAgent([AgentResponse(content="answer", done=True)])

    await ctx.run_agent(agent)

    assert agent.seen_messages[0] == [{"role": "assistant", "content": "scenario message"}]


@pytest.mark.asyncio
async def test_prompt_messages_use_conversation_before_prompt() -> None:
    ctx = HarnessEvalContext(prompt="fallback prompt")
    ctx.conversation = [
        {"role": "assistant", "content": "previous"},
        {"role": "user", "content": "next"},
    ]
    agent = ScriptedAgent([AgentResponse(content="answer", done=True)])

    await ctx.run_agent(agent)

    assert agent.seen_messages[0] == [
        {"role": "assistant", "content": "previous"},
        {"role": "user", "content": "next"},
    ]


@pytest.mark.asyncio
async def test_eval_run_passes_citation_flag_to_agent() -> None:
    ctx = HarnessEvalContext(prompt="Do the task")
    ctx.enable_citations = True
    agent = ScriptedAgent([AgentResponse(content="answer", done=True)])

    await ctx.run_agent(agent)

    assert agent.enable_citations is True


@pytest.mark.asyncio
async def test_eval_run_executes_environment_tool_and_submits_final_answer() -> None:
    ctx = HarnessEvalContext(
        prompt="Use a tool",
        tools=[mcp_tool("lookup")],
        tool_results={"lookup": text_result("looked up")},
    )
    agent = ScriptedAgent(
        [
            AgentResponse(tool_calls=[MCPToolCall(name="lookup", arguments={"q": "hud"})]),
            AgentResponse(content="answer", done=True),
        ]
    )

    result = await ctx.run_agent(agent)

    assert result.content == "answer"
    assert ctx.submitted == "answer"
    assert [(call.name, call.arguments) for call in ctx.environment.calls] == [
        ("lookup", {"q": "hud"})
    ]


@pytest.mark.asyncio
async def test_eval_tool_capability_routes_native_provider_tool_to_environment_tool() -> None:
    ctx = HarnessEvalContext(
        prompt="Use shell",
        tools=[mcp_tool("run_shell", meta={"capability": "shell"})],
    )
    agent = ScriptedAgent(
        [
            AgentResponse(tool_calls=[MCPToolCall(name="shell", arguments={"command": "pwd"})]),
            AgentResponse(content="done", done=True),
        ],
        tools_factory=RoutingHarnessTools,
    )

    result = await ctx.run_agent(agent)

    assert result.content == "done"
    assert [(call.name, call.arguments) for call in ctx.environment.calls] == [
        ("run_shell", {"command": "pwd"})
    ]


@pytest.mark.asyncio
async def test_eval_run_passes_max_steps_to_agent_run() -> None:
    ctx = HarnessEvalContext(prompt="Use a tool", tools=[mcp_tool("lookup")])
    agent = ScriptedAgent(
        [
            AgentResponse(tool_calls=[MCPToolCall(name="lookup", arguments={})]),
            AgentResponse(content="too late", done=True),
        ]
    )

    result = await ctx.run_agent(agent, max_steps=1)

    assert result.isError is True
    assert result.content == "Max steps exceeded"
    assert result.info["error"] == "max_steps_exceeded"
    assert ctx.submitted is None
    assert [(call.name, call.arguments) for call in ctx.environment.calls] == [("lookup", {})]


@pytest.mark.asyncio
async def test_eval_run_records_agent_step_error_on_context() -> None:
    ctx = HarnessEvalContext(prompt="Do the task")
    agent = ScriptedAgent([RuntimeError("agent failed")])

    result = await ctx.run_agent(agent)

    assert result.isError is True
    assert isinstance(ctx.error, Exception)
    assert str(ctx.error) == "agent failed"
    assert ctx.submitted is None


@pytest.mark.asyncio
async def test_submit_result_error_prefers_info_error_message() -> None:
    ctx = HarnessEvalContext(prompt="Do the task")

    result = Trace(isError=True, content="fallback", info={"error": "specific"})

    await ctx.submit_result(result)

    assert isinstance(ctx.error, Exception)
    assert str(ctx.error) == "specific"


def test_prompt_falls_back_to_plain_user_message() -> None:
    ctx = HarnessEvalContext(prompt="hello")

    messages = ctx.prompt_messages()

    assert messages == [
        types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text="hello"),
        )
    ]
