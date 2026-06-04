# pyright: reportPrivateUsage=false
"""``ToolAgent`` plumbing: prompt normalization, catalog→clients, dispatch + loop.

The provider-specific bits are abstract; this drives a tiny concrete subclass with a
scripted ``get_response`` so the loop, dispatch, and message formatting run offline.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mcp.types as mcp_types

from hud.agents.openai.tools.coding import OpenAIShellTool
from hud.agents.tool_agent import RunState, ToolAgent, to_prompt_messages
from hud.capabilities import SSHClient
from hud.types import AgentResponse, MCPToolCall, MCPToolResult, Trace

_Msg = dict[str, Any]


class DictAgent(ToolAgent[_Msg]):
    """Minimal concrete ToolAgent over plain-dict messages."""

    def __init__(self, responses: list[AgentResponse]) -> None:
        self.model = "test-model"
        self.auto_respond = False
        self.hosted_tools = []
        self._responses = list(responses)

    async def _initialize_state(self, *, prompt: Any) -> RunState[_Msg]:
        return RunState(messages=self._initial_messages(prompt))

    async def get_response(
        self, state: RunState[_Msg], *, system_prompt: Any = None, citations_enabled: bool = False
    ) -> AgentResponse:
        return self._responses.pop(0)

    def _format_message(self, role: str, text: str) -> _Msg:
        return {"role": role, "content": text}

    def _format_result(
        self, call: MCPToolCall, result: MCPToolResult, state: RunState[_Msg]
    ) -> _Msg:
        return {"role": "tool", "name": call.name, "isError": result.isError}


# ─── to_prompt_messages ───────────────────────────────────────────────


def test_to_prompt_messages_wraps_plain_text() -> None:
    msgs = to_prompt_messages("hello")
    assert len(msgs) == 1
    assert msgs[0].role == "user"
    assert isinstance(msgs[0].content, mcp_types.TextContent)
    assert msgs[0].content.text == "hello"


def test_to_prompt_messages_none_is_empty_user_turn() -> None:
    assert to_prompt_messages(None)[0].content.text == ""  # type: ignore[union-attr]


def test_to_prompt_messages_normalizes_dicts_and_passthrough() -> None:
    existing = mcp_types.PromptMessage(
        role="assistant", content=mcp_types.TextContent(type="text", text="prior")
    )
    msgs = to_prompt_messages(
        [{"role": "user", "content": {"type": "text", "text": "hi"}}, existing],
    )
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1] is existing


# ─── catalog → clients derivation ─────────────────────────────────────


def test_init_subclass_derives_clients_from_catalog() -> None:
    class WithCatalog(DictAgent):
        tool_catalog = (OpenAIShellTool,)

    assert WithCatalog.clients == (SSHClient,)


# ─── initial messages / user text formatting ──────────────────────────


def test_initial_messages_formats_each_turn() -> None:
    agent = DictAgent([])
    msgs = agent._initial_messages([{"role": "user", "content": {"type": "text", "text": "a"}}])
    assert msgs == [{"role": "user", "content": "a"}]
    assert agent._format_user_text("hey") == {"role": "user", "content": "hey"}


# ─── dispatch + loop ──────────────────────────────────────────────────


async def test_dispatch_unknown_tool_returns_error_result() -> None:
    agent = DictAgent([])
    result = await agent._dispatch_call(MCPToolCall(name="ghost"), RunState())
    assert result.isError is True


async def test_loop_finishes_on_done_response() -> None:
    agent = DictAgent([AgentResponse(content="final answer", done=True)])
    run = SimpleNamespace(trace=Trace())

    await agent._loop(run, RunState(), max_steps=3)  # type: ignore[arg-type]

    assert run.trace.done is True
    assert run.trace.content == "final answer"
    assert run.trace.isError is False


async def test_loop_dispatches_tool_calls_then_finishes() -> None:
    agent = DictAgent(
        [
            AgentResponse(content="", done=False, tool_calls=[MCPToolCall(name="ghost")]),
            AgentResponse(content="done now", done=True),
        ]
    )
    run = SimpleNamespace(trace=Trace())

    await agent._loop(run, RunState(), max_steps=3)  # type: ignore[arg-type]

    assert run.trace.content == "done now"
    # the (unknown) tool call produced a tool message in the trajectory
    assert any(m.get("role") == "tool" for m in run.trace.messages)


async def test_loop_flags_max_steps_exceeded() -> None:
    # Always returns a tool call → never "done" → hits max_steps.
    never_done = [
        AgentResponse(content="", done=False, tool_calls=[MCPToolCall(name="ghost")])
        for _ in range(5)
    ]
    agent = DictAgent(never_done)
    run = SimpleNamespace(trace=Trace())

    await agent._loop(run, RunState(), max_steps=2)  # type: ignore[arg-type]

    assert run.trace.isError is True
    assert run.trace.info.get("error") == "max_steps_exceeded"
