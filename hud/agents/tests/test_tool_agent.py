# pyright: reportPrivateUsage=false
"""``ToolAgent`` plumbing: catalog→clients, message formatting, dispatch + loop.

The provider-specific bits are abstract; this drives a tiny concrete subclass with a
scripted ``get_response`` so the loop, dispatch, and message formatting run offline.
"""

from __future__ import annotations

from typing import Any

import mcp.types as mcp_types

from hud.agents.openai.tools.coding import OpenAIShellTool
from hud.agents.tool_agent import RunState, ToolAgent
from hud.agents.types import AgentConfig, AgentStep, ToolStep
from hud.capabilities import SSHClient
from hud.types import MCPToolCall, MCPToolResult, Step, Trace

_Msg = dict[str, Any]


class _FakeRun:
    """Offline stand-in for ``Run``: records steps onto a local trace only."""

    def __init__(self) -> None:
        self.trace = Trace()

    def record(self, step: Step) -> None:
        self.trace.record(step)


class DictAgent(ToolAgent[_Msg, AgentConfig]):
    """Minimal concrete ToolAgent over plain-dict messages."""

    def __init__(self, turns: list[AgentStep]) -> None:
        self.config = AgentConfig(model="test-model")
        self._turns = list(turns)

    async def _initialize_state(self, *, prompt: Any) -> RunState[_Msg]:
        return RunState(messages=self._initial_messages(prompt))

    async def get_response(
        self, state: RunState[_Msg], *, system_prompt: Any = None, citations_enabled: bool = False
    ) -> AgentStep:
        return self._turns.pop(0)

    def _format_message(self, role: str, text: str) -> _Msg:
        return {"role": role, "content": text}

    def _format_result(
        self, call: MCPToolCall, result: MCPToolResult, state: RunState[_Msg]
    ) -> _Msg:
        return {"role": "tool", "name": call.name, "isError": result.isError}


# ─── catalog → clients derivation ─────────────────────────────────────


def test_init_subclass_derives_clients_from_catalog() -> None:
    class WithCatalog(DictAgent):
        tool_catalog = (OpenAIShellTool,)

    assert WithCatalog.clients == (SSHClient,)


# ─── initial messages / user text formatting ──────────────────────────


def test_initial_messages_formats_each_turn() -> None:
    agent = DictAgent([])
    turn = mcp_types.PromptMessage(
        role="user", content=mcp_types.TextContent(type="text", text="a")
    )
    assert agent._initial_messages([turn]) == [{"role": "user", "content": "a"}]
    assert agent._format_user_text("hey") == {"role": "user", "content": "hey"}


# ─── dispatch + loop ──────────────────────────────────────────────────


async def test_dispatch_unknown_tool_returns_error_result() -> None:
    agent = DictAgent([])
    result = await agent._dispatch_call(MCPToolCall(name="ghost"), RunState())
    assert result.isError is True


async def test_loop_finishes_on_done_response() -> None:
    agent = DictAgent([AgentStep(content="final answer", done=True)])
    run = _FakeRun()

    await agent._loop(run, RunState(), max_steps=3)  # type: ignore[arg-type]

    assert run.trace.status == "completed"
    assert run.trace.content == "final answer"
    assert run.trace.is_error is False
    # The agent turn was recorded directly, with loop-stamped fallbacks.
    (step,) = run.trace.steps
    assert isinstance(step, AgentStep)
    assert step.source == "agent"
    assert step.content == "final answer"
    assert step.model == "test-model"
    assert step.started_at is not None


async def test_loop_dispatches_tool_calls_then_finishes() -> None:
    agent = DictAgent(
        [
            AgentStep(content="", done=False, tool_calls=[MCPToolCall(name="ghost")]),
            AgentStep(content="done now", done=True),
        ]
    )
    run = _FakeRun()

    await agent._loop(run, RunState(), max_steps=3)  # type: ignore[arg-type]

    assert run.trace.content == "done now"
    assert [step.source for step in run.trace.steps] == ["agent", "tool", "agent"]
    # the (unknown) tool call produced an observed tool step in the trajectory
    tool_step = run.trace.steps[1]
    assert isinstance(tool_step, ToolStep)
    assert tool_step.call is not None
    assert tool_step.call.name == "ghost"
    assert tool_step.result is not None
    assert tool_step.result.isError is True  # unknown tool → error result


async def test_loop_max_steps_is_normal_termination() -> None:
    # Always returns a tool call → never "done" → hits max_steps. Exhausting the
    # configured budget is a stop reason, not an agent error (the platform must
    # not paint the rollout or its last tool call as failed).
    never_done = [
        AgentStep(content="", done=False, tool_calls=[MCPToolCall(name="ghost")]) for _ in range(5)
    ]
    agent = DictAgent(never_done)
    run = _FakeRun()

    await agent._loop(run, RunState(), max_steps=2)  # type: ignore[arg-type]

    assert run.trace.is_error is False
    assert run.trace.status == "completed"
    assert run.trace.extra.get("stop_reason") == "max_steps"
    # No synthetic error step — the trajectory ends on the real agent/tool steps.
    assert all(step.source != "system" for step in run.trace.steps)
