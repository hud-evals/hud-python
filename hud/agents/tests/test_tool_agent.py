"""``ToolAgent`` plumbing: catalog→clients, message formatting, dispatch + loop.

The provider-specific bits are abstract; this drives a tiny concrete subclass with a
scripted ``get_response`` so the loop, dispatch, and message formatting run offline.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

import fastmcp
import mcp.types as mcp_types
import pytest
from fastmcp.client.transports import SSETransport, StreamableHttpTransport
from typing_extensions import override

from hud.agents.openai.tools.coding import OpenAIShellTool
from hud.agents.openai.tools.mcp_proxy import OpenAIMCPProxyTool
from hud.agents.tool_agent import RunState, ToolAgent
from hud.agents.types import AgentConfig, AgentStep, ToolStep
from hud.capabilities import Capability, CapabilityClient, MCPClient, RFBClient, SSHClient
from hud.types import MCPToolCall, MCPToolResult, Step, Trace

if TYPE_CHECKING:
    from hud.eval.run import Run

_Msg = dict[str, Any]


class _FakeRun:
    """Offline stand-in for ``Run``: records steps onto a local trace only."""

    def __init__(self) -> None:
        self.trace = Trace()

    def record(self, step: Step) -> None:
        self.trace.record(step)


class DictAgent(ToolAgent[_Msg, AgentConfig]):
    """Minimal concrete ToolAgent over plain-dict messages."""

    def __init__(self, turns: list[AgentStep], **config: Any) -> None:
        self.config = AgentConfig(model="test-model", **config)
        self._turns = list(turns)

    @override
    async def _initialize_state(self, *, prompt: Any) -> RunState[_Msg]:
        return RunState(messages=self._initial_messages(prompt))

    @override
    async def get_response(
        self, state: RunState[_Msg], *, system_prompt: Any = None, citations_enabled: bool = False
    ) -> AgentStep:
        return self._turns.pop(0)

    @override
    def _format_message(self, role: str, text: str) -> _Msg:
        return {"role": role, "content": text}

    @override
    def _format_result(
        self, call: MCPToolCall, result: MCPToolResult, state: RunState[_Msg]
    ) -> _Msg:
        return {"role": "tool", "name": call.name, "isError": result.isError}


# ─── catalog → clients derivation ─────────────────────────────────────


def test_init_subclass_derives_clients_from_catalog() -> None:
    class WithCatalog(DictAgent):
        tool_catalog = (OpenAIShellTool,)

    assert WithCatalog.clients == (SSHClient,)


async def test_agent_opens_every_mcp_capability_by_name() -> None:
    capabilities = [
        Capability.mcp(name="database", url="http://database:8000/mcp"),
        Capability.mcp(name="search", url="http://search:8000/mcp"),
    ]
    opened: list[str] = []

    class Client:
        manifest = SimpleNamespace(bindings=capabilities)

        async def open(self, ref: str) -> CapabilityClient:
            opened.append(ref)
            return cast("CapabilityClient", object())

    class MultiMCPAgent(DictAgent):
        clients = (MCPClient,)

    class LiveRun(_FakeRun):
        def __init__(self) -> None:
            super().__init__()
            self.client = Client()
            self.prompt_messages: list[Any] = []

    await MultiMCPAgent([AgentStep(content="done", done=True)])(cast("Any", LiveRun()))

    assert opened == ["database", "search"]


async def test_agent_opens_only_one_non_mcp_capability_per_protocol() -> None:
    capabilities = [
        Capability.rfb(name="screen-0", url="rfb://display-0", display=0),
        Capability.rfb(name="screen-1", url="rfb://display-1", display=1),
    ]
    opened: list[str] = []

    class Client:
        manifest = SimpleNamespace(bindings=capabilities)

        async def open(self, ref: str) -> CapabilityClient:
            opened.append(ref)
            return cast("CapabilityClient", object())

    class ComputerAgent(DictAgent):
        clients = (RFBClient,)

    class LiveRun(_FakeRun):
        def __init__(self) -> None:
            super().__init__()
            self.client = Client()
            self.prompt_messages: list[Any] = []

    await ComputerAgent([AgentStep(content="done", done=True)])(cast("Any", LiveRun()))

    assert opened == ["screen-0"]


async def test_mcp_capability_names_do_not_collide_with_protocol_keys() -> None:
    capabilities = [
        Capability.mcp(name="ssh/2", url="http://database:8000/mcp"),
        Capability.ssh(name="shell", url="ssh://workspace", host_pubkey="key"),
    ]
    opened: list[str] = []

    class Client:
        manifest = SimpleNamespace(bindings=capabilities)

        async def open(self, ref: str) -> CapabilityClient:
            opened.append(ref)
            return cast("CapabilityClient", object())

    class MCPAndShellAgent(DictAgent):
        clients = (MCPClient, SSHClient)

    class LiveRun(_FakeRun):
        def __init__(self) -> None:
            super().__init__()
            self.client = Client()
            self.prompt_messages: list[Any] = []

    await MCPAndShellAgent([AgentStep(content="done", done=True)])(cast("Any", LiveRun()))

    assert opened == ["ssh/2", "shell"]


@pytest.mark.parametrize(
    ("transport", "expected_type"),
    [("sse", SSETransport), ("streamable-http", StreamableHttpTransport)],
)
async def test_mcp_client_uses_the_declared_http_transport(
    monkeypatch: pytest.MonkeyPatch,
    transport: Literal["sse", "streamable-http"],
    expected_type: type[SSETransport] | type[StreamableHttpTransport],
) -> None:
    transports: list[Any] = []

    class Client:
        def __init__(self, selected: Any, **_kwargs: Any) -> None:
            transports.append(selected)

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr(fastmcp, "Client", Client)
    capability = Capability.mcp(
        url="https://tools.example/events",
        transport=transport,
    )

    client = await MCPClient.connect(capability)
    await client.close()

    assert isinstance(transports[0], expected_type)


async def test_multiple_mcp_capabilities_qualify_tool_names() -> None:
    class Client(MCPClient):
        def __init__(self, *names: str) -> None:
            self.tools = [
                mcp_types.Tool(
                    name=name,
                    description=f"Run {name}",
                    inputSchema={"type": "object", "properties": {}},
                )
                for name in names
            ]
            self.calls: list[str] = []

        @override
        async def list_tools(self) -> list[mcp_types.Tool]:
            return self.tools

        @override
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
            self.calls.append(name)
            return MCPToolResult(content=[])

    class MultiMCPAgent(DictAgent):
        tool_catalog = (OpenAIMCPProxyTool,)

    database = Client("lookup", "write")
    search = Client("lookup", "find")
    tools, params = await MultiMCPAgent([])._build_tools({"database": database, "search": search})

    assert list(tools) == [
        "database__lookup",
        "database__write",
        "search__lookup",
        "search__find",
    ]
    assert [param["name"] for param in params] == list(tools)

    await tools["database__lookup"].execute({})
    assert database.calls == ["lookup"]
    assert search.calls == []

    single, _ = await MultiMCPAgent([])._build_tools({"database": Client("lookup")})
    assert list(single) == ["lookup"]


async def test_multiple_mcp_capabilities_reject_qualified_name_collisions() -> None:
    class Client(MCPClient):
        def __init__(self, name: str) -> None:
            self.tool = mcp_types.Tool(
                name=name,
                description=f"Run {name}",
                inputSchema={"type": "object", "properties": {}},
            )

        @override
        async def list_tools(self) -> list[mcp_types.Tool]:
            return [self.tool]

        @override
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
            raise AssertionError("colliding tools must not be callable")

    class MultiMCPAgent(DictAgent):
        tool_catalog = (OpenAIMCPProxyTool,)

    with pytest.raises(ValueError, match=r"MCP tool name collision.*a__b__c"):
        await MultiMCPAgent([])._build_tools({"a": Client("b__c"), "a__b": Client("c")})


async def test_qualified_mcp_names_are_valid_provider_tool_names() -> None:
    class Client(MCPClient):
        def __init__(self, name: str) -> None:
            self.tool = mcp_types.Tool(
                name=name,
                description=f"Run {name}",
                inputSchema={"type": "object", "properties": {}},
            )

        @override
        async def list_tools(self) -> list[mcp_types.Tool]:
            return [self.tool]

        @override
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
            return MCPToolResult(content=[])

    class MultiMCPAgent(DictAgent):
        tool_catalog = (OpenAIMCPProxyTool,)

    long_name = "lookup_" * 10
    tools, params = await MultiMCPAgent([])._build_tools(
        {
            "company.tools": Client(long_name + "a"),
            "company-tools": Client(long_name + "b"),
        }
    )

    assert len(tools) == 2
    assert set(tools) == {param["name"] for param in params}
    assert all(
        len(name) <= 64 and name.replace("_", "").replace("-", "").isalnum() for name in tools
    )


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


async def test_dispatch_unparsed_arguments_returns_error_result() -> None:
    # A call whose provider arguments never parsed is answered, not executed.
    agent = DictAgent([])
    call = MCPToolCall(name="bash", arguments='{"command": "')
    result = await agent._dispatch_call(call, RunState())
    assert result.isError is True
    content = result.content[0]
    assert isinstance(content, mcp_types.TextContent)
    assert "not executed" in content.text
    assert '{"command": "' in content.text  # the raw prefix re-anchors the model


async def test_loop_finishes_on_done_response() -> None:
    agent = DictAgent([AgentStep(content="final answer", done=True)])
    run = cast("Run", _FakeRun())

    await agent._loop(run, RunState(), max_steps=3)

    assert run.trace.status == "completed"
    assert run.trace.content == "final answer"
    assert run.trace.is_error is False
    assert run.trace.stop_reason == "done"
    assert run.trace.is_truncated is False
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
    run = cast("Run", _FakeRun())

    await agent._loop(run, RunState(), max_steps=3)

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
    run = cast("Run", _FakeRun())

    await agent._loop(run, RunState(), max_steps=2)

    assert run.trace.is_error is False
    assert run.trace.status == "completed"
    assert run.trace.stop_reason == "max_steps"
    assert run.trace.is_truncated is True
    # No synthetic error step — the trajectory ends on the real agent/tool steps.
    assert all(step.source != "system" for step in run.trace.steps)


async def test_loop_marks_length_finish_as_truncated() -> None:
    # A final turn cut off at the provider token cap (e.g. mid-tool-call) ends the
    # rollout normally but is a truncation, not a natural finish — across every
    # provider's finish-reason vocabulary.
    for finish_reason in ("length", "max_output_tokens", "max_tokens", "MAX_TOKENS"):
        agent = DictAgent([AgentStep(content="partial", done=True, finish_reason=finish_reason)])
        run = cast("Run", _FakeRun())

        await agent._loop(run, RunState(), max_steps=3)

        assert run.trace.status == "completed"
        assert run.trace.stop_reason == "length"
        assert run.trace.is_truncated is True


async def test_loop_answers_malformed_call_by_default() -> None:
    # Default "retry": the malformed call gets an error result and the loop continues.
    agent = DictAgent(
        [
            AgentStep(
                content="",
                done=False,
                tool_calls=[MCPToolCall(name="bash", arguments='{"command": "')],
            ),
            AgentStep(content="recovered", done=True),
        ]
    )
    run = cast("Run", _FakeRun())

    await agent._loop(run, RunState(), max_steps=3)

    assert run.trace.content == "recovered"
    tool_step = run.trace.steps[1]
    assert isinstance(tool_step, ToolStep)
    assert tool_step.result is not None
    assert tool_step.result.isError is True


async def test_loop_stops_on_malformed_call_when_configured() -> None:
    # The rollout ends at the malformed-call turn with nothing dispatched, and the
    # fired condition is the recorded stop reason.
    agent = DictAgent(
        [
            AgentStep(
                content="",
                done=False,
                tool_calls=[MCPToolCall(name="bash", arguments='{"command": "')],
            )
        ],
        stop_on={"malformed_tool_call"},
    )
    run = cast("Run", _FakeRun())

    await agent._loop(run, RunState(), max_steps=3)

    assert run.trace.status == "completed"
    assert run.trace.stop_reason == "malformed_tool_call"
    assert run.trace.is_truncated is True
    assert all(not isinstance(step, ToolStep) for step in run.trace.steps)


async def test_loop_stops_on_length_when_configured() -> None:
    # A token-capped turn ends the rollout even when its tool calls parsed.
    agent = DictAgent(
        [
            AgentStep(
                content="",
                done=False,
                finish_reason="length",
                tool_calls=[MCPToolCall(name="bash", arguments={"command": "ls"})],
            )
        ],
        stop_on={"length", "malformed_tool_call"},
    )
    run = cast("Run", _FakeRun())

    await agent._loop(run, RunState(), max_steps=3)

    assert run.trace.stop_reason == "length"
    assert run.trace.is_truncated is True
    assert all(not isinstance(step, ToolStep) for step in run.trace.steps)
