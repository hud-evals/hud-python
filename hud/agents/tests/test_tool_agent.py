"""``ToolAgent`` plumbing: catalog→clients, message formatting, dispatch + loop.

The provider-specific bits are abstract; this drives a tiny concrete subclass with a
scripted ``get_response`` so the loop, dispatch, and message formatting run offline.
"""

from __future__ import annotations

import asyncio
import random
import sys
from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import AsyncMock, Mock

import fastmcp
import mcp.types as mcp_types
import pytest
from fastmcp.client.transports import SSETransport, StreamableHttpTransport
from pydantic import ValidationError

from hud.agents.claude.agent import ClaudeAgent
from hud.agents.claude.tools.coding import ClaudeBashTool, ClaudeTextEditorTool
from hud.agents.openai.agent import OpenAIAgent
from hud.agents.openai.tools.coding import OpenAIShellTool
from hud.agents.openai.tools.mcp_proxy import OpenAIMCPProxyTool
from hud.agents.tool_agent import DegenerateTurnError, RunState, ToolAgent
from hud.agents.tools.base import AgentTool, AgentToolSpec, result_text
from hud.agents.tools.rfb import RFBTool
from hud.agents.tools.ssh import SSHInfrastructureErrorResult
from hud.agents.types import (
    AgentStep,
    ClaudeCLIConfig,
    ClaudeConfig,
    OpenAIConfig,
    ToolAgentConfig,
    ToolStep,
)
from hud.capabilities import (
    Capability,
    CapabilityClient,
    MCPClient,
    RFBClient,
    SSHClient,
)
from hud.capabilities.mcp import get_mcp_trace_id
from hud.capabilities.rfb import PngScreenshotEncoding, WebPScreenshotEncoding
from hud.capabilities.ssh import SSHConnectionError
from hud.environment.workspace import Workspace
from hud.telemetry.context import set_trace_context
from hud.types import MCPToolCall, MCPToolResult, Step, Trace

if TYPE_CHECKING:
    from pathlib import Path

    from hud.eval.run import Run

_Msg = dict[str, Any]


class _FakeRun:
    """Offline stand-in for ``Run``: records steps onto a local trace only."""

    def __init__(self) -> None:
        self.trace = Trace()

    def record(self, step: Step) -> None:
        self.trace.record(step)


class DictAgent(ToolAgent[_Msg, ToolAgentConfig]):
    """Minimal concrete ToolAgent over plain-dict messages."""

    config_cls = ToolAgentConfig

    def __init__(self, turns: list[AgentStep], **config: Any) -> None:
        super().__init__(ToolAgentConfig(model="test-model", **config))
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


def test_claude_defaults_to_configurable_webp_screenshots() -> None:
    assert ToolAgentConfig().screenshot_encoding == PngScreenshotEncoding()
    assert ClaudeConfig().screenshot_encoding == WebPScreenshotEncoding()
    assert ClaudeCLIConfig().screenshot_encoding == WebPScreenshotEncoding()

    configured = ClaudeConfig.model_validate(
        {"screenshot_encoding": {"mime_type": "image/webp", "quality": 42}},
    )
    assert configured.screenshot_encoding == WebPScreenshotEncoding(quality=42)

    with pytest.raises(ValueError):
        ClaudeConfig.model_validate(
            {"screenshot_encoding": {"mime_type": "image/webp", "quality": 101}},
        )


def test_only_claude_provider_has_a_default_tool_timeout() -> None:
    config = ClaudeConfig(timeout_seconds=600)

    assert config.timeout_seconds == 600
    assert config.tool_timeout_seconds == 120
    assert ClaudeConfig(tool_timeout_seconds=None).tool_timeout_seconds is None
    assert ToolAgentConfig().tool_timeout_seconds is None


async def test_agent_passes_screenshot_encoding_to_rfb_tools() -> None:
    class ScreenTool(RFBTool):
        name = "screen"

        @classmethod
        def default_spec(cls, model: str) -> AgentToolSpec:
            del model
            return AgentToolSpec(api_type="screen", api_name="screen")

        async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
            del arguments
            return MCPToolResult(content=[])

        def to_params(self) -> dict[str, str]:
            return {"name": self.name}

    class ScreenAgent(DictAgent):
        tool_catalog = (ScreenTool,)

    encoding = {"mime_type": "image/webp", "quality": 42}
    agent = ScreenAgent([AgentStep(content="done", done=True)], screenshot_encoding=encoding)
    rfb = object.__new__(RFBClient)

    tools, _ = await agent._build_tools({"screen": rfb})

    tool = tools["screen"]
    assert isinstance(tool, ScreenTool)
    assert tool.screenshot_encoding == WebPScreenshotEncoding(quality=42)


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


async def test_mcp_client_propagates_active_trace_context() -> None:
    calls: list[dict[str, Any]] = []

    class Client:
        async def call_tool_mcp(self, **kwargs: Any) -> mcp_types.CallToolResult:
            calls.append(kwargs)
            return mcp_types.CallToolResult(content=[])

    capability = Capability.mcp(url="https://tools.example/mcp")
    client = MCPClient(capability, cast("Any", Client()), AsyncExitStack())

    with set_trace_context("child-trace"):
        await client.call_tool("verify", {})

    assert calls == [
        {
            "name": "verify",
            "arguments": {},
            "meta": {"hud/trace-id": "child-trace"},
        }
    ]


def test_mcp_server_reads_propagated_trace_context(monkeypatch: pytest.MonkeyPatch) -> None:
    request_context = SimpleNamespace(
        meta=SimpleNamespace(model_extra={"hud/trace-id": "child-trace"})
    )
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_context",
        lambda: SimpleNamespace(request_context=request_context),
    )

    assert get_mcp_trace_id() == "child-trace"


def test_mcp_server_reads_propagated_trace_header(monkeypatch: pytest.MonkeyPatch) -> None:
    request_context = SimpleNamespace(meta=SimpleNamespace(model_extra={}))
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_context",
        lambda: SimpleNamespace(request_context=request_context),
    )
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_http_headers",
        lambda: {
            "trace-id": "child-trace",
        },
    )

    assert get_mcp_trace_id() == "child-trace"


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

        async def list_tools(self) -> list[mcp_types.Tool]:
            return self.tools

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

        async def list_tools(self) -> list[mcp_types.Tool]:
            return [self.tool]

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

        async def list_tools(self) -> list[mcp_types.Tool]:
            return [self.tool]

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


async def test_dispatch_marks_exhausted_ssh_reconnect_as_infrastructure_error() -> None:
    agent = DictAgent([])
    tool = SimpleNamespace(
        execute=AsyncMock(side_effect=SSHConnectionError("SSH reconnect failed after 3 attempts"))
    )
    state: RunState[_Msg] = RunState(tools={"bash": cast("Any", tool)})

    result = await agent._dispatch_call(MCPToolCall(name="bash"), state)

    assert result.isError is True
    assert isinstance(result, SSHInfrastructureErrorResult)


async def test_dispatch_reraises_tool_timeout_before_deadline() -> None:
    ssh = SimpleNamespace(run=AsyncMock(side_effect=TimeoutError("remote timed out")))
    tool = ClaudeBashTool(
        spec=ClaudeBashTool.default_spec("claude"),
        client=cast("Any", ssh),
    )
    agent = DictAgent([], tool_timeout_seconds=120)
    state: RunState[_Msg] = RunState(tools={"bash": tool})

    with pytest.raises(TimeoutError, match="remote timed out"):
        await agent._dispatch_call(
            MCPToolCall(name="bash", arguments={"command": "sleep forever"}),
            state,
        )


async def test_claude_bash_timeout_terminates_process_and_continues_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()

    async def wait(*, check: bool, timeout: None) -> None:  # noqa: ASYNC109
        del check, timeout
        started.set()
        await asyncio.Event().wait()

    process = SimpleNamespace(
        wait=AsyncMock(side_effect=wait),
        terminate=Mock(),
        close=Mock(side_effect=AssertionError("graceful termination should succeed")),
        wait_closed=AsyncMock(),
    )
    connection = SimpleNamespace(
        is_closed=Mock(return_value=False),
        create_process=AsyncMock(return_value=process),
    )
    ssh = SSHClient(
        Capability.ssh(url="ssh://workspace", host_pubkey="key"),
        cast("Any", connection),
    )
    tool = ClaudeBashTool(spec=ClaudeBashTool.default_spec("claude"), client=ssh)
    agent = ClaudeAgent(
        ClaudeConfig(
            model="claude-test",
            model_client=cast("Any", object()),
            tool_timeout_seconds=0.01,
        )
    )
    responses = AsyncMock(
        side_effect=[
            AgentStep(
                content="",
                done=False,
                tool_calls=[
                    MCPToolCall(id="tool-1", name="bash", arguments={"command": "sleep forever"})
                ],
            ),
            AgentStep(content="recovered", done=True),
        ]
    )
    monkeypatch.setattr(agent, "get_response", responses)
    state = RunState(messages=[], tools={"bash": tool})
    run = cast("Run", _FakeRun())

    agent.config.max_steps = 3
    await agent._loop(run, state)

    assert started.is_set()
    process.terminate.assert_called_once_with()
    process.wait_closed.assert_awaited_once_with()
    assert responses.await_count == 2
    assert run.trace.status == "completed"
    assert run.trace.content == "recovered"
    tool_result = cast("list[Any]", state.messages[0]["content"])[0]
    error_block = cast("list[Any]", tool_result["content"])[0]
    assert error_block["text"] == (
        "Error: bash timed out after 0.01s; retry with a shorter command"
    )


async def test_composite_editor_uses_one_tool_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deadline = Mock(wraps=asyncio.timeout)
    monkeypatch.setattr(asyncio, "timeout", deadline)
    ssh = SimpleNamespace(
        read_text=AsyncMock(return_value="old"),
        write_text=AsyncMock(),
    )
    tool = ClaudeTextEditorTool(
        spec=ClaudeTextEditorTool.default_spec("claude"),
        client=cast("Any", ssh),
    )
    agent = DictAgent([], tool_timeout_seconds=120)
    state: RunState[_Msg] = RunState(tools={tool.provider_name: tool})

    result = await agent._dispatch_call(
        MCPToolCall(
            name=tool.provider_name,
            arguments={
                "command": "str_replace",
                "path": "/file.txt",
                "old_str": "old",
                "new_str": "new",
            },
        ),
        state,
    )

    assert result.isError is False
    assert result_text(result) == "wrote 3 bytes to /file.txt"
    deadline.assert_called_once_with(120)
    ssh.read_text.assert_awaited_once_with("/file.txt")
    ssh.write_text.assert_awaited_once_with("/file.txt", "new")


async def test_loop_finishes_on_done_response() -> None:
    agent = DictAgent([AgentStep(content="final answer", done=True)])
    run = cast("Run", _FakeRun())

    agent.config.max_steps = 3
    await agent._loop(run, RunState())

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


class _DegenerateDictAgent(DictAgent):
    """DictAgent whose scripted turns may be ``DegenerateTurnError`` instances."""

    def __init__(self, turns: list[AgentStep | DegenerateTurnError], **config: Any) -> None:
        super().__init__(cast("list[AgentStep]", turns), **config)

    async def get_response(
        self, state: RunState[_Msg], *, system_prompt: Any = None, citations_enabled: bool = False
    ) -> AgentStep:
        turn = await super().get_response(
            state, system_prompt=system_prompt, citations_enabled=citations_enabled
        )
        if isinstance(turn, DegenerateTurnError):
            raise turn
        return turn


async def test_loop_discards_degenerate_turn_and_resamples() -> None:
    agent = _DegenerateDictAgent(
        [
            DegenerateTurnError("empty shell_call"),
            AgentStep(content="recovered", done=True),
        ]
    )
    run = cast("Run", _FakeRun())

    agent.config.max_steps = 3
    await agent._loop(run, RunState())

    # The degenerate turn is discarded (consuming a step) and never recorded.
    assert run.trace.status == "completed"
    assert run.trace.content == "recovered"
    assert [step.source for step in run.trace.steps] == ["agent"]


async def test_loop_fails_when_degenerate_turn_exhausts_steps() -> None:
    agent = _DegenerateDictAgent([DegenerateTurnError("empty shell_call")] * 2)
    run = cast("Run", _FakeRun())

    agent.config.max_steps = 2
    await agent._loop(run, RunState())

    assert run.trace.status == "error"
    assert run.trace.error == "empty shell_call"


async def test_loop_dispatches_tool_calls_then_finishes() -> None:
    agent = DictAgent(
        [
            AgentStep(content="", done=False, tool_calls=[MCPToolCall(name="ghost")]),
            AgentStep(content="done now", done=True),
        ]
    )
    run = cast("Run", _FakeRun())

    agent.config.max_steps = 3
    await agent._loop(run, RunState())

    assert run.trace.content == "done now"
    assert [step.source for step in run.trace.steps] == ["agent", "tool", "agent"]
    # the (unknown) tool call produced an observed tool step in the trajectory
    tool_step = run.trace.steps[1]
    assert isinstance(tool_step, ToolStep)
    assert tool_step.call is not None
    assert tool_step.call.name == "ghost"
    assert tool_step.result is not None
    assert tool_step.result.isError is True  # unknown tool → error result


async def test_loop_resets_infrastructure_error_count_after_other_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = [
        AgentStep(content="", done=False, tool_calls=[MCPToolCall(name="bash")]) for _ in range(5)
    ]
    agent = DictAgent(turns)
    infrastructure_error = SSHInfrastructureErrorResult(content=[], isError=True)
    ordinary_error = MCPToolResult(content=[], isError=True)
    dispatch = AsyncMock(
        side_effect=[
            infrastructure_error,
            ordinary_error,
            infrastructure_error,
            infrastructure_error,
            infrastructure_error,
        ]
    )
    monkeypatch.setattr(agent, "_dispatch_call", dispatch)
    run = cast("Run", _FakeRun())

    agent.config.max_steps = 10
    await agent._loop(run, RunState())

    assert dispatch.await_count == 5
    assert run.trace.status == "error"
    assert run.trace.stop_reason is None
    assert run.trace.error == ("SSH tool failure limit reached after 3 consecutive errors")


async def test_loop_max_steps_is_normal_termination() -> None:
    # Always returns a tool call → never "done" → hits max_steps. Exhausting the
    # configured budget is a stop reason, not an agent error (the platform must
    # not paint the rollout or its last tool call as failed).
    never_done = [
        AgentStep(content="", done=False, tool_calls=[MCPToolCall(name="ghost")]) for _ in range(5)
    ]
    agent = DictAgent(never_done)
    run = cast("Run", _FakeRun())

    agent.config.max_steps = 2
    await agent._loop(run, RunState())

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

        agent.config.max_steps = 3
        await agent._loop(run, RunState())

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

    agent.config.max_steps = 3
    await agent._loop(run, RunState())

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

    agent.config.max_steps = 3
    await agent._loop(run, RunState())

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

    agent.config.max_steps = 3
    await agent._loop(run, RunState())

    assert run.trace.stop_reason == "length"
    assert run.trace.is_truncated is True
    assert all(not isinstance(step, ToolStep) for step in run.trace.steps)


@pytest.mark.parametrize("auto_respond", [False, True])
@pytest.mark.parametrize("reason", [None, "malformed_tool_call"])
async def test_provider_error_stops_without_auto_response(
    auto_respond: bool, reason: Literal["malformed_tool_call"] | None
) -> None:
    agent = DictAgent(
        [AgentStep(error="provider failure", stop_reason=reason)],
        auto_respond=auto_respond,
        max_steps=3,
    )
    run = cast("Run", _FakeRun())
    await agent._loop(run, RunState())
    assert run.trace.status == "error"
    assert run.trace.stop_reason == reason
    assert len(run.trace.steps) == 1
    assert run.trace.steps[0].error == "provider failure"


# ─── tool result size limit ───────────────────────────────────────────


class _ProbeTool(AgentTool[CapabilityClient]):
    """Tool that returns one fixed result."""

    name = "probe"
    client_type = CapabilityClient

    def __init__(self, result: MCPToolResult) -> None:
        super().__init__(
            spec=AgentToolSpec(api_type="probe", api_name="probe"), client=cast("Any", None)
        )
        self._result = result

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        del arguments
        return self._result

    def to_params(self) -> dict[str, str]:
        return {"name": self.name}


async def _loop_one_call(result: MCPToolResult, **config: Any) -> MCPToolResult:
    """Run one scripted tool call through the loop; return the recorded result."""
    agent = DictAgent(
        [
            AgentStep(content="", done=False, tool_calls=[MCPToolCall(name="probe")]),
            AgentStep(content="done", done=True),
        ],
        **config,
    )
    run = cast("Run", _FakeRun())

    await agent._loop(run, RunState(tools={"probe": _ProbeTool(result)}))

    tool_step = run.trace.steps[1]
    assert isinstance(tool_step, ToolStep)
    assert tool_step.result is not None
    return tool_step.result


def _texts(result: MCPToolResult) -> list[str]:
    texts: list[str] = []
    for block in result.content:
        if isinstance(block, mcp_types.TextContent):
            texts.append(block.text)
        elif isinstance(block, mcp_types.EmbeddedResource) and isinstance(
            block.resource, mcp_types.TextResourceContents
        ):
            texts.append(block.resource.text)
    return texts


def test_tool_result_limit_defaults_and_rejects_values_too_small_for_the_notice() -> None:
    assert ToolAgentConfig().max_tool_result_chars == 30_000
    assert ClaudeConfig().max_tool_result_chars == 30_000
    assert ToolAgentConfig(max_tool_result_chars=5_000).max_tool_result_chars == 5_000
    with pytest.raises(ValidationError):
        ToolAgentConfig(max_tool_result_chars=999)


async def test_loop_passes_tool_results_within_the_limit_unchanged() -> None:
    result = MCPToolResult(content=[mcp_types.TextContent(type="text", text="x" * 900)])

    recorded = await _loop_one_call(result, max_tool_result_chars=1_000)

    assert recorded == result


async def test_loop_keeps_head_and_tail_of_an_oversized_tool_result() -> None:
    text = "HEAD-" + "x" * 50_000 + "-TAIL"
    result = MCPToolResult(content=[mcp_types.TextContent(type="text", text=text)], isError=True)

    recorded = await _loop_one_call(result, max_tool_result_chars=2_000)

    (bounded,) = _texts(recorded)
    assert len(bounded) <= 2_000
    assert bounded.startswith("HEAD-")
    assert bounded.endswith("-TAIL")
    assert "output truncated" in bounded
    assert "grep" in bounded
    assert recorded.isError is True


async def test_loop_bounds_text_across_blocks_and_keeps_images() -> None:
    image = mcp_types.ImageContent(type="image", data="aW1n", mimeType="image/png")
    result = MCPToolResult(
        content=[
            mcp_types.TextContent(type="text", text="A" * 3_000),
            image,
            mcp_types.TextContent(type="text", text="B" * 3_000),
            mcp_types.EmbeddedResource(
                type="resource",
                resource=mcp_types.TextResourceContents(uri="file:///c.txt", text="C" * 3_000),
            ),
        ]
    )

    recorded = await _loop_one_call(result, max_tool_result_chars=2_000)

    texts = _texts(recorded)
    assert sum(len(text) for text in texts) <= 2_000
    assert recorded.content[1] == image
    assert texts[0].startswith("A")
    assert "output truncated" in texts[0]
    assert texts[-1] == "C" * len(texts[-1])
    assert all("B" not in text for text in texts)


async def test_loop_turns_an_oversized_structured_only_result_into_bounded_text() -> None:
    result = MCPToolResult(content=[], structuredContent={"rows": ["r" * 10] * 1_000})

    recorded = await _loop_one_call(result, max_tool_result_chars=2_000)

    assert recorded.structuredContent is None
    (bounded,) = _texts(recorded)
    assert len(bounded) <= 2_000
    assert bounded.startswith('{"rows": ["rrrrrrrrrr"')
    assert bounded.endswith('"rrrrrrrrrr"]}')


@pytest.mark.parametrize("text", ["summary", "x" * 50_000])
async def test_recorded_tool_step_holds_only_what_the_model_receives(text: str) -> None:
    # Providers send structuredContent only when a result has no content, so a
    # payload beside content must not reach the trace either.
    result = MCPToolResult(
        content=[mcp_types.TextContent(type="text", text=text)],
        structuredContent={"rows": ["r" * 100] * 50_000},
    )

    recorded = await _loop_one_call(result, max_tool_result_chars=2_000)

    assert recorded.structuredContent is None
    (sent,) = _texts(recorded)
    assert sent == text if len(text) <= 2_000 else len(sent) <= 2_000
    assert len(recorded.model_dump_json()) < 3_000


async def _loop_openai_shell(
    outputs: list[tuple[str, str, int]], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Run one OpenAI shell call through the loop; return the provider output item."""
    completed = [
        SimpleNamespace(stdout=stdout, stderr=stderr, returncode=code)
        for stdout, stderr, code in outputs
    ]
    tool = OpenAIShellTool(
        spec=OpenAIShellTool.default_spec("gpt-test"),
        client=cast("Any", SimpleNamespace(run=AsyncMock(side_effect=completed))),
    )
    agent = OpenAIAgent(OpenAIConfig(model="gpt-test", model_client=cast("Any", object())))
    call = MCPToolCall(
        id="call-1",
        name="shell",
        arguments={"commands": [f"step-{n}" for n in range(len(outputs))]},
    )
    responses = AsyncMock(
        side_effect=[
            AgentStep(content="", done=False, tool_calls=[call]),
            AgentStep(content="done", done=True),
        ]
    )
    state: RunState[Any] = RunState(messages=[], tools={"shell": tool})
    monkeypatch.setattr(agent, "get_response", responses)

    await agent._loop(cast("Run", _FakeRun()), state)

    (item,) = state.messages
    return cast("dict[str, Any]", item)


async def test_openai_shell_result_under_the_limit_reaches_the_model_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs: list[tuple[str, str, int]] = [("b" * 12_000, "", 0), ("", "t" * 8_000, 2)]

    item = await _loop_openai_shell(outputs, monkeypatch)

    assert item["type"] == "shell_call_output"
    assert item["output"] == [
        {"stdout": stdout, "stderr": stderr, "outcome": {"type": "exit", "exit_code": code}}
        for stdout, stderr, code in outputs
    ]
    assert item["max_output_length"] == 10 * 1024 * 1024


async def test_openai_shell_result_over_the_limit_keeps_per_command_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs: list[tuple[str, str, int]] = [
        ("ok\n", "", 0),
        ("START-" + "o" * 60_000 + "-END", "", 1),
        ("", "E-" + "e" * 40_000 + "-LAST", 3),
    ]

    item = await _loop_openai_shell(outputs, monkeypatch)

    first, second, third = item["output"]
    assert [entry["outcome"]["exit_code"] for entry in item["output"]] == [0, 1, 3]
    assert first == {"stdout": "ok\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}
    assert sum(len(entry["stdout"]) + len(entry["stderr"]) for entry in item["output"]) <= 30_000
    assert second["stdout"].startswith("START-")
    assert second["stdout"].endswith("-END")
    assert "output truncated" in second["stdout"]
    assert second["stderr"] == ""
    assert third["stdout"] == ""
    assert third["stderr"].startswith("E-")
    assert third["stderr"].endswith("-LAST")
    assert item["max_output_length"] == 10 * 1024 * 1024


async def test_bounded_infrastructure_errors_still_trip_the_ssh_failure_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = [
        AgentStep(content="", done=False, tool_calls=[MCPToolCall(name="bash")]) for _ in range(3)
    ]
    agent = DictAgent(turns, max_tool_result_chars=1_000)
    oversized = SSHInfrastructureErrorResult(
        content=[mcp_types.TextContent(type="text", text="lost " * 10_000)], isError=True
    )
    monkeypatch.setattr(agent, "_dispatch_call", AsyncMock(return_value=oversized))
    run = cast("Run", _FakeRun())

    await agent._loop(run, RunState())

    assert run.trace.error == "SSH tool failure limit reached after 3 consecutive errors"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX workspace semantics")
async def test_claude_agent_bounds_whole_file_reads_from_a_large_case_room(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A multi-MB attachment read whole (cat of a binary, a full CSV view) once
    # became a single ~918K-token tool result and overflowed the model's context.
    root = tmp_path / "case-room"
    root.mkdir()
    rng = random.Random(677)
    (root / "Annual_Report.pdf").write_bytes(b"%PDF-1.7\n" + rng.randbytes(1_500_000))
    rows = [
        f"{n},chat,{n % 10}.5,{n % 5 + 1},customer asked about a refund for order {n}"
        for n in range(30_000)
    ]
    (root / "Tickets.csv").write_text(
        "\n".join(["ticket_id,channel,minutes,csat,summary", *rows]) + "\n"
    )
    workspace = Workspace(root)
    await workspace.start()
    try:
        ssh = await SSHClient.connect(workspace.capability())
        try:
            bash = ClaudeBashTool(spec=ClaudeBashTool.default_spec("claude-test"), client=ssh)
            editor = ClaudeTextEditorTool(
                spec=ClaudeTextEditorTool.default_spec("claude-test"), client=ssh
            )
            agent = ClaudeAgent(
                ClaudeConfig(model="claude-test", model_client=cast("Any", object()))
            )
            calls = [
                MCPToolCall(
                    id="cat-pdf", name="bash", arguments={"command": "cat Annual_Report.pdf"}
                ),
                MCPToolCall(
                    id="view-csv",
                    name=editor.provider_name,
                    arguments={"command": "view", "path": "Tickets.csv"},
                ),
                MCPToolCall(
                    id="view-whole-csv",
                    name=editor.provider_name,
                    arguments={"command": "view", "path": "Tickets.csv", "view_range": [1, -1]},
                ),
            ]
            responses = AsyncMock(
                side_effect=[
                    AgentStep(content="", done=False, tool_calls=calls),
                    AgentStep(content="done", done=True),
                ]
            )
            monkeypatch.setattr(agent, "get_response", responses)
            state = RunState(messages=[], tools={"bash": bash, editor.provider_name: editor})
            run = cast("Run", _FakeRun())

            await agent._loop(run, state)
        finally:
            await ssh.close()
    finally:
        await workspace.stop()

    assert run.trace.status == "completed"
    sent: dict[str, str] = {}
    for message in state.messages:
        for block in cast("list[Any]", message["content"]):
            if block["type"] == "tool_result":
                sent[block["tool_use_id"]] = "".join(part["text"] for part in block["content"])
    assert set(sent) == {"cat-pdf", "view-csv", "view-whole-csv"}
    recorded = {
        step.call.id: step.result
        for step in run.trace.steps
        if isinstance(step, ToolStep) and step.call is not None and step.result is not None
    }
    assert {call_id: result_text(result) for call_id, result in recorded.items()} == sent
    assert all(result.structuredContent is None for result in recorded.values())
    assert all(len(text) <= 30_000 for text in sent.values())
    assert sent["cat-pdf"].startswith("$ cat Annual_Report.pdf\n%PDF-1.7")
    assert sent["cat-pdf"].endswith("(exit 0)")
    assert sent["view-csv"].startswith("     1\tticket_id,channel")
    assert sent["view-csv"].endswith(
        "(Showing lines 1-2000 of 30001. Use view_range [2001, 4000] to continue.)"
    )
    assert sent["view-whole-csv"].endswith("(End of file - total 30001 lines)")
    assert all("output truncated" in text for text in sent.values())
