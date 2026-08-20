"""``ToolAgent`` plumbing: catalog→clients, message formatting, dispatch + loop.

The provider-specific bits are abstract; this drives a tiny concrete subclass with a
scripted ``get_response`` so the loop, dispatch, and message formatting run offline.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import AsyncMock, Mock

import fastmcp
import mcp.types as mcp_types
import pytest
from fastmcp.client.transports import SSETransport, StreamableHttpTransport

from hud.agents.claude.agent import ClaudeAgent
from hud.agents.claude.tools.coding import ClaudeBashTool, ClaudeTextEditorTool
from hud.agents.openai.tools.coding import OpenAIShellTool
from hud.agents.openai.tools.computer import OpenAIComputerTool
from hud.agents.openai.tools.mcp_proxy import OpenAIMCPProxyTool
from hud.agents.tool_agent import RunState, ToolAgent
from hud.agents.tools.base import AgentToolSpec, result_text
from hud.agents.tools.rfb import RFBTool
from hud.agents.tools.ssh import SSHInfrastructureErrorResult
from hud.agents.types import (
    AgentStep,
    ClaudeCLIConfig,
    ClaudeConfig,
    ToolAgentConfig,
    ToolStep,
)
from hud.capabilities import Capability, CapabilityClient, MCPClient, RFBClient
from hud.capabilities.rfb import PngScreenshotEncoding, WebPScreenshotEncoding
from hud.capabilities.ssh import SSHConnectionError
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


class DictAgent(ToolAgent[_Msg, ToolAgentConfig]):
    """Minimal concrete ToolAgent over plain-dict messages."""

    def __init__(self, turns: list[AgentStep], **config: Any) -> None:
        self.config = ToolAgentConfig(model="test-model", **config)
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
    assert AgentConfig().tool_timeout_seconds is None
    assert ClaudeSDKConfig().tool_timeout_seconds is None


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
        tool_catalog = (OpenAIMCPProxyTool,)

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
        tool_catalog = (OpenAIComputerTool,)

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
        tool_catalog = (OpenAIMCPProxyTool, OpenAIShellTool)

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


def test_initial_messages_formats_each_turn() -> None:
    agent = DictAgent([])
    turn = mcp_types.PromptMessage(
        role="user", content=mcp_types.TextContent(type="text", text="a")
    )
    assert agent._initial_messages([turn]) == [{"role": "user", "content": "a"}]


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

    await agent._loop(run, state, max_steps=3)

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

    await agent._loop(run, RunState())

    assert run.trace.status is None
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

    with pytest.raises(RuntimeError, match="SSH tool failure limit reached"):
        await agent._loop(run, RunState())

    assert dispatch.await_count == 5
    assert run.trace.stop_reason is None


async def test_loop_max_steps_is_normal_termination() -> None:
    # Always returns a tool call → never "done" → hits max_steps. Exhausting the
    # configured budget is a stop reason, not an agent error (the platform must
    # not paint the rollout or its last tool call as failed).
    never_done = [
        AgentStep(content="", done=False, tool_calls=[MCPToolCall(name="ghost")]) for _ in range(5)
    ]
    agent = DictAgent(never_done, max_steps=2)
    run = cast("Run", _FakeRun())

    await agent._loop(run, RunState())

    assert run.trace.is_error is False
    assert run.trace.status is None
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

        await agent._loop(run, RunState())

        assert run.trace.status is None
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

    await agent._loop(run, RunState())

    assert run.trace.status is None
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

    await agent._loop(run, RunState())

    assert run.trace.stop_reason == "length"
    assert run.trace.is_truncated is True
    assert all(not isinstance(step, ToolStep) for step in run.trace.steps)
