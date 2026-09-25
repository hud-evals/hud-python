"""ClaudeCLIAgent remote-command construction over the workspace SSH.

The agent runs the ``claude`` CLI on the remote workspace. These cover how the
command is assembled per login shell — especially the Windows path, where the
command must ride a batch file invoked via ``cmd /c``. Bare ``.hud_run.bat`` is
rejected by the remote shell (and silently fails under PowerShell), so the
``cmd /c`` prefix is a regression guard for local Windows setups.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, Mock

import fastmcp
import pytest
from mcp.types import ImageContent, TextContent

from hud.agents import create_agent
from hud.agents.claude.sdk import computer_mcp
from hud.agents.claude.sdk.agent import ClaudeCLIAgent
from hud.agents.tests.cli_fakes import FakeClient as _FakeClient
from hud.agents.tests.cli_fakes import FakeProcess as _FakeStreamProcess
from hud.agents.tests.cli_fakes import fake_run as _fake_run
from hud.agents.types import AgentStep, ClaudeCLIConfig, ToolStep
from hud.capabilities import Capability, Connection, SSHClient
from hud.capabilities.rfb import WebPScreenshotEncoding
from hud.settings import settings
from hud.telemetry.context import set_trace_context
from hud.types import MCPToolResult


@pytest.fixture(autouse=True)
def _clear_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    monkeypatch.setattr(
        "hud.agents.claude.sdk.agent.resolve_executable",
        AsyncMock(return_value="claude"),
    )


async def test_command_follows_explicit_gateway_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", "hud-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "anthropic-key")

    gateway = await _sent_command(ClaudeCLIAgent(ClaudeCLIConfig(gateway=True)))
    provider = await _sent_command(ClaudeCLIAgent(ClaudeCLIConfig(gateway=False)))

    assert f"ANTHROPIC_BASE_URL={settings.hud_gateway_url}" in gateway
    assert "ANTHROPIC_API_KEY=hud-key" in gateway
    assert "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1" in gateway
    assert "DISABLE_AUTO_COMPACT=1" in gateway
    assert "ANTHROPIC_API_KEY=anthropic-key" in provider
    assert "ANTHROPIC_BASE_URL" not in provider
    assert "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS" not in provider
    assert "DISABLE_AUTO_COMPACT" not in provider
    assert "ANTHROPIC_MODEL=claude-sonnet-5" in provider


async def test_default_routing_uses_gateway_only_without_a_provider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "api_key", "hud-key")

    without_provider_key = await _sent_command(ClaudeCLIAgent(ClaudeCLIConfig()))
    monkeypatch.setattr(settings, "anthropic_api_key", "anthropic-key")
    with_provider_key = await _sent_command(ClaudeCLIAgent(ClaudeCLIConfig()))

    assert f"ANTHROPIC_BASE_URL={settings.hud_gateway_url}" in without_provider_key
    assert "ANTHROPIC_API_KEY=anthropic-key" in with_provider_key
    assert "ANTHROPIC_BASE_URL" not in with_provider_key


async def test_create_agent_routes_cli_agent_through_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "api_key", "hud-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "anthropic-key")

    agent = create_agent("claude_cli")

    assert isinstance(agent, ClaudeCLIAgent)
    assert agent.config.model == ClaudeCLIConfig().model
    command = await _sent_command(agent)
    assert f"ANTHROPIC_BASE_URL={settings.hud_gateway_url}" in command
    assert "ANTHROPIC_API_KEY=hud-key" in command


async def test_command_uses_process_bound_connection_without_its_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "api_key", "hud-key")
    connection = Connection(
        name="inference",
        capability="ssh",
        url="https://inference.hud.so",
        headers={"Authorization": "Bearer scoped-runtime-token"},
    )

    with set_trace_context("trace-123"):
        command = await _sent_command(ClaudeCLIAgent(), connection=connection)

    assert f"ANTHROPIC_BASE_URL={connection.client_url}" in command
    assert "ANTHROPIC_AUTH_TOKEN=hud-process-bound" in command
    assert "ANTHROPIC_API_KEY" not in command
    assert "scoped-runtime-token" not in command
    assert "hud-key" not in command
    assert "Trace-Id" not in command
    assert "exec env" in command
    for name in (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ):
        assert f"{name}=claude-sonnet-5" in command


async def test_windows_command_encodes_environment_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "api_key", "hud&key's")
    config = ClaudeCLIConfig(
        gateway=True,
        max_steps=3,
        system_prompt="don't $expand",
    )
    command = await _sent_command(ClaudeCLIAgent(config), "powershell")

    encoded = command.rsplit(" ", 1)[1]
    script = base64.b64decode(encoded).decode("utf-16-le")
    assert "$env:ANTHROPIC_API_KEY='hud&key''s'" in script
    assert "$env:CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS='1'" in script
    assert "$env:DISABLE_AUTO_COMPACT='1'" in script
    assert "'--system-prompt' 'don''t $expand'" in script
    assert "Get-Content -Raw -Encoding UTF8 '.hud_input.jsonl' | & 'claude'" in script
    assert "'--input-format=stream-json'" in script
    assert "python" not in script


class _FakeCompletedProcess:
    async def wait(self, *, check: bool, **kwargs: Any) -> Any:
        del check
        assert kwargs == {"timeout": None}
        return SimpleNamespace(stdout=b"", stderr=b"", exit_status=0, returncode=0)

    def terminate(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


class _FakeConn:
    def __init__(self, sink: dict[str, bytes], process: _FakeStreamProcess) -> None:
        self._sink = sink
        self._process = process
        self.ran: list[str] = []
        self.write_commands: list[str] = []
        self.written: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def is_closed(self) -> bool:
        return False

    async def create_process(self, cmd: str, **kwargs: Any) -> Any:
        input_value = kwargs.get("input")
        if cmd.startswith(("rm -f -- ", "cmd /c del /f /q ")):
            paths = [path for path in self._sink if path in cmd]
            for path in paths:
                self._sink.pop(path)
            self.deleted.extend(paths)
            return _FakeCompletedProcess()
        if input_value is not None or cmd.startswith("powershell "):
            self.write_commands.append(cmd)
            script = cmd
            if match := re.search(r"-EncodedCommand (\S+)", cmd):
                script = base64.b64decode(match.group(1)).decode("utf-16-le")
            name = next(
                path
                for path in (".hud_input.jsonl", ".hud_run.bat", ".hud_mcp_config.json")
                if path in script
            )
            if input_value is not None:
                self._sink[name] = str(input_value).encode()
            elif match := re.search(r"FromBase64String\('([^']+)'\)", script):
                self._sink[name] += base64.b64decode(match.group(1))
            else:
                self._sink[name] = b""
            self.written[name] = self._sink[name]
            return _FakeCompletedProcess()
        assert kwargs["encoding"] is None
        self.env = kwargs.get("env")
        self.ran.append(cmd)
        return self._process


async def run_claude(
    config: ClaudeCLIConfig,
    run: Any,
    *,
    ssh: SSHClient,
    prompt: str,
    bindings: list[Capability] | None = None,
    routed: dict[str, Capability] | None = None,
) -> None:
    run.client = _FakeClient(ssh, bindings, routed)
    run.prompt_text = prompt
    await ClaudeCLIAgent(config)(run)


_STREAM_JSON = (
    '{"type":"assistant","message":{"id":"msg-1","type":"message",'
    '"role":"assistant","model":"claude-test","content":[{"type":"text",'
    '"text":"editing"},{"type":"tool_use","id":"tool-1","name":"Write","input":{}}],'
    '"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":11,'
    '"output_tokens":7,"cache_read_input_tokens":3}}}\n'
    '{"type":"user","message":{"content":[{"type":"tool_result",'
    '"tool_use_id":"tool-1","content":[{"type":"text","text":"wrote a.txt"},'
    '{"type":"image","source":{"type":"base64","media_type":"image/png",'
    '"data":"aW1hZ2U="}}],"is_error":false}]}}\n'
    '{"type":"assistant","message":{"id":"msg-2","type":"message",'
    '"role":"assistant","model":"claude-test","content":[{"type":"text",'
    '"text":"done"}],"stop_reason":"end_turn","stop_sequence":null,'
    '"usage":{"input_tokens":11,"output_tokens":7,"cache_read_input_tokens":3}}}\n'
    '{"type":"result","is_error":false,"result":"done","session_id":"s",'
    '"duration_ms":5,"num_turns":2,"total_cost_usd":0.01}\n'
)


def _ssh_with_conn(shell: str, conn: _FakeConn) -> SSHClient:
    capability = Capability(
        name="shell",
        protocol="ssh/2",
        url="ssh://localhost:22",
        params={"shell": shell, "process_connections": True},
    )
    return SSHClient(capability, cast("Any", conn))


async def _sent_command(
    agent: ClaudeCLIAgent, shell: str = "bash", connection: Connection | None = None
) -> str:
    """The CLI command the agent sends over SSH (unwrapped from the batch file on Windows)."""
    conn = _FakeConn({}, _FakeStreamProcess(_STREAM_JSON))
    connections = {"inference": connection} if connection is not None else None
    await agent(_fake_run(_FakeClient(_ssh_with_conn(shell, conn)), "x", connections))
    if shell in ("cmd", "powershell"):
        return conn.written[".hud_run.bat"].decode().split("\r\n")[1]
    (command,) = conn.ran
    return command


async def test_exec_on_windows_writes_batch_and_execs_via_cmd() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, _FakeStreamProcess(_STREAM_JSON))
    ssh = _ssh_with_conn("cmd", conn)

    run = _fake_run()
    await run_claude(ClaudeCLIConfig(), run, ssh=ssh, prompt="build it")

    assert conn.ran == ["cmd /c .hud_run.bat"]
    assert all(command.startswith("powershell ") for command in conn.write_commands)
    assert conn.written[".hud_run.bat"].startswith(b"@echo off\r\n")
    assert conn.written[".hud_input.jsonl"].endswith(b"\n")
    assert json.loads(conn.written[".hud_input.jsonl"]) == {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "build it"}],
        },
    }
    assert sink == {}
    assert set(conn.deleted) == {".hud_input.jsonl", ".hud_run.bat"}
    assert run.trace.status is None
    assert run.trace.content == "done"
    assert "messages" not in run.trace.extra


async def test_exec_on_bash_runs_inline_without_batch() -> None:
    sink: dict[str, bytes] = {}
    process = _FakeStreamProcess(_STREAM_JSON)
    conn = _FakeConn(sink, process)
    ssh = _ssh_with_conn("bash", conn)

    run = _fake_run()
    await run_claude(ClaudeCLIConfig(), run, ssh=ssh, prompt="build it")

    assert sink == {}
    assert conn.write_commands == []
    assert conn.deleted == []
    assert len(conn.ran) == 1
    assert "claude" in conn.ran[0]
    assert "--input-format=stream-json" in conn.ran[0]
    assert "build it" not in conn.ran[0]
    assert process.stdin.data.endswith(b"\n")
    assert json.loads(process.stdin.data) == {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "build it"}],
        },
    }
    assert process.stdin.eof is True
    assert run.trace.status is None
    assert run.trace.content == "done"
    assert "messages" not in run.trace.extra


async def test_exec_removes_mcp_config_after_run() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, _FakeStreamProcess(_STREAM_JSON))
    await run_claude(
        ClaudeCLIConfig(),
        _fake_run(),
        ssh=_ssh_with_conn("bash", conn),
        bindings=[
            Capability.mcp(name="database", url="http://db/mcp", transport="streamable-http")
        ],
        prompt="build it",
    )

    config = json.loads(conn.written[".hud_mcp_config.json"])
    assert config == {"mcpServers": {"database": {"type": "http", "url": "http://db:80/mcp"}}}
    assert sink == {}
    assert conn.deleted == [".hud_mcp_config.json"]
    assert "--mcp-config .hud_mcp_config.json" in conn.ran[0]


async def test_exec_records_steps_before_process_exit() -> None:
    process = _FakeStreamProcess(_STREAM_JSON, pause_after=1)
    conn = _FakeConn({}, process)
    ssh = _ssh_with_conn("bash", conn)
    run = _fake_run()

    execution = asyncio.create_task(
        run_claude(
            ClaudeCLIConfig(),
            run,
            ssh=ssh,
            prompt="edit it",
        )
    )
    await process.stdout.blocked.wait()

    assert not execution.done()
    assert len(run.steps) == 1
    first = run.steps[0]
    assert isinstance(first, AgentStep)
    assert first.content == "editing"
    assert first.tool_calls[0].id == "tool-1"

    process.stdout.release.set()
    await execution

    assert [type(step) for step in run.steps] == [AgentStep, ToolStep, AgentStep]
    tool = cast("ToolStep", run.steps[1])
    assert tool.started_at == first.ended_at
    assert tool.result is not None
    text = tool.result.content[0]
    assert isinstance(text, TextContent)
    assert text.text == "wrote a.txt"
    image = tool.result.content[1]
    assert isinstance(image, ImageContent)
    assert image.mimeType == "image/png"
    assert image.data == "aW1hZ2U="
    final = cast("AgentStep", run.steps[2])
    assert final.started_at == tool.ended_at
    assert run.trace.status is None
    assert run.trace.content == "done"


async def test_exec_forwards_trace_id_only_to_hud_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "api_key", "hud-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "anthropic-key")

    gateway_conn = _FakeConn({}, _FakeStreamProcess(_STREAM_JSON))
    gateway = ClaudeCLIConfig(gateway=True)
    with set_trace_context("trace-123"):
        await run_claude(
            gateway,
            _fake_run(),
            ssh=_ssh_with_conn("bash", gateway_conn),
            prompt="build it",
        )
    assert "ANTHROPIC_CUSTOM_HEADERS='Trace-Id: trace-123'" in gateway_conn.ran[0]

    provider_conn = _FakeConn({}, _FakeStreamProcess(_STREAM_JSON))
    provider = ClaudeCLIConfig(gateway=False)
    with set_trace_context("trace-123"):
        await run_claude(
            provider,
            _fake_run(),
            ssh=_ssh_with_conn("bash", provider_conn),
            prompt="build it",
        )
    assert "ANTHROPIC_CUSTOM_HEADERS" not in provider_conn.ran[0]


async def test_exec_closes_streaming_process_when_cancelled() -> None:
    process = _FakeStreamProcess(_STREAM_JSON, pause_after=0)
    conn = _FakeConn({}, process)
    execution = asyncio.create_task(
        run_claude(
            ClaudeCLIConfig(),
            _fake_run(),
            ssh=_ssh_with_conn("bash", conn),
            prompt="build it",
        )
    )
    await process.stdout.blocked.wait()
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution

    assert process.closed


async def test_exec_nonzero_exit_with_no_stdout_raises() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, _FakeStreamProcess("", stderr="boom", exit_status=1))
    ssh = _ssh_with_conn("cmd", conn)

    run = _fake_run()
    with pytest.raises(RuntimeError, match="boom"):
        await run_claude(ClaudeCLIConfig(), run, ssh=ssh, prompt="x")

    assert run.trace.extra["returncode"] == 1


async def test_exec_signal_exit_records_the_returncode() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(
        sink,
        _FakeStreamProcess("", exit_status=None, returncode=-15),
    )
    ssh = _ssh_with_conn("bash", conn)

    run = _fake_run()
    with pytest.raises(RuntimeError, match="return code -15"):
        await run_claude(ClaudeCLIConfig(), run, ssh=ssh, prompt="x")

    assert run.trace.extra["returncode"] == -15


async def test_exec_nonzero_exit_with_result_stream_remains_an_error() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(
        sink,
        _FakeStreamProcess(_STREAM_JSON, stderr="transport failed", exit_status=1),
    )
    ssh = _ssh_with_conn("bash", conn)

    run = _fake_run()
    with pytest.raises(RuntimeError, match="transport failed"):
        await run_claude(ClaudeCLIConfig(), run, ssh=ssh, prompt="x")

    assert run.trace.content == "done"
    assert run.trace.extra["returncode"] == 1
    assert run.trace.extra["stderr"] == "transport failed"
    assert "messages" not in run.trace.extra


async def test_exec_zero_exit_without_result_event_is_an_error() -> None:
    sink: dict[str, bytes] = {}
    stdout = _STREAM_JSON.rsplit('{"type":"result"', 1)[0]
    conn = _FakeConn(sink, _FakeStreamProcess(stdout))
    ssh = _ssh_with_conn("bash", conn)

    run = _fake_run()
    with pytest.raises(RuntimeError, match="without a result event"):
        await run_claude(ClaudeCLIConfig(), run, ssh=ssh, prompt="x")

    assert run.trace.content == "done"


@pytest.mark.parametrize(
    ("transport", "claude_type"),
    [("streamable-http", "http"), ("sse", "sse")],
)
async def test_manifest_mcp_capability_is_written_for_remote_claude(
    transport: Literal["streamable-http", "sse"],
    claude_type: str,
) -> None:
    mcp = Capability.mcp(name="database", url="http://database:8000/mcp", transport=transport)
    conn = _FakeConn({}, _FakeStreamProcess(_STREAM_JSON))

    await run_claude(
        ClaudeCLIConfig(),
        _fake_run(),
        ssh=_ssh_with_conn("bash", conn),
        prompt="call the tool",
        bindings=[mcp],
    )

    assert json.loads(conn.written[".hud_mcp_config.json"]) == {
        "mcpServers": {"database": {"type": claude_type, "url": "http://database:8000/mcp"}}
    }


async def test_remote_claude_bridges_computer_mcp_for_the_cli_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = Capability.rfb(name="screen", url="rfb://localhost:5900", display=0)
    routed = Capability.rfb(name="screen", url="rfb://127.0.0.1:41000", display=0)
    encoding = WebPScreenshotEncoding(quality=42)
    conn = _FakeConn({}, _FakeStreamProcess(_STREAM_JSON))
    ssh = _ssh_with_conn("bash", conn)
    runs_while_bridged: list[int] = []

    @asynccontextmanager
    async def bridge(
        bridge_ssh: SSHClient,
        capability: Capability,
        screenshot_encoding: WebPScreenshotEncoding,
        *,
        shell: str,
    ) -> Any:
        assert bridge_ssh is ssh
        assert capability == routed
        assert screenshot_encoding == encoding
        assert shell == "bash"
        yield {"type": "stdio", "command": "sh", "args": ["-c", "relay"]}
        runs_while_bridged.append(len(conn.ran))

    monkeypatch.setattr(computer_mcp, "bridge_computer_mcp", bridge)

    await run_claude(
        ClaudeCLIConfig(screenshot_encoding=encoding),
        _fake_run(),
        ssh=ssh,
        prompt="use the computer",
        bindings=[screen],
        routed={"screen": routed},
    )

    assert runs_while_bridged == [1]
    assert json.loads(conn.written[".hud_mcp_config.json"]) == {
        "mcpServers": {"computer-use": {"type": "stdio", "command": "sh", "args": ["-c", "relay"]}}
    }


async def test_remote_claude_preserves_multiple_rfb_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screens = [
        Capability.rfb(name="screen-0", url="rfb://display-0:5900", display=0),
        Capability.rfb(name="screen-1", url="rfb://display-1:5901", display=1),
    ]
    routed = {
        cap.name: Capability.rfb(name=cap.name, url=f"rfb://127.0.0.1:{41000 + i}", display=i)
        for i, cap in enumerate(screens)
    }
    conn = _FakeConn({}, _FakeStreamProcess(_STREAM_JSON))

    @asynccontextmanager
    async def bridge(
        _ssh: SSHClient,
        capability: Capability,
        _encoding: WebPScreenshotEncoding,
        *,
        shell: str,
    ) -> Any:
        yield {"type": "stdio", "command": "sh", "args": ["-c", capability.name]}

    monkeypatch.setattr(computer_mcp, "bridge_computer_mcp", bridge)

    await run_claude(
        ClaudeCLIConfig(),
        _fake_run(),
        ssh=_ssh_with_conn("bash", conn),
        prompt="use both screens",
        bindings=screens,
        routed=routed,
    )

    assert json.loads(conn.written[".hud_mcp_config.json"]) == {
        "mcpServers": {
            f"computer-use-{name}": {"type": "stdio", "command": "sh", "args": ["-c", name]}
            for name in ("screen-0", "screen-1")
        }
    }


async def test_computer_mcp_stdio_owns_rfb_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = Capability.rfb(name="screen", url="rfb://localhost:5900", display=0)
    encoding = WebPScreenshotEncoding(quality=42)
    rfb = SimpleNamespace(close=AsyncMock())
    connect = AsyncMock(return_value=rfb)
    server = SimpleNamespace(run_async=AsyncMock())
    create = Mock(return_value=server)
    monkeypatch.setattr(computer_mcp.RFBClient, "connect", connect)
    monkeypatch.setattr(computer_mcp, "create_computer_mcp", create)

    await computer_mcp.run_computer_mcp(
        {
            computer_mcp.RFB_CAPABILITY_ENV: json.dumps(screen.to_manifest()),
            computer_mcp.SCREENSHOT_ENCODING_ENV: encoding.model_dump_json(),
        }
    )

    connect.assert_awaited_once_with(screen)
    create.assert_called_once_with(rfb, encoding)
    server.run_async.assert_awaited_once_with(transport="stdio", show_banner=False)
    rfb.close.assert_awaited_once()


async def test_computer_mcp_preserves_tool_result(monkeypatch: pytest.MonkeyPatch) -> None:
    result = MCPToolResult(
        content=[TextContent(type="text", text="failed")],
        isError=True,
    )
    execute = AsyncMock(return_value=result)
    monkeypatch.setattr(computer_mcp.ClaudeComputerTool, "execute", execute)
    server = computer_mcp.create_computer_mcp(cast("Any", object()), WebPScreenshotEncoding())

    async with fastmcp.Client(server) as client:
        received = await client.call_tool_mcp(
            "computer",
            {"action": "left_click", "coordinate": [10, 20]},
        )

    execute.assert_awaited_once_with({"action": "left_click", "coordinate": [10, 20]})
    assert received.isError is True
    assert received.content == result.content


class _ByteWriter:
    def __init__(self) -> None:
        self.closed = False
        self.data = bytearray()

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _LocalComputerProcess:
    def __init__(self) -> None:
        self.stdin = _ByteWriter()
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        assert self.returncode is not None
        return self.returncode


async def test_computer_mcp_bridge_uses_controller_python_and_owns_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_stdout = asyncio.StreamReader()
    bridge_stderr = asyncio.StreamReader()
    bridge_stderr.feed_data(b"ready\n")
    bridge_stdin = _ByteWriter()
    bridge = SimpleNamespace(
        stdin=bridge_stdin,
        stdout=bridge_stdout,
        stderr=bridge_stderr,
        channel=SimpleNamespace(close=Mock()),
        wait_closed=AsyncMock(),
    )
    connection = SimpleNamespace(create_process=AsyncMock(return_value=bridge))
    ssh = SimpleNamespace(create_process=connection.create_process)
    local = _LocalComputerProcess()
    spawn = AsyncMock(return_value=local)
    monkeypatch.setattr(computer_mcp.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(computer_mcp.secrets, "token_hex", lambda _length: "bridge-token")
    screen = Capability.rfb(name="screen", url="rfb://127.0.0.1:41000", display=0)
    encoding = WebPScreenshotEncoding(quality=42)

    async with computer_mcp.bridge_computer_mcp(
        cast("Any", ssh),
        screen,
        encoding,
        shell="bash",
    ) as config:
        assert config == {
            "type": "stdio",
            "command": "sh",
            "args": [
                "-c",
                "cat /tmp/hud-computer-bridge-token.response & reader=$!; "
                "cat > /tmp/hud-computer-bridge-token.request; wait $reader",
            ],
        }
        assert not bridge_stdin.closed
        assert not local.stdin.closed

    bridge_command = connection.create_process.await_args.args[0]
    assert "mkfifo -- /tmp/hud-computer-bridge-token.request" in bridge_command
    assert "printf 'ready\\n' >&2" in bridge_command
    spawn.assert_awaited_once()
    spawn_call = spawn.await_args
    assert spawn_call is not None
    spawn_args = spawn_call.args
    assert spawn_args[:3] == (
        sys.executable,
        "-m",
        "hud.agents.claude.sdk.computer_mcp",
    )
    environ = spawn_call.kwargs["env"]
    assert json.loads(environ[computer_mcp.RFB_CAPABILITY_ENV]) == screen.to_manifest()
    assert environ[computer_mcp.SCREENSHOT_ENCODING_ENV] == encoding.model_dump_json()
    bridge.channel.close.assert_called_once()
    bridge.wait_closed.assert_awaited_once()
    assert bridge_stdin.closed
    assert local.stdin.closed
    assert local.terminated
    assert not local.killed


async def test_computer_mcp_bridge_rejects_windows_before_starting_resources() -> None:
    screen = Capability.rfb(name="screen", url="rfb://127.0.0.1:41000", display=0)
    ssh = SimpleNamespace(create_process=AsyncMock())

    with pytest.raises(RuntimeError, match="requires a POSIX workspace"):
        async with computer_mcp.bridge_computer_mcp(
            cast("Any", ssh),
            screen,
            WebPScreenshotEncoding(),
            shell="powershell",
        ):
            pass

    ssh.create_process.assert_not_awaited()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX FIFO relay")
async def test_computer_mcp_bridge_relays_both_directions(monkeypatch: pytest.MonkeyPatch) -> None:
    """The remote FIFO bridge and the relay Claude runs form one duplex pipe to the controller."""
    spawn = asyncio.create_subprocess_exec

    async def remote_shell(command: str) -> Any:
        process = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return SimpleNamespace(
            stdin=process.stdin,
            stdout=process.stdout,
            stderr=process.stderr,
            channel=SimpleNamespace(close=lambda: None),
            wait_closed=process.wait,
        )

    async def echo_controller(*_args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        return await spawn(
            "cat", stdin=kwargs["stdin"], stdout=kwargs["stdout"], stderr=kwargs["stderr"]
        )

    monkeypatch.setattr(computer_mcp.asyncio, "create_subprocess_exec", echo_controller)
    screen = Capability.rfb(name="screen", url="rfb://127.0.0.1:41000", display=0)

    async with asyncio.timeout(10):
        async with computer_mcp.bridge_computer_mcp(
            cast("Any", SimpleNamespace(create_process=remote_shell)),
            screen,
            WebPScreenshotEncoding(),
            shell="bash",
        ) as config:
            relay = await spawn(
                config["command"],
                *config["args"],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
            assert relay.stdin is not None and relay.stdout is not None
            relay.stdin.write(b'{"method":"tools/list"}\n')
            await relay.stdin.drain()
            assert await relay.stdout.readline() == b'{"method":"tools/list"}\n'
            relay.stdin.close()
        await relay.wait()


async def test_concurrent_runs_keep_their_ssh_state_isolated() -> None:
    first_process = _FakeStreamProcess(_STREAM_JSON, pause_after=0)
    conn_a = _FakeConn({}, first_process)
    conn_b = _FakeConn({}, _FakeStreamProcess(_STREAM_JSON))
    run_a, run_b = _fake_run(), _fake_run()

    first = asyncio.create_task(
        run_claude(ClaudeCLIConfig(), run_a, ssh=_ssh_with_conn("bash", conn_a), prompt="first")
    )
    await first_process.stdout.blocked.wait()
    await run_claude(
        ClaudeCLIConfig(), run_b, ssh=_ssh_with_conn("powershell", conn_b), prompt="second"
    )
    first_process.stdout.release.set()
    await first

    assert conn_a.write_commands == []
    assert "--input-format=stream-json" in conn_a.ran[0]
    assert json.loads(first_process.stdin.data)["message"]["content"][0]["text"] == "first"
    assert conn_b.ran == ["cmd /c .hud_run.bat"]
    assert json.loads(conn_b.written[".hud_input.jsonl"])["message"]["content"][0]["text"] == (
        "second"
    )
    assert run_a.trace.content == run_b.trace.content == "done"
