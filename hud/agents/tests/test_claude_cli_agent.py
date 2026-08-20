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
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import AsyncMock, Mock

import fastmcp
import pytest
from mcp.types import ImageContent, TextContent

from hud.agents.claude.cli import computer_mcp
from hud.agents.claude.cli.agent import ClaudeCLIAgent
from hud.agents.types import AgentStep, ClaudeCLIConfig, ToolStep
from hud.capabilities import Capability, SSHClient
from hud.capabilities.rfb import WebPScreenshotEncoding
from hud.settings import settings
from hud.telemetry.context import set_trace_context
from hud.types import MCPToolResult

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", None)


def test_command_follows_explicit_gateway_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", "hud-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "anthropic-key")

    gateway = ClaudeCLIAgent(ClaudeCLIConfig(use_hud_gateway=True))._build_command(
        shell="bash", prompt="run", mcp_config_path=None
    )
    provider = ClaudeCLIAgent(ClaudeCLIConfig(use_hud_gateway=False))._build_command(
        shell="bash", prompt="run", mcp_config_path=None
    )

    assert f"ANTHROPIC_BASE_URL={settings.hud_gateway_url}" in gateway
    assert "ANTHROPIC_API_KEY=hud-key" in gateway
    assert "ANTHROPIC_API_KEY=anthropic-key" in provider
    assert "ANTHROPIC_BASE_URL" not in provider
    assert "ANTHROPIC_MODEL=claude-sonnet-5" in provider


def test_windows_command_encodes_environment_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "api_key", "hud&key's")
    agent = ClaudeCLIAgent(
        ClaudeCLIConfig(
            use_hud_gateway=True,
            max_steps=3,
            system_prompt="don't $expand",
        )
    )

    command = agent._build_command(
        shell="powershell",
        prompt="not embedded",
    )

    encoded = command.rsplit(" ", 1)[1]
    script = base64.b64decode(encoded).decode("utf-16-le")
    assert "$env:ANTHROPIC_API_KEY='hud&key''s'" in script
    assert "'--system-prompt' 'don''t $expand'" in script
    assert "Get-Content -Raw -Encoding UTF8 '.hud_prompt.txt' | & claude" in script
    assert "not embedded" not in script
    assert "python" not in script


class _FakeReader:
    def __init__(self, value: str, *, pause_after: int | None = None) -> None:
        self._raw = value.encode()
        self._lines = self._raw.splitlines(keepends=True)
        self._pause_after = pause_after
        self._index = 0
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()

    async def readline(self) -> bytes:
        if self._pause_after == self._index:
            self.blocked.set()
            await self.release.wait()
            self._pause_after = None
        if self._index == len(self._lines):
            return b""
        line = self._lines[self._index]
        self._index += 1
        return line

    async def read(self) -> bytes:
        return self._raw


class _FakeStreamProcess:
    def __init__(
        self,
        stdout: str,
        *,
        stderr: str = "",
        exit_status: int | None = 0,
        returncode: int | None = None,
        pause_after: int | None = None,
    ) -> None:
        self.stdout = _FakeReader(stdout, pause_after=pause_after)
        self.stderr = _FakeReader(stderr)
        self.exit_status = exit_status
        self.returncode = exit_status if returncode is None else returncode
        self.closed = False
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


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
                for path in (".hud_prompt.txt", ".hud_run.bat", ".hud_mcp_config.json")
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
        assert kwargs == {"encoding": None}
        self.ran.append(cmd)
        return self._process


def _fake_run() -> Any:
    trace = SimpleNamespace(status=None, content="", extra={})
    steps: list[Any] = []
    return SimpleNamespace(trace=trace, record=steps.append, steps=steps)


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
        params={"shell": shell},
    )
    return SSHClient(capability, cast("Any", conn))


async def test_exec_on_windows_writes_batch_and_execs_via_cmd() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, _FakeStreamProcess(_STREAM_JSON))
    agent = ClaudeCLIAgent()
    ssh = _ssh_with_conn("cmd", conn)

    run = _fake_run()
    await agent._run_cli(run, ssh=ssh, shell="cmd", mcp_servers={}, prompt="build it")

    assert conn.ran == ["cmd /c .hud_run.bat"]
    assert all(command.startswith("powershell ") for command in conn.write_commands)
    assert conn.written[".hud_run.bat"].startswith(b"@echo off\r\n")
    assert conn.written[".hud_prompt.txt"] == b"build it"
    assert sink == {}
    assert set(conn.deleted) == {".hud_prompt.txt", ".hud_run.bat"}
    assert run.trace.status is None
    assert run.trace.content == "done"
    assert "messages" not in run.trace.extra


async def test_exec_on_bash_runs_inline_without_batch() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, _FakeStreamProcess(_STREAM_JSON))
    agent = ClaudeCLIAgent()
    ssh = _ssh_with_conn("bash", conn)

    run = _fake_run()
    await agent._run_cli(run, ssh=ssh, shell="bash", mcp_servers={}, prompt="build it")

    assert sink == {}
    assert conn.write_commands == []
    assert conn.deleted == []
    assert len(conn.ran) == 1
    assert "claude" in conn.ran[0]
    assert run.trace.status is None
    assert run.trace.content == "done"
    assert "messages" not in run.trace.extra


async def test_exec_removes_mcp_config_after_run() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, _FakeStreamProcess(_STREAM_JSON))
    agent = ClaudeCLIAgent()

    await agent._run_cli(
        _fake_run(),
        ssh=_ssh_with_conn("bash", conn),
        shell="bash",
        mcp_servers={"database": {"type": "http", "url": "http://db/mcp"}},
        prompt="build it",
    )

    config = json.loads(conn.written[".hud_mcp_config.json"])
    assert config == {"mcpServers": {"database": {"type": "http", "url": "http://db/mcp"}}}
    assert sink == {}
    assert conn.deleted == [".hud_mcp_config.json"]
    assert "--mcp-config .hud_mcp_config.json" in conn.ran[0]


async def test_exec_records_steps_before_process_exit() -> None:
    process = _FakeStreamProcess(_STREAM_JSON, pause_after=1)
    conn = _FakeConn({}, process)
    agent = ClaudeCLIAgent()
    ssh = _ssh_with_conn("bash", conn)
    run = _fake_run()

    execution = asyncio.create_task(
        agent._run_cli(run, ssh=ssh, shell="bash", mcp_servers={}, prompt="edit it")
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
    gateway = ClaudeCLIAgent(ClaudeCLIConfig(use_hud_gateway=True))
    with set_trace_context("trace-123"):
        await gateway._run_cli(
            _fake_run(),
            ssh=_ssh_with_conn("bash", gateway_conn),
            shell="bash",
            mcp_servers={},
            prompt="build it",
        )
    assert "ANTHROPIC_CUSTOM_HEADERS='Trace-Id: trace-123'" in gateway_conn.ran[0]

    provider_conn = _FakeConn({}, _FakeStreamProcess(_STREAM_JSON))
    provider = ClaudeCLIAgent(ClaudeCLIConfig(use_hud_gateway=False))
    with set_trace_context("trace-123"):
        await provider._run_cli(
            _fake_run(),
            ssh=_ssh_with_conn("bash", provider_conn),
            shell="bash",
            mcp_servers={},
            prompt="build it",
        )
    assert "ANTHROPIC_CUSTOM_HEADERS" not in provider_conn.ran[0]


async def test_exec_closes_streaming_process_when_cancelled() -> None:
    process = _FakeStreamProcess(_STREAM_JSON, pause_after=0)
    conn = _FakeConn({}, process)
    agent = ClaudeCLIAgent()

    execution = asyncio.create_task(
        agent._run_cli(
            _fake_run(),
            ssh=_ssh_with_conn("bash", conn),
            shell="bash",
            mcp_servers={},
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
    agent = ClaudeCLIAgent()
    ssh = _ssh_with_conn("cmd", conn)

    run = _fake_run()
    with pytest.raises(RuntimeError, match="boom"):
        await agent._run_cli(run, ssh=ssh, shell="cmd", mcp_servers={}, prompt="x")

    assert run.trace.extra["returncode"] == 1


async def test_exec_signal_exit_records_the_returncode() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(
        sink,
        _FakeStreamProcess("", exit_status=None, returncode=-15),
    )
    agent = ClaudeCLIAgent()
    ssh = _ssh_with_conn("bash", conn)

    run = _fake_run()
    with pytest.raises(RuntimeError, match="return code -15"):
        await agent._run_cli(run, ssh=ssh, shell="bash", mcp_servers={}, prompt="x")

    assert run.trace.extra["returncode"] == -15


async def test_exec_nonzero_exit_with_result_stream_remains_an_error() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(
        sink,
        _FakeStreamProcess(_STREAM_JSON, stderr="transport failed", exit_status=1),
    )
    agent = ClaudeCLIAgent()
    ssh = _ssh_with_conn("bash", conn)

    run = _fake_run()
    with pytest.raises(RuntimeError, match="transport failed"):
        await agent._run_cli(run, ssh=ssh, shell="bash", mcp_servers={}, prompt="x")

    assert run.trace.content == "done"
    assert run.trace.extra["returncode"] == 1
    assert run.trace.extra["stderr"] == "transport failed"
    assert "messages" not in run.trace.extra


async def test_exec_zero_exit_without_result_event_is_an_error() -> None:
    sink: dict[str, bytes] = {}
    stdout = _STREAM_JSON.rsplit('{"type":"result"', 1)[0]
    conn = _FakeConn(sink, _FakeStreamProcess(stdout))
    agent = ClaudeCLIAgent()
    ssh = _ssh_with_conn("bash", conn)

    run = _fake_run()
    with pytest.raises(RuntimeError, match="without a result event"):
        await agent._run_cli(run, ssh=ssh, shell="bash", mcp_servers={}, prompt="x")

    assert run.trace.content == "done"


@pytest.mark.parametrize(
    ("transport", "claude_type"),
    [("streamable-http", "http"), ("sse", "sse")],
)
async def test_manifest_mcp_capability_is_written_for_remote_claude(
    monkeypatch: pytest.MonkeyPatch,
    transport: Literal["streamable-http", "sse"],
    claude_type: str,
) -> None:
    shell = Capability(
        name="shell",
        protocol="ssh/2",
        url="ssh://localhost:22",
        params={"shell": "bash"},
    )
    mcp = Capability.mcp(
        name="database",
        url="http://database:8000/mcp",
        transport=transport,
    )
    ssh = SSHClient(shell, cast("Any", object()))

    class Client:
        manifest = SimpleNamespace(bindings=[shell, mcp])

        async def open(self, ref: str) -> SSHClient:
            assert ref == "ssh"
            return ssh

    agent = ClaudeCLIAgent()
    execute = AsyncMock()
    monkeypatch.setattr(agent, "_run_cli", execute)

    await agent(
        cast(
            "Any",
            SimpleNamespace(client=Client(), prompt_text="call the tool"),
        )
    )

    await_args = execute.await_args
    assert await_args is not None
    assert await_args.kwargs["mcp_servers"] == {
        "database": {"type": claude_type, "url": "http://database:8000/mcp"}
    }
    execute.assert_awaited_once()


async def test_remote_claude_passes_screenshot_encoding_to_computer_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = Capability(
        name="shell",
        protocol="ssh/2",
        url="ssh://localhost:22",
        params={"shell": "bash"},
    )
    screen = Capability.rfb(name="screen", url="rfb://localhost:5900", display=0)
    routed = Capability.rfb(name="screen", url="rfb://127.0.0.1:41000", display=0)
    ssh = SSHClient(shell, cast("Any", object()))
    opened: list[str] = []
    bridge_active = False

    class Client:
        manifest = SimpleNamespace(bindings=[shell, screen])

        async def open(self, ref: str) -> SSHClient:
            opened.append(ref)
            assert ref == "ssh"
            return ssh

        def binding(self, ref: str) -> Capability:
            assert ref == "screen"
            return routed

    @asynccontextmanager
    async def bridge(
        bridge_ssh: SSHClient,
        capability: Capability,
        screenshot_encoding: WebPScreenshotEncoding,
        *,
        shell: str,
    ) -> Any:
        nonlocal bridge_active
        assert bridge_ssh is ssh
        assert capability == routed
        assert screenshot_encoding == encoding
        assert shell == "bash"
        bridge_active = True
        try:
            yield {"type": "stdio", "command": "sh", "args": ["-c", "relay"]}
        finally:
            bridge_active = False

    encoding = WebPScreenshotEncoding(quality=42)
    agent = ClaudeCLIAgent(ClaudeCLIConfig(screenshot_encoding=encoding))

    async def execute(*_args: Any, **_kwargs: Any) -> None:
        assert bridge_active

    execute_mock = AsyncMock(side_effect=execute)
    monkeypatch.setattr(computer_mcp, "bridge_computer_mcp", bridge)
    monkeypatch.setattr(agent, "_run_cli", execute_mock)

    await agent(
        cast(
            "Any",
            SimpleNamespace(client=Client(), prompt_text="use the computer"),
        )
    )

    assert opened == ["ssh"]
    await_args = execute_mock.await_args
    assert await_args is not None
    server = await_args.kwargs["mcp_servers"]["computer-use"]
    assert server == {
        "type": "stdio",
        "command": "sh",
        "args": ["-c", "relay"],
    }
    assert not bridge_active


async def test_remote_claude_preserves_multiple_rfb_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = Capability(
        name="shell",
        protocol="ssh/2",
        url="ssh://localhost:22",
        params={"shell": "bash"},
    )
    screens = [
        Capability.rfb(name="screen-0", url="rfb://display-0:5900", display=0),
        Capability.rfb(name="screen-1", url="rfb://display-1:5901", display=1),
    ]
    routed = {
        cap.name: Capability.rfb(
            name=cap.name,
            url=f"rfb://127.0.0.1:{41000 + index}",
            display=index,
        )
        for index, cap in enumerate(screens)
    }
    ssh = SSHClient(shell, cast("Any", object()))
    bridged: list[str] = []

    class Client:
        manifest = SimpleNamespace(bindings=[shell, *screens])

        async def open(self, ref: str) -> SSHClient:
            assert ref == "ssh"
            return ssh

        def binding(self, ref: str) -> Capability:
            return routed[ref]

    @asynccontextmanager
    async def bridge(
        _ssh: SSHClient,
        capability: Capability,
        _encoding: WebPScreenshotEncoding,
        *,
        shell: str,
    ) -> Any:
        assert shell == "bash"
        bridged.append(capability.name)
        try:
            yield {"type": "stdio", "command": "sh", "args": ["-c", capability.name]}
        finally:
            bridged.remove(capability.name)

    async def execute(*_args: Any, **kwargs: Any) -> None:
        assert bridged == ["screen-0", "screen-1"]
        assert kwargs["mcp_servers"] == {
            "computer-use-screen-0": {
                "type": "stdio",
                "command": "sh",
                "args": ["-c", "screen-0"],
            },
            "computer-use-screen-1": {
                "type": "stdio",
                "command": "sh",
                "args": ["-c", "screen-1"],
            },
        }

    agent = ClaudeCLIAgent()
    monkeypatch.setattr(computer_mcp, "bridge_computer_mcp", bridge)
    monkeypatch.setattr(agent, "_run_cli", execute)

    await agent(cast("Any", SimpleNamespace(client=Client(), prompt_text="use both screens")))

    assert bridged == []


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
    server = computer_mcp.create_computer_mcp(cast("Any", object()))

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
        "hud.agents.claude.cli.computer_mcp",
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
            shell="powershell",
        ):
            pass

    ssh.create_process.assert_not_awaited()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX FIFO relay")
async def test_computer_mcp_fifo_relay_is_bidirectional(tmp_path: Path) -> None:
    request_path = str(tmp_path / "request")
    response_path = str(tmp_path / "response")
    bridge = await asyncio.create_subprocess_shell(
        computer_mcp._bridge_command(request_path, response_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    relay: asyncio.subprocess.Process | None = None
    try:
        assert bridge.stderr is not None
        assert await asyncio.wait_for(bridge.stderr.readline(), 2) == b"ready\n"
        config = computer_mcp._relay_config(request_path, response_path)
        relay = await asyncio.create_subprocess_exec(
            config["command"],
            *config["args"],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        assert relay.stdin is not None and relay.stdout is not None
        assert bridge.stdin is not None and bridge.stdout is not None

        relay.stdin.write(b'{"method":"tools/list"}\n')
        await relay.stdin.drain()
        assert await asyncio.wait_for(bridge.stdout.readline(), 2) == (b'{"method":"tools/list"}\n')

        bridge.stdin.write(b'{"result":{"tools":[]}}\n')
        await bridge.stdin.drain()
        assert await asyncio.wait_for(relay.stdout.readline(), 2) == b'{"result":{"tools":[]}}\n'
    finally:
        for process in (relay, bridge):
            if process is not None and process.stdin is not None:
                process.stdin.close()
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *(process.wait() for process in (relay, bridge) if process is not None),
                    return_exceptions=True,
                ),
                2,
            )
        except TimeoutError:
            for process in (relay, bridge):
                if process is not None and process.returncode is None:
                    process.kill()


async def test_concurrent_runs_keep_their_ssh_state_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell_a = Capability(
        name="shell-a",
        protocol="ssh/2",
        url="ssh://a:22",
        params={"shell": "bash"},
    )
    shell_b = Capability(
        name="shell-b",
        protocol="ssh/2",
        url="ssh://b:22",
        params={"shell": "powershell"},
    )
    ssh_a = SSHClient(shell_a, cast("Any", object()))
    ssh_b = SSHClient(shell_b, cast("Any", object()))

    class Client:
        def __init__(self, shell: Capability, ssh: SSHClient) -> None:
            self.manifest = SimpleNamespace(bindings=[shell])
            self.ssh = ssh

        async def open(self, ref: str) -> SSHClient:
            assert ref == "ssh"
            return self.ssh

    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    seen: list[tuple[Any, SSHClient, str]] = []

    async def execute(
        run: Any,
        *,
        ssh: SSHClient,
        shell: str,
        mcp_servers: dict[str, dict[str, Any]],
        **_: Any,
    ) -> None:
        assert mcp_servers == {}
        seen.append((run, ssh, shell))
        if run.prompt_text == "first":
            first_entered.set()
            await release_first.wait()

    agent = ClaudeCLIAgent()
    monkeypatch.setattr(agent, "_run_cli", execute)
    run_a = SimpleNamespace(client=Client(shell_a, ssh_a), prompt_text="first")
    run_b = SimpleNamespace(client=Client(shell_b, ssh_b), prompt_text="second")

    first = asyncio.create_task(agent(cast("Any", run_a)))
    await first_entered.wait()
    await agent(cast("Any", run_b))
    release_first.set()
    await first

    assert seen == [(run_a, ssh_a, "bash"), (run_b, ssh_b, "powershell")]
