"""ClaudeSDKAgent remote-command construction over the workspace SSH.

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

import pytest

from hud.agents.claude.sdk import computer_mcp
from hud.agents.claude.sdk.agent import ClaudeSDKAgent, build_remote_invocation
from hud.agents.types import ClaudeSDKConfig
from hud.capabilities import Capability, SSHClient
from hud.capabilities.rfb import WebPScreenshotEncoding
from hud.settings import settings
from hud.telemetry.context import set_trace_context

if TYPE_CHECKING:
    from pathlib import Path

# ─── build_remote_invocation (pure) ───────────────────────────────────


@pytest.mark.parametrize("shell", ["cmd", "powershell"])
def test_windows_shell_runs_batch_file_via_cmd(shell: str) -> None:
    inv = build_remote_invocation(shell, "claude --print -- hi")

    # The bare filename is rejected by the remote shell; cmd /c runs it.
    assert inv.command == "cmd /c .hud_run.bat"
    assert inv.script_name == ".hud_run.bat"
    assert inv.script_body == "@echo off\r\nclaude --print -- hi\r\n"


def test_posix_shell_runs_inline_with_install_check() -> None:
    inv = build_remote_invocation("bash", "claude --print -- hi")

    assert inv.script_name is None
    assert inv.script_body is None
    assert "install.sh" in inv.command  # one-shot bootstrap prefix
    assert inv.command.endswith(" && claude --print -- hi")


def test_gateway_trace_headers_are_forwarded_to_claude_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "api_key", "test-key")
    agent = ClaudeSDKAgent()

    with set_trace_context("child-trace", parent_trace_id="parent-trace"):
        env = agent._build_env_vars()

    assert env["ANTHROPIC_CUSTOM_HEADERS"] == (
        "Trace-Id: child-trace\nX-HUD-Parent-Trace-Id: parent-trace"
    )


# ─── _exec end-to-end over a fake SSH workspace ────────────────────────


class _FakeConn:
    def __init__(self, sink: dict[str, bytes], result: Any) -> None:
        self._sink = sink
        self._result = result
        self.ran: list[str] = []
        self.write_commands: list[str] = []

    def is_closed(self) -> bool:
        return False

    async def run(
        self,
        cmd: str,
        *,
        input: str | None = None,
        check: bool = True,
        encoding: str | None = "utf-8",
    ) -> Any:
        if input is not None or cmd.startswith("powershell "):
            self.write_commands.append(cmd)
            script = cmd
            if match := re.search(r"-EncodedCommand (\S+)", cmd):
                script = base64.b64decode(match.group(1)).decode("utf-16-le")
            name = next(
                path
                for path in (".hud_prompt.txt", ".hud_run.bat", ".hud_mcp_config.json")
                if path in script
            )
            if input is not None:
                self._sink[name] = input.encode()
            elif match := re.search(r"FromBase64String\('([^']+)'\)", script):
                self._sink[name] += base64.b64decode(match.group(1))
            else:
                self._sink[name] = b""
            return SimpleNamespace(stdout="", stderr="", exit_status=0, returncode=0)
        self.ran.append(cmd)
        return self._result

    async def create_process(self, cmd: str, **kwargs: Any) -> _FakeProcess:
        return _FakeProcess(await self.run(cmd, **kwargs))


class _FakeProcess:
    def __init__(self, result: Any) -> None:
        self._result = result

    async def wait(self, *, check: bool, **kwargs: Any) -> Any:
        del check
        assert kwargs == {"timeout": None}
        return self._result

    def terminate(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


def _fake_run() -> Any:
    trace = SimpleNamespace(status="", content="", extra={})
    steps: list[Any] = []
    return SimpleNamespace(trace=trace, record=steps.append, steps=steps)


_STREAM_JSON = (
    '{"type":"assistant","message":{"content":[{"type":"text","text":"working"}]}}\n'
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
    conn = _FakeConn(
        sink,
        SimpleNamespace(stdout=_STREAM_JSON, stderr="", exit_status=0, returncode=0),
    )
    agent = ClaudeSDKAgent()
    ssh = _ssh_with_conn("cmd", conn)

    run = _fake_run()
    await agent._exec(run, ssh=ssh, shell="cmd", mcp_servers={}, prompt="build it", max_steps=5)

    assert conn.ran == ["cmd /c .hud_run.bat"]
    assert all(command.startswith("powershell ") for command in conn.write_commands)
    assert sink[".hud_run.bat"].startswith(b"@echo off\r\n")
    assert sink[".hud_prompt.txt"] == b"build it"
    assert run.trace.status == "completed"
    assert "done" in run.trace.content


async def test_exec_on_bash_runs_inline_without_batch() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(
        sink,
        SimpleNamespace(stdout=_STREAM_JSON, stderr="", exit_status=0, returncode=0),
    )
    agent = ClaudeSDKAgent()
    ssh = _ssh_with_conn("bash", conn)

    run = _fake_run()
    await agent._exec(run, ssh=ssh, shell="bash", mcp_servers={}, prompt="build it", max_steps=5)

    assert ".hud_run.bat" not in sink
    assert conn.write_commands == ["cat > .hud_prompt.txt"]
    assert len(conn.ran) == 1
    assert "install.sh" in conn.ran[0]
    assert "claude" in conn.ran[0]
    assert run.trace.status == "completed"


async def test_exec_nonzero_exit_with_no_stdout_records_system_error() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(
        sink,
        SimpleNamespace(stdout="", stderr="boom", exit_status=1, returncode=1),
    )
    agent = ClaudeSDKAgent()
    ssh = _ssh_with_conn("cmd", conn)

    run = _fake_run()
    await agent._exec(run, ssh=ssh, shell="cmd", mcp_servers={}, prompt="x", max_steps=1)

    assert run.trace.status == "error"
    assert run.trace.extra["returncode"] == 1
    assert run.steps[0].error == "boom"


async def test_exec_signal_exit_records_the_returncode() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(
        sink,
        SimpleNamespace(stdout="", stderr="", exit_status=None, returncode=-15),
    )
    agent = ClaudeSDKAgent()
    ssh = _ssh_with_conn("bash", conn)

    run = _fake_run()
    await agent._exec(run, ssh=ssh, shell="bash", mcp_servers={}, prompt="x", max_steps=1)

    assert run.trace.status == "error"
    assert run.trace.extra["returncode"] == -15
    assert run.steps[0].error == "claude CLI exited with return code -15"


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

    agent = ClaudeSDKAgent()
    execute = AsyncMock()
    monkeypatch.setattr(agent, "_exec", execute)

    with set_trace_context("child-trace"):
        await agent(
            cast(
                "Any",
                SimpleNamespace(client=Client(), prompt_text="call the tool"),
            )
        )

    await_args = execute.await_args
    assert await_args is not None
    assert await_args.kwargs["mcp_servers"] == {
        "database": {
            "type": claude_type,
            "url": "http://database:8000/mcp",
            "headers": {"Trace-Id": "child-trace"},
        }
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
    agent = ClaudeSDKAgent(ClaudeSDKConfig(screenshot_encoding=encoding))

    async def execute(*_args: Any, **_kwargs: Any) -> None:
        assert bridge_active

    execute_mock = AsyncMock(side_effect=execute)
    monkeypatch.setattr(computer_mcp, "bridge_computer_mcp", bridge)
    monkeypatch.setattr(agent, "_exec", execute_mock)

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

    agent = ClaudeSDKAgent()
    monkeypatch.setattr(computer_mcp, "bridge_computer_mcp", bridge)
    monkeypatch.setattr(agent, "_exec", execute)

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

    agent = ClaudeSDKAgent()
    monkeypatch.setattr(agent, "_exec", execute)
    run_a = SimpleNamespace(client=Client(shell_a, ssh_a), prompt_text="first")
    run_b = SimpleNamespace(client=Client(shell_b, ssh_b), prompt_text="second")

    first = asyncio.create_task(agent(cast("Any", run_a)))
    await first_entered.wait()
    await agent(cast("Any", run_b))
    release_first.set()
    await first

    assert seen == [(run_a, ssh_a, "bash"), (run_b, ssh_b, "powershell")]
