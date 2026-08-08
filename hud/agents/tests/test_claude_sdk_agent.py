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
import re
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import pytest

from hud.agents.claude.sdk import computer_mcp
from hud.agents.claude.sdk.agent import ClaudeSDKAgent, build_remote_invocation
from hud.agents.types import AgentStep, ClaudeSDKConfig, ToolStep
from hud.capabilities import Capability, RFBClient, SSHClient
from hud.capabilities.rfb import WebPScreenshotEncoding

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


# ─── _exec end-to-end over a fake SSH workspace ────────────────────────


class _FakeProcess:
    def __init__(
        self,
        stdout: str,
        *,
        stderr: str = "",
        exit_status: int = 0,
        pause_after: int | None = None,
    ) -> None:
        self.stdout = self
        self.stderr = self
        self._lines = stdout.splitlines(keepends=True)
        self._stderr = stderr
        self._pause_after = pause_after
        self._index = 0
        self.exit_status = exit_status
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()

    async def readline(self) -> str:
        if self._pause_after == self._index:
            self.blocked.set()
            await self.release.wait()
            self._pause_after = None
        if self._index == len(self._lines):
            return ""
        line = self._lines[self._index]
        self._index += 1
        return line

    async def read(self) -> str:
        return self._stderr

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


class _FakeConn:
    def __init__(self, sink: dict[str, bytes], process: _FakeProcess) -> None:
        self._sink = sink
        self._process = process
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
            return SimpleNamespace(stdout="", stderr="", exit_status=0)
        raise AssertionError(f"unexpected buffered command: {cmd}")

    async def create_process(self, cmd: str, **kwargs: Any) -> _FakeProcess:
        assert kwargs == {}
        self.ran.append(cmd)
        return self._process


def _fake_run() -> Any:
    trace = SimpleNamespace(status="", content="", extra={})
    steps: list[Any] = []
    return SimpleNamespace(trace=trace, record=steps.append, steps=steps)


_STREAM_JSON = (
    '{"type":"assistant","message":{"content":[{"type":"text","text":"editing"},'
    '{"type":"tool_use","id":"tool-1","name":"Write","input":{}}]}}\n'
    '{"type":"user","message":{"content":[{"type":"tool_result",'
    '"tool_use_id":"tool-1","content":"wrote a.txt","is_error":false}]}}\n'
    '{"type":"assistant","message":{"content":[{"type":"text","text":"finished"}]}}\n'
    '{"type":"result","is_error":false,"result":"finished"}\n'
)


def _agent_with_conn(shell: str, conn: _FakeConn) -> ClaudeSDKAgent:
    agent = ClaudeSDKAgent()
    capability = Capability(
        name="shell",
        protocol="ssh/2",
        url="ssh://localhost:22",
        params={"shell": shell},
    )
    agent._ssh = SSHClient(capability, cast("Any", conn))
    agent._shell = shell
    return agent


async def test_exec_on_windows_writes_batch_and_execs_via_cmd() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, _FakeProcess(_STREAM_JSON))
    agent = _agent_with_conn("cmd", conn)

    run = _fake_run()
    await agent._exec(run, prompt="build it", max_steps=5)

    assert conn.ran == ["cmd /c .hud_run.bat"]
    assert all(command.startswith("powershell ") for command in conn.write_commands)
    assert sink[".hud_run.bat"].startswith(b"@echo off\r\n")
    assert sink[".hud_prompt.txt"] == b"build it"
    assert run.trace.status == "completed"
    assert run.trace.content == "finished"


async def test_exec_on_bash_runs_inline_without_batch() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, _FakeProcess(_STREAM_JSON))
    agent = _agent_with_conn("bash", conn)

    run = _fake_run()
    await agent._exec(run, prompt="build it", max_steps=5)

    assert ".hud_run.bat" not in sink
    assert conn.write_commands == ["cat > .hud_prompt.txt"]
    assert len(conn.ran) == 1
    assert "install.sh" in conn.ran[0]
    assert "claude" in conn.ran[0]
    assert run.trace.status == "completed"


async def test_exec_nonzero_exit_with_no_stdout_records_system_error() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, _FakeProcess("", stderr="boom", exit_status=1))
    agent = _agent_with_conn("cmd", conn)

    run = _fake_run()
    await agent._exec(run, prompt="x", max_steps=1)

    assert run.trace.status == "error"
    assert run.trace.extra["exit_status"] == 1
    assert run.steps[0].error == "boom"


async def test_exec_records_claude_turn_before_process_exit() -> None:
    sink: dict[str, bytes] = {}
    process = _FakeProcess(_STREAM_JSON, pause_after=1)
    conn = _FakeConn(sink, process)
    agent = _agent_with_conn("bash", conn)
    run = _fake_run()

    execution = asyncio.create_task(agent._exec(run, prompt="edit it", max_steps=5))
    await process.blocked.wait()

    assert not execution.done()
    assert len(run.steps) == 1
    first = run.steps[0]
    assert isinstance(first, AgentStep)
    assert first.content == "editing"
    assert first.tool_calls[0].id == "tool-1"

    process.release.set()
    await execution

    assert [type(step) for step in run.steps] == [AgentStep, ToolStep, AgentStep]
    tool = cast("ToolStep", run.steps[1])
    assert tool.started_at == first.ended_at
    final = cast("AgentStep", run.steps[2])
    assert final.started_at == tool.ended_at
    assert run.trace.status == "completed"
    assert run.trace.content == "finished"


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

    await agent(
        cast(
            "Any",
            SimpleNamespace(client=Client(), prompt_text="call the tool"),
        )
    )

    assert agent._mcp_servers == {
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
    ssh = SSHClient(shell, cast("Any", object()))
    rfb = object.__new__(RFBClient)

    class Client:
        manifest = SimpleNamespace(bindings=[shell, screen])

        async def open(self, ref: str) -> Any:
            return ssh if ref == "ssh" else rfb

    encoding = WebPScreenshotEncoding(quality=42)
    agent = ClaudeSDKAgent(ClaudeSDKConfig(screenshot_encoding=encoding))
    execute = AsyncMock()
    serve = AsyncMock(return_value=8765)
    monkeypatch.setattr(agent, "_exec", execute)
    monkeypatch.setattr(computer_mcp, "serve_computer_mcp", serve)

    await agent(
        cast(
            "Any",
            SimpleNamespace(client=Client(), prompt_text="use the computer"),
        )
    )

    serve.assert_awaited_once_with(rfb, encoding)
    assert agent._mcp_servers["computer-use"] == {
        "type": "http",
        "url": "http://127.0.0.1:8765/mcp",
    }
