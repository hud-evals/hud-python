"""ClaudeCLIAgent remote-command construction over the workspace SSH.

The agent runs the ``claude`` CLI on the remote workspace. These cover how the
command is assembled per login shell — especially the Windows path, where the
command must ride a batch file invoked via ``cmd /c``. Bare ``.hud_run.bat`` is
rejected by the remote shell (and silently fails under PowerShell), so the
``cmd /c`` prefix is a regression guard for local Windows setups.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from hud.agents.claude.cli.agent import ClaudeCLIAgent, build_remote_invocation

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


class _FakeFile:
    def __init__(self, name: str, sink: dict[str, bytes]) -> None:
        self._name = name
        self._sink = sink

    async def __aenter__(self) -> _FakeFile:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def write(self, data: bytes) -> None:
        self._sink[self._name] = self._sink.get(self._name, b"") + data


class _FakeSFTP:
    def __init__(self, sink: dict[str, bytes]) -> None:
        self._sink = sink

    async def __aenter__(self) -> _FakeSFTP:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def open(self, name: str, mode: str) -> _FakeFile:
        return _FakeFile(name, self._sink)


class _FakeConn:
    def __init__(self, sink: dict[str, bytes], result: Any) -> None:
        self._sink = sink
        self._result = result
        self.ran: list[str] = []

    def start_sftp_client(self) -> _FakeSFTP:
        return _FakeSFTP(self._sink)

    async def run(self, cmd: str, *, check: bool = True) -> Any:
        self.ran.append(cmd)
        return self._result


def _fake_run() -> Any:
    trace = SimpleNamespace(status="", content="", extra={})
    steps: list[Any] = []
    return SimpleNamespace(trace=trace, record=steps.append, steps=steps)


_STREAM_JSON = (
    '{"type":"assistant","message":{"content":[{"type":"text","text":"working"}]}}\n'
    '{"type":"result","is_error":false,"result":"done","session_id":"s",'
    '"duration_ms":5,"num_turns":2,"total_cost_usd":0.01}\n'
)


def _agent_with_conn(shell: str, conn: _FakeConn) -> ClaudeCLIAgent:
    agent = ClaudeCLIAgent()
    agent._ssh = cast("Any", SimpleNamespace(conn=conn))
    agent._shell = shell
    return agent


async def test_exec_on_windows_writes_batch_and_execs_via_cmd() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, SimpleNamespace(stdout=_STREAM_JSON, stderr="", exit_status=0))
    agent = _agent_with_conn("cmd", conn)
    agent.config.max_steps = 5

    run = _fake_run()
    await agent._exec(run, prompt="build it")

    assert conn.ran == ["cmd /c .hud_run.bat"]
    assert sink[".hud_run.bat"].startswith(b"@echo off\r\n")
    assert sink[".hud_prompt.txt"] == b"build it"
    assert run.trace.status == "completed"
    assert "done" in run.trace.content


async def test_exec_on_bash_runs_inline_without_batch() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, SimpleNamespace(stdout=_STREAM_JSON, stderr="", exit_status=0))
    agent = _agent_with_conn("bash", conn)
    agent.config.max_steps = 5

    run = _fake_run()
    await agent._exec(run, prompt="build it")

    assert ".hud_run.bat" not in sink
    assert len(conn.ran) == 1
    assert "install.sh" in conn.ran[0]
    assert "claude" in conn.ran[0]
    assert run.trace.status == "completed"


async def test_exec_nonzero_exit_with_no_stdout_records_system_error() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, SimpleNamespace(stdout="", stderr="boom", exit_status=1))
    agent = _agent_with_conn("cmd", conn)
    agent.config.max_steps = 1

    run = _fake_run()
    await agent._exec(run, prompt="x")

    assert run.trace.status == "error"
    assert run.trace.extra["exit_status"] == 1
    assert run.steps[0].error == "boom"
