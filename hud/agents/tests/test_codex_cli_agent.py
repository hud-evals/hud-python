"""CodexCLIAgent command construction and JSONL trajectory mapping."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from mcp.types import TextContent

from hud.agents.codex import CodexCLIAgent
from hud.agents.codex.agent import codex_command, run_codex
from hud.agents.types import AgentStep, CodexCLIConfig, ToolStep
from hud.capabilities import Capability, SSHClient
from hud.settings import settings
from hud.telemetry.context import set_trace_context


@pytest.fixture(autouse=True)
def _clear_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "openai_api_key", None)


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


class _FakeWriter:
    def __init__(self) -> None:
        self.data = b""
        self.eof = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        pass

    def write_eof(self) -> None:
        self.eof = True


class _FakeProcess:
    def __init__(
        self,
        stdout: str,
        *,
        stderr: str = "",
        returncode: int | None = 0,
        pause_after: int | None = None,
    ) -> None:
        self.stdin = _FakeWriter()
        self.stdout = _FakeReader(stdout, pause_after=pause_after)
        self.stderr = _FakeReader(stderr)
        self.returncode = returncode
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


class _FakeSSH:
    def __init__(self, process: _FakeProcess, *, shell: str = "bash") -> None:
        self.process = process
        self.capability = Capability(
            name="shell",
            protocol="ssh/2",
            url="ssh://localhost:22",
            params={"shell": shell},
        )
        self.commands: list[str] = []

    async def create_process(self, command: str) -> _FakeProcess:
        self.commands.append(command)
        return self.process


def _fake_run() -> Any:
    trace = SimpleNamespace(status=None, content="", extra={})
    steps: list[Any] = []
    return SimpleNamespace(trace=trace, record=steps.append, steps=steps)


_STREAM_JSON = (
    '{"type":"thread.started","thread_id":"thread-1"}\n'
    '{"type":"turn.started"}\n'
    '{"type":"item.started","item":{"id":"cmd-1","type":"command_execution",'
    '"command":"pytest -q","aggregated_output":"","exit_code":null,'
    '"status":"in_progress"}}\n'
    '{"type":"item.completed","item":{"id":"cmd-1","type":"command_execution",'
    '"command":"pytest -q","aggregated_output":"1 passed\\n","exit_code":0,'
    '"status":"completed"}}\n'
    '{"type":"item.completed","item":{"id":"patch-1","type":"file_change",'
    '"changes":[{"path":"calc.py","kind":"update"}],"status":"completed"}}\n'
    '{"type":"item.started","item":{"id":"mcp-1","type":"mcp_tool_call",'
    '"server":"db","tool":"query","arguments":{"sql":"select 42"},'
    '"result":null,"error":null,"status":"in_progress"}}\n'
    '{"type":"item.completed","item":{"id":"mcp-1","type":"mcp_tool_call",'
    '"server":"db","tool":"query","arguments":{"sql":"select 42"},'
    '"result":{"content":[{"type":"text","text":"42"}],'
    '"structured_content":{"answer":42}},"error":null,"status":"completed"}}\n'
    '{"type":"item.completed","item":{"id":"search-1","type":"web_search",'
    '"query":"HUD evals","action":{"type":"search"}}}\n'
    '{"type":"item.completed","item":{"id":"reason-1","type":"reasoning",'
    '"text":"The test now passes."}}\n'
    '{"type":"item.completed","item":{"id":"message-1","type":"agent_message",'
    '"text":"Implemented and verified."}}\n'
    '{"type":"turn.completed","usage":{"input_tokens":20,"cached_input_tokens":5,'
    '"output_tokens":8,"reasoning_output_tokens":3}}\n'
)


def test_command_follows_explicit_gateway_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", "hud-key")
    monkeypatch.setattr(settings, "openai_api_key", "openai-key")

    with set_trace_context("trace-123"):
        gateway = codex_command(CodexCLIConfig(use_hud_gateway=True), "bash")
    provider = codex_command(CodexCLIConfig(use_hud_gateway=False), "bash")

    assert "HUD_API_KEY=hud-key" in gateway
    assert 'model_provider="hud"' in gateway
    assert f'model_providers.hud.base_url="{settings.hud_gateway_url}"' in gateway
    assert "Trace-Id" in gateway
    assert "CODEX_API_KEY=openai-key" in provider
    assert "model_provider" not in provider
    for command in (gateway, provider):
        assert "codex exec" in command
        assert "--json" in command
        assert "--ephemeral" in command
        assert "--sandbox workspace-write" in command
        assert "--model gpt-5.6-sol" in command
        assert command.endswith(" -")


def test_windows_command_encodes_environment_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "key&value's")
    config = CodexCLIConfig(use_hud_gateway=False, sandbox="danger-full-access")
    command = codex_command(config, "powershell")

    script = base64.b64decode(command.rsplit(" ", 1)[1]).decode("utf-16-le")
    assert "$env:CODEX_API_KEY='key&value''s'" in script
    assert "'--sandbox' 'danger-full-access'" in script
    assert "& codex 'exec'" in script
    assert script.endswith(";exit $LASTEXITCODE")


async def test_exec_streams_prompt_and_records_codex_items() -> None:
    process = _FakeProcess(_STREAM_JSON)
    ssh = _FakeSSH(process)
    run = _fake_run()

    await run_codex(
        CodexCLIConfig(),
        run,
        ssh=cast("SSHClient", ssh),
        shell="bash",
        prompt="Fix the failing test",
    )

    assert process.stdin.data == b"Fix the failing test"
    assert process.stdin.eof
    assert [type(step) for step in run.steps] == [
        ToolStep,
        ToolStep,
        ToolStep,
        ToolStep,
        AgentStep,
        AgentStep,
    ]
    command = cast("ToolStep", run.steps[0])
    assert command.call is not None
    assert command.call.name == "shell"
    assert command.call.arguments == {"command": "pytest -q"}
    assert command.result is not None
    assert command.result.isError is False
    output = command.result.content[0]
    assert isinstance(output, TextContent)
    assert output.text == "1 passed\n"
    patch = cast("ToolStep", run.steps[1])
    assert patch.call is not None
    assert patch.call.name == "apply_patch"
    mcp = cast("ToolStep", run.steps[2])
    assert mcp.call is not None
    assert mcp.call.name == "query"
    assert mcp.call.provider_name == "db.query"
    assert mcp.result is not None
    assert mcp.result.structuredContent == {"answer": 42}
    search = cast("ToolStep", run.steps[3])
    assert search.call is not None
    assert search.call.name == "web_search"
    assert cast("AgentStep", run.steps[4]).reasoning == "The test now passes."
    assert cast("AgentStep", run.steps[5]).content == "Implemented and verified."
    assert run.trace.content == "Implemented and verified."
    assert run.trace.extra["codex_thread_id"] == "thread-1"
    assert run.trace.extra["usage"]["cached_input_tokens"] == 5
    assert run.trace.status is None


async def test_exec_records_completed_items_before_process_exit() -> None:
    process = _FakeProcess(_STREAM_JSON, pause_after=4)
    ssh = _FakeSSH(process)
    run = _fake_run()
    execution = asyncio.create_task(
        run_codex(
            CodexCLIConfig(),
            run,
            ssh=cast("SSHClient", ssh),
            shell="bash",
            prompt="Fix it",
        )
    )
    await process.stdout.blocked.wait()

    assert not execution.done()
    assert len(run.steps) == 1
    assert isinstance(run.steps[0], ToolStep)

    process.stdout.release.set()
    await execution


async def test_exec_turn_failure_raises() -> None:
    stream = (
        '{"type":"thread.started","thread_id":"thread-1"}\n'
        '{"type":"turn.started"}\n'
        '{"type":"turn.failed","error":{"message":"model unavailable"}}\n'
    )
    run = _fake_run()

    with pytest.raises(RuntimeError, match="model unavailable"):
        await run_codex(
            CodexCLIConfig(),
            run,
            ssh=cast("SSHClient", _FakeSSH(_FakeProcess(stream))),
            shell="bash",
            prompt="Fix it",
        )


async def test_exec_nonzero_exit_raises_stderr() -> None:
    run = _fake_run()

    with pytest.raises(RuntimeError, match="authentication failed"):
        await run_codex(
            CodexCLIConfig(),
            run,
            ssh=cast(
                "SSHClient",
                _FakeSSH(_FakeProcess("", stderr="authentication failed", returncode=1)),
            ),
            shell="bash",
            prompt="Fix it",
        )

    assert run.trace.extra["returncode"] == 1


async def test_exec_nonzero_exit_prefers_structured_error() -> None:
    run = _fake_run()
    stream = '{"type":"error","message":"gateway rejected streaming"}\n'

    with pytest.raises(RuntimeError, match="gateway rejected streaming"):
        await run_codex(
            CodexCLIConfig(),
            run,
            ssh=cast(
                "SSHClient",
                _FakeSSH(_FakeProcess(stream, stderr="noisy warning", returncode=1)),
            ),
            shell="bash",
            prompt="Fix it",
        )

    assert "stderr" not in run.trace.extra


async def test_exec_closes_process_when_cancelled() -> None:
    process = _FakeProcess(_STREAM_JSON, pause_after=0)
    execution = asyncio.create_task(
        run_codex(
            CodexCLIConfig(),
            _fake_run(),
            ssh=cast("SSHClient", _FakeSSH(process)),
            shell="bash",
            prompt="Fix it",
        )
    )
    await process.stdout.blocked.wait()
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution

    assert process.closed


async def test_agent_opens_ssh_and_uses_workspace_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    ssh = _FakeSSH(_FakeProcess(_STREAM_JSON), shell="powershell")

    class Client:
        async def open(self, ref: str) -> _FakeSSH:
            assert ref == "ssh"
            return ssh

    agent = CodexCLIAgent()
    execute = AsyncMock()
    monkeypatch.setattr("hud.agents.codex.agent.run_codex", execute)
    run = SimpleNamespace(client=Client(), prompt_text="Fix it")

    await agent(cast("Any", run))

    execute.assert_awaited_once_with(
        agent.config,
        run,
        ssh=ssh,
        shell="powershell",
        prompt="Fix it",
    )
