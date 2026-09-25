"""CodexCLIAgent command construction and JSONL trajectory mapping."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from mcp.types import TextContent

from hud.agents.cli import resolve_executable
from hud.agents.codex import CodexCLIAgent
from hud.agents.tests.cli_fakes import FakeClient as _FakeClient
from hud.agents.tests.cli_fakes import FakeProcess as _FakeProcess
from hud.agents.tests.cli_fakes import fake_run as _fake_run
from hud.agents.types import AgentStep, CodexCLIConfig, ToolStep
from hud.capabilities import Capability, Connection
from hud.eval.runtime import RuntimeConfig, RuntimeResources
from hud.settings import settings
from hud.telemetry.context import set_trace_context


@pytest.fixture(autouse=True)
def _clear_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(
        "hud.agents.codex.agent.resolve_executable",
        AsyncMock(return_value="codex"),
    )


class _FakeSSH:
    def __init__(
        self, process: _FakeProcess, *, shell: str = "bash", executable_head: bytes = b"\x7fE"
    ) -> None:
        self.process = process
        self.executable_head = executable_head
        self.capability = Capability(
            name="shell",
            protocol="ssh/2",
            url="ssh://localhost:22",
            params={"shell": shell},
        )
        self.commands: list[str] = []

    async def run(self, command: str, **_: Any) -> SimpleNamespace:
        assert command.startswith("head -c 2 -- ")
        return SimpleNamespace(returncode=0, stdout=self.executable_head)

    async def create_process(
        self, command: str, *, connections: tuple[Connection, ...] = ()
    ) -> _FakeProcess:
        self.commands.append(command)
        return self.process


async def _sent_command(
    config: CodexCLIConfig, shell: str = "bash", connection: Connection | None = None
) -> str:
    ssh = _FakeSSH(_FakeProcess(_STREAM_JSON), shell=shell)
    connections = {"inference": connection} if connection is not None else None
    await CodexCLIAgent(config)(_fake_run(_FakeClient(ssh), "Fix it", connections))
    (command,) = ssh.commands
    return command


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


async def test_command_follows_explicit_gateway_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", "hud-key")
    monkeypatch.setattr(settings, "openai_api_key", "openai-key")

    with set_trace_context("trace-123"):
        gateway = await _sent_command(CodexCLIConfig(gateway=True))
    provider = await _sent_command(CodexCLIConfig(gateway=False))

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
        assert "--ignore-user-config" not in command
        assert "mktemp -d" in command
        assert 'export CODEX_HOME="$codex_home"' in command
        assert "--sandbox workspace-write" in command
        assert "--model gpt-5.6-sol" in command
        assert command.endswith(" -")


@pytest.mark.parametrize("sandbox", ["read-only", "workspace-write", "danger-full-access"])
async def test_command_uses_process_bound_connection_without_its_credential(
    monkeypatch: pytest.MonkeyPatch, sandbox: str
) -> None:
    monkeypatch.setattr(settings, "api_key", "hud-key")
    connection = Connection(
        name="inference",
        capability="ssh",
        url="https://inference.hud.so",
        headers={"Authorization": "Bearer scoped-runtime-token"},
    )

    with set_trace_context("trace-123"):
        command = await _sent_command(
            CodexCLIConfig.model_validate({"sandbox": sandbox}), connection=connection
        )

    assert "scoped-runtime-token" not in command
    assert "hud-key" not in command
    assert "HUD_CONNECTION_CREDENTIAL=hud-process-bound" in command
    assert 'model_providers.hud.env_key="HUD_CONNECTION_CREDENTIAL"' in command
    assert f'model_providers.hud.base_url="{connection.client_url}"' in command
    assert "Trace-Id" not in command
    assert "exec env" in command
    assert "--sandbox danger-full-access" in command


async def test_process_bound_connection_rejects_a_launcher_script_before_starting() -> None:
    connection = Connection(
        name="inference",
        capability="ssh",
        url="https://inference.hud.so",
        headers={"Authorization": "Bearer scoped-runtime-token"},
    )
    ssh = _FakeSSH(_FakeProcess(_STREAM_JSON), executable_head=b"#!")

    with pytest.raises(RuntimeError, match="launcher script"):
        await CodexCLIAgent()(_fake_run(_FakeClient(ssh), "Fix it", {"inference": connection}))

    assert ssh.commands == []


@pytest.mark.parametrize("shell", ["bash", "powershell"])
async def test_command_preserves_ambient_codex_login_without_explicit_credentials(
    shell: str,
) -> None:
    command = await _sent_command(CodexCLIConfig(), shell)
    script = (
        base64.b64decode(command.rsplit(" ", 1)[1]).decode("utf-16-le")
        if shell == "powershell"
        else command
    )

    assert "CODEX_HOME" not in script
    assert "CODEX_API_KEY" not in script
    assert "HUD_API_KEY" not in script
    assert "mktemp" not in script
    assert "codex exec" in script or "& 'codex' 'exec'" in script


async def test_windows_command_encodes_environment_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "key&value's")
    config = CodexCLIConfig(gateway=False, sandbox="danger-full-access")
    command = await _sent_command(config, "powershell")

    script = base64.b64decode(command.rsplit(" ", 1)[1]).decode("utf-16-le")
    assert "$env:CODEX_API_KEY='key&value''s'" in script
    assert "$env:CODEX_HOME=$codexHome" in script
    assert "[System.Guid]::NewGuid()" in script
    assert "Remove-Item -Recurse -Force $codexHome" in script
    assert "--ignore-user-config" not in script
    assert "'--sandbox' 'danger-full-access'" in script
    assert "& 'codex' 'exec'" in script
    assert script.endswith(";exit $hudExitCode")


async def test_exec_streams_prompt_and_records_codex_items() -> None:
    process = _FakeProcess(_STREAM_JSON)
    run = _fake_run(_FakeClient(_FakeSSH(process)), "Fix the failing test")

    await CodexCLIAgent()(run)

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
    run = _fake_run(_FakeClient(_FakeSSH(process)), "Fix it")
    execution = asyncio.create_task(CodexCLIAgent()(run))
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
    run = _fake_run(_FakeClient(_FakeSSH(_FakeProcess(stream))), "Fix it")

    with pytest.raises(RuntimeError, match="model unavailable"):
        await CodexCLIAgent()(run)


async def test_exec_nonzero_exit_raises_stderr() -> None:
    process = _FakeProcess("", stderr="authentication failed", returncode=1)
    run = _fake_run(_FakeClient(_FakeSSH(process)), "Fix it")

    with pytest.raises(RuntimeError, match="authentication failed"):
        await CodexCLIAgent()(run)

    assert run.trace.extra["returncode"] == 1


async def test_exec_nonzero_exit_prefers_structured_error() -> None:
    stream = '{"type":"error","message":"gateway rejected streaming"}\n'
    process = _FakeProcess(stream, stderr="noisy warning", returncode=1)
    run = _fake_run(_FakeClient(_FakeSSH(process)), "Fix it")

    with pytest.raises(RuntimeError, match="gateway rejected streaming"):
        await CodexCLIAgent()(run)

    assert "stderr" not in run.trace.extra


async def test_exec_closes_process_when_cancelled() -> None:
    process = _FakeProcess(_STREAM_JSON, pause_after=0)
    execution = asyncio.create_task(
        CodexCLIAgent()(_fake_run(_FakeClient(_FakeSSH(process)), "Fix it"))
    )
    await process.stdout.blocked.wait()
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution

    assert process.closed


async def test_agent_runs_in_the_workspace_shell_with_the_run_prompt() -> None:
    process = _FakeProcess(_STREAM_JSON)
    ssh = _FakeSSH(process, shell="powershell")

    await CodexCLIAgent()(_fake_run(_FakeClient(ssh), "Fix it"))

    (command,) = ssh.commands
    script = base64.b64decode(command.rsplit(" ", 1)[1]).decode("utf-16-le")
    assert "& 'codex' 'exec'" in script
    assert process.stdin.data == b"Fix it"


async def test_executable_resolution_prefers_matching_managed_bundle() -> None:
    ssh = SimpleNamespace(
        capability=Capability.ssh(url="ssh://localhost:22", host_pubkey="key", shell="bash"),
        run=AsyncMock(
            side_effect=[
                SimpleNamespace(returncode=0, stdout=b"Linux\nx86_64\ngnu\n"),
                SimpleNamespace(returncode=0, stdout=b""),
            ]
        ),
    )

    executable = await resolve_executable(
        cast("Any", ssh),
        "codex",
        {"linux-x64": "/usr/local/lib/agents/codex/bin/codex"},
        RuntimeConfig(resources=RuntimeResources(os="linux")),
    )

    assert executable == "/usr/local/lib/agents/codex/bin/codex"
    assert ssh.run.await_count == 2


@pytest.mark.parametrize(("reported", "platform"), [("X64", "x64"), ("Arm64", "arm64")])
async def test_executable_resolution_understands_windows_architectures(
    reported: str, platform: str
) -> None:
    ssh = SimpleNamespace(
        capability=Capability.ssh(url="ssh://localhost:22", host_pubkey="key", shell="powershell"),
        run=AsyncMock(
            side_effect=[
                SimpleNamespace(returncode=0, stdout=f"{reported}\r\n".encode()),
                SimpleNamespace(returncode=0, stdout=b"C:\\bin\\codex.exe\r\n"),
            ]
        ),
    )

    executable = await resolve_executable(
        cast("Any", ssh),
        "codex",
        {f"linux-{platform}": "/usr/local/lib/agents/codex/bin/codex"},
        RuntimeConfig(resources=RuntimeResources(os="windows")),
    )

    assert executable == "C:\\bin\\codex.exe"
    assert ssh.run.await_args_list[1].args[0] == "where.exe codex"


async def test_executable_resolution_rejects_runtime_os_mismatch() -> None:
    ssh = SimpleNamespace(
        capability=Capability.ssh(url="ssh://localhost:22", host_pubkey="key", shell="bash"),
        run=AsyncMock(return_value=SimpleNamespace(returncode=0, stdout=b"Linux\nx86_64\ngnu\n")),
    )

    with pytest.raises(RuntimeError, match=r"requested 'windows'.*reports 'linux'"):
        await resolve_executable(
            cast("Any", ssh),
            "codex",
            {},
            RuntimeConfig(resources=RuntimeResources(os="windows")),
        )
