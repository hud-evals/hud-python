"""CLIAgent remote-command construction over workspace SSH."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from hud.agents import create_agent
from hud.agents.cli import CodexAgent, GrokBuildAgent, OpenCodeAgent, Terminus2Agent
from hud.agents.cli.agent import (
    TERMINUS2_RESULT_PATH,
    TERMINUS2_SCRIPT_PATH,
    CLIAgent,
    build_remote_invocation,
)
from hud.agents.types import (
    AiderConfig,
    CLIConfig,
    CodexConfig,
    GrokBuildConfig,
    MiniSweAgentConfig,
    OpenCodeConfig,
    Terminus2Config,
)
from hud.settings import settings

if TYPE_CHECKING:
    import pytest


class _FakeFile:
    def __init__(self, name: str, mode: str, sink: dict[str, bytes]) -> None:
        self._name = name
        self._mode = mode
        self._sink = sink

    async def __aenter__(self) -> _FakeFile:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def write(self, data: bytes) -> None:
        self._sink[self._name] = self._sink.get(self._name, b"") + data

    async def read(self) -> bytes:
        if "r" not in self._mode:
            raise OSError("not opened for reading")
        return self._sink[self._name]


class _FakeSFTP:
    def __init__(self, sink: dict[str, bytes]) -> None:
        self._sink = sink

    async def __aenter__(self) -> _FakeSFTP:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def open(self, name: str, mode: str) -> _FakeFile:
        if "r" in mode and name not in self._sink:
            raise OSError(name)
        return _FakeFile(name, mode, self._sink)


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


def _agent_with_conn(config: CLIConfig, shell: str, conn: _FakeConn) -> CLIAgent:
    agent = CLIAgent(config)
    agent._ssh = cast("Any", SimpleNamespace(conn=conn))
    agent._shell = shell
    return agent


def test_windows_shell_runs_batch_file_via_cmd() -> None:
    inv = build_remote_invocation("powershell", "mini --task x")

    assert inv.command == "cmd /c .hud_run.bat"
    assert inv.script_name == ".hud_run.bat"
    assert inv.script_body == "@echo off\r\nmini --task x\r\n"


def test_presets_build_documented_non_interactive_commands() -> None:
    opencode = CLIAgent(OpenCodeConfig())._build_cli_command(
        prompt="fix the repo",
        mcp_config_path=None,
    )
    codex = CLIAgent(CodexConfig())._build_cli_command(
        prompt="fix the repo",
        mcp_config_path=None,
    )
    aider = CLIAgent(AiderConfig())._build_cli_command(
        prompt="fix the repo",
        mcp_config_path=None,
    )
    grok = CLIAgent(GrokBuildConfig())._build_cli_command(
        prompt="fix the repo",
        mcp_config_path=None,
    )
    mini = CLIAgent(MiniSweAgentConfig())._build_cli_command(
        prompt="fix the repo",
        mcp_config_path=None,
    )
    terminus = CLIAgent(Terminus2Config())._build_cli_command(
        prompt="fix the repo",
        mcp_config_path=None,
    )

    assert "opencode run" in opencode
    assert "--dangerously-skip-permissions" in opencode
    assert "'fix the repo'" in opencode
    assert "codex exec" in codex
    assert "--sandbox workspace-write" in codex
    assert "--skip-git-repo-check" in codex
    assert "'fix the repo'" in codex
    assert "aider --model" in aider
    assert "--message-file .hud_prompt.txt" in aider
    assert "grok -p 'fix the repo'" in grok
    assert "-m grok-build-0.1" in grok
    assert "--always-approve" in grok
    assert "--max-turns" not in grok
    assert "--output-format" not in grok
    assert "mini --model" in mini
    assert "--task 'fix the repo'" in mini
    assert "--yolo" in mini
    assert "uv --no-config run --no-project --quiet" in terminus
    assert "--with harbor==0.6.6" in terminus
    assert "python .hud_terminus2.py" in terminus


def test_create_agent_constructs_opencode_with_default_model() -> None:
    agent = create_agent("opencode")

    assert isinstance(agent, OpenCodeAgent)
    assert agent.config.model == "openai/gpt-5.4"


def test_create_agent_constructs_codex_with_default_model() -> None:
    agent = create_agent("codex")

    assert isinstance(agent, CodexAgent)
    assert agent.config.model == "gpt-5.4"


def test_create_agent_constructs_grok_build_with_default_model() -> None:
    agent = create_agent("grok_build")

    assert isinstance(agent, GrokBuildAgent)
    assert agent.config.model == "grok-build-0.1"


def test_grok_build_uses_isolated_home_with_xai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "xai_api_key", "xai-test")

    env = GrokBuildAgent()._build_env_vars()

    assert env["XAI_API_KEY"] == "xai-test"
    assert env["HOME"] == "/tmp/hud-grok-home"


def test_create_agent_constructs_terminus_2_with_default_model() -> None:
    agent = create_agent("terminus_2")

    assert isinstance(agent, Terminus2Agent)
    assert agent.config.model == "openai/gpt-5.4"
    assert agent.config.result_file == TERMINUS2_RESULT_PATH


async def test_exec_writes_prompt_runs_command_and_records_stdout() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, SimpleNamespace(stdout="done\n", stderr="", exit_status=0))
    config = CLIConfig(
        model="m",
        model_name="TestCLI",
        command="agent",
        args=["--message-file", "{prompt_file}"],
        use_hud_gateway=False,
    )
    agent = _agent_with_conn(config, "bash", conn)
    run = _fake_run()

    await agent._exec(run, prompt="build it")

    assert sink[".hud_prompt.txt"] == b"build it"
    assert len(conn.ran) == 1
    assert conn.ran[0].endswith("agent --message-file .hud_prompt.txt")
    assert run.trace.status == "completed"
    assert run.trace.content == "done"
    assert run.steps[0].content == "done"


async def test_terminus_2_writes_runner_script_before_command() -> None:
    sink: dict[str, bytes] = {TERMINUS2_RESULT_PATH: b"terminus done\n"}
    conn = _FakeConn(sink, SimpleNamespace(stdout="", stderr="", exit_status=0))
    agent = Terminus2Agent(Terminus2Config(use_hud_gateway=False))
    agent._ssh = cast("Any", SimpleNamespace(conn=conn))
    agent._shell = "bash"
    run = _fake_run()

    await agent._exec(run, prompt="solve it")

    assert TERMINUS2_SCRIPT_PATH in sink
    assert b"from harbor.agents.terminus_2 import Terminus2" in sink[TERMINUS2_SCRIPT_PATH]
    assert run.trace.content == "terminus done"


async def test_exec_uses_result_file_when_configured() -> None:
    sink: dict[str, bytes] = {"answer.txt": b"file answer\n"}
    conn = _FakeConn(sink, SimpleNamespace(stdout="stdout answer\n", stderr="", exit_status=0))
    config = CLIConfig(
        model="m",
        command="agent",
        result_file="answer.txt",
        use_hud_gateway=False,
    )
    agent = _agent_with_conn(config, "bash", conn)
    run = _fake_run()

    await agent._exec(run, prompt="x")

    assert run.trace.content == "file answer"


async def test_exec_nonzero_records_system_error() -> None:
    sink: dict[str, bytes] = {}
    conn = _FakeConn(sink, SimpleNamespace(stdout="", stderr="boom", exit_status=2))
    config = CLIConfig(
        model="m",
        model_name="TestCLI",
        command="agent",
        use_hud_gateway=False,
    )
    agent = _agent_with_conn(config, "bash", conn)
    run = _fake_run()

    await agent._exec(run, prompt="x")

    assert run.trace.status == "error"
    assert run.steps[0].error == "boom"
    assert run.steps[1].error == "boom"
