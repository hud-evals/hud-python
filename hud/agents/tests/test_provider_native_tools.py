"""Provider native tool adapters: translate a provider tool call into SSH execution.

Each provider exposes its own LLM-facing schema (``to_params``) but executes over a
shared ``SSHClient`` (``self.bash`` -> ``conn.run``). These tests inject a fake SSH
client and assert the command translation + result shape, fully offline.
"""

from __future__ import annotations

import shlex
from types import SimpleNamespace
from typing import Any, cast

import asyncssh
import mcp.types as mcp_types
import pytest

from hud.agents.claude.tools.coding import ClaudeBashTool, ClaudeTextEditorTool
from hud.agents.gemini.tools.coding import GeminiEditTool, GeminiShellTool
from hud.agents.openai.agent import OpenAIAgent
from hud.agents.openai.tools.coding import OPENAI_SHELL_SPEC, OpenAIShellTool
from hud.agents.openai_compatible.agent import OpenAIChatAgent
from hud.agents.openai_compatible.tools import BashTool, EditTool, ReadTool, WriteTool
from hud.agents.tool_agent import RunState
from hud.agents.tools.base import result_text
from hud.agents.tools.ssh import bound_shell_output
from hud.agents.types import OpenAIChatConfig, OpenAIConfig
from hud.capabilities import Capability, RFBClient, SSHClient
from hud.types import MCPToolCall


class _Completed:
    def __init__(self, *, stdout: str = "", stderr: str = "", exit_status: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_status = exit_status
        self.returncode = exit_status


class _Conn:
    def __init__(self, completed: _Completed, store: dict[str, bytes]) -> None:
        self._completed = completed
        self._store = store
        self.commands: list[str] = []

    def is_closed(self) -> bool:
        return False

    async def run(
        self,
        command: str,
        *,
        input: str | None = None,
        check: bool = False,
        encoding: str | None = "utf-8",
    ) -> _Completed:
        self.commands.append(command)
        parts = shlex.split(command)
        if len(parts) == 3 and parts[:2] == ["cat", "--"]:
            if parts[2] not in self._store:
                if check:
                    raise asyncssh.ProcessError(
                        env=None,
                        command=command,
                        subsystem=None,
                        exit_status=1,
                        exit_signal=None,
                        returncode=1,
                        stdout="",
                        stderr=f"cat: {parts[2]}: No such file or directory",
                    )
                return _Completed(
                    stderr=f"cat: {parts[2]}: No such file or directory", exit_status=1
                )
            return _Completed(stdout=self._store[parts[2]].decode())
        if len(parts) == 3 and parts[:2] == ["cat", ">"]:
            assert input is not None
            self._store[parts[2]] = input.encode()
            return _Completed()
        if len(parts) == 4 and parts[:3] == ["ls", "-1A", "--"]:
            prefix = parts[3].rstrip("/")
            prefix = "/" if not prefix else prefix + "/"
            names = {
                rest.split("/", 1)[0]
                for file_path in self._store
                if file_path.startswith(prefix) and (rest := file_path[len(prefix) :])
            }
            return _Completed(stdout="\n".join(sorted(names)))
        if len(parts) == 3 and parts[:2] in (["test", "-d"], ["test", "-e"]):
            path = parts[2]
            exists = path in self._store or any(
                file_path.startswith(path.rstrip("/") + "/") for file_path in self._store
            )
            if parts[1] == "-d":
                exists = any(
                    file_path.startswith(path.rstrip("/") + "/") for file_path in self._store
                )
            return _Completed(exit_status=0 if exists else 1)
        if len(parts) >= 3 and parts[:2] == ["mkdir", "-p"]:
            return _Completed(exit_status=0)
        return self._completed

    async def create_process(self, command: str, **kwargs: Any) -> _Process:
        return _Process(await self.run(command, **kwargs))


class _Process:
    def __init__(self, completed: _Completed) -> None:
        self.completed = completed
        self.closed = False

    async def wait(self, *, check: bool, timeout: float) -> _Completed:  # noqa: ASYNC109
        del timeout
        if check and self.completed.exit_status:
            raise asyncssh.ProcessError(
                env=None,
                command=None,
                subsystem=None,
                exit_status=self.completed.exit_status,
                exit_signal=None,
                returncode=self.completed.exit_status,
                stdout=self.completed.stdout,
                stderr=self.completed.stderr,
            )
        return self.completed

    def close(self) -> None:
        self.closed = True

    def terminate(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


class _FakeSSH(SSHClient):
    """SSH client with an in-memory exec-channel filesystem."""

    def __init__(
        self,
        *,
        stdout: str = "ok",
        stderr: str = "",
        exit_status: int = 0,
        files: dict[str, bytes] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.files: dict[str, bytes] = files or {}
        params = {"cwd": cwd} if cwd else {}
        super().__init__(
            Capability(name="shell", protocol="ssh/2", url="ssh://localhost:22", params=params),
            cast(
                "Any",
                _Conn(
                    _Completed(stdout=stdout, stderr=stderr, exit_status=exit_status),
                    self.files,
                ),
            ),
        )


def _ssh(**kwargs: Any) -> SSHClient:
    return cast("SSHClient", _FakeSSH(**kwargs))


def _commands(tool: Any) -> list[str]:
    return tool.client.conn.commands


class _OpenAIChatAgentForTest(OpenAIChatAgent):
    async def build_tools_for_test(self, ssh: SSHClient) -> tuple[dict[str, Any], list[Any]]:
        return await self._build_tools({"ssh": ssh})


class _OpenAIAgentForTest(OpenAIAgent):
    async def build_native_tools_for_test(
        self,
        ssh: SSHClient,
        rfb: RFBClient,
    ) -> tuple[dict[str, Any], list[Any]]:
        return await self._build_tools({"shell": ssh, "computer": rfb})


# ─── OpenAI shell ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "expected_types"),
    [
        ("gpt-4o-mini", []),
        ("gpt-5.4-mini", ["shell", "computer"]),
        ("gpt-5.5", ["shell", "computer"]),
        ("gpt-5.6", ["shell", "computer"]),
        ("gpt-5.6-2026-08-07", ["shell", "computer"]),
    ],
)
async def test_openai_native_tools_are_registered_only_for_supported_models(
    model: str,
    expected_types: list[str],
) -> None:
    agent = _OpenAIAgentForTest(OpenAIConfig(model=model, model_client=cast("Any", object())))

    tools, params = await agent.build_native_tools_for_test(
        _ssh(),
        object.__new__(RFBClient),
    )

    assert list(tools) == expected_types
    assert [param["type"] for param in params] == expected_types


@pytest.mark.parametrize(
    ("command", "timeout_ms", "expected"),
    [
        ("pwd", 2500, "timeout 2.5s bash -lc pwd"),
        ("pwd", 500, "timeout 0.5s bash -lc pwd"),
        (
            "echo ready; sleep infinity",
            500,
            "timeout 0.5s bash -lc 'echo ready; sleep infinity'",
        ),
    ],
)
async def test_openai_shell_applies_requested_timeout_to_entire_command(
    command: str,
    timeout_ms: int,
    expected: str,
) -> None:
    tool = OpenAIShellTool(spec=OPENAI_SHELL_SPEC, client=_ssh())

    result = await tool.execute({"commands": [command], "timeout_ms": timeout_ms})

    assert _commands(tool) == [expected]
    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["provider_tool"] == "shell"
    assert len(result.structuredContent["output"]) == 1


async def test_openai_shell_runs_each_command_without_timeout() -> None:
    tool = OpenAIShellTool(spec=OPENAI_SHELL_SPEC, client=_ssh())

    await tool.execute({"commands": ["echo a", "echo b"]})

    assert _commands(tool) == ["echo a", "echo b"]


async def test_openai_shell_has_no_hidden_timeout_across_command_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = OpenAIShellTool(spec=OPENAI_SHELL_SPEC, client=_ssh())
    calls: list[dict[str, Any]] = []

    async def run(*args: object, **kwargs: Any) -> _Completed:
        del args
        calls.append(kwargs)
        return _Completed()

    monkeypatch.setattr(tool.client, "run", run)
    agent = _OpenAIChatAgentForTest(
        OpenAIChatConfig(model="test", model_client=cast("Any", object()))
    )

    await agent._dispatch_call(
        MCPToolCall(name="shell", arguments={"commands": ["echo a", "echo b"]}),
        RunState(tools={"shell": tool}),
    )

    assert len(calls) == 2
    assert all("timeout" not in kwargs for kwargs in calls)


async def test_openai_shell_applies_limit_independently_to_each_command() -> None:
    limit = 80
    tool = OpenAIShellTool(
        spec=OPENAI_SHELL_SPEC,
        client=_ssh(stdout="stdout-start-" + "a" * 100, stderr="b" * 100 + "-stderr-end"),
    )

    result = await tool.execute(
        {"commands": ["first-command", "second-command"], "max_output_length": limit}
    )

    assert result.structuredContent is not None
    outputs = result.structuredContent["output"]
    assert len(outputs) == 2
    assert all(len(output["stdout"]) + len(output["stderr"]) == limit for output in outputs)
    assert all(output["stdout"].startswith("stdout-start-") for output in outputs)
    assert all(output["stderr"].endswith("-stderr-end") for output in outputs)
    assert sum(len(output["stdout"]) + len(output["stderr"]) for output in outputs) == 2 * limit
    text_blocks = [
        block.text for block in result.content if isinstance(block, mcp_types.TextContent)
    ]
    assert len(text_blocks) == 2
    assert all(len(text) == limit for text in text_blocks)
    assert text_blocks == [output["stdout"] + output["stderr"] for output in outputs]


@pytest.mark.parametrize(("limit", "expected"), [(1, "["), (10, "[truncated")])
def test_shared_output_bound_handles_limits_smaller_than_marker(
    limit: int,
    expected: str,
) -> None:
    assert bound_shell_output("x" * 100, "", limit) == (expected, "")


@pytest.mark.parametrize("max_output_length", [0, -1, "20000", 20_000.0, True])
async def test_openai_shell_rejects_invalid_output_limits_without_running(
    max_output_length: Any,
) -> None:
    tool = OpenAIShellTool(spec=OPENAI_SHELL_SPEC, client=_ssh())

    result = await tool.execute(
        {"commands": ["echo should-not-run"], "max_output_length": max_output_length}
    )

    assert result.isError is True
    assert _commands(tool) == []
    assert result.structuredContent is not None
    assert "max_output_length" not in result.structuredContent
    assert result_text(result) == "max_output_length must be a positive integer"


@pytest.mark.parametrize("max_output_length", [None, 20 * 1024 * 1024])
async def test_openai_shell_uses_safe_effective_limit(max_output_length: int | None) -> None:
    tool = OpenAIShellTool(spec=OPENAI_SHELL_SPEC, client=_ssh())
    arguments: dict[str, Any] = {"commands": ["echo ok"]}
    if max_output_length is not None:
        arguments["max_output_length"] = max_output_length

    result = await tool.execute(arguments)

    assert result.structuredContent is not None
    assert result.structuredContent["max_output_length"] == 10 * 1024 * 1024


async def test_openai_shell_rejects_non_list_commands_without_running() -> None:
    tool = OpenAIShellTool(spec=OPENAI_SHELL_SPEC, client=_ssh())

    result = await tool.execute({"commands": 123})

    assert result.isError is True
    assert _commands(tool) == []


def test_openai_shell_to_params_is_shell_type() -> None:
    tool = OpenAIShellTool(spec=OPENAI_SHELL_SPEC, client=_ssh())
    assert tool.to_params()["type"] == "shell"


# ─── OpenAI-compatible OpenCode workspace tools ───────────────────────


async def test_openai_compatible_catalog_matches_opencode_workspace_tools() -> None:
    agent = _OpenAIChatAgentForTest(
        OpenAIChatConfig(model="qwen3.6-plus", model_client=cast("Any", object()))
    )

    tools, params = await agent.build_tools_for_test(_ssh())

    assert list(tools) == ["bash", "read", "glob", "grep", "edit", "write"]
    assert [param["function"]["name"] for param in params] == [
        "bash",
        "read",
        "glob",
        "grep",
        "edit",
        "write",
    ]


async def test_openai_compatible_bash_uses_workdir_and_timeout() -> None:
    tool = BashTool(spec=BashTool.default_spec("qwen"), client=_ssh())

    await tool.execute({"command": "echo hi", "workdir": "/tmp/my dir", "timeout": 2500})

    assert _commands(tool) == ["cd '/tmp/my dir' && timeout 3s bash -lc 'echo hi'"]


async def test_shared_ssh_tool_has_no_hidden_command_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ssh = _ssh()
    seen_kwargs: dict[str, Any] = {}

    async def run(*args: object, **kwargs: Any) -> _Completed:
        del args
        seen_kwargs.update(kwargs)
        return _Completed()

    monkeypatch.setattr(ssh, "run", run)
    tool = BashTool(spec=BashTool.default_spec("qwen"), client=ssh)

    result = await tool.execute({"command": "sleep forever"})

    assert result.isError is False
    assert "timeout" not in seen_kwargs


async def test_shared_ssh_tool_reports_a_signal_returncode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ssh = _ssh()

    async def signaled(*args: object, **kwargs: Any) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            stdout="",
            stderr="terminated",
            exit_status=None,
            returncode=-15,
        )

    monkeypatch.setattr(ssh, "run", signaled)
    tool = BashTool(spec=BashTool.default_spec("qwen"), client=ssh)

    result = await tool.execute({"command": "sleep forever"})

    assert result.isError is True
    assert result_text(result).endswith("(exit -15)")


async def test_shared_ssh_tool_bounds_combined_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hud.agents.tools.ssh.MAX_SHELL_OUTPUT_LENGTH", 80)
    tool = BashTool(
        spec=BashTool.default_spec("qwen"),
        client=_ssh(
            stdout="stdout-start-" + "a" * 100,
            stderr="b" * 100 + "-stderr-end",
        ),
    )

    result = await tool.execute({"command": "noisy"})

    text = result_text(result)
    output = text.split("\n", 1)[1].rsplit("\n(exit 0)", 1)[0]
    stdout, stderr = output.split("\nstderr:\n", 1)
    assert len(stdout) + len(stderr) == 80
    assert stdout.startswith("stdout-start-")
    assert stderr.endswith("-stderr-end")
    assert "[truncated]" in output


async def test_openai_compatible_write_stores_file_via_ssh_exec() -> None:
    ssh = _FakeSSH()
    tool = WriteTool(spec=WriteTool.default_spec("qwen"), client=cast("SSHClient", ssh))

    result = await tool.execute({"filePath": "/REPORT.md", "content": "done"})

    assert result.isError is False
    assert ssh.files["/REPORT.md"] == b"done"


async def test_paths_reach_the_session_verbatim() -> None:
    """A path means what it says in the session's own namespace: file helpers
    and shell commands must never disagree about what a path names, so
    nothing is anchored or rewritten on the way through."""
    ssh = _FakeSSH(cwd="/app", files={"/app/f.txt": b"inside"})
    tool = WriteTool(spec=WriteTool.default_spec("qwen"), client=cast("SSHClient", ssh))

    await tool.execute({"filePath": "/tmp/probe.txt", "content": "done"})

    assert ssh.files["/tmp/probe.txt"] == b"done"
    assert "/app/tmp/probe.txt" not in ssh.files
    assert await cast("SSHClient", ssh).read_text("/app/f.txt") == "inside"


async def test_read_maps_the_directory_predicate_and_listing_together() -> None:
    """`test -d`, listing, and reads must agree on the same path, or
    workspace dirs are misclassified as files."""
    ssh = _FakeSSH(cwd="/workspace", files={"/workspace/pkg/mod.py": b"x = 1\n"})
    tool = ReadTool(spec=ReadTool.default_spec("qwen"), client=cast("SSHClient", ssh))

    result = await tool.execute({"filePath": "/workspace/pkg"})

    text = result_text(result)
    assert "<type>directory</type>" in text
    assert "mod.py" in text
    assert "test -d /workspace/pkg" in cast("Any", ssh).conn.commands


async def test_openai_compatible_edit_rewrites_unique_match() -> None:
    ssh = _FakeSSH(files={"/f.txt": b"hello old world"})
    tool = EditTool(spec=EditTool.default_spec("qwen"), client=cast("SSHClient", ssh))

    result = await tool.execute(
        {"filePath": "/f.txt", "oldString": "old", "newString": "new"},
    )

    assert result.isError is False
    assert ssh.files["/f.txt"] == b"hello new world"


async def test_openai_compatible_edit_rejects_ambiguous_match() -> None:
    ssh = _FakeSSH(files={"/f.txt": b"a a a"})
    tool = EditTool(spec=EditTool.default_spec("qwen"), client=cast("SSHClient", ssh))

    result = await tool.execute(
        {"filePath": "/f.txt", "oldString": "a", "newString": "b"},
    )

    assert result.isError is True
    assert ssh.files["/f.txt"] == b"a a a"


async def test_openai_compatible_read_lists_directories() -> None:
    tool = ReadTool(
        spec=ReadTool.default_spec("qwen"),
        client=_ssh(files={"/work/a.txt": b"a", "/work/nested/b.txt": b"b"}),
    )

    result = await tool.execute({"filePath": "/work"})

    text = result_text(result)
    assert "<type>directory</type>" in text
    assert "a.txt" in text
    assert "nested" in text


async def test_openai_compatible_read_accepts_zero_offset_for_first_page() -> None:
    tool = ReadTool(
        spec=ReadTool.default_spec("qwen"),
        client=_ssh(files={"/f.txt": b"alpha\nbeta\n"}),
    )

    result = await tool.execute({"filePath": "/f.txt", "offset": 0, "limit": 1})

    text = result_text(result)
    assert "1: alpha" in text
    assert "2: beta" not in text


# ─── Gemini shell ─────────────────────────────────────────────────────


async def test_gemini_shell_scopes_command_to_quoted_directory() -> None:
    tool = GeminiShellTool(spec=GeminiShellTool.default_spec("gemini"), client=_ssh())

    await tool.execute({"command": "ls -la", "dir_path": "/tmp/my dir"})

    assert _commands(tool) == ["cd '/tmp/my dir' && ls -la"]


async def test_gemini_shell_runs_bare_command() -> None:
    tool = GeminiShellTool(spec=GeminiShellTool.default_spec("gemini"), client=_ssh())

    await tool.execute({"command": "ls"})

    assert _commands(tool) == ["ls"]


async def test_gemini_shell_requires_command() -> None:
    tool = GeminiShellTool(spec=GeminiShellTool.default_spec("gemini"), client=_ssh())

    with pytest.raises(ValueError, match="command is required"):
        await tool.execute({"command": ""})


# ─── Claude bash ──────────────────────────────────────────────────────


async def test_claude_bash_runs_command() -> None:
    tool = ClaudeBashTool(spec=ClaudeBashTool.default_spec("claude-sonnet-4-6"), client=_ssh())

    await tool.execute({"command": "echo hi"})

    assert _commands(tool) == ["echo hi"]


async def test_claude_bash_restart_is_a_noop() -> None:
    tool = ClaudeBashTool(spec=ClaudeBashTool.default_spec("claude-sonnet-4-6"), client=_ssh())

    result = await tool.execute({"restart": True})

    assert result.isError is False
    assert result_text(result) == (
        "restart is unnecessary; each command runs in a fresh shell session"
    )
    assert _commands(tool) == []  # restart never touches the shell


async def test_claude_bash_requires_command() -> None:
    tool = ClaudeBashTool(spec=ClaudeBashTool.default_spec("claude-sonnet-4-6"), client=_ssh())

    result = await tool.execute({})

    assert result.isError is True
    assert _commands(tool) == []


def test_claude_bash_to_params_carries_native_schema() -> None:
    tool = ClaudeBashTool(spec=ClaudeBashTool.default_spec("claude-sonnet-4-6"), client=_ssh())
    params = tool.to_params()
    assert params == {"type": "bash_20250124", "name": "bash"}


# ─── editor tools over SSH exec ───────────────────────────────────────


async def test_claude_text_editor_creates_file() -> None:
    ssh = _FakeSSH()
    tool = ClaudeTextEditorTool(
        spec=ClaudeTextEditorTool.default_spec("claude"), client=cast("SSHClient", ssh)
    )

    result = await tool.execute({"command": "create", "path": "/f.txt", "file_text": "hello"})

    assert result.isError is False
    assert ssh.files["/f.txt"] == b"hello"


async def test_claude_text_editor_str_replace_rewrites_file() -> None:
    ssh = _FakeSSH(files={"/f.txt": b"hello old world"})
    tool = ClaudeTextEditorTool(
        spec=ClaudeTextEditorTool.default_spec("claude"), client=cast("SSHClient", ssh)
    )

    result = await tool.execute(
        {"command": "str_replace", "path": "/f.txt", "old_str": "old", "new_str": "new"},
    )

    assert result.isError is False
    assert ssh.files["/f.txt"] == b"hello new world"


async def test_claude_text_editor_str_replace_errors_when_not_unique() -> None:
    ssh = _FakeSSH(files={"/f.txt": b"a a a"})
    tool = ClaudeTextEditorTool(
        spec=ClaudeTextEditorTool.default_spec("claude"), client=cast("SSHClient", ssh)
    )

    result = await tool.execute(
        {"command": "str_replace", "path": "/f.txt", "old_str": "a", "new_str": "b"},
    )

    assert result.isError is True  # ambiguous match must not write
    assert ssh.files["/f.txt"] == b"a a a"


async def test_gemini_edit_creates_file_when_old_string_empty() -> None:
    ssh = _FakeSSH()
    tool = GeminiEditTool(spec=GeminiEditTool.default_spec("gemini"), client=cast("SSHClient", ssh))

    await tool.execute({"file_path": "/n.txt", "old_string": "", "new_string": "fresh"})

    assert ssh.files["/n.txt"] == b"fresh"


async def test_reading_a_missing_file_is_a_tool_error_not_a_raised_traceback() -> None:
    """Reading before creating is the first thing an editor tool does; that failure
    must come back as a tool result carrying the shell's message."""
    ssh = _FakeSSH()
    tool = ClaudeTextEditorTool(
        spec=ClaudeTextEditorTool.default_spec("claude"), client=cast("SSHClient", ssh)
    )

    result = await tool.execute({"command": "view", "path": "/nope.txt"})

    assert result.isError is True
    assert "No such file or directory" in result_text(result)
