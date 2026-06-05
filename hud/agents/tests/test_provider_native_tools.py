"""Provider native tool adapters: translate a provider tool call into SSH execution.

Each provider exposes its own LLM-facing schema (``to_params``) but executes over a
shared ``SSHClient`` (``self.bash`` -> ``conn.run``). These tests inject a fake SSH
client and assert the command translation + result shape, fully offline.
"""

from __future__ import annotations

from typing import Any

import pytest

from hud.agents.claude.tools.coding import ClaudeBashTool, ClaudeTextEditorTool
from hud.agents.gemini.tools.coding import GeminiEditTool, GeminiShellTool
from hud.agents.openai.tools.coding import OpenAIShellTool


class _Completed:
    def __init__(self, *, stdout: str = "", stderr: str = "", exit_status: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_status = exit_status


class _FakeOpenFile:
    def __init__(self, store: dict[str, bytes], path: str, mode: str) -> None:
        self._store = store
        self._path = path
        self._mode = mode
        self._written = b""

    async def __aenter__(self) -> _FakeOpenFile:
        return self

    async def __aexit__(self, *_: object) -> bool:
        if "w" in self._mode:
            self._store[self._path] = self._written
        return False

    async def read(self) -> bytes:
        return self._store.get(self._path, b"")

    async def write(self, data: bytes) -> None:
        self._written += data


class _FakeSFTP:
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store

    async def __aenter__(self) -> _FakeSFTP:
        return self

    async def __aexit__(self, *_: object) -> bool:
        return False

    def open(self, path: str, mode: str) -> _FakeOpenFile:
        return _FakeOpenFile(self._store, path, mode)


class _Conn:
    def __init__(self, completed: _Completed, store: dict[str, bytes]) -> None:
        self._completed = completed
        self._store = store
        self.commands: list[str] = []

    async def run(self, command: str, check: bool = False) -> _Completed:
        self.commands.append(command)
        return self._completed

    def start_sftp_client(self) -> _FakeSFTP:
        return _FakeSFTP(self._store)


class _FakeSSH:
    """Duck-typed ``SSHClient``: ``conn.run`` (bash) + ``conn.start_sftp_client`` (files)."""

    def __init__(
        self,
        *,
        stdout: str = "ok",
        exit_status: int = 0,
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.files: dict[str, bytes] = files or {}
        self.conn = _Conn(_Completed(stdout=stdout, exit_status=exit_status), self.files)


def _commands(tool: Any) -> list[str]:
    return tool.client.conn.commands


# ─── OpenAI shell ─────────────────────────────────────────────────────


async def test_openai_shell_wraps_command_with_timeout() -> None:
    tool = OpenAIShellTool(spec=OpenAIShellTool.default_spec("gpt-5.4"), client=_FakeSSH())

    result = await tool.execute({"commands": ["pwd"], "timeout_ms": 2500})

    assert _commands(tool) == ["timeout 2 pwd"]
    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["provider_tool"] == "shell"
    assert len(result.structuredContent["output"]) == 1


async def test_openai_shell_runs_each_command_without_timeout() -> None:
    tool = OpenAIShellTool(spec=OpenAIShellTool.default_spec("gpt-5.4"), client=_FakeSSH())

    await tool.execute({"commands": ["echo a", "echo b"]})

    assert _commands(tool) == ["echo a", "echo b"]


async def test_openai_shell_rejects_non_list_commands_without_running() -> None:
    tool = OpenAIShellTool(spec=OpenAIShellTool.default_spec("gpt-5.4"), client=_FakeSSH())

    result = await tool.execute({"commands": 123})

    assert result.isError is True
    assert _commands(tool) == []


def test_openai_shell_to_params_is_shell_type() -> None:
    tool = OpenAIShellTool(spec=OpenAIShellTool.default_spec("gpt-5.4"), client=_FakeSSH())
    assert tool.to_params()["type"] == "shell"


# ─── Gemini shell ─────────────────────────────────────────────────────


async def test_gemini_shell_scopes_command_to_quoted_directory() -> None:
    tool = GeminiShellTool(spec=GeminiShellTool.default_spec("gemini"), client=_FakeSSH())

    await tool.execute({"command": "ls -la", "dir_path": "/tmp/my dir"})

    assert _commands(tool) == ["cd '/tmp/my dir' && ls -la"]


async def test_gemini_shell_runs_bare_command() -> None:
    tool = GeminiShellTool(spec=GeminiShellTool.default_spec("gemini"), client=_FakeSSH())

    await tool.execute({"command": "ls"})

    assert _commands(tool) == ["ls"]


async def test_gemini_shell_requires_command() -> None:
    tool = GeminiShellTool(spec=GeminiShellTool.default_spec("gemini"), client=_FakeSSH())

    with pytest.raises(ValueError, match="command is required"):
        await tool.execute({"command": ""})


# ─── Claude bash ──────────────────────────────────────────────────────


async def test_claude_bash_runs_command() -> None:
    tool = ClaudeBashTool(spec=ClaudeBashTool.default_spec("claude-sonnet-4-6"), client=_FakeSSH())

    await tool.execute({"command": "echo hi"})

    assert _commands(tool) == ["echo hi"]


async def test_claude_bash_restart_is_a_noop() -> None:
    tool = ClaudeBashTool(spec=ClaudeBashTool.default_spec("claude-sonnet-4-6"), client=_FakeSSH())

    result = await tool.execute({"restart": True})

    assert result.isError is False
    assert _commands(tool) == []  # restart never touches the shell


async def test_claude_bash_requires_command() -> None:
    tool = ClaudeBashTool(spec=ClaudeBashTool.default_spec("claude-sonnet-4-6"), client=_FakeSSH())

    result = await tool.execute({})

    assert result.isError is True
    assert _commands(tool) == []


def test_claude_bash_to_params_carries_native_schema() -> None:
    tool = ClaudeBashTool(spec=ClaudeBashTool.default_spec("claude-sonnet-4-6"), client=_FakeSSH())
    params = tool.to_params()
    assert params == {"type": "bash_20250124", "name": "bash"}


# ─── editor tools over SFTP ───────────────────────────────────────────


async def test_claude_text_editor_creates_file() -> None:
    ssh = _FakeSSH()
    tool = ClaudeTextEditorTool(spec=ClaudeTextEditorTool.default_spec("claude"), client=ssh)

    result = await tool.execute({"command": "create", "path": "/f.txt", "file_text": "hello"})

    assert result.isError is False
    assert ssh.files["/f.txt"] == b"hello"


async def test_claude_text_editor_str_replace_rewrites_file() -> None:
    ssh = _FakeSSH(files={"/f.txt": b"hello old world"})
    tool = ClaudeTextEditorTool(spec=ClaudeTextEditorTool.default_spec("claude"), client=ssh)

    result = await tool.execute(
        {"command": "str_replace", "path": "/f.txt", "old_str": "old", "new_str": "new"},
    )

    assert result.isError is False
    assert ssh.files["/f.txt"] == b"hello new world"


async def test_claude_text_editor_str_replace_errors_when_not_unique() -> None:
    ssh = _FakeSSH(files={"/f.txt": b"a a a"})
    tool = ClaudeTextEditorTool(spec=ClaudeTextEditorTool.default_spec("claude"), client=ssh)

    result = await tool.execute(
        {"command": "str_replace", "path": "/f.txt", "old_str": "a", "new_str": "b"},
    )

    assert result.isError is True  # ambiguous match must not write
    assert ssh.files["/f.txt"] == b"a a a"


async def test_gemini_edit_creates_file_when_old_string_empty() -> None:
    ssh = _FakeSSH()
    tool = GeminiEditTool(spec=GeminiEditTool.default_spec("gemini"), client=ssh)

    await tool.execute({"file_path": "/n.txt", "old_string": "", "new_string": "fresh"})

    assert ssh.files["/n.txt"] == b"fresh"
