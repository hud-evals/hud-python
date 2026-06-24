"""Provider native tool adapters: translate a provider tool call into SSH execution.

Each provider exposes its own LLM-facing schema (``to_params``) but executes over a
shared ``SSHClient`` (``self.bash`` -> ``conn.run``). These tests inject a fake SSH
client and assert the command translation + result shape, fully offline.
"""

from __future__ import annotations

import shlex
from typing import Any, cast

import pytest

from hud.agents.claude.tools.coding import ClaudeBashTool, ClaudeTextEditorTool
from hud.agents.gemini.tools.coding import GeminiEditTool, GeminiShellTool
from hud.agents.openai.tools.coding import OpenAIShellTool
from hud.agents.openai_compatible.agent import OpenAIChatAgent
from hud.agents.openai_compatible.tools import BashTool, EditTool, ReadTool, WriteTool
from hud.agents.tools.base import result_text
from hud.agents.types import OpenAIChatConfig
from hud.capabilities import Capability, SSHClient


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

    async def listdir(self, path: str) -> list[str]:
        prefix = path.rstrip("/")
        if not prefix:
            prefix = "/"
        if prefix != "/":
            prefix += "/"
        names: set[str] = set()
        for file_path in self._store:
            if not file_path.startswith(prefix):
                continue
            rest = file_path[len(prefix) :]
            if rest:
                names.add(rest.split("/", 1)[0])
        return sorted(names)


class _Conn:
    def __init__(self, completed: _Completed, store: dict[str, bytes]) -> None:
        self._completed = completed
        self._store = store
        self.commands: list[str] = []

    async def run(self, command: str, check: bool = False) -> _Completed:
        self.commands.append(command)
        parts = shlex.split(command)
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

    def start_sftp_client(self) -> _FakeSFTP:
        return _FakeSFTP(self._store)


class _FakeSSH(SSHClient):
    """Duck-typed ``SSHClient``: ``conn.run`` (bash) + ``conn.start_sftp_client`` (files)."""

    def __init__(
        self,
        *,
        stdout: str = "ok",
        exit_status: int = 0,
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.files: dict[str, bytes] = files or {}
        super().__init__(
            Capability(name="shell", protocol="ssh/2", url="ssh://localhost:22"),
            cast("Any", _Conn(_Completed(stdout=stdout, exit_status=exit_status), self.files)),
        )


def _ssh(**kwargs: Any) -> SSHClient:
    return cast("SSHClient", _FakeSSH(**kwargs))


def _commands(tool: Any) -> list[str]:
    return tool.client.conn.commands


class _OpenAIChatAgentForTest(OpenAIChatAgent):
    async def build_tools_for_test(self, ssh: SSHClient) -> tuple[dict[str, Any], list[Any]]:
        return await self._build_tools({"ssh": ssh})


# ─── OpenAI shell ─────────────────────────────────────────────────────


async def test_openai_shell_wraps_command_with_timeout() -> None:
    tool = OpenAIShellTool(spec=OpenAIShellTool.default_spec("gpt-5.5"), client=_ssh())

    result = await tool.execute({"commands": ["pwd"], "timeout_ms": 2500})

    assert _commands(tool) == ["timeout 2 pwd"]
    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["provider_tool"] == "shell"
    assert len(result.structuredContent["output"]) == 1


async def test_openai_shell_runs_each_command_without_timeout() -> None:
    tool = OpenAIShellTool(spec=OpenAIShellTool.default_spec("gpt-5.5"), client=_ssh())

    await tool.execute({"commands": ["echo a", "echo b"]})

    assert _commands(tool) == ["echo a", "echo b"]


async def test_openai_shell_rejects_non_list_commands_without_running() -> None:
    tool = OpenAIShellTool(spec=OpenAIShellTool.default_spec("gpt-5.5"), client=_ssh())

    result = await tool.execute({"commands": 123})

    assert result.isError is True
    assert _commands(tool) == []


def test_openai_shell_to_params_is_shell_type() -> None:
    tool = OpenAIShellTool(spec=OpenAIShellTool.default_spec("gpt-5.5"), client=_ssh())
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


async def test_openai_compatible_write_stores_file_via_workspace_sftp() -> None:
    ssh = _FakeSSH()
    tool = WriteTool(spec=WriteTool.default_spec("qwen"), client=cast("SSHClient", ssh))

    result = await tool.execute({"filePath": "/REPORT.md", "content": "done"})

    assert result.isError is False
    assert ssh.files["/REPORT.md"] == b"done"


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


# ─── editor tools over SFTP ───────────────────────────────────────────


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
