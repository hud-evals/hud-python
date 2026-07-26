"""Provider native tool adapters: translate a provider tool call into SSH execution.

Each provider exposes its own LLM-facing schema (``to_params``) but executes over a
shared ``SSHClient`` (``self.bash`` -> ``conn.run``). These tests inject a fake SSH
client and assert the command translation + result shape, fully offline.
"""

from __future__ import annotations

import shlex
from typing import Any, cast

import mcp.types as mcp_types
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


class _Conn:
    def __init__(self, completed: _Completed, store: dict[str, bytes]) -> None:
        self._completed = completed
        self._store = store
        self.commands: list[str] = []

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
            return _Completed(stdout=self._store.get(parts[2], b"").decode())
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


async def test_openai_shell_bounds_large_output_in_every_result_field() -> None:
    limit = 20_000
    stderr_prefix = "stderr-start\n"
    stderr_suffix = "\nstderr-end"
    stderr = (
        stderr_prefix + "x" * (10_523_560 - len(stderr_prefix) - len(stderr_suffix)) + stderr_suffix
    )
    tool = OpenAIShellTool(
        spec=OpenAIShellTool.default_spec("gpt-5.5"),
        client=_ssh(stderr=stderr, exit_status=1),
    )

    result = await tool.execute({"commands": ["noisy-command"], "max_output_length": limit})

    assert result.structuredContent is not None
    assert result.structuredContent["max_output_length"] == limit
    output = result.structuredContent["output"][0]
    assert len(output["stdout"]) + len(output["stderr"]) == limit
    assert output["stderr"].startswith(stderr_prefix)
    assert output["stderr"].endswith(stderr_suffix)
    assert "[truncated]" in output["stderr"]
    text_blocks = [
        block.text for block in result.content if isinstance(block, mcp_types.TextContent)
    ]
    assert len(text_blocks) == 1
    assert len(text_blocks[0]) == limit
    assert text_blocks[0] == output["stdout"] + output["stderr"]
    assert "[truncated]" in text_blocks[0]


async def test_openai_shell_applies_limit_independently_to_each_command() -> None:
    limit = 80
    tool = OpenAIShellTool(
        spec=OpenAIShellTool.default_spec("gpt-5.5"),
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


@pytest.mark.parametrize(
    ("max_output_length", "expected_output"),
    [(1, "["), (10, "[truncated")],
)
async def test_openai_shell_honors_small_positive_output_limits(
    max_output_length: int,
    expected_output: str,
) -> None:
    tool = OpenAIShellTool(
        spec=OpenAIShellTool.default_spec("gpt-5.5"),
        client=_ssh(stdout="x" * 100),
    )

    result = await tool.execute(
        {"commands": ["noisy-command"], "max_output_length": max_output_length}
    )

    assert _commands(tool) == ["noisy-command"]
    assert result.structuredContent is not None
    assert result.structuredContent["max_output_length"] == max_output_length
    output = result.structuredContent["output"][0]
    assert output["stdout"] == expected_output
    assert output["stderr"] == ""
    assert result_text(result) == expected_output


@pytest.mark.parametrize("max_output_length", [0, -1, "20000", 20_000.0, True])
async def test_openai_shell_rejects_invalid_output_limits_without_running(
    max_output_length: Any,
) -> None:
    tool = OpenAIShellTool(spec=OpenAIShellTool.default_spec("gpt-5.5"), client=_ssh())

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
    tool = OpenAIShellTool(spec=OpenAIShellTool.default_spec("gpt-5.5"), client=_ssh())
    arguments: dict[str, Any] = {"commands": ["echo ok"]}
    if max_output_length is not None:
        arguments["max_output_length"] = max_output_length

    result = await tool.execute(arguments)

    assert result.structuredContent is not None
    assert result.structuredContent["max_output_length"] == 10 * 1024 * 1024


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


async def test_openai_compatible_write_stores_file_via_ssh_exec() -> None:
    ssh = _FakeSSH()
    tool = WriteTool(spec=WriteTool.default_spec("qwen"), client=cast("SSHClient", ssh))

    result = await tool.execute({"filePath": "/REPORT.md", "content": "done"})

    assert result.isError is False
    assert ssh.files["/REPORT.md"] == b"done"


async def test_absolute_paths_anchor_to_the_capability_cwd() -> None:
    """The old SFTP chroot resolved ``/REPORT.md`` against the workspace root;
    exec-channel file helpers must keep that contract via the capability cwd."""
    ssh = _FakeSSH(cwd="/workspace", files={"/workspace/f.txt": b"inside"})
    tool = WriteTool(spec=WriteTool.default_spec("qwen"), client=cast("SSHClient", ssh))

    await tool.execute({"filePath": "/REPORT.md", "content": "done"})

    assert ssh.files["/workspace/REPORT.md"] == b"done"
    # Paths already inside the workspace are untouched.
    assert await cast("SSHClient", ssh).read_text("/workspace/f.txt") == "inside"


def test_map_path_clamps_traversal_like_a_chroot() -> None:
    ssh = cast("SSHClient", _FakeSSH(cwd="/workspace"))
    assert ssh.map_path("/workspace/../etc/passwd") == "/workspace/etc/passwd"
    assert ssh.map_path("/../etc/passwd") == "/workspace/etc/passwd"
    assert ssh.map_path("../../etc/passwd") == "/workspace/etc/passwd"
    assert ssh.map_path("a/../b.txt") == "/workspace/b.txt"
    assert ssh.map_path("/") == "/workspace"
    assert ssh.map_path(".") == "/workspace"


def test_map_path_handles_windows_native_paths() -> None:
    """The workspace publishes cwd via as_posix(); callers pass native
    backslash paths, and NTFS compares case-insensitively."""
    cap = Capability(
        name="shell",
        protocol="ssh/2",
        url="ssh://localhost:22",
        params={"shell": "powershell", "cwd": "C:/work"},
    )
    ssh = SSHClient(cap, cast("Any", None))
    assert ssh.map_path("C:\\work\\file.txt") == "C:/work/file.txt"
    assert ssh.map_path("C:\\Work\\sub\\f.txt") == "C:/work/sub/f.txt"
    assert ssh.map_path("D:\\other\\f.txt") == "C:/work/other/f.txt"
    assert ssh.map_path("\\temp\\f.txt") == "C:/work/temp/f.txt"
    assert ssh.map_path("sub\\f.txt") == "C:/work/sub/f.txt"
    assert ssh.map_path("C:\\work\\..\\secrets.txt") == "C:/work/secrets.txt"


async def test_read_maps_the_directory_predicate_and_listing_together() -> None:
    """`test -d`, listing, and reads must agree on the anchored path, or
    absolute workspace dirs are misclassified as files."""
    ssh = _FakeSSH(cwd="/workspace", files={"/workspace/pkg/mod.py": b"x = 1\n"})
    tool = ReadTool(spec=ReadTool.default_spec("qwen"), client=cast("SSHClient", ssh))

    result = await tool.execute({"filePath": "/pkg"})

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
