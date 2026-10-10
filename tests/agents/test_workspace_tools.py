# ruff: noqa: E501 -- snapshots quote product messages verbatim
"""Each provider's native workspace tools act on the files a real workspace serves.

Rows script one or more tool calls against a workspace seeded with files and
pin what the model was sent back and what the workspace holds afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

from hud.agents import ClaudeAgent, GeminiAgent, OpenAIAgent, OpenAIChatAgent
from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIChatConfig, OpenAIConfig
from tests.agents.support import result_line, run_task, tool_results, workspace_env
from tests.harness import call, say, shell_call

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from tests.harness import HudEnv, Models, Turn


def files(root: Path) -> dict[str, str]:
    """Every regular file under ``root``, by relative path."""
    return {
        str(path.relative_to(root)): path.read_text(errors="replace")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@dataclass(frozen=True)
class Row:
    agent: Callable[[], Agent]
    script: Callable[[Path], list[Turn]]
    observed: Any
    seed: dict[str, str] | None = None


def claude() -> Agent:
    return ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6"))


def openai() -> Agent:
    return OpenAIAgent(OpenAIConfig(model="gpt-5.6"))


def gemini() -> Agent:
    return GeminiAgent(GeminiConfig(model="gemini-3.1-pro-preview"))


def chat() -> Agent:
    return OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus"))


def editor(**arguments: Any) -> Turn:
    return call("str_replace_based_edit_tool", **arguments)


NOTES = {"notes.txt": "one\ntwo\nthree\n", "dup.txt": "a a a"}

ROWS = {
    # OpenAI Responses shell
    "openai-shell-commands-run-in-order": Row(
        openai,
        lambda _: [shell_call("printf A > a.txt", "cat a.txt"), say("done")],
        snapshot(
            {
                "results": [
                    '[{"stdout": "", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}, {"stdout": "A", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}] limit=10485760'
                ],
                "files": {"a.txt": "A"},
            }
        ),
    ),
    "openai-shell-timeout-covers-the-whole-command": Row(
        openai,
        lambda _: [shell_call("echo ready; sleep 5; echo late", timeout_ms=500), say("done")],
        snapshot(
            {
                "results": [
                    '[{"stdout": "ready\\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 124}}] limit=10485760'
                ],
                "files": {},
            }
        ),
    ),
    "openai-shell-bounds-each-command-on-its-own": Row(
        openai,
        lambda _: [
            shell_call(
                "printf 'head-%050d' 0; printf '%050d-tail' 0 >&2",
                "printf 'head-%050d' 0",
                max_output_length=30,
            ),
            say("done"),
        ],
        snapshot(
            {
                "results": [
                    '[{"stdout": "head-00000[truncated]", "stderr": "0000-tail", "outcome": {"type": "exit", "exit_code": 0}}, {"stdout": "head-00000[truncated]000000000", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}] limit=30'
                ],
                "files": {},
            }
        ),
    ),
    "openai-shell-limit-smaller-than-the-marker": Row(
        openai,
        lambda _: [shell_call("seq 100", max_output_length=1), say("done")],
        snapshot(
            {
                "results": [
                    '[{"stdout": "[", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}] limit=1'
                ],
                "files": {},
            }
        ),
    ),
    "openai-shell-caps-the-limit": Row(
        openai,
        lambda _: [shell_call("true", max_output_length=20 * 1024 * 1024), say("done")],
        snapshot(
            {
                "results": [
                    '[{"stdout": "", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}] limit=10485760'
                ],
                "files": {},
            }
        ),
    ),
    "openai-shell-rejects-a-zero-limit": Row(
        openai,
        lambda _: [shell_call("touch ran", max_output_length=0), say("done")],
        snapshot(
            {
                "results": [
                    '[{"stdout": "", "stderr": "max_output_length must be a positive integer", "outcome": {"type": "exit", "exit_code": 1}}]'
                ],
                "files": {},
            }
        ),
    ),
    "openai-shell-rejects-commands-that-are-not-a-list": Row(
        openai,
        lambda _: [
            replace(
                shell_call(),
                native=(
                    {
                        "type": "shell_call",
                        "id": "sh_bad",
                        "call_id": "call_bad",
                        "action": {"commands": [1, 2]},
                        "status": "completed",
                    },
                ),
            ),
            say("done"),
        ],
        snapshot(
            {
                "results": [
                    '[{"stdout": "", "stderr": "commands must be a list of strings", "outcome": {"type": "exit", "exit_code": 1}}] limit=10485760'
                ],
                "files": {},
            }
        ),
    ),
    # OpenAI-compatible chat tools
    "chat-bash-workdir-and-timeout": Row(
        chat,
        lambda root: [
            call("bash", command="mkdir -p 'my dir'"),
            call(
                "bash",
                command="pwd; sleep 5; echo late",
                workdir=f"{root}/my dir",
                timeout=500,
            ),
            say("done"),
        ],
        snapshot(
            {
                "results": [
                    """\
$ mkdir -p 'my dir'

(exit 0)\
""",
                    """\
$ cd '<path0>/my dir' && timeout 1s bash -lc 'pwd; sleep 5; echo late'
<path0>/my dir

(exit 124)\
""",
                ],
                "files": {},
            }
        ),
    ),
    "chat-write-creates-parent-directories": Row(
        chat,
        lambda root: [call("write", filePath=f"{root}/deep/er/x.txt", content="x"), say("done")],
        snapshot(
            {"results": ["wrote 1 bytes to <path0>/deep/er/x.txt"], "files": {"deep/er/x.txt": "x"}}
        ),
    ),
    "chat-read-pages-text": Row(
        chat,
        lambda root: [
            call("read", filePath=f"{root}/notes.txt", offset=0, limit=1),
            call("read", filePath=f"{root}/notes.txt", offset=2),
            call("read", filePath=f"{root}/notes.txt", offset=9),
            say("done"),
        ],
        snapshot(
            {
                "results": [
                    """\
<path><path0>/notes.txt</path>
<type>file</type>
<content>
1: one

(Showing lines 1-1 of 3. Use offset=2 to continue.)
</content>\
""",
                    """\
<path><path0>/notes.txt</path>
<type>file</type>
<content>
2: two
3: three

(End of file - total 3 lines)
</content>\
""",
                    "Offset 9 is out of range for this file (3 lines)",
                ],
                "files": {
                    "dup.txt": "a a a",
                    "notes.txt": """\
one
two
three
""",
                },
            }
        ),
        seed=NOTES,
    ),
    "chat-read-lists-a-directory": Row(
        chat,
        lambda root: [
            call("bash", command="mkdir -p pkg/sub && touch pkg/mod.py"),
            call("read", filePath=f"{root}/pkg"),
            say("done"),
        ],
        snapshot(
            {
                "results": [
                    """\
$ mkdir -p pkg/sub && touch pkg/mod.py

(exit 0)\
""",
                    """\
<path><path0>/pkg</path>
<type>directory</type>
<entries>
mod.py
sub

(2 entries)
</entries>\
""",
                ],
                "files": {"pkg/mod.py": ""},
            }
        ),
    ),
    "chat-edit": Row(
        chat,
        lambda root: [
            call("edit", filePath=f"{root}/notes.txt", oldString="two", newString="2"),
            call("edit", filePath=f"{root}/dup.txt", oldString="a", newString="b"),
            call("edit", filePath=f"{root}/dup.txt", oldString="a", newString="b", replaceAll=True),
            call("edit", filePath=f"{root}/notes.txt", oldString="one", newString="one"),
            call("edit", filePath=f"{root}/notes.txt", oldString="", newString="new"),
            call("edit", filePath=f"{root}/fresh/new.txt", oldString="", newString="new"),
            call("edit", filePath=f"{root}/notes.txt", oldString="zzz", newString="y"),
            say("done"),
        ],
        snapshot(
            {
                "results": [
                    "wrote 12 bytes to <path0>/notes.txt",
                    "oldString matches 3 times in <path0>/dup.txt; set replaceAll to true",
                    "wrote 5 bytes to <path0>/dup.txt",
                    "No changes to apply: oldString and newString are identical.",
                    "oldString cannot be empty when editing an existing file. Provide exact text to replace, or use write for full-file replacement.",
                    "wrote 3 bytes to <path0>/fresh/new.txt",
                    "oldString not found in <path0>/notes.txt",
                ],
                "files": {
                    "dup.txt": "b b b",
                    "fresh/new.txt": "new",
                    "notes.txt": """\
one
2
three
""",
                },
            }
        ),
        seed=NOTES,
    ),
    "chat-grep-and-glob": Row(
        chat,
        lambda root: [
            call("grep", pattern="tw.", path=str(root), include="*.txt"),
            call("glob", pattern="*.txt", path=str(root)),
            say("done"),
        ],
        snapshot(
            {
                "results": [
                    """\
$ grep -rn tw. <path0> --include='*.txt'
<path0>/notes.txt:1:two

(exit 0)\
""",
                    """\
$ find <path0> -name '*.txt'
<path0>/notes.txt

(exit 0)\
""",
                ],
                "files": {"notes.txt": "two\n"},
            }
        ),
        seed={"notes.txt": "two\n"},
    ),
    # Gemini tools
    "gemini-shell-in-a-quoted-directory": Row(
        gemini,
        lambda root: [
            call("run_shell_command", command="mkdir -p 'my dir'"),
            call("run_shell_command", command="pwd", dir_path=f"{root}/my dir"),
            say("done"),
        ],
        snapshot(
            {
                "results": [
                    'run_shell_command {"success": true, "output": "$ mkdir -p \'my dir\'\\n\\n(exit 0)"}',
                    'run_shell_command {"success": true, "output": "$ cd \'<path0>/my dir\' && pwd\\n<path0>/my dir\\n\\n(exit 0)"}',
                ],
                "files": {},
            }
        ),
    ),
    "gemini-shell-without-a-command": Row(
        gemini,
        lambda _: [call("run_shell_command", command=""), say("done")],
        snapshot(
            {
                "results": ['run_shell_command {"error": "tool error: command is required"}'],
                "files": {},
            }
        ),
    ),
    "gemini-replace-and-read-lines": Row(
        gemini,
        lambda root: [
            call("replace", file_path=f"{root}/notes.txt", old_string="two", new_string="2"),
            call("replace", file_path=f"{root}/notes.txt", old_string="zzz", new_string="y"),
            call("replace", file_path=f"{root}/made.txt", old_string="", new_string="made"),
            call("read_file", file_path=f"{root}/notes.txt", start_line=2, end_line=3),
            say("done"),
        ],
        snapshot(
            {
                "results": [
                    'replace {"success": true, "output": "wrote 12 bytes to <path0>/notes.txt"}',
                    'replace {"error": "old_string not found in <path0>/notes.txt"}',
                    'replace {"success": true, "output": "wrote 4 bytes to <path0>/made.txt"}',
                    'read_file {"success": true, "output": "2\\nthree\\n"}',
                ],
                "files": {
                    "dup.txt": "a a a",
                    "made.txt": "made",
                    "notes.txt": """\
one
2
three
""",
                },
            }
        ),
        seed=NOTES,
    ),
    "gemini-search-and-glob": Row(
        gemini,
        lambda root: [
            call("grep_search", pattern="tw.", dir_path=str(root), include_pattern="*.txt"),
            call("glob", pattern="*.txt", dir_path=str(root)),
            say("done"),
        ],
        snapshot(
            {
                "results": [
                    'grep_search {"success": true, "output": "$ grep -rn tw. <path0> --include=\'*.txt\'\\n<path0>/notes.txt:1:two\\n\\n(exit 0)"}',
                    'glob {"success": true, "output": "$ find <path0> -name \'*.txt\'\\n<path0>/notes.txt\\n\\n(exit 0)"}',
                ],
                "files": {"notes.txt": "two\n"},
            }
        ),
        seed={"notes.txt": "two\n"},
    ),
    # Claude tools
    "claude-bash-restart-and-missing-command": Row(
        claude,
        lambda _: [call("bash", restart=True), call("bash"), say("done")],
        snapshot(
            {
                "results": [
                    "restart is unnecessary; each command runs in a fresh shell session",
                    "error: Error: command is required unless restart is true",
                ],
                "files": {},
            }
        ),
    ),
    "claude-editor": Row(
        claude,
        lambda root: [
            editor(command="create", path=f"{root}/made.txt", file_text="made\n"),
            editor(command="str_replace", path=f"{root}/notes.txt", old_str="two", new_str="2"),
            editor(command="str_replace", path=f"{root}/dup.txt", old_str="a", new_str="b"),
            editor(command="str_replace", path=f"{root}/notes.txt", old_str="zzz", new_str="y"),
            editor(command="insert", path=f"{root}/notes.txt", insert_line=1, new_str="1.5"),
            editor(command="insert", path=f"{root}/notes.txt", insert_line=99, new_str="x"),
            editor(command="view", path=f"{root}/missing.txt"),
            editor(command="delete", path=f"{root}/notes.txt"),
            say("done"),
        ],
        snapshot(
            {
                "results": [
                    "wrote 5 bytes to <path0>/made.txt",
                    "wrote 12 bytes to <path0>/notes.txt",
                    "error: Error: old_str matches 3 times in <path0>/dup.txt; must be unique",
                    "error: Error: old_str not found in <path0>/notes.txt",
                    "wrote 16 bytes to <path0>/notes.txt",
                    "error: Error: insert_line 99 out of range (file has 4 lines)",
                    "error: Error: cat: <path0>/missing.txt: No such file or directory",
                    "error: Error: unknown editor command: 'delete'",
                ],
                "files": {
                    "dup.txt": "a a a",
                    "made.txt": "made\n",
                    "notes.txt": """\
one
1.5
2
three
""",
                },
            }
        ),
        seed=NOTES,
    ),
}


@pytest.mark.parametrize("row", ROWS.values(), ids=ROWS.keys())
async def test_a_native_tool_acts_on_the_served_workspace(
    row: Row, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    root = tmp_path / "ws"
    models.script(row.script(root))

    run = await run_task(workspace_env(root, files=row.seed), row.agent())

    assert run.trace.status == "completed"
    observed = {
        "results": [result_line(entry, root) for entry in tool_results(models.requests())],
        "files": files(root),
    }
    assert observed == row.observed


async def test_a_path_outside_the_workspace_root_is_used_as_given(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    outside = tmp_path / "elsewhere" / "probe.txt"
    models.script([call("write", filePath=str(outside), content="done"), say("done")])

    await run_task(workspace_env(tmp_path / "ws"), chat())

    assert outside.read_text() == "done"
