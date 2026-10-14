# ruff: noqa: E501 -- snapshots quote product messages and input sequences verbatim
"""Tool calls act on the environment: its workspace, its MCP servers and its screen.

Each row scripts a provider's calls and checks what the environment holds or
received afterwards (files, MCP answers, screen input) and the tool results the
run recorded, which are what the model was handed back.
"""

from __future__ import annotations

import base64
import io
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pytest
from asyncvnc import key_codes
from dirty_equals import IsStr
from fastmcp import FastMCP
from inline_snapshot import snapshot
from PIL import Image
from pydantic import BaseModel, Field

from hud.agents.types import ToolStep
from hud.capabilities import Capability
from hud.capabilities.mcp import get_mcp_trace_id
from tests.agents.support import chat, claude, gemini, mcp_server, openai, run_task, workspace_env
from tests.harness import KeyEvent, call, computer_call, fake_screen, say, shell_call

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from hud.eval import Run
    from tests.harness import FakeScreen, HudEnv, Models, Turn


def results(run: Run, *paths: Path) -> list[str]:
    """Each tool result the run recorded, in a line: its text, ``<image WxH>`` for an image,
    and ``error:`` first when the call failed. ``paths`` read ``<path0>``, ``<path1>``..."""
    lines = []
    for step in run.trace.steps:
        if not isinstance(step, ToolStep) or step.result is None:
            continue
        text = " ".join(block_line(block) for block in step.result.content)
        for index, path in enumerate(paths):
            text = text.replace(str(path), f"<path{index}>")
        lines.append(("error: " if step.result.isError else "") + text)
    return lines


def block_line(block: Any) -> str:
    if block.type == "image":
        with Image.open(io.BytesIO(base64.b64decode(block.data))) as image:
            return f"<image {image.width}x{image.height}>"
    return getattr(block, "text", None) or f"<{block.type}>"


# ─── native workspace tools ─────────────────────────────────────────────


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


def editor(**arguments: Any) -> Turn:
    return call("str_replace_based_edit_tool", **arguments)


NOTES = {"notes.txt": "one\ntwo\nthree\n", "dup.txt": "a a a"}
ROWS = {
    # OpenAI Responses shell
    "openai-shell-commands-run-in-order": Row(
        openai(),
        lambda _: [shell_call("printf A > a.txt", "cat a.txt"), say("done")],
        snapshot({"results": ["A"], "files": {"a.txt": "A"}}),
    ),
    "openai-shell-timeout-covers-the-whole-command": Row(
        openai(),
        lambda _: [shell_call("echo ready; sleep 5; echo late", timeout_ms=500), say("done")],
        snapshot({"results": ["error: ready\n"], "files": {}}),
    ),
    "openai-shell-bounds-each-command-on-its-own": Row(
        openai(),
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
                "results": ["head-00000[truncated]0000-tail head-00000[truncated]000000000"],
                "files": {},
            }
        ),
    ),
    "openai-shell-caps-the-limit": Row(
        openai(),
        lambda _: [shell_call("true", max_output_length=20 * 1024 * 1024), say("done")],
        snapshot({"results": [""], "files": {}}),
    ),
    # OpenAI-compatible chat tools
    "chat-bash-workdir-and-timeout": Row(
        chat(),
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
error: $ cd '<path0>/my dir' && timeout 1s bash -lc 'pwd; sleep 5; echo late'
<path0>/my dir

(exit 124)\
""",
                ],
                "files": {},
            }
        ),
    ),
    "chat-write-creates-parent-directories": Row(
        chat(),
        lambda root: [call("write", filePath=f"{root}/deep/er/x.txt", content="x"), say("done")],
        snapshot(
            {"results": ["wrote 1 bytes to <path0>/deep/er/x.txt"], "files": {"deep/er/x.txt": "x"}}
        ),
    ),
    "chat-read-pages-text": Row(
        chat(),
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
                    "error: Offset 9 is out of range for this file (3 lines)",
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
        chat(),
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
        chat(),
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
                    "error: oldString matches 3 times in <path0>/dup.txt; set replaceAll to true",
                    "wrote 5 bytes to <path0>/dup.txt",
                    "error: No changes to apply: oldString and newString are identical.",
                    "error: oldString cannot be empty when editing an existing file. Provide exact text to replace, or use write for full-file replacement.",
                    "wrote 3 bytes to <path0>/fresh/new.txt",
                    "error: oldString not found in <path0>/notes.txt",
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
        chat(),
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
        gemini(),
        lambda root: [
            call("run_shell_command", command="mkdir -p 'my dir'"),
            call("run_shell_command", command="pwd", dir_path=f"{root}/my dir"),
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
$ cd '<path0>/my dir' && pwd
<path0>/my dir

(exit 0)\
""",
                ],
                "files": {},
            }
        ),
    ),
    "gemini-replace-and-read-lines": Row(
        gemini(),
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
                    "wrote 12 bytes to <path0>/notes.txt",
                    "error: old_string not found in <path0>/notes.txt",
                    "wrote 4 bytes to <path0>/made.txt",
                    """\
2
three
""",
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
        gemini(),
        lambda root: [
            call("grep_search", pattern="tw.", dir_path=str(root), include_pattern="*.txt"),
            call("glob", pattern="*.txt", dir_path=str(root)),
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
    # Claude tools
    "claude-editor": Row(
        claude(),
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
                    "error: old_str matches 3 times in <path0>/dup.txt; must be unique",
                    "error: old_str not found in <path0>/notes.txt",
                    "wrote 16 bytes to <path0>/notes.txt",
                    "error: insert_line 99 out of range (file has 4 lines)",
                    "error: cat: <path0>/missing.txt: No such file or directory",
                    "error: unknown editor command: 'delete'",
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
    assert {"results": results(run, root), "files": files(root)} == row.observed


async def test_a_path_outside_the_workspace_root_is_used_as_given(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    outside = tmp_path / "elsewhere" / "probe.txt"
    models.script([call("write", filePath=str(outside), content="done"), say("done")])

    await run_task(workspace_env(tmp_path / "ws"), chat()())

    assert outside.read_text() == "done"


# ─── MCP tools ──────────────────────────────────────────────────────────


class Address(BaseModel):
    street: str = Field(title="Street", min_length=1)


def database() -> FastMCP:
    server = FastMCP("database")

    @server.tool()
    def lookup(key: str) -> str:
        """Look a key up in the database."""
        return f"database:{key}"

    @server.tool()
    def trace() -> str:
        """Report the trace this call belongs to."""
        return str(get_mcp_trace_id())

    @server.tool()
    def register(
        name: str = Field(title="Name", min_length=2), address: Address | None = None
    ) -> str:
        """Register a person."""
        return f"registered {name}"

    return server


def search() -> FastMCP:
    server = FastMCP("search")

    @server.tool()
    def lookup(key: str) -> str:
        """Look a key up in the search index."""
        return f"search:{key}"

    return server


def named(*tools: str) -> Callable[[], FastMCP]:
    def build() -> FastMCP:
        server = FastMCP("named")
        for tool in tools:
            server.tool(name=tool, description=f"Run {tool}.")(lambda: tool)
        return server

    return build


async def serve(
    stack: AsyncExitStack,
    servers: dict[str, Callable[[], FastMCP]],
    transport: Literal["streamable-http", "sse"] = "streamable-http",
) -> tuple[Capability, ...]:
    return tuple(
        [
            await stack.enter_async_context(mcp_server(build(), name=name, transport=transport))
            for name, build in servers.items()
        ]
    )


def tool_name(tool: dict[str, Any]) -> str:
    return tool.get("name") or tool.get("function", {}).get("name", "")


AGENTS: dict[str, Callable[[], Agent]] = {"openai": openai(), "claude": claude(), "chat": chat()}


@pytest.mark.parametrize("provider", AGENTS.keys())
async def test_tools_from_two_servers_are_offered_qualified_and_reach_their_server(
    provider: str, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script(
        [
            call("database__lookup", key="k1"),
            call("search__lookup", key="k2"),
            call("database__trace"),
            say("done"),
        ]
    )

    async with AsyncExitStack() as stack:
        tools = await serve(stack, {"database": database, "search": search})
        run = await run_task(workspace_env(tmp_path / "ws", capabilities=tools), AGENTS[provider]())

    offered = [name for name in models.requests()[0].tools if "__" in name]
    assert offered == [
        "database__lookup",
        "database__trace",
        "database__register",
        "search__lookup",
    ]
    answers = ("database:k1", "search:k2", run.trace_id)
    assert results(run) == [IsStr(regex=f".*{answer}") for answer in answers]


async def test_a_single_server_offers_its_tools_unqualified(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([call("lookup", key="k1"), say("done")])

    async with AsyncExitStack() as stack:
        tools = await serve(stack, {"search": search}, transport="sse")
        await run_task(workspace_env(tmp_path / "ws", capabilities=tools), chat()())

    first, last = models.requests()
    assert ("lookup" in first.tools, last.tool_results) == (True, ["search:k1"])


async def test_long_and_unusual_server_names_still_give_valid_tool_names(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([say("done")])
    long_name = "lookup_" * 10

    async with AsyncExitStack() as stack:
        tools = await serve(
            stack,
            {
                "company.tools": named(long_name + "a"),
                "company-tools": named(long_name + "b"),
            },
        )
        await run_task(workspace_env(tmp_path / "ws", capabilities=tools), openai()())

    (request,) = models.requests()
    names = [name for name in request.tools if name != "shell"]
    assert names == snapshot(
        [
            "company_tools__lookup_lookup_lookup_lookup_lookup_looku_525f784f",
            "company-tools__lookup_lookup_lookup_lookup_lookup_looku_442a1793",
        ]
    )
    assert all(len(name) <= 64 for name in names)


async def test_tool_names_that_clash_after_qualification_fail_the_run(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")

    async with AsyncExitStack() as stack:
        tools = await serve(stack, {"a": named("b__c"), "a__b": named("c")})
        run = await run_task(workspace_env(tmp_path / "ws", capabilities=tools), openai()())

    assert (run.trace.status, run.trace.error, models.requests()) == (
        "error",
        "[agent loop] ValueError: MCP tool name collision after qualification: 'a__b__c'",
        [],
    )


# ─── computer use ───────────────────────────────────────────────────────


KEY_NAMES: dict[int, str] = {}
for _name, _code in key_codes.items():
    KEY_NAMES.setdefault(_code, _name)


def events(screen: FakeScreen) -> str:
    """The screen's input log in one line.

    ``+Key``/``-Key`` are presses and releases, a bare ``Key`` a press released
    straight away, and ``x,y bN`` a pointer event with button mask ``N``.
    """
    rendered: list[str] = []
    for event in screen.events:
        if isinstance(event, KeyEvent):
            name = KEY_NAMES[event.key]
            if not event.down and rendered and rendered[-1] == f"+{name}":
                rendered[-1] = name
            else:
                rendered.append(f"{'+' if event.down else '-'}{name}")
        else:
            rendered.append(f"{event.x},{event.y} b{event.buttons}")
    return " ".join(rendered)


@dataclass(frozen=True)
class Screen:
    """What the screen received, the error the model was told if the action failed, and
    any other text it was told (a screenshot alone says nothing)."""

    agent: Callable[[], Agent]
    script: list[Turn]
    events: str
    error: str | None = None
    said: list[str] = field(default_factory=list)


def claude_does(*actions: dict[str, Any], **seen: Any) -> Screen:
    turns = [call("computer", **action) for action in actions]
    return Screen(claude(), [*turns, say("done")], **seen)


CLAUDE = {
    "left-click-holding-ctrl": claude_does(
        {"action": "left_click", "coordinate": [10, 20], "text": "ctrl"},
        events="10,20 b0 +Control_L 10,20 b1 10,20 b0 -Control_L",
    ),
    "right-click": claude_does(
        {"action": "right_click", "coordinate": [3, 4]},
        events="3,4 b0 3,4 b4 3,4 b0",
    ),
    "middle-click": claude_does(
        {"action": "middle_click", "coordinate": [3, 4]},
        events="3,4 b0 3,4 b2 3,4 b0",
    ),
    "double-click": claude_does(
        {"action": "double_click", "coordinate": [3, 4]},
        events="3,4 b0 3,4 b1 3,4 b0 3,4 b1 3,4 b0",
    ),
    "triple-click": claude_does(
        {"action": "triple_click", "coordinate": [3, 4]},
        events="3,4 b0 3,4 b1 3,4 b0 3,4 b1 3,4 b0 3,4 b1 3,4 b0",
    ),
    "mouse-move": claude_does(
        {"action": "mouse_move", "coordinate": [7, 8]},
        events="7,8 b0",
    ),
    "mouse-down-then-up": claude_does(
        {"action": "mouse_move", "coordinate": [5, 6]},
        {"action": "left_mouse_down"},
        {"action": "mouse_move", "coordinate": [30, 20]},
        {"action": "left_mouse_up"},
        events="5,6 b0 5,6 b1 30,20 b1 30,20 b0",
    ),
    "type-with-newline-and-tab": claude_does(
        {"action": "type", "text": "a\nb\tC"},
        events="a Return b Tab C",
    ),
    "key-sequence-repeated": claude_does(
        {"action": "key", "text": "ctrl+a Delete", "repeat": 2},
        events="+Control_L a -Control_L Delete +Control_L a -Control_L Delete",
    ),
    "hold-key": claude_does(
        {"action": "hold_key", "text": "shift", "duration": 0.01},
        events="Shift_L",
    ),
    "scroll-down": claude_does(
        {
            "action": "scroll",
            "coordinate": [5, 5],
            "scroll_direction": "down",
            "scroll_amount": 2,
        },
        events="5,5 b0 5,5 b16 5,5 b0 5,5 b16 5,5 b0",
    ),
    "scroll-up-holding-shift": claude_does(
        {
            "action": "scroll",
            "coordinate": [5, 5],
            "scroll_direction": "up",
            "scroll_amount": 1,
            "text": "shift",
        },
        events="5,5 b0 +Shift_L 5,5 b8 5,5 b0 -Shift_L",
    ),
    "scroll-left": claude_does(
        {
            "action": "scroll",
            "coordinate": [5, 5],
            "scroll_direction": "left",
            "scroll_amount": 3,
        },
        events="5,5 b0",
    ),
    "drag": claude_does(
        {"action": "left_click_drag", "start_coordinate": [2, 2], "coordinate": [40, 2]},
        events="2,2 b0 2,2 b1 21,2 b1 40,2 b1 40,2 b0",
    ),
    "cursor-position": claude_does(
        {"action": "mouse_move", "coordinate": [9, 9]},
        {"action": "cursor_position"},
        events="9,9 b0",
        said=["(9, 9)"],
    ),
    "zoom": claude_does(
        {"action": "zoom", "region": [0, 0, 32, 24]},
        events="",
        said=["<image 32x24>"],
    ),
    "key-without-text": claude_does(
        {"action": "key"},
        events="",
        error="`text` (key chord) is required for key",
    ),
    "move-without-coordinate": claude_does(
        {"action": "mouse_move"},
        events="",
        error="computer action 'mouse_move' failed: coordinate.x is required",
    ),
    "unsupported-action": claude_does(
        {"action": "frobnicate"},
        events="",
        error="unsupported computer action: 'frobnicate'",
    ),
}


def openai_does(
    *actions: dict[str, Any],
    events: str,
    error: str | None = None,
    said: tuple[str, ...] = (),
    **call: Any,
) -> Screen:
    return Screen(openai(), [computer_call(*actions, **call), say("done")], events, error, [*said])


OPENAI = {
    "click-holding-ctrl": openai_does(
        {"type": "click", "x": 3, "y": 4, "button": "left", "keys": ["ctrl"]},
        events="3,4 b0 +Control_L 3,4 b1 3,4 b0 -Control_L",
    ),
    "right-and-wheel-click": openai_does(
        {"type": "click", "x": 3, "y": 4, "button": "right"},
        {"type": "click", "x": 3, "y": 4, "button": "wheel"},
        events="3,4 b0 3,4 b4 3,4 b0 3,4 b0 3,4 b2 3,4 b0",
    ),
    "double-click": openai_does(
        {"type": "double_click", "x": 3, "y": 4},
        events="3,4 b0 3,4 b1 3,4 b0 3,4 b1 3,4 b0",
    ),
    "scroll": openai_does(
        {"type": "scroll", "x": 5, "y": 5, "scroll_x": 4, "scroll_y": -2},
        events="5,5 b0 5,5 b8 5,5 b0 5,5 b8 5,5 b0",
    ),
    "type-and-keypress": openai_does(
        {"type": "type", "text": "hi\n"},
        {"type": "keypress", "keys": ["CTRL", "c"]},
        events="h i Return +Control_L c -Control_L",
    ),
    "move-drag-wait-screenshot": openai_does(
        {"type": "move", "x": 1, "y": 1},
        {"type": "drag", "path": [{"x": 2, "y": 2}, {"x": 20, "y": 2}]},
        {"type": "wait", "ms": 10},
        {"type": "screenshot"},
        events="1,1 b0 2,2 b0 2,2 b1 20,2 b1 20,2 b0",
    ),
    "response": openai_does(
        {"type": "response", "text": "all done"},
        events="",
        said=("all done <image 200x100>",),
    ),
    "invalid-action-stops-the-batch": openai_does(
        {"type": "move", "x": 3, "y": 4},
        {"type": "keypress", "keys": "ESC"},
        {"type": "type", "text": "later"},
        events="3,4 b0",
        error="computer action 'keypress' failed: keys must be a list of non-empty strings",
    ),
    "unknown-action-type": openai_does(
        {"type": "frobnicate"},
        events="",
        error="computer action 'frobnicate' failed: Invalid action type: frobnicate",
    ),
    "empty-actions": openai_does(events="", error="actions list is empty"),
    "safety-checks-acknowledged": openai_does(
        {"type": "screenshot"},
        pending_safety_checks=[{"id": "sc_1", "code": "malicious_instructions", "message": "x"}],
        events="",
    ),
    "webp-screenshots": Screen(
        agent=openai(screenshot_encoding={"mime_type": "image/webp", "quality": 42}),
        script=[computer_call({"type": "screenshot"}), say("done")],
        events="",
    ),
}


def gemini_does(*turns: Turn, **seen: Any) -> Screen:
    return Screen(gemini(), [*turns, say("done")], **seen)


GEMINI = {
    "open-web-browser": gemini_does(
        call("open_web_browser"),
        events="",
    ),
    "click-at": gemini_does(
        call("click_at", x=500, y=500),
        events="100,50 b0 100,50 b1 100,50 b0",
    ),
    "hover-at": gemini_does(
        call("hover_at", x=250, y=750),
        events="50,75 b0",
    ),
    "type-text-at-clearing-first": gemini_does(
        call("type_text_at", x=100, y=200, text="hi", press_enter=True),
        events="20,20 b0 20,20 b0 20,20 b1 20,20 b0 +Control_L a -Control_L Delete h i Return",
    ),
    "type-text-at-without-clearing": gemini_does(
        call("type_text_at", x=100, y=200, text="hi", clear_before_typing=False),
        events="20,20 b0 20,20 b0 20,20 b1 20,20 b0 h i",
    ),
    "scroll-document-down": gemini_does(
        call("scroll_document", direction="down"),
        events="0,0 b16 0,0 b0 0,0 b16 0,0 b0 0,0 b16 0,0 b0",
    ),
    "scroll-at-up": gemini_does(
        call("scroll_at", x=500, y=500, direction="up", magnitude=2),
        events="100,50 b0 100,50 b8 100,50 b0 100,50 b8 100,50 b0",
    ),
    "go-back-and-forward": gemini_does(
        call("go_back"),
        call("go_forward"),
        events="+Alt_L Left -Alt_L +Alt_L Right -Alt_L",
    ),
    "search": gemini_does(
        call("search"),
        events="+Control_L l -Control_L h t t p s colon slash slash w w w period g o o g l e period c o m Return",
    ),
    "navigate": gemini_does(
        call("navigate", url="https://a.b"),
        events="+Control_L l -Control_L h t t p s colon slash slash a period b Return",
    ),
    "key-combination": gemini_does(
        call("key_combination", keys="Control+a"),
        events="+Control_L a -Control_L",
    ),
    "drag-and-drop-near-the-edge": gemini_does(
        call("drag_and_drop", x=0, y=500, destination_x=999, destination_y=500),
        events="25,50 b0 25,50 b1 41,50 b1 58,50 b1 75,50 b1 91,50 b1 108,50 b1 125,50 b1 141,50 b1 158,50 b1 175,50 b1 175,50 b0",
    ),
    "key-combination-without-keys": gemini_does(
        call("key_combination", keys=3),
        events="",
        error="keys must be a '+'-separated string",
    ),
}

SCREENSHOT = "<image 200x100>"
SCREENS = {
    **{f"claude-{name}": row for name, row in CLAUDE.items()},
    **{f"openai-{name}": row for name, row in OPENAI.items()},
    **{f"gemini-{name}": row for name, row in GEMINI.items()},
}


@pytest.mark.parametrize("row", SCREENS.values(), ids=SCREENS.keys())
async def test_a_computer_action_reaches_the_screen_as_rfb_input(
    row: Screen, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script(row.script)

    async with fake_screen(width=200, height=100) as screen:
        screen_capability = Capability.rfb(url=screen.url)
        run = await run_task(
            workspace_env(tmp_path / "ws", capabilities=(screen_capability,)), row.agent()
        )

    seen = results(run)
    errors = [
        re.sub(r" <image \d+x\d+>$", "", line.removeprefix("error: "))
        for line in seen
        if line.startswith("error: ")
    ]
    told = [line for line in seen if not line.startswith("error: ") and line != SCREENSHOT]
    assert run.trace.status == "completed"
    assert (events(screen), errors, told) == (
        row.events,
        [row.error] if row.error else [],
        row.said,
    )


async def test_only_the_first_screen_is_driven(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([call("computer", action="left_click", coordinate=[1, 1]), say("done")])

    async with fake_screen() as first, fake_screen() as second:
        screens = (
            Capability.rfb(name="screen-0", url=first.url),
            Capability.rfb(name="screen-1", url=second.url),
        )
        await run_task(workspace_env(tmp_path / "ws", capabilities=screens), claude()())

    assert (events(first), events(second), second.connections) == ("1,1 b0 1,1 b1 1,1 b0", "", 0)
