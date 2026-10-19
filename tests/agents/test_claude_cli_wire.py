"""What only ``ClaudeCLIAgent`` hands the ``claude`` CLI, and how the CLI's exit ends the run.

A stub ``claude`` on the workspace's PATH records its argv, the variables the
agent set, stdin and the ``.hud_*`` files beside it, prints a canned JSONL stream
and exits as told; the real CLI runs in the live lane.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from hud.capabilities import Capability
from tests.agents.cli_stub import jsonl
from tests.agents.families import (
    CliDriver,
    claude_answer,
    claude_assistant,
    claude_result,
    flag,
)
from tests.support import fake_screen

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.agents.families import Driver

pytestmark = pytest.mark.parametrize("family", ["claude_cli"])


def cli(driver: Driver) -> CliDriver:
    assert isinstance(driver, CliDriver)
    return driver


# A configuration, and what the CLI is given for it: read from its argv and environment.
ARGV: dict[str, tuple[dict[str, Any], Callable[[dict[str, Any]], Any], Any]] = {
    "model-names-every-tier": (
        {"model": "claude-opus-5", "gateway": True},
        lambda capture: {
            capture["env"][name]
            for name in (
                "ANTHROPIC_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "CLAUDE_CODE_SUBAGENT_MODEL",
            )
        },
        {"claude-opus-5"},
    ),
    "reasoning-effort": (
        {"reasoning_effort": "max"},
        lambda capture: flag(capture["argv"], "--effort"),
        "max",
    ),
    "no-effort-leaves-the-cli-default": (
        {},
        lambda capture: flag(capture["argv"], "--effort"),
        None,
    ),
    "system-prompt": (
        {"system_prompt": "Be brief."},
        lambda capture: flag(capture["argv"], "--system-prompt"),
        "Be brief.",
    ),
    "permission-mode": (
        {"permission_mode": "acceptEdits"},
        lambda capture: flag(capture["argv"], "--permission-mode"),
        "acceptEdits",
    ),
    "allowed-tools": (
        {"allowed_tools": ["Read", "Bash"]},
        lambda capture: [
            capture["argv"][index + 1]
            for index, arg in enumerate(capture["argv"])
            if arg == "--allowedTools"
        ],
        ["Read", "Bash"],
    ),
    "an-unlimited-budget-sets-no-max-turns": (
        {},
        lambda capture: flag(capture["argv"], "--max-turns"),
        None,
    ),
}


@pytest.mark.parametrize(("config", "read", "expected"), ARGV.values(), ids=ARGV.keys())
async def test_the_claude_cli_is_given_what_its_configuration_says(
    config: dict[str, Any],
    read: Callable[[dict[str, Any]], Any],
    expected: Any,
    family: str,
    driver: Driver,
) -> None:
    stub = cli(driver)
    stub.answer("done")

    await stub.run(stub.agent(**config))

    (capture,) = stub.stub.captures()
    assert read(capture) == expected


WRITE = claude_assistant(
    {"type": "text", "text": "editing"},
    {"type": "tool_use", "id": "tool-1", "name": "Write", "input": {"file_path": "a.txt"}},
    stop="tool_use",
)
DONE = claude_assistant({"type": "text", "text": "done"})

EXITS = {
    "nonzero-exit-without-output": (
        {"stdout": "", "stderr": "boom", "exit": 1},
        ("error", None, "[agent loop] RuntimeError: boom"),
    ),
    "nonzero-exit-after-a-result-with-stderr": (
        {"stdout": claude_answer("done"), "stderr": "transport failed", "exit": 1},
        ("error", "done", "[agent loop] RuntimeError: transport failed"),
    ),
    "nonzero-exit-after-a-result-without-stderr": (
        {"stdout": claude_answer("done"), "exit": 1},
        ("completed", "done", None),
    ),
    "zero-exit-without-a-result": (
        {"stdout": jsonl(DONE)},
        ("error", "done", "[agent loop] RuntimeError: claude CLI exited without a result event"),
    ),
    "result-reports-an-error": (
        {"stdout": jsonl(DONE, claude_result("rate limited upstream", is_error=True))},
        ("error", "rate limited upstream", "[agent loop] RuntimeError: rate limited upstream"),
    ),
    "tool-call-left-without-a-result": (
        {"stdout": jsonl(WRITE, claude_result())},
        (
            "error",
            "done",
            "[agent loop] RuntimeError: claude CLI exited without results for tool calls: tool-1",
        ),
    ),
}


@pytest.mark.parametrize(("behavior", "expected"), EXITS.values(), ids=EXITS.keys())
async def test_how_the_claude_cli_exits_decides_how_the_run_ends(
    behavior: dict[str, Any], expected: Any, family: str, driver: Driver
) -> None:
    stub = cli(driver)
    stub.behave(**behavior)

    run = await stub.run(stub.agent())

    assert (run.trace.status, run.trace.content, run.trace.error) == expected


async def test_mcp_servers_reach_the_claude_cli_in_a_config_removed_after_the_run(
    family: str, driver: Driver
) -> None:
    stub = cli(driver)
    stub.answer("done")
    servers = (
        Capability.mcp(name="database", url="http://db.internal:8000/mcp", auth_token="secret"),
        Capability.mcp(name="search", url="http://search.internal:8000/sse", transport="sse"),
    )

    await stub.run(stub.agent(), capabilities=servers)

    (capture,) = stub.stub.captures()
    assert capture["argv"][-2:] == ["--mcp-config", ".hud_mcp_config.json"]
    assert json.loads(capture["files"][".hud_mcp_config.json"]) == {
        "mcpServers": {
            "database": {
                "type": "http",
                "url": "http://db.internal:8000/mcp",
                "headers": {"Authorization": "Bearer secret"},
            },
            "search": {"type": "sse", "url": "http://search.internal:8000/sse"},
        }
    }
    assert sorted(path.name for path in stub.workspace.iterdir()) == []


async def test_the_claude_cli_drives_the_screen_through_the_computer_bridge(
    family: str, driver: Driver
) -> None:
    stub = cli(driver)
    stub.behave(
        stdout=claude_answer("done"),
        computer=[{"action": "left_click", "coordinate": [3, 4]}, {"action": "frobnicate"}],
    )

    async with fake_screen() as screen:
        run = await stub.run(stub.agent(), capabilities=(Capability.rfb(url=screen.url),))

    (capture,) = stub.stub.captures()
    assert run.trace.status == "completed"
    assert capture["computer"] == [
        {"isError": False, "content": ["image"]},
        {"isError": True, "content": ["unsupported computer action: 'frobnicate'"]},
    ]
    assert [(event.x, event.y, event.buttons) for event in screen.pointer_events()] == [
        (3, 4, 0),
        (3, 4, 1),
        (3, 4, 0),
    ]


async def test_an_mcp_server_named_like_the_computer_bridge_is_refused(
    family: str, driver: Driver
) -> None:
    stub = cli(driver)
    stub.answer("done")

    async with fake_screen() as screen:
        clash = (
            Capability.rfb(url=screen.url),
            Capability.mcp(name="computer-use", url="http://tools.internal:8000/mcp"),
        )
        run = await stub.run(stub.agent(), capabilities=clash)

    assert (run.trace.error, stub.stub.captures()) == (
        "[agent loop] RuntimeError: duplicate MCP server name 'computer-use'",
        [],
    )
