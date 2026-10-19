"""What only ``CodexCLIAgent`` hands ``codex exec``, what it records from its JSONL items,
and how the CLI's exit ends the run.

A stub ``codex`` on the workspace's PATH records its argv, the variables the
agent set and stdin, prints a canned JSONL stream and exits as told; the real CLI
runs in the live lane.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents.types import ToolStep
from tests.agents.cli_stub import jsonl
from tests.agents.families import (
    CODEX_COMPLETED,
    CODEX_STARTED,
    CliDriver,
    codex_item,
    flag,
    overrides,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.agents.families import Driver

pytestmark = pytest.mark.parametrize("family", ["codex_cli"])


def cli(driver: Driver) -> CliDriver:
    assert isinstance(driver, CliDriver)
    return driver


ARGV: dict[str, tuple[dict[str, Any], Callable[[list[str]], Any], Any]] = {
    "model": ({"model": "gpt-5.4-mini"}, lambda argv: flag(argv, "--model"), "gpt-5.4-mini"),
    "sandbox": ({"sandbox": "read-only"}, lambda argv: flag(argv, "--sandbox"), "read-only"),
    "reasoning-effort": (
        {"reasoning_effort": "high"},
        lambda argv: overrides(argv).get("model_reasoning_effort"),
        json.dumps("high"),
    ),
    "no-effort-leaves-the-cli-default": (
        {},
        lambda argv: overrides(argv).get("model_reasoning_effort"),
        None,
    ),
    "the-gateway-speaks-the-responses-api": (
        {"gateway": True},
        lambda argv: overrides(argv).get("model_providers.hud.wire_api"),
        json.dumps("responses"),
    ),
}


@pytest.mark.parametrize(("config", "read", "expected"), ARGV.values(), ids=ARGV.keys())
async def test_codex_is_given_what_its_configuration_says(
    config: dict[str, Any],
    read: Callable[[list[str]], Any],
    expected: Any,
    family: str,
    driver: Driver,
) -> None:
    stub = cli(driver)
    stub.answer("done")

    await stub.run(stub.agent(**config))

    (capture,) = stub.stub.captures()
    assert read(capture["argv"]) == expected


# An item codex reports it ran, and the tool step it lands as: name, error, result text.
ITEMS: dict[str, tuple[dict[str, Any], tuple[str, bool, list[str]]]] = {
    "a-failed-command": (
        codex_item(
            "command_execution",
            "c",
            command="false",
            aggregated_output="",
            exit_code=1,
            status="completed",
        ),
        ("shell", True, [""]),
    ),
    "a-file-change": (
        codex_item(
            "file_change",
            "f",
            changes=[{"kind": "update", "path": "a.txt"}],
            status="completed",
        ),
        ("apply_patch", False, ["update: a.txt"]),
    ),
    "an-mcp-tool-call": (
        codex_item(
            "mcp_tool_call",
            "m",
            server="tools",
            tool="lookup",
            arguments={"key": "k"},
            result={"content": [{"type": "text", "text": "found"}]},
            status="completed",
        ),
        ("lookup", False, ["found"]),
    ),
    "a-failed-mcp-tool-call": (
        codex_item(
            "mcp_tool_call",
            "m",
            server="tools",
            tool="lookup",
            arguments={},
            error={"message": "no such key"},
            status="failed",
        ),
        ("lookup", True, ["no such key"]),
    ),
    "a-web-search": (
        codex_item("web_search", "w", query="sky", action={"type": "search", "query": "sky"}),
        ("web_search", False, ['{"type":"search","query":"sky"}']),
    ),
}


@pytest.mark.parametrize(("item", "expected"), ITEMS.values(), ids=ITEMS.keys())
async def test_each_item_codex_ran_lands_on_the_trace_as_a_tool_step(
    item: dict[str, Any], expected: tuple[str, bool, list[str]], family: str, driver: Driver
) -> None:
    stub = cli(driver)
    stub.behave(stdout=jsonl(CODEX_STARTED, item, CODEX_COMPLETED))

    run = await stub.run(stub.agent())

    (step,) = [step for step in run.trace.steps if isinstance(step, ToolStep)]
    assert step.call is not None and step.result is not None
    texts = [getattr(block, "text", None) for block in step.result.content]
    assert (step.call.name, step.result.isError, texts) == expected


EXITS = {
    "turn-failed": (
        {
            "stdout": jsonl(
                {"type": "turn.started"},
                {"type": "turn.failed", "error": {"message": "model unavailable"}},
            )
        },
        ("error", None, "[agent loop] RuntimeError: model unavailable"),
    ),
    "nonzero-exit": (
        {"stdout": "", "stderr": "authentication failed", "exit": 1},
        ("error", None, "[agent loop] RuntimeError: authentication failed"),
    ),
    "nonzero-exit-with-a-structured-error": (
        {
            "stdout": jsonl({"type": "error", "message": "gateway rejected streaming"}),
            "stderr": "noisy",
            "exit": 1,
        },
        ("error", None, "[agent loop] RuntimeError: gateway rejected streaming"),
    ),
    "zero-exit-without-turn-completed": (
        {
            "stdout": jsonl(
                CODEX_STARTED,
                codex_item("agent_message", "message-1", text="Implemented and verified."),
            )
        },
        (
            "error",
            "Implemented and verified.",
            "[agent loop] RuntimeError: codex CLI exited without a turn.completed event",
        ),
    ),
}


@pytest.mark.parametrize(("behavior", "expected"), EXITS.values(), ids=EXITS.keys())
async def test_how_codex_exits_decides_how_the_run_ends(
    behavior: dict[str, Any], expected: Any, family: str, driver: Driver
) -> None:
    stub = cli(driver)
    stub.behave(**behavior)

    run = await stub.run(stub.agent())

    assert (run.trace.status, run.trace.content, run.trace.error) == expected
