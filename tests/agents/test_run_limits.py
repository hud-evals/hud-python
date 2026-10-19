"""The limits a run's configuration puts on an agent: its step budget, its own timeout,
each tool call's timeout, and the provider's token cap."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from fastmcp import FastMCP

from tests.agents.cli_stub import process_exited
from tests.agents.families import (
    CLAUDE,
    TOOL_FAMILIES,
    NotApplicable,
    over_families,
    over_family_cases,
)
from tests.support import Turn, call, mcp_server, run_task, say, workspace_env

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.agents.families import Driver, ToolFamily
    from tests.support import HudEnv, Models

# A run whose model never finishes, with a budget of two steps: how many model
# requests it made, the budget handed down to a CLI, and how the run stopped.
STEP_BUDGET: dict[str, tuple[int, int | None, str | None, bool] | NotApplicable] = {
    **dict.fromkeys(TOOL_FAMILIES, (2, None, "max_steps", True)),
    "claude_cli": (1, 2, None, False),
    "codex_cli": NotApplicable("codex exec takes no turn budget"),
    "browser_use": (2, None, None, False),
}


@pytest.mark.parametrize(("family", "expected"), over_families(STEP_BUDGET))
async def test_the_step_budget_bounds_a_model_that_never_finishes(
    family: str, expected: tuple[int, int | None, str | None, bool], driver: Driver
) -> None:
    driver.persist()

    run = await driver.run(driver.agent(max_steps=2))

    seen = driver.seen()
    assert (len(seen), seen[0].budget, run.trace.stop_reason, run.trace.is_truncated) == expected


# How many CLI processes each family starts.
TIMEOUT: dict[str, int | NotApplicable] = {
    **dict.fromkeys(TOOL_FAMILIES, 0),
    "claude_cli": 1,
    "codex_cli": 1,
    "browser_use": 0,
}


@pytest.mark.parametrize(("family", "processes"), over_families(TIMEOUT))
async def test_the_agent_timeout_stops_a_run_whose_model_never_answers(
    family: str, processes: int, driver: Driver
) -> None:
    """The timeout is a margin over how long a whole run takes on this machine just now,
    so the agent has started its model (or its CLI) before the timeout lands."""
    driver.answer("done")
    started = time.monotonic()
    await driver.run(driver.agent())
    timeout = 1.0 + 2 * (time.monotonic() - started)
    driver.hang()

    run = await driver.run(driver.agent(timeout_seconds=timeout))

    launched = driver.processes()
    assert (run.trace.status, run.trace.stop_reason) == ("error", "timeout")
    assert (len(launched), all(process_exited(pid) for pid in launched)) == (processes, True)


def slow_tools() -> FastMCP:
    server = FastMCP("slow")

    @server.tool()
    async def wait() -> str:
        """Answer after two seconds."""
        await asyncio.sleep(2.0)
        return "waited"

    return server


@dataclass(frozen=True)
class Slow:
    """The model's turns given the workspace root, what the model is told, and which
    files the workspace holds once a cut-off command would have finished."""

    turns: Callable[[Path], list[Turn]]
    told: str
    files: dict[str, bool] = field(default_factory=dict[str, bool])


def tool_timeouts(family: ToolFamily) -> dict[str, Slow]:
    return {
        "shell": Slow(
            lambda root: [
                family.shell("sleep 3; touch late"),
                family.shell("touch quick"),
                say("done"),
            ],
            "timed out after 1s",
            {"late": False, "quick": True},
        ),
        "mcp-tools-are-left-alone": Slow(lambda root: [call("wait"), say("done")], "waited"),
    }


OWN_TOOLS = NotApplicable("its CLI or SDK runs its tools under its own limits")
TOOL_TIMEOUT: dict[str, dict[str, Slow] | NotApplicable] = {
    **{name: tool_timeouts(family) for name, family in TOOL_FAMILIES.items()},
    "claude_cli": OWN_TOOLS,
    "codex_cli": OWN_TOOLS,
    "browser_use": OWN_TOOLS,
}
TOOL_TIMEOUT["claude"] = {
    **tool_timeouts(CLAUDE),
    "editor-blocked-on-its-file": Slow(
        lambda root: [
            call("bash", command="mkfifo pipe"),
            call(
                "str_replace_based_edit_tool",
                command="str_replace",
                path=f"{root}/pipe",
                old_str="a",
                new_str="b",
            ),
            say("done"),
        ],
        "str_replace_based_edit_tool timed out after 1s",
    ),
}


@pytest.mark.parametrize(("family", "case"), over_family_cases(TOOL_TIMEOUT))
async def test_a_tool_call_past_the_tool_timeout_is_cut_off_and_the_run_goes_on(
    family: str,
    case: Slow,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    root = tmp_path / "ws"
    models.script(case.turns(root))
    tools = TOOL_FAMILIES[family]

    async with mcp_server(slow_tools()) as server:
        env = workspace_env(root, capabilities=(server,))
        run = await run_task(env, tools.agent_cls(tools.config(tool_timeout_seconds=1)))
    if case.files:
        await asyncio.sleep(3)

    assert run.trace.status == "completed"
    assert case.told in json.dumps([request.body for request in models.requests()])
    assert {name: (root / name).exists() for name in case.files} == case.files


CAPPED = NotApplicable("its CLI or SDK continues past a capped reply on its own")


@pytest.mark.parametrize(
    ("family", "row"),
    over_families(
        {
            **dict.fromkeys(TOOL_FAMILIES),
            "claude_cli": CAPPED,
            "codex_cli": CAPPED,
            "browser_use": CAPPED,
        }
    ),
)
async def test_a_reply_cut_at_the_token_cap_ends_the_run_as_truncated(
    family: str, row: None, driver: Driver, models: Models
) -> None:
    models.script([Turn(text="part", truncated=True)])

    run = await driver.run(driver.agent())

    trace = run.trace
    assert (trace.status, trace.stop_reason, trace.is_truncated) == ("completed", "length", True)
