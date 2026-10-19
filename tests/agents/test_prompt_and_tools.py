"""What every agent family hands its model: the prompt, the tools, and nothing of another run.

The model's side is a scripted provider, or a stub CLI that records its argv,
environment, stdin and MCP config; :meth:`Driver.seen` reads each the same way.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from fastmcp import FastMCP

from tests.agents.families import CLAUDE_CLI, FAMILIES, TOOL_FAMILIES, NotApplicable, over_families
from tests.fixtures.environments import run_task
from tests.support import mcp_server

if TYPE_CHECKING:
    from tests.agents.families import Driver

# The environment serves its workspace and one MCP server, ``tools``, with one tool.
GIVEN: dict[str, frozenset[str] | NotApplicable] = {
    **{name: family.workspace_tools | {"lookup"} for name, family in TOOL_FAMILIES.items()},
    "claude_cli": frozenset(CLAUDE_CLI.config().allowed_tools) | {"tools"},
    "codex_cli": NotApplicable(
        "codex is handed no tools: it runs its own, and the env's MCP servers do not reach it"
    ),
    "browser_use": NotApplicable(
        "browser-use is handed the browser, not tools; test_browser_use_wire.py checks it"
    ),
}


def lookup_server() -> FastMCP:
    server = FastMCP("tools")

    @server.tool
    def lookup(key: str) -> str:
        """Look a key up."""
        return key

    return server


@pytest.mark.parametrize(("family", "tools"), over_families(GIVEN))
async def test_the_model_is_given_the_prompt_and_the_environments_tools(
    family: str, tools: frozenset[str], driver: Driver
) -> None:
    driver.answer("done")

    async with mcp_server(lookup_server()) as server:
        run = await driver.run(driver.agent(), prompt="Fix the bug.", capabilities=(server,))

    (seen,) = driver.seen()
    assert run.trace.status == "completed"
    assert (seen.prompt, seen.tools) == ("Fix the bug.", tools)


@pytest.mark.parametrize(("family", "row"), over_families(dict.fromkeys(FAMILIES)))
async def test_one_agent_drives_concurrent_rollouts_each_with_its_own_conversation(
    family: str, row: None, driver: Driver
) -> None:
    driver.answer("done")
    env, agent = driver.environment(), driver.agent()

    runs = await asyncio.gather(
        run_task(env, agent, prompt="first"), run_task(env, agent, prompt="second")
    )

    assert [run.trace.status for run in runs] == ["completed", "completed"]
    assert sorted(seen.prompt for seen in driver.seen()) == ["first", "second"]


def named(name: str, *tools: str) -> FastMCP:
    server = FastMCP(name)
    for tool in tools:
        server.tool(name=tool, description=f"Run {tool}.")(lambda: tool)
    return server


OWN_MCP = NotApplicable("its CLI or SDK names MCP tools itself")


@pytest.mark.parametrize(
    ("family", "row"),
    over_families(
        {
            **dict.fromkeys(TOOL_FAMILIES),
            "claude_cli": OWN_MCP,
            "codex_cli": OWN_MCP,
            "browser_use": OWN_MCP,
        }
    ),
)
async def test_mcp_tools_are_qualified_within_64_characters_and_a_clash_fails_the_run(
    family: str, row: None, driver: Driver
) -> None:
    driver.answer("done")
    long = "lookup_" * 10

    async with (
        mcp_server(named("company.tools", long + "a"), name="company.tools") as dotted,
        mcp_server(named("company-tools", long + "b"), name="company-tools") as dashed,
        mcp_server(named("a", "b__c"), name="a") as first,
        mcp_server(named("a__b", "c"), name="a__b") as second,
    ):
        await driver.run(driver.agent(), capabilities=(dotted, dashed))
        clashed = await driver.run(driver.agent(), capabilities=(first, second))

    (seen,) = driver.seen()
    qualified = [name for name in seen.tools if long[:20] in name]
    assert len(qualified) == 2 and all(len(name) <= 64 for name in qualified)
    assert clashed.trace.error == (
        "[agent loop] ValueError: MCP tool name collision after qualification: 'a__b__c'"
    )
