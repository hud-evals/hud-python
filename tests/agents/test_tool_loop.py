"""What the provider agents' shared tool loop decides: when a run stops, how long a tool
may take, when a lost workspace ends the run, and that colliding MCP tool names fail it.

The loop is one implementation every provider agent runs, so each rule is checked once,
on the chat completions agent; what each provider does differently is
``test_provider_agents``."""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from dataclasses import replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import pytest
from fastmcp import FastMCP

from hud.agents import ClaudeAgent, OpenAIChatAgent
from hud.agents.types import ClaudeConfig, OpenAIChatConfig
from tests.agents.conftest import tool_results
from tests.fixtures.asgi import mcp_server
from tests.fixtures.environments import workspace_task
from tests.fixtures.models import Turn, call
from tests.fixtures.relay import relay

if TYPE_CHECKING:
    from pathlib import Path

    from hud.types import StopCondition
    from tests.conftest import HudEnv
    from tests.fixtures.models import ModelRequest, Models
    from tests.fixtures.relay import FlakyRelay


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


CUT_OFF = Turn(
    native=(
        {
            "id": "call_cut",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ec'},
        },
    )
)


@pytest.mark.parametrize(
    ("reason", "turn"),
    [
        pytest.param(
            "length", replace(call("bash", command="true"), truncated=True), id="a-capped-reply"
        ),
        pytest.param("malformed_tool_call", CUT_OFF, id="a-malformed-call"),
    ],
)
async def test_stop_on_ends_the_run_at_the_call_it_names(
    reason: StopCondition, turn: Turn, models: Models, tmp_path: Path
) -> None:
    models.script([turn])

    task = workspace_task(tmp_path / "ws")
    (run,) = (
        await task().run(OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus", stop_on={reason})))
    ).runs

    assert (run.trace.status, run.trace.stop_reason) == ("completed", reason)
    assert len(models.requests()) == 1


async def test_the_step_budget_ends_a_model_that_never_finishes(
    models: Models, tmp_path: Path
) -> None:
    models.respond(lambda request: call("bash", command="true"))

    task = workspace_task(tmp_path / "ws")
    (run,) = (
        await task().run(OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus", max_steps=2)))
    ).runs

    assert (run.trace.status, run.trace.stop_reason, len(models.requests())) == (
        "completed",
        "max_steps",
        2,
    )


@pytest.mark.parametrize(
    ("calls", "status"),
    [
        pytest.param(["shell", "shell", "shell"], "error", id="three-lost-calls-in-a-row"),
        pytest.param(
            ["shell", "ghost", "shell", "shell"], "completed", id="another-result-resets-the-count"
        ),
    ],
)
async def test_a_lost_workspace_ends_the_run_after_three_consecutive_failures(
    calls: list[str], status: str, models: Models, tmp_path: Path
) -> None:
    turns = [call("bash", command="true")] + [
        call("bash", command="true") if name == "shell" else call(name) for name in calls
    ]
    task = workspace_task(tmp_path / "ws")
    relays: list[FlakyRelay] = []

    async with AsyncExitStack() as stack:

        @task.env.initialize
        async def reach_the_workspace_through_a_relay() -> None:
            shell = task.env.capability("shell")
            address = urlsplit(shell.url)
            assert address.hostname is not None and address.port is not None
            workspace = await stack.enter_async_context(relay(address.hostname, address.port))
            relays.append(workspace)
            task.env.add_capability(replace(shell, url=f"ssh://127.0.0.1:{workspace.port}"))

        def answer(request: ModelRequest) -> Turn:
            if request.turn == 1:
                (workspace,) = relays
                workspace.refuse(sys.maxsize)
                workspace.sever()
            return turns[request.turn] if request.turn < len(turns) else Turn(text="done")

        models.respond(answer)
        (run,) = (
            await task().run(OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus", max_steps=10)))
        ).runs

    assert run.trace.status == status
    if status == "error":
        assert run.trace.error == "SSH tool failure limit reached after 3 consecutive errors"


# ─── the tool timeout ──────────────────────────────────────────────────


async def test_a_shell_call_past_the_tool_timeout_is_cut_off_and_the_run_goes_on(
    models: Models, tmp_path: Path
) -> None:
    root = tmp_path / "ws"
    models.script(
        [
            call("bash", command="sleep 3; touch late"),
            call("bash", command="touch quick"),
            Turn(text="done"),
        ]
    )

    task = workspace_task(root)
    (run,) = (
        await task().run(
            OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus", tool_timeout_seconds=1))
        )
    ).runs
    await asyncio.sleep(3)

    (cut_failed, text), (quick_failed, _) = tool_results(run)
    assert (run.trace.status, cut_failed, quick_failed) == ("completed", True, False)
    assert "timed out after 1s" in text
    assert ((root / "late").exists(), (root / "quick").exists()) == (False, True)


async def test_a_workspace_tool_blocked_on_its_file_is_cut_off_by_the_tool_timeout(
    models: Models, tmp_path: Path
) -> None:
    """Claude's editor reads the file itself, so a fifo blocks it rather than the shell."""
    root = tmp_path / "ws"
    edit = call(
        "str_replace_based_edit_tool",
        command="str_replace",
        path=f"{root}/pipe",
        old_str="a",
        new_str="b",
    )
    models.script([call("bash", command="mkfifo pipe"), edit, Turn(text="done")])

    task = workspace_task(root)
    (run,) = (
        await task().run(
            ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6", tool_timeout_seconds=1))
        )
    ).runs

    *_, (failed, text) = tool_results(run)
    assert (run.trace.status, failed) == ("completed", True)
    assert "str_replace_based_edit_tool timed out after 1s" in text


# ─── MCP tool names ────────────────────────────────────────────────────


def named(name: str, *tools: str) -> FastMCP:
    server = FastMCP(name)
    for tool in tools:
        server.tool(name=tool, description=f"Run {tool}.")(lambda: tool)
    return server


async def test_two_mcp_tools_qualified_to_one_name_fail_the_run(
    models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text="done")])

    async with (
        mcp_server(named("a", "b__c"), name="a") as first,
        mcp_server(named("a__b", "c"), name="a__b") as second,
    ):
        task = workspace_task(tmp_path / "ws", capabilities=(first, second))
        (run,) = (await task().run(OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus")))).runs

    assert run.trace.error == (
        "[agent loop] ValueError: MCP tool name collision after qualification: 'a__b__c'"
    )
    assert models.requests() == []
