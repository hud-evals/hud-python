"""What the provider agents' shared tool loop decides: what the model is told when a call
cannot run, when a run stops, how long a tool may take, and how MCP tools are named.

The loop is one implementation every provider agent runs, so each rule is checked once,
on the chat completions agent; what each provider does differently is
``test_provider_agents``."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from dataclasses import replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import pytest
from fastmcp import FastMCP

from hud.agents import ClaudeAgent, OpenAIChatAgent
from hud.agents.types import ClaudeConfig, OpenAIChatConfig
from tests.fixtures.asgi import mcp_server
from tests.fixtures.environments import run_task, workspace_env
from tests.fixtures.models import Turn, call, fail, say, shell
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


def failing_tools() -> FastMCP:
    server = FastMCP("tools")

    @server.tool
    def explode() -> str:
        """Fail."""
        raise ValueError("kaboom")

    return server


@pytest.mark.parametrize(
    ("turn", "told"),
    [
        pytest.param(call("ghost"), "unknown tool: 'ghost'", id="an-unknown-tool"),
        pytest.param(
            call("explode"), "Error calling tool 'explode': kaboom", id="an-mcp-tool-that-fails"
        ),
    ],
)
async def test_a_call_that_cannot_run_is_told_to_the_model_and_the_run_goes_on(
    turn: Turn, told: str, models: Models, tmp_path: Path
) -> None:
    models.script([turn, say("done")])

    async with mcp_server(failing_tools()) as server:
        env = workspace_env(tmp_path / "ws", capabilities=(server,))
        run = await run_task(env, OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus")))

    assert (run.trace.status, run.trace.content) == ("completed", "done")
    assert told in json.dumps(models.requests()[-1].body)


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
        pytest.param("length", replace(shell("true"), truncated=True), id="a-capped-reply"),
        pytest.param("malformed_tool_call", CUT_OFF, id="a-malformed-call"),
    ],
)
async def test_stop_on_ends_the_run_at_the_call_it_names(
    reason: StopCondition, turn: Turn, models: Models, tmp_path: Path
) -> None:
    models.script([turn])

    run = await run_task(
        workspace_env(tmp_path / "ws"),
        OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus", stop_on={reason})),
    )

    assert (run.trace.status, run.trace.stop_reason) == ("completed", reason)
    assert len(models.requests()) == 1


async def test_the_step_budget_ends_a_model_that_never_finishes(
    models: Models, tmp_path: Path
) -> None:
    models.respond(lambda request: shell("true"))

    run = await run_task(
        workspace_env(tmp_path / "ws"),
        OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus", max_steps=2)),
    )

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
    turns = [shell("true")] + [shell("true") if name == "shell" else call(name) for name in calls]
    env = workspace_env(tmp_path / "ws")
    relays: list[FlakyRelay] = []

    async with AsyncExitStack() as stack:

        @env.initialize
        async def reach_the_workspace_through_a_relay() -> None:
            shell = env.capability("shell")
            address = urlsplit(shell.url)
            assert address.hostname is not None and address.port is not None
            workspace = await stack.enter_async_context(relay(address.hostname, address.port))
            relays.append(workspace)
            env.add_capability(replace(shell, url=f"ssh://127.0.0.1:{workspace.port}"))

        def answer(request: ModelRequest) -> Turn:
            if request.turn == 1:
                (workspace,) = relays
                workspace.refuse(sys.maxsize)
                workspace.sever()
            return turns[request.turn] if request.turn < len(turns) else say("done")

        models.respond(answer)
        run = await run_task(
            env, OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus", max_steps=10))
        )

    assert run.trace.status == status
    if status == "error":
        assert run.trace.error == "SSH tool failure limit reached after 3 consecutive errors"


RESPONDER = "You are an assistant that helps determine"


@pytest.mark.parametrize(
    ("decision", "content"),
    [
        pytest.param(say("CONTINUE"), "finished", id="continue"),
        pytest.param(say("STOP"), "Shall I go on?", id="stop"),
        pytest.param(fail(400), "Shall I go on?", id="the-responder-fails"),
    ],
)
async def test_auto_respond_answers_a_question_or_lets_the_run_end(
    decision: Turn, content: str, models: Models, tmp_path: Path
) -> None:
    """The auto-responder is its own Responses call beside the agent's own requests."""
    decisions = iter([decision, say("STOP")])
    replies = [say("Shall I go on?"), say("finished")]
    models.respond(
        lambda request: (
            next(decisions)
            if str(request.body.get("instructions", "")).startswith(RESPONDER)
            else replies[request.turn]
        )
    )

    run = await run_task(
        workspace_env(tmp_path / "ws"),
        OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus", auto_respond=True)),
    )

    assert run.trace.content == content


# ─── the tool timeout ──────────────────────────────────────────────────


def slow_tools() -> FastMCP:
    server = FastMCP("slow")

    @server.tool()
    async def wait() -> str:
        """Answer after two seconds."""
        await asyncio.sleep(2.0)
        return "waited"

    return server


async def test_a_shell_call_past_the_tool_timeout_is_cut_off_and_the_run_goes_on(
    models: Models, tmp_path: Path
) -> None:
    root = tmp_path / "ws"
    models.script([shell("sleep 3; touch late"), shell("touch quick"), say("done")])

    run = await run_task(
        workspace_env(root),
        OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus", tool_timeout_seconds=1)),
    )
    await asyncio.sleep(3)

    assert run.trace.status == "completed"
    assert "timed out after 1s" in json.dumps(models.requests()[1].body)
    assert ((root / "late").exists(), (root / "quick").exists()) == (False, True)


async def test_an_mcp_tool_is_left_to_its_own_time(models: Models, tmp_path: Path) -> None:
    models.script([call("wait"), say("done")])

    async with mcp_server(slow_tools()) as server:
        env = workspace_env(tmp_path / "ws", capabilities=(server,))
        run = await run_task(
            env, OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus", tool_timeout_seconds=1))
        )

    assert run.trace.status == "completed"
    assert "waited" in json.dumps(models.requests()[-1].body)


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
    models.script([call("bash", command="mkfifo pipe"), edit, say("done")])

    run = await run_task(
        workspace_env(root),
        ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6", tool_timeout_seconds=1)),
    )

    assert run.trace.status == "completed"
    assert "str_replace_based_edit_tool timed out after 1s" in json.dumps(
        models.requests()[-1].body
    )


# ─── MCP tool names ────────────────────────────────────────────────────


def named(name: str, *tools: str) -> FastMCP:
    server = FastMCP(name)
    for tool in tools:
        server.tool(name=tool, description=f"Run {tool}.")(lambda: tool)
    return server


async def test_mcp_tools_from_several_servers_are_qualified_within_64_characters(
    models: Models, tmp_path: Path
) -> None:
    models.script([say("done")])
    long = "lookup_" * 10

    async with (
        mcp_server(named("company.tools", long + "a"), name="company.tools") as dotted,
        mcp_server(named("company-tools", long + "b"), name="company-tools") as dashed,
    ):
        env = workspace_env(tmp_path / "ws", capabilities=(dotted, dashed))
        await run_task(env, OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus")))

    (request,) = models.requests()
    qualified = [name for name in request.tools if long[:20] in name]
    assert len(qualified) == 2 and all(len(name) <= 64 for name in qualified)


async def test_two_mcp_tools_qualified_to_one_name_fail_the_run(
    models: Models, tmp_path: Path
) -> None:
    models.script([say("done")])

    async with (
        mcp_server(named("a", "b__c"), name="a") as first,
        mcp_server(named("a__b", "c"), name="a__b") as second,
    ):
        env = workspace_env(tmp_path / "ws", capabilities=(first, second))
        run = await run_task(env, OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus")))

    assert run.trace.error == (
        "[agent loop] ValueError: MCP tool name collision after qualification: 'a__b__c'"
    )
    assert models.requests() == []
