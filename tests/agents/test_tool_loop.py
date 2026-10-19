"""How an agent's tool calls run, and what the model is told when one goes wrong.

Every family's calls must act on the workspace and land on the trace as tool
steps. The provider tool-call agents share hud's own tool loop, so its rules run
over each of them: an unknown or failing call is told to the model and the run
goes on, a malformed call is explained, ``stop_on`` ends the run at the call it
names, and a workspace that drops away ends the run after three failures in a row.
"""

from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from dataclasses import replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import pytest
from fastmcp import FastMCP

from hud.agents.types import ToolStep
from tests.agents.families import (
    CLAUDE,
    GEMINI,
    OPENAI,
    TOOL_FAMILIES,
    NotApplicable,
    ToolFamily,
    over_families,
    over_family_cases,
)
from tests.support import Turn, call, fail, mcp_server, relay, run_task, say, workspace_env

if TYPE_CHECKING:
    from pathlib import Path

    from tests.agents.families import Driver
    from tests.support import FlakyRelay, HudEnv, ModelRequest, Models

OWN_LOOP = NotApplicable("runs its own tool loop inside its CLI or SDK")

# The name each family's shell call is recorded under.
SHELL_STEPS: dict[str, str | NotApplicable] = {
    "claude": "bash",
    "openai": "shell",
    "gemini": "run_shell_command",
    "openai_compatible": "bash",
    "claude_cli": "Bash",
    "codex_cli": "shell",
    "browser_use": NotApplicable(
        "browser-use acts on the browser and records one coarse step; "
        "test_browser_use_wire.py checks it"
    ),
}


@pytest.mark.parametrize(("family", "step"), over_families(SHELL_STEPS))
async def test_a_shell_call_acts_on_the_workspace_and_lands_on_the_trace(
    family: str, step: str, driver: Driver
) -> None:
    driver.shell("touch made", then="done")

    run = await driver.run(driver.agent())

    calls = [s.call.name for s in run.trace.steps if isinstance(s, ToolStep) and s.call]
    assert (run.trace.status, run.trace.content) == ("completed", "done")
    assert ((driver.workspace / "made").exists(), calls) == (True, [step])


def failing_tools() -> FastMCP:
    server = FastMCP("tools")

    @server.tool
    def explode() -> str:
        """Fail."""
        raise ValueError("kaboom")

    return server


def tool_cases(failing: Turn, told: str) -> dict[str, tuple[Turn, str]]:
    return {
        "unknown-tool": (call("ghost"), "unknown tool: 'ghost'"),
        "own-tool-fails": (failing, told),
        "mcp-tool-fails": (call("explode"), "Error calling tool 'explode': kaboom"),
    }


FAILED_CALLS: dict[str, dict[str, tuple[Turn, str]] | NotApplicable] = {
    "claude": tool_cases(
        call("str_replace_based_edit_tool", command="bogus", path="/x"),
        "unknown editor command: 'bogus'",
    ),
    "openai": tool_cases(
        Turn(native=({**OPENAI.shell("true").native[0], "action": {"commands": 5}},)),
        "commands must be a list of strings",
    ),
    "gemini": tool_cases(call("run_shell_command", command=""), "tool error: command is required"),
    "openai_compatible": tool_cases(
        call("bash", command=""),
        "tool error: command is required",
    ),
    "claude_cli": OWN_LOOP,
    "codex_cli": OWN_LOOP,
    "browser_use": OWN_LOOP,
}


@pytest.mark.parametrize(("family", "case"), over_family_cases(FAILED_CALLS))
async def test_a_call_that_cannot_run_is_told_to_the_model_and_the_run_goes_on(
    family: str, case: tuple[Turn, str], models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    turn, told = case
    models.script([turn, say("done")])
    tools = TOOL_FAMILIES[family]

    async with mcp_server(failing_tools()) as server:
        env = workspace_env(tmp_path / "ws", capabilities=(server,))
        run = await run_task(env, tools.agent_cls(tools.config()))

    assert (run.trace.status, run.trace.content) == ("completed", "done")
    assert told in json.dumps(models.requests()[-1].body)


CUT_OFF_CHAT = Turn(
    native=(
        {
            "id": "call_cut",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ec'},
        },
    )
)
CUT_OFF_RESPONSES = Turn(
    native=(
        {
            "type": "function_call",
            "id": "fc_cut",
            "call_id": "call_cut",
            "name": "bash",
            "arguments": '{"command": "ec',
            "status": "completed",
        },
    )
)
BAD_JSON = Turn(
    native=({"type": "tool_use", "id": "toolu_bad", "name": "bash", "input": '{"command": ls}'},)
)
INCOMPLETE = "the arguments for this 'bash' call arrived incomplete"

# A call whose arguments are not valid JSON: the turns, and what the model is told.
MALFORMED: dict[str, tuple[list[Turn], str] | NotApplicable] = {
    "claude": (
        [BAD_JSON, BAD_JSON, say("done")],
        "Your previous tool-call arguments were invalid JSON.",
    ),
    "openai": ([CUT_OFF_RESPONSES, say("done")], INCOMPLETE),
    "gemini": NotApplicable(
        "Gemini reports a malformed call as its finish reason; test_provider_faults.py has it"
    ),
    "openai_compatible": ([CUT_OFF_CHAT, say("done")], INCOMPLETE),
    "claude_cli": OWN_LOOP,
    "codex_cli": OWN_LOOP,
    "browser_use": OWN_LOOP,
}


@pytest.mark.parametrize(("family", "case"), over_families(MALFORMED))
async def test_a_malformed_call_is_explained_to_the_model_and_the_run_goes_on(
    family: str, case: tuple[list[Turn], str], models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    turns, told = case
    models.queue(turns)
    tools = TOOL_FAMILIES[family]

    run = await run_task(workspace_env(tmp_path / "ws"), tools.agent_cls(tools.config()))

    assert (run.trace.status, run.trace.content) == ("completed", "done")
    assert told in json.dumps(models.requests()[-1].body)


def capped(family: ToolFamily) -> Turn:
    return replace(family.shell("true"), truncated=True)


STOP_ON: dict[str, dict[str, tuple[str, Turn]] | NotApplicable] = {
    "claude": {"token-capped": ("length", capped(CLAUDE))},
    "openai": {
        "token-capped": ("length", capped(OPENAI)),
        "malformed": ("malformed_tool_call", CUT_OFF_RESPONSES),
    },
    "gemini": {"token-capped": ("length", capped(GEMINI))},
    "openai_compatible": {
        "token-capped": ("length", capped(TOOL_FAMILIES["openai_compatible"])),
        "malformed": ("malformed_tool_call", CUT_OFF_CHAT),
    },
    "claude_cli": OWN_LOOP,
    "codex_cli": OWN_LOOP,
    "browser_use": OWN_LOOP,
}


@pytest.mark.parametrize(("family", "case"), over_family_cases(STOP_ON))
async def test_stop_on_ends_the_run_at_a_call_it_names(
    family: str, case: tuple[str, Turn], models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    reason, turn = case
    models.script([turn])
    tools = TOOL_FAMILIES[family]
    agent = tools.agent_cls(tools.config(stop_on={reason}))

    run = await run_task(workspace_env(tmp_path / "ws"), agent)

    assert (run.trace.status, run.trace.stop_reason) == ("completed", reason)
    assert len(models.requests()) == 1


LOST = {
    "three-lost-calls-in-a-row": (["shell", "shell", "shell"], "error"),
    "another-result-resets-the-count": (["shell", "ghost", "shell", "shell"], "completed"),
}
LOST_WORKSPACE: dict[str, dict[str, tuple[list[str], str]] | NotApplicable] = {
    **dict.fromkeys(TOOL_FAMILIES, LOST),
    "claude_cli": OWN_LOOP,
    "codex_cli": OWN_LOOP,
    "browser_use": OWN_LOOP,
}


@pytest.mark.parametrize(("family", "case"), over_family_cases(LOST_WORKSPACE))
async def test_a_lost_workspace_ends_the_run_after_three_consecutive_failures(
    family: str, case: tuple[list[str], str], models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    calls, status = case
    tools = TOOL_FAMILIES[family]
    turns = [tools.shell("true")] + [
        tools.shell("true") if name == "shell" else call(name) for name in calls
    ]
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
        run = await run_task(env, tools.agent_cls(tools.config(max_steps=10)))

    assert run.trace.status == status
    if status == "error":
        assert run.trace.error == "SSH tool failure limit reached after 3 consecutive errors"


RESPONDER = "You are an assistant that helps determine"
AUTO_RESPOND = {
    "continue": (say("CONTINUE"), "finished"),
    "stop": (say("STOP"), "Shall I go on?"),
    "responder-fails": (fail(400), "Shall I go on?"),
}


@pytest.mark.parametrize(
    ("family", "case"),
    over_family_cases(
        {
            **dict.fromkeys(TOOL_FAMILIES, AUTO_RESPOND),
            "claude_cli": OWN_LOOP,
            "codex_cli": OWN_LOOP,
            "browser_use": OWN_LOOP,
        }
    ),
)
async def test_auto_respond_answers_a_question_or_lets_the_run_end(
    family: str, case: tuple[Turn, str], models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    """The auto-responder is its own Responses call beside the agent's own requests."""
    hud_env.set(HUD_API_KEY="k")
    decision, content = case
    decisions = iter([decision, say("STOP")])
    replies = [say("Shall I go on?"), say("finished")]
    models.respond(
        lambda request: (
            next(decisions)
            if str(request.body.get("instructions", "")).startswith(RESPONDER)
            else replies[request.turn]
        )
    )
    tools = TOOL_FAMILIES[family]

    run = await run_task(
        workspace_env(tmp_path / "ws"), tools.agent_cls(tools.config(auto_respond=True))
    )

    assert run.trace.content == content
