"""How an agent's rollout ends, and what the model hears when a call goes wrong.

Each row scripts a provider over a served workspace and checks the trace's status,
stop reason and error: natural finishes, step budgets, token caps, malformed and
unknown calls, failing tools, provider faults and retries, lost workspaces and
timeouts. Where the model is told something, the row checks what it was told.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import pytest
from fastmcp import FastMCP

from hud.agents import ClaudeAgent
from hud.agents.types import ClaudeConfig
from tests.agents.support import chat, claude, gemini, mcp_server, openai, run_task, workspace_env
from tests.harness import (
    Reply,
    Turn,
    call,
    fail,
    interrupted,
    relay,
    say,
    shell_call,
    stream_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from hud.eval import Run
    from tests.harness import FlakyRelay, HudEnv, ModelRequest, Models


def told(requests: list[ModelRequest]) -> list[str]:
    """Every tool result text the model was sent, oldest first."""
    if requests and requests[-1].protocol == "responses":
        return [text for request in requests for text in request.tool_results]
    return requests[-1].tool_results if requests else []


def ending(run: Run) -> tuple[Any, ...]:
    trace = run.trace
    return (trace.status, trace.stop_reason, trace.is_truncated, trace.error)


def sequence(*turns: Turn) -> Callable[[ModelRequest], Turn]:
    """Answer requests with ``turns`` in arrival order, whatever they carry."""
    remaining = iter(turns)
    return lambda _request: next(remaining)


@dataclass(frozen=True)
class Row:
    agent: Callable[[], Agent]
    turns: list[Turn]
    ending: tuple[Any, ...]
    told: list[str] | None = None


MALFORMED = Turn(
    native=(
        {
            "id": "call_cut",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ec'},
        },
    )
)
DONE = ("completed", "done", False, None)
CAPPED = ("completed", "length", True, None)
MALFORMED_GEMINI = (
    "error",
    "malformed_tool_call",
    True,
    "Provider returned a malformed function call",
)

ROWS = {
    "done": Row(chat(), [say("done")], DONE),
    "step-budget-spent": Row(
        chat(max_steps=2),
        [call("bash", command="true"), call("bash", command="true")],
        ("completed", "max_steps", True, None),
    ),
    "chat-token-cap": Row(chat(), [Turn(text="part", truncated=True)], CAPPED),
    "responses-token-cap": Row(openai(), [Turn(text="part", truncated=True)], CAPPED),
    "claude-token-cap": Row(claude(), [Turn(text="part", truncated=True)], CAPPED),
    "gemini-token-cap": Row(gemini(), [Turn(text="part", truncated=True)], CAPPED),
    "malformed-call-answered": Row(
        chat(),
        [MALFORMED, say("recovered")],
        DONE,
        [
            "the arguments for this 'bash' call arrived incomplete (cut off mid-generation or "
            'invalid JSON), so it was not executed. Received: \'{"command": "ec\'. Re-issue '
            "the call in full."
        ],
    ),
    "malformed-call-stops": Row(
        chat(stop_on={"malformed_tool_call"}),
        [MALFORMED],
        ("completed", "malformed_tool_call", True, None),
    ),
    "token-capped-call-stops": Row(
        chat(stop_on={"length"}),
        [Turn(tool_calls=call("bash", command="ls").tool_calls, truncated=True)],
        CAPPED,
    ),
    "unknown-tool": Row(chat(), [call("ghost"), say("done")], DONE, ["unknown tool: 'ghost'"]),
    "tool-raises": Row(
        chat(), [call("bash", command=""), say("done")], DONE, ["tool error: command is required"]
    ),
    "provider-rejects": Row(
        chat(),
        [fail(400, "bad request")],
        (
            "error",
            None,
            False,
            "Error code: 400 - {'error': {'type': 'error', 'message': 'bad request'}}",
        ),
    ),
    "gemini-malformed-function-call": Row(
        gemini(), [Turn(finish="MALFORMED_FUNCTION_CALL")], MALFORMED_GEMINI
    ),
    "gemini-malformed-function-call-auto-respond": Row(
        gemini(auto_respond=True), [Turn(finish="MALFORMED_FUNCTION_CALL")], MALFORMED_GEMINI
    ),
    "gemini-no-candidates": Row(
        gemini(),
        [Turn(fail=Reply(json={"candidates": []}))],
        ("error", None, False, "Gemini returned no candidates for model gemini-3.1-pro-preview"),
    ),
    "responses-empty-shell-call-resampled": Row(openai(), [shell_call(), say("done")], DONE),
    "responses-empty-shell-call-on-last-step": Row(
        openai(max_steps=1),
        [shell_call()],
        (
            "error",
            None,
            False,
            "Model emitted shell_call(s) with an empty commands array (call_shell); continuing "
            "from response resp_1 would fail with OpenAI 500s, so it was not committed to the "
            "run state.",
        ),
    ),
}


@pytest.mark.parametrize("row", ROWS.values(), ids=ROWS.keys())
async def test_a_rollout_ends_for_the_reason_its_last_turn_gives(
    row: Row, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.respond(sequence(*row.turns))

    run = await run_task(workspace_env(tmp_path / "ws"), row.agent())

    assert ending(run) == row.ending
    if row.told is not None:
        assert told(models.requests()) == row.told


async def test_a_malformed_call_replays_as_empty_arguments(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    """History stays parseable for the provider after a call that never parsed."""
    hud_env.set(HUD_API_KEY="k")
    models.script([MALFORMED, say("recovered")])

    await run_task(workspace_env(tmp_path / "ws"), chat()())

    replayed = models.requests()[-1].body["messages"][1]["tool_calls"]
    assert replayed == [
        {"id": "call_cut", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
    ]


async def test_a_discarded_responses_turn_does_not_advance_the_response_chain(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.respond(sequence(shell_call("true"), shell_call(), say("done")))

    run = await run_task(workspace_env(tmp_path / "ws"), openai()())

    chain = [request.body.get("previous_response_id") for request in models.requests()]
    assert (run.trace.content, chain) == ("done", [None, "resp_1", "resp_1"])


def responder(*decisions: Turn) -> Callable[[ModelRequest], Turn]:
    """Chat turns for the agent; the auto-responder's Responses requests get ``decisions``."""
    agent_turns = [say("Shall I go on with the rest?"), say("finished")]
    answers = iter(decisions)

    def answer(request: ModelRequest) -> Turn:
        if request.protocol == "responses":
            return next(answers)
        return agent_turns[request.turn]

    return answer


AUTO_RESPOND = {
    "continue-then-stop": (responder(say("CONTINUE"), say("STOP")), ("finished", "CONTINUE")),
    "stop": (responder(say("STOP")), ("Shall I go on with the rest?", "Do the task.")),
    "responder-fails": (responder(fail(400)), ("Shall I go on with the rest?", "Do the task.")),
}


@pytest.mark.parametrize(("policy", "expected"), AUTO_RESPOND.values(), ids=AUTO_RESPOND.keys())
async def test_auto_respond_answers_a_question_or_lets_the_rollout_end(
    policy: Callable[[ModelRequest], Turn],
    expected: tuple[str, str],
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.respond(policy)

    run = await run_task(workspace_env(tmp_path / "ws"), chat(auto_respond=True)())

    (*_, last) = [request for request in models.requests() if request.protocol == "chat"]
    (said,) = last.body["messages"][-1]["content"]
    assert (run.trace.content, said["text"]) == expected


def cutter(relays: list[FlakyRelay]) -> FastMCP:
    server = FastMCP("cutter")

    @server.tool()
    def cut_workspace() -> str:
        """Drop the connection to the workspace for good."""
        (workspace,) = relays
        workspace.refuse(sys.maxsize)
        workspace.sever()
        return "cut"

    return server


LOST = {
    "three-lost-calls-in-a-row": (
        ["bash", "bash", "bash"],
        ("error", None, False, "SSH tool failure limit reached after 3 consecutive errors"),
    ),
    "another-result-resets-the-count": (["bash", "ghost", "bash", "bash"], DONE),
}


@pytest.mark.parametrize(("tools", "expected"), LOST.values(), ids=LOST.keys())
async def test_a_lost_workspace_ends_the_rollout_after_three_consecutive_failures(
    tools: list[str], expected: tuple[Any, ...], models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    root = tmp_path / "ws"
    script = [call("cut_workspace"), *(call(name, command="true") for name in tools)]
    models.script([*script, say("done")])
    relays: list[FlakyRelay] = []
    env = workspace_env(root)

    async with AsyncExitStack() as stack:

        @env.initialize
        async def reach_the_workspace_through_a_relay() -> None:
            shell = env.capability("shell")
            address = urlsplit(shell.url)
            assert address.hostname is not None and address.port is not None
            workspace = await stack.enter_async_context(relay(address.hostname, address.port))
            relays.append(workspace)
            env.add_capability(replace(shell, url=f"ssh://127.0.0.1:{workspace.port}"))

        cut = await stack.enter_async_context(mcp_server(cutter(relays)))
        env.add_capability(cut)
        run = await run_task(env, chat(max_steps=10)())

    assert ending(run) == expected
    assert told(models.requests())[1] == "tool error: SSH reconnect failed after 3 attempts"


async def test_a_slow_shell_command_times_out_and_the_rollout_goes_on(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    root = tmp_path / "ws"
    models.script(
        [
            call("bash", command="touch started; sleep 1.5; touch late"),
            call("bash", command="sleep 0.1; touch quick"),
            say("done"),
        ]
    )
    env = workspace_env(
        root,
        settle=1.5,
        passes=lambda root: all(
            [(root / "started").exists(), (root / "quick").exists(), not (root / "late").exists()]
        ),
    )

    run = await run_task(env, claude(tool_timeout_seconds=0.5)())

    assert run.reward == 1.0
    assert told(models.requests()) == [
        "Error: bash timed out after 0.5s; retry with a shorter command",
        "$ sleep 0.1; touch quick\n\n(exit 0)",
    ]


async def test_an_editor_blocked_on_its_file_times_out(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    root = tmp_path / "ws"
    models.script(
        [
            call("bash", command="mkfifo pipe"),
            call(
                "str_replace_based_edit_tool",
                command="str_replace",
                path=f"{root}/pipe",
                old_str="a",
                new_str="b",
            ),
            say("done"),
        ]
    )

    run = await run_task(workspace_env(root), claude(tool_timeout_seconds=0.5)())

    assert run.trace.status == "completed"
    assert told(models.requests())[-1] == (
        "Error: str_replace_based_edit_tool timed out after 0.5s; retry with a shorter command"
    )


def slow_tools() -> FastMCP:
    server = FastMCP("slow")

    @server.tool()
    async def slow() -> str:
        """Answer after a second."""
        await asyncio.sleep(1.0)
        return "slow answer"

    return server


async def test_the_tool_timeout_covers_workspace_tools_only(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([call("slow"), say("done")])

    async with mcp_server(slow_tools()) as tools:
        env = workspace_env(tmp_path / "ws", capabilities=(tools,))
        await run_task(env, claude(tool_timeout_seconds=0.5)())

    assert models.requests()[-1].tool_results == ["slow answer"]


async def test_the_agent_timeout_stops_a_rollout_waiting_on_its_provider(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([Turn(fail=Reply(status=500, delay=2.0))])

    run = await run_task(workspace_env(tmp_path / "ws"), chat(timeout_seconds=0.5)())

    assert (run.trace.status, run.trace.stop_reason) == ("error", "timeout")


NO_WAIT = {"retry-after-ms": "0"}


def bad_json(fragment: str) -> Turn:
    """A tool call whose streamed input is not JSON."""
    return Turn(
        native=({"type": "tool_use", "id": "toolu_bad", "name": "bash", "input": fragment},)
    )


BAD_JSON = bad_json('{"command": ls}')
UPSTREAM = "{'type': 'error', 'error': {'type': '%s', 'message': 'scripted %s'}}"
RECOVERED = ("completed", "recovered", None)
TOLD = "previous tool-call arguments were invalid JSON"
STREAMS = {
    "transient-error-then-answer": (
        [stream_error("upstream_error", headers=NO_WAIT), say("recovered")],
        (*RECOVERED, 2, False),
    ),
    "two-transient-errors-then-answer": (
        [
            stream_error("timeout_error", headers=NO_WAIT),
            stream_error("overloaded_error", headers=NO_WAIT),
            say("recovered"),
        ],
        (*RECOVERED, 3, False),
    ),
    "transient-errors-exhaust-the-retries": (
        [stream_error("gateway_timeout", headers=NO_WAIT)] * 3,
        ("error", None, UPSTREAM % ("gateway_timeout", "gateway_timeout"), 3, False),
    ),
    "a-request-error-is-not-retried": (
        [stream_error("invalid_request_error", headers=NO_WAIT)],
        ("error", None, UPSTREAM % ("invalid_request_error", "invalid_request_error"), 1, False),
    ),
    "an-interrupted-stream-is-retried": ([interrupted(), say("recovered")], (*RECOVERED, 2, False)),
    "invalid-tool-json-is-retried-silently": ([BAD_JSON, say("recovered")], (*RECOVERED, 2, False)),
    "invalid-tool-json-twice-is-pointed-out": (
        [BAD_JSON, BAD_JSON, say("recovered")],
        (*RECOVERED, 3, True),
    ),
    "invalid-tool-json-three-times-fails-the-run": (
        [BAD_JSON] * 3,
        (
            "error",
            None,
            "Unable to parse tool parameter JSON from model. Please retry your request or adjust "
            'your prompt. Error: expected value at line 1 column 13. JSON: {"command": ls}',
            3,
            True,
        ),
    ),
}


@pytest.mark.parametrize(("turns", "expected"), STREAMS.values(), ids=STREAMS.keys())
async def test_claude_retries_what_the_stream_lets_it_retry(
    turns: list[Turn], expected: tuple[Any, ...], models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.respond(sequence(*turns))

    run = await run_task(
        workspace_env(tmp_path / "ws"), ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6"))
    )

    requests = models.requests()
    told_invalid = TOLD in str(requests[-1].body["messages"])
    assert (
        run.trace.status,
        run.trace.content,
        run.trace.error,
        len(requests),
        told_invalid,
    ) == expected
