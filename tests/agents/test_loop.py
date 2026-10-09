# ruff: noqa: E501 -- snapshots quote product messages verbatim
"""How a tool-agent rollout ends, and what the model hears when a call goes wrong.

Rows script a provider over a served workspace and pin the trace's status,
stop reason and error, the steps it exported, and the tool results the
model was sent: natural finishes, step budgets, token caps, malformed and
unknown calls, failing tools, provider faults, lost workspaces and timeouts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import pytest
from fastmcp import FastMCP
from inline_snapshot import snapshot

from hud.agents import ClaudeAgent, GeminiAgent, OpenAIAgent, OpenAIChatAgent
from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIChatConfig, OpenAIConfig
from tests.agents.support import Relay, mcp_server, run_task, workspace_env
from tests.harness import Reply, Turn, call, fail, say, shell_call, steps

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from hud.eval import Run
    from tests.harness import HudEnv, ModelRequest, Models


def chat(**config: Any) -> Callable[[], Agent]:
    return lambda: OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus", **config))


def responses(**config: Any) -> Callable[[], Agent]:
    return lambda: OpenAIAgent(OpenAIConfig(model="gpt-5.6", **config))


def claude(**config: Any) -> Callable[[], Agent]:
    return lambda: ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6", **config))


def gemini(**config: Any) -> Callable[[], Agent]:
    return lambda: GeminiAgent(GeminiConfig(model="gemini-3.1-pro-preview", **config))


def result_texts(requests: list[ModelRequest]) -> list[str]:
    """Every tool result text the model was sent, oldest first."""
    if requests and requests[-1].protocol == "responses":
        return [text for request in requests for text in request.tool_results]
    return requests[-1].tool_results if requests else []


def ending(run: Run, requests: list[ModelRequest]) -> dict[str, Any]:
    trace = run.trace
    return {
        "status": trace.status,
        "stop_reason": trace.stop_reason,
        "truncated": trace.is_truncated,
        "error": trace.error,
        "steps": [step["source"] for step in steps(run.trace_id)],
        "results": result_texts(requests),
    }


def sequence(*turns: Turn) -> Callable[[ModelRequest], Turn]:
    """Answer requests with ``turns`` in arrival order, whatever they carry."""
    remaining = iter(turns)
    return lambda _request: next(remaining)


@dataclass(frozen=True)
class Row:
    agent: Callable[[], Agent]
    turns: list[Turn]
    ending: Any


MALFORMED = Turn(
    native=(
        {
            "id": "call_cut",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ec'},
        },
    )
)

ROWS = {
    "done": Row(
        chat(),
        [say("done")],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "done",
                "truncated": False,
                "error": None,
                "steps": ["task", "user", "agent", "task"],
                "results": [],
            }
        ),
    ),
    "step-budget-spent": Row(
        chat(max_steps=2),
        [call("bash", command="true"), call("bash", command="true")],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "max_steps",
                "truncated": True,
                "error": None,
                "steps": ["task", "user", "agent", "tool", "agent", "tool", "task"],
                "results": [
                    """\
$ true

(exit 0)\
"""
                ],
            }
        ),
    ),
    "chat-token-cap": Row(
        chat(),
        [Turn(text="part", truncated=True)],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "length",
                "truncated": True,
                "error": None,
                "steps": ["task", "user", "agent", "task"],
                "results": [],
            }
        ),
    ),
    "responses-token-cap": Row(
        responses(),
        [Turn(text="part", truncated=True)],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "length",
                "truncated": True,
                "error": None,
                "steps": ["task", "user", "agent", "task"],
                "results": [],
            }
        ),
    ),
    "claude-token-cap": Row(
        claude(),
        [Turn(text="part", truncated=True)],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "length",
                "truncated": True,
                "error": None,
                "steps": ["task", "user", "agent", "task"],
                "results": [],
            }
        ),
    ),
    "gemini-token-cap": Row(
        gemini(),
        [Turn(text="part", truncated=True)],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "length",
                "truncated": True,
                "error": None,
                "steps": ["task", "user", "agent", "task"],
                "results": [],
            }
        ),
    ),
    "malformed-call-answered": Row(
        chat(),
        [MALFORMED, say("recovered")],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "done",
                "truncated": False,
                "error": None,
                "steps": ["task", "user", "agent", "tool", "agent", "task"],
                "results": [
                    "the arguments for this 'bash' call arrived incomplete (cut off mid-generation or invalid JSON), so it was not executed. Received: '{\"command\": \"ec'. Re-issue the call in full."
                ],
            }
        ),
    ),
    "malformed-call-stops": Row(
        chat(stop_on={"malformed_tool_call"}),
        [MALFORMED],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "malformed_tool_call",
                "truncated": True,
                "error": None,
                "steps": ["task", "user", "agent", "task"],
                "results": [],
            }
        ),
    ),
    "token-capped-call-stops": Row(
        chat(stop_on={"length"}),
        [Turn(tool_calls=call("bash", command="ls").tool_calls, truncated=True)],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "length",
                "truncated": True,
                "error": None,
                "steps": ["task", "user", "agent", "task"],
                "results": [],
            }
        ),
    ),
    "unknown-tool": Row(
        chat(),
        [call("ghost"), say("done")],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "done",
                "truncated": False,
                "error": None,
                "steps": ["task", "user", "agent", "tool", "agent", "task"],
                "results": ["unknown tool: 'ghost'"],
            }
        ),
    ),
    "tool-raises": Row(
        chat(),
        [call("bash", command=""), say("done")],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "done",
                "truncated": False,
                "error": None,
                "steps": ["task", "user", "agent", "tool", "agent", "task"],
                "results": ["tool error: command is required"],
            }
        ),
    ),
    "failed-command": Row(
        chat(),
        [call("bash", command="exit 3"), say("done")],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "done",
                "truncated": False,
                "error": None,
                "steps": ["task", "user", "agent", "tool", "agent", "task"],
                "results": [
                    """\
$ exit 3

(exit 3)\
"""
                ],
            }
        ),
    ),
    "provider-rejects": Row(
        chat(),
        [fail(400, "bad request")],
        snapshot(
            {
                "status": "error",
                "stop_reason": None,
                "truncated": False,
                "error": "Error code: 400 - {'error': {'type': 'error', 'message': 'bad request'}}",
                "steps": ["task", "user", "system", "task"],
                "results": [],
            }
        ),
    ),
    "gemini-malformed-function-call": Row(
        gemini(),
        [Turn(finish="MALFORMED_FUNCTION_CALL")],
        snapshot(
            {
                "status": "error",
                "stop_reason": "malformed_tool_call",
                "truncated": True,
                "error": "Provider returned a malformed function call",
                "steps": ["task", "user", "agent", "task"],
                "results": [],
            }
        ),
    ),
    "gemini-malformed-function-call-auto-respond": Row(
        gemini(auto_respond=True),
        [Turn(finish="MALFORMED_FUNCTION_CALL")],
        snapshot(
            {
                "status": "error",
                "stop_reason": "malformed_tool_call",
                "truncated": True,
                "error": "Provider returned a malformed function call",
                "steps": ["task", "user", "agent", "task"],
                "results": [],
            }
        ),
    ),
    "gemini-no-candidates": Row(
        gemini(),
        [Turn(fail=Reply(json={"candidates": []}))],
        snapshot(
            {
                "status": "error",
                "stop_reason": None,
                "truncated": False,
                "error": "Gemini returned no candidates for model gemini-3.1-pro-preview",
                "steps": ["task", "user", "system", "task"],
                "results": [],
            }
        ),
    ),
    "responses-empty-shell-call-resampled": Row(
        responses(),
        [shell_call(), say("done")],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "done",
                "truncated": False,
                "error": None,
                "steps": ["task", "user", "agent", "task"],
                "results": [],
            }
        ),
    ),
    "responses-empty-shell-call-on-last-step": Row(
        responses(max_steps=1),
        [shell_call()],
        snapshot(
            {
                "status": "error",
                "stop_reason": None,
                "truncated": False,
                "error": "Model emitted shell_call(s) with an empty commands array (call_shell); continuing from response resp_1 would fail with OpenAI 500s, so it was not committed to the run state.",
                "steps": ["task", "user", "system", "task"],
                "results": [],
            }
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

    assert ending(run, models.requests()) == row.ending


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

    run = await run_task(workspace_env(tmp_path / "ws"), responses()())

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
    "continue-then-stop": (
        responder(say("CONTINUE"), say("STOP")),
        snapshot(
            {
                "content": "finished",
                "steps": ["task", "user", "agent", "user", "agent", "task"],
                "last_user_message": [{"type": "text", "text": "CONTINUE"}],
            }
        ),
    ),
    "stop": (
        responder(say("STOP")),
        snapshot(
            {
                "content": "Shall I go on with the rest?",
                "steps": ["task", "user", "agent", "task"],
                "last_user_message": [{"type": "text", "text": "Do the task."}],
            }
        ),
    ),
    "responder-fails": (
        responder(fail(400)),
        snapshot(
            {
                "content": "Shall I go on with the rest?",
                "steps": ["task", "user", "agent", "task"],
                "last_user_message": [{"type": "text", "text": "Do the task."}],
            }
        ),
    ),
}


@pytest.mark.parametrize(("policy", "expected"), AUTO_RESPOND.values(), ids=AUTO_RESPOND.keys())
async def test_auto_respond_answers_a_question_or_lets_the_rollout_end(
    policy: Callable[[ModelRequest], Turn],
    expected: Any,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.respond(policy)

    run = await run_task(workspace_env(tmp_path / "ws"), chat(auto_respond=True)())

    agent_requests = [request for request in models.requests() if request.protocol == "chat"]
    observed = {
        "content": run.trace.content,
        "steps": [step["source"] for step in steps(run.trace_id)],
        "last_user_message": agent_requests[-1].body["messages"][-1]["content"],
    }
    assert observed == expected


def cutter(relay: Relay) -> FastMCP:
    server = FastMCP("cutter")

    @server.tool()
    def cut_workspace() -> str:
        """Drop the connection to the workspace."""
        relay.cut()
        return "cut"

    return server


LOST = {
    "three-lost-calls-in-a-row": (
        ["bash", "bash", "bash"],
        snapshot(
            {
                "status": "error",
                "stop_reason": None,
                "truncated": False,
                "error": "SSH tool failure limit reached after 3 consecutive errors",
                "steps": [
                    "task",
                    "user",
                    "agent",
                    "tool",
                    "agent",
                    "tool",
                    "agent",
                    "tool",
                    "agent",
                    "tool",
                    "system",
                    "task",
                ],
                "results": [
                    "cut",
                    "tool error: SSH reconnect failed after 3 attempts",
                    "tool error: SSH reconnect failed after 3 attempts",
                ],
            }
        ),
    ),
    "another-result-resets-the-count": (
        ["bash", "ghost", "bash", "bash"],
        snapshot(
            {
                "status": "completed",
                "stop_reason": "done",
                "truncated": False,
                "error": None,
                "steps": [
                    "task",
                    "user",
                    "agent",
                    "tool",
                    "agent",
                    "tool",
                    "agent",
                    "tool",
                    "agent",
                    "tool",
                    "agent",
                    "tool",
                    "agent",
                    "task",
                ],
                "results": [
                    "cut",
                    "tool error: SSH reconnect failed after 3 attempts",
                    "unknown tool: 'ghost'",
                    "tool error: SSH reconnect failed after 3 attempts",
                    "tool error: SSH reconnect failed after 3 attempts",
                ],
            }
        ),
    ),
}


@pytest.mark.parametrize(("tools", "expected"), LOST.values(), ids=LOST.keys())
async def test_a_lost_workspace_ends_the_rollout_after_three_consecutive_failures(
    tools: list[str], expected: Any, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    root = tmp_path / "ws"
    script = [call("cut_workspace"), *(call(name, command="true") for name in tools)]
    models.script([*script, say("done")])
    relay = Relay()
    env = workspace_env(root)

    @env.initialize
    async def reach_the_workspace_through_the_relay() -> None:
        shell = env.capability("shell")
        address = urlsplit(shell.url)
        assert address.hostname is not None and address.port is not None
        port = await relay.start(address.hostname, address.port)
        env.add_capability(replace(shell, url=f"ssh://127.0.0.1:{port}"))

    async with mcp_server(cutter(relay)) as cut:
        env.add_capability(cut)
        run = await run_task(env, chat(max_steps=10)())

    assert ending(run, models.requests()) == expected


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
    assert ending(run, models.requests())["results"] == [
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
    assert ending(run, models.requests())["results"][-1] == (
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
