"""What an agent does when its provider, a tool call or its workspace goes wrong.

A real model will not fail on demand, so a scripted one does here: provider
errors and cut-off streams, malformed and unknown calls, token caps, tools that
raise or hang, and a workspace that drops away. Each test checks how the run
ends and, where the model is told something, what it was told.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from dataclasses import replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import pytest
from fastmcp import FastMCP

from hud.agents import ClaudeAgent, GeminiAgent, OpenAIAgent, OpenAIChatAgent
from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIChatConfig, OpenAIConfig
from tests.agents.support import mcp_server, run_task, workspace_env
from tests.support import (
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
    from tests.support import FlakyRelay, HudEnv, ModelRequest, Models

CLAUDE = ClaudeConfig(model="claude-sonnet-4-6")
OPENAI = OpenAIConfig(model="gpt-5.6")
GEMINI = GeminiConfig(model="gemini-3.1-pro-preview")
CHAT = OpenAIChatConfig(model="qwen3.6-plus")
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
    ("agent", "turns", "error"),
    [
        pytest.param(
            lambda: OpenAIChatAgent(CHAT),
            [fail(400, "bad request")],
            "Error code: 400 - {'error': {'type': 'error', 'message': 'bad request'}}",
            id="chat-provider-rejects",
        ),
        pytest.param(
            lambda: GeminiAgent(GEMINI),
            [Turn(fail=Reply(json={"candidates": []}))],
            "Gemini returned no candidates for model gemini-3.1-pro-preview",
            id="gemini-no-candidates",
        ),
        pytest.param(
            lambda: GeminiAgent(GEMINI),
            [Turn(finish="MALFORMED_FUNCTION_CALL")],
            "Provider returned a malformed function call",
            id="gemini-malformed-function-call",
        ),
        pytest.param(
            lambda: OpenAIAgent(OPENAI.model_copy(update={"max_steps": 1})),
            [shell_call()],
            "Model emitted shell_call(s) with an empty commands array (call_shell); continuing "
            "from response resp_1 would fail with OpenAI 500s, so it was not committed to the "
            "run state.",
            id="responses-empty-shell-call-on-the-last-step",
        ),
    ],
)
async def test_a_provider_fault_ends_the_run_with_its_error(
    agent: Callable[[], Agent],
    turns: list[Turn],
    error: str,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.queue(turns)

    run = await run_task(workspace_env(tmp_path / "ws"), agent())

    assert (run.trace.status, run.trace.error) == ("error", error)


@pytest.mark.parametrize(
    "agent",
    [
        lambda: ClaudeAgent(CLAUDE),
        lambda: OpenAIAgent(OPENAI),
        lambda: GeminiAgent(GEMINI),
        lambda: OpenAIChatAgent(CHAT),
    ],
    ids=["claude", "openai", "gemini", "chat"],
)
async def test_a_reply_cut_at_the_token_cap_ends_the_run_as_truncated(
    agent: Callable[[], Agent], models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.queue([Turn(text="part", truncated=True)])

    run = await run_task(workspace_env(tmp_path / "ws"), agent())

    trace = run.trace
    assert (trace.status, trace.stop_reason, trace.is_truncated) == ("completed", "length", True)


TRANSIENT = {"retry-after-ms": "0"}
BAD_JSON = Turn(
    native=({"type": "tool_use", "id": "toolu_bad", "name": "bash", "input": '{"command": ls}'},)
)


@pytest.mark.parametrize(
    ("turns", "status", "requests"),
    [
        pytest.param(
            [stream_error("overloaded_error", headers=TRANSIENT), say("ok")],
            "completed",
            2,
            id="a-transient-error-is-retried",
        ),
        pytest.param(
            [stream_error("gateway_timeout", headers=TRANSIENT)] * 3,
            "error",
            3,
            id="three-transient-errors-end-the-run",
        ),
        pytest.param(
            [stream_error("invalid_request_error", headers=TRANSIENT)],
            "error",
            1,
            id="a-request-error-is-not-retried",
        ),
        pytest.param([interrupted(), say("ok")], "completed", 2, id="a-cut-stream-is-retried"),
        pytest.param([BAD_JSON, say("ok")], "completed", 2, id="invalid-tool-json-is-retried"),
        pytest.param([BAD_JSON] * 3, "error", 3, id="invalid-tool-json-three-times-ends-the-run"),
    ],
)
async def test_claude_retries_what_its_stream_lets_it_retry(
    turns: list[Turn], status: str, requests: int, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.queue(turns)

    run = await run_task(workspace_env(tmp_path / "ws"), ClaudeAgent(CLAUDE))

    assert (run.trace.status, len(models.requests())) == (status, requests)


async def test_claude_points_out_tool_json_it_could_not_parse_twice(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.queue([BAD_JSON, BAD_JSON, say("ok")])

    run = await run_task(workspace_env(tmp_path / "ws"), ClaudeAgent(CLAUDE))

    retried = models.requests()[-1].body["messages"][-1]["content"][0]["text"]
    assert run.trace.status == "completed"
    assert retried.startswith("Your previous tool-call arguments were invalid JSON.")


async def test_a_call_cut_off_mid_arguments_is_explained_to_the_model_and_replayed_empty(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([CUT_OFF, say("done")])

    run = await run_task(workspace_env(tmp_path / "ws"), OpenAIChatAgent(CHAT))

    last = models.requests()[-1]
    (told,) = last.tool_results
    assert run.trace.status == "completed"
    assert told.startswith("the arguments for this 'bash' call arrived incomplete")
    assert last.body["messages"][1]["tool_calls"][0]["function"]["arguments"] == "{}"


@pytest.mark.parametrize(
    ("stop_on", "turn", "reason"),
    [
        ({"malformed_tool_call"}, CUT_OFF, "malformed_tool_call"),
        ({"length"}, replace(call("bash", command="ls"), truncated=True), "length"),
    ],
    ids=["malformed", "token-capped"],
)
async def test_stop_on_ends_the_run_at_a_call_it_names(
    stop_on: set[str], turn: Turn, reason: str, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([turn])
    agent = OpenAIChatAgent(CHAT.model_copy(update={"stop_on": stop_on}))

    run = await run_task(workspace_env(tmp_path / "ws"), agent)

    assert (run.trace.status, run.trace.stop_reason) == ("completed", reason)
    assert len(models.requests()) == 1


@pytest.mark.parametrize(
    ("turn", "told"),
    [
        (call("ghost"), "unknown tool: 'ghost'"),
        (call("bash", command=""), "tool error: command is required"),
    ],
    ids=["unknown-tool", "tool-raises"],
)
async def test_a_call_that_cannot_run_is_told_to_the_model_and_the_run_goes_on(
    turn: Turn, told: str, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([turn, say("done")])

    run = await run_task(workspace_env(tmp_path / "ws"), OpenAIChatAgent(CHAT))

    assert (run.trace.status, run.trace.content) == ("completed", "done")
    assert models.requests()[-1].tool_results == [told]


async def test_the_step_budget_ends_the_run_as_truncated(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([call("bash", command="true")] * 3)
    agent = OpenAIChatAgent(CHAT.model_copy(update={"max_steps": 2}))

    run = await run_task(workspace_env(tmp_path / "ws"), agent)

    assert (run.trace.stop_reason, run.trace.is_truncated) == ("max_steps", True)


async def test_responses_resamples_an_empty_shell_call_without_advancing_its_chain(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.queue([shell_call("true"), shell_call(), say("done")])

    run = await run_task(workspace_env(tmp_path / "ws"), OpenAIAgent(OPENAI))

    chain = [request.body.get("previous_response_id") for request in models.requests()]
    assert (run.trace.content, chain) == ("done", [None, "resp_1", "resp_1"])


@pytest.mark.parametrize(
    ("tools", "status"),
    [(["bash", "bash", "bash"], "error"), (["bash", "ghost", "bash", "bash"], "completed")],
    ids=["three-lost-calls-in-a-row", "another-result-resets-the-count"],
)
async def test_a_lost_workspace_ends_the_run_after_three_consecutive_failures(
    tools: list[str], status: str, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    turns = [call("bash", command="true"), *(call(name, command="true") for name in tools)]
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
        run = await run_task(env, OpenAIChatAgent(CHAT.model_copy(update={"max_steps": 10})))

    assert run.trace.status == status
    if status == "error":
        assert run.trace.error == "SSH tool failure limit reached after 3 consecutive errors"


async def test_a_slow_shell_command_times_out_and_the_run_goes_on(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    root = tmp_path / "ws"
    models.script(
        [
            call("bash", command="touch started; sleep 1.5; touch late"),
            call("bash", command="touch quick"),
            say("done"),
        ]
    )
    agent = ClaudeAgent(CLAUDE.model_copy(update={"tool_timeout_seconds": 0.5}))

    run = await run_task(workspace_env(root), agent)
    await asyncio.sleep(1.5)

    assert run.trace.status == "completed"
    assert [(root / name).exists() for name in ("started", "quick", "late")] == [True, True, False]
    assert models.requests()[-1].tool_results[0] == (
        "Error: bash timed out after 0.5s; retry with a shorter command"
    )


async def test_an_editor_blocked_on_its_file_times_out(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    root = tmp_path / "ws"
    edit = call(
        "str_replace_based_edit_tool",
        command="str_replace",
        path=f"{root}/pipe",
        old_str="a",
        new_str="b",
    )
    models.script([call("bash", command="mkfifo pipe"), edit, say("done")])
    agent = ClaudeAgent(CLAUDE.model_copy(update={"tool_timeout_seconds": 0.5}))

    run = await run_task(workspace_env(root), agent)

    assert run.trace.status == "completed"
    assert (
        models.requests()[-1]
        .tool_results[-1]
        .startswith("Error: str_replace_based_edit_tool timed out after 0.5s")
    )


async def test_the_tool_timeout_leaves_mcp_tools_alone(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    slow = FastMCP("slow")

    @slow.tool()
    async def wait() -> str:
        """Answer after a second."""
        await asyncio.sleep(1.0)
        return "waited"

    models.script([call("wait"), say("done")])
    agent = ClaudeAgent(CLAUDE.model_copy(update={"tool_timeout_seconds": 0.5}))

    async with mcp_server(slow) as tools:
        await run_task(workspace_env(tmp_path / "ws", capabilities=(tools,)), agent)

    assert models.requests()[-1].tool_results == ["waited"]


async def test_the_agent_timeout_stops_a_run_waiting_on_its_provider(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([Turn(fail=Reply(status=500, delay=2.0))])
    agent = OpenAIChatAgent(CHAT.model_copy(update={"timeout_seconds": 0.5}))

    run = await run_task(workspace_env(tmp_path / "ws"), agent)

    assert (run.trace.status, run.trace.stop_reason) == ("error", "timeout")


@pytest.mark.parametrize(
    ("decision", "content"),
    [
        (say("CONTINUE"), "finished"),
        (say("STOP"), "Shall I go on?"),
        (fail(400), "Shall I go on?"),
    ],
    ids=["continue", "stop", "responder-fails"],
)
async def test_auto_respond_answers_a_question_or_lets_the_run_end(
    decision: Turn, content: str, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    """The auto-responder is its own Responses call beside the agent's Chat Completions."""
    hud_env.set(HUD_API_KEY="k")
    decisions = iter([decision, say("STOP")])
    replies = [say("Shall I go on?"), say("finished")]
    models.respond(
        lambda request: (
            next(decisions) if request.protocol == "responses" else replies[request.turn]
        )
    )
    agent = OpenAIChatAgent(CHAT.model_copy(update={"auto_respond": True}))

    run = await run_task(workspace_env(tmp_path / "ws"), agent)

    assert run.trace.content == content
