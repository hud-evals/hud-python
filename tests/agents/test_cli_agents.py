"""What the coding-CLI agents do with their CLI: one CLI per concurrent rollout, the agent
timeout stopping it, how its exit ends the run, and what Claude Code and Codex each reach
and record. A stand-in on the workspace's ``PATH`` plays each CLI."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents import ClaudeCLIAgent, CodexCLIAgent
from hud.agents.types import ToolStep
from hud.capabilities import Capability, Connection
from hud.environment.egress import ANY_HOST
from hud.eval import LocalRuntime, Shared, rollout
from hud.types import AgentType
from tests.fixtures import clis
from tests.fixtures.environments import workspace_task
from tests.fixtures.screen import fake_screen

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv

#: (agent type, the executable it launches)
CLIS = [("claude_cli", "claude"), ("codex_cli", "codex")]
EVERY_CLI = pytest.mark.parametrize(("agent_type", "cli"), CLIS, ids=[row[0] for row in CLIS])


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


@EVERY_CLI
async def test_one_agent_drives_concurrent_rollouts_each_with_its_own_cli(
    agent_type: str, cli: str, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "stub", cli, stdout=clis.answer(cli, "done"))
    stub.install()
    task, agent = workspace_task(tmp_path / "ws", env=stub.shell_env()), AgentType(agent_type).cls()

    jobs = await asyncio.gather(task(prompt="first").run(agent), task(prompt="second").run(agent))

    assert [run.trace.status for job in jobs for run in job.runs] == ["completed", "completed"]
    assert sorted(clis.prompt(cli, capture["stdin"]) for capture in stub.captures()) == [
        "first",
        "second",
    ]


@EVERY_CLI
async def test_the_agent_timeout_ends_the_run_and_the_cli_while_the_workspace_lives_on(
    agent_type: str, cli: str, tmp_path: Path
) -> None:
    """A pooled workspace outlives the run, so only the agent can end the CLI it started.
    The timeout is a margin over a whole run on this machine just now, so the CLI has
    started before it lands."""
    kind = AgentType(agent_type).cls
    stub = clis.Stub(tmp_path / "stub", cli, stdout=clis.answer(cli, "done"))
    stub.install()
    started = time.monotonic()
    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    await task().run(kind())
    timeout = 1.0 + 2 * (time.monotonic() - started)
    hung = clis.Stub(
        tmp_path / "stub", cli, stdout=clis.answer(cli, "done").splitlines()[0] + "\n", hang=True
    )
    hung.install()
    task = workspace_task(tmp_path / "ws", env=hung.shell_env())

    async with Shared(LocalRuntime(task.env), width=1) as pool:
        run = await rollout(
            task(),
            kind(kind.config_cls(timeout_seconds=timeout)),
            runtime=pool,
        )
        (capture,) = hung.captures()
        exited = clis.process_exited(capture["pid"])

    assert (run.trace.status, run.trace.stop_reason, exited) == ("error", "timeout", True)


WRITE = clis.claude_assistant(
    {"type": "text", "text": "editing"},
    {"type": "tool_use", "id": "tool-1", "name": "Write", "input": {"file_path": "a.txt"}},
    stop="tool_use",
)


DONE = clis.claude_assistant({"type": "text", "text": "done"})


@pytest.mark.parametrize(
    ("stdout", "stderr", "exit", "status", "content", "error"),
    [
        ("", "boom", 1, "error", None, "boom"),
        (clis.answer("claude", "done"), "transport failed", 1, "error", "done", "transport failed"),
        (clis.answer("claude", "done"), "", 1, "completed", "done", None),
        (clis.jsonl(DONE), "", 0, "error", "done", "claude CLI exited without a result event"),
        (
            clis.jsonl(DONE, clis.claude_result("rate limited upstream", is_error=True)),
            "",
            0,
            "error",
            "rate limited upstream",
            "rate limited upstream",
        ),
        (
            clis.jsonl(WRITE, clis.claude_result()),
            "",
            0,
            "error",
            "done",
            "claude CLI exited without results for tool calls: tool-1",
        ),
    ],
    ids=[
        "a-nonzero-exit-without-output",
        "a-nonzero-exit-after-a-result-with-stderr",
        "a-nonzero-exit-after-a-result-without-stderr",
        "a-zero-exit-without-a-result",
        "a-result-reporting-an-error",
        "a-tool-call-left-without-a-result",
    ],
)
async def test_how_the_cli_exits_decides_how_the_run_ends(
    stdout: str,
    stderr: str,
    exit: int,
    status: str,
    content: str | None,
    error: str | None,
    tmp_path: Path,
) -> None:
    stub = clis.Stub(tmp_path / "stub", "claude", stdout=stdout, stderr=stderr, exit=exit)
    stub.install()

    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    (run,) = (await task().run(ClaudeCLIAgent())).runs

    assert (run.trace.status, run.trace.content) == (status, content)
    assert run.trace.error == (None if error is None else f"[agent loop] RuntimeError: {error}")


async def test_the_cli_drives_the_screen_through_the_computer_bridge(tmp_path: Path) -> None:
    actions: list[dict[str, Any]] = [
        {"action": "left_click", "coordinate": [3, 4]},
        {"action": "frobnicate"},
    ]
    stub = clis.Stub(
        tmp_path / "stub", "claude", stdout=clis.answer("claude", "done"), computer=actions
    )
    stub.install()

    async with fake_screen() as screen:
        task = workspace_task(
            tmp_path / "ws", env=stub.shell_env(), capabilities=(Capability.rfb(url=screen.url),)
        )
        (run,) = (await task().run(ClaudeCLIAgent())).runs

    (capture,) = stub.captures()
    assert run.trace.status == "completed"
    assert capture["computer"] == [
        {"isError": False, "content": ["image"]},
        {"isError": True, "content": ["unsupported computer action: 'frobnicate'"]},
    ]
    assert [(e.x, e.y, e.buttons) for e in screen.pointer_events()] == [
        (3, 4, 0),
        (3, 4, 1),
        (3, 4, 0),
    ]


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_the_cli_reaches_a_process_bound_connection_without_its_credential(
    tmp_path: Path,
) -> None:
    connection = Connection(
        name="inference",
        capability="ssh",
        url="https://inference.hud.so/v1",
        headers={"Authorization": "Bearer scoped-runtime-token"},
    )
    stub = clis.Stub(tmp_path / "stub", "claude", stdout=clis.answer("claude", "done"))
    stub.install()
    task = workspace_task(
        tmp_path / "ws",
        env=stub.shell_env(),
        allowed_hosts={ANY_HOST},
        require_isolation=True,
        track_files=False,
    )

    run = await rollout(
        task(), ClaudeCLIAgent(), runtime=LocalRuntime(task.env), connections=(connection,)
    )

    (capture,) = stub.captures()
    assert run.trace.status == "completed"
    assert (capture["env"]["ANTHROPIC_BASE_URL"], capture["env"]["ANTHROPIC_AUTH_TOKEN"]) == (
        connection.client_url,
        "hud-process-bound",
    )
    assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_CUSTOM_HEADERS"} & capture["env"].keys()
    assert "scoped-runtime-token" not in str(capture) and "hud-key" not in str(capture)


@pytest.mark.parametrize(
    ("item", "name", "is_error", "texts"),
    [
        (
            clis.codex_item(
                "command_execution",
                "c",
                command="false",
                aggregated_output="",
                exit_code=1,
                status="completed",
            ),
            "shell",
            True,
            [""],
        ),
        (
            clis.codex_item(
                "file_change",
                "f",
                changes=[{"kind": "update", "path": "a.txt"}],
                status="completed",
            ),
            "apply_patch",
            False,
            ["update: a.txt"],
        ),
        (
            clis.codex_item(
                "mcp_tool_call",
                "m",
                server="tools",
                tool="lookup",
                arguments={"key": "k"},
                result={"content": [{"type": "text", "text": "found"}]},
                status="completed",
            ),
            "lookup",
            False,
            ["found"],
        ),
        (
            clis.codex_item(
                "mcp_tool_call",
                "m",
                server="tools",
                tool="lookup",
                arguments={},
                error={"message": "no such key"},
                status="failed",
            ),
            "lookup",
            True,
            ["no such key"],
        ),
        (
            clis.codex_item(
                "web_search", "w", query="sky", action={"type": "search", "query": "sky"}
            ),
            "web_search",
            False,
            ['{"type":"search","query":"sky"}'],
        ),
    ],
    ids=[
        "a-failed-command",
        "a-file-change",
        "an-mcp-tool-call",
        "a-failed-mcp-tool-call",
        "a-web-search",
    ],
)
async def test_each_item_codex_ran_lands_on_the_trace_as_a_tool_step(
    item: dict[str, Any], name: str, is_error: bool, texts: list[str], tmp_path: Path
) -> None:
    stdout = clis.jsonl(clis.CODEX_STARTED, item, clis.CODEX_COMPLETED)
    stub = clis.Stub(tmp_path / "stub", "codex", stdout=stdout)
    stub.install()

    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    (run,) = (await task().run(CodexCLIAgent())).runs

    (recorded,) = [step for step in run.trace.steps if isinstance(step, ToolStep)]
    assert recorded.call is not None and recorded.result is not None
    sent = [getattr(block, "text", None) for block in recorded.result.content]
    assert (recorded.call.name, recorded.result.isError, sent) == (name, is_error, texts)


@pytest.mark.parametrize(
    ("stdout", "stderr", "exit", "content", "error"),
    [
        (
            clis.jsonl(
                {"type": "turn.started"},
                {"type": "turn.failed", "error": {"message": "model unavailable"}},
            ),
            "",
            0,
            None,
            "model unavailable",
        ),
        ("", "authentication failed", 1, None, "authentication failed"),
        (
            clis.jsonl({"type": "error", "message": "gateway rejected streaming"}),
            "noisy",
            1,
            None,
            "gateway rejected streaming",
        ),
        (
            clis.jsonl(
                clis.CODEX_STARTED,
                clis.codex_item("agent_message", "message-1", text="Implemented and verified."),
            ),
            "",
            0,
            "Implemented and verified.",
            "codex CLI exited without a turn.completed event",
        ),
    ],
    ids=[
        "a-failed-turn",
        "a-nonzero-exit",
        "a-nonzero-exit-with-a-structured-error",
        "a-zero-exit-without-turn-completed",
    ],
)
async def test_how_codex_exits_decides_how_the_run_ends(
    stdout: str, stderr: str, exit: int, content: str | None, error: str, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "stub", "codex", stdout=stdout, stderr=stderr, exit=exit)
    stub.install()

    task = workspace_task(tmp_path / "ws", env=stub.shell_env())
    (run,) = (await task().run(CodexCLIAgent())).runs

    assert (run.trace.status, run.trace.content, run.trace.error) == (
        "error",
        content,
        f"[agent loop] RuntimeError: {error}",
    )


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_codex_refuses_a_launcher_script_for_a_process_bound_connection(
    tmp_path: Path,
) -> None:
    connection = Connection(
        name="inference",
        capability="ssh",
        url="https://inference.hud.so/v1",
        headers={"Authorization": "Bearer scoped-runtime-token"},
    )
    stub = clis.Stub(tmp_path / "stub", "codex", stdout=clis.answer("codex", "done"))
    stub.install()
    task = workspace_task(
        tmp_path / "ws",
        env=stub.shell_env(),
        allowed_hosts={ANY_HOST},
        require_isolation=True,
        track_files=False,
    )

    run = await rollout(
        task(), CodexCLIAgent(), runtime=LocalRuntime(task.env), connections=(connection,)
    )

    assert "/codex is a launcher script, but a process-bound inference connection " in (
        run.trace.error or ""
    )
    assert stub.captures() == []
