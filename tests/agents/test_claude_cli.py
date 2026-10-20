"""What only ``ClaudeCLIAgent`` does: what it hands the ``claude`` CLI (prompt, tools, MCP
servers, where its requests go and which model each tier uses), how the CLI's exit ends
the run, and the computer bridge it serves the CLI. A stand-in ``claude`` on the
workspace's ``PATH`` records what it was handed; the real CLI runs in the live lane."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud.agents import ClaudeCLIAgent
from hud.capabilities import Capability, Connection
from hud.environment.egress import ANY_HOST
from hud.eval import LocalRuntime, rollout
from tests.fixtures import clis
from tests.fixtures.environments import workspace_task
from tests.fixtures.screen import fake_screen

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


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
