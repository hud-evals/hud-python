"""What only ``CodexCLIAgent`` does: where it points ``codex exec`` and with which credential,
what it records from codex's items, and how codex's exit ends the run. A stand-in
``codex`` on the workspace's ``PATH`` records what it was handed; the real CLI runs in the
live lane."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud.agents import CodexCLIAgent
from hud.agents.types import ToolStep
from hud.capabilities import Connection
from hud.environment.egress import ANY_HOST
from hud.eval import LocalRuntime, rollout
from tests.fixtures import clis
from tests.fixtures.environments import workspace_task

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


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
