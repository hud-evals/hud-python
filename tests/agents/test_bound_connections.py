"""CLI agents on a process-bound inference connection, in an isolated workspace.

The connection's relay holds the scoped credential; only the launched CLI process
can reach it. A stub ``claude`` or ``codex`` records what it was handed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.agents import ClaudeCLIAgent, CodexCLIAgent
from hud.capabilities import Connection
from hud.environment.egress import ANY_HOST
from tests.agents.cli_stub import Stub
from tests.agents.families import claude_answer, codex_answer
from tests.support import run_task, workspace_env

if TYPE_CHECKING:
    from pathlib import Path

    from hud import Environment
    from tests.support import HudEnv


def claude_stub(root: Path, **behavior: Any) -> Stub:
    behavior.setdefault("stdout", claude_answer("done"))
    return Stub(root / "stub", "claude", **behavior)


def codex_stub(root: Path, **behavior: Any) -> Stub:
    behavior.setdefault("stdout", codex_answer("Implemented and verified."))
    return Stub(root / "stub", "codex", **behavior)


CONNECTION = Connection(
    name="inference",
    capability="ssh",
    url="https://inference.hud.so/v1",
    headers={"Authorization": "Bearer scoped-runtime-token"},
)


def isolated_env(tmp_path: Path, stub: Stub) -> Environment:
    stub.install()
    return workspace_env(
        tmp_path / "ws",
        env=stub.shell_env(),
        allowed_hosts={ANY_HOST},
        require_isolation=True,
        track_files=False,
    )


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_the_claude_cli_reaches_a_bound_connection_without_its_credential(
    hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="hud-key")
    stub = claude_stub(tmp_path)

    run = await run_task(isolated_env(tmp_path, stub), ClaudeCLIAgent(), connections=(CONNECTION,))

    (capture,) = stub.captures()
    env = capture["env"]
    assert run.trace.status == "completed"
    assert (env["ANTHROPIC_BASE_URL"], env["ANTHROPIC_AUTH_TOKEN"]) == (
        CONNECTION.client_url,
        "hud-process-bound",
    )
    assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_CUSTOM_HEADERS"} & env.keys()
    assert "scoped-runtime-token" not in str(capture) and "hud-key" not in str(capture)


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_codex_refuses_a_launcher_script_for_a_bound_connection(
    hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="hud-key")
    stub = codex_stub(tmp_path)

    run = await run_task(isolated_env(tmp_path, stub), CodexCLIAgent(), connections=(CONNECTION,))

    assert (run.trace.error, stub.captures()) == (
        IsStr(regex=r".*/codex is a launcher script, but a process-bound inference connection .*"),
        [],
    )
