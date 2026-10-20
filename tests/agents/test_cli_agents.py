"""What every coding-CLI agent does with its CLI: one CLI per concurrent rollout, and the
agent timeout stopping it. A stand-in on the workspace's ``PATH`` plays each CLI."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

from hud.eval import LocalRuntime, Shared, rollout
from hud.types import AgentType
from tests.fixtures import clis
from tests.fixtures.environments import workspace_task

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
