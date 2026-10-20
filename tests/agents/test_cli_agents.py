"""What every coding-CLI agent does with its CLI: run it in the workspace, record what it
ran, end the run when it fails, refuse what it cannot route, and stop it at the agent
timeout. A stand-in on the workspace's ``PATH`` plays each CLI; every CLI family in
``AgentType`` is a row."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

from hud.agents.types import ToolStep
from hud.eval import LocalRuntime, Shared, Task, rollout
from hud.types import AgentType
from tests.fixtures import clis
from tests.fixtures.environments import run_task, workspace_env

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv

#: (agent type, the executable it launches)
CLIS = [("claude_cli", "claude"), ("codex_cli", "codex")]
EVERY_CLI = pytest.mark.parametrize(("agent_type", "cli"), CLIS, ids=[row[0] for row in CLIS])


def test_every_cli_family_is_a_row() -> None:
    assert {row[0] for row in CLIS} == {kind.value for kind in AgentType if kind.is_cli}


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


@EVERY_CLI
async def test_a_command_the_cli_ran_acts_on_the_workspace_and_lands_on_the_trace(
    agent_type: str, cli: str, tmp_path: Path
) -> None:
    stub = clis.Stub(
        tmp_path / "stub",
        cli,
        commands=["touch made"],
        stdout=clis.shell(cli, "touch made", "done"),
    )
    stub.install()

    run = await run_task(
        workspace_env(tmp_path / "ws", env=stub.shell_env()), AgentType(agent_type).cls()
    )

    assert (run.trace.status, run.trace.content) == ("completed", "done")
    assert (tmp_path / "ws" / "made").exists()
    assert sum(isinstance(step, ToolStep) for step in run.trace.steps) == 1


@EVERY_CLI
async def test_one_agent_drives_concurrent_rollouts_each_with_its_own_cli(
    agent_type: str, cli: str, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "stub", cli, stdout=clis.answer(cli, "done"))
    stub.install()
    env, agent = workspace_env(tmp_path / "ws", env=stub.shell_env()), AgentType(agent_type).cls()

    runs = await asyncio.gather(
        run_task(env, agent, prompt="first"), run_task(env, agent, prompt="second")
    )

    assert [run.trace.status for run in runs] == ["completed", "completed"]
    assert sorted(clis.prompt(cli, capture["stdin"]) for capture in stub.captures()) == [
        "first",
        "second",
    ]


@EVERY_CLI
async def test_a_failure_the_cli_reports_ends_the_run_with_its_error(
    agent_type: str, cli: str, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "stub", cli, stdout=clis.fault(cli, "model unavailable"))
    stub.install()

    run = await run_task(
        workspace_env(tmp_path / "ws", env=stub.shell_env()), AgentType(agent_type).cls()
    )

    assert run.trace.status == "error"
    assert "model unavailable" in (run.trace.error or "")


@EVERY_CLI
async def test_routing_through_the_gateway_without_a_hud_key_is_refused_before_the_cli_runs(
    agent_type: str, cli: str, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY=None)
    stub = clis.Stub(tmp_path / "stub", cli, stdout=clis.answer(cli, "done"))
    stub.install()
    kind = AgentType(agent_type).cls

    run = await run_task(
        workspace_env(tmp_path / "ws", env=stub.shell_env()), kind(kind.config_cls(gateway=True))
    )

    assert "HUD_API_KEY is required for HUD gateway routing" in (run.trace.error or "")
    assert stub.captures() == []


@EVERY_CLI
async def test_a_cli_missing_from_the_workspace_names_the_platform_it_needs(
    agent_type: str, cli: str, tmp_path: Path
) -> None:
    stub = clis.Stub(tmp_path / "stub", cli, stdout=clis.answer(cli, "done"))
    stub.install()
    env = workspace_env(tmp_path / "ws", env=stub.shell_env(path="/usr/bin:/bin"))

    run = await run_task(env, AgentType(agent_type).cls())

    assert "is unavailable for runtime platform " in (run.trace.error or "")
    assert "install it in the environment or provide a managed runtime bundle" in (
        run.trace.error or ""
    )


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
    await run_task(workspace_env(tmp_path / "ws", env=stub.shell_env()), kind())
    timeout = 1.0 + 2 * (time.monotonic() - started)
    hung = clis.Stub(
        tmp_path / "stub", cli, stdout=clis.answer(cli, "done").splitlines()[0] + "\n", hang=True
    )
    hung.install()
    env = workspace_env(tmp_path / "ws", env=hung.shell_env())

    async with Shared(LocalRuntime(env), width=1) as pool:
        run = await rollout(
            Task(env=env.name, id="task"),
            kind(kind.config_cls(timeout_seconds=timeout)),
            runtime=pool,
        )
        (capture,) = hung.captures()
        exited = clis.process_exited(capture["pid"])

    assert (run.trace.status, run.trace.stop_reason, exited) == ("error", "timeout", True)
