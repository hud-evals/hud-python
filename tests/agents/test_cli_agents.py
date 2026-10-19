"""What both coding CLI agents do with their CLI: run it in the workspace, record what it
ran, end the run when it fails, refuse what they cannot route, and stop it at the agent
timeout. A stub on the workspace's ``PATH`` stands in for each CLI; what one CLI is
handed is in its own module."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.agents import ClaudeCLIAgent, CodexCLIAgent
from hud.agents.types import ClaudeCLIConfig, CodexCLIConfig, ToolStep
from hud.eval import LocalRuntime, Shared, Task, rollout
from tests.agents.cli_stub import (
    Stub,
    claude_answer,
    claude_fault,
    claude_shell,
    codex_answer,
    codex_fault,
    codex_shell,
    process_exited,
)
from tests.fixtures.environments import run_task, workspace_env

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud import Environment
    from hud.agents.base import Agent
    from tests.support import HudEnv


@dataclass(frozen=True)
class Cli:
    """An agent that runs a coding CLI in the workspace and records its JSONL stream."""

    name: str
    agent: Callable[..., Agent]
    executable: str
    #: The stream the CLI prints when it answers, runs a command, or fails.
    answer: Callable[[str], str]
    shell: Callable[[str, str], str]
    fault: Callable[[str], str]
    #: The name its shell call is recorded under.
    shell_step: str
    #: The prompt as the CLI read it from its stdin.
    prompt: Callable[[str], str]


CLAUDE_CLI = Cli(
    "claude_cli",
    lambda **config: ClaudeCLIAgent(ClaudeCLIConfig(**config)),
    "claude",
    claude_answer,
    claude_shell,
    claude_fault,
    "Bash",
    lambda stdin: json.loads(stdin)["message"]["content"][0]["text"],
)
CODEX_CLI = Cli(
    "codex_cli",
    lambda **config: CodexCLIAgent(CodexCLIConfig(**config)),
    "codex",
    codex_answer,
    codex_shell,
    codex_fault,
    "shell",
    lambda stdin: stdin,
)
CLIS = {family.name: family for family in (CLAUDE_CLI, CODEX_CLI)}
EVERY_CLI = pytest.mark.parametrize("family", CLIS.values(), ids=CLIS.keys())


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


def installed(family: Cli, tmp_path: Path, **behavior: Any) -> Stub:
    stub = Stub(tmp_path / "stub", family.executable, **behavior)
    stub.install()
    return stub


def workspace(stub: Stub, tmp_path: Path) -> Environment:
    return workspace_env(tmp_path / "ws", env=stub.shell_env())


@EVERY_CLI
async def test_a_shell_call_the_cli_ran_acts_on_the_workspace_and_lands_on_the_trace(
    family: Cli, tmp_path: Path
) -> None:
    stub = installed(
        family, tmp_path, commands=["touch made"], stdout=family.shell("touch made", "done")
    )

    run = await run_task(workspace(stub, tmp_path), family.agent())

    calls = [step.call.name for step in run.trace.steps if isinstance(step, ToolStep) and step.call]
    assert (run.trace.status, run.trace.content) == ("completed", "done")
    assert ((tmp_path / "ws" / "made").exists(), calls) == (True, [family.shell_step])


@EVERY_CLI
async def test_one_agent_drives_concurrent_rollouts_each_with_its_own_cli(
    family: Cli, tmp_path: Path
) -> None:
    stub = installed(family, tmp_path, stdout=family.answer("done"))
    env, agent = workspace(stub, tmp_path), family.agent()

    runs = await asyncio.gather(
        run_task(env, agent, prompt="first"), run_task(env, agent, prompt="second")
    )

    assert [run.trace.status for run in runs] == ["completed", "completed"]
    assert sorted(family.prompt(capture["stdin"]) for capture in stub.captures()) == [
        "first",
        "second",
    ]


@EVERY_CLI
async def test_a_failure_the_cli_reports_ends_the_run_with_its_error(
    family: Cli, tmp_path: Path
) -> None:
    stub = installed(family, tmp_path, stdout=family.fault("model unavailable"))

    run = await run_task(workspace(stub, tmp_path), family.agent())

    assert run.trace.status == "error"
    assert "model unavailable" in (run.trace.error or "")


@EVERY_CLI
async def test_routing_through_the_gateway_without_a_hud_key_is_refused_before_the_cli_runs(
    family: Cli, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY=None)
    stub = installed(family, tmp_path, stdout=family.answer("done"))

    run = await run_task(workspace(stub, tmp_path), family.agent(gateway=True))

    assert "HUD_API_KEY is required for HUD gateway routing" in (run.trace.error or "")
    assert stub.captures() == []


@EVERY_CLI
async def test_a_cli_missing_from_the_workspace_names_its_runtime_platform(
    family: Cli, tmp_path: Path
) -> None:
    stub = installed(family, tmp_path, stdout=family.answer("done"))
    env = workspace_env(tmp_path / "ws", env=stub.shell_env(path="/usr/bin:/bin"))

    run = await run_task(env, family.agent())

    assert run.trace.error == IsStr(
        regex=r".* is unavailable for runtime platform [a-z]+-\w+; install it in the "
        r"environment or provide a managed runtime bundle"
    )


@EVERY_CLI
async def test_the_agent_timeout_ends_the_run_and_the_cli_while_the_workspace_lives_on(
    family: Cli, tmp_path: Path
) -> None:
    """A pooled workspace outlives the run, so only the agent can end the CLI it started.
    The timeout is a margin over a whole run on this machine just now, so the CLI has
    started before it lands."""
    stub = installed(family, tmp_path, stdout=family.answer("done"))
    started = time.monotonic()
    await run_task(workspace(stub, tmp_path), family.agent())
    timeout = 1.0 + 2 * (time.monotonic() - started)
    hung = installed(
        family, tmp_path, stdout=family.answer("done").splitlines()[0] + "\n", hang=True
    )
    env = workspace(hung, tmp_path)

    async with Shared(LocalRuntime(env), width=1) as pool:
        run = await rollout(
            Task(env=env.name, id="task"), family.agent(timeout_seconds=timeout), runtime=pool
        )
        (capture,) = hung.captures()
        exited = process_exited(capture["pid"])

    assert (run.trace.status, run.trace.stop_reason) == ("error", "timeout")
    assert exited
