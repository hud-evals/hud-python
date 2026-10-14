"""Every provider agent against every fixture, with the provider's real model.

A reward of 1 proves the agent's whole path worked against the real API: the
request it built, the replies it read and the actions it took on the
environment. An agent without a capability earns nothing from it and still
finishes cleanly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud.agents import ClaudeCLIAgent, CodexCLIAgent, create_agent
from hud.agents.types import ClaudeCLIConfig, CodexCLIConfig, ToolStep
from hud.eval import LocalRuntime, Task
from tests.fixtures.envs import source

if TYPE_CHECKING:
    from tests.harness import HudEnv

# The cheapest model of each agent family the gateway serves; Gemini's must support
# computer use.
MODELS = {
    "claude": "claude-haiku-4-5",
    "openai": "gpt-5.4-mini",
    "gemini": "gemini-3.8-flash",
    "chat": "deepseek/deepseek-v4-flash",
}

FIXTURES: dict[str, tuple[str, str, dict[str, Any]]] = {
    "echo": ("echo", "echo", {"word": "tangerine"}),
    "tools": ("tools", "collect_code", {}),
    "servers": ("servers", "look_up_both", {}),
    "workspace": ("workspace", "fix_the_file", {}),
    "picture": ("picture", "name_the_colour", {"colour": "red"}),
    "screen": ("screen", "click_target", {}),
}

# Chat Completions agents have no computer use: offered a screen, they leave it alone.
REWARDS = {(provider, fixture): 1.0 for provider in MODELS for fixture in FIXTURES} | {
    ("chat", "screen"): 0.0
}


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("provider", MODELS)
async def test_each_provider_completes_each_fixture_with_its_real_model(
    provider: str, fixture: str, live: HudEnv
) -> None:
    module, template, args = FIXTURES[fixture]

    job = await Task(env=module, id=template, args=args).run(
        create_agent(MODELS[provider]), runtime=LocalRuntime(source(module))
    )

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.error) == (
        REWARDS[provider, fixture],
        "completed",
        None,
    )


CLIS = {
    "claude-code": lambda: ClaudeCLIAgent(ClaudeCLIConfig(model="claude-haiku-4-5")),
    "codex": lambda: CodexCLIAgent(CodexCLIConfig(model="gpt-5.4-mini")),
}


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.parametrize("cli", CLIS)
async def test_each_cli_agent_fixes_a_file_with_its_real_cli(cli: str, live: HudEnv) -> None:
    job = await Task(env="workspace", id="fix_the_file").run(
        CLIS[cli](), runtime=LocalRuntime(source("workspace"))
    )

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.error) == (1.0, "completed", None)
    assert any(isinstance(step, ToolStep) for step in run.trace.steps)
