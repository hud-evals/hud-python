"""Every agent family against the fixtures, with the family's real model or CLI.

A reward of 1 proves the agent's whole path worked against the real API: the
request it built, the replies it read and the actions it took on the
environment. An agent without a capability earns nothing from it and still
finishes cleanly. Rows come from the agent families the in-process contracts
cover (``tests/agents/families.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from hud.agents import ClaudeCLIAgent, CodexCLIAgent, create_agent
from hud.agents.types import ClaudeCLIConfig, CodexCLIConfig
from hud.eval import LocalRuntime, Task
from tests.agents.families import NotApplicable, over_families
from tests.fixtures.envs import source

if TYPE_CHECKING:
    from collections.abc import Callable

    from hud.agents.base import Agent
    from tests.support import HudEnv

FIXTURES: dict[str, tuple[str, str, dict[str, Any]]] = {
    "echo": ("echo", "echo", {"word": "tangerine"}),
    "tools": ("tools", "collect_code", {}),
    "servers": ("servers", "look_up_both", {}),
    "workspace": ("workspace", "fix_the_file", {}),
    "picture": ("picture", "name_the_colour", {"colour": "red"}),
    "screen": ("screen", "click_target", {}),
}
EVERY_FIXTURE = dict.fromkeys(FIXTURES, 1.0)


# The cheapest model of each family the gateway serves (Gemini's must support computer
# use), and the reward each fixture pays it.
Live = tuple["Callable[[], Agent]", dict[str, float]]
LIVE: dict[str, Live | NotApplicable] = {
    "claude": (lambda: create_agent("claude-haiku-4-5"), EVERY_FIXTURE),
    "openai": (lambda: create_agent("gpt-5.4-mini"), EVERY_FIXTURE),
    "gemini": (lambda: create_agent("gemini-3.8-flash"), EVERY_FIXTURE),
    # Chat Completions agents have no computer use: offered a screen, they leave it alone.
    "openai_compatible": (
        lambda: create_agent("deepseek/deepseek-v4-flash"),
        EVERY_FIXTURE | {"screen": 0.0},
    ),
    "claude_cli": (
        lambda: ClaudeCLIAgent(ClaudeCLIConfig(model="claude-haiku-4-5")),
        {"workspace": 1.0},
    ),
    "codex_cli": (
        lambda: CodexCLIAgent(CodexCLIConfig(model="gpt-5.4-mini")),
        {"workspace": 1.0},
    ),
    "browser_use": NotApplicable(
        "browser-use needs an Anthropic key of its own; the live lane signs in to HUD"
    ),
}


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.parametrize(
    ("agent", "fixture", "reward"),
    [
        pytest.param(build, fixture, reward, id=f"{param.id}-{fixture}")
        for param in over_families(LIVE)
        for build, rewards in [cast("Live", param.values[1])]
        for fixture, reward in rewards.items()
    ],
)
async def test_each_family_completes_each_fixture_with_its_real_model(
    agent: Callable[[], Agent], fixture: str, reward: float, live: HudEnv
) -> None:
    module, template, args = FIXTURES[fixture]

    job = await Task(env=module, id=template, args=args).run(
        agent(), runtime=LocalRuntime(source(module))
    )

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.error) == (reward, "completed", None)
