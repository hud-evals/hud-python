"""Every provider agent against every fixture: a reward of 1 proves its whole path works.

A cell earns its reward only when the agent turned its model's replies into the
right actions on the environment's capabilities (an answer, MCP calls that keep
state, a click on the screen). An agent without a capability earns nothing from
it and still finishes cleanly.

The matrix runs twice. On every PR a scripted model answers from what the agent
sent: the prompt and the tools it was offered. In the live lane each provider's
real model answers through the HUD gateway, so the provider itself is what
judges the requests the agent builds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hud.agents import create_agent
from hud.eval import LocalRuntime, Task
from tests.agents.support import chat, claude, gemini, openai
from tests.fixtures.envs import collect_digits, repeat_word, source
from tests.harness import call, computer_call, say

if TYPE_CHECKING:
    from collections.abc import Callable

    from hud.agents.base import Agent
    from tests.harness import HudEnv, ModelRequest, Models, Turn


AGENTS: dict[str, Callable[[], Agent]] = {
    "claude": claude(),
    "openai": openai(),
    "gemini": gemini(),
    "chat": chat(),
}


def click_once(click: Turn) -> Callable[[ModelRequest], Turn]:
    """Send ``click`` as the first turn, then finish."""
    return lambda request: click if not request.tool_results else say("done")


CLICKS: dict[str, Turn] = {
    "claude": call("computer", action="left_click", coordinate=[100, 50]),
    "openai": computer_call({"type": "click", "x": 100, "y": 50, "button": "left"}),
    # Gemini places clicks on a 0-999 grid; 500,500 is the middle of a 200x100 screen.
    "gemini": call("click_at", x=500, y=500),
}

FIXTURES: dict[str, tuple[str, str, Callable[[str], Callable[[ModelRequest], Turn]]]] = {
    "echo": ("echo", "echo", lambda provider: repeat_word),
    "tools": ("tools", "collect_code", lambda provider: collect_digits()),
    "screen": (
        "screen",
        "click_target",
        lambda provider: click_once(CLICKS[provider]) if provider in CLICKS else repeat_word,
    ),
}

# Chat Completions agents have no computer use: offered a screen, they leave it alone.
REWARDS = {(provider, fixture): 1.0 for provider in AGENTS for fixture in FIXTURES} | {
    ("chat", "screen"): 0.0
}


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("provider", AGENTS)
async def test_each_provider_completes_each_fixture_it_can(
    provider: str, fixture: str, models: Models, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    module, template, policy = FIXTURES[fixture]
    models.respond(policy(provider))
    args = {"word": "tangerine"} if module == "echo" else {}

    job = await Task(env=module, id=template, args=args).run(
        AGENTS[provider](), runtime=LocalRuntime(source(module))
    )

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.error) == (
        REWARDS[provider, fixture],
        "completed",
        None,
    )


# The cheapest model each agent family serves through the gateway.
LIVE_MODELS = {
    "claude": "claude-haiku-4-5",
    "openai": "gpt-5.4-mini",
    "gemini": "gemini-3.8-flash",
    "chat": "deepseek/deepseek-v4-flash",
}


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("provider", AGENTS)
async def test_each_provider_completes_each_fixture_with_its_real_model(
    provider: str, fixture: str, live: HudEnv
) -> None:
    module, template, _ = FIXTURES[fixture]
    args = {"word": "tangerine"} if module == "echo" else {}

    job = await Task(env=module, id=template, args=args).run(
        create_agent(LIVE_MODELS[provider]), runtime=LocalRuntime(source(module))
    )

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.error) == (
        REWARDS[provider, fixture],
        "completed",
        None,
    )
