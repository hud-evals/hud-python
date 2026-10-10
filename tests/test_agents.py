"""Every provider agent against every fixture: a reward of 1 proves its whole path works.

Each scripted model answers the way that provider would, from what the agent
actually sent: the prompt it was shown and the tools it was offered. A cell
earns its reward only when the agent turned the model's replies into the right
actions on the environment's capabilities (an answer, MCP calls that keep state,
a click on the screen). An agent without a capability earns nothing from it and
still finishes cleanly.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from hud.agents import ClaudeAgent, GeminiAgent, OpenAIAgent, OpenAIChatAgent
from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIChatConfig, OpenAIConfig
from hud.eval import LocalRuntime, Task
from tests.fixtures.envs import source
from tests.harness import call, computer_call, say

if TYPE_CHECKING:
    from collections.abc import Callable

    from hud.agents.base import Agent
    from tests.harness import HudEnv, ModelRequest, Models, Turn


AGENTS: dict[str, Callable[[], Agent]] = {
    "claude": lambda: ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6")),
    "openai": lambda: OpenAIAgent(OpenAIConfig(model="gpt-5.6")),
    "gemini": lambda: GeminiAgent(GeminiConfig(model="gemini-3.1-pro-preview")),
    "chat": lambda: OpenAIChatAgent(OpenAIChatConfig(model="m")),
}


def repeat_word(request: ModelRequest) -> Turn:
    """Answer with the word the prompt asks for."""
    match = re.search(r"Reply with: (\S+)", request.prompt)
    return say(match.group(1) if match else "no word in the prompt")


def collect_digits() -> Callable[[ModelRequest], Turn]:
    """Call next_digit until it says done, then answer with every digit it returned.

    Responses chains requests and sends only the newest tool result, so the policy
    keeps the digits it has seen, as the model's own context would.
    """
    digits: list[str] = []

    def policy(request: ModelRequest) -> Turn:
        if request.tool_results:
            latest = re.search(r"\b(\d|done)\b", request.tool_results[-1])
            assert latest is not None, request.tool_results[-1]
            if latest.group(1) == "done":
                return say("".join(digits))
            digits.append(latest.group(1))
        (tool,) = [name for name in request.tools if name.endswith("next_digit")]
        return call(tool)

    return policy


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
