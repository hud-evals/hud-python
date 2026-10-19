"""An agent takes each fixture environment to its reward, on a scripted model or on the
family's real model.

A reward of 1 proves the whole path worked: the prompt reached the model, the
model's tool calls reached the environment, its state survived between calls, and
the grade of record came from the right task. On the scripted model (the default
lane) each fixture is answered by its policy in ``tests.fixtures.envs``, through
``hud eval`` and through ``Taskset.run``, and the trace also shows the tool steps
the policy took and the prompt it was given. On the real model (the ``live`` lane)
every family runs each fixture it can: an agent without a capability earns nothing
from it and still finishes cleanly. The live rows come from the agent families
(``tests/agents/families.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest

from hud.agents import ClaudeAgent, ClaudeCLIAgent, CodexCLIAgent, create_agent
from hud.agents.types import ClaudeCLIConfig, ClaudeConfig, CodexCLIConfig
from hud.eval import LocalRuntime, Task, Taskset
from tests.agents.families import NotApplicable, over_families
from tests.e2e.lifecycle import task_calls
from tests.fixtures.envs import collect_digits, read_token, repeat_word, source
from tests.support import steps

if TYPE_CHECKING:
    from collections.abc import Callable

    from hud.agents.base import Agent
    from tests.support import Hud, HudEnv, ModelRequest, Models, Turn

MODEL = "claude-sonnet-4-6"
MAX_STEPS = 8
EVAL_ARGS = ("claude", "--model", MODEL, "--max-steps", str(MAX_STEPS), "-y", "--json")

#: Each fixture's row, as the live lane runs it.
FIXTURES: dict[str, Task] = {
    "echo": Task(env="echo", id="echo", args={"word": "tangerine"}),
    "tools": Task(env="tools", id="collect_code"),
    "servers": Task(env="servers", id="look_up_both"),
    "workspace": Task(env="workspace", id="fix_the_file"),
    "picture": Task(env="picture", id="name_the_colour", args={"colour": "red"}),
    "screen": Task(env="screen", id="click_target"),
}
ONCE = ("setup", "evaluate")
#: The fixtures a scripted model answers from their tasks file: its policy, the tool
#: steps that takes, and the task phases the trace records.
SCRIPTED: dict[str, tuple[Callable[[], Callable[[ModelRequest], Turn]], int, tuple[str, ...]]] = {
    "echo": (lambda: repeat_word, 0, ONCE),
    "secret": (lambda: read_token, 1, ONCE),
    "tools": (collect_digits, 5, ONCE),
    "verifier": (lambda: repeat_word, 0, (*ONCE, *ONCE)),
}

EVERY_FIXTURE = dict.fromkeys(FIXTURES, 1.0)
Live = tuple["Callable[[], Agent]", dict[str, float]]
# The cheapest model of each family the gateway serves (Gemini's must support computer
# use), and the reward each fixture pays it.
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


@dataclass(frozen=True)
class Backing:
    """A scripted model answering with ``policy``, or the real model ``agent`` builds."""

    entry: str
    reward: float
    agent: Callable[[], Agent] | None = None
    policy: Callable[[], Callable[[ModelRequest], Turn]] | None = None
    tool_steps: int | None = None
    phases: tuple[str, ...] = ()


ROWS = [
    *(
        pytest.param(
            fixture,
            Backing(entry, 1.0, policy=policy, tool_steps=tools, phases=phases),
            id=f"{entry}-{fixture}",
        )
        for fixture, (policy, tools, phases) in SCRIPTED.items()
        for entry in ("hud eval", "Taskset.run")
    ),
    *(
        pytest.param(
            fixture,
            Backing("Taskset.run", reward, agent=build),
            id=f"live-{param.id}-{fixture}",
            marks=[pytest.mark.e2e, pytest.mark.live],
        )
        for param in over_families(LIVE)
        for build, rewards in [cast("Live", param.values[1])]
        for fixture, reward in rewards.items()
    ),
]


@pytest.mark.parametrize(("fixture", "backing"), ROWS)
async def test_an_agent_takes_each_fixture_to_its_reward(
    fixture: str,
    backing: Backing,
    models: Models,
    hud_env: HudEnv,
    hud: Hud,
    request: pytest.FixtureRequest,
) -> None:
    path = source(fixture)
    if backing.policy is not None:
        hud_env.set(HUD_API_KEY="k")
        models.respond(backing.policy())
    else:
        request.getfixturevalue("live")

    run: dict[str, Any]
    if backing.entry == "hud eval":
        result = hud("eval", str(path), *EVAL_ARGS)
        assert result.exit_code == 0, result
        (run,) = result.json["runs"]
    else:
        agent = (
            backing.agent()
            if backing.agent is not None
            else ClaudeAgent(ClaudeConfig(model=MODEL, max_steps=MAX_STEPS))
        )
        rows = (
            Taskset(fixture, [FIXTURES[fixture]])
            if backing.agent is not None
            else Taskset.from_file(path)
        )
        (completed,) = (await rows.run(agent, runtime=LocalRuntime(path))).runs
        run = {
            "reward": completed.reward,
            "is_error": completed.trace.is_error,
            "trace_id": completed.trace_id,
            "error": completed.trace.error,
        }

    assert (run["reward"], run["is_error"], run.get("error")) == (backing.reward, False, None)
    if backing.policy is not None:
        recorded = steps(run["trace_id"])
        assert [step["error"] for step in recorded if step.get("error")] == []
        assert tuple(call["phase"] for call in task_calls(recorded)) == backing.phases
        assert sum(step["source"] == "tool" for step in recorded) == backing.tool_steps
        assert task_calls(recorded)[0]["result"]["prompt"] == models.requests()[0].prompt
