"""An agent takes each fixture environment to its reward, on a scripted model or on its
real model (the ``live`` lane). A reward of 1 proves the prompt reached the model, its
tool calls reached the environment, and the grade came from the right task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud.eval import LocalRuntime, Task, Taskset
from hud.types import AgentType
from tests.conftest import steps
from tests.fixtures.envs import capable_model, source

if TYPE_CHECKING:
    from tests.conftest import Hud, HudEnv
    from tests.fixtures.models import Models

#: (fixture, tool steps its scripted model takes, task phases the trace records)
SCRIPTED = [
    ("echo", 0, ["setup", "evaluate"]),
    ("secret", 1, ["setup", "evaluate"]),
    ("tools", 5, ["setup", "evaluate"]),
    ("verifier", 0, ["setup", "evaluate", "setup", "evaluate"]),
]
#: (agent type, model) for every family that calls a model API itself
SCRIPTED_AGENTS = [
    ("claude", "claude-sonnet-4-6"),
    ("openai", "gpt-5.6"),
    ("gemini", "gemini-3.1-pro-preview"),
    ("openai_compatible", "qwen3.6-plus"),
]


@pytest.mark.parametrize("entry", ["hud eval", "Taskset.run"])
@pytest.mark.parametrize(
    ("agent_type", "model"), SCRIPTED_AGENTS, ids=[row[0] for row in SCRIPTED_AGENTS]
)
@pytest.mark.parametrize(
    ("fixture", "tool_steps", "phases"), SCRIPTED, ids=[r[0] for r in SCRIPTED]
)
async def test_a_scripted_model_takes_each_fixture_to_its_reward(
    entry: str,
    agent_type: str,
    model: str,
    fixture: str,
    tool_steps: int,
    phases: list[str],
    models: Models,
    hud_env: HudEnv,
    hud: Hud,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.respond(capable_model(fixture))
    path = source(fixture)

    run: dict[str, Any]
    if entry == "hud eval":
        result = hud(
            "eval", str(path), agent_type, "--model", model, "--max-steps", "8", "-y", "--json"
        )
        assert result.exit_code == 0, result
        (run,) = result.json["runs"]
    else:
        kind = AgentType(agent_type).cls
        agent = kind(kind.config_cls(model=model))
        (done,) = (await Taskset.from_file(path).run(agent, runtime=LocalRuntime(path))).runs
        run = {"reward": done.reward, "trace_id": done.trace_id, "error": done.trace.error}

    recorded = steps(run["trace_id"])
    calls = [step["task_call"] for step in recorded if step["source"] == "task"]
    assert (run["reward"], run.get("error")) == (1.0, None)
    assert [step["error"] for step in recorded if step.get("error")] == []
    assert [call["phase"] for call in calls] == phases
    assert sum(step["source"] == "tool" for step in recorded) == tool_steps


FIXTURES = {
    "echo": Task(env="echo", id="echo", args={"word": "tangerine"}),
    "tools": Task(env="tools", id="collect_code"),
    "servers": Task(env="servers", id="look_up_both"),
    "workspace": Task(env="workspace", id="fix_the_file"),
    "picture": Task(env="picture", id="name_the_colour", args={"colour": "red"}),
    "screen": Task(env="screen", id="click_target"),
}
#: (agent type, the cheapest model the gateway serves it, the fixtures it runs). Chat
#: Completions agents have no computer use, so a screen pays them nothing.
LIVE = [
    ("claude", "claude-haiku-4-5", list(FIXTURES)),
    ("openai", "gpt-5.4-mini", list(FIXTURES)),
    ("gemini", "gemini-3.8-flash", list(FIXTURES)),
    ("openai_compatible", "deepseek/deepseek-v4-flash", list(FIXTURES)),
    ("claude_cli", "claude-haiku-4-5", ["workspace"]),
    ("codex_cli", "gpt-5.4-mini", ["workspace"]),
]


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.parametrize(
    ("agent_type", "model", "fixture"),
    [(kind, model, fixture) for kind, model, fixtures in LIVE for fixture in fixtures],
)
async def test_a_real_model_takes_each_fixture_to_its_reward(
    agent_type: str, model: str, fixture: str, live: HudEnv
) -> None:
    kind = AgentType(agent_type).cls
    path = source(fixture)

    job = await Taskset(fixture, [FIXTURES[fixture]]).run(
        kind(kind.config_cls(model=model)), runtime=LocalRuntime(path)
    )

    (run,) = job.runs
    pays = 0.0 if (agent_type, fixture) == ("openai_compatible", "screen") else 1.0
    assert (run.reward, run.trace.error) == (pays, None)
