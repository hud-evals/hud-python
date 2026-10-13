"""Fixture environments driven end to end by a scripted model: ``hud eval`` and ``Taskset.run``.

Each oracle's reward is 1 only when its whole path works: the prompt reaches the
model, the model's tool calls reach the environment, the environment's state
survives between calls, and the grade of record comes from the right task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents import ClaudeAgent
from hud.agents.types import ClaudeConfig
from hud.eval import Taskset
from tests.fixtures.envs import collect_digits, read_token, repeat_word, source
from tests.harness import steps

from .lifecycle import task_calls

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.harness import Hud, HudEnv, ModelRequest, Models, Turn

MODEL = "claude-sonnet-4-6"
MAX_STEPS = 8
EVAL_ARGS = ("claude", "--model", MODEL, "--max-steps", str(MAX_STEPS), "-y", "--json")


@dataclass(frozen=True)
class Oracle:
    module: str
    policy: Callable[[], Callable[[ModelRequest], Turn]]
    phases: tuple[str, ...]
    tool_steps: int


ORACLE_TABLE = [
    Oracle("echo", lambda: repeat_word, ("setup", "evaluate"), tool_steps=0),
    Oracle("secret", lambda: read_token, ("setup", "evaluate"), tool_steps=1),
    Oracle("tools", collect_digits, ("setup", "evaluate"), tool_steps=5),
    Oracle(
        "verifier", lambda: repeat_word, ("setup", "evaluate", "setup", "evaluate"), tool_steps=0
    ),
]


@pytest.mark.parametrize("oracle", ORACLE_TABLE, ids=lambda oracle: oracle.module)
@pytest.mark.parametrize("entry", ["hud-eval", "taskset-run"])
async def test_an_oracle_scores_one_only_through_its_whole_path(
    oracle: Oracle, entry: str, models: Models, hud_env: HudEnv, hud: Hud
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.respond(oracle.policy())
    path = source(oracle.module)

    run: dict[str, Any]
    if entry == "hud-eval":
        result = hud("eval", str(path), *EVAL_ARGS)
        assert result.exit_code == 0, result
        (run,) = result.json["runs"]
    else:
        agent = ClaudeAgent(ClaudeConfig(model=MODEL, max_steps=MAX_STEPS))
        (completed,) = (await Taskset.from_file(path).run(agent)).runs
        run = {
            "slug": completed.slug,
            "reward": completed.reward,
            "is_error": completed.trace.is_error,
            "trace_id": completed.trace_id,
        }

    assert run["reward"] == 1.0
    assert run["is_error"] is False
    assert run["slug"] == next(iter(Taskset.from_file(path))).slug
    recorded = steps(run["trace_id"])
    assert [step["error"] for step in recorded if step.get("error")] == []
    assert tuple(call["phase"] for call in task_calls(recorded)) == oracle.phases
    assert sum(step["source"] == "tool" for step in recorded) == oracle.tool_steps
    assert task_calls(recorded)[0]["result"]["prompt"] == models.requests()[0].prompt
