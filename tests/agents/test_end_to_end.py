"""Every agent family takes each fixture environment to its reward on a real model through
the HUD gateway. A reward of 1 proves the prompt reached the model, its tool calls reached
the environment, and the grade came from the right task.

The families are ``AgentType``'s members; a fixture is any module in
``tests/fixtures/envs`` that declares the ``tasks`` it runs. A family added without a model
here fails at collection.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest

from hud.eval import LocalRuntime, Taskset
from hud.types import AgentType
from tests.fixtures.envs import HERE, source

if TYPE_CHECKING:
    from tests.conftest import HudEnv

#: The cheapest current model the gateway serves each family.
MODELS = {
    AgentType.CLAUDE: "claude-haiku-5-5",
    AgentType.OPENAI: "gpt-6-luna",
    AgentType.GEMINI: "gemini-3.8-flash",
    AgentType.OPENAI_COMPATIBLE: "deepseek/deepseek-v4.1-flash",
    AgentType.CLAUDE_CLI: "claude-haiku-5-5",
    AgentType.CODEX_CLI: "gpt-6-luna",
}

FIXTURES = sorted(
    path.stem
    for path in HERE.glob("*.py")
    if any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "tasks" for target in node.targets)
        for node in ast.parse(path.read_text()).body
    )
)

CELLS = [
    pytest.param(kind, MODELS[kind], fixture, id=f"{kind}-{fixture}")
    for kind in AgentType
    for fixture in (["workspace"] if kind.is_cli else FIXTURES)
]


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.parametrize(("kind", "model", "fixture"), CELLS)
async def test_a_real_model_takes_each_fixture_to_its_reward(
    kind: AgentType, model: str, fixture: str, live: HudEnv
) -> None:
    path = source(fixture)
    agent = kind.cls(kind.config_cls(model=model))

    job = await Taskset.from_file(path).run(agent, runtime=LocalRuntime(path))

    (run,) = job.runs
    # Chat Completions has no computer use, so the screen pays it nothing.
    pays = 0.0 if (kind, fixture) == (AgentType.OPENAI_COMPATIBLE, "screen") else 1.0
    assert (run.reward, run.trace.error) == (pays, None)
