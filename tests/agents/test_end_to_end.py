"""Every agent family takes each fixture environment to its reward on a real model through
the HUD gateway. A reward of 1 proves the prompt reached the model, its tool calls reached
the environment, and the grade came from the right task.

The families are ``AgentType``'s members; a fixture is any module in
``tests/fixtures/envs`` that declares the ``tasks`` it runs. A family added without a model
here fails at collection.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

import pytest

from hud.eval import LocalRuntime, Taskset
from hud.types import AgentType
from tests.fixtures.envs import HERE, source

if TYPE_CHECKING:
    from tests.conftest import HudEnv

#: Each family's config: the cheapest current model the gateway serves it. OpenRouter
#: spreads DeepSeek across providers, some of which reply empty or garbled, so it is held
#: to one that answers.
CONFIGS: dict[AgentType, dict[str, Any]] = {
    AgentType.CLAUDE: {"model": "claude-haiku-5-5"},
    AgentType.OPENAI: {"model": "gpt-6-luna"},
    AgentType.GEMINI: {"model": "gemini-3.8-flash"},
    AgentType.OPENAI_COMPATIBLE: {
        "model": "deepseek/deepseek-v4.1-flash",
        "completion_kwargs": {
            "extra_body": {"provider": {"only": ["Together"], "allow_fallbacks": False}}
        },
    },
    AgentType.CLAUDE_CLI: {"model": "claude-haiku-5-5"},
    AgentType.CODEX_CLI: {"model": "gpt-6-luna"},
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
    pytest.param(kind, CONFIGS[kind], fixture, id=f"{kind}-{fixture}")
    for kind in AgentType
    for fixture in (["workspace"] if kind.is_cli else FIXTURES)
]


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.parametrize(("kind", "config", "fixture"), CELLS)
async def test_a_real_model_takes_each_fixture_to_its_reward(
    kind: AgentType, config: dict[str, Any], fixture: str, live: HudEnv
) -> None:
    path = source(fixture)
    agent = kind.cls(kind.config_cls(**config))

    job = await Taskset.from_file(path).run(agent, runtime=LocalRuntime(path))

    (run,) = job.runs
    # Chat Completions has no computer use, so the screen pays it nothing.
    pays = 0.0 if (kind, fixture) == (AgentType.OPENAI_COMPATIBLE, "screen") else 1.0
    assert (run.reward, run.trace.error) == (pays, None), (
        f"answered {run.trace.content!r}, stopped {run.trace.stop_reason!r}"
    )
