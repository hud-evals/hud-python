"""Every agent family takes each fixture environment to its reward on a real model through
the HUD gateway. A reward of 1 proves the prompt reached the model, its tool calls reached
the environment, and the grade came from the right task.

The families are ``AgentType``'s members; a fixture is any module in
``tests/fixtures/envs`` that declares the ``tasks`` it runs. Each family runs its default
model, so this lane is what proves the defaults work.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest

from hud import Taskset
from hud.runtime import LocalRuntime
from hud.types import AgentType
from tests.fixtures.envs import HERE, source

if TYPE_CHECKING:
    from tests.conftest import HudEnv

FIXTURES = sorted(
    path.stem
    for path in HERE.glob("*.py")
    if any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "tasks" for target in node.targets)
        for node in ast.parse(path.read_text()).body
    )
)

# The HUD gateway does not serve Gemini's native API for now; drop this skip once it does.
GATEWAY_DOWN = pytest.mark.skip(reason="the HUD gateway does not serve Gemini's native API")

CELLS = [
    pytest.param(
        kind,
        fixture,
        id=f"{kind}-{fixture}",
        marks=GATEWAY_DOWN if kind is AgentType.GEMINI else (),
    )
    for kind in AgentType
    for fixture in (["workspace"] if kind.is_cli else FIXTURES)
]


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.parametrize(("kind", "fixture"), CELLS)
async def test_a_real_model_takes_each_fixture_to_its_reward(
    kind: AgentType, fixture: str, live: HudEnv
) -> None:
    path = source(fixture)
    agent = kind.cls()

    job = await Taskset.from_file(path).run(agent, runtime=LocalRuntime(path))

    (trace,) = job.traces
    # Chat Completions has no computer use, so the screen pays it nothing.
    pays = 0.0 if (kind, fixture) == (AgentType.OPENAI_COMPATIBLE, "screen") else 1.0
    assert (trace.reward, trace.error) == (pays, None), (
        f"answered {trace.content!r}, stopped {trace.stop_reason!r}"
    )
