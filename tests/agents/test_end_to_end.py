"""Each CLI agent takes the workspace fixture environment to its reward on a real model
through the HUD gateway. A reward of 1 proves the prompt reached the model, its tools
edited the workspace, and the grade came from the right task.

Each agent runs its default model, so this lane is what proves the defaults work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hud import Taskset
from hud.runtime import LocalRuntime
from hud.types import AgentType
from tests.fixtures.envs import source

if TYPE_CHECKING:
    from tests.conftest import HudEnv


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.parametrize("kind", [AgentType.CLAUDE_CLI, AgentType.CODEX_CLI, AgentType.PI])
async def test_a_real_model_takes_the_workspace_fixture_to_its_reward(
    kind: AgentType, live: HudEnv
) -> None:
    path = source("workspace")
    agent = kind.cls()

    job = await Taskset.from_file(path).run(agent, runtime=LocalRuntime(path))

    (trace,) = job.traces
    assert (trace.reward, trace.error) == (1.0, None), (
        f"answered {trace.content!r}, stopped {trace.stop_reason!r}"
    )
