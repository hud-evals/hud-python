"""A policy drives a robot rollout: ``RobotAgent`` on the ``robot`` wire, graded by the sim.

Every case is a real rollout: an env-side bridge serving the ``openpi/0``
WebSocket and its control channel, an agent-side ``RobotAgent`` dialing it
through ``RobotClient``, graded by the rollout engine. The flows are the ones
``docs/v6/advanced/robots.mdx`` promises:

- a single-env custom bridge, where the slot token is optional;
- ``env.gym`` behind a pre-written contract, the shape ``hud-evals/robot-template``
  ships (two cameras and an 8-D state, LIBERO-style);
- a vectorized sim, where the per-episode token pins each rollout to its slot.

The policies are stubs; the plumbing is not.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves at runtime
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from hud.agents.robot import Adapter, Model, RobotAgent
from hud.eval import LocalRuntime, Shared, Task, Taskset, rollout

from .sims import EPISODE_TICKS, StubSim, sim_env, stub_episode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

    from hud.eval import Provider


class EchoAdapter(Adapter):
    """Wires from the contract alone, and records what it wired."""

    def __init__(self) -> None:
        super().__init__()
        self.wired: list[dict[str, Any]] = []

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        data = obs["data"]
        batch = {"state": data[self.state_key], "image": data[self.image_keys[0]], "task": prompt}
        self.wired.append(batch)
        return batch


class EchoModel(Model):
    """Echoes the state's slot id, in the action width its checkpoint was trained for."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def infer(self, batch: Any) -> Any:
        slot = float(np.asarray(batch["state"]).reshape(-1)[0])
        return np.full((1, 1, self.action_dim), slot, dtype=np.float32)  # [N, T, A]


class EchoAgent(RobotAgent):
    """The model/adapter pair; the base owns the loop."""

    max_steps = EPISODE_TICKS + 2  # the env must end the episode, not this cap

    def __init__(self, adapter: EchoAdapter, *, action_dim: int = 2) -> None:
        super().__init__()
        self.model = EchoModel(action_dim)
        self.adapter = adapter


class EarlyStopAgent(EchoAgent):
    """Stops after one action through the public rollout hook."""

    def should_stop(self, obs: dict[str, Any], *, step: int, max_steps: int) -> bool:
        return step >= 1 or super().should_stop(obs, step=step, max_steps=max_steps)


#: ``hud-evals/robot-template``'s ``contract_lerobot.json``: two cameras, an 8-D
#: LIBERO state, a 7-D action. The agent wires the policy from this alone.
TEMPLATE_CONTRACT = {
    "control_rate": 80,
    "features": {
        "observation/image": {"role": "observation", "type": "rgb"},
        "observation/wrist_image": {"role": "observation", "type": "rgb"},
        "observation/state": {"role": "observation", "names": [f"state_{i}" for i in range(8)]},
        "action": {"role": "action", "names": [f"act_{i}" for i in range(7)]},
    },
}

#: Stands in for the template's ``lerobot_sim.py``: LIBERO's shape without LIBERO.
TEMPLATE_SIM = '''\
"""Module-level ``make_env`` for ``env.gym``; suite/id are build args."""

from types import SimpleNamespace

import numpy as np


class _Libero:
    """Two cameras + an 8-D state, 7-D action, env-reported success on the last tick."""

    action_space = SimpleNamespace(shape=(7,), dtype=np.float32)
    task_description = "pick up the black bowl"  # becomes the episode prompt

    def reset(self, seed=None, options=None):
        self.tick = 0
        return self._observation(), {}

    def step(self, action):
        self.tick += 1
        done = self.tick >= 3  # inside the agent's max_steps
        return self._observation(), 1.0, done, False, {"success": done}

    def close(self):
        pass

    def _observation(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        return {
            "observation/image": frame,
            "observation/wrist_image": frame,
            "observation/state": np.zeros(8, dtype=np.float32),
        }


def make_env(task_suite="libero_spatial", task_id=0):
    return _Libero()
'''

#: The template's ``environment/env.py``, unchanged but for the sim it imports.
TEMPLATE_ENV = '''\
"""LIBERO as one declarative HUD env; the task is a template arg."""

from pathlib import Path

from template_sim import make_env

from hud import Environment

env = Environment(name="libero")
sim = env.gym(make_env, contract=str(Path(__file__).parent / "contract.json"))


@env.template()
async def episode(task_suite: str = "libero_spatial", task_id: int = 0, seed: int = 0):
    """One LIBERO episode; the sim provides the instruction as the prompt."""
    ep = await sim.reset(task_suite=task_suite, task_id=task_id, seed=seed)
    yield {"prompt": ep["prompt"]}  # single env: the slot token is optional
    yield await sim.result()
'''


@asynccontextmanager
async def custom_bridge(tmp_path: Path) -> AsyncIterator[tuple[Provider, Task]]:
    async with stub_episode(StubSim(), "stub-sim") as runtime:
        yield runtime, Task(env="stub-sim", id="episode", args={"goal": "lift"})


@asynccontextmanager
async def template_files(tmp_path: Path) -> AsyncIterator[tuple[Provider, Task]]:
    (tmp_path / "template_sim.py").write_text(TEMPLATE_SIM, encoding="utf-8")
    (tmp_path / "contract.json").write_text(json.dumps(TEMPLATE_CONTRACT), encoding="utf-8")
    (tmp_path / "env.py").write_text(TEMPLATE_ENV, encoding="utf-8")
    yield LocalRuntime(tmp_path / "env.py"), Task(env="libero", id="episode", args={"task_id": 0})


@pytest.mark.parametrize(
    ("placement", "agent", "reward", "prompt", "cameras", "state"),
    [
        pytest.param(
            custom_bridge,
            lambda adapter: EchoAgent(adapter),
            EPISODE_TICKS,  # every tick's echoed action reached the sim, which ended it
            "stub task: lift",  # the sim's prompt, built from the task arg
            ["observation/image"],
            [1.0, 0.0],  # slot 1, before any step
            id="single-env-custom-bridge",
        ),
        pytest.param(
            custom_bridge,
            lambda adapter: EarlyStopAgent(adapter),
            1,  # the agent's stop hook ended the rollout after one action
            "stub task: lift",
            ["observation/image"],
            [1.0, 0.0],
            id="the-agent-stop-hook-ends-it-before-the-env",
        ),
        pytest.param(
            template_files,
            lambda adapter: EchoAgent(adapter, action_dim=7),
            1.0,  # the sim's own success outranks accumulated reward
            "pick up the black bowl",  # the sim's instruction, not the task args
            ["observation/image", "observation/wrist_image"],  # file order, state after them
            [0.0] * 8,
            id="gym-behind-a-prewritten-contract",
        ),
    ],
)
async def test_a_policy_rollout_is_wired_from_the_contract_and_graded_by_the_sim(
    tmp_path: Path,
    placement: Callable[[Path], Any],
    agent: Callable[[EchoAdapter], EchoAgent],
    reward: float,
    prompt: str,
    cameras: list[str],
    state: list[float],
) -> None:
    adapter = EchoAdapter()

    async with placement(tmp_path) as (runtime, task):
        run = await rollout(task, agent(adapter), runtime=runtime)

    assert run.reward == reward
    assert run.prompt == prompt
    assert adapter.image_keys == cameras
    assert [batch["task"] for batch in adapter.wired] == [prompt] * len(adapter.wired)
    # A camera arrives HWC uint8, the state as named.
    assert adapter.wired[0]["image"].ndim == 3
    assert adapter.wired[0]["image"].dtype == np.uint8
    assert adapter.wired[0]["state"].tolist() == state


async def test_the_slot_token_pins_each_rollout_to_its_own_slot() -> None:
    sim = StubSim(num_envs=2)
    # Both rollouts must hold a slot before either acts: a sequential pair would each
    # get their own global reset (and slot 0), which is not the case under test.
    claimed = asyncio.Barrier(2)

    async with sim_env(sim, "stub-sim") as (env, endpoint):

        @env.template()
        async def episode(goal: str = "lift") -> AsyncGenerator[Any, Any]:
            ep = await endpoint.reset(goal=goal)
            await claimed.wait()
            # Vectorized: the token is this episode's binding, and grades its slot alone.
            yield {"prompt": ep["prompt"], "bindings": {"robot": {"token": ep["token"]}}}
            yield await endpoint.result(token=ep["token"])

        job = await Taskset("stub", [Task(env="stub-sim", id="episode")]).run(
            EchoAgent(EchoAdapter()),
            runtime=Shared(LocalRuntime(env), width=2),
            group=2,
            max_concurrent=2,
        )

    # Slot i echoes i + 1 per tick, so distinct grades mean neither rollout read or
    # graded the other's slot: a dropped or crossed token grades them the same.
    assert sorted(run.reward for run in job.runs) == [EPISODE_TICKS, 2 * EPISODE_TICKS]
