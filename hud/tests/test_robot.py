"""End-to-end gates for the ``robot`` capability — the flows the docs promise.

Every test drives a real rollout: an env-side bridge serving the ``openpi/0``
WebSocket and its JSON-RPC control channel, an agent-side ``RobotAgent`` dialing
it through ``RobotClient``, graded by the rollout engine. So a change that breaks
the wire, the contract split, or the slot token fails here rather than in a GPU
benchmark. The flows (see ``docs/v6/advanced/robots.mdx``):

- a single-env custom bridge, where the slot token is optional;
- a vectorized sim, where the per-episode token pins each rollout to its slot;
- ``env.gym`` behind a pre-written contract — the shape ``hud-evals/robot-template``
  ships (two cameras + an 8-D state, LIBERO-style).

The policies here are stubs; the plumbing is not.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves at runtime
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from typing_extensions import override

from hud.agents.robot import Adapter, Model, RobotAgent
from hud.environment import Environment
from hud.environment.robot import RobotBridge, RobotEndpoint
from hud.eval import LocalRuntime, Shared, Task, Taskset, rollout
from hud.eval.runtime import _local

if TYPE_CHECKING:
    from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from hud.eval import Provider
    from hud.eval.runtime import Runtime

#: Ticks a stub sim runs before terminating — every gate here is a whole rollout.
EPISODE_TICKS = 3

#: A minimal contract, exactly as documented: one camera, a state vector, an action.
STUB_CONTRACT = {
    "control_rate": 5,
    "features": {
        "observation/image": {"role": "observation", "type": "rgb"},
        "observation/state": {"role": "observation", "names": ["slot", "tick"]},
        "action": {"role": "action", "names": ["a0", "a1"]},
    },
}


# ─── the sim side ──────────────────────────────────────────────────────


class _StubSim(RobotBridge):
    """One camera + a 2-D state, terminating after ``EPISODE_TICKS`` ticks.

    Slot ``i``'s state carries ``i + 1``, and its grade is the sum of the actions
    that slot received — so with an echoing policy the grade names the slot the
    connection actually drove.
    """

    def __init__(self, num_envs: int = 1) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.contract = STUB_CONTRACT
        self.tick = 0
        self.executed = np.zeros(num_envs)

    @override
    def reset(self, **task_args: Any) -> str:
        self.tick = 0
        self.executed[:] = 0.0
        return f"stub task: {task_args.get('goal', 'none')}"

    @override
    def step(self, action: NDArray[Any]) -> None:
        self.tick += 1
        self.executed += np.asarray(action)[:, 0]  # one row per slot

    @override
    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        slots = np.arange(1, self.num_envs + 1, dtype=np.float32)
        data = {
            "observation/image": np.zeros((self.num_envs, 16, 16, 3), dtype=np.uint8),
            "observation/state": np.stack(
                [slots, np.full(self.num_envs, self.tick, dtype=np.float32)], axis=1
            ),
        }
        return data, np.full(self.num_envs, self.tick >= EPISODE_TICKS)

    @override
    def result_slots(self) -> list[dict[str, Any]]:
        # Per-slot grades, never one batch-wide scalar: the token has to resolve.
        return [
            {"score": float(echoed), "success": bool(echoed), "total_reward": float(echoed)}
            for echoed in self.executed
        ]


@asynccontextmanager
async def _sim_env(sim: RobotBridge) -> AsyncIterator[tuple[Environment, RobotEndpoint]]:
    """The docs' custom-bridge env, with *sim* served by this process.

    ``RobotEndpoint.remote`` attaches to a sim someone else runs — here the test
    itself, so the whole control plane is real without a spawned sim program.
    Yields the env and its endpoint; each test declares the template it is about.
    """
    await sim.start()
    control = await sim.serve_control()
    env = Environment("stub-sim")
    endpoint = RobotEndpoint.remote("127.0.0.1", control.sockets[0].getsockname()[1]).attach(env)

    @env.initialize
    async def _up() -> None:
        await endpoint.start()
        for cap in await endpoint.capabilities():
            env.add_capability(cap)

    @env.shutdown
    async def _down() -> None:
        await endpoint.stop()

    try:
        await env.start()  # publish the capability now; serving re-enters as a no-op
        yield env, endpoint
    finally:
        await env.stop()
        control.close()
        await sim.stop()


def _serving(env: Environment) -> Provider:
    """Placement that serves this one env — the substrate a vectorized sim shares."""

    @asynccontextmanager
    async def provider(task: Task) -> AsyncIterator[Runtime]:
        async with _local(env) as runtime:
            yield runtime

    return provider


# ─── the agent side ────────────────────────────────────────────────────


class _EchoAdapter(Adapter):
    """Wires from the contract alone, and records what it wired for the gate."""

    def __init__(self) -> None:
        super().__init__()
        self.wired: list[dict[str, Any]] = []

    @override
    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        data = obs["data"]
        batch = {"state": data[self.state_key], "image": data[self.image_keys[0]], "task": prompt}
        self.wired.append(batch)
        return batch


class _EchoModel(Model):
    """Echoes the state's slot id, in the action width its checkpoint was trained for."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    @override
    def infer(self, batch: Any) -> Any:
        slot = float(np.asarray(batch["state"]).reshape(-1)[0])
        return np.full((1, 1, self.action_dim), slot, dtype=np.float32)  # [N, T, A]


class _EchoAgent(RobotAgent):
    """Stub harness subclass: the model/adapter pair, base owns the loop."""

    max_steps = EPISODE_TICKS + 2  # the env must end the episode, not this cap

    def __init__(self, adapter: _EchoAdapter, *, action_dim: int = 2) -> None:
        self.model = _EchoModel(action_dim)
        self.adapter = adapter


class _EarlyStopAgent(_EchoAgent):
    """Stop after one action through the public rollout hook."""

    @override
    def should_stop(self, obs: dict[str, Any], *, step: int, max_steps: int) -> bool:
        return step >= 1 or super().should_stop(obs, step=step, max_steps=max_steps)


# ─── gates ─────────────────────────────────────────────────────────────


async def test_single_env_rollout_drives_the_policy_and_grades_the_episode() -> None:
    sim = _StubSim()
    adapter = _EchoAdapter()

    async with _sim_env(sim) as (env, endpoint):

        @env.template()
        async def episode(goal: str = "lift") -> AsyncGenerator[Any, Any]:
            ep = await endpoint.reset(goal=goal)
            yield {"prompt": ep["prompt"]}  # one slot: the token is optional
            yield await endpoint.result()

        run = await rollout(
            Task(env="stub-sim", id="episode", args={"goal": "lift"}),
            _EchoAgent(adapter),
            runtime=_serving(env),
        )

    assert run.reward == EPISODE_TICKS  # every tick's echoed action reached the sim
    assert sim.tick == EPISODE_TICKS  # the env terminated the episode, not max_steps
    assert run.prompt == "stub task: lift"  # the sim's prompt, built from the task arg
    # The contract is the whole wiring: its camera arrives HWC uint8, its state as named.
    assert [batch["task"] for batch in adapter.wired] == [run.prompt] * EPISODE_TICKS
    assert adapter.wired[0]["image"].shape == (16, 16, 3)
    assert adapter.wired[0]["image"].dtype == np.uint8
    assert adapter.wired[0]["state"].tolist() == [1.0, 0.0]  # slot 1, before any step


async def test_agent_stop_hook_ends_the_rollout_before_the_env_terminates() -> None:
    sim = _StubSim()

    async with _sim_env(sim) as (env, endpoint):

        @env.template()
        async def episode() -> AsyncGenerator[Any, Any]:
            ep = await endpoint.reset()
            yield {"prompt": ep["prompt"]}
            yield await endpoint.result()

        run = await rollout(
            Task(env="stub-sim", id="episode"),
            _EarlyStopAgent(_EchoAdapter()),
            runtime=_serving(env),
        )

    assert sim.tick == 1
    assert run.reward == 1


async def test_the_slot_token_pins_each_rollout_to_its_own_slot() -> None:
    sim = _StubSim(num_envs=2)
    # Both rollouts must hold a slot before either acts: a sequential pair would each
    # get their own global reset (and slot 0), which is not the case under test.
    claimed = asyncio.Barrier(2)

    async with _sim_env(sim) as (env, endpoint):

        @env.template()
        async def episode(goal: str = "lift") -> AsyncGenerator[Any, Any]:
            ep = await endpoint.reset(goal=goal)
            await claimed.wait()
            # Vectorized: the token is this episode's binding, and grades its slot alone.
            yield {"prompt": ep["prompt"], "bindings": {"robot": {"token": ep["token"]}}}
            yield await endpoint.result(token=ep["token"])

        job = await Taskset("stub", [Task(env="stub-sim", id="episode")]).run(
            _EchoAgent(_EchoAdapter()),
            runtime=Shared(_serving(env), width=2),
            group=2,
            max_concurrent=2,
        )

    # Slot i echoes i + 1 per tick, so distinct grades mean neither rollout read or
    # graded the other's slot — a dropped or crossed token grades them the same.
    assert sorted(run.reward for run in job.runs) == [EPISODE_TICKS, 2 * EPISODE_TICKS]


async def test_a_vectorized_sim_refuses_a_tokenless_grade() -> None:
    sim = _StubSim(num_envs=2)

    async with _sim_env(sim) as (_env, endpoint):
        first = await endpoint.reset(goal="lift")
        second = await endpoint.reset(goal="lift")
        assert first["token"] != second["token"]  # one token per slot, not per batch

        # Two claimed slots: which one to grade is ambiguous, so say so instead of guessing.
        with pytest.raises(RuntimeError, match="tokenless claim"):
            await endpoint.result()


# ─── the robot-template shape: env.gym behind a pre-written contract ───

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

#: Stands in for the template's ``lerobot_sim.py`` — LIBERO's shape without LIBERO.
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
"""LIBERO as one declarative HUD env — the task is a template arg."""

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


async def test_a_gym_sim_behind_a_prewritten_contract_runs_the_template_flow(
    tmp_path: Path,
) -> None:
    (tmp_path / "template_sim.py").write_text(TEMPLATE_SIM, encoding="utf-8")
    (tmp_path / "contract.json").write_text(json.dumps(TEMPLATE_CONTRACT), encoding="utf-8")
    (tmp_path / "env.py").write_text(TEMPLATE_ENV, encoding="utf-8")
    adapter = _EchoAdapter()

    run = await rollout(
        Task(env="libero", id="episode", args={"task_id": 0}),
        _EchoAgent(adapter, action_dim=7),
        runtime=LocalRuntime(tmp_path / "env.py"),
    )

    assert run.reward == 1.0  # the sim's own success outranks accumulated reward
    assert run.prompt == "pick up the black bowl"  # the sim's instruction, not the task args
    # A pre-written contract is loaded verbatim, cameras in file order, state after them.
    assert adapter.image_keys == ["observation/image", "observation/wrist_image"]
    assert adapter.state_key == "observation/state"
    assert adapter.wired[0]["state"].shape == (8,)
