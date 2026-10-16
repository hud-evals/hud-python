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
- ``RobotAgent(save=True)``, recording each rollout as a LeRobot episode with the
  real ``lerobot`` package and reading it back (the ``lerobot`` lane).

The policies here are stubs; the plumbing is not.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import logging
from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves at runtime
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from hud.agents.robot import Adapter, DatasetWriter, Model, RobotAgent
from hud.environment.robot import RobotBridge
from hud.eval import LocalRuntime, Shared, Task, Taskset, rollout
from tests.environment.robot_sims import sim_env

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from hud.environment import Environment
    from hud.eval import Provider, Run
    from tests.support import HudEnv

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

    def __init__(
        self, num_envs: int = 1, *, contract: dict[str, Any] = STUB_CONTRACT, side: int = 16
    ) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.contract = contract
        self.side = side
        self.tick = 0
        self.executed = np.zeros(num_envs)

    def reset(self, **task_args: Any) -> str:
        self.tick = 0
        self.executed[:] = 0.0
        return f"stub task: {task_args.get('goal', 'none')}"

    def step(self, action: NDArray[Any]) -> None:
        self.tick += 1
        self.executed += np.asarray(action)[:, 0]  # one row per slot

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        slots = np.arange(1, self.num_envs + 1, dtype=np.float32)
        cameras = [
            name
            for name, feature in self.contract["features"].items()
            if feature.get("type") == "rgb"
        ]
        data: dict[str, NDArray[Any]] = {
            camera: np.full((self.num_envs, self.side, self.side, 3), self.tick, dtype=np.uint8)
            for camera in cameras
        }
        data["observation/state"] = np.stack(
            [slots, np.full(self.num_envs, self.tick, dtype=np.float32)], axis=1
        )
        return data, np.full(self.num_envs, self.tick >= EPISODE_TICKS)

    def result_slots(self) -> list[dict[str, Any]]:
        # Per-slot grades, never one batch-wide scalar: the token has to resolve.
        return [
            {"score": float(echoed), "success": bool(echoed), "total_reward": float(echoed)}
            for echoed in self.executed
        ]


def _serving(env: Environment) -> Provider:
    """Placement that serves this one env — the substrate a vectorized sim shares."""
    return LocalRuntime(env)


# ─── the agent side ────────────────────────────────────────────────────


class _EchoAdapter(Adapter):
    """Wires from the contract alone, and records what it wired for the gate."""

    def __init__(self) -> None:
        super().__init__()
        self.wired: list[dict[str, Any]] = []

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        data = obs["data"]
        batch = {"state": data[self.state_key], "image": data[self.image_keys[0]], "task": prompt}
        self.wired.append(batch)
        return batch


class _EchoModel(Model):
    """Echoes the state's slot id, in the action width its checkpoint was trained for."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def infer(self, batch: Any) -> Any:
        slot = float(np.asarray(batch["state"]).reshape(-1)[0])
        return np.full((1, 1, self.action_dim), slot, dtype=np.float32)  # [N, T, A]


class _EchoAgent(RobotAgent):
    """Stub harness subclass: the model/adapter pair, base owns the loop."""

    max_steps = EPISODE_TICKS + 2  # the env must end the episode, not this cap

    def __init__(self, adapter: _EchoAdapter, *, action_dim: int = 2) -> None:
        super().__init__()
        self.model = _EchoModel(action_dim)
        self.adapter = adapter


class _EarlyStopAgent(_EchoAgent):
    """Stop after one action through the public rollout hook."""

    def should_stop(self, obs: dict[str, Any], *, step: int, max_steps: int) -> bool:
        return step >= 1 or super().should_stop(obs, step=step, max_steps=max_steps)


# ─── gates ─────────────────────────────────────────────────────────────


async def test_single_env_rollout_drives_the_policy_and_grades_the_episode() -> None:
    sim = _StubSim()
    adapter = _EchoAdapter()

    async with sim_env(sim, "stub-sim") as (env, endpoint):

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

    async with sim_env(sim, "stub-sim") as (env, endpoint):

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

    async with sim_env(sim, "stub-sim") as (env, endpoint):

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

    async with sim_env(sim, "stub-sim") as (_env, endpoint):
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


# ─── recording: RobotAgent(save=True) writes LeRobot datasets ──────────

#: lerobot's video encoder hangs on frames under 32 px.
RECORD_SIDE = 64


def _recording_contract(
    *, rate: int = 5, robot: str = "arm", cameras: tuple[str, ...] = ("observation/image",)
) -> dict[str, Any]:
    features: dict[str, Any] = {
        camera: {"role": "observation", "type": "rgb"} for camera in cameras
    }
    features["observation/state"] = {"role": "observation", "names": ["slot", "tick"]}
    features["action"] = {"role": "action", "names": ["a0", "a1"]}
    return {"control_rate": rate, "robot_type": robot, "features": features}


class _TickModel(Model):
    """Acts ``[tick, 1]``; raises on the tick named by ``fail_at``."""

    def __init__(self, fail_at: int | None) -> None:
        self.fail_at = fail_at

    def infer(self, batch: Any) -> Any:
        state = next(value for value in batch["data"].values() if np.asarray(value).ndim == 1)
        tick = float(np.asarray(state).reshape(-1)[1])
        if tick == self.fail_at:
            raise RuntimeError("policy crashed")
        return np.array([[[tick, 1.0]]], dtype=np.float32)


class _Recorder(RobotAgent):
    save = True
    max_steps = EPISODE_TICKS + 2
    log_every = 0

    def __init__(self, *, fail_at: int | None = None) -> None:
        super().__init__()
        self.model = _TickModel(fail_at)
        self.adapter = None


async def _record(name: str, contract: dict[str, Any], agent: RobotAgent | None = None) -> Run:
    async with sim_env(_StubSim(contract=contract, side=RECORD_SIDE), name) as (env, endpoint):

        @env.template()
        async def episode() -> AsyncGenerator[Any, Any]:
            ep = await endpoint.reset()
            yield {"prompt": ep["prompt"]}
            yield await endpoint.result()

        return await rollout(
            Task(env=name, id="episode"), agent or _Recorder(), runtime=_serving(env)
        )


def _recorded(record_dir: Path) -> list[Any]:
    """Every dataset under ``record_dir`` as ``lerobot`` reads it back, oldest first."""
    lerobot_dataset: Any = importlib.import_module("lerobot.datasets.lerobot_dataset")
    roots = sorted(path for path in record_dir.glob("*") if path.is_dir())
    return [lerobot_dataset.LeRobotDataset(f"hud/{root.name}", root=root) for root in roots]


def _frames(dataset: Any, episode: int) -> list[dict[str, Any]]:
    """One episode's frames as state, action and task."""
    return [
        {
            "state": frame["observation.state"].tolist(),
            "action": frame["action"].tolist(),
            "task": frame["task"],
        }
        for frame in (dataset[index] for index in range(dataset.num_frames))
        if int(frame["episode_index"]) == episode
    ]


@pytest.fixture
def recording(hud_env: HudEnv, tmp_path: Path) -> Iterator[Path]:
    """``RECORD_DIR`` for this test; datasets still open are finalized afterwards."""
    record_dir = tmp_path / "records"
    hud_env.set(RECORD_DIR=str(record_dir))
    yield record_dir
    DatasetWriter.finalize()


@pytest.mark.e2e
@pytest.mark.lerobot
async def test_a_recorded_rollout_saves_one_episode_of_observed_and_executed_frames(
    recording: Path,
) -> None:
    run = await _record("arm", _recording_contract())
    DatasetWriter.finalize()

    (dataset,) = _recorded(recording)
    assert run.reward == sum(range(EPISODE_TICKS))  # the sim saw every action
    assert (dataset.meta.fps, dataset.meta.robot_type, dataset.meta.total_episodes) == (
        5,
        "arm",
        1,
    )
    assert dataset.meta.features["observation.state"]["names"] == ["slot", "tick"]
    image = dataset[0]["observation.images.observation_image"]
    assert tuple(image.shape) == (3, RECORD_SIDE, RECORD_SIDE)
    assert _frames(dataset, 0) == [
        {"state": [1.0, float(tick)], "action": [float(tick), 1.0], "task": "stub task: none"}
        for tick in range(EPISODE_TICKS)
    ]


@pytest.mark.e2e
@pytest.mark.lerobot
async def test_concurrent_rollouts_of_one_contract_share_a_dataset(recording: Path) -> None:
    await asyncio.gather(
        _record("left", _recording_contract()), _record("right", _recording_contract())
    )
    DatasetWriter.finalize()

    (dataset,) = _recorded(recording)
    assert [len(_frames(dataset, episode)) for episode in range(2)] == [EPISODE_TICKS] * 2


@pytest.mark.e2e
@pytest.mark.lerobot
async def test_a_different_rate_or_robot_gets_its_own_dataset(recording: Path) -> None:
    await asyncio.gather(
        _record("five", _recording_contract(rate=5)),
        _record("ten", _recording_contract(rate=10)),
        _record("other", _recording_contract(robot="other")),
    )
    DatasetWriter.finalize()

    found = [(d.meta.fps, d.meta.robot_type, d.meta.total_episodes) for d in _recorded(recording)]
    assert sorted(found) == [(5, "arm", 1), (5, "other", 1), (10, "arm", 1)]


@pytest.mark.e2e
@pytest.mark.lerobot
async def test_a_failed_rollout_keeps_the_frames_it_recorded(recording: Path) -> None:
    run = await _record("arm", _recording_contract(), _Recorder(fail_at=2))
    DatasetWriter.finalize()

    (dataset,) = _recorded(recording)
    assert run.trace.status == "error"
    assert [frame["action"] for frame in _frames(dataset, 0)] == [[0.0, 1.0], [1.0, 1.0]]


@pytest.mark.e2e
@pytest.mark.lerobot
async def test_nested_camera_names_stay_distinct_features(recording: Path) -> None:
    await _record("stereo", _recording_contract(cameras=("left/image", "right/image")))
    DatasetWriter.finalize()

    (dataset,) = _recorded(recording)
    assert {"observation.images.left_image", "observation.images.right_image"} <= set(
        dataset.meta.features
    )


@pytest.mark.e2e
@pytest.mark.lerobot
async def test_features_that_flatten_to_one_key_fail_the_rollout_before_recording(
    recording: Path,
) -> None:
    run = await _record("clash", _recording_contract(cameras=("cam/rgb", "cam_rgb")))

    assert run.trace.status == "error"
    assert _recorded(recording) == []


@pytest.mark.e2e
@pytest.mark.lerobot
@pytest.mark.parametrize("repo", [None, "lab"], ids=["no-repo", "unreachable-hub"])
async def test_a_dataset_stays_on_disk_whether_or_not_its_push_succeeds(
    repo: str | None, recording: Path, hud_env: HudEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    hud_env.set(HF_REPO=repo, HF_ENDPOINT="http://127.0.0.1:9")

    await _record("arm", _recording_contract())
    DatasetWriter.finalize()

    (dataset,) = _recorded(recording)
    assert dataset.meta.total_episodes == 1
    assert ("HF push failed" in capsys.readouterr().out) is (repo is not None)


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is not None, reason="lerobot installed")
async def test_without_lerobot_recording_warns_and_the_rollout_still_runs(
    hud_env: HudEnv, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    hud_env.set(RECORD_DIR=str(tmp_path / "records"))

    with caplog.at_level(logging.WARNING, logger="hud.agents.robot.dataset"):
        run = await _record("arm", _recording_contract())

    assert run.reward == sum(range(EPISODE_TICKS))
    assert any("lerobot" in entry.message for entry in caplog.records)
    assert not (tmp_path / "records").exists()
