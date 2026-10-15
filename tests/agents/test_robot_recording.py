"""``RobotAgent(save=True)`` records each rollout as a LeRobot episode.

A sim serves the ``robot`` capability and a policy acts on it. The recording
scenarios write with the real ``lerobot`` package and read every dataset back
with ``LeRobotDataset``, so they run where it is installed (the ``lerobot``
lane): which rollouts share a dataset, the features taken from the contract,
the frames per tick, failed rollouts kept, and a Hub push that fails.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves at runtime
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from hud.agents.robot import DatasetWriter, Model, RobotAgent
from hud.environment import Environment
from hud.environment.robot import RobotBridge, RobotEndpoint
from hud.eval import LocalRuntime, Task, rollout

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from hud.eval import Run
    from tests.harness import HudEnv

TICKS = 3
# lerobot's video encoder hangs on frames under 32 px.
SIDE = 64


def contract(
    *, rate: int = 5, robot: str = "arm", features: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "control_rate": rate,
        "robot_type": robot,
        "features": features
        or {
            "observation/image": {"role": "observation", "type": "rgb"},
            "observation/state": {"role": "observation", "names": ["tick", "half"]},
            "action": {"role": "action", "names": ["a0", "a1"]},
        },
    }


class Sim(RobotBridge):
    """Cameras and a 2-D state ``[tick, 0.5]``; the episode ends after ``TICKS``."""

    def __init__(self, robot_contract: dict[str, Any]) -> None:
        super().__init__()
        self.num_envs = 1
        self.contract = robot_contract
        self.tick = 0

    def reset(self, **task_args: Any) -> str:
        self.tick = 0
        return "pick it up"

    def step(self, action: NDArray[Any]) -> None:
        self.tick += 1

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        data: dict[str, NDArray[Any]] = {}
        for name, feature in self.contract["features"].items():
            if feature["role"] != "observation":
                continue
            if feature.get("type") == "rgb":
                data[name] = np.full((1, SIDE, SIDE, 3), self.tick, dtype=np.uint8)
            else:
                data[name] = np.array([[self.tick, 0.5]], dtype=np.float32)
        return data, np.array([self.tick >= TICKS])

    def result_slots(self) -> list[dict[str, Any]]:
        return [{"score": 1.0, "success": True, "total_reward": 1.0}]


class Echo(Model):
    """Acts ``[tick, 1]``; raises on the tick named by ``fail_at``."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.fail_at = fail_at

    def infer(self, batch: Any) -> Any:
        state = next(value for value in batch["data"].values() if np.asarray(value).ndim == 1)
        tick = float(np.asarray(state).reshape(-1)[0])
        if tick == self.fail_at:
            raise RuntimeError("policy crashed")
        return np.array([[[tick, 1.0]]], dtype=np.float32)


class Recorder(RobotAgent):
    save = True
    max_steps = TICKS + 2
    log_every = 0

    def __init__(self, model: Model | None = None) -> None:
        super().__init__()
        self.model = model or Echo()
        self.adapter = None


@asynccontextmanager
async def sim_env(name: str, robot_contract: dict[str, Any]) -> AsyncIterator[Environment]:
    sim = Sim(robot_contract)
    await sim.start()
    control = await sim.serve_control()
    env = Environment(name)
    endpoint = RobotEndpoint.remote("127.0.0.1", control.sockets[0].getsockname()[1]).attach(env)

    @env.initialize
    async def up() -> None:
        await endpoint.start()
        for capability in await endpoint.capabilities():
            env.add_capability(capability)

    @env.shutdown
    async def down() -> None:
        await endpoint.stop()

    @env.template()
    async def episode() -> AsyncGenerator[Any, Any]:
        started = await endpoint.reset()
        yield {"prompt": started["prompt"]}
        yield await endpoint.result()

    try:
        await env.start()
        yield env
    finally:
        await env.stop()
        control.close()
        await sim.stop()


async def record(env: Environment, agent: RobotAgent | None = None) -> Run:
    return await rollout(
        Task(env=env.name, id="episode"), agent or Recorder(), runtime=LocalRuntime(env)
    )


def recorded(record_dir: Path) -> list[Any]:
    """Every dataset under ``record_dir`` as ``lerobot`` reads it back, oldest first."""
    lerobot_dataset: Any = importlib.import_module("lerobot.datasets.lerobot_dataset")
    roots = sorted(path for path in record_dir.glob("*") if path.is_dir())
    return [lerobot_dataset.LeRobotDataset(f"hud/{root.name}", root=root) for root in roots]


def frames(dataset: Any, episode: int) -> list[dict[str, Any]]:
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
    async with sim_env("arm", contract()) as env:
        run = await record(env)
    DatasetWriter.finalize()

    (dataset,) = recorded(recording)
    assert run.reward == 1.0
    assert (dataset.meta.fps, dataset.meta.robot_type, dataset.meta.total_episodes) == (
        5,
        "arm",
        1,
    )
    assert dataset.meta.features["observation.state"]["names"] == ["tick", "half"]
    assert tuple(dataset[0]["observation.images.observation_image"].shape) == (3, SIDE, SIDE)
    assert frames(dataset, 0) == [
        {"state": [float(tick), 0.5], "action": [float(tick), 1.0], "task": "pick it up"}
        for tick in range(TICKS)
    ]


@pytest.mark.e2e
@pytest.mark.lerobot
async def test_concurrent_rollouts_of_one_contract_share_a_dataset(recording: Path) -> None:
    async with sim_env("left", contract()) as left, sim_env("right", contract()) as right:
        await asyncio.gather(record(left), record(right))
    DatasetWriter.finalize()

    (dataset,) = recorded(recording)
    assert [len(frames(dataset, episode)) for episode in range(2)] == [TICKS, TICKS]


@pytest.mark.e2e
@pytest.mark.lerobot
async def test_a_different_rate_or_robot_gets_its_own_dataset(recording: Path) -> None:
    async with (
        sim_env("five", contract(rate=5)) as five,
        sim_env("ten", contract(rate=10)) as ten,
        sim_env("other", contract(robot="other")) as other,
    ):
        await asyncio.gather(record(five), record(ten), record(other))
    DatasetWriter.finalize()

    found = [(d.meta.fps, d.meta.robot_type, d.meta.total_episodes) for d in recorded(recording)]
    assert sorted(found) == [(5, "arm", 1), (5, "other", 1), (10, "arm", 1)]


@pytest.mark.e2e
@pytest.mark.lerobot
async def test_a_failed_rollout_keeps_the_frames_it_recorded(recording: Path) -> None:
    async with sim_env("arm", contract()) as env:
        run = await record(env, Recorder(Echo(fail_at=2)))
    DatasetWriter.finalize()

    (dataset,) = recorded(recording)
    assert run.trace.status == "error"
    assert [frame["action"] for frame in frames(dataset, 0)] == [[0.0, 1.0], [1.0, 1.0]]


@pytest.mark.e2e
@pytest.mark.lerobot
async def test_nested_camera_names_stay_distinct_features(recording: Path) -> None:
    stereo = contract(
        features={
            "left/image": {"role": "observation", "type": "rgb"},
            "right/image": {"role": "observation", "type": "rgb"},
            "state": {"role": "observation", "names": ["tick", "half"]},
            "action": {"role": "action", "names": ["a0", "a1"]},
        }
    )
    async with sim_env("stereo", stereo) as env:
        await record(env)
    DatasetWriter.finalize()

    (dataset,) = recorded(recording)
    assert {"observation.images.left_image", "observation.images.right_image"} <= set(
        dataset.meta.features
    )


@pytest.mark.e2e
@pytest.mark.lerobot
async def test_features_that_flatten_to_one_key_fail_the_rollout_before_recording(
    recording: Path,
) -> None:
    clash = contract(
        features={
            "cam/rgb": {"role": "observation", "type": "rgb"},
            "cam_rgb": {"role": "observation", "type": "rgb"},
            "state": {"role": "observation", "names": ["tick", "half"]},
            "action": {"role": "action", "names": ["a0", "a1"]},
        }
    )
    async with sim_env("clash", clash) as env:
        run = await record(env)

    assert run.trace.status == "error"
    assert recorded(recording) == []


@pytest.mark.e2e
@pytest.mark.lerobot
@pytest.mark.parametrize("repo", [None, "lab"], ids=["no-repo", "unreachable-hub"])
async def test_a_dataset_stays_on_disk_whether_or_not_its_push_succeeds(
    repo: str | None,
    recording: Path,
    hud_env: HudEnv,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hud_env.set(HF_REPO=repo, HF_ENDPOINT="http://127.0.0.1:9")
    async with sim_env("arm", contract()) as env:
        await record(env)
    DatasetWriter.finalize()

    (dataset,) = recorded(recording)
    assert dataset.meta.total_episodes == 1
    assert ("HF push failed" in capsys.readouterr().out) is (repo is not None)


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is not None, reason="lerobot installed")
async def test_without_lerobot_recording_warns_and_the_rollout_still_runs(
    hud_env: HudEnv, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    hud_env.set(RECORD_DIR=str(tmp_path / "records"))

    with caplog.at_level(logging.WARNING, logger="hud.agents.robot.dataset"):
        async with sim_env("arm", contract()) as env:
            run = await record(env)

    assert run.reward == 1.0
    assert any("lerobot" in entry.message for entry in caplog.records)
    assert not (tmp_path / "records").exists()
