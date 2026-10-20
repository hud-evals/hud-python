"""Dataset recording: ``RobotAgent(save=True)`` writes each rollout as a LeRobot episode.

The ``lerobot`` lane records with the real ``lerobot`` package and reads the
datasets back with it.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from hud.agents.robot import DatasetWriter, Model, RobotAgent
from hud.eval import Task, rollout

from .sims import EPISODE_TICKS, StubSim, stub_episode

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from hud.eval import Run
    from tests.conftest import HudEnv

#: lerobot's video encoder hangs on frames under 32 px.
RECORD_SIDE = 64


def recording_contract(
    *, rate: int = 5, robot: str = "arm", cameras: tuple[str, ...] = ("observation/image",)
) -> dict[str, Any]:
    features: dict[str, Any] = {
        camera: {"role": "observation", "type": "rgb"} for camera in cameras
    }
    features["observation/state"] = {"role": "observation", "names": ["slot", "tick"]}
    features["action"] = {"role": "action", "names": ["a0", "a1"]}
    return {"control_rate": rate, "robot_type": robot, "features": features}


class TickModel(Model):
    """Acts ``[tick, 1]``; raises on the tick named by ``fail_at``."""

    def __init__(self, fail_at: int | None) -> None:
        self.fail_at = fail_at

    def infer(self, batch: Any) -> Any:
        state = next(value for value in batch["data"].values() if np.asarray(value).ndim == 1)
        tick = float(np.asarray(state).reshape(-1)[1])
        if tick == self.fail_at:
            raise RuntimeError("policy crashed")
        return np.array([[[tick, 1.0]]], dtype=np.float32)


class Recorder(RobotAgent):
    save = True
    max_steps = EPISODE_TICKS + 2
    log_every = 0

    def __init__(self, *, fail_at: int | None = None) -> None:
        super().__init__()
        self.model = TickModel(fail_at)
        self.adapter = None


async def record(name: str, contract: dict[str, Any], *, fail_at: int | None = None) -> Run:
    async with stub_episode(StubSim(contract=contract, side=RECORD_SIDE), name) as runtime:
        return await rollout(
            Task(env=name, id="episode"), Recorder(fail_at=fail_at), runtime=runtime
        )


def datasets(record_dir: Path) -> list[Any]:
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


@pytest.mark.parametrize(
    ("rollouts", "fail_at", "status", "found", "episode_actions", "features"),
    [
        pytest.param(
            {"arm": recording_contract()},
            None,
            "completed",
            [(5, "arm", 1)],
            [[float(tick), 1.0] for tick in range(EPISODE_TICKS)],
            {"observation.images.observation_image"},
            id="one-rollout-is-one-episode",
        ),
        pytest.param(
            {"left": recording_contract(), "right": recording_contract()},
            None,
            "completed",
            [(5, "arm", 2)],
            [[float(tick), 1.0] for tick in range(EPISODE_TICKS)],
            set(),
            id="concurrent-rollouts-of-one-contract-share-a-dataset",
        ),
        pytest.param(
            {
                "five": recording_contract(rate=5),
                "ten": recording_contract(rate=10),
                "other": recording_contract(robot="other"),
            },
            None,
            "completed",
            [(5, "arm", 1), (5, "other", 1), (10, "arm", 1)],
            None,
            set(),
            id="a-different-rate-or-robot-gets-its-own-dataset",
        ),
        pytest.param(
            {"arm": recording_contract()},
            2,
            "error",
            [(5, "arm", 1)],
            [[0.0, 1.0], [1.0, 1.0]],
            set(),
            id="a-failed-rollout-keeps-the-frames-it-recorded",
        ),
        pytest.param(
            {"stereo": recording_contract(cameras=("left/image", "right/image"))},
            None,
            "completed",
            [(5, "arm", 1)],
            None,
            {"observation.images.left_image", "observation.images.right_image"},
            id="nested-camera-names-stay-distinct-features",
        ),
        pytest.param(
            {"clash": recording_contract(cameras=("cam/rgb", "cam_rgb"))},
            None,
            "error",
            [],
            None,
            set(),
            id="features-that-flatten-to-one-key-fail-the-rollout-before-recording",
        ),
    ],
)
@pytest.mark.e2e
@pytest.mark.lerobot
async def test_recorded_rollouts_land_in_one_dataset_per_rate_and_robot(
    recording: Path,
    rollouts: dict[str, dict[str, Any]],
    fail_at: int | None,
    status: str,
    found: list[tuple[int, str, int]],
    episode_actions: list[list[float]] | None,
    features: set[str],
) -> None:
    runs = await asyncio.gather(
        *(record(name, contract, fail_at=fail_at) for name, contract in rollouts.items())
    )
    DatasetWriter.finalize()

    recorded = datasets(recording)
    assert {run.trace.status for run in runs} == {status}
    assert sorted((d.meta.fps, d.meta.robot_type, d.meta.total_episodes) for d in recorded) == found
    for dataset in recorded:
        assert features <= set(dataset.meta.features)
        if episode_actions is not None:
            for episode in range(dataset.meta.total_episodes):
                assert [frame["action"] for frame in frames(dataset, episode)] == episode_actions


@pytest.mark.e2e
@pytest.mark.lerobot
async def test_a_recorded_episode_holds_the_observed_state_executed_action_and_task(
    recording: Path,
) -> None:
    run = await record("arm", recording_contract())
    DatasetWriter.finalize()

    (dataset,) = datasets(recording)
    assert run.reward == sum(range(EPISODE_TICKS))  # the sim saw every action
    assert dataset.meta.features["observation.state"]["names"] == ["slot", "tick"]
    image = dataset[0]["observation.images.observation_image"]
    assert tuple(image.shape) == (3, RECORD_SIDE, RECORD_SIDE)
    assert frames(dataset, 0) == [
        {"state": [1.0, float(tick)], "action": [float(tick), 1.0], "task": "stub task: lift"}
        for tick in range(EPISODE_TICKS)
    ]


@pytest.mark.e2e
@pytest.mark.lerobot
@pytest.mark.parametrize("repo", [None, "lab"], ids=["no-repo", "unreachable-hub"])
async def test_a_dataset_stays_on_disk_whether_or_not_its_push_succeeds(
    repo: str | None, recording: Path, hud_env: HudEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    hud_env.set(HF_REPO=repo, HF_ENDPOINT="http://127.0.0.1:9")

    await record("arm", recording_contract())
    DatasetWriter.finalize()

    (dataset,) = datasets(recording)
    assert dataset.meta.total_episodes == 1
    assert ("HF push failed" in capsys.readouterr().out) is (repo is not None)
