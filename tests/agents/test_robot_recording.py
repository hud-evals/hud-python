# ruff: noqa: E501 -- snapshots quote product messages verbatim
"""``RobotAgent(save=True)`` records each rollout as a LeRobot episode.

A stub sim serves the ``robot`` capability and a stub policy acts on it; the
``lerobot`` package is a stand-in on the import path that writes each dataset
it is asked to create as JSON under ``RECORD_DIR``. Rows pin the datasets on
disk: which rollouts share one, the features taken from the contract, the
frames (observation, executed action) per tick, failed rollouts kept, and the
Hub push.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves at runtime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from inline_snapshot import snapshot

from hud.agents.robot import DatasetWriter, Model, RobotAgent
from hud.environment import Environment
from hud.environment.robot import RobotBridge, RobotEndpoint
from hud.eval import LocalRuntime, Task, rollout

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from numpy.typing import NDArray

    from hud.eval import Run
    from tests.harness import HudEnv

FAKE_LEROBOT = Path(__file__).parent / "fake_lerobot"
TICKS = 3


def contract(*, rate: int = 5, robot: str = "arm", **features: dict[str, Any]) -> dict[str, Any]:
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
    """One camera and a 2-D state ``[tick, 0.5]``; the episode ends after ``TICKS``."""

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
                data[name] = np.full((1, 2, 2, 3), self.tick, dtype=np.uint8)
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


def datasets(record_dir: Path) -> list[dict[str, Any]]:
    """Every dataset written under ``record_dir``, oldest first, without its timestamped name."""
    found = [json.loads(path.read_text()) for path in sorted(record_dir.glob("*/dataset.json"))]
    for dataset in found:
        dataset.pop("repo_id")
    return found


@pytest.fixture
def recording(hud_env: HudEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """``RECORD_DIR`` for this test with the stand-in ``lerobot`` importable. Open
    datasets are finalized and the stand-in unloaded afterwards, so neither
    outlives the test."""
    monkeypatch.syspath_prepend(str(FAKE_LEROBOT))
    record_dir = tmp_path / "records"
    hud_env.set(RECORD_DIR=str(record_dir))
    yield record_dir
    DatasetWriter.finalize()
    for name in [name for name in sys.modules if name.split(".")[0] == "lerobot"]:
        del sys.modules[name]


async def test_a_recorded_rollout_saves_one_episode_of_observed_and_executed_frames(
    recording: Path,
) -> None:
    async with sim_env("arm", contract()) as env:
        run = await record(env)
    DatasetWriter.finalize()

    assert run.reward == 1.0
    assert datasets(recording) == snapshot(
        [
            {
                "fps": 5,
                "robot_type": "arm",
                "use_videos": True,
                "features": {
                    "observation.images.observation_image": {
                        "dtype": "video",
                        "shape": [2, 2, 3],
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [2],
                        "names": ["tick", "half"],
                    },
                    "action": {"dtype": "float32", "shape": [2], "names": ["a0", "a1"]},
                },
                "episodes": [
                    [
                        {
                            "observation.images.observation_image": "<image 2x2x3>",
                            "observation.state": [0.0, 0.5],
                            "action": [0.0, 1.0],
                            "task": "pick it up",
                        },
                        {
                            "observation.images.observation_image": "<image 2x2x3>",
                            "observation.state": [1.0, 0.5],
                            "action": [1.0, 1.0],
                            "task": "pick it up",
                        },
                        {
                            "observation.images.observation_image": "<image 2x2x3>",
                            "observation.state": [2.0, 0.5],
                            "action": [2.0, 1.0],
                            "task": "pick it up",
                        },
                    ]
                ],
                "finalized": True,
                "pushed": None,
            }
        ]
    )


async def test_concurrent_rollouts_of_one_contract_share_a_dataset(recording: Path) -> None:
    async with sim_env("left", contract()) as left, sim_env("right", contract()) as right:
        await asyncio.gather(record(left), record(right))
    DatasetWriter.finalize()

    (dataset,) = datasets(recording)
    assert [len(episode) for episode in dataset["episodes"]] == [TICKS, TICKS]


async def test_a_different_rate_or_robot_gets_its_own_dataset(recording: Path) -> None:
    async with (
        sim_env("five", contract(rate=5)) as five,
        sim_env("ten", contract(rate=10)) as ten,
        sim_env("other", contract(robot="other")) as other,
    ):
        await asyncio.gather(record(five), record(ten), record(other))
    DatasetWriter.finalize()

    assert sorted((d["fps"], d["robot_type"], len(d["episodes"])) for d in datasets(recording)) == [
        (5, "arm", 1),
        (5, "other", 1),
        (10, "arm", 1),
    ]


async def test_a_failed_rollout_keeps_the_frames_it_recorded(recording: Path) -> None:
    async with sim_env("arm", contract()) as env:
        run = await record(env, Recorder(Echo(fail_at=2)))
    DatasetWriter.finalize()

    (dataset,) = datasets(recording)
    assert run.trace.status == "error"
    assert [frame["action"] for frame in dataset["episodes"][0]] == [[0.0, 1.0], [1.0, 1.0]]


async def test_nested_camera_names_stay_distinct_features(recording: Path) -> None:
    stereo = contract(
        **{
            "left/image": {"role": "observation", "type": "rgb"},
            "right/image": {"role": "observation", "type": "rgb"},
            "state": {"role": "observation", "names": ["tick", "half"]},
            "action": {"role": "action", "names": ["a0", "a1"]},
        }
    )
    async with sim_env("stereo", stereo) as env:
        await record(env)
    DatasetWriter.finalize()

    (dataset,) = datasets(recording)
    assert sorted(dataset["features"]) == [
        "action",
        "observation.images.left_image",
        "observation.images.right_image",
        "observation.state",
    ]


async def test_features_that_flatten_to_one_key_fail_the_rollout(recording: Path) -> None:
    clash = contract(
        **{
            "cam/rgb": {"role": "observation", "type": "rgb"},
            "cam_rgb": {"role": "observation", "type": "rgb"},
            "state": {"role": "observation", "names": ["tick", "half"]},
            "action": {"role": "action", "names": ["a0", "a1"]},
        }
    )
    async with sim_env("clash", clash) as env:
        run = await record(env)

    assert run.trace.error == snapshot(
        "[agent loop] ValueError: contract features 'cam/rgb' and 'cam_rgb' both map to LeRobot key 'observation.images.cam_rgb'"
    )
    assert datasets(recording) == []


PUSHES = {
    "no-repo": ({}, snapshot((True, None, []))),
    "repo": ({"HF_REPO": "lab"}, snapshot((True, {"private": False}, []))),
    "private-repo": (
        {"HF_REPO": "lab", "HF_PRIVATE": "1"},
        snapshot((True, {"private": True}, [])),
    ),
    "push-fails": (
        {"HF_REPO": "lab", "FAKE_LEROBOT_PUSH_FAILS": "1"},
        snapshot(
            (
                True,
                None,
                [
                    "[agent] WARNING: HF push failed: ConnectionError('hub unreachable') (dataset still on disk)"
                ],
            )
        ),
    ),
}


@pytest.mark.parametrize(("variables", "expected"), PUSHES.values(), ids=PUSHES.keys())
async def test_finalizing_pushes_to_the_hub_only_with_a_repo_and_keeps_the_dataset(
    variables: dict[str, str],
    expected: Any,
    recording: Path,
    hud_env: HudEnv,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hud_env.set(**variables)
    async with sim_env("arm", contract()) as env:
        await record(env)
    DatasetWriter.finalize()

    (dataset,) = datasets(recording)
    printed = [line for line in capsys.readouterr().out.splitlines() if "HF push" in line]
    assert (dataset["finalized"], dataset["pushed"], printed) == expected


async def test_without_lerobot_recording_warns_and_the_rollout_still_runs(
    hud_env: HudEnv, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    hud_env.set(RECORD_DIR=str(tmp_path / "records"))

    with caplog.at_level(logging.WARNING, logger="hud.agents.robot.dataset"):
        async with sim_env("arm", contract()) as env:
            run = await record(env)

    assert run.reward == 1.0
    assert [record.message for record in caplog.records if "lerobot" in record.message] == [
        "save=True but lerobot is not installed; streaming telemetry only "
        "(pip install 'lerobot[dataset]')"
    ]
    assert not (tmp_path / "records").exists()
