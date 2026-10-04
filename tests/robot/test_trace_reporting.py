"""What a robot run reports: its traces, their outcomes, and every tick as state and video.

Two producers report robot telemetry: ``wrap`` around a vectorized gym env, and
direct control streaming each played tick of a rollout. Each is run as a user runs it;
the cases assert what the platform recorded and the trace spans it can replay.
"""

from __future__ import annotations

import asyncio
import base64
import io
import itertools
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, cast

import av
import numpy as np
import pytest

from hud import Task
from hud.agents.base import Agent
from hud.capabilities import find_capability, open_client
from hud.environment.robot import DirectControl, wrap
from hud.eval.run import rollout
from hud.runtime import LocalRuntime
from tests.conftest import ROBOT_STEP_SCHEMA, steps

from .sims import Arm, ScriptedLLM, arm_env, contract, move, reached

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hud import Run, Trace
    from hud.capabilities import MCPClient
    from tests.fixtures.services import FakeServices


class TwoArmSim:
    """Two envs: env 0 finishes on the second step (succeeding with a shaped reward of
    0.2), env 1 on the third (failing with a shaped reward of 5.0)."""

    num_envs = 2
    metadata: ClassVar[dict[str, int]] = {"render_fps": 10}
    DONES = ([False, False], [True, False], [False, True])

    def __init__(self) -> None:
        self.t = 0

    def observe(self) -> dict[str, Any]:
        return {
            "state": np.full((2, 3), self.t, dtype=np.float32),
            "cam": np.full((2, 16, 16, 3), self.t, dtype=np.uint8),
        }

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        del kwargs
        return self.observe(), {}

    def step(self, action: Any) -> tuple[Any, ...]:
        del action
        done = np.array(self.DONES[self.t])
        self.t += 1
        info = {"success": np.array([True, False])}
        return self.observe(), np.array([0.2, 5.0]), done, np.array([False, False]), info

    def close(self) -> None:
        pass


def test_a_wrapped_vectorized_sim_reports_each_episode_as_its_own_trace(
    reporting: FakeServices,
) -> None:
    env = wrap(TwoArmSim(), contract=None)
    env.reset()
    for _ in TwoArmSim.DONES:
        env.step(np.zeros((2, 2)))
    env.close()

    entered = [
        r.path.split("/")[3] for r in reporting.requests("api", "POST", "/v2/trace/{id}/enter")
    ]
    labels = {trace_id: f"trace-{n}" for n, trace_id in enumerate(entered)}
    exits = {
        labels[r.path.split("/")[3]]: r.json
        for r in reporting.requests("api", "POST", "/v2/trace/{id}/exit")
    }
    spans: dict[str, list[dict[str, Any]]] = {label: [] for label in labels.values()}
    for upload in reporting.requests("telemetry", "POST", "/trace/{id}/telemetry-upload"):
        spans[labels[upload.path.split("/")[2].replace("-", "")]].extend(upload.json["telemetry"])
    observations = {
        trace: sum(span["name"] == "step.observation" for span in recorded)
        for trace, recorded in spans.items()
    }
    assert exits == {
        "trace-0": {
            "status": "completed",
            "reward": 1.0,
            "metadata": {"env_index": 0, "episode_index": 0, "seed": None, "success": True},
        },
        "trace-1": {
            "status": "completed",
            "reward": 0.0,
            "metadata": {"env_index": 1, "episode_index": 0, "seed": None, "success": False},
        },
        "trace-2": {
            "status": "completed",
            "reward": 0.2,
            "metadata": {"env_index": 0, "episode_index": 1, "seed": None},
        },
    }
    assert observations == {"trace-0": 1, "trace-1": 2, "trace-2": 1}


# ─── direct control streams every played tick ──────────────────────────


def frames_in(segments: list[dict[str, Any]]) -> int:
    """How many frames the recorded MP4 segments decode to."""
    mp4 = b"".join(base64.b64decode(segment["segment"]["data"]) for segment in segments)
    with av.open(io.BytesIO(mp4), mode="r") as container:
        return sum(1 for _ in container.decode(video=0))


def recorded(trace: Trace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The trace's observation steps and video segments, in order."""
    payloads = steps(trace.trace_id, schema=ROBOT_STEP_SCHEMA)
    return (
        [p for p in payloads if p.get("source") == "observation"],
        [p for p in payloads if p.get("source") == "video_segment"],
    )


@pytest.mark.parametrize(("control_rate", "tick_seconds"), [(10, 0.1), (0.4, 1.0)])
async def test_every_played_tick_streams_to_the_trace_as_state_and_one_video_per_camera(
    control_rate: float, tick_seconds: float
) -> None:
    sim = Arm(contract("ee_abs", [0.0, -1.0], [1.0, 1.0], rate=control_rate), done=reached)
    agent = ScriptedLLM(move("move_to", x=0.3), move("move_to", x=0.5))

    async with arm_env(sim, DirectControl()) as env:
        trace = await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    observations, segments = recorded(trace)
    starts = [datetime.fromisoformat(obs["started_at"]) for obs in observations]
    assert {segment["camera"] for segment in segments} == {"observation/image"}
    # The opening scene, then one observation and one frame per tick of both moves:
    # the recording keeps every frame however far the encoder falls behind.
    assert frames_in(segments) == 1 + len(sim.actions)
    assert [obs["tick"] for obs in observations] == list(range(1 + len(sim.actions)))
    assert observations[-1]["state"]
    # The viewer's clock is the stamps' span: the video's length, not the calls' latency.
    assert [(b - a).total_seconds() for a, b in itertools.pairwise(starts)] == pytest.approx(
        [tick_seconds] * len(sim.actions)
    )


class AbandonsMidMove(Agent):
    """Starts a long move, then fails once the sim has stepped ``after`` ticks."""

    def __init__(self, moving: threading.Event) -> None:
        super().__init__()
        self.moving = moving

    async def __call__(self, run: Run) -> None:
        async with open_client(find_capability(run.capabilities, "control")) as opened:
            client = cast("MCPClient", opened)
            targets = {"targets": {"x": 0.5, "grip": None}, "note": "move toward the goal"}
            playing = asyncio.create_task(client.call_tool("move_to", targets))
            assert await asyncio.to_thread(self.moving.wait, 30)
            playing.cancel()
            raise RuntimeError("the agent gave up")


async def test_ending_the_episode_mid_move_ends_the_recording_with_the_ticks_played() -> None:
    moving = threading.Event()

    def announcing(arm: Arm, target: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(arm.actions) >= 3:
            moving.set()
        return target

    sim = Arm(contract("ee_abs", [0.0, -1.0], [1.0, 1.0]), follow=announcing)

    async with arm_env(sim, DirectControl(speed=0.001)) as env:
        trace = await rollout(
            Task(env="arm", id="reach"), AbandonsMidMove(moving), runtime=LocalRuntime(env)
        )

    observations, segments = recorded(trace)
    ticks = [obs["tick"] for obs in observations]
    assert trace.status == "error"
    assert ticks == list(range(len(ticks)))
    assert 3 <= len(ticks) < 1 + len(sim.actions)
    assert frames_in(segments) == len(ticks)


async def test_a_camera_the_encoder_cannot_encode_costs_the_video_not_the_rollout() -> None:
    # A 1x1 frame is a valid image for the model but crops to nothing for H.264,
    # so the encoder dies on the first frame while the move keeps streaming ticks.
    sim = Arm(contract("ee_abs", [0.0, -1.0], [1.0, 1.0]), done=reached, side=1)
    agent = ScriptedLLM(move("move_to", x=0.5), move("move_to", grip=1.0))

    async with arm_env(sim, DirectControl(max_step={"grip": 2.0})) as env:
        trace = await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    observations, segments = recorded(trace)
    assert trace.reward == 1.0
    assert [obs["tick"] for obs in observations] == list(range(1 + len(sim.actions)))
    assert segments == []
