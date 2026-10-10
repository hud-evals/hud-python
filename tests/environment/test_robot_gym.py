"""Spawned sims and gym shapes: what a sim process reports and what its env receives.

``RobotEndpoint(bridge)`` and ``env.gym(factory)`` both run the sim in a child
process. These scenarios spawn the sims in ``robot_sims`` and drive them through
the public surface: the endpoint's control calls, the served env's ``robot``
binding, and ``RobotClient``. A gym env reports in its observation the shape of
the action it was stepped with. The recorder rows check the platform reports a
shared job id produces.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import websockets
from openpi_client import msgpack_numpy

from hud.capabilities.robot import RobotClient
from hud.environment import Environment
from hud.environment.robot import RobotEndpoint
from hud.telemetry.robot import JobRecorder
from tests.harness import served

from .robot_sims import Reach, plain_env, vector_env

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.harness import FakeServices, HudEnv

CONTRACT = {"control_rate": 10, "features": {"action": {"role": "action", "names": ["a"]}}}


async def test_a_spawned_bridge_carries_its_declaration_into_the_child() -> None:
    declared = Reach(use_delta=True)
    declared.contract = CONTRACT
    declared.num_envs = 2
    declared.metadata = {"backend": "test"}
    endpoint = RobotEndpoint(declared)

    await endpoint.start()
    try:
        contract = await endpoint.contract()
        (robot,) = await endpoint.capabilities()
        first = await endpoint.reset()
        second = await endpoint.reset()
        async with websockets.connect(robot.url) as socket:
            greeting = msgpack_numpy.unpackb(await socket.recv())
    finally:
        await endpoint.stop()

    assert contract == CONTRACT
    assert (robot.name, robot.protocol, robot.params["contract"]) == ("robot", "openpi/0", CONTRACT)
    assert first["prompt"] == "reach with use_delta=True"
    assert [first["token"][:7], second["token"][:7]] == ["slot-0-", "slot-1-"]
    assert greeting == {"backend": "test", "claim_required": True}
    with pytest.raises(OSError):
        await websockets.connect(robot.url, open_timeout=5)


async def test_a_sim_that_exits_before_announcing_its_port_fails_start() -> None:
    endpoint = RobotEndpoint.spawn([sys.executable, "-c", "import sys; sys.exit(3)"])

    with pytest.raises(RuntimeError, match="exited with code 3 before announcing its port"):
        await endpoint.start()


def gym_env(factory: Callable[..., Any], **defaults: Any) -> Environment:
    env = Environment("gym")
    sim = env.gym(factory, contract=None, **defaults)

    @env.template()
    async def episode():
        claimed = await sim.reset()
        yield {"prompt": claimed["prompt"], "bindings": {"robot": {"token": claimed["token"]}}}
        yield await sim.result(token=claimed["token"])

    return env


@pytest.mark.parametrize(
    ("factory", "defaults", "action", "received", "reward"),
    [
        pytest.param(plain_env, {}, [0.1, 0.2, 0.3, 0.4], [2.0, 2.0], 0.5, id="box-2x2-action"),
        pytest.param(
            vector_env, {"num_envs": 2}, [0.1, 0.2], [2.0, 2.0], 0.25, id="kwargs-vector-env"
        ),
    ],
)
async def test_a_flat_action_reaches_the_gym_env_in_its_own_shape(
    factory: Callable[..., Any],
    defaults: dict[str, Any],
    action: list[float],
    received: list[float],
    reward: float,
) -> None:
    async with served(gym_env(factory, **defaults)) as client:
        started = await client.start_task("episode")
        agent = await RobotClient.connect(
            client.binding("robot"), token=started["bindings"]["robot"]["token"]
        )
        try:
            first = await agent.get_observation()
            await agent.send_action(action)
            stepped = await agent.get_observation()
        finally:
            await agent.close()
        graded = await client.grade({"answer": "done"})

    assert sorted(first["data"]) == sorted(stepped["data"])
    assert stepped["data"]["received"].tolist() == received
    assert float(stepped["reward"]) == reward
    assert stepped["terminated"] is False
    assert graded == {"score": reward, "success": False, "total_reward": reward}


async def test_a_plain_env_observation_is_sliced_to_one_slot_with_its_frames() -> None:
    async with served(gym_env(plain_env)) as client:
        started = await client.start_task("episode")
        agent = await RobotClient.connect(
            client.binding("robot"), token=started["bindings"]["robot"]["token"]
        )
        try:
            observation = await agent.get_observation()
        finally:
            await agent.close()
        await client.cancel()

    shapes = {name: np.shape(value) for name, value in observation["data"].items()}
    assert shapes == {"received": (1,), "camera": (4, 4, 3)}
    assert float(observation["reward"]) == 0.0


COMPACT_JOB = "03dd2a73d3df4d10a54ae3d87c2d530d"
CANONICAL_JOB = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"


@pytest.fixture
def platform(services: FakeServices, hud_env: HudEnv) -> FakeServices:
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1")
    for path in ("/v2/trace/job/{id}/enter", "/v2/trace/{id}/enter", "/v2/trace/{id}/exit"):
        services.route("api", "POST", path, json={})
    services.route("telemetry", "POST", "/trace/{id}/telemetry-upload", json={})
    return services


@pytest.mark.parametrize("job_id", [COMPACT_JOB, CANONICAL_JOB])
def test_a_shared_job_id_is_reported_canonically(platform: FakeServices, job_id: str) -> None:
    recorder = JobRecorder("suite", 1, record_indices=[], job_id=job_id)

    assert recorder.job_url == f"{platform.url('web')}/jobs/{CANONICAL_JOB}"
    assert [request.path for request in platform.requests("api")] == [
        f"/v2/trace/job/{CANONICAL_JOB}/enter"
    ]
    assert platform.bodies("api", "POST", "/v2/trace/job/{id}/enter") == [
        {"name": "suite", "group": 1}
    ]


@pytest.mark.parametrize(
    ("job_id", "trace_id"),
    [
        (COMPACT_JOB, "e9266339c6fa58b08102ce7f41c4f372"),
        (CANONICAL_JOB, "7b3269f689b65c5887542a9bc1ffb5c8"),
    ],
)
def test_a_seeded_recording_keeps_the_trace_id_of_its_job_id_spelling(
    platform: FakeServices, job_id: str, trace_id: str
) -> None:
    JobRecorder("suite", 1, record_indices=[0], seed=7, job_id=job_id).record(done=np.array([True]))

    entered = platform.requests("api", "POST", "/v2/trace/{id}/enter")
    assert [(request.params["id"], request.json) for request in entered] == [
        (trace_id, {"job_id": CANONICAL_JOB})
    ]
    exited = platform.bodies("api", "POST", "/v2/trace/{id}/exit")
    assert exited == [
        {
            "status": "completed",
            "reward": 0.0,
            "metadata": {"env_index": 0, "episode_index": 0, "seed": 7},
        }
    ]


def test_a_job_id_must_be_a_uuid() -> None:
    with pytest.raises(ValueError, match="job_id must be a UUID"):
        JobRecorder("suite", 1, record_indices=[], job_id="shared-suite")
