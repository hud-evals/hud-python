"""The HUD runtime tunnel: the agent loop runs here against an environment the
platform hosts, reached through a runtime session and a WebSocket tunnel.

The fake runtime service relays each tunnel's bytes to an environment served in
this process, so a tunneled rollout really completes and its reward proves it.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.eval import (
    HUDRuntime,
    LocalRuntime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    Task,
    Taskset,
)
from tests.eval.envs import eventually, lab, solve
from tests.harness import Reply, ScriptedAgent
from tests.harness.runtime import SESSION, SESSIONS, TUNNEL, host_on_runtime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.websockets import WebSocket

    from tests.harness import FakeServices, HudEnv

ADD = Task(env="lab", id="add", args={"a": 2, "b": 3})


async def dropped(websocket: WebSocket, params: dict[str, str]) -> None:
    del params
    await websocket.accept()
    await websocket.close()


@contextlib.asynccontextmanager
async def hosted_env(services: FakeServices, hud_env: HudEnv) -> AsyncIterator[None]:
    """Serve ``lab`` here and let the fake runtime service tunnel to it."""
    hud_env.set(HUD_API_KEY="k")
    async with LocalRuntime(lab())(Task(env="lab", id="serve")) as served:
        host_on_runtime(services, int(served.url.rsplit(":", 1)[1]))
        yield


async def test_a_platform_taskset_runs_over_the_tunnel_by_default(
    services: FakeServices, hud_env: HudEnv
) -> None:
    async with hosted_env(services, hud_env):
        job = await Taskset("remote", [ADD], taskset_id="ts-1").run(ScriptedAgent(solve))

    (run,) = job.runs
    assert run.reward == 1.0
    (create,) = services.requests("runtime", "POST", SESSIONS)
    assert (create.bearer, create.json) == (
        "k",
        {"environment": "lab", "trace_id": str(uuid.UUID(run.trace_id))},
    )
    tunnels = services.requests("runtime", "WS", TUNNEL)
    assert tunnels != []
    assert {(tunnel.params["id"], tunnel.bearer) for tunnel in tunnels} == {("sess-1", "k")}
    assert [request.path for request in services.requests("runtime", "DELETE")] == [
        "/runtime/sessions/sess-1"
    ]


async def test_an_acquired_tunnel_is_a_local_address_for_the_session(
    services: FakeServices, hud_env: HudEnv
) -> None:
    async with hosted_env(services, hud_env), HUDRuntime()(ADD) as runtime:
        assert services.requests("runtime", "DELETE") == []

    assert (runtime.url, runtime.params) == (
        IsStr(regex=r"tcp://127\.0\.0\.1:\d+"),
        {
            "session_id": "sess-1",
            "gateway_url": services.url("runtime"),
            "ready_timeout": 300.0,
        },
    )
    assert services.bodies("runtime", "POST", SESSIONS) == [{"environment": "lab"}]
    assert len(services.requests("runtime", "DELETE", SESSION)) == 1


GPU = RuntimeConfig(resources=RuntimeResources(gpu=RuntimeGPU(type="H100")))
REFUSED = (
    "[provisioning] ValueError: HUDRuntime cannot honor this task's declared placement "
    "requirements or limits on an already-deployed env; run it on a placement that provisions them"
)
FAILURES: dict[str, tuple[Task, dict[str, Any], Any, list[str]]] = {
    "a GPU it cannot provision is refused before a session": (
        ADD.model_copy(update={"runtime_config": GPU}),
        {},
        REFUSED,
        [],
    ),
    "explicit limits are refused before a session": (
        ADD.model_copy(
            update={"runtime_config": RuntimeConfig(limits=RuntimeLimits(run_timeout_s=60))}
        ),
        {},
        REFUSED,
        [],
    ),
    "no API key": (
        ADD,
        {"key": None},
        "[provisioning] RuntimeError: HUD runtime tunnel requires HUD_API_KEY",
        [],
    ),
    "a rejected session": (
        ADD,
        {"session": Reply(status=503, json={"detail": "no capacity"})},
        IsStr(regex=r"(?s)\[provisioning\] httpx\.HTTPStatusError: Server error '503 .*"),
        ["POST"],
    ),
    "a session without an id": (
        ADD,
        {"session": Reply(json={"status": "ok"})},
        "[provisioning] RuntimeError: Runtime gateway did not return a session id",
        ["POST"],
    ),
    "a tunnel the gateway keeps dropping, until the deadline": (
        ADD,
        {"tunnel": dropped, "rollout_timeout": 0.5},
        "rollout timed out after 0.5s during starting task",
        ["POST", "WS", "DELETE"],
    ),
    "an agent that raises": (
        ADD,
        {"agent": ScriptedAgent(fail_before=RuntimeError("agent exploded"))},
        "[agent loop] RuntimeError: agent exploded",
        ["POST", "WS", "DELETE"],
    ),
}


@pytest.mark.parametrize(
    ("row", "change", "error", "calls"), FAILURES.values(), ids=FAILURES.keys()
)
async def test_a_failed_tunnel_rollout_is_an_errored_run_and_its_session_is_released(
    row: Task,
    change: dict[str, Any],
    error: Any,
    calls: list[str],
    services: FakeServices,
    hud_env: HudEnv,
) -> None:
    async with hosted_env(services, hud_env):
        hud_env.set(HUD_API_KEY=change.get("key", "k"))
        if "session" in change:
            services.route("runtime", "POST", SESSIONS, change["session"])
        if "tunnel" in change:
            services.websocket("runtime", TUNNEL, change["tunnel"])
        job = await row.run(
            change.get("agent", ScriptedAgent(solve)),
            runtime=HUDRuntime(),
            rollout_timeout=change.get("rollout_timeout"),
        )

    (run,) = job.runs
    assert (run.trace.status, run.trace.error) == ("error", error)

    # A deadline returns before teardown finishes; the session is released after.
    def methods() -> list[str]:
        return list(dict.fromkeys(request.method for request in services.requests()))

    await eventually(lambda: methods() == calls, within=5)


async def test_resources_the_deployed_env_ignores_are_warned_about_once(
    services: FakeServices, hud_env: HudEnv, caplog: pytest.LogCaptureFixture
) -> None:
    sized = RuntimeConfig(image="lab:1", resources=RuntimeResources(cpu=4, memory_mb=8192))
    rows = [Task(env="lab", id="add", args={"a": a, "b": 1}, runtime_config=sized) for a in (1, 2)]

    async with hosted_env(services, hud_env):
        job = await Taskset("sized", rows).run(ScriptedAgent(solve), runtime=HUDRuntime())

    assert [run.reward for run in job.runs] == [1.0, 1.0]
    assert [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ] == [
        "HUDRuntime cannot honor task runtime_config ['resources'] on an already-deployed env; "
        "rollouts proceed on the platform's defaults"
    ]


async def test_the_deprecated_constructor_timeout_is_the_default_deadline(
    services: FakeServices, hud_env: HudEnv
) -> None:
    with pytest.warns(DeprecationWarning, match="rollout_timeout=... to Task.run"):
        runtime = HUDRuntime(run_timeout=0.3)

    async with hosted_env(services, hud_env):
        job = await ADD.run(ScriptedAgent(solve, linger=30), runtime=runtime)

    (run,) = job.runs
    assert (run.trace.stop_reason, run.trace.error) == (
        "timeout",
        "rollout timed out after 0.3s during agent loop",
    )
