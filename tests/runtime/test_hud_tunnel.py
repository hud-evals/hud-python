"""``HUDRuntime``: the agent loop runs here against an environment the platform hosts,
reached through a runtime session and a WebSocket tunnel.

The fake runtime service relays each tunnel's bytes to an environment served in this
process, so a tunneled rollout really completes and its reward proves it.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from hud import Task, Taskset
from hud.runtime import HUDRuntime, LocalRuntime
from tests.conftest import eventually
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.environments import checked
from tests.runtime.conftest import SESSION, SESSIONS, TUNNEL, host_on_runtime

if TYPE_CHECKING:
    from starlette.websockets import WebSocket

    from tests.conftest import HudEnv
    from tests.fixtures.services import FakeServices


async def test_a_reported_run_opens_its_hosted_environment_under_its_trace(
    reporting: FakeServices,
) -> None:
    row = Task(env="checked", id="check")

    async with LocalRuntime(checked())(row) as served:
        host_on_runtime(reporting, served.url)
        job = await Taskset("remote", [row], taskset_id="ts-1").run(ScriptedAgent("yes"))

    (trace,) = job.traces
    assert trace.reward == 1.0
    trace_id = str(uuid.UUID(trace.trace_id))
    (entered,) = reporting.requests("api", "POST", "/v2/trace/{id}/enter")
    assert entered.path == f"/v2/trace/{trace.trace_id}/enter"
    (created,) = reporting.requests("runtime", "POST", SESSIONS)
    assert (created.bearer, created.json) == ("k", {"environment": "checked", "trace_id": trace_id})
    assert {
        (tunnel.params["id"], tunnel.bearer) for tunnel in reporting.requests("runtime", "WS")
    } == {("sess-1", "k")}
    assert len(reporting.requests("runtime", "DELETE", SESSION)) == 1


async def test_with_telemetry_off_the_hosted_environment_opens_without_a_trace(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="0")
    row = Task(env="checked", id="check")

    async with LocalRuntime(checked())(row) as served:
        host_on_runtime(services, served.url)
        job = await Taskset("remote", [row], taskset_id="ts-1").run(ScriptedAgent("yes"))

    (trace,) = job.traces
    assert trace.reward == 1.0
    (created,) = services.requests("runtime", "POST", SESSIONS)
    assert created.json == {"environment": "checked"}
    assert services.requests("api", "POST", "/v2/trace/{id}/enter") == []
    assert len(services.requests("runtime", "DELETE", SESSION)) == 1


async def test_a_refused_session_fails_the_run_with_the_platform_reason(
    reporting: FakeServices,
) -> None:
    reason = "Trace 1234 is not one of the caller's traces"
    reporting.route(
        "runtime",
        "POST",
        SESSIONS,
        status=400,
        json={"error": "bad_request", "source": "user", "detail": reason},
    )

    job = await Task(env="checked", id="check").run(ScriptedAgent("yes"), runtime=HUDRuntime())

    (trace,) = job.traces
    assert trace.grade.is_error
    assert trace.error is not None
    assert reason in trace.error
    assert "'checked'" in trace.error


async def dropped(websocket: WebSocket, params: dict[str, str]) -> None:
    await websocket.accept()
    await websocket.close()


async def test_a_tunnel_that_keeps_dropping_ends_at_the_deadline_and_releases_the_session(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    services.route("runtime", "POST", SESSIONS, json={"id": "sess-1"})
    services.route("runtime", "DELETE", SESSION, json={})
    services.websocket("runtime", TUNNEL, dropped)

    job = await Task(env="checked", id="check").run(
        ScriptedAgent("yes"), runtime=HUDRuntime(), rollout_timeout=0.5
    )

    (trace,) = job.traces
    assert trace.stop_reason == "timeout"
    assert trace.error is not None
    assert "during starting task" in trace.error
    await eventually(lambda: len(services.requests("runtime", "DELETE", SESSION)) == 1)
