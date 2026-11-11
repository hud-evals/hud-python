"""``HUDRuntime``: the agent loop runs here against an environment the platform hosts,
reached through a runtime session and a WebSocket tunnel.

The fake runtime service relays each tunnel's bytes to an environment served in this
process, so a tunneled rollout really completes and its reward proves it.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from hud.eval import HUDRuntime, LocalRuntime, Task, Taskset
from tests.conftest import eventually
from tests.eval.conftest import SESSION, SESSIONS, TUNNEL, host_on_runtime
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.environments import checked

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

    (run,) = job.runs
    assert run.reward == 1.0
    trace_id = str(uuid.UUID(run.trace_id))
    (entered,) = reporting.requests("api", "POST", "/v2/trace/{id}/enter")
    assert entered.path == f"/v2/trace/{run.trace_id}/enter"
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

    assert job.runs[0].reward == 1.0
    (created,) = services.requests("runtime", "POST", SESSIONS)
    assert created.json == {"environment": "checked"}
    assert services.requests("api", "POST", "/v2/trace/{id}/enter") == []
    assert len(services.requests("runtime", "DELETE", SESSION)) == 1


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

    assert job.runs[0].trace.stop_reason == "timeout"
    assert "during starting task" in (job.runs[0].trace.error or "")
    await eventually(lambda: len(services.requests("runtime", "DELETE", SESSION)) == 1)
