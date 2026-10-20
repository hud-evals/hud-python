"""``HUDRuntime``: the agent loop runs here against an environment the platform hosts,
reached through a runtime session and a WebSocket tunnel.

The fake runtime service relays each tunnel's bytes to an environment served in this
process, so a tunneled rollout really completes and its reward proves it.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents.base import Agent
from hud.eval import HUDRuntime, LocalRuntime, Task, Taskset
from tests.conftest import eventually
from tests.eval.conftest import SESSION, SESSIONS, TUNNEL, host_on_runtime
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.environments import checked
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from starlette.websockets import WebSocket

    from hud.eval import Run
    from tests.conftest import HudEnv
    from tests.fixtures.services import FakeServices


async def test_a_platform_taskset_runs_over_the_tunnel_and_releases_its_session(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    row = Task(env="checked", id="check")

    async with LocalRuntime(checked())(row) as served:
        host_on_runtime(services, served.url)
        job = await Taskset("remote", [row], taskset_id="ts-1").run(ScriptedAgent("yes"))

    (run,) = job.runs
    assert run.reward == 1.0
    (created,) = services.requests("runtime", "POST", SESSIONS)
    assert (created.bearer, created.json) == (
        "k",
        {"environment": "checked", "trace_id": str(uuid.UUID(run.trace_id))},
    )
    assert {
        (tunnel.params["id"], tunnel.bearer) for tunnel in services.requests("runtime", "WS")
    } == {("sess-1", "k")}
    assert len(services.requests("runtime", "DELETE", SESSION)) == 1


async def test_runtime_url_points_the_tunnel_at_another_runtime_service(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k", HUD_RUNTIME_URL="http://127.0.0.1:9")
    row = Task(env="checked", id="check")

    async with LocalRuntime(checked())(row) as served:
        host_on_runtime(services, served.url)
        job = await row.run(
            ScriptedAgent("yes"), runtime=HUDRuntime(runtime_url=services.url("runtime") + "/")
        )

    assert job.reward == 1.0


async def dropped(websocket: WebSocket, params: dict[str, str]) -> None:
    await websocket.accept()
    await websocket.close()


@pytest.mark.parametrize(
    ("session", "error"),
    [
        pytest.param(
            Reply(status=503, json={"detail": "no capacity"}),
            "503",
            id="a-rejected-session",
        ),
        pytest.param(
            Reply(json={"status": "ok"}),
            "did not return a session id",
            id="a-session-without-an-id",
        ),
    ],
)
async def test_a_session_the_runtime_service_does_not_grant_fails_the_run(
    session: Reply, error: str, services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    services.route("runtime", "POST", SESSIONS, session)

    job = await Task(env="checked", id="check").run(ScriptedAgent("yes"), runtime=HUDRuntime())

    (run,) = job.runs
    assert (run.trace.error or "").startswith("[provisioning] ")
    assert error in (run.trace.error or "")
    assert [request.method for request in services.requests()] == ["POST"]


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


async def test_resources_the_deployed_env_cannot_change_are_warned_about_once(
    services: FakeServices, hud_env: HudEnv, caplog: pytest.LogCaptureFixture
) -> None:
    hud_env.set(HUD_API_KEY="k")
    sized = {"image": "checked:1", "resources": {"cpu": 4, "memory_mb": 8192}}
    rows = [Task(env="checked", id="check", slug=slug, runtime_config=sized) for slug in "ab"]

    async with LocalRuntime(checked())(Task(env="checked", id="check")) as served:
        host_on_runtime(services, served.url)
        job = await Taskset("sized", rows).run(ScriptedAgent("yes"), runtime=HUDRuntime())

    assert [run.reward for run in job.runs] == [1.0, 1.0]
    (warning,) = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "resources" in warning


@pytest.mark.parametrize(
    "runtime_config",
    [
        {"resources": {"gpu": {"type": "H100"}}},
        {"resources": {"os": "windows"}},
        {"resources": {"tpu": {"type": "v5e", "topology": "2x2"}}},
        {"limits": {"startup_timeout_s": 60}},
    ],
    ids=["a-gpu", "an-os", "a-tpu", "limits"],
)
async def test_requirements_the_deployed_env_cannot_meet_fail_the_run_before_a_session(
    runtime_config: dict[str, Any], services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    row = Task(env="checked", id="check", runtime_config=runtime_config)

    job = await row.run(ScriptedAgent("yes"), runtime=HUDRuntime())

    (run,) = job.runs
    assert (run.trace.status, run.prompt, services.requests()) == ("error", None, [])
    assert (run.trace.error or "").startswith("[provisioning] ")
    assert "cannot honor" in (run.trace.error or "")


async def test_the_tunnel_needs_a_hud_api_key(services: FakeServices) -> None:
    job = await Task(env="checked", id="check").run(ScriptedAgent("yes"), runtime=HUDRuntime())

    (run,) = job.runs
    assert (run.trace.status, run.prompt, services.requests()) == ("error", None, [])
    assert (run.trace.error or "").startswith("[provisioning] ")
    assert "HUD_API_KEY" in (run.trace.error or "")


async def test_the_deprecated_run_timeout_still_ends_a_tunneled_run(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    row = Task(env="checked", id="check")

    class Waits(Agent):
        async def __call__(self, run: Run) -> None:
            await asyncio.sleep(30)

    async with LocalRuntime(checked())(row) as served:
        host_on_runtime(services, served.url)
        with pytest.warns(DeprecationWarning, match="rollout_timeout=... to Task.run"):
            runtime = HUDRuntime(run_timeout=0.5)
        job = await row.run(Waits(), runtime=runtime)
        await eventually(lambda: len(services.requests("runtime", "DELETE", SESSION)) == 1)

    assert job.runs[0].trace.stop_reason == "timeout"
    assert "during agent loop" in (job.runs[0].trace.error or "")
