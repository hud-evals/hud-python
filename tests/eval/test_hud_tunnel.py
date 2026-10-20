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

from hud import Environment
from hud.agents.base import Agent
from hud.eval import HUDRuntime, LocalRuntime, Task, Taskset
from tests.fixtures.agents import ScriptedAgent
from tests.support import Reply, eventually
from tests.support.platform import SESSION, SESSIONS, TUNNEL, host_on_runtime

if TYPE_CHECKING:
    from starlette.websockets import WebSocket

    from hud.eval import Run
    from tests.support import FakeServices, HudEnv


def checked() -> Environment:
    env = Environment("checked")

    @env.template()
    async def check():
        answer = yield "say yes"
        yield 1.0 if answer == "yes" else 0.0

    return env


async def test_a_platform_taskset_runs_over_the_tunnel_and_releases_its_session(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    row = Task(env="checked", id="check")

    async with LocalRuntime(checked())(row) as served:
        host_on_runtime(services, int(served.url.rsplit(":", 1)[1]))
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
    assert [request.path for request in services.requests("runtime", "DELETE")] == [
        "/runtime/sessions/sess-1"
    ]


async def test_runtime_url_points_the_tunnel_at_another_runtime_service(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k", HUD_RUNTIME_URL="http://127.0.0.1:9")
    row = Task(env="checked", id="check")

    async with LocalRuntime(checked())(row) as served:
        host_on_runtime(services, int(served.url.rsplit(":", 1)[1]))
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
            "[provisioning] httpx.HTTPStatusError: Server error '503 Service Unavailable'",
            id="a-rejected-session",
        ),
        pytest.param(
            Reply(json={"status": "ok"}),
            "[provisioning] RuntimeError: Runtime gateway did not return a session id",
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
    assert (run.trace.error or "").startswith(error)
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

    assert job.runs[0].trace.error == "rollout timed out after 0.5s during starting task"
    await eventually(lambda: len(services.requests("runtime", "DELETE", SESSION)) == 1)


async def test_resources_the_deployed_env_cannot_change_are_warned_about_once(
    services: FakeServices, hud_env: HudEnv, caplog: pytest.LogCaptureFixture
) -> None:
    hud_env.set(HUD_API_KEY="k")
    sized = {"image": "checked:1", "resources": {"cpu": 4, "memory_mb": 8192}}
    rows = [Task(env="checked", id="check", slug=slug, runtime_config=sized) for slug in "ab"]

    async with LocalRuntime(checked())(Task(env="checked", id="check")) as served:
        host_on_runtime(services, int(served.url.rsplit(":", 1)[1]))
        job = await Taskset("sized", rows).run(ScriptedAgent("yes"), runtime=HUDRuntime())

    assert [run.reward for run in job.runs] == [1.0, 1.0]
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        "HUDRuntime cannot honor task runtime_config ['resources'] on an already-deployed env; "
        "rollouts proceed on the platform's defaults"
    ]


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

    assert (job.runs[0].trace.error or "").startswith(
        "[provisioning] ValueError: HUDRuntime cannot honor this task's declared placement "
        "requirements or limits"
    )
    assert services.requests() == []


async def test_the_tunnel_needs_a_hud_api_key(services: FakeServices) -> None:
    job = await Task(env="checked", id="check").run(ScriptedAgent("yes"), runtime=HUDRuntime())

    assert job.runs[0].trace.error == (
        "[provisioning] RuntimeError: HUD runtime tunnel requires HUD_API_KEY"
    )
    assert services.requests() == []


async def test_the_deprecated_run_timeout_still_ends_a_tunneled_run(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    row = Task(env="checked", id="check")

    class Waits(Agent):
        async def __call__(self, run: Run) -> None:
            await asyncio.sleep(30)

    async with LocalRuntime(checked())(row) as served:
        host_on_runtime(services, int(served.url.rsplit(":", 1)[1]))
        with pytest.warns(DeprecationWarning, match="rollout_timeout=... to Task.run"):
            runtime = HUDRuntime(run_timeout=0.5)
        job = await row.run(Waits(), runtime=runtime)
        await eventually(lambda: len(services.requests("runtime", "DELETE", SESSION)) == 1)

    assert (job.runs[0].trace.stop_reason, job.runs[0].trace.error) == (
        "timeout",
        "rollout timed out after 0.5s during agent loop",
    )
