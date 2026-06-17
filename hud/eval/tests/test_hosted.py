"""HUD-hosted placement: agent spec, submission/polling, and scheduler dispatch.

The hosted path never opens a local connection — :class:`HUDRuntime` submits the
rollout to the platform, polls the trace until terminal, and folds the result
into a ``Run``. The scheduler (:meth:`Taskset.run`) chooses between ``HUDRuntime``
and a local provider. These tests fake the platform client at the
``PlatformClient`` seam, so they cover everything local: spec serialization,
payload shape, id canonicalization, terminal detection, timeout cancel, the
Run the caller gets back, and the dispatch.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval.run import Run
from hud.eval.runtime import HUDRuntime, Runtime
from hud.eval.task import Task
from hud.settings import settings
from hud.telemetry.context import set_trace_context


class _FakePlatform:
    """Scripted PlatformClient: records posts, serves trace states in order."""

    api_key = "test-key"

    def __init__(self, states: list[dict[str, Any]]) -> None:
        self.states = states
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.polled = 0

    async def apost(self, path: str, *, json: Any | None = None) -> Any:
        self.posts.append((path, json or {}))
        return {"status": "queued"}

    async def aget(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        state = self.states[min(self.polled, len(self.states) - 1)]
        self.polled += 1
        return state


class _FakeResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.body


def _agent() -> OpenAIChatAgent:
    return OpenAIChatAgent(
        OpenAIChatConfig(model="test-model", api_key="k", base_url="http://localhost")
    )


def test_hosted_spec_serializes_full_config() -> None:
    agent = _agent()
    agent.config.system_prompt = "be brief"
    agent.config.max_steps = 7

    spec = agent.hosted_spec()

    assert spec["type"] == "openai_compatible"
    config = spec["config"]
    # The full config travels, so every knob is preserved...
    assert config["model"] == "test-model"
    assert config["max_steps"] == 7
    assert config["system_prompt"] == "be brief"
    # ...minus what can't or shouldn't cross the wire.
    assert "model_client" not in config
    assert "api_key" not in config
    assert "base_url" not in config
    assert "hosted_tools" not in config


def test_hosted_spec_rejects_custom_model_client() -> None:
    agent = _agent()
    agent.config = OpenAIChatConfig(model="m", model_client=object())
    with pytest.raises(ValueError, match="model_client"):
        agent.hosted_spec()


@pytest.mark.asyncio
async def test_run_rejects_non_gateway_agent() -> None:
    """An agent that can't serialize its identity yields a failed Run, not a crash."""
    run = await HUDRuntime(poll_interval=0.0).run(
        Task(env="e", id="x"),
        object(),  # type: ignore[arg-type]
        job_id="j",  # type: ignore[arg-type]
    )
    assert run.trace.is_error
    assert "gateway agent" in (run.trace.error or "")


@pytest.mark.asyncio
async def test_run_submits_and_polls_to_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _FakePlatform(
        [
            {"status": "pending"},
            {"status": "running"},
            {"status": "completed", "reward": 0.5},
        ]
    )
    monkeypatch.setattr(
        "hud.eval.runtime.PlatformClient.from_settings", classmethod(lambda cls: platform)
    )

    hosted = HUDRuntime(poll_interval=0.0)
    trace_id = uuid.uuid4().hex
    job_id = uuid.uuid4().hex
    task = Task(env="sums", id="add", args={"a": 1, "b": 2})

    run = await hosted.run(task, _agent(), job_id=job_id, group_id="g1", trace_id=trace_id)

    assert run.reward == 0.5
    assert run.trace.status == "completed"
    assert run.trace.trace_id == trace_id
    assert run.job_id == job_id
    assert run.group_id == "g1"
    assert platform.polled == 3
    (path, payload) = platform.posts[0]
    assert path == "/rollouts/submit"
    # Hex ids travel as canonical UUID strings.
    assert payload["trace_id"] == str(uuid.UUID(trace_id))
    assert payload["job_id"] == str(uuid.UUID(job_id))
    assert payload["env"] == "sums"
    assert payload["task"] == "add"
    assert payload["args"] == {"a": 1, "b": 2}
    assert payload["group_id"] == "g1"
    assert payload["agent"]["type"] == "openai_compatible"
    assert payload["agent"]["config"]["model"] == "test-model"


@pytest.mark.asyncio
async def test_run_timeout_requests_platform_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _FakePlatform([{"status": "running"}])
    monkeypatch.setattr(
        "hud.eval.runtime.PlatformClient.from_settings", classmethod(lambda cls: platform)
    )

    hosted = HUDRuntime(poll_interval=0.0, run_timeout=0.0)
    task = Task(env="sums", id="add", args={})

    with pytest.raises(TimeoutError, match="hosted rollout"):
        await hosted.run(task, _agent(), job_id=uuid.uuid4().hex)

    cancel_posts = [(p, b) for p, b in platform.posts if p == "/rollouts/cancel"]
    assert len(cancel_posts) == 1


@pytest.mark.asyncio
async def test_run_folds_completed_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _FakePlatform([{"status": "completed", "reward": 1.0, "error": None}])
    monkeypatch.setattr(
        "hud.eval.runtime.PlatformClient.from_settings", classmethod(lambda cls: platform)
    )

    task = Task(env="sums", id="add", args={"a": 2, "b": 3})
    run = await HUDRuntime(poll_interval=0.0).run(task, _agent(), job_id=uuid.uuid4().hex)

    assert run.reward == 1.0
    assert run.trace.status == "completed"
    assert not run.trace.is_error
    assert run.runtime == f"hud://trace/{run.trace.trace_id}"
    # The platform owns the trace lifecycle: no local client ever existed.
    with pytest.raises(RuntimeError, match="no live client"):
        _ = run.client


@pytest.mark.asyncio
async def test_run_folds_error_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _FakePlatform([{"status": "error", "reward": None, "error": "env exploded"}])
    monkeypatch.setattr(
        "hud.eval.runtime.PlatformClient.from_settings", classmethod(lambda cls: platform)
    )

    task = Task(env="sums", id="add", args={})
    run = await HUDRuntime(poll_interval=0.0).run(task, _agent(), job_id=uuid.uuid4().hex)

    assert run.reward == 0.0
    assert run.trace.is_error
    assert "env exploded" in (run.trace.error or "")


@pytest.mark.asyncio
async def test_scheduler_drives_provider_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Provider placement goes through the local rollout atom, not HUDRuntime."""
    import hud.eval.taskset as taskset_mod
    from hud.eval.taskset import Taskset

    seen: dict[str, Any] = {}

    async def fake_rollout(task: Task, agent: Any, **kwargs: Any) -> Run:
        seen.update(kwargs)
        run = Run(None, task.id, {})
        run.trace.status = "completed"
        return run

    monkeypatch.setattr(taskset_mod, "rollout", fake_rollout)

    job = await Taskset("t", [Task(env="e", id="x")]).run(
        _agent(), runtime=Runtime("tcp://127.0.0.1:1")
    )

    assert len(job.runs) == 1
    assert isinstance(seen["runtime"], Runtime)
    assert "job_id" in seen and "group_id" in seen


@pytest.mark.asyncio
async def test_scheduler_delegates_hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A HUDRuntime placement is delegated to via HUDRuntime.run, not the local atom."""
    from hud.eval.taskset import Taskset

    seen: dict[str, Any] = {}

    class _RecordingHUDRuntime(HUDRuntime):
        async def run(self, task: Task, agent: Any, **kwargs: Any) -> Run:  # type: ignore[override]
            seen.update(kwargs)
            run = Run(None, task.id, {})
            run.trace.status = "completed"
            return run

    job = await Taskset("t", [Task(env="e", id="x")]).run(_agent(), runtime=_RecordingHUDRuntime())

    assert len(job.runs) == 1
    assert "job_id" in seen and "group_id" in seen


@pytest.mark.asyncio
async def test_cloud_mode_drives_local_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_rollout(task: Task, agent: Any, **kwargs: Any) -> Run:
        seen.update(kwargs)
        run = Run(None, task.id, {})
        run.trace.status = "completed"
        return run

    monkeypatch.setattr("hud.eval.runtime.rollout", fake_rollout)

    cloud = HUDRuntime(mode="cloud")
    job_id = uuid.uuid4().hex
    trace_id = uuid.uuid4().hex
    run = await cloud.run(
        Task(env="e", id="x"),
        _agent(),
        job_id=job_id,
        group_id="g1",
        trace_id=trace_id,
    )

    assert run.trace.status == "completed"
    assert seen["runtime"] is cloud
    assert seen["job_id"] == job_id
    assert seen["group_id"] == "g1"
    assert seen["trace_id"] == trace_id


@pytest.mark.asyncio
async def test_cloud_session_includes_active_trace_id(monkeypatch: pytest.MonkeyPatch) -> None:
    posts: list[dict[str, Any]] = []
    session_id = str(uuid.uuid4())

    class _RecordingAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _RecordingAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(
            self,
            path: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> _FakeResponse:
            posts.append({"path": path, "headers": headers, "json": json})
            return _FakeResponse({"id": session_id})

    monkeypatch.setattr("hud.eval.runtime.httpx.AsyncClient", _RecordingAsyncClient)

    trace_id = uuid.uuid4().hex
    with set_trace_context(trace_id):
        created = await HUDRuntime(mode="cloud")._create_cloud_session(
            "https://mcp.hud.ai",
            "sk-hud-test",
            Task(env="e", id="x"),
        )

    assert created == session_id
    assert posts == [
        {
            "path": "https://mcp.hud.ai/runtime/sessions",
            "headers": {"Authorization": "Bearer sk-hud-test"},
            "json": {"environment": "e", "trace_id": str(uuid.UUID(trace_id))},
        }
    ]


@pytest.mark.asyncio
async def test_cloud_session_sets_runtime_connection_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = str(uuid.uuid4())
    deleted: list[tuple[str, str, str]] = []

    class _Socket:
        def getsockname(self) -> tuple[str, int]:
            return ("127.0.0.1", 4321)

    class _Server:
        sockets = [_Socket()]

        def __init__(self) -> None:
            self.closed = False
            self.waited = False

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            self.waited = True

    server = _Server()

    async def fake_start_server(*args: Any, **kwargs: Any) -> _Server:
        return server

    async def fake_create_cloud_session(
        self: HUDRuntime,
        runtime_url: str,
        api_key: str,
        task: Task,
    ) -> str:
        assert runtime_url == "https://mcp.hud.ai"
        assert api_key == "sk-hud-test"
        assert task.env == "e"
        return session_id

    async def fake_delete_cloud_session(
        self: HUDRuntime,
        runtime_url: str,
        api_key: str,
        session: str,
    ) -> None:
        deleted.append((runtime_url, api_key, session))

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")
    monkeypatch.setattr("hud.eval.runtime.asyncio.start_server", fake_start_server)
    monkeypatch.setattr(HUDRuntime, "_create_cloud_session", fake_create_cloud_session)
    monkeypatch.setattr(HUDRuntime, "_delete_cloud_session", fake_delete_cloud_session)

    cloud = HUDRuntime(mode="cloud", runtime_url="https://mcp.hud.ai/", run_timeout=600.0)
    async with cloud._cloud_session(Task(env="e", id="x")) as runtime:
        assert runtime.url == "tcp://127.0.0.1:4321"
        assert runtime.params == {
            "session_id": session_id,
            "gateway_url": "https://mcp.hud.ai",
            "ready_timeout": 300.0,
        }

    assert deleted == [("https://mcp.hud.ai", "sk-hud-test", session_id)]
    assert server.closed
    assert server.waited
