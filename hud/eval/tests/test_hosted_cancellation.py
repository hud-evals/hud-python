"""Hosted execution finishes bounded remote cancellation before returning."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents import OpenAIChatAgent
from hud.agents.base import Agent
from hud.agents.types import OpenAIChatConfig
from hud.environment import Environment
from hud.eval import HostedRuntime, LocalRuntime, Task, Taskset

if TYPE_CHECKING:
    from hud.eval.run import Run


class _Platform:
    api_key = "test-key"

    def __init__(self) -> None:
        self.polling = asyncio.Event()
        self.cancel_started = asyncio.Event()
        self.cancel_released = asyncio.Event()
        self.cancel_finished = asyncio.Event()
        self.accepted: list[dict[str, Any]] = []
        self.reject_cancel = False

    async def apost(self, path: str, *, json: Any = None) -> dict[str, str]:
        if path == "/rollouts/cancel":
            self.cancel_started.set()
            try:
                await self.cancel_released.wait()
                if self.reject_cancel:
                    raise RuntimeError("platform unavailable")
                self.accepted.append(json)
            finally:
                self.cancel_finished.set()
        return {"status": "queued"}

    async def aget(self, path: str) -> dict[str, str]:
        self.polling.set()
        return {"status": "running"}


def _agent() -> OpenAIChatAgent:
    return OpenAIChatAgent(
        OpenAIChatConfig(model="test", api_key="unused", base_url="http://model.test")
    )


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
async def test_remote_cancellation_finishes_before_the_hosted_run_exits(
    stop: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    platform = _Platform()
    monkeypatch.setattr("hud.eval.runtime.hosted.PlatformClient.from_settings", lambda: platform)
    pending = asyncio.create_task(
        HostedRuntime(poll_interval=0.01).run(
            Task(env="sums", id="add"),
            _agent(),
            job_id=uuid.uuid4().hex,
            rollout_timeout=0.01 if stop == "timeout" else None,
        )
    )
    await asyncio.wait_for(platform.polling.wait(), 1)
    if stop == "cancel":
        pending.cancel()
    await asyncio.wait_for(platform.cancel_started.wait(), 1)
    try:
        assert not pending.done(), "the caller exited before the platform acknowledged cancellation"
    finally:
        platform.cancel_released.set()
        await asyncio.wait_for(platform.cancel_finished.wait(), 1)
        await asyncio.gather(pending, return_exceptions=True)

    assert len(platform.accepted) == 1
    trace_id = platform.accepted[0]["trace_id"]
    assert str(uuid.UUID(trace_id)) == trace_id
    if stop == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        run = await pending
        assert run.trace.stop_reason == "timeout"
        assert run.trace_id is not None
        assert str(uuid.UUID(run.trace_id)) == trace_id


async def test_remote_cancel_failure_preserves_the_callers_cancellation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    platform = _Platform()
    platform.reject_cancel = True
    platform.cancel_released.set()
    monkeypatch.setattr("hud.eval.runtime.hosted.PlatformClient.from_settings", lambda: platform)
    pending = asyncio.create_task(
        HostedRuntime().run(Task(env="sums", id="add"), _agent(), job_id=uuid.uuid4().hex)
    )
    await asyncio.wait_for(platform.polling.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert platform.cancel_finished.is_set()
    assert "platform unavailable" in caplog.text


async def test_unresponsive_remote_cancel_does_not_hold_the_caller_forever(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    platform = _Platform()
    monkeypatch.setattr("hud.eval.runtime.hosted.PlatformClient.from_settings", lambda: platform)
    pending = asyncio.create_task(
        HostedRuntime().run(Task(env="sums", id="add"), _agent(), job_id=uuid.uuid4().hex)
    )
    await asyncio.wait_for(platform.polling.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 10)
    assert platform.cancel_finished.is_set()
    assert platform.accepted == []
    assert "cancel failed" in caplog.text


def test_remote_cancellation_survives_event_loop_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    class DelayedPlatform(_Platform):
        async def apost(self, path: str, *, json: Any = None) -> dict[str, str]:
            if path == "/rollouts/cancel":
                await asyncio.sleep(0.01)
            return await super().apost(path, json=json)

    platform = DelayedPlatform()
    platform.cancel_released.set()
    monkeypatch.setattr("hud.eval.runtime.hosted.PlatformClient.from_settings", lambda: platform)

    async def driver() -> None:
        pending = asyncio.create_task(
            Task(env="sums", id="add").run(_agent(), runtime=HostedRuntime())
        )
        await asyncio.wait_for(platform.polling.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(driver())
    assert len(platform.accepted) == 1


def test_every_remote_cancellation_finishes_before_a_batch_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BatchPlatform(_Platform):
        def __init__(self) -> None:
            super().__init__()
            self.all_polling = asyncio.Event()
            self.polls = 0
            self.cancels = 0

        async def aget(self, path: str) -> dict[str, str]:
            self.polls += 1
            if self.polls == 2:
                self.all_polling.set()
            return await super().aget(path)

        async def apost(self, path: str, *, json: Any = None) -> dict[str, str]:
            if path == "/rollouts/cancel":
                self.cancels += 1
                if self.cancels == 2:
                    await asyncio.sleep(0.01)
            return await super().apost(path, json=json)

    platform = BatchPlatform()
    platform.cancel_released.set()
    monkeypatch.setattr("hud.eval.runtime.hosted.PlatformClient.from_settings", lambda: platform)

    async def driver() -> None:
        pending = asyncio.create_task(
            Task(env="sums", id="add").run(_agent(), runtime=HostedRuntime(), group=2)
        )
        await asyncio.wait_for(platform.all_polling.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(driver())
    assert len({payload["trace_id"] for payload in platform.accepted}) == 2


async def test_a_rollout_that_cancels_itself_does_not_leave_its_sibling_running() -> None:
    waiting = asyncio.Event()
    stopped = asyncio.Event()

    class CancellingAgent(Agent):
        async def __call__(self, run: Run) -> None:
            if run.prompt_text == "cancel":
                await waiting.wait()
                raise asyncio.CancelledError
            waiting.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    def build_env(task: Task) -> Environment:
        env = Environment("cancellation")

        @env.template()
        async def work(name: str):
            yield name
            yield 1.0

        return env

    tasks = Taskset(
        "siblings",
        [Task(env="cancellation", id="work", args={"name": name}) for name in ("cancel", "wait")],
    )
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(tasks.run(CancellingAgent(), runtime=LocalRuntime(build_env)), 2)
    assert stopped.is_set()
