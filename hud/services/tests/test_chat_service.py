from __future__ import annotations

from typing import Any

import pytest

from hud.eval.task import Task
from hud.services.chat_service import ChatService
from hud.types import Trace


class FakeQueue:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


class FakeContext:
    def __init__(
        self,
        text: str,
        *,
        context_id: str = "ctx-1",
        task_id: str = "task-1",
        message_id: str = "msg-1",
    ) -> None:
        self.context_id = context_id
        self.task_id = task_id
        self.message = type("Msg", (), {"message_id": message_id})
        self._text = text

    def get_user_input(self) -> str:
        return self._text


def _task(scenario: str = "test-env:analysis_chat") -> Task:
    return Task(env={"name": "test-env"}, scenario=scenario)


def test_init_stores_task_and_model() -> None:
    task = _task()
    svc = ChatService(task, model="gpt-4o")
    assert svc._task is task
    assert svc._model == "gpt-4o"
    assert svc._name == "test-env:analysis_chat"


def test_init_uses_explicit_task_scenario() -> None:
    svc = ChatService(_task("other-env:analysis_chat"), model="gpt-4o")
    assert svc._task.scenario == "other-env:analysis_chat"


def test_init_defaults_description_from_task() -> None:
    svc = ChatService(_task("other-env:analysis_chat"), model="gpt-4o")
    assert svc._description == "A2A service for other-env:analysis_chat"


def test_agent_card_basic_fields() -> None:
    svc = ChatService(
        _task(),
        model="gpt-4o",
        name="test",
        description="desc",
    )
    card = svc.agent_card()
    assert card.name == "test"
    assert card.description == "desc"
    assert card.skills == []


@pytest.mark.asyncio
async def test_execute_emits_working_and_input_required(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = ChatService(_task(), model="gpt-4o")
    queue = FakeQueue()
    context = FakeContext("hello")

    async def _fake_send(msg: Any) -> Trace:
        return Trace(content="done")

    chat = svc._get_or_create_chat("ctx-1")
    monkeypatch.setattr(chat, "send", _fake_send)
    svc._sessions["ctx-1"] = chat

    await svc.execute(context, queue)  # type: ignore[arg-type]

    assert len(queue.events) == 2
    assert queue.events[0].status.state.value == "working"
    assert queue.events[1].status.state.value == "input-required"


@pytest.mark.asyncio
async def test_execute_maps_errors_to_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = ChatService(_task(), model="gpt-4o")
    queue = FakeQueue()
    context = FakeContext("hello")

    async def _fail(msg: Any) -> Trace:
        raise RuntimeError("boom")

    chat = svc._get_or_create_chat("ctx-1")
    monkeypatch.setattr(chat, "send", _fail)
    svc._sessions["ctx-1"] = chat

    await svc.execute(context, queue)  # type: ignore[arg-type]

    assert len(queue.events) == 2
    assert queue.events[-1].status.state.value == "failed"
    assert "boom" in queue.events[-1].status.message.parts[0].root.text


@pytest.mark.asyncio
async def test_cancel_clears_session() -> None:
    svc = ChatService(_task(), model="gpt-4o")
    svc._get_or_create_chat("ctx-1")
    assert "ctx-1" in svc._sessions

    queue = FakeQueue()
    context = FakeContext("", context_id="ctx-1", task_id="t")
    await svc.cancel(context, queue)  # type: ignore[arg-type]

    assert "ctx-1" not in svc._sessions
    assert queue.events[-1].status.state.value == "canceled"


def test_get_or_create_reuses_session() -> None:
    svc = ChatService(_task(), model="gpt-4o")
    c1 = svc._get_or_create_chat("ctx-1")
    c2 = svc._get_or_create_chat("ctx-1")
    assert c1 is c2
