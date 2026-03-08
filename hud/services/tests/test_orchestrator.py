from __future__ import annotations

import json
from typing import Any, cast

import pytest
from a2a.types import TaskArtifactUpdateEvent

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


def test_init_stores_task_and_model() -> None:
    orch = ChatService("test-env", model="gpt-4o", scenario="analysis_chat")
    assert orch._task.scenario == "test-env:analysis_chat"
    assert orch._model == "gpt-4o"
    assert orch._name == "hud-test-env"


def test_init_allows_short_scenario_override() -> None:
    orch = ChatService("test-env", model="gpt-4o", scenario="analysis_chat")
    assert orch._task.scenario == "test-env:analysis_chat"


def test_init_allows_qualified_scenario_override() -> None:
    orch = ChatService(
        "test-env",
        model="gpt-4o",
        scenario="other-env:analysis_chat",
    )
    assert orch._task.scenario == "other-env:analysis_chat"


def test_agent_card_basic_fields() -> None:
    orch = ChatService(
        "test-env", model="gpt-4o", scenario="analysis_chat", name="test", description="desc"
    )
    card = orch.agent_card()
    assert card.name == "test"
    assert card.description == "desc"
    assert card.skills == []


@pytest.mark.asyncio
async def test_execute_emits_working_and_input_required(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = ChatService("test-env", model="gpt-4o", scenario="analysis_chat")
    queue = FakeQueue()
    context = FakeContext("hello")

    async def _fake_send(msg: Any) -> Trace:
        return Trace(content="done")

    chat = orch._get_or_create_chat("ctx-1")
    monkeypatch.setattr(chat, "send", _fake_send)
    orch._sessions["ctx-1"] = chat

    await orch.execute(context, queue)  # type: ignore[arg-type]

    assert len(queue.events) == 2
    assert queue.events[0].status.state.value == "working"
    assert queue.events[1].status.state.value == "input-required"


@pytest.mark.asyncio
async def test_execute_emits_metadata_artifact_before_input_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch = ChatService("test-env", model="gpt-4o", scenario="analysis_chat")
    queue = FakeQueue()
    context = FakeContext("hello")

    async def _fake_send(msg: Any) -> Trace:
        return Trace(
            content="done",
            citations=[
                {"type": "url_citation", "source": "https://example.com", "title": "Example"}
            ],
        )

    chat = orch._get_or_create_chat("ctx-1")
    monkeypatch.setattr(chat, "send", _fake_send)
    orch._sessions["ctx-1"] = chat

    await orch.execute(context, queue)  # type: ignore[arg-type]

    assert len(queue.events) == 3
    assert queue.events[0].status.state.value == "working"
    assert isinstance(queue.events[1], TaskArtifactUpdateEvent)
    payload = json.loads(cast(Any, queue.events[1].artifact.parts[0].root).text)
    assert payload["type"] == "hud_reply_metadata"
    assert payload["citations"][0]["source"] == "https://example.com"
    assert payload["data"] is None
    assert queue.events[2].status.state.value == "input-required"


@pytest.mark.asyncio
async def test_execute_maps_errors_to_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = ChatService("test-env", model="gpt-4o", scenario="analysis_chat")
    queue = FakeQueue()
    context = FakeContext("hello")

    async def _fail(msg: Any) -> Trace:
        raise RuntimeError("boom")

    chat = orch._get_or_create_chat("ctx-1")
    monkeypatch.setattr(chat, "send", _fail)
    orch._sessions["ctx-1"] = chat

    await orch.execute(context, queue)  # type: ignore[arg-type]

    assert len(queue.events) == 2
    assert queue.events[-1].status.state.value == "failed"
    assert "boom" in queue.events[-1].status.message.parts[0].root.text


@pytest.mark.asyncio
async def test_cancel_clears_session() -> None:
    orch = ChatService("test-env", model="gpt-4o", scenario="analysis_chat")
    orch._get_or_create_chat("ctx-1")
    assert "ctx-1" in orch._sessions

    queue = FakeQueue()
    context = FakeContext("", context_id="ctx-1", task_id="t")
    await orch.cancel(context, queue)  # type: ignore[arg-type]

    assert "ctx-1" not in orch._sessions
    assert queue.events[-1].status.state.value == "canceled"


def test_get_or_create_reuses_session() -> None:
    orch = ChatService("test-env", model="gpt-4o", scenario="analysis_chat")
    c1 = orch._get_or_create_chat("ctx-1")
    c2 = orch._get_or_create_chat("ctx-1")
    assert c1 is c2
