"""``ChatService`` — per-session ``Chat`` management + A2A execute/cancel flow.

``Chat`` and the reply-metadata builder are faked so no model/network is needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from hud.services import chat_service as cs_mod
from hud.services.chat_service import ChatService


class FakeChat:
    def __init__(self, *_a: Any, **_k: Any) -> None:
        self.cleared = False
        self.loaded: Any = None

    async def send(self, message: str) -> Any:
        return SimpleNamespace(content=f"echo:{message}")

    def clear(self) -> None:
        self.cleared = True

    def export_history(self) -> list[dict[str, Any]]:
        return [{"role": "user"}]

    def load_history(self, messages: list[dict[str, Any]]) -> None:
        self.loaded = messages


class FakeQueue:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def _patch_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs_mod, "Chat", FakeChat)
    monkeypatch.setattr(cs_mod, "build_reply_metadata_event", lambda **_k: None)


def _service() -> ChatService:
    task = cast("Any", SimpleNamespace(id="demo"))
    return ChatService(task, model="gpt-test")


def test_agent_card() -> None:
    card = _service().agent_card("http://host/")
    assert card.name == "demo"
    assert card.url == "http://host/"


async def test_send_reuses_session() -> None:
    service = _service()
    result = await service.send("hi", session_id="s1")
    assert result.content == "echo:hi"
    # Same session id reuses the same Chat instance.
    chat_a = service._get_or_create_chat("s1")  # pyright: ignore[reportPrivateUsage]
    chat_b = service._get_or_create_chat("s1")  # pyright: ignore[reportPrivateUsage]
    assert chat_a is chat_b


def test_export_history_empty_then_populated() -> None:
    service = _service()
    assert service.export_history("none") == []
    service.load_history([{"role": "user"}], session_id="s2")
    assert service.export_history("s2") == [{"role": "user"}]


def test_clear_removes_session() -> None:
    service = _service()
    service.load_history([{"x": 1}], session_id="s3")
    service.clear("s3")
    assert service.export_history("s3") == []


def test_cleanup_stale_sessions() -> None:
    service = _service()
    service.load_history([{"x": 1}], session_id="old")
    service._session_ttl_seconds = -1  # pyright: ignore[reportPrivateUsage]
    service._cleanup_stale_sessions()  # pyright: ignore[reportPrivateUsage]
    assert service.export_history("old") == []


async def test_execute_enqueues_final_status() -> None:
    service = _service()
    queue = FakeQueue()
    context = cast(
        "Any",
        SimpleNamespace(context_id="c1", task_id="t1", get_user_input=lambda: "hello"),
    )
    await service.execute(context, cast("Any", queue))
    assert len(queue.events) >= 2
    assert queue.events[-1].final is True


async def test_cancel_enqueues_canceled() -> None:
    service = _service()
    queue = FakeQueue()
    context = cast("Any", SimpleNamespace(context_id="c1", task_id="t1"))
    await service.cancel(context, cast("Any", queue))
    assert queue.events[-1].final is True
