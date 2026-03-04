from __future__ import annotations

import asyncio
from typing import Any

import pytest

from hud.eval.task import Task
from hud.services.chat_manager import (
    ChatManager,
    SessionBusyError,
    SessionExpiredError,
    SessionFinishedError,
    UnknownChatDefinitionError,
)
from hud.services.types import ChatDefinition
from hud.types import Trace


class FakeChat:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.cleared = False
        self.session_id = "fake-trace"
        self._delay = 0.0

    async def send(self, message: Any) -> Trace:
        await asyncio.sleep(self._delay)
        return Trace(content=str(message))

    def clear(self) -> None:
        self.cleared = True

    def agent_card(self) -> Any:
        class _Card:
            name = "fake-chat"

        return _Card()


def _definition(name: str = "default") -> ChatDefinition:
    return ChatDefinition(
        name=name,
        task=Task(env={"name": "browser"}, scenario="assist"),
        model="gpt-4o",
    )


@pytest.mark.asyncio
async def test_unknown_definition_raises() -> None:
    manager = ChatManager([_definition()])
    with pytest.raises(UnknownChatDefinitionError):
        manager.create_session(definition_name="nope")


@pytest.mark.asyncio
async def test_send_creates_session_and_returns_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hud.services.chat_manager.Chat", FakeChat)
    manager = ChatManager([_definition()])

    result = await manager.send("ctx-1", "hello")
    assert result.content == "hello"
    assert manager.list_sessions()[0]["session_id"] == "ctx-1"


@pytest.mark.asyncio
async def test_per_session_lock_rejects_parallel_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowFakeChat(FakeChat):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._delay = 0.05

    monkeypatch.setattr("hud.services.chat_manager.Chat", SlowFakeChat)
    manager = ChatManager([_definition()])
    session_id = manager.create_session()

    first = asyncio.create_task(manager.send(session_id, "one"))
    await asyncio.sleep(0.01)
    with pytest.raises(SessionBusyError):
        await manager.send(session_id, "two")
    await first


@pytest.mark.asyncio
async def test_finish_then_send_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hud.services.chat_manager.Chat", FakeChat)
    manager = ChatManager([_definition()])
    session_id = manager.create_session()
    manager.finish(session_id)

    with pytest.raises(SessionFinishedError):
        await manager.send(session_id, "hello")


@pytest.mark.asyncio
async def test_expired_session_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hud.services.chat_manager.Chat", FakeChat)
    manager = ChatManager([_definition()], session_ttl_seconds=1)
    session_id = manager.create_session()
    session = manager._sessions[session_id]
    session.last_active_at -= 10
    manager.cleanup_expired()

    with pytest.raises(SessionExpiredError):
        await manager.send(session_id, "hello")
