from __future__ import annotations

import asyncio
from typing import Any

import pytest
from mcp.types import TextContent

from hud.eval.task import Task
from hud.services.chat import Chat
from hud.types import Trace


class _DummyEvalContext:
    pass


class _DummyEval:
    async def __aenter__(self) -> _DummyEvalContext:
        return _DummyEvalContext()

    async def __aexit__(self, *_args: Any) -> None:
        return None


class _FakeAgent:
    async def run(self, _ctx: Any, *, max_steps: int) -> Trace:
        return Trace(content=f"reply-{max_steps}")


@pytest.mark.asyncio()
async def test_send_surfaces_followup_user_messages(monkeypatch: Any) -> None:
    chat = Chat(
        Task(env={"name": "browser"}, scenario="assist"),
        model="gpt-4o",
        max_steps=3,
    )
    captured_task_args: list[dict[str, Any]] = []

    def _capturing_fake_eval(*args: Any, **_kwargs: Any) -> _DummyEval:
        task = args[0]
        captured_task_args.append(task.args or {})
        return _DummyEval()

    monkeypatch.setattr("hud.eval", _capturing_fake_eval)
    monkeypatch.setattr(chat, "_create_agent", lambda: _FakeAgent())

    first = await chat.send("hello")
    second = await chat.send("follow-up")

    assert first.content == "reply-3"
    assert second.content == "reply-3"
    assert [m["role"] for m in chat.messages] == ["user", "assistant", "user", "assistant"]
    assert chat.messages[0]["content"]["text"] == "hello"
    assert chat.messages[2]["content"]["text"] == "follow-up"

    assert len(captured_task_args) == 2
    assert captured_task_args[0]["messages"][0]["content"]["text"] == "hello"
    assert captured_task_args[1]["messages"][2]["content"]["text"] == "follow-up"


@pytest.mark.asyncio()
async def test_send_preserves_all_content_blocks(monkeypatch: Any) -> None:
    chat = Chat(
        Task(env={"name": "browser"}, scenario="assist"),
        model="gpt-4o",
        max_steps=3,
    )
    captured_task_args: list[dict[str, Any]] = []

    def _capturing_fake_eval(*args: Any, **_kwargs: Any) -> _DummyEval:
        task = args[0]
        captured_task_args.append(task.args or {})
        return _DummyEval()

    monkeypatch.setattr("hud.eval", _capturing_fake_eval)
    monkeypatch.setattr(chat, "_create_agent", lambda: _FakeAgent())

    blocks = [
        TextContent(type="text", text="part-1"),
        TextContent(type="text", text="part-2"),
    ]
    await chat.send(blocks)

    user_content = chat.messages[0]["content"]
    assert isinstance(user_content, list)
    assert user_content[0]["text"] == "part-1"
    assert user_content[1]["text"] == "part-2"
    sent_content = captured_task_args[0]["messages"][0]["content"]
    assert isinstance(sent_content, list)
    assert sent_content[1]["text"] == "part-2"


def test_clear_rotates_session_id() -> None:
    chat = Chat(Task(env={"name": "browser"}, scenario="assist"), model="gpt-4o")
    before = chat.session_id
    chat.clear()
    assert chat.session_id != before


@pytest.mark.asyncio()
async def test_execute_resolves_pending_elicitation_without_send(monkeypatch: Any) -> None:
    chat = Chat(Task(env={"name": "browser"}, scenario="assist"), model="gpt-4o")
    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    chat._pending_elicitations["task-1"] = future

    async def _unexpected_send(_message: Any) -> Any:
        raise AssertionError("send() should not run for pending elicitation follow-up")

    monkeypatch.setattr(chat, "send", _unexpected_send)

    class _Ctx:
        context_id = "ctx-1"
        task_id = "task-1"

        @staticmethod
        def get_user_input() -> str:
            return "follow-up answer"

    class _Queue:
        def __init__(self) -> None:
            self.events: list[Any] = []

        async def enqueue_event(self, event: Any) -> None:
            self.events.append(event)

    queue = _Queue()
    await chat.execute(_Ctx(), queue)  # type: ignore[arg-type]

    assert future.done()
    assert future.result() == "follow-up answer"
    assert queue.events == []


@pytest.mark.asyncio()
async def test_cancel_clears_pending_elicitation_and_history() -> None:
    chat = Chat(Task(env={"name": "browser"}, scenario="assist"), model="gpt-4o")
    chat.messages = [{"role": "user", "content": {"type": "text", "text": "hi"}}]
    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    chat._pending_elicitations["task-1"] = future

    class _Ctx:
        context_id = "ctx-1"
        task_id = "task-1"

    class _Queue:
        def __init__(self) -> None:
            self.events: list[Any] = []

        async def enqueue_event(self, event: Any) -> None:
            self.events.append(event)

    queue = _Queue()
    await chat.cancel(_Ctx(), queue)  # type: ignore[arg-type]

    assert chat.messages == []
    assert future.cancelled()
    assert "task-1" not in chat._pending_elicitations
    assert len(queue.events) == 1
