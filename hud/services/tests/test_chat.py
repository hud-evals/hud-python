from __future__ import annotations

from typing import Any

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


def _fake_eval(*_args: Any, **_kwargs: Any) -> _DummyEval:
    return _DummyEval()


async def test_send_surfaces_followup_user_messages(monkeypatch: Any) -> None:
    chat = Chat(Task(env={"name": "browser"}, scenario="assist"), model="gpt-4o", max_steps=3)
    captured_task_args: list[dict[str, Any]] = []

    def _capturing_fake_eval(*args: Any, **_kwargs: Any) -> _DummyEval:
        task = args[0]
        captured_task_args.append(task.args or {})
        return _DummyEval()

    monkeypatch.setattr("hud.eval", _capturing_fake_eval)
    monkeypatch.setattr(chat, "_create_agent", lambda: _FakeAgent())

    first = await chat.send("hello")
    second = await chat.send("follow-up")

    assert [m["role"] for m in first.messages] == ["user", "assistant"]
    assert [m["role"] for m in second.messages] == ["user", "assistant", "user", "assistant"]
    assert second.messages[2]["content"]["text"] == "follow-up"
